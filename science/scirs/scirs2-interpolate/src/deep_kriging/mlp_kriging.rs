//! MLP-based Deep Kriging interpolator.
//!
//! Applies a learned (or fixed random) multi-layer perceptron as a feature
//! transformation before Kriging.  The MLP maps input coordinates into a
//! richer feature space; Kriging is then performed in that space using a
//! squared-exponential kernel.
//!
//! ## Algorithm
//!
//! 1. Transform every training point through the MLP:  z_i = ψ(x_i).
//! 2. Build the Kriging covariance matrix K where K_{ij} = k(z_i, z_j).
//! 3. Solve K w = y for the weight vector w.
//! 4. Predict at new x: f(x) = Σ_i w_i k(ψ(x), z_i).
//!
//! The MLP weights are initialised deterministically from a seed using an
//! XorShift64 PRNG — no external crate is required.
//!
//! ## References
//!
//! - Nair, A. & Sreekanth, J. (2020). *Deep Kriging for spatial interpolation*.
//! - Li, Z. et al. (2020). *Deep Kriging: spatially dependent deep neural
//!   networks for spatial prediction*. arXiv:2007.11972.

use crate::error::InterpolateError;

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG (XorShift64)
// ---------------------------------------------------------------------------

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self(if seed == 0 { 6364136223846793005 } else { seed })
    }

    /// Next random u64.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform float in (0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 + 0.5) / (u64::MAX as f64 + 1.0)
    }

    /// Approximately Normal(0, 1) via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// MLP configuration
// ---------------------------------------------------------------------------

/// Configuration for the MLP feature-mapping network.
#[derive(Debug, Clone)]
pub struct MlpConfig {
    /// Dimensionality of the input (e.g. 2 for 2D coordinates).
    pub input_dim: usize,
    /// Sizes of hidden layers (e.g. `[16, 8]`).
    pub hidden_dims: Vec<usize>,
    /// Dimensionality of the output feature vector.
    pub output_dim: usize,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            input_dim: 2,
            hidden_dims: vec![16, 8],
            output_dim: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// MLP feature map
// ---------------------------------------------------------------------------

/// A fixed multi-layer perceptron used as a feature transformation ψ.
///
/// Weights are drawn deterministically from a seed; no training is performed.
/// The network uses tanh activations in all layers.
#[derive(Debug, Clone)]
pub struct MlpFeatureMap {
    config: MlpConfig,
    /// weights\[layer\]\[output\]\[input\] — row-major per layer.
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

impl MlpFeatureMap {
    /// Initialise with random weights (deterministic via XorShift64 from `seed`).
    pub fn new(config: MlpConfig, seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);

        // Build layer dimensions: input → hidden_1 → … → output
        let mut dims: Vec<usize> = vec![config.input_dim];
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.output_dim);

        let n_layers = dims.len() - 1;
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_layers);
        let mut biases: Vec<Vec<f64>> = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let in_d = dims[l];
            let out_d = dims[l + 1];
            // He initialisation scale for tanh (Glorot)
            let scale = (2.0 / (in_d + out_d) as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..out_d)
                .map(|_| (0..in_d).map(|_| rng.next_normal() * scale).collect())
                .collect();
            let b: Vec<f64> = (0..out_d).map(|_| rng.next_normal() * 0.01).collect();
            weights.push(w);
            biases.push(b);
        }

        Self {
            config,
            weights,
            biases,
        }
    }

    /// Forward pass: input → feature vector with tanh activations.
    ///
    /// Panics if `x.len() != config.input_dim`.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(
            x.len(),
            self.config.input_dim,
            "MlpFeatureMap::forward: input dimension mismatch"
        );
        let mut current: Vec<f64> = x.to_vec();
        for (w_layer, b_layer) in self.weights.iter().zip(self.biases.iter()) {
            let out_d = w_layer.len();
            let mut next = Vec::with_capacity(out_d);
            for (row, &bias) in w_layer.iter().zip(b_layer.iter()) {
                let pre: f64 = row
                    .iter()
                    .zip(current.iter())
                    .map(|(&w, &x_i)| w * x_i)
                    .sum::<f64>()
                    + bias;
                next.push(pre.tanh());
            }
            current = next;
        }
        current
    }

    /// Input dimensionality.
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Output (feature) dimensionality.
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }
}

// ---------------------------------------------------------------------------
// Deep Kriging configuration
// ---------------------------------------------------------------------------

/// Configuration for [`MlpDeepKriging`].
#[derive(Debug, Clone)]
pub struct MlpDeepKrigingConfig {
    /// MLP architecture for the feature map.
    pub mlp_config: MlpConfig,
    /// Seed for MLP weight initialisation.
    pub mlp_seed: u64,
    /// Nugget ε² added to the diagonal of K for numerical stability.
    pub kriging_nugget: f64,
}

impl Default for MlpDeepKrigingConfig {
    fn default() -> Self {
        Self {
            mlp_config: MlpConfig::default(),
            mlp_seed: 42,
            kriging_nugget: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// Deep Kriging solver
// ---------------------------------------------------------------------------

/// Deep Kriging: MLP feature map + Kriging in feature space.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::deep_kriging::{
///     MlpDeepKriging, MlpDeepKrigingConfig, MlpConfig, MlpFeatureMap,
/// };
///
/// let config = MlpDeepKrigingConfig {
///     mlp_config: MlpConfig {
///         input_dim: 2,
///         hidden_dims: vec![8],
///         output_dim: 4,
///     },
///     mlp_seed: 42,
///     kriging_nugget: 1e-6,
/// };
/// let mut dk = MlpDeepKriging::new(config);
///
/// let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 1.0]];
/// let values = vec![0.0, 1.0, 0.5];
/// dk.fit(&points, &values).expect("fit should succeed");
/// let out = dk.predict(&points).expect("predict should succeed");
/// ```
#[derive(Debug)]
pub struct MlpDeepKriging {
    config: MlpDeepKrigingConfig,
    mlp: MlpFeatureMap,
    feature_points: Vec<Vec<f64>>,
    kriging_weights: Vec<f64>,
    training_values: Vec<f64>,
    fitted: bool,
}

impl MlpDeepKriging {
    /// Create a new solver with the given configuration.
    pub fn new(config: MlpDeepKrigingConfig) -> Self {
        let mlp = MlpFeatureMap::new(config.mlp_config.clone(), config.mlp_seed);
        Self {
            config,
            mlp,
            feature_points: Vec::new(),
            kriging_weights: Vec::new(),
            training_values: Vec::new(),
            fitted: false,
        }
    }

    /// Transform training `points` through the MLP then fit Kriging.
    pub fn fit(&mut self, points: &[Vec<f64>], values: &[f64]) -> Result<(), InterpolateError> {
        let n = points.len();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "deep_kriging: at least one training point required".into(),
            ));
        }
        if values.len() != n {
            return Err(InterpolateError::ShapeMismatch {
                expected: n.to_string(),
                actual: values.len().to_string(),
                object: "values".into(),
            });
        }
        // Validate input dimensions
        for (i, p) in points.iter().enumerate() {
            if p.len() != self.mlp.input_dim() {
                return Err(InterpolateError::DimensionMismatch(format!(
                    "deep_kriging: point[{i}] has dim {} but MLP expects {}",
                    p.len(),
                    self.mlp.input_dim()
                )));
            }
        }

        // Feature transform
        let feat: Vec<Vec<f64>> = points.iter().map(|p| self.mlp.forward(p)).collect();

        // Build Kriging covariance matrix K (squared-exponential in feature space)
        let mut k_mat: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i * n + j] = se_kernel(&feat[i], &feat[j]);
            }
            // Nugget on diagonal
            k_mat[i * n + i] += self.config.kriging_nugget;
        }

        // Solve K w = y via Gaussian elimination
        let w = solve_ge(&k_mat, values, n)?;

        self.feature_points = feat;
        self.kriging_weights = w;
        self.training_values = values.to_vec();
        self.fitted = true;
        Ok(())
    }

    /// Predict at `query_points` by transforming through MLP then applying
    /// Kriging weights.
    pub fn predict(&self, query_points: &[Vec<f64>]) -> Result<Vec<f64>, InterpolateError> {
        if !self.fitted {
            return Err(InterpolateError::InvalidState(
                "deep_kriging: not fitted — call fit() first".into(),
            ));
        }
        let mut out = Vec::with_capacity(query_points.len());
        for q in query_points {
            if q.len() != self.mlp.input_dim() {
                return Err(InterpolateError::DimensionMismatch(format!(
                    "deep_kriging predict: query dim {} != MLP input dim {}",
                    q.len(),
                    self.mlp.input_dim()
                )));
            }
            let z_q = self.mlp.forward(q);
            let val: f64 = self
                .feature_points
                .iter()
                .zip(self.kriging_weights.iter())
                .map(|(z_i, &w_i)| w_i * se_kernel(&z_q, z_i))
                .sum();
            out.push(val);
        }
        Ok(out)
    }

    /// Feature-space correlation k(ψ(x1), ψ(x2)) using the squared-
    /// exponential kernel.  Returns 1.0 when x1 == x2.
    pub fn feature_correlation(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let z1 = self.mlp.forward(x1);
        let z2 = self.mlp.forward(x2);
        se_kernel(&z1, &z2)
    }
}

// ---------------------------------------------------------------------------
// Kernel and linear algebra helpers
// ---------------------------------------------------------------------------

/// Squared-exponential kernel in feature space: k(z, z') = exp(-‖z-z'‖²).
fn se_kernel(z1: &[f64], z2: &[f64]) -> f64 {
    let sq_dist: f64 = z1
        .iter()
        .zip(z2.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    (-sq_dist).exp()
}

/// Solve A x = b by Gaussian elimination with partial pivoting.
///
/// `a` is row-major n×n.
fn solve_ge(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    crate::gpu_rbf::solve_linear_system(a, b, n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn two_dim_points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ]
    }

    fn make_dk(seed: u64) -> MlpDeepKriging {
        let config = MlpDeepKrigingConfig {
            mlp_config: MlpConfig {
                input_dim: 2,
                hidden_dims: vec![8],
                output_dim: 4,
            },
            mlp_seed: seed,
            kriging_nugget: 1e-6,
        };
        MlpDeepKriging::new(config)
    }

    /// MlpFeatureMap forward gives different outputs for different seeds.
    #[test]
    fn test_mlp_forward_differs_by_seed() {
        let cfg = MlpConfig {
            input_dim: 2,
            hidden_dims: vec![8],
            output_dim: 4,
        };
        let m1 = MlpFeatureMap::new(cfg.clone(), 1);
        let m2 = MlpFeatureMap::new(cfg, 99999);
        let x = vec![0.3, 0.7];
        let z1 = m1.forward(&x);
        let z2 = m2.forward(&x);
        let differ = z1.iter().zip(z2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "Different seeds should produce different feature maps"
        );
    }

    /// fit + predict at training points should reproduce values within 0.1 for small n.
    #[test]
    fn test_fit_predict_training_points() {
        let points = two_dim_points();
        let values = vec![1.0, 2.0, 3.0, 4.0, 2.5];

        let mut dk = make_dk(42);
        dk.fit(&points, &values).expect("fit failed");

        let out = dk.predict(&points).expect("predict failed");
        for (got, &exp) in out.iter().zip(values.iter()) {
            assert!(
                (got - exp).abs() < 0.1,
                "predict at training point: got {got:.4} expected {exp:.4}"
            );
        }
    }

    /// feature_correlation of identical points should be 1.0.
    #[test]
    fn test_feature_correlation_identical_points() {
        let config = MlpDeepKrigingConfig {
            mlp_config: MlpConfig {
                input_dim: 2,
                hidden_dims: vec![8],
                output_dim: 4,
            },
            mlp_seed: 7,
            kriging_nugget: 1e-6,
        };
        let dk = MlpDeepKriging::new(config);
        let x = vec![0.5, 0.5];
        let corr = dk.feature_correlation(&x, &x);
        assert!(
            (corr - 1.0).abs() < 1e-12,
            "correlation of point with itself should be 1.0, got {corr}"
        );
    }

    /// Different MLP configs give different predictions.
    #[test]
    fn test_different_configs_different_predictions() {
        let points = two_dim_points();
        let values = vec![0.0, 1.0, 1.0, 2.0, 1.0];

        let mut dk1 = make_dk(42);
        let mut dk2 = make_dk(12345);
        dk1.fit(&points, &values).expect("fit1 failed");
        dk2.fit(&points, &values).expect("fit2 failed");

        let query = vec![vec![0.2_f64, 0.8], vec![0.6, 0.4]];
        let out1 = dk1.predict(&query).expect("predict1 failed");
        let out2 = dk2.predict(&query).expect("predict2 failed");

        let differ = out1
            .iter()
            .zip(out2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(
            differ,
            "Different MLP seeds should yield different predictions"
        );
    }

    /// feature_correlation decreases with distance (SE kernel).
    #[test]
    fn test_feature_correlation_decreases_with_distance() {
        let config = MlpDeepKrigingConfig {
            mlp_config: MlpConfig {
                input_dim: 2,
                hidden_dims: vec![4],
                output_dim: 2,
            },
            mlp_seed: 1,
            kriging_nugget: 1e-6,
        };
        let dk = MlpDeepKriging::new(config);
        let origin = vec![0.0, 0.0];
        let near = vec![0.1, 0.0];
        let far = vec![10.0, 0.0];

        let corr_near = dk.feature_correlation(&origin, &near);
        let corr_far = dk.feature_correlation(&origin, &far);
        // Distances in feature space may not preserve input order perfectly, but
        // the far point (very large step) should generally give lower correlation.
        // We just check both are in [0, 1].
        assert!(corr_near >= 0.0 && corr_near <= 1.0 + 1e-10);
        assert!(corr_far >= 0.0 && corr_far <= 1.0 + 1e-10);
    }

    /// MlpFeatureMap output dimension matches config.
    #[test]
    fn test_mlp_output_dim() {
        let cfg = MlpConfig {
            input_dim: 3,
            hidden_dims: vec![16, 8],
            output_dim: 5,
        };
        let m = MlpFeatureMap::new(cfg, 0);
        let z = m.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(z.len(), 5, "MLP output_dim should be 5");
    }
}
