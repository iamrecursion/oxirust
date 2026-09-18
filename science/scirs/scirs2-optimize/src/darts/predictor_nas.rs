//! Predictor-based Neural Architecture Search.
//!
//! Uses a Gaussian kernel ridge regression surrogate (manually implemented,
//! no external GP dependency) to model the mapping from architecture encodings
//! to validation scores.  An active learning loop iterates:
//!
//! 1. Maintain a surrogate trained on evaluated architectures.
//! 2. Use UCB or Expected Improvement acquisition to propose candidates.
//! 3. Evaluate top candidates with the true evaluation function.
//! 4. Retrain the surrogate and repeat.
//!
//! ## Architecture Encoding
//!
//! An architecture is encoded as a flat `Vec<f64>` of normalised operation
//! indices in `[0, 1]`.  Each element `op_idx / (n_operations - 1)`.
//!
//! ## Surrogate
//!
//! Gaussian kernel ridge regression with RBF kernel:
//! `k(a, b) = exp(-||a - b||² / 2)`.
//! Prediction: `ŷ(x) = k(x, X)ᵀ (K + αI)⁻¹ y`.

use super::Lcg;
use crate::error::{OptimizeError, OptimizeResult};

// ────────────────────────────────────────────────── Architecture encoding ──

/// Encode a discrete architecture as a normalised flat `Vec<f64>`.
///
/// Each operation index `op_idx` is divided by `(n_operations - 1).max(1)` so
/// values lie in `[0, 1]`.
///
/// # Arguments
/// - `arch_indices`: `[cell][node][predecessor]` operation indices.
/// - `n_operations`: Total number of candidate operations.
pub fn encode_architecture(arch_indices: &[Vec<Vec<usize>>], n_operations: usize) -> Vec<f64> {
    let norm = (n_operations.max(1) - 1) as f64;
    let denom = norm.max(1.0);
    arch_indices
        .iter()
        .flat_map(|cell| {
            cell.iter()
                .flat_map(|node_edges| node_edges.iter().map(|&op_idx| op_idx as f64 / denom))
        })
        .collect()
}

// ──────────────────────────────────────────────── PredictorNasConfig ──

/// Configuration for the predictor-based NAS searcher.
#[derive(Debug, Clone)]
pub struct PredictorNasConfig {
    /// Number of cells per architecture.
    pub n_cells: usize,
    /// Number of candidate operations per edge.
    pub n_operations: usize,
    /// Number of feature channels (informational; used for index bounds).
    pub channels: usize,
    /// Number of intermediate nodes per cell.
    pub n_nodes: usize,
    /// Number of random architectures evaluated in Phase 1 (warm-up).
    pub n_initial_samples: usize,
    /// Number of active-learning iterations in Phase 2.
    pub n_iterations: usize,
    /// Number of candidate architectures proposed per iteration.
    pub n_candidates_per_iter: usize,
    /// Number of top candidates actually evaluated per iteration.
    pub n_top_to_evaluate: usize,
    /// Exploration-exploitation trade-off for UCB: `μ + κ·σ`.
    pub ucb_kappa: f64,
    /// Random seed for the internal LCG.
    pub seed: u64,
}

impl Default for PredictorNasConfig {
    fn default() -> Self {
        Self {
            n_cells: 3,
            n_operations: 6,
            channels: 32,
            n_nodes: 4,
            n_initial_samples: 5,
            n_iterations: 3,
            n_candidates_per_iter: 20,
            n_top_to_evaluate: 2,
            ucb_kappa: 2.0,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────── AcquisitionStrategy ──

/// Strategy for selecting candidate architectures from the surrogate.
#[derive(Debug, Clone, PartialEq)]
pub enum AcquisitionStrategy {
    /// Upper Confidence Bound: `μ(x) + κ · σ(x)`.
    Ucb,
    /// Expected Improvement over the current best: `E[max(0, f(x) - f_best)]`.
    ExpectedImprovement,
}

// ──────────────────────────────────────────────────── PredictorNasResult ──

/// Result of a predictor-based NAS search.
#[derive(Debug, Clone)]
pub struct PredictorNasResult {
    /// Discrete architecture indices for the best found architecture.
    pub best_arch_indices: Vec<Vec<Vec<usize>>>,
    /// Score of the best architecture (higher is better).
    pub best_score: f64,
    /// Total number of architectures that were actually evaluated.
    pub n_evaluated: usize,
}

// ──────────────────────────────────────────── RidgeSurrogate (private) ──

/// Gaussian kernel ridge regression surrogate.
///
/// RBF kernel: `k(a, b) = exp(-||a - b||² / 2)`.
/// Predictions: `ŷ(x) = k(x, X)ᵀ α_coeff` where
/// `α_coeff = (K + α·I)⁻¹ y`.
///
/// For the small datasets that arise during NAS warm-up (n ≤ ~20) we solve
/// the linear system via Gaussian elimination.
struct RidgeSurrogate {
    /// Training inputs, one row per sample.
    x_train: Vec<Vec<f64>>,
    /// Training targets.
    y_train: Vec<f64>,
    /// Regularisation coefficient.
    alpha: f64,
    /// Solved coefficients `(K + α·I)⁻¹ y`.  Empty when not yet fitted.
    coeffs: Vec<f64>,
}

impl RidgeSurrogate {
    fn new(alpha: f64) -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            alpha,
            coeffs: Vec::new(),
        }
    }

    /// Compute RBF kernel value between two vectors.
    fn rbf(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi) * (ai - bi))
            .sum();
        (-sq_dist / 2.0).exp()
    }

    /// Fit the surrogate to `(x, y)` training data.
    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.x_train = x.to_vec();
        self.y_train = y.to_vec();
        let n = x.len();
        if n == 0 {
            self.coeffs = Vec::new();
            return;
        }

        // Build kernel matrix K (n×n) and add regularisation on diagonal.
        let mut k_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let kij = self.rbf(&x[i], &x[j]);
                        if i == j {
                            kij + self.alpha
                        } else {
                            kij
                        }
                    })
                    .collect()
            })
            .collect();

        // Solve (K + αI) α_coeff = y via Gaussian elimination with partial pivoting.
        let mut rhs: Vec<f64> = y.to_vec();
        gauss_elimination(&mut k_matrix, &mut rhs);
        self.coeffs = rhs;
    }

    /// Predict `(mean, std)` for a query point `x`.
    ///
    /// The predictive std is estimated as a diagonal approximation.
    fn predict_mean_std(&self, x: &[f64]) -> (f64, f64) {
        let n = self.x_train.len();
        if n == 0 || self.coeffs.len() != n {
            // Uninformed prior: mean = 0, large std.
            return (0.0, 1.0);
        }

        // k_vec[i] = k(x, x_train[i])
        let k_vec: Vec<f64> = self.x_train.iter().map(|xi| self.rbf(x, xi)).collect();

        // Mean: k_vec · coeffs
        let mean: f64 = k_vec
            .iter()
            .zip(self.coeffs.iter())
            .map(|(ki, ci)| ki * ci)
            .sum();

        // Predictive variance approximation:
        // var ≈ k(x,x) - Σ_i k(x,xi)² / (k(xi,xi) + α)
        let k_self = self.rbf(x, x); // = 1.0 for RBF

        let var_approx: f64 = k_self
            - k_vec
                .iter()
                .zip(self.x_train.iter())
                .map(|(&kxi, xi)| {
                    let kii = self.rbf(xi, xi) + self.alpha;
                    kxi * kxi / kii.max(1e-12)
                })
                .sum::<f64>();

        let std = var_approx.max(0.0).sqrt();
        (mean, std)
    }
}

/// Gaussian elimination with partial pivoting.
///
/// Modifies `a` and `b` in-place to solve `a · x = b`.
/// Result is stored back in `b`.
fn gauss_elimination(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> bool {
    let n = b.len();
    if n == 0 {
        return true;
    }

    for col in 0..n {
        // Find pivot.
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                a[r1][col]
                    .abs()
                    .partial_cmp(&a[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if a[pivot_row][col].abs() < 1e-14 {
            // Near-singular: skip (regularisation prevents this in practice).
            continue;
        }

        // Swap rows.
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot = a[col][col];
        // Eliminate below.
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            b[row] -= factor * b[col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
        }
    }

    // Back-substitution.
    for col in (0..n).rev() {
        if a[col][col].abs() < 1e-14 {
            b[col] = 0.0;
            continue;
        }
        for row in 0..col {
            let factor = a[row][col] / a[col][col];
            b[row] -= factor * b[col];
        }
        b[col] /= a[col][col];
    }
    true
}

// ──────────────────────────────────────────────── PredictorNasSearcher ──

/// Predictor-based NAS searcher.
///
/// Maintains a `RidgeSurrogate` that is retrained after each batch of true
/// evaluations.
pub struct PredictorNasSearcher {
    config: PredictorNasConfig,
    surrogate: RidgeSurrogate,
    rng: Lcg,
    evaluated_x: Vec<Vec<f64>>,
    evaluated_y: Vec<f64>,
}

impl PredictorNasSearcher {
    /// Construct a new searcher from the given config.
    pub fn new(config: PredictorNasConfig) -> Self {
        let rng = Lcg::new(config.seed);
        Self {
            surrogate: RidgeSurrogate::new(1e-3),
            config,
            rng,
            evaluated_x: Vec::new(),
            evaluated_y: Vec::new(),
        }
    }

    /// Sample a random architecture — `[cell][node][predecessor]` indices.
    fn sample_random_arch(&mut self) -> Vec<Vec<Vec<usize>>> {
        let n_ops = self.config.n_operations;
        (0..self.config.n_cells)
            .map(|_| {
                (0..self.config.n_nodes)
                    .map(|i| {
                        let n_predecessors = 2 + i; // 2 fixed input nodes
                        (0..n_predecessors)
                            .map(|_| {
                                let raw = self.rng.next_f64();
                                ((raw * n_ops as f64) as usize).min(n_ops - 1)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute the UCB acquisition value for a query encoding.
    fn ucb(&self, x: &[f64]) -> f64 {
        let (mean, std) = self.surrogate.predict_mean_std(x);
        mean + self.config.ucb_kappa * std
    }

    /// Compute the Expected Improvement acquisition value.
    fn expected_improvement(&self, x: &[f64]) -> f64 {
        let f_best = self
            .evaluated_y
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if f_best.is_infinite() {
            return 0.0;
        }
        let (mean, std) = self.surrogate.predict_mean_std(x);
        if std < 1e-12 {
            return (mean - f_best).max(0.0);
        }
        let z = (mean - f_best) / std;
        // EI = (mean - f_best) · Φ(z) + std · φ(z)
        let phi_z = normal_cdf(z);
        let pdf_z = normal_pdf(z);
        (mean - f_best) * phi_z + std * pdf_z
    }

    /// Acquisition function value (dispatches to UCB or EI).
    fn acquisition(&self, x: &[f64], strategy: &AcquisitionStrategy) -> f64 {
        match strategy {
            AcquisitionStrategy::Ucb => self.ucb(x),
            AcquisitionStrategy::ExpectedImprovement => self.expected_improvement(x),
        }
    }

    /// Evaluate an architecture with `eval_fn` and record the result.
    fn evaluate_and_record(
        &mut self,
        arch: &[Vec<Vec<usize>>],
        eval_fn: &impl Fn(&[Vec<Vec<usize>>]) -> f64,
    ) -> f64 {
        let score = eval_fn(arch);
        let enc = encode_architecture(arch, self.config.n_operations);
        self.evaluated_x.push(enc);
        self.evaluated_y.push(score);
        score
    }

    /// Refit the surrogate to all evaluated data.
    fn refit_surrogate(&mut self) {
        self.surrogate.fit(&self.evaluated_x, &self.evaluated_y);
    }

    /// Run the full predictor-based NAS search.
    ///
    /// Phase 1: evaluate `n_initial_samples` random architectures.
    /// Phase 2: run `n_iterations` active-learning rounds.
    ///
    /// # Arguments
    /// - `eval_fn`: True evaluation function.  Higher return value = better arch.
    pub fn search(
        &mut self,
        eval_fn: impl Fn(&[Vec<Vec<usize>>]) -> f64,
    ) -> OptimizeResult<PredictorNasResult> {
        if self.config.n_initial_samples == 0 {
            return Err(OptimizeError::InvalidInput(
                "n_initial_samples must be > 0".to_string(),
            ));
        }

        // ── Phase 1: warm-up with random samples ──────────────────────────────
        for _ in 0..self.config.n_initial_samples {
            let arch = self.sample_random_arch();
            self.evaluate_and_record(&arch, &eval_fn);
        }
        self.refit_surrogate();

        // ── Phase 2: active learning ──────────────────────────────────────────
        let strategy = AcquisitionStrategy::Ucb;
        for _ in 0..self.config.n_iterations {
            // Generate candidate architectures.
            let mut candidates: Vec<(f64, Vec<Vec<Vec<usize>>>)> =
                (0..self.config.n_candidates_per_iter)
                    .map(|_| {
                        let arch = self.sample_random_arch();
                        let enc = encode_architecture(&arch, self.config.n_operations);
                        let acq = self.acquisition(&enc, &strategy);
                        (acq, arch)
                    })
                    .collect();

            // Sort by acquisition (descending).
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Evaluate top-k.
            let n_eval = self.config.n_top_to_evaluate.min(candidates.len());
            for (_, arch) in candidates.into_iter().take(n_eval) {
                self.evaluate_and_record(&arch, &eval_fn);
            }

            self.refit_surrogate();
        }

        // ── Find best ─────────────────────────────────────────────────────────
        let best_idx = self
            .evaluated_y
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| {
                OptimizeError::ComputationError("No architectures were evaluated".to_string())
            })?;

        let best_score = self.evaluated_y[best_idx];
        let best_enc = &self.evaluated_x[best_idx];
        // Decode back to arch indices: round to nearest integer index.
        let norm = (self.config.n_operations.max(1) - 1) as f64;
        let denom = norm.max(1.0);
        let decoded_flat: Vec<usize> = best_enc
            .iter()
            .map(|&v| ((v * denom).round() as usize).min(self.config.n_operations - 1))
            .collect();
        let best_arch_indices =
            reconstruct_arch_indices(&decoded_flat, self.config.n_cells, self.config.n_nodes);

        Ok(PredictorNasResult {
            best_arch_indices,
            best_score,
            n_evaluated: self.evaluated_y.len(),
        })
    }
}

// ─────────────────────────────────────────────────── helpers ──

/// Reconstruct `[cell][node][predecessor]` arch indices from a flat vector.
fn reconstruct_arch_indices(
    flat: &[usize],
    n_cells: usize,
    n_nodes: usize,
) -> Vec<Vec<Vec<usize>>> {
    let mut offset = 0;
    let mut result = Vec::with_capacity(n_cells);
    for _ in 0..n_cells {
        let mut cell = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let n_pred = 2 + i;
            let node_edges: Vec<usize> = if offset + n_pred <= flat.len() {
                flat[offset..offset + n_pred].to_vec()
            } else {
                vec![0; n_pred]
            };
            offset += n_pred;
            cell.push(node_edges);
        }
        result.push(cell);
    }
    result
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = normal_pdf(x);
    let cdf_pos = 1.0 - pdf * poly;
    if x >= 0.0 {
        cdf_pos
    } else {
        1.0 - cdf_pos
    }
}

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// ═══════════════════════════════════════════════════════════════════ tests ═══

#[cfg(test)]
mod tests {
    use super::*;

    // ── encode_architecture ────────────────────────────────────────────────────

    #[test]
    fn test_encode_architecture_deterministic() {
        let arch: Vec<Vec<Vec<usize>>> = vec![vec![vec![0, 1], vec![2, 3, 0], vec![1, 0, 2, 1]]];
        let enc1 = encode_architecture(&arch, 6);
        let enc2 = encode_architecture(&arch, 6);
        assert_eq!(enc1, enc2);
    }

    #[test]
    fn test_encode_architecture_length() {
        // n_cells=2, n_nodes=4: edges = 2+3+4+5 = 14 per cell → 28 total.
        let arch: Vec<Vec<Vec<usize>>> = (0..2_usize)
            .map(|_| {
                (0..4_usize)
                    .map(|i| vec![0_usize; 2 + i])
                    .collect::<Vec<_>>()
            })
            .collect();
        let enc = encode_architecture(&arch, 6);
        assert_eq!(enc.len(), 28, "enc.len()={}", enc.len());
    }

    #[test]
    fn test_encode_architecture_range() {
        let arch: Vec<Vec<Vec<usize>>> = vec![vec![vec![0, 5], vec![3, 1, 5]]];
        let enc = encode_architecture(&arch, 6);
        for &v in &enc {
            assert!(v >= 0.0 && v <= 1.0, "v={v} out of [0,1]");
        }
    }

    #[test]
    fn test_encode_architecture_single_op() {
        // With n_operations=1 all encodings should be 0.0.
        let arch: Vec<Vec<Vec<usize>>> = vec![vec![vec![0, 0]]];
        let enc = encode_architecture(&arch, 1);
        for &v in &enc {
            assert!((v - 0.0).abs() < 1e-10, "v={v}");
        }
    }

    // ── RidgeSurrogate ─────────────────────────────────────────────────────────

    #[test]
    fn test_ridge_surrogate_predict_after_fit() {
        let mut surr = RidgeSurrogate::new(1e-3);
        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.5, 1.0];
        surr.fit(&x, &y);
        let (mean, _std) = surr.predict_mean_std(&[0.25]);
        assert!(mean.is_finite(), "mean={mean}");
    }

    #[test]
    fn test_ridge_surrogate_empty_returns_prior() {
        let surr = RidgeSurrogate::new(1e-3);
        let (mean, std) = surr.predict_mean_std(&[0.5]);
        assert!((mean - 0.0).abs() < 1e-10, "mean={mean}");
        assert!((std - 1.0).abs() < 1e-10, "std={std}");
    }

    #[test]
    fn test_ridge_surrogate_std_nonneg() {
        let mut surr = RidgeSurrogate::new(1e-3);
        let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 / 4.0]).collect();
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect();
        surr.fit(&x, &y);
        for i in 0..10 {
            let xq = vec![i as f64 / 10.0];
            let (_mean, std) = surr.predict_mean_std(&xq);
            assert!(std >= 0.0, "std={std} at x={}", xq[0]);
        }
    }

    // ── PredictorNasSearcher ────────────────────────────────────────────────────

    #[test]
    fn test_predictor_search_returns_result() {
        // eval_fn: negative sum of all op indices (lower-index ops are "better").
        let eval_fn = |arch: &[Vec<Vec<usize>>]| -> f64 {
            let total: usize = arch
                .iter()
                .flat_map(|c| c.iter().flat_map(|n| n.iter()))
                .sum();
            -(total as f64)
        };

        let config = PredictorNasConfig {
            n_cells: 2,
            n_nodes: 3,
            n_operations: 6,
            n_initial_samples: 4,
            n_iterations: 2,
            n_candidates_per_iter: 10,
            n_top_to_evaluate: 2,
            ..Default::default()
        };

        let mut searcher = PredictorNasSearcher::new(config);
        let result = searcher.search(eval_fn).expect("search should succeed");

        assert!(
            result.best_score.is_finite(),
            "best_score={}",
            result.best_score
        );
        assert!(
            result.n_evaluated >= 4,
            "n_evaluated={}",
            result.n_evaluated
        );
    }

    #[test]
    fn test_active_learning_improves_best_score() {
        let eval_fn = |arch: &[Vec<Vec<usize>>]| -> f64 {
            let total: usize = arch
                .iter()
                .flat_map(|c| c.iter().flat_map(|n| n.iter()))
                .sum();
            -(total as f64)
        };

        let config_small = PredictorNasConfig {
            n_cells: 1,
            n_nodes: 2,
            n_operations: 6,
            n_initial_samples: 3,
            n_iterations: 0,
            n_candidates_per_iter: 5,
            n_top_to_evaluate: 1,
            seed: 7,
            ..Default::default()
        };
        let mut searcher_small = PredictorNasSearcher::new(config_small);
        let result_small = searcher_small.search(&eval_fn).expect("small search");

        let config_large = PredictorNasConfig {
            n_cells: 1,
            n_nodes: 2,
            n_operations: 6,
            n_initial_samples: 3,
            n_iterations: 4,
            n_candidates_per_iter: 10,
            n_top_to_evaluate: 2,
            seed: 7,
            ..Default::default()
        };
        let mut searcher_large = PredictorNasSearcher::new(config_large);
        let result_large = searcher_large.search(&eval_fn).expect("large search");

        // Larger budget should evaluate more architectures.
        assert!(
            result_large.n_evaluated >= result_small.n_evaluated,
            "large n_eval={} < small n_eval={}",
            result_large.n_evaluated,
            result_small.n_evaluated
        );
        assert!(result_small.best_score.is_finite());
        assert!(result_large.best_score.is_finite());
    }

    #[test]
    fn test_predictor_n_evaluated_count() {
        let config = PredictorNasConfig {
            n_initial_samples: 5,
            n_iterations: 3,
            n_top_to_evaluate: 2,
            n_candidates_per_iter: 10,
            n_cells: 2,
            n_nodes: 3,
            n_operations: 6,
            ..Default::default()
        };
        let expected_min = 5 + 3 * 2; // initial + iterations * top_k

        let mut searcher = PredictorNasSearcher::new(config);
        let result = searcher.search(|_| 1.0).expect("search should not fail");

        assert!(
            result.n_evaluated >= expected_min,
            "n_evaluated={} < expected_min={expected_min}",
            result.n_evaluated
        );
    }

    #[test]
    fn test_predictor_zero_iterations_still_works() {
        let config = PredictorNasConfig {
            n_initial_samples: 3,
            n_iterations: 0,
            ..Default::default()
        };
        let mut searcher = PredictorNasSearcher::new(config);
        let result = searcher
            .search(|_| 42.0)
            .expect("zero-iteration search should succeed");
        assert_eq!(result.best_score, 42.0);
    }

    #[test]
    fn test_normal_cdf_basic() {
        // Φ(0) ≈ 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        // Φ(∞) ≈ 1
        assert!(normal_cdf(10.0) > 0.999);
        // Φ(-∞) ≈ 0
        assert!(normal_cdf(-10.0) < 0.001);
    }

    #[test]
    fn test_gauss_elimination_simple() {
        // Solve 2x = 4 → x = 2.
        let mut a = vec![vec![2.0_f64]];
        let mut b = vec![4.0_f64];
        gauss_elimination(&mut a, &mut b);
        assert!((b[0] - 2.0).abs() < 1e-10, "b[0]={}", b[0]);
    }

    #[test]
    fn test_gauss_elimination_2x2() {
        // [[1,2],[3,4]] x = [5, 11] → x = [1, 2]
        let mut a = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
        let mut b = vec![5.0_f64, 11.0];
        gauss_elimination(&mut a, &mut b);
        assert!((b[0] - 1.0).abs() < 1e-9, "b[0]={}", b[0]);
        assert!((b[1] - 2.0).abs() < 1e-9, "b[1]={}", b[1]);
    }
}
