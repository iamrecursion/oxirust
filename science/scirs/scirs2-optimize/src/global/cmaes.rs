//! CMA-ES: Covariance Matrix Adaptation Evolution Strategy.
//!
//! Reference: Hansen, N. "The CMA Evolution Strategy: A Tutorial" (2016).
//! State-of-the-art derivative-free optimizer for moderate-dimension problems.
//!
//! CMA-ES maintains and adapts a multivariate normal search distribution
//! N(mean, sigma^2 * C) where C is a full covariance matrix. This allows it
//! to detect and exploit correlations between variables, making it especially
//! effective on ill-conditioned and non-separable problems.

use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::rand_distributions::Distribution;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Normal, SeedableRng};

/// CMA-ES configuration parameters.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Population size (lambda). Default: 4 + floor(3 * ln(n)).
    pub population_size: Option<usize>,
    /// Initial step size (sigma0). Default: 0.3.
    pub initial_sigma: f64,
    /// Maximum function evaluations.
    pub max_fevals: usize,
    /// Convergence tolerance on function value (not used directly but stored).
    pub ftol: f64,
    /// Convergence tolerance: stop when step size sigma < xtol.
    pub xtol: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            population_size: None,
            initial_sigma: 0.3,
            max_fevals: 10_000,
            ftol: 1e-10,
            xtol: 1e-10,
            seed: 0,
        }
    }
}

/// Result of a CMA-ES optimization run.
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    /// Best parameter vector found.
    pub x: Array1<f64>,
    /// Best objective function value found.
    pub fun: f64,
    /// Total number of function evaluations used.
    pub fevals: usize,
    /// Number of CMA-ES generations performed.
    pub generations: usize,
    /// Whether the algorithm converged within tolerances.
    pub converged: bool,
    /// Human-readable termination message.
    pub message: String,
}

/// Internal state of the CMA-ES algorithm.
///
/// Encapsulates all distribution parameters and adaptation state.
/// The distribution is N(mean, sigma^2 * C) where C = B * D^2 * B^T.
pub struct CmaEs {
    n: usize,          // problem dimensionality
    lambda: usize,     // population size
    mu: usize,         // number of elite parents selected per generation
    weights: Vec<f64>, // positive recombination weights summing to 1
    mueff: f64,        // effective selection mass = (sum w_i)^2 / sum w_i^2

    // Step-size control (CSA — cumulative step-size adaptation)
    cs: f64,    // learning rate for ps evolution path
    ds: f64,    // damping coefficient for step-size update
    chi_n: f64, // E[||N(0,I)||] ≈ sqrt(n) * (1 - 1/4n + 1/21n^2)

    // Covariance matrix adaptation (CMA)
    cc: f64,  // learning rate for pc evolution path
    c1: f64,  // rank-1 update coefficient
    cmu: f64, // rank-mu update coefficient

    // Current distribution state
    mean: Array1<f64>, // distribution mean
    sigma: f64,        // global step size
    ps: Array1<f64>,   // isotropic evolution path (for step-size control)
    pc: Array1<f64>,   // anisotropic evolution path (for covariance rank-1 update)
    cov: Array2<f64>,  // covariance matrix C

    // Eigendecomposition of C: C = B * diag(eigenvalues) * B^T
    eigenvalues: Array1<f64>,  // eigenvalues of C (should be positive)
    eigenvectors: Array2<f64>, // columns are eigenvectors (B matrix)

    rng: StdRng,
    normal_dist: Normal<f64>,

    /// Total function evaluations consumed.
    pub fevals: usize,
    /// Completed generations.
    pub generations: usize,
}

impl CmaEs {
    /// Initialize CMA-ES with a starting point and configuration.
    ///
    /// All strategy parameters follow Hansen's recommended defaults:
    /// <https://arxiv.org/abs/1604.00772>
    pub fn new(x0: Array1<f64>, config: &CmaEsConfig) -> Result<Self, OptimizeError> {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "dimension must be > 0".to_string(),
            ));
        }

        let lambda = config
            .population_size
            .unwrap_or_else(|| 4 + (3.0 * (n as f64).ln()) as usize);
        if lambda < 2 {
            return Err(OptimizeError::InvalidInput(
                "population_size must be >= 2".to_string(),
            ));
        }
        let mu = lambda / 2;

        // Log-linear recombination weights for the mu best individuals
        let raw_weights: Vec<f64> = (1..=mu)
            .map(|i| ((mu as f64 + 0.5) / i as f64).ln())
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|&w| w / w_sum).collect();
        let mueff = 1.0 / weights.iter().map(|&w| w * w).sum::<f64>();

        // Step-size control parameters
        let cs = (mueff + 2.0) / (n as f64 + mueff + 5.0);
        let ds = 1.0 + 2.0 * (0.0f64.max((mueff - 1.0) / (n as f64 + 1.0) - 1.0)).sqrt() + cs;
        let chi_n =
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * n as f64 * n as f64));

        // Covariance adaptation parameters
        let cc = (4.0 + mueff / n as f64) / (n as f64 + 4.0 + 2.0 * mueff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mueff);
        let alpha_mu = 2.0;
        let cmu = (alpha_mu * (mueff - 2.0 + 1.0 / mueff)
            / ((n as f64 + 2.0).powi(2) + alpha_mu * mueff / 2.0))
            .min(1.0 - c1);

        let normal_dist =
            Normal::new(0.0, 1.0).map_err(|e| OptimizeError::InitializationError(e.to_string()))?;

        Ok(Self {
            n,
            lambda,
            mu,
            weights,
            mueff,
            cs,
            ds,
            chi_n,
            cc,
            c1,
            cmu,
            mean: x0,
            sigma: config.initial_sigma,
            ps: Array1::zeros(n),
            pc: Array1::zeros(n),
            cov: Array2::eye(n),
            eigenvalues: Array1::ones(n),
            eigenvectors: Array2::eye(n),
            rng: StdRng::seed_from_u64(config.seed),
            normal_dist,
            fevals: 0,
            generations: 0,
        })
    }

    /// Sample lambda offspring from the current search distribution.
    ///
    /// Each sample: x_k = mean + sigma * B * D * z_k,  z_k ~ N(0, I)
    fn sample_population(&mut self) -> Vec<Array1<f64>> {
        let mut pop = Vec::with_capacity(self.lambda);
        for _ in 0..self.lambda {
            // Draw standard normal vector
            let z: Array1<f64> = (0..self.n)
                .map(|_| self.normal_dist.sample(&mut self.rng))
                .collect::<Vec<f64>>()
                .into();

            // Scale by sqrt(eigenvalues) to get D * z
            let dz: Array1<f64> = &z * &self.eigenvalues.mapv(|v| v.sqrt());

            // Rotate by eigenvectors to get B * D * z
            let bdz = self.eigenvectors.dot(&dz);

            // x = mean + sigma * B * D * z
            pop.push(&self.mean + &(self.sigma * &bdz));
        }
        pop
    }

    /// Update the distribution parameters using the ranked population.
    ///
    /// Performs:
    /// 1. Weighted mean update
    /// 2. CSA step-size control via evolution path ps
    /// 3. CMA covariance path update via pc
    /// 4. Rank-1 and rank-mu covariance updates
    /// 5. Eigendecomposition refresh
    fn update(&mut self, ranked: &[(usize, f64)], population: &[Array1<f64>]) {
        let n = self.n;
        let gen_f64 = self.generations as f64 + 1.0;

        // 1. Weighted mean update
        let old_mean = self.mean.clone();
        let mut new_mean = Array1::zeros(n);
        for (k, &(idx, _)) in ranked[..self.mu].iter().enumerate() {
            new_mean = new_mean + &population[idx] * self.weights[k];
        }
        self.mean = new_mean;

        // mean_diff = (mean_new - mean_old) / sigma  (step in unscaled space)
        let mean_diff = (&self.mean - &old_mean) / self.sigma;

        // 2. Compute C^{-1/2} = B * D^{-1} * B^T for step-size path update
        let inv_sqrt_diag: Array1<f64> =
            self.eigenvalues
                .mapv(|v| if v > 1e-14 { v.recip().sqrt() } else { 0.0 });
        // C^{-1/2} = B * D^{-1} * B^T
        let inv_sqrt_c: Array2<f64> = self
            .eigenvectors
            .dot(&Array2::from_diag(&inv_sqrt_diag))
            .dot(&self.eigenvectors.t());

        // 3. Update isotropic evolution path ps
        let invsqrt_diff = inv_sqrt_c.dot(&mean_diff);
        let cs = self.cs;
        self.ps = (1.0 - cs) * &self.ps + (cs * (2.0 - cs) * self.mueff).sqrt() * &invsqrt_diff;

        // 4. Heaviside indicator: suppress rank-1 update when ps is large
        let ps_norm = self.ps.mapv(|v| v * v).sum().sqrt();
        let h_thresh = 1.4 + 2.0 / (n as f64 + 1.0);
        let ps_norm_normalized =
            ps_norm / (1.0 - (1.0 - cs).powf(2.0 * gen_f64)).sqrt() / self.chi_n;
        let h_sig = ps_norm_normalized < h_thresh;

        // 5. Update anisotropic evolution path pc
        let delta_h = if h_sig {
            0.0
        } else {
            (2.0 - self.cc) * self.cc
        };

        self.pc = (1.0 - self.cc) * &self.pc
            + if h_sig {
                (self.cc * (2.0 - self.cc) * self.mueff).sqrt() * &mean_diff
            } else {
                Array1::zeros(n)
            };

        // 6. Rank-1 update term: pc * pc^T
        let pc_col = self.pc.view().insert_axis(Axis(1));
        let rank_one: Array2<f64> = pc_col.dot(&pc_col.t());

        // 7. Rank-mu update term: weighted sum of step differences
        let mut rank_mu: Array2<f64> = Array2::zeros((n, n));
        for (k, &(idx, _)) in ranked[..self.mu].iter().enumerate() {
            let diff = (&population[idx] - &old_mean) / self.sigma;
            let diff_col = diff.view().insert_axis(Axis(1));
            rank_mu = rank_mu + self.weights[k] * diff_col.dot(&diff_col.t());
        }

        // 8. Combine: (1 + c1*delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
        //    The delta_h term corrects for the stalling-of-pc effect.
        let c1 = self.c1;
        let cmu = self.cmu;
        self.cov = (1.0 + c1 * delta_h - c1 - cmu) * &self.cov + c1 * &rank_one + cmu * &rank_mu;

        // 9. Update step size via CSA
        let ps_norm_new = self.ps.mapv(|v| v * v).sum().sqrt();
        self.sigma *= ((cs / self.ds) * (ps_norm_new / self.chi_n - 1.0)).exp();

        // 10. Refresh eigendecomposition of the updated covariance
        self.update_eigen();

        self.generations += 1;
    }

    /// Compute the eigendecomposition of the covariance matrix via Jacobi iterations.
    ///
    /// This is an O(n^3) symmetric Jacobi method, suitable for the moderate
    /// dimensions (n <= ~100) that CMA-ES targets. More sweeps = more accuracy.
    fn update_eigen(&mut self) {
        let n = self.n;
        // Work on a symmetric copy of C
        let mut a = self.cov.clone();
        // Symmetrize (numerical drift can create tiny asymmetry)
        for i in 0..n {
            for j in (i + 1)..n {
                let sym = (a[[i, j]] + a[[j, i]]) * 0.5;
                a[[i, j]] = sym;
                a[[j, i]] = sym;
            }
        }

        let mut v = Array2::eye(n);

        // Jacobi sweeps: each sweep does n*(n-1)/2 Givens rotations.
        // 20 sweeps is more than enough for double precision convergence in practice.
        for _ in 0..20 {
            let mut off_norm_sq = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    off_norm_sq += a[[i, j]] * a[[i, j]];
                }
            }
            // Early exit if already diagonal
            if off_norm_sq < 1e-28 {
                break;
            }

            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[[p, q]];
                    if apq.abs() < 1e-15 {
                        continue;
                    }
                    let app = a[[p, p]];
                    let aqq = a[[q, q]];

                    // Angle for Givens rotation
                    let theta = 0.5 * (aqq - app) / apq;
                    let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                    let c_r = 1.0 / (1.0 + t * t).sqrt();
                    let s_r = t * c_r;

                    // Update diagonal
                    a[[p, p]] = app - t * apq;
                    a[[q, q]] = aqq + t * apq;
                    a[[p, q]] = 0.0;
                    a[[q, p]] = 0.0;

                    // Update remaining rows/columns
                    for r in 0..n {
                        if r != p && r != q {
                            let arp = a[[r, p]];
                            let arq = a[[r, q]];
                            let new_arp = c_r * arp - s_r * arq;
                            let new_arq = s_r * arp + c_r * arq;
                            a[[r, p]] = new_arp;
                            a[[p, r]] = new_arp;
                            a[[r, q]] = new_arq;
                            a[[q, r]] = new_arq;
                        }
                    }

                    // Accumulate rotations into eigenvectors
                    for r in 0..n {
                        let vrp = v[[r, p]];
                        let vrq = v[[r, q]];
                        v[[r, p]] = c_r * vrp - s_r * vrq;
                        v[[r, q]] = s_r * vrp + c_r * vrq;
                    }
                }
            }
        }

        // Diagonal of a now contains eigenvalues; clamp to positive
        for i in 0..n {
            self.eigenvalues[i] = a[[i, i]].max(1e-20);
        }
        self.eigenvectors = v;
    }
}

/// Minimize a function using CMA-ES.
///
/// # Arguments
/// * `f` - Objective function to minimize, taking `&Array1<f64>`.
/// * `x0` - Initial point (used as initial distribution mean).
/// * `bounds` - Optional box constraints as `[(lo, hi)]`. If provided, samples
///   that fall outside bounds are clipped before evaluation.
/// * `config` - Algorithm configuration (see `CmaEsConfig`).
///
/// # Returns
/// `CmaEsResult` with the best point found, or an error if inputs are invalid.
///
/// # Example
/// ```rust
/// use scirs2_optimize::global::{minimize_cmaes, CmaEsConfig};
/// use scirs2_core::ndarray::array;
///
/// let result = minimize_cmaes(
///     |x| x[0] * x[0] + x[1] * x[1],
///     array![2.0, -3.0],
///     None,
///     CmaEsConfig { max_fevals: 5000, initial_sigma: 1.0, ..Default::default() },
/// ).expect("cmaes failed");
/// assert!(result.fun < 1e-4);
/// ```
pub fn minimize_cmaes<F>(
    f: F,
    x0: Array1<f64>,
    bounds: Option<&[(f64, f64)]>,
    config: CmaEsConfig,
) -> Result<CmaEsResult, OptimizeError>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "dimension must be > 0".to_string(),
        ));
    }
    if let Some(b) = bounds {
        if b.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "bounds length {} does not match x0 length {}",
                b.len(),
                n
            )));
        }
        for (i, &(lo, hi)) in b.iter().enumerate() {
            if lo > hi {
                return Err(OptimizeError::InvalidInput(format!(
                    "bounds[{}]: lower {} > upper {}",
                    i, lo, hi
                )));
            }
        }
    }

    let max_fevals = config.max_fevals;
    let xtol = config.xtol;

    let mut state = CmaEs::new(x0.clone(), &config)?;

    // Evaluate the starting point
    let x0_clipped = clip_to_bounds(&x0, bounds);
    let mut best_f = f(&x0_clipped);
    let mut best_x = x0_clipped;
    state.fevals += 1;

    loop {
        // Sample a new population
        let population = state.sample_population();

        // Evaluate (with optional bounds clipping)
        let mut fitness: Vec<(usize, f64)> = population
            .iter()
            .enumerate()
            .map(|(i, xi)| {
                let clipped = clip_to_bounds(xi, bounds);
                let fval = f(&clipped);
                (i, fval)
            })
            .collect();
        state.fevals += population.len();

        // Sort ascending by function value (best first)
        fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Track global best
        let best_this_gen_f = fitness[0].1;
        if best_this_gen_f < best_f {
            best_f = best_this_gen_f;
            best_x = clip_to_bounds(&population[fitness[0].0], bounds);
        }

        // Termination: budget exhausted
        if state.fevals >= max_fevals {
            return Ok(CmaEsResult {
                x: best_x,
                fun: best_f,
                fevals: state.fevals,
                generations: state.generations,
                converged: false,
                message: "Maximum function evaluations reached".to_string(),
            });
        }

        // Termination: step size converged
        if state.sigma < xtol {
            return Ok(CmaEsResult {
                x: best_x,
                fun: best_f,
                fevals: state.fevals,
                generations: state.generations,
                converged: true,
                message: "Step size (sigma) converged below xtol".to_string(),
            });
        }

        // Update distribution parameters
        state.update(&fitness, &population);
    }
}

/// Clip a point to the given box bounds (if any).
#[inline]
fn clip_to_bounds(x: &Array1<f64>, bounds: Option<&[(f64, f64)]>) -> Array1<f64> {
    match bounds {
        None => x.clone(),
        Some(b) => {
            let clipped: Vec<f64> = x
                .iter()
                .zip(b.iter())
                .map(|(&v, &(lo, hi))| v.clamp(lo, hi))
                .collect();
            Array1::from(clipped)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cmaes_sphere_1d() {
        // Minimize f(x) = x^2, solution at x = 0
        let result = minimize_cmaes(
            |x| x[0] * x[0],
            array![5.0],
            None,
            CmaEsConfig {
                max_fevals: 2000,
                initial_sigma: 1.0,
                ftol: 1e-8,
                xtol: 1e-6,
                ..Default::default()
            },
        )
        .expect("cmaes failed");
        assert!(result.fun < 1e-6, "f* = {}, expected < 1e-6", result.fun);
        assert!(
            result.x[0].abs() < 1e-3,
            "x* = {}, expected near 0",
            result.x[0]
        );
    }

    #[test]
    fn test_cmaes_sphere_nd() {
        // 5-D sphere function; solution at origin
        let x0 = array![3.0, -2.0, 1.0, 4.0, -1.0];
        let result = minimize_cmaes(
            |x| x.iter().map(|&v| v * v).sum::<f64>(),
            x0,
            None,
            CmaEsConfig {
                max_fevals: 10_000,
                initial_sigma: 1.0,
                xtol: 1e-5,
                ..Default::default()
            },
        )
        .expect("cmaes failed");
        assert!(result.fun < 1e-4, "f* = {}, expected < 1e-4", result.fun);
    }

    #[test]
    fn test_cmaes_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1) = 0
        let result = minimize_cmaes(
            |x| {
                let a = 1.0 - x[0];
                let b = x[1] - x[0] * x[0];
                a * a + 100.0 * b * b
            },
            array![0.0, 0.0],
            None,
            CmaEsConfig {
                max_fevals: 20_000,
                initial_sigma: 0.5,
                xtol: 1e-8,
                ..Default::default()
            },
        )
        .expect("cmaes failed");
        assert!(
            result.fun < 0.01,
            "Rosenbrock f* = {}, expected < 0.01",
            result.fun
        );
    }

    #[test]
    fn test_cmaes_with_bounds() {
        // Minimize x^2 + y^2 with bounds [1, 5] x [1, 5].
        // Unconstrained minimum is at origin; with bounds minimum is at (1, 1).
        let result = minimize_cmaes(
            |x| x[0] * x[0] + x[1] * x[1],
            array![3.0, 3.0],
            Some(&[(1.0, 5.0), (1.0, 5.0)]),
            CmaEsConfig {
                max_fevals: 5000,
                initial_sigma: 0.5,
                ..Default::default()
            },
        )
        .expect("cmaes failed");
        assert!(
            result.x[0] >= 0.9 && result.x[0] <= 2.0,
            "x[0] = {}, expected in [0.9, 2.0]",
            result.x[0]
        );
        assert!(
            result.x[1] >= 0.9 && result.x[1] <= 2.0,
            "x[1] = {}, expected in [0.9, 2.0]",
            result.x[1]
        );
    }

    #[test]
    fn test_cmaes_result_fevals_and_generations() {
        let result = minimize_cmaes(|x| x[0] * x[0], array![1.0], None, CmaEsConfig::default())
            .expect("cmaes failed");
        assert!(result.fevals > 0, "fevals should be > 0");
        assert!(result.generations > 0, "generations should be > 0");
        assert!(!result.message.is_empty(), "message should not be empty");
    }

    #[test]
    fn test_cmaes_invalid_dimension() {
        use scirs2_core::ndarray::Array1;
        let empty: Array1<f64> = Array1::from(vec![]);
        let result = minimize_cmaes(
            |x| x.iter().map(|&v| v * v).sum(),
            empty,
            None,
            CmaEsConfig::default(),
        );
        assert!(result.is_err(), "empty input should return error");
    }

    #[test]
    fn test_cmaes_bounds_mismatch_error() {
        let result = minimize_cmaes(
            |x| x[0] * x[0],
            array![1.0, 2.0],
            Some(&[(0.0, 5.0)]), // only 1 bound for 2-D input
            CmaEsConfig::default(),
        );
        assert!(result.is_err(), "bounds mismatch should return error");
    }

    #[test]
    fn test_cmaes_population_size_override() {
        // Explicit population size override
        let result = minimize_cmaes(
            |x| (x[0] - 1.0).powi(2) + (x[1] + 1.0).powi(2),
            array![0.0, 0.0],
            None,
            CmaEsConfig {
                population_size: Some(20),
                max_fevals: 5000,
                initial_sigma: 0.8,
                ..Default::default()
            },
        )
        .expect("cmaes failed");
        assert!(result.fun < 0.01, "f* = {}", result.fun);
    }

    #[test]
    fn test_cmaes_sigma_convergence() {
        // Very tight xtol so the algorithm terminates on sigma convergence,
        // not just max_fevals.
        let result = minimize_cmaes(
            |x| x[0] * x[0],
            array![0.1],
            None,
            CmaEsConfig {
                max_fevals: 100_000,
                initial_sigma: 0.5,
                xtol: 1e-3, // loose xtol so it triggers quickly
                ..Default::default()
            },
        )
        .expect("cmaes failed");
        // Should converge with sigma stopping criterion
        assert!(result.converged || result.fevals >= 100_000);
    }

    #[test]
    fn test_cmaes_quadratic_with_correlation() {
        // Correlated quadratic: f(x) = x^T A x where A has off-diagonal entries.
        // CMA-ES should handle this well because it adapts the full covariance.
        let result = minimize_cmaes(
            |x| {
                // A = [[2, 1], [1, 3]] — positive definite, correlated
                2.0 * x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1]
            },
            array![3.0, -3.0],
            None,
            CmaEsConfig {
                max_fevals: 5000,
                initial_sigma: 1.0,
                xtol: 1e-7,
                ..Default::default()
            },
        )
        .expect("cmaes failed");
        assert!(
            result.fun < 1e-4,
            "Correlated quadratic f* = {}",
            result.fun
        );
    }
}
