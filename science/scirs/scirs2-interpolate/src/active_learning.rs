//! Active sampling to minimise interpolation error.
//!
//! Provides a model-free, acquisition-function–driven strategy for selecting
//! the next point to query in order to reduce interpolation error most
//! efficiently.  Three acquisition strategies are supported:
//!
//! - **MaximumVariance**: maximise the GP posterior variance at the candidate
//!   points — pure exploration.
//! - **ExpectedImprovement**: standard EI using GP posterior mean and variance.
//! - **LeverageScore**: statistical leverage score of the candidate against the
//!   kernel matrix formed by the observed points.
//!
//! Candidate points are generated using a deterministic quasi-random sequence
//! (XorShift64-based), ensuring reproducibility.
//!
//! ## References
//!
//! - Settles, B. (2009). *Active Learning Literature Survey*.
//! - Srinivas, N. et al. (2010). *Gaussian Process Optimization in the Bandit
//!   Setting: No Regret and Experimental Design*.

use crate::error::InterpolateError;

// ---------------------------------------------------------------------------
// Acquisition function enum
// ---------------------------------------------------------------------------

/// Acquisition function used to rank candidate query points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveAcquisitionFunction {
    /// Select the point with maximum GP posterior variance (pure exploration).
    MaximumVariance,
    /// Standard Expected Improvement (exploits current best observation).
    ExpectedImprovement,
    /// Statistical leverage score: measures the influence of a new point on the
    /// Gram matrix.
    LeverageScore,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ActiveSampler`].
#[derive(Debug, Clone)]
pub struct ActiveSamplerConfig {
    /// Acquisition function to use when ranking candidate points.
    pub acquisition: ActiveAcquisitionFunction,
    /// Number of candidate points sampled per `suggest_next` call.
    pub n_candidates: usize,
    /// Domain bounds for each dimension: `domain[d] = [min, max]`.
    pub domain: Vec<[f64; 2]>,
    /// Seed for the candidate generator.
    pub seed: u64,
}

impl Default for ActiveSamplerConfig {
    fn default() -> Self {
        Self {
            acquisition: ActiveAcquisitionFunction::MaximumVariance,
            n_candidates: 64,
            domain: vec![[0.0, 1.0], [0.0, 1.0]],
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// ActiveSampler
// ---------------------------------------------------------------------------

/// Active sampling strategy for minimising interpolation error.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::active_learning::{
///     ActiveSampler, ActiveSamplerConfig, ActiveAcquisitionFunction,
/// };
///
/// let config = ActiveSamplerConfig {
///     acquisition: ActiveAcquisitionFunction::MaximumVariance,
///     n_candidates: 20,
///     domain: vec![[0.0, 1.0], [0.0, 1.0]],
///     seed: 7,
/// };
/// let mut sampler = ActiveSampler::new(config);
///
/// // Seed with one observation
/// sampler.observe(vec![0.5, 0.5], 1.0);
///
/// let next = sampler.suggest_next();
/// assert_eq!(next.len(), 2);
/// ```
#[derive(Debug)]
pub struct ActiveSampler {
    config: ActiveSamplerConfig,
    observed_points: Vec<Vec<f64>>,
    observed_values: Vec<f64>,
    n_dims: usize,
}

impl ActiveSampler {
    /// Create a new sampler.  The number of dimensions is inferred from
    /// `config.domain.len()`.
    pub fn new(config: ActiveSamplerConfig) -> Self {
        let n_dims = config.domain.len().max(1);
        Self {
            config,
            observed_points: Vec::new(),
            observed_values: Vec::new(),
            n_dims,
        }
    }

    /// Select the next query point by evaluating the acquisition function at
    /// `config.n_candidates` randomly sampled candidate points.
    ///
    /// If no candidates score above zero, returns the first candidate (random).
    pub fn suggest_next(&self) -> Vec<f64> {
        let mut rng = XorShift64::new(self.config.seed.wrapping_add(self.n_observed() as u64));
        let candidates =
            generate_candidates(&self.config.domain, self.config.n_candidates, &mut rng);

        if candidates.is_empty() {
            // Fallback: return domain centre
            return self
                .config
                .domain
                .iter()
                .map(|&[lo, hi]| 0.5 * (lo + hi))
                .collect();
        }

        // Rank candidates
        let best = candidates.iter().cloned().enumerate().fold(
            (0usize, f64::NEG_INFINITY),
            |(bi, bv), (i, ref cand)| {
                let score = self.acquisition_value(cand);
                if score > bv {
                    (i, score)
                } else {
                    (bi, bv)
                }
            },
        );

        candidates.into_iter().nth(best.0).unwrap_or_else(|| {
            self.config
                .domain
                .iter()
                .map(|&[lo, hi]| 0.5 * (lo + hi))
                .collect()
        })
    }

    /// Register a new observation.
    pub fn observe(&mut self, point: Vec<f64>, value: f64) {
        self.observed_points.push(point);
        self.observed_values.push(value);
    }

    /// Compute the acquisition value for a single candidate `point`.
    ///
    /// Returns 0.0 when there are no observations.
    pub fn acquisition_value(&self, point: &[f64]) -> f64 {
        if self.observed_points.is_empty() {
            return 1.0; // no data → treat every point as equally informative
        }
        match self.config.acquisition {
            ActiveAcquisitionFunction::MaximumVariance => {
                gp_posterior_variance(&self.observed_points, &self.observed_values, point, 1e-6)
            }
            ActiveAcquisitionFunction::ExpectedImprovement => {
                expected_improvement(&self.observed_points, &self.observed_values, point, 1e-6)
            }
            ActiveAcquisitionFunction::LeverageScore => {
                leverage_score(&self.observed_points, point, 1e-6)
            }
        }
    }

    /// Leave-one-out cross-validation error estimate.
    ///
    /// For each observed point, fits a simple GP on the remaining n-1 points
    /// and measures the squared prediction error.  Returns the RMS LOO error.
    ///
    /// Returns 0.0 when fewer than 2 observations are available.
    pub fn loo_error(&self) -> f64 {
        let n = self.observed_points.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum_sq = 0.0_f64;
        for leave_out in 0..n {
            // Collect remaining points
            let rem_pts: Vec<Vec<f64>> = self
                .observed_points
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != leave_out)
                .map(|(_, p)| p.clone())
                .collect();
            let rem_vals: Vec<f64> = self
                .observed_values
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != leave_out)
                .map(|(_, &v)| v)
                .collect();

            // Predict the left-out point
            let pred =
                gp_posterior_mean(&rem_pts, &rem_vals, &self.observed_points[leave_out], 1e-6);
            let err = pred - self.observed_values[leave_out];
            sum_sq += err * err;
        }
        (sum_sq / n as f64).sqrt()
    }

    /// Number of observations recorded so far.
    pub fn n_observed(&self) -> usize {
        self.observed_points.len()
    }

    /// Slice of all observed points.
    pub fn observed_points(&self) -> &[Vec<f64>] {
        &self.observed_points
    }

    /// Dimensionality of the domain.
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }
}

// ---------------------------------------------------------------------------
// GP helper functions
// ---------------------------------------------------------------------------

/// Squared-exponential RBF kernel: k(x, x') = exp(-‖x-x'‖² / (2 l²)).
pub fn rbf_kernel_sq(x1: &[f64], x2: &[f64], length_scale: f64) -> f64 {
    let sq_dist: f64 = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    (-sq_dist / (2.0 * length_scale * length_scale)).exp()
}

/// GP posterior variance at `query` given observations.
///
/// Uses a zero-mean GP with SE kernel.  Solves K w = k_star via Gaussian
/// elimination.  Returns max(0, k_star_star - k_star^T (K + nugget I)^{-1} k_star).
pub fn gp_posterior_variance(
    obs_points: &[Vec<f64>],
    obs_vals: &[f64],
    query: &[f64],
    nugget: f64,
) -> f64 {
    let n = obs_points.len();
    if n == 0 {
        return 1.0;
    }
    let ls = auto_length_scale(obs_points);

    // k* = kernel vector between query and observations
    let k_star: Vec<f64> = obs_points
        .iter()
        .map(|p| rbf_kernel_sq(query, p, ls))
        .collect();

    // K + nugget I
    let k_mat = build_kernel_matrix(obs_points, ls, nugget);

    // Solve (K + σI) alpha = k_star  →  alpha = (K+σI)^{-1} k_star
    let alpha = match crate::gpu_rbf::solve_linear_system(&k_mat, &k_star, n) {
        Ok(a) => a,
        Err(_) => return 1.0, // fallback on singular matrix
    };

    let reduction: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();
    let k_ss = rbf_kernel_sq(query, query, ls); // = 1.0 for SE kernel
    let var = k_ss - reduction;
    var.max(0.0)
}

/// GP posterior mean at `query`.
fn gp_posterior_mean(obs_points: &[Vec<f64>], obs_vals: &[f64], query: &[f64], nugget: f64) -> f64 {
    let n = obs_points.len();
    if n == 0 {
        return 0.0;
    }
    let ls = auto_length_scale(obs_points);
    let k_star: Vec<f64> = obs_points
        .iter()
        .map(|p| rbf_kernel_sq(query, p, ls))
        .collect();
    let k_mat = build_kernel_matrix(obs_points, ls, nugget);
    let alpha = match crate::gpu_rbf::solve_linear_system(&k_mat, obs_vals, n) {
        Ok(a) => a,
        Err(_) => return 0.0,
    };
    k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum()
}

/// Expected Improvement acquisition function.
///
/// EI(x) = (μ - y_best) Φ(z) + σ φ(z)  where z = (μ - y_best) / σ.
fn expected_improvement(
    obs_points: &[Vec<f64>],
    obs_vals: &[f64],
    query: &[f64],
    nugget: f64,
) -> f64 {
    if obs_vals.is_empty() {
        return 1.0;
    }
    let y_best = obs_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let ls = auto_length_scale(obs_points);
    let n = obs_points.len();
    let k_star: Vec<f64> = obs_points
        .iter()
        .map(|p| rbf_kernel_sq(query, p, ls))
        .collect();
    let k_mat = build_kernel_matrix(obs_points, ls, nugget);

    let alpha = match crate::gpu_rbf::solve_linear_system(&k_mat, obs_vals, n) {
        Ok(a) => a,
        Err(_) => return 0.0,
    };
    let mu: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

    // Variance
    let alpha_v = match crate::gpu_rbf::solve_linear_system(&k_mat, &k_star, n) {
        Ok(a) => a,
        Err(_) => return 0.0,
    };
    let reduction: f64 = k_star.iter().zip(alpha_v.iter()).map(|(k, a)| k * a).sum();
    let sigma2 = (rbf_kernel_sq(query, query, ls) - reduction).max(1e-18);
    let sigma = sigma2.sqrt();

    let z = (y_best - mu) / sigma;
    // Φ(z) and φ(z) via erf approximation
    let phi_z = 0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2));
    let pdf_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let ei = (y_best - mu) * phi_z + sigma * pdf_z;
    ei.max(0.0)
}

/// Statistical leverage score of `query` given observed points.
///
/// The leverage score measures how much the new point would influence the
/// Gram matrix: h = k*^T (K + σI)^{-1} k*.
fn leverage_score(obs_points: &[Vec<f64>], query: &[f64], nugget: f64) -> f64 {
    let n = obs_points.len();
    if n == 0 {
        return 1.0;
    }
    let ls = auto_length_scale(obs_points);
    let k_star: Vec<f64> = obs_points
        .iter()
        .map(|p| rbf_kernel_sq(query, p, ls))
        .collect();
    let k_mat = build_kernel_matrix(obs_points, ls, nugget);
    let alpha = match crate::gpu_rbf::solve_linear_system(&k_mat, &k_star, n) {
        Ok(a) => a,
        Err(_) => return 0.0,
    };
    k_star
        .iter()
        .zip(alpha.iter())
        .map(|(k, a)| k * a)
        .sum::<f64>()
        .max(0.0)
}

// ---------------------------------------------------------------------------
// Candidate generation
// ---------------------------------------------------------------------------

/// XorShift64 PRNG (local copy for this module).
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 + 0.5) / (u64::MAX as f64 + 1.0)
    }
}

/// Generate `n` uniformly random candidate points inside `domain` from a seed.
pub fn generate_candidates_with_seed(domain: &[[f64; 2]], n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = XorShift64::new(seed);
    generate_candidates(domain, n, &mut rng)
}

/// Generate `n` uniformly random candidate points inside `domain`.
fn generate_candidates(domain: &[[f64; 2]], n: usize, rng: &mut XorShift64) -> Vec<Vec<f64>> {
    if domain.is_empty() || n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|_| {
            domain
                .iter()
                .map(|&[lo, hi]| lo + rng.next_f64() * (hi - lo))
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Numerical helpers
// ---------------------------------------------------------------------------

/// Build SE kernel matrix K_{ij} = k(x_i, x_j) + nugget * δ_{ij}.
fn build_kernel_matrix(obs_points: &[Vec<f64>], ls: f64, nugget: f64) -> Vec<f64> {
    let n = obs_points.len();
    let mut k = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = rbf_kernel_sq(&obs_points[i], &obs_points[j], ls);
        }
        k[i * n + i] += nugget;
    }
    k
}

/// Simple heuristic for the SE length-scale: median pairwise distance / √2.
fn auto_length_scale(points: &[Vec<f64>]) -> f64 {
    let n = points.len();
    if n <= 1 {
        return 1.0;
    }
    let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d2: f64 = points[i]
                .iter()
                .zip(points[j].iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            dists.push(d2.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = if dists.is_empty() {
        1.0
    } else {
        dists[dists.len() / 2]
    };
    (med / std::f64::consts::SQRT_2).max(1e-6)
}

/// Approximation of erf(x) using Abramowitz & Stegun formula 7.1.26.
fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sampler(seed: u64) -> ActiveSampler {
        ActiveSampler::new(ActiveSamplerConfig {
            acquisition: ActiveAcquisitionFunction::MaximumVariance,
            n_candidates: 50,
            domain: vec![[0.0, 1.0], [0.0, 1.0]],
            seed,
        })
    }

    /// suggest_next must return a point within the domain bounds.
    #[test]
    fn test_suggest_next_within_domain() {
        let mut sampler = make_sampler(42);
        sampler.observe(vec![0.5, 0.5], 1.0);
        let next = sampler.suggest_next();
        assert_eq!(next.len(), 2, "suggested point should have 2 dimensions");
        let domain = &sampler.config.domain;
        for (d, &v) in next.iter().enumerate() {
            assert!(
                v >= domain[d][0] && v <= domain[d][1],
                "dim {d}: {v} not in [{}, {}]",
                domain[d][0],
                domain[d][1]
            );
        }
    }

    /// observe increases n_observed by 1 each time.
    #[test]
    fn test_observe_increments_count() {
        let mut sampler = make_sampler(1);
        assert_eq!(sampler.n_observed(), 0);
        sampler.observe(vec![0.1, 0.2], 0.5);
        assert_eq!(sampler.n_observed(), 1);
        sampler.observe(vec![0.8, 0.3], 1.5);
        assert_eq!(sampler.n_observed(), 2);
    }

    /// loo_error changes after adding a new observation.
    #[test]
    fn test_loo_error_changes_after_observation() {
        let mut sampler = make_sampler(3);
        sampler.observe(vec![0.0, 0.0], 0.0);
        sampler.observe(vec![1.0, 0.0], 1.0);
        sampler.observe(vec![0.5, 1.0], 0.5);

        let err_before = sampler.loo_error();

        sampler.observe(vec![0.2, 0.8], 0.2);

        let err_after = sampler.loo_error();
        // The two errors should differ (adding a point changes the LOO estimate)
        // We allow them to be the same only by coincidence, so just check they're finite
        assert!(err_before.is_finite(), "loo_error before should be finite");
        assert!(err_after.is_finite(), "loo_error after should be finite");
        // At least one of them should be non-zero given non-trivial data
        assert!(
            err_before != err_after || err_after == 0.0,
            "loo_error should change (or be 0) after new observation"
        );
    }

    /// Two different seeds should yield different suggested points.
    #[test]
    fn test_different_seeds_different_suggestions() {
        let mut s1 = make_sampler(7);
        let mut s2 = make_sampler(99999);
        s1.observe(vec![0.5, 0.5], 1.0);
        s2.observe(vec![0.5, 0.5], 1.0);

        let n1 = s1.suggest_next();
        let n2 = s2.suggest_next();
        let differ = n1.iter().zip(n2.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            differ,
            "Different seeds should produce different suggested points (got {:?} and {:?})",
            n1, n2
        );
    }

    /// ExpectedImprovement acquisition returns non-negative values.
    #[test]
    fn test_ei_non_negative() {
        let mut sampler = ActiveSampler::new(ActiveSamplerConfig {
            acquisition: ActiveAcquisitionFunction::ExpectedImprovement,
            n_candidates: 20,
            domain: vec![[0.0, 1.0]],
            seed: 5,
        });
        sampler.observe(vec![0.3], 2.0);
        sampler.observe(vec![0.7], 1.0);

        for x in [0.1, 0.5, 0.9] {
            let v = sampler.acquisition_value(&[x]);
            assert!(v >= 0.0, "EI must be non-negative, got {v} at x={x}");
        }
    }

    /// LeverageScore acquisition returns values in [0, 1].
    #[test]
    fn test_leverage_score_range() {
        let mut sampler = ActiveSampler::new(ActiveSamplerConfig {
            acquisition: ActiveAcquisitionFunction::LeverageScore,
            n_candidates: 20,
            domain: vec![[0.0, 1.0], [0.0, 1.0]],
            seed: 10,
        });
        sampler.observe(vec![0.2, 0.3], 1.0);
        sampler.observe(vec![0.8, 0.7], 2.0);

        let v = sampler.acquisition_value(&[0.5, 0.5]);
        assert!(
            v >= 0.0 && v <= 1.0 + 1e-10,
            "leverage score should be in [0, 1], got {v}"
        );
    }

    /// rbf_kernel_sq at identical points returns 1.0.
    #[test]
    fn test_rbf_kernel_sq_at_zero() {
        let x = vec![0.3, 0.7];
        let v = rbf_kernel_sq(&x, &x, 1.0);
        assert!(
            (v - 1.0).abs() < 1e-15,
            "SE kernel at r=0 should be 1.0, got {v}"
        );
    }
}
