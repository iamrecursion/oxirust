//! Variational Bayes Gaussian Mixture Model (VBEM).
//!
//! Implements the variational EM (Variational Bayes EM) algorithm for a
//! univariate Gaussian mixture model with conjugate Dirichlet / Gaussian
//! priors. The derivation follows Bishop (2006), Chapter 10 and the original
//! Attias (1999) / Ghahramani & Beal (2000) formulations.
//!
//! The generative model is:
//!
//! ```text
//!   π ~ Dirichlet(α₀, …, α₀)           (mixing proportions)
//!   μ_k ~ N(m₀, 1/β₀)  k = 1, …, K     (component means, iid)
//!   z_n | π ~ Categorical(π)             (latent assignments)
//!   x_n | z_n = k, μ_k ~ N(μ_k, 1/τ_k) (observed data, precision τ_k known)
//! ```
//!
//! The mean-field variational family factorises as:
//!
//! ```text
//!   q(π, μ₁, …, μ_K, z) = q(π) · Π_k q(μ_k) · Π_n q(z_n)
//! ```
//!
//! The algorithm is self-contained (does not wire into the generic
//! [`super::engine::VariationalMessagePassing`] engine) and follows the same
//! standalone pattern as [`super::gamma`] and [`super::beta`].
//!
//! # References
//!
//! - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*,
//!   §10.2 "Variational mixture of Gaussians".
//! - Attias, H. (1999). Inferring parameters and structure of latent variable
//!   models by variational Bayes. UAI-15.
//! - Ghahramani, Z. & Beal, M. J. (2000). Variational inference for Bayesian
//!   mixtures of factor analysers. NIPS 12.

use crate::error::{PgmError, Result};
use scirs2_core::random::{RngExt, SeedableRng, StdRng};

use super::distributions::{dirichlet_kl, gaussian_kl, DirichletNP, GaussianNP};
use super::exponential_family::ExponentialFamily;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`VariationalGaussianMixture`].
///
/// All fields are validated on [`VariationalGaussianMixture::new`] via the
/// private `VgmmConfig::validate` method. Use the builder helpers to
/// construct a config; the [`VgmmConfig::new`] constructor provides sensible
/// research-preview defaults.
#[derive(Clone, Debug)]
pub struct VgmmConfig {
    /// Number of mixture components K (must be ≥ 1).
    pub n_components: usize,
    /// Symmetric Dirichlet prior concentration α₀ > 0 for each component.
    pub prior_concentration: f64,
    /// Prior mean m₀ for each component's Gaussian prior.
    pub prior_mean: f64,
    /// Prior precision β₀ > 0 for each component's Gaussian prior.
    pub prior_precision: f64,
    /// Shared observation precision τ, used when `component_precisions` is
    /// `None`. All `τ_k` are set to this value.
    pub observation_precision: f64,
    /// Per-component observation precisions `τ_k`. When `Some`, the vector
    /// length must equal `n_components` and every entry must be positive.
    /// When `None`, `observation_precision` is replicated K times.
    pub component_precisions: Option<Vec<f64>>,
    /// Maximum number of VBEM iterations.
    pub max_iterations: usize,
    /// ELBO absolute-change convergence tolerance.
    pub tolerance: f64,
    /// Maximum permissible ELBO decrease (used for divergence detection).
    pub divergence_tolerance: f64,
    /// Seed for the seeded RNG used during mean initialisation.
    pub seed: u64,
}

impl VgmmConfig {
    /// Construct a config for `n_components` mixture components with
    /// sensible defaults:
    ///
    /// | Parameter               | Default |
    /// |-------------------------|---------|
    /// | `prior_concentration`   | 1.0     |
    /// | `prior_mean`            | 0.0     |
    /// | `prior_precision`       | 1e-3    |
    /// | `observation_precision` | 1.0     |
    /// | `max_iterations`        | 200     |
    /// | `tolerance`             | 1e-6    |
    /// | `divergence_tolerance`  | 1e-4    |
    /// | `seed`                  | 0       |
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            prior_concentration: 1.0,
            prior_mean: 0.0,
            prior_precision: 1e-3,
            observation_precision: 1.0,
            component_precisions: None,
            max_iterations: 200,
            tolerance: 1e-6,
            divergence_tolerance: 1e-4,
            seed: 0,
        }
    }

    /// Set the Gaussian prior hyperparameters for all component means.
    ///
    /// - `prior_mean` — prior mean m₀
    /// - `prior_precision` — prior precision β₀ (must be > 0)
    /// - `prior_concentration` — symmetric Dirichlet concentration α₀ (must
    ///   be > 0)
    pub fn with_prior(
        mut self,
        prior_mean: f64,
        prior_precision: f64,
        prior_concentration: f64,
    ) -> Self {
        self.prior_mean = prior_mean;
        self.prior_precision = prior_precision;
        self.prior_concentration = prior_concentration;
        self
    }

    /// Set the shared observation precision τ applied to all components.
    pub fn with_observation_precision(mut self, tau: f64) -> Self {
        self.observation_precision = tau;
        self
    }

    /// Set per-component observation precisions `τ_k`. The vector length must
    /// equal `n_components` and every entry must be strictly positive; this is
    /// validated immediately.
    pub fn with_component_precisions(mut self, taus: Vec<f64>) -> Result<Self> {
        if taus.len() != self.n_components {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.n_components],
                got: vec![taus.len()],
            });
        }
        for (k, &t) in taus.iter().enumerate() {
            if !t.is_finite() || t <= 0.0 {
                return Err(PgmError::InvalidDistribution(format!(
                    "component_precisions[{}] = {} must be positive and finite",
                    k, t
                )));
            }
        }
        self.component_precisions = Some(taus);
        Ok(self)
    }

    /// Set iteration budget and ELBO tolerance.
    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    /// Set the seed for the seeded RNG used during initialisation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate all config fields. Called from
    /// [`VariationalGaussianMixture::new`].
    fn validate(&self) -> Result<()> {
        if self.n_components < 1 {
            return Err(PgmError::InvalidDistribution(
                "VgmmConfig: n_components must be >= 1".to_string(),
            ));
        }
        if !self.prior_concentration.is_finite() || self.prior_concentration <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "VgmmConfig: prior_concentration = {} must be positive and finite",
                self.prior_concentration
            )));
        }
        if !self.prior_mean.is_finite() {
            return Err(PgmError::InvalidDistribution(format!(
                "VgmmConfig: prior_mean = {} must be finite",
                self.prior_mean
            )));
        }
        if !self.prior_precision.is_finite() || self.prior_precision <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "VgmmConfig: prior_precision = {} must be positive and finite",
                self.prior_precision
            )));
        }
        if !self.observation_precision.is_finite() || self.observation_precision <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "VgmmConfig: observation_precision = {} must be positive and finite",
                self.observation_precision
            )));
        }
        if let Some(ref taus) = self.component_precisions {
            if taus.len() != self.n_components {
                return Err(PgmError::DimensionMismatch {
                    expected: vec![self.n_components],
                    got: vec![taus.len()],
                });
            }
            for (k, &t) in taus.iter().enumerate() {
                if !t.is_finite() || t <= 0.0 {
                    return Err(PgmError::InvalidDistribution(format!(
                        "VgmmConfig: component_precisions[{}] = {} must be positive and finite",
                        k, t
                    )));
                }
            }
        }
        Ok(())
    }

    /// Return a `K`-vector of per-component observation precisions.
    fn taus(&self) -> Vec<f64> {
        match &self.component_precisions {
            Some(v) => v.clone(),
            None => vec![self.observation_precision; self.n_components],
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of a completed [`VariationalGaussianMixture::fit`] call.
#[derive(Clone, Debug)]
pub struct VgmmResult {
    /// Soft assignment matrix of shape `N x K`.
    ///
    /// `responsibilities[n][k]` is `r_{nk} = q(z_n = k)`, the posterior
    /// probability that data point `n` belongs to component `k`.
    pub responsibilities: Vec<Vec<f64>>,
    /// Posterior Gaussian distributions for the K component means.
    pub components: Vec<GaussianNP>,
    /// Posterior Dirichlet distribution over mixing weights.
    pub weights: DirichletNP,
    /// ELBO evaluated at initialisation, then after each complete iteration.
    pub elbo_history: Vec<f64>,
    /// Number of VBEM iterations executed (not counting the initialisation pass).
    pub iterations: usize,
    /// `true` if the ELBO converged within `tolerance` before exhausting the
    /// iteration budget.
    pub converged: bool,
}

impl VgmmResult {
    /// Normalised mixing weights `alpha_k / sum(alpha)` derived from the Dirichlet
    /// posterior concentrations.
    pub fn mixing_weights(&self) -> Vec<f64> {
        let total = self.weights.total_concentration();
        self.weights
            .concentration
            .iter()
            .map(|&a| a / total)
            .collect()
    }

    /// Hard cluster assignments `argmax_k r_{nk}` for each data point.
    pub fn hard_assignments(&self) -> Vec<usize> {
        self.responsibilities
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, _)| k)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Effective component sizes `N_k = sum_n r_{nk}`.
    pub fn component_counts(&self) -> Vec<f64> {
        let k = self.components.len();
        let mut counts = vec![0.0_f64; k];
        for row in &self.responsibilities {
            for (j, &r) in row.iter().enumerate() {
                counts[j] += r;
            }
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// VariationalGaussianMixture
// ---------------------------------------------------------------------------

/// Variational Bayes Gaussian Mixture Model fitter (VBEM / VBGMM).
///
/// The algorithm performs coordinate-ascent variational inference on the
/// mean-field factorisation
///
/// ```text
///   q(pi, mu, z) = q(pi) * prod_k q(mu_k) * prod_n q(z_n)
/// ```
///
/// updating the Dirichlet posterior `q(pi)`, Gaussian posteriors `q(mu_k)`, and
/// categorical posteriors `q(z_n)` until the ELBO converges.
///
/// # Example
///
/// ```no_run
/// use tensorlogic_quantrs_hooks::vmp::{VariationalGaussianMixture, VgmmConfig};
///
/// let config = VgmmConfig::new(2)
///     .with_prior(0.0, 1e-3, 1.0)
///     .with_observation_precision(1.0)
///     .with_limits(100, 1e-6)
///     .with_seed(42);
///
/// let vgmm = VariationalGaussianMixture::new(config).unwrap();
/// let data = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
/// let result = vgmm.fit(&data).unwrap();
/// assert!(result.converged);
/// ```
#[derive(Clone, Debug)]
pub struct VariationalGaussianMixture {
    config: VgmmConfig,
}

impl VariationalGaussianMixture {
    /// Construct a new fitter, validating the config immediately.
    pub fn new(config: VgmmConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Run VBEM on `data` and return the variational posterior.
    ///
    /// # Errors
    ///
    /// - [`PgmError::InvalidDistribution`] if `data` is empty or contains
    ///   non-finite values.
    /// - [`PgmError::ConvergenceFailure`] if the ELBO decreases by more than
    ///   `config.divergence_tolerance` in a single iteration (numerical
    ///   breakdown).
    pub fn fit(&self, data: &[f64]) -> Result<VgmmResult> {
        // ----------------------------------------------------------------
        // Input validation
        // ----------------------------------------------------------------
        if data.is_empty() {
            return Err(PgmError::InvalidDistribution(
                "VariationalGaussianMixture::fit: data must not be empty".to_string(),
            ));
        }
        for &x in data {
            if !x.is_finite() {
                return Err(PgmError::InvalidDistribution(format!(
                    "VariationalGaussianMixture::fit: non-finite data value {}",
                    x
                )));
            }
        }

        let k = self.config.n_components;
        let taus = self.config.taus();
        let m0 = self.config.prior_mean;
        let beta0 = self.config.prior_precision;
        let alpha0 = self.config.prior_concentration;

        // ----------------------------------------------------------------
        // Priors (fixed throughout)
        // ----------------------------------------------------------------
        let weights_prior = DirichletNP::new(vec![alpha0; k])?;
        let comp_prior = GaussianNP::new(m0, beta0)?;

        // ----------------------------------------------------------------
        // Initialisation
        // ----------------------------------------------------------------
        let init_m = init_means(data, k, self.config.seed);

        // Initial component posteriors: prior precision, initial mean from data
        let mut components: Vec<GaussianNP> = init_m
            .iter()
            .map(|&m| GaussianNP::new(m, beta0))
            .collect::<Result<Vec<_>>>()?;

        // Initial weight posterior: copy of prior
        let mut weights = weights_prior.clone();

        // Compute initial responsibilities via E-step
        let (mut resp, lse_sum_init) = e_step(data, &components, &weights, &taus);

        // Compute initial ELBO before any M-step
        let elbo0 = assemble_elbo(
            lse_sum_init,
            &weights,
            &weights_prior,
            &components,
            &comp_prior,
        )?;
        let mut elbo_history = vec![elbo0];

        // ----------------------------------------------------------------
        // Main VBEM loop (mirrors engine.rs::run() structure exactly)
        // ----------------------------------------------------------------
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            // M-step: update posteriors from responsibilities
            let (new_components, new_weights) = m_step(data, &resp, &self.config, &taus)?;

            // E-step: update responsibilities from posteriors
            let (new_resp, lse_sum) = e_step(data, &new_components, &new_weights, &taus);

            // ELBO for this iteration
            let elbo_new = assemble_elbo(
                lse_sum,
                &new_weights,
                &weights_prior,
                &new_components,
                &comp_prior,
            )?;

            let prev = *elbo_history
                .last()
                .ok_or_else(|| PgmError::ConvergenceFailure("VBEM elbo history is empty".into()))?;

            // Divergence check: ELBO is guaranteed non-decreasing for conjugate
            // VBEM; a drop beyond tolerance indicates numerical breakdown.
            if elbo_new < prev - self.config.divergence_tolerance {
                return Err(PgmError::ConvergenceFailure(format!(
                    "VBEM ELBO decreased from {} to {} at iteration {}",
                    prev, elbo_new, iter
                )));
            }

            // Accept the update
            resp = new_resp;
            components = new_components;
            weights = new_weights;
            elbo_history.push(elbo_new);
            iterations = iter + 1;

            // Convergence check
            if (elbo_new - prev).abs() < self.config.tolerance {
                converged = true;
                break;
            }
        }

        Ok(VgmmResult {
            responsibilities: resp,
            components,
            weights,
            elbo_history,
            iterations,
            converged,
        })
    }
}

// ---------------------------------------------------------------------------
// Private algorithmic helpers
// ---------------------------------------------------------------------------

/// Initialise K component means by drawing K distinct indices (with cycling
/// fallback when `k > data.len()`) from `data` using a seeded RNG.
fn init_means(data: &[f64], k: usize, seed: u64) -> Vec<f64> {
    if k == 0 {
        return Vec::new();
    }
    let n = data.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut result = Vec::with_capacity(k);
    // Build a shuffled pool of indices; cycle when k > n.
    let mut pool: Vec<usize> = (0..n).collect();
    let mut pool_pos = 0;

    // Fisher-Yates full shuffle of the initial pool
    for i in 0..n {
        let j = i + (rng.random::<f64>() * (n - i) as f64) as usize % (n - i).max(1);
        pool.swap(i, j);
    }

    while result.len() < k {
        if pool_pos >= pool.len() {
            // Reshuffle and cycle
            pool_pos = 0;
            for i in 0..n {
                let j = i + (rng.random::<f64>() * (n - i) as f64) as usize % (n - i).max(1);
                pool.swap(i, j);
            }
        }
        result.push(data[pool[pool_pos]]);
        pool_pos += 1;
    }
    result
}

/// Numerically stable log-sum-exp.
fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// VBEM E-step: compute normalised responsibilities `r_{nk}` for all N data
/// points and all K components.
///
/// Returns `(responsibilities, sum_of_lse_over_n)` where the second element
/// contributes to the ELBO as described in Bishop (2006) eq. (10.65).
///
/// The unnormalised log-responsibility for point n and component k is:
///
/// ```text
///   ln rho_{nk} = E[ln pi_k]
///                 - 0.5 * tau_k * ( m_k^2 + 1/beta_k - 2 x_n m_k + x_n^2 )
/// ```
///
/// where `E[ln pi_k] = psi(alpha_k) - psi(sum alpha)` from
/// `DirichletNP::expected_sufficient_statistics`.
fn e_step(
    data: &[f64],
    components: &[GaussianNP],
    weights: &DirichletNP,
    taus: &[f64],
) -> (Vec<Vec<f64>>, f64) {
    let k = components.len();
    // E[ln pi_k] for each k — DirichletNP::expected_sufficient_statistics
    // returns (psi(alpha_k) - psi(alpha_0)) for each k.
    let e_ln_pi: Vec<f64> = weights.expected_sufficient_statistics();

    let mut lse_sum = 0.0;
    let responsibilities: Vec<Vec<f64>> = data
        .iter()
        .map(|&x| {
            // Compute unnormalised log-responsibilities
            let ln_rho: Vec<f64> = (0..k)
                .map(|j| {
                    let comp = &components[j];
                    let tau_k = taus[j];
                    // E[(x - mu_k)^2] under q(mu_k) = N(m_k, 1/beta_k):
                    //   = (x - m_k)^2 + 1/beta_k
                    //   = x^2 - 2 x m_k + m_k^2 + 1/beta_k
                    let quad =
                        comp.mean * comp.mean + 1.0 / comp.precision - 2.0 * x * comp.mean + x * x;
                    e_ln_pi[j] - 0.5 * tau_k * quad
                })
                .collect();

            let lse = log_sum_exp(&ln_rho);
            lse_sum += lse;

            // Normalised responsibilities
            ln_rho.iter().map(|&l| (l - lse).exp()).collect()
        })
        .collect();

    (responsibilities, lse_sum)
}

/// VBEM M-step: update posteriors `q(pi)` and `q(mu_k)` from the current
/// responsibilities.
///
/// For each component k the sufficient statistics accumulated from the data are:
///
/// ```text
///   N_k = sum_n r_{nk}
///   S_k = sum_n r_{nk} * x_n
/// ```
///
/// The conjugate posterior updates are (Bishop 2006, eq. 10.58-10.63):
///
/// ```text
///   alpha_k = alpha_0 + N_k
///   beta_k  = beta_0 + tau_k * N_k
///   m_k     = (beta_0 * m_0 + tau_k * S_k) / beta_k
/// ```
///
/// Empty components (`N_k < 1e-10`) fall back to the prior to avoid
/// degenerate precisions.
fn m_step(
    data: &[f64],
    resp: &[Vec<f64>],
    config: &VgmmConfig,
    taus: &[f64],
) -> Result<(Vec<GaussianNP>, DirichletNP)> {
    let k = config.n_components;
    let m0 = config.prior_mean;
    let beta0 = config.prior_precision;
    let alpha0 = config.prior_concentration;

    let mut alphas = vec![0.0_f64; k];
    let mut new_components = Vec::with_capacity(k);

    for j in 0..k {
        // Accumulate N_k and S_k
        let n_k: f64 = resp.iter().map(|row| row[j]).sum();
        let s_k: f64 = resp
            .iter()
            .zip(data.iter())
            .map(|(row, &x)| row[j] * x)
            .sum();

        alphas[j] = alpha0 + n_k;

        let comp = if n_k < 1e-10 {
            // Empty component: revert to prior
            GaussianNP::new(m0, beta0)?
        } else {
            let tau_k = taus[j];
            let beta_k = beta0 + tau_k * n_k;
            let m_k = (beta0 * m0 + tau_k * s_k) / beta_k;
            GaussianNP::new(m_k, beta_k)?
        };

        new_components.push(comp);
    }

    let new_weights = DirichletNP::new(alphas)?;
    Ok((new_components, new_weights))
}

/// Compute the VBEM evidence lower bound (ELBO).
///
/// The ELBO decomposes as:
///
/// ```text
///   ELBO = sum_n LSE_n  -  KL[q(pi) || p(pi)]  -  sum_k KL[q(mu_k) || p(mu_k)]
/// ```
///
/// where `sum_n LSE_n` is the accumulated log-normaliser from the E-step,
/// capturing `E[ln p(X, Z | pi, mu)] - E[ln q(Z)]` by the VBEM identity
/// (Bishop 2006, eq. 10.70).
///
/// The KL divergences are closed form for conjugate families:
/// - `KL[Dir(alpha) || Dir(beta)]` from [`super::distributions::dirichlet_kl`]
/// - `KL[N(m_q, 1/beta_q) || N(m_p, 1/beta_p)]` from [`super::distributions::gaussian_kl`]
fn assemble_elbo(
    lse_sum: f64,
    weights: &DirichletNP,
    weights_prior: &DirichletNP,
    components: &[GaussianNP],
    comp_prior: &GaussianNP,
) -> Result<f64> {
    let kl_weights = dirichlet_kl(weights, weights_prior)?;
    let kl_comps: f64 = components
        .iter()
        .map(|c| gaussian_kl(c, comp_prior))
        .collect::<Result<Vec<f64>>>()?
        .iter()
        .sum();
    Ok(lse_sum - kl_weights - kl_comps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_data() -> Vec<f64> {
        vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2]
    }

    fn default_two_cluster_vgmm() -> VariationalGaussianMixture {
        let config = VgmmConfig::new(2)
            .with_prior(0.0, 1e-3, 1.0)
            .with_observation_precision(1.0)
            .with_limits(200, 1e-6)
            .with_seed(1);
        VariationalGaussianMixture::new(config).expect("config valid")
    }

    #[test]
    fn responsibilities_sum_to_one() {
        let data = two_cluster_data();
        let vgmm = default_two_cluster_vgmm();
        let result = vgmm.fit(&data).expect("fit");
        for (n, row) in result.responsibilities.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "row {} sums to {} (expected 1.0)",
                n,
                sum
            );
        }
    }

    #[test]
    fn elbo_is_monotone() {
        let data = two_cluster_data();
        let vgmm = default_two_cluster_vgmm();
        let result = vgmm.fit(&data).expect("fit");
        for w in result.elbo_history.windows(2) {
            assert!(w[1] + 1e-7 >= w[0], "ELBO decreased: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn two_cluster_recovery() {
        let data = two_cluster_data();
        let vgmm = default_two_cluster_vgmm();
        let result = vgmm.fit(&data).expect("fit");
        let mut means: Vec<f64> = result.components.iter().map(|c| c.mean).collect();
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let truth = [0.1_f64, 10.1];
        for (recovered, &t) in means.iter().zip(truth.iter()) {
            assert!(
                (recovered - t).abs() < 0.5,
                "recovered mean {} too far from truth {}",
                recovered,
                t
            );
        }
    }

    #[test]
    fn single_component_posterior() {
        // K=1, data=[1,2,3], essentially uninformative prior (beta_0=1e-6, tau=1).
        // After convergence: N_1 = 3, S_1 = 6
        //   beta_1 = 1e-6 + 1.0 * 3.0 ~= 3.0
        //   m_1 = (1e-6 * 0 + 1.0 * 6.0) / 3.0 ~= 2.0
        let config = VgmmConfig::new(1)
            .with_prior(0.0, 1e-6, 1.0)
            .with_observation_precision(1.0)
            .with_limits(200, 1e-9)
            .with_seed(0);
        let vgmm = VariationalGaussianMixture::new(config).expect("config valid");
        let result = vgmm.fit(&[1.0, 2.0, 3.0]).expect("fit");
        let m1 = result.components[0].mean;
        assert!(
            (m1 - 2.0).abs() < 0.01,
            "single-component posterior mean = {} (expected ~2.0)",
            m1
        );
    }

    #[test]
    fn empty_data_errors() {
        let vgmm = default_two_cluster_vgmm();
        let err = vgmm.fit(&[]);
        assert!(err.is_err(), "empty data should error");
    }

    #[test]
    fn nan_data_errors() {
        let vgmm = default_two_cluster_vgmm();
        let err = vgmm.fit(&[1.0, f64::NAN]);
        assert!(err.is_err(), "NaN data should error");
    }

    #[test]
    fn zero_components_errors() {
        let config = VgmmConfig::new(0);
        let err = VariationalGaussianMixture::new(config);
        assert!(err.is_err(), "K=0 should be rejected by validate()");
    }

    #[test]
    fn mismatched_component_precisions() {
        // n_components = 3 but providing only 1 precision
        let result = VgmmConfig::new(3).with_component_precisions(vec![1.0]);
        assert!(
            result.is_err(),
            "mismatched component_precisions should error"
        );
    }

    #[test]
    fn mixing_weights_sum() {
        let data = two_cluster_data();
        let vgmm = default_two_cluster_vgmm();
        let result = vgmm.fit(&data).expect("fit");
        let sum: f64 = result.mixing_weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "mixing weights sum = {} (expected 1.0)",
            sum
        );
    }

    #[test]
    fn hard_assignments_range() {
        let data = two_cluster_data();
        let config = VgmmConfig::new(2)
            .with_prior(0.0, 1e-3, 1.0)
            .with_observation_precision(1.0)
            .with_limits(200, 1e-6)
            .with_seed(0);
        let vgmm = VariationalGaussianMixture::new(config).expect("config valid");
        let result = vgmm.fit(&data).expect("fit");
        let k = result.components.len();
        for &a in &result.hard_assignments() {
            assert!(a < k, "hard assignment {} out of range [0, {})", a, k);
        }
    }
}
