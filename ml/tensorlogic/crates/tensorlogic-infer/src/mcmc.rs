//! Markov Chain Monte Carlo (MCMC) sampling algorithms.
//!
//! Provides general-purpose posterior sampling via:
//! - **Metropolis-Hastings**: Classic accept/reject sampler with pluggable proposals
//! - **Hamiltonian Monte Carlo (HMC)**: Gradient-based sampler with leapfrog integration
//! - **Chain diagnostics**: ESS (batch means), Gelman-Rubin R-hat, autocorrelation
//!
//! This module is intentionally distinct from the graphical-model-specific Gibbs sampler
//! found in `tensorlogic-quantrs-hooks`.

// ─── Traits ─────────────────────────────────────────────────────────────────

/// A log-probability function (unnormalized): given a parameter vector `θ`, returns log p(θ).
///
/// Implementations must be `Send + Sync` so they can be used across threads.
pub trait LogProb: Send + Sync {
    fn log_prob(&self, theta: &[f64]) -> f64;
}

/// Convenience wrapper: adapts a closure `F: Fn(&[f64]) -> f64` into a [`LogProb`].
pub struct LogProbFn<F: Fn(&[f64]) -> f64 + Send + Sync> {
    f: F,
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> LogProbFn<F> {
    /// Create a new [`LogProbFn`] wrapping the given closure.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F: Fn(&[f64]) -> f64 + Send + Sync> LogProb for LogProbFn<F> {
    fn log_prob(&self, theta: &[f64]) -> f64 {
        (self.f)(theta)
    }
}

/// Proposal distribution for Metropolis-Hastings sampling.
pub trait Proposal: Send + Sync {
    /// Sample a proposed next state given the current state.
    fn propose(&self, current: &[f64], rng: &mut McmcRng) -> Vec<f64>;

    /// Log proposal ratio: `log q(x|y) − log q(y|x)`.
    ///
    /// Returns `0.0` for symmetric proposals (e.g. Gaussian random walk).
    fn log_ratio(&self, proposed: &[f64], current: &[f64]) -> f64;
}

// ─── Proposals ───────────────────────────────────────────────────────────────

/// Gaussian random-walk proposal: `θ' = θ + N(0, step_size²)`.
///
/// This proposal is symmetric so [`Proposal::log_ratio`] always returns `0.0`.
#[derive(Debug, Clone)]
pub struct GaussianProposal {
    pub step_size: f64,
}

impl GaussianProposal {
    /// Create a new Gaussian random-walk proposal with the given step size.
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl Proposal for GaussianProposal {
    fn propose(&self, current: &[f64], rng: &mut McmcRng) -> Vec<f64> {
        current
            .iter()
            .map(|&x| x + rng.next_normal_scaled(0.0, self.step_size))
            .collect()
    }

    fn log_ratio(&self, _proposed: &[f64], _current: &[f64]) -> f64 {
        0.0
    }
}

/// Independent Gaussian proposal: draws each dimension independently from `N(mean_i, std_i²)`.
///
/// Unlike the random-walk proposal this proposal ignores the current state, so the
/// log-ratio is generally non-zero and must be computed explicitly.
#[derive(Debug, Clone)]
pub struct IndependentGaussianProposal {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl IndependentGaussianProposal {
    /// Create a new independent Gaussian proposal.
    ///
    /// # Panics (debug)
    /// Panics in debug builds if `mean.len() != std.len()`.
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Self {
        debug_assert_eq!(
            mean.len(),
            std.len(),
            "mean and std must have the same length"
        );
        Self { mean, std }
    }
}

/// Evaluate `log N(x; mu, sigma²)` (up to constant).
#[inline]
fn log_normal_density(x: f64, mu: f64, sigma: f64) -> f64 {
    let diff = x - mu;
    -0.5 * (diff / sigma).powi(2) - sigma.ln()
}

impl Proposal for IndependentGaussianProposal {
    fn propose(&self, _current: &[f64], rng: &mut McmcRng) -> Vec<f64> {
        self.mean
            .iter()
            .zip(self.std.iter())
            .map(|(&mu, &sigma)| rng.next_normal_scaled(mu, sigma))
            .collect()
    }

    fn log_ratio(&self, proposed: &[f64], current: &[f64]) -> f64 {
        // log q(current | proposed) - log q(proposed | current)
        // Both are independent, so:
        //   log q(x | y) = sum_i log N(x_i; mean_i, std_i)   (ignores y)
        //   same for log q(y | x)
        // Therefore: log q(current) - log q(proposed)
        let log_q_current: f64 = current
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((&x, &mu), &sigma)| log_normal_density(x, mu, sigma))
            .sum();
        let log_q_proposed: f64 = proposed
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((&x, &mu), &sigma)| log_normal_density(x, mu, sigma))
            .sum();
        log_q_current - log_q_proposed
    }
}

// ─── RNG ─────────────────────────────────────────────────────────────────────

/// A simple, reproducible LCG-based pseudo-random number generator.
///
/// Uses the Knuth multiplier LCG with a 64-bit state. Sufficient for MCMC
/// applications where only statistical quality (not cryptographic security)
/// is required.
#[derive(Debug, Clone)]
pub struct McmcRng {
    state: u64,
}

impl McmcRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        // Mix the seed to avoid poor low-bit initializations.
        let state = seed.wrapping_add(6364136223846793005);
        Self { state }
    }

    /// Advance the LCG and return the raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters from Knuth / MMIX
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Return a uniform sample in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for IEEE 754 double precision mantissa.
        (self.next_u64() >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Return a standard normal sample using the Box-Muller transform.
    ///
    /// Samples are generated in pairs; only one is returned per call.
    pub fn next_normal(&mut self) -> f64 {
        // Box-Muller: requires two uniform samples
        let u1 = self.next_f64().max(f64::MIN_POSITIVE); // avoid log(0)
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        r * theta.cos()
    }

    /// Return a normal sample with the given mean and standard deviation.
    pub fn next_normal_scaled(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.next_normal()
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration shared by all MCMC samplers.
#[derive(Debug, Clone)]
pub struct McmcConfig {
    /// Number of post-warmup samples to collect (default: 1000).
    pub n_samples: usize,
    /// Number of burn-in steps to discard (default: 500).
    pub n_warmup: usize,
    /// Thinning factor: keep every `thin`-th sample (default: 1).
    pub thin: usize,
    /// RNG seed for reproducibility (default: 42).
    pub seed: u64,
    /// Target acceptance rate for adaptive step-size tuning (default: 0.234 for MH).
    pub target_acceptance: f64,
}

impl Default for McmcConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            n_warmup: 500,
            thin: 1,
            seed: 42,
            target_acceptance: 0.234,
        }
    }
}

impl McmcConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of post-warmup samples.
    pub fn n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    /// Set the number of burn-in (warmup) steps.
    pub fn n_warmup(mut self, n: usize) -> Self {
        self.n_warmup = n;
        self
    }

    /// Set the thinning factor.
    pub fn thin(mut self, t: usize) -> Self {
        self.thin = t;
        self
    }

    /// Set the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

// ─── Results & Diagnostics ────────────────────────────────────────────────────

/// Per-chain diagnostics computed from the collected samples.
#[derive(Debug, Clone)]
pub struct ChainDiagnostics {
    /// Total number of collected samples.
    pub n_samples: usize,
    /// Fraction of proposals that were accepted.
    pub acceptance_rate: f64,
    /// Per-dimension posterior mean.
    pub mean: Vec<f64>,
    /// Per-dimension posterior variance.
    pub variance: Vec<f64>,
    /// Effective sample size per dimension (batch-means estimator).
    pub effective_sample_size: Vec<f64>,
    /// Gelman-Rubin R-hat per dimension (requires multiple chains; `None` for a single chain).
    pub r_hat: Option<Vec<f64>>,
}

/// Complete result returned by an MCMC sampler.
#[derive(Debug, Clone)]
pub struct McmcResult {
    /// Collected samples: outer index is sample index, inner is parameter dimension.
    pub samples: Vec<Vec<f64>>,
    /// Log-probability at each collected sample.
    pub log_probs: Vec<f64>,
    /// Chain-level diagnostics.
    pub diagnostics: ChainDiagnostics,
}

impl McmcResult {
    /// Number of samples collected.
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Number of parameter dimensions.
    pub fn n_dims(&self) -> usize {
        self.samples.first().map(|s| s.len()).unwrap_or(0)
    }

    /// Extract all samples for a single dimension.
    pub fn marginal_samples(&self, dim: usize) -> Vec<f64> {
        self.samples.iter().map(|s| s[dim]).collect()
    }

    /// Compute the posterior mean across all dimensions.
    pub fn posterior_mean(&self) -> Vec<f64> {
        let n = self.n_samples();
        if n == 0 {
            return vec![];
        }
        let d = self.n_dims();
        let mut mean = vec![0.0_f64; d];
        for sample in &self.samples {
            for (m, &v) in mean.iter_mut().zip(sample.iter()) {
                *m += v;
            }
        }
        mean.iter_mut().for_each(|m| *m /= n as f64);
        mean
    }

    /// Compute the posterior variance (unbiased) across all dimensions.
    pub fn posterior_variance(&self) -> Vec<f64> {
        let n = self.n_samples();
        if n < 2 {
            return vec![0.0; self.n_dims()];
        }
        let mean = self.posterior_mean();
        let d = self.n_dims();
        let mut var = vec![0.0_f64; d];
        for sample in &self.samples {
            for (v, (&x, &mu)) in var.iter_mut().zip(sample.iter().zip(mean.iter())) {
                *v += (x - mu).powi(2);
            }
        }
        var.iter_mut().for_each(|v| *v /= (n - 1) as f64);
        var
    }

    /// Compute the `(alpha/2, 1 - alpha/2)` credible interval for a single dimension.
    ///
    /// Returns `(lower, upper)` quantiles. `alpha = 0.05` gives a 95 % interval.
    pub fn credible_interval(&self, dim: usize, alpha: f64) -> (f64, f64) {
        let mut marginal = self.marginal_samples(dim);
        marginal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = marginal.len();
        if n == 0 {
            return (f64::NAN, f64::NAN);
        }
        let lo_idx = ((alpha / 2.0) * n as f64) as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * n as f64) as usize;
        let lo = marginal[lo_idx.min(n - 1)];
        let hi = marginal[hi_idx.min(n - 1)];
        (lo, hi)
    }
}

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can arise during MCMC sampling.
#[derive(Debug)]
pub enum McmcError {
    /// The sampler configuration is invalid (e.g. zero samples requested).
    InvalidConfig(String),
    /// A dimension mismatch was detected between the initial state and the model.
    DimensionMismatch,
    /// A numerical problem was encountered (e.g. NaN in log-probability).
    NumericalError(String),
}

impl std::fmt::Display for McmcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McmcError::InvalidConfig(msg) => write!(f, "MCMC invalid configuration: {}", msg),
            McmcError::DimensionMismatch => {
                write!(f, "MCMC dimension mismatch between initial state and model")
            }
            McmcError::NumericalError(msg) => write!(f, "MCMC numerical error: {}", msg),
        }
    }
}

impl std::error::Error for McmcError {}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate a configuration and return an error if it is unusable.
fn validate_config(config: &McmcConfig) -> Result<(), McmcError> {
    if config.n_samples == 0 {
        return Err(McmcError::InvalidConfig(
            "n_samples must be > 0".to_string(),
        ));
    }
    if config.thin == 0 {
        return Err(McmcError::InvalidConfig("thin must be > 0".to_string()));
    }
    Ok(())
}

/// Compute basic statistics (mean, variance) over a slice.
fn slice_stats(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = if n < 2 {
        0.0
    } else {
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    };
    (mean, var)
}

// ─── Metropolis-Hastings ──────────────────────────────────────────────────────

/// Metropolis-Hastings MCMC sampler.
///
/// Runs a single chain with an arbitrary [`Proposal`] distribution. The chain
/// is initialized at `initial`, runs for `n_warmup + n_samples * thin` steps,
/// discards the warmup, and returns every `thin`-th sample.
pub struct MetropolisHastings<P: LogProb, Q: Proposal> {
    log_prob: P,
    proposal: Q,
    config: McmcConfig,
}

impl<P: LogProb, Q: Proposal> MetropolisHastings<P, Q> {
    /// Create a new Metropolis-Hastings sampler.
    pub fn new(log_prob: P, proposal: Q, config: McmcConfig) -> Self {
        Self {
            log_prob,
            proposal,
            config,
        }
    }

    /// Run the Metropolis-Hastings chain from `initial` and return the collected samples.
    pub fn sample(&self, initial: &[f64]) -> Result<McmcResult, McmcError> {
        validate_config(&self.config)?;
        if initial.is_empty() {
            return Err(McmcError::InvalidConfig(
                "initial state must be non-empty".to_string(),
            ));
        }

        let mut rng = McmcRng::new(self.config.seed);
        let total_steps = self.config.n_warmup + self.config.n_samples * self.config.thin;

        let mut current: Vec<f64> = initial.to_vec();
        let mut current_lp = self.log_prob.log_prob(&current);
        if !current_lp.is_finite() {
            return Err(McmcError::NumericalError(
                "initial state has non-finite log probability".to_string(),
            ));
        }

        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_samples);
        let mut log_probs: Vec<f64> = Vec::with_capacity(self.config.n_samples);
        let mut n_accepted: usize = 0;
        let mut step_in_sample: usize = 0; // counts post-warmup steps for thinning

        for step in 0..total_steps {
            let proposed = self.proposal.propose(&current, &mut rng);
            let proposed_lp = self.log_prob.log_prob(&proposed);

            let log_accept = if proposed_lp.is_finite() {
                let log_alpha =
                    proposed_lp - current_lp + self.proposal.log_ratio(&proposed, &current);
                log_alpha.min(0.0)
            } else {
                f64::NEG_INFINITY
            };

            let u = rng.next_f64();
            let accepted = u.ln() < log_accept;

            if accepted {
                current = proposed;
                current_lp = proposed_lp;
                if step >= self.config.n_warmup {
                    n_accepted += 1;
                }
            }

            // Collect post-warmup samples, applying thinning
            if step >= self.config.n_warmup {
                step_in_sample += 1;
                if step_in_sample == self.config.thin {
                    samples.push(current.clone());
                    log_probs.push(current_lp);
                    step_in_sample = 0;
                }
            }
        }

        let n_post_warmup_steps = self.config.n_samples * self.config.thin;
        let acceptance_rate = if n_post_warmup_steps > 0 {
            n_accepted as f64 / n_post_warmup_steps as f64
        } else {
            0.0
        };

        let diagnostics = compute_diagnostics_with_acceptance(&samples, acceptance_rate);
        Ok(McmcResult {
            samples,
            log_probs,
            diagnostics,
        })
    }
}

// ─── Hamiltonian Monte Carlo ──────────────────────────────────────────────────

/// Hamiltonian Monte Carlo (HMC) sampler with leapfrog integration.
///
/// Gradients are estimated via central finite differences, so no analytic
/// gradient implementation is required from the user.
pub struct HamiltonianMonteCarlo<P: LogProb> {
    log_prob: P,
    step_size: f64,
    n_leapfrog_steps: usize,
    config: McmcConfig,
}

impl<P: LogProb> HamiltonianMonteCarlo<P> {
    /// Create a new HMC sampler.
    ///
    /// * `step_size`: leapfrog step size `ε`.
    /// * `n_leapfrog_steps`: number of leapfrog steps `L` per proposal.
    pub fn new(log_prob: P, step_size: f64, n_leapfrog_steps: usize, config: McmcConfig) -> Self {
        Self {
            log_prob,
            step_size,
            n_leapfrog_steps,
            config,
        }
    }

    /// Run the HMC chain from `initial` and return collected samples.
    pub fn sample(&self, initial: &[f64]) -> Result<McmcResult, McmcError> {
        validate_config(&self.config)?;
        if initial.is_empty() {
            return Err(McmcError::InvalidConfig(
                "initial state must be non-empty".to_string(),
            ));
        }
        if self.step_size <= 0.0 {
            return Err(McmcError::InvalidConfig(
                "step_size must be positive".to_string(),
            ));
        }
        if self.n_leapfrog_steps == 0 {
            return Err(McmcError::InvalidConfig(
                "n_leapfrog_steps must be > 0".to_string(),
            ));
        }

        let mut rng = McmcRng::new(self.config.seed);
        let total_steps = self.config.n_warmup + self.config.n_samples * self.config.thin;
        let d = initial.len();

        let mut current: Vec<f64> = initial.to_vec();
        let mut current_lp = self.log_prob.log_prob(&current);
        if !current_lp.is_finite() {
            return Err(McmcError::NumericalError(
                "initial state has non-finite log probability".to_string(),
            ));
        }

        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_samples);
        let mut log_probs: Vec<f64> = Vec::with_capacity(self.config.n_samples);
        let mut n_accepted: usize = 0;
        let mut step_in_sample: usize = 0;

        for step in 0..total_steps {
            // Sample momentum r ~ N(0, I)
            let momentum: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();

            // Kinetic energy at start: 0.5 * r^T r
            let ke_old: f64 = momentum.iter().map(|&r| 0.5 * r * r).sum();

            // Leapfrog integration
            let (proposed, new_momentum) = self.leapfrog(&current, &momentum);

            let proposed_lp = self.log_prob.log_prob(&proposed);
            let ke_new: f64 = new_momentum.iter().map(|&r| 0.5 * r * r).sum();

            // Hamiltonian H = -log p(θ) + KE
            let h_old = -current_lp + ke_old;
            let h_new = -proposed_lp + ke_new;

            let log_accept = if proposed_lp.is_finite() {
                (h_old - h_new).min(0.0)
            } else {
                f64::NEG_INFINITY
            };

            let u = rng.next_f64();
            let accepted = u.ln() < log_accept;

            if accepted {
                current = proposed;
                current_lp = proposed_lp;
                if step >= self.config.n_warmup {
                    n_accepted += 1;
                }
            }

            if step >= self.config.n_warmup {
                step_in_sample += 1;
                if step_in_sample == self.config.thin {
                    samples.push(current.clone());
                    log_probs.push(current_lp);
                    step_in_sample = 0;
                }
            }
        }

        let n_post_warmup_steps = self.config.n_samples * self.config.thin;
        let acceptance_rate = if n_post_warmup_steps > 0 {
            n_accepted as f64 / n_post_warmup_steps as f64
        } else {
            0.0
        };

        let diagnostics = compute_diagnostics_with_acceptance(&samples, acceptance_rate);
        Ok(McmcResult {
            samples,
            log_probs,
            diagnostics,
        })
    }

    /// Estimate the gradient of `log_prob` at `theta` using central finite differences.
    ///
    /// Uses step size `eps` for the perturbation.
    fn grad_log_prob(&self, theta: &[f64], eps: f64) -> Vec<f64> {
        let d = theta.len();
        let mut grad = vec![0.0_f64; d];
        let mut theta_plus = theta.to_vec();
        let mut theta_minus = theta.to_vec();
        for i in 0..d {
            theta_plus[i] = theta[i] + eps;
            theta_minus[i] = theta[i] - eps;
            grad[i] = (self.log_prob.log_prob(&theta_plus) - self.log_prob.log_prob(&theta_minus))
                / (2.0 * eps);
            theta_plus[i] = theta[i];
            theta_minus[i] = theta[i];
        }
        grad
    }

    /// Leapfrog integrator: run `L` steps of size `ε` starting from `(theta, momentum)`.
    ///
    /// Returns `(theta*, momentum*)`.
    fn leapfrog(&self, theta: &[f64], momentum: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let eps = self.step_size;
        // Finite-difference step for gradient estimation. Choose adaptively.
        let fd_eps = 1e-5_f64;

        let mut q = theta.to_vec();
        let mut p = momentum.to_vec();
        let d = q.len();

        // Half-step for momentum at the start
        let grad = self.grad_log_prob(&q, fd_eps);
        for i in 0..d {
            p[i] += 0.5 * eps * grad[i];
        }

        for step in 0..self.n_leapfrog_steps {
            // Full step for position
            for i in 0..d {
                q[i] += eps * p[i];
            }

            // Full step for momentum (except at last step, where it is a half-step)
            if step < self.n_leapfrog_steps - 1 {
                let grad_q = self.grad_log_prob(&q, fd_eps);
                for i in 0..d {
                    p[i] += eps * grad_q[i];
                }
            }
        }

        // Final half-step for momentum
        let grad_final = self.grad_log_prob(&q, fd_eps);
        for i in 0..d {
            p[i] += 0.5 * eps * grad_final[i];
        }

        // Negate momentum to make the proposal reversible
        for pi in p.iter_mut() {
            *pi = -*pi;
        }

        (q, p)
    }
}

// ─── Diagnostics ─────────────────────────────────────────────────────────────

/// Compute the effective sample size (ESS) using the batch-means estimator.
///
/// Partitions the chain into `sqrt(n)` batches, computes the batch means,
/// and estimates the variance of the chain mean. Returns a value in `[1, n]`.
pub fn effective_sample_size(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return n as f64;
    }

    let b = (n as f64).sqrt() as usize; // batch size
    let n_batches = n / b;

    if n_batches < 2 {
        return n as f64;
    }

    let overall_mean = samples.iter().sum::<f64>() / n as f64;

    // Variance of the overall chain (naive)
    let chain_var = samples
        .iter()
        .map(|&x| (x - overall_mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;

    if chain_var == 0.0 {
        return 1.0;
    }

    // Variance of batch means
    let batch_mean_var: f64 = (0..n_batches)
        .map(|k| {
            let batch = &samples[k * b..(k + 1) * b];
            let bm = batch.iter().sum::<f64>() / b as f64;
            (bm - overall_mean).powi(2)
        })
        .sum::<f64>()
        / (n_batches - 1) as f64;

    // ESS = n * chain_var / (b * batch_mean_var)
    let ess = n as f64 * chain_var / (b as f64 * batch_mean_var);
    ess.clamp(1.0, n as f64)
}

/// Compute the Gelman-Rubin R-hat statistic for a set of independent chains.
///
/// Returns a value close to 1.0 for converged chains, and > 1.1 indicates
/// potential non-convergence. Requires at least 2 chains.
///
/// # Panics
/// Returns `f64::NAN` if `chains` is empty or all chains have zero variance.
pub fn gelman_rubin(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m < 2 {
        return f64::NAN;
    }

    // All chains should have the same length; use the minimum length.
    let n = chains.iter().map(|c| c.len()).min().unwrap_or(0);
    if n < 2 {
        return f64::NAN;
    }

    let chain_means: Vec<f64> = chains
        .iter()
        .map(|c| c[..n].iter().sum::<f64>() / n as f64)
        .collect();
    let overall_mean = chain_means.iter().sum::<f64>() / m as f64;

    // Between-chain variance B
    let b = n as f64
        * chain_means
            .iter()
            .map(|&mu| (mu - overall_mean).powi(2))
            .sum::<f64>()
        / (m - 1) as f64;

    // Within-chain variance W (average of per-chain variances)
    let w = chains
        .iter()
        .zip(chain_means.iter())
        .map(|(c, &mu)| c[..n].iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1) as f64)
        .sum::<f64>()
        / m as f64;

    if w == 0.0 {
        return f64::NAN;
    }

    // Pooled variance estimate
    let var_hat = (n - 1) as f64 / n as f64 * w + b / n as f64;
    (var_hat / w).sqrt()
}

/// Compute the autocorrelation of `samples` at a given lag `k`.
///
/// Returns a value in `[-1, 1]`; lag 0 always returns `1.0`.
pub fn autocorrelation(samples: &[f64], lag: usize) -> f64 {
    let n = samples.len();
    if n == 0 || lag >= n {
        return 0.0;
    }
    if lag == 0 {
        return 1.0;
    }

    let mean = samples.iter().sum::<f64>() / n as f64;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance == 0.0 {
        return 1.0;
    }

    let n_pairs = n - lag;
    let cov: f64 = samples[..n_pairs]
        .iter()
        .zip(samples[lag..].iter())
        .map(|(&a, &b)| (a - mean) * (b - mean))
        .sum::<f64>()
        / n_pairs as f64;

    cov / variance
}

/// Compute [`ChainDiagnostics`] from a collection of samples (one per row).
///
/// This version assumes a single chain (so `r_hat` will be `None`).
/// Acceptance rate is set to `0.0`; use `compute_diagnostics_with_acceptance`
/// when you have the true acceptance rate.
pub fn compute_diagnostics(samples: &[Vec<f64>]) -> ChainDiagnostics {
    compute_diagnostics_with_acceptance(samples, 0.0)
}

/// Internal helper: compute diagnostics with a known acceptance rate.
pub(crate) fn compute_diagnostics_with_acceptance(
    samples: &[Vec<f64>],
    acceptance_rate: f64,
) -> ChainDiagnostics {
    let n = samples.len();
    if n == 0 {
        return ChainDiagnostics {
            n_samples: 0,
            acceptance_rate,
            mean: vec![],
            variance: vec![],
            effective_sample_size: vec![],
            r_hat: None,
        };
    }

    let d = samples[0].len();
    let mut mean = vec![0.0_f64; d];
    let mut variance = vec![0.0_f64; d];
    let mut ess = vec![0.0_f64; d];

    for dim in 0..d {
        let col: Vec<f64> = samples.iter().map(|s| s[dim]).collect();
        let (m, v) = slice_stats(&col);
        mean[dim] = m;
        variance[dim] = v;
        ess[dim] = effective_sample_size(&col);
    }

    ChainDiagnostics {
        n_samples: n,
        acceptance_rate,
        mean,
        variance,
        effective_sample_size: ess,
        r_hat: None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── McmcRng ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rng_uniform_in_range() {
        let mut rng = McmcRng::new(1234);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!(v >= 0.0, "uniform sample below 0: {}", v);
            assert!(v < 1.0, "uniform sample >= 1: {}", v);
        }
    }

    #[test]
    fn test_rng_normal_mean() {
        let mut rng = McmcRng::new(42);
        let samples: Vec<f64> = (0..1000).map(|_| rng.next_normal()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            mean.abs() < 0.15,
            "Box-Muller mean too far from 0: {}",
            mean
        );
    }

    #[test]
    fn test_rng_normal_std() {
        let mut rng = McmcRng::new(99);
        let samples: Vec<f64> = (0..1000).map(|_| rng.next_normal()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 0.15,
            "Box-Muller std too far from 1: {}",
            std
        );
    }

    // ── GaussianProposal ─────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_proposal_log_ratio_is_zero() {
        let proposal = GaussianProposal::new(0.1);
        let current = vec![1.0, 2.0, 3.0];
        let proposed = vec![1.1, 2.2, 3.3];
        assert_eq!(
            proposal.log_ratio(&proposed, &current),
            0.0,
            "Gaussian RW should be symmetric"
        );
    }

    #[test]
    fn test_gaussian_proposal_changes_state() {
        let proposal = GaussianProposal::new(1.0);
        let mut rng = McmcRng::new(7);
        let current = vec![0.0, 0.0, 0.0];
        let proposed = proposal.propose(&current, &mut rng);
        // It is astronomically unlikely for all three to remain exactly 0.
        assert_ne!(proposed, current, "proposal should change the state");
    }

    // ── MetropolisHastings ───────────────────────────────────────────────────

    /// Standard normal target: log p(θ) = -0.5 * θ²
    fn standard_normal_lp() -> LogProbFn<impl Fn(&[f64]) -> f64 + Send + Sync> {
        LogProbFn::new(|theta: &[f64]| -0.5 * theta[0].powi(2))
    }

    #[test]
    fn test_mh_standard_normal_mean() {
        let lp = standard_normal_lp();
        let proposal = GaussianProposal::new(1.0);
        let config = McmcConfig::new().n_samples(2000).n_warmup(500).seed(123);
        let sampler = MetropolisHastings::new(lp, proposal, config);
        let result = sampler.sample(&[0.0]).expect("sampling failed");
        let mean = result.posterior_mean()[0];
        assert!(
            mean.abs() < 0.3,
            "MH posterior mean too far from 0: {}",
            mean
        );
    }

    #[test]
    fn test_mh_standard_normal_variance() {
        let lp = standard_normal_lp();
        let proposal = GaussianProposal::new(1.0);
        let config = McmcConfig::new().n_samples(2000).n_warmup(500).seed(77);
        let sampler = MetropolisHastings::new(lp, proposal, config);
        let result = sampler.sample(&[0.0]).expect("sampling failed");
        let var = result.posterior_variance()[0];
        assert!(
            (var - 1.0).abs() < 0.5,
            "MH posterior variance too far from 1: {}",
            var
        );
    }

    #[test]
    fn test_mh_acceptance_rate_in_range() {
        let lp = standard_normal_lp();
        let proposal = GaussianProposal::new(1.0);
        let config = McmcConfig::new().n_samples(1000).n_warmup(200).seed(55);
        let sampler = MetropolisHastings::new(lp, proposal, config);
        let result = sampler.sample(&[0.0]).expect("sampling failed");
        let ar = result.diagnostics.acceptance_rate;
        assert!(ar > 0.0, "acceptance rate should be > 0");
        assert!(ar <= 1.0, "acceptance rate should be <= 1");
    }

    #[test]
    fn test_mh_sample_count_matches_config() {
        let lp = standard_normal_lp();
        let proposal = GaussianProposal::new(1.0);
        let n = 300;
        let config = McmcConfig::new().n_samples(n).n_warmup(100).seed(11);
        let sampler = MetropolisHastings::new(lp, proposal, config);
        let result = sampler.sample(&[0.0]).expect("sampling failed");
        assert_eq!(result.n_samples(), n, "sample count should match config");
    }

    #[test]
    fn test_mh_warmup_discarded() {
        let lp = standard_normal_lp();
        let proposal = GaussianProposal::new(1.0);
        let n_samples = 200;
        let n_warmup = 100;
        let config = McmcConfig::new()
            .n_samples(n_samples)
            .n_warmup(n_warmup)
            .seed(42);
        let sampler = MetropolisHastings::new(lp, proposal, config);
        let result = sampler.sample(&[0.0]).expect("sampling failed");
        // Result should contain exactly n_samples, not n_samples + n_warmup
        assert_eq!(
            result.n_samples(),
            n_samples,
            "warmup samples should not be included in result"
        );
    }

    // ── McmcResult ───────────────────────────────────────────────────────────

    #[test]
    fn test_marginal_samples_correct() {
        let samples = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let result = McmcResult {
            log_probs: vec![-1.0, -2.0, -3.0],
            diagnostics: compute_diagnostics(&samples),
            samples,
        };
        let m0 = result.marginal_samples(0);
        assert_eq!(m0, vec![1.0, 2.0, 3.0]);
        let m1 = result.marginal_samples(1);
        assert_eq!(m1, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_credible_interval_contains_true_value() {
        let lp = standard_normal_lp();
        let proposal = GaussianProposal::new(1.0);
        let config = McmcConfig::new().n_samples(2000).n_warmup(500).seed(88);
        let sampler = MetropolisHastings::new(lp, proposal, config);
        let result = sampler.sample(&[0.0]).expect("sampling failed");
        let (lo, hi) = result.credible_interval(0, 0.05); // 95% CI
        assert!(
            lo < 0.0 && 0.0 < hi,
            "95% CI should contain the true mean 0.0; got ({}, {})",
            lo,
            hi
        );
    }

    // ── HamiltonianMonteCarlo ────────────────────────────────────────────────

    #[test]
    fn test_hmc_standard_normal_mean() {
        let lp = LogProbFn::new(|theta: &[f64]| -0.5 * theta[0].powi(2));
        let config = McmcConfig::new().n_samples(1000).n_warmup(500).seed(321);
        let sampler = HamiltonianMonteCarlo::new(lp, 0.3, 10, config);
        let result = sampler.sample(&[0.0]).expect("HMC failed");
        let mean = result.posterior_mean()[0];
        assert!(
            mean.abs() < 0.4,
            "HMC posterior mean too far from 0: {}",
            mean
        );
    }

    #[test]
    fn test_hmc_acceptance_rate_high() {
        let lp = LogProbFn::new(|theta: &[f64]| -0.5 * theta[0].powi(2));
        let config = McmcConfig::new().n_samples(500).n_warmup(200).seed(999);
        // Small step + few leapfrog steps should have high acceptance
        let sampler = HamiltonianMonteCarlo::new(lp, 0.1, 5, config);
        let result = sampler.sample(&[0.0]).expect("HMC failed");
        let ar = result.diagnostics.acceptance_rate;
        assert!(
            ar > 0.5,
            "HMC acceptance rate should be > 0.5 with small step size: {}",
            ar
        );
    }

    #[test]
    fn test_hmc_gradient_finite_difference_accuracy() {
        // For f(x) = -0.5 x^2, the gradient at x=1 should be -1.
        let hmc = HamiltonianMonteCarlo::new(
            LogProbFn::new(|theta: &[f64]| -0.5 * theta[0].powi(2)),
            0.1,
            5,
            McmcConfig::new(),
        );
        let grad = hmc.grad_log_prob(&[1.0], 1e-5);
        assert!(
            (grad[0] - (-1.0)).abs() < 1e-6,
            "gradient inaccurate: expected -1, got {}",
            grad[0]
        );
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    #[test]
    fn test_ess_positive_for_iid() {
        let mut rng = McmcRng::new(1);
        let samples: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
        let ess = effective_sample_size(&samples);
        assert!(ess > 0.0, "ESS should be positive");
    }

    #[test]
    fn test_ess_at_most_n_samples() {
        let mut rng = McmcRng::new(2);
        let samples: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
        let ess = effective_sample_size(&samples);
        assert!(
            ess <= samples.len() as f64,
            "ESS should not exceed number of samples"
        );
    }

    #[test]
    fn test_autocorrelation_lag_zero() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ac = autocorrelation(&samples, 0);
        assert!(
            (ac - 1.0).abs() < 1e-10,
            "autocorrelation at lag 0 should be 1.0, got {}",
            ac
        );
    }

    #[test]
    fn test_autocorrelation_large_lag_near_zero() {
        let mut rng = McmcRng::new(3);
        let samples: Vec<f64> = (0..500).map(|_| rng.next_normal()).collect();
        let ac = autocorrelation(&samples, 100);
        assert!(
            ac.abs() < 0.2,
            "autocorrelation at large lag should be near 0 for iid: {}",
            ac
        );
    }

    #[test]
    fn test_gelman_rubin_converged_chains() {
        let mut rng = McmcRng::new(5);
        let chain1: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
        let chain2: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
        let r_hat = gelman_rubin(&[chain1, chain2]);
        assert!(
            !r_hat.is_nan(),
            "R-hat should not be NaN for well-behaved chains"
        );
        assert!(
            r_hat < 1.2,
            "R-hat should be near 1.0 for converged chains, got {}",
            r_hat
        );
    }

    #[test]
    fn test_gelman_rubin_non_converged_chains() {
        // Two chains drawn from very different distributions
        let chain1: Vec<f64> = (0..200).map(|i| i as f64 * 0.01).collect(); // near 0..2
        let chain2: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.01).collect(); // near 100..102
        let r_hat = gelman_rubin(&[chain1, chain2]);
        assert!(
            r_hat > 1.1,
            "R-hat should be > 1.1 for non-converged chains, got {}",
            r_hat
        );
    }

    // ── McmcConfig builder ───────────────────────────────────────────────────

    #[test]
    fn test_mcmc_config_builder_pattern() {
        let cfg = McmcConfig::new()
            .n_samples(500)
            .n_warmup(250)
            .thin(2)
            .seed(17);
        assert_eq!(cfg.n_samples, 500);
        assert_eq!(cfg.n_warmup, 250);
        assert_eq!(cfg.thin, 2);
        assert_eq!(cfg.seed, 17);
    }

    // ── McmcError Display ────────────────────────────────────────────────────

    #[test]
    fn test_mcmc_error_display() {
        let e = McmcError::InvalidConfig("test error".to_string());
        let s = e.to_string();
        assert!(
            s.contains("test error"),
            "error Display should contain the message"
        );
        let e2 = McmcError::DimensionMismatch;
        assert!(
            e2.to_string().len() > 0,
            "DimensionMismatch display should not be empty"
        );
        let e3 = McmcError::NumericalError("NaN".to_string());
        assert!(
            e3.to_string().contains("NaN"),
            "NumericalError display should contain the message"
        );
    }
}
