//! Sequential Monte Carlo Particle Filter
//!
//! Implements a real-time particle filter (Sequential Importance Resampling, SIR)
//! for Bayesian state estimation of nonlinear, non-Gaussian dynamical systems.
//!
//! ## Algorithm
//!
//! Given a hidden Markov model:
//! ```text
//! x_t = f(x_{t-1}, noise)        (transition / process model)
//! y_t = h(x_t, noise)            (observation model)
//! ```
//!
//! The particle filter maintains a weighted set of particles `{x_t^i, w_t^i}_{i=1}^N`
//! approximating `p(x_t | y_{1:t})`.
//!
//! ## Resampling
//!
//! Three strategies are supported:
//! - **Systematic** (default): O(N), unbiased, low variance
//! - **Stratified**: O(N), similar to systematic
//! - **Multinomial**: O(N log N), highest variance, simplest
//!
//! The filter uses the Effective Sample Size (ESS) criterion to trigger resampling
//! only when particle degeneracy occurs:
//! ```text
//! ESS = 1 / Σ (w_t^i)^2
//! ```
//!
//! ## Usage
//!
//! ```no_run
//! use scirs2_integrate::sde::particle_filter::{ParticleFilter, ParticleFilterConfig, ResamplingStrategy};
//!
//! // 1D linear Gaussian random walk
//! let n_particles = 200;
//! let initial: Vec<f64> = (0..n_particles).map(|_| 0.0f64).collect();
//! let config = ParticleFilterConfig {
//!     n_particles,
//!     resampling: ResamplingStrategy::Systematic,
//!     ess_threshold: 0.5,
//!     seed: 42,
//! };
//! let mut pf = ParticleFilter::new(initial, config).unwrap();
//!
//! // Predict (random walk step with noise 0.1)
//! pf.predict(|&x, rng| x + 0.1 * rng());
//!
//! // Observe y=1.0, Gaussian likelihood sigma=0.5
//! pf.update(|&x| {
//!     let diff = x - 1.0;
//!     -0.5 * diff * diff / (0.5 * 0.5)  // log N(x; 1.0, 0.5^2)
//! });
//!
//! pf.resample_if_needed();
//! let mean = pf.mean(|&x| x);
//! assert!(mean.is_finite());
//! ```

use crate::error::{IntegrateError, IntegrateResult};

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

/// Resampling strategy for the particle filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResamplingStrategy {
    /// Systematic resampling — O(N), low variance, recommended.
    #[default]
    Systematic,
    /// Stratified resampling — O(N), similar to systematic.
    Stratified,
    /// Multinomial resampling — O(N log N), highest variance.
    Multinomial,
}

/// Configuration for [`ParticleFilter`].
#[derive(Debug, Clone)]
pub struct ParticleFilterConfig {
    /// Number of particles.
    pub n_particles: usize,
    /// Resampling strategy to use.
    pub resampling: ResamplingStrategy,
    /// Resample when `ESS < ess_threshold * n_particles`.
    /// Must be in (0, 1]. Default 0.5.
    pub ess_threshold: f64,
    /// Random seed for reproducible results.
    pub seed: u64,
}

impl Default for ParticleFilterConfig {
    fn default() -> Self {
        Self {
            n_particles: 500,
            resampling: ResamplingStrategy::Systematic,
            ess_threshold: 0.5,
            seed: 12345,
        }
    }
}

// ---------------------------------------------------------------------------
// Main particle filter struct
// ---------------------------------------------------------------------------

/// Sequential Importance Resampling (SIR) Particle Filter.
///
/// Maintains a set of `N` particles `x^i` and associated log-weights `log_w^i`
/// to approximate the posterior distribution `p(x_t | y_{1:t})`.
///
/// # Type Parameter
///
/// * `S` — particle state type. Must be `Clone`.
///
/// # Examples
///
/// See the [module-level documentation](self) for a complete example.
#[derive(Debug, Clone)]
pub struct ParticleFilter<S> {
    /// Particle states.
    particles: Vec<S>,
    /// Log-weights (unnormalized). Log-space avoids numerical underflow.
    log_weights: Vec<f64>,
    /// Filter configuration.
    config: ParticleFilterConfig,
    /// Internal PRNG state (PCG-XSH-RR 64-bit).
    rng_state: u64,
    /// PRNG increment (odd number for full-period PCG).
    rng_inc: u64,
}

impl<S: Clone> ParticleFilter<S> {
    /// Create a new particle filter with the given initial particles and config.
    ///
    /// Particles are initialized with equal weights (`1/N` each, i.e. `log(1/N)`).
    ///
    /// # Arguments
    ///
    /// * `initial_particles` — Initial particle states. The number of particles
    ///   is taken from `initial_particles.len()` (not `config.n_particles`).
    /// * `config` — Filter configuration.
    ///
    /// # Errors
    ///
    /// Returns [`IntegrateError::ValueError`] if `initial_particles` is empty.
    pub fn new(initial_particles: Vec<S>, config: ParticleFilterConfig) -> IntegrateResult<Self> {
        let n = initial_particles.len();
        if n == 0 {
            return Err(IntegrateError::ValueError(
                "ParticleFilter requires at least one particle".into(),
            ));
        }
        let log_w = (1.0 / n as f64).ln();
        let log_weights = vec![log_w; n];

        // Seed the PCG-XSH-RR PRNG
        let seed = config.seed;
        let rng_inc = (seed | 1).wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut rng_state = seed
            .wrapping_add(rng_inc)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(rng_inc);
        // Mix the state
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(rng_inc);

        Ok(Self {
            particles: initial_particles,
            log_weights,
            config,
            rng_state,
            rng_inc,
        })
    }

    /// Number of particles.
    #[inline]
    pub fn n_particles(&self) -> usize {
        self.particles.len()
    }

    /// Read-only access to particle states.
    #[inline]
    pub fn particles(&self) -> &[S] {
        &self.particles
    }

    /// Read-only access to log-weights.
    #[inline]
    pub fn log_weights(&self) -> &[f64] {
        &self.log_weights
    }

    /// Effective Sample Size (ESS).
    ///
    /// ```text
    /// ESS = 1 / Σ (w^i)^2
    /// ```
    /// where `w^i` are normalized weights. ESS ∈ [1, N].
    /// ESS = N means all weights are equal; ESS ≈ 1 means severe degeneracy.
    pub fn ess(&self) -> f64 {
        let log_w_norm = self.normalized_log_weights();
        let sum_sq: f64 = log_w_norm
            .iter()
            .map(|&lw| {
                let w = lw.exp();
                w * w
            })
            .sum();
        if sum_sq <= 0.0 {
            self.n_particles() as f64
        } else {
            1.0 / sum_sq
        }
    }

    /// Propagate particles through the transition model.
    ///
    /// # Arguments
    ///
    /// * `transition` — Closure `(particle, rng) -> new_particle`.
    ///   The `rng` argument is a mutable closure that produces `U(0,1)` samples.
    pub fn predict<F>(&mut self, mut transition: F)
    where
        F: FnMut(&S, &mut dyn FnMut() -> f64) -> S,
    {
        let n = self.particles.len();
        let rng_state = &mut self.rng_state;
        let rng_inc = self.rng_inc;
        let mut rng_fn = move || pcg_uniform_f64(rng_state, rng_inc);

        let mut new_particles = Vec::with_capacity(n);
        for i in 0..n {
            let p_new = transition(&self.particles[i], &mut rng_fn);
            new_particles.push(p_new);
        }
        self.particles = new_particles;
    }

    /// Update particle log-weights with the observation log-likelihood.
    ///
    /// # Arguments
    ///
    /// * `log_likelihood` — Closure `particle -> log p(y_t | x_t^i)`.
    pub fn update<F>(&mut self, log_likelihood: F)
    where
        F: Fn(&S) -> f64,
    {
        for (i, p) in self.particles.iter().enumerate() {
            let ll = log_likelihood(p);
            self.log_weights[i] += ll;
        }
    }

    /// Resample particles if ESS drops below the threshold.
    ///
    /// After resampling, log-weights are reset to `log(1/N)`.
    pub fn resample_if_needed(&mut self) {
        let threshold = self.config.ess_threshold * self.n_particles() as f64;
        if self.ess() < threshold {
            self.resample();
        }
    }

    /// Force resampling (regardless of ESS).
    pub fn resample(&mut self) {
        let indices = match self.config.resampling {
            ResamplingStrategy::Systematic => self.systematic_resample(),
            ResamplingStrategy::Stratified => self.stratified_resample(),
            ResamplingStrategy::Multinomial => self.multinomial_resample(),
        };

        let old_particles = std::mem::take(&mut self.particles);
        self.particles = indices.iter().map(|&i| old_particles[i].clone()).collect();

        let n = self.particles.len();
        let log_w = (1.0 / n as f64).ln();
        self.log_weights.iter_mut().for_each(|w| *w = log_w);
    }

    /// Weighted mean of a function over particles.
    ///
    /// Computes `Σ w^i * f(x^i)` where `w^i` are normalized weights.
    ///
    /// # Type constraints
    ///
    /// `T` must implement `Default`, `Clone`, scalar-multiply by `f64`, and
    /// `Sum<T>`. For primitive types like `f64` this is automatic.
    pub fn mean<F, T>(&self, f: F) -> T
    where
        F: Fn(&S) -> T,
        T: Default + Clone + std::ops::Mul<f64, Output = T> + std::iter::Sum,
    {
        let log_w_norm = self.normalized_log_weights();
        self.particles
            .iter()
            .zip(log_w_norm.iter())
            .map(|(p, &lw)| f(p) * lw.exp())
            .sum()
    }

    /// Return a reference to the particle with the highest weight (MAP estimate).
    pub fn map_estimate(&self) -> &S {
        let best_idx = self
            .log_weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        &self.particles[best_idx]
    }

    /// Compute variance of a scalar-valued function over particles (using normalized weights).
    pub fn variance<F>(&self, f: F) -> f64
    where
        F: Fn(&S) -> f64,
    {
        let log_w_norm = self.normalized_log_weights();
        let vals: Vec<f64> = self.particles.iter().map(&f).collect();
        let mean: f64 = vals
            .iter()
            .zip(log_w_norm.iter())
            .map(|(v, &lw)| v * lw.exp())
            .sum();
        let var: f64 = vals
            .iter()
            .zip(log_w_norm.iter())
            .map(|(v, &lw)| {
                let diff = v - mean;
                lw.exp() * diff * diff
            })
            .sum();
        var
    }

    // ---- Private helpers ----

    /// Normalized log-weights (log-sum-exp stabilized).
    fn normalized_log_weights(&self) -> Vec<f64> {
        let max_lw = self
            .log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = self.log_weights.iter().map(|&lw| (lw - max_lw).exp()).sum();
        let log_z = max_lw + sum_exp.ln();
        self.log_weights.iter().map(|&lw| lw - log_z).collect()
    }

    /// Normalized weights in linear space.
    fn normalized_weights(&self) -> Vec<f64> {
        let log_w_norm = self.normalized_log_weights();
        log_w_norm.iter().map(|&lw| lw.exp()).collect()
    }

    /// Systematic resampling: O(N), low variance.
    fn systematic_resample(&mut self) -> Vec<usize> {
        let n = self.n_particles();
        let weights = self.normalized_weights();
        let mut cumsum = vec![0.0f64; n];
        cumsum[0] = weights[0];
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + weights[i];
        }

        let u0 = pcg_uniform_f64(&mut self.rng_state, self.rng_inc) / n as f64;
        let mut indices = Vec::with_capacity(n);
        let mut j = 0usize;

        for i in 0..n {
            let u = u0 + i as f64 / n as f64;
            while j < n - 1 && cumsum[j] < u {
                j += 1;
            }
            indices.push(j);
        }
        indices
    }

    /// Stratified resampling: O(N), similar to systematic.
    fn stratified_resample(&mut self) -> Vec<usize> {
        let n = self.n_particles();
        let weights = self.normalized_weights();
        let mut cumsum = vec![0.0f64; n];
        cumsum[0] = weights[0];
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + weights[i];
        }

        let mut indices = Vec::with_capacity(n);
        let mut j = 0usize;

        for i in 0..n {
            let u = (i as f64 + pcg_uniform_f64(&mut self.rng_state, self.rng_inc)) / n as f64;
            while j < n - 1 && cumsum[j] < u {
                j += 1;
            }
            indices.push(j);
        }
        indices
    }

    /// Multinomial resampling: O(N log N).
    fn multinomial_resample(&mut self) -> Vec<usize> {
        let n = self.n_particles();
        let weights = self.normalized_weights();
        let mut cumsum = vec![0.0f64; n + 1];
        for i in 0..n {
            cumsum[i + 1] = cumsum[i] + weights[i];
        }
        // Normalize last entry to exactly 1 to avoid rounding issues
        cumsum[n] = 1.0;

        let mut indices = Vec::with_capacity(n);
        for _ in 0..n {
            let u = pcg_uniform_f64(&mut self.rng_state, self.rng_inc);
            // Binary search in cumsum
            let idx = cumsum
                .partition_point(|&c| c < u)
                .saturating_sub(1)
                .min(n - 1);
            indices.push(idx);
        }
        indices
    }
}

// ---------------------------------------------------------------------------
// PCG XSH-RR 64-bit PRNG (pure Rust, no external deps)
// ---------------------------------------------------------------------------

/// Single step of PCG-XSH-RR (64-bit state, 32-bit output).
/// Returns a sample uniform on [0, 1).
#[inline]
fn pcg_uniform_f64(state: &mut u64, inc: u64) -> f64 {
    // Advance state
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(inc);
    // XSH-RR output function
    let xorshifted = (((*state >> 18) ^ *state) >> 27) as u32;
    let rot = (*state >> 59) as u32;
    let bits = xorshifted.rotate_right(rot);
    // Map to [0, 1)
    (bits as f64) * (1.0 / (1u64 << 32) as f64)
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Build a particle filter initializing all particles at the same state.
///
/// # Arguments
///
/// * `initial_state` — Initial state, cloned N times.
/// * `config` — Filter configuration.
pub fn particle_filter_uniform_init<S: Clone>(
    initial_state: S,
    config: ParticleFilterConfig,
) -> IntegrateResult<ParticleFilter<S>> {
    let n = config.n_particles;
    let particles = vec![initial_state; n];
    ParticleFilter::new(particles, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate particles as f64 values scattered around `center` with noise `std`.
    fn make_particles(n: usize, center: f64, std: f64, seed: u64) -> Vec<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Box-Muller (approximate) using two uniform samples
                let u1 = (state >> 32) as f64 / u32::MAX as f64;
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u2 = (state >> 32) as f64 / u32::MAX as f64;
                let z =
                    (-2.0 * (u1.max(1e-15)).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                center + std * z
            })
            .collect()
    }

    #[test]
    fn test_particle_filter_new_ok() {
        let particles = make_particles(100, 0.0, 1.0, 42);
        let config = ParticleFilterConfig {
            n_particles: 100,
            ..Default::default()
        };
        let pf = ParticleFilter::new(particles, config);
        assert!(pf.is_ok(), "construction should succeed");
        let pf = pf.expect("already checked ok");
        assert_eq!(pf.n_particles(), 100);
    }

    #[test]
    fn test_particle_filter_new_empty_fails() {
        let config = ParticleFilterConfig::default();
        let pf: Result<ParticleFilter<f64>, _> = ParticleFilter::new(vec![], config);
        assert!(pf.is_err(), "empty particles should fail");
    }

    #[test]
    fn test_ess_normalized() {
        // Equal weights: ESS should be close to N
        let n = 200;
        let particles = make_particles(n, 0.0, 1.0, 7);
        let config = ParticleFilterConfig {
            n_particles: n,
            seed: 7,
            ..Default::default()
        };
        let pf = ParticleFilter::new(particles, config).expect("construct");
        let ess = pf.ess();
        // With equal weights ESS = N exactly
        assert!(
            (ess - n as f64).abs() < 1.0,
            "ESS should be ≈ N={} for equal weights, got {}",
            n,
            ess
        );
        // ESS in [1, N]
        assert!(ess >= 1.0 - 1e-10, "ESS >= 1, got {}", ess);
        assert!(ess <= n as f64 + 1e-10, "ESS <= N, got {}", ess);
    }

    #[test]
    fn test_ess_after_degeneracy() {
        // Give one particle a huge weight, ESS should drop near 1
        let n = 100;
        let particles = make_particles(n, 0.0, 1.0, 99);
        let config = ParticleFilterConfig {
            n_particles: n,
            seed: 99,
            ..Default::default()
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");
        // Set first particle to very high weight, others to very low
        pf.log_weights[0] = 0.0;
        for w in pf.log_weights[1..].iter_mut() {
            *w = -1000.0;
        }
        let ess = pf.ess();
        assert!(
            ess < 2.0,
            "degenerate weights should give ESS ≈ 1, got {}",
            ess
        );
    }

    #[test]
    fn test_particle_filter_linear_gaussian() {
        // Linear Gaussian random walk: x_t = x_{t-1} + N(0, 0.1^2)
        // Observations: y_t = x_t + N(0, 0.5^2)
        // True trajectory starts at 0 and increases to ~2 over 20 steps
        let n = 500;
        let true_x: Vec<f64> = {
            let mut x = 0.0f64;
            let mut traj = vec![x];
            let mut state: u64 = 12345;
            for _ in 0..20 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = (state >> 32) as f64 / u32::MAX as f64 * 0.2 - 0.1;
                x += noise + 0.1; // drift upward
                traj.push(x);
            }
            traj
        };

        let observations: Vec<f64> = {
            let mut state: u64 = 99999;
            true_x
                .iter()
                .map(|&x| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let noise = (state >> 32) as f64 / u32::MAX as f64 * 1.0 - 0.5;
                    x + noise
                })
                .collect()
        };

        let particles = make_particles(n, 0.0, 1.0, 42);
        let config = ParticleFilterConfig {
            n_particles: n,
            resampling: ResamplingStrategy::Systematic,
            ess_threshold: 0.5,
            seed: 42,
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");

        let obs_std = 0.5f64;
        let proc_std = 0.1f64;

        for &obs in &observations {
            // Predict: random walk
            let ps = proc_std;
            pf.predict(|&x, rng| {
                let u1 = rng().max(1e-15);
                let u2 = rng();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                x + ps * z
            });

            // Update: Gaussian log-likelihood
            let os = obs_std;
            pf.update(|&x| {
                let diff = x - obs;
                -0.5 * diff * diff / (os * os)
            });

            pf.resample_if_needed();
        }

        let final_mean = pf.mean(|&x| x);
        let final_true = *true_x.last().unwrap_or(&0.0);

        // The filter should track the true state within 2 standard deviations
        assert!(
            (final_mean - final_true).abs() < 3.0,
            "Filter mean {} should be close to true state {}",
            final_mean,
            final_true
        );
    }

    #[test]
    fn test_resample_maintains_distribution() {
        // After resampling, weighted mean should be consistent before and after
        let n = 200;
        let particles: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let config = ParticleFilterConfig {
            n_particles: n,
            resampling: ResamplingStrategy::Systematic,
            seed: 55,
            ..Default::default()
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");

        // Apply non-uniform weights (favor middle particles)
        for (i, w) in pf.log_weights.iter_mut().enumerate() {
            let x = i as f64 / n as f64 - 0.5;
            *w = -x * x * 10.0; // peak near 0.5
        }

        let mean_before = pf.mean(|&x| x);
        pf.resample();
        let mean_after = pf.mean(|&x| x);

        // Mean should be preserved within sampling error
        assert!(
            (mean_before - mean_after).abs() < 0.1,
            "Resample should preserve distribution mean: before={}, after={}",
            mean_before,
            mean_after
        );
    }

    #[test]
    fn test_map_estimate_is_max_weight_particle() {
        let n = 50;
        let particles: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = ParticleFilterConfig {
            n_particles: n,
            seed: 7,
            ..Default::default()
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");

        // Set particle 37 to have maximum weight
        pf.log_weights[37] = 100.0;

        let map = pf.map_estimate();
        assert!(
            (*map - 37.0f64).abs() < 1e-10,
            "MAP estimate should be particle 37 (value 37.0), got {}",
            map
        );
    }

    #[test]
    fn test_stratified_resample_output_size() {
        let n = 100;
        let particles: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = ParticleFilterConfig {
            n_particles: n,
            resampling: ResamplingStrategy::Stratified,
            seed: 21,
            ..Default::default()
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");
        pf.resample();
        assert_eq!(
            pf.n_particles(),
            n,
            "particle count must stay at N after resample"
        );
    }

    #[test]
    fn test_multinomial_resample_output_size() {
        let n = 80;
        let particles: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = ParticleFilterConfig {
            n_particles: n,
            resampling: ResamplingStrategy::Multinomial,
            seed: 33,
            ..Default::default()
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");
        pf.resample();
        assert_eq!(
            pf.n_particles(),
            n,
            "particle count must stay at N after multinomial resample"
        );
    }

    #[test]
    fn test_predict_changes_particles() {
        let n = 50;
        let particles: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let original: Vec<f64> = particles.clone();
        let config = ParticleFilterConfig {
            n_particles: n,
            seed: 77,
            ..Default::default()
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");

        // Deterministic shift by 1
        pf.predict(|&x, _rng| x + 1.0);

        for (orig, new) in original.iter().zip(pf.particles().iter()) {
            assert!(
                (new - orig - 1.0).abs() < 1e-12,
                "predict should shift each particle by 1"
            );
        }
    }

    #[test]
    fn test_update_modifies_log_weights() {
        let n = 20;
        let particles: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = ParticleFilterConfig {
            n_particles: n,
            seed: 9,
            ..Default::default()
        };
        let mut pf = ParticleFilter::new(particles, config).expect("construct");

        let initial_sum: f64 = pf.log_weights().iter().sum();
        pf.update(|&x| -x); // log-likelihood decreases with x
        let after_sum: f64 = pf.log_weights().iter().sum();

        // With non-constant likelihood, weights must change
        assert!(
            (after_sum - initial_sum).abs() > 1e-3,
            "update must change log-weights"
        );
    }
}
