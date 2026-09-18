//! Mean-field Gaussian Variational Inference.
//!
//! This module implements the standard "black-box" variational inference (BBVI)
//! algorithm for fitting a mean-field Gaussian posterior q(z) = ∏ N(μ_i, σ_i²)
//! to an unnormalised target log p(z).
//!
//! ## Algorithm
//!
//! Maximise the ELBO:
//! ```text
//! L(λ) = E_q[log p(z)] + H[q]
//! ```
//! where H\[q\] = Σ_i log σ_i + const (analytic entropy of a diagonal Gaussian).
//!
//! Gradients of E_q[log p(z)] are estimated via the reparameterisation trick:
//! ```text
//! ∂L/∂μ_i      ≈ (1/S) Σ_s ∂log p(z_s)/∂z_s^i
//! ∂L/∂log_σ_i  ≈ (1/S) Σ_s ∂log p(z_s)/∂z_s^i · ε_s^i · σ_i  +  1
//! ```
//! where z_s = μ + σ ⊙ ε_s, ε_s ~ N(0, I), and gradients of log p are
//! computed by central finite differences.  Adam updates are applied to all
//! variational parameters.

use crate::error::{TlBackendError, TlBackendResult};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Distribution;

// ============================================================================
// MeanFieldGaussian
// ============================================================================

/// Diagonal (mean-field) Gaussian variational distribution q(z) = ∏ N(μ_i, σ_i²).
#[derive(Debug, Clone)]
pub struct MeanFieldGaussian {
    /// Location parameters μ.
    pub mu: Vec<f64>,
    /// Log-scale parameters log σ (σ = exp(log_sigma) > 0 always).
    pub log_sigma: Vec<f64>,
}

impl MeanFieldGaussian {
    /// Dimensionality of the distribution.
    pub fn dim(&self) -> usize {
        self.mu.len()
    }

    /// Scale parameters σ = exp(log_sigma).
    pub fn sigma(&self) -> Vec<f64> {
        self.log_sigma.iter().map(|&ls| ls.exp()).collect()
    }

    /// Draw one sample via the reparameterisation trick: z = μ + σ ⊙ ε, ε ~ N(0, I).
    pub fn sample(&self, rng: &mut impl Rng) -> Vec<f64> {
        let sigma = self.sigma();
        let normal = Normal::new(0.0_f64, 1.0).expect("N(0,1) is always valid");
        self.mu
            .iter()
            .zip(sigma.iter())
            .map(|(&m, &s)| {
                let eps: f64 = normal.sample(rng);
                m + s * eps
            })
            .collect()
    }
}

// ============================================================================
// VariationalConfig
// ============================================================================

/// Hyper-parameters for the variational inference algorithm.
#[derive(Debug, Clone)]
pub struct VariationalConfig {
    /// Number of gradient ascent steps.
    pub steps: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Number of MC samples per ELBO gradient estimate.
    pub mc_samples: usize,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for VariationalConfig {
    fn default() -> Self {
        Self {
            steps: 500,
            learning_rate: 0.01,
            mc_samples: 10,
            seed: None,
        }
    }
}

// ============================================================================
// VariationalInference
// ============================================================================

/// Entry-point for black-box mean-field variational inference.
pub struct VariationalInference;

impl VariationalInference {
    /// Fit a mean-field Gaussian posterior q(z) to the log-joint `log_prob`.
    ///
    /// # Arguments
    /// * `log_prob` — evaluates log p(z) at a given z (up to a constant)
    /// * `dim`      — number of latent dimensions
    /// * `config`   — algorithm hyper-parameters
    ///
    /// # Algorithm
    /// Uses Adam (β₁=0.9, β₂=0.999, ε=1e-8) to maximise the ELBO.  Gradients
    /// of the expected log-joint are estimated by the reparameterisation trick
    /// with central finite differences (h=1e-5) for ∂log_p/∂z.
    ///
    /// # Errors
    /// Returns an error if `dim == 0`.
    pub fn fit(
        log_prob: impl Fn(&[f64]) -> f64,
        dim: usize,
        config: VariationalConfig,
    ) -> TlBackendResult<MeanFieldGaussian> {
        if dim == 0 {
            return Err(TlBackendError::InvalidOperation(
                "VariationalInference::fit: dim must be > 0".to_string(),
            ));
        }

        if let Some(s) = config.seed {
            let mut rng = seeded_rng(s);
            fit_inner(log_prob, dim, &config, &mut rng)
        } else {
            let mut rng = thread_rng();
            fit_inner(log_prob, dim, &config, &mut rng)
        }
    }
}

// ============================================================================
// Inner optimisation loop (concrete RNG type to allow Distribution::sample)
// ============================================================================

/// Core optimisation loop parametric over a concrete RNG type.
///
/// The seeded/unseeded branches in `VariationalInference::fit` delegate here
/// with their respective concrete `Random<StdRng>` / `Random<ThreadRng>` types,
/// following the same dual-branch pattern as `gradient_ops::sample_gumbel`.
fn fit_inner<R: Rng>(
    log_prob: impl Fn(&[f64]) -> f64,
    dim: usize,
    config: &VariationalConfig,
    rng: &mut Random<R>,
) -> TlBackendResult<MeanFieldGaussian> {
    // ---- Initialisation ------------------------------------------------
    // μ = 0, log_σ = 0 (σ = 1)
    let mut mu = vec![0.0_f64; dim];
    let mut log_sigma = vec![0.0_f64; dim];

    // Adam moment buffers for μ and log_σ
    let mut m_mu = vec![0.0_f64; dim];
    let mut v_mu = vec![0.0_f64; dim];
    let mut m_ls = vec![0.0_f64; dim];
    let mut v_ls = vec![0.0_f64; dim];

    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let adam_eps = 1e-8_f64;
    let fd_h = 1e-5_f64;

    let normal_dist = Normal::new(0.0_f64, 1.0).expect("N(0,1) is always valid");

    // ---- Main loop -----------------------------------------------------
    for step in 0..config.steps {
        let adam_t = step + 1;

        // Gradient accumulators
        let mut grad_mu = vec![0.0_f64; dim];
        let mut grad_ls = vec![0.0_f64; dim];

        let sigma: Vec<f64> = log_sigma.iter().map(|&ls| ls.exp()).collect();

        // MC estimate of ∂E_q[log p(z)] / ∂(μ, log_σ)
        for _ in 0..config.mc_samples {
            // Sample ε ~ N(0, I) using the concrete rng type
            let eps: Vec<f64> = (0..dim).map(|_| rng.sample(normal_dist)).collect();

            // z = μ + σ ⊙ ε  (reparameterisation)
            let z: Vec<f64> = mu
                .iter()
                .zip(sigma.iter())
                .zip(eps.iter())
                .map(|((&m, &s), &e)| m + s * e)
                .collect();

            // Central finite differences: ∂log_p(z)/∂z_i ≈ (log_p(z+h) - log_p(z-h)) / 2h
            let grad_log_p = compute_fd_gradient(&log_prob, &z, fd_h);

            // Accumulate gradients
            for i in 0..dim {
                grad_mu[i] += grad_log_p[i];
                // Reparameterisation gradient w.r.t. log_σ_i:
                // ∂z_i/∂log_σ_i = σ_i · ε_i  (chain rule through z = μ + exp(log_σ)*ε)
                grad_ls[i] += grad_log_p[i] * eps[i] * sigma[i];
            }
        }

        let inv_s = 1.0 / config.mc_samples as f64;
        for i in 0..dim {
            grad_mu[i] *= inv_s;
            // Analytic entropy gradient ∂H/∂log_σ_i = 1
            grad_ls[i] = grad_ls[i] * inv_s + 1.0;
        }

        // ---- Adam update for μ -----------------------------------------
        for i in 0..dim {
            m_mu[i] = beta1 * m_mu[i] + (1.0 - beta1) * grad_mu[i];
            v_mu[i] = beta2 * v_mu[i] + (1.0 - beta2) * grad_mu[i].powi(2);
            let m_hat = m_mu[i] / (1.0 - beta1.powi(adam_t as i32));
            let v_hat = v_mu[i] / (1.0 - beta2.powi(adam_t as i32));
            mu[i] += config.learning_rate * m_hat / (v_hat.sqrt() + adam_eps);
        }

        // ---- Adam update for log_σ -------------------------------------
        for i in 0..dim {
            m_ls[i] = beta1 * m_ls[i] + (1.0 - beta1) * grad_ls[i];
            v_ls[i] = beta2 * v_ls[i] + (1.0 - beta2) * grad_ls[i].powi(2);
            let m_hat = m_ls[i] / (1.0 - beta1.powi(adam_t as i32));
            let v_hat = v_ls[i] / (1.0 - beta2.powi(adam_t as i32));
            log_sigma[i] += config.learning_rate * m_hat / (v_hat.sqrt() + adam_eps);
        }
    }

    Ok(MeanFieldGaussian { mu, log_sigma })
}

// ============================================================================
// Finite-difference gradient helper
// ============================================================================

/// Compute the gradient of `f` at `z` via central finite differences with step `h`.
fn compute_fd_gradient(f: &impl Fn(&[f64]) -> f64, z: &[f64], h: f64) -> Vec<f64> {
    let dim = z.len();
    let mut grad = Vec::with_capacity(dim);
    let mut z_plus = z.to_vec();
    let mut z_minus = z.to_vec();

    for i in 0..dim {
        z_plus[i] = z[i] + h;
        z_minus[i] = z[i] - h;
        let g = (f(&z_plus) - f(&z_minus)) / (2.0 * h);
        grad.push(g);
        z_plus[i] = z[i];
        z_minus[i] = z[i];
    }
    grad
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mfg_dim() {
        let mfg = MeanFieldGaussian {
            mu: vec![1.0, 2.0, 3.0],
            log_sigma: vec![0.0, 0.0, 0.0],
        };
        assert_eq!(mfg.dim(), 3);
    }

    #[test]
    fn mfg_sigma() {
        let log_sigma = vec![-1.0, 0.0, 1.0];
        let mfg = MeanFieldGaussian {
            mu: vec![0.0; 3],
            log_sigma: log_sigma.clone(),
        };
        let sigma = mfg.sigma();
        for (got, &ls) in sigma.iter().zip(log_sigma.iter()) {
            assert!(
                (got - ls.exp()).abs() < 1e-12,
                "sigma mismatch: got {got}, expected {}",
                ls.exp()
            );
        }
    }

    #[test]
    fn vi_recovers_gaussian_mean() {
        // Target: log p(z) = -0.5 * ||z - mu_true||^2 / sigma_true^2
        // → posterior mean should converge to mu_true = [2.0, 3.0]
        let mu_true = [2.0_f64, 3.0_f64];
        let sigma_true = 1.0_f64;
        let log_prob = move |z: &[f64]| {
            -0.5 * z
                .iter()
                .zip(mu_true.iter())
                .map(|(&zi, &mi)| ((zi - mi) / sigma_true).powi(2))
                .sum::<f64>()
        };

        let config = VariationalConfig {
            steps: 2000,
            learning_rate: 0.05,
            mc_samples: 20,
            seed: Some(42),
        };
        let mfg = VariationalInference::fit(log_prob, 2, config).expect("fit failed");

        assert!(
            (mfg.mu[0] - mu_true[0]).abs() < 0.3,
            "mu[0]={} not close to {}",
            mfg.mu[0],
            mu_true[0]
        );
        assert!(
            (mfg.mu[1] - mu_true[1]).abs() < 0.3,
            "mu[1]={} not close to {}",
            mfg.mu[1],
            mu_true[1]
        );
    }

    #[test]
    fn vi_recovers_gaussian_variance() {
        // Same target: posterior variance should converge to sigma_true^2 = 1.0
        let mu_true = [2.0_f64, 3.0_f64];
        let sigma_true = 1.0_f64;
        let log_prob = move |z: &[f64]| {
            -0.5 * z
                .iter()
                .zip(mu_true.iter())
                .map(|(&zi, &mi)| ((zi - mi) / sigma_true).powi(2))
                .sum::<f64>()
        };

        let config = VariationalConfig {
            steps: 2000,
            learning_rate: 0.05,
            mc_samples: 20,
            seed: Some(42),
        };
        let mfg = VariationalInference::fit(log_prob, 2, config).expect("fit failed");
        let sigma = mfg.sigma();

        // Each σ_i should be close to sigma_true = 1.0 (within 30%)
        for (i, &s) in sigma.iter().enumerate() {
            assert!(
                (s - sigma_true).abs() < 0.3 * sigma_true,
                "sigma[{i}]={s} not within 30% of {sigma_true}"
            );
        }
    }

    #[test]
    fn vi_runs_without_error() {
        // Arbitrary log_prob; just verify no panic/error
        let log_prob = |z: &[f64]| -z.iter().map(|&v| v.powi(2)).sum::<f64>();
        let config = VariationalConfig {
            steps: 50,
            learning_rate: 0.01,
            mc_samples: 5,
            seed: Some(7),
        };
        VariationalInference::fit(log_prob, 3, config).expect("fit should not fail");
    }
}
