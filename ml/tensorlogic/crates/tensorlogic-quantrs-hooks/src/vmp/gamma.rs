//! Gamma natural parameters for Variational Message Passing.
//!
//! The Gamma distribution `Gamma(α, β)` with shape α > 0 and **rate** β > 0 is
//! the conjugate prior for the Poisson rate and the precision parameter of a
//! univariate Gaussian (the latter is out of scope for the v0.2.0 research
//! preview). In exponential family form:
//!
//! ```text
//!   p(x | α, β) = (β^α / Γ(α)) · x^{α-1} · exp(-β x)   (x > 0)
//!                = h(x) · exp(ηᵀ u(x) − A(η))
//! ```
//!
//! with base measure `h(x) = 1` on `x > 0`, natural parameters
//! `η = (α − 1, −β)`, sufficient statistics `u(x) = (log x, x)`, and log
//! partition `A(η) = ln Γ(η₁ + 1) − (η₁ + 1) ln(−η₂)`.
//!
//! The struct stores α and β directly for ergonomics; conversion to/from the
//! natural-parameter vector is handled at the [`ExponentialFamily`] trait
//! boundary.
//!
//! # Conjugacy cheat-sheet
//!
//! | Conjugate family | Observation likelihood              |
//! |------------------|-------------------------------------|
//! | Poisson          | `y ~ Poisson(λ)`, λ ~ Gamma          |
//! | Exponential      | `y ~ Exp(λ)`, λ ~ Gamma              |
//! | Gaussian (σ²)    | `y ~ N(μ, σ²)`, τ = 1/σ² ~ Gamma     |
//!
//! Only the Poisson pairing is wired into the VMP engine in v0.2.0; the
//! remaining two can be added without touching [`GammaNP`] itself.

use crate::error::{PgmError, Result};

use super::exponential_family::ExponentialFamily;
use super::special::{digamma, ln_gamma};

/// Gamma distribution stored in (shape, rate) moment parameterisation.
///
/// Natural parameters are `η = (α − 1, −β)`. Both α and β must be strictly
/// positive and finite for the distribution to be well-defined; the
/// constructor and [`ExponentialFamily::set_natural`] reject values outside
/// that open half-plane.
#[derive(Clone, Debug)]
pub struct GammaNP {
    /// Shape parameter α > 0.
    pub alpha: f64,
    /// Rate parameter β > 0 (NOT the scale 1/β).
    pub beta: f64,
}

impl GammaNP {
    /// Construct from moment parameters (α, β). Both must be strictly positive
    /// and finite.
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Gamma shape α must be positive and finite (got {})",
                alpha
            )));
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Gamma rate β must be positive and finite (got {})",
                beta
            )));
        }
        Ok(Self { alpha, beta })
    }

    /// Reconstruct a Gamma from natural parameters `η = (α − 1, −β)`.
    pub fn from_natural(natural: &[f64]) -> Result<Self> {
        if natural.len() != 2 {
            return Err(PgmError::DimensionMismatch {
                expected: vec![2],
                got: vec![natural.len()],
            });
        }
        let alpha = natural[0] + 1.0;
        let beta = -natural[1];
        Self::new(alpha, beta)
    }

    /// Expected value `E[x] = α / β`.
    pub fn expected_x(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Expected log value `E[log x] = ψ(α) − ln β`.
    pub fn expected_log_x(&self) -> f64 {
        digamma(self.alpha) - self.beta.ln()
    }

    /// Variance `Var[x] = α / β²`.
    pub fn variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }

    /// Sum the natural parameters of `self` and `other`. Corresponds to the
    /// pointwise product of densities: if both priors are Gamma on the same
    /// variable, their product is another Gamma whose natural parameter is
    /// the sum of the two input natural parameters.
    ///
    /// Concretely: `α_new = α₁ + α₂ − 1` and `β_new = β₁ + β₂`.
    pub fn multiply_naturals(&self, other: &GammaNP) -> Result<GammaNP> {
        let alpha = self.alpha + other.alpha - 1.0;
        let beta = self.beta + other.beta;
        GammaNP::new(alpha, beta)
    }

    /// Closed-form KL divergence `KL(Gamma(α_p, β_p) || Gamma(α_q, β_q))`.
    ///
    /// Standard result (Penny, 2001):
    ///
    /// ```text
    ///   KL = (α_p − α_q) ψ(α_p) − ln Γ(α_p) + ln Γ(α_q)
    ///        + α_q (ln β_p − ln β_q) + α_p (β_q − β_p) / β_p
    /// ```
    pub fn kl_to(&self, other: &GammaNP) -> f64 {
        let ap = self.alpha;
        let bp = self.beta;
        let aq = other.alpha;
        let bq = other.beta;
        (ap - aq) * digamma(ap) - ln_gamma(ap)
            + ln_gamma(aq)
            + aq * (bp.ln() - bq.ln())
            + ap * (bq - bp) / bp
    }
}

impl ExponentialFamily for GammaNP {
    fn family_name(&self) -> &'static str {
        "Gamma"
    }

    fn natural_dim(&self) -> usize {
        2
    }

    fn natural_params(&self) -> Vec<f64> {
        vec![self.alpha - 1.0, -self.beta]
    }

    fn set_natural(&mut self, new_eta: &[f64]) -> Result<()> {
        if new_eta.len() != 2 {
            return Err(PgmError::DimensionMismatch {
                expected: vec![2],
                got: vec![new_eta.len()],
            });
        }
        for &v in new_eta {
            if !v.is_finite() {
                return Err(PgmError::InvalidDistribution(
                    "Gamma natural parameter must be finite".to_string(),
                ));
            }
        }
        let alpha = new_eta[0] + 1.0;
        let beta = -new_eta[1];
        if alpha <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Gamma shape must stay positive (η₁ + 1 = {} ≤ 0)",
                alpha
            )));
        }
        if beta <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Gamma rate must stay positive (−η₂ = {} ≤ 0)",
                beta
            )));
        }
        self.alpha = alpha;
        self.beta = beta;
        Ok(())
    }

    fn sufficient_statistics(&self, value: f64) -> Vec<f64> {
        // u(x) = (log x, x). For `value <= 0` log x is undefined; we return a
        // best-effort NEG_INFINITY so the caller can see that the stat is
        // degenerate without panicking.
        if value > 0.0 {
            vec![value.ln(), value]
        } else {
            vec![f64::NEG_INFINITY, value]
        }
    }

    fn log_partition(&self, natural_params: &[f64]) -> Result<f64> {
        if natural_params.len() != 2 {
            return Err(PgmError::DimensionMismatch {
                expected: vec![2],
                got: vec![natural_params.len()],
            });
        }
        let alpha = natural_params[0] + 1.0;
        let neg_beta = natural_params[1];
        if alpha <= 0.0 || neg_beta >= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Gamma log_partition: α = {} must be positive and −β = {} negative",
                alpha, neg_beta
            )));
        }
        // A(η) = ln Γ(α) − α ln β.
        let beta = -neg_beta;
        Ok(ln_gamma(alpha) - alpha * beta.ln())
    }

    fn expected_sufficient_statistics(&self) -> Vec<f64> {
        // E[u(x)] = (E[log x], E[x]) = (ψ(α) − ln β, α / β).
        vec![self.expected_log_x(), self.expected_x()]
    }
}

/// Gamma-Poisson conjugate posterior update.
///
/// Given a `Gamma(α_prior, β_prior)` prior on the Poisson rate λ and a batch
/// of `N` observed counts `y_i`, the exact posterior is
/// `Gamma(α_prior + Σ y_i, β_prior + N)`.
///
/// This is exact because Poisson is conjugate to Gamma; the update adds the
/// observation-dependent sufficient statistics (Σ y_i, N) into the natural
/// parameters `(α − 1, −β)` of the prior.
pub fn posterior_from_prior_and_observations(
    prior: &GammaNP,
    observations: &[u64],
) -> Result<GammaNP> {
    let n = observations.len() as f64;
    let sum: u64 = observations.iter().sum();
    let posterior_alpha = prior.alpha + sum as f64;
    let posterior_beta = prior.beta + n;
    GammaNP::new(posterior_alpha, posterior_beta)
}

/// `GammaPoissonObservation` captures a Poisson likelihood `y ~ Poisson(λ)`
/// where the rate `λ` is a `GammaNP` variable. It contributes
/// `(Σ y_i, N)` to the posterior natural parameters, i.e. adds `Σ y_i` to
/// `(α − 1)` and `N` to `−(−β) = β`.
///
/// A factor holds a reference to its Gamma-distributed rate variable and a
/// (possibly empty) batch of observations. Posterior inference combining
/// prior + factor is exact in one VMP sweep because Poisson is conjugate to
/// Gamma.
#[derive(Clone, Debug)]
pub struct GammaPoissonObservation {
    /// Name of the `GammaNP` variable in the VMP graph.
    pub rate_variable: String,
    /// Observed Poisson counts.
    pub observations: Vec<u64>,
}

impl GammaPoissonObservation {
    /// Build a new Gamma-Poisson observation factor.
    pub fn new(rate_variable: impl Into<String>, observations: Vec<u64>) -> Self {
        Self {
            rate_variable: rate_variable.into(),
            observations,
        }
    }

    /// Sum of observed counts Σ y_i. Used as the shape-parameter increment.
    pub fn count_sum(&self) -> u64 {
        self.observations.iter().sum()
    }

    /// Number of observations N. Used as the rate-parameter increment.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmp::special::{digamma, ln_gamma};

    #[test]
    fn gamma_expected_x_matches_alpha_over_beta() {
        for &(alpha, beta) in &[(1.0_f64, 1.0_f64), (2.0, 0.5), (3.7, 4.2), (0.25, 10.0)] {
            let g = GammaNP::new(alpha, beta).expect("ctor");
            let ex = g.expected_x();
            assert!(
                (ex - alpha / beta).abs() < 1e-12,
                "E[x] = {} but α/β = {}",
                ex,
                alpha / beta
            );
        }
    }

    #[test]
    fn gamma_expected_log_x_matches_digamma_minus_lnbeta() {
        for &(alpha, beta) in &[(1.0_f64, 1.0_f64), (2.5, 0.5), (4.0, 2.0)] {
            let g = GammaNP::new(alpha, beta).expect("ctor");
            let el = g.expected_log_x();
            let expected = digamma(alpha) - beta.ln();
            assert!(
                (el - expected).abs() < 1e-12,
                "E[log x] = {}, expected ψ(α)−ln β = {}",
                el,
                expected
            );
        }
    }

    #[test]
    fn gamma_multiply_naturals_sums_natural_params() {
        // Gamma(2, 1) has η = (1, -1); Gamma(3, 2) has η = (2, -2).
        // Sum = (3, -3), i.e. Gamma(4, 3).
        let a = GammaNP::new(2.0, 1.0).expect("ctor a");
        let b = GammaNP::new(3.0, 2.0).expect("ctor b");
        let p = a.multiply_naturals(&b).expect("product");
        assert!((p.alpha - 4.0).abs() < 1e-12, "α = {}", p.alpha);
        assert!((p.beta - 3.0).abs() < 1e-12, "β = {}", p.beta);
        // And the round-trip through natural parameters matches.
        let eta_a = a.natural_params();
        let eta_b = b.natural_params();
        let eta_sum: Vec<f64> = eta_a.iter().zip(eta_b.iter()).map(|(x, y)| x + y).collect();
        let p2 = GammaNP::from_natural(&eta_sum).expect("from nat");
        assert!((p2.alpha - p.alpha).abs() < 1e-12);
        assert!((p2.beta - p.beta).abs() < 1e-12);
    }

    #[test]
    fn gamma_kl_is_zero_for_self_positive_otherwise() {
        let g = GammaNP::new(3.0, 2.0).expect("ctor");
        let self_kl = g.kl_to(&g);
        assert!(self_kl.abs() < 1e-10, "KL(g||g) = {}", self_kl);

        let other = GammaNP::new(1.5, 4.0).expect("ctor other");
        let kl = g.kl_to(&other);
        assert!(kl > 0.0, "KL(g||other) should be positive, got {}", kl);

        // Symmetric-ish sanity: cross KL also > 0.
        let kl_rev = other.kl_to(&g);
        assert!(
            kl_rev > 0.0,
            "KL(other||g) should be positive, got {}",
            kl_rev
        );
    }

    #[test]
    fn gamma_poisson_posterior_adds_sum_and_count() {
        let prior = GammaNP::new(1.0, 1.0).expect("prior");
        let obs: [u64; 3] = [3, 5, 2];
        let post = posterior_from_prior_and_observations(&prior, &obs).expect("posterior");
        // Σ y_i = 10, N = 3, so posterior = Gamma(11, 4).
        assert!((post.alpha - 11.0).abs() < 1e-12, "α = {}", post.alpha);
        assert!((post.beta - 4.0).abs() < 1e-12, "β = {}", post.beta);
    }

    #[test]
    fn gamma_log_partition_matches_closed_form() {
        // A(η) = ln Γ(α) − α ln β.
        let g = GammaNP::new(2.5, 3.0).expect("ctor");
        let eta = g.natural_params();
        let a = g.log_partition(&eta).expect("lp");
        let expected = ln_gamma(2.5) - 2.5 * 3.0_f64.ln();
        assert!(
            (a - expected).abs() < 1e-12,
            "A(η) = {}, expected {}",
            a,
            expected
        );

        // ∂A/∂η₁ = ψ(α) − ln β = E[log x].
        // ∂A/∂η₂: Since β = −η₂, we have ∂A/∂η₂ = (∂A/∂β)(∂β/∂η₂)
        //        = (−α/β)(−1) = α/β = E[x].
        let h = 1e-6;
        let a_plus_1 = g.log_partition(&[eta[0] + h, eta[1]]).expect("lp+1");
        let a_minus_1 = g.log_partition(&[eta[0] - h, eta[1]]).expect("lp-1");
        let d1 = (a_plus_1 - a_minus_1) / (2.0 * h);
        let a_plus_2 = g.log_partition(&[eta[0], eta[1] + h]).expect("lp+2");
        let a_minus_2 = g.log_partition(&[eta[0], eta[1] - h]).expect("lp-2");
        let d2 = (a_plus_2 - a_minus_2) / (2.0 * h);
        assert!(
            (d1 - g.expected_log_x()).abs() < 1e-5,
            "dA/dη1 = {}, expected {}",
            d1,
            g.expected_log_x()
        );
        assert!(
            (d2 - g.expected_x()).abs() < 1e-5,
            "dA/dη2 = {}, expected {}",
            d2,
            g.expected_x()
        );
    }

    #[test]
    fn gamma_natural_round_trip() {
        let g = GammaNP::new(4.5, 2.25).expect("ctor");
        let eta = g.natural_params();
        let back = GammaNP::from_natural(&eta).expect("round trip");
        assert!((back.alpha - 4.5).abs() < 1e-12);
        assert!((back.beta - 2.25).abs() < 1e-12);
    }

    #[test]
    fn gamma_set_natural_rejects_negative_alpha() {
        let mut g = GammaNP::new(2.0, 1.0).expect("ctor");
        let err = g.set_natural(&[-1.5, -1.0]); // α = -0.5
        assert!(err.is_err());
        let err = g.set_natural(&[0.5, 1.0]); // β = -1.0
        assert!(err.is_err());
        let err = g.set_natural(&[0.5, -1.0]); // fine
        assert!(err.is_ok());
    }
}
