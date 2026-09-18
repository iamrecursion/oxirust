//! Beta natural parameters for Variational Message Passing.
//!
//! The Beta distribution `Beta(α, β)` with α > 0 and β > 0 is the conjugate
//! prior for the success probability of a Bernoulli / Binomial likelihood. In
//! exponential family form:
//!
//! ```text
//!   p(x | α, β) = (Γ(α + β) / (Γ(α) Γ(β))) · x^{α-1} (1 - x)^{β-1}   (0 < x < 1)
//!                = h(x) · exp(ηᵀ u(x) − A(η))
//! ```
//!
//! with base measure `h(x) = 1` on `(0, 1)`, natural parameters
//! `η = (α − 1, β − 1)`, sufficient statistics `u(x) = (log x, log(1 − x))`, and
//! log partition `A(η) = ln Γ(η₁ + 1) + ln Γ(η₂ + 1) − ln Γ(η₁ + η₂ + 2)`.
//!
//! The struct stores α and β directly for ergonomics; conversion to/from the
//! natural-parameter vector is handled at the [`ExponentialFamily`] trait
//! boundary.
//!
//! # Conjugacy cheat-sheet
//!
//! | Conjugate family | Observation likelihood              |
//! |------------------|-------------------------------------|
//! | Bernoulli        | `y ~ Bernoulli(p)`, p ~ Beta         |
//! | Binomial         | `y ~ Binomial(N, p)`, p ~ Beta       |
//!
//! Only the Bernoulli pairing is wired into the VMP engine in v0.2.0; Binomial
//! can be added with the same machinery (it contributes `(n_s, n_f)` to the
//! natural parameters, just like a batch of Bernoulli draws).

use crate::error::{PgmError, Result};

use super::exponential_family::ExponentialFamily;
use super::special::{digamma, ln_gamma};

/// Beta distribution stored in (α, β) moment parameterisation.
///
/// Natural parameters are `η = (α − 1, β − 1)`. Both α and β must be strictly
/// positive and finite for the distribution to be well-defined; the constructor
/// and [`ExponentialFamily::set_natural`] reject values outside that open
/// positive quadrant.
#[derive(Clone, Debug)]
pub struct BetaNP {
    /// Shape parameter α > 0.
    pub alpha: f64,
    /// Shape parameter β > 0.
    pub beta: f64,
}

impl BetaNP {
    /// Construct from moment parameters (α, β). Both must be strictly positive
    /// and finite.
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Beta shape α must be positive and finite (got {})",
                alpha
            )));
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Beta shape β must be positive and finite (got {})",
                beta
            )));
        }
        Ok(Self { alpha, beta })
    }

    /// Reconstruct a Beta from natural parameters `η = (α − 1, β − 1)`.
    pub fn from_natural(natural: &[f64]) -> Result<Self> {
        if natural.len() != 2 {
            return Err(PgmError::DimensionMismatch {
                expected: vec![2],
                got: vec![natural.len()],
            });
        }
        let alpha = natural[0] + 1.0;
        let beta = natural[1] + 1.0;
        Self::new(alpha, beta)
    }

    /// Expected value `E[x] = α / (α + β)`.
    pub fn expected_x(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Expected log value `E[log x] = ψ(α) − ψ(α + β)`.
    pub fn expected_log_x(&self) -> f64 {
        digamma(self.alpha) - digamma(self.alpha + self.beta)
    }

    /// Expected log of the complement `E[log(1 − x)] = ψ(β) − ψ(α + β)`.
    pub fn expected_log_1mx(&self) -> f64 {
        digamma(self.beta) - digamma(self.alpha + self.beta)
    }

    /// Variance `Var[x] = α β / ((α + β)² (α + β + 1))`.
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + 1.0))
    }

    /// Sum the natural parameters of `self` and `other`. Corresponds to the
    /// pointwise product of densities: if both priors are Beta on the same
    /// variable, their product is another Beta whose natural parameter is the
    /// sum of the two input natural parameters.
    ///
    /// Concretely: `α_new = α₁ + α₂ − 1` and `β_new = β₁ + β₂ − 1`.
    pub fn multiply_naturals(&self, other: &BetaNP) -> Result<BetaNP> {
        let alpha = self.alpha + other.alpha - 1.0;
        let beta = self.beta + other.beta - 1.0;
        BetaNP::new(alpha, beta)
    }

    /// Closed-form KL divergence `KL(Beta(α_p, β_p) || Beta(α_q, β_q))`.
    ///
    /// Standard result:
    ///
    /// ```text
    ///   KL = ln B(α_q, β_q) − ln B(α_p, β_p)
    ///        + (α_p − α_q) ψ(α_p)
    ///        + (β_p − β_q) ψ(β_p)
    ///        + (α_q − α_p + β_q − β_p) ψ(α_p + β_p)
    /// ```
    ///
    /// where `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`.
    pub fn kl_to(&self, other: &BetaNP) -> f64 {
        let ap = self.alpha;
        let bp = self.beta;
        let aq = other.alpha;
        let bq = other.beta;
        let ln_beta_p = ln_gamma(ap) + ln_gamma(bp) - ln_gamma(ap + bp);
        let ln_beta_q = ln_gamma(aq) + ln_gamma(bq) - ln_gamma(aq + bq);
        let psi_ap = digamma(ap);
        let psi_bp = digamma(bp);
        let psi_abp = digamma(ap + bp);
        ln_beta_q - ln_beta_p
            + (ap - aq) * psi_ap
            + (bp - bq) * psi_bp
            + (aq - ap + bq - bp) * psi_abp
    }
}

impl ExponentialFamily for BetaNP {
    fn family_name(&self) -> &'static str {
        "Beta"
    }

    fn natural_dim(&self) -> usize {
        2
    }

    fn natural_params(&self) -> Vec<f64> {
        vec![self.alpha - 1.0, self.beta - 1.0]
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
                    "Beta natural parameter must be finite".to_string(),
                ));
            }
        }
        let alpha = new_eta[0] + 1.0;
        let beta = new_eta[1] + 1.0;
        if alpha <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Beta shape α must stay positive (η₁ + 1 = {} ≤ 0)",
                alpha
            )));
        }
        if beta <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Beta shape β must stay positive (η₂ + 1 = {} ≤ 0)",
                beta
            )));
        }
        self.alpha = alpha;
        self.beta = beta;
        Ok(())
    }

    fn sufficient_statistics(&self, value: f64) -> Vec<f64> {
        // u(x) = (log x, log(1 - x)). For `value` outside (0, 1) the stat is
        // degenerate; we return NEG_INFINITY rather than panicking so the
        // caller can detect the bad input.
        if value > 0.0 && value < 1.0 {
            vec![value.ln(), (1.0 - value).ln()]
        } else {
            vec![f64::NEG_INFINITY, f64::NEG_INFINITY]
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
        let beta = natural_params[1] + 1.0;
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Beta log_partition: α = {} and β = {} must both be positive",
                alpha, beta
            )));
        }
        // A(η) = ln Γ(α) + ln Γ(β) − ln Γ(α + β).
        Ok(ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta))
    }

    fn expected_sufficient_statistics(&self) -> Vec<f64> {
        // E[u(x)] = (E[log x], E[log(1-x)]) = (ψ(α) − ψ(α+β), ψ(β) − ψ(α+β)).
        vec![self.expected_log_x(), self.expected_log_1mx()]
    }
}

/// Beta-Bernoulli conjugate posterior update.
///
/// Given a `Beta(α_prior, β_prior)` prior on the Bernoulli success probability
/// `p` and observed `successes` successes plus `failures` failures, the exact
/// posterior is `Beta(α_prior + successes, β_prior + failures)`.
///
/// This is exact because Bernoulli is conjugate to Beta; the update adds the
/// observation-dependent sufficient statistics `(n_s, n_f)` into the natural
/// parameters `(α − 1, β − 1)` of the prior.
pub fn posterior_from_prior_and_observations(
    prior: &BetaNP,
    successes: u64,
    failures: u64,
) -> Result<BetaNP> {
    let posterior_alpha = prior.alpha + successes as f64;
    let posterior_beta = prior.beta + failures as f64;
    BetaNP::new(posterior_alpha, posterior_beta)
}

/// `BetaBernoulliObservation` captures a Bernoulli likelihood `y ~ Bernoulli(p)`
/// where the success probability `p` is a `BetaNP` variable. It contributes
/// `(n_s, n_f)` to the posterior natural parameters, i.e. adds `n_s` to
/// `(α − 1)` and `n_f` to `(β − 1)`.
///
/// A factor holds a reference to its Beta-distributed probability variable and
/// a (possibly empty) batch of binary outcomes. Posterior inference combining
/// prior + factor is exact in one VMP sweep because Bernoulli is conjugate to
/// Beta.
#[derive(Clone, Debug)]
pub struct BetaBernoulliObservation {
    /// Name of the `BetaNP` variable in the VMP graph.
    pub probability_variable: String,
    /// Observed binary outcomes (true = success, false = failure).
    pub observations: Vec<bool>,
}

impl BetaBernoulliObservation {
    /// Build a new Beta-Bernoulli observation factor from a boolean batch.
    pub fn new(probability_variable: impl Into<String>, observations: Vec<bool>) -> Self {
        Self {
            probability_variable: probability_variable.into(),
            observations,
        }
    }

    /// Convenience constructor from aggregate counts. Often you already have
    /// the sufficient statistics as `(n_s, n_f)` without keeping the raw batch.
    pub fn from_counts(
        probability_variable: impl Into<String>,
        successes: u64,
        failures: u64,
    ) -> Self {
        let mut observations = Vec::with_capacity((successes + failures) as usize);
        observations.extend(std::iter::repeat_n(true, successes as usize));
        observations.extend(std::iter::repeat_n(false, failures as usize));
        Self {
            probability_variable: probability_variable.into(),
            observations,
        }
    }

    /// Number of successes n_s = Σ 1[y_i = 1]. Used as the α-parameter increment.
    pub fn num_successes(&self) -> u64 {
        self.observations.iter().filter(|b| **b).count() as u64
    }

    /// Number of failures n_f = Σ 1[y_i = 0]. Used as the β-parameter increment.
    pub fn num_failures(&self) -> u64 {
        self.observations.iter().filter(|b| !**b).count() as u64
    }

    /// Total number of observations N = n_s + n_f.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmp::special::{digamma, ln_gamma};

    #[test]
    fn beta_expected_x_matches_alpha_over_total() {
        for &(alpha, beta) in &[(1.0_f64, 1.0_f64), (2.0, 3.0), (4.5, 0.5), (0.25, 10.0)] {
            let b = BetaNP::new(alpha, beta).expect("ctor");
            let ex = b.expected_x();
            let expected = alpha / (alpha + beta);
            assert!(
                (ex - expected).abs() < 1e-12,
                "E[x] = {} but α/(α+β) = {}",
                ex,
                expected
            );
        }
    }

    #[test]
    fn beta_expected_log_x_and_1mx_match_digamma() {
        for &(alpha, beta) in &[(1.0_f64, 1.0_f64), (2.5, 1.5), (4.0, 2.0)] {
            let b = BetaNP::new(alpha, beta).expect("ctor");
            let el_x = b.expected_log_x();
            let el_1mx = b.expected_log_1mx();
            let expected_log_x = digamma(alpha) - digamma(alpha + beta);
            let expected_log_1mx = digamma(beta) - digamma(alpha + beta);
            assert!((el_x - expected_log_x).abs() < 1e-12);
            assert!((el_1mx - expected_log_1mx).abs() < 1e-12);
        }
    }

    #[test]
    fn beta_multiply_naturals_sums_natural_params() {
        // Beta(2, 3) has η = (1, 2); Beta(3, 5) has η = (2, 4).
        // Sum = (3, 6), i.e. Beta(4, 7).
        let a = BetaNP::new(2.0, 3.0).expect("ctor a");
        let c = BetaNP::new(3.0, 5.0).expect("ctor b");
        let p = a.multiply_naturals(&c).expect("product");
        assert!((p.alpha - 4.0).abs() < 1e-12, "α = {}", p.alpha);
        assert!((p.beta - 7.0).abs() < 1e-12, "β = {}", p.beta);
        // Round-trip through natural parameters.
        let eta_a = a.natural_params();
        let eta_c = c.natural_params();
        let eta_sum: Vec<f64> = eta_a.iter().zip(eta_c.iter()).map(|(x, y)| x + y).collect();
        let p2 = BetaNP::from_natural(&eta_sum).expect("from nat");
        assert!((p2.alpha - p.alpha).abs() < 1e-12);
        assert!((p2.beta - p.beta).abs() < 1e-12);
    }

    #[test]
    fn beta_kl_is_zero_for_self_positive_otherwise() {
        let b = BetaNP::new(3.0, 2.0).expect("ctor");
        let self_kl = b.kl_to(&b);
        assert!(self_kl.abs() < 1e-10, "KL(b||b) = {}", self_kl);

        let other = BetaNP::new(1.5, 4.0).expect("ctor other");
        let kl = b.kl_to(&other);
        assert!(kl > 0.0, "KL(b||other) should be positive, got {}", kl);

        let kl_rev = other.kl_to(&b);
        assert!(
            kl_rev > 0.0,
            "KL(other||b) should be positive, got {}",
            kl_rev
        );
    }

    #[test]
    fn beta_bernoulli_posterior_adds_counts() {
        // Beta(1, 1) + 7 successes, 3 failures → Beta(8, 4).
        let prior = BetaNP::new(1.0, 1.0).expect("prior");
        let post = posterior_from_prior_and_observations(&prior, 7, 3).expect("posterior");
        assert!((post.alpha - 8.0).abs() < 1e-12, "α = {}", post.alpha);
        assert!((post.beta - 4.0).abs() < 1e-12, "β = {}", post.beta);
    }

    #[test]
    fn beta_log_partition_matches_closed_form() {
        // A(η) = ln Γ(α) + ln Γ(β) − ln Γ(α + β).
        let b = BetaNP::new(2.5, 3.0).expect("ctor");
        let eta = b.natural_params();
        let a = b.log_partition(&eta).expect("lp");
        let expected = ln_gamma(2.5) + ln_gamma(3.0) - ln_gamma(5.5);
        assert!(
            (a - expected).abs() < 1e-12,
            "A(η) = {}, expected {}",
            a,
            expected
        );

        // ∂A/∂η₁ = ψ(α) − ψ(α+β) = E[log x] and ∂A/∂η₂ = ψ(β) − ψ(α+β) = E[log(1-x)].
        let h = 1e-6;
        let a_plus_1 = b.log_partition(&[eta[0] + h, eta[1]]).expect("lp+1");
        let a_minus_1 = b.log_partition(&[eta[0] - h, eta[1]]).expect("lp-1");
        let d1 = (a_plus_1 - a_minus_1) / (2.0 * h);
        let a_plus_2 = b.log_partition(&[eta[0], eta[1] + h]).expect("lp+2");
        let a_minus_2 = b.log_partition(&[eta[0], eta[1] - h]).expect("lp-2");
        let d2 = (a_plus_2 - a_minus_2) / (2.0 * h);
        assert!(
            (d1 - b.expected_log_x()).abs() < 1e-5,
            "dA/dη1 = {}, expected {}",
            d1,
            b.expected_log_x()
        );
        assert!(
            (d2 - b.expected_log_1mx()).abs() < 1e-5,
            "dA/dη2 = {}, expected {}",
            d2,
            b.expected_log_1mx()
        );
    }

    #[test]
    fn beta_natural_round_trip() {
        let b = BetaNP::new(4.5, 2.25).expect("ctor");
        let eta = b.natural_params();
        let back = BetaNP::from_natural(&eta).expect("round trip");
        assert!((back.alpha - 4.5).abs() < 1e-12);
        assert!((back.beta - 2.25).abs() < 1e-12);
    }

    #[test]
    fn beta_set_natural_rejects_invalid_shapes() {
        let mut b = BetaNP::new(2.0, 2.0).expect("ctor");
        // α = 1 + (-1.5) = -0.5 < 0
        let err = b.set_natural(&[-1.5, 0.0]);
        assert!(err.is_err());
        // β = 1 + (-2.0) = -1.0 < 0
        let err = b.set_natural(&[0.0, -2.0]);
        assert!(err.is_err());
        // NaN
        let err = b.set_natural(&[f64::NAN, 0.0]);
        assert!(err.is_err());
        // Wrong length
        let err = b.set_natural(&[0.1]);
        assert!(err.is_err());
        // Valid.
        let ok = b.set_natural(&[0.5, 1.5]);
        assert!(ok.is_ok());
        assert!((b.alpha - 1.5).abs() < 1e-12);
        assert!((b.beta - 2.5).abs() < 1e-12);
    }

    #[test]
    fn beta_bernoulli_observation_counts() {
        let obs = BetaBernoulliObservation::new("p", vec![true, false, true, true, false, true]);
        assert_eq!(obs.num_successes(), 4);
        assert_eq!(obs.num_failures(), 2);
        assert_eq!(obs.num_observations(), 6);

        let from_counts = BetaBernoulliObservation::from_counts("p", 5, 3);
        assert_eq!(from_counts.num_successes(), 5);
        assert_eq!(from_counts.num_failures(), 3);
        assert_eq!(from_counts.num_observations(), 8);
    }
}
