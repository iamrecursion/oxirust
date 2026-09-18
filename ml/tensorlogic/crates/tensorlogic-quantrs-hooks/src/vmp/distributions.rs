//! Concrete `ExponentialFamily` implementations shipped with the v0.2.0 VMP
//! research preview.
//!
//! Three families are provided, each in natural-parameter form:
//!
//! | Family            | Support      | Natural parameters η                  | Sufficient stats u(x) |
//! |-------------------|--------------|---------------------------------------|-----------------------|
//! | Gaussian (fixed τ)| x ∈ ℝ        | η = τμ (scalar — precision `τ` fixed) | \[x\]                   |
//! | Categorical       | k ∈ {0..K-1} | ηₖ = log πₖ (unnormalised)            | one-hot vector        |
//! | Dirichlet         | π ∈ Δ^(K-1)  | ηₖ = αₖ − 1                            | \[log π₁, …, log π_K\]  |
//!
//! The Gaussian case is the *mean-unknown, precision-known* restriction that
//! v0.2.0 targets. Generalising to unknown precision would add a second natural
//! parameter (η₂ = −τ/2) and make the conjugate family Gamma-Normal; that is out
//! of scope for the preview.

use crate::error::{PgmError, Result};

use super::exponential_family::ExponentialFamily;
use super::special::{digamma, ln_gamma};

// ---------------------------------------------------------------------------
// Gaussian (mean unknown, precision known)
// ---------------------------------------------------------------------------

/// Univariate Gaussian with **known precision** τ and unknown mean μ.
///
/// Only the mean is a random variable in VMP; τ is a model constant. The natural
/// parameter is therefore one-dimensional: η = τμ. We still carry `precision`
/// alongside η so that `update_natural` can recover μ without re-deriving it
/// from the conjugate-update context.
#[derive(Clone, Debug)]
pub struct GaussianNP {
    /// Mean μ.
    pub mean: f64,
    /// Precision τ = 1 / σ² (fixed).
    pub precision: f64,
}

impl GaussianNP {
    /// Construct from moment parameters (μ, τ).
    pub fn new(mean: f64, precision: f64) -> Result<Self> {
        if !precision.is_finite() || precision <= 0.0 {
            return Err(PgmError::InvalidDistribution(format!(
                "Gaussian precision must be positive and finite (got {})",
                precision
            )));
        }
        if !mean.is_finite() {
            return Err(PgmError::InvalidDistribution(format!(
                "Gaussian mean must be finite (got {})",
                mean
            )));
        }
        Ok(Self { mean, precision })
    }

    /// Reconstruct a Gaussian from natural parameters with a known precision.
    ///
    /// For the mean-unknown / precision-known case, η = τμ ⇒ μ = η / τ.
    pub fn from_natural(natural: &[f64], precision: f64) -> Result<Self> {
        if natural.len() != 1 {
            return Err(PgmError::DimensionMismatch {
                expected: vec![1],
                got: vec![natural.len()],
            });
        }
        Self::new(natural[0] / precision, precision)
    }

    /// Variance σ² = 1 / τ.
    pub fn variance(&self) -> f64 {
        1.0 / self.precision
    }
}

impl ExponentialFamily for GaussianNP {
    fn family_name(&self) -> &'static str {
        "Gaussian"
    }

    fn natural_dim(&self) -> usize {
        1
    }

    fn natural_params(&self) -> Vec<f64> {
        vec![self.precision * self.mean]
    }

    fn set_natural(&mut self, new_eta: &[f64]) -> Result<()> {
        if new_eta.len() != 1 {
            return Err(PgmError::DimensionMismatch {
                expected: vec![1],
                got: vec![new_eta.len()],
            });
        }
        if !new_eta[0].is_finite() {
            return Err(PgmError::InvalidDistribution(
                "Gaussian natural parameter must be finite".to_string(),
            ));
        }
        self.mean = new_eta[0] / self.precision;
        Ok(())
    }

    fn sufficient_statistics(&self, value: f64) -> Vec<f64> {
        vec![value]
    }

    fn log_partition(&self, natural_params: &[f64]) -> Result<f64> {
        if natural_params.len() != 1 {
            return Err(PgmError::DimensionMismatch {
                expected: vec![1],
                got: vec![natural_params.len()],
            });
        }
        // A(η) = η² / (2τ) with the fixed-precision constant half-log term:
        //        A_full(η) = η² / (2τ) + ½ log(2π / τ).
        let eta = natural_params[0];
        let quad = 0.5 * eta * eta / self.precision;
        let log_norm = 0.5 * (2.0 * std::f64::consts::PI / self.precision).ln();
        Ok(quad + log_norm)
    }

    fn expected_sufficient_statistics(&self) -> Vec<f64> {
        // ∇_η A(η) = η / τ = μ.
        vec![self.mean]
    }

    fn entropy(&self) -> Result<f64> {
        // H = ½ log(2 π e / τ).
        Ok(0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E / self.precision).ln())
    }
}

// ---------------------------------------------------------------------------
// Categorical
// ---------------------------------------------------------------------------

/// Categorical distribution stored as log-probabilities (natural parameters).
///
/// The natural parameters `η = log π` are unnormalised: any constant shift
/// represents the same distribution. We therefore re-centre on every update by
/// subtracting the log-sum-exp — this keeps arithmetic numerically stable without
/// changing semantics.
#[derive(Clone, Debug)]
pub struct CategoricalNP {
    /// Unnormalised log-probabilities.
    pub log_probs: Vec<f64>,
}

impl CategoricalNP {
    /// Construct from probabilities. Validates positivity and normalisation.
    pub fn from_probs(probs: &[f64]) -> Result<Self> {
        if probs.is_empty() {
            return Err(PgmError::InvalidDistribution(
                "Categorical needs at least one category".to_string(),
            ));
        }
        let sum: f64 = probs.iter().sum();
        if !(sum.is_finite()) || sum <= 0.0 {
            return Err(PgmError::InvalidDistribution(
                "Categorical probabilities must be positive and sum to a positive finite value"
                    .to_string(),
            ));
        }
        for &p in probs {
            if !p.is_finite() || p < 0.0 {
                return Err(PgmError::InvalidDistribution(format!(
                    "Categorical probability must be non-negative and finite (got {})",
                    p
                )));
            }
        }
        let log_probs: Vec<f64> = probs
            .iter()
            .map(|&p| if p > 0.0 { (p / sum).ln() } else { -1e12 })
            .collect();
        Ok(Self { log_probs })
    }

    /// Build directly from natural parameters (log-probs, unnormalised).
    pub fn from_natural(natural: &[f64]) -> Result<Self> {
        if natural.is_empty() {
            return Err(PgmError::InvalidDistribution(
                "Categorical needs at least one category".to_string(),
            ));
        }
        for &v in natural {
            if !v.is_finite() {
                return Err(PgmError::InvalidDistribution(
                    "Categorical natural parameter must be finite".to_string(),
                ));
            }
        }
        let mut out = Self {
            log_probs: natural.to_vec(),
        };
        out.renormalise_log_probs();
        Ok(out)
    }

    /// Number of categories K.
    pub fn num_categories(&self) -> usize {
        self.log_probs.len()
    }

    /// Normalised probabilities π = softmax(η).
    pub fn probs(&self) -> Vec<f64> {
        let lse = log_sum_exp(&self.log_probs);
        self.log_probs.iter().map(|&l| (l - lse).exp()).collect()
    }

    /// Shift log-probs so their log-sum-exp is zero (pure cosmetic
    /// normalisation — natural parameters are invariant under a constant shift).
    fn renormalise_log_probs(&mut self) {
        let lse = log_sum_exp(&self.log_probs);
        if lse.is_finite() {
            for v in &mut self.log_probs {
                *v -= lse;
            }
        }
    }
}

impl ExponentialFamily for CategoricalNP {
    fn family_name(&self) -> &'static str {
        "Categorical"
    }

    fn natural_dim(&self) -> usize {
        self.log_probs.len()
    }

    fn natural_params(&self) -> Vec<f64> {
        self.log_probs.clone()
    }

    fn set_natural(&mut self, new_eta: &[f64]) -> Result<()> {
        if new_eta.len() != self.log_probs.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.log_probs.len()],
                got: vec![new_eta.len()],
            });
        }
        for &v in new_eta {
            if !v.is_finite() {
                return Err(PgmError::InvalidDistribution(
                    "Categorical natural parameter must be finite".to_string(),
                ));
            }
        }
        self.log_probs.copy_from_slice(new_eta);
        self.renormalise_log_probs();
        Ok(())
    }

    fn sufficient_statistics(&self, value: f64) -> Vec<f64> {
        // One-hot indicator of the category index. Robust to negative/NaN inputs.
        let k = self.log_probs.len();
        let mut out = vec![0.0; k];
        if value.is_finite() && value >= 0.0 {
            let idx = value.floor() as usize;
            if idx < k {
                out[idx] = 1.0;
            }
        }
        out
    }

    fn log_partition(&self, natural_params: &[f64]) -> Result<f64> {
        if natural_params.len() != self.log_probs.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.log_probs.len()],
                got: vec![natural_params.len()],
            });
        }
        Ok(log_sum_exp(natural_params))
    }

    fn expected_sufficient_statistics(&self) -> Vec<f64> {
        // E_q[u(x)] = softmax(η).
        self.probs()
    }
}

// ---------------------------------------------------------------------------
// Dirichlet
// ---------------------------------------------------------------------------

/// Dirichlet distribution, the conjugate prior for the Categorical.
///
/// Natural parameters are `ηₖ = αₖ − 1`. All updates happen in η-space but we
/// expose the concentration vector α for ergonomics.
#[derive(Clone, Debug)]
pub struct DirichletNP {
    /// Concentration parameters α (all > 0).
    pub concentration: Vec<f64>,
}

impl DirichletNP {
    /// Construct from concentration parameters α.
    pub fn new(concentration: Vec<f64>) -> Result<Self> {
        if concentration.is_empty() {
            return Err(PgmError::InvalidDistribution(
                "Dirichlet needs at least one component".to_string(),
            ));
        }
        for &a in &concentration {
            if !a.is_finite() || a <= 0.0 {
                return Err(PgmError::InvalidDistribution(format!(
                    "Dirichlet concentration must be positive and finite (got {})",
                    a
                )));
            }
        }
        Ok(Self { concentration })
    }

    /// Build from natural parameters η = α − 1. Validates positivity of α.
    pub fn from_natural(natural: &[f64]) -> Result<Self> {
        let alpha: Vec<f64> = natural.iter().map(|&e| e + 1.0).collect();
        Self::new(alpha)
    }

    /// Number of Dirichlet components K.
    pub fn num_components(&self) -> usize {
        self.concentration.len()
    }

    /// Sum of concentration parameters `α₀ = Σαₖ`.
    pub fn total_concentration(&self) -> f64 {
        self.concentration.iter().sum()
    }
}

impl ExponentialFamily for DirichletNP {
    fn family_name(&self) -> &'static str {
        "Dirichlet"
    }

    fn natural_dim(&self) -> usize {
        self.concentration.len()
    }

    fn natural_params(&self) -> Vec<f64> {
        self.concentration.iter().map(|a| a - 1.0).collect()
    }

    fn set_natural(&mut self, new_eta: &[f64]) -> Result<()> {
        if new_eta.len() != self.concentration.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.concentration.len()],
                got: vec![new_eta.len()],
            });
        }
        for &v in new_eta {
            if !v.is_finite() {
                return Err(PgmError::InvalidDistribution(
                    "Dirichlet natural parameter must be finite".to_string(),
                ));
            }
            if v + 1.0 <= 0.0 {
                return Err(PgmError::InvalidDistribution(format!(
                    "Dirichlet concentration must stay positive (η + 1 = {} ≤ 0)",
                    v + 1.0
                )));
            }
        }
        for (a, e) in self.concentration.iter_mut().zip(new_eta.iter()) {
            *a = e + 1.0;
        }
        Ok(())
    }

    fn sufficient_statistics(&self, _value: f64) -> Vec<f64> {
        // Sufficient statistics of the Dirichlet over a K-vector π are
        // u(π) = [log π₁, …, log π_K]. A single scalar `value` is insufficient,
        // so in the VMP setting we only consume `expected_sufficient_statistics`.
        // We still return a best-effort degenerate zero vector here to preserve
        // the uniform trait signature.
        vec![0.0; self.concentration.len()]
    }

    fn log_partition(&self, natural_params: &[f64]) -> Result<f64> {
        if natural_params.len() != self.concentration.len() {
            return Err(PgmError::DimensionMismatch {
                expected: vec![self.concentration.len()],
                got: vec![natural_params.len()],
            });
        }
        // A(η) = Σ_k ln Γ(η_k + 1) − ln Γ(Σ_k (η_k + 1)).
        let alpha: Vec<f64> = natural_params.iter().map(|e| e + 1.0).collect();
        let sum: f64 = alpha.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(PgmError::InvalidDistribution(
                "Dirichlet sum of concentrations must be positive".to_string(),
            ));
        }
        let lgamma_sum = ln_gamma(sum);
        let lgamma_components: f64 = alpha.iter().map(|&a| ln_gamma(a)).sum();
        Ok(lgamma_components - lgamma_sum)
    }

    fn expected_sufficient_statistics(&self) -> Vec<f64> {
        // E_q[log π_k] = ψ(α_k) − ψ(Σ_j α_j).
        let sum = self.total_concentration();
        let psi_sum = digamma(sum);
        self.concentration
            .iter()
            .map(|&a| digamma(a) - psi_sum)
            .collect()
    }
}

/// KL divergence between two Dirichlets `KL(Dir(α) || Dir(β))`.
///
/// Closed form: `ln Γ(α₀) − Σ ln Γ(αₖ) − ln Γ(β₀) + Σ ln Γ(βₖ)
///               + Σ (αₖ − βₖ)(ψ(αₖ) − ψ(α₀))`.
pub fn dirichlet_kl(alpha: &DirichletNP, beta: &DirichletNP) -> Result<f64> {
    if alpha.concentration.len() != beta.concentration.len() {
        return Err(PgmError::DimensionMismatch {
            expected: vec![alpha.concentration.len()],
            got: vec![beta.concentration.len()],
        });
    }
    let a0: f64 = alpha.concentration.iter().sum();
    let b0: f64 = beta.concentration.iter().sum();
    let psi_a0 = digamma(a0);
    let mut kl = ln_gamma(a0) - ln_gamma(b0);
    for (&a, &b) in alpha.concentration.iter().zip(beta.concentration.iter()) {
        kl += ln_gamma(b) - ln_gamma(a);
        kl += (a - b) * (digamma(a) - psi_a0);
    }
    Ok(kl)
}

/// KL divergence between two Gaussians with **the same known precision**
/// (the case that actually arises at the *start* of the mean-unknown VMP setting):
/// `KL(N(μ_q, 1/τ) || N(μ_p, 1/τ)) = τ (μ_q − μ_p)² / 2`.
pub fn gaussian_kl_fixed_precision(q: &GaussianNP, p: &GaussianNP) -> Result<f64> {
    if (q.precision - p.precision).abs() > 1e-12 {
        return Err(PgmError::InvalidDistribution(format!(
            "gaussian_kl_fixed_precision requires equal precisions (got {}, {})",
            q.precision, p.precision
        )));
    }
    let dm = q.mean - p.mean;
    Ok(0.5 * q.precision * dm * dm)
}

/// General univariate Gaussian KL `KL(N(μ_q, 1/τ_q) || N(μ_p, 1/τ_p))`.
///
/// Needed because VMP *changes* the posterior's effective precision (τ_q grows
/// as factors contribute observation precisions) while the prior's precision
/// stays fixed. The closed form is:
///
/// ```text
///   KL = ½ [ ln(τ_q / τ_p) − 1 + τ_p / τ_q + τ_p · (μ_q − μ_p)² ]
/// ```
pub fn gaussian_kl(q: &GaussianNP, p: &GaussianNP) -> Result<f64> {
    if !q.precision.is_finite() || q.precision <= 0.0 {
        return Err(PgmError::InvalidDistribution(format!(
            "gaussian_kl: q.precision must be positive (got {})",
            q.precision
        )));
    }
    if !p.precision.is_finite() || p.precision <= 0.0 {
        return Err(PgmError::InvalidDistribution(format!(
            "gaussian_kl: p.precision must be positive (got {})",
            p.precision
        )));
    }
    let dm = q.mean - p.mean;
    Ok(0.5
        * ((q.precision / p.precision).ln() - 1.0
            + p.precision / q.precision
            + p.precision * dm * dm))
}

/// KL divergence between two Categoricals `KL(q || p) = Σ π_q log(π_q / π_p)`.
pub fn categorical_kl(q: &CategoricalNP, p: &CategoricalNP) -> Result<f64> {
    if q.log_probs.len() != p.log_probs.len() {
        return Err(PgmError::DimensionMismatch {
            expected: vec![q.log_probs.len()],
            got: vec![p.log_probs.len()],
        });
    }
    let q_probs = q.probs();
    let mut kl = 0.0;
    for (i, &pi_q) in q_probs.iter().enumerate() {
        if pi_q <= 0.0 {
            continue;
        }
        let log_pi_q = pi_q.ln();
        let lse_p = log_sum_exp(&p.log_probs);
        let log_pi_p = p.log_probs[i] - lse_p;
        kl += pi_q * (log_pi_q - log_pi_p);
    }
    Ok(kl)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_natural_round_trip() {
        let g = GaussianNP::new(1.5, 2.0).expect("ctor");
        let eta = g.to_natural();
        assert_eq!(eta.len(), 1);
        let back = GaussianNP::from_natural(&eta, 2.0).expect("from nat");
        assert!((back.mean - 1.5).abs() < 1e-12);
        assert!((back.precision - 2.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_expected_sufficient_is_mean() {
        let g = GaussianNP::new(-0.25, 4.0).expect("ctor");
        let ess = g.expected_sufficient_statistics();
        assert!((ess[0] - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn categorical_natural_round_trip_normalises() {
        let c = CategoricalNP::from_probs(&[0.1, 0.2, 0.7]).expect("ctor");
        let probs = c.probs();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // round trip via natural parameters
        let nat = c.natural_params();
        let c2 = CategoricalNP::from_natural(&nat).expect("from nat");
        for (a, b) in probs.iter().zip(c2.probs().iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn categorical_sufficient_stats_one_hot() {
        let c = CategoricalNP::from_probs(&[0.25, 0.25, 0.25, 0.25]).expect("ctor");
        let u = c.sufficient_statistics(2.0);
        assert_eq!(u, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn dirichlet_requires_positive_concentration() {
        let err = DirichletNP::new(vec![0.0, 1.0]);
        assert!(err.is_err());
        let err = DirichletNP::new(vec![-0.1, 1.0]);
        assert!(err.is_err());
        let err = DirichletNP::new(vec![f64::NAN, 1.0]);
        assert!(err.is_err());
        let ok = DirichletNP::new(vec![0.5, 0.5]);
        assert!(ok.is_ok());
    }

    #[test]
    fn dirichlet_expected_log_pi_sums_correctly() {
        let d = DirichletNP::new(vec![2.0, 3.0, 5.0]).expect("ctor");
        let ess = d.expected_sufficient_statistics();
        // Each component should be < 0 since E[log π] is negative.
        for &v in &ess {
            assert!(v < 0.0);
        }
        // Identity: Σ E[log π_k] = Σ ψ(α_k) − K ψ(α₀)
        let manual: f64 =
            [2.0f64, 3.0, 5.0].iter().map(|&a| digamma(a)).sum::<f64>() - 3.0 * digamma(10.0);
        let sum: f64 = ess.iter().sum();
        assert!((sum - manual).abs() < 1e-10);
    }

    #[test]
    fn dirichlet_kl_self_is_zero() {
        let d = DirichletNP::new(vec![1.0, 2.0, 3.0]).expect("ctor");
        let kl = dirichlet_kl(&d, &d).expect("kl");
        assert!(kl.abs() < 1e-10, "kl = {}", kl);
    }

    #[test]
    fn categorical_kl_self_is_zero() {
        let c = CategoricalNP::from_probs(&[0.2, 0.5, 0.3]).expect("ctor");
        let kl = categorical_kl(&c, &c).expect("kl");
        assert!(kl.abs() < 1e-10, "kl = {}", kl);
    }

    #[test]
    fn gaussian_log_partition_identity() {
        // A(η) must be a real number and its derivative match E[u(x)] = μ.
        let g = GaussianNP::new(0.7, 1.3).expect("ctor");
        let eta = g.to_natural();
        let a = g.log_partition(&eta).expect("lp");
        assert!(a.is_finite());
        // Numerical gradient via central difference against ESS[0] = μ.
        let h = 1e-5;
        let plus = g.log_partition(&[eta[0] + h]).expect("lp+");
        let minus = g.log_partition(&[eta[0] - h]).expect("lp-");
        let grad = (plus - minus) / (2.0 * h);
        assert!((grad - g.mean).abs() < 1e-6);
    }
}
