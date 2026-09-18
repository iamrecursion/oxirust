//! Uncertainty quantification over ensembles and sample collections.
//!
//! This module provides tools to characterise the uncertainty in predictions
//! from stochastic models, Monte Carlo Dropout, or deep ensembles:
//!
//! - **`MonteCarloEstimator`**: Computes mean, variance, std-dev and empirical
//!   credible intervals (percentile method) over a collection of tensors.
//! - **`predictive_entropy`**: Shannon entropy of a probability vector.
//! - **`bald_epistemic_uncertainty`**: Bayesian Active Learning by Disagreement —
//!   separates epistemic from aleatoric uncertainty in ensemble predictions.

use crate::error::{TlBackendError, TlBackendResult};
use crate::Scirs2Tensor;
use scirs2_core::ndarray::{ArrayD, Axis, IxDyn};

// ============================================================================
// UncertaintyEstimate
// ============================================================================

/// Full uncertainty breakdown for a collection of tensor samples.
#[derive(Debug, Clone)]
pub struct UncertaintyEstimate {
    /// Per-element posterior mean.
    pub mean: Scirs2Tensor,
    /// Per-element posterior variance.
    pub variance: Scirs2Tensor,
    /// Per-element posterior standard deviation.
    pub std_dev: Scirs2Tensor,
    /// Lower bound of the credible interval (percentile method).
    pub lower_ci: Scirs2Tensor,
    /// Upper bound of the credible interval (percentile method).
    pub upper_ci: Scirs2Tensor,
    /// Confidence level in (0, 1), e.g. 0.95 for 95% CI.
    pub confidence_level: f64,
}

// ============================================================================
// MonteCarloEstimator
// ============================================================================

/// Estimator that summarises a slice of same-shaped tensors as a posterior.
#[derive(Debug, Clone)]
pub struct MonteCarloEstimator {
    /// Confidence level for the credible interval, e.g. 0.95.
    pub confidence_level: f64,
}

impl Default for MonteCarloEstimator {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
        }
    }
}

impl MonteCarloEstimator {
    /// Estimate mean, variance, std-dev, and empirical CI over `samples`.
    ///
    /// Uses a two-pass algorithm for numerical stability:
    /// 1. First pass: compute per-element mean.
    /// 2. Second pass: accumulate squared deviations for variance.
    ///
    /// CI bounds use the percentile method: for each element position, collect
    /// the `n_samples` scalar values, sort them, then return the values at the
    /// `α/2` and `1 – α/2` quantile indices.
    ///
    /// # Errors
    /// - Empty `samples` slice.
    /// - Samples with inconsistent shapes.
    pub fn estimate(&self, samples: &[Scirs2Tensor]) -> TlBackendResult<UncertaintyEstimate> {
        if samples.is_empty() {
            return Err(TlBackendError::InvalidOperation(
                "MonteCarloEstimator::estimate: samples slice must not be empty".to_string(),
            ));
        }

        let ref_shape = samples[0].shape().to_vec();
        for (i, s) in samples.iter().enumerate().skip(1) {
            if s.shape() != ref_shape.as_slice() {
                return Err(TlBackendError::InvalidOperation(format!(
                    "MonteCarloEstimator::estimate: sample {i} has shape {:?}, expected {:?}",
                    s.shape(),
                    ref_shape
                )));
            }
        }

        let n = samples.len() as f64;
        let n_elems = ref_shape.iter().product::<usize>();

        // ---- Pass 1: per-element mean ----------------------------------------
        let mut mean_data = vec![0.0_f64; n_elems];
        for sample in samples.iter() {
            for (acc, &v) in mean_data.iter_mut().zip(sample.iter()) {
                *acc += v;
            }
        }
        for acc in mean_data.iter_mut() {
            *acc /= n;
        }

        // ---- Pass 2: per-element variance (population) -----------------------
        let mut var_data = vec![0.0_f64; n_elems];
        for sample in samples.iter() {
            for ((acc, &v), &m) in var_data.iter_mut().zip(sample.iter()).zip(mean_data.iter()) {
                *acc += (v - m).powi(2);
            }
        }
        for acc in var_data.iter_mut() {
            *acc /= n;
        }

        let std_data: Vec<f64> = var_data.iter().map(|&v| v.sqrt()).collect();

        // ---- Percentile CI ---------------------------------------------------
        let alpha = 1.0 - self.confidence_level;
        let lo_quantile = alpha / 2.0;
        let hi_quantile = 1.0 - alpha / 2.0;

        let mut lower_data = vec![0.0_f64; n_elems];
        let mut upper_data = vec![0.0_f64; n_elems];

        // For each element position, collect all sample values, sort, interpolate.
        let mut elem_values: Vec<f64> = Vec::with_capacity(samples.len());
        for elem_idx in 0..n_elems {
            elem_values.clear();
            for sample in samples.iter() {
                elem_values.push(sample.iter().nth(elem_idx).copied().unwrap_or(f64::NAN));
            }
            elem_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            lower_data[elem_idx] = quantile_sorted(&elem_values, lo_quantile);
            upper_data[elem_idx] = quantile_sorted(&elem_values, hi_quantile);
        }

        let dyn_shape = IxDyn(&ref_shape);
        let mean = ArrayD::from_shape_vec(dyn_shape.clone(), mean_data)
            .map_err(|e| TlBackendError::Internal(e.to_string()))?;
        let variance = ArrayD::from_shape_vec(dyn_shape.clone(), var_data)
            .map_err(|e| TlBackendError::Internal(e.to_string()))?;
        let std_dev = ArrayD::from_shape_vec(dyn_shape.clone(), std_data)
            .map_err(|e| TlBackendError::Internal(e.to_string()))?;
        let lower_ci = ArrayD::from_shape_vec(dyn_shape.clone(), lower_data)
            .map_err(|e| TlBackendError::Internal(e.to_string()))?;
        let upper_ci = ArrayD::from_shape_vec(dyn_shape, upper_data)
            .map_err(|e| TlBackendError::Internal(e.to_string()))?;

        Ok(UncertaintyEstimate {
            mean,
            variance,
            std_dev,
            lower_ci,
            upper_ci,
            confidence_level: self.confidence_level,
        })
    }
}

// ============================================================================
// Quantile helper
// ============================================================================

/// Linear interpolation quantile on a pre-sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 1.0);
    let virtual_idx = q * (n - 1) as f64;
    let lo = virtual_idx.floor() as usize;
    let hi = virtual_idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = virtual_idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ============================================================================
// Predictive entropy
// ============================================================================

/// Compute the predictive entropy H\[p\] = -Σ_k p_k · log(p_k + ε) along `axis`.
///
/// The numerical floor ε = 1e-10 prevents log(0).  The caller is expected to
/// supply a tensor whose entries along `axis` sum to 1 (i.e. a probability
/// simplex), though this is not enforced.
///
/// The returned tensor has the same shape as `probs` with `axis` summed away.
///
/// # Errors
/// Returns an error if `axis` is out of range.
pub fn predictive_entropy(probs: &Scirs2Tensor, axis: usize) -> TlBackendResult<Scirs2Tensor> {
    let ndim = probs.ndim();
    if ndim == 0 {
        return Err(TlBackendError::InvalidOperation(
            "predictive_entropy: probs must have at least one dimension".to_string(),
        ));
    }
    if axis >= ndim {
        return Err(TlBackendError::InvalidOperation(format!(
            "predictive_entropy: axis {axis} out of range for {ndim}-D tensor"
        )));
    }

    const EPS: f64 = 1e-10;
    let term = probs.mapv(|p| {
        let p_safe = (p + EPS).max(EPS);
        -p * p_safe.ln()
    });
    Ok(term.sum_axis(Axis(axis)))
}

// ============================================================================
// BALD epistemic uncertainty
// ============================================================================

/// Compute the epistemic uncertainty via BALD (Bayesian Active Learning by Disagreement).
///
/// BALD = H(mean_probs) − mean_k(H(probs_k))
///
/// where:
/// - `probs_k` is the k-th ensemble member's probability tensor,
/// - `mean_probs` is the element-wise average over ensemble members,
/// - `H(·)` is [`predictive_entropy`] summed along `axis`.
///
/// Intuitively: the entropy of the *average* (total uncertainty) minus the
/// average *individual* entropy (aleatoric uncertainty) = epistemic uncertainty.
///
/// # Arguments
/// * `ensemble` — slice of probability tensors, all same shape; at least 1 required
/// * `axis`     — class/probability axis along which entropy is computed
///
/// # Errors
/// Returns an error if `ensemble` is empty, shapes differ, or `axis` is invalid.
pub fn bald_epistemic_uncertainty(
    ensemble: &[Scirs2Tensor],
    axis: usize,
) -> TlBackendResult<Scirs2Tensor> {
    if ensemble.is_empty() {
        return Err(TlBackendError::InvalidOperation(
            "bald_epistemic_uncertainty: ensemble must not be empty".to_string(),
        ));
    }

    let ref_shape = ensemble[0].shape().to_vec();
    for (i, m) in ensemble.iter().enumerate().skip(1) {
        if m.shape() != ref_shape.as_slice() {
            return Err(TlBackendError::InvalidOperation(format!(
                "bald_epistemic_uncertainty: member {i} has shape {:?}, expected {:?}",
                m.shape(),
                ref_shape
            )));
        }
    }

    // Mean of probabilities over ensemble members
    let n_members = ensemble.len() as f64;
    let mean_probs: Scirs2Tensor = ensemble
        .iter()
        .fold(ArrayD::<f64>::zeros(IxDyn(&ref_shape)), |acc, m| acc + m)
        .mapv(|v| v / n_members);

    // H(mean_probs)
    let entropy_of_mean = predictive_entropy(&mean_probs, axis)?;

    // mean_k H(probs_k)
    let member_entropies: Vec<Scirs2Tensor> = ensemble
        .iter()
        .map(|m| predictive_entropy(m, axis))
        .collect::<TlBackendResult<Vec<_>>>()?;

    let mean_entropy: Scirs2Tensor = {
        let entropy_shape = member_entropies[0].shape().to_vec();
        member_entropies
            .iter()
            .fold(ArrayD::<f64>::zeros(IxDyn(&entropy_shape)), |acc, h| {
                acc + h
            })
            .mapv(|v| v / n_members)
    };

    // BALD = H(mean) − mean(H)
    Ok(entropy_of_mean - mean_entropy)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    #[test]
    fn estimator_basic_statistics() {
        // Known data: two samples [[1,3],[5,7]] and [[3,5],[7,9]]
        // mean per elem: [[2,4],[6,8]]
        // var per elem:  [[1,1],[1,1]]
        let s0 =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 3.0, 5.0, 7.0]).expect("shape ok");
        let s1 =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![3.0, 5.0, 7.0, 9.0]).expect("shape ok");
        let samples = vec![s0, s1];
        let est = MonteCarloEstimator::default();
        let ue = est.estimate(&samples).expect("estimate failed");

        let expected_mean = [2.0, 4.0, 6.0, 8.0];
        for (&got, &exp) in ue.mean.iter().zip(expected_mean.iter()) {
            assert!((got - exp).abs() < 1e-12, "mean: got {got}, expected {exp}");
        }
        for &v in ue.variance.iter() {
            assert!((v - 1.0).abs() < 1e-12, "variance: got {v}, expected 1.0");
        }
    }

    #[test]
    fn predictive_entropy_uniform() {
        // Uniform over 4 classes: p_k = 0.25 → H = log(4) ≈ 1.386
        let p = 0.25_f64;
        let probs = ArrayD::from_elem(IxDyn(&[1, 4]), p);
        let h = predictive_entropy(&probs, 1).expect("entropy failed");
        let expected = -(4.0 * p * (p + 1e-10_f64).ln());
        for &v in h.iter() {
            assert!(
                (v - expected).abs() < 1e-6,
                "H_uniform: got {v}, expected ≈ {expected}"
            );
        }
    }

    #[test]
    fn predictive_entropy_certain() {
        // One-hot: p = [1, 0, 0] → H ≈ 0
        let probs = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 0.0, 0.0]).expect("shape ok");
        let h = predictive_entropy(&probs, 1).expect("entropy failed");
        for &v in h.iter() {
            // -1*ln(1+eps) - 0*ln(eps) - 0*ln(eps) ≈ very small
            assert!(v.abs() < 1e-8, "H_certain: got {v}, expected ≈ 0");
        }
    }

    #[test]
    fn bald_identical_ensemble() {
        // All ensemble members identical → epistemic uncertainty = 0
        let probs = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.2, 0.5, 0.3, 0.1, 0.8, 0.1])
            .expect("shape ok");
        let ensemble = vec![probs.clone(), probs.clone(), probs.clone()];
        let bald = bald_epistemic_uncertainty(&ensemble, 1).expect("bald failed");
        for &v in bald.iter() {
            assert!(
                v.abs() < 1e-10,
                "identical ensemble: bald {v} should be ≈ 0"
            );
        }
    }

    #[test]
    fn bald_diverse_ensemble() {
        // Two polar-opposite members: [1,0] and [0,1] → high epistemic uncertainty
        let m0 = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 0.0]).expect("shape ok");
        let m1 = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 1.0]).expect("shape ok");
        let ensemble = vec![m0, m1];
        let bald = bald_epistemic_uncertainty(&ensemble, 1).expect("bald failed");
        // H(mean=[0.5,0.5]) ≈ ln(2) ≈ 0.693; mean(H) ≈ 0 (each member is certain)
        // → BALD > 0
        for &v in bald.iter() {
            assert!(v > 0.5, "diverse ensemble: bald {v} should be > 0.5");
        }
    }
}
