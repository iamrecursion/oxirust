//! Shazeer-style auxiliary losses for load balancing in MoE layers.
//!
//! These follow Shazeer et al. (2017) "Outrageously Large Neural
//! Networks", §4 and §4.1:
//!
//! * **Importance loss** `L_imp = CV(Σ_b softmax(gate)_i)^2` — the squared
//!   coefficient of variation of the total gating mass each expert
//!   receives across a batch. Drives expected usage toward uniform.
//! * **Load loss** `L_load = CV(count_i)^2` — the squared coefficient of
//!   variation of how many batch items were actually routed to expert
//!   `i`. Drives hard-routing counts toward uniform.
//!
//! They are then combined as `alpha * (L_imp + L_load)` via
//! [`combined_aux_loss`]. `alpha` is a hyperparameter typically in
//! `[1e-3, 1e-1]`; the Switch paper uses `0.01`.
//!
//! All three routines are **pure functions** over the accumulated
//! [`BatchGatingStats`] so they can be unit-tested in isolation.

use ndarray::Array2;

use super::error::{MoeError, MoeResult};

/// Gating statistics accumulated across a batch.
///
/// Callers populate this as they iterate over batch items through a
/// [`super::layer::MoELayer`], then pass it to [`importance_loss`],
/// [`load_loss`], or [`combined_aux_loss`].
///
/// * `gate_scores_per_token` has shape `(batch_size, num_experts)` and
///   stores the **full** softmax distribution (not just the top-k
///   weights). Construct this from
///   [`super::gate::GatingDecision::full_softmax`] per token.
/// * `routed_expert_per_token` is a `batch_size`-long vector of the
///   **primary** expert (argmax / top-1) that each token was dispatched
///   to. For top-k (`k > 1`), the convention matches Shazeer's original
///   counting — use the top-1 expert; the load loss cares about hard
///   routing only.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchGatingStats {
    /// Softmax gate distribution per token, shape `(batch_size, num_experts)`.
    pub gate_scores_per_token: Array2<f64>,
    /// Index of the top-1 expert each token was routed to, length `batch_size`.
    pub routed_expert_per_token: Vec<usize>,
}

impl BatchGatingStats {
    /// Create an empty stats accumulator dimensioned for `batch_size` tokens
    /// over `num_experts` experts. `gate_scores_per_token` is initialised to
    /// zeros; `routed_expert_per_token` is pre-allocated with capacity.
    pub fn empty(batch_size: usize, num_experts: usize) -> Self {
        Self {
            gate_scores_per_token: Array2::<f64>::zeros((batch_size, num_experts)),
            routed_expert_per_token: Vec::with_capacity(batch_size),
        }
    }

    /// Number of batch items (`gate_scores_per_token.nrows()`).
    pub fn batch_size(&self) -> usize {
        self.gate_scores_per_token.nrows()
    }

    /// Number of experts (`gate_scores_per_token.ncols()`).
    pub fn num_experts(&self) -> usize {
        self.gate_scores_per_token.ncols()
    }

    /// Count how many tokens were routed to each expert — output has
    /// length `num_experts`.
    pub fn expert_counts(&self) -> Vec<f64> {
        let mut counts = vec![0.0_f64; self.num_experts()];
        for &idx in &self.routed_expert_per_token {
            if idx < counts.len() {
                counts[idx] += 1.0;
            }
        }
        counts
    }

    /// Sum of softmax gate mass assigned to each expert across the batch.
    /// Output length is `num_experts`.
    pub fn expert_importance(&self) -> Vec<f64> {
        let mut sums = vec![0.0_f64; self.num_experts()];
        for row in self.gate_scores_per_token.rows() {
            for (i, v) in row.iter().enumerate() {
                sums[i] += *v;
            }
        }
        sums
    }
}

/// Squared coefficient of variation helper.
///
/// `CV(x) = std(x) / mean(x)` — we return `CV^2` directly to avoid an
/// unnecessary `sqrt`. When `mean(x) == 0` the loss is defined as 0
/// (nothing to penalise — no gate mass or no tokens at all).
fn cv_squared(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    if mean.abs() <= f64::EPSILON {
        return 0.0;
    }
    // Population variance (matches the Shazeer reference). Using
    // population rather than sample variance keeps the loss
    // scale-invariant to batch size in the limit of many tokens.
    let variance: f64 = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    variance / (mean * mean)
}

/// Importance loss: squared CV of per-expert gate mass.
///
/// Shazeer et al. (2017), eq. (6).
pub fn importance_loss(stats: &BatchGatingStats) -> MoeResult<f64> {
    if stats.num_experts() == 0 {
        return Err(MoeError::EmptyExpertPool);
    }
    let importances = stats.expert_importance();
    Ok(cv_squared(&importances))
}

/// Load loss: squared CV of per-expert routing counts.
///
/// Shazeer et al. (2017), eq. (7) (hard-count variant; the smoothed
/// noisy-top-k version is out of scope for the research preview).
pub fn load_loss(stats: &BatchGatingStats) -> MoeResult<f64> {
    if stats.num_experts() == 0 {
        return Err(MoeError::EmptyExpertPool);
    }
    let counts = stats.expert_counts();
    Ok(cv_squared(&counts))
}

/// Combined auxiliary loss: `alpha * (L_imp + L_load)`.
///
/// `alpha` is the Shazeer "w_importance" coefficient; Switch Transformer
/// uses `0.01`.
pub fn combined_aux_loss(stats: &BatchGatingStats, alpha: f64) -> MoeResult<f64> {
    let l_imp = importance_loss(stats)?;
    let l_load = load_loss(stats)?;
    Ok(alpha * (l_imp + l_load))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cv_squared_zero_for_uniform() {
        let uniform = vec![1.0, 1.0, 1.0, 1.0];
        assert!(cv_squared(&uniform).abs() < 1e-12);
    }

    #[test]
    fn cv_squared_positive_for_skew() {
        let skewed = vec![4.0, 0.0, 0.0, 0.0];
        assert!(cv_squared(&skewed) > 0.0);
    }

    #[test]
    fn cv_squared_zero_on_zero_mean() {
        let zeros = vec![0.0, 0.0, 0.0];
        assert!(cv_squared(&zeros).abs() < 1e-12);
    }
}
