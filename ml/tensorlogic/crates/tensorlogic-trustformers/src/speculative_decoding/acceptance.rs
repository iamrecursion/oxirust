//! Rejection-sampling primitives for speculative decoding.
//!
//! # The acceptance theorem (Leviathan et al. 2023, Theorem 3.5)
//!
//! Let `p_draft(x)` and `p_target(x)` be the probabilities assigned by the
//! draft and target models to a token `x` given a prefix.  Define the
//! per-token acceptance probability
//!
//! ```text
//!     a(x) = min(1, p_target(x) / p_draft(x))
//! ```
//!
//! and, on rejection, resample from the **adjusted target distribution**
//!
//! ```text
//!     p_adj(x) ∝ max(0, p_target(x) - p_draft(x))
//! ```
//!
//! Then the token ultimately emitted is distributed **exactly** as a draw
//! from `p_target` — independently of how poor `p_draft` is.  This module
//! implements both halves of the test as pure functions (no I/O, no global
//! state) so the acceptance logic can be unit-tested in isolation.
//!
//! All inputs are accepted as **log-probabilities**; the routines perform the
//! log→linear conversion locally.  This matches the trait surface in
//! `traits.rs`, which stores distributions as `LogProb` vectors, and avoids
//! the classic bug of calling `accept(log_p_draft, log_p_target)` by accident
//! (type aliases make direction easy to confuse).

use crate::speculative_decoding::error::{SpeculativeDecodingError, SpeculativeDecodingResult};
use crate::speculative_decoding::rng::SpecRng;
use crate::speculative_decoding::traits::{LogProb, TokenId};

/// Per-token Bernoulli acceptance test.
///
/// Returns `true` iff the draft token at this position should be kept,
/// following `accept = min(1, p_target / p_draft)`.
///
/// Both log-probability arguments must be finite; if either is `-infinity`
/// (e.g. impossible token) we fall through to a conservative outcome:
///
/// * `p_target == 0, p_draft > 0` ⇒ always reject.
/// * `p_draft == 0, p_target > 0` ⇒ always accept (the draft was a nonsense
///   pick but the target likes the token; accept and let the downstream step
///   benefit).
/// * `p_draft == 0 && p_target == 0` ⇒ reject and fall through to resample.
pub fn accept(draft_logprob: LogProb, target_logprob: LogProb, rng: &mut dyn SpecRng) -> bool {
    match (draft_logprob.is_finite(), target_logprob.is_finite()) {
        (false, true) => return true,
        (true, false) => return false,
        (false, false) => return false,
        (true, true) => {}
    }

    // ratio = p_target / p_draft = exp(target_logprob - draft_logprob).
    let log_ratio = target_logprob - draft_logprob;
    if log_ratio >= 0.0 {
        // min(1, ratio) = 1 when ratio >= 1.
        return true;
    }

    // 0 < ratio < 1: accept with probability `ratio`.
    let ratio = log_ratio.exp();
    let u = rng.next_unit_f64();
    u < ratio
}

/// Compute the adjusted distribution `q(x) ∝ max(0, p_target(x) - p_draft(x))`
/// as a linear-space vector summing to 1.
///
/// If the two log-prob rows describe identical distributions, every entry in
/// the unnormalized vector is zero; in that pathological case we fall back to
/// the raw target distribution (which still samples correctly, because the
/// token was rejected and we need *some* draw).  If even that is all-zero we
/// surface [`SpeculativeDecodingError::DegenerateDistribution`].
pub fn adjusted_distribution(
    target_logprobs: &[LogProb],
    draft_logprobs: &[LogProb],
) -> SpeculativeDecodingResult<Vec<f64>> {
    if target_logprobs.len() != draft_logprobs.len() {
        return Err(SpeculativeDecodingError::DistributionWidthMismatch {
            expected: target_logprobs.len(),
            got: draft_logprobs.len(),
        });
    }

    let target: Vec<f64> = target_logprobs
        .iter()
        .map(|lp| if lp.is_finite() { lp.exp() } else { 0.0 })
        .collect();
    let draft: Vec<f64> = draft_logprobs
        .iter()
        .map(|lp| if lp.is_finite() { lp.exp() } else { 0.0 })
        .collect();

    let mut adjusted: Vec<f64> = target
        .iter()
        .zip(draft.iter())
        .map(|(t, d)| (t - d).max(0.0))
        .collect();

    let mass: f64 = adjusted.iter().sum();
    if mass > 0.0 {
        for x in adjusted.iter_mut() {
            *x /= mass;
        }
        return Ok(adjusted);
    }

    // Fall back to the raw target distribution (in linear space).
    let target_mass: f64 = target.iter().sum();
    if target_mass <= 0.0 {
        return Err(SpeculativeDecodingError::DegenerateDistribution);
    }
    for x in target.iter() {
        debug_assert!(*x >= 0.0);
    }
    let fallback: Vec<f64> = target.iter().map(|t| t / target_mass).collect();
    Ok(fallback)
}

/// Sample a token from a linear-space probability vector using inverse-CDF
/// sampling against a uniform draw from `rng`.
///
/// Assumes `probs` sums to ≈ 1 (we do *not* re-normalize — callers such as
/// [`adjusted_distribution`] take care of that); returns the last index if
/// numerical drift causes the cumulative sum to fall slightly under the
/// uniform draw.
pub fn sample_index(probs: &[f64], rng: &mut dyn SpecRng) -> SpeculativeDecodingResult<TokenId> {
    if probs.is_empty() {
        return Err(SpeculativeDecodingError::DegenerateDistribution);
    }
    let u = rng.next_unit_f64();
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += *p;
        if u < cum {
            return Ok(i);
        }
    }
    Ok(probs.len() - 1)
}

/// Sample a token from the adjusted target distribution
/// `q ∝ max(0, p_target - p_draft)`.
///
/// This is the routine called when the Bernoulli acceptance test rejects a
/// draft token: its output is, by the Leviathan theorem, drawn from the exact
/// target distribution even though the draft was biased.
pub fn resample_from_adjusted_target(
    target_logprobs: &[LogProb],
    draft_logprobs: &[LogProb],
    rng: &mut dyn SpecRng,
) -> SpeculativeDecodingResult<TokenId> {
    let adjusted = adjusted_distribution(target_logprobs, draft_logprobs)?;
    sample_index(&adjusted, rng)
}

/// Sample a token from a log-prob row by first exponentiating and
/// normalizing.  Used on the bonus position when every draft token is
/// accepted.
pub fn sample_from_logprobs(
    logprobs: &[LogProb],
    rng: &mut dyn SpecRng,
) -> SpeculativeDecodingResult<TokenId> {
    if logprobs.is_empty() {
        return Err(SpeculativeDecodingError::DegenerateDistribution);
    }
    let max = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err(SpeculativeDecodingError::DegenerateDistribution);
    }
    let exps: Vec<f64> = logprobs.iter().map(|lp| (lp - max).exp()).collect();
    let mass: f64 = exps.iter().sum();
    if mass <= 0.0 {
        return Err(SpeculativeDecodingError::DegenerateDistribution);
    }
    let probs: Vec<f64> = exps.iter().map(|e| e / mass).collect();
    sample_index(&probs, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{SeedableRng, StdRng};

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn accept_when_target_dominates() {
        let mut r = rng(1);
        // target much more likely than draft ⇒ always accept.
        for _ in 0..100 {
            assert!(accept(-2.0, -0.1, &mut r));
        }
    }

    #[test]
    fn reject_when_draft_much_more_likely() {
        let mut r = rng(2);
        // ratio = exp(-3.0) ≈ 0.05; most draws should reject.
        let rejects = (0..1000).filter(|_| !accept(-0.1, -3.1, &mut r)).count();
        assert!(rejects > 900, "expected >90% rejects, got {}", rejects);
    }

    #[test]
    fn adjusted_nonnegative_and_normalized() {
        // p_target = [0.1, 0.2, 0.7], p_draft = [0.4, 0.4, 0.2]
        let tgt: Vec<f64> = vec![0.1f64.ln(), 0.2f64.ln(), 0.7f64.ln()];
        let drf: Vec<f64> = vec![0.4f64.ln(), 0.4f64.ln(), 0.2f64.ln()];
        let q = adjusted_distribution(&tgt, &drf).expect("adjusted");
        assert_eq!(q.len(), 3);
        for &p in &q {
            assert!(p >= 0.0);
        }
        let sum: f64 = q.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // max(0, p_t - p_d) = [0, 0, 0.5] → normalize → [0, 0, 1.0].
        assert!((q[0] - 0.0).abs() < 1e-9);
        assert!((q[1] - 0.0).abs() < 1e-9);
        assert!((q[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn adjusted_falls_back_to_target_when_equal() {
        let tgt: Vec<f64> = vec![0.25f64.ln(); 4];
        let drf: Vec<f64> = vec![0.25f64.ln(); 4];
        let q = adjusted_distribution(&tgt, &drf).expect("adjusted");
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for &p in &q {
            assert!((p - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn adjusted_mismatched_widths_errors() {
        let tgt: Vec<f64> = vec![0.5f64.ln(); 3];
        let drf: Vec<f64> = vec![0.5f64.ln(); 4];
        let res = adjusted_distribution(&tgt, &drf);
        assert!(res.is_err());
    }

    #[test]
    fn sample_index_obeys_distribution() {
        let mut r = rng(42);
        let p = vec![0.1, 0.2, 0.3, 0.4];
        let n = 10_000;
        let mut counts = [0usize; 4];
        for _ in 0..n {
            let idx = sample_index(&p, &mut r).expect("sample");
            counts[idx] += 1;
        }
        // Within 3% of expected for this sample count.
        let emp: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
        for (e, t) in emp.iter().zip(p.iter()) {
            assert!((e - t).abs() < 0.03, "emp {:?} vs {:?}", emp, p);
        }
    }

    #[test]
    fn resample_returns_in_range() {
        let mut r = rng(7);
        let tgt: Vec<f64> = vec![0.1f64.ln(), 0.2f64.ln(), 0.7f64.ln()];
        let drf: Vec<f64> = vec![0.4f64.ln(), 0.4f64.ln(), 0.2f64.ln()];
        for _ in 0..200 {
            let idx = resample_from_adjusted_target(&tgt, &drf, &mut r).expect("resample");
            assert!(idx < 3);
        }
    }

    #[test]
    fn sample_from_logprobs_is_normalized() {
        let mut r = rng(13);
        // non-normalized log-probs (max-subtraction handles this).
        let lp = vec![-0.1, -1.0, -2.0, -3.0];
        let mut counts = [0usize; 4];
        for _ in 0..5_000 {
            let idx = sample_from_logprobs(&lp, &mut r).expect("sample");
            counts[idx] += 1;
        }
        // index 0 should dominate.
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
        assert!(counts[2] > counts[3]);
    }
}
