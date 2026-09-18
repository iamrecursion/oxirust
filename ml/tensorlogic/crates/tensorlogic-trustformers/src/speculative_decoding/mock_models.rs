//! Deterministic mock models for testing speculative decoding.
//!
//! These mocks return **fixed** categorical distributions independent of the
//! supplied prefix, which is exactly what the Leviathan theorem needs to make
//! empirical-distribution tests tractable: if target and draft both ignore
//! history, then the marginal distribution over emitted tokens is precisely
//! `p_target`, and the engine's output can be chi-squared against `p_target`.
//!
//! Two mock kinds are exposed:
//!
//! * [`FixedDistDraftModel`] / [`FixedDistTargetModel`] — constant log-probs
//!   over a tiny vocabulary.  Used in both the module-private unit tests and
//!   the crate-level integration test.
//! * [`MockDraftModel`] / [`MockTargetModel`] — thin typedefs exported for
//!   downstream consumers that want a "just works" pair.

use crate::speculative_decoding::error::{SpeculativeDecodingError, SpeculativeDecodingResult};
use crate::speculative_decoding::rng::SpecRng;
use crate::speculative_decoding::traits::{
    DraftModel, DraftProposal, LogProb, TargetModel, TargetScores, TokenId,
};

/// Draft model that always returns the same categorical distribution and
/// samples tokens from it.
#[derive(Debug, Clone)]
pub struct FixedDistDraftModel {
    probs: Vec<f64>,
    logprobs: Vec<LogProb>,
}

impl FixedDistDraftModel {
    /// Build a mock draft from a linear-space probability vector.  The vector
    /// must be non-empty and sum to ≈ 1.
    pub fn new(probs: Vec<f64>) -> SpeculativeDecodingResult<Self> {
        if probs.is_empty() {
            return Err(SpeculativeDecodingError::InvalidConfig(
                "FixedDistDraftModel requires a non-empty probability vector".into(),
            ));
        }
        let sum: f64 = probs.iter().copied().sum();
        if !(sum > 0.0 && sum.is_finite()) {
            return Err(SpeculativeDecodingError::InvalidConfig(
                "FixedDistDraftModel probabilities must have positive finite mass".into(),
            ));
        }
        let normalized: Vec<f64> = probs.iter().map(|p| *p / sum).collect();
        let logprobs: Vec<f64> = normalized
            .iter()
            .map(|p| if *p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
            .collect();
        Ok(Self {
            probs: normalized,
            logprobs,
        })
    }

    /// Access the (normalized) linear-space probabilities.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Access the log-probability row.
    pub fn logprobs(&self) -> &[LogProb] {
        &self.logprobs
    }
}

impl DraftModel for FixedDistDraftModel {
    fn vocab_size(&self) -> usize {
        self.probs.len()
    }

    fn propose(
        &self,
        _prefix: &[TokenId],
        k: usize,
        rng: &mut dyn SpecRng,
    ) -> SpeculativeDecodingResult<DraftProposal> {
        let mut tokens = Vec::with_capacity(k);
        let mut token_logprobs = Vec::with_capacity(k);
        let mut distributions = Vec::with_capacity(k);
        for _ in 0..k {
            let idx = sample_categorical(&self.probs, rng)?;
            tokens.push(idx);
            token_logprobs.push(self.logprobs[idx]);
            distributions.push(self.logprobs.clone());
        }
        Ok(DraftProposal {
            tokens,
            token_logprobs,
            distributions,
        })
    }
}

/// Target model returning the same categorical distribution for every
/// position.
#[derive(Debug, Clone)]
pub struct FixedDistTargetModel {
    probs: Vec<f64>,
    logprobs: Vec<LogProb>,
}

impl FixedDistTargetModel {
    /// Build a mock target from a linear-space probability vector.  See
    /// [`FixedDistDraftModel::new`] for invariants.
    pub fn new(probs: Vec<f64>) -> SpeculativeDecodingResult<Self> {
        if probs.is_empty() {
            return Err(SpeculativeDecodingError::InvalidConfig(
                "FixedDistTargetModel requires a non-empty probability vector".into(),
            ));
        }
        let sum: f64 = probs.iter().copied().sum();
        if !(sum > 0.0 && sum.is_finite()) {
            return Err(SpeculativeDecodingError::InvalidConfig(
                "FixedDistTargetModel probabilities must have positive finite mass".into(),
            ));
        }
        let normalized: Vec<f64> = probs.iter().map(|p| *p / sum).collect();
        let logprobs: Vec<f64> = normalized
            .iter()
            .map(|p| if *p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
            .collect();
        Ok(Self {
            probs: normalized,
            logprobs,
        })
    }

    /// Access the (normalized) linear-space probabilities.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Access the log-probability row.
    pub fn logprobs(&self) -> &[LogProb] {
        &self.logprobs
    }
}

impl TargetModel for FixedDistTargetModel {
    fn vocab_size(&self) -> usize {
        self.probs.len()
    }

    fn verify(
        &self,
        _prefix: &[TokenId],
        draft_tokens: &[TokenId],
    ) -> SpeculativeDecodingResult<TargetScores> {
        let rows = draft_tokens.len() + 1;
        let distributions: Vec<Vec<LogProb>> = (0..rows).map(|_| self.logprobs.clone()).collect();
        Ok(TargetScores { distributions })
    }
}

/// Helper: sample a categorical index via inverse-CDF against `rng`.
pub(crate) fn sample_categorical(
    probs: &[f64],
    rng: &mut dyn SpecRng,
) -> SpeculativeDecodingResult<TokenId> {
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

/// Public alias used by tests: fixed-distribution draft model.
pub type MockDraftModel = FixedDistDraftModel;

/// Public alias used by tests: fixed-distribution target model.
pub type MockTargetModel = FixedDistTargetModel;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{SeedableRng, StdRng};

    #[test]
    fn draft_model_normalizes_input() {
        let d = FixedDistDraftModel::new(vec![2.0, 2.0]).expect("normalize");
        for p in d.probs() {
            assert!((p - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn draft_model_rejects_empty() {
        let r = FixedDistDraftModel::new(vec![]);
        assert!(r.is_err());
    }

    #[test]
    fn draft_model_rejects_zero_mass() {
        let r = FixedDistDraftModel::new(vec![0.0, 0.0, 0.0]);
        assert!(r.is_err());
    }

    #[test]
    fn propose_shapes_are_consistent() {
        let d = FixedDistDraftModel::new(vec![0.25; 4]).expect("d");
        let mut rng = StdRng::seed_from_u64(1);
        let p = d.propose(&[0, 1, 2], 3, &mut rng).expect("propose");
        assert_eq!(p.tokens.len(), 3);
        assert_eq!(p.token_logprobs.len(), 3);
        assert_eq!(p.distributions.len(), 3);
        for row in &p.distributions {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn verify_returns_k_plus_one_rows() {
        let t = FixedDistTargetModel::new(vec![0.25; 4]).expect("t");
        let ts = t.verify(&[0, 1], &[1, 2, 3]).expect("verify");
        assert_eq!(ts.distributions.len(), 4);
        for row in &ts.distributions {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn propose_is_reproducible_with_seed() {
        let d = FixedDistDraftModel::new(vec![0.1, 0.2, 0.3, 0.4]).expect("d");
        let mut r1 = StdRng::seed_from_u64(7);
        let mut r2 = StdRng::seed_from_u64(7);
        let p1 = d.propose(&[0], 8, &mut r1).expect("p1");
        let p2 = d.propose(&[0], 8, &mut r2).expect("p2");
        assert_eq!(p1.tokens, p2.tokens);
    }
}
