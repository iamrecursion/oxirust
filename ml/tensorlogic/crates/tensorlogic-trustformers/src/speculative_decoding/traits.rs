//! Core abstractions for model-level speculative decoding.
//!
//! The speculative decoding protocol of Leviathan et al. (2023) and
//! Chen et al. (2023) is driven by two cooperating models:
//!
//! * A cheap **draft model** that extends the current prefix by *k* candidate
//!   tokens (cf. [`DraftModel`]).
//! * An expensive **target model** that, in a single parallel forward pass,
//!   scores every prefix-continuation of the draft (cf. [`TargetModel`]).
//!
//! The engine then runs the Bernoulli acceptance test
//! `accept = min(1, p_target / p_draft)` position by position, re-sampling
//! the first rejection from the adjusted target distribution
//! `max(0, p_target - p_draft)` (see `acceptance.rs`).  Because this trait
//! layer never references probabilities in linear space directly — the engine
//! converts between log-probs and probs at the call sites — we can host the
//! draft/target models on CPU or GPU, eager or graph-compiled, without
//! leaking those concerns into the acceptance math.
//!
//! ## Why full distributions, not just token log-probs
//!
//! The naive signature
//! `propose(prefix, k) -> Vec<(TokenId, LogProb)>`
//! collapses each step into a single `(token, logprob)` pair.  That is
//! insufficient: the adjusted re-sampling distribution
//! `max(0, p_target - p_draft)` is defined over the **entire vocabulary**,
//! so both draft and target must return full per-position distributions.
//! The trait shapes encode this explicitly.
//!
//! ## Invariants enforced by [`DraftProposal`] / [`TargetScores`]
//!
//! * `tokens.len() == distributions.len() == k`.
//! * `distributions[i].len() == vocab_size` for the configured vocab.
//! * Every `LogProb` row is normalized (log-sum-exp ≈ 0).  The traits do not
//!   re-normalize — it is the implementation's responsibility.
//!
//! The engine defensively checks shapes at runtime and short-circuits with a
//! [`crate::speculative_decoding::SpeculativeDecodingError`] if anything is malformed.

use crate::speculative_decoding::error::SpeculativeDecodingResult;

/// Vocabulary-scoped token identifier.
///
/// Matches the convention used by `rule_guided_decoder::TokenId` so the two
/// decoders can share mappers in future work.
pub type TokenId = usize;

/// Natural-log probability.  We deliberately use a type alias rather than a
/// newtype so callers can freely mix with `f64` arithmetic; the engine is the
/// only place where domains matter (log vs. linear) and it converts locally.
pub type LogProb = f64;

/// Output of a single [`DraftModel::propose`] call.
///
/// Fields are aligned index-wise: `tokens[i]` is the draft's sampled token at
/// step *i*, `token_logprobs[i]` is its log-probability under the draft, and
/// `distributions[i]` is the draft's **full** log-probability row over the
/// vocabulary for that step (needed by the engine for the rejection test and
/// the adjusted re-sampling).
#[derive(Debug, Clone, PartialEq)]
pub struct DraftProposal {
    /// The `k` tokens the draft model proposes to extend the prefix with.
    pub tokens: Vec<TokenId>,
    /// Log-probability of each chosen token under the draft distribution.
    pub token_logprobs: Vec<LogProb>,
    /// Full per-step log-probability rows — `distributions[i]` has length
    /// `vocab_size` and sums (in linear space) to 1.
    pub distributions: Vec<Vec<LogProb>>,
}

impl DraftProposal {
    /// Length of the proposal (number of draft positions, commonly `k`).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Is the proposal empty (no tokens drafted)?
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Output of a single [`TargetModel::verify`] call.
///
/// For `k` draft tokens the target must return `k + 1` distributions: the
/// first `k` at the draft-covered positions (used by the acceptance test),
/// plus one **bonus** distribution at position `k + 1` that the engine uses
/// if every draft token is accepted — see Leviathan et al. 2023 §3.2.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetScores {
    /// `k + 1` log-probability rows, each of length `vocab_size`.
    pub distributions: Vec<Vec<LogProb>>,
}

impl TargetScores {
    /// Number of positions scored (always `k + 1` in canonical usage).
    pub fn len(&self) -> usize {
        self.distributions.len()
    }

    /// Are there no scored positions at all?
    pub fn is_empty(&self) -> bool {
        self.distributions.is_empty()
    }
}

/// A model capable of *cheaply* extending a prefix by `k` tokens while
/// exposing full vocabulary distributions at every step.
///
/// Implementations must be deterministic w.r.t. the supplied RNG so that the
/// engine's empirical-distribution tests are reproducible.
pub trait DraftModel {
    /// Vocabulary cardinality the model emits log-probs over.
    fn vocab_size(&self) -> usize;

    /// Extend `prefix` by `k` draft tokens; return the chosen tokens, their
    /// log-probabilities and the **full** per-position distributions.
    ///
    /// Note `rng` is a dyn-compatible shim: the callee can down-mix it into
    /// whatever PRNG it likes internally, but the engine always drives a
    /// single `StdRng` to keep the acceptance branch of the algorithm
    /// reproducible.
    fn propose(
        &self,
        prefix: &[TokenId],
        k: usize,
        rng: &mut dyn crate::speculative_decoding::rng::SpecRng,
    ) -> SpeculativeDecodingResult<DraftProposal>;
}

/// A model that, given a prefix and up to `k` draft continuations, returns
/// per-position distributions (as log-probs) in a single forward pass.
pub trait TargetModel {
    /// Vocabulary cardinality the target emits log-probs over.  Must match
    /// the draft's `vocab_size()`.
    fn vocab_size(&self) -> usize;

    /// Score `prefix` concatenated with `draft_tokens`: return `k + 1`
    /// distributions (the `k` draft-covered positions plus the bonus).
    fn verify(
        &self,
        prefix: &[TokenId],
        draft_tokens: &[TokenId],
    ) -> SpeculativeDecodingResult<TargetScores>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_proposal_len_matches_tokens() {
        let p = DraftProposal {
            tokens: vec![1, 2, 3],
            token_logprobs: vec![-0.1, -0.2, -0.3],
            distributions: vec![vec![-0.1; 4], vec![-0.2; 4], vec![-0.3; 4]],
        };
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());
    }

    #[test]
    fn empty_proposal_is_empty() {
        let p = DraftProposal {
            tokens: vec![],
            token_logprobs: vec![],
            distributions: vec![],
        };
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn target_scores_len_matches_rows() {
        let t = TargetScores {
            distributions: vec![vec![-0.5; 4], vec![-0.5; 4]],
        };
        assert_eq!(t.len(), 2);
        assert!(!t.is_empty());
    }
}
