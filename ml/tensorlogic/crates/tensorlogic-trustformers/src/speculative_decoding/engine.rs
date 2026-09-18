//! Speculative decoding engine.
//!
//! [`SpeculativeDecoder`] composes a [`DraftModel`] and a [`TargetModel`] and
//! runs the Leviathan / Chen speculative generation loop:
//!
//! ```text
//! loop until max_tokens reached:
//!   1. draft.propose(prefix, k)       → k candidate tokens + distributions
//!   2. target.verify(prefix, draft)   → k+1 target distributions
//!   3. for i in 0..k:
//!        if accept(p_draft_i, p_target_i, rng):
//!            append draft[i]
//!        else:
//!            append resample_from_adjusted_target(p_target_i, p_draft_i, rng)
//!            break
//!   4. if all k accepted:
//!        append sample_from_logprobs(p_target_{k+1}, rng)  (bonus)
//!   5. update metrics; continue.
//! ```
//!
//! The **number of tokens appended per round** is therefore in `1..=k+1`,
//! and — crucially — the marginal distribution of each appended token is
//! *identical* to `p_target(prefix)`.  That correctness is what the empirical
//! chi-square test in `tests.rs` validates against 10 000 samples.

use std::marker::PhantomData;

use scirs2_core::random::{SeedableRng, StdRng};

use crate::speculative_decoding::acceptance::{
    accept, resample_from_adjusted_target, sample_from_logprobs,
};
use crate::speculative_decoding::error::{SpeculativeDecodingError, SpeculativeDecodingResult};
use crate::speculative_decoding::metrics::SpeculativeMetrics;
use crate::speculative_decoding::rng::SpecRng;
use crate::speculative_decoding::traits::{
    DraftModel, DraftProposal, TargetModel, TargetScores, TokenId,
};

/// Configuration for the speculative decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeDecoderConfig {
    /// Number of draft tokens to propose per round (default `4`).
    pub k: usize,
    /// Cost ratio `c_draft / c_target` for speedup modeling (default `0.125`).
    pub cost_ratio: f32,
    /// If `true`, the engine halts the generation loop on the first
    /// `eos_token` it emits.
    pub stop_on_eos: bool,
    /// Optional end-of-sequence token id (ignored unless `stop_on_eos`).
    pub eos_token: Option<TokenId>,
}

impl Default for SpeculativeDecoderConfig {
    fn default() -> Self {
        Self {
            k: 4,
            cost_ratio: 0.125,
            stop_on_eos: false,
            eos_token: None,
        }
    }
}

impl SpeculativeDecoderConfig {
    /// Convenience builder: set draft depth.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Convenience builder: set cost ratio for the speedup estimate.
    pub fn with_cost_ratio(mut self, r: f32) -> Self {
        self.cost_ratio = r;
        self
    }

    /// Convenience builder: attach an EOS token and enable early stopping.
    pub fn with_eos(mut self, eos: TokenId) -> Self {
        self.eos_token = Some(eos);
        self.stop_on_eos = true;
        self
    }

    /// Validate the configuration, returning an [`SpeculativeDecodingError`]
    /// if any invariant is violated.
    pub fn validate(&self) -> SpeculativeDecodingResult<()> {
        if self.k == 0 {
            return Err(SpeculativeDecodingError::InvalidConfig(
                "draft depth `k` must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// Speculative decoder composing a draft model and a target model.
///
/// Trait bounds are intentionally deferred to the `impl` blocks rather than
/// baked into the struct definition so callers can hold a
/// `SpeculativeDecoder` whose inner models do not implement `Debug`.
pub struct SpeculativeDecoder<D: DraftModel, T: TargetModel> {
    draft: D,
    target: T,
    config: SpeculativeDecoderConfig,
    metrics: SpeculativeMetrics,
    _pd: PhantomData<()>,
}

impl<D: DraftModel + std::fmt::Debug, T: TargetModel + std::fmt::Debug> std::fmt::Debug
    for SpeculativeDecoder<D, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculativeDecoder")
            .field("draft", &self.draft)
            .field("target", &self.target)
            .field("config", &self.config)
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl<D: DraftModel, T: TargetModel> SpeculativeDecoder<D, T> {
    /// Build a decoder.  Returns an error if the draft and target disagree on
    /// vocabulary size, or if the config is invalid.
    pub fn new(
        draft: D,
        target: T,
        config: SpeculativeDecoderConfig,
    ) -> SpeculativeDecodingResult<Self> {
        config.validate()?;
        if draft.vocab_size() != target.vocab_size() {
            return Err(SpeculativeDecodingError::VocabMismatch {
                draft: draft.vocab_size(),
                target: target.vocab_size(),
            });
        }
        let metrics = SpeculativeMetrics::new().with_cost_ratio(config.cost_ratio);
        Ok(Self {
            draft,
            target,
            config,
            metrics,
            _pd: PhantomData,
        })
    }

    /// Read-only access to the current metrics snapshot.
    pub fn metrics(&self) -> &SpeculativeMetrics {
        &self.metrics
    }

    /// Reset the metrics counters.
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &SpeculativeDecoderConfig {
        &self.config
    }

    /// Run speculative decoding starting from `prefix` and producing at most
    /// `max_tokens` *new* tokens.  The returned vector contains **only** the
    /// generated continuation, not the original prefix.
    ///
    /// Uses an internally-seeded deterministic [`StdRng`] (seed `42`).
    /// See [`Self::generate_with_rng`] for caller-controlled seeding.
    pub fn generate(
        &mut self,
        prefix: &[TokenId],
        max_tokens: usize,
    ) -> SpeculativeDecodingResult<Vec<TokenId>> {
        let mut rng = StdRng::seed_from_u64(42);
        self.generate_with_rng(prefix, max_tokens, &mut rng)
    }

    /// Run speculative decoding with a caller-supplied RNG.
    pub fn generate_with_rng(
        &mut self,
        prefix: &[TokenId],
        max_tokens: usize,
        rng: &mut dyn SpecRng,
    ) -> SpeculativeDecodingResult<Vec<TokenId>> {
        if prefix.is_empty() {
            return Err(SpeculativeDecodingError::EmptyPrefix);
        }

        let vocab = self.draft.vocab_size();
        let k = self.config.k;
        let mut working = prefix.to_vec();
        let mut output: Vec<TokenId> = Vec::with_capacity(max_tokens);

        while output.len() < max_tokens {
            let remaining = max_tokens - output.len();
            let round_k = k.min(remaining.max(1));

            let proposal = self.draft.propose(&working, round_k, rng)?;
            validate_proposal(&proposal, round_k, vocab)?;

            let target_scores = self.target.verify(&working, &proposal.tokens)?;
            validate_target_scores(&target_scores, round_k, vocab)?;

            let (accepted_count, emitted) =
                run_rejection_loop(&proposal, &target_scores, round_k, vocab, rng)?;

            let mut committed_this_round = 0u32;
            for token in emitted.into_iter() {
                output.push(token);
                working.push(token);
                committed_this_round += 1;
                if output.len() >= max_tokens {
                    break;
                }
                if self.config.stop_on_eos
                    && self
                        .config
                        .eos_token
                        .map(|eos| eos == token)
                        .unwrap_or(false)
                {
                    break;
                }
            }

            self.metrics.record_round(
                round_k as u32,
                accepted_count as u32,
                committed_this_round,
                round_k as u32,
            );

            if self.config.stop_on_eos {
                if let Some(eos) = self.config.eos_token {
                    if output.last().copied() == Some(eos) {
                        break;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Validate the shape and content of a draft proposal.
fn validate_proposal(p: &DraftProposal, k: usize, vocab: usize) -> SpeculativeDecodingResult<()> {
    if p.tokens.len() != k || p.token_logprobs.len() != k || p.distributions.len() != k {
        return Err(SpeculativeDecodingError::DraftShapeMismatch {
            tokens: p.tokens.len(),
            logprobs: p.token_logprobs.len(),
            distributions: p.distributions.len(),
        });
    }
    for row in &p.distributions {
        if row.len() != vocab {
            return Err(SpeculativeDecodingError::DistributionWidthMismatch {
                expected: vocab,
                got: row.len(),
            });
        }
    }
    for &t in &p.tokens {
        if t >= vocab {
            return Err(SpeculativeDecodingError::TokenOutOfRange {
                token: t,
                vocab_size: vocab,
            });
        }
    }
    Ok(())
}

/// Validate the shape and content of target-verification scores.
fn validate_target_scores(
    t: &TargetScores,
    k: usize,
    vocab: usize,
) -> SpeculativeDecodingResult<()> {
    if t.distributions.len() != k + 1 {
        return Err(SpeculativeDecodingError::TargetShapeMismatch {
            expected: k + 1,
            got: t.distributions.len(),
        });
    }
    for row in &t.distributions {
        if row.len() != vocab {
            return Err(SpeculativeDecodingError::DistributionWidthMismatch {
                expected: vocab,
                got: row.len(),
            });
        }
    }
    Ok(())
}

/// Execute the acceptance / rejection sweep for a single speculative round.
///
/// Returns the tuple `(accepted_count, emitted_tokens)` where `emitted_tokens`
/// has length in `1..=k+1`.
fn run_rejection_loop(
    proposal: &DraftProposal,
    target_scores: &TargetScores,
    k: usize,
    vocab: usize,
    rng: &mut dyn SpecRng,
) -> SpeculativeDecodingResult<(usize, Vec<TokenId>)> {
    let mut emitted: Vec<TokenId> = Vec::with_capacity(k + 1);
    let mut accepted: usize = 0;

    for i in 0..k {
        let draft_token = proposal.tokens[i];
        let target_row = &target_scores.distributions[i];
        let draft_row = &proposal.distributions[i];

        let draft_lp = draft_row[draft_token];
        let target_lp = target_row[draft_token];

        if accept(draft_lp, target_lp, rng) {
            emitted.push(draft_token);
            accepted += 1;
            continue;
        }

        // Rejection: resample from adjusted target distribution.
        let resampled = resample_from_adjusted_target(target_row, draft_row, rng)?;
        if resampled >= vocab {
            return Err(SpeculativeDecodingError::TokenOutOfRange {
                token: resampled,
                vocab_size: vocab,
            });
        }
        emitted.push(resampled);
        return Ok((accepted, emitted));
    }

    // All k accepted — draw bonus token from target's (k+1)-th distribution.
    let bonus_row = &target_scores.distributions[k];
    let bonus = sample_from_logprobs(bonus_row, rng)?;
    if bonus >= vocab {
        return Err(SpeculativeDecodingError::TokenOutOfRange {
            token: bonus,
            vocab_size: vocab,
        });
    }
    emitted.push(bonus);
    Ok((accepted, emitted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_is_sensible() {
        let c = SpeculativeDecoderConfig::default();
        assert_eq!(c.k, 4);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn config_k_zero_rejected() {
        let c = SpeculativeDecoderConfig::default().with_k(0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_builders_compose() {
        let c = SpeculativeDecoderConfig::default()
            .with_k(2)
            .with_cost_ratio(0.05)
            .with_eos(7);
        assert_eq!(c.k, 2);
        assert!((c.cost_ratio - 0.05).abs() < 1e-6);
        assert_eq!(c.eos_token, Some(7));
        assert!(c.stop_on_eos);
    }
}
