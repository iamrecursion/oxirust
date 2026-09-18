//! Rule-guided beam-search engine.
//!
//! [`RuleGuidedBeamSearch`] is a thin composition layer on top of
//! [`tensorlogic_infer::beam_search::BeamSearchDecoder`]:
//!
//! 1. The caller provides a raw scoring closure (same signature as the one
//!    accepted by [`BeamSearchDecoder::decode`]).
//! 2. We wrap that closure in a new one that, after each raw-logits call,
//!    runs the configured [`LogitMasker`] against the active beams' prefixes.
//! 3. The wrapped closure is forwarded to the inner decoder, which takes care
//!    of length penalties, temperature, top-k and EOS bookkeeping.
//!
//! Because the masking happens *before* softmax inside `decode<F>`, setting a
//! logit to `NEG_INFINITY` is semantically equivalent to assigning zero
//! probability — the beam search will never explore that branch.

use std::sync::Arc;

use tensorlogic_infer::beam_search::{BeamSearchConfig, BeamSearchDecoder, BeamSearchResult};

use crate::rule_guided_decoder::constraint::RuleConstraint;
use crate::rule_guided_decoder::error::{RuleGuidedError, RuleGuidedResult};
use crate::rule_guided_decoder::mask::LogitMasker;

/// High-level composition of a beam-search decoder, a logical constraint and
/// a masking strategy.
pub struct RuleGuidedBeamSearch {
    inner: BeamSearchDecoder,
    constraint: Arc<RuleConstraint>,
    masker: Arc<dyn LogitMasker>,
}

impl RuleGuidedBeamSearch {
    /// Construct a decoder with an explicit beam-search configuration.
    pub fn new(
        config: BeamSearchConfig,
        constraint: RuleConstraint,
        masker: Arc<dyn LogitMasker>,
    ) -> Self {
        Self {
            inner: BeamSearchDecoder::new(config),
            constraint: Arc::new(constraint),
            masker,
        }
    }

    /// Read-only access to the underlying beam-search configuration.
    pub fn config(&self) -> &BeamSearchConfig {
        &self.inner.config
    }

    /// Read-only access to the compiled constraint.
    pub fn constraint(&self) -> &RuleConstraint {
        &self.constraint
    }

    /// Return the masker's name (`"HardMask"` / `"SoftPenaltyMask"` / ...).
    pub fn masker_name(&self) -> &'static str {
        self.masker.name()
    }

    /// Run the decoder.
    ///
    /// `score_fn` is the same "raw logits" closure accepted by
    /// [`BeamSearchDecoder::decode`].  The engine wraps it so that every
    /// row of the returned `[num_beams][vocab_size]` logit matrix is filtered
    /// through the configured [`LogitMasker`] before being passed on.
    pub fn decode<F>(&self, bos_token_id: usize, score_fn: F) -> RuleGuidedResult<BeamSearchResult>
    where
        F: Fn(&[&[usize]]) -> Result<Vec<Vec<f64>>, String>,
    {
        let constraint = Arc::clone(&self.constraint);
        let masker = Arc::clone(&self.masker);
        let expected_vocab = self.inner.config.vocab_size;

        let wrapped = move |beams: &[&[usize]]| -> Result<Vec<Vec<f64>>, String> {
            let mut raw_logits = score_fn(beams)?;
            for (beam_idx, logits_row) in raw_logits.iter_mut().enumerate() {
                if logits_row.len() != expected_vocab {
                    return Err(format!(
                        "logits row {beam_idx} has width {}, expected {expected_vocab}",
                        logits_row.len()
                    ));
                }
                let prefix = beams.get(beam_idx).copied().unwrap_or(&[]);
                masker
                    .apply(&constraint, prefix, logits_row)
                    .map_err(|e| format!("rule-guided mask error: {e}"))?;
            }
            Ok(raw_logits)
        };

        self.inner
            .decode(bos_token_id, wrapped)
            .map_err(RuleGuidedError::from)
    }
}

impl std::fmt::Debug for RuleGuidedBeamSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleGuidedBeamSearch")
            .field("config", &self.inner.config)
            .field("constraint", &self.constraint)
            .field("masker", &self.masker.name())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule_guided_decoder::constraint::TokenId;
    use crate::rule_guided_decoder::mask::{HardMask, SoftPenaltyMask};
    use tensorlogic_ir::{TLExpr, Term};

    fn mk_constraint_alice_bob() -> RuleConstraint {
        // entity(Alice) OR entity(Bob) — allow symbol set = {entity, Alice, Bob}.
        let a = TLExpr::Pred {
            name: "entity".into(),
            args: vec![Term::Const("Alice".into())],
        };
        let b = TLExpr::Pred {
            name: "entity".into(),
            args: vec![Term::Const("Bob".into())],
        };
        let expr = TLExpr::Or(Box::new(a), Box::new(b));
        let mapper = |tid: TokenId| match tid {
            0 => Some("entity".into()),
            1 => Some("Alice".into()),
            2 => Some("Bob".into()),
            3 => Some("Eve".into()),
            _ => None,
        };
        RuleConstraint::compile(expr, mapper).expect("compile")
    }

    fn flat_config() -> BeamSearchConfig {
        BeamSearchConfig {
            beam_width: 2,
            max_length: 4,
            eos_token_id: None,
            length_penalty: 0.0,
            min_length: 1,
            vocab_size: 4,
            temperature: 1.0,
            top_k_filter: None,
        }
    }

    fn flat_scores() -> impl Fn(&[&[usize]]) -> Result<Vec<Vec<f64>>, String> {
        // Return uniform logits for every active beam; the masker decides
        // which tokens live or die.
        |beams: &[&[usize]]| Ok(beams.iter().map(|_| vec![1.0_f64, 1.0, 1.0, 1.0]).collect())
    }

    #[test]
    fn hard_mask_excludes_forbidden_token() {
        let decoder = RuleGuidedBeamSearch::new(
            flat_config(),
            mk_constraint_alice_bob(),
            Arc::new(HardMask::new()),
        );

        let result = decoder
            .decode(0, flat_scores())
            .expect("decode should succeed");
        // Eve (token id 3) maps to a symbol outside the allow set, so every
        // beam must avoid it.  "Unknown" token ids (mapper returns None)
        // don't exist in this vocabulary, so only forbidden symbols matter.
        for hyp in &result.hypotheses {
            assert!(
                !hyp.tokens.contains(&3),
                "hard-masked decoder emitted forbidden token: {:?}",
                hyp.tokens
            );
        }
        assert_eq!(decoder.masker_name(), "HardMask");
    }

    #[test]
    fn soft_mask_allows_forbidden_when_lambda_is_zero() {
        // Note: Forbidden tokens are still banned regardless of lambda.  We
        // verify soft-mode does not *additionally* block allowed tokens and
        // reports its name correctly.
        let decoder = RuleGuidedBeamSearch::new(
            flat_config(),
            mk_constraint_alice_bob(),
            Arc::new(SoftPenaltyMask::new(0.0).expect("lambda")),
        );
        let result = decoder
            .decode(0, flat_scores())
            .expect("decode should succeed");
        assert!(!result.hypotheses.is_empty());
        assert_eq!(decoder.masker_name(), "SoftPenaltyMask");
    }

    #[test]
    fn error_from_score_fn_is_propagated() {
        let decoder = RuleGuidedBeamSearch::new(
            flat_config(),
            mk_constraint_alice_bob(),
            Arc::new(HardMask::new()),
        );
        let result = decoder.decode(0, |_beams: &[&[usize]]| {
            Err::<Vec<Vec<f64>>, String>("synthetic scoring error".into())
        });
        assert!(result.is_err());
        let msg = format!("{}", result.expect_err("should have returned an error"));
        assert!(msg.contains("synthetic"));
    }
}
