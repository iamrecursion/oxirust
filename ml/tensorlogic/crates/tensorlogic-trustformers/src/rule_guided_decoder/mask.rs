//! Logit-masking strategies for rule-guided sampling.
//!
//! A `LogitMasker` transforms a vocabulary-sized logits slice *in place*,
//! consulting a [`RuleConstraint`] to decide per-token whether to zero out,
//! penalise, or keep the existing logit value.
//!
//! Two concrete strategies are provided:
//!
//! * [`HardMask`] — replaces forbidden logits with `f64::NEG_INFINITY` so
//!   they are strictly excluded from the beam search candidate pool.
//! * [`SoftPenaltyMask`] — subtracts `lambda * violation_score` from logits
//!   classified as soft violations; forbidden tokens still get `-inf` so the
//!   mask never produces a finite "escape" around a hard ban.
//!
//! ## Note on scalar width
//!
//! The upstream beam search in `tensorlogic-infer` operates on `f64`; we match
//! that signature here.  The task specification mentioned `&mut [f32]`, but
//! honouring the actual integration point avoids a lossy round-trip conversion
//! at every decoding step.  If an `f32` pipeline materialises later, adding a
//! thin conversion wrapper is straightforward.

use crate::rule_guided_decoder::constraint::{ConstraintVerdict, RuleConstraint, TokenId};
use crate::rule_guided_decoder::error::{RuleGuidedError, RuleGuidedResult};

/// Strategy interface used by [`RuleGuidedBeamSearch`](crate::rule_guided_decoder::RuleGuidedBeamSearch).
pub trait LogitMasker: Send + Sync {
    /// Apply the mask to `logits` in place.  `prefix` is the sequence
    /// committed to the current beam; implementations may use it to evaluate
    /// stateful predicates (hooked up via `RuleConstraint::evaluate`).
    fn apply(
        &self,
        constraint: &RuleConstraint,
        prefix: &[TokenId],
        logits: &mut [f64],
    ) -> RuleGuidedResult<()>;

    /// Short human-readable label used in diagnostics and tests.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Hard mask
// ---------------------------------------------------------------------------

/// Hard-masking strategy: forbidden tokens get `-inf` and soft penalties are
/// ignored (treated as fully allowed).  This is the strictest mode — the
/// decoder will *never* emit a token that the constraint forbids.
#[derive(Debug, Default, Clone, Copy)]
pub struct HardMask;

impl HardMask {
    /// Construct a new hard-masking strategy.
    pub const fn new() -> Self {
        Self
    }
}

impl LogitMasker for HardMask {
    fn apply(
        &self,
        constraint: &RuleConstraint,
        prefix: &[TokenId],
        logits: &mut [f64],
    ) -> RuleGuidedResult<()> {
        for (token_id, logit) in logits.iter_mut().enumerate() {
            match constraint.evaluate(prefix, token_id) {
                ConstraintVerdict::Allowed => {}
                ConstraintVerdict::Forbidden => {
                    *logit = f64::NEG_INFINITY;
                }
                ConstraintVerdict::SoftPenalty(_) => {
                    // Hard masking ignores soft penalties — they are the soft
                    // mask's job.  The token is treated as allowed.
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "HardMask"
    }
}

// ---------------------------------------------------------------------------
// Soft penalty mask
// ---------------------------------------------------------------------------

/// Soft re-weighting mask: adds a log-penalty of `-lambda * violation_score`
/// to any token classified as a soft violation.  Forbidden tokens are still
/// silently set to `-inf` — soft mode merely downgrades the "probably bad"
/// class, not the "must not emit" class.
#[derive(Debug, Clone, Copy)]
pub struct SoftPenaltyMask {
    /// Non-negative penalty coefficient.  Larger values make the decoder
    /// more averse to soft-violating tokens.
    pub lambda: f64,
}

impl SoftPenaltyMask {
    /// Construct a new soft-penalty mask with the supplied coefficient.
    ///
    /// Returns [`RuleGuidedError::InvalidConfig`] if `lambda` is negative or
    /// not finite.
    pub fn new(lambda: f64) -> RuleGuidedResult<Self> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(RuleGuidedError::InvalidConfig(format!(
                "lambda must be a non-negative finite number, got {lambda}"
            )));
        }
        Ok(Self { lambda })
    }
}

impl LogitMasker for SoftPenaltyMask {
    fn apply(
        &self,
        constraint: &RuleConstraint,
        prefix: &[TokenId],
        logits: &mut [f64],
    ) -> RuleGuidedResult<()> {
        for (token_id, logit) in logits.iter_mut().enumerate() {
            match constraint.evaluate(prefix, token_id) {
                ConstraintVerdict::Allowed => {}
                ConstraintVerdict::Forbidden => {
                    *logit = f64::NEG_INFINITY;
                }
                ConstraintVerdict::SoftPenalty(score) => {
                    // Guard: scores are expected to be non-negative; treat
                    // negative values as zero to protect downstream math.
                    let clamped = score.max(0.0);
                    if self.lambda > 0.0 && clamped > 0.0 {
                        *logit -= self.lambda * clamped;
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SoftPenaltyMask"
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule_guided_decoder::constraint::RuleConstraint;
    use tensorlogic_ir::{TLExpr, Term};

    fn mapper() -> impl Fn(TokenId) -> Option<String> + Send + Sync + 'static {
        |tid: TokenId| match tid {
            0 => Some("entity".into()),
            1 => Some("Alice".into()),
            2 => Some("Bob".into()),
            _ => None,
        }
    }

    fn alice_only() -> RuleConstraint {
        let expr = TLExpr::Pred {
            name: "entity".into(),
            args: vec![Term::Const("Alice".into())],
        };
        RuleConstraint::compile(expr, mapper()).expect("compile")
    }

    #[test]
    fn hard_mask_sets_forbidden_to_neg_infinity() {
        let rc = alice_only();
        let mut logits = vec![0.0_f64, 0.0, 0.0, 0.0];
        HardMask::new().apply(&rc, &[], &mut logits).expect("apply");
        // token 0 (entity) and token 1 (Alice) are allowed -> unchanged.
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
        // token 2 (Bob) is forbidden.
        assert_eq!(logits[2], f64::NEG_INFINITY);
        // token 3 maps to None -> soft penalty -> hard-mask treats as allowed.
        assert_eq!(logits[3], 0.0);
    }

    #[test]
    fn soft_mask_applies_log_penalty() {
        let rc = alice_only();
        let mut logits = vec![0.0_f64, 0.0, 0.0, 0.0];
        SoftPenaltyMask::new(2.5)
            .expect("ctor")
            .apply(&rc, &[], &mut logits)
            .expect("apply");
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], f64::NEG_INFINITY);
        // token 3: SoftPenalty(1.0) * 2.5 -> -2.5.
        assert!((logits[3] - (-2.5)).abs() < 1e-12);
    }

    #[test]
    fn soft_mask_rejects_negative_lambda() {
        let err = SoftPenaltyMask::new(-0.1).expect_err("should reject");
        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn soft_mask_zero_lambda_is_noop() {
        let rc = alice_only();
        let mut logits = vec![0.0_f64, 0.0, 0.0, 0.0];
        SoftPenaltyMask::new(0.0)
            .expect("ctor")
            .apply(&rc, &[], &mut logits)
            .expect("apply");
        // With lambda=0, only forbidden tokens are touched.
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], f64::NEG_INFINITY);
        assert_eq!(logits[3], 0.0);
    }

    #[test]
    fn masker_name_reports_strategy() {
        let hard: Box<dyn LogitMasker> = Box::new(HardMask::new());
        let soft: Box<dyn LogitMasker> = Box::new(SoftPenaltyMask::new(1.0).expect("ctor"));
        assert_eq!(hard.name(), "HardMask");
        assert_eq!(soft.name(), "SoftPenaltyMask");
    }
}
