//! Cross-submodule unit tests for the rule-guided decoder.
//!
//! Submodule-specific tests live next to their respective implementations
//! (`constraint.rs`, `mask.rs`, `engine.rs`).  The cases here exercise
//! interactions that span multiple submodules — e.g. the interplay between a
//! conjunction constraint and the soft-penalty mask.

#![cfg(test)]

use std::sync::Arc;

use tensorlogic_infer::beam_search::BeamSearchConfig;
use tensorlogic_ir::{TLExpr, Term};

use super::constraint::{ConstraintVerdict, RuleConstraint, TokenId};
use super::engine::RuleGuidedBeamSearch;
use super::mask::{HardMask, LogitMasker, SoftPenaltyMask};

fn mapper() -> impl Fn(TokenId) -> Option<String> + Send + Sync + 'static {
    |tid: TokenId| match tid {
        0 => Some("entity".into()),
        1 => Some("Alice".into()),
        2 => Some("Bob".into()),
        3 => Some("Eve".into()),
        _ => None,
    }
}

#[test]
fn conjunction_constraint_hard_masks_all_non_shared_symbols() {
    // entity(Alice) AND entity(Bob) has allow set = {entity}.
    let a = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Alice".into())],
    };
    let b = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Bob".into())],
    };
    let expr = TLExpr::And(Box::new(a), Box::new(b));
    let rc = RuleConstraint::compile(expr, mapper()).expect("compile");

    let mut logits = vec![0.0_f64; 4];
    HardMask::new().apply(&rc, &[], &mut logits).expect("apply");

    // Only token 0 ("entity") survives.
    assert_eq!(logits[0], 0.0);
    assert_eq!(logits[1], f64::NEG_INFINITY);
    assert_eq!(logits[2], f64::NEG_INFINITY);
    assert_eq!(logits[3], f64::NEG_INFINITY);
}

#[test]
fn soft_mask_math_matches_lambda_times_violation() {
    let expr = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Alice".into())],
    };
    let rc = RuleConstraint::compile(expr, mapper()).expect("compile");

    // Token 4 maps to None -> SoftPenalty(1.0).  With lambda = 0.75, the
    // penalty applied is 0.75 * 1.0 = 0.75.
    let mut logits = vec![3.0_f64; 5];
    SoftPenaltyMask::new(0.75)
        .expect("ctor")
        .apply(&rc, &[], &mut logits)
        .expect("apply");
    // token 0 (entity): Allowed -> unchanged
    // token 1 (Alice):  Allowed -> unchanged
    // token 2 (Bob):    Forbidden -> -inf
    // token 3 (Eve):    Forbidden -> -inf
    // token 4 (None):   SoftPenalty(1.0) -> 3.0 - 0.75
    assert!((logits[0] - 3.0).abs() < 1e-12);
    assert!((logits[1] - 3.0).abs() < 1e-12);
    assert_eq!(logits[2], f64::NEG_INFINITY);
    assert_eq!(logits[3], f64::NEG_INFINITY);
    assert!((logits[4] - 2.25).abs() < 1e-12);
}

#[test]
fn predicate_allow_list_evaluates_prefix_independent() {
    // Confirms evaluate() signature respects the (prefix, candidate) API:
    // the current compiler ignores the prefix, so both calls yield the same
    // verdict.  This pins down the contract for stateful predicates.
    let expr = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Alice".into())],
    };
    let rc = RuleConstraint::compile(expr, mapper()).expect("compile");
    assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Allowed);
    assert_eq!(rc.evaluate(&[7, 42, 99], 1), ConstraintVerdict::Allowed);
}

#[test]
fn integration_smoke_with_stubbed_decoder() {
    // End-to-end sanity: a tiny vocabulary where the constraint is
    // "entity(Alice) OR entity(Bob)".  The decoder must emit only
    // {entity, Alice, Bob} tokens under hard masking.
    let a = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Alice".into())],
    };
    let b = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Bob".into())],
    };
    let expr = TLExpr::Or(Box::new(a), Box::new(b));
    let rc = RuleConstraint::compile(expr, mapper()).expect("compile");

    let config = BeamSearchConfig {
        beam_width: 2,
        max_length: 3,
        eos_token_id: None,
        length_penalty: 0.0,
        min_length: 1,
        vocab_size: 4,
        temperature: 1.0,
        top_k_filter: None,
    };

    let decoder = RuleGuidedBeamSearch::new(config, rc, Arc::new(HardMask::new()));

    let result = decoder
        .decode(0, |beams: &[&[usize]]| {
            Ok(beams.iter().map(|_| vec![1.0_f64, 0.5, 0.5, 0.5]).collect())
        })
        .expect("decode");

    assert!(!result.hypotheses.is_empty());
    for hyp in &result.hypotheses {
        for tok in &hyp.tokens {
            assert!(
                (0..=2).contains(tok),
                "hard-masked decoder emitted banned token {tok}: {:?}",
                hyp.tokens
            );
        }
    }
}

#[test]
fn negative_case_when_every_symbol_is_banned_decoder_still_terminates() {
    // entity(Alice) AND user(Charlie) — disjoint predicate names,
    // intersection empty, every symbol except the impossible "both entity
    // and user" symbol is forbidden.  Verify the decoder does not panic and
    // produces at least one (possibly low-score) hypothesis.
    let a = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Alice".into())],
    };
    let c = TLExpr::Pred {
        name: "user".into(),
        args: vec![Term::Const("Charlie".into())],
    };
    let expr = TLExpr::And(Box::new(a), Box::new(c));
    let rc = RuleConstraint::compile(expr, mapper()).expect("compile");

    // Soft mask must succeed even when allow set is empty.
    let masker: Arc<dyn LogitMasker> = Arc::new(SoftPenaltyMask::new(0.1).expect("ctor"));
    let decoder = RuleGuidedBeamSearch::new(
        BeamSearchConfig {
            beam_width: 1,
            max_length: 2,
            eos_token_id: None,
            length_penalty: 0.0,
            min_length: 1,
            vocab_size: 4,
            temperature: 1.0,
            top_k_filter: None,
        },
        rc,
        masker,
    );

    let result = decoder
        .decode(0, |beams: &[&[usize]]| {
            Ok(beams.iter().map(|_| vec![0.0_f64; 4]).collect())
        })
        .expect("decode should not crash");
    // At least one hypothesis must be returned (beams all finalised at
    // max_length).
    assert!(!result.hypotheses.is_empty());
}

#[test]
fn constraint_reports_source_and_support_flag() {
    let expr = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Alice".into())],
    };
    let rc = RuleConstraint::compile(expr.clone(), mapper()).expect("compile");
    assert!(rc.is_supported());
    assert_eq!(rc.source(), &expr);

    // An unsupported expression (Exists requires domain enumeration) still
    // compiles but marks itself as unsupported and evaluates to SoftPenalty(0.0).
    let unsupported = TLExpr::Exists {
        var: "x".to_string(),
        domain: "Person".to_string(),
        body: Box::new(expr),
    };
    let rc2 = RuleConstraint::compile(unsupported, mapper()).expect("compile");
    assert!(!rc2.is_supported());
    assert_eq!(rc2.evaluate(&[], 1), ConstraintVerdict::SoftPenalty(0.0));
}
