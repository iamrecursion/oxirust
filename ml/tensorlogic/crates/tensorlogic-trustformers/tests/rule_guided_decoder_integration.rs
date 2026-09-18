//! End-to-end integration test for the rule-guided decoder.
//!
//! Runs the full pipeline with a hand-built logit stub rather than a live
//! factor-graph model:
//!
//! * Compile a `TLExpr::Or(entity(Alice), entity(Bob))` constraint.
//! * Attach a [`HardMask`] and a [`SoftPenaltyMask`] variant.
//! * Feed synthetic logits through [`RuleGuidedBeamSearch::decode`] and
//!   verify the returned hypotheses respect the constraint.

use std::sync::Arc;

use tensorlogic_infer::beam_search::BeamSearchConfig;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_trustformers::rule_guided_decoder::{
    HardMask, LogitMasker, RuleConstraint, RuleGuidedBeamSearch, SoftPenaltyMask, TokenId,
};

/// Vocabulary: 0 = "entity", 1 = "Alice", 2 = "Bob", 3 = "Eve", 4 = unknown.
fn mapper() -> impl Fn(TokenId) -> Option<String> + Send + Sync + 'static {
    |tid: TokenId| match tid {
        0 => Some("entity".into()),
        1 => Some("Alice".into()),
        2 => Some("Bob".into()),
        3 => Some("Eve".into()),
        _ => None,
    }
}

/// Constraint: allow {entity, Alice, Bob}.
fn constraint() -> RuleConstraint {
    let a = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Alice".into())],
    };
    let b = TLExpr::Pred {
        name: "entity".into(),
        args: vec![Term::Const("Bob".into())],
    };
    RuleConstraint::compile(TLExpr::Or(Box::new(a), Box::new(b)), mapper())
        .expect("constraint compiles")
}

fn base_config() -> BeamSearchConfig {
    BeamSearchConfig {
        beam_width: 3,
        max_length: 5,
        eos_token_id: None,
        length_penalty: 0.0,
        min_length: 1,
        vocab_size: 5,
        temperature: 1.0,
        top_k_filter: None,
    }
}

/// Synthetic scoring function returning biased logits that prefer the banned
/// "Eve" token (token id 3).  The mask must rescue the decoder from picking it.
fn biased_score_fn() -> impl Fn(&[&[usize]]) -> Result<Vec<Vec<f64>>, String> {
    |beams: &[&[usize]]| {
        let row = vec![0.5_f64, 0.5, 0.5, 5.0, 0.0];
        Ok(beams.iter().map(|_| row.clone()).collect())
    }
}

#[test]
fn hard_mask_integration_never_emits_forbidden_token() {
    let decoder = RuleGuidedBeamSearch::new(base_config(), constraint(), Arc::new(HardMask::new()));
    let result = decoder.decode(0, biased_score_fn()).expect("decode");

    assert!(
        !result.hypotheses.is_empty(),
        "decoder returned no hypotheses"
    );
    assert_eq!(decoder.masker_name(), "HardMask");

    for hyp in &result.hypotheses {
        for tok in &hyp.tokens {
            assert!(
                (0..=2).contains(tok),
                "hard mask leaked banned token {tok}: {:?}",
                hyp.tokens
            );
        }
    }
    assert!(result.best_score.is_finite() || result.hypotheses.len() == 1);
}

#[test]
fn soft_mask_integration_prefers_allowed_tokens() {
    // With a large lambda, soft-penalised "unknown" tokens must be avoided,
    // while forbidden tokens (Eve) are still banned.
    let masker: Arc<dyn LogitMasker> =
        Arc::new(SoftPenaltyMask::new(50.0).expect("positive lambda"));
    let decoder = RuleGuidedBeamSearch::new(base_config(), constraint(), masker);
    let result = decoder.decode(0, biased_score_fn()).expect("decode");

    assert_eq!(decoder.masker_name(), "SoftPenaltyMask");
    for hyp in &result.hypotheses {
        for tok in &hyp.tokens {
            assert_ne!(
                *tok, 3,
                "soft mask should still ban hard-forbidden token Eve: {:?}",
                hyp.tokens
            );
        }
    }
}

#[test]
fn hard_and_soft_modes_agree_on_forbidden_ban() {
    // Both strategies must exclude hard-forbidden tokens.
    let hard_decoder =
        RuleGuidedBeamSearch::new(base_config(), constraint(), Arc::new(HardMask::new()));
    let soft_decoder = RuleGuidedBeamSearch::new(
        base_config(),
        constraint(),
        Arc::new(SoftPenaltyMask::new(1.0).expect("lambda")),
    );

    let hard_result = hard_decoder.decode(0, biased_score_fn()).expect("hard");
    let soft_result = soft_decoder.decode(0, biased_score_fn()).expect("soft");

    let hard_has_eve = hard_result.hypotheses.iter().any(|h| h.tokens.contains(&3));
    let soft_has_eve = soft_result.hypotheses.iter().any(|h| h.tokens.contains(&3));

    assert!(!hard_has_eve, "hard decoder emitted Eve");
    assert!(!soft_has_eve, "soft decoder emitted Eve");
}
