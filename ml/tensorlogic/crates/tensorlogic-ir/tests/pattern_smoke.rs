//! Smoke tests for `TLExpr::Match` + `MatchPattern` in tensorlogic-ir.
//!
//! Covers: pattern Display, match_expr builder, validation (no-arms and
//! non-wildcard tail rejection), binary serde round-trip, S-expression serde
//! round-trip, Display contains "match", and pretty_print output.

use tensorlogic_ir::{
    from_binary, from_sexpr, pretty_print, to_binary, to_sexpr, MatchPattern, PrettyConfig, TLExpr,
};

// ---------------------------------------------------------------------------
// 1. MatchPattern Display
// ---------------------------------------------------------------------------

#[test]
fn match_pattern_display_const_symbol() {
    assert_eq!(MatchPattern::ConstSymbol("ok".into()).to_string(), ":ok");
    assert_eq!(MatchPattern::ConstSymbol("foo".into()).to_string(), ":foo");
}

#[test]
fn match_pattern_display_const_number() {
    assert_eq!(MatchPattern::ConstNumber(2.71).to_string(), "2.71");
    assert_eq!(MatchPattern::ConstNumber(0.0).to_string(), "0");
}

#[test]
fn match_pattern_display_wildcard() {
    assert_eq!(MatchPattern::Wildcard.to_string(), "_");
}

// ---------------------------------------------------------------------------
// 2. Validation — no arms
// ---------------------------------------------------------------------------

#[test]
fn match_expr_validation_rejects_no_arms() {
    let scrutinee = TLExpr::SymbolLiteral("x".into());
    let expr = TLExpr::match_expr(scrutinee, vec![]);
    let result = expr.validate_arity();
    assert!(result.is_err(), "expected error for empty arms, got Ok");
    let msg = result.expect_err("already verified is_err");
    assert!(
        msg.contains("arm"),
        "error message should mention 'arm', got: {msg}",
    );
}

// ---------------------------------------------------------------------------
// 3. Validation — non-wildcard last arm
// ---------------------------------------------------------------------------

#[test]
fn match_expr_validation_rejects_non_wildcard_tail() {
    let scrutinee = TLExpr::SymbolLiteral("x".into());
    let arms = vec![
        (MatchPattern::ConstSymbol("a".into()), TLExpr::Constant(1.0)),
        (MatchPattern::ConstSymbol("b".into()), TLExpr::Constant(2.0)),
        // last arm is NOT wildcard — should fail
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    let result = expr.validate_arity();
    assert!(
        result.is_err(),
        "expected error when last arm is not Wildcard, got Ok",
    );
    let msg = result.expect_err("already verified is_err");
    assert!(
        msg.to_lowercase().contains("wildcard"),
        "error message should mention 'Wildcard', got: {msg}",
    );
}

// ---------------------------------------------------------------------------
// 4. Validation — valid match passes
// ---------------------------------------------------------------------------

#[test]
fn match_expr_validation_accepts_valid_arms() {
    let scrutinee = TLExpr::SymbolLiteral("color".into());
    let arms = vec![
        (
            MatchPattern::ConstSymbol("red".into()),
            TLExpr::Constant(1.0),
        ),
        (
            MatchPattern::ConstSymbol("blue".into()),
            TLExpr::Constant(2.0),
        ),
        (MatchPattern::Wildcard, TLExpr::Constant(0.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    expr.validate_arity()
        .expect("valid match should pass validation");
}

// ---------------------------------------------------------------------------
// 5. Binary serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn match_expr_binary_roundtrip() {
    let scrutinee = TLExpr::SymbolLiteral("status".into());
    let arms = vec![
        (
            MatchPattern::ConstSymbol("ok".into()),
            TLExpr::Constant(1.0),
        ),
        (MatchPattern::ConstNumber(42.0), TLExpr::Constant(2.0)),
        (MatchPattern::Wildcard, TLExpr::Constant(-1.0)),
    ];
    let original = TLExpr::match_expr(scrutinee, arms);

    let bytes = to_binary(&original);
    let restored = from_binary(&bytes).expect("binary deserialization failed");

    assert_eq!(
        original, restored,
        "binary round-trip must produce structurally equal expression",
    );
}

// ---------------------------------------------------------------------------
// 6. S-expression serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn match_expr_sexpr_roundtrip() {
    let scrutinee = TLExpr::SymbolLiteral("mode".into());
    let arms = vec![
        (
            MatchPattern::ConstSymbol("fast".into()),
            TLExpr::Constant(10.0),
        ),
        (MatchPattern::Wildcard, TLExpr::Constant(1.0)),
    ];
    let original = TLExpr::match_expr(scrutinee, arms);

    let text = to_sexpr(&original);
    let restored = from_sexpr(&text).expect("s-expression deserialization failed");

    assert_eq!(
        original, restored,
        "s-expr round-trip must produce structurally equal expression",
    );
}

// ---------------------------------------------------------------------------
// 7. Display contains "match"
// ---------------------------------------------------------------------------

#[test]
fn match_expr_display_contains_match_keyword() {
    let scrutinee = TLExpr::SymbolLiteral("x".into());
    let arms = vec![
        (MatchPattern::ConstSymbol("a".into()), TLExpr::Constant(1.0)),
        (MatchPattern::Wildcard, TLExpr::Constant(0.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    let rendered = format!("{expr}");
    assert!(
        rendered.contains("match"),
        "Display output should contain 'match', got: {rendered}",
    );
}

// ---------------------------------------------------------------------------
// 8. pretty_print contains "match"
// ---------------------------------------------------------------------------

#[test]
fn match_expr_pretty_print_contains_match_keyword() {
    let scrutinee = TLExpr::SymbolLiteral("x".into());
    let arms = vec![
        (MatchPattern::ConstNumber(3.0), TLExpr::Constant(99.0)),
        (MatchPattern::Wildcard, TLExpr::Constant(0.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    let pp = pretty_print(&expr, &PrettyConfig::default());
    assert!(
        pp.contains("match"),
        "pretty_print output should contain 'match', got: {pp}",
    );
}
