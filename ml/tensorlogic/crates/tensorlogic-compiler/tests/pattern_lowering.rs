//! Integration tests for `TLExpr::Match` lowering through the bytecode compiler.
//!
//! Each test builds a `Match` expression, compiles it to bytecode, executes it,
//! and asserts the returned `VmValue` matches the expected arm body.

use tensorlogic_compiler::{compile, execute, VmEnv, VmValue};
use tensorlogic_ir::{from_binary, to_binary, MatchPattern, TLExpr};

// ---------------------------------------------------------------------------
// Helper: run a Match expression and return the VM result.
// ---------------------------------------------------------------------------

fn run(expr: &TLExpr) -> VmValue {
    let program = compile(expr).expect("compile failed");
    execute(&program, &VmEnv::new()).expect("execute failed")
}

// ---------------------------------------------------------------------------
// 1. ConstSymbol arm hits — scrutinee matches first arm
// ---------------------------------------------------------------------------

#[test]
fn symbol_arm_matches_first() {
    // match :ok { :ok => 1.0 | _ => 0.0 }
    let scrutinee = TLExpr::SymbolLiteral("ok".into());
    let arms = vec![
        (
            MatchPattern::ConstSymbol("ok".into()),
            TLExpr::Constant(1.0),
        ),
        (MatchPattern::Wildcard, TLExpr::Constant(0.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    assert_eq!(run(&expr), VmValue::Num(1.0));
}

// ---------------------------------------------------------------------------
// 2. ConstSymbol arm misses — falls through to wildcard
// ---------------------------------------------------------------------------

#[test]
fn symbol_arm_falls_through_to_wildcard() {
    // match :err { :ok => 1.0 | _ => 0.0 }
    let scrutinee = TLExpr::SymbolLiteral("err".into());
    let arms = vec![
        (
            MatchPattern::ConstSymbol("ok".into()),
            TLExpr::Constant(1.0),
        ),
        (MatchPattern::Wildcard, TLExpr::Constant(0.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    assert_eq!(run(&expr), VmValue::Num(0.0));
}

// ---------------------------------------------------------------------------
// 3. ConstNumber arm hits
// ---------------------------------------------------------------------------

#[test]
fn number_arm_matches() {
    // match 42.0 { 42.0 => 7.0 | _ => -1.0 }
    let scrutinee = TLExpr::Constant(42.0);
    let arms = vec![
        (MatchPattern::ConstNumber(42.0), TLExpr::Constant(7.0)),
        (MatchPattern::Wildcard, TLExpr::Constant(-1.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    assert_eq!(run(&expr), VmValue::Num(7.0));
}

// ---------------------------------------------------------------------------
// 4. ConstNumber arm misses — wildcard fallthrough
// ---------------------------------------------------------------------------

#[test]
fn number_arm_falls_through_to_wildcard() {
    // match 5.0 { 42.0 => 7.0 | _ => -1.0 }
    let scrutinee = TLExpr::Constant(5.0);
    let arms = vec![
        (MatchPattern::ConstNumber(42.0), TLExpr::Constant(7.0)),
        (MatchPattern::Wildcard, TLExpr::Constant(-1.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    assert_eq!(run(&expr), VmValue::Num(-1.0));
}

// ---------------------------------------------------------------------------
// 5. Three-arm cascade — matches second arm
// ---------------------------------------------------------------------------

#[test]
fn three_arm_cascade_matches_second() {
    // match :b { :a => 1.0 | :b => 2.0 | _ => 3.0 }
    let scrutinee = TLExpr::SymbolLiteral("b".into());
    let arms = vec![
        (MatchPattern::ConstSymbol("a".into()), TLExpr::Constant(1.0)),
        (MatchPattern::ConstSymbol("b".into()), TLExpr::Constant(2.0)),
        (MatchPattern::Wildcard, TLExpr::Constant(3.0)),
    ];
    let expr = TLExpr::match_expr(scrutinee, arms);
    assert_eq!(run(&expr), VmValue::Num(2.0));
}

// ---------------------------------------------------------------------------
// 6. Wildcard-only arm — always returns wildcard body
// ---------------------------------------------------------------------------

#[test]
fn wildcard_only_arm_always_returns_body() {
    // match :anything { _ => 99.0 }
    let scrutinee = TLExpr::SymbolLiteral("anything".into());
    let arms = vec![(MatchPattern::Wildcard, TLExpr::Constant(99.0))];
    let expr = TLExpr::match_expr(scrutinee, arms);
    assert_eq!(run(&expr), VmValue::Num(99.0));
}

// ---------------------------------------------------------------------------
// 7. Binary serde round-trip: serialize the Match expr and re-run
// ---------------------------------------------------------------------------

#[test]
fn match_expr_binary_serde_and_execute() {
    let scrutinee = TLExpr::SymbolLiteral("ping".into());
    let arms = vec![
        (
            MatchPattern::ConstSymbol("ping".into()),
            TLExpr::Constant(1.0),
        ),
        (MatchPattern::Wildcard, TLExpr::Constant(0.0)),
    ];
    let original = TLExpr::match_expr(scrutinee, arms);

    let bytes = to_binary(&original);
    let restored = from_binary(&bytes).expect("binary deserialization failed");

    // Structural equality
    assert_eq!(original, restored);
    // Restored expression compiles and executes identically
    assert_eq!(run(&restored), VmValue::Num(1.0));
}
