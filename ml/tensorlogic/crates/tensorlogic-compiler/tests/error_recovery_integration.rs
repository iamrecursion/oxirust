//! Integration test for partial error recovery (tolerant compilation).
//!
//! Verifies that a mixed program of well-formed and semantically-broken
//! expressions produces a [`PartialCompilationResult`] with:
//!
//! * three `Some(graph)` slots for the three good expressions;
//! * one `None` slot for the broken expression;
//! * exactly one `Error`-severity diagnostic tied to the broken expression;
//! * no `Fatal` diagnostics;
//! * `aborted == false` — tolerant compilation proceeds past the error.

use tensorlogic_compiler::error_recovery::{
    compile_tolerant, compile_tolerant_with_strategy, PartialCompilationResult, RecoveryStrategy,
    Severity,
};
use tensorlogic_ir::{TLExpr, Term};

/// A good expression — a simple unary predicate `name(x)`.
fn good(name: &str) -> TLExpr {
    TLExpr::pred(name, vec![Term::var("x")])
}

/// A semantically-broken expression: an existential quantifier over a
/// domain that was never registered. The strict compiler returns `Err(..)`
/// from `ctx.bind_var` (domain not found).
fn broken() -> TLExpr {
    TLExpr::exists(
        "y",
        "MissingDomain",
        TLExpr::pred("q", vec![Term::var("y")]),
    )
}

#[test]
fn mixed_program_three_good_plus_one_broken() {
    let program = vec![good("a"), good("b"), broken(), good("d")];
    let res: PartialCompilationResult = compile_tolerant(&program);

    // Structural shape: 4 slots total, one None.
    assert_eq!(res.graphs.len(), 4);
    assert_eq!(res.success_count(), 3, "three good expressions compile");
    assert_eq!(res.failure_count(), 1, "one broken expression skipped");
    assert_eq!(res.failures(), vec![2]);

    // Diagnostics: exactly one Error, zero Fatals.
    let errors = res.diagnostics.errors();
    assert_eq!(errors.len(), 1, "exactly one Error diagnostic");
    assert_eq!(
        errors[0].expression_index,
        Some(2),
        "diagnostic tied to broken expression's index"
    );
    assert_eq!(res.diagnostics.count_of(Severity::Fatal), 0);

    // The driver did NOT abort.
    assert!(!res.aborted);
    assert!(res.aborted_at.is_none());
}

#[test]
fn abort_on_any_strategy_stops_at_first_error() {
    let program = vec![good("a"), broken(), good("c"), good("d")];
    let res = compile_tolerant_with_strategy(&program, RecoveryStrategy::AbortOnAny);

    // First slot compiles, broken slot fails, remaining slots are None.
    assert_eq!(res.graphs.len(), 4);
    assert!(res.graphs[0].is_some());
    assert!(res.graphs[1].is_none());
    assert!(res.graphs[2].is_none());
    assert!(res.graphs[3].is_none());

    assert!(res.aborted);
    assert_eq!(res.aborted_at, Some(1));
    assert_eq!(res.diagnostics.errors().len(), 1);
}

#[test]
fn all_good_program_yields_no_diagnostics() {
    let program = vec![good("a"), good("b"), good("c")];
    let res = compile_tolerant(&program);
    assert!(res.is_all_success());
    assert!(res.diagnostics.is_empty());
    assert!(!res.aborted);
}
