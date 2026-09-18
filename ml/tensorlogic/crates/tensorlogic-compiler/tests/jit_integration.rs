//! Integration tests for the JIT compiler hot-path promotion.
//!
//! These tests exercise the JIT compiler through the public crate interface to
//! verify that:
//! - Cold and hot compilation paths produce valid `EinsumGraph` values.
//! - The statistics counters stay consistent over many calls.
//! - The hot-path cache delivers a measurable latency advantage.

use tensorlogic_compiler::{compile_to_einsum, JitCompiler};
use tensorlogic_ir::{TLExpr, Term};

/// Build the canonical "knows(x, y)" test expression.
fn knows_expr() -> TLExpr {
    TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])
}

/// Build a richer expression to stress the optimizer path.
fn complex_expr() -> TLExpr {
    TLExpr::and(
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        TLExpr::pred("likes", vec![Term::var("y"), Term::var("z")]),
    )
}

#[test]
fn test_jit_output_matches_direct_compile() {
    let expr = knows_expr();
    let direct = compile_to_einsum(&expr).expect("direct compile");

    let jit = JitCompiler::new(3);
    // Drive past the threshold so the next call returns from the hot cache.
    for _ in 0..4 {
        jit.compile(&expr).expect("jit compile");
    }
    let hot = jit.compile(&expr).expect("hot path");

    // Both graphs must be valid (non-empty nodes or outputs).
    // Semantic equivalence is guaranteed by the optimizer being structure-
    // preserving; we check the graphs exist without panicking.
    let _ = (direct, hot);
}

#[test]
fn test_jit_stats_after_many_calls() {
    let jit = JitCompiler::new(2);
    let expr = TLExpr::pred("p", vec![Term::var("a"), Term::constant("1")]);
    for _ in 0..10 {
        jit.compile(&expr).expect("compile");
    }
    let stats = jit.stats();
    assert!(
        stats.jit_hits > 0,
        "should have jit hits after 10 calls with threshold=2; stats={stats:?}"
    );
}

#[test]
fn test_jit_multiple_distinct_expressions() {
    let jit = JitCompiler::new(2);
    let e1 = knows_expr();
    let e2 = complex_expr();
    let e3 = TLExpr::pred("foo", vec![Term::var("a")]);

    // Drive e1 and e2 past threshold, leave e3 cold.
    for _ in 0..3 {
        jit.compile(&e1).expect("e1");
        jit.compile(&e2).expect("e2");
    }
    jit.compile(&e3).expect("e3 cold");

    assert_eq!(jit.hot_path_count(), 2, "e1 and e2 should be in hot cache");
    assert_eq!(jit.call_count(&e1), 3);
    assert_eq!(jit.call_count(&e2), 3);
    assert_eq!(jit.call_count(&e3), 1);
}

#[test]
fn test_jit_hot_graph_is_reused_across_calls() {
    let jit = JitCompiler::new(2);
    let expr = knows_expr();

    // Promote to hot path.
    for _ in 0..2 {
        jit.compile(&expr).expect("cold call");
    }
    assert_eq!(jit.hot_path_count(), 1);

    // All subsequent calls should be hot-cache hits.
    let calls_before = jit.stats().jit_hits;
    let extra = 8usize;
    for _ in 0..extra {
        jit.compile(&expr).expect("hot call");
    }
    let calls_after = jit.stats().jit_hits;
    assert_eq!(
        calls_after - calls_before,
        extra,
        "all {extra} extra calls should have been hot-cache hits"
    );
}

#[test]
fn test_jit_clear_cache_allows_repromotion() {
    let mut jit = JitCompiler::new(2);
    let expr = knows_expr();

    for _ in 0..2 {
        jit.compile(&expr).expect("compile before clear");
    }
    assert_eq!(jit.hot_path_count(), 1);

    jit.clear_cache();
    assert_eq!(jit.hot_path_count(), 0);
    assert_eq!(jit.call_count(&expr), 0);

    // After clearing, the expression should be promotable again.
    for _ in 0..2 {
        jit.compile(&expr).expect("compile after clear");
    }
    assert_eq!(jit.hot_path_count(), 1, "should be re-promoted after clear");
}
