//! Let-Inlining pass for TLExpr trees.
//!
//! This module provides a let-inlining optimization pass that substitutes
//! `Let`-bound variables into their usage sites, reducing the number of
//! explicit bindings and enabling downstream passes to work with smaller,
//! simpler trees.
//!
//! # Inlining Strategy
//!
//! Three independent criteria control whether a binding is inlined:
//!
//! 1. **Single-use inlining** (`inline_single_use`): If the bound variable
//!    appears free exactly once in the body, inlining is always safe — it
//!    does not duplicate work.
//!
//! 2. **Constant inlining** (`inline_constants`): If the bound *value* is a
//!    `Constant(f64)`, it is cheap to duplicate so we always inline regardless
//!    of use count.
//!
//! 3. **Variable-alias inlining** (`inline_vars`): If the bound value is a
//!    zero-argument `Pred` (which serves as a variable reference in TLExpr),
//!    we inline it unconditionally because the binding is just a rename.
//!
//! A `max_inline_depth` guard prevents inlining of deeply nested sub-trees
//! to keep code-size growth bounded.
//!
//! # Correctness
//!
//! Substitution respects capture: when descending into a binder that re-uses
//! the same variable name, substitution stops at that binder boundary.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_compiler::inline::{LetInliner, InlineConfig};
//! use tensorlogic_ir::TLExpr;
//!
//! let inliner = LetInliner::with_default();
//! // Let x = 5.0 in Add(x, x)  →  Add(5.0, 5.0)
//! let expr = TLExpr::let_binding(
//!     "x",
//!     TLExpr::Constant(5.0),
//!     TLExpr::add(
//!         TLExpr::pred("x", vec![]),
//!         TLExpr::pred("x", vec![]),
//!     ),
//! );
//! let (result, stats) = inliner.run(expr);
//! assert_eq!(stats.constant_inlines, 1);
//! ```

pub mod config;
pub mod helpers;
pub mod substitute;
pub mod traversal;

pub use config::{InlineConfig, InlineStats};
pub use traversal::LetInliner;

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::TLExpr;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Create a zero-argument predicate (variable reference in let bodies).
    fn var(name: &str) -> TLExpr {
        TLExpr::pred(name, vec![])
    }

    /// Build a deeply nested Add chain of depth `depth` over `Constant(1.0)`.
    fn deep_add(depth: usize) -> TLExpr {
        if depth == 0 {
            return TLExpr::Constant(1.0);
        }
        TLExpr::add(deep_add(depth - 1), TLExpr::Constant(1.0))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // InlineStats tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_inline_stats_default() {
        let stats = InlineStats::default();
        assert_eq!(stats.single_use_inlines, 0);
        assert_eq!(stats.constant_inlines, 0);
        assert_eq!(stats.variable_inlines, 0);
        assert_eq!(stats.total(), 0);
        assert_eq!(stats.nodes_before, 0);
        assert_eq!(stats.nodes_after, 0);
        assert_eq!(stats.passes, 0);
    }

    #[test]
    fn test_inline_stats_summary_nonempty() {
        let stats = InlineStats {
            single_use_inlines: 2,
            constant_inlines: 3,
            variable_inlines: 1,
            nodes_before: 20,
            nodes_after: 14,
            passes: 2,
        };
        let summary = stats.summary();
        assert!(summary.contains("2 passes"));
        assert!(summary.contains("14/20"));
        assert!(summary.contains("2 single-use"));
        assert!(summary.contains("3 constant"));
        assert!(summary.contains("1 variable-alias"));
    }

    #[test]
    fn test_total_inlines() {
        let stats = InlineStats {
            single_use_inlines: 4,
            constant_inlines: 5,
            variable_inlines: 3,
            ..Default::default()
        };
        assert_eq!(stats.total(), 12);
    }

    #[test]
    fn test_reduction_pct() {
        let stats = InlineStats {
            nodes_before: 100,
            nodes_after: 60,
            ..Default::default()
        };
        // (100 - 60) / 100 * 100 = 40%
        let pct = stats.reduction_pct();
        assert!((pct - 40.0).abs() < 1e-9, "expected ~40%, got {pct}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // InlineConfig tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_inline_config_default() {
        let cfg = InlineConfig::default();
        assert!(cfg.inline_single_use);
        assert!(cfg.inline_constants);
        assert!(cfg.inline_vars);
        assert_eq!(cfg.max_passes, 20);
        assert_eq!(cfg.max_inline_depth, 10);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LetInliner construction
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_inliner_with_default() {
        let inliner = LetInliner::with_default();
        // Just verify it constructs without panic and default config is sound.
        assert!(inliner.config.inline_single_use);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // count_free_occurrences tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_count_free_occurrences_zero() {
        // var "z" does not appear in pred("p", [])
        let expr = var("p");
        assert_eq!(LetInliner::count_free_occurrences("z", &expr), 0);
    }

    #[test]
    fn test_count_free_occurrences_one() {
        // var "x" appears once in pred("x", [])
        let expr = var("x");
        assert_eq!(LetInliner::count_free_occurrences("x", &expr), 1);
    }

    #[test]
    fn test_count_free_occurrences_multi() {
        // var "x" appears twice: Add(x, x)
        let expr = TLExpr::add(var("x"), var("x"));
        assert_eq!(LetInliner::count_free_occurrences("x", &expr), 2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // substitute tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_substitute_simple() {
        // substitute "x" with Constant(7.0) in pred("x", []) => Constant(7.0)
        let body = var("x");
        let result = LetInliner::substitute("x", &TLExpr::Constant(7.0), body);
        assert_eq!(result, TLExpr::Constant(7.0));
    }

    #[test]
    fn test_substitute_shadowed() {
        // substitute "x" with Constant(7.0) in Exists{x, D, pred("x",[])}
        // The binder "x" shadows the substitution.
        let inner = TLExpr::exists("x", "D", var("x"));
        let result = LetInliner::substitute("x", &TLExpr::Constant(7.0), inner.clone());
        // Because "x" is shadowed, the result should equal the original (no substitution in body).
        assert_eq!(result, inner);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inlining behaviour tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_inline_constant_binding() {
        // Let x = 5.0 in Add(x, x)  => Add(5.0, 5.0)  (constant, always inlined)
        let inliner = LetInliner::with_default();
        let expr = TLExpr::let_binding("x", TLExpr::Constant(5.0), TLExpr::add(var("x"), var("x")));
        let (result, stats) = inliner.run(expr);
        assert_eq!(stats.constant_inlines, 1);
        assert_eq!(
            result,
            TLExpr::add(TLExpr::Constant(5.0), TLExpr::Constant(5.0))
        );
    }

    #[test]
    fn test_inline_variable_binding() {
        // Let x = y in pred("p", []) where body uses var("x")
        // => pred("p", []) with x replaced by y
        let inliner = LetInliner::with_default();
        let expr = TLExpr::let_binding("x", var("y"), TLExpr::add(var("x"), TLExpr::Constant(1.0)));
        let (result, stats) = inliner.run(expr);
        assert_eq!(stats.variable_inlines, 1);
        assert_eq!(result, TLExpr::add(var("y"), TLExpr::Constant(1.0)));
    }

    #[test]
    fn test_inline_single_use() {
        // Let x = Add(Constant(3.0), Constant(4.0)) in Sqrt(x)
        // x used once, so inline it.
        let inliner = LetInliner::with_default();
        let binding_val = TLExpr::add(TLExpr::Constant(3.0), TLExpr::Constant(4.0));
        let expr = TLExpr::let_binding("x", binding_val.clone(), TLExpr::sqrt(var("x")));
        let (result, stats) = inliner.run(expr);
        assert_eq!(stats.single_use_inlines, 1);
        assert_eq!(result, TLExpr::sqrt(binding_val));
    }

    #[test]
    fn test_no_inline_multi_use_by_default() {
        // With inline_single_use=true but binding is neither constant nor var-alias,
        // and x is used 2 times => should NOT be inlined.
        let cfg = InlineConfig {
            inline_single_use: true,
            inline_constants: false,
            inline_vars: false,
            max_passes: 5,
            max_inline_depth: 10,
        };
        let inliner = LetInliner::new(cfg);
        let binding_val = TLExpr::add(TLExpr::Constant(3.0), TLExpr::Constant(4.0));
        let expr = TLExpr::let_binding("x", binding_val.clone(), TLExpr::add(var("x"), var("x")));
        let (_result, stats) = inliner.run(expr);
        // x used twice, non-simple binding → not inlined
        assert_eq!(stats.single_use_inlines, 0);
        assert_eq!(stats.total(), 0);
    }

    #[test]
    fn test_inline_depth_limit() {
        // Binding value is very deep (depth > max_inline_depth) => not inlined
        let cfg = InlineConfig {
            inline_single_use: true,
            inline_constants: true,
            inline_vars: true,
            max_passes: 5,
            max_inline_depth: 3,
        };
        let inliner = LetInliner::new(cfg);
        // deep_add(5) has depth 6 > 3
        let deep = deep_add(5);
        let expr = TLExpr::let_binding("x", deep, TLExpr::sqrt(var("x")));
        let (_result, stats) = inliner.run(expr);
        assert_eq!(stats.total(), 0, "deep binding should not be inlined");
    }

    #[test]
    fn test_shadowing_respected() {
        // Let x = 5.0 in Let x = 2.0 in pred("x",[])
        // Outer x is inlined (constant). The inner binding re-introduces x.
        // Inner x=2.0 should then also be inlined as a constant.
        let inliner = LetInliner::with_default();
        let expr = TLExpr::let_binding(
            "x",
            TLExpr::Constant(5.0),
            TLExpr::let_binding("x", TLExpr::Constant(2.0), var("x")),
        );
        // After full inlining the result should be Constant(2.0)
        let (result, stats) = inliner.run(expr);
        assert_eq!(result, TLExpr::Constant(2.0));
        // Both constant bindings were inlined.
        assert!(stats.constant_inlines >= 2);
    }

    #[test]
    fn test_run_fixed_point() {
        // Let a = 1.0 in Let b = a in Add(b, b)
        // Pass 1: a=1.0 (constant) inlined → Let b = 1.0 in Add(b, b)
        // Pass 2: b=1.0 (constant) inlined → Add(1.0, 1.0)
        let inliner = LetInliner::with_default();
        let expr = TLExpr::let_binding(
            "a",
            TLExpr::Constant(1.0),
            TLExpr::let_binding("b", var("a"), TLExpr::add(var("b"), var("b"))),
        );
        let (result, stats) = inliner.run(expr);
        assert_eq!(
            result,
            TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(1.0))
        );
        assert!(stats.total() >= 2);
    }

    #[test]
    fn test_run_preserves_non_let() {
        // An expression with no Let bindings is returned unchanged.
        let inliner = LetInliner::with_default();
        let expr = TLExpr::and(TLExpr::pred("P", vec![]), TLExpr::Constant(1.0));
        let (result, stats) = inliner.run(expr.clone());
        assert_eq!(result, expr);
        assert_eq!(stats.total(), 0);
    }

    #[test]
    fn test_inline_disabled() {
        // With all inlining disabled, nothing should be inlined.
        let cfg = InlineConfig {
            inline_single_use: false,
            inline_constants: false,
            inline_vars: false,
            max_passes: 5,
            max_inline_depth: 10,
        };
        let inliner = LetInliner::new(cfg);
        let expr = TLExpr::let_binding("x", TLExpr::Constant(99.0), var("x"));
        let (_result, stats) = inliner.run(expr);
        assert_eq!(stats.total(), 0, "all flags disabled => no inlining");
    }

    #[test]
    fn test_reduction_pct_after_inlining() {
        // Inlining a constant in Let x = C in Add(x, x) should reduce node count.
        let inliner = LetInliner::with_default();
        let expr = TLExpr::let_binding("x", TLExpr::Constant(3.0), TLExpr::add(var("x"), var("x")));
        let (_, stats) = inliner.run(expr);
        assert!(
            stats.nodes_after <= stats.nodes_before,
            "nodes should not grow: before={}, after={}",
            stats.nodes_before,
            stats.nodes_after
        );
        // Should have removed the Let node and the binding Constant
        // Before: Let(C(3), Add(x, x)) = 5 nodes
        // After:  Add(C(3), C(3)) = 3 nodes
        assert!(stats.reduction_pct() > 0.0, "should have some reduction");
    }
}
