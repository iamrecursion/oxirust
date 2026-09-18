//! Tests for dead-code elimination.

#![cfg(test)]

use tensorlogic_ir::{TLExpr, Term};

use super::types::{DceConfig, DeadCodeEliminator};

fn t() -> TLExpr {
    TLExpr::Constant(1.0)
}

fn f() -> TLExpr {
    TLExpr::Constant(0.0)
}

fn var_pred(name: &str) -> TLExpr {
    TLExpr::pred("p", vec![Term::var(name)])
}

fn eliminator() -> DeadCodeEliminator {
    DeadCodeEliminator::with_default()
}

#[test]
fn test_and_false_left_eliminates() {
    let expr = TLExpr::and(f(), var_pred("x"));
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(result, TLExpr::Constant(v) if v == 0.0),
        "Expected Constant(0.0), got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_and_false_right_eliminates() {
    let expr = TLExpr::and(var_pred("x"), f());
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(result, TLExpr::Constant(v) if v == 0.0),
        "Expected Constant(0.0), got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_and_true_left_eliminates() {
    let expr = TLExpr::and(t(), var_pred("x"));
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(&result, TLExpr::Pred { .. }),
        "Expected Pred, got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_and_true_right_eliminates() {
    let expr = TLExpr::and(var_pred("x"), t());
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(&result, TLExpr::Pred { .. }),
        "Expected Pred, got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_or_true_left_eliminates() {
    let expr = TLExpr::or(t(), var_pred("x"));
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(result, TLExpr::Constant(v) if v == 1.0),
        "Expected Constant(1.0), got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_or_true_right_eliminates() {
    let expr = TLExpr::or(var_pred("x"), t());
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(result, TLExpr::Constant(v) if v == 1.0),
        "Expected Constant(1.0), got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_or_false_left_eliminates() {
    let expr = TLExpr::or(f(), var_pred("x"));
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(&result, TLExpr::Pred { .. }),
        "Expected Pred, got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_or_false_right_eliminates() {
    let expr = TLExpr::or(var_pred("x"), f());
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(&result, TLExpr::Pred { .. }),
        "Expected Pred, got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_not_true_eliminates() {
    let expr = TLExpr::negate(t());
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(result, TLExpr::Constant(v) if v == 0.0),
        "Expected Constant(0.0), got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_not_false_eliminates() {
    let expr = TLExpr::negate(f());
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(result, TLExpr::Constant(v) if v == 1.0),
        "Expected Constant(1.0), got {result:?}"
    );
    assert!(stats.constant_folds >= 1);
}

#[test]
fn test_if_true_cond() {
    let then_branch = var_pred("x");
    let else_branch = var_pred("y");
    let expr = TLExpr::IfThenElse {
        condition: Box::new(t()),
        then_branch: Box::new(then_branch),
        else_branch: Box::new(else_branch),
    };
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(&result, TLExpr::Pred { .. }),
        "Expected Pred (then branch), got {result:?}"
    );
    assert!(stats.unreachable_branches >= 1);
}

#[test]
fn test_if_false_cond() {
    let then_branch = var_pred("x");
    let else_branch = var_pred("y");
    let expr = TLExpr::IfThenElse {
        condition: Box::new(f()),
        then_branch: Box::new(then_branch),
        else_branch: Box::new(else_branch),
    };
    let (result, stats) = eliminator().run(expr);
    assert!(
        matches!(&result, TLExpr::Pred { args, .. } if matches!(&args[0], Term::Var(v) if v == "y")),
        "Expected else branch pred(y), got {result:?}"
    );
    assert!(stats.unreachable_branches >= 1);
}

#[test]
fn test_dce_stats_total_eliminations() {
    let expr = TLExpr::and(f(), TLExpr::or(t(), var_pred("x")));
    let (_, stats) = eliminator().run(expr);
    assert!(stats.total_eliminations() >= 1);
}

#[test]
fn test_dce_stats_summary_nonempty() {
    let expr = TLExpr::and(f(), var_pred("x"));
    let (_, stats) = eliminator().run(expr);
    let summary = stats.summary();
    assert!(!summary.is_empty(), "summary() must not be empty");
    assert!(summary.contains("DCE"), "summary should mention DCE");
}

#[test]
fn test_dce_stats_reduction_ratio() {
    let expr = TLExpr::and(f(), var_pred("x"));
    let (_, stats) = eliminator().run(expr);
    let ratio = stats.reduction_ratio();
    assert!(
        (0.0..=1.0).contains(&ratio),
        "reduction_ratio must be in [0, 1], got {ratio}"
    );
    assert!(ratio > 0.0, "Expected some reduction, got 0");
}

#[test]
fn test_dce_config_default() {
    let cfg = DceConfig::default();
    assert!(cfg.eliminate_constant_and);
    assert!(cfg.eliminate_constant_or);
    assert!(cfg.eliminate_constant_not);
    assert!(cfg.eliminate_if_branches);
    assert!(cfg.eliminate_unused_let);
    assert!(cfg.max_passes >= 1);
}

#[test]
fn test_dce_disabled_rule() {
    let cfg = DceConfig {
        eliminate_constant_and: false,
        ..Default::default()
    };
    let eliminator = DeadCodeEliminator::new(cfg);
    let expr = TLExpr::and(f(), var_pred("x"));
    let (result, stats) = eliminator.run(expr);
    assert!(
        matches!(&result, TLExpr::And(_, _)),
        "Expected And to remain when rule is disabled, got {result:?}"
    );
    assert_eq!(
        stats.constant_folds, 0,
        "No folds should occur when And rule disabled"
    );
}

#[test]
fn test_count_nodes_leaf() {
    assert_eq!(
        DeadCodeEliminator::count_nodes(&TLExpr::Constant(42.0)),
        1,
        "A leaf should count as 1 node"
    );
    assert_eq!(
        DeadCodeEliminator::count_nodes(&var_pred("x")),
        1,
        "Pred is a leaf — count as 1"
    );
}

#[test]
fn test_count_nodes_binary() {
    let expr = TLExpr::and(var_pred("a"), var_pred("b"));
    assert_eq!(
        DeadCodeEliminator::count_nodes(&expr),
        3,
        "And(a, b) should have 3 nodes: And + Pred(a) + Pred(b)"
    );
}

#[test]
fn test_dce_fixed_point() {
    let pred = var_pred("x");
    let expr = TLExpr::and(t(), TLExpr::or(f(), pred));
    let (result1, _) = eliminator().run(expr.clone());
    let (result2, _) = eliminator().run(result1.clone());
    assert_eq!(
        result1, result2,
        "Running DCE twice on an already-fixed-point expression should give the same result"
    );
}

#[test]
fn test_nested_and_or_elimination() {
    let pred = var_pred("x");
    let expr = TLExpr::and(t(), TLExpr::or(f(), pred.clone()));
    let (result, stats) = eliminator().run(expr);
    assert_eq!(result, pred, "Expected pred(x), got {result:?}");
    assert!(stats.constant_folds >= 2, "Expected at least 2 folds");
}

#[test]
fn test_dce_passes_count() {
    let expr = TLExpr::and(f(), var_pred("x"));
    let (_, stats) = eliminator().run(expr);
    assert!(stats.passes >= 1, "At least one pass must be recorded");
}
