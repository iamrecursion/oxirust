//! Unit tests for the partial evaluator.

#![cfg(test)]

use tensorlogic_ir::TLExpr;

use super::{partially_evaluate, specialize, specialize_batch, PEConfig, PEEnv};

fn var(name: &str) -> TLExpr {
    TLExpr::pred(name, vec![])
}
fn cnst(v: f64) -> TLExpr {
    TLExpr::Constant(v)
}

// ── 1. Free variable stays symbolic ──────────────────────────────────
#[test]
fn test_free_variable_stays_symbolic() {
    let expr = var("x");
    let env = PEEnv::new(); // no bindings
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(
        matches!(res.expr, TLExpr::Pred { ref name, ref args } if name == "x" && args.is_empty())
    );
    assert!(res.residual_vars.contains(&"x".to_string()));
}

// ── 2. Bound variable replaced by its value ───────────────────────────
#[test]
fn test_bound_variable_replaced() {
    let expr = var("x");
    let env = PEEnv::new().with_f64("x", 42.0);
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if (v - 42.0).abs() < 1e-12));
    assert!(!res.residual_vars.contains(&"x".to_string()));
}

// ── 3. Fully concrete Add folds to constant ───────────────────────────
#[test]
fn test_fully_concrete_add() {
    let expr = TLExpr::add(cnst(2.0), cnst(3.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if (v - 5.0).abs() < 1e-12));
}

// ── 4. Partially concrete Add: x+0 → x ───────────────────────────────
#[test]
fn test_add_identity_x_plus_zero() {
    let expr = TLExpr::add(var("x"), cnst(0.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    // Should reduce to just `x`
    assert!(matches!(res.expr, TLExpr::Pred { ref name, .. } if name == "x"));
}

// ── 5. Mul by zero → 0 ───────────────────────────────────────────────
#[test]
fn test_mul_by_zero() {
    let expr = TLExpr::mul(var("x"), cnst(0.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if v == 0.0));
}

// ── 6. Mul by one → other operand ────────────────────────────────────
#[test]
fn test_mul_by_one() {
    let expr = TLExpr::mul(cnst(1.0), var("y"));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Pred { ref name, .. } if name == "y"));
}

// ── 7. Div by one → numerator ─────────────────────────────────────────
#[test]
fn test_div_by_one() {
    let expr = TLExpr::div(var("z"), cnst(1.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Pred { ref name, .. } if name == "z"));
}

// ── 8. Div by zero does NOT fold ──────────────────────────────────────
#[test]
fn test_div_by_zero_no_fold() {
    let expr = TLExpr::div(cnst(5.0), cnst(0.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    // Must remain as Div(5, 0), not NaN or panicked
    assert!(matches!(res.expr, TLExpr::Div(_, _)));
}

// ── 9. Boolean And short-circuits on false ────────────────────────────
#[test]
fn test_and_short_circuit_false() {
    let expr = TLExpr::and(cnst(0.0), var("x")); // false AND x = false
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if v == 0.0));
    assert!(res.stats.branches_pruned > 0);
}

// ── 10. Boolean And short-circuits on true ────────────────────────────
#[test]
fn test_and_short_circuit_true() {
    let expr = TLExpr::and(cnst(1.0), var("x")); // true AND x = x
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Pred { ref name, .. } if name == "x"));
    assert!(res.stats.branches_pruned > 0);
}

// ── 11. Boolean Or short-circuits on true ────────────────────────────
#[test]
fn test_or_short_circuit_true() {
    let expr = TLExpr::or(cnst(1.0), var("x")); // true OR x = true
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if v == 1.0));
    assert!(res.stats.branches_pruned > 0);
}

// ── 12. Boolean Or short-circuits on false ────────────────────────────
#[test]
fn test_or_short_circuit_false() {
    let expr = TLExpr::or(cnst(0.0), var("x")); // false OR x = x
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Pred { ref name, .. } if name == "x"));
    assert!(res.stats.branches_pruned > 0);
}

// ── 13. Not(true) → false ────────────────────────────────────────────
#[test]
fn test_not_true() {
    let expr = TLExpr::negate(cnst(1.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if v == 0.0));
}

// ── 14. Not(false) → true ────────────────────────────────────────────
#[test]
fn test_not_false() {
    let expr = TLExpr::negate(cnst(0.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if v == 1.0));
}

// ── 15. LetBind with concrete bound var inlines body ──────────────────
#[test]
fn test_let_concrete_inlines() {
    // let a = 5.0 in a + 3.0  → 8.0
    let expr = TLExpr::Let {
        var: "a".to_string(),
        value: Box::new(cnst(5.0)),
        body: Box::new(TLExpr::add(var("a"), cnst(3.0))),
    };
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if (v - 8.0).abs() < 1e-12));
    assert!(res.stats.lets_inlined > 0);
}

// ── 16. LetBind with free bound var keeps let but pe's body ───────────
#[test]
fn test_let_symbolic_keeps_let() {
    // let a = x in a + 3.0  (x is free, so a is symbolic)
    let expr = TLExpr::Let {
        var: "a".to_string(),
        value: Box::new(var("x")),
        body: Box::new(TLExpr::add(var("a"), cnst(3.0))),
    };
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    // Since a = x (symbolic), let is kept
    assert!(matches!(res.expr, TLExpr::Let { .. }));
    // x should be a residual free variable
    assert!(res.residual_vars.contains(&"x".to_string()));
}

// ── 17. Nested expression partially evaluates inner nodes ─────────────
#[test]
fn test_nested_partial_eval() {
    // (x + 2.0) * (3.0 + 4.0)  with x free
    // inner (3+4) should fold to 7; outer becomes x+2 * 7 → Mul(Add(x,2), 7)
    let expr = TLExpr::mul(
        TLExpr::add(var("x"), cnst(2.0)),
        TLExpr::add(cnst(3.0), cnst(4.0)),
    );
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    // The RHS should be folded to 7
    if let TLExpr::Mul(_, rhs) = &res.expr {
        assert!(matches!(rhs.as_ref(), TLExpr::Constant(v) if (v - 7.0).abs() < 1e-12));
    } else {
        panic!("Expected Mul, got {:?}", res.expr);
    }
}

// ── 18. PEStats.nodes_reduced > 0 for concrete reductions ────────────
#[test]
fn test_stats_nodes_reduced() {
    let expr = TLExpr::add(cnst(1.0), cnst(2.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(res.stats.nodes_reduced > 0);
}

// ── 19. PEStats.branches_pruned > 0 for short-circuit branches ────────
#[test]
fn test_stats_branches_pruned() {
    let expr = TLExpr::and(cnst(0.0), var("x"));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(res.stats.branches_pruned > 0);
}

// ── 20. residual_vars lists all remaining free variables ──────────────
#[test]
fn test_residual_vars() {
    // x + y + 5.0 with x = 1.0 bound; y remains free
    let expr = TLExpr::add(TLExpr::add(var("x"), var("y")), cnst(5.0));
    let env = PEEnv::new().with_f64("x", 1.0);
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    // x is bound so not free; y should be free
    assert!(!res.residual_vars.contains(&"x".to_string()));
    assert!(res.residual_vars.contains(&"y".to_string()));
}

// ── 21. specialize() with all bindings → fully concrete ───────────────
#[test]
fn test_specialize_fully_concrete() {
    // x * x with x = 4.0
    let expr = TLExpr::mul(var("x"), var("x"));
    let bindings = vec![("x".to_string(), 4.0)];
    let cfg = PEConfig::default();
    let res = specialize(&expr, &bindings, &cfg);
    assert!(matches!(res.expr, TLExpr::Constant(v) if (v - 16.0).abs() < 1e-12));
    assert!(res.residual_vars.is_empty());
}

// ── 22. specialize_batch() produces one result per binding set ─────────
#[test]
fn test_specialize_batch() {
    let expr = TLExpr::add(var("x"), cnst(10.0));
    let binding_sets = vec![
        vec![("x".to_string(), 1.0)],
        vec![("x".to_string(), 2.0)],
        vec![("x".to_string(), 3.0)],
    ];
    let cfg = PEConfig::default();
    let results = specialize_batch(&expr, &binding_sets, &cfg);
    assert_eq!(results.len(), 3);
    let vals: Vec<f64> = results
        .iter()
        .map(|r| {
            if let TLExpr::Constant(v) = r.expr {
                v
            } else {
                panic!("Expected Constant")
            }
        })
        .collect();
    assert!((vals[0] - 11.0).abs() < 1e-12);
    assert!((vals[1] - 12.0).abs() < 1e-12);
    assert!((vals[2] - 13.0).abs() < 1e-12);
}

// ── 23. PEConfig.fold_arithmetic=false disables arithmetic folding ─────
#[test]
fn test_config_no_fold_arithmetic() {
    let expr = TLExpr::add(cnst(2.0), cnst(3.0));
    let env = PEEnv::new();
    let cfg = PEConfig {
        fold_arithmetic: false,
        ..PEConfig::default()
    };
    let res = partially_evaluate(&expr, &env, &cfg);
    // Should NOT fold to 5.0 because fold_arithmetic is off
    assert!(matches!(res.expr, TLExpr::Add(_, _)));
}

// ── 24. PEConfig.prune_branches=false disables branch pruning ─────────
#[test]
fn test_config_no_prune_branches() {
    let expr = TLExpr::and(cnst(0.0), var("x")); // normally folds to false
    let env = PEEnv::new();
    let cfg = PEConfig {
        prune_branches: false,
        fold_logic: false,
        ..PEConfig::default()
    };
    let res = partially_evaluate(&expr, &env, &cfg);
    // Should NOT prune: result is still And(0, x)
    assert!(matches!(res.expr, TLExpr::And(_, _)));
    assert_eq!(res.stats.branches_pruned, 0);
}

// ── Bonus: reduction_rate helper ──────────────────────────────────────
#[test]
fn test_reduction_rate() {
    let expr = TLExpr::add(cnst(1.0), cnst(2.0));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    let rate = res.stats.reduction_rate();
    assert!(rate > 0.0 && rate <= 1.0);
}

// ── Bonus: PEEnv builder API ──────────────────────────────────────────
#[test]
fn test_env_builder_api() {
    let env = PEEnv::new()
        .with_f64("a", 1.0)
        .with_f64("b", 2.0)
        .with_bool("flag", true);
    assert_eq!(env.len(), 3);
    assert!(!env.is_empty());
    assert!(env.lookup("a").is_some());
    assert!(env.lookup("missing").is_none());
}

// ── Bonus: Imply expands and folds correctly ──────────────────────────
#[test]
fn test_imply_folds() {
    // true → x  should simplify via Not(true)=false, false OR x = x
    let expr = TLExpr::imply(cnst(1.0), var("x"));
    let env = PEEnv::new();
    let cfg = PEConfig::default();
    let res = partially_evaluate(&expr, &env, &cfg);
    assert!(matches!(res.expr, TLExpr::Pred { ref name, .. } if name == "x"));
}

// ── Bonus: Pow special cases ──────────────────────────────────────────
#[test]
fn test_pow_special_cases() {
    let cfg = PEConfig::default();
    let env = PEEnv::new();

    // x^0 = 1
    let r0 = partially_evaluate(&TLExpr::pow(var("x"), cnst(0.0)), &env, &cfg);
    assert!(matches!(r0.expr, TLExpr::Constant(v) if (v - 1.0).abs() < 1e-12));

    // x^1 = x
    let r1 = partially_evaluate(&TLExpr::pow(var("x"), cnst(1.0)), &env, &cfg);
    assert!(matches!(r1.expr, TLExpr::Pred { ref name, .. } if name == "x"));

    // 0^x = 0
    let r2 = partially_evaluate(&TLExpr::pow(cnst(0.0), var("x")), &env, &cfg);
    assert!(matches!(r2.expr, TLExpr::Constant(v) if v == 0.0));

    // 1^x = 1
    let r3 = partially_evaluate(&TLExpr::pow(cnst(1.0), var("x")), &env, &cfg);
    assert!(matches!(r3.expr, TLExpr::Constant(v) if (v - 1.0).abs() < 1e-12));
}
