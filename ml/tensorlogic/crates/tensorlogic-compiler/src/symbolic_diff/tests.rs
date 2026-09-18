//! Tests for symbolic differentiation.

#![cfg(test)]

use tensorlogic_ir::{TLExpr, TNormKind};

use super::api::{differentiate, jacobian};
use super::helpers::{is_constant_value, simplify_derivative};
use super::types::{DiffConfig, DiffError};

/// Convenience: scalar variable as zero-arity predicate.
fn var(name: &str) -> TLExpr {
    TLExpr::pred(name, vec![])
}

fn default_cfg() -> DiffConfig {
    DiffConfig::default()
}

fn no_simplify_cfg() -> DiffConfig {
    DiffConfig {
        simplify_result: false,
        ..DiffConfig::default()
    }
}

fn strict_cfg() -> DiffConfig {
    DiffConfig {
        error_on_unsupported: true,
        simplify_result: false,
        ..DiffConfig::default()
    }
}

// Test 1: d(constant)/dx = 0
#[test]
fn test_diff_constant_is_zero() {
    let expr = TLExpr::Constant(42.0);
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, 0.0),
        "d(42)/dx should be 0, got {:?}",
        result.derivative
    );
}

// Test 2: d(x)/dx = 1
#[test]
fn test_diff_var_wrt_self_is_one() {
    let expr = var("x");
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, 1.0),
        "d(x)/dx should be 1, got {:?}",
        result.derivative
    );
}

// Test 3: d(y)/dx = 0 (different variable)
#[test]
fn test_diff_different_var_is_zero() {
    let expr = var("y");
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, 0.0),
        "d(y)/dx should be 0, got {:?}",
        result.derivative
    );
}

// Test 4: Sum rule
#[test]
fn test_diff_sum_rule() {
    let expr = TLExpr::add(var("x"), var("y"));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, 1.0),
        "d(x+y)/dx should simplify to 1, got {:?}",
        result.derivative
    );
}

// Test 5: Product rule d(x*y)/dx = y
#[test]
fn test_diff_product_rule_xy() {
    let expr = TLExpr::mul(var("x"), var("y"));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Pred { name, args } if name == "y" && args.is_empty()),
        "d(x*y)/dx should simplify to y, got {:?}",
        result.derivative
    );
}

// Test 6: Product rule d(x*x)/dx is non-zero
#[test]
fn test_diff_product_rule_xx() {
    let expr = TLExpr::mul(var("x"), var("x"));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        !is_constant_value(&result.derivative, 0.0),
        "d(x^2 via mul)/dx should not be zero, got {:?}",
        result.derivative
    );
}

// Test 7: Power rule d(x^2)/dx produces Mul with factor 2.0
#[test]
fn test_diff_power_rule() {
    let expr = TLExpr::pow(var("x"), TLExpr::Constant(2.0));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        !is_constant_value(&result.derivative, 0.0),
        "d(x^2)/dx should not be zero, got {:?}",
        result.derivative
    );
    match &result.derivative {
        TLExpr::Mul(l, r) => {
            let has_two = is_constant_value(l, 2.0) || is_constant_value(r, 2.0);
            assert!(
                has_two,
                "d(x^2)/dx should contain factor 2.0, got {:?}",
                result.derivative
            );
        }
        other => panic!("Expected Mul for d(x^2)/dx, got {:?}", other),
    }
}

// Test 8: Quotient rule d(1/x)/dx is Div
#[test]
fn test_diff_quotient_rule() {
    let expr = TLExpr::div(TLExpr::Constant(1.0), var("x"));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Div(_, _)),
        "d(1/x)/dx should be a Div expression, got {:?}",
        result.derivative
    );
}

// Test 9: Chain rule via Apply
#[test]
fn test_diff_chain_rule_apply() {
    let x_sq = TLExpr::pow(var("x"), TLExpr::Constant(2.0));
    let f = var("f");
    let expr = TLExpr::Apply {
        function: Box::new(f),
        argument: Box::new(x_sq),
    };
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Mul(_, _)),
        "d(f(x^2))/dx should be Mul (chain rule), got {:?}",
        result.derivative
    );
}

// Test 10: Negation via Sub(0, x)
#[test]
fn test_diff_negation() {
    let expr = TLExpr::sub(TLExpr::Constant(0.0), var("x"));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, -1.0),
        "d(0-x)/dx should be -1, got {:?}",
        result.derivative
    );
}

// Test 11: Subtraction d(x-y)/dx = 1
#[test]
fn test_diff_subtraction() {
    let expr = TLExpr::sub(var("x"), var("y"));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, 1.0),
        "d(x-y)/dx should be 1, got {:?}",
        result.derivative
    );
}

// Test 12: Logical AND differentiation produces OR
#[test]
fn test_diff_logical_and() {
    let expr = TLExpr::and(var("x"), var("y"));
    let result = differentiate(&expr, "x", &no_simplify_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Or(_, _)),
        "d(AND)/dx should be OR, got {:?}",
        result.derivative
    );
}

// Test 13: Logical OR
#[test]
fn test_diff_logical_or() {
    let expr = TLExpr::or(var("x"), var("y"));
    let result = differentiate(&expr, "x", &no_simplify_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Or(_, _)),
        "d(OR)/dx should be OR, got {:?}",
        result.derivative
    );
}

// Test 14: Logical NOT
#[test]
fn test_diff_logical_not() {
    let expr = TLExpr::negate(var("x"));
    let result = differentiate(&expr, "x", &no_simplify_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Not(_)),
        "d(NOT(x))/dx should be NOT, got {:?}",
        result.derivative
    );
}

// Test 15: Implication
#[test]
fn test_diff_implication() {
    let expr = TLExpr::imply(var("x"), var("y"));
    let result = differentiate(&expr, "x", &no_simplify_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Or(_, _)),
        "d(x→y)/dx should produce an OR, got {:?}",
        result.derivative
    );
}

// Test 16: Let binding differentiation is non-zero
#[test]
fn test_diff_let_binding() {
    let x_times_2 = TLExpr::mul(var("x"), TLExpr::Constant(2.0));
    let body = TLExpr::add(var("z"), var("y"));
    let expr = TLExpr::Let {
        var: "z".to_string(),
        value: Box::new(x_times_2),
        body: Box::new(body),
    };
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        !is_constant_value(&result.derivative, 0.0),
        "d(let z=2x in z+y)/dx should not be zero, got {:?}",
        result.derivative
    );
}

// Test 17: Jacobian with multiple vars
#[test]
fn test_jacobian_multiple_vars() {
    let expr = TLExpr::add(
        TLExpr::pow(var("x"), TLExpr::Constant(2.0)),
        TLExpr::pow(var("y"), TLExpr::Constant(2.0)),
    );
    let jac = jacobian(&expr, &["x", "y"], &default_cfg()).expect("jacobian");
    assert_eq!(jac.len(), 2);
    assert_eq!(jac[0].0, "x");
    assert_eq!(jac[1].0, "y");
    assert!(
        !is_constant_value(&jac[0].1.derivative, 0.0),
        "df/dx should not be zero"
    );
    assert!(
        !is_constant_value(&jac[1].1.derivative, 0.0),
        "df/dy should not be zero"
    );
}

// Test 18: Simplifier identities
#[test]
fn test_simplification_identity() {
    let add_zero = TLExpr::add(TLExpr::Constant(0.0), var("x"));
    let s = simplify_derivative(add_zero);
    assert!(
        matches!(&s, TLExpr::Pred { name, args } if name == "x" && args.is_empty()),
        "0 + x should simplify to x, got {:?}",
        s
    );

    let mul_one = TLExpr::mul(TLExpr::Constant(1.0), var("x"));
    let s = simplify_derivative(mul_one);
    assert!(
        matches!(&s, TLExpr::Pred { name, args } if name == "x" && args.is_empty()),
        "1 * x should simplify to x, got {:?}",
        s
    );

    let mul_zero = TLExpr::mul(var("x"), TLExpr::Constant(0.0));
    let s = simplify_derivative(mul_zero);
    assert!(
        is_constant_value(&s, 0.0),
        "x * 0 should simplify to 0, got {:?}",
        s
    );

    let pow_zero = TLExpr::pow(var("x"), TLExpr::Constant(0.0));
    let s = simplify_derivative(pow_zero);
    assert!(
        is_constant_value(&s, 1.0),
        "x^0 should simplify to 1, got {:?}",
        s
    );

    let pow_one = TLExpr::pow(var("x"), TLExpr::Constant(1.0));
    let s = simplify_derivative(pow_one);
    assert!(
        matches!(&s, TLExpr::Pred { name, args } if name == "x" && args.is_empty()),
        "x^1 should simplify to x, got {:?}",
        s
    );
}

// Test 19: Max depth guard
#[test]
fn test_max_depth_guard() {
    let mut expr = var("x");
    for _ in 0..10 {
        expr = TLExpr::add(expr, TLExpr::Constant(1.0));
    }
    let cfg = DiffConfig {
        max_expr_depth: 3,
        simplify_result: false,
        error_on_unsupported: false,
    };
    let result = differentiate(&expr, "x", &cfg);
    assert!(
        matches!(result, Err(DiffError::MaxDepthExceeded)),
        "should hit MaxDepthExceeded, got {:?}",
        result
    );
}

// Test 20: error_on_unsupported triggers ExprTooComplex for LeastFixpoint
#[test]
fn test_error_on_unsupported() {
    let expr = TLExpr::LeastFixpoint {
        var: "X".to_string(),
        body: Box::new(var("x")),
    };
    let result = differentiate(&expr, "x", &strict_cfg());
    assert!(
        matches!(result, Err(DiffError::ExprTooComplex(_))),
        "LeastFixpoint with error_on_unsupported should return ExprTooComplex, got {:?}",
        result
    );
}

// Test 21: Non-strict mode falls through to Zero for unsupported nodes
#[test]
fn test_non_strict_unsupported_returns_zero() {
    let expr = TLExpr::LeastFixpoint {
        var: "X".to_string(),
        body: Box::new(var("x")),
    };
    let result = differentiate(&expr, "x", &no_simplify_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, 0.0),
        "LeastFixpoint in non-strict mode should return 0, got {:?}",
        result.derivative
    );
    assert!(
        !result.unsupported_nodes.is_empty(),
        "should record unsupported nodes"
    );
}

// Test 22: d(exp(x))/dx = exp(x)
#[test]
fn test_diff_exp() {
    let expr = TLExpr::Exp(Box::new(var("x")));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Exp(_)),
        "d(exp(x))/dx should be Exp, got {:?}",
        result.derivative
    );
}

// Test 23: d(log(x))/dx = 1/x
#[test]
fn test_diff_log() {
    let expr = TLExpr::Log(Box::new(var("x")));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Div(_, _)),
        "d(log(x))/dx should be Div, got {:?}",
        result.derivative
    );
}

// Test 24: d(sin(x))/dx = cos(x)
#[test]
fn test_diff_sin() {
    let expr = TLExpr::Sin(Box::new(var("x")));
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::Cos(_)),
        "d(sin(x))/dx should be Cos, got {:?}",
        result.derivative
    );
}

// Test 25: bound-variable shadowing
#[test]
fn test_diff_quantifier_shadows_var() {
    let p = TLExpr::pred("P", vec![tensorlogic_ir::Term::var("x")]);
    let expr = TLExpr::ForAll {
        var: "x".to_string(),
        domain: "Domain".to_string(),
        body: Box::new(p),
    };
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        is_constant_value(&result.derivative, 0.0),
        "d(∀x.P(x))/dx should be 0 (bound variable), got {:?}",
        result.derivative
    );
}

// Test 26: Weighted rule differentiation
#[test]
fn test_diff_weighted_rule() {
    let expr = TLExpr::WeightedRule {
        weight: 0.5,
        rule: Box::new(var("x")),
    };
    let result = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::WeightedRule { weight, rule }
            if (*weight - 0.5).abs() < f64::EPSILON
                && is_constant_value(rule, 1.0)
        ),
        "d(0.5*x rule)/dx should be WeightedRule{{0.5, 1.0}}, got {:?}",
        result.derivative
    );
}

// Test 27: Jacobian ordering matches input vars
#[test]
fn test_jacobian_ordering() {
    let expr = TLExpr::add(var("a"), var("b"));
    let jac = jacobian(&expr, &["b", "a", "c"], &default_cfg()).expect("jacobian");
    assert_eq!(jac[0].0, "b");
    assert_eq!(jac[1].0, "a");
    assert_eq!(jac[2].0, "c");
    assert!(is_constant_value(&jac[0].1.derivative, 1.0), "df/db=1");
    assert!(is_constant_value(&jac[1].1.derivative, 1.0), "df/da=1");
    assert!(is_constant_value(&jac[2].1.derivative, 0.0), "df/dc=0");
}

// Test 28: Fuzzy TNorm differentiation
#[test]
fn test_diff_tnorm() {
    let expr = TLExpr::TNorm {
        kind: TNormKind::Product,
        left: Box::new(var("x")),
        right: Box::new(var("y")),
    };
    let result = differentiate(&expr, "x", &no_simplify_cfg()).expect("differentiate");
    assert!(
        matches!(&result.derivative, TLExpr::TCoNorm { .. }),
        "d(TNorm(x,y))/dx should be TCoNorm, got {:?}",
        result.derivative
    );
}

// Test 29: simplified flag
#[test]
fn test_diff_result_simplified_flag() {
    let expr = var("x");
    let with_simplify = differentiate(&expr, "x", &default_cfg()).expect("differentiate");
    let without_simplify = differentiate(&expr, "x", &no_simplify_cfg()).expect("differentiate");
    assert!(with_simplify.simplified, "should be marked simplified");
    assert!(
        !without_simplify.simplified,
        "should not be marked simplified"
    );
}

// Test 30: DiffError Display formatting
#[test]
fn test_diff_error_display() {
    let e1 = DiffError::MaxDepthExceeded;
    let e2 = DiffError::ExprTooComplex("LeastFixpoint(X)".to_string());
    assert!(!format!("{}", e1).is_empty());
    assert!(!format!("{}", e2).is_empty());
}
