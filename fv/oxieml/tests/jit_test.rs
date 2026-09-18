//! Integration tests for JIT compilation of `OxiOp` sequences via Cranelift.
//!
//! All tests are gated behind `#[cfg(feature = "jit")]`.

#![cfg(feature = "jit")]

use oxieml::lower::{LoweredOp, OxiOp};
use oxieml::{JitCache, JitFn};
use std::sync::Arc;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Assert approximate equality, tolerating NaN == NaN and Inf == Inf.
fn assert_near(got: f64, expected: f64, eps: f64, label: &str) {
    // Exact match (handles Inf == Inf and -Inf == -Inf via PartialEq).
    if got == expected {
        return;
    }
    // Both NaN is a valid "match".
    if got.is_nan() && expected.is_nan() {
        return;
    }
    assert!(
        (got - expected).abs() < eps,
        "{label}: got {got}, expected {expected}, delta={}",
        (got - expected).abs()
    );
}

// ─── tests ───────────────────────────────────────────────────────────────────

/// A single constant should be returned unchanged.
#[test]
fn jit_const_returns_value() {
    // Use 2.5 — not an approximation of any float constant.
    let ops = vec![OxiOp::Const(2.5)];
    let f = JitFn::compile(&ops, 0).expect("compile const");
    let result = f.call(&[]);
    assert_near(result, 2.5, 1e-12, "const 2.5");
}

/// `[Var(0), Var(1), Add]` should compute `vars[0] + vars[1]`.
#[test]
fn jit_add_two_vars() {
    let ops = vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Add];
    let f = JitFn::compile(&ops, 2).expect("compile add");
    let result = f.call(&[2.0, 3.0]);
    assert_near(result, 5.0, 1e-12, "2 + 3");
}

/// `[Var(0), Exp]` should compute `exp(vars[0])`.
#[test]
fn jit_exp_var() {
    let ops = vec![OxiOp::Var(0), OxiOp::Exp];
    let f = JitFn::compile(&ops, 1).expect("compile exp");
    let result = f.call(&[1.0]);
    assert_near(result, std::f64::consts::E, 1e-10, "exp(1)");
}

/// `[Var(0), Sin]` at `π/2` should give 1.0.
#[test]
fn jit_sin_var() {
    let ops = vec![OxiOp::Var(0), OxiOp::Sin];
    let f = JitFn::compile(&ops, 1).expect("compile sin");
    let result = f.call(&[std::f64::consts::FRAC_PI_2]);
    assert_near(result, 1.0, 1e-10, "sin(π/2)");
}

/// Complex expression: `sin(x) * cos(y) + x^2`.
/// Encodes as: `[Var(0), Sin, Var(1), Cos, Mul, Var(0), Const(2.0), Pow, Add]`.
#[test]
fn jit_complex_expr() {
    // sin(x0) * cos(x1) + x0^2
    let ops = vec![
        OxiOp::Var(0),
        OxiOp::Sin,
        OxiOp::Var(1),
        OxiOp::Cos,
        OxiOp::Mul,
        OxiOp::Var(0),
        OxiOp::Const(2.0),
        OxiOp::Pow,
        OxiOp::Add,
    ];

    let f = JitFn::compile(&ops, 2).expect("compile complex");

    // Compare JIT result against the interpreter.
    let vars = [0.5_f64, 1.0_f64];
    let jit_result = f.call(&vars);
    let interp_result = LoweredOp::eval_ops(&ops, &vars);
    assert_near(jit_result, interp_result, 1e-10, "sin(x)*cos(y)+x^2");
}

/// Calling `get_or_compile` twice with the same ops must return an `Arc` that
/// points to the same allocation (cache hit).
#[test]
fn jit_cache_returns_same_fn() {
    let ops = vec![OxiOp::Var(0), OxiOp::Exp];
    let cache = JitCache::new();

    let a = cache.get_or_compile(&ops, 1).expect("first compile");
    let b = cache
        .get_or_compile(&ops, 1)
        .expect("second compile (cache hit)");

    // Same underlying allocation — both Arcs point to the same JitFn.
    assert!(
        Arc::ptr_eq(&a, &b),
        "cache must return the same Arc on a second call with identical ops"
    );
}

/// Type alias for the parity test table entry.
type ParityCase = (Vec<OxiOp>, usize, Vec<Vec<f64>>);

/// Parity test: for a variety of `OxiOp` sequences and variable sets, the JIT
/// result and the stack-machine interpreter must agree to within 1e-10
/// (or both be NaN).
#[test]
fn jit_parity_with_interpreter() {
    // Collection of (ops, n_vars, var_sets) triples.
    let test_cases: &[ParityCase] = &[
        // 1. Constant
        (vec![OxiOp::Const(42.0)], 0, vec![vec![], vec![]]),
        // 2. Identity: Var(0)
        (
            vec![OxiOp::Var(0)],
            1,
            vec![vec![1.0], vec![-3.5], vec![0.0], vec![100.0], vec![-0.001]],
        ),
        // 3. exp(x)
        (
            vec![OxiOp::Var(0), OxiOp::Exp],
            1,
            vec![vec![0.0], vec![1.0], vec![-1.0], vec![5.0], vec![-5.0]],
        ),
        // 4. ln(x) — includes a negative input (produces NaN)
        (
            vec![OxiOp::Var(0), OxiOp::Ln],
            1,
            vec![
                vec![1.0],
                vec![7.389_f64],
                vec![-1.0],
                vec![0.001],
                vec![1e6],
            ],
        ),
        // 5. x + y
        (
            vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Add],
            2,
            vec![
                vec![1.0, 2.0],
                vec![-3.0, 3.0],
                vec![0.0, 0.0],
                vec![10.0, -10.0],
                vec![0.5, 0.5],
            ],
        ),
        // 6. x * y
        (
            vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Mul],
            2,
            vec![
                vec![2.0, 3.0],
                vec![-1.0, 4.0],
                vec![0.0, 99.0],
                vec![1.5, 1.5],
                vec![-2.5, -2.5],
            ],
        ),
        // 7. sin(x)
        (
            vec![OxiOp::Var(0), OxiOp::Sin],
            1,
            vec![
                vec![0.0],
                vec![std::f64::consts::PI / 6.0],
                vec![std::f64::consts::FRAC_PI_2],
                vec![std::f64::consts::PI],
                vec![2.0 * std::f64::consts::PI],
            ],
        ),
        // 8. tanh(x)
        (
            vec![OxiOp::Var(0), OxiOp::Tanh],
            1,
            vec![vec![0.0], vec![1.0], vec![-1.0], vec![10.0], vec![-10.0]],
        ),
        // 9. x^y (Pow)
        (
            vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Pow],
            2,
            vec![
                vec![2.0, 3.0],
                vec![4.0, 0.5],
                vec![1.0, 100.0],
                vec![10.0, 2.0],
                vec![2.0, -1.0],
            ],
        ),
        // 10. neg(x) = -x
        (
            vec![OxiOp::Var(0), OxiOp::Neg],
            1,
            vec![vec![1.0], vec![-1.0], vec![0.0], vec![1e10], vec![-1e-10]],
        ),
    ];

    let cache = JitCache::new();
    let mut total_checks = 0usize;

    for (ops, n_vars, var_sets) in test_cases {
        let jit_fn = cache
            .get_or_compile(ops, *n_vars)
            .expect("get_or_compile failed");

        for vars in var_sets {
            let jit_val = jit_fn.call(vars);
            let interp_val = LoweredOp::eval_ops(ops, vars);

            let both_nan = jit_val.is_nan() && interp_val.is_nan();
            let close_enough = (jit_val - interp_val).abs() < 1e-10;

            assert!(
                both_nan || close_enough,
                "parity failure for ops={ops:?} vars={vars:?}: \
                 jit={jit_val} interp={interp_val} delta={}",
                (jit_val - interp_val).abs()
            );
            total_checks += 1;
        }
    }

    // Sanity: make sure we actually ran checks.
    assert!(
        total_checks >= 40,
        "expected at least 40 parity checks, got {total_checks}"
    );
}

/// Verify that compiling an empty ops sequence fails gracefully (not a panic).
#[test]
fn jit_empty_ops_is_error() {
    let result = JitFn::compile(&[], 0);
    assert!(
        result.is_err(),
        "empty ops should return an error, not succeed"
    );
}

/// Subtraction: `x - y`.
#[test]
fn jit_sub_two_vars() {
    let ops = vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Sub];
    let f = JitFn::compile(&ops, 2).expect("compile sub");
    assert_near(f.call(&[5.0, 3.0]), 2.0, 1e-12, "5 - 3");
    assert_near(f.call(&[0.0, 1.0]), -1.0, 1e-12, "0 - 1");
}

/// Division: `x / y`.
#[test]
fn jit_div_two_vars() {
    let ops = vec![OxiOp::Var(0), OxiOp::Var(1), OxiOp::Div];
    let f = JitFn::compile(&ops, 2).expect("compile div");
    assert_near(f.call(&[10.0, 4.0]), 2.5, 1e-12, "10 / 4");
    // 1 / 0 → +∞ in IEEE 754; we just check the sign and that it is infinite.
    let inf_result = f.call(&[1.0, 0.0]);
    assert!(
        inf_result.is_infinite() && inf_result.is_sign_positive(),
        "1 / 0 should be +∞, got {inf_result}"
    );
}

/// Negation: `-x`.
#[test]
fn jit_neg_var() {
    let ops = vec![OxiOp::Var(0), OxiOp::Neg];
    let f = JitFn::compile(&ops, 1).expect("compile neg");
    assert_near(f.call(&[3.7]), -3.7, 1e-12, "-3.7");
    assert_near(f.call(&[-2.0]), 2.0, 1e-12, "-(-2)");
}

/// Cosine: `cos(x)`.
#[test]
fn jit_cos_var() {
    let ops = vec![OxiOp::Var(0), OxiOp::Cos];
    let f = JitFn::compile(&ops, 1).expect("compile cos");
    assert_near(f.call(&[0.0]), 1.0, 1e-12, "cos(0)");
    assert_near(f.call(&[std::f64::consts::PI]), -1.0, 1e-10, "cos(π)");
}

/// `ops_hash` produces different values for distinct sequences.
#[test]
fn jit_ops_hash_distinguishes_sequences() {
    use oxieml::jit::ops_hash;
    let a = vec![OxiOp::Var(0), OxiOp::Exp];
    let b = vec![OxiOp::Var(0), OxiOp::Ln];
    let c = vec![OxiOp::Var(0), OxiOp::Exp];
    assert_ne!(ops_hash(&a), ops_hash(&b), "exp vs ln hashes must differ");
    assert_eq!(
        ops_hash(&a),
        ops_hash(&c),
        "identical sequences must hash the same"
    );
}
