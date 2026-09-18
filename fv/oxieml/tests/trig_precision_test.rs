//! Integration tests for trig-precision lowering recognition.
//!
//! Canonical `sin`/`cos` constructions produce deep EML trees that evaluate
//! through complex arithmetic with ~1e-2 precision drift. The lowering pass
//! recognises these canonical tree shapes and rewrites them to
//! `LoweredOp::Sin` / `LoweredOp::Cos`, dispatching to `f64::sin` / `f64::cos`
//! in the stack machine — giving full f64 precision (~1e-15) via
//! `EmlTree::eval_real_lowered`.

use oxieml::{Canonical, EmlTree, EvalCtx, LoweredOp};
use std::f64::consts::PI;

#[test]
fn sin_half_pi_lowered_is_one() {
    let x = EmlTree::var(0);
    let tree = Canonical::sin(&x);
    let ctx = EvalCtx::new(&[PI / 2.0]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-12,
        "sin(PI/2) via lowered = {result}, expected 1.0"
    );
}

#[test]
fn cos_zero_lowered_is_one() {
    let x = EmlTree::var(0);
    let tree = Canonical::cos(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-12,
        "cos(0) via lowered = {result}, expected 1.0"
    );
}

#[test]
fn sin_pi_over_six_lowered_is_half() {
    let x = EmlTree::var(0);
    let tree = Canonical::sin(&x);
    let ctx = EvalCtx::new(&[PI / 6.0]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    assert!(
        (result - 0.5).abs() < 1e-12,
        "sin(PI/6) via lowered = {result}, expected 0.5"
    );
}

#[test]
fn cos_pi_over_three_lowered_is_half() {
    let x = EmlTree::var(0);
    let tree = Canonical::cos(&x);
    let ctx = EvalCtx::new(&[PI / 3.0]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    assert!(
        (result - 0.5).abs() < 1e-12,
        "cos(PI/3) via lowered = {result}, expected 0.5"
    );
}

#[test]
fn sin_negative_half_pi_lowered_is_negative_one() {
    let x = EmlTree::var(0);
    let tree = Canonical::sin(&x);
    let ctx = EvalCtx::new(&[-PI / 2.0]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    assert!(
        (result - (-1.0)).abs() < 1e-12,
        "sin(-PI/2) via lowered = {result}, expected -1.0"
    );
}

#[test]
fn cos_pi_lowered_is_negative_one() {
    let x = EmlTree::var(0);
    let tree = Canonical::cos(&x);
    let ctx = EvalCtx::new(&[PI]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    assert!(
        (result - (-1.0)).abs() < 1e-12,
        "cos(PI) via lowered = {result}, expected -1.0"
    );
}

#[test]
fn pythagorean_identity_lowered() {
    // sin^2 + cos^2 = 1 at an arbitrary angle via the lowered path.
    let x = EmlTree::var(0);
    let sin_x = Canonical::sin(&x);
    let cos_x = Canonical::cos(&x);
    let theta = 0.7_f64;
    let ctx = EvalCtx::new(&[theta]);
    let s = sin_x.eval_real_lowered(&ctx).unwrap();
    let c = cos_x.eval_real_lowered(&ctx).unwrap();
    assert!(
        (s * s + c * c - 1.0).abs() < 1e-12,
        "sin^2 + cos^2 at {theta} = {}, expected 1.0",
        s * s + c * c
    );
}

#[test]
fn cosh_sinh_identity_lowered() {
    // cosh^2 - sinh^2 = 1 — pure-real-domain, already lowers cleanly via
    // Exp/Sub/Add/Div without any new matcher.
    let x = EmlTree::var(0);
    let sinh = Canonical::sinh(&x);
    let cosh = Canonical::cosh(&x);
    let ctx = EvalCtx::new(&[1.5]);
    let s = sinh.eval_real_lowered(&ctx).unwrap();
    let c = cosh.eval_real_lowered(&ctx).unwrap();
    assert!(
        (c * c - s * s - 1.0).abs() < 1e-12,
        "cosh^2 - sinh^2 at 1.5 = {}, expected 1.0",
        c * c - s * s
    );
}

#[test]
fn tanh_zero_lowered_is_zero() {
    let x = EmlTree::var(0);
    let tree = Canonical::tanh(&x);
    let ctx = EvalCtx::new(&[0.0]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    assert!(
        result.abs() < 1e-12,
        "tanh(0) via lowered = {result}, expected 0.0"
    );
}

#[test]
fn tanh_half_matches_f64() {
    let x = EmlTree::var(0);
    let tree = Canonical::tanh(&x);
    let ctx = EvalCtx::new(&[0.5]);
    let result = tree.eval_real_lowered(&ctx).unwrap();
    let expected = 0.5_f64.tanh();
    assert!(
        (result - expected).abs() < 1e-12,
        "tanh(0.5) via lowered = {result}, expected {expected}"
    );
}

#[test]
fn sin_structure_lowers_to_sin_opcode() {
    let x = EmlTree::var(0);
    let tree = Canonical::sin(&x);
    let lowered = tree.lower();

    fn contains_sin(op: &LoweredOp) -> bool {
        match op {
            LoweredOp::Sin(_) => true,
            LoweredOp::Cos(_) => false,
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => contains_sin(a) || contains_sin(b),
            LoweredOp::Exp(a) | LoweredOp::Ln(a) | LoweredOp::Neg(a) => contains_sin(a),
            _ => false,
        }
    }

    assert!(
        contains_sin(&lowered),
        "sin(x) must lower to LoweredOp::Sin, got {lowered:?}"
    );
}

#[test]
fn cos_structure_lowers_to_cos_opcode() {
    let x = EmlTree::var(0);
    let tree = Canonical::cos(&x);
    let lowered = tree.lower();

    fn contains_cos(op: &LoweredOp) -> bool {
        match op {
            LoweredOp::Cos(_) => true,
            LoweredOp::Sin(_) => false,
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => contains_cos(a) || contains_cos(b),
            LoweredOp::Exp(a) | LoweredOp::Ln(a) | LoweredOp::Neg(a) => contains_cos(a),
            _ => false,
        }
    }

    assert!(
        contains_cos(&lowered),
        "cos(x) must lower to LoweredOp::Cos, got {lowered:?}"
    );
}

#[test]
fn precision_gap_vs_raw_eval_real_is_huge() {
    // Demonstrates why `eval_real_lowered` is worth having: the raw complex
    // evaluation path drifts to ~1e-2 on canonical sin, while the lowered
    // path hits true f64 precision.
    let x = EmlTree::var(0);
    let tree = Canonical::sin(&x);
    let ctx = EvalCtx::new(&[PI / 2.0]);

    let lowered_result = tree.eval_real_lowered(&ctx).unwrap();
    let lowered_err = (lowered_result - 1.0).abs();
    assert!(
        lowered_err < 1e-12,
        "lowered sin(PI/2) error = {lowered_err} exceeds 1e-12"
    );

    // Raw eval_real goes through complex arithmetic on the Euler formula.
    // We do *not* assert a particular error magnitude for the raw path
    // (that's by definition drifty) — we only assert the lowered path is
    // strictly tighter if eval_real succeeds at all.
    if let Ok(raw_result) = tree.eval_real(&ctx) {
        let raw_err = (raw_result - 1.0).abs();
        assert!(
            lowered_err <= raw_err,
            "lowered path ({lowered_err}) must be at least as accurate as raw eval_real ({raw_err})"
        );
    }
}
