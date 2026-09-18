//! End-to-end tests for temporal operator forward pass and gradient correctness.
//!
//! These tests verify:
//! 1. `temporal_next:<axis>` round-trips correctly through the executor.
//! 2. `temporal_until:<tag>:<axis>` (ProbSumProduct) produces the expected scan values.
//! 3. The VJP for `until_scan` ProbSumProduct matches central-difference finite differences.
//!
//! No GPU, no `integration-tests` feature flag — these are standalone unit-style tests
//! that directly exercise the backend primitives and the compiled graph machinery.

use scirs2_core::ndarray::{arr1, Array, ArrayD};
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{EinsumGraph, EinsumNode};
use tensorlogic_scirs_backend::{temporal_ops, Scirs2Exec};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn vec_to_arrayd(data: &[f64]) -> ArrayD<f64> {
    arr1(data).into_dyn()
}

fn assert_close(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: f64, msg: &str) {
    assert_eq!(
        a.shape(),
        b.shape(),
        "{msg}: shape mismatch {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    for (av, bv) in a.iter().zip(b.iter()) {
        assert!(
            (av - bv).abs() < tol,
            "{msg}: element differs by {} (got {}, expected {})",
            (av - bv).abs(),
            av,
            bv
        );
    }
}

// ---------------------------------------------------------------------------
// Helper: build a minimal single-op graph for testing.
// ---------------------------------------------------------------------------

/// Build a single-unary-op graph that applies `op` to `input_name`.
fn build_unary_graph(input_name: &str, op: &str) -> EinsumGraph {
    let mut g = EinsumGraph::new();
    let t_in = g.add_tensor(input_name.to_string());
    let t_out = g.add_tensor("__out__".to_string());
    g.add_node(EinsumNode::elem_unary(op, t_in, t_out))
        .expect("add_node failed");
    g.add_input(t_in).expect("add_input");
    g.add_output(t_out).expect("add_output");
    g
}

/// Build a single-binary-op graph.
fn build_binary_graph(in_a: &str, in_b: &str, op: &str) -> EinsumGraph {
    let mut g = EinsumGraph::new();
    let t_a = g.add_tensor(in_a.to_string());
    let t_b = g.add_tensor(in_b.to_string());
    let t_out = g.add_tensor("__out__".to_string());
    g.add_node(EinsumNode::elem_binary(op, t_a, t_b, t_out))
        .expect("add_node failed");
    g.add_input(t_a).expect("add_input a");
    g.add_input(t_b).expect("add_input b");
    g.add_output(t_out).expect("add_output");
    g
}

// ---------------------------------------------------------------------------
// Test 1: shift_next via executor forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_executor_shift_next_forward() {
    let x = vec_to_arrayd(&[0.2, 0.8, 0.5]);

    let graph = build_unary_graph("x", "temporal_next:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("x", x);

    let out = exec.forward(&graph).expect("forward failed");

    let expected = vec_to_arrayd(&[0.8, 0.5, 0.0]);
    assert_close(&out, &expected, 1e-12, "executor shift_next forward");
}

// ---------------------------------------------------------------------------
// Test 2: shift_next backward (VJP) via executor
// ---------------------------------------------------------------------------

#[test]
fn test_executor_shift_next_backward() {
    let x = vec_to_arrayd(&[0.1, 0.3, 0.7, 0.2]);

    let graph = build_unary_graph("x", "temporal_next:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("x", x);

    // Forward pass first.
    exec.forward(&graph).expect("forward failed");

    // Backward pass with all-ones gradient.
    let g_out = vec_to_arrayd(&[1.0, 1.0, 1.0, 1.0]);
    let tape = exec.backward(&graph, &g_out).expect("backward failed");

    // grad_x = shift_prev(g_out) = [0, g[0], g[1], g[2]] = [0, 1, 1, 1]
    let expected = vec_to_arrayd(&[0.0, 1.0, 1.0, 1.0]);
    let grad = tape.tensors[0]
        .as_ref()
        .expect("gradient for x should be present");
    assert_close(grad, &expected, 1e-12, "executor shift_next backward");
}

// ---------------------------------------------------------------------------
// Test 3: until_scan ProbSumProduct via executor forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_executor_until_scan_prob_sum_product_forward() {
    // a = [0.5, 0.5], b = [0.3, 0.4]  → u = [0.44, 0.4]
    let a = vec_to_arrayd(&[0.5, 0.5]);
    let b = vec_to_arrayd(&[0.3, 0.4]);

    let graph = build_binary_graph("a", "b", "temporal_until:prod:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a);
    exec.add_tensor("b", b);

    let out = exec.forward(&graph).expect("forward failed");
    let expected = vec_to_arrayd(&[0.44, 0.4]);
    assert_close(&out, &expected, 1e-12, "executor until_scan prod forward");
}

// ---------------------------------------------------------------------------
// Test 4: until_scan MaxMin via executor forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_executor_until_scan_max_min_forward() {
    // a = [0.9, 0.9, 0.9], b = [0.0, 0.0, 0.4] → u = [0.4, 0.4, 0.4]
    let a = vec_to_arrayd(&[0.9, 0.9, 0.9]);
    let b = vec_to_arrayd(&[0.0, 0.0, 0.4]);

    let graph = build_binary_graph("a", "b", "temporal_until:max:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a);
    exec.add_tensor("b", b);

    let out = exec.forward(&graph).expect("forward failed");
    let expected = vec_to_arrayd(&[0.4, 0.4, 0.4]);
    assert_close(&out, &expected, 1e-12, "executor until_scan max forward");
}

// ---------------------------------------------------------------------------
// Test 5: until_scan ProbSumProduct VJP via executor backward pass
// ---------------------------------------------------------------------------

#[test]
fn test_executor_until_scan_prob_sum_product_backward() {
    let a0 = vec_to_arrayd(&[0.4, 0.6, 0.8]);
    let b0 = vec_to_arrayd(&[0.2, 0.5, 0.3]);
    let g_out = vec_to_arrayd(&[1.0, 1.0, 1.0]);

    let graph = build_binary_graph("a", "b", "temporal_until:prod:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a0.clone());
    exec.add_tensor("b", b0.clone());

    exec.forward(&graph).expect("forward failed");
    let tape = exec.backward(&graph, &g_out).expect("backward failed");

    // Analytic VJP from the library function.
    let (grad_a_ref, grad_b_ref) = temporal_ops::until_scan_vjp(
        &a0.view(),
        &b0.view(),
        &g_out.view(),
        0,
        temporal_ops::UntilSemantics::ProbSumProduct,
    );

    // The graph has tensors [a, b, __out__] at indices [0, 1, 2].
    // Gradient for a is at index 0, gradient for b at index 1.
    let grad_a = tape.tensors[0]
        .as_ref()
        .expect("gradient for a should be present");
    let grad_b = tape.tensors[1]
        .as_ref()
        .expect("gradient for b should be present");

    assert_close(
        grad_a,
        &grad_a_ref,
        1e-10,
        "executor until VJP grad_a matches library",
    );
    assert_close(
        grad_b,
        &grad_b_ref,
        1e-10,
        "executor until VJP grad_b matches library",
    );
}

// ---------------------------------------------------------------------------
// Test 6: VJP finite-difference check for until_scan ProbSumProduct (smooth)
// ---------------------------------------------------------------------------

#[test]
fn test_until_scan_vjp_prob_sum_product_finite_difference_e2e() {
    // Use the direct backend functions (not through the executor) for a clean
    // finite-difference check.  This validates mathematical correctness.
    let a0 = vec_to_arrayd(&[0.3, 0.5, 0.7]);
    let b0 = vec_to_arrayd(&[0.1, 0.4, 0.6]);
    let g_out = vec_to_arrayd(&[0.5, 1.5, -0.3]);

    let sem = temporal_ops::UntilSemantics::ProbSumProduct;

    let (ga_analytic, gb_analytic) =
        temporal_ops::until_scan_vjp(&a0.view(), &b0.view(), &g_out.view(), 0, sem);

    let n = a0.len();
    let eps = 1e-5;
    let tol = 1e-4;

    // Central-difference grad_a.
    let mut ga_fd = ArrayD::zeros(a0.raw_dim());
    for j in 0..n {
        let mut ap = a0.clone();
        *ap.iter_mut().nth(j).expect("j in bounds") += eps;
        let mut am = a0.clone();
        *am.iter_mut().nth(j).expect("j in bounds") -= eps;
        let up = temporal_ops::until_scan(&ap.view(), &b0.view(), 0, sem);
        let um = temporal_ops::until_scan(&am.view(), &b0.view(), 0, sem);
        let diff = (&up - &um) / (2.0 * eps);
        let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
        if let Some(v) = ga_fd.iter_mut().nth(j) {
            *v = val;
        }
    }

    // Central-difference grad_b.
    let mut gb_fd = ArrayD::zeros(b0.raw_dim());
    for j in 0..n {
        let mut bp = b0.clone();
        *bp.iter_mut().nth(j).expect("j in bounds") += eps;
        let mut bm = b0.clone();
        *bm.iter_mut().nth(j).expect("j in bounds") -= eps;
        let up = temporal_ops::until_scan(&a0.view(), &bp.view(), 0, sem);
        let um = temporal_ops::until_scan(&a0.view(), &bm.view(), 0, sem);
        let diff = (&up - &um) / (2.0 * eps);
        let val: f64 = g_out.iter().zip(diff.iter()).map(|(&g, &d)| g * d).sum();
        if let Some(v) = gb_fd.iter_mut().nth(j) {
            *v = val;
        }
    }

    assert_close(
        &ga_analytic,
        &ga_fd,
        tol,
        "until VJP grad_a vs FD (e2e test)",
    );
    assert_close(
        &gb_analytic,
        &gb_fd,
        tol,
        "until VJP grad_b vs FD (e2e test)",
    );
}

// ---------------------------------------------------------------------------
// Test 7: shape preservation for rank-2 temporal operations
// ---------------------------------------------------------------------------

#[test]
fn test_shift_next_rank2_shape_preserved_via_executor() {
    // shape [2, 4], time axis = 0
    let x: ArrayD<f64> = Array::from_shape_fn((2, 4), |(i, j)| (i * 4 + j + 1) as f64).into_dyn();
    let shape = x.shape().to_vec();

    let graph = build_unary_graph("x", "temporal_next:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("x", x);

    let out = exec.forward(&graph).expect("forward failed rank-2");
    assert_eq!(
        out.shape(),
        shape.as_slice(),
        "rank-2 shift_next must preserve shape"
    );
}

// ---------------------------------------------------------------------------
// Test 8: WeakUntil forward pass via executor
// ---------------------------------------------------------------------------

#[test]
fn test_executor_weak_until_forward() {
    // a=[1,1,1], b=[0,0,0], WeakUntil MaxMin → [1,1,1]
    let a = vec_to_arrayd(&[1.0, 1.0, 1.0]);
    let b = vec_to_arrayd(&[0.0, 0.0, 0.0]);

    let graph = build_binary_graph("a", "b", "temporal_weakuntil:max:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a);
    exec.add_tensor("b", b);

    let out = exec.forward(&graph).expect("weak_until forward failed");
    let expected = vec_to_arrayd(&[1.0, 1.0, 1.0]);
    assert_close(&out, &expected, 1e-12, "executor weak_until MaxMin forward");
}

// ---------------------------------------------------------------------------
// Test 9: Release forward pass via executor
// ---------------------------------------------------------------------------

#[test]
fn test_executor_release_forward() {
    // a=[0,0,1], b=[1,1,1], Release MaxMin → [1,1,1]
    let a = vec_to_arrayd(&[0.0, 0.0, 1.0]);
    let b = vec_to_arrayd(&[1.0, 1.0, 1.0]);

    let graph = build_binary_graph("a", "b", "temporal_release:max:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a);
    exec.add_tensor("b", b);

    let out = exec.forward(&graph).expect("release forward failed");
    let expected = vec_to_arrayd(&[1.0, 1.0, 1.0]);
    assert_close(&out, &expected, 1e-12, "executor release MaxMin forward");
}

// ---------------------------------------------------------------------------
// Test 10: StrongRelease forward pass via executor
// ---------------------------------------------------------------------------

#[test]
fn test_executor_strong_release_forward() {
    // a=[0,1,0], b=[1,0,1], StrongRelease MaxMin → [0,0,0]
    let a = vec_to_arrayd(&[0.0, 1.0, 0.0]);
    let b = vec_to_arrayd(&[1.0, 0.0, 1.0]);

    let graph = build_binary_graph("a", "b", "temporal_strongrelease:max:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a);
    exec.add_tensor("b", b);

    let out = exec.forward(&graph).expect("strong_release forward failed");
    let expected = vec_to_arrayd(&[0.0, 0.0, 0.0]);
    assert_close(
        &out,
        &expected,
        1e-12,
        "executor strong_release MaxMin forward",
    );
}

// ---------------------------------------------------------------------------
// Test 11: WeakUntil ProbSumProduct backward pass (gradient check)
// ---------------------------------------------------------------------------

#[test]
fn test_executor_weak_until_backward_grad_check() {
    let a0 = vec_to_arrayd(&[0.4, 0.6, 0.7]);
    let b0 = vec_to_arrayd(&[0.2, 0.5, 0.3]);
    let g_out = vec_to_arrayd(&[1.0, 0.5, -0.3]);

    let graph = build_binary_graph("a", "b", "temporal_weakuntil:prod:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a0.clone());
    exec.add_tensor("b", b0.clone());

    exec.forward(&graph).expect("weak_until forward failed");
    let tape = exec
        .backward(&graph, &g_out)
        .expect("weak_until backward failed");

    // Reference from direct function call
    let (ga_ref, gb_ref) = temporal_ops::temporal_binary_scan_vjp(
        &a0.view(),
        &b0.view(),
        &g_out.view(),
        0,
        temporal_ops::TemporalBinaryForm::WeakUntil,
        temporal_ops::UntilSemantics::ProbSumProduct,
    );

    let grad_a = tape.tensors[0]
        .as_ref()
        .expect("gradient for a should be present");
    let grad_b = tape.tensors[1]
        .as_ref()
        .expect("gradient for b should be present");

    assert_close(grad_a, &ga_ref, 1e-10, "executor weak_until VJP grad_a");
    assert_close(grad_b, &gb_ref, 1e-10, "executor weak_until VJP grad_b");
}

// ---------------------------------------------------------------------------
// Test 12: Release ProbSumProduct backward pass (gradient check)
// ---------------------------------------------------------------------------

#[test]
fn test_executor_release_backward_grad_check() {
    let a0 = vec_to_arrayd(&[0.3, 0.7, 0.5]);
    let b0 = vec_to_arrayd(&[0.6, 0.4, 0.8]);
    let g_out = vec_to_arrayd(&[0.5, -0.5, 1.0]);

    let graph = build_binary_graph("a", "b", "temporal_release:prod:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a0.clone());
    exec.add_tensor("b", b0.clone());

    exec.forward(&graph).expect("release forward failed");
    let tape = exec
        .backward(&graph, &g_out)
        .expect("release backward failed");

    let (ga_ref, gb_ref) = temporal_ops::temporal_binary_scan_vjp(
        &a0.view(),
        &b0.view(),
        &g_out.view(),
        0,
        temporal_ops::TemporalBinaryForm::Release,
        temporal_ops::UntilSemantics::ProbSumProduct,
    );

    let grad_a = tape.tensors[0]
        .as_ref()
        .expect("gradient for a should be present");
    let grad_b = tape.tensors[1]
        .as_ref()
        .expect("gradient for b should be present");

    assert_close(grad_a, &ga_ref, 1e-10, "executor release VJP grad_a");
    assert_close(grad_b, &gb_ref, 1e-10, "executor release VJP grad_b");
}

// ---------------------------------------------------------------------------
// Test 13: StrongRelease ProbSumProduct backward pass (gradient check)
// ---------------------------------------------------------------------------

#[test]
fn test_executor_strong_release_backward_grad_check() {
    let a0 = vec_to_arrayd(&[0.5, 0.4, 0.6]);
    let b0 = vec_to_arrayd(&[0.3, 0.7, 0.5]);
    let g_out = vec_to_arrayd(&[-0.5, 1.0, 0.5]);

    let graph = build_binary_graph("a", "b", "temporal_strongrelease:prod:0");

    let mut exec = Scirs2Exec::new();
    exec.add_tensor("a", a0.clone());
    exec.add_tensor("b", b0.clone());

    exec.forward(&graph).expect("strong_release forward failed");
    let tape = exec
        .backward(&graph, &g_out)
        .expect("strong_release backward failed");

    let (ga_ref, gb_ref) = temporal_ops::temporal_binary_scan_vjp(
        &a0.view(),
        &b0.view(),
        &g_out.view(),
        0,
        temporal_ops::TemporalBinaryForm::StrongRelease,
        temporal_ops::UntilSemantics::ProbSumProduct,
    );

    let grad_a = tape.tensors[0]
        .as_ref()
        .expect("gradient for a should be present");
    let grad_b = tape.tensors[1]
        .as_ref()
        .expect("gradient for b should be present");

    assert_close(grad_a, &ga_ref, 1e-10, "executor strong_release VJP grad_a");
    assert_close(grad_b, &gb_ref, 1e-10, "executor strong_release VJP grad_b");
}
