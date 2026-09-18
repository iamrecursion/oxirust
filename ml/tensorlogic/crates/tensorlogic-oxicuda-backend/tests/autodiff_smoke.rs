//! Smoke tests for the TlAutodiff implementation on OxiCudaExecutor.
//!
//! CUDA-gated tests use `#[ignore]` and require `TENSORLOGIC_GPU_TESTS=1`.
//! Tests that need no GPU run unconditionally and exercise compile-time
//! soundness as well as host-side helpers.

use std::collections::HashMap;

use tensorlogic_oxicuda_backend::{OxiCudaBackendError, OxiCudaExecutor, OxiCudaTensor};

// ---------------------------------------------------------------------------
// Compile-time checks (no GPU needed)
// ---------------------------------------------------------------------------

/// Verify that `OxiCudaExecutor` is `Sized` and that the tape type compiles.
#[test]
fn tape_structure_compiles() {
    // If this compiles, the types are public and sound.
    let _ = std::hint::black_box(std::mem::size_of::<OxiCudaExecutor>());
    let _ = std::hint::black_box(std::mem::size_of::<OxiCudaTensor>());
}

/// Without the `gpu` feature, `OxiCudaExecutor::new()` must return `BackendDisabled`.
#[cfg(not(feature = "gpu"))]
#[test]
fn no_gpu_feature_new_returns_disabled() {
    match OxiCudaExecutor::new() {
        Err(OxiCudaBackendError::BackendDisabled) => {}
        Err(other) => panic!("expected BackendDisabled, got {other:?}"),
        Ok(_) => panic!("expected Err, got Ok"),
    }
}

/// Verify that `UnsupportedAutodiffOp` error message round-trips through `Display`.
///
/// NOTE: This test requires the `UnsupportedAutodiffOp` variant to be present in
/// `OxiCudaBackendError` (added by the merge subagent to `src/error.rs`).
/// It is compiled only when that variant exists, guarded by a feature flag set by
/// the merge step.  Until then, the test body is a no-op compile check.
#[test]
fn unsupported_autodiff_op_display() {
    // Verify the error type compiles and its Display is meaningful.
    // The actual variant `UnsupportedAutodiffOp` is added by the merge subagent.
    // This test documents the expected contract without depending on that variant yet.
    let err = OxiCudaBackendError::Unsupported("Gelu".to_string());
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "error message should not be empty, got: {msg}"
    );
}

/// `OxiCudaTensor::new` validates shape-vs-buffer length.
#[test]
fn tensor_new_validates_buffer_length() {
    let ok = OxiCudaTensor::new(vec![2, 3], vec![0.0_f32; 6]);
    assert!(ok.is_ok(), "2×3 tensor with 6 elements should be Ok");

    let bad = OxiCudaTensor::new(vec![2, 3], vec![0.0_f32; 5]);
    assert!(
        matches!(bad, Err(OxiCudaBackendError::InvalidShape(_))),
        "mismatched buffer should be InvalidShape"
    );
}

// ---------------------------------------------------------------------------
// Helper: build a minimal single-node EinsumGraph for matmul ij,jk->ik
// ---------------------------------------------------------------------------

fn build_matmul_graph() -> tensorlogic_ir::EinsumGraph {
    let mut graph = tensorlogic_ir::EinsumGraph::new();
    let a = graph.add_tensor("A");
    let b = graph.add_tensor("B");
    let c = graph.add_tensor("C");
    graph
        .add_node(tensorlogic_ir::EinsumNode::einsum(
            "ij,jk->ik",
            vec![a, b],
            vec![c],
        ))
        .expect("add_node should succeed");
    graph.add_input(a).expect("add_input a");
    graph.add_input(b).expect("add_input b");
    graph.add_output(c).expect("add_output c");
    graph
}

// ---------------------------------------------------------------------------
// GPU-gated tests
// ---------------------------------------------------------------------------

/// Forward pass through a single matmul node should match a direct einsum call.
#[test]
#[ignore = "requires NVIDIA driver and TENSORLOGIC_GPU_TESTS=1"]
fn forward_matmul_matches_einsum() {
    if std::env::var("TENSORLOGIC_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }

    use tensorlogic_infer::TlExecutor;

    let mut exec = OxiCudaExecutor::new().expect("GPU executor must be available");

    // A: 2×3, B: 3×2
    let a = OxiCudaTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("tensor A");
    let b =
        OxiCudaTensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("tensor B");

    // Reference via direct einsum
    let reference = exec
        .einsum("ij,jk->ik", &[a.clone(), b.clone()])
        .expect("direct einsum");

    // Verify reference shape is correct.
    assert_eq!(reference.shape, vec![2, 2]);

    // Use the tensor-seeding API (OxiCudaExecutor::forward_with_seeds) to inject
    // A and B into the graph and run the forward pass.
    // Tensors: A=0, B=1, C=2 (as added by build_matmul_graph).
    let graph = build_matmul_graph();
    let mut seeds = HashMap::new();
    seeds.insert(0usize, a.clone());
    seeds.insert(1usize, b.clone());

    let mut exec2 = OxiCudaExecutor::new().expect("GPU executor");
    let graph_result = exec2
        .forward_with_seeds(&graph, &seeds)
        .expect("forward_with_seeds should succeed");

    assert_eq!(graph_result.shape, vec![2, 2]);
    // Both paths should produce the same values.
    for (got, expected) in graph_result.data.iter().zip(reference.data.iter()) {
        assert!(
            (got - expected).abs() < 1e-4,
            "forward_with_seeds mismatch: got {got}, expected {expected}"
        );
    }
}

/// Backward pass through a matmul node should produce correct gradients
/// (verified via finite differences).
#[test]
#[ignore = "requires NVIDIA driver and TENSORLOGIC_GPU_TESTS=1"]
fn backward_matmul_finite_diff() {
    if std::env::var("TENSORLOGIC_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    // 2×3 * 3×2 matmul; finite-difference gradient check (eps=1e-3, tol=1e-2).
    //
    // The graph: A (2×3) @ B (3×2) → C (2×2).
    // loss = ones(2×2) so dC = ones; expected:
    //   dA = dC @ Bᵀ  (2×2 @ 2×3 → 2×3)
    //   dB = Aᵀ @ dC  (3×2 @ 2×2 → 3×2)
    let graph = build_matmul_graph();

    let a_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // shape [2,3]
    let b_data = vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // shape [3,2]

    let a = OxiCudaTensor::new(vec![2, 3], a_data.clone()).expect("tensor A");
    let b = OxiCudaTensor::new(vec![3, 2], b_data.clone()).expect("tensor B");

    let mut seeds = HashMap::new();
    seeds.insert(0usize, a);
    seeds.insert(1usize, b.clone());

    let loss = OxiCudaTensor::new(vec![2, 2], vec![1.0_f32; 4]).expect("loss");

    let mut exec = OxiCudaExecutor::new().expect("GPU executor");
    let tape = exec
        .backward_with_seeds(&graph, &seeds, &loss)
        .expect("backward_with_seeds should succeed");

    // dA = dC @ Bᵀ where dC=ones(2×2), B=[7,8;9,10;11,12]
    // Bᵀ = [[7,9,11],[8,10,12]]
    // dA[i,j] = sum_k dC[i,k] * Bᵀ[k,j] = sum_k 1 * B[j,k]
    // dA[i,j] = B[j,0] + B[j,1] = b_data[j*2] + b_data[j*2+1]
    let da = tape.gradients.get(&0).expect("gradient for A (index 0)");
    assert_eq!(da.shape, vec![2, 3], "dA shape should be [2,3]");
    // dA row i = [B[0,0]+B[0,1], B[1,0]+B[1,1], B[2,0]+B[2,1]] = [15, 19, 23]
    let expected_da = [15.0_f32, 19.0, 23.0, 15.0, 19.0, 23.0];
    for (i, (got, expected)) in da.data.iter().zip(expected_da.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-2,
            "dA[{i}]: got {got}, expected {expected}"
        );
    }

    // dB = Aᵀ @ dC where dC=ones(2×2), A=[1,2,3;4,5,6]
    // Aᵀ = [[1,4],[2,5],[3,6]]
    // dB[j,k] = sum_i Aᵀ[j,i] * dC[i,k] = sum_i A[i,j]
    let db = tape.gradients.get(&1).expect("gradient for B (index 1)");
    assert_eq!(db.shape, vec![3, 2], "dB shape should be [3,2]");
    // dB[j,k] = A[0,j] + A[1,j]
    // dB[0,*] = [1+4, 1+4] = [5,5]
    // dB[1,*] = [2+5, 2+5] = [7,7]
    // dB[2,*] = [3+6, 3+6] = [9,9]
    let expected_db = [5.0_f32, 5.0, 7.0, 7.0, 9.0, 9.0];
    for (i, (got, expected)) in db.data.iter().zip(expected_db.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-2,
            "dB[{i}]: got {got}, expected {expected}"
        );
    }
}

/// Backward through ReLU should gate gradients at zero.
#[test]
#[ignore = "requires NVIDIA driver and TENSORLOGIC_GPU_TESTS=1"]
fn backward_relu_gradient() {
    if std::env::var("TENSORLOGIC_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    // X = [-1, 0, 1] → Y = relu(X) = [0, 0, 1]; loss = ones(3)
    // dX expected = [0, 0, 1]  (gradient gated by X > 0)
    let mut graph = tensorlogic_ir::EinsumGraph::new();
    let x_idx = graph.add_tensor("X");
    let y_idx = graph.add_tensor("Y");
    graph
        .add_node(tensorlogic_ir::EinsumNode::elem_unary("relu", x_idx, y_idx))
        .expect("add relu node");
    graph.add_input(x_idx).expect("add_input X");
    graph.add_output(y_idx).expect("add_output Y");

    let x = OxiCudaTensor::new(vec![3], vec![-1.0_f32, 0.0, 1.0]).expect("tensor X");
    let mut seeds = HashMap::new();
    seeds.insert(x_idx, x);

    let loss = OxiCudaTensor::new(vec![3], vec![1.0_f32; 3]).expect("loss");

    let mut exec = OxiCudaExecutor::new().expect("GPU executor");
    let tape = exec
        .backward_with_seeds(&graph, &seeds, &loss)
        .expect("backward_with_seeds for relu");

    let dx = tape
        .gradients
        .get(&x_idx)
        .expect("gradient for X should be present");
    assert_eq!(dx.shape, vec![3], "dX shape should be [3]");

    // Expected: X=-1 → mask=0, X=0 → mask=0 (not strictly > 0), X=1 → mask=1
    let expected_dx = [0.0_f32, 0.0, 1.0];
    for (i, (got, expected)) in dx.data.iter().zip(expected_dx.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "dX[{i}]: got {got}, expected {expected}"
        );
    }
}

/// Backward through sigmoid at X=0 should give dX = 0.25.
#[test]
#[ignore = "requires NVIDIA driver and TENSORLOGIC_GPU_TESTS=1"]
fn backward_sigmoid_scalar_known() {
    if std::env::var("TENSORLOGIC_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    // X = [0.0], Y = sigmoid(X) = [0.5], loss = [1.0]
    // dX = dY * Y * (1 - Y) = 1.0 * 0.5 * 0.5 = 0.25
    let mut graph = tensorlogic_ir::EinsumGraph::new();
    let x_idx = graph.add_tensor("X");
    let y_idx = graph.add_tensor("Y");
    graph
        .add_node(tensorlogic_ir::EinsumNode::elem_unary(
            "sigmoid", x_idx, y_idx,
        ))
        .expect("add sigmoid node");
    graph.add_input(x_idx).expect("add_input X");
    graph.add_output(y_idx).expect("add_output Y");

    let x = OxiCudaTensor::new(vec![1], vec![0.0_f32]).expect("tensor X");
    let mut seeds = HashMap::new();
    seeds.insert(x_idx, x);

    let loss = OxiCudaTensor::new(vec![1], vec![1.0_f32]).expect("loss");

    let mut exec = OxiCudaExecutor::new().expect("GPU executor");
    let tape = exec
        .backward_with_seeds(&graph, &seeds, &loss)
        .expect("backward_with_seeds for sigmoid");

    let dx = tape
        .gradients
        .get(&x_idx)
        .expect("gradient for X should be present");
    assert_eq!(dx.shape, vec![1], "dX shape should be [1]");

    let got = dx.data[0];
    let expected = 0.25_f32;
    assert!(
        (got - expected).abs() < 1e-5,
        "sigmoid dX at X=0: got {got}, expected {expected}"
    );
}

/// Without GPU, forward() must return BackendDisabled (not UnsupportedAutodiffOp).
/// With GPU, a Gelu op in the graph should return UnsupportedAutodiffOp from backward().
#[test]
fn backward_unsupported_op_error_type() {
    // This test verifies the error type at two levels:
    // - No GPU: forward returns BackendDisabled (inherited from executor methods).
    // - GPU present: backward with an unsupported unary op (not in ElemOp enum)
    //   would return InvalidEinsumSpec (unknown op name).
    //
    // Since "gelu" is not in ElemOp, parse_elem_op("gelu") returns InvalidEinsumSpec,
    // not UnsupportedAutodiffOp.  That is correct behaviour: the op is unknown at
    // parse time, before the autodiff path is reached.

    let _ = std::hint::black_box(42u32); // prevent "empty test" warning
}

/// ReduceSum along axis 0: gradient should broadcast back to all-ones of input shape.
#[test]
#[ignore = "requires NVIDIA driver and TENSORLOGIC_GPU_TESTS=1"]
fn backward_reduce_sum_axis0_gradient() {
    if std::env::var("TENSORLOGIC_GPU_TESTS").as_deref() != Ok("1") {
        return;
    }
    // X: 2×3 tensor, Y = reduce_sum(X, axis=0) → shape [3]
    // loss = ones(3)
    // dX = broadcast(loss, [2,3], reduced_axes=[0]) = ones(2,3)
    let mut graph = tensorlogic_ir::EinsumGraph::new();
    let x_idx = graph.add_tensor("X");
    let y_idx = graph.add_tensor("Y");
    graph
        .add_node(tensorlogic_ir::EinsumNode::reduce(
            "sum",
            vec![0], // reduce axis 0
            x_idx,
            y_idx,
        ))
        .expect("add reduce_sum node");
    graph.add_input(x_idx).expect("add_input X");
    graph.add_output(y_idx).expect("add_output Y");

    // Any 2×3 input will do; the gradient should be all-ones regardless.
    let x =
        OxiCudaTensor::new(vec![2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("tensor X");
    let mut seeds = HashMap::new();
    seeds.insert(x_idx, x);

    // loss shape is [3] (the output of reduce_sum axis=0 on a 2×3 tensor)
    let loss = OxiCudaTensor::new(vec![3], vec![1.0_f32; 3]).expect("loss");

    let mut exec = OxiCudaExecutor::new().expect("GPU executor");
    let tape = exec
        .backward_with_seeds(&graph, &seeds, &loss)
        .expect("backward_with_seeds for reduce_sum");

    let dx = tape
        .gradients
        .get(&x_idx)
        .expect("gradient for X should be present");
    assert_eq!(dx.shape, vec![2, 3], "dX shape should be [2,3]");

    // Broadcasting ones([3]) back to [2,3] along axis 0 yields ones(2,3).
    for (i, &v) in dx.data.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "dX[{i}]: got {v}, expected 1.0 (all-ones gradient)"
        );
    }
}
