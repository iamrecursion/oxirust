//! Smoke tests for elementwise unary and binary op dispatch.
//!
//! Non-GPU path: verifies that `elem_op` and `elem_op_binary` return
//! `BackendDisabled` when the `gpu` feature is off.
//!
//! GPU path (gated by `TENSORLOGIC_GPU_TESTS=1` + `#[ignore]`): round-trips
//! each supported op through the GPU and compares the result against a CPU
//! oracle within epsilon.
//!
//! Run GPU tests with:
//! ```bash
//! TENSORLOGIC_GPU_TESTS=1 cargo test -p tensorlogic-oxicuda-backend \
//!   --features gpu -- --include-ignored elem_ops
//! ```

#[cfg(feature = "gpu")]
use tensorlogic_infer::ElemOp;
use tensorlogic_oxicuda_backend::{OxiCudaBackendError, OxiCudaExecutor};

#[cfg(feature = "gpu")]
use tensorlogic_oxicuda_backend::OxiCudaTensor;

// ---------------------------------------------------------------------------
// Without GPU feature: BackendDisabled
// ---------------------------------------------------------------------------

#[cfg(not(feature = "gpu"))]
#[test]
fn no_gpu_elem_op_returns_backend_disabled() {
    // Cannot construct a real executor — just verify the error variant.
    match OxiCudaExecutor::new() {
        Err(OxiCudaBackendError::BackendDisabled) => {}
        other => panic!("expected BackendDisabled, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn gpu_tests_enabled() -> bool {
    std::env::var("TENSORLOGIC_GPU_TESTS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[cfg(feature = "gpu")]
fn try_init_exec() -> Option<OxiCudaExecutor> {
    match OxiCudaExecutor::new() {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; skipping");
            None
        }
    }
}

#[cfg(feature = "gpu")]
fn make_tensor(shape: Vec<usize>, data: Vec<f32>) -> OxiCudaTensor {
    match OxiCudaTensor::new(shape, data) {
        Ok(t) => t,
        Err(e) => panic!("tensor build failed: {e}"),
    }
}

/// Apply `op` element-wise to `data` using the CPU oracle.
#[cfg(feature = "gpu")]
fn cpu_unary(op: ElemOp, data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&x| match op {
            ElemOp::Relu => x.max(0.0),
            ElemOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            _ => panic!("cpu_unary: unsupported op {op:?}"),
        })
        .collect()
}

/// Apply binary `op` element-wise to `a` and `b` using the CPU oracle.
#[cfg(feature = "gpu")]
fn cpu_binary(op: ElemOp, a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| match op {
            ElemOp::Add => x + y,
            ElemOp::Multiply => x * y,
            _ => panic!("cpu_binary: unsupported op {op:?}"),
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn assert_close(got: &[f32], want: &[f32], eps: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= eps,
            "{label}[{i}]: got {g}, want {w}, eps {eps}"
        );
    }
}

// ---------------------------------------------------------------------------
// GPU unary tests (ignored without env var)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_relu_small_buffer() {
    use tensorlogic_infer::TlExecutor;

    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0, 3.0];
    let x = make_tensor(vec![6], data.clone());
    let out = match exec.elem_op(ElemOp::Relu, &x) {
        Ok(t) => t,
        Err(e) => panic!("relu failed: {e}"),
    };

    let expected = cpu_unary(ElemOp::Relu, &data);
    assert_close(&out.data, &expected, 1e-5, "relu");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_sigmoid_small_buffer() {
    use tensorlogic_infer::TlExecutor;

    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let data = vec![-1.0_f32, 0.0, 1.0, 2.0];
    let x = make_tensor(vec![4], data.clone());
    let out = match exec.elem_op(ElemOp::Sigmoid, &x) {
        Ok(t) => t,
        Err(e) => panic!("sigmoid failed: {e}"),
    };

    let expected = cpu_unary(ElemOp::Sigmoid, &data);
    assert_close(&out.data, &expected, 1e-5, "sigmoid");
}

// ---------------------------------------------------------------------------
// GPU binary tests (ignored without env var)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_add_small_buffers() {
    use tensorlogic_infer::TlExecutor;

    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let a_data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let b_data = vec![10.0_f32, 20.0, 30.0, 40.0];
    let x = make_tensor(vec![4], a_data.clone());
    let y = make_tensor(vec![4], b_data.clone());

    let out = match exec.elem_op_binary(ElemOp::Add, &x, &y) {
        Ok(t) => t,
        Err(e) => panic!("add failed: {e}"),
    };

    let expected = cpu_binary(ElemOp::Add, &a_data, &b_data);
    assert_close(&out.data, &expected, 1e-5, "add");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn gpu_multiply_small_buffers() {
    use tensorlogic_infer::TlExecutor;

    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let a_data = vec![2.0_f32, 3.0, 4.0, 5.0];
    let b_data = vec![1.5_f32, 2.0, 2.5, 3.0];
    let x = make_tensor(vec![4], a_data.clone());
    let y = make_tensor(vec![4], b_data.clone());

    let out = match exec.elem_op_binary(ElemOp::Multiply, &x, &y) {
        Ok(t) => t,
        Err(e) => panic!("multiply failed: {e}"),
    };

    let expected = cpu_binary(ElemOp::Multiply, &a_data, &b_data);
    assert_close(&out.data, &expected, 1e-5, "multiply");
}

// ---------------------------------------------------------------------------
// Error path: unsupported op does NOT require GPU
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
fn gpu_unsupported_unary_returns_err() {
    use tensorlogic_infer::TlExecutor;

    let mut exec = match OxiCudaExecutor::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; test is compile-only");
            return;
        }
    };

    let x = make_tensor(vec![4], vec![1.0; 4]);
    // OneMinus is a supported unary op; use Add (a binary op) to trigger UnsupportedUnary.
    let result = exec.elem_op(ElemOp::Add, &x);
    match result {
        Err(OxiCudaBackendError::UnsupportedUnary(_)) => {}
        other => panic!("expected UnsupportedUnary, got {other:?}"),
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_unsupported_binary_returns_err() {
    use tensorlogic_infer::TlExecutor;

    let mut exec = match OxiCudaExecutor::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; test is compile-only");
            return;
        }
    };

    let x = make_tensor(vec![4], vec![1.0; 4]);
    let y = make_tensor(vec![4], vec![2.0; 4]);
    // Subtract is a supported binary op; use Relu (a unary op) to trigger UnsupportedBinary.
    let result = exec.elem_op_binary(ElemOp::Relu, &x, &y);
    match result {
        Err(OxiCudaBackendError::UnsupportedBinary(_)) => {}
        other => panic!("expected UnsupportedBinary, got {other:?}"),
    }
}
