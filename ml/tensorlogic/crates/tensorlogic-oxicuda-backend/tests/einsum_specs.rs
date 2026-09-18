//! Tests for the einsum spec parser and dispatcher.
//!
//! - Error-path tests (unknown spec → `UnsupportedSpec`) do NOT require a GPU.
//! - Functional tests (batched matmul, identity) are gated behind
//!   `TENSORLOGIC_GPU_TESTS=1` and `#[ignore]`.
//!
//! Run GPU tests with:
//! ```bash
//! TENSORLOGIC_GPU_TESTS=1 cargo test -p tensorlogic-oxicuda-backend \
//!   --features gpu -- --include-ignored einsum
//! ```

use tensorlogic_oxicuda_backend::{OxiCudaBackendError, OxiCudaExecutor};

#[cfg(feature = "gpu")]
use tensorlogic_oxicuda_backend::OxiCudaTensor;

#[cfg(feature = "gpu")]
use tensorlogic_infer::TlExecutor;

// ---------------------------------------------------------------------------
// Helpers
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
// Error-path test: unknown spec returns UnsupportedSpec (no GPU needed)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
fn unknown_spec_returns_unsupported_spec() {
    let mut exec = match OxiCudaExecutor::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; test is compile-only");
            return;
        }
    };

    let a = make_tensor(vec![2, 2], vec![0.0_f32; 4]);
    let result = exec.einsum("ii->i", &[a]);
    match result {
        Err(OxiCudaBackendError::UnsupportedSpec(_)) => {}
        other => panic!("expected UnsupportedSpec, got {other:?}"),
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn no_gpu_executor_returns_backend_disabled() {
    match OxiCudaExecutor::new() {
        Err(OxiCudaBackendError::BackendDisabled) => {}
        other => panic!("expected BackendDisabled, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Identity spec: "ij->ij" (ignored, GPU, but cheapest possible test)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn identity_spec_returns_clone() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let a = make_tensor(vec![2, 2], data.clone());
    let out = match exec.einsum("ij->ij", &[a]) {
        Ok(t) => t,
        Err(e) => panic!("identity einsum failed: {e}"),
    };

    assert_eq!(out.shape, vec![2, 2]);
    assert_close(&out.data, &data, 1e-6, "identity");
}

// ---------------------------------------------------------------------------
// Batched matmul: "bij,bjk->bik" with [2,3,4] x [2,4,5]
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn batched_matmul_2x3x4_times_2x4x5() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    // A: [2, 3, 4] — filled with 1.0
    // B: [2, 4, 5] — filled with 1.0
    // C = A*B: [2, 3, 5], each element = sum of k=4 ones = 4.0
    let batch = 2;
    let m = 3;
    let k = 4;
    let n = 5;
    let a = make_tensor(vec![batch, m, k], vec![1.0_f32; batch * m * k]);
    let b = make_tensor(vec![batch, k, n], vec![1.0_f32; batch * k * n]);

    let out = match exec.einsum("bij,bjk->bik", &[a, b]) {
        Ok(t) => t,
        Err(e) => panic!("batched matmul failed: {e}"),
    };

    assert_eq!(out.shape, vec![batch, m, n], "output shape mismatch");
    let expected = vec![k as f32; batch * m * n];
    assert_close(&out.data, &expected, 1e-3, "bij,bjk->bik (ones)");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore]
fn batched_matmul_non_trivial_values() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match try_init_exec() {
        Some(e) => e,
        None => return,
    };

    // batch=1, A: [1,2,3], B: [1,3,2]
    // A[0] = [[1,2,3],[4,5,6]], B[0] = [[1,2],[3,4],[5,6]]
    // C[0] = A[0]*B[0] = [[22,28],[49,64]]
    let a_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = make_tensor(vec![1, 2, 3], a_data);
    let b = make_tensor(vec![1, 3, 2], b_data);

    let out = match exec.einsum("bij,bjk->bik", &[a, b]) {
        Ok(t) => t,
        Err(e) => panic!("batched matmul failed: {e}"),
    };

    assert_eq!(out.shape, vec![1, 2, 2]);
    let expected = vec![22.0_f32, 28.0, 49.0, 64.0];
    assert_close(&out.data, &expected, 1e-3, "batched non-trivial");
}
