//! GPU integration test for the SGEMM matmul path.
//!
//! Requires:
//! - `--features gpu` at compile time.
//! - `TENSORLOGIC_GPU_TESTS=1` environment variable at runtime.
//! - A machine with an NVIDIA GPU and a working driver.
//!
//! Run with:
//! ```bash
//! TENSORLOGIC_GPU_TESTS=1 cargo test -p tensorlogic-oxicuda-backend --features gpu -- --include-ignored gpu_sgemm
//! ```

#![cfg(feature = "gpu")]

use tensorlogic_infer::TlExecutor;
use tensorlogic_oxicuda_backend::{OxiCudaExecutor, OxiCudaTensor};

fn gpu_tests_enabled() -> bool {
    std::env::var("TENSORLOGIC_GPU_TESTS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
#[ignore]
fn sgemm_identity_2x2() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match OxiCudaExecutor::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; skipping");
            return;
        }
    };

    // A = [[1, 2], [3, 4]], B = I_2
    let a = OxiCudaTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = OxiCudaTensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let a = match a {
        Ok(t) => t,
        Err(e) => panic!("tensor build A failed: {e}"),
    };
    let b = match b {
        Ok(t) => t,
        Err(e) => panic!("tensor build B failed: {e}"),
    };

    let c = match exec.einsum("ij,jk->ik", &[a.clone(), b]) {
        Ok(t) => t,
        Err(e) => panic!("SGEMM failed: {e}"),
    };

    assert_eq!(c.shape, vec![2, 2]);
    for (got, want) in c.data.iter().zip(a.data.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "mismatch: got {got}, want {want}"
        );
    }
}

#[test]
#[ignore]
fn sgemm_3x2_times_2x4() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match OxiCudaExecutor::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; skipping");
            return;
        }
    };

    // A (3x2): [[1,2],[3,4],[5,6]]
    // B (2x4): [[7,8,9,10],[11,12,13,14]]
    // C (3x4) = A*B row-major
    let a = OxiCudaTensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = OxiCudaTensor::new(
        vec![2, 4],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
    );
    let a = match a {
        Ok(t) => t,
        Err(e) => panic!("tensor build A failed: {e}"),
    };
    let b = match b {
        Ok(t) => t,
        Err(e) => panic!("tensor build B failed: {e}"),
    };

    let c = match exec.einsum("ij,jk->ik", &[a, b]) {
        Ok(t) => t,
        Err(e) => panic!("SGEMM failed: {e}"),
    };

    assert_eq!(c.shape, vec![3, 4]);

    #[rustfmt::skip]
    let expected: [f32; 12] = [
        29.0, 32.0, 35.0, 38.0,
        65.0, 72.0, 79.0, 86.0,
        101.0, 112.0, 123.0, 134.0,
    ];

    for (i, (got, want)) in c.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "element {i}: got {got}, want {want}"
        );
    }
}

#[test]
#[ignore]
fn sgemm_rejects_wrong_input_count_on_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("TENSORLOGIC_GPU_TESTS != 1; skipping");
        return;
    }
    let mut exec = match OxiCudaExecutor::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("GPU executor init failed: {err}; skipping");
            return;
        }
    };

    let a = match OxiCudaTensor::new(vec![2, 3], vec![0.0; 6]) {
        Ok(t) => t,
        Err(e) => panic!("tensor build failed: {e}"),
    };

    let result = exec.einsum("ij,jk->ik", &[a]);
    assert!(result.is_err(), "single-input matmul should fail");
}
