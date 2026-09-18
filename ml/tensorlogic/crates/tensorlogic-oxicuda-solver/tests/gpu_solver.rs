//! GPU solver integration tests.
//!
//! These tests are compiled and run only when the `gpu` feature is enabled.
//! In CI (no CUDA device) they should be skipped by not enabling the feature.
//! They exist to verify the dispatch layer compiles and the GPU-unavailable
//! path returns the expected error variant.

// When the gpu feature is off, this whole file provides nothing —
// that is intentional (the test binary still compiles cleanly).

#[cfg(feature = "gpu")]
mod gpu_dispatch_tests {
    use tensorlogic_oxicuda_solver::{
        cg_solve, solve_cholesky, solve_lu, solve_qr_lstsq, SolverError,
    };

    /// Confirm that on a machine without a real CUDA device (CI), the GPU dispatch
    /// falls back to the CPU path rather than panicking.  The Round-5 stub always
    /// returns `gpu_available() == false`, so CPU is always taken here.
    #[test]
    fn gpu_lu_falls_back_to_cpu_when_unavailable() {
        let a = vec![1f32, 0., 0., 1.];
        let b = vec![3f32, 7.];
        // Must succeed because the stub routes to CPU.
        let x = solve_lu(&a, 2, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 7.0).abs() < 1e-4, "x[1]={}", x[1]);
    }

    #[test]
    fn gpu_cholesky_falls_back_to_cpu_when_unavailable() {
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let x = solve_cholesky(&a, 2, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }

    #[test]
    fn gpu_qr_falls_back_to_cpu_when_unavailable() {
        let a = vec![1f32, 0., 0., 1., 1., 1.];
        let b = vec![1f32, 1., 2.];
        let x = solve_qr_lstsq(&a, 3, 2, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }

    #[test]
    fn gpu_cg_falls_back_to_cpu_when_unavailable() {
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let x = cg_solve(&a, 2, &b, 100, 1e-6).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }

    /// Verify the SolverError::GpuError variant is constructable and formats correctly.
    /// This exercises the error path that will be live when GPU is genuinely unavailable
    /// at runtime in Round 6.
    #[test]
    fn gpu_error_variant_formats_correctly() {
        let err = SolverError::GpuError("test error".to_string());
        let msg = err.to_string();
        assert!(msg.contains("GPU solver error"), "unexpected msg: {msg}");
        assert!(msg.contains("test error"), "unexpected msg: {msg}");
    }
}

// Always-compiled placeholder so rustc does not complain about an empty file
// when the `gpu` feature is off.
#[allow(dead_code)]
fn _placeholder() {}
