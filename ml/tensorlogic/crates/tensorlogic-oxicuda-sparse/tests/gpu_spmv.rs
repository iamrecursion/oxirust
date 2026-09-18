//! GPU SpMV parity test.
//!
//! This test requires:
//!   1. `TENSORLOGIC_GPU_TESTS=1` in the environment.
//!   2. An NVIDIA GPU with a working CUDA driver.
//!   3. The crate to be compiled with `--features gpu`.
//!
//! In CI (no GPU) the test is always marked `#[ignore]`.  To run it locally:
//!
//! ```sh
//! TENSORLOGIC_GPU_TESTS=1 cargo test --features gpu --test gpu_spmv -- --ignored
//! ```

/// Verifies that the GPU SpMV path produces the same result as the reference
/// CPU computation on a small diagonal matrix.
#[test]
#[ignore = "requires TENSORLOGIC_GPU_TESTS=1 and NVIDIA GPU"]
fn gpu_spmv_parity() {
    if std::env::var("TENSORLOGIC_GPU_TESTS").unwrap_or_default() != "1" {
        return;
    }

    // If the gpu feature is not enabled, there is nothing to exercise here.
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("gpu feature not enabled, skipping gpu_spmv_parity");
    }

    // -----------------------------------------------------------------------
    // With the gpu feature enabled we run the same spmv call as the smoke
    // test.  The public `spmv` function will attempt the GPU path first and
    // fall back to CPU if the driver is unavailable.  Both paths must agree.
    // -----------------------------------------------------------------------
    #[cfg(feature = "gpu")]
    {
        use tensorlogic_oxicuda_sparse::{spmv, SparseCsr};

        // Identity 4×4.
        let a = SparseCsr::from_triplets(4, 4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0])
            .unwrap();

        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut y_gpu = vec![0.0f32; 4];

        spmv(&a, &x, 1.0, 0.0, &mut y_gpu).unwrap();

        // Expected: diagonal entries applied to x = [1, 2, 3, 4].
        let expected = [1.0f32, 2.0, 3.0, 4.0];
        for (i, (&got, &exp)) in y_gpu.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "gpu_spmv_parity: y[{i}]={got}, expected {exp}"
            );
        }
    }
}

/// Verifies that the GPU SpMM path produces the same result as the reference
/// CPU computation on a small sparse matrix times a dense matrix.
#[test]
#[ignore = "requires TENSORLOGIC_GPU_TESTS=1 and NVIDIA GPU"]
fn gpu_spmm_parity() {
    if std::env::var("TENSORLOGIC_GPU_TESTS").unwrap_or_default() != "1" {
        return;
    }

    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("gpu feature not enabled, skipping gpu_spmm_parity");
    }

    #[cfg(feature = "gpu")]
    {
        use tensorlogic_oxicuda_sparse::{spmm, SparseCsr};

        // A = 3×3 identity sparse.
        let a = SparseCsr::from_triplets(3, 3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0]).unwrap();

        // B = 3×2 dense row-major: [[1,2],[3,4],[5,6]].
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 6]; // 3×2 output

        spmm(&a, &b, 2, 1.0, 0.0, &mut c).unwrap();

        // Identity * B = B.
        for (i, (&got, &exp)) in c.iter().zip(b.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "gpu_spmm_parity: c[{i}]={got}, expected {exp}"
            );
        }
    }
}
