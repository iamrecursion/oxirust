//! Compilation test for v0.2.0 SIMD enhancements
//!
//! This test ensures all new modules compile correctly.

#[test]
fn test_modules_compile() {
    // Import all new modules to ensure they compile
    use scirs2_stats::correlation_simd_enhanced;
    use scirs2_stats::parallel_simd_stats;
    use scirs2_stats::sampling_simd;

    // This test passes if compilation succeeds
}

#[test]
fn test_basic_functionality() {
    use scirs2_core::ndarray::array;
    use scirs2_stats::correlation_simd_enhanced::spearman_r_simd;
    use scirs2_stats::sampling_simd::box_muller_simd;

    // Test Box-Muller sampling
    let samples = box_muller_simd(100, 0.0, 1.0, Some(42));
    assert!(samples.is_ok());

    // Test Spearman correlation
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
    let rho = spearman_r_simd(&x.view(), &y.view());
    assert!(rho.is_ok());
}
