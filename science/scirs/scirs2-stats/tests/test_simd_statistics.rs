//! SIMD statistics integration tests
//!
//! This module tests the SIMD-accelerated variance, std, and weighted statistics
//! implementations in scirs2-stats, ensuring correctness and performance.

use scirs2_core::ndarray::{array, Array1};
use scirs2_stats::{std, var, weighted_mean};

// ============================================================================
// Variance Tests (ddof=1 uses SIMD, ddof=0 uses fallback)
// ============================================================================

#[test]
fn test_simd_variance_f64_basic() {
    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];

    // Sample variance (ddof=1) - should use SIMD
    let result = var(&data.view(), 1, None).expect("Test: operation failed");

    // Expected sample variance: 2.5
    let expected = 2.5f64;
    assert!(
        (result - expected).abs() < 1e-10,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_variance_f32_basic() {
    let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];

    // Sample variance (ddof=1) - should use SIMD
    let result = var(&data.view(), 1, None).expect("Test: operation failed");

    let expected = 2.5f32;
    assert!(
        (result - expected).abs() < 1e-6,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_variance_f64_large() {
    // Test with large array to ensure SIMD path is used
    let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());

    // Sample variance (ddof=1)
    let result = var(&data.view(), 1, None).expect("Test: operation failed");

    // Expected variance for 0..1000: approximately 83416.67
    let expected = 83416.66666666667f64;
    assert!(
        (result - expected).abs() / expected < 1e-10,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_variance_f32_large() {
    let data: Array1<f32> = Array1::from_vec((0..1000).map(|i| i as f32).collect());

    // Sample variance (ddof=1)
    let result = var(&data.view(), 1, None).expect("Test: operation failed");

    let expected = 83416.67f32;
    assert!(
        (result - expected).abs() / expected < 1e-4,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_variance_ddof_comparison() {
    // Test that ddof=0 (population variance) vs ddof=1 (sample variance) work correctly
    let data = array![2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    // Population variance (ddof=0) - uses fallback path
    let pop_var = var(&data.view(), 0, None).expect("Test: operation failed");
    // Expected: 4.0
    assert!((pop_var - 4.0).abs() < 1e-10);

    // Sample variance (ddof=1) - uses SIMD path
    let sample_var = var(&data.view(), 1, None).expect("Test: operation failed");
    // Expected: 4.571428571428571
    assert!((sample_var - 4.571428571428571).abs() < 1e-10);
}

// ============================================================================
// Standard Deviation Tests (ddof=1 uses SIMD, ddof=0 uses fallback)
// ============================================================================

#[test]
fn test_simd_std_f64_basic() {
    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];

    // Sample std (ddof=1) - should use SIMD
    let result = std(&data.view(), 1, None).expect("Test: operation failed");

    // Expected sample std: sqrt(2.5) = 1.5811388300841898
    let expected = 1.5811388300841898f64;
    assert!(
        (result - expected).abs() < 1e-10,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_std_f32_basic() {
    let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];

    // Sample std (ddof=1) - should use SIMD
    let result = std(&data.view(), 1, None).expect("Test: operation failed");

    let expected = 1.5811388f32;
    assert!(
        (result - expected).abs() < 1e-5,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_std_f64_large() {
    // Test with large array to ensure SIMD path is used
    let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());

    // Sample std (ddof=1)
    let result = std(&data.view(), 1, None).expect("Test: operation failed");

    // Expected std for 0..1000: approximately 288.82
    let expected = 288.8194360957494f64;
    assert!(
        (result - expected).abs() / expected < 1e-10,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_std_f32_large() {
    let data: Array1<f32> = Array1::from_vec((0..1000).map(|i| i as f32).collect());

    // Sample std (ddof=1)
    let result = std(&data.view(), 1, None).expect("Test: operation failed");

    let expected = 288.8194f32;
    assert!(
        (result - expected).abs() / expected < 1e-4,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_std_ddof_comparison() {
    // Test that ddof=0 (population std) vs ddof=1 (sample std) work correctly
    let data = array![2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    // Population std (ddof=0) - uses fallback path
    let pop_std = std(&data.view(), 0, None).expect("Test: operation failed");
    // Expected: 2.0
    assert!((pop_std - 2.0).abs() < 1e-10);

    // Sample std (ddof=1) - uses SIMD path
    let sample_std = std(&data.view(), 1, None).expect("Test: operation failed");
    // Expected: 2.138089935299395
    assert!((sample_std - 2.138089935299395).abs() < 1e-10);
}

// ============================================================================
// Weighted Mean Tests (uses SIMD for large arrays)
// ============================================================================

#[test]
fn test_simd_weighted_mean_f64_basic() {
    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let weights = array![1.0f64, 1.0, 1.0, 1.0, 1.0];

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    // With equal weights, should equal regular mean: 3.0
    let expected = 3.0f64;
    assert!(
        (result - expected).abs() < 1e-10,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_weighted_mean_f32_basic() {
    let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let weights = array![1.0f32, 1.0, 1.0, 1.0, 1.0];

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    let expected = 3.0f32;
    assert!(
        (result - expected).abs() < 1e-6,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_weighted_mean_f64_unequal_weights() {
    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let weights = array![5.0f64, 4.0, 3.0, 2.0, 1.0];

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    // Expected: (1*5 + 2*4 + 3*3 + 4*2 + 5*1) / (5+4+3+2+1) = 35/15 = 2.333...
    let expected = 2.3333333333333335f64;
    assert!(
        (result - expected).abs() < 1e-10,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_weighted_mean_f32_unequal_weights() {
    let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let weights = array![5.0f32, 4.0, 3.0, 2.0, 1.0];

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    let expected = 2.3333333f32;
    assert!(
        (result - expected).abs() < 1e-5,
        "Got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn test_simd_weighted_mean_f64_large() {
    // Test with large array to ensure SIMD path is used
    let data: Array1<f64> = Array1::from_vec((0..1000).map(|i| i as f64).collect());
    let weights: Array1<f64> =
        Array1::from_vec((0..1000).map(|i| 1.0 + (i as f64 / 1000.0)).collect());

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    // With increasing weights, mean is biased toward higher values: approximately 555
    let expected = 555.0740246748916f64;
    assert!(
        (result - expected).abs() < 1.0,
        "Got {}, expected approximately {}",
        result,
        expected
    );
}

#[test]
fn test_simd_weighted_mean_f32_large() {
    let data: Array1<f32> = Array1::from_vec((0..1000).map(|i| i as f32).collect());
    let weights: Array1<f32> =
        Array1::from_vec((0..1000).map(|i| 1.0 + (i as f32 / 1000.0)).collect());

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    // With increasing weights, mean is biased toward higher values: approximately 555
    let expected = 555.07404f32;
    assert!(
        (result - expected).abs() < 1.0,
        "Got {}, expected approximately {}",
        result,
        expected
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_simd_variance_negative_values() {
    let data = array![-5.0f64, -3.0, -1.0, 1.0, 3.0, 5.0];

    // Sample variance (ddof=1)
    let result = var(&data.view(), 1, None).expect("Test: operation failed");

    // Expected variance: 14.0
    let expected = 14.0f64;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_std_negative_values() {
    let data = array![-5.0f64, -3.0, -1.0, 1.0, 3.0, 5.0];

    // Sample std (ddof=1)
    let result = std(&data.view(), 1, None).expect("Test: operation failed");

    // Expected std: sqrt(14.0) = 3.7416573867739413
    let expected = 3.7416573867739413f64;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_variance_uniform_values() {
    // All same values should have variance = 0
    let data = array![5.0f64, 5.0, 5.0, 5.0, 5.0];

    let result = var(&data.view(), 1, None).expect("Test: operation failed");

    assert!(
        result.abs() < 1e-10,
        "Variance of uniform values should be 0"
    );
}

#[test]
fn test_simd_std_uniform_values() {
    // All same values should have std = 0
    let data = array![5.0f64, 5.0, 5.0, 5.0, 5.0];

    let result = std(&data.view(), 1, None).expect("Test: operation failed");

    assert!(result.abs() < 1e-10, "Std of uniform values should be 0");
}

#[test]
fn test_simd_weighted_mean_single_weight() {
    // Only one non-zero weight
    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let weights = array![0.0f64, 0.0, 1.0, 0.0, 0.0];

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    // Should equal the value at index 2: 3.0
    let expected = 3.0f64;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_simd_variance_very_large_array() {
    // Test with very large array (10K elements)
    let data: Array1<f64> = Array1::from_vec((0..10000).map(|i| (i as f64) * 0.1).collect());

    // Sample variance (ddof=1) - should definitely use SIMD
    let result = var(&data.view(), 1, None).expect("Test: operation failed");

    // Expected variance for 0..10000 with scale 0.1: 0.01 * 8333416.67 = 83334.17
    assert!(
        result > 83300.0 && result < 83400.0,
        "Variance should be approximately 83334, got {}",
        result
    );
}

#[test]
fn test_simd_std_very_large_array() {
    // Test with very large array (10K elements)
    let data: Array1<f64> = Array1::from_vec((0..10000).map(|i| (i as f64) * 0.1).collect());

    // Sample std (ddof=1) - should definitely use SIMD
    let result = std(&data.view(), 1, None).expect("Test: operation failed");

    // Expected std: sqrt(83334.17) = approximately 288.68
    assert!(
        result > 288.0 && result < 289.0,
        "Std should be approximately 288.68, got {}",
        result
    );
}

#[test]
fn test_simd_weighted_mean_very_large_array() {
    // Test with very large array (10K elements)
    let data: Array1<f64> = Array1::from_vec((0..10000).map(|i| i as f64).collect());
    let weights: Array1<f64> = Array1::ones(10000);

    let result = weighted_mean(&data.view(), &weights.view()).expect("Test: operation failed");

    // With equal weights, should equal regular mean: 4999.5
    let expected = 4999.5f64;
    assert!(
        (result - expected).abs() < 1e-6,
        "Got {}, expected {}",
        result,
        expected
    );
}
