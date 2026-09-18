//! Comprehensive tests for SIMD-accelerated preprocessing operations
//!
//! Tests normalize_simd and standardize_simd functions with:
//! - Basic correctness (f32/f64)
//! - Edge cases (empty, single element, zero vector, constant)
//! - Large arrays (SIMD path)
//! - SIMD vs scalar equivalence
//! - Numerical properties (unit norm, zero mean, unit variance)

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::preprocessing::{clip_simd, normalize_simd, standardize_simd};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============================================================================
// Normalize SIMD Tests
// ============================================================================

#[test]
fn test_normalize_simd_f64_basic() {
    let x = array![3.0, 4.0];
    let result = normalize_simd(&x.view());

    // 3-4-5 triangle: norm = 5, so [0.6, 0.8]
    assert!((result[0] - 0.6_f64).abs() < 1e-10);
    assert!((result[1] - 0.8_f64).abs() < 1e-10);

    // Verify unit norm
    let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_normalize_simd_f32_basic() {
    let x = array![3.0f32, 4.0];
    let result = normalize_simd(&x.view());

    assert!((result[0] - 0.6).abs() < 1e-6);
    assert!((result[1] - 0.8).abs() < 1e-6);

    // Verify unit norm
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_normalize_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = normalize_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_normalize_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = normalize_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_normalize_simd_f64_single() {
    let x = array![5.0];
    let result = normalize_simd(&x.view());

    // Single element: norm = 5, so result = 1.0
    assert!((result[0] - 1.0_f64).abs() < 1e-10);
}

#[test]
fn test_normalize_simd_f32_single() {
    let x = array![5.0f32];
    let result = normalize_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_normalize_simd_f64_zero_vector() {
    let x = array![0.0, 0.0, 0.0];
    let result = normalize_simd(&x.view());

    // Zero norm should return zero array
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test]
fn test_normalize_simd_f32_zero_vector() {
    let x = array![0.0f32, 0.0, 0.0];
    let result = normalize_simd(&x.view());

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test]
fn test_normalize_simd_f64_negative() {
    let x = array![-3.0, 4.0];
    let result = normalize_simd(&x.view());

    // norm = 5, so [-0.6, 0.8]
    assert!((result[0] - (-0.6_f64)).abs() < 1e-10);
    assert!((result[1] - 0.8_f64).abs() < 1e-10);

    // Verify unit norm
    let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_normalize_simd_f32_negative() {
    let x = array![-3.0f32, 4.0];
    let result = normalize_simd(&x.view());

    assert!((result[0] - (-0.6)).abs() < 1e-6);
    assert!((result[1] - 0.8).abs() < 1e-6);

    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_normalize_simd_f64_large_array() {
    // Create large array for SIMD path
    let data: Vec<f64> = (0..10000).map(|i| (i as f64).sin()).collect();
    let x = Array1::from_vec(data);
    let result = normalize_simd(&x.view());

    // Verify unit norm
    let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-8,
        "Large array norm should be ~1, got {}",
        norm
    );
}

#[test]
fn test_normalize_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000).map(|i| (i as f32).sin()).collect();
    let x = Array1::from_vec(data);
    let result = normalize_simd(&x.view());

    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Large array norm should be ~1, got {}",
        norm
    );
}

#[cfg(feature = "random")]
#[test]
fn test_normalize_simd_equivalence_random() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // SIMD result
    let simd_result = normalize_simd(&x.view());

    // Scalar result (manual computation)
    let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    let expected: Vec<f64> = data.iter().map(|x| x / norm).collect();

    // Compare SIMD vs scalar
    for i in 0..data.len() {
        let diff = (simd_result[i] - expected[i]).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            simd_result[i],
            expected[i]
        );
    }

    // Verify unit norm
    let result_norm: f64 = simd_result.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((result_norm - 1.0).abs() < 1e-10);
}

// ============================================================================
// Standardize SIMD Tests
// ============================================================================

#[test]
fn test_standardize_simd_f64_basic() {
    let x = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let result = standardize_simd(&x.view());

    // Verify mean ≈ 0
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    assert!(mean.abs() < 1e-10, "Mean should be ~0, got {}", mean);

    // Verify std ≈ 1 (sample std with ddof=1)
    let variance: f64 = result.iter().map(|&x| x * x).sum::<f64>() / (result.len() - 1) as f64;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-10, "Std should be ~1, got {}", std);
}

#[test]
fn test_standardize_simd_f32_basic() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = standardize_simd(&x.view());

    // Verify mean ≈ 0
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

    // Verify std ≈ 1
    let variance: f32 = result.iter().map(|&x| x * x).sum::<f32>() / (result.len() - 1) as f32;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-5, "Std should be ~1, got {}", std);
}

#[test]
fn test_standardize_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = standardize_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_standardize_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = standardize_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_standardize_simd_f64_single_element() {
    let x = array![5.0];
    let result = standardize_simd(&x.view());

    // Single element has undefined std, should return zero
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_standardize_simd_f32_single_element() {
    let x = array![5.0f32];
    let result = standardize_simd(&x.view());

    assert_eq!(result[0], 0.0);
}

#[test]
fn test_standardize_simd_f64_constant() {
    let x = array![5.0, 5.0, 5.0, 5.0];
    let result = standardize_simd(&x.view());

    // Constant array has zero std, should return zero array
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 0.0);
}

#[test]
fn test_standardize_simd_f32_constant() {
    let x = array![5.0f32, 5.0, 5.0, 5.0];
    let result = standardize_simd(&x.view());

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 0.0);
}

#[test]
fn test_standardize_simd_f64_negative() {
    let x = array![-10.0, -5.0, 0.0, 5.0, 10.0];
    let result = standardize_simd(&x.view());

    // Verify mean ≈ 0
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    assert!(mean.abs() < 1e-10, "Mean should be ~0, got {}", mean);

    // Verify std ≈ 1
    let variance: f64 = result.iter().map(|&x| x * x).sum::<f64>() / (result.len() - 1) as f64;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-10, "Std should be ~1, got {}", std);
}

#[test]
fn test_standardize_simd_f32_negative() {
    let x = array![-10.0f32, -5.0, 0.0, 5.0, 10.0];
    let result = standardize_simd(&x.view());

    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5);

    let variance: f32 = result.iter().map(|&x| x * x).sum::<f32>() / (result.len() - 1) as f32;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-5);
}

#[test]
fn test_standardize_simd_f64_large_array() {
    // Create large array for SIMD path
    let data: Vec<f64> = (0..10000).map(|i| (i as f64).sin() * 100.0).collect();
    let x = Array1::from_vec(data);
    let result = standardize_simd(&x.view());

    // Verify mean ≈ 0
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    assert!(
        mean.abs() < 1e-10,
        "Large array mean should be ~0, got {}",
        mean
    );

    // Verify std ≈ 1
    let variance: f64 = result.iter().map(|&x| x * x).sum::<f64>() / (result.len() - 1) as f64;
    let std = variance.sqrt();
    assert!(
        (std - 1.0).abs() < 1e-8,
        "Large array std should be ~1, got {}",
        std
    );
}

#[test]
fn test_standardize_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000).map(|i| (i as f32).sin() * 100.0).collect();
    let x = Array1::from_vec(data);
    let result = standardize_simd(&x.view());

    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5);

    let variance: f32 = result.iter().map(|&x| x * x).sum::<f32>() / (result.len() - 1) as f32;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-5);
}

#[cfg(feature = "random")]
#[test]
fn test_standardize_simd_equivalence_random() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // SIMD result
    let simd_result = standardize_simd(&x.view());

    // Scalar result (manual computation)
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();
    let expected: Vec<f64> = data.iter().map(|x| (x - mean) / std).collect();

    // Compare SIMD vs scalar
    for i in 0..data.len() {
        let diff = (simd_result[i] - expected[i]).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            simd_result[i],
            expected[i]
        );
    }

    // Verify mean ≈ 0
    let result_mean: f64 = simd_result.iter().sum::<f64>() / simd_result.len() as f64;
    assert!(result_mean.abs() < 1e-10);

    // Verify std ≈ 1
    let result_variance: f64 =
        simd_result.iter().map(|&x| x * x).sum::<f64>() / (simd_result.len() - 1) as f64;
    let result_std = result_variance.sqrt();
    assert!((result_std - 1.0).abs() < 1e-10);
}

// ============================================================================
// Edge Case and Property Tests
// ============================================================================

#[test]
fn test_normalize_simd_all_same_sign() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = normalize_simd(&x.view());

    // All positive, should still have unit norm
    let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);

    // All elements should be positive
    for &val in result.iter() {
        assert!(val > 0.0);
    }
}

#[test]
fn test_standardize_simd_symmetric_distribution() {
    let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = standardize_simd(&x.view());

    // Symmetric distribution should maintain symmetry
    let sum_first_last: f64 = result[0] + result[4];
    assert!(sum_first_last.abs() < 1e-10); // First and last should be opposites
    let sum_second_fourth: f64 = result[1] + result[3];
    assert!(sum_second_fourth.abs() < 1e-10); // Second and fourth should be opposites
    let middle: f64 = result[2];
    assert!(middle.abs() < 1e-10); // Middle should be ~0
}

#[test]
fn test_normalize_simd_f64_precision() {
    // Test with very small values to check numerical stability
    let x = array![1e-100, 2e-100, 3e-100];
    let result = normalize_simd(&x.view());

    // Should still have unit norm
    let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_standardize_simd_f64_precision() {
    // Test with very large values to check numerical stability
    let x = array![1e100, 2e100, 3e100, 4e100, 5e100];
    let result = standardize_simd(&x.view());

    // Should still have mean ≈ 0 and std ≈ 1
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    assert!(mean.abs() < 1e-10);

    let variance: f64 = result.iter().map(|&x| x * x).sum::<f64>() / (result.len() - 1) as f64;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-10);
}
// ============================================================================
// Clip SIMD Tests
// ============================================================================

#[test]
fn test_clip_simd_f64_basic() {
    let x = array![-10.0, -5.0, 0.0, 5.0, 10.0];
    let result = clip_simd(&x.view(), -3.0, 7.0);

    assert_eq!(result[0], -3.0); // -10 clipped to -3
    assert_eq!(result[1], -3.0); // -5 clipped to -3
    assert_eq!(result[2], 0.0); // 0 unchanged
    assert_eq!(result[3], 5.0); // 5 unchanged
    assert_eq!(result[4], 7.0); // 10 clipped to 7
}

#[test]
fn test_clip_simd_f32_basic() {
    let x = array![-10.0f32, -5.0, 0.0, 5.0, 10.0];
    let result = clip_simd(&x.view(), -3.0, 7.0);

    assert_eq!(result[0], -3.0);
    assert_eq!(result[1], -3.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 5.0);
    assert_eq!(result[4], 7.0);
}

#[test]
fn test_clip_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = clip_simd(&x.view(), -1.0, 1.0);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_clip_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = clip_simd(&x.view(), -1.0, 1.0);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_clip_simd_f64_all_within_range() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = clip_simd(&x.view(), 0.0, 10.0);

    // All values should be unchanged
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_clip_simd_f32_all_within_range() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = clip_simd(&x.view(), 0.0, 10.0);

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_clip_simd_f64_all_below_min() {
    let x = array![-10.0, -8.0, -6.0, -4.0, -2.0];
    let result = clip_simd(&x.view(), 0.0, 10.0);

    // All values should be clipped to min
    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_clip_simd_f32_all_below_min() {
    let x = array![-10.0f32, -8.0, -6.0, -4.0, -2.0];
    let result = clip_simd(&x.view(), 0.0, 10.0);

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_clip_simd_f64_all_above_max() {
    let x = array![12.0, 14.0, 16.0, 18.0, 20.0];
    let result = clip_simd(&x.view(), 0.0, 10.0);

    // All values should be clipped to max
    for &val in result.iter() {
        assert_eq!(val, 10.0);
    }
}

#[test]
fn test_clip_simd_f32_all_above_max() {
    let x = array![12.0f32, 14.0, 16.0, 18.0, 20.0];
    let result = clip_simd(&x.view(), 0.0, 10.0);

    for &val in result.iter() {
        assert_eq!(val, 10.0);
    }
}

#[test]
fn test_clip_simd_f64_equal_bounds() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = clip_simd(&x.view(), 3.0, 3.0);

    // All values should be set to 3.0
    for &val in result.iter() {
        assert_eq!(val, 3.0);
    }
}

#[test]
fn test_clip_simd_f32_equal_bounds() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = clip_simd(&x.view(), 3.0, 3.0);

    for &val in result.iter() {
        assert_eq!(val, 3.0);
    }
}

#[test]
fn test_clip_simd_f64_negative_range() {
    let x = array![-10.0, -5.0, 0.0, 5.0, 10.0];
    let result = clip_simd(&x.view(), -8.0, -2.0);

    assert_eq!(result[0], -8.0); // -10 clipped to -8
    assert_eq!(result[1], -5.0); // -5 unchanged
    assert_eq!(result[2], -2.0); // 0 clipped to -2
    assert_eq!(result[3], -2.0); // 5 clipped to -2
    assert_eq!(result[4], -2.0); // 10 clipped to -2
}

#[test]
fn test_clip_simd_f32_negative_range() {
    let x = array![-10.0f32, -5.0, 0.0, 5.0, 10.0];
    let result = clip_simd(&x.view(), -8.0, -2.0);

    assert_eq!(result[0], -8.0);
    assert_eq!(result[1], -5.0);
    assert_eq!(result[2], -2.0);
    assert_eq!(result[3], -2.0);
    assert_eq!(result[4], -2.0);
}

#[test]
fn test_clip_simd_f64_large_array() {
    // Test SIMD path with large array
    let data: Vec<f64> = (0..10000).map(|i| (i as f64).sin() * 100.0).collect();
    let x = Array1::from_vec(data.clone());
    let result = clip_simd(&x.view(), -50.0, 50.0);

    // Verify all values are within bounds
    for &val in result.iter() {
        assert!((-50.0..=50.0).contains(&val));
    }

    // Verify clipping actually occurred
    let min_original = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_original = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(min_original < -50.0 || max_original > 50.0);
}

#[test]
fn test_clip_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000).map(|i| (i as f32).sin() * 100.0).collect();
    let x = Array1::from_vec(data);
    let result = clip_simd(&x.view(), -50.0, 50.0);

    for &val in result.iter() {
        assert!((-50.0..=50.0).contains(&val));
    }
}

#[cfg(feature = "random")]
#[test]
fn test_clip_simd_equivalence_random() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    let min_val = -30.0;
    let max_val = 40.0;

    // SIMD result
    let simd_result = clip_simd(&x.view(), min_val, max_val);

    // Scalar result (manual computation)
    let expected: Vec<f64> = data
        .iter()
        .map(|&x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        })
        .collect();

    // Compare SIMD vs scalar
    for i in 0..data.len() {
        let diff = (simd_result[i] - expected[i]).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            simd_result[i],
            expected[i]
        );
    }

    // Verify bounds
    for &val in simd_result.iter() {
        assert!(val >= min_val && val <= max_val);
    }
}

#[test]
fn test_clip_simd_gradient_clipping_use_case() {
    // Simulate gradient clipping scenario
    let gradients = array![10.5, -8.2, 0.3, 15.7, -12.1, 2.5, -0.9, 7.3];
    let clip_value = 5.0;
    let clipped = clip_simd(&gradients.view(), -clip_value, clip_value);

    // Verify all gradients are within [-5, 5]
    for &val in clipped.iter() {
        assert!(val >= -clip_value && val <= clip_value);
    }

    // Check specific values
    assert_eq!(clipped[0], 5.0); // 10.5 -> 5.0
    assert_eq!(clipped[1], -5.0); // -8.2 -> -5.0
    assert_eq!(clipped[2], 0.3); // unchanged
    assert_eq!(clipped[3], 5.0); // 15.7 -> 5.0
    assert_eq!(clipped[4], -5.0); // -12.1 -> -5.0
    assert_eq!(clipped[5], 2.5); // unchanged
    assert_eq!(clipped[6], -0.9); // unchanged
    assert_eq!(clipped[7], 5.0); // 7.3 -> 5.0
}

#[test]
#[should_panic(expected = "min_val must be <= max_val")]
fn test_clip_simd_invalid_bounds() {
    let x = array![1.0, 2.0, 3.0];
    let _ = clip_simd(&x.view(), 10.0, 5.0); // min > max, should panic
}

#[test]
fn test_clip_simd_zero_centered() {
    let x = array![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];
    let result = clip_simd(&x.view(), -2.0, 2.0);

    assert_eq!(result[0], -2.0);
    assert_eq!(result[1], -2.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], 0.0);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 2.0);
    assert_eq!(result[6], 2.0);
}
