//! Basic math SIMD tests: abs, sqrt, floor, ceil, round, sign, clamp

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{
    abs_simd, ceil_simd, clamp_simd, floor_simd, recip_simd, round_simd, sign_simd, sqrt_simd,
};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============================================================================
// Abs SIMD Tests
// ============================================================================

#[test]
fn test_abs_simd_f64_basic() {
    let x = array![-3.0, -1.5, 0.0, 1.5, 3.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 1.5);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.5);
    assert_eq!(result[4], 3.0);
}

#[test]
fn test_abs_simd_f32_basic() {
    let x = array![-3.0f32, -1.5, 0.0, 1.5, 3.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 1.5);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.5);
    assert_eq!(result[4], 3.0);
}

#[test]
fn test_abs_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = abs_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_abs_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = abs_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_abs_simd_f64_all_positive() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = abs_simd(&x.view());

    // All positive values should be unchanged
    for i in 0..x.len() {
        assert_eq!(result[i], x[i]);
    }
}

#[test]
fn test_abs_simd_f32_all_positive() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = abs_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i]);
    }
}

#[test]
fn test_abs_simd_f64_all_negative() {
    let x = array![-1.0, -2.0, -3.0, -4.0, -5.0];
    let result = abs_simd(&x.view());

    // All negative values should become positive
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_abs_simd_f32_all_negative() {
    let x = array![-1.0f32, -2.0, -3.0, -4.0, -5.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_abs_simd_f64_zero() {
    let x = array![0.0, -0.0, 0.0];
    let result = abs_simd(&x.view());

    // Both +0 and -0 should become +0
    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_abs_simd_f32_zero() {
    let x = array![0.0f32, -0.0, 0.0];
    let result = abs_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_abs_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
    assert_eq!(result[1], f64::INFINITY); // -∞ becomes +∞
}

#[test]
fn test_abs_simd_f32_infinity() {
    let x = array![f32::INFINITY, f32::NEG_INFINITY];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
    assert_eq!(result[1], f32::INFINITY);
}

#[test]
fn test_abs_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = abs_simd(&x.view());

    // abs(NaN) should be NaN
    assert!(result[0].is_nan());
}

#[test]
fn test_abs_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = abs_simd(&x.view());

    assert!(result[0].is_nan());
}

#[test]
fn test_abs_simd_f64_large_array() {
    // Test SIMD path with large array
    let data: Vec<f64> = (0..10000)
        .map(|i| ((i as f64) * 0.1).sin() * 100.0)
        .collect();
    let x = Array1::from_vec(data.clone());
    let result = abs_simd(&x.view());

    // Verify all values are non-negative
    for &val in result.iter() {
        assert!(val >= 0.0);
    }

    // Verify correctness
    for i in 0..data.len() {
        assert_eq!(result[i], data[i].abs());
    }
}

#[test]
fn test_abs_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000)
        .map(|i| ((i as f32) * 0.1).sin() * 100.0)
        .collect();
    let x = Array1::from_vec(data.clone());
    let result = abs_simd(&x.view());

    for &val in result.iter() {
        assert!(val >= 0.0);
    }

    for i in 0..data.len() {
        assert_eq!(result[i], data[i].abs());
    }
}

#[cfg(feature = "random")]
#[test]
fn test_abs_simd_equivalence_random() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // SIMD result
    let simd_result = abs_simd(&x.view());

    // Scalar result
    let expected: Vec<f64> = data.iter().map(|&x| x.abs()).collect();

    // Compare
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
}

// ============================================================================
// ============================================================================
// Sqrt SIMD Tests
// ============================================================================

#[test]
fn test_sqrt_simd_f64_basic() {
    let x = array![1.0, 4.0, 9.0, 16.0, 25.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_sqrt_simd_f32_basic() {
    let x = array![1.0f32, 4.0, 9.0, 16.0, 25.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_sqrt_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = sqrt_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sqrt_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = sqrt_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sqrt_simd_f64_zero() {
    let x = array![0.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 0.0);
}

#[test]
fn test_sqrt_simd_f32_zero() {
    let x = array![0.0f32];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 0.0);
}

#[test]
fn test_sqrt_simd_f64_negative() {
    let x = array![-1.0];
    let result = sqrt_simd(&x.view());

    // sqrt of negative should be NaN
    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_sqrt_simd_f32_negative() {
    let x = array![-1.0f32];
    let result = sqrt_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_sqrt_simd_f64_infinity() {
    let x = array![f64::INFINITY];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
}

#[test]
fn test_sqrt_simd_f32_infinity() {
    let x = array![f32::INFINITY];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
}

#[test]
fn test_sqrt_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = sqrt_simd(&x.view());

    // sqrt(NaN) should be NaN
    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_sqrt_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = sqrt_simd(&x.view());

    assert!(result[0].is_nan());
}

#[test]
fn test_sqrt_simd_f64_small_values() {
    let x = array![0.01, 0.04, 0.09, 0.16, 0.25];
    let result = sqrt_simd(&x.view());

    let diff0: f64 = result[0] - 0.1;
    assert!(diff0.abs() < 1e-10);
    let diff1: f64 = result[1] - 0.2;
    assert!(diff1.abs() < 1e-10);
    let diff2: f64 = result[2] - 0.3;
    assert!(diff2.abs() < 1e-10);
    let diff3: f64 = result[3] - 0.4;
    assert!(diff3.abs() < 1e-10);
    let diff4: f64 = result[4] - 0.5;
    assert!(diff4.abs() < 1e-10);
}

#[test]
fn test_sqrt_simd_f32_small_values() {
    let x = array![0.01f32, 0.04, 0.09, 0.16, 0.25];
    let result = sqrt_simd(&x.view());

    assert!((result[0] - 0.1).abs() < 1e-6);
    assert!((result[1] - 0.2).abs() < 1e-6);
    assert!((result[2] - 0.3).abs() < 1e-6);
    assert!((result[3] - 0.4).abs() < 1e-6);
    assert!((result[4] - 0.5).abs() < 1e-6);
}

#[test]
fn test_sqrt_simd_f64_large_array() {
    // Test SIMD path with large array
    let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    let x = Array1::from_vec(data.clone());
    let result = sqrt_simd(&x.view());

    // Verify correctness
    for i in 0..data.len() {
        let expected = data[i].sqrt();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            result[i],
            expected
        );
    }
}

#[test]
fn test_sqrt_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
    let x = Array1::from_vec(data.clone());
    let result = sqrt_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].sqrt();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            result[i],
            expected
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_sqrt_simd_equivalence_random() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 100.0).expect("Test: operation failed"); // Only positive values
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // SIMD result
    let simd_result = sqrt_simd(&x.view());

    // Scalar result
    let expected: Vec<f64> = data.iter().map(|&x| x.sqrt()).collect();

    // Compare
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
}

// ============================================================================
// Property Tests
// ============================================================================

#[test]
fn test_abs_simd_idempotent() {
    // abs(abs(x)) = abs(x)
    let x = array![-3.0, -1.0, 0.0, 1.0, 3.0];
    let abs_once = abs_simd(&x.view());
    let abs_twice = abs_simd(&abs_once.view());

    for i in 0..x.len() {
        assert_eq!(abs_once[i], abs_twice[i]);
    }
}

#[cfg(feature = "random")]
#[test]
fn test_abs_simd_non_negative() {
    // abs(x) >= 0 for all x
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data);
    let result = abs_simd(&x.view());

    for &val in result.iter() {
        assert!(val >= 0.0 || val.is_nan());
    }
}

#[test]
fn test_sqrt_abs_non_negative() {
    // sqrt(abs(x)) is always defined
    let x = array![-9.0, -4.0, 0.0, 4.0, 9.0];
    let abs_x = abs_simd(&x.view());
    let sqrt_abs_x = sqrt_simd(&abs_x.view());

    // All values should be non-NaN
    for &val in sqrt_abs_x.iter() {
        let v: f64 = val;
        assert!(!v.is_nan());
    }

    // Check specific values
    assert_eq!(sqrt_abs_x[0], 3.0); // sqrt(abs(-9)) = sqrt(9) = 3
    assert_eq!(sqrt_abs_x[1], 2.0); // sqrt(abs(-4)) = sqrt(4) = 2
    assert_eq!(sqrt_abs_x[2], 0.0); // sqrt(abs(0)) = sqrt(0) = 0
    assert_eq!(sqrt_abs_x[3], 2.0); // sqrt(abs(4)) = sqrt(4) = 2
    assert_eq!(sqrt_abs_x[4], 3.0); // sqrt(abs(9)) = sqrt(9) = 3
}

// ============================================================================
// ============================================================================
// Floor SIMD Tests
// ============================================================================

#[test]
fn test_floor_simd_f64_basic() {
    let x = array![1.2_f64, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = floor_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], -2.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_floor_simd_f32_basic() {
    let x = array![1.2_f32, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = floor_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], -2.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_floor_simd_empty() {
    let x: Array1<f64> = array![];
    let result = floor_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_floor_simd_integers() {
    let x = array![1.0_f64, 2.0, 3.0, -1.0, -2.0];
    let result = floor_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i], "floor(integer) should equal integer");
    }
}

#[test]
fn test_floor_simd_nan_propagation() {
    let x = array![1.5_f64, f64::NAN, 2.5];
    let result = floor_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 2.0);
}

// Mathematical property: floor(x) <= x
#[test]
fn test_floor_property_less_than_or_equal() {
    let x = array![1.1_f64, 2.9, -0.5, -1.7, 3.5, 0.01];
    let result = floor_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] <= x[i],
            "floor(x) should be <= x, but floor({}) = {} > {}",
            x[i],
            result[i],
            x[i]
        );
    }
}

// Relationship: floor(-x) = -ceil(x)
#[test]
fn test_floor_ceil_relationship() {
    let x = array![1.2_f64, 2.7, 3.5, 0.1, 0.9];
    let neg_x = x.mapv(|v| -v);

    let floor_neg_x = floor_simd(&neg_x.view());
    let ceil_x = ceil_simd(&x.view());
    let neg_ceil_x = ceil_x.mapv(|v| -v);

    for i in 0..x.len() {
        assert_eq!(
            floor_neg_x[i], neg_ceil_x[i],
            "floor(-x) should equal -ceil(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_floor_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = floor_simd(&x.view());

    // Verify correctness for first 100 elements
    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].floor());
    }
}

// ============================================================================
// Ceil SIMD Tests
// ============================================================================

#[test]
fn test_ceil_simd_f64_basic() {
    let x = array![1.2_f64, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = ceil_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -2.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 1.0);
    assert_eq!(result[6], 0.0);
}

#[test]
fn test_ceil_simd_f32_basic() {
    let x = array![1.2_f32, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = ceil_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -2.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 1.0);
    assert_eq!(result[6], 0.0);
}

#[test]
fn test_ceil_simd_empty() {
    let x: Array1<f64> = array![];
    let result = ceil_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ceil_simd_integers() {
    let x = array![1.0_f64, 2.0, 3.0, -1.0, -2.0];
    let result = ceil_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i], "ceil(integer) should equal integer");
    }
}

#[test]
fn test_ceil_simd_nan_propagation() {
    let x = array![1.5_f64, f64::NAN, 2.5];
    let result = ceil_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 3.0);
}

// Mathematical property: ceil(x) >= x
#[test]
fn test_ceil_property_greater_than_or_equal() {
    let x = array![1.1_f64, 2.9, -0.5, -1.7, 3.5, 0.01];
    let result = ceil_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] >= x[i],
            "ceil(x) should be >= x, but ceil({}) = {} < {}",
            x[i],
            result[i],
            x[i]
        );
    }
}

// Relationship: ceil(-x) = -floor(x)
#[test]
fn test_ceil_floor_relationship() {
    let x = array![1.2_f64, 2.7, 3.5, 0.1, 0.9];
    let neg_x = x.mapv(|v| -v);

    let ceil_neg_x = ceil_simd(&neg_x.view());
    let floor_x = floor_simd(&x.view());
    let neg_floor_x = floor_x.mapv(|v| -v);

    for i in 0..x.len() {
        assert_eq!(
            ceil_neg_x[i], neg_floor_x[i],
            "ceil(-x) should equal -floor(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_ceil_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = ceil_simd(&x.view());

    // Verify correctness for first 100 elements
    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].ceil());
    }
}

// ============================================================================
// Round SIMD Tests
// ============================================================================

#[test]
fn test_round_simd_f64_basic() {
    let x = array![1.2_f64, 2.7, -1.3, -2.9, 3.0, 0.4, -0.6];
    let result = round_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_round_simd_f32_basic() {
    let x = array![1.2_f32, 2.7, -1.3, -2.9, 3.0, 0.4, -0.6];
    let result = round_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_round_simd_empty() {
    let x: Array1<f64> = array![];
    let result = round_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_round_simd_integers() {
    let x = array![1.0_f64, 2.0, 3.0, -1.0, -2.0];
    let result = round_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i], "round(integer) should equal integer");
    }
}

// Round half away from zero (Rust default behavior)
#[test]
fn test_round_simd_half_ties() {
    let x = array![0.5_f64, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5];
    let result = round_simd(&x.view());

    // Rust rounds half away from zero
    assert_eq!(result[0], 1.0); // 0.5 -> 1.0
    assert_eq!(result[1], 2.0); // 1.5 -> 2.0
    assert_eq!(result[2], 3.0); // 2.5 -> 3.0
    assert_eq!(result[3], 4.0); // 3.5 -> 4.0
    assert_eq!(result[4], 5.0); // 4.5 -> 5.0
    assert_eq!(result[5], -1.0); // -0.5 -> -1.0
    assert_eq!(result[6], -2.0); // -1.5 -> -2.0
    assert_eq!(result[7], -3.0); // -2.5 -> -3.0
}

#[test]
fn test_round_simd_nan_propagation() {
    let x = array![1.5_f64, f64::NAN, 2.5];
    let result = round_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 3.0);
}

// Mathematical property: |round(x) - x| <= 0.5
#[test]
fn test_round_property_distance() {
    let x = array![1.1_f64, 2.9, 3.5, -0.7, -1.5, 0.01, 4.99];
    let result = round_simd(&x.view());

    for i in 0..x.len() {
        let distance = (result[i] - x[i]).abs();
        assert!(
            distance <= 0.5,
            "|round({}) - {}| = {} should be <= 0.5",
            x[i],
            x[i],
            distance
        );
    }
}

// Symmetry: round(-x) = -round(x)
#[test]
fn test_round_symmetry() {
    let x = array![1.2_f64, 2.7, 3.3, 0.9];
    let neg_x = x.mapv(|v| -v);

    let round_x = round_simd(&x.view());
    let round_neg_x = round_simd(&neg_x.view());

    for i in 0..x.len() {
        assert_eq!(
            round_neg_x[i], -round_x[i],
            "round(-x) should equal -round(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_round_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = round_simd(&x.view());

    // Verify correctness for first 100 elements
    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].round());
    }
}

// ============================================================================
// Cross-function Rounding Tests
// ============================================================================

// Property: floor(x) <= round(x) <= ceil(x)
#[test]
fn test_rounding_ordering() {
    let x = array![1.1_f64, 2.9, 3.5, -0.7, -1.5];

    let floor_result = floor_simd(&x.view());
    let round_result = round_simd(&x.view());
    let ceil_result = ceil_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            floor_result[i] <= round_result[i] && round_result[i] <= ceil_result[i],
            "floor({}) <= round({}) <= ceil({}) not satisfied",
            x[i],
            x[i],
            x[i]
        );
    }
}

// Property: ceil(x) - floor(x) is 0 or 1
#[test]
fn test_ceil_minus_floor() {
    let x = array![1.1_f64, 2.0, 3.5, -0.7, -1.0];

    let floor_result = floor_simd(&x.view());
    let ceil_result = ceil_simd(&x.view());

    for i in 0..x.len() {
        let diff = ceil_result[i] - floor_result[i];
        assert!(
            diff == 0.0 || diff == 1.0,
            "ceil(x) - floor(x) should be 0 or 1, got {} for x={}",
            diff,
            x[i]
        );
    }
}

// ============================================================================
// ============================================================================
// Sign SIMD Tests
// ============================================================================

// Basic correctness f64
#[test]
fn test_sign_simd_f64_basic() {
    let x = array![-3.0, -1.5, 0.0, 1.5, 3.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0);
    assert_eq!(result[1], -1.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 1.0);
}

// Basic correctness f32
#[test]
fn test_sign_simd_f32_basic() {
    let x = array![-3.0_f32, -1.5, 0.0, 1.5, 3.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0);
    assert_eq!(result[1], -1.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 1.0);
}

// Empty array
#[test]
fn test_sign_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let result = sign_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// All positive values
#[test]
fn test_sign_simd_all_positive() {
    let x = array![0.1_f64, 1.0, 5.0, 100.0, 1e6];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

// All negative values
#[test]
fn test_sign_simd_all_negative() {
    let x = array![-0.1_f64, -1.0, -5.0, -100.0, -1e6];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, -1.0);
    }
}

// All zeros
#[test]
fn test_sign_simd_all_zeros() {
    let x = array![0.0_f64, 0.0, 0.0, 0.0, 0.0];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// Mixed values
#[test]
fn test_sign_simd_mixed() {
    let x = array![-100.0_f64, -0.001, 0.0, 0.001, 100.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0);
    assert_eq!(result[1], -1.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 1.0);
}

// Property: sign(-x) = -sign(x) (odd function)
#[test]
fn test_sign_odd_function() {
    let x = array![-5.0_f64, -2.0, -1.0, 1.0, 2.0, 5.0];
    let neg_x = -&x;

    let sign_x = sign_simd(&x.view());
    let sign_neg_x = sign_simd(&neg_x.view());

    for i in 0..x.len() {
        assert_eq!(sign_neg_x[i], -sign_x[i]);
    }
}

// Property: sign(x) * |x| = x
#[test]
fn test_sign_times_abs_equals_x() {
    let x = array![-5.0_f64, -2.0, 0.0, 2.0, 5.0];
    let signs = sign_simd(&x.view());
    let abs_values = x.mapv(|v| v.abs());
    let reconstructed = signs * abs_values;

    for i in 0..x.len() {
        assert!((reconstructed[i] - x[i]).abs() < 1e-10);
    }
}

// Property: sign(x) * sign(y) = sign(x * y)
#[test]
fn test_sign_product_rule() {
    let x = array![-3.0_f64, -1.0, 2.0, 5.0];
    let y = array![2.0_f64, -4.0, -1.0, 3.0];

    let sign_x = sign_simd(&x.view());
    let sign_y = sign_simd(&y.view());
    let product = &x * &y;
    let sign_product = sign_simd(&product.view());

    let sign_x_times_sign_y = &sign_x * &sign_y;

    for i in 0..x.len() {
        assert_eq!(sign_x_times_sign_y[i], sign_product[i]);
    }
}

// Property: |sign(x)| ≤ 1 for all x
#[test]
fn test_sign_bounded() {
    let x = array![-1e10_f64, -100.0, -0.001, 0.0, 0.001, 100.0, 1e10];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert!(val.abs() <= 1.0);
    }
}

// NaN handling (NaN comparisons are false, so sign(NaN) = 0.0)
#[test]
fn test_sign_nan_handling() {
    let x = array![f64::NAN, 1.0, -1.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], 0.0); // NaN → 0.0 (neither > 0 nor < 0)
    assert_eq!(result[1], 1.0);
    assert_eq!(result[2], -1.0);
}

// Infinity handling
#[test]
fn test_sign_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], 1.0); // +inf → +1
    assert_eq!(result[1], -1.0); // -inf → -1
    assert_eq!(result[2], 0.0); // 0 → 0
}

// Large array SIMD path
#[test]
fn test_sign_large_array() {
    let size = 10_000;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let val = (i as f64) - (size as f64 / 2.0);
        data.push(val);
    }
    let x = Array1::from_vec(data);
    let result = sign_simd(&x.view());

    for i in 0..size {
        let expected = if x[i] > 0.0 {
            1.0
        } else if x[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        assert_eq!(result[i], expected);
    }
}

// Very small values (near zero)
#[test]
fn test_sign_very_small_values() {
    let x = array![-1e-300_f64, -1e-100, 1e-100, 1e-300];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0); // very small negative → -1
    assert_eq!(result[1], -1.0); // small negative → -1
    assert_eq!(result[2], 1.0); // small positive → +1
    assert_eq!(result[3], 1.0); // very small positive → +1
}

// Application: gradient direction
#[test]
fn test_sign_gradient_direction() {
    let gradients = array![-0.5_f32, -0.1, 0.0, 0.1, 0.5];
    let directions = sign_simd(&gradients.view());

    // In gradient descent, we move opposite to the gradient
    let descent_directions = -directions;

    assert_eq!(descent_directions[0], 1.0); // negative gradient → move right
    assert_eq!(descent_directions[1], 1.0);
    assert_eq!(descent_directions[2], 0.0); // zero gradient → no movement
    assert_eq!(descent_directions[3], -1.0); // positive gradient → move left
    assert_eq!(descent_directions[4], -1.0);
}

// ============================================================================
// ============================================================================
// Clamp SIMD Tests
// ============================================================================

#[test]
fn test_clamp_simd_f64_basic() {
    let x = array![-1.0_f64, 0.5, 1.5, 2.5, 3.5];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    assert_eq!(result[0], 0.0); // Below min
    assert_eq!(result[1], 0.5); // Within range
    assert_eq!(result[2], 1.5); // Within range
    assert_eq!(result[3], 2.0); // Above max
    assert_eq!(result[4], 2.0); // Above max
}

#[test]
fn test_clamp_simd_f32_basic() {
    let x = array![-1.0_f32, 0.5, 1.5, 2.5, 3.5];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.5);
    assert_eq!(result[2], 1.5);
    assert_eq!(result[3], 2.0);
    assert_eq!(result[4], 2.0);
}

#[test]
fn test_clamp_simd_empty() {
    let x: Array1<f64> = array![];
    let result = clamp_simd(&x.view(), 0.0, 1.0);
    assert_eq!(result.len(), 0);
}

// Boundary conditions: values at min/max
#[test]
fn test_clamp_boundary_values() {
    let x = array![0.0_f64, 1.0, 2.0];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    assert_eq!(result[0], 0.0); // At min
    assert_eq!(result[1], 1.0); // Within
    assert_eq!(result[2], 2.0); // At max
}

// All values below min
#[test]
fn test_clamp_all_below_min() {
    let x = array![-5.0_f64, -3.0, -1.0];
    let result = clamp_simd(&x.view(), 0.0, 10.0);

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// All values above max
#[test]
fn test_clamp_all_above_max() {
    let x = array![11.0_f64, 15.0, 20.0];
    let result = clamp_simd(&x.view(), 0.0, 10.0);

    for &val in result.iter() {
        assert_eq!(val, 10.0);
    }
}

// All values within range (should remain unchanged)
#[test]
fn test_clamp_all_within_range() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = clamp_simd(&x.view(), 0.0, 10.0);

    for i in 0..x.len() {
        assert_eq!(result[i], x[i]);
    }
}

// Idempotency: clamp(clamp(x)) = clamp(x)
#[test]
fn test_clamp_idempotent() {
    let x = array![-5.0_f64, 0.5, 5.0, 15.0];
    let result1 = clamp_simd(&x.view(), 0.0, 10.0);
    let result2 = clamp_simd(&result1.view(), 0.0, 10.0);

    for i in 0..x.len() {
        assert_eq!(result1[i], result2[i], "Clamp should be idempotent");
    }
}

// Monotonicity: if x1 <= x2, then clamp(x1) <= clamp(x2)
#[test]
fn test_clamp_preserves_monotonicity() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    for i in 1..x.len() {
        assert!(
            result[i - 1] <= result[i],
            "Clamp should preserve monotonicity"
        );
    }
}

// Edge case: min == max (all values should equal this value)
#[test]
fn test_clamp_min_equals_max() {
    let x = array![-1.0_f64, 0.5, 1.5, 2.5];
    let result = clamp_simd(&x.view(), 1.0, 1.0);

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

// Range bounds test
#[test]
fn test_clamp_result_in_bounds() {
    let x = array![-10.0_f64, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0];
    let min = -5.0;
    let max = 10.0;
    let result = clamp_simd(&x.view(), min, max);

    for &val in result.iter() {
        assert!(
            val >= min && val <= max,
            "Clamped value {} should be in [{}, {}]",
            val,
            min,
            max
        );
    }
}

// Property: clamp(x, min, max) = max(min, min(x, max))
#[test]
fn test_clamp_mathematical_definition() {
    let x = array![-5.0_f64, 0.0, 2.5, 7.0, 12.0];
    let min = 0.0;
    let max = 10.0;

    let result = clamp_simd(&x.view(), min, max);
    let expected = x.mapv(|val| val.max(min).min(max));

    for i in 0..x.len() {
        assert_eq!(result[i], expected[i]);
    }
}

// NaN handling
#[test]
fn test_clamp_nan_propagation() {
    let x = array![1.0_f64, f64::NAN, 3.0];
    let result = clamp_simd(&x.view(), 0.0, 5.0);

    assert_eq!(result[0], 1.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 3.0);
}

// Negative range
#[test]
fn test_clamp_negative_range() {
    let x = array![-10.0_f64, -5.0, 0.0, 5.0];
    let result = clamp_simd(&x.view(), -7.0, -2.0);

    assert_eq!(result[0], -7.0); // Below min
    assert_eq!(result[1], -5.0); // Within
    assert_eq!(result[2], -2.0); // Above max
    assert_eq!(result[3], -2.0); // Above max
}

// Large array (SIMD path)
#[cfg(feature = "random")]
#[test]
fn test_clamp_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let min = -50.0;
    let max = 50.0;
    let result = clamp_simd(&x.view(), min, max);

    for i in 0..100.min(data.len()) {
        let expected = data[i].clamp(min, max);
        assert_eq!(result[i], expected);
        assert!(result[i] >= min && result[i] <= max);
    }
}

// Pixel normalization use case (0.0 - 1.0)
#[test]
fn test_clamp_pixel_normalization() {
    let pixels = array![-0.5_f64, 0.0, 0.3, 0.7, 1.0, 1.5];
    let result = clamp_simd(&pixels.view(), 0.0, 1.0);

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.3);
    assert_eq!(result[3], 0.7);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 1.0);
}

// Byte range normalization (0 - 255)
#[test]
fn test_clamp_byte_range() {
    let values = array![-50.0_f32, 0.0, 127.5, 255.0, 300.0];
    let result = clamp_simd(&values.view(), 0.0, 255.0);

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 127.5);
    assert_eq!(result[3], 255.0);
    assert_eq!(result[4], 255.0);
}

// ============================================================================
