//! Phase 62: SIMD geometry tests (hypot and copysign)

use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Phase 62: SIMD hypot (hypotenuse) tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::copysign_simd;
use scirs2_core::ndarray_ext::elementwise::hypot_simd;

/// Test hypot basic f32 correctness with Pythagorean triples
#[test]
fn test_hypot_simd_f32_basic() {
    // Well-known Pythagorean triples
    let x = array![3.0_f32, 5.0, 8.0, 7.0];
    let y = array![4.0_f32, 12.0, 15.0, 24.0];

    let result = hypot_simd(&x.view(), &y.view());

    assert!((result[0] - 5.0).abs() < 1e-5); // 3-4-5
    assert!((result[1] - 13.0).abs() < 1e-5); // 5-12-13
    assert!((result[2] - 17.0).abs() < 1e-5); // 8-15-17
    assert!((result[3] - 25.0).abs() < 1e-5); // 7-24-25
}

/// Test hypot basic f64 correctness
#[test]
fn test_hypot_simd_f64_basic() {
    let x = array![3.0_f64, 5.0, 8.0, 7.0];
    let y = array![4.0_f64, 12.0, 15.0, 24.0];

    let result = hypot_simd(&x.view(), &y.view());

    assert!((result[0] - 5.0).abs() < 1e-14);
    assert!((result[1] - 13.0).abs() < 1e-14);
    assert!((result[2] - 17.0).abs() < 1e-14);
    assert!((result[3] - 25.0).abs() < 1e-14);
}

/// Test hypot empty array
#[test]
fn test_hypot_simd_empty() {
    let x: Array1<f64> = array![];
    let y: Array1<f64> = array![];
    let result = hypot_simd(&x.view(), &y.view());
    assert_eq!(result.len(), 0);
}

/// Test hypot large array (SIMD path)
#[test]
fn test_hypot_simd_large_array() {
    let n = 10000;
    // Generate values that form 3-4-5 scaled triangles
    let x = Array1::from_vec((0..n).map(|i| 3.0 * i as f64).collect());
    let y = Array1::from_vec((0..n).map(|i| 4.0 * i as f64).collect());

    let result = hypot_simd(&x.view(), &y.view());
    assert_eq!(result.len(), n);

    // Check: hypot(3i, 4i) = 5i
    for i in [0, 100, 1000, 5000, 9999] {
        let expected = 5.0 * i as f64;
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "hypot[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test hypot with zeros
#[test]
fn test_hypot_simd_zeros() {
    let x = array![0.0_f64, 5.0, 0.0];
    let y = array![3.0_f64, 0.0, 0.0];

    let result = hypot_simd(&x.view(), &y.view());

    assert!((result[0] - 3.0).abs() < 1e-14); // hypot(0, 3) = 3
    assert!((result[1] - 5.0).abs() < 1e-14); // hypot(5, 0) = 5
    assert!((result[2] - 0.0).abs() < 1e-14); // hypot(0, 0) = 0
}

/// Test hypot with negative values
#[test]
fn test_hypot_simd_negative_values() {
    let x = array![-3.0_f64, -5.0, 3.0, -3.0];
    let y = array![4.0_f64, -12.0, -4.0, -4.0];

    let result = hypot_simd(&x.view(), &y.view());

    // hypot should always return positive values regardless of input signs
    assert!((result[0] - 5.0).abs() < 1e-14);
    assert!((result[1] - 13.0).abs() < 1e-14);
    assert!((result[2] - 5.0).abs() < 1e-14);
    assert!((result[3] - 5.0).abs() < 1e-14);
}

/// Test hypot overflow protection
#[test]
fn test_hypot_simd_overflow_protection() {
    // Large values that would overflow with naive x*x + y*y
    let large = 1e150_f64;
    let x = array![large, large];
    let y = array![large, large];

    let result = hypot_simd(&x.view(), &y.view());

    // Result should be finite, approximately large * sqrt(2)
    assert!(result[0].is_finite(), "hypot should handle large values");
    let expected = large * std::f64::consts::SQRT_2;
    let relative_error = ((result[0] - expected) / expected).abs();
    assert!(
        relative_error < 1e-10,
        "hypot({}, {}) = {}, expected {}",
        large,
        large,
        result[0],
        expected
    );
}

/// Test hypot underflow protection
#[test]
fn test_hypot_simd_underflow_protection() {
    // Small values that would underflow with naive x*x + y*y
    let small = 1e-200_f64;
    let x = array![small, small];
    let y = array![small, small];

    let result = hypot_simd(&x.view(), &y.view());

    // Result should be finite and non-zero
    assert!(result[0].is_finite(), "hypot should handle small values");
    assert!(result[0] > 0.0, "hypot should not underflow to zero");
    let expected = small * std::f64::consts::SQRT_2;
    let relative_error = ((result[0] - expected) / expected).abs();
    assert!(
        relative_error < 1e-10,
        "hypot({}, {}) = {}, expected {}",
        small,
        small,
        result[0],
        expected
    );
}

/// Test hypot property: hypot(x, 0) = |x|
#[test]
fn test_hypot_simd_identity_property() {
    let x = array![5.0_f64, -5.0, 0.0, std::f64::consts::PI];
    let zeros = Array1::zeros(4);

    let result = hypot_simd(&x.view(), &zeros.view());

    for i in 0..x.len() {
        assert!(
            (result[i] - x[i].abs()).abs() < 1e-14,
            "hypot(x, 0) should equal |x|"
        );
    }
}

/// Test hypot distance calculation use case
#[test]
fn test_hypot_simd_distance_calculation() {
    // Calculate distances from origin to various 2D points
    let points_x = array![1.0_f64, 1.0, 2.0, 3.0];
    let points_y = array![0.0_f64, 1.0, 2.0, 4.0];

    let distances = hypot_simd(&points_x.view(), &points_y.view());

    assert!((distances[0] - 1.0).abs() < 1e-14); // Point (1,0)
    assert!((distances[1] - std::f64::consts::SQRT_2).abs() < 1e-14); // Point (1,1)
    assert!((distances[2] - (2.0 * std::f64::consts::SQRT_2)).abs() < 1e-14); // Point (2,2)
    assert!((distances[3] - 5.0).abs() < 1e-14); // Point (3,4) -> 3-4-5 triangle
}

// ============================================================================
// Phase 62: SIMD copysign tests
// ============================================================================

/// Test copysign basic f32 correctness
#[test]
fn test_copysign_simd_f32_basic() {
    let x = array![1.0_f32, -2.0, 3.0, -4.0];
    let y = array![-1.0_f32, 1.0, 1.0, -1.0];

    let result = copysign_simd(&x.view(), &y.view());

    assert!((result[0] - (-1.0)).abs() < 1e-6); // |1| with sign of -1 = -1
    assert!((result[1] - 2.0).abs() < 1e-6); // |-2| with sign of 1 = 2
    assert!((result[2] - 3.0).abs() < 1e-6); // |3| with sign of 1 = 3
    assert!((result[3] - (-4.0)).abs() < 1e-6); // |-4| with sign of -1 = -4
}

/// Test copysign basic f64 correctness
#[test]
fn test_copysign_simd_f64_basic() {
    let x = array![1.0_f64, -2.0, 3.0, -4.0];
    let y = array![-1.0_f64, 1.0, 1.0, -1.0];

    let result = copysign_simd(&x.view(), &y.view());

    assert!((result[0] - (-1.0)).abs() < 1e-14);
    assert!((result[1] - 2.0).abs() < 1e-14);
    assert!((result[2] - 3.0).abs() < 1e-14);
    assert!((result[3] - (-4.0)).abs() < 1e-14);
}

/// Test copysign empty array
#[test]
fn test_copysign_simd_empty() {
    let x: Array1<f64> = array![];
    let y: Array1<f64> = array![];
    let result = copysign_simd(&x.view(), &y.view());
    assert_eq!(result.len(), 0);
}

/// Test copysign large array (SIMD path)
#[test]
fn test_copysign_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| i as f64 + 1.0).collect());
    let y = Array1::from_vec(
        (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
    );

    let result = copysign_simd(&x.view(), &y.view());
    assert_eq!(result.len(), n);

    // Check: even indices positive, odd indices negative
    for i in [0, 101, 1000, 5001, 9998] {
        if i % 2 == 0 {
            assert!(result[i] > 0.0, "Even index {} should be positive", i);
        } else {
            assert!(result[i] < 0.0, "Odd index {} should be negative", i);
        }
        assert!(
            (result[i].abs() - (i as f64 + 1.0)).abs() < 1e-10,
            "Magnitude should be preserved"
        );
    }
}

/// Test copysign with zeros
#[test]
fn test_copysign_simd_zeros() {
    let x = array![5.0_f64, 0.0, -3.0];
    let y = array![0.0_f64, -1.0, 0.0];

    let result = copysign_simd(&x.view(), &y.view());

    // Zero has positive sign in IEEE 754
    assert!((result[0] - 5.0).abs() < 1e-14); // 5 with sign of +0 = 5
                                              // copysign(0, -1) should be -0
    assert!(result[1] == 0.0 || result[1] == -0.0);
    assert!((result[2] - 3.0).abs() < 1e-14); // -3 with sign of +0 = 3
}

/// Test copysign property: |copysign(x, y)| = |x|
#[test]
fn test_copysign_simd_magnitude_preservation() {
    let x = array![1.5_f64, -2.5, 3.5, -4.5];
    let y = array![-1.0_f64, -1.0, 1.0, 1.0];

    let result = copysign_simd(&x.view(), &y.view());

    for i in 0..x.len() {
        assert!(
            (result[i].abs() - x[i].abs()).abs() < 1e-14,
            "copysign should preserve magnitude"
        );
    }
}

/// Test copysign property: sign(copysign(x, y)) = sign(y)
#[test]
fn test_copysign_simd_sign_transfer() {
    let x = array![1.0_f64, -2.0, 3.0, -4.0, 5.0];
    let y = array![-10.0_f64, 20.0, -30.0, 40.0, -50.0];

    let result = copysign_simd(&x.view(), &y.view());

    for i in 0..x.len() {
        let result_sign = if result[i] >= 0.0 { 1 } else { -1 };
        let y_sign = if y[i] >= 0.0 { 1 } else { -1 };
        assert_eq!(result_sign, y_sign, "Sign should match y at index {}", i);
    }
}

/// Test copysign use case: implementing absolute value
#[test]
fn test_copysign_simd_abs_implementation() {
    let x = array![-5.0_f64, 3.0, -2.0, 0.0, -0.0];
    let ones = array![1.0_f64, 1.0, 1.0, 1.0, 1.0];

    let result = copysign_simd(&x.view(), &ones.view());

    // copysign(x, 1) = |x|
    for i in 0..x.len() {
        assert!(
            (result[i] - x[i].abs()).abs() < 1e-14,
            "copysign(x, 1) should equal |x|"
        );
    }
}

/// Test copysign use case: implementing negation
#[test]
fn test_copysign_simd_negation_implementation() {
    let x = array![5.0_f64, -3.0, 2.0];
    let neg_ones = array![-1.0_f64, -1.0, -1.0];

    let result = copysign_simd(&x.view(), &neg_ones.view());

    // copysign(x, -1) = -|x|
    for i in 0..x.len() {
        assert!(
            (result[i] - (-x[i].abs())).abs() < 1e-14,
            "copysign(x, -1) should equal -|x|"
        );
    }
}
