//! Phase 63: SIMD smootherstep tests (Ken Perlin's improved smoothstep)

use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Phase 63: SIMD smootherstep (Ken Perlin's improved smoothstep) tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::smootherstep_simd;
use scirs2_core::ndarray_ext::elementwise::smoothstep_simd;

/// Test smootherstep basic f32 correctness
#[test]
fn test_smootherstep_simd_f32_basic() {
    let x = array![0.0_f32, 0.25, 0.5, 0.75, 1.0];

    let result = smootherstep_simd(0.0_f32, 1.0_f32, &x.view());

    // smootherstep(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // smootherstep(0.5) = 0.5 (symmetric about midpoint)
    assert!((result[2] - 0.5).abs() < 1e-6);
    // smootherstep(1) = 1
    assert!((result[4] - 1.0).abs() < 1e-6);

    // Values should be monotonically increasing
    for i in 0..result.len() - 1 {
        assert!(
            result[i] <= result[i + 1],
            "smootherstep should be monotonic"
        );
    }
}

/// Test smootherstep basic f64 correctness
#[test]
fn test_smootherstep_simd_f64_basic() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Verify the Perlin polynomial: 6t⁵ - 15t⁴ + 10t³
    for i in 0..x.len() {
        let t = x[i];
        let t3 = t * t * t;
        let expected = t3 * (t * (t * 6.0 - 15.0) + 10.0);
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "smootherstep[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test smootherstep empty array
#[test]
fn test_smootherstep_simd_empty() {
    let x: Array1<f64> = array![];
    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), 0);
}

/// Test smootherstep clamping at edges
#[test]
fn test_smootherstep_simd_clamping() {
    let x = array![-1.0_f64, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Values below edge0 should be 0
    assert!((result[0] - 0.0).abs() < 1e-14); // x = -1
    assert!((result[1] - 0.0).abs() < 1e-14); // x = -0.5
    assert!((result[2] - 0.0).abs() < 1e-14); // x = 0

    // Values above edge1 should be 1
    assert!((result[4] - 1.0).abs() < 1e-14); // x = 1
    assert!((result[5] - 1.0).abs() < 1e-14); // x = 1.5
    assert!((result[6] - 1.0).abs() < 1e-14); // x = 2
}

/// Test smootherstep large array (SIMD path)
#[test]
fn test_smootherstep_simd_large_array() {
    let n = 10000;
    let x = Array1::linspace(0.0, 1.0, n);

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), n);

    // Verify first, middle, and last values
    assert!((result[0] - 0.0).abs() < 1e-14);
    let mid = n / 2;
    let t_mid = x[mid];
    let t3 = t_mid * t_mid * t_mid;
    let expected_mid = t3 * (t_mid * (t_mid * 6.0 - 15.0) + 10.0);
    assert!((result[mid] - expected_mid).abs() < 1e-10);
    assert!((result[n - 1] - 1.0).abs() < 1e-14);
}

/// Test smootherstep property: first derivative at edges is zero
#[test]
fn test_smootherstep_simd_first_derivative_at_edges() {
    // The first derivative of 6t⁵ - 15t⁴ + 10t³ is 30t⁴ - 60t³ + 30t² = 30t²(t-1)²
    // At t=0 and t=1, the derivative is 0

    let eps = 1e-8_f64;
    let x_near_0 = array![0.0, eps];
    let x_near_1 = array![1.0 - eps, 1.0];

    let result_0 = smootherstep_simd(0.0_f64, 1.0_f64, &x_near_0.view());
    let result_1 = smootherstep_simd(0.0_f64, 1.0_f64, &x_near_1.view());

    // Derivative near 0 should be very small
    let deriv_near_0 = (result_0[1] - result_0[0]) / eps;
    assert!(
        deriv_near_0.abs() < 1e-5,
        "First derivative at edge0 should be near 0, got {}",
        deriv_near_0
    );

    // Derivative near 1 should be very small
    let deriv_near_1 = (result_1[1] - result_1[0]) / eps;
    assert!(
        deriv_near_1.abs() < 1e-5,
        "First derivative at edge1 should be near 0, got {}",
        deriv_near_1
    );
}

/// Test smootherstep property: second derivative at edges is zero
#[test]
fn test_smootherstep_simd_second_derivative_at_edges() {
    // The second derivative is 120t³ - 180t² + 60t = 60t(2t-1)(t-1)
    // At t=0 and t=1, the second derivative is 0

    let eps = 1e-6_f64;
    let x = array![0.0, eps, 2.0 * eps];

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Approximate second derivative at t=0 using central difference
    let d1 = (result[1] - result[0]) / eps;
    let d2 = (result[2] - result[1]) / eps;
    let second_deriv = (d2 - d1) / eps;

    assert!(
        second_deriv.abs() < 1e-2,
        "Second derivative at edge0 should be near 0, got {}",
        second_deriv
    );
}

/// Test smootherstep comparison with smoothstep (should be smoother)
#[test]
fn test_smootherstep_vs_smoothstep() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];

    let result_smoother = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());
    let result_smooth = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Both should agree at 0, 0.5, and 1
    assert!((result_smoother[0] - result_smooth[0]).abs() < 1e-14);
    assert!((result_smoother[2] - result_smooth[2]).abs() < 1e-14);
    assert!((result_smoother[4] - result_smooth[4]).abs() < 1e-14);

    // At t=0.25 and t=0.75, smootherstep differs from smoothstep
    // smoothstep(0.25) = 3(0.0625) - 2(0.015625) = 0.1875 - 0.03125 = 0.15625
    // smootherstep(0.25) = 6(0.000976...) - 15(0.00390625) + 10(0.015625)
    //                    = 0.005859... - 0.058594... + 0.15625 ≈ 0.103516
    // They should be different
    assert!(
        (result_smoother[1] - result_smooth[1]).abs() > 0.01,
        "smootherstep and smoothstep should differ at t=0.25"
    );
}

/// Test smootherstep Perlin noise use case
#[test]
fn test_smootherstep_simd_perlin_noise_use_case() {
    // In Perlin noise, smootherstep is used for gradient interpolation
    // This tests the fade function behavior that Perlin noise expects

    let t_values = array![0.0_f64, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    let fade_result = smootherstep_simd(0.0_f64, 1.0_f64, &t_values.view());

    // All values should be in [0, 1]
    for (i, &v) in fade_result.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "Perlin fade at t={} should be in [0,1], got {}",
            t_values[i],
            v
        );
    }

    // Should be strictly monotonically increasing (except at endpoints)
    for i in 1..fade_result.len() {
        assert!(
            fade_result[i] >= fade_result[i - 1],
            "Perlin fade should be monotonic"
        );
    }

    // The curve should be more "S-shaped" than linear
    // At t=0.5, smootherstep should equal exactly 0.5 (symmetry)
    assert!((fade_result[5] - 0.5).abs() < 1e-14);
}
