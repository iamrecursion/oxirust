//! Phase 61: SIMD interpolation tests (lerp and smoothstep)

use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Phase 61: SIMD lerp (linear interpolation) tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::lerp_simd;
use scirs2_core::ndarray_ext::elementwise::smoothstep_simd;

/// Test lerp basic f32 correctness
#[test]
fn test_lerp_simd_f32_basic() {
    let a = array![0.0_f32, 10.0, -5.0, 100.0];
    let b = array![10.0_f32, 20.0, 5.0, 200.0];

    // t = 0: should return a
    let result = lerp_simd(&a.view(), &b.view(), 0.0_f32);
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 10.0).abs() < 1e-6);
    assert!((result[2] - (-5.0)).abs() < 1e-6);
    assert!((result[3] - 100.0).abs() < 1e-6);

    // t = 1: should return b
    let result = lerp_simd(&a.view(), &b.view(), 1.0_f32);
    assert!((result[0] - 10.0).abs() < 1e-6);
    assert!((result[1] - 20.0).abs() < 1e-6);
    assert!((result[2] - 5.0).abs() < 1e-6);
    assert!((result[3] - 200.0).abs() < 1e-6);

    // t = 0.5: should return midpoint
    let result = lerp_simd(&a.view(), &b.view(), 0.5_f32);
    assert!((result[0] - 5.0).abs() < 1e-6);
    assert!((result[1] - 15.0).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
    assert!((result[3] - 150.0).abs() < 1e-6);
}

/// Test lerp basic f64 correctness
#[test]
fn test_lerp_simd_f64_basic() {
    let a = array![0.0_f64, 100.0, -50.0];
    let b = array![100.0_f64, 200.0, 50.0];

    // t = 0.25
    let result = lerp_simd(&a.view(), &b.view(), 0.25_f64);
    assert!((result[0] - 25.0).abs() < 1e-14);
    assert!((result[1] - 125.0).abs() < 1e-14);
    assert!((result[2] - (-25.0)).abs() < 1e-14);

    // t = 0.75
    let result = lerp_simd(&a.view(), &b.view(), 0.75_f64);
    assert!((result[0] - 75.0).abs() < 1e-14);
    assert!((result[1] - 175.0).abs() < 1e-14);
    assert!((result[2] - 25.0).abs() < 1e-14);
}

/// Test lerp empty array
#[test]
fn test_lerp_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = lerp_simd(&a.view(), &b.view(), 0.5_f64);
    assert_eq!(result.len(), 0);
}

/// Test lerp large array (SIMD path)
#[test]
fn test_lerp_simd_large_array() {
    let n = 10000;
    let a = Array1::from_vec((0..n).map(|i| i as f64).collect());
    let b = Array1::from_vec((0..n).map(|i| (i * 2) as f64).collect());

    let result = lerp_simd(&a.view(), &b.view(), 0.5_f64);
    assert_eq!(result.len(), n);

    // Check a few values: lerp(i, 2*i, 0.5) = i + 0.5*(2i - i) = i + 0.5*i = 1.5*i
    for i in [0, 100, 1000, 5000, 9999] {
        let expected = 1.5 * i as f64;
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "lerp[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test lerp extrapolation (t outside [0, 1])
#[test]
fn test_lerp_simd_extrapolation() {
    let a = array![0.0_f64, 0.0];
    let b = array![10.0_f64, 20.0];

    // t = -1: extrapolate backward
    let result = lerp_simd(&a.view(), &b.view(), -1.0_f64);
    assert!((result[0] - (-10.0)).abs() < 1e-14);
    assert!((result[1] - (-20.0)).abs() < 1e-14);

    // t = 2: extrapolate forward
    let result = lerp_simd(&a.view(), &b.view(), 2.0_f64);
    assert!((result[0] - 20.0).abs() < 1e-14);
    assert!((result[1] - 40.0).abs() < 1e-14);
}

/// Test lerp property: lerp(a, b, t) = a + t*(b-a)
#[test]
fn test_lerp_simd_formula_verification() {
    let a = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let b = array![11.0_f64, 12.0, 13.0, 14.0, 15.0];
    let t = 0.3_f64;

    let result = lerp_simd(&a.view(), &b.view(), t);

    for i in 0..a.len() {
        let expected = a[i] + t * (b[i] - a[i]);
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "Formula verification failed at index {}",
            i
        );
    }
}

/// Test lerp with same values (a == b)
#[test]
fn test_lerp_simd_same_values() {
    let a = array![5.0_f64, 5.0, 5.0];
    let b = array![5.0_f64, 5.0, 5.0];

    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let result = lerp_simd(&a.view(), &b.view(), t);
        for i in 0..a.len() {
            assert!(
                (result[i] - 5.0).abs() < 1e-14,
                "lerp of identical values should be the same value"
            );
        }
    }
}

/// Test lerp animation blending use case
#[test]
fn test_lerp_simd_animation_blending() {
    // Simulating position interpolation between keyframes
    let keyframe_0 = array![0.0_f64, 0.0, 0.0]; // Position at t=0
    let keyframe_1 = array![10.0_f64, 5.0, 2.0]; // Position at t=1

    // Interpolate at various animation times
    for time in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let position = lerp_simd(&keyframe_0.view(), &keyframe_1.view(), time);
        for i in 0..3 {
            let expected = keyframe_0[i] + time * (keyframe_1[i] - keyframe_0[i]);
            assert!(
                (position[i] - expected).abs() < 1e-14,
                "Animation interpolation at t={} failed",
                time
            );
        }
    }
}

// ============================================================================
// Phase 61: SIMD smoothstep tests
// ============================================================================

/// Test smoothstep basic f32 correctness
#[test]
fn test_smoothstep_simd_f32_basic() {
    let x = array![0.0_f32, 0.25, 0.5, 0.75, 1.0];

    let result = smoothstep_simd(0.0_f32, 1.0_f32, &x.view());

    // smoothstep(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // smoothstep(0.5) = 0.5 (symmetric about midpoint)
    assert!((result[2] - 0.5).abs() < 1e-6);
    // smoothstep(1) = 1
    assert!((result[4] - 1.0).abs() < 1e-6);

    // Values should be monotonically increasing
    for i in 0..result.len() - 1 {
        assert!(result[i] <= result[i + 1], "smoothstep should be monotonic");
    }
}

/// Test smoothstep basic f64 correctness
#[test]
fn test_smoothstep_simd_f64_basic() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];

    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Verify the Hermite polynomial: 3t² - 2t³
    for i in 0..x.len() {
        let t = x[i];
        let expected = t * t * (3.0 - 2.0 * t);
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "smoothstep[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test smoothstep empty array
#[test]
fn test_smoothstep_simd_empty() {
    let x: Array1<f64> = array![];
    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), 0);
}

/// Test smoothstep clamping at edges
#[test]
fn test_smoothstep_simd_clamping() {
    let x = array![-1.0_f64, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];

    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Values below edge0 should be 0
    assert!((result[0] - 0.0).abs() < 1e-14); // x = -1
    assert!((result[1] - 0.0).abs() < 1e-14); // x = -0.5
    assert!((result[2] - 0.0).abs() < 1e-14); // x = 0

    // Values above edge1 should be 1
    assert!((result[4] - 1.0).abs() < 1e-14); // x = 1
    assert!((result[5] - 1.0).abs() < 1e-14); // x = 1.5
    assert!((result[6] - 1.0).abs() < 1e-14); // x = 2
}

/// Test smoothstep with custom edges
#[test]
fn test_smoothstep_simd_custom_edges() {
    let x = array![0.0_f64, 2.5, 5.0, 7.5, 10.0];

    // Edges at [0, 10]
    let result = smoothstep_simd(0.0_f64, 10.0_f64, &x.view());

    // x=0: t=0, smoothstep=0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // x=5: t=0.5, smoothstep=0.5
    assert!((result[2] - 0.5).abs() < 1e-14);
    // x=10: t=1, smoothstep=1
    assert!((result[4] - 1.0).abs() < 1e-14);
}

/// Test smoothstep large array (SIMD path)
#[test]
fn test_smoothstep_simd_large_array() {
    let n = 10000;
    let x = Array1::linspace(0.0, 1.0, n);

    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), n);

    // Verify first, middle, and last values
    assert!((result[0] - 0.0).abs() < 1e-14);
    let mid = n / 2;
    let t_mid = x[mid];
    let expected_mid = t_mid * t_mid * (3.0 - 2.0 * t_mid);
    assert!((result[mid] - expected_mid).abs() < 1e-10);
    assert!((result[n - 1] - 1.0).abs() < 1e-14);
}

/// Test smoothstep property: derivative at edges is zero
#[test]
fn test_smoothstep_simd_derivative_at_edges() {
    // The derivative of 3t² - 2t³ is 6t - 6t² = 6t(1-t)
    // At t=0: derivative = 0, at t=1: derivative = 0

    // Approximate derivative at boundaries with very small epsilon
    let eps = 1e-8_f64;
    let x_near_0 = array![0.0, eps];
    let x_near_1 = array![1.0 - eps, 1.0];

    let result_0 = smoothstep_simd(0.0_f64, 1.0_f64, &x_near_0.view());
    let result_1 = smoothstep_simd(0.0_f64, 1.0_f64, &x_near_1.view());

    // Derivative near 0 should be very small
    let deriv_near_0 = (result_0[1] - result_0[0]) / eps;
    assert!(
        deriv_near_0.abs() < 1e-5,
        "Derivative at edge0 should be near 0, got {}",
        deriv_near_0
    );

    // Derivative near 1 should be very small
    let deriv_near_1 = (result_1[1] - result_1[0]) / eps;
    assert!(
        deriv_near_1.abs() < 1e-5,
        "Derivative at edge1 should be near 0, got {}",
        deriv_near_1
    );
}

/// Test smoothstep with reversed edges (edge0 > edge1)
#[test]
fn test_smoothstep_simd_reversed_edges() {
    let x = array![0.0_f64, 0.5, 1.0];

    // Reversed: edge0=1, edge1=0
    let result = smoothstep_simd(1.0_f64, 0.0_f64, &x.view());

    // With reversed edges, x=0 maps to t=(0-1)/(0-1)=1, x=1 maps to t=0
    // smoothstep(1.0, 0.0, 0) should be 1 (since x < edge1)
    // smoothstep(1.0, 0.0, 1) should be 0 (since x > edge0)
    assert!((result[0] - 1.0).abs() < 1e-14); // x=0 gives smoothstep=1
    assert!((result[2] - 0.0).abs() < 1e-14); // x=1 gives smoothstep=0
}

/// Test smoothstep shader use case: soft shadow transition
#[test]
fn test_smoothstep_simd_shadow_transition() {
    // Simulating soft shadow from light source
    // Shadow starts at distance 2.0, fully dark at 5.0
    let distances = array![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let shadow_start = 2.0_f64;
    let shadow_end = 5.0_f64;

    let shadow_factor = smoothstep_simd(shadow_start, shadow_end, &distances.view());

    // Before shadow_start: no shadow (0)
    assert!((shadow_factor[0] - 0.0).abs() < 1e-14);
    assert!((shadow_factor[1] - 0.0).abs() < 1e-14);
    assert!((shadow_factor[2] - 0.0).abs() < 1e-14);

    // After shadow_end: full shadow (1)
    assert!((shadow_factor[5] - 1.0).abs() < 1e-14);
    assert!((shadow_factor[6] - 1.0).abs() < 1e-14);
    assert!((shadow_factor[7] - 1.0).abs() < 1e-14);

    // Middle region: smooth transition
    assert!(shadow_factor[3] > 0.0 && shadow_factor[3] < 1.0);
    assert!(shadow_factor[4] > 0.0 && shadow_factor[4] < 1.0);
}
