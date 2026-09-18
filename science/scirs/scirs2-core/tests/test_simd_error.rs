//! Error function SIMD tests: erf, erfc, erfinv, erfcinv

use scirs2_core::ndarray::{array, Array1};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// =============================================================================
// =============================================================================
// Phase 45: Error Function Tests
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{erf_simd, erfc_simd};

/// Test erf(0) = 0
#[test]
fn test_erf_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = erf_simd(&x.view());
    assert!(
        result[0].abs() < 1e-15,
        "erf(0) should be 0, got {}",
        result[0]
    );
}

/// Test erf(1) ≈ 0.8427007929497148
#[test]
fn test_erf_simd_f64_at_one() {
    let x = array![1.0_f64];
    let result = erf_simd(&x.view());
    let expected = 0.8427007929497148_f64;
    // A&S approximation has max error ~1.5e-7
    assert!(
        (result[0] - expected).abs() < 2e-6,
        "erf(1) should be {}, got {}",
        expected,
        result[0]
    );
}

/// Test erf is an odd function: erf(-x) = -erf(x)
#[test]
fn test_erf_simd_f64_odd_function() {
    let x_pos = array![0.5_f64, 1.0, 1.5, 2.0, 3.0];
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0, -3.0];
    let result_pos = erf_simd(&x_pos.view());
    let result_neg = erf_simd(&x_neg.view());

    for i in 0..5 {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-14,
            "erf({}) should equal -erf({}): {} vs {}",
            x_pos[i],
            x_neg[i],
            result_pos[i],
            result_neg[i]
        );
    }
}

/// Test erf approaches 1 for large positive x
#[test]
fn test_erf_simd_f64_large_positive() {
    let x = array![3.0_f64, 4.0, 5.0, 6.0];
    let result = erf_simd(&x.view());

    // erf(3) ≈ 0.99998, erf(4+) is very close to 1
    for i in 0..4 {
        assert!(
            (result[i] - 1.0).abs() < 1e-4,
            "erf({}) should be close to 1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test erf approaches -1 for large negative x
#[test]
fn test_erf_simd_f64_large_negative() {
    let x = array![-3.0_f64, -4.0, -5.0, -6.0];
    let result = erf_simd(&x.view());

    // erf(-3) ≈ -0.99998, erf(-4-) is very close to -1
    for i in 0..4 {
        assert!(
            (result[i] + 1.0).abs() < 1e-4,
            "erf({}) should be close to -1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test known values of erf
#[test]
fn test_erf_simd_f64_known_values() {
    // Known values from reference tables
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let expected = [
        0.5204998778130465_f64, // erf(0.5)
        0.8427007929497148_f64, // erf(1.0)
        0.9661051464753108_f64, // erf(1.5)
        0.9953222650189527_f64, // erf(2.0)
    ];
    let result = erf_simd(&x.view());

    // A&S approximation has max error ~1.5e-7
    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 2e-6,
            "erf({}) should be {}, got {}",
            x[i],
            expected[i],
            result[i]
        );
    }
}

/// Test erf for very small x (linear regime)
#[test]
fn test_erf_simd_f64_small_values() {
    // For very small x, erf(x) ≈ 2x/√π
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let x = array![1e-10_f64, 1e-8, 1e-6];
    let result = erf_simd(&x.view());

    for i in 0..3 {
        let expected = x[i] * two_over_sqrt_pi;
        assert!(
            (result[i] - expected).abs() / expected.abs() < 1e-10,
            "erf({}) should be approximately {}, got {}",
            x[i],
            expected,
            result[i]
        );
    }
}

/// Test erf(infinity) = 1
#[test]
fn test_erf_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = erf_simd(&x.view());

    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "erf(∞) should be 1, got {}",
        result[0]
    );
    assert!(
        (result[1] + 1.0).abs() < 1e-10,
        "erf(-∞) should be -1, got {}",
        result[1]
    );
}

/// Test erf(NaN) = NaN
#[test]
fn test_erf_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = erf_simd(&x.view());
    assert!(result[0].is_nan(), "erf(NaN) should be NaN");
}

/// Test erf f32 version
#[test]
fn test_erf_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0, -1.0];
    let result = erf_simd(&x.view());

    assert!(
        result[0].abs() < 1e-7,
        "erf(0) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.8427008_f32).abs() < 1e-5,
        "erf(1) should be ~0.8427, got {}",
        result[1]
    );
    assert!(
        (result[3] + 0.8427008_f32).abs() < 1e-5,
        "erf(-1) should be ~-0.8427, got {}",
        result[3]
    );
}

/// Test empty input for erf
#[test]
fn test_erf_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = erf_simd(&x.view());
    assert!(result.is_empty(), "erf of empty array should be empty");
}

/// Test large array for erf (performance)
#[test]
fn test_erf_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-3.0, 3.0, n);
    let result = erf_simd(&x.view());

    assert_eq!(result.len(), n);
    // Check bounds: -1 <= erf(x) <= 1
    for val in result.iter() {
        assert!(
            *val >= -1.0 - 1e-10 && *val <= 1.0 + 1e-10,
            "erf should be in [-1, 1], got {}",
            val
        );
    }
}

// =============================================================================
// Complementary Error Function (erfc) Tests
// =============================================================================

/// Test erfc(0) = 1
#[test]
fn test_erfc_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = erfc_simd(&x.view());
    assert!(
        (result[0] - 1.0).abs() < 1e-15,
        "erfc(0) should be 1, got {}",
        result[0]
    );
}

/// Test erfc(1) ≈ 0.1572992070502852
#[test]
fn test_erfc_simd_f64_at_one() {
    let x = array![1.0_f64];
    let result = erfc_simd(&x.view());
    let expected = 0.1572992070502852_f64;
    // A&S approximation has max error ~1.5e-7
    assert!(
        (result[0] - expected).abs() < 2e-6,
        "erfc(1) should be {}, got {}",
        expected,
        result[0]
    );
}

/// Test erfc(-x) = 2 - erfc(x) (reflection property)
#[test]
fn test_erfc_simd_f64_reflection() {
    let x_pos = array![0.5_f64, 1.0, 1.5, 2.0];
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let result_pos = erfc_simd(&x_pos.view());
    let result_neg = erfc_simd(&x_neg.view());

    for i in 0..4 {
        assert!(
            (result_neg[i] - (2.0 - result_pos[i])).abs() < 1e-14,
            "erfc({}) should equal 2 - erfc({}): {} vs {}",
            x_neg[i],
            x_pos[i],
            result_neg[i],
            2.0 - result_pos[i]
        );
    }
}

/// Test erfc approaches 0 for large positive x
#[test]
fn test_erfc_simd_f64_large_positive() {
    let x = array![3.0_f64, 4.0, 5.0, 6.0];
    let result = erfc_simd(&x.view());

    // erfc(3) ≈ 2.2e-5, erfc(4+) is very close to 0
    for i in 0..4 {
        assert!(
            result[i] < 1e-4,
            "erfc({}) should be close to 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test erfc approaches 2 for large negative x
#[test]
fn test_erfc_simd_f64_large_negative() {
    let x = array![-3.0_f64, -4.0, -5.0, -6.0];
    let result = erfc_simd(&x.view());

    // erfc(-3) ≈ 1.99998, erfc(-4-) is very close to 2
    for i in 0..4 {
        assert!(
            (result[i] - 2.0).abs() < 1e-4,
            "erfc({}) should be close to 2, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test consistency: erfc(x) = 1 - erf(x)
#[test]
fn test_erfc_simd_f64_consistency_with_erf() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0];
    let erf_result = erf_simd(&x.view());
    let erfc_result = erfc_simd(&x.view());

    for i in 0..x.len() {
        let expected = 1.0 - erf_result[i];
        assert!(
            (erfc_result[i] - expected).abs() < 1e-14,
            "erfc({}) should equal 1 - erf({}): got {} vs {}",
            x[i],
            x[i],
            erfc_result[i],
            expected
        );
    }
}

/// Test erfc(infinity) = 0, erfc(-infinity) = 2
#[test]
fn test_erfc_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = erfc_simd(&x.view());

    assert!(
        result[0].abs() < 1e-10,
        "erfc(∞) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 2.0).abs() < 1e-10,
        "erfc(-∞) should be 2, got {}",
        result[1]
    );
}

/// Test erfc(NaN) = NaN
#[test]
fn test_erfc_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = erfc_simd(&x.view());
    assert!(result[0].is_nan(), "erfc(NaN) should be NaN");
}

/// Test erfc f32 version
#[test]
fn test_erfc_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0];
    let result = erfc_simd(&x.view());

    assert!(
        (result[0] - 1.0).abs() < 1e-7,
        "erfc(0) should be 1, got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.157299_f32).abs() < 1e-5,
        "erfc(1) should be ~0.157, got {}",
        result[1]
    );
}

/// Test empty input for erfc
#[test]
fn test_erfc_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = erfc_simd(&x.view());
    assert!(result.is_empty(), "erfc of empty array should be empty");
}

/// Test normal distribution CDF using erf
/// Φ(x) = 0.5 * (1 + erf(x/√2))
#[test]
fn test_erf_simd_normal_cdf() {
    let sqrt_2 = std::f64::consts::SQRT_2;
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let x_scaled: Array1<f64> = x.mapv(|v| v / sqrt_2);
    let erf_result = erf_simd(&x_scaled.view());
    let cdf: Array1<f64> = erf_result.mapv(|v| 0.5 * (1.0 + v));

    // Known CDF values for standard normal
    let expected_cdf = [
        0.5_f64,              // Φ(0) = 0.5
        0.8413447460685429,   // Φ(1) ≈ 0.8413
        0.15865525393145707,  // Φ(-1) ≈ 0.1587
        0.9772498680518208,   // Φ(2) ≈ 0.9772
        0.022750131948179195, // Φ(-2) ≈ 0.0228
    ];

    for i in 0..5 {
        assert!(
            (cdf[i] - expected_cdf[i]).abs() < 1e-6,
            "Φ({}) should be {}, got {}",
            x[i],
            expected_cdf[i],
            cdf[i]
        );
    }
}

// =============================================================================
// Phase 46: Inverse Error Function Tests
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{erfcinv_simd, erfinv_simd};

/// Test erfinv(0) = 0
#[test]
fn test_erfinv_simd_f64_at_zero() {
    let y = array![0.0_f64];
    let result = erfinv_simd(&y.view());
    assert!(
        result[0].abs() < 1e-15,
        "erfinv(0) should be 0, got {}",
        result[0]
    );
}

/// Test erfinv is the inverse of erf: erf(erfinv(y)) = y
#[test]
fn test_erfinv_simd_f64_inverse_property() {
    let y = array![0.0_f64, 0.1, 0.3, 0.5, 0.7, 0.9, -0.3, -0.7];
    let x = erfinv_simd(&y.view());
    let y_back = erf_simd(&x.view());

    for i in 0..y.len() {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-10,
            "erf(erfinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test erfinv is odd: erfinv(-y) = -erfinv(y)
#[test]
fn test_erfinv_simd_f64_odd_function() {
    let y_pos = array![0.1_f64, 0.3, 0.5, 0.7, 0.9];
    let y_neg = array![-0.1_f64, -0.3, -0.5, -0.7, -0.9];
    let result_pos = erfinv_simd(&y_pos.view());
    let result_neg = erfinv_simd(&y_neg.view());

    for i in 0..5 {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-12,
            "erfinv({}) should equal -erfinv({}): {} vs {}",
            y_pos[i],
            y_neg[i],
            result_pos[i],
            result_neg[i]
        );
    }
}

/// Test erfinv at boundaries: erfinv(1) = ∞, erfinv(-1) = -∞
#[test]
fn test_erfinv_simd_f64_boundaries() {
    let y = array![1.0_f64, -1.0];
    let result = erfinv_simd(&y.view());

    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "erfinv(1) should be ∞, got {}",
        result[0]
    );
    assert!(
        result[1].is_infinite() && result[1] < 0.0,
        "erfinv(-1) should be -∞, got {}",
        result[1]
    );
}

/// Test erfinv outside domain returns NaN
#[test]
fn test_erfinv_simd_f64_outside_domain() {
    let y = array![1.5_f64, -1.5, 2.0, -3.0];
    let result = erfinv_simd(&y.view());

    for i in 0..4 {
        assert!(
            result[i].is_nan(),
            "erfinv({}) should be NaN, got {}",
            y[i],
            result[i]
        );
    }
}

/// Test erfinv(NaN) = NaN
#[test]
fn test_erfinv_simd_f64_nan() {
    let y = array![f64::NAN];
    let result = erfinv_simd(&y.view());
    assert!(result[0].is_nan(), "erfinv(NaN) should be NaN");
}

/// Test erfinv f32 version
#[test]
fn test_erfinv_simd_f32_basic() {
    let y = array![0.0_f32, 0.5, -0.5];
    let result = erfinv_simd(&y.view());

    assert!(
        result[0].abs() < 1e-7,
        "erfinv(0) should be 0, got {}",
        result[0]
    );
    // Verify inverse property for f32
    let y_back = erf_simd(&result.view());
    for i in 0..3 {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-5,
            "erf(erfinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test empty input for erfinv
#[test]
fn test_erfinv_simd_empty() {
    let y = array![] as Array1<f64>;
    let result = erfinv_simd(&y.view());
    assert!(result.is_empty(), "erfinv of empty array should be empty");
}

/// Test large array for erfinv (performance)
#[test]
fn test_erfinv_simd_large_array() {
    let n = 10000;
    let y: Array1<f64> = Array1::linspace(-0.99, 0.99, n);
    let result = erfinv_simd(&y.view());

    assert_eq!(result.len(), n);
    // Verify inverse property
    let y_back = erf_simd(&result.view());
    for i in 0..n {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-8,
            "erf(erfinv({})) roundtrip error too large: got {}",
            y[i],
            y_back[i]
        );
    }
}

// =============================================================================
// Inverse Complementary Error Function (erfcinv) Tests
// =============================================================================

/// Test erfcinv(1) = 0
#[test]
fn test_erfcinv_simd_f64_at_one() {
    let y = array![1.0_f64];
    let result = erfcinv_simd(&y.view());
    assert!(
        result[0].abs() < 1e-15,
        "erfcinv(1) should be 0, got {}",
        result[0]
    );
}

/// Test erfcinv is the inverse of erfc: erfc(erfcinv(y)) = y
#[test]
fn test_erfcinv_simd_f64_inverse_property() {
    let y = array![0.1_f64, 0.5, 1.0, 1.5, 1.9];
    let x = erfcinv_simd(&y.view());
    let y_back = erfc_simd(&x.view());

    for i in 0..y.len() {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-10,
            "erfc(erfcinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test erfcinv at boundaries: erfcinv(0) = ∞, erfcinv(2) = -∞
#[test]
fn test_erfcinv_simd_f64_boundaries() {
    let y = array![0.0_f64, 2.0];
    let result = erfcinv_simd(&y.view());

    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "erfcinv(0) should be ∞, got {}",
        result[0]
    );
    assert!(
        result[1].is_infinite() && result[1] < 0.0,
        "erfcinv(2) should be -∞, got {}",
        result[1]
    );
}

/// Test erfcinv outside domain returns NaN
#[test]
fn test_erfcinv_simd_f64_outside_domain() {
    let y = array![-0.5_f64, 2.5, 3.0, -1.0];
    let result = erfcinv_simd(&y.view());

    for i in 0..4 {
        assert!(
            result[i].is_nan(),
            "erfcinv({}) should be NaN, got {}",
            y[i],
            result[i]
        );
    }
}

/// Test erfcinv(NaN) = NaN
#[test]
fn test_erfcinv_simd_f64_nan() {
    let y = array![f64::NAN];
    let result = erfcinv_simd(&y.view());
    assert!(result[0].is_nan(), "erfcinv(NaN) should be NaN");
}

/// Test erfcinv f32 version
#[test]
fn test_erfcinv_simd_f32_basic() {
    let y = array![1.0_f32, 0.5, 1.5];
    let result = erfcinv_simd(&y.view());

    assert!(
        result[0].abs() < 1e-6,
        "erfcinv(1) should be 0, got {}",
        result[0]
    );
    // Verify inverse property for f32
    let y_back = erfc_simd(&result.view());
    for i in 0..3 {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-5,
            "erfc(erfcinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test empty input for erfcinv
#[test]
fn test_erfcinv_simd_empty() {
    let y = array![] as Array1<f64>;
    let result = erfcinv_simd(&y.view());
    assert!(result.is_empty(), "erfcinv of empty array should be empty");
}

/// Test consistency: erfcinv(y) = erfinv(1 - y)
#[test]
fn test_erfcinv_simd_f64_consistency_with_erfinv() {
    let y = array![0.5_f64, 1.0, 1.5];
    let erfcinv_result = erfcinv_simd(&y.view());
    let y_minus_one: Array1<f64> = y.mapv(|v| 1.0 - v);
    let erfinv_result = erfinv_simd(&y_minus_one.view());

    for i in 0..y.len() {
        assert!(
            (erfcinv_result[i] - erfinv_result[i]).abs() < 1e-10,
            "erfcinv({}) should equal erfinv({}): got {} vs {}",
            y[i],
            1.0 - y[i],
            erfcinv_result[i],
            erfinv_result[i]
        );
    }
}

/// Test inverse normal CDF (probit function) using erfinv
/// Φ⁻¹(p) = √2 * erfinv(2p - 1)
#[test]
fn test_erfinv_simd_inverse_normal_cdf() {
    let sqrt_2 = std::f64::consts::SQRT_2;

    // Known quantiles of standard normal
    let p = array![0.5_f64, 0.84134, 0.97725]; // Φ(0), Φ(1), Φ(2) approximately
    let expected_x = [0.0_f64, 1.0, 2.0];

    let two_p_minus_one: Array1<f64> = p.mapv(|v| 2.0 * v - 1.0);
    let erfinv_result = erfinv_simd(&two_p_minus_one.view());
    let probit: Array1<f64> = erfinv_result.mapv(|v| sqrt_2 * v);

    for i in 0..3 {
        assert!(
            (probit[i] - expected_x[i]).abs() < 0.01,
            "Φ⁻¹({}) should be approximately {}, got {}",
            p[i],
            expected_x[i],
            probit[i]
        );
    }
}
