//! Hyperbolic function SIMD tests: sinh, cosh, tanh, asinh, acosh, atanh

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{cosh_simd, sinh_simd, tanh_simd};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// =============================================================================
// Hyperbolic Function Tests (Phase 17)
// =============================================================================

// Basic correctness tests
#[test]
fn test_sinh_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0];
    let result = sinh_simd(&x.view());

    // sinh(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-10);
    // sinh(1) ≈ 1.175201194
    assert!((result[1] - 1.1752011936438014).abs() < 1e-10);
    // sinh(-1) ≈ -1.175201194
    assert!((result[2] + 1.1752011936438014).abs() < 1e-10);
    // sinh(2) ≈ 3.626860408
    assert!((result[3] - 3.626860407847019).abs() < 1e-10);
}

#[test]
fn test_sinh_simd_f32_basic() {
    let x = array![0.0f32, 1.0, -1.0, 2.0];
    let result = sinh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.1752012).abs() < 1e-6);
    assert!((result[2] + 1.1752012).abs() < 1e-6);
    assert!((result[3] - 3.6268604).abs() < 1e-6);
}

#[test]
fn test_cosh_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0];
    let result = cosh_simd(&x.view());

    // cosh(0) = 1
    assert!((result[0] - 1.0).abs() < 1e-10);
    // cosh(1) ≈ 1.543080635
    assert!((result[1] - 1.5430806348152437).abs() < 1e-10);
    // cosh(-1) ≈ 1.543080635 (symmetric)
    assert!((result[2] - 1.5430806348152437).abs() < 1e-10);
    // cosh(2) ≈ 3.762195691
    assert!((result[3] - 3.7621956910836314).abs() < 1e-10);
}

#[test]
fn test_cosh_simd_f32_basic() {
    let x = array![0.0f32, 1.0, -1.0, 2.0];
    let result = cosh_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 1.5430807).abs() < 1e-6);
    assert!((result[2] - 1.5430807).abs() < 1e-6);
    assert!((result[3] - 3.7621956).abs() < 1e-6);
}

#[test]
fn test_tanh_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, 10.0];
    let result = tanh_simd(&x.view());

    // tanh(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-10);
    // tanh(1) ≈ 0.761594156
    assert!((result[1] - 0.7615941559557649).abs() < 1e-10);
    // tanh(-1) ≈ -0.761594156
    assert!((result[2] + 0.7615941559557649).abs() < 1e-10);
    // tanh(2) ≈ 0.964027580
    assert!((result[3] - 0.9640275800758169).abs() < 1e-10);
    // tanh(10) ≈ 1.0 (asymptotic)
    assert!((result[4] - 1.0).abs() < 1e-8);
}

#[test]
fn test_tanh_simd_f32_basic() {
    let x = array![0.0f32, 1.0, -1.0, 2.0, 10.0];
    let result = tanh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 0.7615942).abs() < 1e-6);
    assert!((result[2] + 0.7615942).abs() < 1e-6);
    assert!((result[3] - 0.9640276).abs() < 1e-6);
    assert!((result[4] - 1.0).abs() < 1e-6);
}

// Empty array tests
#[test]
fn test_sinh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = sinh_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_cosh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cosh_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_tanh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = tanh_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Zero tests
#[test]
fn test_sinh_zero() {
    let x = array![0.0];
    let result = sinh_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_cosh_zero() {
    let x = array![0.0];
    let result = cosh_simd(&x.view());
    assert_eq!(result[0], 1.0);
}

#[test]
fn test_tanh_zero() {
    let x = array![0.0];
    let result = tanh_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

// NaN handling
#[test]
fn test_sinh_nan() {
    let x = array![f64::NAN];
    let result = sinh_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_cosh_nan() {
    let x = array![f64::NAN];
    let result = cosh_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_tanh_nan() {
    let x = array![f64::NAN];
    let result = tanh_simd(&x.view());
    assert!(result[0].is_nan());
}

// Hyperbolic identity: cosh²(x) - sinh²(x) = 1
#[test]
fn test_hyperbolic_identity_cosh_squared_minus_sinh_squared() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0];
    let sinh_result = sinh_simd(&x.view());
    let cosh_result = cosh_simd(&x.view());

    for i in 0..x.len() {
        let identity = cosh_result[i] * cosh_result[i] - sinh_result[i] * sinh_result[i];
        assert!(
            (identity - 1.0).abs() < 1e-10,
            "cosh²(x) - sinh²(x) != 1 at index {}",
            i
        );
    }
}

// Hyperbolic identity: tanh(x) = sinh(x) / cosh(x)
#[test]
fn test_hyperbolic_identity_tanh_equals_sinh_over_cosh() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0];
    let sinh_result = sinh_simd(&x.view());
    let cosh_result = cosh_simd(&x.view());
    let tanh_result = tanh_simd(&x.view());

    for i in 0..x.len() {
        let tanh_from_ratio = sinh_result[i] / cosh_result[i];
        assert!(
            (tanh_result[i] - tanh_from_ratio).abs() < 1e-10,
            "tanh(x) != sinh(x)/cosh(x) at index {}",
            i
        );
    }
}

// Anti-symmetry: sinh(-x) = -sinh(x)
#[test]
fn test_sinh_anti_symmetry() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let sinh_pos = sinh_simd(&x.view());
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let sinh_neg = sinh_simd(&x_neg.view());

    for i in 0..x.len() {
        assert!(
            (sinh_neg[i] + sinh_pos[i]).abs() < 1e-10,
            "sinh(-x) != -sinh(x) at index {}",
            i
        );
    }
}

// Anti-symmetry: tanh(-x) = -tanh(x)
#[test]
fn test_tanh_anti_symmetry() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let tanh_pos = tanh_simd(&x.view());
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let tanh_neg = tanh_simd(&x_neg.view());

    for i in 0..x.len() {
        assert!(
            (tanh_neg[i] + tanh_pos[i]).abs() < 1e-10,
            "tanh(-x) != -tanh(x) at index {}",
            i
        );
    }
}

// Symmetry: cosh(-x) = cosh(x)
#[test]
fn test_cosh_symmetry() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let cosh_pos = cosh_simd(&x.view());
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let cosh_neg = cosh_simd(&x_neg.view());

    for i in 0..x.len() {
        assert!(
            (cosh_neg[i] - cosh_pos[i]).abs() < 1e-10,
            "cosh(-x) != cosh(x) at index {}",
            i
        );
    }
}

// Asymptotic behavior: tanh(x) → ±1 as x → ±∞
#[test]
fn test_tanh_asymptotic_behavior() {
    let x_large_pos = array![10.0_f64, 20.0, 50.0];
    let result_pos = tanh_simd(&x_large_pos.view());

    for i in 0..x_large_pos.len() {
        assert!(
            (result_pos[i] - 1.0).abs() < 1e-8,
            "tanh(large_positive) should approach 1"
        );
    }

    let x_large_neg = array![-10.0_f64, -20.0, -50.0];
    let result_neg = tanh_simd(&x_large_neg.view());

    for i in 0..x_large_neg.len() {
        assert!(
            (result_neg[i] + 1.0).abs() < 1e-8,
            "tanh(large_negative) should approach -1"
        );
    }
}

// Range constraint: cosh(x) >= 1 always
#[test]
fn test_cosh_range_constraint() {
    let x = array![-10.0_f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
    let result = cosh_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] >= 1.0,
            "cosh(x) must be >= 1, but got {} at index {}",
            result[i],
            i
        );
    }
}

// Range constraint: |tanh(x)| < 1 always
#[test]
fn test_tanh_range_constraint() {
    let x = array![-10.0_f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
    let result = tanh_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].abs() <= 1.0,
            "|tanh(x)| must be <= 1, but got {} at index {}",
            result[i],
            i
        );
    }
}

// Derivative identity: d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
#[test]
fn test_tanh_derivative_identity() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5];
    let tanh_result = tanh_simd(&x.view());

    for i in 0..x.len() {
        // d/dx tanh(x) = 1 - tanh²(x)
        let derivative = 1.0 - tanh_result[i] * tanh_result[i];

        // Also equals sech²(x) = 1/cosh²(x)
        let cosh_val = x[i].cosh();
        let sech_squared = 1.0 / (cosh_val * cosh_val);

        assert!(
            (derivative - sech_squared).abs() < 1e-10,
            "1 - tanh²(x) != sech²(x) at index {}",
            i
        );
    }
}

// Relation with exponential: sinh(x) = (e^x - e^(-x))/2
#[test]
fn test_sinh_exponential_definition() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0];
    let sinh_result = sinh_simd(&x.view());

    for i in 0..x.len() {
        let sinh_from_exp = (x[i].exp() - (-x[i]).exp()) / 2.0;
        assert!(
            (sinh_result[i] - sinh_from_exp).abs() < 1e-10,
            "sinh(x) != (e^x - e^(-x))/2 at index {}",
            i
        );
    }
}

// Relation with exponential: cosh(x) = (e^x + e^(-x))/2
#[test]
fn test_cosh_exponential_definition() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0];
    let cosh_result = cosh_simd(&x.view());

    for i in 0..x.len() {
        let cosh_from_exp = (x[i].exp() + (-x[i]).exp()) / 2.0;
        assert!(
            (cosh_result[i] - cosh_from_exp).abs() < 1e-10,
            "cosh(x) != (e^x + e^(-x))/2 at index {}",
            i
        );
    }
}

// ============================================================================
// =============================================================================
// Inverse Hyperbolic Function Tests (Phase 59)
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{acosh_simd, asinh_simd, atanh_simd};

// -----------------------------------------------------------------------------
// asinh (Inverse Hyperbolic Sine) Tests
// -----------------------------------------------------------------------------

/// Test asinh basic f64
#[test]
fn test_asinh_simd_basic_f64() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = asinh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10, "asinh(0) should be 0");
    assert!(
        (result[1] - 0.881373587).abs() < 1e-6,
        "asinh(1) should be ~0.881"
    );
    assert!(
        (result[1] + result[2]).abs() < 1e-10,
        "asinh should be odd function"
    );
}

/// Test asinh basic f32
#[test]
fn test_asinh_simd_basic_f32() {
    let x = array![0.0_f32, 1.0, -1.0, 5.0];
    let result = asinh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-5, "asinh(0) should be 0");
    assert!(
        (result[1] - 0.881_373_6).abs() < 1e-4,
        "asinh(1) should be ~0.881"
    );
}

/// Test asinh empty array
#[test]
fn test_asinh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = asinh_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test asinh large array (SIMD path)
#[test]
fn test_asinh_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-100.0, 100.0, n);
    let result = asinh_simd(&x.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "asinh should be finite at index {}", i);
    }
}

/// Test asinh odd function property
#[test]
fn test_asinh_simd_odd_function() {
    let x = array![0.5_f64, 1.5, 2.5, 10.0, 100.0];
    let neg_x = x.mapv(|v| -v);

    let result_pos = asinh_simd(&x.view());
    let result_neg = asinh_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-10,
            "asinh(-x) should equal -asinh(x) at index {}",
            i
        );
    }
}

/// Test asinh inverse of sinh
#[test]
fn test_asinh_simd_inverse_of_sinh() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, 5.0];
    let sinh_x = x.mapv(|v| v.sinh());
    let asinh_sinh_x = asinh_simd(&sinh_x.view());

    for i in 0..x.len() {
        assert!(
            (asinh_sinh_x[i] - x[i]).abs() < 1e-10,
            "asinh(sinh(x)) should equal x, got {} vs {} at index {}",
            asinh_sinh_x[i],
            x[i],
            i
        );
    }
}

/// Test asinh NaN handling
#[test]
fn test_asinh_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN];
    let result = asinh_simd(&x.view());

    assert!(result[0].is_nan(), "asinh(NaN) should be NaN");
    assert!(result[2].is_nan(), "asinh(NaN) should be NaN");
    assert!(result[1].is_finite(), "asinh(1) should be finite");
}

// -----------------------------------------------------------------------------
// acosh (Inverse Hyperbolic Cosine) Tests
// -----------------------------------------------------------------------------

/// Test acosh basic f64
#[test]
fn test_acosh_simd_basic_f64() {
    let x = array![1.0_f64, 2.0, 3.0, 10.0];
    let result = acosh_simd(&x.view());

    assert!(
        (result[0] - 0.0).abs() < 1e-10,
        "acosh(1) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 1.31695790).abs() < 1e-6,
        "acosh(2) should be ~1.317"
    );
}

/// Test acosh basic f32
#[test]
fn test_acosh_simd_basic_f32() {
    let x = array![1.0_f32, 2.0, 5.0];
    let result = acosh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-5, "acosh(1) should be 0");
    assert!(
        (result[1] - 1.316_958).abs() < 1e-4,
        "acosh(2) should be ~1.317"
    );
}

/// Test acosh empty array
#[test]
fn test_acosh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = acosh_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test acosh large array (SIMD path)
#[test]
fn test_acosh_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(1.0, 1000.0, n);
    let result = acosh_simd(&x.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "acosh should be finite at index {}", i);
        assert!(
            v >= 0.0,
            "acosh should be non-negative, got {} at index {}",
            v,
            i
        );
    }
}

/// Test acosh out of domain (x < 1)
#[test]
fn test_acosh_simd_out_of_domain() {
    let x = array![0.0_f64, 0.5, 0.99, -1.0];
    let result = acosh_simd(&x.view());

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_nan(),
            "acosh({}) should be NaN for x < 1, got {}",
            x[i],
            v
        );
    }
}

/// Test acosh inverse of cosh
#[test]
fn test_acosh_simd_inverse_of_cosh() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, 5.0];
    let cosh_x = x.mapv(|v| v.cosh());
    let acosh_cosh_x = acosh_simd(&cosh_x.view());

    for i in 0..x.len() {
        // Note: acosh(cosh(x)) = |x| since cosh is even
        assert!(
            (acosh_cosh_x[i] - x[i].abs()).abs() < 1e-10,
            "acosh(cosh(x)) should equal |x|, got {} vs {} at index {}",
            acosh_cosh_x[i],
            x[i].abs(),
            i
        );
    }
}

/// Test acosh NaN handling
#[test]
fn test_acosh_simd_nan() {
    let x = array![f64::NAN, 2.0, f64::NAN];
    let result = acosh_simd(&x.view());

    assert!(result[0].is_nan(), "acosh(NaN) should be NaN");
    assert!(result[2].is_nan(), "acosh(NaN) should be NaN");
    assert!(result[1].is_finite(), "acosh(2) should be finite");
}

// -----------------------------------------------------------------------------
// atanh (Inverse Hyperbolic Tangent) Tests
// -----------------------------------------------------------------------------

/// Test atanh basic f64
#[test]
fn test_atanh_simd_basic_f64() {
    let x = array![0.0_f64, 0.5, -0.5, 0.9];
    let result = atanh_simd(&x.view());

    assert!(
        (result[0] - 0.0).abs() < 1e-10,
        "atanh(0) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.54930614).abs() < 1e-6,
        "atanh(0.5) should be ~0.549"
    );
    assert!(
        (result[1] + result[2]).abs() < 1e-10,
        "atanh should be odd function"
    );
}

/// Test atanh basic f32
#[test]
fn test_atanh_simd_basic_f32() {
    let x = array![0.0_f32, 0.5, -0.5];
    let result = atanh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-5, "atanh(0) should be 0");
    assert!(
        (result[1] - 0.54930614).abs() < 1e-4,
        "atanh(0.5) should be ~0.549"
    );
}

/// Test atanh empty array
#[test]
fn test_atanh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = atanh_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test atanh large array (SIMD path)
#[test]
fn test_atanh_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-0.99, 0.99, n);
    let result = atanh_simd(&x.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "atanh should be finite at index {}", i);
    }
}

/// Test atanh odd function property
#[test]
fn test_atanh_simd_odd_function() {
    let x = array![0.1_f64, 0.3, 0.5, 0.7, 0.9];
    let neg_x = x.mapv(|v| -v);

    let result_pos = atanh_simd(&x.view());
    let result_neg = atanh_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-10,
            "atanh(-x) should equal -atanh(x) at index {}",
            i
        );
    }
}

/// Test atanh boundaries (±1 → ±∞)
#[test]
fn test_atanh_simd_boundaries() {
    let x = array![1.0_f64, -1.0];
    let result = atanh_simd(&x.view());

    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "atanh(1) should be +∞"
    );
    assert!(
        result[1].is_infinite() && result[1] < 0.0,
        "atanh(-1) should be -∞"
    );
}

/// Test atanh out of domain (|x| > 1)
#[test]
fn test_atanh_simd_out_of_domain() {
    let x = array![1.5_f64, -1.5, 10.0];
    let result = atanh_simd(&x.view());

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_nan(),
            "atanh({}) should be NaN for |x| > 1, got {}",
            x[i],
            v
        );
    }
}

/// Test atanh inverse of tanh
#[test]
fn test_atanh_simd_inverse_of_tanh() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, -0.5];
    let tanh_x = x.mapv(|v| v.tanh());
    let atanh_tanh_x = atanh_simd(&tanh_x.view());

    for i in 0..x.len() {
        assert!(
            (atanh_tanh_x[i] - x[i]).abs() < 1e-10,
            "atanh(tanh(x)) should equal x, got {} vs {} at index {}",
            atanh_tanh_x[i],
            x[i],
            i
        );
    }
}

/// Test atanh NaN handling
#[test]
fn test_atanh_simd_nan() {
    let x = array![f64::NAN, 0.5, f64::NAN];
    let result = atanh_simd(&x.view());

    assert!(result[0].is_nan(), "atanh(NaN) should be NaN");
    assert!(result[2].is_nan(), "atanh(NaN) should be NaN");
    assert!(result[1].is_finite(), "atanh(0.5) should be finite");
}

/// Test atanh Fisher's z-transformation use case
#[test]
fn test_atanh_simd_fisher_z_use_case() {
    // Fisher's z-transformation: z = atanh(r) where r is correlation coefficient
    // This is used to make correlation coefficient sampling distributions more normal
    let correlations = array![0.0_f64, 0.3, 0.5, 0.7, 0.9, -0.5];
    let fisher_z = atanh_simd(&correlations.view());

    // z values should be finite for valid correlations
    for (i, &z) in fisher_z.iter().enumerate() {
        assert!(
            z.is_finite(),
            "Fisher z for correlation {} should be finite, got {}",
            correlations[i],
            z
        );
    }

    // z should preserve order (atanh is monotonically increasing)
    for i in 0..correlations.len() - 1 {
        if correlations[i] < correlations[i + 1] {
            assert!(
                fisher_z[i] < fisher_z[i + 1],
                "atanh should be monotonically increasing"
            );
        }
    }
}

// =============================================================================
