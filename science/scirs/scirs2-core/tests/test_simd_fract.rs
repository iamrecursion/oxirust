//! Fractional part SIMD tests: fract

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{fract_simd, gamma_simd, powi_simd, recip_simd};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============================================================================
// Fract SIMD Tests
// ============================================================================

// Basic correctness f64
#[test]
fn test_fract_simd_f64_basic() {
    let x = array![1.5_f64, 2.7, -1.3, 0.0, 3.0];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.5).abs() < 1e-10); // 1.5.fract() = 0.5
    assert!((result[1] - 0.7).abs() < 1e-10); // 2.7.fract() = 0.7
    assert!((result[2] - (-0.3)).abs() < 1e-10); // -1.3.fract() = -0.3 (signed!)
    assert_eq!(result[3], 0.0); // 0.0.fract() = 0.0
    assert_eq!(result[4], 0.0); // 3.0.fract() = 0.0
}

// Basic correctness f32
#[test]
fn test_fract_simd_f32_basic() {
    let x = array![1.5_f32, 2.7, -1.3, 0.0, 3.0];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.5_f32).abs() < 1e-6);
    assert!((result[1] - 0.7_f32).abs() < 1e-6);
    assert!((result[2] - (-0.3_f32)).abs() < 1e-6);
    assert_eq!(result[3], 0.0);
    assert_eq!(result[4], 0.0);
}

// Empty array
#[test]
fn test_fract_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let result = fract_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// All positive integers
#[test]
fn test_fract_simd_positive_integers() {
    let x = array![1.0_f64, 2.0, 3.0, 10.0, 100.0];
    let result = fract_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// All negative integers
#[test]
fn test_fract_simd_negative_integers() {
    let x = array![-1.0_f64, -2.0, -3.0, -10.0, -100.0];
    let result = fract_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// All positive non-integers
#[test]
fn test_fract_simd_positive_fractions() {
    let x = array![0.1_f64, 0.5, 0.9, 1.1, 1.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.1).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - 0.9).abs() < 1e-10);
    assert!((result[3] - 0.1).abs() < 1e-10);
    assert!((result[4] - 0.5).abs() < 1e-10);
}

// All negative non-integers
#[test]
fn test_fract_simd_negative_fractions() {
    let x = array![-0.1_f64, -0.5, -0.9, -1.1, -1.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - (-0.1)).abs() < 1e-10); // -0.1.fract() = -0.1
    assert!((result[1] - (-0.5)).abs() < 1e-10); // -0.5.fract() = -0.5
    assert!((result[2] - (-0.9)).abs() < 1e-10); // -0.9.fract() = -0.9
    assert!((result[3] - (-0.1)).abs() < 1e-10); // -1.1.fract() = -0.1
    assert!((result[4] - (-0.5)).abs() < 1e-10); // -1.5.fract() = -0.5
}

// Property: fract(x) + trunc(x) = x
#[test]
fn test_fract_plus_trunc_equals_x() {
    let x = array![5.25_f64, -2.75, 0.5, 10.0, -7.3];
    let fract_parts = fract_simd(&x.view());
    let trunc_parts = x.mapv(|v| v.trunc());
    let reconstructed = &fract_parts + &trunc_parts;

    for i in 0..x.len() {
        assert!((reconstructed[i] - x[i]).abs() < 1e-10);
    }
}

// Property: fract(x + n) = fract(x) for integer n (same sign)
#[test]
fn test_fract_periodic_property() {
    let x = array![0.3_f64, 0.7, 0.5, 1.2, 2.9];
    let n = 5.0;

    let fract_x = fract_simd(&x.view());
    let x_plus_n = &x + n;
    let fract_x_plus_n = fract_simd(&x_plus_n.view());

    for i in 0..x.len() {
        assert!((fract_x[i] - fract_x_plus_n[i]).abs() < 1e-10);
    }
}

// Property: -1 < fract(x) < 1 for all finite x
#[test]
fn test_fract_range_property() {
    let x = array![-100.5_f64, -10.3, -0.1, 0.0, 0.1, 10.3, 100.5];
    let result = fract_simd(&x.view());

    for &val in result.iter() {
        assert!(val > -1.0 && val < 1.0);
    }
}

// Property: fract(x) = 0 if and only if x is an integer
#[test]
fn test_fract_zero_iff_integer() {
    let integers = array![-5.0_f64, -1.0, 0.0, 1.0, 5.0];
    let result = fract_simd(&integers.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// Property: fract(-x) = -fract(x) (odd function)
#[test]
fn test_fract_odd_function() {
    let x = array![0.3_f64, 0.7, 1.5, 2.9];
    let neg_x = -&x;

    let fract_x = fract_simd(&x.view());
    let fract_neg_x = fract_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!((fract_neg_x[i] + fract_x[i]).abs() < 1e-10);
    }
}

// NaN handling
#[test]
fn test_fract_nan_handling() {
    let x = array![f64::NAN, 1.5, -1.5];
    let result = fract_simd(&x.view());

    assert!(result[0].is_nan());
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (-0.5)).abs() < 1e-10);
}

// Infinity handling
#[test]
fn test_fract_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let result = fract_simd(&x.view());

    assert!(result[0].is_nan()); // fract(+inf) = NaN
    assert!(result[1].is_nan()); // fract(-inf) = NaN
    assert_eq!(result[2], 0.0);
}

// Large array SIMD path
#[test]
fn test_fract_large_array() {
    let size = 10_000;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let val = ((i as f64) - 5000.0) * 0.137; // Mix of positive and negative
        data.push(val);
    }
    let x = Array1::from_vec(data);
    let result = fract_simd(&x.view());

    for i in 0..size {
        let expected = x[i].fract();
        assert!((result[i] - expected).abs() < 1e-10);
        assert!(result[i] > -1.0 && result[i] < 1.0);
    }
}

// Very large values
#[test]
fn test_fract_very_large_values() {
    let x = array![1e10_f64 + 0.5, -1e10 - 0.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.5).abs() < 1e-6);
    assert!((result[1] - (-0.5)).abs() < 1e-6);
}

// Signed fractional parts
#[test]
fn test_fract_signed_values() {
    let x = array![-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - (-0.5_f32)).abs() < 1e-6);
    assert!((result[1] - (-0.5_f32)).abs() < 1e-6);
    assert!((result[2] - (-0.5_f32)).abs() < 1e-6);
    assert!((result[3] - 0.5_f32).abs() < 1e-6);
    assert!((result[4] - 0.5_f32).abs() < 1e-6);
    assert!((result[5] - 0.5_f32).abs() < 1e-6);
}

// ============================================================================
// recip_simd Tests (Phase 24)
// ============================================================================

// Basic correctness f32
#[test]
fn test_recip_simd_f32_basic() {
    let x = array![2.0_f32, 4.0, 0.5, 1.0, -2.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - 0.5_f32).abs() < 1e-6);
    assert!((result[1] - 0.25_f32).abs() < 1e-6);
    assert!((result[2] - 2.0_f32).abs() < 1e-6);
    assert!((result[3] - 1.0_f32).abs() < 1e-6);
    assert!((result[4] - (-0.5_f32)).abs() < 1e-6);
}

// Basic correctness f64
#[test]
fn test_recip_simd_f64_basic() {
    let x = array![2.0_f64, 4.0, 0.5, 1.0, -2.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - 0.5).abs() < 1e-10);
    assert!((result[1] - 0.25).abs() < 1e-10);
    assert!((result[2] - 2.0).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
    assert!((result[4] - (-0.5)).abs() < 1e-10);
}

// Empty array
#[test]
fn test_recip_simd_empty() {
    let x: Array1<f64> = array![];
    let result = recip_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// All positive values
#[test]
fn test_recip_simd_all_positive() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (1.0 / 3.0)).abs() < 1e-10);
    assert!((result[3] - 0.25).abs() < 1e-10);
    assert!((result[4] - 0.2).abs() < 1e-10);
}

// All negative values
#[test]
fn test_recip_simd_all_negative() {
    let x = array![-1.0_f64, -2.0, -4.0, -5.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - (-0.5)).abs() < 1e-10);
    assert!((result[2] - (-0.25)).abs() < 1e-10);
    assert!((result[3] - (-0.2)).abs() < 1e-10);
}

// Mixed positive and negative
#[test]
fn test_recip_simd_mixed() {
    let x = array![-4.0_f64, -2.0, -1.0, 1.0, 2.0, 4.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - (-0.25)).abs() < 1e-10);
    assert!((result[1] - (-0.5)).abs() < 1e-10);
    assert!((result[2] - (-1.0)).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
    assert!((result[4] - 0.5).abs() < 1e-10);
    assert!((result[5] - 0.25).abs() < 1e-10);
}

// Property: recip(recip(x)) = x (involutive property)
#[test]
fn test_recip_involutive() {
    let x = array![2.0_f64, 3.0, 5.0, 7.0, 11.0];
    let recip_x = recip_simd(&x.view());
    let recip_recip_x = recip_simd(&recip_x.view());

    for i in 0..x.len() {
        assert!((recip_recip_x[i] - x[i]).abs() < 1e-10);
    }
}

// Property: recip(x * y) = recip(x) * recip(y)
#[test]
fn test_recip_multiplicative() {
    let x = array![2.0_f64, 3.0, 5.0];
    let y = array![4.0_f64, 6.0, 10.0];
    let xy = &x * &y;

    let recip_xy = recip_simd(&xy.view());
    let recip_x = recip_simd(&x.view());
    let recip_y = recip_simd(&y.view());
    let product = &recip_x * &recip_y;

    for i in 0..x.len() {
        assert!((recip_xy[i] - product[i]).abs() < 1e-10);
    }
}

// Property: recip(x/y) = y/x
#[test]
fn test_recip_division_inversion() {
    let x = array![8.0_f64, 12.0, 20.0];
    let y = array![2.0_f64, 3.0, 5.0];
    let x_div_y = &x / &y;

    let recip_result = recip_simd(&x_div_y.view());
    let expected = &y / &x;

    for i in 0..x.len() {
        assert!((recip_result[i] - expected[i]).abs() < 1e-10);
    }
}

// Property: recip(1) = 1 (identity element)
#[test]
fn test_recip_identity() {
    let x = array![1.0_f64, 1.0, 1.0, 1.0];
    let result = recip_simd(&x.view());

    for &val in result.iter() {
        assert!((val - 1.0).abs() < 1e-10);
    }
}

// Property: recip(-x) = -recip(x) (odd function)
#[test]
fn test_recip_odd_function() {
    let x = array![2.0_f64, 3.0, 5.0, 7.0];
    let neg_x = -&x;

    let recip_x = recip_simd(&x.view());
    let recip_neg_x = recip_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!((recip_neg_x[i] + recip_x[i]).abs() < 1e-10);
    }
}

// Property: recip(x^n) = (recip(x))^n
#[test]
fn test_recip_power_property() {
    let x = array![2.0_f64, 3.0, 4.0];
    let n = 3.0;

    let x_pow_n = x.mapv(|v| v.powf(n));
    let recip_x_pow_n = recip_simd(&x_pow_n.view());

    let recip_x = recip_simd(&x.view());
    let recip_x_pow_n_expected = recip_x.mapv(|v| v.powf(n));

    for i in 0..x.len() {
        assert!((recip_x_pow_n[i] - recip_x_pow_n_expected[i]).abs() < 1e-10);
    }
}

// Zero handling (returns infinity)
#[test]
fn test_recip_zero() {
    let x = array![0.0_f64];
    let result = recip_simd(&x.view());
    assert!(result[0].is_infinite() && result[0].is_sign_positive());

    // Test negative zero
    let x = array![-0.0_f64];
    let result = recip_simd(&x.view());
    assert!(result[0].is_infinite() && result[0].is_sign_negative());
}

// NaN propagation
#[test]
fn test_recip_nan_propagation() {
    let x = array![f64::NAN, 2.0, 3.0];
    let result = recip_simd(&x.view());

    assert!(result[0].is_nan());
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (1.0 / 3.0)).abs() < 1e-10);
}

// Infinity handling
#[test]
fn test_recip_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 2.0];
    let result = recip_simd(&x.view());

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], -0.0);
    assert!((result[2] - 0.5).abs() < 1e-10);
}

// Large array (triggers SIMD path)
#[test]
fn test_recip_large_array() {
    let size = 10_000;
    let data: Array1<f64> = Array1::from_vec((1..=size).map(|i| i as f64).collect());

    let result = recip_simd(&data.view());

    assert_eq!(result.len(), size);
    // Check a few values
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[99] - 0.01).abs() < 1e-10);
    assert!((result[999] - 0.001).abs() < 1e-10);
}

// Very small values (near zero)
#[test]
fn test_recip_very_small_values() {
    let x = array![1e-10_f64, 1e-20, 1e-30];
    let result = recip_simd(&x.view());

    assert!((result[0] - 1e10).abs() < 1e-10);
    assert!((result[1] - 1e20).abs() < 1e10);
    assert!((result[2] - 1e30).abs() < 1e20);
}

// Very large values
#[test]
fn test_recip_very_large_values() {
    let x = array![1e10_f64, 1e20, 1e30];
    let result = recip_simd(&x.view());

    assert!((result[0] - 1e-10).abs() < 1e-20);
    assert!((result[1] - 1e-20).abs() < 1e-30);
    // Note: 1e30 reciprocal is too small for standard comparison
    assert!(result[2] < 1e-29 && result[2] > 0.0);
}

// Application: Inverse square law (physics)
#[test]
fn test_recip_inverse_square_law() {
    // F = k / r^2 for various distances
    let distances = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let r_squared = distances.mapv(|r| r * r);
    let forces = recip_simd(&r_squared.view()); // Proportional to force

    // Verify inverse square relationship
    assert!((forces[0] - 1.0).abs() < 1e-10); // 1/1^2 = 1
    assert!((forces[1] - 0.25).abs() < 1e-10); // 1/2^2 = 0.25
    assert!((forces[2] - (1.0 / 9.0)).abs() < 1e-10); // 1/3^2 = 1/9
    assert!((forces[3] - 0.0625).abs() < 1e-10); // 1/4^2 = 1/16
    assert!((forces[4] - 0.04).abs() < 1e-10); // 1/5^2 = 1/25
}

// ============================================================================
// powi_simd Tests (Phase 25)
// ============================================================================

// Basic correctness f32 - square
#[test]
fn test_powi_simd_f32_square() {
    let x = array![2.0_f32, 3.0, 4.0, 5.0];
    let result = powi_simd(&x.view(), 2);

    assert!((result[0] - 4.0_f32).abs() < 1e-6);
    assert!((result[1] - 9.0_f32).abs() < 1e-6);
    assert!((result[2] - 16.0_f32).abs() < 1e-6);
    assert!((result[3] - 25.0_f32).abs() < 1e-6);
}

// Basic correctness f64 - cube
#[test]
fn test_powi_simd_f64_cube() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0];
    let result = powi_simd(&x.view(), 3);

    assert!((result[0] - 8.0).abs() < 1e-10);
    assert!((result[1] - 27.0).abs() < 1e-10);
    assert!((result[2] - 64.0).abs() < 1e-10);
    assert!((result[3] - 125.0).abs() < 1e-10);
}

// Empty array
#[test]
fn test_powi_simd_empty() {
    let x: Array1<f64> = array![];
    let result = powi_simd(&x.view(), 2);
    assert_eq!(result.len(), 0);
}

// Exponent zero (should return all ones)
#[test]
fn test_powi_simd_exp_zero() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0, -2.0];
    let result = powi_simd(&x.view(), 0);

    for &val in result.iter() {
        assert!((val - 1.0).abs() < 1e-10);
    }
}

// Exponent one (should return input)
#[test]
fn test_powi_simd_exp_one() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0, -2.0];
    let result = powi_simd(&x.view(), 1);

    for i in 0..x.len() {
        assert!((result[i] - x[i]).abs() < 1e-10);
    }
}

// Negative exponent (reciprocal powers)
#[test]
fn test_powi_simd_negative_exp() {
    let x = array![2.0_f64, 4.0, 5.0];
    let result = powi_simd(&x.view(), -2);

    assert!((result[0] - 0.25).abs() < 1e-10); // 2^(-2) = 0.25
    assert!((result[1] - 0.0625).abs() < 1e-10); // 4^(-2) = 0.0625
    assert!((result[2] - 0.04).abs() < 1e-10); // 5^(-2) = 0.04
}

// Negative bases with even exponent
#[test]
fn test_powi_simd_negative_base_even() {
    let x = array![-2.0_f64, -3.0, -4.0];
    let result = powi_simd(&x.view(), 2);

    assert!((result[0] - 4.0).abs() < 1e-10); // (-2)^2 = 4
    assert!((result[1] - 9.0).abs() < 1e-10); // (-3)^2 = 9
    assert!((result[2] - 16.0).abs() < 1e-10); // (-4)^2 = 16
}

// Negative bases with odd exponent
#[test]
fn test_powi_simd_negative_base_odd() {
    let x = array![-2.0_f64, -3.0, -4.0];
    let result = powi_simd(&x.view(), 3);

    assert!((result[0] - (-8.0)).abs() < 1e-10); // (-2)^3 = -8
    assert!((result[1] - (-27.0)).abs() < 1e-10); // (-3)^3 = -27
    assert!((result[2] - (-64.0)).abs() < 1e-10); // (-4)^3 = -64
}

// Property: powi(x, n+m) = powi(x, n) * powi(x, m)
#[test]
fn test_powi_exponent_addition() {
    let x = array![2.0_f64, 3.0, 5.0];
    let n = 3;
    let m = 2;

    let powi_n_plus_m = powi_simd(&x.view(), n + m);
    let powi_n = powi_simd(&x.view(), n);
    let powi_m = powi_simd(&x.view(), m);
    let product = &powi_n * &powi_m;

    for i in 0..x.len() {
        assert!((powi_n_plus_m[i] - product[i]).abs() < 1e-10);
    }
}

// Property: powi(x*y, n) = powi(x, n) * powi(y, n)
#[test]
fn test_powi_distributive() {
    let x = array![2.0_f64, 3.0, 5.0];
    let y = array![4.0_f64, 2.0, 3.0];
    let n = 3;

    let xy = &x * &y;
    let powi_xy = powi_simd(&xy.view(), n);

    let powi_x = powi_simd(&x.view(), n);
    let powi_y = powi_simd(&y.view(), n);
    let product = &powi_x * &powi_y;

    for i in 0..x.len() {
        assert!((powi_xy[i] - product[i]).abs() < 1e-10);
    }
}

// Property: powi(x, n*m) = powi(powi(x, n), m)
#[test]
fn test_powi_exponent_multiplication() {
    let x = array![2.0_f64, 3.0, 4.0];
    let n = 2;
    let m = 3;

    let powi_nm = powi_simd(&x.view(), n * m);
    let powi_n = powi_simd(&x.view(), n);
    let powi_powi = powi_simd(&powi_n.view(), m);

    for i in 0..x.len() {
        assert!((powi_nm[i] - powi_powi[i]).abs() < 1e-10);
    }
}

// Property: powi(x, -n) = 1 / powi(x, n)
#[test]
fn test_powi_negative_exp_reciprocal() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0];
    let n = 3;

    let powi_neg_n = powi_simd(&x.view(), -n);
    let powi_n = powi_simd(&x.view(), n);
    let recip_powi_n = recip_simd(&powi_n.view());

    for i in 0..x.len() {
        assert!((powi_neg_n[i] - recip_powi_n[i]).abs() < 1e-10);
    }
}

// Zero base with positive exponent
#[test]
fn test_powi_zero_base_positive_exp() {
    let x = array![0.0_f64, 0.0, 0.0];
    let result = powi_simd(&x.view(), 3);

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// Zero base with zero exponent (0^0 = 1 convention)
#[test]
fn test_powi_zero_base_zero_exp() {
    let x = array![0.0_f64];
    let result = powi_simd(&x.view(), 0);

    assert_eq!(result[0], 1.0);
}

// NaN propagation
#[test]
fn test_powi_nan_propagation() {
    let x = array![f64::NAN, 2.0, 3.0];
    let result = powi_simd(&x.view(), 2);

    assert!(result[0].is_nan());
    assert!((result[1] - 4.0).abs() < 1e-10);
    assert!((result[2] - 9.0).abs() < 1e-10);
}

// Infinity handling
#[test]
fn test_powi_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 2.0];
    let result_pos = powi_simd(&x.view(), 2);
    let result_neg = powi_simd(&x.view(), -2);

    // Positive exponent
    assert!(result_pos[0].is_infinite() && result_pos[0].is_sign_positive());
    assert!(result_pos[1].is_infinite() && result_pos[1].is_sign_positive()); // (-∞)^2 = ∞

    // Negative exponent
    assert_eq!(result_neg[0], 0.0); // ∞^(-2) = 0
    assert_eq!(result_neg[1], 0.0); // (-∞)^(-2) = 0
}

// Large array (triggers SIMD path)
#[test]
fn test_powi_large_array() {
    let size = 10_000;
    let data: Array1<f64> = Array1::from_vec((1..=size).map(|i| i as f64).collect());

    let result = powi_simd(&data.view(), 2);

    assert_eq!(result.len(), size);
    // Check a few values
    assert!((result[0] - 1.0).abs() < 1e-10); // 1^2 = 1
    assert!((result[1] - 4.0).abs() < 1e-10); // 2^2 = 4
    assert!((result[99] - 10000.0).abs() < 1e-10); // 100^2 = 10000
    assert!((result[999] - 1000000.0).abs() < 1e-10); // 1000^2 = 1000000
}

// High exponent
#[test]
fn test_powi_high_exponent() {
    let x = array![2.0_f64, 3.0];
    let result = powi_simd(&x.view(), 10);

    assert!((result[0] - 1024.0).abs() < 1e-10); // 2^10 = 1024
    assert!((result[1] - 59049.0).abs() < 1e-10); // 3^10 = 59049
}

// Fractional bases
#[test]
fn test_powi_fractional_bases() {
    let x = array![0.5_f64, 0.1, 0.25];
    let result = powi_simd(&x.view(), 2);

    assert!((result[0] - 0.25).abs() < 1e-10); // 0.5^2 = 0.25
    assert!((result[1] - 0.01).abs() < 1e-10); // 0.1^2 = 0.01
    assert!((result[2] - 0.0625).abs() < 1e-10); // 0.25^2 = 0.0625
}

// Application: Variance calculation (second moment)
#[test]
fn test_powi_variance_application() {
    // Computing (x - mean)^2 for variance
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mean = 3.0;
    let deviations = x.mapv(|v| v - mean);
    let squared_deviations = powi_simd(&deviations.view(), 2);

    assert!((squared_deviations[0] - 4.0).abs() < 1e-10); // (1-3)^2 = 4
    assert!((squared_deviations[1] - 1.0).abs() < 1e-10); // (2-3)^2 = 1
    assert!((squared_deviations[2] - 0.0).abs() < 1e-10); // (3-3)^2 = 0
    assert!((squared_deviations[3] - 1.0).abs() < 1e-10); // (4-3)^2 = 1
    assert!((squared_deviations[4] - 4.0).abs() < 1e-10); // (5-3)^2 = 4
}

// ============================================================================
// gamma_simd Tests (Phase 26)
// ============================================================================

// Basic correctness f32
#[test]
fn test_gamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = gamma_simd(&x.view());

    // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(5) = 24
    // f32 has lower precision, so we use looser tolerances
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 2.0).abs() < 1e-4);
    assert!((result[3] - 6.0).abs() < 1e-3);
    assert!((result[4] - 24.0).abs() < 0.01);
}

// Basic correctness f64
#[test]
fn test_gamma_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = gamma_simd(&x.view());

    // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(5) = 24
    assert!((result[0] - 1.0).abs() < 1e-14);
    assert!((result[1] - 1.0).abs() < 1e-14);
    assert!((result[2] - 2.0).abs() < 1e-14);
    assert!((result[3] - 6.0).abs() < 1e-13);
    assert!((result[4] - 24.0).abs() < 1e-12);
}

// Empty array
#[test]
fn test_gamma_simd_empty() {
    let x: Array1<f64> = array![];
    let result = gamma_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Half-integer value: Γ(1/2) = √π
#[test]
fn test_gamma_half() {
    let x = array![0.5_f64];
    let result = gamma_simd(&x.view());
    let expected = std::f64::consts::PI.sqrt();
    assert!(
        (result[0] - expected).abs() < 1e-14,
        "Γ(1/2) should be √π, got {}, expected {}",
        result[0],
        expected
    );
}

// Half-integer values: Γ(3/2) = √π/2, Γ(5/2) = 3√π/4
#[test]
fn test_gamma_half_integers() {
    let x = array![0.5_f64, 1.5, 2.5];
    let result = gamma_simd(&x.view());

    let sqrt_pi = std::f64::consts::PI.sqrt();
    assert!((result[0] - sqrt_pi).abs() < 1e-14); // Γ(1/2) = √π
    assert!((result[1] - sqrt_pi / 2.0).abs() < 1e-14); // Γ(3/2) = √π/2
    assert!((result[2] - 3.0 * sqrt_pi / 4.0).abs() < 1e-13); // Γ(5/2) = 3√π/4
}

// Functional equation: Γ(z+1) = z·Γ(z)
#[test]
fn test_gamma_functional_equation() {
    let z = array![2.5_f64, 3.5, 4.5];
    let gamma_z = gamma_simd(&z.view());
    let z_plus_1 = z.mapv(|v| v + 1.0);
    let gamma_z_plus_1 = gamma_simd(&z_plus_1.view());

    for i in 0..z.len() {
        let expected = z[i] * gamma_z[i];
        let relative_error = ((gamma_z_plus_1[i] - expected) / expected).abs();
        assert!(
            relative_error < 1e-14,
            "Functional equation failed: Γ({}) = {}, but {}·Γ({}) = {}",
            z[i] + 1.0,
            gamma_z_plus_1[i],
            z[i],
            z[i],
            expected
        );
    }
}

// Negative values with reflection formula
#[test]
fn test_gamma_negative_values() {
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = gamma_simd(&x.view());

    // These values use the reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
    // For z = -0.5: Γ(-0.5) = -2√π
    let expected_minus_half = -2.0 * std::f64::consts::PI.sqrt();
    assert!((result[0] - expected_minus_half).abs() < 1e-13);

    // All should be finite
    assert!(result[1].is_finite());
    assert!(result[2].is_finite());
}

// Negative integers return infinity (poles)
#[test]
fn test_gamma_negative_integers() {
    let x = array![-1.0_f64, -2.0, -3.0];
    let result = gamma_simd(&x.view());

    // Gamma has poles at negative integers
    // The reflection formula involves sin(πx) which is ~0 for integers
    // Due to numerical precision, we expect very large values or infinity
    for &val in result.iter() {
        assert!(
            val.abs() > 1e10 || val.is_infinite(),
            "Gamma at negative integers should be very large or infinite"
        );
    }
}

// NaN propagation
#[test]
fn test_gamma_nan() {
    let x = array![f64::NAN, 1.0, 2.0];
    let result = gamma_simd(&x.view());

    assert!(result[0].is_nan());
    assert!(result[1].is_finite());
    assert!(result[2].is_finite());
}

// Large positive values
#[test]
fn test_gamma_large_positive() {
    let x = array![10.0_f64, 15.0, 20.0];
    let result = gamma_simd(&x.view());

    // Γ(10) = 9! = 362880
    assert!((result[0] - 362880.0).abs() < 1.0);

    // All should be large and finite (or infinity for very large inputs)
    assert!(result[1] > 1e10);
    assert!(result[2] > 1e15);
}

// Small positive values (< 1)
#[test]
fn test_gamma_small_positive() {
    let x = array![0.1_f64, 0.2, 0.3];
    let result = gamma_simd(&x.view());

    // For small x, Γ(x) ≈ 1/x (dominant term)
    // Γ(0.1) ≈ 9.51, Γ(0.2) ≈ 4.59, Γ(0.3) ≈ 2.99
    assert!((result[0] - 9.51350769866873).abs() < 0.01);
    assert!((result[1] - 4.590843711998803).abs() < 0.01);
    assert!((result[2] - 2.9915689876875904).abs() < 0.01);
}

// Large array (triggers SIMD path)
#[test]
fn test_gamma_large_array() {
    let size = 10_000;
    let data: Array1<f64> =
        Array1::from_vec((0..size).map(|i| 1.0 + (i as f64) * 0.0001).collect());

    let result = gamma_simd(&data.view());

    assert_eq!(result.len(), size);
    // Γ(1) = 1
    assert!((result[0] - 1.0).abs() < 1e-13);
    // Values should increase monotonically for x > 1.46
    for i in 5000..5010 {
        assert!(result[i] < result[i + 1]);
    }
}

// Comparison with known values
#[test]
fn test_gamma_known_values() {
    let x = array![
        1.0_f64, // Γ(1) = 1
        2.0,     // Γ(2) = 1
        6.0,     // Γ(6) = 120
        0.5,     // Γ(1/2) = √π ≈ 1.772453850905516
    ];
    let result = gamma_simd(&x.view());
    let expected = array![1.0, 1.0, 120.0, std::f64::consts::PI.sqrt()];

    for i in 0..x.len() {
        let relative_error = ((result[i] - expected[i]) / expected[i]).abs();
        assert!(
            relative_error < 1e-13,
            "Γ({}) = {}, expected {}, relative error = {}",
            x[i],
            result[i],
            expected[i],
            relative_error
        );
    }
}

// Statistical application: Beta function B(α,β) = Γ(α)Γ(β)/Γ(α+β)
#[test]
fn test_gamma_beta_function() {
    let alpha = 2.0_f64;
    let beta = 3.0_f64;

    let params = array![alpha, beta, alpha + beta];
    let gammas = gamma_simd(&params.view());

    let beta_function = (gammas[0] * gammas[1]) / gammas[2];
    let expected = 1.0 / 12.0; // B(2,3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12

    assert!((beta_function - expected).abs() < 1e-14);
}

// Ratio property: Γ(n+1)/Γ(n) = n
#[test]
fn test_gamma_ratio_property() {
    let n = array![5.0_f64, 7.5, 10.0];
    let n_plus_1 = n.mapv(|v| v + 1.0);

    let gamma_n = gamma_simd(&n.view());
    let gamma_n_plus_1 = gamma_simd(&n_plus_1.view());

    for i in 0..n.len() {
        let ratio = gamma_n_plus_1[i] / gamma_n[i];
        let relative_error = ((ratio - n[i]) / n[i]).abs();
        assert!(
            relative_error < 1e-14,
            "Ratio property failed: Γ({})/Γ({}) = {}, expected {}",
            n[i] + 1.0,
            n[i],
            ratio,
            n[i]
        );
    }
}

// Symmetry around 2: Γ(2+x) and Γ(2-x) relationship
#[test]
fn test_gamma_symmetry() {
    let x = array![0.3_f64, 0.5, 0.7];
    let two_plus_x = x.mapv(|v| 2.0 + v);
    let two_minus_x = x.mapv(|v| 2.0 - v);

    let gamma_plus = gamma_simd(&two_plus_x.view());
    let gamma_minus = gamma_simd(&two_minus_x.view());

    // Both should be finite and positive
    for i in 0..x.len() {
        assert!(gamma_plus[i] > 0.0 && gamma_plus[i].is_finite());
        assert!(gamma_minus[i] > 0.0 && gamma_minus[i].is_finite());
    }
}

// Values near 1
#[test]
fn test_gamma_near_one() {
    let x = array![0.9_f64, 0.95, 1.0, 1.05, 1.1];
    let result = gamma_simd(&x.view());

    // Γ(1) = 1, and gamma is continuous, so values near 1 should be close to 1
    assert!((result[2] - 1.0).abs() < 1e-14); // Γ(1) = 1 exactly
    assert!((result[0] - 1.0).abs() < 0.1); // Γ(0.9) ≈ 1.068
    assert!((result[4] - 1.0).abs() < 0.1); // Γ(1.1) ≈ 0.951
}

// Duplication formula test: Γ(2z) related to Γ(z) and Γ(z+1/2)
#[test]
fn test_gamma_duplication_formula() {
    let z = 2.0_f64;
    let params = array![z, z + 0.5, 2.0 * z];
    let gammas = gamma_simd(&params.view());

    // Legendre duplication formula: Γ(2z) = (2^(2z-1))/√π · Γ(z)Γ(z+1/2)
    let two_power = 2.0_f64.powf(2.0 * z - 1.0);
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let expected = (two_power / sqrt_pi) * gammas[0] * gammas[1];

    let relative_error = ((gammas[2] - expected) / expected).abs();
    assert!(
        relative_error < 1e-12,
        "Duplication formula failed: got {}, expected {}",
        gammas[2],
        expected
    );
}

// Monotonicity for x > 1.46 (local minimum at x ≈ 1.46)
#[test]
fn test_gamma_monotonicity() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0, 6.0];
    let result = gamma_simd(&x.view());

    // For x > 1.46, gamma is strictly increasing
    for i in 0..(x.len() - 1) {
        assert!(
            result[i] < result[i + 1],
            "Γ should be monotonically increasing for x > 1.46"
        );
    }
}

// ============================================================================
