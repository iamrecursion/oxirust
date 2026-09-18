//! Power function SIMD tests: powf, pow, powi

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{ln_simd, pow_simd, powf_simd, powi_simd, sqrt_simd};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============================================================================
// Powf SIMD Tests (scalar exponent)
// ============================================================================

#[test]
fn test_powf_simd_f64_basic() {
    let base = array![2.0, 3.0, 4.0, 5.0];
    let result = powf_simd(&base.view(), 2.0);

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 9.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 25.0);
}

#[test]
fn test_powf_simd_f32_basic() {
    let base = array![2.0f32, 3.0, 4.0, 5.0];
    let result = powf_simd(&base.view(), 2.0);

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 9.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 25.0);
}

#[test]
fn test_powf_simd_f64_empty() {
    let base: Array1<f64> = array![];
    let result = powf_simd(&base.view(), 2.0);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_powf_simd_f64_exp_zero() {
    // x^0 = 1 for any x
    let base = array![2.0, 3.0, 0.0, -5.0];
    let result = powf_simd(&base.view(), 0.0);

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_powf_simd_f64_exp_one() {
    // x^1 = x
    let base = array![2.0, 3.0, 4.0, 5.0];
    let result = powf_simd(&base.view(), 1.0);

    for i in 0..base.len() {
        assert_eq!(result[i], base[i]);
    }
}

#[test]
fn test_powf_simd_f64_negative_base() {
    let base = array![-2.0, -3.0];
    let result = powf_simd(&base.view(), 3.0); // Odd integer exponent

    assert_eq!(result[0], -8.0);
    assert_eq!(result[1], -27.0);
}

#[test]
fn test_powf_simd_f64_negative_base_non_integer() {
    let base = array![-2.0];
    let result = powf_simd(&base.view(), 2.5); // Non-integer exponent

    // Negative base with non-integer exponent should return NaN
    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_powf_simd_f64_zero_base_negative_exp() {
    let base = array![0.0];
    let result = powf_simd(&base.view(), -1.0);

    // 0^negative = infinity
    let val: f64 = result[0];
    assert!(val.is_infinite() && val.is_sign_positive());
}

#[test]
fn test_powf_simd_f64_nan() {
    let base = array![f64::NAN];
    let result = powf_simd(&base.view(), 2.0);
    assert!(result[0].is_nan());
}

#[test]
fn test_powf_simd_f64_large_array() {
    let base: Array1<f64> = Array1::linspace(1.0, 100.0, 10000);
    let result = powf_simd(&base.view(), 2.0);

    // Check some values
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 10000.0).abs() < 1e-8);
}

#[test]
fn test_powf_simd_f32_large_array() {
    let base: Array1<f32> = Array1::linspace(1.0, 100.0, 10000);
    let result = powf_simd(&base.view(), 2.0);

    // All values should be finite and positive
    for &val in result.iter() {
        assert!(val.is_finite() && val >= 0.0);
    }
}

// ============================================================================
// ============================================================================
// Pow SIMD Tests (array exponent)
// ============================================================================

#[test]
fn test_pow_simd_f64_basic() {
    let base = array![2.0, 3.0, 4.0, 5.0];
    let exp = array![2.0, 3.0, 2.0, 1.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 27.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 5.0);
}

#[test]
fn test_pow_simd_f32_basic() {
    let base = array![2.0f32, 3.0, 4.0, 5.0];
    let exp = array![2.0f32, 3.0, 2.0, 1.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 27.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 5.0);
}

#[test]
fn test_pow_simd_f64_empty() {
    let base: Array1<f64> = array![];
    let exp: Array1<f64> = array![];
    let result = pow_simd(&base.view(), &exp.view());
    assert_eq!(result.len(), 0);
}

#[test]
#[should_panic(expected = "Base and exponent arrays must have the same length")]
fn test_pow_simd_f64_mismatched_lengths() {
    let base = array![2.0, 3.0, 4.0];
    let exp = array![2.0, 3.0];
    let _ = pow_simd(&base.view(), &exp.view());
}

#[test]
fn test_pow_simd_f64_exp_zero() {
    // x^0 = 1 for any x
    let base = array![2.0, 3.0, 0.0, -5.0];
    let exp = array![0.0, 0.0, 0.0, 0.0];
    let result = pow_simd(&base.view(), &exp.view());

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_pow_simd_f64_exp_one() {
    // x^1 = x
    let base = array![2.0, 3.0, 4.0, 5.0];
    let exp = array![1.0, 1.0, 1.0, 1.0];
    let result = pow_simd(&base.view(), &exp.view());

    for i in 0..base.len() {
        assert_eq!(result[i], base[i]);
    }
}

#[test]
fn test_pow_simd_f64_mixed_exponents() {
    let base = array![2.0, 3.0, 4.0];
    let exp = array![0.0, 1.0, 2.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], 1.0); // 2^0 = 1
    assert_eq!(result[1], 3.0); // 3^1 = 3
    assert_eq!(result[2], 16.0); // 4^2 = 16
}

#[test]
fn test_pow_simd_f64_negative_bases() {
    let base = array![-2.0, -3.0];
    let exp = array![3.0, 2.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], -8.0); // (-2)^3 = -8
    assert_eq!(result[1], 9.0); // (-3)^2 = 9
}

#[test]
fn test_pow_simd_f64_nan_propagation() {
    let base = array![f64::NAN, 2.0];
    let exp = array![2.0, f64::NAN];
    let result = pow_simd(&base.view(), &exp.view());

    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_pow_simd_f64_large_array() {
    let base: Array1<f64> = Array1::linspace(1.0, 10.0, 10000);
    let exp: Array1<f64> = Array1::from_elem(10000, 2.0);
    let result = pow_simd(&base.view(), &exp.view());

    // Check first and last values
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 100.0).abs() < 1e-8);
}

#[test]
fn test_pow_simd_f32_large_array() {
    let base: Array1<f32> = Array1::linspace(1.0, 10.0, 10000);
    let exp: Array1<f32> = Array1::from_elem(10000, 2.0);
    let result = pow_simd(&base.view(), &exp.view());

    // All values should be finite and positive
    for &val in result.iter() {
        assert!(val.is_finite() && val >= 0.0);
    }
}

// ============================================================================
// Power Function Property Tests
// ============================================================================

#[test]
fn test_pow_exp_log_identity() {
    // log(x^y) = y * log(x)
    let base = array![2.0, 3.0, 5.0, 10.0];
    let exp = array![2.0, 3.0, 2.0, 2.0];

    let pow_result = pow_simd(&base.view(), &exp.view());
    let log_pow = ln_simd(&pow_result.view());

    let log_base = ln_simd(&base.view());
    let exp_log = exp
        .iter()
        .zip(log_base.iter())
        .map(|(&e, &l)| e * l)
        .collect::<Array1<_>>();

    for i in 0..base.len() {
        let diff: f64 = log_pow[i] - exp_log[i];
        assert!(diff.abs() < 1e-10, "log(x^y) != y*log(x) at index {}", i);
    }
}

#[test]
fn test_powf_square_sqrt_inverse() {
    // sqrt(x^2) = |x| for x >= 0
    let base = array![2.0, 3.0, 4.0, 5.0];
    let squared = powf_simd(&base.view(), 2.0);
    let sqrt_result = sqrt_simd(&squared.view());

    for i in 0..base.len() {
        let diff: f64 = sqrt_result[i] - base[i];
        assert!(diff.abs() < 1e-10, "sqrt(x^2) != x at index {}", i);
    }
}

// =============================================================================
