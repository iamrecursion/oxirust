//! Exponential and logarithm SIMD tests: exp, ln, log10, log2, exp2, cbrt, ln_1p, exp_m1

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{exp_simd, ln_simd, log10_simd, log2_simd};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============================================================================
// Exp SIMD Tests
// ============================================================================

#[test]
fn test_exp_simd_f64_basic() {
    let x = array![0.0, 1.0, 2.0];
    let result = exp_simd(&x.view());

    let diff0: f64 = result[0] - 1.0;
    assert!(diff0.abs() < 1e-10);
    let diff1: f64 = result[1] - std::f64::consts::E;
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - std::f64::consts::E.powi(2);
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_exp_simd_f32_basic() {
    let x = array![0.0f32, 1.0, 2.0];
    let result = exp_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-6);
    assert!((result[2] - std::f32::consts::E.powi(2)).abs() < 1e-6);
}

#[test]
fn test_exp_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = exp_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_exp_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = exp_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_exp_simd_f64_zero() {
    let x = array![0.0];
    let result = exp_simd(&x.view());

    let diff: f64 = result[0] - 1.0;
    assert!(diff.abs() < 1e-10);
}

#[test]
fn test_exp_simd_f32_zero() {
    let x = array![0.0f32];
    let result = exp_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_exp_simd_f64_negative() {
    let x = array![-1.0, -2.0, -5.0];
    let result = exp_simd(&x.view());

    let diff0: f64 = result[0] - 0.367879441;
    assert!(diff0.abs() < 1e-9);
    let diff1: f64 = result[1] - 0.135335283;
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - 0.006737947;
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_exp_simd_f32_negative() {
    let x = array![-1.0f32, -2.0, -5.0];
    let result = exp_simd(&x.view());

    assert!((result[0] - 0.367_879_45).abs() < 1e-6);
    assert!((result[1] - 0.135_335_28).abs() < 1e-6);
    assert!((result[2] - 0.006737947).abs() < 1e-6);
}

#[test]
fn test_exp_simd_f64_large_positive() {
    let x = array![10.0, 20.0];
    let result = exp_simd(&x.view());

    let diff: f64 = result[0] - 22026.465794806718;
    assert!(diff.abs() < 1e-9);
    assert!(result[1] > 1e8); // exp(20) is very large
}

#[test]
fn test_exp_simd_f32_large_positive() {
    let x = array![10.0f32];
    let result = exp_simd(&x.view());

    assert!((result[0] - 22_026.465).abs() < 1e-2);
}

#[test]
fn test_exp_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = exp_simd(&x.view());

    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_exp_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = exp_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_exp_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = exp_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
    assert_eq!(result[1], 0.0); // exp(-∞) = 0
}

#[test]
fn test_exp_simd_f32_infinity() {
    let x = array![f32::INFINITY, f32::NEG_INFINITY];
    let result = exp_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
    assert_eq!(result[1], 0.0);
}

#[test]
fn test_exp_simd_f64_large_array() {
    let data: Vec<f64> = (0..10000).map(|i| ((i as f64) * 0.001) - 5.0).collect();
    let x = Array1::from_vec(data.clone());
    let result = exp_simd(&x.view());

    // Verify correctness
    for i in 0..data.len() {
        let expected = data[i].exp();
        let diff = (result[i] - expected).abs();
        let rel_error = if expected.abs() > 1e-10 {
            diff / expected.abs()
        } else {
            diff
        };
        assert!(
            rel_error < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            result[i],
            expected
        );
    }
}

#[test]
fn test_exp_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000).map(|i| ((i as f32) * 0.001) - 5.0).collect();
    let x = Array1::from_vec(data.clone());
    let result = exp_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].exp();
        let diff = (result[i] - expected).abs();
        let rel_error = if expected.abs() > 1e-6 {
            diff / expected.abs()
        } else {
            diff
        };
        assert!(rel_error < 1e-5, "Mismatch at index {}", i);
    }
}

// ============================================================================
// ============================================================================
// Ln (Natural Logarithm) SIMD Tests
// ============================================================================

#[test]
fn test_ln_simd_f64_basic() {
    let x = array![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)];
    let result = ln_simd(&x.view());

    let diff0: f64 = result[0] - 0.0;
    assert!(diff0.abs() < 1e-10);
    let diff1: f64 = result[1] - 1.0;
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - 2.0;
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_ln_simd_f32_basic() {
    let x = array![1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)];
    let result = ln_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_ln_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = ln_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ln_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = ln_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ln_simd_f64_one() {
    let x = array![1.0];
    let result = ln_simd(&x.view());

    let diff: f64 = result[0] - 0.0;
    assert!(diff.abs() < 1e-10);
}

#[test]
fn test_ln_simd_f32_one() {
    let x = array![1.0f32];
    let result = ln_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_ln_simd_f64_zero() {
    let x = array![0.0];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f64::NEG_INFINITY);
}

#[test]
fn test_ln_simd_f32_zero() {
    let x = array![0.0f32];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f32::NEG_INFINITY);
}

#[test]
fn test_ln_simd_f64_negative() {
    let x = array![-1.0];
    let result = ln_simd(&x.view());

    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f32_negative() {
    let x = array![-1.0f32];
    let result = ln_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f64_infinity() {
    let x = array![f64::INFINITY];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
}

#[test]
fn test_ln_simd_f32_infinity() {
    let x = array![f32::INFINITY];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
}

#[test]
fn test_ln_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = ln_simd(&x.view());

    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = ln_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f64_small_values() {
    let x = array![0.1, 0.01, 0.001];
    let result = ln_simd(&x.view());

    let diff0: f64 = result[0] - (-std::f64::consts::LN_10);
    assert!(diff0.abs() < 1e-9);
    let diff1: f64 = result[1] - (-2.0 * std::f64::consts::LN_10);
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - (-3.0 * std::f64::consts::LN_10);
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_ln_simd_f32_small_values() {
    let x = array![0.1f32, 0.01, 0.001];
    let result = ln_simd(&x.view());

    assert!((result[0] - (-std::f32::consts::LN_10)).abs() < 1e-6);
    assert!((result[1] - (-2.0 * std::f32::consts::LN_10)).abs() < 1e-6);
    assert!((result[2] - (-3.0 * std::f32::consts::LN_10)).abs() < 1e-6);
}

#[test]
fn test_ln_simd_f64_large_array() {
    let data: Vec<f64> = (1..=10000).map(|i| (i as f64) * 0.1).collect();
    let x = Array1::from_vec(data.clone());
    let result = ln_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].ln();
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
fn test_ln_simd_f32_large_array() {
    let data: Vec<f32> = (1..=10000).map(|i| (i as f32) * 0.1).collect();
    let x = Array1::from_vec(data.clone());
    let result = ln_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].ln();
        let diff = (result[i] - expected).abs();
        assert!(diff < 1e-5, "Mismatch at index {}", i);
    }
}

// ============================================================================
// Combined Property Tests
// ============================================================================

#[test]
fn test_exp_ln_inverse() {
    // exp(ln(x)) = x for positive x
    let x = array![1.0, 2.0, 5.0, 10.0, 100.0];
    let ln_x = ln_simd(&x.view());
    let exp_ln_x = exp_simd(&ln_x.view());

    for i in 0..x.len() {
        let diff: f64 = exp_ln_x[i] - x[i];
        let abs_diff = diff.abs();
        let rel_error = abs_diff / x[i];
        assert!(rel_error < 1e-10, "exp(ln(x)) != x at index {}", i);
    }
}

#[test]
fn test_ln_exp_inverse() {
    // ln(exp(x)) = x
    let x = array![-5.0, -1.0, 0.0, 1.0, 5.0];
    let exp_x = exp_simd(&x.view());
    let ln_exp_x = ln_simd(&exp_x.view());

    for i in 0..x.len() {
        let diff: f64 = ln_exp_x[i] - x[i];
        let abs_diff = diff.abs();
        assert!(abs_diff < 1e-10, "ln(exp(x)) != x at index {}", i);
    }
}

// ============================================================================
// ============================================================================
// Log10 SIMD Tests
// ============================================================================

#[test]
fn test_log10_simd_f64_basic() {
    let x = array![1.0_f64, 10.0, 100.0, 1000.0];
    let result = log10_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!((result[2] - 2.0).abs() < 1e-10);
    assert!((result[3] - 3.0).abs() < 1e-10);
}

#[test]
fn test_log10_simd_f32_basic() {
    let x = array![1.0_f32, 10.0, 100.0, 1000.0];
    let result = log10_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - 3.0).abs() < 1e-6);
}

#[test]
fn test_log10_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log10_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Domain constraint: log10(negative) = NaN
#[test]
fn test_log10_negative_returns_nan() {
    let x = array![1.0_f64, -1.0, 10.0, -10.0];
    let result = log10_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(!result[2].is_nan());
    assert!(result[3].is_nan());
}

// Domain constraint: log10(0) = -∞
#[test]
fn test_log10_zero_returns_neg_inf() {
    let x = array![0.0_f64, 1.0];
    let result = log10_simd(&x.view());

    assert!(result[0].is_infinite() && result[0].is_sign_negative());
    assert!((result[1] - 0.0).abs() < 1e-10);
}

// Logarithm property: log10(a * b) = log10(a) + log10(b)
#[test]
fn test_log10_product_property() {
    let a = array![2.0_f64, 5.0, 10.0];
    let b = array![5.0_f64, 2.0, 10.0];

    let product = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .collect::<Vec<_>>();
    let product_array = Array1::from(product);

    let log_a = log10_simd(&a.view());
    let log_b = log10_simd(&b.view());
    let log_product = log10_simd(&product_array.view());

    for i in 0..a.len() {
        assert!(
            (log_product[i] - (log_a[i] + log_b[i])).abs() < 1e-10,
            "log10(a*b) should equal log10(a) + log10(b)"
        );
    }
}

// Logarithm property: log10(a / b) = log10(a) - log10(b)
#[test]
fn test_log10_quotient_property() {
    let a = array![100.0_f64, 1000.0, 10000.0];
    let b = array![10.0_f64, 100.0, 1000.0];

    let quotient = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x / y)
        .collect::<Vec<_>>();
    let quotient_array = Array1::from(quotient);

    let log_a = log10_simd(&a.view());
    let log_b = log10_simd(&b.view());
    let log_quotient = log10_simd(&quotient_array.view());

    for i in 0..a.len() {
        assert!(
            (log_quotient[i] - (log_a[i] - log_b[i])).abs() < 1e-10,
            "log10(a/b) should equal log10(a) - log10(b)"
        );
    }
}

// Logarithm property: log10(a^n) = n * log10(a)
#[test]
fn test_log10_power_property() {
    let a = array![2.0_f64, 5.0, 10.0];
    let n = 3.0;

    let power = a.mapv(|x| x.powf(n));

    let log_a = log10_simd(&a.view());
    let log_power = log10_simd(&power.view());

    for i in 0..a.len() {
        assert!(
            (log_power[i] - n * log_a[i]).abs() < 1e-10,
            "log10(a^n) should equal n * log10(a)"
        );
    }
}

// Change of base formula: log10(x) = ln(x) / ln(10)
#[test]
fn test_log10_change_of_base() {
    let x = array![1.0_f64, 2.0, 5.0, 10.0, 50.0, 100.0];
    let log10_result = log10_simd(&x.view());
    let ln_result = ln_simd(&x.view());
    let ln10 = 10.0_f64.ln();

    for i in 0..x.len() {
        assert!(
            (log10_result[i] - ln_result[i] / ln10).abs() < 1e-10,
            "log10(x) should equal ln(x) / ln(10)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_log10_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(1.0, 1000.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = log10_simd(&x.view());

    for i in 0..100.min(data.len()) {
        let expected = data[i].log10();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-14,
            "log10 mismatch at {}: {} vs {} (diff: {})",
            i,
            result[i],
            expected,
            diff
        );
    }
}

// ============================================================================
// ============================================================================
// Log2 SIMD Tests
// ============================================================================

#[test]
fn test_log2_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 4.0, 8.0, 16.0];
    let result = log2_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!((result[2] - 2.0).abs() < 1e-10);
    assert!((result[3] - 3.0).abs() < 1e-10);
    assert!((result[4] - 4.0).abs() < 1e-10);
}

#[test]
fn test_log2_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 4.0, 8.0, 16.0];
    let result = log2_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - 3.0).abs() < 1e-6);
    assert!((result[4] - 4.0).abs() < 1e-6);
}

#[test]
fn test_log2_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log2_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Domain constraint: log2(negative) = NaN
#[test]
fn test_log2_negative_returns_nan() {
    let x = array![1.0_f64, -1.0, 2.0, -2.0];
    let result = log2_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(!result[2].is_nan());
    assert!(result[3].is_nan());
}

// Domain constraint: log2(0) = -∞
#[test]
fn test_log2_zero_returns_neg_inf() {
    let x = array![0.0_f64, 1.0];
    let result = log2_simd(&x.view());

    assert!(result[0].is_infinite() && result[0].is_sign_negative());
    assert!((result[1] - 0.0).abs() < 1e-10);
}

// Logarithm property: log2(a * b) = log2(a) + log2(b)
#[test]
fn test_log2_product_property() {
    let a = array![2.0_f64, 4.0, 8.0];
    let b = array![4.0_f64, 2.0, 8.0];

    let product = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .collect::<Vec<_>>();
    let product_array = Array1::from(product);

    let log_a = log2_simd(&a.view());
    let log_b = log2_simd(&b.view());
    let log_product = log2_simd(&product_array.view());

    for i in 0..a.len() {
        assert!(
            (log_product[i] - (log_a[i] + log_b[i])).abs() < 1e-10,
            "log2(a*b) should equal log2(a) + log2(b)"
        );
    }
}

// Logarithm property: log2(a / b) = log2(a) - log2(b)
#[test]
fn test_log2_quotient_property() {
    let a = array![16.0_f64, 64.0, 256.0];
    let b = array![2.0_f64, 4.0, 8.0];

    let quotient = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x / y)
        .collect::<Vec<_>>();
    let quotient_array = Array1::from(quotient);

    let log_a = log2_simd(&a.view());
    let log_b = log2_simd(&b.view());
    let log_quotient = log2_simd(&quotient_array.view());

    for i in 0..a.len() {
        assert!(
            (log_quotient[i] - (log_a[i] - log_b[i])).abs() < 1e-10,
            "log2(a/b) should equal log2(a) - log2(b)"
        );
    }
}

// Logarithm property: log2(a^n) = n * log2(a)
#[test]
fn test_log2_power_property() {
    let a = array![2.0_f64, 3.0, 5.0];
    let n = 4.0;

    let power = a.mapv(|x| x.powf(n));

    let log_a = log2_simd(&a.view());
    let log_power = log2_simd(&power.view());

    for i in 0..a.len() {
        assert!(
            (log_power[i] - n * log_a[i]).abs() < 1e-10,
            "log2(a^n) should equal n * log2(a)"
        );
    }
}

// Change of base formula: log2(x) = ln(x) / ln(2)
#[test]
fn test_log2_change_of_base() {
    let x = array![1.0_f64, 2.0, 4.0, 8.0, 16.0, 32.0];
    let log2_result = log2_simd(&x.view());
    let ln_result = ln_simd(&x.view());
    let ln2 = 2.0_f64.ln();

    for i in 0..x.len() {
        assert!(
            (log2_result[i] - ln_result[i] / ln2).abs() < 1e-10,
            "log2(x) should equal ln(x) / ln(2)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_log2_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(1.0, 1024.0).expect("Test: operation failed");
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = log2_simd(&x.view());

    for i in 0..100.min(data.len()) {
        let expected = data[i].log2();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-14,
            "log2 mismatch at {}: {} vs {} (diff: {})",
            i,
            result[i],
            expected,
            diff
        );
    }
}

// ============================================================================
// ============================================================================
// Exp2 SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::exp2_simd;

#[test]
fn test_exp2_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, 2.0, 3.0, -1.0];
    let result = exp2_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14);
    assert!((result[1] - 2.0).abs() < 1e-14);
    assert!((result[2] - 4.0).abs() < 1e-14);
    assert!((result[3] - 8.0).abs() < 1e-14);
    assert!((result[4] - 0.5).abs() < 1e-14);
}

#[test]
fn test_exp2_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0, 3.0, -1.0];
    let result = exp2_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 2.0).abs() < 1e-6);
    assert!((result[2] - 4.0).abs() < 1e-6);
    assert!((result[3] - 8.0).abs() < 1e-6);
    assert!((result[4] - 0.5).abs() < 1e-6);
}

#[test]
fn test_exp2_simd_empty() {
    let x: Array1<f64> = array![];
    let result = exp2_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// ============================================================================
// Cbrt SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::cbrt_simd;

#[test]
fn test_cbrt_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, 8.0, 27.0, -8.0, -27.0];
    let result = cbrt_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - 1.0).abs() < 1e-14);
    assert!((result[2] - 2.0).abs() < 1e-14);
    assert!((result[3] - 3.0).abs() < 1e-14);
    assert!((result[4] - (-2.0)).abs() < 1e-14);
    assert!((result[5] - (-3.0)).abs() < 1e-14);
}

#[test]
fn test_cbrt_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 8.0, 27.0, -8.0];
    let result = cbrt_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - 3.0).abs() < 1e-6);
    assert!((result[4] - (-2.0)).abs() < 1e-6);
}

#[test]
fn test_cbrt_simd_inverse_cube() {
    // cbrt(x^3) should equal x
    let x = array![-5.0_f64, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let cubed = x.mapv(|v| v * v * v);
    let result = cbrt_simd(&cubed.view());

    for i in 0..x.len() {
        assert!(
            (result[i] - x[i]).abs() < 1e-14,
            "cbrt(x^3) should equal x at index {}: got {}, expected {}",
            i,
            result[i],
            x[i]
        );
    }
}

// ============================================================================
// Ln_1p SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::ln_1p_simd;

#[test]
fn test_ln_1p_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, std::f64::consts::E - 1.0];
    let result = ln_1p_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-14); // ln(1+0) = ln(1) = 0
    assert!((result[1] - 2.0_f64.ln()).abs() < 1e-14); // ln(1+1) = ln(2)
    assert!((result[2] - 1.0).abs() < 1e-14); // ln(e) = 1
}

#[test]
fn test_ln_1p_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, std::f32::consts::E - 1.0];
    let result = ln_1p_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 2.0_f32.ln()).abs() < 1e-6);
    assert!((result[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_ln_1p_simd_small_values() {
    // Test numerical stability for small values
    let x = array![1e-15_f64, 1e-10, 1e-5];
    let result = ln_1p_simd(&x.view());

    // For small x, ln(1+x) ≈ x
    assert!((result[0] - x[0]).abs() < 1e-28);
    assert!((result[1] - x[1]).abs() < 1e-20);
    assert!((result[2] - x[2]).abs() < 1e-10);
}

// ============================================================================
// Exp_m1 SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::exp_m1_simd;

#[test]
fn test_exp_m1_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, 2.0];
    let result = exp_m1_simd(&x.view());

    let e: f64 = std::f64::consts::E;
    assert!((result[0] - 0.0).abs() < 1e-14); // exp(0) - 1 = 0
    assert!((result[1] - (e - 1.0)).abs() < 1e-14); // exp(1) - 1 = e - 1
    assert!((result[2] - (e * e - 1.0)).abs() < 1e-14); // exp(2) - 1 = e^2 - 1
}

#[test]
fn test_exp_m1_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0];
    let result = exp_m1_simd(&x.view());

    let e: f32 = std::f32::consts::E;
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - (e - 1.0)).abs() < 1e-6);
    assert!((result[2] - (e * e - 1.0)).abs() < 1e-5);
}

#[test]
fn test_exp_m1_simd_small_values() {
    // Test numerical stability for small values
    let x = array![1e-15_f64, 1e-10, 1e-5];
    let result = exp_m1_simd(&x.view());

    // For small x, exp(x)-1 ≈ x
    assert!((result[0] - x[0]).abs() < 1e-28);
    assert!((result[1] - x[1]).abs() < 1e-20);
    assert!((result[2] - x[2]).abs() < 1e-10);
}

#[test]
fn test_exp_m1_ln_1p_inverse() {
    // exp_m1 and ln_1p are inverse functions
    let x = array![-0.5_f64, 0.0, 0.5, 1.0, 2.0];
    let exp_m1_result = exp_m1_simd(&x.view());
    let back = ln_1p_simd(&exp_m1_result.view());

    for i in 0..x.len() {
        assert!(
            (back[i] - x[i]).abs() < 1e-14,
            "ln_1p(exp_m1(x)) should equal x at index {}: got {}, expected {}",
            i,
            back[i],
            x[i]
        );
    }
}

// ============================================================================
