//! Phase 68: SIMD log/exp tests (expm1 and log1p)

use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Phase 68: SIMD expm1 and log1p tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::expm1_simd;
use scirs2_core::ndarray_ext::elementwise::log1p_simd;

// ============================================================================
// expm1 tests
// ============================================================================

/// Test expm1 basic f32 correctness
#[test]
fn test_expm1_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0];

    let result = expm1_simd(&x.view());

    // exp(0) - 1 = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // exp(1) - 1 ≈ 1.718
    assert!((result[1] - (1.0_f32.exp() - 1.0)).abs() < 1e-5);
    // exp(-1) - 1 ≈ -0.632
    assert!((result[2] - ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
}

/// Test expm1 basic f64 correctness
#[test]
fn test_expm1_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0];

    let result = expm1_simd(&x.view());

    // exp(0) - 1 = 0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // exp(1) - 1 ≈ 1.718
    assert!((result[1] - (1.0_f64.exp() - 1.0)).abs() < 1e-14);
    // exp(-1) - 1 ≈ -0.632
    assert!((result[2] - ((-1.0_f64).exp() - 1.0)).abs() < 1e-14);
}

/// Test expm1 empty array
#[test]
fn test_expm1_simd_empty() {
    let x: Array1<f64> = array![];
    let result = expm1_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test expm1 large array (SIMD path)
#[test]
fn test_expm1_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| (i as f64 - 5000.0) * 0.001).collect());

    let result = expm1_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 5000, 9999] {
        let expected = x[i].exp_m1();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "expm1[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test expm1 numerical stability for small values
#[test]
fn test_expm1_simd_small_values() {
    // For very small x, exp(x) - 1 ≈ x (first term of Taylor series)
    // Direct calculation exp(x) - 1 would lose precision here

    let x = array![1e-15_f64, 1e-14, 1e-13, 1e-12, 1e-10];

    let result = expm1_simd(&x.view());

    // For very small x, expm1(x) ≈ x with high precision
    for i in 0..x.len() {
        // The relative error should be very small
        let rel_error = if x[i].abs() > 1e-20 {
            (result[i] - x[i]).abs() / x[i].abs()
        } else {
            (result[i] - x[i]).abs()
        };
        assert!(
            rel_error < 1e-10,
            "expm1 should be numerically stable for small x, relative error = {} at x = {}",
            rel_error,
            x[i]
        );
    }
}

/// Test expm1 vs naive exp(x) - 1 comparison for small values
#[test]
fn test_expm1_simd_vs_naive_small() {
    // This demonstrates the numerical advantage of expm1 over exp(x) - 1

    let x = array![1e-16_f64];
    let result = expm1_simd(&x.view());

    // Naive calculation would give 0 due to floating point precision loss
    let naive = x[0].exp() - 1.0;

    // expm1 should preserve precision
    // For x = 1e-16, expm1(x) should be approximately 1e-16
    // while naive exp(x) - 1 = 0 (precision loss)
    assert!(
        (result[0] - x[0]).abs() < 1e-30,
        "expm1 should preserve precision for tiny values"
    );
    // The naive calculation has lost precision
    assert_eq!(naive, 0.0, "naive exp(x)-1 loses precision for tiny x");
}

/// Test expm1 identity: expm1(-x) = -expm1(x) / (1 + expm1(x))
#[test]
fn test_expm1_simd_identity() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];

    let result_pos = expm1_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let result_neg = expm1_simd(&neg_x.view());

    // expm1(-x) = -expm1(x) / (1 + expm1(x)) = -expm1(x) / exp(x)
    for i in 0..x.len() {
        let expected_neg = -result_pos[i] / (1.0 + result_pos[i]);
        assert!(
            (result_neg[i] - expected_neg).abs() < 1e-14,
            "expm1(-x) identity failed at x = {}",
            x[i]
        );
    }
}

/// Test expm1 financial compound interest use case
#[test]
fn test_expm1_simd_compound_interest() {
    // Continuous compound interest: A = P * e^(rt)
    // Interest earned = P * (e^(rt) - 1) = P * expm1(rt)
    // For small rates, this is critical for accuracy

    let principal = 10000.0_f64;
    let rate = 0.0001_f64; // 0.01% daily rate
    let time = 1.0_f64; // 1 day

    let rt = array![rate * time];
    let growth_factor = expm1_simd(&rt.view());
    let interest = principal * growth_factor[0];

    // For small rates: interest ≈ P * r * t (simple interest approximation)
    let simple_interest = principal * rate * time;

    // The difference between compound and simple interest is very small:
    // compound_interest - simple_interest ≈ P * r² * t² / 2 = 10000 * 1e-8 / 2 = 5e-5
    // So the interest should be very close to simple interest
    let diff = (interest - simple_interest).abs();
    assert!(
        diff < 1e-3,
        "expm1 preserves precision for small compound interest calculations, diff = {}",
        diff
    );

    // Also verify the compound interest is slightly higher than simple interest
    // (due to compounding effect)
    assert!(
        interest >= simple_interest,
        "Compound interest should be >= simple interest"
    );
}

// ============================================================================
// log1p tests
// ============================================================================

/// Test log1p basic f32 correctness
#[test]
fn test_log1p_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -0.5, 3.0];

    let result = log1p_simd(&x.view());

    // ln(1 + 0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // ln(1 + 1) = ln(2)
    assert!((result[1] - 2.0_f32.ln()).abs() < 1e-6);
    // ln(1 + (-0.5)) = ln(0.5)
    assert!((result[2] - 0.5_f32.ln()).abs() < 1e-6);
}

/// Test log1p basic f64 correctness
#[test]
fn test_log1p_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -0.5, 3.0];

    let result = log1p_simd(&x.view());

    // ln(1 + 0) = 0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // ln(1 + 1) = ln(2)
    assert!((result[1] - 2.0_f64.ln()).abs() < 1e-14);
    // ln(1 + (-0.5)) = ln(0.5)
    assert!((result[2] - 0.5_f64.ln()).abs() < 1e-14);
}

/// Test log1p empty array
#[test]
fn test_log1p_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log1p_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test log1p large array (SIMD path)
#[test]
fn test_log1p_simd_large_array() {
    let n = 10000;
    // Use values in (-0.9, 10) to avoid ln of negative numbers
    let x = Array1::from_vec(
        (0..n)
            .map(|i| -0.9 + (i as f64 * 11.0 / n as f64))
            .collect(),
    );

    let result = log1p_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 5000, 9999] {
        let expected = x[i].ln_1p();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "log1p[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test log1p numerical stability for small values
#[test]
fn test_log1p_simd_small_values() {
    // For very small x, ln(1 + x) ≈ x (first term of Taylor series)
    // Direct calculation ln(1 + x) would lose precision here

    let x = array![1e-15_f64, 1e-14, 1e-13, 1e-12, 1e-10];

    let result = log1p_simd(&x.view());

    // For very small x, log1p(x) ≈ x with high precision
    for i in 0..x.len() {
        let rel_error = if x[i].abs() > 1e-20 {
            (result[i] - x[i]).abs() / x[i].abs()
        } else {
            (result[i] - x[i]).abs()
        };
        assert!(
            rel_error < 1e-10,
            "log1p should be numerically stable for small x, relative error = {} at x = {}",
            rel_error,
            x[i]
        );
    }
}

/// Test log1p vs naive ln(1 + x) comparison for small values
#[test]
fn test_log1p_simd_vs_naive_small() {
    // This demonstrates the numerical advantage of log1p over ln(1 + x)

    let x = array![1e-16_f64];
    let result = log1p_simd(&x.view());

    // Naive calculation would give 0 due to floating point precision loss
    let naive = (1.0 + x[0]).ln();

    // log1p should preserve precision
    // For x = 1e-16, log1p(x) should be approximately 1e-16
    // while naive ln(1 + x) = 0 (precision loss because 1 + 1e-16 = 1 in f64)
    assert!(
        (result[0] - x[0]).abs() < 1e-30,
        "log1p should preserve precision for tiny values"
    );
    assert_eq!(naive, 0.0, "naive ln(1+x) loses precision for tiny x");
}

/// Test log1p edge cases
#[test]
fn test_log1p_simd_edge_cases() {
    let x = array![-1.0_f64, -2.0];

    let result = log1p_simd(&x.view());

    // ln(1 + (-1)) = ln(0) = -∞
    assert!(result[0].is_infinite() && result[0] < 0.0);
    // ln(1 + (-2)) = ln(-1) = NaN
    assert!(result[1].is_nan());
}

/// Test log1p identity: log1p(x) + log1p(y) = log1p(x + y + xy)
#[test]
fn test_log1p_simd_addition_identity() {
    // log(a) + log(b) = log(ab)
    // log1p(x) + log1p(y) = ln(1+x) + ln(1+y) = ln((1+x)(1+y)) = ln(1 + x + y + xy)
    // = log1p(x + y + xy)

    let x = array![0.1_f64, 0.2, 0.3, 0.5];
    let y = array![0.2_f64, 0.3, 0.1, 0.5];

    let log1p_x = log1p_simd(&x.view());
    let log1p_y = log1p_simd(&y.view());

    let combined = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi + yi + xi * yi)
        .collect::<Vec<_>>();
    let combined_arr = Array1::from_vec(combined);
    let log1p_combined = log1p_simd(&combined_arr.view());

    for i in 0..x.len() {
        let sum = log1p_x[i] + log1p_y[i];
        assert!(
            (sum - log1p_combined[i]).abs() < 1e-14,
            "log1p addition identity failed at i = {}",
            i
        );
    }
}

/// Test log1p inverse relationship with expm1
#[test]
fn test_log1p_simd_inverse_of_expm1() {
    // log1p(expm1(x)) = x for any x
    // expm1(log1p(x)) = x for x > -1

    let x = array![0.1_f64, 0.5, 1.0, 2.0, -0.5];

    let expm1_x = expm1_simd(&x.view());
    let log1p_expm1_x = log1p_simd(&expm1_x.view());

    for i in 0..x.len() {
        assert!(
            (log1p_expm1_x[i] - x[i]).abs() < 1e-14,
            "log1p(expm1(x)) should equal x"
        );
    }

    // Also test expm1(log1p(x)) = x
    let log1p_x = log1p_simd(&x.view());
    let expm1_log1p_x = expm1_simd(&log1p_x.view());

    for i in 0..x.len() {
        assert!(
            (expm1_log1p_x[i] - x[i]).abs() < 1e-14,
            "expm1(log1p(x)) should equal x"
        );
    }
}

/// Test log1p binary cross-entropy use case
#[test]
fn test_log1p_simd_cross_entropy() {
    // Binary cross-entropy: -y*log(p) - (1-y)*log(1-p)
    // When p is very close to 0 or 1, we need log1p for stability
    // log(1-p) = log1p(-p)

    let p = array![0.99999_f64, 0.9999, 0.999]; // p close to 1
    let neg_p = p.mapv(|v| -v);

    let log_1_minus_p = log1p_simd(&neg_p.view());

    // For p close to 1, log(1-p) should be a large negative number
    for (i, &v) in log_1_minus_p.iter().enumerate() {
        assert!(
            v < 0.0,
            "log(1-p) should be negative for p close to 1, got {} at index {}",
            v,
            i
        );
        assert!(v.is_finite(), "log(1-p) should be finite for p < 1");
    }
}
