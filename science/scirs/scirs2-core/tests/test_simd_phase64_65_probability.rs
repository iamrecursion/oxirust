//! Phase 64-65: SIMD probability tests (logaddexp and logit)

use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Phase 64-65: SIMD logaddexp and logit tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::logaddexp_simd;
use scirs2_core::ndarray_ext::elementwise::logit_simd;

/// Test logaddexp basic f32 correctness
#[test]
fn test_logaddexp_simd_f32_basic() {
    let a = array![0.0_f32, 1.0, 2.0, -1.0];
    let b = array![0.0_f32, 1.0, 2.0, 1.0];

    let result = logaddexp_simd(&a.view(), &b.view());

    // log(exp(0) + exp(0)) = log(2)
    assert!((result[0] - 2.0_f32.ln()).abs() < 1e-5);
    // log(exp(1) + exp(1)) = log(2) + 1
    assert!((result[1] - (2.0_f32.ln() + 1.0)).abs() < 1e-5);
    // log(exp(2) + exp(2)) = log(2) + 2
    assert!((result[2] - (2.0_f32.ln() + 2.0)).abs() < 1e-5);
}

/// Test logaddexp basic f64 correctness
#[test]
fn test_logaddexp_simd_f64_basic() {
    let a = array![0.0_f64, 1.0, 2.0, -1.0];
    let b = array![0.0_f64, 1.0, 2.0, 1.0];

    let result = logaddexp_simd(&a.view(), &b.view());

    // log(exp(0) + exp(0)) = log(2)
    assert!((result[0] - 2.0_f64.ln()).abs() < 1e-14);
    // log(exp(1) + exp(1)) = log(2) + 1
    assert!((result[1] - (2.0_f64.ln() + 1.0)).abs() < 1e-14);
    // log(exp(2) + exp(2)) = log(2) + 2
    assert!((result[2] - (2.0_f64.ln() + 2.0)).abs() < 1e-14);
}

/// Test logaddexp empty array
#[test]
fn test_logaddexp_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = logaddexp_simd(&a.view(), &b.view());
    assert_eq!(result.len(), 0);
}

/// Test logaddexp large array (SIMD path)
#[test]
fn test_logaddexp_simd_large_array() {
    let n = 10000;
    // Use smaller values to avoid exp() overflow in the naive verification
    let a = Array1::from_vec((0..n).map(|i| i as f64 * 0.01).collect());
    let b = Array1::from_vec((0..n).map(|i| i as f64 * 0.01 + 0.5).collect());

    let result = logaddexp_simd(&a.view(), &b.view());
    assert_eq!(result.len(), n);

    // Check a few values (keeping values small enough for naive exp())
    for i in [0, 100, 500, 1000] {
        let expected = (a[i].exp() + b[i].exp()).ln();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "logaddexp[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }

    // For large indices, verify monotonicity and finiteness
    assert!(result[9999].is_finite());
    assert!(result[9999] > result[5000]);
}

/// Test logaddexp numerical stability with large values
#[test]
fn test_logaddexp_simd_large_values() {
    // Large positive values that would overflow with naive exp()
    let a = array![700.0_f64, 700.0, -700.0];
    let b = array![700.0_f64, 500.0, -500.0];

    let result = logaddexp_simd(&a.view(), &b.view());

    // All results should be finite
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "logaddexp should handle large values, got {} at index {}",
            v,
            i
        );
    }

    // log(exp(700) + exp(700)) = 700 + log(2)
    assert!((result[0] - (700.0 + 2.0_f64.ln())).abs() < 1e-10);
    // log(exp(700) + exp(500)) ≈ 700 (dominated by larger term)
    assert!((result[1] - 700.0).abs() < 1e-10);
}

/// Test logaddexp property: logaddexp(a, a) = a + log(2)
#[test]
fn test_logaddexp_simd_equal_inputs() {
    let a = array![0.0_f64, 1.0, -1.0, 10.0, -10.0];

    let result = logaddexp_simd(&a.view(), &a.view());

    let ln2 = 2.0_f64.ln();
    for i in 0..a.len() {
        let expected = a[i] + ln2;
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "logaddexp(a, a) should be a + log(2)"
        );
    }
}

/// Test logaddexp commutativity: logaddexp(a, b) = logaddexp(b, a)
#[test]
fn test_logaddexp_simd_commutativity() {
    let a = array![1.0_f64, 2.0, -3.0, 4.0];
    let b = array![5.0_f64, -2.0, 3.0, -4.0];

    let result_ab = logaddexp_simd(&a.view(), &b.view());
    let result_ba = logaddexp_simd(&b.view(), &a.view());

    for i in 0..a.len() {
        assert!(
            (result_ab[i] - result_ba[i]).abs() < 1e-14,
            "logaddexp should be commutative"
        );
    }
}

/// Test logaddexp log-probability use case
#[test]
fn test_logaddexp_simd_log_probability() {
    // In log-probability space, logaddexp combines probabilities
    // log(P(A or B)) = logaddexp(log(P(A)), log(P(B))) when A, B are mutually exclusive

    let log_p_a = array![-1.0_f64, -2.0, -3.0]; // P(A) = exp(-1), exp(-2), exp(-3)
    let log_p_b = array![-1.0_f64, -2.0, -3.0]; // P(B) = exp(-1), exp(-2), exp(-3)

    let log_p_union = logaddexp_simd(&log_p_a.view(), &log_p_b.view());

    // P(A or B) = 2 * P(A) when P(A) = P(B), so log(P) = log(2) + log(P(A))
    let ln2 = 2.0_f64.ln();
    for i in 0..log_p_a.len() {
        let expected = log_p_a[i] + ln2;
        assert!(
            (log_p_union[i] - expected).abs() < 1e-14,
            "Log-probability combination failed"
        );
    }
}

// ============================================================================
// Logit tests
// ============================================================================

/// Test logit basic f32 correctness
#[test]
fn test_logit_simd_f32_basic() {
    let p = array![0.5_f32, 0.1, 0.9, 0.01, 0.99];

    let result = logit_simd(&p.view());

    // logit(0.5) = log(1) = 0
    assert!((result[0] - 0.0).abs() < 1e-5);
    // logit(0.1) = log(0.1/0.9)
    assert!((result[1] - (0.1_f32 / 0.9).ln()).abs() < 1e-5);
    // logit(0.9) = log(0.9/0.1)
    assert!((result[2] - (0.9_f32 / 0.1).ln()).abs() < 1e-5);
}

/// Test logit basic f64 correctness
#[test]
fn test_logit_simd_f64_basic() {
    let p = array![0.5_f64, 0.1, 0.9, 0.01, 0.99];

    let result = logit_simd(&p.view());

    // logit(0.5) = log(1) = 0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // logit(0.1) = log(0.1/0.9)
    assert!((result[1] - (0.1_f64 / 0.9).ln()).abs() < 1e-14);
    // logit(0.9) = log(0.9/0.1)
    assert!((result[2] - (0.9_f64 / 0.1).ln()).abs() < 1e-14);
}

/// Test logit empty array
#[test]
fn test_logit_simd_empty() {
    let p: Array1<f64> = array![];
    let result = logit_simd(&p.view());
    assert_eq!(result.len(), 0);
}

/// Test logit large array (SIMD path)
#[test]
fn test_logit_simd_large_array() {
    let n = 10000;
    // Generate probabilities in (0.01, 0.99) range
    let p = Array1::from_vec(
        (0..n)
            .map(|i| 0.01 + 0.98 * (i as f64 / n as f64))
            .collect(),
    );

    let result = logit_simd(&p.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 1000, 5000, 9999] {
        let expected = (p[i] / (1.0 - p[i])).ln();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "logit[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test logit edge cases
#[test]
fn test_logit_simd_edge_cases() {
    let p = array![0.0_f64, 1.0];

    let result = logit_simd(&p.view());

    // logit(0) = -∞
    assert!(result[0].is_infinite() && result[0] < 0.0);
    // logit(1) = +∞
    assert!(result[1].is_infinite() && result[1] > 0.0);
}

/// Test logit symmetry: logit(p) = -logit(1-p)
#[test]
fn test_logit_simd_symmetry() {
    let p = array![0.1_f64, 0.2, 0.3, 0.4];
    let one_minus_p = p.mapv(|x| 1.0 - x);

    let result_p = logit_simd(&p.view());
    let result_1mp = logit_simd(&one_minus_p.view());

    for i in 0..p.len() {
        assert!(
            (result_p[i] + result_1mp[i]).abs() < 1e-14,
            "logit(p) + logit(1-p) should be 0"
        );
    }
}

/// Test logit as inverse of sigmoid
#[test]
fn test_logit_simd_inverse_of_sigmoid() {
    use scirs2_core::ndarray_ext::elementwise::sigmoid_simd;

    let p = array![0.1_f64, 0.3, 0.5, 0.7, 0.9];

    // logit(p) then sigmoid should return p
    let logit_p = logit_simd(&p.view());
    let sigmoid_logit_p = sigmoid_simd(&logit_p.view());

    for i in 0..p.len() {
        assert!(
            (sigmoid_logit_p[i] - p[i]).abs() < 1e-14,
            "sigmoid(logit(p)) should equal p"
        );
    }
}

/// Test logit logistic regression use case
#[test]
fn test_logit_simd_logistic_regression() {
    // In logistic regression, we model log-odds = β₀ + β₁x₁ + ...
    // Converting predicted probabilities to log-odds should give linear values

    // Simulating probabilities from a logistic model with linear log-odds
    let true_log_odds = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let probabilities = true_log_odds.mapv(|lo| 1.0 / (1.0 + (-lo).exp()));

    let recovered_log_odds = logit_simd(&probabilities.view());

    for i in 0..true_log_odds.len() {
        assert!(
            (recovered_log_odds[i] - true_log_odds[i]).abs() < 1e-14,
            "Logit should recover log-odds from probabilities"
        );
    }
}
