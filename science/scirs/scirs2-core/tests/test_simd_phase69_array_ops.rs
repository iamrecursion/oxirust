//! Phase 69: SIMD array operations tests (clip, cumsum, cumprod, diff)

use scirs2_core::ndarray::{array, Array1};

// ============================================================================
// Phase 69: SIMD clip, cumsum, cumprod, diff tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::clip_simd;
use scirs2_core::ndarray_ext::elementwise::cumprod_simd;
use scirs2_core::ndarray_ext::elementwise::cumsum_simd;
use scirs2_core::ndarray_ext::elementwise::diff_simd;

// ============================================================================
// clip tests
// ============================================================================

/// Test clip basic f32 correctness
#[test]
fn test_clip_simd_f32_basic() {
    let x = array![-2.0_f32, -1.0, 0.0, 1.0, 2.0, 3.0];

    let result = clip_simd(&x.view(), 0.0, 1.0);

    assert!((result[0] - 0.0).abs() < 1e-6); // -2 clipped to 0
    assert!((result[1] - 0.0).abs() < 1e-6); // -1 clipped to 0
    assert!((result[2] - 0.0).abs() < 1e-6); // 0 unchanged
    assert!((result[3] - 1.0).abs() < 1e-6); // 1 unchanged
    assert!((result[4] - 1.0).abs() < 1e-6); // 2 clipped to 1
    assert!((result[5] - 1.0).abs() < 1e-6); // 3 clipped to 1
}

/// Test clip basic f64 correctness
#[test]
fn test_clip_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0];

    let result = clip_simd(&x.view(), 0.0, 1.0);

    assert!((result[0] - 0.0).abs() < 1e-14); // -2 clipped to 0
    assert!((result[1] - 0.0).abs() < 1e-14); // -1 clipped to 0
    assert!((result[2] - 0.0).abs() < 1e-14); // 0 unchanged
    assert!((result[3] - 1.0).abs() < 1e-14); // 1 unchanged
    assert!((result[4] - 1.0).abs() < 1e-14); // 2 clipped to 1
    assert!((result[5] - 1.0).abs() < 1e-14); // 3 clipped to 1
}

/// Test clip empty array
#[test]
fn test_clip_simd_empty() {
    let x: Array1<f64> = array![];
    let result = clip_simd(&x.view(), 0.0, 1.0);
    assert_eq!(result.len(), 0);
}

/// Test clip large array (SIMD path)
#[test]
fn test_clip_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| (i as f64 - 5000.0) * 0.001).collect());

    let result = clip_simd(&x.view(), -1.0, 1.0);
    assert_eq!(result.len(), n);

    // All values should be in [-1, 1]
    for &v in result.iter() {
        assert!((-1.0..=1.0).contains(&v), "Value {} out of clip range", v);
    }
}

/// Test clip gradient clipping use case
#[test]
fn test_clip_simd_gradient_clipping() {
    // In neural networks, gradients are often clipped to prevent exploding gradients
    let gradients = array![100.0_f64, -50.0, 0.5, -0.1, 200.0];

    let clipped = clip_simd(&gradients.view(), -1.0, 1.0);

    assert!((clipped[0] - 1.0).abs() < 1e-14); // 100 clipped to 1
    assert!((clipped[1] - (-1.0)).abs() < 1e-14); // -50 clipped to -1
    assert!((clipped[2] - 0.5).abs() < 1e-14); // 0.5 unchanged
    assert!((clipped[3] - (-0.1)).abs() < 1e-14); // -0.1 unchanged
    assert!((clipped[4] - 1.0).abs() < 1e-14); // 200 clipped to 1
}

// ============================================================================
// cumsum tests
// ============================================================================

/// Test cumsum basic f32 correctness
#[test]
fn test_cumsum_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];

    let result = cumsum_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-5); // 1
    assert!((result[1] - 3.0).abs() < 1e-5); // 1+2
    assert!((result[2] - 6.0).abs() < 1e-5); // 1+2+3
    assert!((result[3] - 10.0).abs() < 1e-5); // 1+2+3+4
    assert!((result[4] - 15.0).abs() < 1e-5); // 1+2+3+4+5
}

/// Test cumsum basic f64 correctness
#[test]
fn test_cumsum_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let result = cumsum_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1
    assert!((result[1] - 3.0).abs() < 1e-14); // 1+2
    assert!((result[2] - 6.0).abs() < 1e-14); // 1+2+3
    assert!((result[3] - 10.0).abs() < 1e-14); // 1+2+3+4
    assert!((result[4] - 15.0).abs() < 1e-14); // 1+2+3+4+5
}

/// Test cumsum empty array
#[test]
fn test_cumsum_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cumsum_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test cumsum large array (SIMD path)
#[test]
fn test_cumsum_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((1..=n).map(|i| i as f64).collect());

    let result = cumsum_simd(&x.view());
    assert_eq!(result.len(), n);

    // Last element should be n*(n+1)/2 = 50005000
    let expected_last = (n * (n + 1) / 2) as f64;
    assert!(
        (result[n - 1] - expected_last).abs() < 1e-6,
        "cumsum last element should be {}, got {}",
        expected_last,
        result[n - 1]
    );
}

/// Test cumsum CDF computation use case
#[test]
fn test_cumsum_simd_cdf() {
    // Computing CDF from PDF
    let pdf = array![0.1_f64, 0.2, 0.3, 0.25, 0.15]; // Should sum to 1

    let cdf = cumsum_simd(&pdf.view());

    assert!((cdf[0] - 0.1).abs() < 1e-14);
    assert!((cdf[1] - 0.3).abs() < 1e-14);
    assert!((cdf[2] - 0.6).abs() < 1e-14);
    assert!((cdf[3] - 0.85).abs() < 1e-14);
    assert!((cdf[4] - 1.0).abs() < 1e-14); // CDF should end at 1
}

// ============================================================================
// cumprod tests
// ============================================================================

/// Test cumprod basic f32 correctness
#[test]
fn test_cumprod_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];

    let result = cumprod_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-5); // 1
    assert!((result[1] - 2.0).abs() < 1e-5); // 1*2
    assert!((result[2] - 6.0).abs() < 1e-5); // 1*2*3
    assert!((result[3] - 24.0).abs() < 1e-5); // 1*2*3*4 = 4!
}

/// Test cumprod basic f64 correctness
#[test]
fn test_cumprod_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let result = cumprod_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1
    assert!((result[1] - 2.0).abs() < 1e-14); // 1*2
    assert!((result[2] - 6.0).abs() < 1e-14); // 1*2*3
    assert!((result[3] - 24.0).abs() < 1e-14); // 1*2*3*4 = 4!
    assert!((result[4] - 120.0).abs() < 1e-14); // 1*2*3*4*5 = 5!
}

/// Test cumprod empty array
#[test]
fn test_cumprod_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cumprod_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test cumprod factorial computation
#[test]
fn test_cumprod_simd_factorial() {
    // Computing factorials: cumprod([1,2,3,...,n]) = [1!, 2!, 3!, ..., n!]
    let n = 10;
    let seq = Array1::from_vec((1..=n).map(|i| i as f64).collect());

    let factorials = cumprod_simd(&seq.view());

    // Known factorials
    let expected = [
        1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0,
    ];

    for i in 0..n {
        assert!(
            (factorials[i] - expected[i]).abs() < 1e-10,
            "{}! should be {}, got {}",
            i + 1,
            expected[i],
            factorials[i]
        );
    }
}

/// Test cumprod survival probability use case
#[test]
fn test_cumprod_simd_survival() {
    // Computing survival probability from hazard rates
    // Survival at time t = product of (1 - hazard_i) for i < t
    let survival_probs = array![0.99_f64, 0.98, 0.97, 0.96, 0.95];

    let cumulative_survival = cumprod_simd(&survival_probs.view());

    // Cumulative survival should be decreasing
    for i in 1..survival_probs.len() {
        assert!(
            cumulative_survival[i] < cumulative_survival[i - 1],
            "Cumulative survival should be decreasing"
        );
    }

    // Last value should be product of all survival probabilities
    let expected = 0.99 * 0.98 * 0.97 * 0.96 * 0.95;
    assert!(
        (cumulative_survival[4] - expected).abs() < 1e-14,
        "Final survival should be {}",
        expected
    );
}

// ============================================================================
// diff tests
// ============================================================================

/// Test diff basic f32 correctness
#[test]
fn test_diff_simd_f32_basic() {
    let x = array![1.0_f32, 3.0, 6.0, 10.0, 15.0];

    let result = diff_simd(&x.view());
    assert_eq!(result.len(), 4); // n-1 elements

    assert!((result[0] - 2.0).abs() < 1e-5); // 3-1
    assert!((result[1] - 3.0).abs() < 1e-5); // 6-3
    assert!((result[2] - 4.0).abs() < 1e-5); // 10-6
    assert!((result[3] - 5.0).abs() < 1e-5); // 15-10
}

/// Test diff basic f64 correctness
#[test]
fn test_diff_simd_f64_basic() {
    let x = array![1.0_f64, 3.0, 6.0, 10.0, 15.0];

    let result = diff_simd(&x.view());
    assert_eq!(result.len(), 4); // n-1 elements

    assert!((result[0] - 2.0).abs() < 1e-14); // 3-1
    assert!((result[1] - 3.0).abs() < 1e-14); // 6-3
    assert!((result[2] - 4.0).abs() < 1e-14); // 10-6
    assert!((result[3] - 5.0).abs() < 1e-14); // 15-10
}

/// Test diff empty and single element arrays
#[test]
fn test_diff_simd_edge_cases() {
    let x_empty: Array1<f64> = array![];
    let result_empty = diff_simd(&x_empty.view());
    assert_eq!(result_empty.len(), 0);

    let x_single = array![1.0_f64];
    let result_single = diff_simd(&x_single.view());
    assert_eq!(result_single.len(), 0); // Need at least 2 elements
}

/// Test diff large array (SIMD path)
#[test]
fn test_diff_simd_large_array() {
    let n = 10000;
    // Create quadratic sequence: x[i] = i^2
    let x = Array1::from_vec((0..n).map(|i| (i as f64).powi(2)).collect());

    let result = diff_simd(&x.view());
    assert_eq!(result.len(), n - 1);

    // diff of x^2 gives 2x+1 = 1, 3, 5, 7, ... (odd numbers)
    for i in 0..result.len() {
        let expected = (2 * i + 1) as f64;
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "diff[{}] should be {}, got {}",
            i,
            expected,
            result[i]
        );
    }
}

/// Test diff constant array gives zeros
#[test]
fn test_diff_simd_constant() {
    let x = array![5.0_f64, 5.0, 5.0, 5.0, 5.0];

    let result = diff_simd(&x.view());

    for &v in result.iter() {
        assert!((v - 0.0).abs() < 1e-14, "diff of constant should be 0");
    }
}

/// Test diff numerical differentiation use case
#[test]
fn test_diff_simd_numerical_derivative() {
    // Approximate derivative of sin(x) should be cos(x)
    let n = 100;
    let h = 0.01_f64;
    let x = Array1::from_vec((0..n).map(|i| i as f64 * h).collect());
    let sin_x = x.mapv(|xi| xi.sin());

    let diff_sin = diff_simd(&sin_x.view());

    // diff[i] / h ≈ cos(x[i])
    for i in 0..diff_sin.len() {
        let numerical_derivative = diff_sin[i] / h;
        let analytical_derivative = x[i].cos();
        // Should be close (within O(h) error)
        assert!(
            (numerical_derivative - analytical_derivative).abs() < 0.02,
            "Numerical derivative should approximate cos(x)"
        );
    }
}

/// Test diff and cumsum are inverse operations
#[test]
fn test_diff_simd_inverse_of_cumsum() {
    // For a sequence starting from 0: diff(cumsum(x)) = x (except for first element)
    // cumsum(diff(x)) = x - x[0] (offset by first element)

    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let cumsum_x = cumsum_simd(&x.view());
    let diff_cumsum_x = diff_simd(&cumsum_x.view());

    // diff(cumsum(x))[i] should equal x[i+1]
    for i in 0..diff_cumsum_x.len() {
        assert!(
            (diff_cumsum_x[i] - x[i + 1]).abs() < 1e-14,
            "diff(cumsum(x))[{}] should equal x[{}]",
            i,
            i + 1
        );
    }
}

// =============================================================================
