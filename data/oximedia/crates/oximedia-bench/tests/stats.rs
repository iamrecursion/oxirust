//! Tests for Welch's t-test and bootstrap confidence interval functions.

use oximedia_bench::stats::{bootstrap_ci, welch_t_test, StatsError};

// ── Welch's t-test ─────────────────────────────────────────────────────────────

#[test]
fn test_welch_identical_samples_t_near_zero() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = a.clone();
    let result = welch_t_test(&a, &b).expect("welch ok");
    assert!(
        result.t_statistic.abs() < 1e-9,
        "identical samples → t ≈ 0, got {}",
        result.t_statistic
    );
    // p-value should be 1.0 (or very close) when t = 0.
    assert!(
        result.p_value > 0.99,
        "t = 0 → p ≈ 1.0, got {}",
        result.p_value
    );
}

#[test]
fn test_welch_well_separated_samples_significant() {
    // Two clearly different distributions.
    let a: Vec<f64> = vec![1.0, 1.1, 0.9, 1.0, 1.05, 0.95];
    let b: Vec<f64> = vec![10.0, 10.1, 9.9, 10.0, 10.05, 9.95];
    let result = welch_t_test(&a, &b).expect("welch ok");
    assert!(
        result.t_statistic.abs() > 10.0,
        "well-separated means → |t| >> 1, got {}",
        result.t_statistic
    );
    assert!(
        result.p_value < 0.001,
        "well-separated → p < 0.001, got {}",
        result.p_value
    );
}

#[test]
fn test_welch_p_value_in_range() {
    let a = vec![5.0, 5.5, 4.8, 5.2, 5.0];
    let b = vec![5.5, 6.0, 5.3, 5.7, 5.6];
    let result = welch_t_test(&a, &b).expect("welch ok");
    assert!(
        result.p_value >= 0.0 && result.p_value <= 1.0,
        "p-value must be in [0, 1], got {}",
        result.p_value
    );
    assert!(
        result.degrees_of_freedom > 0.0,
        "df must be positive, got {}",
        result.degrees_of_freedom
    );
}

#[test]
fn test_welch_insufficient_data_returns_error() {
    let a = vec![1.0]; // too small
    let b = vec![2.0, 3.0];
    let result = welch_t_test(&a, &b);
    assert!(
        matches!(result, Err(StatsError::InsufficientData { got: 1 })),
        "single-element sample should return InsufficientData, got {result:?}"
    );
}

#[test]
fn test_welch_symmetric_t_statistic() {
    // t(a, b) = -t(b, a)
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let r1 = welch_t_test(&a, &b).expect("welch ok");
    let r2 = welch_t_test(&b, &a).expect("welch ok");
    assert!(
        (r1.t_statistic + r2.t_statistic).abs() < 1e-9,
        "t(a,b) should equal -t(b,a)"
    );
    assert!(
        (r1.p_value - r2.p_value).abs() < 1e-9,
        "p-value should be symmetric"
    );
}

// ── Bootstrap CI ───────────────────────────────────────────────────────────────

#[test]
fn test_bootstrap_ci_true_mean_covered_for_normals() {
    // Generate a pseudo-normal sample with known mean 5.0.
    // Use a fixed seed for reproducibility.
    let data: Vec<f64> = (0..200)
        .map(|i| 5.0 + (((i as f64 * 1.618) % 2.0) - 1.0))
        .collect();
    let ci = bootstrap_ci(&data, 0.95, 1000).expect("bootstrap ok");
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    assert!(
        ci.lower <= mean && mean <= ci.upper,
        "95% CI [{:.4}, {:.4}] should contain sample mean {:.4}",
        ci.lower,
        ci.upper,
        mean
    );
}

#[test]
fn test_bootstrap_ci_lower_lt_upper() {
    let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
    let ci = bootstrap_ci(&data, 0.95, 500).expect("bootstrap ok");
    assert!(ci.lower <= ci.upper, "lower must be ≤ upper");
    assert_eq!(ci.n_resamples, 500);
}

#[test]
fn test_bootstrap_ci_default_resamples() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ci = bootstrap_ci(&data, 0.95, 0).expect("bootstrap ok");
    assert_eq!(
        ci.n_resamples, 1000,
        "passing 0 should use default 1000 resamples"
    );
}

#[test]
fn test_bootstrap_ci_invalid_confidence() {
    let data = vec![1.0, 2.0, 3.0];
    let result = bootstrap_ci(&data, 1.0, 100);
    assert!(
        matches!(result, Err(StatsError::InvalidConfidence { .. })),
        "confidence = 1.0 should return InvalidConfidence, got {result:?}"
    );
    let result2 = bootstrap_ci(&data, 0.0, 100);
    assert!(
        matches!(result2, Err(StatsError::InvalidConfidence { .. })),
        "confidence = 0.0 should return InvalidConfidence, got {result2:?}"
    );
}

#[test]
fn test_bootstrap_ci_insufficient_data() {
    let data = vec![42.0];
    let result = bootstrap_ci(&data, 0.95, 100);
    assert!(
        matches!(result, Err(StatsError::InsufficientData { got: 1 })),
        "single element should return InsufficientData, got {result:?}"
    );
}

#[test]
fn test_bootstrap_ci_constant_data() {
    // All values identical → CI should degenerate to a single point.
    let data = vec![7.0; 50];
    let ci = bootstrap_ci(&data, 0.95, 200).expect("bootstrap ok");
    assert!(
        (ci.lower - 7.0).abs() < 1e-9 && (ci.upper - 7.0).abs() < 1e-9,
        "constant data → CI = [7, 7], got [{}, {}]",
        ci.lower,
        ci.upper
    );
}

// ── Cross-validation: Welch + bootstrap on same data ─────────────────────────

#[test]
fn test_welch_and_bootstrap_consistent() {
    let a: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64) * 0.1).collect();
    let b: Vec<f64> = (0..30).map(|i| 11.0 + (i as f64) * 0.1).collect();

    let welch = welch_t_test(&a, &b).expect("welch ok");
    let ci_a = bootstrap_ci(&a, 0.95, 1000).expect("bootstrap a ok");
    let ci_b = bootstrap_ci(&b, 0.95, 1000).expect("bootstrap b ok");

    // The means differ by 1.0; if Welch says p < 0.05, the CIs should not overlap.
    if welch.p_value < 0.05 {
        assert!(
            ci_b.lower > ci_a.upper || ci_a.lower > ci_b.upper,
            "significant Welch (p={:.4}) but CIs overlap: [{:.3},{:.3}] vs [{:.3},{:.3}]",
            welch.p_value,
            ci_a.lower,
            ci_a.upper,
            ci_b.lower,
            ci_b.upper
        );
    }
}
