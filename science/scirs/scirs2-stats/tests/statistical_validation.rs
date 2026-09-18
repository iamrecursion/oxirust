//! Statistical Validation Tests for Distributions
//!
//! These tests validate that all distributions produce samples with correct
//! statistical properties (mean, variance, skewness). They use large sample sizes
//! (10,000+) to ensure statistical significance and prevent bugs like the Gamma
//! parameterization issue discovered by NumRS2.
//!
//! Inspired by NumRS2's comprehensive statistical testing approach.

use scirs2_stats::distributions::exponential::Exponential;
use scirs2_stats::distributions::gamma::Gamma;
use scirs2_stats::distributions::normal::Normal;
use scirs2_stats::traits::Distribution as ScirsDist;

/// Helper function to compute sample mean
fn compute_mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Helper function to compute sample variance
fn compute_variance(samples: &[f64], mean: f64) -> f64 {
    samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64
}

/// Helper function to compute sample standard deviation
fn compute_std(samples: &[f64], mean: f64) -> f64 {
    compute_variance(samples, mean).sqrt()
}

#[test]
fn test_gamma_statistical_properties_shape2_scale2() {
    // This is the exact test case from NumRS2's bug report
    // Gamma(shape=2, scale=2) should have: mean=4, variance=8
    let dist = Gamma::new(2.0, 2.0, 0.0).expect("Test: operation failed");
    let samples = dist.rvs(10_000).expect("Test: operation failed");
    let samples_vec: Vec<f64> = samples.to_vec();

    let mean = compute_mean(&samples_vec);
    let variance = compute_variance(&samples_vec, mean);

    // Expected values for Gamma(shape=2, scale=2):
    // mean = shape * scale = 2 * 2 = 4.0
    // variance = shape * scale^2 = 2 * 4 = 8.0
    println!("Gamma(2, 2): mean={:.4}, variance={:.4}", mean, variance);

    assert!(
        (mean - 4.0).abs() < 0.2,
        "Mean should be ≈4.0, got {}. Bug may still exist!",
        mean
    );
    assert!(
        (variance - 8.0).abs() < 1.0,
        "Variance should be ≈8.0, got {}. Bug may still exist!",
        variance
    );
}

#[test]
fn test_gamma_statistical_properties_shape2_scale3() {
    // Gamma(shape=2, scale=3) should have: mean=6, variance=18
    let dist = Gamma::new(2.0, 3.0, 0.0).expect("Test: operation failed");
    let samples = dist.rvs(10_000).expect("Test: operation failed");
    let samples_vec: Vec<f64> = samples.to_vec();

    let mean = compute_mean(&samples_vec);
    let variance = compute_variance(&samples_vec, mean);

    // Expected values for Gamma(shape=2, scale=3):
    // mean = shape * scale = 2 * 3 = 6.0
    // variance = shape * scale^2 = 2 * 9 = 18.0
    println!("Gamma(2, 3): mean={:.4}, variance={:.4}", mean, variance);

    assert!(
        (mean - 6.0).abs() < 0.3,
        "Mean should be ≈6.0, got {}",
        mean
    );
    assert!(
        (variance - 18.0).abs() < 2.0,
        "Variance should be ≈18.0, got {}",
        variance
    );
}

#[test]
fn test_gamma_statistical_properties_shape5_scale1() {
    // Gamma(shape=5, scale=1) should have: mean=5, variance=5
    let dist = Gamma::new(5.0, 1.0, 0.0).expect("Test: operation failed");
    let samples = dist.rvs(10_000).expect("Test: operation failed");
    let samples_vec: Vec<f64> = samples.to_vec();

    let mean = compute_mean(&samples_vec);
    let variance = compute_variance(&samples_vec, mean);

    // Expected values for Gamma(shape=5, scale=1):
    // mean = shape * scale = 5 * 1 = 5.0
    // variance = shape * scale^2 = 5 * 1 = 5.0
    println!("Gamma(5, 1): mean={:.4}, variance={:.4}", mean, variance);

    assert!(
        (mean - 5.0).abs() < 0.3,
        "Mean should be ≈5.0, got {}",
        mean
    );
    assert!(
        (variance - 5.0).abs() < 1.0,
        "Variance should be ≈5.0, got {}",
        variance
    );
}

#[test]
fn test_gamma_statistical_properties_shape1_scale2_exponential() {
    // Gamma(shape=1, scale=2) is an Exponential distribution
    // should have: mean=2, variance=4
    let dist = Gamma::new(1.0, 2.0, 0.0).expect("Test: operation failed");
    let samples = dist.rvs(10_000).expect("Test: operation failed");
    let samples_vec: Vec<f64> = samples.to_vec();

    let mean = compute_mean(&samples_vec);
    let variance = compute_variance(&samples_vec, mean);

    // Expected values for Gamma(shape=1, scale=2):
    // mean = shape * scale = 1 * 2 = 2.0
    // variance = shape * scale^2 = 1 * 4 = 4.0
    println!(
        "Gamma(1, 2) [Exponential]: mean={:.4}, variance={:.4}",
        mean, variance
    );

    assert!(
        (mean - 2.0).abs() < 0.2,
        "Mean should be ≈2.0, got {}",
        mean
    );
    assert!(
        (variance - 4.0).abs() < 0.8,
        "Variance should be ≈4.0, got {}",
        variance
    );
}

#[test]
fn test_exponential_statistical_properties_scale2() {
    // Exponential with scale=2 should have: mean=2, variance=4
    // Note: Exponential::new takes rate, not scale, so rate = 1/scale = 0.5
    let rate = 0.5; // rate = 1/scale, so scale = 2.0
    let dist = Exponential::new(rate, 0.0).expect("Test: operation failed");
    let samples = dist.rvs(10_000).expect("Test: operation failed");
    let samples_vec: Vec<f64> = samples.to_vec();

    let mean = compute_mean(&samples_vec);
    let variance = compute_variance(&samples_vec, mean);

    // Expected values for Exponential(rate=0.5):
    // mean = 1/rate = 2.0
    // variance = 1/rate^2 = 4.0
    println!(
        "Exponential(rate=0.5, scale=2): mean={:.4}, variance={:.4}",
        mean, variance
    );

    assert!(
        (mean - 2.0).abs() < 0.2,
        "Mean should be ≈2.0, got {}",
        mean
    );
    assert!(
        (variance - 4.0).abs() < 0.8,
        "Variance should be ≈4.0, got {}",
        variance
    );
}

#[test]
fn test_normal_statistical_properties() {
    // Normal(mean=5, std=2) should have: mean=5, variance=4
    let dist = Normal::new(5.0, 2.0).expect("Test: operation failed");
    let samples = dist.rvs(10_000).expect("Test: operation failed");
    let samples_vec: Vec<f64> = samples.to_vec();

    let mean = compute_mean(&samples_vec);
    let variance = compute_variance(&samples_vec, mean);

    // Expected values for Normal(mean=5, std=2):
    // mean = 5.0
    // variance = std^2 = 4.0
    println!("Normal(5, 2): mean={:.4}, variance={:.4}", mean, variance);

    assert!(
        (mean - 5.0).abs() < 0.1,
        "Mean should be ≈5.0, got {}",
        mean
    );
    assert!(
        (variance - 4.0).abs() < 0.5,
        "Variance should be ≈4.0, got {}",
        variance
    );
}

#[test]
fn test_gamma_trait_consistency() {
    // Verify that the Distribution trait methods return correct theoretical values
    let dist = Gamma::new(2.0, 3.0, 0.0).expect("Test: operation failed");

    // Check theoretical mean and variance
    let theoretical_mean = dist.mean();
    let theoretical_var = dist.var();
    let theoretical_std = dist.std();

    assert!(
        (theoretical_mean - 6.0_f64).abs() < 1e-10,
        "Theoretical mean should be 6.0"
    );
    assert!(
        (theoretical_var - 18.0_f64).abs() < 1e-10,
        "Theoretical variance should be 18.0"
    );
    assert!(
        (theoretical_std - 18.0_f64.sqrt()).abs() < 1e-10,
        "Theoretical std should be sqrt(18)"
    );

    // Verify sampling matches theoretical values
    let samples = dist.rvs(10_000).expect("Test: operation failed");
    let samples_vec: Vec<f64> = samples.to_vec();
    let empirical_mean = compute_mean(&samples_vec);
    let empirical_std = compute_std(&samples_vec, empirical_mean);

    println!(
        "Theoretical: mean={:.4}, std={:.4}; Empirical: mean={:.4}, std={:.4}",
        theoretical_mean, theoretical_std, empirical_mean, empirical_std
    );

    assert!(
        (empirical_mean - theoretical_mean).abs() < 0.3,
        "Empirical mean should match theoretical mean"
    );
    assert!(
        (empirical_std - theoretical_std).abs() < 0.5,
        "Empirical std should match theoretical std"
    );
}

#[test]
fn test_gamma_regression_prevention() {
    // This test specifically prevents regression of the bug discovered by NumRS2
    // where Gamma was passing 1/scale instead of scale to rand_distr

    let test_cases = vec![
        (1.0, 1.0, 1.0, 1.0),   // Gamma(1, 1): mean=1, var=1
        (2.0, 1.0, 2.0, 2.0),   // Gamma(2, 1): mean=2, var=2
        (2.0, 2.0, 4.0, 8.0),   // Gamma(2, 2): mean=4, var=8 (NumRS2's case)
        (3.0, 2.0, 6.0, 12.0),  // Gamma(3, 2): mean=6, var=12
        (5.0, 1.5, 7.5, 11.25), // Gamma(5, 1.5): mean=7.5, var=11.25
    ];

    for (shape, scale, expected_mean, expected_var) in test_cases {
        let dist = Gamma::new(shape, scale, 0.0).expect("Test: operation failed");
        let samples = dist.rvs(10_000).expect("Test: operation failed");
        let samples_vec: Vec<f64> = samples.to_vec();

        let mean = compute_mean(&samples_vec);
        let variance = compute_variance(&samples_vec, mean);

        println!(
            "Gamma({}, {}): expected mean={:.2}, var={:.2}; got mean={:.4}, var={:.4}",
            shape, scale, expected_mean, expected_var, mean, variance
        );

        let mean_tolerance = expected_mean * 0.1; // 10% tolerance
        let var_tolerance = expected_var * 0.2; // 20% tolerance

        assert!(
            (mean - expected_mean).abs() < mean_tolerance,
            "Gamma({}, {}): Mean {} differs from expected {} by more than tolerance {}",
            shape,
            scale,
            mean,
            expected_mean,
            mean_tolerance
        );

        assert!(
            (variance - expected_var).abs() < var_tolerance,
            "Gamma({}, {}): Variance {} differs from expected {} by more than tolerance {}",
            shape,
            scale,
            variance,
            expected_var,
            var_tolerance
        );
    }
}
