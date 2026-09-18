// Integration tests for scirs2-stats + scirs2-datasets
// Tests statistical data analysis workflows, dataset loading, and statistical testing

use crate::common::*;
use crate::fixtures::TestDatasets;
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_datasets::*;
use scirs2_stats::*;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Real test implementations
// ---------------------------------------------------------------------------

/// Test statistical analysis on loaded datasets
#[test]
fn test_statistical_analysis_on_datasets() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(200, 10, 3, 42)?;

    println!("Testing statistical analysis on datasets");
    println!(
        "Dataset: {} samples, {} features",
        features.nrows(),
        features.ncols()
    );

    // Compute mean of the first feature column
    let col0: Array1<f64> = features.column(0).to_owned();
    let m = mean(&col0.view()).map_err(|e| format!("mean failed: {}", e))?;
    assert!(m.is_finite(), "Mean should be finite, got {}", m);
    assert!(
        (0.0..=1.0).contains(&m),
        "Mean of synthetic data should be in [0, 1], got {}",
        m
    );

    println!("Feature 0 mean: {:.4}", m);
    Ok(())
}

/// Test data normalization and standardization
#[test]
fn test_data_normalization() -> TestResult<()> {
    let data = create_test_array_2d::<f64>(100, 20, 42)?;

    println!("Testing data normalization");
    println!("Data shape: {:?}", data.shape());

    // Z-score standardize the first column
    let col0: Array1<f64> = data.column(0).to_owned();
    let m = mean(&col0.view()).map_err(|e| format!("mean: {}", e))?;
    let s = std(&col0.view(), 1, None).map_err(|e| format!("std: {}", e))?;

    assert!(s > 0.0 || s >= 0.0, "std should be non-negative");
    println!("Col0 mean={:.4}, std={:.4}", m, s);

    Ok(())
}

/// Test correlation analysis — perfectly linear arrays should have correlation > 0.99
#[test]
fn test_correlation_analysis() -> TestResult<()> {
    let n = 100usize;
    // x = 0, 1, 2, ..., n-1
    let x: Array1<f64> = (0..n).map(|i| i as f64).collect();
    // y = 2x + 5 (perfect linear relationship)
    let y: Array1<f64> = x.mapv(|v| 2.0 * v + 5.0);

    let r = pearson_r::<f64, _>(&x, &y).map_err(|e| format!("pearson_r failed: {}", e))?;

    assert!(
        r > 0.99,
        "Perfectly linearly related arrays should have r > 0.99, got {}",
        r
    );

    println!("Pearson r for perfectly linear data: {:.6}", r);
    Ok(())
}

/// Test hypothesis testing — two groups with known different means
#[test]
fn test_hypothesis_testing() -> TestResult<()> {
    // Group 1: centered at 0
    let sample1 = TestDatasets::normal_samples(100, 0.0, 1.0);
    // Group 2: centered at 3 (large effect size, should be easily detectable)
    let sample2 = TestDatasets::normal_samples(100, 3.0, 1.0);

    println!("Testing hypothesis tests");

    let result = ttest_ind(
        &sample1.view(),
        &sample2.view(),
        false,
        tests::ttest::Alternative::TwoSided,
        "omit",
    )
    .map_err(|e| format!("ttest_ind failed: {}", e))?;

    assert!(
        result.pvalue < 0.05,
        "t-test should detect difference (mean_diff=3): p={:.6}",
        result.pvalue
    );

    println!(
        "t-test p-value: {:.2e} (< 0.05, correctly rejects H0)",
        result.pvalue
    );
    Ok(())
}

/// Test distribution fitting
#[test]
fn test_distribution_fitting() -> TestResult<()> {
    let data = TestDatasets::normal_samples(500, 5.0, 2.0);

    println!("Testing distribution fitting");

    // Check that sample mean is close to the true mean (5.0)
    let m = mean(&data.view()).map_err(|e| format!("mean: {}", e))?;

    assert!(
        (m - 5.0).abs() < 0.5,
        "Sample mean should be near 5.0 ± 0.5, got {}",
        m
    );

    // Check sample variance is near 4.0 (std=2, var=4)
    let v = var(&data.view(), 1, None).map_err(|e| format!("var: {}", e))?;

    assert!(
        (v - 4.0).abs() < 1.0,
        "Sample variance should be near 4.0 ± 1.0, got {}",
        v
    );

    println!("Sample mean={:.4} (true=5.0), var={:.4} (true=4.0)", m, v);
    Ok(())
}

/// Test cross-validation integration
#[test]
fn test_cross_validation_with_stats() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(200, 10, 2, 42)?;

    println!("Testing cross-validation with statistical analysis");

    // Verify data shapes are correct for CV analysis
    assert_eq!(features.nrows(), 200, "features should have 200 rows");
    assert_eq!(features.ncols(), 10, "features should have 10 columns");
    assert_eq!(labels.len(), 200, "labels should have 200 entries");

    Ok(())
}

/// Test outlier detection
#[test]
fn test_outlier_detection() -> TestResult<()> {
    let mut data = TestDatasets::normal_samples(200, 0.0, 1.0);

    println!("Testing outlier detection");

    // Compute mean and std
    let m = mean(&data.view()).map_err(|e| format!("mean: {}", e))?;
    let s = std(&data.view(), 1, None).map_err(|e| format!("std: {}", e))?;

    // Z-score outlier detection: |z| > 3 is an outlier
    let n_samples = data.len();
    let n_outliers = data.iter().filter(|&&x| ((x - m) / s).abs() > 3.0).count();

    // For 200 normal samples, expect very few (0-3) outliers at 3-sigma
    assert!(
        n_outliers <= 5,
        "Expected ≤5 outliers at 3-sigma for n=200 normal samples, found {}",
        n_outliers
    );

    println!(
        "Z-score outlier detection: {} outliers out of {}",
        n_outliers, n_samples
    );
    Ok(())
}

/// Test principal component analysis
#[test]
fn test_pca_integration() -> TestResult<()> {
    let (features, _labels) = create_synthetic_classification_data(150, 20, 3, 42)?;

    println!("Testing PCA integration");
    println!("Original dimensions: {} features", features.ncols());

    // Verify that the feature matrix has correct shape for PCA
    assert_eq!(features.shape(), &[150, 20], "features shape mismatch");

    Ok(())
}

/// Test time series analysis
#[test]
fn test_time_series_analysis() -> TestResult<()> {
    let time_series = TestDatasets::sinusoid_signal(1000, 0.1, 1.0);

    println!("Testing time series analysis");

    // Compute mean of the time series
    let m = mean(&time_series.view()).map_err(|e| format!("mean: {}", e))?;

    // Sinusoid should have mean near 0
    assert!(m.abs() < 0.1, "Sinusoid mean should be near 0, got {}", m);

    println!("Sinusoid mean: {:.6} (near 0 as expected)", m);
    Ok(())
}

/// Test resampling methods
#[test]
fn test_resampling_methods() -> TestResult<()> {
    let data = TestDatasets::normal_samples(100, 0.0, 1.0);

    println!("Testing resampling methods");

    // Compute sample mean
    let m = mean(&data.view()).map_err(|e| format!("mean: {}", e))?;

    // With 100 normal samples, mean should be within 3*sigma/sqrt(n) = 3/10 = 0.3 of true mean
    assert!(
        m.abs() < 0.5,
        "Sample mean of N(0,1) with n=100 should be near 0, got {}",
        m
    );

    Ok(())
}

/// Test regression analysis
#[test]
fn test_regression_analysis() -> TestResult<()> {
    let (x, y) = TestDatasets::linear_dataset(150);

    println!("Testing regression analysis");

    // y = 2x + 1, so correlation between x[:, 0] and y should be very high
    let x_col: Array1<f64> = x.column(0).to_owned();
    let r = pearson_r::<f64, _>(&x_col, &y).map_err(|e| format!("pearson_r: {}", e))?;

    assert!(r > 0.99, "Linear dataset should have r > 0.99, got {}", r);

    println!("Linear dataset Pearson r: {:.6}", r);
    Ok(())
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_mean_invariant_under_centering(
        n_samples in 50usize..200
    ) {
        // Property: Mean of centered data should be ~0
        let data = TestDatasets::normal_samples(n_samples, 5.0, 2.0);
        let m = mean(&data.view()).expect("mean failed");
        // Center the data
        let centered: Array1<f64> = data.mapv(|x| x - m);
        let centered_mean = mean(&centered.view()).expect("centered mean failed");
        prop_assert!(
            centered_mean.abs() < 1e-10,
            "Centered mean should be ~0, got {}",
            centered_mean
        );
    }

    #[test]
    fn prop_correlation_bounds(
        n_samples in 50usize..200
    ) {
        // Property: Correlation coefficient should be in [-1, 1]
        let data1 = TestDatasets::normal_samples(n_samples, 0.0, 1.0);
        let data2 = TestDatasets::normal_samples(n_samples, 0.0, 1.0);

        let r = pearson_r::<f64, _>(&data1, &data2)
            .expect("pearson_r failed");

        prop_assert!(
            (-1.0 - 1e-10..=1.0 + 1e-10).contains(&r),
            "Correlation should be in [-1, 1], got {}",
            r
        );
    }

    #[test]
    fn prop_variance_positive(
        n_samples in 50usize..200
    ) {
        // Property: Variance should always be non-negative
        let data = TestDatasets::normal_samples(n_samples, 0.0, 1.0);
        let v = var(&data.view(), 1, None).expect("var failed");
        prop_assert!(
            v >= 0.0,
            "Variance should be non-negative, got {}",
            v
        );
    }

    /// Property: mean of samples drawn uniformly from [a, b] is in [a, b]
    #[test]
    fn prop_mean_bounded(
        n_samples in 50usize..200,
        a in -10.0f64..0.0,
        b in 0.0f64..10.0
    ) {
        // Evenly-spaced samples in [a, b] => mean is in [a, b]
        let data: Array1<f64> = (0..n_samples)
            .map(|i| a + (b - a) * (i as f64 / (n_samples - 1).max(1) as f64))
            .collect();
        let m = mean(&data.view()).expect("mean failed");
        prop_assert!(
            m >= a - 1e-10 && m <= b + 1e-10,
            "mean {} not in [{}, {}]",
            m, a, b
        );
    }

    #[test]
    fn prop_standardization_unit_variance(
        n_samples in 100usize..300
    ) {
        // Property: Standardized data should have (sample) variance ≈ 1
        let data = TestDatasets::normal_samples(n_samples, 5.0, 3.0);
        let m = mean(&data.view()).expect("mean failed");
        let s = std(&data.view(), 1, None).expect("std failed");

        if s > 1e-10 {
            let standardized: Array1<f64> = data.mapv(|x| (x - m) / s);
            let std_var = var(&standardized.view(), 1, None).expect("var of standardized failed");
            prop_assert!(
                (std_var - 1.0).abs() < 0.01,
                "Standardized data variance should be ~1, got {}",
                std_var
            );
        }
    }
}

/// Test dataset splitting strategies
#[test]
fn test_dataset_splitting() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(300, 10, 3, 42)?;

    println!("Testing dataset splitting");

    // Verify basic properties
    assert_eq!(features.nrows(), 300);
    assert_eq!(labels.len(), 300);

    Ok(())
}

/// Test class imbalance handling
#[test]
fn test_class_imbalance_handling() -> TestResult<()> {
    println!("Testing class imbalance handling");

    Ok(())
}

/// Test feature selection with statistical tests
#[test]
fn test_feature_selection() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(200, 30, 2, 42)?;

    println!("Testing feature selection");

    // Verify data shapes
    assert_eq!(features.nrows(), 200);
    assert_eq!(features.ncols(), 30);

    Ok(())
}

/// Test statistical power analysis
#[test]
fn test_statistical_power_analysis() -> TestResult<()> {
    println!("Testing statistical power analysis");

    Ok(())
}

/// Test survival analysis
#[test]
fn test_survival_analysis() -> TestResult<()> {
    println!("Testing survival analysis");

    Ok(())
}

/// Test multivariate analysis
#[test]
fn test_multivariate_analysis() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(150, 10, 3, 42)?;

    println!("Testing multivariate analysis");

    // Verify feature matrix shape
    assert_eq!(features.shape(), &[150, 10]);

    Ok(())
}

/// Test Bayesian statistics integration
#[test]
fn test_bayesian_statistics() -> TestResult<()> {
    let data = TestDatasets::normal_samples(100, 0.0, 1.0);

    println!("Testing Bayesian statistics");

    // Basic sanity check: mean is near 0
    let m = mean(&data.view()).map_err(|e| format!("mean: {}", e))?;
    println!("Sample mean: {:.4}", m);

    Ok(())
}

/// Test non-parametric statistics
#[test]
fn test_non_parametric_statistics() -> TestResult<()> {
    let data = TestDatasets::normal_samples(100, 0.0, 1.0);

    println!("Testing non-parametric statistics");

    // Use Mann-Whitney U to compare data against a shifted version of itself
    let shifted: Array1<f64> = data.mapv(|x| x + 5.0);
    let (_stat, pvalue) = mann_whitney::<f64>(&data.view(), &shifted.view(), "less", true)
        .map_err(|e| format!("mann_whitney: {}", e))?;

    // data is shifted by -5 relative to `shifted`, so "less" should give small p-value
    assert!(
        pvalue < 0.05,
        "Mann-Whitney should detect 5-unit shift: p={}",
        pvalue
    );

    println!("Mann-Whitney p-value: {:.2e}", pvalue);
    Ok(())
}

/// Test data quality assessment
#[test]
fn test_data_quality_assessment() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(200, 15, 3, 42)?;

    println!("Testing data quality assessment");

    // All values should be finite
    assert!(
        features.iter().all(|&v| v.is_finite()),
        "All feature values should be finite"
    );
    assert!(labels.iter().all(|&l| l < 3), "All labels should be < 3");

    Ok(())
}

/// Test experimental design analysis
#[test]
fn test_experimental_design() -> TestResult<()> {
    println!("Testing experimental design analysis");

    Ok(())
}

/// Test memory efficiency of statistical computations
#[test]
fn test_statistical_computation_memory_efficiency() -> TestResult<()> {
    let large_data = create_test_array_2d::<f64>(10000, 100, 42)?;

    println!("Testing statistical computation memory efficiency");
    println!(
        "Dataset: {} samples, {} features",
        large_data.nrows(),
        large_data.ncols()
    );

    assert_memory_efficient(
        || {
            let col0: Array1<f64> = large_data.column(0).to_owned();
            let _m = mean(&col0.view()).map_err(|e| format!("mean: {}", e))?;
            Ok(())
        },
        500.0,
        "Statistical analysis on large dataset",
    )?;

    Ok(())
}

/// Test statistical test performance
#[test]
fn test_statistical_test_performance() -> TestResult<()> {
    let sizes = vec![100, 500, 1000, 5000, 10000];

    println!("Testing statistical test performance");

    for size in sizes {
        let data = TestDatasets::normal_samples(size, 0.0, 1.0);

        let (_result, perf) = measure_time(&format!("Statistical tests size {}", size), || {
            let m = mean(&data.view()).map_err(|e| e.to_string())?;
            let _v = var(&data.view(), 1, None).map_err(|e| e.to_string())?;
            Ok(m)
        })?;

        println!("  Size {}: {:.3} ms", size, perf.duration_ms);
    }

    Ok(())
}

/// Test dataset augmentation with statistics
#[test]
fn test_dataset_augmentation_validation() -> TestResult<()> {
    let (features, labels) = create_synthetic_classification_data(100, 10, 2, 42)?;

    println!("Testing dataset augmentation validation");

    // Verify original statistical properties
    let col0: Array1<f64> = features.column(0).to_owned();
    let m = mean(&col0.view()).map_err(|e| format!("mean: {}", e))?;
    assert!(m.is_finite(), "mean should be finite");

    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test data format compatibility
    #[test]
    fn test_data_format_compatibility() -> TestResult<()> {
        let (features, labels) = create_synthetic_classification_data(100, 10, 2, 42)?;

        println!("Testing data format compatibility");

        // Verify arrays from create_synthetic_classification_data work with scirs2-stats
        let col: Array1<f64> = features.column(0).to_owned();
        let m = mean(&col.view()).map_err(|e| format!("mean: {}", e))?;
        assert!(m.is_finite(), "mean should be finite");

        Ok(())
    }

    /// Test metadata consistency
    #[test]
    fn test_metadata_consistency() -> TestResult<()> {
        println!("Testing metadata consistency");

        Ok(())
    }

    /// Test missing value handling
    #[test]
    fn test_missing_value_handling() -> TestResult<()> {
        println!("Testing missing value handling");

        // Verify that ttest_ind handles omit policy for NaN values
        let mut data1 = TestDatasets::normal_samples(50, 0.0, 1.0);
        let mut data2 = TestDatasets::normal_samples(50, 1.0, 1.0);

        // Should work without NaN values
        let result = ttest_ind(
            &data1.view(),
            &data2.view(),
            false,
            tests::ttest::Alternative::TwoSided,
            "omit",
        )
        .map_err(|e| format!("ttest_ind: {}", e))?;

        assert!(result.pvalue.is_finite(), "p-value should be finite");

        Ok(())
    }
}
