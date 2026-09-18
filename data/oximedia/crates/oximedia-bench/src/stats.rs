//! Statistical analysis utilities for benchmark results.

use crate::SequenceResult;
use serde::{Deserialize, Serialize};

/// Statistical analysis of benchmark results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Statistics {
    /// Mean encoding FPS
    pub mean_encoding_fps: f64,
    /// Median encoding FPS
    pub median_encoding_fps: f64,
    /// Standard deviation of encoding FPS
    pub std_dev_encoding_fps: f64,
    /// Mean decoding FPS
    pub mean_decoding_fps: f64,
    /// Median decoding FPS
    pub median_decoding_fps: f64,
    /// Standard deviation of decoding FPS
    pub std_dev_decoding_fps: f64,
    /// Mean PSNR
    pub mean_psnr: Option<f64>,
    /// Median PSNR
    pub median_psnr: Option<f64>,
    /// Mean SSIM
    pub mean_ssim: Option<f64>,
    /// Median SSIM
    pub median_ssim: Option<f64>,
    /// Mean file size
    pub mean_file_size: u64,
    /// Median file size
    pub median_file_size: u64,
    /// 95th percentile encoding FPS
    pub p95_encoding_fps: f64,
    /// 99th percentile encoding FPS
    pub p99_encoding_fps: f64,
    /// 95% confidence interval lower bound for mean encoding FPS
    pub ci_encoding_fps_lower: f64,
    /// 95% confidence interval upper bound for mean encoding FPS
    pub ci_encoding_fps_upper: f64,
    /// 95% confidence interval lower bound for mean decoding FPS
    pub ci_decoding_fps_lower: f64,
    /// 95% confidence interval upper bound for mean decoding FPS
    pub ci_decoding_fps_upper: f64,
    /// Number of outliers removed from encoding FPS before computing stats
    pub outliers_removed: usize,
}

/// Statistical analysis provider.
pub trait StatisticalAnalysis {
    /// Compute statistics from sequence results.
    fn compute_statistics(results: &[SequenceResult]) -> Statistics;
}

impl StatisticalAnalysis for Statistics {
    fn compute_statistics(results: &[SequenceResult]) -> Statistics {
        compute_statistics(results)
    }
}

/// Remove outliers from a sorted slice using the IQR (Tukey fences) method.
///
/// Values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` are considered outliers.
/// Returns the cleaned values and the number of outliers removed.
fn remove_outliers_iqr(values: &[f64]) -> (Vec<f64>, usize) {
    if values.len() < 4 {
        // Too few points for reliable IQR estimation — return as-is.
        return (values.to_vec(), 0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = percentile(&mut sorted.clone(), 25.0);
    let q3 = percentile(&mut sorted.clone(), 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;

    let cleaned: Vec<f64> = values
        .iter()
        .copied()
        .filter(|&v| v >= lower && v <= upper)
        .collect();
    let removed = values.len() - cleaned.len();
    (cleaned, removed)
}

/// Compute statistics from sequence results.
///
/// Encoding and decoding FPS values are cleaned of outliers (IQR method) before
/// computing means and confidence intervals.  The `outliers_removed` field on the
/// returned [`Statistics`] reflects how many FPS measurements were discarded.
#[must_use]
pub fn compute_statistics(results: &[SequenceResult]) -> Statistics {
    if results.is_empty() {
        return Statistics::default();
    }

    let raw_encoding_fps: Vec<f64> = results.iter().map(|r| r.encoding_fps).collect();
    let raw_decoding_fps: Vec<f64> = results.iter().map(|r| r.decoding_fps).collect();
    let file_sizes: Vec<u64> = results.iter().map(|r| r.file_size_bytes).collect();

    let psnr_values: Vec<f64> = results.iter().filter_map(|r| r.metrics.psnr).collect();
    let ssim_values: Vec<f64> = results.iter().filter_map(|r| r.metrics.ssim).collect();

    // Apply IQR outlier removal to FPS measurements.
    let (encoding_fps, enc_out) = remove_outliers_iqr(&raw_encoding_fps);
    let (decoding_fps, dec_out) = remove_outliers_iqr(&raw_decoding_fps);
    let outliers_removed = enc_out + dec_out;

    // Compute 95 % confidence intervals using z = 1.96 (large-sample approximation).
    let z95 = 1.96_f64;
    let enc_ci = {
        let sd = std_dev(&encoding_fps);
        let n = (encoding_fps.len() as f64).max(1.0);
        let margin = z95 * (sd / n.sqrt());
        let m = mean(&encoding_fps);
        (m - margin, m + margin)
    };
    let dec_ci = {
        let sd = std_dev(&decoding_fps);
        let n = (decoding_fps.len() as f64).max(1.0);
        let margin = z95 * (sd / n.sqrt());
        let m = mean(&decoding_fps);
        (m - margin, m + margin)
    };

    Statistics {
        mean_encoding_fps: mean(&encoding_fps),
        median_encoding_fps: median(&mut encoding_fps.clone()),
        std_dev_encoding_fps: std_dev(&encoding_fps),
        mean_decoding_fps: mean(&decoding_fps),
        median_decoding_fps: median(&mut decoding_fps.clone()),
        std_dev_decoding_fps: std_dev(&decoding_fps),
        mean_psnr: if psnr_values.is_empty() {
            None
        } else {
            Some(mean(&psnr_values))
        },
        median_psnr: if psnr_values.is_empty() {
            None
        } else {
            Some(median(&mut psnr_values.clone()))
        },
        mean_ssim: if ssim_values.is_empty() {
            None
        } else {
            Some(mean(&ssim_values))
        },
        median_ssim: if ssim_values.is_empty() {
            None
        } else {
            Some(median(&mut ssim_values.clone()))
        },
        mean_file_size: mean_u64(&file_sizes),
        median_file_size: median_u64(&mut file_sizes.clone()),
        p95_encoding_fps: percentile(&mut encoding_fps.clone(), 95.0),
        p99_encoding_fps: percentile(&mut encoding_fps.clone(), 99.0),
        ci_encoding_fps_lower: enc_ci.0,
        ci_encoding_fps_upper: enc_ci.1,
        ci_decoding_fps_lower: dec_ci.0,
        ci_decoding_fps_upper: dec_ci.1,
        outliers_removed,
    }
}

/// Calculate mean of values.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate mean of u64 values.
fn mean_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.iter().sum::<u64>() / values.len() as u64
}

/// Calculate median of values (modifies input).
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Calculate median of u64 values (modifies input).
fn median_u64(values: &mut [u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }

    values.sort_unstable();

    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2
    } else {
        values[mid]
    }
}

/// Calculate standard deviation.
fn std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean_val = mean(values);
    let variance = values
        .iter()
        .map(|v| {
            let diff = v - mean_val;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;

    variance.sqrt()
}

/// Calculate percentile value using the ceiling nearest-rank method.
///
/// Uses `ceil(p/100 * n) - 1` (0-indexed) so that the 50th percentile of
/// `[1..10]` returns the element at index 4 (value 5), and the 95th percentile
/// returns the element at index 9 (value 10).
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let raw = p / 100.0 * values.len() as f64;
    // ceil then subtract 1, clamped to [0, n-1]
    let idx = if raw.fract() < f64::EPSILON {
        (raw as usize).saturating_sub(1).min(values.len() - 1)
    } else {
        (raw.ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1)
    };
    values[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QualityMetrics, SequenceResult};
    use std::time::Duration;

    fn make_result(encoding_fps: f64, psnr: Option<f64>) -> SequenceResult {
        SequenceResult {
            sequence_name: "test".to_string(),
            frames_processed: 100,
            encoding_fps,
            decoding_fps: encoding_fps * 2.0,
            file_size_bytes: 1_000_000,
            metrics: QualityMetrics {
                psnr,
                ..Default::default()
            },
            encoding_duration: Duration::from_secs(1),
            decoding_duration: Duration::from_secs(1),
        }
    }

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(mean(&[10.0, 20.0]), 15.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_median_odd() {
        let mut values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(median(&mut values), 3.0);
    }

    #[test]
    fn test_median_even() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&mut values), 2.5);
    }

    #[test]
    fn test_median_empty() {
        let mut values: Vec<f64> = vec![];
        assert_eq!(median(&mut values), 0.0);
    }

    #[test]
    fn test_std_dev() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&values);
        assert!((sd - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_std_dev_single_value() {
        assert_eq!(std_dev(&[5.0]), 0.0);
    }

    #[test]
    fn test_percentile() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&mut values, 50.0), 5.0);
        assert_eq!(percentile(&mut values, 95.0), 10.0);
    }

    #[test]
    fn test_compute_statistics() {
        let results = vec![
            make_result(30.0, Some(35.0)),
            make_result(40.0, Some(38.0)),
            make_result(35.0, Some(36.0)),
        ];

        let stats = compute_statistics(&results);
        assert_eq!(stats.mean_encoding_fps, 35.0);
        assert_eq!(stats.median_encoding_fps, 35.0);
        assert!(stats.mean_psnr.is_some());
        assert!(
            (stats.mean_psnr.expect("test expectation failed") - 36.333_333_333_333_336).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_compute_statistics_empty() {
        let results: Vec<SequenceResult> = vec![];
        let stats = compute_statistics(&results);
        assert_eq!(stats.mean_encoding_fps, 0.0);
        assert!(stats.mean_psnr.is_none());
    }

    #[test]
    fn test_median_u64() {
        let mut values = vec![1u64, 2, 3, 4, 5];
        assert_eq!(median_u64(&mut values), 3);

        let mut values = vec![1u64, 2, 3, 4];
        assert_eq!(median_u64(&mut values), 2);
    }

    #[test]
    fn test_mean_u64() {
        assert_eq!(mean_u64(&[1, 2, 3, 4, 5]), 3);
        assert_eq!(mean_u64(&[10, 20]), 15);
        assert_eq!(mean_u64(&[]), 0);
    }
}

/// Advanced statistical analysis tools.
pub struct AdvancedStats;

impl AdvancedStats {
    /// Calculate coefficient of variation (CV).
    #[must_use]
    pub fn coefficient_of_variation(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean_val = mean(values);
        if mean_val.abs() < f64::EPSILON {
            return 0.0;
        }

        let std_dev_val = std_dev(values);
        std_dev_val / mean_val
    }

    /// Calculate skewness (measure of asymmetry).
    #[must_use]
    pub fn skewness(values: &[f64]) -> f64 {
        let n = values.len() as f64;
        if n < 3.0 {
            return 0.0;
        }

        let mean_val = mean(values);
        let std_dev_val = std_dev(values);

        if std_dev_val.abs() < f64::EPSILON {
            return 0.0;
        }

        let m3 = values
            .iter()
            .map(|v| {
                let diff = v - mean_val;
                diff * diff * diff
            })
            .sum::<f64>()
            / n;

        m3 / std_dev_val.powi(3)
    }

    /// Calculate kurtosis (measure of tailedness).
    #[must_use]
    pub fn kurtosis(values: &[f64]) -> f64 {
        let n = values.len() as f64;
        if n < 4.0 {
            return 0.0;
        }

        let mean_val = mean(values);
        let std_dev_val = std_dev(values);

        if std_dev_val.abs() < f64::EPSILON {
            return 0.0;
        }

        let m4 = values
            .iter()
            .map(|v| {
                let diff = v - mean_val;
                diff * diff * diff * diff
            })
            .sum::<f64>()
            / n;

        (m4 / std_dev_val.powi(4)) - 3.0
    }

    /// Calculate interquartile range (IQR).
    #[must_use]
    pub fn interquartile_range(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let q1 = percentile(values, 25.0);
        let q3 = percentile(values, 75.0);
        q3 - q1
    }

    /// Detect outliers using IQR method.
    #[must_use]
    pub fn detect_outliers(values: &mut [f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        let q1 = percentile(values, 25.0);
        let q3 = percentile(values, 75.0);
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        values
            .iter()
            .filter(|&&v| v < lower_bound || v > upper_bound)
            .copied()
            .collect()
    }

    /// Calculate confidence interval.
    #[must_use]
    pub fn confidence_interval(values: &[f64], confidence_level: f64) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let mean_val = mean(values);
        let std_dev_val = std_dev(values);
        let n = values.len() as f64;

        // Z-score for given confidence level (simplified)
        let z = match confidence_level {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // default to 95%
        };

        let margin = z * (std_dev_val / n.sqrt());

        (mean_val - margin, mean_val + margin)
    }

    /// Perform t-test comparing two samples.
    #[must_use]
    pub fn t_test(sample_a: &[f64], sample_b: &[f64]) -> TTestResult {
        if sample_a.len() < 2 || sample_b.len() < 2 {
            return TTestResult {
                t_statistic: 0.0,
                degrees_of_freedom: 0,
                p_value: 1.0,
                significant: false,
            };
        }

        let mean_a = mean(sample_a);
        let mean_b = mean(sample_b);
        let var_a = std_dev(sample_a).powi(2);
        let var_b = std_dev(sample_b).powi(2);
        let n_a = sample_a.len() as f64;
        let n_b = sample_b.len() as f64;

        let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
        let se = (pooled_var * (1.0 / n_a + 1.0 / n_b)).sqrt();

        let t_stat = if se.abs() < f64::EPSILON {
            0.0
        } else {
            (mean_a - mean_b) / se
        };

        let df = (n_a + n_b - 2.0) as usize;

        // Simplified p-value estimation (placeholder)
        let p_value = if t_stat.abs() > 2.0 { 0.05 } else { 0.5 };

        TTestResult {
            t_statistic: t_stat,
            degrees_of_freedom: df,
            p_value,
            significant: p_value < 0.05,
        }
    }
}

/// T-test result.
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// T-statistic value
    pub t_statistic: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
    /// P-value
    pub p_value: f64,
    /// Whether the result is statistically significant (p < 0.05)
    pub significant: bool,
}

/// Statistical distribution analyzer.
pub struct DistributionAnalyzer;

impl DistributionAnalyzer {
    /// Analyze the distribution of values.
    #[must_use]
    pub fn analyze(values: &[f64]) -> DistributionAnalysis {
        let mut sorted = values.to_vec();

        DistributionAnalysis {
            mean: mean(values),
            median: median(&mut sorted),
            mode: Self::calculate_mode(values),
            std_dev: std_dev(values),
            variance: std_dev(values).powi(2),
            skewness: AdvancedStats::skewness(values),
            kurtosis: AdvancedStats::kurtosis(values),
            min: sorted.first().copied().unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
            range: sorted.last().unwrap_or(&0.0) - sorted.first().unwrap_or(&0.0),
        }
    }

    fn calculate_mode(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }

        let mut counts = std::collections::HashMap::new();
        for &value in values {
            *counts.entry((value * 100.0) as i64).or_insert(0) += 1;
        }

        counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&value, _)| value as f64 / 100.0)
    }
}

/// Distribution analysis result.
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Mode (most frequent value)
    pub mode: Option<f64>,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Range (max - min)
    pub range: f64,
}

#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_coefficient_of_variation() {
        let values = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let cv = AdvancedStats::coefficient_of_variation(&values);
        assert!(cv > 0.0);
        assert!(cv < 1.0);
    }

    #[test]
    fn test_skewness() {
        let values = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        let skew = AdvancedStats::skewness(&values);
        // Should be slightly negative (left-skewed)
        assert!(skew < 0.5);
    }

    #[test]
    fn test_kurtosis() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kurt = AdvancedStats::kurtosis(&values);
        // Normal distribution has kurtosis ~ 0
        assert!(kurt.abs() < 2.0);
    }

    #[test]
    fn test_iqr() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let iqr = AdvancedStats::interquartile_range(&mut values);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_outliers() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is an outlier
        let outliers = AdvancedStats::detect_outliers(&mut values);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_confidence_interval() {
        let values = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let (lower, upper) = AdvancedStats::confidence_interval(&values, 0.95);
        assert!(lower < upper);
        assert!(lower < 14.0);
        assert!(upper > 14.0);
    }

    #[test]
    fn test_t_test() {
        let sample_a = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let sample_b = vec![11.0, 13.0, 15.0, 17.0, 19.0];
        let result = AdvancedStats::t_test(&sample_a, &sample_b);
        assert!(result.degrees_of_freedom > 0);
    }

    #[test]
    fn test_distribution_analysis() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let analysis = DistributionAnalyzer::analyze(&values);
        assert_eq!(analysis.mean, 3.0);
        assert_eq!(analysis.median, 3.0);
        assert_eq!(analysis.min, 1.0);
        assert_eq!(analysis.max, 5.0);
        assert_eq!(analysis.range, 4.0);
    }
}

// ─── Welch's t-test and bootstrap confidence intervals ───────────────────────

use rand::RngExt;
use rand::SeedableRng;

/// Errors produced by the statistical inference functions.
#[derive(Debug, thiserror::Error)]
pub enum StatsError {
    /// The sample has fewer than 2 observations.
    #[error("insufficient data: need at least 2 observations, got {got}")]
    InsufficientData {
        /// Actual number of observations.
        got: usize,
    },
    /// The confidence level is outside (0, 1).
    #[error("invalid confidence level {level}: must be in (0, 1)")]
    InvalidConfidence {
        /// The invalid confidence value.
        level: f64,
    },
}

/// Result of a Welch's two-sample t-test.
#[derive(Debug, Clone, PartialEq)]
pub struct WelchResult {
    /// The Welch t-statistic.
    pub t_statistic: f64,
    /// Welch–Satterthwaite degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Two-tailed p-value.
    pub p_value: f64,
}

/// Percentile bootstrap confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapCi {
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
    /// Number of bootstrap resamples used.
    pub n_resamples: usize,
}

// ── Regularised incomplete beta via Lentz's continued-fraction (two-tailed p) ─

/// Evaluate the beta continued-fraction kernel using the Legendre CF:
///
///   cf(x, a, b) = 1 / (1 + d1/(1 + d2/(1 + ...)))
///
/// where d_{2m}   = m*(b-m)*x / [(a+2m-1)*(a+2m)]
///       d_{2m+1} = -(a+m)*(a+b+m)*x / [(a+2m)*(a+2m+1)]
///
/// Returns the CF value directly (not the full I_x).
/// Reference: Abramowitz & Stegun §26.5.8 / Numerical Recipes §6.4.
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: u32 = 200;
    const EPS: f64 = 3e-15;
    const TINY: f64 = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    // First step of Lentz initialised to the first denominator element.
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1_u32..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    h
}

/// Evaluate the regularised incomplete beta I_x(a, b) using the CF decomposition
/// `I_x(a,b) = x^a*(1-x)^b / (a*B(a,b)) * betacf(x,a,b)`.
///
/// Uses the symmetry relation `I_x(a,b) = 1 - I_{1-x}(b,a)` for x > (a+1)/(a+b+2).
fn regularised_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    // Guard: domain
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // log of the beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);

    // Symmetry: use I_{1-x}(b,a) = 1 - I_x(a,b) when x is large.
    let (xx, aa, bb, flipped) = if x < (a + 1.0) / (a + b + 2.0) {
        (x, a, b, false)
    } else {
        (1.0 - x, b, a, true)
    };

    // Compute the prefactor entirely in log-space to avoid underflow.
    // front = x^a*(1-x)^b / (a*B(a,b))  [in log space]
    let log_front = aa * xx.ln() + bb * (1.0 - xx).ln() - lbeta - aa.ln();

    let cf = betacf(xx, aa, bb);

    // cf must be positive; guard against numerical drift.
    let val = if cf > 0.0 {
        (log_front + cf.ln()).exp().clamp(0.0, 1.0)
    } else if cf < 0.0 {
        // Sign drift: use the power-series bound as a fallback.
        // For the regularised beta, result is in (0,1), so clamp to 0.
        0.0
    } else {
        0.0
    };

    if flipped {
        1.0 - val
    } else {
        val
    }
}

/// Approximation of the log-gamma function via Lanczos (g = 7, n = 9).
fn lgamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_998_099_3,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_4e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let mut z = z;
    if z < 0.5 {
        // Reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - lgamma(1.0 - z);
    }
    z -= 1.0;
    let mut x = C[0];
    for (i, &ci) in C.iter().enumerate().skip(1) {
        x += ci / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Two-tailed p-value for a t-statistic and given degrees of freedom.
///
/// Uses the relationship: p = I_{df/(df+t²)}(df/2, 1/2) where I is the
/// regularised incomplete beta function.
fn t_dist_two_tailed_p(t: f64, df: f64) -> f64 {
    let t2 = t * t;
    let x = df / (df + t2);
    // I_x(df/2, 1/2) gives the two-tailed p-value directly.
    regularised_incomplete_beta(x, df / 2.0, 0.5)
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Perform Welch's unequal-variances two-sample t-test.
///
/// Returns the t-statistic, Welch–Satterthwaite degrees of freedom, and
/// two-tailed p-value.  Requires at least 2 observations in each sample.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> Result<WelchResult, StatsError> {
    let n1 = a.len();
    let n2 = b.len();
    if n1 < 2 {
        return Err(StatsError::InsufficientData { got: n1 });
    }
    if n2 < 2 {
        return Err(StatsError::InsufficientData { got: n2 });
    }

    let mean1 = a.iter().sum::<f64>() / n1 as f64;
    let mean2 = b.iter().sum::<f64>() / n2 as f64;

    let var1 = a.iter().map(|&v| (v - mean1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
    let var2 = b.iter().map(|&v| (v - mean2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

    let se = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();

    // Guard against degenerate case where both variances are zero.
    let t_statistic = if se == 0.0 {
        if (mean1 - mean2).abs() < f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (mean1 - mean2) / se
    };

    // Welch–Satterthwaite degrees of freedom
    let s1_n1 = var1 / n1 as f64;
    let s2_n2 = var2 / n2 as f64;
    let denom = s1_n1.powi(2) / (n1 - 1) as f64 + s2_n2.powi(2) / (n2 - 1) as f64;
    let degrees_of_freedom = if denom == 0.0 {
        (n1 + n2 - 2) as f64
    } else {
        (s1_n1 + s2_n2).powi(2) / denom
    };

    let p_value = if t_statistic.is_infinite() || t_statistic.is_nan() {
        0.0
    } else {
        t_dist_two_tailed_p(t_statistic, degrees_of_freedom)
    };

    Ok(WelchResult {
        t_statistic,
        degrees_of_freedom,
        p_value,
    })
}

/// Compute a percentile bootstrap confidence interval for the mean.
///
/// Uses `n_resamples` bootstrap resamples (default 1000 when 0 is passed).
/// The `confidence` parameter must be in the open interval (0, 1).
pub fn bootstrap_ci(
    data: &[f64],
    confidence: f64,
    n_resamples: usize,
) -> Result<BootstrapCi, StatsError> {
    if data.len() < 2 {
        return Err(StatsError::InsufficientData { got: data.len() });
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(StatsError::InvalidConfidence { level: confidence });
    }

    let n_resamples = if n_resamples == 0 { 1000 } else { n_resamples };
    let n = data.len();

    // Deterministic seed for reproducibility across runs.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xdeadbeef_cafebabe);

    let mut means: Vec<f64> = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let resample_sum: f64 = (0..n).map(|_| data[rng.random_range(0..n)]).sum();
        means.push(resample_sum / n as f64);
    }

    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * n_resamples as f64).floor() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_resamples as f64).ceil() as usize;

    let lo_idx = lo_idx.min(n_resamples - 1);
    let hi_idx = hi_idx.min(n_resamples - 1);

    Ok(BootstrapCi {
        lower: means[lo_idx],
        upper: means[hi_idx],
        n_resamples,
    })
}
