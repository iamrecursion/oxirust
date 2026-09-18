//! Cost estimator: predict job duration and resource usage from historical data.
//!
//! Unlike [`crate::resource_estimator`] which uses static heuristics, this module
//! learns from completed jobs.  It maintains a history of job executions and uses
//! statistical analysis (mean, median, percentiles, linear regression) to predict
//! future job costs with increasing accuracy over time.

use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{BatchError, Result};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A completed job record used to build the prediction model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    /// Category key (e.g. "transcode:h264:1080p", "file_op:copy").
    pub category: String,
    /// Input size in bytes.
    pub input_size_bytes: u64,
    /// Actual wall-clock duration in seconds.
    pub duration_secs: f64,
    /// Peak CPU utilization (0.0 ..= cores).
    pub peak_cpu: f64,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Peak disk I/O in bytes.
    pub peak_disk_bytes: u64,
    /// Whether the job succeeded.
    pub succeeded: bool,
    /// Unix timestamp when the job completed.
    pub completed_at_secs: u64,
}

impl JobRecord {
    /// Create a new completed job record.
    #[must_use]
    pub fn new(
        category: impl Into<String>,
        input_size_bytes: u64,
        duration_secs: f64,
        peak_cpu: f64,
        peak_memory_bytes: u64,
    ) -> Self {
        Self {
            category: category.into(),
            input_size_bytes,
            duration_secs,
            peak_cpu,
            peak_memory_bytes,
            peak_disk_bytes: 0,
            succeeded: true,
            completed_at_secs: current_timestamp(),
        }
    }

    /// Builder: set peak disk bytes.
    #[must_use]
    pub fn with_disk(mut self, bytes: u64) -> Self {
        self.peak_disk_bytes = bytes;
        self
    }

    /// Builder: mark as failed.
    #[must_use]
    pub fn failed(mut self) -> Self {
        self.succeeded = false;
        self
    }

    /// Processing speed in bytes per second (0.0 if duration is zero).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bytes_per_sec(&self) -> f64 {
        if self.duration_secs > 0.0 {
            self.input_size_bytes as f64 / self.duration_secs
        } else {
            0.0
        }
    }
}

/// Predicted cost for a future job.
#[derive(Debug, Clone)]
pub struct CostPrediction {
    /// Predicted duration in seconds.
    pub predicted_duration_secs: f64,
    /// Predicted peak CPU utilization.
    pub predicted_cpu: f64,
    /// Predicted peak memory in bytes.
    pub predicted_memory_bytes: u64,
    /// Predicted disk I/O in bytes.
    pub predicted_disk_bytes: u64,
    /// Confidence level (0.0 ..= 1.0).
    pub confidence: f64,
    /// Number of historical samples used.
    pub sample_count: usize,
    /// Prediction method used.
    pub method: PredictionMethod,
}

/// How the prediction was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMethod {
    /// Simple average of all records in the category.
    Mean,
    /// Median of all records in the category.
    Median,
    /// Linear regression on input size vs. duration.
    LinearRegression,
    /// Exponentially weighted moving average (recent data weighted more).
    Ewma,
    /// No data available; used the global fallback.
    Fallback,
}

impl std::fmt::Display for PredictionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Median => write!(f, "median"),
            Self::LinearRegression => write!(f, "linear_regression"),
            Self::Ewma => write!(f, "ewma"),
            Self::Fallback => write!(f, "fallback"),
        }
    }
}

// ---------------------------------------------------------------------------
// History store
// ---------------------------------------------------------------------------

/// Maximum records kept per category to bound memory usage.
const MAX_RECORDS_PER_CATEGORY: usize = 1000;

/// Thread-safe store of historical job records.
#[derive(Debug, Default)]
pub struct JobHistory {
    records: RwLock<HashMap<String, Vec<JobRecord>>>,
}

impl JobHistory {
    /// Create a new empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed job.
    pub fn record(&self, record: JobRecord) {
        let mut map = self.records.write();
        let bucket = map.entry(record.category.clone()).or_default();
        bucket.push(record);
        // Evict oldest records if the bucket is too large.
        if bucket.len() > MAX_RECORDS_PER_CATEGORY {
            let excess = bucket.len() - MAX_RECORDS_PER_CATEGORY;
            bucket.drain(..excess);
        }
    }

    /// Get all records for a category.
    #[must_use]
    pub fn get_records(&self, category: &str) -> Vec<JobRecord> {
        self.records
            .read()
            .get(category)
            .cloned()
            .unwrap_or_default()
    }

    /// Get only successful records for a category.
    #[must_use]
    pub fn get_successful_records(&self, category: &str) -> Vec<JobRecord> {
        self.records
            .read()
            .get(category)
            .map(|recs| recs.iter().filter(|r| r.succeeded).cloned().collect())
            .unwrap_or_default()
    }

    /// Total number of records across all categories.
    #[must_use]
    pub fn total_records(&self) -> usize {
        self.records.read().values().map(|v| v.len()).sum()
    }

    /// Number of distinct categories.
    #[must_use]
    pub fn category_count(&self) -> usize {
        self.records.read().len()
    }

    /// List all known categories.
    #[must_use]
    pub fn categories(&self) -> Vec<String> {
        self.records.read().keys().cloned().collect()
    }

    /// Clear all history.
    pub fn clear(&self) {
        self.records.write().clear();
    }
}

// ---------------------------------------------------------------------------
// Cost estimator
// ---------------------------------------------------------------------------

/// Predicts future job costs from historical data.
#[derive(Debug)]
pub struct CostEstimator {
    history: JobHistory,
    /// Minimum number of samples needed for linear regression.
    min_regression_samples: usize,
    /// EWMA decay factor (0.0..1.0). Higher = more weight on recent data.
    ewma_alpha: f64,
    /// Fallback duration estimate when no history is available.
    fallback_duration_secs: f64,
    /// Fallback memory estimate when no history is available.
    fallback_memory_bytes: u64,
}

impl CostEstimator {
    /// Create a new cost estimator with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: JobHistory::new(),
            min_regression_samples: 10,
            ewma_alpha: 0.3,
            fallback_duration_secs: 60.0,
            fallback_memory_bytes: 256 * 1024 * 1024,
        }
    }

    /// Create a cost estimator with a shared history.
    #[must_use]
    pub fn with_history(history: JobHistory) -> Self {
        Self {
            history,
            min_regression_samples: 10,
            ewma_alpha: 0.3,
            fallback_duration_secs: 60.0,
            fallback_memory_bytes: 256 * 1024 * 1024,
        }
    }

    /// Set the EWMA decay factor.
    pub fn set_ewma_alpha(&mut self, alpha: f64) {
        self.ewma_alpha = alpha.clamp(0.01, 0.99);
    }

    /// Set the fallback duration.
    pub fn set_fallback_duration(&mut self, secs: f64) {
        self.fallback_duration_secs = secs.max(0.0);
    }

    /// Record a completed job for future predictions.
    pub fn record(&self, record: JobRecord) {
        self.history.record(record);
    }

    /// Get a reference to the underlying history.
    #[must_use]
    pub fn history(&self) -> &JobHistory {
        &self.history
    }

    /// Predict the cost for a job in the given category with the given input size.
    ///
    /// The estimator automatically selects the best prediction method based on
    /// the number and quality of available historical records.
    ///
    /// # Errors
    ///
    /// This function does not currently return errors; it falls back to default
    /// estimates when no history is available. The `Result` return type is used
    /// for forward compatibility.
    pub fn predict(&self, category: &str, input_size_bytes: u64) -> Result<CostPrediction> {
        let records = self.history.get_successful_records(category);

        if records.is_empty() {
            return Ok(self.fallback_prediction());
        }

        // Choose method based on sample count.
        if records.len() >= self.min_regression_samples {
            Ok(self.predict_regression(&records, input_size_bytes))
        } else if records.len() >= 3 {
            Ok(self.predict_ewma(&records, input_size_bytes))
        } else {
            Ok(self.predict_mean(&records))
        }
    }

    /// Predict using simple mean (duration via median for outlier robustness).
    #[allow(clippy::cast_precision_loss)]
    fn predict_mean(&self, records: &[JobRecord]) -> CostPrediction {
        let n = records.len() as f64;
        // Use median for duration to resist outliers; mean for CPU/memory.
        let mut durations: Vec<f64> = records.iter().map(|r| r.duration_secs).collect();
        let avg_duration = median(&mut durations);
        let avg_cpu = records.iter().map(|r| r.peak_cpu).sum::<f64>() / n;
        let avg_mem =
            records.iter().map(|r| r.peak_memory_bytes).sum::<u64>() / records.len() as u64;
        let avg_disk =
            records.iter().map(|r| r.peak_disk_bytes).sum::<u64>() / records.len() as u64;

        // Confidence based on sample count (caps at 0.7 for mean-only).
        let confidence = (records.len() as f64 / 20.0).min(0.7);

        CostPrediction {
            predicted_duration_secs: avg_duration,
            predicted_cpu: avg_cpu,
            predicted_memory_bytes: avg_mem,
            predicted_disk_bytes: avg_disk,
            confidence,
            sample_count: records.len(),
            method: PredictionMethod::Mean,
        }
    }

    /// Predict using exponentially weighted moving average.
    #[allow(clippy::cast_precision_loss)]
    fn predict_ewma(&self, records: &[JobRecord], _input_size_bytes: u64) -> CostPrediction {
        let alpha = self.ewma_alpha;
        let mut ewma_duration = records[0].duration_secs;
        let mut ewma_cpu = records[0].peak_cpu;
        let mut ewma_mem = records[0].peak_memory_bytes as f64;
        let mut ewma_disk = records[0].peak_disk_bytes as f64;

        for r in &records[1..] {
            ewma_duration = alpha * r.duration_secs + (1.0 - alpha) * ewma_duration;
            ewma_cpu = alpha * r.peak_cpu + (1.0 - alpha) * ewma_cpu;
            ewma_mem = alpha * r.peak_memory_bytes as f64 + (1.0 - alpha) * ewma_mem;
            ewma_disk = alpha * r.peak_disk_bytes as f64 + (1.0 - alpha) * ewma_disk;
        }

        let confidence = (records.len() as f64 / 15.0).min(0.8);

        CostPrediction {
            predicted_duration_secs: ewma_duration,
            predicted_cpu: ewma_cpu,
            predicted_memory_bytes: ewma_mem as u64,
            predicted_disk_bytes: ewma_disk as u64,
            confidence,
            sample_count: records.len(),
            method: PredictionMethod::Ewma,
        }
    }

    /// Predict using simple linear regression (input_size -> duration).
    ///
    /// Memory and CPU are predicted using the mean since they correlate less
    /// linearly with input size.
    #[allow(clippy::cast_precision_loss)]
    fn predict_regression(&self, records: &[JobRecord], input_size_bytes: u64) -> CostPrediction {
        let n = records.len() as f64;

        // Linear regression: duration = slope * input_size + intercept
        let sum_x: f64 = records.iter().map(|r| r.input_size_bytes as f64).sum();
        let sum_y: f64 = records.iter().map(|r| r.duration_secs).sum();
        let sum_xy: f64 = records
            .iter()
            .map(|r| r.input_size_bytes as f64 * r.duration_secs)
            .sum();
        let sum_xx: f64 = records
            .iter()
            .map(|r| {
                let x = r.input_size_bytes as f64;
                x * x
            })
            .sum();

        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let denominator = sum_xx - n * mean_x * mean_x;
        let (slope, intercept) = if denominator.abs() > f64::EPSILON {
            let s = (sum_xy - n * mean_x * mean_y) / denominator;
            let i = mean_y - s * mean_x;
            (s, i)
        } else {
            // All inputs are the same size; fall back to mean.
            (0.0, mean_y)
        };

        let predicted_duration = (slope * input_size_bytes as f64 + intercept).max(0.0);

        // For CPU, memory, disk: use mean.
        let avg_cpu = records.iter().map(|r| r.peak_cpu).sum::<f64>() / n;
        let avg_mem =
            records.iter().map(|r| r.peak_memory_bytes).sum::<u64>() / records.len() as u64;
        let avg_disk =
            records.iter().map(|r| r.peak_disk_bytes).sum::<u64>() / records.len() as u64;

        // Compute R-squared for confidence.
        let ss_tot: f64 = records
            .iter()
            .map(|r| (r.duration_secs - mean_y).powi(2))
            .sum();
        let ss_res: f64 = records
            .iter()
            .map(|r| {
                let predicted = slope * r.input_size_bytes as f64 + intercept;
                (r.duration_secs - predicted).powi(2)
            })
            .sum();
        let r_squared = if ss_tot.abs() > f64::EPSILON {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Confidence combines R-squared with sample count.
        let sample_confidence = (records.len() as f64 / 30.0).min(1.0);
        let confidence = (r_squared * 0.6 + sample_confidence * 0.4).min(0.95);

        CostPrediction {
            predicted_duration_secs: predicted_duration,
            predicted_cpu: avg_cpu,
            predicted_memory_bytes: avg_mem,
            predicted_disk_bytes: avg_disk,
            confidence,
            sample_count: records.len(),
            method: PredictionMethod::LinearRegression,
        }
    }

    /// Fallback when no history is available.
    fn fallback_prediction(&self) -> CostPrediction {
        CostPrediction {
            predicted_duration_secs: self.fallback_duration_secs,
            predicted_cpu: 1.0,
            predicted_memory_bytes: self.fallback_memory_bytes,
            predicted_disk_bytes: 0,
            confidence: 0.1,
            sample_count: 0,
            method: PredictionMethod::Fallback,
        }
    }

    /// Get a summary of prediction accuracy for a category by comparing
    /// predictions against actual records (leave-one-out cross-validation).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if no records exist for the category.
    #[allow(clippy::cast_precision_loss)]
    pub fn accuracy_report(&self, category: &str) -> Result<AccuracyReport> {
        let records = self.history.get_successful_records(category);
        if records.is_empty() {
            return Err(BatchError::JobNotFound(format!(
                "No records for category: {category}"
            )));
        }

        let mut errors = Vec::with_capacity(records.len());
        for (i, record) in records.iter().enumerate() {
            // Leave-one-out: predict using all records except this one.
            let mut subset: Vec<JobRecord> = records.clone();
            subset.remove(i);
            if subset.is_empty() {
                continue;
            }
            let pred = self.predict_mean(&subset);
            let error_pct = if record.duration_secs.abs() > f64::EPSILON {
                ((pred.predicted_duration_secs - record.duration_secs) / record.duration_secs).abs()
                    * 100.0
            } else {
                0.0
            };
            errors.push(error_pct);
        }

        if errors.is_empty() {
            return Ok(AccuracyReport {
                category: category.to_string(),
                sample_count: records.len(),
                mean_error_pct: 0.0,
                median_error_pct: 0.0,
                p95_error_pct: 0.0,
                max_error_pct: 0.0,
            });
        }

        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_err = errors.iter().sum::<f64>() / errors.len() as f64;
        let median_err = errors[errors.len() / 2];
        let p95_idx = ((errors.len() as f64) * 0.95) as usize;
        let p95_err = errors[p95_idx.min(errors.len() - 1)];
        let max_err = errors.last().copied().unwrap_or(0.0);

        Ok(AccuracyReport {
            category: category.to_string(),
            sample_count: records.len(),
            mean_error_pct: mean_err,
            median_error_pct: median_err,
            p95_error_pct: p95_err,
            max_error_pct: max_err,
        })
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Accuracy report for a prediction category.
#[derive(Debug, Clone)]
pub struct AccuracyReport {
    /// Category name.
    pub category: String,
    /// Number of samples used.
    pub sample_count: usize,
    /// Mean absolute percentage error.
    pub mean_error_pct: f64,
    /// Median absolute percentage error.
    pub median_error_pct: f64,
    /// 95th percentile error.
    pub p95_error_pct: f64,
    /// Maximum error observed.
    pub max_error_pct: f64,
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Compute the median of a slice of f64 values. Returns 0.0 for empty input.
#[allow(clippy::cast_precision_loss)]
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

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(category: &str, size: u64, duration: f64) -> JobRecord {
        JobRecord::new(category, size, duration, 2.0, 512 * 1024 * 1024)
    }

    #[test]
    fn test_job_record_bytes_per_sec() {
        let r = sample_record("cat", 1_000_000, 10.0);
        assert!((r.bytes_per_sec() - 100_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_record_bytes_per_sec_zero_duration() {
        let r = sample_record("cat", 1_000_000, 0.0);
        assert!((r.bytes_per_sec()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_history_record_and_retrieve() {
        let history = JobHistory::new();
        history.record(sample_record("transcode:h264", 1000, 10.0));
        history.record(sample_record("transcode:h264", 2000, 20.0));
        let recs = history.get_records("transcode:h264");
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn test_job_history_get_successful_records() {
        let history = JobHistory::new();
        history.record(sample_record("cat", 1000, 10.0));
        history.record(sample_record("cat", 2000, 20.0).failed());
        let recs = history.get_successful_records("cat");
        assert_eq!(recs.len(), 1);
    }

    #[test]
    fn test_job_history_total_records() {
        let history = JobHistory::new();
        history.record(sample_record("a", 1000, 10.0));
        history.record(sample_record("b", 2000, 20.0));
        history.record(sample_record("a", 3000, 30.0));
        assert_eq!(history.total_records(), 3);
    }

    #[test]
    fn test_job_history_category_count() {
        let history = JobHistory::new();
        history.record(sample_record("a", 1000, 10.0));
        history.record(sample_record("b", 2000, 20.0));
        assert_eq!(history.category_count(), 2);
    }

    #[test]
    fn test_job_history_categories() {
        let history = JobHistory::new();
        history.record(sample_record("alpha", 1000, 10.0));
        history.record(sample_record("beta", 2000, 20.0));
        let cats = history.categories();
        assert_eq!(cats.len(), 2);
    }

    #[test]
    fn test_job_history_clear() {
        let history = JobHistory::new();
        history.record(sample_record("a", 1000, 10.0));
        history.clear();
        assert_eq!(history.total_records(), 0);
    }

    #[test]
    fn test_job_history_evicts_old_records() {
        let history = JobHistory::new();
        for i in 0..1100 {
            history.record(sample_record("big", i, i as f64));
        }
        let recs = history.get_records("big");
        assert!(recs.len() <= MAX_RECORDS_PER_CATEGORY);
    }

    #[test]
    fn test_cost_estimator_fallback_when_no_history() {
        let estimator = CostEstimator::new();
        let pred = estimator
            .predict("unknown_category", 1000)
            .expect("predict should succeed");
        assert_eq!(pred.method, PredictionMethod::Fallback);
        assert_eq!(pred.sample_count, 0);
        assert!(pred.confidence < 0.2);
    }

    #[test]
    fn test_cost_estimator_mean_with_few_samples() {
        let estimator = CostEstimator::new();
        estimator.record(sample_record("test", 1000, 10.0));
        estimator.record(sample_record("test", 2000, 20.0));
        let pred = estimator
            .predict("test", 1500)
            .expect("predict should succeed");
        assert_eq!(pred.method, PredictionMethod::Mean);
        assert!((pred.predicted_duration_secs - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_estimator_ewma_with_moderate_samples() {
        let estimator = CostEstimator::new();
        for i in 0..5 {
            estimator.record(sample_record(
                "ewma_test",
                (i + 1) * 1000,
                (i + 1) as f64 * 10.0,
            ));
        }
        let pred = estimator
            .predict("ewma_test", 3000)
            .expect("predict should succeed");
        assert_eq!(pred.method, PredictionMethod::Ewma);
        assert!(pred.predicted_duration_secs > 0.0);
    }

    #[test]
    fn test_cost_estimator_regression_with_many_samples() {
        let estimator = CostEstimator::new();
        // Linear relationship: duration = 0.001 * input_size
        for i in 1..=15 {
            let size = i as u64 * 10_000;
            let duration = size as f64 * 0.001;
            estimator.record(sample_record("regression_test", size, duration));
        }
        let pred = estimator
            .predict("regression_test", 80_000)
            .expect("predict should succeed");
        assert_eq!(pred.method, PredictionMethod::LinearRegression);
        // Predicted duration should be close to 80.0 (0.001 * 80000).
        assert!((pred.predicted_duration_secs - 80.0).abs() < 5.0);
    }

    #[test]
    fn test_cost_estimator_regression_confidence() {
        let estimator = CostEstimator::new();
        // Perfect linear data should have high confidence.
        for i in 1..=20 {
            let size = i as u64 * 1000;
            let duration = size as f64 * 0.01;
            estimator.record(sample_record("perfect", size, duration));
        }
        let pred = estimator
            .predict("perfect", 5000)
            .expect("predict should succeed");
        assert!(pred.confidence > 0.5);
    }

    #[test]
    fn test_cost_estimator_ignores_failed_records() {
        let estimator = CostEstimator::new();
        estimator.record(sample_record("fail_test", 1000, 10.0));
        estimator.record(sample_record("fail_test", 2000, 999.0).failed());
        let pred = estimator
            .predict("fail_test", 1500)
            .expect("predict should succeed");
        // Should only use the successful record.
        assert_eq!(pred.sample_count, 1);
        assert!((pred.predicted_duration_secs - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_estimator_accuracy_report() {
        let estimator = CostEstimator::new();
        for i in 1..=5 {
            estimator.record(sample_record("accuracy", i as u64 * 1000, i as f64 * 10.0));
        }
        let report = estimator
            .accuracy_report("accuracy")
            .expect("should produce report");
        assert_eq!(report.sample_count, 5);
        assert!(report.mean_error_pct >= 0.0);
    }

    #[test]
    fn test_cost_estimator_accuracy_report_no_data() {
        let estimator = CostEstimator::new();
        let result = estimator.accuracy_report("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_cost_estimator_set_ewma_alpha() {
        let mut estimator = CostEstimator::new();
        estimator.set_ewma_alpha(0.5);
        assert!((estimator.ewma_alpha - 0.5).abs() < f64::EPSILON);
        // Clamping
        estimator.set_ewma_alpha(2.0);
        assert!((estimator.ewma_alpha - 0.99).abs() < f64::EPSILON);
        estimator.set_ewma_alpha(-1.0);
        assert!((estimator.ewma_alpha - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_estimator_default() {
        let estimator = CostEstimator::default();
        assert_eq!(estimator.history().total_records(), 0);
    }

    #[test]
    fn test_median_helper() {
        assert!((median(&mut []) - 0.0).abs() < f64::EPSILON);
        assert!((median(&mut [5.0]) - 5.0).abs() < f64::EPSILON);
        assert!((median(&mut [1.0, 3.0]) - 2.0).abs() < f64::EPSILON);
        assert!((median(&mut [1.0, 2.0, 3.0]) - 2.0).abs() < f64::EPSILON);
        assert!((median(&mut [4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prediction_method_display() {
        assert_eq!(PredictionMethod::Mean.to_string(), "mean");
        assert_eq!(
            PredictionMethod::LinearRegression.to_string(),
            "linear_regression"
        );
        assert_eq!(PredictionMethod::Fallback.to_string(), "fallback");
    }

    #[test]
    fn test_job_record_builder() {
        let r = sample_record("cat", 1000, 10.0).with_disk(5000).failed();
        assert_eq!(r.peak_disk_bytes, 5000);
        assert!(!r.succeeded);
    }

    #[test]
    fn test_cost_estimator_with_history() {
        let history = JobHistory::new();
        history.record(sample_record("shared", 1000, 10.0));
        let estimator = CostEstimator::with_history(history);
        let pred = estimator.predict("shared", 1000).expect("should succeed");
        assert_eq!(pred.sample_count, 1);
    }
}
