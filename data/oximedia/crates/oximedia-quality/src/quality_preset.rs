#![allow(dead_code)]
//! Quality assessment presets and profiles.
//!
//! Defines reusable quality assessment configurations for common workflows
//! such as broadcast delivery, streaming, archival, and post-production QC.
//! Each preset specifies which metrics to evaluate, their thresholds, and
//! the pooling strategy for aggregation.

use std::collections::HashMap;

/// Name of a built-in preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresetName {
    /// Broadcast delivery (EBU / SMPTE compliance)
    Broadcast,
    /// OTT streaming delivery
    Streaming,
    /// High-quality archival
    Archival,
    /// Fast screening for dailies review
    Dailies,
    /// Full post-production QC
    PostProduction,
    /// User-generated content ingest
    Ugc,
}

/// Pooling strategy for aggregating per-frame scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolStrategy {
    /// Arithmetic mean
    Mean,
    /// Harmonic mean (penalises low outliers)
    HarmonicMean,
    /// Minimum value across all frames
    Min,
    /// Percentile (e.g. 5th percentile)
    Percentile(u8),
}

impl PoolStrategy {
    /// Applies the pooling strategy to a slice of values.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        match self {
            Self::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Self::HarmonicMean => {
                let sum_inv: f64 = values.iter().map(|v| 1.0 / v.max(1e-10)).sum();
                values.len() as f64 / sum_inv
            }
            Self::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
            Self::Percentile(p) => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = (f64::from(*p) / 100.0 * sorted.len() as f64) as usize;
                sorted[idx.min(sorted.len() - 1)]
            }
        }
    }
}

/// Threshold definition for a single metric.
#[derive(Debug, Clone)]
pub struct MetricThreshold {
    /// Metric name (e.g. "PSNR", "SSIM", "VMAF")
    pub metric: String,
    /// Minimum acceptable value (scores below this trigger an error)
    pub error_below: Option<f64>,
    /// Warning threshold (scores below this trigger a warning)
    pub warn_below: Option<f64>,
    /// Maximum acceptable value (scores above this trigger an error)
    pub error_above: Option<f64>,
    /// Warning threshold (scores above this trigger a warning)
    pub warn_above: Option<f64>,
    /// Whether this metric is mandatory (must be computed)
    pub required: bool,
    /// Pooling strategy for this metric
    pub pool: PoolStrategy,
}

impl MetricThreshold {
    /// Creates a new threshold with a minimum-value error boundary.
    pub fn min_required(metric: impl Into<String>, min_value: f64) -> Self {
        Self {
            metric: metric.into(),
            error_below: Some(min_value),
            warn_below: None,
            error_above: None,
            warn_above: None,
            required: true,
            pool: PoolStrategy::Mean,
        }
    }

    /// Sets the warning threshold below which a warning is emitted.
    #[must_use]
    pub fn with_warn_below(mut self, val: f64) -> Self {
        self.warn_below = Some(val);
        self
    }

    /// Sets the maximum error threshold.
    #[must_use]
    pub fn with_error_above(mut self, val: f64) -> Self {
        self.error_above = Some(val);
        self
    }

    /// Sets the pooling strategy.
    #[must_use]
    pub fn with_pool(mut self, pool: PoolStrategy) -> Self {
        self.pool = pool;
        self
    }

    /// Evaluates a score against this threshold.
    #[must_use]
    pub fn evaluate(&self, score: f64) -> ThresholdResult {
        if let Some(min) = self.error_below {
            if score < min {
                return ThresholdResult::Error(format!(
                    "{}: {:.3} below minimum {:.3}",
                    self.metric, score, min
                ));
            }
        }
        if let Some(max) = self.error_above {
            if score > max {
                return ThresholdResult::Error(format!(
                    "{}: {:.3} above maximum {:.3}",
                    self.metric, score, max
                ));
            }
        }
        if let Some(warn_min) = self.warn_below {
            if score < warn_min {
                return ThresholdResult::Warning(format!(
                    "{}: {:.3} below recommended {:.3}",
                    self.metric, score, warn_min
                ));
            }
        }
        if let Some(warn_max) = self.warn_above {
            if score > warn_max {
                return ThresholdResult::Warning(format!(
                    "{}: {:.3} above recommended {:.3}",
                    self.metric, score, warn_max
                ));
            }
        }
        ThresholdResult::Pass
    }
}

/// Result of evaluating a score against a threshold.
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdResult {
    /// Score is within acceptable limits
    Pass,
    /// Score triggered a warning
    Warning(String),
    /// Score triggered an error
    Error(String),
}

/// A complete quality assessment preset.
#[derive(Debug, Clone)]
pub struct QualityPreset {
    /// Preset name
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Metric thresholds
    pub thresholds: Vec<MetricThreshold>,
    /// Default pooling strategy if metric doesn't specify one
    pub default_pool: PoolStrategy,
    /// Extra tags for identification
    pub tags: HashMap<String, String>,
}

impl QualityPreset {
    /// Creates a new preset.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            thresholds: Vec::new(),
            default_pool: PoolStrategy::Mean,
            tags: HashMap::new(),
        }
    }

    /// Adds a metric threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: MetricThreshold) -> Self {
        self.thresholds.push(threshold);
        self
    }

    /// Sets the default pooling strategy.
    #[must_use]
    pub fn with_default_pool(mut self, pool: PoolStrategy) -> Self {
        self.default_pool = pool;
        self
    }

    /// Adds a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Returns the number of required metrics.
    #[must_use]
    pub fn required_metric_count(&self) -> usize {
        self.thresholds.iter().filter(|t| t.required).count()
    }

    /// Evaluates a map of `metric_name` -> score against the preset.
    #[must_use]
    pub fn evaluate(&self, scores: &HashMap<String, f64>) -> Vec<ThresholdResult> {
        let mut results = Vec::new();
        for threshold in &self.thresholds {
            if let Some(&score) = scores.get(&threshold.metric) {
                results.push(threshold.evaluate(score));
            } else if threshold.required {
                results.push(ThresholdResult::Error(format!(
                    "{}: metric not computed",
                    threshold.metric
                )));
            }
        }
        results
    }

    /// Returns true if all evaluations pass.
    #[must_use]
    pub fn all_pass(&self, scores: &HashMap<String, f64>) -> bool {
        self.evaluate(scores)
            .iter()
            .all(|r| matches!(r, ThresholdResult::Pass))
    }
}

/// Returns a built-in preset by name.
#[must_use]
pub fn builtin_preset(name: PresetName) -> QualityPreset {
    match name {
        PresetName::Broadcast => QualityPreset::new("Broadcast", "EBU/SMPTE broadcast delivery QC")
            .with_threshold(MetricThreshold::min_required("PSNR", 30.0).with_warn_below(35.0))
            .with_threshold(MetricThreshold::min_required("SSIM", 0.90).with_warn_below(0.95))
            .with_threshold(MetricThreshold::min_required("VMAF", 70.0).with_warn_below(80.0))
            .with_default_pool(PoolStrategy::Percentile(5)),

        PresetName::Streaming => QualityPreset::new("Streaming", "OTT adaptive streaming delivery")
            .with_threshold(MetricThreshold::min_required("VMAF", 60.0).with_warn_below(70.0))
            .with_threshold(MetricThreshold::min_required("SSIM", 0.85).with_warn_below(0.90))
            .with_default_pool(PoolStrategy::Mean),

        PresetName::Archival => {
            QualityPreset::new("Archival", "High-quality archival preservation")
                .with_threshold(MetricThreshold::min_required("PSNR", 40.0).with_warn_below(45.0))
                .with_threshold(MetricThreshold::min_required("SSIM", 0.97).with_warn_below(0.99))
                .with_threshold(MetricThreshold::min_required("VMAF", 90.0).with_warn_below(95.0))
                .with_default_pool(PoolStrategy::Min)
        }

        PresetName::Dailies => QualityPreset::new("Dailies", "Fast screening for editorial review")
            .with_threshold(MetricThreshold::min_required("PSNR", 25.0))
            .with_threshold(MetricThreshold::min_required("SSIM", 0.80))
            .with_default_pool(PoolStrategy::Mean),

        PresetName::PostProduction => {
            QualityPreset::new("PostProduction", "Full post-production QC pass")
                .with_threshold(MetricThreshold::min_required("PSNR", 35.0).with_warn_below(40.0))
                .with_threshold(MetricThreshold::min_required("SSIM", 0.93).with_warn_below(0.96))
                .with_threshold(MetricThreshold::min_required("VMAF", 80.0).with_warn_below(90.0))
                .with_default_pool(PoolStrategy::HarmonicMean)
        }

        PresetName::Ugc => QualityPreset::new("UGC", "User-generated content ingest")
            .with_threshold(MetricThreshold::min_required("PSNR", 20.0).with_warn_below(25.0))
            .with_threshold(MetricThreshold::min_required("SSIM", 0.70).with_warn_below(0.80))
            .with_default_pool(PoolStrategy::Mean),
    }
}

/// Lists all built-in preset names.
#[must_use]
pub fn list_builtin_presets() -> Vec<PresetName> {
    vec![
        PresetName::Broadcast,
        PresetName::Streaming,
        PresetName::Archival,
        PresetName::Dailies,
        PresetName::PostProduction,
        PresetName::Ugc,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_strategy_mean() {
        let vals = vec![2.0, 4.0, 6.0];
        assert!((PoolStrategy::Mean.apply(&vals) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_pool_strategy_min() {
        let vals = vec![5.0, 3.0, 8.0];
        assert!((PoolStrategy::Min.apply(&vals) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_pool_strategy_harmonic() {
        let vals = vec![2.0, 2.0, 2.0];
        assert!((PoolStrategy::HarmonicMean.apply(&vals) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_pool_strategy_empty() {
        assert_eq!(PoolStrategy::Mean.apply(&[]), 0.0);
    }

    #[test]
    fn test_threshold_pass() {
        let t = MetricThreshold::min_required("PSNR", 30.0);
        assert_eq!(t.evaluate(35.0), ThresholdResult::Pass);
    }

    #[test]
    fn test_threshold_error_below() {
        let t = MetricThreshold::min_required("PSNR", 30.0);
        assert!(matches!(t.evaluate(25.0), ThresholdResult::Error(_)));
    }

    #[test]
    fn test_threshold_warn_below() {
        let t = MetricThreshold::min_required("PSNR", 30.0).with_warn_below(35.0);
        assert!(matches!(t.evaluate(32.0), ThresholdResult::Warning(_)));
    }

    #[test]
    fn test_threshold_error_above() {
        let t = MetricThreshold::min_required("Noise", 0.0).with_error_above(10.0);
        assert!(matches!(t.evaluate(15.0), ThresholdResult::Error(_)));
    }

    #[test]
    fn test_preset_broadcast() {
        let preset = builtin_preset(PresetName::Broadcast);
        assert_eq!(preset.name, "Broadcast");
        assert!(preset.thresholds.len() >= 3);
    }

    #[test]
    fn test_preset_evaluate_all_pass() {
        let preset = builtin_preset(PresetName::Dailies);
        let mut scores = HashMap::new();
        scores.insert("PSNR".to_string(), 40.0);
        scores.insert("SSIM".to_string(), 0.95);
        assert!(preset.all_pass(&scores));
    }

    #[test]
    fn test_preset_evaluate_fail() {
        let preset = builtin_preset(PresetName::Archival);
        let mut scores = HashMap::new();
        scores.insert("PSNR".to_string(), 25.0); // way below 40
        scores.insert("SSIM".to_string(), 0.99);
        scores.insert("VMAF".to_string(), 95.0);
        assert!(!preset.all_pass(&scores));
    }

    #[test]
    fn test_preset_required_metric_count() {
        let preset = builtin_preset(PresetName::Broadcast);
        assert_eq!(preset.required_metric_count(), 3);
    }

    #[test]
    fn test_list_builtin_presets() {
        let names = list_builtin_presets();
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn test_custom_preset() {
        let preset = QualityPreset::new("Custom", "Custom test")
            .with_threshold(MetricThreshold::min_required("MyMetric", 50.0))
            .with_tag("version", "1");
        assert_eq!(
            preset.tags.get("version").expect("should succeed in test"),
            "1"
        );
        assert_eq!(preset.required_metric_count(), 1);
    }
}
