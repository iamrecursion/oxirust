//! Automated quality gating for media processing pipelines.
//!
//! Provides QC pass/fail rules, configurable thresholds, and remediation hints
//! for automated quality control decisions.

#![allow(dead_code)]

use std::collections::HashMap;

/// Result of a quality gate evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum GateVerdict {
    /// Content passed all quality checks.
    Pass,
    /// Content failed one or more checks.
    Fail(Vec<QualityIssue>),
    /// Content passed but has warnings.
    PassWithWarnings(Vec<QualityIssue>),
}

impl GateVerdict {
    /// Returns true if the gate passed (with or without warnings).
    pub fn is_pass(&self) -> bool {
        matches!(self, GateVerdict::Pass | GateVerdict::PassWithWarnings(_))
    }

    /// Returns true if the gate failed.
    pub fn is_fail(&self) -> bool {
        matches!(self, GateVerdict::Fail(_))
    }

    /// Collect all issues from any verdict variant.
    pub fn issues(&self) -> &[QualityIssue] {
        match self {
            GateVerdict::Pass => &[],
            GateVerdict::Fail(issues) | GateVerdict::PassWithWarnings(issues) => issues.as_slice(),
        }
    }
}

/// Severity of a quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueSeverity {
    /// Informational hint.
    Info,
    /// Warning that does not fail the gate.
    Warning,
    /// Error that causes gate failure.
    Error,
}

/// A specific quality issue found during evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityIssue {
    /// Human-readable description.
    pub description: String,
    /// Severity level.
    pub severity: IssueSeverity,
    /// Remediation hint.
    pub remediation: String,
    /// Measured value that triggered the issue.
    pub measured_value: f64,
    /// Threshold that was violated.
    pub threshold: f64,
}

impl QualityIssue {
    /// Create a new quality issue.
    pub fn new(
        description: impl Into<String>,
        severity: IssueSeverity,
        remediation: impl Into<String>,
        measured_value: f64,
        threshold: f64,
    ) -> Self {
        Self {
            description: description.into(),
            severity,
            remediation: remediation.into(),
            measured_value,
            threshold,
        }
    }
}

/// Thresholds for a single quality metric.
#[derive(Debug, Clone)]
pub struct MetricThreshold {
    /// Minimum acceptable value (None = no lower bound).
    pub min: Option<f64>,
    /// Maximum acceptable value (None = no upper bound).
    pub max: Option<f64>,
    /// If violated, issue is error (else warning).
    pub severity: IssueSeverity,
    /// Remediation hint text.
    pub remediation: String,
}

impl MetricThreshold {
    /// Create a threshold requiring a minimum value.
    pub fn at_least(min: f64, severity: IssueSeverity, remediation: impl Into<String>) -> Self {
        Self {
            min: Some(min),
            max: None,
            severity,
            remediation: remediation.into(),
        }
    }

    /// Create a threshold requiring a maximum value.
    pub fn at_most(max: f64, severity: IssueSeverity, remediation: impl Into<String>) -> Self {
        Self {
            min: None,
            max: Some(max),
            severity,
            remediation: remediation.into(),
        }
    }

    /// Create a threshold with both min and max.
    pub fn between(
        min: f64,
        max: f64,
        severity: IssueSeverity,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
            severity,
            remediation: remediation.into(),
        }
    }

    /// Evaluate this threshold against a measured value.
    pub fn evaluate(&self, name: &str, value: f64) -> Option<QualityIssue> {
        if let Some(min) = self.min {
            if value < min {
                return Some(QualityIssue::new(
                    format!("{name} ({value:.3}) is below minimum ({min:.3})"),
                    self.severity,
                    self.remediation.clone(),
                    value,
                    min,
                ));
            }
        }
        if let Some(max) = self.max {
            if value > max {
                return Some(QualityIssue::new(
                    format!("{name} ({value:.3}) exceeds maximum ({max:.3})"),
                    self.severity,
                    self.remediation.clone(),
                    value,
                    max,
                ));
            }
        }
        None
    }
}

/// Configuration for the quality gate.
#[derive(Debug, Clone)]
pub struct QualityGateConfig {
    /// Named metric thresholds.
    pub thresholds: HashMap<String, MetricThreshold>,
    /// Whether warnings alone cause failure.
    pub warnings_cause_failure: bool,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();

        thresholds.insert(
            "video_bitrate_kbps".to_string(),
            MetricThreshold::at_least(500.0, IssueSeverity::Error, "Increase video bitrate"),
        );
        thresholds.insert(
            "audio_bitrate_kbps".to_string(),
            MetricThreshold::at_least(64.0, IssueSeverity::Error, "Increase audio bitrate"),
        );
        thresholds.insert(
            "psnr_db".to_string(),
            MetricThreshold::at_least(
                30.0,
                IssueSeverity::Warning,
                "Reduce compression or re-encode",
            ),
        );
        thresholds.insert(
            "loudness_lufs".to_string(),
            MetricThreshold::between(
                -23.0,
                -16.0,
                IssueSeverity::Warning,
                "Normalize audio loudness",
            ),
        );
        thresholds.insert(
            "true_peak_dbtp".to_string(),
            MetricThreshold::at_most(-1.0, IssueSeverity::Error, "Apply true-peak limiter"),
        );

        Self {
            thresholds,
            warnings_cause_failure: false,
        }
    }
}

/// Metric measurements to evaluate.
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Named metric values.
    pub values: HashMap<String, f64>,
}

impl QualityMetrics {
    /// Create metrics from a list of (name, value) pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (impl Into<String>, f64)>) -> Self {
        let mut values = HashMap::new();
        for (name, value) in pairs {
            values.insert(name.into(), value);
        }
        Self { values }
    }

    /// Insert a metric value.
    pub fn insert(&mut self, name: impl Into<String>, value: f64) {
        self.values.insert(name.into(), value);
    }
}

/// The quality gate evaluator.
#[derive(Debug, Clone)]
pub struct QualityGate {
    config: QualityGateConfig,
}

impl QualityGate {
    /// Create a new quality gate.
    pub fn new(config: QualityGateConfig) -> Self {
        Self { config }
    }

    /// Evaluate metrics against configured thresholds.
    pub fn evaluate(&self, metrics: &QualityMetrics) -> GateVerdict {
        let mut errors: Vec<QualityIssue> = Vec::new();
        let mut warnings: Vec<QualityIssue> = Vec::new();

        for (name, threshold) in &self.config.thresholds {
            if let Some(&value) = metrics.values.get(name) {
                if let Some(issue) = threshold.evaluate(name, value) {
                    match issue.severity {
                        IssueSeverity::Error => errors.push(issue),
                        IssueSeverity::Warning => warnings.push(issue),
                        IssueSeverity::Info => warnings.push(issue),
                    }
                }
            }
        }

        if !errors.is_empty() {
            let mut all = errors;
            all.extend(warnings);
            return GateVerdict::Fail(all);
        }

        if self.config.warnings_cause_failure && !warnings.is_empty() {
            return GateVerdict::Fail(warnings);
        }

        if warnings.is_empty() {
            GateVerdict::Pass
        } else {
            GateVerdict::PassWithWarnings(warnings)
        }
    }

    /// Add or update a threshold.
    pub fn set_threshold(&mut self, name: impl Into<String>, threshold: MetricThreshold) {
        self.config.thresholds.insert(name.into(), threshold);
    }
}

impl Default for QualityGate {
    fn default() -> Self {
        Self::new(QualityGateConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gate() -> QualityGate {
        QualityGate::default()
    }

    fn passing_metrics() -> QualityMetrics {
        QualityMetrics::from_pairs([
            ("video_bitrate_kbps", 2000.0),
            ("audio_bitrate_kbps", 128.0),
            ("psnr_db", 38.0),
            ("loudness_lufs", -20.0),
            ("true_peak_dbtp", -2.0),
        ])
    }

    #[test]
    fn test_all_passing_metrics_returns_pass() {
        let gate = default_gate();
        let verdict = gate.evaluate(&passing_metrics());
        assert!(verdict.is_pass());
    }

    #[test]
    fn test_low_video_bitrate_causes_failure() {
        let gate = default_gate();
        let mut m = passing_metrics();
        m.insert("video_bitrate_kbps", 100.0);
        let verdict = gate.evaluate(&m);
        assert!(verdict.is_fail());
    }

    #[test]
    fn test_low_audio_bitrate_causes_failure() {
        let gate = default_gate();
        let mut m = passing_metrics();
        m.insert("audio_bitrate_kbps", 32.0);
        let verdict = gate.evaluate(&m);
        assert!(verdict.is_fail());
    }

    #[test]
    fn test_low_psnr_causes_warning_not_error() {
        let gate = default_gate();
        let mut m = passing_metrics();
        m.insert("psnr_db", 25.0);
        let verdict = gate.evaluate(&m);
        // psnr threshold is Warning severity, so gate passes with warning
        assert!(verdict.is_pass());
        assert!(!verdict.issues().is_empty());
    }

    #[test]
    fn test_true_peak_too_high_causes_failure() {
        let gate = default_gate();
        let mut m = passing_metrics();
        m.insert("true_peak_dbtp", 0.0); // above -1.0 max
        let verdict = gate.evaluate(&m);
        assert!(verdict.is_fail());
    }

    #[test]
    fn test_loudness_out_of_range_causes_warning() {
        let gate = default_gate();
        let mut m = passing_metrics();
        m.insert("loudness_lufs", -30.0); // below -23.0 min
        let verdict = gate.evaluate(&m);
        // loudness threshold is Warning
        assert!(verdict.is_pass());
        assert!(verdict
            .issues()
            .iter()
            .any(|i| i.description.contains("loudness_lufs")));
    }

    #[test]
    fn test_warnings_cause_failure_mode() {
        let config = QualityGateConfig {
            warnings_cause_failure: true,
            ..Default::default()
        };
        let gate = QualityGate::new(config);
        let mut m = passing_metrics();
        m.insert("psnr_db", 25.0); // triggers warning
        let verdict = gate.evaluate(&m);
        assert!(verdict.is_fail());
    }

    #[test]
    fn test_missing_metric_not_evaluated() {
        let gate = default_gate();
        // Only provide some metrics; missing ones are ignored
        let m = QualityMetrics::from_pairs([("video_bitrate_kbps", 2000.0)]);
        let verdict = gate.evaluate(&m);
        assert!(verdict.is_pass());
    }

    #[test]
    fn test_issue_remediation_present() {
        let gate = default_gate();
        let mut m = passing_metrics();
        m.insert("video_bitrate_kbps", 100.0);
        let verdict = gate.evaluate(&m);
        assert!(verdict.issues().iter().any(|i| !i.remediation.is_empty()));
    }

    #[test]
    fn test_metric_threshold_at_least() {
        let t = MetricThreshold::at_least(10.0, IssueSeverity::Error, "fix it");
        assert!(t.evaluate("x", 9.9).is_some());
        assert!(t.evaluate("x", 10.0).is_none());
        assert!(t.evaluate("x", 100.0).is_none());
    }

    #[test]
    fn test_metric_threshold_at_most() {
        let t = MetricThreshold::at_most(5.0, IssueSeverity::Warning, "reduce it");
        assert!(t.evaluate("x", 5.1).is_some());
        assert!(t.evaluate("x", 5.0).is_none());
        assert!(t.evaluate("x", 0.0).is_none());
    }

    #[test]
    fn test_metric_threshold_between() {
        let t = MetricThreshold::between(1.0, 10.0, IssueSeverity::Error, "adjust it");
        assert!(t.evaluate("x", 0.5).is_some());
        assert!(t.evaluate("x", 5.0).is_none());
        assert!(t.evaluate("x", 10.5).is_some());
    }

    #[test]
    fn test_gate_verdict_is_pass() {
        assert!(GateVerdict::Pass.is_pass());
        assert!(GateVerdict::PassWithWarnings(vec![]).is_pass());
        assert!(!GateVerdict::Fail(vec![]).is_pass());
    }

    #[test]
    fn test_gate_verdict_is_fail() {
        assert!(GateVerdict::Fail(vec![]).is_fail());
        assert!(!GateVerdict::Pass.is_fail());
    }

    #[test]
    fn test_custom_threshold_added() {
        let mut gate = default_gate();
        gate.set_threshold(
            "frame_rate",
            MetricThreshold::at_least(24.0, IssueSeverity::Error, "Increase frame rate"),
        );
        let mut m = passing_metrics();
        m.insert("frame_rate", 10.0);
        let verdict = gate.evaluate(&m);
        assert!(verdict.is_fail());
    }
}
