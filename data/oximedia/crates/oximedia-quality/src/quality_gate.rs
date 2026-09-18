//! Quality gate evaluation for enforcing minimum quality thresholds.
//!
//! Two quality-gate flavours are provided:
//!
//! 1. [`QualityGate`] — generic threshold-based gate for any named metric.
//! 2. [`PipelineQualityGate`] — opinionated CI/CD gate that works directly
//!    with a [`TemporalQualityAnalysisReport`] and provides broadcast /
//!    streaming / preview presets.
//!
//! [`TemporalQualityAnalysisReport`]:
//!     crate::temporal_quality::TemporalQualityAnalysisReport

#![allow(dead_code)]

use crate::temporal_quality::TemporalQualityAnalysisReport;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A minimum (or maximum) threshold for a named quality metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateThreshold {
    /// Name of the metric this threshold applies to (e.g. `"vmaf"`, `"psnr"`).
    pub metric: String,
    /// Minimum acceptable score.  If `None`, no lower bound is applied.
    pub min_score: Option<f64>,
    /// Maximum acceptable score.  If `None`, no upper bound is applied.
    pub max_score: Option<f64>,
    /// Human-readable description of this threshold.
    pub description: String,
}

impl GateThreshold {
    /// Creates a lower-bound threshold: the metric must be ≥ `min_score`.
    #[must_use]
    pub fn at_least(metric: impl Into<String>, min_score: f64) -> Self {
        Self {
            metric: metric.into(),
            min_score: Some(min_score),
            max_score: None,
            description: String::new(),
        }
    }

    /// Creates an upper-bound threshold: the metric must be ≤ `max_score`.
    #[must_use]
    pub fn at_most(metric: impl Into<String>, max_score: f64) -> Self {
        Self {
            metric: metric.into(),
            min_score: None,
            max_score: Some(max_score),
            description: String::new(),
        }
    }

    /// Attaches a human-readable description, consuming and returning `self`.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Evaluates the threshold against a single numeric score.
    ///
    /// Returns `true` if the score satisfies all configured bounds.
    #[must_use]
    pub fn passes(&self, score: f64) -> bool {
        if let Some(min) = self.min_score {
            if score < min {
                return false;
            }
        }
        if let Some(max) = self.max_score {
            if score > max {
                return false;
            }
        }
        true
    }
}

/// Detailed outcome of evaluating a single threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdOutcome {
    /// The metric name.
    pub metric: String,
    /// Whether the threshold was satisfied.
    pub passed: bool,
    /// Actual score supplied to the gate.
    pub actual_score: f64,
    /// Threshold that was applied.
    pub threshold: GateThreshold,
}

/// Aggregate result of evaluating all thresholds in a [`QualityGate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// `true` iff every threshold passed.
    pub passed: bool,
    /// Per-threshold outcomes (in evaluation order).
    pub outcomes: Vec<ThresholdOutcome>,
}

impl GateResult {
    /// Returns the outcomes for thresholds that failed.
    #[must_use]
    pub fn failures(&self) -> Vec<&ThresholdOutcome> {
        self.outcomes.iter().filter(|o| !o.passed).collect()
    }

    /// Returns the number of thresholds that failed.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.outcomes.iter().filter(|o| !o.passed).count()
    }

    /// Returns the number of thresholds that passed.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.outcomes.iter().filter(|o| o.passed).count()
    }
}

/// A configurable quality gate composed of one or more [`GateThreshold`]s.
///
/// Call [`QualityGate::evaluate`] with a map of metric scores to obtain a
/// [`GateResult`].
#[derive(Debug, Clone, Default)]
pub struct QualityGate {
    thresholds: Vec<GateThreshold>,
}

impl QualityGate {
    /// Creates an empty quality gate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a threshold to this gate.
    pub fn add_threshold(&mut self, threshold: GateThreshold) {
        self.thresholds.push(threshold);
    }

    /// Builder-style threshold addition.
    #[must_use]
    pub fn with_threshold(mut self, threshold: GateThreshold) -> Self {
        self.add_threshold(threshold);
        self
    }

    /// Returns the number of configured thresholds.
    #[must_use]
    pub fn threshold_count(&self) -> usize {
        self.thresholds.len()
    }

    /// Evaluates the gate against the supplied metric scores.
    ///
    /// Metrics that appear in the gate but are absent from `scores` are treated
    /// as `0.0`.
    #[must_use]
    pub fn evaluate(&self, scores: &HashMap<String, f64>) -> GateResult {
        let mut outcomes = Vec::with_capacity(self.thresholds.len());
        let mut all_passed = true;

        for threshold in &self.thresholds {
            let actual_score = scores.get(&threshold.metric).copied().unwrap_or(0.0);
            let passed = threshold.passes(actual_score);
            if !passed {
                all_passed = false;
            }
            outcomes.push(ThresholdOutcome {
                metric: threshold.metric.clone(),
                passed,
                actual_score,
                threshold: threshold.clone(),
            });
        }

        GateResult {
            passed: all_passed,
            outcomes,
        }
    }

    /// Convenience method: returns `true` iff all thresholds pass for `scores`.
    #[must_use]
    pub fn passes(&self, scores: &HashMap<String, f64>) -> bool {
        self.evaluate(scores).passed
    }
}

// ---------------------------------------------------------------------------
// PipelineQualityGate — CI/CD oriented gate
// ---------------------------------------------------------------------------

/// Result of evaluating a [`PipelineQualityGate`].
#[derive(Debug, Clone)]
pub struct QualityGateResult {
    /// Whether all conditions were satisfied.
    pub passed: bool,
    /// Human-readable descriptions of each violated condition.
    pub violations: Vec<String>,
    /// Name of the gate that produced this result.
    pub gate_name: String,
}

impl QualityGateResult {
    /// Returns `true` iff the gate passed.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        self.passed
    }

    /// Returns the number of violations.
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

/// A CI/CD quality gate that evaluates a [`TemporalQualityAnalysisReport`]
/// against a set of named thresholds.
///
/// # Presets
///
/// | Preset      | Min PSNR | Min SSIM | Max PSNR drop |
/// |-------------|----------|----------|---------------|
/// | broadcast   | 40 dB    | 0.98     | 3 dB          |
/// | streaming   | 35 dB    | 0.95     | 5 dB          |
/// | preview     | 30 dB    | 0.90     | 8 dB          |
#[derive(Debug, Clone)]
pub struct PipelineQualityGate {
    /// Human-readable name of this gate configuration.
    pub name: String,
    /// Minimum acceptable mean PSNR (dB). `None` disables the check.
    pub min_psnr: Option<f32>,
    /// Minimum acceptable mean SSIM in \[0, 1\]. `None` disables the check.
    pub min_ssim: Option<f32>,
    /// Minimum acceptable VMAF estimate. `None` disables the check.
    pub min_vmaf: Option<f32>,
    /// Maximum allowed drop from best-frame PSNR to worst-frame PSNR.
    /// `None` disables the check.
    pub max_psnr_drop: Option<f32>,
}

impl PipelineQualityGate {
    /// Create a gate for broadcast delivery (highest quality).
    ///
    /// Requirements: mean PSNR ≥ 40 dB, mean SSIM ≥ 0.98, PSNR drop ≤ 3 dB.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            name: "broadcast".to_string(),
            min_psnr: Some(40.0),
            min_ssim: Some(0.98),
            min_vmaf: None,
            max_psnr_drop: Some(3.0),
        }
    }

    /// Create a gate for adaptive streaming (standard quality).
    ///
    /// Requirements: mean PSNR ≥ 35 dB, mean SSIM ≥ 0.95, PSNR drop ≤ 5 dB.
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            name: "streaming".to_string(),
            min_psnr: Some(35.0),
            min_ssim: Some(0.95),
            min_vmaf: None,
            max_psnr_drop: Some(5.0),
        }
    }

    /// Create a gate for preview/proxy files (relaxed quality).
    ///
    /// Requirements: mean PSNR ≥ 30 dB, mean SSIM ≥ 0.90, PSNR drop ≤ 8 dB.
    #[must_use]
    pub fn preview() -> Self {
        Self {
            name: "preview".to_string(),
            min_psnr: Some(30.0),
            min_ssim: Some(0.90),
            min_vmaf: None,
            max_psnr_drop: Some(8.0),
        }
    }

    /// Create a fully custom gate with all thresholds explicitly specified.
    #[must_use]
    pub fn custom(
        name: impl Into<String>,
        min_psnr: Option<f32>,
        min_ssim: Option<f32>,
        min_vmaf: Option<f32>,
        max_psnr_drop: Option<f32>,
    ) -> Self {
        Self {
            name: name.into(),
            min_psnr,
            min_ssim,
            min_vmaf,
            max_psnr_drop,
        }
    }

    /// Evaluate this gate against a [`TemporalQualityAnalysisReport`].
    #[must_use]
    pub fn check(&self, report: &TemporalQualityAnalysisReport) -> QualityGateResult {
        let mut violations = Vec::new();

        // PSNR mean threshold
        if let Some(min) = self.min_psnr {
            if report.psnr_stats.mean < min {
                violations.push(format!(
                    "mean PSNR {:.2} dB < required {:.2} dB",
                    report.psnr_stats.mean, min
                ));
            }
        }

        // SSIM mean threshold
        if let Some(min) = self.min_ssim {
            if report.ssim_stats.mean < min {
                violations.push(format!(
                    "mean SSIM {:.4} < required {:.4}",
                    report.ssim_stats.mean, min
                ));
            }
        }

        // VMAF estimate threshold
        if let Some(min) = self.min_vmaf {
            if report.overall_vmaf_estimate < min {
                violations.push(format!(
                    "VMAF estimate {:.2} < required {:.2}",
                    report.overall_vmaf_estimate, min
                ));
            }
        }

        // PSNR drop (best − worst)
        if let Some(max_drop) = self.max_psnr_drop {
            let drop = report.psnr_stats.max - report.psnr_stats.min;
            if drop > max_drop {
                violations.push(format!("PSNR drop {drop:.2} dB > allowed {max_drop:.2} dB"));
            }
        }

        QualityGateResult {
            passed: violations.is_empty(),
            violations,
            gate_name: self.name.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// CompositeQualityGate — multi-metric gate with AND/OR logic
// ---------------------------------------------------------------------------

/// Logical combination mode for composite gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositeMode {
    /// All sub-gates must pass (logical AND).
    All,
    /// At least one sub-gate must pass (logical OR).
    Any,
    /// At least `n` sub-gates must pass.
    AtLeast(usize),
}

/// Result of evaluating a [`CompositeQualityGate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeGateResult {
    /// Whether the composite gate passed overall.
    pub passed: bool,
    /// The combination mode used.
    pub mode: CompositeMode,
    /// Per-sub-gate results (in evaluation order).
    pub sub_results: Vec<GateResult>,
    /// Number of sub-gates that passed.
    pub passed_count: usize,
    /// Total number of sub-gates.
    pub total_count: usize,
}

impl CompositeGateResult {
    /// Returns sub-gate results that failed.
    #[must_use]
    pub fn failed_sub_gates(&self) -> Vec<&GateResult> {
        self.sub_results.iter().filter(|r| !r.passed).collect()
    }

    /// Returns a human-readable summary of the composite evaluation.
    #[must_use]
    pub fn summary(&self) -> String {
        let status = if self.passed { "PASS" } else { "FAIL" };
        let mode_desc = match self.mode {
            CompositeMode::All => "ALL".to_string(),
            CompositeMode::Any => "ANY".to_string(),
            CompositeMode::AtLeast(n) => format!("AT_LEAST({n})"),
        };
        format!(
            "[{status}] mode={mode_desc} passed={}/{} total_violations={}",
            self.passed_count,
            self.total_count,
            self.sub_results
                .iter()
                .map(|r| r.failure_count())
                .sum::<usize>()
        )
    }
}

/// A composite gate that combines multiple [`QualityGate`]s with configurable logic.
///
/// # Example
///
/// ```
/// use oximedia_quality::quality_gate::{
///     CompositeQualityGate, CompositeMode, QualityGate, GateThreshold,
/// };
/// use std::collections::HashMap;
///
/// let ssim_gate = QualityGate::new()
///     .with_threshold(GateThreshold::at_least("ssim", 0.95));
/// let vmaf_gate = QualityGate::new()
///     .with_threshold(GateThreshold::at_least("vmaf", 80.0));
///
/// let composite = CompositeQualityGate::new(CompositeMode::All)
///     .with_gate(ssim_gate)
///     .with_gate(vmaf_gate);
///
/// let mut scores = HashMap::new();
/// scores.insert("ssim".to_string(), 0.97);
/// scores.insert("vmaf".to_string(), 85.0);
/// let result = composite.evaluate(&scores);
/// assert!(result.passed);
/// ```
#[derive(Debug, Clone)]
pub struct CompositeQualityGate {
    /// How sub-gates are combined.
    mode: CompositeMode,
    /// The list of sub-gates.
    gates: Vec<QualityGate>,
}

impl CompositeQualityGate {
    /// Creates a new composite gate with the given combination mode.
    #[must_use]
    pub fn new(mode: CompositeMode) -> Self {
        Self {
            mode,
            gates: Vec::new(),
        }
    }

    /// Adds a sub-gate to this composite.
    pub fn add_gate(&mut self, gate: QualityGate) {
        self.gates.push(gate);
    }

    /// Builder-style sub-gate addition.
    #[must_use]
    pub fn with_gate(mut self, gate: QualityGate) -> Self {
        self.add_gate(gate);
        self
    }

    /// Returns the number of sub-gates.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Returns the combination mode.
    #[must_use]
    pub fn mode(&self) -> CompositeMode {
        self.mode
    }

    /// Evaluates all sub-gates against the supplied metric scores.
    #[must_use]
    pub fn evaluate(&self, scores: &HashMap<String, f64>) -> CompositeGateResult {
        let sub_results: Vec<GateResult> = self.gates.iter().map(|g| g.evaluate(scores)).collect();
        let passed_count = sub_results.iter().filter(|r| r.passed).count();
        let total_count = sub_results.len();

        let passed = match self.mode {
            CompositeMode::All => sub_results.iter().all(|r| r.passed),
            CompositeMode::Any => {
                if sub_results.is_empty() {
                    true
                } else {
                    sub_results.iter().any(|r| r.passed)
                }
            }
            CompositeMode::AtLeast(n) => passed_count >= n,
        };

        CompositeGateResult {
            passed,
            mode: self.mode,
            sub_results,
            passed_count,
            total_count,
        }
    }

    /// Convenience: returns `true` iff the composite gate passes for `scores`.
    #[must_use]
    pub fn passes(&self, scores: &HashMap<String, f64>) -> bool {
        self.evaluate(scores).passed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_threshold_at_least_passes() {
        let t = GateThreshold::at_least("vmaf", 80.0);
        assert!(t.passes(90.0));
        assert!(t.passes(80.0));
    }

    #[test]
    fn test_threshold_at_least_fails() {
        let t = GateThreshold::at_least("vmaf", 80.0);
        assert!(!t.passes(79.9));
    }

    #[test]
    fn test_threshold_at_most_passes() {
        let t = GateThreshold::at_most("noise", 5.0);
        assert!(t.passes(3.0));
        assert!(t.passes(5.0));
    }

    #[test]
    fn test_threshold_at_most_fails() {
        let t = GateThreshold::at_most("noise", 5.0);
        assert!(!t.passes(5.1));
    }

    #[test]
    fn test_threshold_with_description() {
        let t = GateThreshold::at_least("vmaf", 90.0).with_description("Broadcast quality");
        assert_eq!(t.description, "Broadcast quality");
    }

    #[test]
    fn test_gate_all_pass() {
        let gate = QualityGate::new()
            .with_threshold(GateThreshold::at_least("vmaf", 80.0))
            .with_threshold(GateThreshold::at_least("psnr", 35.0));
        let result = gate.evaluate(&scores(&[("vmaf", 92.0), ("psnr", 40.0)]));
        assert!(result.passed);
        assert_eq!(result.failure_count(), 0);
        assert_eq!(result.pass_count(), 2);
    }

    #[test]
    fn test_gate_one_fail() {
        let gate = QualityGate::new()
            .with_threshold(GateThreshold::at_least("vmaf", 80.0))
            .with_threshold(GateThreshold::at_least("psnr", 35.0));
        let result = gate.evaluate(&scores(&[("vmaf", 92.0), ("psnr", 30.0)]));
        assert!(!result.passed);
        assert_eq!(result.failure_count(), 1);
        assert_eq!(result.failures()[0].metric, "psnr");
    }

    #[test]
    fn test_gate_missing_metric_treated_as_zero() {
        let gate = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 80.0));
        let result = gate.evaluate(&scores(&[]));
        assert!(!result.passed);
        assert_eq!(result.outcomes[0].actual_score, 0.0);
    }

    #[test]
    fn test_gate_passes_convenience() {
        let gate = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        assert!(gate.passes(&scores(&[("ssim", 0.95)])));
        assert!(!gate.passes(&scores(&[("ssim", 0.85)])));
    }

    #[test]
    fn test_gate_threshold_count() {
        let gate = QualityGate::new()
            .with_threshold(GateThreshold::at_least("a", 0.0))
            .with_threshold(GateThreshold::at_least("b", 0.0));
        assert_eq!(gate.threshold_count(), 2);
    }

    #[test]
    fn test_empty_gate_passes_anything() {
        let gate = QualityGate::new();
        assert!(gate.passes(&scores(&[])));
    }

    #[test]
    fn test_gate_result_failures_list() {
        let gate = QualityGate::new()
            .with_threshold(GateThreshold::at_least("vmaf", 80.0))
            .with_threshold(GateThreshold::at_least("psnr", 40.0));
        let result = gate.evaluate(&scores(&[("vmaf", 70.0), ("psnr", 38.0)]));
        assert_eq!(result.failures().len(), 2);
    }

    #[test]
    fn test_threshold_both_bounds() {
        let t = GateThreshold {
            metric: "brightness".to_string(),
            min_score: Some(0.1),
            max_score: Some(0.9),
            description: String::new(),
        };
        assert!(t.passes(0.5));
        assert!(!t.passes(0.05));
        assert!(!t.passes(0.95));
    }

    #[test]
    fn test_gate_result_pass_count() {
        let gate = QualityGate::new()
            .with_threshold(GateThreshold::at_least("a", 50.0))
            .with_threshold(GateThreshold::at_least("b", 50.0))
            .with_threshold(GateThreshold::at_least("c", 50.0));
        let result = gate.evaluate(&scores(&[("a", 60.0), ("b", 40.0), ("c", 70.0)]));
        assert_eq!(result.pass_count(), 2);
        assert_eq!(result.failure_count(), 1);
    }

    // ── PipelineQualityGate ────────────────────────────────────────────────

    use crate::temporal_quality::{QualityStats, TemporalQualityAnalysisReport};

    fn make_report(
        psnr_mean: f32,
        psnr_min: f32,
        psnr_max: f32,
        ssim_mean: f32,
        vmaf_est: f32,
    ) -> TemporalQualityAnalysisReport {
        TemporalQualityAnalysisReport {
            total_frames: 10,
            psnr_stats: QualityStats {
                mean: psnr_mean,
                min: psnr_min,
                max: psnr_max,
                std_dev: 1.0,
                frame_count: 10,
            },
            ssim_stats: QualityStats {
                mean: ssim_mean,
                min: ssim_mean - 0.01,
                max: ssim_mean + 0.01,
                std_dev: 0.005,
                frame_count: 10,
            },
            quality_drops: vec![],
            worst_frame: 0,
            best_frame: 9,
            overall_vmaf_estimate: vmaf_est,
        }
    }

    #[test]
    fn test_pipeline_gate_broadcast_passes() {
        let gate = PipelineQualityGate::broadcast();
        // Exactly at threshold for PSNR and SSIM, drop = 0
        let report = make_report(40.0, 40.0, 40.0, 0.98, 80.0);
        let result = gate.check(&report);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_pipeline_gate_broadcast_fails_psnr() {
        let gate = PipelineQualityGate::broadcast();
        let report = make_report(38.0, 38.0, 38.0, 0.99, 85.0);
        let result = gate.check(&report);
        assert!(!result.passed);
        assert!(
            result.violations.iter().any(|v| v.contains("PSNR")),
            "expected PSNR violation, got: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_pipeline_gate_broadcast_fails_ssim() {
        let gate = PipelineQualityGate::broadcast();
        let report = make_report(42.0, 42.0, 42.0, 0.97, 85.0);
        let result = gate.check(&report);
        assert!(!result.passed);
        assert!(
            result.violations.iter().any(|v| v.contains("SSIM")),
            "expected SSIM violation"
        );
    }

    #[test]
    fn test_pipeline_gate_broadcast_fails_psnr_drop() {
        let gate = PipelineQualityGate::broadcast();
        // drop = 48 - 42 = 6 > 3
        let report = make_report(45.0, 42.0, 48.0, 0.985, 88.0);
        let result = gate.check(&report);
        assert!(!result.passed);
        assert!(
            result.violations.iter().any(|v| v.contains("drop")),
            "expected drop violation"
        );
    }

    #[test]
    fn test_pipeline_gate_streaming_passes() {
        let gate = PipelineQualityGate::streaming();
        let report = make_report(37.0, 36.0, 38.0, 0.96, 70.0);
        let result = gate.check(&report);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_pipeline_gate_preview_passes() {
        let gate = PipelineQualityGate::preview();
        let report = make_report(32.0, 28.0, 34.0, 0.92, 55.0);
        let result = gate.check(&report);
        assert!(result.passed, "violations: {:?}", result.violations);
    }

    #[test]
    fn test_pipeline_gate_preview_fails_ssim() {
        let gate = PipelineQualityGate::preview();
        let report = make_report(32.0, 30.0, 33.0, 0.85, 50.0);
        let result = gate.check(&report);
        assert!(!result.passed);
    }

    #[test]
    fn test_pipeline_gate_name() {
        assert_eq!(PipelineQualityGate::broadcast().name, "broadcast");
        assert_eq!(PipelineQualityGate::streaming().name, "streaming");
        assert_eq!(PipelineQualityGate::preview().name, "preview");
    }

    #[test]
    fn test_pipeline_gate_result_gate_name() {
        let gate = PipelineQualityGate::streaming();
        let report = make_report(36.0, 35.0, 37.0, 0.96, 72.0);
        let result = gate.check(&report);
        assert_eq!(result.gate_name, "streaming");
    }

    #[test]
    fn test_pipeline_gate_custom_vmaf() {
        let gate = PipelineQualityGate::custom("my-gate", Some(35.0), Some(0.95), Some(70.0), None);
        // VMAF too low
        let report = make_report(38.0, 37.0, 39.0, 0.96, 65.0);
        let result = gate.check(&report);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("VMAF")));
    }

    #[test]
    fn test_pipeline_gate_multiple_violations() {
        let gate = PipelineQualityGate::broadcast();
        // Both PSNR and SSIM fail
        let report = make_report(30.0, 28.0, 32.0, 0.90, 60.0);
        let result = gate.check(&report);
        assert!(!result.passed);
        assert!(result.violation_count() >= 2, "expected >= 2 violations");
    }

    #[test]
    fn test_quality_gate_result_is_pass() {
        let gate = PipelineQualityGate::preview();
        let pass_report = make_report(35.0, 34.0, 36.0, 0.93, 65.0);
        let fail_report = make_report(20.0, 18.0, 22.0, 0.70, 30.0);
        assert!(gate.check(&pass_report).is_pass());
        assert!(!gate.check(&fail_report).is_pass());
    }

    // ── CompositeQualityGate ─────────────────────────────────────────────

    use super::{CompositeMode, CompositeQualityGate};

    #[test]
    fn test_composite_all_pass() {
        let g1 = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        let g2 = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 70.0));
        let composite = CompositeQualityGate::new(CompositeMode::All)
            .with_gate(g1)
            .with_gate(g2);
        let result = composite.evaluate(&scores(&[("ssim", 0.95), ("vmaf", 80.0)]));
        assert!(result.passed);
        assert_eq!(result.passed_count, 2);
        assert_eq!(result.total_count, 2);
    }

    #[test]
    fn test_composite_all_one_fails() {
        let g1 = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        let g2 = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 70.0));
        let composite = CompositeQualityGate::new(CompositeMode::All)
            .with_gate(g1)
            .with_gate(g2);
        let result = composite.evaluate(&scores(&[("ssim", 0.95), ("vmaf", 60.0)]));
        assert!(!result.passed);
        assert_eq!(result.passed_count, 1);
    }

    #[test]
    fn test_composite_any_one_passes() {
        let g1 = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        let g2 = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 70.0));
        let composite = CompositeQualityGate::new(CompositeMode::Any)
            .with_gate(g1)
            .with_gate(g2);
        let result = composite.evaluate(&scores(&[("ssim", 0.85), ("vmaf", 80.0)]));
        assert!(result.passed);
    }

    #[test]
    fn test_composite_any_none_passes() {
        let g1 = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        let g2 = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 70.0));
        let composite = CompositeQualityGate::new(CompositeMode::Any)
            .with_gate(g1)
            .with_gate(g2);
        let result = composite.evaluate(&scores(&[("ssim", 0.80), ("vmaf", 50.0)]));
        assert!(!result.passed);
    }

    #[test]
    fn test_composite_at_least_passes() {
        let g1 = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        let g2 = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 70.0));
        let g3 = QualityGate::new().with_threshold(GateThreshold::at_least("psnr", 35.0));
        let composite = CompositeQualityGate::new(CompositeMode::AtLeast(2))
            .with_gate(g1)
            .with_gate(g2)
            .with_gate(g3);
        let result = composite.evaluate(&scores(&[("ssim", 0.95), ("vmaf", 80.0), ("psnr", 30.0)]));
        assert!(result.passed);
        assert_eq!(result.passed_count, 2);
    }

    #[test]
    fn test_composite_at_least_fails() {
        let g1 = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        let g2 = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 70.0));
        let g3 = QualityGate::new().with_threshold(GateThreshold::at_least("psnr", 35.0));
        let composite = CompositeQualityGate::new(CompositeMode::AtLeast(3))
            .with_gate(g1)
            .with_gate(g2)
            .with_gate(g3);
        let result = composite.evaluate(&scores(&[("ssim", 0.95), ("vmaf", 80.0), ("psnr", 30.0)]));
        assert!(!result.passed);
    }

    #[test]
    fn test_composite_empty_all_passes() {
        let composite = CompositeQualityGate::new(CompositeMode::All);
        let result = composite.evaluate(&scores(&[]));
        assert!(result.passed);
    }

    #[test]
    fn test_composite_empty_any_passes() {
        let composite = CompositeQualityGate::new(CompositeMode::Any);
        let result = composite.evaluate(&scores(&[]));
        assert!(result.passed);
    }

    #[test]
    fn test_composite_gate_count() {
        let composite = CompositeQualityGate::new(CompositeMode::All)
            .with_gate(QualityGate::new())
            .with_gate(QualityGate::new());
        assert_eq!(composite.gate_count(), 2);
    }

    #[test]
    fn test_composite_mode_accessor() {
        let composite = CompositeQualityGate::new(CompositeMode::Any);
        assert_eq!(composite.mode(), CompositeMode::Any);
    }

    #[test]
    fn test_composite_passes_convenience() {
        let g = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 80.0));
        let composite = CompositeQualityGate::new(CompositeMode::All).with_gate(g);
        assert!(composite.passes(&scores(&[("vmaf", 90.0)])));
        assert!(!composite.passes(&scores(&[("vmaf", 70.0)])));
    }

    #[test]
    fn test_composite_failed_sub_gates() {
        let g1 = QualityGate::new().with_threshold(GateThreshold::at_least("ssim", 0.9));
        let g2 = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 70.0));
        let composite = CompositeQualityGate::new(CompositeMode::All)
            .with_gate(g1)
            .with_gate(g2);
        let result = composite.evaluate(&scores(&[("ssim", 0.85), ("vmaf", 80.0)]));
        let failed = result.failed_sub_gates();
        assert_eq!(failed.len(), 1);
    }

    #[test]
    fn test_composite_summary_format() {
        let g = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 80.0));
        let composite = CompositeQualityGate::new(CompositeMode::All).with_gate(g);
        let result = composite.evaluate(&scores(&[("vmaf", 90.0)]));
        let summary = result.summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("ALL"));
        assert!(summary.contains("1/1"));
    }

    #[test]
    fn test_composite_at_least_zero_always_passes() {
        let g = QualityGate::new().with_threshold(GateThreshold::at_least("vmaf", 999.0));
        let composite = CompositeQualityGate::new(CompositeMode::AtLeast(0)).with_gate(g);
        assert!(composite.passes(&scores(&[("vmaf", 0.0)])));
    }
}
