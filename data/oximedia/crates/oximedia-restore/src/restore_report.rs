//! Structured reporting of restoration operations.
//!
//! A [`RestoreReport`] captures every significant step carried out during a
//! restoration session, together with timing data and per-step metrics.
//! The companion [`RestoreReportBuilder`] provides a fluent API for
//! constructing reports incrementally.

#![allow(dead_code)]

use std::time::Duration;

/// Identifies a single step in a restoration pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreStepKind {
    /// DC offset removal.
    DcRemoval,
    /// Click / pop removal.
    ClickRemoval,
    /// Hiss reduction.
    HissReduction,
    /// Hum removal (50/60 Hz).
    HumRemoval,
    /// Declipping (peak restoration).
    Declipping,
    /// Wow / flutter correction.
    WowFlutterCorrection,
    /// Colour correction / fade restoration.
    ColorCorrection,
    /// Synthetic grain addition.
    GrainSynthesis,
    /// Video de-flicker.
    Deflicker,
    /// Telecine detection and removal.
    TelecineRemoval,
    /// Video upscaling.
    Upscale,
    /// Debanding.
    Deband,
    /// A custom step.
    Custom(String),
}

impl RestoreStepKind {
    /// Human-readable label for this step kind.
    pub fn label(&self) -> &str {
        match self {
            Self::DcRemoval => "DC Removal",
            Self::ClickRemoval => "Click Removal",
            Self::HissReduction => "Hiss Reduction",
            Self::HumRemoval => "Hum Removal",
            Self::Declipping => "Declipping",
            Self::WowFlutterCorrection => "Wow/Flutter Correction",
            Self::ColorCorrection => "Colour Correction",
            Self::GrainSynthesis => "Grain Synthesis",
            Self::Deflicker => "Deflicker",
            Self::TelecineRemoval => "Telecine Removal",
            Self::Upscale => "Upscale",
            Self::Deband => "Deband",
            Self::Custom(name) => name.as_str(),
        }
    }
}

/// Outcome of a single restoration step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// Step completed successfully.
    Success,
    /// Step completed with minor issues.
    PartialSuccess,
    /// Step was skipped (not applicable to this media).
    Skipped,
    /// Step failed.
    Failed,
}

/// Record of one step inside a restoration report.
#[derive(Debug, Clone)]
pub struct StepRecord {
    /// Kind of restoration step.
    pub kind: RestoreStepKind,
    /// Outcome of the step.
    pub outcome: StepOutcome,
    /// Wall-clock duration of this step.
    pub duration: Duration,
    /// Optional human-readable note.
    pub note: Option<String>,
}

/// Complete report of a restoration session.
#[derive(Debug, Clone)]
pub struct RestoreReport {
    /// Name or identifier for this report.
    pub name: String,
    /// Source file or asset that was restored.
    pub source: String,
    /// Ordered list of step records.
    steps: Vec<StepRecord>,
    /// Total wall-clock time for the session.
    pub total_duration: Duration,
}

impl RestoreReport {
    /// Number of steps recorded.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// All step records.
    pub fn steps(&self) -> &[StepRecord] {
        &self.steps
    }

    /// Count of steps with a given outcome.
    pub fn count_outcome(&self, outcome: StepOutcome) -> usize {
        self.steps.iter().filter(|s| s.outcome == outcome).count()
    }

    /// Success rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Both `Success` and `PartialSuccess` are counted as successes.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let ok = self
            .steps
            .iter()
            .filter(|s| {
                matches!(
                    s.outcome,
                    StepOutcome::Success | StepOutcome::PartialSuccess
                )
            })
            .count();
        ok as f64 / self.steps.len() as f64
    }

    /// Summed duration of all individual steps.
    pub fn steps_duration(&self) -> Duration {
        self.steps.iter().map(|s| s.duration).sum()
    }

    /// Generate a plain-text summary.
    pub fn summary(&self) -> String {
        let success = self.count_outcome(StepOutcome::Success);
        let partial = self.count_outcome(StepOutcome::PartialSuccess);
        let failed = self.count_outcome(StepOutcome::Failed);
        let skipped = self.count_outcome(StepOutcome::Skipped);
        format!(
            "Restore report '{}': {} steps | {} ok | {} partial | {} failed | {} skipped | {:.1}% success",
            self.name,
            self.steps.len(),
            success,
            partial,
            failed,
            skipped,
            self.success_rate() * 100.0,
        )
    }

    /// Return the names of all failed steps.
    pub fn failed_steps(&self) -> Vec<&str> {
        self.steps
            .iter()
            .filter(|s| s.outcome == StepOutcome::Failed)
            .map(|s| s.kind.label())
            .collect()
    }
}

/// Fluent builder for constructing a [`RestoreReport`].
#[derive(Debug, Default)]
pub struct RestoreReportBuilder {
    name: String,
    source: String,
    steps: Vec<StepRecord>,
    total_duration: Duration,
}

impl RestoreReportBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the report name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the source identifier.
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Record a step.
    pub fn add_step(
        mut self,
        kind: RestoreStepKind,
        outcome: StepOutcome,
        duration: Duration,
        note: Option<String>,
    ) -> Self {
        self.steps.push(StepRecord {
            kind,
            outcome,
            duration,
            note,
        });
        self
    }

    /// Record a successful step with the given duration.
    pub fn add_ok(self, kind: RestoreStepKind, duration: Duration) -> Self {
        self.add_step(kind, StepOutcome::Success, duration, None)
    }

    /// Record a failed step with a reason.
    pub fn add_fail(self, kind: RestoreStepKind, duration: Duration, reason: &str) -> Self {
        self.add_step(kind, StepOutcome::Failed, duration, Some(reason.to_owned()))
    }

    /// Set the total session duration.
    pub fn total_duration(mut self, dur: Duration) -> Self {
        self.total_duration = dur;
        self
    }

    /// Consume the builder and produce the report.
    pub fn build(self) -> RestoreReport {
        RestoreReport {
            name: self.name,
            source: self.source,
            steps: self.steps,
            total_duration: self.total_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> RestoreReport {
        RestoreReportBuilder::new()
            .name("test_report")
            .source("sample.wav")
            .add_ok(RestoreStepKind::DcRemoval, Duration::from_millis(100))
            .add_ok(RestoreStepKind::ClickRemoval, Duration::from_millis(500))
            .add_fail(
                RestoreStepKind::Declipping,
                Duration::from_millis(200),
                "no clipping detected",
            )
            .total_duration(Duration::from_millis(900))
            .build()
    }

    #[test]
    fn test_step_count() {
        let report = sample_report();
        assert_eq!(report.step_count(), 3);
    }

    #[test]
    fn test_count_outcome_success() {
        let report = sample_report();
        assert_eq!(report.count_outcome(StepOutcome::Success), 2);
    }

    #[test]
    fn test_count_outcome_failed() {
        let report = sample_report();
        assert_eq!(report.count_outcome(StepOutcome::Failed), 1);
    }

    #[test]
    fn test_success_rate() {
        let report = sample_report();
        let rate = report.success_rate();
        // 2 out of 3
        assert!((rate - 2.0 / 3.0).abs() < 1e-9, "rate={rate}");
    }

    #[test]
    fn test_empty_report_success_rate() {
        let report = RestoreReportBuilder::new().build();
        assert_eq!(report.success_rate(), 0.0);
    }

    #[test]
    fn test_steps_duration_sum() {
        let report = sample_report();
        let sum = report.steps_duration();
        assert_eq!(sum, Duration::from_millis(800));
    }

    #[test]
    fn test_total_duration() {
        let report = sample_report();
        assert_eq!(report.total_duration, Duration::from_millis(900));
    }

    #[test]
    fn test_summary_contains_name() {
        let report = sample_report();
        let s = report.summary();
        assert!(s.contains("test_report"), "summary: {s}");
    }

    #[test]
    fn test_summary_contains_counts() {
        let report = sample_report();
        let s = report.summary();
        assert!(s.contains("3 steps"), "summary: {s}");
        assert!(s.contains("2 ok"), "summary: {s}");
        assert!(s.contains("1 failed"), "summary: {s}");
    }

    #[test]
    fn test_failed_steps_list() {
        let report = sample_report();
        let failed = report.failed_steps();
        assert_eq!(failed, vec!["Declipping"]);
    }

    #[test]
    fn test_no_failed_steps() {
        let report = RestoreReportBuilder::new()
            .add_ok(RestoreStepKind::HumRemoval, Duration::from_millis(50))
            .build();
        assert!(report.failed_steps().is_empty());
    }

    #[test]
    fn test_step_labels_non_empty() {
        let kinds = [
            RestoreStepKind::DcRemoval,
            RestoreStepKind::ClickRemoval,
            RestoreStepKind::HissReduction,
            RestoreStepKind::HumRemoval,
            RestoreStepKind::Declipping,
            RestoreStepKind::WowFlutterCorrection,
            RestoreStepKind::ColorCorrection,
            RestoreStepKind::GrainSynthesis,
            RestoreStepKind::Deflicker,
            RestoreStepKind::TelecineRemoval,
            RestoreStepKind::Upscale,
            RestoreStepKind::Deband,
        ];
        for k in &kinds {
            assert!(!k.label().is_empty(), "label for {k:?} should not be empty");
        }
    }

    #[test]
    fn test_custom_step_label() {
        let kind = RestoreStepKind::Custom("MyCustom".to_string());
        assert_eq!(kind.label(), "MyCustom");
    }

    #[test]
    fn test_partial_success_counted() {
        let report = RestoreReportBuilder::new()
            .add_step(
                RestoreStepKind::DcRemoval,
                StepOutcome::PartialSuccess,
                Duration::from_millis(100),
                None,
            )
            .add_fail(
                RestoreStepKind::ClickRemoval,
                Duration::from_millis(200),
                "fail",
            )
            .build();
        // PartialSuccess counts toward success rate
        assert!((report.success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_skipped_outcome() {
        let report = RestoreReportBuilder::new()
            .add_step(
                RestoreStepKind::GrainSynthesis,
                StepOutcome::Skipped,
                Duration::ZERO,
                Some("audio only file".to_string()),
            )
            .build();
        assert_eq!(report.count_outcome(StepOutcome::Skipped), 1);
    }

    #[test]
    fn test_builder_source_stored() {
        let report = RestoreReportBuilder::new()
            .source("archive/tape01.wav")
            .build();
        assert_eq!(report.source, "archive/tape01.wav");
    }

    #[test]
    fn test_note_preserved_in_step() {
        let report = RestoreReportBuilder::new()
            .add_fail(
                RestoreStepKind::Declipping,
                Duration::from_millis(10),
                "saturated",
            )
            .build();
        assert_eq!(report.steps()[0].note.as_deref(), Some("saturated"));
    }
}
