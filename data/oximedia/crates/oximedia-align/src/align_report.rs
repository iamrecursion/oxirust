#![allow(dead_code)]
//! Alignment quality reporting and diagnostics.
//!
//! This module generates comprehensive reports about alignment quality,
//! including per-frame accuracy metrics, drift analysis, and confidence
//! scoring for multi-camera synchronization workflows.

use std::collections::BTreeMap;

/// Overall quality grade for an alignment result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlignGrade {
    /// Perfect alignment (sub-sample accuracy)
    Excellent,
    /// Good alignment (within 1 frame)
    Good,
    /// Acceptable alignment (within 2-3 frames)
    Acceptable,
    /// Poor alignment (more than 3 frames off)
    Poor,
    /// Alignment failed entirely
    Failed,
}

impl AlignGrade {
    /// Convert a numeric error (in frames) to a quality grade.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_frame_error(error_frames: f64) -> Self {
        if error_frames < 0.5 {
            Self::Excellent
        } else if error_frames < 1.5 {
            Self::Good
        } else if error_frames < 3.5 {
            Self::Acceptable
        } else if error_frames < f64::INFINITY {
            Self::Poor
        } else {
            Self::Failed
        }
    }

    /// Return a human-readable label for this grade.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Acceptable => "Acceptable",
            Self::Poor => "Poor",
            Self::Failed => "Failed",
        }
    }

    /// Return a numeric score from 0.0 (Failed) to 1.0 (Excellent).
    #[must_use]
    pub fn score(&self) -> f64 {
        match self {
            Self::Excellent => 1.0,
            Self::Good => 0.8,
            Self::Acceptable => 0.6,
            Self::Poor => 0.3,
            Self::Failed => 0.0,
        }
    }
}

/// A single per-frame alignment measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMeasurement {
    /// Frame index in the source timeline.
    pub frame_index: u64,
    /// Measured offset in seconds relative to reference.
    pub offset_secs: f64,
    /// Confidence of this measurement (0.0..1.0).
    pub confidence: f64,
    /// Spatial displacement in pixels (if applicable).
    pub spatial_error_px: f64,
}

impl FrameMeasurement {
    /// Create a new frame measurement.
    #[must_use]
    pub fn new(frame_index: u64, offset_secs: f64, confidence: f64, spatial_error_px: f64) -> Self {
        Self {
            frame_index,
            offset_secs,
            confidence: confidence.clamp(0.0, 1.0),
            spatial_error_px,
        }
    }
}

/// Drift statistics computed from a series of frame measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftStats {
    /// Mean drift in seconds.
    pub mean_drift: f64,
    /// Maximum absolute drift in seconds.
    pub max_drift: f64,
    /// Standard deviation of drift in seconds.
    pub std_dev: f64,
    /// Linear drift rate (seconds per frame).
    pub drift_rate: f64,
}

impl DriftStats {
    /// Compute drift statistics from a slice of offset values.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute(offsets: &[f64]) -> Self {
        if offsets.is_empty() {
            return Self {
                mean_drift: 0.0,
                max_drift: 0.0,
                std_dev: 0.0,
                drift_rate: 0.0,
            };
        }

        let n = offsets.len() as f64;
        let mean = offsets.iter().sum::<f64>() / n;
        let max_abs = offsets.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let variance = offsets.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Simple linear regression for drift rate
        let drift_rate = if offsets.len() >= 2 {
            let n_minus_1 = (offsets.len() - 1) as f64;
            let first = offsets[0];
            let last = offsets[offsets.len() - 1];
            (last - first) / n_minus_1
        } else {
            0.0
        };

        Self {
            mean_drift: mean,
            max_drift: max_abs,
            std_dev,
            drift_rate,
        }
    }

    /// Check whether drift exceeds a threshold (in seconds).
    #[must_use]
    pub fn exceeds_threshold(&self, threshold_secs: f64) -> bool {
        self.max_drift > threshold_secs
    }
}

/// A comprehensive alignment quality report.
#[derive(Debug, Clone)]
pub struct AlignReport {
    /// Human-readable title for this report.
    pub title: String,
    /// Per-frame measurements indexed by frame number.
    pub measurements: BTreeMap<u64, FrameMeasurement>,
    /// Computed drift statistics.
    pub drift_stats: Option<DriftStats>,
    /// Overall quality grade.
    pub grade: AlignGrade,
    /// Textual notes and warnings.
    pub notes: Vec<String>,
}

impl AlignReport {
    /// Create a new empty alignment report.
    #[must_use]
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            measurements: BTreeMap::new(),
            drift_stats: None,
            grade: AlignGrade::Failed,
            notes: Vec::new(),
        }
    }

    /// Add a frame measurement to the report.
    pub fn add_measurement(&mut self, m: FrameMeasurement) {
        self.measurements.insert(m.frame_index, m);
    }

    /// Add a textual note or warning.
    pub fn add_note(&mut self, note: &str) {
        self.notes.push(note.to_string());
    }

    /// Finalize the report: compute drift stats and assign a grade.
    #[allow(clippy::cast_precision_loss)]
    pub fn finalize(&mut self) {
        let offsets: Vec<f64> = self.measurements.values().map(|m| m.offset_secs).collect();
        let drift = DriftStats::compute(&offsets);
        self.drift_stats = Some(drift);

        // Grade based on max drift (assuming 30fps -> 1 frame = ~0.0333s)
        let frame_error = drift.max_drift / 0.0333;
        self.grade = AlignGrade::from_frame_error(frame_error);

        if drift.drift_rate.abs() > 1e-6 {
            self.add_note(&format!(
                "Linear drift detected: {:.6} s/frame",
                drift.drift_rate
            ));
        }
    }

    /// Return the number of measurements in this report.
    #[must_use]
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Return the average confidence across all measurements.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_confidence(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        let total: f64 = self.measurements.values().map(|m| m.confidence).sum();
        total / self.measurements.len() as f64
    }

    /// Generate a plain-text summary of this report.
    #[must_use]
    pub fn summary_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== {} ===", self.title));
        lines.push(format!("Grade: {}", self.grade.label()));
        lines.push(format!("Measurements: {}", self.measurement_count()));
        lines.push(format!("Avg confidence: {:.3}", self.average_confidence()));
        if let Some(ref drift) = self.drift_stats {
            lines.push(format!("Mean drift: {:.6} s", drift.mean_drift));
            lines.push(format!("Max drift:  {:.6} s", drift.max_drift));
            lines.push(format!("Drift rate: {:.9} s/frame", drift.drift_rate));
        }
        for note in &self.notes {
            lines.push(format!("NOTE: {note}"));
        }
        lines.join("\n")
    }
}

/// Builder for creating alignment reports incrementally.
#[derive(Debug)]
pub struct AlignReportBuilder {
    /// Title for the report being built.
    title: String,
    /// Accumulated measurements.
    measurements: Vec<FrameMeasurement>,
    /// Accumulated notes.
    notes: Vec<String>,
}

impl AlignReportBuilder {
    /// Create a new report builder with the given title.
    #[must_use]
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            measurements: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Add a measurement.
    #[must_use]
    pub fn measurement(mut self, m: FrameMeasurement) -> Self {
        self.measurements.push(m);
        self
    }

    /// Add a note.
    #[must_use]
    pub fn note(mut self, note: &str) -> Self {
        self.notes.push(note.to_string());
        self
    }

    /// Build and finalize the report.
    #[must_use]
    pub fn build(self) -> AlignReport {
        let mut report = AlignReport::new(&self.title);
        for m in self.measurements {
            report.add_measurement(m);
        }
        for note in &self.notes {
            report.add_note(note);
        }
        report.finalize();
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_from_frame_error_excellent() {
        assert_eq!(AlignGrade::from_frame_error(0.1), AlignGrade::Excellent);
        assert_eq!(AlignGrade::from_frame_error(0.0), AlignGrade::Excellent);
    }

    #[test]
    fn test_grade_from_frame_error_good() {
        assert_eq!(AlignGrade::from_frame_error(1.0), AlignGrade::Good);
    }

    #[test]
    fn test_grade_from_frame_error_acceptable() {
        assert_eq!(AlignGrade::from_frame_error(2.0), AlignGrade::Acceptable);
        assert_eq!(AlignGrade::from_frame_error(3.0), AlignGrade::Acceptable);
    }

    #[test]
    fn test_grade_from_frame_error_poor() {
        assert_eq!(AlignGrade::from_frame_error(5.0), AlignGrade::Poor);
        assert_eq!(AlignGrade::from_frame_error(100.0), AlignGrade::Poor);
    }

    #[test]
    fn test_grade_labels() {
        assert_eq!(AlignGrade::Excellent.label(), "Excellent");
        assert_eq!(AlignGrade::Failed.label(), "Failed");
    }

    #[test]
    fn test_grade_scores() {
        assert!((AlignGrade::Excellent.score() - 1.0).abs() < f64::EPSILON);
        assert!((AlignGrade::Failed.score()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_measurement_confidence_clamped() {
        let m = FrameMeasurement::new(0, 0.0, 1.5, 0.0);
        assert!((m.confidence - 1.0).abs() < f64::EPSILON);

        let m2 = FrameMeasurement::new(0, 0.0, -0.5, 0.0);
        assert!((m2.confidence).abs() < f64::EPSILON);
    }

    #[test]
    fn test_drift_stats_empty() {
        let stats = DriftStats::compute(&[]);
        assert!((stats.mean_drift).abs() < f64::EPSILON);
        assert!((stats.max_drift).abs() < f64::EPSILON);
    }

    #[test]
    fn test_drift_stats_constant_offset() {
        let offsets = vec![0.01, 0.01, 0.01, 0.01];
        let stats = DriftStats::compute(&offsets);
        assert!((stats.mean_drift - 0.01).abs() < 1e-10);
        assert!((stats.max_drift - 0.01).abs() < 1e-10);
        assert!(stats.std_dev < 1e-10);
        assert!(stats.drift_rate.abs() < 1e-10);
    }

    #[test]
    fn test_drift_stats_linear_drift() {
        let offsets = vec![0.0, 0.001, 0.002, 0.003];
        let stats = DriftStats::compute(&offsets);
        assert!((stats.drift_rate - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_drift_exceeds_threshold() {
        let stats = DriftStats {
            mean_drift: 0.05,
            max_drift: 0.1,
            std_dev: 0.02,
            drift_rate: 0.0001,
        };
        assert!(stats.exceeds_threshold(0.05));
        assert!(!stats.exceeds_threshold(0.2));
    }

    #[test]
    fn test_report_finalize_and_grade() {
        let mut report = AlignReport::new("Test Report");
        for i in 0..10 {
            report.add_measurement(FrameMeasurement::new(i, 0.001, 0.9, 0.5));
        }
        report.finalize();
        assert_eq!(report.grade, AlignGrade::Excellent);
        assert!(report.drift_stats.is_some());
    }

    #[test]
    fn test_report_average_confidence() {
        let mut report = AlignReport::new("Conf test");
        report.add_measurement(FrameMeasurement::new(0, 0.0, 0.8, 0.0));
        report.add_measurement(FrameMeasurement::new(1, 0.0, 0.6, 0.0));
        assert!((report.average_confidence() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_report_empty_confidence() {
        let report = AlignReport::new("Empty");
        assert!((report.average_confidence()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_summary_text_contains_title() {
        let mut report = AlignReport::new("My Alignment");
        report.finalize();
        let text = report.summary_text();
        assert!(text.contains("My Alignment"));
        assert!(text.contains("Grade:"));
    }

    #[test]
    fn test_builder_builds_finalized_report() {
        let report = AlignReportBuilder::new("Builder Test")
            .measurement(FrameMeasurement::new(0, 0.0, 0.95, 0.1))
            .measurement(FrameMeasurement::new(1, 0.001, 0.92, 0.2))
            .note("Test note")
            .build();
        assert_eq!(report.measurement_count(), 2);
        // 1 user note + 1 auto-generated drift note from finalize()
        assert_eq!(report.notes.len(), 2);
        assert!(report.drift_stats.is_some());
    }

    #[test]
    fn test_grade_ordering() {
        assert!(AlignGrade::Excellent < AlignGrade::Good);
        assert!(AlignGrade::Good < AlignGrade::Acceptable);
        assert!(AlignGrade::Poor < AlignGrade::Failed);
    }
}
