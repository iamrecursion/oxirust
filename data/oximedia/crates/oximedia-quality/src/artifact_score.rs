//! Compression artifact detection and severity scoring.
//!
//! Provides classification of common video compression artifacts
//! ([`ArtifactType`]), per-artifact scoring ([`ArtifactScore`]), and an
//! aggregated report ([`ArtifactReport`]) with an overall severity assessment.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Categories of compression artifact that can be detected in video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Block-boundary discontinuities from DCT-based codecs (blocking).
    Blocking,
    /// Loss of fine detail due to over-quantisation (blurring).
    Blurring,
    /// Checkerboard-like noise from upsampling chroma planes.
    Ringing,
    /// Temporal flickering due to frame-level quantisation variation.
    Flickering,
    /// Colour banding from insufficient bit-depth or strong filtering.
    Banding,
    /// Visible mosquito noise around sharp edges.
    MosquitoNoise,
    /// Interlacing artefacts (combing, aliasing on horizontal edges).
    Interlacing,
    /// Any other or unknown artifact.
    Other,
}

impl ArtifactType {
    /// Returns a short display name for the artifact type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Blocking => "blocking",
            Self::Blurring => "blurring",
            Self::Ringing => "ringing",
            Self::Flickering => "flickering",
            Self::Banding => "banding",
            Self::MosquitoNoise => "mosquito_noise",
            Self::Interlacing => "interlacing",
            Self::Other => "other",
        }
    }
}

/// Severity level derived from an artifact score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Score ≤ threshold considered negligible.
    None,
    /// Low level artifact, likely imperceptible on most displays.
    Low,
    /// Moderate artifact, visible on careful inspection.
    Medium,
    /// Severe artifact, clearly visible during normal viewing.
    High,
    /// Critical artifact, significantly impairs viewing experience.
    Critical,
}

impl Severity {
    /// Derives a `Severity` from a normalised score in `[0, 1]`.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score < 0.05 {
            Self::None
        } else if score < 0.25 {
            Self::Low
        } else if score < 0.55 {
            Self::Medium
        } else if score < 0.80 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Returns a numeric weight used for aggregation (0–4).
    #[must_use]
    pub fn weight(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }

    /// Returns `true` if this severity is at or above [`Severity::Medium`].
    #[must_use]
    pub fn is_actionable(&self) -> bool {
        *self >= Self::Medium
    }
}

/// Score measuring the strength of a specific artifact type in a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactScore {
    /// Type of artifact being scored.
    pub artifact_type: ArtifactType,
    /// Normalised strength of the artifact in `[0.0, 1.0]`.
    pub score: f64,
    /// Frame index at which this score was measured.
    pub frame_index: usize,
    /// Spatial location hint (x, y, width, height) in pixels, if available.
    pub region: Option<(u32, u32, u32, u32)>,
}

impl ArtifactScore {
    /// Creates a new `ArtifactScore` for a given frame.
    ///
    /// `score` is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(artifact_type: ArtifactType, score: f64, frame_index: usize) -> Self {
        Self {
            artifact_type,
            score: score.clamp(0.0, 1.0),
            frame_index,
            region: None,
        }
    }

    /// Attaches a spatial region hint and returns `self`.
    #[must_use]
    pub fn with_region(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.region = Some((x, y, width, height));
        self
    }

    /// Returns the severity derived from this score.
    #[must_use]
    pub fn severity(&self) -> Severity {
        Severity::from_score(self.score)
    }

    /// Returns `true` if the artifact is actionable (≥ `Medium`).
    #[must_use]
    pub fn is_actionable(&self) -> bool {
        self.severity().is_actionable()
    }
}

/// Aggregated artifact report for a video sequence.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ArtifactReport {
    /// All per-frame artifact scores contributing to this report.
    pub scores: Vec<ArtifactScore>,
    /// Optional description of the content being assessed.
    pub description: String,
}

impl ArtifactReport {
    /// Creates a new, empty `ArtifactReport`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a report with a description.
    #[must_use]
    pub fn with_description(description: impl Into<String>) -> Self {
        Self {
            scores: Vec::new(),
            description: description.into(),
        }
    }

    /// Adds an artifact score to the report.
    pub fn add(&mut self, score: ArtifactScore) {
        self.scores.push(score);
    }

    /// Returns the number of artifact scores in the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns `true` if no scores have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Returns scores for the specified artifact type.
    #[must_use]
    pub fn by_type(&self, artifact_type: ArtifactType) -> Vec<&ArtifactScore> {
        self.scores
            .iter()
            .filter(|s| s.artifact_type == artifact_type)
            .collect()
    }

    /// Returns the overall severity derived from the worst single artifact
    /// score in the report.
    ///
    /// Returns [`Severity::None`] for an empty report.
    #[must_use]
    pub fn severity(&self) -> Severity {
        self.scores
            .iter()
            .map(ArtifactScore::severity)
            .max()
            .unwrap_or(Severity::None)
    }

    /// Returns the mean artifact score across all entries.
    ///
    /// Returns `0.0` if the report is empty.
    #[must_use]
    pub fn mean_score(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.scores.iter().map(|s| s.score).sum();
        sum / self.scores.len() as f64
    }

    /// Returns scores whose severity is at or above the given level.
    #[must_use]
    pub fn filter_by_severity(&self, min: Severity) -> Vec<&ArtifactScore> {
        self.scores.iter().filter(|s| s.severity() >= min).collect()
    }

    /// Returns `true` if any score has an actionable (≥ `Medium`) severity.
    #[must_use]
    pub fn has_actionable(&self) -> bool {
        self.scores.iter().any(ArtifactScore::is_actionable)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_from_score_none() {
        assert_eq!(Severity::from_score(0.0), Severity::None);
        assert_eq!(Severity::from_score(0.04), Severity::None);
    }

    #[test]
    fn test_severity_from_score_boundaries() {
        assert_eq!(Severity::from_score(0.05), Severity::Low);
        assert_eq!(Severity::from_score(0.25), Severity::Medium);
        assert_eq!(Severity::from_score(0.55), Severity::High);
        assert_eq!(Severity::from_score(0.80), Severity::Critical);
    }

    #[test]
    fn test_severity_weight() {
        assert_eq!(Severity::None.weight(), 0);
        assert_eq!(Severity::Critical.weight(), 4);
    }

    #[test]
    fn test_severity_is_actionable() {
        assert!(!Severity::Low.is_actionable());
        assert!(Severity::Medium.is_actionable());
        assert!(Severity::High.is_actionable());
    }

    #[test]
    fn test_artifact_score_clamp() {
        let s = ArtifactScore::new(ArtifactType::Blocking, 1.5, 0);
        assert_eq!(s.score, 1.0);
        let s2 = ArtifactScore::new(ArtifactType::Blocking, -0.5, 0);
        assert_eq!(s2.score, 0.0);
    }

    #[test]
    fn test_artifact_score_severity() {
        let s = ArtifactScore::new(ArtifactType::Blurring, 0.6, 0);
        assert_eq!(s.severity(), Severity::High);
    }

    #[test]
    fn test_artifact_score_with_region() {
        let s = ArtifactScore::new(ArtifactType::Ringing, 0.3, 0).with_region(10, 20, 100, 80);
        assert_eq!(s.region, Some((10, 20, 100, 80)));
    }

    #[test]
    fn test_artifact_score_is_actionable() {
        let s = ArtifactScore::new(ArtifactType::Blocking, 0.5, 0);
        assert!(s.is_actionable());
    }

    #[test]
    fn test_report_severity_worst_wins() {
        let mut r = ArtifactReport::new();
        r.add(ArtifactScore::new(ArtifactType::Blocking, 0.1, 0)); // Low
        r.add(ArtifactScore::new(ArtifactType::Blurring, 0.9, 1)); // Critical
        assert_eq!(r.severity(), Severity::Critical);
    }

    #[test]
    fn test_report_empty_severity_none() {
        let r = ArtifactReport::new();
        assert_eq!(r.severity(), Severity::None);
    }

    #[test]
    fn test_report_by_type() {
        let mut r = ArtifactReport::new();
        r.add(ArtifactScore::new(ArtifactType::Blocking, 0.3, 0));
        r.add(ArtifactScore::new(ArtifactType::Blurring, 0.2, 1));
        r.add(ArtifactScore::new(ArtifactType::Blocking, 0.4, 2));
        let blocking = r.by_type(ArtifactType::Blocking);
        assert_eq!(blocking.len(), 2);
    }

    #[test]
    fn test_report_filter_by_severity() {
        let mut r = ArtifactReport::new();
        r.add(ArtifactScore::new(ArtifactType::Blocking, 0.1, 0)); // Low
        r.add(ArtifactScore::new(ArtifactType::Blurring, 0.6, 1)); // High
        let high_plus = r.filter_by_severity(Severity::High);
        assert_eq!(high_plus.len(), 1);
    }

    #[test]
    fn test_report_mean_score() {
        let mut r = ArtifactReport::new();
        r.add(ArtifactScore::new(ArtifactType::Blocking, 0.4, 0));
        r.add(ArtifactScore::new(ArtifactType::Blurring, 0.6, 1));
        assert!((r.mean_score() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_report_has_actionable() {
        let mut r = ArtifactReport::new();
        r.add(ArtifactScore::new(ArtifactType::Blocking, 0.1, 0)); // Low — not actionable
        assert!(!r.has_actionable());
        r.add(ArtifactScore::new(ArtifactType::Blurring, 0.4, 1)); // Medium — actionable
        assert!(r.has_actionable());
    }

    #[test]
    fn test_artifact_type_name() {
        assert_eq!(ArtifactType::Blocking.name(), "blocking");
        assert_eq!(ArtifactType::MosquitoNoise.name(), "mosquito_noise");
    }

    #[test]
    fn test_report_len_and_empty() {
        let mut r = ArtifactReport::new();
        assert!(r.is_empty());
        r.add(ArtifactScore::new(ArtifactType::Banding, 0.2, 0));
        assert_eq!(r.len(), 1);
    }
}
