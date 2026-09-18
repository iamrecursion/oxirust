//! Detection metrics and scoring for interlace and telecine analysis.
//!
//! This module provides various metrics for quantifying interlacing artifacts,
//! field differences, and telecine patterns. These metrics are used to make
//! decisions about content type and appropriate deinterlacing methods.

use std::fmt;

/// Metrics for interlace detection.
///
/// These metrics quantify the amount of interlacing artifacts present in a frame
/// or sequence of frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterlaceMetrics {
    /// Comb detection score (0.0 = progressive, 1.0 = heavily interlaced).
    pub comb_score: f64,
    /// Field difference score (measures inter-field motion).
    pub field_diff: f64,
    /// Spatial comb metric (frequency domain analysis).
    pub spatial_comb: f64,
    /// Temporal comb metric (frame-to-frame analysis).
    pub temporal_comb: f64,
    /// Edge combing score (focus on edges where combing is most visible).
    pub edge_comb: f64,
}

impl InterlaceMetrics {
    /// Creates new interlace metrics with default (progressive) values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            comb_score: 0.0,
            field_diff: 0.0,
            spatial_comb: 0.0,
            temporal_comb: 0.0,
            edge_comb: 0.0,
        }
    }

    /// Creates interlace metrics from individual components.
    #[must_use]
    pub const fn from_components(
        comb_score: f64,
        field_diff: f64,
        spatial_comb: f64,
        temporal_comb: f64,
        edge_comb: f64,
    ) -> Self {
        Self {
            comb_score,
            field_diff,
            spatial_comb,
            temporal_comb,
            edge_comb,
        }
    }

    /// Returns the overall interlacing confidence (0.0-1.0).
    ///
    /// This combines all metrics into a single confidence score, with higher
    /// weight given to more reliable indicators.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        // Weighted average of metrics
        let weights = [0.3, 0.2, 0.2, 0.15, 0.15];
        let scores = [
            self.comb_score,
            self.field_diff,
            self.spatial_comb,
            self.temporal_comb,
            self.edge_comb,
        ];

        let weighted_sum: f64 = scores
            .iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum();

        weighted_sum.clamp(0.0, 1.0)
    }

    /// Returns true if metrics indicate interlaced content.
    ///
    /// Uses a threshold-based approach with the overall confidence score.
    #[must_use]
    pub fn is_interlaced(&self, threshold: f64) -> bool {
        self.confidence() > threshold
    }

    /// Normalizes all metrics to the 0.0-1.0 range.
    #[must_use]
    pub fn normalize(&self) -> Self {
        Self {
            comb_score: self.comb_score.clamp(0.0, 1.0),
            field_diff: self.field_diff.clamp(0.0, 1.0),
            spatial_comb: self.spatial_comb.clamp(0.0, 1.0),
            temporal_comb: self.temporal_comb.clamp(0.0, 1.0),
            edge_comb: self.edge_comb.clamp(0.0, 1.0),
        }
    }
}

impl Default for InterlaceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for InterlaceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InterlaceMetrics(comb: {:.3}, field_diff: {:.3}, spatial: {:.3}, temporal: {:.3}, edge: {:.3}, confidence: {:.3})",
            self.comb_score,
            self.field_diff,
            self.spatial_comb,
            self.temporal_comb,
            self.edge_comb,
            self.confidence()
        )
    }
}

/// Telecine detection metrics.
///
/// Metrics for detecting various telecine patterns (film-to-video transfer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelecineMetrics {
    /// Pulldown pattern confidence (0.0-1.0).
    pub pattern_confidence: f64,
    /// Cadence stability score (how consistent the pattern is).
    pub cadence_stability: f64,
    /// Frame difference variance (used to detect repeated fields).
    pub frame_variance: f64,
    /// Field match quality (how well fields align with pattern).
    pub field_match_quality: f64,
}

impl TelecineMetrics {
    /// Creates new telecine metrics with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pattern_confidence: 0.0,
            cadence_stability: 0.0,
            frame_variance: 0.0,
            field_match_quality: 0.0,
        }
    }

    /// Creates telecine metrics from individual components.
    #[must_use]
    pub const fn from_components(
        pattern_confidence: f64,
        cadence_stability: f64,
        frame_variance: f64,
        field_match_quality: f64,
    ) -> Self {
        Self {
            pattern_confidence,
            cadence_stability,
            frame_variance,
            field_match_quality,
        }
    }

    /// Returns the overall telecine detection confidence (0.0-1.0).
    #[must_use]
    pub fn confidence(&self) -> f64 {
        // Pattern confidence and cadence stability are most important
        let weighted = self.pattern_confidence * 0.4
            + self.cadence_stability * 0.3
            + self.frame_variance * 0.15
            + self.field_match_quality * 0.15;

        weighted.clamp(0.0, 1.0)
    }

    /// Returns true if metrics indicate telecine content.
    #[must_use]
    pub fn is_telecine(&self, threshold: f64) -> bool {
        self.confidence() > threshold
    }
}

impl Default for TelecineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TelecineMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TelecineMetrics(pattern: {:.3}, cadence: {:.3}, variance: {:.3}, match: {:.3}, confidence: {:.3})",
            self.pattern_confidence,
            self.cadence_stability,
            self.frame_variance,
            self.field_match_quality,
            self.confidence()
        )
    }
}

/// Combined detection score for content classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionScore {
    /// Progressive content score (0.0-1.0).
    pub progressive: f64,
    /// Interlaced content score (0.0-1.0).
    pub interlaced: f64,
    /// Telecine content score (0.0-1.0).
    pub telecine: f64,
    /// Mixed content score (combination of film and video).
    pub mixed: f64,
}

impl DetectionScore {
    /// Creates a new detection score with all values at 0.0.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            progressive: 0.0,
            interlaced: 0.0,
            telecine: 0.0,
            mixed: 0.0,
        }
    }

    /// Creates a detection score from individual components.
    #[must_use]
    pub const fn from_components(
        progressive: f64,
        interlaced: f64,
        telecine: f64,
        mixed: f64,
    ) -> Self {
        Self {
            progressive,
            interlaced,
            telecine,
            mixed,
        }
    }

    /// Returns the most likely content type based on scores.
    #[must_use]
    pub fn dominant_type(&self) -> ContentType {
        let max_score = self
            .progressive
            .max(self.interlaced)
            .max(self.telecine)
            .max(self.mixed);

        if (self.progressive - max_score).abs() < f64::EPSILON {
            ContentType::Progressive
        } else if (self.interlaced - max_score).abs() < f64::EPSILON {
            ContentType::Interlaced
        } else if (self.telecine - max_score).abs() < f64::EPSILON {
            ContentType::Telecine
        } else {
            ContentType::Mixed
        }
    }

    /// Normalizes scores so they sum to 1.0.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let sum = self.progressive + self.interlaced + self.telecine + self.mixed;
        if sum < f64::EPSILON {
            return Self::new();
        }

        Self {
            progressive: self.progressive / sum,
            interlaced: self.interlaced / sum,
            telecine: self.telecine / sum,
            mixed: self.mixed / sum,
        }
    }

    /// Returns the confidence in the dominant type.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        let normalized = self.normalize();
        match normalized.dominant_type() {
            ContentType::Progressive => normalized.progressive,
            ContentType::Interlaced => normalized.interlaced,
            ContentType::Telecine => normalized.telecine,
            ContentType::Mixed => normalized.mixed,
        }
    }
}

impl Default for DetectionScore {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DetectionScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let normalized = self.normalize();
        write!(
            f,
            "DetectionScore(progressive: {:.1}%, interlaced: {:.1}%, telecine: {:.1}%, mixed: {:.1}%, type: {:?})",
            normalized.progressive * 100.0,
            normalized.interlaced * 100.0,
            normalized.telecine * 100.0,
            normalized.mixed * 100.0,
            normalized.dominant_type()
        )
    }
}

/// Content type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// Progressive scan (no interlacing).
    Progressive,
    /// Interlaced video content.
    Interlaced,
    /// Telecine (film transferred to video).
    Telecine,
    /// Mixed content (combination of film and video).
    Mixed,
}

impl fmt::Display for ContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Progressive => write!(f, "Progressive"),
            Self::Interlaced => write!(f, "Interlaced"),
            Self::Telecine => write!(f, "Telecine"),
            Self::Mixed => write!(f, "Mixed"),
        }
    }
}

/// Field order for interlaced content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldOrder {
    /// Top field first (TFF).
    TopFieldFirst,
    /// Bottom field first (BFF).
    BottomFieldFirst,
    /// Unknown field order.
    Unknown,
}

impl fmt::Display for FieldOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TopFieldFirst => write!(f, "TFF"),
            Self::BottomFieldFirst => write!(f, "BFF"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Confidence level for detection results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConfidenceLevel {
    /// Very low confidence (< 0.3).
    VeryLow,
    /// Low confidence (0.3-0.5).
    Low,
    /// Medium confidence (0.5-0.7).
    Medium,
    /// High confidence (0.7-0.9).
    High,
    /// Very high confidence (>= 0.9).
    VeryHigh,
}

impl ConfidenceLevel {
    /// Converts a confidence value to a confidence level.
    #[must_use]
    pub fn from_value(confidence: f64) -> Self {
        if confidence < 0.3 {
            Self::VeryLow
        } else if confidence < 0.5 {
            Self::Low
        } else if confidence < 0.7 {
            Self::Medium
        } else if confidence < 0.9 {
            Self::High
        } else {
            Self::VeryHigh
        }
    }

    /// Returns the threshold value for this confidence level.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        match self {
            Self::VeryLow => 0.0,
            Self::Low => 0.3,
            Self::Medium => 0.5,
            Self::High => 0.7,
            Self::VeryHigh => 0.9,
        }
    }
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VeryLow => write!(f, "Very Low"),
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::VeryHigh => write!(f, "Very High"),
        }
    }
}
