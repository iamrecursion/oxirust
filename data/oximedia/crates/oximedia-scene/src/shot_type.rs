//! Shot type classification for cinematographic framing analysis.
//!
//! Provides enumerations and detectors for standard cinematographic shot sizes,
//! from extreme wide shots through extreme close-ups.

#![allow(dead_code)]

use std::collections::HashMap;

/// Standard cinematographic shot sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShotType {
    /// Extreme wide shot — vast environment, tiny or no visible subject.
    ExtremeWide,
    /// Wide shot — full subject visible with surrounding environment.
    Wide,
    /// Medium wide shot — subject from knees up.
    MediumWide,
    /// Medium shot — subject from waist up.
    Medium,
    /// Medium close-up — subject from chest up.
    MediumCloseup,
    /// Close-up — subject's face fills the frame.
    Closeup,
    /// Extreme close-up — tight detail (eyes, hands, object).
    ExtremeCloseup,
    /// Point-of-view shot — the scene as seen by a character.
    Pov,
    /// Over-the-shoulder shot — perspective over one subject toward another.
    OverShoulder,
}

impl ShotType {
    /// Returns the industry-standard abbreviation for this shot type.
    #[must_use]
    pub fn abbreviation(self) -> &'static str {
        match self {
            Self::ExtremeWide => "EWS",
            Self::Wide => "WS",
            Self::MediumWide => "MWS",
            Self::Medium => "MS",
            Self::MediumCloseup => "MCU",
            Self::Closeup => "CU",
            Self::ExtremeCloseup => "ECU",
            Self::Pov => "POV",
            Self::OverShoulder => "OTS",
        }
    }

    /// Returns the typical focal length range in mm for a full-frame camera.
    ///
    /// Returns `(min_mm, max_mm)` as a guide for lens selection.
    #[must_use]
    pub fn typical_focal_mm(self) -> (u32, u32) {
        match self {
            Self::ExtremeWide => (10, 20),
            Self::Wide => (20, 35),
            Self::MediumWide => (35, 50),
            Self::Medium => (50, 85),
            Self::MediumCloseup => (85, 100),
            Self::Closeup => (85, 135),
            Self::ExtremeCloseup => (100, 200),
            Self::Pov => (24, 50),
            Self::OverShoulder => (50, 85),
        }
    }

    /// Returns a human-readable name for this shot type.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::ExtremeWide => "Extreme Wide Shot",
            Self::Wide => "Wide Shot",
            Self::MediumWide => "Medium Wide Shot",
            Self::Medium => "Medium Shot",
            Self::MediumCloseup => "Medium Close-Up",
            Self::Closeup => "Close-Up",
            Self::ExtremeCloseup => "Extreme Close-Up",
            Self::Pov => "Point of View",
            Self::OverShoulder => "Over the Shoulder",
        }
    }

    /// Returns `true` if this is a close-range shot (closer than medium).
    #[must_use]
    pub fn is_close(self) -> bool {
        matches!(
            self,
            Self::MediumCloseup | Self::Closeup | Self::ExtremeCloseup
        )
    }

    /// Returns `true` if this is a wide-range shot.
    #[must_use]
    pub fn is_wide(self) -> bool {
        matches!(self, Self::ExtremeWide | Self::Wide | Self::MediumWide)
    }
}

/// Heuristic parameters for shot-type detection.
#[derive(Debug, Clone)]
pub struct ShotTypeDetector {
    /// Fraction of frame height occupied by the face/subject (0.0–1.0).
    pub subject_fill_ratio: f32,
    /// Estimated depth-of-field blur amount (0 = sharp, 1 = very blurred background).
    pub background_blur: f32,
}

impl ShotTypeDetector {
    /// Creates a new `ShotTypeDetector` with the given parameters.
    #[must_use]
    pub fn new(subject_fill_ratio: f32, background_blur: f32) -> Self {
        Self {
            subject_fill_ratio: subject_fill_ratio.clamp(0.0, 1.0),
            background_blur: background_blur.clamp(0.0, 1.0),
        }
    }

    /// Classifies the shot type based on how much of the frame the subject fills.
    #[must_use]
    pub fn classify(&self) -> ShotType {
        match self.subject_fill_ratio {
            r if r < 0.05 => ShotType::ExtremeWide,
            r if r < 0.15 => ShotType::Wide,
            r if r < 0.25 => ShotType::MediumWide,
            r if r < 0.40 => ShotType::Medium,
            r if r < 0.55 => ShotType::MediumCloseup,
            r if r < 0.75 => ShotType::Closeup,
            _ => ShotType::ExtremeCloseup,
        }
    }

    /// Returns a confidence score (0.0–1.0) for the classification.
    ///
    /// High blur with a moderate fill ratio suggests a close-up.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        let fill_confidence = 1.0 - (self.subject_fill_ratio % 0.1) * 5.0;
        fill_confidence.clamp(0.5, 1.0)
    }
}

/// Statistics about shot type distribution in a sequence.
#[derive(Debug, Clone, Default)]
pub struct ShotTypeStats {
    counts: HashMap<String, usize>,
    total: usize,
}

impl ShotTypeStats {
    /// Creates a new, empty `ShotTypeStats`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a shot type observation.
    pub fn record(&mut self, shot: ShotType) {
        *self
            .counts
            .entry(shot.abbreviation().to_string())
            .or_insert(0) += 1;
        self.total += 1;
    }

    /// Returns the count for a given shot type.
    #[must_use]
    pub fn count_by_type(&self, shot: ShotType) -> usize {
        *self.counts.get(shot.abbreviation()).unwrap_or(&0)
    }

    /// Returns total number of shots recorded.
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Returns the fraction of shots of the given type (0.0–1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fraction(&self, shot: ShotType) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count_by_type(shot) as f64 / self.total as f64
    }

    /// Returns the most common shot type, or `None` if no shots recorded.
    #[must_use]
    pub fn most_common(&self) -> Option<ShotType> {
        let abbrev = self
            .counts
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.as_str())?;
        // Map abbreviation back to ShotType
        let all = [
            ShotType::ExtremeWide,
            ShotType::Wide,
            ShotType::MediumWide,
            ShotType::Medium,
            ShotType::MediumCloseup,
            ShotType::Closeup,
            ShotType::ExtremeCloseup,
            ShotType::Pov,
            ShotType::OverShoulder,
        ];
        all.iter().find(|s| s.abbreviation() == abbrev).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abbreviations_non_empty() {
        for shot in [
            ShotType::ExtremeWide,
            ShotType::Wide,
            ShotType::Medium,
            ShotType::Closeup,
            ShotType::ExtremeCloseup,
            ShotType::Pov,
        ] {
            assert!(!shot.abbreviation().is_empty());
        }
    }

    #[test]
    fn test_abbreviation_ews() {
        assert_eq!(ShotType::ExtremeWide.abbreviation(), "EWS");
    }

    #[test]
    fn test_abbreviation_cu() {
        assert_eq!(ShotType::Closeup.abbreviation(), "CU");
    }

    #[test]
    fn test_typical_focal_ranges_ordered() {
        let (min_ews, _) = ShotType::ExtremeWide.typical_focal_mm();
        let (min_ws, _) = ShotType::Wide.typical_focal_mm();
        assert!(min_ews < min_ws);
    }

    #[test]
    fn test_is_close() {
        assert!(ShotType::Closeup.is_close());
        assert!(ShotType::ExtremeCloseup.is_close());
        assert!(ShotType::MediumCloseup.is_close());
        assert!(!ShotType::Wide.is_close());
    }

    #[test]
    fn test_is_wide() {
        assert!(ShotType::ExtremeWide.is_wide());
        assert!(ShotType::Wide.is_wide());
        assert!(!ShotType::Closeup.is_wide());
    }

    #[test]
    fn test_detector_classify_extreme_wide() {
        let det = ShotTypeDetector::new(0.02, 0.0);
        assert_eq!(det.classify(), ShotType::ExtremeWide);
    }

    #[test]
    fn test_detector_classify_wide() {
        let det = ShotTypeDetector::new(0.10, 0.0);
        assert_eq!(det.classify(), ShotType::Wide);
    }

    #[test]
    fn test_detector_classify_medium() {
        let det = ShotTypeDetector::new(0.35, 0.1);
        assert_eq!(det.classify(), ShotType::Medium);
    }

    #[test]
    fn test_detector_classify_closeup() {
        let det = ShotTypeDetector::new(0.65, 0.5);
        assert_eq!(det.classify(), ShotType::Closeup);
    }

    #[test]
    fn test_detector_classify_extreme_closeup() {
        let det = ShotTypeDetector::new(0.90, 0.8);
        assert_eq!(det.classify(), ShotType::ExtremeCloseup);
    }

    #[test]
    fn test_detector_confidence_in_range() {
        let det = ShotTypeDetector::new(0.5, 0.3);
        let c = det.confidence();
        assert!(c >= 0.5 && c <= 1.0);
    }

    #[test]
    fn test_stats_count_by_type() {
        let mut stats = ShotTypeStats::new();
        stats.record(ShotType::Medium);
        stats.record(ShotType::Medium);
        stats.record(ShotType::Wide);
        assert_eq!(stats.count_by_type(ShotType::Medium), 2);
        assert_eq!(stats.count_by_type(ShotType::Wide), 1);
    }

    #[test]
    fn test_stats_total() {
        let mut stats = ShotTypeStats::new();
        stats.record(ShotType::Closeup);
        stats.record(ShotType::Closeup);
        assert_eq!(stats.total(), 2);
    }

    #[test]
    fn test_stats_fraction() {
        let mut stats = ShotTypeStats::new();
        stats.record(ShotType::Wide);
        stats.record(ShotType::Wide);
        stats.record(ShotType::Medium);
        let frac = stats.fraction(ShotType::Wide);
        assert!((frac - 2.0 / 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_stats_most_common() {
        let mut stats = ShotTypeStats::new();
        stats.record(ShotType::Closeup);
        stats.record(ShotType::Closeup);
        stats.record(ShotType::Closeup);
        stats.record(ShotType::Wide);
        assert_eq!(stats.most_common(), Some(ShotType::Closeup));
    }

    #[test]
    fn test_stats_most_common_empty() {
        let stats = ShotTypeStats::new();
        assert_eq!(stats.most_common(), None);
    }
}
