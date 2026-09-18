//! Shot size and camera angle classification for `oximedia-shots`.
//!
//! Provides enumerations and a classifier for determining the framing size
//! (e.g. Extreme Wide, Close-up, Macro) and the camera angle (High, Eye-level,
//! Low, etc.) from simple numeric descriptors.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Shot size ────────────────────────────────────────────────────────────────

/// Canonical shot-size classification.
///
/// Ordered from largest (most environmental context) to smallest (most detail).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShotSize {
    /// Extreme Wide Shot – vast environment, subject is tiny or absent.
    ExtremeWide,
    /// Wide Shot – full body with room to breathe.
    Wide,
    /// Medium Shot – waist up.
    Medium,
    /// Close-up – head and shoulders.
    CloseUp,
    /// Macro / Extreme Close-up – fine detail (eye, ring, insect).
    Macro,
}

impl ShotSize {
    /// Returns a human-readable abbreviation used in shot lists.
    #[must_use]
    pub fn abbreviation(self) -> &'static str {
        match self {
            Self::ExtremeWide => "EWS",
            Self::Wide => "WS",
            Self::Medium => "MS",
            Self::CloseUp => "CU",
            Self::Macro => "ECU",
        }
    }

    /// Returns `true` when the shot reveals the subject's surroundings more
    /// than the subject itself (EWS or WS).
    #[must_use]
    pub fn is_establishing(self) -> bool {
        matches!(self, Self::ExtremeWide | Self::Wide)
    }

    /// Returns `true` when the shot emphasises facial or object detail.
    #[must_use]
    pub fn is_detail(self) -> bool {
        matches!(self, Self::CloseUp | Self::Macro)
    }
}

impl std::fmt::Display for ShotSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.abbreviation())
    }
}

// ── Camera angle ─────────────────────────────────────────────────────────────

/// Camera angle relative to the subject's eye-line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CameraAngle {
    /// Bird's-eye view – directly overhead.
    BirdsEye,
    /// High angle – camera is above eye-line, looking down.
    High,
    /// Eye-level – neutral, matches the subject's eye-line.
    EyeLevel,
    /// Low angle – camera is below eye-line, looking up.
    Low,
    /// Worm's-eye view – extreme low, near ground level.
    WormsEye,
    /// Dutch/canted angle – camera is tilted on its roll axis.
    Dutch,
}

impl CameraAngle {
    /// Returns the typical emotional or narrative connotation of this angle.
    #[must_use]
    pub fn connotation(self) -> &'static str {
        match self {
            Self::BirdsEye => "overview / god-like perspective",
            Self::High => "diminutive / vulnerable",
            Self::EyeLevel => "neutral / relatable",
            Self::Low => "powerful / threatening",
            Self::WormsEye => "extreme dominance / awe",
            Self::Dutch => "unease / tension",
        }
    }
}

impl std::fmt::Display for CameraAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::BirdsEye => "Bird's Eye",
            Self::High => "High Angle",
            Self::EyeLevel => "Eye Level",
            Self::Low => "Low Angle",
            Self::WormsEye => "Worm's Eye",
            Self::Dutch => "Dutch Angle",
        };
        write!(f, "{s}")
    }
}

// ── Classifier ───────────────────────────────────────────────────────────────

/// Combined result of shot-size + camera-angle classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotTypeResult {
    /// Classified shot size.
    pub size: ShotSize,
    /// Classified camera angle.
    pub angle: CameraAngle,
    /// Classifier confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Classifies shot size and camera angle from normalised numeric descriptors.
///
/// In a production system these values would come from a neural-network or
/// feature-extraction pipeline; here we accept simple heuristic inputs to keep
/// the crate dependency-free.
#[derive(Debug, Default, Clone)]
pub struct ShotTypeClassifier {
    /// Minimum confidence threshold below which results are flagged uncertain.
    pub min_confidence: f32,
}

impl ShotTypeClassifier {
    /// Creates a new classifier with a default minimum confidence of `0.5`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_confidence: 0.5,
        }
    }

    /// Classifies shot size from `subject_area_ratio` (0 = absent, 1 = fills frame).
    ///
    /// | ratio range | classification |
    /// |-------------|----------------|
    /// | < 0.05      | `ExtremeWide`  |
    /// | 0.05–0.20   | `Wide`         |
    /// | 0.20–0.50   | `Medium`       |
    /// | 0.50–0.80   | `CloseUp`      |
    /// | > 0.80      | `Macro`        |
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn classify_size(&self, subject_area_ratio: f32) -> ShotSize {
        let r = subject_area_ratio.clamp(0.0, 1.0);
        if r < 0.05 {
            ShotSize::ExtremeWide
        } else if r < 0.20 {
            ShotSize::Wide
        } else if r < 0.50 {
            ShotSize::Medium
        } else if r < 0.80 {
            ShotSize::CloseUp
        } else {
            ShotSize::Macro
        }
    }

    /// Classifies camera angle from `tilt_degrees` relative to horizontal.
    ///
    /// Positive values mean the camera looks down; negative means it looks up.
    ///
    /// | tilt_degrees | classification |
    /// |--------------|----------------|
    /// | > 60         | `BirdsEye`     |
    /// | 15–60        | `High`         |
    /// | -15–15       | `EyeLevel`     |
    /// | -60 – -15    | `Low`          |
    /// | < -60        | `WormsEye`     |
    #[must_use]
    pub fn classify_angle(&self, tilt_degrees: f32) -> CameraAngle {
        if tilt_degrees > 60.0 {
            CameraAngle::BirdsEye
        } else if tilt_degrees > 15.0 {
            CameraAngle::High
        } else if tilt_degrees > -15.0 {
            CameraAngle::EyeLevel
        } else if tilt_degrees > -60.0 {
            CameraAngle::Low
        } else {
            CameraAngle::WormsEye
        }
    }

    /// Full classification returning a [`ShotTypeResult`].
    ///
    /// `roll_degrees` is used only to detect Dutch angles; any roll outside
    /// ±5° marks the shot as Dutch regardless of tilt.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn classify(
        &self,
        subject_area_ratio: f32,
        tilt_degrees: f32,
        roll_degrees: f32,
        confidence: f32,
    ) -> ShotTypeResult {
        let size = self.classify_size(subject_area_ratio);
        let angle = if roll_degrees.abs() > 5.0 {
            CameraAngle::Dutch
        } else {
            self.classify_angle(tilt_degrees)
        };
        ShotTypeResult {
            size,
            angle,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Returns `true` when the result meets the minimum confidence threshold.
    #[must_use]
    pub fn is_confident(&self, result: &ShotTypeResult) -> bool {
        result.confidence >= self.min_confidence
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shot_size_extreme_wide() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_size(0.01), ShotSize::ExtremeWide);
    }

    #[test]
    fn test_shot_size_wide() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_size(0.10), ShotSize::Wide);
    }

    #[test]
    fn test_shot_size_medium() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_size(0.35), ShotSize::Medium);
    }

    #[test]
    fn test_shot_size_close_up() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_size(0.65), ShotSize::CloseUp);
    }

    #[test]
    fn test_shot_size_macro() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_size(0.90), ShotSize::Macro);
    }

    #[test]
    fn test_shot_size_boundary_clamp_high() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_size(1.5), ShotSize::Macro);
    }

    #[test]
    fn test_shot_size_boundary_clamp_low() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_size(-1.0), ShotSize::ExtremeWide);
    }

    #[test]
    fn test_camera_angle_birds_eye() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_angle(75.0), CameraAngle::BirdsEye);
    }

    #[test]
    fn test_camera_angle_high() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_angle(30.0), CameraAngle::High);
    }

    #[test]
    fn test_camera_angle_eye_level() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_angle(0.0), CameraAngle::EyeLevel);
    }

    #[test]
    fn test_camera_angle_low() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_angle(-30.0), CameraAngle::Low);
    }

    #[test]
    fn test_camera_angle_worms_eye() {
        let c = ShotTypeClassifier::new();
        assert_eq!(c.classify_angle(-70.0), CameraAngle::WormsEye);
    }

    #[test]
    fn test_dutch_angle_overrides_tilt() {
        let c = ShotTypeClassifier::new();
        let result = c.classify(0.4, 0.0, 15.0, 0.9);
        assert_eq!(result.angle, CameraAngle::Dutch);
    }

    #[test]
    fn test_classify_full_result() {
        let c = ShotTypeClassifier::new();
        let result = c.classify(0.3, 5.0, 0.0, 0.85);
        assert_eq!(result.size, ShotSize::Medium);
        assert_eq!(result.angle, CameraAngle::EyeLevel);
        assert!((result.confidence - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_is_confident_above_threshold() {
        let c = ShotTypeClassifier::new();
        let result = ShotTypeResult {
            size: ShotSize::Medium,
            angle: CameraAngle::EyeLevel,
            confidence: 0.8,
        };
        assert!(c.is_confident(&result));
    }

    #[test]
    fn test_is_confident_below_threshold() {
        let c = ShotTypeClassifier::new();
        let result = ShotTypeResult {
            size: ShotSize::Wide,
            angle: CameraAngle::High,
            confidence: 0.3,
        };
        assert!(!c.is_confident(&result));
    }

    #[test]
    fn test_abbreviation_roundtrip() {
        assert_eq!(ShotSize::ExtremeWide.abbreviation(), "EWS");
        assert_eq!(ShotSize::Wide.abbreviation(), "WS");
        assert_eq!(ShotSize::Medium.abbreviation(), "MS");
        assert_eq!(ShotSize::CloseUp.abbreviation(), "CU");
        assert_eq!(ShotSize::Macro.abbreviation(), "ECU");
    }

    #[test]
    fn test_is_establishing() {
        assert!(ShotSize::ExtremeWide.is_establishing());
        assert!(ShotSize::Wide.is_establishing());
        assert!(!ShotSize::Medium.is_establishing());
        assert!(!ShotSize::CloseUp.is_establishing());
        assert!(!ShotSize::Macro.is_establishing());
    }

    #[test]
    fn test_is_detail() {
        assert!(!ShotSize::ExtremeWide.is_detail());
        assert!(!ShotSize::Wide.is_detail());
        assert!(!ShotSize::Medium.is_detail());
        assert!(ShotSize::CloseUp.is_detail());
        assert!(ShotSize::Macro.is_detail());
    }
}
