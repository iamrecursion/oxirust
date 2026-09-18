//! Sign language video quality assessment and window specification.
//!
//! Provides quality standards, detected-signer region analysis, and
//! picture-in-picture window specifications for sign language overlays.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sign language enum
// ---------------------------------------------------------------------------

/// Sign language variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignLanguage {
    /// American Sign Language.
    Asl,
    /// British Sign Language.
    Bsl,
    /// Deutsche Gebärdensprache (German Sign Language).
    DeutschGebaerdensprache,
    /// Auslan (Australian Sign Language).
    AuslanAustralia,
    /// Langue des Signes Française (French Sign Language).
    Lsf,
}

impl SignLanguage {
    /// Human-readable name for the sign language.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Asl => "American Sign Language (ASL)",
            Self::Bsl => "British Sign Language (BSL)",
            Self::DeutschGebaerdensprache => "Deutsche Gebärdensprache (DGS)",
            Self::AuslanAustralia => "Auslan (Australian Sign Language)",
            Self::Lsf => "Langue des Signes Française (LSF)",
        }
    }
}

// ---------------------------------------------------------------------------
// Video quality
// ---------------------------------------------------------------------------

/// Technical quality parameters for a sign language video stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignVideoQuality {
    /// Frames per second of the sign video.
    pub frame_rate: f32,
    /// Width × height in pixels.
    pub resolution: (u32, u32),
    /// Bit depth (typically 8 or 10).
    pub bit_depth: u8,
    /// Compression ratio (`original_size` / `compressed_size`).
    pub compression_ratio: f32,
}

impl SignVideoQuality {
    /// Create a new `SignVideoQuality`.
    #[must_use]
    pub const fn new(
        frame_rate: f32,
        resolution: (u32, u32),
        bit_depth: u8,
        compression_ratio: f32,
    ) -> Self {
        Self {
            frame_rate,
            resolution,
            bit_depth,
            compression_ratio,
        }
    }

    /// Check whether the video meets the minimum quality standard.
    ///
    /// Standards:
    /// - Minimum 25 fps
    /// - Minimum 480 p height (the "visible hands" rule)
    /// - Bit depth ≥ 8
    #[must_use]
    pub fn meets_standard(&self) -> bool {
        self.frame_rate >= 25.0 && self.resolution.1 >= 480 && self.bit_depth >= 8
    }

    /// Return a human-readable description of any quality issues.
    #[must_use]
    pub fn quality_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.frame_rate < 25.0 {
            issues.push(format!(
                "Frame rate {:.1} fps is below minimum 25 fps",
                self.frame_rate
            ));
        }
        if self.resolution.1 < 480 {
            issues.push(format!(
                "Resolution height {} px is below minimum 480 px (visible hands rule)",
                self.resolution.1
            ));
        }
        if self.bit_depth < 8 {
            issues.push(format!("Bit depth {} is below minimum 8", self.bit_depth));
        }
        issues
    }
}

// ---------------------------------------------------------------------------
// Signer region
// ---------------------------------------------------------------------------

/// Bounding box for a detected signer within a video frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignerRegion {
    /// Left edge of the bounding box in pixels.
    pub x: u32,
    /// Top edge of the bounding box in pixels.
    pub y: u32,
    /// Width of the bounding box in pixels.
    pub width: u32,
    /// Height of the bounding box in pixels.
    pub height: u32,
    /// Detection confidence (0.0–1.0).
    pub confidence: f32,
}

impl SignerRegion {
    /// Create a new `SignerRegion`.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32, confidence: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Composition analyser
// ---------------------------------------------------------------------------

/// Analyses sign-video framing and composition.
pub struct SignVideoAnalyzer;

impl SignVideoAnalyzer {
    /// Check the composition of a signer region within a frame and return
    /// human-readable framing warnings.
    ///
    /// # Arguments
    ///
    /// * `region` — Detected bounding box of the signer.
    /// * `frame_w` — Total frame width in pixels.
    /// * `frame_h` — Total frame height in pixels.
    #[must_use]
    pub fn check_composition(region: &SignerRegion, frame_w: u32, frame_h: u32) -> Vec<String> {
        let mut warnings = Vec::new();

        if frame_w == 0 || frame_h == 0 {
            warnings.push("Frame dimensions are zero — cannot assess composition.".to_string());
            return warnings;
        }

        let region_right = region.x + region.width;
        let region_bottom = region.y + region.height;

        // Signer extends beyond frame boundary
        if region_right > frame_w {
            warnings.push(format!(
                "Signer region extends {px} px beyond right edge of frame.",
                px = region_right - frame_w
            ));
        }
        if region_bottom > frame_h {
            warnings.push(format!(
                "Signer region extends {px} px below bottom edge of frame.",
                px = region_bottom - frame_h
            ));
        }

        // Coverage checks — the signer should occupy a visible portion
        let coverage_w = region.width as f32 / frame_w as f32;
        let coverage_h = region.height as f32 / frame_h as f32;

        if coverage_w < 0.15 {
            warnings.push(format!(
                "Signer is too narrow ({:.0}% of frame width); recommend at least 15%.",
                coverage_w * 100.0
            ));
        }
        if coverage_h < 0.30 {
            warnings.push(format!(
                "Signer is too short ({:.0}% of frame height); recommend at least 30% for hand visibility.",
                coverage_h * 100.0
            ));
        }

        // Low confidence detection
        if region.confidence < 0.5 {
            warnings.push(format!(
                "Signer detection confidence is low ({:.2}); framing may be unreliable.",
                region.confidence
            ));
        }

        warnings
    }
}

// ---------------------------------------------------------------------------
// PiP window specification
// ---------------------------------------------------------------------------

/// Position of the sign-language picture-in-picture window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignPosition {
    /// Bottom-right corner.
    BottomRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Top-right corner.
    TopRight,
    /// Top-left corner.
    TopLeft,
    /// Full-screen (side-by-side or overlay).
    FullScreen,
}

/// Specification for a sign-language picture-in-picture window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignWindowSpec {
    /// Position of the window within the main video frame.
    pub position: SignPosition,
    /// Size of the `PiP` window as a percentage of the main frame (0–100).
    pub size_pct: f32,
}

impl SignWindowSpec {
    /// Create a new window specification.
    #[must_use]
    pub fn new(position: SignPosition, size_pct: f32) -> Self {
        Self {
            position,
            size_pct: size_pct.clamp(0.0, 100.0),
        }
    }

    /// Default `PiP`: bottom-right at 25%.
    #[must_use]
    pub fn default_pip() -> Self {
        Self::new(SignPosition::BottomRight, 25.0)
    }

    /// Whether this spec represents full-screen mode.
    #[must_use]
    pub fn is_full_screen(&self) -> bool {
        self.position == SignPosition::FullScreen
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_language_names() {
        assert_eq!(SignLanguage::Asl.name(), "American Sign Language (ASL)");
        assert_eq!(SignLanguage::Bsl.name(), "British Sign Language (BSL)");
        assert!(SignLanguage::DeutschGebaerdensprache.name().contains("DGS"));
        assert!(SignLanguage::AuslanAustralia.name().contains("Auslan"));
        assert!(SignLanguage::Lsf.name().contains("LSF"));
    }

    #[test]
    fn test_video_quality_meets_standard() {
        let good = SignVideoQuality::new(25.0, (854, 480), 8, 10.0);
        assert!(good.meets_standard());
    }

    #[test]
    fn test_video_quality_fails_fps() {
        let bad = SignVideoQuality::new(24.0, (854, 480), 8, 10.0);
        assert!(!bad.meets_standard());
    }

    #[test]
    fn test_video_quality_fails_resolution() {
        let bad = SignVideoQuality::new(25.0, (640, 360), 8, 10.0);
        assert!(!bad.meets_standard());
    }

    #[test]
    fn test_video_quality_issues_list() {
        let bad = SignVideoQuality::new(20.0, (320, 240), 6, 10.0);
        let issues = bad.quality_issues();
        assert_eq!(issues.len(), 3);
    }

    #[test]
    fn test_signer_region_out_of_bounds() {
        let region = SignerRegion::new(900, 0, 300, 300, 0.9);
        let warnings = SignVideoAnalyzer::check_composition(&region, 1000, 600);
        assert!(warnings.iter().any(|w| w.contains("right edge")));
    }

    #[test]
    fn test_signer_region_too_small() {
        let region = SignerRegion::new(0, 0, 50, 50, 0.9);
        let warnings = SignVideoAnalyzer::check_composition(&region, 1920, 1080);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_signer_low_confidence() {
        let region = SignerRegion::new(100, 100, 500, 400, 0.3);
        let warnings = SignVideoAnalyzer::check_composition(&region, 1920, 1080);
        assert!(warnings.iter().any(|w| w.contains("confidence")));
    }

    #[test]
    fn test_sign_window_spec_default_pip() {
        let spec = SignWindowSpec::default_pip();
        assert_eq!(spec.position, SignPosition::BottomRight);
        assert!((spec.size_pct - 25.0).abs() < f32::EPSILON);
        assert!(!spec.is_full_screen());
    }

    #[test]
    fn test_sign_window_spec_full_screen() {
        let spec = SignWindowSpec::new(SignPosition::FullScreen, 100.0);
        assert!(spec.is_full_screen());
    }

    #[test]
    fn test_sign_position_variants() {
        let positions = [
            SignPosition::BottomRight,
            SignPosition::BottomLeft,
            SignPosition::TopRight,
            SignPosition::TopLeft,
            SignPosition::FullScreen,
        ];
        assert_eq!(positions.len(), 5);
    }
}
