#![allow(dead_code)]
//! Vectorscope colour target definitions and pixel analysis.
//!
//! Provides:
//! * [`ColorTarget`]          – named reference targets with expected hue angles.
//! * [`VectorscopePoint`]     – a single Cb/Cr sample with polar coordinates.
//! * [`VectorscopeAnalyzer`]  – extracts points from a video frame and classifies them.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// ColorTarget
// ---------------------------------------------------------------------------

/// Named colour targets displayed on a vectorscope graticule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorTarget {
    /// Caucasian skin-tone reference (~12–15° on a vectorscope).
    SkinTone,
    /// Red primary (approximately 103° in YUV space).
    RedPrimary,
    /// Green primary (approximately 241° in YUV space).
    GreenPrimary,
    /// Blue primary (approximately 347° in YUV space).
    BluePrimary,
    /// Cyan secondary.
    Cyan,
    /// Magenta secondary.
    Magenta,
    /// Yellow secondary.
    Yellow,
}

impl ColorTarget {
    /// The expected hue angle on the vectorscope in degrees `[0, 360)`.
    ///
    /// Values are approximate YUV angles for SMPTE 75% colour bars in BT.709.
    #[must_use]
    pub fn expected_angle(self) -> f32 {
        match self {
            Self::SkinTone => 13.0,
            Self::RedPrimary => 103.0,
            Self::GreenPrimary => 241.0,
            Self::BluePrimary => 347.0,
            Self::Cyan => 180.0 + 103.0 - 360.0 + 360.0, // opposite Red ≈ 283
            Self::Magenta => 180.0 + 241.0 - 360.0 + 360.0, // opposite Green ≈ 61
            Self::Yellow => 180.0 + 347.0 - 360.0 + 360.0, // opposite Blue ≈ 167
        }
    }

    /// Returns the complementary (opposite) target.
    #[must_use]
    pub const fn complement(self) -> Self {
        match self {
            Self::RedPrimary => Self::Cyan,
            Self::GreenPrimary => Self::Magenta,
            Self::BluePrimary => Self::Yellow,
            Self::Cyan => Self::RedPrimary,
            Self::Magenta => Self::GreenPrimary,
            Self::Yellow => Self::BluePrimary,
            Self::SkinTone => Self::SkinTone,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SkinTone => "Skin Tone",
            Self::RedPrimary => "Red",
            Self::GreenPrimary => "Green",
            Self::BluePrimary => "Blue",
            Self::Cyan => "Cyan",
            Self::Magenta => "Magenta",
            Self::Yellow => "Yellow",
        }
    }
}

// ---------------------------------------------------------------------------
// VectorscopePoint
// ---------------------------------------------------------------------------

/// A single pixel sample plotted on the vectorscope as a Cb/Cr coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorscopePoint {
    /// Normalised Cb component `[-0.5, 0.5]`.
    pub cb: f32,
    /// Normalised Cr component `[-0.5, 0.5]`.
    pub cr: f32,
    /// Normalised luma value `[0.0, 1.0]`.
    pub luma: f32,
}

impl VectorscopePoint {
    /// Create from normalised Cb, Cr and luma values.
    #[must_use]
    pub fn new(cb: f32, cr: f32, luma: f32) -> Self {
        Self { cb, cr, luma }
    }

    /// Hue angle in degrees `[0, 360)` measured from positive Cr axis.
    #[must_use]
    pub fn angle_deg(&self) -> f32 {
        let rad = self.cr.atan2(self.cb);
        let deg = rad * 180.0 / PI;
        if deg < 0.0 {
            deg + 360.0
        } else {
            deg
        }
    }

    /// Chroma magnitude (distance from the achromatic centre).
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.cb * self.cb + self.cr * self.cr).sqrt()
    }

    /// Returns `true` when the point is within `tolerance` degrees of `target`.
    #[must_use]
    pub fn near_target(&self, target: ColorTarget, tolerance_deg: f32) -> bool {
        let diff = (self.angle_deg() - target.expected_angle()).abs();
        let diff = diff.min(360.0 - diff); // wrap-around distance
        diff <= tolerance_deg
    }
}

// ---------------------------------------------------------------------------
// VectorscopeAnalyzer
// ---------------------------------------------------------------------------

/// Extracts [`VectorscopePoint`] samples from RGB24 video frames.
pub struct VectorscopeAnalyzer {
    /// Maximum number of points to collect (sub-sampling for performance).
    max_points: usize,
    /// Sub-sampling step (1 = every pixel, 2 = every other pixel, …).
    step: usize,
}

impl VectorscopeAnalyzer {
    /// Create an analyser.
    ///
    /// `max_points` caps memory usage; `step` controls sub-sampling.
    #[must_use]
    pub fn new(max_points: usize, step: usize) -> Self {
        Self {
            max_points,
            step: step.max(1),
        }
    }

    /// Analyse `frame` (RGB24, `width * height * 3` bytes) and return sampled points.
    ///
    /// The BT.709 RGB→YCbCr conversion is used.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze(&self, frame: &[u8], width: usize, height: usize) -> Vec<VectorscopePoint> {
        let total_pixels = width * height;
        if frame.len() < total_pixels * 3 {
            return Vec::new();
        }

        let mut points = Vec::with_capacity(self.max_points.min(total_pixels / self.step + 1));

        let mut i = 0usize;
        while i < total_pixels && points.len() < self.max_points {
            let r = frame[i * 3] as f32 / 255.0;
            let g = frame[i * 3 + 1] as f32 / 255.0;
            let b = frame[i * 3 + 2] as f32 / 255.0;

            // BT.709 coefficients
            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let cb = -0.1146 * r - 0.3854 * g + 0.5 * b;
            let cr = 0.5 * r - 0.4542 * g - 0.0458 * b;

            points.push(VectorscopePoint::new(cb, cr, y));
            i += self.step;
        }

        points
    }
}

impl Default for VectorscopeAnalyzer {
    fn default() -> Self {
        Self::new(50_000, 1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_target_skin_tone_angle() {
        let angle = ColorTarget::SkinTone.expected_angle();
        assert!(angle > 10.0 && angle < 20.0);
    }

    #[test]
    fn test_color_target_red_angle() {
        let angle = ColorTarget::RedPrimary.expected_angle();
        assert!((angle - 103.0).abs() < 1.0);
    }

    #[test]
    fn test_color_target_complement_red_cyan() {
        assert_eq!(ColorTarget::RedPrimary.complement(), ColorTarget::Cyan);
        assert_eq!(ColorTarget::Cyan.complement(), ColorTarget::RedPrimary);
    }

    #[test]
    fn test_color_target_labels_non_empty() {
        for t in [
            ColorTarget::SkinTone,
            ColorTarget::RedPrimary,
            ColorTarget::GreenPrimary,
            ColorTarget::BluePrimary,
            ColorTarget::Cyan,
            ColorTarget::Magenta,
            ColorTarget::Yellow,
        ] {
            assert!(!t.label().is_empty());
        }
    }

    #[test]
    fn test_vectorscope_point_angle_positive_cr() {
        // atan2(cr=1, cb=0) → 90°
        let p = VectorscopePoint::new(0.0, 1.0, 0.5);
        let angle = p.angle_deg();
        assert!((angle - 90.0).abs() < 1.0, "angle was {angle}");
    }

    #[test]
    fn test_vectorscope_point_angle_positive_cb() {
        // atan2(cr=0, cb=1) → 0°
        let p = VectorscopePoint::new(1.0, 0.0, 0.5);
        let angle = p.angle_deg();
        assert!(angle < 5.0 || angle > 355.0, "angle was {angle}");
    }

    #[test]
    fn test_vectorscope_point_magnitude_unit() {
        let p = VectorscopePoint::new(0.6, 0.8, 0.5);
        assert!((p.magnitude() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vectorscope_point_magnitude_zero() {
        let p = VectorscopePoint::new(0.0, 0.0, 0.5);
        assert!(p.magnitude().abs() < 1e-6);
    }

    #[test]
    fn test_vectorscope_point_near_target() {
        // Create a point at 103° (red primary)
        let angle_rad = 103.0_f32 * PI / 180.0;
        let p = VectorscopePoint::new(angle_rad.cos() * 0.5, angle_rad.sin() * 0.5, 0.5);
        // The angle_deg from atan2(cr, cb) – here cr=sin, cb=cos
        // actual angle = atan2(sin, cos) = 103°
        assert!(p.near_target(ColorTarget::RedPrimary, 10.0));
    }

    #[test]
    fn test_vectorscope_analyzer_empty_frame() {
        let analyzer = VectorscopeAnalyzer::default();
        let points = analyzer.analyze(&[], 0, 0);
        assert!(points.is_empty());
    }

    #[test]
    fn test_vectorscope_analyzer_black_frame() {
        let analyzer = VectorscopeAnalyzer::new(1000, 1);
        let frame = vec![0u8; 4 * 4 * 3];
        let points = analyzer.analyze(&frame, 4, 4);
        assert_eq!(points.len(), 16);
        for p in &points {
            assert!(p.luma.abs() < 1e-4);
        }
    }

    #[test]
    fn test_vectorscope_analyzer_max_points_cap() {
        let analyzer = VectorscopeAnalyzer::new(5, 1);
        let frame = vec![128u8; 100 * 3];
        let points = analyzer.analyze(&frame, 100, 1);
        assert!(points.len() <= 5);
    }

    #[test]
    fn test_vectorscope_analyzer_step_subsampling() {
        let analyzer = VectorscopeAnalyzer::new(1000, 2);
        let frame = vec![128u8; 10 * 3];
        let points = analyzer.analyze(&frame, 10, 1);
        assert_eq!(points.len(), 5);
    }

    #[test]
    fn test_vectorscope_analyzer_insufficient_data() {
        let analyzer = VectorscopeAnalyzer::default();
        let points = analyzer.analyze(&[0u8; 6], 4, 4);
        assert!(points.is_empty());
    }
}
