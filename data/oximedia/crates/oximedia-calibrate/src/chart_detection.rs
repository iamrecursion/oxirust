//! Calibration chart detection (color checker, resolution chart).
//!
//! Provides types and logic for detecting and validating standard calibration charts
//! such as X-Rite `ColorChecker` 24, Xrite, `GretagMacbeth`, ISO resolution charts, and DSCI charts.

#![allow(dead_code)]

// ── ChartType ─────────────────────────────────────────────────────────────────

/// Type of calibration chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    /// X-Rite `ColorChecker` Classic 24-patch chart.
    ColorChecker24,
    /// X-Rite chart (generic).
    Xrite,
    /// `GretagMacbeth` `ColorChecker` (legacy name for the 24-patch chart).
    GretagMacbeth,
    /// ISO resolution test chart.
    IsoResolution,
    /// DSCI (Digital Still Camera Imaging) chart.
    Dsci,
}

impl ChartType {
    /// Returns the number of color/resolution patches in this chart.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        match self {
            Self::ColorChecker24 => 24,
            Self::Xrite => 24,
            Self::GretagMacbeth => 24,
            Self::IsoResolution => 0,
            Self::Dsci => 12,
        }
    }

    /// Returns `true` if this chart contains color patches with known reference values.
    #[must_use]
    pub fn has_color_patches(&self) -> bool {
        match self {
            Self::ColorChecker24 | Self::Xrite | Self::GretagMacbeth | Self::Dsci => true,
            Self::IsoResolution => false,
        }
    }
}

// ── ChartPatch ────────────────────────────────────────────────────────────────

/// A single detected patch on a calibration chart.
#[derive(Debug, Clone)]
pub struct ChartPatch {
    /// Zero-based patch identifier.
    pub id: u8,
    /// X coordinate of the top-left corner (pixels).
    pub x: f32,
    /// Y coordinate of the top-left corner (pixels).
    pub y: f32,
    /// Width of the patch (pixels).
    pub width: f32,
    /// Height of the patch (pixels).
    pub height: f32,
    /// Expected sRGB values for this patch (0–255).
    pub expected_rgb: [u8; 3],
}

impl ChartPatch {
    /// Returns the center coordinates `(cx, cy)` of the patch.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    /// Returns the area of the patch in square pixels.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

// ── DetectedChart ─────────────────────────────────────────────────────────────

/// Result of detecting a calibration chart in an image.
#[derive(Debug, Clone)]
pub struct DetectedChart {
    /// The type of chart that was detected.
    pub chart_type: ChartType,
    /// Detection confidence in the range \[0.0, 1.0\].
    pub confidence: f32,
    /// All detected patches on the chart.
    pub patches: Vec<ChartPatch>,
    /// Estimated rotation of the chart in degrees (positive = clockwise).
    pub rotation_deg: f32,
}

impl DetectedChart {
    /// Returns `true` if the detection confidence meets the minimum threshold.
    #[must_use]
    pub fn is_valid(&self, min_confidence: f32) -> bool {
        self.confidence >= min_confidence
    }

    /// Returns the number of detected patches.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Returns `true` if the chart rotation is within ±5°.
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        self.rotation_deg.abs() < 5.0
    }
}

// ── Helper constructor ─────────────────────────────────────────────────────────

/// Build a simple `ChartPatch` with default expected colour.
#[must_use]
pub fn make_patch(id: u8, x: f32, y: f32, w: f32, h: f32) -> ChartPatch {
    ChartPatch {
        id,
        x,
        y,
        width: w,
        height: h,
        expected_rgb: [128, 128, 128],
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChartType ──────────────────────────────────────────────────────────────

    #[test]
    fn test_colorchecker24_patch_count() {
        assert_eq!(ChartType::ColorChecker24.patch_count(), 24);
    }

    #[test]
    fn test_xrite_patch_count() {
        assert_eq!(ChartType::Xrite.patch_count(), 24);
    }

    #[test]
    fn test_gretagmacbeth_patch_count() {
        assert_eq!(ChartType::GretagMacbeth.patch_count(), 24);
    }

    #[test]
    fn test_iso_resolution_patch_count() {
        assert_eq!(ChartType::IsoResolution.patch_count(), 0);
    }

    #[test]
    fn test_dsci_patch_count() {
        assert_eq!(ChartType::Dsci.patch_count(), 12);
    }

    #[test]
    fn test_has_color_patches_true() {
        assert!(ChartType::ColorChecker24.has_color_patches());
        assert!(ChartType::Xrite.has_color_patches());
        assert!(ChartType::GretagMacbeth.has_color_patches());
        assert!(ChartType::Dsci.has_color_patches());
    }

    #[test]
    fn test_has_color_patches_false_for_resolution() {
        assert!(!ChartType::IsoResolution.has_color_patches());
    }

    // ── ChartPatch ──────────────────────────────────────────────────────────────

    #[test]
    fn test_patch_center() {
        let patch = make_patch(0, 10.0, 20.0, 50.0, 40.0);
        let (cx, cy) = patch.center();
        assert!((cx - 35.0).abs() < 1e-5);
        assert!((cy - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_patch_area() {
        let patch = make_patch(1, 0.0, 0.0, 30.0, 20.0);
        assert!((patch.area() - 600.0).abs() < 1e-5);
    }

    #[test]
    fn test_patch_area_zero() {
        let patch = make_patch(2, 5.0, 5.0, 0.0, 10.0);
        assert!((patch.area() - 0.0).abs() < 1e-5);
    }

    // ── DetectedChart ───────────────────────────────────────────────────────────

    #[test]
    fn test_detected_chart_is_valid_above_threshold() {
        let chart = DetectedChart {
            chart_type: ChartType::ColorChecker24,
            confidence: 0.9,
            patches: vec![],
            rotation_deg: 0.0,
        };
        assert!(chart.is_valid(0.8));
    }

    #[test]
    fn test_detected_chart_is_valid_below_threshold() {
        let chart = DetectedChart {
            chart_type: ChartType::Xrite,
            confidence: 0.5,
            patches: vec![],
            rotation_deg: 2.0,
        };
        assert!(!chart.is_valid(0.75));
    }

    #[test]
    fn test_detected_chart_patch_count() {
        let patches = vec![
            make_patch(0, 0.0, 0.0, 10.0, 10.0),
            make_patch(1, 20.0, 0.0, 10.0, 10.0),
        ];
        let chart = DetectedChart {
            chart_type: ChartType::Dsci,
            confidence: 0.8,
            patches,
            rotation_deg: 1.0,
        };
        assert_eq!(chart.patch_count(), 2);
    }

    #[test]
    fn test_detected_chart_is_aligned_within_5deg() {
        let chart = DetectedChart {
            chart_type: ChartType::ColorChecker24,
            confidence: 0.95,
            patches: vec![],
            rotation_deg: 4.9,
        };
        assert!(chart.is_aligned());
    }

    #[test]
    fn test_detected_chart_not_aligned_beyond_5deg() {
        let chart = DetectedChart {
            chart_type: ChartType::ColorChecker24,
            confidence: 0.95,
            patches: vec![],
            rotation_deg: 6.0,
        };
        assert!(!chart.is_aligned());
    }

    #[test]
    fn test_detected_chart_negative_rotation_aligned() {
        let chart = DetectedChart {
            chart_type: ChartType::GretagMacbeth,
            confidence: 0.85,
            patches: vec![],
            rotation_deg: -3.0,
        };
        assert!(chart.is_aligned());
    }

    #[test]
    fn test_detected_chart_exactly_5deg_not_aligned() {
        // |5.0| is NOT < 5.0
        let chart = DetectedChart {
            chart_type: ChartType::IsoResolution,
            confidence: 0.7,
            patches: vec![],
            rotation_deg: 5.0,
        };
        assert!(!chart.is_aligned());
    }
}
