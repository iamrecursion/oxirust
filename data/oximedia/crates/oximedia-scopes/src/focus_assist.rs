#![allow(dead_code)]
//! Focus assist tools for camera operators.
//!
//! Provides focus peaking (edge detection overlay) and related methods
//! for confirming critical focus on a video frame.

/// Methods available for focus assist visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusMethod {
    /// Color highlight applied to high-contrast edges.
    PeakingHighlight,
    /// Zebra-style striped lines drawn over in-focus regions.
    ZebralLines,
    /// Magnified (zoomed) view of a region of interest for precise focus.
    MagnifiedView,
}

impl FocusMethod {
    /// Returns whether this method produces an overlay on the original image
    /// rather than replacing it.
    #[must_use]
    pub fn is_overlay(self) -> bool {
        matches!(self, Self::PeakingHighlight | Self::ZebralLines)
    }

    /// Returns a short label for display.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::PeakingHighlight => "Peaking",
            Self::ZebralLines => "Zebra",
            Self::MagnifiedView => "Magnified",
        }
    }
}

/// Configuration for the focus peaking algorithm.
#[derive(Debug, Clone)]
pub struct FocusPeaking {
    /// Gradient magnitude threshold (`[0.0, 1.0]`).
    pub threshold: f32,
    /// Peaking highlight color as RGBA bytes.
    pub color: [u8; 4],
    /// Sensitivity multiplier applied before thresholding.
    pub sensitivity: f32,
}

impl Default for FocusPeaking {
    fn default() -> Self {
        Self {
            threshold: 0.15,
            color: [255, 0, 0, 200],
            sensitivity: 1.0,
        }
    }
}

impl FocusPeaking {
    /// Creates a new `FocusPeaking` config.
    #[must_use]
    pub fn new(threshold: f32, color: [u8; 4], sensitivity: f32) -> Self {
        Self {
            threshold,
            color,
            sensitivity,
        }
    }

    /// Returns `true` when the given gradient magnitude would trigger
    /// the peaking highlight.
    #[must_use]
    pub fn threshold_ok(&self, gradient: f32) -> bool {
        gradient * self.sensitivity >= self.threshold
    }

    /// Returns the effective threshold after applying sensitivity.
    #[must_use]
    pub fn effective_threshold(&self) -> f32 {
        if self.sensitivity > 0.0 {
            self.threshold / self.sensitivity
        } else {
            f32::MAX
        }
    }
}

/// Focus assist processor that detects edges and computes focus metrics.
#[derive(Debug, Clone)]
pub struct FocusAssist {
    peaking: FocusPeaking,
    method: FocusMethod,
}

impl FocusAssist {
    /// Creates a new `FocusAssist` with the given method and peaking config.
    #[must_use]
    pub fn new(method: FocusMethod, peaking: FocusPeaking) -> Self {
        Self { peaking, method }
    }

    /// Returns the configured focus method.
    #[must_use]
    pub fn method(&self) -> FocusMethod {
        self.method
    }

    /// Detects edges in a luma plane using a simple Sobel-like gradient.
    ///
    /// `luma` must be `width * height` values in `[0.0, 1.0]`.
    ///
    /// Returns a flat `Vec<bool>` where `true` means the pixel is considered
    /// in focus (gradient exceeds threshold).
    #[must_use]
    pub fn detect_edges(&self, luma: &[f32], width: usize, height: usize) -> Vec<bool> {
        let mut result = vec![false; width * height];
        if width < 2 || height < 2 {
            return result;
        }
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let gx = luma[idx + 1] - luma[idx - 1];
                let gy = luma[idx + width] - luma[idx - width];
                let mag = (gx * gx + gy * gy).sqrt();
                result[idx] = self.peaking.threshold_ok(mag);
            }
        }
        result
    }

    /// Computes edge density — the fraction of pixels flagged as in-focus.
    ///
    /// Returns a value in `[0.0, 1.0]`. Higher values indicate a sharper
    /// (more in-focus) image.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn edge_density(&self, luma: &[f32], width: usize, height: usize) -> f32 {
        let edges = self.detect_edges(luma, width, height);
        if edges.is_empty() {
            return 0.0;
        }
        let count = edges.iter().filter(|&&e| e).count();
        count as f32 / edges.len() as f32
    }

    /// Returns whether any pixel in the frame exceeds the peaking threshold.
    #[must_use]
    pub fn has_focus_region(&self, luma: &[f32], width: usize, height: usize) -> bool {
        self.edge_density(luma, width, height) > 0.0
    }
}

impl Default for FocusAssist {
    fn default() -> Self {
        Self::new(FocusMethod::PeakingHighlight, FocusPeaking::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3×3 luma with a strong gradient at the centre pixel (1,1).
    /// Centre gradient: gx = luma[5]-luma[3] = 1.0-0.0 = 1.0, gy = luma[7]-luma[1] = 1.0-0.0 = 1.0
    fn sharp_luma() -> Vec<f32> {
        vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0]
    }

    /// Flat 3×3 luma — no edges.
    fn flat_luma() -> Vec<f32> {
        vec![0.5; 9]
    }

    #[test]
    fn test_focus_method_is_overlay_peaking() {
        assert!(FocusMethod::PeakingHighlight.is_overlay());
    }

    #[test]
    fn test_focus_method_is_overlay_zebra() {
        assert!(FocusMethod::ZebralLines.is_overlay());
    }

    #[test]
    fn test_focus_method_is_not_overlay_magnified() {
        assert!(!FocusMethod::MagnifiedView.is_overlay());
    }

    #[test]
    fn test_focus_method_labels() {
        assert_eq!(FocusMethod::PeakingHighlight.label(), "Peaking");
        assert_eq!(FocusMethod::ZebralLines.label(), "Zebra");
        assert_eq!(FocusMethod::MagnifiedView.label(), "Magnified");
    }

    #[test]
    fn test_peaking_threshold_ok_above() {
        let p = FocusPeaking::new(0.1, [255, 0, 0, 255], 1.0);
        assert!(p.threshold_ok(0.5));
    }

    #[test]
    fn test_peaking_threshold_ok_below() {
        let p = FocusPeaking::new(0.5, [255, 0, 0, 255], 1.0);
        assert!(!p.threshold_ok(0.1));
    }

    #[test]
    fn test_peaking_effective_threshold_with_sensitivity() {
        let p = FocusPeaking::new(0.4, [0; 4], 2.0);
        // effective = 0.4 / 2.0 = 0.2
        assert!((p.effective_threshold() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_peaking_effective_threshold_zero_sensitivity() {
        let p = FocusPeaking::new(0.1, [0; 4], 0.0);
        assert_eq!(p.effective_threshold(), f32::MAX);
    }

    #[test]
    fn test_detect_edges_sharp() {
        let assist = FocusAssist::default();
        let edges = assist.detect_edges(&sharp_luma(), 3, 3);
        // The centre pixel (1,1) should be flagged
        assert!(edges[4]);
    }

    #[test]
    fn test_detect_edges_flat() {
        let assist = FocusAssist::default();
        let edges = assist.detect_edges(&flat_luma(), 3, 3);
        // No edges in a flat image
        assert!(edges.iter().all(|&e| !e));
    }

    #[test]
    fn test_edge_density_range() {
        let assist = FocusAssist::default();
        let density = assist.edge_density(&sharp_luma(), 3, 3);
        assert!((0.0..=1.0).contains(&density));
    }

    #[test]
    fn test_edge_density_flat_is_zero() {
        let assist = FocusAssist::default();
        assert_eq!(assist.edge_density(&flat_luma(), 3, 3), 0.0);
    }

    #[test]
    fn test_has_focus_region_sharp() {
        let assist = FocusAssist::default();
        assert!(assist.has_focus_region(&sharp_luma(), 3, 3));
    }

    #[test]
    fn test_has_focus_region_flat() {
        let assist = FocusAssist::default();
        assert!(!assist.has_focus_region(&flat_luma(), 3, 3));
    }

    #[test]
    fn test_detect_edges_too_small() {
        let assist = FocusAssist::default();
        let luma = vec![0.5, 0.8];
        let edges = assist.detect_edges(&luma, 2, 1);
        assert!(edges.iter().all(|&e| !e));
    }
}
