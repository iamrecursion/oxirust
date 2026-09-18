//! Dolby Vision mastering display metadata.
//!
//! Provides structures and utilities for mastering display metadata
//! used in Dolby Vision streams, including color primaries, white point,
//! and light-level parameters.

#![allow(dead_code)]

/// Color volume metadata describing the content's light level range.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorVolume {
    /// Peak luminance of the content in nits (cd/m²).
    pub max_luminance: f64,
    /// Minimum luminance of the content in nits (cd/m²).
    pub min_luminance: f64,
    /// Maximum Content Light Level in cd/m².
    pub max_content_light: u32,
    /// Maximum Frame-Average Light Level in cd/m².
    pub max_frame_avg_light: u32,
}

impl ColorVolume {
    /// Create a new `ColorVolume`.
    #[must_use]
    pub const fn new(
        max_luminance: f64,
        min_luminance: f64,
        max_content_light: u32,
        max_frame_avg_light: u32,
    ) -> Self {
        Self {
            max_luminance,
            min_luminance,
            max_content_light,
            max_frame_avg_light,
        }
    }

    /// Dynamic range in stops (log2 ratio).
    #[must_use]
    pub fn dynamic_range_stops(&self) -> f64 {
        if self.min_luminance <= 0.0 {
            return 0.0;
        }
        (self.max_luminance / self.min_luminance).log2()
    }
}

/// Mastering display descriptor, including chromaticity and luminance.
///
/// Color primaries and white point are in CIE xy chromaticity coordinates.
/// Luminance values are in nits (cd/m²).
#[derive(Debug, Clone, PartialEq)]
pub struct MasteringDisplay {
    /// Peak luminance of the mastering display in nits.
    pub max_luminance: f64,
    /// Minimum luminance of the mastering display in nits.
    pub min_luminance: f64,
    /// CIE xy chromaticity of [R, G, B] primaries.
    pub primaries: [[f64; 2]; 3],
    /// CIE xy chromaticity of the display white point.
    pub white_point: [f64; 2],
    /// Color volume metadata.
    pub color_volume: ColorVolume,
}

impl MasteringDisplay {
    /// Create a DCI-P3 D65 1000-nit mastering display.
    ///
    /// Common mastering target for streaming and broadcast HDR10/DV content.
    #[must_use]
    pub fn p3_d65_1000nit() -> Self {
        Self {
            max_luminance: 1000.0,
            min_luminance: 0.005,
            // DCI-P3 D65 primaries (CIE xy)
            primaries: [
                [0.680, 0.320], // R
                [0.265, 0.690], // G
                [0.150, 0.060], // B
            ],
            // D65 white point
            white_point: [0.3127, 0.3290],
            color_volume: ColorVolume::new(1000.0, 0.005, 1000, 400),
        }
    }

    /// Create a BT.2020 4000-nit mastering display.
    ///
    /// Common mastering target for theatrical and premium HDR content.
    #[must_use]
    pub fn bt2020_4000nit() -> Self {
        Self {
            max_luminance: 4000.0,
            min_luminance: 0.005,
            // BT.2020 primaries (CIE xy)
            primaries: [
                [0.708, 0.292], // R
                [0.170, 0.797], // G
                [0.131, 0.046], // B
            ],
            // D65 white point
            white_point: [0.3127, 0.3290],
            color_volume: ColorVolume::new(4000.0, 0.005, 4000, 1000),
        }
    }

    /// Validate the mastering display metadata for consistency.
    ///
    /// Returns `true` if all parameters are within plausible ranges.
    #[must_use]
    pub fn validate(&self) -> bool {
        // Luminance sanity
        if self.max_luminance <= self.min_luminance {
            return false;
        }
        if self.min_luminance < 0.0 || self.min_luminance > 1.0 {
            return false;
        }
        if self.max_luminance < 100.0 || self.max_luminance > 10_000.0 {
            return false;
        }

        // xy chromaticity must be in [0,1] range and sum < 1
        for primary in &self.primaries {
            if primary[0] < 0.0 || primary[0] > 1.0 {
                return false;
            }
            if primary[1] < 0.0 || primary[1] > 1.0 {
                return false;
            }
        }

        if self.white_point[0] < 0.0
            || self.white_point[0] > 1.0
            || self.white_point[1] < 0.0
            || self.white_point[1] > 1.0
        {
            return false;
        }

        true
    }

    /// Dynamic range expressed in decibels (10 * log10 ratio).
    #[must_use]
    pub fn dynamic_range_db(&self) -> f64 {
        if self.min_luminance <= 0.0 {
            return 0.0;
        }
        10.0 * (self.max_luminance / self.min_luminance).log10()
    }
}

/// Associates mastering display metadata with a specific frame.
#[derive(Debug, Clone)]
pub struct MasteringMetadata {
    /// Mastering display descriptor.
    pub display: MasteringDisplay,
    /// Frame index this metadata applies to.
    pub frame_id: u64,
}

impl MasteringMetadata {
    /// Create a new mastering metadata record.
    #[must_use]
    pub const fn new(display: MasteringDisplay, frame_id: u64) -> Self {
        Self { display, frame_id }
    }
}

/// Compute target nits for content given the display's peak capability.
///
/// Returns the value scaled so that `content_nits` maps onto the
/// `display_peak` headroom, clamped to `[0, display_peak]`.
#[must_use]
pub fn compute_target_nits(content_nits: f64, display_peak: f64) -> f64 {
    if display_peak <= 0.0 || content_nits <= 0.0 {
        return 0.0;
    }
    // Simple linear headroom scaling
    let ratio = content_nits / display_peak;
    (ratio * display_peak).clamp(0.0, display_peak)
}

/// Format mastering display metadata as an MDCV/CLL string.
///
/// The returned string can be used in container metadata such as
/// HEVC SEI messages or Matroska `MasteringMetadata` elements.
#[must_use]
pub fn hdr_static_to_string(display: &MasteringDisplay) -> String {
    let [r, g, b] = display.primaries;
    format!(
        "G({gx:.4},{gy:.4})B({bx:.4},{by:.4})R({rx:.4},{ry:.4})WP({wx:.4},{wy:.4})L({lmax:.0},{lmin:.4})",
        gx = g[0],
        gy = g[1],
        bx = b[0],
        by = b[1],
        rx = r[0],
        ry = r[1],
        wx = display.white_point[0],
        wy = display.white_point[1],
        lmax = display.max_luminance,
        lmin = display.min_luminance,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p3_d65_1000nit_valid() {
        let disp = MasteringDisplay::p3_d65_1000nit();
        assert!(disp.validate(), "p3_d65_1000nit should be valid");
    }

    #[test]
    fn test_bt2020_4000nit_valid() {
        let disp = MasteringDisplay::bt2020_4000nit();
        assert!(disp.validate(), "bt2020_4000nit should be valid");
    }

    #[test]
    fn test_dynamic_range_db_1000nit() {
        let disp = MasteringDisplay::p3_d65_1000nit();
        let db = disp.dynamic_range_db();
        // 1000 / 0.005 = 200,000 → ~53 dB
        assert!(db > 50.0 && db < 60.0, "Expected ~53 dB, got {db}");
    }

    #[test]
    fn test_dynamic_range_db_4000nit() {
        let disp = MasteringDisplay::bt2020_4000nit();
        let db = disp.dynamic_range_db();
        // 4000 / 0.005 = 800,000 → ~59 dB
        assert!(db > 55.0 && db < 65.0, "Expected ~59 dB, got {db}");
    }

    #[test]
    fn test_validate_invalid_luminance_inverted() {
        let mut disp = MasteringDisplay::p3_d65_1000nit();
        disp.min_luminance = 2000.0;
        assert!(!disp.validate());
    }

    #[test]
    fn test_validate_invalid_max_luminance_too_low() {
        let mut disp = MasteringDisplay::p3_d65_1000nit();
        disp.max_luminance = 10.0;
        assert!(!disp.validate());
    }

    #[test]
    fn test_validate_invalid_primary_out_of_range() {
        let mut disp = MasteringDisplay::p3_d65_1000nit();
        disp.primaries[0][0] = 1.5; // x > 1.0
        assert!(!disp.validate());
    }

    #[test]
    fn test_compute_target_nits_identity() {
        let result = compute_target_nits(500.0, 1000.0);
        assert!((result - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_target_nits_clamp() {
        let result = compute_target_nits(5000.0, 1000.0);
        assert!((result - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_target_nits_zero_display() {
        let result = compute_target_nits(500.0, 0.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_hdr_static_to_string_contains_luminance() {
        let disp = MasteringDisplay::p3_d65_1000nit();
        let s = hdr_static_to_string(&disp);
        assert!(s.contains("1000"), "String should contain peak luminance");
        assert!(s.contains("WP"), "String should contain white point");
        assert!(s.contains("G("), "String should contain green primary");
    }

    #[test]
    fn test_mastering_metadata_frame_id() {
        let disp = MasteringDisplay::p3_d65_1000nit();
        let meta = MasteringMetadata::new(disp, 42);
        assert_eq!(meta.frame_id, 42);
    }

    #[test]
    fn test_color_volume_dynamic_range_stops() {
        let cv = ColorVolume::new(1000.0, 0.005, 1000, 400);
        let stops = cv.dynamic_range_stops();
        // log2(1000 / 0.005) = log2(200000) ≈ 17.6
        assert!(
            stops > 17.0 && stops < 18.0,
            "Expected ~17.6 stops, got {stops}"
        );
    }

    #[test]
    fn test_color_volume_dynamic_range_zero_min() {
        let cv = ColorVolume::new(1000.0, 0.0, 1000, 400);
        assert_eq!(cv.dynamic_range_stops(), 0.0);
    }
}
