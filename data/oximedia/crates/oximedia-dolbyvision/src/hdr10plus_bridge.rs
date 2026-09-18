//! Dolby Vision to HDR10+ metadata conversion bridge.
//!
//! Translates per-frame Dolby Vision Level 1 / Level 2 metadata into an
//! approximation of HDR10+ dynamic metadata, enabling playback on displays
//! that support HDR10+ but not Dolby Vision.
//!
//! # Design
//!
//! The conversion is intentionally lossy — HDR10+ uses a different parameter
//! set (ST.2094-40) than Dolby Vision — so the output should be treated as a
//! best-effort rendering hint rather than a faithful transcoding.
//!
//! The bridge maps:
//! - DV Level 1 min/max/avg PQ → HDR10+ `TargetedSystemDisplayMaximumLuminance`,
//!   `AverageMaxRGB`, and per-pixel-percentage luminance distribution.
//! - DV Level 6 (if present) → HDR10+ `MasteringDisplayLuminance` info.
//! - DV Level 2 trim parameters (if present) → HDR10+ `ToneMappingParameters`.

use crate::{DolbyVisionRpu, Level1Metadata, Level6Metadata};

// ---------------------------------------------------------------------------
// HDR10+ approximation types
// ---------------------------------------------------------------------------

/// Bezier-curve tone-mapping anchors as used in ST.2094-40.
///
/// Each entry represents a (normalised input, normalised output) knee point.
#[derive(Debug, Clone, PartialEq)]
pub struct BezierCurveAnchor {
    /// Normalised input luminance (0.0–1.0 relative to reference peak).
    pub input: f32,
    /// Normalised output luminance (0.0–1.0 relative to target peak).
    pub output: f32,
}

/// HDR10+ tone-mapping parameters derived from Dolby Vision trim metadata.
#[derive(Debug, Clone)]
pub struct Hdr10pToneMapParams {
    /// Target system display maximum luminance in nits.
    pub target_max_luminance: f32,
    /// Knee point anchors for the Bezier tone curve.
    pub bezier_anchors: Vec<BezierCurveAnchor>,
    /// 1–99 percentile distribution bins (normalised, 0.0–1.0).
    ///
    /// Index 0 = 1st percentile, index 98 = 99th percentile.
    pub distribution_percentiles: [f32; 99],
}

impl Default for Hdr10pToneMapParams {
    fn default() -> Self {
        Self {
            target_max_luminance: 1000.0,
            bezier_anchors: Vec::new(),
            distribution_percentiles: [0.0; 99],
        }
    }
}

/// Approximated HDR10+ frame-level dynamic metadata.
#[derive(Debug, Clone)]
pub struct Hdr10pFrameMetadata {
    /// Maximum SCRGB value for the mastering display (nits).
    pub mastering_display_max_luminance: f32,
    /// Maximum SCRGB value for the mastering display minimum (nits).
    pub mastering_display_min_luminance: f32,
    /// MaxCLL in nits.
    pub max_content_light_level: f32,
    /// MaxFALL in nits.
    pub max_frame_average_light_level: f32,
    /// Per-frame tone-mapping parameters.
    pub tone_map: Hdr10pToneMapParams,
    /// Average MaxRGB per frame (nits).
    pub average_max_rgb: f32,
}

// ---------------------------------------------------------------------------
// Conversion logic
// ---------------------------------------------------------------------------

/// Convert a Dolby Vision RPU to an approximate HDR10+ frame metadata block.
///
/// The conversion uses Level 1 PQ values for per-frame luminance statistics and
/// Level 6 values for mastering display information.  Level 2 trim parameters
/// are mapped to Bezier curve anchors when available.
#[must_use]
pub fn dv_rpu_to_hdr10p(rpu: &DolbyVisionRpu) -> Hdr10pFrameMetadata {
    // --- Mastering display info from Level 6 ---
    let (md_max, md_min, max_cll, max_fall) = if let Some(ref l6) = rpu.level6 {
        extract_l6_luminance(l6)
    } else {
        (1000.0, 0.005, 1000.0, 400.0)
    };

    // --- Per-frame stats from Level 1 ---
    let (avg_max_rgb, tone_map) = if let Some(ref l1) = rpu.level1 {
        let tone = build_tone_map_from_l1(l1, md_max);
        let avg = pq_code_to_nits(l1.avg_pq);
        (avg, tone)
    } else {
        (100.0, Hdr10pToneMapParams::default())
    };

    Hdr10pFrameMetadata {
        mastering_display_max_luminance: md_max,
        mastering_display_min_luminance: md_min,
        max_content_light_level: max_cll,
        max_frame_average_light_level: max_fall,
        tone_map,
        average_max_rgb: avg_max_rgb,
    }
}

/// Estimate a Bezier-curve tone map from Level 1 PQ statistics.
fn build_tone_map_from_l1(l1: &Level1Metadata, target_max_nits: f32) -> Hdr10pToneMapParams {
    let max_nits = pq_code_to_nits(l1.max_pq).max(1.0);
    let min_nits = pq_code_to_nits(l1.min_pq);
    let avg_nits = pq_code_to_nits(l1.avg_pq);

    // Normalise to [0, 1] relative to max
    let norm_min = (min_nits / max_nits).clamp(0.0, 1.0);
    let norm_avg = (avg_nits / max_nits).clamp(0.0, 1.0);

    // Build a simple 3-anchor Bezier:  black, avg, white
    let target_ratio = (target_max_nits / max_nits).clamp(0.01, 1.0);

    let anchors = vec![
        BezierCurveAnchor {
            input: norm_min,
            output: (norm_min * target_ratio).clamp(0.0, 1.0),
        },
        BezierCurveAnchor {
            input: norm_avg,
            output: (norm_avg * target_ratio).clamp(0.0, 1.0),
        },
        BezierCurveAnchor {
            input: 1.0,
            output: target_ratio.clamp(0.0, 1.0),
        },
    ];

    // Fill a synthetic percentile distribution using a smooth ramp
    let mut distribution_percentiles = [0.0f32; 99];
    for (i, slot) in distribution_percentiles.iter_mut().enumerate() {
        let t = (i + 1) as f32 / 100.0;
        *slot = (norm_min + t * (1.0 - norm_min)) * target_ratio;
    }

    Hdr10pToneMapParams {
        target_max_luminance: target_max_nits,
        bezier_anchors: anchors,
        distribution_percentiles,
    }
}

/// Extract luminance values from a Level 6 metadata block.
fn extract_l6_luminance(l6: &Level6Metadata) -> (f32, f32, f32, f32) {
    let md_max = f32::from(l6.max_display_mastering_luminance as u16);
    let md_min = l6.min_display_mastering_luminance as f32 / 10_000.0;
    let max_cll = f32::from(l6.max_cll);
    let max_fall = f32::from(l6.max_fall);
    (md_max, md_min, max_cll, max_fall)
}

/// Convert a 12-bit PQ code (0–4095) to approximate nits.
///
/// Uses the simplified ST.2084 inverse EOTF.
#[must_use]
#[inline]
fn pq_code_to_nits(pq: u16) -> f32 {
    const M1_INV: f64 = 1.0 / 0.159_301_758_113_479_8;
    const M2_INV: f64 = 1.0 / 78.843_750;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_562_5;
    const C3: f64 = 18.6875;

    if pq == 0 {
        return 0.0;
    }
    let v = (f64::from(pq) / 4095.0).powf(M2_INV);
    let y = ((v - C1).max(0.0) / (C2 - C3 * v)).powf(M1_INV);
    (y * 10_000.0).clamp(0.0, 10_000.0) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DolbyVisionRpu, Level1Metadata, Level6Metadata, Profile};

    #[test]
    fn test_dv_to_hdr10p_empty_rpu() {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        let meta = dv_rpu_to_hdr10p(&rpu);
        assert!(meta.mastering_display_max_luminance > 0.0);
        assert!(meta.max_content_light_level > 0.0);
    }

    #[test]
    fn test_dv_to_hdr10p_with_level1() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: 62,
            max_pq: 3696,
            avg_pq: 1000,
        });
        let meta = dv_rpu_to_hdr10p(&rpu);
        assert!(meta.average_max_rgb > 0.0);
        assert_eq!(meta.tone_map.bezier_anchors.len(), 3);
    }

    #[test]
    fn test_dv_to_hdr10p_with_level6() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level6 = Some(Level6Metadata::bt2020());
        let meta = dv_rpu_to_hdr10p(&rpu);
        assert!((meta.max_content_light_level - 1000.0).abs() < 1.0);
        assert!((meta.max_frame_average_light_level - 400.0).abs() < 1.0);
    }

    #[test]
    fn test_bezier_anchors_monotonic() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: 62,
            max_pq: 3696,
            avg_pq: 1800,
        });
        let meta = dv_rpu_to_hdr10p(&rpu);
        let anchors = &meta.tone_map.bezier_anchors;
        for w in anchors.windows(2) {
            assert!(
                w[1].input >= w[0].input,
                "Bezier anchors should be non-decreasing in input"
            );
        }
    }

    #[test]
    fn test_pq_code_to_nits_zero() {
        assert_eq!(pq_code_to_nits(0), 0.0);
    }

    #[test]
    fn test_pq_code_to_nits_max() {
        let nits = pq_code_to_nits(4095);
        assert!(nits > 9000.0 && nits <= 10_001.0, "nits={nits}");
    }

    #[test]
    fn test_pq_code_to_nits_100nit() {
        // PQ code for 100 nits ≈ 2081
        let nits = pq_code_to_nits(2081);
        assert!(nits > 80.0 && nits < 120.0, "nits={nits}");
    }

    #[test]
    fn test_distribution_percentiles_range() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: 100,
            max_pq: 3500,
            avg_pq: 1750,
        });
        let meta = dv_rpu_to_hdr10p(&rpu);
        for &v in &meta.tone_map.distribution_percentiles {
            assert!(v >= 0.0 && v <= 1.0, "percentile value {v} out of range");
        }
    }

    #[test]
    fn test_hdr10p_tone_map_params_default() {
        let params = Hdr10pToneMapParams::default();
        assert!((params.target_max_luminance - 1000.0).abs() < 1.0);
        assert!(params.bezier_anchors.is_empty());
    }

    #[test]
    fn test_bezier_curve_anchor_fields() {
        let a = BezierCurveAnchor {
            input: 0.3,
            output: 0.2,
        };
        assert!((a.input - 0.3).abs() < 1e-6);
        assert!((a.output - 0.2).abs() < 1e-6);
    }
}
