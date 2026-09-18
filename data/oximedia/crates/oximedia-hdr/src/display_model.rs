//! Display model characterisation for HDR tone-mapping target displays.
//!
//! A `DisplayModel` captures the key perceptual characteristics of a display
//! (peak luminance, black level, colour gamut, and optional MaxCLL), and
//! provides a `tone_map_to` method that applies a display-aware HDR→SDR or
//! HDR→HDR tone mapping between two models.

use crate::gamut::ColorGamut;
use crate::{HdrError, Result};

// ── DisplayModel ──────────────────────────────────────────────────────────────

/// Physical and perceptual characteristics of a display device.
///
/// Used as the source (mastering) or target (viewing) model for
/// display-referred tone mapping via [`DisplayModel::tone_map_to`].
#[derive(Debug, Clone)]
pub struct DisplayModel {
    /// Peak white luminance of the display in nits (cd/m²).
    pub peak_luminance_nits: f32,
    /// Absolute black level of the display in nits.
    /// Typical OLED: 0.0001. Typical LCD: 0.05.
    pub black_level_nits: f32,
    /// Primary colour gamut of the display.
    pub color_gamut: ColorGamut,
    /// Optional override for the maximum single-pixel luminance (MaxCLL) in nits.
    ///
    /// When `Some`, tone mapping uses this as the effective HDR peak rather
    /// than `peak_luminance_nits`.  When `None`, `peak_luminance_nits` is used.
    pub max_cll: Option<f32>,
}

impl DisplayModel {
    /// Construct a display model with explicit parameters.
    pub fn new(
        peak_luminance_nits: f32,
        black_level_nits: f32,
        color_gamut: ColorGamut,
        max_cll: Option<f32>,
    ) -> Self {
        Self {
            peak_luminance_nits,
            black_level_nits,
            color_gamut,
            max_cll,
        }
    }

    /// Reference SDR display: BT.709 gamut, 100 nit peak, 0.05 nit black.
    ///
    /// Corresponds to an ITU-R BT.1886 reference monitor in a dim-surround
    /// environment.
    pub fn sdr_rec709() -> Self {
        Self {
            peak_luminance_nits: 100.0,
            black_level_nits: 0.05,
            color_gamut: ColorGamut::Rec709,
            max_cll: None,
        }
    }

    /// HDR display: BT.2020 gamut, 1 000 nit peak, 0.005 nit black.
    ///
    /// Typical OLED/miniLED consumer HDR display (e.g. HDR10 / HLG certified).
    pub fn hdr_rec2020_1000nit() -> Self {
        Self {
            peak_luminance_nits: 1000.0,
            black_level_nits: 0.005,
            color_gamut: ColorGamut::Rec2020,
            max_cll: Some(1000.0),
        }
    }

    /// HDR display: BT.2020 gamut, 4 000 nit peak, 0.0005 nit black.
    ///
    /// High-end professional HDR reference monitor (e.g. Sony BVM-HX310).
    pub fn hdr_rec2020_4000nit() -> Self {
        Self {
            peak_luminance_nits: 4000.0,
            black_level_nits: 0.0005,
            color_gamut: ColorGamut::Rec2020,
            max_cll: Some(4000.0),
        }
    }

    /// Effective HDR peak luminance for tone mapping decisions.
    ///
    /// Returns `max_cll` if set, otherwise `peak_luminance_nits`.
    pub fn effective_peak_nits(&self) -> f32 {
        self.max_cll.unwrap_or(self.peak_luminance_nits)
    }

    /// Dynamic range ratio: `peak / black_level` (linear).
    ///
    /// Represents the contrast ratio of the display.  Higher values indicate
    /// a wider dynamic range (more detail in shadows relative to peak white).
    pub fn dynamic_range_ratio(&self) -> f32 {
        self.peak_luminance_nits / self.black_level_nits.max(1e-6)
    }

    /// Tone-map a single linear-light RGB pixel from `self` (source display)
    /// to `target` (destination display).
    ///
    /// The pixel values are expected to be normalised to [0, 1] relative to
    /// `self.effective_peak_nits()`.  The output is normalised to [0, 1]
    /// relative to `target.effective_peak_nits()`.
    ///
    /// The mapping pipeline is:
    /// 1. Convert from source normalised [0, 1] to absolute nits
    /// 2. Apply black-level lift to account for the source display's black
    /// 3. Compress/expand luminance to the target display's range using a
    ///    display-aware roll-off curve
    /// 4. Apply black-level offset of the target display
    /// 5. Normalise back to [0, 1] relative to the target peak
    ///
    /// # Errors
    /// Returns `HdrError::ToneMappingError` if either display peak is zero or
    /// negative.
    pub fn tone_map_to(&self, target: &DisplayModel, pixel: &[f32; 3]) -> Result<[f32; 3]> {
        let src_peak = self.effective_peak_nits();
        let dst_peak = target.effective_peak_nits();

        if src_peak <= 0.0 {
            return Err(HdrError::ToneMappingError(
                "source display peak luminance must be positive".to_string(),
            ));
        }
        if dst_peak <= 0.0 {
            return Err(HdrError::ToneMappingError(
                "target display peak luminance must be positive".to_string(),
            ));
        }

        // Compute luminance for scaling (BT.2100 coefficients)
        let lum_norm = 0.2627 * pixel[0] + 0.6780 * pixel[1] + 0.0593 * pixel[2];

        // Convert to absolute nits
        let lum_nits = lum_norm * src_peak;

        // Black-level adaptation: lift source black to scene-linear reference
        let lum_adapted = lum_nits + self.black_level_nits;

        // Map luminance from source to target using a display-aware roll-off.
        // We use the BT.2390 roll-off function which was designed specifically
        // for cross-display HDR tone mapping.
        let mapped_nits = display_rolloff(lum_adapted, src_peak, dst_peak, self.black_level_nits);

        // Apply target black offset
        let final_nits = (mapped_nits - target.black_level_nits).max(0.0);

        // Normalise back to [0, 1]
        let mapped_norm = final_nits / dst_peak;

        // Scale the RGB channels proportionally to preserve hue
        let (r_out, g_out, b_out) = if lum_norm > 1e-7 {
            let scale = mapped_norm / lum_norm;
            (
                (pixel[0] * scale).clamp(0.0, 1.0),
                (pixel[1] * scale).clamp(0.0, 1.0),
                (pixel[2] * scale).clamp(0.0, 1.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        Ok([r_out, g_out, b_out])
    }

    /// Is this display a wide-gamut HDR display (peak > 400 nits)?
    pub fn is_hdr(&self) -> bool {
        self.peak_luminance_nits > 400.0
    }

    /// Is this display an SDR display (peak <= 200 nits)?
    pub fn is_sdr(&self) -> bool {
        self.peak_luminance_nits <= 200.0
    }
}

// ── Display roll-off ──────────────────────────────────────────────────────────

/// BT.2390-inspired display-aware roll-off tone mapping function.
///
/// Maps an absolute luminance value `lum_nits` from a source display with
/// peak `src_peak` to a target display with peak `dst_peak`.
///
/// For downward mapping (src_peak > dst_peak): applies a smooth shoulder
/// that compresses highlights while preserving shadows and mid-tones.
///
/// For upward mapping (src_peak < dst_peak): applies a gentle expansion
/// that lifts highlights toward the target peak.
#[inline]
fn display_rolloff(lum_nits: f32, src_peak: f32, dst_peak: f32, black_level: f32) -> f32 {
    if lum_nits <= 0.0 {
        return 0.0;
    }

    // Normalise to source peak
    let x = (lum_nits / src_peak).clamp(0.0, 2.0);

    // Compute the tone-mapped normalised output using a modified Reinhard with
    // display-specific white point.
    let peak_ratio = dst_peak / src_peak;

    let mapped_norm = if peak_ratio >= 1.0 {
        // Upward mapping: gentle expansion.
        // Use a linear-plus-boost function that gradually lifts highlights.
        let boost = (peak_ratio - 1.0).min(40.0);
        let expanded = x * (1.0 + boost * x);
        expanded.min(peak_ratio)
    } else {
        // Downward mapping: modified Reinhard with parametric white point.
        // w is chosen so that f(1) = peak_ratio (maps source peak to destination peak).
        // Reinhard: f(x) = x * (1 + x/w²) / (1 + x)
        //   f(1) = peak_ratio => solve for w:
        //   peak_ratio * (1 + 1) = 1 + 1/w²  => w² = 1/(2*peak_ratio - 1)
        //
        // For very dark targets, fall back to simple Reinhard.
        let w_sq = if peak_ratio > 0.5 {
            1.0 / (2.0 * peak_ratio - 1.0).max(1e-6)
        } else {
            4.0 // fallback for very dark targets
        };
        x * (1.0 + x / w_sq) / (1.0 + x)
    };

    // Bring back into nit space
    let result_nits = mapped_norm * dst_peak;

    // Add back the black level of the *target* display
    result_nits + black_level
}

// ── Extended display characterisation types ───────────────────────────────────

/// CIE 1931 xy chromaticity coordinates for display primaries.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayPrimaries {
    /// Red primary xy chromaticity.
    pub red_xy: (f32, f32),
    /// Green primary xy chromaticity.
    pub green_xy: (f32, f32),
    /// Blue primary xy chromaticity.
    pub blue_xy: (f32, f32),
}

/// CIE 1931 xy white-point chromaticity.
#[derive(Debug, Clone, PartialEq)]
pub struct WhitePoint {
    pub x: f32,
    pub y: f32,
}

impl WhitePoint {
    /// D65 standard illuminant.
    pub fn d65() -> Self {
        Self {
            x: 0.3127,
            y: 0.3290,
        }
    }

    /// DCI white point (P3 theatre).
    pub fn dci() -> Self {
        Self {
            x: 0.3140,
            y: 0.3510,
        }
    }
}

/// Electro-optical transfer function (EOTF) / gamma characteristic of a display.
#[derive(Debug, Clone, PartialEq)]
pub enum DisplayGamma {
    /// IEC 61966-2-1 sRGB piece-wise curve.
    Srgb,
    /// ITU-R BT.1886 reference EOTF for BT.709 displays (power 2.4 with black lift).
    Bt1886,
    /// SMPTE ST 2084 / BT.2100 Perceptual Quantiser (PQ).
    Pq,
    /// ITU-R BT.2100 Hybrid Log-Gamma (HLG).
    Hlg,
    /// Simple power-law gamma curve: `L = V ^ gamma`.
    Pure(f32),
}

/// HDR container or signalling format supported by the display.
#[derive(Debug, Clone, PartialEq)]
pub enum HdrFormatDisplay {
    /// SMPTE ST 2086 + CTA-861 static metadata (HDR10).
    Hdr10,
    /// China HDR Vivid standard (UHD Alliance).
    HdrVivid,
    /// Dolby Vision (profile-dependent single- or dual-layer).
    DolbyVision,
    /// SMPTE ST 2094-40 dynamic metadata (HDR10+).
    HdrPlus,
    /// ITU-R BT.2100 Hybrid Log-Gamma broadcast (HLG).
    Hlg,
}

/// Recommended tone-mapping parameters derived from a [`FullDisplayModel`].
#[derive(Debug, Clone)]
pub struct ToneMapDisplayParams {
    /// Source (mastering) peak luminance in nits.
    pub input_peak_nits: f32,
    /// Target display peak luminance in nits.
    pub output_peak_nits: f32,
    /// Target display minimum (black-level) luminance in nits.
    pub output_black_nits: f32,
    /// Linear contrast ratio of the target display (`peak / black`).
    pub contrast_ratio: f32,
    /// Name of the recommended tone-mapping algorithm for this combination.
    ///
    /// One of `"BT.2446A"`, `"BT.2446C"`, `"BT.2390"`, or `"ACES"`.
    pub recommended_algorithm: String,
}

/// Full display characterisation for display-model-aware HDR tone mapping.
///
/// Unlike the lightweight [`DisplayModel`], this struct carries the complete
/// set of display characteristics needed to recommend a tone-mapping algorithm
/// and to populate HDR mastering metadata in a container.
#[derive(Debug, Clone)]
pub struct FullDisplayModel {
    /// Human-readable display name / profile label.
    pub name: String,
    /// Peak white luminance (cd/m²).
    pub max_luminance_nits: f32,
    /// Black-level luminance (cd/m²). Typical OLED: 0.0001, LCD: 0.05.
    pub min_luminance_nits: f32,
    /// CIE 1931 xy colour primaries.
    pub primaries: DisplayPrimaries,
    /// CIE 1931 xy white point.
    pub white_point: WhitePoint,
    /// Display EOTF / gamma characteristic.
    pub gamma: DisplayGamma,
    /// Whether this display supports HDR signalling at all.
    pub hdr_capable: bool,
    /// HDR container formats supported by this display.
    pub hdr_formats: Vec<HdrFormatDisplay>,
}

impl FullDisplayModel {
    // ── Standard reference display presets ────────────────────────────────────

    /// ITU-R BT.709 SDR reference monitor (100 nit, BT.1886 EOTF).
    ///
    /// Corresponds to an EBU/SMPTE-level SDR reference display in a dim
    /// surround environment (L* = 50 cd/m² adapted white).
    pub fn rec709_reference() -> Self {
        Self {
            name: "Rec. 709 SDR Reference".to_string(),
            max_luminance_nits: 100.0,
            min_luminance_nits: 0.05,
            primaries: DisplayPrimaries {
                red_xy: (0.640, 0.330),
                green_xy: (0.300, 0.600),
                blue_xy: (0.150, 0.060),
            },
            white_point: WhitePoint::d65(),
            gamma: DisplayGamma::Bt1886,
            hdr_capable: false,
            hdr_formats: vec![],
        }
    }

    /// DCI-P3 D65 cinema reference display (48 nit screen, PQ EOTF).
    ///
    /// Used for mastering theatrical content; also common in Apple Pro Display XDR
    /// "P3 Reference" mode.
    pub fn p3_d65_reference() -> Self {
        Self {
            name: "P3-D65 Cinema Reference".to_string(),
            max_luminance_nits: 48.0,
            min_luminance_nits: 0.005,
            primaries: DisplayPrimaries {
                red_xy: (0.680, 0.320),
                green_xy: (0.265, 0.690),
                blue_xy: (0.150, 0.060),
            },
            white_point: WhitePoint::d65(),
            gamma: DisplayGamma::Pq,
            hdr_capable: true,
            hdr_formats: vec![HdrFormatDisplay::Hdr10, HdrFormatDisplay::DolbyVision],
        }
    }

    /// Rec. 2020 / BT.2100 HDR10 consumer display at 1 000 nit peak.
    ///
    /// Representative of a high-end OLED or miniLED consumer TV certified for
    /// HDR10 / HDR10+.
    pub fn rec2020_1000nit() -> Self {
        Self {
            name: "Rec. 2020 HDR10 1000 nit".to_string(),
            max_luminance_nits: 1000.0,
            min_luminance_nits: 0.001,
            primaries: DisplayPrimaries {
                red_xy: (0.708, 0.292),
                green_xy: (0.170, 0.797),
                blue_xy: (0.131, 0.046),
            },
            white_point: WhitePoint::d65(),
            gamma: DisplayGamma::Pq,
            hdr_capable: true,
            hdr_formats: vec![HdrFormatDisplay::Hdr10, HdrFormatDisplay::HdrPlus],
        }
    }

    /// Rec. 2020 / BT.2100 professional HDR mastering monitor at 4 000 nit peak.
    ///
    /// Corresponds to a Sony BVM-HX310 or similar professional HDR reference
    /// monitor used for HDR mastering.
    pub fn rec2020_4000nit() -> Self {
        Self {
            name: "Rec. 2020 HDR Mastering 4000 nit".to_string(),
            max_luminance_nits: 4000.0,
            min_luminance_nits: 0.0005,
            primaries: DisplayPrimaries {
                red_xy: (0.708, 0.292),
                green_xy: (0.170, 0.797),
                blue_xy: (0.131, 0.046),
            },
            white_point: WhitePoint::d65(),
            gamma: DisplayGamma::Pq,
            hdr_capable: true,
            hdr_formats: vec![
                HdrFormatDisplay::Hdr10,
                HdrFormatDisplay::HdrPlus,
                HdrFormatDisplay::DolbyVision,
            ],
        }
    }

    /// Rec. 2020 / BT.2100 MaxCLL reference display at 10 000 nit peak.
    ///
    /// Represents the absolute MaxCLL reference level defined by SMPTE ST 2084.
    /// No consumer display reaches this level; it is used for scene-referred
    /// reference and MaxCLL measurement.
    pub fn rec2020_10000nit() -> Self {
        Self {
            name: "Rec. 2020 MaxCLL Reference 10000 nit".to_string(),
            max_luminance_nits: 10000.0,
            min_luminance_nits: 0.0001,
            primaries: DisplayPrimaries {
                red_xy: (0.708, 0.292),
                green_xy: (0.170, 0.797),
                blue_xy: (0.131, 0.046),
            },
            white_point: WhitePoint::d65(),
            gamma: DisplayGamma::Pq,
            hdr_capable: true,
            hdr_formats: vec![
                HdrFormatDisplay::Hdr10,
                HdrFormatDisplay::HdrPlus,
                HdrFormatDisplay::DolbyVision,
                HdrFormatDisplay::HdrVivid,
                HdrFormatDisplay::Hlg,
            ],
        }
    }

    // ── Tone-mapping parameter recommendation ─────────────────────────────────

    /// Compute recommended tone-mapping parameters to map content mastered at
    /// `source_peak_nits` onto this display.
    ///
    /// Algorithm selection heuristic:
    /// - `BT.2446A`  — source ≤ 1 000 nit, SDR target (≤ 100 nit)
    /// - `BT.2446C`  — source > 1 000 nit, SDR target (≤ 100 nit)
    /// - `ACES`      — HDR target ≥ 1 000 nit
    /// - `BT.2390`   — all other cases (mid-range HDR targets)
    pub fn compute_tone_map_params(&self, source_peak_nits: f32) -> ToneMapDisplayParams {
        let output_peak = self.max_luminance_nits;
        let output_black = self.min_luminance_nits;
        let contrast_ratio = output_peak / output_black.max(1e-6);

        let recommended_algorithm = if source_peak_nits <= 1000.0 && output_peak <= 100.0 {
            "BT.2446A".to_string()
        } else if output_peak <= 100.0 {
            "BT.2446C".to_string()
        } else if output_peak >= 1000.0 {
            "ACES".to_string()
        } else {
            "BT.2390".to_string()
        };

        ToneMapDisplayParams {
            input_peak_nits: source_peak_nits,
            output_peak_nits: output_peak,
            output_black_nits: output_black,
            contrast_ratio,
            recommended_algorithm,
        }
    }

    /// Return the linear contrast ratio of this display.
    pub fn contrast_ratio(&self) -> f32 {
        self.max_luminance_nits / self.min_luminance_nits.max(1e-6)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // 1. sdr_rec709 factory
    #[test]
    fn test_sdr_rec709_properties() {
        let m = DisplayModel::sdr_rec709();
        assert!(approx(m.peak_luminance_nits, 100.0, 0.1));
        assert!(approx(m.black_level_nits, 0.05, 0.01));
        assert_eq!(m.color_gamut, ColorGamut::Rec709);
        assert!(m.max_cll.is_none());
        assert!(m.is_sdr());
        assert!(!m.is_hdr());
    }

    // 2. hdr_rec2020_1000nit factory
    #[test]
    fn test_hdr_1000nit_properties() {
        let m = DisplayModel::hdr_rec2020_1000nit();
        assert!(approx(m.peak_luminance_nits, 1000.0, 0.1));
        assert_eq!(m.color_gamut, ColorGamut::Rec2020);
        assert!(m.is_hdr());
        assert!(!m.is_sdr());
    }

    // 3. hdr_rec2020_4000nit factory
    #[test]
    fn test_hdr_4000nit_properties() {
        let m = DisplayModel::hdr_rec2020_4000nit();
        assert!(approx(m.peak_luminance_nits, 4000.0, 0.1));
        assert!(m.is_hdr());
    }

    // 4. Black pixel maps to near-black on any target
    #[test]
    fn test_black_pixel_maps_near_black() {
        let src = DisplayModel::hdr_rec2020_1000nit();
        let dst = DisplayModel::sdr_rec709();
        let result = src.tone_map_to(&dst, &[0.0, 0.0, 0.0]).expect("tone_map");
        for &c in result.iter() {
            assert!(c <= 0.05, "black pixel channel out of range: {c}");
        }
    }

    // 5. White pixel maps within [0, 1] on SDR target
    #[test]
    fn test_white_pixel_maps_in_range() {
        let src = DisplayModel::hdr_rec2020_1000nit();
        let dst = DisplayModel::sdr_rec709();
        let result = src.tone_map_to(&dst, &[1.0, 1.0, 1.0]).expect("tone_map");
        for &c in result.iter() {
            assert!(
                (0.0..=1.0).contains(&c),
                "white pixel channel out of range: {c}"
            );
        }
    }

    // 6. Mapping to same display should be near-identity for mid-tones
    #[test]
    fn test_identity_mapping_same_display() {
        let m = DisplayModel::hdr_rec2020_1000nit();
        let pixel = [0.3f32, 0.5, 0.4];
        let result = m.tone_map_to(&m, &pixel).expect("identity");
        for i in 0..3 {
            assert!(
                approx(result[i], pixel[i], 0.15),
                "identity mapping large deviation at channel {i}: {} vs {}",
                result[i],
                pixel[i]
            );
        }
    }

    // 7. Dynamic range ratio
    #[test]
    fn test_dynamic_range_ratio() {
        let m = DisplayModel::hdr_rec2020_1000nit();
        let ratio = m.dynamic_range_ratio();
        assert!(
            ratio > 100_000.0,
            "HDR display should have very high ratio: {ratio}"
        );
    }

    // 8. effective_peak_nits uses max_cll when set
    #[test]
    fn test_effective_peak_uses_max_cll() {
        let m = DisplayModel::new(4000.0, 0.0005, ColorGamut::Rec2020, Some(2000.0));
        assert!(approx(m.effective_peak_nits(), 2000.0, 0.1));
    }

    // 9. effective_peak_nits falls back to peak_luminance when max_cll is None
    #[test]
    fn test_effective_peak_fallback() {
        let m = DisplayModel::sdr_rec709();
        assert!(approx(m.effective_peak_nits(), 100.0, 0.1));
    }

    // 10. Zero peak luminance returns error
    #[test]
    fn test_zero_peak_error() {
        let bad = DisplayModel::new(0.0, 0.0, ColorGamut::Rec709, None);
        let good = DisplayModel::sdr_rec709();
        assert!(bad.tone_map_to(&good, &[0.5, 0.5, 0.5]).is_err());
        assert!(good.tone_map_to(&bad, &[0.5, 0.5, 0.5]).is_err());
    }

    // 11. HDR→SDR mapped luminance is lower than HDR→HDR (both to same normalised range)
    #[test]
    fn test_hdr_to_sdr_compresses_highlights() {
        let src = DisplayModel::hdr_rec2020_4000nit();
        let sdr = DisplayModel::sdr_rec709();
        let hdr1000 = DisplayModel::hdr_rec2020_1000nit();

        let pixel = [0.9f32, 0.9, 0.9];
        let to_sdr = src.tone_map_to(&sdr, &pixel).expect("to_sdr");
        let to_hdr = src.tone_map_to(&hdr1000, &pixel).expect("to_hdr");

        // The HDR→SDR result should be lower (compressed) than HDR→HDR
        let lum_sdr = 0.2627 * to_sdr[0] + 0.6780 * to_sdr[1] + 0.0593 * to_sdr[2];
        let lum_hdr = 0.2627 * to_hdr[0] + 0.6780 * to_hdr[1] + 0.0593 * to_hdr[2];
        assert!(
            lum_sdr <= lum_hdr + 0.01,
            "SDR output ({lum_sdr}) should not exceed HDR output ({lum_hdr})"
        );
    }

    // ── FullDisplayModel tests ────────────────────────────────────────────────

    // 12. rec709_reference properties
    #[test]
    fn test_full_rec709_reference_properties() {
        let m = FullDisplayModel::rec709_reference();
        assert!(approx(m.max_luminance_nits, 100.0, 0.1));
        assert!(!m.hdr_capable);
        assert!(m.hdr_formats.is_empty());
        assert_eq!(m.gamma, DisplayGamma::Bt1886);
        assert!(approx(m.white_point.x, 0.3127, 0.001));
        assert!(approx(m.white_point.y, 0.3290, 0.001));
    }

    // 13. p3_d65_reference is HDR-capable with correct formats
    #[test]
    fn test_full_p3_d65_reference_properties() {
        let m = FullDisplayModel::p3_d65_reference();
        assert!(m.hdr_capable);
        assert!(m.hdr_formats.contains(&HdrFormatDisplay::DolbyVision));
        assert!(m.hdr_formats.contains(&HdrFormatDisplay::Hdr10));
        assert_eq!(m.gamma, DisplayGamma::Pq);
        assert!(approx(m.primaries.red_xy.0, 0.680, 0.001));
    }

    // 14. rec2020_1000nit has correct peak and formats
    #[test]
    fn test_full_rec2020_1000nit_properties() {
        let m = FullDisplayModel::rec2020_1000nit();
        assert!(approx(m.max_luminance_nits, 1000.0, 0.1));
        assert!(m.hdr_capable);
        assert!(m.hdr_formats.contains(&HdrFormatDisplay::Hdr10));
        assert!(m.hdr_formats.contains(&HdrFormatDisplay::HdrPlus));
        assert!(!m.hdr_formats.contains(&HdrFormatDisplay::DolbyVision));
    }

    // 15. rec2020_4000nit includes DolbyVision support
    #[test]
    fn test_full_rec2020_4000nit_properties() {
        let m = FullDisplayModel::rec2020_4000nit();
        assert!(approx(m.max_luminance_nits, 4000.0, 0.1));
        assert!(m.hdr_formats.contains(&HdrFormatDisplay::DolbyVision));
    }

    // 16. rec2020_10000nit has all 5 formats
    #[test]
    fn test_full_rec2020_10000nit_has_all_formats() {
        let m = FullDisplayModel::rec2020_10000nit();
        assert!(approx(m.max_luminance_nits, 10000.0, 1.0));
        assert_eq!(m.hdr_formats.len(), 5);
        assert!(m.hdr_formats.contains(&HdrFormatDisplay::HdrVivid));
        assert!(m.hdr_formats.contains(&HdrFormatDisplay::Hlg));
    }

    // 17. compute_tone_map_params: SDR target <= 100 nit, source <= 1000 → BT.2446A
    #[test]
    fn test_tone_map_params_bt2446a() {
        let display = FullDisplayModel::rec709_reference();
        let params = display.compute_tone_map_params(1000.0);
        assert_eq!(params.recommended_algorithm, "BT.2446A");
        assert!(approx(params.input_peak_nits, 1000.0, 0.1));
        assert!(approx(params.output_peak_nits, 100.0, 0.1));
    }

    // 18. compute_tone_map_params: SDR target, source > 1000 → BT.2446C
    #[test]
    fn test_tone_map_params_bt2446c() {
        let display = FullDisplayModel::rec709_reference();
        let params = display.compute_tone_map_params(4000.0);
        assert_eq!(params.recommended_algorithm, "BT.2446C");
    }

    // 19. compute_tone_map_params: HDR target >= 1000 → ACES
    #[test]
    fn test_tone_map_params_aces() {
        let display = FullDisplayModel::rec2020_1000nit();
        let params = display.compute_tone_map_params(4000.0);
        assert_eq!(params.recommended_algorithm, "ACES");
        assert!(params.contrast_ratio > 500_000.0);
    }

    // 20. compute_tone_map_params: mid-range HDR target → BT.2390
    #[test]
    fn test_tone_map_params_bt2390() {
        let mut display = FullDisplayModel::rec2020_1000nit();
        display.max_luminance_nits = 500.0; // mid-range HDR target
        let params = display.compute_tone_map_params(4000.0);
        assert_eq!(params.recommended_algorithm, "BT.2390");
    }

    // 21. contrast_ratio is correct
    #[test]
    fn test_full_contrast_ratio() {
        let m = FullDisplayModel::rec2020_4000nit();
        let ratio = m.contrast_ratio();
        // 4000 / 0.0005 = 8_000_000
        assert!(
            ratio > 5_000_000.0,
            "expected contrast ratio > 5M, got {ratio}"
        );
    }

    // 22. WhitePoint::d65 returns correct values
    #[test]
    fn test_white_point_d65() {
        let wp = WhitePoint::d65();
        assert!(approx(wp.x, 0.3127, 0.0001));
        assert!(approx(wp.y, 0.3290, 0.0001));
    }
}
