//! Pre-built photographic and cinematic LUT presets.
//!
//! Each [`PhotoLutPreset`](crate::photographic_luts::PhotoLutPreset) variant encodes a signature colour transformation
//! and can be materialised as a 33³ [`Lut3DData`](crate::hald_clut::Lut3DData) or applied
//! directly to a single pixel.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::hald_clut::Lut3DData;

// ---------------------------------------------------------------------------
// HSL helpers
// ---------------------------------------------------------------------------

/// Convert linear RGB `[0, 1]` → HSL `(h°, s, l)`.
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;
    let delta = max - min;

    if delta < 1e-7 {
        return (0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if (max - r).abs() < 1e-7 {
        (g - b) / delta + if g < b { 6.0 } else { 0.0 }
    } else if (max - g).abs() < 1e-7 {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };

    (h * 60.0, s, l)
}

/// Convert HSL `(h°, s, l)` → linear RGB `[0, 1]`.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s < 1e-7 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue = h / 360.0;

    let channel = |t: f32| -> f32 {
        let t = if t < 0.0 {
            t + 1.0
        } else if t > 1.0 {
            t - 1.0
        } else {
            t
        };
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };

    (
        channel(hue + 1.0 / 3.0),
        channel(hue),
        channel(hue - 1.0 / 3.0),
    )
}

// ---------------------------------------------------------------------------
// S-curve / contrast helpers
// ---------------------------------------------------------------------------

/// Smooth S-curve contrast boost centred at 0.5.
///
/// `strength` 0.0 = identity, 1.0 = strong S-curve.
#[inline]
fn s_curve(v: f32, strength: f32) -> f32 {
    // Use a simple polynomial S-curve
    let x = v.clamp(0.0, 1.0);
    let shaped = x * x * (3.0 - 2.0 * x); // smoothstep
    (1.0 - strength) * x + strength * shaped
}

/// Power-law contrast centred at mid-grey.
#[inline]
fn contrast_adjust(v: f32, factor: f32) -> f32 {
    let x = v.clamp(0.0, 1.0) - 0.5;
    (0.5 + x * factor).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// PhotoLutPreset
// ---------------------------------------------------------------------------

/// Pre-built photographic / cinematic colour grade presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotoLutPreset {
    /// High-contrast, desaturated blacks — classic film noir look.
    FilmNoir,
    /// Warm reds/yellows, punchy Kodachrome-inspired colours.
    Kodachrome,
    /// Cool blues, natural greens — Fuji chrome aesthetic.
    FujiChrome,
    /// Complementary teal–orange grade popular in modern cinema.
    CinematicTeal,
    /// Bleach bypass: high contrast, heavily desaturated.
    Bleach,
    /// Faded, warm shadows with lifted blacks.
    Vintage,
    /// Cool blue tones, darkened midtones — moonlight simulation.
    Moonlight,
    /// Warm orange, lifted shadows — sunrise / golden hour.
    Sunrise,
    /// Clean, slightly warm, lifted blacks — commercial / advertising look.
    Commercial,
    /// High-contrast black & white conversion.
    BwHigh,
    /// Low-contrast, soft black & white conversion.
    BwLow,
    /// Generic log-curve → Rec.709 linearisation.
    LogToRec709,
}

impl PhotoLutPreset {
    /// Human-readable name for this preset.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::FilmNoir => "Film Noir",
            Self::Kodachrome => "Kodachrome",
            Self::FujiChrome => "FujiChrome",
            Self::CinematicTeal => "Cinematic Teal/Orange",
            Self::Bleach => "Bleach Bypass",
            Self::Vintage => "Vintage",
            Self::Moonlight => "Moonlight",
            Self::Sunrise => "Sunrise",
            Self::Commercial => "Commercial",
            Self::BwHigh => "B&W High Contrast",
            Self::BwLow => "B&W Low Contrast",
            Self::LogToRec709 => "Log to Rec.709",
        }
    }

    /// Apply the preset directly to a single `(r, g, b)` pixel.
    ///
    /// All values are expected in `[0, 1]`; output is also clamped to `[0, 1]`.
    #[must_use]
    pub fn apply_to_pixel(r: f32, g: f32, b: f32, preset: &Self) -> (f32, f32, f32) {
        let (r, g, b) = preset.transform(r, g, b);
        (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
    }

    /// Generate a 33³ [`Lut3DData`] for this preset.
    #[must_use]
    pub fn to_lut3d(&self) -> Lut3DData {
        let size = 33_usize;
        let mut data = Vec::with_capacity(size * size * size);
        let scale = (size - 1) as f32;
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let rf = r as f32 / scale;
                    let gf = g as f32 / scale;
                    let bf = b as f32 / scale;
                    let (ro, go, bo) = self.transform(rf, gf, bf);
                    data.push([ro.clamp(0.0, 1.0), go.clamp(0.0, 1.0), bo.clamp(0.0, 1.0)]);
                }
            }
        }
        Lut3DData { size, data }
    }

    // -----------------------------------------------------------------------
    // Per-preset transform functions
    // -----------------------------------------------------------------------

    fn transform(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        match self {
            Self::FilmNoir => Self::film_noir(r, g, b),
            Self::Kodachrome => Self::kodachrome(r, g, b),
            Self::FujiChrome => Self::fuji_chrome(r, g, b),
            Self::CinematicTeal => Self::cinematic_teal(r, g, b),
            Self::Bleach => Self::bleach(r, g, b),
            Self::Vintage => Self::vintage(r, g, b),
            Self::Moonlight => Self::moonlight(r, g, b),
            Self::Sunrise => Self::sunrise(r, g, b),
            Self::Commercial => Self::commercial(r, g, b),
            Self::BwHigh => Self::bw_high(r, g, b),
            Self::BwLow => Self::bw_low(r, g, b),
            Self::LogToRec709 => Self::log_to_rec709(r, g, b),
        }
    }

    // -- Film Noir -----------------------------------------------------------
    fn film_noir(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Desaturate and apply high-contrast S-curve, crush shadows
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let desat = 0.35_f32; // keep some colour
        let r2 = r * (1.0 - desat) + luma * desat;
        let g2 = g * (1.0 - desat) + luma * desat;
        let b2 = b * (1.0 - desat) + luma * desat;
        // Strong S-curve contrast
        let r3 = s_curve(r2, 0.8);
        let g3 = s_curve(g2, 0.8);
        let b3 = s_curve(b2, 0.8);
        // Shadow crush
        let crush = |v: f32| (v - 0.08).max(0.0) / (1.0 - 0.08);
        (crush(r3), crush(g3), crush(b3))
    }

    // -- Kodachrome ----------------------------------------------------------
    fn kodachrome(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Warm reds/yellows: lift red, slight green boost, reduce blue
        let r2 = (r * 1.10 + 0.02).min(1.0);
        let g2 = (g * 1.04).min(1.0);
        let b2 = (b * 0.90).max(0.0);
        // Boost saturation via HSL
        let (h, s, l) = rgb_to_hsl(r2, g2, b2);
        let s2 = (s * 1.25).min(1.0);
        // Slight hue rotation toward red (+3° for warm-red bias)
        let h2 = (h + 3.0).rem_euclid(360.0);
        hsl_to_rgb(h2, s2, l)
    }

    // -- FujiChrome ----------------------------------------------------------
    fn fuji_chrome(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Cool blues, natural greens — reduce red slightly, lift blue
        let r2 = (r * 0.93).max(0.0);
        let g2 = g;
        let b2 = (b * 1.08 + 0.02).min(1.0);
        let (h, s, l) = rgb_to_hsl(r2, g2, b2);
        // Slight hue shift toward blue-cyan (-4°)
        let h2 = (h - 4.0).rem_euclid(360.0);
        let s2 = (s * 1.1).min(1.0);
        hsl_to_rgb(h2, s2, l)
    }

    // -- Cinematic Teal/Orange -----------------------------------------------
    fn cinematic_teal(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        // Shadows → teal (lift blue, suppress red in shadows)
        // Highlights → orange (lift red/green in highlights)
        let shadow_weight = (1.0 - luma).powf(1.5);
        let highlight_weight = luma.powf(1.5);
        let r2 = r - 0.08 * shadow_weight + 0.12 * highlight_weight;
        let g2 = g + 0.02 * shadow_weight + 0.06 * highlight_weight;
        let b2 = b + 0.14 * shadow_weight - 0.10 * highlight_weight;
        (r2.clamp(0.0, 1.0), g2.clamp(0.0, 1.0), b2.clamp(0.0, 1.0))
    }

    // -- Bleach Bypass -------------------------------------------------------
    fn bleach(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // 60% desaturation + 30% contrast boost
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let desat = 0.60_f32;
        let r2 = r * (1.0 - desat) + luma * desat;
        let g2 = g * (1.0 - desat) + luma * desat;
        let b2 = b * (1.0 - desat) + luma * desat;
        let contrast = 1.30_f32;
        (
            contrast_adjust(r2, contrast),
            contrast_adjust(g2, contrast),
            contrast_adjust(b2, contrast),
        )
    }

    // -- Vintage -------------------------------------------------------------
    fn vintage(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Lift blacks to 0.05, reduce contrast, warm (+0.05R, -0.05B)
        let lift_black = |v: f32| 0.05 + v * (1.0 - 0.05);
        let r2 = lift_black(r) + 0.05;
        let g2 = lift_black(g);
        let b2 = lift_black(b) - 0.05;
        // Reduce contrast (compress toward mid-grey)
        let compress = 0.85_f32;
        (
            contrast_adjust(r2, compress),
            contrast_adjust(g2, compress),
            contrast_adjust(b2, compress),
        )
    }

    // -- Moonlight -----------------------------------------------------------
    fn moonlight(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Cool blue, dark — reduce all channels, lift blue
        let darken = 0.75_f32;
        let r2 = r * darken * 0.85;
        let g2 = g * darken * 0.90;
        let b2 = (b * darken + 0.07).min(1.0);
        (r2, g2, b2)
    }

    // -- Sunrise -------------------------------------------------------------
    fn sunrise(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Warm orange, lifted shadows
        let lift = |v: f32| 0.03 + v * 0.97;
        let r2 = (lift(r) * 1.15).min(1.0);
        let g2 = (lift(g) * 1.02).min(1.0);
        let b2 = lift(b) * 0.78;
        (r2, g2, b2)
    }

    // -- Commercial ----------------------------------------------------------
    fn commercial(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Clean, slightly warm, lifted blacks
        let lift_black = |v: f32| 0.04 + v * 0.96;
        let r2 = (lift_black(r) * 1.04).min(1.0);
        let g2 = lift_black(g);
        let b2 = lift_black(b) * 0.97;
        (r2, g2, b2)
    }

    // -- B&W High Contrast ---------------------------------------------------
    fn bw_high(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let v = s_curve(luma, 0.9);
        (v, v, v)
    }

    // -- B&W Low Contrast ----------------------------------------------------
    fn bw_low(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        // Compress toward mid-grey
        let v = 0.1 + luma * 0.8;
        (v, v, v)
    }

    // -- Log to Rec.709 ------------------------------------------------------
    fn log_to_rec709(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Approximate: log-C-like linearisation then Rec.709 gamma
        let log_expand = |v: f32| -> f32 {
            let v = v.clamp(0.0, 1.0);
            // Simple log-linear approximation: v_lin = 10^((v - 0.385) / 0.247)
            // shifted / scaled to keep blacks near zero
            if v < 0.09 {
                v / 0.09 * 0.018
            } else {
                10.0_f32.powf((v - 0.385) / 0.247) * 0.18
            }
        };
        let gamma709 = |v: f32| -> f32 {
            let v = v.clamp(0.0, 1.0);
            if v <= 0.018 {
                4.5 * v
            } else {
                1.099 * v.powf(0.45) - 0.099
            }
        };
        let r2 = gamma709(log_expand(r));
        let g2 = gamma709(log_expand(g));
        let b2 = gamma709(log_expand(b));
        (r2, g2, b2)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rgb_in_range(r: f32, g: f32, b: f32) {
        assert!(r >= 0.0 && r <= 1.0, "r out of range: {r}");
        assert!(g >= 0.0 && g <= 1.0, "g out of range: {g}");
        assert!(b >= 0.0 && b <= 1.0, "b out of range: {b}");
    }

    #[test]
    fn test_all_presets_have_names() {
        let presets = [
            PhotoLutPreset::FilmNoir,
            PhotoLutPreset::Kodachrome,
            PhotoLutPreset::FujiChrome,
            PhotoLutPreset::CinematicTeal,
            PhotoLutPreset::Bleach,
            PhotoLutPreset::Vintage,
            PhotoLutPreset::Moonlight,
            PhotoLutPreset::Sunrise,
            PhotoLutPreset::Commercial,
            PhotoLutPreset::BwHigh,
            PhotoLutPreset::BwLow,
            PhotoLutPreset::LogToRec709,
        ];
        for p in &presets {
            assert!(!p.name().is_empty(), "empty name for {p:?}");
        }
    }

    #[test]
    fn test_film_noir_desaturates() {
        // Use a mid-range colour so the S-curve + shadow crush don't clip channels to zero
        let (r, g, b) = PhotoLutPreset::apply_to_pixel(0.8, 0.2, 0.5, &PhotoLutPreset::FilmNoir);
        // After 35% desaturation the channel spread should be smaller than the input spread (0.6)
        let spread_in = (0.8_f32 - 0.2_f32).abs();
        let spread_out = (r - g).abs().max((g - b).abs()).max((r - b).abs());
        assert!(
            spread_out <= spread_in + 0.05,
            "spread_out={spread_out} spread_in={spread_in}"
        );
        assert_rgb_in_range(r, g, b);
    }

    #[test]
    fn test_bw_high_makes_greyscale() {
        let (r, g, b) = PhotoLutPreset::apply_to_pixel(0.8, 0.3, 0.5, &PhotoLutPreset::BwHigh);
        assert!((r - g).abs() < 1e-5, "r={r} g={g}");
        assert!((g - b).abs() < 1e-5, "g={g} b={b}");
    }

    #[test]
    fn test_bw_low_makes_greyscale() {
        let (r, g, b) = PhotoLutPreset::apply_to_pixel(0.2, 0.7, 0.4, &PhotoLutPreset::BwLow);
        assert!((r - g).abs() < 1e-5);
        assert!((g - b).abs() < 1e-5);
    }

    #[test]
    fn test_vintage_lifts_blacks() {
        let (r, g, b) = PhotoLutPreset::apply_to_pixel(0.0, 0.0, 0.0, &PhotoLutPreset::Vintage);
        // Black input must produce lifted output (> 0.0)
        assert!(
            r > 0.0 || g > 0.0 || b > 0.0,
            "blacks not lifted: {r} {g} {b}"
        );
        assert_rgb_in_range(r, g, b);
    }

    #[test]
    fn test_bleach_reduces_saturation() {
        // Compare a mid-range saturated colour before and after bleach
        let (ri, gi, bi) = (0.9_f32, 0.1, 0.1);
        let (_, s_in, _) = rgb_to_hsl(ri, gi, bi);
        let (r, g, b) = PhotoLutPreset::apply_to_pixel(ri, gi, bi, &PhotoLutPreset::Bleach);
        let (_, s_out, _) = rgb_to_hsl(r, g, b);
        // 60% desaturation should reduce saturation noticeably
        assert!(
            s_out < s_in,
            "saturation should decrease: s_in={s_in} s_out={s_out}"
        );
        assert_rgb_in_range(r, g, b);
    }

    #[test]
    fn test_all_presets_output_in_range() {
        let presets = [
            PhotoLutPreset::FilmNoir,
            PhotoLutPreset::Kodachrome,
            PhotoLutPreset::FujiChrome,
            PhotoLutPreset::CinematicTeal,
            PhotoLutPreset::Bleach,
            PhotoLutPreset::Vintage,
            PhotoLutPreset::Moonlight,
            PhotoLutPreset::Sunrise,
            PhotoLutPreset::Commercial,
            PhotoLutPreset::BwHigh,
            PhotoLutPreset::BwLow,
            PhotoLutPreset::LogToRec709,
        ];
        let test_pixels = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.3, 0.8)];
        for &p in &presets {
            for &(r, g, b) in &test_pixels {
                let (ro, go, bo) = PhotoLutPreset::apply_to_pixel(r, g, b, &p);
                assert_rgb_in_range(ro, go, bo);
            }
        }
    }

    #[test]
    fn test_to_lut3d_correct_size() {
        let lut = PhotoLutPreset::FilmNoir.to_lut3d();
        assert_eq!(lut.size, 33);
        assert_eq!(lut.data.len(), 33 * 33 * 33);
    }

    #[test]
    fn test_to_lut3d_data_in_range() {
        let lut = PhotoLutPreset::Kodachrome.to_lut3d();
        for &[r, g, b] in &lut.data {
            assert!(r >= 0.0 && r <= 1.0, "r={r}");
            assert!(g >= 0.0 && g <= 1.0, "g={g}");
            assert!(b >= 0.0 && b <= 1.0, "b={b}");
        }
    }

    #[test]
    fn test_cinematic_teal_shifts_shadows() {
        // Pure dark tone should gain blue-green component
        let (r, g, b) =
            PhotoLutPreset::apply_to_pixel(0.1, 0.1, 0.1, &PhotoLutPreset::CinematicTeal);
        assert!(b > r, "shadows should be teal: r={r} b={b}");
        assert_rgb_in_range(r, g, b);
    }

    #[test]
    fn test_moonlight_darkens_image() {
        let (r, g, b) = PhotoLutPreset::apply_to_pixel(0.8, 0.8, 0.8, &PhotoLutPreset::Moonlight);
        let luma_in = 0.2126 * 0.8 + 0.7152 * 0.8 + 0.0722 * 0.8;
        let luma_out = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        assert!(
            luma_out < luma_in,
            "moonlight should darken: {luma_out} vs {luma_in}"
        );
    }

    #[test]
    fn test_log_to_rec709_maps_mid_grey() {
        // Log mid-grey (≈0.385) should map to a visible mid-grey
        let (r, g, b) =
            PhotoLutPreset::apply_to_pixel(0.385, 0.385, 0.385, &PhotoLutPreset::LogToRec709);
        // r, g, b should all be equal (neutral)
        assert!((r - g).abs() < 1e-4, "not neutral: r={r} g={g}");
        assert!((g - b).abs() < 1e-4, "not neutral: g={g} b={b}");
        assert_rgb_in_range(r, g, b);
    }

    #[test]
    fn test_hsl_roundtrip() {
        let (r0, g0, b0) = (0.6, 0.2, 0.9);
        let (h, s, l) = rgb_to_hsl(r0, g0, b0);
        let (r1, g1, b1) = hsl_to_rgb(h, s, l);
        assert!((r0 - r1).abs() < 1e-4, "r roundtrip fail: {r0} vs {r1}");
        assert!((g0 - g1).abs() < 1e-4, "g roundtrip fail: {g0} vs {g1}");
        assert!((b0 - b1).abs() < 1e-4, "b roundtrip fail: {b0} vs {b1}");
    }
}
