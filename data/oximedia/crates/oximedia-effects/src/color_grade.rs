//! Color grading effects using CDL (Color Decision List) and film-look models.
//!
//! Operates on linear-light `f32` RGB values in the range [0.0, 1.0].

#![allow(dead_code)]

/// ASC CDL (Color Decision List) parameters.
///
/// Each channel uses the formula: `out = clamp((in * slope + offset) ^ power)`
#[derive(Debug, Clone)]
pub struct CdlParameters {
    /// Per-channel slope multiplier [R, G, B]. 1.0 = neutral.
    pub slope: [f32; 3],
    /// Per-channel additive offset [R, G, B]. 0.0 = neutral.
    pub offset: [f32; 3],
    /// Per-channel power (gamma) curve [R, G, B]. 1.0 = neutral.
    pub power: [f32; 3],
    /// Saturation multiplier applied after slope/offset/power. 1.0 = neutral.
    pub saturation: f32,
}

impl CdlParameters {
    /// Identity CDL: no colour change.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            slope: [1.0; 3],
            offset: [0.0; 3],
            power: [1.0; 3],
            saturation: 1.0,
        }
    }

    /// Apply CDL to a single RGB pixel `[R, G, B]` (linear [0, 1]).
    ///
    /// Formula per channel: `out = clamp((in * slope + offset) ^ power, 0, 1)`
    /// Then saturation is applied using Rec. 709 luma weights.
    #[must_use]
    pub fn apply_to_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let v = rgb[i] * self.slope[i] + self.offset[i];
            let v = v.max(0.0).powf(self.power[i]);
            out[i] = v.clamp(0.0, 1.0);
        }
        // Saturation via Rec.709 luma
        let luma = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
        for c in &mut out {
            *c = (luma + (*c - luma) * self.saturation).clamp(0.0, 1.0);
        }
        out
    }
}

/// Simple parametric film-look tone curve.
///
/// Applies contrast, lift, gain, and gamma to a single luminance value.
#[derive(Debug, Clone)]
pub struct FilmLook {
    /// Contrast scale around 0.5 midpoint. 1.0 = neutral.
    pub contrast: f32,
    /// Shadow lift offset. 0.0 = neutral.
    pub lift: f32,
    /// Highlight gain multiplier. 1.0 = neutral.
    pub gain: f32,
    /// Gamma power curve. 1.0 = neutral; >1 brightens midtones.
    pub gamma: f32,
}

impl FilmLook {
    /// Apply the film-look curve to a single scalar value `x` in [0, 1].
    #[must_use]
    pub fn apply(&self, x: f32) -> f32 {
        // 1. Contrast around 0.5
        let v = ((x - 0.5) * self.contrast + 0.5).clamp(0.0, 1.0);
        // 2. Lift + gain
        let v = (v * (1.0 - self.lift) + self.lift) * self.gain;
        // 3. Gamma (power curve), guard against non-positive
        let gamma_safe = self.gamma.max(1e-6);
        v.max(0.0).powf(1.0 / gamma_safe).clamp(0.0, 1.0)
    }
}

impl Default for FilmLook {
    fn default() -> Self {
        Self {
            contrast: 1.0,
            lift: 0.0,
            gain: 1.0,
            gamma: 1.0,
        }
    }
}

/// Radial vignette effect.
#[derive(Debug, Clone)]
pub struct VignetteEffect {
    /// Vignette strength [0.0, 1.0]; 0.0 = no effect.
    pub strength: f32,
    /// Radius of the clear (unaffected) centre as a fraction of image diagonal.
    pub radius: f32,
    /// Feather/softness of the vignette edge [0.0, 1.0].
    pub feather: f32,
}

impl VignetteEffect {
    /// Compute the vignette multiplier [0.0, 1.0] for a pixel at `distance_from_center`
    /// (normalised to the image diagonal, so center=0.0, corner=1.0).
    ///
    /// A factor of 1.0 means fully unaffected; 0.0 means fully darkened.
    #[must_use]
    pub fn factor_at(&self, distance_from_center: f32) -> f32 {
        if self.strength <= 0.0 {
            return 1.0;
        }
        // Normalise distance relative to radius
        let edge_start = self.radius;
        let edge_end = (self.radius + self.feather.max(1e-6)).max(edge_start + 1e-6);
        let t = ((distance_from_center - edge_start) / (edge_end - edge_start)).clamp(0.0, 1.0);
        // Smooth step
        let smooth = t * t * (3.0 - 2.0 * t);
        1.0 - self.strength * smooth
    }
}

/// Film grain simulation using an LCG hash noise function.
#[derive(Debug, Clone)]
pub struct GrainEffect {
    /// Grain intensity [0.0, 1.0].
    pub intensity: f32,
    /// Grain structure size (unused in hash noise, kept for API completeness).
    pub size: f32,
}

impl GrainEffect {
    /// Sample the grain value at pixel coordinates `(x, y)` with the given
    /// `seed`.  Returns a value in [0.0, 1.0].
    ///
    /// Uses a combined LCG / xor-shift hash so adjacent pixels are uncorrelated.
    #[must_use]
    pub fn sample(&self, x: u32, y: u32, seed: u32) -> f32 {
        // Combine position and seed into a single u32 hash
        let mut h = x
            .wrapping_mul(1_664_525)
            .wrapping_add(y.wrapping_mul(1_013_904_223))
            .wrapping_add(seed.wrapping_mul(22_695_477));
        // Xor-shift mix
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9_f3b7);
        h ^= h >> 16;
        #[allow(clippy::cast_precision_loss)]
        let normalized = (h as f32) / (u32::MAX as f32);
        // Scale by intensity and shift to be centred around 0.5
        (0.5 + (normalized - 0.5) * self.intensity).clamp(0.0, 1.0)
    }
}

// ── ColorGradeParams API ──────────────────────────────────────────────────────

/// Parameters for the full lift/gamma/gain color grading pipeline (f64).
#[derive(Debug, Clone)]
pub struct ColorGradeParams {
    /// Shadow offset (lift) per channel [R, G, B]. 0.0 = neutral.
    pub lift: [f64; 3],
    /// Midtone power curve (gamma) per channel [R, G, B]. 1.0 = neutral.
    pub gamma: [f64; 3],
    /// Highlight multiplier (gain) per channel [R, G, B]. 1.0 = neutral.
    pub gain: [f64; 3],
    /// Saturation multiplier. 1.0 = neutral, 0.0 = desaturate.
    pub saturation: f64,
    /// Contrast around midpoint 0.5. 1.0 = neutral.
    pub contrast: f64,
    /// Additive brightness offset in linear \[0,1\] space. 0.0 = neutral.
    pub brightness: f64,
}

impl Default for ColorGradeParams {
    fn default() -> Self {
        Self::default_neutral()
    }
}

impl ColorGradeParams {
    /// Identity / neutral grade: no colour change.
    #[must_use]
    pub fn default_neutral() -> Self {
        Self {
            lift: [0.0, 0.0, 0.0],
            gamma: [1.0, 1.0, 1.0],
            gain: [1.0, 1.0, 1.0],
            saturation: 1.0,
            contrast: 1.0,
            brightness: 0.0,
        }
    }
}

/// Apply lift/gamma/gain to an RGB pixel (f64 precision).
///
/// Processing order per channel: lift → gain → gamma.
#[must_use]
pub fn apply_lift_gamma_gain(rgb: (f64, f64, f64), params: &ColorGradeParams) -> (f64, f64, f64) {
    let apply_channel = |v: f64, lift: f64, gamma: f64, gain: f64| -> f64 {
        let lifted = v * (1.0 - lift) + lift;
        let gained = lifted * gain;
        let gamma_safe = gamma.max(1e-6);
        gained.max(0.0).powf(1.0 / gamma_safe)
    };
    let (r, g, b) = rgb;
    (
        apply_channel(r, params.lift[0], params.gamma[0], params.gain[0]),
        apply_channel(g, params.lift[1], params.gamma[1], params.gain[1]),
        apply_channel(b, params.lift[2], params.gamma[2], params.gain[2]),
    )
}

/// Adjust saturation of an RGB pixel using Rec.709 luminance weights.
#[must_use]
pub fn adjust_saturation(rgb: (f64, f64, f64), saturation: f64) -> (f64, f64, f64) {
    let (r, g, b) = rgb;
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    (
        luma + (r - luma) * saturation,
        luma + (g - luma) * saturation,
        luma + (b - luma) * saturation,
    )
}

/// Apply an S-curve contrast adjustment around midpoint 0.5.
#[must_use]
#[inline]
pub fn adjust_contrast(value: f64, contrast: f64) -> f64 {
    ((value - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
}

/// Apply all color grading operations to a single pixel.
#[must_use]
pub fn apply_color_grade(rgb: (f64, f64, f64), params: &ColorGradeParams) -> (f64, f64, f64) {
    let (r, g, b) = apply_lift_gamma_gain(rgb, params);
    let r = r + params.brightness;
    let g = g + params.brightness;
    let b = b + params.brightness;
    let r = adjust_contrast(r, params.contrast);
    let g = adjust_contrast(g, params.contrast);
    let b = adjust_contrast(b, params.contrast);
    let (r, g, b) = adjust_saturation((r, g, b), params.saturation);
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Apply color grading to an entire frame of linear-light f64 pixels in-place.
pub fn apply_color_grade_frame(pixels: &mut [(f64, f64, f64)], params: &ColorGradeParams) {
    for pixel in pixels.iter_mut() {
        *pixel = apply_color_grade(*pixel, params);
    }
}

/// Parameters for split-tone effect (shadow/highlight tinting).
#[derive(Debug, Clone)]
pub struct SplitToneParams {
    /// Shadow tone hue in degrees [0, 360).
    pub shadow_hue: f64,
    /// Shadow tint saturation [0.0, 1.0].
    pub shadow_saturation: f64,
    /// Highlight tone hue in degrees [0, 360).
    pub highlight_hue: f64,
    /// Highlight tint saturation [0.0, 1.0].
    pub highlight_saturation: f64,
    /// Balance between shadows and highlights [-1.0, 1.0]. 0.0 = balanced.
    pub balance: f64,
}

impl Default for SplitToneParams {
    fn default() -> Self {
        Self {
            shadow_hue: 210.0,
            shadow_saturation: 0.0,
            highlight_hue: 40.0,
            highlight_saturation: 0.0,
            balance: 0.0,
        }
    }
}

/// Apply split-tone effect to an RGB pixel.
#[must_use]
pub fn apply_split_tone(rgb: (f64, f64, f64), params: &SplitToneParams) -> (f64, f64, f64) {
    use std::f64::consts::PI;
    let (r, g, b) = rgb;
    let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
    let shadow_weight = ((1.0 - luma) - params.balance * 0.5).clamp(0.0, 1.0);
    let highlight_weight = (luma + params.balance * 0.5).clamp(0.0, 1.0);
    let shadow_tint = hue_to_rgb(params.shadow_hue * PI / 180.0);
    let highlight_tint = hue_to_rgb(params.highlight_hue * PI / 180.0);
    let s_sat = params.shadow_saturation * shadow_weight;
    let h_sat = params.highlight_saturation * highlight_weight;
    let ro = r + (shadow_tint.0 - luma) * s_sat + (highlight_tint.0 - luma) * h_sat;
    let go = g + (shadow_tint.1 - luma) * s_sat + (highlight_tint.1 - luma) * h_sat;
    let bo = b + (shadow_tint.2 - luma) * s_sat + (highlight_tint.2 - luma) * h_sat;
    (ro.clamp(0.0, 1.0), go.clamp(0.0, 1.0), bo.clamp(0.0, 1.0))
}

/// Convert hue angle (radians) to an RGB triplet.
#[inline]
fn hue_to_rgb(hue_rad: f64) -> (f64, f64, f64) {
    use std::f64::consts::PI;
    let r = ((hue_rad).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
    let g = ((hue_rad + 2.0 * PI / 3.0).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
    let b = ((hue_rad - 2.0 * PI / 3.0).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
    (r, g, b)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CdlParameters ─────────────────────────────────────────────────────────

    #[test]
    fn test_cdl_identity_is_passthrough() {
        let cdl = CdlParameters::identity();
        let rgb = [0.4, 0.6, 0.2];
        let out = cdl.apply_to_rgb(rgb);
        for i in 0..3 {
            assert!(
                (out[i] - rgb[i]).abs() < 1e-5,
                "channel {i}: {} != {}",
                out[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn test_cdl_slope_multiplies() {
        let mut cdl = CdlParameters::identity();
        cdl.slope = [2.0, 1.0, 1.0];
        let out = cdl.apply_to_rgb([0.3, 0.5, 0.5]);
        // R: 0.3 * 2 = 0.6; clamped to 0.6
        assert!((out[0] - 0.6).abs() < 1e-5, "R: {}", out[0]);
    }

    #[test]
    fn test_cdl_offset_adds() {
        let mut cdl = CdlParameters::identity();
        cdl.offset = [0.1, 0.0, 0.0];
        let out = cdl.apply_to_rgb([0.4, 0.5, 0.5]);
        // R: 0.4 + 0.1 = 0.5
        assert!((out[0] - 0.5).abs() < 1e-5, "R: {}", out[0]);
    }

    #[test]
    fn test_cdl_power_darkens() {
        let mut cdl = CdlParameters::identity();
        cdl.power = [2.0, 1.0, 1.0]; // power=2 darkens midtones
        let out = cdl.apply_to_rgb([0.5, 0.5, 0.5]);
        // 0.5^2 = 0.25
        assert!((out[0] - 0.25).abs() < 1e-5, "R: {}", out[0]);
    }

    #[test]
    fn test_cdl_output_clamped() {
        let mut cdl = CdlParameters::identity();
        cdl.slope = [10.0, 10.0, 10.0];
        let out = cdl.apply_to_rgb([1.0, 1.0, 1.0]);
        for &c in &out {
            assert!(c <= 1.0, "channel must be <= 1.0: {c}");
        }
    }

    #[test]
    fn test_cdl_saturation_zero_is_grey() {
        let mut cdl = CdlParameters::identity();
        cdl.saturation = 0.0;
        let out = cdl.apply_to_rgb([0.8, 0.2, 0.5]);
        // All channels should equal the Rec.709 luma
        let luma = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.5;
        for &c in &out {
            assert!(
                (c - luma).abs() < 1e-5,
                "channel should be luma {luma}: {c}"
            );
        }
    }

    // ── FilmLook ──────────────────────────────────────────────────────────────

    #[test]
    fn test_film_look_identity() {
        let fl = FilmLook::default();
        let v = 0.6;
        assert!((fl.apply(v) - v).abs() < 1e-5, "identity: {}", fl.apply(v));
    }

    #[test]
    fn test_film_look_contrast_spreads() {
        let fl = FilmLook {
            contrast: 2.0,
            ..Default::default()
        };
        assert!(fl.apply(0.2) < 0.2, "contrast>1 should darken shadows");
        assert!(fl.apply(0.8) > 0.8, "contrast>1 should brighten highlights");
    }

    #[test]
    fn test_film_look_output_in_range() {
        let fl = FilmLook {
            contrast: 3.0,
            lift: 0.1,
            gain: 1.5,
            gamma: 1.2,
        };
        for x in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let out = fl.apply(x);
            assert!((0.0..=1.0).contains(&out), "out of range: {out} for {x}");
        }
    }

    // ── VignetteEffect ────────────────────────────────────────────────────────

    #[test]
    fn test_vignette_center_unaffected() {
        let v = VignetteEffect {
            strength: 1.0,
            radius: 0.5,
            feather: 0.2,
        };
        // Exactly at center (0.0) — well inside the radius
        assert!((v.factor_at(0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vignette_zero_strength() {
        let v = VignetteEffect {
            strength: 0.0,
            radius: 0.5,
            feather: 0.2,
        };
        // Zero strength → always 1.0
        for d in [0.0f32, 0.5, 1.0] {
            assert!((v.factor_at(d) - 1.0).abs() < 1e-5, "d={d}");
        }
    }

    #[test]
    fn test_vignette_corner_darkened() {
        let v = VignetteEffect {
            strength: 1.0,
            radius: 0.3,
            feather: 0.1,
        };
        // A distance of 1.0 is well past the edge
        let factor = v.factor_at(1.0);
        assert!(factor < 1.0, "corner should be darkened: {factor}");
    }

    // ── GrainEffect ───────────────────────────────────────────────────────────

    #[test]
    fn test_grain_in_range() {
        let g = GrainEffect {
            intensity: 1.0,
            size: 1.0,
        };
        for x in 0..10u32 {
            for y in 0..10u32 {
                let s = g.sample(x, y, 42);
                assert!((0.0..=1.0).contains(&s), "sample out of range: {s}");
            }
        }
    }

    #[test]
    fn test_grain_deterministic() {
        let g = GrainEffect {
            intensity: 0.5,
            size: 1.0,
        };
        assert_eq!(g.sample(5, 7, 99), g.sample(5, 7, 99));
    }

    #[test]
    fn test_grain_zero_intensity_near_half() {
        let g = GrainEffect {
            intensity: 0.0,
            size: 1.0,
        };
        // With intensity 0 the output should be exactly 0.5
        let s = g.sample(0, 0, 0);
        assert!((s - 0.5).abs() < 1e-5, "zero-intensity grain: {s}");
    }

    // ── ColorGradeParams ──────────────────────────────────────────────────────

    #[test]
    fn test_color_grade_params_default_neutral() {
        let params = ColorGradeParams::default_neutral();
        assert_eq!(params.lift, [0.0, 0.0, 0.0]);
        assert_eq!(params.gamma, [1.0, 1.0, 1.0]);
        assert_eq!(params.gain, [1.0, 1.0, 1.0]);
        assert_eq!(params.saturation, 1.0);
    }

    #[test]
    fn test_apply_lift_gamma_gain_identity() {
        let params = ColorGradeParams::default_neutral();
        let rgb = (0.4, 0.6, 0.2);
        let (r, g, b) = apply_lift_gamma_gain(rgb, &params);
        assert!((r - 0.4).abs() < 1e-10, "r: {r}");
        assert!((g - 0.6).abs() < 1e-10, "g: {g}");
        assert!((b - 0.2).abs() < 1e-10, "b: {b}");
    }

    #[test]
    fn test_apply_lift_gamma_gain_lift_raises_black() {
        let mut params = ColorGradeParams::default_neutral();
        params.lift = [0.2, 0.2, 0.2];
        let (r, _, _) = apply_lift_gamma_gain((0.0, 0.0, 0.0), &params);
        assert!((r - 0.2).abs() < 1e-10, "Lift should raise black: {r}");
    }

    #[test]
    fn test_adjust_saturation_desaturate() {
        let gray = adjust_saturation((0.8, 0.4, 0.2), 0.0);
        let luma = 0.2126 * 0.8 + 0.7152 * 0.4 + 0.0722 * 0.2;
        assert!(
            (gray.0 - luma).abs() < 1e-10,
            "R should be luma: {}",
            gray.0
        );
        assert!(
            (gray.1 - luma).abs() < 1e-10,
            "G should be luma: {}",
            gray.1
        );
        assert!(
            (gray.2 - luma).abs() < 1e-10,
            "B should be luma: {}",
            gray.2
        );
    }

    #[test]
    fn test_adjust_contrast_midpoint_unchanged() {
        let v = adjust_contrast(0.5, 2.0);
        assert!((v - 0.5).abs() < 1e-10, "Midpoint should not shift: {v}");
    }

    #[test]
    fn test_apply_color_grade_output_in_range() {
        let params = ColorGradeParams::default_neutral();
        for px in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.3, 0.8)] {
            let (r, g, b) = apply_color_grade(px, &params);
            assert!(r >= 0.0 && r <= 1.0, "r={r}");
            assert!(g >= 0.0 && g <= 1.0, "g={g}");
            assert!(b >= 0.0 && b <= 1.0, "b={b}");
        }
    }

    #[test]
    fn test_apply_color_grade_frame_length() {
        let params = ColorGradeParams::default_neutral();
        let mut pixels = vec![(0.5f64, 0.3, 0.8); 100];
        apply_color_grade_frame(&mut pixels, &params);
        assert_eq!(pixels.len(), 100);
    }

    #[test]
    fn test_split_tone_neutral_saturation() {
        let params = SplitToneParams {
            shadow_saturation: 0.0,
            highlight_saturation: 0.0,
            ..Default::default()
        };
        let rgb = (0.5, 0.4, 0.6);
        let out = apply_split_tone(rgb, &params);
        assert!((out.0 - 0.5).abs() < 1e-10, "r unchanged: {}", out.0);
        assert!((out.1 - 0.4).abs() < 1e-10, "g unchanged: {}", out.1);
        assert!((out.2 - 0.6).abs() < 1e-10, "b unchanged: {}", out.2);
    }

    #[test]
    fn test_split_tone_output_in_range() {
        let params = SplitToneParams {
            shadow_hue: 210.0,
            shadow_saturation: 0.5,
            highlight_hue: 40.0,
            highlight_saturation: 0.5,
            balance: 0.0,
        };
        for rgb in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.3, 0.7)] {
            let (r, g, b) = apply_split_tone(rgb, &params);
            assert!(r >= 0.0 && r <= 1.0, "r={r}");
            assert!(g >= 0.0 && g <= 1.0, "g={g}");
            assert!(b >= 0.0 && b <= 1.0, "b={b}");
        }
    }
}
