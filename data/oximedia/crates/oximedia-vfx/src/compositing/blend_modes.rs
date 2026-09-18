//! Additional blend modes for compositing operations.
//!
//! This module provides a `BlendMode` enum with per-channel and per-pixel
//! blending functions, plus full-frame compositing helpers.  It complements the
//! existing `blend::BlendMode` with a richer API (including `apply` /
//! `apply_with_opacity` over whole frame buffers).

// ── BlendMode ──────────────────────────────────────────────────────────────────

/// Blend mode applied to each pixel channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Normal alpha compositing: `dst = src * src_a + dst * (1 − src_a)`.
    Normal,
    /// Multiply: `dst *= src`.
    Multiply,
    /// Screen: `dst = 1 − (1 − src) * (1 − dst)`.
    Screen,
    /// Overlay: Multiply for dark pixels, Screen for light pixels.
    Overlay,
    /// Soft Light: gentler version of Overlay.
    SoftLight,
    /// Hard Light: like Overlay but source/dest roles reversed.
    HardLight,
    /// Color Dodge: brightens destination.
    Dodge,
    /// Color Burn: darkens destination.
    Burn,
    /// Difference: `abs(src − dst)`.
    Difference,
    /// Exclusion: `src + dst − 2 * src * dst`.
    Exclusion,
    /// Hue: hue from source, saturation + luminosity from destination.
    Hue,
    /// Saturation: saturation from source, hue + luminosity from destination.
    Saturation,
    /// Color: hue + saturation from source, luminosity from destination.
    Color,
    /// Luminosity: luminosity from source, hue + saturation from destination.
    Luminosity,
    /// Additive: `dst + src` clamped to 1.
    Add,
    /// Subtract: `dst − src` clamped to 0.
    Subtract,
    /// Darken: `min(src, dst)` per channel.
    Darken,
    /// Lighten: `max(src, dst)` per channel.
    Lighten,
    /// Vivid Light: combines Color Dodge and Color Burn.
    /// Dodges (brightens) if src > 0.5, burns (darkens) if src <= 0.5.
    VividLight,
    /// Pin Light: replaces dark pixels with darker of src/dst, light with lighter.
    /// If src > 0.5: max(dst, 2*(src-0.5)), else min(dst, 2*src).
    PinLight,
    /// Hard Mix: quantises output to 0 or 1 per channel based on Vivid Light result.
    HardMix,
    /// Linear Light: combines Linear Dodge and Linear Burn.
    /// dst + 2*src - 1.
    LinearLight,
    /// Linear Burn: dst + src - 1.
    LinearBurn,
    /// Linear Dodge (Add): dst + src (same as Add but included for Photoshop parity).
    LinearDodge,
    /// Darker Color: picks the darker pixel overall (by luminance).
    DarkerColor,
    /// Lighter Color: picks the lighter pixel overall (by luminance).
    LighterColor,
    /// Divide: dst / src.
    Divide,
}

impl BlendMode {
    /// Human-readable name of this blend mode.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Multiply => "Multiply",
            Self::Screen => "Screen",
            Self::Overlay => "Overlay",
            Self::SoftLight => "Soft Light",
            Self::HardLight => "Hard Light",
            Self::Dodge => "Dodge",
            Self::Burn => "Burn",
            Self::Difference => "Difference",
            Self::Exclusion => "Exclusion",
            Self::Hue => "Hue",
            Self::Saturation => "Saturation",
            Self::Color => "Color",
            Self::Luminosity => "Luminosity",
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Darken => "Darken",
            Self::Lighten => "Lighten",
            Self::VividLight => "Vivid Light",
            Self::PinLight => "Pin Light",
            Self::HardMix => "Hard Mix",
            Self::LinearLight => "Linear Light",
            Self::LinearBurn => "Linear Burn",
            Self::LinearDodge => "Linear Dodge",
            Self::DarkerColor => "Darker Color",
            Self::LighterColor => "Lighter Color",
            Self::Divide => "Divide",
        }
    }

    /// Blend a single channel value; both `src` and `dst` must be in [0.0, 1.0].
    ///
    /// Returns the blended value clamped to [0.0, 1.0].
    #[must_use]
    pub fn blend_channel(&self, src: f32, dst: f32) -> f32 {
        let result = match self {
            Self::Normal => src,
            Self::Multiply => src * dst,
            Self::Screen => 1.0 - (1.0 - src) * (1.0 - dst),
            Self::Overlay => {
                if dst < 0.5 {
                    2.0 * dst * src
                } else {
                    1.0 - 2.0 * (1.0 - dst) * (1.0 - src)
                }
            }
            Self::SoftLight => {
                if src < 0.5 {
                    dst - (1.0 - 2.0 * src) * dst * (1.0 - dst)
                } else {
                    dst + (2.0 * src - 1.0) * (d_func(dst) - dst)
                }
            }
            Self::HardLight => {
                if src < 0.5 {
                    2.0 * dst * src
                } else {
                    1.0 - 2.0 * (1.0 - dst) * (1.0 - src)
                }
            }
            Self::Dodge => {
                if src < 1.0 {
                    (dst / (1.0 - src)).min(1.0)
                } else {
                    1.0
                }
            }
            Self::Burn => {
                if src > 0.0 {
                    1.0 - ((1.0 - dst) / src).min(1.0)
                } else {
                    0.0
                }
            }
            Self::Difference => (src - dst).abs(),
            Self::Exclusion => src + dst - 2.0 * src * dst,
            // HSL modes fall back to Normal for the per-channel call.
            Self::Hue | Self::Saturation | Self::Color | Self::Luminosity => src,
            Self::Add => (src + dst).min(1.0),
            Self::Subtract => (dst - src).max(0.0),
            Self::Darken => src.min(dst),
            Self::Lighten => src.max(dst),
            Self::VividLight => {
                if src <= 0.5 {
                    // Color Burn with 2*src
                    let s2 = 2.0 * src;
                    if s2 > 0.0 {
                        1.0 - ((1.0 - dst) / s2).min(1.0)
                    } else {
                        0.0
                    }
                } else {
                    // Color Dodge with 2*(src-0.5)
                    let s2 = 2.0 * (src - 0.5);
                    if s2 < 1.0 {
                        (dst / (1.0 - s2)).min(1.0)
                    } else {
                        1.0
                    }
                }
            }
            Self::PinLight => {
                if src > 0.5 {
                    dst.max(2.0 * (src - 0.5))
                } else {
                    dst.min(2.0 * src)
                }
            }
            Self::HardMix => {
                // Use Vivid Light result, then threshold at 0.5
                let vl = Self::VividLight.blend_channel(src, dst);
                if vl < 0.5 {
                    0.0
                } else {
                    1.0
                }
            }
            Self::LinearLight => {
                // dst + 2*src - 1
                (dst + 2.0 * src - 1.0).clamp(0.0, 1.0)
            }
            Self::LinearBurn => (dst + src - 1.0).max(0.0),
            Self::LinearDodge => (dst + src).min(1.0),
            // DarkerColor / LighterColor need full pixel context; per-channel fallback
            Self::DarkerColor | Self::LighterColor => src,
            Self::Divide => {
                if src > 0.0 {
                    (dst / src).min(1.0)
                } else {
                    1.0
                }
            }
        };
        result.clamp(0.0, 1.0)
    }

    /// Blend two RGBA pixels.
    ///
    /// `src_rgba` and `dst_rgba` are `(r, g, b, a)` tuples in [0.0, 1.0].
    /// Returns the composited `(r, g, b, a)`.
    #[must_use]
    pub fn blend_pixel(
        &self,
        src_rgba: (f32, f32, f32, f32),
        dst_rgba: (f32, f32, f32, f32),
    ) -> (f32, f32, f32, f32) {
        let (sr, sg, sb, sa) = src_rgba;
        let (dr, dg, db, da) = dst_rgba;

        if sa <= 0.0 {
            return dst_rgba;
        }

        // For HSL modes and whole-pixel modes we need full RGB context.
        let (br, bg, bb) = match self {
            Self::Hue => {
                let b_hsl = rgb_to_hsl(dr, dg, db);
                let s_hsl = rgb_to_hsl(sr, sg, sb);
                hsl_to_rgb(s_hsl.0, b_hsl.1, b_hsl.2)
            }
            Self::Saturation => {
                let b_hsl = rgb_to_hsl(dr, dg, db);
                let s_hsl = rgb_to_hsl(sr, sg, sb);
                hsl_to_rgb(b_hsl.0, s_hsl.1, b_hsl.2)
            }
            Self::Color => {
                let b_hsl = rgb_to_hsl(dr, dg, db);
                let s_hsl = rgb_to_hsl(sr, sg, sb);
                hsl_to_rgb(s_hsl.0, s_hsl.1, b_hsl.2)
            }
            Self::Luminosity => {
                let b_hsl = rgb_to_hsl(dr, dg, db);
                let s_hsl = rgb_to_hsl(sr, sg, sb);
                hsl_to_rgb(b_hsl.0, b_hsl.1, s_hsl.2)
            }
            Self::DarkerColor => {
                let src_lum = 0.299 * sr + 0.587 * sg + 0.114 * sb;
                let dst_lum = 0.299 * dr + 0.587 * dg + 0.114 * db;
                if src_lum < dst_lum {
                    (sr, sg, sb)
                } else {
                    (dr, dg, db)
                }
            }
            Self::LighterColor => {
                let src_lum = 0.299 * sr + 0.587 * sg + 0.114 * sb;
                let dst_lum = 0.299 * dr + 0.587 * dg + 0.114 * db;
                if src_lum > dst_lum {
                    (sr, sg, sb)
                } else {
                    (dr, dg, db)
                }
            }
            _ => (
                self.blend_channel(sr, dr),
                self.blend_channel(sg, dg),
                self.blend_channel(sb, db),
            ),
        };

        // Porter-Duff "over" for alpha compositing of the blended color.
        let out_a = sa + da * (1.0 - sa);
        if out_a <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let composite = |blended: f32, dst_ch: f32| -> f32 {
            (blended * sa + dst_ch * da * (1.0 - sa)) / out_a
        };

        (
            composite(br, dr).clamp(0.0, 1.0),
            composite(bg, dg).clamp(0.0, 1.0),
            composite(bb, db).clamp(0.0, 1.0),
            out_a.clamp(0.0, 1.0),
        )
    }

    /// Composite `src_frame` over `dst_frame` using this blend mode.
    ///
    /// Both frames must be RGBA (4 bytes per pixel) with `width * height * 4`
    /// bytes.  Silently returns if sizes are mismatched.
    pub fn apply(&self, src_frame: &[u8], dst_frame: &mut [u8], width: u32, height: u32) {
        self.apply_with_opacity(src_frame, dst_frame, width, height, 1.0);
    }

    /// Composite `src_frame` over `dst_frame` with an additional `opacity`
    /// factor in [0.0, 1.0].  `opacity = 1.0` is equivalent to [`apply`].
    ///
    /// [`apply`]: BlendMode::apply
    pub fn apply_with_opacity(
        &self,
        src_frame: &[u8],
        dst_frame: &mut [u8],
        width: u32,
        height: u32,
        opacity: f32,
    ) {
        let pixel_count = (width as usize) * (height as usize);
        let expected = pixel_count * 4;
        if src_frame.len() < expected || dst_frame.len() < expected {
            return;
        }

        let opacity = opacity.clamp(0.0, 1.0);

        for i in 0..pixel_count {
            let base = i * 4;

            let sr = src_frame[base] as f32 / 255.0;
            let sg = src_frame[base + 1] as f32 / 255.0;
            let sb = src_frame[base + 2] as f32 / 255.0;
            let sa = src_frame[base + 3] as f32 / 255.0 * opacity;

            let dr = dst_frame[base] as f32 / 255.0;
            let dg = dst_frame[base + 1] as f32 / 255.0;
            let db = dst_frame[base + 2] as f32 / 255.0;
            let da = dst_frame[base + 3] as f32 / 255.0;

            let (or, og, ob, oa) = self.blend_pixel((sr, sg, sb, sa), (dr, dg, db, da));

            dst_frame[base] = (or * 255.0).clamp(0.0, 255.0) as u8;
            dst_frame[base + 1] = (og * 255.0).clamp(0.0, 255.0) as u8;
            dst_frame[base + 2] = (ob * 255.0).clamp(0.0, 255.0) as u8;
            dst_frame[base + 3] = (oa * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Soft-light D function per the W3C compositing spec.
fn d_func(x: f32) -> f32 {
    if x <= 0.25 {
        ((16.0 * x - 12.0) * x + 4.0) * x
    } else {
        x.sqrt()
    }
}

/// Convert RGB (each in [0,1]) to HSL.
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let l = (max + min) / 2.0;

    if delta < 1e-7 {
        return (0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if (max - r).abs() < 1e-7 {
        ((g - b) / delta + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if (max - g).abs() < 1e-7 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    (h, s, l)
}

/// Convert HSL to RGB.
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

    let hue_to_rgb = |p: f32, q: f32, mut t: f32| -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 0.5 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };

    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(r: u8, g: u8, b: u8, a: u8, pixels: usize) -> Vec<u8> {
        let mut f = vec![0u8; pixels * 4];
        for i in 0..pixels {
            f[i * 4] = r;
            f[i * 4 + 1] = g;
            f[i * 4 + 2] = b;
            f[i * 4 + 3] = a;
        }
        f
    }

    #[test]
    fn test_normal_blend_full_alpha() {
        // With src fully opaque, dst should become src color.
        let (or, og, _ob, oa) =
            BlendMode::Normal.blend_pixel((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0));
        assert!((or - 1.0).abs() < 1e-4, "r should be 1.0, got {or}");
        assert!((og - 0.0).abs() < 1e-4, "g should be 0.0, got {og}");
        assert!((oa - 1.0).abs() < 1e-4, "alpha should be 1.0, got {oa}");
    }

    #[test]
    fn test_multiply_white() {
        // Multiply by white (1.0) should leave destination unchanged.
        let (or, og, ob, oa) =
            BlendMode::Multiply.blend_pixel((1.0, 1.0, 1.0, 1.0), (0.5, 0.3, 0.7, 1.0));
        assert!((or - 0.5).abs() < 0.02, "r: expected ~0.5, got {or}");
        assert!((og - 0.3).abs() < 0.02, "g: expected ~0.3, got {og}");
        assert!((ob - 0.7).abs() < 0.02, "b: expected ~0.7, got {ob}");
        let _ = oa;
    }

    #[test]
    fn test_multiply_black() {
        // Multiply by black should produce black.
        let (or, og, ob, _oa) =
            BlendMode::Multiply.blend_pixel((0.0, 0.0, 0.0, 1.0), (0.8, 0.6, 0.4, 1.0));
        assert!(or < 0.05, "r should be ~0, got {or}");
        assert!(og < 0.05, "g should be ~0, got {og}");
        assert!(ob < 0.05, "b should be ~0, got {ob}");
    }

    #[test]
    fn test_screen_black() {
        // Screen with black src (alpha=1) leaves dst unchanged.
        let dr = 0.6_f32;
        let dg = 0.4_f32;
        let db = 0.2_f32;
        let (or, og, ob, _) =
            BlendMode::Screen.blend_pixel((0.0, 0.0, 0.0, 1.0), (dr, dg, db, 1.0));
        // screen(0, dst) = 1 - (1-0)*(1-dst) = dst  → should preserve dst
        assert!((or - dr).abs() < 0.02, "r: expected {dr}, got {or}");
        assert!((og - dg).abs() < 0.02, "g: expected {dg}, got {og}");
        assert!((ob - db).abs() < 0.02, "b: expected {db}, got {ob}");
    }

    #[test]
    fn test_difference_identical() {
        // Difference of identical pixels should give 0 for RGB channels.
        let v = 0.7_f32;
        let ch = BlendMode::Difference.blend_channel(v, v);
        assert!(
            ch.abs() < 1e-5,
            "difference of identical should be 0, got {ch}"
        );
    }

    #[test]
    fn test_add_clamped() {
        // 0.8 + 0.5 should clamp to 1.0.
        let ch = BlendMode::Add.blend_channel(0.8, 0.5);
        assert!(
            (ch - 1.0).abs() < 1e-5,
            "add clamped: expected 1.0, got {ch}"
        );
    }

    #[test]
    fn test_darken_picks_min() {
        let ch = BlendMode::Darken.blend_channel(0.3, 0.7);
        assert!((ch - 0.3).abs() < 1e-5, "darken should pick 0.3, got {ch}");
    }

    #[test]
    fn test_lighten_picks_max() {
        let ch = BlendMode::Lighten.blend_channel(0.3, 0.7);
        assert!((ch - 0.7).abs() < 1e-5, "lighten should pick 0.7, got {ch}");
    }

    #[test]
    fn test_blend_pixel_alpha_zero() {
        // Fully transparent src → dst should be unchanged.
        let dst = (0.4, 0.5, 0.6, 1.0);
        let (or, og, ob, oa) = BlendMode::Normal.blend_pixel((1.0, 0.0, 0.0, 0.0), dst);
        assert!(
            (or - dst.0).abs() < 1e-5,
            "r unchanged: expected {}, got {or}",
            dst.0
        );
        assert!(
            (og - dst.1).abs() < 1e-5,
            "g unchanged: expected {}, got {og}",
            dst.1
        );
        assert!(
            (ob - dst.2).abs() < 1e-5,
            "b unchanged: expected {}, got {ob}",
            dst.2
        );
        assert!(
            (oa - dst.3).abs() < 1e-5,
            "a unchanged: expected {}, got {oa}",
            dst.3
        );
    }

    #[test]
    fn test_vivid_light_bright_src() {
        // With src=0.8 (> 0.5), acts like Dodge
        let ch = BlendMode::VividLight.blend_channel(0.8, 0.3);
        assert!(ch > 0.3, "vivid light dodge should brighten, got {ch}");
    }

    #[test]
    fn test_vivid_light_dark_src() {
        // With src=0.2 (< 0.5), acts like Burn
        let ch = BlendMode::VividLight.blend_channel(0.2, 0.8);
        assert!(ch < 0.8, "vivid light burn should darken, got {ch}");
    }

    #[test]
    fn test_pin_light_bright_src() {
        // src=0.8 > 0.5: max(dst, 2*(0.8-0.5)) = max(0.3, 0.6) = 0.6
        let ch = BlendMode::PinLight.blend_channel(0.8, 0.3);
        assert!(
            (ch - 0.6).abs() < 0.01,
            "pin light bright: expected 0.6, got {ch}"
        );
    }

    #[test]
    fn test_pin_light_dark_src() {
        // src=0.2 <= 0.5: min(dst, 2*0.2) = min(0.7, 0.4) = 0.4
        let ch = BlendMode::PinLight.blend_channel(0.2, 0.7);
        assert!(
            (ch - 0.4).abs() < 0.01,
            "pin light dark: expected 0.4, got {ch}"
        );
    }

    #[test]
    fn test_hard_mix_quantizes() {
        // Hard Mix should output only 0 or 1
        for s in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for d in [0.1, 0.3, 0.5, 0.7, 0.9] {
                let ch = BlendMode::HardMix.blend_channel(s, d);
                assert!(
                    ch == 0.0 || ch == 1.0,
                    "hard mix should be 0 or 1, got {ch} for src={s}, dst={d}"
                );
            }
        }
    }

    #[test]
    fn test_linear_light() {
        // dst + 2*src - 1 = 0.4 + 2*0.6 - 1 = 0.6
        let ch = BlendMode::LinearLight.blend_channel(0.6, 0.4);
        assert!(
            (ch - 0.6).abs() < 0.01,
            "linear light: expected 0.6, got {ch}"
        );
    }

    #[test]
    fn test_linear_burn() {
        // dst + src - 1 = 0.7 + 0.5 - 1 = 0.2
        let ch = BlendMode::LinearBurn.blend_channel(0.5, 0.7);
        assert!(
            (ch - 0.2).abs() < 0.01,
            "linear burn: expected 0.2, got {ch}"
        );
    }

    #[test]
    fn test_linear_dodge_same_as_add() {
        let ch_add = BlendMode::Add.blend_channel(0.3, 0.4);
        let ch_ld = BlendMode::LinearDodge.blend_channel(0.3, 0.4);
        assert!(
            (ch_add - ch_ld).abs() < 0.01,
            "linear dodge should match add: {ch_add} vs {ch_ld}"
        );
    }

    #[test]
    fn test_divide() {
        // dst / src = 0.5 / 1.0 = 0.5
        let ch = BlendMode::Divide.blend_channel(1.0, 0.5);
        assert!((ch - 0.5).abs() < 0.01, "divide: expected 0.5, got {ch}");
    }

    #[test]
    fn test_divide_by_zero() {
        let ch = BlendMode::Divide.blend_channel(0.0, 0.5);
        assert!(
            (ch - 1.0).abs() < 0.01,
            "divide by zero should return 1.0, got {ch}"
        );
    }

    #[test]
    fn test_darker_color_pixel() {
        // src is darker (lower luminance) than dst
        let (or, og, ob, _) =
            BlendMode::DarkerColor.blend_pixel((0.1, 0.1, 0.1, 1.0), (0.9, 0.9, 0.9, 1.0));
        assert!(or < 0.5, "darker color should pick src, got r={or}");
        assert!(og < 0.5, "darker color should pick src, got g={og}");
        assert!(ob < 0.5, "darker color should pick src, got b={ob}");
    }

    #[test]
    fn test_lighter_color_pixel() {
        let (or, og, ob, _) =
            BlendMode::LighterColor.blend_pixel((0.9, 0.9, 0.9, 1.0), (0.1, 0.1, 0.1, 1.0));
        assert!(or > 0.5, "lighter color should pick src, got r={or}");
        assert!(og > 0.5, "lighter color should pick src, got g={og}");
        assert!(ob > 0.5, "lighter color should pick src, got b={ob}");
    }

    #[test]
    fn test_all_new_blend_modes_have_names() {
        let modes = [
            BlendMode::VividLight,
            BlendMode::PinLight,
            BlendMode::HardMix,
            BlendMode::LinearLight,
            BlendMode::LinearBurn,
            BlendMode::LinearDodge,
            BlendMode::DarkerColor,
            BlendMode::LighterColor,
            BlendMode::Divide,
        ];
        let mut names = std::collections::HashSet::new();
        for mode in &modes {
            let name = mode.name();
            assert!(!name.is_empty());
            assert!(names.insert(name), "duplicate name: {name}");
        }
    }

    #[test]
    fn test_apply_frame_size() {
        // Output frame should have same byte count as input.
        let width = 4_u32;
        let height = 4_u32;
        let pixels = (width * height) as usize;
        let src = make_frame(255, 0, 0, 128, pixels);
        let mut dst = make_frame(0, 255, 0, 255, pixels);
        BlendMode::Normal.apply(&src, &mut dst, width, height);
        assert_eq!(dst.len(), pixels * 4, "output frame should have same size");
    }
}
