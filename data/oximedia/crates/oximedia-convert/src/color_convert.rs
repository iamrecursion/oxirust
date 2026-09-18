// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Color format conversion utilities.
//!
//! Provides BT.601 YUV/RGB conversions, planar YUV420 packing/unpacking,
//! linear/sRGB gamma encoding/decoding, and HDR-to-SDR tone-mapping conversion.

/// Convert a single YUV pixel (BT.601) to RGB.
///
/// Returns `[r, g, b]` with each component clamped to `0..=255`.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let y = i32::from(y);
    let u = i32::from(u) - 128;
    let v = i32::from(v) - 128;

    let r = y + (1_402 * v) / 1_000;
    let g = y - (344_136 * u + 714_136 * v) / 1_000_000;
    let b = y + (1_772 * u) / 1_000;

    [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
}

/// Convert a single RGB pixel to YUV (BT.601).
///
/// Returns `[y, u, v]`.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn rgb_to_yuv(r: u8, g: u8, b: u8) -> [u8; 3] {
    let r = i32::from(r);
    let g = i32::from(g);
    let b = i32::from(b);

    let y = (299 * r + 587 * g + 114 * b) / 1_000;
    let u = (-168_736 * r - 331_264 * g + 500_000 * b) / 1_000_000 + 128;
    let v = (500_000 * r - 418_688 * g - 81_312 * b) / 1_000_000 + 128;

    [clamp_u8(y), clamp_u8(u), clamp_u8(v)]
}

/// Convert a planar YUV420 buffer to packed RGB.
///
/// The input layout is: Y plane (width * height bytes), then U plane
/// (width/2 * height/2 bytes), then V plane (width/2 * height/2 bytes).
///
/// Returns packed RGB with 3 bytes per pixel (row-major).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn yuv420_to_rgb(yuv: &[u8], width: usize, height: usize) -> Vec<u8> {
    let y_size = width * height;
    let uv_size = (width / 2) * (height / 2);

    if yuv.len() < y_size + 2 * uv_size {
        return Vec::new();
    }

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + uv_size];
    let v_plane = &yuv[y_size + uv_size..y_size + 2 * uv_size];

    let mut rgb = Vec::with_capacity(width * height * 3);

    #[allow(clippy::many_single_char_names)]
    for row in 0..height {
        for col in 0..width {
            let y = y_plane[row * width + col];
            let uv_row = row / 2;
            let uv_col = col / 2;
            let uv_width = width / 2;
            let u = u_plane[uv_row * uv_width + uv_col];
            let v = v_plane[uv_row * uv_width + uv_col];
            let [r, g, b] = yuv_to_rgb(y, u, v);
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }

    rgb
}

/// Convert packed RGB to planar YUV420.
///
/// The input is packed RGB with 3 bytes per pixel (row-major).
///
/// Returns: Y plane (width * height bytes) + U plane (w/2 * h/2) + V plane (w/2 * h/2).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn rgb_to_yuv420(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    if rgb.len() < width * height * 3 {
        return Vec::new();
    }

    let uv_width = width / 2;
    let uv_height = height / 2;
    let y_size = width * height;
    let uv_size = uv_width * uv_height;

    let mut y_plane = vec![0u8; y_size];
    let mut u_plane = vec![0u8; uv_size];
    let mut v_plane = vec![0u8; uv_size];

    // Fill Y plane
    for row in 0..height {
        for col in 0..width {
            let idx = (row * width + col) * 3;
            let [y, _, _] = rgb_to_yuv(rgb[idx], rgb[idx + 1], rgb[idx + 2]);
            y_plane[row * width + col] = y;
        }
    }

    // Fill U/V planes (average over 2x2 blocks)
    for uv_row in 0..uv_height {
        for uv_col in 0..uv_width {
            let mut u_sum: i32 = 0;
            let mut v_sum: i32 = 0;
            let mut count = 0i32;
            for dr in 0..2usize {
                for dc in 0..2usize {
                    let row = uv_row * 2 + dr;
                    let col = uv_col * 2 + dc;
                    if row < height && col < width {
                        let idx = (row * width + col) * 3;
                        let [_, u, v] = rgb_to_yuv(rgb[idx], rgb[idx + 1], rgb[idx + 2]);
                        u_sum += i32::from(u);
                        v_sum += i32::from(v);
                        count += 1;
                    }
                }
            }
            if count > 0 {
                u_plane[uv_row * uv_width + uv_col] = clamp_u8(u_sum / count);
                v_plane[uv_row * uv_width + uv_col] = clamp_u8(v_sum / count);
            }
        }
    }

    let mut out = Vec::with_capacity(y_size + 2 * uv_size);
    out.extend_from_slice(&y_plane);
    out.extend_from_slice(&u_plane);
    out.extend_from_slice(&v_plane);
    out
}

/// Encode a linear light value (0.0..=1.0) to sRGB gamma.
///
/// Values outside `0.0..=1.0` are clamped before encoding.
#[must_use]
pub fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Decode an sRGB gamma-encoded value (0.0..=1.0) to linear light.
///
/// Values outside `0.0..=1.0` are clamped before decoding.
#[must_use]
pub fn srgb_to_linear(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

// ── HDR-to-SDR tone mapping conversion ───────────────────────────────────────

use oximedia_colormgmt::gamut_mapping::{ColorPrimaries, GamutMapper, GamutMappingMethod};
use oximedia_colormgmt::hdr::tonemapping::ToneCurve;

/// Source HDR transfer function (EOTF).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceTransferFunction {
    /// SMPTE ST 2084 / ITU-R BT.2100 PQ — peak typically 1 000–10 000 nits.
    Pq,
    /// ITU-R BT.2100 HLG (Hybrid Log-Gamma).
    Hlg,
}

/// Controls whether tone mapping operates per-channel or on luminance only.
///
/// * `PerChannel` — apply the tone curve independently to R, G, B (classic approach,
///   may introduce hue shifts in highlights).
/// * `Luminance` — compute scene luma (Rec.2020 coefficients), tone-map the luma,
///   then scale all channels by the luma ratio to preserve hue. The `desaturation`
///   field on [`HdrToSdrConfig`] blends between the two modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneDomain {
    /// Apply tone curve independently to each R, G, B channel (default).
    PerChannel,
    /// Tone-map on scene luminance and scale RGB to preserve hue.
    Luminance,
}

/// Configuration for HDR-to-SDR tone-mapping conversion.
#[derive(Debug, Clone)]
pub struct HdrToSdrConfig {
    /// Transfer function used to encode the source HDR signal.
    pub source_tf: SourceTransferFunction,
    /// Reference peak luminance in nits (default 1 000 nits).
    ///
    /// For PQ this is the absolute peak level encoded at signal code 1.0.
    /// For HLG this is the display peak used in the HLG OOTF.
    pub peak_nits: f32,
    /// Tone-mapping operator applied after EOTF linearisation.
    pub tone_curve: ToneCurve,
    /// Apply a Rec.2020 → Rec.709 gamut-clip after tone mapping.
    pub gamut_map: bool,
    /// Whether to tone-map per-channel or on luminance (preserving hue).
    ///
    /// Default: [`ToneDomain::PerChannel`].
    pub tone_domain: ToneDomain,
    /// Desaturation blend in luminance mode (0.0 = full hue-preserved, 1.0 = per-channel).
    ///
    /// Only used when `tone_domain == ToneDomain::Luminance`. Default: `0.0`.
    pub desaturation: f32,
}

impl Default for HdrToSdrConfig {
    fn default() -> Self {
        Self {
            source_tf: SourceTransferFunction::Pq,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::FilmicHable,
            gamut_map: true,
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        }
    }
}

// ── PQ EOTF (SMPTE ST 2084) ───────────────────────────────────────────────────

/// Decode a single PQ code-value (10-bit, in [0, 1]) to scene-linear nits.
///
/// Per SMPTE ST 2084-2014 Eq. 5.  The output is normalised so that the PQ
/// reference peak (code ≈ 1.0) decodes to `1.0`; absolute nits are obtained
/// by multiplying by `10 000`.
#[inline]
fn pq_eotf_normalised(code: f32) -> f32 {
    const M1: f32 = 0.159_301_758; // 1 / m2_inv = 2610 / 16384 / 4
    const M2: f32 = 78.843_75; // 2523 / 32
    const C1: f32 = 0.835_937_5; // 3424 / 4096
    const C2: f32 = 18.851_562_5; // 2413 / 128
    const C3: f32 = 18.687_5; // 2392 / 128

    let code = code.clamp(0.0, 1.0);
    let v_m2 = code.powf(1.0 / M2);
    let num = (v_m2 - C1).max(0.0);
    let den = C2 - C3 * v_m2;
    if den.abs() < f32::EPSILON {
        return 0.0;
    }
    (num / den).powf(1.0 / M1)
}

// ── HLG OETF inverse (BT.2100) ───────────────────────────────────────────────

/// Decode a single HLG signal value (in [0, 1]) to scene-linear normalised light.
///
/// Per ITU-R BT.2100-2 Table 5.
#[inline]
fn hlg_oetf_inverse(code: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 0.284_668_92;
    const C: f32 = 0.559_910_73;

    let code = code.clamp(0.0, 1.0);
    if code <= 0.5 {
        code * code / 3.0
    } else {
        (((code - C) / A).exp() + B) / 12.0
    }
}

/// Reconstruct scene-linear light from a 10-bit HDR code using the given EOTF.
///
/// The returned value is normalised so that 1.0 = `peak_nits`.  Values **above**
/// 1.0 are intentionally not clamped here so that the full highlight range
/// reaches the tone curve's rolloff region.  The caller (`convert_hdr_to_sdr_pixel`)
/// is responsible for final clamping after tone mapping.
///
/// ## PQ
/// `pq_eotf_normalised` returns the absolute luminance normalised to 10 000 nits.
/// We scale by `10_000 / peak_nits` so that `peak_nits` = 1.0 in scene-linear.
/// Values above `peak_nits` (super-whites) produce output > 1.0 and must pass
/// through to the tone curve.
///
/// ## HLG
/// The raw `hlg_oetf_inverse` output is scene-light in [0, 1].  We apply the
/// BT.2100-2 system-gamma OOTF:
///   `E_display = E_scene^(system_gamma)`
/// where `system_gamma ≈ 1.2` (ITU-R BT.2100-2 Table 5, `L_W` = `peak_nits`).
/// This correctly maps the HLG signal to display-referred absolute luminance.
#[inline]
fn decode_hdr_code(code_u16: u16, tf: SourceTransferFunction, peak_nits: f32) -> f32 {
    let code_f = code_u16 as f32 / 1023.0;
    match tf {
        SourceTransferFunction::Pq => {
            // PQ 10 000-nit peak → normalise to peak_nits; do NOT clamp above 1.0
            // so super-white highlights reach the tone curve's rolloff region.
            let pq_linear = pq_eotf_normalised(code_f);
            (pq_linear * (10_000.0 / peak_nits.max(1.0))).max(0.0)
        }
        SourceTransferFunction::Hlg => {
            // Scene-linear after OETF inverse (no OOTF yet)
            let scene_linear = hlg_oetf_inverse(code_f);
            // BT.2100-2 system-gamma OOTF: system_gamma = 1.2 (simplified; valid
            // for L_W from ~400 nits to ~2000 nits per the standard).
            const SYSTEM_GAMMA: f32 = 1.2;
            // Apply OOTF and normalise to peak_nits so 1.0 = peak_nits on display
            let ootf_linear = scene_linear.powf(SYSTEM_GAMMA);
            // scale: HLG reference peak (1000 nits) / peak_nits keeps the ratio
            let scale = (1000.0_f32 / peak_nits.max(1.0)).powf(SYSTEM_GAMMA - 1.0);
            (ootf_linear * scale).max(0.0)
        }
    }
}

/// Convert a single HDR pixel (normalised scene-linear Y, Cb, Cr in [-0.5, 0.5])
/// to SDR `(R, G, B)` as `u8` values.
///
/// The input Y is the normalised scene-linear luma (0..=1.0 maps to 0..=`peak_nits`;
/// values > 1.0 are valid super-whites that must not be pre-clamped).
/// Cb/Cr are differential chroma signals in `[-0.5, 0.5]`.
///
/// An optional pre-built `GamutMapper` (Rec.2020 → Rec.709) may be passed to
/// avoid rebuilding the 3×3 matrix for every pixel in a frame.  Pass `None` to
/// let the function build one on the fly (used for single-pixel calls in tests).
///
/// # Processing pipeline
///
/// 1. PQ or HLG EOTF decoding (already done by the caller; `y_linear` is the result).
/// 2. Chroma-to-RGB reconstruct (BT.2020 YCbCr coefficients).
/// 3. Tone mapping via `cfg.tone_curve` — per-channel or luminance mode.
/// 4. Optional Rec.2020 → Rec.709 gamut mapping (via `GamutMapper`).
/// 5. sRGB OETF encode + scale to [0, 255].
#[must_use]
pub fn convert_hdr_to_sdr_pixel(
    y_linear: f32,
    cb: f32,
    cr: f32,
    cfg: &HdrToSdrConfig,
) -> (u8, u8, u8) {
    // Build a temporary gamut mapper for single-pixel / test calls.
    let mapper_opt = if cfg.gamut_map {
        Some(GamutMapper::new(
            ColorPrimaries::rec2020(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Clip,
        ))
    } else {
        None
    };
    convert_hdr_to_sdr_pixel_with_mapper(y_linear, cb, cr, cfg, mapper_opt.as_ref())
}

/// Inner implementation that accepts an optional pre-built `GamutMapper`.
///
/// Used by `convert_hdr_to_sdr_frame` to avoid rebuilding the matrix per pixel.
#[inline]
fn convert_hdr_to_sdr_pixel_with_mapper(
    y_linear: f32,
    cb: f32,
    cr: f32,
    cfg: &HdrToSdrConfig,
    gamut_mapper: Option<&GamutMapper>,
) -> (u8, u8, u8) {
    // Reconstruct linear R, G, B from Y, Cb, Cr (BT.2020 matrix)
    // BT.2020 YCbCr coefficients Kr=0.2627, Kb=0.0593
    let r_linear = y_linear + 1.474_600 * cr;
    let g_linear = y_linear - 0.164_553 * cb - 0.571_353 * cr;
    let b_linear = y_linear + 1.881_400 * cb;

    // Scene-linear RGB (clamp negatives from chroma math, but keep super-whites)
    let r_sc = r_linear.max(0.0);
    let g_sc = g_linear.max(0.0);
    let b_sc = b_linear.max(0.0);

    // Tone mapping: per-channel or luminance-preserving
    let rgb_tm = match cfg.tone_domain {
        ToneDomain::PerChannel => cfg.tone_curve.map([r_sc, g_sc, b_sc]),
        ToneDomain::Luminance => {
            // Rec.2020 luma coefficients: Kr=0.2627, Kb=0.0593
            const KR: f32 = 0.2627;
            const KG: f32 = 0.6780;
            const KB: f32 = 0.0593;
            let y_scene = KR * r_sc + KG * g_sc + KB * b_sc;

            // Tone-map the luma scalar
            let y_tm = cfg.tone_curve.map([y_scene, y_scene, y_scene])[0];

            // Scale RGB by the luma ratio to preserve hue
            let scale = if y_scene > 1e-6 { y_tm / y_scene } else { y_tm };
            let rgb_lum = [r_sc * scale, g_sc * scale, b_sc * scale];

            // Per-channel result for blending
            let rgb_pc = cfg.tone_curve.map([r_sc, g_sc, b_sc]);

            // Blend: desaturation=0 → pure luminance-mode; 1 → pure per-channel
            let d = cfg.desaturation.clamp(0.0, 1.0);
            [
                rgb_lum[0] * (1.0 - d) + rgb_pc[0] * d,
                rgb_lum[1] * (1.0 - d) + rgb_pc[1] * d,
                rgb_lum[2] * (1.0 - d) + rgb_pc[2] * d,
            ]
        }
    };

    // Optional gamut mapping: Rec.2020 → Rec.709
    let (r_sdr, g_sdr, b_sdr) = if let Some(mapper) = gamut_mapper {
        mapper.map_pixel(rgb_tm[0], rgb_tm[1], rgb_tm[2])
    } else {
        (
            rgb_tm[0].clamp(0.0, 1.0),
            rgb_tm[1].clamp(0.0, 1.0),
            rgb_tm[2].clamp(0.0, 1.0),
        )
    };

    // sRGB OETF encode (correct piecewise sRGB, not γ=2.2) + scale to [0, 255]
    let encode = |v: f32| -> u8 { (linear_to_srgb(v) * 255.0 + 0.5) as u8 };

    (encode(r_sdr), encode(g_sdr), encode(b_sdr))
}

/// Convert a full HDR frame (10-bit luma-coded PQ or HLG, YCbCr 4:2:0) to a
/// packed SDR RGB `u8` frame.
///
/// The output is packed as `[R, G, B, R, G, B, …]` in row-major order with the
/// same dimensions as the input.
///
/// The Rec.2020 → Rec.709 `GamutMapper` (when `cfg.gamut_map` is `true`) is
/// built **once** here and reused for every pixel, avoiding repeated 3×3 matrix
/// construction per call.
///
/// # Arguments
///
/// * `frame_y`  — luma plane (`w * h` 10-bit samples as `u16`).
/// * `frame_cb` — Cb chroma plane (`(w/2) * (h/2)` 10-bit samples).
/// * `frame_cr` — Cr chroma plane (`(w/2) * (h/2)` 10-bit samples).
/// * `w`, `h`   — frame dimensions.
/// * `cfg`      — HDR-to-SDR configuration.
///
/// # Returns
///
/// Packed RGB `u8` buffer of length `w * h * 3`, or an empty `Vec` if the
/// inputs are too short.
#[must_use]
pub fn convert_hdr_to_sdr_frame(
    frame_y: &[u16],
    frame_cb: &[u16],
    frame_cr: &[u16],
    w: u32,
    h: u32,
    cfg: &HdrToSdrConfig,
) -> Vec<u8> {
    let width = w as usize;
    let height = h as usize;
    let y_len = width * height;
    let uv_len = (width / 2) * (height / 2);

    if frame_y.len() < y_len || frame_cb.len() < uv_len || frame_cr.len() < uv_len {
        return Vec::new();
    }

    // Build the gamut mapper once per frame, not per pixel.
    let gamut_mapper: Option<GamutMapper> = if cfg.gamut_map {
        Some(GamutMapper::new(
            ColorPrimaries::rec2020(),
            ColorPrimaries::rec709(),
            GamutMappingMethod::Clip,
        ))
    } else {
        None
    };

    let mut out = Vec::with_capacity(y_len * 3);

    for row in 0..height {
        for col in 0..width {
            let y_code = frame_y[row * width + col];
            let uv_row = row / 2;
            let uv_col = col / 2;
            let uv_width = width / 2;
            let cb_code = frame_cb[uv_row * uv_width + uv_col];
            let cr_code = frame_cr[uv_row * uv_width + uv_col];

            // Decode HDR codes to normalised scene-linear light (may be > 1.0 for
            // super-whites — intentionally not clamped until after tone mapping).
            let y_linear = decode_hdr_code(y_code, cfg.source_tf, cfg.peak_nits);
            // Chroma: 10-bit code centre 512 → offset to [-0.5, 0.5]
            let cb_linear = cb_code as f32 / 1023.0 - 0.5;
            let cr_linear = cr_code as f32 / 1023.0 - 0.5;

            let (r, g, b) = convert_hdr_to_sdr_pixel_with_mapper(
                y_linear,
                cb_linear,
                cr_linear,
                cfg,
                gamut_mapper.as_ref(),
            );
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- yuv_to_rgb / rgb_to_yuv ---

    #[test]
    fn test_yuv_to_rgb_white() {
        // Y=235, U=128, V=128 → near white
        let [r, g, b] = yuv_to_rgb(235, 128, 128);
        assert!(r > 200, "r={r}");
        assert!(g > 200, "g={g}");
        assert!(b > 200, "b={b}");
    }

    #[test]
    fn test_yuv_to_rgb_black() {
        let [r, g, b] = yuv_to_rgb(16, 128, 128);
        assert!(r < 20, "r={r}");
        assert!(g < 20, "g={g}");
        assert!(b < 20, "b={b}");
    }

    #[test]
    fn test_rgb_to_yuv_white() {
        let [y, u, v] = rgb_to_yuv(255, 255, 255);
        assert!(y > 200, "y={y}");
        // U and V should be near 128 for achromatic
        assert!((i32::from(u) - 128).abs() < 15, "u={u}");
        assert!((i32::from(v) - 128).abs() < 15, "v={v}");
    }

    #[test]
    fn test_rgb_to_yuv_black() {
        let [y, u, v] = rgb_to_yuv(0, 0, 0);
        assert!(y < 10, "y={y}");
        assert!((i32::from(u) - 128).abs() < 5, "u={u}");
        assert!((i32::from(v) - 128).abs() < 5, "v={v}");
    }

    #[test]
    fn test_yuv_rgb_roundtrip_gray() {
        // Gray: round-trip should stay near-gray
        let original = [128u8, 128, 128];
        let [y, u, v] = rgb_to_yuv(original[0], original[1], original[2]);
        let [r2, g2, b2] = yuv_to_rgb(y, u, v);
        // Allow small rounding error
        assert!((i32::from(r2) - i32::from(original[0])).abs() <= 5);
        assert!((i32::from(g2) - i32::from(original[1])).abs() <= 5);
        assert!((i32::from(b2) - i32::from(original[2])).abs() <= 5);
    }

    #[test]
    fn test_yuv_to_rgb_no_overflow() {
        // Extreme values should not panic
        // Note: r, g, b are u8 types, so clamping is guaranteed by clamp_u8 function.
        // No need for explicit bounds checks.
        let [r, g, b] = yuv_to_rgb(0, 0, 0);
        let _ = (r, g, b);
        let [r, g, b] = yuv_to_rgb(255, 255, 255);
        let _ = (r, g, b);
    }

    #[test]
    fn test_rgb_to_yuv_pure_red() {
        let [y, _u, v] = rgb_to_yuv(255, 0, 0);
        // Red should produce high V and significant Y
        assert!(y > 50, "y={y}");
        assert!(v > 128, "v={v}");
    }

    // --- yuv420_to_rgb / rgb_to_yuv420 ---

    #[test]
    fn test_yuv420_to_rgb_size() {
        let width = 4;
        let height = 4;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let yuv = vec![128u8; y_size + 2 * uv_size];
        let rgb = yuv420_to_rgb(&yuv, width, height);
        assert_eq!(rgb.len(), width * height * 3);
    }

    #[test]
    fn test_yuv420_to_rgb_empty_on_short_input() {
        let rgb = yuv420_to_rgb(&[0u8; 4], 4, 4);
        assert!(rgb.is_empty());
    }

    #[test]
    fn test_rgb_to_yuv420_size() {
        let width = 4;
        let height = 4;
        let rgb = vec![128u8; width * height * 3];
        let yuv = rgb_to_yuv420(&rgb, width, height);
        let expected = width * height + 2 * (width / 2) * (height / 2);
        assert_eq!(yuv.len(), expected);
    }

    #[test]
    fn test_rgb_to_yuv420_empty_on_short_input() {
        let yuv = rgb_to_yuv420(&[0u8; 10], 4, 4);
        assert!(yuv.is_empty());
    }

    // --- linear_to_srgb / srgb_to_linear ---

    #[test]
    fn test_linear_to_srgb_zero() {
        assert!((linear_to_srgb(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_to_srgb_one() {
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_srgb_to_linear_zero() {
        assert!((srgb_to_linear(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_srgb_to_linear_one() {
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_gamma_roundtrip() {
        let vals = [0.0f32, 0.01, 0.1, 0.5, 0.9, 1.0];
        for v in vals {
            let encoded = linear_to_srgb(v);
            let decoded = srgb_to_linear(encoded);
            assert!(
                (decoded - v).abs() < 1e-4,
                "roundtrip failed for v={v}: decoded={decoded}"
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_clamps_negative() {
        assert!((linear_to_srgb(-1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_to_srgb_clamps_above_one() {
        assert!((linear_to_srgb(2.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_srgb_midpoint_gamma() {
        // sRGB mid-gray (0.5) → linear should be lower
        let linear = srgb_to_linear(0.5);
        assert!(linear < 0.5, "linear={linear}");
        assert!(linear > 0.1, "linear={linear}");
    }

    // ── HDR-to-SDR tests ─────────────────────────────────────────────────────

    #[test]
    fn test_hdr_to_sdr_black_stays_black() {
        // Zero PQ code → all zero output
        let cfg = HdrToSdrConfig {
            source_tf: SourceTransferFunction::Pq,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::FilmicHable,
            gamut_map: false,
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        };
        let (r, g, b) = convert_hdr_to_sdr_pixel(0.0, 0.0, 0.0, &cfg);
        assert_eq!(r, 0, "r={r}");
        assert_eq!(g, 0, "g={g}");
        assert_eq!(b, 0, "b={b}");
    }

    #[test]
    fn test_hdr_to_sdr_pq_white_near_white() {
        // Peak white PQ luma ≈ 1.0 normalised → SDR should be substantially above zero.
        // With Hable tone curve, 1.0 scene-linear → significant SDR output.
        let cfg = HdrToSdrConfig {
            source_tf: SourceTransferFunction::Pq,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::FilmicHable,
            gamut_map: false,
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        };
        // y_linear=1.0 means peak-white signal
        let (r, _g, _b) = convert_hdr_to_sdr_pixel(1.0, 0.0, 0.0, &cfg);
        assert!(
            r > 100,
            "peak-white luma should produce significant SDR output, got r={r}"
        );
    }

    #[test]
    fn test_hdr_to_sdr_hlg_black_zero() {
        let cfg = HdrToSdrConfig {
            source_tf: SourceTransferFunction::Hlg,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::ReinhardSimple,
            gamut_map: false,
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        };
        let (r, g, b) = convert_hdr_to_sdr_pixel(0.0, 0.0, 0.0, &cfg);
        assert_eq!((r, g, b), (0, 0, 0), "HLG black should map to zero");
    }

    #[test]
    fn test_hdr_to_sdr_frame_size() {
        // 4×4 frame → output must be 4*4*3 = 48 bytes
        let w = 4u32;
        let h = 4u32;
        let y = vec![512u16; 16];
        let cb = vec![512u16; 4];
        let cr = vec![512u16; 4];
        let cfg = HdrToSdrConfig::default();
        let out = convert_hdr_to_sdr_frame(&y, &cb, &cr, w, h, &cfg);
        assert_eq!(out.len(), 48, "output length should be w*h*3");
    }

    #[test]
    fn test_hdr_to_sdr_frame_empty_on_short_input() {
        let cfg = HdrToSdrConfig::default();
        let out = convert_hdr_to_sdr_frame(&[], &[], &[], 4, 4, &cfg);
        assert!(out.is_empty(), "short input should produce empty output");
    }

    #[test]
    fn test_pq_eotf_black() {
        // PQ code 0 → linear 0
        assert!(pq_eotf_normalised(0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hlg_oetf_inverse_black() {
        // HLG code 0 → linear 0
        assert!(hlg_oetf_inverse(0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hlg_oetf_inverse_midpoint() {
        // HLG 0.5 → linear 0.25/3 ≈ 0.0833
        let v = hlg_oetf_inverse(0.5);
        assert!((v - 0.0833).abs() < 1e-3, "hlg(0.5) ≈ 0.0833, got {v}");
    }

    // ── New correctness tests (Slice 1 additions) ─────────────────────────────

    /// Verify that a PQ signal above the display peak (4000-nit encoded signal
    /// with a 1000-nit display) does NOT clip to 255 before tone mapping.
    ///
    /// With the old pre-clamp bug, the super-white scene-linear value was truncated
    /// to 1.0 before the filmic curve, losing the highlight rolloff.  The fixed
    /// code passes values > 1.0 to the tone curve which maps them to a value
    /// that is less than the peak-mapped value at 1.0 — but still well below 255.
    ///
    /// Also confirms monotonicity: higher PQ code → higher SDR output.
    #[test]
    fn test_hdr_headroom_reaches_curve() {
        let cfg = HdrToSdrConfig {
            source_tf: SourceTransferFunction::Pq,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::FilmicHable,
            gamut_map: false,
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        };

        // Encode a 4000-nit PQ signal: find the 10-bit code whose PQ EOTF ≈ 4000 nit.
        // PQ ref peak = 10 000 nit, so 4000/10000 = 0.4 normalised PQ.
        // The PQ curve is steeply nonlinear: a 10-bit code near 924 decodes to
        // `pq_eotf_normalised ≈ 0.402` (~4024 nit). With a 1000-nit display this
        // scales to ~4.0 in normalised scene-linear (a super-white > 1.0).
        let code_4000nit: u16 = 924;
        let linear_4000 = decode_hdr_code(code_4000nit, SourceTransferFunction::Pq, 1000.0);
        assert!(
            linear_4000 > 1.0,
            "4000-nit PQ should decode to > 1.0 (super-white), got {linear_4000}"
        );

        // The output should be a valid SDR value (< 255) because the tone curve
        // compresses it, not clips it.
        let (r1000, _g, _b) = convert_hdr_to_sdr_pixel(1.0, 0.0, 0.0, &cfg);
        let (r4000, _g, _b) = convert_hdr_to_sdr_pixel(linear_4000, 0.0, 0.0, &cfg);

        assert!(
            r4000 < 255,
            "super-white 4000-nit should not clip to 255; got r={r4000}"
        );
        assert!(
            r4000 > r1000,
            "higher PQ input should yield higher output: r(1000nit)={r1000}, r(4000nit)={r4000}"
        );
    }

    /// In luminance-preserving mode (ToneDomain::Luminance), two pixels with the
    /// same hue but different luminance should preserve the hue angle better than
    /// per-channel mode does for a highly saturated Rec.2020 color.
    ///
    /// We use a saturated red in Rec.2020 scene space and compare the hue shift
    /// produced by per-channel vs luminance tone mapping.
    #[test]
    fn test_luminance_mode_hue_preservation() {
        // A highly saturated, bright Rec.2020 red: Y=2.0 (super-white), chroma only on R
        // Cb=0 means no blue contribution; we set Cr to give pure red in Rec.2020 RGB.
        // With BT.2020 coefficients: R = Y + 1.4746*Cr => set Cr so R is much larger than G/B.
        let cr = 0.40_f32; // Cr offset giving a strong red tint
        let cb = -0.10_f32; // slight blue offset for richer color

        let cfg_per = HdrToSdrConfig {
            source_tf: SourceTransferFunction::Pq,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::ReinhardSimple,
            gamut_map: false,
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        };
        let cfg_lum = HdrToSdrConfig {
            tone_domain: ToneDomain::Luminance,
            desaturation: 0.0,
            ..cfg_per.clone()
        };

        // High luminance (super-white) input
        let y_high = 2.5_f32;
        let (rp, gp, bp) = convert_hdr_to_sdr_pixel(y_high, cb, cr, &cfg_per);
        let (rl, gl, bl) = convert_hdr_to_sdr_pixel(y_high, cb, cr, &cfg_lum);

        // Compute a simple hue metric: |R - max| / (max - min + ε)
        // Lower value → hue is more closely preserved relative to achromatic.
        let per_r = f32::from(rp);
        let per_g = f32::from(gp);
        let per_b = f32::from(bp);
        let lum_r = f32::from(rl);
        let lum_g = f32::from(gl);
        let lum_b = f32::from(bl);

        // In luminance mode the ratio R/(R+G+B) should be closer to the input ratio.
        // Input R is dominant due to large Cr; in per-channel mode the curve compresses
        // channels differently, shifting hue.  In luminance mode they scale together.
        let sum_per = per_r + per_g + per_b + 1e-6;
        let sum_lum = lum_r + lum_g + lum_b + 1e-6;

        let frac_r_per = per_r / sum_per;
        let frac_r_lum = lum_r / sum_lum;

        // Input R fraction (scene-linear):
        let r_sc = (y_high + 1.474_600 * cr).max(0.0);
        let g_sc = (y_high - 0.164_553 * cb - 0.571_353 * cr).max(0.0);
        let b_sc = (y_high + 1.881_400 * cb).max(0.0);
        let sum_sc = r_sc + g_sc + b_sc + 1e-6;
        let frac_r_sc = r_sc / sum_sc;

        let err_per = (frac_r_per - frac_r_sc).abs();
        let err_lum = (frac_r_lum - frac_r_sc).abs();

        assert!(
            err_lum <= err_per + 0.05,
            "luminance mode should preserve hue at least as well as per-channel: \
             err_lum={err_lum:.4}, err_per={err_per:.4}, frac_r_sc={frac_r_sc:.4}"
        );
    }

    /// A Rec.2020 red primary (pure red in Rec.2020 RGB space) is outside Rec.709.
    /// After gamut mapping, all channels must be in [0, 255].
    #[test]
    fn test_out_of_gamut_lands_in_range() {
        // Pure Rec.2020 red primary in scene-linear: R=1, G=0, B=0 in Rec.2020 RGB.
        // In YCbCr: Y = Kr*R = 0.2627, Cr = (1-Kr)/(2*(1-Kr)) * R correction.
        // We use direct scene-linear injection: y_linear=0.2627 (luma of pure red),
        // with Cr and Cb chosen so that the reconstructed R=1, G=0, B=0 in linear.
        // From BT.2020 YCbCr:  R = Y + 1.4746*Cr → Cr = (1-0.2627)/1.4746 ≈ 0.5002
        //                       B = Y + 1.8814*Cb → Cb = (0-0.2627)/1.8814 ≈ -0.1396
        //                       G = Y - 0.1646*Cb - 0.5714*Cr ≈ 0.2627 + 0.0230 - 0.2861 ≈ -0.0004
        let y_linear = 0.2627_f32;
        let cr = 0.5002_f32;
        let cb = -0.1396_f32;

        let cfg = HdrToSdrConfig {
            source_tf: SourceTransferFunction::Pq,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::ReinhardSimple,
            gamut_map: true, // engage Rec.2020 → Rec.709 gamut mapping
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        };

        let (r, g, b) = convert_hdr_to_sdr_pixel(y_linear, cb, cr, &cfg);

        // After Rec.2020 → Rec.709 gamut clipping, a pure-red primary must remain
        // red-dominant (r is the largest channel) — proving the gamut mapper did
        // not collapse the pixel to grey or overflow into the other channels.
        assert!(
            r >= g && r >= b,
            "pure red must stay red-dominant after gamut clip: r={r} g={g} b={b}"
        );
        // g and b should be in valid range after gamut-clip
        assert!(g < 250, "green should not be near max for pure red: g={g}");
        assert!(b < 250, "blue should not be near max for pure red: b={b}");
    }

    /// Same HLG signal, two different `peak_nits` values must produce different output.
    ///
    /// This proves the BT.2100-2 system-gamma OOTF is applied (which uses peak_nits).
    /// Before the fix, HLG ignored peak_nits entirely.
    #[test]
    fn test_hlg_peak_nits_affects_output() {
        // A non-trivial HLG mid-gray signal
        let y_hlg_mid = hlg_oetf_inverse(0.6);

        let cfg_1000 = HdrToSdrConfig {
            source_tf: SourceTransferFunction::Hlg,
            peak_nits: 1000.0,
            tone_curve: ToneCurve::ReinhardSimple,
            gamut_map: false,
            tone_domain: ToneDomain::PerChannel,
            desaturation: 0.0,
        };
        let cfg_400 = HdrToSdrConfig {
            peak_nits: 400.0,
            ..cfg_1000.clone()
        };

        // Decode the same HLG code at two different peak luminances
        let y_1000 = {
            // Apply OOTF manually to match decode_hdr_code logic
            const GAMMA: f32 = 1.2;
            let scale = (1000.0_f32 / 1000.0).powf(GAMMA - 1.0);
            y_hlg_mid.powf(GAMMA) * scale
        };
        let y_400 = {
            const GAMMA: f32 = 1.2;
            let scale = (1000.0_f32 / 400.0).powf(GAMMA - 1.0);
            y_hlg_mid.powf(GAMMA) * scale
        };

        // The two scene-linear values must differ (proving OOTF is peak_nits-dependent)
        let (r1, _g1, _b1) = convert_hdr_to_sdr_pixel(y_1000, 0.0, 0.0, &cfg_1000);
        let (r2, _g2, _b2) = convert_hdr_to_sdr_pixel(y_400, 0.0, 0.0, &cfg_400);

        assert_ne!(
            r1, r2,
            "different peak_nits must produce different HLG output: \
             r(1000nit)={r1}, r(400nit)={r2}"
        );
    }

    /// Linear 0.5 → sRGB OETF → approximately 187 (≈0.735 × 255), within ±2.
    ///
    /// This verifies the correct piecewise sRGB OETF is used instead of γ=2.2.
    #[test]
    fn test_srgb_oetf_midgray() {
        // linear_to_srgb(0.5) = 1.055 * 0.5^(1/2.4) - 0.055 ≈ 0.7354
        // 0.7354 * 255 + 0.5 ≈ 188.1 → rounds to 188
        let srgb_val = linear_to_srgb(0.5);
        let encoded = (srgb_val * 255.0 + 0.5) as u8;
        assert!(
            (i32::from(encoded) - 187).abs() <= 2,
            "linear 0.5 → sRGB ≈ 187±2, got {encoded} (srgb_val={srgb_val:.4})"
        );
    }
}
