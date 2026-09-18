//! Color conversion and tone mapping operations.
//!
//! Provides pure-Rust CPU implementations for:
//! - YUV ↔ RGB conversion using BT.601 and BT.709 coefficient matrices
//! - HDR tone mapping: PQ (ST.2084) → SDR and HLG → SDR
//! - Alpha blending with pre-multiplied and straight alpha support

use crate::error::{AccelError, AccelResult};

// ─────────────────────────────────────────────────────────────────────────────
// YUV ↔ RGB
// ─────────────────────────────────────────────────────────────────────────────

/// YUV colour-space standard (selects the coefficient matrix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvStandard {
    /// ITU-R BT.601 — standard definition, used in MPEG-1/2/MJPEG.
    Bt601,
    /// ITU-R BT.709 — high definition, used in Blu-ray / most HD video.
    Bt709,
}

/// Range of luma/chroma sample values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvRange {
    /// "TV / limited" range: luma 16–235, chroma 16–240.
    Limited,
    /// "PC / full" range: luma and chroma 0–255.
    Full,
}

/// Convert a planar YUV420 buffer to interleaved RGB24.
///
/// # Arguments
/// * `yuv`    – Input buffer: Y plane followed by U plane then V plane
///              (each U/V plane is `(width/2) * (height/2)` bytes).
/// * `width`  – Image width in pixels (must be even).
/// * `height` – Image height in pixels (must be even).
/// * `std`    – BT.601 or BT.709 coefficient matrix.
/// * `range`  – Full or limited sample range.
///
/// # Errors
/// Returns [`AccelError::InvalidDimensions`] if the buffer length does not match.
pub fn yuv420_to_rgb(
    yuv: &[u8],
    width: u32,
    height: u32,
    std: YuvStandard,
    range: YuvRange,
) -> AccelResult<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let expected = y_size + 2 * uv_size;

    if yuv.len() != expected {
        return Err(AccelError::InvalidDimensions(format!(
            "yuv420_to_rgb: expected {expected} bytes ({}x{}), got {}",
            width,
            height,
            yuv.len()
        )));
    }

    let y_plane = &yuv[..y_size];
    let u_plane = &yuv[y_size..y_size + uv_size];
    let v_plane = &yuv[y_size + uv_size..];

    let mut rgb = vec![0u8; w * h * 3];

    for row in 0..h {
        for col in 0..w {
            let y_raw = y_plane[row * w + col];
            let uv_row = row / 2;
            let uv_col = col / 2;
            let u_raw = u_plane[uv_row * (w / 2) + uv_col];
            let v_raw = v_plane[uv_row * (w / 2) + uv_col];

            let (r, g, b) = yuv_to_rgb_pixel(y_raw, u_raw, v_raw, std, range);

            let out_idx = (row * w + col) * 3;
            rgb[out_idx] = r;
            rgb[out_idx + 1] = g;
            rgb[out_idx + 2] = b;
        }
    }

    Ok(rgb)
}

/// Convert interleaved RGB24 to planar YUV420.
///
/// # Arguments
/// * `rgb`    – Input buffer: interleaved R G B bytes, `width * height * 3` bytes total.
/// * `width`  – Image width in pixels (must be even).
/// * `height` – Image height in pixels (must be even).
/// * `std`    – BT.601 or BT.709 coefficient matrix.
/// * `range`  – Full or limited sample range.
///
/// # Errors
/// Returns [`AccelError::InvalidDimensions`] if the buffer length does not match.
pub fn rgb_to_yuv420(
    rgb: &[u8],
    width: u32,
    height: u32,
    std: YuvStandard,
    range: YuvRange,
) -> AccelResult<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * 3;

    if rgb.len() != expected {
        return Err(AccelError::InvalidDimensions(format!(
            "rgb_to_yuv420: expected {expected} bytes ({}x{}), got {}",
            width,
            height,
            rgb.len()
        )));
    }

    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let mut yuv = vec![0u8; y_size + 2 * uv_size];

    // Y plane
    for row in 0..h {
        for col in 0..w {
            let idx = (row * w + col) * 3;
            let r = rgb[idx];
            let g = rgb[idx + 1];
            let b = rgb[idx + 2];
            let (y, _u, _v) = rgb_to_yuv_pixel(r, g, b, std, range);
            yuv[row * w + col] = y;
        }
    }

    // U and V planes — average 2x2 block
    for uv_row in 0..(h / 2) {
        for uv_col in 0..(w / 2) {
            let mut u_sum: f32 = 0.0;
            let mut v_sum: f32 = 0.0;

            for dr in 0..2usize {
                for dc in 0..2usize {
                    let row = uv_row * 2 + dr;
                    let col = uv_col * 2 + dc;
                    let idx = (row * w + col) * 3;
                    let r = rgb[idx];
                    let g = rgb[idx + 1];
                    let b = rgb[idx + 2];
                    let (_y, u, v) = rgb_to_yuv_pixel(r, g, b, std, range);
                    u_sum += u as f32;
                    v_sum += v as f32;
                }
            }

            let uv_idx = uv_row * (w / 2) + uv_col;
            yuv[y_size + uv_idx] = (u_sum / 4.0).round() as u8;
            yuv[y_size + uv_size + uv_idx] = (v_sum / 4.0).round() as u8;
        }
    }

    Ok(yuv)
}

/// Single-pixel YUV → RGB conversion.
///
/// Returns `(R, G, B)` as `u8` values clamped to 0–255.
fn yuv_to_rgb_pixel(
    y_raw: u8,
    u_raw: u8,
    v_raw: u8,
    std: YuvStandard,
    range: YuvRange,
) -> (u8, u8, u8) {
    // Expand to float in 0..=1 or remove offsets
    let (y, u, v) = match range {
        YuvRange::Limited => {
            let y_f = (y_raw as f32 - 16.0) / 219.0;
            let u_f = (u_raw as f32 - 128.0) / 224.0;
            let v_f = (v_raw as f32 - 128.0) / 224.0;
            (y_f, u_f, v_f)
        }
        YuvRange::Full => {
            let y_f = y_raw as f32 / 255.0;
            let u_f = (u_raw as f32 - 128.0) / 255.0;
            let v_f = (v_raw as f32 - 128.0) / 255.0;
            (y_f, u_f, v_f)
        }
    };

    // BT.601 / BT.709 inverse matrices (Y, Cb, Cr → R, G, B)
    let (r, g, b) = match std {
        YuvStandard::Bt601 => {
            // BT.601 inverse
            let r = y + 1.402 * v;
            let g = y - 0.344_136 * u - 0.714_136 * v;
            let b = y + 1.772 * u;
            (r, g, b)
        }
        YuvStandard::Bt709 => {
            // BT.709 inverse
            let r = y + 1.574_800 * v;
            let g = y - 0.187_324 * u - 0.468_124 * v;
            let b = y + 1.855_600 * u;
            (r, g, b)
        }
    };

    (
        clamp_to_u8(r * 255.0),
        clamp_to_u8(g * 255.0),
        clamp_to_u8(b * 255.0),
    )
}

/// Single-pixel RGB → YUV conversion.
///
/// Returns `(Y, U, V)` as `u8` values.
fn rgb_to_yuv_pixel(r: u8, g: u8, b: u8, std: YuvStandard, range: YuvRange) -> (u8, u8, u8) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let (y, u, v) = match std {
        YuvStandard::Bt601 => {
            let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
            let u = -0.168_736 * rf - 0.331_264 * gf + 0.500 * bf;
            let v = 0.500 * rf - 0.418_688 * gf - 0.081_312 * bf;
            (y, u, v)
        }
        YuvStandard::Bt709 => {
            let y = 0.212_600 * rf + 0.715_200 * gf + 0.072_200 * bf;
            let u = -0.114_572 * rf - 0.385_428 * gf + 0.500 * bf;
            let v = 0.500 * rf - 0.454_153 * gf - 0.045_847 * bf;
            (y, u, v)
        }
    };

    match range {
        YuvRange::Limited => {
            let y_out = y * 219.0 + 16.0;
            let u_out = u * 224.0 + 128.0;
            let v_out = v * 224.0 + 128.0;
            (clamp_to_u8(y_out), clamp_to_u8(u_out), clamp_to_u8(v_out))
        }
        YuvRange::Full => {
            let y_out = y * 255.0;
            let u_out = u * 255.0 + 128.0;
            let v_out = v * 255.0 + 128.0;
            (clamp_to_u8(y_out), clamp_to_u8(u_out), clamp_to_u8(v_out))
        }
    }
}

#[inline]
fn clamp_to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

// ─────────────────────────────────────────────────────────────────────────────
// HDR Tone Mapping
// ─────────────────────────────────────────────────────────────────────────────

/// PQ (SMPTE ST.2084 / HDR10) inverse EOTF constants.
mod pq_constants {
    pub const M1: f64 = 0.158_515_463_379_877_85; // 2610 / (4096 * 4.0)
    pub const M2: f64 = 78.843_750; // 2523 / 4096 * 128
    pub const C1: f64 = 0.834_960_937_5; // 3424 / 4096
    pub const C2: f64 = 18.851_562_5; // 2413 / 4096 * 32
    pub const C3: f64 = 18.686_523_437_5; // 2392 / 4096 * 32
    /// Peak luminance for PQ signal (nits).
    pub const ST2084_PEAK: f64 = 10_000.0;
}

/// HLG OETF system gamma and related constants (ITU-R BT.2390).
mod hlg_constants {
    pub const A: f64 = 0.178_832_77;
    pub const B: f64 = 0.284_668_92;
    pub const C: f64 = 0.559_910_73;
    /// HLG reference display white luminance in nits.
    pub const REFERENCE_WHITE: f64 = 203.0;
    /// HLG system gamma (for 1000 nit reference display).
    pub const GAMMA: f64 = 1.2;
}

/// Apply the PQ inverse-EOTF to a single normalised signal value [0, 1],
/// returning scene linear luminance in nits.
fn pq_eotf(signal: f64) -> f64 {
    use pq_constants::*;
    if signal <= 0.0 {
        return 0.0;
    }
    let sp = signal.powf(1.0 / M2);
    let num = (sp - C1).max(0.0);
    let den = C2 - C3 * sp;
    if den <= 0.0 {
        return 0.0;
    }
    let linear = (num / den).powf(1.0 / M1);
    linear * ST2084_PEAK
}

/// Apply the HLG inverse-OETF to a normalised signal value [0, 1],
/// returning relative scene luminance (0–12 for HLG).
fn hlg_inverse_oetf(signal: f64) -> f64 {
    use hlg_constants::*;
    if signal <= 0.5 {
        // linear region
        (signal * signal) / 3.0
    } else {
        // gamma region
        ((signal - C) / A).exp() + B
    }
}

/// Simple Reinhard tone-mapping operator applied in linear light.
#[inline]
fn reinhard(l: f64, white_point: f64) -> f64 {
    l * (1.0 + l / (white_point * white_point)) / (1.0 + l)
}

/// Tone-map a single channel sample from PQ HDR to SDR [0, 1].
///
/// * `signal`     – PQ-encoded input in [0, 1].
/// * `peak_nits`  – The HDR mastering peak luminance (e.g. 1000.0, 4000.0, 10000.0).
///                  A higher value reduces the exposure of the SDR output.
fn pq_channel_to_sdr(signal: f64, peak_nits: f64) -> f64 {
    let linear_nits = pq_eotf(signal);
    // Normalise to 0..1 against the declared peak
    let normalised = linear_nits / peak_nits.max(1.0);
    // Apply Reinhard with white-point = 1.0 (after normalisation)
    reinhard(normalised, 1.0)
}

/// Tone-map a single channel sample from HLG to SDR [0, 1].
fn hlg_channel_to_sdr(signal: f64) -> f64 {
    use hlg_constants::*;
    // Convert HLG signal to scene linear
    let scene_linear = hlg_inverse_oetf(signal);
    // Apply system gamma to get display linear
    let display_linear = scene_linear.powf(GAMMA - 1.0) * scene_linear;
    // Normalise by reference white (203 nit → 1.0 SDR)
    let normalised = display_linear * REFERENCE_WHITE / 100.0;
    reinhard(normalised, 10.0)
}

/// Apply PQ (ST.2084 HDR10) → SDR tone mapping to an RGB24 image.
///
/// Each channel is assumed to be PQ-encoded in the range [0, 255].
/// The output is SDR RGB24 with BT.1886 gamma (γ ≈ 2.4 approximated
/// by a simple power function).
///
/// # Arguments
/// * `input`     – PQ-encoded RGB24 pixel data, length `width * height * 3`.
/// * `peak_nits` – Declared HDR peak luminance (e.g. 1000.0 for HDR10 grade).
///
/// # Errors
/// Returns [`AccelError::InvalidDimensions`] if `input.len()` is not a multiple of 3.
pub fn pq_to_sdr_tonemap(input: &[u8], peak_nits: f32) -> AccelResult<Vec<u8>> {
    if input.len() % 3 != 0 {
        return Err(AccelError::InvalidDimensions(
            "pq_to_sdr_tonemap: input length must be a multiple of 3".to_string(),
        ));
    }

    let peak = peak_nits as f64;
    let mut output = vec![0u8; input.len()];

    for (chunk_in, chunk_out) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        let r = pq_channel_to_sdr(chunk_in[0] as f64 / 255.0, peak);
        let g = pq_channel_to_sdr(chunk_in[1] as f64 / 255.0, peak);
        let b = pq_channel_to_sdr(chunk_in[2] as f64 / 255.0, peak);

        // BT.1886 gamma ≈ 2.4 (linearise then re-encode)
        chunk_out[0] = clamp_to_u8((r.powf(1.0 / 2.4) * 255.0) as f32);
        chunk_out[1] = clamp_to_u8((g.powf(1.0 / 2.4) * 255.0) as f32);
        chunk_out[2] = clamp_to_u8((b.powf(1.0 / 2.4) * 255.0) as f32);
    }

    Ok(output)
}

/// Apply HLG (ITU-R BT.2100) → SDR tone mapping to an RGB24 image.
///
/// Each channel is HLG-OETF-encoded in [0, 255].
/// The output is SDR RGB24 with sRGB gamma.
///
/// # Errors
/// Returns [`AccelError::InvalidDimensions`] if `input.len()` is not a multiple of 3.
pub fn hlg_to_sdr_tonemap(input: &[u8]) -> AccelResult<Vec<u8>> {
    if input.len() % 3 != 0 {
        return Err(AccelError::InvalidDimensions(
            "hlg_to_sdr_tonemap: input length must be a multiple of 3".to_string(),
        ));
    }

    let mut output = vec![0u8; input.len()];

    for (chunk_in, chunk_out) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        let r = hlg_channel_to_sdr(chunk_in[0] as f64 / 255.0);
        let g = hlg_channel_to_sdr(chunk_in[1] as f64 / 255.0);
        let b = hlg_channel_to_sdr(chunk_in[2] as f64 / 255.0);

        // sRGB gamma encode
        chunk_out[0] = clamp_to_u8((srgb_encode(r) * 255.0) as f32);
        chunk_out[1] = clamp_to_u8((srgb_encode(g) * 255.0) as f32);
        chunk_out[2] = clamp_to_u8((srgb_encode(b) * 255.0) as f32);
    }

    Ok(output)
}

/// sRGB gamma encoding function.
#[inline]
fn srgb_encode(linear: f64) -> f64 {
    let c = linear.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Alpha Blending
// ─────────────────────────────────────────────────────────────────────────────

/// Blend `src` over `dst` with a uniform `alpha` factor (straight alpha).
///
/// Formula: `out[i] = src[i] * alpha + dst[i] * (1 - alpha)`
///
/// Both `src` and `dst` must have the same length (RGBA or RGB — any byte
/// sequence is accepted; blending is per-byte).
///
/// # Arguments
/// * `src`   – Source pixel data.
/// * `dst`   – Destination pixel data.
/// * `alpha` – Blend factor in [0.0, 1.0]. 0 = fully dst, 1 = fully src.
///
/// # Errors
/// Returns [`AccelError::InvalidDimensions`] if `src` and `dst` differ in length.
pub fn alpha_blend(src: &[u8], dst: &[u8], alpha: f32) -> AccelResult<Vec<u8>> {
    if src.len() != dst.len() {
        return Err(AccelError::InvalidDimensions(format!(
            "alpha_blend: src ({}) and dst ({}) must have the same length",
            src.len(),
            dst.len()
        )));
    }

    let a = alpha.clamp(0.0, 1.0);
    let ia = 1.0 - a;

    let output = src
        .iter()
        .zip(dst.iter())
        .map(|(&s, &d)| {
            let blended = s as f32 * a + d as f32 * ia;
            blended.round().clamp(0.0, 255.0) as u8
        })
        .collect();

    Ok(output)
}

/// Blend `src` (with embedded per-pixel alpha in 4th byte) over `dst` (RGBA).
///
/// Uses the standard Porter-Duff "src over dst" compositing formula:
/// ```text
/// alpha_s = src[3] / 255.0
/// out_rgb = src_rgb * alpha_s + dst_rgb * (1 - alpha_s)
/// out_a   = alpha_s + dst_a * (1 - alpha_s)
/// ```
///
/// # Errors
/// Returns [`AccelError::InvalidDimensions`] if either buffer is not a multiple
/// of 4 or the lengths differ.
pub fn alpha_blend_rgba(src: &[u8], dst: &[u8]) -> AccelResult<Vec<u8>> {
    if src.len() % 4 != 0 || dst.len() % 4 != 0 {
        return Err(AccelError::InvalidDimensions(
            "alpha_blend_rgba: buffers must be multiples of 4 bytes".to_string(),
        ));
    }
    if src.len() != dst.len() {
        return Err(AccelError::InvalidDimensions(format!(
            "alpha_blend_rgba: src ({}) and dst ({}) must have the same length",
            src.len(),
            dst.len()
        )));
    }

    let mut output = vec![0u8; src.len()];

    for (i, (s_chunk, d_chunk)) in src.chunks_exact(4).zip(dst.chunks_exact(4)).enumerate() {
        let alpha_s = s_chunk[3] as f32 / 255.0;
        let alpha_d = d_chunk[3] as f32 / 255.0;
        let ia = 1.0 - alpha_s;

        let out_idx = i * 4;
        output[out_idx] = clamp_to_u8(s_chunk[0] as f32 * alpha_s + d_chunk[0] as f32 * ia);
        output[out_idx + 1] = clamp_to_u8(s_chunk[1] as f32 * alpha_s + d_chunk[1] as f32 * ia);
        output[out_idx + 2] = clamp_to_u8(s_chunk[2] as f32 * alpha_s + d_chunk[2] as f32 * ia);
        // Porter-Duff alpha composite
        let out_a = alpha_s + alpha_d * ia;
        output[out_idx + 3] = clamp_to_u8(out_a * 255.0);
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── YUV ↔ RGB round-trip ────────────────────────────────────────────────

    fn make_solid_rgb(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 3) as usize];
        for chunk in buf.chunks_exact_mut(3) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
        buf
    }

    #[test]
    fn test_yuv_rgb_roundtrip_bt601_full_grey() {
        // Grey pixel: R=G=B=128 → Y≈128, U=V≈128 → R≈G≈B≈128
        let w = 2u32;
        let h = 2u32;
        let rgb_in = make_solid_rgb(128, 128, 128, w, h);
        let yuv = rgb_to_yuv420(&rgb_in, w, h, YuvStandard::Bt601, YuvRange::Full)
            .expect("encode failed");
        let rgb_out =
            yuv420_to_rgb(&yuv, w, h, YuvStandard::Bt601, YuvRange::Full).expect("decode failed");

        for i in 0..rgb_in.len() {
            let diff = (rgb_in[i] as i32 - rgb_out[i] as i32).abs();
            assert!(
                diff <= 2,
                "channel {i}: expected ~{}, got {}",
                rgb_in[i],
                rgb_out[i]
            );
        }
    }

    #[test]
    fn test_yuv_rgb_roundtrip_bt709_limited_red() {
        let w = 4u32;
        let h = 4u32;
        let rgb_in = make_solid_rgb(200, 50, 50, w, h);
        let yuv =
            rgb_to_yuv420(&rgb_in, w, h, YuvStandard::Bt709, YuvRange::Limited).expect("encode");
        let rgb_out =
            yuv420_to_rgb(&yuv, w, h, YuvStandard::Bt709, YuvRange::Limited).expect("decode");

        for i in 0..rgb_in.len() {
            let diff = (rgb_in[i] as i32 - rgb_out[i] as i32).abs();
            assert!(
                diff <= 4,
                "channel {i}: expected ~{}, got {}",
                rgb_in[i],
                rgb_out[i]
            );
        }
    }

    #[test]
    fn test_yuv_rgb_roundtrip_bt601_limited_green() {
        let w = 2u32;
        let h = 2u32;
        let rgb_in = make_solid_rgb(20, 200, 30, w, h);
        let yuv =
            rgb_to_yuv420(&rgb_in, w, h, YuvStandard::Bt601, YuvRange::Limited).expect("encode");
        let rgb_out =
            yuv420_to_rgb(&yuv, w, h, YuvStandard::Bt601, YuvRange::Limited).expect("decode");

        for i in 0..rgb_in.len() {
            let diff = (rgb_in[i] as i32 - rgb_out[i] as i32).abs();
            assert!(
                diff <= 4,
                "channel {i}: expected ~{}, got {}",
                rgb_in[i],
                rgb_out[i]
            );
        }
    }

    #[test]
    fn test_yuv420_invalid_buffer() {
        let bad = vec![0u8; 10];
        let result = yuv420_to_rgb(&bad, 4, 4, YuvStandard::Bt601, YuvRange::Full);
        assert!(result.is_err());
    }

    #[test]
    fn test_rgb_to_yuv420_invalid_buffer() {
        let bad = vec![0u8; 10];
        let result = rgb_to_yuv420(&bad, 4, 4, YuvStandard::Bt601, YuvRange::Full);
        assert!(result.is_err());
    }

    // ── HDR Tone Mapping ─────────────────────────────────────────────────────

    #[test]
    fn test_pq_to_sdr_black_stays_black() {
        let input = vec![0u8, 0, 0, 0, 0, 0];
        let out = pq_to_sdr_tonemap(&input, 1000.0).expect("pq tonemap");
        for &v in &out {
            assert_eq!(v, 0, "black input should stay black");
        }
    }

    #[test]
    fn test_pq_to_sdr_white_maps_to_high() {
        // Full PQ signal (255, 255, 255) at 1000 nit peak
        let input = vec![255u8, 255, 255];
        let out = pq_to_sdr_tonemap(&input, 1000.0).expect("pq tonemap");
        for &v in &out {
            // White should be high but may be compressed; must be > 200
            assert!(v > 200, "full PQ white should map to bright SDR, got {v}");
        }
    }

    #[test]
    fn test_pq_to_sdr_invalid_length() {
        let input = vec![1u8, 2]; // not a multiple of 3
        assert!(pq_to_sdr_tonemap(&input, 1000.0).is_err());
    }

    #[test]
    fn test_pq_to_sdr_monotonic() {
        // A brighter PQ input should yield a brighter SDR output (same channel)
        let dark = vec![100u8, 0, 0];
        let bright = vec![200u8, 0, 0];
        let out_dark = pq_to_sdr_tonemap(&dark, 1000.0).expect("dark");
        let out_bright = pq_to_sdr_tonemap(&bright, 1000.0).expect("bright");
        assert!(out_bright[0] >= out_dark[0], "tone map must be monotonic");
    }

    #[test]
    fn test_hlg_to_sdr_black_stays_black() {
        let input = vec![0u8, 0, 0];
        let out = hlg_to_sdr_tonemap(&input).expect("hlg tonemap");
        for &v in &out {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_hlg_to_sdr_white_maps_nonnegative() {
        let input = vec![255u8, 255, 255];
        let out = hlg_to_sdr_tonemap(&input).expect("hlg tonemap");
        for &v in &out {
            assert!(v > 0, "HLG white should map to a positive SDR value");
        }
    }

    #[test]
    fn test_hlg_to_sdr_invalid_length() {
        let input = vec![1u8, 2]; // not a multiple of 3
        assert!(hlg_to_sdr_tonemap(&input).is_err());
    }

    #[test]
    fn test_hlg_to_sdr_monotonic() {
        let dark = vec![50u8, 0, 0];
        let bright = vec![200u8, 0, 0];
        let out_dark = hlg_to_sdr_tonemap(&dark).expect("dark");
        let out_bright = hlg_to_sdr_tonemap(&bright).expect("bright");
        assert!(
            out_bright[0] >= out_dark[0],
            "HLG tonemap must be monotonic"
        );
    }

    // ── Alpha Blending ───────────────────────────────────────────────────────

    #[test]
    fn test_alpha_blend_fully_src() {
        let src = vec![200u8, 100, 50];
        let dst = vec![10u8, 20, 30];
        let out = alpha_blend(&src, &dst, 1.0).expect("blend");
        assert_eq!(out, src, "alpha=1.0 must return src");
    }

    #[test]
    fn test_alpha_blend_fully_dst() {
        let src = vec![200u8, 100, 50];
        let dst = vec![10u8, 20, 30];
        let out = alpha_blend(&src, &dst, 0.0).expect("blend");
        assert_eq!(out, dst, "alpha=0.0 must return dst");
    }

    #[test]
    fn test_alpha_blend_midpoint() {
        let src = vec![200u8];
        let dst = vec![100u8];
        let out = alpha_blend(&src, &dst, 0.5).expect("blend");
        // Expected: 200*0.5 + 100*0.5 = 150
        assert_eq!(out[0], 150);
    }

    #[test]
    fn test_alpha_blend_length_mismatch() {
        let src = vec![1u8, 2, 3];
        let dst = vec![1u8, 2];
        assert!(alpha_blend(&src, &dst, 0.5).is_err());
    }

    #[test]
    fn test_alpha_blend_clamped() {
        // alpha > 1 gets clamped to 1
        let src = vec![200u8];
        let dst = vec![100u8];
        let out = alpha_blend(&src, &dst, 2.0).expect("blend");
        assert_eq!(out[0], 200, "clamped alpha=1 → src");
    }

    #[test]
    fn test_alpha_blend_rgba_over() {
        // Fully opaque src (alpha=255) over any dst → output = src
        let src = vec![100u8, 150, 200, 255];
        let dst = vec![50u8, 60, 70, 255];
        let out = alpha_blend_rgba(&src, &dst).expect("rgba blend");
        assert_eq!(&out[..3], &[100u8, 150, 200]);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn test_alpha_blend_rgba_transparent_src() {
        // Fully transparent src (alpha=0) → output = dst
        let src = vec![100u8, 150, 200, 0];
        let dst = vec![50u8, 60, 70, 255];
        let out = alpha_blend_rgba(&src, &dst).expect("rgba blend");
        assert_eq!(&out[..3], &[50u8, 60, 70]);
    }

    #[test]
    fn test_alpha_blend_rgba_invalid_length() {
        // Not multiple of 4
        let src = vec![1u8, 2, 3];
        let dst = vec![1u8, 2, 3];
        assert!(alpha_blend_rgba(&src, &dst).is_err());
    }

    #[test]
    fn test_alpha_blend_rgba_mismatched_lengths() {
        let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let dst = vec![1u8, 2, 3, 4];
        assert!(alpha_blend_rgba(&src, &dst).is_err());
    }
}
