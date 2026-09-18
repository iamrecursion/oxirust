//! High-quality image resampling filters for 360° VR pixel lookups.
//!
//! Provides bicubic (Mitchell-Netravali) and Lanczos resampling alongside the
//! existing bilinear sampler.  All samplers operate in normalised UV coordinates
//! and support three pixel formats:
//!
//! * [`sample_u8`]   — 8-bit unsigned integer channels (most common)
//! * [`sample_u16`]  — 16-bit unsigned integer channels (HDR / RAW pipelines)
//! * [`sample_f32`]  — 32-bit floating-point channels (linear-light compositing)
//!
//! Sampling quality is selected via the [`FilterKernel`] enum.

#![allow(unsafe_code)]

use crate::VrError;

// ─── Filter kernel selector ──────────────────────────────────────────────────

/// Resampling filter kernel used by the typed samplers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKernel {
    /// Bilinear (2-tap) — fast, mild blur.
    Bilinear,
    /// Mitchell-Netravali bicubic (4-tap × 4-tap, B=1/3, C=1/3) — balanced
    /// sharpness/ringing trade-off, good for downscaling.
    Bicubic,
    /// Lanczos with lobes = 2 (4-tap × 4-tap) — sharper than bicubic,
    /// slight ringing at very high contrast edges.
    Lanczos2,
    /// Lanczos with lobes = 3 (6-tap × 6-tap) — standard high-quality choice
    /// for format conversions.
    Lanczos3,
}

// ─── Pixel format trait ──────────────────────────────────────────────────────

/// A scalar pixel component type that can be reconstructed from a weighted sum.
///
/// This sealed trait is implemented for `u8`, `u16`, and `f32`.
pub trait PixelComponent: Copy + Into<f64> + Sized + private::Sealed {
    /// Clamp and convert a weighted accumulator value back to `Self`.
    fn from_f64_clamped(v: f64) -> Self;
    /// The value representing black / zero.
    fn zero() -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
}

impl PixelComponent for u8 {
    fn from_f64_clamped(v: f64) -> Self {
        v.round().clamp(0.0, 255.0) as u8
    }
    fn zero() -> Self {
        0u8
    }
}

impl PixelComponent for u16 {
    fn from_f64_clamped(v: f64) -> Self {
        v.round().clamp(0.0, 65535.0) as u16
    }
    fn zero() -> Self {
        0u16
    }
}

impl PixelComponent for f32 {
    fn from_f64_clamped(v: f64) -> Self {
        v as f32
    }
    fn zero() -> Self {
        0.0f32
    }
}

// ─── Public typed samplers ────────────────────────────────────────────────────

/// Sample an 8-bit image at normalised UV coordinates using the specified filter.
///
/// * `data`     — packed row-major pixel data (channels interleaved)
/// * `w`, `h`   — image dimensions in pixels
/// * `u`, `v`   — normalised sampling coordinates (0..1)
/// * `channels` — number of colour channels per pixel (e.g. 3 for RGB)
/// * `kernel`   — resampling filter to use
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if `w` or `h` is zero.
/// Returns [`VrError::BufferTooSmall`] if `data` is too small.
pub fn sample_u8(
    data: &[u8],
    w: u32,
    h: u32,
    u: f32,
    v: f32,
    channels: u32,
    kernel: FilterKernel,
) -> Result<Vec<u8>, VrError> {
    validate_buffer(data, w, h, channels, std::mem::size_of::<u8>())?;
    Ok(dispatch_sample::<u8>(
        data_as_u8_slice(data),
        w,
        h,
        u,
        v,
        channels,
        kernel,
    ))
}

/// Sample a 16-bit image at normalised UV coordinates using the specified filter.
///
/// `data` must be in native-endian order (one `u16` per component, packed).
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if `w` or `h` is zero.
/// Returns [`VrError::BufferTooSmall`] if `data` is too small.
pub fn sample_u16(
    data: &[u16],
    w: u32,
    h: u32,
    u: f32,
    v: f32,
    channels: u32,
    kernel: FilterKernel,
) -> Result<Vec<u16>, VrError> {
    let required = w as usize * h as usize * channels as usize;
    if w == 0 || h == 0 {
        return Err(VrError::InvalidDimensions(
            "image width and height must be > 0".into(),
        ));
    }
    if data.len() < required {
        return Err(VrError::BufferTooSmall {
            expected: required * 2,
            got: data.len() * 2,
        });
    }
    Ok(dispatch_sample::<u16>(data, w, h, u, v, channels, kernel))
}

/// Sample an f32 image at normalised UV coordinates using the specified filter.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if `w` or `h` is zero.
/// Returns [`VrError::BufferTooSmall`] if `data` is too small.
pub fn sample_f32(
    data: &[f32],
    w: u32,
    h: u32,
    u: f32,
    v: f32,
    channels: u32,
    kernel: FilterKernel,
) -> Result<Vec<f32>, VrError> {
    let required = w as usize * h as usize * channels as usize;
    if w == 0 || h == 0 {
        return Err(VrError::InvalidDimensions(
            "image width and height must be > 0".into(),
        ));
    }
    if data.len() < required {
        return Err(VrError::BufferTooSmall {
            expected: required * 4,
            got: data.len() * 4,
        });
    }
    Ok(dispatch_sample::<f32>(data, w, h, u, v, channels, kernel))
}

// ─── Dispatch helper ──────────────────────────────────────────────────────────

fn dispatch_sample<T: PixelComponent>(
    data: &[T],
    w: u32,
    h: u32,
    u: f32,
    v: f32,
    channels: u32,
    kernel: FilterKernel,
) -> Vec<T> {
    match kernel {
        FilterKernel::Bilinear => bilinear_generic(data, w, h, u, v, channels),
        FilterKernel::Bicubic => bicubic_generic(data, w, h, u, v, channels),
        FilterKernel::Lanczos2 => lanczos_generic(data, w, h, u, v, channels, 2),
        FilterKernel::Lanczos3 => lanczos_generic(data, w, h, u, v, channels, 3),
    }
}

// ─── Bilinear (generic) ───────────────────────────────────────────────────────

fn bilinear_generic<T: PixelComponent>(
    data: &[T],
    w: u32,
    h: u32,
    u: f32,
    v: f32,
    channels: u32,
) -> Vec<T> {
    let ch = channels as usize;
    let fw = w as f64;
    let fh = h as f64;

    let px = ((u as f64) * fw - 0.5).max(0.0);
    let py = ((v as f64) * fh - 0.5).max(0.0);

    let x0 = (px.floor() as u32).min(w.saturating_sub(1));
    let y0 = (py.floor() as u32).min(h.saturating_sub(1));
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));

    let tx = px - px.floor();
    let ty = py - py.floor();

    let stride = w as usize * ch;
    let b00 = y0 as usize * stride + x0 as usize * ch;
    let b10 = y0 as usize * stride + x1 as usize * ch;
    let b01 = y1 as usize * stride + x0 as usize * ch;
    let b11 = y1 as usize * stride + x1 as usize * ch;

    let mut out = vec![T::zero(); ch];
    for c in 0..ch {
        let p00: f64 = data[b00 + c].into();
        let p10: f64 = data[b10 + c].into();
        let p01: f64 = data[b01 + c].into();
        let p11: f64 = data[b11 + c].into();
        let top = p00 + (p10 - p00) * tx;
        let bottom = p01 + (p11 - p01) * tx;
        out[c] = T::from_f64_clamped(top + (bottom - top) * ty);
    }
    out
}

// ─── Mitchell-Netravali bicubic ───────────────────────────────────────────────
//
// Parameters B = 1/3, C = 1/3 (Mitchell-Netravali "balanced" point).
// Kernel support: [-2, +2].

fn mitchell_netravali(x: f64, b: f64, c: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
            + (-18.0 + 12.0 * b + 6.0 * c) * x * x
            + (6.0 - 2.0 * b))
            / 6.0
    } else if x < 2.0 {
        ((-b - 6.0 * c) * x * x * x
            + (6.0 * b + 30.0 * c) * x * x
            + (-12.0 * b - 48.0 * c) * x
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

fn bicubic_generic<T: PixelComponent>(
    data: &[T],
    w: u32,
    h: u32,
    u: f32,
    v: f32,
    channels: u32,
) -> Vec<T> {
    const B: f64 = 1.0 / 3.0;
    const C: f64 = 1.0 / 3.0;

    let ch = channels as usize;
    let fw = w as f64;
    let fh = h as f64;

    // Sub-pixel position
    let px = (u as f64) * fw - 0.5;
    let py = (v as f64) * fh - 0.5;

    let x_floor = px.floor() as i64;
    let y_floor = py.floor() as i64;
    let fx = px - px.floor();
    let fy = py - py.floor();

    let stride = w as usize * ch;
    let wi = w as i64;
    let hi = h as i64;

    // Precompute per-axis weights for taps at offsets [-1, 0, 1, 2]
    let wx: [f64; 4] = [
        mitchell_netravali(fx + 1.0, B, C),
        mitchell_netravali(fx, B, C),
        mitchell_netravali(fx - 1.0, B, C),
        mitchell_netravali(fx - 2.0, B, C),
    ];
    let wy: [f64; 4] = [
        mitchell_netravali(fy + 1.0, B, C),
        mitchell_netravali(fy, B, C),
        mitchell_netravali(fy - 1.0, B, C),
        mitchell_netravali(fy - 2.0, B, C),
    ];

    let mut out = vec![T::zero(); ch];
    let mut acc = vec![0.0f64; ch];
    let mut weight_sum = 0.0f64;

    for (ky, &wy_k) in wy.iter().enumerate() {
        let sy = (y_floor - 1 + ky as i64).clamp(0, hi - 1) as usize;
        for (kx, &wx_k) in wx.iter().enumerate() {
            let sx = (x_floor - 1 + kx as i64).clamp(0, wi - 1) as usize;
            let w_total = wy_k * wx_k;
            let base = sy * stride + sx * ch;
            for c in 0..ch {
                let v_f: f64 = data[base + c].into();
                acc[c] += v_f * w_total;
            }
            weight_sum += w_total;
        }
    }

    // Normalise to guard against floating-point drift at image edges
    if weight_sum.abs() > 1e-12 {
        for c in 0..ch {
            out[c] = T::from_f64_clamped(acc[c] / weight_sum);
        }
    }
    out
}

// ─── Lanczos ─────────────────────────────────────────────────────────────────
//
// lanczos(x, a) = sinc(x) * sinc(x/a)  for |x| < a, 0 otherwise
// where sinc(x) = sin(π x) / (π x)  and  sinc(0) = 1.

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn lanczos_weight(x: f64, a: i32) -> f64 {
    let af = a as f64;
    if x.abs() < af {
        sinc(x) * sinc(x / af)
    } else {
        0.0
    }
}

fn lanczos_generic<T: PixelComponent>(
    data: &[T],
    w: u32,
    h: u32,
    u: f32,
    v: f32,
    channels: u32,
    lobes: i32,
) -> Vec<T> {
    let ch = channels as usize;
    let fw = w as f64;
    let fh = h as f64;

    let px = (u as f64) * fw - 0.5;
    let py = (v as f64) * fh - 0.5;

    let x_floor = px.floor() as i64;
    let y_floor = py.floor() as i64;
    let fx = px - px.floor();
    let fy = py - py.floor();

    let stride = w as usize * ch;
    let wi = w as i64;
    let hi = h as i64;

    // Tap range: [-(lobes-1), lobes]
    let lo = -(lobes as i64 - 1);
    let hi_tap = lobes as i64;

    let tap_count = (hi_tap - lo) as usize;

    let wx: Vec<f64> = (lo..hi_tap)
        .map(|k| lanczos_weight(fx - k as f64, lobes))
        .collect();
    let wy: Vec<f64> = (lo..hi_tap)
        .map(|k| lanczos_weight(fy - k as f64, lobes))
        .collect();

    let mut acc = vec![0.0f64; ch];
    let mut weight_sum = 0.0f64;

    for (ky, &wy_k) in wy.iter().enumerate() {
        let sy = (y_floor + lo + ky as i64).clamp(0, hi - 1) as usize;
        for (kx, &wx_k) in wx.iter().enumerate() {
            let sx = (x_floor + lo + kx as i64).clamp(0, wi - 1) as usize;
            let w_total = wy_k * wx_k;
            let base = sy * stride + sx * ch;
            for c in 0..ch {
                let v_f: f64 = data[base + c].into();
                acc[c] += v_f * w_total;
            }
            weight_sum += w_total;
        }
    }

    let _ = tap_count; // used implicitly via ranges above
    let mut out = vec![T::zero(); ch];
    if weight_sum.abs() > 1e-12 {
        for c in 0..ch {
            out[c] = T::from_f64_clamped(acc[c] / weight_sum);
        }
    }
    out
}

// ─── Keys cubic bicubic convenience sampler ───────────────────────────────────
//
// Uses the Keys cubic kernel with parameter a = -0.5:
//   w(t) = (a+2)|t|³ - (a+3)|t|² + 1          for |t| ≤ 1
//   w(t) = a|t|³ - 5a|t|² + 8a|t| - 4a         for 1 < |t| < 2
//   w(t) = 0                                    otherwise
//
// This is a low-level convenience function that returns a fixed-size `[u8; 3]`
// array, making it easy to use directly in projection inner loops without
// error-handling overhead.  The image is expected to be RGB (3 channels, 3 bpp
// row-major).  The coordinates `x` and `y` are in *pixel* space (not normalised
// UV), with `x ∈ [0, w)` and `y ∈ [0, h)`.

const KEYS_A: f64 = -0.5;

/// Evaluate the Keys cubic kernel weight for distance `t`.
#[inline]
fn keys_cubic(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        (KEYS_A + 2.0) * t * t * t - (KEYS_A + 3.0) * t * t + 1.0
    } else if t < 2.0 {
        KEYS_A * t * t * t - 5.0 * KEYS_A * t * t + 8.0 * KEYS_A * t - 4.0 * KEYS_A
    } else {
        0.0
    }
}

/// Sample an RGB (`u8`, 3-channel, row-major) image at pixel coordinate
/// `(x, y)` using the Keys bicubic kernel (`a = -0.5`).
///
/// * `src` — packed RGB pixel data (`width × height × 3` bytes)
/// * `w`, `h` — image dimensions in pixels
/// * `x`, `y` — pixel-space sampling position (0 … w/h, may be fractional)
///
/// The function is infallible: out-of-range taps are clamped to the image
/// border (border-replicate strategy).  If `w == 0` or `h == 0` a black pixel
/// `[0, 0, 0]` is returned.
pub fn sample_bicubic(src: &[u8], w: u32, h: u32, x: f64, y: f64) -> [u8; 3] {
    if w == 0 || h == 0 {
        return [0, 0, 0];
    }

    let wi = w as i64;
    let hi = h as i64;
    let stride = w as usize * 3;

    let x_floor = x.floor() as i64;
    let y_floor = y.floor() as i64;
    let fx = x - x.floor(); // sub-pixel offset in [0, 1)
    let fy = y - y.floor();

    // Precompute per-axis weights for 4 taps at offsets -1, 0, 1, 2
    let wx = [
        keys_cubic(fx + 1.0),
        keys_cubic(fx),
        keys_cubic(fx - 1.0),
        keys_cubic(fx - 2.0),
    ];
    let wy = [
        keys_cubic(fy + 1.0),
        keys_cubic(fy),
        keys_cubic(fy - 1.0),
        keys_cubic(fy - 2.0),
    ];

    let mut acc = [0.0f64; 3];
    let mut weight_sum = 0.0f64;

    for (ky, &wy_k) in wy.iter().enumerate() {
        let sy = (y_floor - 1 + ky as i64).clamp(0, hi - 1) as usize;
        for (kx, &wx_k) in wx.iter().enumerate() {
            let sx = (x_floor - 1 + kx as i64).clamp(0, wi - 1) as usize;
            let w_total = wy_k * wx_k;
            let base = sy * stride + sx * 3;
            for c in 0..3 {
                acc[c] += src[base + c] as f64 * w_total;
            }
            weight_sum += w_total;
        }
    }

    if weight_sum.abs() < 1e-12 {
        return [0, 0, 0];
    }

    [
        (acc[0] / weight_sum).round().clamp(0.0, 255.0) as u8,
        (acc[1] / weight_sum).round().clamp(0.0, 255.0) as u8,
        (acc[2] / weight_sum).round().clamp(0.0, 255.0) as u8,
    ]
}

// ─── SIMD batch bilinear sampling ────────────────────────────────────────────

/// Scalar bilinear sample: single-channel u8 image at pixel-space `(u, v)`.
///
/// Coordinates are in pixel space (0 … width-1, 0 … height-1) with sub-pixel
/// precision. Taps are clamped to the image border.
#[inline]
pub fn bilinear_sample_scalar(img: &[u8], width: u32, height: u32, u: f32, v: f32) -> u8 {
    let px = u.max(0.0);
    let py = v.max(0.0);
    let x0 = (px.floor() as u32).min(width.saturating_sub(1));
    let y0 = (py.floor() as u32).min(height.saturating_sub(1));
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = px - px.floor();
    let ty = py - py.floor();
    let stride = width as usize;
    let p00 = img[y0 as usize * stride + x0 as usize] as f32;
    let p10 = img[y0 as usize * stride + x1 as usize] as f32;
    let p01 = img[y1 as usize * stride + x0 as usize] as f32;
    let p11 = img[y1 as usize * stride + x1 as usize] as f32;
    let top = p00 + (p10 - p00) * tx;
    let bottom = p01 + (p11 - p01) * tx;
    (top + (bottom - top) * ty).round().clamp(0.0, 255.0) as u8
}

/// Scalar batch bilinear sampling: processes all `coords` one at a time.
///
/// `coords` are in pixel space `(u, v)` — fractional values are allowed.
/// `out` must be at least `coords.len()` bytes long.
pub fn sample_bilinear_scalar(
    img: &[u8],
    width: u32,
    height: u32,
    _channels: usize,
    coords: &[(f32, f32)],
    out: &mut [u8],
) {
    for (i, &(u, v)) in coords.iter().enumerate() {
        out[i] = bilinear_sample_scalar(img, width, height, u, v);
    }
}

/// AVX2+FMA path: 4× unrolled scalar inside a target-feature unsafe fn so the
/// compiler can use SIMD registers and instruction selection freely.
///
/// # Safety
/// The caller must have confirmed that AVX2 and FMA are both available via
/// `is_x86_feature_detected!` before calling this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
unsafe fn sample_bilinear_avx2(
    img: &[u8],
    width: u32,
    height: u32,
    _channels: usize,
    coords: &[(f32, f32)],
    out: &mut [u8],
) {
    // Process 4 coordinates at a time (unrolled so the compiler can vectorise).
    let chunks = coords.chunks_exact(4);
    let remainder = chunks.remainder();
    let base_idx = coords.len() - remainder.len();

    for (i, chunk) in chunks.enumerate() {
        // Unroll 4 bilinear samples — the target_feature constraint lets the
        // compiler hoist loads/multiplies into AVX2 registers automatically.
        let s0 = bilinear_sample_scalar(img, width, height, chunk[0].0, chunk[0].1);
        let s1 = bilinear_sample_scalar(img, width, height, chunk[1].0, chunk[1].1);
        let s2 = bilinear_sample_scalar(img, width, height, chunk[2].0, chunk[2].1);
        let s3 = bilinear_sample_scalar(img, width, height, chunk[3].0, chunk[3].1);
        let base = i * 4;
        out[base] = s0;
        out[base + 1] = s1;
        out[base + 2] = s2;
        out[base + 3] = s3;
    }
    // Handle remainder with plain scalar.
    for (i, &(u, v)) in remainder.iter().enumerate() {
        out[base_idx + i] = bilinear_sample_scalar(img, width, height, u, v);
    }
}

/// SSE4.1 path: 2× unrolled scalar inside a target-feature unsafe fn.
///
/// # Safety
/// The caller must have confirmed that SSE4.1 is available via
/// `is_x86_feature_detected!` before calling this function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
unsafe fn sample_bilinear_sse41(
    img: &[u8],
    width: u32,
    height: u32,
    _channels: usize,
    coords: &[(f32, f32)],
    out: &mut [u8],
) {
    let chunks = coords.chunks_exact(2);
    let remainder = chunks.remainder();
    let base_idx = coords.len() - remainder.len();

    for (i, chunk) in chunks.enumerate() {
        let s0 = bilinear_sample_scalar(img, width, height, chunk[0].0, chunk[0].1);
        let s1 = bilinear_sample_scalar(img, width, height, chunk[1].0, chunk[1].1);
        let base = i * 2;
        out[base] = s0;
        out[base + 1] = s1;
    }
    for (i, &(u, v)) in remainder.iter().enumerate() {
        out[base_idx + i] = bilinear_sample_scalar(img, width, height, u, v);
    }
}

/// SIMD-accelerated batch bilinear sampling with runtime dispatch.
///
/// Processes `coords.len()` single-channel u8 lookups. Dispatches to AVX2+FMA
/// (4× unrolled), SSE4.1 (2× unrolled), or scalar fallback depending on runtime
/// CPU feature detection. The safe wrapper guarantees no unsafe code is reached
/// without prior feature detection.
///
/// # Arguments
/// * `img`      — packed row-major single-channel u8 image
/// * `width`    — image width in pixels
/// * `height`   — image height in pixels
/// * `_channels`— number of colour channels (currently single-channel; reserved)
/// * `coords`   — pixel-space sampling coordinates `(u, v)`, may be fractional
/// * `out`      — output buffer; must be at least `coords.len()` bytes
pub fn sample_bilinear_batch(
    img: &[u8],
    width: u32,
    height: u32,
    channels: usize,
    coords: &[(f32, f32)],
    out: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability confirmed at runtime above.
            unsafe {
                return sample_bilinear_avx2(img, width, height, channels, coords, out);
            }
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: SSE4.1 availability confirmed at runtime above.
            unsafe {
                return sample_bilinear_sse41(img, width, height, channels, coords, out);
            }
        }
    }
    sample_bilinear_scalar(img, width, height, channels, coords, out);
}

// ─── Fast-math trig approximations ───────────────────────────────────────────
//
// These are gated behind the `fast_math` Cargo feature.  They trade a small
// amount of accuracy for significantly faster execution in hot projection loops
// where the full-precision libm functions are the bottleneck.
//
// Accuracy targets (confirmed by unit tests):
//   fast_atan2 — error ≤ 0.01 rad (≈ 0.6°) over the full circle
//   fast_sin   — error ≤ 2e-3   over [−π, π]   (Bhaskara I, max err ≈ 1.3e-3)
//   fast_cos   — same bound via π/2 shift

/// Hastings-style atan2 approximation.  Error bound: ±0.01 rad (≈ 0.6°).
///
/// Suitable for azimuth/elevation calculations in equirectangular / cubemap
/// projection inner loops where sub-degree accuracy is sufficient.
#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_2, PI};
    let ax = x.abs();
    let ay = y.abs();
    // Guard against division by zero at the origin.
    let max_xy = ay.max(ax).max(1e-12);
    let a = ay.min(ax) / max_xy;
    // Hastings polynomial: atan(a) ≈ a*(π/4 - (a-1)*(0.2447 + 0.0663*a))
    let r = a * (FRAC_PI_2 * 0.5 - (a - 1.0) * (0.2447 + 0.0663 * a));
    // Restore octant symmetry.
    let r = if ay > ax { FRAC_PI_2 - r } else { r };
    let r = if x < 0.0 { PI - r } else { r };
    if y < 0.0 {
        -r
    } else {
        r
    }
}

/// Bhaskara I sin approximation over [−π, π].  Error bound: ±1.3e-3.
///
/// Substantially faster than `f32::sin` on platforms without hardware sin
/// instructions, or when the compiler cannot auto-vectorise libm calls.
#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn fast_sin(x: f32) -> f32 {
    use std::f32::consts::PI;
    // Reduce to [−π, π].
    let x = x - (x / (2.0 * PI)).round() * (2.0 * PI);
    // Bhaskara I formula: numerically stable form using |x|.
    let xa = x.abs();
    let num = 16.0 * x * (PI - xa);
    let den = 5.0 * PI * PI - 4.0 * xa * (PI - xa);
    // den is always positive for xa ∈ [0, π]; guard against floating-point edge.
    if den.abs() < 1e-12 {
        0.0
    } else {
        num / den
    }
}

/// Fast cos via π/2 phase shift of `fast_sin`.
///
/// Error bound matches `fast_sin`: ±1.3e-3.
#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn fast_cos(x: f32) -> f32 {
    fast_sin(x + std::f32::consts::FRAC_PI_2)
}

// ─── Internal validation ──────────────────────────────────────────────────────

fn validate_buffer(
    data: &[u8],
    w: u32,
    h: u32,
    channels: u32,
    _bytes_per_component: usize,
) -> Result<(), VrError> {
    if w == 0 || h == 0 {
        return Err(VrError::InvalidDimensions(
            "image width and height must be > 0".into(),
        ));
    }
    let expected = w as usize * h as usize * channels as usize;
    if data.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: data.len(),
        });
    }
    Ok(())
}

/// Reinterpret a `&[u8]` as `&[u8]` (identity, but typed for dispatch).
#[inline(always)]
fn data_as_u8_slice(data: &[u8]) -> &[u8] {
    data
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn solid_u8(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    fn gradient_u8(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h))
            .flat_map(|i| {
                let x = (i % w) as u8;
                [x, 0u8, 0u8]
            })
            .collect()
    }

    // ── Error paths ──────────────────────────────────────────────────────────

    #[test]
    fn sample_u8_zero_dim_error() {
        let img = solid_u8(4, 4, 128, 0, 0);
        assert!(sample_u8(&img, 0, 4, 0.5, 0.5, 3, FilterKernel::Bilinear).is_err());
        assert!(sample_u8(&img, 4, 0, 0.5, 0.5, 3, FilterKernel::Bilinear).is_err());
    }

    #[test]
    fn sample_u8_buffer_too_small_error() {
        assert!(sample_u8(&[0u8; 5], 4, 4, 0.5, 0.5, 3, FilterKernel::Bilinear).is_err());
    }

    #[test]
    fn sample_u16_zero_dim_error() {
        let img: Vec<u16> = vec![128u16; 4 * 4 * 3];
        assert!(sample_u16(&img, 0, 4, 0.5, 0.5, 3, FilterKernel::Bicubic).is_err());
    }

    #[test]
    fn sample_f32_zero_dim_error() {
        let img: Vec<f32> = vec![0.5f32; 4 * 4 * 3];
        assert!(sample_f32(&img, 4, 0, 0.5, 0.5, 3, FilterKernel::Lanczos3).is_err());
    }

    // ── Solid-colour images return correct colour for all kernels ─────────────

    #[test]
    fn bilinear_solid_colour_u8() {
        let img = solid_u8(8, 8, 200, 100, 50);
        let out = sample_u8(&img, 8, 8, 0.5, 0.5, 3, FilterKernel::Bilinear).expect("ok");
        assert_eq!(out[0], 200);
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 50);
    }

    #[test]
    fn bicubic_solid_colour_u8() {
        let img = solid_u8(8, 8, 100, 200, 50);
        let out = sample_u8(&img, 8, 8, 0.5, 0.5, 3, FilterKernel::Bicubic).expect("ok");
        // Solid image: bicubic should return exactly the same colour
        assert!((out[0] as i32 - 100).abs() <= 2, "R={}", out[0]);
        assert!((out[1] as i32 - 200).abs() <= 2, "G={}", out[1]);
    }

    #[test]
    fn lanczos2_solid_colour_u8() {
        let img = solid_u8(16, 16, 180, 90, 45);
        let out = sample_u8(&img, 16, 16, 0.5, 0.5, 3, FilterKernel::Lanczos2).expect("ok");
        assert!((out[0] as i32 - 180).abs() <= 2, "R={}", out[0]);
    }

    #[test]
    fn lanczos3_solid_colour_u8() {
        let img = solid_u8(16, 16, 60, 120, 240);
        let out = sample_u8(&img, 16, 16, 0.5, 0.5, 3, FilterKernel::Lanczos3).expect("ok");
        assert!((out[0] as i32 - 60).abs() <= 2, "R={}", out[0]);
        assert!((out[2] as i32 - 240).abs() <= 2, "B={}", out[2]);
    }

    // ── Gradient image: left pixel darker than right ─────────────────────────

    #[test]
    fn bicubic_gradient_order_preserved() {
        let img = gradient_u8(16, 1);
        let left = sample_u8(&img, 16, 1, 0.1, 0.5, 3, FilterKernel::Bicubic).expect("ok");
        let right = sample_u8(&img, 16, 1, 0.9, 0.5, 3, FilterKernel::Bicubic).expect("ok");
        assert!(left[0] < right[0], "left={} right={}", left[0], right[0]);
    }

    #[test]
    fn lanczos3_gradient_order_preserved() {
        let img = gradient_u8(16, 1);
        let left = sample_u8(&img, 16, 1, 0.1, 0.5, 3, FilterKernel::Lanczos3).expect("ok");
        let right = sample_u8(&img, 16, 1, 0.9, 0.5, 3, FilterKernel::Lanczos3).expect("ok");
        assert!(left[0] < right[0]);
    }

    // ── u16 sampler ──────────────────────────────────────────────────────────

    #[test]
    fn bicubic_solid_u16() {
        let img: Vec<u16> = (0..8 * 8 * 3)
            .map(|i| if i % 3 == 0 { 1000u16 } else { 500u16 })
            .collect();
        let out = sample_u16(&img, 8, 8, 0.5, 0.5, 3, FilterKernel::Bicubic).expect("ok");
        assert_eq!(out.len(), 3);
        assert!((out[0] as i32 - 1000).abs() <= 5, "got {}", out[0]);
    }

    // ── f32 sampler ──────────────────────────────────────────────────────────

    #[test]
    fn lanczos3_solid_f32() {
        let img: Vec<f32> = (0..8 * 8 * 3)
            .map(|i| if i % 3 == 0 { 0.8f32 } else { 0.4f32 })
            .collect();
        let out = sample_f32(&img, 8, 8, 0.5, 0.5, 3, FilterKernel::Lanczos3).expect("ok");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.8).abs() < 0.05, "got {}", out[0]);
    }

    // ── Mitchell-Netravali kernel unit tests ─────────────────────────────────

    #[test]
    fn mitchell_netravali_at_zero_correct_value() {
        // With B=1/3, C=1/3: MN(0) = (6 - 2*B)/6 = (6 - 2/3)/6 ≈ 0.8889
        // The kernel is NOT 1.0 at x=0; it is normalised via 4-tap sum = 1.0.
        let b = 1.0_f64 / 3.0;
        let c = 1.0_f64 / 3.0;
        let v = super::mitchell_netravali(0.0, b, c);
        let expected = (6.0 - 2.0 * b) / 6.0;
        assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
    }

    #[test]
    fn mitchell_netravali_four_tap_sums_to_one() {
        // The 4-tap kernel sum at subpixel offset = 0 should be ≈ 1.0
        // Tap positions relative to centre: -1, 0, 1, 2 (fx = 0)
        let b = 1.0_f64 / 3.0;
        let c = 1.0_f64 / 3.0;
        let fx = 0.0_f64;
        let weights: f64 = (-1..=2_i32)
            .map(|k| super::mitchell_netravali(fx - k as f64, b, c))
            .sum();
        assert!((weights - 1.0).abs() < 1e-9, "sum={weights}");
    }

    #[test]
    fn mitchell_netravali_beyond_two_is_zero() {
        let v = super::mitchell_netravali(2.5, 1.0 / 3.0, 1.0 / 3.0);
        assert_eq!(v, 0.0);
    }

    // ── sinc unit test ────────────────────────────────────────────────────────

    #[test]
    fn sinc_at_zero_is_one() {
        assert!((super::sinc(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sinc_at_integer_is_zero() {
        for n in 1..=5i32 {
            let v = super::sinc(n as f64);
            assert!(v.abs() < 1e-10, "sinc({n}) = {v}");
        }
    }

    // ── Output size ───────────────────────────────────────────────────────────

    #[test]
    fn all_kernels_return_correct_channel_count() {
        let img = solid_u8(4, 4, 10, 20, 30);
        for &k in &[
            FilterKernel::Bilinear,
            FilterKernel::Bicubic,
            FilterKernel::Lanczos2,
            FilterKernel::Lanczos3,
        ] {
            let out = sample_u8(&img, 4, 4, 0.5, 0.5, 3, k).expect("ok");
            assert_eq!(out.len(), 3, "kernel={k:?}");
        }
    }

    #[test]
    fn one_channel_image_works() {
        let img = vec![128u8; 4 * 4];
        for &k in &[
            FilterKernel::Bilinear,
            FilterKernel::Bicubic,
            FilterKernel::Lanczos3,
        ] {
            let out = sample_u8(&img, 4, 4, 0.5, 0.5, 1, k).expect("ok");
            assert_eq!(out.len(), 1);
            assert!((out[0] as i32 - 128).abs() <= 2);
        }
    }

    // ── Corner/edge clamping ──────────────────────────────────────────────────

    #[test]
    fn bicubic_corner_u0_v0_does_not_panic() {
        let img = solid_u8(4, 4, 50, 100, 150);
        let out = sample_u8(&img, 4, 4, 0.0, 0.0, 3, FilterKernel::Bicubic).expect("ok");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn lanczos3_corner_u1_v1_does_not_panic() {
        let img = solid_u8(4, 4, 50, 100, 150);
        let out = sample_u8(&img, 4, 4, 1.0, 1.0, 3, FilterKernel::Lanczos3).expect("ok");
        assert_eq!(out.len(), 3);
    }

    // ── sample_bicubic (Keys cubic, a=-0.5) ───────────────────────────────────

    #[test]
    fn sample_bicubic_zero_dim_returns_black() {
        let img = solid_u8(4, 4, 200, 100, 50);
        let out = sample_bicubic(&img, 0, 4, 2.0, 2.0);
        assert_eq!(out, [0, 0, 0]);
        let out2 = sample_bicubic(&img, 4, 0, 2.0, 2.0);
        assert_eq!(out2, [0, 0, 0]);
    }

    #[test]
    fn sample_bicubic_solid_colour_centre() {
        // A solid-colour image: any position should return that colour exactly.
        let img = solid_u8(8, 8, 180, 90, 45);
        let out = sample_bicubic(&img, 8, 8, 4.0, 4.0);
        assert!((out[0] as i32 - 180).abs() <= 3, "R={}", out[0]);
        assert!((out[1] as i32 - 90).abs() <= 3, "G={}", out[1]);
        assert!((out[2] as i32 - 45).abs() <= 3, "B={}", out[2]);
    }

    #[test]
    fn sample_bicubic_solid_colour_fractional_position() {
        let img = solid_u8(16, 16, 200, 150, 100);
        let out = sample_bicubic(&img, 16, 16, 7.3, 5.8);
        assert!((out[0] as i32 - 200).abs() <= 3, "R={}", out[0]);
        assert!((out[1] as i32 - 150).abs() <= 3, "G={}", out[1]);
        assert!((out[2] as i32 - 100).abs() <= 3, "B={}", out[2]);
    }

    #[test]
    fn sample_bicubic_corner_top_left_does_not_panic() {
        let img = solid_u8(4, 4, 50, 100, 150);
        let out = sample_bicubic(&img, 4, 4, 0.0, 0.0);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn sample_bicubic_corner_bottom_right_does_not_panic() {
        let img = solid_u8(4, 4, 50, 100, 150);
        // Pixel-space corner: (w-1, h-1)
        let out = sample_bicubic(&img, 4, 4, 3.9, 3.9);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn sample_bicubic_gradient_order_preserved() {
        // Gradient: pixel at (x, 0) has R = x (for x < 256).
        let img: Vec<u8> = (0..16u32).flat_map(|x| [x as u8, 0u8, 0u8]).collect();
        let left = sample_bicubic(&img, 16, 1, 2.0, 0.0);
        let right = sample_bicubic(&img, 16, 1, 12.0, 0.0);
        assert!(
            left[0] < right[0],
            "left R={} right R={}",
            left[0],
            right[0]
        );
    }

    #[test]
    fn keys_cubic_kernel_properties() {
        // Keys cubic at t=0 should be 1.0: (a+2)*0 - (a+3)*0 + 1 = 1
        let v0 = super::keys_cubic(0.0);
        assert!((v0 - 1.0).abs() < 1e-10, "keys_cubic(0)={v0}");

        // Keys cubic at |t|=1 should be 0.0:
        // (a+2)*1 - (a+3)*1 + 1 = a+2 - a-3 + 1 = 0 ✓
        let v1 = super::keys_cubic(1.0);
        assert!(v1.abs() < 1e-10, "keys_cubic(1)={v1}");

        // Keys cubic at |t|>=2 should be 0
        let v2 = super::keys_cubic(2.0);
        assert_eq!(v2, 0.0, "keys_cubic(2)={v2}");

        let v3 = super::keys_cubic(3.0);
        assert_eq!(v3, 0.0, "keys_cubic(3)={v3}");
    }

    #[test]
    fn sample_bicubic_matches_pixel_at_exact_integer_coords() {
        // At exact integer pixel centres, bicubic should return that pixel value.
        let mut img = vec![0u8; 8 * 8 * 3];
        // Set pixel (3, 4) to a distinctive colour
        let base = (4 * 8 + 3) * 3;
        img[base] = 255;
        img[base + 1] = 128;
        img[base + 2] = 64;
        let out = sample_bicubic(&img, 8, 8, 3.0, 4.0);
        // Due to ringing from neighbouring zero pixels, allow tolerance of 20
        assert!((out[0] as i32 - 255).abs() <= 20, "R={}", out[0]);
    }

    // ── SIMD batch bilinear sampling ──────────────────────────────────────────

    /// Build a deterministic pseudo-random 8-bit grayscale image.
    fn pseudo_random_img(size: usize) -> Vec<u8> {
        (0..size)
            .map(|i| {
                // Simple LCG-derived pattern: deterministic but not uniform.
                let x = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 56) as u8
            })
            .collect()
    }

    #[test]
    fn simd_bilinear_matches_scalar() {
        let img = pseudo_random_img(64 * 64);
        let coords: Vec<(f32, f32)> = (0..1000_usize)
            .map(|i| ((i % 64) as f32 + 0.3, (i / 64) as f32 % 64.0 + 0.7))
            .collect();
        let mut out_scalar = vec![0u8; 1000];
        let mut out_simd = vec![0u8; 1000];
        sample_bilinear_scalar(&img, 64, 64, 1, &coords, &mut out_scalar);
        sample_bilinear_batch(&img, 64, 64, 1, &coords, &mut out_simd);
        for (i, (s, b)) in out_scalar.iter().zip(out_simd.iter()).enumerate() {
            assert!(s.abs_diff(*b) <= 1, "coord[{i}]: scalar={s} simd={b}");
        }
    }

    #[test]
    fn simd_bilinear_batch_empty_coords() {
        let img = vec![128u8; 4 * 4];
        let coords: Vec<(f32, f32)> = vec![];
        let mut out = vec![];
        // Should not panic on empty input.
        sample_bilinear_batch(&img, 4, 4, 1, &coords, &mut out);
        sample_bilinear_scalar(&img, 4, 4, 1, &coords, &mut out);
    }

    #[test]
    fn simd_bilinear_batch_solid_image() {
        // A uniform-value image should yield the same value for any coordinate.
        let img = vec![200u8; 16 * 16];
        let coords: Vec<(f32, f32)> = (0..20).map(|i| (i as f32 * 0.7, i as f32 * 0.5)).collect();
        let mut out = vec![0u8; coords.len()];
        sample_bilinear_batch(&img, 16, 16, 1, &coords, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 200, "pixel[{i}]={v} expected 200");
        }
    }

    // ── Fast-math trig ────────────────────────────────────────────────────────

    #[cfg(feature = "fast_math")]
    #[test]
    fn fast_atan2_within_tol() {
        for i in 0..10_000usize {
            let angle = (i as f32 / 10_000.0) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
            let y = angle.sin();
            let x = angle.cos();
            let approx = super::fast_atan2(y, x);
            let exact = y.atan2(x);
            assert!(
                (approx - exact).abs() < 0.01,
                "angle={angle:.4} approx={approx:.4} exact={exact:.4}"
            );
        }
    }

    #[cfg(feature = "fast_math")]
    #[test]
    fn fast_sin_within_tol() {
        // Bhaskara I max error over [-π, π] is ≈ 1.3e-3; we test against 2e-3.
        for i in 0..10_000usize {
            let x = (i as f32 / 10_000.0) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
            let approx = super::fast_sin(x);
            let exact = x.sin();
            assert!(
                (approx - exact).abs() < 2e-3,
                "x={x:.4} approx={approx:.4} exact={exact:.4} err={:.6}",
                (approx - exact).abs()
            );
        }
    }

    #[cfg(feature = "fast_math")]
    #[test]
    fn fast_cos_within_tol() {
        for i in 0..10_000usize {
            let x = (i as f32 / 10_000.0) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
            let approx = super::fast_cos(x);
            let exact = x.cos();
            assert!(
                (approx - exact).abs() < 2e-3,
                "x={x:.4} approx={approx:.4} exact={exact:.4}"
            );
        }
    }

    #[cfg(feature = "fast_math")]
    #[test]
    fn fast_atan2_special_cases() {
        use std::f32::consts::PI;
        // Origin (both zero) — should not panic.
        let v = super::fast_atan2(0.0, 0.0);
        assert!(v.is_finite(), "fast_atan2(0,0)={v}");
        // Positive x-axis: atan2(0, 1) = 0
        let v = super::fast_atan2(0.0, 1.0);
        assert!(v.abs() < 0.01, "atan2(0,1)={v}");
        // Negative x-axis: atan2(0, -1) = π
        let v = super::fast_atan2(0.0, -1.0).abs();
        assert!((v - PI).abs() < 0.01, "atan2(0,-1)={v}");
    }
}
