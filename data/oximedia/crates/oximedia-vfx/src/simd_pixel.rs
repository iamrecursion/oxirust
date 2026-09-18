//! SIMD-accelerated pixel-level operations for VFX frame processing.
//!
//! This module provides high-throughput implementations of the fundamental
//! pixel operations used throughout the VFX pipeline:
//!
//! * **`fill_rgba`** — fill a packed RGBA byte buffer with a constant colour.
//!   Uses 128-bit write chunks on all platforms; on x86_64 with AVX2 available
//!   at runtime, 256-bit (32-byte) chunks are used.
//!
//! * **`blend_rgba_buffers`** — alpha-composite (Porter-Duff "over") two
//!   equal-length packed RGBA buffers: `dst[i] = src[i].a/255 * src[i] + (1 - src[i].a/255) * dst[i]`.
//!   Implemented with integer fixed-point arithmetic to avoid f32 on the hot
//!   path; processes four channels in one pass per pixel.
//!
//! * **`lerp_rgba_buffers`** — linearly interpolate two buffers at a uniform
//!   parameter `t ∈ [0, 1]`.  Used by dissolve transitions and cross-fades.
//!
//! All functions are `forbid(unsafe_code)` compliant — no `unsafe` blocks are
//! needed because the SIMD widening is achieved by packing pixels into wider
//! integer types via `u128::to_ne_bytes()` and standard slice chunking.
//!
//! # Example
//!
//! ```
//! use oximedia_vfx::simd_pixel::{fill_rgba, blend_rgba_buffers, lerp_rgba_buffers};
//!
//! // Fill 1920×1080 frame with opaque red
//! let mut buf = vec![0u8; 1920 * 1080 * 4];
//! fill_rgba(&mut buf, [255, 0, 0, 255]);
//! assert_eq!(&buf[0..4], &[255, 0, 0, 255]);
//!
//! // Blend a semi-transparent green layer over the red
//! let green = vec![0u8, 200u8, 0u8, 128u8].repeat(1920 * 1080);
//! let mut dst = buf;
//! blend_rgba_buffers(&green, &mut dst);
//! // dst now contains the over-composited result
//!
//! // Lerp two buffers at t=0.5
//! let a = vec![100u8, 100, 100, 255].repeat(4);
//! let b = vec![200u8, 200, 200, 255].repeat(4);
//! let mut out = vec![0u8; 4 * 4];
//! lerp_rgba_buffers(&a, &b, &mut out, 0.5);
//! assert_eq!(out[0], 150);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// fill_rgba
// ─────────────────────────────────────────────────────────────────────────────

/// Fill `buf` with the constant RGBA pixel `rgba`.
///
/// * On **x86_64 + AVX2** (detected at runtime): writes 32-byte chunks.
/// * On all other targets: writes 16-byte chunks via `u128`.
///
/// `buf` must have a length that is a multiple of 4 (one pixel = 4 bytes).
/// Any trailing bytes (< 4) are left unchanged (this should not occur for
/// well-formed RGBA buffers).
pub fn fill_rgba(buf: &mut [u8], rgba: [u8; 4]) {
    // Pack 4-byte pixel → u32 → u128 (four pixels)
    let px32 = u32::from_ne_bytes(rgba);
    let px64 = (px32 as u64) | ((px32 as u64) << 32);
    let px128: u128 = (px64 as u128) | ((px64 as u128) << 64);
    let bytes128 = px128.to_ne_bytes();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // 256-bit write: two u128 side by side
            let mut bytes256 = [0u8; 32];
            bytes256[..16].copy_from_slice(&bytes128);
            bytes256[16..].copy_from_slice(&bytes128);

            let chunks32 = buf.len() / 32;
            for i in 0..chunks32 {
                buf[i * 32..(i + 1) * 32].copy_from_slice(&bytes256);
            }
            let rem_start = chunks32 * 32;
            // Handle remaining 16-byte blocks
            let rem = &mut buf[rem_start..];
            let chunks16 = rem.len() / 16;
            for i in 0..chunks16 {
                rem[i * 16..(i + 1) * 16].copy_from_slice(&bytes128);
            }
            let tail_start = rem_start + chunks16 * 16;
            for chunk in buf[tail_start..].chunks_exact_mut(4) {
                chunk.copy_from_slice(&rgba);
            }
            return;
        }
    }

    // Generic 16-byte path
    let chunks16 = buf.len() / 16;
    for i in 0..chunks16 {
        buf[i * 16..(i + 1) * 16].copy_from_slice(&bytes128);
    }
    let tail = chunks16 * 16;
    for chunk in buf[tail..].chunks_exact_mut(4) {
        chunk.copy_from_slice(&rgba);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// blend_rgba_buffers
// ─────────────────────────────────────────────────────────────────────────────

/// Porter-Duff "over" composite: `dst = src ◦ over ◦ dst`.
///
/// For each pixel:
/// ```text
/// alpha = src.a
/// dst.r = (src.r * alpha + dst.r * (255 - alpha) + 127) / 255
/// dst.g = (src.g * alpha + dst.g * (255 - alpha) + 127) / 255
/// dst.b = (src.b * alpha + dst.b * (255 - alpha) + 127) / 255
/// dst.a = max(dst.a, src.a)
/// ```
///
/// Uses integer fixed-point arithmetic; no f32 divisions on the hot path.
///
/// # Panics
///
/// Panics in debug mode if `src.len() != dst.len()` or if either length is not
/// a multiple of 4.
pub fn blend_rgba_buffers(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len(), "src and dst must be the same length");
    debug_assert_eq!(src.len() % 4, 0, "buffer length must be a multiple of 4");

    let len = src.len().min(dst.len());
    let mut i = 0;
    while i + 4 <= len {
        let alpha = src[i + 3] as u32;
        if alpha == 0 {
            // Fully transparent src: dst unchanged
            i += 4;
            continue;
        }
        if alpha == 255 {
            // Fully opaque src: straight copy
            dst[i..i + 4].copy_from_slice(&src[i..i + 4]);
            i += 4;
            continue;
        }
        let inv = 255 - alpha;
        // Blend with rounding: (a * fg + b * bg + 127) / 255
        let blend_ch =
            |s: u8, d: u8| -> u8 { ((s as u32 * alpha + d as u32 * inv + 127) / 255) as u8 };
        dst[i] = blend_ch(src[i], dst[i]);
        dst[i + 1] = blend_ch(src[i + 1], dst[i + 1]);
        dst[i + 2] = blend_ch(src[i + 2], dst[i + 2]);
        dst[i + 3] = dst[i + 3].max(src[i + 3]);
        i += 4;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// lerp_rgba_buffers
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation between two RGBA buffers at parameter `t ∈ [0, 1]`.
///
/// For each byte: `out[i] = a[i] + t * (b[i] - a[i])`.
///
/// Uses fixed-point: `t` is scaled to `[0, 256]` so the division becomes a
/// right-shift.  Result is clamped to `[0, 255]`.
///
/// # Panics
///
/// Panics in debug mode if `a.len() != b.len()` or `out.len() < a.len()`.
pub fn lerp_rgba_buffers(a: &[u8], b: &[u8], out: &mut [u8], t: f32) {
    debug_assert_eq!(a.len(), b.len(), "a and b must be the same length");
    debug_assert!(out.len() >= a.len(), "out must be at least as long as a");

    let t = t.clamp(0.0, 1.0);
    let t_fp = (t * 256.0) as u32; // [0, 256]
    let inv_fp = 256 - t_fp;

    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    // Process 4 bytes (one pixel) at a time for locality
    while i + 4 <= len {
        let lerp_ch = |av: u8, bv: u8| -> u8 {
            ((av as u32 * inv_fp + bv as u32 * t_fp + 128) >> 8).min(255) as u8
        };
        out[i] = lerp_ch(a[i], b[i]);
        out[i + 1] = lerp_ch(a[i + 1], b[i + 1]);
        out[i + 2] = lerp_ch(a[i + 2], b[i + 2]);
        out[i + 3] = lerp_ch(a[i + 3], b[i + 3]);
        i += 4;
    }
    // Tail (< 4 bytes — should not occur for RGBA buffers)
    while i < len {
        out[i] = ((a[i] as u32 * inv_fp + b[i] as u32 * t_fp + 128) >> 8).min(255) as u8;
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_color_multiply
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply every pixel's RGB channels by a constant colour, preserving alpha.
///
/// Each channel: `dst[ch] = (src[ch] * colour[ch] + 127) / 255`.
///
/// Useful for tinting operations.
pub fn apply_color_multiply(buf: &mut [u8], colour: [u8; 3]) {
    let cr = colour[0] as u32;
    let cg = colour[1] as u32;
    let cb = colour[2] as u32;
    for chunk in buf.chunks_exact_mut(4) {
        chunk[0] = ((chunk[0] as u32 * cr + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u32 * cg + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u32 * cb + 127) / 255) as u8;
        // alpha unchanged
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── fill_rgba ──────────────────────────────────────────────────────────

    #[test]
    fn test_fill_rgba_all_pixels() {
        let mut buf = vec![0u8; 4 * 10];
        fill_rgba(&mut buf, [255, 128, 64, 255]);
        for chunk in buf.chunks_exact(4) {
            assert_eq!(chunk, &[255, 128, 64, 255]);
        }
    }

    #[test]
    fn test_fill_rgba_large_buffer() {
        let size = 1920 * 1080 * 4;
        let mut buf = vec![0u8; size];
        fill_rgba(&mut buf, [0, 0, 0, 255]);
        assert!(buf.iter().step_by(4).all(|&b| b == 0));
        assert!(buf.iter().skip(3).step_by(4).all(|&b| b == 255));
    }

    #[test]
    fn test_fill_rgba_single_pixel() {
        let mut buf = vec![0u8; 4];
        fill_rgba(&mut buf, [1, 2, 3, 4]);
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    // ── blend_rgba_buffers ─────────────────────────────────────────────────

    #[test]
    fn test_blend_fully_opaque_src() {
        let src = vec![100u8, 150, 200, 255]; // opaque
        let mut dst = vec![50u8, 50, 50, 255];
        blend_rgba_buffers(&src, &mut dst);
        assert_eq!(dst, [100, 150, 200, 255]);
    }

    #[test]
    fn test_blend_fully_transparent_src() {
        let src = vec![100u8, 150, 200, 0]; // fully transparent
        let mut dst = vec![50u8, 60, 70, 255];
        blend_rgba_buffers(&src, &mut dst);
        assert_eq!(dst, [50, 60, 70, 255]); // unchanged
    }

    #[test]
    fn test_blend_semi_transparent() {
        // 50% alpha (128/255 ≈ 0.502)
        let src = vec![200u8, 100, 50, 128];
        let mut dst = vec![0u8, 0, 0, 255];
        blend_rgba_buffers(&src, &mut dst);
        // Expected: (200*128 + 0*127 + 127) / 255 ≈ 100
        let expected_r = (200u32 * 128 + 0 * 127 + 127) / 255;
        assert!((dst[0] as i32 - expected_r as i32).abs() <= 1);
    }

    // ── lerp_rgba_buffers ──────────────────────────────────────────────────

    #[test]
    fn test_lerp_t0_returns_a() {
        let a = vec![100u8, 100, 100, 255];
        let b = vec![200u8, 200, 200, 255];
        let mut out = vec![0u8; 4];
        lerp_rgba_buffers(&a, &b, &mut out, 0.0);
        assert_eq!(out[0], 100);
    }

    #[test]
    fn test_lerp_t1_returns_b() {
        let a = vec![100u8, 100, 100, 255];
        let b = vec![200u8, 200, 200, 255];
        let mut out = vec![0u8; 4];
        lerp_rgba_buffers(&a, &b, &mut out, 1.0);
        assert_eq!(out[0], 200);
    }

    #[test]
    fn test_lerp_t_half() {
        let a = vec![100u8, 100, 100, 255].repeat(4);
        let b = vec![200u8, 200, 200, 255].repeat(4);
        let mut out = vec![0u8; 4 * 4];
        lerp_rgba_buffers(&a, &b, &mut out, 0.5);
        // 100 + 0.5 * (200 - 100) = 150, with fixed-point rounding ±1
        assert!((out[0] as i32 - 150).abs() <= 1);
    }

    #[test]
    fn test_lerp_multi_pixel() {
        let a = vec![0u8, 0, 0, 0];
        let b = vec![255u8, 255, 255, 255];
        let mut out = vec![0u8; 4];
        lerp_rgba_buffers(&a, &b, &mut out, 0.25);
        // 0 + 0.25*255 ≈ 63 or 64
        assert!((out[0] as i32 - 63).abs() <= 1);
    }

    // ── apply_color_multiply ───────────────────────────────────────────────

    #[test]
    fn test_color_multiply_white_is_identity() {
        let mut buf = vec![100u8, 150, 200, 255];
        apply_color_multiply(&mut buf, [255, 255, 255]);
        assert!((buf[0] as i32 - 100).abs() <= 1);
        assert!((buf[1] as i32 - 150).abs() <= 1);
        assert!((buf[2] as i32 - 200).abs() <= 1);
    }

    #[test]
    fn test_color_multiply_black_zeroes_rgb() {
        let mut buf = vec![100u8, 150, 200, 255];
        apply_color_multiply(&mut buf, [0, 0, 0]);
        assert_eq!(buf[0], 0);
        assert_eq!(buf[1], 0);
        assert_eq!(buf[2], 0);
        assert_eq!(buf[3], 255); // alpha preserved
    }

    #[test]
    fn test_color_multiply_preserves_alpha() {
        let mut buf = vec![255u8, 255, 255, 128];
        apply_color_multiply(&mut buf, [128, 128, 128]);
        assert_eq!(buf[3], 128); // alpha unchanged
    }
}
