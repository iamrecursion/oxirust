//! Alpha blending and compositing operations.
//!
//! Provides Porter-Duff compositing, premultiplied-alpha helpers, and a
//! selection of Photoshop-style blend modes. All arithmetic is performed in
//! the `u32` / `f32` domain and then clamped back to `u8`.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(dead_code)]

// ── Low-level pixel blend modes ──────────────────────────────────────────────

/// Screen blend mode: `1 - (1-a)*(1-b)` mapped to `[0, 255]`.
#[must_use]
pub fn screen_blend(a: u8, b: u8) -> u8 {
    let a = u32::from(a);
    let b = u32::from(b);
    let result = 255 - (255 - a) * (255 - b) / 255;
    result as u8
}

/// Multiply blend mode: `a * b / 255`.
#[must_use]
pub fn multiply_blend(a: u8, b: u8) -> u8 {
    ((u32::from(a) * u32::from(b)) / 255) as u8
}

/// Overlay blend mode: multiply for dark base, screen for light base.
#[must_use]
pub fn overlay_blend(a: u8, b: u8) -> u8 {
    // `a` is the base layer.
    if a < 128 {
        multiply_blend(a, b).saturating_mul(2)
    } else {
        let s = u32::from(screen_blend(a, b));
        // 2*screen - 255, clamped
        (s.saturating_sub(127)) as u8
    }
}

/// Hard-light blend mode: overlay with layers swapped.
#[must_use]
pub fn hard_light_blend(a: u8, b: u8) -> u8 {
    overlay_blend(b, a)
}

// ── Porter-Duff "over" compositing ───────────────────────────────────────────

/// Porter-Duff "over" compositing for two RGBA pixels.
///
/// Computes `fg over bg` and returns the resulting RGBA pixel. Channels are
/// stored as `[R, G, B, A]`.
#[must_use]
pub fn composite_over(fg: [u8; 4], bg: [u8; 4]) -> [u8; 4] {
    let fa = u32::from(fg[3]);
    let ba = u32::from(bg[3]);

    // out_a = fa + ba * (255 - fa) / 255
    let out_a = fa + ba * (255 - fa) / 255;
    if out_a == 0 {
        return [0, 0, 0, 0];
    }

    let mut out = [0u8; 4];
    for i in 0..3 {
        let fc = u32::from(fg[i]);
        let bc = u32::from(bg[i]);
        // pre-multiply each channel, composite, then un-premultiply
        let val = (fc * fa + bc * ba * (255 - fa) / 255)
            .checked_div(out_a)
            .unwrap_or(0);
        out[i] = val.min(255) as u8;
    }
    out[3] = out_a.min(255) as u8;
    out
}

// ── Premultiplied-alpha helpers ───────────────────────────────────────────────

/// Convert RGBA pixels from straight alpha to premultiplied alpha in-place.
///
/// `pixels` must be a flat RGBA buffer (4 bytes per pixel).
pub fn premultiply_alpha(pixels: &mut [u8]) {
    let count = pixels.len() / 4;
    for i in 0..count {
        let base = i * 4;
        let a = u32::from(pixels[base + 3]);
        pixels[base] = (u32::from(pixels[base]) * a / 255) as u8;
        pixels[base + 1] = (u32::from(pixels[base + 1]) * a / 255) as u8;
        pixels[base + 2] = (u32::from(pixels[base + 2]) * a / 255) as u8;
    }
}

/// Convert RGBA pixels from premultiplied alpha back to straight alpha in-place.
///
/// `pixels` must be a flat RGBA buffer (4 bytes per pixel).
pub fn unpremultiply_alpha(pixels: &mut [u8]) {
    let count = pixels.len() / 4;
    for i in 0..count {
        let base = i * 4;
        let a = u32::from(pixels[base + 3]);
        pixels[base] = (u32::from(pixels[base]) * 255)
            .checked_div(a)
            .unwrap_or(0)
            .min(255) as u8;
        pixels[base + 1] = (u32::from(pixels[base + 1]) * 255)
            .checked_div(a)
            .unwrap_or(0)
            .min(255) as u8;
        pixels[base + 2] = (u32::from(pixels[base + 2]) * 255)
            .checked_div(a)
            .unwrap_or(0)
            .min(255) as u8;
    }
}

// ── Batch alpha blending ──────────────────────────────────────────────────────

/// Blend `src` over `dst` using the alpha channel from `src`.
///
/// All three slices must hold RGBA data (4 bytes/pixel) and be the same
/// length. The output is written to `out`.
///
/// # Panics
///
/// Panics if `out.len() < (src.len().min(dst.len()) / 4) * 4`.
pub fn alpha_blend(src: &[u8], dst: &[u8], out: &mut [u8]) {
    let pixels = src.len().min(dst.len()) / 4;
    for i in 0..pixels {
        let base = i * 4;
        let sa = u32::from(src[base + 3]);
        let inv_sa = 255 - sa;

        out[base] = ((u32::from(src[base]) * sa + u32::from(dst[base]) * inv_sa) / 255) as u8;
        out[base + 1] =
            ((u32::from(src[base + 1]) * sa + u32::from(dst[base + 1]) * inv_sa) / 255) as u8;
        out[base + 2] =
            ((u32::from(src[base + 2]) * sa + u32::from(dst[base + 2]) * inv_sa) / 255) as u8;
        out[base + 3] = (sa + u32::from(dst[base + 3]) * inv_sa / 255) as u8;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_blend_zeros() {
        assert_eq!(multiply_blend(0, 255), 0);
        assert_eq!(multiply_blend(255, 0), 0);
    }

    #[test]
    fn test_multiply_blend_identity() {
        // Multiplying by 255 should give the original value (±1 rounding).
        let v = multiply_blend(200, 255);
        assert!((i32::from(v) - 200).abs() <= 1);
    }

    #[test]
    fn test_screen_blend_zeros() {
        // screen(0, x) = x
        assert_eq!(screen_blend(0, 200), 200);
    }

    #[test]
    fn test_screen_blend_max() {
        // screen(255, x) = 255
        assert_eq!(screen_blend(255, 100), 255);
    }

    #[test]
    fn test_screen_blend_symmetric() {
        // screen is commutative
        assert_eq!(screen_blend(100, 150), screen_blend(150, 100));
    }

    #[test]
    fn test_overlay_dark_base() {
        // For base < 128 overlay is closer to multiply (darker result)
        let result = overlay_blend(50, 100);
        assert!(result < 128);
    }

    #[test]
    fn test_hard_light_swap() {
        // hard_light(a, b) == overlay(b, a)
        assert_eq!(hard_light_blend(80, 200), overlay_blend(200, 80));
    }

    #[test]
    fn test_composite_over_opaque_fg() {
        // Opaque foreground should fully replace background.
        let fg = [255, 0, 0, 255];
        let bg = [0, 0, 255, 255];
        let out = composite_over(fg, bg);
        assert_eq!(out[0], 255); // R
        assert_eq!(out[2], 0); // B
        assert_eq!(out[3], 255); // A
    }

    #[test]
    fn test_composite_over_transparent_fg() {
        // Fully transparent foreground should pass background through.
        let fg = [255, 0, 0, 0];
        let bg = [0, 100, 200, 255];
        let out = composite_over(fg, bg);
        // Background RGB should be preserved.
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 200);
    }

    #[test]
    fn test_premultiply_round_trip() {
        let mut pixels = vec![200u8, 100, 50, 128];
        let original = pixels.clone();
        premultiply_alpha(&mut pixels);
        unpremultiply_alpha(&mut pixels);
        // Allow ±2 rounding error per channel.
        for i in 0..3 {
            let diff = (i32::from(pixels[i]) - i32::from(original[i])).abs();
            assert!(
                diff <= 2,
                "channel {i}: orig={} got={}",
                original[i],
                pixels[i]
            );
        }
        assert_eq!(pixels[3], original[3]);
    }

    #[test]
    fn test_premultiply_transparent_pixel() {
        let mut pixels = vec![255u8, 255, 255, 0];
        premultiply_alpha(&mut pixels);
        assert_eq!(pixels[0], 0);
        assert_eq!(pixels[1], 0);
        assert_eq!(pixels[2], 0);
    }

    #[test]
    fn test_unpremultiply_zero_alpha() {
        let mut pixels = vec![50u8, 60, 70, 0];
        unpremultiply_alpha(&mut pixels);
        // Division by zero path should zero channels, not panic.
        assert_eq!(pixels[0], 0);
        assert_eq!(pixels[3], 0);
    }

    #[test]
    fn test_alpha_blend_opaque_src() {
        let src = vec![255u8, 0, 0, 255];
        let dst = vec![0u8, 0, 255, 255];
        let mut out = vec![0u8; 4];
        alpha_blend(&src, &dst, &mut out);
        assert_eq!(out[0], 255); // R from src
        assert_eq!(out[2], 0); // B gone
    }

    #[test]
    fn test_alpha_blend_transparent_src() {
        let src = vec![255u8, 0, 0, 0];
        let dst = vec![0u8, 100, 200, 255];
        let mut out = vec![0u8; 4];
        alpha_blend(&src, &dst, &mut out);
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 200);
    }

    #[test]
    fn test_alpha_blend_preserves_length() {
        let src = vec![255u8; 8];
        let dst = vec![0u8; 8];
        let mut out = vec![0u8; 8];
        alpha_blend(&src, &dst, &mut out);
        assert_eq!(out.len(), 8);
    }
}
