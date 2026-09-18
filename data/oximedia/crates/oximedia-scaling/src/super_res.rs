//! Super resolution upscaling with `u8` pixel API.
//!
//! Provides bicubic upscaling and bicubic + unsharp-mask post-sharpening
//! for 8-bit RGBA images.  For floating-point / edge-guided methods see
//! `super_resolution`.

/// Super-resolution upscaler that works on 8-bit (RGBA) pixel buffers.
///
/// Both 2× and 4× integer scale factors are supported.  The input must be a
/// tightly-packed RGBA (4-byte-per-pixel) buffer of length
/// `src_w * src_h * 4`.
#[derive(Debug, Clone, Copy)]
pub struct SuperResolutionUpscaler {
    /// Integer scale factor — either 2 or 4.
    pub scale_factor: u32,
}

impl SuperResolutionUpscaler {
    /// Create a new upscaler.  `scale_factor` should be 2 or 4; any other
    /// value is accepted but may yield unexpected quality.
    #[must_use]
    pub fn new(scale_factor: u32) -> Self {
        Self { scale_factor }
    }

    /// Bicubic upscale from `src_w × src_h` to `dst_w × dst_h`.
    ///
    /// The `src` slice must contain `src_w * src_h * 4` bytes (RGBA).
    /// Returns a `Vec<u8>` of length `dst_w * dst_h * 4`.
    ///
    /// Returns an empty `Vec` when any dimension is zero.
    #[must_use]
    pub fn upscale_bicubic(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Vec<u8> {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }

        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = dst_w as usize;
        let dh = dst_h as usize;

        let scale_x = sw as f64 / dw as f64;
        let scale_y = sh as f64 / dh as f64;

        let mut dst = vec![0u8; dw * dh * 4];

        for dy in 0..dh {
            for dx in 0..dw {
                // Map destination pixel back to source coordinates (center alignment).
                let fx = (dx as f64 + 0.5) * scale_x - 0.5;
                let fy = (dy as f64 + 0.5) * scale_y - 0.5;

                let base = (dy * dw + dx) * 4;
                let pixel = bicubic_sample_rgba(src, sw, sh, fx, fy);
                dst[base] = pixel[0];
                dst[base + 1] = pixel[1];
                dst[base + 2] = pixel[2];
                dst[base + 3] = pixel[3];
            }
        }

        dst
    }

    /// Bicubic upscale followed by an unsharp-mask post-sharpening pass.
    ///
    /// `sharpen_strength` controls the amplitude of the sharpening mask.
    /// Values in [0.0, 1.0] are typical; 0.0 gives the same result as
    /// [`upscale_bicubic`].
    ///
    /// [`upscale_bicubic`]: SuperResolutionUpscaler::upscale_bicubic
    #[must_use]
    pub fn upscale_with_sharpening(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        sharpen_strength: f32,
    ) -> Vec<u8> {
        let bicubic = self.upscale_bicubic(src, src_w, src_h, dst_w, dst_h);
        if bicubic.is_empty() {
            return bicubic;
        }
        apply_unsharp_mask_rgba(&bicubic, dst_w as usize, dst_h as usize, sharpen_strength)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Catmull-Rom cubic weights for fractional position `t` in [0, 1].
fn catmull_rom_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Clamp a floating-point value to `[0, 255]` and convert to `u8`.
#[inline]
fn clamp_u8(v: f64) -> u8 {
    v.clamp(0.0, 255.0).round() as u8
}

/// Sample a single RGBA pixel from `src` using bicubic (Catmull-Rom)
/// interpolation at continuous source coordinates `(fx, fy)`.
fn bicubic_sample_rgba(src: &[u8], sw: usize, sh: usize, fx: f64, fy: f64) -> [u8; 4] {
    let ix = fx.floor() as i64;
    let iy = fy.floor() as i64;
    let tx = fx - ix as f64;
    let ty = fy - iy as f64;

    let wx = catmull_rom_weights(tx);
    let wy = catmull_rom_weights(ty);

    let mut channels = [0.0f64; 4];

    for (j, &wy_j) in wy.iter().enumerate() {
        for (i, &wx_i) in wx.iter().enumerate() {
            let px = (ix + i as i64 - 1).clamp(0, sw as i64 - 1) as usize;
            let py = (iy + j as i64 - 1).clamp(0, sh as i64 - 1) as usize;
            let base = (py * sw + px) * 4;
            for c in 0..4 {
                channels[c] += src[base + c] as f64 * wx_i * wy_j;
            }
        }
    }

    [
        clamp_u8(channels[0]),
        clamp_u8(channels[1]),
        clamp_u8(channels[2]),
        clamp_u8(channels[3]),
    ]
}

/// Apply a 3×3 box-blur unsharp mask to an RGBA buffer.
///
/// `output = clamp(src + strength × (src − blur), 0, 255)`
fn apply_unsharp_mask_rgba(src: &[u8], w: usize, h: usize, strength: f32) -> Vec<u8> {
    if strength == 0.0 {
        return src.to_vec();
    }

    // Compute 3×3 box blur.
    let mut blur = src.to_vec();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            for c in 0..4 {
                let sum: u32 = [
                    ((y - 1) * w + (x - 1)),
                    ((y - 1) * w + x),
                    ((y - 1) * w + (x + 1)),
                    (y * w + (x - 1)),
                    (y * w + x),
                    (y * w + (x + 1)),
                    ((y + 1) * w + (x - 1)),
                    ((y + 1) * w + x),
                    ((y + 1) * w + (x + 1)),
                ]
                .iter()
                .map(|&idx| src[idx * 4 + c] as u32)
                .sum();
                blur[(y * w + x) * 4 + c] = (sum / 9) as u8;
            }
        }
    }

    // Unsharp mask: original + strength × (original − blurred).
    src.iter()
        .zip(blur.iter())
        .map(|(&s, &b)| {
            let diff = s as f32 - b as f32;
            clamp_u8((s as f32 + strength * diff) as f64)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat RGBA image of `w×h` pixels filled with `pixel`.
    fn solid(w: u32, h: u32, pixel: [u8; 4]) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut buf = Vec::with_capacity(n);
        for _ in 0..(w * h) {
            buf.extend_from_slice(&pixel);
        }
        buf
    }

    #[test]
    fn test_upscale_bicubic_dimensions() {
        // 2×2 → 4×4
        let src = solid(2, 2, [128, 64, 32, 255]);
        let upscaler = SuperResolutionUpscaler::new(2);
        let dst = upscaler.upscale_bicubic(&src, 2, 2, 4, 4);
        assert_eq!(dst.len(), 4 * 4 * 4, "output length must be dst_w*dst_h*4");
    }

    #[test]
    fn test_upscale_bicubic_uniform_color_preserved() {
        // A uniform-color image should remain uniform after upscaling.
        let pixel = [200u8, 100, 50, 255];
        let src = solid(2, 2, pixel);
        let upscaler = SuperResolutionUpscaler::new(2);
        let dst = upscaler.upscale_bicubic(&src, 2, 2, 4, 4);
        for chunk in dst.chunks_exact(4) {
            // Allow ±2 for floating-point rounding.
            for (c, &expected) in chunk.iter().zip(pixel.iter()) {
                let diff = (c.saturating_sub(expected)).max(expected.saturating_sub(*c));
                assert!(diff <= 2, "channel mismatch: got {c}, expected {expected}");
            }
        }
    }

    #[test]
    fn test_upscale_bicubic_zero_dimensions_returns_empty() {
        let upscaler = SuperResolutionUpscaler::new(2);
        assert!(upscaler.upscale_bicubic(&[], 0, 0, 4, 4).is_empty());
        assert!(upscaler.upscale_bicubic(&[0u8; 16], 2, 2, 0, 4).is_empty());
    }

    #[test]
    fn test_upscale_with_sharpening_dimensions() {
        let src = solid(2, 2, [100, 150, 200, 255]);
        let upscaler = SuperResolutionUpscaler::new(2);
        let dst = upscaler.upscale_with_sharpening(&src, 2, 2, 4, 4, 0.5);
        assert_eq!(dst.len(), 4 * 4 * 4);
    }

    #[test]
    fn test_upscale_with_zero_strength_matches_bicubic() {
        // With strength = 0.0, sharpening should be a no-op.
        let src = solid(2, 2, [80, 160, 240, 255]);
        let upscaler = SuperResolutionUpscaler::new(2);
        let bicubic = upscaler.upscale_bicubic(&src, 2, 2, 4, 4);
        let sharpened = upscaler.upscale_with_sharpening(&src, 2, 2, 4, 4, 0.0);
        assert_eq!(bicubic, sharpened);
    }

    #[test]
    fn test_scale_factor_4x() {
        let src = solid(2, 2, [255, 0, 128, 255]);
        let upscaler = SuperResolutionUpscaler::new(4);
        let dst = upscaler.upscale_bicubic(&src, 2, 2, 8, 8);
        assert_eq!(dst.len(), 8 * 8 * 4);
    }
}
