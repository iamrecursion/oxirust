//! High-level `Compositor` providing composite, gradient overlay, vignette, and
//! multi-layer masked compositing operations.
//!
//! All operations work on in-memory RGBA8 buffers.

use super::blend::{blend_pixels, BlendMode};

/// Stateless compositor for common image compositing operations.
pub struct Compositor;

impl Compositor {
    /// Composite `top` over `bottom` using the specified blend mode and opacity,
    /// writing the result into `output`.
    ///
    /// All three buffers must be RGBA8 and exactly `width * height * 4` bytes long.
    ///
    /// # Panics
    ///
    /// Panics if any buffer length does not match `width * height * 4`.
    pub fn composite(
        bottom: &[u8],
        top: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        mode: BlendMode,
        opacity: f32,
    ) {
        let expected = (width as usize) * (height as usize) * 4;
        assert_eq!(bottom.len(), expected, "bottom buffer size mismatch");
        assert_eq!(top.len(), expected, "top buffer size mismatch");
        assert_eq!(output.len(), expected, "output buffer size mismatch");

        for (chunk_idx, out_chunk) in output.chunks_exact_mut(4).enumerate() {
            let base = chunk_idx * 4;
            let b = [
                bottom[base],
                bottom[base + 1],
                bottom[base + 2],
                bottom[base + 3],
            ];
            let t = [top[base], top[base + 1], top[base + 2], top[base + 3]];
            let r = blend_pixels(b, t, mode, opacity);
            out_chunk.copy_from_slice(&r);
        }
    }

    /// Overlay a linear gradient on `target`.
    ///
    /// The gradient runs from `color_start` to `color_end` at the given
    /// `angle_deg` (0° = left-to-right, 90° = top-to-bottom).
    /// Compositing uses standard alpha-over blending.
    ///
    /// # Panics
    ///
    /// Panics if `target.len() != width * height * 4`.
    pub fn gradient_overlay(
        target: &mut [u8],
        width: u32,
        height: u32,
        color_start: [u8; 4],
        color_end: [u8; 4],
        angle_deg: f32,
    ) {
        let expected = (width as usize) * (height as usize) * 4;
        assert_eq!(target.len(), expected, "target buffer size mismatch");

        if width == 0 || height == 0 {
            return;
        }

        let angle_rad = angle_deg.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        // Gradient is projected onto the direction vector (cos, sin).
        // Centre of the image is the zero-point; we normalise so that
        // the projection spans [0, 1] across the longest diagonal.
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;

        // Half-length of the projection of the image corners onto the direction
        let half_len = (cx * cos_a.abs() + cy * sin_a.abs()).max(1.0);

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let proj = dx * cos_a + dy * sin_a;
                let t = ((proj / half_len) * 0.5 + 0.5).clamp(0.0, 1.0);

                // Linearly interpolate the gradient color
                let grad = interpolate_color(color_start, color_end, t);

                let idx = ((y * width + x) * 4) as usize;
                let existing = [
                    target[idx],
                    target[idx + 1],
                    target[idx + 2],
                    target[idx + 3],
                ];
                let blended = alpha_over(existing, grad);
                target[idx..idx + 4].copy_from_slice(&blended);
            }
        }
    }

    /// Apply a vignette effect to `target`.
    ///
    /// The vignette darkens (or tints) pixels towards the edges of the frame.
    ///
    /// # Parameters
    /// - `strength` – How strongly the vignette blends (0.0 = no effect, 1.0 = full).
    /// - `radius`   – Normalised radius of the clear centre (0.0–1.0).  At `radius = 0.5`
    ///                the vignette starts half-way between the centre and the nearest edge.
    /// - `color`    – RGBA vignette colour (typically `[0, 0, 0, 255]` for a dark vignette).
    ///
    /// # Panics
    ///
    /// Panics if `target.len() != width * height * 4`.
    pub fn vignette(
        target: &mut [u8],
        width: u32,
        height: u32,
        strength: f32,
        radius: f32,
        color: [u8; 4],
    ) {
        let expected = (width as usize) * (height as usize) * 4;
        assert_eq!(target.len(), expected, "target buffer size mismatch");

        if width == 0 || height == 0 {
            return;
        }

        let strength = strength.clamp(0.0, 1.0);
        let radius = radius.clamp(0.0, 1.0);

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        // Normalise distances so the corners are at distance 1.0
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..height {
            for x in 0..width {
                let dx = (x as f32 - cx) / max_dist;
                let dy = (y as f32 - cy) / max_dist;
                let dist = (dx * dx + dy * dy).sqrt();

                // How far outside the clear radius are we?
                let excess = ((dist - radius) / (1.0 - radius)).clamp(0.0, 1.0);
                // Smooth the falloff using a squared curve
                let blend = strength * excess * excess;

                let mut vig_color = color;
                vig_color[3] = (blend * f32::from(color[3])) as u8;

                let idx = ((y * width + x) * 4) as usize;
                let existing = [
                    target[idx],
                    target[idx + 1],
                    target[idx + 2],
                    target[idx + 3],
                ];
                let blended = alpha_over(existing, vig_color);
                target[idx..idx + 4].copy_from_slice(&blended);
            }
        }
    }

    /// Composite `top` over `bottom` through a greyscale mask.
    ///
    /// The mask is a single-channel (greyscale) buffer where 255 = fully composite
    /// the top layer and 0 = show only the bottom layer.
    ///
    /// All RGBA buffers must be `width * height * 4` bytes.
    /// The mask buffer must be `width * height` bytes (one byte per pixel).
    ///
    /// # Panics
    ///
    /// Panics if any buffer has the wrong length.
    pub fn composite_masked(
        bottom: &[u8],
        top: &[u8],
        mask: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        mode: BlendMode,
    ) {
        let n = (width as usize) * (height as usize);
        let rgba_expected = n * 4;
        assert_eq!(bottom.len(), rgba_expected, "bottom buffer size mismatch");
        assert_eq!(top.len(), rgba_expected, "top buffer size mismatch");
        assert_eq!(mask.len(), n, "mask buffer size mismatch");
        assert_eq!(output.len(), rgba_expected, "output buffer size mismatch");

        for (chunk_idx, out_chunk) in output.chunks_exact_mut(4).enumerate() {
            let base = chunk_idx * 4;
            let b = [
                bottom[base],
                bottom[base + 1],
                bottom[base + 2],
                bottom[base + 3],
            ];
            let t = [top[base], top[base + 1], top[base + 2], top[base + 3]];

            // Scale opacity by mask value
            let opacity = f32::from(mask[chunk_idx]) / 255.0;
            let r = blend_pixels(b, t, mode, opacity);
            out_chunk.copy_from_slice(&r);
        }
    }

    /// Flatten a stack of RGBA layers onto `output`.
    ///
    /// Layers are composited in order (index 0 = bottom) using `BlendMode::Normal`
    /// with each layer's individual opacity.
    ///
    /// `layers` is a slice of `(&[u8], f32)` tuples: (RGBA buffer, opacity).
    /// All buffers must be `width * height * 4` bytes.
    ///
    /// # Panics
    ///
    /// Panics if any buffer has the wrong length or `layers` is empty.
    pub fn flatten_layers(layers: &[(&[u8], f32)], output: &mut [u8], width: u32, height: u32) {
        assert!(!layers.is_empty(), "At least one layer is required");
        let expected = (width as usize) * (height as usize) * 4;
        assert_eq!(output.len(), expected, "output buffer size mismatch");

        // Seed output with the bottom layer
        let (first, first_opacity) = layers[0];
        assert_eq!(first.len(), expected, "layer 0 buffer size mismatch");
        for (i, chunk) in output.chunks_exact_mut(4).enumerate() {
            let base = i * 4;
            let p = [
                first[base],
                first[base + 1],
                first[base + 2],
                first[base + 3],
            ];
            let alpha = (f32::from(p[3]) * first_opacity).clamp(0.0, 255.0) as u8;
            chunk.copy_from_slice(&[p[0], p[1], p[2], alpha]);
        }

        // Composite remaining layers
        let n_pixels = (width as usize) * (height as usize);
        for (layer_buf, opacity) in &layers[1..] {
            assert_eq!(layer_buf.len(), expected, "layer buffer size mismatch");
            for i in 0..n_pixels {
                let base = i * 4;
                let bottom = [
                    output[base],
                    output[base + 1],
                    output[base + 2],
                    output[base + 3],
                ];
                let top = [
                    layer_buf[base],
                    layer_buf[base + 1],
                    layer_buf[base + 2],
                    layer_buf[base + 3],
                ];
                let blended = blend_pixels(bottom, top, BlendMode::Normal, *opacity);
                output[base..base + 4].copy_from_slice(&blended);
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Standard alpha-over composite: place `src` over `dst`.
#[inline]
fn alpha_over(dst: [u8; 4], src: [u8; 4]) -> [u8; 4] {
    let sa = f32::from(src[3]) / 255.0;
    let da = f32::from(dst[3]) / 255.0;
    let inv_sa = 1.0 - sa;
    let out_a = sa + da * inv_sa;
    if out_a < 1e-6 {
        return [0, 0, 0, 0];
    }
    [
        ((f32::from(src[0]) * sa + f32::from(dst[0]) * da * inv_sa) / out_a).clamp(0.0, 255.0)
            as u8,
        ((f32::from(src[1]) * sa + f32::from(dst[1]) * da * inv_sa) / out_a).clamp(0.0, 255.0)
            as u8,
        ((f32::from(src[2]) * sa + f32::from(dst[2]) * da * inv_sa) / out_a).clamp(0.0, 255.0)
            as u8,
        (out_a * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// Linearly interpolate between two RGBA colors.
#[inline]
fn interpolate_color(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    [
        lerp_u8(a[0], b[0], t),
        lerp_u8(a[1], b[1], t),
        lerp_u8(a[2], b[2], t),
        lerp_u8(a[3], b[3], t),
    ]
}

#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (f32::from(a) + (f32::from(b) - f32::from(a)) * t).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.extend_from_slice(&color);
        }
        buf
    }

    #[test]
    fn test_composite_normal_full_opacity() {
        let bottom = solid(4, 4, [255, 0, 0, 255]);
        let top = solid(4, 4, [0, 255, 0, 255]);
        let mut out = vec![0u8; 4 * 4 * 4];
        Compositor::composite(&bottom, &top, &mut out, 4, 4, BlendMode::Normal, 1.0);
        // Normal blend at full opacity → output should equal top
        assert_eq!(&out[0..4], &[0, 255, 0, 255]);
    }

    #[test]
    fn test_composite_zero_opacity() {
        let bottom = solid(4, 4, [255, 0, 0, 255]);
        let top = solid(4, 4, [0, 255, 0, 255]);
        let mut out = vec![0u8; 4 * 4 * 4];
        Compositor::composite(&bottom, &top, &mut out, 4, 4, BlendMode::Normal, 0.0);
        // Zero opacity → output should equal bottom
        assert_eq!(&out[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn test_composite_multiply() {
        let bottom = solid(2, 2, [200, 200, 200, 255]);
        let top = solid(2, 2, [128, 128, 128, 255]);
        let mut out = vec![0u8; 2 * 2 * 4];
        Compositor::composite(&bottom, &top, &mut out, 2, 2, BlendMode::Multiply, 1.0);
        // Multiply should darken
        assert!(
            out[0] < 200,
            "Multiply should darken red channel, got {}",
            out[0]
        );
    }

    #[test]
    fn test_gradient_overlay_horizontal() {
        let w: u32 = 64;
        let h: u32 = 32;
        let mut buf = vec![255u8; (w * h * 4) as usize];
        // Make alpha=255 so gradient blends correctly
        for i in 0..buf.len() / 4 {
            buf[i * 4 + 3] = 255;
        }
        Compositor::gradient_overlay(
            &mut buf,
            w,
            h,
            [255, 0, 0, 200], // start: red
            [0, 0, 255, 200], // end: blue
            0.0,              // angle: left→right
        );
        // Left-most pixels should be more red than blue
        let left_r = buf[0];
        let left_b = buf[2];
        // Right-most pixels should be more blue than red
        let right_base = ((h / 2 * w + w - 1) * 4) as usize;
        let right_r = buf[right_base];
        let right_b = buf[right_base + 2];
        assert!(
            left_r > right_r,
            "Left edge should be more red: {left_r} vs {right_r}"
        );
        assert!(
            right_b > left_b,
            "Right edge should be more blue: {right_b} vs {left_b}"
        );
    }

    #[test]
    fn test_gradient_overlay_no_panic_zero_dims() {
        // Should early-return without panicking
        let mut buf: Vec<u8> = Vec::new();
        Compositor::gradient_overlay(&mut buf, 0, 0, [0; 4], [0; 4], 45.0);
    }

    #[test]
    fn test_vignette_darkens_corners() {
        let w: u32 = 32;
        let h: u32 = 32;
        // Start with a fully white image
        let mut buf = solid(w, h, [255, 255, 255, 255]);
        Compositor::vignette(&mut buf, w, h, 1.0, 0.0, [0, 0, 0, 255]);

        // Centre pixel should be brighter than corner pixel
        let centre_idx = ((h / 2 * w + w / 2) * 4) as usize;
        let corner_idx = 0;
        let centre_r = buf[centre_idx];
        let corner_r = buf[corner_idx];
        assert!(
            centre_r > corner_r,
            "Centre ({centre_r}) should be brighter than corner ({corner_r})"
        );
    }

    #[test]
    fn test_vignette_zero_strength() {
        let w: u32 = 16;
        let h: u32 = 16;
        let original = solid(w, h, [200, 150, 100, 255]);
        let mut buf = original.clone();
        Compositor::vignette(&mut buf, w, h, 0.0, 0.5, [0, 0, 0, 255]);
        // Zero strength → no change
        assert_eq!(buf, original);
    }

    #[test]
    fn test_alpha_over_transparent_src() {
        let dst = [100u8, 150, 200, 255];
        let src = [0u8, 0, 0, 0]; // fully transparent
        let result = alpha_over(dst, src);
        // Transparent source → result equals destination
        assert_eq!(result, dst);
    }

    #[test]
    fn test_alpha_over_opaque_src() {
        let dst = [255u8, 0, 0, 255]; // red
        let src = [0u8, 255, 0, 255]; // green, opaque
        let result = alpha_over(dst, src);
        // Opaque source → result should be green
        assert_eq!(result[1], 255);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn test_interpolate_color() {
        let a = [0u8, 0, 0, 0];
        let b = [200u8, 100, 50, 255];
        let mid = interpolate_color(a, b, 0.5);
        assert_eq!(mid[0], 100);
        assert_eq!(mid[3], 127);
    }

    // ── New masked compositing tests ──────────────────────────────────────────

    #[test]
    fn test_composite_masked_full_mask_shows_top() {
        let bottom = solid(4, 4, [255, 0, 0, 255]); // red
        let top = solid(4, 4, [0, 255, 0, 255]); // green
        let mask = vec![255u8; 4 * 4]; // fully reveal top
        let mut out = vec![0u8; 4 * 4 * 4];
        Compositor::composite_masked(&bottom, &top, &mask, &mut out, 4, 4, BlendMode::Normal);
        // Full mask → top should dominate
        assert!(out[1] > 200, "Green should dominate with full mask");
    }

    #[test]
    fn test_composite_masked_zero_mask_shows_bottom() {
        let bottom = solid(4, 4, [255, 0, 0, 255]); // red
        let top = solid(4, 4, [0, 255, 0, 255]); // green
        let mask = vec![0u8; 4 * 4]; // fully hide top
        let mut out = vec![0u8; 4 * 4 * 4];
        Compositor::composite_masked(&bottom, &top, &mask, &mut out, 4, 4, BlendMode::Normal);
        // Zero mask → bottom should show
        assert_eq!(out[0], 255, "Red bottom should show when mask is 0");
    }

    #[test]
    fn test_flatten_layers_single_layer() {
        let layer = solid(4, 4, [100, 150, 200, 255]);
        let layers: Vec<(&[u8], f32)> = vec![(&layer, 1.0)];
        let mut out = vec![0u8; 4 * 4 * 4];
        Compositor::flatten_layers(&layers, &mut out, 4, 4);
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 150);
        assert_eq!(out[2], 200);
    }

    #[test]
    fn test_flatten_layers_two_layers_opacity() {
        let bottom = solid(4, 4, [255, 0, 0, 255]); // red opaque
        let top = solid(4, 4, [0, 255, 0, 255]); // green
        let layers: Vec<(&[u8], f32)> = vec![(&bottom, 1.0), (&top, 0.5)];
        let mut out = vec![0u8; 4 * 4 * 4];
        Compositor::flatten_layers(&layers, &mut out, 4, 4);
        // At 50% opacity the green layer blends with red
        assert!(out[1] > 50, "Some green should show through");
        assert!(out[0] > 0, "Some red should remain");
    }

    #[test]
    fn test_flatten_layers_full_top_opacity_replaces_bottom() {
        let bottom = solid(4, 4, [255, 0, 0, 255]); // red
        let top = solid(4, 4, [0, 0, 255, 255]); // blue
        let layers: Vec<(&[u8], f32)> = vec![(&bottom, 1.0), (&top, 1.0)];
        let mut out = vec![0u8; 4 * 4 * 4];
        Compositor::flatten_layers(&layers, &mut out, 4, 4);
        // Full top layer opacity, opaque blue → blue dominates
        assert!(out[2] > 200, "Blue should dominate at 1.0 opacity");
    }

    #[test]
    fn test_composite_masked_half_mask_blends() {
        let bottom = solid(2, 2, [200, 0, 0, 255]);
        let top = solid(2, 2, [0, 200, 0, 255]);
        let mask = vec![128u8; 2 * 2]; // ~50% reveal
        let mut out = vec![0u8; 2 * 2 * 4];
        Compositor::composite_masked(&bottom, &top, &mask, &mut out, 2, 2, BlendMode::Normal);
        // Both channels should be present
        assert!(out[0] > 0, "Some red should remain");
        assert!(out[1] > 0, "Some green should show");
    }
}
