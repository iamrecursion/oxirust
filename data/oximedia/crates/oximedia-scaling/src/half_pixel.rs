//! Sub-pixel accurate scaling with half-pixel correction.
//!
//! Different encoders and decoders use subtly different conventions for how
//! integer destination pixels map to fractional source coordinates.  Getting
//! this wrong introduces a systematic half-pixel shift that accumulates over
//! successive encode/decode cycles.
//!
//! This module implements three modes matching real-world usage:
//!
//! | Mode     | Description                                          |
//! |----------|------------------------------------------------------|
//! | `Center` | Sample centres aligned — standard MPEG convention    |
//! | `Edge`   | Outer-edge aligned — used by some scalers (OpenCV)   |
//! | `Legacy` | Top-left-aligned — simple but shifted                |

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ── HalfPixelMode ─────────────────────────────────────────────────────────────

/// Convention used to map destination pixel coordinates to source coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfPixelMode {
    /// Centre-of-pixel alignment (MPEG / ONNX standard).
    ///
    /// `src = (dst + 0.5) × (src_size / dst_size) − 0.5`
    ///
    /// This ensures the centres of the outer destination pixels map to the
    /// centres of the outer source pixels.
    Center,

    /// Edge-to-edge alignment (used by OpenCV and some other frameworks).
    ///
    /// `src = dst × (src_size − 1) / (dst_size − 1)`
    ///
    /// The first and last destination pixels map exactly to the first and last
    /// source pixels.
    Edge,

    /// Top-left / legacy alignment (naïve mapping).
    ///
    /// `src = dst × (src_size / dst_size)`
    ///
    /// Produces a systematic half-pixel offset compared to `Center`.
    Legacy,
}

// ── CoordinateMapper ──────────────────────────────────────────────────────────

/// Maps destination pixel coordinates to fractional source coordinates.
#[derive(Debug, Clone)]
pub struct CoordinateMapper {
    /// Source image dimensions `(width, height)`.
    pub src_size: (usize, usize),
    /// Destination image dimensions `(width, height)`.
    pub dst_size: (usize, usize),
    /// Coordinate mapping convention.
    pub mode: HalfPixelMode,
}

impl CoordinateMapper {
    /// Create a new coordinate mapper.
    #[must_use]
    pub fn new(src_size: (usize, usize), dst_size: (usize, usize), mode: HalfPixelMode) -> Self {
        Self {
            src_size,
            dst_size,
            mode,
        }
    }

    /// Map a destination pixel `(dst_x, dst_y)` to fractional source
    /// coordinates `(src_x, src_y)`.
    ///
    /// Floating-point destination coordinates allow sub-pixel queries.
    #[must_use]
    pub fn map_src_coord(&self, dst_x: f32, dst_y: f32) -> (f32, f32) {
        let src_w = self.src_size.0 as f32;
        let src_h = self.src_size.1 as f32;
        let dst_w = self.dst_size.0 as f32;
        let dst_h = self.dst_size.1 as f32;

        match self.mode {
            HalfPixelMode::Center => {
                let sx = (dst_x + 0.5) * (src_w / dst_w.max(1.0)) - 0.5;
                let sy = (dst_y + 0.5) * (src_h / dst_h.max(1.0)) - 0.5;
                (sx, sy)
            }

            HalfPixelMode::Edge => {
                let sx = if dst_w <= 1.0 {
                    0.0
                } else {
                    dst_x * (src_w - 1.0) / (dst_w - 1.0)
                };
                let sy = if dst_h <= 1.0 {
                    0.0
                } else {
                    dst_y * (src_h - 1.0) / (dst_h - 1.0)
                };
                (sx, sy)
            }

            HalfPixelMode::Legacy => {
                let sx = dst_x * (src_w / dst_w.max(1.0));
                let sy = dst_y * (src_h / dst_h.max(1.0));
                (sx, sy)
            }
        }
    }
}

// ── Interpolation primitives ──────────────────────────────────────────────────

/// Bilinear interpolation of a single-channel float image at fractional
/// coordinates `(x, y)`.
///
/// Coordinates outside the image are clamped.
#[must_use]
pub fn bilinear_interp(src: &[f32], x: f32, y: f32, width: usize, height: usize) -> f32 {
    if width == 0 || height == 0 || src.is_empty() {
        return 0.0;
    }

    let x0 = x.floor().clamp(0.0, (width - 1) as f32) as usize;
    let y0 = y.floor().clamp(0.0, (height - 1) as f32) as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let tx = (x - x.floor()).clamp(0.0, 1.0);
    let ty = (y - y.floor()).clamp(0.0, 1.0);

    let v00 = src[y0 * width + x0];
    let v10 = src[y0 * width + x1];
    let v01 = src[y1 * width + x0];
    let v11 = src[y1 * width + x1];

    let top = v00 * (1.0 - tx) + v10 * tx;
    let bot = v01 * (1.0 - tx) + v11 * tx;
    top * (1.0 - ty) + bot * ty
}

/// Cubic interpolation helper — Catmull-Rom weights for fractional `t`.
#[inline]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Sample a row of 4 values from a clamped 1-D array starting at `base_idx`.
#[inline]
fn sample_4(src: &[f32], base_idx: i64, stride: usize, count: usize, offset: usize) -> [f32; 4] {
    let mut values = [0.0f32; 4];
    for k in 0..4_i64 {
        let idx = (base_idx + k).clamp(0, count as i64 - 1) as usize;
        values[k as usize] = src[idx * stride + offset];
    }
    values
}

/// Bicubic (Catmull-Rom) interpolation of a single-channel float image at
/// fractional coordinates `(x, y)` using a 4×4 neighbourhood.
///
/// Coordinates outside the image are clamped.
#[must_use]
pub fn cubic_interp(src: &[f32], x: f32, y: f32, width: usize, height: usize) -> f32 {
    if width == 0 || height == 0 || src.is_empty() {
        return 0.0;
    }

    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let tx = x - ix as f32;
    let ty = y - iy as f32;

    let wx = catmull_rom_weights(tx);
    let wy = catmull_rom_weights(ty);

    let mut result = 0.0f32;
    for j in 0..4_i64 {
        let row = iy + j - 1;
        let row_clamped = row.clamp(0, height as i64 - 1) as usize;
        let cols = sample_4(src, ix - 1, 1, width, row_clamped * width);
        let row_val = wx[0] * cols[0] + wx[1] * cols[1] + wx[2] * cols[2] + wx[3] * cols[3];
        result += wy[j as usize] * row_val;
    }
    result
}

/// Bicubic interpolation alternative using per-row source addressing.
///
/// This variant explicitly builds the 4×4 neighbourhood from the 2-D
/// `src` buffer, properly clamping row and column indices.
#[must_use]
pub fn cubic_interp_2d(src: &[f32], x: f32, y: f32, width: usize, height: usize) -> f32 {
    if width == 0 || height == 0 || src.is_empty() {
        return 0.0;
    }

    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let tx = x - ix as f32;
    let ty = y - iy as f32;

    let wx = catmull_rom_weights(tx);
    let wy = catmull_rom_weights(ty);

    let clamp_x = |v: i64| v.clamp(0, width as i64 - 1) as usize;
    let clamp_y = |v: i64| v.clamp(0, height as i64 - 1) as usize;

    let mut result = 0.0f32;
    for j in 0..4_i64 {
        let row = clamp_y(iy + j - 1);
        let mut row_val = 0.0f32;
        for i in 0..4_i64 {
            let col = clamp_x(ix + i - 1);
            row_val += wx[i as usize] * src[row * width + col];
        }
        result += wy[j as usize] * row_val;
    }
    result
}

// ── ScaleKernel ───────────────────────────────────────────────────────────────

/// Which interpolation filter the `ScaleKernel` should apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleInterp {
    /// Bilinear 2×2 interpolation.
    Bilinear,
    /// Bicubic 4×4 Catmull-Rom interpolation.
    Cubic,
}

/// A complete scaling kernel combining coordinate mapping with interpolation.
#[derive(Debug, Clone)]
pub struct ScaleKernel {
    /// Coordinate mapping convention.
    pub mode: HalfPixelMode,
    /// Interpolation filter.
    pub interp: ScaleInterp,
}

impl ScaleKernel {
    /// Create a scale kernel.
    #[must_use]
    pub fn new(mode: HalfPixelMode, interp: ScaleInterp) -> Self {
        Self { mode, interp }
    }

    /// Scale a single-channel float image.
    ///
    /// - `src`: source pixels in row-major order.
    /// - `src_w`, `src_h`: source dimensions.
    /// - `dst_w`, `dst_h`: destination dimensions.
    ///
    /// Returns a `dst_w × dst_h` output buffer.
    #[must_use]
    pub fn apply(
        &self,
        src: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<f32> {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }

        let mapper = CoordinateMapper::new((src_w, src_h), (dst_w, dst_h), self.mode);
        let mut dst = vec![0.0f32; dst_w * dst_h];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let (sx, sy) = mapper.map_src_coord(dx as f32, dy as f32);
                dst[dy * dst_w + dx] = match self.interp {
                    ScaleInterp::Bilinear => bilinear_interp(src, sx, sy, src_w, src_h),
                    ScaleInterp::Cubic => cubic_interp_2d(src, sx, sy, src_w, src_h),
                };
            }
        }

        dst
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CoordinateMapper ──────────────────────────────────────────────────────

    #[test]
    fn center_mode_first_pixel_near_zero() {
        // With Center mode, dst pixel 0 should map to src around 0.
        let m = CoordinateMapper::new((4, 4), (4, 4), HalfPixelMode::Center);
        let (sx, sy) = m.map_src_coord(0.0, 0.0);
        // (0 + 0.5) * (4/4) - 0.5 = 0.5 - 0.5 = 0.0
        assert!((sx - 0.0).abs() < 1e-5, "Center x={sx}");
        assert!((sy - 0.0).abs() < 1e-5, "Center y={sy}");
    }

    #[test]
    fn center_mode_2x_upscale() {
        // 4→8: dst pixel 1 → src (1+0.5)*(4/8)-0.5 = 0.75 - 0.5 = 0.25
        let m = CoordinateMapper::new((4, 1), (8, 1), HalfPixelMode::Center);
        let (sx, _) = m.map_src_coord(1.0, 0.0);
        assert!((sx - 0.25).abs() < 1e-4, "Center 2x up dst=1 → src={sx}");
    }

    #[test]
    fn edge_mode_first_pixel_is_zero() {
        let m = CoordinateMapper::new((8, 8), (4, 4), HalfPixelMode::Edge);
        let (sx, sy) = m.map_src_coord(0.0, 0.0);
        assert!((sx).abs() < 1e-5, "Edge first x={sx}");
        assert!((sy).abs() < 1e-5, "Edge first y={sy}");
    }

    #[test]
    fn edge_mode_last_pixel_is_src_last() {
        let m = CoordinateMapper::new((8, 8), (4, 4), HalfPixelMode::Edge);
        let (sx, sy) = m.map_src_coord(3.0, 3.0);
        // dst=3 (last) → src = 3*(8-1)/(4-1) = 7.0
        assert!((sx - 7.0).abs() < 1e-4, "Edge last x={sx}");
        assert!((sy - 7.0).abs() < 1e-4, "Edge last y={sy}");
    }

    #[test]
    fn legacy_mode_zero_maps_to_zero() {
        let m = CoordinateMapper::new((8, 8), (4, 4), HalfPixelMode::Legacy);
        let (sx, sy) = m.map_src_coord(0.0, 0.0);
        assert!(sx.abs() < 1e-5);
        assert!(sy.abs() < 1e-5);
    }

    #[test]
    fn center_vs_legacy_differ_at_nonzero_pixel() {
        let mc = CoordinateMapper::new((4, 1), (8, 1), HalfPixelMode::Center);
        let ml = CoordinateMapper::new((4, 1), (8, 1), HalfPixelMode::Legacy);
        let (sx_c, _) = mc.map_src_coord(2.0, 0.0);
        let (sx_l, _) = ml.map_src_coord(2.0, 0.0);
        // They should differ
        assert!(
            (sx_c - sx_l).abs() > 0.01,
            "Center and Legacy should differ: center={sx_c} legacy={sx_l}"
        );
    }

    #[test]
    fn edge_vs_center_differ_at_nonzero_pixel() {
        let mc = CoordinateMapper::new((4, 1), (8, 1), HalfPixelMode::Center);
        let me = CoordinateMapper::new((4, 1), (8, 1), HalfPixelMode::Edge);
        let (sx_c, _) = mc.map_src_coord(3.0, 0.0);
        let (sx_e, _) = me.map_src_coord(3.0, 0.0);
        assert!(
            (sx_c - sx_e).abs() > 0.01,
            "Center and Edge should differ: center={sx_c} edge={sx_e}"
        );
    }

    // ── bilinear_interp ───────────────────────────────────────────────────────

    #[test]
    fn bilinear_interp_exact_pixel() {
        let src = vec![0.1f32, 0.2, 0.3, 0.4];
        let v = bilinear_interp(&src, 0.0, 0.0, 2, 2);
        assert!((v - 0.1).abs() < 1e-5, "bilinear at (0,0) = {v}");
    }

    #[test]
    fn bilinear_interp_midpoint() {
        // All four corners equal 0.5 → midpoint should be 0.5
        let src = vec![0.5f32; 4];
        let v = bilinear_interp(&src, 0.5, 0.5, 2, 2);
        assert!((v - 0.5).abs() < 1e-5);
    }

    #[test]
    fn bilinear_interp_clamped_oob() {
        let src = vec![1.0f32; 4];
        let v = bilinear_interp(&src, -5.0, -5.0, 2, 2);
        assert!((v - 1.0).abs() < 1e-4);
    }

    #[test]
    fn bilinear_interp_zero_dimensions() {
        assert_eq!(bilinear_interp(&[], 0.5, 0.5, 0, 4), 0.0);
        assert_eq!(bilinear_interp(&[], 0.5, 0.5, 4, 0), 0.0);
    }

    // ── cubic_interp_2d ───────────────────────────────────────────────────────

    #[test]
    fn cubic_interp_at_integer_pixel() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let v = cubic_interp_2d(&src, 2.0, 2.0, 4, 4);
        let expected = src[2 * 4 + 2];
        assert!(
            (v - expected).abs() < 0.01,
            "cubic at integer: {v} vs {expected}"
        );
    }

    #[test]
    fn cubic_interp_uniform_image() {
        let src = vec![0.5f32; 16];
        let v = cubic_interp_2d(&src, 1.5, 1.5, 4, 4);
        assert!((v - 0.5).abs() < 0.01, "cubic uniform: {v}");
    }

    // ── ScaleKernel ───────────────────────────────────────────────────────────

    #[test]
    fn scale_kernel_output_size_bilinear_center() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let k = ScaleKernel::new(HalfPixelMode::Center, ScaleInterp::Bilinear);
        let dst = k.apply(&src, 4, 4, 8, 8);
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn scale_kernel_output_size_cubic_edge() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let k = ScaleKernel::new(HalfPixelMode::Edge, ScaleInterp::Cubic);
        let dst = k.apply(&src, 4, 4, 8, 8);
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn scale_kernel_empty_returns_empty() {
        let k = ScaleKernel::new(HalfPixelMode::Center, ScaleInterp::Bilinear);
        assert!(k.apply(&[], 0, 0, 8, 8).is_empty());
        assert!(k.apply(&[], 4, 4, 0, 0).is_empty());
    }

    #[test]
    fn scale_kernel_uniform_image_stays_uniform() {
        let src = vec![0.6f32; 16];
        let k = ScaleKernel::new(HalfPixelMode::Center, ScaleInterp::Bilinear);
        let dst = k.apply(&src, 4, 4, 8, 8);
        for &v in &dst {
            assert!((v - 0.6).abs() < 0.001, "uniform: {v}");
        }
    }

    #[test]
    fn scale_kernel_center_vs_legacy_differ() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let k_c = ScaleKernel::new(HalfPixelMode::Center, ScaleInterp::Bilinear);
        let k_l = ScaleKernel::new(HalfPixelMode::Legacy, ScaleInterp::Bilinear);
        let dst_c = k_c.apply(&src, 4, 4, 8, 8);
        let dst_l = k_l.apply(&src, 4, 4, 8, 8);
        let diff: f32 = dst_c
            .iter()
            .zip(dst_l.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Center and Legacy should produce different outputs"
        );
    }

    #[test]
    fn scale_kernel_edge_mode_endpoints() {
        // Edge mode: first and last output pixels should match src boundary.
        let src: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let k = ScaleKernel::new(HalfPixelMode::Edge, ScaleInterp::Bilinear);
        // 1-D 4-sample → 8-sample scale.
        let dst = k.apply(&src, 4, 1, 8, 1);
        assert!(
            (dst[0] - src[0]).abs() < 0.01,
            "first edge pixel: {} vs {}",
            dst[0],
            src[0]
        );
        assert!(
            (dst[7] - src[3]).abs() < 0.01,
            "last edge pixel: {} vs {}",
            dst[7],
            src[3]
        );
    }

    #[test]
    fn scale_kernel_downscale_output_size() {
        let src = vec![0.5f32; 64];
        let k = ScaleKernel::new(HalfPixelMode::Center, ScaleInterp::Cubic);
        let dst = k.apply(&src, 8, 8, 4, 4);
        assert_eq!(dst.len(), 16);
    }
}
