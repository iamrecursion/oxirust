//! Automatic color matching between source and destination clips.
//!
//! Implements stats-based Reinhard color transfer.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Per-channel color statistics computed from a pixel buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorStats {
    /// Mean of red channel (0.0-255.0).
    pub mean_r: f32,
    /// Mean of green channel (0.0-255.0).
    pub mean_g: f32,
    /// Mean of blue channel (0.0-255.0).
    pub mean_b: f32,
    /// Standard deviation of red channel.
    pub std_r: f32,
    /// Standard deviation of green channel.
    pub std_g: f32,
    /// Standard deviation of blue channel.
    pub std_b: f32,
}

impl ColorStats {
    /// Compute color statistics from an interleaved RGB pixel buffer.
    ///
    /// `pixels` must be a slice of bytes in `[R, G, B, R, G, B, ...]` order.
    /// Returns a zeroed `ColorStats` if the slice is empty or not a multiple of 3.
    pub fn compute_from_pixels(pixels: &[u8]) -> Self {
        if pixels.len() < 3 || pixels.len() % 3 != 0 {
            return Self {
                mean_r: 0.0,
                mean_g: 0.0,
                mean_b: 0.0,
                std_r: 0.0,
                std_g: 0.0,
                std_b: 0.0,
            };
        }

        let n = (pixels.len() / 3) as f32;
        let mut sum_r = 0f64;
        let mut sum_g = 0f64;
        let mut sum_b = 0f64;

        for chunk in pixels.chunks_exact(3) {
            sum_r += f64::from(chunk[0]);
            sum_g += f64::from(chunk[1]);
            sum_b += f64::from(chunk[2]);
        }

        let mean_r = (sum_r / f64::from(n)) as f32;
        let mean_g = (sum_g / f64::from(n)) as f32;
        let mean_b = (sum_b / f64::from(n)) as f32;

        let mut var_r = 0f64;
        let mut var_g = 0f64;
        let mut var_b = 0f64;

        for chunk in pixels.chunks_exact(3) {
            let dr = f64::from(chunk[0]) - f64::from(mean_r);
            let dg = f64::from(chunk[1]) - f64::from(mean_g);
            let db = f64::from(chunk[2]) - f64::from(mean_b);
            var_r += dr * dr;
            var_g += dg * dg;
            var_b += db * db;
        }

        let std_r = ((var_r / f64::from(n)) as f32).sqrt();
        let std_g = ((var_g / f64::from(n)) as f32).sqrt();
        let std_b = ((var_b / f64::from(n)) as f32).sqrt();

        Self {
            mean_r,
            mean_g,
            mean_b,
            std_r,
            std_g,
            std_b,
        }
    }
}

/// A per-channel linear transfer (scale + offset) to match source statistics to destination.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorMatchTransfer {
    /// Multiplicative scale per channel `[R, G, B]`.
    pub scale: [f32; 3],
    /// Additive offset per channel `[R, G, B]`.
    pub offset: [f32; 3],
}

impl ColorMatchTransfer {
    /// Apply this transfer to a single RGB pixel (values in 0.0-255.0).
    ///
    /// Returns clamped `[R, G, B]` values in 0.0-255.0.
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        [
            (r * self.scale[0] + self.offset[0]).clamp(0.0, 255.0),
            (g * self.scale[1] + self.offset[1]).clamp(0.0, 255.0),
            (b * self.scale[2] + self.offset[2]).clamp(0.0, 255.0),
        ]
    }

    /// Return an identity transfer (no color change).
    pub fn identity() -> Self {
        Self {
            scale: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
        }
    }
}

/// Compute a Reinhard-style color transfer from `src` statistics to `dst` statistics.
///
/// For each channel:
/// `scale = dst.std / src.std` (or 1.0 when `src.std` is near zero)
/// `offset = dst.mean - scale * src.mean`
pub fn compute_color_transfer(src: &ColorStats, dst: &ColorStats) -> ColorMatchTransfer {
    const EPS: f32 = 1e-6;

    let scale_r = if src.std_r > EPS {
        dst.std_r / src.std_r
    } else {
        1.0
    };
    let scale_g = if src.std_g > EPS {
        dst.std_g / src.std_g
    } else {
        1.0
    };
    let scale_b = if src.std_b > EPS {
        dst.std_b / src.std_b
    } else {
        1.0
    };

    let offset_r = dst.mean_r - scale_r * src.mean_r;
    let offset_g = dst.mean_g - scale_g * src.mean_g;
    let offset_b = dst.mean_b - scale_b * src.mean_b;

    ColorMatchTransfer {
        scale: [scale_r, scale_g, scale_b],
        offset: [offset_r, offset_g, offset_b],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ColorStats tests ----

    #[test]
    fn test_compute_from_pixels_uniform() {
        // All pixels are (100, 150, 200) → mean == value, std == 0
        let pixels: Vec<u8> = vec![100, 150, 200, 100, 150, 200, 100, 150, 200];
        let stats = ColorStats::compute_from_pixels(&pixels);
        assert!((stats.mean_r - 100.0).abs() < 1e-3);
        assert!((stats.mean_g - 150.0).abs() < 1e-3);
        assert!((stats.mean_b - 200.0).abs() < 1e-3);
        assert!(stats.std_r < 1e-3);
        assert!(stats.std_g < 1e-3);
        assert!(stats.std_b < 1e-3);
    }

    #[test]
    fn test_compute_from_pixels_two_pixels() {
        // Two pixels: (0, 0, 0) and (200, 200, 200) → mean = 100, std ≈ 100
        let pixels: Vec<u8> = vec![0, 0, 0, 200, 200, 200];
        let stats = ColorStats::compute_from_pixels(&pixels);
        assert!((stats.mean_r - 100.0).abs() < 1e-3);
        assert!((stats.mean_g - 100.0).abs() < 1e-3);
        assert!((stats.mean_b - 100.0).abs() < 1e-3);
        assert!((stats.std_r - 100.0).abs() < 1e-2);
    }

    #[test]
    fn test_compute_from_pixels_empty() {
        let stats = ColorStats::compute_from_pixels(&[]);
        assert_eq!(stats.mean_r, 0.0);
        assert_eq!(stats.std_r, 0.0);
    }

    #[test]
    fn test_compute_from_pixels_invalid_length() {
        // Length not a multiple of 3
        let stats = ColorStats::compute_from_pixels(&[10, 20]);
        assert_eq!(stats.mean_r, 0.0);
    }

    #[test]
    fn test_compute_from_single_pixel() {
        let pixels: Vec<u8> = vec![255, 0, 128];
        let stats = ColorStats::compute_from_pixels(&pixels);
        assert!((stats.mean_r - 255.0).abs() < 1e-3);
        assert!((stats.mean_g - 0.0).abs() < 1e-3);
        assert!((stats.mean_b - 128.0).abs() < 1e-3);
    }

    // ---- ColorMatchTransfer tests ----

    #[test]
    fn test_identity_transfer_noop() {
        let xfer = ColorMatchTransfer::identity();
        let result = xfer.apply_pixel(100.0, 150.0, 200.0);
        assert!((result[0] - 100.0).abs() < 1e-4);
        assert!((result[1] - 150.0).abs() < 1e-4);
        assert!((result[2] - 200.0).abs() < 1e-4);
    }

    #[test]
    fn test_apply_pixel_clamps_high() {
        let xfer = ColorMatchTransfer {
            scale: [2.0, 2.0, 2.0],
            offset: [0.0, 0.0, 0.0],
        };
        let result = xfer.apply_pixel(200.0, 200.0, 200.0);
        // 200*2 = 400 → clamped to 255
        assert!((result[0] - 255.0).abs() < 1e-4);
    }

    #[test]
    fn test_apply_pixel_clamps_low() {
        let xfer = ColorMatchTransfer {
            scale: [1.0, 1.0, 1.0],
            offset: [-300.0, -300.0, -300.0],
        };
        let result = xfer.apply_pixel(50.0, 50.0, 50.0);
        assert!((result[0] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_apply_pixel_simple_scale_and_offset() {
        let xfer = ColorMatchTransfer {
            scale: [1.0, 0.5, 2.0],
            offset: [10.0, 5.0, -20.0],
        };
        let [r, g, b] = xfer.apply_pixel(50.0, 50.0, 50.0);
        assert!((r - 60.0).abs() < 1e-4); // 50*1 + 10
        assert!((g - 30.0).abs() < 1e-4); // 50*0.5 + 5
        assert!((b - 80.0).abs() < 1e-4); // 50*2 - 20
    }

    // ---- compute_color_transfer tests ----

    #[test]
    fn test_transfer_same_stats_is_identity() {
        let stats = ColorStats {
            mean_r: 128.0,
            mean_g: 100.0,
            mean_b: 90.0,
            std_r: 30.0,
            std_g: 25.0,
            std_b: 20.0,
        };
        let xfer = compute_color_transfer(&stats, &stats);
        // scale should be ~1.0, offset ~0.0
        assert!((xfer.scale[0] - 1.0).abs() < 1e-4);
        assert!((xfer.offset[0]).abs() < 1e-2);
    }

    #[test]
    fn test_transfer_zero_std_uses_identity_scale() {
        let src = ColorStats {
            mean_r: 100.0,
            mean_g: 100.0,
            mean_b: 100.0,
            std_r: 0.0,
            std_g: 0.0,
            std_b: 0.0,
        };
        let dst = ColorStats {
            mean_r: 200.0,
            mean_g: 200.0,
            mean_b: 200.0,
            std_r: 50.0,
            std_g: 50.0,
            std_b: 50.0,
        };
        let xfer = compute_color_transfer(&src, &dst);
        // scale falls back to 1.0 for zero std
        assert!((xfer.scale[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_transfer_scales_up() {
        let src = ColorStats {
            mean_r: 100.0,
            mean_g: 100.0,
            mean_b: 100.0,
            std_r: 10.0,
            std_g: 10.0,
            std_b: 10.0,
        };
        let dst = ColorStats {
            mean_r: 100.0,
            mean_g: 100.0,
            mean_b: 100.0,
            std_r: 20.0,
            std_g: 20.0,
            std_b: 20.0,
        };
        let xfer = compute_color_transfer(&src, &dst);
        assert!((xfer.scale[0] - 2.0).abs() < 1e-4);
        // offset = 100 - 2*100 = -100
        assert!((xfer.offset[0] + 100.0).abs() < 1e-2);
    }

    #[test]
    fn test_roundtrip_pixel_after_transfer() {
        // Source pixels: uniform red channel 80, green 120, blue 160
        let src_pixels: Vec<u8> = vec![80, 120, 160, 80, 120, 160, 80, 120, 160];
        let dst_pixels: Vec<u8> = vec![160, 60, 80, 160, 60, 80, 160, 60, 80];
        let src_stats = ColorStats::compute_from_pixels(&src_pixels);
        let dst_stats = ColorStats::compute_from_pixels(&dst_pixels);
        let xfer = compute_color_transfer(&src_stats, &dst_stats);
        let result = xfer.apply_pixel(80.0, 120.0, 160.0);
        // Result should be close to dst mean since both src and dst are uniform
        assert!((result[0] - 160.0).abs() < 1.0);
        assert!((result[1] - 60.0).abs() < 1.0);
        assert!((result[2] - 80.0).abs() < 1.0);
    }

    /// `compute_color_transfer` is a pure function: calling it twice with
    /// identical `ColorStats` arguments must produce bit-identical results.
    #[test]
    fn test_color_transfer_identical_inputs_deterministic() {
        let src = ColorStats {
            mean_r: 120.0,
            mean_g: 80.0,
            mean_b: 60.0,
            std_r: 25.0,
            std_g: 15.0,
            std_b: 10.0,
        };
        let dst = ColorStats {
            mean_r: 180.0,
            mean_g: 140.0,
            mean_b: 90.0,
            std_r: 40.0,
            std_g: 30.0,
            std_b: 20.0,
        };

        let xfer1 = compute_color_transfer(&src, &dst);
        let xfer2 = compute_color_transfer(&src, &dst);

        // All scale and offset values must be bit-identical (pure arithmetic)
        assert_eq!(
            xfer1.scale[0], xfer2.scale[0],
            "R scale must be deterministic"
        );
        assert_eq!(
            xfer1.scale[1], xfer2.scale[1],
            "G scale must be deterministic"
        );
        assert_eq!(
            xfer1.scale[2], xfer2.scale[2],
            "B scale must be deterministic"
        );
        assert_eq!(
            xfer1.offset[0], xfer2.offset[0],
            "R offset must be deterministic"
        );
        assert_eq!(
            xfer1.offset[1], xfer2.offset[1],
            "G offset must be deterministic"
        );
        assert_eq!(
            xfer1.offset[2], xfer2.offset[2],
            "B offset must be deterministic"
        );

        // Sanity-check: the transfer must actually shift the R channel upward
        // (src mean 120 → dst mean 180, so the transformed pixel should move up)
        let pixel = xfer1.apply_pixel(100.0, 70.0, 50.0);
        assert!(
            pixel[0] > 100.0,
            "R channel should shift toward dst mean (180 > 120)"
        );
    }
}
