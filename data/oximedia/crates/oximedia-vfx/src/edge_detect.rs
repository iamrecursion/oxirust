#![allow(dead_code)]
//! Edge detection filters for VFX.
//!
//! Implements Sobel, Prewitt, and Laplacian edge detection operators
//! that operate on RGBA frame buffers. Useful for stylization,
//! outline effects, and analytical preprocessing.

/// Edge detection algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeAlgorithm {
    /// Sobel operator (3x3 gradient magnitude).
    Sobel,
    /// Prewitt operator (3x3 gradient magnitude).
    Prewitt,
    /// Laplacian operator (3x3 second-derivative).
    Laplacian,
    /// Roberts cross operator (2x2 diagonal gradient).
    Roberts,
}

/// Configuration for edge detection.
#[derive(Debug, Clone)]
pub struct EdgeDetectConfig {
    /// Algorithm to use.
    pub algorithm: EdgeAlgorithm,
    /// Output intensity scale (1.0 = normal).
    pub intensity: f32,
    /// Whether to invert the output (white edges on black vs black edges on white).
    pub invert: bool,
    /// Whether to output grayscale (true) or preserve original colour at edge pixels (false).
    pub grayscale: bool,
    /// Threshold below which edge magnitude is zeroed (0..255).
    pub threshold: u8,
}

impl Default for EdgeDetectConfig {
    fn default() -> Self {
        Self {
            algorithm: EdgeAlgorithm::Sobel,
            intensity: 1.0,
            invert: false,
            grayscale: true,
            threshold: 0,
        }
    }
}

/// Convert an RGBA pixel to luminance (0..255).
#[allow(clippy::cast_precision_loss)]
fn luminance(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Get luminance from a buffer at (x, y), clamping to edges.
#[allow(clippy::cast_precision_loss)]
fn get_lum(buf: &[u8], width: usize, height: usize, x: isize, y: isize) -> f32 {
    let cx = x.clamp(0, (width as isize) - 1) as usize;
    let cy = y.clamp(0, (height as isize) - 1) as usize;
    let idx = (cy * width + cx) * 4;
    luminance(buf[idx], buf[idx + 1], buf[idx + 2])
}

/// Compute Sobel gradient magnitude at a pixel.
fn sobel_magnitude(buf: &[u8], width: usize, height: usize, x: usize, y: usize) -> f32 {
    let ix = x as isize;
    let iy = y as isize;

    let gx = -get_lum(buf, width, height, ix - 1, iy - 1)
        + get_lum(buf, width, height, ix + 1, iy - 1)
        - 2.0 * get_lum(buf, width, height, ix - 1, iy)
        + 2.0 * get_lum(buf, width, height, ix + 1, iy)
        - get_lum(buf, width, height, ix - 1, iy + 1)
        + get_lum(buf, width, height, ix + 1, iy + 1);

    let gy = -get_lum(buf, width, height, ix - 1, iy - 1)
        - 2.0 * get_lum(buf, width, height, ix, iy - 1)
        - get_lum(buf, width, height, ix + 1, iy - 1)
        + get_lum(buf, width, height, ix - 1, iy + 1)
        + 2.0 * get_lum(buf, width, height, ix, iy + 1)
        + get_lum(buf, width, height, ix + 1, iy + 1);

    (gx * gx + gy * gy).sqrt()
}

/// Compute Prewitt gradient magnitude at a pixel.
fn prewitt_magnitude(buf: &[u8], width: usize, height: usize, x: usize, y: usize) -> f32 {
    let ix = x as isize;
    let iy = y as isize;

    let gx = -get_lum(buf, width, height, ix - 1, iy - 1)
        + get_lum(buf, width, height, ix + 1, iy - 1)
        - get_lum(buf, width, height, ix - 1, iy)
        + get_lum(buf, width, height, ix + 1, iy)
        - get_lum(buf, width, height, ix - 1, iy + 1)
        + get_lum(buf, width, height, ix + 1, iy + 1);

    let gy = -get_lum(buf, width, height, ix - 1, iy - 1)
        - get_lum(buf, width, height, ix, iy - 1)
        - get_lum(buf, width, height, ix + 1, iy - 1)
        + get_lum(buf, width, height, ix - 1, iy + 1)
        + get_lum(buf, width, height, ix, iy + 1)
        + get_lum(buf, width, height, ix + 1, iy + 1);

    (gx * gx + gy * gy).sqrt()
}

/// Compute Laplacian at a pixel.
fn laplacian_magnitude(buf: &[u8], width: usize, height: usize, x: usize, y: usize) -> f32 {
    let ix = x as isize;
    let iy = y as isize;

    let center = get_lum(buf, width, height, ix, iy);
    let neighbours = get_lum(buf, width, height, ix - 1, iy)
        + get_lum(buf, width, height, ix + 1, iy)
        + get_lum(buf, width, height, ix, iy - 1)
        + get_lum(buf, width, height, ix, iy + 1);

    (neighbours - 4.0 * center).abs()
}

/// Compute Roberts cross gradient magnitude at a pixel.
fn roberts_magnitude(buf: &[u8], width: usize, height: usize, x: usize, y: usize) -> f32 {
    let ix = x as isize;
    let iy = y as isize;

    let p00 = get_lum(buf, width, height, ix, iy);
    let p11 = get_lum(buf, width, height, ix + 1, iy + 1);
    let p10 = get_lum(buf, width, height, ix + 1, iy);
    let p01 = get_lum(buf, width, height, ix, iy + 1);

    let gx = p00 - p11;
    let gy = p10 - p01;
    (gx * gx + gy * gy).sqrt()
}

/// Edge detection processor.
#[derive(Debug, Clone)]
pub struct EdgeDetect {
    /// Configuration.
    config: EdgeDetectConfig,
}

impl EdgeDetect {
    /// Create a new edge detector with default settings.
    pub fn new() -> Self {
        Self {
            config: EdgeDetectConfig::default(),
        }
    }

    /// Create with a specific configuration.
    pub fn with_config(config: EdgeDetectConfig) -> Self {
        Self { config }
    }

    /// Get configuration reference.
    pub fn config(&self) -> &EdgeDetectConfig {
        &self.config
    }

    /// Set configuration.
    pub fn set_config(&mut self, config: EdgeDetectConfig) {
        self.config = config;
    }

    /// Compute edge magnitude at a single pixel.
    fn edge_magnitude(&self, buf: &[u8], width: usize, height: usize, x: usize, y: usize) -> f32 {
        match self.config.algorithm {
            EdgeAlgorithm::Sobel => sobel_magnitude(buf, width, height, x, y),
            EdgeAlgorithm::Prewitt => prewitt_magnitude(buf, width, height, x, y),
            EdgeAlgorithm::Laplacian => laplacian_magnitude(buf, width, height, x, y),
            EdgeAlgorithm::Roberts => roberts_magnitude(buf, width, height, x, y),
        }
    }

    /// Apply edge detection to an RGBA buffer, writing to `dst`.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(&self, src: &[u8], dst: &mut [u8], width: u32, height: u32) {
        let w = width as usize;
        let h = height as usize;
        let expected = w * h * 4;
        if src.len() < expected || dst.len() < expected {
            return;
        }

        for y in 0..h {
            for x in 0..w {
                let mag = self.edge_magnitude(src, w, h, x, y) * self.config.intensity;
                let mut val = mag.clamp(0.0, 255.0) as u8;

                if val < self.config.threshold {
                    val = 0;
                }

                if self.config.invert {
                    val = 255 - val;
                }

                let dst_idx = (y * w + x) * 4;
                let src_idx = (y * w + x) * 4;

                if self.config.grayscale {
                    dst[dst_idx] = val;
                    dst[dst_idx + 1] = val;
                    dst[dst_idx + 2] = val;
                } else {
                    // Multiply original colour by edge strength
                    let scale = val as f32 / 255.0;
                    dst[dst_idx] = (src[src_idx] as f32 * scale).clamp(0.0, 255.0) as u8;
                    dst[dst_idx + 1] = (src[src_idx + 1] as f32 * scale).clamp(0.0, 255.0) as u8;
                    dst[dst_idx + 2] = (src[src_idx + 2] as f32 * scale).clamp(0.0, 255.0) as u8;
                }
                dst[dst_idx + 3] = src[src_idx + 3]; // preserve alpha
            }
        }
    }
}

impl Default for EdgeDetect {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_buffer(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height * 4]
    }

    fn make_gradient_buffer(width: usize, height: usize) -> Vec<u8> {
        let mut buf = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                #[allow(clippy::cast_possible_truncation)]
                let v = (x * 255 / width.max(1)) as u8;
                buf[idx] = v;
                buf[idx + 1] = v;
                buf[idx + 2] = v;
                buf[idx + 3] = 255;
            }
        }
        buf
    }

    #[test]
    fn test_luminance() {
        let l = luminance(255, 255, 255);
        assert!((l - 255.0).abs() < 0.01);
        let l = luminance(0, 0, 0);
        assert!(l.abs() < 0.01);
    }

    #[test]
    fn test_edge_detect_flat_image() {
        // A flat image should produce zero edges
        let w = 8usize;
        let h = 8usize;
        let src = make_flat_buffer(w, h, 128);
        let mut dst = vec![0u8; w * h * 4];
        let ed = EdgeDetect::new();
        ed.apply(&src, &mut dst, w as u32, h as u32);
        // All edge values should be 0
        for chunk in dst.chunks_exact(4) {
            assert_eq!(chunk[0], 0);
            assert_eq!(chunk[1], 0);
            assert_eq!(chunk[2], 0);
        }
    }

    #[test]
    fn test_edge_detect_gradient_image() {
        let w = 16usize;
        let h = 16usize;
        let src = make_gradient_buffer(w, h);
        let mut dst = vec![0u8; w * h * 4];
        let ed = EdgeDetect::new();
        ed.apply(&src, &mut dst, w as u32, h as u32);
        // Interior pixels should detect horizontal edges
        let mid_idx = (8 * w + 8) * 4;
        assert!(dst[mid_idx] > 0);
    }

    #[test]
    fn test_sobel_algorithm() {
        let cfg = EdgeDetectConfig {
            algorithm: EdgeAlgorithm::Sobel,
            ..Default::default()
        };
        let ed = EdgeDetect::with_config(cfg);
        assert_eq!(ed.config().algorithm, EdgeAlgorithm::Sobel);
    }

    #[test]
    fn test_prewitt_algorithm() {
        let w = 16usize;
        let h = 16usize;
        let src = make_gradient_buffer(w, h);
        let mut dst = vec![0u8; w * h * 4];
        let cfg = EdgeDetectConfig {
            algorithm: EdgeAlgorithm::Prewitt,
            ..Default::default()
        };
        let ed = EdgeDetect::with_config(cfg);
        ed.apply(&src, &mut dst, w as u32, h as u32);
        let mid_idx = (8 * w + 8) * 4;
        assert!(dst[mid_idx] > 0);
    }

    #[test]
    fn test_laplacian_algorithm() {
        let w = 16usize;
        let h = 16usize;
        let src = make_gradient_buffer(w, h);
        let mut dst = vec![0u8; w * h * 4];
        let cfg = EdgeDetectConfig {
            algorithm: EdgeAlgorithm::Laplacian,
            ..Default::default()
        };
        let ed = EdgeDetect::with_config(cfg);
        ed.apply(&src, &mut dst, w as u32, h as u32);
        // Laplacian on linear gradient may be near-zero, but should not crash
        assert_eq!(dst.len(), w * h * 4);
    }

    #[test]
    fn test_roberts_algorithm() {
        let w = 16usize;
        let h = 16usize;
        let src = make_gradient_buffer(w, h);
        let mut dst = vec![0u8; w * h * 4];
        let cfg = EdgeDetectConfig {
            algorithm: EdgeAlgorithm::Roberts,
            ..Default::default()
        };
        let ed = EdgeDetect::with_config(cfg);
        ed.apply(&src, &mut dst, w as u32, h as u32);
        assert_eq!(dst.len(), w * h * 4);
    }

    #[test]
    fn test_invert_flag() {
        let w = 8usize;
        let h = 8usize;
        let src = make_flat_buffer(w, h, 128);
        let mut dst = vec![0u8; w * h * 4];
        let cfg = EdgeDetectConfig {
            invert: true,
            ..Default::default()
        };
        let ed = EdgeDetect::with_config(cfg);
        ed.apply(&src, &mut dst, w as u32, h as u32);
        // Flat image with invert: edges are 0, inverted = 255
        for chunk in dst.chunks_exact(4) {
            assert_eq!(chunk[0], 255);
        }
    }

    #[test]
    fn test_threshold() {
        let w = 16usize;
        let h = 16usize;
        let src = make_gradient_buffer(w, h);
        let mut dst = vec![0u8; w * h * 4];
        let cfg = EdgeDetectConfig {
            threshold: 200,
            ..Default::default()
        };
        let ed = EdgeDetect::with_config(cfg);
        ed.apply(&src, &mut dst, w as u32, h as u32);
        // High threshold should suppress weak edges
        let non_zero_count = dst.chunks_exact(4).filter(|c| c[0] > 0).count();
        // Most pixels should be zeroed with high threshold
        assert!(non_zero_count < w * h);
    }

    #[test]
    fn test_alpha_preservation() {
        let w = 4usize;
        let h = 4usize;
        let mut src = make_gradient_buffer(w, h);
        // Set specific alpha values
        for chunk in src.chunks_exact_mut(4) {
            chunk[3] = 200;
        }
        let mut dst = vec![0u8; w * h * 4];
        let ed = EdgeDetect::new();
        ed.apply(&src, &mut dst, w as u32, h as u32);
        for chunk in dst.chunks_exact(4) {
            assert_eq!(chunk[3], 200);
        }
    }

    #[test]
    fn test_intensity_scaling() {
        let w = 16usize;
        let h = 16usize;
        let src = make_gradient_buffer(w, h);

        let mut dst_normal = vec![0u8; w * h * 4];
        let ed_normal = EdgeDetect::with_config(EdgeDetectConfig {
            intensity: 1.0,
            ..Default::default()
        });
        ed_normal.apply(&src, &mut dst_normal, w as u32, h as u32);

        let mut dst_boosted = vec![0u8; w * h * 4];
        let ed_boosted = EdgeDetect::with_config(EdgeDetectConfig {
            intensity: 2.0,
            ..Default::default()
        });
        ed_boosted.apply(&src, &mut dst_boosted, w as u32, h as u32);

        // Boosted should have higher or equal values
        let sum_normal: u64 = dst_normal.chunks_exact(4).map(|c| c[0] as u64).sum();
        let sum_boosted: u64 = dst_boosted.chunks_exact(4).map(|c| c[0] as u64).sum();
        assert!(sum_boosted >= sum_normal);
    }

    #[test]
    fn test_undersized_buffer_no_panic() {
        let ed = EdgeDetect::new();
        let mut dst = vec![0u8; 4];
        ed.apply(&[0u8; 4], &mut dst, 8, 8);
        assert_eq!(dst, vec![0u8; 4]);
    }
}
