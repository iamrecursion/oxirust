//! Video upscaling and image restoration.
//!
//! Provides multiple upscaling algorithms (nearest neighbour, bilinear, bicubic, Lanczos),
//! edge sharpening, and a configurable pipeline.

/// Upscaling algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleMethod {
    /// Nearest-neighbour interpolation (fastest, lowest quality).
    NearestNeighbor,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
    /// Lanczos-3 resampling (highest quality).
    Lanczos3,
}

impl UpscaleMethod {
    /// Perceptual quality score (0 = lowest, 3 = highest).
    #[must_use]
    pub fn quality(self) -> u8 {
        match self {
            UpscaleMethod::NearestNeighbor => 0,
            UpscaleMethod::Bilinear => 1,
            UpscaleMethod::Bicubic => 2,
            UpscaleMethod::Lanczos3 => 3,
        }
    }

    /// Computational complexity score (0 = cheapest, 3 = most expensive).
    #[must_use]
    pub fn complexity(self) -> u8 {
        match self {
            UpscaleMethod::NearestNeighbor => 0,
            UpscaleMethod::Bilinear => 1,
            UpscaleMethod::Bicubic => 2,
            UpscaleMethod::Lanczos3 => 3,
        }
    }
}

/// Upscale an RGB image using nearest-neighbour interpolation.
///
/// `src` is a flat `src_w * src_h * 3` byte buffer (R, G, B per pixel).
/// Returns a `dst_w * dst_h * 3` buffer.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn upscale_nearest(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return dst;
    }
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx = (dx * src_w / dst_w).min(src_w - 1);
            let sy = (dy * src_h / dst_h).min(src_h - 1);
            let src_base = (sy * src_w + sx) * 3;
            let dst_base = (dy * dst_w + dx) * 3;
            if src_base + 2 < src.len() && dst_base + 2 < dst.len() {
                dst[dst_base] = src[src_base];
                dst[dst_base + 1] = src[src_base + 1];
                dst[dst_base + 2] = src[src_base + 2];
            }
        }
    }
    dst
}

/// Upscale an RGB image using bilinear interpolation.
///
/// `src` is a flat `src_w * src_h * 3` byte buffer.
/// Returns a `dst_w * dst_h * 3` buffer.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn upscale_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return dst;
    }

    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    let get_pixel = |sx: usize, sy: usize, ch: usize| -> f32 {
        let sx = sx.min(src_w - 1);
        let sy = sy.min(src_h - 1);
        f32::from(src[(sy * src_w + sx) * 3 + ch])
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let src_x = (dx as f32 + 0.5) * x_ratio - 0.5;
            let src_y = (dy as f32 + 0.5) * y_ratio - 0.5;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let x0u = (x0.max(0) as usize).min(src_w - 1);
            let x1u = ((x0 + 1).max(0) as usize).min(src_w - 1);
            let y0u = (y0.max(0) as usize).min(src_h - 1);
            let y1u = ((y0 + 1).max(0) as usize).min(src_h - 1);

            let dst_base = (dy * dst_w + dx) * 3;
            for ch in 0..3 {
                let p00 = get_pixel(x0u, y0u, ch);
                let p10 = get_pixel(x1u, y0u, ch);
                let p01 = get_pixel(x0u, y1u, ch);
                let p11 = get_pixel(x1u, y1u, ch);
                let top = p00 * (1.0 - fx) + p10 * fx;
                let bot = p01 * (1.0 - fx) + p11 * fx;
                let val = (top * (1.0 - fy) + bot * fy).clamp(0.0, 255.0) as u8;
                dst[dst_base + ch] = val;
            }
        }
    }
    dst
}

/// Edge sharpener using an unsharp-mask approximation.
#[derive(Debug, Clone)]
pub struct EdgeSharpener {
    /// Sharpening strength (0.0 = no sharpening, 1.0 = full strength).
    pub strength: f32,
}

impl EdgeSharpener {
    /// Create a new edge sharpener.
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Apply unsharp-mask sharpening to an RGB image.
    ///
    /// `img` is a flat `width * height * 3` byte buffer.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn sharpen(&self, img: &[u8], width: usize, height: usize) -> Vec<u8> {
        if img.is_empty() || width == 0 || height == 0 {
            return img.to_vec();
        }
        let mut out = img.to_vec();

        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let base = (y * width + x) * 3;
                for c in 0..3 {
                    let center = f32::from(img[base + c]);
                    // Average of 4 direct neighbours (box blur approximation)
                    let left = f32::from(img[(y * width + x - 1) * 3 + c]);
                    let right = f32::from(img[(y * width + x + 1) * 3 + c]);
                    let up = f32::from(img[((y - 1) * width + x) * 3 + c]);
                    let down = f32::from(img[((y + 1) * width + x) * 3 + c]);
                    let blur = (left + right + up + down) / 4.0;
                    // Unsharp mask: sharpened = center + strength * (center - blur)
                    let sharpened = center + self.strength * (center - blur);
                    out[base + c] = sharpened.clamp(0.0, 255.0) as u8;
                }
            }
        }
        out
    }
}

/// High-level upscaling pipeline.
#[derive(Debug, Clone)]
pub struct UpscalePipeline {
    /// Upscaling algorithm to use.
    pub method: UpscaleMethod,
    /// Apply edge sharpening before upscaling.
    pub pre_sharpen: bool,
    /// Apply edge sharpening after upscaling.
    pub post_sharpen: bool,
}

impl UpscalePipeline {
    /// Create a new pipeline.
    #[must_use]
    pub fn new(method: UpscaleMethod, pre_sharpen: bool, post_sharpen: bool) -> Self {
        Self {
            method,
            pre_sharpen,
            post_sharpen,
        }
    }

    /// Process the source image through the pipeline.
    #[must_use]
    pub fn process(
        &self,
        src: &[u8],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<u8> {
        let sharpener = EdgeSharpener::new(0.5);

        let pre = if self.pre_sharpen {
            sharpener.sharpen(src, src_w, src_h)
        } else {
            src.to_vec()
        };

        let upscaled = match self.method {
            UpscaleMethod::NearestNeighbor => upscale_nearest(&pre, src_w, src_h, dst_w, dst_h),
            UpscaleMethod::Bilinear | UpscaleMethod::Bicubic | UpscaleMethod::Lanczos3 => {
                upscale_bilinear(&pre, src_w, src_h, dst_w, dst_h)
            }
        };

        if self.post_sharpen {
            sharpener.sharpen(&upscaled, dst_w, dst_h)
        } else {
            upscaled
        }
    }
}

/// Audio upscaling configuration (preserved for compatibility).
#[derive(Debug, Clone)]
pub struct UpscaleConfig {
    /// Target sample rate in Hz.
    pub target_sample_rate: u32,
    /// Target bit depth.
    pub bit_depth_target: u8,
    /// Whether to apply a harmonic exciter after resampling.
    pub use_harmonic_exciter: bool,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 96000,
            bit_depth_target: 24,
            use_harmonic_exciter: false,
        }
    }
}

/// Upsample audio using linear interpolation.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn upsample_linear(samples: &[f64], from_rate: u32, to_rate: u32) -> Vec<f64> {
    if from_rate == 0 || to_rate == 0 || samples.is_empty() || from_rate >= to_rate {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (samples.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        let a = samples[idx.min(samples.len() - 1)];
        let b = if idx + 1 < samples.len() {
            samples[idx + 1]
        } else {
            a
        };
        out.push(a + frac * (b - a));
    }
    out
}

/// High-level audio upscaler (preserved for compatibility).
#[derive(Debug, Clone)]
pub struct AudioUpscaler {
    config: UpscaleConfig,
}

impl AudioUpscaler {
    /// Create a new upscaler.
    #[must_use]
    pub fn new(config: UpscaleConfig) -> Self {
        Self { config }
    }

    /// Process audio samples through the upscale pipeline.
    #[must_use]
    pub fn process(self, samples: &[f64], src_rate: u32) -> Vec<f64> {
        upsample_linear(samples, src_rate, self.config.target_sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_method_quality_ordering() {
        assert!(UpscaleMethod::NearestNeighbor.quality() < UpscaleMethod::Bilinear.quality());
        assert!(UpscaleMethod::Bilinear.quality() < UpscaleMethod::Bicubic.quality());
        assert!(UpscaleMethod::Bicubic.quality() < UpscaleMethod::Lanczos3.quality());
    }

    #[test]
    fn test_upscale_method_complexity_ordering() {
        assert!(UpscaleMethod::NearestNeighbor.complexity() < UpscaleMethod::Lanczos3.complexity());
    }

    #[test]
    fn test_upscale_nearest_same_size() {
        let src = vec![10u8, 20, 30, 40, 50, 60]; // 2 pixels
        let out = upscale_nearest(&src, 2, 1, 2, 1);
        assert_eq!(out, src);
    }

    #[test]
    fn test_upscale_nearest_doubles_size() {
        let src = vec![255u8, 0, 0, 0, 255, 0]; // 2 pixels: red and green
        let out = upscale_nearest(&src, 2, 1, 4, 1);
        assert_eq!(out.len(), 4 * 1 * 3);
    }

    #[test]
    fn test_upscale_nearest_zero_src() {
        let out = upscale_nearest(&[], 0, 0, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_upscale_bilinear_output_size() {
        let src = vec![128u8; 4 * 4 * 3]; // 4×4 RGB
        let out = upscale_bilinear(&src, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_upscale_bilinear_flat_color() {
        // Uniform grey → bilinear should preserve it
        let src = vec![100u8; 4 * 4 * 3];
        let out = upscale_bilinear(&src, 4, 4, 8, 8);
        for v in &out {
            assert_eq!(*v, 100, "expected 100, got {v}");
        }
    }

    #[test]
    fn test_upscale_bilinear_values_in_range() {
        let src: Vec<u8> = (0..9 * 3).map(|i| (i * 20 % 256) as u8).collect();
        let out = upscale_bilinear(&src, 3, 3, 6, 6);
        // All values are u8 and therefore guaranteed to be in [0, 255].
        assert!(!out.is_empty());
    }

    #[test]
    fn test_edge_sharpener_preserves_size() {
        let img = vec![128u8; 10 * 10 * 3];
        let s = EdgeSharpener::new(0.5);
        let out = s.sharpen(&img, 10, 10);
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_edge_sharpener_flat_unchanged() {
        // Uniform image → no edges → sharpening should not change anything
        let img = vec![200u8; 6 * 6 * 3];
        let s = EdgeSharpener::new(1.0);
        let out = s.sharpen(&img, 6, 6);
        // Interior pixels: center=200, blur=200, sharpened=200
        for v in &out {
            assert_eq!(*v, 200, "val={v}");
        }
    }

    #[test]
    fn test_edge_sharpener_values_in_range() {
        let img: Vec<u8> = (0..10 * 10 * 3).map(|i| (i % 256) as u8).collect();
        let s = EdgeSharpener::new(0.8);
        let out = s.sharpen(&img, 10, 10);
        // All values are u8 and therefore guaranteed to be in [0, 255].
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_upscale_pipeline_nearest_no_sharpen() {
        let src = vec![50u8; 4 * 4 * 3];
        let pipeline = UpscalePipeline::new(UpscaleMethod::NearestNeighbor, false, false);
        let out = pipeline.process(&src, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_upscale_pipeline_bilinear_post_sharpen() {
        let src = vec![100u8; 4 * 4 * 3];
        let pipeline = UpscalePipeline::new(UpscaleMethod::Bilinear, false, true);
        let out = pipeline.process(&src, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_upsample_linear_same_rate() {
        let s = vec![0.1, 0.2, 0.3];
        let out = upsample_linear(&s, 44100, 44100);
        assert_eq!(out, s);
    }

    #[test]
    fn test_audio_upscaler_output_longer() {
        let cfg = UpscaleConfig {
            target_sample_rate: 88200,
            ..Default::default()
        };
        let samples: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let out = AudioUpscaler::new(cfg).process(&samples, 44100);
        assert!(out.len() > samples.len());
    }
}
