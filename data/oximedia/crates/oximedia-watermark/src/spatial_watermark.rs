//! Spatial-domain invisible watermarking for image/video frames.
#![allow(dead_code)]

/// A spatial watermark pattern defined over a pixel grid.
#[derive(Debug, Clone)]
pub struct SpatialPattern {
    /// Pattern width in pixels.
    pub width: u32,
    /// Pattern height in pixels.
    pub height: u32,
    /// Normalised displacement values in [-1.0, 1.0] (one per pixel).
    pub values: Vec<f32>,
}

impl SpatialPattern {
    /// Create a pattern from a flat value vector.
    ///
    /// Returns `None` if `values.len() != width * height`.
    #[must_use]
    pub fn new(width: u32, height: u32, values: Vec<f32>) -> Option<Self> {
        if values.len() != (width as usize) * (height as usize) {
            return None;
        }
        Some(Self {
            width,
            height,
            values,
        })
    }

    /// Create a pattern filled with a constant value.
    #[must_use]
    pub fn constant(width: u32, height: u32, value: f32) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            values: vec![value; count],
        }
    }

    /// Total number of pixels in the pattern.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.values.len()
    }

    /// Value at pixel `(x, y)`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.values[(y as usize) * (self.width as usize) + (x as usize)])
    }
}

/// Configuration for spatial-domain watermarking.
#[derive(Debug, Clone)]
pub struct SpatialConfig {
    /// Embedding strength — maximum per-pixel luminance delta (0.0..1.0).
    pub strength: f32,
    /// Use tiling: repeat the pattern across the entire frame.
    pub tile: bool,
    /// If `true`, apply a perceptual mask (edges get stronger watermark).
    pub perceptual_mask: bool,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            strength: 0.02,
            tile: true,
            perceptual_mask: false,
        }
    }
}

impl SpatialConfig {
    /// Returns `true` if the config produces a robust watermark.
    ///
    /// Robustness is defined as strength > 0.05 OR perceptual masking enabled.
    #[must_use]
    pub fn is_robust(&self) -> bool {
        self.strength > 0.05 || self.perceptual_mask
    }
}

/// Invisible spatial watermarker operating on normalised luminance frames.
#[derive(Debug, Clone)]
pub struct SpatialWatermark {
    config: SpatialConfig,
    pattern: SpatialPattern,
}

impl SpatialWatermark {
    /// Create a new spatial watermarker.
    #[must_use]
    pub fn new(config: SpatialConfig, pattern: SpatialPattern) -> Self {
        Self { config, pattern }
    }

    /// Embed the watermark invisibly into `frame` (in-place).
    ///
    /// `frame` is a flat row-major slice of normalised [0.0, 1.0] luminance values
    /// with dimensions `(frame_width, frame_height)`.
    pub fn embed_invisible(&self, frame: &mut [f32], frame_width: u32, frame_height: u32) {
        let pw = self.pattern.width as usize;
        let ph = self.pattern.height as usize;
        let fw = frame_width as usize;
        let fh = frame_height as usize;

        for fy in 0..fh {
            for fx in 0..fw {
                let px = if self.config.tile {
                    fx % pw
                } else {
                    fx.min(pw - 1)
                };
                let py = if self.config.tile {
                    fy % ph
                } else {
                    fy.min(ph - 1)
                };
                let pattern_val = self.pattern.values[py * pw + px];
                let delta = pattern_val * self.config.strength;
                let idx = fy * fw + fx;
                frame[idx] = (frame[idx] + delta).clamp(0.0, 1.0);
            }
        }
    }

    /// Extract the watermark pattern correlation from `frame`.
    ///
    /// Returns the mean signed correlation between the frame residual and the pattern.
    /// A positive value suggests the watermark is present.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn extract_invisible(
        &self,
        frame: &[f32],
        original: &[f32],
        frame_width: u32,
        frame_height: u32,
    ) -> f32 {
        if frame.len() != original.len() {
            return 0.0;
        }
        let pw = self.pattern.width as usize;
        let ph = self.pattern.height as usize;
        let fw = frame_width as usize;
        let fh = frame_height as usize;

        let mut correlation = 0.0_f32;
        for fy in 0..fh {
            for fx in 0..fw {
                let px = if self.config.tile {
                    fx % pw
                } else {
                    fx.min(pw - 1)
                };
                let py = if self.config.tile {
                    fy % ph
                } else {
                    fy.min(ph - 1)
                };
                let pattern_val = self.pattern.values[py * pw + px];
                let residual = frame[fy * fw + fx] - original[fy * fw + fx];
                correlation += residual * pattern_val;
            }
        }
        let n = (fw * fh) as f32;
        if n > 0.0 {
            correlation / n
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: u32, h: u32) -> SpatialPattern {
        let values: Vec<f32> = (0..(w as usize * h as usize))
            .map(|i| {
                let x = i % w as usize;
                let y = i / w as usize;
                if (x + y) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        SpatialPattern::new(w, h, values).expect("should succeed in test")
    }

    #[test]
    fn test_spatial_pattern_pixel_count() {
        let p = SpatialPattern::constant(4, 4, 0.5);
        assert_eq!(p.pixel_count(), 16);
    }

    #[test]
    fn test_spatial_pattern_new_wrong_size() {
        assert!(SpatialPattern::new(3, 3, vec![0.0; 10]).is_none());
    }

    #[test]
    fn test_spatial_pattern_get_in_bounds() {
        let p = checkerboard(4, 4);
        assert_eq!(p.get(0, 0), Some(1.0));
        assert_eq!(p.get(1, 0), Some(-1.0));
    }

    #[test]
    fn test_spatial_pattern_get_out_of_bounds() {
        let p = SpatialPattern::constant(4, 4, 0.5);
        assert!(p.get(10, 0).is_none());
    }

    #[test]
    fn test_spatial_config_is_robust_low_strength() {
        let c = SpatialConfig {
            strength: 0.01,
            tile: true,
            perceptual_mask: false,
        };
        assert!(!c.is_robust());
    }

    #[test]
    fn test_spatial_config_is_robust_high_strength() {
        let c = SpatialConfig {
            strength: 0.1,
            tile: true,
            perceptual_mask: false,
        };
        assert!(c.is_robust());
    }

    #[test]
    fn test_spatial_config_is_robust_via_perceptual() {
        let c = SpatialConfig {
            strength: 0.01,
            tile: true,
            perceptual_mask: true,
        };
        assert!(c.is_robust());
    }

    #[test]
    fn test_embed_changes_frame() {
        let pattern = SpatialPattern::constant(2, 2, 1.0);
        let config = SpatialConfig {
            strength: 0.1,
            tile: true,
            perceptual_mask: false,
        };
        let wm = SpatialWatermark::new(config, pattern);
        let mut frame = vec![0.5_f32; 4];
        let original = frame.clone();
        wm.embed_invisible(&mut frame, 2, 2);
        assert!(frame
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn test_embed_clamps_to_valid_range() {
        let pattern = SpatialPattern::constant(2, 2, 1.0);
        let config = SpatialConfig {
            strength: 1.0,
            tile: true,
            perceptual_mask: false,
        };
        let wm = SpatialWatermark::new(config, pattern);
        let mut frame = vec![0.99_f32; 4];
        wm.embed_invisible(&mut frame, 2, 2);
        assert!(frame.iter().all(|&v| v <= 1.0));
    }

    #[test]
    fn test_extract_positive_correlation() {
        let pattern = SpatialPattern::constant(2, 2, 1.0);
        let config = SpatialConfig {
            strength: 0.1,
            tile: true,
            perceptual_mask: false,
        };
        let wm = SpatialWatermark::new(config, pattern);
        let original = vec![0.5_f32; 4];
        let mut watermarked = original.clone();
        wm.embed_invisible(&mut watermarked, 2, 2);
        let corr = wm.extract_invisible(&watermarked, &original, 2, 2);
        assert!(corr > 0.0, "correlation should be positive: {corr}");
    }

    #[test]
    fn test_extract_length_mismatch_returns_zero() {
        let pattern = SpatialPattern::constant(2, 2, 1.0);
        let wm = SpatialWatermark::new(SpatialConfig::default(), pattern);
        let frame = vec![0.5_f32; 4];
        let orig = vec![0.5_f32; 5]; // different length
        let corr = wm.extract_invisible(&frame, &orig, 2, 2);
        assert_eq!(corr, 0.0);
    }

    #[test]
    fn test_tiling_covers_larger_frame() {
        let pattern = SpatialPattern::constant(2, 2, 0.5);
        let config = SpatialConfig {
            strength: 0.1,
            tile: true,
            perceptual_mask: false,
        };
        let wm = SpatialWatermark::new(config, pattern);
        let mut frame = vec![0.5_f32; 16]; // 4×4
        wm.embed_invisible(&mut frame, 4, 4);
        // All pixels should be shifted by the same constant delta.
        let expected = 0.5 + 0.5 * 0.1;
        for &v in &frame {
            assert!((v - expected).abs() < 1e-5);
        }
    }
}
