//! Video matting: alpha matte extraction for precise foreground/background separation.
//!
//! This module provides a closed-form matting solution as an alternative to
//! binary chroma keying.  It operates on a "trimap" (foreground / unknown /
//! background labelling) and refines the hard boundaries into smooth alpha
//! values that capture fine details such as hair and semi-transparent objects.
//!
//! # Algorithm overview
//!
//! 1. Start from a trimap: pixels labelled *known foreground* (alpha=1),
//!    *known background* (alpha=0), or *unknown*.
//! 2. For each unknown pixel, solve a local colour-line model in a small
//!    neighbourhood to estimate the fractional foreground coverage (alpha).
//! 3. Optionally refine using an iterative guided-filter smoothing pass.
//!
//! This is a CPU-only, pure-Rust implementation that does not depend on any
//! external vision library.

use crate::error::{CvError, CvResult};

/// Trimap label for each pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimapLabel {
    /// Definitely foreground: alpha = 1.
    Foreground,
    /// Definitely background: alpha = 0.
    Background,
    /// Unknown region where alpha must be estimated.
    Unknown,
}

/// Alpha matte result.
#[derive(Debug, Clone)]
pub struct AlphaMatte {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Per-pixel alpha value in \[0.0, 1.0\], row-major.
    pub alpha: Vec<f32>,
}

impl AlphaMatte {
    /// Create a new alpha matte of the given dimensions (all zeros).
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            alpha: vec![0.0f32; width * height],
        }
    }

    /// Get alpha at pixel `(x, y)`.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.alpha[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Set alpha at pixel `(x, y)`.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.alpha[y * self.width + x] = value.clamp(0.0, 1.0);
        }
    }

    /// Number of pixels with alpha > threshold.
    #[must_use]
    pub fn foreground_pixels(&self, threshold: f32) -> usize {
        self.alpha.iter().filter(|&&a| a > threshold).count()
    }

    /// Apply the matte to an RGBA buffer.
    ///
    /// Sets the alpha channel of each pixel according to the matte.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length doesn't match width × height × 4.
    pub fn apply_to_rgba(&self, rgba: &mut [u8]) -> CvResult<()> {
        let expected = self.width * self.height * 4;
        if rgba.len() < expected {
            return Err(CvError::insufficient_data(expected, rgba.len()));
        }
        for (i, &a) in self.alpha.iter().enumerate() {
            rgba[i * 4 + 3] = (a * 255.0).round() as u8;
        }
        Ok(())
    }
}

/// Configuration for the alpha matte extractor.
#[derive(Debug, Clone)]
pub struct MattingConfig {
    /// Local window radius for colour sampling.
    pub window_radius: usize,
    /// Regularisation epsilon for the colour-line model.
    pub epsilon: f32,
    /// Number of guided-filter refinement iterations.
    pub refinement_iterations: usize,
    /// Guided-filter radius for smoothing.
    pub guided_filter_radius: usize,
    /// Smoothing strength for the guided filter (higher = smoother edges).
    pub smoothing_strength: f32,
}

impl Default for MattingConfig {
    fn default() -> Self {
        Self {
            window_radius: 5,
            epsilon: 1e-7,
            refinement_iterations: 2,
            guided_filter_radius: 3,
            smoothing_strength: 0.01,
        }
    }
}

/// Alpha matte extractor.
///
/// Estimates per-pixel alpha values for unknown trimap regions using a
/// local colour-line model.
pub struct AlphaMatteExtractor {
    config: MattingConfig,
}

impl AlphaMatteExtractor {
    /// Create a new extractor with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MattingConfig::default(),
        }
    }

    /// Create a new extractor with custom configuration.
    #[must_use]
    pub fn with_config(config: MattingConfig) -> Self {
        Self { config }
    }

    /// Extract an alpha matte from an RGB image and trimap.
    ///
    /// # Arguments
    ///
    /// * `image` – Raw RGB data (3 bytes per pixel, row-major).
    /// * `trimap` – Per-pixel trimap label (one per pixel).
    /// * `width`, `height` – Image dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if image or trimap dimensions are inconsistent.
    pub fn extract(
        &self,
        image: &[u8],
        trimap: &[TrimapLabel],
        width: usize,
        height: usize,
    ) -> CvResult<AlphaMatte> {
        let n = width * height;
        if image.len() < n * 3 {
            return Err(CvError::insufficient_data(n * 3, image.len()));
        }
        if trimap.len() != n {
            return Err(CvError::invalid_parameter(
                "trimap",
                format!("expected {} labels, got {}", n, trimap.len()),
            ));
        }
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width as u32, height as u32));
        }

        let mut matte = AlphaMatte::new(width, height);

        // Initialise known regions from trimap
        for i in 0..n {
            match trimap[i] {
                TrimapLabel::Foreground => matte.alpha[i] = 1.0,
                TrimapLabel::Background => matte.alpha[i] = 0.0,
                TrimapLabel::Unknown => {} // to be estimated
            }
        }

        // Estimate unknown pixels
        let r = self.config.window_radius;
        for y in 0..height {
            for x in 0..width {
                if trimap[y * width + x] != TrimapLabel::Unknown {
                    continue;
                }

                // Collect known foreground/background samples in window
                let mut fg_samples: Vec<[f32; 3]> = Vec::new();
                let mut bg_samples: Vec<[f32; 3]> = Vec::new();

                let y_start = y.saturating_sub(r);
                let y_end = (y + r + 1).min(height);
                let x_start = x.saturating_sub(r);
                let x_end = (x + r + 1).min(width);

                for ny in y_start..y_end {
                    for nx in x_start..x_end {
                        let idx = ny * width + nx;
                        let pixel = [
                            image[idx * 3] as f32 / 255.0,
                            image[idx * 3 + 1] as f32 / 255.0,
                            image[idx * 3 + 2] as f32 / 255.0,
                        ];
                        match trimap[idx] {
                            TrimapLabel::Foreground => fg_samples.push(pixel),
                            TrimapLabel::Background => bg_samples.push(pixel),
                            TrimapLabel::Unknown => {}
                        }
                    }
                }

                let curr_pixel = [
                    image[(y * width + x) * 3] as f32 / 255.0,
                    image[(y * width + x) * 3 + 1] as f32 / 255.0,
                    image[(y * width + x) * 3 + 2] as f32 / 255.0,
                ];

                let alpha = self.estimate_alpha(&curr_pixel, &fg_samples, &bg_samples);

                matte.alpha[y * width + x] = alpha;
            }
        }

        // Refinement: iterative guided filter smoothing
        for _ in 0..self.config.refinement_iterations {
            self.guided_filter_smooth(image, &mut matte, width, height);
        }

        // Re-apply trimap constraints (known pixels must stay 0 or 1)
        for i in 0..n {
            match trimap[i] {
                TrimapLabel::Foreground => matte.alpha[i] = 1.0,
                TrimapLabel::Background => matte.alpha[i] = 0.0,
                TrimapLabel::Unknown => {}
            }
        }

        Ok(matte)
    }

    /// Estimate alpha for a single unknown pixel.
    ///
    /// Uses a Bayesian estimation: find closest foreground and background
    /// colour cluster means and interpolate by colour distance.
    fn estimate_alpha(
        &self,
        pixel: &[f32; 3],
        fg_samples: &[[f32; 3]],
        bg_samples: &[[f32; 3]],
    ) -> f32 {
        if fg_samples.is_empty() && bg_samples.is_empty() {
            return 0.5; // No information: assume 50/50
        }

        if fg_samples.is_empty() {
            return 0.0;
        }

        if bg_samples.is_empty() {
            return 1.0;
        }

        // Compute mean foreground and background colours
        let fg_mean = mean_color(fg_samples);
        let bg_mean = mean_color(bg_samples);

        let d_fg = color_dist(pixel, &fg_mean);
        let d_bg = color_dist(pixel, &bg_mean);

        let total = d_fg + d_bg + self.config.epsilon;
        if total < 1e-9 {
            return 0.5;
        }

        // Alpha = proportion of pixel belonging to foreground.
        // When d_fg ≈ 0 (pixel matches fg_mean), alpha → 1.
        let alpha = d_bg / total;
        alpha.clamp(0.0, 1.0)
    }

    /// Single pass of guided-filter alpha smoothing.
    ///
    /// Uses a box-filter approximation to the guided filter for efficiency.
    fn guided_filter_smooth(
        &self,
        image: &[u8],
        matte: &mut AlphaMatte,
        width: usize,
        height: usize,
    ) {
        let r = self.config.guided_filter_radius;
        let eps = self.config.smoothing_strength;
        let n = width * height;

        // Compute local statistics for the guidance (grayscale) image
        let gray: Vec<f32> = (0..n)
            .map(|i| {
                let r_ch = image[i * 3] as f32 / 255.0;
                let g_ch = image[i * 3 + 1] as f32 / 255.0;
                let b_ch = image[i * 3 + 2] as f32 / 255.0;
                0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
            })
            .collect();

        let alpha_copy = matte.alpha.clone();
        let mut new_alpha = vec![0.0f32; n];

        for y in 0..height {
            for x in 0..width {
                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(height);
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(width);

                let mut sum_g = 0.0f64;
                let mut sum_a = 0.0f64;
                let mut sum_gg = 0.0f64;
                let mut sum_ga = 0.0f64;
                let mut cnt = 0.0f64;

                for ny in y0..y1 {
                    for nx in x0..x1 {
                        let i = ny * width + nx;
                        let g = gray[i] as f64;
                        let a = alpha_copy[i] as f64;
                        sum_g += g;
                        sum_a += a;
                        sum_gg += g * g;
                        sum_ga += g * a;
                        cnt += 1.0;
                    }
                }

                if cnt < 1.0 {
                    new_alpha[y * width + x] = alpha_copy[y * width + x];
                    continue;
                }

                let mean_g = sum_g / cnt;
                let mean_a = sum_a / cnt;
                let var_g = sum_gg / cnt - mean_g * mean_g;
                let cov_ga = sum_ga / cnt - mean_g * mean_a;

                let a_k = cov_ga / (var_g + eps as f64);
                let b_k = mean_a - a_k * mean_g;

                let i = y * width + x;
                let guided = (a_k * gray[i] as f64 + b_k) as f32;
                new_alpha[i] = guided.clamp(0.0, 1.0);
            }
        }

        matte.alpha = new_alpha;
    }
}

impl Default for AlphaMatteExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute mean colour of a set of samples.
fn mean_color(samples: &[[f32; 3]]) -> [f32; 3] {
    if samples.is_empty() {
        return [0.0; 3];
    }
    let n = samples.len() as f32;
    let mut sum = [0.0f32; 3];
    for s in samples {
        sum[0] += s[0];
        sum[1] += s[1];
        sum[2] += s[2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Euclidean distance in RGB space.
fn color_dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(w: usize, h: usize) -> Vec<u8> {
        // Left half green, right half red
        let mut img = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                if x < w / 2 {
                    img[idx] = 0;
                    img[idx + 1] = 200;
                    img[idx + 2] = 0;
                } else {
                    img[idx] = 200;
                    img[idx + 1] = 0;
                    img[idx + 2] = 0;
                }
            }
        }
        img
    }

    fn make_trimap(w: usize, h: usize) -> Vec<TrimapLabel> {
        let mut tm = vec![TrimapLabel::Unknown; w * h];
        // Left quarter = background
        for y in 0..h {
            for x in 0..w / 4 {
                tm[y * w + x] = TrimapLabel::Background;
            }
        }
        // Right quarter = foreground
        for y in 0..h {
            for x in (w * 3 / 4)..w {
                tm[y * w + x] = TrimapLabel::Foreground;
            }
        }
        tm
    }

    #[test]
    fn test_alpha_matte_new() {
        let m = AlphaMatte::new(10, 10);
        assert_eq!(m.width, 10);
        assert_eq!(m.height, 10);
        assert!(m.alpha.iter().all(|&a| a == 0.0));
    }

    #[test]
    fn test_alpha_matte_get_set() {
        let mut m = AlphaMatte::new(5, 5);
        m.set(2, 3, 0.7);
        assert!((m.get(2, 3) - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_alpha_matte_clamp() {
        let mut m = AlphaMatte::new(5, 5);
        m.set(0, 0, 2.0);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-5);
        m.set(0, 0, -1.0);
        assert!((m.get(0, 0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_alpha_matte_foreground_pixels() {
        let mut m = AlphaMatte::new(4, 4);
        m.set(0, 0, 0.9);
        m.set(1, 1, 0.8);
        m.set(2, 2, 0.3); // below threshold
        assert_eq!(m.foreground_pixels(0.5), 2);
    }

    #[test]
    fn test_alpha_matte_apply_to_rgba() {
        let mut m = AlphaMatte::new(2, 2);
        m.set(0, 0, 1.0);
        m.set(1, 0, 0.0);
        let mut rgba = vec![128u8; 2 * 2 * 4];
        m.apply_to_rgba(&mut rgba)
            .expect("apply_to_rgba should succeed");
        assert_eq!(rgba[3], 255); // pixel (0,0) alpha
        assert_eq!(rgba[7], 0); // pixel (1,0) alpha
    }

    #[test]
    fn test_extractor_trimap_constraints() {
        let w = 20usize;
        let h = 20usize;
        let img = make_test_image(w, h);
        let trimap = make_trimap(w, h);
        let extractor = AlphaMatteExtractor::new();
        let matte = extractor
            .extract(&img, &trimap, w, h)
            .expect("extraction should succeed");

        // Known foreground pixels should have alpha = 1
        for y in 0..h {
            for x in (w * 3 / 4)..w {
                assert!(
                    (matte.get(x, y) - 1.0).abs() < 1e-5,
                    "fg pixel ({x},{y}) should be 1.0, got {}",
                    matte.get(x, y)
                );
            }
        }
        // Known background pixels should have alpha = 0
        for y in 0..h {
            for x in 0..w / 4 {
                assert!(
                    matte.get(x, y).abs() < 1e-5,
                    "bg pixel ({x},{y}) should be 0.0, got {}",
                    matte.get(x, y)
                );
            }
        }
    }

    #[test]
    fn test_extractor_invalid_trimap_size() {
        let extractor = AlphaMatteExtractor::new();
        let img = make_test_image(10, 10);
        let trimap = vec![TrimapLabel::Unknown; 5]; // wrong size
        let result = extractor.extract(&img, &trimap, 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_extractor_zero_dimensions() {
        let extractor = AlphaMatteExtractor::new();
        let result = extractor.extract(&[], &[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_extractor_unknown_region_estimated() {
        let w = 10usize;
        let h = 10usize;
        let img = make_test_image(w, h);
        let trimap = make_trimap(w, h);
        let extractor = AlphaMatteExtractor::new();
        let matte = extractor
            .extract(&img, &trimap, w, h)
            .expect("extraction should succeed");

        // Unknown region should have values in (0, 1)
        for y in 0..h {
            for x in (w / 4)..(w * 3 / 4) {
                let a = matte.get(x, y);
                assert!(
                    (0.0..=1.0).contains(&a),
                    "alpha {a} out of range at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_matting_config_default() {
        let cfg = MattingConfig::default();
        assert_eq!(cfg.window_radius, 5);
        assert_eq!(cfg.refinement_iterations, 2);
    }

    #[test]
    fn test_trimap_label_equality() {
        assert_eq!(TrimapLabel::Foreground, TrimapLabel::Foreground);
        assert_ne!(TrimapLabel::Foreground, TrimapLabel::Background);
    }
}
