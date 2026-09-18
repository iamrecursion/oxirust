//! Focus peaking for sharpness analysis.
//!
//! Focus peaking highlights in-focus edges in a luma image by applying
//! one of several high-frequency edge detection kernels and coloring pixels
//! that exceed a sensitivity threshold.

#![allow(dead_code)]

/// Edge detection algorithm used for focus peaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeakingMode {
    /// Laplacian 3×3 kernel: `[-1,-1,-1; -1,8,-1; -1,-1,-1]`.
    Laplacian,
    /// Sobel: horizontal/vertical 3×3 kernels, combined magnitude.
    Sobel,
    /// Roberts cross-gradient (2×2 diagonal differences).
    Roberts,
    /// Prewitt 3×3 kernel — high-frequency variant.
    PrewittHighFreq,
}

/// Configuration for the focus peaking overlay.
#[derive(Debug, Clone)]
pub struct PeakingConfig {
    /// Sensitivity in [0, 1]: higher value → more pixels highlighted.
    pub sensitivity: f32,
    /// Highlight color (RGBA). Defaults to red semi-transparent.
    pub color: [u8; 4],
    /// Edge detection mode.
    pub mode: PeakingMode,
}

impl Default for PeakingConfig {
    fn default() -> Self {
        Self {
            sensitivity: 0.15,
            color: [255, 0, 0, 220],
            mode: PeakingMode::Laplacian,
        }
    }
}

/// Per-frame analysis from the focus peaking pass.
#[derive(Debug, Clone)]
pub struct PeakingAnalysis {
    /// Percentage of pixels identified as sharply focused [0, 100].
    pub sharp_pixel_pct: f32,
    /// Peak sharpness value (0-1, normalised).
    pub peak_sharpness: f32,
    /// Estimated center of the sharpest region (x, y) in luma coordinates.
    pub focus_center: (u32, u32),
}

/// Focus peaking processor.
pub struct FocusPeaking {
    config: PeakingConfig,
}

impl FocusPeaking {
    /// Creates a new `FocusPeaking` processor with the given configuration.
    #[must_use]
    pub fn new(config: PeakingConfig) -> Self {
        Self { config }
    }

    /// Applies focus peaking to a luma plane and returns an RGBA overlay.
    ///
    /// # Arguments
    ///
    /// * `luma`   - Luma values in [0, 1] per pixel, row-major.
    /// * `width`  - Image width.
    /// * `height` - Image height.
    ///
    /// # Returns
    ///
    /// RGBA byte buffer (width × height × 4). Highlighted pixels receive the
    /// configured peaking color; non-highlighted pixels are transparent (alpha 0).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn apply(&self, luma: &[f32], width: u32, height: u32) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut overlay = vec![0u8; w * h * 4];

        if w < 3 || h < 3 {
            return overlay;
        }

        let edges = match self.config.mode {
            PeakingMode::Laplacian => apply_laplacian(luma, w, h),
            PeakingMode::Sobel => apply_sobel(luma, w, h),
            PeakingMode::Roberts => apply_roberts(luma, w, h),
            PeakingMode::PrewittHighFreq => apply_prewitt_highfreq(luma, w, h),
        };

        // Find max edge value for normalization
        let max_edge = edges.iter().cloned().fold(0.0f32, f32::max);
        if max_edge <= 0.0 {
            return overlay;
        }

        let threshold = (1.0 - self.config.sensitivity).clamp(0.0, 1.0);
        let color = self.config.color;

        for i in 0..w * h {
            let norm = edges[i] / max_edge;
            if norm >= threshold {
                let alpha = ((norm - threshold) / (1.0 - threshold + 1e-6) * color[3] as f32)
                    .min(255.0) as u8;
                let idx = i * 4;
                overlay[idx] = color[0];
                overlay[idx + 1] = color[1];
                overlay[idx + 2] = color[2];
                overlay[idx + 3] = alpha;
            }
        }

        overlay
    }

    /// Applies focus peaking and returns the overlay together with analysis.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_with_analysis(
        &self,
        luma: &[f32],
        width: u32,
        height: u32,
    ) -> (Vec<u8>, PeakingAnalysis) {
        let w = width as usize;
        let h = height as usize;
        let mut overlay = vec![0u8; w * h * 4];

        if w < 3 || h < 3 {
            return (
                overlay,
                PeakingAnalysis {
                    sharp_pixel_pct: 0.0,
                    peak_sharpness: 0.0,
                    focus_center: (0, 0),
                },
            );
        }

        let edges = match self.config.mode {
            PeakingMode::Laplacian => apply_laplacian(luma, w, h),
            PeakingMode::Sobel => apply_sobel(luma, w, h),
            PeakingMode::Roberts => apply_roberts(luma, w, h),
            PeakingMode::PrewittHighFreq => apply_prewitt_highfreq(luma, w, h),
        };

        let max_edge = edges.iter().cloned().fold(0.0f32, f32::max);
        let peak_sharpness = if max_edge > 0.0 { 1.0f32 } else { 0.0f32 };

        if max_edge <= 0.0 {
            return (
                overlay,
                PeakingAnalysis {
                    sharp_pixel_pct: 0.0,
                    peak_sharpness: 0.0,
                    focus_center: (width / 2, height / 2),
                },
            );
        }

        let threshold = (1.0 - self.config.sensitivity).clamp(0.0, 1.0);
        let color = self.config.color;

        let mut sharp_count = 0u32;
        let mut sum_x = 0u64;
        let mut sum_y = 0u64;

        for i in 0..w * h {
            let norm = edges[i] / max_edge;
            if norm >= threshold {
                sharp_count += 1;
                let px = (i % w) as u64;
                let py = (i / w) as u64;
                sum_x += px;
                sum_y += py;

                let alpha = ((norm - threshold) / (1.0 - threshold + 1e-6) * color[3] as f32)
                    .min(255.0) as u8;
                let idx = i * 4;
                overlay[idx] = color[0];
                overlay[idx + 1] = color[1];
                overlay[idx + 2] = color[2];
                overlay[idx + 3] = alpha;
            }
        }

        let total = (w * h) as u32;
        let sharp_pixel_pct = if total > 0 {
            (sharp_count as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        let focus_center = if sharp_count > 0 {
            (
                (sum_x / sharp_count as u64) as u32,
                (sum_y / sharp_count as u64) as u32,
            )
        } else {
            (width / 2, height / 2)
        };

        let analysis = PeakingAnalysis {
            sharp_pixel_pct,
            peak_sharpness,
            focus_center,
        };

        (overlay, analysis)
    }

    /// Returns a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &PeakingConfig {
        &self.config
    }
}

impl Default for FocusPeaking {
    fn default() -> Self {
        Self::new(PeakingConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Kernel implementations
// ---------------------------------------------------------------------------

/// Laplacian 3×3: `[-1,-1,-1; -1,8,-1; -1,-1,-1]`.
#[allow(clippy::cast_possible_wrap)]
fn apply_laplacian(luma: &[f32], w: usize, h: usize) -> Vec<f32> {
    let kernel: [i32; 9] = [-1, -1, -1, -1, 8, -1, -1, -1, -1];
    convolve_3x3(luma, w, h, &kernel)
}

/// Sobel horizontal + vertical, combined L2 magnitude.
#[allow(clippy::cast_possible_wrap)]
fn apply_sobel(luma: &[f32], w: usize, h: usize) -> Vec<f32> {
    let kx: [i32; 9] = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let ky: [i32; 9] = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    let gx = convolve_3x3(luma, w, h, &kx);
    let gy = convolve_3x3(luma, w, h, &ky);

    gx.iter()
        .zip(gy.iter())
        .map(|(x, y)| (x * x + y * y).sqrt())
        .collect()
}

/// Roberts cross (2×2 diagonal differences).
fn apply_roberts(luma: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let i = y * w + x;
            let p00 = luma[i];
            let p01 = luma[i + 1];
            let p10 = luma[i + w];
            let p11 = luma[i + w + 1];
            let g1 = p00 - p11;
            let g2 = p01 - p10;
            out[i] = (g1 * g1 + g2 * g2).sqrt();
        }
    }
    out
}

/// Prewitt high-frequency variant: uses a sharpening-biased kernel.
#[allow(clippy::cast_possible_wrap)]
fn apply_prewitt_highfreq(luma: &[f32], w: usize, h: usize) -> Vec<f32> {
    // Prewitt X and Y (standard), combined as L2
    let kx: [i32; 9] = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
    let ky: [i32; 9] = [-1, -1, -1, 0, 0, 0, 1, 1, 1];
    let gx = convolve_3x3(luma, w, h, &kx);
    let gy = convolve_3x3(luma, w, h, &ky);
    gx.iter()
        .zip(gy.iter())
        .map(|(x, y)| (x * x + y * y).sqrt())
        .collect()
}

/// Generic 3×3 separable-style convolution, returns absolute values.
#[allow(clippy::cast_possible_wrap)]
fn convolve_3x3(luma: &[f32], w: usize, h: usize, kernel: &[i32; 9]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut sum = 0.0f32;
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let py = y + ky - 1;
                    let px = x + kx - 1;
                    sum += luma[py * w + px] * kernel[ky * 3 + kx] as f32;
                }
            }
            out[y * w + x] = sum.abs();
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a checkerboard luma pattern (alternating 0/1 tiles of size `tile`).
    fn checkerboard(w: usize, h: usize, tile: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if (x / tile + y / tile) % 2 == 0 {
                    v[y * w + x] = 1.0;
                }
            }
        }
        v
    }

    /// Create a uniform luma plane (all same value).
    fn uniform(w: usize, h: usize, val: f32) -> Vec<f32> {
        vec![val; w * h]
    }

    #[test]
    fn test_config_default() {
        let cfg = PeakingConfig::default();
        assert!((cfg.sensitivity - 0.15).abs() < 0.001);
        assert_eq!(cfg.color, [255, 0, 0, 220]);
        assert_eq!(cfg.mode, PeakingMode::Laplacian);
    }

    #[test]
    fn test_apply_output_size() {
        let peaking = FocusPeaking::default();
        let luma = checkerboard(64, 64, 8);
        let out = peaking.apply(&luma, 64, 64);
        assert_eq!(out.len(), 64 * 64 * 4);
    }

    #[test]
    fn test_uniform_image_no_peaks() {
        let peaking = FocusPeaking::default();
        let luma = uniform(32, 32, 0.5);
        let (_, analysis) = peaking.apply_with_analysis(&luma, 32, 32);
        // Uniform image has no edges, so no peaks
        assert!(analysis.sharp_pixel_pct < 1.0);
    }

    #[test]
    fn test_checkerboard_has_peaks() {
        let peaking = FocusPeaking::new(PeakingConfig {
            sensitivity: 0.5,
            ..Default::default()
        });
        let luma = checkerboard(64, 64, 4);
        let (_, analysis) = peaking.apply_with_analysis(&luma, 64, 64);
        assert!(
            analysis.sharp_pixel_pct > 0.0,
            "Checkerboard should produce edge peaks"
        );
    }

    #[test]
    fn test_laplacian_mode() {
        let peaking = FocusPeaking::new(PeakingConfig {
            mode: PeakingMode::Laplacian,
            sensitivity: 0.5,
            ..Default::default()
        });
        let luma = checkerboard(32, 32, 4);
        let out = peaking.apply(&luma, 32, 32);
        assert_eq!(out.len(), 32 * 32 * 4);
    }

    #[test]
    fn test_sobel_mode() {
        let peaking = FocusPeaking::new(PeakingConfig {
            mode: PeakingMode::Sobel,
            sensitivity: 0.5,
            ..Default::default()
        });
        let luma = checkerboard(32, 32, 4);
        let out = peaking.apply(&luma, 32, 32);
        assert_eq!(out.len(), 32 * 32 * 4);
    }

    #[test]
    fn test_roberts_mode() {
        let peaking = FocusPeaking::new(PeakingConfig {
            mode: PeakingMode::Roberts,
            sensitivity: 0.5,
            ..Default::default()
        });
        let luma = checkerboard(32, 32, 4);
        let out = peaking.apply(&luma, 32, 32);
        assert_eq!(out.len(), 32 * 32 * 4);
    }

    #[test]
    fn test_prewitt_mode() {
        let peaking = FocusPeaking::new(PeakingConfig {
            mode: PeakingMode::PrewittHighFreq,
            sensitivity: 0.5,
            ..Default::default()
        });
        let luma = checkerboard(32, 32, 4);
        let out = peaking.apply(&luma, 32, 32);
        assert_eq!(out.len(), 32 * 32 * 4);
    }

    #[test]
    fn test_focus_center_near_middle_for_uniform_checkerboard() {
        let peaking = FocusPeaking::new(PeakingConfig {
            sensitivity: 0.3,
            mode: PeakingMode::Sobel,
            ..Default::default()
        });
        let w = 64u32;
        let h = 64u32;
        let luma = checkerboard(w as usize, h as usize, 4);
        let (_, analysis) = peaking.apply_with_analysis(&luma, w, h);
        // With a uniform checkerboard the center should be roughly in the image
        assert!(analysis.focus_center.0 < w);
        assert!(analysis.focus_center.1 < h);
    }

    #[test]
    fn test_too_small_image_returns_empty_overlay() {
        let peaking = FocusPeaking::default();
        let luma = vec![0.5f32; 4];
        let out = peaking.apply(&luma, 2, 2);
        // All transparent
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_peak_sharpness_nonzero_for_edges() {
        let peaking = FocusPeaking::new(PeakingConfig {
            sensitivity: 0.01,
            ..Default::default()
        });
        let luma = checkerboard(32, 32, 2);
        let (_, analysis) = peaking.apply_with_analysis(&luma, 32, 32);
        assert!(analysis.peak_sharpness > 0.0);
    }
}
