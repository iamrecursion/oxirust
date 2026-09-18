#![allow(dead_code)]
//! Chroma-specific noise reduction.
//!
//! Chroma (color) channels in video are often noisier than luma, especially
//! in high-ISO footage. This module provides chroma-targeted filters that
//! aggressively reduce color noise while preserving luminance detail.

/// Chroma denoising method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaMethod {
    /// Simple box-blur on chroma planes.
    BoxBlur,
    /// Gaussian blur on chroma planes.
    Gaussian,
    /// Bilateral filter preserving chroma edges.
    Bilateral,
    /// Weighted median filter.
    Median,
}

impl Default for ChromaMethod {
    fn default() -> Self {
        Self::Bilateral
    }
}

/// Configuration for chroma denoising.
#[derive(Clone, Debug)]
pub struct ChromaDenoiseConfig {
    /// Filter method.
    pub method: ChromaMethod,
    /// Filter strength for Cb channel (0.0 = off, 1.0 = maximum).
    pub strength_cb: f32,
    /// Filter strength for Cr channel (0.0 = off, 1.0 = maximum).
    pub strength_cr: f32,
    /// Kernel radius in pixels.
    pub radius: u32,
    /// Preserve luma-correlated chroma edges.
    pub preserve_edges: bool,
    /// Edge threshold for bilateral/edge-aware modes.
    pub edge_threshold: f32,
}

impl Default for ChromaDenoiseConfig {
    fn default() -> Self {
        Self {
            method: ChromaMethod::default(),
            strength_cb: 0.5,
            strength_cr: 0.5,
            radius: 3,
            preserve_edges: true,
            edge_threshold: 10.0,
        }
    }
}

impl ChromaDenoiseConfig {
    /// Create a light chroma denoise configuration.
    pub fn light() -> Self {
        Self {
            strength_cb: 0.3,
            strength_cr: 0.3,
            radius: 2,
            ..Default::default()
        }
    }

    /// Create a strong chroma denoise configuration.
    pub fn strong() -> Self {
        Self {
            strength_cb: 0.8,
            strength_cr: 0.8,
            radius: 5,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.strength_cb) {
            return Err("strength_cb must be in [0.0, 1.0]".to_string());
        }
        if !(0.0..=1.0).contains(&self.strength_cr) {
            return Err("strength_cr must be in [0.0, 1.0]".to_string());
        }
        if self.radius == 0 || self.radius > 32 {
            return Err("radius must be in [1, 32]".to_string());
        }
        Ok(())
    }
}

/// Apply box blur to a chroma plane.
#[allow(clippy::cast_precision_loss)]
pub fn box_blur_chroma(plane: &[f32], width: usize, height: usize, radius: u32) -> Vec<f32> {
    let r = radius as usize;
    let mut output = vec![0.0f32; plane.len()];
    for y in 0..height {
        for x in 0..width {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(height);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(width);
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += f64::from(plane[yy * width + xx]);
                    count += 1;
                }
            }
            output[y * width + x] = (sum / f64::from(count)) as f32;
        }
    }
    output
}

/// Apply Gaussian blur to a chroma plane.
#[allow(clippy::cast_precision_loss)]
pub fn gaussian_blur_chroma(
    plane: &[f32],
    width: usize,
    height: usize,
    radius: u32,
    sigma: f32,
) -> Vec<f32> {
    let r = radius as i32;
    let sigma_sq = 2.0 * f64::from(sigma) * f64::from(sigma);
    // Build 1D kernel
    let kernel_size = (2 * r + 1) as usize;
    let mut kernel = vec![0.0f64; kernel_size];
    let mut kernel_sum = 0.0f64;
    for i in 0..kernel_size {
        let d = (i as i32 - r) as f64;
        let w = (-d * d / sigma_sq).exp();
        kernel[i] = w;
        kernel_sum += w;
    }
    for k in &mut kernel {
        *k /= kernel_sum;
    }

    // Horizontal pass
    let mut temp = vec![0.0f32; plane.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            for ki in 0..kernel_size {
                let sx = (x as i32 + ki as i32 - r).clamp(0, (width - 1) as i32) as usize;
                acc += f64::from(plane[y * width + sx]) * kernel[ki];
            }
            temp[y * width + x] = acc as f32;
        }
    }

    // Vertical pass
    let mut output = vec![0.0f32; plane.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            for ki in 0..kernel_size {
                let sy = (y as i32 + ki as i32 - r).clamp(0, (height - 1) as i32) as usize;
                acc += f64::from(temp[sy * width + x]) * kernel[ki];
            }
            output[y * width + x] = acc as f32;
        }
    }
    output
}

/// Apply median filter to a chroma plane.
pub fn median_filter_chroma(plane: &[f32], width: usize, height: usize, radius: u32) -> Vec<f32> {
    let r = radius as usize;
    let mut output = vec![0.0f32; plane.len()];
    for y in 0..height {
        for x in 0..width {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r + 1).min(height);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(width);
            let mut window: Vec<f32> = Vec::new();
            for yy in y0..y1 {
                for xx in x0..x1 {
                    window.push(plane[yy * width + xx]);
                }
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = window.len() / 2;
            output[y * width + x] = window[mid];
        }
    }
    output
}

/// Blend original and filtered planes by strength.
#[allow(clippy::cast_precision_loss)]
pub fn blend_chroma(original: &[f32], filtered: &[f32], strength: f32) -> Vec<f32> {
    let s = strength.clamp(0.0, 1.0);
    original
        .iter()
        .zip(filtered.iter())
        .map(|(&o, &f)| o * (1.0 - s) + f * s)
        .collect()
}

/// Statistics from chroma denoising.
#[derive(Clone, Debug)]
pub struct ChromaDenoiseStats {
    /// Mean absolute difference on Cb.
    pub cb_mad: f32,
    /// Mean absolute difference on Cr.
    pub cr_mad: f32,
    /// Peak signal-to-noise ratio improvement estimate on Cb.
    pub cb_psnr_gain: f32,
    /// Peak signal-to-noise ratio improvement estimate on Cr.
    pub cr_psnr_gain: f32,
}

/// Compute mean absolute difference between two buffers.
#[allow(clippy::cast_precision_loss)]
pub fn mean_absolute_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from((x - y).abs()))
        .sum();
    (sum / a.len() as f64) as f32
}

/// Estimate PSNR between two buffers (assuming peak = 1.0).
#[allow(clippy::cast_precision_loss)]
pub fn estimate_psnr(original: &[f32], processed: &[f32]) -> f32 {
    if original.is_empty() {
        return 0.0;
    }
    let mse: f64 = original
        .iter()
        .zip(processed.iter())
        .map(|(&o, &p)| {
            let d = f64::from(o - p);
            d * d
        })
        .sum::<f64>()
        / original.len() as f64;
    if mse < 1e-10 {
        return 100.0;
    }
    (10.0 * (1.0f64 / mse).log10()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_method_default() {
        assert_eq!(ChromaMethod::default(), ChromaMethod::Bilateral);
    }

    #[test]
    fn test_config_default() {
        let cfg = ChromaDenoiseConfig::default();
        assert!((cfg.strength_cb - 0.5).abs() < f32::EPSILON);
        assert!((cfg.strength_cr - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.radius, 3);
        assert!(cfg.preserve_edges);
    }

    #[test]
    fn test_config_presets() {
        let light = ChromaDenoiseConfig::light();
        assert!((light.strength_cb - 0.3).abs() < f32::EPSILON);
        let strong = ChromaDenoiseConfig::strong();
        assert!((strong.strength_cb - 0.8).abs() < f32::EPSILON);
        assert_eq!(strong.radius, 5);
    }

    #[test]
    fn test_config_validate() {
        let cfg = ChromaDenoiseConfig::default();
        assert!(cfg.validate().is_ok());

        let bad = ChromaDenoiseConfig {
            strength_cb: 2.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad_radius = ChromaDenoiseConfig {
            radius: 0,
            ..Default::default()
        };
        assert!(bad_radius.validate().is_err());
    }

    #[test]
    fn test_box_blur_uniform() {
        let plane = vec![0.5f32; 16];
        let result = box_blur_chroma(&plane, 4, 4, 1);
        for v in &result {
            assert!((*v - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_box_blur_impulse() {
        let mut plane = vec![0.0f32; 25];
        plane[12] = 1.0; // center of 5x5
        let result = box_blur_chroma(&plane, 5, 5, 1);
        // Center value should decrease (spread)
        assert!(result[12] < 1.0);
        assert!(result[12] > 0.0);
    }

    #[test]
    fn test_gaussian_blur_uniform() {
        let plane = vec![0.5f32; 16];
        let result = gaussian_blur_chroma(&plane, 4, 4, 1, 1.0);
        for v in &result {
            assert!((*v - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_median_filter_uniform() {
        let plane = vec![0.7f32; 9];
        let result = median_filter_chroma(&plane, 3, 3, 1);
        for v in &result {
            assert!((*v - 0.7).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_blend_zero_strength() {
        let orig = vec![1.0, 2.0, 3.0];
        let filt = vec![0.0, 0.0, 0.0];
        let result = blend_chroma(&orig, &filt, 0.0);
        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        assert!((result[1] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_blend_full_strength() {
        let orig = vec![1.0, 2.0, 3.0];
        let filt = vec![0.0, 0.0, 0.0];
        let result = blend_chroma(&orig, &filt, 1.0);
        for v in &result {
            assert!((*v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_mean_absolute_diff() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.5, 2.5, 3.5];
        let mad = mean_absolute_diff(&a, &b);
        assert!((mad - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_estimate_psnr_identical() {
        let a = vec![0.5f32; 100];
        let b = vec![0.5f32; 100];
        let psnr = estimate_psnr(&a, &b);
        assert!(psnr >= 99.0);
    }

    #[test]
    fn test_estimate_psnr_different() {
        let a = vec![1.0f32; 100];
        let b = vec![0.0f32; 100];
        let psnr = estimate_psnr(&a, &b);
        assert!(psnr < 1.0);
    }
}
