//! Approximate Non-Local Means (NLM) and enhanced bilateral spatial denoising.
//!
//! Provides fast approximations of NLM suitable for real-time use, plus
//! utility functions for patch similarity and kernel computation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Gaussian kernel weights for a 1-D window of radius `r`.
pub fn gaussian_kernel_1d(radius: usize, sigma: f32) -> Vec<f32> {
    let size = 2 * radius + 1;
    let mut kernel: Vec<f32> = (0..size)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    if sum > 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }
    kernel
}

/// Compute the mean squared error between two patches.
///
/// Both patches must have the same length.
pub fn patch_mse(patch_a: &[u8], patch_b: &[u8]) -> f32 {
    if patch_a.len() != patch_b.len() || patch_a.is_empty() {
        return f32::MAX;
    }
    let n = patch_a.len() as f32;
    patch_a
        .iter()
        .zip(patch_b.iter())
        .map(|(&a, &b)| {
            let d = a as f32 - b as f32;
            d * d
        })
        .sum::<f32>()
        / n
}

/// Compute the NLM weight for two patches given filter parameter `h`.
///
/// w = exp(-MSE(p, q) / (h * h))
pub fn nlm_weight(patch_a: &[u8], patch_b: &[u8], h: f32) -> f32 {
    let mse = patch_mse(patch_a, patch_b);
    (-mse / (h * h)).exp()
}

/// Configuration for approximate NLM denoising.
#[derive(Debug, Clone)]
pub struct NlmApproxConfig {
    /// Patch radius (patch size = 2*radius + 1 in each dimension).
    pub patch_radius: usize,
    /// Search radius for candidate patches.
    pub search_radius: usize,
    /// Filter parameter h (controls blur amount).
    pub h: f32,
}

impl Default for NlmApproxConfig {
    fn default() -> Self {
        Self {
            patch_radius: 2,
            search_radius: 5,
            h: 10.0,
        }
    }
}

impl NlmApproxConfig {
    /// Create a fast (small search radius) configuration.
    pub fn fast() -> Self {
        Self {
            patch_radius: 1,
            search_radius: 3,
            h: 15.0,
        }
    }

    /// Create a high-quality (large search radius) configuration.
    pub fn quality() -> Self {
        Self {
            patch_radius: 3,
            search_radius: 10,
            h: 8.0,
        }
    }
}

/// Extract a flat patch from a 1-D image of dimensions `width × height` at position `(cx, cy)`.
///
/// Returns a patch of size `(2*radius+1)^2`. Edge pixels use border clamping.
pub fn extract_patch(
    image: &[u8],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
) -> Vec<u8> {
    let size = 2 * radius + 1;
    let mut patch = Vec::with_capacity(size * size);
    for dy in 0..size {
        let row = cy + dy;
        let row = if radius <= row {
            (row - radius).min(height.saturating_sub(1))
        } else {
            0
        };
        for dx in 0..size {
            let col = cx + dx;
            let col = if radius <= col {
                (col - radius).min(width.saturating_sub(1))
            } else {
                0
            };
            patch.push(image[row * width + col]);
        }
    }
    patch
}

/// Apply a fast approximate NLM filter to a grayscale image.
///
/// `image` is a flat `width * height` byte slice.
/// Returns the filtered image as a `Vec<u8>`.
pub fn nlm_approx_filter(
    image: &[u8],
    width: usize,
    height: usize,
    config: &NlmApproxConfig,
) -> Vec<u8> {
    let mut output = vec![0u8; width * height];
    for cy in 0..height {
        for cx in 0..width {
            let ref_patch = extract_patch(image, width, height, cx, cy, config.patch_radius);

            let row_start = cy.saturating_sub(config.search_radius);
            let row_end = (cy + config.search_radius + 1).min(height);
            let col_start = cx.saturating_sub(config.search_radius);
            let col_end = (cx + config.search_radius + 1).min(width);

            let mut weighted_sum = 0.0_f32;
            let mut weight_total = 0.0_f32;

            for sy in row_start..row_end {
                for sx in col_start..col_end {
                    let cand_patch =
                        extract_patch(image, width, height, sx, sy, config.patch_radius);
                    let w = nlm_weight(&ref_patch, &cand_patch, config.h);
                    weighted_sum += w * image[sy * width + sx] as f32;
                    weight_total += w;
                }
            }

            let filtered = if weight_total > 0.0 {
                (weighted_sum / weight_total).round().clamp(0.0, 255.0) as u8
            } else {
                image[cy * width + cx]
            };
            output[cy * width + cx] = filtered;
        }
    }
    output
}

/// Apply an edge-enhanced bilateral-style filter using pre-computed range weights.
///
/// `sigma_space` controls spatial weighting, `sigma_range` controls range (intensity) weighting.
pub fn enhanced_bilateral(
    image: &[u8],
    width: usize,
    height: usize,
    radius: usize,
    sigma_space: f32,
    sigma_range: f32,
) -> Vec<u8> {
    let mut output = vec![0u8; width * height];
    for cy in 0..height {
        for cx in 0..width {
            let centre_val = image[cy * width + cx] as f32;
            let mut weighted_sum = 0.0_f32;
            let mut weight_total = 0.0_f32;

            let row_start = cy.saturating_sub(radius);
            let row_end = (cy + radius + 1).min(height);
            let col_start = cx.saturating_sub(radius);
            let col_end = (cx + radius + 1).min(width);

            for sy in row_start..row_end {
                for sx in col_start..col_end {
                    let dx = sx as f32 - cx as f32;
                    let dy = sy as f32 - cy as f32;
                    let spatial = -(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space);
                    let neighbour_val = image[sy * width + sx] as f32;
                    let range_diff = neighbour_val - centre_val;
                    let range = -(range_diff * range_diff) / (2.0 * sigma_range * sigma_range);
                    let w = (spatial + range).exp();
                    weighted_sum += w * neighbour_val;
                    weight_total += w;
                }
            }
            let filtered = if weight_total > 0.0 {
                (weighted_sum / weight_total).round().clamp(0.0, 255.0) as u8
            } else {
                image[cy * width + cx]
            };
            output[cy * width + cx] = filtered;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let kernel = gaussian_kernel_1d(2, 1.0);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_kernel_length() {
        let kernel = gaussian_kernel_1d(3, 1.5);
        assert_eq!(kernel.len(), 7);
    }

    #[test]
    fn test_gaussian_kernel_symmetry() {
        let kernel = gaussian_kernel_1d(3, 1.5);
        let n = kernel.len();
        for i in 0..n / 2 {
            assert!((kernel[i] - kernel[n - 1 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_patch_mse_identical() {
        let patch = vec![100u8; 9];
        assert_eq!(patch_mse(&patch, &patch), 0.0);
    }

    #[test]
    fn test_patch_mse_different() {
        let a = vec![0u8; 4];
        let b = vec![10u8; 4];
        let mse = patch_mse(&a, &b);
        assert!((mse - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_patch_mse_length_mismatch() {
        let a = vec![1u8; 3];
        let b = vec![1u8; 4];
        assert_eq!(patch_mse(&a, &b), f32::MAX);
    }

    #[test]
    fn test_nlm_weight_identical_patches() {
        let patch = vec![128u8; 9];
        let w = nlm_weight(&patch, &patch, 10.0);
        assert!((w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_nlm_weight_different_patches() {
        let a = vec![0u8; 9];
        let b = vec![100u8; 9];
        let w = nlm_weight(&a, &b, 10.0);
        assert!(w < 1.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_extract_patch_centre() {
        let image: Vec<u8> = (0..25).collect();
        // 5x5 image, extract 3x3 patch around (2,2)
        let patch = extract_patch(&image, 5, 5, 2, 2, 1);
        assert_eq!(patch.len(), 9);
        // centre pixel is image[2*5+2] = 12
        assert_eq!(patch[4], 12);
    }

    #[test]
    fn test_extract_patch_corner() {
        let image: Vec<u8> = (0..16).collect();
        // 4x4 image, extract 3x3 patch at (0,0) with border clamping
        let patch = extract_patch(&image, 4, 4, 0, 0, 1);
        assert_eq!(patch.len(), 9);
    }

    #[test]
    fn test_nlm_approx_filter_uniform() {
        // Uniform image: output should equal input
        let image = vec![128u8; 25];
        let config = NlmApproxConfig::fast();
        let output = nlm_approx_filter(&image, 5, 5, &config);
        assert_eq!(output, image);
    }

    #[test]
    fn test_nlm_approx_filter_size_preserved() {
        let image: Vec<u8> = (0..64).map(|i| (i % 256) as u8).collect();
        let config = NlmApproxConfig::fast();
        let output = nlm_approx_filter(&image, 8, 8, &config);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_enhanced_bilateral_uniform() {
        let image = vec![100u8; 36];
        let output = enhanced_bilateral(&image, 6, 6, 1, 2.0, 30.0);
        assert_eq!(output, image);
    }

    #[test]
    fn test_enhanced_bilateral_size_preserved() {
        let image: Vec<u8> = (0..64).map(|i| (i * 3 % 256) as u8).collect();
        let output = enhanced_bilateral(&image, 8, 8, 2, 3.0, 40.0);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_nlm_config_defaults() {
        let config = NlmApproxConfig::default();
        assert_eq!(config.patch_radius, 2);
        assert_eq!(config.search_radius, 5);
        assert!(config.h > 0.0);
    }

    #[test]
    fn test_nlm_config_fast() {
        let config = NlmApproxConfig::fast();
        assert!(config.search_radius < NlmApproxConfig::default().search_radius);
    }
}
