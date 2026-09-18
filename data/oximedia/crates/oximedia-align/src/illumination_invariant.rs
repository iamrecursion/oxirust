//! Exposure-invariant feature descriptors.
//!
//! Provides [`IlluminationInvariantDescriptor`] that normalizes patch intensity
//! before computing ORB-like binary descriptors. This makes feature matching
//! robust to global and local illumination changes between frames (e.g. when
//! cameras have different exposure settings or when lighting changes over time).
//!
//! # Algorithm
//!
//! 1. Extract a patch around each keypoint.
//! 2. Normalize the patch: subtract mean intensity, divide by standard deviation.
//! 3. Compute binary tests on the normalized patch (BRIEF-style).
//!
//! This simple normalization removes additive (brightness) and multiplicative
//! (contrast) illumination effects, making the descriptor invariant to affine
//! intensity transforms `I' = a*I + b`.

#![allow(clippy::cast_precision_loss)]

use crate::features::{BinaryDescriptor, Keypoint};
use crate::{AlignError, AlignResult};

/// Configuration for illumination-invariant descriptor extraction.
#[derive(Debug, Clone)]
pub struct IlluminationInvariantConfig {
    /// Half-size of the normalization patch (full patch is `2*half + 1`).
    pub patch_half_size: usize,
    /// Whether to apply Gaussian weighting to the patch before normalization.
    pub gaussian_weighting: bool,
    /// Sigma for Gaussian weighting (only used if `gaussian_weighting` is true).
    pub gaussian_sigma: f64,
}

impl Default for IlluminationInvariantConfig {
    fn default() -> Self {
        Self {
            patch_half_size: 15,
            gaussian_weighting: true,
            gaussian_sigma: 5.0,
        }
    }
}

/// Illumination-invariant binary descriptor extractor.
///
/// Normalizes local patch intensity before computing binary tests,
/// making the descriptor robust to exposure and lighting changes.
pub struct IlluminationInvariantDescriptor {
    /// Configuration.
    pub config: IlluminationInvariantConfig,
    /// Pre-computed sampling pattern (256 pairs of offsets).
    pattern: Vec<(isize, isize, isize, isize)>,
}

impl Default for IlluminationInvariantDescriptor {
    fn default() -> Self {
        Self::new(IlluminationInvariantConfig::default())
    }
}

impl IlluminationInvariantDescriptor {
    /// Create a new illumination-invariant descriptor extractor.
    #[must_use]
    pub fn new(config: IlluminationInvariantConfig) -> Self {
        let pattern = Self::generate_pattern(config.patch_half_size);
        Self { config, pattern }
    }

    /// Generate a deterministic sampling pattern within the patch.
    fn generate_pattern(half_size: usize) -> Vec<(isize, isize, isize, isize)> {
        let mut pattern = Vec::with_capacity(256);
        let half = half_size as isize;
        let full = (2 * half + 1) as u64;

        let mut seed = 0x5EED_CAFE_u64;
        for _ in 0..256 {
            let x1 = (lcg_next(&mut seed) % full) as isize - half;
            let y1 = (lcg_next(&mut seed) % full) as isize - half;
            let x2 = (lcg_next(&mut seed) % full) as isize - half;
            let y2 = (lcg_next(&mut seed) % full) as isize - half;
            pattern.push((x1, y1, x2, y2));
        }

        pattern
    }

    /// Extract an illumination-invariant descriptor at a single keypoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the keypoint is too close to the image border.
    pub fn extract(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        keypoint: &Keypoint,
    ) -> AlignResult<BinaryDescriptor> {
        let half = self.config.patch_half_size as isize;
        let cx = keypoint.point.x.round() as isize;
        let cy = keypoint.point.y.round() as isize;

        if cx < half || cy < half || cx >= (width as isize - half) || cy >= (height as isize - half)
        {
            return Err(AlignError::FeatureError(
                "Keypoint too close to border for illumination-invariant descriptor".to_string(),
            ));
        }

        // Extract and normalize the patch
        let patch_size = (2 * half + 1) as usize;
        let mut patch = vec![0.0_f64; patch_size * patch_size];

        for dy in -half..=half {
            for dx in -half..=half {
                let px = (cx + dx) as usize;
                let py = (cy + dy) as usize;
                let pidx = (dy + half) as usize * patch_size + (dx + half) as usize;
                patch[pidx] = f64::from(image[py * width + px]);
            }
        }

        // Apply Gaussian weighting if configured
        if self.config.gaussian_weighting {
            let sigma2 = self.config.gaussian_sigma * self.config.gaussian_sigma;
            for dy in -half..=half {
                for dx in -half..=half {
                    let pidx = (dy + half) as usize * patch_size + (dx + half) as usize;
                    let r2 = (dx * dx + dy * dy) as f64;
                    let weight = (-0.5 * r2 / sigma2).exp();
                    patch[pidx] *= weight;
                }
            }
        }

        // Compute mean and standard deviation
        let n = patch.len() as f64;
        let mean = patch.iter().sum::<f64>() / n;
        let variance = patch.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let std_dev = variance.sqrt().max(1e-6); // avoid division by zero

        // Normalize: (I - mean) / std_dev
        for v in &mut patch {
            *v = (*v - mean) / std_dev;
        }

        // Compute binary descriptor from normalized patch
        let mut descriptor = [0u8; 32];
        for (bit_idx, &(x1, y1, x2, y2)) in self.pattern.iter().enumerate() {
            let idx1 = (y1 + half) as usize * patch_size + (x1 + half) as usize;
            let idx2 = (y2 + half) as usize * patch_size + (x2 + half) as usize;

            if idx1 < patch.len() && idx2 < patch.len() && patch[idx1] < patch[idx2] {
                let byte_idx = bit_idx / 8;
                let bit_pos = bit_idx % 8;
                descriptor[byte_idx] |= 1 << bit_pos;
            }
        }

        Ok(BinaryDescriptor::new(descriptor))
    }

    /// Extract descriptors for multiple keypoints.
    ///
    /// # Errors
    ///
    /// Returns an error if any extraction fails.
    pub fn extract_batch(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        keypoints: &[Keypoint],
    ) -> AlignResult<Vec<BinaryDescriptor>> {
        keypoints
            .iter()
            .map(|kp| self.extract(image, width, height, kp))
            .collect()
    }
}

/// Simple LCG PRNG for deterministic pattern generation.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient_image(w: usize, h: usize, brightness: u8, contrast: f32) -> Vec<u8> {
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let base = ((x as f32 / w as f32) * 128.0 + (y as f32 / h as f32) * 64.0) as f32;
                let val = (base * contrast + f32::from(brightness)).clamp(0.0, 255.0);
                img[y * w + x] = val as u8;
            }
        }
        img
    }

    #[test]
    fn test_config_default() {
        let config = IlluminationInvariantConfig::default();
        assert_eq!(config.patch_half_size, 15);
        assert!(config.gaussian_weighting);
    }

    #[test]
    fn test_descriptor_creation() {
        let desc = IlluminationInvariantDescriptor::default();
        assert_eq!(desc.pattern.len(), 256);
    }

    #[test]
    fn test_extract_center_keypoint() {
        let w = 64usize;
        let h = 64usize;
        let image = make_gradient_image(w, h, 0, 1.0);
        let desc = IlluminationInvariantDescriptor::default();
        let kp = Keypoint::new(32.0, 32.0, 1.0, 0.0, 100.0);

        let result = desc.extract(&image, w, h, &kp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_border_keypoint_fails() {
        let w = 64usize;
        let h = 64usize;
        let image = vec![128u8; w * h];
        let desc = IlluminationInvariantDescriptor::default();
        let kp = Keypoint::new(2.0, 2.0, 1.0, 0.0, 100.0);

        let result = desc.extract(&image, w, h, &kp);
        assert!(result.is_err());
    }

    #[test]
    fn test_illumination_invariance() {
        // Two images of the same scene with different exposure
        let w = 128usize;
        let h = 128usize;
        let img_dark = make_gradient_image(w, h, 10, 0.5);
        let img_bright = make_gradient_image(w, h, 80, 1.5);

        let desc = IlluminationInvariantDescriptor::new(IlluminationInvariantConfig {
            patch_half_size: 12,
            gaussian_weighting: true,
            gaussian_sigma: 4.0,
        });

        let kp = Keypoint::new(64.0, 64.0, 1.0, 0.0, 100.0);
        let d1 = desc.extract(&img_dark, w, h, &kp).expect("should succeed");
        let d2 = desc
            .extract(&img_bright, w, h, &kp)
            .expect("should succeed");

        let hamming = d1.hamming_distance(&d2);
        // With normalization, the descriptors should be more similar than different
        // (Hamming < 128 out of 256 bits means more similar than random)
        assert!(
            hamming < 128,
            "Illumination-invariant descriptors should be similar across exposures, hamming={hamming}"
        );
    }

    #[test]
    fn test_extract_batch() {
        let w = 128usize;
        let h = 128usize;
        let image = make_gradient_image(w, h, 0, 1.0);
        let desc = IlluminationInvariantDescriptor::default();

        let keypoints = vec![
            Keypoint::new(40.0, 40.0, 1.0, 0.0, 100.0),
            Keypoint::new(80.0, 80.0, 1.0, 0.0, 90.0),
        ];

        let result = desc.extract_batch(&image, w, h, &keypoints);
        assert!(result.is_ok());
        let descs = result.expect("should succeed");
        assert_eq!(descs.len(), 2);
    }

    #[test]
    fn test_constant_image_descriptor() {
        // On a constant image, normalization should still produce a valid descriptor
        let w = 64usize;
        let h = 64usize;
        let image = vec![128u8; w * h];
        let desc = IlluminationInvariantDescriptor::new(IlluminationInvariantConfig {
            patch_half_size: 10,
            gaussian_weighting: false,
            gaussian_sigma: 5.0,
        });
        let kp = Keypoint::new(32.0, 32.0, 1.0, 0.0, 100.0);

        let result = desc.extract(&image, w, h, &kp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_no_gaussian_weighting() {
        let w = 64usize;
        let h = 64usize;
        let image = make_gradient_image(w, h, 0, 1.0);
        let desc = IlluminationInvariantDescriptor::new(IlluminationInvariantConfig {
            patch_half_size: 10,
            gaussian_weighting: false,
            gaussian_sigma: 5.0,
        });
        let kp = Keypoint::new(32.0, 32.0, 1.0, 0.0, 100.0);

        let result = desc.extract(&image, w, h, &kp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_self_match_zero_distance() {
        let w = 128usize;
        let h = 128usize;
        let image = make_gradient_image(w, h, 50, 1.0);
        let desc = IlluminationInvariantDescriptor::default();
        let kp = Keypoint::new(64.0, 64.0, 1.0, 0.0, 100.0);

        let d = desc.extract(&image, w, h, &kp).expect("should succeed");
        assert_eq!(d.hamming_distance(&d), 0);
    }
}
