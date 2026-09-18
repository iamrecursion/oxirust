//! Multi-Scale Structural Similarity Index (MS-SSIM) calculation.
//!
//! MS-SSIM extends SSIM by computing it at multiple scales through iterative
//! low-pass filtering and downsampling. This better captures quality degradation
//! at different viewing distances.
//!
//! # Formula
//!
//! MS-SSIM = [lₘ(x,y)]^αₘ · ∏ᵢ₌₁ᴹ [cᵢ(x,y)]^βᵢ · [sᵢ(x,y)]^γᵢ
//!
//! where M is the number of scales (typically 5).

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// MS-SSIM calculator for video quality assessment.
pub struct MsSsimCalculator {
    /// Number of scales (typically 5)
    num_scales: usize,
    /// Exponent weights for each scale
    weights: Vec<f64>,
    /// Window size for SSIM calculation
    window_size: usize,
    /// K1 constant
    k1: f64,
    /// K2 constant
    k2: f64,
}

impl MsSsimCalculator {
    /// Creates a new MS-SSIM calculator with default parameters.
    ///
    /// Uses 5 scales with standard weights.
    #[must_use]
    pub fn new() -> Self {
        // Standard MS-SSIM weights from Wang et al. 2003
        let weights = vec![0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

        Self {
            num_scales: 5,
            weights,
            window_size: 11,
            k1: 0.01,
            k2: 0.03,
        }
    }

    /// Creates MS-SSIM calculator with custom number of scales.
    #[must_use]
    pub fn with_scales(num_scales: usize) -> Self {
        let mut weights = vec![1.0 / num_scales as f64; num_scales];
        // Redistribute weights to emphasize middle scales
        if num_scales >= 3 {
            weights[0] *= 0.5;
            weights[num_scales - 1] *= 0.5;
            let redistribution = (weights[0] + weights[num_scales - 1]) / (num_scales - 2) as f64;
            for w in weights.iter_mut().skip(1).take(num_scales - 2) {
                *w += redistribution;
            }
        }

        Self {
            num_scales,
            weights,
            window_size: 11,
            k1: 0.01,
            k2: 0.03,
        }
    }

    /// Calculates MS-SSIM between reference and distorted frames.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match or are too small.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> OxiResult<QualityScore> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match".to_string(),
            ));
        }

        let mut score = QualityScore::new(MetricType::MsSsim, 0.0);

        // Calculate MS-SSIM for Y plane
        let y_msssim = self.calculate_plane(
            &reference.planes[0],
            &distorted.planes[0],
            reference.width,
            reference.height,
        )?;
        score.add_component("Y", y_msssim);

        score.score = y_msssim;
        Ok(score)
    }

    /// Calculates MS-SSIM for a single plane.
    fn calculate_plane(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
    ) -> OxiResult<f64> {
        let min_dimension = width.min(height);
        let max_scales = ((min_dimension as f64).log2() - 4.0).floor() as usize;
        let num_scales = self.num_scales.min(max_scales);

        if num_scales < 2 {
            return Err(oximedia_core::OxiError::InvalidData(
                "Image too small for MS-SSIM".to_string(),
            ));
        }

        let mut ref_current = ref_plane.to_vec();
        let mut dist_current = dist_plane.to_vec();
        let mut current_width = width;
        let mut current_height = height;

        let mut contrast_sensitivity = Vec::with_capacity(num_scales);
        let mut structure_comparison = Vec::with_capacity(num_scales);
        let mut luminance = 0.0;

        for scale in 0..num_scales {
            let (lum, cs, sc) = self.calculate_ssim_components(
                &ref_current,
                &dist_current,
                current_width,
                current_height,
            )?;

            if scale == num_scales - 1 {
                luminance = lum;
            }

            contrast_sensitivity.push(cs);
            structure_comparison.push(sc);

            // Downsample for next scale (except last)
            if scale < num_scales - 1 {
                let (new_ref, new_dist, new_width, new_height) = self.downsample_by_2(
                    &ref_current,
                    &dist_current,
                    current_width,
                    current_height,
                );
                ref_current = new_ref;
                dist_current = new_dist;
                current_width = new_width;
                current_height = new_height;
            }
        }

        // Compute MS-SSIM as weighted product
        let mut msssim = luminance.powf(self.weights[num_scales - 1]);

        for (i, (&cs, &sc)) in contrast_sensitivity
            .iter()
            .zip(structure_comparison.iter())
            .enumerate()
        {
            let component = (cs * sc).powf(self.weights[i]);
            msssim *= component;
        }

        Ok(msssim.clamp(0.0, 1.0))
    }

    /// Calculates SSIM components (luminance, contrast, structure) at one scale.
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_ssim_components(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
    ) -> OxiResult<(f64, f64, f64)> {
        let half_window = self.window_size / 2;
        let l = 255.0;
        let c1 = (self.k1 * l) * (self.k1 * l);
        let c2 = (self.k2 * l) * (self.k2 * l);
        let c3 = c2 / 2.0;

        let components: Vec<(f64, f64, f64)> = (half_window..height - half_window)
            .into_par_iter()
            .flat_map(|y| {
                (half_window..width - half_window)
                    .map(|x| {
                        self.calculate_components_at_position(
                            ref_plane, dist_plane, x, y, width, c1, c2, c3,
                        )
                    })
                    .collect::<Vec<(f64, f64, f64)>>()
            })
            .collect();

        if components.is_empty() {
            return Ok((1.0, 1.0, 1.0));
        }

        let mean_lum = components.iter().map(|c| c.0).sum::<f64>() / components.len() as f64;
        let mean_cs = components.iter().map(|c| c.1).sum::<f64>() / components.len() as f64;
        let mean_sc = components.iter().map(|c| c.2).sum::<f64>() / components.len() as f64;

        Ok((mean_lum, mean_cs, mean_sc))
    }

    /// Calculates SSIM components at a specific position.
    fn calculate_components_at_position(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        cx: usize,
        cy: usize,
        width: usize,
        c1: f64,
        c2: f64,
        c3: f64,
    ) -> (f64, f64, f64) {
        let half_window = self.window_size / 2;
        let mut sum_ref = 0.0;
        let mut sum_dist = 0.0;
        let mut sum_ref_sq = 0.0;
        let mut sum_dist_sq = 0.0;
        let mut sum_ref_dist = 0.0;
        let mut count = 0.0;

        for dy in 0..self.window_size {
            let y = cy - half_window + dy;
            for dx in 0..self.window_size {
                let x = cx - half_window + dx;
                let idx = y * width + x;

                let ref_val = f64::from(ref_plane[idx]);
                let dist_val = f64::from(dist_plane[idx]);

                sum_ref += ref_val;
                sum_dist += dist_val;
                sum_ref_sq += ref_val * ref_val;
                sum_dist_sq += dist_val * dist_val;
                sum_ref_dist += ref_val * dist_val;
                count += 1.0;
            }
        }

        let mu_ref = sum_ref / count;
        let mu_dist = sum_dist / count;
        let sigma_ref_sq = (sum_ref_sq / count) - mu_ref * mu_ref;
        let sigma_dist_sq = (sum_dist_sq / count) - mu_dist * mu_dist;
        let sigma_ref_dist = (sum_ref_dist / count) - mu_ref * mu_dist;

        let luminance = (2.0 * mu_ref * mu_dist + c1) / (mu_ref * mu_ref + mu_dist * mu_dist + c1);
        let contrast = (2.0 * sigma_ref_sq.sqrt() * sigma_dist_sq.sqrt() + c2)
            / (sigma_ref_sq + sigma_dist_sq + c2);
        let structure = (sigma_ref_dist + c3) / (sigma_ref_sq.sqrt() * sigma_dist_sq.sqrt() + c3);

        (luminance, contrast, structure)
    }

    /// Downsamples plane by factor of 2 using simple averaging.
    fn downsample_by_2(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
    ) -> (Vec<u8>, Vec<u8>, usize, usize) {
        let new_width = width / 2;
        let new_height = height / 2;

        let mut new_ref = vec![0u8; new_width * new_height];
        let mut new_dist = vec![0u8; new_width * new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_y = y * 2;
                let src_x = x * 2;

                // 2x2 average
                let ref_sum = u32::from(ref_plane[src_y * width + src_x])
                    + u32::from(ref_plane[src_y * width + src_x + 1])
                    + u32::from(ref_plane[(src_y + 1) * width + src_x])
                    + u32::from(ref_plane[(src_y + 1) * width + src_x + 1]);

                let dist_sum = u32::from(dist_plane[src_y * width + src_x])
                    + u32::from(dist_plane[src_y * width + src_x + 1])
                    + u32::from(dist_plane[(src_y + 1) * width + src_x])
                    + u32::from(dist_plane[(src_y + 1) * width + src_x + 1]);

                new_ref[y * new_width + x] = (ref_sum / 4) as u8;
                new_dist[y * new_width + x] = (dist_sum / 4) as u8;
            }
        }

        (new_ref, new_dist, new_width, new_height)
    }
}

impl Default for MsSsimCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        frame.planes[0].fill(value);
        frame
    }

    #[test]
    fn test_msssim_identical_frames() {
        let calc = MsSsimCalculator::new();
        let frame1 = create_test_frame(256, 256, 128);
        let frame2 = create_test_frame(256, 256, 128);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!((result.score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_msssim_different_frames() {
        let calc = MsSsimCalculator::new();
        let frame1 = create_test_frame(256, 256, 100);
        let frame2 = create_test_frame(256, 256, 110);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score > 0.0 && result.score < 1.0);
    }

    #[test]
    fn test_msssim_small_image() {
        let calc = MsSsimCalculator::new();
        let frame1 = create_test_frame(32, 32, 128);
        let frame2 = create_test_frame(32, 32, 128);

        // Should fail because image is too small for 5 scales
        assert!(calc.calculate(&frame1, &frame2).is_err());
    }

    #[test]
    fn test_downsampling() {
        let calc = MsSsimCalculator::new();
        let ref_plane = vec![100u8; 256];
        let dist_plane = vec![110u8; 256];

        let (new_ref, _new_dist, new_width, new_height) =
            calc.downsample_by_2(&ref_plane, &dist_plane, 16, 16);

        assert_eq!(new_width, 8);
        assert_eq!(new_height, 8);
        assert_eq!(new_ref.len(), 64);
        assert_eq!(new_ref[0], 100); // Should be average of 100s
    }

    #[test]
    fn test_custom_scales() {
        let calc = MsSsimCalculator::with_scales(3);
        assert_eq!(calc.num_scales, 3);
        assert_eq!(calc.weights.len(), 3);

        let frame1 = create_test_frame(256, 256, 128);
        let frame2 = create_test_frame(256, 256, 128);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!((result.score - 1.0).abs() < 0.01);
    }
}
