//! Visual Information Fidelity (VIF) calculation.
//!
//! VIF is a full-reference image quality assessment metric based on natural
//! scene statistics and information theory. It quantifies the information
//! shared between the reference and distorted images.
//!
//! # Reference
//!
//! H.R. Sheikh and A.C. Bovik, "Image Information and Visual Quality,"
//! IEEE Transactions on Image Processing, 2006.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;

/// VIF calculator for video quality assessment.
pub struct VifCalculator {
    /// Number of scales (subbands)
    num_scales: usize,
    /// Sigma for Gaussian noise
    sigma_nsq: f64,
}

impl VifCalculator {
    /// Creates a new VIF calculator with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_scales: 4,
            sigma_nsq: 2.0,
        }
    }

    /// Calculates VIF between reference and distorted frames.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> OxiResult<QualityScore> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match".to_string(),
            ));
        }

        let mut score = QualityScore::new(MetricType::Vif, 0.0);

        // Calculate VIF for Y plane
        let y_vif = self.calculate_plane(
            &reference.planes[0],
            &distorted.planes[0],
            reference.width,
            reference.height,
        )?;

        score.add_component("Y", y_vif);
        score.score = y_vif;

        Ok(score)
    }

    /// Calculates VIF for a single plane.
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_plane(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
    ) -> OxiResult<f64> {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        let mut ref_current = self.plane_to_f64(ref_plane);
        let mut dist_current = self.plane_to_f64(dist_plane);
        let mut current_width = width;
        let mut current_height = height;

        for _scale in 0..self.num_scales {
            let (num, denom) = self.compute_vif_subband(
                &ref_current,
                &dist_current,
                current_width,
                current_height,
            );

            numerator += num;
            denominator += denom;

            // Downsample for next scale
            if current_width > 32 && current_height > 32 {
                let (new_ref, new_dist, new_w, new_h) =
                    self.downsample(&ref_current, &dist_current, current_width, current_height);
                ref_current = new_ref;
                dist_current = new_dist;
                current_width = new_w;
                current_height = new_h;
            } else {
                break;
            }
        }

        let vif = if denominator > 1e-10 {
            numerator / denominator
        } else {
            1.0
        };

        Ok(vif.clamp(0.0, 1.0))
    }

    /// Computes VIF for a single subband.
    fn compute_vif_subband(
        &self,
        ref_plane: &[f64],
        dist_plane: &[f64],
        width: usize,
        height: usize,
    ) -> (f64, f64) {
        let block_size = 3;
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for y in (0..height).step_by(block_size) {
            for x in (0..width).step_by(block_size) {
                let end_y = (y + block_size).min(height);
                let end_x = (x + block_size).min(width);

                // Extract blocks
                let ref_block = self.extract_block(ref_plane, x, y, end_x, end_y, width);
                let dist_block = self.extract_block(dist_plane, x, y, end_x, end_y, width);

                // Compute statistics
                let ref_mean = ref_block.iter().sum::<f64>() / ref_block.len() as f64;
                let dist_mean = dist_block.iter().sum::<f64>() / dist_block.len() as f64;

                let ref_var = ref_block
                    .iter()
                    .map(|&v| (v - ref_mean).powi(2))
                    .sum::<f64>()
                    / ref_block.len() as f64;

                let dist_var = dist_block
                    .iter()
                    .map(|&v| (v - dist_mean).powi(2))
                    .sum::<f64>()
                    / dist_block.len() as f64;

                let covariance = ref_block
                    .iter()
                    .zip(dist_block.iter())
                    .map(|(&r, &d)| (r - ref_mean) * (d - dist_mean))
                    .sum::<f64>()
                    / ref_block.len() as f64;

                // VIF computation using information-theoretic approach
                let sigma_ref_sq = ref_var.max(1e-10);
                let sigma_dist_sq = dist_var.max(1e-10);

                let g = covariance / sigma_ref_sq;
                let sv_sq = sigma_dist_sq - g * g * sigma_ref_sq;

                // Information extracted from reference
                let num = if sigma_ref_sq > self.sigma_nsq {
                    ((sigma_ref_sq + self.sigma_nsq) / (sv_sq + self.sigma_nsq)).ln()
                } else {
                    0.0
                };

                let denom = if sigma_ref_sq > self.sigma_nsq {
                    ((sigma_ref_sq + self.sigma_nsq) / self.sigma_nsq).ln()
                } else {
                    0.0
                };

                numerator += num;
                denominator += denom;
            }
        }

        (numerator, denominator)
    }

    /// Extracts a block from the plane.
    fn extract_block(
        &self,
        plane: &[f64],
        x: usize,
        y: usize,
        end_x: usize,
        end_y: usize,
        width: usize,
    ) -> Vec<f64> {
        let mut block = Vec::new();
        for row in y..end_y {
            for col in x..end_x {
                block.push(plane[row * width + col]);
            }
        }
        block
    }

    /// Converts u8 plane to f64 plane.
    fn plane_to_f64(&self, plane: &[u8]) -> Vec<f64> {
        plane.iter().map(|&v| f64::from(v)).collect()
    }

    /// Downsamples plane by factor of 2.
    fn downsample(
        &self,
        ref_plane: &[f64],
        dist_plane: &[f64],
        width: usize,
        height: usize,
    ) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let new_width = width / 2;
        let new_height = height / 2;

        let mut new_ref = vec![0.0; new_width * new_height];
        let mut new_dist = vec![0.0; new_width * new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_y = y * 2;
                let src_x = x * 2;

                // 2x2 average
                let ref_sum = ref_plane[src_y * width + src_x]
                    + ref_plane[src_y * width + src_x + 1]
                    + ref_plane[(src_y + 1) * width + src_x]
                    + ref_plane[(src_y + 1) * width + src_x + 1];

                let dist_sum = dist_plane[src_y * width + src_x]
                    + dist_plane[src_y * width + src_x + 1]
                    + dist_plane[(src_y + 1) * width + src_x]
                    + dist_plane[(src_y + 1) * width + src_x + 1];

                new_ref[y * new_width + x] = ref_sum / 4.0;
                new_dist[y * new_width + x] = dist_sum / 4.0;
            }
        }

        (new_ref, new_dist, new_width, new_height)
    }
}

impl Default for VifCalculator {
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
    fn test_vif_identical_frames() {
        let calc = VifCalculator::new();
        let frame1 = create_test_frame(128, 128, 128);
        let frame2 = create_test_frame(128, 128, 128);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn test_vif_different_frames() {
        let calc = VifCalculator::new();
        let frame1 = create_test_frame(128, 128, 100);
        let frame2 = create_test_frame(128, 128, 110);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_plane_to_f64() {
        let calc = VifCalculator::new();
        let plane = vec![100u8, 150, 200];
        let f64_plane = calc.plane_to_f64(&plane);

        assert_eq!(f64_plane.len(), 3);
        assert!((f64_plane[0] - 100.0).abs() < 0.01);
        assert!((f64_plane[1] - 150.0).abs() < 0.01);
        assert!((f64_plane[2] - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_downsample() {
        let calc = VifCalculator::new();
        let ref_plane = vec![100.0; 256];
        let dist_plane = vec![110.0; 256];

        let (new_ref, _new_dist, new_width, new_height) =
            calc.downsample(&ref_plane, &dist_plane, 16, 16);

        assert_eq!(new_width, 8);
        assert_eq!(new_height, 8);
        assert_eq!(new_ref.len(), 64);
        assert!((new_ref[0] - 100.0).abs() < 0.01);
    }
}
