//! Structural Similarity Index (SSIM) calculation.
//!
//! SSIM is a perceptual metric that quantifies image quality degradation
//! caused by processing such as data compression. Unlike PSNR, which is based
//! on pixel-wise error, SSIM considers structural information.
//!
//! # Formula
//!
//! SSIM(x,y) = [l(x,y)ᵅ · c(x,y)ᵝ · s(x,y)ᵞ]
//!
//! where:
//! - l(x,y) is the luminance comparison
//! - c(x,y) is the contrast comparison
//! - s(x,y) is the structure comparison
//! - α, β, γ are parameters to adjust the relative importance (typically all 1)

use crate::psnr::extract_plane_roi;
use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// SSIM calculator for video quality assessment.
pub struct SsimCalculator {
    /// Window size (typically 11x11)
    window_size: usize,
    /// Gaussian window weights
    window: Vec<f64>,
    /// K1 constant for luminance (typically 0.01)
    k1: f64,
    /// K2 constant for contrast (typically 0.03)
    k2: f64,
    /// Weight for luma component
    luma_weight: f64,
    /// Weight for chroma components
    chroma_weight: f64,
}

impl SsimCalculator {
    /// Creates a new SSIM calculator with default parameters.
    ///
    /// Uses 11x11 Gaussian window, K1=0.01, K2=0.03.
    #[must_use]
    pub fn new() -> Self {
        Self::with_window_size(11)
    }

    /// Creates a new SSIM calculator with custom window size.
    #[must_use]
    pub fn with_window_size(window_size: usize) -> Self {
        let window = Self::create_gaussian_window(window_size);
        Self {
            window_size,
            window,
            k1: 0.01,
            k2: 0.03,
            luma_weight: 4.0 / 6.0,
            chroma_weight: 1.0 / 6.0,
        }
    }

    /// Creates a Gaussian window for SSIM calculation.
    fn create_gaussian_window(size: usize) -> Vec<f64> {
        let sigma = 1.5;
        let center = (size - 1) as f64 / 2.0;
        let mut window = Vec::with_capacity(size * size);
        let mut sum = 0.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                let value = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
                window.push(value);
                sum += value;
            }
        }

        // Normalize
        for value in &mut window {
            *value /= sum;
        }

        window
    }

    /// Calculates SSIM between reference and distorted frames.
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

        let mut score = QualityScore::new(MetricType::Ssim, 0.0);

        // Calculate Y SSIM
        let y_ssim = self.calculate_plane(
            &reference.planes[0],
            &distorted.planes[0],
            reference.width,
            reference.height,
            reference.strides[0],
            distorted.strides[0],
        )?;
        score.add_component("Y", y_ssim);

        let mut weighted_ssim = self.luma_weight * y_ssim;

        // Calculate chroma SSIM if available
        if reference.planes.len() >= 3 && distorted.planes.len() >= 3 {
            let (h_sub, v_sub) = reference.format.chroma_subsampling();
            let chroma_width = reference.width / h_sub as usize;
            let chroma_height = reference.height / v_sub as usize;

            let cb_ssim = self.calculate_plane(
                &reference.planes[1],
                &distorted.planes[1],
                chroma_width,
                chroma_height,
                reference.strides[1],
                distorted.strides[1],
            )?;

            let cr_ssim = self.calculate_plane(
                &reference.planes[2],
                &distorted.planes[2],
                chroma_width,
                chroma_height,
                reference.strides[2],
                distorted.strides[2],
            )?;

            score.add_component("Cb", cb_ssim);
            score.add_component("Cr", cr_ssim);

            weighted_ssim += self.chroma_weight * (cb_ssim + cr_ssim);
        }

        score.score = weighted_ssim;
        Ok(score)
    }

    /// Calculates SSIM for a specific region-of-interest (ROI) within the frames.
    ///
    /// `roi` is `(x, y, width, height)` in luma pixels.  Chroma planes are
    /// cropped proportionally.  The extracted sub-planes are treated as
    /// independent, stride-free images for the SSIM computation.
    ///
    /// # Errors
    ///
    /// Returns an error if the ROI extends outside the frame or frame sizes differ.
    pub fn calculate_region(
        &self,
        reference: &Frame,
        distorted: &Frame,
        roi: (usize, usize, usize, usize),
    ) -> OxiResult<QualityScore> {
        let (rx, ry, rw, rh) = roi;
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match for region SSIM".to_string(),
            ));
        }
        if rx + rw > reference.width || ry + rh > reference.height {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "ROI ({rx},{ry},{rw},{rh}) exceeds frame bounds {}×{}",
                reference.width, reference.height
            )));
        }

        let mut score = QualityScore::new(MetricType::Ssim, 0.0);

        // Extract luma ROI (packed, stride = rw)
        let ref_y = extract_plane_roi(&reference.planes[0], reference.strides[0], rx, ry, rw, rh);
        let dist_y = extract_plane_roi(&distorted.planes[0], distorted.strides[0], rx, ry, rw, rh);

        let y_ssim = self.calculate_plane(&ref_y, &dist_y, rw, rh, rw, rw)?;
        score.add_component("Y", y_ssim);
        let mut weighted = self.luma_weight * y_ssim;

        // Chroma ROI
        if reference.planes.len() >= 3 && distorted.planes.len() >= 3 {
            let (h_sub, v_sub) = reference.format.chroma_subsampling();
            let cx = rx / h_sub as usize;
            let cy = ry / v_sub as usize;
            let cw = (rw / h_sub as usize).max(1);
            let ch = (rh / v_sub as usize).max(1);

            let ref_cb =
                extract_plane_roi(&reference.planes[1], reference.strides[1], cx, cy, cw, ch);
            let dist_cb =
                extract_plane_roi(&distorted.planes[1], distorted.strides[1], cx, cy, cw, ch);
            let ref_cr =
                extract_plane_roi(&reference.planes[2], reference.strides[2], cx, cy, cw, ch);
            let dist_cr =
                extract_plane_roi(&distorted.planes[2], distorted.strides[2], cx, cy, cw, ch);

            let cb_ssim = self.calculate_plane(&ref_cb, &dist_cb, cw, ch, cw, cw)?;
            let cr_ssim = self.calculate_plane(&ref_cr, &dist_cr, cw, ch, cw, cw)?;
            score.add_component("Cb", cb_ssim);
            score.add_component("Cr", cr_ssim);
            weighted += self.chroma_weight * (cb_ssim + cr_ssim);
        }

        score.score = weighted;
        Ok(score)
    }

    /// Calculates SSIM for a single plane.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_plane(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
        ref_stride: usize,
        dist_stride: usize,
    ) -> OxiResult<f64> {
        let half_window = self.window_size / 2;

        // Dynamic range (for 8-bit: 255)
        let l = 255.0;
        let c1 = (self.k1 * l) * (self.k1 * l);
        let c2 = (self.k2 * l) * (self.k2 * l);

        // Calculate SSIM for each valid window position
        let ssim_values: Vec<f64> = (half_window..height - half_window)
            .into_par_iter()
            .flat_map(|y| {
                (half_window..width - half_window)
                    .map(|x| {
                        self.calculate_ssim_at_position(
                            ref_plane,
                            dist_plane,
                            x,
                            y,
                            width,
                            height,
                            ref_stride,
                            dist_stride,
                            c1,
                            c2,
                        )
                    })
                    .collect::<Vec<f64>>()
            })
            .collect();

        if ssim_values.is_empty() {
            return Ok(1.0);
        }

        // Mean SSIM
        let mean_ssim = ssim_values.iter().sum::<f64>() / ssim_values.len() as f64;
        Ok(mean_ssim)
    }

    /// Calculates SSIM at a specific position.
    #[allow(clippy::too_many_arguments)]
    fn calculate_ssim_at_position(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        cx: usize,
        cy: usize,
        _width: usize,
        _height: usize,
        ref_stride: usize,
        dist_stride: usize,
        c1: f64,
        c2: f64,
    ) -> f64 {
        let half_window = self.window_size / 2;

        let mut sum_ref = 0.0;
        let mut sum_dist = 0.0;
        let mut sum_ref_sq = 0.0;
        let mut sum_dist_sq = 0.0;
        let mut sum_ref_dist = 0.0;

        for dy in 0..self.window_size {
            let y = cy - half_window + dy;
            for dx in 0..self.window_size {
                let x = cx - half_window + dx;
                let w = self.window[dy * self.window_size + dx];

                let ref_idx = y * ref_stride + x;
                let dist_idx = y * dist_stride + x;

                let ref_val = f64::from(ref_plane[ref_idx]);
                let dist_val = f64::from(dist_plane[dist_idx]);

                sum_ref += w * ref_val;
                sum_dist += w * dist_val;
                sum_ref_sq += w * ref_val * ref_val;
                sum_dist_sq += w * dist_val * dist_val;
                sum_ref_dist += w * ref_val * dist_val;
            }
        }

        // Mean
        let mu_ref = sum_ref;
        let mu_dist = sum_dist;

        // Variance and covariance
        let sigma_ref_sq = sum_ref_sq - mu_ref * mu_ref;
        let sigma_dist_sq = sum_dist_sq - mu_dist * mu_dist;
        let sigma_ref_dist = sum_ref_dist - mu_ref * mu_dist;

        // SSIM formula
        let numerator = (2.0 * mu_ref * mu_dist + c1) * (2.0 * sigma_ref_dist + c2);
        let denominator =
            (mu_ref * mu_ref + mu_dist * mu_dist + c1) * (sigma_ref_sq + sigma_dist_sq + c2);

        numerator / denominator
    }
}

impl Default for SsimCalculator {
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
        frame.planes[1].fill(128);
        frame.planes[2].fill(128);
        frame
    }

    #[test]
    fn test_ssim_identical_frames() {
        let calc = SsimCalculator::new();
        let frame1 = create_test_frame(64, 64, 128);
        let frame2 = create_test_frame(64, 64, 128);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!((result.score - 1.0).abs() < 0.01); // SSIM should be ~1.0
    }

    #[test]
    fn test_ssim_different_frames() {
        let calc = SsimCalculator::new();
        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(64, 64, 110);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score > 0.0 && result.score < 1.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_gaussian_window() {
        let window = SsimCalculator::create_gaussian_window(11);
        assert_eq!(window.len(), 121); // 11x11

        // Sum should be ~1.0 (normalized)
        let sum: f64 = window.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Center value should be largest
        let center = window[5 * 11 + 5];
        assert!(center > window[0]); // Center > corner
    }

    #[test]
    fn test_ssim_custom_window() {
        let calc = SsimCalculator::with_window_size(7);
        assert_eq!(calc.window_size, 7);
        assert_eq!(calc.window.len(), 49);
    }

    #[test]
    fn test_ssim_range() {
        let calc = SsimCalculator::new();

        // Identical frames should give SSIM close to 1
        let frame1 = create_test_frame(64, 64, 128);
        let frame2 = create_test_frame(64, 64, 128);
        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score >= 0.99);

        // Very different frames should give lower SSIM
        let frame3 = create_test_frame(64, 64, 0);
        let frame4 = create_test_frame(64, 64, 255);
        let result2 = calc
            .calculate(&frame3, &frame4)
            .expect("should succeed in test");
        assert!(result2.score < 0.5);
    }

    // ── Region (ROI) SSIM tests ──────────────────────────────────────────

    #[test]
    fn test_ssim_region_identical() {
        let calc = SsimCalculator::new();
        let f = create_test_frame(64, 64, 128);
        let result = calc
            .calculate_region(&f, &f, (8, 8, 32, 32))
            .expect("region SSIM should succeed");
        assert!(
            (result.score - 1.0).abs() < 0.02,
            "SSIM of identical region must be ~1.0, got {}",
            result.score
        );
    }

    #[test]
    fn test_ssim_region_oob_errors() {
        let calc = SsimCalculator::new();
        let f = create_test_frame(64, 64, 128);
        let result = calc.calculate_region(&f, &f, (50, 50, 32, 32));
        assert!(result.is_err(), "out-of-bounds ROI must return error");
    }

    #[test]
    fn test_ssim_region_has_chroma_components() {
        let calc = SsimCalculator::new();
        let f1 = create_test_frame(64, 64, 100);
        let f2 = create_test_frame(64, 64, 110);
        let result = calc
            .calculate_region(&f1, &f2, (0, 0, 32, 32))
            .expect("region should succeed");
        assert!(result.components.contains_key("Y"));
        assert!(result.components.contains_key("Cb"));
        assert!(result.components.contains_key("Cr"));
    }

    #[test]
    fn test_ssim_region_mismatched_dims_errors() {
        let calc = SsimCalculator::new();
        let f1 = create_test_frame(64, 64, 128);
        let f2 = create_test_frame(32, 32, 128);
        let result = calc.calculate_region(&f1, &f2, (0, 0, 16, 16));
        assert!(result.is_err());
    }
}
