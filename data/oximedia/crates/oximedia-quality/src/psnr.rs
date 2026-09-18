//! Peak Signal-to-Noise Ratio (PSNR) calculation.
//!
//! PSNR is a widely used objective metric for measuring reconstruction quality
//! in lossy compression. It's based on the mean squared error (MSE) between
//! the reference and distorted signals.
//!
//! # Formula
//!
//! PSNR = 10 × log₁₀(MAX²/MSE)
//!
//! where MAX is the maximum possible pixel value (255 for 8-bit)
//! and MSE is the mean squared error.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// PSNR calculator for video quality assessment.
pub struct PsnrCalculator {
    /// Weight for luma (Y) component
    luma_weight: f64,
    /// Weight for chroma (Cb, Cr) components
    chroma_weight: f64,
}

impl PsnrCalculator {
    /// Creates a new PSNR calculator with default weights.
    ///
    /// Uses standard weights: Y=4, Cb=1, Cr=1 (normalized to sum to 1).
    #[must_use]
    pub fn new() -> Self {
        Self {
            luma_weight: 4.0 / 6.0,
            chroma_weight: 1.0 / 6.0,
        }
    }

    /// Creates a PSNR calculator with custom component weights.
    #[must_use]
    pub fn with_weights(luma_weight: f64, chroma_weight: f64) -> Self {
        let total = luma_weight + 2.0 * chroma_weight;
        Self {
            luma_weight: luma_weight / total,
            chroma_weight: chroma_weight / total,
        }
    }

    /// Calculates PSNR between reference and distorted frames.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match or if frames are invalid.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> OxiResult<QualityScore> {
        let mut score = QualityScore::new(MetricType::Psnr, 0.0);

        // Calculate Y PSNR
        let y_psnr = self.calculate_plane(&reference.planes[0], &distorted.planes[0])?;
        score.add_component("Y", y_psnr);

        let mut weighted_psnr = self.luma_weight * y_psnr;

        // Calculate chroma PSNR if available
        if reference.planes.len() >= 3 && distorted.planes.len() >= 3 {
            let cb_psnr = self.calculate_plane(&reference.planes[1], &distorted.planes[1])?;
            let cr_psnr = self.calculate_plane(&reference.planes[2], &distorted.planes[2])?;

            score.add_component("Cb", cb_psnr);
            score.add_component("Cr", cr_psnr);

            weighted_psnr += self.chroma_weight * (cb_psnr + cr_psnr);
        }

        score.score = weighted_psnr;
        Ok(score)
    }

    /// Calculates PSNR for a single plane.
    fn calculate_plane(&self, reference: &[u8], distorted: &[u8]) -> OxiResult<f64> {
        if reference.len() != distorted.len() {
            return Err(oximedia_core::OxiError::InvalidData(
                "Plane sizes must match".to_string(),
            ));
        }

        if reference.is_empty() {
            return Ok(f64::INFINITY);
        }

        // Calculate MSE using parallel processing for large planes
        let mse = if reference.len() > 10000 {
            self.calculate_mse_parallel(reference, distorted)
        } else {
            self.calculate_mse(reference, distorted)
        };

        if mse < 1e-10 {
            // Practically identical
            return Ok(100.0); // Return a very high PSNR
        }

        // PSNR = 10 * log10(MAX^2 / MSE)
        // For 8-bit: MAX = 255
        let max_value = 255.0;
        let psnr = 10.0 * (max_value * max_value / mse).log10();

        Ok(psnr)
    }

    /// Calculates mean squared error sequentially.
    fn calculate_mse(&self, reference: &[u8], distorted: &[u8]) -> f64 {
        let sum_squared_diff: u64 = reference
            .iter()
            .zip(distorted.iter())
            .map(|(r, d)| {
                let diff = i32::from(*r) - i32::from(*d);
                (diff * diff) as u64
            })
            .sum();

        sum_squared_diff as f64 / reference.len() as f64
    }

    /// Calculates mean squared error using parallel processing.
    fn calculate_mse_parallel(&self, reference: &[u8], distorted: &[u8]) -> f64 {
        let sum_squared_diff: u64 = reference
            .par_iter()
            .zip(distorted.par_iter())
            .map(|(r, d)| {
                let diff = i32::from(*r) - i32::from(*d);
                (diff * diff) as u64
            })
            .sum();

        sum_squared_diff as f64 / reference.len() as f64
    }

    /// Calculates PSNR for a specific region-of-interest (ROI) within the frames.
    ///
    /// `roi` is `(x, y, width, height)` in pixels — all relative to the luma plane.
    /// Chroma planes are cropped proportionally using the frame's chroma subsampling.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match, or if the ROI extends
    /// outside the frame boundaries.
    pub fn calculate_region(
        &self,
        reference: &Frame,
        distorted: &Frame,
        roi: (usize, usize, usize, usize),
    ) -> OxiResult<QualityScore> {
        let (rx, ry, rw, rh) = roi;
        if rx + rw > reference.width || ry + rh > reference.height {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "ROI ({rx},{ry},{rw},{rh}) exceeds frame bounds {}×{}",
                reference.width, reference.height
            )));
        }
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match for region PSNR".to_string(),
            ));
        }

        let mut score = QualityScore::new(MetricType::Psnr, 0.0);

        // Extract luma ROI and compute PSNR
        let ref_y_roi =
            extract_plane_roi(&reference.planes[0], reference.strides[0], rx, ry, rw, rh);
        let dist_y_roi =
            extract_plane_roi(&distorted.planes[0], distorted.strides[0], rx, ry, rw, rh);
        let y_psnr = self.calculate_plane(&ref_y_roi, &dist_y_roi)?;
        score.add_component("Y", y_psnr);
        let mut weighted = self.luma_weight * y_psnr;

        // Chroma ROI — scale by subsampling factors
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

            let cb_psnr = self.calculate_plane(&ref_cb, &dist_cb)?;
            let cr_psnr = self.calculate_plane(&ref_cr, &dist_cr)?;
            score.add_component("Cb", cb_psnr);
            score.add_component("Cr", cr_psnr);
            weighted += self.chroma_weight * (cb_psnr + cr_psnr);
        }

        score.score = weighted;
        Ok(score)
    }

    /// Calculates PSNR for 10-bit or 12-bit content.
    #[allow(dead_code)]
    fn calculate_plane_high_bit_depth(
        &self,
        reference: &[u16],
        distorted: &[u16],
        bit_depth: u32,
    ) -> OxiResult<f64> {
        if reference.len() != distorted.len() {
            return Err(oximedia_core::OxiError::InvalidData(
                "Plane sizes must match".to_string(),
            ));
        }

        if reference.is_empty() {
            return Ok(f64::INFINITY);
        }

        let sum_squared_diff: u64 = reference
            .iter()
            .zip(distorted.iter())
            .map(|(r, d)| {
                let diff = i64::from(*r) - i64::from(*d);
                (diff * diff) as u64
            })
            .sum();

        let mse = sum_squared_diff as f64 / reference.len() as f64;

        if mse < 1e-10 {
            return Ok(100.0);
        }

        let max_value = f64::from((1u32 << bit_depth) - 1);
        let psnr = 10.0 * (max_value * max_value / mse).log10();

        Ok(psnr)
    }
}

/// Extracts a rectangular region from a plane as a packed (stride-removed) buffer.
///
/// `stride` is the number of bytes per row. `x`, `y`, `w`, `h` are in pixels.
pub(crate) fn extract_plane_roi(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h);
    for row in y..y + h {
        let start = row * stride + x;
        let end = start + w;
        if end <= plane.len() {
            out.extend_from_slice(&plane[start..end]);
        }
    }
    out
}

impl Default for PsnrCalculator {
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
    fn test_psnr_identical_frames() {
        let calc = PsnrCalculator::new();
        let frame1 = create_test_frame(64, 64, 128);
        let frame2 = create_test_frame(64, 64, 128);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score > 99.0); // Very high PSNR for identical frames
    }

    #[test]
    fn test_psnr_different_frames() {
        let calc = PsnrCalculator::new();
        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(64, 64, 110);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score > 0.0 && result.score < 100.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_psnr_custom_weights() {
        let calc = PsnrCalculator::with_weights(1.0, 0.5);
        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(64, 64, 110);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_mse_calculation() {
        let calc = PsnrCalculator::new();
        let ref_plane = vec![100u8; 100];
        let dist_plane = vec![110u8; 100];

        let mse = calc.calculate_mse(&ref_plane, &dist_plane);
        assert!((mse - 100.0).abs() < 0.01); // (110-100)^2 = 100
    }

    #[test]
    fn test_psnr_formula() {
        let calc = PsnrCalculator::new();

        // Create frames with known MSE
        let mut frame1 = create_test_frame(10, 10, 0);
        let mut frame2 = create_test_frame(10, 10, 0);

        // Set specific values to get MSE = 100 for Y plane
        frame1.planes[0].fill(100);
        frame2.planes[0].fill(110);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");

        // Y PSNR ≈ 28.13, Cb/Cr PSNR ≈ 100 (identical chroma)
        // Weighted: (4/6)*28 + (2/6)*100 ≈ 52
        assert!(result.score > 45.0 && result.score < 60.0);
    }

    // ── Region (ROI) PSNR tests ──────────────────────────────────────────

    #[test]
    fn test_psnr_region_identical() {
        let calc = PsnrCalculator::new();
        let f = create_test_frame(64, 64, 128);
        let result = calc
            .calculate_region(&f, &f, (8, 8, 32, 32))
            .expect("region PSNR should succeed");
        assert!(
            result.score > 99.0,
            "identical region PSNR must be very high"
        );
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_psnr_region_oob_errors() {
        let calc = PsnrCalculator::new();
        let f1 = create_test_frame(64, 64, 100);
        let f2 = create_test_frame(64, 64, 110);
        // ROI extends beyond frame bounds
        let result = calc.calculate_region(&f1, &f2, (50, 50, 32, 32));
        assert!(result.is_err(), "out-of-bounds ROI must return error");
    }

    #[test]
    fn test_psnr_region_smaller_than_full() {
        let calc = PsnrCalculator::new();
        // Create a frame where left half is 100 and right half is 200.
        let mut f1 = create_test_frame(64, 64, 100);
        let mut f2 = create_test_frame(64, 64, 100);
        // Set right half of distorted to a different value
        for row in 0..64 {
            for col in 32..64 {
                f2.planes[0][row * 64 + col] = 200;
            }
        }
        // Full frame includes the large-difference right half → lower PSNR
        let full_result = calc.calculate(&f1, &f2).expect("full");
        // Left half only — f1 and f2 are identical in this region → high PSNR
        let region_result = calc
            .calculate_region(&f1, &f2, (0, 0, 32, 64))
            .expect("left region");
        assert!(
            region_result.score > full_result.score,
            "left region (identical) must have higher PSNR than full frame, \
             region={:.2} full={:.2}",
            region_result.score,
            full_result.score
        );
        // Suppress unused mut warning
        let _ = &mut f1;
    }

    #[test]
    fn test_extract_plane_roi_basic() {
        // 4×4 plane with stride=4, values 0..15
        let plane: Vec<u8> = (0..16).collect();
        // Extract 2×2 starting at (1,1)
        let roi = extract_plane_roi(&plane, 4, 1, 1, 2, 2);
        assert_eq!(roi, vec![5, 6, 9, 10]);
    }
}
