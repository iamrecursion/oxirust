//! Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE).
//!
//! BRISQUE is a no-reference image quality assessment metric that operates
//! in the spatial domain. It uses scene statistics of locally normalized
//! luminance coefficients to quantify possible losses of naturalness.
//!
//! # Reference
//!
//! A. Mittal et al., "No-Reference Image Quality Assessment in the Spatial Domain,"
//! IEEE Transactions on Image Processing, 2012.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;

/// BRISQUE assessor for no-reference quality assessment.
pub struct BrisqueAssessor {
    /// Number of scales to analyze
    num_scales: usize,
}

impl BrisqueAssessor {
    /// Creates a new BRISQUE assessor with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self { num_scales: 2 }
    }

    /// Assesses quality of a frame (no reference needed).
    ///
    /// Returns lower scores for better quality (0-100 scale).
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is too small.
    pub fn assess(&self, frame: &Frame) -> OxiResult<QualityScore> {
        let mut score = QualityScore::new(MetricType::Brisque, 0.0);

        // Calculate BRISQUE for Y plane
        let y_brisque = self.assess_plane(&frame.planes[0], frame.width, frame.height)?;

        score.add_component("Y", y_brisque);
        score.score = y_brisque;

        Ok(score)
    }

    /// Assesses a single plane.
    fn assess_plane(&self, plane: &[u8], width: usize, height: usize) -> OxiResult<f64> {
        if width < 32 || height < 32 {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame too small for BRISQUE assessment".to_string(),
            ));
        }

        let mut all_features = Vec::new();

        let mut current_plane = self.plane_to_f64(plane);
        let mut current_width = width;
        let mut current_height = height;

        // Extract features at multiple scales
        for _scale in 0..self.num_scales {
            let mscn = self.compute_mscn(&current_plane, current_width, current_height);
            let features = self.extract_features(&mscn, current_width, current_height);
            all_features.extend(features);

            // Downsample for next scale
            if current_width > 64 && current_height > 64 {
                let (downsampled, new_w, new_h) =
                    self.downsample(&current_plane, current_width, current_height);
                current_plane = downsampled;
                current_width = new_w;
                current_height = new_h;
            } else {
                break;
            }
        }

        // Compute quality score from features
        let quality_score = self.compute_quality_score(&all_features);

        Ok(quality_score)
    }

    /// Computes Mean Subtracted Contrast Normalized (MSCN) coefficients.
    fn compute_mscn(&self, plane: &[f64], width: usize, height: usize) -> Vec<f64> {
        let window_size = 7;
        let half_window = window_size / 2;
        let c = 1.0; // Constant to avoid instability

        let mut mscn = vec![0.0; plane.len()];

        for y in half_window..height - half_window {
            for x in half_window..width - half_window {
                let center_idx = y * width + x;

                // Gaussian weighted local mean and variance
                let (mean, variance) =
                    self.compute_local_statistics(plane, x, y, width, window_size);

                let std_dev = (variance + c).sqrt();

                // MSCN coefficient
                mscn[center_idx] = (plane[center_idx] - mean) / std_dev;
            }
        }

        mscn
    }

    /// Computes local mean and variance with Gaussian weighting.
    fn compute_local_statistics(
        &self,
        plane: &[f64],
        cx: usize,
        cy: usize,
        width: usize,
        window_size: usize,
    ) -> (f64, f64) {
        let half_window = window_size / 2;
        let sigma = 7.0 / 6.0;

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut weight_sum = 0.0;

        for dy in 0..window_size {
            let y = cy + dy - half_window;
            for dx in 0..window_size {
                let x = cx + dx - half_window;

                let dist_sq = (dx as f64 - half_window as f64).powi(2)
                    + (dy as f64 - half_window as f64).powi(2);
                let weight = (-dist_sq / (2.0 * sigma * sigma)).exp();

                let idx = y * width + x;
                let value = plane[idx];

                sum += weight * value;
                sum_sq += weight * value * value;
                weight_sum += weight;
            }
        }

        let mean = sum / weight_sum;
        let variance = (sum_sq / weight_sum) - mean * mean;

        (mean, variance.max(0.0))
    }

    /// Extracts statistical features from MSCN coefficients.
    fn extract_features(&self, mscn: &[f64], width: usize, height: usize) -> Vec<f64> {
        let mut features = Vec::new();

        // Filter valid coefficients (non-zero)
        let valid_coeffs: Vec<f64> = mscn.iter().copied().filter(|&v| v != 0.0).collect();

        if valid_coeffs.is_empty() {
            return vec![0.0; 36];
        }

        // 1. Shape parameters of MSCN coefficient distribution
        features.extend(self.fit_aggd_parameters(&valid_coeffs));

        // 2. Pairwise products in four directions
        let shifts = [(1, 0), (0, 1), (1, 1), (1, -1)]; // H, V, D1, D2

        for &(dx, dy) in &shifts {
            let products = self.compute_pairwise_products(mscn, width, height, dx, dy);
            if products.is_empty() {
                features.extend(vec![0.0; 4]);
            } else {
                features.extend(self.fit_aggd_parameters(&products));
            }
        }

        // Pad to fixed size
        while features.len() < 36 {
            features.push(0.0);
        }

        features.truncate(36);
        features
    }

    /// Fits Asymmetric Generalized Gaussian Distribution (AGGD) parameters.
    fn fit_aggd_parameters(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![0.0; 4];
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64;

        let std_dev = variance.sqrt();

        // Skewness
        let skewness = if std_dev > 1e-10 {
            data.iter()
                .map(|v| ((v - mean) / std_dev).powi(3))
                .sum::<f64>()
                / data.len() as f64
        } else {
            0.0
        };

        // Kurtosis
        let kurtosis = if std_dev > 1e-10 {
            data.iter()
                .map(|v| ((v - mean) / std_dev).powi(4))
                .sum::<f64>()
                / data.len() as f64
                - 3.0
        } else {
            0.0
        };

        vec![mean, variance, skewness, kurtosis]
    }

    /// Computes pairwise products of MSCN coefficients.
    fn compute_pairwise_products(
        &self,
        mscn: &[f64],
        width: usize,
        height: usize,
        dx: i32,
        dy: i32,
    ) -> Vec<f64> {
        let mut products = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let idx1 = y * width + x;
                    let idx2 = ny as usize * width + nx as usize;

                    if mscn[idx1] != 0.0 && mscn[idx2] != 0.0 {
                        products.push(mscn[idx1] * mscn[idx2]);
                    }
                }
            }
        }

        products
    }

    /// Computes quality score from features.
    fn compute_quality_score(&self, features: &[f64]) -> f64 {
        // Simplified scoring using feature statistics
        // In a full implementation, this would use SVR (Support Vector Regression)
        // with pre-trained model

        if features.is_empty() {
            return 50.0; // Default mid-range score
        }

        let mut score = 0.0;
        let mut count = 0.0;

        for &feature in features {
            score += feature.abs();
            count += 1.0;
        }

        let avg_deviation = score / count;

        // Map to 0-100 scale (lower is better)
        (avg_deviation * 10.0).min(100.0)
    }

    /// Converts u8 plane to f64.
    fn plane_to_f64(&self, plane: &[u8]) -> Vec<f64> {
        plane.iter().map(|&v| f64::from(v)).collect()
    }

    /// Downsamples plane by factor of 2.
    fn downsample(&self, plane: &[f64], width: usize, height: usize) -> (Vec<f64>, usize, usize) {
        let new_width = width / 2;
        let new_height = height / 2;
        let mut downsampled = vec![0.0; new_width * new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let sum = plane[2 * y * width + 2 * x]
                    + plane[2 * y * width + 2 * x + 1]
                    + plane[(2 * y + 1) * width + 2 * x]
                    + plane[(2 * y + 1) * width + 2 * x + 1];
                downsampled[y * new_width + x] = sum / 4.0;
            }
        }

        (downsampled, new_width, new_height)
    }
}

impl Default for BrisqueAssessor {
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
    fn test_brisque_assessment() {
        let assessor = BrisqueAssessor::new();
        let frame = create_test_frame(128, 128, 128);

        let result = assessor.assess(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0 && result.score <= 100.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_mscn_computation() {
        let assessor = BrisqueAssessor::new();
        let plane: Vec<f64> = (0..10000).map(|i| (i % 256) as f64).collect();

        let mscn = assessor.compute_mscn(&plane, 100, 100);
        assert_eq!(mscn.len(), 10000);
        assert!(mscn.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_feature_extraction() {
        let assessor = BrisqueAssessor::new();
        let mscn: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 100.0).collect();

        let features = assessor.extract_features(&mscn, 10, 100);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_aggd_fitting() {
        let assessor = BrisqueAssessor::new();
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();

        let params = assessor.fit_aggd_parameters(&data);
        assert_eq!(params.len(), 4);
        // Mean should be around middle of range
        assert!((params[0] - 4.95).abs() < 1.0);
    }

    #[test]
    fn test_pairwise_products() {
        let assessor = BrisqueAssessor::new();
        let mscn: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();

        let products = assessor.compute_pairwise_products(&mscn, 10, 10, 1, 0);
        assert!(!products.is_empty());
    }
}
