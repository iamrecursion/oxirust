//! Natural Image Quality Evaluator (NIQE) - No-reference quality assessment.
//!
//! NIQE is a completely blind/no-reference image quality analyzer based on
//! measuring deviations from statistical regularities observed in natural images.
//! Lower NIQE scores indicate better perceptual quality.
//!
//! # Reference
//!
//! A. Mittal et al., "Making a 'Completely Blind' Image Quality Analyzer,"
//! IEEE Signal Processing Letters, 2013.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;

/// NIQE assessor for no-reference quality assessment.
pub struct NiqeAssessor {
    /// Patch size for local statistics
    patch_size: usize,
    /// Number of orientations for Gabor-like filtering
    num_orientations: usize,
}

impl NiqeAssessor {
    /// Creates a new NIQE assessor with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            patch_size: 96,
            num_orientations: 2,
        }
    }

    /// Assesses quality of a frame (no reference needed).
    ///
    /// Returns lower scores for better quality.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is too small.
    pub fn assess(&self, frame: &Frame) -> OxiResult<QualityScore> {
        let mut score = QualityScore::new(MetricType::Niqe, 0.0);

        // Calculate NIQE for Y plane
        let y_niqe = self.assess_plane(&frame.planes[0], frame.width, frame.height)?;

        score.add_component("Y", y_niqe);
        score.score = y_niqe;

        Ok(score)
    }

    /// Assesses a single plane.
    fn assess_plane(&self, plane: &[u8], width: usize, height: usize) -> OxiResult<f64> {
        if width < self.patch_size || height < self.patch_size {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame too small for NIQE assessment".to_string(),
            ));
        }

        // Convert to f64
        let plane_f64: Vec<f64> = plane.iter().map(|&v| f64::from(v)).collect();

        // Compute local mean-subtracted contrast normalized (MSCN) coefficients
        let mscn = self.compute_mscn(&plane_f64, width, height);

        // Extract features from MSCN coefficients
        let features = self.extract_features(&mscn, width, height);

        // Compute distance from natural scene statistics model
        // In a full implementation, this would compare against a pre-trained model
        // Here we use a simplified heuristic
        let quality_score = self.compute_quality_score(&features);

        Ok(quality_score)
    }

    /// Computes Mean Subtracted Contrast Normalized (MSCN) coefficients.
    fn compute_mscn(&self, plane: &[f64], width: usize, height: usize) -> Vec<f64> {
        let window_size = 7;
        let half_window = window_size / 2;

        let mut mscn = vec![0.0; plane.len()];

        for y in half_window..height - half_window {
            for x in half_window..width - half_window {
                let center_idx = y * width + x;

                // Compute local mean
                let mut sum = 0.0;
                let mut count = 0.0;

                for dy in 0..window_size {
                    for dx in 0..window_size {
                        let idx = (y + dy - half_window) * width + (x + dx - half_window);
                        sum += plane[idx];
                        count += 1.0;
                    }
                }

                let mean = sum / count;

                // Compute local variance
                let mut var_sum = 0.0;
                for dy in 0..window_size {
                    for dx in 0..window_size {
                        let idx = (y + dy - half_window) * width + (x + dx - half_window);
                        let diff = plane[idx] - mean;
                        var_sum += diff * diff;
                    }
                }

                let variance = var_sum / count;
                let std_dev = (variance + 1.0).sqrt(); // Add 1 to avoid division by zero

                // MSCN coefficient
                mscn[center_idx] = (plane[center_idx] - mean) / std_dev;
            }
        }

        mscn
    }

    /// Extracts statistical features from MSCN coefficients.
    fn extract_features(&self, mscn: &[f64], _width: usize, _height: usize) -> Vec<f64> {
        let mut features = Vec::new();

        // Filter out zero values (unprocessed border pixels)
        let valid_coeffs: Vec<f64> = mscn.iter().copied().filter(|&v| v != 0.0).collect();

        if valid_coeffs.is_empty() {
            return vec![0.0; 18];
        }

        // Shape parameter (approximation using moments)
        let mean = valid_coeffs.iter().sum::<f64>() / valid_coeffs.len() as f64;
        let variance = valid_coeffs.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / valid_coeffs.len() as f64;

        let std_dev = variance.sqrt();

        features.push(mean.abs());
        features.push(variance);
        features.push(std_dev);

        // Skewness
        let skewness = if std_dev > 1e-10 {
            valid_coeffs
                .iter()
                .map(|v| ((v - mean) / std_dev).powi(3))
                .sum::<f64>()
                / valid_coeffs.len() as f64
        } else {
            0.0
        };
        features.push(skewness);

        // Kurtosis
        let kurtosis = if std_dev > 1e-10 {
            valid_coeffs
                .iter()
                .map(|v| ((v - mean) / std_dev).powi(4))
                .sum::<f64>()
                / valid_coeffs.len() as f64
                - 3.0
        } else {
            0.0
        };
        features.push(kurtosis);

        // Pairwise products for adjacent coefficients (simplified)
        let products: Vec<f64> = valid_coeffs.windows(2).map(|w| w[0] * w[1]).collect();

        if products.is_empty() {
            features.push(0.0);
            features.push(0.0);
        } else {
            let prod_mean = products.iter().sum::<f64>() / products.len() as f64;
            let prod_var = products
                .iter()
                .map(|v| (v - prod_mean).powi(2))
                .sum::<f64>()
                / products.len() as f64;

            features.push(prod_mean);
            features.push(prod_var);
        }

        // Pad to fixed size
        while features.len() < 18 {
            features.push(0.0);
        }

        features
    }

    /// Computes quality score from features.
    ///
    /// Lower scores indicate better quality.
    fn compute_quality_score(&self, features: &[f64]) -> f64 {
        // Simplified scoring based on deviation from expected natural statistics
        // In a full implementation, this would use Mahalanobis distance with
        // pre-trained multivariate Gaussian model

        let expected_mean = 0.0;
        let expected_variance = 1.0;

        let mean_deviation = (features[0] - expected_mean).abs();
        let variance_deviation = (features[1] - expected_variance).abs();
        let skewness_deviation = features[3].abs();
        let kurtosis_deviation = features[4].abs();

        // Weighted combination (lower is better)
        let score = mean_deviation * 0.3
            + variance_deviation * 0.3
            + skewness_deviation * 0.2
            + kurtosis_deviation * 0.2;

        // Scale to reasonable range (0-100, lower is better)
        (score * 20.0).min(100.0)
    }
}

impl Default for NiqeAssessor {
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

    fn create_noisy_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        for i in 0..frame.planes[0].len() {
            frame.planes[0][i] = ((i * 37) % 256) as u8; // Pseudo-random pattern
        }
        frame
    }

    #[test]
    fn test_niqe_assessment() {
        let assessor = NiqeAssessor::new();
        let frame = create_test_frame(128, 128, 128);

        let result = assessor.assess(&frame).expect("should succeed in test");
        assert!(result.score >= 0.0);
        assert!(result.components.contains_key("Y"));
    }

    #[test]
    fn test_niqe_different_qualities() {
        let assessor = NiqeAssessor::new();

        // Uniform frame (unnatural)
        let uniform = create_test_frame(128, 128, 128);
        let uniform_score = assessor.assess(&uniform).expect("should succeed in test");

        // Noisy frame (also unnatural)
        let noisy = create_noisy_frame(128, 128);
        let noisy_score = assessor.assess(&noisy).expect("should succeed in test");

        // Both should have some score
        assert!(uniform_score.score >= 0.0);
        assert!(noisy_score.score >= 0.0);
    }

    #[test]
    fn test_mscn_computation() {
        let assessor = NiqeAssessor::new();
        let plane: Vec<f64> = (0..10000).map(|i| (i % 256) as f64).collect();

        let mscn = assessor.compute_mscn(&plane, 100, 100);
        assert_eq!(mscn.len(), 10000);

        // Check that some coefficients are non-zero
        assert!(mscn.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn test_feature_extraction() {
        let assessor = NiqeAssessor::new();
        let mscn: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 100.0).collect();

        let features = assessor.extract_features(&mscn, 10, 100);
        assert!(!features.is_empty());
        assert!(features.len() >= 4); // At least mean, variance, skewness, kurtosis
    }
}
