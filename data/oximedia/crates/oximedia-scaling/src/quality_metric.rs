//! Scale quality assessment metrics.
//!
//! Provides numerical metrics for evaluating the quality of a scaling
//! operation by comparing reference and scaled pixel buffers. Includes
//! PSNR, MSE, MAE, and a simple sharpness estimate.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use serde::{Deserialize, Serialize};

// -- QualityScore ------------------------------------------------------------

/// A single quality score with its metric name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityScore {
    /// Name of the metric (e.g. "PSNR", "MSE").
    pub metric: String,
    /// Numeric value of the metric.
    pub value: f64,
    /// Unit of measurement.
    pub unit: String,
}

impl QualityScore {
    /// Create a new score.
    pub fn new(metric: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            metric: metric.into(),
            value,
            unit: unit.into(),
        }
    }
}

impl std::fmt::Display for QualityScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:.4} {}", self.metric, self.value, self.unit)
    }
}

// -- ScaleQualityMetrics -----------------------------------------------------

/// Computes quality metrics between a reference and processed image.
///
/// Both images are represented as flat `f32` buffers with values in `[0.0, 1.0]`
/// (normalised). All metrics assume matching buffer lengths.
///
/// # Example
/// ```
/// use oximedia_scaling::quality_metric::ScaleQualityMetrics;
///
/// let reference = vec![0.5f32; 100];
/// let processed = vec![0.5f32; 100];
/// let mse = ScaleQualityMetrics::mse(&reference, &processed);
/// assert!(mse < 1e-10);
/// ```
pub struct ScaleQualityMetrics;

impl ScaleQualityMetrics {
    /// Mean Squared Error between two buffers.
    pub fn mse(reference: &[f32], processed: &[f32]) -> f64 {
        if reference.is_empty() {
            return 0.0;
        }
        let n = reference.len().min(processed.len());
        let sum: f64 = reference[..n]
            .iter()
            .zip(processed[..n].iter())
            .map(|(&r, &p)| {
                let d = (r as f64) - (p as f64);
                d * d
            })
            .sum();
        sum / n as f64
    }

    /// Peak Signal-to-Noise Ratio (PSNR) in dB.
    ///
    /// `max_val` is the peak signal value (1.0 for normalised buffers,
    /// 255.0 for 8-bit). Returns `f64::INFINITY` when MSE is zero.
    pub fn psnr(reference: &[f32], processed: &[f32], max_val: f64) -> f64 {
        let mse = Self::mse(reference, processed);
        if mse < 1e-15 {
            return f64::INFINITY;
        }
        10.0 * (max_val * max_val / mse).log10()
    }

    /// Mean Absolute Error.
    pub fn mae(reference: &[f32], processed: &[f32]) -> f64 {
        if reference.is_empty() {
            return 0.0;
        }
        let n = reference.len().min(processed.len());
        let sum: f64 = reference[..n]
            .iter()
            .zip(processed[..n].iter())
            .map(|(&r, &p)| ((r as f64) - (p as f64)).abs())
            .sum();
        sum / n as f64
    }

    /// Maximum absolute pixel difference.
    pub fn max_error(reference: &[f32], processed: &[f32]) -> f64 {
        let n = reference.len().min(processed.len());
        reference[..n]
            .iter()
            .zip(processed[..n].iter())
            .map(|(&r, &p)| ((r as f64) - (p as f64)).abs())
            .fold(0.0f64, f64::max)
    }

    /// Simple sharpness estimate using variance of the Laplacian.
    ///
    /// Operates on a row-major 2-D buffer of dimensions `width x height`.
    /// Higher values indicate sharper images.
    pub fn sharpness_estimate(buf: &[f32], width: usize, height: usize) -> f64 {
        if width < 3 || height < 3 || buf.len() < width * height {
            return 0.0;
        }
        let mut sum = 0.0f64;
        let mut count = 0u64;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let center = buf[idx] as f64;
                let laplacian = -4.0 * center
                    + buf[idx - 1] as f64
                    + buf[idx + 1] as f64
                    + buf[idx - width] as f64
                    + buf[idx + width] as f64;
                sum += laplacian * laplacian;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Compute a full quality report as a list of [`QualityScore`] values.
    pub fn full_report(reference: &[f32], processed: &[f32]) -> Vec<QualityScore> {
        vec![
            QualityScore::new("MSE", Self::mse(reference, processed), ""),
            QualityScore::new("PSNR", Self::psnr(reference, processed, 1.0), "dB"),
            QualityScore::new("MAE", Self::mae(reference, processed), ""),
            QualityScore::new("MaxError", Self::max_error(reference, processed), ""),
        ]
    }

    /// Return `true` if PSNR exceeds a given threshold (common pass criterion).
    pub fn passes_quality_gate(reference: &[f32], processed: &[f32], min_psnr: f64) -> bool {
        Self::psnr(reference, processed, 1.0) >= min_psnr
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_identical() {
        let a = vec![0.5f32; 100];
        let b = vec![0.5f32; 100];
        assert!(ScaleQualityMetrics::mse(&a, &b) < 1e-10);
    }

    #[test]
    fn test_mse_different() {
        let a = vec![1.0f32; 100];
        let b = vec![0.0f32; 100];
        let mse = ScaleQualityMetrics::mse(&a, &b);
        assert!((mse - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_empty() {
        assert!(ScaleQualityMetrics::mse(&[], &[]) < 1e-10);
    }

    #[test]
    fn test_psnr_identical() {
        let a = vec![0.5f32; 100];
        let b = vec![0.5f32; 100];
        assert!(ScaleQualityMetrics::psnr(&a, &b, 1.0).is_infinite());
    }

    #[test]
    fn test_psnr_different() {
        let a = vec![1.0f32; 100];
        let b = vec![0.9f32; 100];
        let psnr = ScaleQualityMetrics::psnr(&a, &b, 1.0);
        assert!(psnr > 0.0 && psnr < 100.0);
    }

    #[test]
    fn test_mae_identical() {
        let a = vec![0.5f32; 100];
        let b = vec![0.5f32; 100];
        assert!(ScaleQualityMetrics::mae(&a, &b) < 1e-10);
    }

    #[test]
    fn test_mae_known_value() {
        let a = vec![1.0f32; 100];
        let b = vec![0.5f32; 100];
        let mae = ScaleQualityMetrics::mae(&a, &b);
        assert!((mae - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_error() {
        let a = vec![0.0f32, 0.5, 1.0];
        let b = vec![0.0f32, 0.3, 0.6];
        let me = ScaleQualityMetrics::max_error(&a, &b);
        assert!((me - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_sharpness_flat() {
        // Flat image -> laplacian = 0 everywhere
        let buf = vec![0.5f32; 10 * 10];
        let sharp = ScaleQualityMetrics::sharpness_estimate(&buf, 10, 10);
        assert!(sharp < 1e-10);
    }

    #[test]
    fn test_sharpness_edge() {
        // Image with a vertical edge should have non-zero sharpness
        let mut buf = vec![0.0f32; 10 * 10];
        for y in 0..10 {
            for x in 5..10 {
                buf[y * 10 + x] = 1.0;
            }
        }
        let sharp = ScaleQualityMetrics::sharpness_estimate(&buf, 10, 10);
        assert!(sharp > 0.0);
    }

    #[test]
    fn test_sharpness_too_small() {
        let buf = vec![0.5f32; 4];
        let sharp = ScaleQualityMetrics::sharpness_estimate(&buf, 2, 2);
        assert!(sharp < 1e-10);
    }

    #[test]
    fn test_full_report_length() {
        let a = vec![0.5f32; 100];
        let b = vec![0.5f32; 100];
        let report = ScaleQualityMetrics::full_report(&a, &b);
        assert_eq!(report.len(), 4);
    }

    #[test]
    fn test_quality_score_display() {
        let s = QualityScore::new("PSNR", 45.123, "dB");
        let disp = s.to_string();
        assert!(disp.contains("PSNR"));
        assert!(disp.contains("dB"));
    }

    #[test]
    fn test_passes_quality_gate_true() {
        let a = vec![1.0f32; 100];
        let b = vec![0.99f32; 100];
        // PSNR for MSE=0.0001 → 10*log10(1/0.0001) = 40 dB
        assert!(ScaleQualityMetrics::passes_quality_gate(&a, &b, 30.0));
    }

    #[test]
    fn test_passes_quality_gate_false() {
        let a = vec![1.0f32; 100];
        let b = vec![0.0f32; 100];
        // PSNR = 0 dB (MSE=1.0)
        assert!(!ScaleQualityMetrics::passes_quality_gate(&a, &b, 10.0));
    }

    #[test]
    fn test_mae_empty() {
        assert!(ScaleQualityMetrics::mae(&[], &[]) < 1e-10);
    }
}
