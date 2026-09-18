//! Quality metrics for evaluating denoising effectiveness.
//!
//! Provides objective metrics — SNR, PSNR, MSE, SSIM proxy — that can be
//! computed between a reference (pre-denoise) and a processed (post-denoise)
//! pixel buffer to quantify noise reduction quality.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Which objective quality metric to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMetric {
    /// Mean Squared Error between reference and processed.
    Mse,
    /// Peak Signal-to-Noise Ratio in dB.
    Psnr,
    /// Signal-to-Noise Ratio in dB (power ratio).
    Snr,
    /// Structural Similarity Index (simplified luminance component).
    Ssim,
    /// Total Variation — measures spatial smoothness of the output.
    TotalVariation,
}

impl std::fmt::Display for DenoiseMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Mse => "MSE",
            Self::Psnr => "PSNR",
            Self::Snr => "SNR",
            Self::Ssim => "SSIM",
            Self::TotalVariation => "TotalVariation",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// MetricResult
// ---------------------------------------------------------------------------

/// Result of computing a single quality metric.
#[derive(Debug, Clone)]
pub struct MetricResult {
    /// The metric that was computed.
    pub metric: DenoiseMetric,
    /// The computed value.
    pub value: f64,
    /// Optional human-readable unit label (e.g. "dB", "unitless").
    pub unit: &'static str,
}

impl MetricResult {
    /// Create a new metric result.
    #[must_use]
    pub fn new(metric: DenoiseMetric, value: f64, unit: &'static str) -> Self {
        Self {
            metric,
            value,
            unit,
        }
    }

    /// Returns `true` if the metric indicates acceptable quality (heuristic
    /// thresholds).
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        match self.metric {
            DenoiseMetric::Mse => self.value < 100.0,
            DenoiseMetric::Psnr => self.value > 30.0,
            DenoiseMetric::Snr => self.value > 20.0,
            DenoiseMetric::Ssim => self.value > 0.85,
            DenoiseMetric::TotalVariation => true, // no fixed threshold
        }
    }
}

impl std::fmt::Display for MetricResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:.4} {}", self.metric, self.value, self.unit)
    }
}

// ---------------------------------------------------------------------------
// DenoiseMetrics — the computation engine
// ---------------------------------------------------------------------------

/// Computes and stores a collection of quality metrics for a denoising
/// operation.
///
/// All methods expect raw luma (Y) pixel buffers of identical length.
///
/// # Example
///
/// ```rust
/// use oximedia_denoise::denoise_metrics::DenoiseMetrics;
///
/// let ref_px = vec![128u8; 256];
/// let out_px = vec![130u8; 256];
/// let m = DenoiseMetrics::new(&ref_px, &out_px);
/// let psnr = m.psnr();
/// assert!(psnr > 40.0);
/// ```
pub struct DenoiseMetrics {
    mse_val: f64,
    snr_val: f64,
    ssim_val: f64,
    tv_val: f64,
    n: usize,
}

impl DenoiseMetrics {
    /// Create metrics by computing all values from `reference` and `processed`
    /// pixel buffers.
    ///
    /// # Panics
    ///
    /// Panics if the buffers have different lengths.
    #[must_use]
    pub fn new(reference: &[u8], processed: &[u8]) -> Self {
        assert_eq!(
            reference.len(),
            processed.len(),
            "reference and processed must have the same length"
        );

        let n = reference.len();
        if n == 0 {
            return Self {
                mse_val: 0.0,
                snr_val: 0.0,
                ssim_val: 1.0,
                tv_val: 0.0,
                n: 0,
            };
        }

        let mse_val = Self::compute_mse(reference, processed);
        let snr_val = Self::compute_snr(reference, mse_val);
        let ssim_val = Self::compute_ssim(reference, processed);
        let tv_val = Self::compute_tv(processed);

        Self {
            mse_val,
            snr_val,
            ssim_val,
            tv_val,
            n,
        }
    }

    /// Mean Squared Error.
    #[must_use]
    pub fn mse(&self) -> f64 {
        self.mse_val
    }

    /// Peak Signal-to-Noise Ratio in dB.  Returns `f64::INFINITY` for zero MSE.
    #[must_use]
    pub fn psnr(&self) -> f64 {
        if self.mse_val <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0_f64 * 255.0 / self.mse_val).log10()
    }

    /// Signal-to-Noise Ratio in dB.
    #[must_use]
    pub fn snr_db(&self) -> f64 {
        self.snr_val
    }

    /// Structural Similarity Index (simplified, luminance only).
    #[must_use]
    pub fn ssim(&self) -> f64 {
        self.ssim_val
    }

    /// Total Variation of the processed buffer (lower = smoother).
    #[must_use]
    pub fn total_variation(&self) -> f64 {
        self.tv_val
    }

    /// Number of pixels used.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.n
    }

    /// Compute a specific metric by enum tag.
    #[must_use]
    pub fn compute(&self, metric: DenoiseMetric) -> MetricResult {
        let (value, unit) = match metric {
            DenoiseMetric::Mse => (self.mse_val, ""),
            DenoiseMetric::Psnr => (self.psnr(), "dB"),
            DenoiseMetric::Snr => (self.snr_val, "dB"),
            DenoiseMetric::Ssim => (self.ssim_val, "unitless"),
            DenoiseMetric::TotalVariation => (self.tv_val, ""),
        };
        MetricResult::new(metric, value, unit)
    }

    /// Return all metrics as a `Vec<MetricResult>`.
    #[must_use]
    pub fn all_metrics(&self) -> Vec<MetricResult> {
        vec![
            self.compute(DenoiseMetric::Mse),
            self.compute(DenoiseMetric::Psnr),
            self.compute(DenoiseMetric::Snr),
            self.compute(DenoiseMetric::Ssim),
            self.compute(DenoiseMetric::TotalVariation),
        ]
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn compute_mse(reference: &[u8], processed: &[u8]) -> f64 {
        let sum: f64 = reference
            .iter()
            .zip(processed.iter())
            .map(|(&r, &p)| {
                let diff = r as f64 - p as f64;
                diff * diff
            })
            .sum();
        sum / reference.len() as f64
    }

    fn compute_snr(reference: &[u8], mse: f64) -> f64 {
        if mse <= 0.0 {
            return f64::INFINITY;
        }
        let signal_power: f64 = reference
            .iter()
            .map(|&p| {
                let v = p as f64;
                v * v
            })
            .sum::<f64>()
            / reference.len() as f64;

        if signal_power <= 0.0 {
            return 0.0;
        }
        10.0 * (signal_power / mse).log10()
    }

    fn compute_ssim(reference: &[u8], processed: &[u8]) -> f64 {
        // Simplified SSIM: luminance-only, single global window
        let n = reference.len() as f64;
        let mu_x: f64 = reference.iter().map(|&p| p as f64).sum::<f64>() / n;
        let mu_y: f64 = processed.iter().map(|&p| p as f64).sum::<f64>() / n;

        let var_x: f64 = reference
            .iter()
            .map(|&p| {
                let d = p as f64 - mu_x;
                d * d
            })
            .sum::<f64>()
            / n;
        let var_y: f64 = processed
            .iter()
            .map(|&p| {
                let d = p as f64 - mu_y;
                d * d
            })
            .sum::<f64>()
            / n;
        let cov: f64 = reference
            .iter()
            .zip(processed.iter())
            .map(|(&r, &p)| (r as f64 - mu_x) * (p as f64 - mu_y))
            .sum::<f64>()
            / n;

        // SSIM constants
        let c1 = (0.01 * 255.0_f64).powi(2);
        let c2 = (0.03 * 255.0_f64).powi(2);

        let num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2);
        let den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);

        if den == 0.0 {
            1.0
        } else {
            num / den
        }
    }

    fn compute_tv(processed: &[u8]) -> f64 {
        if processed.len() < 2 {
            return 0.0;
        }
        processed
            .windows(2)
            .map(|w| (w[1] as f64 - w[0] as f64).abs())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identical_frames(value: u8, n: usize) -> (Vec<u8>, Vec<u8>) {
        (vec![value; n], vec![value; n])
    }

    fn offset_frames(base: u8, offset: i16, n: usize) -> (Vec<u8>, Vec<u8>) {
        let ref_px = vec![base; n];
        let proc_px: Vec<u8> = ref_px
            .iter()
            .map(|&p| (p as i16 + offset).clamp(0, 255) as u8)
            .collect();
        (ref_px, proc_px)
    }

    #[test]
    fn test_metric_display() {
        assert_eq!(DenoiseMetric::Psnr.to_string(), "PSNR");
        assert_eq!(DenoiseMetric::TotalVariation.to_string(), "TotalVariation");
    }

    #[test]
    fn test_metric_result_display() {
        let r = MetricResult::new(DenoiseMetric::Psnr, 42.5, "dB");
        let s = r.to_string();
        assert!(s.contains("PSNR"));
        assert!(s.contains("42.5"));
    }

    #[test]
    fn test_mse_identical_frames() {
        let (r, p) = identical_frames(128, 256);
        let m = DenoiseMetrics::new(&r, &p);
        assert!((m.mse() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_psnr_identical_frames() {
        let (r, p) = identical_frames(128, 256);
        let m = DenoiseMetrics::new(&r, &p);
        assert!(m.psnr().is_infinite());
    }

    #[test]
    fn test_psnr_offset_frames() {
        let (r, p) = offset_frames(128, 5, 256);
        let m = DenoiseMetrics::new(&r, &p);
        // MSE = 25, PSNR = 10*log10(255^2/25) ≈ 34.15 dB
        assert!(m.psnr() > 30.0, "psnr={}", m.psnr());
    }

    #[test]
    fn test_snr_identical_frames() {
        let (r, p) = identical_frames(100, 256);
        let m = DenoiseMetrics::new(&r, &p);
        assert!(m.snr_db().is_infinite());
    }

    #[test]
    fn test_snr_positive_for_offset() {
        let (r, p) = offset_frames(200, 2, 512);
        let m = DenoiseMetrics::new(&r, &p);
        assert!(m.snr_db() > 0.0);
    }

    #[test]
    fn test_ssim_identical() {
        let (r, p) = identical_frames(128, 256);
        let m = DenoiseMetrics::new(&r, &p);
        assert!((m.ssim() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ssim_range() {
        let ref_px: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let proc_px: Vec<u8> = (0..256).map(|i| ((i + 10) % 256) as u8).collect();
        let m = DenoiseMetrics::new(&ref_px, &proc_px);
        assert!((-1.0..=1.0).contains(&m.ssim()));
    }

    #[test]
    fn test_total_variation_flat() {
        let (_, p) = identical_frames(100, 256);
        let m = DenoiseMetrics::new(&p, &p);
        assert!((m.total_variation() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_total_variation_alternating() {
        let ref_px: Vec<u8> = (0..256).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        let m = DenoiseMetrics::new(&ref_px, &ref_px);
        // TV should be large for alternating 0/255
        assert!(m.total_variation() > 10.0);
    }

    #[test]
    fn test_pixel_count() {
        let (r, p) = identical_frames(0, 512);
        let m = DenoiseMetrics::new(&r, &p);
        assert_eq!(m.pixel_count(), 512);
    }

    #[test]
    fn test_all_metrics_count() {
        let (r, p) = offset_frames(100, 3, 128);
        let m = DenoiseMetrics::new(&r, &p);
        assert_eq!(m.all_metrics().len(), 5);
    }

    #[test]
    fn test_compute_by_tag() {
        let (r, p) = offset_frames(100, 10, 256);
        let m = DenoiseMetrics::new(&r, &p);
        let res = m.compute(DenoiseMetric::Mse);
        assert!((res.value - m.mse()).abs() < 1e-10);
    }

    #[test]
    fn test_is_acceptable_psnr_good() {
        let (r, p) = offset_frames(128, 1, 256);
        let m = DenoiseMetrics::new(&r, &p);
        let res = m.compute(DenoiseMetric::Psnr);
        // offset of 1 gives very high PSNR → acceptable
        assert!(res.is_acceptable());
    }

    #[test]
    fn test_is_acceptable_psnr_bad() {
        // large offset → low PSNR
        let r = vec![0u8; 256];
        let p = vec![255u8; 256];
        let m = DenoiseMetrics::new(&r, &p);
        let res = m.compute(DenoiseMetric::Psnr);
        assert!(!res.is_acceptable());
    }

    #[test]
    fn test_empty_buffers() {
        let m = DenoiseMetrics::new(&[], &[]);
        assert!((m.mse() - 0.0).abs() < 1e-10);
        assert!((m.ssim() - 1.0).abs() < 1e-10);
        assert_eq!(m.pixel_count(), 0);
    }
}
