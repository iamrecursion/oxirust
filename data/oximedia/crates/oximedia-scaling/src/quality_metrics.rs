//! Quality metrics for evaluating scaling operations.
//!
//! Provides PSNR, SSIM-like, MSE, MAE, and histogram-based quality
//! measurements for comparing original and scaled frames.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A quality score from a single metric.
#[derive(Debug, Clone, Copy)]
pub struct QualityScore {
    /// The metric that was computed.
    pub metric: MetricKind,
    /// The numeric score value.
    pub value: f64,
    /// Whether a higher score means better quality.
    pub higher_is_better: bool,
}

impl QualityScore {
    /// Create a new quality score.
    pub fn new(metric: MetricKind, value: f64) -> Self {
        let higher_is_better = matches!(
            metric,
            MetricKind::Psnr | MetricKind::Ssim | MetricKind::HistogramCorrelation
        );
        Self {
            metric,
            value,
            higher_is_better,
        }
    }

    /// Returns true if this score indicates "good" quality.
    /// Thresholds are metric-specific heuristics.
    #[allow(clippy::cast_precision_loss)]
    pub fn is_good(&self) -> bool {
        match self.metric {
            MetricKind::Psnr => self.value >= 30.0,
            MetricKind::Ssim => self.value >= 0.9,
            MetricKind::Mse => self.value <= 100.0,
            MetricKind::Mae => self.value <= 10.0,
            MetricKind::HistogramCorrelation => self.value >= 0.95,
        }
    }
}

impl fmt::Display for QualityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.4}", self.metric, self.value)
    }
}

/// The kind of quality metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// Peak Signal-to-Noise Ratio (dB).
    Psnr,
    /// Structural Similarity Index (simplified).
    Ssim,
    /// Mean Squared Error.
    Mse,
    /// Mean Absolute Error.
    Mae,
    /// Histogram correlation.
    HistogramCorrelation,
}

impl fmt::Display for MetricKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Psnr => write!(f, "PSNR"),
            Self::Ssim => write!(f, "SSIM"),
            Self::Mse => write!(f, "MSE"),
            Self::Mae => write!(f, "MAE"),
            Self::HistogramCorrelation => write!(f, "HistCorr"),
        }
    }
}

/// Aggregate report containing all computed metrics.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Source dimensions (width, height).
    pub src_dims: (u32, u32),
    /// Destination dimensions (width, height).
    pub dst_dims: (u32, u32),
    /// Individual scores.
    pub scores: Vec<QualityScore>,
}

impl QualityReport {
    /// Create a new empty report.
    pub fn new(src_dims: (u32, u32), dst_dims: (u32, u32)) -> Self {
        Self {
            src_dims,
            dst_dims,
            scores: Vec::new(),
        }
    }

    /// Add a score.
    pub fn add(&mut self, score: QualityScore) {
        self.scores.push(score);
    }

    /// Get a score by metric kind.
    pub fn get(&self, kind: MetricKind) -> Option<&QualityScore> {
        self.scores.iter().find(|s| s.metric == kind)
    }

    /// Check whether all scores indicate good quality.
    pub fn all_good(&self) -> bool {
        self.scores.iter().all(|s| s.is_good())
    }

    /// Format as a summary string.
    pub fn summary(&self) -> String {
        let parts: Vec<String> = self.scores.iter().map(|s| s.to_string()).collect();
        format!(
            "{}x{} -> {}x{}: {}",
            self.src_dims.0,
            self.src_dims.1,
            self.dst_dims.0,
            self.dst_dims.1,
            parts.join(", ")
        )
    }
}

// ---------------------------------------------------------------------------
// Metric computations
// ---------------------------------------------------------------------------

/// Compute Mean Squared Error between two equal-length u8 buffers.
#[allow(clippy::cast_precision_loss)]
pub fn mse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffers must have equal length");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    sum / a.len() as f64
}

/// Compute Mean Absolute Error.
#[allow(clippy::cast_precision_loss)]
pub fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffers must have equal length");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum();
    sum / a.len() as f64
}

/// Compute PSNR in dB. Returns `f64::INFINITY` if the images are identical.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let m = mse(a, b);
    if m < f64::EPSILON {
        return f64::INFINITY;
    }
    let max_val = 255.0_f64;
    10.0 * (max_val * max_val / m).log10()
}

/// Compute a simplified SSIM on two equal-length u8 buffers.
///
/// This is a global (non-windowed) approximation. For real SSIM you would
/// slide an 8x8 or 11x11 window, but this gives a useful approximation.
#[allow(clippy::cast_precision_loss)]
pub fn ssim_simple(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffers must have equal length");
    let n = a.len() as f64;
    if n < 1.0 {
        return 1.0;
    }

    let mean_a: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    let mut cov = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x as f64 - mean_a;
        let dy = y as f64 - mean_b;
        var_a += dx * dx;
        var_b += dy * dy;
        cov += dx * dy;
    }
    var_a /= n;
    var_b /= n;
    cov /= n;

    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2);
    let denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);
    numerator / denominator
}

/// Build a 256-bin histogram of a u8 buffer.
pub fn histogram(data: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &v in data {
        h[v as usize] += 1;
    }
    h
}

/// Histogram correlation (Pearson) between two histograms.
#[allow(clippy::cast_precision_loss)]
pub fn histogram_correlation(h1: &[u32; 256], h2: &[u32; 256]) -> f64 {
    let n = 256.0_f64;
    let mean1: f64 = h1.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean2: f64 = h2.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut num = 0.0_f64;
    let mut den1 = 0.0_f64;
    let mut den2 = 0.0_f64;
    for i in 0..256 {
        let d1 = h1[i] as f64 - mean1;
        let d2 = h2[i] as f64 - mean2;
        num += d1 * d2;
        den1 += d1 * d1;
        den2 += d2 * d2;
    }
    let denom = (den1 * den2).sqrt();
    if denom < f64::EPSILON {
        return 1.0;
    }
    num / denom
}

/// Run all metrics and return a report.
pub fn full_report(
    a: &[u8],
    b: &[u8],
    src_dims: (u32, u32),
    dst_dims: (u32, u32),
) -> QualityReport {
    let mut report = QualityReport::new(src_dims, dst_dims);
    report.add(QualityScore::new(MetricKind::Mse, mse(a, b)));
    report.add(QualityScore::new(MetricKind::Mae, mae(a, b)));
    report.add(QualityScore::new(MetricKind::Psnr, psnr(a, b)));
    report.add(QualityScore::new(MetricKind::Ssim, ssim_simple(a, b)));

    let h1 = histogram(a);
    let h2 = histogram(b);
    report.add(QualityScore::new(
        MetricKind::HistogramCorrelation,
        histogram_correlation(&h1, &h2),
    ));
    report
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_identical() {
        let a = vec![128u8; 100];
        assert!(mse(&a, &a) < f64::EPSILON);
    }

    #[test]
    fn test_mse_known_value() {
        let a = vec![0u8; 4];
        let b = vec![10u8; 4];
        assert!((mse(&a, &b) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mae_identical() {
        let a = vec![42u8; 100];
        assert!(mae(&a, &a) < f64::EPSILON);
    }

    #[test]
    fn test_mae_known_value() {
        let a = vec![0u8; 4];
        let b = vec![10u8; 4];
        assert!((mae(&a, &b) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_psnr_identical() {
        let a = vec![100u8; 64];
        assert_eq!(psnr(&a, &a), f64::INFINITY);
    }

    #[test]
    fn test_psnr_not_identical() {
        let a = vec![100u8; 64];
        let b = vec![110u8; 64];
        let p = psnr(&a, &b);
        assert!(p > 0.0 && p < 100.0);
    }

    #[test]
    fn test_ssim_identical() {
        let a = vec![128u8; 64];
        let s = ssim_simple(&a, &a);
        assert!((s - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ssim_different() {
        let a = vec![0u8; 64];
        let b = vec![255u8; 64];
        let s = ssim_simple(&a, &b);
        assert!(s < 0.5);
    }

    #[test]
    fn test_histogram_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let h = histogram(&data);
        assert!(h.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_histogram_correlation_identical() {
        let data = vec![128u8; 100];
        let h = histogram(&data);
        let c = histogram_correlation(&h, &h);
        assert!((c - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quality_score_display() {
        let s = QualityScore::new(MetricKind::Psnr, 35.5);
        assert_eq!(s.to_string(), "PSNR=35.5000");
    }

    #[test]
    fn test_quality_score_is_good() {
        assert!(QualityScore::new(MetricKind::Psnr, 40.0).is_good());
        assert!(!QualityScore::new(MetricKind::Psnr, 20.0).is_good());
        assert!(QualityScore::new(MetricKind::Mse, 50.0).is_good());
        assert!(!QualityScore::new(MetricKind::Mse, 200.0).is_good());
    }

    #[test]
    fn test_full_report_has_all_metrics() {
        let a = vec![128u8; 64];
        let b = vec![130u8; 64];
        let r = full_report(&a, &b, (8, 8), (8, 8));
        assert_eq!(r.scores.len(), 5);
        assert!(r.get(MetricKind::Psnr).is_some());
        assert!(r.get(MetricKind::Ssim).is_some());
        assert!(r.get(MetricKind::HistogramCorrelation).is_some());
    }

    #[test]
    fn test_report_all_good() {
        let a = vec![128u8; 64];
        let r = full_report(&a, &a, (8, 8), (8, 8));
        assert!(r.all_good());
    }

    #[test]
    fn test_report_summary() {
        let a = vec![128u8; 64];
        let r = full_report(&a, &a, (8, 8), (4, 4));
        let s = r.summary();
        assert!(s.contains("8x8"));
        assert!(s.contains("4x4"));
    }

    #[test]
    fn test_metric_kind_display() {
        assert_eq!(MetricKind::Psnr.to_string(), "PSNR");
        assert_eq!(MetricKind::Mae.to_string(), "MAE");
        assert_eq!(MetricKind::HistogramCorrelation.to_string(), "HistCorr");
    }

    #[test]
    fn test_empty_buffers() {
        let a: Vec<u8> = vec![];
        assert!(mse(&a, &a) < f64::EPSILON);
        assert!(mae(&a, &a) < f64::EPSILON);
    }
}
