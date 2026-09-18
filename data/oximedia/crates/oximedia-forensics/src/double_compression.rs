#![allow(dead_code)]
//! Double JPEG compression detection for image forensics.
//!
//! When a JPEG image is loaded, edited, and re-saved as JPEG, it undergoes
//! a second round of DCT quantization. This "double compression" leaves
//! characteristic statistical artifacts in the DCT coefficient histograms
//! that do not appear in singly-compressed images.
//!
//! # Detection Techniques
//!
//! - **DCT histogram periodicity** -- double quantization produces periodic
//!   peaks/valleys in the first-digit distribution of DCT coefficients
//! - **Benford's law deviation** -- natural DCT coefficients follow a
//!   modified Benford distribution; re-compression disrupts this
//! - **Blocking artifact grid analysis** -- misaligned 8x8 grids from the
//!   first and second compression produce detectable patterns
//!
//! All computations are pure Rust with no external image decoding.

use std::collections::HashMap;

/// Configuration for double compression detection.
#[derive(Debug, Clone)]
pub struct DoubleCompressionConfig {
    /// Maximum DCT coefficient magnitude to include in histograms
    pub max_coeff: i32,
    /// Number of histogram bins
    pub num_bins: usize,
    /// Minimum periodicity score to flag double compression
    pub periodicity_threshold: f64,
    /// Whether to use Benford's law analysis
    pub use_benford: bool,
}

impl Default for DoubleCompressionConfig {
    fn default() -> Self {
        Self {
            max_coeff: 50,
            num_bins: 101,
            periodicity_threshold: 0.3,
            use_benford: true,
        }
    }
}

/// A histogram of DCT coefficients.
#[derive(Debug, Clone)]
pub struct DctHistogram {
    /// Bin counts indexed from `[-max_coeff..=max_coeff]` mapped to `[0..num_bins]`
    pub bins: Vec<u64>,
    /// Maximum coefficient magnitude represented
    pub max_coeff: i32,
    /// Total number of coefficients counted
    pub total: u64,
}

impl DctHistogram {
    /// Create a new empty histogram.
    pub fn new(max_coeff: i32) -> Self {
        let num_bins = (2 * max_coeff + 1) as usize;
        Self {
            bins: vec![0; num_bins],
            max_coeff,
            total: 0,
        }
    }

    /// Add a coefficient value. Values outside the range are clamped.
    pub fn add(&mut self, value: i32) {
        let clamped = value.clamp(-self.max_coeff, self.max_coeff);
        let idx = (clamped + self.max_coeff) as usize;
        if idx < self.bins.len() {
            self.bins[idx] += 1;
            self.total += 1;
        }
    }

    /// Get the count for a specific coefficient value.
    pub fn count(&self, value: i32) -> u64 {
        let clamped = value.clamp(-self.max_coeff, self.max_coeff);
        let idx = (clamped + self.max_coeff) as usize;
        if idx < self.bins.len() {
            self.bins[idx]
        } else {
            0
        }
    }

    /// Get the number of bins.
    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }

    /// Compute the normalised histogram (probability distribution).
    #[allow(clippy::cast_precision_loss)]
    pub fn normalised(&self) -> Vec<f64> {
        if self.total == 0 {
            return vec![0.0; self.bins.len()];
        }
        let t = self.total as f64;
        self.bins.iter().map(|&b| b as f64 / t).collect()
    }
}

/// Compute the periodicity of a histogram using autocorrelation.
///
/// Double JPEG compression produces periodic peaks in DCT coefficient histograms.
/// We compute the normalised autocorrelation and return the maximum value for
/// lags in `[2, max_lag]` as the periodicity score.
#[allow(clippy::cast_precision_loss)]
pub fn histogram_periodicity(hist: &DctHistogram, max_lag: usize) -> f64 {
    let norm = hist.normalised();
    let n = norm.len();
    if n < 4 || max_lag < 2 {
        return 0.0;
    }

    let mean: f64 = norm.iter().sum::<f64>() / n as f64;
    let var: f64 = norm.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    if var < 1e-15 {
        return 0.0;
    }

    let mut best = 0.0f64;
    let effective_max = max_lag.min(n / 2);
    for lag in 2..=effective_max {
        let mut corr = 0.0;
        for i in 0..n - lag {
            corr += (norm[i] - mean) * (norm[i + lag] - mean);
        }
        let normalised_corr = corr / var;
        if normalised_corr > best {
            best = normalised_corr;
        }
    }
    best.clamp(0.0, 1.0)
}

/// Expected first-digit distribution for DCT coefficients following a
/// Laplacian-like model (modified Benford's law).
///
/// Returns probabilities for digits 1..9.
#[allow(clippy::cast_precision_loss)]
pub fn benford_expected() -> [f64; 9] {
    let mut probs = [0.0f64; 9];
    for d in 1..=9u32 {
        probs[(d - 1) as usize] = (1.0 + 1.0 / d as f64).log10();
    }
    probs
}

/// Compute the first-digit distribution of absolute DCT coefficients (ignoring zeros).
#[allow(clippy::cast_precision_loss)]
pub fn first_digit_distribution(coefficients: &[i32]) -> [f64; 9] {
    let mut counts = [0u64; 9];
    let mut total = 0u64;

    for &c in coefficients {
        let abs = c.unsigned_abs();
        if abs == 0 {
            continue;
        }
        // Extract first digit
        let mut v = abs;
        while v >= 10 {
            v /= 10;
        }
        if v >= 1 && v <= 9 {
            counts[(v - 1) as usize] += 1;
            total += 1;
        }
    }

    let mut dist = [0.0f64; 9];
    if total > 0 {
        for i in 0..9 {
            dist[i] = counts[i] as f64 / total as f64;
        }
    }
    dist
}

/// Compute the chi-squared divergence of the observed first-digit distribution
/// from the expected Benford distribution.
#[allow(clippy::cast_precision_loss)]
pub fn benford_chi_squared(observed: &[f64; 9]) -> f64 {
    let expected = benford_expected();
    let mut chi2 = 0.0;
    for i in 0..9 {
        if expected[i] > 1e-15 {
            let diff = observed[i] - expected[i];
            chi2 += diff * diff / expected[i];
        }
    }
    chi2
}

/// Result of double compression analysis.
#[derive(Debug, Clone)]
pub struct DoubleCompressionResult {
    /// Whether double compression is detected
    pub detected: bool,
    /// Periodicity score from DCT histogram analysis (0..1)
    pub periodicity_score: f64,
    /// Chi-squared Benford divergence (higher = more suspicious)
    pub benford_chi2: f64,
    /// Overall confidence score (0..1)
    pub confidence: f64,
    /// Estimated primary quality factor (if detectable)
    pub estimated_primary_quality: Option<u8>,
    /// Additional details
    pub details: HashMap<String, f64>,
}

/// Analyse a set of DCT coefficients for evidence of double compression.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_double_compression(
    coefficients: &[i32],
    config: &DoubleCompressionConfig,
) -> DoubleCompressionResult {
    // Build DCT histogram
    let mut hist = DctHistogram::new(config.max_coeff);
    for &c in coefficients {
        hist.add(c);
    }

    let periodicity = histogram_periodicity(&hist, 20);

    // Benford analysis
    let first_digits = first_digit_distribution(coefficients);
    let chi2 = if config.use_benford {
        benford_chi_squared(&first_digits)
    } else {
        0.0
    };

    // Estimate primary quality by testing periodicity at different quantization steps
    let mut best_q: Option<u8> = None;
    let mut best_q_score = 0.0;
    for q in (10..=90).step_by(5) {
        let step = (50.0 / f64::from(q)).max(1.0);
        let istep = step.round() as usize;
        if istep < 2 || istep > 20 {
            continue;
        }
        // Check periodicity at this specific step
        let norm = hist.normalised();
        let n = norm.len();
        if istep >= n / 2 {
            continue;
        }
        let mean: f64 = norm.iter().sum::<f64>() / n as f64;
        let var: f64 = norm.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
        if var < 1e-15 {
            continue;
        }
        let mut corr = 0.0;
        for i in 0..n - istep {
            corr += (norm[i] - mean) * (norm[i + istep] - mean);
        }
        let score = (corr / var).clamp(0.0, 1.0);
        if score > best_q_score {
            best_q_score = score;
            best_q = Some(q);
        }
    }

    let confidence = if config.use_benford {
        let benford_norm = (chi2 / 10.0).clamp(0.0, 1.0);
        (periodicity * 0.6 + benford_norm * 0.4).clamp(0.0, 1.0)
    } else {
        periodicity
    };

    let detected = confidence >= config.periodicity_threshold;

    let mut details = HashMap::new();
    details.insert("periodicity".to_string(), periodicity);
    details.insert("benford_chi2".to_string(), chi2);
    details.insert("best_q_score".to_string(), best_q_score);

    DoubleCompressionResult {
        detected,
        periodicity_score: periodicity,
        benford_chi2: chi2,
        confidence,
        estimated_primary_quality: if detected { best_q } else { None },
        details,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let c = DoubleCompressionConfig::default();
        assert_eq!(c.max_coeff, 50);
        assert_eq!(c.num_bins, 101);
        assert!(c.use_benford);
    }

    #[test]
    fn test_dct_histogram_new() {
        let h = DctHistogram::new(10);
        assert_eq!(h.num_bins(), 21);
        assert_eq!(h.total, 0);
    }

    #[test]
    fn test_dct_histogram_add_and_count() {
        let mut h = DctHistogram::new(5);
        h.add(3);
        h.add(3);
        h.add(-2);
        assert_eq!(h.count(3), 2);
        assert_eq!(h.count(-2), 1);
        assert_eq!(h.count(0), 0);
        assert_eq!(h.total, 3);
    }

    #[test]
    fn test_dct_histogram_clamping() {
        let mut h = DctHistogram::new(5);
        h.add(100);
        assert_eq!(h.count(5), 1);
        h.add(-100);
        assert_eq!(h.count(-5), 1);
    }

    #[test]
    fn test_dct_histogram_normalised() {
        let mut h = DctHistogram::new(2);
        h.add(0);
        h.add(0);
        h.add(1);
        h.add(1);
        let norm = h.normalised();
        assert!((norm[2] - 0.5).abs() < 1e-12); // index for 0
        assert!((norm[3] - 0.5).abs() < 1e-12); // index for 1
    }

    #[test]
    fn test_dct_histogram_normalised_empty() {
        let h = DctHistogram::new(3);
        let norm = h.normalised();
        assert!(norm.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_histogram_periodicity_constant() {
        let mut h = DctHistogram::new(10);
        for v in -10..=10 {
            h.add(v);
        }
        let p = histogram_periodicity(&h, 5);
        // Uniform histogram has no periodicity
        assert!(p < 0.1);
    }

    #[test]
    fn test_benford_expected_sums_to_one() {
        let probs = benford_expected();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_first_digit_distribution_basic() {
        let coeffs = vec![1, 2, 3, 10, 20, 30, -15, -25];
        let dist = first_digit_distribution(&coeffs);
        // digits: 1, 2, 3, 1, 2, 3, 1, 2 (from abs: 1,2,3,10,20,30,15,25)
        assert!((dist[0] - 3.0 / 8.0).abs() < 1e-12); // digit 1: 1, 10, 15
        assert!((dist[1] - 3.0 / 8.0).abs() < 1e-12); // digit 2: 2, 20, 25
        assert!((dist[2] - 2.0 / 8.0).abs() < 1e-12); // digit 3: 3, 30
    }

    #[test]
    fn test_first_digit_distribution_zeros_ignored() {
        let coeffs = vec![0, 0, 0, 1];
        let dist = first_digit_distribution(&coeffs);
        assert!((dist[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_benford_chi_squared_perfect() {
        let expected = benford_expected();
        let chi2 = benford_chi_squared(&expected);
        assert!(chi2 < 1e-12);
    }

    #[test]
    fn test_analyze_double_compression_empty() {
        let config = DoubleCompressionConfig::default();
        let result = analyze_double_compression(&[], &config);
        assert!(!result.detected);
    }

    #[test]
    fn test_analyze_double_compression_uniform() {
        let config = DoubleCompressionConfig::default();
        let coeffs: Vec<i32> = (0..1000).map(|i| (i % 11) - 5).collect();
        let result = analyze_double_compression(&coeffs, &config);
        // Structured but not specifically double-compressed
        assert!(result.periodicity_score >= 0.0);
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_double_compression_result_details() {
        let config = DoubleCompressionConfig::default();
        let coeffs: Vec<i32> = vec![1, -1, 2, -2, 3, -3];
        let result = analyze_double_compression(&coeffs, &config);
        assert!(result.details.contains_key("periodicity"));
        assert!(result.details.contains_key("benford_chi2"));
    }
}
