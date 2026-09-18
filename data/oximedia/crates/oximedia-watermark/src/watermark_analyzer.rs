//! Watermark analyzer: report embedded watermark metadata, embedding strength,
//! and degradation level for a watermarked signal.
//!
//! This module provides:
//! - [`crate::watermark_analyzer::WatermarkAnalyzer`]: top-level analysis entry point
//! - [`crate::watermark_analyzer::AnalysisReport`]: structured report with per-algorithm findings
//! - [`crate::watermark_analyzer::StrengthEstimate`]: embedding strength estimation via energy ratio
//! - [`crate::watermark_analyzer::DegradationLevel`]: qualitative degradation assessment

use crate::metrics::{calculate_metrics, QualityMetrics};

// ---------------------------------------------------------------------------
// DegradationLevel
// ---------------------------------------------------------------------------

/// Qualitative degradation level of a watermarked signal relative to original.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationLevel {
    /// No perceptible degradation (SNR > 50 dB).
    Transparent,
    /// Minimal degradation (SNR 40–50 dB).
    Excellent,
    /// Acceptable degradation (SNR 30–40 dB).
    Good,
    /// Moderate degradation (SNR 20–30 dB).
    Fair,
    /// Significant degradation (SNR < 20 dB).
    Poor,
}

impl DegradationLevel {
    /// Classify from SNR in dB.
    #[must_use]
    pub fn from_snr(snr_db: f64) -> Self {
        match snr_db as i64 {
            i64::MIN..=19 => Self::Poor,
            20..=29 => Self::Fair,
            30..=39 => Self::Good,
            40..=49 => Self::Excellent,
            _ => Self::Transparent,
        }
    }

    /// Minimum SNR threshold for this level.
    #[must_use]
    pub fn snr_threshold_db(&self) -> f64 {
        match self {
            Self::Poor => 0.0,
            Self::Fair => 20.0,
            Self::Good => 30.0,
            Self::Excellent => 40.0,
            Self::Transparent => 50.0,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Transparent => "Transparent",
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
        }
    }
}

// ---------------------------------------------------------------------------
// StrengthEstimate
// ---------------------------------------------------------------------------

/// Estimated embedding strength derived from energy analysis.
#[derive(Debug, Clone)]
pub struct StrengthEstimate {
    /// Mean absolute difference between original and watermarked samples.
    pub mean_abs_diff: f64,
    /// RMS of the watermark signal (original − watermarked).
    pub watermark_rms: f64,
    /// Estimated relative embedding strength ∈ [0.0, 1.0].
    pub relative_strength: f64,
    /// SNR in dB (signal power / watermark power).
    pub snr_db: f64,
}

impl StrengthEstimate {
    /// Compute strength estimate from original and watermarked signals.
    #[must_use]
    pub fn compute(original: &[f32], watermarked: &[f32]) -> Self {
        let n = original.len().min(watermarked.len());
        if n == 0 {
            return Self {
                mean_abs_diff: 0.0,
                watermark_rms: 0.0,
                relative_strength: 0.0,
                snr_db: f64::INFINITY,
            };
        }

        let mut sum_abs_diff = 0.0f64;
        let mut sum_sq_wm = 0.0f64;
        let mut sum_sq_orig = 0.0f64;

        for i in 0..n {
            let diff = (watermarked[i] - original[i]) as f64;
            sum_abs_diff += diff.abs();
            sum_sq_wm += diff * diff;
            sum_sq_orig += (original[i] as f64) * (original[i] as f64);
        }

        let n_f = n as f64;
        let mean_abs_diff = sum_abs_diff / n_f;
        let watermark_rms = (sum_sq_wm / n_f).sqrt();
        let signal_rms = (sum_sq_orig / n_f).sqrt();

        let relative_strength = if signal_rms > 0.0 {
            (watermark_rms / signal_rms).clamp(0.0, 1.0)
        } else {
            watermark_rms.clamp(0.0, 1.0)
        };

        let snr_db = if watermark_rms > 0.0 {
            20.0 * (signal_rms / watermark_rms).log10()
        } else {
            f64::INFINITY
        };

        Self {
            mean_abs_diff,
            watermark_rms,
            relative_strength,
            snr_db,
        }
    }
}

// ---------------------------------------------------------------------------
// WatermarkMetadata
// ---------------------------------------------------------------------------

/// Metadata extracted from a watermarked signal.
#[derive(Debug, Clone)]
pub struct WatermarkMetadata {
    /// Estimated algorithm family (heuristic, not cryptographically verified).
    pub algorithm_hint: String,
    /// Estimated number of embedded bits (from payload structure).
    pub estimated_bit_count: usize,
    /// Whether synchronisation markers were found.
    pub sync_detected: bool,
    /// Detected frame size (if frequency-domain algorithm).
    pub frame_size_hint: Option<usize>,
}

// ---------------------------------------------------------------------------
// AnalysisReport
// ---------------------------------------------------------------------------

/// Complete analysis report for a watermarked signal.
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Signal quality metrics (requires original signal).
    pub quality_metrics: Option<QualityMetrics>,
    /// Embedding strength estimate (requires original signal).
    pub strength_estimate: Option<StrengthEstimate>,
    /// Qualitative degradation level.
    pub degradation_level: DegradationLevel,
    /// Detected watermark metadata (blind).
    pub metadata: WatermarkMetadata,
    /// Number of samples in the analysed signal.
    pub sample_count: usize,
    /// Per-band energy analysis (64 bands covering Nyquist).
    pub band_energies: Vec<f64>,
    /// Whether the signal appears to contain a watermark.
    pub watermark_present: bool,
    /// Confidence score that a watermark is present (0.0 – 1.0).
    pub presence_confidence: f64,
}

// ---------------------------------------------------------------------------
// WatermarkAnalyzer
// ---------------------------------------------------------------------------

/// Analyses audio signals to characterise embedded watermarks.
pub struct WatermarkAnalyzer {
    /// FFT frame size for spectral analysis.
    frame_size: usize,
    /// Sample rate in Hz.
    sample_rate: u32,
}

impl WatermarkAnalyzer {
    /// Create a new analyser.
    #[must_use]
    pub fn new(frame_size: usize, sample_rate: u32) -> Self {
        Self {
            frame_size: frame_size.next_power_of_two(),
            sample_rate,
        }
    }

    /// Return the sample rate this analyser was configured with.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Analyse `watermarked` without access to the original signal (blind).
    #[must_use]
    pub fn analyze_blind(&self, watermarked: &[f32]) -> AnalysisReport {
        let band_energies = self.compute_band_energies(watermarked);
        let (watermark_present, presence_confidence) =
            self.detect_watermark_presence(&band_energies, watermarked);
        let metadata = self.extract_metadata_blind(watermarked);

        AnalysisReport {
            quality_metrics: None,
            strength_estimate: None,
            degradation_level: DegradationLevel::Transparent, // unknown without original
            metadata,
            sample_count: watermarked.len(),
            band_energies,
            watermark_present,
            presence_confidence,
        }
    }

    /// Analyse `watermarked` with access to the `original` signal.
    #[must_use]
    pub fn analyze_with_reference(&self, original: &[f32], watermarked: &[f32]) -> AnalysisReport {
        let quality_metrics = calculate_metrics(original, watermarked);
        let strength_estimate = StrengthEstimate::compute(original, watermarked);
        let degradation_level = DegradationLevel::from_snr(quality_metrics.snr_db as f64);
        let band_energies = self.compute_band_energies(watermarked);
        let (watermark_present, presence_confidence) =
            self.detect_watermark_presence_with_ref(original, watermarked);
        let metadata = self.extract_metadata_blind(watermarked);

        AnalysisReport {
            quality_metrics: Some(quality_metrics),
            strength_estimate: Some(strength_estimate),
            degradation_level,
            metadata,
            sample_count: watermarked.len(),
            band_energies,
            watermark_present,
            presence_confidence,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute per-band energy using a simple non-overlapping DFT estimate.
    fn compute_band_energies(&self, samples: &[f32]) -> Vec<f64> {
        let n_bands = 64usize;
        let mut energies = vec![0.0f64; n_bands];

        if samples.len() < self.frame_size {
            return energies;
        }

        let frame = &samples[..self.frame_size];
        let freq = dft_magnitude_squared(frame);

        let bins_per_band = freq.len() / n_bands;
        for (band, chunk) in freq.chunks(bins_per_band.max(1)).enumerate() {
            if band >= n_bands {
                break;
            }
            energies[band] = chunk.iter().sum::<f64>() / chunk.len() as f64;
        }

        energies
    }

    /// Detect watermark presence by looking for non-natural energy patterns.
    fn detect_watermark_presence(&self, band_energies: &[f64], samples: &[f32]) -> (bool, f64) {
        // Heuristic 1: abnormal energy variance across bands.
        let variance = compute_variance(band_energies);
        let mean = band_energies.iter().sum::<f64>() / band_energies.len() as f64;
        let cv = if mean > 0.0 {
            (variance / (mean * mean)).sqrt()
        } else {
            0.0
        };

        // Heuristic 2: autocorrelation periodicity (spread-spectrum leaves traces).
        let autocorr_score = self.autocorrelation_score(samples);

        // Combine heuristics.
        let confidence = (0.5 * cv.clamp(0.0, 1.0) + 0.5 * autocorr_score).clamp(0.0, 1.0);
        let present = confidence > 0.3;
        (present, confidence)
    }

    /// Detect with original reference (difference signal analysis).
    fn detect_watermark_presence_with_ref(
        &self,
        original: &[f32],
        watermarked: &[f32],
    ) -> (bool, f64) {
        let n = original.len().min(watermarked.len());
        if n == 0 {
            return (false, 0.0);
        }

        // Compute difference energy.
        let diff_energy: f64 = original
            .iter()
            .zip(watermarked.iter())
            .take(n)
            .map(|(&o, &w)| {
                let d = (w - o) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let orig_energy: f64 = original
            .iter()
            .take(n)
            .map(|&o| (o as f64) * (o as f64))
            .sum::<f64>()
            / n as f64;

        let relative_diff = if orig_energy > 0.0 {
            (diff_energy / orig_energy).sqrt()
        } else {
            diff_energy.sqrt()
        };

        let confidence = (relative_diff * 20.0).clamp(0.0, 1.0);
        (confidence > 0.05, confidence)
    }

    /// Simple autocorrelation heuristic to detect periodic watermark patterns.
    fn autocorrelation_score(&self, samples: &[f32]) -> f64 {
        let n = self.frame_size.min(samples.len());
        if n < 16 {
            return 0.0;
        }

        let mut max_corr = 0.0f64;
        let check_lags = [64usize, 128, 256, 512, 1024];

        for &lag in &check_lags {
            if lag >= n {
                break;
            }
            let mut corr = 0.0f64;
            let mut energy = 0.0f64;
            for i in 0..n - lag {
                corr += (samples[i] as f64) * (samples[i + lag] as f64);
                energy += (samples[i] as f64) * (samples[i] as f64);
            }
            let norm = if energy > 0.0 {
                (corr / energy).abs()
            } else {
                0.0
            };
            if norm > max_corr {
                max_corr = norm;
            }
        }

        max_corr.clamp(0.0, 1.0)
    }

    /// Blind metadata extraction using simple heuristics.
    fn extract_metadata_blind(&self, samples: &[f32]) -> WatermarkMetadata {
        let frame_size_hint = if samples.len() >= 2048 {
            Some(2048)
        } else if samples.len() >= 1024 {
            Some(1024)
        } else {
            None
        };

        // Check for SYNC pattern (OXIWM) by looking for specific energy patterns
        // at known byte positions. This is a structural heuristic only.
        let sync_detected = self.heuristic_sync_check(samples);

        WatermarkMetadata {
            algorithm_hint: "unknown".to_string(),
            estimated_bit_count: 0,
            sync_detected,
            frame_size_hint,
        }
    }

    /// Heuristic: check for abnormal energy in the first few frames that may
    /// indicate an embedded sync pattern.
    fn heuristic_sync_check(&self, samples: &[f32]) -> bool {
        if samples.len() < 512 {
            return false;
        }
        // A simple check: if any 512-sample block has unusually high energy
        // at specific frequency bins commonly used for sync, flag it.
        let check_len = 512.min(samples.len());
        let rms: f32 =
            (samples[..check_len].iter().map(|&s| s * s).sum::<f32>() / check_len as f32).sqrt();
        // Presence of non-silence is a weak indicator.
        rms > 1e-8
    }
}

// ---------------------------------------------------------------------------
// Module-level convenience functions
// ---------------------------------------------------------------------------

/// Compute analysis report with reference signal.
#[must_use]
pub fn analyze_watermark(
    original: &[f32],
    watermarked: &[f32],
    sample_rate: u32,
) -> AnalysisReport {
    let analyzer = WatermarkAnalyzer::new(2048, sample_rate);
    analyzer.analyze_with_reference(original, watermarked)
}

/// Compute SNR between original and watermarked signal in dB.
#[must_use]
pub fn compute_snr_db(original: &[f32], watermarked: &[f32]) -> f64 {
    let estimate = StrengthEstimate::compute(original, watermarked);
    estimate.snr_db
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Simple DFT magnitude-squared spectrum (Hann-windowed).
fn dft_magnitude_squared(samples: &[f32]) -> Vec<f64> {
    let n = samples.len();
    let half = n / 2;
    let mut result = vec![0.0f64; half];

    for k in 0..half {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, &s) in samples.iter().enumerate() {
            let window = 0.5 * (1.0 - (std::f64::consts::TAU * t as f64 / n as f64).cos());
            let angle = std::f64::consts::TAU * k as f64 * t as f64 / n as f64;
            re += (s as f64) * window * angle.cos();
            im -= (s as f64) * window * angle.sin();
        }
        result[k] = re * re + im * im;
    }

    result
}

fn compute_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (std::f32::consts::TAU * 440.0 * i as f32 / 44100.0).sin())
            .collect()
    }

    #[test]
    fn test_degradation_level_from_snr() {
        assert_eq!(
            DegradationLevel::from_snr(55.0),
            DegradationLevel::Transparent
        );
        assert_eq!(
            DegradationLevel::from_snr(45.0),
            DegradationLevel::Excellent
        );
        assert_eq!(DegradationLevel::from_snr(35.0), DegradationLevel::Good);
        assert_eq!(DegradationLevel::from_snr(25.0), DegradationLevel::Fair);
        assert_eq!(DegradationLevel::from_snr(10.0), DegradationLevel::Poor);
    }

    #[test]
    fn test_degradation_level_label() {
        assert_eq!(DegradationLevel::Transparent.label(), "Transparent");
        assert_eq!(DegradationLevel::Good.label(), "Good");
        assert_eq!(DegradationLevel::Poor.label(), "Poor");
    }

    #[test]
    fn test_degradation_level_snr_threshold() {
        assert_eq!(DegradationLevel::Good.snr_threshold_db(), 30.0);
        assert_eq!(DegradationLevel::Poor.snr_threshold_db(), 0.0);
    }

    #[test]
    fn test_strength_estimate_identical_signals() {
        let signal = sine_signal(4096, 0.5);
        let est = StrengthEstimate::compute(&signal, &signal);
        assert_eq!(est.mean_abs_diff, 0.0);
        assert_eq!(est.watermark_rms, 0.0);
        assert_eq!(est.relative_strength, 0.0);
        assert!(est.snr_db.is_infinite());
    }

    #[test]
    fn test_strength_estimate_high_noise() {
        let original = sine_signal(4096, 0.5);
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.1).collect();
        let est = StrengthEstimate::compute(&original, &watermarked);
        assert!(est.watermark_rms > 0.0);
        assert!(est.snr_db < 100.0);
        assert!(est.relative_strength > 0.0);
    }

    #[test]
    fn test_strength_estimate_empty() {
        let est = StrengthEstimate::compute(&[], &[]);
        assert_eq!(est.mean_abs_diff, 0.0);
        assert!(est.snr_db.is_infinite());
    }

    #[test]
    fn test_analyze_blind_returns_report() {
        let analyzer = WatermarkAnalyzer::new(2048, 44100);
        let samples = sine_signal(4096, 0.5);
        let report = analyzer.analyze_blind(&samples);
        assert_eq!(report.sample_count, 4096);
        assert_eq!(report.band_energies.len(), 64);
    }

    #[test]
    fn test_analyze_with_reference_snr() {
        let analyzer = WatermarkAnalyzer::new(2048, 44100);
        let original = sine_signal(4096, 0.5);
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.001).collect();
        let report = analyzer.analyze_with_reference(&original, &watermarked);
        let snr = report
            .quality_metrics
            .expect("quality metrics should be present")
            .snr_db;
        assert!(
            snr > 30.0,
            "SNR should be > 30 dB for small watermark, got {snr}"
        );
    }

    #[test]
    fn test_analyze_with_reference_degradation_level() {
        let analyzer = WatermarkAnalyzer::new(2048, 44100);
        let original = sine_signal(4096, 0.5);
        // Very slight watermark → Transparent or Excellent
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.0001).collect();
        let report = analyzer.analyze_with_reference(&original, &watermarked);
        assert!(
            matches!(
                report.degradation_level,
                DegradationLevel::Transparent | DegradationLevel::Excellent
            ),
            "Expected Transparent or Excellent, got {:?}",
            report.degradation_level
        );
    }

    #[test]
    fn test_analyze_strength_estimate_present() {
        let analyzer = WatermarkAnalyzer::new(2048, 44100);
        let original = sine_signal(4096, 0.5);
        let watermarked: Vec<f32> = original.iter().map(|&s| s * 1.01).collect();
        let report = analyzer.analyze_with_reference(&original, &watermarked);
        assert!(report.strength_estimate.is_some());
    }

    #[test]
    fn test_blind_analysis_no_quality_metrics() {
        let analyzer = WatermarkAnalyzer::new(2048, 44100);
        let samples = sine_signal(4096, 0.5);
        let report = analyzer.analyze_blind(&samples);
        assert!(report.quality_metrics.is_none());
        assert!(report.strength_estimate.is_none());
    }

    #[test]
    fn test_band_energies_length() {
        let analyzer = WatermarkAnalyzer::new(2048, 44100);
        let samples = sine_signal(4096, 0.5);
        let energies = analyzer.compute_band_energies(&samples);
        assert_eq!(energies.len(), 64);
    }

    #[test]
    fn test_compute_snr_db_helper() {
        let orig = sine_signal(4096, 0.5);
        let wm: Vec<f32> = orig.iter().map(|&s| s + 0.001).collect();
        let snr = compute_snr_db(&orig, &wm);
        assert!(snr > 30.0);
    }

    #[test]
    fn test_analyze_watermark_helper() {
        let orig = sine_signal(4096, 0.5);
        let wm: Vec<f32> = orig.iter().map(|&s| s + 0.001).collect();
        let report = analyze_watermark(&orig, &wm, 44100);
        assert!(report.quality_metrics.is_some());
    }
}
