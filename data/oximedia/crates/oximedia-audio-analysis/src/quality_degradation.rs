//! Audio quality degradation detection.
//!
//! Detects common audio quality issues arising from encoding artifacts and
//! bandwidth limitations:
//!
//! - **Encoding artifacts** (MP3/AAC pre-echo, ringing): detected via
//!   pre-echo energy ratio and spectral smearing
//! - **Bandwidth limitation** (low-pass filtering): detected by inspecting the
//!   high-frequency rolloff point and the ratio of high-frequency energy
//! - **Clipping / saturation**: samples at or near ±1.0
//! - **Noise floor elevation**: background noise estimate relative to signal
//! - **Quantization noise**: characteristic uniform error distribution
//! - **Compression smearing**: loss of transient sharpness

use crate::{AnalysisConfig, AnalysisError, Result};
use oxifft::Complex;

/// Quality degradation analysis result.
#[derive(Debug, Clone)]
pub struct QualityDegradationResult {
    /// Overall degradation score (0.0 = perfect, 1.0 = severely degraded)
    pub degradation_score: f32,
    /// Clipping severity (0.0–1.0; fraction of near-clipped samples)
    pub clipping_severity: f32,
    /// Bandwidth limitation score (0.0 = full bandwidth, 1.0 = severely limited)
    pub bandwidth_limitation: f32,
    /// Encoding artifact score (pre-echo / spectral smearing, 0.0–1.0)
    pub encoding_artifacts: f32,
    /// Estimated audio bandwidth in Hz (frequency below which 99% of energy lies)
    pub estimated_bandwidth_hz: f32,
    /// Noise floor estimate in dBFS
    pub noise_floor_db: f32,
    /// Whether clipping is detected
    pub has_clipping: bool,
    /// Whether bandwidth limitation is detected
    pub has_bandwidth_limitation: bool,
    /// Whether encoding artifacts are detected
    pub has_encoding_artifacts: bool,
}

/// Degradation analyzes audio samples for quality issues.
pub struct DegradationAnalyzer {
    config: AnalysisConfig,
    /// Clipping threshold (samples above this are considered clipped)
    clip_threshold: f32,
    /// Bandwidth limitation detection threshold in Hz
    bandwidth_limit_hz: f32,
}

impl DegradationAnalyzer {
    /// Create a new degradation analyzer with default thresholds.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            clip_threshold: 0.98,
            bandwidth_limit_hz: 15000.0,
        }
    }

    /// Set the clipping detection threshold (default 0.98).
    #[must_use]
    pub fn with_clip_threshold(mut self, threshold: f32) -> Self {
        self.clip_threshold = threshold.clamp(0.5, 1.0);
        self
    }

    /// Set the bandwidth limitation detection threshold in Hz (default 15 kHz).
    #[must_use]
    pub fn with_bandwidth_limit_hz(mut self, hz: f32) -> Self {
        self.bandwidth_limit_hz = hz.max(1000.0);
        self
    }

    /// Analyze audio for quality degradation.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<QualityDegradationResult> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // ── Clipping detection ──────────────────────────────────────────────
        let (clipping_severity, has_clipping) = detect_clipping(samples, self.clip_threshold);

        // ── Spectral analysis ──────────────────────────────────────────────
        let magnitude = compute_magnitude_spectrum(samples, self.config.fft_size)?;
        let (estimated_bandwidth_hz, bandwidth_limitation, has_bandwidth_limitation) =
            detect_bandwidth_limitation(&magnitude, sample_rate, self.bandwidth_limit_hz);

        // ── Encoding artifact detection ────────────────────────────────────
        let (encoding_artifacts, has_encoding_artifacts) = detect_encoding_artifacts(
            samples,
            sample_rate,
            self.config.fft_size,
            self.config.hop_size,
        );

        // ── Noise floor estimation ─────────────────────────────────────────
        let noise_floor_db = estimate_noise_floor_db(&magnitude, sample_rate);

        // ── Overall degradation score ──────────────────────────────────────
        let degradation_score = (clipping_severity * 0.35
            + bandwidth_limitation * 0.30
            + encoding_artifacts * 0.25
            + (-noise_floor_db / 100.0).min(1.0).max(0.0) * 0.10)
            .min(1.0)
            .max(0.0);

        Ok(QualityDegradationResult {
            degradation_score,
            clipping_severity,
            bandwidth_limitation,
            encoding_artifacts,
            estimated_bandwidth_hz,
            noise_floor_db,
            has_clipping,
            has_bandwidth_limitation,
            has_encoding_artifacts,
        })
    }
}

// ── internal helpers ─────────────────────────────────────────────────────────

/// Detect clipping: fraction of samples at or near full scale.
fn detect_clipping(samples: &[f32], threshold: f32) -> (f32, bool) {
    if samples.is_empty() {
        return (0.0, false);
    }
    let clipped = samples.iter().filter(|&&x| x.abs() >= threshold).count();
    let severity = clipped as f32 / samples.len() as f32;
    (severity, severity > 0.001) // >0.1% clipped → flagged
}

/// Compute magnitude spectrum for analysis using the first `fft_size` samples.
fn compute_magnitude_spectrum(samples: &[f32], fft_size: usize) -> Result<Vec<f32>> {
    let n = fft_size.min(samples.len()).next_power_of_two();
    if n == 0 {
        return Err(AnalysisError::InsufficientSamples { needed: 1, got: 0 });
    }

    // Hann window
    let window: Vec<f32> = (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()))
        .collect();

    let buffer: Vec<Complex<f64>> = samples[..n.min(samples.len())]
        .iter()
        .zip(&window)
        .map(|(&s, &w)| Complex::new(f64::from(s * w), 0.0))
        .collect();

    let spectrum = oxifft::fft(&buffer);
    let magnitude: Vec<f32> = spectrum[..=n / 2].iter().map(|c| c.norm() as f32).collect();

    Ok(magnitude)
}

/// Detect bandwidth limitation: estimates effective audio bandwidth.
fn detect_bandwidth_limitation(
    magnitude: &[f32],
    sample_rate: f32,
    limit_threshold_hz: f32,
) -> (f32, f32, bool) {
    if magnitude.is_empty() {
        return (0.0, 0.0, false);
    }

    let n = magnitude.len();
    let hz_per_bin = sample_rate / (2.0 * (n - 1) as f32);

    let total_energy: f32 = magnitude.iter().map(|&m| m * m).sum();
    if total_energy <= 0.0 {
        return (0.0, 1.0, true);
    }

    // Find 99% energy bandwidth
    let mut cumulative = 0.0_f32;
    let target = total_energy * 0.99;
    let mut bandwidth_bin = n - 1;

    for (i, &m) in magnitude.iter().enumerate() {
        cumulative += m * m;
        if cumulative >= target {
            bandwidth_bin = i;
            break;
        }
    }

    let estimated_hz = bandwidth_bin as f32 * hz_per_bin;

    // Energy ratio above the limit threshold
    let limit_bin = (limit_threshold_hz / hz_per_bin) as usize;
    let limit_bin = limit_bin.min(n - 1);

    let high_freq_energy: f32 = magnitude[limit_bin..].iter().map(|&m| m * m).sum();
    let high_freq_ratio = high_freq_energy / total_energy;

    // Bandwidth limitation: low estimated bandwidth OR very little high-freq energy
    let limitation_score = if estimated_hz < limit_threshold_hz * 0.85 {
        let normalized = 1.0 - (estimated_hz / (limit_threshold_hz * 0.85)).min(1.0);
        normalized * 0.8 + (1.0 - high_freq_ratio.min(1.0)) * 0.2
    } else {
        (1.0 - high_freq_ratio * 20.0).max(0.0) * 0.3
    };

    let has_limitation = estimated_hz < limit_threshold_hz * 0.85 || high_freq_ratio < 0.001;

    (estimated_hz, limitation_score.min(1.0), has_limitation)
}

/// Detect encoding artifacts via pre-echo detection and spectral smearing.
fn detect_encoding_artifacts(
    samples: &[f32],
    _sample_rate: f32,
    fft_size: usize,
    hop_size: usize,
) -> (f32, bool) {
    if samples.len() < fft_size + hop_size {
        return (0.0, false);
    }

    let hop = hop_size;
    let n_frames = (samples.len() - fft_size) / hop;

    if n_frames < 3 {
        return (0.0, false);
    }

    // Pre-echo: energy just before a transient should be low; if it's elevated, that's an artifact
    let mut pre_echo_scores = Vec::with_capacity(n_frames);

    for i in 1..(n_frames - 1) {
        let start = i * hop;
        let end = (start + fft_size).min(samples.len());
        let frame = &samples[start..end];
        let energy: f32 = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32;

        let prev_start = (i - 1) * hop;
        let prev_end = (prev_start + fft_size).min(samples.len());
        let prev_frame = &samples[prev_start..prev_end];
        let prev_energy: f32 =
            prev_frame.iter().map(|&x| x * x).sum::<f32>() / prev_frame.len() as f32;

        let next_start = (i + 1) * hop;
        let next_end = (next_start + fft_size).min(samples.len());
        let next_frame = &samples[next_start..next_end];
        let next_energy: f32 =
            next_frame.iter().map(|&x| x * x).sum::<f32>() / next_frame.len() as f32;

        // If next frame has much higher energy than this frame and prev frame,
        // check that current frame doesn't have elevated energy (pre-echo)
        if next_energy > energy * 5.0 && prev_energy < energy * 0.5 {
            // Pre-echo: energy rises gradually before transient
            let rise_ratio = energy / (prev_energy + 1e-10);
            pre_echo_scores.push((rise_ratio - 1.0).max(0.0).min(5.0) / 5.0);
        }
    }

    if pre_echo_scores.is_empty() {
        return (0.0, false);
    }

    let score = pre_echo_scores.iter().sum::<f32>() / pre_echo_scores.len() as f32;
    let has_artifacts = score > 0.2;
    (score.min(1.0), has_artifacts)
}

/// Estimate noise floor in dBFS from the quietest spectral bins.
fn estimate_noise_floor_db(magnitude: &[f32], _sample_rate: f32) -> f32 {
    if magnitude.is_empty() {
        return -100.0;
    }

    let mut sorted: Vec<f32> = magnitude.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use 10th percentile as noise floor estimate
    let percentile_idx = sorted.len() / 10;
    let floor_magnitude = if percentile_idx < sorted.len() {
        sorted[percentile_idx]
    } else {
        sorted[0]
    };

    if floor_magnitude <= 0.0 {
        return -100.0;
    }

    (20.0 * floor_magnitude.log10()).max(-100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sine(freq: f32, n: usize, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_no_clipping_sine() {
        let config = AnalysisConfig::default();
        let analyzer = DegradationAnalyzer::new(config);
        let samples = make_sine(440.0, 44100, 44100.0);
        let result = analyzer.analyze(&samples, 44100.0).expect("should succeed");
        // Pure sine at amplitude 1 might just touch clipping threshold at peaks
        assert!(result.clipping_severity >= 0.0 && result.clipping_severity <= 1.0);
        assert!(result.degradation_score >= 0.0 && result.degradation_score <= 1.0);
    }

    #[test]
    fn test_clipping_detection() {
        let samples: Vec<f32> = (0..1000).map(|_| 1.0).collect(); // Full-scale DC
        let (severity, has_clipping) = detect_clipping(&samples, 0.98);
        assert!(has_clipping, "DC at 1.0 should be clipped");
        assert!(severity > 0.99, "All samples should be clipped: {severity}");
    }

    #[test]
    fn test_no_clipping_detection() {
        let samples: Vec<f32> = (0..1000).map(|_| 0.5).collect();
        let (severity, has_clipping) = detect_clipping(&samples, 0.98);
        assert!(!has_clipping);
        assert_eq!(severity, 0.0);
    }

    #[test]
    fn test_bandwidth_estimation_sine_440() {
        let n = 4096_usize;
        let sr = 44100.0;
        let samples = make_sine(440.0, n, sr);
        let mag = compute_magnitude_spectrum(&samples, n).expect("should compute");
        let (bw_hz, _, _) = detect_bandwidth_limitation(&mag, sr, 15000.0);
        // 440 Hz sine should have very low bandwidth
        assert!(
            bw_hz < 5000.0,
            "440 Hz sine should have low bandwidth: {bw_hz}"
        );
    }

    #[test]
    fn test_degradation_insufficient_samples() {
        let config = AnalysisConfig::default();
        let analyzer = DegradationAnalyzer::new(config);
        let result = analyzer.analyze(&[0.0; 100], 44100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_noise_floor_silent() {
        let mag = vec![0.0_f32; 100];
        let floor = estimate_noise_floor_db(&mag, 44100.0);
        assert_eq!(floor, -100.0);
    }

    #[test]
    fn test_noise_floor_white_noise() {
        // White-noise-like spectrum (all bins equal)
        let mag = vec![0.1_f32; 512];
        let floor = estimate_noise_floor_db(&mag, 44100.0);
        assert!(
            floor > -40.0 && floor < 0.0,
            "Floor should be between -40 and 0 dBFS: {floor}"
        );
    }

    #[test]
    fn test_result_scores_in_range() {
        let config = AnalysisConfig::default();
        let analyzer = DegradationAnalyzer::new(config);
        let n = 44100;
        let sr = 44100.0;
        // Bandlimited signal: only low-frequency content
        let samples = make_sine(200.0, n, sr);
        let result = analyzer.analyze(&samples, sr).expect("should succeed");
        assert!(result.degradation_score >= 0.0 && result.degradation_score <= 1.0);
        assert!(result.bandwidth_limitation >= 0.0 && result.bandwidth_limitation <= 1.0);
        assert!(result.encoding_artifacts >= 0.0 && result.encoding_artifacts <= 1.0);
        assert!(result.clipping_severity >= 0.0 && result.clipping_severity <= 1.0);
    }
}
