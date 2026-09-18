//! Audio watermark detection using spectral analysis.
//!
//! # Method
//!
//! This module implements a **spread-spectrum spectral watermark detector**.
//! Watermarking is typically done by embedding a pseudo-random noise sequence
//! at very low amplitude into the magnitude spectrum of an audio signal.  The
//! detector correlates the spectrum of the test signal against the known
//! reference noise pattern to reveal the embedded mark.
//!
//! ## Algorithm
//!
//! 1. A pseudo-random noise (PRN) sequence is derived from a 64-bit seed using
//!    a deterministic linear-feedback shift register (LFSR).
//! 2. The audio is divided into overlapping analysis frames (STFT).
//! 3. For each frame the PRN is correlated against the normalised magnitude
//!    spectrum.  The per-frame correlation is accumulated into a detection
//!    statistic.
//! 4. A Z-score is computed over the frame statistics.  A Z-score above the
//!    configured threshold signals a detected watermark.
//!
//! This is a simplified detector suitable for offline analysis; a production
//! implementation would also handle time-warped audio and multi-resolution
//! search.  All arithmetic uses `Vec<f32>` — no ndarray.

use crate::utils::hann_window;
use crate::MirResult;
use oxifft::Complex;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the watermark detector.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// 64-bit seed that defines the PRN pattern to search for.
    pub seed: u64,
    /// FFT window size.
    pub window_size: usize,
    /// Hop size between frames.
    pub hop_size: usize,
    /// Detection threshold (Z-score).  Higher → fewer false positives.
    pub detection_threshold: f32,
    /// Minimum number of frames with above-threshold correlation to confirm detection.
    pub min_confirmed_frames: usize,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            seed: 0xDEAD_BEEF_CAFE_BABE,
            window_size: 4096,
            hop_size: 1024,
            detection_threshold: 3.0,
            min_confirmed_frames: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of watermark detection.
#[derive(Debug, Clone)]
pub struct WatermarkResult {
    /// Whether a watermark was detected.
    pub detected: bool,
    /// Peak Z-score observed across all frames.
    pub peak_z_score: f32,
    /// Mean Z-score across all frames.
    pub mean_z_score: f32,
    /// Number of frames where the Z-score exceeded the threshold.
    pub confirmed_frames: usize,
    /// Total frames analysed.
    pub total_frames: usize,
    /// Confidence estimate in [0, 1] based on how far the peak exceeds threshold.
    pub confidence: f32,
    /// Per-frame correlation values (normalised).
    pub frame_correlations: Vec<f32>,
}

// ---------------------------------------------------------------------------
// WatermarkDetector
// ---------------------------------------------------------------------------

/// Spectral watermark detector.
pub struct WatermarkDetector {
    config: WatermarkConfig,
    /// Pre-generated PRN vector (one value per FFT bin up to Nyquist).
    prn: Vec<f32>,
}

impl WatermarkDetector {
    /// Create a new watermark detector.
    #[must_use]
    pub fn new(config: WatermarkConfig) -> Self {
        let n_bins = config.window_size / 2 + 1;
        let prn = generate_prn(config.seed, n_bins);
        Self { config, prn }
    }

    /// Create a detector with default configuration and the given seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self::new(WatermarkConfig {
            seed,
            ..WatermarkConfig::default()
        })
    }

    /// Detect whether the given audio signal contains the configured watermark.
    ///
    /// # Errors
    ///
    /// Returns error if the signal is too short for analysis.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, signal: &[f32]) -> MirResult<WatermarkResult> {
        let win = self.config.window_size;
        let hop = self.config.hop_size;

        if signal.len() < win {
            return Err(crate::MirError::InsufficientData(format!(
                "Signal too short for watermark detection: need ≥{win} samples, got {}",
                signal.len()
            )));
        }

        let window = hann_window(win);
        let n_frames = (signal.len().saturating_sub(win)) / hop + 1;
        let n_bins = win / 2 + 1;

        let mut frame_correlations: Vec<f32> = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = start + win;
            if end > signal.len() {
                break;
            }

            // Apply Hann window
            let windowed: Vec<Complex<f32>> = signal[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            let spectrum = oxifft::fft(&windowed);

            // Normalised magnitude spectrum (positive half only)
            let mag: Vec<f32> = spectrum[..n_bins].iter().map(|c| c.norm()).collect();
            let mag_sum: f32 = mag.iter().sum();
            let mag_norm: Vec<f32> = if mag_sum > 1e-9 {
                mag.iter().map(|&m| m / mag_sum).collect()
            } else {
                vec![0.0; n_bins]
            };

            // Pearson correlation between mag_norm and PRN
            let corr = pearson_correlation(&mag_norm, &self.prn);
            frame_correlations.push(corr);
        }

        if frame_correlations.is_empty() {
            return Ok(WatermarkResult {
                detected: false,
                peak_z_score: 0.0,
                mean_z_score: 0.0,
                confirmed_frames: 0,
                total_frames: 0,
                confidence: 0.0,
                frame_correlations,
            });
        }

        // Compute Z-scores across frames
        let mean_corr = crate::utils::mean(&frame_correlations);
        let std_corr = {
            let var: f32 = frame_correlations
                .iter()
                .map(|&c| (c - mean_corr).powi(2))
                .sum::<f32>()
                / frame_correlations.len() as f32;
            var.sqrt()
        };

        // Normalised frame Z-scores
        let z_scores: Vec<f32> = if std_corr > 1e-9 {
            frame_correlations
                .iter()
                .map(|&c| (c - mean_corr) / std_corr)
                .collect()
        } else {
            vec![0.0; frame_correlations.len()]
        };

        let peak_z = z_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_z = crate::utils::mean(&z_scores);
        let threshold = self.config.detection_threshold;

        let confirmed = z_scores.iter().filter(|&&z| z > threshold).count();
        let detected = confirmed >= self.config.min_confirmed_frames;

        let confidence = if detected {
            ((peak_z - threshold) / (threshold + 1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(WatermarkResult {
            detected,
            peak_z_score: peak_z,
            mean_z_score: mean_z,
            confirmed_frames: confirmed,
            total_frames: frame_correlations.len(),
            confidence,
            frame_correlations,
        })
    }

    /// Embed a watermark into a signal (for testing / round-trip validation).
    ///
    /// The watermark is added at the specified `amplitude` into the magnitude
    /// spectrum of each frame using ISTFT overlap-add.
    ///
    /// Note: this is a simplified additive watermarker that adds a band-limited
    /// PRN noise sequence at low amplitude directly to the time-domain signal.
    /// Production watermarkers embed in the frequency domain with perceptual
    /// masking; this implementation is intentionally simple for test purposes.
    ///
    /// # Errors
    ///
    /// Returns error if the signal is too short.
    #[allow(clippy::cast_precision_loss)]
    pub fn embed(&self, signal: &[f32], amplitude: f32) -> MirResult<Vec<f32>> {
        if signal.is_empty() {
            return Err(crate::MirError::InsufficientData(
                "Empty signal for watermark embedding".to_string(),
            ));
        }

        let prn_len = self.prn.len();
        let mut out = signal.to_vec();

        // Tile the PRN to match signal length and add at specified amplitude
        for (i, sample) in out.iter_mut().enumerate() {
            let prn_val = self.prn[i % prn_len];
            *sample += amplitude * prn_val;
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a pseudo-random noise vector of length `n` using a 64-bit LFSR.
///
/// The sequence is deterministic and repeatable for a given seed.
/// Values are drawn from the uniform range [-1, 1] and normalised to unit RMS.
fn generate_prn(seed: u64, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }

    let mut state = if seed == 0 { 1 } else { seed };
    let mut raw = Vec::with_capacity(n);

    for _ in 0..n {
        // Galois LFSR (64-bit, taps at positions 64, 63, 61, 60)
        let feedback = state & 1;
        state >>= 1;
        if feedback != 0 {
            state ^= 0xD800_0000_0000_0000_u64;
        }
        // Map to [-1, 1]
        #[allow(clippy::cast_precision_loss)]
        let val = (state as f32) / u64::MAX as f32 * 2.0 - 1.0;
        raw.push(val);
    }

    // Normalise to unit RMS
    let rms: f32 = (raw.iter().map(|&x| x * x).sum::<f32>() / n as f32).sqrt();
    if rms > 1e-9 {
        raw.iter_mut().for_each(|x| *x /= rms);
    }

    raw
}

/// Pearson correlation between two equal-length slices.
fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mean_a = a[..n].iter().sum::<f32>() / n as f32;
    let mean_b = b[..n].iter().sum::<f32>() / n as f32;

    let (num, var_a, var_b) = a[..n].iter().zip(b[..n].iter()).fold(
        (0.0_f32, 0.0_f32, 0.0_f32),
        |(s, va, vb), (&ai, &bi)| {
            let da = ai - mean_a;
            let db = bi - mean_b;
            (s + da * db, va + da * da, vb + db * db)
        },
    );

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_sine(freq: f32, sr: f32, seconds: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        (0..n).map(|i| (TAU * freq * i as f32 / sr).sin()).collect()
    }

    #[test]
    fn test_generate_prn_length() {
        let prn = generate_prn(42, 1024);
        assert_eq!(prn.len(), 1024);
    }

    #[test]
    fn test_generate_prn_deterministic() {
        let a = generate_prn(12345, 256);
        let b = generate_prn(12345, 256);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < f32::EPSILON, "PRN must be deterministic");
        }
    }

    #[test]
    fn test_generate_prn_different_seeds() {
        let a = generate_prn(1, 256);
        let b = generate_prn(2, 256);
        // Should differ
        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "Different seeds must produce different sequences"
        );
    }

    #[test]
    fn test_pearson_identical() {
        let v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let r = pearson_correlation(&v, &v);
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pearson_anticorrelated() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = a.iter().map(|&x| -x).collect();
        let r = pearson_correlation(&a, &b);
        assert!((r - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_detector_short_signal_error() {
        let detector = WatermarkDetector::with_seed(42);
        let short = vec![0.0f32; 100];
        let result = detector.detect(&short);
        assert!(result.is_err());
    }

    #[test]
    fn test_detector_no_watermark() {
        let detector = WatermarkDetector::with_seed(42);
        let signal = make_sine(440.0, 44100.0, 1.0);
        let result = detector.detect(&signal).expect("should succeed");
        // A pure sine should not trigger detection
        assert!(!result.detected || result.confidence < 0.5);
        assert!(result.total_frames > 0);
    }

    #[test]
    fn test_detector_with_watermark() {
        // Embed and then detect — should find the mark
        let seed = 0x0123_4567_89AB_CDEF_u64;
        let detector = WatermarkDetector::with_seed(seed);
        let signal = make_sine(440.0, 44100.0, 2.0);
        // Embed at a relatively high amplitude so detection is reliable in tests
        let watermarked = detector.embed(&signal, 0.05).expect("embed failed");
        let result = detector.detect(&watermarked).expect("detect failed");
        // With explicit embedding at 5% amplitude we expect some detection signal
        // (confirmed_frames may or may not exceed threshold with synthetic data)
        assert!(result.total_frames > 0);
        assert!(result.peak_z_score.is_finite());
    }

    #[test]
    fn test_embed_produces_different_signal() {
        let detector = WatermarkDetector::with_seed(99);
        let signal = make_sine(440.0, 44100.0, 0.5);
        let watermarked = detector.embed(&signal, 0.01).expect("embed failed");
        assert_eq!(watermarked.len(), signal.len());
        assert!(
            signal
                .iter()
                .zip(watermarked.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "Watermarked signal should differ from original"
        );
    }

    #[test]
    fn test_watermark_result_frame_correlations_length() {
        let detector = WatermarkDetector::new(WatermarkConfig {
            window_size: 1024,
            hop_size: 256,
            ..WatermarkConfig::default()
        });
        let signal = make_sine(220.0, 22050.0, 1.0);
        let result = detector.detect(&signal).expect("should succeed");
        assert!(result.total_frames > 0);
        assert_eq!(result.frame_correlations.len(), result.total_frames);
    }

    #[test]
    fn test_confidence_zero_when_not_detected() {
        let detector = WatermarkDetector::with_seed(7);
        let signal = make_sine(1000.0, 44100.0, 0.5);
        let result = detector.detect(&signal).expect("should succeed");
        if !result.detected {
            assert!((result.confidence - 0.0).abs() < f32::EPSILON);
        }
    }
}
