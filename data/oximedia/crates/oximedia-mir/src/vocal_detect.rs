#![allow(dead_code)]
//! Vocal presence detection and vocal/instrumental classification.
//!
//! Uses spectral analysis heuristics (harmonic-to-noise ratio, formant
//! energy bands) to estimate whether a segment of audio contains singing
//! or speech versus purely instrumental content.

use std::f32::consts::PI;

/// Result of a vocal detection analysis on one audio segment.
#[derive(Debug, Clone)]
pub struct VocalDetectionResult {
    /// Probability (0.0 .. 1.0) that the segment contains vocals.
    pub vocal_probability: f32,
    /// Classification derived from `vocal_probability`.
    pub classification: VocalClass,
    /// Per-frame vocal-presence scores (one per hop).
    pub frame_scores: Vec<f32>,
    /// Duration of the analysed segment in seconds.
    pub duration_secs: f32,
}

/// High-level vocal / instrumental classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocalClass {
    /// Predominantly vocal content.
    Vocal,
    /// Predominantly instrumental content.
    Instrumental,
    /// Mixed or uncertain.
    Mixed,
}

/// Configuration for the vocal detector.
#[derive(Debug, Clone)]
pub struct VocalDetectorConfig {
    /// Sample rate of the input audio.
    pub sample_rate: f32,
    /// FFT window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Lower edge of the vocal formant band (Hz).
    pub formant_low_hz: f32,
    /// Upper edge of the vocal formant band (Hz).
    pub formant_high_hz: f32,
    /// Threshold above which a frame is considered vocal.
    pub vocal_threshold: f32,
}

impl Default for VocalDetectorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            window_size: 2048,
            hop_size: 512,
            formant_low_hz: 300.0,
            formant_high_hz: 3400.0,
            vocal_threshold: 0.45,
        }
    }
}

/// Vocal presence detector.
pub struct VocalDetector {
    config: VocalDetectorConfig,
}

impl VocalDetector {
    /// Create a new vocal detector with the given configuration.
    #[must_use]
    pub fn new(config: VocalDetectorConfig) -> Self {
        Self { config }
    }

    /// Analyse a mono audio buffer and return vocal detection results.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, samples: &[f32]) -> VocalDetectionResult {
        if samples.is_empty() {
            return VocalDetectionResult {
                vocal_probability: 0.0,
                classification: VocalClass::Instrumental,
                frame_scores: Vec::new(),
                duration_secs: 0.0,
            };
        }

        let mut frame_scores = Vec::new();
        let n = self.config.window_size;
        let hop = self.config.hop_size;
        let sr = self.config.sample_rate;

        let mut pos = 0;
        while pos + n <= samples.len() {
            let frame = &samples[pos..pos + n];
            let score = self.frame_vocal_score(frame, n, sr);
            frame_scores.push(score);
            pos += hop;
        }

        let vocal_probability = if frame_scores.is_empty() {
            0.0
        } else {
            frame_scores.iter().sum::<f32>() / frame_scores.len() as f32
        };

        let classification = classify_vocal(vocal_probability, self.config.vocal_threshold);
        let duration_secs = samples.len() as f32 / sr;

        VocalDetectionResult {
            vocal_probability,
            classification,
            frame_scores,
            duration_secs,
        }
    }

    /// Compute a vocal likelihood score for a single windowed frame.
    #[allow(clippy::cast_precision_loss)]
    fn frame_vocal_score(&self, frame: &[f32], n: usize, sr: f32) -> f32 {
        // Compute magnitude spectrum via DFT approximation (simplified).
        let magnitudes = simple_magnitude_spectrum(frame, n);

        // Compute energy in the vocal formant band vs total.
        let bin_hz = sr / n as f32;
        let lo_bin = (self.config.formant_low_hz / bin_hz) as usize;
        let hi_bin = ((self.config.formant_high_hz / bin_hz) as usize).min(magnitudes.len());

        let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
        if total_energy < 1e-12 {
            return 0.0;
        }

        let formant_energy: f32 = magnitudes[lo_bin..hi_bin].iter().map(|m| m * m).sum();
        let ratio = formant_energy / total_energy;

        // Apply a simple sigmoid-style mapping to [0, 1].
        sigmoid(4.0 * (ratio - 0.3))
    }
}

/// Classify based on probability and threshold.
#[must_use]
fn classify_vocal(prob: f32, threshold: f32) -> VocalClass {
    if prob >= threshold + 0.15 {
        VocalClass::Vocal
    } else if prob < threshold - 0.15 {
        VocalClass::Instrumental
    } else {
        VocalClass::Mixed
    }
}

/// Simplified magnitude spectrum (DFT of first `n` bins).
///
/// For production use a proper FFT; this is a lightweight pure-Rust fallback.
#[allow(clippy::cast_precision_loss)]
fn simple_magnitude_spectrum(frame: &[f32], n: usize) -> Vec<f32> {
    let half = n / 2 + 1;
    let mut mags = vec![0.0_f32; half];
    let n_f = n as f32;
    for (k, mag) in mags.iter_mut().enumerate() {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (i, &sample) in frame.iter().enumerate().take(n) {
            let angle = 2.0 * PI * k as f32 * i as f32 / n_f;
            re += sample * angle.cos();
            im -= sample * angle.sin();
        }
        *mag = (re * re + im * im).sqrt();
    }
    mags
}

/// Standard sigmoid function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the harmonic-to-noise ratio (simplified) of a frame.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn harmonic_to_noise_ratio(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let energy: f32 = frame.iter().map(|s| s * s).sum();
    let n = frame.len();
    // Autocorrelation at lag 1 as a rough harmonic proxy.
    let mut auto = 0.0_f32;
    for i in 0..n - 1 {
        auto += frame[i] * frame[i + 1];
    }
    if energy < 1e-12 {
        return 0.0;
    }
    (auto.abs() / energy).min(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, sr: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_detect_silence() {
        let det = VocalDetector::new(VocalDetectorConfig::default());
        let silence = vec![0.0_f32; 4096];
        let result = det.detect(&silence);
        assert_eq!(result.classification, VocalClass::Instrumental);
        assert!(result.vocal_probability < 0.1);
    }

    #[test]
    fn test_detect_empty() {
        let det = VocalDetector::new(VocalDetectorConfig::default());
        let result = det.detect(&[]);
        assert_eq!(result.classification, VocalClass::Instrumental);
        assert!(result.frame_scores.is_empty());
    }

    #[test]
    fn test_detect_low_freq_instrumental() {
        // Use a small window/hop and reduced sample count so the O(N²) DFT stays fast.
        let cfg = VocalDetectorConfig {
            sample_rate: 8000.0,
            window_size: 256,
            hop_size: 128,
            ..VocalDetectorConfig::default()
        };
        let det = VocalDetector::new(cfg);
        // 80 Hz bass -- well below the vocal formant band; 0.5 s at 8 kHz = 4000 samples
        let bass = sine_wave(80.0, 8000.0, 4000);
        let result = det.detect(&bass);
        assert!(
            result.vocal_probability < 0.5,
            "low bass should not be classified as vocal, got {}",
            result.vocal_probability
        );
    }

    #[test]
    fn test_detect_vocal_range_sine() {
        // Use a small window/hop and reduced sample count so the O(N²) DFT stays fast.
        let cfg = VocalDetectorConfig {
            sample_rate: 8000.0,
            window_size: 256,
            hop_size: 128,
            ..VocalDetectorConfig::default()
        };
        let det = VocalDetector::new(cfg);
        // 1000 Hz sine sits inside the vocal formant band; 0.5 s at 8 kHz = 4000 samples
        let vocal_sine = sine_wave(1000.0, 8000.0, 4000);
        let result = det.detect(&vocal_sine);
        // Should have higher vocal probability than the bass case
        assert!(result.vocal_probability > 0.3);
    }

    #[test]
    fn test_classify_vocal() {
        assert_eq!(classify_vocal(0.8, 0.45), VocalClass::Vocal);
        assert_eq!(classify_vocal(0.1, 0.45), VocalClass::Instrumental);
        assert_eq!(classify_vocal(0.45, 0.45), VocalClass::Mixed);
    }

    #[test]
    fn test_sigmoid_center() {
        let val = sigmoid(0.0);
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let val = sigmoid(10.0);
        assert!(val > 0.99);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let val = sigmoid(-10.0);
        assert!(val < 0.01);
    }

    #[test]
    fn test_hnr_silence() {
        let silence = vec![0.0_f32; 1024];
        assert!((harmonic_to_noise_ratio(&silence) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hnr_pure_tone() {
        let tone = sine_wave(440.0, 44100.0, 4096);
        let hnr = harmonic_to_noise_ratio(&tone);
        // A pure tone should have high autocorrelation.
        assert!(hnr > 0.5, "pure tone HNR should be high, got {hnr}");
    }

    #[test]
    fn test_duration_calculation() {
        // Use 8 kHz / small window to avoid O(N²) DFT overhead.
        let cfg = VocalDetectorConfig {
            sample_rate: 8000.0,
            window_size: 256,
            hop_size: 128,
            ..VocalDetectorConfig::default()
        };
        let det = VocalDetector::new(cfg);
        let samples = vec![0.0_f32; 8000]; // 1 second at 8 kHz
        let result = det.detect(&samples);
        assert!((result.duration_secs - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_frame_scores_count() {
        let cfg = VocalDetectorConfig {
            window_size: 1024,
            hop_size: 512,
            ..VocalDetectorConfig::default()
        };
        let det = VocalDetector::new(cfg);
        let samples = vec![0.0_f32; 4096];
        let result = det.detect(&samples);
        // Expected frames: (4096 - 1024) / 512 + 1 = 7
        assert_eq!(result.frame_scores.len(), 7);
    }

    #[test]
    fn test_config_default_values() {
        let cfg = VocalDetectorConfig::default();
        assert!((cfg.sample_rate - 44100.0).abs() < f32::EPSILON);
        assert_eq!(cfg.window_size, 2048);
        assert!((cfg.vocal_threshold - 0.45).abs() < f32::EPSILON);
    }
}
