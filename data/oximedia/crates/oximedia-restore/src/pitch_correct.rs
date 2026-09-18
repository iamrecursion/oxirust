#![allow(dead_code)]
//! Pitch correction and intonation repair for audio restoration.
//!
//! This module provides tools for detecting and correcting pitch drift,
//! tape-speed-induced pitch shifts, and micro-tonal intonation problems
//! commonly found in archival audio recordings.

use std::f64::consts::PI;

/// Standard concert pitch reference frequency in Hz.
const A4_HZ: f64 = 440.0;

/// Semitone ratio (equal temperament).
const SEMITONE_RATIO: f64 = 1.059_463_094_359_295_3;

/// Detected pitch measurement at a point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchMeasurement {
    /// Time offset in seconds from the start of the audio.
    pub time_secs: f64,
    /// Detected fundamental frequency in Hz.
    pub frequency_hz: f64,
    /// Confidence of the pitch detection (0.0 to 1.0).
    pub confidence: f64,
}

impl PitchMeasurement {
    /// Create a new pitch measurement.
    pub fn new(time_secs: f64, frequency_hz: f64, confidence: f64) -> Self {
        Self {
            time_secs,
            frequency_hz,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Calculate how many cents this measurement deviates from the nearest
    /// semitone on a standard equal-temperament scale.
    #[allow(clippy::cast_precision_loss)]
    pub fn cents_deviation(&self) -> f64 {
        if self.frequency_hz <= 0.0 {
            return 0.0;
        }
        let semitones_from_a4 = 12.0 * (self.frequency_hz / A4_HZ).ln() / 2.0_f64.ln();
        let nearest = semitones_from_a4.round();
        (semitones_from_a4 - nearest) * 100.0
    }
}

/// Strategy for correcting pitch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionStrategy {
    /// Snap to nearest semitone in equal temperament.
    SnapToSemitone,
    /// Apply a constant offset in cents across the whole file.
    ConstantOffset,
    /// Smooth drift gradually using a low-pass envelope.
    SmoothDrift,
}

/// Configuration for the pitch corrector.
#[derive(Debug, Clone)]
pub struct PitchCorrectorConfig {
    /// Reference tuning frequency for A4 in Hz.
    pub reference_hz: f64,
    /// Maximum correction range in cents.
    pub max_correction_cents: f64,
    /// Correction speed (0.0 = slow/smooth, 1.0 = instant snap).
    pub correction_speed: f64,
    /// Minimum confidence to apply correction.
    pub min_confidence: f64,
    /// Strategy to use.
    pub strategy: CorrectionStrategy,
}

impl Default for PitchCorrectorConfig {
    fn default() -> Self {
        Self {
            reference_hz: A4_HZ,
            max_correction_cents: 100.0,
            correction_speed: 0.5,
            min_confidence: 0.6,
            strategy: CorrectionStrategy::SnapToSemitone,
        }
    }
}

/// Pitch corrector that repairs intonation drift.
#[derive(Debug)]
pub struct PitchCorrector {
    /// Configuration.
    config: PitchCorrectorConfig,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Running phase accumulator for resampling.
    phase: f64,
}

impl PitchCorrector {
    /// Create a new pitch corrector with the given configuration and sample rate.
    pub fn new(config: PitchCorrectorConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
            phase: 0.0,
        }
    }

    /// Create a pitch corrector with default configuration.
    pub fn with_defaults(sample_rate: u32) -> Self {
        Self::new(PitchCorrectorConfig::default(), sample_rate)
    }

    /// Estimate the pitch shift ratio needed for a given deviation in cents.
    #[allow(clippy::cast_precision_loss)]
    fn shift_ratio_for_cents(cents: f64) -> f64 {
        2.0_f64.powf(cents / 1200.0)
    }

    /// Compute a correction envelope from a series of pitch measurements.
    ///
    /// Returns a vector of per-sample pitch-shift ratios.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_correction_envelope(
        &self,
        measurements: &[PitchMeasurement],
        num_samples: usize,
    ) -> Vec<f64> {
        if measurements.is_empty() || num_samples == 0 {
            return vec![1.0; num_samples];
        }

        let mut envelope = vec![1.0; num_samples];
        let sr = f64::from(self.sample_rate);

        match self.config.strategy {
            CorrectionStrategy::SnapToSemitone => {
                for m in measurements {
                    if m.confidence < self.config.min_confidence {
                        continue;
                    }
                    let deviation = m.cents_deviation();
                    let clamped = deviation.clamp(
                        -self.config.max_correction_cents,
                        self.config.max_correction_cents,
                    );
                    let correction = -clamped * self.config.correction_speed;
                    let ratio = Self::shift_ratio_for_cents(correction);

                    let sample_idx = (m.time_secs * sr) as usize;
                    if sample_idx < num_samples {
                        envelope[sample_idx] = ratio;
                    }
                }
                // Interpolate between keyframes
                Self::interpolate_envelope(&mut envelope);
            }
            CorrectionStrategy::ConstantOffset => {
                let total_dev: f64 = measurements
                    .iter()
                    .filter(|m| m.confidence >= self.config.min_confidence)
                    .map(|m| m.cents_deviation())
                    .sum();
                let count = measurements
                    .iter()
                    .filter(|m| m.confidence >= self.config.min_confidence)
                    .count();
                if count > 0 {
                    let avg_dev = total_dev / count as f64;
                    let clamped = avg_dev.clamp(
                        -self.config.max_correction_cents,
                        self.config.max_correction_cents,
                    );
                    let ratio = Self::shift_ratio_for_cents(-clamped);
                    for v in &mut envelope {
                        *v = ratio;
                    }
                }
            }
            CorrectionStrategy::SmoothDrift => {
                // Place corrections at measurement times, then low-pass
                for m in measurements {
                    if m.confidence < self.config.min_confidence {
                        continue;
                    }
                    let deviation = m.cents_deviation();
                    let clamped = deviation.clamp(
                        -self.config.max_correction_cents,
                        self.config.max_correction_cents,
                    );
                    let ratio = Self::shift_ratio_for_cents(-clamped);
                    let sample_idx = (m.time_secs * sr) as usize;
                    if sample_idx < num_samples {
                        envelope[sample_idx] = ratio;
                    }
                }
                Self::interpolate_envelope(&mut envelope);
                Self::smooth_envelope(&mut envelope, 0.01);
            }
        }

        envelope
    }

    /// Apply a pitch-shift envelope to audio samples using linear interpolation resampling.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_correction(&mut self, samples: &[f32], envelope: &[f64]) -> Vec<f32> {
        let len = samples.len();
        if len == 0 {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(len);
        self.phase = 0.0;

        while (self.phase as usize) < len {
            let idx = self.phase as usize;
            let frac = self.phase - idx as f64;

            // Linear interpolation
            let s0 = f64::from(samples[idx]);
            let s1 = if idx + 1 < len {
                f64::from(samples[idx + 1])
            } else {
                s0
            };
            let interpolated = s0 + frac * (s1 - s0);
            output.push(interpolated as f32);

            // Advance by the pitch-shift ratio
            let ratio = if idx < envelope.len() {
                envelope[idx]
            } else {
                1.0
            };
            self.phase += ratio;
        }

        // Pad or truncate to match input length
        output.resize(len, 0.0);
        output
    }

    /// Linearly interpolate between set keyframes (values != 1.0) in an envelope.
    fn interpolate_envelope(envelope: &mut [f64]) {
        if envelope.is_empty() {
            return;
        }
        let mut last_key_idx = 0;
        let mut last_key_val = envelope[0];

        for i in 1..envelope.len() {
            if (envelope[i] - 1.0).abs() > 1e-12 {
                // Interpolate from last keyframe to here
                let span = i - last_key_idx;
                if span > 1 {
                    for j in 1..span {
                        let t = j as f64 / span as f64;
                        envelope[last_key_idx + j] =
                            last_key_val + t * (envelope[i] - last_key_val);
                    }
                }
                last_key_idx = i;
                last_key_val = envelope[i];
            }
        }
    }

    /// Simple single-pole low-pass smoothing of an envelope.
    fn smooth_envelope(envelope: &mut [f64], alpha: f64) {
        if envelope.len() < 2 {
            return;
        }
        let coeff = alpha.clamp(0.0, 1.0);
        for i in 1..envelope.len() {
            envelope[i] = envelope[i - 1] + coeff * (envelope[i] - envelope[i - 1]);
        }
    }
}

/// Convert a frequency in Hz to the nearest MIDI note number.
#[allow(clippy::cast_precision_loss)]
pub fn hz_to_midi(freq: f64) -> f64 {
    if freq <= 0.0 {
        return 0.0;
    }
    69.0 + 12.0 * (freq / A4_HZ).log2()
}

/// Convert a MIDI note number to frequency in Hz.
#[allow(clippy::cast_precision_loss)]
pub fn midi_to_hz(midi: f64) -> f64 {
    A4_HZ * 2.0_f64.powf((midi - 69.0) / 12.0)
}

/// Simple autocorrelation-based pitch detector for mono audio.
#[derive(Debug)]
pub struct AutocorrelationPitchDetector {
    /// Minimum expected frequency in Hz.
    min_freq: f64,
    /// Maximum expected frequency in Hz.
    max_freq: f64,
    /// Sample rate in Hz.
    sample_rate: u32,
}

impl AutocorrelationPitchDetector {
    /// Create a new detector for the given frequency range.
    pub fn new(min_freq: f64, max_freq: f64, sample_rate: u32) -> Self {
        Self {
            min_freq: min_freq.max(20.0),
            max_freq: max_freq.min(f64::from(sample_rate) / 2.0),
            sample_rate,
        }
    }

    /// Detect pitch of a windowed audio frame.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, frame: &[f32]) -> Option<PitchMeasurement> {
        let sr = f64::from(self.sample_rate);
        let min_lag = (sr / self.max_freq) as usize;
        let max_lag = (sr / self.min_freq).min(frame.len() as f64 - 1.0) as usize;

        if min_lag >= max_lag || max_lag >= frame.len() {
            return None;
        }

        // Compute normalized autocorrelation
        let energy: f64 = frame.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        if energy < 1e-10 {
            return None;
        }

        let mut best_lag = min_lag;
        let mut best_corr = f64::NEG_INFINITY;

        for lag in min_lag..=max_lag {
            let mut corr = 0.0_f64;
            for i in 0..frame.len() - lag {
                corr += f64::from(frame[i]) * f64::from(frame[i + lag]);
            }
            let norm_corr = corr / energy;
            if norm_corr > best_corr {
                best_corr = norm_corr;
                best_lag = lag;
            }
        }

        let frequency = sr / best_lag as f64;
        let confidence = best_corr.clamp(0.0, 1.0);

        Some(PitchMeasurement::new(0.0, frequency, confidence))
    }
}

/// Apply a Hann window to a frame of audio samples.
#[allow(clippy::cast_precision_loss)]
pub fn apply_hann_window(frame: &mut [f32]) {
    let n = frame.len();
    if n == 0 {
        return;
    }
    for i in 0..n {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1).max(1) as f64).cos());
        frame[i] *= w as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_measurement_new() {
        let m = PitchMeasurement::new(1.0, 440.0, 0.95);
        assert_eq!(m.time_secs, 1.0);
        assert_eq!(m.frequency_hz, 440.0);
        assert_eq!(m.confidence, 0.95);
    }

    #[test]
    fn test_pitch_measurement_confidence_clamped() {
        let m = PitchMeasurement::new(0.0, 440.0, 1.5);
        assert_eq!(m.confidence, 1.0);
        let m2 = PitchMeasurement::new(0.0, 440.0, -0.3);
        assert_eq!(m2.confidence, 0.0);
    }

    #[test]
    fn test_cents_deviation_exact_a4() {
        let m = PitchMeasurement::new(0.0, 440.0, 1.0);
        assert!((m.cents_deviation()).abs() < 0.01);
    }

    #[test]
    fn test_cents_deviation_one_semitone_up() {
        // A#4 = 440 * 2^(1/12) ≈ 466.16
        let freq = 440.0 * SEMITONE_RATIO;
        let m = PitchMeasurement::new(0.0, freq, 1.0);
        assert!(m.cents_deviation().abs() < 1.0);
    }

    #[test]
    fn test_cents_deviation_quarter_tone() {
        // 25 cents sharp of A4 -- unambiguously rounds to A4
        let freq = 440.0 * 2.0_f64.powf(25.0 / 1200.0);
        let m = PitchMeasurement::new(0.0, freq, 1.0);
        assert!((m.cents_deviation() - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_cents_deviation_zero_freq() {
        let m = PitchMeasurement::new(0.0, 0.0, 1.0);
        assert_eq!(m.cents_deviation(), 0.0);
    }

    #[test]
    fn test_hz_to_midi_a4() {
        let midi = hz_to_midi(440.0);
        assert!((midi - 69.0).abs() < 0.01);
    }

    #[test]
    fn test_midi_to_hz_a4() {
        let freq = midi_to_hz(69.0);
        assert!((freq - 440.0).abs() < 0.01);
    }

    #[test]
    fn test_hz_to_midi_roundtrip() {
        let original = 261.63; // C4
        let midi = hz_to_midi(original);
        let back = midi_to_hz(midi);
        assert!((back - original).abs() < 0.01);
    }

    #[test]
    fn test_corrector_constant_offset() {
        let config = PitchCorrectorConfig {
            strategy: CorrectionStrategy::ConstantOffset,
            min_confidence: 0.5,
            max_correction_cents: 100.0,
            correction_speed: 1.0,
            ..Default::default()
        };
        let corrector = PitchCorrector::new(config, 44100);
        // All measurements 20 cents sharp
        let freq_20_cents_sharp = 440.0 * 2.0_f64.powf(20.0 / 1200.0);
        let measurements = vec![
            PitchMeasurement::new(0.0, freq_20_cents_sharp, 0.9),
            PitchMeasurement::new(0.5, freq_20_cents_sharp, 0.9),
        ];
        let env = corrector.compute_correction_envelope(&measurements, 1000);
        assert_eq!(env.len(), 1000);
        // All values should be the same constant
        let first = env[0];
        for &v in &env {
            assert!((v - first).abs() < 1e-9);
        }
        // The ratio should shift pitch downward (ratio < 1 means slow-down = lower pitch)
        assert!(first < 1.0);
    }

    #[test]
    fn test_corrector_empty_measurements() {
        let corrector = PitchCorrector::with_defaults(44100);
        let env = corrector.compute_correction_envelope(&[], 500);
        assert_eq!(env.len(), 500);
        for &v in &env {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_apply_correction_identity() {
        let mut corrector = PitchCorrector::with_defaults(44100);
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let envelope = vec![1.0; 100];
        let result = corrector.apply_correction(&samples, &envelope);
        assert_eq!(result.len(), samples.len());
        for (a, b) in samples.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_hann_window() {
        let mut frame = vec![1.0_f32; 64];
        apply_hann_window(&mut frame);
        // Endpoints should be near zero
        assert!(frame[0].abs() < 1e-5);
        assert!(frame[63].abs() < 1e-5);
        // Middle should be near 1.0
        assert!((frame[32] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_autocorrelation_detector_sine() {
        let sr = 8000_u32;
        let freq = 400.0;
        let n = 512;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(sr)).sin() as f32)
            .collect();
        let detector = AutocorrelationPitchDetector::new(100.0, 2000.0, sr);
        let result = detector.detect(&samples);
        assert!(result.is_some());
        let m = result.expect("should succeed in test");
        assert!((m.frequency_hz - freq).abs() < 20.0);
    }

    #[test]
    fn test_autocorrelation_detector_silence() {
        let samples = vec![0.0_f32; 256];
        let detector = AutocorrelationPitchDetector::new(100.0, 2000.0, 8000);
        let result = detector.detect(&samples);
        assert!(result.is_none());
    }
}
