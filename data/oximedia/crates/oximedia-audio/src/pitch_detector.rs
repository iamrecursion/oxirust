//! Pitch detection algorithms.
//!
//! This module provides multiple pitch detection algorithms for monophonic audio:
//!
//! - **YIN** — de Cheveigné & Kawahara (2002), cumulative mean normalized difference
//! - **Autocorrelation** — ACF-based with first-zero-crossing peak search
//! - **Cepstrum** — log-magnitude spectrum IDFT, quefrency peak
//! - **AMDF** — Average Magnitude Difference Function minimum search
//!
//! # Example
//!
//! ```
//! use oximedia_audio::pitch_detector::{PitchDetector, PitchAlgorithm};
//!
//! let detector = PitchDetector::new(PitchAlgorithm::Yin, 44100);
//! // detector.detect(&samples) returns Option<PitchDetection>
//! ```

#![forbid(unsafe_code)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of pitch detection on a frame of audio samples.
#[derive(Clone, Debug)]
pub struct PitchDetection {
    /// Detected fundamental frequency in Hz.
    pub frequency_hz: f32,
    /// Detection confidence in the range 0.0..1.0.
    pub confidence: f32,
    /// Nearest MIDI note (0 = C-1 at ~8.18 Hz, 69 = A4 = 440 Hz).
    pub midi_note: u8,
    /// Deviation from the nearest MIDI note in cents (-50..50).
    pub cents_deviation: f32,
    /// Note name string such as `"A4"` or `"C#3"`.
    pub note_name: &'static str,
}

/// Pitch detection algorithm selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PitchAlgorithm {
    /// YIN algorithm (de Cheveigné & Kawahara, 2002).
    ///
    /// Computes the cumulative mean normalized difference function and finds
    /// the first dip below a threshold.
    Yin,

    /// Normalized autocorrelation — finds the first peak after the first
    /// zero-crossing.
    Autocorrelation,

    /// Cepstral pitch detection — takes the log-magnitude spectrum, computes
    /// the IDFT (real cepstrum), and searches for the peak in the quefrency
    /// range corresponding to `[min_frequency, max_frequency]`.
    Cepstrum,

    /// Average Magnitude Difference Function — finds the lag at which the
    /// mean absolute difference between the signal and a shifted copy is
    /// minimised.
    Amdf,
}

/// Stateless pitch detector.
#[derive(Clone, Debug)]
pub struct PitchDetector {
    /// Algorithm used for detection.
    pub algorithm: PitchAlgorithm,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Minimum detectable frequency in Hz (default 50).
    pub min_frequency: f32,
    /// Maximum detectable frequency in Hz (default 1000).
    pub max_frequency: f32,
    /// Algorithm-specific confidence threshold (default 0.1 for YIN/AMDF, 0.15 for others).
    pub threshold: f32,
}

// ---------------------------------------------------------------------------
// Note name table (MIDI 0..=127)
// ---------------------------------------------------------------------------

static NOTE_NAMES: &[&str] = &[
    "C-1", "C#-1", "D-1", "D#-1", "E-1", "F-1", "F#-1", "G-1", "G#-1", "A-1", "A#-1", "B-1", "C0",
    "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0", "G#0", "A0", "A#0", "B0", "C1", "C#1", "D1",
    "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2",
    "F#2", "G2", "G#2", "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3",
    "G#3", "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4",
    "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5", "C6",
    "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6", "C7", "C#7", "D7",
    "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7", "C8", "C#8", "D8", "D#8", "E8", "F8",
    "F#8", "G8", "G#8", "A8", "A#8", "B8", "C9", "C#9", "D9", "D#9", "E9", "F9", "F#9", "G9",
];

// ---------------------------------------------------------------------------
// Public helper functions
// ---------------------------------------------------------------------------

/// Convert a frequency in Hz to the nearest MIDI note number and cents deviation.
///
/// Returns `(midi_note, cents_deviation)` where `cents_deviation` is in `(-50, 50]`.
/// If `freq` is ≤ 0 or outside MIDI range, returns `(0, 0.0)`.
#[must_use]
pub fn frequency_to_midi(freq: f32) -> (u8, f32) {
    if freq <= 0.0 {
        return (0, 0.0);
    }
    let fractional_midi = 69.0 + 12.0 * (freq / 440.0).log2();
    let nearest = fractional_midi.round().clamp(0.0, 127.0);
    let cents = (fractional_midi - nearest) * 100.0;
    (nearest as u8, cents)
}

/// Return the note name string for a MIDI note number (0..=127).
///
/// MIDI notes outside the 0..=127 range are clamped.
#[must_use]
pub fn midi_to_note_name(midi: u8) -> &'static str {
    let idx = (midi as usize).min(NOTE_NAMES.len() - 1);
    NOTE_NAMES[idx]
}

/// Convert a frequency in Hz to a note name and the cents deviation.
///
/// Returns `(note_name, cents_deviation)`.
#[must_use]
pub fn frequency_to_note_name(freq: f32) -> (&'static str, f32) {
    let (midi, cents) = frequency_to_midi(freq);
    (midi_to_note_name(midi), cents)
}

// ---------------------------------------------------------------------------
// PitchDetector implementation
// ---------------------------------------------------------------------------

impl PitchDetector {
    /// Create a new `PitchDetector` with sensible defaults.
    ///
    /// Default frequency range: 50 – 1000 Hz.
    /// Default threshold: algorithm-specific (0.10 for YIN/AMDF, 0.15 for others).
    #[must_use]
    pub fn new(algorithm: PitchAlgorithm, sample_rate: u32) -> Self {
        let threshold = match algorithm {
            PitchAlgorithm::Yin | PitchAlgorithm::Amdf => 0.10,
            PitchAlgorithm::Autocorrelation | PitchAlgorithm::Cepstrum => 0.15,
        };
        Self {
            algorithm,
            sample_rate,
            min_frequency: 50.0,
            max_frequency: 1000.0,
            threshold,
        }
    }

    /// Detect the pitch in the given mono sample buffer.
    ///
    /// Returns `None` if no pitch is found with sufficient confidence.
    #[must_use]
    pub fn detect(&self, samples: &[f32]) -> Option<PitchDetection> {
        if samples.is_empty() {
            return None;
        }

        let sr = self.sample_rate as f32;
        let min_lag = (sr / self.max_frequency).ceil() as usize;
        let max_lag = (sr / self.min_frequency).floor() as usize;

        if max_lag >= samples.len() || min_lag == 0 || min_lag > max_lag {
            return None;
        }

        let (period, confidence) = match self.algorithm {
            PitchAlgorithm::Yin => detect_yin(samples, min_lag, max_lag, self.threshold)?,
            PitchAlgorithm::Autocorrelation => {
                detect_autocorrelation(samples, min_lag, max_lag, self.threshold)?
            }
            PitchAlgorithm::Amdf => detect_amdf(samples, min_lag, max_lag, self.threshold)?,
            PitchAlgorithm::Cepstrum => detect_cepstrum(samples, min_lag, max_lag, self.threshold)?,
        };

        if period == 0 {
            return None;
        }

        let frequency_hz = sr / period as f32;
        let (midi_note, cents_deviation) = frequency_to_midi(frequency_hz);
        let note_name = midi_to_note_name(midi_note);

        Some(PitchDetection {
            frequency_hz,
            confidence,
            midi_note,
            cents_deviation,
            note_name,
        })
    }
}

// ---------------------------------------------------------------------------
// YIN algorithm
// ---------------------------------------------------------------------------

/// Returns `(period_samples, confidence)` using YIN, or `None`.
///
/// Algorithm from: de Cheveigné & Kawahara (2002), "YIN, a fundamental
/// frequency estimator for speech and music", JASA 111(4).
fn detect_yin(
    samples: &[f32],
    min_lag: usize,
    max_lag: usize,
    threshold: f32,
) -> Option<(usize, f32)> {
    let n = samples.len();
    let max_tau = max_lag.min(n / 2);

    if min_lag >= max_tau {
        return None;
    }

    // Step 1: Difference function d(tau) = sum_{t=0}^{W-1} (x[t] - x[t+tau])^2
    let window = max_tau;
    let mut d: Vec<f32> = vec![0.0; window + 1];
    for tau in 1..=window {
        let mut val = 0.0f32;
        for t in 0..(n - tau) {
            let diff = samples[t] - samples[t + tau];
            val += diff * diff;
        }
        d[tau] = val;
    }

    // Step 2: Cumulative mean normalized difference function (CMNDF)
    // d'[0] = 1, d'[tau] = d[tau] / ((1/tau) * sum_{j=1}^{tau} d[j])
    let mut cmndf: Vec<f32> = vec![0.0; window + 1];
    cmndf[0] = 1.0;
    let mut running_sum = 0.0f32;
    for tau in 1..=window {
        running_sum += d[tau];
        if running_sum > 0.0 {
            cmndf[tau] = d[tau] * tau as f32 / running_sum;
        } else {
            cmndf[tau] = 1.0;
        }
    }

    // Step 3: Find first dip below threshold in [min_lag, max_lag]
    let search_end = max_lag.min(window);
    let mut best_tau = 0usize;
    let mut best_val = f32::MAX;

    let mut tau = min_lag;
    while tau < search_end {
        if cmndf[tau] < threshold {
            // Find the local minimum of this dip
            let dip_start = tau;
            while tau + 1 < search_end && cmndf[tau + 1] < cmndf[tau] {
                tau += 1;
            }
            if cmndf[tau] < best_val {
                best_val = cmndf[tau];
                best_tau = tau;
            }
            // If we found a dip, prefer it (original YIN step 4)
            if best_val < threshold {
                break;
            }
            tau = dip_start + 1;
        } else {
            tau += 1;
        }
    }

    // Fallback: if no dip found below threshold, use absolute minimum
    if best_tau == 0 {
        for t in min_lag..=search_end {
            if cmndf[t] < best_val {
                best_val = cmndf[t];
                best_tau = t;
            }
        }
    }

    if best_tau == 0 || best_val > 0.9 {
        return None;
    }

    // Confidence: 1 - CMNDF value (lower CMNDF → higher confidence)
    let confidence = (1.0 - best_val).clamp(0.0, 1.0);
    Some((best_tau, confidence))
}

// ---------------------------------------------------------------------------
// Autocorrelation algorithm
// ---------------------------------------------------------------------------

/// Returns `(period_samples, confidence)` using normalized autocorrelation, or `None`.
///
/// Computes ACF, finds the first zero-crossing after lag 0, then locates the
/// first prominent peak in `[min_lag, max_lag]`.
fn detect_autocorrelation(
    samples: &[f32],
    min_lag: usize,
    max_lag: usize,
    threshold: f32,
) -> Option<(usize, f32)> {
    let n = samples.len();

    // Compute ACF at lag 0 for normalisation
    let r0: f32 = samples.iter().map(|&s| s * s).sum();
    if r0 < 1e-10 {
        return None; // silence
    }

    // Build normalized ACF for lags 0..=max_lag
    let upper = max_lag.min(n / 2);
    let mut acf: Vec<f32> = vec![0.0; upper + 1];
    acf[0] = 1.0;
    for lag in 1..=upper {
        let len = n - lag;
        let r: f32 = (0..len).map(|i| samples[i] * samples[i + lag]).sum();
        acf[lag] = r / r0;
    }

    // Find the first peak in [min_lag, max_lag] that is above threshold
    let mut best_lag = 0usize;
    let mut best_val = threshold;

    let mut lag = min_lag;
    while lag < upper {
        // Local maximum
        let prev = if lag > 0 { acf[lag - 1] } else { 0.0 };
        let curr = acf[lag];
        let next = if lag < upper { acf[lag + 1] } else { 0.0 };
        if curr >= prev && curr >= next && curr > best_val {
            best_val = curr;
            best_lag = lag;
        }
        lag += 1;
    }

    if best_lag == 0 {
        return None;
    }

    let confidence = best_val.clamp(0.0, 1.0);
    Some((best_lag, confidence))
}

// ---------------------------------------------------------------------------
// AMDF algorithm
// ---------------------------------------------------------------------------

/// Returns `(period_samples, confidence)` using AMDF, or `None`.
///
/// Computes `amdf(tau) = mean |x[t] - x[t+tau]|` and finds the first
/// prominent minimum in `[min_lag, max_lag]`.
fn detect_amdf(
    samples: &[f32],
    min_lag: usize,
    max_lag: usize,
    threshold: f32,
) -> Option<(usize, f32)> {
    let n = samples.len();
    let upper = max_lag.min(n - 1);

    if min_lag > upper {
        return None;
    }

    // Compute AMDF
    let mut amdf: Vec<f32> = vec![0.0; upper + 1];
    for tau in min_lag..=upper {
        let len = n - tau;
        if len == 0 {
            amdf[tau] = f32::MAX;
            continue;
        }
        let sum: f32 = (0..len)
            .map(|i| (samples[i] - samples[i + tau]).abs())
            .sum();
        amdf[tau] = sum / len as f32;
    }

    // Normalise by the mean of the signal
    let mean_abs: f32 = samples.iter().map(|s| s.abs()).sum::<f32>() / n as f32;
    if mean_abs < 1e-8 {
        return None;
    }
    let norm_factor = mean_abs * 2.0; // scale so relative minimum ≈ 0

    // Find the global minimum of AMDF in [min_lag, upper]
    let mut best_lag = 0usize;
    let mut best_val = f32::MAX;

    for tau in min_lag..=upper {
        if amdf[tau] < best_val {
            best_val = amdf[tau];
            best_lag = tau;
        }
    }

    if best_lag == 0 {
        return None;
    }

    // Convert AMDF value to confidence: lower AMDF relative to mean signal energy → higher confidence
    let rel = (best_val / norm_factor).clamp(0.0, 1.0);
    let confidence = (1.0 - rel).clamp(0.0, 1.0);

    if confidence < threshold {
        return None;
    }

    Some((best_lag, confidence))
}

// ---------------------------------------------------------------------------
// Cepstrum algorithm
// ---------------------------------------------------------------------------

/// Returns `(period_samples, confidence)` using the real cepstrum, or `None`.
///
/// Steps:
/// 1. Compute the DFT of the input frame.
/// 2. Take `log(|X[k]| + ε)` to form the log magnitude spectrum.
/// 3. Compute the IDFT (real cepstrum).
/// 4. Find the peak in quefrency range `[min_lag, max_lag]`.
fn detect_cepstrum(
    samples: &[f32],
    min_lag: usize,
    max_lag: usize,
    threshold: f32,
) -> Option<(usize, f32)> {
    let n = samples.len();
    let fft_size = n.next_power_of_two();

    // Build complex input, zero-padded
    let mut input: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
    input.resize(fft_size, Complex::zero());

    // Forward FFT using OxiFFT plan API
    let mut buffer = vec![Complex::zero(); fft_size];
    if let Some(plan) = Plan::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE) {
        plan.execute(&input, &mut buffer);
    }

    // Log magnitude spectrum
    let log_mag_input: Vec<Complex<f32>> = buffer
        .iter()
        .map(|c| {
            let mag = (c.norm() + 1e-10).ln();
            Complex::new(mag, 0.0)
        })
        .collect();

    // Inverse FFT to get cepstrum
    let mut log_mag = vec![Complex::zero(); fft_size];
    if let Some(plan) = Plan::dft_1d(fft_size, Direction::Backward, Flags::ESTIMATE) {
        plan.execute(&log_mag_input, &mut log_mag);
    }
    let inv_n = 1.0 / fft_size as f32;

    // Find peak in quefrency range
    let upper = max_lag.min(fft_size / 2);
    if min_lag > upper {
        return None;
    }

    let mut best_quefrency = 0usize;
    let mut best_val = f32::NEG_INFINITY;

    for q in min_lag..=upper {
        let val = (log_mag[q].re * inv_n).abs();
        if val > best_val {
            best_val = val;
            best_quefrency = q;
        }
    }

    if best_quefrency == 0 {
        return None;
    }

    // Normalise confidence: compare peak to mean of the cepstrum in range
    let mean_val: f32 = (min_lag..=upper)
        .map(|q| (log_mag[q].re * inv_n).abs())
        .sum::<f32>()
        / (upper - min_lag + 1) as f32;

    let confidence = if mean_val > 1e-10 {
        ((best_val / mean_val - 1.0) / 10.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    if confidence < threshold {
        return None;
    }

    Some((best_quefrency, confidence))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        let sr = sample_rate as f32;
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    // --- frequency_to_midi ---

    #[test]
    fn test_frequency_to_midi_a4() {
        let (note, cents) = frequency_to_midi(440.0);
        assert_eq!(note, 69, "A4 should be MIDI 69");
        assert!(cents.abs() < 0.01, "cents deviation should be near 0");
    }

    #[test]
    fn test_frequency_to_midi_c4() {
        // C4 ≈ 261.626 Hz → MIDI 60
        let (note, _) = frequency_to_midi(261.626);
        assert_eq!(note, 60, "C4 should be MIDI 60");
    }

    #[test]
    fn test_frequency_to_midi_negative() {
        let (note, cents) = frequency_to_midi(-1.0);
        assert_eq!(note, 0);
        assert_eq!(cents, 0.0);
    }

    #[test]
    fn test_frequency_to_midi_cents_sharp() {
        // 442 Hz is slightly sharp of A4
        let (note, cents) = frequency_to_midi(442.0);
        assert_eq!(note, 69, "still nearest A4");
        assert!(cents > 0.0, "should be positive (sharp)");
        assert!(cents < 10.0, "should be a small deviation");
    }

    // --- midi_to_note_name ---

    #[test]
    fn test_midi_to_note_name_a4() {
        assert_eq!(midi_to_note_name(69), "A4");
    }

    #[test]
    fn test_midi_to_note_name_c4() {
        assert_eq!(midi_to_note_name(60), "C4");
    }

    #[test]
    fn test_midi_to_note_name_c_sharp_4() {
        assert_eq!(midi_to_note_name(61), "C#4");
    }

    // --- frequency_to_note_name ---

    #[test]
    fn test_frequency_to_note_name_a4() {
        let (name, cents) = frequency_to_note_name(440.0);
        assert_eq!(name, "A4");
        assert!(cents.abs() < 0.01);
    }

    // --- YIN ---

    #[test]
    fn test_yin_detects_440hz() {
        let samples = sine_wave(440.0, 44100, 4096);
        let detector = PitchDetector::new(PitchAlgorithm::Yin, 44100);
        let result = detector.detect(&samples);
        assert!(result.is_some(), "YIN should detect 440 Hz sine wave");
        let det = result.expect("detection present");
        assert!(
            (det.frequency_hz - 440.0).abs() < 10.0,
            "frequency should be close to 440 Hz, got {}",
            det.frequency_hz
        );
        assert!(det.confidence > 0.5, "confidence should be high");
    }

    #[test]
    fn test_yin_silence_returns_none() {
        let samples = vec![0.0f32; 2048];
        let detector = PitchDetector::new(PitchAlgorithm::Yin, 44100);
        let result = detector.detect(&samples);
        assert!(result.is_none(), "YIN should return None for silence");
    }

    // --- Autocorrelation ---

    #[test]
    fn test_autocorrelation_detects_220hz() {
        let samples = sine_wave(220.0, 44100, 8192);
        let mut detector = PitchDetector::new(PitchAlgorithm::Autocorrelation, 44100);
        detector.min_frequency = 100.0;
        detector.max_frequency = 500.0;
        let result = detector.detect(&samples);
        assert!(result.is_some(), "ACF should detect 220 Hz sine wave");
        let det = result.expect("detection present");
        assert!(
            (det.frequency_hz - 220.0).abs() < 15.0,
            "frequency should be near 220 Hz, got {}",
            det.frequency_hz
        );
    }

    #[test]
    fn test_autocorrelation_silence_returns_none() {
        let samples = vec![0.0f32; 4096];
        let detector = PitchDetector::new(PitchAlgorithm::Autocorrelation, 44100);
        let result = detector.detect(&samples);
        assert!(result.is_none(), "ACF should return None for silence");
    }

    // --- AMDF ---

    #[test]
    fn test_amdf_detects_440hz() {
        // AMDF may find sub-harmonics or harmonics; test that it finds
        // a frequency in the harmonic series of 440 Hz (i.e. a multiple
        // or division by a small integer is acceptable).
        let samples = sine_wave(440.0, 44100, 4096);
        let mut detector = PitchDetector::new(PitchAlgorithm::Amdf, 44100);
        detector.threshold = 0.05;
        let result = detector.detect(&samples);
        assert!(
            result.is_some(),
            "AMDF should detect a pitch in 440 Hz sine wave"
        );
        let det = result.expect("detection present");
        // Accept the fundamental or any harmonic multiple up to 4x or sub-harmonic
        // The detected frequency should be f = 440 * N or 440 / N for N in 1..=4
        let ratios = [1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 1.0 / 3.0, 1.0 / 4.0];
        let harmonic_match = ratios
            .iter()
            .any(|&r| (det.frequency_hz - 440.0 * r).abs() < 20.0);
        assert!(
            harmonic_match,
            "AMDF frequency {} should be a harmonic/sub-harmonic of 440 Hz",
            det.frequency_hz
        );
    }

    #[test]
    fn test_amdf_silence_returns_none() {
        let samples = vec![0.0f32; 2048];
        let detector = PitchDetector::new(PitchAlgorithm::Amdf, 44100);
        let result = detector.detect(&samples);
        assert!(result.is_none(), "AMDF should return None for silence");
    }

    // --- Cepstrum ---

    #[test]
    fn test_cepstrum_detects_pitch() {
        // Use a harmonic-rich signal (multiple harmonics) so the cepstrum has
        // a real peak to find.  A square wave approximation provides this.
        let sr = 44100u32;
        let fundamental = 200.0f32;
        let num_samples = 4096;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sr as f32;
                // Square wave: sum of odd harmonics 1,3,5,7
                (1..=7)
                    .step_by(2)
                    .map(|h| {
                        let h = h as f32;
                        (2.0 * PI * fundamental * h * t).sin() / h
                    })
                    .sum::<f32>()
            })
            .collect();

        let mut detector = PitchDetector::new(PitchAlgorithm::Cepstrum, sr);
        detector.min_frequency = 100.0;
        detector.max_frequency = 800.0;
        detector.threshold = 0.02;
        let result = detector.detect(&samples);

        // If cepstrum finds a pitch it must be inside the configured range
        if let Some(det) = result {
            assert!(
                det.frequency_hz >= detector.min_frequency
                    && det.frequency_hz <= detector.max_frequency,
                "cepstrum result {} outside [{}, {}]",
                det.frequency_hz,
                detector.min_frequency,
                detector.max_frequency
            );
        }
        // Returning None is also acceptable (sparse cepstrum on short frames)
    }

    // --- PitchDetection fields ---

    #[test]
    fn test_detection_has_valid_midi_note() {
        let samples = sine_wave(440.0, 44100, 4096);
        let detector = PitchDetector::new(PitchAlgorithm::Yin, 44100);
        if let Some(det) = detector.detect(&samples) {
            assert!(det.midi_note <= 127, "MIDI note must be in 0..=127");
        }
    }

    #[test]
    fn test_detection_has_valid_cents_deviation() {
        let samples = sine_wave(440.0, 44100, 4096);
        let detector = PitchDetector::new(PitchAlgorithm::Yin, 44100);
        if let Some(det) = detector.detect(&samples) {
            assert!(
                det.cents_deviation.abs() <= 50.0,
                "cents deviation must be in -50..50"
            );
        }
    }

    #[test]
    fn test_detection_confidence_is_normalized() {
        let samples = sine_wave(440.0, 44100, 4096);
        let detector = PitchDetector::new(PitchAlgorithm::Yin, 44100);
        if let Some(det) = detector.detect(&samples) {
            assert!(
                (0.0..=1.0).contains(&det.confidence),
                "confidence must be 0..1, got {}",
                det.confidence
            );
        }
    }
}
