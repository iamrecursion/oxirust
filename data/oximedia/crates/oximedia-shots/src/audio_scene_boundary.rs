//! Audio-based scene boundary detection for `oximedia-shots`.
//!
//! Detects scene changes in audio streams by analysing spectral flux, energy
//! transitions, and zero-crossing rate (ZCR) per analysis frame.  Peak-picking
//! with minimum-scene-length suppression converts a continuous flux curve into
//! discrete [`AudioSceneBoundary`] events, each carrying an
//! [`AudioChangeType`] label derived from energy and ZCR heuristics.

use std::f32::consts::PI;

// ── Public types ─────────────────────────────────────────────────────────────

/// A detected audio scene boundary with classification metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSceneBoundary {
    /// Index of the analysis frame where the boundary was detected.
    pub frame_index: u64,
    /// Estimated time of the boundary in milliseconds.
    pub time_ms: u64,
    /// Confidence score in `[0, 1]` derived from the normalised spectral flux peak.
    pub confidence: f32,
    /// Semantic classification of the audio change.
    pub audio_change_type: AudioChangeType,
}

/// Semantic label for the type of audio change at a scene boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioChangeType {
    /// Audio transitions from silence to speech.
    SilenceToSpeech,
    /// Audio transitions from speech to silence.
    SpeechToSilence,
    /// Audio transitions from music to speech.
    MusicToSpeech,
    /// Audio transitions from speech to music.
    SpeechToMusic,
    /// A large, sudden increase in overall energy.
    EnergySpike,
    /// A broad spectral shift without a clear silence/speech/music label.
    SpectralShift,
}

/// Configuration and state for the audio-based scene detector.
///
/// # Example
///
/// ```
/// use oximedia_shots::audio_scene_boundary::AudioSceneDetector;
///
/// let detector = AudioSceneDetector::new(48_000);
/// let sine: Vec<f32> = (0..48_000)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
///     .collect();
/// let boundaries = detector.detect_boundaries(&sine);
/// println!("{} boundaries found", boundaries.len());
/// ```
#[derive(Debug, Clone)]
pub struct AudioSceneDetector {
    /// Sample rate of the input audio in Hz.
    pub sample_rate: u32,
    /// Number of samples per analysis frame.
    pub frame_size: usize,
    /// Hop size between consecutive analysis frames (in samples).
    pub hop_size: usize,
    /// Silence threshold in dB (default `−40.0`).  Frames whose RMS energy is
    /// below this level are considered silent.
    pub silence_threshold_db: f32,
    /// Minimum normalised spectral flux (in `[0, 1]`) required for a peak to be
    /// recognised as a boundary (default `0.5`).
    pub change_threshold: f32,
    /// Minimum number of analysis frames that must separate consecutive
    /// boundary detections (default `8`).
    pub min_scene_frames: usize,
}

impl Default for AudioSceneDetector {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            frame_size: 1024,
            hop_size: 512,
            silence_threshold_db: -40.0,
            change_threshold: 0.5,
            min_scene_frames: 8,
        }
    }
}

impl AudioSceneDetector {
    /// Create a detector with sensible defaults for `sample_rate`.
    ///
    /// Frame size is set to roughly 23 ms (the standard for speech analysis).
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        // ~23 ms frame, ~11.5 ms hop — standard for speech/music analysis
        let frame_size = next_power_of_two((sample_rate as f32 * 0.023) as usize);
        let hop_size = frame_size / 2;
        Self {
            sample_rate,
            frame_size,
            hop_size,
            silence_threshold_db: -40.0,
            change_threshold: 0.5,
            min_scene_frames: 8,
        }
    }

    /// Detect scene boundaries in a block of mono PCM samples (f32, normalised
    /// to `[−1, 1]`).
    ///
    /// The algorithm:
    /// 1. Slice the signal into overlapping frames.
    /// 2. Compute a Hann-windowed DFT magnitude spectrum per frame.
    /// 3. Compute half-wave rectified spectral flux between consecutive frames.
    /// 4. Normalise flux to `[0, 1]` and peak-pick with `min_scene_frames`
    ///    suppression.
    /// 5. Classify each peak using energy and ZCR context.
    #[must_use]
    pub fn detect_boundaries(&self, samples: &[f32]) -> Vec<AudioSceneBoundary> {
        if samples.len() < self.frame_size {
            return Vec::new();
        }

        // ── Step 1: collect spectra for every hop-aligned frame ──────────────
        let num_frames = 1 + (samples.len().saturating_sub(self.frame_size)) / self.hop_size;
        let mut spectra: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        let mut energies: Vec<f32> = Vec::with_capacity(num_frames);
        let mut zcrs: Vec<f32> = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = (start + self.frame_size).min(samples.len());
            let frame = &samples[start..end];

            let spectrum = compute_spectrum(frame, self.frame_size);
            let energy = rms_energy(frame);
            let zcr = zero_crossing_rate(frame);

            spectra.push(spectrum);
            energies.push(energy);
            zcrs.push(zcr);
        }

        if spectra.len() < 2 {
            return Vec::new();
        }

        // ── Step 2: spectral flux curve ───────────────────────────────────────
        let mut flux: Vec<f32> = Vec::with_capacity(spectra.len());
        flux.push(0.0); // no previous frame for the first frame
        for t in 1..spectra.len() {
            flux.push(spectral_flux(&spectra[t - 1], &spectra[t]));
        }

        // ── Step 3: normalise flux to [0, 1] ─────────────────────────────────
        let max_flux = flux.iter().cloned().fold(0.0_f32, f32::max);
        let normalised: Vec<f32> = if max_flux > f32::EPSILON {
            flux.iter().map(|&f| f / max_flux).collect()
        } else {
            flux.clone()
        };

        // ── Step 4: peak-pick with suppression ────────────────────────────────
        let silence_linear = db_to_linear(self.silence_threshold_db);
        let mut boundaries = Vec::new();
        let mut last_boundary_frame: Option<usize> = None;

        for t in 1..normalised.len().saturating_sub(1) {
            // Local maximum check
            if normalised[t] <= normalised[t - 1] || normalised[t] <= normalised[t + 1] {
                continue;
            }
            // Threshold check
            if normalised[t] < self.change_threshold {
                continue;
            }
            // Min-scene-frames suppression
            if let Some(last) = last_boundary_frame {
                if t - last < self.min_scene_frames {
                    continue;
                }
            }

            // ── Step 5: classify ─────────────────────────────────────────────
            let prev_energy = energies[t - 1];
            let curr_energy = energies[t];
            let prev_zcr = zcrs[t - 1];
            let curr_zcr = zcrs[t];

            let change_type =
                classify_change(prev_energy, curr_energy, prev_zcr, curr_zcr, silence_linear);

            let time_ms = (t as u64 * self.hop_size as u64 * 1000) / self.sample_rate as u64;

            boundaries.push(AudioSceneBoundary {
                frame_index: t as u64,
                time_ms,
                confidence: normalised[t].clamp(0.0, 1.0),
                audio_change_type: change_type,
            });

            last_boundary_frame = Some(t);
        }

        boundaries
    }
}

// ── Free functions (public) ───────────────────────────────────────────────────

/// Compute the DFT magnitude spectrum of `samples` using a Hann window.
///
/// Only the first `frame_size / 2 + 1` (non-redundant) bins are returned.
/// The DFT is computed via a naïve O(N²) real-DFT suitable for moderate
/// `frame_size` values (≤ 4096).
#[must_use]
pub fn compute_spectrum(samples: &[f32], frame_size: usize) -> Vec<f32> {
    let n = frame_size.min(samples.len());
    if n == 0 {
        return Vec::new();
    }

    // Apply Hann window
    let windowed: Vec<f32> = samples[..n]
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos());
            s * w
        })
        .collect();

    let num_bins = n / 2 + 1;
    let mut magnitudes = vec![0.0_f32; num_bins];

    for k in 0..num_bins {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        let angle_step = 2.0 * PI * k as f32 / n as f32;
        for (i, &x) in windowed.iter().enumerate() {
            let angle = angle_step * i as f32;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        magnitudes[k] = (re * re + im * im).sqrt();
    }

    magnitudes
}

/// Compute the spectral centroid (in Hz) of a magnitude spectrum.
///
/// Returns `0.0` for an all-zero spectrum.
#[must_use]
pub fn spectral_centroid(spectrum: &[f32], sample_rate: u32) -> f32 {
    let num_bins = spectrum.len();
    if num_bins == 0 {
        return 0.0;
    }
    let bin_hz = sample_rate as f32 / (2.0 * (num_bins as f32 - 1.0).max(1.0));
    let (weighted, total) = spectrum
        .iter()
        .enumerate()
        .fold((0.0_f32, 0.0_f32), |(w, t), (k, &mag)| {
            (w + k as f32 * bin_hz * mag, t + mag)
        });
    if total < f32::EPSILON {
        0.0
    } else {
        weighted / total
    }
}

/// Compute the half-wave rectified spectral flux between two magnitude spectra.
///
/// `flux = Σ max(0, |curr[k]| − |prev[k]|)` over all bins.
#[must_use]
pub fn spectral_flux(prev: &[f32], curr: &[f32]) -> f32 {
    let len = prev.len().min(curr.len());
    (0..len).map(|k| (curr[k] - prev[k]).max(0.0)).sum()
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// RMS energy of a frame.
fn rms_energy(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let mean_sq: f32 = frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32;
    mean_sq.sqrt()
}

/// Zero-crossing rate (normalised to crossings per sample).
fn zero_crossing_rate(frame: &[f32]) -> f32 {
    if frame.len() < 2 {
        return 0.0;
    }
    let crossings = frame
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();
    crossings as f32 / (frame.len() - 1) as f32
}

/// Convert a dB value to a linear amplitude scale.
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Classify the type of audio change given energy and ZCR context.
///
/// Heuristics (tuned for 44.1/48 kHz, normalised audio):
/// - Speech ZCR is typically > 0.05 crossings/sample (voiced+unvoiced mix).
/// - Music with low ZCR + sustained energy → music.
/// - Silence threshold determined by `silence_linear`.
fn classify_change(
    prev_energy: f32,
    curr_energy: f32,
    prev_zcr: f32,
    curr_zcr: f32,
    silence_linear: f32,
) -> AudioChangeType {
    let prev_silent = prev_energy < silence_linear;
    let curr_silent = curr_energy < silence_linear;

    // High ZCR heuristic for speech (voiced + unvoiced phonemes)
    let zcr_speech_threshold = 0.05_f32;

    match (prev_silent, curr_silent) {
        (true, false) => AudioChangeType::SilenceToSpeech,
        (false, true) => AudioChangeType::SpeechToSilence,
        (false, false) => {
            let prev_speech = prev_zcr > zcr_speech_threshold;
            let curr_speech = curr_zcr > zcr_speech_threshold;
            match (prev_speech, curr_speech) {
                (false, true) => AudioChangeType::MusicToSpeech,
                (true, false) => AudioChangeType::SpeechToMusic,
                _ => {
                    // Both speech or both music — check energy spike
                    let ratio = if prev_energy > f32::EPSILON {
                        curr_energy / prev_energy
                    } else {
                        f32::MAX
                    };
                    if ratio > 4.0 {
                        AudioChangeType::EnergySpike
                    } else {
                        AudioChangeType::SpectralShift
                    }
                }
            }
        }
        (true, true) => AudioChangeType::SpectralShift,
    }
}

/// Return the smallest power of two ≥ `n` (minimum 64).
fn next_power_of_two(n: usize) -> usize {
    let mut p = 64_usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq_hz: f32, sample_rate: u32, num_samples: usize, amplitude: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn silence(num_samples: usize) -> Vec<f32> {
        vec![0.0_f32; num_samples]
    }

    // 1. Constructor sets reasonable frame_size for 44100 Hz
    #[test]
    fn test_new_sets_frame_size_44100() {
        let det = AudioSceneDetector::new(44_100);
        assert!(det.frame_size >= 64, "frame_size should be at least 64");
        assert!(
            det.frame_size.is_power_of_two(),
            "frame_size should be a power of two"
        );
    }

    // 2. Constructor sets reasonable frame_size for 48000 Hz
    #[test]
    fn test_new_sets_frame_size_48000() {
        let det = AudioSceneDetector::new(48_000);
        assert!(det.frame_size >= 64);
        assert!(det.frame_size.is_power_of_two());
    }

    // 3. Empty input returns no boundaries
    #[test]
    fn test_empty_input_no_boundaries() {
        let det = AudioSceneDetector::new(44_100);
        assert!(det.detect_boundaries(&[]).is_empty());
    }

    // 4. Pure silence returns no boundaries
    #[test]
    fn test_silence_no_boundaries() {
        let det = AudioSceneDetector::new(44_100);
        let samples = silence(44_100);
        let boundaries = det.detect_boundaries(&samples);
        // All frames are silent, flux is zero everywhere
        assert!(boundaries.is_empty());
    }

    // 5. compute_spectrum returns correct number of bins
    #[test]
    fn test_compute_spectrum_bin_count() {
        let frame: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let spectrum = compute_spectrum(&frame, 1024);
        assert_eq!(spectrum.len(), 513); // N/2 + 1
    }

    // 6. compute_spectrum of zeros is all zeros
    #[test]
    fn test_compute_spectrum_zeros() {
        let frame = vec![0.0_f32; 512];
        let spectrum = compute_spectrum(&frame, 512);
        for &bin in &spectrum {
            assert!(bin.abs() < 1e-5, "expected zero spectrum, got {bin}");
        }
    }

    // 7. spectral_flux is zero for identical spectra
    #[test]
    fn test_spectral_flux_identical() {
        let spectrum: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let flux = spectral_flux(&spectrum, &spectrum);
        assert!(
            flux.abs() < 1e-6,
            "flux of identical spectra should be 0, got {flux}"
        );
    }

    // 8. spectral_flux is positive when current has more energy
    #[test]
    fn test_spectral_flux_positive_when_louder() {
        let prev = vec![1.0_f32; 64];
        let curr = vec![2.0_f32; 64];
        let flux = spectral_flux(&prev, &curr);
        assert!(flux > 0.0, "flux should be positive when curr > prev");
    }

    // 9. spectral_centroid returns 0 for all-zero spectrum
    #[test]
    fn test_spectral_centroid_zeros() {
        let spectrum = vec![0.0_f32; 64];
        assert_eq!(spectral_centroid(&spectrum, 44_100), 0.0);
    }

    // 10. Silence→signal transition is detected and classified
    #[test]
    fn test_silence_to_signal_detected() {
        let sample_rate = 44_100_u32;
        let half = sample_rate as usize / 2;
        let mut samples = silence(half);
        // Second half: a high-ZCR-like signal (pseudo-speech: alternating polarity noise)
        for i in 0..half {
            samples.push(if i % 2 == 0 { 0.3 } else { -0.3 });
        }
        let det = AudioSceneDetector {
            sample_rate,
            frame_size: 1024,
            hop_size: 512,
            silence_threshold_db: -40.0,
            change_threshold: 0.3,
            min_scene_frames: 4,
        };
        let boundaries = det.detect_boundaries(&samples);
        assert!(
            !boundaries.is_empty(),
            "should detect at least one boundary at silence→signal transition"
        );
        // The first boundary should be SilenceToSpeech (high ZCR alternating polarity)
        assert_eq!(
            boundaries[0].audio_change_type,
            AudioChangeType::SilenceToSpeech
        );
    }
}
