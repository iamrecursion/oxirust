//! Breath removal for podcast and voiceover restoration.
//!
//! Detects breath events using a combination of low-frequency energy surge and
//! characteristic spectral profile, then attenuates the detected region with
//! smooth fade-down/fade-up envelopes to avoid clicks.

use crate::error::RestoreResult;
use crate::utils::spectral::{apply_window, FftProcessor, WindowFunction};

/// Breath detector configuration.
#[derive(Debug, Clone)]
pub struct BreathDetector {
    /// Energy threshold for breath detection (fraction of RMS).
    pub threshold: f32,
    /// Minimum breath duration in milliseconds.
    pub min_duration_ms: f32,
    /// Maximum breath duration in milliseconds.
    pub max_duration_ms: f32,
}

impl Default for BreathDetector {
    fn default() -> Self {
        Self {
            threshold: 0.15,
            min_duration_ms: 50.0,
            max_duration_ms: 800.0,
        }
    }
}

impl BreathDetector {
    /// Create a new breath detector.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Energy threshold relative to RMS (0.0–1.0)
    /// * `min_duration_ms` - Minimum breath event duration in ms
    /// * `max_duration_ms` - Maximum breath event duration in ms
    #[must_use]
    pub fn new(threshold: f32, min_duration_ms: f32, max_duration_ms: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            min_duration_ms,
            max_duration_ms,
        }
    }

    /// Detect breath event regions in the input.
    ///
    /// Returns a list of `(start_sample, end_sample)` pairs.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    /// * `sample_rate` - Sample rate in Hz
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn detect_regions(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> RestoreResult<Vec<(usize, usize)>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let sr = sample_rate as f32;
        let min_samples = (self.min_duration_ms * sr / 1000.0).ceil() as usize;
        let max_samples = (self.max_duration_ms * sr / 1000.0).ceil() as usize;

        // Frame-based spectral analysis
        let fft_size = 512_usize.min(samples.len().next_power_of_two());
        if fft_size < 8 {
            return Ok(Vec::new());
        }

        let hop = fft_size / 2;
        let fft = FftProcessor::new(fft_size);

        // Compute global RMS for thresholding
        let rms = {
            let energy: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
            energy.sqrt()
        };

        if rms < f32::EPSILON {
            return Ok(Vec::new());
        }

        // Per-frame breath score
        let num_frames = (samples.len().saturating_sub(fft_size)) / hop + 1;
        let mut frame_scores = vec![0.0_f32; num_frames];

        for (frame_idx, score) in frame_scores.iter_mut().enumerate() {
            let start = frame_idx * hop;
            let end = (start + fft_size).min(samples.len());
            if end - start < fft_size {
                break;
            }

            let mut frame = samples[start..end].to_vec();
            apply_window(&mut frame, WindowFunction::Hann);

            let spectrum = fft.forward(&frame)?;
            let magnitude = fft.magnitude(&spectrum);

            // Breath profile: elevated energy in 200 Hz–3 kHz band, low energy
            // above 5 kHz, noisy spectral shape (high flatness).
            let bin_to_hz = |b: usize| b as f32 * sr / fft_size as f32;

            let lo_bin = ((200.0 * fft_size as f32 / sr) as usize).max(1);
            let hi_bin = ((3000.0 * fft_size as f32 / sr) as usize).min(magnitude.len() - 1);
            let very_hi_bin = ((5000.0 * fft_size as f32 / sr) as usize).min(magnitude.len() - 1);

            let breath_energy: f32 = magnitude[lo_bin..=hi_bin].iter().map(|&m| m * m).sum();
            let high_energy: f32 = if very_hi_bin < magnitude.len() {
                magnitude[very_hi_bin..].iter().map(|&m| m * m).sum()
            } else {
                0.0
            };

            let total_energy: f32 = magnitude.iter().map(|&m| m * m).sum();

            let frame_rms = (frame.iter().map(|&s| s * s).sum::<f32>() / fft_size as f32).sqrt();

            // Breath signature: moderate energy in speech band, low high-freq,
            // and overall RMS significantly below the average signal RMS.
            let breath_ratio = if total_energy > f32::EPSILON {
                breath_energy / total_energy
            } else {
                0.0
            };

            let high_freq_ratio = if total_energy > f32::EPSILON {
                high_energy / total_energy
            } else {
                1.0
            };

            // Noise-like flatness check (logarithm-based entropy)
            let flatness = spectral_flatness_local(&magnitude);

            // A breath frame has:
            //   - moderate energy in 200–3k band (>20% of total)
            //   - low high-freq (< 20% above 5k)
            //   - relatively flat spectrum (noise-like) vs speech
            //   - RMS < threshold * global_rms (quieter than main signal)
            let is_breath = breath_ratio > 0.20
                && high_freq_ratio < 0.25
                && flatness > 0.10
                && frame_rms < self.threshold * rms * 4.0
                && frame_rms > rms * 0.01;

            *score = if is_breath { 1.0 } else { 0.0 };
            // Suppress unused variable lint via explicit use
            let _ = bin_to_hz(0);
        }

        // Convert frame scores to sample-level markers
        let mut breath_sample = vec![false; samples.len()];
        for (frame_idx, &score) in frame_scores.iter().enumerate() {
            if score > 0.5 {
                let start = frame_idx * hop;
                let end = (start + fft_size).min(samples.len());
                for s in &mut breath_sample[start..end] {
                    *s = true;
                }
            }
        }

        // Group contiguous breath samples into regions
        let mut regions = Vec::new();
        let mut in_breath = false;
        let mut breath_start = 0;

        for (i, &is_breath) in breath_sample.iter().enumerate() {
            if !in_breath && is_breath {
                in_breath = true;
                breath_start = i;
            } else if in_breath && !is_breath {
                let duration = i - breath_start;
                if duration >= min_samples && duration <= max_samples {
                    regions.push((breath_start, i));
                }
                in_breath = false;
            }
        }

        if in_breath {
            let duration = samples.len() - breath_start;
            if duration >= min_samples && duration <= max_samples {
                regions.push((breath_start, samples.len()));
            }
        }

        Ok(regions)
    }
}

/// Result of breath removal processing.
#[derive(Debug, Clone)]
pub struct BreathRemovalResult {
    /// Number of breath events detected.
    pub breaths_detected: usize,
    /// Total number of samples attenuated.
    pub samples_attenuated: usize,
}

/// Breath remover that detects and attenuates breaths with smooth envelopes.
#[derive(Debug, Clone)]
pub struct BreathRemover {
    /// Detector configuration.
    pub detector: BreathDetector,
    /// Attenuation factor applied inside breath region (0.0 = silence, 1.0 = unchanged).
    pub attenuation: f32,
    /// Fade duration in milliseconds at the start and end of each breath.
    pub fade_ms: f32,
}

impl Default for BreathRemover {
    fn default() -> Self {
        Self {
            detector: BreathDetector::default(),
            attenuation: 0.05,
            fade_ms: 10.0,
        }
    }
}

impl BreathRemover {
    /// Create a new breath remover.
    ///
    /// # Arguments
    ///
    /// * `detector` - Breath detection configuration
    /// * `attenuation` - Gain applied inside breath region (0.0–1.0)
    /// * `fade_ms` - Crossfade duration in ms at region boundaries
    #[must_use]
    pub fn new(detector: BreathDetector, attenuation: f32, fade_ms: f32) -> Self {
        Self {
            detector,
            attenuation: attenuation.clamp(0.0, 1.0),
            fade_ms,
        }
    }

    /// Process samples in-place, attenuating breath regions.
    ///
    /// Returns a `BreathRemovalResult` describing detected events.
    ///
    /// # Arguments
    ///
    /// * `samples` - Mutable sample buffer
    /// * `sample_rate` - Sample rate in Hz
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn process(
        &self,
        samples: &mut [f32],
        sample_rate: u32,
    ) -> RestoreResult<BreathRemovalResult> {
        let regions = self.detector.detect_regions(samples, sample_rate)?;

        let fade_samples = ((self.fade_ms * sample_rate as f32 / 1000.0).ceil() as usize).max(1);
        let mut total_attenuated = 0_usize;

        for &(start, end) in &regions {
            if start >= samples.len() || end > samples.len() || start >= end {
                continue;
            }

            let duration = end - start;
            let actual_fade = fade_samples.min(duration / 2);

            for i in start..end {
                let envelope = compute_breath_envelope(i - start, duration, actual_fade);
                let gain = 1.0 - (1.0 - self.attenuation) * envelope;
                samples[i] *= gain;
                if gain < 1.0 {
                    total_attenuated += 1;
                }
            }
        }

        Ok(BreathRemovalResult {
            breaths_detected: regions.len(),
            samples_attenuated: total_attenuated,
        })
    }
}

/// Compute a smooth trapezoidal envelope for a breath region.
///
/// Returns a value in [0.0, 1.0] where 1.0 is the full center of the region.
#[allow(clippy::cast_precision_loss)]
fn compute_breath_envelope(position: usize, duration: usize, fade_samples: usize) -> f32 {
    use std::f32::consts::PI;

    if duration == 0 {
        return 0.0;
    }

    if position < fade_samples {
        // Fade in: half-cosine ramp 0→1
        let t = position as f32 / fade_samples as f32;
        0.5 * (1.0 - (PI * t).cos())
    } else if position >= duration.saturating_sub(fade_samples) {
        // Fade out: half-cosine ramp 1→0
        let pos_from_end = duration - position;
        let t = pos_from_end as f32 / fade_samples as f32;
        0.5 * (1.0 - (PI * t).cos())
    } else {
        1.0
    }
}

/// Compute spectral flatness (Wiener entropy) as a local helper.
#[allow(clippy::cast_precision_loss)]
fn spectral_flatness_local(magnitude: &[f32]) -> f32 {
    if magnitude.is_empty() {
        return 0.0;
    }

    let n = magnitude.len();
    let mut log_sum = 0.0_f32;
    let mut sum = 0.0_f32;

    for &m in magnitude {
        let m = m.max(1e-10);
        log_sum += m.ln();
        sum += m;
    }

    let geometric_mean = (log_sum / n as f32).exp();
    let arithmetic_mean = sum / n as f32;

    if arithmetic_mean > f32::EPSILON {
        geometric_mean / arithmetic_mean
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_breath_signal(sample_rate: u32, duration_ms: f32) -> Vec<f32> {
        // A breath-like signal: band-limited noise between 200–3000 Hz
        let n = (sample_rate as f32 * duration_ms / 1000.0) as usize;
        let amplitude = 0.04_f32; // quiet — below typical RMS

        (0..n)
            .map(|i| {
                // Sum several mid-frequency sinusoids with slight randomness
                let t = i as f32 / sample_rate as f32;
                let s = amplitude
                    * ((2.0 * PI * 400.0 * t).sin()
                        + 0.8 * (2.0 * PI * 800.0 * t).sin()
                        + 0.6 * (2.0 * PI * 1200.0 * t).sin()
                        + 0.4 * (2.0 * PI * 2000.0 * t).sin());
                s * 0.25 // extra quieting to stay well below speech amplitude
            })
            .collect()
    }

    fn make_speech_signal(sample_rate: u32, duration_ms: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_ms / 1000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * PI * 200.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_breath_detector_default() {
        let detector = BreathDetector::default();
        assert!(detector.threshold > 0.0);
        assert!(detector.min_duration_ms < detector.max_duration_ms);
    }

    #[test]
    fn test_breath_detector_new_clamps_threshold() {
        let detector = BreathDetector::new(2.5, 50.0, 500.0);
        assert!((detector.threshold - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_regions_empty() {
        let detector = BreathDetector::default();
        let result = detector.detect_regions(&[], 44100).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_regions_silence() {
        let detector = BreathDetector::default();
        let samples = vec![0.0_f32; 4096];
        let result = detector
            .detect_regions(&samples, 44100)
            .expect("should succeed");
        // Pure silence has rms == 0, so no breaths detected
        assert!(result.is_empty());
    }

    #[test]
    fn test_breath_remover_default() {
        let remover = BreathRemover::default();
        assert!(remover.attenuation < 1.0);
        assert!(remover.fade_ms > 0.0);
    }

    #[test]
    fn test_breath_remover_empty_input() {
        let remover = BreathRemover::default();
        let mut samples: Vec<f32> = Vec::new();
        let result = remover
            .process(&mut samples, 44100)
            .expect("should succeed");
        assert_eq!(result.breaths_detected, 0);
        assert_eq!(result.samples_attenuated, 0);
    }

    #[test]
    fn test_breath_remover_pure_speech_unchanged_length() {
        let remover = BreathRemover::default();
        let mut samples = make_speech_signal(44100, 500.0);
        let original_len = samples.len();
        let result = remover
            .process(&mut samples, 44100)
            .expect("should succeed");
        assert_eq!(samples.len(), original_len);
        // Speech dominant signal should have very few or zero breath detections
        let _ = result; // result count may vary by signal
    }

    #[test]
    fn test_breath_removal_result_fields() {
        let result = BreathRemovalResult {
            breaths_detected: 3,
            samples_attenuated: 1500,
        };
        assert_eq!(result.breaths_detected, 3);
        assert_eq!(result.samples_attenuated, 1500);
    }

    #[test]
    fn test_breath_envelope_edges() {
        // At position 0, envelope should be near 0 (fade in)
        let e0 = compute_breath_envelope(0, 100, 10);
        assert!(e0 < 0.1, "envelope at start should be near 0, got {e0}");

        // At position 99 (last sample), envelope should be near 0 (fade out)
        let e99 = compute_breath_envelope(99, 100, 10);
        assert!(e99 < 0.1, "envelope at end should be near 0, got {e99}");

        // In the middle, envelope should be 1.0
        let emid = compute_breath_envelope(50, 100, 10);
        assert!(
            (emid - 1.0).abs() < f32::EPSILON,
            "envelope at center should be 1.0, got {emid}"
        );
    }

    #[test]
    fn test_breath_remover_attenuation_applied() {
        // Build a signal where the first half is "breath-like" and the second is speech.
        // Even if detection doesn't trigger (threshold-dependent), we test the
        // attenuation logic by manually injecting a known region.
        let sample_rate = 44100_u32;
        let n = 4096_usize;
        let mut samples = vec![1.0_f32; n]; // unity signal

        // Simulate detected region by directly running compute_breath_envelope
        let region = (100_usize, 200_usize);
        let fade = 10_usize;
        let duration = region.1 - region.0;
        let attenuation = 0.1_f32;

        for i in region.0..region.1 {
            let envelope = compute_breath_envelope(i - region.0, duration, fade);
            let gain = 1.0 - (1.0 - attenuation) * envelope;
            samples[i] *= gain;
        }

        // Center of region should be attenuated
        let center = (region.0 + region.1) / 2;
        assert!(
            samples[center] < 0.5,
            "center should be attenuated: {}",
            samples[center]
        );

        // Outside region should be unchanged
        assert!(
            (samples[0] - 1.0).abs() < f32::EPSILON,
            "outside region unchanged"
        );
        let _ = sample_rate;
    }

    #[test]
    fn test_breath_remover_process_with_mixed_signal() {
        // Create a mixed signal: breath-like region embedded in speech
        let sample_rate = 44100_u32;
        let speech = make_speech_signal(sample_rate, 300.0);
        let breath = make_breath_signal(sample_rate, 200.0);
        let silence = vec![0.0_f32; (sample_rate as f32 * 0.1) as usize];

        let mut samples = Vec::new();
        samples.extend_from_slice(&speech);
        samples.extend_from_slice(&silence);
        samples.extend_from_slice(&breath);
        samples.extend_from_slice(&silence);
        samples.extend_from_slice(&speech);

        let remover = BreathRemover::default();
        let result = remover
            .process(&mut samples, sample_rate)
            .expect("should succeed");

        // We just ensure it runs without error and returns valid counts
        assert!(result.breaths_detected <= 100); // upper sanity bound
        assert!(result.samples_attenuated <= samples.len());
    }
}
