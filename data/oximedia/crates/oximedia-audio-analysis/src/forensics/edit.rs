//! Edit detection (cuts, splices, insertions).

use std::f64::consts::PI;

use crate::spectral::SpectralAnalyzer;
use crate::{AnalysisConfig, Result};

/// Estimated splice probability at a single STFT frame boundary.
#[derive(Debug, Clone)]
pub struct SpliceProbability {
    /// Index of the frame (0-based) at which the phase discontinuity was measured.
    pub frame_idx: usize,
    /// Normalised confidence in the range \[0.0, 1.0\]; higher = more likely a splice.
    pub confidence: f64,
}

/// Detector for phase discontinuities indicating audio splices.
///
/// Uses the **second derivative of the instantaneous phase** (d²φ/dt²) to
/// identify splice points:
///
/// 1. Compute STFT frames with a Hann window.
/// 2. For each bin, compute the first-order phase difference between successive
///    frames (wrapped to (−π, π]) to obtain the instantaneous frequency estimate.
/// 3. Compute the second-order phase difference (change in instantaneous frequency)
///    between consecutive frame pairs.
/// 4. Average over magnitude-weighted bins; frames that exceed `threshold` are
///    returned as [`SpliceProbability`] candidates.
///
/// A clean, steady sinusoid has a near-constant instantaneous frequency so its
/// second derivative is ~0.  A hard splice introduces a sudden phase jump that
/// appears as a large second derivative spike.
///
/// # Reference
/// Phase-vocoder instantaneous-frequency estimation: Dolson, "The Phase Vocoder:
/// A Tutorial", CMJ 1986.
pub struct PhaseDiscontinuityDetector {
    /// Analysis window size (samples). Defaults to 2048.
    pub window_size: usize,
    /// Hop size (samples) between successive frames. Defaults to 512.
    pub hop_size: usize,
    /// Magnitude-weighted mean second-derivative threshold (radians) above which
    /// a frame is flagged as a potential splice. Defaults to 0.1.
    pub threshold: f64,
}

impl Default for PhaseDiscontinuityDetector {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_size: 512,
            threshold: 0.1,
        }
    }
}

impl PhaseDiscontinuityDetector {
    /// Create a new detector with the supplied parameters.
    #[must_use]
    pub fn new(window_size: usize, hop_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            hop_size,
            threshold,
        }
    }

    /// Analyse `samples` and return a list of [`SpliceProbability`] entries for
    /// frames whose magnitude-weighted mean second-derivative of instantaneous phase
    /// exceeds `threshold`.
    ///
    /// At least three STFT frames are required; shorter signals return an empty vec.
    #[must_use]
    pub fn detect_phase_discontinuities(&self, samples: &[f32]) -> Vec<SpliceProbability> {
        let n = self.window_size;

        if samples.len() < n * 3 {
            return Vec::new();
        }

        let phase_frames = self.compute_phase_frames(samples);
        if phase_frames.len() < 3 {
            return Vec::new();
        }

        let num_bins = n / 2 + 1;

        // Global peak magnitude for relative energy gating (10% floor).
        let global_peak: f64 = phase_frames
            .iter()
            .flat_map(|(_, mags)| mags.iter().copied())
            .fold(0.0_f64, f64::max);
        let mag_floor = (global_peak * 0.1).max(1e-6);

        let mut results = Vec::new();

        // Iterate over triples (prev, curr, next) — frame index i refers to curr.
        for i in 1..phase_frames.len().saturating_sub(1) {
            let (ph_prev, _) = &phase_frames[i - 1];
            let (ph_curr, mag_curr) = &phase_frames[i];
            let (ph_next, _) = &phase_frames[i + 1];

            let mut weighted_d2 = 0.0_f64;
            let mut total_weight = 0.0_f64;

            for k in 1..num_bins {
                let mag = mag_curr[k];
                if mag < mag_floor {
                    continue;
                }

                // First-order instantaneous frequency estimates (wrapped Δφ).
                let d1_prev = wrap_angle(ph_curr[k] - ph_prev[k]);
                let d1_curr = wrap_angle(ph_next[k] - ph_curr[k]);

                // Second derivative: change in instantaneous frequency.
                let d2 = wrap_angle(d1_curr - d1_prev).abs();

                weighted_d2 += d2 * mag;
                total_weight += mag;
            }

            if total_weight > 0.0 {
                let mean_d2 = weighted_d2 / total_weight;
                if mean_d2 > self.threshold {
                    // Normalise confidence: clamp [threshold, π] → [0, 1].
                    let confidence =
                        ((mean_d2 - self.threshold) / (PI - self.threshold)).clamp(0.0, 1.0);
                    results.push(SpliceProbability {
                        frame_idx: i,
                        confidence,
                    });
                }
            }
        }

        results
    }

    /// Compute windowed phase and magnitude spectra for every overlapping frame.
    fn compute_phase_frames(&self, samples: &[f32]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let n = self.window_size;
        let hop = self.hop_size;
        let num_bins = n / 2 + 1;
        let mut frames: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

        let mut pos = 0_usize;
        while pos + n <= samples.len() {
            let frame = &samples[pos..pos + n];
            let (phases, magnitudes) = compute_stft_phases_and_magnitudes(frame, n, num_bins);
            frames.push((phases, magnitudes));
            pos += hop;
        }

        frames
    }
}

/// Compute the phase and magnitude spectrum of a single windowed frame.
///
/// Returns `(phases, magnitudes)` each of length `num_bins`.
fn compute_stft_phases_and_magnitudes(
    frame: &[f32],
    n: usize,
    num_bins: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut phases = vec![0.0_f64; num_bins];
    let mut magnitudes = vec![0.0_f64; num_bins];

    for k in 0..num_bins {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (j, &s) in frame.iter().enumerate() {
            // Hann window.
            let w = 0.5 * (1.0 - (2.0 * PI * j as f64 / (n - 1) as f64).cos());
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            let sv = f64::from(s) * w;
            re += sv * angle.cos();
            im += sv * angle.sin();
        }
        magnitudes[k] = (re * re + im * im).sqrt();
        phases[k] = im.atan2(re);
    }

    (phases, magnitudes)
}

/// Wrap an angle to the (−π, π] interval.
#[inline]
fn wrap_angle(angle: f64) -> f64 {
    let mut a = angle;
    while a > PI {
        a -= 2.0 * PI;
    }
    while a <= -PI {
        a += 2.0 * PI;
    }
    a
}

/// Edit detector for detecting cuts and splices.
pub struct EditDetector {
    spectral_analyzer: SpectralAnalyzer,
    hop_size: usize,
    phase_detector: PhaseDiscontinuityDetector,
}

impl EditDetector {
    /// Create a new edit detector.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let hop_size = config.hop_size;
        let phase_detector = PhaseDiscontinuityDetector::new(
            config.fft_size,
            hop_size,
            0.1, // default threshold (second-derivative units)
        );
        Self {
            spectral_analyzer: SpectralAnalyzer::new(config),
            hop_size,
            phase_detector,
        }
    }

    /// Detect edits in audio.
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Result<EditResult> {
        // Look for discontinuities in:
        // 1. Amplitude envelope
        // 2. Spectral characteristics
        // 3. Phase continuity

        let edit_times = self.detect_discontinuities(samples, sample_rate)?;

        Ok(EditResult {
            num_edits: edit_times.len(),
            edit_times,
        })
    }

    /// Detect discontinuities indicating edits.
    fn detect_discontinuities(&self, samples: &[f32], sample_rate: f32) -> Result<Vec<f32>> {
        let window_size = 2048;
        let mut edits = Vec::new();

        if samples.len() < window_size * 2 {
            return Ok(edits);
        }

        // Compute spectral features over time
        let num_frames = (samples.len() - window_size) / self.hop_size;
        let mut spectral_centroids = Vec::new();
        let mut energies = Vec::new();

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = (start + window_size).min(samples.len());

            if end - start < window_size {
                break;
            }

            let frame = &samples[start..end];

            // Compute energy
            let energy: f32 = frame.iter().map(|&x| x * x).sum();
            energies.push(energy);

            // Compute spectral centroid
            let features = self.spectral_analyzer.analyze_frame(frame, sample_rate)?;
            spectral_centroids.push(features.centroid);
        }

        // Look for sudden changes
        let threshold = 3.0; // Standard deviations

        // Check energy discontinuities
        for i in 1..energies.len() {
            let diff = (energies[i] - energies[i - 1]).abs();
            let mean = (energies[i] + energies[i - 1]) / 2.0;

            if mean > 0.0 && diff / mean > threshold {
                let time = (i * self.hop_size) as f32 / sample_rate;
                edits.push(time);
            }
        }

        // Check spectral centroid discontinuities
        for i in 1..spectral_centroids.len() {
            let diff = (spectral_centroids[i] - spectral_centroids[i - 1]).abs();

            if diff > 500.0 {
                // Large change in spectral centroid
                let time = (i * self.hop_size) as f32 / sample_rate;
                if !edits.contains(&time) {
                    edits.push(time);
                }
            }
        }

        // Phase discontinuity analysis — additional signal for splice detection.
        let phase_candidates = self.phase_detector.detect_phase_discontinuities(samples);
        for candidate in &phase_candidates {
            let time = (candidate.frame_idx * self.hop_size) as f32 / sample_rate;
            // Only add high-confidence phase discontinuities (confidence ≥ 0.3).
            if candidate.confidence >= 0.3 && !edits.contains(&time) {
                edits.push(time);
            }
        }

        edits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Ok(edits)
    }
}

/// Edit detection result.
#[derive(Debug, Clone)]
pub struct EditResult {
    /// Number of detected edits
    pub num_edits: usize,
    /// Times of detected edits in seconds
    pub edit_times: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_detector() {
        let config = AnalysisConfig::default();
        let detector = EditDetector::new(config);

        // Generate signal with splice
        let sample_rate = 44100.0;
        let mut samples = Vec::new();

        // First part: 440 Hz
        for i in 0..22050 {
            samples.push((2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin() * 0.5);
        }

        // Second part: 880 Hz (sudden change)
        for i in 0..22050 {
            samples.push((2.0 * std::f32::consts::PI * 880.0 * i as f32 / sample_rate).sin() * 0.5);
        }

        let result = detector.detect(&samples, sample_rate);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Phase discontinuity tests (Item 1)
    // -----------------------------------------------------------------------

    /// A clean, phase-continuous sine wave should produce no phase-discontinuity
    /// detections because its instantaneous frequency is constant (second
    /// derivative of phase ≈ 0 everywhere).
    #[test]
    fn test_phase_continuity_clean_signal() {
        // Threshold 0.1 rad — well above numerical noise (~0.001) for a clean sine.
        let detector = PhaseDiscontinuityDetector::new(512, 128, 0.1);

        // Use 8000 Hz sample rate for shorter absolute sample count while keeping
        // 440 Hz well below the Nyquist frequency.
        let sample_rate = 8000.0_f32;
        let freq = 440.0_f32;
        let num_samples = 8000_usize; // 1 second at 8 kHz

        // Continuous sine — instantaneous frequency is constant → d²φ/dt² ≈ 0.
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin() * 0.8)
            .collect();

        let detections = detector.detect_phase_discontinuities(&samples);

        // A clean sinusoid should produce zero detections.
        assert!(
            detections.is_empty(),
            "Clean signal should produce 0 phase detections, got {}",
            detections.len()
        );
    }

    /// A signal constructed by hard-concatenating two same-frequency but
    /// out-of-phase sinusoids should trigger a phase-discontinuity detection
    /// near the splice point.
    #[test]
    fn test_phase_discontinuity_detected_at_splice() {
        let window_size = 512;
        let hop_size = 128;
        // Threshold 0.1 rad — very sensitive, phase π-jump gives ~1.5 rad d²φ.
        let detector = PhaseDiscontinuityDetector::new(window_size, hop_size, 0.1);

        let sample_rate = 44100.0_f32;
        let freq = 440.0_f32;
        let half = 4096_usize; // about 93 ms each

        // First segment: sine starting at phase 0.
        let mut samples: Vec<f32> = (0..half)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin() * 0.8)
            .collect();

        // Second segment: sine starting at phase π (fully out of phase).
        let phase_offset = std::f32::consts::PI;
        let second: Vec<f32> = (0..half)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate + phase_offset).sin()
                    * 0.8
            })
            .collect();
        samples.extend_from_slice(&second);

        let detections = detector.detect_phase_discontinuities(&samples);

        assert!(
            !detections.is_empty(),
            "Expected at least one phase discontinuity at the splice, got none"
        );

        // The highest-confidence detection should be within the expected region.
        // The splice is at frame ~ half / hop_size.
        let splice_frame = half / hop_size;
        let found_near_splice = detections
            .iter()
            .any(|d| d.frame_idx.abs_diff(splice_frame) <= 4);
        assert!(
            found_near_splice,
            "Expected a detection near frame {} (splice), detections at frames: {:?}",
            splice_frame,
            detections.iter().map(|d| d.frame_idx).collect::<Vec<_>>()
        );
    }
}
