#![allow(dead_code)]
//! Musical tuning detection for audio content.
//!
//! Detects the reference pitch (concert pitch) of audio recordings, determining
//! whether they are tuned to standard A440, or an alternative like A432, A442,
//! etc. Also detects equal-temperament vs. just-intonation characteristics and
//! overall pitch drift over time.

/// Standard concert pitch in Hz (A4).
pub const A440: f32 = 440.0;

/// Minimum detectable reference pitch in Hz.
const MIN_REF_PITCH: f32 = 415.0;

/// Maximum detectable reference pitch in Hz.
const MAX_REF_PITCH: f32 = 466.0;

/// Result of tuning detection.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningResult {
    /// Detected reference pitch for A4 in Hz.
    pub reference_pitch_hz: f32,
    /// Deviation from A440 in cents.
    pub deviation_cents: f32,
    /// Confidence of the detection (0.0 - 1.0).
    pub confidence: f32,
    /// Detected temperament type.
    pub temperament: Temperament,
    /// Per-frame pitch drift values in cents (relative to detected reference).
    pub drift_profile: Vec<f32>,
    /// Overall stability of the tuning (0.0 = very unstable, 1.0 = rock solid).
    pub stability: f32,
}

/// Temperament classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Temperament {
    /// Standard 12-tone equal temperament.
    EqualTemperament,
    /// Just intonation characteristics detected.
    JustIntonation,
    /// Pythagorean tuning characteristics.
    Pythagorean,
    /// Indeterminate / mixed.
    Unknown,
}

/// Configuration for tuning detection.
#[derive(Debug, Clone)]
pub struct TuningDetectConfig {
    /// FFT window size (must be power of 2).
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Minimum frequency to consider for pitch (Hz).
    pub min_freq: f32,
    /// Maximum frequency to consider for pitch (Hz).
    pub max_freq: f32,
    /// Number of harmonics to examine.
    pub num_harmonics: usize,
}

impl Default for TuningDetectConfig {
    fn default() -> Self {
        Self {
            window_size: 4096,
            hop_size: 1024,
            min_freq: 80.0,
            max_freq: 4000.0,
            num_harmonics: 6,
        }
    }
}

/// Calculates the deviation in cents between two frequencies.
#[must_use]
pub fn cents_between(freq_a: f32, freq_b: f32) -> f32 {
    if freq_a <= 0.0 || freq_b <= 0.0 {
        return 0.0;
    }
    1200.0 * (freq_b / freq_a).log2()
}

/// Snaps a frequency to the nearest semitone assuming a given reference A4.
#[must_use]
pub fn nearest_semitone(freq: f32, reference_a4: f32) -> f32 {
    if freq <= 0.0 || reference_a4 <= 0.0 {
        return 0.0;
    }
    let semitones_from_a4 = 12.0 * (freq / reference_a4).log2();
    let rounded = semitones_from_a4.round();
    reference_a4 * 2.0_f32.powf(rounded / 12.0)
}

/// Calculates the residual tuning error in cents for a set of detected peaks
/// against a given reference pitch.
#[must_use]
fn residual_error(peaks_hz: &[f32], reference_a4: f32) -> f32 {
    if peaks_hz.is_empty() || reference_a4 <= 0.0 {
        return f32::MAX;
    }
    let sum: f32 = peaks_hz
        .iter()
        .map(|&f| {
            let nearest = nearest_semitone(f, reference_a4);
            let c = cents_between(nearest, f);
            c * c
        })
        .sum();
    #[allow(clippy::cast_precision_loss)]
    let count = peaks_hz.len() as f32;
    (sum / count).sqrt()
}

/// Detect the most likely reference pitch from a set of spectral peaks.
#[must_use]
pub fn detect_reference_pitch(peaks_hz: &[f32]) -> (f32, f32) {
    if peaks_hz.is_empty() {
        return (A440, 0.0);
    }

    let steps = 200;
    let step_size = (MAX_REF_PITCH - MIN_REF_PITCH) / steps as f32;

    let mut best_ref = A440;
    let mut best_error = f32::MAX;

    for i in 0..=steps {
        #[allow(clippy::cast_precision_loss)]
        let candidate = MIN_REF_PITCH + step_size * i as f32;
        let err = residual_error(peaks_hz, candidate);
        if err < best_error {
            best_error = err;
            best_ref = candidate;
        }
    }

    let confidence = if best_error < 5.0 {
        1.0 - (best_error / 5.0).min(1.0)
    } else {
        0.0
    };

    (best_ref, confidence)
}

/// Estimate temperament from interval deviations.
#[must_use]
pub fn estimate_temperament(peaks_hz: &[f32], reference_a4: f32) -> Temperament {
    if peaks_hz.len() < 3 || reference_a4 <= 0.0 {
        return Temperament::Unknown;
    }

    // In equal temperament all intervals deviate similarly from just ratios.
    // We check the deviation of the major third (4 semitones = ratio 5/4 in JI).
    let just_major_third_ratio: f32 = 5.0 / 4.0;
    let et_major_third_cents: f32 = 400.0; // 4 semitones * 100 cents

    let mut ji_score: f32 = 0.0;
    let mut et_score: f32 = 0.0;
    let mut count = 0_u32;

    for i in 0..peaks_hz.len() {
        for j in (i + 1)..peaks_hz.len() {
            let ratio = peaks_hz[j] / peaks_hz[i];
            let interval_cents = cents_between(peaks_hz[i], peaks_hz[j]).abs();

            // Check if close to a major third
            let ji_cents = cents_between(1.0, just_major_third_ratio).abs();
            if (interval_cents - ji_cents).abs() < 30.0 {
                ji_score += 1.0;
                count += 1;
            } else if (interval_cents - et_major_third_cents).abs() < 30.0 {
                et_score += 1.0;
                count += 1;
            }

            let _ = ratio; // acknowledged
        }
    }

    if count == 0 {
        return Temperament::Unknown;
    }

    if ji_score > et_score {
        Temperament::JustIntonation
    } else if et_score > ji_score {
        Temperament::EqualTemperament
    } else {
        Temperament::Unknown
    }
}

/// Calculate pitch stability from a series of drift values (in cents).
#[must_use]
pub fn calculate_stability(drift_cents: &[f32]) -> f32 {
    if drift_cents.is_empty() {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = drift_cents.len() as f32;
    let mean = drift_cents.iter().sum::<f32>() / n;
    let variance = drift_cents
        .iter()
        .map(|&d| (d - mean) * (d - mean))
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();
    // Map std_dev to 0..1 range: 0 cents std_dev = 1.0 stability, 50+ cents = 0.0
    (1.0 - (std_dev / 50.0)).clamp(0.0, 1.0)
}

/// Simple peak picker: finds local maxima in a magnitude spectrum.
#[must_use]
pub fn pick_peaks(magnitudes: &[f32], threshold: f32) -> Vec<usize> {
    let mut peaks = Vec::new();
    if magnitudes.len() < 3 {
        return peaks;
    }
    for i in 1..magnitudes.len() - 1 {
        if magnitudes[i] > threshold
            && magnitudes[i] > magnitudes[i - 1]
            && magnitudes[i] > magnitudes[i + 1]
        {
            peaks.push(i);
        }
    }
    peaks
}

/// Convert bin index to frequency.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn bin_to_freq(bin: usize, sample_rate: f32, fft_size: usize) -> f32 {
    if fft_size == 0 {
        return 0.0;
    }
    bin as f32 * sample_rate / fft_size as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cents_between_octave() {
        let c = cents_between(220.0, 440.0);
        assert!(
            (c - 1200.0).abs() < 0.1,
            "Octave should be 1200 cents, got {c}"
        );
    }

    #[test]
    fn test_cents_between_unison() {
        let c = cents_between(440.0, 440.0);
        assert!(c.abs() < 0.01, "Unison should be 0 cents");
    }

    #[test]
    fn test_cents_between_zero() {
        assert!((cents_between(0.0, 440.0)).abs() < f32::EPSILON);
        assert!((cents_between(440.0, 0.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nearest_semitone_exact() {
        let f = nearest_semitone(440.0, 440.0);
        assert!((f - 440.0).abs() < 0.1);
    }

    #[test]
    fn test_nearest_semitone_slight_detune() {
        // 442 Hz should still snap to A4 = 440 (when ref is 440)
        let f = nearest_semitone(442.0, 440.0);
        assert!((f - 440.0).abs() < 1.0, "Should snap to 440, got {f}");
    }

    #[test]
    fn test_nearest_semitone_zero() {
        assert!((nearest_semitone(0.0, 440.0)).abs() < f32::EPSILON);
        assert!((nearest_semitone(440.0, 0.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_detect_reference_pitch_a440() {
        // Peaks that are exact A440 semitones (octaves + fifth)
        let peaks = vec![440.0, 880.0, 1760.0, 659.26];
        let (ref_pitch, confidence) = detect_reference_pitch(&peaks);
        assert!(
            (ref_pitch - 440.0).abs() < 3.0,
            "Should detect ~440 Hz, got {ref_pitch}"
        );
        assert!(
            confidence > 0.5,
            "Confidence should be high, got {confidence}"
        );
    }

    #[test]
    fn test_detect_reference_pitch_empty() {
        let (ref_pitch, _) = detect_reference_pitch(&[]);
        assert!((ref_pitch - A440).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_stability_perfect() {
        let drift = vec![0.0, 0.0, 0.0, 0.0];
        let stability = calculate_stability(&drift);
        assert!((stability - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_stability_unstable() {
        let drift = vec![-50.0, 50.0, -50.0, 50.0];
        let stability = calculate_stability(&drift);
        assert!(
            stability < 0.1,
            "Highly varying drift should have low stability, got {stability}"
        );
    }

    #[test]
    fn test_calculate_stability_empty() {
        assert!((calculate_stability(&[]) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pick_peaks() {
        let mags = vec![0.0, 0.5, 1.0, 0.5, 0.0, 0.3, 0.8, 0.2];
        let peaks = pick_peaks(&mags, 0.1);
        assert_eq!(peaks, vec![2, 6]);
    }

    #[test]
    fn test_pick_peaks_below_threshold() {
        let mags = vec![0.0, 0.05, 0.1, 0.05, 0.0];
        let peaks = pick_peaks(&mags, 0.5);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_bin_to_freq() {
        // Bin 1 at 44100 Hz with FFT size 44100 = 1.0 Hz
        let f = bin_to_freq(1, 44100.0, 44100);
        assert!((f - 1.0).abs() < f32::EPSILON);

        // Bin 0 should always be 0 Hz
        let f0 = bin_to_freq(0, 44100.0, 4096);
        assert!(f0.abs() < f32::EPSILON);
    }

    #[test]
    fn test_bin_to_freq_zero_fft() {
        assert!((bin_to_freq(5, 44100.0, 0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_estimate_temperament_unknown_small() {
        let peaks = vec![440.0];
        assert_eq!(estimate_temperament(&peaks, 440.0), Temperament::Unknown);
    }

    #[test]
    fn test_tuning_detect_config_default() {
        let cfg = TuningDetectConfig::default();
        assert_eq!(cfg.window_size, 4096);
        assert_eq!(cfg.hop_size, 1024);
        assert_eq!(cfg.num_harmonics, 6);
    }
}
