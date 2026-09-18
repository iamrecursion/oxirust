//! Chromagram and tonal centroid (tonnetz) computation.
//!
//! Provides top-level chroma feature extraction from power/magnitude spectra,
//! producing 12-bin pitch-class profiles (one bin per semitone of the equal-
//! temperament chromatic scale) and 6-dimensional tonal centroid vectors for
//! harmonic / key analysis.
//!
//! The mapping from FFT bins to pitch classes follows the equal-temperament
//! formula with A4 = 440 Hz as the reference pitch:
//!
//! ```text
//! pitch_class(freq) = round(12 * log2(freq / A4) + 9) mod 12
//! ```

#![forbid(unsafe_code)]
#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::f32::consts::PI;

/// Names of the 12 pitch classes in chromatic order, starting from C.
pub const CHROMA_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Number of pitch classes in the Western chromatic scale.
pub const NUM_CHROMA_BINS: usize = 12;

/// A4 reference frequency (Hz).
const A4_HZ: f32 = 440.0;
/// Minimum frequency considered for chroma mapping (A0 ≈ 27.5 Hz).
const MIN_FREQ_HZ: f32 = 27.5;
/// Maximum frequency considered for chroma mapping (C8 ≈ 4186 Hz).
const MAX_FREQ_HZ: f32 = 4186.0;

/// Compute a 12-bin chromagram from a magnitude (or power) spectrum.
///
/// Each FFT bin is mapped to one of the 12 pitch classes using the equal-
/// temperament formula.  Bins whose corresponding frequency falls outside
/// `[MIN_FREQ_HZ, MAX_FREQ_HZ]` are ignored.  The output vector is
/// L1-normalised so that its values sum to 1 (or remains all-zero if no bin
/// contributes).
///
/// # Arguments
/// * `spectrum`    – Magnitude spectrum.  Length should be `n_fft / 2 + 1`
///                   (the positive-frequency half of a real FFT of size `n_fft`).
/// * `sample_rate` – Audio sample rate in Hz.
/// * `n_fft`       – FFT size used to produce `spectrum` (must be > 0 and even).
///
/// # Returns
/// 12-element array with normalised pitch-class energies (index 0 = C).
#[must_use]
pub fn compute_chromagram(spectrum: &[f32], sample_rate: u32, n_fft: usize) -> [f32; 12] {
    let mut chroma = [0.0_f32; NUM_CHROMA_BINS];

    if spectrum.is_empty() || sample_rate == 0 || n_fft == 0 {
        return chroma;
    }

    let sr = sample_rate as f32;
    let n = n_fft;

    for (bin, &mag) in spectrum.iter().enumerate() {
        if mag <= 0.0 {
            continue;
        }
        // Frequency of this bin: freq = bin * sr / n_fft
        let freq = bin as f32 * sr / n as f32;
        if freq < MIN_FREQ_HZ || freq > MAX_FREQ_HZ {
            continue;
        }

        // Map to pitch class via equal temperament.
        // semitones from A4: st = 12 * log2(freq / A4)
        // A = pitch class 9, so pc = round(st) + 9  (mod 12)
        let semitones_from_a4 = 12.0 * (freq / A4_HZ).log2();
        let rounded = semitones_from_a4.round() as i32;
        let pc = ((rounded + 9).rem_euclid(12)) as usize;

        chroma[pc] += mag;
    }

    // L1-normalise
    let sum: f32 = chroma.iter().sum();
    if sum > 0.0 {
        for v in &mut chroma {
            *v /= sum;
        }
    }

    chroma
}

/// Normalization method for chromagram vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaNorm {
    /// No normalization (raw magnitudes).
    None,
    /// L1 normalization: values sum to 1.
    L1,
    /// L2 normalization: Euclidean norm equals 1.
    L2,
    /// Max normalization: maximum value equals 1.
    Max,
}

/// Normalize a 12-bin chromagram in place using the specified norm.
///
/// # Arguments
/// * `chroma` - Mutable reference to 12-bin chromagram array.
/// * `norm`   - Normalization method to apply.
///
/// If all values are zero or negative, the array is left unchanged.
pub fn normalize_chromagram(chroma: &mut [f32; 12], norm: ChromaNorm) {
    match norm {
        ChromaNorm::None => {}
        ChromaNorm::L1 => {
            let sum: f32 = chroma.iter().map(|&v| v.abs()).sum();
            if sum > f32::EPSILON {
                for v in chroma.iter_mut() {
                    *v /= sum;
                }
            }
        }
        ChromaNorm::L2 => {
            let sq_sum: f32 = chroma.iter().map(|&v| v * v).sum();
            let norm_val = sq_sum.sqrt();
            if norm_val > f32::EPSILON {
                for v in chroma.iter_mut() {
                    *v /= norm_val;
                }
            }
        }
        ChromaNorm::Max => {
            let max_val = chroma.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_val > f32::EPSILON {
                for v in chroma.iter_mut() {
                    *v /= max_val;
                }
            }
        }
    }
}

/// Compute a 12-bin chromagram with configurable normalization.
///
/// Like [`compute_chromagram`] but accepts a [`ChromaNorm`] parameter for
/// the normalization method instead of always using L1.
///
/// # Arguments
/// * `spectrum`    - Magnitude spectrum (length n_fft / 2 + 1).
/// * `sample_rate` - Audio sample rate in Hz.
/// * `n_fft`       - FFT size used to produce `spectrum`.
/// * `norm`        - Normalization method.
///
/// # Returns
/// 12-element array with pitch-class energies (index 0 = C).
#[must_use]
pub fn compute_chromagram_normalized(
    spectrum: &[f32],
    sample_rate: u32,
    n_fft: usize,
    norm: ChromaNorm,
) -> [f32; 12] {
    let mut chroma = [0.0_f32; NUM_CHROMA_BINS];

    if spectrum.is_empty() || sample_rate == 0 || n_fft == 0 {
        return chroma;
    }

    let sr = sample_rate as f32;
    let n = n_fft;

    for (bin, &mag) in spectrum.iter().enumerate() {
        if mag <= 0.0 {
            continue;
        }
        let freq = bin as f32 * sr / n as f32;
        if freq < MIN_FREQ_HZ || freq > MAX_FREQ_HZ {
            continue;
        }
        let semitones_from_a4 = 12.0 * (freq / A4_HZ).log2();
        let rounded = semitones_from_a4.round() as i32;
        let pc = ((rounded + 9).rem_euclid(12)) as usize;
        chroma[pc] += mag;
    }

    normalize_chromagram(&mut chroma, norm);
    chroma
}

/// Compute the 6-dimensional tonal centroid (tonnetz) from a 12-bin chromagram.
///
/// The tonnetz representation projects chroma features onto three circles that
/// encode the circle of fifths, minor thirds, and major thirds.  Each circle
/// contributes two dimensions (sin and cos), giving a 6D vector.
///
/// The projection uses the standard Harte et al. (2006) formula:
///
/// ```text
/// r₁ = 1.0  (fifths)
/// r₂ = 1.0  (minor thirds)
/// r₃ = 0.5  (major thirds)
///
/// dim 0: Σ chroma[k] * r₁ * sin(2π * 7k / 12)
/// dim 1: Σ chroma[k] * r₁ * cos(2π * 7k / 12)
/// dim 2: Σ chroma[k] * r₂ * sin(2π * 3k / 12)
/// dim 3: Σ chroma[k] * r₂ * cos(2π * 3k / 12)
/// dim 4: Σ chroma[k] * r₃ * sin(2π * 4k / 12)
/// dim 5: Σ chroma[k] * r₃ * cos(2π * 4k / 12)
/// ```
///
/// # Arguments
/// * `chroma` – 12-bin chromagram (should be normalised, but need not be).
///
/// # Returns
/// 6-element tonal centroid vector.
#[must_use]
pub fn tonal_centroid(chroma: &[f32; 12]) -> [f32; 6] {
    // Interval class step sizes (in semitones) for the three circles.
    // Circle of fifths: 7 semitones
    // Circle of minor thirds: 3 semitones
    // Circle of major thirds: 4 semitones
    const STEPS: [f32; 3] = [7.0, 3.0, 4.0];
    const RADII: [f32; 3] = [1.0, 1.0, 0.5];

    let mut centroid = [0.0_f32; 6];

    for (k, &c) in chroma.iter().enumerate() {
        if c == 0.0 {
            continue;
        }
        let kf = k as f32;
        for (circle, (&step, &radius)) in STEPS.iter().zip(RADII.iter()).enumerate() {
            let angle = 2.0 * PI * step * kf / 12.0;
            centroid[circle * 2] += c * radius * angle.sin();
            centroid[circle * 2 + 1] += c * radius * angle.cos();
        }
    }

    centroid
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a magnitude spectrum with energy only at the FFT bin nearest to
    /// `freq_hz` for a given FFT size and sample rate.
    fn single_freq_spectrum(freq_hz: f32, sample_rate: u32, n_fft: usize) -> Vec<f32> {
        let sr = sample_rate as f32;
        let bin = ((freq_hz / sr) * n_fft as f32).round() as usize;
        let bin = bin.min(n_fft / 2);
        let mut spectrum = vec![0.0_f32; n_fft / 2 + 1];
        spectrum[bin] = 1.0;
        spectrum
    }

    // ── CHROMA_NAMES ──────────────────────────────────────────────────────────

    #[test]
    fn test_chroma_names_count() {
        assert_eq!(CHROMA_NAMES.len(), 12);
    }

    #[test]
    fn test_chroma_names_first_last() {
        assert_eq!(CHROMA_NAMES[0], "C");
        assert_eq!(CHROMA_NAMES[11], "B");
        assert_eq!(CHROMA_NAMES[9], "A");
    }

    // ── compute_chromagram ────────────────────────────────────────────────────

    #[test]
    fn test_chromagram_empty_spectrum() {
        let chroma = compute_chromagram(&[], 44100, 2048);
        assert_eq!(chroma, [0.0_f32; 12]);
    }

    #[test]
    fn test_chromagram_zero_sample_rate() {
        let spectrum = vec![1.0_f32; 1025];
        let chroma = compute_chromagram(&spectrum, 0, 2048);
        assert_eq!(chroma, [0.0_f32; 12]);
    }

    #[test]
    fn test_chromagram_zero_n_fft() {
        let spectrum = vec![1.0_f32; 1025];
        let chroma = compute_chromagram(&spectrum, 44100, 0);
        assert_eq!(chroma, [0.0_f32; 12]);
    }

    #[test]
    fn test_chromagram_a4_maps_to_pitch_class_9() {
        // A4 (440 Hz) should map to pitch class 9 (A).
        let sr = 44100_u32;
        let n_fft = 4096_usize;
        let spectrum = single_freq_spectrum(440.0, sr, n_fft);
        let chroma = compute_chromagram(&spectrum, sr, n_fft);
        let dominant = chroma
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(
            dominant, 9,
            "A4 should map to pitch class 9 (A), got {dominant}"
        );
    }

    #[test]
    fn test_chromagram_a5_octave_invariance() {
        // A5 (880 Hz) should also map to pitch class 9 (A).
        let sr = 44100_u32;
        let n_fft = 4096_usize;
        let spectrum = single_freq_spectrum(880.0, sr, n_fft);
        let chroma = compute_chromagram(&spectrum, sr, n_fft);
        let dominant = chroma
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(
            dominant, 9,
            "A5 should also map to pitch class 9 (A), got {dominant}"
        );
    }

    #[test]
    fn test_chromagram_normalised_sums_to_one() {
        let sr = 44100_u32;
        let n_fft = 2048_usize;
        let spectrum = single_freq_spectrum(440.0, sr, n_fft);
        let chroma = compute_chromagram(&spectrum, sr, n_fft);
        let sum: f32 = chroma.iter().sum();
        // Either sums to 1.0 (if frequency is in range) or 0.0 (if not mapped)
        assert!(
            (sum - 1.0).abs() < 1e-5 || sum == 0.0,
            "Chromagram should sum to 1.0 or 0.0, got {sum}"
        );
    }

    #[test]
    fn test_chromagram_dc_and_subsonic_ignored() {
        // DC bin (0 Hz) and very low frequencies should not contribute.
        let sr = 44100_u32;
        let n_fft = 2048_usize;
        let mut spectrum = vec![0.0_f32; n_fft / 2 + 1];
        spectrum[0] = 100.0; // DC
        spectrum[1] = 100.0; // ~21 Hz — below MIN_FREQ_HZ of 27.5 Hz
        let chroma = compute_chromagram(&spectrum, sr, n_fft);
        let sum: f32 = chroma.iter().sum();
        assert!(
            sum == 0.0,
            "Sub-sonic bins should not contribute, sum={sum}"
        );
    }

    // ── tonal_centroid ────────────────────────────────────────────────────────

    #[test]
    fn test_tonal_centroid_length() {
        let chroma = [0.0_f32; 12];
        let tc = tonal_centroid(&chroma);
        assert_eq!(tc.len(), 6);
    }

    #[test]
    fn test_tonal_centroid_zero_chroma_is_all_zero() {
        let chroma = [0.0_f32; 12];
        let tc = tonal_centroid(&chroma);
        for &v in tc.iter() {
            assert!(v.abs() < f32::EPSILON, "Expected 0.0, got {v}");
        }
    }

    #[test]
    fn test_tonal_centroid_single_pitch_class_non_zero() {
        // With only C (pitch class 0), the centroid should have non-zero
        // components unless sin/cos happen to be exactly 0 (which they are not
        // for all three circles simultaneously).
        let mut chroma = [0.0_f32; 12];
        chroma[0] = 1.0; // C
        let tc = tonal_centroid(&chroma);
        let magnitude: f32 = tc.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(magnitude > 0.0, "Tonal centroid should be non-zero for C");
    }

    #[test]
    fn test_tonal_centroid_symmetry_chromatic_scale() {
        // Uniform chroma (all pitch classes equally weighted) should produce
        // a centroid close to the origin due to symmetry of the projection.
        let chroma = [1.0_f32 / 12.0; 12];
        let tc = tonal_centroid(&chroma);
        for &v in tc.iter() {
            assert!(
                v.abs() < 1e-5,
                "Uniform chroma tonal centroid should be near zero, got {v}"
            );
        }
    }

    #[test]
    fn test_chroma_names_contains_all_sharps() {
        assert!(CHROMA_NAMES.contains(&"C#"));
        assert!(CHROMA_NAMES.contains(&"D#"));
        assert!(CHROMA_NAMES.contains(&"F#"));
        assert!(CHROMA_NAMES.contains(&"G#"));
        assert!(CHROMA_NAMES.contains(&"A#"));
    }

    // ── Normalization tests ─────────────────────────────────────────────────

    #[test]
    fn test_normalize_l1_sums_to_one() {
        let mut chroma = [3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        normalize_chromagram(&mut chroma, ChromaNorm::L1);
        let sum: f32 = chroma.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "L1 norm should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_normalize_l2_unit_norm() {
        let mut chroma = [3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        normalize_chromagram(&mut chroma, ChromaNorm::L2);
        let norm: f32 = chroma.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2 norm should be 1.0, got {norm}"
        );
    }

    #[test]
    fn test_normalize_max_one() {
        let mut chroma = [3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        normalize_chromagram(&mut chroma, ChromaNorm::Max);
        let max_val = chroma.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_val - 1.0).abs() < 1e-5,
            "Max norm should set max to 1.0, got {max_val}"
        );
        // Original max was at index 7 (value 4.0)
        assert!(
            (chroma[0] - 0.75).abs() < 1e-5,
            "3/4 = 0.75, got {}",
            chroma[0]
        );
    }

    #[test]
    fn test_normalize_none_unchanged() {
        let original = [3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let mut chroma = original;
        normalize_chromagram(&mut chroma, ChromaNorm::None);
        for (a, b) in chroma.iter().zip(original.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_normalize_zero_array() {
        let mut chroma = [0.0_f32; 12];
        normalize_chromagram(&mut chroma, ChromaNorm::L1);
        assert_eq!(chroma, [0.0_f32; 12]);
        normalize_chromagram(&mut chroma, ChromaNorm::L2);
        assert_eq!(chroma, [0.0_f32; 12]);
        normalize_chromagram(&mut chroma, ChromaNorm::Max);
        assert_eq!(chroma, [0.0_f32; 12]);
    }

    #[test]
    fn test_compute_chromagram_normalized_l2() {
        let sr = 44100_u32;
        let n_fft = 4096_usize;
        let spectrum = single_freq_spectrum(440.0, sr, n_fft);
        let chroma = compute_chromagram_normalized(&spectrum, sr, n_fft, ChromaNorm::L2);
        let norm: f32 = chroma.iter().map(|&v| v * v).sum::<f32>().sqrt();
        // Should be 1.0 if there's any energy
        if norm > 0.0 {
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "L2-normalized chromagram norm should be 1.0, got {norm}"
            );
        }
    }

    #[test]
    fn test_compute_chromagram_normalized_max() {
        let sr = 44100_u32;
        let n_fft = 4096_usize;
        let spectrum = single_freq_spectrum(440.0, sr, n_fft);
        let chroma = compute_chromagram_normalized(&spectrum, sr, n_fft, ChromaNorm::Max);
        let max_val = chroma.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val > 0.0 {
            assert!(
                (max_val - 1.0).abs() < 1e-5,
                "Max-normalized max should be 1.0, got {max_val}"
            );
        }
    }

    #[test]
    fn test_l1_l2_max_preserve_relative_ordering() {
        let mut chroma_l1 = [5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut chroma_l2 = chroma_l1;
        let mut chroma_max = chroma_l1;
        normalize_chromagram(&mut chroma_l1, ChromaNorm::L1);
        normalize_chromagram(&mut chroma_l2, ChromaNorm::L2);
        normalize_chromagram(&mut chroma_max, ChromaNorm::Max);

        // Relative ordering should be preserved: bin[0] > bin[1] > bin[2]
        assert!(chroma_l1[0] > chroma_l1[1] && chroma_l1[1] > chroma_l1[2]);
        assert!(chroma_l2[0] > chroma_l2[1] && chroma_l2[1] > chroma_l2[2]);
        assert!(chroma_max[0] > chroma_max[1] && chroma_max[1] > chroma_max[2]);
    }
}
