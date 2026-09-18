//! Audio feature extraction for Music Information Retrieval.
//!
//! Provides spectral feature computation including centroid, rolloff,
//! flatness, and MFCC approximation from spectrum data.

#![allow(dead_code)]

use std::f32::consts::PI;

/// Spectral feature set computed from a magnitude spectrum.
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral centroid (Hz) — the "centre of gravity" of the spectrum.
    pub centroid: f32,
    /// Spectral spread — standard deviation around the centroid.
    pub spread: f32,
    /// Spectral skewness — asymmetry of the spectral distribution.
    pub skewness: f32,
    /// Spectral kurtosis — peakedness of the spectral distribution.
    pub kurtosis: f32,
    /// Spectral rolloff — frequency (Hz) below which 85 % of energy lies.
    pub rolloff_85: f32,
    /// Spectral flatness — geometric mean / arithmetic mean (closer to 1 = noise-like).
    pub flatness: f32,
}

impl SpectralFeatures {
    /// Returns `true` if the spectrum is "bright" (centroid above 4 kHz).
    #[must_use]
    pub fn is_bright(&self) -> bool {
        self.centroid > 4000.0
    }
}

/// Compute the spectral centroid in Hz from magnitude spectrum.
///
/// # Arguments
/// * `magnitudes`   – non-negative magnitude values (one per bin).
/// * `sample_rate`  – audio sample rate in Hz.
///
/// Returns 0.0 if the total energy is zero.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_spectral_centroid(magnitudes: &[f32], sample_rate: u32) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let n_bins = magnitudes.len();
    let hz_per_bin = sample_rate as f32 / (2.0 * n_bins as f32);
    let total: f32 = magnitudes.iter().sum();
    if total < 1e-12 {
        return 0.0;
    }
    magnitudes
        .iter()
        .enumerate()
        .map(|(i, &m)| i as f32 * hz_per_bin * m)
        .sum::<f32>()
        / total
}

/// Compute the spectral rolloff bin index.
///
/// Returns the first bin index where the cumulative power is ≥ `threshold`
/// fraction of the total power. `threshold` should be in `[0, 1]`.
///
/// Returns `magnitudes.len() - 1` if threshold is not reached (saturating).
#[must_use]
pub fn compute_spectral_rolloff(magnitudes: &[f32], threshold: f32) -> usize {
    if magnitudes.is_empty() {
        return 0;
    }
    let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
    if total_energy < 1e-12 {
        return 0;
    }
    let target = threshold.clamp(0.0, 1.0) * total_energy;
    let mut cumulative = 0.0_f32;
    for (i, &m) in magnitudes.iter().enumerate() {
        cumulative += m * m;
        if cumulative >= target {
            return i;
        }
    }
    magnitudes.len() - 1
}

/// Compute spectral flatness (Wiener entropy).
///
/// Defined as the ratio of the geometric mean to the arithmetic mean of the
/// magnitude spectrum. Values near 1.0 indicate noise-like content; values
/// near 0.0 indicate tonal content.
///
/// Returns 0.0 for an empty or all-zero spectrum.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_spectral_flatness(magnitudes: &[f32]) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let n = magnitudes.len() as f32;
    let arithmetic_mean: f32 = magnitudes.iter().sum::<f32>() / n;
    if arithmetic_mean < 1e-12 {
        return 0.0;
    }
    // Geometric mean via log-sum-exp for numerical stability
    let log_sum: f32 = magnitudes.iter().map(|&m| (m.max(1e-12)).ln()).sum::<f32>();
    let geometric_mean = (log_sum / n).exp();
    (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
}

/// MFCC coefficients computed from a magnitude spectrum.
#[derive(Debug, Clone)]
pub struct MfccCoeffs {
    /// The raw coefficient values.
    pub coeffs: Vec<f32>,
    /// Number of coefficients stored.
    pub num_coeffs: usize,
}

impl MfccCoeffs {
    /// Euclidean distance between two `MfccCoeffs` vectors.
    ///
    /// Returns `f32::MAX` if the two vectors have different lengths.
    #[must_use]
    pub fn distance(&self, other: &MfccCoeffs) -> f32 {
        if self.coeffs.len() != other.coeffs.len() {
            return f32::MAX;
        }
        self.coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }
}

/// Convert Hz to the mel scale.
#[allow(clippy::cast_precision_loss)]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Compute approximate MFCC coefficients from a magnitude spectrum.
///
/// The implementation uses a triangular mel filterbank followed by a
/// Type-II DCT approximation.
///
/// # Arguments
/// * `magnitudes`   – magnitude spectrum (one value per FFT bin).
/// * `sample_rate`  – audio sample rate in Hz.
/// * `n_coeffs`     – number of MFCC coefficients to return (typically 13).
///
/// # Returns
/// `MfccCoeffs` with `n_coeffs` values.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
pub fn compute_mfcc_from_spectrum(
    magnitudes: &[f32],
    sample_rate: u32,
    n_coeffs: usize,
) -> MfccCoeffs {
    if magnitudes.is_empty() || n_coeffs == 0 {
        return MfccCoeffs {
            coeffs: vec![0.0; n_coeffs],
            num_coeffs: n_coeffs,
        };
    }

    let n_filters: usize = (n_coeffs * 2).max(26);
    let n_bins = magnitudes.len();
    let sr = sample_rate as f32;

    let f_min = 0.0_f32;
    let f_max = sr / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Mel filterbank centre points (including edges)
    let mel_points: Vec<f32> = (0..=(n_filters + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_filters + 1) as f32)
        .collect();

    // Convert mel points to FFT bin indices
    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&m| {
            let hz = 700.0 * (10.0_f32.powf(m / 2595.0) - 1.0);
            let bin = (hz / sr * 2.0 * n_bins as f32).round() as usize;
            bin.min(n_bins - 1)
        })
        .collect();

    // Apply mel filterbank
    let mut mel_energies = vec![0.0_f32; n_filters];
    for k in 0..n_filters {
        let lo = bin_points[k];
        let cen = bin_points[k + 1];
        let hi = bin_points[k + 2];
        // Rising slope
        for b in lo..=cen {
            if cen > lo && b < n_bins {
                let w = (b - lo) as f32 / (cen - lo).max(1) as f32;
                mel_energies[k] += magnitudes[b] * w;
            }
        }
        // Falling slope
        for b in cen..=hi {
            if hi > cen && b < n_bins {
                let w = 1.0 - (b - cen) as f32 / (hi - cen).max(1) as f32;
                mel_energies[k] += magnitudes[b] * w;
            }
        }
        // Log compression
        mel_energies[k] = (mel_energies[k].max(1e-10)).ln();
    }

    // DCT-II approximation
    let mut coeffs = vec![0.0_f32; n_coeffs];
    for n in 0..n_coeffs {
        let mut sum = 0.0_f32;
        for (m, &e) in mel_energies.iter().enumerate() {
            sum += e * (PI * n as f32 * (2 * m + 1) as f32 / (2 * n_filters) as f32).cos();
        }
        coeffs[n] = sum;
    }

    MfccCoeffs {
        coeffs,
        num_coeffs: n_coeffs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SpectralFeatures ──────────────────────────────────────────────────────

    #[test]
    fn test_is_bright_above_4k() {
        let sf = SpectralFeatures {
            centroid: 5000.0,
            spread: 100.0,
            skewness: 0.0,
            kurtosis: 0.0,
            rolloff_85: 8000.0,
            flatness: 0.5,
        };
        assert!(sf.is_bright());
    }

    #[test]
    fn test_is_bright_below_4k() {
        let sf = SpectralFeatures {
            centroid: 2000.0,
            spread: 100.0,
            skewness: 0.0,
            kurtosis: 0.0,
            rolloff_85: 4000.0,
            flatness: 0.3,
        };
        assert!(!sf.is_bright());
    }

    #[test]
    fn test_is_bright_exactly_4k_is_not_bright() {
        let sf = SpectralFeatures {
            centroid: 4000.0,
            spread: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            rolloff_85: 0.0,
            flatness: 0.0,
        };
        assert!(!sf.is_bright());
    }

    // ── compute_spectral_centroid ─────────────────────────────────────────────

    #[test]
    fn test_centroid_empty() {
        assert!((compute_spectral_centroid(&[], 44100) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_centroid_all_zeros() {
        let mags = vec![0.0_f32; 512];
        assert!((compute_spectral_centroid(&mags, 44100) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_centroid_single_bin() {
        // Single non-zero bin at index 100, hz_per_bin = 44100 / (2*512) ≈ 43.07
        let mut mags = vec![0.0_f32; 512];
        mags[100] = 1.0;
        let centroid = compute_spectral_centroid(&mags, 44100);
        let expected = 100.0 * (44100.0 / (2.0 * 512.0));
        assert!((centroid - expected).abs() < 1.0);
    }

    #[test]
    fn test_centroid_symmetric_peak_is_middle() {
        // Equal energy at bins 50 and 150 → centroid at bin 100
        let mut mags = vec![0.0_f32; 512];
        mags[50] = 1.0;
        mags[150] = 1.0;
        let centroid = compute_spectral_centroid(&mags, 44100);
        let hz_per_bin = 44100.0 / (2.0 * 512.0);
        let expected_bin = 100.0 * hz_per_bin;
        assert!((centroid - expected_bin).abs() < 1.0);
    }

    // ── compute_spectral_rolloff ──────────────────────────────────────────────

    #[test]
    fn test_rolloff_empty() {
        assert_eq!(compute_spectral_rolloff(&[], 0.85), 0);
    }

    #[test]
    fn test_rolloff_all_energy_first_bin() {
        let mut mags = vec![0.0_f32; 10];
        mags[0] = 1.0;
        // 100 % at bin 0 → rolloff at bin 0 for any threshold ≤ 1.0
        assert_eq!(compute_spectral_rolloff(&mags, 0.85), 0);
    }

    #[test]
    fn test_rolloff_uniform_85_percent() {
        // Uniform spectrum: cumulative at bin k = (k+1)/N
        // 85 % → first bin where (k+1)/N >= 0.85 → k = ceil(0.85 * N) - 1
        let n = 20_usize;
        let mags = vec![1.0_f32; n];
        let idx = compute_spectral_rolloff(&mags, 0.85);
        // cumulative energy^2 = (idx+1) / n >= 0.85
        let expected = ((0.85 * n as f32).ceil() as usize).saturating_sub(1);
        assert_eq!(idx, expected);
    }

    #[test]
    fn test_rolloff_threshold_one_returns_last() {
        let mags = vec![1.0_f32; 10];
        let idx = compute_spectral_rolloff(&mags, 1.0);
        assert_eq!(idx, mags.len() - 1);
    }

    // ── compute_spectral_flatness ─────────────────────────────────────────────

    #[test]
    fn test_flatness_empty() {
        assert!((compute_spectral_flatness(&[]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_flatness_all_zeros() {
        let mags = vec![0.0_f32; 8];
        assert!((compute_spectral_flatness(&mags) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_flatness_uniform_is_one() {
        let mags = vec![1.0_f32; 16];
        let f = compute_spectral_flatness(&mags);
        assert!((f - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_flatness_single_spike_near_zero() {
        let mut mags = vec![0.0_f32; 16];
        mags[0] = 1.0;
        // Very low flatness (single tone)
        let f = compute_spectral_flatness(&mags);
        assert!(f < 0.5);
    }

    // ── MfccCoeffs::distance ─────────────────────────────────────────────────

    #[test]
    fn test_mfcc_distance_zero_self() {
        let m = MfccCoeffs {
            coeffs: vec![1.0, 2.0, 3.0],
            num_coeffs: 3,
        };
        assert!(m.distance(&m).abs() < 1e-5);
    }

    #[test]
    fn test_mfcc_distance_known() {
        let a = MfccCoeffs {
            coeffs: vec![0.0, 0.0],
            num_coeffs: 2,
        };
        let b = MfccCoeffs {
            coeffs: vec![3.0, 4.0],
            num_coeffs: 2,
        };
        assert!((a.distance(&b) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_mfcc_distance_different_len() {
        let a = MfccCoeffs {
            coeffs: vec![1.0, 2.0],
            num_coeffs: 2,
        };
        let b = MfccCoeffs {
            coeffs: vec![1.0],
            num_coeffs: 1,
        };
        assert_eq!(a.distance(&b), f32::MAX);
    }

    // ── compute_mfcc_from_spectrum ────────────────────────────────────────────

    #[test]
    fn test_mfcc_output_length() {
        let mags = vec![1.0_f32; 512];
        let m = compute_mfcc_from_spectrum(&mags, 44100, 13);
        assert_eq!(m.coeffs.len(), 13);
        assert_eq!(m.num_coeffs, 13);
    }

    #[test]
    fn test_mfcc_empty_input_returns_zeros() {
        let m = compute_mfcc_from_spectrum(&[], 44100, 13);
        assert_eq!(m.coeffs, vec![0.0_f32; 13]);
    }

    #[test]
    fn test_mfcc_zero_coeffs_returns_empty() {
        let mags = vec![1.0_f32; 64];
        let m = compute_mfcc_from_spectrum(&mags, 44100, 0);
        assert!(m.coeffs.is_empty());
    }

    #[test]
    fn test_mfcc_finite_output() {
        let mags: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin().abs()).collect();
        let m = compute_mfcc_from_spectrum(&mags, 44100, 13);
        for c in &m.coeffs {
            assert!(c.is_finite(), "coefficient is not finite: {c}");
        }
    }
}
