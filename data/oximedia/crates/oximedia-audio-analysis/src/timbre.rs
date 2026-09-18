//! Timbre analysis: MFCC coefficients, spectral centroid, brightness, roughness.
//!
//! Timbral features describe the "colour" or "texture" of a sound and are widely
//! used in music information retrieval and sound classification tasks.

#![allow(dead_code)]

use std::f32::consts::PI;

/// Number of Mel filter banks used internally.
const N_MEL_FILTERS: usize = 40;

/// Mel-frequency cepstral coefficients (MFCCs).
///
/// The first coefficient (`c0`) corresponds to the overall energy of the frame.
#[derive(Debug, Clone)]
pub struct Mfcc {
    /// Cepstral coefficients (typically 12–20 values).
    pub coefficients: Vec<f32>,
}

impl Mfcc {
    /// Creates a new [`Mfcc`] from a coefficient vector.
    #[must_use]
    pub fn new(coefficients: Vec<f32>) -> Self {
        Self { coefficients }
    }

    /// Returns the number of cepstral coefficients.
    #[must_use]
    pub fn num_coefficients(&self) -> usize {
        self.coefficients.len()
    }

    /// Euclidean distance to another MFCC vector.
    ///
    /// Both vectors are truncated to the shorter length before comparison.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let len = self.coefficients.len().min(other.coefficients.len());
        self.coefficients[..len]
            .iter()
            .zip(&other.coefficients[..len])
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Timbral feature set for a single analysis frame.
#[derive(Debug, Clone)]
pub struct TimbreFeatures {
    /// MFCC coefficients.
    pub mfcc: Mfcc,
    /// Spectral centroid in Hz.
    pub centroid_hz: f32,
    /// Spectral brightness: fraction of energy above a threshold (typically 1500 Hz).
    pub brightness: f32,
    /// Spectral roughness (measure of sensory dissonance).
    pub roughness: f32,
    /// Spectral spread (RMS bandwidth around the centroid).
    pub spread_hz: f32,
    /// Spectral roll-off frequency in Hz (frequency below which X% of energy lies).
    pub rolloff_hz: f32,
}

/// Converts a frequency in Hz to the Mel scale.
#[must_use]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts a Mel-scale value back to Hz.
#[must_use]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Builds a triangular Mel filter bank.
///
/// Returns a matrix of shape `[n_filters][n_fft_bins]` where each row is one
/// triangular filter.
///
/// # Arguments
/// * `n_filters` – Number of Mel filters.
/// * `n_fft` – FFT size (number of bins = `n_fft` / 2 + 1).
/// * `sample_rate` – Audio sample rate in Hz.
/// * `f_min` – Minimum frequency (Hz).
/// * `f_max` – Maximum frequency (Hz).
#[must_use]
pub fn mel_filter_bank(
    n_filters: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Equally-spaced Mel points (n_filters + 2 to include edges)
    let mel_points: Vec<f32> = (0..=n_filters + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_filters + 1) as f32)
        .collect();

    // Convert back to Hz, then to FFT bin indices
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&h| (n_fft + 1) as f32 * h / sample_rate)
        .collect();

    // Build triangular filters
    (0..n_filters)
        .map(|m| {
            let mut filter = vec![0.0f32; n_bins];
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for (k, f) in filter.iter_mut().enumerate() {
                let k_f = k as f32;
                if k_f >= left && k_f <= center && center > left {
                    *f = (k_f - left) / (center - left);
                } else if k_f >= center && k_f <= right && right > center {
                    *f = (right - k_f) / (right - center);
                }
            }
            filter
        })
        .collect()
}

/// Computes MFCCs from a power spectrum.
///
/// # Arguments
/// * `power_spectrum` – Magnitude or power values for each FFT bin.
/// * `n_coeffs` – Number of cepstral coefficients to return.
/// * `sample_rate` – Audio sample rate in Hz.
#[must_use]
pub fn compute_mfcc(power_spectrum: &[f32], n_coeffs: usize, sample_rate: f32) -> Mfcc {
    let n_fft = (power_spectrum.len() - 1) * 2;
    let filters = mel_filter_bank(N_MEL_FILTERS, n_fft, sample_rate, 0.0, sample_rate / 2.0);

    // Apply Mel filters and take log
    let mel_energies: Vec<f32> = filters
        .iter()
        .map(|filter| {
            let energy: f32 = filter
                .iter()
                .zip(power_spectrum)
                .map(|(&w, &p)| w * p)
                .sum();
            (energy + 1e-10).ln()
        })
        .collect();

    // DCT-II to get cepstral coefficients
    let n = mel_energies.len();
    let coeffs: Vec<f32> = (0..n_coeffs)
        .map(|k| {
            mel_energies
                .iter()
                .enumerate()
                .map(|(m, &e)| {
                    e * (PI * k as f32 * (2.0 * m as f32 + 1.0) / (2.0 * n as f32)).cos()
                })
                .sum::<f32>()
                * if k == 0 {
                    (1.0 / n as f32).sqrt()
                } else {
                    (2.0 / n as f32).sqrt()
                }
        })
        .collect();

    Mfcc::new(coeffs)
}

/// Computes the spectral centroid from a magnitude spectrum.
///
/// Returns 0.0 if the spectrum is silent.
#[must_use]
pub fn spectral_centroid(magnitudes: &[f32], sample_rate: f32) -> f32 {
    let n_bins = magnitudes.len();
    if n_bins == 0 {
        return 0.0;
    }
    let n_fft = (n_bins - 1) * 2;

    let (weighted_sum, total_mag): (f32, f32) = magnitudes
        .iter()
        .enumerate()
        .map(|(k, &m)| {
            let freq = k as f32 * sample_rate / n_fft as f32;
            (m * freq, m)
        })
        .fold((0.0, 0.0), |(ws, tm), (wf, mg)| (ws + wf, tm + mg));

    if total_mag < 1e-10 {
        0.0
    } else {
        weighted_sum / total_mag
    }
}

/// Computes spectral brightness: proportion of energy above `threshold_hz`.
#[must_use]
pub fn spectral_brightness(magnitudes: &[f32], sample_rate: f32, threshold_hz: f32) -> f32 {
    let n_bins = magnitudes.len();
    if n_bins == 0 {
        return 0.0;
    }
    let n_fft = (n_bins - 1) * 2;

    let (above, total): (f32, f32) = magnitudes
        .iter()
        .enumerate()
        .map(|(k, &m)| {
            let freq = k as f32 * sample_rate / n_fft as f32;
            (if freq >= threshold_hz { m } else { 0.0 }, m)
        })
        .fold((0.0, 0.0), |(a, t), (ab, mg)| (a + ab, t + mg));

    if total < 1e-10 {
        0.0
    } else {
        above / total
    }
}

/// Computes spectral spread (RMS bandwidth) around the centroid.
#[must_use]
pub fn spectral_spread(magnitudes: &[f32], sample_rate: f32) -> f32 {
    let centroid = spectral_centroid(magnitudes, sample_rate);
    let n_bins = magnitudes.len();
    if n_bins == 0 {
        return 0.0;
    }
    let n_fft = (n_bins - 1) * 2;

    let (weighted_var, total_mag): (f32, f32) = magnitudes
        .iter()
        .enumerate()
        .map(|(k, &m)| {
            let freq = k as f32 * sample_rate / n_fft as f32;
            (m * (freq - centroid).powi(2), m)
        })
        .fold((0.0, 0.0), |(wv, tm), (wvi, mg)| (wv + wvi, tm + mg));

    if total_mag < 1e-10 {
        0.0
    } else {
        (weighted_var / total_mag).sqrt()
    }
}

/// Computes the spectral roll-off frequency.
///
/// Returns the frequency below which `rolloff_frac` (e.g. 0.85) of the total
/// energy is concentrated.
#[must_use]
pub fn spectral_rolloff(magnitudes: &[f32], sample_rate: f32, rolloff_frac: f32) -> f32 {
    let n_bins = magnitudes.len();
    if n_bins == 0 {
        return 0.0;
    }
    let n_fft = (n_bins - 1) * 2;

    let total: f32 = magnitudes.iter().sum();
    if total < 1e-10 {
        return 0.0;
    }

    let threshold = total * rolloff_frac.clamp(0.0, 1.0);
    let mut cumsum = 0.0_f32;
    for (k, &m) in magnitudes.iter().enumerate() {
        cumsum += m;
        if cumsum >= threshold {
            return k as f32 * sample_rate / n_fft as f32;
        }
    }
    sample_rate / 2.0
}

/// Estimates spectral roughness as the normalised variance of the magnitude
/// spectrum (a proxy for sensory dissonance).
#[must_use]
pub fn spectral_roughness(magnitudes: &[f32]) -> f32 {
    let n = magnitudes.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f32 = magnitudes.iter().sum::<f32>() / n as f32;
    let variance: f32 = magnitudes.iter().map(|&m| (m - mean).powi(2)).sum::<f32>() / n as f32;
    if mean.abs() < 1e-10 {
        0.0
    } else {
        variance.sqrt() / mean
    }
}

/// Computes a complete set of timbral features from a magnitude spectrum.
#[must_use]
pub fn compute_timbre_features(
    magnitudes: &[f32],
    sample_rate: f32,
    n_mfcc: usize,
    brightness_threshold_hz: f32,
) -> TimbreFeatures {
    let mfcc = compute_mfcc(magnitudes, n_mfcc, sample_rate);
    let centroid_hz = spectral_centroid(magnitudes, sample_rate);
    let brightness = spectral_brightness(magnitudes, sample_rate, brightness_threshold_hz);
    let roughness = spectral_roughness(magnitudes);
    let spread_hz = spectral_spread(magnitudes, sample_rate);
    let rolloff_hz = spectral_rolloff(magnitudes, sample_rate, 0.85);

    TimbreFeatures {
        mfcc,
        centroid_hz,
        brightness,
        roughness,
        spread_hz,
        rolloff_hz,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat spectrum (white noise approximation).
    fn flat_spectrum(n_bins: usize) -> Vec<f32> {
        vec![1.0f32; n_bins]
    }

    /// Impulse spectrum: energy only in the lowest bin.
    fn low_impulse(n_bins: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n_bins];
        v[1] = 1.0;
        v
    }

    #[test]
    fn test_hz_to_mel_zero() {
        assert!((hz_to_mel(0.0) - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_mel_roundtrip() {
        let freq = 1000.0_f32;
        let mel = hz_to_mel(freq);
        let back = mel_to_hz(mel);
        assert!((back - freq).abs() < 0.01, "roundtrip error: {back}");
    }

    #[test]
    fn test_mel_filter_bank_shape() {
        let filters = mel_filter_bank(20, 1024, 44100.0, 0.0, 22050.0);
        assert_eq!(filters.len(), 20);
        assert_eq!(filters[0].len(), 513); // n_fft/2 + 1
    }

    #[test]
    fn test_mel_filter_bank_non_negative() {
        let filters = mel_filter_bank(20, 1024, 44100.0, 0.0, 22050.0);
        for row in &filters {
            for &v in row {
                assert!(v >= 0.0, "negative filter weight: {v}");
            }
        }
    }

    #[test]
    fn test_compute_mfcc_length() {
        let spec = flat_spectrum(513);
        let mfcc = compute_mfcc(&spec, 13, 44100.0);
        assert_eq!(mfcc.num_coefficients(), 13);
    }

    #[test]
    fn test_mfcc_distance_zero_self() {
        let spec = flat_spectrum(513);
        let mfcc = compute_mfcc(&spec, 13, 44100.0);
        let d = mfcc.distance(&mfcc);
        assert!(d < 1e-4, "distance to self should be ~0, got {d}");
    }

    #[test]
    fn test_mfcc_distance_positive_different() {
        let s1 = flat_spectrum(513);
        let mut s2 = flat_spectrum(513);
        s2[256] = 100.0; // very different spectrum
        let m1 = compute_mfcc(&s1, 13, 44100.0);
        let m2 = compute_mfcc(&s2, 13, 44100.0);
        let d = m1.distance(&m2);
        assert!(d > 0.0, "distance should be positive for different spectra");
    }

    #[test]
    fn test_spectral_centroid_flat_is_middle() {
        let n_bins = 513;
        let spec = flat_spectrum(n_bins);
        let centroid = spectral_centroid(&spec, 44100.0);
        // Centroid of flat spectrum should be near Nyquist / 2
        assert!(
            centroid > 5000.0 && centroid < 17000.0,
            "centroid = {centroid}"
        );
    }

    #[test]
    fn test_spectral_centroid_low_energy_is_low() {
        let n_bins = 513;
        let spec = low_impulse(n_bins);
        let centroid = spectral_centroid(&spec, 44100.0);
        assert!(centroid < 500.0, "low energy centroid too high: {centroid}");
    }

    #[test]
    fn test_spectral_brightness_flat_near_half() {
        let spec = flat_spectrum(513);
        let b = spectral_brightness(&spec, 44100.0, 11025.0);
        // Flat spectrum: roughly 50% above Nyquist/2
        assert!(b > 0.4 && b < 0.6, "brightness = {b}");
    }

    #[test]
    fn test_spectral_brightness_all_low_freq_is_zero() {
        let spec = low_impulse(513);
        let b = spectral_brightness(&spec, 44100.0, 5000.0);
        assert!(b < 0.01, "brightness should be ~0 for low impulse, got {b}");
    }

    #[test]
    fn test_spectral_spread_positive() {
        let spec = flat_spectrum(513);
        let spread = spectral_spread(&spec, 44100.0);
        assert!(spread > 0.0);
    }

    #[test]
    fn test_spectral_rolloff_within_range() {
        let spec = flat_spectrum(513);
        let rolloff = spectral_rolloff(&spec, 44100.0, 0.85);
        assert!(rolloff > 0.0 && rolloff <= 22050.0, "rolloff = {rolloff}");
    }

    #[test]
    fn test_spectral_roughness_flat_near_zero() {
        // Flat spectrum has zero variance → roughness = 0
        let spec = flat_spectrum(513);
        let r = spectral_roughness(&spec);
        assert!(r < 1e-5, "roughness of flat spectrum should be ~0, got {r}");
    }

    #[test]
    fn test_compute_timbre_features_compiles_and_runs() {
        let spec = flat_spectrum(513);
        let f = compute_timbre_features(&spec, 44100.0, 13, 1500.0);
        assert_eq!(f.mfcc.num_coefficients(), 13);
        assert!(f.centroid_hz >= 0.0);
        assert!(f.brightness >= 0.0 && f.brightness <= 1.0);
        assert!(f.roughness >= 0.0);
    }
}
