#![allow(dead_code)]
//! Psychoacoustic modeling for audio analysis.
//!
//! This module implements psychoacoustic models including:
//! - Bark and ERB critical band scales
//! - Equal-loudness contour approximation (ISO 226)
//! - Masking threshold estimation (simultaneous masking)
//! - Specific loudness computation (Zwicker model approximation)

// ── Critical-band scales ──────────────────────────────────────────────

/// Convert frequency in Hz to Bark scale.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn hz_to_bark(hz: f64) -> f64 {
    13.0 * (0.00076 * hz).atan() + 3.5 * ((hz / 7500.0).powi(2)).atan()
}

/// Convert Bark scale value back to Hz (iterative Newton approximation).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn bark_to_hz(bark: f64) -> f64 {
    // Newton iteration starting from a rough estimate
    let mut hz = bark * 100.0; // rough seed
    for _ in 0..20 {
        let b = hz_to_bark(hz);
        let db = 13.0 * 0.00076 / (1.0 + (0.00076 * hz).powi(2))
            + 3.5 * 2.0 * hz / (7500.0 * 7500.0) / (1.0 + (hz / 7500.0).powi(4));
        if db.abs() < 1e-15 {
            break;
        }
        hz -= (b - bark) / db;
        if hz < 0.0 {
            hz = 0.0;
        }
    }
    hz
}

/// Convert frequency in Hz to ERB (Equivalent Rectangular Bandwidth) rate.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn hz_to_erb_rate(hz: f64) -> f64 {
    21.4 * (0.00437 * hz + 1.0).log10()
}

/// Compute ERB bandwidth at a given frequency in Hz.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn erb_bandwidth(hz: f64) -> f64 {
    24.7 * (0.00437 * hz + 1.0)
}

// ── Equal-loudness contour (ISO 226 approximation) ────────────────────

/// Approximate A-weighting at a given frequency in Hz.
///
/// Returns the weighting in dB. This follows the standard IEC 61672-1 formula.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn a_weighting_db(hz: f64) -> f64 {
    if hz <= 0.0 {
        return -200.0;
    }
    let f2 = hz * hz;
    let f4 = f2 * f2;
    let num = 12194.0_f64.powi(2) * f4;
    let den = (f2 + 20.6_f64.powi(2))
        * ((f2 + 107.7_f64.powi(2)) * (f2 + 737.9_f64.powi(2))).sqrt()
        * (f2 + 12194.0_f64.powi(2));
    if den <= 0.0 {
        return -200.0;
    }
    20.0 * (num / den).log10() + 2.0
}

/// Approximate equal-loudness level at a given frequency for a given phon level.
///
/// Returns the SPL in dB that produces the given loudness in phon at that frequency.
/// This is a simplified approximation.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn equal_loudness_spl(hz: f64, phon: f64) -> f64 {
    // Simple approximation: at 1 kHz, phon = dB SPL.
    // Adjust using A-weighting as a rough proxy.
    let weight = a_weighting_db(hz);
    phon - weight
}

// ── Masking model ─────────────────────────────────────────────────────

/// Configuration for the masking model.
#[derive(Debug, Clone)]
pub struct MaskingConfig {
    /// FFT size for spectral analysis.
    pub fft_size: usize,
    /// Offset for tonality estimation (dB).
    pub tonal_offset_db: f64,
    /// Offset for noise masking (dB).
    pub noise_offset_db: f64,
}

impl Default for MaskingConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            tonal_offset_db: 14.5,
            noise_offset_db: 5.5,
        }
    }
}

/// A masking threshold at a specific frequency.
#[derive(Debug, Clone, Copy)]
pub struct MaskingThreshold {
    /// Center frequency in Hz.
    pub frequency_hz: f64,
    /// Masking threshold in dB SPL.
    pub threshold_db: f64,
    /// Whether the masker is tonal.
    pub is_tonal: bool,
}

/// Compute the spreading function for simultaneous masking.
///
/// `bark_diff` is the distance in Bark between masker and maskee.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn spreading_function(bark_diff: f64) -> f64 {
    // Schroeder spreading function approximation

    // Convert dB attenuation to linear, then back (keep in dB domain)
    if bark_diff < -1.0 {
        27.0 * bark_diff // steep lower slope
    } else if bark_diff <= 1.0 {
        -6.7 * bark_diff // gentler near masker / below 1 Bark
    } else {
        (-27.0 + 0.37 * 60.0) * bark_diff // upper slope level-dependent approx
    }
}

/// Estimate masking thresholds from a power spectrum.
///
/// `power_spectrum_db` should have `fft_size / 2 + 1` entries in dB.
/// `sample_rate` is in Hz.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn estimate_masking_thresholds(
    power_spectrum_db: &[f64],
    sample_rate: f64,
    config: &MaskingConfig,
) -> Vec<MaskingThreshold> {
    let num_bins = power_spectrum_db.len();
    let bin_hz = sample_rate / config.fft_size as f64;

    // Find local peaks as potential tonal maskers
    let mut thresholds = Vec::new();
    for i in 1..num_bins.saturating_sub(1) {
        let freq = i as f64 * bin_hz;
        if freq < 20.0 || freq > sample_rate / 2.0 {
            continue;
        }

        let is_peak = power_spectrum_db[i] > power_spectrum_db[i - 1]
            && power_spectrum_db[i] > power_spectrum_db[i + 1];

        let (threshold, is_tonal) = if is_peak && power_spectrum_db[i] > -60.0 {
            (power_spectrum_db[i] - config.tonal_offset_db, true)
        } else {
            (power_spectrum_db[i] - config.noise_offset_db, false)
        };

        thresholds.push(MaskingThreshold {
            frequency_hz: freq,
            threshold_db: threshold,
            is_tonal,
        });
    }
    thresholds
}

// ── Specific loudness (Zwicker approximation) ─────────────────────────

/// Result of specific loudness computation.
#[derive(Debug, Clone)]
pub struct LoudnessResult {
    /// Specific loudness per Bark band (sone/Bark).
    pub specific_loudness: Vec<f64>,
    /// Total loudness in sone.
    pub total_sone: f64,
    /// Total loudness in phon.
    pub total_phon: f64,
}

/// Compute total loudness from a power spectrum using a simplified Zwicker model.
///
/// `power_spectrum_db` should have `fft_size / 2 + 1` entries in dB SPL.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_loudness(
    power_spectrum_db: &[f64],
    sample_rate: f64,
    fft_size: usize,
) -> LoudnessResult {
    let num_bins = power_spectrum_db.len();
    let bin_hz = sample_rate / fft_size as f64;

    // Accumulate energy into Bark bands (0..24 Bark)
    let num_bands = 24_usize;
    let mut band_energy_db = vec![-100.0_f64; num_bands];

    #[allow(clippy::needless_range_loop)]
    for i in 0..num_bins {
        let freq = i as f64 * bin_hz;
        let bark = hz_to_bark(freq);
        let band = (bark.floor() as usize).min(num_bands - 1);
        // Energetic sum in linear domain
        let current_lin = 10.0_f64.powf(band_energy_db[band] / 10.0);
        let add_lin = 10.0_f64.powf(power_spectrum_db[i] / 10.0);
        band_energy_db[band] = 10.0 * (current_lin + add_lin).log10();
    }

    // Simplified specific loudness: power law
    let specific_loudness: Vec<f64> = band_energy_db
        .iter()
        .map(|&db| {
            if db > -60.0 {
                let lin = 10.0_f64.powf(db / 10.0);
                lin.powf(0.23) // Stevens' power law approximation
            } else {
                0.0
            }
        })
        .collect();

    let total_sone: f64 = specific_loudness.iter().sum();
    let total_phon = if total_sone > 0.0 {
        40.0 + 10.0 * (total_sone / 0.0625_f64).log2() // inverse of sone = 2^((phon-40)/10) * 0.0625
    } else {
        0.0
    };

    LoudnessResult {
        specific_loudness,
        total_sone,
        total_phon,
    }
}

/// Compute the absolute threshold of hearing (in quiet) at a given frequency.
///
/// Returns threshold in dB SPL. Uses the ISO 226:2003 approximation formula.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn threshold_in_quiet(hz: f64) -> f64 {
    if hz <= 0.0 {
        return 200.0;
    }
    let khz = hz / 1000.0;
    3.64 * khz.powf(-0.8) - 6.5 * (-0.6 * (khz - 3.3).powi(2)).exp() + 1e-3 * khz.powf(4.0)
}

/// Compute critical bandwidth at a given frequency in Hz.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn critical_bandwidth(hz: f64) -> f64 {
    25.0 + 75.0 * (1.0 + 1.4 * (hz / 1000.0).powi(2)).powf(0.69)
}

/// Determine if a signal bin is above the absolute threshold of hearing.
#[must_use]
pub fn is_audible(frequency_hz: f64, level_db_spl: f64) -> bool {
    level_db_spl > threshold_in_quiet(frequency_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_bark_zero() {
        assert!((hz_to_bark(0.0)).abs() < 0.01);
    }

    #[test]
    fn test_hz_to_bark_1000() {
        let bark = hz_to_bark(1000.0);
        // ~8.5 Bark at 1 kHz
        assert!(bark > 7.5 && bark < 9.5, "Bark at 1kHz = {bark}");
    }

    #[test]
    fn test_bark_to_hz_roundtrip() {
        for &freq in &[100.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let bark = hz_to_bark(freq);
            let back = bark_to_hz(bark);
            assert!(
                (freq - back).abs() < 5.0,
                "Roundtrip failed for {freq} Hz: got {back}"
            );
        }
    }

    #[test]
    fn test_hz_to_erb_rate() {
        let rate = hz_to_erb_rate(1000.0);
        // ~15.6 ERB rate at 1 kHz
        assert!(rate > 14.0 && rate < 17.0, "ERB rate at 1kHz = {rate}");
    }

    #[test]
    fn test_erb_bandwidth() {
        let bw = erb_bandwidth(1000.0);
        // ~132 Hz at 1 kHz
        assert!(bw > 100.0 && bw < 200.0, "ERB bw at 1kHz = {bw}");
    }

    #[test]
    fn test_a_weighting_1000hz() {
        let w = a_weighting_db(1000.0);
        // A-weighting at 1 kHz should be ~0 dB
        assert!(w.abs() < 1.0, "A-weight at 1kHz = {w}");
    }

    #[test]
    fn test_a_weighting_low_freq() {
        let w = a_weighting_db(50.0);
        // A-weighting strongly attenuates low freqs
        assert!(w < -20.0, "A-weight at 50Hz = {w}");
    }

    #[test]
    fn test_spreading_function_zero() {
        let s = spreading_function(0.0);
        assert!((s - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_spreading_function_symmetric_falloff() {
        let s_pos = spreading_function(2.0);
        let s_neg = spreading_function(-2.0);
        // Both should be negative (attenuation)
        assert!(s_pos < 0.0);
        assert!(s_neg < 0.0);
    }

    #[test]
    fn test_threshold_in_quiet_1000hz() {
        let t = threshold_in_quiet(1000.0);
        // Around 3-4 dB at 1 kHz
        assert!(t > -5.0 && t < 15.0, "TiQ at 1kHz = {t}");
    }

    #[test]
    fn test_threshold_in_quiet_100hz() {
        let t = threshold_in_quiet(100.0);
        // Should be much higher at 100 Hz
        assert!(t > 15.0, "TiQ at 100Hz = {t}");
    }

    #[test]
    fn test_critical_bandwidth_low() {
        let cb = critical_bandwidth(100.0);
        assert!(cb > 25.0 && cb < 200.0, "CB at 100Hz = {cb}");
    }

    #[test]
    fn test_is_audible() {
        // 1 kHz at 60 dB SPL should be audible
        assert!(is_audible(1000.0, 60.0));
        // 1 kHz at -50 dB SPL should not be audible
        assert!(!is_audible(1000.0, -50.0));
    }

    #[test]
    fn test_compute_loudness_basic() {
        let fft_size = 1024;
        let num_bins = fft_size / 2 + 1;
        // Moderate level spectrum
        let spectrum: Vec<f64> = vec![40.0; num_bins];
        let result = compute_loudness(&spectrum, 44100.0, fft_size);
        assert_eq!(result.specific_loudness.len(), 24);
        assert!(result.total_sone > 0.0);
        assert!(result.total_phon > 0.0);
    }

    #[test]
    fn test_compute_loudness_silence() {
        let fft_size = 512;
        let num_bins = fft_size / 2 + 1;
        let spectrum: Vec<f64> = vec![-100.0; num_bins];
        let result = compute_loudness(&spectrum, 44100.0, fft_size);
        assert!((result.total_sone).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_masking_thresholds() {
        let fft_size = 512;
        let num_bins = fft_size / 2 + 1;
        let mut spectrum = vec![-80.0_f64; num_bins];
        // Add a tonal peak
        spectrum[50] = 60.0;
        let config = MaskingConfig {
            fft_size,
            ..MaskingConfig::default()
        };
        let thresholds = estimate_masking_thresholds(&spectrum, 44100.0, &config);
        assert!(!thresholds.is_empty());
        // The peak should produce a tonal masker
        assert!(thresholds.iter().any(|t| t.is_tonal));
    }
}
