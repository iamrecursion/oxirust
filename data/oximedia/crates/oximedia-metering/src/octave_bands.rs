//! Multi-band octave spectrum analysis.
//!
//! Provides ISO 1/3-octave and full-octave band definitions, SPL computation
//! from FFT magnitude data, and a simple equality comparison helper.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single octave or fractional-octave band.
#[derive(Debug, Clone, PartialEq)]
pub struct OctaveBand {
    /// Nominal centre frequency in Hz.
    pub center_hz: f32,
    /// Lower edge frequency in Hz.
    pub low_hz: f32,
    /// Upper edge frequency in Hz.
    pub high_hz: f32,
    /// Sound pressure level (or signal level) in dB for this band.
    pub spl_db: f32,
}

impl OctaveBand {
    /// Create a band with `spl_db = 0.0`.
    #[must_use]
    pub fn new(center_hz: f32, low_hz: f32, high_hz: f32) -> Self {
        Self {
            center_hz,
            low_hz,
            high_hz,
            spl_db: 0.0,
        }
    }

    /// Bandwidth of this band in Hz.
    #[must_use]
    pub fn bandwidth_hz(&self) -> f32 {
        self.high_hz - self.low_hz
    }
}

/// Analyser that builds octave-band sets and fills SPL values from FFT data.
pub struct OctaveAnalyzer;

impl OctaveAnalyzer {
    /// Return the 31 ISO 266 standard 1/3-octave bands covering 20 Hz – 20 kHz.
    ///
    /// Centre frequencies follow the preferred series (ISO 266:1997).
    /// All `spl_db` fields are initialised to `0.0`.
    #[must_use]
    pub fn third_octave_bands() -> Vec<OctaveBand> {
        // Factor between adjacent 1/3-octave centre frequencies: 2^(1/3).
        let factor: f32 = 2.0f32.powf(1.0 / 3.0);
        // ISO preferred centre frequencies (Hz) for 1/3-octave bands.
        let centres: [f32; 31] = [
            20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0,
            400.0, 500.0, 630.0, 800.0, 1_000.0, 1_250.0, 1_600.0, 2_000.0, 2_500.0, 3_150.0,
            4_000.0, 5_000.0, 6_300.0, 8_000.0, 10_000.0, 12_500.0, 16_000.0, 20_000.0,
        ];
        centres
            .iter()
            .map(|&c| OctaveBand::new(c, c / factor.sqrt(), c * factor.sqrt()))
            .collect()
    }

    /// Return the 10 standard full-octave bands covering 31.5 Hz – 16 kHz.
    ///
    /// Centre frequencies: 31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16 kHz.
    #[must_use]
    pub fn full_octave_bands() -> Vec<OctaveBand> {
        let factor: f32 = 2.0f32.sqrt(); // 2^(1/2) — half-octave boundary
        let centres: [f32; 10] = [
            31.5, 63.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0,
        ];
        centres
            .iter()
            .map(|&c| OctaveBand::new(c, c / factor, c * factor))
            .collect()
    }
}

/// Fill `spl_db` for each band by summing FFT magnitudes that fall within the
/// band's frequency range, then converting to dB.
///
/// * `bands` — mutable slice of bands to fill.
/// * `magnitudes` — linear FFT magnitudes (DC to Nyquist).
/// * `sample_rate` — sample rate of the source signal in Hz.
pub fn fill_spl(bands: &mut [OctaveBand], magnitudes: &[f32], sample_rate: u32) {
    let n_bins = magnitudes.len();
    if n_bins == 0 || sample_rate == 0 {
        return;
    }
    let nyquist = sample_rate as f32 / 2.0;
    let bin_width = nyquist / n_bins as f32;

    for band in bands.iter_mut() {
        let mut power: f32 = 0.0;
        for (bin, &mag) in magnitudes.iter().enumerate() {
            let bin_hz = bin as f32 * bin_width;
            if bin_hz >= band.low_hz && bin_hz < band.high_hz {
                power += mag * mag;
            }
        }
        band.spl_db = if power > 0.0 {
            10.0 * power.log10()
        } else {
            f32::NEG_INFINITY
        };
    }
}

/// Compute the mean absolute deviation in dB between two sets of octave bands.
///
/// Uses the shorter of the two slices.  Returns `0.0` when both slices are
/// empty.
pub struct SpectrumEquality;

impl SpectrumEquality {
    /// Mean absolute deviation in dB between `measured` and `target`.
    #[must_use]
    pub fn compare(measured: &[OctaveBand], target: &[OctaveBand]) -> f32 {
        let len = measured.len().min(target.len());
        if len == 0 {
            return 0.0;
        }
        let sum: f32 = measured
            .iter()
            .zip(target.iter())
            .take(len)
            .map(|(m, t)| (m.spl_db - t.spl_db).abs())
            .sum();
        sum / len as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- OctaveBand ----------

    #[test]
    fn test_octave_band_bandwidth() {
        let b = OctaveBand::new(1_000.0, 707.0, 1_414.0);
        assert!((b.bandwidth_hz() - 707.0).abs() < 1.0);
    }

    #[test]
    fn test_octave_band_default_spl_zero() {
        let b = OctaveBand::new(500.0, 354.0, 707.0);
        assert_eq!(b.spl_db, 0.0);
    }

    // ---------- OctaveAnalyzer::third_octave_bands ----------

    #[test]
    fn test_third_octave_band_count() {
        let bands = OctaveAnalyzer::third_octave_bands();
        assert_eq!(bands.len(), 31);
    }

    #[test]
    fn test_third_octave_first_band_center() {
        let bands = OctaveAnalyzer::third_octave_bands();
        assert!((bands[0].center_hz - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_third_octave_last_band_center() {
        let bands = OctaveAnalyzer::third_octave_bands();
        assert!((bands[30].center_hz - 20_000.0).abs() < 1.0);
    }

    #[test]
    fn test_third_octave_all_spl_zero_initially() {
        let bands = OctaveAnalyzer::third_octave_bands();
        for b in &bands {
            assert_eq!(b.spl_db, 0.0);
        }
    }

    #[test]
    fn test_third_octave_low_below_center() {
        let bands = OctaveAnalyzer::third_octave_bands();
        for b in &bands {
            assert!(b.low_hz < b.center_hz, "{} not < {}", b.low_hz, b.center_hz);
        }
    }

    // ---------- OctaveAnalyzer::full_octave_bands ----------

    #[test]
    fn test_full_octave_band_count() {
        let bands = OctaveAnalyzer::full_octave_bands();
        assert_eq!(bands.len(), 10);
    }

    #[test]
    fn test_full_octave_first_center() {
        let bands = OctaveAnalyzer::full_octave_bands();
        assert!((bands[0].center_hz - 31.5).abs() < 0.5);
    }

    #[test]
    fn test_full_octave_last_center() {
        let bands = OctaveAnalyzer::full_octave_bands();
        assert!((bands[9].center_hz - 16_000.0).abs() < 1.0);
    }

    // ---------- fill_spl ----------

    #[test]
    fn test_fill_spl_nonzero_magnitudes() {
        let mut bands = OctaveAnalyzer::full_octave_bands();
        let mags = vec![1.0f32; 1024];
        fill_spl(&mut bands, &mags, 48_000);
        // At least some bands should have finite SPL
        let finite_count = bands.iter().filter(|b| b.spl_db.is_finite()).count();
        assert!(finite_count > 0);
    }

    #[test]
    fn test_fill_spl_empty_magnitudes_no_change() {
        let mut bands = OctaveAnalyzer::third_octave_bands();
        fill_spl(&mut bands, &[], 48_000);
        // spl_db should remain 0.0 (no change since loop body never runs)
        for b in &bands {
            assert_eq!(b.spl_db, 0.0);
        }
    }

    #[test]
    fn test_fill_spl_zero_magnitudes_neg_inf() {
        let mut bands = OctaveAnalyzer::full_octave_bands();
        let mags = vec![0.0f32; 1024];
        fill_spl(&mut bands, &mags, 48_000);
        for b in &bands {
            assert!(b.spl_db.is_infinite() || b.spl_db == 0.0);
        }
    }

    // ---------- SpectrumEquality ----------

    #[test]
    fn test_spectrum_equality_identical_bands() {
        let bands = OctaveAnalyzer::full_octave_bands();
        let mad = SpectrumEquality::compare(&bands, &bands);
        assert!((mad - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectrum_equality_empty() {
        let mad = SpectrumEquality::compare(&[], &[]);
        assert!((mad - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectrum_equality_known_deviation() {
        let mut a = vec![OctaveBand::new(1_000.0, 700.0, 1_400.0)];
        let mut b = vec![OctaveBand::new(1_000.0, 700.0, 1_400.0)];
        a[0].spl_db = 10.0;
        b[0].spl_db = 5.0;
        let mad = SpectrumEquality::compare(&a, &b);
        assert!((mad - 5.0).abs() < 1e-5);
    }
}
