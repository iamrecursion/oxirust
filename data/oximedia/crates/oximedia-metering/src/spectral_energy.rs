//! Spectral energy balance metering: frequency-band energy analysis and
//! deviation measurement against a target balance curve.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A named frequency band defined by its low and high cutoff frequencies.
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    /// Human-readable label (e.g. "Sub-bass").
    pub name: String,
    /// Lower edge of the band in Hz.
    pub low_hz: f32,
    /// Upper edge of the band in Hz.
    pub high_hz: f32,
}

impl FrequencyBand {
    /// Create a new frequency band.
    #[must_use]
    pub fn new(name: impl Into<String>, low_hz: f32, high_hz: f32) -> Self {
        Self {
            name: name.into(),
            low_hz,
            high_hz,
        }
    }

    /// Sub-bass: 20 – 60 Hz.
    #[must_use]
    pub fn sub_bass() -> Self {
        Self::new("Sub-bass", 20.0, 60.0)
    }

    /// Bass: 60 – 250 Hz.
    #[must_use]
    pub fn bass() -> Self {
        Self::new("Bass", 60.0, 250.0)
    }

    /// Low-mid: 250 – 500 Hz.
    #[must_use]
    pub fn low_mid() -> Self {
        Self::new("Low-mid", 250.0, 500.0)
    }

    /// Mid: 500 – 2 000 Hz.
    #[must_use]
    pub fn mid() -> Self {
        Self::new("Mid", 500.0, 2_000.0)
    }

    /// High-mid: 2 000 – 4 000 Hz.
    #[must_use]
    pub fn high_mid() -> Self {
        Self::new("High-mid", 2_000.0, 4_000.0)
    }

    /// Presence: 4 000 – 6 000 Hz.
    #[must_use]
    pub fn presence() -> Self {
        Self::new("Presence", 4_000.0, 6_000.0)
    }

    /// Brilliance: 6 000 – 20 000 Hz.
    #[must_use]
    pub fn brilliance() -> Self {
        Self::new("Brilliance", 6_000.0, 20_000.0)
    }
}

/// Measured spectral balance: per-band energy values.
#[derive(Debug, Clone)]
pub struct SpectralBalance {
    /// Frequency bands used in the measurement.
    pub bands: Vec<FrequencyBand>,
    /// Energy for each band (linear, arbitrary units).
    pub energy: Vec<f32>,
}

impl SpectralBalance {
    /// Return the energy for the band whose name matches `name`, or `None`.
    #[must_use]
    pub fn energy_in_band(&self, name: &str) -> Option<f32> {
        self.bands
            .iter()
            .zip(self.energy.iter())
            .find_map(|(b, &e)| if b.name == name { Some(e) } else { None })
    }

    /// Return the name of the band with the highest energy, or `None` when
    /// there are no bands.
    #[must_use]
    pub fn dominant_band(&self) -> Option<&str> {
        if self.energy.is_empty() {
            return None;
        }
        let idx = self
            .energy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)?;
        Some(&self.bands[idx].name)
    }

    /// Return `true` when the combined energy of "Sub-bass" and "Bass" bands
    /// exceeds 50 % of the total energy.
    #[must_use]
    pub fn is_bass_heavy(&self) -> bool {
        let total: f32 = self.energy.iter().sum();
        if total == 0.0 {
            return false;
        }
        let bass_energy: f32 = self
            .bands
            .iter()
            .zip(self.energy.iter())
            .filter_map(|(b, &e)| {
                if b.name == "Sub-bass" || b.name == "Bass" {
                    Some(e)
                } else {
                    None
                }
            })
            .sum();
        bass_energy / total > 0.5
    }
}

/// Analyses FFT magnitude data and distributes energy into frequency bands.
pub struct SpectrumAnalyzer;

impl SpectrumAnalyzer {
    /// Distribute `magnitudes` (linear FFT bin magnitudes, DC-to-Nyquist) into
    /// the standard seven frequency bands and return a [`SpectralBalance`].
    ///
    /// * `magnitudes` — linear magnitude per FFT bin.
    /// * `sample_rate` — sample rate of the source audio in Hz.
    #[must_use]
    pub fn analyze(magnitudes: &[f32], sample_rate: u32) -> SpectralBalance {
        let bands = vec![
            FrequencyBand::sub_bass(),
            FrequencyBand::bass(),
            FrequencyBand::low_mid(),
            FrequencyBand::mid(),
            FrequencyBand::high_mid(),
            FrequencyBand::presence(),
            FrequencyBand::brilliance(),
        ];

        let n_bins = magnitudes.len();
        let nyquist = sample_rate as f32 / 2.0;

        let bin_to_hz = |bin: usize| bin as f32 * nyquist / n_bins as f32;

        let mut energy = vec![0.0f32; bands.len()];

        for (bin, &mag) in magnitudes.iter().enumerate() {
            let hz = bin_to_hz(bin);
            for (b_idx, band) in bands.iter().enumerate() {
                if hz >= band.low_hz && hz < band.high_hz {
                    energy[b_idx] += mag;
                    break;
                }
            }
        }

        SpectralBalance { bands, energy }
    }
}

/// A target spectral balance used for deviation measurement.
#[derive(Debug, Clone)]
pub struct SpectralBalanceTarget {
    /// Target energy per band (must match the band count of measured data).
    pub target_energy: Vec<f32>,
}

impl SpectralBalanceTarget {
    /// Create a target from a slice of desired energies.
    #[must_use]
    pub fn new(target_energy: Vec<f32>) -> Self {
        Self { target_energy }
    }

    /// Compute the per-band signed deviation between `measured` and this target.
    ///
    /// Returns `measured - target` for each band.  If the lengths differ the
    /// shorter length is used.
    #[must_use]
    pub fn deviation(&self, measured: &SpectralBalance) -> Vec<f32> {
        measured
            .energy
            .iter()
            .zip(self.target_energy.iter())
            .map(|(&m, &t)| m - t)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_bands() -> Vec<FrequencyBand> {
        vec![
            FrequencyBand::sub_bass(),
            FrequencyBand::bass(),
            FrequencyBand::low_mid(),
            FrequencyBand::mid(),
            FrequencyBand::high_mid(),
            FrequencyBand::presence(),
            FrequencyBand::brilliance(),
        ]
    }

    #[test]
    fn test_frequency_band_sub_bass_range() {
        let b = FrequencyBand::sub_bass();
        assert_eq!(b.low_hz, 20.0);
        assert_eq!(b.high_hz, 60.0);
        assert_eq!(b.name, "Sub-bass");
    }

    #[test]
    fn test_frequency_band_bass_range() {
        let b = FrequencyBand::bass();
        assert_eq!(b.low_hz, 60.0);
        assert_eq!(b.high_hz, 250.0);
    }

    #[test]
    fn test_frequency_band_brilliance_range() {
        let b = FrequencyBand::brilliance();
        assert_eq!(b.low_hz, 6_000.0);
        assert_eq!(b.high_hz, 20_000.0);
    }

    #[test]
    fn test_seven_standard_bands() {
        let bands = standard_bands();
        assert_eq!(bands.len(), 7);
    }

    #[test]
    fn test_energy_in_band_found() {
        let sb = SpectralBalance {
            bands: standard_bands(),
            energy: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };
        assert_eq!(sb.energy_in_band("Bass"), Some(2.0));
        assert_eq!(sb.energy_in_band("Mid"), Some(4.0));
    }

    #[test]
    fn test_energy_in_band_not_found() {
        let sb = SpectralBalance {
            bands: standard_bands(),
            energy: vec![1.0; 7],
        };
        assert!(sb.energy_in_band("Nonexistent").is_none());
    }

    #[test]
    fn test_dominant_band_single() {
        let bands = vec![FrequencyBand::bass(), FrequencyBand::mid()];
        let sb = SpectralBalance {
            bands,
            energy: vec![0.5, 3.0],
        };
        assert_eq!(sb.dominant_band(), Some("Mid"));
    }

    #[test]
    fn test_dominant_band_empty_returns_none() {
        let sb = SpectralBalance {
            bands: vec![],
            energy: vec![],
        };
        assert!(sb.dominant_band().is_none());
    }

    #[test]
    fn test_is_bass_heavy_true() {
        let bands = standard_bands();
        // sub-bass(1) + bass(9) = 10, total = 10 + 0*5 = 10 → 100 % bass
        let energy = vec![1.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sb = SpectralBalance { bands, energy };
        assert!(sb.is_bass_heavy());
    }

    #[test]
    fn test_is_bass_heavy_false() {
        let bands = standard_bands();
        let energy = vec![0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 5.0];
        let sb = SpectralBalance { bands, energy };
        assert!(!sb.is_bass_heavy());
    }

    #[test]
    fn test_spectrum_analyzer_returns_seven_bands() {
        let mags = vec![1.0f32; 1024];
        let sb = SpectrumAnalyzer::analyze(&mags, 44_100);
        assert_eq!(sb.bands.len(), 7);
        assert_eq!(sb.energy.len(), 7);
    }

    #[test]
    fn test_spectrum_analyzer_energy_is_non_negative() {
        let mags: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let sb = SpectrumAnalyzer::analyze(&mags, 48_000);
        for &e in &sb.energy {
            assert!(e >= 0.0);
        }
    }

    #[test]
    fn test_spectral_balance_target_deviation() {
        let bands = vec![FrequencyBand::bass(), FrequencyBand::mid()];
        let measured = SpectralBalance {
            bands,
            energy: vec![3.0, 5.0],
        };
        let target = SpectralBalanceTarget::new(vec![2.0, 4.0]);
        let dev = target.deviation(&measured);
        assert!((dev[0] - 1.0).abs() < 1e-6);
        assert!((dev[1] - 1.0).abs() < 1e-6);
    }
}
