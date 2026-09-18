//! Spectral balance metering: frequency band energy, tilt measurement, warmth/brightness.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A frequency band with low/high cutoff frequencies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyBand {
    /// Lower cutoff frequency in Hz.
    pub low_hz: f64,
    /// Upper cutoff frequency in Hz.
    pub high_hz: f64,
    /// Human-readable label.
    pub label: &'static str,
}

impl FrequencyBand {
    /// Create a new frequency band.
    #[must_use]
    pub const fn new(low_hz: f64, high_hz: f64, label: &'static str) -> Self {
        Self {
            low_hz,
            high_hz,
            label,
        }
    }

    /// Return the geometric centre frequency of the band.
    #[must_use]
    pub fn centre_hz(&self) -> f64 {
        (self.low_hz * self.high_hz).sqrt()
    }
}

/// Standard octave bands used for spectral balance analysis.
pub static STANDARD_BANDS: &[FrequencyBand] = &[
    FrequencyBand::new(20.0, 80.0, "Sub"),
    FrequencyBand::new(80.0, 250.0, "Low"),
    FrequencyBand::new(250.0, 800.0, "Low-Mid"),
    FrequencyBand::new(800.0, 2500.0, "High-Mid"),
    FrequencyBand::new(2500.0, 8000.0, "Presence"),
    FrequencyBand::new(8000.0, 20000.0, "Air"),
];

/// Spectral balance snapshot: energy per band.
#[derive(Debug, Clone)]
pub struct SpectralBalance {
    /// Energy in dBFS for each frequency band.
    pub band_energy_db: Vec<f64>,
    /// The bands corresponding to each energy value.
    pub bands: Vec<FrequencyBand>,
}

impl SpectralBalance {
    /// Create a new snapshot from band energies.
    #[must_use]
    pub fn new(bands: Vec<FrequencyBand>, band_energy_db: Vec<f64>) -> Self {
        Self {
            band_energy_db,
            bands,
        }
    }

    /// Return the number of bands.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Compute spectral tilt: slope of a least-squares line fit to the band
    /// energies as a function of log-centre-frequency.
    /// Units: dB per octave (negative = high-frequency roll-off = warm, positive = bright).
    #[must_use]
    pub fn spectral_tilt_db_per_octave(&self) -> f64 {
        let n = self.bands.len();
        if n < 2 {
            return 0.0;
        }

        // X values: log2 of centre frequency (octaves)
        let xs: Vec<f64> = self.bands.iter().map(|b| b.centre_hz().log2()).collect();
        let ys: &[f64] = &self.band_energy_db;

        let n_f = n as f64;
        let sum_x: f64 = xs.iter().sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = xs.iter().map(|x| x * x).sum();

        let denom = n_f * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return 0.0;
        }

        (n_f * sum_xy - sum_x * sum_y) / denom
    }

    /// Warmth: average energy of bands below 500 Hz relative to overall average.
    /// Positive = warm (more low frequency energy).
    #[must_use]
    pub fn warmth_db(&self) -> f64 {
        let low_energies: Vec<f64> = self
            .bands
            .iter()
            .zip(self.band_energy_db.iter())
            .filter(|(b, _)| b.high_hz <= 500.0)
            .map(|(_, &e)| e)
            .collect();

        if low_energies.is_empty() || self.band_energy_db.is_empty() {
            return 0.0;
        }

        let low_avg = low_energies.iter().sum::<f64>() / low_energies.len() as f64;
        let overall_avg =
            self.band_energy_db.iter().sum::<f64>() / self.band_energy_db.len() as f64;
        low_avg - overall_avg
    }

    /// Brightness: average energy of bands above 5000 Hz relative to overall average.
    /// Positive = bright (more high frequency energy).
    #[must_use]
    pub fn brightness_db(&self) -> f64 {
        let high_energies: Vec<f64> = self
            .bands
            .iter()
            .zip(self.band_energy_db.iter())
            .filter(|(b, _)| b.low_hz >= 5000.0)
            .map(|(_, &e)| e)
            .collect();

        if high_energies.is_empty() || self.band_energy_db.is_empty() {
            return 0.0;
        }

        let high_avg = high_energies.iter().sum::<f64>() / high_energies.len() as f64;
        let overall_avg =
            self.band_energy_db.iter().sum::<f64>() / self.band_energy_db.len() as f64;
        high_avg - overall_avg
    }

    /// Return the index of the dominant band (highest energy).
    #[must_use]
    pub fn dominant_band_index(&self) -> Option<usize> {
        self.band_energy_db
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
}

/// Configuration for the spectral balance meter.
#[derive(Debug, Clone)]
pub struct SpectralBalanceMeterConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// FFT size.
    pub fft_size: usize,
    /// Smoothing decay factor [0, 1). Higher = slower response.
    pub smoothing: f64,
    /// Frequency bands to measure.
    pub bands: Vec<FrequencyBand>,
}

impl SpectralBalanceMeterConfig {
    /// Create a default config for 48 kHz stereo audio.
    #[must_use]
    pub fn default_48k() -> Self {
        Self {
            sample_rate: 48000.0,
            fft_size: 2048,
            smoothing: 0.85,
            bands: STANDARD_BANDS.to_vec(),
        }
    }
}

/// Spectral balance meter that measures energy in configurable frequency bands.
///
/// Uses a simple power-spectrum estimator built from the squared magnitude of
/// sample blocks (no FFT dependency – uses a single-pole band-pass approximation
/// via first-order IIR high- and low-pass filters).
pub struct SpectralBalanceMeter {
    config: SpectralBalanceMeterConfig,
    /// Per-band smoothed energy in dBFS.
    band_energy_db: Vec<f64>,
    /// Per-band, per-filter state for simple bandpass: [band][0=hp_state, 1=lp_state].
    filter_states: Vec<[f64; 2]>,
    /// Total samples processed.
    total_samples: usize,
}

impl SpectralBalanceMeter {
    /// Create a new meter.
    #[must_use]
    pub fn new(config: SpectralBalanceMeterConfig) -> Self {
        let n = config.bands.len();
        Self {
            band_energy_db: vec![-96.0; n],
            filter_states: vec![[0.0; 2]; n],
            total_samples: 0,
            config,
        }
    }

    /// Create a default 48 kHz meter.
    #[must_use]
    pub fn default_48k() -> Self {
        Self::new(SpectralBalanceMeterConfig::default_48k())
    }

    /// Process a mono sample block. For each band, estimates the band energy.
    pub fn process_mono(&mut self, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }

        let sr = self.config.sample_rate;
        let smoothing = self.config.smoothing;

        for (band_idx, band) in self.config.bands.iter().enumerate() {
            // Simple 1st-order IIR bandpass energy estimate:
            // High-pass the signal at band.low_hz, then low-pass at band.high_hz.
            let hp_alpha = Self::hp_alpha(band.low_hz, sr);
            let lp_alpha = Self::lp_alpha(band.high_hz, sr);

            let mut hp_state = self.filter_states[band_idx][0];
            let mut lp_state = self.filter_states[band_idx][1];
            let mut power_sum = 0.0_f64;

            for &s in samples {
                // First-order high-pass
                let hp_out = hp_alpha * (hp_state + s - lp_state);
                hp_state = hp_out;

                // First-order low-pass on the high-passed signal
                lp_state += lp_alpha * (hp_out - lp_state);

                power_sum += lp_state * lp_state;
            }

            self.filter_states[band_idx][0] = hp_state;
            self.filter_states[band_idx][1] = lp_state;

            let rms = (power_sum / samples.len() as f64).sqrt().max(1e-10);
            let instant_db = 20.0 * rms.log10();

            // Smooth
            self.band_energy_db[band_idx] =
                smoothing * self.band_energy_db[band_idx] + (1.0 - smoothing) * instant_db;
        }

        self.total_samples += samples.len();
    }

    /// Process interleaved stereo by summing channels to mono first.
    pub fn process_interleaved_stereo(&mut self, samples: &[f64]) {
        let mono: Vec<f64> = samples
            .chunks_exact(2)
            .map(|ch| (ch[0] + ch[1]) * 0.5)
            .collect();
        self.process_mono(&mono);
    }

    /// Return current spectral balance snapshot.
    #[must_use]
    pub fn snapshot(&self) -> SpectralBalance {
        SpectralBalance::new(self.config.bands.clone(), self.band_energy_db.clone())
    }

    /// Return energy in dBFS for a specific band index.
    #[must_use]
    pub fn band_energy_db(&self, band_idx: usize) -> Option<f64> {
        self.band_energy_db.get(band_idx).copied()
    }

    /// Total samples processed.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        for e in &mut self.band_energy_db {
            *e = -96.0;
        }
        for state in &mut self.filter_states {
            *state = [0.0; 2];
        }
        self.total_samples = 0;
    }

    /// Compute first-order high-pass alpha coefficient.
    fn hp_alpha(cutoff_hz: f64, sample_rate: f64) -> f64 {
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        rc / (rc + dt)
    }

    /// Compute first-order low-pass alpha coefficient.
    fn lp_alpha(cutoff_hz: f64, sample_rate: f64) -> f64 {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
        dt / (rc + dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_signal(amp: f64, len: usize) -> Vec<f64> {
        // Generate a 1 kHz sine wave at the given amplitude, which sits in the Low-Mid band
        let sr = 48000.0_f64;
        let freq = 1000.0_f64;
        (0..len)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_frequency_band_centre() {
        let band = FrequencyBand::new(100.0, 400.0, "Test");
        let centre = band.centre_hz();
        assert!((centre - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_frequency_band_label() {
        let band = FrequencyBand::new(20.0, 80.0, "Sub");
        assert_eq!(band.label, "Sub");
    }

    #[test]
    fn test_spectral_balance_band_count() {
        let balance =
            SpectralBalance::new(STANDARD_BANDS.to_vec(), vec![-20.0; STANDARD_BANDS.len()]);
        assert_eq!(balance.band_count(), STANDARD_BANDS.len());
    }

    #[test]
    fn test_spectral_balance_dominant_band() {
        let energies = vec![-30.0, -20.0, -10.0, -25.0, -35.0, -40.0];
        let balance = SpectralBalance::new(STANDARD_BANDS.to_vec(), energies);
        assert_eq!(balance.dominant_band_index(), Some(2));
    }

    #[test]
    fn test_spectral_balance_dominant_band_empty() {
        let balance = SpectralBalance::new(vec![], vec![]);
        assert_eq!(balance.dominant_band_index(), None);
    }

    #[test]
    fn test_spectral_balance_warmth_positive() {
        // Higher low-freq energy
        let energies = vec![-10.0, -10.0, -30.0, -30.0, -30.0, -30.0];
        let balance = SpectralBalance::new(STANDARD_BANDS.to_vec(), energies);
        assert!(balance.warmth_db() > 0.0, "warmth={}", balance.warmth_db());
    }

    #[test]
    fn test_spectral_balance_brightness_positive() {
        // Higher high-freq energy
        let energies = vec![-30.0, -30.0, -30.0, -30.0, -10.0, -10.0];
        let balance = SpectralBalance::new(STANDARD_BANDS.to_vec(), energies);
        assert!(
            balance.brightness_db() > 0.0,
            "brightness={}",
            balance.brightness_db()
        );
    }

    #[test]
    fn test_spectral_balance_tilt_flat() {
        // All bands at same energy → tilt ≈ 0
        let energies = vec![-20.0; STANDARD_BANDS.len()];
        let balance = SpectralBalance::new(STANDARD_BANDS.to_vec(), energies);
        let tilt = balance.spectral_tilt_db_per_octave();
        assert!(tilt.abs() < 1e-6, "tilt={}", tilt);
    }

    #[test]
    fn test_spectral_balance_tilt_direction() {
        // High-freq bands have lower energy → negative tilt (warm)
        let n = STANDARD_BANDS.len();
        let energies: Vec<f64> = (0..n).map(|i| -(i as f64) * 6.0).collect();
        let balance = SpectralBalance::new(STANDARD_BANDS.to_vec(), energies);
        let tilt = balance.spectral_tilt_db_per_octave();
        assert!(tilt < 0.0, "tilt={}", tilt);
    }

    #[test]
    fn test_meter_default_creation() {
        let meter = SpectralBalanceMeter::default_48k();
        assert_eq!(meter.config.bands.len(), STANDARD_BANDS.len());
    }

    #[test]
    fn test_meter_total_samples_after_processing() {
        let mut meter = SpectralBalanceMeter::default_48k();
        let sig = flat_signal(0.5, 4800);
        meter.process_mono(&sig);
        assert_eq!(meter.total_samples(), 4800);
    }

    #[test]
    fn test_meter_band_energy_returns_some() {
        let meter = SpectralBalanceMeter::default_48k();
        assert!(meter.band_energy_db(0).is_some());
        assert!(meter.band_energy_db(100).is_none());
    }

    #[test]
    fn test_meter_reset() {
        let mut meter = SpectralBalanceMeter::default_48k();
        let sig = flat_signal(0.5, 4800);
        meter.process_mono(&sig);
        meter.reset();
        assert_eq!(meter.total_samples(), 0);
        for e in &meter.band_energy_db {
            assert_eq!(*e, -96.0);
        }
    }

    #[test]
    fn test_meter_snapshot_band_count() {
        let meter = SpectralBalanceMeter::default_48k();
        let snap = meter.snapshot();
        assert_eq!(snap.band_count(), STANDARD_BANDS.len());
    }

    #[test]
    fn test_meter_process_stereo() {
        let mut meter = SpectralBalanceMeter::default_48k();
        let stereo: Vec<f64> = vec![0.3; 9600]; // 4800 stereo frames
        meter.process_interleaved_stereo(&stereo);
        assert_eq!(meter.total_samples(), 4800);
    }

    #[test]
    fn test_meter_hp_lp_alpha_range() {
        let alpha_hp = SpectralBalanceMeter::hp_alpha(80.0, 48000.0);
        let alpha_lp = SpectralBalanceMeter::lp_alpha(250.0, 48000.0);
        assert!(alpha_hp > 0.0 && alpha_hp < 1.0);
        assert!(alpha_lp > 0.0 && alpha_lp < 1.0);
    }
}
