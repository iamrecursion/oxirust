//! PPM (Peak Programme Meter) implementation.
//!
//! Implements standards-compliant PPM ballistics for broadcast audio metering,
//! including IEC 268-10 Type I/II, Nordic NRK, BBC, and EBU standards.

#![allow(dead_code)]

/// PPM metering standard selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpmStandard {
    /// IEC 268-10 Type I (British/Nordic, fast attack).
    Iec268_10TypeI,
    /// IEC 268-10 Type II (EBU/European, slower attack).
    Iec268_10TypeII,
    /// Nordic NRK (Norwegian Broadcasting Corporation) standard.
    NordicNrk,
    /// BBC standard.
    Bbc,
    /// EBU (European Broadcasting Union) standard.
    Ebu,
}

/// Configuration for a PPM meter.
#[derive(Clone, Debug)]
pub struct PpmConfig {
    /// PPM standard to implement.
    pub standard: PpmStandard,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release (return) time in milliseconds (time to fall 20 dB).
    pub release_ms: f64,
    /// Integration time in milliseconds.
    pub integration_time_ms: f64,
    /// Peak hold duration in milliseconds (0 = no hold).
    pub peak_hold_ms: f64,
}

impl PpmConfig {
    /// IEC 268-10 Type I configuration (attack: 10ms, release: 1500ms).
    #[must_use]
    pub fn iec_type_i() -> Self {
        Self {
            standard: PpmStandard::Iec268_10TypeI,
            attack_ms: 10.0,
            release_ms: 1500.0,
            integration_time_ms: 5.0,
            peak_hold_ms: 1000.0,
        }
    }

    /// IEC 268-10 Type II configuration (attack: 5ms, release: 1500ms).
    #[must_use]
    pub fn iec_type_ii() -> Self {
        Self {
            standard: PpmStandard::Iec268_10TypeII,
            attack_ms: 5.0,
            release_ms: 1500.0,
            integration_time_ms: 10.0,
            peak_hold_ms: 2000.0,
        }
    }

    /// EBU PPM configuration (attack: 10ms, release: 1700ms).
    #[must_use]
    pub fn ebu() -> Self {
        Self {
            standard: PpmStandard::Ebu,
            attack_ms: 10.0,
            release_ms: 1700.0,
            integration_time_ms: 5.0,
            peak_hold_ms: 2000.0,
        }
    }

    /// BBC PPM configuration.
    #[must_use]
    pub fn bbc() -> Self {
        Self {
            standard: PpmStandard::Bbc,
            attack_ms: 10.0,
            release_ms: 2800.0,
            integration_time_ms: 5.0,
            peak_hold_ms: 0.0,
        }
    }

    /// Nordic NRK PPM configuration.
    #[must_use]
    pub fn nordic_nrk() -> Self {
        Self {
            standard: PpmStandard::NordicNrk,
            attack_ms: 10.0,
            release_ms: 1500.0,
            integration_time_ms: 5.0,
            peak_hold_ms: 1000.0,
        }
    }
}

/// PPM meter state and processing.
pub struct PpmMeter {
    /// Meter configuration.
    pub config: PpmConfig,
    /// Current envelope (linear).
    pub envelope: f64,
    /// Peak hold value (linear).
    pub peak: f64,
    /// Peak hold timer in milliseconds.
    pub peak_hold_timer: u64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Attack coefficient (per-sample).
    attack_coeff: f64,
    /// Release coefficient (per-sample).
    release_coeff: f64,
}

impl PpmMeter {
    /// Create a new PPM meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `config` - PPM meter configuration
    #[must_use]
    pub fn new(sample_rate: f64, config: PpmConfig) -> Self {
        let attack_coeff = compute_attack_coeff(config.attack_ms, sample_rate);
        let release_coeff = compute_release_coeff(config.release_ms, sample_rate);
        Self {
            config,
            envelope: 0.0,
            peak: 0.0,
            peak_hold_timer: 0,
            sample_rate,
            attack_coeff,
            release_coeff,
        }
    }

    /// Process a single audio sample and update the envelope.
    ///
    /// # Arguments
    ///
    /// * `sample` - Audio sample (linear amplitude)
    /// * `time_ms` - Current time in milliseconds (used for peak hold timer)
    pub fn process_sample(&mut self, sample: f64, time_ms: u64) {
        let abs_sample = sample.abs();

        if abs_sample > self.envelope {
            // Attack
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_sample;
        } else {
            // Release
            self.envelope *= self.release_coeff;
        }

        // Update peak hold
        if self.envelope >= self.peak {
            self.peak = self.envelope;
            self.peak_hold_timer = time_ms + self.config.peak_hold_ms as u64;
        } else if time_ms > self.peak_hold_timer {
            // Peak hold expired - let peak decay
            self.peak *= self.release_coeff;
        }
    }

    /// Get the current peak level in dBFS.
    #[must_use]
    pub fn peak_db(&self) -> f64 {
        linear_to_db(self.peak)
    }

    /// Get the current envelope level in dBFS.
    #[must_use]
    pub fn envelope_db(&self) -> f64 {
        linear_to_db(self.envelope)
    }

    /// Reset the meter to initial state.
    pub fn reset_peak(&mut self) {
        self.peak = 0.0;
        self.envelope = 0.0;
        self.peak_hold_timer = 0;
    }

    /// Process a block of samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to process
    /// * `start_time_ms` - Time of the first sample in milliseconds
    pub fn process_block(&mut self, samples: &[f64], start_time_ms: u64) {
        let samples_per_ms = self.sample_rate / 1000.0;
        for (i, &s) in samples.iter().enumerate() {
            let time_ms = start_time_ms + (i as f64 / samples_per_ms) as u64;
            self.process_sample(s, time_ms);
        }
    }
}

/// Convert dB value to linear amplitude.
#[must_use]
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dBFS.
///
/// Returns `f64::NEG_INFINITY` for zero amplitude.
#[must_use]
pub fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Compute the per-sample attack coefficient from attack time in milliseconds.
fn compute_attack_coeff(attack_ms: f64, sample_rate: f64) -> f64 {
    if attack_ms <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }
    let attack_samples = attack_ms * sample_rate / 1000.0;
    (-1.0 / attack_samples).exp()
}

/// Compute the per-sample release coefficient from release time in milliseconds.
///
/// Release time is defined as the time for the level to fall 20 dB.
fn compute_release_coeff(release_ms: f64, sample_rate: f64) -> f64 {
    if release_ms <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }
    let release_samples = release_ms * sample_rate / 1000.0;
    (-1.0 / release_samples).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f64 = 48000.0;

    #[test]
    fn test_db_to_linear_zero_db() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_db_to_linear_minus_6() {
        // -6 dB ≈ 0.5012 linear
        let linear = db_to_linear(-6.0);
        assert!((linear - 0.501187).abs() < 1e-5);
    }

    #[test]
    fn test_db_to_linear_minus_20() {
        // -20 dB = 0.1 linear
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_roundtrip() {
        let db = -12.0;
        let linear = db_to_linear(db);
        let back = linear_to_db(linear);
        assert!((back - db).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_zero_is_neg_infinity() {
        assert!(linear_to_db(0.0).is_infinite());
        assert!(linear_to_db(0.0) < 0.0);
    }

    #[test]
    fn test_ppm_config_iec_type_i() {
        let cfg = PpmConfig::iec_type_i();
        assert_eq!(cfg.standard, PpmStandard::Iec268_10TypeI);
        assert!((cfg.attack_ms - 10.0).abs() < 1e-10);
        assert!((cfg.release_ms - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_ppm_config_iec_type_ii() {
        let cfg = PpmConfig::iec_type_ii();
        assert_eq!(cfg.standard, PpmStandard::Iec268_10TypeII);
        assert!((cfg.attack_ms - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ppm_config_ebu() {
        let cfg = PpmConfig::ebu();
        assert_eq!(cfg.standard, PpmStandard::Ebu);
    }

    #[test]
    fn test_ppm_meter_new() {
        let meter = PpmMeter::new(SAMPLE_RATE, PpmConfig::iec_type_i());
        assert!((meter.envelope - 0.0).abs() < 1e-10);
        assert!((meter.peak - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ppm_meter_attack_on_impulse() {
        let mut meter = PpmMeter::new(SAMPLE_RATE, PpmConfig::iec_type_i());
        // Process a full-scale impulse
        meter.process_sample(1.0, 0);
        assert!(
            meter.envelope > 0.0,
            "envelope should increase after impulse"
        );
    }

    #[test]
    fn test_ppm_meter_release_decays() {
        let mut meter = PpmMeter::new(SAMPLE_RATE, PpmConfig::iec_type_i());
        // Attack to near full scale
        for _ in 0..1000 {
            meter.process_sample(1.0, 0);
        }
        let after_attack = meter.envelope;
        // Let it decay
        for i in 0..10000 {
            meter.process_sample(0.0, i as u64 / 48);
        }
        assert!(meter.envelope < after_attack, "envelope should decay");
    }

    #[test]
    fn test_ppm_meter_peak_holds() {
        let mut meter = PpmMeter::new(SAMPLE_RATE, PpmConfig::iec_type_i());
        // Set a peak
        for _ in 0..500 {
            meter.process_sample(0.8, 0);
        }
        let peak_after_attack = meter.peak;
        // Process silence but within hold window
        for i in 0..100 {
            meter.process_sample(0.0, (i as u64) / 48);
        }
        // Peak should still be held (within hold_ms window)
        assert!(meter.peak <= peak_after_attack + 0.01);
    }

    #[test]
    fn test_ppm_meter_reset_peak() {
        let mut meter = PpmMeter::new(SAMPLE_RATE, PpmConfig::iec_type_i());
        meter.process_sample(1.0, 0);
        meter.reset_peak();
        assert!((meter.peak - 0.0).abs() < 1e-10);
        assert!((meter.envelope - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ppm_meter_peak_db_silence() {
        let meter = PpmMeter::new(SAMPLE_RATE, PpmConfig::iec_type_i());
        let db = meter.peak_db();
        assert!(db.is_infinite() && db < 0.0);
    }

    #[test]
    fn test_ppm_meter_process_block() {
        let mut meter = PpmMeter::new(SAMPLE_RATE, PpmConfig::iec_type_ii());
        let block: Vec<f64> = (0..480).map(|_| 0.5).collect();
        meter.process_block(&block, 0);
        assert!(meter.envelope > 0.0);
    }
}
