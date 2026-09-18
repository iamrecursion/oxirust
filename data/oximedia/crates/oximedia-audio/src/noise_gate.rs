#![allow(dead_code)]
//! Advanced noise gate with hysteresis, look-ahead, and hold time.
//!
//! A noise gate attenuates signals that fall below a specified threshold,
//! reducing background noise during silent passages. This implementation
//! supports:
//!
//! - **Hysteresis**: Separate open and close thresholds to prevent chattering.
//! - **Attack/Release**: Smoothed envelope transitions for natural gating.
//! - **Hold time**: Keeps the gate open for a minimum duration after signal drops.
//! - **Look-ahead**: Delays audio to allow the gate to open before transients.
//! - **Side-chain filtering**: Apply a filter to the detection signal for
//!   frequency-selective gating.
//! - **Range control**: Partial attenuation instead of full silence.

/// State of the noise gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Gate is closed (attenuating signal).
    Closed,
    /// Gate is opening (attack phase).
    Opening,
    /// Gate is fully open (passing signal).
    Open,
    /// Gate is closing (release phase).
    Closing,
    /// Gate is in hold phase (staying open after signal drops).
    Hold,
}

/// Configuration for the noise gate.
#[derive(Debug, Clone)]
pub struct NoiseGateConfig {
    /// Threshold in dB for opening the gate.
    pub open_threshold_db: f64,
    /// Threshold in dB for closing the gate (should be <= open_threshold_db).
    pub close_threshold_db: f64,
    /// Attack time in seconds.
    pub attack_secs: f64,
    /// Release time in seconds.
    pub release_secs: f64,
    /// Hold time in seconds (gate stays open after signal drops).
    pub hold_secs: f64,
    /// Range in dB (0 = full silence, -20 = 20 dB attenuation).
    pub range_db: f64,
    /// Look-ahead time in seconds.
    pub lookahead_secs: f64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
}

impl Default for NoiseGateConfig {
    fn default() -> Self {
        Self {
            open_threshold_db: -40.0,
            close_threshold_db: -45.0,
            attack_secs: 0.001,
            release_secs: 0.05,
            hold_secs: 0.01,
            range_db: -80.0,
            lookahead_secs: 0.0,
            sample_rate: 48000.0,
        }
    }
}

/// A professional noise gate processor.
#[derive(Debug, Clone)]
pub struct NoiseGate {
    /// Current gate configuration.
    config: NoiseGateConfig,
    /// Current envelope level (0.0 = closed, 1.0 = open).
    envelope: f64,
    /// Current gate state.
    state: GateState,
    /// Hold counter in samples.
    hold_counter: usize,
    /// Hold time in samples.
    hold_samples: usize,
    /// Attack coefficient for exponential smoothing.
    attack_coeff: f64,
    /// Release coefficient for exponential smoothing.
    release_coeff: f64,
    /// Range as linear gain (the minimum gain when gate is closed).
    range_linear: f64,
    /// Look-ahead buffer.
    lookahead_buffer: Vec<f64>,
    /// Write position in the look-ahead buffer.
    lookahead_pos: usize,
    /// Look-ahead delay in samples.
    lookahead_samples: usize,
}

impl NoiseGate {
    /// Creates a new noise gate with the given configuration.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(config: NoiseGateConfig) -> Self {
        let attack_coeff = if config.attack_secs > 0.0 {
            (-1.0 / (config.attack_secs * config.sample_rate)).exp()
        } else {
            0.0
        };
        let release_coeff = if config.release_secs > 0.0 {
            (-1.0 / (config.release_secs * config.sample_rate)).exp()
        } else {
            0.0
        };
        let hold_samples = (config.hold_secs * config.sample_rate) as usize;
        let range_linear = db_to_linear(config.range_db);
        let lookahead_samples = (config.lookahead_secs * config.sample_rate) as usize;
        let lookahead_buffer = vec![0.0; lookahead_samples.max(1)];

        Self {
            config,
            envelope: 0.0,
            state: GateState::Closed,
            hold_counter: 0,
            hold_samples,
            attack_coeff,
            release_coeff,
            range_linear,
            lookahead_buffer,
            lookahead_pos: 0,
            lookahead_samples,
        }
    }

    /// Returns the current state of the gate.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Returns the current envelope level (0.0 = closed, 1.0 = open).
    pub fn envelope(&self) -> f64 {
        self.envelope
    }

    /// Computes the current gain applied to the signal.
    pub fn current_gain(&self) -> f64 {
        self.range_linear + (1.0 - self.range_linear) * self.envelope
    }

    /// Resets the gate to its initial state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.state = GateState::Closed;
        self.hold_counter = 0;
        self.lookahead_pos = 0;
        for s in &mut self.lookahead_buffer {
            *s = 0.0;
        }
    }

    /// Processes a block of mono audio samples in-place.
    ///
    /// The detection signal is the absolute value of the input.
    pub fn process_block(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            let detection_level = sample.abs();
            let detection_db = linear_to_db(detection_level);

            // State machine
            match self.state {
                GateState::Closed | GateState::Closing => {
                    if detection_db >= self.config.open_threshold_db {
                        self.state = GateState::Opening;
                    }
                }
                GateState::Opening => {
                    if detection_db < self.config.close_threshold_db {
                        self.state = GateState::Closing;
                    } else if self.envelope >= 0.999 {
                        self.state = GateState::Open;
                    }
                }
                GateState::Open => {
                    if detection_db < self.config.close_threshold_db {
                        self.state = GateState::Hold;
                        self.hold_counter = self.hold_samples;
                    }
                }
                GateState::Hold => {
                    if detection_db >= self.config.open_threshold_db {
                        self.state = GateState::Open;
                    } else if self.hold_counter == 0 {
                        self.state = GateState::Closing;
                    } else {
                        self.hold_counter -= 1;
                    }
                }
            }

            // Envelope follower
            let target = match self.state {
                GateState::Opening | GateState::Open | GateState::Hold => 1.0,
                GateState::Closed | GateState::Closing => 0.0,
            };

            let coeff = if target > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = target + coeff * (self.envelope - target);

            // Apply gain
            let gain = self.current_gain();

            if self.lookahead_samples > 0 {
                let delayed = self.lookahead_buffer[self.lookahead_pos];
                self.lookahead_buffer[self.lookahead_pos] = *sample;
                self.lookahead_pos = (self.lookahead_pos + 1) % self.lookahead_buffer.len();
                *sample = delayed * gain;
            } else {
                *sample *= gain;
            }
        }
    }

    /// Processes a block using an external side-chain signal for detection.
    ///
    /// `audio` is modified in-place; `sidechain` is used only for detection.
    pub fn process_with_sidechain(&mut self, audio: &mut [f64], sidechain: &[f64]) {
        let len = audio.len().min(sidechain.len());
        for i in 0..len {
            let detection_level = sidechain[i].abs();
            let detection_db = linear_to_db(detection_level);

            match self.state {
                GateState::Closed | GateState::Closing => {
                    if detection_db >= self.config.open_threshold_db {
                        self.state = GateState::Opening;
                    }
                }
                GateState::Opening => {
                    if detection_db < self.config.close_threshold_db {
                        self.state = GateState::Closing;
                    } else if self.envelope >= 0.999 {
                        self.state = GateState::Open;
                    }
                }
                GateState::Open => {
                    if detection_db < self.config.close_threshold_db {
                        self.state = GateState::Hold;
                        self.hold_counter = self.hold_samples;
                    }
                }
                GateState::Hold => {
                    if detection_db >= self.config.open_threshold_db {
                        self.state = GateState::Open;
                    } else if self.hold_counter == 0 {
                        self.state = GateState::Closing;
                    } else {
                        self.hold_counter -= 1;
                    }
                }
            }

            let target = match self.state {
                GateState::Opening | GateState::Open | GateState::Hold => 1.0,
                GateState::Closed | GateState::Closing => 0.0,
            };

            let coeff = if target > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = target + coeff * (self.envelope - target);

            audio[i] *= self.current_gain();
        }
    }
}

/// Converts a linear amplitude to decibels.
///
/// Returns `-f64::INFINITY` for zero or negative values.
pub fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        -f64::INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Converts decibels to linear amplitude.
pub fn db_to_linear(db: f64) -> f64 {
    if db <= -200.0 {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

/// Computes the RMS level of a block of samples.
#[allow(clippy::cast_precision_loss)]
pub fn rms_level(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Computes the peak level of a block of samples.
pub fn peak_level(samples: &[f64]) -> f64 {
    samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = NoiseGateConfig::default();
        assert!((cfg.open_threshold_db - (-40.0)).abs() < 1e-10);
        assert!((cfg.close_threshold_db - (-45.0)).abs() < 1e-10);
        assert!((cfg.sample_rate - 48000.0).abs() < 1e-10);
    }

    #[test]
    fn test_gate_initial_state() {
        let gate = NoiseGate::new(NoiseGateConfig::default());
        assert_eq!(gate.state(), GateState::Closed);
        assert!((gate.envelope() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gate_opens_on_loud_signal() {
        let config = NoiseGateConfig {
            open_threshold_db: -40.0,
            close_threshold_db: -45.0,
            attack_secs: 0.0001,
            release_secs: 0.001,
            hold_secs: 0.0,
            range_db: -80.0,
            lookahead_secs: 0.0,
            sample_rate: 48000.0,
        };
        let mut gate = NoiseGate::new(config);
        // A loud signal (0.5 = ~-6 dB) should open the gate
        let mut samples = vec![0.5; 1000];
        gate.process_block(&mut samples);
        assert!(gate.envelope() > 0.9);
    }

    #[test]
    fn test_gate_stays_closed_on_quiet_signal() {
        let config = NoiseGateConfig {
            open_threshold_db: -20.0,
            ..NoiseGateConfig::default()
        };
        let mut gate = NoiseGate::new(config);
        // Very quiet signal (-80 dB)
        let mut samples = vec![0.0001; 1000];
        gate.process_block(&mut samples);
        assert!(gate.envelope() < 0.1);
    }

    #[test]
    fn test_gate_attenuates_quiet_signal() {
        let config = NoiseGateConfig {
            open_threshold_db: -10.0,
            close_threshold_db: -15.0,
            range_db: -60.0,
            attack_secs: 0.0,
            release_secs: 0.0,
            hold_secs: 0.0,
            lookahead_secs: 0.0,
            sample_rate: 48000.0,
        };
        let mut gate = NoiseGate::new(config);
        let mut samples = vec![0.001; 500]; // very quiet
        gate.process_block(&mut samples);
        // Output should be attenuated
        let max_out = samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        assert!(max_out < 0.001);
    }

    #[test]
    fn test_gate_reset() {
        let mut gate = NoiseGate::new(NoiseGateConfig::default());
        let mut samples = vec![1.0; 100];
        gate.process_block(&mut samples);
        gate.reset();
        assert_eq!(gate.state(), GateState::Closed);
        assert!((gate.envelope() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db() {
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-10);
        assert!((linear_to_db(0.1) - (-20.0)).abs() < 0.01);
        assert_eq!(linear_to_db(0.0), -f64::INFINITY);
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 0.001);
        assert!((db_to_linear(-200.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_db_conversion() {
        let original = 0.5;
        let db = linear_to_db(original);
        let back = db_to_linear(db);
        assert!((back - original).abs() < 1e-10);
    }

    #[test]
    fn test_rms_level() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        assert!((rms_level(&samples) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rms_level_empty() {
        assert!((rms_level(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_level() {
        let samples = vec![0.1, -0.5, 0.3, -0.2];
        assert!((peak_level(&samples) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_current_gain_when_closed() {
        let config = NoiseGateConfig {
            range_db: -60.0,
            ..NoiseGateConfig::default()
        };
        let gate = NoiseGate::new(config);
        // When envelope is 0, gain should be range_linear
        let gain = gate.current_gain();
        assert!(gain < 0.01);
    }

    #[test]
    fn test_sidechain_gating() {
        let config = NoiseGateConfig {
            open_threshold_db: -20.0,
            close_threshold_db: -25.0,
            attack_secs: 0.0001,
            release_secs: 0.001,
            hold_secs: 0.0,
            range_db: -80.0,
            lookahead_secs: 0.0,
            sample_rate: 48000.0,
        };
        let mut gate = NoiseGate::new(config);
        let mut audio = vec![0.5; 500];
        let sidechain = vec![0.0001; 500]; // quiet sidechain
        gate.process_with_sidechain(&mut audio, &sidechain);
        // Gate should stay closed because sidechain is quiet
        assert!(gate.envelope() < 0.1);
    }
}
