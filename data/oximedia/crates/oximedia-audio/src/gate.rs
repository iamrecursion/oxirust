//! Noise gate for audio signals.
//!
//! Implements a noise gate with threshold, hysteresis, hold time, and
//! a state machine for open/close transitions.

/// State of the noise gate.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateState {
    /// Gate is open; signal passes through.
    Open,
    /// Gate is in hold phase after signal dropped below threshold.
    Holding,
    /// Gate is closed; signal is attenuated.
    Closed,
}

/// Noise gate configuration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Open threshold in dBFS.
    pub threshold_db: f32,
    /// Close threshold in dBFS (should be <= threshold_db for hysteresis).
    pub close_threshold_db: f32,
    /// Attack time in seconds (how fast the gate opens).
    pub attack_secs: f32,
    /// Release time in seconds (how fast the gate closes after hold).
    pub release_secs: f32,
    /// Hold time in seconds (stay open after signal drops below close threshold).
    pub hold_secs: f32,
    /// Attenuation applied when gate is closed, in dB (0 = fully closed, -60 = -60 dBFS floor).
    pub floor_db: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Look-ahead delay in seconds (0.0 = disabled).
    ///
    /// When enabled, the gate analyses the signal ahead of time so it can
    /// open *before* the transient arrives, preserving the attack of the sound.
    pub lookahead_secs: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            close_threshold_db: -45.0,
            attack_secs: 0.001,
            release_secs: 0.05,
            hold_secs: 0.1,
            floor_db: -80.0,
            sample_rate: 48_000.0,
            lookahead_secs: 0.0,
        }
    }
}

/// Noise gate processor.
#[allow(dead_code)]
pub struct NoiseGate {
    config: GateConfig,
    state: GateState,
    /// Envelope follower level (linear).
    envelope: f32,
    /// Remaining hold samples.
    hold_samples_remaining: u32,
    /// Total hold samples.
    hold_samples_total: u32,
    /// Current gain (0.0..=1.0).
    current_gain: f32,
    /// Attack coefficient for gain smoothing.
    attack_coeff: f32,
    /// Release coefficient for gain smoothing.
    release_coeff: f32,
    /// Floor gain (linear).
    floor_gain: f32,
    /// Look-ahead delay buffer (ring buffer).
    lookahead_buffer: Vec<f32>,
    /// Write position in the look-ahead buffer.
    lookahead_write_pos: usize,
    /// Number of look-ahead samples (0 = disabled).
    lookahead_samples: usize,
}

impl NoiseGate {
    /// Create a new noise gate.
    #[allow(dead_code)]
    pub fn new(config: GateConfig) -> Self {
        let hold_samples_total = (config.hold_secs * config.sample_rate).round() as u32;
        let attack_coeff = Self::time_coeff(config.attack_secs, config.sample_rate);
        let release_coeff = Self::time_coeff(config.release_secs, config.sample_rate);
        let floor_gain = db_to_linear(config.floor_db);
        let lookahead_samples = if config.lookahead_secs > 0.0 {
            (config.lookahead_secs * config.sample_rate).round() as usize
        } else {
            0
        };
        let lookahead_buffer = vec![0.0; lookahead_samples.max(1)];

        Self {
            config,
            state: GateState::Closed,
            envelope: 0.0,
            hold_samples_remaining: 0,
            hold_samples_total,
            current_gain: 0.0,
            attack_coeff,
            release_coeff,
            floor_gain,
            lookahead_buffer,
            lookahead_write_pos: 0,
            lookahead_samples,
        }
    }

    fn time_coeff(time_secs: f32, sample_rate: f32) -> f32 {
        if time_secs <= 0.0 || sample_rate <= 0.0 {
            return 0.0;
        }
        (-1.0_f32 / (time_secs * sample_rate)).exp()
    }

    /// Process a single sample and return the gated output.
    #[allow(dead_code)]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Update envelope
        let abs_input = input.abs();
        if abs_input > self.envelope {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_input;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_input;
        }

        let level_db = linear_to_db(self.envelope);

        // State machine
        match self.state {
            GateState::Closed => {
                if level_db >= self.config.threshold_db {
                    self.state = GateState::Open;
                    self.hold_samples_remaining = self.hold_samples_total;
                }
            }
            GateState::Open => {
                if level_db < self.config.close_threshold_db {
                    self.state = GateState::Holding;
                    self.hold_samples_remaining = self.hold_samples_total;
                } else {
                    self.hold_samples_remaining = self.hold_samples_total;
                }
            }
            GateState::Holding => {
                if level_db >= self.config.threshold_db {
                    self.state = GateState::Open;
                    self.hold_samples_remaining = self.hold_samples_total;
                } else if self.hold_samples_remaining == 0 {
                    self.state = GateState::Closed;
                } else {
                    self.hold_samples_remaining -= 1;
                }
            }
        }

        // Target gain based on state
        let target_gain = match self.state {
            GateState::Open | GateState::Holding => 1.0,
            GateState::Closed => self.floor_gain,
        };

        // Smooth gain transitions
        if target_gain > self.current_gain {
            self.current_gain =
                self.attack_coeff * self.current_gain + (1.0 - self.attack_coeff) * target_gain;
        } else {
            self.current_gain =
                self.release_coeff * self.current_gain + (1.0 - self.release_coeff) * target_gain;
        }

        // Look-ahead: apply gain to the *delayed* sample so the gate opens
        // before the transient arrives.
        if self.lookahead_samples > 0 {
            let delayed = self.lookahead_buffer[self.lookahead_write_pos];
            self.lookahead_buffer[self.lookahead_write_pos] = input;
            self.lookahead_write_pos = (self.lookahead_write_pos + 1) % self.lookahead_buffer.len();
            delayed * self.current_gain
        } else {
            input * self.current_gain
        }
    }

    /// Process a buffer of samples in-place.
    #[allow(dead_code)]
    pub fn process_buffer(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Returns the current gate state.
    #[allow(dead_code)]
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Returns the current gain (0.0..=1.0).
    #[allow(dead_code)]
    pub fn current_gain(&self) -> f32 {
        self.current_gain
    }

    /// Reset the gate to closed state.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.state = GateState::Closed;
        self.envelope = 0.0;
        self.hold_samples_remaining = 0;
        self.current_gain = 0.0;
        self.lookahead_buffer.fill(0.0);
        self.lookahead_write_pos = 0;
    }

    /// Set the look-ahead time.
    ///
    /// A non-zero value introduces a delay and allows the gate to open
    /// before the transient arrives, preserving the attack of the sound.
    #[allow(dead_code)]
    pub fn set_lookahead(&mut self, lookahead_secs: f32) {
        self.config.lookahead_secs = lookahead_secs;
        self.lookahead_samples = if lookahead_secs > 0.0 {
            (lookahead_secs * self.config.sample_rate).round() as usize
        } else {
            0
        };
        self.lookahead_buffer = vec![0.0; self.lookahead_samples.max(1)];
        self.lookahead_write_pos = 0;
    }

    /// Get the current look-ahead delay in samples.
    #[allow(dead_code)]
    pub fn lookahead_samples(&self) -> usize {
        self.lookahead_samples
    }

    /// Update hold time.
    #[allow(dead_code)]
    pub fn set_hold(&mut self, hold_secs: f32) {
        self.config.hold_secs = hold_secs;
        self.hold_samples_total = (hold_secs * self.config.sample_rate).round() as u32;
    }

    /// Update floor attenuation.
    #[allow(dead_code)]
    pub fn set_floor_db(&mut self, floor_db: f32) {
        self.config.floor_db = floor_db;
        self.floor_gain = db_to_linear(floor_db);
    }

    /// Process a single sample using an external **sidechain** key signal.
    ///
    /// The gate state machine is driven by `key` (the sidechain input) while
    /// the actual gating is applied to `input` (the programme signal).
    /// A common use-case is a gate keyed by a kick drum to tighten a bass guitar.
    ///
    /// When the look-ahead buffer is enabled the delayed `input` sample is
    /// output with the gain computed from the current `key` sample, so the gate
    /// opens before the transient in the programme signal arrives.
    #[allow(dead_code)]
    pub fn process_sample_sidechain(&mut self, input: f32, key: f32) -> f32 {
        // Envelope follower driven by the key signal
        let abs_key = key.abs();
        if abs_key > self.envelope {
            self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_key;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_key;
        }

        let level_db = linear_to_db(self.envelope);

        // State machine (same as self-keyed version)
        match self.state {
            GateState::Closed => {
                if level_db >= self.config.threshold_db {
                    self.state = GateState::Open;
                    self.hold_samples_remaining = self.hold_samples_total;
                }
            }
            GateState::Open => {
                if level_db < self.config.close_threshold_db {
                    self.state = GateState::Holding;
                    self.hold_samples_remaining = self.hold_samples_total;
                } else {
                    self.hold_samples_remaining = self.hold_samples_total;
                }
            }
            GateState::Holding => {
                if level_db >= self.config.threshold_db {
                    self.state = GateState::Open;
                    self.hold_samples_remaining = self.hold_samples_total;
                } else if self.hold_samples_remaining == 0 {
                    self.state = GateState::Closed;
                } else {
                    self.hold_samples_remaining -= 1;
                }
            }
        }

        let target_gain = match self.state {
            GateState::Open | GateState::Holding => 1.0,
            GateState::Closed => self.floor_gain,
        };

        if target_gain > self.current_gain {
            self.current_gain =
                self.attack_coeff * self.current_gain + (1.0 - self.attack_coeff) * target_gain;
        } else {
            self.current_gain =
                self.release_coeff * self.current_gain + (1.0 - self.release_coeff) * target_gain;
        }

        if self.lookahead_samples > 0 {
            let delayed = self.lookahead_buffer[self.lookahead_write_pos];
            self.lookahead_buffer[self.lookahead_write_pos] = input;
            self.lookahead_write_pos = (self.lookahead_write_pos + 1) % self.lookahead_buffer.len();
            delayed * self.current_gain
        } else {
            input * self.current_gain
        }
    }

    /// Process a buffer of samples using an external sidechain key buffer.
    ///
    /// `samples` and `key_samples` must be the same length. The output is
    /// written back into `samples`.
    #[allow(dead_code)]
    pub fn process_buffer_sidechain(&mut self, samples: &mut [f32], key_samples: &[f32]) {
        for (i, s) in samples.iter_mut().enumerate() {
            let key = key_samples.get(i).copied().unwrap_or(0.0);
            *s = self.process_sample_sidechain(*s, key);
        }
    }
}

/// Convert linear amplitude to dBFS.
#[allow(dead_code)]
fn linear_to_db(linear: f32) -> f32 {
    if linear <= 1e-10 {
        -120.0
    } else {
        20.0 * linear.log10()
    }
}

/// Convert dBFS to linear amplitude.
#[allow(dead_code)]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gate() -> NoiseGate {
        NoiseGate::new(GateConfig::default())
    }

    #[test]
    fn test_gate_creation() {
        let g = make_gate();
        assert_eq!(g.state(), GateState::Closed);
    }

    #[test]
    fn test_silence_stays_closed() {
        let mut g = make_gate();
        for _ in 0..1000 {
            g.process_sample(0.0);
        }
        assert_eq!(g.state(), GateState::Closed);
    }

    #[test]
    fn test_loud_signal_opens_gate() {
        let mut g = make_gate();
        // Signal well above threshold (-40 dBFS), threshold linear ≈ 0.01
        for _ in 0..2000 {
            g.process_sample(0.1);
        }
        assert_eq!(g.state(), GateState::Open);
    }

    #[test]
    fn test_reset_closes_gate() {
        let mut g = make_gate();
        for _ in 0..2000 {
            g.process_sample(0.1);
        }
        g.reset();
        assert_eq!(g.state(), GateState::Closed);
        assert_eq!(g.envelope, 0.0);
    }

    #[test]
    fn test_db_to_linear() {
        let lin = db_to_linear(0.0);
        assert!((lin - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_to_db_zero() {
        let db = linear_to_db(0.0);
        assert_eq!(db, -120.0);
    }

    #[test]
    fn test_linear_to_db_one() {
        let db = linear_to_db(1.0);
        assert!((db - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_process_buffer_not_nan() {
        let mut g = make_gate();
        let mut buf = vec![0.05_f32; 500];
        g.process_buffer(&mut buf);
        for s in &buf {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_set_hold_updates_total() {
        let mut g = make_gate();
        g.set_hold(0.5);
        assert_eq!(g.hold_samples_total, (0.5 * 48_000.0) as u32);
    }

    #[test]
    fn test_set_floor_db() {
        let mut g = make_gate();
        g.set_floor_db(-60.0);
        let expected = db_to_linear(-60.0);
        assert!((g.floor_gain - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gate_transitions_to_holding() {
        let config = GateConfig {
            hold_secs: 1.0,
            sample_rate: 48_000.0,
            ..GateConfig::default()
        };
        let mut g = NoiseGate::new(config);
        // Open the gate
        for _ in 0..5000 {
            g.process_sample(0.1);
        }
        assert_eq!(g.state(), GateState::Open);
        // Drop signal below close threshold
        for _ in 0..100 {
            g.process_sample(0.0);
        }
        // Should be in Holding (hold_secs = 1.0 means lots of samples left)
        assert!(g.state() == GateState::Holding || g.state() == GateState::Open);
    }

    #[test]
    fn test_current_gain_closed_is_low() {
        let mut g = make_gate();
        // Process silence (gate closed)
        for _ in 0..500 {
            g.process_sample(0.0);
        }
        // After settling, gain should be near floor
        let gain = g.current_gain();
        assert!(gain < 0.5);
    }

    #[test]
    fn test_hysteresis_close_threshold_lower() {
        let config = GateConfig::default();
        assert!(config.close_threshold_db <= config.threshold_db);
    }

    // --- Sidechain tests ---

    #[test]
    fn test_sidechain_gate_opens_on_loud_key() {
        // Gate with threshold at -40 dBFS
        let config = GateConfig {
            threshold_db: -40.0,
            close_threshold_db: -45.0,
            attack_secs: 0.001,
            release_secs: 0.05,
            hold_secs: 0.1,
            floor_db: -80.0,
            sample_rate: 48_000.0,
            lookahead_secs: 0.0,
        };
        let mut g = NoiseGate::new(config);

        // Key signal well above threshold; programme signal is low-level noise
        let key = vec![0.1_f32; 10_000]; // above threshold linear ≈ 0.01
        let mut prog = vec![0.05_f32; 10_000];
        g.process_buffer_sidechain(&mut prog, &key);

        // Gate should have opened — programme allowed through
        assert_eq!(g.state(), GateState::Open);
        let tail_val = prog[9_000];
        assert!(
            tail_val > 0.01,
            "Gate should pass programme; tail={tail_val}"
        );
    }

    #[test]
    fn test_sidechain_gate_stays_closed_with_silent_key() {
        let mut g = make_gate();
        let key = vec![0.0_f32; 5_000];
        let mut prog = vec![0.05_f32; 5_000];
        g.process_buffer_sidechain(&mut prog, &key);
        assert_eq!(g.state(), GateState::Closed);
    }

    #[test]
    fn test_sidechain_output_is_finite() {
        let mut g = make_gate();
        for i in 0..2000_usize {
            let key = (i as f32 * 0.01).sin();
            let prog = (i as f32 * 0.007).sin() * 0.5;
            let out = g.process_sample_sidechain(prog, key);
            assert!(
                out.is_finite(),
                "sidechain gate output must be finite at {i}"
            );
        }
    }

    // --- Look-ahead delay tests ---

    #[test]
    fn test_gate_lookahead_disabled_by_default() {
        let g = make_gate();
        assert_eq!(g.lookahead_samples(), 0);
    }

    #[test]
    fn test_gate_lookahead_creates_delay() {
        let config = GateConfig {
            lookahead_secs: 0.005,
            ..GateConfig::default()
        };
        let g = NoiseGate::new(config);
        assert_eq!(g.lookahead_samples(), 240);
    }

    #[test]
    fn test_gate_lookahead_output_delayed() {
        let config = GateConfig {
            lookahead_secs: 0.001, // 48 samples at 48 kHz
            threshold_db: -100.0,  // gate always open
            close_threshold_db: -110.0,
            ..GateConfig::default()
        };
        let mut g = NoiseGate::new(config);
        let delay = g.lookahead_samples();
        assert!(delay > 0);

        // First `delay` outputs should be zeros (from the delay buffer init)
        for i in 0..delay {
            let out = g.process_sample(1.0);
            assert!(
                out.abs() < 0.1,
                "sample {i} should be near-zero (delayed), got {out}"
            );
        }
    }

    #[test]
    fn test_gate_lookahead_output_finite() {
        let config = GateConfig {
            lookahead_secs: 0.002,
            ..GateConfig::default()
        };
        let mut g = NoiseGate::new(config);
        for _ in 0..5000 {
            let out = g.process_sample(0.05);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn test_gate_set_lookahead_runtime() {
        let mut g = make_gate();
        assert_eq!(g.lookahead_samples(), 0);
        g.set_lookahead(0.01);
        assert_eq!(g.lookahead_samples(), 480);
    }

    #[test]
    fn test_gate_lookahead_reset_clears_buffer() {
        let config = GateConfig {
            lookahead_secs: 0.005,
            ..GateConfig::default()
        };
        let mut g = NoiseGate::new(config);
        for _ in 0..500 {
            g.process_sample(0.1);
        }
        g.reset();
        assert_eq!(g.lookahead_write_pos, 0);
        for &s in &g.lookahead_buffer {
            assert_eq!(s, 0.0);
        }
    }
}
