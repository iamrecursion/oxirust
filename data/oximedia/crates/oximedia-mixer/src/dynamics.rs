//! Audio dynamics processing: compressor, expander, gate, and limiter.

/// Convert dB value to linear amplitude.
#[must_use]
#[allow(dead_code)]
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear amplitude to dB.
#[must_use]
#[allow(dead_code)]
pub fn linear_to_db(x: f32) -> f32 {
    if x <= 0.0 {
        -f32::INFINITY
    } else {
        20.0 * x.log10()
    }
}

// ---------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------

/// Configuration for a dynamic range compressor.
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Threshold in dB above which compression is applied.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 4.0 = 4:1).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Make-up gain in dB applied after compression.
    pub makeup_gain_db: f32,
    /// Soft-knee width in dB (0 = hard knee).
    pub knee_db: f32,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            makeup_gain_db: 0.0,
            knee_db: 6.0,
        }
    }
}

/// Dynamic range compressor with ballistic smoothing.
#[derive(Debug, Clone)]
pub struct Compressor {
    config: CompressorConfig,
    /// Current envelope level in dB (smoothed).
    envelope_db: f32,
}

impl Compressor {
    /// Create a new compressor from config.
    #[must_use]
    pub fn new(config: CompressorConfig) -> Self {
        Self {
            config,
            envelope_db: -120.0,
        }
    }

    /// Compute static gain reduction in dB for a given input level (dB).
    fn gain_computer_db(&self, input_db: f32) -> f32 {
        let t = self.config.threshold_db;
        let r = self.config.ratio;
        let k = self.config.knee_db;

        if k <= 0.0 {
            // Hard knee
            if input_db < t {
                0.0
            } else {
                (input_db - t) * (1.0 - 1.0 / r)
            }
        } else {
            // Soft knee
            let half_k = k / 2.0;
            if input_db < t - half_k {
                0.0
            } else if input_db > t + half_k {
                (input_db - t) * (1.0 - 1.0 / r)
            } else {
                let x = input_db - t + half_k;
                (1.0 - 1.0 / r) * x * x / (2.0 * k)
            }
        }
    }

    /// Process a single sample through the compressor.
    #[must_use]
    pub fn process_sample(&mut self, sample: f32, sample_rate: u32) -> f32 {
        let input_db = linear_to_db(sample.abs());

        // Ballistic smoothing (attack/release)
        let coeff = if input_db > self.envelope_db {
            let attack_secs = self.config.attack_ms / 1000.0;
            #[allow(clippy::cast_precision_loss)]
            (-1.0_f32 / (attack_secs * sample_rate as f32)).exp()
        } else {
            let release_secs = self.config.release_ms / 1000.0;
            #[allow(clippy::cast_precision_loss)]
            (-1.0_f32 / (release_secs * sample_rate as f32)).exp()
        };

        self.envelope_db = coeff * self.envelope_db + (1.0 - coeff) * input_db;

        let gain_reduction_db = self.gain_computer_db(self.envelope_db);
        let makeup_linear = db_to_linear(self.config.makeup_gain_db);
        let gain_linear = db_to_linear(-gain_reduction_db) * makeup_linear;

        sample * gain_linear
    }
}

// ---------------------------------------------------------------------------
// Expander
// ---------------------------------------------------------------------------

/// Configuration for a downward expander.
#[derive(Debug, Clone)]
pub struct ExpanderConfig {
    /// Threshold in dB below which expansion is applied.
    pub threshold_db: f32,
    /// Expansion ratio (e.g. 2.0 = 1:2).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
}

impl Default for ExpanderConfig {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            ratio: 2.0,
            attack_ms: 5.0,
            release_ms: 50.0,
        }
    }
}

/// Downward expander.
#[derive(Debug, Clone)]
pub struct Expander {
    config: ExpanderConfig,
    envelope_db: f32,
}

impl Expander {
    /// Create a new expander.
    #[must_use]
    pub fn new(config: ExpanderConfig) -> Self {
        Self {
            config,
            envelope_db: 0.0,
        }
    }

    /// Process a single sample through the expander.
    #[must_use]
    pub fn process_sample(&mut self, sample: f32, sample_rate: u32) -> f32 {
        let input_db = linear_to_db(sample.abs());

        let coeff = if input_db > self.envelope_db {
            let attack_secs = self.config.attack_ms / 1000.0;
            #[allow(clippy::cast_precision_loss)]
            (-1.0_f32 / (attack_secs * sample_rate as f32)).exp()
        } else {
            let release_secs = self.config.release_ms / 1000.0;
            #[allow(clippy::cast_precision_loss)]
            (-1.0_f32 / (release_secs * sample_rate as f32)).exp()
        };

        self.envelope_db = coeff * self.envelope_db + (1.0 - coeff) * input_db;

        let gain_db = if self.envelope_db < self.config.threshold_db {
            // Below threshold: apply downward expansion
            (self.envelope_db - self.config.threshold_db) * (self.config.ratio - 1.0)
        } else {
            0.0
        };

        sample * db_to_linear(gain_db)
    }
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

/// Configuration for a noise gate.
#[derive(Debug, Clone)]
pub struct GateConfig {
    /// Open threshold in dB.
    pub threshold_db: f32,
    /// Hysteresis: close threshold = `threshold_db` - `hysteresis_db`.
    pub hysteresis_db: f32,
    /// Attack time in milliseconds (time to open).
    pub attack_ms: f32,
    /// Release time in milliseconds (time to close).
    pub release_ms: f32,
    /// Hold time in milliseconds (minimum open duration).
    pub hold_ms: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            threshold_db: -50.0,
            hysteresis_db: 6.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            hold_ms: 10.0,
        }
    }
}

/// Noise gate state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    /// Gate is fully open (passing signal).
    Open,
    /// Gate is fully closed (blocking signal).
    Closed,
    /// Gate is transitioning from closed to open.
    Opening,
    /// Gate is transitioning from open to closed.
    Closing,
}

/// Noise gate with hysteresis, attack, hold, and release.
#[derive(Debug, Clone)]
pub struct Gate {
    config: GateConfig,
    envelope_db: f32,
    state: GateState,
    gain: f32,
    /// Hold timer in samples.
    hold_samples: u64,
    hold_counter: u64,
}

impl Gate {
    /// Create a new gate.
    #[must_use]
    pub fn new(config: GateConfig) -> Self {
        Self {
            config,
            envelope_db: -120.0,
            state: GateState::Closed,
            gain: 0.0,
            hold_samples: 0,
            hold_counter: 0,
        }
    }

    /// Current gate state.
    #[must_use]
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Process a single sample.
    #[must_use]
    pub fn process_sample(&mut self, sample: f32, sample_rate: u32) -> f32 {
        let input_db = linear_to_db(sample.abs());

        #[allow(clippy::cast_precision_loss)]
        let sr = sample_rate as f32;

        // Envelope follower
        let coeff = (-1.0_f32 / (0.001 * sr)).exp();
        self.envelope_db = coeff * self.envelope_db + (1.0 - coeff) * input_db;

        let open_threshold = self.config.threshold_db;
        let close_threshold = self.config.threshold_db - self.config.hysteresis_db;

        // Hold samples update (only done once per init, cached thereafter)
        self.hold_samples = (self.config.hold_ms / 1000.0 * sr) as u64;

        // State machine
        match self.state {
            GateState::Closed | GateState::Closing => {
                if self.envelope_db >= open_threshold {
                    self.state = GateState::Opening;
                    self.hold_counter = 0;
                }
            }
            GateState::Open | GateState::Opening => {
                if self.envelope_db < close_threshold {
                    if self.hold_counter < self.hold_samples {
                        self.hold_counter += 1;
                    } else {
                        self.state = GateState::Closing;
                    }
                } else {
                    self.hold_counter = 0;
                }
            }
        }

        // Gain smoothing
        let target = match self.state {
            GateState::Open | GateState::Opening => 1.0_f32,
            GateState::Closed | GateState::Closing => 0.0_f32,
        };

        let rate = if target > self.gain {
            self.config.attack_ms / 1000.0
        } else {
            self.config.release_ms / 1000.0
        };

        let gain_coeff = (-1.0_f32 / (rate * sr)).exp();
        self.gain = gain_coeff * self.gain + (1.0 - gain_coeff) * target;

        // Finalise state based on gain proximity
        if self.gain > 0.99 {
            self.state = GateState::Open;
        } else if self.gain < 0.01 {
            self.state = GateState::Closed;
        }

        sample * self.gain
    }
}

// ---------------------------------------------------------------------------
// Limiter
// ---------------------------------------------------------------------------

/// Configuration for a hard limiter (brickwall).
#[derive(Debug, Clone)]
pub struct LimiterConfig {
    /// Ceiling in dB (output will not exceed this level).
    pub ceiling_db: f32,
    /// Look-ahead in milliseconds.
    pub lookahead_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_db: 0.0,
            lookahead_ms: 0.0,
            release_ms: 50.0,
        }
    }
}

/// Brickwall limiter.
#[derive(Debug, Clone)]
pub struct Limiter {
    config: LimiterConfig,
    gain: f32,
}

impl Limiter {
    /// Create a new limiter.
    #[must_use]
    pub fn new(config: LimiterConfig) -> Self {
        Self { config, gain: 1.0 }
    }

    /// Process a single sample.
    #[must_use]
    pub fn process_sample(&mut self, sample: f32, sample_rate: u32) -> f32 {
        let ceiling_linear = db_to_linear(self.config.ceiling_db);
        let abs_sample = sample.abs();

        // If sample exceeds ceiling, clamp gain immediately
        let target_gain = if abs_sample > ceiling_linear {
            ceiling_linear / abs_sample
        } else {
            1.0_f32
        };

        // Release smoothing
        if target_gain < self.gain {
            self.gain = target_gain;
        } else {
            let release_secs = self.config.release_ms / 1000.0;
            #[allow(clippy::cast_precision_loss)]
            let coeff = (-1.0_f32 / (release_secs * sample_rate as f32)).exp();
            self.gain = coeff * self.gain + (1.0 - coeff) * target_gain;
        }

        // Hard clip as final safety net
        (sample * self.gain).clamp(-ceiling_linear, ceiling_linear)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_to_linear_unity() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_db_to_linear_minus_6() {
        // -6 dB ≈ 0.5012
        let l = db_to_linear(-6.0);
        assert!((l - 0.501_187).abs() < 1e-4, "got {l}");
    }

    #[test]
    fn test_linear_to_db_unity() {
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_to_db_roundtrip() {
        let original_db = -12.345_f32;
        let roundtrip = linear_to_db(db_to_linear(original_db));
        assert!((roundtrip - original_db).abs() < 1e-4);
    }

    #[test]
    fn test_linear_to_db_zero() {
        assert!(linear_to_db(0.0).is_infinite());
    }

    #[test]
    fn test_compressor_default_config() {
        let cfg = CompressorConfig::default();
        assert_eq!(cfg.threshold_db, -20.0);
        assert_eq!(cfg.ratio, 4.0);
    }

    #[test]
    fn test_compressor_below_threshold_no_reduction() {
        let mut comp = Compressor::new(CompressorConfig {
            threshold_db: -10.0,
            ratio: 4.0,
            attack_ms: 0.01,
            release_ms: 0.01,
            makeup_gain_db: 0.0,
            knee_db: 0.0,
        });
        // Silence should pass through unchanged (apart from floating point)
        let out = comp.process_sample(0.0, 48000);
        assert!(out.abs() < 1e-6);
    }

    #[test]
    fn test_compressor_attenuates_loud_signal() {
        let cfg = CompressorConfig {
            threshold_db: -20.0,
            ratio: 10.0,
            attack_ms: 0.01,
            release_ms: 0.01,
            makeup_gain_db: 0.0,
            knee_db: 0.0,
        };
        let mut comp = Compressor::new(cfg);
        // Drive the compressor for many samples to let envelope settle
        let input = 0.5_f32; // ≈ -6 dB, well above threshold of -20 dB
        let mut out = 0.0_f32;
        for _ in 0..10_000 {
            out = comp.process_sample(input, 48000);
        }
        // Output should be lower than input due to compression
        assert!(out.abs() < input, "out={out} should be < input={input}");
    }

    #[test]
    fn test_expander_default_config() {
        let cfg = ExpanderConfig::default();
        assert_eq!(cfg.threshold_db, -40.0);
        assert_eq!(cfg.ratio, 2.0);
    }

    #[test]
    fn test_expander_above_threshold_passthrough() {
        let mut exp = Expander::new(ExpanderConfig {
            threshold_db: -60.0,
            ratio: 2.0,
            attack_ms: 0.01,
            release_ms: 0.01,
        });
        let input = 0.9_f32; // well above threshold
        let out = exp.process_sample(input, 48000);
        // Gain should be near 1.0
        assert!((out - input).abs() < 0.1, "out={out}");
    }

    #[test]
    fn test_gate_default_config() {
        let cfg = GateConfig::default();
        assert_eq!(cfg.threshold_db, -50.0);
        assert_eq!(cfg.hysteresis_db, 6.0);
    }

    #[test]
    fn test_gate_starts_closed() {
        let gate = Gate::new(GateConfig::default());
        assert_eq!(gate.state(), GateState::Closed);
    }

    #[test]
    fn test_gate_opens_above_threshold() {
        let mut gate = Gate::new(GateConfig {
            threshold_db: -40.0,
            hysteresis_db: 6.0,
            attack_ms: 0.1,
            release_ms: 10.0,
            hold_ms: 0.0,
        });
        // Feed a loud signal to open the gate
        let input = 0.5_f32; // ≈ -6 dB
        for _ in 0..1000 {
            let _ = gate.process_sample(input, 48000);
        }
        assert_eq!(gate.state(), GateState::Open);
    }

    #[test]
    fn test_limiter_default_config() {
        let cfg = LimiterConfig::default();
        assert_eq!(cfg.ceiling_db, 0.0);
    }

    #[test]
    fn test_limiter_clamps_output() {
        let mut limiter = Limiter::new(LimiterConfig {
            ceiling_db: -6.0,
            lookahead_ms: 0.0,
            release_ms: 10.0,
        });
        let ceiling = db_to_linear(-6.0);
        let input = 0.9_f32; // above ceiling
        let out = limiter.process_sample(input, 48000);
        assert!(out.abs() <= ceiling + 1e-5, "out={out} ceiling={ceiling}");
    }

    #[test]
    fn test_limiter_passes_quiet_signal() {
        let mut limiter = Limiter::new(LimiterConfig::default());
        let input = 0.01_f32; // well below 0 dBFS
        let out = limiter.process_sample(input, 48000);
        assert!((out - input).abs() < 1e-5);
    }
}
