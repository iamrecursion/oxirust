//! Dynamic range compressor for audio signals.
//!
//! Implements a full-featured compressor with attack/release envelope,
//! ratio, knee, and gain reduction tracking.

/// Compression ratio modes.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KneeMode {
    /// Hard knee: abrupt transition at threshold.
    Hard,
    /// Soft knee: gradual transition around threshold.
    Soft(f32),
}

/// Dynamic range compressor configuration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Threshold in dBFS above which compression begins.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 4.0 = 4:1).
    pub ratio: f32,
    /// Attack time in seconds.
    pub attack_secs: f32,
    /// Release time in seconds.
    pub release_secs: f32,
    /// Make-up gain in dB applied after compression.
    pub makeup_gain_db: f32,
    /// Knee mode (hard or soft with width in dB).
    pub knee: KneeMode,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Look-ahead delay in seconds (0.0 = disabled).
    ///
    /// When enabled, the compressor detects peaks ahead of time and applies
    /// gain reduction before the transient arrives, producing transparent
    /// compression without overshoot.
    pub lookahead_secs: f32,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_secs: 0.01,
            release_secs: 0.1,
            makeup_gain_db: 0.0,
            knee: KneeMode::Hard,
            sample_rate: 48_000.0,
            lookahead_secs: 0.0,
        }
    }
}

/// Dynamic range compressor state.
#[allow(dead_code)]
pub struct Compressor {
    config: CompressorConfig,
    /// Current envelope follower level (linear).
    envelope: f32,
    /// Attack coefficient.
    attack_coeff: f32,
    /// Release coefficient.
    release_coeff: f32,
    /// Last computed gain reduction in dB (positive = reduction).
    last_gain_reduction_db: f32,
    /// Look-ahead delay buffer (ring buffer).
    lookahead_buffer: Vec<f32>,
    /// Write position in the look-ahead buffer.
    lookahead_write_pos: usize,
    /// Number of look-ahead samples (0 = disabled).
    lookahead_samples: usize,
}

impl Compressor {
    /// Create a new compressor from the given configuration.
    #[allow(dead_code)]
    pub fn new(config: CompressorConfig) -> Self {
        let attack_coeff = Self::time_to_coeff(config.attack_secs, config.sample_rate);
        let release_coeff = Self::time_to_coeff(config.release_secs, config.sample_rate);
        let lookahead_samples = if config.lookahead_secs > 0.0 {
            (config.lookahead_secs * config.sample_rate).round() as usize
        } else {
            0
        };
        let lookahead_buffer = vec![0.0; lookahead_samples.max(1)];
        Self {
            config,
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            last_gain_reduction_db: 0.0,
            lookahead_buffer,
            lookahead_write_pos: 0,
            lookahead_samples,
        }
    }

    /// Convert a time constant (seconds) to a one-pole IIR coefficient.
    #[allow(dead_code)]
    fn time_to_coeff(time_secs: f32, sample_rate: f32) -> f32 {
        if time_secs <= 0.0 || sample_rate <= 0.0 {
            return 0.0;
        }
        (-1.0_f32 / (time_secs * sample_rate)).exp()
    }

    /// Compute gain reduction in dB for an input level in dBFS.
    #[allow(dead_code)]
    fn compute_gain_reduction_db(&self, input_db: f32) -> f32 {
        let threshold = self.config.threshold_db;
        let ratio = self.config.ratio;

        match self.config.knee {
            KneeMode::Hard => {
                if input_db <= threshold {
                    0.0
                } else {
                    (input_db - threshold) * (1.0 - 1.0 / ratio)
                }
            }
            KneeMode::Soft(knee_width) => {
                let half_knee = knee_width / 2.0;
                if input_db < threshold - half_knee {
                    0.0
                } else if input_db > threshold + half_knee {
                    (input_db - threshold) * (1.0 - 1.0 / ratio)
                } else {
                    // Smooth interpolation within knee
                    let x = input_db - (threshold - half_knee);
                    let gain = x * x / (2.0 * knee_width);
                    gain * (1.0 - 1.0 / ratio)
                }
            }
        }
    }

    /// Process a single audio sample, returning the compressed output.
    ///
    /// When look-ahead is enabled, the compressor analyses the input signal
    /// ahead of time. The *current* input is written into a delay buffer while
    /// the *delayed* sample is output with gain reduction computed from the
    /// current (future) peak. This allows the compressor to react to transients
    /// before they arrive, producing transparent limiting with no overshoot.
    #[allow(dead_code)]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let abs_input = input.abs();

        // Envelope follower
        if abs_input > self.envelope {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_input;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_input;
        }

        // Compute gain reduction
        let level_db = if self.envelope > 1e-10 {
            20.0 * self.envelope.log10()
        } else {
            -120.0
        };

        let gr_db = self.compute_gain_reduction_db(level_db);
        self.last_gain_reduction_db = gr_db;

        // Apply gain: reduction + makeup
        let total_gain_db = -gr_db + self.config.makeup_gain_db;
        let gain_linear = 10.0_f32.powf(total_gain_db / 20.0);

        // Look-ahead: apply gain to the *delayed* sample
        if self.lookahead_samples > 0 {
            let delayed = self.lookahead_buffer[self.lookahead_write_pos];
            self.lookahead_buffer[self.lookahead_write_pos] = input;
            self.lookahead_write_pos = (self.lookahead_write_pos + 1) % self.lookahead_buffer.len();
            delayed * gain_linear
        } else {
            input * gain_linear
        }
    }

    /// Process a buffer of samples in-place.
    #[allow(dead_code)]
    pub fn process_buffer(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Returns the last computed gain reduction in dB (positive = gain reduction applied).
    #[allow(dead_code)]
    pub fn gain_reduction_db(&self) -> f32 {
        self.last_gain_reduction_db
    }

    /// Reset the compressor state.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.last_gain_reduction_db = 0.0;
        self.lookahead_buffer.fill(0.0);
        self.lookahead_write_pos = 0;
    }

    /// Set the look-ahead time.
    ///
    /// A non-zero value introduces a delay equal to `lookahead_secs` and allows
    /// the compressor to anticipate transients for transparent compression.
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

    /// Update the attack time.
    #[allow(dead_code)]
    pub fn set_attack(&mut self, attack_secs: f32) {
        self.config.attack_secs = attack_secs;
        self.attack_coeff = Self::time_to_coeff(attack_secs, self.config.sample_rate);
    }

    /// Update the release time.
    #[allow(dead_code)]
    pub fn set_release(&mut self, release_secs: f32) {
        self.config.release_secs = release_secs;
        self.release_coeff = Self::time_to_coeff(release_secs, self.config.sample_rate);
    }

    /// Update the threshold.
    #[allow(dead_code)]
    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.config.threshold_db = threshold_db;
    }

    /// Update the ratio.
    #[allow(dead_code)]
    pub fn set_ratio(&mut self, ratio: f32) {
        self.config.ratio = ratio.max(1.0);
    }

    /// Process a single sample using an external **sidechain** key signal.
    ///
    /// The gain reduction is computed from `key` (the sidechain input) while
    /// the actual compression is applied to `input` (the programme signal).
    /// This allows classic sidechain effects such as ducking (e.g. compressing
    /// music whenever a voice is present on the key channel).
    ///
    /// When the look-ahead buffer is enabled the *delayed* version of `input`
    /// is output with the gain derived from the *current* `key` sample.
    #[allow(dead_code)]
    pub fn process_sample_sidechain(&mut self, input: f32, key: f32) -> f32 {
        let abs_key = key.abs();

        // Envelope follower driven by the key signal
        if abs_key > self.envelope {
            self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_key;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_key;
        }

        // Compute gain reduction from key level
        let level_db = if self.envelope > 1e-10 {
            20.0 * self.envelope.log10()
        } else {
            -120.0
        };

        let gr_db = self.compute_gain_reduction_db(level_db);
        self.last_gain_reduction_db = gr_db;

        // Apply gain: reduction + makeup — applied to the programme signal
        let total_gain_db = -gr_db + self.config.makeup_gain_db;
        let gain_linear = 10.0_f32.powf(total_gain_db / 20.0);

        if self.lookahead_samples > 0 {
            let delayed = self.lookahead_buffer[self.lookahead_write_pos];
            self.lookahead_buffer[self.lookahead_write_pos] = input;
            self.lookahead_write_pos = (self.lookahead_write_pos + 1) % self.lookahead_buffer.len();
            delayed * gain_linear
        } else {
            input * gain_linear
        }
    }

    /// Process a buffer of samples using an external sidechain key buffer.
    ///
    /// `samples` and `key_samples` must be the same length. The output is
    /// written back into `samples`.
    ///
    /// # Panics
    ///
    /// Does not panic. If `key_samples` is shorter than `samples`, the
    /// remaining input samples are processed without sidechain (key = 0).
    #[allow(dead_code)]
    pub fn process_buffer_sidechain(&mut self, samples: &mut [f32], key_samples: &[f32]) {
        for (i, s) in samples.iter_mut().enumerate() {
            let key = key_samples.get(i).copied().unwrap_or(0.0);
            *s = self.process_sample_sidechain(*s, key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_compressor() -> Compressor {
        Compressor::new(CompressorConfig::default())
    }

    #[test]
    fn test_compressor_creation() {
        let c = make_compressor();
        assert_eq!(c.config.threshold_db, -20.0);
        assert_eq!(c.config.ratio, 4.0);
    }

    #[test]
    fn test_silence_passes_through() {
        let mut c = make_compressor();
        let out = c.process_sample(0.0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_envelope_rises_on_signal() {
        let mut c = make_compressor();
        for _ in 0..100 {
            c.process_sample(1.0);
        }
        assert!(c.envelope > 0.0);
    }

    #[test]
    fn test_gain_reduction_hard_knee_below_threshold() {
        let c = make_compressor();
        let gr = c.compute_gain_reduction_db(-30.0);
        assert_eq!(gr, 0.0, "no reduction below threshold");
    }

    #[test]
    fn test_gain_reduction_hard_knee_above_threshold() {
        let c = make_compressor();
        // input_db = -10, threshold = -20, ratio = 4
        // reduction = 10 * (1 - 1/4) = 7.5 dB
        let gr = c.compute_gain_reduction_db(-10.0);
        assert!((gr - 7.5).abs() < 1e-4);
    }

    #[test]
    fn test_soft_knee_within_knee() {
        let config = CompressorConfig {
            knee: KneeMode::Soft(10.0),
            ..CompressorConfig::default()
        };
        let c = Compressor::new(config);
        // At threshold (boundary), reduction should be positive
        let gr = c.compute_gain_reduction_db(-20.0);
        assert!(gr >= 0.0);
    }

    #[test]
    fn test_process_buffer_modifies_samples() {
        let mut c = make_compressor();
        let mut buf = vec![0.5_f32; 1000];
        c.process_buffer(&mut buf);
        // Samples should still be valid (not NaN/inf)
        for s in &buf {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_reset_clears_envelope() {
        let mut c = make_compressor();
        for _ in 0..500 {
            c.process_sample(1.0);
        }
        assert!(c.envelope > 0.0);
        c.reset();
        assert_eq!(c.envelope, 0.0);
    }

    #[test]
    fn test_time_to_coeff_zero_time() {
        let coeff = Compressor::time_to_coeff(0.0, 48_000.0);
        assert_eq!(coeff, 0.0);
    }

    #[test]
    fn test_time_to_coeff_positive() {
        let coeff = Compressor::time_to_coeff(0.01, 48_000.0);
        assert!(coeff > 0.0 && coeff < 1.0);
    }

    #[test]
    fn test_set_attack_updates_coeff() {
        let mut c = make_compressor();
        let old_coeff = c.attack_coeff;
        c.set_attack(0.001);
        assert_ne!(c.attack_coeff, old_coeff);
    }

    #[test]
    fn test_set_ratio_clamps_to_one() {
        let mut c = make_compressor();
        c.set_ratio(0.5);
        assert_eq!(c.config.ratio, 1.0);
    }

    #[test]
    fn test_makeup_gain_increases_output() {
        let config = CompressorConfig {
            makeup_gain_db: 6.0,
            threshold_db: 0.0, // no compression
            ..CompressorConfig::default()
        };
        let mut c = Compressor::new(config);
        // Run many samples to let envelope stabilize
        for _ in 0..10000 {
            c.process_sample(0.1);
        }
        let out = c.process_sample(0.1);
        // 6 dB makeup ~ factor of 2
        assert!(out > 0.1);
    }

    #[test]
    fn test_gain_reduction_accessor() {
        let mut c = make_compressor();
        for _ in 0..1000 {
            c.process_sample(1.0);
        }
        // After signal above threshold, gain reduction should be positive
        let gr = c.gain_reduction_db();
        assert!(gr >= 0.0);
    }

    // --- Look-ahead delay tests ---

    #[test]
    fn test_lookahead_disabled_by_default() {
        let c = make_compressor();
        assert_eq!(c.lookahead_samples(), 0);
    }

    #[test]
    fn test_lookahead_creates_delay() {
        let config = CompressorConfig {
            lookahead_secs: 0.005, // 5ms
            sample_rate: 48_000.0,
            ..CompressorConfig::default()
        };
        let c = Compressor::new(config);
        // 5ms at 48 kHz = 240 samples
        assert_eq!(c.lookahead_samples(), 240);
    }

    #[test]
    fn test_lookahead_output_is_delayed() {
        let config = CompressorConfig {
            lookahead_secs: 0.001, // 1ms = 48 samples
            threshold_db: 0.0,     // high threshold so no compression
            sample_rate: 48_000.0,
            ..CompressorConfig::default()
        };
        let mut c = Compressor::new(config);
        let delay = c.lookahead_samples();
        assert!(delay > 0, "should have non-zero look-ahead");

        // First `delay` outputs should be zeros (from buffer init)
        for i in 0..delay {
            let out = c.process_sample(1.0);
            assert!(
                out.abs() < 1e-6,
                "sample {i} should be zero (from delay), got {out}"
            );
        }

        // After the delay period, signal should pass through
        let out = c.process_sample(1.0);
        assert!(out.abs() > 0.5, "signal should appear after delay");
    }

    #[test]
    fn test_lookahead_output_is_finite() {
        let config = CompressorConfig {
            lookahead_secs: 0.002,
            ..CompressorConfig::default()
        };
        let mut c = Compressor::new(config);
        for _ in 0..5000 {
            let out = c.process_sample(0.7);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn test_set_lookahead_runtime() {
        let mut c = make_compressor();
        assert_eq!(c.lookahead_samples(), 0);
        c.set_lookahead(0.01); // 10ms
        assert_eq!(c.lookahead_samples(), 480);
    }

    // --- Sidechain tests ---

    #[test]
    fn test_sidechain_compresses_programme_when_key_loud() {
        // Threshold at -20 dBFS, ratio 4:1
        let config = CompressorConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_secs: 0.001,
            release_secs: 0.05,
            makeup_gain_db: 0.0,
            sample_rate: 48_000.0,
            ..CompressorConfig::default()
        };
        let mut c = Compressor::new(config);

        // Key signal at 0 dBFS — well above threshold
        let key = vec![1.0_f32; 10_000];
        // Programme signal at moderate level
        let mut prog = vec![0.5_f32; 10_000];
        c.process_buffer_sidechain(&mut prog, &key);

        // After attack settles, programme should be reduced
        let tail_max = prog[5_000..].iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            tail_max < 0.5,
            "Sidechain should reduce programme; tail_max={tail_max}"
        );
    }

    #[test]
    fn test_sidechain_no_compression_when_key_silent() {
        let config = CompressorConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            makeup_gain_db: 0.0,
            sample_rate: 48_000.0,
            ..CompressorConfig::default()
        };
        let mut c = Compressor::new(config);

        // Key signal is silence — below threshold
        let key = vec![0.0_f32; 5_000];
        let mut prog = vec![0.1_f32; 5_000]; // low level programme signal
        c.process_buffer_sidechain(&mut prog, &key);

        // With key silent, no compression — output ≈ input
        let tail_avg: f32 = prog[4_000..].iter().sum::<f32>() / 1_000.0;
        assert!(
            (tail_avg - 0.1).abs() < 0.02,
            "No compression without key; tail_avg={tail_avg}"
        );
    }

    #[test]
    fn test_sidechain_output_is_finite() {
        let mut c = make_compressor();
        for i in 0..2000_usize {
            let key = (i as f32 * 0.01).sin();
            let prog = (i as f32 * 0.007).sin() * 0.5;
            let out = c.process_sample_sidechain(prog, key);
            assert!(out.is_finite(), "sidechain output must be finite at {i}");
        }
    }

    #[test]
    fn test_lookahead_reset_clears_buffer() {
        let config = CompressorConfig {
            lookahead_secs: 0.005,
            ..CompressorConfig::default()
        };
        let mut c = Compressor::new(config);
        for _ in 0..500 {
            c.process_sample(0.9);
        }
        c.reset();
        assert_eq!(c.lookahead_write_pos, 0);
        for &s in &c.lookahead_buffer {
            assert_eq!(s, 0.0);
        }
    }
}
