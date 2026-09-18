#![allow(dead_code)]
//! RMS envelope follower with configurable time constants.
//!
//! Computes a smoothed RMS (Root Mean Square) envelope of an audio signal
//! using configurable attack and release time constants. Useful for level
//! metering, dynamics visualization, and broadcast loudness monitoring.
//! Supports both per-channel and downmixed operation modes.
//!
//! # Relationship to the main metering pipeline
//!
//! This module is independent of the [`crate::ebu_r128_impl::EbuR128Meter`]
//! filter → LKFS → gating → LRA pipeline (see the crate-level docs, "Pipeline
//! Architecture") — it shares no state with it and applies no K-weighting.
//! Where the Momentary (400 ms) and Short-term (3 s) LKFS windows have fixed,
//! standards-mandated durations, this envelope follower's attack/release times
//! are freely configurable, making it suited to VU-style ballistics and
//! dynamics visualisation rather than standards-accurate loudness. Run it
//! alongside [`crate::LoudnessMeter`] on the same audio buffer. See also
//! [`crate::clip_counter`] and [`crate::silence_detect`] for the other two
//! complementary, independently-run QC modules.

/// Configuration for the RMS envelope follower.
#[derive(Clone, Debug)]
pub struct RmsEnvelopeConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of audio channels.
    pub channels: usize,
    /// Attack time constant in seconds (how fast the envelope rises).
    pub attack_time: f64,
    /// Release time constant in seconds (how fast the envelope falls).
    pub release_time: f64,
    /// RMS window size in samples for the initial RMS calculation.
    pub window_size: usize,
}

impl RmsEnvelopeConfig {
    /// Create a new configuration.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels: channels.max(1),
            attack_time: 0.005,
            release_time: 0.050,
            window_size: 1024,
        }
    }

    /// Set attack time in seconds.
    pub fn with_attack(mut self, seconds: f64) -> Self {
        self.attack_time = seconds.max(0.0001);
        self
    }

    /// Set release time in seconds.
    pub fn with_release(mut self, seconds: f64) -> Self {
        self.release_time = seconds.max(0.0001);
        self
    }

    /// Set the RMS window size in samples.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(1);
        self
    }

    /// Compute the attack coefficient from the time constant.
    fn attack_coeff(&self) -> f64 {
        if self.sample_rate <= 0.0 {
            return 0.0;
        }
        (-1.0 / (self.attack_time * self.sample_rate)).exp()
    }

    /// Compute the release coefficient from the time constant.
    fn release_coeff(&self) -> f64 {
        if self.sample_rate <= 0.0 {
            return 0.0;
        }
        (-1.0 / (self.release_time * self.sample_rate)).exp()
    }
}

/// Per-channel envelope state.
#[derive(Clone, Debug)]
struct ChannelEnvelopeState {
    /// Current squared envelope value.
    envelope_sq: f64,
    /// Ring buffer for running RMS computation.
    ring_buffer: Vec<f64>,
    /// Write position in the ring buffer.
    write_pos: usize,
    /// Running sum of squared samples in the window.
    running_sum_sq: f64,
    /// Number of valid samples in the buffer.
    valid_count: usize,
    /// Peak envelope value seen.
    peak_rms: f64,
}

impl ChannelEnvelopeState {
    /// Create a new channel state.
    fn new(window_size: usize) -> Self {
        Self {
            envelope_sq: 0.0,
            ring_buffer: vec![0.0; window_size],
            write_pos: 0,
            running_sum_sq: 0.0,
            valid_count: 0,
            peak_rms: 0.0,
        }
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.envelope_sq = 0.0;
        self.running_sum_sq = 0.0;
        self.valid_count = 0;
        self.write_pos = 0;
        self.peak_rms = 0.0;
        for v in &mut self.ring_buffer {
            *v = 0.0;
        }
    }
}

/// RMS envelope follower.
///
/// Tracks the RMS level of audio signals with smooth attack/release behavior,
/// suitable for metering displays and level-dependent processing.
#[derive(Clone, Debug)]
pub struct RmsEnvelopeFollower {
    /// Configuration.
    config: RmsEnvelopeConfig,
    /// Attack smoothing coefficient.
    attack_coeff: f64,
    /// Release smoothing coefficient.
    release_coeff: f64,
    /// Per-channel state.
    channel_states: Vec<ChannelEnvelopeState>,
    /// Total samples processed (per channel).
    samples_processed: u64,
}

impl RmsEnvelopeFollower {
    /// Create a new RMS envelope follower.
    pub fn new(config: RmsEnvelopeConfig) -> Self {
        let attack_coeff = config.attack_coeff();
        let release_coeff = config.release_coeff();
        let channels = config.channels;
        let window_size = config.window_size;
        Self {
            config,
            attack_coeff,
            release_coeff,
            channel_states: (0..channels)
                .map(|_| ChannelEnvelopeState::new(window_size))
                .collect(),
            samples_processed: 0,
        }
    }

    /// Process interleaved audio samples.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frame_count = samples.len() / self.config.channels;
        for frame in 0..frame_count {
            for ch in 0..self.config.channels {
                let sample = samples[frame * self.config.channels + ch];
                self.process_sample(ch, sample);
            }
            self.samples_processed += 1;
        }
    }

    /// Process a single sample on a specific channel.
    fn process_sample(&mut self, channel: usize, sample: f64) {
        let state = &mut self.channel_states[channel];
        let sq = sample * sample;
        let window_size = state.ring_buffer.len();

        // Update running sum: subtract old, add new
        let old_sq = state.ring_buffer[state.write_pos];
        state.running_sum_sq += sq - old_sq;
        // Prevent negative from floating-point drift
        if state.running_sum_sq < 0.0 {
            state.running_sum_sq = 0.0;
        }
        state.ring_buffer[state.write_pos] = sq;
        state.write_pos = (state.write_pos + 1) % window_size;
        if state.valid_count < window_size {
            state.valid_count += 1;
        }

        // Compute instantaneous RMS^2 from the window
        #[allow(clippy::cast_precision_loss)]
        let instant_rms_sq = if state.valid_count > 0 {
            state.running_sum_sq / state.valid_count as f64
        } else {
            0.0
        };

        // Apply attack/release envelope
        let coeff = if instant_rms_sq > state.envelope_sq {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        state.envelope_sq = coeff * state.envelope_sq + (1.0 - coeff) * instant_rms_sq;

        // Track peak
        let current_rms = state.envelope_sq.sqrt();
        if current_rms > state.peak_rms {
            state.peak_rms = current_rms;
        }
    }

    /// Get the current RMS level for a channel (linear scale).
    pub fn rms_linear(&self, channel: usize) -> f64 {
        self.channel_states
            .get(channel)
            .map_or(0.0, |s| s.envelope_sq.sqrt())
    }

    /// Get the current RMS level for a channel in dBFS.
    pub fn rms_dbfs(&self, channel: usize) -> f64 {
        let linear = self.rms_linear(channel);
        if linear <= 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * linear.log10()
        }
    }

    /// Get peak RMS seen for a channel (linear scale).
    pub fn peak_rms_linear(&self, channel: usize) -> f64 {
        self.channel_states.get(channel).map_or(0.0, |s| s.peak_rms)
    }

    /// Get peak RMS for a channel in dBFS.
    pub fn peak_rms_dbfs(&self, channel: usize) -> f64 {
        let linear = self.peak_rms_linear(channel);
        if linear <= 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * linear.log10()
        }
    }

    /// Get all channel RMS levels in dBFS.
    pub fn all_rms_dbfs(&self) -> Vec<f64> {
        (0..self.config.channels)
            .map(|ch| self.rms_dbfs(ch))
            .collect()
    }

    /// Get the maximum RMS across all channels (linear).
    pub fn max_rms_linear(&self) -> f64 {
        (0..self.config.channels)
            .map(|ch| self.rms_linear(ch))
            .fold(0.0_f64, f64::max)
    }

    /// Get total samples processed (per channel).
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }

    /// Reset the envelope follower.
    pub fn reset(&mut self) {
        for state in &mut self.channel_states {
            state.reset();
        }
        self.samples_processed = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_follower(channels: usize) -> RmsEnvelopeFollower {
        let config = RmsEnvelopeConfig::new(48000.0, channels)
            .with_attack(0.001)
            .with_release(0.010)
            .with_window_size(256);
        RmsEnvelopeFollower::new(config)
    }

    #[test]
    fn test_config_defaults() {
        let config = RmsEnvelopeConfig::new(48000.0, 2);
        assert!((config.attack_time - 0.005).abs() < 1e-12);
        assert!((config.release_time - 0.050).abs() < 1e-12);
        assert_eq!(config.window_size, 1024);
    }

    #[test]
    fn test_config_builder() {
        let config = RmsEnvelopeConfig::new(44100.0, 1)
            .with_attack(0.01)
            .with_release(0.1)
            .with_window_size(512);
        assert!((config.attack_time - 0.01).abs() < 1e-12);
        assert!((config.release_time - 0.1).abs() < 1e-12);
        assert_eq!(config.window_size, 512);
    }

    #[test]
    fn test_silence_gives_neg_infinity() {
        let follower = make_follower(1);
        assert!(follower.rms_dbfs(0).is_infinite());
        assert!(follower.rms_dbfs(0) < 0.0);
    }

    #[test]
    fn test_full_scale_sine() {
        let mut follower = make_follower(1);
        // Generate enough full-scale sine to fill the window and stabilize
        let samples: Vec<f64> = (0..10000)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();
        for ch_samples in samples.chunks(1) {
            follower.process_interleaved(ch_samples);
        }
        let rms = follower.rms_linear(0);
        // Sine wave RMS = 1/sqrt(2) ~ 0.707
        assert!(rms > 0.5, "RMS should be significant for full-scale sine");
        assert!(rms < 0.85, "RMS should be near 0.707 for sine");
    }

    #[test]
    fn test_dc_signal() {
        let mut follower = make_follower(1);
        let samples = vec![0.5; 5000];
        for s in &samples {
            follower.process_interleaved(&[*s]);
        }
        let rms = follower.rms_linear(0);
        // DC 0.5 -> RMS should approach 0.5
        assert!(
            (rms - 0.5).abs() < 0.1,
            "RMS for DC 0.5 should be near 0.5, got {rms}"
        );
    }

    #[test]
    fn test_peak_rms_tracking() {
        let mut follower = make_follower(1);
        // Process loud then quiet
        let loud = vec![0.8; 2000];
        let quiet = vec![0.01; 2000];
        for s in &loud {
            follower.process_interleaved(&[*s]);
        }
        for s in &quiet {
            follower.process_interleaved(&[*s]);
        }
        let peak = follower.peak_rms_linear(0);
        let current = follower.rms_linear(0);
        assert!(
            peak > current,
            "Peak should be higher than current after going quiet"
        );
    }

    #[test]
    fn test_multichannel() {
        let mut follower = make_follower(2);
        // Left channel loud, right channel quiet
        let mut samples = Vec::new();
        for _ in 0..5000 {
            samples.push(0.8); // L
            samples.push(0.1); // R
        }
        follower.process_interleaved(&samples);
        let left = follower.rms_linear(0);
        let right = follower.rms_linear(1);
        assert!(left > right * 2.0, "Left should be much louder than right");
    }

    #[test]
    fn test_all_rms_dbfs() {
        let mut follower = make_follower(2);
        let samples = vec![0.5; 10000]; // 5000 frames of stereo
        follower.process_interleaved(&samples);
        let levels = follower.all_rms_dbfs();
        assert_eq!(levels.len(), 2);
        // Both channels fed same data => similar levels
        assert!((levels[0] - levels[1]).abs() < 1.0);
    }

    #[test]
    fn test_max_rms_linear() {
        let mut follower = make_follower(2);
        let mut samples = Vec::new();
        for _ in 0..5000 {
            samples.push(0.9); // L
            samples.push(0.1); // R
        }
        follower.process_interleaved(&samples);
        let max = follower.max_rms_linear();
        let left = follower.rms_linear(0);
        assert!(
            (max - left).abs() < 0.01,
            "Max should match the louder channel"
        );
    }

    #[test]
    fn test_reset() {
        let mut follower = make_follower(1);
        follower.process_interleaved(&vec![0.5; 2000]);
        assert!(follower.rms_linear(0) > 0.0);
        follower.reset();
        assert!(follower.rms_linear(0) < 1e-12);
        assert_eq!(follower.samples_processed(), 0);
    }

    #[test]
    fn test_samples_processed() {
        let mut follower = make_follower(2);
        follower.process_interleaved(&vec![0.5; 200]); // 100 frames
        assert_eq!(follower.samples_processed(), 100);
    }

    #[test]
    fn test_dbfs_conversion() {
        let mut follower = make_follower(1);
        // Process a constant 1.0 signal
        follower.process_interleaved(&vec![1.0; 5000]);
        let dbfs = follower.rms_dbfs(0);
        // Should be close to 0 dBFS
        assert!(
            dbfs > -3.0,
            "Full-scale DC should be near 0 dBFS, got {dbfs}"
        );
    }

    #[test]
    fn test_invalid_channel() {
        let follower = make_follower(1);
        // Accessing a non-existent channel returns 0
        assert!((follower.rms_linear(99)).abs() < 1e-12);
    }
}
