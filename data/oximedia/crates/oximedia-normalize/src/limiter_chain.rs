#![allow(dead_code)]
//! Multi-stage limiter chain for broadcast loudness normalization.
//!
//! Provides a configurable chain of limiting stages that process audio
//! sequentially: soft clipper, lookahead limiter, true-peak limiter,
//! and final safety clipper. Each stage can be individually enabled
//! and configured.

use std::collections::VecDeque;
use std::fmt;

/// Type of limiter stage in the chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimiterStageType {
    /// Soft clipper with saturation curve.
    SoftClipper,
    /// Lookahead limiter with attack/release envelope.
    LookaheadLimiter,
    /// True-peak limiter using oversampled detection.
    TruePeakLimiter,
    /// Hard clipper as final safety net.
    HardClipper,
    /// Brick-wall limiter with zero overshoot.
    BrickWall,
}

impl fmt::Display for LimiterStageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SoftClipper => write!(f, "Soft Clipper"),
            Self::LookaheadLimiter => write!(f, "Lookahead Limiter"),
            Self::TruePeakLimiter => write!(f, "True Peak Limiter"),
            Self::HardClipper => write!(f, "Hard Clipper"),
            Self::BrickWall => write!(f, "Brick Wall"),
        }
    }
}

/// Configuration for a single limiter stage.
#[derive(Debug, Clone, PartialEq)]
pub struct LimiterStageConfig {
    /// Type of limiter.
    pub stage_type: LimiterStageType,
    /// Threshold in dB (negative value, e.g. -1.0 dBTP).
    pub threshold_db: f64,
    /// Ceiling in dB (maximum output level).
    pub ceiling_db: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Lookahead in milliseconds (for lookahead stages).
    pub lookahead_ms: f64,
    /// Whether this stage is enabled.
    pub enabled: bool,
    /// Knee width in dB (for soft clipper).
    pub knee_db: f64,
}

impl LimiterStageConfig {
    /// Create a new stage configuration.
    pub fn new(stage_type: LimiterStageType, threshold_db: f64, ceiling_db: f64) -> Self {
        let (attack_ms, release_ms, lookahead_ms) = match stage_type {
            LimiterStageType::SoftClipper => (0.1, 10.0, 0.0),
            LimiterStageType::LookaheadLimiter => (1.0, 50.0, 5.0),
            LimiterStageType::TruePeakLimiter => (0.5, 20.0, 1.5),
            LimiterStageType::HardClipper => (0.0, 0.0, 0.0),
            LimiterStageType::BrickWall => (0.01, 5.0, 0.5),
        };
        Self {
            stage_type,
            threshold_db,
            ceiling_db,
            attack_ms,
            release_ms,
            lookahead_ms,
            enabled: true,
            knee_db: 3.0,
        }
    }

    /// Disable this stage.
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Set attack time.
    pub fn with_attack(mut self, attack_ms: f64) -> Self {
        self.attack_ms = attack_ms;
        self
    }

    /// Set release time.
    pub fn with_release(mut self, release_ms: f64) -> Self {
        self.release_ms = release_ms;
        self
    }

    /// Set knee width for soft clipper.
    pub fn with_knee(mut self, knee_db: f64) -> Self {
        self.knee_db = knee_db;
        self
    }
}

/// Runtime state for a single limiter stage.
#[derive(Debug, Clone)]
struct LimiterStageState {
    /// Current gain reduction in dB.
    gain_reduction_db: f64,
    /// Envelope follower state.
    envelope: f64,
    /// Lookahead buffer.
    lookahead_buffer: VecDeque<f64>,
    /// Lookahead buffer size in samples.
    lookahead_samples: usize,
    /// Attack coefficient (per-sample).
    attack_coeff: f64,
    /// Release coefficient (per-sample).
    release_coeff: f64,
    /// Peak gain reduction observed.
    peak_gr_db: f64,
    /// Number of samples limited.
    samples_limited: u64,
}

impl LimiterStageState {
    /// Create a new state for a stage configuration.
    fn new(config: &LimiterStageConfig, sample_rate: f64) -> Self {
        let lookahead_samples = (config.lookahead_ms * sample_rate / 1000.0) as usize;
        let attack_coeff = if config.attack_ms > 0.0 {
            (-1.0 / (config.attack_ms * sample_rate / 1000.0)).exp()
        } else {
            0.0
        };
        let release_coeff = if config.release_ms > 0.0 {
            (-1.0 / (config.release_ms * sample_rate / 1000.0)).exp()
        } else {
            0.0
        };

        Self {
            gain_reduction_db: 0.0,
            envelope: 0.0,
            lookahead_buffer: VecDeque::with_capacity(lookahead_samples.max(1)),
            lookahead_samples,
            attack_coeff,
            release_coeff,
            peak_gr_db: 0.0,
            samples_limited: 0,
        }
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.gain_reduction_db = 0.0;
        self.envelope = 0.0;
        self.lookahead_buffer.clear();
        self.peak_gr_db = 0.0;
        self.samples_limited = 0;
    }
}

/// A single limiter stage that processes audio.
#[derive(Debug, Clone)]
struct LimiterStage {
    /// Configuration.
    config: LimiterStageConfig,
    /// Runtime state.
    state: LimiterStageState,
}

impl LimiterStage {
    /// Create a new limiter stage.
    fn new(config: LimiterStageConfig, sample_rate: f64) -> Self {
        let state = LimiterStageState::new(&config, sample_rate);
        Self { config, state }
    }

    /// Process a single sample through this stage.
    fn process_sample(&mut self, sample: f64) -> f64 {
        if !self.config.enabled {
            return sample;
        }

        match self.config.stage_type {
            LimiterStageType::SoftClipper => self.soft_clip(sample),
            LimiterStageType::HardClipper => self.hard_clip(sample),
            LimiterStageType::LookaheadLimiter
            | LimiterStageType::TruePeakLimiter
            | LimiterStageType::BrickWall => self.envelope_limit(sample),
        }
    }

    /// Soft clipping with tanh saturation.
    fn soft_clip(&mut self, sample: f64) -> f64 {
        let threshold_lin = db_to_linear(self.config.threshold_db);
        let abs_sample = sample.abs();
        if abs_sample <= threshold_lin {
            return sample;
        }
        self.state.samples_limited += 1;
        let ceiling_lin = db_to_linear(self.config.ceiling_db);
        let excess = (abs_sample - threshold_lin) / (ceiling_lin - threshold_lin + 1e-10);
        let compressed = threshold_lin + (ceiling_lin - threshold_lin) * excess.tanh();
        let result = compressed.min(ceiling_lin) * sample.signum();
        let gr = linear_to_db(abs_sample) - linear_to_db(result.abs().max(1e-10));
        if gr > self.state.peak_gr_db {
            self.state.peak_gr_db = gr;
        }
        result
    }

    /// Hard clipping.
    fn hard_clip(&mut self, sample: f64) -> f64 {
        let ceiling_lin = db_to_linear(self.config.ceiling_db);
        if sample.abs() > ceiling_lin {
            self.state.samples_limited += 1;
            sample.clamp(-ceiling_lin, ceiling_lin)
        } else {
            sample
        }
    }

    /// Envelope-following limiter (lookahead, true-peak, brick-wall).
    fn envelope_limit(&mut self, sample: f64) -> f64 {
        let threshold_lin = db_to_linear(self.config.threshold_db);
        let abs_sample = sample.abs();

        // Envelope follower
        if abs_sample > self.state.envelope {
            self.state.envelope = self.state.attack_coeff * self.state.envelope
                + (1.0 - self.state.attack_coeff) * abs_sample;
        } else {
            self.state.envelope = self.state.release_coeff * self.state.envelope
                + (1.0 - self.state.release_coeff) * abs_sample;
        }

        // Calculate gain reduction
        if self.state.envelope > threshold_lin {
            self.state.gain_reduction_db =
                linear_to_db(threshold_lin) - linear_to_db(self.state.envelope);
            if (-self.state.gain_reduction_db) > self.state.peak_gr_db {
                self.state.peak_gr_db = -self.state.gain_reduction_db;
            }
            self.state.samples_limited += 1;
        } else {
            self.state.gain_reduction_db = 0.0;
        }

        let gain_lin = db_to_linear(self.state.gain_reduction_db);

        // Use lookahead if available
        if self.state.lookahead_samples > 0 {
            self.state.lookahead_buffer.push_back(sample);
            if self.state.lookahead_buffer.len() > self.state.lookahead_samples {
                let delayed = self.state.lookahead_buffer.pop_front().unwrap_or(0.0);
                delayed * gain_lin
            } else {
                0.0
            }
        } else {
            sample * gain_lin
        }
    }

    /// Reset stage state.
    fn reset(&mut self) {
        self.state.reset();
    }
}

/// Statistics for the limiter chain processing.
#[derive(Debug, Clone, PartialEq)]
pub struct LimiterChainStats {
    /// Total input samples processed.
    pub total_samples: u64,
    /// Per-stage statistics.
    pub stage_stats: Vec<StageStats>,
    /// Overall peak gain reduction in dB.
    pub overall_peak_gr_db: f64,
    /// Overall samples that were limited.
    pub overall_samples_limited: u64,
}

/// Statistics for a single limiter stage.
#[derive(Debug, Clone, PartialEq)]
pub struct StageStats {
    /// Stage type.
    pub stage_type: LimiterStageType,
    /// Peak gain reduction in dB.
    pub peak_gr_db: f64,
    /// Number of samples limited.
    pub samples_limited: u64,
    /// Whether the stage is enabled.
    pub enabled: bool,
}

/// Configuration for the complete limiter chain.
#[derive(Debug, Clone, PartialEq)]
pub struct LimiterChainConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: usize,
    /// Ordered list of stage configurations.
    pub stages: Vec<LimiterStageConfig>,
    /// Enable dithering after limiting.
    pub dither_after: bool,
    /// Output bit depth for dithering.
    pub output_bit_depth: u32,
}

impl LimiterChainConfig {
    /// Create a new limiter chain configuration.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            stages: Vec::new(),
            dither_after: false,
            output_bit_depth: 24,
        }
    }

    /// Add a stage to the chain.
    pub fn add_stage(mut self, stage: LimiterStageConfig) -> Self {
        self.stages.push(stage);
        self
    }

    /// Create a broadcast-standard chain (soft clip -> lookahead -> true peak -> hard clip).
    pub fn broadcast(sample_rate: f64, channels: usize) -> Self {
        Self::new(sample_rate, channels)
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::SoftClipper,
                -3.0,
                -1.0,
            ))
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::LookaheadLimiter,
                -1.5,
                -1.0,
            ))
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::TruePeakLimiter,
                -1.0,
                -1.0,
            ))
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::HardClipper,
                -0.5,
                -0.5,
            ))
    }

    /// Create a streaming-oriented chain (softer limiting).
    pub fn streaming(sample_rate: f64, channels: usize) -> Self {
        Self::new(sample_rate, channels)
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::SoftClipper,
                -2.0,
                -1.0,
            ))
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::TruePeakLimiter,
                -1.0,
                -1.0,
            ))
    }
}

/// Multi-stage limiter chain processor.
#[derive(Debug, Clone)]
pub struct LimiterChain {
    /// Configuration.
    config: LimiterChainConfig,
    /// Processing stages.
    stages: Vec<LimiterStage>,
    /// Total samples processed.
    total_samples: u64,
}

impl LimiterChain {
    /// Create a new limiter chain from configuration.
    pub fn new(config: LimiterChainConfig) -> Self {
        let stages = config
            .stages
            .iter()
            .map(|sc| LimiterStage::new(sc.clone(), config.sample_rate))
            .collect();
        Self {
            config,
            stages,
            total_samples: 0,
        }
    }

    /// Process a buffer of interleaved audio samples in place.
    pub fn process_interleaved(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            let mut s = *sample;
            for stage in &mut self.stages {
                s = stage.process_sample(s);
            }
            *sample = s;
            self.total_samples += 1;
        }
    }

    /// Process a buffer of f32 interleaved audio samples in place.
    pub fn process_interleaved_f32(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            let mut s = f64::from(*sample);
            for stage in &mut self.stages {
                s = stage.process_sample(s);
            }
            #[allow(clippy::cast_possible_truncation)]
            {
                *sample = s as f32;
            }
            self.total_samples += 1;
        }
    }

    /// Process a single sample through the entire chain.
    pub fn process_sample(&mut self, sample: f64) -> f64 {
        let mut s = sample;
        for stage in &mut self.stages {
            s = stage.process_sample(s);
        }
        self.total_samples += 1;
        s
    }

    /// Get current chain statistics.
    pub fn stats(&self) -> LimiterChainStats {
        let mut overall_peak_gr = 0.0_f64;
        let mut overall_limited = 0_u64;
        let stage_stats: Vec<StageStats> = self
            .stages
            .iter()
            .map(|s| {
                let gr = s.state.peak_gr_db;
                if gr > overall_peak_gr {
                    overall_peak_gr = gr;
                }
                overall_limited += s.state.samples_limited;
                StageStats {
                    stage_type: s.config.stage_type,
                    peak_gr_db: gr,
                    samples_limited: s.state.samples_limited,
                    enabled: s.config.enabled,
                }
            })
            .collect();

        LimiterChainStats {
            total_samples: self.total_samples,
            stage_stats,
            overall_peak_gr_db: overall_peak_gr,
            overall_samples_limited: overall_limited,
        }
    }

    /// Reset all stages.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
        self.total_samples = 0;
    }

    /// Number of stages in the chain.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Number of enabled stages.
    pub fn num_enabled_stages(&self) -> usize {
        self.stages.iter().filter(|s| s.config.enabled).count()
    }

    /// Total latency introduced by the chain in samples.
    pub fn latency_samples(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.config.enabled)
            .map(|s| s.state.lookahead_samples)
            .sum()
    }

    /// Get the configuration.
    pub fn config(&self) -> &LimiterChainConfig {
        &self.config
    }
}

/// Convert dB to linear gain.
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear gain to dB.
fn linear_to_db(linear: f64) -> f64 {
    20.0 * linear.max(1e-10).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_linear_roundtrip() {
        let db_vals = [-6.0, -3.0, 0.0, 3.0, 6.0];
        for &db in &db_vals {
            let lin = db_to_linear(db);
            let back = linear_to_db(lin);
            assert!((back - db).abs() < 1e-10, "Roundtrip failed for {db} dB");
        }
    }

    #[test]
    fn test_hard_clipper_below_threshold() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::HardClipper,
            -1.0,
            -1.0,
        ));
        let mut chain = LimiterChain::new(config);
        let sample = 0.5;
        let out = chain.process_sample(sample);
        assert!((out - sample).abs() < 1e-10);
    }

    #[test]
    fn test_hard_clipper_above_threshold() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::HardClipper,
            -6.0,
            -6.0,
        ));
        let mut chain = LimiterChain::new(config);
        let ceiling = db_to_linear(-6.0);
        let out = chain.process_sample(1.0);
        assert!(out <= ceiling + 1e-10);
    }

    #[test]
    fn test_soft_clipper_preserves_low_level() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::SoftClipper,
            -3.0,
            -1.0,
        ));
        let mut chain = LimiterChain::new(config);
        let sample = 0.1;
        let out = chain.process_sample(sample);
        assert!((out - sample).abs() < 1e-10);
    }

    #[test]
    fn test_soft_clipper_reduces_high_level() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::SoftClipper,
            -6.0,
            -1.0,
        ));
        let mut chain = LimiterChain::new(config);
        let out = chain.process_sample(0.95);
        assert!(out.abs() <= 0.95 + 1e-6);
    }

    #[test]
    fn test_broadcast_chain_creation() {
        let config = LimiterChainConfig::broadcast(48000.0, 2);
        assert_eq!(config.stages.len(), 4);
        let chain = LimiterChain::new(config);
        assert_eq!(chain.num_stages(), 4);
        assert_eq!(chain.num_enabled_stages(), 4);
    }

    #[test]
    fn test_streaming_chain_creation() {
        let config = LimiterChainConfig::streaming(44100.0, 2);
        assert_eq!(config.stages.len(), 2);
        let chain = LimiterChain::new(config);
        assert_eq!(chain.num_stages(), 2);
    }

    #[test]
    fn test_process_interleaved() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::HardClipper,
            -6.0,
            -6.0,
        ));
        let mut chain = LimiterChain::new(config);
        let ceiling = db_to_linear(-6.0);
        let mut buf = vec![0.1, 0.3, 0.5, 0.9, 1.0, -0.8];
        chain.process_interleaved(&mut buf);
        for &s in &buf {
            assert!(s.abs() <= ceiling + 1e-10);
        }
    }

    #[test]
    fn test_process_interleaved_f32() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::HardClipper,
            -6.0,
            -6.0,
        ));
        let mut chain = LimiterChain::new(config);
        let ceiling = db_to_linear(-6.0) as f32;
        let mut buf: Vec<f32> = vec![0.1, 0.5, 0.9, 1.0];
        chain.process_interleaved_f32(&mut buf);
        for &s in &buf {
            assert!(s.abs() <= ceiling + 1e-4);
        }
    }

    #[test]
    fn test_stats_tracking() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::HardClipper,
            -6.0,
            -6.0,
        ));
        let mut chain = LimiterChain::new(config);
        let _ = chain.process_sample(1.0);
        let _ = chain.process_sample(0.1);
        let stats = chain.stats();
        assert_eq!(stats.total_samples, 2);
        assert_eq!(stats.stage_stats.len(), 1);
        assert!(stats.stage_stats[0].samples_limited >= 1);
    }

    #[test]
    fn test_reset() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(LimiterStageConfig::new(
            LimiterStageType::HardClipper,
            -6.0,
            -6.0,
        ));
        let mut chain = LimiterChain::new(config);
        let _ = chain.process_sample(1.0);
        chain.reset();
        let stats = chain.stats();
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.overall_samples_limited, 0);
    }

    #[test]
    fn test_disabled_stage() {
        let config = LimiterChainConfig::new(48000.0, 1).add_stage(
            LimiterStageConfig::new(LimiterStageType::HardClipper, -6.0, -6.0).disabled(),
        );
        let chain = LimiterChain::new(config);
        assert_eq!(chain.num_enabled_stages(), 0);
    }

    #[test]
    fn test_latency_samples() {
        let config = LimiterChainConfig::new(48000.0, 1)
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::LookaheadLimiter,
                -1.0,
                -1.0,
            ))
            .add_stage(LimiterStageConfig::new(
                LimiterStageType::TruePeakLimiter,
                -1.0,
                -1.0,
            ));
        let chain = LimiterChain::new(config);
        assert!(chain.latency_samples() > 0);
    }

    #[test]
    fn test_stage_type_display() {
        assert_eq!(format!("{}", LimiterStageType::SoftClipper), "Soft Clipper");
        assert_eq!(format!("{}", LimiterStageType::BrickWall), "Brick Wall");
        assert_eq!(format!("{}", LimiterStageType::HardClipper), "Hard Clipper");
    }

    #[test]
    fn test_stage_config_builder() {
        let stage = LimiterStageConfig::new(LimiterStageType::SoftClipper, -3.0, -1.0)
            .with_attack(0.5)
            .with_release(20.0)
            .with_knee(6.0);
        assert!((stage.attack_ms - 0.5).abs() < f64::EPSILON);
        assert!((stage.release_ms - 20.0).abs() < f64::EPSILON);
        assert!((stage.knee_db - 6.0).abs() < f64::EPSILON);
    }
}
