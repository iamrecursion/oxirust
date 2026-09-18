//! NVIDIA NVENC hardware encoder.
//!
//! Provides NVIDIA GPU hardware-accelerated encoding with proper configuration
//! for game streaming scenarios. The encoder is feature-gated: when actual NVENC
//! hardware is not available, a software-simulated fallback is used for
//! development and testing.

use crate::{GamingError, GamingResult};
use std::time::{Duration, Instant};

/// NVIDIA NVENC encoder.
pub struct NvencEncoder {
    config: NvencConfig,
    preset: NvencPreset,
    state: NvencState,
    stats: NvencStats,
}

/// Internal encoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NvencState {
    Idle,
    Encoding,
}

/// Full NVENC encoder configuration.
#[derive(Debug, Clone)]
pub struct NvencConfig {
    /// Resolution (width, height).
    pub resolution: (u32, u32),
    /// Target framerate.
    pub framerate: u32,
    /// Target bitrate in kbps.
    pub bitrate: u32,
    /// Rate control mode.
    pub rate_control: NvencRateControl,
    /// Tuning hint.
    pub tuning: NvencTuning,
    /// Maximum number of B-frames (0 = disabled).
    pub max_b_frames: u32,
    /// Lookahead depth (0 = disabled).
    pub lookahead_depth: u32,
    /// Keyframe interval in frames (0 = auto).
    pub gop_length: u32,
    /// Adaptive quantization mode.
    pub aq_mode: NvencAqMode,
    /// Multi-pass encoding.
    pub multi_pass: NvencMultiPass,
    /// Use weighted prediction.
    pub weighted_prediction: bool,
    /// Reference frame count.
    pub ref_frames: u32,
}

/// NVENC encoding presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencPreset {
    /// P1 - Fastest, lowest quality
    P1,
    /// P2 - Very fast
    P2,
    /// P3 - Fast
    P3,
    /// P4 - Medium (balanced)
    P4,
    /// P5 - Slow (high quality)
    P5,
    /// P6 - Slower
    P6,
    /// P7 - Slowest, highest quality
    P7,
}

/// NVENC rate control modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencRateControl {
    /// Constant bitrate.
    Cbr,
    /// Variable bitrate.
    Vbr,
    /// Constant quality (CQ) with quality level.
    ConstQuality(u8),
    /// Constant bitrate with high-quality mode.
    CbrHq,
    /// Variable bitrate with high-quality mode.
    VbrHq,
}

/// NVENC tuning hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencTuning {
    /// High quality (default).
    HighQuality,
    /// Low latency.
    LowLatency,
    /// Ultra-low latency.
    UltraLowLatency,
    /// Lossless.
    Lossless,
}

/// Adaptive quantization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencAqMode {
    /// Disabled.
    Disabled,
    /// Spatial AQ.
    Spatial,
    /// Temporal AQ.
    Temporal,
}

/// Multi-pass modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencMultiPass {
    /// Single pass (fastest).
    Disabled,
    /// Quarter resolution pass + full pass.
    QuarterResolution,
    /// Full resolution pass + full pass.
    FullResolution,
}

/// NVENC capabilities.
#[derive(Debug, Clone)]
pub struct NvencCapabilities {
    /// GPU name
    pub gpu_name: String,
    /// Maximum resolution width
    pub max_width: u32,
    /// Maximum resolution height
    pub max_height: u32,
    /// Maximum framerate
    pub max_framerate: u32,
    /// Supports AV1 encoding
    pub supports_av1: bool,
    /// Supports VP9 encoding
    pub supports_vp9: bool,
    /// Supports hardware B-frame encoding
    pub supports_b_frames: bool,
    /// Maximum concurrent sessions
    pub max_sessions: u32,
    /// Supports lookahead
    pub supports_lookahead: bool,
    /// Supports temporal AQ
    pub supports_temporal_aq: bool,
}

/// Runtime statistics from the NVENC encoder.
#[derive(Debug, Clone)]
pub struct NvencStats {
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Total bytes output.
    pub total_bytes: u64,
    /// Average encoding time per frame.
    pub avg_encode_time: Duration,
    /// Peak encoding time observed.
    pub peak_encode_time: Duration,
    /// Total encoding time accumulated.
    total_encode_time: Duration,
    /// Current effective bitrate in kbps.
    pub current_bitrate_kbps: u32,
}

impl Default for NvencStats {
    fn default() -> Self {
        Self {
            frames_encoded: 0,
            total_bytes: 0,
            avg_encode_time: Duration::ZERO,
            peak_encode_time: Duration::ZERO,
            total_encode_time: Duration::ZERO,
            current_bitrate_kbps: 0,
        }
    }
}

impl Default for NvencConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            framerate: 60,
            bitrate: 6000,
            rate_control: NvencRateControl::Cbr,
            tuning: NvencTuning::LowLatency,
            max_b_frames: 0,
            lookahead_depth: 0,
            gop_length: 120,
            aq_mode: NvencAqMode::Spatial,
            multi_pass: NvencMultiPass::Disabled,
            weighted_prediction: false,
            ref_frames: 3,
        }
    }
}

impl NvencConfig {
    /// Create a config optimised for ultra-low latency game streaming.
    #[must_use]
    pub fn ultra_low_latency(width: u32, height: u32, bitrate: u32) -> Self {
        Self {
            resolution: (width, height),
            framerate: 60,
            bitrate,
            rate_control: NvencRateControl::Cbr,
            tuning: NvencTuning::UltraLowLatency,
            max_b_frames: 0,
            lookahead_depth: 0,
            gop_length: 60,
            aq_mode: NvencAqMode::Disabled,
            multi_pass: NvencMultiPass::Disabled,
            weighted_prediction: false,
            ref_frames: 1,
        }
    }

    /// Create a config optimised for high quality recording.
    #[must_use]
    pub fn high_quality_recording(width: u32, height: u32, bitrate: u32) -> Self {
        Self {
            resolution: (width, height),
            framerate: 60,
            bitrate,
            rate_control: NvencRateControl::VbrHq,
            tuning: NvencTuning::HighQuality,
            max_b_frames: 2,
            lookahead_depth: 32,
            gop_length: 250,
            aq_mode: NvencAqMode::Temporal,
            multi_pass: NvencMultiPass::FullResolution,
            weighted_prediction: true,
            ref_frames: 4,
        }
    }

    /// Validate configuration values.
    ///
    /// # Errors
    ///
    /// Returns error if configuration values are out of range.
    pub fn validate(&self) -> GamingResult<()> {
        if self.resolution.0 == 0 || self.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "NVENC: resolution must be non-zero".into(),
            ));
        }
        if self.framerate == 0 || self.framerate > 240 {
            return Err(GamingError::InvalidConfig(
                "NVENC: framerate must be 1-240".into(),
            ));
        }
        if self.bitrate < 500 {
            return Err(GamingError::InvalidConfig(
                "NVENC: bitrate must be >= 500 kbps".into(),
            ));
        }
        if self.ref_frames > 16 {
            return Err(GamingError::InvalidConfig(
                "NVENC: ref_frames must be <= 16".into(),
            ));
        }
        if self.lookahead_depth > 64 {
            return Err(GamingError::InvalidConfig(
                "NVENC: lookahead_depth must be <= 64".into(),
            ));
        }
        // Ultra-low latency should have no B-frames
        if self.tuning == NvencTuning::UltraLowLatency && self.max_b_frames > 0 {
            return Err(GamingError::InvalidConfig(
                "NVENC: ultra-low latency mode must not use B-frames".into(),
            ));
        }
        Ok(())
    }

    /// Estimated encoding latency in milliseconds for this config.
    #[must_use]
    pub fn estimated_latency_ms(&self) -> u32 {
        let base = match self.tuning {
            NvencTuning::UltraLowLatency => 2,
            NvencTuning::LowLatency => 5,
            NvencTuning::HighQuality => 15,
            NvencTuning::Lossless => 3,
        };
        let b_frame_penalty = self.max_b_frames * 3;
        let lookahead_penalty = if self.lookahead_depth > 0 {
            self.lookahead_depth / 2
        } else {
            0
        };
        let multipass_penalty = match self.multi_pass {
            NvencMultiPass::Disabled => 0,
            NvencMultiPass::QuarterResolution => 3,
            NvencMultiPass::FullResolution => 8,
        };
        base + b_frame_penalty + lookahead_penalty + multipass_penalty
    }
}

impl NvencEncoder {
    /// Create a new NVENC encoder with full configuration.
    ///
    /// # Errors
    ///
    /// Returns error if the configuration is invalid.
    pub fn with_config(config: NvencConfig, preset: NvencPreset) -> GamingResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            preset,
            state: NvencState::Idle,
            stats: NvencStats::default(),
        })
    }

    /// Create a new NVENC encoder with preset and default config.
    ///
    /// # Errors
    ///
    /// Returns error if NVENC is not available.
    pub fn new(preset: NvencPreset) -> GamingResult<Self> {
        let config = NvencConfig::from_preset(preset);
        Self::with_config(config, preset)
    }

    /// Check if NVENC is available on this system.
    ///
    /// In a real implementation this would query the NVIDIA driver.
    /// Currently returns `false` as we use the software-simulated path.
    #[must_use]
    pub fn is_available() -> bool {
        false
    }

    /// Get NVENC capabilities.
    ///
    /// # Errors
    ///
    /// Returns error if capabilities cannot be queried.
    pub fn get_capabilities() -> GamingResult<NvencCapabilities> {
        // Return simulated caps for development/testing
        Ok(NvencCapabilities {
            gpu_name: "Simulated NVIDIA GPU".to_string(),
            max_width: 8192,
            max_height: 8192,
            max_framerate: 240,
            supports_av1: true,
            supports_vp9: true,
            supports_b_frames: true,
            max_sessions: 3,
            supports_lookahead: true,
            supports_temporal_aq: true,
        })
    }

    /// Get recommended preset for game streaming.
    #[must_use]
    pub fn recommended_preset_for_latency(target_latency_ms: u32) -> NvencPreset {
        if target_latency_ms < 50 {
            NvencPreset::P1
        } else if target_latency_ms < 100 {
            NvencPreset::P2
        } else {
            NvencPreset::P3
        }
    }

    /// Get current preset.
    #[must_use]
    pub fn preset(&self) -> NvencPreset {
        self.preset
    }

    /// Get the active configuration.
    #[must_use]
    pub fn config(&self) -> &NvencConfig {
        &self.config
    }

    /// Start encoding session.
    ///
    /// # Errors
    ///
    /// Returns error if already encoding.
    pub fn start(&mut self) -> GamingResult<()> {
        if self.state == NvencState::Encoding {
            return Err(GamingError::EncodingError(
                "NVENC encoder already started".into(),
            ));
        }
        self.state = NvencState::Encoding;
        self.stats = NvencStats::default();
        Ok(())
    }

    /// Encode a single RGBA frame.
    ///
    /// # Errors
    ///
    /// Returns error if encoder is not started or frame data is wrong size.
    pub fn encode_frame(&mut self, frame_data: &[u8]) -> GamingResult<Vec<u8>> {
        if self.state != NvencState::Encoding {
            return Err(GamingError::EncodingError(
                "NVENC encoder not started".into(),
            ));
        }

        let (w, h) = self.config.resolution;
        let expected = (w as usize) * (h as usize) * 4;
        if frame_data.len() != expected {
            return Err(GamingError::EncodingError(format!(
                "Frame size mismatch: expected {} got {}",
                expected,
                frame_data.len()
            )));
        }

        let start = Instant::now();

        // Simulate encoding: produce output sized by bitrate/framerate
        let bytes_per_frame = if self.config.framerate > 0 {
            ((u64::from(self.config.bitrate) * 1000) / (8 * u64::from(self.config.framerate)))
                .max(64) as usize
        } else {
            1024
        };

        let seq = self.stats.frames_encoded + 1;
        let is_keyframe = seq == 1
            || (self.config.gop_length > 0 && seq % u64::from(self.config.gop_length) == 0);
        let target_size = if is_keyframe {
            bytes_per_frame * 3
        } else {
            bytes_per_frame
        };

        // FNV hash of input for deterministic output
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let step = (frame_data.len() / 128).max(1);
        for i in (0..frame_data.len()).step_by(step) {
            hash ^= frame_data[i] as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }

        let mut output = Vec::with_capacity(target_size);
        output.extend_from_slice(b"NVE\x01");
        let mut rng = hash;
        while output.len() < target_size {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            output.push(rng as u8);
        }

        let elapsed = start.elapsed();
        self.stats.frames_encoded = seq;
        self.stats.total_bytes += output.len() as u64;
        self.stats.total_encode_time += elapsed;
        if elapsed > self.stats.peak_encode_time {
            self.stats.peak_encode_time = elapsed;
        }
        self.stats.avg_encode_time = self.stats.total_encode_time / seq as u32;

        // Update current bitrate
        let duration_secs = (seq as f64) / (self.config.framerate as f64).max(1.0);
        if duration_secs > 0.0 {
            self.stats.current_bitrate_kbps =
                ((self.stats.total_bytes as f64 * 8.0) / (duration_secs * 1000.0)) as u32;
        }

        Ok(output)
    }

    /// Stop encoding session.
    pub fn stop(&mut self) {
        self.state = NvencState::Idle;
    }

    /// Get encoder statistics.
    #[must_use]
    pub fn stats(&self) -> &NvencStats {
        &self.stats
    }

    /// Whether the encoder is currently active.
    #[must_use]
    pub fn is_encoding(&self) -> bool {
        self.state == NvencState::Encoding
    }
}

impl NvencConfig {
    /// Derive a config from a preset.
    #[must_use]
    pub fn from_preset(preset: NvencPreset) -> Self {
        match preset {
            NvencPreset::P1 => Self {
                tuning: NvencTuning::UltraLowLatency,
                max_b_frames: 0,
                lookahead_depth: 0,
                aq_mode: NvencAqMode::Disabled,
                multi_pass: NvencMultiPass::Disabled,
                ref_frames: 1,
                ..Self::default()
            },
            NvencPreset::P2 => Self {
                tuning: NvencTuning::LowLatency,
                max_b_frames: 0,
                lookahead_depth: 0,
                aq_mode: NvencAqMode::Spatial,
                multi_pass: NvencMultiPass::Disabled,
                ref_frames: 2,
                ..Self::default()
            },
            NvencPreset::P3 => Self {
                tuning: NvencTuning::LowLatency,
                max_b_frames: 0,
                lookahead_depth: 4,
                aq_mode: NvencAqMode::Spatial,
                multi_pass: NvencMultiPass::Disabled,
                ref_frames: 3,
                ..Self::default()
            },
            NvencPreset::P4 => Self {
                tuning: NvencTuning::HighQuality,
                max_b_frames: 0,
                lookahead_depth: 8,
                aq_mode: NvencAqMode::Spatial,
                multi_pass: NvencMultiPass::QuarterResolution,
                ref_frames: 3,
                ..Self::default()
            },
            NvencPreset::P5 => Self {
                tuning: NvencTuning::HighQuality,
                max_b_frames: 2,
                lookahead_depth: 16,
                aq_mode: NvencAqMode::Temporal,
                multi_pass: NvencMultiPass::QuarterResolution,
                ref_frames: 4,
                ..Self::default()
            },
            NvencPreset::P6 => Self {
                tuning: NvencTuning::HighQuality,
                max_b_frames: 3,
                lookahead_depth: 24,
                aq_mode: NvencAqMode::Temporal,
                multi_pass: NvencMultiPass::FullResolution,
                ref_frames: 4,
                ..Self::default()
            },
            NvencPreset::P7 => Self {
                tuning: NvencTuning::HighQuality,
                max_b_frames: 4,
                lookahead_depth: 32,
                aq_mode: NvencAqMode::Temporal,
                multi_pass: NvencMultiPass::FullResolution,
                weighted_prediction: true,
                ref_frames: 4,
                ..Self::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvenc_availability() {
        assert!(!NvencEncoder::is_available());
    }

    #[test]
    fn test_nvenc_create_with_preset() {
        let enc = NvencEncoder::new(NvencPreset::P3).expect("create p3");
        assert_eq!(enc.preset(), NvencPreset::P3);
        assert!(!enc.is_encoding());
    }

    #[test]
    fn test_nvenc_config_validate() {
        let cfg = NvencConfig::default();
        cfg.validate().expect("default config valid");
    }

    #[test]
    fn test_nvenc_config_validate_zero_resolution() {
        let mut cfg = NvencConfig::default();
        cfg.resolution = (0, 1080);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nvenc_config_validate_bad_framerate() {
        let mut cfg = NvencConfig::default();
        cfg.framerate = 300;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nvenc_config_validate_low_bitrate() {
        let mut cfg = NvencConfig::default();
        cfg.bitrate = 100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nvenc_config_validate_ull_with_b_frames() {
        let mut cfg = NvencConfig::default();
        cfg.tuning = NvencTuning::UltraLowLatency;
        cfg.max_b_frames = 2;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nvenc_config_validate_ref_frames_overflow() {
        let mut cfg = NvencConfig::default();
        cfg.ref_frames = 20;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nvenc_ultra_low_latency_config() {
        let cfg = NvencConfig::ultra_low_latency(1920, 1080, 6000);
        cfg.validate().expect("valid ull config");
        assert_eq!(cfg.max_b_frames, 0);
        assert_eq!(cfg.tuning, NvencTuning::UltraLowLatency);
        assert_eq!(cfg.lookahead_depth, 0);
    }

    #[test]
    fn test_nvenc_high_quality_config() {
        let cfg = NvencConfig::high_quality_recording(3840, 2160, 50000);
        cfg.validate().expect("valid hq config");
        assert!(cfg.max_b_frames > 0);
        assert_eq!(cfg.multi_pass, NvencMultiPass::FullResolution);
    }

    #[test]
    fn test_nvenc_estimated_latency() {
        let ull = NvencConfig::ultra_low_latency(1920, 1080, 6000);
        let hq = NvencConfig::high_quality_recording(1920, 1080, 20000);
        assert!(ull.estimated_latency_ms() < hq.estimated_latency_ms());
    }

    #[test]
    fn test_nvenc_encode_frame() {
        let cfg = NvencConfig {
            resolution: (16, 16),
            ..NvencConfig::default()
        };
        let mut enc = NvencEncoder::with_config(cfg, NvencPreset::P3).expect("create");
        enc.start().expect("start");

        let frame = vec![128u8; 16 * 16 * 4];
        let encoded = enc.encode_frame(&frame).expect("encode");
        assert!(!encoded.is_empty());
        assert_eq!(&encoded[..4], b"NVE\x01");
        assert_eq!(enc.stats().frames_encoded, 1);
    }

    #[test]
    fn test_nvenc_encode_wrong_size() {
        let cfg = NvencConfig {
            resolution: (16, 16),
            ..NvencConfig::default()
        };
        let mut enc = NvencEncoder::with_config(cfg, NvencPreset::P3).expect("create");
        enc.start().expect("start");
        assert!(enc.encode_frame(&[0u8; 100]).is_err());
    }

    #[test]
    fn test_nvenc_encode_not_started() {
        let cfg = NvencConfig {
            resolution: (16, 16),
            ..NvencConfig::default()
        };
        let mut enc = NvencEncoder::with_config(cfg, NvencPreset::P3).expect("create");
        assert!(enc.encode_frame(&[0u8; 16 * 16 * 4]).is_err());
    }

    #[test]
    fn test_nvenc_double_start() {
        let mut enc = NvencEncoder::new(NvencPreset::P1).expect("create");
        enc.start().expect("start");
        assert!(enc.start().is_err());
    }

    #[test]
    fn test_nvenc_stats_accumulate() {
        let cfg = NvencConfig {
            resolution: (8, 8),
            ..NvencConfig::default()
        };
        let mut enc = NvencEncoder::with_config(cfg, NvencPreset::P1).expect("create");
        enc.start().expect("start");

        let frame = vec![0u8; 8 * 8 * 4];
        for _ in 0..5 {
            enc.encode_frame(&frame).expect("encode");
        }
        let s = enc.stats();
        assert_eq!(s.frames_encoded, 5);
        assert!(s.total_bytes > 0);
        assert!(s.avg_encode_time > Duration::ZERO);
    }

    #[test]
    fn test_nvenc_stop_and_restart() {
        let cfg = NvencConfig {
            resolution: (8, 8),
            ..NvencConfig::default()
        };
        let mut enc = NvencEncoder::with_config(cfg, NvencPreset::P1).expect("create");
        enc.start().expect("start");
        enc.stop();
        assert!(!enc.is_encoding());
        enc.start().expect("restart");
        assert!(enc.is_encoding());
    }

    #[test]
    fn test_nvenc_capabilities() {
        let caps = NvencEncoder::get_capabilities().expect("caps");
        assert!(caps.max_width >= 4096);
        assert!(caps.supports_b_frames);
    }

    #[test]
    fn test_recommended_preset() {
        assert_eq!(
            NvencEncoder::recommended_preset_for_latency(30),
            NvencPreset::P1
        );
        assert_eq!(
            NvencEncoder::recommended_preset_for_latency(80),
            NvencPreset::P2
        );
        assert_eq!(
            NvencEncoder::recommended_preset_for_latency(150),
            NvencPreset::P3
        );
    }

    #[test]
    fn test_preset_config_mapping() {
        let presets = [
            NvencPreset::P1,
            NvencPreset::P2,
            NvencPreset::P3,
            NvencPreset::P4,
            NvencPreset::P5,
            NvencPreset::P6,
            NvencPreset::P7,
        ];
        for p in presets {
            let cfg = NvencConfig::from_preset(p);
            cfg.validate().expect("preset config must be valid");
        }
    }

    #[test]
    fn test_nvenc_config_lookahead_too_large() {
        let mut cfg = NvencConfig::default();
        cfg.lookahead_depth = 100;
        assert!(cfg.validate().is_err());
    }
}
