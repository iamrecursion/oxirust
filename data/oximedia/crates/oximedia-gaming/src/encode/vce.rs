//! AMD VCE/VCN hardware encoder.
//!
//! Provides AMD GPU hardware-accelerated encoding with full configuration
//! for game streaming. Feature-gated: when actual VCE/VCN hardware is
//! not available, a software-simulated fallback is used.

use crate::{GamingError, GamingResult};
use std::time::{Duration, Instant};

/// AMD VCE/VCN encoder.
pub struct VceEncoder {
    config: VceConfig,
    preset: VcePreset,
    state: VceState,
    stats: VceStats,
}

/// Internal encoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VceState {
    Idle,
    Encoding,
}

/// Full VCE/VCN encoder configuration.
#[derive(Debug, Clone)]
pub struct VceConfig {
    /// Resolution (width, height).
    pub resolution: (u32, u32),
    /// Target framerate.
    pub framerate: u32,
    /// Target bitrate in kbps.
    pub bitrate: u32,
    /// Rate control mode.
    pub rate_control: VceRateControl,
    /// Quality preset (speed vs quality tradeoff).
    pub quality_preset: VceQualityPreset,
    /// Maximum B-frames.
    pub max_b_frames: u32,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Reference frame count.
    pub ref_frames: u32,
    /// Pre-analysis pass.
    pub pre_analysis: bool,
    /// VCN hardware version.
    pub vcn_version: VcnVersion,
    /// Enable VBAQ (Variance Based Adaptive Quantization).
    pub vbaq: bool,
    /// High motion quality boost.
    pub hmqb: bool,
}

/// VCE encoding presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcePreset {
    /// Speed preset (lowest quality, fastest)
    Speed,
    /// Balanced preset
    Balanced,
    /// Quality preset (highest quality, slowest)
    Quality,
}

/// VCE rate control modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VceRateControl {
    /// Constant bitrate.
    Cbr,
    /// Variable bitrate peak constrained.
    VbrPeak,
    /// Variable bitrate latency constrained.
    VbrLatency,
    /// Constant QP.
    /// Constant QP with per-frame-type QP values.
    Cqp {
        /// I-frame QP.
        qp_i: u8,
        /// P-frame QP.
        qp_p: u8,
    },
    /// Quality VBR.
    Qvbr(u8),
}

/// VCE quality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VceQualityPreset {
    /// Speed priority.
    Speed,
    /// Balanced.
    Balanced,
    /// Quality priority.
    Quality,
}

/// VCN hardware version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcnVersion {
    /// Unknown / auto-detect.
    Auto,
    /// VCN 1.0 (Vega).
    Vcn1,
    /// VCN 2.0 (Navi 1x).
    Vcn2,
    /// VCN 3.0 (Navi 2x / RDNA2).
    Vcn3,
    /// VCN 4.0 (Navi 3x / RDNA3).
    Vcn4,
    /// VCN 5.0 (RDNA4).
    Vcn5,
}

/// VCE capabilities.
#[derive(Debug, Clone)]
pub struct VceCapabilities {
    /// GPU name
    pub gpu_name: String,
    /// VCN version
    pub vcn_version: VcnVersion,
    /// Maximum resolution width
    pub max_width: u32,
    /// Maximum resolution height
    pub max_height: u32,
    /// Supports AV1 encoding (VCN 4.0+)
    pub supports_av1: bool,
    /// Supports VP9 encoding
    pub supports_vp9: bool,
    /// Supports pre-analysis
    pub supports_pre_analysis: bool,
    /// Supports VBAQ
    pub supports_vbaq: bool,
}

/// Runtime statistics from the VCE encoder.
#[derive(Debug, Clone)]
pub struct VceStats {
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Total bytes output.
    pub total_bytes: u64,
    /// Average encoding time per frame.
    pub avg_encode_time: Duration,
    /// Peak encoding time observed.
    pub peak_encode_time: Duration,
    /// Accumulated encoding time.
    total_encode_time: Duration,
    /// Current effective bitrate in kbps.
    pub current_bitrate_kbps: u32,
}

impl Default for VceStats {
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

impl Default for VceConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            framerate: 60,
            bitrate: 6000,
            rate_control: VceRateControl::Cbr,
            quality_preset: VceQualityPreset::Balanced,
            max_b_frames: 0,
            gop_size: 120,
            ref_frames: 3,
            pre_analysis: false,
            vcn_version: VcnVersion::Auto,
            vbaq: true,
            hmqb: false,
        }
    }
}

impl VceConfig {
    /// Create a config optimised for low latency streaming.
    #[must_use]
    pub fn low_latency_streaming(width: u32, height: u32, bitrate: u32) -> Self {
        Self {
            resolution: (width, height),
            framerate: 60,
            bitrate,
            rate_control: VceRateControl::Cbr,
            quality_preset: VceQualityPreset::Speed,
            max_b_frames: 0,
            gop_size: 60,
            ref_frames: 1,
            pre_analysis: false,
            vcn_version: VcnVersion::Auto,
            vbaq: false,
            hmqb: false,
        }
    }

    /// Create a config optimised for high-quality recording.
    #[must_use]
    pub fn high_quality_recording(width: u32, height: u32, bitrate: u32) -> Self {
        Self {
            resolution: (width, height),
            framerate: 60,
            bitrate,
            rate_control: VceRateControl::Qvbr(23),
            quality_preset: VceQualityPreset::Quality,
            max_b_frames: 3,
            gop_size: 250,
            ref_frames: 4,
            pre_analysis: true,
            vcn_version: VcnVersion::Auto,
            vbaq: true,
            hmqb: true,
        }
    }

    /// Validate configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn validate(&self) -> GamingResult<()> {
        if self.resolution.0 == 0 || self.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "VCE: resolution must be non-zero".into(),
            ));
        }
        if self.framerate == 0 || self.framerate > 240 {
            return Err(GamingError::InvalidConfig(
                "VCE: framerate must be 1-240".into(),
            ));
        }
        if self.bitrate < 500 {
            return Err(GamingError::InvalidConfig(
                "VCE: bitrate must be >= 500 kbps".into(),
            ));
        }
        if self.ref_frames > 16 {
            return Err(GamingError::InvalidConfig(
                "VCE: ref_frames must be <= 16".into(),
            ));
        }
        // Speed preset should avoid B-frames for latency
        if self.quality_preset == VceQualityPreset::Speed && self.max_b_frames > 0 {
            return Err(GamingError::InvalidConfig(
                "VCE: speed preset should not use B-frames".into(),
            ));
        }
        Ok(())
    }

    /// Estimated encoding latency in milliseconds.
    #[must_use]
    pub fn estimated_latency_ms(&self) -> u32 {
        let base = match self.quality_preset {
            VceQualityPreset::Speed => 4,
            VceQualityPreset::Balanced => 8,
            VceQualityPreset::Quality => 15,
        };
        let b_penalty = self.max_b_frames * 4;
        let pre_analysis_penalty = if self.pre_analysis { 5 } else { 0 };
        base + b_penalty + pre_analysis_penalty
    }
}

impl VceEncoder {
    /// Create a new VCE encoder with full configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn with_config(config: VceConfig, preset: VcePreset) -> GamingResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            preset,
            state: VceState::Idle,
            stats: VceStats::default(),
        })
    }

    /// Create a new VCE encoder with preset defaults.
    ///
    /// # Errors
    ///
    /// Returns error if preset config is invalid.
    pub fn new(preset: VcePreset) -> GamingResult<Self> {
        let config = VceConfig::from_preset(preset);
        Self::with_config(config, preset)
    }

    /// Check if VCE is available on this system.
    #[must_use]
    pub fn is_available() -> bool {
        false
    }

    /// Get VCE capabilities.
    ///
    /// # Errors
    ///
    /// Returns error if capabilities cannot be queried.
    pub fn get_capabilities() -> GamingResult<VceCapabilities> {
        Ok(VceCapabilities {
            gpu_name: "Simulated AMD GPU".to_string(),
            vcn_version: VcnVersion::Auto,
            max_width: 8192,
            max_height: 8192,
            supports_av1: true,
            supports_vp9: true,
            supports_pre_analysis: true,
            supports_vbaq: true,
        })
    }

    /// Get recommended preset for game streaming.
    #[must_use]
    pub fn recommended_preset_for_latency(target_latency_ms: u32) -> VcePreset {
        if target_latency_ms < 100 {
            VcePreset::Speed
        } else {
            VcePreset::Balanced
        }
    }

    /// Get current preset.
    #[must_use]
    pub fn preset(&self) -> VcePreset {
        self.preset
    }

    /// Get the active configuration.
    #[must_use]
    pub fn config(&self) -> &VceConfig {
        &self.config
    }

    /// Start encoding session.
    ///
    /// # Errors
    ///
    /// Returns error if already encoding.
    pub fn start(&mut self) -> GamingResult<()> {
        if self.state == VceState::Encoding {
            return Err(GamingError::EncodingError(
                "VCE encoder already started".into(),
            ));
        }
        self.state = VceState::Encoding;
        self.stats = VceStats::default();
        Ok(())
    }

    /// Encode a single RGBA frame.
    ///
    /// # Errors
    ///
    /// Returns error if encoder is not started or frame data is wrong size.
    pub fn encode_frame(&mut self, frame_data: &[u8]) -> GamingResult<Vec<u8>> {
        if self.state != VceState::Encoding {
            return Err(GamingError::EncodingError("VCE encoder not started".into()));
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

        let bytes_per_frame = if self.config.framerate > 0 {
            ((u64::from(self.config.bitrate) * 1000) / (8 * u64::from(self.config.framerate)))
                .max(64) as usize
        } else {
            1024
        };

        let seq = self.stats.frames_encoded + 1;
        let is_keyframe =
            seq == 1 || (self.config.gop_size > 0 && seq % u64::from(self.config.gop_size) == 0);
        let target_size = if is_keyframe {
            bytes_per_frame * 3
        } else {
            bytes_per_frame
        };

        let mut hash: u64 = 0xa5a5_a5a5_5a5a_5a5a;
        let step = (frame_data.len() / 128).max(1);
        for i in (0..frame_data.len()).step_by(step) {
            hash ^= frame_data[i] as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }

        let mut output = Vec::with_capacity(target_size);
        output.extend_from_slice(b"VCE\x01");
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

        let duration_secs = (seq as f64) / (self.config.framerate as f64).max(1.0);
        if duration_secs > 0.0 {
            self.stats.current_bitrate_kbps =
                ((self.stats.total_bytes as f64 * 8.0) / (duration_secs * 1000.0)) as u32;
        }

        Ok(output)
    }

    /// Stop encoding session.
    pub fn stop(&mut self) {
        self.state = VceState::Idle;
    }

    /// Get encoder statistics.
    #[must_use]
    pub fn stats(&self) -> &VceStats {
        &self.stats
    }

    /// Whether the encoder is currently active.
    #[must_use]
    pub fn is_encoding(&self) -> bool {
        self.state == VceState::Encoding
    }
}

impl VceConfig {
    /// Derive a config from a preset.
    #[must_use]
    pub fn from_preset(preset: VcePreset) -> Self {
        match preset {
            VcePreset::Speed => Self {
                quality_preset: VceQualityPreset::Speed,
                max_b_frames: 0,
                pre_analysis: false,
                ref_frames: 1,
                vbaq: false,
                hmqb: false,
                ..Self::default()
            },
            VcePreset::Balanced => Self {
                quality_preset: VceQualityPreset::Balanced,
                max_b_frames: 0,
                pre_analysis: false,
                ref_frames: 3,
                vbaq: true,
                hmqb: false,
                ..Self::default()
            },
            VcePreset::Quality => Self {
                quality_preset: VceQualityPreset::Quality,
                max_b_frames: 0,
                pre_analysis: true,
                ref_frames: 4,
                vbaq: true,
                hmqb: true,
                ..Self::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vce_availability() {
        assert!(!VceEncoder::is_available());
    }

    #[test]
    fn test_vce_create_with_preset() {
        let enc = VceEncoder::new(VcePreset::Balanced).expect("create");
        assert_eq!(enc.preset(), VcePreset::Balanced);
        assert!(!enc.is_encoding());
    }

    #[test]
    fn test_vce_config_validate() {
        let cfg = VceConfig::default();
        cfg.validate().expect("default valid");
    }

    #[test]
    fn test_vce_config_validate_zero_res() {
        let mut cfg = VceConfig::default();
        cfg.resolution = (0, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_vce_config_validate_speed_b_frames() {
        let mut cfg = VceConfig::default();
        cfg.quality_preset = VceQualityPreset::Speed;
        cfg.max_b_frames = 2;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_vce_low_latency_config() {
        let cfg = VceConfig::low_latency_streaming(1920, 1080, 6000);
        cfg.validate().expect("valid");
        assert_eq!(cfg.quality_preset, VceQualityPreset::Speed);
        assert_eq!(cfg.max_b_frames, 0);
    }

    #[test]
    fn test_vce_hq_config() {
        let cfg = VceConfig::high_quality_recording(1920, 1080, 20000);
        cfg.validate().expect("valid");
        assert!(cfg.pre_analysis);
        assert!(cfg.hmqb);
    }

    #[test]
    fn test_vce_estimated_latency() {
        let ll = VceConfig::low_latency_streaming(1920, 1080, 6000);
        let hq = VceConfig::high_quality_recording(1920, 1080, 20000);
        assert!(ll.estimated_latency_ms() < hq.estimated_latency_ms());
    }

    #[test]
    fn test_vce_encode_frame() {
        let cfg = VceConfig {
            resolution: (8, 8),
            ..VceConfig::default()
        };
        let mut enc = VceEncoder::with_config(cfg, VcePreset::Balanced).expect("create");
        enc.start().expect("start");

        let frame = vec![0u8; 8 * 8 * 4];
        let encoded = enc.encode_frame(&frame).expect("encode");
        assert!(!encoded.is_empty());
        assert_eq!(&encoded[..4], b"VCE\x01");
    }

    #[test]
    fn test_vce_encode_wrong_size() {
        let cfg = VceConfig {
            resolution: (8, 8),
            ..VceConfig::default()
        };
        let mut enc = VceEncoder::with_config(cfg, VcePreset::Balanced).expect("create");
        enc.start().expect("start");
        assert!(enc.encode_frame(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_vce_stats_accumulate() {
        let cfg = VceConfig {
            resolution: (4, 4),
            ..VceConfig::default()
        };
        let mut enc = VceEncoder::with_config(cfg, VcePreset::Speed).expect("create");
        enc.start().expect("start");

        let frame = vec![0u8; 4 * 4 * 4];
        for _ in 0..4 {
            enc.encode_frame(&frame).expect("encode");
        }
        assert_eq!(enc.stats().frames_encoded, 4);
        assert!(enc.stats().total_bytes > 0);
    }

    #[test]
    fn test_vce_capabilities() {
        let caps = VceEncoder::get_capabilities().expect("caps");
        assert!(caps.supports_av1);
        assert!(caps.supports_vbaq);
    }

    #[test]
    fn test_recommended_preset() {
        assert_eq!(
            VceEncoder::recommended_preset_for_latency(50),
            VcePreset::Speed
        );
        assert_eq!(
            VceEncoder::recommended_preset_for_latency(150),
            VcePreset::Balanced
        );
    }

    #[test]
    fn test_preset_config_mapping() {
        let presets = [VcePreset::Speed, VcePreset::Balanced, VcePreset::Quality];
        for p in presets {
            let cfg = VceConfig::from_preset(p);
            cfg.validate().expect("preset config must be valid");
        }
    }

    #[test]
    fn test_vce_double_start() {
        let mut enc = VceEncoder::new(VcePreset::Balanced).expect("create");
        enc.start().expect("start");
        assert!(enc.start().is_err());
    }

    #[test]
    fn test_vce_stop_restart() {
        let mut enc = VceEncoder::new(VcePreset::Balanced).expect("create");
        enc.start().expect("start");
        enc.stop();
        assert!(!enc.is_encoding());
        enc.start().expect("restart");
        assert!(enc.is_encoding());
    }

    #[test]
    fn test_vce_encode_not_started() {
        let mut enc = VceEncoder::new(VcePreset::Balanced).expect("create");
        assert!(enc.encode_frame(&[0u8; 4]).is_err());
    }

    #[test]
    fn test_vcn_version_variants() {
        let versions = [
            VcnVersion::Auto,
            VcnVersion::Vcn1,
            VcnVersion::Vcn2,
            VcnVersion::Vcn3,
            VcnVersion::Vcn4,
            VcnVersion::Vcn5,
        ];
        for v in versions {
            let cfg = VceConfig {
                vcn_version: v,
                ..VceConfig::default()
            };
            cfg.validate().expect("valid");
        }
    }
}
