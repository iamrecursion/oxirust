//! Quality control modes and rate control for video encoding.

use serde::{Deserialize, Serialize};

/// Quality preset levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityMode {
    /// Low quality (faster encoding, lower bitrate).
    Low,
    /// Medium quality (balanced speed and quality).
    Medium,
    /// High quality (slower encoding, higher bitrate).
    High,
    /// Very high quality (very slow encoding, very high bitrate).
    VeryHigh,
    /// Custom quality with specific parameters.
    Custom,
}

/// Quality preset configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreset {
    /// Ultra-fast encoding with minimal quality.
    UltraFast,
    /// Super-fast encoding with low quality.
    SuperFast,
    /// Very fast encoding with reduced quality.
    VeryFast,
    /// Fast encoding with acceptable quality.
    Fast,
    /// Medium speed and quality (balanced).
    Medium,
    /// Slow encoding with good quality.
    Slow,
    /// Very slow encoding with excellent quality.
    VerySlow,
    /// Placebo (extremely slow, diminishing returns).
    Placebo,
}

/// Rate control modes for video encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateControlMode {
    /// Constant Rate Factor (quality-based).
    ///
    /// Maintains consistent quality across the video.
    /// Lower values = higher quality, larger file.
    /// Typical range: 18-28 (23 is default for many codecs).
    Crf(u8),

    /// Constant Bitrate (CBR).
    ///
    /// Maintains a constant bitrate throughout.
    /// Good for streaming with bandwidth constraints.
    /// Value in bits per second.
    Cbr(u64),

    /// Variable Bitrate (VBR).
    ///
    /// Allows bitrate to vary based on complexity.
    /// Target bitrate in bits per second, with quality factor.
    Vbr {
        /// Target average bitrate.
        target: u64,
        /// Maximum bitrate.
        max: u64,
    },

    /// Constrained Variable Bitrate.
    ///
    /// VBR with strict maximum bitrate limits.
    ConstrainedVbr {
        /// Target average bitrate.
        target: u64,
        /// Maximum bitrate (hard limit).
        max: u64,
        /// Buffer size for rate control.
        buffer_size: u64,
    },

    /// Average Bitrate (ABR).
    ///
    /// Targets an average bitrate over the entire video.
    Abr(u64),
}

/// Quality configuration for encoding.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Quality preset.
    pub preset: QualityPreset,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Enable two-pass encoding for better quality.
    pub two_pass: bool,
    /// Look-ahead frames for better quality decisions.
    pub lookahead: Option<u32>,
    /// Tune for specific content type.
    pub tune: Option<TuneMode>,
}

/// Tuning modes for different content types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuneMode {
    /// Optimize for film content.
    Film,
    /// Optimize for animation.
    Animation,
    /// Optimize for grain preservation.
    Grain,
    /// Optimize for still images.
    StillImage,
    /// Optimize for fast decode.
    FastDecode,
    /// Optimize for zero latency (live streaming).
    ZeroLatency,
    /// Optimize for PSNR quality metric.
    Psnr,
    /// Optimize for SSIM quality metric.
    Ssim,
}

impl QualityMode {
    /// Converts quality mode to a CRF value for most codecs.
    ///
    /// # Returns
    ///
    /// CRF value (lower = higher quality):
    /// - Low: 28
    /// - Medium: 23
    /// - High: 20
    /// - `VeryHigh`: 18
    #[must_use]
    pub fn to_crf(self) -> u8 {
        match self {
            Self::Low => 28,
            Self::Medium => 23,
            Self::High => 20,
            Self::VeryHigh => 18,
            Self::Custom => 23, // Default fallback
        }
    }

    /// Converts quality mode to a preset.
    #[must_use]
    pub fn to_preset(self) -> QualityPreset {
        match self {
            Self::Low => QualityPreset::VeryFast,
            Self::Medium => QualityPreset::Medium,
            Self::High => QualityPreset::Slow,
            Self::VeryHigh => QualityPreset::VerySlow,
            Self::Custom => QualityPreset::Medium,
        }
    }

    /// Estimates encoding speed factor relative to real-time.
    ///
    /// # Returns
    ///
    /// Approximate speed factor:
    /// - Low: 5.0x (encodes 5x faster than real-time)
    /// - Medium: 1.5x
    /// - High: 0.5x
    /// - `VeryHigh`: 0.2x
    #[must_use]
    pub fn speed_factor(self) -> f64 {
        match self {
            Self::Low => 5.0,
            Self::Medium => 1.5,
            Self::High => 0.5,
            Self::VeryHigh => 0.2,
            Self::Custom => 1.0,
        }
    }
}

impl QualityPreset {
    /// Gets the preset name as a string (compatible with x264/x265/libvpx naming).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UltraFast => "ultrafast",
            Self::SuperFast => "superfast",
            Self::VeryFast => "veryfast",
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::Slow => "slow",
            Self::VerySlow => "veryslow",
            Self::Placebo => "placebo",
        }
    }

    /// Gets the CPU usage level (0-8, higher = slower/better).
    #[must_use]
    pub fn cpu_used(self) -> u8 {
        match self {
            Self::UltraFast => 8,
            Self::SuperFast => 7,
            Self::VeryFast => 6,
            Self::Fast => 5,
            Self::Medium => 4,
            Self::Slow => 2,
            Self::VerySlow => 1,
            Self::Placebo => 0,
        }
    }
}

impl RateControlMode {
    /// Gets the target bitrate for this rate control mode.
    #[must_use]
    pub fn target_bitrate(&self) -> Option<u64> {
        match self {
            Self::Crf(_) => None,
            Self::Cbr(bitrate) | Self::Abr(bitrate) => Some(*bitrate),
            Self::Vbr { target, .. } | Self::ConstrainedVbr { target, .. } => Some(*target),
        }
    }

    /// Gets the maximum bitrate for this rate control mode.
    #[must_use]
    pub fn max_bitrate(&self) -> Option<u64> {
        match self {
            Self::Crf(_) | Self::Cbr(_) | Self::Abr(_) => None,
            Self::Vbr { max, .. } | Self::ConstrainedVbr { max, .. } => Some(*max),
        }
    }

    /// Checks if this is a constant quality mode.
    #[must_use]
    pub fn is_constant_quality(&self) -> bool {
        matches!(self, Self::Crf(_))
    }

    /// Checks if this is a bitrate-based mode.
    #[must_use]
    pub fn is_bitrate_mode(&self) -> bool {
        !self.is_constant_quality()
    }
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            preset: QualityPreset::Medium,
            rate_control: RateControlMode::Crf(23),
            two_pass: false,
            lookahead: Some(40),
            tune: None,
        }
    }
}

impl TuneMode {
    /// Gets the tune mode name as a string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Film => "film",
            Self::Animation => "animation",
            Self::Grain => "grain",
            Self::StillImage => "stillimage",
            Self::FastDecode => "fastdecode",
            Self::ZeroLatency => "zerolatency",
            Self::Psnr => "psnr",
            Self::Ssim => "ssim",
        }
    }
}

/// Builder for quality configuration.
#[allow(dead_code)]
pub struct QualityConfigBuilder {
    config: QualityConfig,
}

#[allow(dead_code)]
impl QualityConfigBuilder {
    /// Creates a new quality config builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: QualityConfig::default(),
        }
    }

    /// Sets the quality preset.
    #[must_use]
    pub fn preset(mut self, preset: QualityPreset) -> Self {
        self.config.preset = preset;
        self
    }

    /// Sets the rate control mode.
    #[must_use]
    pub fn rate_control(mut self, mode: RateControlMode) -> Self {
        self.config.rate_control = mode;
        self
    }

    /// Enables two-pass encoding.
    #[must_use]
    pub fn two_pass(mut self, enable: bool) -> Self {
        self.config.two_pass = enable;
        self
    }

    /// Sets the lookahead frame count.
    #[must_use]
    pub fn lookahead(mut self, frames: u32) -> Self {
        self.config.lookahead = Some(frames);
        self
    }

    /// Sets the tune mode.
    #[must_use]
    pub fn tune(mut self, mode: TuneMode) -> Self {
        self.config.tune = Some(mode);
        self
    }

    /// Builds the quality configuration.
    #[must_use]
    pub fn build(self) -> QualityConfig {
        self.config
    }
}

impl Default for QualityConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_mode_crf() {
        assert_eq!(QualityMode::Low.to_crf(), 28);
        assert_eq!(QualityMode::Medium.to_crf(), 23);
        assert_eq!(QualityMode::High.to_crf(), 20);
        assert_eq!(QualityMode::VeryHigh.to_crf(), 18);
    }

    #[test]
    fn test_quality_preset_str() {
        assert_eq!(QualityPreset::UltraFast.as_str(), "ultrafast");
        assert_eq!(QualityPreset::Medium.as_str(), "medium");
        assert_eq!(QualityPreset::VerySlow.as_str(), "veryslow");
    }

    #[test]
    fn test_quality_preset_cpu_used() {
        assert_eq!(QualityPreset::UltraFast.cpu_used(), 8);
        assert_eq!(QualityPreset::Medium.cpu_used(), 4);
        assert_eq!(QualityPreset::Placebo.cpu_used(), 0);
    }

    #[test]
    fn test_rate_control_crf() {
        let crf = RateControlMode::Crf(23);
        assert!(crf.is_constant_quality());
        assert!(!crf.is_bitrate_mode());
        assert_eq!(crf.target_bitrate(), None);
        assert_eq!(crf.max_bitrate(), None);
    }

    #[test]
    fn test_rate_control_cbr() {
        let cbr = RateControlMode::Cbr(5_000_000);
        assert!(!cbr.is_constant_quality());
        assert!(cbr.is_bitrate_mode());
        assert_eq!(cbr.target_bitrate(), Some(5_000_000));
        assert_eq!(cbr.max_bitrate(), None);
    }

    #[test]
    fn test_rate_control_vbr() {
        let vbr = RateControlMode::Vbr {
            target: 5_000_000,
            max: 8_000_000,
        };
        assert!(!vbr.is_constant_quality());
        assert!(vbr.is_bitrate_mode());
        assert_eq!(vbr.target_bitrate(), Some(5_000_000));
        assert_eq!(vbr.max_bitrate(), Some(8_000_000));
    }

    #[test]
    fn test_rate_control_constrained_vbr() {
        let cvbr = RateControlMode::ConstrainedVbr {
            target: 5_000_000,
            max: 8_000_000,
            buffer_size: 10_000_000,
        };
        assert_eq!(cvbr.target_bitrate(), Some(5_000_000));
        assert_eq!(cvbr.max_bitrate(), Some(8_000_000));
    }

    #[test]
    fn test_tune_mode_str() {
        assert_eq!(TuneMode::Film.as_str(), "film");
        assert_eq!(TuneMode::Animation.as_str(), "animation");
        assert_eq!(TuneMode::ZeroLatency.as_str(), "zerolatency");
    }

    #[test]
    fn test_quality_config_builder() {
        let config = QualityConfigBuilder::new()
            .preset(QualityPreset::Slow)
            .rate_control(RateControlMode::Crf(20))
            .two_pass(true)
            .lookahead(60)
            .tune(TuneMode::Film)
            .build();

        assert_eq!(config.preset, QualityPreset::Slow);
        assert_eq!(config.rate_control, RateControlMode::Crf(20));
        assert!(config.two_pass);
        assert_eq!(config.lookahead, Some(60));
        assert_eq!(config.tune, Some(TuneMode::Film));
    }

    #[test]
    fn test_quality_mode_speed_factor() {
        assert_eq!(QualityMode::Low.speed_factor(), 5.0);
        assert_eq!(QualityMode::Medium.speed_factor(), 1.5);
        assert_eq!(QualityMode::High.speed_factor(), 0.5);
        assert_eq!(QualityMode::VeryHigh.speed_factor(), 0.2);
    }
}
