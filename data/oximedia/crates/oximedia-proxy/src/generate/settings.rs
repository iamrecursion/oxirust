//! Proxy generation settings and configuration.

use serde::{Deserialize, Serialize};

/// Proxy generation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyGenerationSettings {
    /// Target resolution as a scale factor (0.25, 0.5, 1.0).
    pub scale_factor: f32,

    /// Target codec name (e.g., "h264", "vp9").
    pub codec: String,

    /// Target bitrate in bits per second.
    pub bitrate: u64,

    /// Audio codec name (e.g., "aac", "opus").
    pub audio_codec: String,

    /// Audio bitrate in bits per second.
    pub audio_bitrate: u64,

    /// Frame rate preservation mode.
    pub preserve_frame_rate: bool,

    /// Timecode preservation mode.
    pub preserve_timecode: bool,

    /// Metadata preservation mode.
    pub preserve_metadata: bool,

    /// Container format (e.g., "mp4", "mov").
    pub container: String,

    /// Use hardware acceleration if available.
    pub use_hw_accel: bool,

    /// Number of encoding threads (0 = auto).
    pub threads: u32,

    /// Quality preset name (e.g., "fast", "medium", "slow").
    pub quality_preset: String,
}

impl Default for ProxyGenerationSettings {
    fn default() -> Self {
        Self {
            scale_factor: 0.25, // Quarter resolution by default
            codec: "h264".to_string(),
            bitrate: 5_000_000, // 5 Mbps
            audio_codec: "aac".to_string(),
            audio_bitrate: 128_000, // 128 kbps
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mp4".to_string(),
            use_hw_accel: true,
            threads: 0, // Auto
            quality_preset: "medium".to_string(),
        }
    }
}

impl ProxyGenerationSettings {
    /// Create settings for quarter resolution H.264 proxy.
    #[must_use]
    pub fn quarter_res_h264() -> Self {
        Self {
            scale_factor: 0.25,
            codec: "h264".to_string(),
            bitrate: 2_000_000,
            ..Self::default()
        }
    }

    /// Create settings for half resolution H.264 proxy.
    #[must_use]
    pub fn half_res_h264() -> Self {
        Self {
            scale_factor: 0.5,
            codec: "h264".to_string(),
            bitrate: 5_000_000,
            ..Self::default()
        }
    }

    /// Create settings for full resolution H.264 proxy.
    #[must_use]
    pub fn full_res_h264() -> Self {
        Self {
            scale_factor: 1.0,
            codec: "h264".to_string(),
            bitrate: 10_000_000,
            ..Self::default()
        }
    }

    /// Create settings for quarter resolution VP9 proxy.
    #[must_use]
    pub fn quarter_res_vp9() -> Self {
        Self {
            scale_factor: 0.25,
            codec: "vp9".to_string(),
            bitrate: 1_500_000,
            audio_codec: "opus".to_string(),
            container: "webm".to_string(),
            ..Self::default()
        }
    }

    /// Set the scale factor.
    #[must_use]
    pub fn with_scale_factor(mut self, scale_factor: f32) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    /// Set the codec.
    #[must_use]
    pub fn with_codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = codec.into();
        self
    }

    /// Set the bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set the container format.
    #[must_use]
    pub fn with_container(mut self, container: impl Into<String>) -> Self {
        self.container = container.into();
        self
    }

    /// Set hardware acceleration.
    #[must_use]
    pub fn with_hw_accel(mut self, enable: bool) -> Self {
        self.use_hw_accel = enable;
        self
    }

    /// Validate the settings.
    pub fn validate(&self) -> crate::Result<()> {
        if self.scale_factor <= 0.0 || self.scale_factor > 1.0 {
            return Err(crate::ProxyError::InvalidInput(
                "Scale factor must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.bitrate == 0 {
            return Err(crate::ProxyError::InvalidInput(
                "Bitrate must be greater than 0".to_string(),
            ));
        }

        if self.codec.is_empty() {
            return Err(crate::ProxyError::InvalidInput(
                "Codec must be specified".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = ProxyGenerationSettings::default();
        assert_eq!(settings.scale_factor, 0.25);
        assert_eq!(settings.codec, "h264");
        assert_eq!(settings.bitrate, 5_000_000);
    }

    #[test]
    fn test_quarter_res_preset() {
        let settings = ProxyGenerationSettings::quarter_res_h264();
        assert_eq!(settings.scale_factor, 0.25);
        assert_eq!(settings.bitrate, 2_000_000);
    }

    #[test]
    fn test_half_res_preset() {
        let settings = ProxyGenerationSettings::half_res_h264();
        assert_eq!(settings.scale_factor, 0.5);
        assert_eq!(settings.bitrate, 5_000_000);
    }

    #[test]
    fn test_builder_pattern() {
        let settings = ProxyGenerationSettings::default()
            .with_scale_factor(0.5)
            .with_codec("vp9")
            .with_bitrate(8_000_000);

        assert_eq!(settings.scale_factor, 0.5);
        assert_eq!(settings.codec, "vp9");
        assert_eq!(settings.bitrate, 8_000_000);
    }

    #[test]
    fn test_validation() {
        let mut settings = ProxyGenerationSettings::default();
        assert!(settings.validate().is_ok());

        settings.scale_factor = 0.0;
        assert!(settings.validate().is_err());

        settings.scale_factor = 0.5;
        settings.bitrate = 0;
        assert!(settings.validate().is_err());
    }
}
