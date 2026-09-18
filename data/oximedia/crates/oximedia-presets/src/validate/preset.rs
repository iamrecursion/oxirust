//! Preset validation functionality.

use crate::{Preset, PresetError, Result};

/// Validate a preset configuration.
pub fn validate_preset(preset: &Preset) -> Result<()> {
    // Validate resolution
    if let (Some(width), Some(height)) = (preset.config.width, preset.config.height) {
        if width == 0 || height == 0 {
            return Err(PresetError::Validation(
                "Invalid resolution: width or height is zero".to_string(),
            ));
        }
        if width > 7680 || height > 4320 {
            return Err(PresetError::Validation(
                "Resolution exceeds maximum supported (8K)".to_string(),
            ));
        }
    }

    // Validate frame rate
    if let Some((num, den)) = preset.config.frame_rate {
        if den == 0 {
            return Err(PresetError::Validation(
                "Invalid frame rate: denominator is zero".to_string(),
            ));
        }
        let fps = num as f64 / den as f64;
        if fps <= 0.0 || fps > 120.0 {
            return Err(PresetError::Validation(format!(
                "Frame rate {fps} is out of valid range (0-120 fps)"
            )));
        }
    }

    // Validate bitrates
    if let Some(bitrate) = preset.config.video_bitrate {
        if bitrate > 500_000_000 {
            return Err(PresetError::Validation(
                "Video bitrate exceeds reasonable maximum (500 Mbps)".to_string(),
            ));
        }
    }

    if let Some(bitrate) = preset.config.audio_bitrate {
        if bitrate > 1_000_000 {
            return Err(PresetError::Validation(
                "Audio bitrate exceeds reasonable maximum (1 Mbps)".to_string(),
            ));
        }
    }

    Ok(())
}

/// Check if preset is complete (has all required fields).
#[must_use]
pub fn is_complete(preset: &Preset) -> bool {
    preset.config.video_codec.is_some()
        && preset.config.audio_codec.is_some()
        && preset.config.container.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PresetCategory, PresetMetadata};
    use oximedia_transcode::{PresetConfig, QualityMode};

    #[test]
    fn test_valid_preset() {
        let metadata =
            PresetMetadata::new("test", "Test", PresetCategory::Platform("Test".to_string()));
        let config = PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(5_000_000),
            audio_bitrate: Some(192_000),
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        };
        let preset = Preset::new(metadata, config);
        assert!(validate_preset(&preset).is_ok());
    }

    #[test]
    fn test_invalid_resolution() {
        let metadata =
            PresetMetadata::new("test", "Test", PresetCategory::Platform("Test".to_string()));
        let config = PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(5_000_000),
            audio_bitrate: Some(192_000),
            width: Some(0),
            height: Some(1080),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        };
        let preset = Preset::new(metadata, config);
        assert!(validate_preset(&preset).is_err());
    }
}
