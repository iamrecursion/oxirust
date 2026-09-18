//! Compatibility checking for presets.

use crate::{Preset, PresetError, Result};

/// Check if preset is compatible with a target platform.
pub fn check_platform_compatibility(preset: &Preset, platform: &str) -> Result<()> {
    let platform_lower = platform.to_lowercase();

    // Check YouTube compatibility
    if platform_lower.contains("youtube") {
        if let Some(container) = &preset.config.container {
            if !["mp4", "webm", "mov"].contains(&container.as_str()) {
                return Err(PresetError::Compatibility(format!(
                    "Container {container} not supported by YouTube (use mp4, webm, or mov)"
                )));
            }
        }
    }

    // Check Instagram compatibility
    if platform_lower.contains("instagram") {
        if let Some(container) = &preset.config.container {
            if container != "mp4" {
                return Err(PresetError::Compatibility(format!(
                    "Container {container} not supported by Instagram (use mp4)"
                )));
            }
        }
    }

    Ok(())
}

/// Check if preset is compatible with a specific codec.
pub fn check_codec_compatibility(preset: &Preset) -> Result<()> {
    // Check if video codec matches container
    if let (Some(video_codec), Some(container)) =
        (&preset.config.video_codec, &preset.config.container)
    {
        match container.as_str() {
            "mp4" => {
                if !["h264", "hevc", "av1", "mpeg4", "mpeg2video"].contains(&video_codec.as_str()) {
                    return Err(PresetError::Compatibility(format!(
                        "Video codec {video_codec} not typically used with MP4 container"
                    )));
                }
            }
            "webm" => {
                if !["vp8", "vp9", "av1"].contains(&video_codec.as_str()) {
                    return Err(PresetError::Compatibility(format!(
                        "Video codec {video_codec} not compatible with WebM container"
                    )));
                }
            }
            "mkv" => {
                // MKV supports almost everything
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PresetCategory, PresetMetadata};
    use oximedia_transcode::{PresetConfig, QualityMode};

    #[test]
    fn test_youtube_compatibility() {
        let metadata = PresetMetadata::new(
            "test",
            "Test",
            PresetCategory::Platform("YouTube".to_string()),
        );
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
        assert!(check_platform_compatibility(&preset, "YouTube").is_ok());
    }

    #[test]
    fn test_codec_compatibility() {
        let metadata = PresetMetadata::new("test", "Test", PresetCategory::Web("WebM".to_string()));
        let config = PresetConfig {
            video_codec: Some("vp9".to_string()),
            audio_codec: Some("opus".to_string()),
            video_bitrate: Some(3_000_000),
            audio_bitrate: Some(128_000),
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("webm".to_string()),
            audio_channel_layout: None,
        };
        let preset = Preset::new(metadata, config);
        assert!(check_codec_compatibility(&preset).is_ok());
    }
}
