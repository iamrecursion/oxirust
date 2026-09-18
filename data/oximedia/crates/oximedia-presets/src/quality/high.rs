//! High quality presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all high quality presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![high_720p(), high_1080p()]
}

/// Returns the high quality 720p preset.
#[must_use]
pub fn high_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "high-720p",
        "High Quality 720p",
        PresetCategory::Quality("High".to_string()),
    )
    .with_description("High quality 720p")
    .with_tag("high")
    .with_tag("720p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the high quality 1080p preset.
#[must_use]
pub fn high_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "high-1080p",
        "High Quality 1080p",
        PresetCategory::Quality("High".to_string()),
    )
    .with_description("High quality 1080p")
    .with_tag("high")
    .with_tag("1080p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(10_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_high_presets_count() {
        assert_eq!(all_presets().len(), 2);
    }
}
