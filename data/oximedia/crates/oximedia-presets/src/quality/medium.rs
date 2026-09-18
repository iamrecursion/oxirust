//! Medium quality presets for balanced quality/size.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all medium quality presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![medium_480p(), medium_720p()]
}

/// Returns the medium quality 480p preset.
#[must_use]
pub fn medium_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "medium-480p",
        "Medium Quality 480p",
        PresetCategory::Quality("Medium".to_string()),
    )
    .with_description("Balanced 480p")
    .with_tag("medium")
    .with_tag("480p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_000_000),
        audio_bitrate: Some(128_000),
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the medium quality 720p preset.
#[must_use]
pub fn medium_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "medium-720p",
        "Medium Quality 720p",
        PresetCategory::Quality("Medium".to_string()),
    )
    .with_description("Balanced 720p")
    .with_tag("medium")
    .with_tag("720p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_000_000),
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_medium_presets_count() {
        assert_eq!(all_presets().len(), 2);
    }
}
