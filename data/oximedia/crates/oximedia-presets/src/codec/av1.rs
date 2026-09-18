//! AV1 codec-specific profiles.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all AV1 codec presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![av1_720p(), av1_1080p(), av1_2160p()]
}

/// Returns the AV1 720p preset.
#[must_use]
pub fn av1_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-720p",
        "AV1 720p",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description("AV1 720p encoding")
    .with_tag("av1")
    .with_tag("720p");
    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(1_500_000),
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the AV1 1080p preset.
#[must_use]
pub fn av1_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-1080p",
        "AV1 1080p",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description("AV1 1080p encoding")
    .with_tag("av1")
    .with_tag("1080p");
    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
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
    Preset::new(metadata, config)
}

/// Returns the AV1 2160p (4K) preset.
#[must_use]
pub fn av1_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-2160p",
        "AV1 2160p/4K",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description("AV1 4K encoding")
    .with_tag("av1")
    .with_tag("2160p")
    .with_tag("4k");
    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(10_000_000),
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_av1_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }
}
