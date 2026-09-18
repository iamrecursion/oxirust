//! Highest quality presets (renamed from ultra to avoid "Ultra" naming).

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all highest quality presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        highest_1080p(),
        highest_1440p(),
        highest_2160p(),
        highest_4320p(),
    ]
}

/// Returns the highest quality 1080p preset.
#[must_use]
pub fn highest_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "highest-1080p",
        "Highest Quality 1080p",
        PresetCategory::Quality("Highest".to_string()),
    )
    .with_description("Maximum quality 1080p")
    .with_tag("highest")
    .with_tag("1080p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(15_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the highest quality 1440p preset.
#[must_use]
pub fn highest_1440p() -> Preset {
    let metadata = PresetMetadata::new(
        "highest-1440p",
        "Highest Quality 1440p",
        PresetCategory::Quality("Highest".to_string()),
    )
    .with_description("Maximum quality 1440p")
    .with_tag("highest")
    .with_tag("1440p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(25_000_000),
        audio_bitrate: Some(256_000),
        width: Some(2560),
        height: Some(1440),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the highest quality 2160p (4K) preset.
#[must_use]
pub fn highest_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "highest-2160p",
        "Highest Quality 2160p/4K",
        PresetCategory::Quality("Highest".to_string()),
    )
    .with_description("Maximum quality 4K")
    .with_tag("highest")
    .with_tag("2160p")
    .with_tag("4k");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(50_000_000),
        audio_bitrate: Some(320_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the highest quality 4320p (8K) preset.
#[must_use]
pub fn highest_4320p() -> Preset {
    let metadata = PresetMetadata::new(
        "highest-4320p",
        "Highest Quality 4320p/8K",
        PresetCategory::Quality("Highest".to_string()),
    )
    .with_description("Maximum quality 8K")
    .with_tag("highest")
    .with_tag("4320p")
    .with_tag("8k");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(100_000_000),
        audio_bitrate: Some(320_000),
        width: Some(7680),
        height: Some(4320),
        frame_rate: Some((60, 1)),
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
    fn test_highest_presets_count() {
        assert_eq!(all_presets().len(), 4);
    }
}
