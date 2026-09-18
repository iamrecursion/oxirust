//! Social media feed post presets (square/landscape).

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all social media feed presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![feed_square(), feed_landscape(), feed_portrait()]
}

/// Returns the square (1:1) feed post preset.
#[must_use]
pub fn feed_square() -> Preset {
    let metadata = PresetMetadata::new(
        "feed-square",
        "Feed Square (1:1)",
        PresetCategory::Social("Feed".to_string()),
    )
    .with_description("Square feed post")
    .with_tag("feed")
    .with_tag("1:1")
    .with_tag("square");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_500_000),
        audio_bitrate: Some(128_000),
        width: Some(1080),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the landscape (16:9) feed post preset.
#[must_use]
pub fn feed_landscape() -> Preset {
    let metadata = PresetMetadata::new(
        "feed-landscape",
        "Feed Landscape (16:9)",
        PresetCategory::Social("Feed".to_string()),
    )
    .with_description("Landscape feed post")
    .with_tag("feed")
    .with_tag("16:9")
    .with_tag("landscape");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_500_000),
        audio_bitrate: Some(128_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the portrait (4:5) feed post preset.
#[must_use]
pub fn feed_portrait() -> Preset {
    let metadata = PresetMetadata::new(
        "feed-portrait",
        "Feed Portrait (4:5)",
        PresetCategory::Social("Feed".to_string()),
    )
    .with_description("Portrait feed post")
    .with_tag("feed")
    .with_tag("4:5")
    .with_tag("portrait");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_500_000),
        audio_bitrate: Some(128_000),
        width: Some(1080),
        height: Some(1350),
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
    fn test_feed_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }
}
