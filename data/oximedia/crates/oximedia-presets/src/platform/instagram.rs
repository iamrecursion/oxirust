//! Instagram video encoding presets for feed, stories, and reels.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all Instagram presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        instagram_feed_square(),
        instagram_feed_portrait(),
        instagram_feed_landscape(),
        instagram_story(),
        instagram_reel(),
        instagram_igtv_vertical(),
        instagram_igtv_horizontal(),
    ]
}

/// Instagram Feed Square 1:1 (H.264/AAC).
#[must_use]
pub fn instagram_feed_square() -> Preset {
    let metadata = PresetMetadata::new(
        "instagram-feed-square",
        "Instagram Feed Square (1:1)",
        PresetCategory::Platform("Instagram".to_string()),
    )
    .with_description("Square aspect ratio for Instagram feed")
    .with_target("Instagram Feed")
    .with_tag("instagram")
    .with_tag("feed")
    .with_tag("square")
    .with_tag("1:1");

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

/// Instagram Feed Portrait 4:5 (H.264/AAC).
#[must_use]
pub fn instagram_feed_portrait() -> Preset {
    let metadata = PresetMetadata::new(
        "instagram-feed-portrait",
        "Instagram Feed Portrait (4:5)",
        PresetCategory::Platform("Instagram".to_string()),
    )
    .with_description("Portrait aspect ratio for Instagram feed")
    .with_target("Instagram Feed")
    .with_tag("instagram")
    .with_tag("feed")
    .with_tag("portrait")
    .with_tag("4:5");

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

/// Instagram Feed Landscape 16:9 (H.264/AAC).
#[must_use]
pub fn instagram_feed_landscape() -> Preset {
    let metadata = PresetMetadata::new(
        "instagram-feed-landscape",
        "Instagram Feed Landscape (16:9)",
        PresetCategory::Platform("Instagram".to_string()),
    )
    .with_description("Landscape aspect ratio for Instagram feed")
    .with_target("Instagram Feed")
    .with_tag("instagram")
    .with_tag("feed")
    .with_tag("landscape")
    .with_tag("16:9");

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

/// Instagram Story 9:16 (H.264/AAC).
#[must_use]
pub fn instagram_story() -> Preset {
    let metadata = PresetMetadata::new(
        "instagram-story",
        "Instagram Story (9:16)",
        PresetCategory::Platform("Instagram".to_string()),
    )
    .with_description("Vertical video for Instagram Stories")
    .with_target("Instagram Stories")
    .with_tag("instagram")
    .with_tag("story")
    .with_tag("vertical")
    .with_tag("9:16");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_500_000),
        audio_bitrate: Some(128_000),
        width: Some(1080),
        height: Some(1920),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Instagram Reel 9:16 (H.264/AAC).
#[must_use]
pub fn instagram_reel() -> Preset {
    let metadata = PresetMetadata::new(
        "instagram-reel",
        "Instagram Reel (9:16)",
        PresetCategory::Platform("Instagram".to_string()),
    )
    .with_description("Vertical video for Instagram Reels (up to 90s)")
    .with_target("Instagram Reels")
    .with_tag("instagram")
    .with_tag("reel")
    .with_tag("vertical")
    .with_tag("9:16");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1080),
        height: Some(1920),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Instagram IGTV Vertical 9:16 (H.264/AAC).
#[must_use]
pub fn instagram_igtv_vertical() -> Preset {
    let metadata = PresetMetadata::new(
        "instagram-igtv-vertical",
        "Instagram IGTV Vertical (9:16)",
        PresetCategory::Platform("Instagram".to_string()),
    )
    .with_description("Vertical long-form video for IGTV")
    .with_target("Instagram IGTV")
    .with_tag("instagram")
    .with_tag("igtv")
    .with_tag("vertical")
    .with_tag("9:16");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1080),
        height: Some(1920),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Instagram IGTV Horizontal 16:9 (H.264/AAC).
#[must_use]
pub fn instagram_igtv_horizontal() -> Preset {
    let metadata = PresetMetadata::new(
        "instagram-igtv-horizontal",
        "Instagram IGTV Horizontal (16:9)",
        PresetCategory::Platform("Instagram".to_string()),
    )
    .with_description("Horizontal long-form video for IGTV")
    .with_target("Instagram IGTV")
    .with_tag("instagram")
    .with_tag("igtv")
    .with_tag("horizontal")
    .with_tag("16:9");

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

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instagram_presets_count() {
        assert_eq!(all_presets().len(), 7);
    }

    #[test]
    fn test_instagram_square() {
        let preset = instagram_feed_square();
        assert_eq!(preset.config.width, Some(1080));
        assert_eq!(preset.config.height, Some(1080));
    }

    #[test]
    fn test_instagram_story() {
        let preset = instagram_story();
        assert_eq!(preset.config.width, Some(1080));
        assert_eq!(preset.config.height, Some(1920));
    }
}
