//! Twitter video encoding presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all Twitter presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        twitter_video_sd(),
        twitter_video_720p(),
        twitter_video_1080p(),
        twitter_ad_720p(),
        twitter_ad_1080p(),
    ]
}

/// Twitter Video SD (H.264/AAC).
#[must_use]
pub fn twitter_video_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "twitter-video-sd",
        "Twitter Video SD",
        PresetCategory::Platform("Twitter".to_string()),
    )
    .with_description("Standard definition video for Twitter")
    .with_target("Twitter Video")
    .with_tag("twitter")
    .with_tag("sd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(1_000_000),
        audio_bitrate: Some(128_000),
        width: Some(640),
        height: Some(360),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Twitter Video 720p (H.264/AAC).
#[must_use]
pub fn twitter_video_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "twitter-video-720p",
        "Twitter Video 720p",
        PresetCategory::Platform("Twitter".to_string()),
    )
    .with_description("HD video for Twitter (max 512MB)")
    .with_target("Twitter Video")
    .with_tag("twitter")
    .with_tag("720p")
    .with_tag("hd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_000_000),
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Twitter Video 1080p (H.264/AAC).
#[must_use]
pub fn twitter_video_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "twitter-video-1080p",
        "Twitter Video 1080p",
        PresetCategory::Platform("Twitter".to_string()),
    )
    .with_description("Full HD video for Twitter (max 512MB)")
    .with_target("Twitter Video")
    .with_tag("twitter")
    .with_tag("1080p")
    .with_tag("full-hd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
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

/// Twitter Ad 720p (H.264/AAC).
#[must_use]
pub fn twitter_ad_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "twitter-ad-720p",
        "Twitter Ad 720p",
        PresetCategory::Platform("Twitter".to_string()),
    )
    .with_description("HD video ad for Twitter")
    .with_target("Twitter Ads")
    .with_tag("twitter")
    .with_tag("ad")
    .with_tag("720p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_000_000),
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

/// Twitter Ad 1080p (H.264/AAC).
#[must_use]
pub fn twitter_ad_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "twitter-ad-1080p",
        "Twitter Ad 1080p",
        PresetCategory::Platform("Twitter".to_string()),
    )
    .with_description("Full HD video ad for Twitter")
    .with_target("Twitter Ads")
    .with_tag("twitter")
    .with_tag("ad")
    .with_tag("1080p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000),
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
    fn test_twitter_presets_count() {
        assert_eq!(all_presets().len(), 5);
    }

    #[test]
    fn test_twitter_1080p() {
        let preset = twitter_video_1080p();
        assert_eq!(preset.config.width, Some(1920));
        assert_eq!(preset.config.height, Some(1080));
    }
}
