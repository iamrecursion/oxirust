//! Facebook video encoding presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all Facebook presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        facebook_feed_sd(),
        facebook_feed_720p(),
        facebook_feed_1080p(),
        facebook_story_720p(),
        facebook_story_1080p(),
        facebook_ad_720p(),
        facebook_ad_1080p(),
        facebook_live_720p(),
        facebook_live_1080p(),
    ]
}

/// Facebook Feed SD (H.264/AAC).
#[must_use]
pub fn facebook_feed_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-feed-sd",
        "Facebook Feed SD",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("Standard definition feed video")
    .with_target("Facebook Feed")
    .with_tag("facebook")
    .with_tag("feed")
    .with_tag("sd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(1_500_000),
        audio_bitrate: Some(128_000),
        width: Some(640),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Facebook Feed 720p (H.264/AAC).
#[must_use]
pub fn facebook_feed_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-feed-720p",
        "Facebook Feed 720p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("HD feed video optimized for auto-play")
    .with_target("Facebook Feed")
    .with_tag("facebook")
    .with_tag("feed")
    .with_tag("720p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_000_000),
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

/// Facebook Feed 1080p (H.264/AAC).
#[must_use]
pub fn facebook_feed_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-feed-1080p",
        "Facebook Feed 1080p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("Full HD feed video")
    .with_target("Facebook Feed")
    .with_tag("facebook")
    .with_tag("feed")
    .with_tag("1080p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000),
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

/// Facebook Story 720p (H.264/AAC) - Vertical 9:16.
#[must_use]
pub fn facebook_story_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-story-720p",
        "Facebook Story 720p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("Vertical HD story video (9:16)")
    .with_target("Facebook Stories")
    .with_tag("facebook")
    .with_tag("story")
    .with_tag("720p")
    .with_tag("vertical");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_000_000),
        audio_bitrate: Some(128_000),
        width: Some(720),
        height: Some(1280),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Facebook Story 1080p (H.264/AAC) - Vertical 9:16.
#[must_use]
pub fn facebook_story_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-story-1080p",
        "Facebook Story 1080p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("Vertical Full HD story video (9:16)")
    .with_target("Facebook Stories")
    .with_tag("facebook")
    .with_tag("story")
    .with_tag("1080p")
    .with_tag("vertical");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
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

/// Facebook Ad 720p (H.264/AAC).
#[must_use]
pub fn facebook_ad_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-ad-720p",
        "Facebook Ad 720p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("HD video ad format")
    .with_target("Facebook Ads")
    .with_tag("facebook")
    .with_tag("ad")
    .with_tag("720p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
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

/// Facebook Ad 1080p (H.264/AAC).
#[must_use]
pub fn facebook_ad_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-ad-1080p",
        "Facebook Ad 1080p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("Full HD video ad format")
    .with_target("Facebook Ads")
    .with_tag("facebook")
    .with_tag("ad")
    .with_tag("1080p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(8_000_000),
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

/// Facebook Live 720p (H.264/AAC).
#[must_use]
pub fn facebook_live_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-live-720p",
        "Facebook Live 720p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("HD live streaming preset")
    .with_target("Facebook Live")
    .with_tag("facebook")
    .with_tag("live")
    .with_tag("720p")
    .with_tag("streaming");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_000_000),
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

/// Facebook Live 1080p (H.264/AAC).
#[must_use]
pub fn facebook_live_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "facebook-live-1080p",
        "Facebook Live 1080p",
        PresetCategory::Platform("Facebook".to_string()),
    )
    .with_description("Full HD live streaming preset")
    .with_target("Facebook Live")
    .with_tag("facebook")
    .with_tag("live")
    .with_tag("1080p")
    .with_tag("streaming");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facebook_presets_count() {
        assert_eq!(all_presets().len(), 9);
    }

    #[test]
    fn test_facebook_story_aspect_ratio() {
        let preset = facebook_story_1080p();
        assert_eq!(preset.config.width, Some(1080));
        assert_eq!(preset.config.height, Some(1920));
    }
}
