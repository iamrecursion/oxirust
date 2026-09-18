//! TikTok video encoding presets optimized for vertical mobile content.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all TikTok presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![tiktok_standard(), tiktok_hd(), tiktok_high_quality()]
}

/// TikTok Standard Quality 9:16 (H.264/AAC).
#[must_use]
pub fn tiktok_standard() -> Preset {
    let metadata = PresetMetadata::new(
        "tiktok-standard",
        "TikTok Standard",
        PresetCategory::Platform("TikTok".to_string()),
    )
    .with_description("Standard quality vertical video for TikTok")
    .with_target("TikTok")
    .with_tag("tiktok")
    .with_tag("vertical")
    .with_tag("9:16")
    .with_tag("standard");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_500_000),
        audio_bitrate: Some(192_000), // High quality audio for music
        width: Some(720),
        height: Some(1280),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// TikTok HD Quality 9:16 (H.264/AAC).
#[must_use]
pub fn tiktok_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "tiktok-hd",
        "TikTok HD",
        PresetCategory::Platform("TikTok".to_string()),
    )
    .with_description("HD vertical video for TikTok")
    .with_target("TikTok HD")
    .with_tag("tiktok")
    .with_tag("vertical")
    .with_tag("9:16")
    .with_tag("hd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_000_000),
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

/// TikTok High Quality 9:16 (H.264/AAC) - 60fps.
#[must_use]
pub fn tiktok_high_quality() -> Preset {
    let metadata = PresetMetadata::new(
        "tiktok-high-quality",
        "TikTok High Quality",
        PresetCategory::Platform("TikTok".to_string()),
    )
    .with_description("High quality 60fps vertical video for TikTok")
    .with_target("TikTok High Quality")
    .with_tag("tiktok")
    .with_tag("vertical")
    .with_tag("9:16")
    .with_tag("60fps")
    .with_tag("high-quality");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1080),
        height: Some(1920),
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
    fn test_tiktok_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }

    #[test]
    fn test_tiktok_aspect_ratio() {
        let preset = tiktok_hd();
        assert_eq!(preset.config.width, Some(1080));
        assert_eq!(preset.config.height, Some(1920));
    }
}
