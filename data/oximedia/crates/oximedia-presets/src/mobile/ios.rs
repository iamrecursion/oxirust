//! iOS device optimization presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all iOS presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        ios_phone_sd(),
        ios_phone_hd(),
        ios_tablet_hd(),
        ios_tablet_uhd(),
    ]
}

/// iOS Phone SD (H.264/AAC).
#[must_use]
pub fn ios_phone_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "ios-phone-sd",
        "iOS Phone SD",
        PresetCategory::Mobile("iOS".to_string()),
    )
    .with_description("Optimized for iPhone SD playback")
    .with_target("iPhone")
    .with_tag("ios")
    .with_tag("iphone")
    .with_tag("sd")
    .with_tag("mobile");

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

/// iOS Phone HD (H.264/AAC).
#[must_use]
pub fn ios_phone_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "ios-phone-hd",
        "iOS Phone HD",
        PresetCategory::Mobile("iOS".to_string()),
    )
    .with_description("Optimized for iPhone HD playback")
    .with_target("iPhone")
    .with_tag("ios")
    .with_tag("iphone")
    .with_tag("hd")
    .with_tag("mobile");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_000_000),
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

/// iOS Tablet HD (H.264/AAC).
#[must_use]
pub fn ios_tablet_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "ios-tablet-hd",
        "iOS Tablet HD",
        PresetCategory::Mobile("iOS".to_string()),
    )
    .with_description("Optimized for iPad HD playback")
    .with_target("iPad")
    .with_tag("ios")
    .with_tag("ipad")
    .with_tag("hd")
    .with_tag("tablet");

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

/// iOS Tablet UHD (H.264/AAC).
#[must_use]
pub fn ios_tablet_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "ios-tablet-uhd",
        "iOS Tablet UHD",
        PresetCategory::Mobile("iOS".to_string()),
    )
    .with_description("Optimized for iPad Pro UHD playback")
    .with_target("iPad Pro")
    .with_tag("ios")
    .with_tag("ipad")
    .with_tag("uhd")
    .with_tag("tablet");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(12_000_000),
        audio_bitrate: Some(192_000),
        width: Some(2732),
        height: Some(2048),
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
    fn test_ios_presets_count() {
        assert_eq!(all_presets().len(), 4);
    }
}
