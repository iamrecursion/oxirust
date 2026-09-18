//! Android device optimization presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all Android presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        android_phone_sd(),
        android_phone_hd(),
        android_tablet_hd(),
        android_tablet_uhd(),
    ]
}

/// Android Phone SD (H.264/AAC).
#[must_use]
pub fn android_phone_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "android-phone-sd",
        "Android Phone SD",
        PresetCategory::Mobile("Android".to_string()),
    )
    .with_description("Optimized for Android phone SD playback")
    .with_target("Android Phone")
    .with_tag("android")
    .with_tag("phone")
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

/// Android Phone HD (H.264/AAC).
#[must_use]
pub fn android_phone_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "android-phone-hd",
        "Android Phone HD",
        PresetCategory::Mobile("Android".to_string()),
    )
    .with_description("Optimized for Android phone HD playback")
    .with_target("Android Phone")
    .with_tag("android")
    .with_tag("phone")
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

/// Android Tablet HD (H.264/AAC).
#[must_use]
pub fn android_tablet_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "android-tablet-hd",
        "Android Tablet HD",
        PresetCategory::Mobile("Android".to_string()),
    )
    .with_description("Optimized for Android tablet HD playback")
    .with_target("Android Tablet")
    .with_tag("android")
    .with_tag("tablet")
    .with_tag("hd");

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

/// Android Tablet UHD (H.264/AAC).
#[must_use]
pub fn android_tablet_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "android-tablet-uhd",
        "Android Tablet UHD",
        PresetCategory::Mobile("Android".to_string()),
    )
    .with_description("Optimized for Android tablet UHD playback")
    .with_target("Android Tablet")
    .with_tag("android")
    .with_tag("tablet")
    .with_tag("uhd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(12_000_000),
        audio_bitrate: Some(192_000),
        width: Some(2560),
        height: Some(1600),
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
    fn test_android_presets_count() {
        assert_eq!(all_presets().len(), 4);
    }
}
