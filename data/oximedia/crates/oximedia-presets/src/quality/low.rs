//! Low quality presets for bandwidth-constrained scenarios.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all low quality presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![low_240p(), low_360p(), low_480p()]
}

/// Returns the low quality 240p preset.
#[must_use]
pub fn low_240p() -> Preset {
    let metadata = PresetMetadata::new(
        "low-240p",
        "Low Quality 240p",
        PresetCategory::Quality("Low".to_string()),
    )
    .with_description("Very low bitrate 240p")
    .with_tag("low")
    .with_tag("240p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(300_000),
        audio_bitrate: Some(64_000),
        width: Some(426),
        height: Some(240),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the low quality 360p preset.
#[must_use]
pub fn low_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "low-360p",
        "Low Quality 360p",
        PresetCategory::Quality("Low".to_string()),
    )
    .with_description("Low bitrate 360p")
    .with_tag("low")
    .with_tag("360p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(500_000),
        audio_bitrate: Some(64_000),
        width: Some(640),
        height: Some(360),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the low quality 480p preset.
#[must_use]
pub fn low_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "low-480p",
        "Low Quality 480p",
        PresetCategory::Quality("Low".to_string()),
    )
    .with_description("Low bitrate 480p")
    .with_tag("low")
    .with_tag("480p");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(800_000),
        audio_bitrate: Some(96_000),
        width: Some(854),
        height: Some(480),
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
    fn test_low_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }
}
