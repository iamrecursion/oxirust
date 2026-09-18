//! VP8 codec-specific profiles.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all VP8 codec presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![vp8_480p(), vp8_720p(), vp8_1080p()]
}

/// Returns the VP8 480p preset.
#[must_use]
pub fn vp8_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "vp8-480p",
        "VP8 480p",
        PresetCategory::Codec("VP8".to_string()),
    )
    .with_description("VP8 480p encoding")
    .with_tag("vp8")
    .with_tag("480p");
    let config = PresetConfig {
        video_codec: Some("vp8".to_string()),
        audio_codec: Some("vorbis".to_string()),
        video_bitrate: Some(1_000_000),
        audio_bitrate: Some(128_000),
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the VP8 720p preset.
#[must_use]
pub fn vp8_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "vp8-720p",
        "VP8 720p",
        PresetCategory::Codec("VP8".to_string()),
    )
    .with_description("VP8 720p encoding")
    .with_tag("vp8")
    .with_tag("720p");
    let config = PresetConfig {
        video_codec: Some("vp8".to_string()),
        audio_codec: Some("vorbis".to_string()),
        video_bitrate: Some(2_000_000),
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

/// Returns the VP8 1080p preset.
#[must_use]
pub fn vp8_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "vp8-1080p",
        "VP8 1080p",
        PresetCategory::Codec("VP8".to_string()),
    )
    .with_description("VP8 1080p encoding")
    .with_tag("vp8")
    .with_tag("1080p");
    let config = PresetConfig {
        video_codec: Some("vp8".to_string()),
        audio_codec: Some("vorbis".to_string()),
        video_bitrate: Some(4_000_000),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vp8_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }
}
