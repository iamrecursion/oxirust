//! VP9 codec-specific profiles.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all VP9 codec presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![vp9_720p(), vp9_1080p(), vp9_2160p()]
}

/// Returns the VP9 720p preset.
#[must_use]
pub fn vp9_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "vp9-720p",
        "VP9 720p",
        PresetCategory::Codec("VP9".to_string()),
    )
    .with_description("VP9 720p encoding")
    .with_tag("vp9")
    .with_tag("720p");
    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
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

/// Returns the VP9 1080p preset.
#[must_use]
pub fn vp9_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "vp9-1080p",
        "VP9 1080p",
        PresetCategory::Codec("VP9".to_string()),
    )
    .with_description("VP9 1080p encoding")
    .with_tag("vp9")
    .with_tag("1080p");
    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
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

/// Returns the VP9 2160p (4K) preset.
#[must_use]
pub fn vp9_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "vp9-2160p",
        "VP9 2160p/4K",
        PresetCategory::Codec("VP9".to_string()),
    )
    .with_description("VP9 4K encoding")
    .with_tag("vp9")
    .with_tag("2160p")
    .with_tag("4k");
    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(15_000_000),
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
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
    fn test_vp9_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }
}
