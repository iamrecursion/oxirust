//! Social media stories format presets (vertical 9:16).

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all social media stories presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![stories_hd(), stories_uhd()]
}

/// Returns the vertical HD stories preset.
#[must_use]
pub fn stories_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "stories-hd",
        "Stories HD (9:16)",
        PresetCategory::Social("Stories".to_string()),
    )
    .with_description("Vertical HD stories")
    .with_tag("stories")
    .with_tag("9:16")
    .with_tag("vertical");
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

/// Returns the vertical UHD stories preset.
#[must_use]
pub fn stories_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "stories-uhd",
        "Stories UHD (9:16)",
        PresetCategory::Social("Stories".to_string()),
    )
    .with_description("Vertical UHD stories")
    .with_tag("stories")
    .with_tag("9:16")
    .with_tag("vertical")
    .with_tag("uhd");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1080),
        height: Some(2160),
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
    fn test_stories_presets_count() {
        assert_eq!(all_presets().len(), 2);
    }
}
