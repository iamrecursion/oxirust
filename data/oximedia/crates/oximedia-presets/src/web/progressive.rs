//! Progressive download presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all progressive download presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![progressive_sd(), progressive_hd()]
}

/// Returns the progressive download SD preset.
#[must_use]
pub fn progressive_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "progressive-sd",
        "Progressive SD",
        PresetCategory::Web("Progressive".to_string()),
    )
    .with_description("Progressive download SD")
    .with_tag("progressive")
    .with_tag("sd");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_000_000),
        audio_bitrate: Some(128_000),
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the progressive download HD preset.
#[must_use]
pub fn progressive_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "progressive-hd",
        "Progressive HD",
        PresetCategory::Web("Progressive".to_string()),
    )
    .with_description("Progressive download HD")
    .with_tag("progressive")
    .with_tag("hd");
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
    fn test_progressive_presets_count() {
        assert_eq!(all_presets().len(), 2);
    }
}
