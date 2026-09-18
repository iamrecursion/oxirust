//! Opus audio codec profiles.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all Opus codec presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![opus_voice(), opus_music(), opus_hq()]
}

/// Returns the Opus voice-optimized preset.
#[must_use]
pub fn opus_voice() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-voice",
        "Opus Voice",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("Opus optimized for voice")
    .with_tag("opus")
    .with_tag("voice");
    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(32_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::Medium),
        container: Some("ogg".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the Opus music-optimized preset.
#[must_use]
pub fn opus_music() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-music",
        "Opus Music",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("Opus optimized for music")
    .with_tag("opus")
    .with_tag("music");
    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(128_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::High),
        container: Some("ogg".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

/// Returns the Opus high-quality preset.
#[must_use]
pub fn opus_hq() -> Preset {
    let metadata = PresetMetadata::new(
        "opus-hq",
        "Opus High Quality",
        PresetCategory::Codec("Opus".to_string()),
    )
    .with_description("Opus high quality music")
    .with_tag("opus")
    .with_tag("music")
    .with_tag("hq");
    let config = PresetConfig {
        video_codec: None,
        audio_codec: Some("opus".to_string()),
        video_bitrate: None,
        audio_bitrate: Some(256_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::High),
        container: Some("ogg".to_string()),
        audio_channel_layout: None,
    };
    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_opus_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }
}
