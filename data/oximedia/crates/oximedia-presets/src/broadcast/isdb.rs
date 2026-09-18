//! ISDB (Integrated Services Digital Broadcasting) presets for Japanese broadcast standards.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all ISDB presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![isdb_t_sd(), isdb_t_hd(), isdb_t_one_seg()]
}

/// ISDB-T SD (MPEG-2/AAC).
#[must_use]
pub fn isdb_t_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "isdb-t-sd",
        "ISDB-T SD",
        PresetCategory::Broadcast("ISDB".to_string()),
    )
    .with_description("ISDB-T standard definition terrestrial broadcast")
    .with_target("ISDB-T")
    .with_tag("isdb")
    .with_tag("isdb-t")
    .with_tag("sd")
    .with_tag("terrestrial");

    let config = PresetConfig {
        video_codec: Some("mpeg2video".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_000_000),
        audio_bitrate: Some(192_000),
        width: Some(720),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ISDB-T HD (H.264/AAC).
#[must_use]
pub fn isdb_t_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "isdb-t-hd",
        "ISDB-T HD",
        PresetCategory::Broadcast("ISDB".to_string()),
    )
    .with_description("ISDB-T HD terrestrial broadcast")
    .with_target("ISDB-T")
    .with_tag("isdb")
    .with_tag("isdb-t")
    .with_tag("hd")
    .with_tag("terrestrial");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(12_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ISDB-T One-Seg (H.264/AAC) - Mobile segment.
#[must_use]
pub fn isdb_t_one_seg() -> Preset {
    let metadata = PresetMetadata::new(
        "isdb-t-one-seg",
        "ISDB-T One-Seg",
        PresetCategory::Broadcast("ISDB".to_string()),
    )
    .with_description("ISDB-T One-Seg mobile broadcast")
    .with_target("ISDB-T One-Seg")
    .with_tag("isdb")
    .with_tag("isdb-t")
    .with_tag("one-seg")
    .with_tag("mobile");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(350_000),
        audio_bitrate: Some(64_000),
        width: Some(320),
        height: Some(240),
        frame_rate: Some((15, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isdb_presets_count() {
        assert_eq!(all_presets().len(), 3);
    }

    #[test]
    fn test_isdb_one_seg() {
        let preset = isdb_t_one_seg();
        assert!(preset.has_tag("mobile"));
        assert_eq!(preset.config.width, Some(320));
    }
}
