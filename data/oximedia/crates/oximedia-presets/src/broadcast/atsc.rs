//! ATSC (Advanced Television Systems Committee) broadcast presets.
//!
//! Supports both ATSC 1.0 (traditional HD broadcast) and ATSC 3.0 (NextGen TV).

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all ATSC presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        atsc_1_0_720p(),
        atsc_1_0_1080i(),
        atsc_1_0_1080p(),
        atsc_3_0_1080p(),
        atsc_3_0_4k(),
    ]
}

/// ATSC 1.0 720p (MPEG-2/AC-3).
#[must_use]
pub fn atsc_1_0_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "atsc-1.0-720p",
        "ATSC 1.0 720p",
        PresetCategory::Broadcast("ATSC".to_string()),
    )
    .with_description("ATSC 1.0 HD 720p broadcast standard")
    .with_target("ATSC 1.0")
    .with_tag("atsc")
    .with_tag("broadcast")
    .with_tag("720p");

    let config = PresetConfig {
        video_codec: Some("mpeg2video".to_string()),
        audio_codec: Some("ac3".to_string()),
        video_bitrate: Some(15_000_000),
        audio_bitrate: Some(384_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ATSC 1.0 1080i (MPEG-2/AC-3).
#[must_use]
pub fn atsc_1_0_1080i() -> Preset {
    let metadata = PresetMetadata::new(
        "atsc-1.0-1080i",
        "ATSC 1.0 1080i",
        PresetCategory::Broadcast("ATSC".to_string()),
    )
    .with_description("ATSC 1.0 Full HD 1080i broadcast standard")
    .with_target("ATSC 1.0")
    .with_tag("atsc")
    .with_tag("broadcast")
    .with_tag("1080i")
    .with_tag("interlaced");

    let config = PresetConfig {
        video_codec: Some("mpeg2video".to_string()),
        audio_codec: Some("ac3".to_string()),
        video_bitrate: Some(19_400_000), // Maximum ATSC bitrate
        audio_bitrate: Some(384_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ATSC 1.0 1080p (H.264/AC-3).
#[must_use]
pub fn atsc_1_0_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "atsc-1.0-1080p",
        "ATSC 1.0 1080p",
        PresetCategory::Broadcast("ATSC".to_string()),
    )
    .with_description("ATSC 1.0 Full HD 1080p with H.264")
    .with_target("ATSC 1.0")
    .with_tag("atsc")
    .with_tag("broadcast")
    .with_tag("1080p")
    .with_tag("h264");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("ac3".to_string()),
        video_bitrate: Some(18_000_000),
        audio_bitrate: Some(384_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ATSC 3.0 1080p (HEVC/AC-4).
#[must_use]
pub fn atsc_3_0_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "atsc-3.0-1080p",
        "ATSC 3.0 1080p",
        PresetCategory::Broadcast("ATSC".to_string()),
    )
    .with_description("ATSC 3.0 NextGen TV Full HD")
    .with_target("ATSC 3.0")
    .with_tag("atsc")
    .with_tag("atsc3")
    .with_tag("nextgen")
    .with_tag("1080p")
    .with_tag("hevc");

    let config = PresetConfig {
        video_codec: Some("hevc".to_string()),
        audio_codec: Some("ac4".to_string()),
        video_bitrate: Some(10_000_000),
        audio_bitrate: Some(448_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ATSC 3.0 4K (HEVC/AC-4).
#[must_use]
pub fn atsc_3_0_4k() -> Preset {
    let metadata = PresetMetadata::new(
        "atsc-3.0-4k",
        "ATSC 3.0 4K",
        PresetCategory::Broadcast("ATSC".to_string()),
    )
    .with_description("ATSC 3.0 NextGen TV 4K UHD")
    .with_target("ATSC 3.0")
    .with_tag("atsc")
    .with_tag("atsc3")
    .with_tag("nextgen")
    .with_tag("4k")
    .with_tag("uhd")
    .with_tag("hevc");

    let config = PresetConfig {
        video_codec: Some("hevc".to_string()),
        audio_codec: Some("ac4".to_string()),
        video_bitrate: Some(20_000_000),
        audio_bitrate: Some(512_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atsc_presets_count() {
        assert_eq!(all_presets().len(), 5);
    }

    #[test]
    fn test_atsc_1_0_1080i() {
        let preset = atsc_1_0_1080i();
        assert_eq!(preset.config.video_codec, Some("mpeg2video".to_string()));
        assert_eq!(preset.config.width, Some(1920));
        assert_eq!(preset.config.height, Some(1080));
    }

    #[test]
    fn test_atsc_3_0_4k() {
        let preset = atsc_3_0_4k();
        assert_eq!(preset.config.video_codec, Some("hevc".to_string()));
        assert!(preset.has_tag("4k"));
    }
}
