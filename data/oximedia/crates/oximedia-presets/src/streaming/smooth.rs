//! Microsoft Smooth Streaming ABR ladder presets.

use crate::{AbrLadder, Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all Smooth Streaming presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        smooth_240p(),
        smooth_360p(),
        smooth_480p(),
        smooth_720p(),
        smooth_1080p(),
    ]
}

/// Get complete Smooth Streaming ABR ladder.
#[must_use]
pub fn smooth_abr_ladder() -> AbrLadder {
    AbrLadder::new("Smooth Streaming Ladder", "SmoothStreaming")
        .add_rung(240, 500_000, smooth_240p())
        .add_rung(360, 1_000_000, smooth_360p())
        .add_rung(480, 2_000_000, smooth_480p())
        .add_rung(720, 4_000_000, smooth_720p())
        .add_rung(1080, 8_000_000, smooth_1080p())
}

/// Smooth Streaming 240p rung (H.264/AAC).
#[must_use]
pub fn smooth_240p() -> Preset {
    let metadata = PresetMetadata::new(
        "smooth-240p",
        "Smooth Streaming 240p",
        PresetCategory::Streaming("SmoothStreaming".to_string()),
    )
    .with_description("Smooth Streaming ABR - 240p @ 500kbps")
    .with_target("Smooth Streaming")
    .with_tag("smooth")
    .with_tag("240p")
    .with_tag("abr");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(500_000),
        audio_bitrate: Some(64_000),
        width: Some(426),
        height: Some(240),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("ismv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Smooth Streaming 360p rung (H.264/AAC).
#[must_use]
pub fn smooth_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "smooth-360p",
        "Smooth Streaming 360p",
        PresetCategory::Streaming("SmoothStreaming".to_string()),
    )
    .with_description("Smooth Streaming ABR - 360p @ 1Mbps")
    .with_target("Smooth Streaming")
    .with_tag("smooth")
    .with_tag("360p")
    .with_tag("abr");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(1_000_000),
        audio_bitrate: Some(96_000),
        width: Some(640),
        height: Some(360),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("ismv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Smooth Streaming 480p rung (H.264/AAC).
#[must_use]
pub fn smooth_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "smooth-480p",
        "Smooth Streaming 480p",
        PresetCategory::Streaming("SmoothStreaming".to_string()),
    )
    .with_description("Smooth Streaming ABR - 480p @ 2Mbps")
    .with_target("Smooth Streaming")
    .with_tag("smooth")
    .with_tag("480p")
    .with_tag("abr");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_000_000),
        audio_bitrate: Some(128_000),
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("ismv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Smooth Streaming 720p rung (H.264/AAC).
#[must_use]
pub fn smooth_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "smooth-720p",
        "Smooth Streaming 720p",
        PresetCategory::Streaming("SmoothStreaming".to_string()),
    )
    .with_description("Smooth Streaming ABR - 720p @ 4Mbps")
    .with_target("Smooth Streaming")
    .with_tag("smooth")
    .with_tag("720p")
    .with_tag("abr");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_000_000),
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("ismv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Smooth Streaming 1080p rung (H.264/AAC).
#[must_use]
pub fn smooth_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "smooth-1080p",
        "Smooth Streaming 1080p",
        PresetCategory::Streaming("SmoothStreaming".to_string()),
    )
    .with_description("Smooth Streaming ABR - 1080p @ 8Mbps")
    .with_target("Smooth Streaming")
    .with_tag("smooth")
    .with_tag("1080p")
    .with_tag("abr");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(8_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("ismv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_presets_count() {
        assert_eq!(all_presets().len(), 5);
    }

    #[test]
    fn test_smooth_abr_ladder() {
        let ladder = smooth_abr_ladder();
        assert_eq!(ladder.rungs.len(), 5);
        assert_eq!(ladder.protocol, "SmoothStreaming");
    }
}
