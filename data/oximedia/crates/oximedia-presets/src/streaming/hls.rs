//! HLS (HTTP Live Streaming) ABR ladder presets.

use crate::{AbrLadder, Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all HLS presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        hls_240p(),
        hls_360p(),
        hls_480p(),
        hls_720p(),
        hls_1080p(),
        hls_1440p(),
        hls_2160p(),
    ]
}

/// Get complete HLS ABR ladder.
#[must_use]
pub fn hls_abr_ladder() -> AbrLadder {
    AbrLadder::new("HLS Standard Ladder", "HLS")
        .add_rung(240, 500_000, hls_240p())
        .add_rung(360, 1_000_000, hls_360p())
        .add_rung(480, 2_000_000, hls_480p())
        .add_rung(720, 4_000_000, hls_720p())
        .add_rung(1080, 8_000_000, hls_1080p())
        .add_rung(1440, 12_000_000, hls_1440p())
        .add_rung(2160, 20_000_000, hls_2160p())
}

/// HLS 240p rung (H.264/AAC).
#[must_use]
pub fn hls_240p() -> Preset {
    let metadata = PresetMetadata::new(
        "hls-240p",
        "HLS 240p",
        PresetCategory::Streaming("HLS".to_string()),
    )
    .with_description("HLS ABR ladder - 240p @ 500kbps")
    .with_target("HLS")
    .with_tag("hls")
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
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// HLS 360p rung (H.264/AAC).
#[must_use]
pub fn hls_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "hls-360p",
        "HLS 360p",
        PresetCategory::Streaming("HLS".to_string()),
    )
    .with_description("HLS ABR ladder - 360p @ 1Mbps")
    .with_target("HLS")
    .with_tag("hls")
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
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// HLS 480p rung (H.264/AAC).
#[must_use]
pub fn hls_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "hls-480p",
        "HLS 480p",
        PresetCategory::Streaming("HLS".to_string()),
    )
    .with_description("HLS ABR ladder - 480p @ 2Mbps")
    .with_target("HLS")
    .with_tag("hls")
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
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// HLS 720p rung (H.264/AAC).
#[must_use]
pub fn hls_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "hls-720p",
        "HLS 720p",
        PresetCategory::Streaming("HLS".to_string()),
    )
    .with_description("HLS ABR ladder - 720p @ 4Mbps")
    .with_target("HLS")
    .with_tag("hls")
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
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// HLS 1080p rung (H.264/AAC).
#[must_use]
pub fn hls_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "hls-1080p",
        "HLS 1080p",
        PresetCategory::Streaming("HLS".to_string()),
    )
    .with_description("HLS ABR ladder - 1080p @ 8Mbps")
    .with_target("HLS")
    .with_tag("hls")
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
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// HLS 1440p rung (H.264/AAC).
#[must_use]
pub fn hls_1440p() -> Preset {
    let metadata = PresetMetadata::new(
        "hls-1440p",
        "HLS 1440p",
        PresetCategory::Streaming("HLS".to_string()),
    )
    .with_description("HLS ABR ladder - 1440p @ 12Mbps")
    .with_target("HLS")
    .with_tag("hls")
    .with_tag("1440p")
    .with_tag("abr");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(12_000_000),
        audio_bitrate: Some(192_000),
        width: Some(2560),
        height: Some(1440),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// HLS 2160p/4K rung (H.264/AAC).
#[must_use]
pub fn hls_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "hls-2160p",
        "HLS 2160p/4K",
        PresetCategory::Streaming("HLS".to_string()),
    )
    .with_description("HLS ABR ladder - 2160p @ 20Mbps")
    .with_target("HLS")
    .with_tag("hls")
    .with_tag("2160p")
    .with_tag("4k")
    .with_tag("abr");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(20_000_000),
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
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
    fn test_hls_presets_count() {
        assert_eq!(all_presets().len(), 7);
    }

    #[test]
    fn test_hls_abr_ladder() {
        let ladder = hls_abr_ladder();
        assert_eq!(ladder.rungs.len(), 7);
        assert_eq!(ladder.protocol, "HLS");
    }
}
