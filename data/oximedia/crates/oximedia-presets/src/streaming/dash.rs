//! DASH (Dynamic Adaptive Streaming over HTTP) ABR ladder presets.

use crate::{AbrLadder, Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all DASH presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        dash_240p(),
        dash_360p(),
        dash_480p(),
        dash_720p(),
        dash_1080p(),
        dash_1440p(),
        dash_2160p(),
    ]
}

/// Get complete DASH ABR ladder.
#[must_use]
pub fn dash_abr_ladder() -> AbrLadder {
    AbrLadder::new("DASH Standard Ladder", "DASH")
        .add_rung(240, 500_000, dash_240p())
        .add_rung(360, 1_000_000, dash_360p())
        .add_rung(480, 2_000_000, dash_480p())
        .add_rung(720, 4_000_000, dash_720p())
        .add_rung(1080, 8_000_000, dash_1080p())
        .add_rung(1440, 12_000_000, dash_1440p())
        .add_rung(2160, 20_000_000, dash_2160p())
}

/// DASH 240p rung (H.264/AAC).
#[must_use]
pub fn dash_240p() -> Preset {
    let metadata = PresetMetadata::new(
        "dash-240p",
        "DASH 240p",
        PresetCategory::Streaming("DASH".to_string()),
    )
    .with_description("DASH ABR ladder - 240p @ 500kbps")
    .with_target("DASH")
    .with_tag("dash")
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
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DASH 360p rung (H.264/AAC).
#[must_use]
pub fn dash_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "dash-360p",
        "DASH 360p",
        PresetCategory::Streaming("DASH".to_string()),
    )
    .with_description("DASH ABR ladder - 360p @ 1Mbps")
    .with_target("DASH")
    .with_tag("dash")
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
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DASH 480p rung (H.264/AAC).
#[must_use]
pub fn dash_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "dash-480p",
        "DASH 480p",
        PresetCategory::Streaming("DASH".to_string()),
    )
    .with_description("DASH ABR ladder - 480p @ 2Mbps")
    .with_target("DASH")
    .with_tag("dash")
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
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DASH 720p rung (H.264/AAC).
#[must_use]
pub fn dash_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "dash-720p",
        "DASH 720p",
        PresetCategory::Streaming("DASH".to_string()),
    )
    .with_description("DASH ABR ladder - 720p @ 4Mbps")
    .with_target("DASH")
    .with_tag("dash")
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
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DASH 1080p rung (H.264/AAC).
#[must_use]
pub fn dash_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "dash-1080p",
        "DASH 1080p",
        PresetCategory::Streaming("DASH".to_string()),
    )
    .with_description("DASH ABR ladder - 1080p @ 8Mbps")
    .with_target("DASH")
    .with_tag("dash")
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
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DASH 1440p rung (H.264/AAC).
#[must_use]
pub fn dash_1440p() -> Preset {
    let metadata = PresetMetadata::new(
        "dash-1440p",
        "DASH 1440p",
        PresetCategory::Streaming("DASH".to_string()),
    )
    .with_description("DASH ABR ladder - 1440p @ 12Mbps")
    .with_target("DASH")
    .with_tag("dash")
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
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DASH 2160p/4K rung (H.264/AAC).
#[must_use]
pub fn dash_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "dash-2160p",
        "DASH 2160p/4K",
        PresetCategory::Streaming("DASH".to_string()),
    )
    .with_description("DASH ABR ladder - 2160p @ 20Mbps")
    .with_target("DASH")
    .with_tag("dash")
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
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dash_presets_count() {
        assert_eq!(all_presets().len(), 7);
    }

    #[test]
    fn test_dash_abr_ladder() {
        let ladder = dash_abr_ladder();
        assert_eq!(ladder.rungs.len(), 7);
        assert_eq!(ladder.protocol, "DASH");
    }
}
