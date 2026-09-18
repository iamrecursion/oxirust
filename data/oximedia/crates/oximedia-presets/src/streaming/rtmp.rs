//! RTMP (Real-Time Messaging Protocol) streaming presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all RTMP presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        rtmp_360p(),
        rtmp_480p(),
        rtmp_720p(),
        rtmp_1080p(),
        rtmp_1080p_60fps(),
        rtmp_low_latency(),
        rtmp_high_quality(),
    ]
}

/// RTMP 360p preset (H.264/AAC) for low-bandwidth streaming.
#[must_use]
pub fn rtmp_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "rtmp-360p",
        "RTMP 360p",
        PresetCategory::Streaming("RTMP".to_string()),
    )
    .with_description("RTMP live streaming - 360p @ 800kbps")
    .with_target("RTMP")
    .with_tag("rtmp")
    .with_tag("live")
    .with_tag("360p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(800_000),
        audio_bitrate: Some(96_000),
        width: Some(640),
        height: Some(360),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// RTMP 480p preset (H.264/AAC).
#[must_use]
pub fn rtmp_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "rtmp-480p",
        "RTMP 480p",
        PresetCategory::Streaming("RTMP".to_string()),
    )
    .with_description("RTMP live streaming - 480p @ 1.5Mbps")
    .with_target("RTMP")
    .with_tag("rtmp")
    .with_tag("live")
    .with_tag("480p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(1_500_000),
        audio_bitrate: Some(128_000),
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// RTMP 720p preset (H.264/AAC).
#[must_use]
pub fn rtmp_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "rtmp-720p",
        "RTMP 720p",
        PresetCategory::Streaming("RTMP".to_string()),
    )
    .with_description("RTMP live streaming - 720p @ 3Mbps")
    .with_target("RTMP")
    .with_tag("rtmp")
    .with_tag("live")
    .with_tag("720p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_000_000),
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// RTMP 1080p preset (H.264/AAC).
#[must_use]
pub fn rtmp_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "rtmp-1080p",
        "RTMP 1080p",
        PresetCategory::Streaming("RTMP".to_string()),
    )
    .with_description("RTMP live streaming - 1080p @ 6Mbps")
    .with_target("RTMP")
    .with_tag("rtmp")
    .with_tag("live")
    .with_tag("1080p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// RTMP 1080p 60fps preset (H.264/AAC) for high-framerate streaming.
#[must_use]
pub fn rtmp_1080p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "rtmp-1080p-60fps",
        "RTMP 1080p 60fps",
        PresetCategory::Streaming("RTMP".to_string()),
    )
    .with_description("RTMP live streaming - 1080p @ 60fps / 9Mbps")
    .with_target("RTMP")
    .with_tag("rtmp")
    .with_tag("live")
    .with_tag("1080p")
    .with_tag("60fps");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(9_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// RTMP low-latency preset optimized for interactive streaming.
#[must_use]
pub fn rtmp_low_latency() -> Preset {
    let metadata = PresetMetadata::new(
        "rtmp-low-latency",
        "RTMP Low Latency 720p",
        PresetCategory::Streaming("RTMP".to_string()),
    )
    .with_description("RTMP low-latency streaming - 720p @ 2.5Mbps, tuned=zerolatency")
    .with_target("RTMP")
    .with_tag("rtmp")
    .with_tag("live")
    .with_tag("low-latency")
    .with_tag("720p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_500_000),
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// RTMP high-quality preset for broadcast-grade streaming.
#[must_use]
pub fn rtmp_high_quality() -> Preset {
    let metadata = PresetMetadata::new(
        "rtmp-high-quality",
        "RTMP High Quality 1080p",
        PresetCategory::Streaming("RTMP".to_string()),
    )
    .with_description("RTMP high-quality streaming - 1080p @ 8Mbps, broadcast grade")
    .with_target("RTMP")
    .with_tag("rtmp")
    .with_tag("live")
    .with_tag("broadcast")
    .with_tag("1080p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(8_000_000),
        audio_bitrate: Some(320_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("flv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtmp_presets_count() {
        assert_eq!(all_presets().len(), 7);
    }

    #[test]
    fn test_rtmp_360p_preset() {
        let preset = rtmp_360p();
        assert_eq!(preset.metadata.id, "rtmp-360p");
        assert!(preset.has_tag("rtmp"));
        assert!(preset.has_tag("live"));
    }

    #[test]
    fn test_rtmp_1080p_60fps_preset() {
        let preset = rtmp_1080p_60fps();
        assert!(preset.has_tag("60fps"));
        assert_eq!(preset.config.frame_rate, Some((60, 1)));
    }

    #[test]
    fn test_rtmp_low_latency_preset() {
        let preset = rtmp_low_latency();
        assert!(preset.has_tag("low-latency"));
    }

    #[test]
    fn test_rtmp_container_is_flv() {
        for preset in all_presets() {
            assert_eq!(
                preset.config.container.as_deref(),
                Some("flv"),
                "RTMP preset {} should use FLV container",
                preset.metadata.id
            );
        }
    }

    #[test]
    fn test_rtmp_presets_category() {
        for preset in all_presets() {
            assert!(
                matches!(
                    &preset.metadata.category,
                    crate::PresetCategory::Streaming(p) if p == "RTMP"
                ),
                "Expected Streaming(RTMP) category for {}",
                preset.metadata.id
            );
        }
    }
}
