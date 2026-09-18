//! SRT (Secure Reliable Transport) streaming presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all SRT presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        srt_360p(),
        srt_480p(),
        srt_720p(),
        srt_1080p(),
        srt_1080p_60fps(),
        srt_low_latency(),
        srt_broadcast(),
    ]
}

/// SRT 360p preset (H.264/AAC) for low-bandwidth contribution.
#[must_use]
pub fn srt_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "srt-360p",
        "SRT 360p",
        PresetCategory::Streaming("SRT".to_string()),
    )
    .with_description("SRT contribution - 360p @ 1Mbps")
    .with_target("SRT")
    .with_tag("srt")
    .with_tag("contribution")
    .with_tag("360p");

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

/// SRT 480p preset (H.264/AAC).
#[must_use]
pub fn srt_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "srt-480p",
        "SRT 480p",
        PresetCategory::Streaming("SRT".to_string()),
    )
    .with_description("SRT contribution - 480p @ 2Mbps")
    .with_target("SRT")
    .with_tag("srt")
    .with_tag("contribution")
    .with_tag("480p");

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

/// SRT 720p preset (H.264/AAC).
#[must_use]
pub fn srt_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "srt-720p",
        "SRT 720p",
        PresetCategory::Streaming("SRT".to_string()),
    )
    .with_description("SRT contribution - 720p @ 4Mbps")
    .with_target("SRT")
    .with_tag("srt")
    .with_tag("contribution")
    .with_tag("720p");

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

/// SRT 1080p preset (H.264/AAC).
#[must_use]
pub fn srt_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "srt-1080p",
        "SRT 1080p",
        PresetCategory::Streaming("SRT".to_string()),
    )
    .with_description("SRT contribution - 1080p @ 8Mbps")
    .with_target("SRT")
    .with_tag("srt")
    .with_tag("contribution")
    .with_tag("1080p");

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

/// SRT 1080p 60fps preset for sports and high-motion content.
#[must_use]
pub fn srt_1080p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "srt-1080p-60fps",
        "SRT 1080p 60fps",
        PresetCategory::Streaming("SRT".to_string()),
    )
    .with_description("SRT contribution - 1080p @ 60fps / 12Mbps")
    .with_target("SRT")
    .with_tag("srt")
    .with_tag("contribution")
    .with_tag("1080p")
    .with_tag("60fps")
    .with_tag("sports");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(12_000_000),
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// SRT low-latency preset for real-time contribution over unreliable networks.
#[must_use]
pub fn srt_low_latency() -> Preset {
    let metadata = PresetMetadata::new(
        "srt-low-latency",
        "SRT Low Latency 720p",
        PresetCategory::Streaming("SRT".to_string()),
    )
    .with_description("SRT low-latency contribution - 720p @ 3Mbps, minimal buffering")
    .with_target("SRT")
    .with_tag("srt")
    .with_tag("low-latency")
    .with_tag("720p")
    .with_tag("real-time");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_000_000),
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// SRT broadcast-quality preset for professional contribution links.
#[must_use]
pub fn srt_broadcast() -> Preset {
    let metadata = PresetMetadata::new(
        "srt-broadcast",
        "SRT Broadcast 1080p",
        PresetCategory::Streaming("SRT".to_string()),
    )
    .with_description("SRT broadcast-quality contribution - 1080p @ 15Mbps")
    .with_target("SRT")
    .with_tag("srt")
    .with_tag("broadcast")
    .with_tag("1080p")
    .with_tag("professional");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(15_000_000),
        audio_bitrate: Some(320_000),
        width: Some(1920),
        height: Some(1080),
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
    fn test_srt_presets_count() {
        assert_eq!(all_presets().len(), 7);
    }

    #[test]
    fn test_srt_360p_preset() {
        let preset = srt_360p();
        assert_eq!(preset.metadata.id, "srt-360p");
        assert!(preset.has_tag("srt"));
    }

    #[test]
    fn test_srt_1080p_60fps_preset() {
        let preset = srt_1080p_60fps();
        assert!(preset.has_tag("60fps"));
        assert!(preset.has_tag("sports"));
        assert_eq!(preset.config.frame_rate, Some((60, 1)));
    }

    #[test]
    fn test_srt_broadcast_bitrate() {
        let preset = srt_broadcast();
        assert_eq!(preset.config.video_bitrate, Some(15_000_000));
        assert!(preset.has_tag("professional"));
    }

    #[test]
    fn test_srt_container_is_mpegts() {
        for preset in all_presets() {
            assert_eq!(
                preset.config.container.as_deref(),
                Some("mpegts"),
                "SRT preset {} should use MPEG-TS container",
                preset.metadata.id
            );
        }
    }

    #[test]
    fn test_srt_presets_category() {
        for preset in all_presets() {
            assert!(
                matches!(
                    &preset.metadata.category,
                    crate::PresetCategory::Streaming(p) if p == "SRT"
                ),
                "Expected Streaming(SRT) category for {}",
                preset.metadata.id
            );
        }
    }
}
