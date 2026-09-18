//! Streaming presets for HLS, DASH, and adaptive bitrate.

use crate::{AbrLadder, AbrLadderBuilder, AbrStrategy, PresetConfig, QualityMode};

/// HLS (HTTP Live Streaming) ABR ladder.
#[must_use]
pub fn hls_ladder() -> AbrLadder {
    AbrLadder::hls_standard()
}

/// DASH (MPEG-DASH) ABR ladder.
#[must_use]
pub fn dash_ladder() -> AbrLadder {
    AbrLadder::hls_standard() // Similar to HLS
}

/// Low-latency streaming preset (720p).
#[must_use]
pub fn low_latency_720p() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(3_000_000), // 3 Mbps
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Low-latency streaming preset (1080p).
#[must_use]
pub fn low_latency_1080p() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000), // 6 Mbps
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// CMAF (Common Media Application Format) preset.
#[must_use]
pub fn cmaf() -> PresetConfig {
    PresetConfig {
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
    }
}

/// Creates a custom ABR ladder for streaming.
#[must_use]
pub fn custom_ladder(
    min_resolution: (u32, u32),
    max_resolution: (u32, u32),
    strategy: AbrStrategy,
) -> AbrLadder {
    AbrLadderBuilder::new(strategy)
        .min_resolution(min_resolution.0, min_resolution.1)
        .max_resolution(max_resolution.0, max_resolution.1)
        .build()
}

/// Twitch streaming preset (1080p60).
#[must_use]
pub fn twitch_1080p60() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(6_000_000), // 6 Mbps (Twitch recommendation)
        audio_bitrate: Some(160_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Twitch streaming preset (720p60).
#[must_use]
pub fn twitch_720p60() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(4_500_000), // 4.5 Mbps
        audio_bitrate: Some(160_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hls_ladder() {
        let ladder = hls_ladder();
        assert!(ladder.rung_count() > 0);
        assert_eq!(ladder.strategy, AbrStrategy::AppleHls);
    }

    #[test]
    fn test_low_latency_720p() {
        let preset = low_latency_720p();
        assert_eq!(preset.width, Some(1280));
        assert_eq!(preset.height, Some(720));
        assert_eq!(preset.quality_mode, Some(QualityMode::Medium));
    }

    #[test]
    fn test_twitch_1080p60() {
        let preset = twitch_1080p60();
        assert_eq!(preset.width, Some(1920));
        assert_eq!(preset.height, Some(1080));
        assert_eq!(preset.frame_rate, Some((60, 1)));
        assert_eq!(preset.video_bitrate, Some(6_000_000));
    }

    #[test]
    fn test_custom_ladder() {
        let ladder = custom_ladder((640, 360), (1920, 1080), AbrStrategy::Conservative);
        assert_eq!(ladder.strategy, AbrStrategy::Conservative);
    }
}
