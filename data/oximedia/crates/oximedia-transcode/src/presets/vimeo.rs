//! Vimeo presets following platform recommendations.

use crate::{PresetConfig, QualityMode};

/// Vimeo SD (H.264/AAC, 480p).
#[must_use]
pub fn vimeo_sd() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_000_000), // 2 Mbps
        audio_bitrate: Some(128_000),
        width: Some(640),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Vimeo HD (H.264/AAC, 720p).
#[must_use]
pub fn vimeo_hd() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000), // 5 Mbps
        audio_bitrate: Some(192_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Vimeo Full HD (H.264/AAC, 1080p).
#[must_use]
pub fn vimeo_full_hd() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(10_000_000), // 10 Mbps
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Vimeo 4K (H.264/AAC, 2160p).
#[must_use]
pub fn vimeo_4k() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(40_000_000), // 40 Mbps
        audio_bitrate: Some(256_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Vimeo Pro (H.264/AAC, 1080p, high bitrate for professionals).
#[must_use]
pub fn vimeo_pro() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(20_000_000), // 20 Mbps
        audio_bitrate: Some(320_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((24, 1)), // Film standard
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vimeo_hd() {
        let preset = vimeo_hd();
        assert_eq!(preset.width, Some(1280));
        assert_eq!(preset.height, Some(720));
        assert_eq!(preset.video_bitrate, Some(5_000_000));
    }

    #[test]
    fn test_vimeo_full_hd() {
        let preset = vimeo_full_hd();
        assert_eq!(preset.width, Some(1920));
        assert_eq!(preset.height, Some(1080));
        assert_eq!(preset.quality_mode, Some(QualityMode::VeryHigh));
    }

    #[test]
    fn test_vimeo_4k() {
        let preset = vimeo_4k();
        assert_eq!(preset.width, Some(3840));
        assert_eq!(preset.height, Some(2160));
        assert_eq!(preset.video_bitrate, Some(40_000_000));
    }

    #[test]
    fn test_vimeo_pro() {
        let preset = vimeo_pro();
        assert_eq!(preset.frame_rate, Some((24, 1)));
        assert_eq!(preset.audio_bitrate, Some(320_000));
    }
}
