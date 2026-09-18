//! Archive and preservation presets for long-term storage.

use crate::{PresetConfig, QualityMode};

/// Lossless archival preset using VP9 (best compression).
///
/// Note: In a real implementation, this would use FFV1 for true lossless.
/// VP9 lossless mode is used here as it's part of our royalty-free codec set.
#[must_use]
pub fn lossless_vp9() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: None, // Lossless mode
        audio_bitrate: Some(256_000),
        width: None,      // Preserve source resolution
        height: None,     // Preserve source resolution
        frame_rate: None, // Preserve source frame rate
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    }
}

/// High-quality archival preset (VP9, near-lossless).
#[must_use]
pub fn high_quality_vp9() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(50_000_000), // 50 Mbps
        audio_bitrate: Some(256_000),
        width: None,      // Preserve source
        height: None,     // Preserve source
        frame_rate: None, // Preserve source
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    }
}

/// High-quality archival preset (AV1, smaller file size).
#[must_use]
pub fn high_quality_av1() -> PresetConfig {
    PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(30_000_000), // 30 Mbps (AV1 more efficient)
        audio_bitrate: Some(256_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    }
}

/// Master archival preset (highest quality).
#[must_use]
pub fn master_archive() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(100_000_000), // 100 Mbps
        audio_bitrate: Some(512_000),     // High-quality audio
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    }
}

/// Intermediate archival preset (balance of quality and size).
#[must_use]
pub fn intermediate_archive() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(25_000_000), // 25 Mbps
        audio_bitrate: Some(192_000),
        width: None,
        height: None,
        frame_rate: None,
        quality_mode: Some(QualityMode::High),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    }
}

/// Preservation preset with metadata (1080p).
#[must_use]
pub fn preservation_1080p() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(20_000_000), // 20 Mbps
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((24, 1)), // Film standard
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    }
}

/// Preservation preset with metadata (4K).
#[must_use]
pub fn preservation_4k() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(60_000_000), // 60 Mbps
        audio_bitrate: Some(320_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossless_vp9() {
        let preset = lossless_vp9();
        assert_eq!(preset.video_codec, Some("vp9".to_string()));
        assert_eq!(preset.container, Some("mkv".to_string()));
        assert!(preset.width.is_none()); // Preserve source
    }

    #[test]
    fn test_high_quality_av1() {
        let preset = high_quality_av1();
        assert_eq!(preset.video_codec, Some("av1".to_string()));
        assert_eq!(preset.quality_mode, Some(QualityMode::VeryHigh));
    }

    #[test]
    fn test_master_archive() {
        let preset = master_archive();
        assert_eq!(preset.video_bitrate, Some(100_000_000));
        assert_eq!(preset.audio_bitrate, Some(512_000));
    }

    #[test]
    fn test_preservation_4k() {
        let preset = preservation_4k();
        assert_eq!(preset.width, Some(3840));
        assert_eq!(preset.height, Some(2160));
        assert_eq!(preset.frame_rate, Some((24, 1)));
    }
}
