//! `YouTube` presets following official recommendations.

use crate::{PresetConfig, QualityMode};

/// `YouTube` 360p (H.264/AAC).
#[must_use]
pub fn youtube_360p() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(1_000_000), // 1 Mbps
        audio_bitrate: Some(128_000),   // 128 kbps
        width: Some(640),
        height: Some(360),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 480p (H.264/AAC).
#[must_use]
pub fn youtube_480p() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_500_000), // 2.5 Mbps
        audio_bitrate: Some(128_000),
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 720p (H.264/AAC).
#[must_use]
pub fn youtube_720p() -> PresetConfig {
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

/// `YouTube` 720p60 (H.264/AAC, 60fps).
#[must_use]
pub fn youtube_720p60() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(7_500_000), // 7.5 Mbps
        audio_bitrate: Some(192_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 1080p (H.264/AAC).
#[must_use]
pub fn youtube_1080p() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(8_000_000), // 8 Mbps
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 1080p60 (H.264/AAC, 60fps).
#[must_use]
pub fn youtube_1080p60() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(12_000_000), // 12 Mbps
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 1440p (H.264/AAC, 2K).
#[must_use]
pub fn youtube_1440p() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(16_000_000), // 16 Mbps
        audio_bitrate: Some(192_000),
        width: Some(2560),
        height: Some(1440),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 1440p60 (H.264/AAC, 2K 60fps).
#[must_use]
pub fn youtube_1440p60() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(24_000_000), // 24 Mbps
        audio_bitrate: Some(192_000),
        width: Some(2560),
        height: Some(1440),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 4K (H.264/AAC, 2160p).
#[must_use]
pub fn youtube_4k() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(35_000_000), // 35 Mbps
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 4K60 (H.264/AAC, 2160p 60fps).
#[must_use]
pub fn youtube_4k60() -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(53_000_000), // 53 Mbps
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 1080p VP9/Opus (modern codec).
#[must_use]
pub fn youtube_1080p_vp9() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(4_500_000), // 4.5 Mbps (VP9 more efficient)
        audio_bitrate: Some(128_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    }
}

/// `YouTube` 4K VP9/Opus (modern codec).
#[must_use]
pub fn youtube_4k_vp9() -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(18_000_000), // 18 Mbps (VP9 more efficient than H.264)
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_youtube_720p() {
        let preset = youtube_720p();
        assert_eq!(preset.width, Some(1280));
        assert_eq!(preset.height, Some(720));
        assert_eq!(preset.video_bitrate, Some(5_000_000));
        assert_eq!(preset.frame_rate, Some((30, 1)));
    }

    #[test]
    fn test_youtube_1080p60() {
        let preset = youtube_1080p60();
        assert_eq!(preset.width, Some(1920));
        assert_eq!(preset.height, Some(1080));
        assert_eq!(preset.frame_rate, Some((60, 1)));
        assert_eq!(preset.video_bitrate, Some(12_000_000));
    }

    #[test]
    fn test_youtube_4k() {
        let preset = youtube_4k();
        assert_eq!(preset.width, Some(3840));
        assert_eq!(preset.height, Some(2160));
        assert_eq!(preset.quality_mode, Some(QualityMode::VeryHigh));
    }

    #[test]
    fn test_youtube_vp9() {
        let preset = youtube_1080p_vp9();
        assert_eq!(preset.video_codec, Some("vp9".to_string()));
        assert_eq!(preset.audio_codec, Some("opus".to_string()));
        assert_eq!(preset.container, Some("webm".to_string()));
    }
}
