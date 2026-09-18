//! Industry-standard presets for common transcoding scenarios.

pub mod archive;
pub mod broadcast;
pub mod streaming;
pub mod vimeo;
pub mod youtube;

use crate::{PresetConfig, QualityMode};

/// Common preset categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetCategory {
    /// `YouTube` presets.
    YouTube,
    /// Vimeo presets.
    Vimeo,
    /// Broadcast/professional presets.
    Broadcast,
    /// Streaming (HLS/DASH) presets.
    Streaming,
    /// Archive/preservation presets.
    Archive,
    /// Social media presets.
    SocialMedia,
}

/// Creates a basic H.264/AAC preset with specified resolution.
#[must_use]
pub fn h264_aac(width: u32, height: u32, video_bitrate: u64, audio_bitrate: u64) -> PresetConfig {
    PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(video_bitrate),
        audio_bitrate: Some(audio_bitrate),
        width: Some(width),
        height: Some(height),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Creates a basic VP9/Opus preset with specified resolution.
#[must_use]
pub fn vp9_opus(width: u32, height: u32, video_bitrate: u64, audio_bitrate: u64) -> PresetConfig {
    PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(video_bitrate),
        audio_bitrate: Some(audio_bitrate),
        width: Some(width),
        height: Some(height),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    }
}

/// Creates a basic AV1/Opus preset with specified resolution.
#[must_use]
pub fn av1_opus(width: u32, height: u32, video_bitrate: u64, audio_bitrate: u64) -> PresetConfig {
    PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(video_bitrate),
        audio_bitrate: Some(audio_bitrate),
        width: Some(width),
        height: Some(height),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    }
}

/// Social media presets for Instagram, `TikTok`, Twitter, etc.
pub mod social {
    use super::{PresetConfig, QualityMode};

    /// Instagram feed (1080x1080, square).
    #[must_use]
    pub fn instagram_feed() -> PresetConfig {
        PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(3_500_000),
            audio_bitrate: Some(128_000),
            width: Some(1080),
            height: Some(1080),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        }
    }

    /// Instagram stories (1080x1920, vertical).
    #[must_use]
    pub fn instagram_stories() -> PresetConfig {
        PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(3_500_000),
            audio_bitrate: Some(128_000),
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        }
    }

    /// `TikTok` (1080x1920, vertical).
    #[must_use]
    pub fn tiktok() -> PresetConfig {
        PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(4_000_000),
            audio_bitrate: Some(128_000),
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        }
    }

    /// Twitter (1280x720, landscape).
    #[must_use]
    pub fn twitter() -> PresetConfig {
        PresetConfig {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(5_000_000),
            audio_bitrate: Some(128_000),
            width: Some(1280),
            height: Some(720),
            frame_rate: Some((30, 1)),
            quality_mode: Some(QualityMode::High),
            container: Some("mp4".to_string()),
            audio_channel_layout: None,
        }
    }

    /// Facebook (1280x720, landscape).
    #[must_use]
    pub fn facebook() -> PresetConfig {
        PresetConfig {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h264_aac_preset() {
        let preset = h264_aac(1920, 1080, 5_000_000, 192_000);
        assert_eq!(preset.video_codec, Some("h264".to_string()));
        assert_eq!(preset.audio_codec, Some("aac".to_string()));
        assert_eq!(preset.width, Some(1920));
        assert_eq!(preset.height, Some(1080));
        assert_eq!(preset.video_bitrate, Some(5_000_000));
        assert_eq!(preset.audio_bitrate, Some(192_000));
    }

    #[test]
    fn test_vp9_opus_preset() {
        let preset = vp9_opus(1280, 720, 2_500_000, 128_000);
        assert_eq!(preset.video_codec, Some("vp9".to_string()));
        assert_eq!(preset.audio_codec, Some("opus".to_string()));
        assert_eq!(preset.container, Some("webm".to_string()));
    }

    #[test]
    fn test_av1_opus_preset() {
        let preset = av1_opus(1920, 1080, 4_000_000, 128_000);
        assert_eq!(preset.video_codec, Some("av1".to_string()));
        assert_eq!(preset.audio_codec, Some("opus".to_string()));
        assert_eq!(preset.quality_mode, Some(QualityMode::High));
    }

    #[test]
    fn test_social_instagram_feed() {
        let preset = social::instagram_feed();
        assert_eq!(preset.width, Some(1080));
        assert_eq!(preset.height, Some(1080)); // Square
        assert_eq!(preset.video_codec, Some("h264".to_string()));
    }

    #[test]
    fn test_social_instagram_stories() {
        let preset = social::instagram_stories();
        assert_eq!(preset.width, Some(1080));
        assert_eq!(preset.height, Some(1920)); // Vertical
    }

    #[test]
    fn test_social_tiktok() {
        let preset = social::tiktok();
        assert_eq!(preset.width, Some(1080));
        assert_eq!(preset.height, Some(1920)); // Vertical
    }

    #[test]
    fn test_social_twitter() {
        let preset = social::twitter();
        assert_eq!(preset.width, Some(1280));
        assert_eq!(preset.height, Some(720));
    }
}
