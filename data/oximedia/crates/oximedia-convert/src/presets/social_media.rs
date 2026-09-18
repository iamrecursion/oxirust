// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Social media platform conversion presets.
//!
//! Platform-specific presets that conform to each platform's recommended
//! upload specifications including resolution, bitrate, frame rate, aspect
//! ratio, and duration constraints. All codecs are patent-free.

use super::{AudioPresetSettings, EncodingSpeed, Preset, VideoPresetSettings};
use crate::formats::{AudioCodec, ChannelLayout, ContainerFormat, VideoCodec};
use crate::Result;
use serde::{Deserialize, Serialize};

/// Social media platform identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SocialPlatform {
    /// YouTube (all variants)
    YouTube,
    /// TikTok
    TikTok,
    /// Instagram (all variants)
    Instagram,
    /// Twitter / X
    Twitter,
}

impl SocialPlatform {
    /// Get human-readable platform name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::YouTube => "YouTube",
            Self::TikTok => "TikTok",
            Self::Instagram => "Instagram",
            Self::Twitter => "Twitter/X",
        }
    }

    /// Get maximum upload file size in bytes for the platform.
    #[must_use]
    pub const fn max_file_size_bytes(self) -> u64 {
        match self {
            Self::YouTube => 256 * 1024 * 1024 * 1024, // 256 GB
            Self::TikTok => 287 * 1024 * 1024,         // 287 MB
            Self::Instagram => 650 * 1024 * 1024,      // 650 MB (feed)
            Self::Twitter => 512 * 1024 * 1024,        // 512 MB
        }
    }

    /// Get maximum video duration in seconds.
    #[must_use]
    pub const fn max_duration_seconds(self) -> u64 {
        match self {
            Self::YouTube => 43200,  // 12 hours
            Self::TikTok => 600,     // 10 minutes
            Self::Instagram => 3600, // 60 minutes (IGTV)
            Self::Twitter => 140,    // 2:20
        }
    }

    /// Get all available preset variants for this platform.
    #[must_use]
    pub fn available_presets(self) -> Vec<&'static str> {
        match self {
            Self::YouTube => vec![
                "youtube-shorts",
                "youtube-1080p-60",
                "youtube-4k-hdr",
                "youtube-live",
            ],
            Self::TikTok => vec!["tiktok-standard", "tiktok-hd", "tiktok-ads"],
            Self::Instagram => vec![
                "instagram-feed-square",
                "instagram-feed-portrait",
                "instagram-feed-landscape",
                "instagram-reels",
                "instagram-stories",
            ],
            Self::Twitter => vec!["twitter-landscape", "twitter-portrait", "twitter-square"],
        }
    }
}

// ── YouTube Presets ──────────────────────────────────────────────────────────

/// YouTube Shorts preset (9:16 vertical, up to 60 seconds).
pub fn youtube_shorts() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube Shorts".to_string(),
        description: "YouTube Shorts: 1080x1920 9:16, max 60 seconds".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(6_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// YouTube 1080p 60fps preset (high quality for gaming/sports content).
pub fn youtube_1080p_60fps() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube 1080p 60fps".to_string(),
        description: "YouTube 1080p at 60fps for high-motion content".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(60.0),
            bitrate: Some(12_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(192_000),
        }),
    })
}

/// YouTube 4K HDR preset (premium quality).
pub fn youtube_4k_hdr() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube 4K HDR".to_string(),
        description: "YouTube 4K HDR for premium quality uploads".to_string(),
        container: ContainerFormat::Matroska,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Av1,
            width: Some(3840),
            height: Some(2160),
            frame_rate: Some(30.0),
            bitrate: Some(50_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Slow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Surround5_1,
            bitrate: Some(384_000),
        }),
    })
}

/// YouTube Live preset (optimized for live streaming).
pub fn youtube_live() -> Result<Preset> {
    Ok(Preset {
        name: "YouTube Live".to_string(),
        description: "YouTube Live streaming preset, fast encoding".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp8,
            width: Some(1920),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(4_500_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

// ── TikTok Presets ──────────────────────────────────────────────────────────

/// TikTok standard preset (1080x1920, 9:16).
pub fn tiktok_standard() -> Result<Preset> {
    Ok(Preset {
        name: "TikTok Standard".to_string(),
        description: "TikTok standard quality 1080x1920 9:16".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(4_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// TikTok HD preset (higher bitrate for better quality).
pub fn tiktok_hd() -> Result<Preset> {
    Ok(Preset {
        name: "TikTok HD".to_string(),
        description: "TikTok high definition 1080x1920 with higher bitrate".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(6_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Medium,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(192_000),
        }),
    })
}

/// TikTok Ads preset (optimized for advertising content).
pub fn tiktok_ads() -> Result<Preset> {
    Ok(Preset {
        name: "TikTok Ads".to_string(),
        description: "TikTok advertising preset, max quality within platform limits".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(8_000_000),
            quality: None,
            two_pass: true,
            speed: EncodingSpeed::Slow,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(192_000),
        }),
    })
}

// ── Instagram Presets ───────────────────────────────────────────────────────

/// Instagram feed square preset (1:1 aspect ratio).
pub fn instagram_feed_square() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Feed Square".to_string(),
        description: "Instagram feed 1080x1080 square (1:1)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1080),
            frame_rate: Some(30.0),
            bitrate: Some(3_500_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Instagram feed portrait preset (4:5 aspect ratio).
pub fn instagram_feed_portrait() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Feed Portrait".to_string(),
        description: "Instagram feed 1080x1350 portrait (4:5)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1350),
            frame_rate: Some(30.0),
            bitrate: Some(4_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Instagram feed landscape preset (1.91:1 aspect ratio).
pub fn instagram_feed_landscape() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Feed Landscape".to_string(),
        description: "Instagram feed 1080x566 landscape (1.91:1)".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(566),
            frame_rate: Some(30.0),
            bitrate: Some(3_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Instagram Reels preset (9:16 aspect ratio, up to 90 seconds).
pub fn instagram_reels() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Reels".to_string(),
        description: "Instagram Reels 1080x1920 (9:16), max 90 seconds".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(4_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Instagram Stories preset (9:16 aspect ratio, up to 60 seconds).
pub fn instagram_stories() -> Result<Preset> {
    Ok(Preset {
        name: "Instagram Stories".to_string(),
        description: "Instagram Stories 1080x1920 (9:16), max 60 seconds".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1080),
            height: Some(1920),
            frame_rate: Some(30.0),
            bitrate: Some(3_500_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

// ── Twitter/X Presets ───────────────────────────────────────────────────────

/// Twitter landscape preset (16:9).
pub fn twitter_landscape() -> Result<Preset> {
    Ok(Preset {
        name: "Twitter Landscape".to_string(),
        description: "Twitter landscape video 1280x720 (16:9), max 2:20".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(1280),
            height: Some(720),
            frame_rate: Some(30.0),
            bitrate: Some(5_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Twitter portrait preset (9:16).
pub fn twitter_portrait() -> Result<Preset> {
    Ok(Preset {
        name: "Twitter Portrait".to_string(),
        description: "Twitter portrait video 720x1280 (9:16), max 2:20".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(720),
            height: Some(1280),
            frame_rate: Some(30.0),
            bitrate: Some(4_000_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

/// Twitter square preset (1:1).
pub fn twitter_square() -> Result<Preset> {
    Ok(Preset {
        name: "Twitter Square".to_string(),
        description: "Twitter square video 720x720 (1:1), max 2:20".to_string(),
        container: ContainerFormat::Mp4,
        video: Some(VideoPresetSettings {
            codec: VideoCodec::Vp9,
            width: Some(720),
            height: Some(720),
            frame_rate: Some(30.0),
            bitrate: Some(3_500_000),
            quality: None,
            two_pass: false,
            speed: EncodingSpeed::Fast,
        }),
        audio: Some(AudioPresetSettings {
            codec: AudioCodec::Opus,
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            bitrate: Some(128_000),
        }),
    })
}

// ── Lookup ──────────────────────────────────────────────────────────────────

/// Look up a social media preset by name.
pub fn social_media_preset(name: &str) -> Result<Preset> {
    match name.to_lowercase().as_str() {
        "youtube-shorts" => youtube_shorts(),
        "youtube-1080p-60" | "youtube-1080p-60fps" => youtube_1080p_60fps(),
        "youtube-4k-hdr" => youtube_4k_hdr(),
        "youtube-live" => youtube_live(),
        "tiktok-standard" | "tiktok-std" => tiktok_standard(),
        "tiktok-hd" => tiktok_hd(),
        "tiktok-ads" => tiktok_ads(),
        "instagram-feed-square" | "ig-square" => instagram_feed_square(),
        "instagram-feed-portrait" | "ig-portrait" => instagram_feed_portrait(),
        "instagram-feed-landscape" | "ig-landscape" => instagram_feed_landscape(),
        "instagram-reels" | "ig-reels" => instagram_reels(),
        "instagram-stories" | "ig-stories" => instagram_stories(),
        "twitter-landscape" | "x-landscape" => twitter_landscape(),
        "twitter-portrait" | "x-portrait" => twitter_portrait(),
        "twitter-square" | "x-square" => twitter_square(),
        _ => Err(crate::ConversionError::InvalidProfile(format!(
            "Unknown social media preset: {name}"
        ))),
    }
}

/// Get all available social media preset names.
#[must_use]
pub fn all_social_media_presets() -> Vec<&'static str> {
    vec![
        "youtube-shorts",
        "youtube-1080p-60",
        "youtube-4k-hdr",
        "youtube-live",
        "tiktok-standard",
        "tiktok-hd",
        "tiktok-ads",
        "instagram-feed-square",
        "instagram-feed-portrait",
        "instagram-feed-landscape",
        "instagram-reels",
        "instagram-stories",
        "twitter-landscape",
        "twitter-portrait",
        "twitter-square",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_social_presets_valid() {
        for name in all_social_media_presets() {
            let preset = social_media_preset(name);
            assert!(preset.is_ok(), "Preset '{}' should be valid", name);
            let p = preset.expect("checked above");
            assert!(!p.name.is_empty());
            assert!(!p.description.is_empty());
        }
    }

    #[test]
    fn test_unknown_social_preset() {
        assert!(social_media_preset("nonexistent").is_err());
    }

    #[test]
    fn test_youtube_shorts_specs() {
        let p = youtube_shorts().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.width, Some(1080));
        assert_eq!(v.height, Some(1920));
        assert_eq!(v.frame_rate, Some(30.0));
        assert!(!v.two_pass); // Shorts are fast uploads
    }

    #[test]
    fn test_youtube_1080p_60fps_specs() {
        let p = youtube_1080p_60fps().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.width, Some(1920));
        assert_eq!(v.height, Some(1080));
        assert_eq!(v.frame_rate, Some(60.0));
        assert!(v.two_pass);
    }

    #[test]
    fn test_youtube_4k_hdr_specs() {
        let p = youtube_4k_hdr().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.width, Some(3840));
        assert_eq!(v.height, Some(2160));
        assert_eq!(v.codec, VideoCodec::Av1); // AV1 for HDR
        let a = p.audio.expect("should have audio");
        assert_eq!(a.channels, ChannelLayout::Surround5_1);
    }

    #[test]
    fn test_youtube_live_fast_encoding() {
        let p = youtube_live().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.speed, EncodingSpeed::Fast);
        assert!(!v.two_pass); // Live can't do two-pass
    }

    #[test]
    fn test_tiktok_standard_vertical() {
        let p = tiktok_standard().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.width, Some(1080));
        assert_eq!(v.height, Some(1920));
        // Aspect ratio: 9:16 (vertical)
        assert!(v.height > v.width);
    }

    #[test]
    fn test_tiktok_hd_higher_bitrate() {
        let std_p = tiktok_standard().expect("should be valid");
        let hd_p = tiktok_hd().expect("should be valid");
        let std_br = std_p.video.expect("v").bitrate.unwrap_or(0);
        let hd_br = hd_p.video.expect("v").bitrate.unwrap_or(0);
        assert!(hd_br > std_br);
    }

    #[test]
    fn test_tiktok_ads_best_quality() {
        let p = tiktok_ads().expect("should be valid");
        let v = p.video.expect("should have video");
        assert!(v.two_pass);
        assert_eq!(v.speed, EncodingSpeed::Slow);
    }

    #[test]
    fn test_instagram_aspect_ratios() {
        let square = instagram_feed_square().expect("should be valid");
        let portrait = instagram_feed_portrait().expect("should be valid");
        let landscape = instagram_feed_landscape().expect("should be valid");

        let sv = square.video.expect("v");
        assert_eq!(sv.width, sv.height); // 1:1

        let pv = portrait.video.expect("v");
        assert!(pv.height.unwrap_or(0) > pv.width.unwrap_or(0)); // 4:5

        let lv = landscape.video.expect("v");
        assert!(lv.width.unwrap_or(0) > lv.height.unwrap_or(0)); // 1.91:1
    }

    #[test]
    fn test_instagram_reels_vertical() {
        let p = instagram_reels().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.width, Some(1080));
        assert_eq!(v.height, Some(1920));
    }

    #[test]
    fn test_twitter_landscape() {
        let p = twitter_landscape().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.width, Some(1280));
        assert_eq!(v.height, Some(720));
    }

    #[test]
    fn test_twitter_square() {
        let p = twitter_square().expect("should be valid");
        let v = p.video.expect("should have video");
        assert_eq!(v.width, Some(720));
        assert_eq!(v.height, Some(720));
    }

    #[test]
    fn test_platform_properties() {
        assert_eq!(SocialPlatform::YouTube.name(), "YouTube");
        assert_eq!(SocialPlatform::TikTok.name(), "TikTok");
        assert_eq!(SocialPlatform::Instagram.name(), "Instagram");
        assert_eq!(SocialPlatform::Twitter.name(), "Twitter/X");
    }

    #[test]
    fn test_platform_max_file_size() {
        assert!(
            SocialPlatform::YouTube.max_file_size_bytes()
                > SocialPlatform::TikTok.max_file_size_bytes()
        );
        assert!(SocialPlatform::TikTok.max_file_size_bytes() > 0);
    }

    #[test]
    fn test_platform_max_duration() {
        assert!(
            SocialPlatform::YouTube.max_duration_seconds()
                > SocialPlatform::Twitter.max_duration_seconds()
        );
        assert_eq!(SocialPlatform::Twitter.max_duration_seconds(), 140);
    }

    #[test]
    fn test_platform_available_presets() {
        for platform in &[
            SocialPlatform::YouTube,
            SocialPlatform::TikTok,
            SocialPlatform::Instagram,
            SocialPlatform::Twitter,
        ] {
            let presets = platform.available_presets();
            assert!(!presets.is_empty(), "{:?} should have presets", platform);
            // Each preset name should be resolvable
            for name in &presets {
                assert!(
                    social_media_preset(name).is_ok(),
                    "Preset '{}' from {:?} should be valid",
                    name,
                    platform
                );
            }
        }
    }

    #[test]
    fn test_alias_lookup() {
        // Test aliases
        assert!(social_media_preset("tiktok-std").is_ok());
        assert!(social_media_preset("ig-square").is_ok());
        assert!(social_media_preset("ig-reels").is_ok());
        assert!(social_media_preset("x-landscape").is_ok());
        assert!(social_media_preset("youtube-1080p-60fps").is_ok());
    }

    #[test]
    fn test_all_presets_use_patent_free_codecs() {
        for name in all_social_media_presets() {
            let p = social_media_preset(name).expect("should be valid");
            if let Some(v) = &p.video {
                assert!(
                    matches!(
                        v.codec,
                        VideoCodec::Av1 | VideoCodec::Vp9 | VideoCodec::Vp8 | VideoCodec::Theora
                    ),
                    "Preset '{}' uses non-patent-free video codec",
                    name
                );
            }
            if let Some(a) = &p.audio {
                assert!(
                    matches!(
                        a.codec,
                        AudioCodec::Opus | AudioCodec::Vorbis | AudioCodec::Flac | AudioCodec::Pcm
                    ),
                    "Preset '{}' uses non-patent-free audio codec",
                    name
                );
            }
        }
    }

    #[test]
    fn test_all_presets_have_audio() {
        // Social media videos should always include audio
        for name in all_social_media_presets() {
            let p = social_media_preset(name).expect("should be valid");
            assert!(
                p.audio.is_some(),
                "Preset '{}' should include audio settings",
                name
            );
        }
    }
}
