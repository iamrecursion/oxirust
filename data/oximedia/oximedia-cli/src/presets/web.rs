//! Web platform presets.
//!
//! Presets optimized for web platforms including:
//! - YouTube (various quality levels)
//! - Vimeo (high quality)
//! - Social media (Twitter, Facebook, Instagram)

use super::{AudioConfig, Preset, PresetCategory, VideoConfig};

/// Get all web platform presets.
pub fn get_web_presets() -> Vec<Preset> {
    vec![
        // YouTube presets
        youtube_4k(),
        youtube_1440p(),
        youtube_1080p(),
        youtube_720p(),
        youtube_480p(),
        youtube_360p(),
        // Vimeo presets
        vimeo_4k(),
        vimeo_1080p(),
        vimeo_720p(),
        // Social media presets
        twitter_1080p(),
        twitter_720p(),
        facebook_1080p(),
        facebook_720p(),
        instagram_feed(),
        instagram_story(),
        instagram_reels(),
    ]
}

/// YouTube 4K (2160p) preset.
fn youtube_4k() -> Preset {
    Preset {
        name: "youtube-4k".to_string(),
        description: "YouTube 4K (2160p) upload - VP9 codec with high bitrate".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("35M".to_string()),
            crf: None,
            width: Some(3840),
            height: Some(2160),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("53M".to_string()),
            min_bitrate: None,
            buffer_size: Some("70M".to_string()),
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("192k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec!["youtube".to_string(), "4k".to_string(), "uhd".to_string()],
    }
}

/// YouTube 1440p preset.
fn youtube_1440p() -> Preset {
    Preset {
        name: "youtube-1440p".to_string(),
        description: "YouTube 1440p (QHD) upload - VP9 codec".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("16M".to_string()),
            crf: None,
            width: Some(2560),
            height: Some(1440),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("24M".to_string()),
            min_bitrate: None,
            buffer_size: Some("32M".to_string()),
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("192k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "youtube".to_string(),
            "1440p".to_string(),
            "qhd".to_string(),
        ],
    }
}

/// YouTube 1080p preset.
fn youtube_1080p() -> Preset {
    Preset {
        name: "youtube-1080p".to_string(),
        description: "YouTube 1080p (Full HD) upload - VP9 codec with optimal quality".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("8M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("12M".to_string()),
            min_bitrate: None,
            buffer_size: Some("16M".to_string()),
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("128k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "youtube".to_string(),
            "1080p".to_string(),
            "fhd".to_string(),
        ],
    }
}

/// YouTube 720p preset.
fn youtube_720p() -> Preset {
    Preset {
        name: "youtube-720p".to_string(),
        description: "YouTube 720p (HD) upload - VP9 codec".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("5M".to_string()),
            crf: None,
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("7M".to_string()),
            min_bitrate: None,
            buffer_size: Some("10M".to_string()),
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("128k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec!["youtube".to_string(), "720p".to_string(), "hd".to_string()],
    }
}

/// YouTube 480p preset.
fn youtube_480p() -> Preset {
    Preset {
        name: "youtube-480p".to_string(),
        description: "YouTube 480p (SD) upload - VP9 codec".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("2.5M".to_string()),
            crf: None,
            width: Some(854),
            height: Some(480),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("4M".to_string()),
            min_bitrate: None,
            buffer_size: Some("5M".to_string()),
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("96k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec!["youtube".to_string(), "480p".to_string(), "sd".to_string()],
    }
}

/// YouTube 360p preset.
fn youtube_360p() -> Preset {
    Preset {
        name: "youtube-360p".to_string(),
        description: "YouTube 360p upload - VP9 codec for low bandwidth".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("1M".to_string()),
            crf: None,
            width: Some(640),
            height: Some(360),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("1.5M".to_string()),
            min_bitrate: None,
            buffer_size: Some("2M".to_string()),
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("96k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "youtube".to_string(),
            "360p".to_string(),
            "low-bandwidth".to_string(),
        ],
    }
}

/// Vimeo 4K preset.
fn vimeo_4k() -> Preset {
    Preset {
        name: "vimeo-4k".to_string(),
        description: "Vimeo 4K upload - High quality VP9".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(15),
            width: Some(3840),
            height: Some(2160),
            fps: Some(30.0),
            preset: Some("slow".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("256k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "vimeo".to_string(),
            "4k".to_string(),
            "high-quality".to_string(),
        ],
    }
}

/// Vimeo 1080p preset.
fn vimeo_1080p() -> Preset {
    Preset {
        name: "vimeo-1080p".to_string(),
        description: "Vimeo 1080p upload - High quality VP9".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(18),
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("slow".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("192k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "vimeo".to_string(),
            "1080p".to_string(),
            "high-quality".to_string(),
        ],
    }
}

/// Vimeo 720p preset.
fn vimeo_720p() -> Preset {
    Preset {
        name: "vimeo-720p".to_string(),
        description: "Vimeo 720p upload - High quality VP9".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(20),
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("slow".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("128k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "vimeo".to_string(),
            "720p".to_string(),
            "high-quality".to_string(),
        ],
    }
}

/// Twitter 1080p preset.
fn twitter_1080p() -> Preset {
    Preset {
        name: "twitter-1080p".to_string(),
        description: "Twitter 1080p video - Optimized for social media".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("6M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("10M".to_string()),
            min_bitrate: None,
            buffer_size: Some("12M".to_string()),
            keyframe_interval: Some(150),
            min_keyframe_interval: Some(15),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("128k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "twitter".to_string(),
            "1080p".to_string(),
            "social".to_string(),
        ],
    }
}

/// Twitter 720p preset.
fn twitter_720p() -> Preset {
    Preset {
        name: "twitter-720p".to_string(),
        description: "Twitter 720p video - Fast encoding for social media".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("3M".to_string()),
            crf: None,
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("fast".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("5M".to_string()),
            min_bitrate: None,
            buffer_size: Some("6M".to_string()),
            keyframe_interval: Some(150),
            min_keyframe_interval: Some(15),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("96k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "twitter".to_string(),
            "720p".to_string(),
            "social".to_string(),
        ],
    }
}

/// Facebook 1080p preset.
fn facebook_1080p() -> Preset {
    Preset {
        name: "facebook-1080p".to_string(),
        description: "Facebook 1080p video upload".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("8M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("12M".to_string()),
            min_bitrate: None,
            buffer_size: Some("16M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(12),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("128k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "facebook".to_string(),
            "1080p".to_string(),
            "social".to_string(),
        ],
    }
}

/// Facebook 720p preset.
fn facebook_720p() -> Preset {
    Preset {
        name: "facebook-720p".to_string(),
        description: "Facebook 720p video upload".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("4M".to_string()),
            crf: None,
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("6M".to_string()),
            min_bitrate: None,
            buffer_size: Some("8M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(12),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("96k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "facebook".to_string(),
            "720p".to_string(),
            "social".to_string(),
        ],
    }
}

/// Instagram feed post preset.
fn instagram_feed() -> Preset {
    Preset {
        name: "instagram-feed".to_string(),
        description: "Instagram feed post - Square or 4:5 aspect ratio".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("5M".to_string()),
            crf: None,
            width: Some(1080),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("8M".to_string()),
            min_bitrate: None,
            buffer_size: Some("10M".to_string()),
            keyframe_interval: Some(90),
            min_keyframe_interval: Some(9),
            aspect_ratio: Some("1:1".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("96k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "instagram".to_string(),
            "feed".to_string(),
            "square".to_string(),
        ],
    }
}

/// Instagram story preset.
fn instagram_story() -> Preset {
    Preset {
        name: "instagram-story".to_string(),
        description: "Instagram story - Vertical 9:16 aspect ratio".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("4M".to_string()),
            crf: None,
            width: Some(1080),
            height: Some(1920),
            fps: Some(30.0),
            preset: Some("fast".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("6M".to_string()),
            min_bitrate: None,
            buffer_size: Some("8M".to_string()),
            keyframe_interval: Some(90),
            min_keyframe_interval: Some(9),
            aspect_ratio: Some("9:16".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("96k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "instagram".to_string(),
            "story".to_string(),
            "vertical".to_string(),
        ],
    }
}

/// Instagram reels preset.
fn instagram_reels() -> Preset {
    Preset {
        name: "instagram-reels".to_string(),
        description: "Instagram reels - Vertical 9:16 optimized".to_string(),
        category: PresetCategory::Web,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("5M".to_string()),
            crf: None,
            width: Some(1080),
            height: Some(1920),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("8M".to_string()),
            min_bitrate: None,
            buffer_size: Some("10M".to_string()),
            keyframe_interval: Some(90),
            min_keyframe_interval: Some(9),
            aspect_ratio: Some("9:16".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("128k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "instagram".to_string(),
            "reels".to_string(),
            "vertical".to_string(),
        ],
    }
}
