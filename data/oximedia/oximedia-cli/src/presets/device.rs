//! Device-specific presets.
//!
//! Presets optimized for specific devices including:
//! - iPhone (various models)
//! - iPad
//! - Android phones and tablets
//! - Smart TVs (4K, 1080p)
//! - Gaming consoles

use super::{AudioConfig, Preset, PresetCategory, VideoConfig};

/// Get all device-specific presets.
pub fn get_device_presets() -> Vec<Preset> {
    vec![
        // iPhone presets
        iphone_15_pro(),
        iphone_standard(),
        iphone_legacy(),
        // iPad presets
        ipad_pro(),
        ipad_standard(),
        // Android presets
        android_flagship(),
        android_midrange(),
        android_tablet(),
        // TV presets
        tv_4k(),
        tv_1080p(),
        tv_720p(),
        // Console presets
        console_4k(),
        console_1080p(),
    ]
}

/// iPhone 15 Pro preset (latest flagship).
fn iphone_15_pro() -> Preset {
    Preset {
        name: "iphone-15-pro".to_string(),
        description: "iPhone 15 Pro - High quality for OLED display".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("12M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(60.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("16M".to_string()),
            min_bitrate: None,
            buffer_size: Some("20M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(12),
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
            "iphone".to_string(),
            "apple".to_string(),
            "mobile".to_string(),
        ],
    }
}

/// Standard iPhone preset (iPhone 12 and newer).
fn iphone_standard() -> Preset {
    Preset {
        name: "iphone-standard".to_string(),
        description: "Standard iPhone (iPhone 12+) - Balanced quality and file size".to_string(),
        category: PresetCategory::Device,
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
            "iphone".to_string(),
            "apple".to_string(),
            "mobile".to_string(),
        ],
    }
}

/// Legacy iPhone preset (iPhone 11 and older).
fn iphone_legacy() -> Preset {
    Preset {
        name: "iphone-legacy".to_string(),
        description: "Legacy iPhone (iPhone 11 and older) - Lower bitrate".to_string(),
        category: PresetCategory::Device,
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
            "iphone".to_string(),
            "apple".to_string(),
            "mobile".to_string(),
            "legacy".to_string(),
        ],
    }
}

/// iPad Pro preset.
fn ipad_pro() -> Preset {
    Preset {
        name: "ipad-pro".to_string(),
        description: "iPad Pro - High resolution for large display".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("16M".to_string()),
            crf: None,
            width: Some(2732),
            height: Some(2048),
            fps: Some(60.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("24M".to_string()),
            min_bitrate: None,
            buffer_size: Some("32M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(12),
            aspect_ratio: Some("4:3".to_string()),
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
            "ipad".to_string(),
            "apple".to_string(),
            "tablet".to_string(),
        ],
    }
}

/// Standard iPad preset.
fn ipad_standard() -> Preset {
    Preset {
        name: "ipad-standard".to_string(),
        description: "Standard iPad - Balanced quality".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("10M".to_string()),
            crf: None,
            width: Some(2048),
            height: Some(1536),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("15M".to_string()),
            min_bitrate: None,
            buffer_size: Some("20M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(12),
            aspect_ratio: Some("4:3".to_string()),
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
            "ipad".to_string(),
            "apple".to_string(),
            "tablet".to_string(),
        ],
    }
}

/// Android flagship phone preset.
fn android_flagship() -> Preset {
    Preset {
        name: "android-flagship".to_string(),
        description: "Android flagship phones (Samsung S24, Pixel 8, etc.)".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("10M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(60.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("15M".to_string()),
            min_bitrate: None,
            buffer_size: Some("20M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(12),
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
            "android".to_string(),
            "mobile".to_string(),
            "flagship".to_string(),
        ],
    }
}

/// Android mid-range phone preset.
fn android_midrange() -> Preset {
    Preset {
        name: "android-midrange".to_string(),
        description: "Android mid-range phones - Balanced performance".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("6M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("9M".to_string()),
            min_bitrate: None,
            buffer_size: Some("12M".to_string()),
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
            "android".to_string(),
            "mobile".to_string(),
            "midrange".to_string(),
        ],
    }
}

/// Android tablet preset.
fn android_tablet() -> Preset {
    Preset {
        name: "android-tablet".to_string(),
        description: "Android tablets - Large screen optimization".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("12M".to_string()),
            crf: None,
            width: Some(2560),
            height: Some(1600),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("18M".to_string()),
            min_bitrate: None,
            buffer_size: Some("24M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(12),
            aspect_ratio: Some("16:10".to_string()),
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
        tags: vec!["android".to_string(), "tablet".to_string()],
    }
}

/// 4K TV preset.
fn tv_4k() -> Preset {
    Preset {
        name: "tv-4k".to_string(),
        description: "4K Smart TV - Ultra HD quality".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("40M".to_string()),
            crf: None,
            width: Some(3840),
            height: Some(2160),
            fps: Some(60.0),
            preset: Some("slow".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("60M".to_string()),
            min_bitrate: None,
            buffer_size: Some("80M".to_string()),
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
        tags: vec!["tv".to_string(), "4k".to_string(), "uhd".to_string()],
    }
}

/// 1080p TV preset.
fn tv_1080p() -> Preset {
    Preset {
        name: "tv-1080p".to_string(),
        description: "1080p Smart TV - Full HD quality".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("15M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(60.0),
            preset: Some("slow".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("20M".to_string()),
            min_bitrate: None,
            buffer_size: Some("30M".to_string()),
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
        tags: vec!["tv".to_string(), "1080p".to_string(), "fhd".to_string()],
    }
}

/// 720p TV preset.
fn tv_720p() -> Preset {
    Preset {
        name: "tv-720p".to_string(),
        description: "720p TV - HD quality for older displays".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("8M".to_string()),
            crf: None,
            width: Some(1280),
            height: Some(720),
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
        tags: vec!["tv".to_string(), "720p".to_string(), "hd".to_string()],
    }
}

/// Gaming console 4K preset.
fn console_4k() -> Preset {
    Preset {
        name: "console-4k".to_string(),
        description: "Gaming console 4K (PS5, Xbox Series X)".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("50M".to_string()),
            crf: None,
            width: Some(3840),
            height: Some(2160),
            fps: Some(60.0),
            preset: Some("slow".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("75M".to_string()),
            min_bitrate: None,
            buffer_size: Some("100M".to_string()),
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
            "console".to_string(),
            "gaming".to_string(),
            "4k".to_string(),
        ],
    }
}

/// Gaming console 1080p preset.
fn console_1080p() -> Preset {
    Preset {
        name: "console-1080p".to_string(),
        description: "Gaming console 1080p (PS4, Xbox One)".to_string(),
        category: PresetCategory::Device,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("20M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(60.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("30M".to_string()),
            min_bitrate: None,
            buffer_size: Some("40M".to_string()),
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
            "console".to_string(),
            "gaming".to_string(),
            "1080p".to_string(),
        ],
    }
}
