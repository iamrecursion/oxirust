//! Streaming presets for adaptive bitrate streaming.
//!
//! Presets optimized for HLS/DASH adaptive streaming including:
//! - Multiple bitrate ladders (4K, 1080p, 720p, 480p, 360p, 240p)
//! - Low latency streaming
//! - Live streaming configurations

use super::{AudioConfig, Preset, PresetCategory, VideoConfig};

/// Get all streaming presets.
pub fn get_streaming_presets() -> Vec<Preset> {
    vec![
        // Standard ABR ladder
        hls_4k(),
        hls_1080p(),
        hls_720p(),
        hls_480p(),
        hls_360p(),
        hls_240p(),
        // DASH variants
        dash_4k(),
        dash_1080p(),
        dash_720p(),
        dash_480p(),
        // Low latency
        low_latency_1080p(),
        low_latency_720p(),
        // Live streaming
        live_1080p(),
        live_720p(),
    ]
}

/// HLS 4K variant (highest quality).
fn hls_4k() -> Preset {
    Preset {
        name: "hls-4k".to_string(),
        description: "HLS 4K variant - Adaptive streaming highest quality".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("25M".to_string()),
            crf: None,
            width: Some(3840),
            height: Some(2160),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("35M".to_string()),
            min_bitrate: Some("20M".to_string()),
            buffer_size: Some("50M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
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
        tags: vec!["hls".to_string(), "streaming".to_string(), "4k".to_string()],
    }
}

/// HLS 1080p variant.
fn hls_1080p() -> Preset {
    Preset {
        name: "hls-1080p".to_string(),
        description: "HLS 1080p variant - High quality adaptive streaming".to_string(),
        category: PresetCategory::Streaming,
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
            max_bitrate: Some("8M".to_string()),
            min_bitrate: Some("5M".to_string()),
            buffer_size: Some("12M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
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
            "hls".to_string(),
            "streaming".to_string(),
            "1080p".to_string(),
        ],
    }
}

/// HLS 720p variant.
fn hls_720p() -> Preset {
    Preset {
        name: "hls-720p".to_string(),
        description: "HLS 720p variant - Medium quality adaptive streaming".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("3M".to_string()),
            crf: None,
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("4M".to_string()),
            min_bitrate: Some("2.5M".to_string()),
            buffer_size: Some("6M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
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
            "hls".to_string(),
            "streaming".to_string(),
            "720p".to_string(),
        ],
    }
}

/// HLS 480p variant.
fn hls_480p() -> Preset {
    Preset {
        name: "hls-480p".to_string(),
        description: "HLS 480p variant - Standard definition streaming".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("1.5M".to_string()),
            crf: None,
            width: Some(854),
            height: Some(480),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("2M".to_string()),
            min_bitrate: Some("1M".to_string()),
            buffer_size: Some("3M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("64k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "hls".to_string(),
            "streaming".to_string(),
            "480p".to_string(),
        ],
    }
}

/// HLS 360p variant.
fn hls_360p() -> Preset {
    Preset {
        name: "hls-360p".to_string(),
        description: "HLS 360p variant - Low bandwidth streaming".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("800k".to_string()),
            crf: None,
            width: Some(640),
            height: Some(360),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("1M".to_string()),
            min_bitrate: Some("600k".to_string()),
            buffer_size: Some("1.5M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("48k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "hls".to_string(),
            "streaming".to_string(),
            "360p".to_string(),
            "low-bandwidth".to_string(),
        ],
    }
}

/// HLS 240p variant (minimum quality).
fn hls_240p() -> Preset {
    Preset {
        name: "hls-240p".to_string(),
        description: "HLS 240p variant - Minimum quality for very low bandwidth".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("400k".to_string()),
            crf: None,
            width: Some(426),
            height: Some(240),
            fps: Some(30.0),
            preset: Some("fast".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("500k".to_string()),
            min_bitrate: Some("300k".to_string()),
            buffer_size: Some("800k".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("32k".to_string()),
            sample_rate: Some(48000),
            channels: Some(1),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "hls".to_string(),
            "streaming".to_string(),
            "240p".to_string(),
            "low-bandwidth".to_string(),
        ],
    }
}

/// DASH 4K variant.
fn dash_4k() -> Preset {
    Preset {
        name: "dash-4k".to_string(),
        description: "DASH 4K variant - Highest quality DASH streaming".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("30M".to_string()),
            crf: None,
            width: Some(3840),
            height: Some(2160),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("40M".to_string()),
            min_bitrate: Some("25M".to_string()),
            buffer_size: Some("60M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
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
            "dash".to_string(),
            "streaming".to_string(),
            "4k".to_string(),
        ],
    }
}

/// DASH 1080p variant.
fn dash_1080p() -> Preset {
    Preset {
        name: "dash-1080p".to_string(),
        description: "DASH 1080p variant - High quality DASH streaming".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("7M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("10M".to_string()),
            min_bitrate: Some("6M".to_string()),
            buffer_size: Some("14M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
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
            "dash".to_string(),
            "streaming".to_string(),
            "1080p".to_string(),
        ],
    }
}

/// DASH 720p variant.
fn dash_720p() -> Preset {
    Preset {
        name: "dash-720p".to_string(),
        description: "DASH 720p variant - Medium quality DASH streaming".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("3.5M".to_string()),
            crf: None,
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("5M".to_string()),
            min_bitrate: Some("3M".to_string()),
            buffer_size: Some("7M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
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
            "dash".to_string(),
            "streaming".to_string(),
            "720p".to_string(),
        ],
    }
}

/// DASH 480p variant.
fn dash_480p() -> Preset {
    Preset {
        name: "dash-480p".to_string(),
        description: "DASH 480p variant - Standard definition DASH streaming".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("1.8M".to_string()),
            crf: None,
            width: Some(854),
            height: Some(480),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: Some("2.5M".to_string()),
            min_bitrate: Some("1.5M".to_string()),
            buffer_size: Some("3.6M".to_string()),
            keyframe_interval: Some(120),
            min_keyframe_interval: Some(120),
            aspect_ratio: Some("16:9".to_string()),
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("64k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "webm".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "dash".to_string(),
            "streaming".to_string(),
            "480p".to_string(),
        ],
    }
}

/// Low latency 1080p streaming.
fn low_latency_1080p() -> Preset {
    Preset {
        name: "low-latency-1080p".to_string(),
        description: "Low latency 1080p streaming - Optimized for real-time".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("5M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("veryfast".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("7M".to_string()),
            min_bitrate: Some("4M".to_string()),
            buffer_size: Some("2M".to_string()),
            keyframe_interval: Some(60),
            min_keyframe_interval: Some(60),
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
            "low-latency".to_string(),
            "streaming".to_string(),
            "1080p".to_string(),
            "real-time".to_string(),
        ],
    }
}

/// Low latency 720p streaming.
fn low_latency_720p() -> Preset {
    Preset {
        name: "low-latency-720p".to_string(),
        description: "Low latency 720p streaming - Fast encoding for real-time".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("2.5M".to_string()),
            crf: None,
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("veryfast".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("3.5M".to_string()),
            min_bitrate: Some("2M".to_string()),
            buffer_size: Some("1M".to_string()),
            keyframe_interval: Some(60),
            min_keyframe_interval: Some(60),
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
            "low-latency".to_string(),
            "streaming".to_string(),
            "720p".to_string(),
            "real-time".to_string(),
        ],
    }
}

/// Live streaming 1080p.
fn live_1080p() -> Preset {
    Preset {
        name: "live-1080p".to_string(),
        description: "Live streaming 1080p - Optimized for broadcast".to_string(),
        category: PresetCategory::Streaming,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: Some("6M".to_string()),
            crf: None,
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("fast".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: Some("9M".to_string()),
            min_bitrate: Some("5M".to_string()),
            buffer_size: Some("3M".to_string()),
            keyframe_interval: Some(60),
            min_keyframe_interval: Some(30),
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
            "live".to_string(),
            "streaming".to_string(),
            "1080p".to_string(),
            "broadcast".to_string(),
        ],
    }
}

/// Live streaming 720p.
fn live_720p() -> Preset {
    Preset {
        name: "live-720p".to_string(),
        description: "Live streaming 720p - Fast encoding for broadcast".to_string(),
        category: PresetCategory::Streaming,
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
            max_bitrate: Some("4.5M".to_string()),
            min_bitrate: Some("2.5M".to_string()),
            buffer_size: Some("1.5M".to_string()),
            keyframe_interval: Some(60),
            min_keyframe_interval: Some(30),
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
            "live".to_string(),
            "streaming".to_string(),
            "720p".to_string(),
            "broadcast".to_string(),
        ],
    }
}
