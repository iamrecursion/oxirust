//! Built-in quality and archival presets.
//!
//! Generic quality tier presets and archival presets for long-term storage.

use super::{AudioConfig, Preset, PresetCategory, VideoConfig};

/// Get quality tier presets (4K, 1080p, 720p, 480p).
pub fn get_quality_presets() -> Vec<Preset> {
    vec![
        quality_4k(),
        quality_1080p(),
        quality_720p(),
        quality_480p(),
        quality_360p(),
    ]
}

/// Get archival presets for long-term storage.
pub fn get_archival_presets() -> Vec<Preset> {
    vec![
        archival_lossless(),
        archival_high_quality(),
        archival_balanced(),
        audio_flac(),
        audio_opus_high(),
    ]
}

/// Generic 4K quality preset.
fn quality_4k() -> Preset {
    Preset {
        name: "quality-4k".to_string(),
        description: "Generic 4K (2160p) quality - High bitrate VP9".to_string(),
        category: PresetCategory::Quality,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(24),
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
            bitrate: Some("192k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec!["quality".to_string(), "4k".to_string(), "uhd".to_string()],
    }
}

/// Generic 1080p quality preset.
fn quality_1080p() -> Preset {
    Preset {
        name: "quality-1080p".to_string(),
        description: "Generic 1080p (Full HD) quality - Balanced VP9".to_string(),
        category: PresetCategory::Quality,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(31),
            width: Some(1920),
            height: Some(1080),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
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
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "quality".to_string(),
            "1080p".to_string(),
            "fhd".to_string(),
        ],
    }
}

/// Generic 720p quality preset.
fn quality_720p() -> Preset {
    Preset {
        name: "quality-720p".to_string(),
        description: "Generic 720p (HD) quality - Efficient VP9".to_string(),
        category: PresetCategory::Quality,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(32),
            width: Some(1280),
            height: Some(720),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
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
            bitrate: Some("96k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec!["quality".to_string(), "720p".to_string(), "hd".to_string()],
    }
}

/// Generic 480p quality preset.
fn quality_480p() -> Preset {
    Preset {
        name: "quality-480p".to_string(),
        description: "Generic 480p (SD) quality - Small file size VP9".to_string(),
        category: PresetCategory::Quality,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(35),
            width: Some(854),
            height: Some(480),
            fps: Some(30.0),
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
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
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec!["quality".to_string(), "480p".to_string(), "sd".to_string()],
    }
}

/// Generic 360p quality preset.
fn quality_360p() -> Preset {
    Preset {
        name: "quality-360p".to_string(),
        description: "Generic 360p quality - Minimal file size VP9".to_string(),
        category: PresetCategory::Quality,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(40),
            width: Some(640),
            height: Some(360),
            fps: Some(30.0),
            preset: Some("fast".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
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
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "quality".to_string(),
            "360p".to_string(),
            "low-res".to_string(),
        ],
    }
}

/// Lossless archival preset.
fn archival_lossless() -> Preset {
    Preset {
        name: "archival-lossless".to_string(),
        description: "Lossless archival - AV1 lossless + FLAC audio".to_string(),
        category: PresetCategory::Archival,
        video: VideoConfig {
            codec: "av1".to_string(),
            bitrate: None,
            crf: Some(0),
            width: None,
            height: None,
            fps: None,
            preset: Some("veryslow".to_string()),
            pixel_format: Some("yuv444p".to_string()),
            two_pass: false,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: None,
        },
        audio: AudioConfig {
            codec: "flac".to_string(),
            bitrate: None,
            sample_rate: Some(48000),
            channels: None,
            quality: None,
            compression_level: Some(8),
        },
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "archival".to_string(),
            "lossless".to_string(),
            "preservation".to_string(),
        ],
    }
}

/// High quality archival preset.
fn archival_high_quality() -> Preset {
    Preset {
        name: "archival-high-quality".to_string(),
        description: "High quality archival - AV1 CRF 15 + FLAC audio".to_string(),
        category: PresetCategory::Archival,
        video: VideoConfig {
            codec: "av1".to_string(),
            bitrate: None,
            crf: Some(15),
            width: None,
            height: None,
            fps: None,
            preset: Some("veryslow".to_string()),
            pixel_format: Some("yuv420p10le".to_string()),
            two_pass: true,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: None,
        },
        audio: AudioConfig {
            codec: "flac".to_string(),
            bitrate: None,
            sample_rate: Some(48000),
            channels: None,
            quality: None,
            compression_level: Some(8),
        },
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "archival".to_string(),
            "high-quality".to_string(),
            "preservation".to_string(),
        ],
    }
}

/// Balanced archival preset.
fn archival_balanced() -> Preset {
    Preset {
        name: "archival-balanced".to_string(),
        description: "Balanced archival - AV1 CRF 25 + Opus audio".to_string(),
        category: PresetCategory::Archival,
        video: VideoConfig {
            codec: "av1".to_string(),
            bitrate: None,
            crf: Some(25),
            width: None,
            height: None,
            fps: None,
            preset: Some("slow".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: true,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: None,
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("192k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "mkv".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "archival".to_string(),
            "balanced".to_string(),
            "preservation".to_string(),
        ],
    }
}

/// Audio-only FLAC preset.
fn audio_flac() -> Preset {
    Preset {
        name: "audio-flac".to_string(),
        description: "Audio-only FLAC - Lossless audio archival".to_string(),
        category: PresetCategory::Archival,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(31),
            width: None,
            height: None,
            fps: None,
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: None,
            min_keyframe_interval: None,
            aspect_ratio: None,
        },
        audio: AudioConfig {
            codec: "flac".to_string(),
            bitrate: None,
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: Some(8),
        },
        container: "flac".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "audio".to_string(),
            "flac".to_string(),
            "lossless".to_string(),
        ],
    }
}

/// Audio-only high quality Opus preset.
fn audio_opus_high() -> Preset {
    Preset {
        name: "audio-opus-high".to_string(),
        description: "Audio-only Opus - High quality lossy audio".to_string(),
        category: PresetCategory::Archival,
        video: VideoConfig {
            codec: "vp9".to_string(),
            bitrate: None,
            crf: Some(31),
            width: None,
            height: None,
            fps: None,
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: None,
            min_keyframe_interval: None,
            aspect_ratio: None,
        },
        audio: AudioConfig {
            codec: "opus".to_string(),
            bitrate: Some("256k".to_string()),
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container: "ogg".to_string(),
        filters: None,
        builtin: true,
        tags: vec![
            "audio".to_string(),
            "opus".to_string(),
            "high-quality".to_string(),
        ],
    }
}
