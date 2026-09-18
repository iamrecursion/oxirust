//! YouTube encoding presets following official recommendations.
//!
//! Supports multiple quality tiers from 360p to 8K, with both H.264 and VP9 options.
//! Includes HDR and high frame rate support.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all YouTube presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        youtube_360p(),
        youtube_480p(),
        youtube_720p(),
        youtube_720p_60fps(),
        youtube_1080p(),
        youtube_1080p_60fps(),
        youtube_1440p(),
        youtube_1440p_60fps(),
        youtube_2160p(),
        youtube_2160p_60fps(),
        youtube_4320p(),
        youtube_vp9_360p(),
        youtube_vp9_480p(),
        youtube_vp9_720p(),
        youtube_vp9_720p_60fps(),
        youtube_vp9_1080p(),
        youtube_vp9_1080p_60fps(),
        youtube_vp9_1440p(),
        youtube_vp9_1440p_60fps(),
        youtube_vp9_2160p(),
        youtube_vp9_2160p_60fps(),
        youtube_vp9_4320p(),
        youtube_hdr_1080p(),
        youtube_hdr_1440p(),
        youtube_hdr_2160p(),
    ]
}

/// YouTube 360p (H.264/AAC) - Low quality for slow connections.
#[must_use]
pub fn youtube_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-360p",
        "YouTube 360p (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Low quality preset for slow connections")
    .with_target("YouTube 360p")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("360p")
    .with_tag("low-quality");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 480p (H.264/AAC) - Standard definition.
#[must_use]
pub fn youtube_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-480p",
        "YouTube 480p (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Standard definition preset")
    .with_target("YouTube 480p")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("480p")
    .with_tag("sd");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 720p (H.264/AAC) - HD quality.
#[must_use]
pub fn youtube_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-720p",
        "YouTube 720p (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("HD quality preset (recommended minimum)")
    .with_target("YouTube 720p")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("720p")
    .with_tag("hd");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 720p 60fps (H.264/AAC) - HD high framerate.
#[must_use]
pub fn youtube_720p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-720p-60fps",
        "YouTube 720p 60fps (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("HD high framerate preset for smooth motion")
    .with_target("YouTube 720p60")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("720p")
    .with_tag("60fps")
    .with_tag("hd");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 1080p (H.264/AAC) - Full HD quality.
#[must_use]
pub fn youtube_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-1080p",
        "YouTube 1080p (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Full HD quality preset (most popular)")
    .with_target("YouTube 1080p")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("1080p")
    .with_tag("full-hd");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 1080p 60fps (H.264/AAC) - Full HD high framerate.
#[must_use]
pub fn youtube_1080p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-1080p-60fps",
        "YouTube 1080p 60fps (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Full HD high framerate for smooth gaming/sports content")
    .with_target("YouTube 1080p60")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("1080p")
    .with_tag("60fps")
    .with_tag("full-hd");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 1440p (H.264/AAC) - 2K quality.
#[must_use]
pub fn youtube_1440p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-1440p",
        "YouTube 1440p (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("2K/QHD quality preset")
    .with_target("YouTube 1440p")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("1440p")
    .with_tag("2k")
    .with_tag("qhd");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 1440p 60fps (H.264/AAC) - 2K high framerate.
#[must_use]
pub fn youtube_1440p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-1440p-60fps",
        "YouTube 1440p 60fps (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("2K/QHD high framerate preset")
    .with_target("YouTube 1440p60")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("1440p")
    .with_tag("60fps")
    .with_tag("2k")
    .with_tag("qhd");

    let config = PresetConfig {
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
    };

    Preset::new(metadata, config)
}

/// YouTube 2160p/4K (H.264/AAC) - 4K quality.
#[must_use]
pub fn youtube_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-2160p",
        "YouTube 2160p/4K (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("4K/UHD quality preset")
    .with_target("YouTube 2160p")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("2160p")
    .with_tag("4k")
    .with_tag("uhd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(35_000_000), // 35 Mbps
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 2160p/4K 60fps (H.264/AAC) - 4K high framerate.
#[must_use]
pub fn youtube_2160p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-2160p-60fps",
        "YouTube 2160p/4K 60fps (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("4K/UHD high framerate preset")
    .with_target("YouTube 2160p60")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("2160p")
    .with_tag("60fps")
    .with_tag("4k")
    .with_tag("uhd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(53_000_000), // 53 Mbps
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 4320p/8K (H.264/AAC) - 8K quality.
#[must_use]
pub fn youtube_4320p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-4320p",
        "YouTube 4320p/8K (H.264)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("8K quality preset for future-proof content")
    .with_target("YouTube 4320p")
    .with_tag("youtube")
    .with_tag("h264")
    .with_tag("4320p")
    .with_tag("8k");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(100_000_000), // 100 Mbps
        audio_bitrate: Some(256_000),
        width: Some(7680),
        height: Some(4320),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

// VP9 variants for better compression

/// YouTube 360p (VP9/Opus) - Low quality with modern codecs.
#[must_use]
pub fn youtube_vp9_360p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-360p",
        "YouTube 360p (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Low quality with modern VP9 codec")
    .with_target("YouTube 360p")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("360p")
    .with_tag("low-quality");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(500_000), // 500 kbps
        audio_bitrate: Some(96_000),
        width: Some(640),
        height: Some(360),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 480p (VP9/Opus) - SD with modern codecs.
#[must_use]
pub fn youtube_vp9_480p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-480p",
        "YouTube 480p (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Standard definition with modern VP9 codec")
    .with_target("YouTube 480p")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("480p")
    .with_tag("sd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(1_500_000), // 1.5 Mbps
        audio_bitrate: Some(96_000),
        width: Some(854),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 720p (VP9/Opus) - HD with modern codecs.
#[must_use]
pub fn youtube_vp9_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-720p",
        "YouTube 720p (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("HD quality with modern VP9 codec")
    .with_target("YouTube 720p")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("720p")
    .with_tag("hd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(2_500_000), // 2.5 Mbps
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 720p 60fps (VP9/Opus) - HD high framerate with modern codecs.
#[must_use]
pub fn youtube_vp9_720p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-720p-60fps",
        "YouTube 720p 60fps (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("HD high framerate with modern VP9 codec")
    .with_target("YouTube 720p60")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("720p")
    .with_tag("60fps")
    .with_tag("hd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(4_000_000), // 4 Mbps
        audio_bitrate: Some(128_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 1080p (VP9/Opus) - Full HD with modern codecs.
#[must_use]
pub fn youtube_vp9_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-1080p",
        "YouTube 1080p (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Full HD with modern VP9 codec")
    .with_target("YouTube 1080p")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("1080p")
    .with_tag("full-hd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(4_500_000), // 4.5 Mbps
        audio_bitrate: Some(128_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 1080p 60fps (VP9/Opus) - Full HD high framerate with modern codecs.
#[must_use]
pub fn youtube_vp9_1080p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-1080p-60fps",
        "YouTube 1080p 60fps (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Full HD high framerate with modern VP9 codec")
    .with_target("YouTube 1080p60")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("1080p")
    .with_tag("60fps")
    .with_tag("full-hd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(7_000_000), // 7 Mbps
        audio_bitrate: Some(128_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 1440p (VP9/Opus) - 2K with modern codecs.
#[must_use]
pub fn youtube_vp9_1440p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-1440p",
        "YouTube 1440p (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("2K/QHD with modern VP9 codec")
    .with_target("YouTube 1440p")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("1440p")
    .with_tag("2k")
    .with_tag("qhd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(9_000_000), // 9 Mbps
        audio_bitrate: Some(128_000),
        width: Some(2560),
        height: Some(1440),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 1440p 60fps (VP9/Opus) - 2K high framerate with modern codecs.
#[must_use]
pub fn youtube_vp9_1440p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-1440p-60fps",
        "YouTube 1440p 60fps (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("2K/QHD high framerate with modern VP9 codec")
    .with_target("YouTube 1440p60")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("1440p")
    .with_tag("60fps")
    .with_tag("2k")
    .with_tag("qhd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(13_000_000), // 13 Mbps
        audio_bitrate: Some(128_000),
        width: Some(2560),
        height: Some(1440),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 2160p/4K (VP9/Opus) - 4K with modern codecs.
#[must_use]
pub fn youtube_vp9_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-2160p",
        "YouTube 2160p/4K (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("4K/UHD with modern VP9 codec")
    .with_target("YouTube 2160p")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("2160p")
    .with_tag("4k")
    .with_tag("uhd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(18_000_000), // 18 Mbps
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 2160p/4K 60fps (VP9/Opus) - 4K high framerate with modern codecs.
#[must_use]
pub fn youtube_vp9_2160p_60fps() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-2160p-60fps",
        "YouTube 2160p/4K 60fps (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("4K/UHD high framerate with modern VP9 codec")
    .with_target("YouTube 2160p60")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("2160p")
    .with_tag("60fps")
    .with_tag("4k")
    .with_tag("uhd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(25_000_000), // 25 Mbps
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((60, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 4320p/8K (VP9/Opus) - 8K with modern codecs.
#[must_use]
pub fn youtube_vp9_4320p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-vp9-4320p",
        "YouTube 4320p/8K (VP9)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("8K quality with modern VP9 codec")
    .with_target("YouTube 4320p")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("4320p")
    .with_tag("8k");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(50_000_000), // 50 Mbps
        audio_bitrate: Some(256_000),
        width: Some(7680),
        height: Some(4320),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

// HDR variants

/// YouTube 1080p HDR (VP9.2/Opus) - Full HD with HDR.
#[must_use]
pub fn youtube_hdr_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-hdr-1080p",
        "YouTube 1080p HDR (VP9.2)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("Full HD with HDR10 support")
    .with_target("YouTube 1080p HDR")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("hdr")
    .with_tag("1080p")
    .with_tag("full-hd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(10_000_000), // 10 Mbps
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 1440p HDR (VP9.2/Opus) - 2K with HDR.
#[must_use]
pub fn youtube_hdr_1440p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-hdr-1440p",
        "YouTube 1440p HDR (VP9.2)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("2K/QHD with HDR10 support")
    .with_target("YouTube 1440p HDR")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("hdr")
    .with_tag("1440p")
    .with_tag("2k")
    .with_tag("qhd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(15_000_000), // 15 Mbps
        audio_bitrate: Some(192_000),
        width: Some(2560),
        height: Some(1440),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// YouTube 2160p/4K HDR (VP9.2/Opus) - 4K with HDR.
#[must_use]
pub fn youtube_hdr_2160p() -> Preset {
    let metadata = PresetMetadata::new(
        "youtube-hdr-2160p",
        "YouTube 2160p/4K HDR (VP9.2)",
        PresetCategory::Platform("YouTube".to_string()),
    )
    .with_description("4K/UHD with HDR10 support")
    .with_target("YouTube 2160p HDR")
    .with_tag("youtube")
    .with_tag("vp9")
    .with_tag("hdr")
    .with_tag("2160p")
    .with_tag("4k")
    .with_tag("uhd");

    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(25_000_000), // 25 Mbps
        audio_bitrate: Some(192_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("webm".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_youtube_presets_count() {
        let presets = all_presets();
        assert_eq!(presets.len(), 25);
    }

    #[test]
    fn test_youtube_1080p_config() {
        let preset = youtube_1080p();
        assert_eq!(preset.metadata.id, "youtube-1080p");
        assert_eq!(preset.config.width, Some(1920));
        assert_eq!(preset.config.height, Some(1080));
        assert!(preset.has_tag("youtube"));
        assert!(preset.has_tag("1080p"));
    }

    #[test]
    fn test_youtube_vp9_config() {
        let preset = youtube_vp9_1080p();
        assert_eq!(preset.config.video_codec, Some("vp9".to_string()));
        assert_eq!(preset.config.audio_codec, Some("opus".to_string()));
        assert_eq!(preset.config.container, Some("webm".to_string()));
    }

    #[test]
    fn test_youtube_hdr_config() {
        let preset = youtube_hdr_2160p();
        assert!(preset.has_tag("hdr"));
        assert!(preset.has_tag("4k"));
    }
}
