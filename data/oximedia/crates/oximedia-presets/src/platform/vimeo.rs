//! Vimeo encoding presets for professional quality video.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all Vimeo presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        vimeo_sd(),
        vimeo_720p(),
        vimeo_1080p(),
        vimeo_2k(),
        vimeo_4k(),
        vimeo_pro_720p(),
        vimeo_pro_1080p(),
        vimeo_pro_4k(),
    ]
}

/// Vimeo SD (H.264/AAC).
#[must_use]
pub fn vimeo_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-sd",
        "Vimeo SD",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("Standard definition for Vimeo")
    .with_target("Vimeo SD")
    .with_tag("vimeo")
    .with_tag("sd")
    .with_tag("h264");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(2_000_000),
        audio_bitrate: Some(128_000),
        width: Some(640),
        height: Some(480),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Vimeo 720p (H.264/AAC).
#[must_use]
pub fn vimeo_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-720p",
        "Vimeo 720p",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("HD quality for Vimeo")
    .with_target("Vimeo 720p")
    .with_tag("vimeo")
    .with_tag("720p")
    .with_tag("hd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
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

/// Vimeo 1080p (H.264/AAC).
#[must_use]
pub fn vimeo_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-1080p",
        "Vimeo 1080p",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("Full HD quality for Vimeo")
    .with_target("Vimeo 1080p")
    .with_tag("vimeo")
    .with_tag("1080p")
    .with_tag("full-hd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(10_000_000),
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

/// Vimeo 2K (H.264/AAC).
#[must_use]
pub fn vimeo_2k() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-2k",
        "Vimeo 2K",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("2K quality for Vimeo")
    .with_target("Vimeo 2K")
    .with_tag("vimeo")
    .with_tag("2k")
    .with_tag("1440p");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(20_000_000),
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

/// Vimeo 4K (H.264/AAC).
#[must_use]
pub fn vimeo_4k() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-4k",
        "Vimeo 4K",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("4K quality for Vimeo")
    .with_target("Vimeo 4K")
    .with_tag("vimeo")
    .with_tag("4k")
    .with_tag("uhd");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(40_000_000),
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

/// Vimeo Pro 720p (H.264/AAC) - Professional quality.
#[must_use]
pub fn vimeo_pro_720p() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-pro-720p",
        "Vimeo Pro 720p",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("Professional HD quality for Vimeo Pro")
    .with_target("Vimeo Pro 720p")
    .with_tag("vimeo")
    .with_tag("720p")
    .with_tag("pro")
    .with_tag("high-quality");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(8_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1280),
        height: Some(720),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Vimeo Pro 1080p (H.264/AAC) - Professional quality.
#[must_use]
pub fn vimeo_pro_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-pro-1080p",
        "Vimeo Pro 1080p",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("Professional Full HD quality for Vimeo Pro")
    .with_target("Vimeo Pro 1080p")
    .with_tag("vimeo")
    .with_tag("1080p")
    .with_tag("pro")
    .with_tag("high-quality");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(15_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Vimeo Pro 4K (H.264/AAC) - Professional quality.
#[must_use]
pub fn vimeo_pro_4k() -> Preset {
    let metadata = PresetMetadata::new(
        "vimeo-pro-4k",
        "Vimeo Pro 4K",
        PresetCategory::Platform("Vimeo".to_string()),
    )
    .with_description("Professional 4K quality for Vimeo Pro")
    .with_target("Vimeo Pro 4K")
    .with_tag("vimeo")
    .with_tag("4k")
    .with_tag("pro")
    .with_tag("high-quality");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(60_000_000),
        audio_bitrate: Some(320_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((30, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mp4".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vimeo_presets_count() {
        assert_eq!(all_presets().len(), 8);
    }

    #[test]
    fn test_vimeo_1080p() {
        let preset = vimeo_1080p();
        assert_eq!(preset.config.width, Some(1920));
        assert_eq!(preset.config.height, Some(1080));
    }
}
