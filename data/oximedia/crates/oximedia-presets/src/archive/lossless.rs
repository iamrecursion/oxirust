//! Lossless archive presets for long-term preservation.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all lossless archive presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        ffv1_sd(),
        ffv1_hd(),
        ffv1_uhd(),
        utvideo_hd(),
        utvideo_uhd(),
        lossless_h264_hd(),
    ]
}

/// FFV1 SD lossless (FFV1/FLAC).
#[must_use]
pub fn ffv1_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "ffv1-sd",
        "FFV1 SD Lossless",
        PresetCategory::Archive("Lossless".to_string()),
    )
    .with_description("FFV1 lossless SD for archival")
    .with_target("Archival Storage")
    .with_tag("ffv1")
    .with_tag("lossless")
    .with_tag("sd")
    .with_tag("archive");

    let config = PresetConfig {
        video_codec: Some("ffv1".to_string()),
        audio_codec: Some("flac".to_string()),
        video_bitrate: None, // Lossless
        audio_bitrate: None,
        width: Some(720),
        height: Some(576),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// FFV1 HD lossless (FFV1/FLAC).
#[must_use]
pub fn ffv1_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "ffv1-hd",
        "FFV1 HD Lossless",
        PresetCategory::Archive("Lossless".to_string()),
    )
    .with_description("FFV1 lossless HD for archival")
    .with_target("Archival Storage")
    .with_tag("ffv1")
    .with_tag("lossless")
    .with_tag("hd")
    .with_tag("archive");

    let config = PresetConfig {
        video_codec: Some("ffv1".to_string()),
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// FFV1 UHD lossless (FFV1/FLAC).
#[must_use]
pub fn ffv1_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "ffv1-uhd",
        "FFV1 UHD Lossless",
        PresetCategory::Archive("Lossless".to_string()),
    )
    .with_description("FFV1 lossless 4K for archival")
    .with_target("Archival Storage")
    .with_tag("ffv1")
    .with_tag("lossless")
    .with_tag("4k")
    .with_tag("uhd")
    .with_tag("archive");

    let config = PresetConfig {
        video_codec: Some("ffv1".to_string()),
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: None,
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// UT Video HD lossless (UT Video/FLAC).
#[must_use]
pub fn utvideo_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "utvideo-hd",
        "UT Video HD Lossless",
        PresetCategory::Archive("Lossless".to_string()),
    )
    .with_description("UT Video lossless HD for archival")
    .with_target("Archival Storage")
    .with_tag("utvideo")
    .with_tag("lossless")
    .with_tag("hd")
    .with_tag("archive");

    let config = PresetConfig {
        video_codec: Some("utvideo".to_string()),
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// UT Video UHD lossless (UT Video/FLAC).
#[must_use]
pub fn utvideo_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "utvideo-uhd",
        "UT Video UHD Lossless",
        PresetCategory::Archive("Lossless".to_string()),
    )
    .with_description("UT Video lossless 4K for archival")
    .with_target("Archival Storage")
    .with_tag("utvideo")
    .with_tag("lossless")
    .with_tag("4k")
    .with_tag("uhd")
    .with_tag("archive");

    let config = PresetConfig {
        video_codec: Some("utvideo".to_string()),
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: None,
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Lossless H.264 HD (H.264 lossless/FLAC).
#[must_use]
pub fn lossless_h264_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "lossless-h264-hd",
        "H.264 HD Lossless",
        PresetCategory::Archive("Lossless".to_string()),
    )
    .with_description("H.264 lossless HD for archival")
    .with_target("Archival Storage")
    .with_tag("h264")
    .with_tag("lossless")
    .with_tag("hd")
    .with_tag("archive");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("flac".to_string()),
        video_bitrate: None,
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossless_presets_count() {
        assert_eq!(all_presets().len(), 6);
    }

    #[test]
    fn test_ffv1_hd() {
        let preset = ffv1_hd();
        assert_eq!(preset.config.video_codec, Some("ffv1".to_string()));
        assert!(preset.has_tag("lossless"));
    }
}
