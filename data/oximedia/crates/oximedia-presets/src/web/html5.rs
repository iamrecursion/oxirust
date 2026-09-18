//! HTML5 video optimization presets.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Returns all HTML5 video presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![html5_sd(), html5_hd(), html5_uhd(), html5_webm_hd()]
}

/// Returns the HTML5 SD video preset.
#[must_use]
pub fn html5_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "html5-sd",
        "HTML5 SD",
        PresetCategory::Web("HTML5".to_string()),
    )
    .with_description("HTML5 SD video")
    .with_tag("html5")
    .with_tag("sd");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(1_500_000),
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

/// Returns the HTML5 HD video preset.
#[must_use]
pub fn html5_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "html5-hd",
        "HTML5 HD",
        PresetCategory::Web("HTML5".to_string()),
    )
    .with_description("HTML5 HD video")
    .with_tag("html5")
    .with_tag("hd");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(5_000_000),
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

/// Returns the HTML5 UHD (4K) video preset.
#[must_use]
pub fn html5_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "html5-uhd",
        "HTML5 UHD",
        PresetCategory::Web("HTML5".to_string()),
    )
    .with_description("HTML5 4K video")
    .with_tag("html5")
    .with_tag("uhd");
    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(15_000_000),
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

/// Returns the HTML5 WebM HD video preset using VP9/Opus.
#[must_use]
pub fn html5_webm_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "html5-webm-hd",
        "HTML5 WebM HD",
        PresetCategory::Web("HTML5".to_string()),
    )
    .with_description("HTML5 WebM HD video")
    .with_tag("html5")
    .with_tag("webm")
    .with_tag("hd");
    let config = PresetConfig {
        video_codec: Some("vp9".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(3_000_000),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_html5_presets_count() {
        assert_eq!(all_presets().len(), 4);
    }
}
