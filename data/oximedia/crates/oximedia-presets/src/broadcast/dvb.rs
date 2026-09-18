//! DVB (Digital Video Broadcasting) presets for European broadcast standards.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all DVB presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        dvb_t_sd(),
        dvb_t_hd(),
        dvb_t2_hd(),
        dvb_t2_uhd(),
        dvb_s2_hd(),
        dvb_s2_uhd(),
    ]
}

/// DVB-T SD (MPEG-2/MP2).
#[must_use]
pub fn dvb_t_sd() -> Preset {
    let metadata = PresetMetadata::new(
        "dvb-t-sd",
        "DVB-T SD",
        PresetCategory::Broadcast("DVB".to_string()),
    )
    .with_description("DVB-T standard definition terrestrial broadcast")
    .with_target("DVB-T")
    .with_tag("dvb")
    .with_tag("dvb-t")
    .with_tag("sd")
    .with_tag("terrestrial");

    let config = PresetConfig {
        video_codec: Some("mpeg2video".to_string()),
        audio_codec: Some("mp2".to_string()),
        video_bitrate: Some(3_500_000),
        audio_bitrate: Some(192_000),
        width: Some(720),
        height: Some(576),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::Medium),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DVB-T HD (H.264/MP2).
#[must_use]
pub fn dvb_t_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "dvb-t-hd",
        "DVB-T HD",
        PresetCategory::Broadcast("DVB".to_string()),
    )
    .with_description("DVB-T HD terrestrial broadcast")
    .with_target("DVB-T")
    .with_tag("dvb")
    .with_tag("dvb-t")
    .with_tag("hd")
    .with_tag("terrestrial");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("mp2".to_string()),
        video_bitrate: Some(8_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DVB-T2 HD (H.264/AAC).
#[must_use]
pub fn dvb_t2_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "dvb-t2-hd",
        "DVB-T2 HD",
        PresetCategory::Broadcast("DVB".to_string()),
    )
    .with_description("DVB-T2 HD terrestrial broadcast (second generation)")
    .with_target("DVB-T2")
    .with_tag("dvb")
    .with_tag("dvb-t2")
    .with_tag("hd")
    .with_tag("terrestrial");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(10_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DVB-T2 UHD (HEVC/AAC).
#[must_use]
pub fn dvb_t2_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "dvb-t2-uhd",
        "DVB-T2 UHD",
        PresetCategory::Broadcast("DVB".to_string()),
    )
    .with_description("DVB-T2 UHD 4K terrestrial broadcast")
    .with_target("DVB-T2")
    .with_tag("dvb")
    .with_tag("dvb-t2")
    .with_tag("4k")
    .with_tag("uhd")
    .with_tag("terrestrial");

    let config = PresetConfig {
        video_codec: Some("hevc".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(20_000_000),
        audio_bitrate: Some(256_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DVB-S2 HD (H.264/AAC) - Satellite broadcast.
#[must_use]
pub fn dvb_s2_hd() -> Preset {
    let metadata = PresetMetadata::new(
        "dvb-s2-hd",
        "DVB-S2 HD",
        PresetCategory::Broadcast("DVB".to_string()),
    )
    .with_description("DVB-S2 HD satellite broadcast")
    .with_target("DVB-S2")
    .with_tag("dvb")
    .with_tag("dvb-s2")
    .with_tag("hd")
    .with_tag("satellite");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(12_000_000),
        audio_bitrate: Some(256_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DVB-S2 UHD (HEVC/AAC) - Satellite broadcast.
#[must_use]
pub fn dvb_s2_uhd() -> Preset {
    let metadata = PresetMetadata::new(
        "dvb-s2-uhd",
        "DVB-S2 UHD",
        PresetCategory::Broadcast("DVB".to_string()),
    )
    .with_description("DVB-S2 UHD 4K satellite broadcast")
    .with_target("DVB-S2")
    .with_tag("dvb")
    .with_tag("dvb-s2")
    .with_tag("4k")
    .with_tag("uhd")
    .with_tag("satellite");

    let config = PresetConfig {
        video_codec: Some("hevc".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(25_000_000),
        audio_bitrate: Some(256_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mpegts".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dvb_presets_count() {
        assert_eq!(all_presets().len(), 6);
    }

    #[test]
    fn test_dvb_t2_uhd() {
        let preset = dvb_t2_uhd();
        assert_eq!(preset.config.video_codec, Some("hevc".to_string()));
        assert!(preset.has_tag("4k"));
    }
}
