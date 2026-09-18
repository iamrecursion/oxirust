//! Mezzanine format presets for high-quality intermediate files.

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Get all mezzanine presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        prores_proxy(),
        prores_lt(),
        prores_standard(),
        prores_hq(),
        prores_4444(),
        dnxhd_proxy(),
        dnxhd_hq(),
        h264_mezzanine(),
    ]
}

/// ProRes Proxy for editing.
#[must_use]
pub fn prores_proxy() -> Preset {
    let metadata = PresetMetadata::new(
        "prores-proxy",
        "ProRes Proxy",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("ProRes Proxy for offline editing")
    .with_target("Editing")
    .with_tag("prores")
    .with_tag("proxy")
    .with_tag("mezzanine");

    let config = PresetConfig {
        video_codec: Some("prores".to_string()),
        audio_codec: Some("pcm_s16le".to_string()),
        video_bitrate: Some(45_000_000),
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mov".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ProRes LT for editing.
#[must_use]
pub fn prores_lt() -> Preset {
    let metadata = PresetMetadata::new(
        "prores-lt",
        "ProRes LT",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("ProRes LT for editing")
    .with_target("Editing")
    .with_tag("prores")
    .with_tag("lt")
    .with_tag("mezzanine");

    let config = PresetConfig {
        video_codec: Some("prores".to_string()),
        audio_codec: Some("pcm_s16le".to_string()),
        video_bitrate: Some(100_000_000),
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mov".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ProRes Standard for editing.
#[must_use]
pub fn prores_standard() -> Preset {
    let metadata = PresetMetadata::new(
        "prores-standard",
        "ProRes Standard",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("ProRes Standard 422 for editing")
    .with_target("Editing")
    .with_tag("prores")
    .with_tag("422")
    .with_tag("mezzanine");

    let config = PresetConfig {
        video_codec: Some("prores".to_string()),
        audio_codec: Some("pcm_s24le".to_string()),
        video_bitrate: Some(147_000_000),
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mov".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ProRes HQ for editing.
#[must_use]
pub fn prores_hq() -> Preset {
    let metadata = PresetMetadata::new(
        "prores-hq",
        "ProRes HQ",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("ProRes HQ 422 for high-quality editing")
    .with_target("Editing")
    .with_tag("prores")
    .with_tag("hq")
    .with_tag("422")
    .with_tag("mezzanine");

    let config = PresetConfig {
        video_codec: Some("prores".to_string()),
        audio_codec: Some("pcm_s24le".to_string()),
        video_bitrate: Some(220_000_000),
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mov".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// ProRes 4444 for editing with alpha.
#[must_use]
pub fn prores_4444() -> Preset {
    let metadata = PresetMetadata::new(
        "prores-4444",
        "ProRes 4444",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("ProRes 4444 for editing with alpha channel")
    .with_target("Editing")
    .with_tag("prores")
    .with_tag("4444")
    .with_tag("alpha")
    .with_tag("mezzanine");

    let config = PresetConfig {
        video_codec: Some("prores".to_string()),
        audio_codec: Some("pcm_s24le".to_string()),
        video_bitrate: Some(330_000_000),
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mov".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DNxHD Proxy for editing.
#[must_use]
pub fn dnxhd_proxy() -> Preset {
    let metadata = PresetMetadata::new(
        "dnxhd-proxy",
        "DNxHD Proxy",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("DNxHD proxy for Avid editing")
    .with_target("Editing")
    .with_tag("dnxhd")
    .with_tag("proxy")
    .with_tag("avid")
    .with_tag("mezzanine");

    let config = PresetConfig {
        video_codec: Some("dnxhd".to_string()),
        audio_codec: Some("pcm_s16le".to_string()),
        video_bitrate: Some(36_000_000),
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// DNxHD HQ for editing.
#[must_use]
pub fn dnxhd_hq() -> Preset {
    let metadata = PresetMetadata::new(
        "dnxhd-hq",
        "DNxHD HQ",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("DNxHD HQ for Avid editing")
    .with_target("Editing")
    .with_tag("dnxhd")
    .with_tag("hq")
    .with_tag("avid")
    .with_tag("mezzanine");

    let config = PresetConfig {
        video_codec: Some("dnxhd".to_string()),
        audio_codec: Some("pcm_s24le".to_string()),
        video_bitrate: Some(185_000_000),
        audio_bitrate: None,
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
        quality_mode: Some(QualityMode::High),
        container: Some("mxf".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// H.264 high-quality mezzanine.
#[must_use]
pub fn h264_mezzanine() -> Preset {
    let metadata = PresetMetadata::new(
        "h264-mezzanine",
        "H.264 Mezzanine",
        PresetCategory::Archive("Mezzanine".to_string()),
    )
    .with_description("High-quality H.264 mezzanine file")
    .with_target("Intermediate")
    .with_tag("h264")
    .with_tag("mezzanine")
    .with_tag("high-quality");

    let config = PresetConfig {
        video_codec: Some("h264".to_string()),
        audio_codec: Some("aac".to_string()),
        video_bitrate: Some(50_000_000),
        audio_bitrate: Some(320_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((25, 1)),
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
    fn test_mezzanine_presets_count() {
        assert_eq!(all_presets().len(), 8);
    }

    #[test]
    fn test_prores_hq() {
        let preset = prores_hq();
        assert_eq!(preset.config.video_codec, Some("prores".to_string()));
        assert!(preset.has_tag("mezzanine"));
    }
}
