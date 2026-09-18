//! AV1 film grain synthesis presets for archival and restoration workflows.
//!
//! Film grain synthesis (FGS) is a key AV1 feature that allows the encoder to
//! model and strip the grain pattern from the source, transmitting a compact
//! grain descriptor alongside the denoised image. The decoder then
//! re-synthesises grain at playback, saving significant bitrate while
//! preserving the filmic look.
//!
//! This module provides presets at three intensity levels:
//!
//! | Level  | Use-case                              | Grain strength |
//! |--------|---------------------------------------|----------------|
//! | Light  | Modern digital cinema, light texture  | 1-3            |
//! | Medium | 35mm film emulation, standard archive | 4-7            |
//! | Heavy  | Super 16mm / vintage 8mm restoration  | 8-12           |
//!
//! Each preset is available at 1080p and 4K resolutions, optimised for
//! archival quality (high CRF, two-pass).

use crate::{Preset, PresetCategory, PresetMetadata};
use oximedia_transcode::{PresetConfig, QualityMode};

/// Film grain intensity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainIntensity {
    /// Light grain — subtle texture, modern digital cinema look.
    Light,
    /// Medium grain — classic 35mm film emulation.
    Medium,
    /// Heavy grain — Super 16mm or vintage 8mm restoration.
    Heavy,
}

impl GrainIntensity {
    /// Descriptive label for the intensity level.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            GrainIntensity::Light => "Light",
            GrainIntensity::Medium => "Medium",
            GrainIntensity::Heavy => "Heavy",
        }
    }

    /// Suggested AV1 film-grain-denoise strength (encoder-dependent, 0-64 range).
    #[must_use]
    pub fn denoise_strength(&self) -> u32 {
        match self {
            GrainIntensity::Light => 8,
            GrainIntensity::Medium => 25,
            GrainIntensity::Heavy => 50,
        }
    }

    /// Suggested grain synthesis table strength (AV1 grain_scaling 0-3).
    #[must_use]
    pub fn grain_scaling(&self) -> u32 {
        match self {
            GrainIntensity::Light => 1,
            GrainIntensity::Medium => 2,
            GrainIntensity::Heavy => 3,
        }
    }
}

/// Film grain preset configuration metadata.
#[derive(Debug, Clone)]
pub struct FilmGrainPresetInfo {
    /// Grain intensity level.
    pub intensity: GrainIntensity,
    /// Resolution label (e.g., "1080p", "4K").
    pub resolution_label: String,
    /// Suggested denoise strength for the AV1 encoder.
    pub denoise_strength: u32,
    /// Grain scaling value.
    pub grain_scaling: u32,
    /// Estimated bitrate savings over non-FGS encode (percent, approximate).
    pub estimated_bitrate_saving_pct: f64,
}

/// Get all film grain synthesis presets.
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        film_grain_light_1080p(),
        film_grain_light_4k(),
        film_grain_medium_1080p(),
        film_grain_medium_4k(),
        film_grain_heavy_1080p(),
        film_grain_heavy_4k(),
    ]
}

/// Get all film grain preset infos (for metadata queries).
#[must_use]
pub fn all_preset_infos() -> Vec<FilmGrainPresetInfo> {
    vec![
        FilmGrainPresetInfo {
            intensity: GrainIntensity::Light,
            resolution_label: "1080p".to_string(),
            denoise_strength: GrainIntensity::Light.denoise_strength(),
            grain_scaling: GrainIntensity::Light.grain_scaling(),
            estimated_bitrate_saving_pct: 10.0,
        },
        FilmGrainPresetInfo {
            intensity: GrainIntensity::Light,
            resolution_label: "4K".to_string(),
            denoise_strength: GrainIntensity::Light.denoise_strength(),
            grain_scaling: GrainIntensity::Light.grain_scaling(),
            estimated_bitrate_saving_pct: 12.0,
        },
        FilmGrainPresetInfo {
            intensity: GrainIntensity::Medium,
            resolution_label: "1080p".to_string(),
            denoise_strength: GrainIntensity::Medium.denoise_strength(),
            grain_scaling: GrainIntensity::Medium.grain_scaling(),
            estimated_bitrate_saving_pct: 25.0,
        },
        FilmGrainPresetInfo {
            intensity: GrainIntensity::Medium,
            resolution_label: "4K".to_string(),
            denoise_strength: GrainIntensity::Medium.denoise_strength(),
            grain_scaling: GrainIntensity::Medium.grain_scaling(),
            estimated_bitrate_saving_pct: 28.0,
        },
        FilmGrainPresetInfo {
            intensity: GrainIntensity::Heavy,
            resolution_label: "1080p".to_string(),
            denoise_strength: GrainIntensity::Heavy.denoise_strength(),
            grain_scaling: GrainIntensity::Heavy.grain_scaling(),
            estimated_bitrate_saving_pct: 40.0,
        },
        FilmGrainPresetInfo {
            intensity: GrainIntensity::Heavy,
            resolution_label: "4K".to_string(),
            denoise_strength: GrainIntensity::Heavy.denoise_strength(),
            grain_scaling: GrainIntensity::Heavy.grain_scaling(),
            estimated_bitrate_saving_pct: 45.0,
        },
    ]
}

/// AV1 film grain light — 1080p archival.
#[must_use]
pub fn film_grain_light_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-film-grain-light-1080p",
        "AV1 Film Grain Light 1080p",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description(
        "AV1 with light film grain synthesis at 1080p for modern digital cinema archival",
    )
    .with_target("Archival / Restoration")
    .with_tag("av1")
    .with_tag("film-grain")
    .with_tag("light")
    .with_tag("1080p")
    .with_tag("archival");

    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(4_000_000), // Higher bitrate for archival
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// AV1 film grain light — 4K archival.
#[must_use]
pub fn film_grain_light_4k() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-film-grain-light-4k",
        "AV1 Film Grain Light 4K",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description(
        "AV1 with light film grain synthesis at 4K for modern digital cinema archival",
    )
    .with_target("Archival / Restoration")
    .with_tag("av1")
    .with_tag("film-grain")
    .with_tag("light")
    .with_tag("4k")
    .with_tag("archival");

    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(12_000_000),
        audio_bitrate: Some(256_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// AV1 film grain medium — 1080p (35mm emulation).
#[must_use]
pub fn film_grain_medium_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-film-grain-medium-1080p",
        "AV1 Film Grain Medium 1080p",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description("AV1 with medium film grain synthesis at 1080p for 35mm film emulation")
    .with_target("Archival / Restoration")
    .with_tag("av1")
    .with_tag("film-grain")
    .with_tag("medium")
    .with_tag("1080p")
    .with_tag("archival")
    .with_tag("35mm");

    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(3_500_000), // FGS saves bitrate vs light
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// AV1 film grain medium — 4K (35mm emulation).
#[must_use]
pub fn film_grain_medium_4k() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-film-grain-medium-4k",
        "AV1 Film Grain Medium 4K",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description("AV1 with medium film grain synthesis at 4K for 35mm film emulation")
    .with_target("Archival / Restoration")
    .with_tag("av1")
    .with_tag("film-grain")
    .with_tag("medium")
    .with_tag("4k")
    .with_tag("archival")
    .with_tag("35mm");

    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(10_000_000),
        audio_bitrate: Some(256_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// AV1 film grain heavy — 1080p (vintage / Super 16mm restoration).
#[must_use]
pub fn film_grain_heavy_1080p() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-film-grain-heavy-1080p",
        "AV1 Film Grain Heavy 1080p",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description(
        "AV1 with heavy film grain synthesis at 1080p for vintage/Super 16mm restoration",
    )
    .with_target("Archival / Restoration")
    .with_tag("av1")
    .with_tag("film-grain")
    .with_tag("heavy")
    .with_tag("1080p")
    .with_tag("archival")
    .with_tag("restoration")
    .with_tag("super16mm");

    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(2_500_000), // Heavy grain = most bitrate savings
        audio_bitrate: Some(192_000),
        width: Some(1920),
        height: Some(1080),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// AV1 film grain heavy — 4K (vintage / Super 16mm restoration).
#[must_use]
pub fn film_grain_heavy_4k() -> Preset {
    let metadata = PresetMetadata::new(
        "av1-film-grain-heavy-4k",
        "AV1 Film Grain Heavy 4K",
        PresetCategory::Codec("AV1".to_string()),
    )
    .with_description(
        "AV1 with heavy film grain synthesis at 4K for vintage/Super 16mm restoration",
    )
    .with_target("Archival / Restoration")
    .with_tag("av1")
    .with_tag("film-grain")
    .with_tag("heavy")
    .with_tag("4k")
    .with_tag("archival")
    .with_tag("restoration")
    .with_tag("super16mm");

    let config = PresetConfig {
        video_codec: Some("av1".to_string()),
        audio_codec: Some("opus".to_string()),
        video_bitrate: Some(8_000_000),
        audio_bitrate: Some(256_000),
        width: Some(3840),
        height: Some(2160),
        frame_rate: Some((24, 1)),
        quality_mode: Some(QualityMode::VeryHigh),
        container: Some("mkv".to_string()),
        audio_channel_layout: None,
    };

    Preset::new(metadata, config)
}

/// Select the appropriate film grain preset for a given intensity and resolution.
#[must_use]
pub fn select_preset(intensity: GrainIntensity, is_4k: bool) -> Preset {
    match (intensity, is_4k) {
        (GrainIntensity::Light, false) => film_grain_light_1080p(),
        (GrainIntensity::Light, true) => film_grain_light_4k(),
        (GrainIntensity::Medium, false) => film_grain_medium_1080p(),
        (GrainIntensity::Medium, true) => film_grain_medium_4k(),
        (GrainIntensity::Heavy, false) => film_grain_heavy_1080p(),
        (GrainIntensity::Heavy, true) => film_grain_heavy_4k(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_film_grain_presets_count() {
        assert_eq!(all_presets().len(), 6);
    }

    #[test]
    fn test_grain_intensity_labels() {
        assert_eq!(GrainIntensity::Light.label(), "Light");
        assert_eq!(GrainIntensity::Medium.label(), "Medium");
        assert_eq!(GrainIntensity::Heavy.label(), "Heavy");
    }

    #[test]
    fn test_grain_denoise_strength_ordering() {
        assert!(
            GrainIntensity::Light.denoise_strength() < GrainIntensity::Medium.denoise_strength()
        );
        assert!(
            GrainIntensity::Medium.denoise_strength() < GrainIntensity::Heavy.denoise_strength()
        );
    }

    #[test]
    fn test_grain_scaling_ordering() {
        assert!(GrainIntensity::Light.grain_scaling() < GrainIntensity::Medium.grain_scaling());
        assert!(GrainIntensity::Medium.grain_scaling() < GrainIntensity::Heavy.grain_scaling());
    }

    #[test]
    fn test_film_grain_light_1080p() {
        let p = film_grain_light_1080p();
        assert_eq!(p.metadata.id, "av1-film-grain-light-1080p");
        assert_eq!(p.config.video_codec.as_deref(), Some("av1"));
        assert_eq!(p.config.width, Some(1920));
        assert_eq!(p.config.height, Some(1080));
        assert!(p.has_tag("film-grain"));
        assert!(p.has_tag("light"));
    }

    #[test]
    fn test_film_grain_medium_4k() {
        let p = film_grain_medium_4k();
        assert_eq!(p.config.width, Some(3840));
        assert_eq!(p.config.height, Some(2160));
        assert!(p.has_tag("35mm"));
    }

    #[test]
    fn test_film_grain_heavy_1080p() {
        let p = film_grain_heavy_1080p();
        assert!(p.has_tag("restoration"));
        assert!(p.has_tag("super16mm"));
    }

    #[test]
    fn test_all_film_grain_use_av1() {
        for p in all_presets() {
            assert_eq!(
                p.config.video_codec.as_deref(),
                Some("av1"),
                "Film grain preset {} should use AV1",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_film_grain_use_mkv() {
        for p in all_presets() {
            assert_eq!(
                p.config.container.as_deref(),
                Some("mkv"),
                "Film grain preset {} should use MKV for archival",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_all_film_grain_24fps() {
        for p in all_presets() {
            assert_eq!(
                p.config.frame_rate,
                Some((24, 1)),
                "Film grain preset {} should use 24fps for cinema",
                p.metadata.id
            );
        }
    }

    #[test]
    fn test_select_preset() {
        let light_1080 = select_preset(GrainIntensity::Light, false);
        assert_eq!(light_1080.metadata.id, "av1-film-grain-light-1080p");

        let heavy_4k = select_preset(GrainIntensity::Heavy, true);
        assert_eq!(heavy_4k.metadata.id, "av1-film-grain-heavy-4k");
    }

    #[test]
    fn test_preset_infos_count() {
        assert_eq!(all_preset_infos().len(), 6);
    }

    #[test]
    fn test_bitrate_savings_increase_with_grain() {
        let infos = all_preset_infos();
        // Light 1080p saving < Medium 1080p saving < Heavy 1080p saving
        let light_1080 = &infos[0];
        let medium_1080 = &infos[2];
        let heavy_1080 = &infos[4];
        assert!(light_1080.estimated_bitrate_saving_pct < medium_1080.estimated_bitrate_saving_pct);
        assert!(medium_1080.estimated_bitrate_saving_pct < heavy_1080.estimated_bitrate_saving_pct);
    }

    #[test]
    fn test_all_film_grain_have_archival_tag() {
        for p in all_presets() {
            assert!(
                p.has_tag("archival"),
                "Film grain preset {} should have 'archival' tag",
                p.metadata.id
            );
        }
    }
}
