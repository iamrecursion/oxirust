//! Extended denoising configuration presets and strength enumerations.
//!
//! This module supplements the top-level [`DenoiseConfig`] with a richer
//! strength taxonomy and named presets for common workflows such as
//! broadcast delivery, archival mastering, and real-time streaming.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Semantic denoising strength levels.
///
/// These map to numeric strength values so callers can choose a named level
/// without picking an arbitrary float.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DenoiseStrength {
    /// Barely perceptible — only the finest grain is removed.
    Hairline,
    /// Light touch — preserves most texture, removes coarse noise.
    Light,
    /// Moderate reduction suitable for web delivery.
    Medium,
    /// Aggressive reduction for highly compressed sources.
    Strong,
    /// Maximum — intended for restoration of heavily degraded media.
    Maximum,
}

impl DenoiseStrength {
    /// Convert to the equivalent floating-point strength value (0.0–1.0).
    #[must_use]
    pub fn as_f32(self) -> f32 {
        match self {
            Self::Hairline => 0.15,
            Self::Light => 0.30,
            Self::Medium => 0.50,
            Self::Strong => 0.75,
            Self::Maximum => 0.95,
        }
    }

    /// Create from a float, selecting the nearest named level.
    #[must_use]
    pub fn from_f32(v: f32) -> Self {
        let v = v.clamp(0.0, 1.0);
        if v < 0.225 {
            Self::Hairline
        } else if v < 0.40 {
            Self::Light
        } else if v < 0.625 {
            Self::Medium
        } else if v < 0.85 {
            Self::Strong
        } else {
            Self::Maximum
        }
    }
}

impl std::fmt::Display for DenoiseStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Hairline => "hairline",
            Self::Light => "light",
            Self::Medium => "medium",
            Self::Strong => "strong",
            Self::Maximum => "maximum",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// DenoiseConfig (extended)
// ---------------------------------------------------------------------------

/// Extended denoising configuration with semantic controls.
///
/// Unlike the top-level `DenoiseConfig` which exposes raw floats, this struct
/// provides clearly named parameters for common use-cases.
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// Named strength level.
    pub strength: DenoiseStrength,
    /// Number of reference frames used for temporal filtering (1 = spatial only).
    pub temporal_radius: usize,
    /// Edge-preservation factor (0.0 = ignore edges, 1.0 = hard-preserve).
    pub edge_preservation: f32,
    /// Whether to retain the natural film-grain character of the source.
    pub preserve_grain: bool,
    /// Enable chroma (Cb/Cr) denoising in addition to luma.
    pub denoise_chroma: bool,
    /// Reduce blocking artefacts from prior lossy compression.
    pub deblock: bool,
    /// Override temporal radius for fast/real-time mode (disables temporal).
    pub realtime: bool,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            strength: DenoiseStrength::Medium,
            temporal_radius: 2,
            edge_preservation: 0.7,
            preserve_grain: false,
            denoise_chroma: true,
            deblock: false,
            realtime: false,
        }
    }
}

impl DenoiseConfig {
    /// Effective temporal window size derived from `temporal_radius`.
    ///
    /// Returns 1 when in real-time mode (spatial only).
    #[must_use]
    pub fn temporal_window(&self) -> usize {
        if self.realtime {
            1
        } else {
            self.temporal_radius * 2 + 1
        }
    }

    /// Validate that all parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.edge_preservation) {
            return Err(format!(
                "edge_preservation must be 0.0–1.0, got {}",
                self.edge_preservation
            ));
        }
        if self.temporal_radius > 8 {
            return Err(format!(
                "temporal_radius must be ≤ 8, got {}",
                self.temporal_radius
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConfigPreset
// ---------------------------------------------------------------------------

/// A named configuration preset for common denoising workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigPreset {
    /// Real-time streaming — minimal latency, spatial only.
    Streaming,
    /// Web delivery — balanced quality/performance.
    WebDelivery,
    /// Broadcast SD/HD — meets EBU R 118 / SMPTE guidelines.
    Broadcast,
    /// Archival mastering — best quality, slowest.
    Archival,
    /// Film restoration — grain-aware, strong spatial.
    FilmRestore,
    /// Surveillance / security footage — aggressive, fast.
    Surveillance,
}

impl ConfigPreset {
    /// Instantiate the full `DenoiseConfig` for this preset.
    #[must_use]
    pub fn defaults(self) -> DenoiseConfig {
        match self {
            Self::Streaming => DenoiseConfig {
                strength: DenoiseStrength::Light,
                temporal_radius: 0,
                edge_preservation: 0.9,
                preserve_grain: true,
                denoise_chroma: false,
                deblock: false,
                realtime: true,
            },
            Self::WebDelivery => DenoiseConfig {
                strength: DenoiseStrength::Medium,
                temporal_radius: 2,
                edge_preservation: 0.7,
                preserve_grain: false,
                denoise_chroma: true,
                deblock: true,
                realtime: false,
            },
            Self::Broadcast => DenoiseConfig {
                strength: DenoiseStrength::Light,
                temporal_radius: 3,
                edge_preservation: 0.8,
                preserve_grain: false,
                denoise_chroma: true,
                deblock: false,
                realtime: false,
            },
            Self::Archival => DenoiseConfig {
                strength: DenoiseStrength::Medium,
                temporal_radius: 5,
                edge_preservation: 0.95,
                preserve_grain: true,
                denoise_chroma: true,
                deblock: false,
                realtime: false,
            },
            Self::FilmRestore => DenoiseConfig {
                strength: DenoiseStrength::Strong,
                temporal_radius: 4,
                edge_preservation: 0.85,
                preserve_grain: true,
                denoise_chroma: true,
                deblock: true,
                realtime: false,
            },
            Self::Surveillance => DenoiseConfig {
                strength: DenoiseStrength::Strong,
                temporal_radius: 1,
                edge_preservation: 0.4,
                preserve_grain: false,
                denoise_chroma: false,
                deblock: false,
                realtime: true,
            },
        }
    }

    /// Human-readable description of the preset.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::Streaming => "Real-time spatial denoising for live streaming",
            Self::WebDelivery => "Balanced quality and speed for web video",
            Self::Broadcast => "Quality-first temporal denoising for broadcast",
            Self::Archival => "Grain-preserving archival mastering",
            Self::FilmRestore => "Film restoration with grain awareness",
            Self::Surveillance => "Aggressive fast denoising for surveillance footage",
        }
    }
}

impl std::fmt::Display for ConfigPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Streaming => "Streaming",
            Self::WebDelivery => "WebDelivery",
            Self::Broadcast => "Broadcast",
            Self::Archival => "Archival",
            Self::FilmRestore => "FilmRestore",
            Self::Surveillance => "Surveillance",
        };
        write!(f, "{name}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strength_as_f32() {
        assert!((DenoiseStrength::Hairline.as_f32() - 0.15).abs() < 1e-5);
        assert!((DenoiseStrength::Light.as_f32() - 0.30).abs() < 1e-5);
        assert!((DenoiseStrength::Medium.as_f32() - 0.50).abs() < 1e-5);
        assert!((DenoiseStrength::Strong.as_f32() - 0.75).abs() < 1e-5);
        assert!((DenoiseStrength::Maximum.as_f32() - 0.95).abs() < 1e-5);
    }

    #[test]
    fn test_strength_from_f32_boundaries() {
        assert_eq!(DenoiseStrength::from_f32(0.0), DenoiseStrength::Hairline);
        assert_eq!(DenoiseStrength::from_f32(1.0), DenoiseStrength::Maximum);
        assert_eq!(DenoiseStrength::from_f32(0.5), DenoiseStrength::Medium);
    }

    #[test]
    fn test_strength_from_f32_clamps() {
        assert_eq!(DenoiseStrength::from_f32(-1.0), DenoiseStrength::Hairline);
        assert_eq!(DenoiseStrength::from_f32(2.0), DenoiseStrength::Maximum);
    }

    #[test]
    fn test_strength_ordering() {
        assert!(DenoiseStrength::Hairline < DenoiseStrength::Light);
        assert!(DenoiseStrength::Light < DenoiseStrength::Medium);
        assert!(DenoiseStrength::Medium < DenoiseStrength::Strong);
        assert!(DenoiseStrength::Strong < DenoiseStrength::Maximum);
    }

    #[test]
    fn test_strength_display() {
        assert_eq!(DenoiseStrength::Medium.to_string(), "medium");
        assert_eq!(DenoiseStrength::Maximum.to_string(), "maximum");
    }

    #[test]
    fn test_config_default() {
        let cfg = DenoiseConfig::default();
        assert_eq!(cfg.strength, DenoiseStrength::Medium);
        assert_eq!(cfg.temporal_radius, 2);
        assert!(!cfg.realtime);
    }

    #[test]
    fn test_temporal_window_realtime() {
        let mut cfg = DenoiseConfig::default();
        cfg.realtime = true;
        assert_eq!(cfg.temporal_window(), 1);
    }

    #[test]
    fn test_temporal_window_normal() {
        let cfg = DenoiseConfig {
            temporal_radius: 3,
            ..Default::default()
        };
        assert_eq!(cfg.temporal_window(), 7);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = DenoiseConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_edge() {
        let cfg = DenoiseConfig {
            edge_preservation: 1.5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_radius() {
        let cfg = DenoiseConfig {
            temporal_radius: 9,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_preset_streaming() {
        let cfg = ConfigPreset::Streaming.defaults();
        assert!(cfg.realtime);
        assert_eq!(cfg.temporal_radius, 0);
        assert!(!cfg.denoise_chroma);
    }

    #[test]
    fn test_preset_archival() {
        let cfg = ConfigPreset::Archival.defaults();
        assert!(cfg.preserve_grain);
        assert!(cfg.temporal_radius >= 4);
    }

    #[test]
    fn test_preset_display() {
        assert_eq!(ConfigPreset::WebDelivery.to_string(), "WebDelivery");
        assert_eq!(ConfigPreset::FilmRestore.to_string(), "FilmRestore");
    }

    #[test]
    fn test_preset_description_nonempty() {
        for p in [
            ConfigPreset::Streaming,
            ConfigPreset::WebDelivery,
            ConfigPreset::Broadcast,
            ConfigPreset::Archival,
            ConfigPreset::FilmRestore,
            ConfigPreset::Surveillance,
        ] {
            assert!(!p.description().is_empty(), "empty description for {p}");
        }
    }

    #[test]
    fn test_all_presets_validate() {
        for p in [
            ConfigPreset::Streaming,
            ConfigPreset::WebDelivery,
            ConfigPreset::Broadcast,
            ConfigPreset::Archival,
            ConfigPreset::FilmRestore,
            ConfigPreset::Surveillance,
        ] {
            let cfg = p.defaults();
            assert!(cfg.validate().is_ok(), "preset {p} failed validation");
        }
    }
}
