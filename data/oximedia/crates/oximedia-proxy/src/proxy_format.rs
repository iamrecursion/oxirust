//! Format-oriented proxy configuration presets.
//!
//! Provides [`QualityPreset`], [`ProxyFormatConfig`], and [`FormatSelector`]
//! for choosing an encoding format and bitrate configuration that suits a
//! given production context (editing, review, archive, etc.).

#![allow(dead_code)]

/// High-level quality preset that bundles codec, resolution, and bitrate choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityPreset {
    /// Ultra-low bandwidth for remote / offline mobile editing.
    Mobile,
    /// Web-quality proxy suitable for browser-based review tools.
    Web,
    /// Editorial proxy — good enough for a Resolve / Premiere timeline.
    Editorial,
    /// Near-lossless proxy for critical colour grading preparation.
    ColorGrade,
    /// Archival intermediate — visually lossless (e.g. DNxHD / ProRes 4444).
    Archive,
}

impl QualityPreset {
    /// Return a short human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mobile => "Mobile",
            Self::Web => "Web",
            Self::Editorial => "Editorial",
            Self::ColorGrade => "Color Grade",
            Self::Archive => "Archive",
        }
    }

    /// Return the next higher preset, or `None` if already at [`Archive`](Self::Archive).
    pub fn upgrade(self) -> Option<Self> {
        match self {
            Self::Mobile => Some(Self::Web),
            Self::Web => Some(Self::Editorial),
            Self::Editorial => Some(Self::ColorGrade),
            Self::ColorGrade => Some(Self::Archive),
            Self::Archive => None,
        }
    }

    /// Return the next lower preset, or `None` if already at [`Mobile`](Self::Mobile).
    pub fn downgrade(self) -> Option<Self> {
        match self {
            Self::Mobile => None,
            Self::Web => Some(Self::Mobile),
            Self::Editorial => Some(Self::Web),
            Self::ColorGrade => Some(Self::Editorial),
            Self::Archive => Some(Self::ColorGrade),
        }
    }

    /// Return all presets in ascending quality order.
    pub fn all() -> &'static [QualityPreset] {
        &[
            Self::Mobile,
            Self::Web,
            Self::Editorial,
            Self::ColorGrade,
            Self::Archive,
        ]
    }
}

impl std::fmt::Display for QualityPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Concrete encoding parameters for a particular [`QualityPreset`].
#[derive(Debug, Clone)]
pub struct ProxyFormatConfig {
    /// The preset this config was derived from.
    pub preset: QualityPreset,
    /// Target video bitrate in kilobits per second.
    pub video_kbps: u32,
    /// Target audio bitrate in kilobits per second.
    pub audio_kbps: u32,
    /// Maximum horizontal resolution in pixels.
    pub max_width: u32,
    /// Maximum vertical resolution in pixels.
    pub max_height: u32,
    /// Container format hint (e.g. `"mp4"`, `"mov"`, `"mxf"`).
    pub container: &'static str,
    /// Video codec hint (e.g. `"h264"`, `"prores"`, `"dnxhd"`).
    pub codec: &'static str,
}

impl ProxyFormatConfig {
    /// Build a config with sensible defaults for `preset`.
    pub fn for_preset(preset: QualityPreset) -> Self {
        match preset {
            QualityPreset::Mobile => Self {
                preset,
                video_kbps: 400,
                audio_kbps: 64,
                max_width: 640,
                max_height: 360,
                container: "mp4",
                codec: "h264",
            },
            QualityPreset::Web => Self {
                preset,
                video_kbps: 1_500,
                audio_kbps: 128,
                max_width: 1_280,
                max_height: 720,
                container: "mp4",
                codec: "h264",
            },
            QualityPreset::Editorial => Self {
                preset,
                video_kbps: 8_000,
                audio_kbps: 192,
                max_width: 1_920,
                max_height: 1_080,
                container: "mp4",
                codec: "h264",
            },
            QualityPreset::ColorGrade => Self {
                preset,
                video_kbps: 45_000,
                audio_kbps: 320,
                max_width: 3_840,
                max_height: 2_160,
                container: "mov",
                codec: "prores",
            },
            QualityPreset::Archive => Self {
                preset,
                video_kbps: 185_000,
                audio_kbps: 320,
                max_width: 3_840,
                max_height: 2_160,
                container: "mxf",
                codec: "dnxhd",
            },
        }
    }

    /// Total bitrate in kbps (video + audio).
    pub fn total_kbps(&self) -> u32 {
        self.video_kbps + self.audio_kbps
    }

    /// Return `true` if this config fits within `budget_kbps`.
    pub fn fits_budget(&self, budget_kbps: u32) -> bool {
        self.total_kbps() <= budget_kbps
    }

    /// Return `true` if this config supports the given resolution.
    pub fn supports_resolution(&self, width: u32, height: u32) -> bool {
        width <= self.max_width && height <= self.max_height
    }
}

/// Selects the most suitable [`ProxyFormatConfig`] for a given constraint.
#[derive(Default)]
pub struct FormatSelector {
    configs: Vec<ProxyFormatConfig>,
}

impl FormatSelector {
    /// Create a selector populated with default configs for every preset.
    pub fn new() -> Self {
        let configs = QualityPreset::all()
            .iter()
            .map(|&p| ProxyFormatConfig::for_preset(p))
            .collect();
        Self { configs }
    }

    /// Create a selector with custom configs.
    pub fn with_configs(configs: Vec<ProxyFormatConfig>) -> Self {
        Self { configs }
    }

    /// Return the highest-quality preset that fits `budget_kbps`.
    ///
    /// Returns `None` if even the lowest-quality preset exceeds the budget.
    pub fn select_for_budget(&self, budget_kbps: u32) -> Option<&ProxyFormatConfig> {
        // Try from highest to lowest quality
        let mut sorted: Vec<&ProxyFormatConfig> = self.configs.iter().collect();
        sorted.sort_by(|a, b| b.video_kbps.cmp(&a.video_kbps));
        sorted.into_iter().find(|c| c.fits_budget(budget_kbps))
    }

    /// Return the config for `preset`, if present.
    pub fn get(&self, preset: QualityPreset) -> Option<&ProxyFormatConfig> {
        self.configs.iter().find(|c| c.preset == preset)
    }

    /// Return all configs sorted from lowest to highest video bitrate.
    pub fn all_ascending(&self) -> Vec<&ProxyFormatConfig> {
        let mut v: Vec<&ProxyFormatConfig> = self.configs.iter().collect();
        v.sort_by_key(|c| c.video_kbps);
        v
    }

    /// Number of configs in the selector.
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Return `true` if the selector contains no configs.
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_preset_name() {
        assert_eq!(QualityPreset::Mobile.name(), "Mobile");
        assert_eq!(QualityPreset::Archive.name(), "Archive");
    }

    #[test]
    fn quality_preset_display() {
        assert_eq!(format!("{}", QualityPreset::Editorial), "Editorial");
    }

    #[test]
    fn quality_preset_upgrade_chain() {
        assert_eq!(QualityPreset::Mobile.upgrade(), Some(QualityPreset::Web));
        assert_eq!(QualityPreset::Archive.upgrade(), None);
    }

    #[test]
    fn quality_preset_downgrade_chain() {
        assert_eq!(QualityPreset::Web.downgrade(), Some(QualityPreset::Mobile));
        assert_eq!(QualityPreset::Mobile.downgrade(), None);
    }

    #[test]
    fn quality_preset_all_count() {
        assert_eq!(QualityPreset::all().len(), 5);
    }

    #[test]
    fn proxy_format_config_total_kbps() {
        let cfg = ProxyFormatConfig::for_preset(QualityPreset::Mobile);
        assert_eq!(cfg.total_kbps(), 400 + 64);
    }

    #[test]
    fn proxy_format_config_fits_budget_true() {
        let cfg = ProxyFormatConfig::for_preset(QualityPreset::Mobile);
        assert!(cfg.fits_budget(1_000));
    }

    #[test]
    fn proxy_format_config_fits_budget_false() {
        let cfg = ProxyFormatConfig::for_preset(QualityPreset::Archive);
        assert!(!cfg.fits_budget(1_000));
    }

    #[test]
    fn proxy_format_config_supports_resolution_ok() {
        let cfg = ProxyFormatConfig::for_preset(QualityPreset::Editorial);
        assert!(cfg.supports_resolution(1_920, 1_080));
    }

    #[test]
    fn proxy_format_config_supports_resolution_fail() {
        let cfg = ProxyFormatConfig::for_preset(QualityPreset::Mobile);
        assert!(!cfg.supports_resolution(1_920, 1_080));
    }

    #[test]
    fn proxy_format_config_codec_preset_mobile() {
        let cfg = ProxyFormatConfig::for_preset(QualityPreset::Mobile);
        assert_eq!(cfg.codec, "h264");
        assert_eq!(cfg.container, "mp4");
    }

    #[test]
    fn proxy_format_config_codec_preset_archive() {
        let cfg = ProxyFormatConfig::for_preset(QualityPreset::Archive);
        assert_eq!(cfg.codec, "dnxhd");
        assert_eq!(cfg.container, "mxf");
    }

    #[test]
    fn format_selector_new_has_all_presets() {
        let sel = FormatSelector::new();
        assert_eq!(sel.len(), QualityPreset::all().len());
    }

    #[test]
    fn format_selector_get_by_preset() {
        let sel = FormatSelector::new();
        let cfg = sel.get(QualityPreset::Web);
        assert!(cfg.is_some());
        assert_eq!(
            cfg.expect("should succeed in test").preset,
            QualityPreset::Web
        );
    }

    #[test]
    fn format_selector_select_for_budget_returns_none_if_too_low() {
        let sel = FormatSelector::new();
        // budget of 1 kbps should match nothing
        assert!(sel.select_for_budget(1).is_none());
    }

    #[test]
    fn format_selector_select_for_budget_returns_mobile_for_small_budget() {
        let sel = FormatSelector::new();
        let cfg = sel.select_for_budget(500);
        assert!(cfg.is_some());
        assert_eq!(
            cfg.expect("should succeed in test").preset,
            QualityPreset::Mobile
        );
    }

    #[test]
    fn format_selector_all_ascending_ordered() {
        let sel = FormatSelector::new();
        let ascending = sel.all_ascending();
        for pair in ascending.windows(2) {
            assert!(pair[0].video_kbps <= pair[1].video_kbps);
        }
    }

    #[test]
    fn format_selector_is_empty_false() {
        let sel = FormatSelector::new();
        assert!(!sel.is_empty());
    }
}
