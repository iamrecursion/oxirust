//! Color-space preset management for encoding pipelines.
#![allow(dead_code)]

/// Standardised colour-space / transfer-function pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpacePreset {
    /// ITU-R BT.709 – standard HD colour space.
    Rec709,
    /// ITU-R BT.2020 – wide colour gamut for HDR/UHD.
    Rec2020,
    /// DCI-P3 with D65 white point – cinema & display HDR.
    P3D65,
    /// ACES Colour Corrected – wide dynamic range production space.
    AcesCc,
    /// sRGB – common web / display colour space.
    Srgb,
    /// HDR10 / SMPTE ST 2084 (PQ) colour space.
    Hdr10,
    /// Hybrid Log-Gamma broadcast HDR.
    Hlg,
}

impl ColorSpacePreset {
    /// Nominal bit-depth associated with this colour space.
    #[must_use]
    pub fn bit_depth(&self) -> u8 {
        match self {
            ColorSpacePreset::Rec709 => 8,
            ColorSpacePreset::Srgb => 8,
            ColorSpacePreset::Rec2020 => 10,
            ColorSpacePreset::P3D65 => 10,
            ColorSpacePreset::Hdr10 => 10,
            ColorSpacePreset::Hlg => 10,
            ColorSpacePreset::AcesCc => 16,
        }
    }

    /// Returns `true` when this colour space is intended for HDR content.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        matches!(
            self,
            ColorSpacePreset::Rec2020
                | ColorSpacePreset::P3D65
                | ColorSpacePreset::AcesCc
                | ColorSpacePreset::Hdr10
                | ColorSpacePreset::Hlg
        )
    }

    /// Short string tag usable in file names or codec parameters.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            ColorSpacePreset::Rec709 => "rec709",
            ColorSpacePreset::Rec2020 => "rec2020",
            ColorSpacePreset::P3D65 => "p3d65",
            ColorSpacePreset::AcesCc => "acescc",
            ColorSpacePreset::Srgb => "srgb",
            ColorSpacePreset::Hdr10 => "hdr10",
            ColorSpacePreset::Hlg => "hlg",
        }
    }
}

/// A named colour preset coupling a colour-space with a transfer function label.
#[derive(Debug, Clone)]
pub struct ColorPreset {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Colour-space / transfer pair.
    pub color_space: ColorSpacePreset,
    /// Transfer function label (e.g. `"gamma22"`, `"pq"`, `"hlg"`).
    pub transfer_function: String,
    /// Colour primaries label (e.g. `"bt709"`, `"p3"`, `"bt2020"`).
    pub primaries: String,
    /// Whether tone-mapping is applied during output.
    pub tone_map: bool,
}

impl ColorPreset {
    /// Create a new colour preset.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        color_space: ColorSpacePreset,
    ) -> Self {
        let space = color_space;
        let transfer = match space {
            ColorSpacePreset::Rec709 | ColorSpacePreset::Srgb => "gamma22",
            ColorSpacePreset::Rec2020 | ColorSpacePreset::Hdr10 => "pq",
            ColorSpacePreset::P3D65 => "pq",
            ColorSpacePreset::AcesCc => "acescc",
            ColorSpacePreset::Hlg => "hlg",
        };
        let primaries = match space {
            ColorSpacePreset::Rec709 | ColorSpacePreset::Srgb => "bt709",
            ColorSpacePreset::Rec2020 | ColorSpacePreset::Hdr10 | ColorSpacePreset::Hlg => "bt2020",
            ColorSpacePreset::P3D65 => "p3",
            ColorSpacePreset::AcesCc => "aces",
        };
        Self {
            id: id.into(),
            name: name.into(),
            color_space,
            transfer_function: transfer.to_string(),
            primaries: primaries.to_string(),
            tone_map: false,
        }
    }

    /// Returns `true` when the preset targets HDR colour.
    #[must_use]
    pub fn is_hdr_color(&self) -> bool {
        self.color_space.is_hdr()
    }

    /// Bit depth required by this preset's colour space.
    #[must_use]
    pub fn bit_depth(&self) -> u8 {
        self.color_space.bit_depth()
    }
}

/// Manages a collection of colour presets.
#[derive(Debug, Default)]
pub struct ColorPresetManager {
    presets: Vec<ColorPreset>,
}

impl ColorPresetManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a preset.
    pub fn add(&mut self, preset: ColorPreset) {
        self.presets.push(preset);
    }

    /// Find a preset by ID.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&ColorPreset> {
        self.presets.iter().find(|p| p.id == id)
    }

    /// Return all presets using the given colour space.
    #[must_use]
    pub fn find_by_space(&self, space: ColorSpacePreset) -> Vec<&ColorPreset> {
        self.presets
            .iter()
            .filter(|p| p.color_space == space)
            .collect()
    }

    /// Return all HDR presets.
    #[must_use]
    pub fn hdr_presets(&self) -> Vec<&ColorPreset> {
        self.presets.iter().filter(|p| p.is_hdr_color()).collect()
    }

    /// Total number of registered presets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rec709_bit_depth() {
        assert_eq!(ColorSpacePreset::Rec709.bit_depth(), 8);
    }

    #[test]
    fn test_rec2020_bit_depth() {
        assert_eq!(ColorSpacePreset::Rec2020.bit_depth(), 10);
    }

    #[test]
    fn test_acescc_bit_depth() {
        assert_eq!(ColorSpacePreset::AcesCc.bit_depth(), 16);
    }

    #[test]
    fn test_rec709_is_not_hdr() {
        assert!(!ColorSpacePreset::Rec709.is_hdr());
    }

    #[test]
    fn test_rec2020_is_hdr() {
        assert!(ColorSpacePreset::Rec2020.is_hdr());
    }

    #[test]
    fn test_hlg_is_hdr() {
        assert!(ColorSpacePreset::Hlg.is_hdr());
    }

    #[test]
    fn test_color_space_tags() {
        assert_eq!(ColorSpacePreset::Rec709.tag(), "rec709");
        assert_eq!(ColorSpacePreset::Hdr10.tag(), "hdr10");
        assert_eq!(ColorSpacePreset::AcesCc.tag(), "acescc");
    }

    #[test]
    fn test_color_preset_new_sets_transfer() {
        let p = ColorPreset::new("r709", "Rec709", ColorSpacePreset::Rec709);
        assert_eq!(p.transfer_function, "gamma22");
    }

    #[test]
    fn test_color_preset_hdr10_transfer_is_pq() {
        let p = ColorPreset::new("hdr10", "HDR10", ColorSpacePreset::Hdr10);
        assert_eq!(p.transfer_function, "pq");
    }

    #[test]
    fn test_color_preset_is_hdr_color_true() {
        let p = ColorPreset::new("p3", "P3 D65", ColorSpacePreset::P3D65);
        assert!(p.is_hdr_color());
    }

    #[test]
    fn test_color_preset_is_hdr_color_false() {
        let p = ColorPreset::new("srgb", "sRGB", ColorSpacePreset::Srgb);
        assert!(!p.is_hdr_color());
    }

    #[test]
    fn test_color_preset_bit_depth_delegates() {
        let p = ColorPreset::new("r2020", "Rec2020", ColorSpacePreset::Rec2020);
        assert_eq!(p.bit_depth(), 10);
    }

    #[test]
    fn test_manager_add_and_count() {
        let mut mgr = ColorPresetManager::new();
        assert_eq!(mgr.count(), 0);
        mgr.add(ColorPreset::new("a", "A", ColorSpacePreset::Rec709));
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_manager_find_by_id() {
        let mut mgr = ColorPresetManager::new();
        mgr.add(ColorPreset::new("id-a", "A", ColorSpacePreset::Rec709));
        assert!(mgr.find("id-a").is_some());
        assert!(mgr.find("id-b").is_none());
    }

    #[test]
    fn test_manager_find_by_space() {
        let mut mgr = ColorPresetManager::new();
        mgr.add(ColorPreset::new("a", "A", ColorSpacePreset::Rec709));
        mgr.add(ColorPreset::new("b", "B", ColorSpacePreset::Rec709));
        mgr.add(ColorPreset::new("c", "C", ColorSpacePreset::Hdr10));
        let rec709 = mgr.find_by_space(ColorSpacePreset::Rec709);
        assert_eq!(rec709.len(), 2);
    }

    #[test]
    fn test_manager_hdr_presets() {
        let mut mgr = ColorPresetManager::new();
        mgr.add(ColorPreset::new("sdr", "SDR", ColorSpacePreset::Rec709));
        mgr.add(ColorPreset::new("hdr", "HDR", ColorSpacePreset::Hdr10));
        mgr.add(ColorPreset::new("hlg", "HLG", ColorSpacePreset::Hlg));
        let hdrs = mgr.hdr_presets();
        assert_eq!(hdrs.len(), 2);
    }
}
