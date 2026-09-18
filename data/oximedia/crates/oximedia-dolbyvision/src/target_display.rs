//! Dolby Vision target display characterization.
//!
//! This module models the characteristics of the display that will render
//! Dolby Vision content. The RPU metadata must be tailored to the target
//! display's peak luminance, color gamut, and electro-optical transfer
//! function (EOTF) capabilities.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// DisplayType
// ---------------------------------------------------------------------------

/// Broad category of display technology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayType {
    /// OLED panel.
    Oled,
    /// LCD with LED back-lighting.
    LcdLed,
    /// Micro-LED panel.
    MicroLed,
    /// Laser-phosphor projector.
    LaserProjector,
    /// Conventional DLP projector.
    DlpProjector,
    /// Reference mastering monitor.
    ReferenceMaster,
    /// Generic / unspecified.
    Generic,
}

impl DisplayType {
    /// Returns `true` for self-emissive panel technologies (pixel-level dimming).
    #[must_use]
    pub const fn is_emissive(self) -> bool {
        matches!(self, Self::Oled | Self::MicroLed)
    }

    /// Returns `true` for projection-based display types.
    #[must_use]
    pub const fn is_projector(self) -> bool {
        matches!(self, Self::LaserProjector | Self::DlpProjector)
    }

    /// Short label for display type.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Oled => "OLED",
            Self::LcdLed => "LCD-LED",
            Self::MicroLed => "MicroLED",
            Self::LaserProjector => "Laser",
            Self::DlpProjector => "DLP",
            Self::ReferenceMaster => "Reference",
            Self::Generic => "Generic",
        }
    }
}

impl fmt::Display for DisplayType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// DvTargetDisplay
// ---------------------------------------------------------------------------

/// Describes a specific target display used by Dolby Vision tone mapping.
///
/// The DV RPU metadata uses target display information to tailor the final
/// pixel values for accurate reproduction on the viewing device.
#[derive(Debug, Clone)]
pub struct DvTargetDisplay {
    /// Unique target display identifier (e.g., SMPTE target ID).
    pub target_id: u16,
    /// Display type / technology.
    pub display_type: DisplayType,
    /// Peak luminance in nits (cd/m^2), must be > 0.
    pub peak_luminance_nits: f64,
    /// Minimum black level in nits, must be >= 0.
    pub min_luminance_nits: f64,
    /// Diagonal size in inches (0 for unspecified).
    pub diagonal_inches: f32,
    /// Supports wide color gamut (Rec. 2020).
    pub supports_widecolor: bool,
    /// Human-readable display name.
    pub name: String,
}

impl DvTargetDisplay {
    /// Create a new target display descriptor.
    #[must_use]
    pub fn new(target_id: u16, display_type: DisplayType, peak_luminance_nits: f64) -> Self {
        Self {
            target_id,
            display_type,
            peak_luminance_nits,
            min_luminance_nits: 0.005,
            diagonal_inches: 0.0,
            supports_widecolor: false,
            name: String::new(),
        }
    }

    /// Set minimum black level.
    #[must_use]
    pub fn with_min_luminance(mut self, nits: f64) -> Self {
        self.min_luminance_nits = nits.max(0.0);
        self
    }

    /// Set display name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set diagonal size.
    #[must_use]
    pub fn with_diagonal(mut self, inches: f32) -> Self {
        self.diagonal_inches = inches.max(0.0);
        self
    }

    /// Set wide-color-gamut support.
    #[must_use]
    pub fn with_widecolor(mut self, flag: bool) -> Self {
        self.supports_widecolor = flag;
        self
    }

    /// Dynamic range ratio: peak / min luminance.
    #[must_use]
    pub fn dynamic_range_ratio(&self) -> f64 {
        if self.min_luminance_nits <= 0.0 {
            return f64::INFINITY;
        }
        self.peak_luminance_nits / self.min_luminance_nits
    }

    /// Returns `true` if peak luminance is high enough for HDR (>= 600 nits).
    #[must_use]
    pub fn is_hdr_capable(&self) -> bool {
        self.peak_luminance_nits >= 600.0
    }

    /// Validate the display descriptor for sane values.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.peak_luminance_nits > 0.0
            && self.min_luminance_nits >= 0.0
            && self.peak_luminance_nits > self.min_luminance_nits
    }
}

impl fmt::Display for DvTargetDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Target#{} {} {:.0}nits",
            self.target_id, self.display_type, self.peak_luminance_nits
        )
    }
}

// ---------------------------------------------------------------------------
// DisplayCapabilities
// ---------------------------------------------------------------------------

/// Aggregated capabilities for a set of target displays.
///
/// When authoring content for multiple targets, this struct captures the
/// common denominator and the full range of display parameters.
#[derive(Debug, Clone)]
pub struct DisplayCapabilities {
    displays: Vec<DvTargetDisplay>,
}

impl DisplayCapabilities {
    /// Create an empty capability set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            displays: Vec::new(),
        }
    }

    /// Add a target display.
    pub fn add(&mut self, display: DvTargetDisplay) {
        self.displays.push(display);
    }

    /// Number of registered displays.
    #[must_use]
    pub fn count(&self) -> usize {
        self.displays.len()
    }

    /// Returns `true` when no displays have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.displays.is_empty()
    }

    /// Maximum peak luminance across all registered displays.
    #[must_use]
    pub fn max_peak_luminance(&self) -> f64 {
        self.displays
            .iter()
            .map(|d| d.peak_luminance_nits)
            .fold(0.0_f64, f64::max)
    }

    /// Minimum peak luminance across all registered displays.
    #[must_use]
    pub fn min_peak_luminance(&self) -> f64 {
        self.displays
            .iter()
            .map(|d| d.peak_luminance_nits)
            .fold(f64::INFINITY, f64::min)
    }

    /// Check if all displays support HDR (>= 600 nits).
    #[must_use]
    pub fn all_hdr_capable(&self) -> bool {
        !self.displays.is_empty() && self.displays.iter().all(|d| d.is_hdr_capable())
    }

    /// Check if any display supports wide color gamut.
    #[must_use]
    pub fn any_widecolor(&self) -> bool {
        self.displays.iter().any(|d| d.supports_widecolor)
    }

    /// Iterate over all registered displays.
    pub fn iter(&self) -> impl Iterator<Item = &DvTargetDisplay> {
        self.displays.iter()
    }

    /// Returns displays filtered by display type.
    #[must_use]
    pub fn by_type(&self, dt: DisplayType) -> Vec<&DvTargetDisplay> {
        self.displays
            .iter()
            .filter(|d| d.display_type == dt)
            .collect()
    }
}

impl Default for DisplayCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn oled_display(id: u16, peak: f64) -> DvTargetDisplay {
        DvTargetDisplay::new(id, DisplayType::Oled, peak)
    }

    #[test]
    fn test_display_type_emissive() {
        assert!(DisplayType::Oled.is_emissive());
        assert!(DisplayType::MicroLed.is_emissive());
        assert!(!DisplayType::LcdLed.is_emissive());
    }

    #[test]
    fn test_display_type_projector() {
        assert!(DisplayType::LaserProjector.is_projector());
        assert!(DisplayType::DlpProjector.is_projector());
        assert!(!DisplayType::Oled.is_projector());
    }

    #[test]
    fn test_display_type_label() {
        assert_eq!(DisplayType::Oled.label(), "OLED");
        assert_eq!(DisplayType::Generic.label(), "Generic");
    }

    #[test]
    fn test_display_type_display() {
        assert_eq!(format!("{}", DisplayType::MicroLed), "MicroLED");
    }

    #[test]
    fn test_target_display_new() {
        let d = oled_display(1, 1000.0);
        assert_eq!(d.target_id, 1);
        assert!((d.peak_luminance_nits - 1000.0).abs() < f64::EPSILON);
        assert!(d.is_valid());
    }

    #[test]
    fn test_target_display_builders() {
        let d = DvTargetDisplay::new(2, DisplayType::LcdLed, 600.0)
            .with_min_luminance(0.01)
            .with_name("Test LCD")
            .with_diagonal(55.0)
            .with_widecolor(true);
        assert_eq!(d.name, "Test LCD");
        assert!(d.supports_widecolor);
        assert!((d.diagonal_inches - 55.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_target_display_dynamic_range() {
        let d = oled_display(1, 1000.0).with_min_luminance(0.001);
        let ratio = d.dynamic_range_ratio();
        assert!((ratio - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_target_display_dynamic_range_zero_min() {
        let d = oled_display(1, 1000.0).with_min_luminance(0.0);
        assert!(d.dynamic_range_ratio().is_infinite());
    }

    #[test]
    fn test_target_display_hdr_capable() {
        assert!(oled_display(1, 1000.0).is_hdr_capable());
        assert!(oled_display(2, 600.0).is_hdr_capable());
        assert!(!oled_display(3, 400.0).is_hdr_capable());
    }

    #[test]
    fn test_target_display_is_valid() {
        assert!(oled_display(1, 1000.0).is_valid());
        let bad = DvTargetDisplay::new(2, DisplayType::Oled, 0.0);
        assert!(!bad.is_valid());
    }

    #[test]
    fn test_target_display_format() {
        let d = oled_display(42, 1000.0);
        let s = format!("{d}");
        assert!(s.contains("42"));
        assert!(s.contains("OLED"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn test_capabilities_empty() {
        let cap = DisplayCapabilities::new();
        assert!(cap.is_empty());
        assert_eq!(cap.count(), 0);
    }

    #[test]
    fn test_capabilities_add_and_count() {
        let mut cap = DisplayCapabilities::new();
        cap.add(oled_display(1, 1000.0));
        cap.add(oled_display(2, 4000.0));
        assert_eq!(cap.count(), 2);
    }

    #[test]
    fn test_capabilities_max_peak_luminance() {
        let mut cap = DisplayCapabilities::new();
        cap.add(oled_display(1, 1000.0));
        cap.add(oled_display(2, 4000.0));
        assert!((cap.max_peak_luminance() - 4000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_capabilities_min_peak_luminance() {
        let mut cap = DisplayCapabilities::new();
        cap.add(oled_display(1, 1000.0));
        cap.add(oled_display(2, 4000.0));
        assert!((cap.min_peak_luminance() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_capabilities_all_hdr_capable() {
        let mut cap = DisplayCapabilities::new();
        cap.add(oled_display(1, 1000.0));
        cap.add(oled_display(2, 600.0));
        assert!(cap.all_hdr_capable());
        cap.add(oled_display(3, 300.0));
        assert!(!cap.all_hdr_capable());
    }

    #[test]
    fn test_capabilities_any_widecolor() {
        let mut cap = DisplayCapabilities::new();
        cap.add(oled_display(1, 1000.0));
        assert!(!cap.any_widecolor());
        cap.add(oled_display(2, 4000.0).with_widecolor(true));
        assert!(cap.any_widecolor());
    }

    #[test]
    fn test_capabilities_by_type() {
        let mut cap = DisplayCapabilities::new();
        cap.add(oled_display(1, 1000.0));
        cap.add(DvTargetDisplay::new(2, DisplayType::LcdLed, 600.0));
        cap.add(oled_display(3, 4000.0));
        let oleds = cap.by_type(DisplayType::Oled);
        assert_eq!(oleds.len(), 2);
    }
}
