//! Color harmony analysis and generation utilities.
//!
//! Provides tools for computing harmonious color relationships based on
//! classical color theory (complementary, triadic, analogous, etc.).

#![allow(dead_code)]

/// Types of classical color harmony relationships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarmonyType {
    /// Two colors directly opposite on the color wheel (180°).
    Complementary,
    /// Three colors evenly spaced at 120° intervals.
    Triadic,
    /// Colors adjacent on the wheel (±30° from primary).
    Analogous,
    /// Primary plus two colors adjacent to its complement (±150°, ±210°).
    SplitComplementary,
    /// Four colors forming a rectangle on the wheel (60° + 120° pattern).
    Tetradic,
    /// Four colors evenly spaced at 90° intervals.
    Square,
}

impl HarmonyType {
    /// Returns the hue offsets (in degrees) from the primary hue to produce each harmony.
    ///
    /// The primary hue is always at offset 0°; returned slice includes all harmony members.
    #[must_use]
    pub fn angle_offsets(self) -> &'static [f64] {
        match self {
            Self::Complementary => &[0.0, 180.0],
            Self::Triadic => &[0.0, 120.0, 240.0],
            Self::Analogous => &[0.0, 30.0, 60.0],
            Self::SplitComplementary => &[0.0, 150.0, 210.0],
            Self::Tetradic => &[0.0, 60.0, 180.0, 240.0],
            Self::Square => &[0.0, 90.0, 180.0, 270.0],
        }
    }

    /// Returns the number of colors in this harmony scheme.
    #[must_use]
    pub fn color_count(self) -> usize {
        self.angle_offsets().len()
    }

    /// Returns a human-readable name for this harmony type.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Complementary => "Complementary",
            Self::Triadic => "Triadic",
            Self::Analogous => "Analogous",
            Self::SplitComplementary => "Split Complementary",
            Self::Tetradic => "Tetradic",
            Self::Square => "Square",
        }
    }
}

/// An HSL color (hue 0–360, saturation 0–1, lightness 0–1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HslColor {
    /// Hue in degrees (0–360).
    pub hue: f64,
    /// Saturation (0.0–1.0).
    pub saturation: f64,
    /// Lightness (0.0–1.0).
    pub lightness: f64,
}

impl HslColor {
    /// Creates a new `HslColor`, clamping values to valid ranges.
    #[must_use]
    pub fn new(hue: f64, saturation: f64, lightness: f64) -> Self {
        Self {
            hue: hue.rem_euclid(360.0),
            saturation: saturation.clamp(0.0, 1.0),
            lightness: lightness.clamp(0.0, 1.0),
        }
    }

    /// Returns a new `HslColor` with the hue rotated by `degrees`.
    #[must_use]
    pub fn with_hue_offset(self, degrees: f64) -> Self {
        Self::new(self.hue + degrees, self.saturation, self.lightness)
    }
}

/// A set of harmonically related colors.
#[derive(Debug, Clone)]
pub struct HarmonySet {
    /// The harmony type that generated this set.
    pub harmony_type: HarmonyType,
    colors: Vec<HslColor>,
}

impl HarmonySet {
    /// Creates a `HarmonySet` from a primary color and harmony type.
    #[must_use]
    pub fn from_primary(primary: HslColor, harmony_type: HarmonyType) -> Self {
        let colors = harmony_type
            .angle_offsets()
            .iter()
            .map(|&offset| primary.with_hue_offset(offset))
            .collect();
        Self {
            harmony_type,
            colors,
        }
    }

    /// Returns the primary (first) color in the harmony.
    #[must_use]
    pub fn primary(&self) -> HslColor {
        self.colors[0]
    }

    /// Returns all harmony colors excluding the primary.
    #[must_use]
    pub fn harmonics(&self) -> &[HslColor] {
        &self.colors[1..]
    }

    /// Returns all colors in the harmony set (including primary).
    #[must_use]
    pub fn all_colors(&self) -> &[HslColor] {
        &self.colors
    }

    /// Returns the number of colors in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Returns `true` if the set has no colors (should never happen in normal use).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

/// Finds color harmonies for a given primary color.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorHarmonyFinder;

impl ColorHarmonyFinder {
    /// Creates a new `ColorHarmonyFinder`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Generates a [`HarmonySet`] for the given primary color and harmony type.
    #[must_use]
    pub fn find(&self, primary: HslColor, harmony_type: HarmonyType) -> HarmonySet {
        HarmonySet::from_primary(primary, harmony_type)
    }

    /// Generates harmony sets for all known harmony types for the given color.
    #[must_use]
    pub fn find_all(&self, primary: HslColor) -> Vec<HarmonySet> {
        [
            HarmonyType::Complementary,
            HarmonyType::Triadic,
            HarmonyType::Analogous,
            HarmonyType::SplitComplementary,
            HarmonyType::Tetradic,
            HarmonyType::Square,
        ]
        .iter()
        .map(|&ht| self.find(primary, ht))
        .collect()
    }

    /// Returns the contrast ratio (hue distance) between primary and the best complementary hue.
    #[must_use]
    pub fn complementary_hue(primary: HslColor) -> HslColor {
        primary.with_hue_offset(180.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complementary_angle_offsets() {
        let offsets = HarmonyType::Complementary.angle_offsets();
        assert_eq!(offsets, &[0.0, 180.0]);
    }

    #[test]
    fn test_triadic_angle_offsets() {
        let offsets = HarmonyType::Triadic.angle_offsets();
        assert_eq!(offsets.len(), 3);
        assert!((offsets[1] - 120.0).abs() < 1e-10);
        assert!((offsets[2] - 240.0).abs() < 1e-10);
    }

    #[test]
    fn test_analogous_angle_offsets() {
        let offsets = HarmonyType::Analogous.angle_offsets();
        assert_eq!(offsets[1], 30.0);
        assert_eq!(offsets[2], 60.0);
    }

    #[test]
    fn test_split_complementary_offsets() {
        let offsets = HarmonyType::SplitComplementary.angle_offsets();
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[1], 150.0);
        assert_eq!(offsets[2], 210.0);
    }

    #[test]
    fn test_harmony_type_color_count() {
        assert_eq!(HarmonyType::Complementary.color_count(), 2);
        assert_eq!(HarmonyType::Triadic.color_count(), 3);
        assert_eq!(HarmonyType::Square.color_count(), 4);
        assert_eq!(HarmonyType::Tetradic.color_count(), 4);
    }

    #[test]
    fn test_harmony_type_names_non_empty() {
        for ht in [
            HarmonyType::Complementary,
            HarmonyType::Triadic,
            HarmonyType::Analogous,
            HarmonyType::SplitComplementary,
            HarmonyType::Tetradic,
            HarmonyType::Square,
        ] {
            assert!(!ht.name().is_empty());
        }
    }

    #[test]
    fn test_hsl_hue_wrapping() {
        let c = HslColor::new(390.0, 0.5, 0.5);
        assert!((c.hue - 30.0).abs() < 1e-8);
    }

    #[test]
    fn test_hsl_clamping() {
        let c = HslColor::new(180.0, 1.5, -0.2);
        assert!((c.saturation - 1.0).abs() < 1e-10);
        assert!((c.lightness - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hue_offset() {
        let c = HslColor::new(60.0, 0.8, 0.5);
        let complementary = c.with_hue_offset(180.0);
        assert!((complementary.hue - 240.0).abs() < 1e-8);
    }

    #[test]
    fn test_harmony_set_from_primary() {
        let primary = HslColor::new(30.0, 0.8, 0.5);
        let set = HarmonySet::from_primary(primary, HarmonyType::Complementary);
        assert_eq!(set.len(), 2);
        assert!((set.primary().hue - 30.0).abs() < 1e-8);
        assert!((set.harmonics()[0].hue - 210.0).abs() < 1e-8);
    }

    #[test]
    fn test_harmony_set_not_empty() {
        let primary = HslColor::new(120.0, 1.0, 0.5);
        let set = HarmonySet::from_primary(primary, HarmonyType::Triadic);
        assert!(!set.is_empty());
    }

    #[test]
    fn test_color_harmony_finder_find() {
        let finder = ColorHarmonyFinder::new();
        let primary = HslColor::new(0.0, 1.0, 0.5);
        let set = finder.find(primary, HarmonyType::Square);
        assert_eq!(set.all_colors().len(), 4);
        assert!((set.all_colors()[1].hue - 90.0).abs() < 1e-8);
        assert!((set.all_colors()[2].hue - 180.0).abs() < 1e-8);
        assert!((set.all_colors()[3].hue - 270.0).abs() < 1e-8);
    }

    #[test]
    fn test_find_all_returns_six_sets() {
        let finder = ColorHarmonyFinder::new();
        let primary = HslColor::new(200.0, 0.7, 0.5);
        let all = finder.find_all(primary);
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_complementary_hue_static() {
        let primary = HslColor::new(100.0, 0.9, 0.4);
        let comp = ColorHarmonyFinder::complementary_hue(primary);
        assert!((comp.hue - 280.0).abs() < 1e-8);
    }
}
