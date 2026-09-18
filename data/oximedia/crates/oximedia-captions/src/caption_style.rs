//! Caption styling: alignment, font properties, accessibility checking, and style presets.

#![allow(dead_code)]

/// Horizontal alignment of a caption block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionAlignment {
    /// Align text to the left edge of the safe area.
    Left,
    /// Center text horizontally.
    Center,
    /// Align text to the right edge of the safe area.
    Right,
    /// Stretch text across the full width (justified).
    Justify,
}

impl CaptionAlignment {
    /// Return the CSS `text-align` value corresponding to this alignment.
    #[must_use]
    pub fn css_value(&self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Center => "center",
            Self::Right => "right",
            Self::Justify => "justify",
        }
    }

    /// Returns `true` if this alignment is suitable for RTL text.
    #[must_use]
    pub fn is_rtl_compatible(&self) -> bool {
        matches!(self, Self::Right | Self::Center | Self::Justify)
    }
}

/// Font descriptor for a caption.
#[derive(Debug, Clone)]
pub struct CaptionFont {
    /// Font family name (e.g. "Arial", "Courier New").
    pub family: String,
    /// Font size in points.
    pub size_pt: f32,
    /// Whether the font is bold.
    pub bold: bool,
    /// Whether the font is italic.
    pub italic: bool,
}

impl CaptionFont {
    /// Create a new caption font descriptor.
    #[must_use]
    pub fn new(family: impl Into<String>, size_pt: f32) -> Self {
        Self {
            family: family.into(),
            size_pt,
            bold: false,
            italic: false,
        }
    }

    /// Returns `true` if this is a monospace font.
    ///
    /// Heuristic: checks the family name against common monospace families.
    #[must_use]
    pub fn is_monospace(&self) -> bool {
        let lower = self.family.to_lowercase();
        lower.contains("mono")
            || lower.contains("courier")
            || lower.contains("consolas")
            || lower.contains("inconsolata")
            || lower.contains("fira code")
            || lower.contains("source code")
            || lower.contains("hack")
    }

    /// Enable bold styling.
    #[must_use]
    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    /// Enable italic styling.
    #[must_use]
    pub fn italic(mut self) -> Self {
        self.italic = true;
        self
    }

    /// CSS `font-weight` string.
    #[must_use]
    pub fn css_font_weight(&self) -> &'static str {
        if self.bold {
            "bold"
        } else {
            "normal"
        }
    }
}

impl Default for CaptionFont {
    fn default() -> Self {
        Self::new("Arial", 24.0)
    }
}

/// RGBA colour (each channel 0–255).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba(pub u8, pub u8, pub u8, pub u8);

impl Rgba {
    /// Fully opaque white.
    pub const WHITE: Self = Self(255, 255, 255, 255);
    /// Fully opaque black.
    pub const BLACK: Self = Self(0, 0, 0, 255);
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self(0, 0, 0, 0);

    /// Relative luminance (WCAG 2.1 simplified — assumes sRGB).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn relative_luminance(&self) -> f64 {
        let linearize = |c: u8| {
            let s = f64::from(c) / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * linearize(self.0) + 0.7152 * linearize(self.1) + 0.0722 * linearize(self.2)
    }

    /// WCAG contrast ratio against another colour.
    #[must_use]
    pub fn contrast_ratio(&self, other: &Self) -> f64 {
        let l1 = self.relative_luminance();
        let l2 = other.relative_luminance();
        let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
        (lighter + 0.05) / (darker + 0.05)
    }
}

/// Full style descriptor for a caption element.
#[derive(Debug, Clone)]
pub struct CaptionStyle {
    /// Horizontal alignment.
    pub alignment: CaptionAlignment,
    /// Font settings.
    pub font: CaptionFont,
    /// Text colour.
    pub color: Rgba,
    /// Background/box colour.
    pub background: Rgba,
    /// Line spacing multiplier (1.0 = normal).
    pub line_spacing: f32,
    /// Whether an outline/drop-shadow is rendered.
    pub has_outline: bool,
}

impl CaptionStyle {
    /// Create a default accessible style (white text on semi-transparent black).
    #[must_use]
    pub fn default_accessible() -> Self {
        Self {
            alignment: CaptionAlignment::Center,
            font: CaptionFont::new("Arial", 24.0),
            color: Rgba::WHITE,
            background: Rgba(0, 0, 0, 180),
            line_spacing: 1.2,
            has_outline: true,
        }
    }

    /// Returns `true` if the style meets WCAG AA contrast requirements (≥ 4.5:1).
    #[must_use]
    pub fn is_accessible(&self) -> bool {
        self.color.contrast_ratio(&self.background) >= 4.5
    }

    /// Returns the contrast ratio between text and background colours.
    #[must_use]
    pub fn contrast_ratio(&self) -> f64 {
        self.color.contrast_ratio(&self.background)
    }
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self::default_accessible()
    }
}

/// Pre-defined style presets for common captioning scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StylePreset {
    /// White text, black outline — suitable for most backgrounds.
    Standard,
    /// Larger white text on a semi-transparent black bar.
    HighVisibility,
    /// Monospace font for broadcast/technical captions.
    Broadcast,
    /// Minimal styling — transparent background, outline only.
    Minimal,
    /// Forced-narrative style — italic, semi-transparent.
    ForcedNarrative,
}

impl StylePreset {
    /// Build a `CaptionStyle` for this preset.
    #[must_use]
    pub fn style(&self) -> CaptionStyle {
        match self {
            Self::Standard => CaptionStyle {
                alignment: CaptionAlignment::Center,
                font: CaptionFont::new("Arial", 24.0),
                color: Rgba::WHITE,
                background: Rgba(0, 0, 0, 180),
                line_spacing: 1.2,
                has_outline: true,
            },
            Self::HighVisibility => CaptionStyle {
                alignment: CaptionAlignment::Center,
                font: CaptionFont::new("Arial", 32.0).bold(),
                color: Rgba::WHITE,
                background: Rgba(0, 0, 0, 220),
                line_spacing: 1.4,
                has_outline: true,
            },
            Self::Broadcast => CaptionStyle {
                alignment: CaptionAlignment::Left,
                font: CaptionFont::new("Courier New", 22.0),
                color: Rgba::WHITE,
                background: Rgba(0, 0, 0, 200),
                line_spacing: 1.0,
                has_outline: false,
            },
            Self::Minimal => CaptionStyle {
                alignment: CaptionAlignment::Center,
                font: CaptionFont::new("Arial", 20.0),
                color: Rgba::WHITE,
                background: Rgba::TRANSPARENT,
                line_spacing: 1.1,
                has_outline: true,
            },
            Self::ForcedNarrative => CaptionStyle {
                alignment: CaptionAlignment::Center,
                font: CaptionFont::new("Arial", 20.0).italic(),
                color: Rgba::WHITE,
                background: Rgba(0, 0, 0, 120),
                line_spacing: 1.2,
                has_outline: true,
            },
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::HighVisibility => "High Visibility",
            Self::Broadcast => "Broadcast",
            Self::Minimal => "Minimal",
            Self::ForcedNarrative => "Forced Narrative",
        }
    }
}

// ─── unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1 — CaptionAlignment::css_value
    #[test]
    fn test_alignment_css_value() {
        assert_eq!(CaptionAlignment::Left.css_value(), "left");
        assert_eq!(CaptionAlignment::Center.css_value(), "center");
        assert_eq!(CaptionAlignment::Right.css_value(), "right");
        assert_eq!(CaptionAlignment::Justify.css_value(), "justify");
    }

    // 2 — CaptionAlignment RTL compatibility
    #[test]
    fn test_alignment_rtl_compatible() {
        assert!(CaptionAlignment::Right.is_rtl_compatible());
        assert!(CaptionAlignment::Center.is_rtl_compatible());
        assert!(!CaptionAlignment::Left.is_rtl_compatible());
    }

    // 3 — CaptionFont::is_monospace — monospace families
    #[test]
    fn test_font_is_monospace_true() {
        let f = CaptionFont::new("Courier New", 20.0);
        assert!(f.is_monospace());
        let f2 = CaptionFont::new("JetBrains Mono", 18.0);
        assert!(f2.is_monospace());
    }

    // 4 — CaptionFont::is_monospace — proportional families
    #[test]
    fn test_font_is_monospace_false() {
        let f = CaptionFont::new("Arial", 24.0);
        assert!(!f.is_monospace());
    }

    // 5 — CaptionFont builder methods
    #[test]
    fn test_font_builder() {
        let f = CaptionFont::new("Arial", 24.0).bold().italic();
        assert!(f.bold);
        assert!(f.italic);
        assert_eq!(f.css_font_weight(), "bold");
    }

    // 6 — Rgba::relative_luminance for white
    #[test]
    fn test_rgba_luminance_white() {
        let lum = Rgba::WHITE.relative_luminance();
        assert!((lum - 1.0).abs() < 0.01);
    }

    // 7 — Rgba::relative_luminance for black
    #[test]
    fn test_rgba_luminance_black() {
        let lum = Rgba::BLACK.relative_luminance();
        assert!(lum < 0.01);
    }

    // 8 — Rgba::contrast_ratio white on black ≥ 21
    #[test]
    fn test_contrast_ratio_white_on_black() {
        let ratio = Rgba::WHITE.contrast_ratio(&Rgba::BLACK);
        assert!(ratio >= 21.0);
    }

    // 9 — CaptionStyle::is_accessible — default accessible style
    #[test]
    fn test_default_style_is_accessible() {
        let style = CaptionStyle::default_accessible();
        assert!(
            style.is_accessible(),
            "contrast: {}",
            style.contrast_ratio()
        );
    }

    // 10 — CaptionStyle::is_accessible — same-colour fails
    #[test]
    fn test_same_color_not_accessible() {
        let mut style = CaptionStyle::default_accessible();
        style.background = Rgba::WHITE;
        style.color = Rgba::WHITE;
        assert!(!style.is_accessible());
    }

    // 11 — StylePreset::Standard produces accessible style
    #[test]
    fn test_preset_standard_accessible() {
        let style = StylePreset::Standard.style();
        assert!(style.is_accessible());
    }

    // 12 — StylePreset::Broadcast uses monospace font
    #[test]
    fn test_preset_broadcast_monospace() {
        let style = StylePreset::Broadcast.style();
        assert!(style.font.is_monospace());
    }

    // 13 — StylePreset::HighVisibility bold
    #[test]
    fn test_preset_high_visibility_bold() {
        let style = StylePreset::HighVisibility.style();
        assert!(style.font.bold);
    }

    // 14 — StylePreset::ForcedNarrative italic
    #[test]
    fn test_preset_forced_narrative_italic() {
        let style = StylePreset::ForcedNarrative.style();
        assert!(style.font.italic);
    }

    // 15 — StylePreset::label
    #[test]
    fn test_preset_label() {
        assert_eq!(StylePreset::Standard.label(), "Standard");
        assert_eq!(StylePreset::HighVisibility.label(), "High Visibility");
        assert_eq!(StylePreset::Broadcast.label(), "Broadcast");
        assert_eq!(StylePreset::Minimal.label(), "Minimal");
        assert_eq!(StylePreset::ForcedNarrative.label(), "Forced Narrative");
    }
}
