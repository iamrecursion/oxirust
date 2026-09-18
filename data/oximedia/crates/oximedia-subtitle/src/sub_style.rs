//! Subtitle styling types for the OxiMedia subtitle crate.
//!
//! Provides color, font style, alignment, and composite style definitions
//! for subtitles distinct from the existing low-level [`crate::style`] module.

/// An RGBA color for subtitle rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubtitleColor {
    /// Red component (0–255).
    pub r: u8,
    /// Green component (0–255).
    pub g: u8,
    /// Blue component (0–255).
    pub b: u8,
    /// Alpha component (0 = transparent, 255 = fully opaque).
    pub a: u8,
}

impl SubtitleColor {
    /// Create an RGBA color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque white.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(255, 255, 255, 255)
    }

    /// Opaque yellow.
    #[must_use]
    pub const fn yellow() -> Self {
        Self::new(255, 255, 0, 255)
    }

    /// Opaque black.
    #[must_use]
    pub const fn black() -> Self {
        Self::new(0, 0, 0, 255)
    }

    /// Format as a hex RGBA string, e.g. `"#RRGGBBAA"`.
    #[must_use]
    pub fn to_hex_rgba(&self) -> String {
        format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
    }

    /// Whether this color is fully opaque (alpha = 255).
    #[must_use]
    pub const fn is_opaque(&self) -> bool {
        self.a == 255
    }
}

/// Font style flags for subtitle text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FontStyle {
    /// Bold text.
    pub bold: bool,
    /// Italic text.
    pub italic: bool,
    /// Underlined text.
    pub underline: bool,
    /// Strikethrough text.
    pub strikethrough: bool,
}

impl FontStyle {
    /// Plain (unstyled) text.
    #[must_use]
    pub const fn plain() -> Self {
        Self {
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }

    /// Bold-only style.
    #[must_use]
    pub const fn bold_style() -> Self {
        Self {
            bold: true,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }

    /// Italic-only style.
    #[must_use]
    pub const fn italic_style() -> Self {
        Self {
            bold: false,
            italic: true,
            underline: false,
            strikethrough: false,
        }
    }
}

/// Alignment position for a subtitle block on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubtitleAlignment {
    /// Top-left corner.
    TopLeft,
    /// Top center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Vertically centered, left-aligned.
    MiddleLeft,
    /// Vertically and horizontally centered.
    MiddleCenter,
    /// Vertically centered, right-aligned.
    MiddleRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom center (most common for subtitles).
    #[default]
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl SubtitleAlignment {
    /// Whether the alignment places the subtitle at the bottom of the frame.
    #[must_use]
    pub const fn is_bottom(&self) -> bool {
        matches!(
            self,
            Self::BottomLeft | Self::BottomCenter | Self::BottomRight
        )
    }

    /// Whether the alignment is horizontally centered.
    #[must_use]
    pub const fn is_centered(&self) -> bool {
        matches!(
            self,
            Self::TopCenter | Self::MiddleCenter | Self::BottomCenter
        )
    }
}

/// A complete subtitle style definition.
#[derive(Debug, Clone)]
pub struct SubtitleStyle {
    /// Style name (e.g. "Default", "Forced").
    pub name: String,
    /// Font family name.
    pub font_family: String,
    /// Font size in points.
    pub font_size_pt: f32,
    /// Primary (text) color.
    pub primary_color: SubtitleColor,
    /// Outline color.
    pub outline_color: SubtitleColor,
    /// Shadow color.
    pub shadow_color: SubtitleColor,
    /// Whether the text is bold.
    pub bold: bool,
    /// Whether the text is italic.
    pub italic: bool,
    /// Alignment on screen.
    pub alignment: SubtitleAlignment,
}

impl SubtitleStyle {
    /// Create a default broadcast subtitle style.
    ///
    /// White 48pt Arial at the bottom center with a black outline.
    #[must_use]
    pub fn default_style() -> Self {
        Self {
            name: "Default".to_string(),
            font_family: "Arial".to_string(),
            font_size_pt: 48.0,
            primary_color: SubtitleColor::white(),
            outline_color: SubtitleColor::black(),
            shadow_color: SubtitleColor::new(0, 0, 0, 128),
            bold: false,
            italic: false,
            alignment: SubtitleAlignment::BottomCenter,
        }
    }

    /// Create a forced-subtitle style (smaller, yellow).
    #[must_use]
    pub fn forced_style() -> Self {
        Self {
            name: "Forced".to_string(),
            font_family: "Arial".to_string(),
            font_size_pt: 36.0,
            primary_color: SubtitleColor::yellow(),
            outline_color: SubtitleColor::black(),
            shadow_color: SubtitleColor::new(0, 0, 0, 96),
            bold: false,
            italic: true,
            alignment: SubtitleAlignment::BottomCenter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_white() {
        let c = SubtitleColor::white();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 255);
        assert_eq!(c.b, 255);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_color_yellow() {
        let c = SubtitleColor::yellow();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 255);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn test_color_black() {
        let c = SubtitleColor::black();
        assert_eq!(c.r, 0);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn test_color_to_hex_rgba_white() {
        let hex = SubtitleColor::white().to_hex_rgba();
        assert_eq!(hex, "#FFFFFFFF");
    }

    #[test]
    fn test_color_to_hex_rgba_black() {
        let hex = SubtitleColor::black().to_hex_rgba();
        assert_eq!(hex, "#000000FF");
    }

    #[test]
    fn test_color_is_opaque_true() {
        assert!(SubtitleColor::white().is_opaque());
    }

    #[test]
    fn test_color_is_opaque_false() {
        let c = SubtitleColor::new(0, 0, 0, 128);
        assert!(!c.is_opaque());
    }

    #[test]
    fn test_font_style_plain() {
        let f = FontStyle::plain();
        assert!(!f.bold && !f.italic && !f.underline && !f.strikethrough);
    }

    #[test]
    fn test_font_style_bold() {
        let f = FontStyle::bold_style();
        assert!(f.bold);
        assert!(!f.italic);
    }

    #[test]
    fn test_font_style_italic() {
        let f = FontStyle::italic_style();
        assert!(f.italic);
        assert!(!f.bold);
    }

    #[test]
    fn test_alignment_is_bottom() {
        assert!(SubtitleAlignment::BottomCenter.is_bottom());
        assert!(SubtitleAlignment::BottomLeft.is_bottom());
        assert!(!SubtitleAlignment::TopCenter.is_bottom());
    }

    #[test]
    fn test_alignment_is_centered() {
        assert!(SubtitleAlignment::BottomCenter.is_centered());
        assert!(SubtitleAlignment::MiddleCenter.is_centered());
        assert!(!SubtitleAlignment::BottomLeft.is_centered());
    }

    #[test]
    fn test_subtitle_style_default() {
        let s = SubtitleStyle::default_style();
        assert_eq!(s.name, "Default");
        assert_eq!(s.font_family, "Arial");
        assert!((s.font_size_pt - 48.0).abs() < 1e-4);
        assert_eq!(s.alignment, SubtitleAlignment::BottomCenter);
    }

    #[test]
    fn test_subtitle_style_forced_is_yellow() {
        let s = SubtitleStyle::forced_style();
        assert_eq!(s.primary_color, SubtitleColor::yellow());
        assert!(s.italic);
    }
}
