#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
//! Interactive colour picker model for broadcast graphics.
//!
//! Supports RGB, HSL, and hex representations with conversions between
//! all formats — useful for UI colour selection in live graphics tools.

/// Colour space identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Red, Green, Blue (linear, 0–255 each).
    Rgb,
    /// Hue (0–360°), Saturation (0–100%), Lightness (0–100%).
    Hsl,
    /// Red, Green, Blue, Alpha (linear, 0–255 each).
    Rgba,
    /// Cyan, Magenta, Yellow, Key (0–100% each).
    Cmyk,
}

impl ColorSpace {
    /// Returns the component names for this colour space.
    pub fn component_names(self) -> &'static [&'static str] {
        match self {
            Self::Rgb => &["R", "G", "B"],
            Self::Hsl => &["H", "S", "L"],
            Self::Rgba => &["R", "G", "B", "A"],
            Self::Cmyk => &["C", "M", "Y", "K"],
        }
    }

    /// Returns the number of components in this colour space.
    pub fn component_count(self) -> usize {
        self.component_names().len()
    }
}

// ── ColorPicker ───────────────────────────────────────────────────────────

/// A stateful colour picker that stores a colour in RGB internally and
/// converts to/from hex and HSL on demand.
#[derive(Debug, Clone)]
pub struct ColorPicker {
    r: u8,
    g: u8,
    b: u8,
    /// Optional alpha channel (0–255).
    a: u8,
}

impl Default for ColorPicker {
    fn default() -> Self {
        Self {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        }
    }
}

impl ColorPicker {
    /// Creates a colour picker initialised to opaque white.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a colour picker from explicit RGBA components.
    pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Creates a colour picker from explicit RGB components (alpha = 255).
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Sets the colour from a hex string (`"#RRGGBB"` or `"#RRGGBBAA"`).
    ///
    /// Returns `false` if the string is not a valid hex colour.
    pub fn set_hex(&mut self, hex: &str) -> bool {
        let hex = hex.trim_start_matches('#');
        match hex.len() {
            6 => {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    u8::from_str_radix(&hex[0..2], 16),
                    u8::from_str_radix(&hex[2..4], 16),
                    u8::from_str_radix(&hex[4..6], 16),
                ) {
                    self.r = r;
                    self.g = g;
                    self.b = b;
                    self.a = 255;
                    true
                } else {
                    false
                }
            }
            8 => {
                if let (Ok(r), Ok(g), Ok(b), Ok(a)) = (
                    u8::from_str_radix(&hex[0..2], 16),
                    u8::from_str_radix(&hex[2..4], 16),
                    u8::from_str_radix(&hex[4..6], 16),
                    u8::from_str_radix(&hex[6..8], 16),
                ) {
                    self.r = r;
                    self.g = g;
                    self.b = b;
                    self.a = a;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Encodes the current colour as `"#RRGGBB"`.
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Encodes the current colour as `"#RRGGBBAA"` (includes alpha).
    pub fn to_hex_alpha(&self) -> String {
        format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
    }

    /// Returns the RGB components as `(r, g, b)` in the range 0–255.
    pub fn to_rgb(&self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }

    /// Returns the RGBA components.
    pub fn to_rgba(&self) -> (u8, u8, u8, u8) {
        (self.r, self.g, self.b, self.a)
    }

    /// Converts to HSL: `(hue 0–360, saturation 0–100, lightness 0–100)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_hsl(&self) -> (f32, f32, f32) {
        let r = self.r as f32 / 255.0;
        let g = self.g as f32 / 255.0;
        let b = self.b as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let l = (max + min) / 2.0;

        if delta < f32::EPSILON {
            return (0.0, 0.0, l * 100.0);
        }

        let s = if l < 0.5 {
            delta / (max + min)
        } else {
            delta / (2.0 - max - min)
        };

        let h = if (max - r).abs() < f32::EPSILON {
            ((g - b) / delta).rem_euclid(6.0) / 6.0
        } else if (max - g).abs() < f32::EPSILON {
            ((b - r) / delta + 2.0) / 6.0
        } else {
            ((r - g) / delta + 4.0) / 6.0
        };

        (h * 360.0, s * 100.0, l * 100.0)
    }

    /// Returns `true` if the perceived luminance is below 50% (dark colour).
    #[allow(clippy::cast_precision_loss)]
    pub fn is_dark(&self) -> bool {
        // Relative luminance per WCAG 2.1
        let linearise = |c: u8| {
            let v = c as f32 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        let lum =
            0.2126 * linearise(self.r) + 0.7152 * linearise(self.g) + 0.0722 * linearise(self.b);
        lum < 0.5
    }

    /// Sets the alpha channel directly.
    pub fn set_alpha(&mut self, a: u8) {
        self.a = a;
    }

    /// Returns the alpha channel value.
    pub fn alpha(&self) -> u8 {
        self.a
    }
}

// ─────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space_component_names_rgb() {
        assert_eq!(ColorSpace::Rgb.component_names(), &["R", "G", "B"]);
    }

    #[test]
    fn test_color_space_component_count() {
        assert_eq!(ColorSpace::Rgb.component_count(), 3);
        assert_eq!(ColorSpace::Rgba.component_count(), 4);
        assert_eq!(ColorSpace::Cmyk.component_count(), 4);
        assert_eq!(ColorSpace::Hsl.component_count(), 3);
    }

    #[test]
    fn test_default_is_white() {
        let p = ColorPicker::default();
        assert_eq!(p.to_rgb(), (255, 255, 255));
        assert_eq!(p.alpha(), 255);
    }

    #[test]
    fn test_from_rgb() {
        let p = ColorPicker::from_rgb(10, 20, 30);
        assert_eq!(p.to_rgb(), (10, 20, 30));
    }

    #[test]
    fn test_set_hex_valid_six_digit() {
        let mut p = ColorPicker::new();
        assert!(p.set_hex("#FF8800"));
        assert_eq!(p.to_rgb(), (255, 136, 0));
        assert_eq!(p.alpha(), 255);
    }

    #[test]
    fn test_set_hex_without_hash() {
        let mut p = ColorPicker::new();
        assert!(p.set_hex("00FF00"));
        assert_eq!(p.to_rgb(), (0, 255, 0));
    }

    #[test]
    fn test_set_hex_invalid_returns_false() {
        let mut p = ColorPicker::new();
        assert!(!p.set_hex("#ZZZZZZ"));
        assert!(!p.set_hex("#123")); // too short
    }

    #[test]
    fn test_set_hex_eight_digit_includes_alpha() {
        let mut p = ColorPicker::new();
        assert!(p.set_hex("#AABBCCDD"));
        assert_eq!(p.to_rgba(), (0xAA, 0xBB, 0xCC, 0xDD));
    }

    #[test]
    fn test_to_hex_roundtrip() {
        let mut p = ColorPicker::new();
        p.set_hex("#1A2B3C");
        assert_eq!(p.to_hex(), "#1A2B3C");
    }

    #[test]
    fn test_to_hsl_red() {
        let p = ColorPicker::from_rgb(255, 0, 0);
        let (h, s, l) = p.to_hsl();
        assert!((h - 0.0).abs() < 1.0 || (h - 360.0).abs() < 1.0);
        assert!(s > 90.0);
        assert!((l - 50.0).abs() < 2.0);
    }

    #[test]
    fn test_to_hsl_white_has_zero_saturation() {
        let p = ColorPicker::from_rgb(255, 255, 255);
        let (_, s, l) = p.to_hsl();
        assert!(s < 1.0);
        assert!((l - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_is_dark_black() {
        let p = ColorPicker::from_rgb(0, 0, 0);
        assert!(p.is_dark());
    }

    #[test]
    fn test_is_dark_white() {
        let p = ColorPicker::from_rgb(255, 255, 255);
        assert!(!p.is_dark());
    }

    #[test]
    fn test_set_alpha() {
        let mut p = ColorPicker::from_rgb(100, 100, 100);
        p.set_alpha(128);
        assert_eq!(p.alpha(), 128);
    }

    #[test]
    fn test_to_hex_alpha_format() {
        let p = ColorPicker::from_rgba(255, 0, 0, 128);
        let h = p.to_hex_alpha();
        assert_eq!(h, "#FF000080");
    }
}
