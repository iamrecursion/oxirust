//! Color and style options for drawings.

use serde::{Deserialize, Serialize};

/// RGB color.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Color {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
    /// Alpha component (0.0-1.0).
    pub a: f32,
}

impl Color {
    /// Create a new color.
    #[must_use]
    pub fn new(r: u8, g: u8, b: u8, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a: a.clamp(0.0, 1.0),
        }
    }

    /// Create an RGB color with full opacity.
    #[must_use]
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 1.0)
    }

    /// Create an RGBA color.
    #[must_use]
    pub fn rgba(r: u8, g: u8, b: u8, a: f32) -> Self {
        Self::new(r, g, b, a)
    }

    /// Create from hex string (e.g., "#FF0000" or "#FF000080").
    ///
    /// # Errors
    ///
    /// Returns error if hex string is invalid.
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        let hex = hex.trim_start_matches('#');

        match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).map_err(|e| e.to_string())?;
                let g = u8::from_str_radix(&hex[2..4], 16).map_err(|e| e.to_string())?;
                let b = u8::from_str_radix(&hex[4..6], 16).map_err(|e| e.to_string())?;
                Ok(Self::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).map_err(|e| e.to_string())?;
                let g = u8::from_str_radix(&hex[2..4], 16).map_err(|e| e.to_string())?;
                let b = u8::from_str_radix(&hex[4..6], 16).map_err(|e| e.to_string())?;
                let a = u8::from_str_radix(&hex[6..8], 16).map_err(|e| e.to_string())?;
                Ok(Self::rgba(r, g, b, f32::from(a) / 255.0))
            }
            _ => Err("Invalid hex color length".to_string()),
        }
    }

    /// Convert to hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        if (self.a - 1.0).abs() < 0.001 {
            format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
        } else {
            format!(
                "#{:02X}{:02X}{:02X}{:02X}",
                self.r,
                self.g,
                self.b,
                (self.a * 255.0) as u8
            )
        }
    }

    /// Predefined red color.
    #[must_use]
    pub fn red() -> Self {
        Self::rgb(255, 0, 0)
    }

    /// Predefined green color.
    #[must_use]
    pub fn green() -> Self {
        Self::rgb(0, 255, 0)
    }

    /// Predefined blue color.
    #[must_use]
    pub fn blue() -> Self {
        Self::rgb(0, 0, 255)
    }

    /// Predefined yellow color.
    #[must_use]
    pub fn yellow() -> Self {
        Self::rgb(255, 255, 0)
    }

    /// Predefined white color.
    #[must_use]
    pub fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    /// Predefined black color.
    #[must_use]
    pub fn black() -> Self {
        Self::rgb(0, 0, 0)
    }
}

/// Stroke style for drawings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StrokeStyle {
    /// Stroke color.
    pub color: Color,
    /// Stroke width.
    pub width: f32,
    /// Line cap style.
    pub cap: LineCap,
    /// Line join style.
    pub join: LineJoin,
    /// Dash pattern (if any).
    pub dash_pattern: Option<DashPattern>,
}

impl StrokeStyle {
    /// Create a new stroke style.
    #[must_use]
    pub fn new(color: Color, width: f32) -> Self {
        Self {
            color,
            width,
            cap: LineCap::Round,
            join: LineJoin::Round,
            dash_pattern: None,
        }
    }

    /// Create a solid stroke.
    #[must_use]
    pub fn solid(color: Color, width: f32) -> Self {
        Self::new(color, width)
    }

    /// Create a dashed stroke.
    #[must_use]
    pub fn dashed(color: Color, width: f32, dash_length: f32, gap_length: f32) -> Self {
        Self {
            color,
            width,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            dash_pattern: Some(DashPattern {
                dash_length,
                gap_length,
            }),
        }
    }
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self::solid(Color::black(), 2.0)
    }
}

/// Line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineCap {
    /// Butt cap.
    Butt,
    /// Round cap.
    Round,
    /// Square cap.
    Square,
}

/// Line join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineJoin {
    /// Miter join.
    Miter,
    /// Round join.
    Round,
    /// Bevel join.
    Bevel,
}

/// Dash pattern for strokes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DashPattern {
    /// Length of dash.
    pub dash_length: f32,
    /// Length of gap.
    pub gap_length: f32,
}

/// Fill style for shapes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FillStyle {
    /// Fill color.
    pub color: Color,
}

impl FillStyle {
    /// Create a new fill style.
    #[must_use]
    pub fn new(color: Color) -> Self {
        Self { color }
    }

    /// Create a semi-transparent fill.
    #[must_use]
    pub fn semi_transparent(r: u8, g: u8, b: u8, alpha: f32) -> Self {
        Self::new(Color::rgba(r, g, b, alpha))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_creation() {
        let color = Color::new(255, 0, 0, 1.0);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
        assert!((color.a - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#FF0000").expect("should succeed in test");
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);

        let color_with_alpha = Color::from_hex("#FF000080").expect("should succeed in test");
        assert_eq!(color_with_alpha.r, 255);
        assert!((color_with_alpha.a - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_color_to_hex() {
        let color = Color::rgb(255, 0, 0);
        assert_eq!(color.to_hex(), "#FF0000");

        let color_with_alpha = Color::rgba(255, 0, 0, 0.5);
        assert_eq!(color_with_alpha.to_hex(), "#FF00007F");
    }

    #[test]
    fn test_color_presets() {
        assert_eq!(Color::red(), Color::rgb(255, 0, 0));
        assert_eq!(Color::green(), Color::rgb(0, 255, 0));
        assert_eq!(Color::blue(), Color::rgb(0, 0, 255));
    }

    #[test]
    fn test_stroke_style_solid() {
        let style = StrokeStyle::solid(Color::red(), 3.0);
        assert_eq!(style.color, Color::red());
        assert!((style.width - 3.0).abs() < 0.001);
        assert!(style.dash_pattern.is_none());
    }

    #[test]
    fn test_stroke_style_dashed() {
        let style = StrokeStyle::dashed(Color::blue(), 2.0, 5.0, 3.0);
        assert_eq!(style.color, Color::blue());
        assert!(style.dash_pattern.is_some());

        if let Some(pattern) = style.dash_pattern {
            assert!((pattern.dash_length - 5.0).abs() < 0.001);
            assert!((pattern.gap_length - 3.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_fill_style() {
        let fill = FillStyle::new(Color::red());
        assert_eq!(fill.color, Color::red());
    }
}
