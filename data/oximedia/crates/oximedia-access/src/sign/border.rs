//! Sign language video border and styling.

use serde::{Deserialize, Serialize};

/// Border style for sign language video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignBorderStyle {
    /// Solid border.
    Solid,
    /// Rounded border.
    Rounded,
    /// No border.
    None,
}

/// Border configuration for sign language video.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SignBorder {
    /// Border style.
    pub style: SignBorderStyle,
    /// Border width in pixels.
    pub width: u32,
    /// Border color (RGBA).
    pub color: (u8, u8, u8, u8),
    /// Corner radius for rounded borders.
    pub radius: u32,
}

impl Default for SignBorder {
    fn default() -> Self {
        Self {
            style: SignBorderStyle::Solid,
            width: 2,
            color: (255, 255, 255, 255),
            radius: 0,
        }
    }
}

impl SignBorder {
    /// Create a solid border.
    #[must_use]
    pub const fn solid(width: u32, color: (u8, u8, u8, u8)) -> Self {
        Self {
            style: SignBorderStyle::Solid,
            width,
            color,
            radius: 0,
        }
    }

    /// Create a rounded border.
    #[must_use]
    pub const fn rounded(width: u32, color: (u8, u8, u8, u8), radius: u32) -> Self {
        Self {
            style: SignBorderStyle::Rounded,
            width,
            color,
            radius,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_border() {
        let border = SignBorder::default();
        assert_eq!(border.style, SignBorderStyle::Solid);
        assert_eq!(border.width, 2);
    }

    #[test]
    fn test_border_creation() {
        let border = SignBorder::solid(3, (255, 0, 0, 255));
        assert_eq!(border.width, 3);
        assert_eq!(border.color, (255, 0, 0, 255));
    }
}
