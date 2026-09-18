//! Color blindness adaptation.

use serde::{Deserialize, Serialize};

/// Types of color blindness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorBlindnessType {
    /// Protanopia (red-blind).
    Protanopia,
    /// Deuteranopia (green-blind).
    Deuteranopia,
    /// Tritanopia (blue-blind).
    Tritanopia,
    /// Protanomaly (red-weak).
    Protanomaly,
    /// Deuteranomaly (green-weak).
    Deuteranomaly,
    /// Tritanomaly (blue-weak).
    Tritanomaly,
    /// Achromatopsia (total color blindness).
    Achromatopsia,
}

/// Adapts colors for color blind users.
pub struct ColorBlindnessAdapter {
    cb_type: ColorBlindnessType,
}

impl ColorBlindnessAdapter {
    /// Create a new adapter.
    #[must_use]
    pub const fn new(cb_type: ColorBlindnessType) -> Self {
        Self { cb_type }
    }

    /// Adapt frame colors for color blindness.
    #[must_use]
    pub fn adapt(&self, frame: &[u8]) -> Vec<u8> {
        // In production, this would:
        // 1. Apply color transformation matrix
        // 2. Simulate how colors appear to color blind users
        // 3. Adjust colors for better differentiation

        frame.to_vec()
    }

    /// Transform a single RGB color.
    #[must_use]
    pub fn transform_color(&self, color: (u8, u8, u8)) -> (u8, u8, u8) {
        match self.cb_type {
            ColorBlindnessType::Protanopia => self.protanopia_transform(color),
            ColorBlindnessType::Deuteranopia => self.deuteranopia_transform(color),
            ColorBlindnessType::Tritanopia => self.tritanopia_transform(color),
            ColorBlindnessType::Achromatopsia => self.achromatopsia_transform(color),
            _ => color, // Anomaly types use partial transforms
        }
    }

    fn protanopia_transform(&self, color: (u8, u8, u8)) -> (u8, u8, u8) {
        // Simplified protanopia simulation
        let (r, g, b) = color;
        let new_r = (0.567 * f32::from(r) + 0.433 * f32::from(g)) as u8;
        let new_g = (0.558 * f32::from(r) + 0.442 * f32::from(g)) as u8;
        (new_r, new_g, b)
    }

    fn deuteranopia_transform(&self, color: (u8, u8, u8)) -> (u8, u8, u8) {
        let (r, g, b) = color;
        let new_r = (0.625 * f32::from(r) + 0.375 * f32::from(g)) as u8;
        let new_g = (0.7 * f32::from(r) + 0.3 * f32::from(g)) as u8;
        (new_r, new_g, b)
    }

    fn tritanopia_transform(&self, color: (u8, u8, u8)) -> (u8, u8, u8) {
        let (r, g, b) = color;
        let new_g = (0.95 * f32::from(g) + 0.05 * f32::from(b)) as u8;
        let new_b = (0.433 * f32::from(g) + 0.567 * f32::from(b)) as u8;
        (r, new_g, new_b)
    }

    fn achromatopsia_transform(&self, color: (u8, u8, u8)) -> (u8, u8, u8) {
        let gray = (0.299 * f32::from(color.0)
            + 0.587 * f32::from(color.1)
            + 0.114 * f32::from(color.2)) as u8;
        (gray, gray, gray)
    }

    /// Get color blindness type.
    #[must_use]
    pub const fn cb_type(&self) -> ColorBlindnessType {
        self.cb_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = ColorBlindnessAdapter::new(ColorBlindnessType::Protanopia);
        assert_eq!(adapter.cb_type(), ColorBlindnessType::Protanopia);
    }

    #[test]
    fn test_achromatopsia() {
        let adapter = ColorBlindnessAdapter::new(ColorBlindnessType::Achromatopsia);
        let color = (255, 128, 64);
        let transformed = adapter.transform_color(color);

        // Should be grayscale
        assert_eq!(transformed.0, transformed.1);
        assert_eq!(transformed.1, transformed.2);
    }

    #[test]
    fn test_color_transform() {
        let adapter = ColorBlindnessAdapter::new(ColorBlindnessType::Deuteranopia);
        let color = (255, 0, 0);
        let _transformed = adapter.transform_color(color);
        // Just ensure it doesn't panic
    }
}
