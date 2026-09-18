//! Advanced keying algorithms.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Key color type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyColor {
    /// Green screen.
    Green,
    /// Blue screen.
    Blue,
    /// Custom color.
    Custom,
}

/// Advanced keying effect.
///
/// Provides professional green/blue screen keying with multiple algorithms.
pub struct AdvancedKey {
    key_color: KeyColor,
    custom_color: Color,
    threshold: f32,
    tolerance: f32,
    edge_feather: f32,
    despill_strength: f32,
}

impl AdvancedKey {
    /// Create a new advanced key effect.
    #[must_use]
    pub const fn new(key_color: KeyColor) -> Self {
        Self {
            key_color,
            custom_color: Color::rgb(0, 255, 0),
            threshold: 0.4,
            tolerance: 0.2,
            edge_feather: 0.1,
            despill_strength: 0.5,
        }
    }

    /// Set custom key color.
    #[must_use]
    pub const fn with_custom_color(mut self, color: Color) -> Self {
        self.custom_color = color;
        self
    }

    /// Set threshold (0.0 - 1.0).
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set tolerance (0.0 - 1.0).
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance.clamp(0.0, 1.0);
        self
    }

    /// Set edge feathering (0.0 - 1.0).
    #[must_use]
    pub fn with_feather(mut self, feather: f32) -> Self {
        self.edge_feather = feather.clamp(0.0, 1.0);
        self
    }

    /// Set despill strength (0.0 - 1.0).
    #[must_use]
    pub fn with_despill(mut self, strength: f32) -> Self {
        self.despill_strength = strength.clamp(0.0, 1.0);
        self
    }

    fn get_key_color(&self) -> Color {
        match self.key_color {
            KeyColor::Green => Color::rgb(0, 255, 0),
            KeyColor::Blue => Color::rgb(0, 0, 255),
            KeyColor::Custom => self.custom_color,
        }
    }

    fn calculate_key(&self, pixel: [u8; 4]) -> f32 {
        let key = self.get_key_color();
        let r = f32::from(pixel[0]) / 255.0;
        let g = f32::from(pixel[1]) / 255.0;
        let b = f32::from(pixel[2]) / 255.0;

        let key_r = f32::from(key.r) / 255.0;
        let key_g = f32::from(key.g) / 255.0;
        let key_b = f32::from(key.b) / 255.0;

        // Color difference
        let diff = ((r - key_r).powi(2) + (g - key_g).powi(2) + (b - key_b).powi(2)).sqrt();

        // Calculate alpha based on difference
        let alpha = if diff < self.threshold {
            0.0
        } else if diff < self.threshold + self.tolerance {
            (diff - self.threshold) / self.tolerance
        } else {
            1.0
        };

        // Apply edge feathering
        if self.edge_feather > 0.0 && alpha > 0.0 && alpha < 1.0 {
            let feathered = (alpha / self.edge_feather).min(1.0);
            feathered
        } else {
            alpha
        }
    }

    fn despill(&self, pixel: [u8; 4], alpha: f32) -> [u8; 4] {
        if self.despill_strength == 0.0 || alpha == 0.0 {
            return pixel;
        }

        let r = f32::from(pixel[0]);
        let g = f32::from(pixel[1]);
        let b = f32::from(pixel[2]);

        let (new_g, new_b) = match self.key_color {
            KeyColor::Green => {
                let spill = (g - r.max(b)).max(0.0);
                let despilled_g = g - spill * self.despill_strength;
                (despilled_g, b)
            }
            KeyColor::Blue => {
                let spill = (b - r.max(g)).max(0.0);
                let despilled_b = b - spill * self.despill_strength;
                (g, despilled_b)
            }
            KeyColor::Custom => (g, b),
        };

        [pixel[0], new_g as u8, new_b as u8, pixel[3]]
    }
}

impl VideoEffect for AdvancedKey {
    fn name(&self) -> &'static str {
        "Advanced Key"
    }

    fn description(&self) -> &'static str {
        "Professional green/blue screen keying"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let alpha = self.calculate_key(pixel);
                let despilled = self.despill(pixel, alpha);

                let final_alpha = (alpha * 255.0) as u8;
                output.set_pixel(
                    x,
                    y,
                    [despilled[0], despilled[1], despilled[2], final_alpha],
                );
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_key() {
        let mut key = AdvancedKey::new(KeyColor::Green);
        let mut input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");

        // Fill with green
        for y in 0..100 {
            for x in 0..100 {
                input.set_pixel(x, y, [0, 255, 0, 255]);
            }
        }

        let params = EffectParams::new();
        key.apply(&input, &mut output, &params)
            .expect("should succeed in test");

        // Green pixels should be keyed out
        let pixel = output.get_pixel(50, 50).expect("should succeed in test");
        assert!(pixel[3] < 128);
    }

    #[test]
    fn test_key_parameters() {
        let key = AdvancedKey::new(KeyColor::Blue)
            .with_threshold(0.5)
            .with_tolerance(0.3)
            .with_feather(0.2);

        assert_eq!(key.threshold, 0.5);
        assert_eq!(key.tolerance, 0.3);
        assert_eq!(key.edge_feather, 0.2);
    }
}
