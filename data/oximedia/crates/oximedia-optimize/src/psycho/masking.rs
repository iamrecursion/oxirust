//! Visual masking models.

/// Visual masking effect calculator.
pub struct VisualMasking {
    luminance_masking: bool,
    texture_masking: bool,
}

impl Default for VisualMasking {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualMasking {
    /// Creates a new visual masking calculator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            luminance_masking: true,
            texture_masking: true,
        }
    }

    /// Calculates masking strength for a block.
    #[must_use]
    pub fn calculate_masking(&self, luminance: u8, variance: f64) -> MaskingStrength {
        let luminance_factor = if self.luminance_masking {
            self.luminance_masking_factor(luminance)
        } else {
            1.0
        };

        let texture_factor = if self.texture_masking {
            self.texture_masking_factor(variance)
        } else {
            1.0
        };

        let total_masking = luminance_factor * texture_factor;

        MaskingStrength {
            luminance_factor,
            texture_factor,
            total_masking,
        }
    }

    /// Luminance masking: darker and brighter areas can hide more distortion.
    fn luminance_masking_factor(&self, luminance: u8) -> f64 {
        // Maximum masking at very dark and very bright areas
        let luma = f64::from(luminance) / 255.0;
        let distance_from_mid = (luma - 0.5).abs();
        1.0 + 0.5 * distance_from_mid
    }

    /// Texture masking: textured areas can hide more distortion.
    fn texture_masking_factor(&self, variance: f64) -> f64 {
        // Higher variance = more texture = more masking
        1.0 + (variance / 1000.0).min(0.5)
    }
}

/// Masking strength components.
#[derive(Debug, Clone, Copy)]
pub struct MaskingStrength {
    /// Luminance masking component.
    pub luminance_factor: f64,
    /// Texture masking component.
    pub texture_factor: f64,
    /// Total masking strength.
    pub total_masking: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_masking_creation() {
        let masking = VisualMasking::new();
        assert!(masking.luminance_masking);
        assert!(masking.texture_masking);
    }

    #[test]
    fn test_luminance_masking_mid() {
        let masking = VisualMasking::new();
        let strength = masking.calculate_masking(128, 0.0);
        assert!((strength.luminance_factor - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_luminance_masking_dark() {
        let masking = VisualMasking::new();
        let strength = masking.calculate_masking(0, 0.0);
        assert!(strength.luminance_factor > 1.0);
    }

    #[test]
    fn test_luminance_masking_bright() {
        let masking = VisualMasking::new();
        let strength = masking.calculate_masking(255, 0.0);
        assert!(strength.luminance_factor > 1.0);
    }

    #[test]
    fn test_texture_masking() {
        let masking = VisualMasking::new();
        let low_variance = masking.calculate_masking(128, 10.0);
        let high_variance = masking.calculate_masking(128, 500.0);
        assert!(high_variance.texture_factor > low_variance.texture_factor);
    }
}
