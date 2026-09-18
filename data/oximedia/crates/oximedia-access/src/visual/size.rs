//! Text size adjustment.

/// Adjusts text size for better readability.
pub struct TextSizeAdjuster {
    multiplier: f32,
}

impl TextSizeAdjuster {
    /// Create a new text size adjuster.
    #[must_use]
    pub fn new(multiplier: f32) -> Self {
        Self {
            multiplier: multiplier.clamp(0.5, 3.0),
        }
    }

    /// Adjust font size.
    #[must_use]
    pub fn adjust_size(&self, original_size: u32) -> u32 {
        ((original_size as f32 * self.multiplier) as u32).max(8)
    }

    /// Get multiplier.
    #[must_use]
    pub const fn multiplier(&self) -> f32 {
        self.multiplier
    }
}

impl Default for TextSizeAdjuster {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjuster_creation() {
        let adjuster = TextSizeAdjuster::new(1.5);
        assert!((adjuster.multiplier() - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adjust_size() {
        let adjuster = TextSizeAdjuster::new(2.0);
        assert_eq!(adjuster.adjust_size(12), 24);
    }

    #[test]
    fn test_minimum_size() {
        let adjuster = TextSizeAdjuster::new(0.5);
        assert!(adjuster.adjust_size(10) >= 8);
    }
}
