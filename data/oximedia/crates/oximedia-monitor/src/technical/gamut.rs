//! Color gamut violation detection.

use crate::MonitorResult;
use serde::{Deserialize, Serialize};

/// Gamut violation metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GamutMetrics {
    /// Number of out-of-gamut pixels.
    pub violation_count: u64,

    /// Percentage of pixels out of gamut.
    pub violation_percentage: f32,

    /// Maximum gamut overflow.
    pub max_overflow: f32,
}

/// Gamut checker.
pub struct GamutChecker {
    metrics: GamutMetrics,
}

impl GamutChecker {
    /// Create a new gamut checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: GamutMetrics::default(),
        }
    }

    /// Check frame for gamut violations.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check(&mut self, frame: &[u8], width: u32, height: u32) -> MonitorResult<()> {
        let mut violations = 0u64;
        let mut max_overflow = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    // Check Rec.709 legal range
                    let overflow = Self::calculate_overflow(r, g, b);
                    if overflow > 0.0 {
                        violations += 1;
                        max_overflow = max_overflow.max(overflow);
                    }
                }
            }
        }

        let pixel_count = (width * height) as f32;
        self.metrics.violation_count = violations;
        self.metrics.violation_percentage = (violations as f32 / pixel_count) * 100.0;
        self.metrics.max_overflow = max_overflow;

        Ok(())
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &GamutMetrics {
        &self.metrics
    }

    /// Reset checker.
    pub fn reset(&mut self) {
        self.metrics = GamutMetrics::default();
    }

    fn calculate_overflow(r: f32, g: f32, b: f32) -> f32 {
        let mut overflow = 0.0f32;

        // Legal broadcast range is 16-235 for 8-bit
        if r < 16.0 {
            overflow = overflow.max(16.0 - r);
        } else if r > 235.0 {
            overflow = overflow.max(r - 235.0);
        }

        if g < 16.0 {
            overflow = overflow.max(16.0 - g);
        } else if g > 235.0 {
            overflow = overflow.max(g - 235.0);
        }

        if b < 16.0 {
            overflow = overflow.max(16.0 - b);
        } else if b > 235.0 {
            overflow = overflow.max(b - 235.0);
        }

        overflow / 255.0
    }
}

impl Default for GamutChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamut_checker() {
        let mut checker = GamutChecker::new();
        let frame = vec![128u8; 100 * 100 * 3];
        assert!(checker.check(&frame, 100, 100).is_ok());
    }
}
