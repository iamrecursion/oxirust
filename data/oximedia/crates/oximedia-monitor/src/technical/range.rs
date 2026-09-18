//! Luma and chroma range checking.

use crate::MonitorResult;
use serde::{Deserialize, Serialize};

/// Range checking metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RangeMetrics {
    /// Luma range violations.
    pub luma_violations: u64,

    /// Chroma range violations.
    pub chroma_violations: u64,

    /// Percentage of pixels in legal range.
    pub legal_range_percentage: f32,
}

/// Range checker.
pub struct RangeChecker {
    metrics: RangeMetrics,
}

impl RangeChecker {
    /// Create a new range checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: RangeMetrics::default(),
        }
    }

    /// Check frame ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check(&mut self, frame: &[u8], width: u32, height: u32) -> MonitorResult<()> {
        let mut luma_violations = 0u64;
        let mut chroma_violations = 0u64;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    if luma < 16.0 || luma > 235.0 {
                        luma_violations += 1;
                    }

                    let cb = (b - luma) * 0.5;
                    let cr = (r - luma) * 0.5;
                    if cb.abs() > 112.0 || cr.abs() > 112.0 {
                        chroma_violations += 1;
                    }
                }
            }
        }

        let pixel_count = (width * height) as f32;
        let legal_pixels = pixel_count - luma_violations as f32;

        self.metrics.luma_violations = luma_violations;
        self.metrics.chroma_violations = chroma_violations;
        self.metrics.legal_range_percentage = (legal_pixels / pixel_count) * 100.0;

        Ok(())
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &RangeMetrics {
        &self.metrics
    }

    /// Reset checker.
    pub fn reset(&mut self) {
        self.metrics = RangeMetrics::default();
    }
}

impl Default for RangeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_checker() {
        let mut checker = RangeChecker::new();
        let frame = vec![128u8; 100 * 100 * 3];
        assert!(checker.check(&frame, 100, 100).is_ok());
    }
}
