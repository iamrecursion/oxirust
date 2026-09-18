//! Black and white level analysis.

use crate::{MonitorError, MonitorResult};
use serde::{Deserialize, Serialize};

/// Level analysis metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LevelMetrics {
    /// Black level (0.0-1.0).
    pub black_level: f32,

    /// White level (0.0-1.0).
    pub white_level: f32,

    /// Average luma (0.0-1.0).
    pub avg_luma: f32,

    /// Contrast ratio.
    pub contrast_ratio: f32,

    /// Pixels below black (16 in 8-bit).
    pub below_black_count: u64,

    /// Pixels above white (235 in 8-bit).
    pub above_white_count: u64,
}

/// Level analyzer.
pub struct LevelAnalyzer {
    metrics: LevelMetrics,
}

impl LevelAnalyzer {
    /// Create a new level analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: LevelMetrics::default(),
        }
    }

    /// Analyze frame levels.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    pub fn analyze(&mut self, frame: &[u8], width: u32, height: u32) -> MonitorResult<()> {
        if frame.len() < (width * height * 3) as usize {
            return Err(MonitorError::ProcessingError(
                "Frame buffer too small".to_string(),
            ));
        }

        let mut min_luma = 255.0f32;
        let mut max_luma = 0.0f32;
        let mut sum_luma = 0.0f32;
        let mut below_black = 0u64;
        let mut above_white = 0u64;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    min_luma = min_luma.min(luma);
                    max_luma = max_luma.max(luma);
                    sum_luma += luma;

                    if luma < 16.0 {
                        below_black += 1;
                    }
                    if luma > 235.0 {
                        above_white += 1;
                    }
                }
            }
        }

        let pixel_count = (width * height) as f32;
        self.metrics.black_level = min_luma / 255.0;
        self.metrics.white_level = max_luma / 255.0;
        self.metrics.avg_luma = sum_luma / (pixel_count * 255.0);
        self.metrics.contrast_ratio = if min_luma > 0.0 {
            max_luma / min_luma
        } else {
            0.0
        };
        self.metrics.below_black_count = below_black;
        self.metrics.above_white_count = above_white;

        Ok(())
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &LevelMetrics {
        &self.metrics
    }

    /// Reset analyzer.
    pub fn reset(&mut self) {
        self.metrics = LevelMetrics::default();
    }
}

impl Default for LevelAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_analyzer() {
        let mut analyzer = LevelAnalyzer::new();
        let frame = vec![128u8; 100 * 100 * 3];
        assert!(analyzer.analyze(&frame, 100, 100).is_ok());
    }
}
