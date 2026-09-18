//! Frame rate and pulldown detection.

use serde::{Deserialize, Serialize};

/// Cadence detection metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CadenceMetrics {
    /// Detected frame rate.
    pub detected_fps: f32,

    /// Pulldown pattern detected.
    pub pulldown_detected: bool,

    /// Pulldown pattern (e.g., "3:2", "2:2").
    pub pulldown_pattern: String,

    /// Frame count.
    pub frame_count: u64,
}

/// Cadence detector.
pub struct CadenceDetector {
    metrics: CadenceMetrics,
}

impl CadenceDetector {
    /// Create a new cadence detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: CadenceMetrics::default(),
        }
    }

    /// Process a frame.
    pub fn process_frame(&mut self) {
        self.metrics.frame_count += 1;
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &CadenceMetrics {
        &self.metrics
    }

    /// Reset detector.
    pub fn reset(&mut self) {
        self.metrics = CadenceMetrics::default();
    }
}

impl Default for CadenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cadence_detector() {
        let mut detector = CadenceDetector::new();
        detector.process_frame();
        assert_eq!(detector.metrics().frame_count, 1);
    }
}
