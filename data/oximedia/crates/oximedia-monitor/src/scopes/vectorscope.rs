//! Vectorscope monitor implementation.

use crate::MonitorResult;
use oximedia_scopes::ScopeConfig;
use serde::{Deserialize, Serialize};

/// Vectorscope monitoring metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorscopeMetrics {
    /// Maximum saturation (0.0-1.0).
    pub max_saturation: f32,

    /// Average saturation (0.0-1.0).
    pub avg_saturation: f32,

    /// Dominant hue angle (0-360 degrees).
    pub dominant_hue: f32,

    /// Color gamut violations.
    pub gamut_violations: u64,
}

/// Vectorscope monitor.
pub struct VectorscopeMonitor {
    config: ScopeConfig,
    metrics: VectorscopeMetrics,
    gain: f32,
}

impl VectorscopeMonitor {
    /// Create a new vectorscope monitor.
    #[must_use]
    pub fn new(config: ScopeConfig) -> Self {
        Self {
            config,
            metrics: VectorscopeMetrics::default(),
            gain: 1.0,
        }
    }

    /// Process a video frame.
    ///
    /// # Errors
    ///
    /// Returns an error if frame processing fails.
    pub fn process(&mut self, frame: &[u8], width: u32, height: u32) -> MonitorResult<()> {
        self.update_metrics(frame, width, height);
        Ok(())
    }

    /// Set vectorscope gain.
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(0.5, 4.0);
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &VectorscopeMetrics {
        &self.metrics
    }

    /// Reset monitor state.
    pub fn reset(&mut self) {
        self.metrics = VectorscopeMetrics::default();
    }

    fn update_metrics(&mut self, frame: &[u8], width: u32, height: u32) {
        let mut total_saturation = 0.0f32;
        let mut max_saturation = 0.0f32;
        let mut gamut_violations = 0u64;
        let mut pixel_count = 0u32;

        // Hue histogram for dominant hue calculation
        let mut hue_bins = vec![0u32; 360];

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]) / 255.0;
                    let g = f32::from(frame[idx + 1]) / 255.0;
                    let b = f32::from(frame[idx + 2]) / 255.0;

                    let max_c = r.max(g).max(b);
                    let min_c = r.min(g).min(b);
                    let delta = max_c - min_c;

                    let saturation = if max_c > 0.0 { delta / max_c } else { 0.0 };

                    total_saturation += saturation;
                    max_saturation = max_saturation.max(saturation);

                    // Calculate hue
                    if delta > 0.0 {
                        let hue = if max_c == r {
                            60.0 * (((g - b) / delta) % 6.0)
                        } else if max_c == g {
                            60.0 * (((b - r) / delta) + 2.0)
                        } else {
                            60.0 * (((r - g) / delta) + 4.0)
                        };

                        let hue_normalized = if hue < 0.0 { hue + 360.0 } else { hue };

                        #[allow(clippy::cast_possible_truncation)]
                        #[allow(clippy::cast_sign_loss)]
                        let hue_bin = (hue_normalized.clamp(0.0, 359.9)) as usize;
                        if hue_bin < 360 {
                            hue_bins[hue_bin] += 1;
                        }
                    }

                    // Check gamut
                    if r < 16.0 / 255.0
                        || r > 235.0 / 255.0
                        || g < 16.0 / 255.0
                        || g > 235.0 / 255.0
                        || b < 16.0 / 255.0
                        || b > 235.0 / 255.0
                    {
                        gamut_violations += 1;
                    }

                    pixel_count += 1;
                }
            }
        }

        self.metrics.avg_saturation = if pixel_count > 0 {
            total_saturation / pixel_count as f32
        } else {
            0.0
        };
        self.metrics.max_saturation = max_saturation;
        self.metrics.gamut_violations = gamut_violations;

        // Find dominant hue
        if let Some((hue, _)) = hue_bins.iter().enumerate().max_by_key(|(_, &count)| count) {
            self.metrics.dominant_hue = hue as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorscope_monitor() {
        let config = ScopeConfig::default();
        let mut monitor = VectorscopeMonitor::new(config);

        let frame = vec![128u8; 1920 * 1080 * 3];
        assert!(monitor.process(&frame, 1920, 1080).is_ok());

        let metrics = monitor.metrics();
        assert!(metrics.max_saturation >= 0.0);
    }

    #[test]
    fn test_vectorscope_gain() {
        let config = ScopeConfig::default();
        let mut monitor = VectorscopeMonitor::new(config);

        monitor.set_gain(2.0);
        assert_eq!(monitor.gain, 2.0);

        monitor.set_gain(10.0); // Should clamp to 4.0
        assert_eq!(monitor.gain, 4.0);
    }
}
