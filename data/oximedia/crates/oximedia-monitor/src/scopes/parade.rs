//! RGB parade monitor implementation.

use crate::MonitorResult;
use oximedia_scopes::ScopeConfig;
use serde::{Deserialize, Serialize};

/// Parade monitoring metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParadeMetrics {
    /// Red channel min/max.
    pub red_range: (f32, f32),

    /// Green channel min/max.
    pub green_range: (f32, f32),

    /// Blue channel min/max.
    pub blue_range: (f32, f32),

    /// Color balance deviation.
    pub color_balance: f32,
}

/// RGB parade monitor.
pub struct ParadeMonitor {
    config: ScopeConfig,
    metrics: ParadeMetrics,
}

impl ParadeMonitor {
    /// Create a new parade monitor.
    #[must_use]
    pub fn new(config: ScopeConfig) -> Self {
        Self {
            config,
            metrics: ParadeMetrics::default(),
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

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &ParadeMetrics {
        &self.metrics
    }

    /// Reset monitor state.
    pub fn reset(&mut self) {
        self.metrics = ParadeMetrics::default();
    }

    fn update_metrics(&mut self, frame: &[u8], width: u32, height: u32) {
        let mut r_min = 255.0f32;
        let mut r_max = 0.0f32;
        let mut g_min = 255.0f32;
        let mut g_max = 0.0f32;
        let mut b_min = 255.0f32;
        let mut b_max = 0.0f32;

        let mut r_sum = 0.0f32;
        let mut g_sum = 0.0f32;
        let mut b_sum = 0.0f32;
        let mut pixel_count = 0u32;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    r_min = r_min.min(r);
                    r_max = r_max.max(r);
                    g_min = g_min.min(g);
                    g_max = g_max.max(g);
                    b_min = b_min.min(b);
                    b_max = b_max.max(b);

                    r_sum += r;
                    g_sum += g;
                    b_sum += b;
                    pixel_count += 1;
                }
            }
        }

        self.metrics.red_range = (r_min / 255.0, r_max / 255.0);
        self.metrics.green_range = (g_min / 255.0, g_max / 255.0);
        self.metrics.blue_range = (b_min / 255.0, b_max / 255.0);

        // Calculate color balance (deviation from neutral)
        if pixel_count > 0 {
            let r_avg = r_sum / pixel_count as f32;
            let g_avg = g_sum / pixel_count as f32;
            let b_avg = b_sum / pixel_count as f32;

            let avg_all = (r_avg + g_avg + b_avg) / 3.0;
            let r_dev = (r_avg - avg_all).abs();
            let g_dev = (g_avg - avg_all).abs();
            let b_dev = (b_avg - avg_all).abs();

            self.metrics.color_balance = (r_dev + g_dev + b_dev) / (3.0 * 255.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parade_monitor() {
        let config = ScopeConfig::default();
        let mut monitor = ParadeMonitor::new(config);

        let frame = vec![128u8; 1920 * 1080 * 3];
        assert!(monitor.process(&frame, 1920, 1080).is_ok());

        let metrics = monitor.metrics();
        assert!(metrics.color_balance >= 0.0);
    }
}
