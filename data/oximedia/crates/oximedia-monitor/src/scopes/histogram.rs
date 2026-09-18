//! Histogram monitor implementation.

use crate::MonitorResult;
use oximedia_scopes::ScopeConfig;
use serde::{Deserialize, Serialize};

/// Histogram monitoring metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HistogramMetrics {
    /// Luma histogram (256 bins).
    pub luma_histogram: Vec<u32>,

    /// Red histogram (256 bins).
    pub red_histogram: Vec<u32>,

    /// Green histogram (256 bins).
    pub green_histogram: Vec<u32>,

    /// Blue histogram (256 bins).
    pub blue_histogram: Vec<u32>,

    /// Luma mean.
    pub luma_mean: f32,

    /// Luma standard deviation.
    pub luma_stddev: f32,
}

/// Histogram monitor.
pub struct HistogramMonitor {
    config: ScopeConfig,
    metrics: HistogramMetrics,
}

impl HistogramMonitor {
    /// Create a new histogram monitor.
    #[must_use]
    pub fn new(config: ScopeConfig) -> Self {
        Self {
            config,
            metrics: HistogramMetrics::default(),
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
    pub const fn metrics(&self) -> &HistogramMetrics {
        &self.metrics
    }

    /// Reset monitor state.
    pub fn reset(&mut self) {
        self.metrics = HistogramMetrics::default();
    }

    fn update_metrics(&mut self, frame: &[u8], width: u32, height: u32) {
        let mut luma_hist = vec![0u32; 256];
        let mut red_hist = vec![0u32; 256];
        let mut green_hist = vec![0u32; 256];
        let mut blue_hist = vec![0u32; 256];

        let mut luma_sum = 0.0f32;
        let mut luma_sq_sum = 0.0f32;
        let mut pixel_count = 0u32;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = frame[idx];
                    let g = frame[idx + 1];
                    let b = frame[idx + 2];

                    // Update RGB histograms
                    red_hist[r as usize] += 1;
                    green_hist[g as usize] += 1;
                    blue_hist[b as usize] += 1;

                    // Calculate luma
                    let luma =
                        0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b);

                    #[allow(clippy::cast_possible_truncation)]
                    #[allow(clippy::cast_sign_loss)]
                    let luma_bin = luma.clamp(0.0, 255.0) as usize;
                    if luma_bin < 256 {
                        luma_hist[luma_bin] += 1;
                    }

                    luma_sum += luma;
                    luma_sq_sum += luma * luma;
                    pixel_count += 1;
                }
            }
        }

        self.metrics.luma_histogram = luma_hist;
        self.metrics.red_histogram = red_hist;
        self.metrics.green_histogram = green_hist;
        self.metrics.blue_histogram = blue_hist;

        if pixel_count > 0 {
            let mean = luma_sum / pixel_count as f32;
            let variance = (luma_sq_sum / pixel_count as f32) - (mean * mean);
            self.metrics.luma_mean = mean / 255.0;
            self.metrics.luma_stddev = variance.sqrt() / 255.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_monitor() {
        let config = ScopeConfig::default();
        let mut monitor = HistogramMonitor::new(config);

        let frame = vec![128u8; 1920 * 1080 * 3];
        assert!(monitor.process(&frame, 1920, 1080).is_ok());

        let metrics = monitor.metrics();
        assert_eq!(metrics.luma_histogram.len(), 256);
    }

    #[test]
    fn test_histogram_statistics() {
        let config = ScopeConfig::default();
        let mut monitor = HistogramMonitor::new(config);

        // All gray pixels
        let frame = vec![128u8; 100 * 100 * 3];
        assert!(monitor.process(&frame, 100, 100).is_ok());

        let metrics = monitor.metrics();
        assert!(metrics.luma_mean > 0.0);
    }
}
