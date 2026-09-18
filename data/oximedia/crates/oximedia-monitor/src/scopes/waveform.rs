//! Waveform monitor implementation.

use crate::MonitorResult;
use oximedia_scopes::ScopeConfig;
use serde::{Deserialize, Serialize};

/// Waveform display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveformMode {
    /// Luma waveform (Y channel only).
    Luma,

    /// RGB overlay waveform.
    RgbOverlay,

    /// RGB parade waveform.
    RgbParade,

    /// `YCbCr` waveform.
    Ycbcr,
}

/// Waveform monitoring metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WaveformMetrics {
    /// Peak luma value.
    pub peak_luma: f32,

    /// Minimum luma value.
    pub min_luma: f32,

    /// Average luma value.
    pub avg_luma: f32,

    /// Luma distribution histogram (256 bins).
    pub luma_histogram: Vec<u32>,
}

/// Waveform monitor.
pub struct WaveformMonitor {
    config: ScopeConfig,
    mode: WaveformMode,
    metrics: WaveformMetrics,
}

impl WaveformMonitor {
    /// Create a new waveform monitor.
    #[must_use]
    pub fn new(config: ScopeConfig) -> Self {
        Self {
            config,
            mode: WaveformMode::Luma,
            metrics: WaveformMetrics::default(),
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

    /// Set waveform mode.
    pub fn set_mode(&mut self, mode: WaveformMode) {
        self.mode = mode;
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &WaveformMetrics {
        &self.metrics
    }

    /// Reset monitor state.
    pub fn reset(&mut self) {
        self.metrics = WaveformMetrics::default();
    }

    fn update_metrics(&mut self, frame: &[u8], width: u32, height: u32) {
        let mut histogram = vec![0u32; 256];
        let mut sum = 0.0f32;
        let mut min_val = 255.0f32;
        let mut max_val = 0.0f32;
        let mut pixel_count = 0u32;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    // Rec.709 luma
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    sum += luma;
                    min_val = min_val.min(luma);
                    max_val = max_val.max(luma);

                    #[allow(clippy::cast_possible_truncation)]
                    #[allow(clippy::cast_sign_loss)]
                    let bin = luma.clamp(0.0, 255.0) as usize;
                    if bin < 256 {
                        histogram[bin] += 1;
                    }

                    pixel_count += 1;
                }
            }
        }

        self.metrics.min_luma = min_val / 255.0;
        self.metrics.peak_luma = max_val / 255.0;
        self.metrics.avg_luma = if pixel_count > 0 {
            sum / (pixel_count as f32 * 255.0)
        } else {
            0.0
        };
        self.metrics.luma_histogram = histogram;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_monitor() {
        let config = ScopeConfig::default();
        let mut monitor = WaveformMonitor::new(config);

        // Test with a simple gradient frame
        let frame = vec![128u8; 1920 * 1080 * 3];
        assert!(monitor.process(&frame, 1920, 1080).is_ok());

        let metrics = monitor.metrics();
        assert!(metrics.avg_luma > 0.0);
    }

    #[test]
    fn test_waveform_mode() {
        let config = ScopeConfig::default();
        let mut monitor = WaveformMonitor::new(config);

        monitor.set_mode(WaveformMode::RgbParade);
        assert_eq!(monitor.mode, WaveformMode::RgbParade);
    }
}
