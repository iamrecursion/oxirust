//! Video scope monitoring.
//!
//! This module provides wrappers around the oximedia-scopes crate for broadcast monitoring.

pub mod focus;
pub mod histogram;
pub mod parade;
pub mod vectorscope;
pub mod waveform;

use crate::{MonitorError, MonitorResult};
use oximedia_scopes::{ScopeConfig, ScopeData, ScopeType, VideoScopes};
use serde::{Deserialize, Serialize};

pub use focus::FocusAssist;
pub use histogram::HistogramMonitor;
pub use parade::ParadeMonitor;
pub use vectorscope::VectorscopeMonitor;
pub use waveform::WaveformMonitor;

/// Scope monitoring metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeMetrics {
    /// Luma range (min, max).
    pub luma_range: (f32, f32),

    /// Chroma range (min, max).
    pub chroma_range: (f32, f32),

    /// Out-of-gamut pixel count.
    pub out_of_gamut_pixels: u64,

    /// Black level violations.
    pub black_level_violations: u64,

    /// White level violations.
    pub white_level_violations: u64,

    /// Vectorscope saturation (0.0-1.0).
    pub saturation: f32,

    /// Histogram mean luminance.
    pub mean_luminance: f32,
}

/// Scope monitor integrating all video scopes.
pub struct ScopeMonitor {
    config: ScopeConfig,
    scopes: VideoScopes,
    waveform_monitor: WaveformMonitor,
    vectorscope_monitor: VectorscopeMonitor,
    histogram_monitor: HistogramMonitor,
    parade_monitor: ParadeMonitor,
    focus_assist: FocusAssist,
    metrics: ScopeMetrics,
}

impl ScopeMonitor {
    /// Create a new scope monitor.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(config: ScopeConfig) -> MonitorResult<Self> {
        Ok(Self {
            scopes: VideoScopes::new(config.clone()),
            waveform_monitor: WaveformMonitor::new(config.clone()),
            vectorscope_monitor: VectorscopeMonitor::new(config.clone()),
            histogram_monitor: HistogramMonitor::new(config.clone()),
            parade_monitor: ParadeMonitor::new(config.clone()),
            focus_assist: FocusAssist::new(),
            config,
            metrics: ScopeMetrics::default(),
        })
    }

    /// Process a video frame.
    ///
    /// # Errors
    ///
    /// Returns an error if frame processing fails.
    pub fn process_frame(&mut self, frame: &[u8], width: u32, height: u32) -> MonitorResult<()> {
        // Update individual scope monitors
        self.waveform_monitor.process(frame, width, height)?;
        self.vectorscope_monitor.process(frame, width, height)?;
        self.histogram_monitor.process(frame, width, height)?;
        self.parade_monitor.process(frame, width, height)?;

        // Update metrics
        self.update_metrics(frame, width, height)?;

        Ok(())
    }

    /// Generate waveform scope.
    ///
    /// # Errors
    ///
    /// Returns an error if scope generation fails.
    pub fn generate_waveform(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> MonitorResult<ScopeData> {
        self.scopes
            .analyze(frame, width, height, ScopeType::WaveformLuma)
            .map_err(|e| MonitorError::ScopeError(e.to_string()))
    }

    /// Generate vectorscope.
    ///
    /// # Errors
    ///
    /// Returns an error if scope generation fails.
    pub fn generate_vectorscope(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> MonitorResult<ScopeData> {
        self.scopes
            .analyze(frame, width, height, ScopeType::Vectorscope)
            .map_err(|e| MonitorError::ScopeError(e.to_string()))
    }

    /// Generate histogram.
    ///
    /// # Errors
    ///
    /// Returns an error if scope generation fails.
    pub fn generate_histogram(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> MonitorResult<ScopeData> {
        self.scopes
            .analyze(frame, width, height, ScopeType::HistogramLuma)
            .map_err(|e| MonitorError::ScopeError(e.to_string()))
    }

    /// Generate RGB parade.
    ///
    /// # Errors
    ///
    /// Returns an error if scope generation fails.
    pub fn generate_parade(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> MonitorResult<ScopeData> {
        self.scopes
            .analyze(frame, width, height, ScopeType::ParadeRgb)
            .map_err(|e| MonitorError::ScopeError(e.to_string()))
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &ScopeMetrics {
        &self.metrics
    }

    /// Reset scope monitor.
    pub fn reset(&mut self) {
        self.metrics = ScopeMetrics::default();
        self.waveform_monitor.reset();
        self.vectorscope_monitor.reset();
        self.histogram_monitor.reset();
        self.parade_monitor.reset();
    }

    /// Update metrics from frame data.
    fn update_metrics(&mut self, frame: &[u8], width: u32, height: u32) -> MonitorResult<()> {
        // Calculate luma and chroma ranges
        let (luma_min, luma_max) = self.calculate_luma_range(frame, width, height);
        self.metrics.luma_range = (luma_min, luma_max);

        // Calculate chroma range
        let (chroma_min, chroma_max) = self.calculate_chroma_range(frame, width, height);
        self.metrics.chroma_range = (chroma_min, chroma_max);

        // Count violations
        self.metrics.black_level_violations = self.count_black_violations(frame, width, height);
        self.metrics.white_level_violations = self.count_white_violations(frame, width, height);
        self.metrics.out_of_gamut_pixels = self.count_gamut_violations(frame, width, height);

        // Calculate saturation
        self.metrics.saturation = self.calculate_saturation(frame, width, height);

        // Calculate mean luminance
        self.metrics.mean_luminance = self.calculate_mean_luminance(frame, width, height);

        Ok(())
    }

    fn calculate_luma_range(&self, frame: &[u8], width: u32, height: u32) -> (f32, f32) {
        if frame.len() < (width * height * 3) as usize {
            return (0.0, 0.0);
        }

        let mut min_luma = 255.0f32;
        let mut max_luma = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    // Rec.709 luma coefficients
                    #[allow(clippy::cast_possible_truncation)]
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    min_luma = min_luma.min(luma);
                    max_luma = max_luma.max(luma);
                }
            }
        }

        (min_luma / 255.0, max_luma / 255.0)
    }

    fn calculate_chroma_range(&self, frame: &[u8], width: u32, height: u32) -> (f32, f32) {
        if frame.len() < (width * height * 3) as usize {
            return (0.0, 0.0);
        }

        let mut min_chroma = f32::MAX;
        let mut max_chroma = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    let cb = (b - luma) * 0.5;
                    let cr = (r - luma) * 0.5;
                    let chroma = (cb * cb + cr * cr).sqrt();

                    min_chroma = min_chroma.min(chroma);
                    max_chroma = max_chroma.max(chroma);
                }
            }
        }

        (min_chroma / 255.0, max_chroma / 255.0)
    }

    fn count_black_violations(&self, frame: &[u8], width: u32, height: u32) -> u64 {
        let mut count = 0u64;
        let black_threshold = 16.0; // Broadcast black level (16 in 8-bit)

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    if luma < black_threshold {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn count_white_violations(&self, frame: &[u8], width: u32, height: u32) -> u64 {
        let mut count = 0u64;
        let white_threshold = 235.0; // Broadcast white level (235 in 8-bit)

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                    if luma > white_threshold {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn count_gamut_violations(&self, frame: &[u8], width: u32, height: u32) -> u64 {
        let mut count = 0u64;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    // Simple gamut check - RGB values should be in legal range
                    if r < 16.0 || r > 235.0 || g < 16.0 || g > 235.0 || b < 16.0 || b > 235.0 {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    fn calculate_saturation(&self, frame: &[u8], width: u32, height: u32) -> f32 {
        if frame.len() < (width * height * 3) as usize {
            return 0.0;
        }

        let mut total_saturation = 0.0f32;
        let pixel_count = width * height;

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
                }
            }
        }

        total_saturation / pixel_count as f32
    }

    fn calculate_mean_luminance(&self, frame: &[u8], width: u32, height: u32) -> f32 {
        if frame.len() < (width * height * 3) as usize {
            return 0.0;
        }

        let mut total_luma = 0.0f32;
        let pixel_count = width * height;

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < frame.len() {
                    let r = f32::from(frame[idx]);
                    let g = f32::from(frame[idx + 1]);
                    let b = f32::from(frame[idx + 2]);

                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    total_luma += luma;
                }
            }
        }

        total_luma / (pixel_count as f32 * 255.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_monitor_creation() {
        let config = ScopeConfig::default();
        let result = ScopeMonitor::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scope_metrics_default() {
        let metrics = ScopeMetrics::default();
        assert_eq!(metrics.luma_range, (0.0, 0.0));
        assert_eq!(metrics.out_of_gamut_pixels, 0);
    }
}
