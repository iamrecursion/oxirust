#![allow(dead_code)]
//! Flicker detection for video content.
//!
//! This module detects luminance flicker in video frames, which can be caused by:
//!
//! - **Lighting frequency mismatch** - 50 Hz lights recorded at 60 fps (or vice versa)
//! - **LED/PWM artifacts** - LED screens or displays captured at incompatible rates
//! - **Encoding artifacts** - Unstable quantization causing brightness oscillation
//! - **Strobe effects** - Intentional or accidental rapid brightness changes
//!
//! The detector tracks per-frame average luminance and analyzes the temporal signal
//! for periodic oscillations using autocorrelation and peak detection.

use std::collections::VecDeque;

/// Configuration for flicker detection.
#[derive(Debug, Clone)]
pub struct FlickerConfig {
    /// Minimum luminance change (0.0-1.0) to consider significant.
    pub min_delta: f64,
    /// Maximum number of frames to keep in the analysis buffer.
    pub buffer_size: usize,
    /// Threshold for autocorrelation peak to confirm periodic flicker (0.0-1.0).
    pub autocorr_threshold: f64,
    /// Minimum flicker frequency in Hz to detect.
    pub min_freq_hz: f64,
    /// Maximum flicker frequency in Hz to detect.
    pub max_freq_hz: f64,
}

impl Default for FlickerConfig {
    fn default() -> Self {
        Self {
            min_delta: 0.005,
            buffer_size: 300,
            autocorr_threshold: 0.3,
            min_freq_hz: 5.0,
            max_freq_hz: 120.0,
        }
    }
}

/// A detected flicker event.
#[derive(Debug, Clone)]
pub struct FlickerEvent {
    /// Frame index where the flicker starts.
    pub start_frame: usize,
    /// Frame index where the flicker ends.
    pub end_frame: usize,
    /// Estimated flicker frequency in Hz.
    pub frequency_hz: f64,
    /// Severity of the flicker (0.0 = imperceptible, 1.0 = severe).
    pub severity: f64,
    /// Average luminance amplitude of the oscillation (0-255 scale).
    pub amplitude: f64,
}

/// Per-frame luminance measurement.
#[derive(Debug, Clone, Copy)]
pub struct FrameLuminance {
    /// Frame index.
    pub frame_index: usize,
    /// Average luminance (0.0-255.0).
    pub avg_luminance: f64,
    /// Luminance standard deviation.
    pub std_luminance: f64,
}

/// Flicker detector that processes video frames sequentially.
pub struct FlickerDetector {
    /// Configuration.
    config: FlickerConfig,
    /// Frame rate in fps.
    fps: f64,
    /// Buffer of recent luminance measurements.
    luminance_buffer: VecDeque<f64>,
    /// Frame indices corresponding to the luminance buffer.
    frame_indices: VecDeque<usize>,
    /// All detected flicker events.
    events: Vec<FlickerEvent>,
    /// Total frames processed.
    frame_count: usize,
    /// Cumulative luminance sum (for global stats).
    lum_sum: f64,
    /// Cumulative luminance squared sum.
    lum_sum_sq: f64,
}

impl FlickerDetector {
    /// Create a new flicker detector.
    ///
    /// # Arguments
    /// * `fps` - Frame rate of the video.
    pub fn new(fps: f64) -> Self {
        Self::with_config(fps, FlickerConfig::default())
    }

    /// Create a new flicker detector with custom configuration.
    pub fn with_config(fps: f64, config: FlickerConfig) -> Self {
        Self {
            luminance_buffer: VecDeque::with_capacity(config.buffer_size),
            frame_indices: VecDeque::with_capacity(config.buffer_size),
            config,
            fps: fps.max(1.0),
            events: Vec::new(),
            frame_count: 0,
            lum_sum: 0.0,
            lum_sum_sq: 0.0,
        }
    }

    /// Compute average luminance of a Y plane.
    #[allow(clippy::cast_precision_loss)]
    fn compute_avg_luminance(y_plane: &[u8]) -> f64 {
        if y_plane.is_empty() {
            return 0.0;
        }
        let sum: u64 = y_plane.iter().map(|&v| u64::from(v)).sum();
        sum as f64 / y_plane.len() as f64
    }

    /// Compute luminance standard deviation.
    #[allow(clippy::cast_precision_loss)]
    fn compute_std_luminance(y_plane: &[u8], mean: f64) -> f64 {
        if y_plane.len() < 2 {
            return 0.0;
        }
        let variance: f64 = y_plane
            .iter()
            .map(|&v| {
                let diff = f64::from(v) - mean;
                diff * diff
            })
            .sum::<f64>()
            / y_plane.len() as f64;
        variance.sqrt()
    }

    /// Process a video frame (Y plane only).
    ///
    /// Returns the luminance measurement for this frame.
    pub fn process_frame(&mut self, y_plane: &[u8], frame_index: usize) -> FrameLuminance {
        let avg = Self::compute_avg_luminance(y_plane);
        let std_dev = Self::compute_std_luminance(y_plane, avg);

        // Update buffer
        if self.luminance_buffer.len() >= self.config.buffer_size {
            self.luminance_buffer.pop_front();
            self.frame_indices.pop_front();
        }
        self.luminance_buffer.push_back(avg);
        self.frame_indices.push_back(frame_index);

        // Update running stats
        self.lum_sum += avg;
        self.lum_sum_sq += avg * avg;
        self.frame_count += 1;

        // Periodically check for flicker when buffer is full enough
        if self.luminance_buffer.len() >= 30 {
            self.analyze_buffer();
        }

        FrameLuminance {
            frame_index,
            avg_luminance: avg,
            std_luminance: std_dev,
        }
    }

    /// Analyze the luminance buffer for periodic flicker patterns.
    #[allow(clippy::cast_precision_loss)]
    fn analyze_buffer(&mut self) {
        let buf: Vec<f64> = self.luminance_buffer.iter().copied().collect();
        let n = buf.len();
        if n < 10 {
            return;
        }

        // Remove DC offset (mean)
        let mean: f64 = buf.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = buf.iter().map(|&v| v - mean).collect();

        // Compute autocorrelation for different lags
        let energy: f64 = centered.iter().map(|v| v * v).sum();
        if energy < 1e-10 {
            return;
        }

        // Lag range based on frequency bounds
        let min_lag = (self.fps / self.config.max_freq_hz).ceil() as usize;
        let max_lag = ((self.fps / self.config.min_freq_hz).floor() as usize).min(n / 2);
        if min_lag >= max_lag || max_lag >= n {
            return;
        }

        let mut best_lag = 0;
        let mut best_corr = 0.0_f64;

        for lag in min_lag..=max_lag {
            let mut corr = 0.0;
            for i in 0..n - lag {
                corr += centered[i] * centered[i + lag];
            }
            let normalized = corr / energy;
            if normalized > best_corr {
                best_corr = normalized;
                best_lag = lag;
            }
        }

        if best_corr >= self.config.autocorr_threshold && best_lag > 0 {
            let frequency = self.fps / best_lag as f64;

            // Compute amplitude (peak-to-peak of oscillation)
            let max_val = centered.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_val = centered.iter().copied().fold(f64::INFINITY, f64::min);
            let amplitude = max_val - min_val;

            if amplitude / 255.0 >= self.config.min_delta {
                let severity = (best_corr * (amplitude / 50.0)).min(1.0);

                let start = self.frame_indices.front().copied().unwrap_or(0);
                let end = self.frame_indices.back().copied().unwrap_or(0);

                // Don't add duplicate events for the same region
                let dominated = self.events.last().is_some_and(|e| e.end_frame >= start);
                if !dominated {
                    self.events.push(FlickerEvent {
                        start_frame: start,
                        end_frame: end,
                        frequency_hz: frequency,
                        severity,
                        amplitude,
                    });
                }
            }
        }
    }

    /// Finalize detection and return all found events.
    pub fn finalize(self) -> Vec<FlickerEvent> {
        self.events
    }

    /// Get the number of frames processed so far.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get the global average luminance across all frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn global_avg_luminance(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.lum_sum / self.frame_count as f64
    }

    /// Get the global luminance standard deviation across all frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn global_std_luminance(&self) -> f64 {
        if self.frame_count < 2 {
            return 0.0;
        }
        let n = self.frame_count as f64;
        let mean = self.lum_sum / n;
        let variance = (self.lum_sum_sq / n - mean * mean).max(0.0);
        variance.sqrt()
    }
}

/// Compute the flicker score for a pair of consecutive frames.
///
/// Returns the absolute difference in average luminance, normalized to 0.0-1.0.
#[allow(clippy::cast_precision_loss)]
pub fn frame_pair_flicker(y1: &[u8], y2: &[u8]) -> f64 {
    let avg1 = FlickerDetector::compute_avg_luminance(y1);
    let avg2 = FlickerDetector::compute_avg_luminance(y2);
    (avg1 - avg2).abs() / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_frame(size: usize, value: u8) -> Vec<u8> {
        vec![value; size]
    }

    fn make_sine_frames(count: usize, size: usize, period: usize, amplitude: f64) -> Vec<Vec<u8>> {
        let mut frames = Vec::with_capacity(count);
        let base = 128.0;
        for i in 0..count {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
            let val = (base + amplitude * phase.sin()).clamp(0.0, 255.0) as u8;
            frames.push(vec![val; size]);
        }
        frames
    }

    #[test]
    fn test_flat_frames_no_flicker() {
        let mut detector = FlickerDetector::new(30.0);
        for i in 0..60 {
            let frame = make_flat_frame(1024, 128);
            detector.process_frame(&frame, i);
        }
        let events = detector.finalize();
        assert!(events.is_empty(), "flat frames should produce no flicker");
    }

    #[test]
    fn test_compute_avg_luminance() {
        let frame = vec![100u8; 100];
        let avg = FlickerDetector::compute_avg_luminance(&frame);
        assert!((avg - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_avg_luminance_empty() {
        let avg = FlickerDetector::compute_avg_luminance(&[]);
        assert_eq!(avg, 0.0);
    }

    #[test]
    fn test_compute_std_luminance() {
        let frame = vec![100u8; 100];
        let std = FlickerDetector::compute_std_luminance(&frame, 100.0);
        assert!(std < 0.001, "uniform frame should have zero std");
    }

    #[test]
    fn test_compute_std_luminance_varied() {
        let mut frame = vec![0u8; 100];
        for i in 0..100 {
            frame[i] = if i < 50 { 0 } else { 200 };
        }
        let mean = FlickerDetector::compute_avg_luminance(&frame);
        let std = FlickerDetector::compute_std_luminance(&frame, mean);
        assert!(
            std > 50.0,
            "bimodal distribution should have high std: {}",
            std
        );
    }

    #[test]
    fn test_oscillating_frames_detected() {
        let config = FlickerConfig {
            min_delta: 0.001,
            buffer_size: 120,
            autocorr_threshold: 0.2,
            min_freq_hz: 3.0,
            max_freq_hz: 30.0,
        };
        let mut detector = FlickerDetector::with_config(30.0, config);
        let frames = make_sine_frames(120, 1024, 6, 40.0); // 5 Hz flicker at 30fps

        for (i, frame) in frames.iter().enumerate() {
            detector.process_frame(frame, i);
        }
        let events = detector.finalize();
        assert!(
            !events.is_empty(),
            "oscillating frames should be detected as flicker"
        );
        if let Some(ev) = events.first() {
            assert!(
                ev.frequency_hz > 3.0 && ev.frequency_hz < 8.0,
                "detected freq should be near 5 Hz, got {}",
                ev.frequency_hz
            );
        }
    }

    #[test]
    fn test_frame_pair_flicker_identical() {
        let f1 = make_flat_frame(100, 128);
        let f2 = make_flat_frame(100, 128);
        let score = frame_pair_flicker(&f1, &f2);
        assert!(score < 0.001);
    }

    #[test]
    fn test_frame_pair_flicker_different() {
        let f1 = make_flat_frame(100, 0);
        let f2 = make_flat_frame(100, 255);
        let score = frame_pair_flicker(&f1, &f2);
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_global_stats() {
        let mut detector = FlickerDetector::new(30.0);
        for i in 0..10 {
            let frame = make_flat_frame(100, 100);
            detector.process_frame(&frame, i);
        }
        assert_eq!(detector.frame_count(), 10);
        assert!((detector.global_avg_luminance() - 100.0).abs() < 0.01);
        assert!(detector.global_std_luminance() < 0.01);
    }

    #[test]
    fn test_global_stats_empty() {
        let detector = FlickerDetector::new(30.0);
        assert_eq!(detector.global_avg_luminance(), 0.0);
        assert_eq!(detector.global_std_luminance(), 0.0);
    }

    #[test]
    fn test_buffer_eviction() {
        let config = FlickerConfig {
            buffer_size: 10,
            ..Default::default()
        };
        let mut detector = FlickerDetector::with_config(30.0, config);
        for i in 0..20 {
            let frame = make_flat_frame(64, 128);
            detector.process_frame(&frame, i);
        }
        // Buffer should only contain last 10 entries
        assert_eq!(detector.luminance_buffer.len(), 10);
        assert_eq!(detector.frame_count(), 20);
    }
}
