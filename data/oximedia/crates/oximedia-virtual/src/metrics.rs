//! Performance metrics and monitoring for virtual production
//!
//! Provides real-time performance monitoring, latency tracking,
//! and quality metrics for virtual production systems.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average frame time in milliseconds
    pub avg_frame_time_ms: f64,
    /// Minimum frame time in milliseconds
    pub min_frame_time_ms: f64,
    /// Maximum frame time in milliseconds
    pub max_frame_time_ms: f64,
    /// Current FPS
    pub current_fps: f64,
    /// Average FPS
    pub avg_fps: f64,
    /// Frame drops
    pub frame_drops: u64,
    /// Total frames processed
    pub total_frames: u64,
}

impl PerformanceMetrics {
    /// Create new metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            avg_frame_time_ms: 0.0,
            min_frame_time_ms: 0.0,
            max_frame_time_ms: 0.0,
            current_fps: 0.0,
            avg_fps: 0.0,
            frame_drops: 0,
            total_frames: 0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Camera tracking latency in microseconds
    pub tracking_latency_us: u64,
    /// Rendering latency in microseconds
    pub render_latency_us: u64,
    /// Compositing latency in microseconds
    pub composite_latency_us: u64,
    /// Total end-to-end latency in microseconds
    pub total_latency_us: u64,
}

impl LatencyMetrics {
    /// Create new latency metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracking_latency_us: 0,
            render_latency_us: 0,
            composite_latency_us: 0,
            total_latency_us: 0,
        }
    }

    /// Check if latency is acceptable (< 20ms total)
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.total_latency_us < 20_000
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Tracking confidence (0.0 - 1.0)
    pub tracking_confidence: f32,
    /// Color accuracy score (0.0 - 1.0)
    pub color_accuracy: f32,
    /// Sync accuracy in microseconds
    pub sync_accuracy_us: u64,
    /// LED wall brightness uniformity (0.0 - 1.0)
    pub brightness_uniformity: f32,
}

impl QualityMetrics {
    /// Create new quality metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracking_confidence: 1.0,
            color_accuracy: 1.0,
            sync_accuracy_us: 0,
            brightness_uniformity: 1.0,
        }
    }

    /// Overall quality score
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        (self.tracking_confidence + self.color_accuracy + self.brightness_uniformity) / 3.0
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics collector
pub struct MetricsCollector {
    frame_times: VecDeque<Duration>,
    window_size: usize,
    last_frame_time: Option<Instant>,
    performance: PerformanceMetrics,
    latency: LatencyMetrics,
    quality: QualityMetrics,
}

impl MetricsCollector {
    /// Create new metrics collector
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(window_size),
            window_size,
            last_frame_time: None,
            performance: PerformanceMetrics::new(),
            latency: LatencyMetrics::new(),
            quality: QualityMetrics::new(),
        }
    }

    /// Record frame
    pub fn record_frame(&mut self) {
        let now = Instant::now();

        if let Some(last_time) = self.last_frame_time {
            let frame_time = now.duration_since(last_time);

            // Add to window
            self.frame_times.push_back(frame_time);
            if self.frame_times.len() > self.window_size {
                self.frame_times.pop_front();
            }

            // Update metrics
            self.update_performance_metrics();
        }

        self.last_frame_time = Some(now);
        self.performance.total_frames += 1;
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self) {
        if self.frame_times.is_empty() {
            return;
        }

        let total_time: Duration = self.frame_times.iter().sum();
        let count = self.frame_times.len();

        self.performance.avg_frame_time_ms = total_time.as_secs_f64() * 1000.0 / count as f64;
        self.performance.avg_fps = 1000.0 / self.performance.avg_frame_time_ms;

        self.performance.min_frame_time_ms = self
            .frame_times
            .iter()
            .min()
            .unwrap_or(&Duration::ZERO)
            .as_secs_f64()
            * 1000.0;

        self.performance.max_frame_time_ms = self
            .frame_times
            .iter()
            .max()
            .unwrap_or(&Duration::ZERO)
            .as_secs_f64()
            * 1000.0;

        if let Some(last) = self.frame_times.back() {
            self.performance.current_fps = 1000.0 / (last.as_secs_f64() * 1000.0);
        }
    }

    /// Record tracking latency
    pub fn record_tracking_latency(&mut self, latency: Duration) {
        self.latency.tracking_latency_us = latency.as_micros() as u64;
    }

    /// Record render latency
    pub fn record_render_latency(&mut self, latency: Duration) {
        self.latency.render_latency_us = latency.as_micros() as u64;
    }

    /// Record composite latency
    pub fn record_composite_latency(&mut self, latency: Duration) {
        self.latency.composite_latency_us = latency.as_micros() as u64;
    }

    /// Update total latency
    pub fn update_total_latency(&mut self) {
        self.latency.total_latency_us = self.latency.tracking_latency_us
            + self.latency.render_latency_us
            + self.latency.composite_latency_us;
    }

    /// Update quality metrics
    pub fn update_quality(&mut self, quality: QualityMetrics) {
        self.quality = quality;
    }

    /// Get performance metrics
    #[must_use]
    pub fn performance(&self) -> &PerformanceMetrics {
        &self.performance
    }

    /// Get latency metrics
    #[must_use]
    pub fn latency(&self) -> &LatencyMetrics {
        &self.latency
    }

    /// Get quality metrics
    #[must_use]
    pub fn quality(&self) -> &QualityMetrics {
        &self.quality
    }

    /// Reset metrics
    pub fn reset(&mut self) {
        self.frame_times.clear();
        self.last_frame_time = None;
        self.performance = PerformanceMetrics::new();
        self.latency = LatencyMetrics::new();
        self.quality = QualityMetrics::new();
    }

    /// Get report
    #[must_use]
    pub fn generate_report(&self) -> String {
        format!(
            "Performance Report\n\
             =================\n\
             FPS: {:.2} (avg: {:.2})\n\
             Frame Time: {:.2}ms (min: {:.2}ms, max: {:.2}ms)\n\
             Frame Drops: {}\n\
             Total Frames: {}\n\
             \n\
             Latency Report\n\
             ==============\n\
             Tracking: {}µs\n\
             Rendering: {}µs\n\
             Compositing: {}µs\n\
             Total: {}µs ({}ms)\n\
             Acceptable: {}\n\
             \n\
             Quality Report\n\
             ==============\n\
             Tracking Confidence: {:.2}%\n\
             Color Accuracy: {:.2}%\n\
             Sync Accuracy: {}µs\n\
             Brightness Uniformity: {:.2}%\n\
             Overall Score: {:.2}%\n",
            self.performance.current_fps,
            self.performance.avg_fps,
            self.performance.avg_frame_time_ms,
            self.performance.min_frame_time_ms,
            self.performance.max_frame_time_ms,
            self.performance.frame_drops,
            self.performance.total_frames,
            self.latency.tracking_latency_us,
            self.latency.render_latency_us,
            self.latency.composite_latency_us,
            self.latency.total_latency_us,
            self.latency.total_latency_us as f64 / 1000.0,
            self.latency.is_acceptable(),
            self.quality.tracking_confidence * 100.0,
            self.quality.color_accuracy * 100.0,
            self.quality.sync_accuracy_us,
            self.quality.brightness_uniformity * 100.0,
            self.quality.overall_score() * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new();
        assert_eq!(metrics.total_frames, 0);
    }

    #[test]
    fn test_latency_metrics() {
        let mut metrics = LatencyMetrics::new();
        assert!(metrics.is_acceptable());

        metrics.total_latency_us = 25_000;
        assert!(!metrics.is_acceptable());
    }

    #[test]
    fn test_quality_metrics() {
        let metrics = QualityMetrics::new();
        assert_eq!(metrics.overall_score(), 1.0);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new(60);

        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(16));
            collector.record_frame();
        }

        assert!(collector.performance().total_frames > 0);
        assert!(collector.performance().avg_fps > 0.0);
    }

    #[test]
    fn test_metrics_collector_latency() {
        let mut collector = MetricsCollector::new(60);

        collector.record_tracking_latency(Duration::from_millis(1));
        collector.record_render_latency(Duration::from_millis(5));
        collector.record_composite_latency(Duration::from_millis(2));
        collector.update_total_latency();

        assert_eq!(collector.latency().total_latency_us, 8000);
    }

    #[test]
    fn test_generate_report() {
        let collector = MetricsCollector::new(60);
        let report = collector.generate_report();
        assert!(report.contains("Performance Report"));
        assert!(report.contains("Latency Report"));
        assert!(report.contains("Quality Report"));
    }
}
