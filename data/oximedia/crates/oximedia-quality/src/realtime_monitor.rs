//! Real-time quality monitoring with running statistics.
//!
//! Provides [`RealtimeQualityMonitor`] which tracks quality metrics across a
//! stream of frames using configurable sliding windows, exponential moving
//! averages, and threshold-based alerting.
//!
//! # Example
//!
//! ```
//! use oximedia_quality::realtime_monitor::{RealtimeQualityMonitor, MonitorConfig};
//!
//! let config = MonitorConfig::new(30); // 30-frame sliding window
//! let mut monitor = RealtimeQualityMonitor::new(config);
//!
//! monitor.push("psnr", 38.5);
//! monitor.push("psnr", 37.2);
//! monitor.push("ssim", 0.96);
//!
//! let psnr_avg = monitor.running_mean("psnr");
//! assert!(psnr_avg.is_some());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the real-time quality monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Number of recent samples to keep in the sliding window per metric.
    pub window_size: usize,
    /// Smoothing factor for exponential moving average (0.0..1.0).
    /// Higher values give more weight to recent samples.
    pub ema_alpha: f64,
    /// Alert thresholds: metric name -> (min, max).
    /// A sample outside [min, max] triggers an alert.
    pub thresholds: HashMap<String, (f64, f64)>,
}

impl MonitorConfig {
    /// Creates a new configuration with the given window size.
    ///
    /// Uses default EMA alpha of 0.1 and no thresholds.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            ema_alpha: 0.1,
            thresholds: HashMap::new(),
        }
    }

    /// Sets the EMA smoothing factor.
    #[must_use]
    pub fn with_ema_alpha(mut self, alpha: f64) -> Self {
        self.ema_alpha = alpha.clamp(0.001, 1.0);
        self
    }

    /// Adds an alert threshold for a metric.
    #[must_use]
    pub fn with_threshold(mut self, metric: impl Into<String>, min: f64, max: f64) -> Self {
        self.thresholds.insert(metric.into(), (min, max));
        self
    }
}

/// Internal state for a single metric's sliding window.
#[derive(Debug, Clone)]
struct MetricWindow {
    /// Circular buffer of recent samples.
    samples: Vec<f64>,
    /// Write index into the circular buffer.
    write_idx: usize,
    /// Number of samples written (may exceed window_size; capped for stats).
    total_count: u64,
    /// Running sum for the window (maintained incrementally).
    running_sum: f64,
    /// Running sum-of-squares for the window (maintained incrementally).
    running_sum_sq: f64,
    /// Exponential moving average.
    ema: f64,
    /// Whether the EMA has been initialized.
    ema_initialized: bool,
}

impl MetricWindow {
    fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            write_idx: 0,
            total_count: 0,
            running_sum: 0.0,
            running_sum_sq: 0.0,
            ema: 0.0,
            ema_initialized: false,
        }
    }

    fn push(&mut self, value: f64, capacity: usize, alpha: f64) {
        if self.samples.len() < capacity {
            // Buffer not yet full
            self.samples.push(value);
            self.running_sum += value;
            self.running_sum_sq += value * value;
        } else {
            // Overwrite oldest sample
            let old = self.samples[self.write_idx];
            self.running_sum -= old;
            self.running_sum_sq -= old * old;
            self.samples[self.write_idx] = value;
            self.running_sum += value;
            self.running_sum_sq += value * value;
        }
        self.write_idx = (self.write_idx + 1) % capacity;
        self.total_count += 1;

        // Update EMA
        if self.ema_initialized {
            self.ema = alpha * value + (1.0 - alpha) * self.ema;
        } else {
            self.ema = value;
            self.ema_initialized = true;
        }
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            None
        } else {
            Some(self.running_sum / self.samples.len() as f64)
        }
    }

    fn variance(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let n = self.samples.len() as f64;
        let mean = self.running_sum / n;
        let mean_sq = self.running_sum_sq / n;
        // Clamp to avoid negative values from floating-point drift
        Some((mean_sq - mean * mean).max(0.0))
    }

    fn stddev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    fn min(&self) -> Option<f64> {
        self.samples
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn max(&self) -> Option<f64> {
        self.samples
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn latest(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let idx = if self.write_idx == 0 {
            self.samples.len() - 1
        } else {
            self.write_idx - 1
        };
        Some(self.samples[idx])
    }
}

/// An alert triggered when a metric value exceeds its configured threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAlert {
    /// Metric that triggered the alert.
    pub metric: String,
    /// The value that triggered the alert.
    pub value: f64,
    /// The configured lower bound.
    pub min_threshold: f64,
    /// The configured upper bound.
    pub max_threshold: f64,
    /// Total sample index when the alert was triggered.
    pub sample_index: u64,
}

/// Running statistics snapshot for a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Metric name.
    pub metric: String,
    /// Number of samples currently in the window.
    pub window_count: usize,
    /// Total samples ever pushed.
    pub total_count: u64,
    /// Mean of samples in the window.
    pub mean: f64,
    /// Standard deviation of samples in the window.
    pub stddev: f64,
    /// Minimum in the window.
    pub min: f64,
    /// Maximum in the window.
    pub max: f64,
    /// Most recent value.
    pub latest: f64,
    /// Exponential moving average.
    pub ema: f64,
}

/// Real-time quality monitor that tracks multiple metrics with sliding windows.
#[derive(Debug, Clone)]
pub struct RealtimeQualityMonitor {
    /// Configuration.
    config: MonitorConfig,
    /// Per-metric sliding windows.
    windows: HashMap<String, MetricWindow>,
    /// Accumulated alerts.
    alerts: Vec<QualityAlert>,
}

impl RealtimeQualityMonitor {
    /// Creates a new monitor with the given configuration.
    #[must_use]
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            windows: HashMap::new(),
            alerts: Vec::new(),
        }
    }

    /// Pushes a new sample for the named metric.
    ///
    /// If a threshold is configured for this metric and the value is outside
    /// the allowed range, an alert is recorded.
    pub fn push(&mut self, metric: &str, value: f64) {
        let window = self
            .windows
            .entry(metric.to_string())
            .or_insert_with(|| MetricWindow::new(self.config.window_size));

        window.push(value, self.config.window_size, self.config.ema_alpha);

        // Check threshold
        if let Some(&(min_t, max_t)) = self.config.thresholds.get(metric) {
            if value < min_t || value > max_t {
                self.alerts.push(QualityAlert {
                    metric: metric.to_string(),
                    value,
                    min_threshold: min_t,
                    max_threshold: max_t,
                    sample_index: window.total_count,
                });
            }
        }
    }

    /// Returns the running mean for the named metric in the current window.
    #[must_use]
    pub fn running_mean(&self, metric: &str) -> Option<f64> {
        self.windows.get(metric).and_then(|w| w.mean())
    }

    /// Returns the running standard deviation for the named metric.
    #[must_use]
    pub fn running_stddev(&self, metric: &str) -> Option<f64> {
        self.windows.get(metric).and_then(|w| w.stddev())
    }

    /// Returns the exponential moving average for the named metric.
    #[must_use]
    pub fn ema(&self, metric: &str) -> Option<f64> {
        self.windows
            .get(metric)
            .filter(|w| w.ema_initialized)
            .map(|w| w.ema)
    }

    /// Returns the most recent value for the named metric.
    #[must_use]
    pub fn latest(&self, metric: &str) -> Option<f64> {
        self.windows.get(metric).and_then(|w| w.latest())
    }

    /// Returns the min within the current window for the named metric.
    #[must_use]
    pub fn window_min(&self, metric: &str) -> Option<f64> {
        self.windows.get(metric).and_then(|w| w.min())
    }

    /// Returns the max within the current window for the named metric.
    #[must_use]
    pub fn window_max(&self, metric: &str) -> Option<f64> {
        self.windows.get(metric).and_then(|w| w.max())
    }

    /// Returns the total number of samples ever pushed for a metric.
    #[must_use]
    pub fn total_count(&self, metric: &str) -> u64 {
        self.windows.get(metric).map_or(0, |w| w.total_count)
    }

    /// Returns the number of samples currently in the window.
    #[must_use]
    pub fn window_count(&self, metric: &str) -> usize {
        self.windows.get(metric).map_or(0, |w| w.len())
    }

    /// Takes a snapshot of a single metric's current state.
    #[must_use]
    pub fn snapshot(&self, metric: &str) -> Option<MetricSnapshot> {
        let w = self.windows.get(metric)?;
        Some(MetricSnapshot {
            metric: metric.to_string(),
            window_count: w.len(),
            total_count: w.total_count,
            mean: w.mean().unwrap_or(0.0),
            stddev: w.stddev().unwrap_or(0.0),
            min: w.min().unwrap_or(0.0),
            max: w.max().unwrap_or(0.0),
            latest: w.latest().unwrap_or(0.0),
            ema: w.ema,
        })
    }

    /// Takes snapshots of all tracked metrics.
    #[must_use]
    pub fn snapshot_all(&self) -> Vec<MetricSnapshot> {
        let mut keys: Vec<&String> = self.windows.keys().collect();
        keys.sort();
        keys.into_iter().filter_map(|k| self.snapshot(k)).collect()
    }

    /// Returns all accumulated alerts.
    #[must_use]
    pub fn alerts(&self) -> &[QualityAlert] {
        &self.alerts
    }

    /// Returns the number of alerts triggered so far.
    #[must_use]
    pub fn alert_count(&self) -> usize {
        self.alerts.len()
    }

    /// Clears all accumulated alerts.
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    /// Resets all metric windows and alerts.
    pub fn reset(&mut self) {
        self.windows.clear();
        self.alerts.clear();
    }

    /// Returns a list of all metric names being tracked.
    #[must_use]
    pub fn tracked_metrics(&self) -> Vec<String> {
        let mut names: Vec<String> = self.windows.keys().cloned().collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_push_and_mean() {
        let config = MonitorConfig::new(10);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("psnr", 30.0);
        monitor.push("psnr", 40.0);
        let mean = monitor.running_mean("psnr");
        assert!(mean.is_some());
        assert!((mean.unwrap_or(0.0) - 35.0).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_missing_metric_returns_none() {
        let config = MonitorConfig::new(10);
        let monitor = RealtimeQualityMonitor::new(config);
        assert!(monitor.running_mean("psnr").is_none());
        assert!(monitor.ema("psnr").is_none());
        assert!(monitor.latest("psnr").is_none());
    }

    #[test]
    fn test_monitor_sliding_window_evicts() {
        let config = MonitorConfig::new(3);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("x", 10.0);
        monitor.push("x", 20.0);
        monitor.push("x", 30.0);
        // Window full: [10, 20, 30], mean = 20
        assert!((monitor.running_mean("x").unwrap_or(0.0) - 20.0).abs() < 1e-6);

        monitor.push("x", 40.0);
        // Window: [40, 20, 30], mean = 30
        assert!((monitor.running_mean("x").unwrap_or(0.0) - 30.0).abs() < 1e-6);
        assert_eq!(monitor.window_count("x"), 3);
        assert_eq!(monitor.total_count("x"), 4);
    }

    #[test]
    fn test_monitor_ema_tracks_recent() {
        let config = MonitorConfig::new(100).with_ema_alpha(0.5);
        let mut monitor = RealtimeQualityMonitor::new(config);

        // Push many low values then a high value
        for _ in 0..20 {
            monitor.push("v", 10.0);
        }
        let ema_before = monitor.ema("v").unwrap_or(0.0);
        monitor.push("v", 100.0);
        let ema_after = monitor.ema("v").unwrap_or(0.0);

        // EMA should jump toward 100 after the high value
        assert!(ema_after > ema_before);
    }

    #[test]
    fn test_monitor_stddev() {
        let config = MonitorConfig::new(100);
        let mut monitor = RealtimeQualityMonitor::new(config);
        // All same values => stddev = 0
        for _ in 0..10 {
            monitor.push("c", 5.0);
        }
        let sd = monitor.running_stddev("c").unwrap_or(-1.0);
        assert!(sd.abs() < 1e-6);

        // Different values => stddev > 0
        let config2 = MonitorConfig::new(100);
        let mut monitor2 = RealtimeQualityMonitor::new(config2);
        monitor2.push("d", 10.0);
        monitor2.push("d", 20.0);
        let sd2 = monitor2.running_stddev("d").unwrap_or(0.0);
        assert!(sd2 > 0.0);
    }

    #[test]
    fn test_monitor_min_max_latest() {
        let config = MonitorConfig::new(10);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("q", 5.0);
        monitor.push("q", 15.0);
        monitor.push("q", 10.0);
        assert!((monitor.window_min("q").unwrap_or(0.0) - 5.0).abs() < 1e-6);
        assert!((monitor.window_max("q").unwrap_or(0.0) - 15.0).abs() < 1e-6);
        assert!((monitor.latest("q").unwrap_or(0.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_threshold_alert() {
        let config = MonitorConfig::new(10).with_threshold("psnr", 30.0, 50.0);
        let mut monitor = RealtimeQualityMonitor::new(config);

        monitor.push("psnr", 35.0); // OK
        assert_eq!(monitor.alert_count(), 0);

        monitor.push("psnr", 25.0); // Below min
        assert_eq!(monitor.alert_count(), 1);
        assert!((monitor.alerts()[0].value - 25.0).abs() < 1e-6);

        monitor.push("psnr", 55.0); // Above max
        assert_eq!(monitor.alert_count(), 2);
    }

    #[test]
    fn test_monitor_clear_alerts() {
        let config = MonitorConfig::new(10).with_threshold("x", 0.0, 10.0);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("x", 20.0);
        assert_eq!(monitor.alert_count(), 1);
        monitor.clear_alerts();
        assert_eq!(monitor.alert_count(), 0);
    }

    #[test]
    fn test_monitor_reset() {
        let config = MonitorConfig::new(10);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("a", 1.0);
        monitor.push("b", 2.0);
        assert_eq!(monitor.tracked_metrics().len(), 2);
        monitor.reset();
        assert!(monitor.tracked_metrics().is_empty());
        assert!(monitor.running_mean("a").is_none());
    }

    #[test]
    fn test_monitor_snapshot() {
        let config = MonitorConfig::new(10);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("psnr", 30.0);
        monitor.push("psnr", 40.0);
        let snap = monitor.snapshot("psnr").expect("should have snapshot");
        assert_eq!(snap.metric, "psnr");
        assert_eq!(snap.window_count, 2);
        assert_eq!(snap.total_count, 2);
        assert!((snap.mean - 35.0).abs() < 1e-6);
        assert!((snap.latest - 40.0).abs() < 1e-6);
        assert!((snap.min - 30.0).abs() < 1e-6);
        assert!((snap.max - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_snapshot_all() {
        let config = MonitorConfig::new(10);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("psnr", 30.0);
        monitor.push("ssim", 0.95);
        let snaps = monitor.snapshot_all();
        assert_eq!(snaps.len(), 2);
        // Should be sorted alphabetically
        assert_eq!(snaps[0].metric, "psnr");
        assert_eq!(snaps[1].metric, "ssim");
    }

    #[test]
    fn test_monitor_tracked_metrics() {
        let config = MonitorConfig::new(10);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("vmaf", 80.0);
        monitor.push("psnr", 35.0);
        let metrics = monitor.tracked_metrics();
        assert_eq!(metrics, vec!["psnr", "vmaf"]);
    }

    #[test]
    fn test_monitor_config_ema_clamped() {
        let config = MonitorConfig::new(10).with_ema_alpha(2.0);
        assert!((config.ema_alpha - 1.0).abs() < 1e-6);
        let config2 = MonitorConfig::new(10).with_ema_alpha(-1.0);
        assert!((config2.ema_alpha - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_large_window_stress() {
        let config = MonitorConfig::new(5);
        let mut monitor = RealtimeQualityMonitor::new(config);
        for i in 0..1000u64 {
            monitor.push("v", i as f64);
        }
        assert_eq!(monitor.total_count("v"), 1000);
        assert_eq!(monitor.window_count("v"), 5);
        // Window should contain the last 5 values: 995..=999
        let mean = monitor.running_mean("v").unwrap_or(0.0);
        assert!((mean - 997.0).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_single_sample() {
        let config = MonitorConfig::new(10);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("x", 42.0);
        assert!((monitor.running_mean("x").unwrap_or(0.0) - 42.0).abs() < 1e-6);
        assert!((monitor.running_stddev("x").unwrap_or(-1.0)).abs() < 1e-6);
        assert!((monitor.ema("x").unwrap_or(0.0) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_alert_sample_index() {
        let config = MonitorConfig::new(10).with_threshold("x", 0.0, 10.0);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("x", 5.0);
        monitor.push("x", 5.0);
        monitor.push("x", 20.0); // sample 3 triggers alert
        assert_eq!(monitor.alerts()[0].sample_index, 3);
    }

    #[test]
    fn test_snapshot_missing_metric() {
        let config = MonitorConfig::new(10);
        let monitor = RealtimeQualityMonitor::new(config);
        assert!(monitor.snapshot("missing").is_none());
    }

    #[test]
    fn test_monitor_window_size_one() {
        let config = MonitorConfig::new(1);
        let mut monitor = RealtimeQualityMonitor::new(config);
        monitor.push("x", 10.0);
        assert!((monitor.running_mean("x").unwrap_or(0.0) - 10.0).abs() < 1e-6);
        monitor.push("x", 20.0);
        assert!((monitor.running_mean("x").unwrap_or(0.0) - 20.0).abs() < 1e-6);
        assert_eq!(monitor.window_count("x"), 1);
    }
}
