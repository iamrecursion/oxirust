//! Metrics collector for monitoring server performance.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Counter (monotonically increasing).
    Counter,
    /// Gauge (can increase or decrease).
    Gauge,
    /// Histogram (distribution of values).
    Histogram,
}

/// Metric value.
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name.
    pub name: String,

    /// Metric type.
    pub metric_type: MetricType,

    /// Current value.
    pub value: f64,

    /// Labels.
    pub labels: HashMap<String, String>,

    /// Last updated.
    pub updated_at: Instant,
}

impl Metric {
    /// Creates a new metric.
    #[must_use]
    pub fn new(name: impl Into<String>, metric_type: MetricType) -> Self {
        Self {
            name: name.into(),
            metric_type,
            value: 0.0,
            labels: HashMap::new(),
            updated_at: Instant::now(),
        }
    }

    /// Sets a label.
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Increments the metric.
    pub fn increment(&mut self, amount: f64) {
        self.value += amount;
        self.updated_at = Instant::now();
    }

    /// Sets the metric value.
    pub fn set(&mut self, value: f64) {
        self.value = value;
        self.updated_at = Instant::now();
    }
}

/// Metrics collector.
pub struct MetricsCollector {
    /// Registered metrics.
    metrics: Arc<RwLock<HashMap<String, Metric>>>,

    /// Stream-specific metrics.
    stream_metrics: Arc<RwLock<HashMap<String, StreamMetrics>>>,

    /// Start time.
    start_time: Instant,
}

/// Stream-specific metrics.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct StreamMetrics {
    /// Stream key.
    stream_key: String,

    /// Bytes received.
    bytes_received: u64,

    /// Bytes sent.
    bytes_sent: u64,

    /// Packets received.
    packets_received: u64,

    /// Packets sent.
    packets_sent: u64,

    /// Current viewers.
    current_viewers: u64,

    /// Peak viewers.
    peak_viewers: u64,

    /// Start time.
    start_time: Instant,

    /// Last updated.
    last_updated: Instant,
}

impl MetricsCollector {
    /// Creates a new metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            stream_metrics: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Registers a metric.
    pub fn register_metric(&self, metric: Metric) {
        let mut metrics = self.metrics.write();
        metrics.insert(metric.name.clone(), metric);
    }

    /// Increments a counter.
    pub fn increment_counter(&self, name: &str, amount: f64) {
        let mut metrics = self.metrics.write();
        if let Some(metric) = metrics.get_mut(name) {
            metric.increment(amount);
        } else {
            let mut metric = Metric::new(name, MetricType::Counter);
            metric.increment(amount);
            metrics.insert(name.to_string(), metric);
        }
    }

    /// Sets a gauge value.
    pub fn set_gauge(&self, name: &str, value: f64) {
        let mut metrics = self.metrics.write();
        if let Some(metric) = metrics.get_mut(name) {
            metric.set(value);
        } else {
            let mut metric = Metric::new(name, MetricType::Gauge);
            metric.set(value);
            metrics.insert(name.to_string(), metric);
        }
    }

    /// Records bytes received.
    pub fn record_bytes_received(&self, bytes: u64) {
        self.increment_counter("bytes_received_total", bytes as f64);
    }

    /// Records bytes sent.
    pub fn record_bytes_sent(&self, bytes: u64) {
        self.increment_counter("bytes_sent_total", bytes as f64);
    }

    /// Records packet received.
    pub fn record_packet_received(&self) {
        self.increment_counter("packets_received_total", 1.0);
    }

    /// Records packet sent.
    pub fn record_packet_sent(&self) {
        self.increment_counter("packets_sent_total", 1.0);
    }

    /// Records stream activity.
    pub fn record_stream_active(&self, app_name: &str, stream_key: &str) {
        let key = format!("{}/{}", app_name, stream_key);
        let mut stream_metrics = self.stream_metrics.write();

        if let Some(metrics) = stream_metrics.get_mut(&key) {
            metrics.last_updated = Instant::now();
        } else {
            stream_metrics.insert(
                key.clone(),
                StreamMetrics {
                    stream_key: key,
                    bytes_received: 0,
                    bytes_sent: 0,
                    packets_received: 0,
                    packets_sent: 0,
                    current_viewers: 0,
                    peak_viewers: 0,
                    start_time: Instant::now(),
                    last_updated: Instant::now(),
                },
            );
        }
    }

    /// Increments viewer count.
    pub fn increment_viewers(&self, app_name: &str, stream_key: &str) {
        let key = format!("{}/{}", app_name, stream_key);
        let mut stream_metrics = self.stream_metrics.write();

        if let Some(metrics) = stream_metrics.get_mut(&key) {
            metrics.current_viewers += 1;
            if metrics.current_viewers > metrics.peak_viewers {
                metrics.peak_viewers = metrics.current_viewers;
            }
        }

        self.increment_counter("viewers_total", 1.0);
    }

    /// Decrements viewer count.
    pub fn decrement_viewers(&self, app_name: &str, stream_key: &str) {
        let key = format!("{}/{}", app_name, stream_key);
        let mut stream_metrics = self.stream_metrics.write();

        if let Some(metrics) = stream_metrics.get_mut(&key) {
            if metrics.current_viewers > 0 {
                metrics.current_viewers -= 1;
            }
        }
    }

    /// Gets all metrics.
    #[must_use]
    pub fn get_metrics(&self) -> HashMap<String, Metric> {
        let metrics = self.metrics.read();
        metrics.clone()
    }

    /// Gets a specific metric.
    #[must_use]
    pub fn get_metric(&self, name: &str) -> Option<Metric> {
        let metrics = self.metrics.read();
        metrics.get(name).cloned()
    }

    /// Gets uptime.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        Instant::now().duration_since(self.start_time)
    }

    /// Clears all metrics.
    pub fn clear(&self) {
        let mut metrics = self.metrics.write();
        metrics.clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
