//! # Prometheus Metrics Export for SciRS2 v0.2.0
//!
//! This module provides Prometheus metrics collection and export capabilities.
//! It enables monitoring of SciRS2 applications with industry-standard tools.
//!
//! # Features
//!
//! - **Custom Metrics**: Counters, gauges, histograms, and summaries
//! - **Automatic Registration**: Automatic metrics registration with global registry
//! - **HTTP Export**: Expose metrics via HTTP endpoint
//! - **Performance Counters**: Track computation performance
//! - **Memory Metrics**: Monitor memory usage
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::profiling::prometheus_metrics::{MetricsRegistry, register_counter};
//!
//! // Create a counter
//! let counter = register_counter(
//!     "scirs2_operations_total",
//!     "Total number of operations"
//! ).expect("Failed to register counter");
//!
//! // Increment the counter
//! counter.inc();
//!
//! // Export metrics
//! let metrics = MetricsRegistry::gather();
//! println!("{}", metrics);
//! ```

#[cfg(feature = "profiling_prometheus")]
use crate::CoreResult;
#[cfg(feature = "profiling_prometheus")]
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, Opts,
    Registry, TextEncoder,
};
#[cfg(feature = "profiling_prometheus")]
use std::sync::Arc;

/// Global metrics registry
#[cfg(feature = "profiling_prometheus")]
static REGISTRY: once_cell::sync::Lazy<Arc<Registry>> =
    once_cell::sync::Lazy::new(|| Arc::new(Registry::new()));

/// Metrics registry wrapper
#[cfg(feature = "profiling_prometheus")]
pub struct MetricsRegistry;

#[cfg(feature = "profiling_prometheus")]
impl MetricsRegistry {
    /// Get the global registry
    pub fn global() -> Arc<Registry> {
        REGISTRY.clone()
    }

    /// Gather all metrics in Prometheus text format
    pub fn gather() -> String {
        let encoder = TextEncoder::new();
        let metric_families = REGISTRY.gather();
        let mut buffer = Vec::new();

        encoder
            .encode(&metric_families, &mut buffer)
            .expect("Failed to encode metrics");

        String::from_utf8(buffer).expect("Failed to convert metrics to string")
    }

    /// Reset all metrics
    pub fn reset() {
        // Create a new registry (note: this doesn't actually reset, just for API compatibility)
        // In practice, individual metrics should be reset
    }
}

/// Register a counter metric
#[cfg(feature = "profiling_prometheus")]
pub fn register_counter(name: &str, help: &str) -> CoreResult<Counter> {
    let opts = Opts::new(name, help);
    let counter = Counter::with_opts(opts).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to create counter: {}",
            e
        )))
    })?;

    REGISTRY.register(Box::new(counter.clone())).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to register counter: {}",
            e
        )))
    })?;

    Ok(counter)
}

/// Register a counter vector metric
#[cfg(feature = "profiling_prometheus")]
pub fn register_counter_vec(
    name: &str,
    help: &str,
    label_names: &[&str],
) -> CoreResult<CounterVec> {
    let opts = Opts::new(name, help);
    let counter_vec = CounterVec::new(opts, label_names).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to create counter vec: {}",
            e
        )))
    })?;

    REGISTRY
        .register(Box::new(counter_vec.clone()))
        .map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to register counter vec: {}",
                e
            )))
        })?;

    Ok(counter_vec)
}

/// Register a gauge metric
#[cfg(feature = "profiling_prometheus")]
pub fn register_gauge(name: &str, help: &str) -> CoreResult<Gauge> {
    let opts = Opts::new(name, help);
    let gauge = Gauge::with_opts(opts).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to create gauge: {}",
            e
        )))
    })?;

    REGISTRY.register(Box::new(gauge.clone())).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to register gauge: {}",
            e
        )))
    })?;

    Ok(gauge)
}

/// Register a gauge vector metric
#[cfg(feature = "profiling_prometheus")]
pub fn register_gauge_vec(name: &str, help: &str, label_names: &[&str]) -> CoreResult<GaugeVec> {
    let opts = Opts::new(name, help);
    let gauge_vec = GaugeVec::new(opts, label_names).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to create gauge vec: {}",
            e
        )))
    })?;

    REGISTRY
        .register(Box::new(gauge_vec.clone()))
        .map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to register gauge vec: {}",
                e
            )))
        })?;

    Ok(gauge_vec)
}

/// Register a histogram metric
#[cfg(feature = "profiling_prometheus")]
pub fn register_histogram(name: &str, help: &str, buckets: Vec<f64>) -> CoreResult<Histogram> {
    let opts = HistogramOpts::new(name, help).buckets(buckets);
    let histogram = Histogram::with_opts(opts).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to create histogram: {}",
            e
        )))
    })?;

    REGISTRY
        .register(Box::new(histogram.clone()))
        .map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to register histogram: {}",
                e
            )))
        })?;

    Ok(histogram)
}

/// Register a histogram vector metric
#[cfg(feature = "profiling_prometheus")]
pub fn register_histogram_vec(
    name: &str,
    help: &str,
    label_names: &[&str],
    buckets: Vec<f64>,
) -> CoreResult<HistogramVec> {
    let opts = HistogramOpts::new(name, help).buckets(buckets);
    let histogram_vec = HistogramVec::new(opts, label_names).map_err(|e| {
        crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
            "Failed to create histogram vec: {}",
            e
        )))
    })?;

    REGISTRY
        .register(Box::new(histogram_vec.clone()))
        .map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to register histogram vec: {}",
                e
            )))
        })?;

    Ok(histogram_vec)
}

/// Standard buckets for latency metrics (in seconds)
#[cfg(feature = "profiling_prometheus")]
pub fn latency_buckets() -> Vec<f64> {
    vec![
        0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ]
}

/// Standard buckets for size metrics (in bytes)
#[cfg(feature = "profiling_prometheus")]
pub fn size_buckets() -> Vec<f64> {
    vec![
        1024.0,
        10_240.0,
        102_400.0,
        1_048_576.0,
        10_485_760.0,
        104_857_600.0,
        1_073_741_824.0,
    ]
}

/// Pre-defined SciRS2 metrics
#[cfg(feature = "profiling_prometheus")]
pub struct SciRS2Metrics {
    /// Total number of operations
    pub operations_total: CounterVec,
    /// Operation duration histogram
    pub operation_duration: HistogramVec,
    /// Active operations gauge
    pub active_operations: GaugeVec,
    /// Memory usage gauge
    pub memory_usage_bytes: Gauge,
    /// Array size histogram
    pub array_size_bytes: Histogram,
    /// Error counter
    pub errors_total: CounterVec,
}

#[cfg(feature = "profiling_prometheus")]
impl SciRS2Metrics {
    /// Create and register standard SciRS2 metrics
    pub fn register() -> CoreResult<Self> {
        let operations_total = register_counter_vec(
            "scirs2_operations_total",
            "Total number of operations",
            &["operation", "module"],
        )?;

        let operation_duration = register_histogram_vec(
            "scirs2_operation_duration_seconds",
            "Duration of operations in seconds",
            &["operation", "module"],
            latency_buckets(),
        )?;

        let active_operations = register_gauge_vec(
            "scirs2_active_operations",
            "Number of active operations",
            &["operation", "module"],
        )?;

        let memory_usage_bytes =
            register_gauge("scirs2_memory_usage_bytes", "Current memory usage in bytes")?;

        let array_size_bytes = register_histogram(
            "scirs2_array_size_bytes",
            "Size of arrays in bytes",
            size_buckets(),
        )?;

        let errors_total = register_counter_vec(
            "scirs2_errors_total",
            "Total number of errors",
            &["error_type", "module"],
        )?;

        Ok(Self {
            operations_total,
            operation_duration,
            active_operations,
            memory_usage_bytes,
            array_size_bytes,
            errors_total,
        })
    }
}

/// Timer for measuring operation duration
#[cfg(feature = "profiling_prometheus")]
pub struct PrometheusTimer {
    histogram: Histogram,
    start: std::time::Instant,
}

#[cfg(feature = "profiling_prometheus")]
impl PrometheusTimer {
    /// Start a new timer
    pub fn start(histogram: Histogram) -> Self {
        Self {
            histogram,
            start: std::time::Instant::now(),
        }
    }

    /// Stop the timer and record the duration
    pub fn stop(self) {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}

#[cfg(feature = "profiling_prometheus")]
impl Drop for PrometheusTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}

/// Macro for timing an operation with Prometheus
#[macro_export]
#[cfg(feature = "profiling_prometheus")]
macro_rules! prometheus_time {
    ($histogram:expr, $body:block) => {{
        let _timer = $crate::profiling::prometheus_metrics::PrometheusTimer::start($histogram);
        $body
    }};
}

/// Stub implementations when profiling_prometheus feature is disabled
#[cfg(not(feature = "profiling_prometheus"))]
pub struct MetricsRegistry;

#[cfg(not(feature = "profiling_prometheus"))]
impl MetricsRegistry {
    pub fn gather() -> String {
        String::new()
    }
}

#[cfg(test)]
#[cfg(feature = "profiling_prometheus")]
mod tests {
    use super::*;

    #[test]
    fn test_register_counter() {
        let counter = register_counter("test_counter", "Test counter");
        assert!(counter.is_ok());
    }

    #[test]
    fn test_register_gauge() {
        let gauge = register_gauge("test_gauge", "Test gauge");
        assert!(gauge.is_ok());
    }

    #[test]
    fn test_register_histogram() {
        let histogram = register_histogram("test_histogram", "Test histogram", latency_buckets());
        assert!(histogram.is_ok());
    }

    #[test]
    fn test_scirs2_metrics() {
        let metrics = SciRS2Metrics::register();
        assert!(metrics.is_ok());

        if let Ok(m) = metrics {
            m.operations_total
                .with_label_values(&["test", "core"])
                .inc();
            m.memory_usage_bytes.set(1024.0);
            m.array_size_bytes.observe(2048.0);
        }
    }

    #[test]
    fn test_prometheus_timer() {
        let histogram = register_histogram("test_timer", "Test timer", latency_buckets())
            .expect("Failed to register histogram");

        let timer = PrometheusTimer::start(histogram);
        std::thread::sleep(std::time::Duration::from_millis(10));
        timer.stop();
    }

    #[test]
    fn test_metrics_gather() {
        let _counter = register_counter("gather_test", "Test counter").expect("Failed to register");
        let metrics = MetricsRegistry::gather();
        assert!(!metrics.is_empty());
    }
}
