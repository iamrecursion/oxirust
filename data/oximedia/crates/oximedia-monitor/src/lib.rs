//! Comprehensive system monitoring and alerting for `OxiMedia`.
//!
//! This crate provides professional-grade system and application monitoring with:
//!
//! - **System Metrics**: CPU, memory, disk, network, GPU, temperature
//! - **Application Metrics**: Encoding throughput, job statistics, worker status
//! - **Quality Metrics**: Bitrate, quality scores (PSNR, SSIM, VMAF)
//! - **Time Series Storage**: In-memory ring buffer + `SQLite` historical storage
//! - **Alerting**: Multiple channels (email, Slack, Discord, webhook, SMS, file)
//! - **REST API**: Query metrics, manage alerts, health checks
//! - **WebSocket**: Real-time metric streaming
//! - **Health Checks**: Component health monitoring
//! - **Log Aggregation**: Structured logging with search
//! - **Dashboards**: Data provider for external visualization tools
//! - **Prometheus**: Compatible exposition format
//!
//! # Example
//!
//! ```no_run
//! use oximedia_monitor::{InMemoryMonitor, MonitorConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = MonitorConfig::default();
//!     let monitor = InMemoryMonitor::new(config)?;
//!
//!     // Start monitoring
//!     monitor.start().await?;
//!
//!     // Get application metrics
//!     let app_metrics = monitor.application_metrics();
//!     println!("Total frames encoded: {}", app_metrics.encoding.total_frames);
//!
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_async,
    clippy::unused_self
)]

pub mod alert;
pub mod alert_rule;
pub mod alert_rules;
pub mod alerting_pipeline;
pub mod anomaly;
pub mod api;
/// Capacity planning and resource utilisation forecasting.
pub mod capacity_planner;
/// Metric cardinality limiter to prevent unbounded label growth.
pub mod cardinality_limiter;
pub mod config;
pub mod correlation;
/// Monotonic counter metrics for discrete event tracking.
pub mod counter_metrics;
pub mod dashboard;
pub mod dashboard_metric;
pub mod dashboard_widget;
/// Exponential backoff delivery retrier for webhook/email alert failures.
pub mod delivery_retrier;
pub mod error;
pub mod event_bus;
pub mod health;
pub mod health_check;
/// Incident tracking and lifecycle management.
pub mod incident_tracker;
pub mod integration;
pub mod log_aggregator;
pub mod logs;
pub mod metric_aggregation;
/// Metric batching to reduce storage write frequency under high load.
pub mod metric_batcher;
/// Multi-resolution metric downsampling for historical data retention.
pub mod metric_downsampler;
/// Metric export in Prometheus, JSON, CSV, and StatsD formats.
pub mod metric_export;
/// Metric processing pipeline with transformations and aggregations.
pub mod metric_pipeline;
/// Metric recording and playback for debugging.
pub mod metric_recorder;
pub mod metric_store;
pub mod metrics;
pub mod panel_view;
pub mod reporting;
/// Resource usage forecasting and trend analysis.
pub mod resource_forecast;
pub mod retention;
/// Seasonal decomposition for time-series anomaly detection.
pub mod seasonal_decomposition;
pub mod simple;
pub mod sla;
pub mod slo_tracker;
pub mod storage;
/// System-level metrics: CPU, memory, disk, and network.
pub mod system_metrics;
/// Distributed trace-span tracking for latency measurement.
pub mod trace_span;
pub mod uptime_tracker;

// ── New modules (0.1.2 enhancements) ────────────────────────────────────────
/// Alert correlation engine: groups related alerts firing within a time window.
pub mod alert_correlation;
/// Dashboard templating with `{{variable}}` substitution.
pub mod dashboard_template;
/// PagerDuty and OpsGenie alerting channel integrations.
pub mod notification_channels;
/// OpsGenie Alerts API integration (pure Rust, no reqwest).
pub mod opsgenie;
/// OpenTelemetry Protocol (OTLP) metric export alongside Prometheus.
pub mod otlp_export;
/// PagerDuty Events API v2 integration (pure Rust, no reqwest).
pub mod pagerduty;
/// StatsD line-protocol ingestion endpoint.
pub mod statsd_ingestion;
/// W3C Trace Context propagation (traceparent/tracestate headers).
pub mod w3c_trace_context;

// ── New modules (0.1.8 orphan wiring) ─────────────────────────────────────
/// Multi-dimensional capacity advisor with headroom and scale-up recommendations.
pub mod capacity_advisor;
/// Metric cardinality enforcement with configurable overflow actions.
pub mod cardinality;
/// Dashboard layout template system for configuring monitoring views.
pub mod dashboard_layout;
/// GPU metrics collection via Linux sysfs and macOS IOKit paths (non-NVIDIA).
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_metrics_sysfs;
/// Full incident lifecycle management with MTTD/MTTR analytics.
pub mod incident_manager;
/// Multi-tier metric downsampling with retention policies.
pub mod metric_downsample;
/// High-frequency time-series downsampling (LTTB, Average, MaxMin, LastValue).
pub mod metric_downsampling;
/// OpenTelemetry-compatible metrics export (wire-format only, no OTel SDK).
pub mod opentelemetry_export;
/// Seasonal decomposition for periodic anomaly detection.
pub mod seasonal;
/// Comprehensive SLA tracking with tier definitions and breach detection.
pub mod sla_tracker;
/// SQLite connection pool for concurrent metric storage access.
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod sqlite_pool;
/// StatsD wire-format parser (free-function API).
pub mod statsd_parser;
/// W3C Trace Context propagation using `traceparent`/`tracestate` headers.
pub mod trace_context;
/// Distributed trace export with Jaeger JSON format and tail-based sampling.
pub mod trace_exporter;

// ── Subdir modules (broadcast monitoring) ─────────────────────────────────
/// Audio metering and monitoring (VU, PPM, loudness, phase, spectrum).
pub mod audio;
/// Closed caption monitoring and validation.
pub mod cc;
/// Broadcast compliance checking (EBU R128, SMPTE, ATSC).
pub mod compliance;
/// Focus peaking and assist (depends on scopes, declared after).
pub mod focus;
/// Multi-viewer layout and composition.
pub mod multiviewer;
/// Reference signal comparison (MSE, PSNR, SSIM).
pub mod reference;
/// Video scope monitoring (waveform, vectorscope, histogram, parade).
pub mod scopes;
/// Signal status and error monitoring.
pub mod status;
/// Technical video analysis (levels, range, gamut, cadence, sync).
pub mod technical;
/// Timecode monitoring and validation.
pub mod timecode;

#[cfg(not(target_arch = "wasm32"))]
pub use alert::AlertManager;
pub use alert::{Alert, AlertRule, AlertSeverity};
pub use alert_correlation::{AlertCorrelationEngine, AlertGroup};
pub use config::{AlertConfig, ApiConfig, MetricsConfig, MonitorConfig, StorageConfig};
pub use error::{MonitorError, MonitorResult};
pub use metrics::{
    ApplicationMetrics, EncodingMetrics, JobMetrics, QualityMetrics, WorkerMetrics, WorkerStatus,
};
#[cfg(not(target_arch = "wasm32"))]
pub use metrics::{MetricsCollector, SystemMetrics, SystemMetricsConfig};
pub use simple::{
    CodecMetrics, Comparison, FiredAlert, HealthCheck, HealthCheckAggregator, HealthStatus,
    NotificationAction, SimpleAlertManager, SimpleAlertRule, SimpleMetricsCollector,
    SimpleMetricsSnapshot,
};
pub use storage::RingBuffer;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub use storage::{QueryEngine, SqliteStorage, TimeRange, TimeSeriesQuery};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

/// Main monitoring system (SQLite-backed).
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub struct OximediaMonitor {
    config: MonitorConfig,
    metrics_collector: Arc<MetricsCollector>,
    storage: Arc<SqliteStorage>,
    query_engine: Arc<QueryEngine>,
    alert_manager: Option<Arc<AlertManager>>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
impl OximediaMonitor {
    /// Create a new monitoring system.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub async fn new(config: MonitorConfig) -> MonitorResult<Self> {
        config.validate()?;

        let storage = Arc::new(SqliteStorage::new(&config.storage.db_path)?);
        let query_engine = Arc::new(QueryEngine::new((*storage).clone()));

        let metrics_collector = Arc::new(MetricsCollector::new(config.metrics.clone())?);

        let alert_manager = if config.alerts.enabled {
            Some(Arc::new(AlertManager::new(config.alerts.clone())))
        } else {
            None
        };

        Ok(Self {
            config,
            metrics_collector,
            storage,
            query_engine,
            alert_manager,
        })
    }

    /// Start the monitoring system.
    ///
    /// # Errors
    ///
    /// Returns an error if start fails.
    pub async fn start(&self) -> MonitorResult<()> {
        self.metrics_collector.start().await?;

        if let Some(ref alert_manager) = self.alert_manager {
            alert_manager.start().await?;
        }

        Ok(())
    }

    /// Stop the monitoring system.
    pub async fn stop(&self) {
        self.metrics_collector.stop().await;

        if let Some(ref alert_manager) = self.alert_manager {
            alert_manager.stop().await;
        }
    }

    /// Get current system metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if collection fails.
    pub async fn system_metrics(&self) -> MonitorResult<Option<SystemMetrics>> {
        self.metrics_collector.collect_system_metrics().await
    }

    /// Get application metrics.
    #[must_use]
    pub fn application_metrics(&self) -> ApplicationMetrics {
        self.metrics_collector.application_metrics()
    }

    /// Get quality metrics.
    #[must_use]
    pub fn quality_metrics(&self) -> QualityMetrics {
        self.metrics_collector.quality_metrics()
    }

    /// Get the query engine.
    #[must_use]
    pub fn query_engine(&self) -> Arc<QueryEngine> {
        self.query_engine.clone()
    }

    /// Get the metrics collector.
    #[must_use]
    pub fn metrics_collector(&self) -> Arc<MetricsCollector> {
        self.metrics_collector.clone()
    }

    /// Get the storage.
    #[must_use]
    pub fn storage(&self) -> Arc<SqliteStorage> {
        self.storage.clone()
    }

    /// Get the alert manager.
    #[must_use]
    pub fn alert_manager(&self) -> Option<Arc<AlertManager>> {
        self.alert_manager.clone()
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &MonitorConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// In-memory only monitor (works without SQLite feature)
// ---------------------------------------------------------------------------

/// In-memory monitoring system that works without the `sqlite` feature.
///
/// Uses a [`RingBuffer`] for metric storage and the alerting pipeline
/// for rule evaluation, providing a fully functional monitoring system
/// that requires no external dependencies.
#[cfg(not(target_arch = "wasm32"))]
pub struct InMemoryMonitor {
    config: MonitorConfig,
    metrics_collector: Arc<MetricsCollector>,
    ring_buffer: Arc<RingBuffer>,
    alert_manager: Option<Arc<AlertManager>>,
    alerting_pipeline: Arc<parking_lot::RwLock<alerting_pipeline::AlertingPipeline>>,
    metric_batcher: Arc<parking_lot::RwLock<metric_batcher::MetricBatcher>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl InMemoryMonitor {
    /// Create a new in-memory monitoring system.
    ///
    /// # Errors
    ///
    /// Returns an error if metrics collector initialization fails.
    pub fn new(config: MonitorConfig) -> MonitorResult<Self> {
        let metrics_collector = Arc::new(MetricsCollector::new(config.metrics.clone())?);
        let ring_buffer = Arc::new(RingBuffer::new(config.storage.ring_buffer_capacity));
        let alert_manager = if config.alerts.enabled {
            Some(Arc::new(AlertManager::new(config.alerts.clone())))
        } else {
            None
        };
        let pipeline = alerting_pipeline::AlertingPipeline::new();
        let batcher = metric_batcher::MetricBatcher::new(metric_batcher::BatcherConfig::default());

        Ok(Self {
            config,
            metrics_collector,
            ring_buffer,
            alert_manager,
            alerting_pipeline: Arc::new(parking_lot::RwLock::new(pipeline)),
            metric_batcher: Arc::new(parking_lot::RwLock::new(batcher)),
        })
    }

    /// Start the monitoring system.
    ///
    /// # Errors
    ///
    /// Returns an error if start fails.
    pub async fn start(&self) -> MonitorResult<()> {
        self.metrics_collector.start().await?;
        if let Some(ref am) = self.alert_manager {
            am.start().await?;
        }
        Ok(())
    }

    /// Stop the monitoring system.
    pub async fn stop(&self) {
        self.metrics_collector.stop().await;
        if let Some(ref am) = self.alert_manager {
            am.stop().await;
        }
    }

    /// Record a named metric value.
    ///
    /// The value is batched and written to the ring buffer. Any matching
    /// alerting rules are also evaluated.
    pub fn record_metric(&self, name: &str, value: f64) {
        // Write through batcher.
        let mut batcher = self.metric_batcher.write();
        let flushed = batcher.add(name, value);

        // Store flushed batch entries in ring buffer.
        for entry in flushed {
            self.ring_buffer.push_value(entry.value);
        }

        // Evaluate alerting rules.
        let mut pipeline = self.alerting_pipeline.write();
        let _fired = pipeline.evaluate(name, value);
    }

    /// Add an alerting rule to the pipeline.
    pub fn add_alert_rule(&self, rule: alerting_pipeline::PipelineRule) {
        self.alerting_pipeline.write().add_rule(rule);
    }

    /// Get current system metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if collection fails.
    pub async fn system_metrics(&self) -> MonitorResult<Option<SystemMetrics>> {
        self.metrics_collector.collect_system_metrics().await
    }

    /// Get application metrics.
    #[must_use]
    pub fn application_metrics(&self) -> ApplicationMetrics {
        self.metrics_collector.application_metrics()
    }

    /// Get quality metrics.
    #[must_use]
    pub fn quality_metrics(&self) -> QualityMetrics {
        self.metrics_collector.quality_metrics()
    }

    /// Get the metrics collector.
    #[must_use]
    pub fn metrics_collector(&self) -> Arc<MetricsCollector> {
        self.metrics_collector.clone()
    }

    /// Get the alert manager.
    #[must_use]
    pub fn alert_manager(&self) -> Option<Arc<AlertManager>> {
        self.alert_manager.clone()
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Number of entries currently in the ring buffer.
    #[must_use]
    pub fn ring_buffer_len(&self) -> usize {
        self.ring_buffer.len()
    }

    /// Access the ring buffer.
    #[must_use]
    pub fn ring_buffer(&self) -> Arc<RingBuffer> {
        self.ring_buffer.clone()
    }
}

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;

    /// Build a [`MonitorConfig`] suitable for fast unit tests.
    ///
    /// System-metrics collection is disabled so no expensive sysinfo I/O
    /// occurs during construction or during the start/stop lifecycle.  Each
    /// test provides its own temp-dir backed database to avoid conflicts.
    fn fast_monitor_config(dir: &tempfile::TempDir) -> MonitorConfig {
        let mut config = MonitorConfig::default();
        config.storage.db_path = dir.path().join("monitor.db");
        config.metrics.enable_system_metrics = false;
        config.metrics.collection_interval = Duration::from_millis(100);
        config
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_monitor_creation() {
        let dir = tempdir().expect("failed to create temp dir");
        let monitor = OximediaMonitor::new(fast_monitor_config(&dir))
            .await
            .expect("operation should succeed");
        assert!(monitor.alert_manager().is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_monitor_start_stop() {
        let dir = tempdir().expect("failed to create temp dir");
        let monitor = OximediaMonitor::new(fast_monitor_config(&dir))
            .await
            .expect("operation should succeed");

        monitor.start().await.expect("await should be valid");
        assert!(monitor.metrics_collector().is_running().await);

        monitor.stop().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(!monitor.metrics_collector().is_running().await);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_collect_metrics() {
        let dir = tempdir().expect("failed to create temp dir");
        // Enable system metrics (CPU + memory) but disable disk I/O so the
        // test completes quickly on macOS with many mount points.
        let mut config = MonitorConfig::default();
        config.storage.db_path = dir.path().join("monitor.db");
        config.metrics.enable_disk_metrics = false;

        let monitor = OximediaMonitor::new(config)
            .await
            .expect("failed to create");

        let system_metrics = monitor
            .system_metrics()
            .await
            .expect("await should be valid");
        assert!(system_metrics.is_some());

        let app_metrics = monitor.application_metrics();
        assert_eq!(app_metrics.encoding.total_frames, 0);

        let quality_metrics = monitor.quality_metrics();
        assert_eq!(quality_metrics.bitrate.video_bitrate_bps, 0);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod in_memory_tests {
    use super::*;
    use std::time::Duration;

    fn fast_in_memory_config() -> MonitorConfig {
        let mut config = MonitorConfig::default();
        config.metrics.enable_system_metrics = false;
        config.metrics.collection_interval = Duration::from_millis(100);
        config
    }

    #[tokio::test]
    async fn test_in_memory_monitor_creation() {
        let monitor =
            InMemoryMonitor::new(fast_in_memory_config()).expect("should create in-memory monitor");
        assert!(monitor.alert_manager().is_some());
    }

    #[tokio::test]
    async fn test_in_memory_monitor_start_stop() {
        let monitor =
            InMemoryMonitor::new(fast_in_memory_config()).expect("should create in-memory monitor");
        monitor.start().await.expect("start should succeed");
        assert!(monitor.metrics_collector().is_running().await);
        monitor.stop().await;
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(!monitor.metrics_collector().is_running().await);
    }

    #[tokio::test]
    async fn test_in_memory_record_metric() {
        let monitor =
            InMemoryMonitor::new(fast_in_memory_config()).expect("should create in-memory monitor");
        monitor.record_metric("cpu_usage", 85.0);
        monitor.record_metric("cpu_usage", 90.0);
        monitor.record_metric("memory_usage", 72.0);
        // Ring buffer may have entries (some may still be batched).
        let _ = monitor.ring_buffer_len();
    }

    #[tokio::test]
    async fn test_in_memory_alert_rule() {
        let monitor =
            InMemoryMonitor::new(fast_in_memory_config()).expect("should create in-memory monitor");
        monitor.add_alert_rule(
            alerting_pipeline::PipelineRule::new(
                "high_cpu",
                "cpu",
                alerting_pipeline::Comparator::Gt,
                90.0,
                alerting_pipeline::Priority::Critical,
            )
            .with_silence(Duration::from_millis(0)),
        );
        monitor.record_metric("cpu", 95.0);
    }
}
