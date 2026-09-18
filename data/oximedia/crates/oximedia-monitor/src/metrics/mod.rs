//! Metrics collection and management.

pub mod application;
#[cfg(not(target_arch = "wasm32"))]
pub mod collector;
pub mod quality;
pub mod registry;
#[cfg(not(target_arch = "wasm32"))]
pub mod system;
pub mod types;

pub use application::{
    ApplicationMetrics, ApplicationMetricsTracker, EncodingMetrics, JobMetrics, WorkerMetrics,
    WorkerStatus,
};
#[cfg(not(target_arch = "wasm32"))]
pub use collector::MetricsCollector;
pub use quality::{BitrateMetrics, QualityMetrics, QualityMetricsTracker, QualityScore};
pub use registry::{MetricRegistry, MetricValue};
#[cfg(all(not(target_arch = "wasm32"), feature = "gpu"))]
pub use system::GpuMetrics;
#[cfg(not(target_arch = "wasm32"))]
pub use system::{
    CpuMetrics, DiskMetrics, MemoryMetrics, NetworkMetrics, SystemMetrics, SystemMetricsCollector,
    SystemMetricsConfig, TemperatureMetrics,
};
pub use types::{Counter, Gauge, Histogram, Metric, MetricKind, MetricLabels, MetricName, Summary};
