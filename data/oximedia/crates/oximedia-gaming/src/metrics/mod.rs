//! Performance metrics and stream statistics.

pub mod performance;
pub mod stats;

pub use performance::{PerformanceMetrics, PerformanceMonitor};
pub use stats::{StatisticsCollector, StreamStatistics};
