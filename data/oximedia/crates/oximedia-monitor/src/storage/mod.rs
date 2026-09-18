//! Time series storage for metrics.

pub mod aggregation;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod query;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod retention;
pub mod ringbuffer;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub mod sqlite;

pub use aggregation::{AggregateFunction, AggregatedMetric, Aggregator};
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub use query::{QueryEngine, TimeRange, TimeSeriesQuery, TimeSeriesResult};
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub use retention::RetentionManager;
pub use ringbuffer::RingBuffer;
#[cfg(all(not(target_arch = "wasm32"), feature = "sqlite"))]
pub use sqlite::{SqliteStorage, TimeSeriesPoint};
