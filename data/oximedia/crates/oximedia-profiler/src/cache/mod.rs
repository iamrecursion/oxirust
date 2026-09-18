//! Cache analysis modules.

pub mod analyze;
pub mod miss;

pub use analyze::{CacheAnalyzer, CacheStats};
pub use miss::{CacheMissProfiler, MissPattern};
