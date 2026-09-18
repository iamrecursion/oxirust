//! Memory-efficient partitioned tensor reductions.
//!
//! Reductions are processed in configurable chunks to bound peak memory usage
//! independently of total tensor size.

pub mod config;
pub mod reducer;

pub use config::{AccumulationStrategy, PartitionConfig};
pub use reducer::{PartitionedError, PartitionedReducer, PartitionedStats};
