//! Partition optimization module.
//!
//! Block size selection and complexity-based partitioning.

pub mod complexity;
pub mod split;

pub use complexity::{BlockComplexity, ComplexityAnalyzer};
pub use split::{PartitionDecision, PartitionMode, SplitOptimizer};
