//! Performance optimization modules
//!
//! This module contains advanced optimization techniques including
//! dynamic load balancing and NUMA-aware scheduling.

pub mod load_balancing;
pub mod numa_awareness;

// Re-export main types
pub use load_balancing::{DynamicLoadBalancer, LoadBalancingStats};
pub use numa_awareness::{AdvancedWorkStealingScheduler, NumaTopology};