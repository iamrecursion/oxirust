//! # AdaptiveMemoryManager - accessors Methods
//!
//! This module contains method implementations for `AdaptiveMemoryManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::{PlatformCapabilities, SimdUnifiedOps},
};

use super::types::MemoryUsageStatistics;

use super::adaptivememorymanager_type::AdaptiveMemoryManager;

impl<F> AdaptiveMemoryManager<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + 'static
        + std::fmt::Display,
{
    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryUsageStatistics {
        MemoryUsageStatistics {
            total_allocated: self.calculate_total_allocated(),
            peak_allocated: self.calculate_peak_allocated(),
            fragmentation_ratio: self.calculate_fragmentation(),
            cache_hit_ratio: self.cache_manager.get_hit_ratio(),
            numa_efficiency: self.numa_manager.get_efficiency(),
            gc_overhead: self.gc_manager.get_overhead(),
            pressure_level: self.pressure_monitor.get_current_pressure(),
            out_of_core_ratio: self.out_of_core_manager.get_ratio(),
        }
    }
    /// Calculate total allocated memory
    pub fn calculate_total_allocated(&self) -> usize {
        self.memory_pools
            .read()
            .expect("Operation failed")
            .values()
            .map(|pool| pool.get_allocatedsize())
            .sum()
    }
    /// Calculate peak allocated memory
    pub fn calculate_peak_allocated(&self) -> usize {
        self.memory_pools
            .read()
            .expect("Operation failed")
            .values()
            .map(|pool| pool.get_peaksize())
            .max()
            .unwrap_or(0)
    }
    /// Calculate memory fragmentation ratio
    pub fn calculate_fragmentation(&self) -> f64 {
        let total_allocated = self.calculate_total_allocated() as f64;
        let total_requested = self.calculate_total_requested() as f64;
        if total_requested > 0.0 {
            (total_allocated - total_requested) / total_allocated
        } else {
            0.0
        }
    }
}
