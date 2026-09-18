//! # AdaptiveChunkingParams - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveChunkingParams`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::memory_efficient::adaptive_feedback::SharedPredictor;
use std::time::{Duration, Instant};

use super::functions::SharedMemoryMonitor;
use super::types::{AdaptiveChunkingParams, MemoryLimits};

impl std::fmt::Debug for AdaptiveChunkingParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveChunkingParams")
            .field("target_memory_usage", &self.target_memory_usage)
            .field("max_chunksize", &self.max_chunksize)
            .field("min_chunksize", &self.min_chunksize)
            .field("target_chunk_duration", &self.target_chunk_duration)
            .field("consider_distribution", &self.consider_distribution)
            .field("optimize_for_parallel", &self.optimize_for_parallel)
            .field("numworkers", &self.numworkers)
            .field(
                "predictor",
                &self.predictor.as_ref().map(|_| "Some(SharedPredictor)"),
            )
            .field("memory_limits", &self.memory_limits)
            .field("enable_oom_prevention", &self.enable_oom_prevention)
            .field("enable_dynamic_adjustment", &self.enable_dynamic_adjustment)
            .field(
                "memory_monitor",
                &self
                    .memory_monitor
                    .as_ref()
                    .map(|_| "Some(SharedMemoryMonitor)"),
            )
            .finish()
    }
}

impl Default for AdaptiveChunkingParams {
    fn default() -> Self {
        let available_memory = Self::detect_available_memory();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let target_memory = if let Some(mem) = available_memory {
            (mem / 8).clamp(16 * 1024 * 1024, 256 * 1024 * 1024)
        } else {
            64 * 1024 * 1024
        };
        Self {
            target_memory_usage: target_memory,
            max_chunksize: usize::MAX,
            min_chunksize: 1024,
            target_chunk_duration: Some(Duration::from_millis(100)),
            consider_distribution: true,
            optimize_for_parallel: cpu_cores > 1,
            numworkers: Some(cpu_cores),
            predictor: None,
            memory_limits: Some(MemoryLimits::auto_detect()),
            enable_oom_prevention: true,
            enable_dynamic_adjustment: true,
            memory_monitor: None,
        }
    }
}
