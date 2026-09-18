//! # DynamicChunkAdjuster - Trait Implementations
//!
//! This module contains trait implementations for `DynamicChunkAdjuster`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::types::DynamicChunkAdjuster;

impl Clone for DynamicChunkAdjuster {
    fn clone(&self) -> Self {
        Self {
            current_size: AtomicUsize::new(self.current_size.load(Ordering::Relaxed)),
            initial_size: self.initial_size,
            min_size: self.min_size,
            max_size: self.max_size,
            monitor: self.monitor.clone(),
            chunk_times: std::sync::Mutex::new(Vec::new()),
            target_time: self.target_time,
            adjustments: AtomicUsize::new(self.adjustments.load(Ordering::Relaxed)),
            enabled: AtomicBool::new(self.enabled.load(Ordering::Relaxed)),
        }
    }
}
