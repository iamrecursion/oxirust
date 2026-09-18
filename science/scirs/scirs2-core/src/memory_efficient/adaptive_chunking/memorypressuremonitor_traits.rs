//! # MemoryPressureMonitor - Trait Implementations
//!
//! This module contains trait implementations for `MemoryPressureMonitor`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::types::MemoryPressureMonitor;

impl Clone for MemoryPressureMonitor {
    fn clone(&self) -> Self {
        Self {
            limits: self.limits.clone(),
            current_level: AtomicUsize::new(self.current_level.load(Ordering::Relaxed)),
            active: AtomicBool::new(self.active.load(Ordering::Relaxed)),
            last_check: std::sync::Mutex::new(Instant::now()),
            check_interval: self.check_interval,
            pressure_history: std::sync::Mutex::new(Vec::new()),
            max_history: self.max_history,
        }
    }
}
