//! Continuous monitoring functionality

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Continuous monitoring system
pub struct ContinuousMonitor {
    running: Arc<AtomicBool>,
    interval: Duration,
}

impl ContinuousMonitor {
    /// Create new continuous monitor
    pub fn new(interval: Duration) -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            interval,
        }
    }

    /// Start monitoring
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    /// Stop monitoring
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}
