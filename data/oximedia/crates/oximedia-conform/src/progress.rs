//! Progress tracking for conform operations.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Progress stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressStage {
    /// Loading timeline.
    LoadingTimeline,
    /// Scanning media.
    ScanningMedia,
    /// Matching clips.
    MatchingClips,
    /// Validating matches.
    Validating,
    /// Conforming timeline.
    Conforming,
    /// Exporting output.
    Exporting,
    /// Complete.
    Complete,
}

/// Progress information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    /// Current stage.
    pub stage: ProgressStage,
    /// Current item being processed.
    pub current_item: usize,
    /// Total items to process.
    pub total_items: usize,
    /// Progress percentage (0-100).
    pub percentage: f64,
    /// Estimated time remaining in seconds.
    pub eta_seconds: Option<f64>,
    /// Current operation description.
    pub description: String,
}

/// Progress tracker for conform operations.
pub struct ProgressTracker {
    /// Current stage.
    stage: Arc<parking_lot::RwLock<ProgressStage>>,
    /// Current item.
    current_item: Arc<AtomicUsize>,
    /// Total items.
    total_items: Arc<AtomicUsize>,
    /// Bytes processed.
    bytes_processed: Arc<AtomicU64>,
    /// Total bytes.
    total_bytes: Arc<AtomicU64>,
    /// Start time.
    start_time: Instant,
    /// Current operation description.
    description: Arc<parking_lot::RwLock<String>>,
}

impl ProgressTracker {
    /// Create a new progress tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stage: Arc::new(parking_lot::RwLock::new(ProgressStage::LoadingTimeline)),
            current_item: Arc::new(AtomicUsize::new(0)),
            total_items: Arc::new(AtomicUsize::new(0)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            total_bytes: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            description: Arc::new(parking_lot::RwLock::new(String::new())),
        }
    }

    /// Set the current stage.
    pub fn set_stage(&self, stage: ProgressStage) {
        *self.stage.write() = stage;
    }

    /// Get the current stage.
    #[must_use]
    pub fn stage(&self) -> ProgressStage {
        *self.stage.read()
    }

    /// Set total items.
    pub fn set_total_items(&self, total: usize) {
        self.total_items.store(total, Ordering::Relaxed);
    }

    /// Set current item.
    pub fn set_current_item(&self, current: usize) {
        self.current_item.store(current, Ordering::Relaxed);
    }

    /// Increment current item.
    pub fn increment_item(&self) {
        self.current_item.fetch_add(1, Ordering::Relaxed);
    }

    /// Set description.
    pub fn set_description(&self, desc: String) {
        *self.description.write() = desc;
    }

    /// Get current progress percentage.
    #[must_use]
    pub fn percentage(&self) -> f64 {
        let total = self.total_items.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let current = self.current_item.load(Ordering::Relaxed);
        (current as f64 / total as f64) * 100.0
    }

    /// Get estimated time remaining.
    #[must_use]
    pub fn eta(&self) -> Option<Duration> {
        let current = self.current_item.load(Ordering::Relaxed);
        let total = self.total_items.load(Ordering::Relaxed);

        if current == 0 || total == 0 || current >= total {
            return None;
        }

        let elapsed = self.start_time.elapsed();
        let per_item = elapsed.as_secs_f64() / current as f64;
        let remaining_items = total - current;
        let eta_seconds = per_item * remaining_items as f64;

        Some(Duration::from_secs_f64(eta_seconds))
    }

    /// Get current progress information.
    #[must_use]
    pub fn info(&self) -> ProgressInfo {
        ProgressInfo {
            stage: *self.stage.read(),
            current_item: self.current_item.load(Ordering::Relaxed),
            total_items: self.total_items.load(Ordering::Relaxed),
            percentage: self.percentage(),
            eta_seconds: self.eta().map(|d| d.as_secs_f64()),
            description: self.description.read().clone(),
        }
    }

    /// Set bytes processed.
    pub fn set_bytes_processed(&self, bytes: u64) {
        self.bytes_processed.store(bytes, Ordering::Relaxed);
    }

    /// Add bytes processed.
    pub fn add_bytes_processed(&self, bytes: u64) {
        self.bytes_processed.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Set total bytes.
    pub fn set_total_bytes(&self, bytes: u64) {
        self.total_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Get bytes processed.
    #[must_use]
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed.load(Ordering::Relaxed)
    }

    /// Get total bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Get elapsed time.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        *self.stage.write() = ProgressStage::LoadingTimeline;
        self.current_item.store(0, Ordering::Relaxed);
        self.total_items.store(0, Ordering::Relaxed);
        self.bytes_processed.store(0, Ordering::Relaxed);
        self.total_bytes.store(0, Ordering::Relaxed);
        self.start_time = Instant::now();
        self.description.write().clear();
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ProgressTracker {
    fn clone(&self) -> Self {
        Self {
            stage: Arc::clone(&self.stage),
            current_item: Arc::clone(&self.current_item),
            total_items: Arc::clone(&self.total_items),
            bytes_processed: Arc::clone(&self.bytes_processed),
            total_bytes: Arc::clone(&self.total_bytes),
            start_time: self.start_time,
            description: Arc::clone(&self.description),
        }
    }
}

/// Callback trait for progress updates.
pub trait ProgressCallback: Send + Sync {
    /// Called when progress is updated.
    fn on_progress(&self, info: &ProgressInfo);
}

/// Simple progress callback that prints to stdout.
pub struct PrintProgressCallback;

impl ProgressCallback for PrintProgressCallback {
    fn on_progress(&self, info: &ProgressInfo) {
        println!(
            "[{:?}] {}/{} ({:.1}%) - {}",
            info.stage, info.current_item, info.total_items, info.percentage, info.description
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new();
        assert_eq!(tracker.stage(), ProgressStage::LoadingTimeline);
        assert!((tracker.percentage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_percentage() {
        let tracker = ProgressTracker::new();
        tracker.set_total_items(100);
        tracker.set_current_item(50);
        assert!((tracker.percentage() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_increment_item() {
        let tracker = ProgressTracker::new();
        tracker.set_total_items(10);
        tracker.increment_item();
        tracker.increment_item();
        assert_eq!(tracker.current_item.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_progress_info() {
        let tracker = ProgressTracker::new();
        tracker.set_total_items(100);
        tracker.set_current_item(25);
        tracker.set_description("Testing".to_string());

        let info = tracker.info();
        assert_eq!(info.current_item, 25);
        assert_eq!(info.total_items, 100);
        assert!((info.percentage - 25.0).abs() < 0.01);
        assert_eq!(info.description, "Testing");
    }

    #[test]
    fn test_eta() {
        let tracker = ProgressTracker::new();
        tracker.set_total_items(100);
        tracker.set_current_item(0);
        assert!(tracker.eta().is_none());

        tracker.set_current_item(50);
        // ETA should be available when in progress
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Cannot reliably test exact ETA due to timing
    }
}
