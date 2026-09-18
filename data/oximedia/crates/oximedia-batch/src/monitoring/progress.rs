//! Progress tracking implementation

use std::time::{Duration, Instant};

/// Progress tracker for individual jobs
pub struct ProgressTracker {
    total_items: u64,
    completed_items: u64,
    start_time: Instant,
    last_update: Option<Instant>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    #[must_use]
    pub fn new(total_items: u64) -> Self {
        Self {
            total_items,
            completed_items: 0,
            start_time: Instant::now(),
            last_update: None,
        }
    }

    /// Update completed items
    pub fn update(&mut self, completed_items: u64) {
        self.completed_items = completed_items.min(self.total_items);
        self.last_update = Some(Instant::now());
    }

    /// Get progress percentage
    #[must_use]
    pub fn progress_percentage(&self) -> f64 {
        if self.total_items == 0 {
            return 100.0;
        }

        #[allow(clippy::cast_precision_loss)]
        let result = (self.completed_items as f64 / self.total_items as f64) * 100.0;
        result
    }

    /// Get elapsed time
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get estimated remaining time in seconds
    #[must_use]
    pub fn estimated_remaining_secs(&self) -> Option<u64> {
        if self.completed_items == 0 {
            return None;
        }

        let elapsed_secs = self.elapsed().as_secs();
        #[allow(clippy::cast_precision_loss)]
        let rate = self.completed_items as f64 / elapsed_secs as f64;

        if rate > 0.0 {
            let remaining_items = self.total_items - self.completed_items;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            Some((remaining_items as f64 / rate) as u64)
        } else {
            None
        }
    }

    /// Get processing speed (items per second)
    #[must_use]
    pub fn processing_speed(&self) -> f64 {
        let elapsed_secs = self.elapsed().as_secs_f64();
        if elapsed_secs > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let speed = self.completed_items as f64 / elapsed_secs;
            speed
        } else {
            0.0
        }
    }

    /// Get total items
    #[must_use]
    pub const fn total_items(&self) -> u64 {
        self.total_items
    }

    /// Get completed items
    #[must_use]
    pub const fn completed_items(&self) -> u64 {
        self.completed_items
    }

    /// Check if completed
    #[must_use]
    pub const fn is_completed(&self) -> bool {
        self.completed_items >= self.total_items
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new(100);
        assert_eq!(tracker.total_items(), 100);
        assert_eq!(tracker.completed_items(), 0);
        assert!(!tracker.is_completed());
    }

    #[test]
    fn test_progress_update() {
        let mut tracker = ProgressTracker::new(100);
        tracker.update(50);

        assert_eq!(tracker.completed_items(), 50);
        assert_eq!(tracker.progress_percentage(), 50.0);
    }

    #[test]
    fn test_progress_completion() {
        let mut tracker = ProgressTracker::new(100);
        tracker.update(100);

        assert_eq!(tracker.progress_percentage(), 100.0);
        assert!(tracker.is_completed());
    }

    #[test]
    fn test_progress_over_limit() {
        let mut tracker = ProgressTracker::new(100);
        tracker.update(150);

        assert_eq!(tracker.completed_items(), 100);
        assert_eq!(tracker.progress_percentage(), 100.0);
    }

    #[test]
    fn test_zero_total_items() {
        let tracker = ProgressTracker::new(0);
        assert_eq!(tracker.progress_percentage(), 100.0);
        assert!(tracker.is_completed());
    }

    #[test]
    fn test_elapsed_time() {
        let tracker = ProgressTracker::new(100);
        thread::sleep(Duration::from_millis(10));

        let elapsed = tracker.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_processing_speed() {
        let mut tracker = ProgressTracker::new(100);
        thread::sleep(Duration::from_millis(100));
        tracker.update(50);

        let speed = tracker.processing_speed();
        assert!(speed > 0.0);
    }

    #[test]
    fn test_estimated_remaining() {
        let mut tracker = ProgressTracker::new(100);
        thread::sleep(Duration::from_millis(100));
        tracker.update(50);

        let estimated = tracker.estimated_remaining_secs();
        assert!(estimated.is_some());
    }

    #[test]
    fn test_estimated_remaining_zero_completed() {
        let tracker = ProgressTracker::new(100);
        let estimated = tracker.estimated_remaining_secs();
        assert!(estimated.is_none());
    }
}
