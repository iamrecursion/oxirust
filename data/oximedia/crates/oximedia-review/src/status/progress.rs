//! Progress calculation and tracking.

use serde::{Deserialize, Serialize};

/// Progress tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressTracker {
    /// Total items.
    pub total: usize,
    /// Completed items.
    pub completed: usize,
    /// In progress items.
    pub in_progress: usize,
    /// Blocked items.
    pub blocked: usize,
}

impl ProgressTracker {
    /// Create a new progress tracker.
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self {
            total,
            completed: 0,
            in_progress: 0,
            blocked: 0,
        }
    }

    /// Record completion of an item.
    pub fn complete_item(&mut self) {
        if self.in_progress > 0 {
            self.in_progress -= 1;
        }
        self.completed += 1;
    }

    /// Start an item.
    pub fn start_item(&mut self) {
        self.in_progress += 1;
    }

    /// Block an item.
    pub fn block_item(&mut self) {
        if self.in_progress > 0 {
            self.in_progress -= 1;
        }
        self.blocked += 1;
    }

    /// Unblock an item.
    pub fn unblock_item(&mut self) {
        if self.blocked > 0 {
            self.blocked -= 1;
            self.in_progress += 1;
        }
    }

    /// Calculate completion percentage.
    #[must_use]
    pub fn completion_percentage(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }

        (self.completed as f64 / self.total as f64) * 100.0
    }

    /// Check if all items are complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed >= self.total
    }

    /// Get remaining items.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.total, 10);
        assert_eq!(tracker.completed, 0);
        assert!((tracker.completion_percentage() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_progress_tracker_complete_item() {
        let mut tracker = ProgressTracker::new(5);
        tracker.start_item();
        tracker.complete_item();

        assert_eq!(tracker.completed, 1);
        assert_eq!(tracker.in_progress, 0);
        assert!((tracker.completion_percentage() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_progress_tracker_block_unblock() {
        let mut tracker = ProgressTracker::new(5);
        tracker.start_item();
        tracker.block_item();

        assert_eq!(tracker.blocked, 1);
        assert_eq!(tracker.in_progress, 0);

        tracker.unblock_item();
        assert_eq!(tracker.blocked, 0);
        assert_eq!(tracker.in_progress, 1);
    }

    #[test]
    fn test_progress_tracker_is_complete() {
        let mut tracker = ProgressTracker::new(2);
        assert!(!tracker.is_complete());

        tracker.complete_item();
        tracker.complete_item();
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_progress_tracker_remaining() {
        let mut tracker = ProgressTracker::new(10);
        assert_eq!(tracker.remaining(), 10);

        tracker.complete_item();
        assert_eq!(tracker.remaining(), 9);
    }
}
