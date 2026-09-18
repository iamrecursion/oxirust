#![allow(dead_code)]
//! Progress tracking for long-running Python operations.
//!
//! Tracks elapsed time, throughput, and estimated completion for media
//! processing jobs that are driven from Python code.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// ProgressState
// ---------------------------------------------------------------------------

/// Current state of a tracked operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressState {
    /// The operation has not started yet.
    Pending,
    /// The operation is actively running.
    Running,
    /// The operation completed successfully.
    Completed,
    /// The operation was cancelled.
    Cancelled,
    /// The operation failed with an error.
    Failed,
}

impl std::fmt::Display for ProgressState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

// ---------------------------------------------------------------------------
// ProgressSnapshot
// ---------------------------------------------------------------------------

/// An immutable snapshot of progress at a single point in time.
#[derive(Debug, Clone)]
pub struct ProgressSnapshot {
    /// Units completed so far.
    pub completed: u64,
    /// Total units expected (if known).
    pub total: Option<u64>,
    /// Fraction completed in [0.0, 1.0] (only meaningful when total is known).
    pub fraction: f64,
    /// Current throughput in units per second.
    pub throughput: f64,
    /// Estimated seconds remaining (if total is known).
    pub eta_seconds: Option<f64>,
    /// State at the time of the snapshot.
    pub state: ProgressState,
}

// ---------------------------------------------------------------------------
// ThroughputSample
// ---------------------------------------------------------------------------

/// A single throughput measurement.
#[derive(Debug, Clone, Copy)]
struct ThroughputSample {
    /// Units processed in this sample window.
    units: u64,
    /// Duration of the sample window in seconds.
    duration_secs: f64,
}

// ---------------------------------------------------------------------------
// ThroughputEstimator
// ---------------------------------------------------------------------------

/// Rolling-window throughput estimator.
#[derive(Debug)]
pub struct ThroughputEstimator {
    /// Maximum number of samples to keep.
    window: usize,
    /// Recent samples.
    samples: VecDeque<ThroughputSample>,
}

impl ThroughputEstimator {
    /// Create an estimator with a given window size.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            samples: VecDeque::with_capacity(window),
        }
    }

    /// Record a sample of `units` processed over `duration_secs`.
    pub fn record(&mut self, units: u64, duration_secs: f64) {
        if duration_secs <= 0.0 {
            return;
        }
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(ThroughputSample {
            units,
            duration_secs,
        });
    }

    /// Current estimated throughput (units/sec) averaged over the window.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total_units: u64 = self.samples.iter().map(|s| s.units).sum();
        let total_time: f64 = self.samples.iter().map(|s| s.duration_secs).sum();
        if total_time <= 0.0 {
            0.0
        } else {
            total_units as f64 / total_time
        }
    }

    /// Number of samples currently stored.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

// ---------------------------------------------------------------------------
// ProgressTracker
// ---------------------------------------------------------------------------

/// Tracks progress of a long-running operation.
#[derive(Debug)]
pub struct ProgressTracker {
    /// Human-readable label for the operation.
    pub label: String,
    /// Total units expected (if known).
    total: Option<u64>,
    /// Units completed so far.
    completed: u64,
    /// Current state.
    state: ProgressState,
    /// Throughput estimator.
    estimator: ThroughputEstimator,
}

impl ProgressTracker {
    /// Create a new tracker with an optional total.
    pub fn new(label: impl Into<String>, total: Option<u64>) -> Self {
        Self {
            label: label.into(),
            total,
            completed: 0,
            state: ProgressState::Pending,
            estimator: ThroughputEstimator::new(10),
        }
    }

    /// Mark the operation as running.
    pub fn start(&mut self) {
        self.state = ProgressState::Running;
    }

    /// Record that `units` were processed over `duration_secs`.
    pub fn advance(&mut self, units: u64, duration_secs: f64) {
        self.completed += units;
        self.estimator.record(units, duration_secs);
    }

    /// Mark the operation as completed.
    pub fn complete(&mut self) {
        self.state = ProgressState::Completed;
    }

    /// Mark the operation as cancelled.
    pub fn cancel(&mut self) {
        self.state = ProgressState::Cancelled;
    }

    /// Mark the operation as failed.
    pub fn fail(&mut self) {
        self.state = ProgressState::Failed;
    }

    /// Take an immutable snapshot of the current progress.
    #[allow(clippy::cast_precision_loss)]
    pub fn snapshot(&self) -> ProgressSnapshot {
        let throughput = self.estimator.throughput();
        let fraction = self
            .total
            .map(|t| {
                if t == 0 {
                    1.0
                } else {
                    self.completed as f64 / t as f64
                }
            })
            .unwrap_or(0.0);
        let eta_seconds = self.total.and_then(|t| {
            if throughput <= 0.0 || self.completed >= t {
                None
            } else {
                Some((t - self.completed) as f64 / throughput)
            }
        });
        ProgressSnapshot {
            completed: self.completed,
            total: self.total,
            fraction,
            throughput,
            eta_seconds,
            state: self.state,
        }
    }

    /// Current state.
    pub fn state(&self) -> ProgressState {
        self.state
    }

    /// Units completed so far.
    pub fn completed(&self) -> u64 {
        self.completed
    }
}

// ---------------------------------------------------------------------------
// ProgressBar (text-based)
// ---------------------------------------------------------------------------

/// Simple text-based progress bar renderer.
#[derive(Debug)]
pub struct ProgressBar {
    /// Width of the bar in characters.
    pub width: usize,
    /// Character used for the filled portion.
    pub fill_char: char,
    /// Character used for the empty portion.
    pub empty_char: char,
}

impl Default for ProgressBar {
    fn default() -> Self {
        Self {
            width: 40,
            fill_char: '#',
            empty_char: '-',
        }
    }
}

impl ProgressBar {
    /// Create a progress bar with given width.
    pub fn new(width: usize) -> Self {
        Self {
            width: width.max(1),
            ..Self::default()
        }
    }

    /// Render the bar for a given fraction in [0.0, 1.0].
    pub fn render(&self, fraction: f64) -> String {
        let clamped = fraction.clamp(0.0, 1.0);
        #[allow(clippy::cast_precision_loss)]
        let filled = (clamped * self.width as f64).round() as usize;
        let empty = self.width.saturating_sub(filled);
        let fill_str: String = std::iter::repeat_n(self.fill_char, filled).collect();
        let empty_str: String = std::iter::repeat_n(self.empty_char, empty).collect();
        format!("[{fill_str}{empty_str}]")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_state_display() {
        assert_eq!(ProgressState::Pending.to_string(), "pending");
        assert_eq!(ProgressState::Running.to_string(), "running");
        assert_eq!(ProgressState::Completed.to_string(), "completed");
        assert_eq!(ProgressState::Cancelled.to_string(), "cancelled");
        assert_eq!(ProgressState::Failed.to_string(), "failed");
    }

    #[test]
    fn test_progress_state_equality() {
        assert_eq!(ProgressState::Pending, ProgressState::Pending);
        assert_ne!(ProgressState::Running, ProgressState::Completed);
    }

    #[test]
    fn test_throughput_estimator_empty() {
        let e = ThroughputEstimator::new(5);
        assert_eq!(e.throughput(), 0.0);
        assert_eq!(e.sample_count(), 0);
    }

    #[test]
    fn test_throughput_estimator_single() {
        let mut e = ThroughputEstimator::new(5);
        e.record(100, 2.0);
        assert!((e.throughput() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_estimator_window() {
        let mut e = ThroughputEstimator::new(2);
        e.record(100, 1.0);
        e.record(200, 1.0);
        e.record(300, 1.0); // evicts first
        assert_eq!(e.sample_count(), 2);
        // 200 + 300 = 500 over 2 seconds = 250
        assert!((e.throughput() - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_estimator_zero_duration() {
        let mut e = ThroughputEstimator::new(5);
        e.record(100, 0.0);
        assert_eq!(e.sample_count(), 0);
    }

    #[test]
    fn test_tracker_lifecycle() {
        let mut t = ProgressTracker::new("encode", Some(1000));
        assert_eq!(t.state(), ProgressState::Pending);
        t.start();
        assert_eq!(t.state(), ProgressState::Running);
        t.advance(500, 1.0);
        assert_eq!(t.completed(), 500);
        t.complete();
        assert_eq!(t.state(), ProgressState::Completed);
    }

    #[test]
    fn test_tracker_snapshot_with_total() {
        let mut t = ProgressTracker::new("test", Some(200));
        t.start();
        t.advance(100, 2.0);
        let snap = t.snapshot();
        assert_eq!(snap.completed, 100);
        assert_eq!(snap.total, Some(200));
        assert!((snap.fraction - 0.5).abs() < f64::EPSILON);
        assert!(snap.throughput > 0.0);
        assert!(snap.eta_seconds.is_some());
    }

    #[test]
    fn test_tracker_snapshot_no_total() {
        let mut t = ProgressTracker::new("test", None);
        t.start();
        t.advance(50, 1.0);
        let snap = t.snapshot();
        assert_eq!(snap.fraction, 0.0);
        assert!(snap.eta_seconds.is_none());
    }

    #[test]
    fn test_tracker_cancel() {
        let mut t = ProgressTracker::new("x", Some(10));
        t.start();
        t.cancel();
        assert_eq!(t.state(), ProgressState::Cancelled);
    }

    #[test]
    fn test_tracker_fail() {
        let mut t = ProgressTracker::new("x", Some(10));
        t.start();
        t.fail();
        assert_eq!(t.state(), ProgressState::Failed);
    }

    #[test]
    fn test_progress_bar_full() {
        let bar = ProgressBar::new(10);
        assert_eq!(bar.render(1.0), "[##########]");
    }

    #[test]
    fn test_progress_bar_empty() {
        let bar = ProgressBar::new(10);
        assert_eq!(bar.render(0.0), "[----------]");
    }

    #[test]
    fn test_progress_bar_half() {
        let bar = ProgressBar::new(10);
        assert_eq!(bar.render(0.5), "[#####-----]");
    }

    #[test]
    fn test_progress_bar_clamp() {
        let bar = ProgressBar::new(4);
        // >1.0 clamped to 1.0
        assert_eq!(bar.render(2.0), "[####]");
        // <0.0 clamped to 0.0
        assert_eq!(bar.render(-1.0), "[----]");
    }
}
