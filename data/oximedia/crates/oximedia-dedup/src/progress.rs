//! Progress reporting callbacks for long-running deduplication operations.
//!
//! Provides a [`ProgressReporter`] trait and concrete implementations for
//! monitoring the progress of `DuplicateDetector::find_duplicates()` and
//! similar batch operations in large media libraries.
//!
//! # Example
//!
//! ```
//! use oximedia_dedup::progress::{ProgressReporter, ProgressEvent, LoggingReporter};
//!
//! let reporter = LoggingReporter::new();
//! reporter.on_event(&ProgressEvent::PhaseStarted {
//!     phase: "exact_hash",
//!     total_items: 1000,
//! });
//! reporter.on_event(&ProgressEvent::ItemProcessed {
//!     current: 500,
//!     total: 1000,
//! });
//! reporter.on_event(&ProgressEvent::PhaseCompleted {
//!     phase: "exact_hash",
//!     groups_found: 42,
//!     elapsed_ms: 1234,
//! });
//! ```

#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Progress events
// ---------------------------------------------------------------------------

/// An event emitted during deduplication progress.
#[derive(Debug, Clone)]
pub enum ProgressEvent<'a> {
    /// A detection phase has started.
    PhaseStarted {
        /// Name of the phase (e.g., "exact_hash", "perceptual_hash", "ssim").
        phase: &'a str,
        /// Total number of items to process in this phase.
        total_items: usize,
    },

    /// A single item has been processed.
    ItemProcessed {
        /// Current item index (1-based).
        current: usize,
        /// Total items in this phase.
        total: usize,
    },

    /// A batch of items has been processed.
    BatchProcessed {
        /// Number of items in this batch.
        batch_size: usize,
        /// Cumulative items processed so far.
        cumulative: usize,
        /// Total items in this phase.
        total: usize,
    },

    /// A detection phase has completed.
    PhaseCompleted {
        /// Name of the phase.
        phase: &'a str,
        /// Number of duplicate groups found in this phase.
        groups_found: usize,
        /// Wall-clock time in milliseconds.
        elapsed_ms: u64,
    },

    /// The entire deduplication run has completed.
    RunCompleted {
        /// Total duplicate groups found across all phases.
        total_groups: usize,
        /// Total wall-clock time in milliseconds.
        total_elapsed_ms: u64,
    },

    /// An error occurred during processing (non-fatal; processing continues).
    ItemError {
        /// Item identifier (e.g., file path).
        item_id: &'a str,
        /// Error description.
        error: &'a str,
    },
}

impl ProgressEvent<'_> {
    /// Returns the progress percentage (0.0 - 100.0) if applicable.
    #[must_use]
    pub fn percentage(&self) -> Option<f64> {
        match self {
            Self::ItemProcessed { current, total } if *total > 0 => {
                Some(*current as f64 / *total as f64 * 100.0)
            }
            Self::BatchProcessed {
                cumulative, total, ..
            } if *total > 0 => Some(*cumulative as f64 / *total as f64 * 100.0),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ProgressReporter trait
// ---------------------------------------------------------------------------

/// Trait for receiving progress updates during deduplication.
///
/// Implementations should be efficient -- `on_event` may be called thousands
/// of times per second for large libraries.
pub trait ProgressReporter: Send + Sync {
    /// Called when a progress event occurs.
    fn on_event(&self, event: &ProgressEvent<'_>);

    /// Returns `true` if the operation should be cancelled.
    ///
    /// Implementations can use this to allow user-initiated cancellation.
    /// The default returns `false` (never cancel).
    fn is_cancelled(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Null reporter (no-op)
// ---------------------------------------------------------------------------

/// A no-op progress reporter that discards all events.
///
/// This is the default when no progress reporting is needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullReporter;

impl NullReporter {
    /// Create a new null reporter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ProgressReporter for NullReporter {
    fn on_event(&self, _event: &ProgressEvent<'_>) {
        // intentionally empty
    }
}

// ---------------------------------------------------------------------------
// Logging reporter
// ---------------------------------------------------------------------------

/// A progress reporter that logs events to a Vec for later inspection.
///
/// Useful for testing and batch processing where you want to review
/// progress after the fact.
#[derive(Debug, Default)]
pub struct LoggingReporter {
    /// Logged messages.
    messages: std::sync::Mutex<Vec<String>>,
}

impl LoggingReporter {
    /// Create a new logging reporter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return all logged messages.
    pub fn messages(&self) -> Vec<String> {
        self.messages
            .lock()
            .map(|msgs| msgs.clone())
            .unwrap_or_default()
    }

    /// Return the number of logged messages.
    pub fn message_count(&self) -> usize {
        self.messages.lock().map(|m| m.len()).unwrap_or(0)
    }
}

impl ProgressReporter for LoggingReporter {
    fn on_event(&self, event: &ProgressEvent<'_>) {
        let msg = match event {
            ProgressEvent::PhaseStarted { phase, total_items } => {
                format!("[START] {phase}: {total_items} items")
            }
            ProgressEvent::ItemProcessed { current, total } => {
                format!("[ITEM] {current}/{total}")
            }
            ProgressEvent::BatchProcessed {
                batch_size,
                cumulative,
                total,
            } => {
                format!("[BATCH] +{batch_size} ({cumulative}/{total})")
            }
            ProgressEvent::PhaseCompleted {
                phase,
                groups_found,
                elapsed_ms,
            } => {
                format!("[DONE] {phase}: {groups_found} groups in {elapsed_ms}ms")
            }
            ProgressEvent::RunCompleted {
                total_groups,
                total_elapsed_ms,
            } => {
                format!("[COMPLETE] {total_groups} groups in {total_elapsed_ms}ms")
            }
            ProgressEvent::ItemError { item_id, error } => {
                format!("[ERROR] {item_id}: {error}")
            }
        };

        if let Ok(mut msgs) = self.messages.lock() {
            msgs.push(msg);
        }
    }
}

// ---------------------------------------------------------------------------
// Callback reporter
// ---------------------------------------------------------------------------

/// A progress reporter backed by a user-supplied closure.
pub struct CallbackReporter<F>
where
    F: Fn(&ProgressEvent<'_>) + Send + Sync,
{
    callback: F,
    cancelled: AtomicBool,
}

impl<F> CallbackReporter<F>
where
    F: Fn(&ProgressEvent<'_>) + Send + Sync,
{
    /// Create a new callback reporter.
    pub fn new(callback: F) -> Self {
        Self {
            callback,
            cancelled: AtomicBool::new(false),
        }
    }

    /// Signal that the operation should be cancelled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

impl<F> ProgressReporter for CallbackReporter<F>
where
    F: Fn(&ProgressEvent<'_>) + Send + Sync,
{
    fn on_event(&self, event: &ProgressEvent<'_>) {
        (self.callback)(event);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Throttled reporter
// ---------------------------------------------------------------------------

/// A progress reporter that throttles `ItemProcessed` and `BatchProcessed`
/// events to at most one per `interval_ms` milliseconds.
///
/// Phase start/complete and error events are always forwarded immediately.
pub struct ThrottledReporter<R: ProgressReporter> {
    inner: R,
    interval_ms: u64,
    last_emit_ms: AtomicU64,
}

impl<R: ProgressReporter> ThrottledReporter<R> {
    /// Create a new throttled reporter wrapping `inner`.
    ///
    /// # Arguments
    /// * `inner` - The underlying reporter to forward events to.
    /// * `interval_ms` - Minimum milliseconds between forwarded progress events.
    pub fn new(inner: R, interval_ms: u64) -> Self {
        Self {
            inner,
            interval_ms,
            last_emit_ms: AtomicU64::new(0),
        }
    }

    /// Current wall-clock time in milliseconds since an arbitrary epoch.
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

impl<R: ProgressReporter> ProgressReporter for ThrottledReporter<R> {
    fn on_event(&self, event: &ProgressEvent<'_>) {
        let should_throttle = matches!(
            event,
            ProgressEvent::ItemProcessed { .. } | ProgressEvent::BatchProcessed { .. }
        );

        if should_throttle {
            let now = self.now_ms();
            let last = self.last_emit_ms.load(Ordering::Relaxed);
            if now.saturating_sub(last) < self.interval_ms {
                return;
            }
            self.last_emit_ms.store(now, Ordering::Relaxed);
        }

        self.inner.on_event(event);
    }

    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

// ---------------------------------------------------------------------------
// Multi-reporter (fan-out)
// ---------------------------------------------------------------------------

/// Forwards events to multiple reporters.
pub struct MultiReporter {
    reporters: Vec<Arc<dyn ProgressReporter>>,
}

impl MultiReporter {
    /// Create a new multi-reporter.
    #[must_use]
    pub fn new(reporters: Vec<Arc<dyn ProgressReporter>>) -> Self {
        Self { reporters }
    }
}

impl ProgressReporter for MultiReporter {
    fn on_event(&self, event: &ProgressEvent<'_>) {
        for r in &self.reporters {
            r.on_event(event);
        }
    }

    fn is_cancelled(&self) -> bool {
        self.reporters.iter().any(|r| r.is_cancelled())
    }
}

// ---------------------------------------------------------------------------
// Progress tracker (helper for emitting events)
// ---------------------------------------------------------------------------

/// Helper struct for tracking and emitting progress events from dedup algorithms.
pub struct ProgressTracker<'a> {
    reporter: &'a dyn ProgressReporter,
    phase: String,
    total: usize,
    processed: usize,
    start_time_ms: u64,
}

impl<'a> ProgressTracker<'a> {
    /// Create a new tracker for a phase.
    pub fn new(reporter: &'a dyn ProgressReporter, phase: &str, total: usize) -> Self {
        let start = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        reporter.on_event(&ProgressEvent::PhaseStarted {
            phase,
            total_items: total,
        });

        Self {
            reporter,
            phase: phase.to_string(),
            total,
            processed: 0,
            start_time_ms: start,
        }
    }

    /// Report that one item has been processed.
    pub fn tick(&mut self) {
        self.processed += 1;
        self.reporter.on_event(&ProgressEvent::ItemProcessed {
            current: self.processed,
            total: self.total,
        });
    }

    /// Report that a batch of items has been processed.
    pub fn tick_batch(&mut self, batch_size: usize) {
        self.processed += batch_size;
        self.reporter.on_event(&ProgressEvent::BatchProcessed {
            batch_size,
            cumulative: self.processed,
            total: self.total,
        });
    }

    /// Report an error for an item.
    pub fn report_error(&self, item_id: &str, error: &str) {
        self.reporter
            .on_event(&ProgressEvent::ItemError { item_id, error });
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.reporter.is_cancelled()
    }

    /// Complete the phase and emit a `PhaseCompleted` event.
    pub fn complete(self, groups_found: usize) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let elapsed = now.saturating_sub(self.start_time_ms);

        self.reporter.on_event(&ProgressEvent::PhaseCompleted {
            phase: &self.phase,
            groups_found,
            elapsed_ms: elapsed,
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_reporter() {
        let reporter = NullReporter::new();
        reporter.on_event(&ProgressEvent::PhaseStarted {
            phase: "test",
            total_items: 100,
        });
        assert!(!reporter.is_cancelled());
    }

    #[test]
    fn test_logging_reporter_captures_events() {
        let reporter = LoggingReporter::new();
        reporter.on_event(&ProgressEvent::PhaseStarted {
            phase: "exact_hash",
            total_items: 500,
        });
        reporter.on_event(&ProgressEvent::ItemProcessed {
            current: 1,
            total: 500,
        });
        reporter.on_event(&ProgressEvent::PhaseCompleted {
            phase: "exact_hash",
            groups_found: 10,
            elapsed_ms: 250,
        });

        assert_eq!(reporter.message_count(), 3);
        let msgs = reporter.messages();
        assert!(msgs[0].contains("[START]"));
        assert!(msgs[0].contains("exact_hash"));
        assert!(msgs[1].contains("[ITEM]"));
        assert!(msgs[2].contains("[DONE]"));
    }

    #[test]
    fn test_logging_reporter_error_event() {
        let reporter = LoggingReporter::new();
        reporter.on_event(&ProgressEvent::ItemError {
            item_id: "bad_file.mp4",
            error: "Permission denied",
        });
        let msgs = reporter.messages();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].contains("[ERROR]"));
        assert!(msgs[0].contains("bad_file.mp4"));
    }

    #[test]
    fn test_callback_reporter() {
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        let reporter = CallbackReporter::new(move |_event| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        reporter.on_event(&ProgressEvent::PhaseStarted {
            phase: "test",
            total_items: 10,
        });
        reporter.on_event(&ProgressEvent::ItemProcessed {
            current: 1,
            total: 10,
        });

        assert_eq!(counter.load(Ordering::Relaxed), 2);
        assert!(!reporter.is_cancelled());
    }

    #[test]
    fn test_callback_reporter_cancellation() {
        let reporter = CallbackReporter::new(|_| {});
        assert!(!reporter.is_cancelled());
        reporter.cancel();
        assert!(reporter.is_cancelled());
    }

    #[test]
    fn test_progress_event_percentage() {
        let event = ProgressEvent::ItemProcessed {
            current: 50,
            total: 200,
        };
        let pct = event.percentage().expect("should have percentage");
        assert!((pct - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_event_percentage_batch() {
        let event = ProgressEvent::BatchProcessed {
            batch_size: 10,
            cumulative: 75,
            total: 100,
        };
        let pct = event.percentage().expect("should have percentage");
        assert!((pct - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_progress_event_no_percentage_for_phase() {
        let event = ProgressEvent::PhaseStarted {
            phase: "test",
            total_items: 100,
        };
        assert!(event.percentage().is_none());
    }

    #[test]
    fn test_multi_reporter() {
        let r1 = Arc::new(LoggingReporter::new());
        let r2 = Arc::new(LoggingReporter::new());

        let multi = MultiReporter::new(vec![r1.clone(), r2.clone()]);
        multi.on_event(&ProgressEvent::PhaseStarted {
            phase: "test",
            total_items: 50,
        });

        assert_eq!(r1.message_count(), 1);
        assert_eq!(r2.message_count(), 1);
        assert!(!multi.is_cancelled());
    }

    #[test]
    fn test_progress_tracker_lifecycle() {
        let reporter = LoggingReporter::new();
        let mut tracker = ProgressTracker::new(&reporter, "scan", 3);

        tracker.tick();
        tracker.tick();
        tracker.tick();
        assert!(!tracker.is_cancelled());
        tracker.complete(1);

        let msgs = reporter.messages();
        // 1 start + 3 items + 1 complete = 5 messages
        assert_eq!(msgs.len(), 5);
        assert!(msgs[0].contains("[START]"));
        assert!(msgs[4].contains("[DONE]"));
    }

    #[test]
    fn test_progress_tracker_batch() {
        let reporter = LoggingReporter::new();
        let mut tracker = ProgressTracker::new(&reporter, "index", 100);

        tracker.tick_batch(25);
        tracker.tick_batch(25);
        tracker.complete(5);

        let msgs = reporter.messages();
        assert_eq!(msgs.len(), 4); // start + 2 batches + complete
        assert!(msgs[1].contains("[BATCH]"));
    }

    #[test]
    fn test_progress_tracker_error_reporting() {
        let reporter = LoggingReporter::new();
        let tracker = ProgressTracker::new(&reporter, "scan", 10);

        tracker.report_error("corrupt.mp4", "invalid header");
        tracker.complete(0);

        let msgs = reporter.messages();
        assert!(msgs.iter().any(|m| m.contains("[ERROR]")));
        assert!(msgs.iter().any(|m| m.contains("corrupt.mp4")));
    }

    #[test]
    fn test_throttled_reporter_forwards_phase_events() {
        let inner = LoggingReporter::new();
        let throttled = ThrottledReporter::new(inner, 1000);

        // Phase events should always be forwarded
        throttled.on_event(&ProgressEvent::PhaseStarted {
            phase: "test",
            total_items: 100,
        });
        throttled.on_event(&ProgressEvent::PhaseCompleted {
            phase: "test",
            groups_found: 5,
            elapsed_ms: 500,
        });

        assert_eq!(throttled.inner.message_count(), 2);
    }

    #[test]
    fn test_throttled_reporter_throttles_items() {
        let inner = LoggingReporter::new();
        // Very long interval to ensure throttling
        let throttled = ThrottledReporter::new(inner, 60_000);

        // First item should be forwarded
        throttled.on_event(&ProgressEvent::ItemProcessed {
            current: 1,
            total: 100,
        });
        // Subsequent items within the interval should be throttled
        throttled.on_event(&ProgressEvent::ItemProcessed {
            current: 2,
            total: 100,
        });
        throttled.on_event(&ProgressEvent::ItemProcessed {
            current: 3,
            total: 100,
        });

        // Only 1 item event should have been forwarded
        assert_eq!(throttled.inner.message_count(), 1);
    }

    #[test]
    fn test_run_completed_event() {
        let reporter = LoggingReporter::new();
        reporter.on_event(&ProgressEvent::RunCompleted {
            total_groups: 15,
            total_elapsed_ms: 5000,
        });
        let msgs = reporter.messages();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].contains("[COMPLETE]"));
        assert!(msgs[0].contains("15 groups"));
    }
}
