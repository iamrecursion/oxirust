//! Step result types for workflow execution tracking.
//!
//! Provides `StepStatus`, `StepResult`, and `StepResultLog` for
//! recording and querying per-step outcomes in a workflow run.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// StepStatus
// ---------------------------------------------------------------------------

/// The execution status of a single workflow step.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StepStatus {
    /// The step has not yet started.
    Pending,
    /// The step is currently executing.
    Running,
    /// The step completed successfully.
    Success,
    /// The step completed with an error.
    Failed,
    /// The step was intentionally skipped.
    Skipped,
}

impl StepStatus {
    /// Returns `true` if no further execution can follow from this status.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Success | Self::Failed | Self::Skipped)
    }

    /// Human-readable label for the status.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }
}

// ---------------------------------------------------------------------------
// StepResult
// ---------------------------------------------------------------------------

/// The outcome of a single step execution.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Unique identifier for the step.
    pub step_id: String,
    /// Final status of the step.
    pub status: StepStatus,
    /// Wall-clock milliseconds from start to finish.
    elapsed_ms: u64,
    /// Optional human-readable message.
    pub message: Option<String>,
}

impl StepResult {
    /// Create a new `StepResult`.
    #[must_use]
    pub fn new(step_id: impl Into<String>, status: StepStatus, elapsed_ms: u64) -> Self {
        Self {
            step_id: step_id.into(),
            status,
            elapsed_ms,
            message: None,
        }
    }

    /// Attach a message to this result.
    #[must_use]
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Wall-clock elapsed time in milliseconds.
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed_ms
    }

    /// Convenience: was this step successful?
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status == StepStatus::Success
    }

    /// Convenience: did this step fail?
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.status == StepStatus::Failed
    }
}

// ---------------------------------------------------------------------------
// StepResultLog
// ---------------------------------------------------------------------------

/// A collection of `StepResult`s grouped by workflow run identifier.
#[derive(Debug, Default)]
pub struct StepResultLog {
    entries: HashMap<String, Vec<StepResult>>,
}

impl StepResultLog {
    /// Create an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a result for the given `run_id`.
    pub fn add(&mut self, run_id: impl Into<String>, result: StepResult) {
        self.entries.entry(run_id.into()).or_default().push(result);
    }

    /// Fraction of terminal steps that succeeded for a given run.
    ///
    /// Returns `None` if the run is unknown or has no terminal steps.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn success_rate(&self, run_id: &str) -> Option<f64> {
        let results = self.entries.get(run_id)?;
        let terminal: Vec<_> = results.iter().filter(|r| r.status.is_terminal()).collect();
        if terminal.is_empty() {
            return None;
        }
        let successes = terminal.iter().filter(|r| r.is_success()).count();
        Some(successes as f64 / terminal.len() as f64)
    }

    /// Return only the failed steps for a given run.
    #[must_use]
    pub fn failed_steps(&self, run_id: &str) -> Vec<&StepResult> {
        self.entries
            .get(run_id)
            .map(|v| v.iter().filter(|r| r.is_failed()).collect())
            .unwrap_or_default()
    }

    /// Total number of step results recorded across all runs.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }

    /// All run identifiers currently in the log.
    #[must_use]
    pub fn run_ids(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_pending_not_terminal() {
        assert!(!StepStatus::Pending.is_terminal());
    }

    #[test]
    fn test_status_running_not_terminal() {
        assert!(!StepStatus::Running.is_terminal());
    }

    #[test]
    fn test_status_success_is_terminal() {
        assert!(StepStatus::Success.is_terminal());
    }

    #[test]
    fn test_status_failed_is_terminal() {
        assert!(StepStatus::Failed.is_terminal());
    }

    #[test]
    fn test_status_skipped_is_terminal() {
        assert!(StepStatus::Skipped.is_terminal());
    }

    #[test]
    fn test_status_labels() {
        assert_eq!(StepStatus::Pending.label(), "pending");
        assert_eq!(StepStatus::Running.label(), "running");
        assert_eq!(StepStatus::Success.label(), "success");
        assert_eq!(StepStatus::Failed.label(), "failed");
        assert_eq!(StepStatus::Skipped.label(), "skipped");
    }

    #[test]
    fn test_step_result_elapsed_ms() {
        let r = StepResult::new("step1", StepStatus::Success, 250);
        assert_eq!(r.elapsed_ms(), 250);
    }

    #[test]
    fn test_step_result_is_success() {
        let r = StepResult::new("step1", StepStatus::Success, 10);
        assert!(r.is_success());
        assert!(!r.is_failed());
    }

    #[test]
    fn test_step_result_is_failed() {
        let r = StepResult::new("step2", StepStatus::Failed, 5);
        assert!(r.is_failed());
        assert!(!r.is_success());
    }

    #[test]
    fn test_step_result_with_message() {
        let r = StepResult::new("s", StepStatus::Failed, 0).with_message("oops");
        assert_eq!(r.message.as_deref(), Some("oops"));
    }

    #[test]
    fn test_log_add_and_total_entries() {
        let mut log = StepResultLog::new();
        log.add("run1", StepResult::new("a", StepStatus::Success, 1));
        log.add("run1", StepResult::new("b", StepStatus::Failed, 2));
        log.add("run2", StepResult::new("c", StepStatus::Skipped, 0));
        assert_eq!(log.total_entries(), 3);
    }

    #[test]
    fn test_success_rate_all_success() {
        let mut log = StepResultLog::new();
        log.add("r", StepResult::new("a", StepStatus::Success, 1));
        log.add("r", StepResult::new("b", StepStatus::Success, 2));
        let rate = log.success_rate("r").expect("should succeed in test");
        assert!((rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_mixed() {
        let mut log = StepResultLog::new();
        log.add("r", StepResult::new("a", StepStatus::Success, 1));
        log.add("r", StepResult::new("b", StepStatus::Failed, 2));
        let rate = log.success_rate("r").expect("should succeed in test");
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_unknown_run() {
        let log = StepResultLog::new();
        assert!(log.success_rate("unknown").is_none());
    }

    #[test]
    fn test_failed_steps() {
        let mut log = StepResultLog::new();
        log.add("r", StepResult::new("ok", StepStatus::Success, 1));
        log.add("r", StepResult::new("bad", StepStatus::Failed, 2));
        let failed = log.failed_steps("r");
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].step_id, "bad");
    }

    #[test]
    fn test_run_ids() {
        let mut log = StepResultLog::new();
        log.add("alpha", StepResult::new("x", StepStatus::Success, 0));
        log.add("beta", StepResult::new("y", StepStatus::Success, 0));
        let mut ids = log.run_ids();
        ids.sort_unstable();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_success_rate_no_terminal_steps() {
        let mut log = StepResultLog::new();
        log.add("r", StepResult::new("x", StepStatus::Running, 0));
        assert!(log.success_rate("r").is_none());
    }
}
