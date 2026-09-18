#![allow(dead_code)]
//! Output collection — gather, classify, and summarise task output entries.

/// A single captured output entry from a batch task.
#[derive(Debug, Clone)]
pub struct OutputEntry {
    /// The task that produced this output.
    pub task_name: String,
    /// The captured text content.
    pub content: String,
    /// Whether this entry represents an error condition.
    pub error: bool,
}

impl OutputEntry {
    /// Create a normal (non-error) output entry.
    #[must_use]
    pub fn new(task_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            task_name: task_name.into(),
            content: content.into(),
            error: false,
        }
    }

    /// Create an error output entry.
    #[must_use]
    pub fn error(task_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            task_name: task_name.into(),
            content: content.into(),
            error: true,
        }
    }

    /// Returns `true` if this entry is an error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.error
    }

    /// Returns the byte length of the content.
    #[must_use]
    pub fn content_len(&self) -> usize {
        self.content.len()
    }
}

/// Accumulates output entries produced by batch tasks.
#[derive(Debug, Clone, Default)]
pub struct OutputCollector {
    entries: Vec<OutputEntry>,
}

impl OutputCollector {
    /// Create a new, empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an output entry into the collector.
    pub fn push(&mut self, entry: OutputEntry) {
        self.entries.push(entry);
    }

    /// Return all entries that are errors.
    #[must_use]
    pub fn errors(&self) -> Vec<&OutputEntry> {
        self.entries.iter().filter(|e| e.is_error()).collect()
    }

    /// Return all entries that are not errors.
    #[must_use]
    pub fn successes(&self) -> Vec<&OutputEntry> {
        self.entries.iter().filter(|e| !e.is_error()).collect()
    }

    /// Total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries have been collected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all entries for a specific task.
    #[must_use]
    pub fn entries_for(&self, task_name: &str) -> Vec<&OutputEntry> {
        self.entries
            .iter()
            .filter(|e| e.task_name == task_name)
            .collect()
    }

    /// Produce a summary view of this collector.
    #[must_use]
    pub fn summarize(&self) -> OutputSummary {
        let total = self.entries.len();
        let error_count = self.errors().len();
        let success_count = self.successes().len();
        OutputSummary {
            total,
            error_count,
            success_count,
        }
    }

    /// Return a reference to the raw entries slice.
    #[must_use]
    pub fn all_entries(&self) -> &[OutputEntry] {
        &self.entries
    }
}

/// A lightweight summary of what an [`OutputCollector`] captured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSummary {
    /// Total entries collected.
    pub total: usize,
    /// Number of error entries.
    pub error_count: usize,
    /// Number of success (non-error) entries.
    pub success_count: usize,
}

impl OutputSummary {
    /// Fraction of entries that are errors, in `[0.0, 1.0]`.
    /// Returns `0.0` when `total` is zero.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.error_count as f64 / self.total as f64
    }

    /// Returns `true` if there are no errors.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.error_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_entry_new_not_error() {
        let e = OutputEntry::new("task1", "some output");
        assert!(!e.is_error());
        assert_eq!(e.task_name, "task1");
    }

    #[test]
    fn test_output_entry_error_is_error() {
        let e = OutputEntry::error("task2", "fatal error");
        assert!(e.is_error());
    }

    #[test]
    fn test_output_entry_content_len() {
        let e = OutputEntry::new("t", "hello");
        assert_eq!(e.content_len(), 5);
    }

    #[test]
    fn test_collector_push_and_len() {
        let mut c = OutputCollector::new();
        assert!(c.is_empty());
        c.push(OutputEntry::new("t", "ok"));
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());
    }

    #[test]
    fn test_collector_errors_and_successes() {
        let mut c = OutputCollector::new();
        c.push(OutputEntry::new("t1", "ok"));
        c.push(OutputEntry::error("t2", "bad"));
        c.push(OutputEntry::new("t3", "ok2"));
        assert_eq!(c.errors().len(), 1);
        assert_eq!(c.successes().len(), 2);
    }

    #[test]
    fn test_collector_entries_for() {
        let mut c = OutputCollector::new();
        c.push(OutputEntry::new("alpha", "a1"));
        c.push(OutputEntry::new("beta", "b1"));
        c.push(OutputEntry::new("alpha", "a2"));
        let alpha = c.entries_for("alpha");
        assert_eq!(alpha.len(), 2);
    }

    #[test]
    fn test_collector_all_entries_slice() {
        let mut c = OutputCollector::new();
        c.push(OutputEntry::new("t", "x"));
        assert_eq!(c.all_entries().len(), 1);
    }

    #[test]
    fn test_summarize_total_counts() {
        let mut c = OutputCollector::new();
        c.push(OutputEntry::new("t", "ok"));
        c.push(OutputEntry::error("t", "err"));
        let s = c.summarize();
        assert_eq!(s.total, 2);
        assert_eq!(s.error_count, 1);
        assert_eq!(s.success_count, 1);
    }

    #[test]
    fn test_output_summary_error_rate_non_zero() {
        let s = OutputSummary {
            total: 4,
            error_count: 1,
            success_count: 3,
        };
        let rate = s.error_rate();
        assert!((rate - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_output_summary_error_rate_zero_total() {
        let s = OutputSummary {
            total: 0,
            error_count: 0,
            success_count: 0,
        };
        assert_eq!(s.error_rate(), 0.0);
    }

    #[test]
    fn test_output_summary_is_clean_true() {
        let s = OutputSummary {
            total: 3,
            error_count: 0,
            success_count: 3,
        };
        assert!(s.is_clean());
    }

    #[test]
    fn test_output_summary_is_clean_false() {
        let s = OutputSummary {
            total: 3,
            error_count: 1,
            success_count: 2,
        };
        assert!(!s.is_clean());
    }

    #[test]
    fn test_collector_default_is_empty() {
        let c = OutputCollector::default();
        assert!(c.is_empty());
    }
}
