//! Batch job reporting and summary aggregation.
//!
//! Provides per-job results and rolled-up summary statistics
//! for a collection of batch reports.

#![allow(dead_code)]

/// Outcome of a single batch job execution.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchResult {
    /// Job completed without errors.
    Success,
    /// Job failed with the given error message.
    Failure(String),
    /// Job was cancelled before completion.
    Cancelled,
    /// Job was partially successful (some outputs produced).
    Partial {
        /// Number of outputs successfully completed.
        completed: u32,
        /// Total number of expected outputs.
        total: u32,
    },
}

impl BatchResult {
    /// Returns `true` if the result is `Success`.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` if the result is `Failure` or `Cancelled`.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failure(_) | Self::Cancelled)
    }
}

/// Report for a single batch job.
#[derive(Debug, Clone)]
pub struct BatchJobReport {
    /// Job identifier
    pub job_id: String,
    /// Job label / description
    pub label: String,
    /// Outcome of the job
    pub result: BatchResult,
    /// Wall-clock duration in milliseconds
    pub duration_ms: u64,
}

impl BatchJobReport {
    /// Creates a new job report.
    #[must_use]
    pub fn new(job_id: &str, label: &str, result: BatchResult, duration_ms: u64) -> Self {
        Self {
            job_id: job_id.to_owned(),
            label: label.to_owned(),
            result,
            duration_ms,
        }
    }

    /// Returns the success rate for this report: 1.0 for `Success`, 0.0 otherwise,
    /// or the partial fraction for `Partial`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        match &self.result {
            BatchResult::Success => 1.0,
            BatchResult::Partial { completed, total } => {
                if *total == 0 {
                    0.0
                } else {
                    f64::from(*completed) / f64::from(*total)
                }
            }
            _ => 0.0,
        }
    }

    /// Returns `true` if this report represents a failing job.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.result.is_failure()
            || matches!(&self.result, BatchResult::Partial { completed, total } if completed < total)
    }
}

/// Aggregated summary over a collection of `BatchJobReport`s.
#[derive(Debug, Default)]
pub struct BatchReportSummary {
    reports: Vec<BatchJobReport>,
}

impl BatchReportSummary {
    /// Creates a new empty summary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a job report to the summary.
    pub fn add_report(&mut self, report: BatchJobReport) {
        self.reports.push(report);
    }

    /// Returns the total number of jobs in the summary.
    #[must_use]
    pub fn total_jobs(&self) -> usize {
        self.reports.len()
    }

    /// Returns the number of fully successful jobs.
    #[must_use]
    pub fn successful_jobs(&self) -> usize {
        self.reports
            .iter()
            .filter(|r| r.result.is_success())
            .count()
    }

    /// Returns the number of failed or cancelled jobs.
    #[must_use]
    pub fn failed_jobs(&self) -> usize {
        self.reports.iter().filter(|r| r.has_failures()).count()
    }

    /// Returns the mean success rate (0.0–1.0) across all reports.
    /// Returns `0.0` if no reports have been added.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_success_rate(&self) -> f64 {
        if self.reports.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.reports.iter().map(BatchJobReport::success_rate).sum();
        sum / self.reports.len() as f64
    }

    /// Returns the average job duration in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_duration_ms(&self) -> f64 {
        if self.reports.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.reports.iter().map(|r| r.duration_ms).sum();
        sum as f64 / self.reports.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn success_report(id: &str) -> BatchJobReport {
        BatchJobReport::new(id, "label", BatchResult::Success, 100)
    }

    fn failure_report(id: &str) -> BatchJobReport {
        BatchJobReport::new(id, "label", BatchResult::Failure("err".to_owned()), 50)
    }

    fn partial_report(id: &str, done: u32, total: u32) -> BatchJobReport {
        BatchJobReport::new(
            id,
            "label",
            BatchResult::Partial {
                completed: done,
                total,
            },
            200,
        )
    }

    #[test]
    fn test_batch_result_is_success() {
        assert!(BatchResult::Success.is_success());
        assert!(!BatchResult::Failure("e".to_owned()).is_success());
    }

    #[test]
    fn test_batch_result_is_failure() {
        assert!(BatchResult::Failure("e".to_owned()).is_failure());
        assert!(BatchResult::Cancelled.is_failure());
        assert!(!BatchResult::Success.is_failure());
    }

    #[test]
    fn test_job_report_success_rate_full() {
        assert!((success_report("j1").success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_report_success_rate_failure() {
        assert!((failure_report("j1").success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_report_success_rate_partial() {
        let r = partial_report("j1", 3, 4);
        assert!((r.success_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_job_report_success_rate_partial_zero_total() {
        let r = partial_report("j1", 0, 0);
        assert!((r.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_report_has_failures_success() {
        assert!(!success_report("j1").has_failures());
    }

    #[test]
    fn test_job_report_has_failures_failure() {
        assert!(failure_report("j1").has_failures());
    }

    #[test]
    fn test_job_report_has_failures_partial() {
        assert!(partial_report("j1", 3, 4).has_failures());
        assert!(!partial_report("j2", 4, 4).has_failures()); // all completed
    }

    #[test]
    fn test_summary_total_jobs() {
        let mut s = BatchReportSummary::new();
        s.add_report(success_report("j1"));
        s.add_report(failure_report("j2"));
        assert_eq!(s.total_jobs(), 2);
    }

    #[test]
    fn test_summary_successful_jobs() {
        let mut s = BatchReportSummary::new();
        s.add_report(success_report("j1"));
        s.add_report(success_report("j2"));
        s.add_report(failure_report("j3"));
        assert_eq!(s.successful_jobs(), 2);
    }

    #[test]
    fn test_summary_failed_jobs() {
        let mut s = BatchReportSummary::new();
        s.add_report(success_report("j1"));
        s.add_report(failure_report("j2"));
        assert_eq!(s.failed_jobs(), 1);
    }

    #[test]
    fn test_summary_total_success_rate_empty() {
        let s = BatchReportSummary::new();
        assert!((s.total_success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summary_total_success_rate_mixed() {
        let mut s = BatchReportSummary::new();
        s.add_report(success_report("j1")); // 1.0
        s.add_report(failure_report("j2")); // 0.0
                                            // mean = 0.5
        assert!((s.total_success_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_summary_avg_duration() {
        let mut s = BatchReportSummary::new();
        s.add_report(success_report("j1")); // 100ms
        s.add_report(failure_report("j2")); // 50ms
        assert!((s.avg_duration_ms() - 75.0).abs() < 1e-9);
    }
}
