#![allow(dead_code)]
//! Render job archival and historical query system.
//!
//! Completed, failed, and cancelled render jobs are archived with full
//! metadata so that farm operators can analyse historical utilisation,
//! identify recurring failures, and produce compliance reports.

use std::collections::HashMap;
use std::fmt;

// ─── Archive Status ─────────────────────────────────────────────────────────

/// Terminal state of an archived job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveStatus {
    /// Job completed successfully.
    Completed,
    /// Job failed after exhausting retries.
    Failed,
    /// Job was cancelled by user or system.
    Cancelled,
    /// Job timed out.
    TimedOut,
}

impl fmt::Display for ArchiveStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
            Self::TimedOut => "Timed Out",
        };
        f.write_str(label)
    }
}

// ─── Archived Job ───────────────────────────────────────────────────────────

/// An archived render job record.
#[derive(Debug, Clone)]
pub struct ArchivedJob {
    /// Unique job identifier (UUID string).
    pub job_id: String,
    /// Human-readable job name.
    pub name: String,
    /// Project this job belonged to.
    pub project: String,
    /// Terminal status.
    pub status: ArchiveStatus,
    /// Total wall-clock render time in seconds.
    pub render_time_secs: f64,
    /// Total CPU-hours consumed.
    pub cpu_hours: f64,
    /// Total GPU-hours consumed.
    pub gpu_hours: f64,
    /// Number of frames rendered.
    pub frames_rendered: u64,
    /// Total frames requested.
    pub frames_total: u64,
    /// Submitting user.
    pub submitted_by: String,
    /// Epoch timestamp of submission (seconds since Unix epoch).
    pub submitted_at: u64,
    /// Epoch timestamp of completion / failure.
    pub finished_at: u64,
    /// Number of retry attempts before terminal state.
    pub retry_count: u32,
    /// Error message (if failed).
    pub error_message: Option<String>,
    /// Arbitrary metadata tags.
    pub tags: HashMap<String, String>,
}

impl ArchivedJob {
    /// Create a new archived job with minimal fields.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        name: impl Into<String>,
        project: impl Into<String>,
        status: ArchiveStatus,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            name: name.into(),
            project: project.into(),
            status,
            render_time_secs: 0.0,
            cpu_hours: 0.0,
            gpu_hours: 0.0,
            frames_rendered: 0,
            frames_total: 0,
            submitted_by: String::new(),
            submitted_at: 0,
            finished_at: 0,
            retry_count: 0,
            error_message: None,
            tags: HashMap::new(),
        }
    }

    /// Duration from submission to finish in seconds.
    #[must_use]
    pub fn turnaround_secs(&self) -> u64 {
        self.finished_at.saturating_sub(self.submitted_at)
    }

    /// Frame completion ratio.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn completion_ratio(&self) -> f64 {
        if self.frames_total == 0 {
            return 0.0;
        }
        self.frames_rendered as f64 / self.frames_total as f64
    }

    /// Returns `true` if the job succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status == ArchiveStatus::Completed
    }

    /// Returns `true` if the job had retries.
    #[must_use]
    pub fn was_retried(&self) -> bool {
        self.retry_count > 0
    }
}

// ─── Archive Store ──────────────────────────────────────────────────────────

/// In-memory archive of completed jobs, keyed by job ID.
#[derive(Debug, Clone, Default)]
pub struct JobArchive {
    /// Archived jobs by job ID.
    jobs: HashMap<String, ArchivedJob>,
}

impl JobArchive {
    /// Create a new empty archive.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Archive a job.
    pub fn insert(&mut self, job: ArchivedJob) {
        self.jobs.insert(job.job_id.clone(), job);
    }

    /// Look up an archived job by ID.
    #[must_use]
    pub fn get(&self, job_id: &str) -> Option<&ArchivedJob> {
        self.jobs.get(job_id)
    }

    /// Total number of archived jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Returns `true` if the archive is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// List all jobs for a given project.
    #[must_use]
    pub fn jobs_for_project(&self, project: &str) -> Vec<&ArchivedJob> {
        self.jobs
            .values()
            .filter(|j| j.project == project)
            .collect()
    }

    /// List all failed jobs.
    #[must_use]
    pub fn failed_jobs(&self) -> Vec<&ArchivedJob> {
        self.jobs
            .values()
            .filter(|j| j.status == ArchiveStatus::Failed)
            .collect()
    }

    /// Total CPU-hours across all archived jobs.
    #[must_use]
    pub fn total_cpu_hours(&self) -> f64 {
        self.jobs.values().map(|j| j.cpu_hours).sum()
    }

    /// Total GPU-hours across all archived jobs.
    #[must_use]
    pub fn total_gpu_hours(&self) -> f64 {
        self.jobs.values().map(|j| j.gpu_hours).sum()
    }

    /// Success rate (fraction of completed jobs over total).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.jobs.is_empty() {
            return 0.0;
        }
        let completed = self
            .jobs
            .values()
            .filter(|j| j.status == ArchiveStatus::Completed)
            .count();
        completed as f64 / self.jobs.len() as f64
    }

    /// Remove all jobs older than `cutoff_epoch` (by `finished_at`).
    pub fn purge_before(&mut self, cutoff_epoch: u64) {
        self.jobs.retain(|_, j| j.finished_at >= cutoff_epoch);
    }

    /// Produce a summary of job counts by status.
    #[must_use]
    pub fn status_summary(&self) -> HashMap<ArchiveStatus, usize> {
        let mut summary = HashMap::new();
        for job in self.jobs.values() {
            *summary.entry(job.status).or_insert(0) += 1;
        }
        summary
    }
}

// ─── Archive Query ──────────────────────────────────────────────────────────

/// Filter criteria for querying the archive.
#[derive(Debug, Clone, Default)]
pub struct ArchiveQuery {
    /// Filter by project name (exact match).
    pub project: Option<String>,
    /// Filter by status.
    pub status: Option<ArchiveStatus>,
    /// Filter by submitter.
    pub submitted_by: Option<String>,
    /// Minimum submission epoch.
    pub submitted_after: Option<u64>,
    /// Maximum submission epoch.
    pub submitted_before: Option<u64>,
}

impl ArchiveQuery {
    /// Create a new empty query (matches all).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: filter by project.
    #[must_use]
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Builder: filter by status.
    #[must_use]
    pub fn with_status(mut self, status: ArchiveStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Builder: filter by submitter.
    #[must_use]
    pub fn with_submitter(mut self, user: impl Into<String>) -> Self {
        self.submitted_by = Some(user.into());
        self
    }

    /// Execute the query against an archive.
    #[must_use]
    pub fn execute<'a>(&self, archive: &'a JobArchive) -> Vec<&'a ArchivedJob> {
        archive
            .jobs
            .values()
            .filter(|j| {
                if let Some(ref p) = self.project {
                    if j.project != *p {
                        return false;
                    }
                }
                if let Some(s) = self.status {
                    if j.status != s {
                        return false;
                    }
                }
                if let Some(ref u) = self.submitted_by {
                    if j.submitted_by != *u {
                        return false;
                    }
                }
                if let Some(after) = self.submitted_after {
                    if j.submitted_at < after {
                        return false;
                    }
                }
                if let Some(before) = self.submitted_before {
                    if j.submitted_at > before {
                        return false;
                    }
                }
                true
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_job(id: &str, status: ArchiveStatus) -> ArchivedJob {
        let mut j = ArchivedJob::new(id, format!("Job {id}"), "project-alpha", status);
        j.submitted_by = "alice".into();
        j.submitted_at = 1000;
        j.finished_at = 2000;
        j.frames_total = 100;
        j.frames_rendered = if status == ArchiveStatus::Completed {
            100
        } else {
            50
        };
        j.cpu_hours = 10.0;
        j.gpu_hours = 5.0;
        j.render_time_secs = 3600.0;
        j
    }

    #[test]
    fn test_archived_job_new() {
        let j = ArchivedJob::new("abc", "Test", "proj", ArchiveStatus::Completed);
        assert_eq!(j.job_id, "abc");
        assert_eq!(j.status, ArchiveStatus::Completed);
    }

    #[test]
    fn test_turnaround_secs() {
        let j = sample_job("1", ArchiveStatus::Completed);
        assert_eq!(j.turnaround_secs(), 1000);
    }

    #[test]
    fn test_completion_ratio_full() {
        let j = sample_job("1", ArchiveStatus::Completed);
        assert!((j.completion_ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_completion_ratio_partial() {
        let j = sample_job("1", ArchiveStatus::Failed);
        assert!((j.completion_ratio() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_completion_ratio_zero_total() {
        let j = ArchivedJob::new("x", "x", "p", ArchiveStatus::Completed);
        assert!((j.completion_ratio() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_is_success() {
        assert!(sample_job("1", ArchiveStatus::Completed).is_success());
        assert!(!sample_job("2", ArchiveStatus::Failed).is_success());
    }

    #[test]
    fn test_was_retried() {
        let mut j = sample_job("1", ArchiveStatus::Completed);
        assert!(!j.was_retried());
        j.retry_count = 2;
        assert!(j.was_retried());
    }

    #[test]
    fn test_archive_insert_get() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("a", ArchiveStatus::Completed));
        assert_eq!(archive.len(), 1);
        assert!(archive.get("a").is_some());
        assert!(archive.get("b").is_none());
    }

    #[test]
    fn test_archive_is_empty() {
        let archive = JobArchive::new();
        assert!(archive.is_empty());
    }

    #[test]
    fn test_archive_jobs_for_project() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        let mut other = ArchivedJob::new("2", "J2", "other-project", ArchiveStatus::Completed);
        other.submitted_by = "bob".into();
        archive.insert(other);
        let alpha_jobs = archive.jobs_for_project("project-alpha");
        assert_eq!(alpha_jobs.len(), 1);
    }

    #[test]
    fn test_archive_failed_jobs() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        archive.insert(sample_job("2", ArchiveStatus::Failed));
        assert_eq!(archive.failed_jobs().len(), 1);
    }

    #[test]
    fn test_archive_total_hours() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        archive.insert(sample_job("2", ArchiveStatus::Completed));
        assert!((archive.total_cpu_hours() - 20.0).abs() < 1e-12);
        assert!((archive.total_gpu_hours() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_archive_success_rate() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        archive.insert(sample_job("2", ArchiveStatus::Failed));
        assert!((archive.success_rate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_archive_success_rate_empty() {
        let archive = JobArchive::new();
        assert!((archive.success_rate() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_archive_purge_before() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed)); // finished_at = 2000
        let mut old = sample_job("2", ArchiveStatus::Completed);
        old.finished_at = 500;
        archive.insert(old);
        archive.purge_before(1000);
        assert_eq!(archive.len(), 1);
        assert!(archive.get("1").is_some());
    }

    #[test]
    fn test_archive_status_summary() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        archive.insert(sample_job("2", ArchiveStatus::Failed));
        archive.insert(sample_job("3", ArchiveStatus::Completed));
        let summary = archive.status_summary();
        assert_eq!(summary[&ArchiveStatus::Completed], 2);
        assert_eq!(summary[&ArchiveStatus::Failed], 1);
    }

    #[test]
    fn test_archive_query_all() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        archive.insert(sample_job("2", ArchiveStatus::Failed));
        let results = ArchiveQuery::new().execute(&archive);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_archive_query_by_status() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        archive.insert(sample_job("2", ArchiveStatus::Failed));
        let results = ArchiveQuery::new()
            .with_status(ArchiveStatus::Failed)
            .execute(&archive);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].job_id, "2");
    }

    #[test]
    fn test_archive_query_by_project() {
        let mut archive = JobArchive::new();
        archive.insert(sample_job("1", ArchiveStatus::Completed));
        let results = ArchiveQuery::new()
            .with_project("project-alpha")
            .execute(&archive);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_archive_status_display() {
        assert_eq!(format!("{}", ArchiveStatus::Completed), "Completed");
        assert_eq!(format!("{}", ArchiveStatus::TimedOut), "Timed Out");
    }
}
