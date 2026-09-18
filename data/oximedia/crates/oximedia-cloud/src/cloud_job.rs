//! Cloud job lifecycle management.

#![allow(dead_code)]

use std::collections::VecDeque;

/// The kind of work a cloud job performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudJobType {
    /// Transcode a media file to a different format/codec.
    Transcode,
    /// Generate thumbnails from a video.
    Thumbnail,
    /// Package media for adaptive streaming (HLS/DASH).
    Package,
    /// Analyse media content (scene detection, QC, etc.).
    Analyse,
    /// Transfer data between storage buckets or regions.
    Transfer,
}

impl CloudJobType {
    /// Rough estimated cost in USD for an average job of this type.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_cost_usd(&self) -> f64 {
        match self {
            CloudJobType::Transcode => 0.50,
            CloudJobType::Thumbnail => 0.02,
            CloudJobType::Package => 0.30,
            CloudJobType::Analyse => 0.15,
            CloudJobType::Transfer => 0.08,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            CloudJobType::Transcode => "Transcode",
            CloudJobType::Thumbnail => "Thumbnail",
            CloudJobType::Package => "Package",
            CloudJobType::Analyse => "Analyse",
            CloudJobType::Transfer => "Transfer",
        }
    }
}

/// Status of a cloud job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Waiting in the queue.
    Queued,
    /// Currently executing on cloud infrastructure.
    Running,
    /// Finished successfully.
    Succeeded,
    /// Finished with an error.
    Failed(String),
}

/// A single cloud processing job.
#[derive(Debug, Clone)]
pub struct CloudJob {
    /// Unique job identifier.
    pub id: String,
    /// Type of work.
    pub job_type: CloudJobType,
    /// Current status.
    pub status: JobStatus,
    /// ID of the source asset.
    pub asset_id: String,
}

impl CloudJob {
    /// Create a new job in the `Queued` state.
    pub fn new(id: impl Into<String>, job_type: CloudJobType, asset_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            job_type,
            status: JobStatus::Queued,
            asset_id: asset_id.into(),
        }
    }

    /// Returns `true` when the job has finished (either success or failure).
    pub fn is_complete(&self) -> bool {
        matches!(self.status, JobStatus::Succeeded | JobStatus::Failed(_))
    }

    /// Returns `true` when the job succeeded.
    pub fn is_success(&self) -> bool {
        self.status == JobStatus::Succeeded
    }

    /// Transition the job to the `Running` state.
    pub fn start(&mut self) {
        if self.status == JobStatus::Queued {
            self.status = JobStatus::Running;
        }
    }

    /// Mark the job as succeeded.
    pub fn complete(&mut self) {
        self.status = JobStatus::Succeeded;
    }

    /// Mark the job as failed with an error message.
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = JobStatus::Failed(reason.into());
    }

    /// Estimated cost in USD for this job.
    pub fn estimated_cost_usd(&self) -> f64 {
        self.job_type.estimated_cost_usd()
    }
}

/// A queue that manages submission and execution of `CloudJob` instances.
#[derive(Debug, Default)]
pub struct CloudJobQueue {
    pending: VecDeque<CloudJob>,
    all: Vec<CloudJob>,
}

impl CloudJobQueue {
    /// Create an empty job queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a new job to the queue.
    pub fn submit(&mut self, job: CloudJob) {
        self.pending.push_back(job.clone());
        self.all.push(job);
    }

    /// Pop the next pending job, marking it as running.
    pub fn next_job(&mut self) -> Option<CloudJob> {
        let mut job = self.pending.pop_front()?;
        job.start();
        // Update in the all-jobs store
        if let Some(stored) = self.all.iter_mut().find(|j| j.id == job.id) {
            stored.start();
        }
        Some(job)
    }

    /// Count of jobs currently in the Running state.
    pub fn running_count(&self) -> usize {
        self.all
            .iter()
            .filter(|j| j.status == JobStatus::Running)
            .count()
    }

    /// Count of jobs waiting to be picked up.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Produce aggregate statistics.
    pub fn stats(&self) -> CloudJobStats {
        let total = self.all.len();
        let succeeded = self.all.iter().filter(|j| j.is_success()).count();
        let failed = self
            .all
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Failed(_)))
            .count();
        CloudJobStats {
            total,
            succeeded,
            failed,
        }
    }
}

/// Aggregate statistics about jobs processed by a `CloudJobQueue`.
#[derive(Debug, Clone)]
pub struct CloudJobStats {
    /// Total jobs submitted.
    pub total: usize,
    /// Number of successful jobs.
    pub succeeded: usize,
    /// Number of failed jobs.
    pub failed: usize,
}

impl CloudJobStats {
    /// Success rate as a value in [0.0, 1.0]. Returns 0.0 when no jobs have been submitted.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.succeeded as f64 / self.total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcode_job(id: &str) -> CloudJob {
        CloudJob::new(id, CloudJobType::Transcode, "asset-1")
    }

    #[test]
    fn test_transcode_cost() {
        assert!((CloudJobType::Transcode.estimated_cost_usd() - 0.50).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thumbnail_cost() {
        assert!((CloudJobType::Thumbnail.estimated_cost_usd() - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn test_labels() {
        assert_eq!(CloudJobType::Package.label(), "Package");
        assert_eq!(CloudJobType::Analyse.label(), "Analyse");
        assert_eq!(CloudJobType::Transfer.label(), "Transfer");
    }

    #[test]
    fn test_new_job_is_queued() {
        let job = transcode_job("j1");
        assert_eq!(job.status, JobStatus::Queued);
        assert!(!job.is_complete());
    }

    #[test]
    fn test_job_start_changes_status() {
        let mut job = transcode_job("j1");
        job.start();
        assert_eq!(job.status, JobStatus::Running);
    }

    #[test]
    fn test_job_complete() {
        let mut job = transcode_job("j1");
        job.start();
        job.complete();
        assert!(job.is_complete());
        assert!(job.is_success());
    }

    #[test]
    fn test_job_fail() {
        let mut job = transcode_job("j1");
        job.start();
        job.fail("disk full");
        assert!(job.is_complete());
        assert!(!job.is_success());
    }

    #[test]
    fn test_queue_submit_and_pending_count() {
        let mut q = CloudJobQueue::new();
        q.submit(transcode_job("j1"));
        q.submit(transcode_job("j2"));
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn test_queue_next_job() {
        let mut q = CloudJobQueue::new();
        q.submit(transcode_job("j1"));
        let job = q.next_job();
        assert!(job.is_some());
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn test_queue_running_count() {
        let mut q = CloudJobQueue::new();
        q.submit(transcode_job("j1"));
        q.next_job(); // transitions to running in `all`
        assert_eq!(q.running_count(), 1);
    }

    #[test]
    fn test_stats_success_rate_full() {
        let stats = CloudJobStats {
            total: 10,
            succeeded: 10,
            failed: 0,
        };
        assert!((stats.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_success_rate_zero_total() {
        let stats = CloudJobStats {
            total: 0,
            succeeded: 0,
            failed: 0,
        };
        assert!((stats.success_rate()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_success_rate_partial() {
        let stats = CloudJobStats {
            total: 4,
            succeeded: 3,
            failed: 1,
        };
        assert!((stats.success_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_job_estimated_cost() {
        let job = transcode_job("j1");
        assert!((job.estimated_cost_usd() - 0.50).abs() < f64::EPSILON);
    }
}
