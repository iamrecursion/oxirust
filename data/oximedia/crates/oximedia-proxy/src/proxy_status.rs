#![allow(dead_code)]

//! Proxy generation status tracking and lifecycle management.
//!
//! This module provides a status tracker for proxy generation jobs,
//! supporting state transitions, progress reporting, error tracking,
//! and batch status aggregation.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Possible states of a proxy generation job.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProxyState {
    /// Job has been created but not yet started.
    Queued,
    /// Job is currently being processed.
    InProgress,
    /// Job completed successfully.
    Completed,
    /// Job failed with an error.
    Failed,
    /// Job was cancelled before completion.
    Cancelled,
    /// Job is paused (e.g., waiting for resources).
    Paused,
    /// Job is being retried after a previous failure.
    Retrying,
}

impl std::fmt::Display for ProxyState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Queued => "Queued",
            Self::InProgress => "In Progress",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
            Self::Paused => "Paused",
            Self::Retrying => "Retrying",
        };
        write!(f, "{label}")
    }
}

/// Status information for a single proxy generation job.
#[derive(Debug, Clone)]
pub struct ProxyJobStatus {
    /// Unique job identifier.
    pub job_id: String,
    /// Source media file path.
    pub source_path: String,
    /// Output proxy file path.
    pub output_path: String,
    /// Current state of the job.
    pub state: ProxyState,
    /// Progress percentage (0.0 to 100.0).
    pub progress_percent: f64,
    /// Number of frames processed so far.
    pub frames_processed: u64,
    /// Total frames expected.
    pub total_frames: u64,
    /// Error message (if state is Failed).
    pub error_message: Option<String>,
    /// Number of retry attempts so far.
    pub retry_count: u32,
    /// Maximum number of retries allowed.
    pub max_retries: u32,
}

impl ProxyJobStatus {
    /// Create a new job status in the Queued state.
    pub fn new(job_id: &str, source: &str, output: &str) -> Self {
        Self {
            job_id: job_id.to_string(),
            source_path: source.to_string(),
            output_path: output.to_string(),
            state: ProxyState::Queued,
            progress_percent: 0.0,
            frames_processed: 0,
            total_frames: 0,
            error_message: None,
            retry_count: 0,
            max_retries: 3,
        }
    }

    /// Set the total frame count.
    pub fn with_total_frames(mut self, total: u64) -> Self {
        self.total_frames = total;
        self
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Check if the job is in a terminal state (Completed, Failed, or Cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            ProxyState::Completed | ProxyState::Failed | ProxyState::Cancelled
        )
    }

    /// Check if the job can be retried.
    pub fn can_retry(&self) -> bool {
        self.state == ProxyState::Failed && self.retry_count < self.max_retries
    }

    /// Update the progress based on frames processed.
    #[allow(clippy::cast_precision_loss)]
    pub fn update_progress(&mut self, frames: u64) {
        self.frames_processed = frames;
        if self.total_frames > 0 {
            self.progress_percent = (frames as f64 / self.total_frames as f64) * 100.0;
            if self.progress_percent > 100.0 {
                self.progress_percent = 100.0;
            }
        }
    }
}

/// Tracker that manages multiple proxy job statuses.
#[derive(Debug)]
pub struct ProxyStatusTracker {
    /// Map of job_id to status.
    jobs: HashMap<String, ProxyJobStatus>,
    /// Timestamp of tracker creation.
    created_at: Instant,
}

impl ProxyStatusTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    /// Register a new job.
    pub fn add_job(&mut self, status: ProxyJobStatus) {
        self.jobs.insert(status.job_id.clone(), status);
    }

    /// Get the status of a specific job.
    pub fn get_job(&self, job_id: &str) -> Option<&ProxyJobStatus> {
        self.jobs.get(job_id)
    }

    /// Transition a job to a new state.
    pub fn transition(&mut self, job_id: &str, new_state: ProxyState) -> bool {
        if let Some(job) = self.jobs.get_mut(job_id) {
            if job.is_terminal() && new_state != ProxyState::Retrying {
                return false;
            }
            if new_state == ProxyState::Retrying {
                job.retry_count += 1;
            }
            if new_state == ProxyState::Completed {
                job.progress_percent = 100.0;
            }
            job.state = new_state;
            true
        } else {
            false
        }
    }

    /// Record a failure with an error message.
    pub fn fail_job(&mut self, job_id: &str, error: &str) -> bool {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.state = ProxyState::Failed;
            job.error_message = Some(error.to_string());
            true
        } else {
            false
        }
    }

    /// Update frame progress on a job.
    pub fn update_progress(&mut self, job_id: &str, frames: u64) -> bool {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.update_progress(frames);
            true
        } else {
            false
        }
    }

    /// Return the number of tracked jobs.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Return a count of jobs by state.
    pub fn count_by_state(&self) -> HashMap<ProxyState, usize> {
        let mut counts: HashMap<ProxyState, usize> = HashMap::new();
        for job in self.jobs.values() {
            *counts.entry(job.state.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Return the overall progress across all non-terminal jobs.
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_progress(&self) -> f64 {
        let active: Vec<&ProxyJobStatus> =
            self.jobs.values().filter(|j| !j.is_terminal()).collect();
        if active.is_empty() {
            return 100.0;
        }
        let sum: f64 = active.iter().map(|j| j.progress_percent).sum();
        sum / active.len() as f64
    }

    /// Return all jobs that are in the Failed state and can be retried.
    pub fn retryable_jobs(&self) -> Vec<&ProxyJobStatus> {
        self.jobs.values().filter(|j| j.can_retry()).collect()
    }

    /// Elapsed time since the tracker was created.
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Remove all jobs that are in a terminal state.
    pub fn clear_terminal(&mut self) -> usize {
        let before = self.jobs.len();
        self.jobs.retain(|_, j| !j.is_terminal());
        before - self.jobs.len()
    }
}

impl Default for ProxyStatusTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: &str) -> ProxyJobStatus {
        ProxyJobStatus::new(id, "/src/clip.mov", "/proxy/clip.mp4")
            .with_total_frames(1000)
            .with_max_retries(2)
    }

    #[test]
    fn test_new_job_is_queued() {
        let job = make_job("j1");
        assert_eq!(job.state, ProxyState::Queued);
        assert!((job.progress_percent - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_terminal() {
        let mut job = make_job("j1");
        assert!(!job.is_terminal());
        job.state = ProxyState::Completed;
        assert!(job.is_terminal());
        job.state = ProxyState::Failed;
        assert!(job.is_terminal());
        job.state = ProxyState::Cancelled;
        assert!(job.is_terminal());
    }

    #[test]
    fn test_can_retry() {
        let mut job = make_job("j1");
        job.state = ProxyState::Failed;
        job.retry_count = 0;
        assert!(job.can_retry());
        job.retry_count = 2;
        assert!(!job.can_retry());
    }

    #[test]
    fn test_update_progress() {
        let mut job = make_job("j1");
        job.update_progress(500);
        assert!((job.progress_percent - 50.0).abs() < f64::EPSILON);
        assert_eq!(job.frames_processed, 500);
    }

    #[test]
    fn test_progress_caps_at_100() {
        let mut job = make_job("j1");
        job.update_progress(2000);
        assert!((job.progress_percent - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracker_add_and_get() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        assert_eq!(tracker.job_count(), 1);
        assert!(tracker.get_job("j1").is_some());
        assert!(tracker.get_job("j999").is_none());
    }

    #[test]
    fn test_transition() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        assert!(tracker.transition("j1", ProxyState::InProgress));
        assert_eq!(
            tracker.get_job("j1").expect("should succeed in test").state,
            ProxyState::InProgress
        );
    }

    #[test]
    fn test_transition_terminal_blocked() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        tracker.transition("j1", ProxyState::Completed);
        // Cannot go back to InProgress from Completed
        assert!(!tracker.transition("j1", ProxyState::InProgress));
    }

    #[test]
    fn test_transition_retry_allowed() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        tracker.transition("j1", ProxyState::Failed);
        // Retrying is allowed from Failed
        assert!(tracker.transition("j1", ProxyState::Retrying));
        assert_eq!(
            tracker
                .get_job("j1")
                .expect("should succeed in test")
                .retry_count,
            1
        );
    }

    #[test]
    fn test_fail_job() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        assert!(tracker.fail_job("j1", "codec not found"));
        let job = tracker.get_job("j1").expect("should succeed in test");
        assert_eq!(job.state, ProxyState::Failed);
        assert_eq!(job.error_message.as_deref(), Some("codec not found"));
    }

    #[test]
    fn test_fail_nonexistent_job() {
        let mut tracker = ProxyStatusTracker::new();
        assert!(!tracker.fail_job("nonexistent", "err"));
    }

    #[test]
    fn test_count_by_state() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        tracker.add_job(make_job("j2"));
        tracker.add_job(make_job("j3"));
        tracker.transition("j2", ProxyState::InProgress);
        tracker.transition("j3", ProxyState::Completed);

        let counts = tracker.count_by_state();
        assert_eq!(*counts.get(&ProxyState::Queued).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&ProxyState::InProgress).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&ProxyState::Completed).unwrap_or(&0), 1);
    }

    #[test]
    fn test_overall_progress() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        tracker.add_job(make_job("j2"));
        tracker.update_progress("j1", 500); // 50%
        tracker.update_progress("j2", 250); // 25%
        let overall = tracker.overall_progress();
        assert!((overall - 37.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_overall_progress_all_done() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        tracker.transition("j1", ProxyState::Completed);
        assert!((tracker.overall_progress() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_retryable_jobs() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        tracker.add_job(make_job("j2"));
        tracker.fail_job("j1", "err");
        let retryable = tracker.retryable_jobs();
        assert_eq!(retryable.len(), 1);
        assert_eq!(retryable[0].job_id, "j1");
    }

    #[test]
    fn test_clear_terminal() {
        let mut tracker = ProxyStatusTracker::new();
        tracker.add_job(make_job("j1"));
        tracker.add_job(make_job("j2"));
        tracker.add_job(make_job("j3"));
        tracker.transition("j1", ProxyState::Completed);
        tracker.fail_job("j2", "err");
        let cleared = tracker.clear_terminal();
        assert_eq!(cleared, 2);
        assert_eq!(tracker.job_count(), 1);
    }

    #[test]
    fn test_proxy_state_display() {
        assert_eq!(format!("{}", ProxyState::Queued), "Queued");
        assert_eq!(format!("{}", ProxyState::InProgress), "In Progress");
        assert_eq!(format!("{}", ProxyState::Completed), "Completed");
        assert_eq!(format!("{}", ProxyState::Failed), "Failed");
    }

    #[test]
    fn test_default_tracker() {
        let tracker = ProxyStatusTracker::default();
        assert_eq!(tracker.job_count(), 0);
    }
}
