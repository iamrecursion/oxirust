//! Asset transfer manager for MAM.
//!
//! Tracks file transfer jobs between storage locations, supporting
//! progress monitoring, cancellation, and retry logic.

#![allow(dead_code)]

use std::collections::HashMap;

/// Current status of a transfer job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferStatus {
    /// Job is queued but not yet started.
    Queued,
    /// Transfer is actively running.
    InProgress,
    /// Transfer completed successfully.
    Completed,
    /// Transfer failed; `error` field has details.
    Failed,
    /// Transfer was cancelled by the user.
    Cancelled,
    /// Transfer is paused and can be resumed.
    Paused,
}

impl TransferStatus {
    /// Return `true` if the job is in a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Return `true` if the job is still active (queued, running, or paused).
    #[must_use]
    pub const fn is_active(&self) -> bool {
        !self.is_terminal()
    }
}

/// Direction / type of transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// Local-to-local copy.
    LocalToLocal,
    /// Upload to remote / cloud storage.
    Upload,
    /// Download from remote / cloud storage.
    Download,
    /// Remote-to-remote transfer without local staging.
    RemoteToRemote,
}

/// A single file-transfer job.
#[derive(Clone, Debug)]
pub struct TransferJob {
    /// Unique job identifier.
    pub id: u64,
    /// Asset ID being transferred.
    pub asset_id: u64,
    /// Source path or URI.
    pub source: String,
    /// Destination path or URI.
    pub destination: String,
    /// Direction of the transfer.
    pub direction: TransferDirection,
    /// Current status.
    pub status: TransferStatus,
    /// Total bytes to transfer.
    pub total_bytes: u64,
    /// Bytes transferred so far.
    pub transferred_bytes: u64,
    /// Number of retry attempts performed.
    pub retry_count: u32,
    /// Maximum retries allowed before marking as failed.
    pub max_retries: u32,
    /// Most recent error message, if any.
    pub last_error: Option<String>,
    /// Priority: higher values are scheduled first.
    pub priority: u32,
}

impl TransferJob {
    /// Create a new queued transfer job.
    #[must_use]
    pub fn new(
        id: u64,
        asset_id: u64,
        source: impl Into<String>,
        destination: impl Into<String>,
        direction: TransferDirection,
        total_bytes: u64,
    ) -> Self {
        Self {
            id,
            asset_id,
            source: source.into(),
            destination: destination.into(),
            direction,
            status: TransferStatus::Queued,
            total_bytes,
            transferred_bytes: 0,
            retry_count: 0,
            max_retries: 3,
            last_error: None,
            priority: 0,
        }
    }

    /// Completion fraction in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        self.transferred_bytes as f64 / self.total_bytes as f64
    }

    /// Remaining bytes.
    #[must_use]
    pub fn remaining_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.transferred_bytes)
    }

    /// Return `true` if another retry attempt is allowed.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
}

/// Manages a collection of [`TransferJob`] instances.
///
/// # Example
/// ```
/// use oximedia_mam::transfer_manager::{TransferDirection, TransferManager, TransferStatus};
///
/// let mut mgr = TransferManager::new();
/// let id = mgr.submit(1, "/src/file.mxf", "/dst/file.mxf", TransferDirection::LocalToLocal, 1024);
/// mgr.start(id);
/// assert_eq!(mgr.get(id).expect("transfer exists").status, TransferStatus::InProgress);
/// ```
#[derive(Default)]
pub struct TransferManager {
    jobs: HashMap<u64, TransferJob>,
    next_id: u64,
}

impl TransferManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a new transfer job and return its ID.
    pub fn submit(
        &mut self,
        asset_id: u64,
        source: impl Into<String>,
        destination: impl Into<String>,
        direction: TransferDirection,
        total_bytes: u64,
    ) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        let job = TransferJob::new(id, asset_id, source, destination, direction, total_bytes);
        self.jobs.insert(id, job);
        id
    }

    /// Look up a job by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&TransferJob> {
        self.jobs.get(&id)
    }

    /// Return the total number of jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Return `true` when no jobs are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Transition a queued job to `InProgress`.
    ///
    /// Returns `false` if the job is not in the `Queued` state.
    pub fn start(&mut self, id: u64) -> bool {
        if let Some(job) = self.jobs.get_mut(&id) {
            if job.status == TransferStatus::Queued || job.status == TransferStatus::Paused {
                job.status = TransferStatus::InProgress;
                return true;
            }
        }
        false
    }

    /// Update progress for an in-progress job.
    ///
    /// Clamps `bytes` to `total_bytes`.
    pub fn update_progress(&mut self, id: u64, bytes_transferred: u64) {
        if let Some(job) = self.jobs.get_mut(&id) {
            job.transferred_bytes = bytes_transferred.min(job.total_bytes);
        }
    }

    /// Mark a job as completed.
    pub fn complete(&mut self, id: u64) {
        if let Some(job) = self.jobs.get_mut(&id) {
            job.transferred_bytes = job.total_bytes;
            job.status = TransferStatus::Completed;
        }
    }

    /// Mark a job as failed with an error message.
    pub fn fail(&mut self, id: u64, error: impl Into<String>) {
        if let Some(job) = self.jobs.get_mut(&id) {
            job.status = TransferStatus::Failed;
            job.last_error = Some(error.into());
        }
    }

    /// Cancel a non-terminal job.  Returns `false` if already terminal.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(job) = self.jobs.get_mut(&id) {
            if !job.status.is_terminal() {
                job.status = TransferStatus::Cancelled;
                return true;
            }
        }
        false
    }

    /// Pause an in-progress job.
    pub fn pause(&mut self, id: u64) -> bool {
        if let Some(job) = self.jobs.get_mut(&id) {
            if job.status == TransferStatus::InProgress {
                job.status = TransferStatus::Paused;
                return true;
            }
        }
        false
    }

    /// Retry a failed job (increments `retry_count`, resets to `Queued`).
    ///
    /// Returns `false` if the job cannot be retried.
    pub fn retry(&mut self, id: u64) -> bool {
        if let Some(job) = self.jobs.get_mut(&id) {
            if job.status == TransferStatus::Failed && job.can_retry() {
                job.retry_count += 1;
                job.status = TransferStatus::Queued;
                job.last_error = None;
                return true;
            }
        }
        false
    }

    /// Return all jobs with the given status.
    #[must_use]
    pub fn jobs_with_status(&self, status: TransferStatus) -> Vec<&TransferJob> {
        self.jobs.values().filter(|j| j.status == status).collect()
    }

    /// Return jobs ordered by priority (highest first), then by ID (FIFO).
    #[must_use]
    pub fn queued_by_priority(&self) -> Vec<&TransferJob> {
        let mut queued: Vec<&TransferJob> = self
            .jobs
            .values()
            .filter(|j| j.status == TransferStatus::Queued)
            .collect();
        queued.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.id.cmp(&b.id)));
        queued
    }

    /// Remove all terminal jobs, returning the count removed.
    pub fn purge_terminal(&mut self) -> usize {
        let before = self.jobs.len();
        self.jobs.retain(|_, j| !j.status.is_terminal());
        before - self.jobs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submit_job(mgr: &mut TransferManager) -> u64 {
        mgr.submit(
            1,
            "/src/clip.mxf",
            "/dst/clip.mxf",
            TransferDirection::LocalToLocal,
            1024 * 1024,
        )
    }

    #[test]
    fn test_submit_creates_queued_job() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        assert_eq!(
            mgr.get(id).expect("should succeed in test").status,
            TransferStatus::Queued
        );
    }

    #[test]
    fn test_start_transitions_to_in_progress() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        assert!(mgr.start(id));
        assert_eq!(
            mgr.get(id).expect("should succeed in test").status,
            TransferStatus::InProgress
        );
    }

    #[test]
    fn test_update_progress_clamps_to_total() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        mgr.start(id);
        let total = mgr.get(id).expect("should succeed in test").total_bytes;
        mgr.update_progress(id, total + 9999);
        assert_eq!(
            mgr.get(id)
                .expect("should succeed in test")
                .transferred_bytes,
            total
        );
    }

    #[test]
    fn test_complete_sets_status_and_bytes() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        mgr.start(id);
        mgr.complete(id);
        let job = mgr.get(id).expect("should succeed in test");
        assert_eq!(job.status, TransferStatus::Completed);
        assert_eq!(job.transferred_bytes, job.total_bytes);
    }

    #[test]
    fn test_fail_stores_error() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        mgr.fail(id, "network timeout");
        let job = mgr.get(id).expect("should succeed in test");
        assert_eq!(job.status, TransferStatus::Failed);
        assert_eq!(job.last_error.as_deref(), Some("network timeout"));
    }

    #[test]
    fn test_cancel_active_job() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        assert!(mgr.cancel(id));
        assert_eq!(
            mgr.get(id).expect("should succeed in test").status,
            TransferStatus::Cancelled
        );
    }

    #[test]
    fn test_cancel_terminal_job_returns_false() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        mgr.complete(id);
        assert!(!mgr.cancel(id));
    }

    #[test]
    fn test_pause_and_resume() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        mgr.start(id);
        assert!(mgr.pause(id));
        assert_eq!(
            mgr.get(id).expect("should succeed in test").status,
            TransferStatus::Paused
        );
        assert!(mgr.start(id));
        assert_eq!(
            mgr.get(id).expect("should succeed in test").status,
            TransferStatus::InProgress
        );
    }

    #[test]
    fn test_retry_increments_count() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        mgr.fail(id, "error");
        assert!(mgr.retry(id));
        assert_eq!(mgr.get(id).expect("should succeed in test").retry_count, 1);
        assert_eq!(
            mgr.get(id).expect("should succeed in test").status,
            TransferStatus::Queued
        );
    }

    #[test]
    fn test_retry_exhausted() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        if let Some(job) = mgr.jobs.get_mut(&id) {
            job.retry_count = job.max_retries;
            job.status = TransferStatus::Failed;
        }
        assert!(!mgr.retry(id));
    }

    #[test]
    fn test_jobs_with_status() {
        let mut mgr = TransferManager::new();
        let id1 = submit_job(&mut mgr);
        let id2 = submit_job(&mut mgr);
        mgr.start(id1);
        let in_progress = mgr.jobs_with_status(TransferStatus::InProgress);
        assert_eq!(in_progress.len(), 1);
        let queued = mgr.jobs_with_status(TransferStatus::Queued);
        assert_eq!(queued.len(), 1);
        let _ = id2; // used above
    }

    #[test]
    fn test_purge_terminal() {
        let mut mgr = TransferManager::new();
        let id1 = submit_job(&mut mgr);
        let _id2 = submit_job(&mut mgr);
        mgr.complete(id1);
        let removed = mgr.purge_terminal();
        assert_eq!(removed, 1);
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_progress_fraction() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);
        mgr.start(id);
        let total = mgr.get(id).expect("should succeed in test").total_bytes;
        mgr.update_progress(id, total / 2);
        let p = mgr.get(id).expect("should succeed in test").progress();
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_is_terminal_flags() {
        assert!(TransferStatus::Completed.is_terminal());
        assert!(TransferStatus::Failed.is_terminal());
        assert!(!TransferStatus::InProgress.is_terminal());
    }

    #[test]
    fn test_queued_by_priority_ordering() {
        let mut mgr = TransferManager::new();
        let id_low = submit_job(&mut mgr);
        let id_high = submit_job(&mut mgr);
        if let Some(j) = mgr.jobs.get_mut(&id_high) {
            j.priority = 10;
        }
        let ordered = mgr.queued_by_priority();
        assert_eq!(ordered[0].id, id_high);
        assert_eq!(ordered[1].id, id_low);
    }

    // -----------------------------------------------------------------------
    // Simulated network failure and partial-transfer tests
    // -----------------------------------------------------------------------

    /// Simulate a partial transfer that stalls mid-way, then fails with a
    /// network error.  Verify that the error is recorded, the transferred byte
    /// count reflects the last known progress, and status becomes `Failed`.
    #[test]
    fn test_partial_transfer_then_network_failure() {
        let mut mgr = TransferManager::new();
        let id = mgr.submit(
            42,
            "remote://host/asset.mxf",
            "/mnt/local/asset.mxf",
            TransferDirection::Download,
            1_000_000,
        );

        mgr.start(id);
        // Transfer progresses to 40 %.
        mgr.update_progress(id, 400_000);
        {
            let job = mgr.get(id).expect("job must exist");
            assert_eq!(job.transferred_bytes, 400_000);
            assert_eq!(job.status, TransferStatus::InProgress);
        }

        // Network failure at 40 %.
        mgr.fail(id, "network error: connection reset by peer");
        {
            let job = mgr.get(id).expect("job must exist");
            assert_eq!(job.status, TransferStatus::Failed);
            // Byte count is frozen at the last progress snapshot.
            assert_eq!(job.transferred_bytes, 400_000);
            assert_eq!(
                job.last_error.as_deref(),
                Some("network error: connection reset by peer")
            );
        }
    }

    /// After a partial transfer failure the byte count is preserved across a
    /// retry (the retry only resets status to Queued; bytes_transferred is NOT
    /// reset by the manager — the caller controls that via `update_progress`).
    #[test]
    fn test_retry_preserves_partial_byte_count() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);

        mgr.start(id);
        mgr.update_progress(id, 512 * 1024); // 50 % of 1 MiB
        mgr.fail(id, "timeout");

        // Retry resets status to Queued but must not lose progress bytes.
        assert!(mgr.retry(id), "first retry must succeed");
        let job = mgr.get(id).expect("job must exist after retry");
        assert_eq!(job.status, TransferStatus::Queued);
        assert_eq!(job.retry_count, 1);
        // transferred_bytes unchanged — the caller will re-set it upon resume.
        assert_eq!(job.transferred_bytes, 512 * 1024);
        assert!(job.last_error.is_none());
    }

    /// A job that has exhausted all its retry budget must remain `Failed` and
    /// `can_retry()` must return `false`.
    ///
    /// With `max_retries = 3`:
    ///   - Failures 1–3 can each be retried (retry_count goes 0 → 1 → 2 → 3).
    ///   - Failure 4 cannot: retry_count (3) == max_retries (3) → `can_retry() = false`.
    #[test]
    fn test_max_retries_exhausted_after_repeated_failures() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr); // max_retries = 3

        // Consume the retry budget: fail → retry, repeated max_retries times.
        for attempt in 1..=3_u32 {
            mgr.start(id);
            mgr.fail(id, format!("error attempt {attempt}"));
            assert!(
                mgr.get(id).expect("must exist").can_retry(),
                "should still be retryable at attempt {attempt}"
            );
            assert!(mgr.retry(id), "retry() must succeed at attempt {attempt}");
        }

        // Now retry_count == max_retries (3). One more failure exhausts the budget.
        mgr.start(id);
        mgr.fail(id, "final error");
        let job = mgr.get(id).expect("must exist");
        assert!(
            !job.can_retry(),
            "must not be retryable after {} retries",
            job.max_retries
        );
        assert!(!mgr.retry(id), "retry() must refuse after budget exhausted");
        assert_eq!(
            mgr.get(id).expect("must exist").status,
            TransferStatus::Failed,
            "status must remain Failed after exhaustion"
        );
    }

    /// Simulates the typical failure-then-complete scenario:
    ///   1. Transfer starts.
    ///   2. Transient network hiccup → fail at 30 %.
    ///   3. Retry → resume, complete.
    ///
    /// Verifies that the job ends in `Completed` and `transferred_bytes == total_bytes`.
    #[test]
    fn test_transient_failure_then_eventual_success() {
        let mut mgr = TransferManager::new();
        let total: u64 = 2_000_000;
        let id = mgr.submit(
            7,
            "s3://bucket/clip.mov",
            "/local/clip.mov",
            TransferDirection::Download,
            total,
        );

        // First attempt: fails after 30 %.
        mgr.start(id);
        mgr.update_progress(id, total * 30 / 100);
        mgr.fail(id, "TCP RST");

        // Retry and complete.
        assert!(mgr.retry(id));
        mgr.start(id);
        mgr.complete(id);

        let job = mgr.get(id).expect("must exist");
        assert_eq!(job.status, TransferStatus::Completed);
        assert_eq!(job.transferred_bytes, total);
        assert_eq!(job.retry_count, 1);
    }

    /// Multiple concurrent jobs can fail independently; each tracks its own
    /// error message and retry counter.
    #[test]
    fn test_multiple_independent_failures() {
        let mut mgr = TransferManager::new();
        let ids: Vec<u64> = (0..5)
            .map(|i| {
                mgr.submit(
                    i,
                    format!("remote://host/file_{i}.mxf"),
                    format!("/local/file_{i}.mxf"),
                    TransferDirection::Download,
                    100_000 * (i + 1),
                )
            })
            .collect();

        // Start all, fail each with a unique message.
        for &id in &ids {
            mgr.start(id);
        }
        for (idx, &id) in ids.iter().enumerate() {
            mgr.update_progress(id, 10_000 * (idx as u64 + 1));
            mgr.fail(id, format!("packet loss at node {idx}"));
        }

        // Verify each job has its own distinct error and failed status.
        for (idx, &id) in ids.iter().enumerate() {
            let job = mgr.get(id).expect("must exist");
            assert_eq!(job.status, TransferStatus::Failed);
            let expected_err = format!("packet loss at node {idx}");
            assert_eq!(job.last_error.as_deref(), Some(expected_err.as_str()));
            assert_eq!(job.transferred_bytes, 10_000 * (idx as u64 + 1));
        }

        // Purge terminal jobs — all 5 are Failed, so all should be removed.
        let removed = mgr.purge_terminal();
        assert_eq!(removed, 5);
        assert!(mgr.is_empty());
    }

    /// Failing an already-failed job is idempotent (status stays `Failed`,
    /// last_error is overwritten with the new message).
    #[test]
    fn test_fail_is_idempotent_and_overwrites_error() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);

        mgr.fail(id, "first error");
        mgr.fail(id, "second error — overwrites first");

        let job = mgr.get(id).expect("must exist");
        assert_eq!(job.status, TransferStatus::Failed);
        assert_eq!(
            job.last_error.as_deref(),
            Some("second error — overwrites first")
        );
    }

    /// A paused job that receives a simulated network failure should be
    /// cancellable (cancel is valid for any non-terminal state).
    #[test]
    fn test_cancel_paused_job_after_partial_transfer() {
        let mut mgr = TransferManager::new();
        let id = submit_job(&mut mgr);

        mgr.start(id);
        mgr.update_progress(id, 256 * 1024); // 25 %
        mgr.pause(id);
        assert_eq!(
            mgr.get(id).expect("must exist").status,
            TransferStatus::Paused
        );

        // Network comes back but user decides to cancel.
        assert!(mgr.cancel(id));
        assert_eq!(
            mgr.get(id).expect("must exist").status,
            TransferStatus::Cancelled
        );
        // Cancelling a cancelled job must return false (terminal).
        assert!(!mgr.cancel(id));
    }

    /// Priority ordering must be preserved even when some higher-priority jobs
    /// have failed: only `Queued` jobs appear in `queued_by_priority`.
    #[test]
    fn test_priority_ordering_unaffected_by_failed_jobs() {
        let mut mgr = TransferManager::new();

        // Submit three jobs with ascending priorities.
        let id_low = mgr.submit(
            1,
            "/src/low",
            "/dst/low",
            TransferDirection::LocalToLocal,
            1,
        );
        let id_mid = mgr.submit(
            2,
            "/src/mid",
            "/dst/mid",
            TransferDirection::LocalToLocal,
            1,
        );
        let id_high = mgr.submit(
            3,
            "/src/high",
            "/dst/high",
            TransferDirection::LocalToLocal,
            1,
        );

        if let Some(j) = mgr.jobs.get_mut(&id_low) {
            j.priority = 1;
        }
        if let Some(j) = mgr.jobs.get_mut(&id_mid) {
            j.priority = 5;
        }
        if let Some(j) = mgr.jobs.get_mut(&id_high) {
            j.priority = 10;
        }

        // Fail the highest-priority job.
        mgr.start(id_high);
        mgr.fail(id_high, "NFS timeout");

        // queued_by_priority must now return only the two remaining queued jobs.
        let queued = mgr.queued_by_priority();
        assert_eq!(queued.len(), 2, "only non-terminal queued jobs returned");
        assert_eq!(queued[0].id, id_mid, "mid-priority first");
        assert_eq!(queued[1].id, id_low, "low-priority second");
    }

    /// Zero-byte transfer completes immediately (progress() == 1.0 from the start).
    #[test]
    fn test_zero_byte_transfer_progress_is_complete() {
        let mut mgr = TransferManager::new();
        let id = mgr.submit(
            99,
            "/src/empty",
            "/dst/empty",
            TransferDirection::LocalToLocal,
            0,
        );
        let job = mgr.get(id).expect("must exist");
        assert_eq!(job.progress(), 1.0, "zero-byte transfer must report 100 %");
    }
}
