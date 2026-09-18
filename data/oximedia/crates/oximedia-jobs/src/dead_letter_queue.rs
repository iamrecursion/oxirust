// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Dead letter queue for jobs that have exceeded their maximum retry attempts.
//!
//! When a job fails more times than its configured `max_retries`, it is moved
//! here instead of being silently dropped. Operators can then inspect, retry, or
//! purge entries as needed.

use crate::job::Job;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use thiserror::Error;
use uuid::Uuid;

/// Errors produced by dead-letter queue operations.
#[derive(Debug, Error)]
pub enum DlqError {
    /// The requested entry was not found in the DLQ.
    #[error("DLQ entry not found: {0}")]
    NotFound(Uuid),

    /// The DLQ has reached its capacity and cannot accept more entries.
    #[error("DLQ capacity exceeded (max {0})")]
    CapacityExceeded(usize),
}

/// A single entry in the dead letter queue, holding the failed job alongside
/// diagnostic metadata.
#[derive(Debug, Clone)]
pub struct DeadLetterEntry {
    /// The job that could not be completed.
    pub job: Job,
    /// Wall-clock time at which the job was admitted to the DLQ.
    pub failed_at: Instant,
    /// The error message / reason for ultimate failure.
    pub reason: String,
    /// Total number of execution attempts that were made.
    pub retry_count: u32,
}

impl DeadLetterEntry {
    /// Create a new DLQ entry.
    pub fn new(job: Job, reason: String, retry_count: u32) -> Self {
        Self {
            job,
            failed_at: Instant::now(),
            reason,
            retry_count,
        }
    }

    /// How long the entry has been sitting in the DLQ.
    pub fn age(&self) -> Duration {
        self.failed_at.elapsed()
    }

    /// Job identifier shortcut.
    pub fn job_id(&self) -> Uuid {
        self.job.id
    }
}

/// A bounded dead letter queue.
///
/// Entries are ordered by admission time (oldest first). When the queue is full
/// the caller receives [`DlqError::CapacityExceeded`] rather than silently
/// dropping data.
#[derive(Debug)]
pub struct DeadLetterQueue {
    /// Maximum number of entries the queue can hold simultaneously.
    max_jobs: usize,
    /// Storage — front is the oldest entry.
    jobs: VecDeque<DeadLetterEntry>,
}

impl DeadLetterQueue {
    /// Create a new dead letter queue with the specified capacity.
    ///
    /// A `max_jobs` of `0` means unlimited.
    pub fn new(max_jobs: usize) -> Self {
        Self {
            max_jobs,
            jobs: VecDeque::new(),
        }
    }

    /// Returns the configured maximum capacity.
    pub fn max_jobs(&self) -> usize {
        self.max_jobs
    }

    /// Current number of entries held in the queue.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Returns `true` when the queue holds no entries.
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Admit a failed job into the DLQ.
    ///
    /// # Errors
    ///
    /// Returns [`DlqError::CapacityExceeded`] when `max_jobs > 0` and the
    /// queue already holds `max_jobs` entries.
    pub fn admit(&mut self, job: Job, reason: String, retry_count: u32) -> Result<(), DlqError> {
        if self.max_jobs > 0 && self.jobs.len() >= self.max_jobs {
            return Err(DlqError::CapacityExceeded(self.max_jobs));
        }
        self.jobs
            .push_back(DeadLetterEntry::new(job, reason, retry_count));
        Ok(())
    }

    /// Look up an entry by job ID, returning an immutable reference.
    pub fn get(&self, id: Uuid) -> Option<&DeadLetterEntry> {
        self.jobs.iter().find(|e| e.job_id() == id)
    }

    /// Remove and return the entry with the given job ID so it can be
    /// requeued or inspected externally.
    ///
    /// # Errors
    ///
    /// Returns [`DlqError::NotFound`] if no entry with that ID exists.
    pub fn requeue(&mut self, id: Uuid) -> Result<Job, DlqError> {
        let pos = self
            .jobs
            .iter()
            .position(|e| e.job_id() == id)
            .ok_or(DlqError::NotFound(id))?;

        let entry = self.jobs.remove(pos).ok_or(DlqError::NotFound(id))?;

        let mut job = entry.job;
        // Reset the job so it is eligible for immediate processing.
        job.reset_for_retry();
        Ok(job)
    }

    /// Remove all entries whose [`DeadLetterEntry::age`] exceeds `max_age`.
    ///
    /// Returns the number of entries that were purged.
    pub fn purge_older_than(&mut self, max_age: Duration) -> usize {
        let before = self.jobs.len();
        self.jobs.retain(|e| e.age() < max_age);
        before - self.jobs.len()
    }

    /// Remove and return the oldest entry (front of the queue), if any.
    pub fn pop_oldest(&mut self) -> Option<DeadLetterEntry> {
        self.jobs.pop_front()
    }

    /// Iterate over all entries from oldest to newest (immutable).
    pub fn iter(&self) -> impl Iterator<Item = &DeadLetterEntry> {
        self.jobs.iter()
    }

    /// Drain the entire queue, returning all entries.
    pub fn drain_all(&mut self) -> Vec<DeadLetterEntry> {
        self.jobs.drain(..).collect()
    }

    /// Number of entries that have been in the queue longer than `age`.
    pub fn count_older_than(&self, age: Duration) -> usize {
        self.jobs.iter().filter(|e| e.age() >= age).count()
    }

    /// Peek at the oldest entry without removing it.
    pub fn peek_oldest(&self) -> Option<&DeadLetterEntry> {
        self.jobs.front()
    }
}

impl Default for DeadLetterQueue {
    /// Creates an unbounded dead letter queue.
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{Job, JobPayload, Priority, TranscodeParams};
    use std::thread;

    fn make_job(name: &str) -> Job {
        let params = TranscodeParams {
            input: "in.mp4".to_string(),
            output: "out.mp4".to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            video_bitrate: 4_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "fast".to_string(),
            hw_accel: None,
        };
        Job::new(
            name.to_string(),
            Priority::Normal,
            JobPayload::Transcode(params),
        )
    }

    // ── basic structure ────────────────────────────────────────────────────────

    #[test]
    fn test_new_queue_is_empty() {
        let dlq = DeadLetterQueue::new(10);
        assert!(dlq.is_empty());
        assert_eq!(dlq.len(), 0);
        assert_eq!(dlq.max_jobs(), 10);
    }

    #[test]
    fn test_default_is_unbounded() {
        let dlq = DeadLetterQueue::default();
        assert_eq!(dlq.max_jobs(), 0);
    }

    // ── admit ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_admit_single_job() {
        let mut dlq = DeadLetterQueue::new(5);
        let job = make_job("job-1");
        dlq.admit(job, "timeout".to_string(), 3)
            .expect("admit should succeed");
        assert_eq!(dlq.len(), 1);
    }

    #[test]
    fn test_admit_multiple_jobs() {
        let mut dlq = DeadLetterQueue::new(5);
        for i in 0..5_usize {
            dlq.admit(make_job(&format!("j{i}")), "err".to_string(), i as u32)
                .expect("admit should succeed");
        }
        assert_eq!(dlq.len(), 5);
    }

    #[test]
    fn test_admit_exceeds_capacity_returns_error() {
        let mut dlq = DeadLetterQueue::new(2);
        dlq.admit(make_job("j0"), "e".to_string(), 1)
            .expect("admit should succeed");
        dlq.admit(make_job("j1"), "e".to_string(), 1)
            .expect("admit should succeed");
        let result = dlq.admit(make_job("j2"), "e".to_string(), 1);
        assert!(matches!(result, Err(DlqError::CapacityExceeded(2))));
    }

    #[test]
    fn test_admit_unbounded_never_fails_capacity() {
        let mut dlq = DeadLetterQueue::new(0);
        for i in 0..100_usize {
            dlq.admit(make_job(&format!("j{i}")), "e".to_string(), 1)
                .expect("unlimited queue should always accept");
        }
        assert_eq!(dlq.len(), 100);
    }

    // ── get ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_get_existing_entry() {
        let mut dlq = DeadLetterQueue::new(10);
        let job = make_job("findme");
        let id = job.id;
        dlq.admit(job, "reason".to_string(), 2)
            .expect("admit should succeed");
        let entry = dlq.get(id).expect("entry should be found");
        assert_eq!(entry.job_id(), id);
        assert_eq!(entry.reason, "reason");
        assert_eq!(entry.retry_count, 2);
    }

    #[test]
    fn test_get_missing_returns_none() {
        let dlq = DeadLetterQueue::new(10);
        assert!(dlq.get(Uuid::new_v4()).is_none());
    }

    // ── requeue ────────────────────────────────────────────────────────────────

    #[test]
    fn test_requeue_returns_reset_job() {
        let mut dlq = DeadLetterQueue::new(10);
        let job = make_job("requeue-me");
        let id = job.id;
        dlq.admit(job, "err".to_string(), 3)
            .expect("admit should succeed");
        let requeued = dlq.requeue(id).expect("requeue should succeed");
        assert_eq!(requeued.id, id);
        // After reset_for_retry the job should be Pending again
        assert_eq!(requeued.status, crate::job::JobStatus::Pending);
        // Entry should be removed
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_requeue_nonexistent_returns_error() {
        let mut dlq = DeadLetterQueue::new(10);
        let result = dlq.requeue(Uuid::new_v4());
        assert!(matches!(result, Err(DlqError::NotFound(_))));
    }

    // ── purge_older_than ───────────────────────────────────────────────────────

    #[test]
    fn test_purge_older_than_removes_old_entries() {
        let mut dlq = DeadLetterQueue::new(10);
        dlq.admit(make_job("old"), "e".to_string(), 1)
            .expect("admit should succeed");

        // Wait a tiny bit so the entry's age is > 0 ns
        thread::sleep(Duration::from_millis(10));

        // Entries older than 1 ns should include our entry
        let purged = dlq.purge_older_than(Duration::from_nanos(1));
        assert_eq!(purged, 1);
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_purge_older_than_keeps_fresh_entries() {
        let mut dlq = DeadLetterQueue::new(10);
        dlq.admit(make_job("fresh"), "e".to_string(), 1)
            .expect("admit should succeed");
        // Purge anything older than a very long age — nothing should be removed
        let purged = dlq.purge_older_than(Duration::from_secs(3600));
        assert_eq!(purged, 0);
        assert_eq!(dlq.len(), 1);
    }

    // ── drain_all ──────────────────────────────────────────────────────────────

    #[test]
    fn test_drain_all_empties_queue() {
        let mut dlq = DeadLetterQueue::new(10);
        for i in 0..3_usize {
            dlq.admit(make_job(&format!("j{i}")), "e".to_string(), 1)
                .expect("admit should succeed");
        }
        let drained = dlq.drain_all();
        assert_eq!(drained.len(), 3);
        assert!(dlq.is_empty());
    }

    // ── iter / peek ────────────────────────────────────────────────────────────

    #[test]
    fn test_iter_order_is_oldest_first() {
        let mut dlq = DeadLetterQueue::new(10);
        let j1 = make_job("first");
        let id1 = j1.id;
        let j2 = make_job("second");
        let id2 = j2.id;
        dlq.admit(j1, "e".to_string(), 1)
            .expect("admit should succeed");
        dlq.admit(j2, "e".to_string(), 1)
            .expect("admit should succeed");

        let ids: Vec<Uuid> = dlq.iter().map(|e| e.job_id()).collect();
        assert_eq!(ids, vec![id1, id2]);
    }

    #[test]
    fn test_peek_oldest_does_not_remove() {
        let mut dlq = DeadLetterQueue::new(10);
        let job = make_job("peekable");
        let id = job.id;
        dlq.admit(job, "e".to_string(), 1)
            .expect("admit should succeed");
        let peeked = dlq.peek_oldest().expect("peek should return entry");
        assert_eq!(peeked.job_id(), id);
        assert_eq!(dlq.len(), 1); // still present
    }

    #[test]
    fn test_pop_oldest_removes_front() {
        let mut dlq = DeadLetterQueue::new(10);
        let j1 = make_job("first");
        let id1 = j1.id;
        dlq.admit(j1, "e".to_string(), 1)
            .expect("admit should succeed");
        dlq.admit(make_job("second"), "e".to_string(), 1)
            .expect("admit should succeed");

        let popped = dlq.pop_oldest().expect("pop should return entry");
        assert_eq!(popped.job_id(), id1);
        assert_eq!(dlq.len(), 1);
    }
}
