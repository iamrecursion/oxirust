// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job failure detection and recovery.
//!
//! Provides enumerations for classifying failures, records for storing their
//! history, and composable recovery strategies that can be applied
//! automatically when failures occur.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FailureType
// ---------------------------------------------------------------------------

/// Classification of the root cause of a render job failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureType {
    /// The worker process crashed unexpectedly (SIGSEGV, OOM-kill, etc.).
    WorkerCrash,
    /// The worker ran out of memory and was terminated.
    OutOfMemory,
    /// The job exceeded its allowed wall-clock time.
    Timeout,
    /// A network partition prevented communication with the worker.
    NetworkError,
    /// The job produced output that failed validation checks.
    InvalidOutput,
    /// The worker's scratch disk filled up during rendering.
    DiskFull,
    /// A human operator cancelled the job.
    UserCancel,
    /// A prerequisite job failed, making this job non-runnable.
    DependencyFailed,
}

impl FailureType {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::WorkerCrash => "WorkerCrash",
            Self::OutOfMemory => "OutOfMemory",
            Self::Timeout => "Timeout",
            Self::NetworkError => "NetworkError",
            Self::InvalidOutput => "InvalidOutput",
            Self::DiskFull => "DiskFull",
            Self::UserCancel => "UserCancel",
            Self::DependencyFailed => "DependencyFailed",
        }
    }

    /// Whether the failure is likely transient (and thus worth retrying).
    #[must_use]
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::NetworkError | Self::Timeout | Self::WorkerCrash)
    }
}

// ---------------------------------------------------------------------------
// FailureRecord
// ---------------------------------------------------------------------------

/// A record of a single failure event for a specific job attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    /// Identifier of the failed job.
    pub job_id: String,
    /// Identifier of the worker on which the failure occurred.
    pub worker_id: String,
    /// Classified root cause.
    pub failure_type: FailureType,
    /// Unix-millisecond timestamp of the failure.
    pub timestamp_ms: i64,
    /// Which attempt this was (1-based).
    pub attempt_number: u8,
    /// The raw error message from the worker.
    pub error_message: String,
}

impl FailureRecord {
    /// Construct a new failure record.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        worker_id: impl Into<String>,
        failure_type: FailureType,
        timestamp_ms: i64,
        attempt_number: u8,
        error_message: impl Into<String>,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            worker_id: worker_id.into(),
            failure_type,
            timestamp_ms,
            attempt_number,
            error_message: error_message.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// RecoveryStrategy
// ---------------------------------------------------------------------------

/// How the system should respond when a job fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Re-queue the job on the same (or any) worker with optional backoff.
    Retry {
        /// Maximum number of retry attempts.
        max_attempts: u8,
        /// Fixed delay in milliseconds before each retry.
        backoff_ms: u64,
    },

    /// Re-queue the job, but steer it away from the failing worker.
    RetryOnDifferentWorker {
        /// Maximum number of retry attempts.
        max_attempts: u8,
    },

    /// Restart rendering from a specific frame (e.g. from the last checkpoint).
    PartialRestart {
        /// The frame number from which to resume.
        resume_from_frame: u64,
    },

    /// Give up and mark the job as permanently failed.
    Abort,

    /// Mark the job for manual human review.
    Escalate,
}

impl RecoveryStrategy {
    /// Whether this strategy involves retrying at all.
    #[must_use]
    pub fn will_retry(&self) -> bool {
        matches!(
            self,
            Self::Retry { .. } | Self::RetryOnDifferentWorker { .. }
        )
    }

    /// Maximum allowed attempts (returns 1 for non-retry strategies).
    #[must_use]
    pub fn max_attempts(&self) -> u8 {
        match self {
            Self::Retry { max_attempts, .. } | Self::RetryOnDifferentWorker { max_attempts } => {
                *max_attempts
            }
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// RecoveryPolicy
// ---------------------------------------------------------------------------

/// A mapping from failure types to the recovery strategy that should be used.
#[derive(Debug, Clone)]
pub struct RecoveryPolicy {
    /// Per-failure-type strategies.
    pub strategies: HashMap<FailureType, RecoveryStrategy>,
    /// Fallback strategy when no specific mapping exists.
    pub default_strategy: RecoveryStrategy,
}

impl RecoveryPolicy {
    /// Create a policy with no entries and an `Abort` fallback.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            strategies: HashMap::new(),
            default_strategy: RecoveryStrategy::Abort,
        }
    }

    /// Register a strategy for a specific failure type.
    pub fn set(&mut self, failure_type: FailureType, strategy: RecoveryStrategy) {
        self.strategies.insert(failure_type, strategy);
    }

    /// Look up the recovery strategy for a failure, falling back to the default.
    #[must_use]
    pub fn get_strategy(&self, failure: &FailureType) -> &RecoveryStrategy {
        self.strategies
            .get(failure)
            .unwrap_or(&self.default_strategy)
    }
}

impl Default for RecoveryPolicy {
    /// Opinionated defaults covering the most common failure patterns.
    fn default() -> Self {
        let mut policy = Self::empty();

        policy.set(
            FailureType::WorkerCrash,
            RecoveryStrategy::RetryOnDifferentWorker { max_attempts: 3 },
        );
        policy.set(
            FailureType::OutOfMemory,
            RecoveryStrategy::RetryOnDifferentWorker { max_attempts: 2 },
        );
        policy.set(
            FailureType::Timeout,
            RecoveryStrategy::Retry {
                max_attempts: 2,
                backoff_ms: 5_000,
            },
        );
        policy.set(
            FailureType::NetworkError,
            RecoveryStrategy::Retry {
                max_attempts: 5,
                backoff_ms: 1_000,
            },
        );
        policy.set(FailureType::InvalidOutput, RecoveryStrategy::Abort);
        policy.set(FailureType::DiskFull, RecoveryStrategy::Escalate);
        policy.set(FailureType::UserCancel, RecoveryStrategy::Abort);
        policy.set(FailureType::DependencyFailed, RecoveryStrategy::Abort);

        policy
    }
}

// ---------------------------------------------------------------------------
// JobRetry
// ---------------------------------------------------------------------------

/// Represents a rescheduled attempt at a previously-failed job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRetry {
    /// The original job identifier.
    pub original_job_id: String,
    /// The attempt number for this retry (1-based).
    pub attempt: u8,
    /// Optional preferred worker to route this retry to.
    pub preferred_worker: Option<String>,
    /// How long to wait before dispatching (milliseconds).
    pub backoff_ms: u64,
}

// ---------------------------------------------------------------------------
// RecoveryAction
// ---------------------------------------------------------------------------

/// The concrete action to take following a failure analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Reschedule the job as a new attempt.
    Schedule(JobRetry),
    /// Re-run from the given frame index (partial restart).
    PartialReschedule(u64),
    /// Permanently cancel the job.
    Cancel,
    /// Send an alert with the given message (for human review).
    Alert(String),
    /// Retry a specific chunk/tile instead of the entire frame.
    RetryChunk(ChunkId),
}

// ---------------------------------------------------------------------------
// ChunkId / ChunkRetryState / ChunkRetryManager
// ---------------------------------------------------------------------------

/// Identifies a single rendered tile/chunk within a tiled frame.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId {
    /// The frame this chunk belongs to.
    pub frame_id: u64,
    /// Index of this tile within the frame's tile grid.
    pub tile_index: u32,
    /// Pixel region this tile covers: (x, y, width, height).
    pub tile_region: (u32, u32, u32, u32),
}

impl ChunkId {
    /// Create a new chunk identifier.
    pub fn new(frame_id: u64, tile_index: u32, tile_region: (u32, u32, u32, u32)) -> Self {
        Self {
            frame_id,
            tile_index,
            tile_region,
        }
    }
}

/// Tracks retry state for a single in-flight chunk.
#[derive(Debug, Clone)]
pub struct ChunkRetryState {
    /// The chunk being retried.
    pub chunk: ChunkId,
    /// Current attempt number (1-based; incremented on each failure).
    pub attempt: u32,
    /// Maximum allowed attempts before giving up.
    pub max_attempts: u32,
}

/// Manages fine-grained chunk-level retry state for tiled frame rendering.
///
/// Instead of retrying an entire frame when one tile fails, the manager
/// tracks individual tile outcomes and only re-dispatches failed tiles.
pub struct ChunkRetryManager {
    /// In-flight chunks: (frame_id, tile_index) -> ChunkRetryState.
    pending: HashMap<(u64, u32), ChunkRetryState>,
    /// Default maximum retry attempts for new chunks.
    max_attempts: u32,
}

impl ChunkRetryManager {
    /// Create a new manager with the given default max attempts.
    pub fn new(max_attempts: u32) -> Self {
        Self {
            pending: HashMap::new(),
            max_attempts,
        }
    }

    /// Called when a chunk fails. Returns the retry state if retries remain,
    /// or `None` if the chunk has exhausted its allowed attempts.
    ///
    /// Internally increments the attempt counter. The first call for a chunk
    /// initialises it with `attempt = 1`; once `attempt >= max_attempts` the
    /// chunk is considered permanently failed and `None` is returned.
    pub fn on_chunk_failure(&mut self, chunk: ChunkId) -> Option<ChunkRetryState> {
        let key = (chunk.frame_id, chunk.tile_index);
        let max = self.max_attempts;
        let state = self.pending.entry(key).or_insert_with(|| ChunkRetryState {
            chunk: chunk.clone(),
            attempt: 0,
            max_attempts: max,
        });
        state.attempt += 1;
        if state.attempt < state.max_attempts {
            Some(state.clone())
        } else {
            None
        }
    }

    /// Called when a chunk succeeds; removes it from the pending map.
    pub fn on_chunk_success(&mut self, frame_id: u64, tile_index: u32) {
        self.pending.remove(&(frame_id, tile_index));
    }

    /// Returns all pending retry states for a given frame.
    pub fn pending_chunks_for_frame(&self, frame_id: u64) -> Vec<&ChunkRetryState> {
        self.pending
            .iter()
            .filter(|((fid, _), _)| *fid == frame_id)
            .map(|(_, state)| state)
            .collect()
    }

    /// Returns `true` if no chunks for `frame_id` remain pending.
    ///
    /// `total_tiles` is accepted for API symmetry and guards against the
    /// degenerate case of a zero-tile frame (which can never be complete).
    pub fn is_frame_complete(&self, frame_id: u64, total_tiles: u32) -> bool {
        if total_tiles == 0 {
            return false;
        }
        self.pending
            .keys()
            .filter(|(fid, _)| *fid == frame_id)
            .count()
            == 0
    }
}

// ---------------------------------------------------------------------------
// FailureAnalyzer
// ---------------------------------------------------------------------------

/// Accumulates failure records and derives insights or recovery actions.
#[derive(Debug, Default)]
pub struct FailureAnalyzer {
    /// All recorded failure events.
    pub records: Vec<FailureRecord>,
}

impl FailureAnalyzer {
    /// Create an empty analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a failure record.
    pub fn record(&mut self, record: FailureRecord) {
        self.records.push(record);
    }

    /// Return the failure type that appears most often across all records.
    ///
    /// Returns `None` when there are no records.
    #[must_use]
    pub fn most_common_failure_type(&self) -> Option<FailureType> {
        let mut counts: HashMap<FailureType, usize> = HashMap::new();
        for r in &self.records {
            *counts.entry(r.failure_type).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(ft, _)| ft)
    }

    /// Fraction of this worker's jobs that have failed (0.0 when no records).
    #[must_use]
    pub fn worker_failure_rate(&self, worker_id: &str) -> f32 {
        let total = self
            .records
            .iter()
            .filter(|r| r.worker_id == worker_id)
            .count();
        if total == 0 {
            return 0.0;
        }
        // All records stored here represent failures, so rate = 1.0 per job attempt.
        // More usefully: unique jobs that failed vs total attempts on this worker.
        let unique_jobs: std::collections::HashSet<&str> = self
            .records
            .iter()
            .filter(|r| r.worker_id == worker_id)
            .map(|r| r.job_id.as_str())
            .collect();
        unique_jobs.len() as f32 / total as f32
    }

    /// Determine the recovery action for a given failure, consulting the policy.
    #[must_use]
    pub fn suggest_recovery(
        &self,
        record: &FailureRecord,
        policy: &RecoveryPolicy,
    ) -> RecoveryAction {
        let strategy = policy.get_strategy(&record.failure_type);

        match strategy {
            RecoveryStrategy::Retry {
                max_attempts,
                backoff_ms,
            } => {
                if record.attempt_number >= *max_attempts {
                    return RecoveryAction::Alert(format!(
                        "Job {} exhausted {} retry attempts: {}",
                        record.job_id, max_attempts, record.error_message
                    ));
                }
                let delay = exponential_backoff_ms(record.attempt_number, *backoff_ms, u64::MAX);
                RecoveryAction::Schedule(JobRetry {
                    original_job_id: record.job_id.clone(),
                    attempt: record.attempt_number + 1,
                    preferred_worker: None,
                    backoff_ms: delay,
                })
            }

            RecoveryStrategy::RetryOnDifferentWorker { max_attempts } => {
                if record.attempt_number >= *max_attempts {
                    return RecoveryAction::Alert(format!(
                        "Job {} exhausted {} retries on different workers",
                        record.job_id, max_attempts
                    ));
                }
                RecoveryAction::Schedule(JobRetry {
                    original_job_id: record.job_id.clone(),
                    attempt: record.attempt_number + 1,
                    preferred_worker: None,
                    backoff_ms: 0,
                })
            }

            RecoveryStrategy::PartialRestart { resume_from_frame } => {
                RecoveryAction::PartialReschedule(*resume_from_frame)
            }

            RecoveryStrategy::Abort => RecoveryAction::Cancel,

            RecoveryStrategy::Escalate => RecoveryAction::Alert(format!(
                "Manual intervention required for job {}: {}",
                record.job_id, record.error_message
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute an exponential backoff delay, capped at `max_ms`.
///
/// Formula: `min(base_ms * 2^attempt, max_ms)`.
///
/// Uses saturating arithmetic to avoid overflow on large `attempt` values.
#[must_use]
pub fn exponential_backoff_ms(attempt: u8, base_ms: u64, max_ms: u64) -> u64 {
    let shift = u32::from(attempt);
    let multiplier = if shift >= 64 { u64::MAX } else { 1u64 << shift };
    let raw = base_ms.saturating_mul(multiplier);
    raw.min(max_ms)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        job_id: &str,
        worker_id: &str,
        failure_type: FailureType,
        attempt: u8,
    ) -> FailureRecord {
        FailureRecord::new(job_id, worker_id, failure_type, 0, attempt, "test error")
    }

    // --- FailureType ---

    #[test]
    fn test_failure_type_labels() {
        assert_eq!(FailureType::WorkerCrash.label(), "WorkerCrash");
        assert_eq!(FailureType::UserCancel.label(), "UserCancel");
    }

    #[test]
    fn test_failure_type_transient() {
        assert!(FailureType::NetworkError.is_transient());
        assert!(FailureType::Timeout.is_transient());
        assert!(FailureType::WorkerCrash.is_transient());
        assert!(!FailureType::InvalidOutput.is_transient());
        assert!(!FailureType::UserCancel.is_transient());
    }

    // --- RecoveryStrategy ---

    #[test]
    fn test_strategy_will_retry() {
        assert!(RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 1000
        }
        .will_retry());
        assert!(RecoveryStrategy::RetryOnDifferentWorker { max_attempts: 2 }.will_retry());
        assert!(!RecoveryStrategy::Abort.will_retry());
        assert!(!RecoveryStrategy::Escalate.will_retry());
    }

    #[test]
    fn test_strategy_max_attempts() {
        assert_eq!(
            RecoveryStrategy::Retry {
                max_attempts: 5,
                backoff_ms: 0
            }
            .max_attempts(),
            5
        );
        assert_eq!(RecoveryStrategy::Abort.max_attempts(), 1);
    }

    // --- RecoveryPolicy::default ---

    #[test]
    fn test_policy_default_worker_crash() {
        let policy = RecoveryPolicy::default();
        let s = policy.get_strategy(&FailureType::WorkerCrash);
        assert!(s.will_retry());
        assert_eq!(s.max_attempts(), 3);
    }

    #[test]
    fn test_policy_default_oom() {
        let policy = RecoveryPolicy::default();
        let s = policy.get_strategy(&FailureType::OutOfMemory);
        assert!(s.will_retry());
        assert_eq!(s.max_attempts(), 2);
    }

    #[test]
    fn test_policy_default_timeout() {
        let policy = RecoveryPolicy::default();
        match policy.get_strategy(&FailureType::Timeout) {
            RecoveryStrategy::Retry {
                max_attempts,
                backoff_ms,
            } => {
                assert_eq!(*max_attempts, 2);
                assert_eq!(*backoff_ms, 5_000);
            }
            other => panic!("unexpected strategy: {other:?}"),
        }
    }

    #[test]
    fn test_policy_default_network() {
        let policy = RecoveryPolicy::default();
        match policy.get_strategy(&FailureType::NetworkError) {
            RecoveryStrategy::Retry {
                max_attempts,
                backoff_ms,
            } => {
                assert_eq!(*max_attempts, 5);
                assert_eq!(*backoff_ms, 1_000);
            }
            other => panic!("unexpected strategy: {other:?}"),
        }
    }

    #[test]
    fn test_policy_default_invalid_output_aborts() {
        let policy = RecoveryPolicy::default();
        assert!(matches!(
            policy.get_strategy(&FailureType::InvalidOutput),
            RecoveryStrategy::Abort
        ));
    }

    // --- FailureAnalyzer ---

    #[test]
    fn test_analyzer_most_common_empty() {
        let analyzer = FailureAnalyzer::new();
        assert!(analyzer.most_common_failure_type().is_none());
    }

    #[test]
    fn test_analyzer_most_common_single() {
        let mut analyzer = FailureAnalyzer::new();
        analyzer.record(make_record("j1", "w1", FailureType::Timeout, 1));
        assert_eq!(
            analyzer.most_common_failure_type(),
            Some(FailureType::Timeout)
        );
    }

    #[test]
    fn test_analyzer_most_common_mixed() {
        let mut analyzer = FailureAnalyzer::new();
        analyzer.record(make_record("j1", "w1", FailureType::NetworkError, 1));
        analyzer.record(make_record("j2", "w1", FailureType::NetworkError, 1));
        analyzer.record(make_record("j3", "w1", FailureType::Timeout, 1));
        assert_eq!(
            analyzer.most_common_failure_type(),
            Some(FailureType::NetworkError)
        );
    }

    #[test]
    fn test_analyzer_worker_failure_rate_zero() {
        let analyzer = FailureAnalyzer::new();
        assert_eq!(analyzer.worker_failure_rate("w1"), 0.0);
    }

    #[test]
    fn test_analyzer_worker_failure_rate_nonzero() {
        let mut analyzer = FailureAnalyzer::new();
        analyzer.record(make_record("j1", "w1", FailureType::Timeout, 1));
        analyzer.record(make_record("j2", "w1", FailureType::Timeout, 1));
        let rate = analyzer.worker_failure_rate("w1");
        assert!(rate > 0.0);
    }

    #[test]
    fn test_suggest_recovery_retry_within_limit() {
        let analyzer = FailureAnalyzer::new();
        let policy = RecoveryPolicy::default();
        let record = make_record("job-1", "worker-1", FailureType::NetworkError, 1);
        match analyzer.suggest_recovery(&record, &policy) {
            RecoveryAction::Schedule(retry) => {
                assert_eq!(retry.original_job_id, "job-1");
                assert_eq!(retry.attempt, 2);
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn test_suggest_recovery_retry_exhausted() {
        let analyzer = FailureAnalyzer::new();
        let policy = RecoveryPolicy::default();
        // NetworkError allows 5 attempts; attempt 5 should trigger alert
        let record = make_record("job-1", "worker-1", FailureType::NetworkError, 5);
        assert!(matches!(
            analyzer.suggest_recovery(&record, &policy),
            RecoveryAction::Alert(_)
        ));
    }

    #[test]
    fn test_suggest_recovery_abort() {
        let analyzer = FailureAnalyzer::new();
        let policy = RecoveryPolicy::default();
        let record = make_record("job-2", "worker-1", FailureType::InvalidOutput, 1);
        assert!(matches!(
            analyzer.suggest_recovery(&record, &policy),
            RecoveryAction::Cancel
        ));
    }

    #[test]
    fn test_suggest_recovery_escalate() {
        let analyzer = FailureAnalyzer::new();
        let policy = RecoveryPolicy::default();
        let record = make_record("job-3", "worker-1", FailureType::DiskFull, 1);
        assert!(matches!(
            analyzer.suggest_recovery(&record, &policy),
            RecoveryAction::Alert(_)
        ));
    }

    // --- exponential_backoff_ms ---

    #[test]
    fn test_backoff_attempt_0() {
        assert_eq!(exponential_backoff_ms(0, 1_000, 60_000), 1_000);
    }

    #[test]
    fn test_backoff_attempt_1() {
        assert_eq!(exponential_backoff_ms(1, 1_000, 60_000), 2_000);
    }

    #[test]
    fn test_backoff_attempt_3() {
        assert_eq!(exponential_backoff_ms(3, 1_000, 60_000), 8_000);
    }

    #[test]
    fn test_backoff_capped() {
        assert_eq!(exponential_backoff_ms(10, 1_000, 5_000), 5_000);
    }

    #[test]
    fn test_backoff_no_overflow() {
        // Large attempt should not panic
        let v = exponential_backoff_ms(63, 1, u64::MAX);
        assert!(v > 0);
    }

    // --- ChunkRetryManager ---

    #[test]
    fn test_chunk_retry_manager_retry_within_limit() {
        let mut mgr = ChunkRetryManager::new(3);
        let chunk = ChunkId::new(1, 0, (0, 0, 256, 256));
        let state = mgr.on_chunk_failure(chunk.clone());
        assert!(state.is_some());
        let s = state.expect("state present");
        assert_eq!(s.attempt, 1);
    }

    #[test]
    fn test_chunk_retry_manager_exhausted() {
        let mut mgr = ChunkRetryManager::new(2);
        let chunk = ChunkId::new(1, 0, (0, 0, 256, 256));
        mgr.on_chunk_failure(chunk.clone()); // attempt 1
        let result = mgr.on_chunk_failure(chunk.clone()); // attempt 2 = max
        assert!(result.is_none(), "should be exhausted at max_attempts");
    }

    #[test]
    fn test_chunk_retry_manager_success_removes() {
        let mut mgr = ChunkRetryManager::new(3);
        let chunk = ChunkId::new(2, 5, (512, 0, 256, 256));
        mgr.on_chunk_failure(chunk.clone());
        mgr.on_chunk_success(2, 5);
        assert!(mgr.pending_chunks_for_frame(2).is_empty());
    }

    #[test]
    fn test_chunk_retry_manager_pending_for_frame() {
        let mut mgr = ChunkRetryManager::new(5);
        mgr.on_chunk_failure(ChunkId::new(10, 0, (0, 0, 128, 128)));
        mgr.on_chunk_failure(ChunkId::new(10, 1, (128, 0, 128, 128)));
        mgr.on_chunk_failure(ChunkId::new(11, 0, (0, 0, 128, 128)));
        let pending = mgr.pending_chunks_for_frame(10);
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_chunk_retry_manager_frame_complete_no_pending() {
        let mgr = ChunkRetryManager::new(3);
        // No pending chunks and total_tiles > 0 → complete
        assert!(mgr.is_frame_complete(99, 4));
    }

    #[test]
    fn test_chunk_retry_manager_frame_not_complete_with_pending() {
        let mut mgr = ChunkRetryManager::new(5);
        mgr.on_chunk_failure(ChunkId::new(7, 0, (0, 0, 64, 64)));
        assert!(!mgr.is_frame_complete(7, 4));
    }

    #[test]
    fn test_chunk_retry_manager_zero_tiles_not_complete() {
        let mgr = ChunkRetryManager::new(3);
        // total_tiles == 0 → never complete
        assert!(!mgr.is_frame_complete(99, 0));
    }
}
