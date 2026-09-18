#![allow(dead_code)]
//! Worker health scoring with automatic quarantine of failing workers.
//!
//! ## Model
//!
//! Each worker has a **health score** in `[0.0, 1.0]` where `1.0` is fully
//! healthy and `0.0` is completely failed.  The score is updated via an
//! exponentially-weighted moving average (EWMA) of individual job outcomes.
//!
//! A worker is automatically **quarantined** when its score drops below
//! `quarantine_threshold`.  A quarantined worker is excluded from new job
//! assignments and re-evaluated after a configurable cool-down.  If the score
//! recovers above `recovery_threshold` the worker is reinstated.
//!
//! Workers that stay quarantined beyond `max_quarantine_duration` are
//! permanently marked as `Evicted` and removed from the scheduling pool.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Score and outcome
// ---------------------------------------------------------------------------

/// Outcome of a single job/task execution on a worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JobOutcome {
    /// The job completed successfully.
    Success,
    /// The job failed with a transient error (counted against the score).
    TransientFailure,
    /// The job failed with a permanent error (counted more heavily).
    PermanentFailure,
    /// The job timed out (moderate penalty).
    Timeout,
}

impl JobOutcome {
    /// Map the outcome to a sample value in `[0.0, 1.0]` used for EWMA.
    ///
    /// - `1.0` → success (boosts health)
    /// - Values < `1.0` → failure (degrades health, weight varies by severity)
    #[must_use]
    pub fn sample_value(self) -> f64 {
        match self {
            Self::Success => 1.0,
            Self::Timeout => 0.5,
            Self::TransientFailure => 0.2,
            Self::PermanentFailure => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Worker health record
// ---------------------------------------------------------------------------

/// Health scoring policy.
#[derive(Debug, Clone)]
pub struct HealthScoringPolicy {
    /// EWMA smoothing factor `α ∈ (0, 1]`.
    ///
    /// A higher value gives more weight to recent outcomes; a lower value
    /// smooths over a longer history.
    pub ewma_alpha: f64,
    /// Score below which the worker is quarantined.
    pub quarantine_threshold: f64,
    /// Score above which a quarantined worker is reinstated.
    pub recovery_threshold: f64,
    /// Minimum time a worker must remain quarantined before re-evaluation.
    pub quarantine_cool_down: Duration,
    /// Maximum time a worker may remain quarantined before eviction.
    pub max_quarantine_duration: Duration,
    /// Minimum number of job outcomes before a score can trigger quarantine.
    pub min_samples_for_quarantine: usize,
}

impl Default for HealthScoringPolicy {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.3,
            quarantine_threshold: 0.4,
            recovery_threshold: 0.7,
            quarantine_cool_down: Duration::from_secs(60),
            max_quarantine_duration: Duration::from_secs(600),
            min_samples_for_quarantine: 5,
        }
    }
}

/// Operational status of a worker from the health-scoring perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealthStatus {
    /// Worker is healthy and accepting jobs.
    Healthy,
    /// Worker score is degraded but still above the quarantine threshold.
    Degraded,
    /// Worker has been excluded from new job assignments pending recovery.
    Quarantined,
    /// Worker remained quarantined too long and has been removed.
    Evicted,
}

impl std::fmt::Display for WorkerHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Quarantined => write!(f, "Quarantined"),
            Self::Evicted => write!(f, "Evicted"),
        }
    }
}

/// Health record for a single worker.
#[derive(Debug)]
pub struct WorkerHealthRecord {
    /// Worker identifier.
    pub worker_id: String,
    /// Current EWMA health score (`0.0` = fully failed, `1.0` = perfect).
    pub score: f64,
    /// Current operational status.
    pub status: WorkerHealthStatus,
    /// Total number of job outcomes recorded.
    pub sample_count: usize,
    /// Instant when the worker was quarantined (if applicable).
    quarantined_at: Option<Instant>,
    /// Monotonic instant of the last score update.
    last_updated: Instant,
}

impl WorkerHealthRecord {
    /// Create a new record starting at full health.
    #[must_use]
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            score: 1.0,
            status: WorkerHealthStatus::Healthy,
            sample_count: 0,
            quarantined_at: None,
            last_updated: Instant::now(),
        }
    }

    /// Return `true` when the worker can be assigned new jobs.
    #[must_use]
    pub fn is_assignable(&self) -> bool {
        matches!(
            self.status,
            WorkerHealthStatus::Healthy | WorkerHealthStatus::Degraded
        )
    }

    /// Duration since the worker was quarantined.  Returns `None` if the
    /// worker is not currently quarantined.
    #[must_use]
    pub fn quarantine_duration(&self) -> Option<Duration> {
        self.quarantined_at.map(|t| t.elapsed())
    }
}

// ---------------------------------------------------------------------------
// Health scorer
// ---------------------------------------------------------------------------

/// Manages health scoring and quarantine for a pool of workers.
pub struct WorkerHealthScorer {
    records: HashMap<String, WorkerHealthRecord>,
    policy: HealthScoringPolicy,
}

impl WorkerHealthScorer {
    /// Create a scorer with the given policy.
    #[must_use]
    pub fn new(policy: HealthScoringPolicy) -> Self {
        Self {
            records: HashMap::new(),
            policy,
        }
    }

    /// Create a scorer with the default policy.
    #[must_use]
    pub fn default_policy() -> Self {
        Self::new(HealthScoringPolicy::default())
    }

    /// Register a worker with initial full-health score.
    ///
    /// If the worker is already registered, this is a no-op.
    pub fn register_worker(&mut self, worker_id: impl Into<String>) {
        let id = worker_id.into();
        self.records
            .entry(id.clone())
            .or_insert_with(|| WorkerHealthRecord::new(id));
    }

    /// Record a job outcome for a worker and update its health score.
    ///
    /// If the worker is not registered it is silently added with a starting
    /// score of `1.0`.
    pub fn record_outcome(&mut self, worker_id: &str, outcome: JobOutcome) {
        let policy = &self.policy;
        let sample = outcome.sample_value();
        let alpha = policy.ewma_alpha;

        let record = self
            .records
            .entry(worker_id.to_string())
            .or_insert_with(|| WorkerHealthRecord::new(worker_id));

        // EWMA update
        record.score = alpha * sample + (1.0 - alpha) * record.score;
        record.sample_count += 1;
        record.last_updated = Instant::now();

        // Re-classify status.
        self.classify(worker_id);
    }

    /// Re-evaluate the health status of a worker based on its current score,
    /// quarantine age, and policy thresholds.
    fn classify(&mut self, worker_id: &str) {
        let policy = &self.policy;

        let record = match self.records.get_mut(worker_id) {
            Some(r) => r,
            None => return,
        };

        match record.status {
            WorkerHealthStatus::Evicted => {
                // Evicted workers are final; do not change status.
            }
            WorkerHealthStatus::Quarantined => {
                let q_at = record.quarantined_at.expect("quarantined_at set");
                let elapsed = q_at.elapsed();

                if elapsed >= policy.max_quarantine_duration {
                    // Too long in quarantine → evict.
                    record.status = WorkerHealthStatus::Evicted;
                } else if elapsed >= policy.quarantine_cool_down
                    && record.score >= policy.recovery_threshold
                {
                    // Recovered after cool-down.
                    record.status = WorkerHealthStatus::Healthy;
                    record.quarantined_at = None;
                }
                // Otherwise stay quarantined.
            }
            WorkerHealthStatus::Healthy | WorkerHealthStatus::Degraded => {
                if record.sample_count >= policy.min_samples_for_quarantine
                    && record.score < policy.quarantine_threshold
                {
                    record.status = WorkerHealthStatus::Quarantined;
                    record.quarantined_at = Some(Instant::now());
                } else if record.score >= policy.recovery_threshold {
                    record.status = WorkerHealthStatus::Healthy;
                } else {
                    record.status = WorkerHealthStatus::Degraded;
                }
            }
        }
    }

    /// Trigger a periodic re-evaluation of all quarantined workers, advancing
    /// them to `Evicted` if they have exceeded `max_quarantine_duration`.
    ///
    /// Returns the IDs of any newly evicted workers.
    pub fn sweep_quarantine(&mut self) -> Vec<String> {
        let worker_ids: Vec<String> = self.records.keys().cloned().collect();
        let mut evicted = Vec::new();

        for id in worker_ids {
            let prev = self.records[&id].status;
            self.classify(&id);
            let curr = self.records[&id].status;
            if prev != WorkerHealthStatus::Evicted && curr == WorkerHealthStatus::Evicted {
                evicted.push(id);
            }
        }

        evicted
    }

    /// Get the health record for a worker.
    #[must_use]
    pub fn get_record(&self, worker_id: &str) -> Option<&WorkerHealthRecord> {
        self.records.get(worker_id)
    }

    /// Return IDs of all workers that are currently assignable (Healthy or Degraded).
    #[must_use]
    pub fn assignable_workers(&self) -> Vec<&str> {
        self.records
            .values()
            .filter(|r| r.is_assignable())
            .map(|r| r.worker_id.as_str())
            .collect()
    }

    /// Return IDs of all quarantined workers.
    #[must_use]
    pub fn quarantined_workers(&self) -> Vec<&str> {
        self.records
            .values()
            .filter(|r| r.status == WorkerHealthStatus::Quarantined)
            .map(|r| r.worker_id.as_str())
            .collect()
    }

    /// Return IDs of all evicted workers.
    #[must_use]
    pub fn evicted_workers(&self) -> Vec<&str> {
        self.records
            .values()
            .filter(|r| r.status == WorkerHealthStatus::Evicted)
            .map(|r| r.worker_id.as_str())
            .collect()
    }

    /// Total number of tracked workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.records.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scorer_low_threshold() -> WorkerHealthScorer {
        WorkerHealthScorer::new(HealthScoringPolicy {
            ewma_alpha: 0.5,
            quarantine_threshold: 0.4,
            recovery_threshold: 0.7,
            quarantine_cool_down: Duration::from_millis(10),
            max_quarantine_duration: Duration::from_millis(500),
            min_samples_for_quarantine: 3,
        })
    }

    #[test]
    fn test_register_worker_starts_healthy() {
        let mut scorer = WorkerHealthScorer::default_policy();
        scorer.register_worker("w1");
        let rec = scorer.get_record("w1").expect("record");
        assert_eq!(rec.status, WorkerHealthStatus::Healthy);
        assert!((rec.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_keeps_score_high() {
        let mut scorer = make_scorer_low_threshold();
        scorer.register_worker("w1");
        for _ in 0..10 {
            scorer.record_outcome("w1", JobOutcome::Success);
        }
        let rec = scorer.get_record("w1").expect("record");
        assert!(rec.score > 0.9);
        assert_eq!(rec.status, WorkerHealthStatus::Healthy);
    }

    #[test]
    fn test_repeated_failures_trigger_quarantine() {
        let mut scorer = make_scorer_low_threshold();
        scorer.register_worker("w1");
        for _ in 0..10 {
            scorer.record_outcome("w1", JobOutcome::PermanentFailure);
        }
        let rec = scorer.get_record("w1").expect("record");
        assert!(
            matches!(
                rec.status,
                WorkerHealthStatus::Quarantined | WorkerHealthStatus::Evicted
            ),
            "expected quarantined or evicted, got {:?}",
            rec.status
        );
    }

    #[test]
    fn test_min_samples_prevents_early_quarantine() {
        let mut scorer = WorkerHealthScorer::new(HealthScoringPolicy {
            min_samples_for_quarantine: 10,
            ewma_alpha: 0.9,
            quarantine_threshold: 0.4,
            recovery_threshold: 0.7,
            quarantine_cool_down: Duration::from_millis(10),
            max_quarantine_duration: Duration::from_millis(500),
        });
        scorer.register_worker("w1");
        // Only 2 failures, but min_samples = 10
        scorer.record_outcome("w1", JobOutcome::PermanentFailure);
        scorer.record_outcome("w1", JobOutcome::PermanentFailure);
        let rec = scorer.get_record("w1").expect("record");
        // Should be Degraded, not Quarantined yet
        assert_ne!(rec.status, WorkerHealthStatus::Quarantined);
    }

    #[test]
    fn test_quarantine_evicts_after_max_duration() {
        let mut scorer = WorkerHealthScorer::new(HealthScoringPolicy {
            ewma_alpha: 0.9,
            quarantine_threshold: 0.4,
            recovery_threshold: 0.7,
            quarantine_cool_down: Duration::from_millis(1),
            max_quarantine_duration: Duration::from_millis(5),
            min_samples_for_quarantine: 3,
        });
        scorer.register_worker("w1");
        for _ in 0..10 {
            scorer.record_outcome("w1", JobOutcome::PermanentFailure);
        }
        // Give the quarantine time to expire.
        std::thread::sleep(Duration::from_millis(20));
        let evicted = scorer.sweep_quarantine();
        // If it was quarantined before, it should now be evicted.
        let rec = scorer.get_record("w1").expect("record");
        assert!(
            evicted.contains(&"w1".to_string()) || rec.status == WorkerHealthStatus::Evicted,
            "expected eviction"
        );
    }

    #[test]
    fn test_assignable_workers_excludes_quarantined() {
        let mut scorer = make_scorer_low_threshold();
        scorer.register_worker("good");
        scorer.register_worker("bad");
        for _ in 0..10 {
            scorer.record_outcome("bad", JobOutcome::PermanentFailure);
        }
        let assignable = scorer.assignable_workers();
        assert!(
            assignable.contains(&"good"),
            "good worker should be assignable"
        );
        assert!(
            !assignable.contains(&"bad"),
            "bad worker should be quarantined"
        );
    }

    #[test]
    fn test_quarantined_workers_list() {
        let mut scorer = make_scorer_low_threshold();
        scorer.register_worker("w1");
        for _ in 0..10 {
            scorer.record_outcome("w1", JobOutcome::PermanentFailure);
        }
        // Should be either quarantined or evicted.
        let quarantined = scorer.quarantined_workers();
        let evicted = scorer.evicted_workers();
        assert!(
            quarantined.contains(&"w1") || evicted.contains(&"w1"),
            "w1 should be quarantined or evicted"
        );
    }

    #[test]
    fn test_status_display() {
        assert_eq!(WorkerHealthStatus::Healthy.to_string(), "Healthy");
        assert_eq!(WorkerHealthStatus::Quarantined.to_string(), "Quarantined");
        assert_eq!(WorkerHealthStatus::Evicted.to_string(), "Evicted");
    }

    #[test]
    fn test_outcome_sample_values() {
        assert!((JobOutcome::Success.sample_value() - 1.0).abs() < f64::EPSILON);
        assert!(
            JobOutcome::PermanentFailure.sample_value()
                < JobOutcome::TransientFailure.sample_value()
        );
        assert!(JobOutcome::TransientFailure.sample_value() < JobOutcome::Timeout.sample_value());
        assert!(JobOutcome::Timeout.sample_value() < JobOutcome::Success.sample_value());
    }

    #[test]
    fn test_worker_count() {
        let mut scorer = WorkerHealthScorer::default_policy();
        assert_eq!(scorer.worker_count(), 0);
        scorer.register_worker("w1");
        scorer.register_worker("w2");
        assert_eq!(scorer.worker_count(), 2);
    }
}
