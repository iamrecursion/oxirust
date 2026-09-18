//! Per-job timeout tracking and enforcement.
//!
//! [`TimeoutEnforcer`] maintains a live registry of in-flight jobs and their
//! deadlines.  Call [`TimeoutEnforcer::check_all`] periodically (e.g., from a
//! monitoring task) to receive a stream of [`TimeoutEvent`]s without consuming
//! the registry.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ─── JobTimeout ───────────────────────────────────────────────────────────────

/// Tracking record for a single in-flight job.
#[derive(Debug)]
pub struct JobTimeout {
    /// Job identifier.
    pub job_id: String,
    /// Hard deadline measured from `started_at`.
    pub timeout: Duration,
    /// Monotonic clock reading when the job was registered.
    pub started_at: Instant,
    /// Optional early-warning threshold.  When `elapsed >= warn_at` a
    /// [`TimeoutEvent::Warning`] is emitted.
    pub warn_at: Option<Duration>,
}

impl JobTimeout {
    /// Elapsed time since the job was registered.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Remaining time before the hard deadline, saturating at zero.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.timeout.saturating_sub(self.elapsed())
    }

    /// `true` when the hard deadline has passed.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.elapsed() >= self.timeout
    }

    /// `true` when the warning threshold has been crossed (and one exists).
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.warn_at.map_or(false, |w| self.elapsed() >= w)
    }
}

// ─── TimeoutEvent ─────────────────────────────────────────────────────────────

/// An event produced by [`TimeoutEnforcer::check_all`].
#[derive(Debug, Clone)]
pub enum TimeoutEvent {
    /// The job has passed its optional early-warning threshold.
    Warning {
        /// Job identifier.
        job_id: String,
        /// How long the job has been running.
        elapsed: Duration,
        /// Time left until the hard deadline.
        remaining: Duration,
    },
    /// The job has passed its hard deadline.
    Expired {
        /// Job identifier.
        job_id: String,
        /// How long the job has been running (≥ timeout).
        elapsed: Duration,
    },
}

// ─── TimeoutEnforcer ──────────────────────────────────────────────────────────

/// Registry of in-flight job timeouts.
///
/// All methods are `&self` or `&mut self`; there is no async or background
/// thread involved.  The caller is expected to drive the polling loop.
#[derive(Debug, Default)]
pub struct TimeoutEnforcer {
    jobs: HashMap<String, JobTimeout>,
}

impl TimeoutEnforcer {
    /// Create an empty enforcer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a job with a hard `timeout` and an optional early-warning
    /// `warn_at` threshold.
    ///
    /// Registering the same `job_id` twice replaces the previous entry.
    pub fn register(&mut self, job_id: String, timeout: Duration, warn_at: Option<Duration>) {
        self.jobs.insert(
            job_id.clone(),
            JobTimeout {
                job_id,
                timeout,
                started_at: Instant::now(),
                warn_at,
            },
        );
    }

    /// Remove a job from the registry (e.g., when it completes normally).
    pub fn deregister(&mut self, job_id: &str) {
        self.jobs.remove(job_id);
    }

    /// Inspect all registered jobs and return any [`TimeoutEvent`]s that apply
    /// **right now** without removing jobs from the registry.
    ///
    /// A job may generate at most two events simultaneously: a `Warning` *and*
    /// an `Expired` if both thresholds are passed.
    #[must_use]
    pub fn check_all(&self) -> Vec<TimeoutEvent> {
        let mut events = Vec::new();
        for jt in self.jobs.values() {
            let elapsed = jt.elapsed();
            // Hard expiry check first.
            if jt.is_expired() {
                events.push(TimeoutEvent::Expired {
                    job_id: jt.job_id.clone(),
                    elapsed,
                });
            } else if jt.is_warning() {
                // Only emit Warning when not yet expired.
                events.push(TimeoutEvent::Warning {
                    job_id: jt.job_id.clone(),
                    elapsed,
                    remaining: jt.remaining(),
                });
            }
        }
        events
    }

    /// Return the IDs of all jobs that have exceeded their hard deadline.
    #[must_use]
    pub fn expired_jobs(&self) -> Vec<String> {
        self.jobs
            .values()
            .filter(|jt| jt.is_expired())
            .map(|jt| jt.job_id.clone())
            .collect()
    }

    /// Return the IDs of all jobs that have crossed their warning threshold
    /// but have **not** yet expired.
    #[must_use]
    pub fn warning_jobs(&self) -> Vec<String> {
        self.jobs
            .values()
            .filter(|jt| jt.is_warning() && !jt.is_expired())
            .map(|jt| jt.job_id.clone())
            .collect()
    }

    /// Return the remaining time for a specific job, or `None` if the job is
    /// not registered.
    #[must_use]
    pub fn time_remaining(&self, job_id: &str) -> Option<Duration> {
        self.jobs.get(job_id).map(|jt| jt.remaining())
    }

    /// Return `true` if the job is registered **and** has expired.
    #[must_use]
    pub fn is_expired(&self, job_id: &str) -> bool {
        self.jobs.get(job_id).map_or(false, |jt| jt.is_expired())
    }

    /// Number of jobs currently in the registry.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.jobs.len()
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_enforcer_is_empty() {
        let enforcer = TimeoutEnforcer::new();
        assert_eq!(enforcer.active_count(), 0);
    }

    #[test]
    fn test_register_increases_count() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register("job-1".to_string(), Duration::from_secs(30), None);
        assert_eq!(enforcer.active_count(), 1);
    }

    #[test]
    fn test_deregister_removes_job() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register("job-1".to_string(), Duration::from_secs(10), None);
        enforcer.deregister("job-1");
        assert_eq!(enforcer.active_count(), 0);
    }

    #[test]
    fn test_time_remaining_returns_none_for_unknown_job() {
        let enforcer = TimeoutEnforcer::new();
        assert!(enforcer.time_remaining("ghost").is_none());
    }

    #[test]
    fn test_is_expired_returns_false_for_unknown_job() {
        let enforcer = TimeoutEnforcer::new();
        assert!(!enforcer.is_expired("ghost"));
    }

    #[test]
    fn test_fresh_job_not_expired() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register("j".to_string(), Duration::from_secs(60), None);
        assert!(!enforcer.is_expired("j"));
    }

    #[test]
    fn test_expired_job_detected() {
        let mut enforcer = TimeoutEnforcer::new();
        // Register with a zero-duration timeout — immediately expired.
        enforcer.register("j".to_string(), Duration::ZERO, None);
        assert!(enforcer.is_expired("j"));
    }

    #[test]
    fn test_expired_jobs_list() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register("fast".to_string(), Duration::ZERO, None);
        enforcer.register("slow".to_string(), Duration::from_secs(3600), None);
        let expired = enforcer.expired_jobs();
        assert!(expired.contains(&"fast".to_string()));
        assert!(!expired.contains(&"slow".to_string()));
    }

    #[test]
    fn test_check_all_returns_expired_event() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register("j".to_string(), Duration::ZERO, None);
        let events = enforcer.check_all();
        assert!(events
            .iter()
            .any(|e| matches!(e, TimeoutEvent::Expired { job_id, .. } if job_id == "j")));
    }

    #[test]
    fn test_check_all_returns_warning_event() {
        let mut enforcer = TimeoutEnforcer::new();
        // Timeout is 1 hour but warn_at is zero → warning fires immediately.
        enforcer.register(
            "j".to_string(),
            Duration::from_secs(3600),
            Some(Duration::ZERO),
        );
        let events = enforcer.check_all();
        assert!(events
            .iter()
            .any(|e| matches!(e, TimeoutEvent::Warning { job_id, .. } if job_id == "j")));
    }

    #[test]
    fn test_warning_jobs_list() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register(
            "warn-me".to_string(),
            Duration::from_secs(3600),
            Some(Duration::ZERO),
        );
        enforcer.register("fine".to_string(), Duration::from_secs(3600), None);
        let warns = enforcer.warning_jobs();
        assert!(warns.contains(&"warn-me".to_string()));
        assert!(!warns.contains(&"fine".to_string()));
    }

    #[test]
    fn test_re_register_replaces_entry() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register("j".to_string(), Duration::ZERO, None); // expires immediately
                                                                  // Re-register with a long timeout — should no longer be expired.
        enforcer.register("j".to_string(), Duration::from_secs(3600), None);
        assert!(!enforcer.is_expired("j"));
        assert_eq!(enforcer.active_count(), 1);
    }

    #[test]
    fn test_check_all_does_not_remove_entries() {
        let mut enforcer = TimeoutEnforcer::new();
        enforcer.register("j".to_string(), Duration::ZERO, None);
        let _ = enforcer.check_all();
        assert_eq!(enforcer.active_count(), 1); // still there
    }
}
