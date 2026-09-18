// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job quota and throttling for oximedia-jobs.
//!
//! `QuotaEnforcer` gates job submissions using configurable limits on
//! concurrency, throughput, queue depth, and CPU usage.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// JobQuota
// ---------------------------------------------------------------------------

/// Configurable resource limits for job submission.
#[derive(Debug, Clone)]
pub struct JobQuota {
    /// Maximum number of jobs that may be active (running) simultaneously.
    pub max_concurrent: usize,
    /// Maximum number of job submissions allowed per 60-second sliding window.
    pub max_per_minute: u32,
    /// Maximum number of jobs allowed in the pending queue at any time.
    pub max_queued: usize,
    /// Maximum allowed CPU utilisation percentage (0.0–100.0).
    pub max_cpu_pct: f64,
}

impl Default for JobQuota {
    fn default() -> Self {
        Self {
            max_concurrent: 8,
            max_per_minute: 120,
            max_queued: 256,
            max_cpu_pct: 90.0,
        }
    }
}

// ---------------------------------------------------------------------------
// QuotaState
// ---------------------------------------------------------------------------

/// Runtime state tracked by `QuotaEnforcer`.
#[derive(Debug, Default)]
pub struct QuotaState {
    /// Number of currently active (running) jobs.
    pub active_jobs: usize,
    /// Number of jobs in the pending queue.
    pub queued_jobs: usize,
    /// Millisecond timestamps of recent job submissions (sliding window).
    pub recent_submissions: VecDeque<u64>,
}

// ---------------------------------------------------------------------------
// QuotaDecision
// ---------------------------------------------------------------------------

/// Outcome of a quota check.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum QuotaDecision {
    /// The submission is within all configured limits.
    Allow,
    /// The submission exceeds a rate or capacity limit; caller should retry.
    Throttle {
        /// Suggested delay before retrying, in milliseconds.
        retry_after_ms: u64,
    },
    /// The submission is unconditionally rejected (e.g., queue full).
    Reject {
        /// Human-readable rejection reason.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// QuotaEnforcer
// ---------------------------------------------------------------------------

const WINDOW_MS: u64 = 60_000; // 1-minute sliding window

/// Enforces `JobQuota` limits on job submissions.
pub struct QuotaEnforcer {
    /// The configured resource quota.
    pub quota: JobQuota,
    /// Current runtime state.
    pub state: QuotaState,
}

impl QuotaEnforcer {
    /// Create a new enforcer with the given quota.
    #[must_use]
    pub fn new(quota: JobQuota) -> Self {
        Self {
            quota,
            state: QuotaState::default(),
        }
    }

    /// Evaluate whether a new job submission is permitted.
    ///
    /// * `now_ms` – current time in milliseconds (monotonic or wall-clock).
    ///
    /// Call this *before* the job is submitted to the queue. If `Allow` is
    /// returned, the caller should then call `job_started` when the job
    /// actually begins execution, or adjust `state.queued_jobs` manually if
    /// the job enters a waiting queue first.
    pub fn check_submit(&mut self, now_ms: u64) -> QuotaDecision {
        // 1. Check queue capacity
        if self.state.queued_jobs >= self.quota.max_queued {
            return QuotaDecision::Reject {
                reason: format!(
                    "queue full: {} queued jobs (max {})",
                    self.state.queued_jobs, self.quota.max_queued
                ),
            };
        }

        // 2. Evict stale entries from the sliding window
        let window_start = now_ms.saturating_sub(WINDOW_MS);
        while self
            .state
            .recent_submissions
            .front()
            .map(|&t| t < window_start)
            .unwrap_or(false)
        {
            self.state.recent_submissions.pop_front();
        }

        // 3. Check per-minute rate
        let count_in_window = self.state.recent_submissions.len() as u32;
        if count_in_window >= self.quota.max_per_minute {
            // Estimate when the oldest submission will leave the window
            let oldest = *self.state.recent_submissions.front().unwrap_or(&now_ms);
            let retry_after_ms = (oldest + WINDOW_MS).saturating_sub(now_ms);
            return QuotaDecision::Throttle { retry_after_ms };
        }

        // 4. All checks passed – record the submission timestamp
        self.state.recent_submissions.push_back(now_ms);
        QuotaDecision::Allow
    }

    /// Notify the enforcer that a job has moved from "queued" to "active".
    pub fn job_started(&mut self) {
        self.state.active_jobs += 1;
        if self.state.queued_jobs > 0 {
            self.state.queued_jobs -= 1;
        }
    }

    /// Notify the enforcer that an active job has finished.
    pub fn job_finished(&mut self) {
        if self.state.active_jobs > 0 {
            self.state.active_jobs -= 1;
        }
    }

    /// Returns the current utilisation as a fraction of `max_concurrent` in
    /// the range `[0.0, 1.0]` (clamped; may exceed 1.0 temporarily under
    /// miscounting).
    #[must_use]
    pub fn current_utilization(&self) -> f64 {
        if self.quota.max_concurrent == 0 {
            return 0.0;
        }
        (self.state.active_jobs as f64 / self.quota.max_concurrent as f64).min(1.0)
    }

    /// Returns `true` when the concurrency limit has been reached.
    #[must_use]
    pub fn is_at_capacity(&self) -> bool {
        self.state.active_jobs >= self.quota.max_concurrent
    }

    /// Returns the number of submissions recorded in the current window.
    #[must_use]
    pub fn submissions_in_window(&self, now_ms: u64) -> u32 {
        let window_start = now_ms.saturating_sub(WINDOW_MS);
        self.state
            .recent_submissions
            .iter()
            .filter(|&&t| t >= window_start)
            .count() as u32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn enforcer_with(
        max_concurrent: usize,
        max_per_minute: u32,
        max_queued: usize,
    ) -> QuotaEnforcer {
        QuotaEnforcer::new(JobQuota {
            max_concurrent,
            max_per_minute,
            max_queued,
            max_cpu_pct: 90.0,
        })
    }

    #[test]
    fn test_allow_under_limits() {
        let mut e = enforcer_with(4, 100, 10);
        let decision = e.check_submit(1000);
        assert_eq!(decision, QuotaDecision::Allow);
    }

    #[test]
    fn test_reject_queue_full() {
        let mut e = enforcer_with(4, 100, 2);
        e.state.queued_jobs = 2;
        let decision = e.check_submit(1000);
        assert!(matches!(decision, QuotaDecision::Reject { .. }));
    }

    #[test]
    fn test_throttle_rate_exceeded() {
        let mut e = enforcer_with(4, 3, 100);
        let now = 10_000u64;
        // Fill the window
        e.check_submit(now);
        e.check_submit(now + 100);
        e.check_submit(now + 200);
        // Fourth attempt should be throttled
        let decision = e.check_submit(now + 300);
        assert!(matches!(decision, QuotaDecision::Throttle { .. }));
    }

    #[test]
    fn test_rate_resets_after_window() {
        let mut e = enforcer_with(4, 2, 100);
        let t0 = 0u64;
        e.check_submit(t0);
        e.check_submit(t0 + 100);
        // Both slots used – next submission in same window throttled
        assert!(matches!(
            e.check_submit(t0 + 200),
            QuotaDecision::Throttle { .. }
        ));
        // Jump past 60-second window – old entries evicted
        let t1 = t0 + WINDOW_MS + 1;
        let decision = e.check_submit(t1);
        assert_eq!(decision, QuotaDecision::Allow);
    }

    #[test]
    fn test_job_started_increments_active() {
        let mut e = enforcer_with(4, 100, 10);
        e.state.queued_jobs = 1;
        e.job_started();
        assert_eq!(e.state.active_jobs, 1);
        assert_eq!(e.state.queued_jobs, 0);
    }

    #[test]
    fn test_job_finished_decrements_active() {
        let mut e = enforcer_with(4, 100, 10);
        e.state.active_jobs = 3;
        e.job_finished();
        assert_eq!(e.state.active_jobs, 2);
    }

    #[test]
    fn test_job_finished_no_underflow() {
        let mut e = enforcer_with(4, 100, 10);
        e.state.active_jobs = 0;
        e.job_finished(); // should not panic or underflow
        assert_eq!(e.state.active_jobs, 0);
    }

    #[test]
    fn test_current_utilization_empty() {
        let e = enforcer_with(4, 100, 10);
        assert!((e.current_utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_current_utilization_full() {
        let mut e = enforcer_with(4, 100, 10);
        e.state.active_jobs = 4;
        assert!((e.current_utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_current_utilization_partial() {
        let mut e = enforcer_with(8, 100, 10);
        e.state.active_jobs = 4;
        assert!((e.current_utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_current_utilization_zero_max_concurrent() {
        let mut e = enforcer_with(0, 100, 10);
        e.state.active_jobs = 5;
        assert!((e.current_utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_at_capacity() {
        let mut e = enforcer_with(2, 100, 10);
        assert!(!e.is_at_capacity());
        e.state.active_jobs = 2;
        assert!(e.is_at_capacity());
    }

    #[test]
    fn test_submissions_in_window() {
        let mut e = enforcer_with(4, 100, 10);
        let now = 5_000u64;
        e.check_submit(now - 1_000); // within window
        e.check_submit(now - 500); // within window
        assert_eq!(e.submissions_in_window(now), 2);
    }

    #[test]
    fn test_default_quota() {
        let q = JobQuota::default();
        assert_eq!(q.max_concurrent, 8);
        assert_eq!(q.max_per_minute, 120);
        assert_eq!(q.max_queued, 256);
        assert!((q.max_cpu_pct - 90.0).abs() < f64::EPSILON);
    }
}
