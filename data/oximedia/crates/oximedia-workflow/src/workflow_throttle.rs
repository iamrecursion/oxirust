//! Workflow throttling and rate-limiting for `oximedia-workflow`.
//!
//! [`WorkflowThrottler`] controls how many workflows or tasks may execute
//! concurrently, applying a [`ThrottlePolicy`] and tracking current load via
//! [`ThrottleState`].

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Throttle policy
// ---------------------------------------------------------------------------

/// Strategy used to limit concurrent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThrottlePolicy {
    /// No throttling; allow unlimited concurrency.
    Unlimited,
    /// Fixed maximum number of concurrent executions.
    FixedMax,
    /// Token-bucket rate limiter (tokens refill over time).
    TokenBucket,
    /// Adaptive: adjusts concurrency based on system load.
    Adaptive,
}

impl ThrottlePolicy {
    /// Returns a short label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unlimited => "Unlimited",
            Self::FixedMax => "Fixed Max",
            Self::TokenBucket => "Token Bucket",
            Self::Adaptive => "Adaptive",
        }
    }

    /// Returns all variants.
    #[must_use]
    pub const fn all() -> &'static [ThrottlePolicy] {
        &[
            Self::Unlimited,
            Self::FixedMax,
            Self::TokenBucket,
            Self::Adaptive,
        ]
    }
}

impl std::fmt::Display for ThrottlePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Throttle state
// ---------------------------------------------------------------------------

/// Current load snapshot used by the throttler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ThrottleState {
    /// Number of currently running executions.
    pub active: usize,
    /// Number of executions waiting in the queue.
    pub queued: usize,
    /// Total number of executions admitted since the throttler was created.
    pub total_admitted: u64,
    /// Total number of executions rejected / delayed.
    pub total_rejected: u64,
}

impl ThrottleState {
    /// Returns utilisation as a fraction `[0.0, 1.0]` relative to a capacity.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilisation(&self, capacity: usize) -> f64 {
        if capacity == 0 {
            return 0.0;
        }
        self.active as f64 / capacity as f64
    }

    /// Returns `true` if there is room for at least one more execution
    /// given the specified capacity.
    #[must_use]
    pub fn has_capacity(&self, capacity: usize) -> bool {
        self.active < capacity
    }
}

// ---------------------------------------------------------------------------
// Workflow throttler
// ---------------------------------------------------------------------------

/// Controls concurrent workflow / task execution according to a policy.
#[derive(Debug, Clone)]
pub struct WorkflowThrottler {
    policy: ThrottlePolicy,
    max_concurrent: usize,
    state: ThrottleState,
    wait_queue: VecDeque<String>,
}

impl Default for WorkflowThrottler {
    fn default() -> Self {
        Self {
            policy: ThrottlePolicy::Unlimited,
            max_concurrent: usize::MAX,
            state: ThrottleState::default(),
            wait_queue: VecDeque::new(),
        }
    }
}

impl WorkflowThrottler {
    /// Creates a throttler with a fixed maximum concurrency.
    #[must_use]
    pub fn fixed(max_concurrent: usize) -> Self {
        Self {
            policy: ThrottlePolicy::FixedMax,
            max_concurrent,
            state: ThrottleState::default(),
            wait_queue: VecDeque::new(),
        }
    }

    /// Creates an unlimited throttler.
    #[must_use]
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Returns the current policy.
    #[must_use]
    pub fn policy(&self) -> ThrottlePolicy {
        self.policy
    }

    /// Returns the maximum concurrency limit.
    #[must_use]
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Returns a snapshot of the current state.
    #[must_use]
    pub fn state(&self) -> &ThrottleState {
        &self.state
    }

    /// Attempts to admit a workflow for execution.
    ///
    /// Returns `true` if admitted, `false` if queued.
    pub fn try_admit(&mut self, workflow_id: impl Into<String>) -> bool {
        let wf_id = workflow_id.into();
        if self.state.active < self.max_concurrent {
            self.state.active += 1;
            self.state.total_admitted += 1;
            true
        } else {
            self.wait_queue.push_back(wf_id);
            self.state.queued = self.wait_queue.len();
            self.state.total_rejected += 1;
            false
        }
    }

    /// Signals that a workflow has finished, releasing a concurrency slot.
    ///
    /// Returns the next queued workflow ID to admit, if any.
    pub fn release(&mut self) -> Option<String> {
        if self.state.active > 0 {
            self.state.active -= 1;
        }
        if let Some(next) = self.wait_queue.pop_front() {
            self.state.active += 1;
            self.state.total_admitted += 1;
            self.state.queued = self.wait_queue.len();
            Some(next)
        } else {
            None
        }
    }

    /// Returns the number of workflows currently waiting.
    #[must_use]
    pub fn queued_count(&self) -> usize {
        self.wait_queue.len()
    }

    /// Returns the current utilisation as a fraction.
    #[must_use]
    pub fn utilisation(&self) -> f64 {
        self.state.utilisation(self.max_concurrent)
    }

    /// Resets the throttler to its initial state.
    pub fn reset(&mut self) {
        self.state = ThrottleState::default();
        self.wait_queue.clear();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ThrottlePolicy -----------------------------------------------------

    #[test]
    fn test_policy_label() {
        assert_eq!(ThrottlePolicy::Unlimited.label(), "Unlimited");
        assert_eq!(ThrottlePolicy::FixedMax.label(), "Fixed Max");
    }

    #[test]
    fn test_policy_display() {
        assert_eq!(format!("{}", ThrottlePolicy::TokenBucket), "Token Bucket");
    }

    #[test]
    fn test_policy_all() {
        assert_eq!(ThrottlePolicy::all().len(), 4);
    }

    // -- ThrottleState ------------------------------------------------------

    #[test]
    fn test_state_default() {
        let s = ThrottleState::default();
        assert_eq!(s.active, 0);
        assert_eq!(s.queued, 0);
    }

    #[test]
    fn test_state_utilisation() {
        let s = ThrottleState {
            active: 3,
            ..Default::default()
        };
        assert!((s.utilisation(10) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_state_utilisation_zero_capacity() {
        let s = ThrottleState::default();
        assert_eq!(s.utilisation(0), 0.0);
    }

    #[test]
    fn test_state_has_capacity() {
        let s = ThrottleState {
            active: 5,
            ..Default::default()
        };
        assert!(s.has_capacity(10));
        assert!(!s.has_capacity(5));
    }

    // -- WorkflowThrottler --------------------------------------------------

    #[test]
    fn test_throttler_unlimited() {
        let mut t = WorkflowThrottler::unlimited();
        assert!(t.try_admit("wf-1"));
        assert!(t.try_admit("wf-2"));
        assert_eq!(t.state().active, 2);
    }

    #[test]
    fn test_throttler_fixed_admit() {
        let mut t = WorkflowThrottler::fixed(2);
        assert!(t.try_admit("wf-1"));
        assert!(t.try_admit("wf-2"));
        // Third should be queued
        assert!(!t.try_admit("wf-3"));
        assert_eq!(t.queued_count(), 1);
    }

    #[test]
    fn test_throttler_release_promotes_queued() {
        let mut t = WorkflowThrottler::fixed(1);
        assert!(t.try_admit("wf-1"));
        assert!(!t.try_admit("wf-2"));

        let next = t.release();
        assert_eq!(next.as_deref(), Some("wf-2"));
        assert_eq!(t.state().active, 1);
        assert_eq!(t.queued_count(), 0);
    }

    #[test]
    fn test_throttler_release_empty_queue() {
        let mut t = WorkflowThrottler::fixed(2);
        assert!(t.try_admit("wf-1"));
        let next = t.release();
        assert!(next.is_none());
        assert_eq!(t.state().active, 0);
    }

    #[test]
    fn test_throttler_utilisation() {
        let mut t = WorkflowThrottler::fixed(4);
        t.try_admit("wf-1");
        t.try_admit("wf-2");
        assert!((t.utilisation() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_throttler_reset() {
        let mut t = WorkflowThrottler::fixed(2);
        t.try_admit("wf-1");
        t.try_admit("wf-2");
        t.try_admit("wf-3");
        t.reset();
        assert_eq!(t.state().active, 0);
        assert_eq!(t.queued_count(), 0);
    }

    #[test]
    fn test_throttler_total_counters() {
        let mut t = WorkflowThrottler::fixed(1);
        t.try_admit("wf-1");
        t.try_admit("wf-2"); // rejected/queued
        assert_eq!(t.state().total_admitted, 1);
        assert_eq!(t.state().total_rejected, 1);
        t.release(); // promotes wf-2
        assert_eq!(t.state().total_admitted, 2);
    }
}
