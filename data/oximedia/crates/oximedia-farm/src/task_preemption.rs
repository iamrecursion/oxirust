//! Task preemption support for the render farm.
//!
//! Provides policies and management for preempting running tasks based on
//! timeouts, priority inversion, or resource contention.

#![allow(dead_code)]

/// Policy controlling how and when tasks may be preempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionPolicy {
    /// Tasks run to completion; no preemption is allowed.
    NonPreemptive,
    /// Preempt tasks that exceed a maximum wall-clock runtime.
    Timeout,
    /// Preempt lower-priority tasks when a higher-priority task arrives.
    Priority,
    /// Preempt tasks when their node is experiencing resource contention.
    ResourceContention,
}

impl PreemptionPolicy {
    /// Return `true` if this policy permits preempting a running task.
    #[must_use]
    pub fn allows_preemption(&self) -> bool {
        !matches!(self, Self::NonPreemptive)
    }

    /// A short human-readable description of the policy.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::NonPreemptive => "Non-preemptive: tasks run to completion",
            Self::Timeout => "Timeout: preempt tasks exceeding max runtime",
            Self::Priority => "Priority: preempt low-priority tasks for high-priority ones",
            Self::ResourceContention => {
                "Resource contention: preempt tasks when node is overloaded"
            }
        }
    }
}

/// A record of a single preemption event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreemptionEvent {
    /// Identifier of the preempted task.
    pub task_id: u64,
    /// Policy that triggered this preemption.
    pub reason: PreemptionPolicy,
    /// Epoch timestamp (milliseconds) at which preemption occurred.
    pub preempted_at_epoch: u64,
    /// Amount of work already completed, in milliseconds.
    pub elapsed_work_ms: u64,
}

impl PreemptionEvent {
    /// Create a new preemption event.
    #[must_use]
    pub fn new(
        task_id: u64,
        reason: PreemptionPolicy,
        preempted_at_epoch: u64,
        elapsed_work_ms: u64,
    ) -> Self {
        Self {
            task_id,
            reason,
            preempted_at_epoch,
            elapsed_work_ms,
        }
    }

    /// Return `true` if the preempted task is eligible to be resumed later.
    ///
    /// Tasks preempted due to `ResourceContention` are **not** eligible for
    /// resumption because the node state is unknown; all other preemption
    /// reasons allow the task to be re-queued.
    #[must_use]
    pub fn can_resume(&self) -> bool {
        self.reason != PreemptionPolicy::ResourceContention
    }
}

/// Configuration for the preemption manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreemptionConfig {
    /// Active preemption policy.
    pub policy: PreemptionPolicy,
    /// Maximum task runtime in milliseconds before a `Timeout` preemption fires.
    pub max_task_runtime_ms: u64,
    /// Minimum priority of an incoming task required to trigger a `Priority` preemption.
    pub priority_threshold: u32,
}

impl PreemptionConfig {
    /// Create the default configuration (non-preemptive).
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            policy: PreemptionPolicy::NonPreemptive,
            max_task_runtime_ms: u64::MAX,
            priority_threshold: 100,
        }
    }

    /// Create a timeout-based preemption configuration.
    #[must_use]
    pub fn timeout_policy(max_ms: u64) -> Self {
        Self {
            policy: PreemptionPolicy::Timeout,
            max_task_runtime_ms: max_ms,
            priority_threshold: 100,
        }
    }
}

impl Default for PreemptionConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Manages preemption decisions and records preemption history.
#[derive(Debug, Clone, Default)]
pub struct PreemptionManager {
    /// Active configuration.
    pub config: PreemptionConfig,
    /// History of preemption events.
    pub events: Vec<PreemptionEvent>,
}

impl PreemptionManager {
    /// Create a new preemption manager with the given configuration.
    #[must_use]
    pub fn new(config: PreemptionConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
        }
    }

    /// Decide whether the running task should be preempted.
    ///
    /// # Arguments
    ///
    /// * `started_at` – epoch timestamp (ms) when the task started.
    /// * `now` – current epoch timestamp (ms).
    /// * `priority` – priority value of the incoming competing task.
    #[must_use]
    pub fn should_preempt(&self, started_at: u64, now: u64, priority: u32) -> bool {
        match self.config.policy {
            PreemptionPolicy::NonPreemptive => false,
            PreemptionPolicy::Timeout => {
                let elapsed = now.saturating_sub(started_at);
                elapsed >= self.config.max_task_runtime_ms
            }
            PreemptionPolicy::Priority => priority >= self.config.priority_threshold,
            PreemptionPolicy::ResourceContention => true,
        }
    }

    /// Record a preemption event.
    pub fn record_preemption(&mut self, event: PreemptionEvent) {
        self.events.push(event);
    }

    /// Return the total number of preemption events recorded.
    #[must_use]
    pub fn total_preemptions(&self) -> usize {
        self.events.len()
    }

    /// Return the total elapsed work (ms) across all preempted tasks.
    #[must_use]
    pub fn preempted_ms_total(&self) -> u64 {
        self.events.iter().map(|e| e.elapsed_work_ms).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PreemptionPolicy tests ---

    #[test]
    fn test_non_preemptive_does_not_allow_preemption() {
        assert!(!PreemptionPolicy::NonPreemptive.allows_preemption());
    }

    #[test]
    fn test_timeout_allows_preemption() {
        assert!(PreemptionPolicy::Timeout.allows_preemption());
    }

    #[test]
    fn test_priority_allows_preemption() {
        assert!(PreemptionPolicy::Priority.allows_preemption());
    }

    #[test]
    fn test_resource_contention_allows_preemption() {
        assert!(PreemptionPolicy::ResourceContention.allows_preemption());
    }

    #[test]
    fn test_description_non_preemptive() {
        let desc = PreemptionPolicy::NonPreemptive.description();
        assert!(desc.contains("Non-preemptive"));
    }

    #[test]
    fn test_description_timeout() {
        let desc = PreemptionPolicy::Timeout.description();
        assert!(desc.contains("Timeout"));
    }

    // --- PreemptionEvent tests ---

    #[test]
    fn test_can_resume_timeout_preemption() {
        let event = PreemptionEvent::new(1, PreemptionPolicy::Timeout, 1000, 500);
        assert!(event.can_resume());
    }

    #[test]
    fn test_can_resume_priority_preemption() {
        let event = PreemptionEvent::new(2, PreemptionPolicy::Priority, 2000, 300);
        assert!(event.can_resume());
    }

    #[test]
    fn test_cannot_resume_resource_contention() {
        let event = PreemptionEvent::new(3, PreemptionPolicy::ResourceContention, 3000, 200);
        assert!(!event.can_resume());
    }

    #[test]
    fn test_can_resume_non_preemptive() {
        // In practice NonPreemptive won't generate events, but the struct allows it
        let event = PreemptionEvent::new(4, PreemptionPolicy::NonPreemptive, 4000, 100);
        assert!(event.can_resume());
    }

    // --- PreemptionConfig tests ---

    #[test]
    fn test_default_config_is_non_preemptive() {
        let cfg = PreemptionConfig::default();
        assert_eq!(cfg.policy, PreemptionPolicy::NonPreemptive);
    }

    #[test]
    fn test_timeout_policy_sets_policy() {
        let cfg = PreemptionConfig::timeout_policy(30_000);
        assert_eq!(cfg.policy, PreemptionPolicy::Timeout);
        assert_eq!(cfg.max_task_runtime_ms, 30_000);
    }

    // --- PreemptionManager tests ---

    #[test]
    fn test_should_not_preempt_non_preemptive() {
        let mgr = PreemptionManager::new(PreemptionConfig::default());
        assert!(!mgr.should_preempt(0, 999_999, 200));
    }

    #[test]
    fn test_should_preempt_on_timeout() {
        let cfg = PreemptionConfig::timeout_policy(5_000);
        let mgr = PreemptionManager::new(cfg);
        // elapsed = 6000 >= 5000
        assert!(mgr.should_preempt(0, 6_000, 0));
    }

    #[test]
    fn test_should_not_preempt_before_timeout() {
        let cfg = PreemptionConfig::timeout_policy(5_000);
        let mgr = PreemptionManager::new(cfg);
        assert!(!mgr.should_preempt(0, 4_000, 0));
    }

    #[test]
    fn test_should_preempt_on_priority() {
        let cfg = PreemptionConfig {
            policy: PreemptionPolicy::Priority,
            max_task_runtime_ms: u64::MAX,
            priority_threshold: 10,
        };
        let mgr = PreemptionManager::new(cfg);
        assert!(mgr.should_preempt(0, 0, 10));
    }

    #[test]
    fn test_should_not_preempt_below_priority_threshold() {
        let cfg = PreemptionConfig {
            policy: PreemptionPolicy::Priority,
            max_task_runtime_ms: u64::MAX,
            priority_threshold: 10,
        };
        let mgr = PreemptionManager::new(cfg);
        assert!(!mgr.should_preempt(0, 0, 9));
    }

    #[test]
    fn test_record_and_total_preemptions() {
        let mut mgr = PreemptionManager::new(PreemptionConfig::timeout_policy(1_000));
        mgr.record_preemption(PreemptionEvent::new(
            1,
            PreemptionPolicy::Timeout,
            1000,
            800,
        ));
        mgr.record_preemption(PreemptionEvent::new(
            2,
            PreemptionPolicy::Timeout,
            2000,
            600,
        ));
        assert_eq!(mgr.total_preemptions(), 2);
    }

    #[test]
    fn test_preempted_ms_total() {
        let mut mgr = PreemptionManager::new(PreemptionConfig::default());
        mgr.record_preemption(PreemptionEvent::new(
            1,
            PreemptionPolicy::Priority,
            1000,
            400,
        ));
        mgr.record_preemption(PreemptionEvent::new(
            2,
            PreemptionPolicy::Priority,
            2000,
            600,
        ));
        assert_eq!(mgr.preempted_ms_total(), 1_000);
    }
}
