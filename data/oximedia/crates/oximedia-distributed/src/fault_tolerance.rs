//! Distributed fault tolerance.
//!
//! Provides circuit-breaker logic and per-node failure tracking.

use std::collections::HashMap;

/// Describes why a node failed.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureType {
    NodeCrash,
    NetworkPartition,
    SlowNode,
    MemoryExhausted,
    DiskFull,
}

/// A recorded failure event for a node.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct NodeFailure {
    pub node_id: u64,
    pub failure_type: FailureType,
    pub detected_at: u64,
    pub recovered_at: Option<u64>,
}

/// State of a circuit breaker.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation; calls are allowed.
    Closed,
    /// Too many failures; calls are blocked.
    Open,
    /// After reset timeout; one probe call is allowed.
    HalfOpen,
}

/// Per-node circuit breaker.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub failure_count: u32,
    pub threshold: u32,
    pub state: CircuitState,
    pub last_failure: u64,
    pub reset_timeout_ms: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker in the `Closed` state.
    #[must_use]
    pub fn new(threshold: u32, reset_timeout_ms: u64) -> Self {
        Self {
            failure_count: 0,
            threshold,
            state: CircuitState::Closed,
            last_failure: 0,
            reset_timeout_ms,
        }
    }

    /// Record a successful call; resets the breaker to `Closed`.
    pub fn call_succeeded(&mut self, _now: u64) {
        self.failure_count = 0;
        self.state = CircuitState::Closed;
    }

    /// Record a failed call; may trip the breaker to `Open`.
    pub fn call_failed(&mut self, now: u64) {
        self.failure_count += 1;
        self.last_failure = now;
        if self.failure_count >= self.threshold {
            self.state = CircuitState::Open;
        }
    }

    /// Returns `true` if a call is permitted right now.
    ///
    /// - `Closed` -> always allowed.
    /// - `Open`   -> allowed only after `reset_timeout_ms` has elapsed (and the
    ///               state transitions to `HalfOpen`).
    /// - `HalfOpen` -> allowed (one probe).
    pub fn is_allowed(&mut self, now: u64) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                if now.saturating_sub(self.last_failure) >= self.reset_timeout_ms {
                    self.state = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            }
        }
    }
}

/// Tracks failures across all nodes and manages their circuit breakers.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct FaultManager {
    failures: Vec<NodeFailure>,
    breakers: HashMap<u64, CircuitBreaker>,
}

impl FaultManager {
    /// Create an empty fault manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            failures: Vec::new(),
            breakers: HashMap::new(),
        }
    }

    /// Report a new failure for a node.
    pub fn report_failure(&mut self, node_id: u64, ftype: FailureType, now: u64) {
        self.failures.push(NodeFailure {
            node_id,
            failure_type: ftype,
            detected_at: now,
            recovered_at: None,
        });
        // Trip the corresponding circuit breaker.
        let breaker = self
            .breakers
            .entry(node_id)
            .or_insert_with(|| CircuitBreaker::new(3, 5000));
        breaker.call_failed(now);
    }

    /// Mark a node as recovered at `now`.
    pub fn node_recovered(&mut self, node_id: u64, now: u64) {
        for f in &mut self.failures {
            if f.node_id == node_id && f.recovered_at.is_none() {
                f.recovered_at = Some(now);
            }
        }
        if let Some(b) = self.breakers.get_mut(&node_id) {
            b.call_succeeded(now);
        }
    }

    /// Return all failures that have not yet been recovered.
    #[must_use]
    pub fn active_failures(&self) -> Vec<&NodeFailure> {
        self.failures
            .iter()
            .filter(|f| f.recovered_at.is_none())
            .collect()
    }

    /// Returns `true` if the node's circuit breaker permits a call at `now`.
    /// Nodes without a registered breaker are considered healthy.
    pub fn is_node_healthy(&mut self, node_id: u64, now: u64) -> bool {
        match self.breakers.get_mut(&node_id) {
            Some(b) => b.is_allowed(now),
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new(3, 1000);
        assert_eq!(cb.state, CircuitState::Closed);
        assert_eq!(cb.failure_count, 0);
    }

    #[test]
    fn test_circuit_breaker_trips_on_threshold() {
        let mut cb = CircuitBreaker::new(3, 1000);
        cb.call_failed(1);
        cb.call_failed(2);
        assert_eq!(cb.state, CircuitState::Closed);
        cb.call_failed(3);
        assert_eq!(cb.state, CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_blocked_when_open() {
        let mut cb = CircuitBreaker::new(1, 5000);
        cb.call_failed(100);
        assert!(!cb.is_allowed(200)); // within timeout
    }

    #[test]
    fn test_circuit_breaker_half_open_after_timeout() {
        let mut cb = CircuitBreaker::new(1, 1000);
        cb.call_failed(0);
        assert!(cb.is_allowed(1001));
        assert_eq!(cb.state, CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(1, 1000);
        cb.call_failed(0);
        cb.call_succeeded(2000);
        assert_eq!(cb.state, CircuitState::Closed);
        assert_eq!(cb.failure_count, 0);
    }

    #[test]
    fn test_fault_manager_report_failure() {
        let mut fm = FaultManager::new();
        fm.report_failure(1, FailureType::NodeCrash, 100);
        assert_eq!(fm.active_failures().len(), 1);
    }

    #[test]
    fn test_fault_manager_node_recovered() {
        let mut fm = FaultManager::new();
        fm.report_failure(1, FailureType::NetworkPartition, 100);
        fm.node_recovered(1, 200);
        assert!(fm.active_failures().is_empty());
    }

    #[test]
    fn test_fault_manager_active_failures_partial() {
        let mut fm = FaultManager::new();
        fm.report_failure(1, FailureType::SlowNode, 100);
        fm.report_failure(2, FailureType::DiskFull, 110);
        fm.node_recovered(1, 150);
        let active = fm.active_failures();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].node_id, 2);
    }

    #[test]
    fn test_fault_manager_is_node_healthy_unknown() {
        let mut fm = FaultManager::new();
        assert!(fm.is_node_healthy(99, 1000));
    }

    #[test]
    fn test_fault_manager_is_node_unhealthy_after_failures() {
        let mut fm = FaultManager::new();
        // Default threshold is 3
        fm.report_failure(5, FailureType::MemoryExhausted, 10);
        fm.report_failure(5, FailureType::MemoryExhausted, 20);
        fm.report_failure(5, FailureType::MemoryExhausted, 30);
        assert!(!fm.is_node_healthy(5, 35));
    }

    #[test]
    fn test_fault_manager_recovery_re_enables() {
        let mut fm = FaultManager::new();
        fm.report_failure(7, FailureType::NodeCrash, 10);
        fm.report_failure(7, FailureType::NodeCrash, 20);
        fm.report_failure(7, FailureType::NodeCrash, 30);
        fm.node_recovered(7, 200);
        assert!(fm.is_node_healthy(7, 200));
    }

    #[test]
    fn test_all_failure_types() {
        let mut fm = FaultManager::new();
        let types = [
            FailureType::NodeCrash,
            FailureType::NetworkPartition,
            FailureType::SlowNode,
            FailureType::MemoryExhausted,
            FailureType::DiskFull,
        ];
        for (i, ft) in types.into_iter().enumerate() {
            fm.report_failure(i as u64, ft, i as u64 * 10);
        }
        assert_eq!(fm.active_failures().len(), 5);
    }

    #[test]
    fn test_multiple_failures_same_node() {
        let mut fm = FaultManager::new();
        fm.report_failure(1, FailureType::SlowNode, 1);
        fm.report_failure(1, FailureType::SlowNode, 2);
        // Both recorded; both active
        assert_eq!(fm.active_failures().len(), 2);
    }
}
