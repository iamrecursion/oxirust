#![allow(dead_code)]
//! Approval gate system for workflow steps.
//!
//! Provides configurable approval gates that can pause workflow execution
//! until human or automated approval is received. Supports multi-approver
//! policies, escalation rules, and time-based auto-approval.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Unique identifier for an approval gate instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GateId(u64);

impl GateId {
    /// Create a new gate identifier from a raw value.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw numeric identifier.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Policy that governs how many approvals are required.
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalPolicy {
    /// Any single approver is sufficient.
    Any,
    /// All listed approvers must approve.
    All,
    /// At least `n` out of the total approvers must approve.
    Quorum(usize),
    /// A specific named role must approve.
    Role(String),
}

/// The current state of an approval gate.
#[derive(Debug, Clone, PartialEq)]
pub enum GateState {
    /// Gate is waiting for approval.
    Pending,
    /// Gate has been approved.
    Approved,
    /// Gate has been rejected.
    Rejected,
    /// Gate was auto-approved after timeout.
    AutoApproved,
    /// Gate has timed out without response.
    TimedOut,
    /// Gate was escalated to a higher authority.
    Escalated,
}

/// A single approval decision from an approver.
#[derive(Debug, Clone)]
pub struct ApprovalDecision {
    /// Who made the decision.
    pub approver: String,
    /// Whether it was approved or rejected.
    pub approved: bool,
    /// Optional comment or reason.
    pub comment: Option<String>,
    /// When the decision was made.
    pub decided_at: Instant,
}

/// Escalation configuration for when approval is not received in time.
#[derive(Debug, Clone)]
pub struct EscalationRule {
    /// How long to wait before escalating.
    pub after: Duration,
    /// Who to escalate to.
    pub escalate_to: String,
    /// Optional message for the escalation.
    pub message: Option<String>,
}

impl EscalationRule {
    /// Create a new escalation rule.
    pub fn new(after: Duration, escalate_to: impl Into<String>) -> Self {
        Self {
            after,
            escalate_to: escalate_to.into(),
            message: None,
        }
    }

    /// Set the escalation message.
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }
}

/// Configuration for an approval gate.
#[derive(Debug, Clone)]
pub struct ApprovalGateConfig {
    /// Human-readable name for this gate.
    pub name: String,
    /// Description of what is being approved.
    pub description: Option<String>,
    /// Who can approve this gate.
    pub approvers: Vec<String>,
    /// Policy governing how many approvals are needed.
    pub policy: ApprovalPolicy,
    /// If set, the gate auto-approves after this duration.
    pub auto_approve_after: Option<Duration>,
    /// If set, the gate times out after this duration.
    pub timeout: Option<Duration>,
    /// Escalation rules (applied in order).
    pub escalation_rules: Vec<EscalationRule>,
    /// Metadata attached to this gate.
    pub metadata: HashMap<String, String>,
}

impl ApprovalGateConfig {
    /// Create a new approval gate configuration.
    pub fn new(name: impl Into<String>, approvers: Vec<String>, policy: ApprovalPolicy) -> Self {
        Self {
            name: name.into(),
            description: None,
            approvers,
            policy,
            auto_approve_after: None,
            timeout: None,
            escalation_rules: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the gate description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set auto-approve timeout.
    #[must_use]
    pub fn with_auto_approve(mut self, after: Duration) -> Self {
        self.auto_approve_after = Some(after);
        self
    }

    /// Set the hard timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add an escalation rule.
    #[must_use]
    pub fn add_escalation(mut self, rule: EscalationRule) -> Self {
        self.escalation_rules.push(rule);
        self
    }

    /// Add metadata to the gate.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A live approval gate instance tracking approvals and state.
#[derive(Debug)]
pub struct ApprovalGate {
    /// Unique identifier for this gate.
    pub id: GateId,
    /// Configuration for this gate.
    pub config: ApprovalGateConfig,
    /// Current state of the gate.
    pub state: GateState,
    /// Collected approval decisions.
    pub decisions: Vec<ApprovalDecision>,
    /// When the gate was created / opened.
    pub opened_at: Instant,
    /// When the gate was closed (approved/rejected/timed-out).
    pub closed_at: Option<Instant>,
}

impl ApprovalGate {
    /// Create a new approval gate from configuration.
    #[must_use]
    pub fn new(id: GateId, config: ApprovalGateConfig) -> Self {
        Self {
            id,
            config,
            state: GateState::Pending,
            decisions: Vec::new(),
            opened_at: Instant::now(),
            closed_at: None,
        }
    }

    /// Submit an approval decision.
    pub fn submit_decision(&mut self, decision: ApprovalDecision) {
        if self.state != GateState::Pending {
            return;
        }
        self.decisions.push(decision);
        self.evaluate();
    }

    /// Check whether the gate should auto-approve or time out.
    pub fn check_timeouts(&mut self) {
        if self.state != GateState::Pending {
            return;
        }
        let elapsed = self.opened_at.elapsed();

        if let Some(auto_dur) = self.config.auto_approve_after {
            if elapsed >= auto_dur {
                self.state = GateState::AutoApproved;
                self.closed_at = Some(Instant::now());
                return;
            }
        }

        if let Some(timeout) = self.config.timeout {
            if elapsed >= timeout {
                self.state = GateState::TimedOut;
                self.closed_at = Some(Instant::now());
                return;
            }
        }

        // Check escalation rules
        for rule in &self.config.escalation_rules {
            if elapsed >= rule.after && self.state == GateState::Pending {
                self.state = GateState::Escalated;
                return;
            }
        }
    }

    /// Return whether the gate is still pending.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.state == GateState::Pending
    }

    /// Return whether the gate has been approved (including auto-approved).
    #[must_use]
    pub fn is_approved(&self) -> bool {
        matches!(self.state, GateState::Approved | GateState::AutoApproved)
    }

    /// Return whether the gate has been rejected.
    #[must_use]
    pub fn is_rejected(&self) -> bool {
        self.state == GateState::Rejected
    }

    /// Count how many positive approvals have been received.
    #[must_use]
    pub fn approval_count(&self) -> usize {
        self.decisions.iter().filter(|d| d.approved).count()
    }

    /// Count how many rejections have been received.
    #[must_use]
    pub fn rejection_count(&self) -> usize {
        self.decisions.iter().filter(|d| !d.approved).count()
    }

    /// Get the elapsed time since the gate was opened.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.opened_at.elapsed()
    }

    /// Evaluate decisions against the policy and update state.
    fn evaluate(&mut self) {
        let approvals = self.approval_count();
        let rejections = self.rejection_count();
        let total = self.config.approvers.len();

        match &self.config.policy {
            ApprovalPolicy::Any => {
                if approvals >= 1 {
                    self.state = GateState::Approved;
                    self.closed_at = Some(Instant::now());
                } else if rejections == total {
                    self.state = GateState::Rejected;
                    self.closed_at = Some(Instant::now());
                }
            }
            ApprovalPolicy::All => {
                if approvals == total {
                    self.state = GateState::Approved;
                    self.closed_at = Some(Instant::now());
                } else if rejections >= 1 {
                    self.state = GateState::Rejected;
                    self.closed_at = Some(Instant::now());
                }
            }
            ApprovalPolicy::Quorum(n) => {
                if approvals >= *n {
                    self.state = GateState::Approved;
                    self.closed_at = Some(Instant::now());
                } else if rejections > total.saturating_sub(*n) {
                    self.state = GateState::Rejected;
                    self.closed_at = Some(Instant::now());
                }
            }
            ApprovalPolicy::Role(role) => {
                // Check if any approver with the matching role has approved
                for decision in &self.decisions {
                    if decision.approver == *role {
                        if decision.approved {
                            self.state = GateState::Approved;
                        } else {
                            self.state = GateState::Rejected;
                        }
                        self.closed_at = Some(Instant::now());
                        return;
                    }
                }
            }
        }
    }
}

/// Registry that manages multiple approval gates.
#[derive(Debug)]
pub struct ApprovalGateRegistry {
    /// All registered gates keyed by ID.
    gates: HashMap<GateId, ApprovalGate>,
    /// Counter for generating gate IDs.
    next_id: u64,
}

impl Default for ApprovalGateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ApprovalGateRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gates: HashMap::new(),
            next_id: 1,
        }
    }

    /// Open a new approval gate and return its ID.
    pub fn open_gate(&mut self, config: ApprovalGateConfig) -> GateId {
        let id = GateId::new(self.next_id);
        self.next_id += 1;
        let gate = ApprovalGate::new(id, config);
        self.gates.insert(id, gate);
        id
    }

    /// Get a reference to a gate by ID.
    #[must_use]
    pub fn get_gate(&self, id: GateId) -> Option<&ApprovalGate> {
        self.gates.get(&id)
    }

    /// Get a mutable reference to a gate by ID.
    pub fn get_gate_mut(&mut self, id: GateId) -> Option<&mut ApprovalGate> {
        self.gates.get_mut(&id)
    }

    /// Submit a decision to a specific gate.
    pub fn submit_decision(&mut self, gate_id: GateId, decision: ApprovalDecision) -> bool {
        if let Some(gate) = self.gates.get_mut(&gate_id) {
            gate.submit_decision(decision);
            true
        } else {
            false
        }
    }

    /// Check timeouts on all pending gates.
    pub fn check_all_timeouts(&mut self) {
        for gate in self.gates.values_mut() {
            gate.check_timeouts();
        }
    }

    /// List all pending gate IDs.
    #[must_use]
    pub fn pending_gates(&self) -> Vec<GateId> {
        self.gates
            .iter()
            .filter(|(_, g)| g.is_pending())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Return the total number of gates in the registry.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Remove a closed gate from the registry.
    pub fn remove_gate(&mut self, id: GateId) -> Option<ApprovalGate> {
        self.gates.remove(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(approvers: Vec<&str>, policy: ApprovalPolicy) -> ApprovalGateConfig {
        ApprovalGateConfig::new(
            "test-gate",
            approvers.into_iter().map(String::from).collect(),
            policy,
        )
    }

    fn approve(approver: &str) -> ApprovalDecision {
        ApprovalDecision {
            approver: approver.to_string(),
            approved: true,
            comment: None,
            decided_at: Instant::now(),
        }
    }

    fn reject(approver: &str) -> ApprovalDecision {
        ApprovalDecision {
            approver: approver.to_string(),
            approved: false,
            comment: Some("Not ready".to_string()),
            decided_at: Instant::now(),
        }
    }

    #[test]
    fn test_gate_id() {
        let id = GateId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_new_gate_is_pending() {
        let config = make_config(vec!["alice"], ApprovalPolicy::Any);
        let gate = ApprovalGate::new(GateId::new(1), config);
        assert!(gate.is_pending());
        assert!(!gate.is_approved());
        assert!(!gate.is_rejected());
    }

    #[test]
    fn test_any_policy_single_approval() {
        let config = make_config(vec!["alice", "bob"], ApprovalPolicy::Any);
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(approve("alice"));
        assert!(gate.is_approved());
        assert_eq!(gate.approval_count(), 1);
    }

    #[test]
    fn test_all_policy_requires_all() {
        let config = make_config(vec!["alice", "bob"], ApprovalPolicy::All);
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(approve("alice"));
        assert!(gate.is_pending());
        gate.submit_decision(approve("bob"));
        assert!(gate.is_approved());
    }

    #[test]
    fn test_all_policy_rejects_on_single_rejection() {
        let config = make_config(vec!["alice", "bob"], ApprovalPolicy::All);
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(reject("alice"));
        assert!(gate.is_rejected());
    }

    #[test]
    fn test_quorum_policy() {
        let config = make_config(vec!["a", "b", "c"], ApprovalPolicy::Quorum(2));
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(approve("a"));
        assert!(gate.is_pending());
        gate.submit_decision(approve("b"));
        assert!(gate.is_approved());
    }

    #[test]
    fn test_quorum_policy_rejection() {
        let config = make_config(vec!["a", "b", "c"], ApprovalPolicy::Quorum(2));
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(reject("a"));
        gate.submit_decision(reject("b"));
        assert!(gate.is_rejected());
    }

    #[test]
    fn test_role_policy_approved() {
        let config = make_config(vec!["admin"], ApprovalPolicy::Role("admin".to_string()));
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(approve("admin"));
        assert!(gate.is_approved());
    }

    #[test]
    fn test_role_policy_rejected() {
        let config = make_config(vec!["admin"], ApprovalPolicy::Role("admin".to_string()));
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(reject("admin"));
        assert!(gate.is_rejected());
    }

    #[test]
    fn test_auto_approve_timeout() {
        let config = make_config(vec!["alice"], ApprovalPolicy::Any)
            .with_auto_approve(Duration::from_millis(0));
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.check_timeouts();
        assert_eq!(gate.state, GateState::AutoApproved);
        assert!(gate.is_approved());
    }

    #[test]
    fn test_hard_timeout() {
        let config =
            make_config(vec!["alice"], ApprovalPolicy::Any).with_timeout(Duration::from_millis(0));
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.check_timeouts();
        assert_eq!(gate.state, GateState::TimedOut);
    }

    #[test]
    fn test_registry_open_and_get() {
        let mut registry = ApprovalGateRegistry::new();
        let config = make_config(vec!["alice"], ApprovalPolicy::Any);
        let id = registry.open_gate(config);
        assert!(registry.get_gate(id).is_some());
        assert_eq!(registry.gate_count(), 1);
    }

    #[test]
    fn test_registry_submit_and_pending() {
        let mut registry = ApprovalGateRegistry::new();
        let config1 = make_config(vec!["alice"], ApprovalPolicy::Any);
        let config2 = make_config(vec!["bob"], ApprovalPolicy::Any);
        let id1 = registry.open_gate(config1);
        let id2 = registry.open_gate(config2);
        assert_eq!(registry.pending_gates().len(), 2);

        registry.submit_decision(id1, approve("alice"));
        assert_eq!(registry.pending_gates().len(), 1);
        assert_eq!(registry.pending_gates()[0], id2);
    }

    #[test]
    fn test_registry_remove_gate() {
        let mut registry = ApprovalGateRegistry::new();
        let config = make_config(vec!["alice"], ApprovalPolicy::Any);
        let id = registry.open_gate(config);
        assert_eq!(registry.gate_count(), 1);
        let removed = registry.remove_gate(id);
        assert!(removed.is_some());
        assert_eq!(registry.gate_count(), 0);
    }

    #[test]
    fn test_config_builder_methods() {
        let config = make_config(vec!["alice"], ApprovalPolicy::Any)
            .with_description("Review final output")
            .with_timeout(Duration::from_secs(3600))
            .with_metadata("project", "alpha");
        assert_eq!(config.description.as_deref(), Some("Review final output"));
        assert!(config.timeout.is_some());
        assert_eq!(
            config.metadata.get("project").map(|s| s.as_str()),
            Some("alpha")
        );
    }

    #[test]
    fn test_escalation_rule() {
        let rule = EscalationRule::new(Duration::from_secs(60), "manager")
            .with_message("Urgent: please review");
        assert_eq!(rule.escalate_to, "manager");
        assert_eq!(rule.message.as_deref(), Some("Urgent: please review"));
    }

    #[test]
    fn test_decision_after_close_is_ignored() {
        let config = make_config(vec!["alice", "bob"], ApprovalPolicy::Any);
        let mut gate = ApprovalGate::new(GateId::new(1), config);
        gate.submit_decision(approve("alice"));
        assert!(gate.is_approved());
        // submit another decision -- should be ignored
        gate.submit_decision(reject("bob"));
        assert!(gate.is_approved()); // still approved
        assert_eq!(gate.decisions.len(), 1);
    }

    #[test]
    fn test_default_registry() {
        let registry = ApprovalGateRegistry::default();
        assert_eq!(registry.gate_count(), 0);
    }
}
