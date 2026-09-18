#![allow(dead_code)]
//! Policy-driven routing decision engine.
//!
//! Provides [`PolicyType`], [`RoutingPolicy`], and [`PolicyEngine`] for
//! evaluating and selecting routes based on configurable policies.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Policy type
// ---------------------------------------------------------------------------

/// The kind of policy to apply when selecting among candidate routes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyType {
    /// Always prefer the route with the lowest metric.
    LowestMetric,
    /// Always prefer the route on the specified interface.
    PreferInterface(String),
    /// Round-robin across all matching routes.
    RoundRobin,
    /// Weighted routing: interfaces mapped to weights (higher = more traffic).
    Weighted(HashMap<String, u32>),
    /// Failover: primary interface with automatic fallback.
    Failover {
        /// Primary interface name.
        primary: String,
        /// Backup interface name.
        backup: String,
    },
}

// ---------------------------------------------------------------------------
// Route candidate
// ---------------------------------------------------------------------------

/// A candidate route for policy evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteCandidate {
    /// Destination identifier (e.g., IP string or symbolic name).
    pub destination: String,
    /// Outgoing interface name.
    pub interface: String,
    /// Administrative metric (lower = preferred for `LowestMetric`).
    pub metric: u32,
    /// Whether this route's interface is currently up.
    pub interface_up: bool,
}

impl RouteCandidate {
    /// Creates a new candidate with the interface marked as up.
    pub fn new(destination: impl Into<String>, interface: impl Into<String>, metric: u32) -> Self {
        Self {
            destination: destination.into(),
            interface: interface.into(),
            metric,
            interface_up: true,
        }
    }

    /// Marks the interface as down.
    pub fn with_interface_down(mut self) -> Self {
        self.interface_up = false;
        self
    }
}

// ---------------------------------------------------------------------------
// RoutingPolicy
// ---------------------------------------------------------------------------

/// A named routing policy with an associated type.
#[derive(Debug, Clone)]
pub struct RoutingPolicy {
    /// Human-readable policy name.
    pub name: String,
    /// The type/logic of the policy.
    pub policy_type: PolicyType,
    /// Whether this policy is currently active.
    pub enabled: bool,
}

impl RoutingPolicy {
    /// Creates a new enabled policy.
    pub fn new(name: impl Into<String>, policy_type: PolicyType) -> Self {
        Self {
            name: name.into(),
            policy_type,
            enabled: true,
        }
    }

    /// Disables the policy.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enables the policy.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

// ---------------------------------------------------------------------------
// PolicyEngine
// ---------------------------------------------------------------------------

/// Evaluates a list of [`RouteCandidate`]s against registered policies and
/// returns the best candidate according to the active policy.
#[derive(Debug, Default)]
pub struct PolicyEngine {
    policies: Vec<RoutingPolicy>,
    rr_counter: usize,
}

impl PolicyEngine {
    /// Creates an empty policy engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a policy.
    pub fn add_policy(&mut self, policy: RoutingPolicy) {
        self.policies.push(policy);
    }

    /// Returns the number of registered policies.
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    /// Evaluates `candidates` using the first *enabled* policy in the list.
    ///
    /// Returns a reference to the selected candidate, or `None` if no
    /// candidates are available or no policy is enabled.
    pub fn evaluate<'a>(&mut self, candidates: &'a [RouteCandidate]) -> Option<&'a RouteCandidate> {
        let up: Vec<&RouteCandidate> = candidates.iter().filter(|c| c.interface_up).collect();

        if up.is_empty() {
            return None;
        }

        let policy = self.policies.iter().find(|p| p.enabled)?;

        match &policy.policy_type {
            PolicyType::LowestMetric => up.into_iter().min_by_key(|c| c.metric),

            PolicyType::PreferInterface(iface) => {
                // Find preferred interface; fall back to lowest metric.
                up.iter()
                    .copied()
                    .find(|c| c.interface == *iface)
                    .or_else(|| up.into_iter().min_by_key(|c| c.metric))
            }

            PolicyType::RoundRobin => {
                let idx = self.rr_counter % up.len();
                self.rr_counter += 1;
                Some(up[idx])
            }

            PolicyType::Weighted(weights) => {
                // Select the interface with the highest weight that is present
                // among the up candidates.
                let best = up
                    .iter()
                    .copied()
                    .max_by_key(|c| weights.get(&c.interface).copied().unwrap_or(0));
                best
            }

            PolicyType::Failover { primary, backup } => up
                .iter()
                .copied()
                .find(|c| &c.interface == primary)
                .or_else(|| up.iter().copied().find(|c| &c.interface == backup))
                .or_else(|| up.into_iter().min_by_key(|c| c.metric)),
        }
    }

    /// Removes all registered policies.
    pub fn clear_policies(&mut self) {
        self.policies.clear();
    }

    /// Returns an iterator over policy names.
    pub fn policy_names(&self) -> impl Iterator<Item = &str> {
        self.policies.iter().map(|p| p.name.as_str())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> Vec<RouteCandidate> {
        vec![
            RouteCandidate::new("10.0.0.0/8", "eth0", 10),
            RouteCandidate::new("10.0.0.0/8", "eth1", 5),
            RouteCandidate::new("10.0.0.0/8", "eth2", 20),
        ]
    }

    #[test]
    fn test_lowest_metric_selects_min() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("lowest", PolicyType::LowestMetric));
        let cands = candidates();
        let result = engine.evaluate(&cands).expect("should succeed in test");
        assert_eq!(result.interface, "eth1"); // metric 5
    }

    #[test]
    fn test_prefer_interface_found() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new(
            "prefer",
            PolicyType::PreferInterface("eth2".to_string()),
        ));
        let cands = candidates();
        let result = engine.evaluate(&cands).expect("should succeed in test");
        assert_eq!(result.interface, "eth2");
    }

    #[test]
    fn test_prefer_interface_fallback_to_lowest() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new(
            "prefer",
            PolicyType::PreferInterface("eth99".to_string()),
        ));
        let cands = candidates();
        let result = engine.evaluate(&cands).expect("should succeed in test");
        // Falls back to lowest metric = eth1 (metric 5)
        assert_eq!(result.interface, "eth1");
    }

    #[test]
    fn test_round_robin_cycles() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("rr", PolicyType::RoundRobin));
        let cands = candidates();
        let first = engine
            .evaluate(&cands)
            .expect("should succeed in test")
            .interface
            .clone();
        let second = engine
            .evaluate(&cands)
            .expect("should succeed in test")
            .interface
            .clone();
        // Should differ (first call → idx 0, second → idx 1)
        assert_ne!(first, second);
    }

    #[test]
    fn test_weighted_selects_highest_weight() {
        let mut weights = HashMap::new();
        weights.insert("eth0".to_string(), 1);
        weights.insert("eth1".to_string(), 10);
        weights.insert("eth2".to_string(), 2);
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("w", PolicyType::Weighted(weights)));
        let cands = candidates();
        let result = engine.evaluate(&cands).expect("should succeed in test");
        assert_eq!(result.interface, "eth1");
    }

    #[test]
    fn test_failover_primary_preferred() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new(
            "fo",
            PolicyType::Failover {
                primary: "eth0".to_string(),
                backup: "eth1".to_string(),
            },
        ));
        let cands = candidates();
        let result = engine.evaluate(&cands).expect("should succeed in test");
        assert_eq!(result.interface, "eth0");
    }

    #[test]
    fn test_failover_uses_backup_when_primary_down() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new(
            "fo",
            PolicyType::Failover {
                primary: "eth0".to_string(),
                backup: "eth1".to_string(),
            },
        ));
        let cands = vec![
            RouteCandidate::new("10.0.0.0/8", "eth0", 10).with_interface_down(),
            RouteCandidate::new("10.0.0.0/8", "eth1", 5),
        ];
        let result = engine.evaluate(&cands).expect("should succeed in test");
        assert_eq!(result.interface, "eth1");
    }

    #[test]
    fn test_no_candidates_returns_none() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("lowest", PolicyType::LowestMetric));
        let result = engine.evaluate(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_all_interfaces_down_returns_none() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("lowest", PolicyType::LowestMetric));
        let cands = vec![
            RouteCandidate::new("dst", "eth0", 1).with_interface_down(),
            RouteCandidate::new("dst", "eth1", 2).with_interface_down(),
        ];
        assert!(engine.evaluate(&cands).is_none());
    }

    #[test]
    fn test_disabled_policy_skipped() {
        let mut engine = PolicyEngine::new();
        let mut p1 =
            RoutingPolicy::new("disabled", PolicyType::PreferInterface("eth2".to_string()));
        p1.disable();
        let p2 = RoutingPolicy::new("lowest", PolicyType::LowestMetric);
        engine.add_policy(p1);
        engine.add_policy(p2);
        let cands = candidates();
        // p1 is disabled → falls through to p2 (LowestMetric → eth1)
        let result = engine.evaluate(&cands).expect("should succeed in test");
        assert_eq!(result.interface, "eth1");
    }

    #[test]
    fn test_policy_enable_disable() {
        let mut policy = RoutingPolicy::new("p", PolicyType::LowestMetric);
        assert!(policy.enabled);
        policy.disable();
        assert!(!policy.enabled);
        policy.enable();
        assert!(policy.enabled);
    }

    #[test]
    fn test_policy_count() {
        let mut engine = PolicyEngine::new();
        assert_eq!(engine.policy_count(), 0);
        engine.add_policy(RoutingPolicy::new("a", PolicyType::LowestMetric));
        engine.add_policy(RoutingPolicy::new("b", PolicyType::RoundRobin));
        assert_eq!(engine.policy_count(), 2);
    }

    #[test]
    fn test_clear_policies() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("a", PolicyType::LowestMetric));
        engine.clear_policies();
        assert_eq!(engine.policy_count(), 0);
    }

    #[test]
    fn test_policy_names_iterator() {
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("alpha", PolicyType::LowestMetric));
        engine.add_policy(RoutingPolicy::new("beta", PolicyType::RoundRobin));
        let names: Vec<&str> = engine.policy_names().collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn test_route_candidate_with_interface_down() {
        let c = RouteCandidate::new("dst", "eth0", 1).with_interface_down();
        assert!(!c.interface_up);
    }

    #[test]
    fn test_weighted_unknown_interface_treated_as_zero() {
        let weights: HashMap<String, u32> = HashMap::new(); // no weights → all 0
        let mut engine = PolicyEngine::new();
        engine.add_policy(RoutingPolicy::new("w", PolicyType::Weighted(weights)));
        let cands = candidates();
        // All weights 0 — any candidate may be returned; just confirm no panic.
        assert!(engine.evaluate(&cands).is_some());
    }
}
