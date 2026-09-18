#![allow(dead_code)]

//! Synchronization chain analysis and tracking.
//!
//! This module models the chain of clock references from a grandmaster
//! down to endpoint clocks, enabling analysis of cumulative error,
//! traceability, and chain-of-trust for timing paths.

use std::collections::HashMap;

/// Unique identifier for a node in the sync chain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChainNodeId(String);

impl ChainNodeId {
    /// Creates a new node identifier.
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    /// Returns the identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ChainNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Role of a node within the synchronization chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// Grandmaster clock (top of the chain).
    Grandmaster,
    /// Boundary clock (transparent relay with clock recovery).
    BoundaryClock,
    /// Transparent clock (forwards with correction field).
    TransparentClock,
    /// Ordinary clock (endpoint / slave).
    OrdinaryClock,
    /// NTP server acting as a reference.
    NtpServer,
}

impl NodeRole {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Grandmaster => "Grandmaster",
            Self::BoundaryClock => "Boundary Clock",
            Self::TransparentClock => "Transparent Clock",
            Self::OrdinaryClock => "Ordinary Clock",
            Self::NtpServer => "NTP Server",
        }
    }

    /// Returns whether this role is a reference source.
    #[must_use]
    pub const fn is_reference(&self) -> bool {
        matches!(self, Self::Grandmaster | Self::NtpServer)
    }
}

/// A node in the synchronization chain.
#[derive(Debug, Clone)]
pub struct ChainNode {
    /// Unique identifier.
    pub id: ChainNodeId,
    /// Role of this node.
    pub role: NodeRole,
    /// Human-readable description.
    pub description: String,
    /// Estimated error contribution in nanoseconds.
    pub error_ns: f64,
    /// Parent node ID (None if this is the grandmaster).
    pub parent_id: Option<ChainNodeId>,
    /// Hop count from the grandmaster.
    pub hop_count: u32,
}

impl ChainNode {
    /// Creates a new grandmaster node.
    #[must_use]
    pub fn grandmaster(id: &str, description: &str, error_ns: f64) -> Self {
        Self {
            id: ChainNodeId::new(id),
            role: NodeRole::Grandmaster,
            description: description.to_string(),
            error_ns,
            parent_id: None,
            hop_count: 0,
        }
    }

    /// Creates a new child node connected to a parent.
    #[must_use]
    pub fn child(
        id: &str,
        role: NodeRole,
        description: &str,
        error_ns: f64,
        parent: &ChainNodeId,
        parent_hops: u32,
    ) -> Self {
        Self {
            id: ChainNodeId::new(id),
            role,
            description: description.to_string(),
            error_ns,
            parent_id: Some(parent.clone()),
            hop_count: parent_hops + 1,
        }
    }
}

/// A synchronization chain representing the full path from grandmaster to endpoints.
#[derive(Debug, Clone)]
pub struct SyncChain {
    /// All nodes in the chain, keyed by their ID.
    nodes: HashMap<ChainNodeId, ChainNode>,
}

impl SyncChain {
    /// Creates a new empty sync chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Adds a node to the chain. Returns false if a node with the same ID exists.
    pub fn add_node(&mut self, node: ChainNode) -> bool {
        if self.nodes.contains_key(&node.id) {
            return false;
        }
        self.nodes.insert(node.id.clone(), node);
        true
    }

    /// Returns the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Looks up a node by ID.
    #[must_use]
    pub fn get_node(&self, id: &ChainNodeId) -> Option<&ChainNode> {
        self.nodes.get(id)
    }

    /// Returns all grandmaster nodes.
    #[must_use]
    pub fn grandmasters(&self) -> Vec<&ChainNode> {
        self.nodes
            .values()
            .filter(|n| n.role == NodeRole::Grandmaster)
            .collect()
    }

    /// Returns all endpoint (ordinary clock) nodes.
    #[must_use]
    pub fn endpoints(&self) -> Vec<&ChainNode> {
        self.nodes
            .values()
            .filter(|n| n.role == NodeRole::OrdinaryClock)
            .collect()
    }

    /// Computes the path from grandmaster to a given node ID.
    ///
    /// Returns the chain of node IDs from grandmaster (first) to the target (last),
    /// or `None` if the node is not found or the chain is broken.
    #[must_use]
    pub fn path_to(&self, target: &ChainNodeId) -> Option<Vec<ChainNodeId>> {
        let mut path = Vec::new();
        let mut current_id = target.clone();

        loop {
            let node = self.nodes.get(&current_id)?;
            path.push(current_id.clone());

            match &node.parent_id {
                Some(parent) => current_id = parent.clone(),
                None => break, // Reached the root
            }

            // Safety: prevent infinite loops
            if path.len() > self.nodes.len() {
                return None;
            }
        }

        path.reverse();
        Some(path)
    }

    /// Computes the cumulative error along the path to a node.
    #[must_use]
    pub fn cumulative_error_ns(&self, target: &ChainNodeId) -> Option<f64> {
        let path = self.path_to(target)?;
        let total = path
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .map(|n| n.error_ns)
            .sum();
        Some(total)
    }

    /// Returns the maximum hop count across all nodes.
    #[must_use]
    pub fn max_hops(&self) -> u32 {
        self.nodes.values().map(|n| n.hop_count).max().unwrap_or(0)
    }

    /// Returns a summary of node counts by role.
    #[must_use]
    pub fn role_summary(&self) -> HashMap<&'static str, usize> {
        let mut summary = HashMap::new();
        for node in self.nodes.values() {
            *summary.entry(node.role.label()).or_insert(0) += 1;
        }
        summary
    }

    /// Validates the chain for structural issues.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        let grandmasters = self.grandmasters();
        if grandmasters.is_empty() {
            issues.push("No grandmaster node found".to_string());
        }

        for node in self.nodes.values() {
            if let Some(ref parent_id) = node.parent_id {
                if !self.nodes.contains_key(parent_id) {
                    issues.push(format!(
                        "Node {} references missing parent {}",
                        node.id, parent_id
                    ));
                }
            }
        }

        issues
    }
}

impl Default for SyncChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_chain() -> SyncChain {
        let mut chain = SyncChain::new();
        let gm = ChainNode::grandmaster("GM-1", "GPS Grandmaster", 50.0);
        chain.add_node(gm);

        let bc = ChainNode::child(
            "BC-1",
            NodeRole::BoundaryClock,
            "Boundary Clock 1",
            100.0,
            &ChainNodeId::new("GM-1"),
            0,
        );
        chain.add_node(bc);

        let oc = ChainNode::child(
            "OC-1",
            NodeRole::OrdinaryClock,
            "Endpoint 1",
            200.0,
            &ChainNodeId::new("BC-1"),
            1,
        );
        chain.add_node(oc);

        chain
    }

    #[test]
    fn test_chain_node_id_display() {
        let id = ChainNodeId::new("GM-1");
        assert_eq!(id.to_string(), "GM-1");
    }

    #[test]
    fn test_node_role_labels() {
        assert_eq!(NodeRole::Grandmaster.label(), "Grandmaster");
        assert_eq!(NodeRole::OrdinaryClock.label(), "Ordinary Clock");
        assert_eq!(NodeRole::NtpServer.label(), "NTP Server");
    }

    #[test]
    fn test_node_role_is_reference() {
        assert!(NodeRole::Grandmaster.is_reference());
        assert!(NodeRole::NtpServer.is_reference());
        assert!(!NodeRole::BoundaryClock.is_reference());
        assert!(!NodeRole::OrdinaryClock.is_reference());
    }

    #[test]
    fn test_chain_node_count() {
        let chain = build_simple_chain();
        assert_eq!(chain.node_count(), 3);
    }

    #[test]
    fn test_chain_grandmasters() {
        let chain = build_simple_chain();
        let gms = chain.grandmasters();
        assert_eq!(gms.len(), 1);
        assert_eq!(gms[0].id.as_str(), "GM-1");
    }

    #[test]
    fn test_chain_endpoints() {
        let chain = build_simple_chain();
        let eps = chain.endpoints();
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].id.as_str(), "OC-1");
    }

    #[test]
    fn test_path_to_endpoint() {
        let chain = build_simple_chain();
        let path = chain
            .path_to(&ChainNodeId::new("OC-1"))
            .expect("should succeed in test");
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].as_str(), "GM-1");
        assert_eq!(path[1].as_str(), "BC-1");
        assert_eq!(path[2].as_str(), "OC-1");
    }

    #[test]
    fn test_path_to_grandmaster() {
        let chain = build_simple_chain();
        let path = chain
            .path_to(&ChainNodeId::new("GM-1"))
            .expect("should succeed in test");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].as_str(), "GM-1");
    }

    #[test]
    fn test_path_to_missing_node() {
        let chain = build_simple_chain();
        assert!(chain.path_to(&ChainNodeId::new("MISSING")).is_none());
    }

    #[test]
    fn test_cumulative_error() {
        let chain = build_simple_chain();
        let error = chain
            .cumulative_error_ns(&ChainNodeId::new("OC-1"))
            .expect("should succeed in test");
        // 50 + 100 + 200 = 350
        assert!((error - 350.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_hops() {
        let chain = build_simple_chain();
        assert_eq!(chain.max_hops(), 2);
    }

    #[test]
    fn test_role_summary() {
        let chain = build_simple_chain();
        let summary = chain.role_summary();
        assert_eq!(*summary.get("Grandmaster").unwrap_or(&0), 1);
        assert_eq!(*summary.get("Boundary Clock").unwrap_or(&0), 1);
        assert_eq!(*summary.get("Ordinary Clock").unwrap_or(&0), 1);
    }

    #[test]
    fn test_validate_ok() {
        let chain = build_simple_chain();
        let issues = chain.validate();
        assert!(issues.is_empty(), "unexpected: {issues:?}");
    }

    #[test]
    fn test_validate_no_grandmaster() {
        let chain = SyncChain::new();
        let issues = chain.validate();
        assert!(issues.iter().any(|i| i.contains("grandmaster")));
    }

    #[test]
    fn test_add_duplicate_node() {
        let mut chain = SyncChain::new();
        let gm1 = ChainNode::grandmaster("GM-1", "First", 10.0);
        let gm2 = ChainNode::grandmaster("GM-1", "Duplicate", 20.0);
        assert!(chain.add_node(gm1));
        assert!(!chain.add_node(gm2));
        assert_eq!(chain.node_count(), 1);
    }
}
