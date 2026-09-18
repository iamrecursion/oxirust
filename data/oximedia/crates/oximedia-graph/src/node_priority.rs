//! Node priority and scheduling weight assignment.
//!
//! This module assigns priority values to graph nodes for scheduling
//! decisions. Higher-priority nodes are processed first when multiple
//! nodes are ready, enabling latency-sensitive paths (e.g., live preview)
//! to be prioritized over background tasks (e.g., proxy generation).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// Priority class for a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityClass {
    /// Real-time path (lowest latency, highest priority).
    RealTime,
    /// Interactive path (user-facing, high priority).
    Interactive,
    /// Normal processing (default).
    Normal,
    /// Background / best-effort processing.
    Background,
    /// Idle processing (only when nothing else is pending).
    Idle,
}

impl PriorityClass {
    /// Get the numeric weight for this priority class.
    ///
    /// Higher values mean higher priority.
    pub fn weight(&self) -> i32 {
        match self {
            Self::RealTime => 1000,
            Self::Interactive => 500,
            Self::Normal => 100,
            Self::Background => 10,
            Self::Idle => 1,
        }
    }

    /// Return all priority classes in descending priority order.
    pub fn all_descending() -> &'static [PriorityClass] {
        &[
            PriorityClass::RealTime,
            PriorityClass::Interactive,
            PriorityClass::Normal,
            PriorityClass::Background,
            PriorityClass::Idle,
        ]
    }
}

impl fmt::Display for PriorityClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RealTime => write!(f, "RealTime"),
            Self::Interactive => write!(f, "Interactive"),
            Self::Normal => write!(f, "Normal"),
            Self::Background => write!(f, "Background"),
            Self::Idle => write!(f, "Idle"),
        }
    }
}

impl Default for PriorityClass {
    fn default() -> Self {
        Self::Normal
    }
}

impl Ord for PriorityClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight().cmp(&other.weight())
    }
}

impl PartialOrd for PriorityClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A unique node identifier used within the priority system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PriorityNodeId(pub u64);

impl fmt::Display for PriorityNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Describes the priority assignment for a single node.
#[derive(Debug, Clone)]
pub struct NodePriorityEntry {
    /// The node identifier.
    pub node_id: PriorityNodeId,
    /// The base priority class.
    pub class: PriorityClass,
    /// An additional boost or penalty (-500 to +500).
    pub boost: i32,
    /// Computed effective priority (class weight + boost).
    pub effective: i32,
    /// Optional label for debugging.
    pub label: String,
}

impl NodePriorityEntry {
    /// Create a new priority entry.
    pub fn new(node_id: PriorityNodeId, class: PriorityClass, label: &str) -> Self {
        let effective = class.weight();
        Self {
            node_id,
            class,
            boost: 0,
            effective,
            label: label.to_string(),
        }
    }

    /// Apply a priority boost (positive) or penalty (negative).
    pub fn with_boost(mut self, boost: i32) -> Self {
        self.boost = boost.clamp(-500, 500);
        self.effective = self.class.weight() + self.boost;
        self
    }

    /// Recalculate effective priority after changes.
    pub fn recalculate(&mut self) {
        self.effective = self.class.weight() + self.boost;
    }
}

impl fmt::Display for NodePriorityEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] priority={} ({}{})",
            self.label,
            self.node_id,
            self.effective,
            self.class,
            if self.boost != 0 {
                format!("{:+}", self.boost)
            } else {
                String::new()
            }
        )
    }
}

/// Manages priority assignments for all nodes in a graph.
pub struct PriorityManager {
    /// Priority entries indexed by node ID.
    entries: HashMap<PriorityNodeId, NodePriorityEntry>,
    /// Default class for new nodes.
    default_class: PriorityClass,
}

impl PriorityManager {
    /// Create a new priority manager with Normal as the default class.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            default_class: PriorityClass::Normal,
        }
    }

    /// Set the default priority class for newly registered nodes.
    pub fn set_default_class(&mut self, class: PriorityClass) {
        self.default_class = class;
    }

    /// Get the default priority class.
    pub fn default_class(&self) -> PriorityClass {
        self.default_class
    }

    /// Register a node with a specific priority class.
    pub fn register(&mut self, node_id: PriorityNodeId, class: PriorityClass, label: &str) {
        self.entries
            .insert(node_id, NodePriorityEntry::new(node_id, class, label));
    }

    /// Register a node with the default priority class.
    pub fn register_default(&mut self, node_id: PriorityNodeId, label: &str) {
        let class = self.default_class;
        self.register(node_id, class, label);
    }

    /// Get the priority entry for a node.
    pub fn get(&self, node_id: PriorityNodeId) -> Option<&NodePriorityEntry> {
        self.entries.get(&node_id)
    }

    /// Get the effective priority for a node (returns 0 if not registered).
    pub fn effective_priority(&self, node_id: PriorityNodeId) -> i32 {
        self.entries.get(&node_id).map_or(0, |e| e.effective)
    }

    /// Apply a boost to a node's priority.
    pub fn apply_boost(&mut self, node_id: PriorityNodeId, boost: i32) -> bool {
        if let Some(entry) = self.entries.get_mut(&node_id) {
            entry.boost = boost.clamp(-500, 500);
            entry.recalculate();
            true
        } else {
            false
        }
    }

    /// Change a node's priority class.
    pub fn set_class(&mut self, node_id: PriorityNodeId, class: PriorityClass) -> bool {
        if let Some(entry) = self.entries.get_mut(&node_id) {
            entry.class = class;
            entry.recalculate();
            true
        } else {
            false
        }
    }

    /// Unregister a node.
    pub fn unregister(&mut self, node_id: PriorityNodeId) -> bool {
        self.entries.remove(&node_id).is_some()
    }

    /// Get the number of registered nodes.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Get all node IDs sorted by effective priority (highest first).
    pub fn sorted_by_priority(&self) -> Vec<PriorityNodeId> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by(|a, b| b.effective.cmp(&a.effective));
        entries.iter().map(|e| e.node_id).collect()
    }

    /// Get all nodes in a specific priority class.
    pub fn nodes_in_class(&self, class: PriorityClass) -> Vec<PriorityNodeId> {
        self.entries
            .values()
            .filter(|e| e.class == class)
            .map(|e| e.node_id)
            .collect()
    }

    /// Promote all nodes of one class to a higher class.
    pub fn promote_class(&mut self, from: PriorityClass, to: PriorityClass) -> usize {
        let mut count = 0;
        for entry in self.entries.values_mut() {
            if entry.class == from {
                entry.class = to;
                entry.recalculate();
                count += 1;
            }
        }
        count
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for PriorityManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_class_weight_ordering() {
        assert!(PriorityClass::RealTime.weight() > PriorityClass::Interactive.weight());
        assert!(PriorityClass::Interactive.weight() > PriorityClass::Normal.weight());
        assert!(PriorityClass::Normal.weight() > PriorityClass::Background.weight());
        assert!(PriorityClass::Background.weight() > PriorityClass::Idle.weight());
    }

    #[test]
    fn test_priority_class_default() {
        assert_eq!(PriorityClass::default(), PriorityClass::Normal);
    }

    #[test]
    fn test_priority_class_display() {
        assert_eq!(format!("{}", PriorityClass::RealTime), "RealTime");
        assert_eq!(format!("{}", PriorityClass::Idle), "Idle");
    }

    #[test]
    fn test_priority_class_ord() {
        assert!(PriorityClass::RealTime > PriorityClass::Normal);
        assert!(PriorityClass::Idle < PriorityClass::Background);
    }

    #[test]
    fn test_all_descending() {
        let all = PriorityClass::all_descending();
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], PriorityClass::RealTime);
        assert_eq!(all[4], PriorityClass::Idle);
    }

    #[test]
    fn test_node_priority_entry_new() {
        let entry =
            NodePriorityEntry::new(PriorityNodeId(1), PriorityClass::Interactive, "preview");
        assert_eq!(entry.effective, 500);
        assert_eq!(entry.boost, 0);
        assert_eq!(entry.label, "preview");
    }

    #[test]
    fn test_node_priority_entry_with_boost() {
        let entry =
            NodePriorityEntry::new(PriorityNodeId(1), PriorityClass::Normal, "n").with_boost(50);
        assert_eq!(entry.effective, 150);
        assert_eq!(entry.boost, 50);
    }

    #[test]
    fn test_boost_clamp() {
        let entry =
            NodePriorityEntry::new(PriorityNodeId(1), PriorityClass::Normal, "n").with_boost(9999);
        assert_eq!(entry.boost, 500);
    }

    #[test]
    fn test_priority_manager_register() {
        let mut mgr = PriorityManager::new();
        mgr.register(PriorityNodeId(1), PriorityClass::RealTime, "rt_node");
        assert_eq!(mgr.count(), 1);
        assert_eq!(mgr.effective_priority(PriorityNodeId(1)), 1000);
    }

    #[test]
    fn test_priority_manager_register_default() {
        let mut mgr = PriorityManager::new();
        mgr.register_default(PriorityNodeId(1), "default_node");
        assert_eq!(mgr.effective_priority(PriorityNodeId(1)), 100);
    }

    #[test]
    fn test_priority_manager_unregistered_returns_zero() {
        let mgr = PriorityManager::new();
        assert_eq!(mgr.effective_priority(PriorityNodeId(999)), 0);
    }

    #[test]
    fn test_priority_manager_sorted() {
        let mut mgr = PriorityManager::new();
        mgr.register(PriorityNodeId(1), PriorityClass::Background, "bg");
        mgr.register(PriorityNodeId(2), PriorityClass::RealTime, "rt");
        mgr.register(PriorityNodeId(3), PriorityClass::Normal, "norm");
        let sorted = mgr.sorted_by_priority();
        assert_eq!(sorted[0], PriorityNodeId(2)); // RealTime first
        assert_eq!(sorted[2], PriorityNodeId(1)); // Background last
    }

    #[test]
    fn test_priority_manager_apply_boost() {
        let mut mgr = PriorityManager::new();
        mgr.register(PriorityNodeId(1), PriorityClass::Normal, "n");
        assert!(mgr.apply_boost(PriorityNodeId(1), 200));
        assert_eq!(mgr.effective_priority(PriorityNodeId(1)), 300);
        assert!(!mgr.apply_boost(PriorityNodeId(99), 10));
    }

    #[test]
    fn test_priority_manager_set_class() {
        let mut mgr = PriorityManager::new();
        mgr.register(PriorityNodeId(1), PriorityClass::Normal, "n");
        mgr.set_class(PriorityNodeId(1), PriorityClass::RealTime);
        assert_eq!(mgr.effective_priority(PriorityNodeId(1)), 1000);
    }

    #[test]
    fn test_priority_manager_promote_class() {
        let mut mgr = PriorityManager::new();
        mgr.register(PriorityNodeId(1), PriorityClass::Background, "b1");
        mgr.register(PriorityNodeId(2), PriorityClass::Background, "b2");
        mgr.register(PriorityNodeId(3), PriorityClass::Normal, "n1");
        let promoted = mgr.promote_class(PriorityClass::Background, PriorityClass::Normal);
        assert_eq!(promoted, 2);
        assert_eq!(mgr.effective_priority(PriorityNodeId(1)), 100);
    }

    #[test]
    fn test_priority_manager_nodes_in_class() {
        let mut mgr = PriorityManager::new();
        mgr.register(PriorityNodeId(1), PriorityClass::Idle, "i1");
        mgr.register(PriorityNodeId(2), PriorityClass::Idle, "i2");
        mgr.register(PriorityNodeId(3), PriorityClass::Normal, "n1");
        let idle_nodes = mgr.nodes_in_class(PriorityClass::Idle);
        assert_eq!(idle_nodes.len(), 2);
    }
}
