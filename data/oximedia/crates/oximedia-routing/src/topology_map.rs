#![allow(dead_code)]
//! Network topology mapping for routing infrastructure.
//!
//! Models the physical and logical topology of media routing networks,
//! including nodes (devices), links (connections), and their properties.
//! Supports topology discovery, path enumeration, and health tracking.

use std::collections::HashMap;

/// Unique identifier for a topology node.
pub type NodeId = u64;

/// Unique identifier for a link between nodes.
pub type LinkId = u64;

/// Type of device represented by a topology node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Audio/video source (camera, microphone, deck).
    Source,
    /// Audio/video destination (monitor, recorder).
    Destination,
    /// Router or switch (crosspoint matrix, IP switch).
    Router,
    /// Processing node (frame sync, colour corrector).
    Processor,
    /// Gateway between network segments.
    Gateway,
    /// Multiviewer/monitoring device.
    Multiviewer,
}

/// Health status of a node or link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Fully operational.
    Healthy,
    /// Operational but with warnings (e.g. high utilisation).
    Degraded,
    /// Not responding or faulted.
    Down,
    /// Status is not yet known.
    Unknown,
}

/// A node in the topology graph.
#[derive(Debug, Clone)]
pub struct TopologyNode {
    /// Node identifier.
    pub id: NodeId,
    /// Human-readable label.
    pub label: String,
    /// Type of this node.
    pub node_type: NodeType,
    /// Number of input ports.
    pub input_count: u32,
    /// Number of output ports.
    pub output_count: u32,
    /// Current health status.
    pub health: HealthStatus,
    /// Arbitrary key-value properties.
    pub properties: HashMap<String, String>,
}

impl TopologyNode {
    /// Create a new topology node.
    pub fn new(id: NodeId, label: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id,
            label: label.into(),
            node_type,
            input_count: 0,
            output_count: 0,
            health: HealthStatus::Unknown,
            properties: HashMap::new(),
        }
    }

    /// Set port counts.
    pub fn with_ports(mut self, inputs: u32, outputs: u32) -> Self {
        self.input_count = inputs;
        self.output_count = outputs;
        self
    }

    /// Set health status.
    pub fn with_health(mut self, health: HealthStatus) -> Self {
        self.health = health;
        self
    }

    /// Total port count.
    pub fn total_ports(&self) -> u32 {
        self.input_count + self.output_count
    }
}

/// Media transport type for a link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkTransport {
    /// SDI (serial digital interface).
    Sdi,
    /// SMPTE ST 2110 IP.
    St2110,
    /// NDI.
    Ndi,
    /// MADI.
    Madi,
    /// Dante audio-over-IP.
    Dante,
    /// AES67.
    Aes67,
    /// HDMI.
    Hdmi,
    /// Internal (virtual / software link).
    Internal,
}

/// A directional link between two topology nodes.
#[derive(Debug, Clone)]
pub struct TopologyLink {
    /// Link identifier.
    pub id: LinkId,
    /// Source node.
    pub from_node: NodeId,
    /// Source port index.
    pub from_port: u32,
    /// Destination node.
    pub to_node: NodeId,
    /// Destination port index.
    pub to_port: u32,
    /// Transport type.
    pub transport: LinkTransport,
    /// Bandwidth capacity in megabits per second.
    pub bandwidth_mbps: f64,
    /// Current utilisation as a fraction [0.0, 1.0].
    pub utilisation: f64,
    /// Health status.
    pub health: HealthStatus,
    /// Latency in microseconds (measured or estimated).
    pub latency_us: u64,
}

impl TopologyLink {
    /// Create a new link.
    pub fn new(
        id: LinkId,
        from_node: NodeId,
        from_port: u32,
        to_node: NodeId,
        to_port: u32,
        transport: LinkTransport,
    ) -> Self {
        Self {
            id,
            from_node,
            from_port,
            to_node,
            to_port,
            transport,
            bandwidth_mbps: 1000.0,
            utilisation: 0.0,
            health: HealthStatus::Unknown,
            latency_us: 0,
        }
    }

    /// Whether the link is currently healthy enough to carry traffic.
    pub fn is_usable(&self) -> bool {
        self.health == HealthStatus::Healthy || self.health == HealthStatus::Degraded
    }

    /// Available bandwidth in megabits per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn available_bandwidth_mbps(&self) -> f64 {
        self.bandwidth_mbps * (1.0 - self.utilisation)
    }
}

/// An ordered path through the topology from source to destination.
#[derive(Debug, Clone)]
pub struct TopologyPath {
    /// Sequence of node IDs in the path (source first, destination last).
    pub nodes: Vec<NodeId>,
    /// Link IDs connecting consecutive nodes.
    pub links: Vec<LinkId>,
    /// Total latency of this path in microseconds.
    pub total_latency_us: u64,
    /// Minimum available bandwidth along the path (bottleneck), in Mbps.
    pub bottleneck_bandwidth_mbps: f64,
}

impl TopologyPath {
    /// Number of hops (links) in the path.
    pub fn hop_count(&self) -> usize {
        self.links.len()
    }

    /// Whether the path is direct (single hop).
    pub fn is_direct(&self) -> bool {
        self.links.len() == 1
    }
}

/// The full topology map holding all nodes and links.
#[derive(Debug)]
pub struct TopologyMap {
    /// All nodes keyed by their ID.
    nodes: HashMap<NodeId, TopologyNode>,
    /// All links keyed by their ID.
    links: HashMap<LinkId, TopologyLink>,
    /// Next auto-assigned node ID.
    next_node_id: NodeId,
    /// Next auto-assigned link ID.
    next_link_id: LinkId,
}

impl TopologyMap {
    /// Create an empty topology map.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            links: HashMap::new(),
            next_node_id: 1,
            next_link_id: 1,
        }
    }

    /// Add a node and return its assigned ID.
    pub fn add_node(&mut self, label: impl Into<String>, node_type: NodeType) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        let node = TopologyNode::new(id, label, node_type);
        self.nodes.insert(id, node);
        id
    }

    /// Add a node with full specification.
    pub fn insert_node(&mut self, node: TopologyNode) {
        if node.id >= self.next_node_id {
            self.next_node_id = node.id + 1;
        }
        self.nodes.insert(node.id, node);
    }

    /// Add a link and return its assigned ID.
    pub fn add_link(
        &mut self,
        from_node: NodeId,
        from_port: u32,
        to_node: NodeId,
        to_port: u32,
        transport: LinkTransport,
    ) -> Option<LinkId> {
        if !self.nodes.contains_key(&from_node) || !self.nodes.contains_key(&to_node) {
            return None;
        }
        let id = self.next_link_id;
        self.next_link_id += 1;
        let link = TopologyLink::new(id, from_node, from_port, to_node, to_port, transport);
        self.links.insert(id, link);
        Some(id)
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&TopologyNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut TopologyNode> {
        self.nodes.get_mut(&id)
    }

    /// Get a link by ID.
    pub fn link(&self, id: LinkId) -> Option<&TopologyLink> {
        self.links.get(&id)
    }

    /// Get a mutable link by ID.
    pub fn link_mut(&mut self, id: LinkId) -> Option<&mut TopologyLink> {
        self.links.get_mut(&id)
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of links.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Return all nodes of a specific type.
    pub fn nodes_of_type(&self, node_type: NodeType) -> Vec<&TopologyNode> {
        self.nodes
            .values()
            .filter(|n| n.node_type == node_type)
            .collect()
    }

    /// Return all links originating from a given node.
    pub fn outgoing_links(&self, node_id: NodeId) -> Vec<&TopologyLink> {
        self.links
            .values()
            .filter(|l| l.from_node == node_id)
            .collect()
    }

    /// Return all links arriving at a given node.
    pub fn incoming_links(&self, node_id: NodeId) -> Vec<&TopologyLink> {
        self.links
            .values()
            .filter(|l| l.to_node == node_id)
            .collect()
    }

    /// Find all direct links between two nodes.
    pub fn links_between(&self, from: NodeId, to: NodeId) -> Vec<&TopologyLink> {
        self.links
            .values()
            .filter(|l| l.from_node == from && l.to_node == to)
            .collect()
    }

    /// Count nodes currently in a given health state.
    pub fn nodes_with_health(&self, status: HealthStatus) -> usize {
        self.nodes.values().filter(|n| n.health == status).count()
    }

    /// Count links currently in a given health state.
    pub fn links_with_health(&self, status: HealthStatus) -> usize {
        self.links.values().filter(|l| l.health == status).count()
    }

    /// Remove a node and all its connected links.
    pub fn remove_node(&mut self, id: NodeId) -> Option<TopologyNode> {
        let node = self.nodes.remove(&id)?;
        self.links
            .retain(|_, l| l.from_node != id && l.to_node != id);
        Some(node)
    }

    /// Remove a link.
    pub fn remove_link(&mut self, id: LinkId) -> Option<TopologyLink> {
        self.links.remove(&id)
    }
}

impl Default for TopologyMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let n = TopologyNode::new(1, "Cam1", NodeType::Source);
        assert_eq!(n.id, 1);
        assert_eq!(n.label, "Cam1");
        assert_eq!(n.node_type, NodeType::Source);
        assert_eq!(n.health, HealthStatus::Unknown);
    }

    #[test]
    fn test_node_with_ports() {
        let n = TopologyNode::new(1, "Router", NodeType::Router).with_ports(16, 16);
        assert_eq!(n.input_count, 16);
        assert_eq!(n.output_count, 16);
        assert_eq!(n.total_ports(), 32);
    }

    #[test]
    fn test_link_usability() {
        let mut link = TopologyLink::new(1, 1, 0, 2, 0, LinkTransport::Sdi);
        link.health = HealthStatus::Healthy;
        assert!(link.is_usable());
        link.health = HealthStatus::Down;
        assert!(!link.is_usable());
    }

    #[test]
    fn test_link_available_bandwidth() {
        let mut link = TopologyLink::new(1, 1, 0, 2, 0, LinkTransport::St2110);
        link.bandwidth_mbps = 1000.0;
        link.utilisation = 0.6;
        let avail = link.available_bandwidth_mbps();
        assert!((avail - 400.0).abs() < 1e-6);
    }

    #[test]
    fn test_topology_path_hop_count() {
        let path = TopologyPath {
            nodes: vec![1, 2, 3],
            links: vec![10, 11],
            total_latency_us: 500,
            bottleneck_bandwidth_mbps: 900.0,
        };
        assert_eq!(path.hop_count(), 2);
        assert!(!path.is_direct());
    }

    #[test]
    fn test_topology_path_direct() {
        let path = TopologyPath {
            nodes: vec![1, 2],
            links: vec![10],
            total_latency_us: 100,
            bottleneck_bandwidth_mbps: 1000.0,
        };
        assert!(path.is_direct());
    }

    #[test]
    fn test_map_add_node() {
        let mut map = TopologyMap::new();
        let id = map.add_node("Source1", NodeType::Source);
        assert_eq!(id, 1);
        assert_eq!(map.node_count(), 1);
        assert!(map.node(id).is_some());
    }

    #[test]
    fn test_map_add_link() {
        let mut map = TopologyMap::new();
        let s = map.add_node("Src", NodeType::Source);
        let d = map.add_node("Dst", NodeType::Destination);
        let lid = map.add_link(s, 0, d, 0, LinkTransport::Sdi);
        assert!(lid.is_some());
        assert_eq!(map.link_count(), 1);
    }

    #[test]
    fn test_map_add_link_invalid_node() {
        let mut map = TopologyMap::new();
        let _ = map.add_node("Src", NodeType::Source);
        let lid = map.add_link(1, 0, 999, 0, LinkTransport::Sdi);
        assert!(lid.is_none());
    }

    #[test]
    fn test_map_nodes_of_type() {
        let mut map = TopologyMap::new();
        map.add_node("Src1", NodeType::Source);
        map.add_node("Src2", NodeType::Source);
        map.add_node("Router", NodeType::Router);
        assert_eq!(map.nodes_of_type(NodeType::Source).len(), 2);
        assert_eq!(map.nodes_of_type(NodeType::Router).len(), 1);
    }

    #[test]
    fn test_map_outgoing_incoming_links() {
        let mut map = TopologyMap::new();
        let a = map.add_node("A", NodeType::Source);
        let b = map.add_node("B", NodeType::Router);
        let c = map.add_node("C", NodeType::Destination);
        map.add_link(a, 0, b, 0, LinkTransport::St2110);
        map.add_link(b, 0, c, 0, LinkTransport::St2110);
        assert_eq!(map.outgoing_links(b).len(), 1);
        assert_eq!(map.incoming_links(b).len(), 1);
    }

    #[test]
    fn test_map_remove_node_cascades() {
        let mut map = TopologyMap::new();
        let a = map.add_node("A", NodeType::Source);
        let b = map.add_node("B", NodeType::Destination);
        map.add_link(a, 0, b, 0, LinkTransport::Sdi);
        assert_eq!(map.link_count(), 1);
        map.remove_node(a);
        assert_eq!(map.node_count(), 1);
        assert_eq!(map.link_count(), 0);
    }

    #[test]
    fn test_map_links_between() {
        let mut map = TopologyMap::new();
        let a = map.add_node("A", NodeType::Source);
        let b = map.add_node("B", NodeType::Destination);
        map.add_link(a, 0, b, 0, LinkTransport::Sdi);
        map.add_link(a, 1, b, 1, LinkTransport::St2110);
        assert_eq!(map.links_between(a, b).len(), 2);
        assert_eq!(map.links_between(b, a).len(), 0);
    }
}
