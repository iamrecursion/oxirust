//! Signal path analysis for professional broadcast infrastructure.
//!
//! Models signal nodes and connections between them, with support for
//! path tracing (BFS) and bandwidth bottleneck detection.

#![allow(dead_code)]

/// The physical or logical format of a signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalFormat {
    /// 3G-SDI (SMPTE 424M) — up to 3 Gbit/s.
    Sdi3G,
    /// 12G-SDI (SMPTE 2082) — up to 12 Gbit/s.
    Sdi12G,
    /// HDMI 2.0 — up to 18 Gbit/s.
    Hdmi20,
    /// `DisplayPort` 1.4 — up to 32.4 Gbit/s.
    Dp14,
    /// SMPTE ST 2110 over IP.
    Ip2110,
}

impl SignalFormat {
    /// Maximum raw bandwidth in Gbit/s for this format.
    #[must_use]
    pub fn max_bandwidth_gbps(&self) -> f32 {
        match self {
            Self::Sdi3G => 3.0,
            Self::Sdi12G => 12.0,
            Self::Hdmi20 => 18.0,
            Self::Dp14 => 32.4,
            Self::Ip2110 => 10.0,
        }
    }

    /// Returns `true` if this format can carry a 4K (UHD) signal.
    #[must_use]
    pub fn supports_4k(&self) -> bool {
        match self {
            Self::Sdi3G => false,
            Self::Sdi12G | Self::Hdmi20 | Self::Dp14 | Self::Ip2110 => true,
        }
    }
}

/// A node in the signal path (device, converter, router, etc.).
#[derive(Debug, Clone)]
pub struct SignalNode {
    /// Unique identifier for this node.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Signal format used by this node.
    pub format: SignalFormat,
    /// IDs of nodes that feed into this node.
    pub inputs: Vec<u32>,
    /// IDs of nodes that this node feeds.
    pub outputs: Vec<u32>,
}

impl SignalNode {
    /// Creates a new `SignalNode`.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>, format: SignalFormat) -> Self {
        Self {
            id,
            name: name.into(),
            format,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Returns the number of input connections.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the number of output connections.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

/// A directed graph of signal nodes and connections.
#[derive(Debug, Clone, Default)]
pub struct SignalPath {
    /// All nodes in the path.
    pub nodes: Vec<SignalNode>,
    /// Directed edges: (`from_id`, `to_id`) pairs.
    pub connections: Vec<(u32, u32)>,
}

impl SignalPath {
    /// Creates an empty `SignalPath`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node to the path.
    pub fn add_node(&mut self, node: SignalNode) {
        self.nodes.push(node);
    }

    /// Connects two nodes by ID.
    ///
    /// Returns `false` if either ID does not exist.
    pub fn connect(&mut self, from_id: u32, to_id: u32) -> bool {
        let has_from = self.nodes.iter().any(|n| n.id == from_id);
        let has_to = self.nodes.iter().any(|n| n.id == to_id);
        if !has_from || !has_to {
            return false;
        }
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == from_id) {
            if !n.outputs.contains(&to_id) {
                n.outputs.push(to_id);
            }
        }
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == to_id) {
            if !n.inputs.contains(&from_id) {
                n.inputs.push(from_id);
            }
        }
        if !self.connections.contains(&(from_id, to_id)) {
            self.connections.push((from_id, to_id));
        }
        true
    }

    /// Traces a path from `from_id` to `to_id` using breadth-first search.
    ///
    /// Returns `Some(vec_of_ids)` with the node IDs along the path, or `None`
    /// if no path exists.
    #[must_use]
    pub fn trace_path(&self, from_id: u32, to_id: u32) -> Option<Vec<u32>> {
        use std::collections::{HashMap, VecDeque};

        if from_id == to_id {
            return Some(vec![from_id]);
        }

        let mut queue: VecDeque<u32> = VecDeque::new();
        let mut prev: HashMap<u32, u32> = HashMap::new();
        queue.push_back(from_id);

        while let Some(current) = queue.pop_front() {
            // Find neighbours
            let neighbours: Vec<u32> = self
                .connections
                .iter()
                .filter_map(|&(a, b)| if a == current { Some(b) } else { None })
                .collect();

            for next in neighbours {
                if prev.contains_key(&next) {
                    continue;
                }
                prev.insert(next, current);
                if next == to_id {
                    // Reconstruct path
                    let mut path = vec![to_id];
                    let mut cur = to_id;
                    while cur != from_id {
                        cur = *prev.get(&cur)?;
                        path.push(cur);
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(next);
            }
        }
        None
    }

    /// Returns the minimum bandwidth (Gbit/s) along any connection in the graph,
    /// or `None` if there are no connections.
    #[must_use]
    pub fn bandwidth_bottleneck(&self) -> Option<f32> {
        if self.connections.is_empty() {
            return None;
        }
        self.connections
            .iter()
            .filter_map(|&(a, _b)| {
                self.nodes
                    .iter()
                    .find(|n| n.id == a)
                    .map(|n| n.format.max_bandwidth_gbps())
            })
            .reduce(f32::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_format_bandwidth() {
        assert!((SignalFormat::Sdi3G.max_bandwidth_gbps() - 3.0).abs() < f32::EPSILON);
        assert!((SignalFormat::Sdi12G.max_bandwidth_gbps() - 12.0).abs() < f32::EPSILON);
        assert!((SignalFormat::Hdmi20.max_bandwidth_gbps() - 18.0).abs() < f32::EPSILON);
        assert!((SignalFormat::Dp14.max_bandwidth_gbps() - 32.4).abs() < 0.01);
        assert!((SignalFormat::Ip2110.max_bandwidth_gbps() - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_signal_format_supports_4k() {
        assert!(!SignalFormat::Sdi3G.supports_4k());
        assert!(SignalFormat::Sdi12G.supports_4k());
        assert!(SignalFormat::Hdmi20.supports_4k());
        assert!(SignalFormat::Dp14.supports_4k());
        assert!(SignalFormat::Ip2110.supports_4k());
    }

    #[test]
    fn test_signal_node_counts() {
        let mut node = SignalNode::new(1, "Camera", SignalFormat::Sdi12G);
        node.inputs.push(0);
        node.outputs.push(2);
        node.outputs.push(3);
        assert_eq!(node.input_count(), 1);
        assert_eq!(node.output_count(), 2);
    }

    #[test]
    fn test_signal_path_connect_returns_false_for_missing_node() {
        let mut path = SignalPath::new();
        path.add_node(SignalNode::new(1, "A", SignalFormat::Sdi3G));
        assert!(!path.connect(1, 99));
        assert!(!path.connect(99, 1));
    }

    #[test]
    fn test_signal_path_connect_success() {
        let mut path = SignalPath::new();
        path.add_node(SignalNode::new(1, "A", SignalFormat::Sdi3G));
        path.add_node(SignalNode::new(2, "B", SignalFormat::Sdi3G));
        assert!(path.connect(1, 2));
        assert_eq!(path.connections.len(), 1);
    }

    #[test]
    fn test_signal_path_no_duplicate_connections() {
        let mut path = SignalPath::new();
        path.add_node(SignalNode::new(1, "A", SignalFormat::Sdi3G));
        path.add_node(SignalNode::new(2, "B", SignalFormat::Sdi3G));
        path.connect(1, 2);
        path.connect(1, 2); // duplicate
        assert_eq!(path.connections.len(), 1);
    }

    #[test]
    fn test_trace_path_same_node() {
        let mut path = SignalPath::new();
        path.add_node(SignalNode::new(1, "A", SignalFormat::Sdi3G));
        assert_eq!(path.trace_path(1, 1), Some(vec![1]));
    }

    #[test]
    fn test_trace_path_direct() {
        let mut path = SignalPath::new();
        path.add_node(SignalNode::new(1, "A", SignalFormat::Sdi3G));
        path.add_node(SignalNode::new(2, "B", SignalFormat::Sdi3G));
        path.connect(1, 2);
        assert_eq!(path.trace_path(1, 2), Some(vec![1, 2]));
    }

    #[test]
    fn test_trace_path_multi_hop() {
        let mut path = SignalPath::new();
        for id in 1..=4 {
            path.add_node(SignalNode::new(
                id,
                format!("Node{id}"),
                SignalFormat::Sdi12G,
            ));
        }
        path.connect(1, 2);
        path.connect(2, 3);
        path.connect(3, 4);
        let result = path.trace_path(1, 4);
        assert_eq!(result, Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_trace_path_no_path() {
        let mut path = SignalPath::new();
        path.add_node(SignalNode::new(1, "A", SignalFormat::Sdi3G));
        path.add_node(SignalNode::new(2, "B", SignalFormat::Sdi3G));
        // No connection
        assert_eq!(path.trace_path(1, 2), None);
    }

    #[test]
    fn test_bandwidth_bottleneck_empty() {
        let path = SignalPath::new();
        assert!(path.bandwidth_bottleneck().is_none());
    }

    #[test]
    fn test_bandwidth_bottleneck_min_selected() {
        let mut path = SignalPath::new();
        path.add_node(SignalNode::new(1, "A", SignalFormat::Sdi3G)); // 3 Gbps
        path.add_node(SignalNode::new(2, "B", SignalFormat::Sdi12G)); // 12 Gbps
        path.add_node(SignalNode::new(3, "C", SignalFormat::Hdmi20)); // 18 Gbps
        path.connect(1, 2);
        path.connect(2, 3);
        let bottleneck = path.bandwidth_bottleneck().expect("should succeed in test");
        assert!((bottleneck - 3.0).abs() < 0.01);
    }
}
