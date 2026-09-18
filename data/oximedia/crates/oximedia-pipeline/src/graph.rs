//! Pipeline graph — a directed acyclic graph (DAG) of media processing nodes.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::node::{NodeId, NodeSpec, NodeType};
use crate::PipelineError;

// ── DFS colour state for cycle detection ─────────────────────────────────────

/// DFS visit colour following Tarjan's three-colour DFS convention.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Colour {
    /// Not yet visited.
    White,
    /// Currently on the DFS stack (in-progress).
    Grey,
    /// Fully explored.
    Black,
}

// ── Edge ─────────────────────────────────────────────────────────────────────

/// A directed connection between two pads in the pipeline graph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Edge {
    /// Source node identifier.
    pub from_node: NodeId,
    /// Name of the output pad on the source node.
    pub from_pad: String,
    /// Destination node identifier.
    pub to_node: NodeId,
    /// Name of the input pad on the destination node.
    pub to_pad: String,
}

// ── PipelineGraph ────────────────────────────────────────────────────────────

/// A directed acyclic graph of media processing nodes connected by edges.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineGraph {
    /// All nodes in the graph, keyed by their unique identifier.
    pub nodes: HashMap<NodeId, NodeSpec>,
    /// All edges (connections) in the graph.
    pub edges: Vec<Edge>,
}

impl PipelineGraph {
    /// Create an empty pipeline graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph and return its identifier.
    pub fn add_node(&mut self, spec: NodeSpec) -> NodeId {
        let id = spec.id;
        self.nodes.insert(id, spec);
        id
    }

    /// Connect an output pad of one node to an input pad of another.
    ///
    /// Returns an error if either node or pad does not exist, or if the streams
    /// are incompatible (video connected to audio, etc.).
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_pad: &str,
        to_node: NodeId,
        to_pad: &str,
    ) -> Result<(), PipelineError> {
        // Validate from_node exists
        let from_spec = self
            .nodes
            .get(&from_node)
            .ok_or_else(|| PipelineError::NodeNotFound(from_node.to_string()))?;

        // Validate from_pad exists
        let from_pad_spec = from_spec
            .output_pads
            .iter()
            .find(|(name, _)| name == from_pad)
            .ok_or_else(|| PipelineError::PadNotFound {
                node: from_spec.name.clone(),
                pad: from_pad.to_string(),
            })?;
        let from_stream = from_pad_spec.1.clone();

        // Validate to_node exists
        let to_spec = self
            .nodes
            .get(&to_node)
            .ok_or_else(|| PipelineError::NodeNotFound(to_node.to_string()))?;

        // Validate to_pad exists
        let to_pad_spec = to_spec
            .input_pads
            .iter()
            .find(|(name, _)| name == to_pad)
            .ok_or_else(|| PipelineError::PadNotFound {
                node: to_spec.name.clone(),
                pad: to_pad.to_string(),
            })?;
        let to_stream = &to_pad_spec.1;

        // Check stream kind compatibility
        if !from_stream.kind_compatible(to_stream) {
            return Err(PipelineError::IncompatibleStreams);
        }

        self.edges.push(Edge {
            from_node,
            from_pad: from_pad.to_string(),
            to_node,
            to_pad: to_pad.to_string(),
        });

        Ok(())
    }

    /// Remove a node and all edges connected to it.
    pub fn remove_node(&mut self, id: &NodeId) -> Result<(), PipelineError> {
        if self.nodes.remove(id).is_none() {
            return Err(PipelineError::NodeNotFound(id.to_string()));
        }
        self.edges
            .retain(|e| e.from_node != *id && e.to_node != *id);
        Ok(())
    }

    /// Validate the graph and return a list of all detected errors.
    ///
    /// Checks for: cycles (with full path reporting), dangling edges, and
    /// unconnected required pads.
    pub fn validate(&self) -> Vec<PipelineError> {
        let mut errors = Vec::new();

        // Check for dangling edges (referencing non-existent nodes)
        for edge in &self.edges {
            if !self.nodes.contains_key(&edge.from_node) {
                errors.push(PipelineError::NodeNotFound(edge.from_node.to_string()));
            }
            if !self.nodes.contains_key(&edge.to_node) {
                errors.push(PipelineError::NodeNotFound(edge.to_node.to_string()));
            }
        }

        // Check that all input pads of non-source nodes are connected
        for (id, spec) in &self.nodes {
            match &spec.node_type {
                NodeType::Source(_) => {
                    // Sources have no input pads — nothing to check
                }
                _ => {
                    for (pad_name, _) in &spec.input_pads {
                        let connected = self
                            .edges
                            .iter()
                            .any(|e| e.to_node == *id && e.to_pad == *pad_name);
                        if !connected {
                            errors.push(PipelineError::PadNotFound {
                                node: spec.name.clone(),
                                pad: pad_name.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Check for cycles with detailed path reporting via DFS
        if let Err(e) = self.detect_cycle() {
            errors.push(e);
        }

        errors
    }

    /// Detect any cycle in the graph using a DFS three-colour algorithm.
    ///
    /// Returns `Ok(())` if the graph is acyclic, or `Err(PipelineError::CycleDetected)`
    /// with a full ordered cycle path when a back-edge is found.
    pub fn detect_cycle(&self) -> Result<(), PipelineError> {
        // Build adjacency list (NodeId → Vec<NodeId>) from the edge list.
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for id in self.nodes.keys() {
            adj.entry(*id).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.from_node).or_default().push(edge.to_node);
        }

        let mut colour: HashMap<NodeId, Colour> =
            self.nodes.keys().map(|id| (*id, Colour::White)).collect();

        // Iterative DFS using an explicit stack to avoid recursion limits on
        // large graphs.  Each stack frame stores (node, iterator-index).
        let all_ids: Vec<NodeId> = self.nodes.keys().copied().collect();

        for &start in &all_ids {
            if colour.get(&start) != Some(&Colour::White) {
                continue;
            }

            // path_stack: ordered list of NodeIds currently in DFS stack (grey).
            let mut path_stack: Vec<NodeId> = Vec::new();
            // dfs_stack: (node_id, next_child_index).
            let mut dfs_stack: Vec<(NodeId, usize)> = vec![(start, 0)];
            *colour.entry(start).or_insert(Colour::White) = Colour::Grey;
            path_stack.push(start);

            while let Some((current, child_idx)) = dfs_stack.last_mut() {
                let current = *current;
                let neighbours = adj.get(&current).map(|v| v.as_slice()).unwrap_or(&[]);

                if *child_idx >= neighbours.len() {
                    // All children explored: mark black and pop.
                    *colour.entry(current).or_insert(Colour::Black) = Colour::Black;
                    dfs_stack.pop();
                    path_stack.pop();
                } else {
                    let neighbour = neighbours[*child_idx];
                    *child_idx += 1;

                    match colour.get(&neighbour).copied().unwrap_or(Colour::White) {
                        Colour::Grey => {
                            // Back-edge found: build cycle path.
                            // Find where `neighbour` first appears in path_stack.
                            let cycle_start_pos = path_stack
                                .iter()
                                .position(|&id| id == neighbour)
                                .unwrap_or(0);
                            let cycle_ids = &path_stack[cycle_start_pos..];

                            // Resolve node names (fall back to UUID string).
                            let mut cycle_path: Vec<String> = cycle_ids
                                .iter()
                                .map(|id| {
                                    self.nodes
                                        .get(id)
                                        .map(|s| s.name.clone())
                                        .unwrap_or_else(|| id.to_string())
                                })
                                .collect();
                            // Repeat the first element to close the cycle loop.
                            if let Some(first) = cycle_path.first().cloned() {
                                cycle_path.push(first);
                            }

                            return Err(PipelineError::CycleDetected { path: cycle_path });
                        }
                        Colour::White => {
                            *colour.entry(neighbour).or_insert(Colour::White) = Colour::Grey;
                            path_stack.push(neighbour);
                            dfs_stack.push((neighbour, 0));
                        }
                        Colour::Black => {
                            // Already fully explored; no cycle through this edge.
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Topological sort using Kahn's algorithm (BFS over in-degree).
    ///
    /// Returns nodes in dependency order, or `CycleDetected` (with full path)
    /// if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, PipelineError> {
        // Fast-path: detect and report cycle with full path first.
        self.detect_cycle()?;

        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();

        for &id in &node_ids {
            in_degree.insert(id, 0);
        }

        for edge in &self.edges {
            if let Some(deg) = in_degree.get_mut(&edge.to_node) {
                *deg += 1;
            }
        }

        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for (&id, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(id);
            }
        }

        let mut sorted = Vec::with_capacity(node_ids.len());

        while let Some(node) = queue.pop_front() {
            sorted.push(node);
            for edge in &self.edges {
                if edge.from_node == node {
                    if let Some(deg) = in_degree.get_mut(&edge.to_node) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(edge.to_node);
                        }
                    }
                }
            }
        }

        // This branch should not be reachable given detect_cycle passed,
        // but kept as a safety net returning a generic cycle error.
        if sorted.len() != node_ids.len() {
            return Err(PipelineError::CycleDetected {
                path: vec!["<unknown cycle>".to_string()],
            });
        }

        Ok(sorted)
    }

    /// Return all source nodes (nodes with no incoming edges).
    pub fn source_nodes(&self) -> Vec<NodeId> {
        let has_incoming: HashSet<NodeId> = self.edges.iter().map(|e| e.to_node).collect();
        self.nodes
            .keys()
            .filter(|id| !has_incoming.contains(id))
            .copied()
            .collect()
    }

    /// Return all sink nodes (nodes with no outgoing edges).
    pub fn sink_nodes(&self) -> Vec<NodeId> {
        let has_outgoing: HashSet<NodeId> = self.edges.iter().map(|e| e.from_node).collect();
        self.nodes
            .keys()
            .filter(|id| !has_outgoing.contains(id))
            .copied()
            .collect()
    }

    /// Return the direct upstream predecessors of a node.
    pub fn predecessors(&self, node: &NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.to_node == *node)
            .map(|e| e.from_node)
            .collect()
    }

    /// Return the direct downstream successors of a node.
    pub fn successors(&self, node: &NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.from_node == *node)
            .map(|e| e.to_node)
            .collect()
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Merge another `PipelineGraph` into this one.
    ///
    /// All nodes and edges from `other` are incorporated. Nodes that already
    /// exist in `self` (same `NodeId`) are considered **shared** — they are
    /// kept as-is and not duplicated. Edges from `other` are added only when
    /// they do not already appear in `self`.
    ///
    /// Returns a `Vec` of all `NodeId`s that were newly added (i.e. were in
    /// `other` but not previously in `self`).
    pub fn merge_graph(&mut self, other: &PipelineGraph) -> Vec<NodeId> {
        let mut added = Vec::new();

        // Add nodes that don't yet exist
        for (id, spec) in &other.nodes {
            if !self.nodes.contains_key(id) {
                self.nodes.insert(*id, spec.clone());
                added.push(*id);
            }
        }

        // Add edges that don't already exist
        for edge in &other.edges {
            let already_exists = self.edges.iter().any(|e| {
                e.from_node == edge.from_node
                    && e.from_pad == edge.from_pad
                    && e.to_node == edge.to_node
                    && e.to_pad == edge.to_pad
            });
            if !already_exists {
                self.edges.push(edge.clone());
            }
        }

        added
    }

    /// Merge another graph and validate the result.
    ///
    /// Like [`merge_graph`](Self::merge_graph), but also runs
    /// [`validate`](Self::validate) on the combined graph. If validation
    /// produces errors, they are returned and the merge is **still applied**
    /// (the caller can inspect the graph and decide whether to undo).
    pub fn merge_validated(
        &mut self,
        other: &PipelineGraph,
    ) -> Result<Vec<NodeId>, Vec<PipelineError>> {
        let added = self.merge_graph(other);
        let errors = self.validate();
        if errors.is_empty() {
            Ok(added)
        } else {
            Err(errors)
        }
    }

    /// Return the set of `NodeId`s shared between this graph and another.
    pub fn shared_nodes(&self, other: &PipelineGraph) -> HashSet<NodeId> {
        let self_ids: HashSet<NodeId> = self.nodes.keys().copied().collect();
        let other_ids: HashSet<NodeId> = other.nodes.keys().copied().collect();
        self_ids.intersection(&other_ids).copied().collect()
    }
}

impl Default for PipelineGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::*;

    fn video_spec() -> StreamSpec {
        StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
    }

    fn audio_spec() -> StreamSpec {
        StreamSpec::audio(FrameFormat::S16Interleaved, 48000, 2)
    }

    #[test]
    fn new_graph_is_empty() {
        let g = PipelineGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn add_node_returns_id() {
        let mut g = PipelineGraph::new();
        let spec = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let id = g.add_node(spec);
        assert_eq!(g.node_count(), 1);
        assert!(g.nodes.contains_key(&id));
    }

    #[test]
    fn connect_valid_pads() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let sink_id = g.add_node(sink);
        let result = g.connect(src_id, "default", sink_id, "default");
        assert!(result.is_ok());
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn connect_incompatible_streams() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, audio_spec());
        let src_id = g.add_node(src);
        let sink_id = g.add_node(sink);
        let result = g.connect(src_id, "default", sink_id, "default");
        assert!(result.is_err());
    }

    #[test]
    fn connect_missing_from_node() {
        let mut g = PipelineGraph::new();
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let sink_id = g.add_node(sink);
        let fake_id = NodeId::new();
        let result = g.connect(fake_id, "default", sink_id, "default");
        assert!(result.is_err());
    }

    #[test]
    fn connect_missing_to_node() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let src_id = g.add_node(src);
        let fake_id = NodeId::new();
        let result = g.connect(src_id, "default", fake_id, "default");
        assert!(result.is_err());
    }

    #[test]
    fn connect_missing_pad() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let sink_id = g.add_node(sink);
        let result = g.connect(src_id, "nonexistent", sink_id, "default");
        assert!(result.is_err());
    }

    #[test]
    fn remove_node_removes_edges() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            video_spec(),
            video_spec(),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let filt_id = g.add_node(filt);
        let sink_id = g.add_node(sink);
        let _ = g.connect(src_id, "default", filt_id, "default");
        let _ = g.connect(filt_id, "default", sink_id, "default");
        assert_eq!(g.edge_count(), 2);
        let result = g.remove_node(&filt_id);
        assert!(result.is_ok());
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn remove_node_not_found() {
        let mut g = PipelineGraph::new();
        let fake = NodeId::new();
        let result = g.remove_node(&fake);
        assert!(result.is_err());
    }

    #[test]
    fn topological_sort_linear() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            video_spec(),
            video_spec(),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let filt_id = g.add_node(filt);
        let sink_id = g.add_node(sink);
        let _ = g.connect(src_id, "default", filt_id, "default");
        let _ = g.connect(filt_id, "default", sink_id, "default");

        let sorted = g.topological_sort().expect("no cycle");
        // src must come before filt, filt before sink
        let pos_src = sorted.iter().position(|x| *x == src_id).expect("src");
        let pos_filt = sorted.iter().position(|x| *x == filt_id).expect("filt");
        let pos_sink = sorted.iter().position(|x| *x == sink_id).expect("sink");
        assert!(pos_src < pos_filt);
        assert!(pos_filt < pos_sink);
    }

    #[test]
    fn topological_sort_empty() {
        let g = PipelineGraph::new();
        let sorted = g.topological_sort().expect("no cycle");
        assert!(sorted.is_empty());
    }

    #[test]
    fn source_nodes_and_sink_nodes() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let sink_id = g.add_node(sink);
        let _ = g.connect(src_id, "default", sink_id, "default");

        let sources = g.source_nodes();
        let sinks = g.sink_nodes();
        assert_eq!(sources.len(), 1);
        assert_eq!(sinks.len(), 1);
        assert!(sources.contains(&src_id));
        assert!(sinks.contains(&sink_id));
    }

    #[test]
    fn predecessors_and_successors() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            video_spec(),
            video_spec(),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let filt_id = g.add_node(filt);
        let sink_id = g.add_node(sink);
        let _ = g.connect(src_id, "default", filt_id, "default");
        let _ = g.connect(filt_id, "default", sink_id, "default");

        assert_eq!(g.predecessors(&filt_id), vec![src_id]);
        assert_eq!(g.successors(&filt_id), vec![sink_id]);
        assert!(g.predecessors(&src_id).is_empty());
        assert!(g.successors(&sink_id).is_empty());
    }

    #[test]
    fn validate_connected_graph() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let sink_id = g.add_node(sink);
        let _ = g.connect(src_id, "default", sink_id, "default");
        let errors = g.validate();
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn validate_unconnected_sink() {
        let mut g = PipelineGraph::new();
        let _src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let _sink_id = g.add_node(sink);
        let errors = g.validate();
        // Sink's input pad is not connected
        assert!(!errors.is_empty());
    }

    #[test]
    fn node_count_and_edge_count() {
        let mut g = PipelineGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let sink_id = g.add_node(sink);
        assert_eq!(g.node_count(), 2);
        let _ = g.connect(src_id, "default", sink_id, "default");
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn default_creates_empty_graph() {
        let g = PipelineGraph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn multiple_sources_fan_in() {
        let mut g = PipelineGraph::new();
        let vs = video_spec();
        let src1 = NodeSpec::source("src1", SourceConfig::File("a.mp4".into()), vs.clone());
        let src2 = NodeSpec::source("src2", SourceConfig::File("b.mp4".into()), vs.clone());
        // Merge has two input pads
        let merge = NodeSpec::new(
            "merge",
            NodeType::Merge,
            vec![
                ("input0".to_string(), vs.clone()),
                ("input1".to_string(), vs.clone()),
            ],
            vec![("default".to_string(), vs.clone())],
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs);
        let s1 = g.add_node(src1);
        let s2 = g.add_node(src2);
        let m = g.add_node(merge);
        let sk = g.add_node(sink);
        let _ = g.connect(s1, "default", m, "input0");
        let _ = g.connect(s2, "default", m, "input1");
        let _ = g.connect(m, "default", sk, "default");

        let sorted = g.topological_sort().expect("no cycle");
        let pos_m = sorted.iter().position(|x| *x == m).expect("merge");
        let pos_s1 = sorted.iter().position(|x| *x == s1).expect("src1");
        let pos_s2 = sorted.iter().position(|x| *x == s2).expect("src2");
        assert!(pos_s1 < pos_m);
        assert!(pos_s2 < pos_m);
    }

    #[test]
    fn fan_out_split() {
        let mut g = PipelineGraph::new();
        let vs = video_spec();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), vs.clone());
        let split = NodeSpec::new(
            "split",
            NodeType::Split,
            vec![("default".to_string(), vs.clone())],
            vec![
                ("out0".to_string(), vs.clone()),
                ("out1".to_string(), vs.clone()),
            ],
        );
        let sink1 = NodeSpec::sink("sink1", SinkConfig::File("o1.mp4".into()), vs.clone());
        let sink2 = NodeSpec::sink("sink2", SinkConfig::File("o2.mp4".into()), vs);
        let s = g.add_node(src);
        let sp = g.add_node(split);
        let sk1 = g.add_node(sink1);
        let sk2 = g.add_node(sink2);
        let _ = g.connect(s, "default", sp, "default");
        let _ = g.connect(sp, "out0", sk1, "default");
        let _ = g.connect(sp, "out1", sk2, "default");

        assert_eq!(g.successors(&sp).len(), 2);
        let sinks = g.sink_nodes();
        assert_eq!(sinks.len(), 2);
    }

    #[test]
    fn edge_equality() {
        let a = NodeId::new();
        let b = NodeId::new();
        let e1 = Edge {
            from_node: a,
            from_pad: "default".to_string(),
            to_node: b,
            to_pad: "default".to_string(),
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn validate_source_needs_no_input() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let _id = g.add_node(src);
        // Source alone is valid (no unconnected input pads)
        let errors = g.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn single_node_is_both_source_and_sink() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let id = g.add_node(src);
        assert!(g.source_nodes().contains(&id));
        assert!(g.sink_nodes().contains(&id));
    }

    // ── Merge tests ─────────────────────────────────────────────────────────

    #[test]
    fn merge_disjoint_graphs() {
        let mut g1 = PipelineGraph::new();
        let src1 = NodeSpec::source("src1", SourceConfig::File("a.mp4".into()), video_spec());
        let sink1 = NodeSpec::sink("sink1", SinkConfig::Null, video_spec());
        let s1 = g1.add_node(src1);
        let sk1 = g1.add_node(sink1);
        let _ = g1.connect(s1, "default", sk1, "default");

        let mut g2 = PipelineGraph::new();
        let src2 = NodeSpec::source("src2", SourceConfig::File("b.mp4".into()), video_spec());
        let sink2 = NodeSpec::sink("sink2", SinkConfig::Null, video_spec());
        let s2 = g2.add_node(src2);
        let sk2 = g2.add_node(sink2);
        let _ = g2.connect(s2, "default", sk2, "default");

        let added = g1.merge_graph(&g2);
        assert_eq!(added.len(), 2);
        assert_eq!(g1.node_count(), 4);
        assert_eq!(g1.edge_count(), 2);
    }

    #[test]
    fn merge_with_shared_node() {
        let mut g1 = PipelineGraph::new();
        let src = NodeSpec::source(
            "shared_src",
            SourceConfig::File("a.mp4".into()),
            video_spec(),
        );
        let src_id = g1.add_node(src.clone());
        let sink1 = NodeSpec::sink("sink1", SinkConfig::Null, video_spec());
        let sk1 = g1.add_node(sink1);
        let _ = g1.connect(src_id, "default", sk1, "default");

        // Build g2 with the same shared source node
        let mut g2 = PipelineGraph::new();
        g2.nodes.insert(src_id, src);
        let sink2 = NodeSpec::sink("sink2", SinkConfig::File("out.mp4".into()), video_spec());
        let sk2 = g2.add_node(sink2);
        let _ = g2.connect(src_id, "default", sk2, "default");

        let shared = g1.shared_nodes(&g2);
        assert_eq!(shared.len(), 1);
        assert!(shared.contains(&src_id));

        let added = g1.merge_graph(&g2);
        // Only sink2 is new; shared_src already exists
        assert_eq!(added.len(), 1);
        assert_eq!(g1.node_count(), 3);
        assert_eq!(g1.edge_count(), 2);
    }

    #[test]
    fn merge_idempotent() {
        let mut g1 = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let s = g1.add_node(src);
        let sk = g1.add_node(sink);
        let _ = g1.connect(s, "default", sk, "default");

        let g2 = g1.clone();
        let added = g1.merge_graph(&g2);
        assert!(added.is_empty());
        assert_eq!(g1.node_count(), 2);
        assert_eq!(g1.edge_count(), 1);
    }

    #[test]
    fn shared_nodes_empty_when_disjoint() {
        let mut g1 = PipelineGraph::new();
        let _ = g1.add_node(NodeSpec::source(
            "src",
            SourceConfig::File("a.mp4".into()),
            video_spec(),
        ));
        let mut g2 = PipelineGraph::new();
        let _ = g2.add_node(NodeSpec::source(
            "src2",
            SourceConfig::File("b.mp4".into()),
            video_spec(),
        ));
        let shared = g1.shared_nodes(&g2);
        assert!(shared.is_empty());
    }

    #[test]
    fn merge_validated_succeeds_on_valid_graphs() {
        let mut g1 = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let src_id = g1.add_node(src.clone());

        let mut g2 = PipelineGraph::new();
        g2.nodes.insert(src_id, src);
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let sk_id = g2.add_node(sink);
        let _ = g2.connect(src_id, "default", sk_id, "default");

        let result = g1.merge_validated(&g2);
        assert!(result.is_ok());
        let added = result.expect("valid merge");
        assert_eq!(added.len(), 1); // only sink was new
        assert_eq!(g1.node_count(), 2);
        assert_eq!(g1.edge_count(), 1);
    }

    #[test]
    fn merge_validated_detects_invalid_result() {
        // Create g1 with just a sink (no source connected -> invalid)
        let mut g1 = PipelineGraph::new();
        let sink = NodeSpec::sink("sink1", SinkConfig::Null, video_spec());
        let _sk = g1.add_node(sink);

        // g2 also has just a disconnected sink
        let mut g2 = PipelineGraph::new();
        let sink2 = NodeSpec::sink("sink2", SinkConfig::Null, video_spec());
        let _sk2 = g2.add_node(sink2);

        let result = g1.merge_validated(&g2);
        assert!(result.is_err());
    }

    #[test]
    fn merge_preserves_edges_from_both_graphs() {
        let mut g1 = PipelineGraph::new();
        let vs = video_spec();
        let src1 = NodeSpec::source("src1", SourceConfig::File("a.mp4".into()), vs.clone());
        let sink1 = NodeSpec::sink("sink1", SinkConfig::Null, vs.clone());
        let s1 = g1.add_node(src1);
        let sk1 = g1.add_node(sink1);
        let _ = g1.connect(s1, "default", sk1, "default");

        let mut g2 = PipelineGraph::new();
        let src2 = NodeSpec::source("src2", SourceConfig::File("b.mp4".into()), vs.clone());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 640,
                height: 480,
            },
            vs.clone(),
            vs.clone(),
        );
        let sink2 = NodeSpec::sink("sink2", SinkConfig::Null, vs);
        let s2 = g2.add_node(src2);
        let f = g2.add_node(filt);
        let sk2 = g2.add_node(sink2);
        let _ = g2.connect(s2, "default", f, "default");
        let _ = g2.connect(f, "default", sk2, "default");

        g1.merge_graph(&g2);
        assert_eq!(g1.node_count(), 5);
        assert_eq!(g1.edge_count(), 3); // 1 from g1 + 2 from g2
                                        // Both subgraphs should topo-sort independently
        let sorted = g1.topological_sort();
        assert!(sorted.is_ok());
    }

    // ── Serde round-trip tests ────────────────────────────────────────────

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        #[test]
        fn pipeline_graph_roundtrip() {
            let mut g = PipelineGraph::new();
            let vs = video_spec();
            let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), vs.clone());
            let sink = NodeSpec::sink("sink", SinkConfig::Null, vs);
            let s = g.add_node(src);
            let sk = g.add_node(sink);
            let _ = g.connect(s, "default", sk, "default");

            let json = serde_json::to_string(&g).expect("serialize");
            let g2: PipelineGraph = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(g2.node_count(), 2);
            assert_eq!(g2.edge_count(), 1);
        }

        #[test]
        fn edge_roundtrip() {
            let a = NodeId::new();
            let b = NodeId::new();
            let edge = Edge {
                from_node: a,
                from_pad: "default".to_string(),
                to_node: b,
                to_pad: "default".to_string(),
            };
            let json = serde_json::to_string(&edge).expect("serialize");
            let edge2: Edge = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(edge, edge2);
        }

        #[test]
        fn filter_config_parametric_roundtrip() {
            let base = FilterConfig::Scale {
                width: 1280,
                height: 720,
            };
            let param = base
                .with_property("quality", "high")
                .with_property("threads", "4");

            let json = serde_json::to_string(&param).expect("serialize");
            let param2: FilterConfig = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(param2.get_property("quality"), Some("high"));
            assert_eq!(param2.get_property("threads"), Some("4"));
            assert_eq!(param2.cost_estimate(), 8);
        }

        #[test]
        fn complex_graph_roundtrip() {
            let mut g = PipelineGraph::new();
            let vs = video_spec();
            let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), vs.clone());
            let filt = NodeSpec::filter(
                "scale",
                FilterConfig::Scale {
                    width: 1280,
                    height: 720,
                },
                vs.clone(),
                vs.clone(),
            );
            let sink = NodeSpec::sink("sink", SinkConfig::Null, vs);
            let s = g.add_node(src);
            let f = g.add_node(filt);
            let sk = g.add_node(sink);
            let _ = g.connect(s, "default", f, "default");
            let _ = g.connect(f, "default", sk, "default");

            let json = serde_json::to_string(&g).expect("serialize");
            let g2: PipelineGraph = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(g2.node_count(), 3);
            assert_eq!(g2.edge_count(), 2);

            let sorted = g2.topological_sort().expect("no cycle");
            assert_eq!(sorted.len(), 3);
        }
    }
}
