//! Shot scene graph: nodes, edges, and transitions between shots.
//!
//! Models an edit as a directed graph of `ShotNode` vertices connected by
//! `ShotEdge` arcs. Each edge carries the type of transition (cut, dissolve,
//! fade, or wipe) and the frame number at which the transition occurs.

#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// Transition
// ──────────────────────────────────────────────────────────────────────────────

/// The kind of transition between two shots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotTransition {
    /// Hard cut — zero-duration transition.
    Cut,
    /// Dissolve — one shot fades in while the other fades out.
    Dissolve {
        /// Duration of the overlap in frames.
        frames: u32,
    },
    /// Fade to/from black (or white).
    Fade {
        /// Duration of the fade in frames.
        frames: u32,
    },
    /// Wipe — a geometric boundary sweeps across the frame.
    Wipe {
        /// Duration of the wipe in frames.
        frames: u32,
    },
}

impl ShotTransition {
    /// Number of frames the transition occupies.
    ///
    /// A `Cut` is instantaneous and returns `0`.
    #[must_use]
    pub fn duration_frames(self) -> u32 {
        match self {
            Self::Cut => 0,
            Self::Dissolve { frames } | Self::Fade { frames } | Self::Wipe { frames } => frames,
        }
    }

    /// Returns `true` for a hard cut (zero duration).
    #[must_use]
    pub fn is_cut(self) -> bool {
        matches!(self, Self::Cut)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Nodes and edges
// ──────────────────────────────────────────────────────────────────────────────

/// A single shot represented as a node in the scene graph.
#[derive(Debug, Clone)]
pub struct ShotNode {
    /// Unique identifier for this shot.
    pub id: u64,
    /// First frame of the shot (inclusive).
    pub frame_in: u64,
    /// Last frame of the shot (exclusive).
    pub frame_out: u64,
    /// Camera that captured this shot.
    pub camera_id: u32,
    /// Optional location / set name.
    pub location: Option<String>,
}

impl ShotNode {
    /// Creates a new `ShotNode`.
    #[must_use]
    pub fn new(id: u64, frame_in: u64, frame_out: u64, camera_id: u32) -> Self {
        Self {
            id,
            frame_in,
            frame_out,
            camera_id,
            location: None,
        }
    }

    /// Number of frames in the shot.
    ///
    /// Returns `0` when `frame_out <= frame_in`.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.frame_out.saturating_sub(self.frame_in)
    }

    /// Returns `true` when this shot's frame range overlaps with `other`'s.
    #[must_use]
    pub fn overlaps(&self, other: &ShotNode) -> bool {
        self.frame_in < other.frame_out && other.frame_in < self.frame_out
    }
}

/// A directed edge between two shots in the scene graph.
#[derive(Debug, Clone)]
pub struct ShotEdge {
    /// ID of the outgoing (earlier) shot.
    pub from_id: u64,
    /// ID of the incoming (later) shot.
    pub to_id: u64,
    /// The kind of transition.
    pub transition: ShotTransition,
    /// The edit-line frame number where the transition begins.
    pub cut_frame: u64,
}

impl ShotEdge {
    /// Returns `true` when the transition is a hard cut.
    #[must_use]
    pub fn is_cut(&self) -> bool {
        self.transition.is_cut()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scene graph
// ──────────────────────────────────────────────────────────────────────────────

/// A directed graph of shots and the transitions between them.
#[derive(Debug, Clone, Default)]
pub struct ShotGraph {
    /// All shots (nodes) in the graph.
    pub nodes: Vec<ShotNode>,
    /// All transitions (edges) in the graph.
    pub edges: Vec<ShotEdge>,
}

impl ShotGraph {
    /// Creates an empty `ShotGraph`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a shot node to the graph.
    ///
    /// If a node with the same `id` already exists it is replaced.
    pub fn add_shot(&mut self, node: ShotNode) {
        if let Some(existing) = self.nodes.iter_mut().find(|n| n.id == node.id) {
            *existing = node;
        } else {
            self.nodes.push(node);
        }
    }

    /// Adds a transition edge to the graph.
    pub fn add_transition(&mut self, edge: ShotEdge) {
        self.edges.push(edge);
    }

    /// Returns all shots captured by `camera_id`.
    #[must_use]
    pub fn shots_for_camera(&self, camera_id: u32) -> Vec<&ShotNode> {
        self.nodes
            .iter()
            .filter(|n| n.camera_id == camera_id)
            .collect()
    }

    /// Returns all edges that originate from `shot_id`.
    #[must_use]
    pub fn transitions_from(&self, shot_id: u64) -> Vec<&ShotEdge> {
        self.edges.iter().filter(|e| e.from_id == shot_id).collect()
    }

    /// Returns all edges that lead into `shot_id`.
    #[must_use]
    pub fn transitions_to(&self, shot_id: u64) -> Vec<&ShotEdge> {
        self.edges.iter().filter(|e| e.to_id == shot_id).collect()
    }

    /// Total duration in frames, defined as the maximum `frame_out` across all
    /// nodes.
    ///
    /// Returns `0` when the graph has no nodes.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.nodes.iter().map(|n| n.frame_out).max().unwrap_or(0)
    }

    /// Number of shots in the graph.
    #[must_use]
    pub fn shot_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of transitions in the graph.
    #[must_use]
    pub fn transition_count(&self) -> usize {
        self.edges.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, frame_in: u64, frame_out: u64, cam: u32) -> ShotNode {
        ShotNode::new(id, frame_in, frame_out, cam)
    }

    fn cut_edge(from: u64, to: u64, cut_frame: u64) -> ShotEdge {
        ShotEdge {
            from_id: from,
            to_id: to,
            transition: ShotTransition::Cut,
            cut_frame,
        }
    }

    // ShotTransition
    #[test]
    fn test_cut_duration_is_zero() {
        assert_eq!(ShotTransition::Cut.duration_frames(), 0);
    }

    #[test]
    fn test_dissolve_duration() {
        assert_eq!(
            ShotTransition::Dissolve { frames: 12 }.duration_frames(),
            12
        );
    }

    #[test]
    fn test_fade_duration() {
        assert_eq!(ShotTransition::Fade { frames: 25 }.duration_frames(), 25);
    }

    #[test]
    fn test_wipe_duration() {
        assert_eq!(ShotTransition::Wipe { frames: 6 }.duration_frames(), 6);
    }

    #[test]
    fn test_is_cut_true() {
        assert!(ShotTransition::Cut.is_cut());
    }

    #[test]
    fn test_is_cut_false_for_dissolve() {
        assert!(!ShotTransition::Dissolve { frames: 10 }.is_cut());
    }

    // ShotNode
    #[test]
    fn test_shot_node_duration() {
        let n = node(1, 100, 200, 0);
        assert_eq!(n.duration_frames(), 100);
    }

    #[test]
    fn test_shot_node_duration_zero_when_inverted() {
        let n = node(1, 200, 100, 0);
        assert_eq!(n.duration_frames(), 0);
    }

    #[test]
    fn test_shot_node_overlaps_true() {
        let a = node(1, 0, 100, 0);
        let b = node(2, 50, 150, 0);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_shot_node_overlaps_false() {
        let a = node(1, 0, 100, 0);
        let b = node(2, 100, 200, 0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_shot_edge_is_cut() {
        let e = cut_edge(1, 2, 100);
        assert!(e.is_cut());
    }

    #[test]
    fn test_shot_edge_not_cut_for_dissolve() {
        let e = ShotEdge {
            from_id: 1,
            to_id: 2,
            transition: ShotTransition::Dissolve { frames: 12 },
            cut_frame: 100,
        };
        assert!(!e.is_cut());
    }

    // ShotGraph
    #[test]
    fn test_add_shot_and_shot_count() {
        let mut g = ShotGraph::new();
        g.add_shot(node(1, 0, 50, 0));
        g.add_shot(node(2, 50, 100, 1));
        assert_eq!(g.shot_count(), 2);
    }

    #[test]
    fn test_add_shot_replaces_existing() {
        let mut g = ShotGraph::new();
        g.add_shot(node(1, 0, 50, 0));
        g.add_shot(node(1, 0, 75, 1)); // replace
        assert_eq!(g.shot_count(), 1);
        assert_eq!(g.nodes[0].camera_id, 1);
    }

    #[test]
    fn test_shots_for_camera() {
        let mut g = ShotGraph::new();
        g.add_shot(node(1, 0, 50, 0));
        g.add_shot(node(2, 50, 100, 1));
        g.add_shot(node(3, 100, 150, 0));
        let cam0 = g.shots_for_camera(0);
        assert_eq!(cam0.len(), 2);
    }

    #[test]
    fn test_transitions_from() {
        let mut g = ShotGraph::new();
        g.add_shot(node(1, 0, 50, 0));
        g.add_shot(node(2, 50, 100, 0));
        g.add_transition(cut_edge(1, 2, 50));
        assert_eq!(g.transitions_from(1).len(), 1);
        assert_eq!(g.transitions_from(2).len(), 0);
    }

    #[test]
    fn test_transitions_to() {
        let mut g = ShotGraph::new();
        g.add_shot(node(1, 0, 50, 0));
        g.add_shot(node(2, 50, 100, 0));
        g.add_transition(cut_edge(1, 2, 50));
        assert_eq!(g.transitions_to(2).len(), 1);
        assert_eq!(g.transitions_to(1).len(), 0);
    }

    #[test]
    fn test_total_duration_frames() {
        let mut g = ShotGraph::new();
        g.add_shot(node(1, 0, 100, 0));
        g.add_shot(node(2, 100, 300, 0));
        assert_eq!(g.total_duration_frames(), 300);
    }

    #[test]
    fn test_total_duration_empty_graph() {
        let g = ShotGraph::new();
        assert_eq!(g.total_duration_frames(), 0);
    }

    #[test]
    fn test_transition_count() {
        let mut g = ShotGraph::new();
        g.add_shot(node(1, 0, 50, 0));
        g.add_shot(node(2, 50, 100, 0));
        g.add_shot(node(3, 100, 150, 0));
        g.add_transition(cut_edge(1, 2, 50));
        g.add_transition(cut_edge(2, 3, 100));
        assert_eq!(g.transition_count(), 2);
    }
}
