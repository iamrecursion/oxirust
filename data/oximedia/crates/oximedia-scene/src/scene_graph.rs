//! Scene graph structure for hierarchical narrative representation.
//!
//! Provides a tree-based scene graph where nodes represent shots, acts,
//! sequences, episodes, or seasons, and edges represent parent-child
//! containment relationships.

/// The type of a scene graph node in the narrative hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneNodeType {
    /// An individual shot (smallest unit).
    Shot,
    /// A narrative act (collection of shots).
    Act,
    /// A sequence of scenes forming a unit.
    Sequence,
    /// A full episode.
    Episode,
    /// A season (collection of episodes).
    Season,
}

impl SceneNodeType {
    /// Hierarchy level: lower numbers are more granular.
    ///
    /// `Shot=0`, `Act=1`, `Sequence=2`, `Episode=3`, `Season=4`
    #[must_use]
    pub fn level(&self) -> u8 {
        match self {
            SceneNodeType::Shot => 0,
            SceneNodeType::Act => 1,
            SceneNodeType::Sequence => 2,
            SceneNodeType::Episode => 3,
            SceneNodeType::Season => 4,
        }
    }
}

/// A single node in the scene graph.
#[derive(Debug, Clone)]
pub struct SceneNode {
    /// Unique identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Type of this node.
    pub node_type: SceneNodeType,
    /// IDs of child nodes.
    pub children: Vec<u64>,
    /// Start frame index (inclusive).
    pub start_frame: u64,
    /// End frame index (exclusive).
    pub end_frame: u64,
}

impl SceneNode {
    /// Duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` when frame `f` falls within `[start_frame, end_frame)`.
    #[must_use]
    pub fn contains_frame(&self, f: u64) -> bool {
        f >= self.start_frame && f < self.end_frame
    }
}

/// A directed scene graph with parent-child relationships.
#[derive(Debug, Default)]
pub struct SceneGraph {
    nodes: Vec<SceneNode>,
    next_id: u64,
}

impl SceneGraph {
    /// Create a new empty `SceneGraph`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node and return its assigned ID.
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        node_type: SceneNodeType,
        start_frame: u64,
        end_frame: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(SceneNode {
            id,
            name: name.into(),
            node_type,
            children: Vec::new(),
            start_frame,
            end_frame,
        });
        id
    }

    /// Register `child_id` as a child of `parent_id`.
    ///
    /// Returns `false` when either ID does not exist.
    pub fn add_child(&mut self, parent_id: u64, child_id: u64) -> bool {
        // Verify child exists
        if !self.nodes.iter().any(|n| n.id == child_id) {
            return false;
        }
        if let Some(parent) = self.nodes.iter_mut().find(|n| n.id == parent_id) {
            if !parent.children.contains(&child_id) {
                parent.children.push(child_id);
            }
            true
        } else {
            false
        }
    }

    /// Find a node by ID.
    #[must_use]
    pub fn find(&self, id: u64) -> Option<&SceneNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// All nodes that are not the child of any other node.
    #[must_use]
    pub fn root_nodes(&self) -> Vec<&SceneNode> {
        let child_ids: std::collections::HashSet<u64> = self
            .nodes
            .iter()
            .flat_map(|n| n.children.iter().copied())
            .collect();
        self.nodes
            .iter()
            .filter(|n| !child_ids.contains(&n.id))
            .collect()
    }
}

/// Depth-first walker over a `SceneGraph` rooted at a given node.
pub struct SceneGraphWalker<'a> {
    graph: &'a SceneGraph,
    stack: Vec<u64>,
}

impl<'a> SceneGraphWalker<'a> {
    /// Create a walker starting from `root_id`.
    ///
    /// Returns `None` when `root_id` does not exist in the graph.
    #[must_use]
    pub fn new(graph: &'a SceneGraph, root_id: u64) -> Option<Self> {
        if graph.find(root_id).is_some() {
            Some(Self {
                graph,
                stack: vec![root_id],
            })
        } else {
            None
        }
    }
}

impl<'a> Iterator for SceneGraphWalker<'a> {
    type Item = &'a SceneNode;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.stack.pop()?;
        let node = self.graph.find(id)?;
        // Push children in reverse order so the first child is visited first.
        for &child_id in node.children.iter().rev() {
            self.stack.push(child_id);
        }
        Some(node)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. SceneNodeType::level
    #[test]
    fn test_node_type_level() {
        assert_eq!(SceneNodeType::Shot.level(), 0);
        assert_eq!(SceneNodeType::Act.level(), 1);
        assert_eq!(SceneNodeType::Sequence.level(), 2);
        assert_eq!(SceneNodeType::Episode.level(), 3);
        assert_eq!(SceneNodeType::Season.level(), 4);
    }

    // 2. SceneNode::duration_frames
    #[test]
    fn test_node_duration_frames() {
        let node = SceneNode {
            id: 0,
            name: "shot1".into(),
            node_type: SceneNodeType::Shot,
            children: vec![],
            start_frame: 100,
            end_frame: 250,
        };
        assert_eq!(node.duration_frames(), 150);
    }

    // 3. SceneNode::duration_frames – saturates to 0
    #[test]
    fn test_node_duration_frames_saturates() {
        let node = SceneNode {
            id: 1,
            name: "empty".into(),
            node_type: SceneNodeType::Shot,
            children: vec![],
            start_frame: 200,
            end_frame: 100, // end < start
        };
        assert_eq!(node.duration_frames(), 0);
    }

    // 4. SceneNode::contains_frame
    #[test]
    fn test_node_contains_frame() {
        let node = SceneNode {
            id: 2,
            name: "s".into(),
            node_type: SceneNodeType::Shot,
            children: vec![],
            start_frame: 10,
            end_frame: 20,
        };
        assert!(node.contains_frame(10));
        assert!(node.contains_frame(19));
        assert!(!node.contains_frame(20)); // end is exclusive
        assert!(!node.contains_frame(9));
    }

    // 5. SceneGraph::add_node assigns sequential IDs
    #[test]
    fn test_scene_graph_add_node_ids() {
        let mut g = SceneGraph::new();
        let id0 = g.add_node("A", SceneNodeType::Shot, 0, 10);
        let id1 = g.add_node("B", SceneNodeType::Act, 10, 20);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    // 6. SceneGraph::find – existing and missing node
    #[test]
    fn test_scene_graph_find() {
        let mut g = SceneGraph::new();
        let id = g.add_node("X", SceneNodeType::Sequence, 0, 100);
        assert!(g.find(id).is_some());
        assert!(g.find(999).is_none());
    }

    // 7. SceneGraph::add_child – success
    #[test]
    fn test_scene_graph_add_child_success() {
        let mut g = SceneGraph::new();
        let parent = g.add_node("P", SceneNodeType::Act, 0, 100);
        let child = g.add_node("C", SceneNodeType::Shot, 0, 50);
        assert!(g.add_child(parent, child));
        let p = g.find(parent).expect("should succeed in test");
        assert!(p.children.contains(&child));
    }

    // 8. SceneGraph::add_child – missing parent returns false
    #[test]
    fn test_scene_graph_add_child_missing_parent() {
        let mut g = SceneGraph::new();
        let child = g.add_node("C", SceneNodeType::Shot, 0, 10);
        assert!(!g.add_child(999, child));
    }

    // 9. SceneGraph::add_child – missing child returns false
    #[test]
    fn test_scene_graph_add_child_missing_child() {
        let mut g = SceneGraph::new();
        let parent = g.add_node("P", SceneNodeType::Act, 0, 100);
        assert!(!g.add_child(parent, 999));
    }

    // 10. SceneGraph::root_nodes – single root
    #[test]
    fn test_root_nodes_single() {
        let mut g = SceneGraph::new();
        let root = g.add_node("root", SceneNodeType::Season, 0, 10_000);
        let child = g.add_node("ep1", SceneNodeType::Episode, 0, 5_000);
        g.add_child(root, child);
        let roots = g.root_nodes();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, root);
    }

    // 11. SceneGraph::root_nodes – multiple roots
    #[test]
    fn test_root_nodes_multiple() {
        let mut g = SceneGraph::new();
        g.add_node("A", SceneNodeType::Shot, 0, 10);
        g.add_node("B", SceneNodeType::Shot, 10, 20);
        assert_eq!(g.root_nodes().len(), 2);
    }

    // 12. SceneGraphWalker – visits nodes depth-first
    #[test]
    fn test_walker_visits_all() {
        let mut g = SceneGraph::new();
        let root = g.add_node("root", SceneNodeType::Act, 0, 100);
        let c1 = g.add_node("c1", SceneNodeType::Shot, 0, 50);
        let c2 = g.add_node("c2", SceneNodeType::Shot, 50, 100);
        g.add_child(root, c1);
        g.add_child(root, c2);
        let visited: Vec<u64> = SceneGraphWalker::new(&g, root)
            .expect("should succeed in test")
            .map(|n| n.id)
            .collect();
        assert_eq!(visited.len(), 3);
        assert!(visited.contains(&root));
        assert!(visited.contains(&c1));
        assert!(visited.contains(&c2));
    }

    // 13. SceneGraphWalker::new – missing root returns None
    #[test]
    fn test_walker_missing_root() {
        let g = SceneGraph::new();
        assert!(SceneGraphWalker::new(&g, 0).is_none());
    }

    // 14. SceneGraph node data preserved
    #[test]
    fn test_scene_graph_node_data() {
        let mut g = SceneGraph::new();
        let id = g.add_node("MyShot", SceneNodeType::Shot, 5, 15);
        let n = g.find(id).expect("should succeed in test");
        assert_eq!(n.name, "MyShot");
        assert_eq!(n.start_frame, 5);
        assert_eq!(n.end_frame, 15);
        assert_eq!(n.node_type, SceneNodeType::Shot);
    }

    // 15. Add child is idempotent (duplicate child not added twice)
    #[test]
    fn test_add_child_idempotent() {
        let mut g = SceneGraph::new();
        let p = g.add_node("P", SceneNodeType::Act, 0, 100);
        let c = g.add_node("C", SceneNodeType::Shot, 0, 50);
        g.add_child(p, c);
        g.add_child(p, c); // second call should not duplicate
        let parent = g.find(p).expect("should succeed in test");
        assert_eq!(parent.children.len(), 1);
    }
}
