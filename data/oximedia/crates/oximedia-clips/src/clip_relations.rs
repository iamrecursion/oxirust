//! Clip relationship graph for `OxiMedia`.
//!
//! Tracks directed relationships between clips such as subclip derivation,
//! duplication, and versioning. Supports transitive relationship queries.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

/// Identifier for a clip (mirrors the one in the clip module).
pub type ClipId = u64;

/// The type of relationship between two clips.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    /// The destination is a subclip of the source.
    Subclip,
    /// The destination is a byte-for-byte or perceptual duplicate.
    Duplicate,
    /// The destination is an alternate angle or take.
    Alternate,
    /// The destination is a newer version of the source.
    Version,
}

impl RelationType {
    /// Returns `true` if this relation type implies the destination was
    /// *derived* from the source (i.e. the source is "original").
    #[must_use]
    pub fn is_derivative(&self) -> bool {
        matches!(self, Self::Subclip | Self::Version)
    }

    /// Returns a short human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Subclip => "subclip",
            Self::Duplicate => "duplicate",
            Self::Alternate => "alternate",
            Self::Version => "version",
        }
    }
}

/// A single directed relationship between two clips.
#[derive(Debug, Clone)]
pub struct ClipRelation {
    /// The originating clip.
    pub from_id: ClipId,
    /// The destination clip.
    pub to_id: ClipId,
    /// The kind of relationship.
    pub relation_type: RelationType,
}

impl ClipRelation {
    /// Create a new relation.
    #[must_use]
    pub fn new(from_id: ClipId, to_id: ClipId, relation_type: RelationType) -> Self {
        Self {
            from_id,
            to_id,
            relation_type,
        }
    }

    /// Returns `true` if this is a "forward" (derivative) relation.
    #[must_use]
    pub fn is_forward(&self) -> bool {
        self.relation_type.is_derivative()
    }
}

/// Graph of clip relationships supporting directed edges.
#[derive(Debug, Default)]
pub struct ClipRelationGraph {
    /// Adjacency list: from_id → list of (to_id, relation_type)
    edges: HashMap<ClipId, Vec<(ClipId, RelationType)>>,
}

impl ClipRelationGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a directed relationship to the graph.
    pub fn add(&mut self, relation: ClipRelation) {
        self.edges
            .entry(relation.from_id)
            .or_default()
            .push((relation.to_id, relation.relation_type));
    }

    /// Return all clips directly related to `clip_id` (outgoing edges).
    #[must_use]
    pub fn find_related(&self, clip_id: ClipId) -> Vec<ClipId> {
        self.edges
            .get(&clip_id)
            .map_or_else(Vec::new, |v| v.iter().map(|(id, _)| *id).collect())
    }

    /// Return all clips directly related to `clip_id` with a specific relation type.
    #[must_use]
    pub fn find_related_by_type(&self, clip_id: ClipId, rel_type: &RelationType) -> Vec<ClipId> {
        self.edges.get(&clip_id).map_or_else(Vec::new, |v| {
            v.iter()
                .filter(|(_, t)| t == rel_type)
                .map(|(id, _)| *id)
                .collect()
        })
    }

    /// Return all clips reachable from `clip_id` through any chain of edges
    /// (breadth-first traversal, does not include `clip_id` itself).
    #[must_use]
    pub fn transitively_related(&self, clip_id: ClipId) -> Vec<ClipId> {
        let mut visited: HashSet<ClipId> = HashSet::new();
        let mut queue: VecDeque<ClipId> = VecDeque::new();
        queue.push_back(clip_id);
        visited.insert(clip_id);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.edges.get(&current) {
                for (neighbor, _) in neighbors {
                    if visited.insert(*neighbor) {
                        queue.push_back(*neighbor);
                    }
                }
            }
        }

        visited.remove(&clip_id);
        visited.into_iter().collect()
    }

    /// Returns the total number of edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(Vec::len).sum()
    }

    /// Returns `true` if there is at least one direct edge from `from` to `to`.
    #[must_use]
    pub fn has_direct_relation(&self, from: ClipId, to: ClipId) -> bool {
        self.edges
            .get(&from)
            .map_or(false, |v| v.iter().any(|(id, _)| *id == to))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subclip_is_derivative() {
        assert!(RelationType::Subclip.is_derivative());
    }

    #[test]
    fn test_version_is_derivative() {
        assert!(RelationType::Version.is_derivative());
    }

    #[test]
    fn test_duplicate_not_derivative() {
        assert!(!RelationType::Duplicate.is_derivative());
    }

    #[test]
    fn test_alternate_not_derivative() {
        assert!(!RelationType::Alternate.is_derivative());
    }

    #[test]
    fn test_relation_labels() {
        assert_eq!(RelationType::Subclip.label(), "subclip");
        assert_eq!(RelationType::Duplicate.label(), "duplicate");
    }

    #[test]
    fn test_clip_relation_is_forward_true() {
        let r = ClipRelation::new(1, 2, RelationType::Subclip);
        assert!(r.is_forward());
    }

    #[test]
    fn test_clip_relation_is_forward_false() {
        let r = ClipRelation::new(1, 2, RelationType::Alternate);
        assert!(!r.is_forward());
    }

    #[test]
    fn test_find_related_empty() {
        let g = ClipRelationGraph::new();
        assert!(g.find_related(1).is_empty());
    }

    #[test]
    fn test_add_and_find_related() {
        let mut g = ClipRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Subclip));
        g.add(ClipRelation::new(1, 3, RelationType::Duplicate));
        let mut related = g.find_related(1);
        related.sort_unstable();
        assert_eq!(related, vec![2, 3]);
    }

    #[test]
    fn test_find_related_by_type() {
        let mut g = ClipRelationGraph::new();
        g.add(ClipRelation::new(10, 20, RelationType::Version));
        g.add(ClipRelation::new(10, 30, RelationType::Alternate));
        let versions = g.find_related_by_type(10, &RelationType::Version);
        assert_eq!(versions, vec![20]);
    }

    #[test]
    fn test_transitively_related_chain() {
        let mut g = ClipRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Version));
        g.add(ClipRelation::new(2, 3, RelationType::Version));
        let mut tr = g.transitively_related(1);
        tr.sort_unstable();
        assert_eq!(tr, vec![2, 3]);
    }

    #[test]
    fn test_transitively_related_no_cycle_explosion() {
        let mut g = ClipRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Duplicate));
        g.add(ClipRelation::new(2, 1, RelationType::Duplicate));
        // Should not loop infinitely
        let tr = g.transitively_related(1);
        assert_eq!(tr.len(), 1);
    }

    #[test]
    fn test_edge_count() {
        let mut g = ClipRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Subclip));
        g.add(ClipRelation::new(1, 3, RelationType::Alternate));
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_has_direct_relation_true() {
        let mut g = ClipRelationGraph::new();
        g.add(ClipRelation::new(5, 6, RelationType::Version));
        assert!(g.has_direct_relation(5, 6));
    }

    #[test]
    fn test_has_direct_relation_false() {
        let g = ClipRelationGraph::new();
        assert!(!g.has_direct_relation(5, 6));
    }
}
