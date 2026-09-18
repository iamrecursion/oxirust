//! Bidirectional clip relation graph with automatic symmetry enforcement.
//!
//! When a relation `A → B` of type `T` is added via
//! `BidirectionalRelationGraph::add`, the reverse edge `B → A` of the same
//! type is also inserted automatically.  This ensures that queries from either
//! endpoint return consistent results without requiring the caller to
//! explicitly add both directions.
//!
//! For asymmetric relation types (`Subclip`, `Version`) the automatic reverse
//! edge still uses the **same** `RelationType` so that a "subclip of" query
//! from the child clip is also possible.  Callers who need to distinguish
//! direction can use the lower-level `ClipRelationGraph` from
//! `clip_relations`.

#![allow(dead_code)]

use crate::clip_relations::{ClipId, ClipRelation, ClipRelationGraph, RelationType};
use std::collections::HashSet;

/// A relation graph that automatically maintains symmetric (bidirectional)
/// edges.
///
/// All mutation goes through `add` which writes both `(from, to, T)` and
/// `(to, from, T)`.  Read-only queries are the same as `ClipRelationGraph`.
#[derive(Debug, Default)]
pub struct BidirectionalRelationGraph {
    inner: ClipRelationGraph,
    /// Tracks which `(min_id, max_id, type)` pairs have already been inserted
    /// to prevent duplicate edges (both directions count as one canonical pair).
    inserted: HashSet<(ClipId, ClipId, String)>,
}

impl BidirectionalRelationGraph {
    /// Creates an empty bidirectional graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a relation between `from_id` and `to_id`.  The reverse direction
    /// is automatically added.
    ///
    /// If the pair `(from_id, to_id, relation_type)` has already been added
    /// (in either direction), this is a no-op.
    pub fn add(&mut self, relation: ClipRelation) {
        let key = canonical_key(relation.from_id, relation.to_id, &relation.relation_type);
        if self.inserted.contains(&key) {
            return;
        }
        self.inserted.insert(key);

        // Forward edge.
        let forward = ClipRelation::new(
            relation.from_id,
            relation.to_id,
            relation.relation_type.clone(),
        );
        // Reverse edge.
        let reverse = ClipRelation::new(relation.to_id, relation.from_id, relation.relation_type);

        self.inner.add(forward);
        self.inner.add(reverse);
    }

    /// Returns all clips that have a direct outgoing edge from `clip_id`
    /// (in either direction because the graph is symmetric).
    #[must_use]
    pub fn find_related(&self, clip_id: ClipId) -> Vec<ClipId> {
        self.inner.find_related(clip_id)
    }

    /// Returns all clips directly related to `clip_id` with a specific type.
    #[must_use]
    pub fn find_related_by_type(&self, clip_id: ClipId, rel_type: &RelationType) -> Vec<ClipId> {
        self.inner.find_related_by_type(clip_id, rel_type)
    }

    /// Returns all clips reachable from `clip_id` through any chain of edges
    /// (breadth-first traversal, does not include `clip_id` itself).
    #[must_use]
    pub fn transitively_related(&self, clip_id: ClipId) -> Vec<ClipId> {
        self.inner.transitively_related(clip_id)
    }

    /// Returns `true` if there is at least one direct edge between `a` and `b`
    /// (direction-agnostic).
    #[must_use]
    pub fn has_relation(&self, a: ClipId, b: ClipId) -> bool {
        self.inner.has_direct_relation(a, b) || self.inner.has_direct_relation(b, a)
    }

    /// Returns the number of canonical (undirected) pairs stored.
    #[must_use]
    pub fn canonical_edge_count(&self) -> usize {
        self.inserted.len()
    }

    /// Returns the total number of directed edges (always `2 × canonical`).
    #[must_use]
    pub fn directed_edge_count(&self) -> usize {
        self.inner.edge_count()
    }
}

/// Produces a stable canonical key for a relation pair so that `(A, B)` and
/// `(B, A)` map to the same entry.
fn canonical_key(a: ClipId, b: ClipId, rel: &RelationType) -> (ClipId, ClipId, String) {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    (lo, hi, rel.label().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_creates_both_directions() {
        let mut g = BidirectionalRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Duplicate));

        // Both 1→2 and 2→1 must exist.
        assert!(g.has_relation(1, 2));
        assert!(g.has_relation(2, 1));
    }

    #[test]
    fn test_find_related_from_both_endpoints() {
        let mut g = BidirectionalRelationGraph::new();
        g.add(ClipRelation::new(10, 20, RelationType::Alternate));

        let from_10 = g.find_related(10);
        let from_20 = g.find_related(20);

        assert!(from_10.contains(&20));
        assert!(from_20.contains(&10));
    }

    #[test]
    fn test_duplicate_add_is_idempotent() {
        let mut g = BidirectionalRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Subclip));
        g.add(ClipRelation::new(1, 2, RelationType::Subclip)); // duplicate
        g.add(ClipRelation::new(2, 1, RelationType::Subclip)); // reverse duplicate

        assert_eq!(g.canonical_edge_count(), 1);
        assert_eq!(g.directed_edge_count(), 2);
    }

    #[test]
    fn test_find_related_by_type_from_both_sides() {
        let mut g = BidirectionalRelationGraph::new();
        g.add(ClipRelation::new(5, 6, RelationType::Version));

        let versions_from_5 = g.find_related_by_type(5, &RelationType::Version);
        let versions_from_6 = g.find_related_by_type(6, &RelationType::Version);

        assert!(versions_from_5.contains(&6));
        assert!(versions_from_6.contains(&5));
    }

    #[test]
    fn test_transitively_related_bidirectional_chain() {
        // 1 ↔ 2 ↔ 3  (all bidirectional)
        let mut g = BidirectionalRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Version));
        g.add(ClipRelation::new(2, 3, RelationType::Version));

        let mut tr = g.transitively_related(1);
        tr.sort_unstable();
        assert_eq!(tr, vec![2, 3]);
    }

    #[test]
    fn test_canonical_edge_count_multiple_types() {
        let mut g = BidirectionalRelationGraph::new();
        // Same endpoints, different types → two canonical edges.
        g.add(ClipRelation::new(1, 2, RelationType::Duplicate));
        g.add(ClipRelation::new(1, 2, RelationType::Alternate));

        assert_eq!(g.canonical_edge_count(), 2);
        assert_eq!(g.directed_edge_count(), 4);
    }

    #[test]
    fn test_empty_graph_has_no_relations() {
        let g = BidirectionalRelationGraph::new();
        assert!(!g.has_relation(1, 2));
        assert!(g.find_related(1).is_empty());
    }

    #[test]
    fn test_multiple_clips_connected_to_one() {
        let mut g = BidirectionalRelationGraph::new();
        g.add(ClipRelation::new(1, 2, RelationType::Subclip));
        g.add(ClipRelation::new(1, 3, RelationType::Subclip));
        g.add(ClipRelation::new(1, 4, RelationType::Subclip));

        let related = g.find_related(1);
        assert_eq!(related.len(), 3);

        // All of 2, 3, 4 should also see clip 1 as related.
        assert!(g.find_related(2).contains(&1));
        assert!(g.find_related(3).contains(&1));
        assert!(g.find_related(4).contains(&1));
    }
}
