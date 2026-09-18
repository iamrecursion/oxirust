//! Asset relationship graph for tracking connections between media assets.
//!
//! Provides `RelationKind`, `AssetRelation`, and `RelationGraph` to model
//! and query directed relationships such as "derived from", "part of", and
//! "references".

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// The semantic kind of a relationship between two assets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationKind {
    /// The target asset was derived from the source (e.g. proxy from master).
    DerivedFrom,
    /// The source asset is part of the target (e.g. clip in a compilation).
    PartOf,
    /// The source references the target (e.g. colour-graded version).
    References,
    /// The source replaces the target (new version supersedes old).
    Replaces,
    /// Custom relationship label.
    Custom(String),
}

impl RelationKind {
    /// Return a stable string key for this kind.
    #[must_use]
    pub fn key(&self) -> &str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::PartOf => "part_of",
            Self::References => "references",
            Self::Replaces => "replaces",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// A directed relationship from one asset to another.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetRelation {
    /// Source asset id.
    pub from: u64,
    /// Target asset id.
    pub to: u64,
    /// Kind of the relationship.
    pub kind: RelationKind,
    /// Optional human-readable note.
    pub note: Option<String>,
}

impl AssetRelation {
    /// Create a new relation.
    #[must_use]
    pub fn new(from: u64, to: u64, kind: RelationKind) -> Self {
        Self {
            from,
            to,
            kind,
            note: None,
        }
    }

    /// Attach a textual note to this relation.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

/// A directed graph of asset relationships.
///
/// Edges are stored as [`AssetRelation`] values indexed for fast lookup by
/// both source and target.
#[derive(Debug, Default)]
pub struct RelationGraph {
    /// All relations stored in insertion order.
    relations: Vec<AssetRelation>,
    /// Index: from → list of relation indices.
    out_edges: HashMap<u64, Vec<usize>>,
    /// Index: to → list of relation indices.
    in_edges: HashMap<u64, Vec<usize>>,
}

impl RelationGraph {
    /// Create an empty relation graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a relation to the graph.
    ///
    /// Duplicate (same `from`, `to`, same `kind`) relations are silently ignored.
    pub fn add_relation(&mut self, relation: AssetRelation) {
        // De-duplicate
        for existing in &self.relations {
            if existing.from == relation.from
                && existing.to == relation.to
                && existing.kind == relation.kind
            {
                return;
            }
        }
        let idx = self.relations.len();
        self.out_edges.entry(relation.from).or_default().push(idx);
        self.in_edges.entry(relation.to).or_default().push(idx);
        self.relations.push(relation);
    }

    /// Return all relations originating from `asset_id`.
    #[must_use]
    pub fn outgoing(&self, asset_id: u64) -> Vec<&AssetRelation> {
        self.out_edges
            .get(&asset_id)
            .map(|idxs| idxs.iter().map(|&i| &self.relations[i]).collect())
            .unwrap_or_default()
    }

    /// Return all relations pointing to `asset_id`.
    #[must_use]
    pub fn incoming(&self, asset_id: u64) -> Vec<&AssetRelation> {
        self.in_edges
            .get(&asset_id)
            .map(|idxs| idxs.iter().map(|&i| &self.relations[i]).collect())
            .unwrap_or_default()
    }

    /// Return all relations of a specific kind.
    #[must_use]
    pub fn by_kind(&self, kind: &RelationKind) -> Vec<&AssetRelation> {
        self.relations.iter().filter(|r| &r.kind == kind).collect()
    }

    /// Return the set of all asset ids that are directly related to `asset_id`
    /// (either as source or target).
    #[must_use]
    pub fn neighbours(&self, asset_id: u64) -> HashSet<u64> {
        let mut result = HashSet::new();
        for r in self.outgoing(asset_id) {
            result.insert(r.to);
        }
        for r in self.incoming(asset_id) {
            result.insert(r.from);
        }
        result
    }

    /// Return the total number of relations in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.relations.len()
    }

    /// Return `true` if the graph contains no relations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    /// Remove all relations involving `asset_id` (as source or target).
    pub fn remove_asset(&mut self, asset_id: u64) {
        self.relations
            .retain(|r| r.from != asset_id && r.to != asset_id);
        // Rebuild indices from scratch after removal
        self.out_edges.clear();
        self.in_edges.clear();
        for (idx, rel) in self.relations.iter().enumerate() {
            self.out_edges.entry(rel.from).or_default().push(idx);
            self.in_edges.entry(rel.to).or_default().push(idx);
        }
    }

    /// Check whether a direct relation from `from` to `to` of any kind exists.
    #[must_use]
    pub fn has_direct_relation(&self, from: u64, to: u64) -> bool {
        self.outgoing(from).iter().any(|r| r.to == to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> RelationGraph {
        let mut g = RelationGraph::new();
        // 1 → 2 DerivedFrom
        g.add_relation(AssetRelation::new(1, 2, RelationKind::DerivedFrom));
        // 3 → 1 PartOf
        g.add_relation(AssetRelation::new(3, 1, RelationKind::PartOf));
        // 2 → 4 References
        g.add_relation(AssetRelation::new(2, 4, RelationKind::References));
        g
    }

    #[test]
    fn test_add_and_len() {
        let g = sample_graph();
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn test_is_empty_false() {
        let g = sample_graph();
        assert!(!g.is_empty());
    }

    #[test]
    fn test_is_empty_true() {
        let g = RelationGraph::new();
        assert!(g.is_empty());
    }

    #[test]
    fn test_outgoing() {
        let g = sample_graph();
        let out = g.outgoing(1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to, 2);
    }

    #[test]
    fn test_incoming() {
        let g = sample_graph();
        let inc = g.incoming(1);
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].from, 3);
    }

    #[test]
    fn test_neighbours() {
        let g = sample_graph();
        let n = g.neighbours(1);
        assert!(n.contains(&2));
        assert!(n.contains(&3));
    }

    #[test]
    fn test_by_kind() {
        let g = sample_graph();
        let derived = g.by_kind(&RelationKind::DerivedFrom);
        assert_eq!(derived.len(), 1);
        assert_eq!(derived[0].from, 1);
    }

    #[test]
    fn test_duplicate_relation_ignored() {
        let mut g = sample_graph();
        g.add_relation(AssetRelation::new(1, 2, RelationKind::DerivedFrom));
        assert_eq!(g.len(), 3); // still 3
    }

    #[test]
    fn test_has_direct_relation() {
        let g = sample_graph();
        assert!(g.has_direct_relation(1, 2));
        assert!(!g.has_direct_relation(1, 4));
    }

    #[test]
    fn test_remove_asset() {
        let mut g = sample_graph();
        g.remove_asset(1);
        // Both (1→2) and (3→1) should be removed; only (2→4) remains
        assert_eq!(g.len(), 1);
        assert_eq!(g.outgoing(2)[0].to, 4);
    }

    #[test]
    fn test_relation_with_note() {
        let rel = AssetRelation::new(10, 20, RelationKind::Replaces)
            .with_note("Version 2 supersedes version 1");
        assert!(rel.note.is_some());
        assert!(rel
            .note
            .as_ref()
            .expect("should succeed in test")
            .contains("supersedes"));
    }

    #[test]
    fn test_relation_kind_key() {
        assert_eq!(RelationKind::DerivedFrom.key(), "derived_from");
        assert_eq!(RelationKind::PartOf.key(), "part_of");
        assert_eq!(RelationKind::References.key(), "references");
        assert_eq!(RelationKind::Replaces.key(), "replaces");
        assert_eq!(RelationKind::Custom("linked".into()).key(), "linked");
    }

    #[test]
    fn test_no_outgoing_for_unknown_asset() {
        let g = sample_graph();
        assert!(g.outgoing(999).is_empty());
    }

    #[test]
    fn test_no_incoming_for_unknown_asset() {
        let g = sample_graph();
        assert!(g.incoming(999).is_empty());
    }
}
