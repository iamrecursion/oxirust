//! Related media linking and graph traversal
//!
//! Provides a directed graph of links between media assets, with:
//! - Multiple link types (derived, related, part-of, sequence, alternate, translation)
//! - BFS-based related-asset discovery with depth limiting
//! - Shortest-path search between assets
//! - DFS-based cycle detection
//! - Link statistics

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

/// The semantic relationship between two linked assets
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinkType {
    /// Target was produced from source (e.g. a transcode)
    DerivedFrom,
    /// Source and target are editorially related
    RelatedTo,
    /// Source is a component that belongs to the target asset
    PartOf,
    /// Source precedes target in a sequence
    Sequence,
    /// Target is an alternate version of source (different format/language)
    AlternateVersion,
    /// Target is a translation of source
    Translation,
}

impl LinkType {
    /// Human-readable label for the link type
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::RelatedTo => "related_to",
            Self::PartOf => "part_of",
            Self::Sequence => "sequence",
            Self::AlternateVersion => "alternate_version",
            Self::Translation => "translation",
        }
    }
}

/// A directed link between two media assets
#[derive(Debug, Clone)]
pub struct MediaLink {
    /// Unique link identifier
    pub id: u64,
    /// Source asset ID
    pub source_id: String,
    /// Target asset ID
    pub target_id: String,
    /// Type of relationship
    pub link_type: LinkType,
    /// Additional metadata about the link
    pub metadata: HashMap<String, String>,
}

impl MediaLink {
    /// Create a new media link
    #[must_use]
    pub fn new(
        id: u64,
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        link_type: LinkType,
    ) -> Self {
        Self {
            id,
            source_id: source_id.into(),
            target_id: target_id.into(),
            link_type,
            metadata: HashMap::new(),
        }
    }

    /// Create a link with metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A directed graph of media asset links
#[derive(Debug, Default)]
pub struct MediaLinkGraph {
    /// All links in the graph
    pub links: Vec<MediaLink>,
    /// Adjacency list: source_id -> list of target_ids
    adjacency: HashMap<String, Vec<String>>,
}

impl MediaLinkGraph {
    /// Create an empty graph
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a link to the graph
    pub fn add_link(&mut self, link: MediaLink) {
        self.adjacency
            .entry(link.source_id.clone())
            .or_default()
            .push(link.target_id.clone());
        // Ensure the target also appears as a node (even with no outgoing edges)
        self.adjacency.entry(link.target_id.clone()).or_default();
        self.links.push(link);
    }

    /// BFS traversal from `asset_id` up to `max_depth` hops.
    ///
    /// Returns `(asset_id, depth)` pairs for every reachable asset
    /// (excluding the start node itself), sorted by depth then ID.
    #[must_use]
    pub fn find_related(&self, asset_id: &str, max_depth: u32) -> Vec<(String, u32)> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        let mut results: Vec<(String, u32)> = Vec::new();

        visited.insert(asset_id.to_string());
        queue.push_back((asset_id.to_string(), 0));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(neighbors) = self.adjacency.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        results.push((neighbor.clone(), depth + 1));
                        queue.push_back((neighbor.clone(), depth + 1));
                    }
                }
            }
        }

        results
    }

    /// BFS shortest path from `from` to `to`.
    ///
    /// Returns `Some(path)` where `path[0] == from` and `path[last] == to`,
    /// or `None` if no path exists.
    #[must_use]
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<Vec<String>> = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back(vec![from.to_string()]);

        while let Some(path) = queue.pop_front() {
            let current = path
                .last()
                .expect("invariant: path non-empty (pushed via push_back with non-empty vec)");

            if let Some(neighbors) = self.adjacency.get(current) {
                for neighbor in neighbors {
                    if neighbor == to {
                        let mut result = path.clone();
                        result.push(neighbor.clone());
                        return Some(result);
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        queue.push_back(new_path);
                    }
                }
            }
        }

        None
    }

    /// Return all node IDs in the graph (all sources and targets)
    #[must_use]
    pub fn nodes(&self) -> Vec<&str> {
        self.adjacency.keys().map(String::as_str).collect()
    }

    /// Return all links originating from a given asset
    #[must_use]
    pub fn links_from(&self, asset_id: &str) -> Vec<&MediaLink> {
        self.links
            .iter()
            .filter(|l| l.source_id == asset_id)
            .collect()
    }
}

/// Detects cycles in a `MediaLinkGraph` using DFS
pub struct LinkValidator;

impl LinkValidator {
    /// Detect all cycles in the graph using DFS.
    ///
    /// Returns a list of cycles, where each cycle is a list of node IDs
    /// starting and ending at the same node.  Only unique cycles are returned.
    #[must_use]
    pub fn detect_cycles(graph: &MediaLinkGraph) -> Vec<Vec<String>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut rec_stack: HashSet<String> = HashSet::new();
        let mut cycles: Vec<Vec<String>> = Vec::new();

        let nodes: Vec<String> = graph.adjacency.keys().cloned().collect();

        for node in &nodes {
            if !visited.contains(node) {
                let mut path = Vec::new();
                Self::dfs(
                    node,
                    graph,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                );
            }
        }

        // Deduplicate cycles by their canonical (sorted) form
        let mut seen_keys: HashSet<String> = HashSet::new();
        let mut unique_cycles = Vec::new();
        for cycle in cycles {
            let key = Self::cycle_key(&cycle);
            if seen_keys.insert(key) {
                unique_cycles.push(cycle);
            }
        }

        unique_cycles
    }

    fn dfs(
        node: &str,
        graph: &MediaLinkGraph,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = graph.adjacency.get(node) {
            for neighbor in neighbors {
                if rec_stack.contains(neighbor) {
                    // Found a cycle - extract it from the current path
                    if let Some(start) = path.iter().position(|n| n == neighbor) {
                        let mut cycle = path[start..].to_vec();
                        cycle.push(neighbor.clone());
                        cycles.push(cycle);
                    }
                } else if !visited.contains(neighbor) {
                    Self::dfs(neighbor, graph, visited, rec_stack, path, cycles);
                }
            }
        }

        path.pop();
        rec_stack.remove(node);
    }

    /// Create a canonical string key for a cycle to support deduplication
    fn cycle_key(cycle: &[String]) -> String {
        // Rotate so the smallest element is first, then stringify
        if cycle.is_empty() {
            return String::new();
        }
        let inner = &cycle[..cycle.len().saturating_sub(1)]; // exclude trailing repeat
        if inner.is_empty() {
            return cycle.join(",");
        }
        let min_pos = inner
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let rotated: Vec<&String> = inner[min_pos..]
            .iter()
            .chain(inner[..min_pos].iter())
            .collect();
        rotated
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Statistics about a `MediaLinkGraph`
#[derive(Debug, Clone)]
pub struct LinkStats {
    /// Total number of links in the graph
    pub total_links: u64,
    /// Average number of outgoing connections per node
    pub avg_connections: f32,
    /// Maximum outgoing connections from any single node
    pub max_connections: u32,
}

impl LinkStats {
    /// Compute statistics for a graph
    #[must_use]
    pub fn compute(graph: &MediaLinkGraph) -> Self {
        let total_links = graph.links.len() as u64;

        if graph.adjacency.is_empty() {
            return Self {
                total_links,
                avg_connections: 0.0,
                max_connections: 0,
            };
        }

        let counts: Vec<u32> = graph.adjacency.values().map(|v| v.len() as u32).collect();

        let max_connections = counts.iter().copied().max().unwrap_or(0);
        let sum: u32 = counts.iter().sum();
        let avg_connections = sum as f32 / counts.len() as f32;

        Self {
            total_links,
            avg_connections,
            max_connections,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> MediaLinkGraph {
        let mut g = MediaLinkGraph::new();
        g.add_link(MediaLink::new(1, "A", "B", LinkType::DerivedFrom));
        g.add_link(MediaLink::new(2, "B", "C", LinkType::RelatedTo));
        g.add_link(MediaLink::new(3, "A", "D", LinkType::PartOf));
        g
    }

    #[test]
    fn test_link_type_labels() {
        assert_eq!(LinkType::DerivedFrom.label(), "derived_from");
        assert_eq!(LinkType::RelatedTo.label(), "related_to");
        assert_eq!(LinkType::PartOf.label(), "part_of");
        assert_eq!(LinkType::Sequence.label(), "sequence");
        assert_eq!(LinkType::AlternateVersion.label(), "alternate_version");
        assert_eq!(LinkType::Translation.label(), "translation");
    }

    #[test]
    fn test_add_link_and_nodes() {
        let g = make_graph();
        let nodes = g.nodes();
        assert!(nodes.contains(&"A"));
        assert!(nodes.contains(&"B"));
        assert!(nodes.contains(&"C"));
        assert!(nodes.contains(&"D"));
    }

    #[test]
    fn test_find_related_depth_1() {
        let g = make_graph();
        let related = g.find_related("A", 1);
        let ids: HashSet<&str> = related.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains("B"));
        assert!(ids.contains("D"));
        assert!(!ids.contains("C")); // C is at depth 2
    }

    #[test]
    fn test_find_related_depth_2() {
        let g = make_graph();
        let related = g.find_related("A", 2);
        let ids: HashSet<&str> = related.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains("C")); // Now reachable
    }

    #[test]
    fn test_find_related_empty() {
        let g = MediaLinkGraph::new();
        assert!(g.find_related("A", 5).is_empty());
    }

    #[test]
    fn test_shortest_path_direct() {
        let g = make_graph();
        let path = g.shortest_path("A", "B").expect("should succeed in test");
        assert_eq!(path, vec!["A", "B"]);
    }

    #[test]
    fn test_shortest_path_two_hops() {
        let g = make_graph();
        let path = g.shortest_path("A", "C").expect("should succeed in test");
        assert_eq!(path, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_shortest_path_no_path() {
        let g = make_graph();
        // C has no outgoing edges -> no path from C to A
        assert!(g.shortest_path("C", "A").is_none());
    }

    #[test]
    fn test_shortest_path_self() {
        let g = make_graph();
        let path = g.shortest_path("A", "A").expect("should succeed in test");
        assert_eq!(path, vec!["A"]);
    }

    #[test]
    fn test_detect_cycles_none() {
        let g = make_graph(); // A->B->C, A->D -- no cycles
        let cycles = LinkValidator::detect_cycles(&g);
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_detect_cycles_simple() {
        let mut g = MediaLinkGraph::new();
        g.add_link(MediaLink::new(1, "X", "Y", LinkType::DerivedFrom));
        g.add_link(MediaLink::new(2, "Y", "X", LinkType::DerivedFrom));
        let cycles = LinkValidator::detect_cycles(&g);
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_link_stats_basic() {
        let g = make_graph();
        let stats = LinkStats::compute(&g);
        assert_eq!(stats.total_links, 3);
        assert!(stats.max_connections >= 2); // A has 2 outgoing
        assert!(stats.avg_connections > 0.0);
    }

    #[test]
    fn test_link_stats_empty() {
        let g = MediaLinkGraph::new();
        let stats = LinkStats::compute(&g);
        assert_eq!(stats.total_links, 0);
        assert_eq!(stats.max_connections, 0);
        assert_eq!(stats.avg_connections, 0.0);
    }

    #[test]
    fn test_media_link_with_metadata() {
        let link =
            MediaLink::new(1, "A", "B", LinkType::Translation).with_metadata("language", "fr");
        assert_eq!(
            link.metadata.get("language").map(String::as_str),
            Some("fr")
        );
    }
}
