//! RDF graph statistics and structural analysis.
//!
//! Computes graph metrics from triple stores: node/edge counts,
//! degree distributions, predicate frequencies, and density.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Comprehensive statistics about an RDF graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of unique subjects
    pub subject_count: usize,
    /// Total number of unique objects
    pub object_count: usize,
    /// Total number of unique nodes (subjects union objects)
    pub node_count: usize,
    /// Total number of triples (edges)
    pub edge_count: usize,
    /// Number of unique predicates
    pub predicate_count: usize,
    /// Graph density: edges / (nodes * (nodes-1)) for directed graphs
    pub density: f64,
    /// Average out-degree (edges per subject)
    pub avg_out_degree: f64,
    /// Maximum out-degree
    pub max_out_degree: usize,
    /// Average in-degree (edges per object)
    pub avg_in_degree: f64,
    /// Maximum in-degree
    pub max_in_degree: usize,
    /// Self-loop count (subject == object)
    pub self_loop_count: usize,
}

impl GraphStats {
    /// Compute stats from a list of (subject, predicate, object) triples.
    pub fn compute(triples: &[(String, String, String)]) -> Self {
        if triples.is_empty() {
            return Self {
                subject_count: 0,
                object_count: 0,
                node_count: 0,
                edge_count: 0,
                predicate_count: 0,
                density: 0.0,
                avg_out_degree: 0.0,
                max_out_degree: 0,
                avg_in_degree: 0.0,
                max_in_degree: 0,
                self_loop_count: 0,
            };
        }

        let mut subjects: HashSet<&str> = HashSet::new();
        let mut objects: HashSet<&str> = HashSet::new();
        let mut predicates: HashSet<&str> = HashSet::new();
        let mut out_degree: HashMap<&str, usize> = HashMap::new();
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut self_loop_count: usize = 0;

        for (s, p, o) in triples {
            subjects.insert(s.as_str());
            objects.insert(o.as_str());
            predicates.insert(p.as_str());

            *out_degree.entry(s.as_str()).or_insert(0) += 1;
            *in_degree.entry(o.as_str()).or_insert(0) += 1;

            if s == o {
                self_loop_count += 1;
            }
        }

        let subject_count = subjects.len();
        let object_count = objects.len();
        let nodes: HashSet<&str> = subjects.union(&objects).copied().collect();
        let node_count = nodes.len();
        let edge_count = triples.len();
        let predicate_count = predicates.len();

        let density = if node_count > 1 {
            edge_count as f64 / (node_count as f64 * (node_count as f64 - 1.0))
        } else {
            0.0
        };

        let avg_out_degree = if subject_count > 0 {
            edge_count as f64 / subject_count as f64
        } else {
            0.0
        };

        let max_out_degree = out_degree.values().copied().max().unwrap_or(0);

        let avg_in_degree = if object_count > 0 {
            edge_count as f64 / object_count as f64
        } else {
            0.0
        };

        let max_in_degree = in_degree.values().copied().max().unwrap_or(0);

        Self {
            subject_count,
            object_count,
            node_count,
            edge_count,
            predicate_count,
            density,
            avg_out_degree,
            max_out_degree,
            avg_in_degree,
            max_in_degree,
            self_loop_count,
        }
    }

    /// Is the graph empty?
    pub fn is_empty(&self) -> bool {
        self.edge_count == 0
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "Graph: {} nodes, {} edges, {} predicates | \
             density={:.6} | out-degree: avg={:.2}, max={} | \
             in-degree: avg={:.2}, max={} | self-loops: {}",
            self.node_count,
            self.edge_count,
            self.predicate_count,
            self.density,
            self.avg_out_degree,
            self.max_out_degree,
            self.avg_in_degree,
            self.max_in_degree,
            self.self_loop_count,
        )
    }
}

/// Per-predicate statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateStats {
    /// Predicate name (IRI)
    pub name: String,
    /// Number of triples using this predicate
    pub count: usize,
    /// Number of unique subjects for this predicate
    pub unique_subjects: usize,
    /// Number of unique objects for this predicate
    pub unique_objects: usize,
    /// Whether each subject maps to at most one object (functional property)
    pub is_functional: bool,
    /// Whether each object maps to at most one subject (inverse functional property)
    pub is_inverse_functional: bool,
}

/// Compute per-predicate statistics from triples.
pub fn predicate_statistics(triples: &[(String, String, String)]) -> Vec<PredicateStats> {
    let mut pred_data: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();

    for (s, p, o) in triples {
        pred_data
            .entry(p.as_str())
            .or_default()
            .push((s.as_str(), o.as_str()));
    }

    let mut results: Vec<PredicateStats> = pred_data
        .into_iter()
        .map(|(name, pairs)| {
            let count = pairs.len();

            let mut subj_set: HashSet<&str> = HashSet::new();
            let mut obj_set: HashSet<&str> = HashSet::new();
            let mut subj_to_objs: HashMap<&str, HashSet<&str>> = HashMap::new();
            let mut obj_to_subjs: HashMap<&str, HashSet<&str>> = HashMap::new();

            for &(s, o) in &pairs {
                subj_set.insert(s);
                obj_set.insert(o);
                subj_to_objs.entry(s).or_default().insert(o);
                obj_to_subjs.entry(o).or_default().insert(s);
            }

            let is_functional = subj_to_objs.values().all(|objs| objs.len() <= 1);
            let is_inverse_functional = obj_to_subjs.values().all(|subjs| subjs.len() <= 1);

            PredicateStats {
                name: name.to_string(),
                count,
                unique_subjects: subj_set.len(),
                unique_objects: obj_set.len(),
                is_functional,
                is_inverse_functional,
            }
        })
        .collect();

    results.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));
    results
}

/// Degree distribution: maps degree to count of nodes with that degree.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DegreeDistribution {
    /// Out-degree distribution: degree -> number of nodes
    pub out_degrees: HashMap<usize, usize>,
    /// In-degree distribution: degree -> number of nodes
    pub in_degrees: HashMap<usize, usize>,
}

impl DegreeDistribution {
    /// Compute degree distributions from triples.
    pub fn compute(triples: &[(String, String, String)]) -> Self {
        let mut out_deg: HashMap<&str, usize> = HashMap::new();
        let mut in_deg: HashMap<&str, usize> = HashMap::new();

        for (s, _p, o) in triples {
            *out_deg.entry(s.as_str()).or_insert(0) += 1;
            *in_deg.entry(o.as_str()).or_insert(0) += 1;
        }

        let mut out_degrees: HashMap<usize, usize> = HashMap::new();
        for &deg in out_deg.values() {
            *out_degrees.entry(deg).or_insert(0) += 1;
        }

        let mut in_degrees: HashMap<usize, usize> = HashMap::new();
        for &deg in in_deg.values() {
            *in_degrees.entry(deg).or_insert(0) += 1;
        }

        Self {
            out_degrees,
            in_degrees,
        }
    }

    /// Median out-degree.
    pub fn median_out_degree(&self) -> f64 {
        compute_median_from_distribution(&self.out_degrees)
    }

    /// Median in-degree.
    pub fn median_in_degree(&self) -> f64 {
        compute_median_from_distribution(&self.in_degrees)
    }
}

/// Compute median from a frequency distribution (degree -> count).
fn compute_median_from_distribution(dist: &HashMap<usize, usize>) -> f64 {
    if dist.is_empty() {
        return 0.0;
    }

    let total: usize = dist.values().sum();
    if total == 0 {
        return 0.0;
    }

    let mut sorted_degrees: Vec<usize> = dist.keys().copied().collect();
    sorted_degrees.sort_unstable();

    // Expand into sorted values to find median
    let mut values: Vec<usize> = Vec::with_capacity(total);
    for &deg in &sorted_degrees {
        let count = dist.get(&deg).copied().unwrap_or(0);
        for _ in 0..count {
            values.push(deg);
        }
    }

    if values.len() % 2 == 1 {
        values[values.len() / 2] as f64
    } else {
        let mid = values.len() / 2;
        (values[mid - 1] as f64 + values[mid] as f64) / 2.0
    }
}

/// Compute the number of weakly connected components using union-find.
pub fn connected_components(triples: &[(String, String, String)]) -> usize {
    let mut nodes: HashSet<String> = HashSet::new();
    let mut uf = UnionFind::new();

    for (s, _p, o) in triples {
        nodes.insert(s.clone());
        nodes.insert(o.clone());
        uf.union(s, o);
    }

    if nodes.is_empty() {
        return 0;
    }

    uf.component_count(&nodes)
}

/// Simple union-find (disjoint set) structure for graph connectivity.
struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
        }
    }

    fn find(&mut self, x: &str) -> String {
        // Ensure node exists
        if !self.parent.contains_key(x) {
            self.parent.insert(x.to_string(), x.to_string());
            return x.to_string();
        }

        // Path compression via iterative approach
        let mut current = x.to_string();
        let mut path = Vec::new();

        loop {
            let p = self
                .parent
                .get(&current)
                .cloned()
                .unwrap_or_else(|| current.clone());
            if p == current {
                break;
            }
            path.push(current.clone());
            current = p;
        }

        // Compress path
        let root = current;
        for node in path {
            self.parent.insert(node, root.clone());
        }

        root
    }

    fn union(&mut self, x: &str, y: &str) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent.insert(rx, ry);
        }
    }

    fn component_count(&mut self, nodes: &HashSet<String>) -> usize {
        let mut roots: HashSet<String> = HashSet::new();
        for node in nodes {
            let root = self.find(node);
            roots.insert(root);
        }
        roots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triples(data: &[(&str, &str, &str)]) -> Vec<(String, String, String)> {
        data.iter()
            .map(|(s, p, o)| (s.to_string(), p.to_string(), o.to_string()))
            .collect()
    }

    #[test]
    fn test_graph_stats_empty() {
        let stats = GraphStats::compute(&[]);
        assert_eq!(stats.subject_count, 0);
        assert_eq!(stats.object_count, 0);
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.predicate_count, 0);
        assert_eq!(stats.density, 0.0);
        assert_eq!(stats.avg_out_degree, 0.0);
        assert_eq!(stats.max_out_degree, 0);
        assert_eq!(stats.avg_in_degree, 0.0);
        assert_eq!(stats.max_in_degree, 0);
        assert_eq!(stats.self_loop_count, 0);
    }

    #[test]
    fn test_graph_stats_single_triple() {
        let triples = make_triples(&[("a", "p", "b")]);
        let stats = GraphStats::compute(&triples);
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
        assert_eq!(stats.subject_count, 1);
        assert_eq!(stats.object_count, 1);
    }

    #[test]
    fn test_graph_stats_node_count() {
        // "a" appears as both subject and object, should be deduplicated
        let triples = make_triples(&[("a", "p", "b"), ("b", "q", "a")]);
        let stats = GraphStats::compute(&triples);
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.subject_count, 2);
        assert_eq!(stats.object_count, 2);
    }

    #[test]
    fn test_graph_stats_density() {
        // 3 nodes, 3 edges => density = 3 / (3 * 2) = 0.5
        let triples = make_triples(&[("a", "p", "b"), ("b", "p", "c"), ("c", "p", "a")]);
        let stats = GraphStats::compute(&triples);
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 3);
        let expected_density = 3.0 / (3.0 * 2.0);
        assert!((stats.density - expected_density).abs() < 1e-10);
    }

    #[test]
    fn test_graph_stats_degrees() {
        // a->b, a->c, b->c
        let triples = make_triples(&[("a", "p", "b"), ("a", "p", "c"), ("b", "p", "c")]);
        let stats = GraphStats::compute(&triples);
        // out-degrees: a=2, b=1 => avg = 3/2 = 1.5, max = 2
        assert!((stats.avg_out_degree - 1.5).abs() < 1e-10);
        assert_eq!(stats.max_out_degree, 2);
        // in-degrees: b=1, c=2 => avg = 3/2 = 1.5, max = 2
        assert!((stats.avg_in_degree - 1.5).abs() < 1e-10);
        assert_eq!(stats.max_in_degree, 2);
    }

    #[test]
    fn test_graph_stats_self_loop() {
        let triples = make_triples(&[("a", "p", "a"), ("a", "q", "b")]);
        let stats = GraphStats::compute(&triples);
        assert_eq!(stats.self_loop_count, 1);
    }

    #[test]
    fn test_graph_stats_summary() {
        let triples = make_triples(&[("a", "p", "b")]);
        let stats = GraphStats::compute(&triples);
        let summary = stats.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("2 nodes"));
        assert!(summary.contains("1 edges"));
    }

    #[test]
    fn test_graph_stats_is_empty() {
        let stats = GraphStats::compute(&[]);
        assert!(stats.is_empty());

        let triples = make_triples(&[("a", "p", "b")]);
        let stats2 = GraphStats::compute(&triples);
        assert!(!stats2.is_empty());
    }

    #[test]
    fn test_predicate_stats_count() {
        let triples = make_triples(&[
            ("a", "knows", "b"),
            ("b", "knows", "c"),
            ("a", "likes", "c"),
        ]);
        let stats = predicate_statistics(&triples);
        assert_eq!(stats.len(), 2);
        // sorted by count desc: knows=2, likes=1
        assert_eq!(stats[0].name, "knows");
        assert_eq!(stats[0].count, 2);
        assert_eq!(stats[1].name, "likes");
        assert_eq!(stats[1].count, 1);
    }

    #[test]
    fn test_predicate_stats_functional() {
        // Each subject has exactly one object for "name"
        let triples = make_triples(&[("a", "name", "Alice"), ("b", "name", "Bob")]);
        let stats = predicate_statistics(&triples);
        assert_eq!(stats.len(), 1);
        assert!(stats[0].is_functional);
    }

    #[test]
    fn test_predicate_stats_not_functional() {
        // Subject "a" has two objects for "knows"
        let triples = make_triples(&[("a", "knows", "b"), ("a", "knows", "c")]);
        let stats = predicate_statistics(&triples);
        assert_eq!(stats.len(), 1);
        assert!(!stats[0].is_functional);
    }

    #[test]
    fn test_predicate_stats_inverse_functional() {
        // Each object has exactly one subject
        let triples = make_triples(&[("a", "id", "1"), ("b", "id", "2")]);
        let stats = predicate_statistics(&triples);
        assert_eq!(stats.len(), 1);
        assert!(stats[0].is_inverse_functional);
    }

    #[test]
    fn test_degree_distribution_compute() {
        // a->b, a->c, b->c
        let triples = make_triples(&[("a", "p", "b"), ("a", "p", "c"), ("b", "p", "c")]);
        let dist = DegreeDistribution::compute(&triples);
        // out: a=2, b=1 => {2: 1, 1: 1}
        assert_eq!(dist.out_degrees.get(&2), Some(&1));
        assert_eq!(dist.out_degrees.get(&1), Some(&1));
        // in: b=1, c=2 => {1: 1, 2: 1}
        assert_eq!(dist.in_degrees.get(&1), Some(&1));
        assert_eq!(dist.in_degrees.get(&2), Some(&1));
    }

    #[test]
    fn test_degree_distribution_median() {
        // 4 nodes with out-degrees: 1, 1, 2, 3 => median = (1+2)/2 = 1.5
        let triples = make_triples(&[
            ("a", "p", "x"),
            ("b", "p", "x"),
            ("c", "p", "x"),
            ("c", "p", "y"),
            ("d", "p", "x"),
            ("d", "p", "y"),
            ("d", "p", "z"),
        ]);
        let dist = DegreeDistribution::compute(&triples);
        // out-degrees: a=1, b=1, c=2, d=3 => sorted: [1,1,2,3] => median=(1+2)/2=1.5
        assert!((dist.median_out_degree() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_connected_components_single() {
        // Fully connected triangle => 1 component
        let triples = make_triples(&[("a", "p", "b"), ("b", "p", "c"), ("c", "p", "a")]);
        assert_eq!(connected_components(&triples), 1);
    }

    #[test]
    fn test_connected_components_two() {
        // Two disconnected edges => 2 components
        let triples = make_triples(&[("a", "p", "b"), ("c", "p", "d")]);
        assert_eq!(connected_components(&triples), 2);
    }

    #[test]
    fn test_connected_components_isolated() {
        // a-b connected, c-d connected, but c also connects to e
        // Actually: make truly isolated by having separate components
        let triples = make_triples(&[("a", "p", "b"), ("c", "p", "d"), ("e", "p", "f")]);
        assert_eq!(connected_components(&triples), 3);
    }

    #[test]
    fn test_graph_stats_multiple_predicates() {
        let triples = make_triples(&[
            ("a", "knows", "b"),
            ("a", "likes", "c"),
            ("b", "hates", "c"),
        ]);
        let stats = GraphStats::compute(&triples);
        assert_eq!(stats.predicate_count, 3);
    }
}
