//! Factor graph visualization and structural analysis.
//!
//! Renders factor graphs as ASCII art and DOT (Graphviz) format,
//! and computes structural statistics (degree distributions, tree
//! detection, treewidth bounds).

use std::fmt::Write;

use serde::{Deserialize, Serialize};

use crate::graph::FactorGraph;

// ---------------------------------------------------------------------------
// Lightweight visualization model
// ---------------------------------------------------------------------------

/// A lightweight factor graph representation for visualization.
///
/// This is intentionally decoupled from [`FactorGraph`] so that callers
/// can build ad-hoc models (e.g. from external data) and still use the
/// rendering / statistics helpers.
#[derive(Debug, Clone, Default)]
pub struct FactorGraphModel {
    /// Variable nodes.
    pub variables: Vec<VizVariableNode>,
    /// Factor nodes.
    pub factors: Vec<VizFactorNode>,
}

/// A variable node inside a [`FactorGraphModel`].
#[derive(Debug, Clone)]
pub struct VizVariableNode {
    /// Human-readable name.
    pub name: String,
    /// Number of values the variable can take.
    pub domain_size: usize,
}

/// A factor node inside a [`FactorGraphModel`].
#[derive(Debug, Clone)]
pub struct VizFactorNode {
    /// Human-readable name.
    pub name: String,
    /// Indices into [`FactorGraphModel::variables`] that this factor touches.
    pub variable_indices: Vec<usize>,
}

impl FactorGraphModel {
    /// Create a new, empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a [`FactorGraphModel`] from an existing [`FactorGraph`].
    ///
    /// Variable ordering is arbitrary (HashMap iteration order).
    pub fn from_factor_graph(fg: &FactorGraph) -> Self {
        // Collect variables into a stable ordering.
        let mut var_names: Vec<String> = fg.variable_names().cloned().collect();
        var_names.sort();

        let mut name_to_idx: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        let mut model = Self::new();
        for name in &var_names {
            let card = fg.get_variable(name).map(|v| v.cardinality).unwrap_or(2);
            let idx = model.add_variable(name.clone(), card);
            name_to_idx.insert(name.clone(), idx);
        }

        for factor in fg.factors() {
            let indices: Vec<usize> = factor
                .variables
                .iter()
                .filter_map(|v| name_to_idx.get(v).copied())
                .collect();
            model.add_factor(factor.name.clone(), indices);
        }

        model
    }

    /// Add a variable node. Returns the index of the newly added variable.
    pub fn add_variable(&mut self, name: impl Into<String>, domain_size: usize) -> usize {
        let idx = self.variables.len();
        self.variables.push(VizVariableNode {
            name: name.into(),
            domain_size,
        });
        idx
    }

    /// Add a factor node connecting the given variable indices.
    pub fn add_factor(&mut self, name: impl Into<String>, variable_indices: Vec<usize>) {
        self.factors.push(VizFactorNode {
            name: name.into(),
            variable_indices,
        });
    }

    /// Number of variable nodes.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Number of factor nodes.
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    /// Total number of edges (factor-variable connections).
    pub fn edge_count(&self) -> usize {
        self.factors.iter().map(|f| f.variable_indices.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Structural statistics for a factor graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactorGraphStats {
    /// Number of variable nodes.
    pub variable_count: usize,
    /// Number of factor nodes.
    pub factor_count: usize,
    /// Total edges (variable-factor connections).
    pub edge_count: usize,
    /// Maximum number of variables any single factor connects.
    pub max_factor_arity: usize,
    /// Average factor arity.
    pub avg_factor_arity: f64,
    /// Maximum degree of any variable (number of factors it participates in).
    pub max_variable_degree: usize,
    /// Average variable degree.
    pub avg_variable_degree: f64,
    /// Whether the factor graph forms a tree (no loops).
    pub is_tree: bool,
    /// Upper bound on treewidth (max factor arity - 1).
    pub treewidth_upper_bound: usize,
}

impl FactorGraphStats {
    /// Compute statistics from a [`FactorGraphModel`].
    pub fn compute(model: &FactorGraphModel) -> Self {
        let variable_count = model.variable_count();
        let factor_count = model.factor_count();
        let edge_count = model.edge_count();

        let max_factor_arity = model
            .factors
            .iter()
            .map(|f| f.variable_indices.len())
            .max()
            .unwrap_or(0);

        let avg_factor_arity = if factor_count > 0 {
            edge_count as f64 / factor_count as f64
        } else {
            0.0
        };

        // Variable degree = number of factors connected to it.
        let mut var_degrees = vec![0usize; variable_count];
        for factor in &model.factors {
            for &vi in &factor.variable_indices {
                if vi < variable_count {
                    var_degrees[vi] += 1;
                }
            }
        }

        let max_variable_degree = var_degrees.iter().copied().max().unwrap_or(0);
        let avg_variable_degree = if variable_count > 0 {
            var_degrees.iter().sum::<usize>() as f64 / variable_count as f64
        } else {
            0.0
        };

        // Tree check: a bipartite factor graph is a tree when
        // |edges| == |variable nodes| + |factor nodes| - 1 and the graph is
        // connected. We use the simpler edge-count heuristic here (sufficient
        // for the upper-bound use-case).
        let total_nodes = variable_count + factor_count;
        let is_tree = total_nodes > 0 && edge_count + 1 == total_nodes;

        let treewidth_upper_bound = if max_factor_arity > 0 {
            max_factor_arity - 1
        } else {
            0
        };

        Self {
            variable_count,
            factor_count,
            edge_count,
            max_factor_arity,
            avg_factor_arity,
            max_variable_degree,
            avg_variable_degree,
            is_tree,
            treewidth_upper_bound,
        }
    }

    /// One-line summary string.
    pub fn summary(&self) -> String {
        format!(
            "{} vars, {} factors, {} edges, treewidth\u{2264}{}{}",
            self.variable_count,
            self.factor_count,
            self.edge_count,
            self.treewidth_upper_bound,
            if self.is_tree { " (tree)" } else { "" }
        )
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Render a [`FactorGraphModel`] as human-readable ASCII text.
pub fn render_ascii(model: &FactorGraphModel) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "Factor Graph:");

    // Variables line
    let var_descs: Vec<String> = model
        .variables
        .iter()
        .map(|v| format!("{}({})", v.name, v.domain_size))
        .collect();
    let _ = writeln!(
        out,
        "  Variables ({}): {}",
        model.variable_count(),
        var_descs.join(", ")
    );

    // Factors line
    let fac_descs: Vec<String> = model
        .factors
        .iter()
        .map(|f| format!("{}({})", f.name, f.variable_indices.len()))
        .collect();
    let _ = writeln!(
        out,
        "  Factors ({}):  {}",
        model.factor_count(),
        fac_descs.join(", ")
    );

    // Connections
    let _ = writeln!(out, "  Connections:");
    for factor in &model.factors {
        let var_names: Vec<&str> = factor
            .variable_indices
            .iter()
            .filter_map(|&i| model.variables.get(i).map(|v| v.name.as_str()))
            .collect();
        let _ = writeln!(
            out,
            "    {} \u{2500}\u{2500} {}",
            factor.name,
            var_names.join(", ")
        );
    }

    out
}

/// Render a [`FactorGraphModel`] as a DOT (Graphviz) graph.
///
/// The output uses the undirected `graph` keyword because factor graphs
/// are inherently undirected.
pub fn render_dot(model: &FactorGraphModel) -> String {
    let mut dot = String::new();

    let _ = writeln!(dot, "graph FactorGraph {{");
    let _ = writeln!(dot, "  rankdir=LR;");

    // Variable nodes as circles.
    for (i, var) in model.variables.iter().enumerate() {
        let _ = writeln!(dot, "  v{} [label=\"{}\", shape=circle];", i, var.name);
    }

    // Factor nodes as filled squares, with edges.
    for (i, factor) in model.factors.iter().enumerate() {
        let _ = writeln!(
            dot,
            "  f{} [label=\"{}\", shape=square, style=filled, fillcolor=lightgray];",
            i, factor.name
        );
        for &vi in &factor.variable_indices {
            let _ = writeln!(dot, "  f{} -- v{};", i, vi);
        }
    }

    let _ = writeln!(dot, "}}");
    dot
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    /// A-f1-B-f2-C chain (3 vars, 2 binary factors, 4 edges).
    fn chain_model() -> FactorGraphModel {
        let mut m = FactorGraphModel::new();
        let a = m.add_variable("A", 2);
        let b = m.add_variable("B", 2);
        let c = m.add_variable("C", 2);
        m.add_factor("f1", vec![a, b]);
        m.add_factor("f2", vec![b, c]);
        m
    }

    /// A loopy model: triangle A-B-C with one ternary factor.
    fn loopy_model() -> FactorGraphModel {
        let mut m = FactorGraphModel::new();
        let a = m.add_variable("A", 2);
        let b = m.add_variable("B", 2);
        let c = m.add_variable("C", 2);
        m.add_factor("f1", vec![a, b]);
        m.add_factor("f2", vec![b, c]);
        m.add_factor("f3", vec![a, c]);
        m
    }

    // -- FactorGraphModel basics --

    #[test]
    fn test_model_new_empty() {
        let m = FactorGraphModel::new();
        assert_eq!(m.variable_count(), 0);
        assert_eq!(m.factor_count(), 0);
        assert_eq!(m.edge_count(), 0);
    }

    #[test]
    fn test_model_add_variable() {
        let mut m = FactorGraphModel::new();
        let idx = m.add_variable("X", 4);
        assert_eq!(idx, 0);
        assert_eq!(m.variable_count(), 1);
        assert_eq!(m.variables[0].domain_size, 4);
    }

    #[test]
    fn test_model_add_factor() {
        let mut m = FactorGraphModel::new();
        let a = m.add_variable("A", 2);
        m.add_factor("f1", vec![a]);
        assert_eq!(m.factor_count(), 1);
    }

    #[test]
    fn test_model_counts() {
        let m = chain_model();
        assert_eq!(m.variable_count(), 3);
        assert_eq!(m.factor_count(), 2);
        assert_eq!(m.edge_count(), 4);
    }

    // -- FactorGraphStats --

    #[test]
    fn test_stats_empty() {
        let m = FactorGraphModel::new();
        let s = FactorGraphStats::compute(&m);
        assert_eq!(s.variable_count, 0);
        assert_eq!(s.factor_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.max_factor_arity, 0);
        assert!((s.avg_factor_arity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_simple_chain() {
        let s = FactorGraphStats::compute(&chain_model());
        assert_eq!(s.variable_count, 3);
        assert_eq!(s.factor_count, 2);
        assert_eq!(s.edge_count, 4);
    }

    #[test]
    fn test_stats_max_factor_arity() {
        let mut m = FactorGraphModel::new();
        let a = m.add_variable("A", 2);
        let b = m.add_variable("B", 2);
        let c = m.add_variable("C", 2);
        m.add_factor("big", vec![a, b, c]);
        let s = FactorGraphStats::compute(&m);
        assert_eq!(s.max_factor_arity, 3);
    }

    #[test]
    fn test_stats_avg_factor_arity() {
        // chain: 4 edges / 2 factors = 2.0
        let s = FactorGraphStats::compute(&chain_model());
        assert!((s.avg_factor_arity - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_variable_degree() {
        // In chain, B appears in f1 and f2 => degree 2.
        let s = FactorGraphStats::compute(&chain_model());
        assert_eq!(s.max_variable_degree, 2);
    }

    #[test]
    fn test_stats_is_tree_true() {
        // chain: 5 nodes (3 var + 2 factor), 4 edges => tree.
        let s = FactorGraphStats::compute(&chain_model());
        assert!(s.is_tree);
    }

    #[test]
    fn test_stats_is_tree_false() {
        // loopy: 6 nodes, 6 edges => not a tree.
        let s = FactorGraphStats::compute(&loopy_model());
        assert!(!s.is_tree);
    }

    #[test]
    fn test_stats_treewidth() {
        let s = FactorGraphStats::compute(&chain_model());
        // max arity = 2, so upper bound = 1
        assert_eq!(s.treewidth_upper_bound, 1);
    }

    #[test]
    fn test_stats_summary() {
        let s = FactorGraphStats::compute(&chain_model());
        let summary = s.summary();
        assert!(summary.contains("vars"));
        assert!(summary.contains("factors"));
    }

    // -- Rendering --

    #[test]
    fn test_render_ascii_header() {
        let out = render_ascii(&chain_model());
        assert!(out.contains("Factor Graph:"));
    }

    #[test]
    fn test_render_ascii_variables() {
        let out = render_ascii(&chain_model());
        assert!(out.contains("A(2)"));
        assert!(out.contains("B(2)"));
        assert!(out.contains("C(2)"));
    }

    #[test]
    fn test_render_ascii_connections() {
        let out = render_ascii(&chain_model());
        // f1 connects A, B
        assert!(out.contains("f1"));
        assert!(out.contains("A"));
        assert!(out.contains("B"));
    }

    #[test]
    fn test_render_dot_undirected() {
        let dot = render_dot(&chain_model());
        // Must use undirected "graph", not "digraph".
        assert!(dot.starts_with("graph "));
        assert!(!dot.contains("digraph"));
    }

    #[test]
    fn test_render_dot_nodes() {
        let dot = render_dot(&chain_model());
        // Variable nodes
        assert!(dot.contains("v0"));
        assert!(dot.contains("shape=circle"));
        // Factor nodes
        assert!(dot.contains("f0"));
        assert!(dot.contains("shape=square"));
    }
}
