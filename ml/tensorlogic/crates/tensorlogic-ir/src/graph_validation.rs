//! EinsumGraph validation and sanitization utilities.
//!
//! Checks graph integrity: missing inputs, duplicate outputs, cycles,
//! unreachable nodes. Provides sanitization to fix common issues.
//!
//! This module provides a complementary validation layer on top of
//! the `crate::graph::validation` module, using a unified issue-based reporting
//! model with severity levels (Error, Warning, Info).

use crate::EinsumGraph;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Critical problem that makes the graph invalid.
    Error,
    /// Non-critical issue that may indicate a problem.
    Warning,
    /// Informational note about the graph structure.
    Info,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Error => write!(f, "ERROR"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Info => write!(f, "INFO"),
        }
    }
}

/// A single validation issue found in a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// How severe this issue is.
    pub severity: IssueSeverity,
    /// Machine-readable issue code (e.g., "empty-graph", "duplicate-output").
    pub code: String,
    /// Human-readable description of the issue.
    pub message: String,
    /// Index of the node related to this issue, if applicable.
    pub node_index: Option<usize>,
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(idx) = self.node_index {
            write!(
                f,
                "[{}] {} (node {}): {}",
                self.severity, self.code, idx, self.message
            )
        } else {
            write!(f, "[{}] {}: {}", self.severity, self.code, self.message)
        }
    }
}

/// Result of graph validation containing all discovered issues.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    /// All issues found during validation.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Returns `true` if the graph has no error-level issues.
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }

    /// Count of error-level issues.
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count()
    }

    /// Count of warning-level issues.
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count()
    }

    /// Count of info-level issues.
    pub fn info_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Info)
            .count()
    }

    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "{} errors, {} warnings",
            self.error_count(),
            self.warning_count()
        )
    }

    /// Filter issues by severity.
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// Filter issues by code prefix.
    pub fn issues_by_code(&self, code: &str) -> Vec<&ValidationIssue> {
        self.issues.iter().filter(|i| i.code == code).collect()
    }
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Validation: {}", self.summary())?;
        for issue in &self.issues {
            writeln!(f, "  {}", issue)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Validation entry point
// ---------------------------------------------------------------------------

/// Validate an `EinsumGraph` for structural integrity.
///
/// Runs all structural checks and returns a unified [`ValidationResult`].
///
/// # Checks performed
///
/// 1. **Empty graph** (warning) -- graph has zero nodes
/// 2. **Duplicate graph outputs** (warning) -- same tensor index listed twice
/// 3. **Node input references** (error) -- node inputs reference non-existent tensors
/// 4. **Node output references** (error) -- node outputs reference non-existent tensors
/// 5. **Unreachable nodes** (warning) -- nodes whose outputs are never consumed
/// 6. **Output references** (error) -- graph outputs reference non-existent tensors
/// 7. **Outputs without producer** (error) -- output tensors not produced by any node or input
/// 8. **Cycle detection** (error) -- data-flow cycles in the graph
/// 9. **Empty node outputs** (error) -- nodes that produce no tensors
/// 10. **Duplicate tensor names** (info) -- two tensors share the same name
pub fn validate_einsum_graph(graph: &EinsumGraph) -> ValidationResult {
    let mut result = ValidationResult::default();

    check_empty_graph(graph, &mut result);
    check_duplicate_outputs(graph, &mut result);
    check_node_input_refs(graph, &mut result);
    check_node_output_refs(graph, &mut result);
    check_unreachable_nodes(graph, &mut result);
    check_output_refs(graph, &mut result);
    check_outputs_have_producers(graph, &mut result);
    check_cycles(graph, &mut result);
    check_empty_node_outputs(graph, &mut result);
    check_duplicate_tensor_names(graph, &mut result);

    result
}

// ---------------------------------------------------------------------------
// Individual checks
// ---------------------------------------------------------------------------

fn check_empty_graph(graph: &EinsumGraph, result: &mut ValidationResult) {
    if graph.nodes.is_empty() {
        result.issues.push(ValidationIssue {
            severity: IssueSeverity::Warning,
            code: "empty-graph".to_string(),
            message: "Graph has no nodes".to_string(),
            node_index: None,
        });
    }
}

fn check_duplicate_outputs(graph: &EinsumGraph, result: &mut ValidationResult) {
    let mut seen = HashSet::new();
    for &output in &graph.outputs {
        if !seen.insert(output) {
            result.issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                code: "duplicate-output".to_string(),
                message: format!("Duplicate output tensor index: {}", output),
                node_index: None,
            });
        }
    }
}

fn check_node_input_refs(graph: &EinsumGraph, result: &mut ValidationResult) {
    let num_tensors = graph.tensors.len();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &input_idx in &node.inputs {
            if input_idx >= num_tensors {
                result.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "invalid-input-ref".to_string(),
                    message: format!(
                        "Node {} input references tensor index {} but only {} tensors exist",
                        node_idx, input_idx, num_tensors
                    ),
                    node_index: Some(node_idx),
                });
            }
        }
    }
}

fn check_node_output_refs(graph: &EinsumGraph, result: &mut ValidationResult) {
    let num_tensors = graph.tensors.len();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &output_idx in &node.outputs {
            if output_idx >= num_tensors {
                result.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "invalid-output-ref".to_string(),
                    message: format!(
                        "Node {} output references tensor index {} but only {} tensors exist",
                        node_idx, output_idx, num_tensors
                    ),
                    node_index: Some(node_idx),
                });
            }
        }
    }
}

fn check_unreachable_nodes(graph: &EinsumGraph, result: &mut ValidationResult) {
    if graph.nodes.is_empty() {
        return;
    }

    // A node is "reachable" if at least one of its outputs is consumed by
    // another node or is a graph output.
    let output_set: HashSet<usize> = graph.outputs.iter().copied().collect();

    // Collect all tensor indices consumed as inputs by any node.
    let mut consumed_tensors: HashSet<usize> = HashSet::new();
    for node in &graph.nodes {
        for &inp in &node.inputs {
            consumed_tensors.insert(inp);
        }
    }

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let any_output_used = node
            .outputs
            .iter()
            .any(|o| consumed_tensors.contains(o) || output_set.contains(o));
        if !any_output_used {
            result.issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                code: "unreachable-node".to_string(),
                message: format!(
                    "Node {} outputs are never consumed and not graph outputs",
                    node_idx
                ),
                node_index: Some(node_idx),
            });
        }
    }
}

fn check_output_refs(graph: &EinsumGraph, result: &mut ValidationResult) {
    let num_tensors = graph.tensors.len();
    for &output_idx in &graph.outputs {
        if output_idx >= num_tensors {
            result.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "invalid-graph-output".to_string(),
                message: format!(
                    "Graph output references tensor index {} but only {} tensors exist",
                    output_idx, num_tensors
                ),
                node_index: None,
            });
        }
    }
}

fn check_outputs_have_producers(graph: &EinsumGraph, result: &mut ValidationResult) {
    // Build set of tensors produced by nodes.
    let mut produced: HashSet<usize> = HashSet::new();
    for node in &graph.nodes {
        for &out in &node.outputs {
            produced.insert(out);
        }
    }

    let input_set: HashSet<usize> = graph.inputs.iter().copied().collect();

    for &output_idx in &graph.outputs {
        if output_idx >= graph.tensors.len() {
            continue; // Already reported in check_output_refs
        }
        if !produced.contains(&output_idx) && !input_set.contains(&output_idx) {
            result.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "output-no-producer".to_string(),
                message: format!(
                    "Output tensor {} ('{}') is not produced by any node and is not a graph input",
                    output_idx, graph.tensors[output_idx]
                ),
                node_index: None,
            });
        }
    }
}

fn check_cycles(graph: &EinsumGraph, result: &mut ValidationResult) {
    if graph.nodes.is_empty() {
        return;
    }

    // Build adjacency: node -> set of successor nodes (via tensor flow).
    let num_nodes = graph.nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];

    // Map tensor index -> producing node index.
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();
    for (nidx, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            tensor_producer.insert(out, nidx);
        }
    }

    // For each node, find successor nodes: nodes that consume tensors this node produces.
    for (nidx, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            for (other_idx, other_node) in graph.nodes.iter().enumerate() {
                if other_idx != nidx && other_node.inputs.contains(&out) {
                    adj[nidx].push(other_idx);
                }
            }
        }
    }

    // Standard DFS cycle detection.
    let mut visited = vec![0u8; num_nodes]; // 0=unvisited, 1=in-stack, 2=done

    for start in 0..num_nodes {
        if visited[start] == 0 && dfs_has_cycle(start, &adj, &mut visited) {
            result.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "cycle-detected".to_string(),
                message: format!("Cyclic dependency detected involving node {}", start),
                node_index: Some(start),
            });
        }
    }
}

fn dfs_has_cycle(node: usize, adj: &[Vec<usize>], visited: &mut [u8]) -> bool {
    visited[node] = 1; // in-stack
    for &next in &adj[node] {
        if visited[next] == 1 {
            return true;
        }
        if visited[next] == 0 && dfs_has_cycle(next, adj, visited) {
            return true;
        }
    }
    visited[node] = 2; // done
    false
}

fn check_empty_node_outputs(graph: &EinsumGraph, result: &mut ValidationResult) {
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        if node.outputs.is_empty() {
            result.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "node-no-outputs".to_string(),
                message: format!("Node {} produces no outputs", node_idx),
                node_index: Some(node_idx),
            });
        }
    }
}

fn check_duplicate_tensor_names(graph: &EinsumGraph, result: &mut ValidationResult) {
    let mut name_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, name) in graph.tensors.iter().enumerate() {
        name_indices.entry(name.as_str()).or_default().push(idx);
    }
    for (name, indices) in &name_indices {
        if indices.len() > 1 {
            result.issues.push(ValidationIssue {
                severity: IssueSeverity::Info,
                code: "duplicate-tensor-name".to_string(),
                message: format!("Tensor name '{}' is used by indices {:?}", name, indices),
                node_index: None,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Graph statistics
// ---------------------------------------------------------------------------

/// Graph statistics computed during validation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSanitizationStats {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Number of graph-level output tensor indices.
    pub output_count: usize,
    /// Number of tensors in the graph.
    pub tensor_count: usize,
    /// Whether the graph contains cycles.
    pub has_cycles: bool,
    /// Number of unreachable nodes.
    pub unreachable_count: usize,
    /// Maximum depth of the data-flow DAG (0 for empty graphs).
    pub max_depth: usize,
}

/// Compute statistics about an `EinsumGraph`.
pub fn compute_graph_stats(graph: &EinsumGraph) -> GraphSanitizationStats {
    let validation = validate_einsum_graph(graph);

    let unreachable_count = validation
        .issues
        .iter()
        .filter(|i| i.code == "unreachable-node")
        .count();

    let has_cycles = validation.issues.iter().any(|i| i.code == "cycle-detected");

    let max_depth = compute_max_depth(graph);

    GraphSanitizationStats {
        node_count: graph.nodes.len(),
        output_count: graph.outputs.len(),
        tensor_count: graph.tensors.len(),
        has_cycles,
        unreachable_count,
        max_depth,
    }
}

/// Compute the maximum depth of the data-flow DAG using BFS from inputs.
fn compute_max_depth(graph: &EinsumGraph) -> usize {
    if graph.nodes.is_empty() {
        return 0;
    }

    // Map tensor -> producing node.
    let mut tensor_producer: HashMap<usize, usize> = HashMap::new();
    for (nidx, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            tensor_producer.insert(out, nidx);
        }
    }

    // Build adjacency: node -> successor nodes.
    let num_nodes = graph.nodes.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
    let mut in_degree: Vec<usize> = vec![0; num_nodes];

    for (nidx, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            for (other_idx, other_node) in graph.nodes.iter().enumerate() {
                if other_idx != nidx && other_node.inputs.contains(&out) {
                    adj[nidx].push(other_idx);
                    in_degree[other_idx] += 1;
                }
            }
        }
    }

    // BFS topological order to compute depth.
    let mut depth = vec![0usize; num_nodes];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut max_d = 0usize;
    while let Some(n) = queue.pop_front() {
        for &next in &adj[n] {
            let new_depth = depth[n] + 1;
            if new_depth > depth[next] {
                depth[next] = new_depth;
            }
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
        if depth[n] > max_d {
            max_d = depth[n];
        }
    }

    max_d
}

// ---------------------------------------------------------------------------
// Sanitization
// ---------------------------------------------------------------------------

/// Sanitize a graph by fixing common issues.
///
/// Currently performs:
/// - Deduplication of graph output indices.
///
/// The returned graph is independent of the input (cloned).
pub fn sanitize_graph(graph: &EinsumGraph) -> EinsumGraph {
    let mut sanitized = graph.clone();

    // Deduplicate output indices, preserving order.
    let mut seen = HashSet::new();
    sanitized.outputs.retain(|o| seen.insert(*o));

    sanitized
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EinsumGraph, EinsumNode};

    /// Helper: build a minimal valid graph (input -> relu -> output).
    fn make_valid_graph() -> EinsumGraph {
        let mut g = EinsumGraph::new();
        let t0 = g.add_tensor("input");
        let t1 = g.add_tensor("output");
        g.inputs = vec![t0];
        g.outputs = vec![t1];
        g.add_node(EinsumNode::elem_unary("relu", t0, t1))
            .expect("failed to add node");
        g
    }

    // 1
    #[test]
    fn test_validate_empty_graph() {
        let graph = EinsumGraph::new();
        let result = validate_einsum_graph(&graph);
        assert!(
            result.is_valid(),
            "empty graph should be valid (only warnings)"
        );
        assert!(
            result.issues.iter().any(|i| i.code == "empty-graph"),
            "should have empty-graph warning"
        );
    }

    // 2
    #[test]
    fn test_validate_valid_graph() {
        let graph = make_valid_graph();
        let result = validate_einsum_graph(&graph);
        assert!(
            result.is_valid(),
            "well-formed graph should be valid: {:?}",
            result
        );
    }

    // 3
    #[test]
    fn test_validate_duplicate_outputs() {
        let mut graph = make_valid_graph();
        // outputs is [1]; push duplicate
        graph.outputs.push(graph.outputs[0]);
        let result = validate_einsum_graph(&graph);
        assert!(
            result.issues.iter().any(|i| i.code == "duplicate-output"),
            "should detect duplicate outputs"
        );
    }

    // 4
    #[test]
    fn test_validate_result_summary() {
        let graph = EinsumGraph::new();
        let result = validate_einsum_graph(&graph);
        let summary = result.summary();
        assert!(
            summary.contains("errors") && summary.contains("warnings"),
            "summary should mention errors and warnings: {}",
            summary
        );
    }

    // 5
    #[test]
    fn test_validate_error_count() {
        // Build a graph where output has no producer and is not an input.
        let mut graph = EinsumGraph::new();
        let t0 = graph.add_tensor("a");
        let _t1 = graph.add_tensor("b");
        graph.outputs = vec![t0]; // t0 not produced, not in inputs
        let result = validate_einsum_graph(&graph);
        assert!(
            result.error_count() >= 1,
            "should have at least one error for output without producer"
        );
    }

    // 6
    #[test]
    fn test_validate_warning_count() {
        let graph = EinsumGraph::new();
        let result = validate_einsum_graph(&graph);
        assert!(
            result.warning_count() >= 1,
            "empty graph should have at least one warning"
        );
    }

    // 7
    #[test]
    fn test_graph_stats_node_count() {
        let graph = make_valid_graph();
        let stats = compute_graph_stats(&graph);
        assert_eq!(stats.node_count, 1);
    }

    // 8
    #[test]
    fn test_graph_stats_output_count() {
        let graph = make_valid_graph();
        let stats = compute_graph_stats(&graph);
        assert_eq!(stats.output_count, 1);
    }

    // 9
    #[test]
    fn test_sanitize_dedup_outputs() {
        let mut graph = make_valid_graph();
        graph.outputs.push(graph.outputs[0]);
        assert_eq!(graph.outputs.len(), 2);
        let sanitized = sanitize_graph(&graph);
        assert_eq!(sanitized.outputs.len(), 1, "duplicates should be removed");
    }

    // 10
    #[test]
    fn test_sanitize_preserves_valid() {
        let graph = make_valid_graph();
        let sanitized = sanitize_graph(&graph);
        assert_eq!(sanitized.tensors, graph.tensors);
        assert_eq!(sanitized.nodes, graph.nodes);
        assert_eq!(sanitized.outputs, graph.outputs);
        assert_eq!(sanitized.inputs, graph.inputs);
    }

    // 11
    #[test]
    fn test_issue_severity_eq() {
        assert_eq!(IssueSeverity::Error, IssueSeverity::Error);
        assert_eq!(IssueSeverity::Warning, IssueSeverity::Warning);
        assert_eq!(IssueSeverity::Info, IssueSeverity::Info);
        assert_ne!(IssueSeverity::Error, IssueSeverity::Warning);
    }

    // 12
    #[test]
    fn test_validation_result_default() {
        let result = ValidationResult::default();
        assert!(result.issues.is_empty());
        assert!(result.is_valid());
    }

    // 13
    #[test]
    fn test_validation_result_is_valid_no_errors() {
        let mut result = ValidationResult::default();
        result.issues.push(ValidationIssue {
            severity: IssueSeverity::Warning,
            code: "test".to_string(),
            message: "just a warning".to_string(),
            node_index: None,
        });
        assert!(result.is_valid(), "warnings only => valid");
    }

    // 14
    #[test]
    fn test_validation_result_is_valid_with_errors() {
        let mut result = ValidationResult::default();
        result.issues.push(ValidationIssue {
            severity: IssueSeverity::Error,
            code: "test-error".to_string(),
            message: "an error".to_string(),
            node_index: None,
        });
        assert!(!result.is_valid(), "errors => not valid");
    }

    // 15
    #[test]
    fn test_graph_stats_default() {
        let stats = GraphSanitizationStats::default();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.output_count, 0);
        assert_eq!(stats.tensor_count, 0);
        assert!(!stats.has_cycles);
        assert_eq!(stats.unreachable_count, 0);
        assert_eq!(stats.max_depth, 0);
    }

    // 16
    #[test]
    fn test_validate_outputs_reference() {
        // Graph outputs reference a tensor that doesn't exist.
        let mut graph = EinsumGraph::new();
        graph.add_tensor("a");
        graph.outputs = vec![999]; // non-existent
        let result = validate_einsum_graph(&graph);
        assert!(
            result
                .issues
                .iter()
                .any(|i| i.code == "invalid-graph-output"),
            "should detect invalid graph output reference"
        );
        assert!(!result.is_valid());
    }

    // 17
    #[test]
    fn test_sanitize_returns_clone() {
        let graph = make_valid_graph();
        let sanitized = sanitize_graph(&graph);
        // Mutating original should not affect sanitized.
        let mut original = graph;
        original.tensors.push("extra".to_string());
        assert_ne!(original.tensors.len(), sanitized.tensors.len());
    }

    // 18
    #[test]
    fn test_compute_stats_empty() {
        let graph = EinsumGraph::new();
        let stats = compute_graph_stats(&graph);
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.output_count, 0);
        assert_eq!(stats.tensor_count, 0);
        assert!(!stats.has_cycles);
        assert_eq!(stats.unreachable_count, 0);
        assert_eq!(stats.max_depth, 0);
    }
}
