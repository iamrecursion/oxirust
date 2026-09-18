//! Graph visualization: DOT export and ASCII rendering.
//!
//! Converts [`EinsumGraph`] into visual representations for debugging and documentation.
//! Builds on top of `tensorlogic_ir::dot_export` while adding CLI-specific features
//! such as configurable ASCII rendering, graph summary statistics, and file I/O.
//!
//! # Example
//!
//! ```rust,no_run
//! use tensorlogic_cli::visualization::{
//!     AsciiRenderer, DotExporter, GraphSummary, VisualizationConfig,
//! };
//! use tensorlogic_ir::{EinsumGraph, EinsumNode};
//!
//! let mut graph = EinsumGraph::new();
//! let t0 = graph.add_tensor("x".to_string());
//! let t1 = graph.add_tensor("y".to_string());
//! let node = EinsumNode::elem_unary("relu", t0, t1);
//! graph.add_node(node).expect("should add node");
//!
//! let config = VisualizationConfig::default();
//! let dot = DotExporter::export(&graph, &config);
//! let ascii = AsciiRenderer::render(&graph, &config);
//! let summary = GraphSummary::compute(&graph);
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use tensorlogic_ir::{DotExportOptions, EinsumGraph, OpType};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for graph visualization.
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Show operation details (spec strings, etc.)
    pub show_details: bool,
    /// Show tensor shapes in node labels
    pub show_shapes: bool,
    /// Maximum depth for ASCII rendering (0 = unlimited)
    pub max_depth: usize,
    /// Use colour in DOT output
    pub use_color: bool,
    /// Indent string for ASCII rendering
    pub indent: String,
    /// Show tensor indices alongside names
    pub show_tensor_ids: bool,
    /// Show node indices alongside operation labels
    pub show_node_ids: bool,
    /// Use horizontal (LR) layout in DOT
    pub horizontal_layout: bool,
    /// Cluster operations by type in DOT
    pub cluster_by_operation: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        VisualizationConfig {
            show_details: true,
            show_shapes: true,
            max_depth: 0,
            use_color: true,
            indent: "  ".to_string(),
            show_tensor_ids: false,
            show_node_ids: true,
            horizontal_layout: false,
            cluster_by_operation: false,
        }
    }
}

impl VisualizationConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: toggle detail display.
    pub fn with_details(mut self, v: bool) -> Self {
        self.show_details = v;
        self
    }

    /// Builder: toggle shape display.
    pub fn with_shapes(mut self, v: bool) -> Self {
        self.show_shapes = v;
        self
    }

    /// Builder: set maximum ASCII rendering depth.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Builder: toggle colour in DOT output.
    pub fn with_color(mut self, v: bool) -> Self {
        self.use_color = v;
        self
    }

    /// Builder: toggle tensor id display.
    pub fn with_tensor_ids(mut self, v: bool) -> Self {
        self.show_tensor_ids = v;
        self
    }

    /// Builder: toggle node id display.
    pub fn with_node_ids(mut self, v: bool) -> Self {
        self.show_node_ids = v;
        self
    }

    /// Builder: toggle horizontal layout.
    pub fn with_horizontal_layout(mut self, v: bool) -> Self {
        self.horizontal_layout = v;
        self
    }

    /// Builder: toggle operation clustering.
    pub fn with_clustering(mut self, v: bool) -> Self {
        self.cluster_by_operation = v;
        self
    }

    /// A minimal preset that hides most detail.
    pub fn minimal() -> Self {
        VisualizationConfig {
            show_details: false,
            show_shapes: false,
            show_tensor_ids: false,
            show_node_ids: false,
            ..Self::default()
        }
    }

    /// Convert to the upstream `DotExportOptions`.
    fn to_dot_options(&self) -> DotExportOptions {
        DotExportOptions {
            show_tensor_ids: self.show_tensor_ids,
            show_node_ids: self.show_node_ids,
            show_metadata: self.show_details,
            show_shapes: self.show_shapes,
            cluster_by_operation: self.cluster_by_operation,
            horizontal_layout: self.horizontal_layout,
            highlight_tensors: Vec::new(),
            highlight_nodes: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// DOT exporter
// ---------------------------------------------------------------------------

/// Export an [`EinsumGraph`] to Graphviz DOT format.
///
/// This wraps the lower-level `tensorlogic_ir::export_to_dot_with_options` with
/// the CLI-level [`VisualizationConfig`] and optionally strips colour attributes
/// when `use_color` is `false`.
pub struct DotExporter;

impl DotExporter {
    /// Export a graph to a DOT format string.
    pub fn export(graph: &EinsumGraph, config: &VisualizationConfig) -> String {
        let options = config.to_dot_options();
        let dot = tensorlogic_ir::export_to_dot_with_options(graph, &options);

        if config.use_color {
            dot
        } else {
            Self::strip_fill_colors(&dot)
        }
    }

    /// Remove `fillcolor=...` and `style=filled` from a DOT string so that
    /// the output is monochrome.
    fn strip_fill_colors(dot: &str) -> String {
        let mut result = String::with_capacity(dot.len());
        for line in dot.lines() {
            let cleaned = line
                .replace(", style=filled", "")
                .replace("style=filled, ", "")
                .replace("style=filled", "");

            // Remove fillcolor=<word>
            let cleaned = strip_attr(&cleaned, "fillcolor=");
            // Remove trailing ", ]" artifacts that may remain
            let cleaned = cleaned.replace(", ];", "];").replace(",];", "];");
            let _ = writeln!(result, "{}", cleaned);
        }
        result
    }
}

/// Remove a DOT attribute of the form `key=value` (unquoted single word).
fn strip_attr(line: &str, prefix: &str) -> String {
    if let Some(start) = line.find(prefix) {
        let before = &line[..start];
        let after_key = &line[start + prefix.len()..];
        // value ends at comma, space, semicolon, or ']'
        let end = after_key
            .find([',', ';', ']', ' '])
            .unwrap_or(after_key.len());
        let rest = &after_key[end..];
        // Trim a leading ", " from rest
        let rest = rest.strip_prefix(", ").unwrap_or(rest);
        let rest = rest.strip_prefix(',').unwrap_or(rest);
        format!("{}{}", before.trim_end_matches(", "), rest)
    } else {
        line.to_string()
    }
}

/// Write DOT output to a file.
pub fn write_dot_file(
    path: &std::path::Path,
    graph: &EinsumGraph,
    config: &VisualizationConfig,
) -> std::io::Result<()> {
    let dot = DotExporter::export(graph, config);
    std::fs::write(path, dot)
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

/// Render an [`EinsumGraph`] as ASCII art for terminal display.
pub struct AsciiRenderer;

impl AsciiRenderer {
    /// Render a graph to an ASCII string.
    pub fn render(graph: &EinsumGraph, config: &VisualizationConfig) -> String {
        let mut out = String::new();

        let _ = writeln!(out, "=== EinsumGraph ===");
        let _ = writeln!(out, "Nodes: {}", graph.nodes.len());
        let _ = writeln!(
            out,
            "Tensors: {} ({} inputs, {} outputs)",
            graph.tensors.len(),
            graph.inputs.len(),
            graph.outputs.len()
        );

        // List output tensor names
        if !graph.outputs.is_empty() {
            let names: Vec<&str> = graph
                .outputs
                .iter()
                .filter_map(|&idx| graph.tensors.get(idx).map(|s| s.as_str()))
                .collect();
            let _ = writeln!(out, "Outputs: [{}]", names.join(", "));
        }

        let _ = writeln!(out);

        // Render each node
        let depth_limit = if config.max_depth == 0 {
            usize::MAX
        } else {
            config.max_depth
        };

        for (i, node) in graph.nodes.iter().enumerate() {
            if i >= depth_limit {
                let _ = writeln!(
                    out,
                    "{}... ({} more nodes)",
                    config.indent,
                    graph.nodes.len() - i
                );
                break;
            }
            Self::render_node(&mut out, graph, node, i, config);
        }

        let _ = writeln!(out, "===================");
        out
    }

    fn render_node(
        out: &mut String,
        graph: &EinsumGraph,
        node: &tensorlogic_ir::EinsumNode,
        idx: usize,
        config: &VisualizationConfig,
    ) {
        let indent = &config.indent;
        let _ = write!(out, "{}[{}] ", indent, idx);

        // Operation description
        let _ = writeln!(out, "{}", node.operation_description());

        if config.show_details {
            // Inputs
            let input_names: Vec<String> = node
                .inputs
                .iter()
                .map(|&i| {
                    graph
                        .tensors
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("?{}", i))
                })
                .collect();
            let _ = writeln!(
                out,
                "{}{} inputs: [{}]",
                indent,
                indent,
                input_names.join(", ")
            );

            // Outputs
            let output_names: Vec<String> = node
                .outputs
                .iter()
                .map(|&i| {
                    graph
                        .tensors
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("?{}", i))
                })
                .collect();
            let _ = writeln!(
                out,
                "{}{} outputs: [{}]",
                indent,
                indent,
                output_names.join(", ")
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Graph summary / statistics
// ---------------------------------------------------------------------------

/// Lightweight statistics computed from an [`EinsumGraph`].
#[derive(Debug, Clone)]
pub struct GraphSummary {
    /// Number of computation nodes.
    pub node_count: usize,
    /// Number of named tensors.
    pub tensor_count: usize,
    /// Number of graph outputs.
    pub output_count: usize,
    /// Number of graph inputs.
    pub input_count: usize,
    /// Maximum fan-in (number of inputs) across all nodes.
    pub max_fan_in: usize,
    /// Maximum fan-out (number of outputs) across all nodes.
    pub max_fan_out: usize,
    /// Longest path through the dataflow graph (in nodes).
    pub depth: usize,
    /// Operation type distribution.
    pub op_counts: HashMap<String, usize>,
}

impl GraphSummary {
    /// Compute summary statistics for the given graph.
    pub fn compute(graph: &EinsumGraph) -> Self {
        let node_count = graph.nodes.len();
        let tensor_count = graph.tensors.len();
        let output_count = graph.outputs.len();
        let input_count = graph.inputs.len();

        let max_fan_in = graph
            .nodes
            .iter()
            .map(|n| n.inputs.len())
            .max()
            .unwrap_or(0);
        let max_fan_out = graph
            .nodes
            .iter()
            .map(|n| n.outputs.len())
            .max()
            .unwrap_or(0);

        let mut op_counts: HashMap<String, usize> = HashMap::new();
        for node in &graph.nodes {
            let key = match &node.op {
                OpType::Einsum { .. } => "Einsum",
                OpType::ElemUnary { .. } => "ElemUnary",
                OpType::ElemBinary { .. } => "ElemBinary",
                OpType::Reduce { .. } => "Reduce",
            };
            *op_counts.entry(key.to_string()).or_insert(0) += 1;
        }

        let depth = Self::compute_depth(graph);

        GraphSummary {
            node_count,
            tensor_count,
            output_count,
            input_count,
            max_fan_in,
            max_fan_out,
            depth,
            op_counts,
        }
    }

    /// Compute the longest path through the dataflow graph using topological
    /// traversal. Each node is assigned a depth equal to 1 + max depth of any
    /// node producing one of its input tensors.
    fn compute_depth(graph: &EinsumGraph) -> usize {
        if graph.nodes.is_empty() {
            return 0;
        }

        // Build a map: tensor_idx -> producing node index
        let mut producer: HashMap<usize, usize> = HashMap::new();
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &out_t in &node.outputs {
                producer.insert(out_t, node_idx);
            }
        }

        // Memo for node depths
        let num_nodes = graph.nodes.len();
        let mut memo: Vec<Option<usize>> = vec![None; num_nodes];

        fn depth_of(
            node_idx: usize,
            graph: &EinsumGraph,
            producer: &HashMap<usize, usize>,
            memo: &mut [Option<usize>],
            visited: &mut HashSet<usize>,
        ) -> usize {
            if let Some(d) = memo[node_idx] {
                return d;
            }
            // Cycle guard
            if !visited.insert(node_idx) {
                return 0;
            }
            let node = &graph.nodes[node_idx];
            let mut max_pred = 0usize;
            for &inp_t in &node.inputs {
                if let Some(&pred_node) = producer.get(&inp_t) {
                    let d = depth_of(pred_node, graph, producer, memo, visited);
                    if d + 1 > max_pred {
                        max_pred = d + 1;
                    }
                }
            }
            memo[node_idx] = Some(max_pred);
            max_pred
        }

        let mut max_depth = 0usize;
        for i in 0..num_nodes {
            let mut visited = HashSet::new();
            let d = depth_of(i, graph, &producer, &mut memo, &mut visited);
            if d > max_depth {
                max_depth = d;
            }
        }

        // depth counts edges; add 1 so a single-node graph has depth 1
        max_depth + 1
    }

    /// Pretty-print the summary.
    pub fn display(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Graph Summary:");
        let _ = writeln!(out, "  Nodes:   {}", self.node_count);
        let _ = writeln!(out, "  Tensors: {}", self.tensor_count);
        let _ = writeln!(out, "  Inputs:  {}", self.input_count);
        let _ = writeln!(out, "  Outputs: {}", self.output_count);
        let _ = writeln!(out, "  Depth:   {}", self.depth);
        let _ = writeln!(out, "  Max fan-in:  {}", self.max_fan_in);
        let _ = writeln!(out, "  Max fan-out: {}", self.max_fan_out);
        if !self.op_counts.is_empty() {
            let _ = writeln!(out, "  Operations:");
            let mut sorted: Vec<_> = self.op_counts.iter().collect();
            sorted.sort_by_key(|(k, _)| (*k).clone());
            for (op, count) in sorted {
                let _ = writeln!(out, "    {}: {}", op, count);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumGraph, EinsumNode};

    /// Helper: build an empty graph.
    fn empty_graph() -> EinsumGraph {
        EinsumGraph::new()
    }

    /// Helper: build a small graph with two nodes.
    fn small_graph() -> EinsumGraph {
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("a".to_string());
        let b = g.add_tensor("b".to_string());
        let c = g.add_tensor("c".to_string());
        let d = g.add_tensor("d".to_string());
        g.inputs = vec![a, b];
        g.outputs = vec![d];
        g.add_node(EinsumNode::elem_binary("add", a, b, c))
            .expect("node add");
        g.add_node(EinsumNode::elem_unary("relu", c, d))
            .expect("node relu");
        g
    }

    // -----------------------------------------------------------------------
    // DOT export tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_export_empty_graph() {
        let g = empty_graph();
        let dot = DotExporter::export(&g, &VisualizationConfig::default());
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_dot_export_contains_nodes() {
        let g = small_graph();
        let dot = DotExporter::export(&g, &VisualizationConfig::default());
        assert!(dot.contains("op_0"));
        assert!(dot.contains("op_1"));
    }

    #[test]
    fn test_dot_export_contains_edges() {
        let g = small_graph();
        let dot = DotExporter::export(&g, &VisualizationConfig::default());
        // a->add, b->add
        assert!(dot.contains("tensor_0 -> op_0"));
        assert!(dot.contains("tensor_1 -> op_0"));
        // add->c
        assert!(dot.contains("op_0 -> tensor_2"));
        // c->relu
        assert!(dot.contains("tensor_2 -> op_1"));
        // relu->d
        assert!(dot.contains("op_1 -> tensor_3"));
    }

    #[test]
    fn test_dot_export_no_color() {
        let g = small_graph();
        let config = VisualizationConfig::new().with_color(false);
        let dot = DotExporter::export(&g, &config);
        // fillcolor attributes should be stripped
        assert!(!dot.contains("fillcolor"));
    }

    #[test]
    fn test_dot_export_minimal_config() {
        let g = small_graph();
        let full = DotExporter::export(&g, &VisualizationConfig::default());
        let minimal = DotExporter::export(&g, &VisualizationConfig::minimal());
        // Minimal should still be valid DOT but may be shorter (no node ids etc.)
        assert!(minimal.contains("digraph"));
        assert!(minimal.len() <= full.len());
    }

    #[test]
    fn test_write_dot_file() {
        let g = small_graph();
        let dir = std::env::temp_dir();
        let path = dir.join("tensorlogic_test_viz.dot");
        write_dot_file(&path, &g, &VisualizationConfig::default()).expect("should write file");
        let contents = std::fs::read_to_string(&path).expect("should read file");
        assert!(contents.contains("digraph"));
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // ASCII renderer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ascii_render_header() {
        let g = empty_graph();
        let ascii = AsciiRenderer::render(&g, &VisualizationConfig::default());
        assert!(ascii.starts_with("=== EinsumGraph ==="));
    }

    #[test]
    fn test_ascii_render_node_count() {
        let g = small_graph();
        let ascii = AsciiRenderer::render(&g, &VisualizationConfig::default());
        assert!(ascii.contains("Nodes: 2"));
    }

    #[test]
    fn test_ascii_render_output_count() {
        let g = small_graph();
        let ascii = AsciiRenderer::render(&g, &VisualizationConfig::default());
        // Output tensor is "d"
        assert!(ascii.contains("Outputs: [d]"));
    }

    #[test]
    fn test_ascii_render_details() {
        let g = small_graph();
        let config = VisualizationConfig::new().with_details(true);
        let ascii = AsciiRenderer::render(&g, &config);
        assert!(ascii.contains("inputs:"));
        assert!(ascii.contains("outputs:"));
    }

    #[test]
    fn test_ascii_render_no_details() {
        let g = small_graph();
        let with_details =
            AsciiRenderer::render(&g, &VisualizationConfig::new().with_details(true));
        let without = AsciiRenderer::render(&g, &VisualizationConfig::new().with_details(false));
        assert!(without.len() < with_details.len());
        assert!(!without.contains("inputs:"));
    }

    // -----------------------------------------------------------------------
    // Config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let c = VisualizationConfig::default();
        assert!(c.show_details);
        assert!(c.show_shapes);
        assert_eq!(c.max_depth, 0);
        assert!(c.use_color);
        assert_eq!(c.indent, "  ");
    }

    #[test]
    fn test_config_builder() {
        let c = VisualizationConfig::new()
            .with_details(false)
            .with_shapes(false)
            .with_max_depth(5)
            .with_color(false);
        assert!(!c.show_details);
        assert!(!c.show_shapes);
        assert_eq!(c.max_depth, 5);
        assert!(!c.use_color);
    }

    #[test]
    fn test_config_minimal() {
        let c = VisualizationConfig::minimal();
        assert!(!c.show_details);
        assert!(!c.show_shapes);
        assert!(!c.show_tensor_ids);
        assert!(!c.show_node_ids);
    }

    // -----------------------------------------------------------------------
    // Graph summary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_summary_empty() {
        let g = empty_graph();
        let s = GraphSummary::compute(&g);
        assert_eq!(s.node_count, 0);
        assert_eq!(s.tensor_count, 0);
        assert_eq!(s.output_count, 0);
        assert_eq!(s.input_count, 0);
        assert_eq!(s.max_fan_in, 0);
        assert_eq!(s.max_fan_out, 0);
        assert_eq!(s.depth, 0);
    }

    #[test]
    fn test_graph_summary_basic() {
        let g = small_graph();
        let s = GraphSummary::compute(&g);
        assert_eq!(s.node_count, 2);
        assert_eq!(s.tensor_count, 4);
        assert_eq!(s.output_count, 1);
        assert_eq!(s.input_count, 2);
        assert_eq!(s.max_fan_in, 2); // binary add has 2 inputs
        assert_eq!(s.max_fan_out, 1);
        assert_eq!(s.depth, 2); // add -> relu chain
        assert_eq!(s.op_counts.get("ElemBinary"), Some(&1));
        assert_eq!(s.op_counts.get("ElemUnary"), Some(&1));
    }

    // -----------------------------------------------------------------------
    // Determinism tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_deterministic() {
        let g = small_graph();
        let config = VisualizationConfig::default();
        let a = DotExporter::export(&g, &config);
        let b = DotExporter::export(&g, &config);
        assert_eq!(a, b);
    }

    #[test]
    fn test_ascii_deterministic() {
        let g = small_graph();
        let config = VisualizationConfig::default();
        let a = AsciiRenderer::render(&g, &config);
        let b = AsciiRenderer::render(&g, &config);
        assert_eq!(a, b);
    }
}
