//! DOT/Graphviz format export for [`PipelineGraph`].
//!
//! Produces a `.dot` file that can be rendered with `dot -Tpng pipeline.dot`
//! or visualised in any Graphviz-compatible tool.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig};
//! use oximedia_pipeline::dot::{DotExporter, DotExportOptions};
//!
//! let graph = PipelineBuilder::new()
//!     .source("input", SourceConfig::File("video.mkv".into()))
//!     .scale(1280, 720)
//!     .hflip()
//!     .sink("output", SinkConfig::File("out.mkv".into()))
//!     .build()
//!     .expect("valid pipeline");
//!
//! let dot = DotExporter::new(DotExportOptions::default()).export(&graph);
//! assert!(dot.contains("digraph"));
//! assert!(dot.contains("input"));
//! assert!(dot.contains("output"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::graph::PipelineGraph;
use crate::node::{NodeId, NodeType};

// ── DotNodeStyle ─────────────────────────────────────────────────────────────

/// Visual style applied to a node shape in the DOT output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DotNodeStyle {
    /// Plain rectangle (default).
    Box,
    /// Rounded rectangle.
    RoundedBox,
    /// Ellipse / oval shape.
    Ellipse,
    /// Diamond (decision / condition node).
    Diamond,
    /// Parallelogram (data input/output).
    Parallelogram,
    /// Hexagon (complex processing node).
    Hexagon,
}

impl DotNodeStyle {
    #[allow(dead_code)]
    fn as_dot_shape(&self) -> &'static str {
        match self {
            DotNodeStyle::Box => "box",
            DotNodeStyle::RoundedBox => "box",
            DotNodeStyle::Ellipse => "ellipse",
            DotNodeStyle::Diamond => "diamond",
            DotNodeStyle::Parallelogram => "parallelogram",
            DotNodeStyle::Hexagon => "hexagon",
        }
    }

    fn extra_attrs(&self) -> &'static str {
        match self {
            DotNodeStyle::RoundedBox => ", style=\"rounded,filled\"",
            DotNodeStyle::Box => ", style=filled",
            _ => ", style=filled",
        }
    }
}

// ── DotExportOptions ─────────────────────────────────────────────────────────

/// Options controlling how the DOT output is generated.
#[derive(Debug, Clone)]
pub struct DotExportOptions {
    /// Name to give the `digraph` declaration (default: `"pipeline"`).
    pub graph_name: String,
    /// Include pad names on edge labels (default: `true`).
    pub show_pad_names: bool,
    /// Include node type information in node labels (default: `true`).
    pub show_node_types: bool,
    /// Color for source nodes (default: `"#a8d8a8"` — light green).
    pub source_color: String,
    /// Color for sink nodes (default: `"#f7c59f"` — light orange).
    pub sink_color: String,
    /// Color for filter nodes (default: `"#b2c9e0"` — light blue).
    pub filter_color: String,
    /// Color for split/merge/null nodes (default: `"#e0c9e0"` — light purple).
    pub special_color: String,
    /// Edge color (default: `"#555555"` — dark grey).
    pub edge_color: String,
    /// Font name used throughout the graph (default: `"Helvetica"`).
    pub font_name: String,
    /// Global rank direction: `"LR"` (left-right) or `"TB"` (top-bottom,
    /// default).
    pub rankdir: String,
    /// Highlight nodes whose IDs are in this list with a bold border.
    pub highlight_nodes: Vec<NodeId>,
}

impl Default for DotExportOptions {
    fn default() -> Self {
        Self {
            graph_name: "pipeline".to_string(),
            show_pad_names: true,
            show_node_types: true,
            source_color: "#a8d8a8".to_string(),
            sink_color: "#f7c59f".to_string(),
            filter_color: "#b2c9e0".to_string(),
            special_color: "#e0c9e0".to_string(),
            edge_color: "#555555".to_string(),
            font_name: "Helvetica".to_string(),
            rankdir: "TB".to_string(),
            highlight_nodes: Vec::new(),
        }
    }
}

// ── DotExporter ──────────────────────────────────────────────────────────────

/// Exports a [`PipelineGraph`] to Graphviz DOT format.
pub struct DotExporter {
    options: DotExportOptions,
}

impl DotExporter {
    /// Create a new `DotExporter` with the given options.
    pub fn new(options: DotExportOptions) -> Self {
        Self { options }
    }

    /// Export `graph` to a DOT-format `String`.
    ///
    /// The returned string is valid DOT syntax that can be rendered with
    /// `dot`, `neato`, `fdp`, etc.
    pub fn export(&self, graph: &PipelineGraph) -> String {
        let mut out = String::with_capacity(512 + graph.node_count() * 128);
        let opts = &self.options;

        let _ = writeln!(out, "digraph {} {{", escape_id(&opts.graph_name));
        let _ = writeln!(out, "  rankdir={};", opts.rankdir);
        let _ = writeln!(out, "  fontname=\"{}\";", opts.font_name);
        let _ = writeln!(out, "  node [fontname=\"{}\"];", opts.font_name);
        let _ = writeln!(
            out,
            "  edge [fontname=\"{}\", color=\"{}\"];",
            opts.font_name, opts.edge_color
        );
        let _ = writeln!(out);

        // ── Sorted node output (stable order for reproducible output) ─────────
        let mut node_ids: Vec<NodeId> = graph.nodes.keys().copied().collect();
        node_ids.sort_by_key(|id| id.to_string());

        for id in &node_ids {
            let spec = match graph.nodes.get(id) {
                Some(s) => s,
                None => continue,
            };

            let (color, style, shape) = match &spec.node_type {
                NodeType::Source(_) => (&opts.source_color, DotNodeStyle::RoundedBox, "box"),
                NodeType::Sink(_) => (&opts.sink_color, DotNodeStyle::RoundedBox, "box"),
                NodeType::Filter(_) => (&opts.filter_color, DotNodeStyle::Box, "box"),
                NodeType::Split => (&opts.special_color, DotNodeStyle::Diamond, "diamond"),
                NodeType::Merge => (&opts.special_color, DotNodeStyle::Diamond, "diamond"),
                NodeType::Null => (&opts.special_color, DotNodeStyle::Ellipse, "ellipse"),
                NodeType::Conditional(_) => (&opts.special_color, DotNodeStyle::Diamond, "diamond"),
            };

            let type_label = if opts.show_node_types {
                match &spec.node_type {
                    NodeType::Source(c) => format!("\\n[source: {}]", source_label(c)),
                    NodeType::Sink(c) => format!("\\n[sink: {}]", sink_label(c)),
                    NodeType::Filter(c) => format!("\\n[filter: {}]", filter_label(c)),
                    NodeType::Split => "\\n[split]".to_string(),
                    NodeType::Merge => "\\n[merge]".to_string(),
                    NodeType::Null => "\\n[null]".to_string(),
                    NodeType::Conditional(cond) => {
                        format!("\\n[if: {}]", escape_label(&cond.description))
                    }
                }
            } else {
                String::new()
            };

            let node_label = format!("{}{}", escape_label(&spec.name), type_label);
            let dot_id = node_dot_id(id);

            let highlight = if opts.highlight_nodes.contains(id) {
                ", penwidth=3, color=\"#cc0000\""
            } else {
                ""
            };

            let _ = writeln!(
                out,
                "  {dot_id} [label=\"{node_label}\", shape={shape}, fillcolor=\"{color}\"{extra}{highlight}];",
                extra = style.extra_attrs(),
            );
        }

        let _ = writeln!(out);

        // ── Edges ─────────────────────────────────────────────────────────────
        for edge in &graph.edges {
            let from = node_dot_id(&edge.from_node);
            let to = node_dot_id(&edge.to_node);

            let label = if opts.show_pad_names
                && (edge.from_pad != "default" || edge.to_pad != "default")
            {
                format!(
                    " [label=\"{} → {}\"]",
                    escape_label(&edge.from_pad),
                    escape_label(&edge.to_pad)
                )
            } else {
                String::new()
            };

            let _ = writeln!(out, "  {from} -> {to}{label};");
        }

        let _ = writeln!(out, "}}");
        out
    }

    /// Export and write to a file at `path`.
    ///
    /// Returns an `io::Error` on failure.
    pub fn export_to_file(
        &self,
        graph: &PipelineGraph,
        path: &std::path::Path,
    ) -> std::io::Result<()> {
        let content = self.export(graph);
        std::fs::write(path, content)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn node_dot_id(id: &NodeId) -> String {
    // DOT identifiers must not start with a digit; prefix with 'n'
    format!("n_{}", id.to_string().replace('-', "_"))
}

fn escape_id(s: &str) -> String {
    // Wrap in double quotes to handle special characters
    format!("\"{}\"", s.replace('"', "\\\""))
}

fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn source_label(config: &crate::node::SourceConfig) -> String {
    match config {
        crate::node::SourceConfig::File(p) => truncate_path(p, 32),
        crate::node::SourceConfig::Network(u) => truncate_path(u, 32),
        crate::node::SourceConfig::Synthetic(s) => match s {
            crate::node::SyntheticSource::Silence { .. } => "silence".to_string(),
            crate::node::SyntheticSource::BlackFrame { .. } => "black".to_string(),
            crate::node::SyntheticSource::TestPattern { .. } => "pattern".to_string(),
        },
    }
}

fn sink_label(config: &crate::node::SinkConfig) -> String {
    match config {
        crate::node::SinkConfig::File(p) => truncate_path(p, 32),
        crate::node::SinkConfig::Null => "null".to_string(),
        crate::node::SinkConfig::Memory(k) => format!("mem:{k}"),
    }
}

fn filter_label(config: &crate::node::FilterConfig) -> String {
    match config {
        crate::node::FilterConfig::Scale { width, height } => format!("scale {width}×{height}"),
        crate::node::FilterConfig::Crop { x, y, w, h } => format!("crop {w}×{h}@{x},{y}"),
        crate::node::FilterConfig::Trim { start_ms, end_ms } => {
            format!("trim {start_ms}–{end_ms}ms")
        }
        crate::node::FilterConfig::Volume { gain_db } => format!("vol {gain_db:+.1}dB"),
        crate::node::FilterConfig::Fps { fps } => format!("fps {fps}"),
        crate::node::FilterConfig::Format(f) => format!("fmt {f}"),
        crate::node::FilterConfig::Overlay => "overlay".to_string(),
        crate::node::FilterConfig::Concat { count } => format!("concat×{count}"),
        crate::node::FilterConfig::Pad { width, height } => format!("pad {width}×{height}"),
        crate::node::FilterConfig::Hflip => "hflip".to_string(),
        crate::node::FilterConfig::Vflip => "vflip".to_string(),
        crate::node::FilterConfig::Transpose(r) => format!("transpose {r}"),
        crate::node::FilterConfig::Custom { name, .. } => name.clone(),
        crate::node::FilterConfig::Parametric { base, .. } => {
            format!("{}+params", filter_label(base))
        }
    }
}

fn truncate_path(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("…{}", &s[s.len().saturating_sub(max - 1)..])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::node::{FilterConfig, FrameFormat, NodeSpec, SinkConfig, SourceConfig, StreamSpec};

    fn video_spec() -> StreamSpec {
        StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
    }

    #[test]
    fn dot_export_basic_structure() {
        let graph = PipelineBuilder::new()
            .source("input", SourceConfig::File("video.mkv".into()))
            .scale(1280, 720)
            .sink("output", SinkConfig::File("out.mkv".into()))
            .build()
            .expect("valid pipeline");

        let dot = DotExporter::new(DotExportOptions::default()).export(&graph);

        assert!(
            dot.contains("digraph"),
            "should contain digraph declaration"
        );
        assert!(dot.contains("rankdir=TB"), "should have rankdir");
        assert!(dot.contains("input"), "should contain input node label");
        assert!(dot.contains("output"), "should contain output node label");
        assert!(dot.contains("scale"), "should contain scale node label");
        assert!(dot.contains("->"), "should have edges");
    }

    #[test]
    fn dot_export_contains_node_count() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("a.mp4".into()))
            .hflip()
            .vflip()
            .sink("snk", SinkConfig::Null)
            .build()
            .expect("valid pipeline");

        let dot = DotExporter::new(DotExportOptions::default()).export(&graph);
        // 4 nodes means 4 node declarations
        let node_declarations = dot.matches("[label=").count();
        assert_eq!(node_declarations, 4, "should have 4 node declarations");
    }

    #[test]
    fn dot_export_no_pad_names_option() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("a.mp4".into()))
            .sink("snk", SinkConfig::Null)
            .build()
            .expect("valid pipeline");

        let opts = DotExportOptions {
            show_pad_names: false,
            show_node_types: false,
            ..DotExportOptions::default()
        };
        let dot = DotExporter::new(opts).export(&graph);
        assert!(!dot.contains("→"), "should not show pad names");
    }

    #[test]
    fn dot_export_left_right_rankdir() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("a.mp4".into()))
            .sink("snk", SinkConfig::Null)
            .build()
            .expect("valid pipeline");

        let opts = DotExportOptions {
            rankdir: "LR".to_string(),
            ..DotExportOptions::default()
        };
        let dot = DotExporter::new(opts).export(&graph);
        assert!(dot.contains("rankdir=LR"), "should use LR direction");
    }

    #[test]
    fn dot_export_custom_graph_name() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("a.mp4".into()))
            .sink("snk", SinkConfig::Null)
            .build()
            .expect("valid pipeline");

        let opts = DotExportOptions {
            graph_name: "my_pipeline".to_string(),
            ..DotExportOptions::default()
        };
        let dot = DotExporter::new(opts).export(&graph);
        assert!(dot.contains("my_pipeline"), "should use custom graph name");
    }

    #[test]
    fn dot_export_highlighted_node() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("a.mp4".into()))
            .sink("snk", SinkConfig::Null)
            .build()
            .expect("valid pipeline");

        let src_id = graph.source_nodes().into_iter().next().expect("has source");

        let opts = DotExportOptions {
            highlight_nodes: vec![src_id],
            ..DotExportOptions::default()
        };
        let dot = DotExporter::new(opts).export(&graph);
        assert!(
            dot.contains("penwidth=3"),
            "should have penwidth for highlighted node"
        );
        assert!(dot.contains("#cc0000"), "should have red highlight color");
    }

    #[test]
    fn dot_export_source_sink_colors_present() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("a.mp4".into()))
            .sink("snk", SinkConfig::Null)
            .build()
            .expect("valid pipeline");

        let opts = DotExportOptions::default();
        let dot = DotExporter::new(opts.clone()).export(&graph);
        assert!(dot.contains(&opts.source_color), "should have source color");
        assert!(dot.contains(&opts.sink_color), "should have sink color");
    }

    #[test]
    fn dot_export_branch_pipeline() {
        let mut g = crate::graph::PipelineGraph::new();
        let vs = video_spec();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), vs.clone());
        let split = NodeSpec::new(
            "split",
            crate::node::NodeType::Split,
            vec![("default".to_string(), vs.clone())],
            vec![
                ("out0".to_string(), vs.clone()),
                ("out1".to_string(), vs.clone()),
            ],
        );
        let sink1 = NodeSpec::sink("out_hd", SinkConfig::File("hd.mp4".into()), vs.clone());
        let sink2 = NodeSpec::sink("out_sd", SinkConfig::File("sd.mp4".into()), vs);
        let s = g.add_node(src);
        let sp = g.add_node(split);
        let sk1 = g.add_node(sink1);
        let sk2 = g.add_node(sink2);
        let _ = g.connect(s, "default", sp, "default");
        let _ = g.connect(sp, "out0", sk1, "default");
        let _ = g.connect(sp, "out1", sk2, "default");

        let dot = DotExporter::new(DotExportOptions::default()).export(&g);
        assert!(dot.contains("split"), "should contain split node");
        assert!(dot.contains("out_hd"), "should contain hd sink");
        assert!(dot.contains("out_sd"), "should contain sd sink");
        // 2 edge labels from split pads
        assert!(dot.contains("out0"), "should show out0 pad");
        assert!(dot.contains("out1"), "should show out1 pad");
    }

    #[test]
    fn dot_export_filter_labels() {
        // Build graph manually using audio stream throughout so Volume connects cleanly.
        let mut g = crate::graph::PipelineGraph::new();
        let aus =
            crate::node::StreamSpec::audio(crate::node::FrameFormat::S16Interleaved, 48000, 2);
        let src =
            crate::node::NodeSpec::source("src", SourceConfig::File("a.mp4".into()), aus.clone());
        let vol = crate::node::NodeSpec::filter(
            "vol_boost",
            FilterConfig::Volume { gain_db: 6.0 },
            aus.clone(),
            aus.clone(),
        );
        let snk = crate::node::NodeSpec::sink("snk", SinkConfig::Null, aus.clone());
        let s = g.add_node(src);
        let v = g.add_node(vol);
        let sk = g.add_node(snk);
        let _ = g.connect(s, "default", v, "default");
        let _ = g.connect(v, "default", sk, "default");

        let dot = DotExporter::new(DotExportOptions::default()).export(&g);
        assert!(dot.contains("vol"), "should include vol in filter label");
    }

    #[test]
    fn dot_export_to_file() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("a.mp4".into()))
            .sink("snk", SinkConfig::Null)
            .build()
            .expect("valid pipeline");

        let tmp = std::env::temp_dir().join("oximedia_pipeline_test.dot");
        let exporter = DotExporter::new(DotExportOptions::default());
        exporter
            .export_to_file(&graph, &tmp)
            .expect("should write file");
        let content = std::fs::read_to_string(&tmp).expect("should read file");
        assert!(
            content.contains("digraph"),
            "file should contain DOT content"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn dot_export_empty_graph() {
        let g = crate::graph::PipelineGraph::new();
        let dot = DotExporter::new(DotExportOptions::default()).export(&g);
        assert!(
            dot.contains("digraph"),
            "empty graph should still have digraph"
        );
        // The DOT format closes with a single `}`
        assert!(dot.contains('}'), "should have closing brace");
    }
}
