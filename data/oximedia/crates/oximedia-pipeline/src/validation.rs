//! Pipeline graph validation — structural correctness checks with detailed
//! error and warning reporting.
//!
//! [`PipelineValidator::validate`] inspects a [`PipelineGraph`] and produces a
//! [`ValidationReport`] that classifies problems as errors (hard failures) or
//! warnings (style / best-practice hints).
//!
//! # Checks performed
//!
//! **Errors**
//! - [`ValidationError::DisconnectedNode`] — a node that is neither a source
//!   nor reachable from any source via directed edges.
//! - [`ValidationError::UnsatisfiedInput`] — a non-source node whose required
//!   input pad is not connected.
//! - [`ValidationError::IncompatibleCodec`] — an edge that links pads carrying
//!   incompatible stream kinds (e.g. video → audio).
//! - [`ValidationError::CyclicDependency`] — the graph contains a directed
//!   cycle (topological sort is impossible).
//!
//! **Warnings**
//! - [`ValidationWarning::UnusedSource`] — a source node whose only output pad
//!   is not connected to any downstream node.
//! - [`ValidationWarning::IsolatedSink`] — a sink node whose input pad is
//!   connected but its predecessor chain leads back to no source.
//! - [`ValidationWarning::DuplicateNodeName`] — two or more nodes share the
//!   same human-readable name.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::graph::PipelineGraph;
//! use oximedia_pipeline::node::{NodeSpec, SourceConfig, SinkConfig, StreamSpec, FrameFormat};
//! use oximedia_pipeline::validation::PipelineValidator;
//!
//! let mut g = PipelineGraph::new();
//! let vs = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
//! let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs.clone());
//! let sink = NodeSpec::sink("sink", SinkConfig::Null, vs);
//! let s = g.add_node(src);
//! let sk = g.add_node(sink);
//! g.connect(s, "default", sk, "default").expect("connect ok");
//!
//! let report = PipelineValidator::new().validate(&g);
//! assert!(report.is_valid);
//! assert!(report.errors.is_empty());
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::PipelineGraph;
use crate::node::{NodeId, NodeType, StreamKind};

// ── ValidationError ───────────────────────────────────────────────────────────

/// A hard error that makes a pipeline graph invalid.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum ValidationError {
    /// A node is not reachable from any source node via directed edges.
    ///
    /// This usually means the node is either floating (not connected to
    /// anything) or is part of a subgraph that has no source attached.
    #[error("disconnected node: '{node_name}'")]
    DisconnectedNode {
        /// The human-readable name of the disconnected node.
        node_name: String,
    },

    /// A non-source node has at least one required input pad that is not
    /// connected to any upstream node.
    #[error("unsatisfied input pad '{pad_name}' on node '{node_name}'")]
    UnsatisfiedInput {
        /// The node that is missing an input connection.
        node_name: String,
        /// The name of the unconnected input pad.
        pad_name: String,
    },

    /// An edge connects pads whose stream kinds are incompatible
    /// (e.g. a video output pad wired to an audio input pad).
    #[error("incompatible codec between '{from_node}' and '{to_node}': {reason}")]
    IncompatibleCodec {
        /// Name of the upstream node.
        from_node: String,
        /// Name of the downstream node.
        to_node: String,
        /// Human-readable explanation of the mismatch.
        reason: String,
    },

    /// The graph contains a directed cycle; topological execution is
    /// impossible.  The `path` field lists the nodes forming the cycle.
    #[error("cyclic dependency detected: {}", path.join(" → "))]
    CyclicDependency {
        /// Ordered list of node names forming the cycle (first == last to
        /// show the loop explicitly).
        path: Vec<String>,
    },
}

// ── ValidationWarning ─────────────────────────────────────────────────────────

/// A non-fatal warning that indicates a potential problem in the pipeline
/// graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationWarning {
    /// A source node has no outgoing edges and therefore produces no output.
    UnusedSource {
        /// Human-readable name of the idle source.
        node_name: String,
    },

    /// A sink node that has its input satisfied but whose upstream chain cannot
    /// be traced back to any recognised source.  This may occur after manual
    /// graph surgery.
    IsolatedSink {
        /// Human-readable name of the isolated sink.
        node_name: String,
    },

    /// Two or more nodes in the graph share the same human-readable `name`.
    /// While not structurally invalid (nodes are keyed by `NodeId`), this
    /// makes debugging output ambiguous.
    DuplicateNodeName {
        /// The name that appears more than once.
        name: String,
    },
}

impl std::fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationWarning::UnusedSource { node_name } => {
                write!(f, "unused source node: '{node_name}'")
            }
            ValidationWarning::IsolatedSink { node_name } => {
                write!(f, "isolated sink node: '{node_name}'")
            }
            ValidationWarning::DuplicateNodeName { name } => {
                write!(f, "duplicate node name: '{name}'")
            }
        }
    }
}

// ── ValidationReport ──────────────────────────────────────────────────────────

/// The result of running [`PipelineValidator::validate`] on a
/// [`PipelineGraph`].
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Hard errors that must be resolved before the pipeline can execute.
    pub errors: Vec<ValidationError>,
    /// Non-fatal warnings (best-practice violations, dead code, etc.).
    pub warnings: Vec<ValidationWarning>,
    /// `true` when [`errors`](Self::errors) is empty.
    pub is_valid: bool,
}

impl ValidationReport {
    /// Returns `true` when there are no errors.
    pub fn is_empty_errors(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns the number of errors.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Returns the number of warnings.
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    /// Returns a human-readable one-line summary.
    pub fn summary(&self) -> String {
        if self.is_valid {
            format!("Pipeline valid ({} warnings)", self.warnings.len())
        } else {
            format!(
                "Pipeline invalid: {} error(s), {} warning(s)",
                self.errors.len(),
                self.warnings.len()
            )
        }
    }
}

// ── PipelineValidator ─────────────────────────────────────────────────────────

/// Validates the structural integrity of a [`PipelineGraph`].
///
/// Create with [`PipelineValidator::new`] and call
/// [`PipelineValidator::validate`].
#[derive(Debug, Default, Clone)]
pub struct PipelineValidator {
    /// When `true`, warnings about duplicate node names are suppressed.
    pub suppress_duplicate_name_warnings: bool,
}

impl PipelineValidator {
    /// Create a new validator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the given `pipeline` graph and return a [`ValidationReport`].
    pub fn validate(&self, pipeline: &PipelineGraph) -> ValidationReport {
        let mut errors: Vec<ValidationError> = Vec::new();
        let mut warnings: Vec<ValidationWarning> = Vec::new();

        // 1. Cycle detection (using the graph's own algorithm which produces
        //    full path information).
        if let Err(cycle_err) = pipeline.detect_cycle() {
            // Convert PipelineError::CycleDetected into our ValidationError.
            if let crate::PipelineError::CycleDetected { path } = cycle_err {
                errors.push(ValidationError::CyclicDependency { path });
            }
        }

        // 2. Unsatisfied input pads — any non-source node that has an input
        //    pad not wired to an upstream output pad.
        for (id, spec) in &pipeline.nodes {
            match &spec.node_type {
                NodeType::Source(_) => {
                    // Sources are allowed to have no inputs.
                }
                _ => {
                    for (pad_name, _) in &spec.input_pads {
                        let connected = pipeline
                            .edges
                            .iter()
                            .any(|e| e.to_node == *id && e.to_pad == *pad_name);
                        if !connected {
                            errors.push(ValidationError::UnsatisfiedInput {
                                node_name: spec.name.clone(),
                                pad_name: pad_name.clone(),
                            });
                        }
                    }
                }
            }
        }

        // 3. Codec compatibility — edges that connect pads with incompatible
        //    stream kinds.
        for edge in &pipeline.edges {
            let from_spec = pipeline.nodes.get(&edge.from_node);
            let to_spec = pipeline.nodes.get(&edge.to_node);

            if let (Some(from), Some(to)) = (from_spec, to_spec) {
                // Find the stream kind produced by the source pad.
                let from_kind = from
                    .output_pads
                    .iter()
                    .find(|(n, _)| n == &edge.from_pad)
                    .map(|(_, s)| s.kind);

                // Find the stream kind expected by the sink pad.
                let to_kind = to
                    .input_pads
                    .iter()
                    .find(|(n, _)| n == &edge.to_pad)
                    .map(|(_, s)| s.kind);

                if let (Some(fk), Some(tk)) = (from_kind, to_kind) {
                    if fk != tk {
                        errors.push(ValidationError::IncompatibleCodec {
                            from_node: from.name.clone(),
                            to_node: to.name.clone(),
                            reason: format!(
                                "output pad '{}' produces {} but input pad '{}' expects {}",
                                edge.from_pad,
                                stream_kind_name(fk),
                                edge.to_pad,
                                stream_kind_name(tk),
                            ),
                        });
                    }
                }
            }
        }

        // 4. Reachability — nodes that cannot be reached from any source via
        //    a forward BFS/DFS.  Disconnected nodes indicate wiring mistakes.
        let reachable = self.compute_reachable(pipeline);
        for (id, spec) in &pipeline.nodes {
            if !reachable.contains(id) {
                // Sources are trivially reachable from themselves; anything
                // else that isn't reachable is a disconnected non-source.
                if !matches!(spec.node_type, NodeType::Source(_)) {
                    errors.push(ValidationError::DisconnectedNode {
                        node_name: spec.name.clone(),
                    });
                }
            }
        }

        // 5. Warnings: unused source nodes (no outgoing edges).
        let has_outgoing: HashSet<NodeId> = pipeline.edges.iter().map(|e| e.from_node).collect();

        for (id, spec) in &pipeline.nodes {
            if matches!(spec.node_type, NodeType::Source(_)) && !has_outgoing.contains(id) {
                warnings.push(ValidationWarning::UnusedSource {
                    node_name: spec.name.clone(),
                });
            }
        }

        // 6. Warnings: isolated sink nodes (not reachable from any source,
        //    but have at least one connected input pad).
        let has_incoming: HashSet<NodeId> = pipeline.edges.iter().map(|e| e.to_node).collect();

        for (id, spec) in &pipeline.nodes {
            if matches!(spec.node_type, NodeType::Sink(_))
                && has_incoming.contains(id)
                && !reachable.contains(id)
            {
                warnings.push(ValidationWarning::IsolatedSink {
                    node_name: spec.name.clone(),
                });
            }
        }

        // 7. Warnings: duplicate node names.
        if !self.suppress_duplicate_name_warnings {
            let mut name_counts: HashMap<&str, usize> = HashMap::new();
            for spec in pipeline.nodes.values() {
                *name_counts.entry(spec.name.as_str()).or_insert(0) += 1;
            }
            let mut seen_dup: HashSet<&str> = HashSet::new();
            for (name, count) in &name_counts {
                if *count > 1 && seen_dup.insert(name) {
                    warnings.push(ValidationWarning::DuplicateNodeName {
                        name: (*name).to_string(),
                    });
                }
            }
        }

        let is_valid = errors.is_empty();
        ValidationReport {
            errors,
            warnings,
            is_valid,
        }
    }

    /// Compute the set of `NodeId`s reachable from any source node via
    /// directed forward BFS.
    fn compute_reachable(&self, pipeline: &PipelineGraph) -> HashSet<NodeId> {
        // Start BFS from all source nodes.
        let mut reachable: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        for (id, spec) in &pipeline.nodes {
            if matches!(spec.node_type, NodeType::Source(_)) {
                if reachable.insert(*id) {
                    queue.push_back(*id);
                }
            }
        }

        while let Some(current) = queue.pop_front() {
            for edge in &pipeline.edges {
                if edge.from_node == current && reachable.insert(edge.to_node) {
                    queue.push_back(edge.to_node);
                }
            }
        }

        reachable
    }
}

/// Return a display name for a `StreamKind` value without importing Display.
fn stream_kind_name(kind: StreamKind) -> &'static str {
    match kind {
        StreamKind::Video => "video",
        StreamKind::Audio => "audio",
        StreamKind::Data => "data",
        StreamKind::Subtitle => "subtitle",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{FilterConfig, FrameFormat, NodeSpec, SinkConfig, SourceConfig, StreamSpec};

    fn vs() -> StreamSpec {
        StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
    }

    fn audio_spec() -> StreamSpec {
        StreamSpec::audio(FrameFormat::S16Interleaved, 48000, 2)
    }

    fn validator() -> PipelineValidator {
        PipelineValidator::new()
    }

    // ── Valid graph ──────────────────────────────────────────────────────────

    #[test]
    fn valid_source_sink_graph() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");
        let report = validator().validate(&g);
        assert!(report.is_valid);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn valid_with_filter_chain() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            vs(),
            StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25),
        );
        let sink = NodeSpec::sink(
            "sink",
            SinkConfig::Null,
            StreamSpec::video(FrameFormat::Yuv420p, 1280, 720, 25),
        );
        let s = g.add_node(src);
        let f = g.add_node(filt);
        let sk = g.add_node(sink);
        g.connect(s, "default", f, "default").expect("connect");
        g.connect(f, "default", sk, "default").expect("connect");
        let report = validator().validate(&g);
        assert!(report.is_valid);
    }

    // ── UnsatisfiedInput ─────────────────────────────────────────────────────

    #[test]
    fn detects_unsatisfied_input_pad() {
        let mut g = PipelineGraph::new();
        // Add a sink with no incoming edge
        let sink = NodeSpec::sink("orphan_sink", SinkConfig::Null, vs());
        g.add_node(sink);
        let report = validator().validate(&g);
        assert!(!report.is_valid);
        let has_unsatisfied = report.errors.iter().any(|e| {
            matches!(e, ValidationError::UnsatisfiedInput { node_name, .. } if node_name == "orphan_sink")
        });
        assert!(has_unsatisfied, "expected UnsatisfiedInput error");
    }

    // ── DisconnectedNode ─────────────────────────────────────────────────────

    #[test]
    fn detects_disconnected_filter_node() {
        let mut g = PipelineGraph::new();
        // Source→sink connected; a floating filter not connected to anything useful
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect");

        // Add a disconnected filter (it has an unconnected input AND is
        // unreachable from any source)
        let float_filt = NodeSpec::filter("floating", FilterConfig::Hflip, vs(), vs());
        g.add_node(float_filt);

        let report = validator().validate(&g);
        assert!(!report.is_valid);
        let has_disconnected = report.errors.iter().any(|e| {
            matches!(e, ValidationError::DisconnectedNode { node_name } if node_name == "floating")
        });
        assert!(has_disconnected, "expected DisconnectedNode for 'floating'");
    }

    // ── IncompatibleCodec ────────────────────────────────────────────────────

    #[test]
    fn detects_incompatible_codec_video_to_audio() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        let sink = NodeSpec::sink("audio_sink", SinkConfig::Null, audio_spec());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        // Force-add an incompatible edge by directly pushing (bypassing graph.connect() validation)
        g.edges.push(crate::graph::Edge {
            from_node: s,
            from_pad: "default".to_string(),
            to_node: sk,
            to_pad: "default".to_string(),
        });

        let report = validator().validate(&g);
        let has_incompat = report.errors.iter().any(|e| {
            matches!(e, ValidationError::IncompatibleCodec { from_node, .. } if from_node == "src")
        });
        assert!(has_incompat, "expected IncompatibleCodec error");
    }

    // ── CyclicDependency ─────────────────────────────────────────────────────

    #[test]
    fn detects_cyclic_dependency() {
        let mut g = PipelineGraph::new();
        let a = NodeSpec::filter("a", FilterConfig::Hflip, vs(), vs());
        let b = NodeSpec::filter("b", FilterConfig::Vflip, vs(), vs());
        let a_id = g.add_node(a);
        let b_id = g.add_node(b);
        // Manually add both directions to create a cycle
        g.edges.push(crate::graph::Edge {
            from_node: a_id,
            from_pad: "default".to_string(),
            to_node: b_id,
            to_pad: "default".to_string(),
        });
        g.edges.push(crate::graph::Edge {
            from_node: b_id,
            from_pad: "default".to_string(),
            to_node: a_id,
            to_pad: "default".to_string(),
        });

        let report = validator().validate(&g);
        let has_cycle = report
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::CyclicDependency { .. }));
        assert!(has_cycle, "expected CyclicDependency error");
    }

    // ── UnusedSource warning ─────────────────────────────────────────────────

    #[test]
    fn warns_unused_source() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("idle_src", SourceConfig::File("in.mp4".into()), vs());
        g.add_node(src);
        let report = validator().validate(&g);
        // No errors (source alone is structurally valid) …
        // but there should be an UnusedSource warning.
        let has_warn = report.warnings.iter().any(|w| {
            matches!(w, ValidationWarning::UnusedSource { node_name } if node_name == "idle_src")
        });
        assert!(has_warn, "expected UnusedSource warning");
    }

    // ── DuplicateNodeName warning ────────────────────────────────────────────

    #[test]
    fn warns_duplicate_node_names() {
        let mut g = PipelineGraph::new();
        let src1 = NodeSpec::source("same_name", SourceConfig::File("a.mp4".into()), vs());
        let src2 = NodeSpec::source("same_name", SourceConfig::File("b.mp4".into()), vs());
        let sink1 = NodeSpec::sink("sink1", SinkConfig::Null, vs());
        let sink2 = NodeSpec::sink("sink2", SinkConfig::Null, vs());
        let s1 = g.add_node(src1);
        let s2 = g.add_node(src2);
        let sk1 = g.add_node(sink1);
        let sk2 = g.add_node(sink2);
        g.connect(s1, "default", sk1, "default").expect("connect");
        g.connect(s2, "default", sk2, "default").expect("connect");

        let report = validator().validate(&g);
        let has_dup = report.warnings.iter().any(
            |w| matches!(w, ValidationWarning::DuplicateNodeName { name } if name == "same_name"),
        );
        assert!(has_dup, "expected DuplicateNodeName warning");
    }

    #[test]
    fn suppress_duplicate_name_warning() {
        let mut g = PipelineGraph::new();
        let src1 = NodeSpec::source("dup", SourceConfig::File("a.mp4".into()), vs());
        let src2 = NodeSpec::source("dup", SourceConfig::File("b.mp4".into()), vs());
        let sink1 = NodeSpec::sink("sink1", SinkConfig::Null, vs());
        let sink2 = NodeSpec::sink("sink2", SinkConfig::Null, vs());
        let s1 = g.add_node(src1);
        let s2 = g.add_node(src2);
        let sk1 = g.add_node(sink1);
        let sk2 = g.add_node(sink2);
        g.connect(s1, "default", sk1, "default").expect("connect");
        g.connect(s2, "default", sk2, "default").expect("connect");

        let mut v = PipelineValidator::new();
        v.suppress_duplicate_name_warnings = true;
        let report = v.validate(&g);
        let has_dup = report
            .warnings
            .iter()
            .any(|w| matches!(w, ValidationWarning::DuplicateNodeName { .. }));
        assert!(!has_dup, "duplicate name warning should be suppressed");
    }

    // ── ValidationReport helpers ─────────────────────────────────────────────

    #[test]
    fn report_summary_valid() {
        let report = ValidationReport {
            errors: vec![],
            warnings: vec![],
            is_valid: true,
        };
        let s = report.summary();
        assert!(s.contains("valid"));
    }

    #[test]
    fn report_summary_invalid() {
        let report = ValidationReport {
            errors: vec![ValidationError::DisconnectedNode {
                node_name: "x".into(),
            }],
            warnings: vec![],
            is_valid: false,
        };
        let s = report.summary();
        assert!(s.contains("invalid"));
    }

    #[test]
    fn report_counts() {
        let report = ValidationReport {
            errors: vec![
                ValidationError::DisconnectedNode {
                    node_name: "a".into(),
                },
                ValidationError::DisconnectedNode {
                    node_name: "b".into(),
                },
            ],
            warnings: vec![ValidationWarning::UnusedSource {
                node_name: "c".into(),
            }],
            is_valid: false,
        };
        assert_eq!(report.error_count(), 2);
        assert_eq!(report.warning_count(), 1);
        assert!(!report.is_empty_errors());
    }

    #[test]
    fn empty_graph_is_valid() {
        let g = PipelineGraph::new();
        let report = validator().validate(&g);
        assert!(report.is_valid);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn source_only_graph_valid_with_warning() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs());
        g.add_node(src);
        let report = validator().validate(&g);
        // No structural errors: a lone source is valid (it just doesn't go anywhere).
        assert!(report.is_valid);
        // Should have an UnusedSource warning.
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn multiple_unsatisfied_pads_all_reported() {
        let mut g = PipelineGraph::new();
        // Merge node with two input pads, neither connected
        let vs_spec = vs();
        let merge = crate::node::NodeSpec::new(
            "merge",
            crate::node::NodeType::Merge,
            vec![
                ("in0".to_string(), vs_spec.clone()),
                ("in1".to_string(), vs_spec.clone()),
            ],
            vec![("default".to_string(), vs_spec)],
        );
        g.add_node(merge);
        let report = validator().validate(&g);
        // Should have at least 2 UnsatisfiedInput errors
        let unsatisfied_count = report
            .errors
            .iter()
            .filter(|e| matches!(e, ValidationError::UnsatisfiedInput { .. }))
            .count();
        assert!(
            unsatisfied_count >= 2,
            "expected 2+ UnsatisfiedInput errors, got {unsatisfied_count}"
        );
    }

    #[test]
    fn fan_out_graph_valid() {
        let mut g = PipelineGraph::new();
        let vs_spec = vs();
        let src = NodeSpec::source("src", SourceConfig::File("in.mp4".into()), vs_spec.clone());
        let split = crate::node::NodeSpec::new(
            "split",
            crate::node::NodeType::Split,
            vec![("default".to_string(), vs_spec.clone())],
            vec![
                ("out0".to_string(), vs_spec.clone()),
                ("out1".to_string(), vs_spec.clone()),
            ],
        );
        let sink1 = NodeSpec::sink("sink1", SinkConfig::File("a.mp4".into()), vs_spec.clone());
        let sink2 = NodeSpec::sink("sink2", SinkConfig::File("b.mp4".into()), vs_spec);
        let s = g.add_node(src);
        let sp = g.add_node(split);
        let sk1 = g.add_node(sink1);
        let sk2 = g.add_node(sink2);
        g.connect(s, "default", sp, "default").expect("connect");
        g.connect(sp, "out0", sk1, "default").expect("connect");
        g.connect(sp, "out1", sk2, "default").expect("connect");

        let report = validator().validate(&g);
        assert!(
            report.is_valid,
            "fan-out graph should be valid, errors: {:?}",
            report.errors
        );
    }

    #[test]
    fn validation_error_display() {
        let e = ValidationError::DisconnectedNode {
            node_name: "x".into(),
        };
        let s = e.to_string();
        assert!(s.contains("disconnected") || s.contains("x"));
    }

    #[test]
    fn validation_warning_display() {
        let w = ValidationWarning::UnusedSource {
            node_name: "src".into(),
        };
        let s = w.to_string();
        assert!(s.contains("src"));
    }
}
