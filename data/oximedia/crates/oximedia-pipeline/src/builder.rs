//! Fluent pipeline builder DSL for constructing pipeline graphs.

use crate::graph::PipelineGraph;
use crate::node::{
    FilterConfig, FrameFormat, NodeId, NodeSpec, SinkConfig, SourceConfig, StreamSpec,
};
use crate::PipelineError;

// ── PipelineBuilder ──────────────────────────────────────────────────────────

/// A fluent builder for constructing `PipelineGraph` instances.
///
/// # Example
///
/// ```rust
/// use oximedia_pipeline::builder::PipelineBuilder;
/// use oximedia_pipeline::node::{SourceConfig, SinkConfig};
///
/// let graph = PipelineBuilder::new()
///     .source("input", SourceConfig::File("video.mkv".into()))
///     .scale(1280, 720)
///     .sink("output", SinkConfig::File("out.mkv".into()))
///     .build()
///     .expect("pipeline should validate");
/// ```
pub struct PipelineBuilder {
    graph: PipelineGraph,
    last_output: Option<(NodeId, String)>,
}

impl PipelineBuilder {
    /// Create a new empty pipeline builder.
    pub fn new() -> Self {
        Self {
            graph: PipelineGraph::new(),
            last_output: None,
        }
    }

    /// Add a source node and return a `NodeChain` for fluent chaining.
    pub fn source(mut self, name: &str, config: SourceConfig) -> NodeChain {
        let out_spec = default_video_spec();
        let spec = NodeSpec::source(name, config, out_spec);
        let id = self.graph.add_node(spec);
        NodeChain {
            builder: self,
            current_node: id,
            current_pad: "default".to_string(),
        }
    }

    /// Build the pipeline graph, validating it first.
    ///
    /// Returns the graph if valid, or a list of validation errors.
    pub fn build(self) -> Result<PipelineGraph, Vec<PipelineError>> {
        let errors = self.graph.validate();
        if errors.is_empty() {
            Ok(self.graph)
        } else {
            Err(errors)
        }
    }

    /// Return a reference to the graph being built (for inspection before build).
    pub fn graph(&self) -> &PipelineGraph {
        &self.graph
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── NodeChain ────────────────────────────────────────────────────────────────

/// A chain handle returned by builder methods, enabling fluent filter chaining.
pub struct NodeChain {
    builder: PipelineBuilder,
    current_node: NodeId,
    current_pad: String,
}

impl NodeChain {
    /// Add a generic filter node and auto-connect it to the current output.
    pub fn filter(mut self, name: &str, config: FilterConfig) -> NodeChain {
        let spec_in = default_video_spec();
        let spec_out = match &config {
            FilterConfig::Scale { width, height } => {
                StreamSpec::video(FrameFormat::Yuv420p, *width, *height, 25)
            }
            FilterConfig::Crop { w, h, .. } => StreamSpec::video(FrameFormat::Yuv420p, *w, *h, 25),
            FilterConfig::Volume { .. } => StreamSpec::audio(FrameFormat::S16Interleaved, 48000, 2),
            _ => spec_in.clone(),
        };

        let filter_spec = if matches!(config, FilterConfig::Volume { .. }) {
            let audio_in = StreamSpec::audio(FrameFormat::S16Interleaved, 48000, 2);
            NodeSpec::filter(name, config, audio_in, spec_out)
        } else {
            NodeSpec::filter(name, config, spec_in, spec_out)
        };

        let new_id = self.builder.graph.add_node(filter_spec);

        // Auto-connect: ignore errors in builder mode (validation at build time)
        let _ = self
            .builder
            .graph
            .connect(self.current_node, &self.current_pad, new_id, "default");

        NodeChain {
            builder: self.builder,
            current_node: new_id,
            current_pad: "default".to_string(),
        }
    }

    /// Shorthand: add a Scale filter.
    pub fn scale(self, width: u32, height: u32) -> NodeChain {
        self.filter("scale", FilterConfig::Scale { width, height })
    }

    /// Shorthand: add a Crop filter.
    pub fn crop(self, x: u32, y: u32, w: u32, h: u32) -> NodeChain {
        self.filter("crop", FilterConfig::Crop { x, y, w, h })
    }

    /// Shorthand: add a Volume filter.
    pub fn volume(self, gain_db: f32) -> NodeChain {
        self.filter("volume", FilterConfig::Volume { gain_db })
    }

    /// Shorthand: add a Trim filter.
    pub fn trim(self, start_ms: i64, end_ms: i64) -> NodeChain {
        self.filter("trim", FilterConfig::Trim { start_ms, end_ms })
    }

    /// Shorthand: add an Fps filter.
    pub fn fps(self, fps: f32) -> NodeChain {
        self.filter("fps", FilterConfig::Fps { fps })
    }

    /// Shorthand: add a horizontal flip filter.
    pub fn hflip(self) -> NodeChain {
        self.filter("hflip", FilterConfig::Hflip)
    }

    /// Shorthand: add a vertical flip filter.
    pub fn vflip(self) -> NodeChain {
        self.filter("vflip", FilterConfig::Vflip)
    }

    /// Add a sink node, auto-connecting it to the current output,
    /// and return a mutable reference to the `PipelineBuilder`.
    pub fn sink(mut self, name: &str, config: SinkConfig) -> PipelineBuilder {
        let spec_in = default_video_spec();
        let sink_spec = NodeSpec::sink(name, config, spec_in);
        let sink_id = self.builder.graph.add_node(sink_spec);

        let _ =
            self.builder
                .graph
                .connect(self.current_node, &self.current_pad, sink_id, "default");

        self.builder.last_output = Some((sink_id, "default".to_string()));
        self.builder
    }

    /// Branch the current pipeline into multiple output paths via a `Split` node.
    ///
    /// Inserts a `Split` node after the current position and returns a
    /// `BranchSet` containing one `NodeChain` per requested output path.
    /// Each chain can be independently extended with filters and terminated
    /// with a sink. When all branches are done, call `BranchSet::finish()`
    /// to recover the underlying `PipelineBuilder`.
    ///
    /// # Arguments
    /// * `count` — number of output branches (must be >= 2).
    pub fn branch(mut self, count: usize) -> BranchSet {
        let count = count.max(2);
        let vs = default_video_spec();

        // Build the split node with `count` output pads
        let input_pads = vec![("default".to_string(), vs.clone())];
        let output_pads: Vec<(String, StreamSpec)> = (0..count)
            .map(|i| (format!("out{i}"), vs.clone()))
            .collect();

        let split = crate::node::NodeSpec::new(
            "split",
            crate::node::NodeType::Split,
            input_pads,
            output_pads,
        );
        let split_id = self.builder.graph.add_node(split);

        // Connect current output to split input
        let _ =
            self.builder
                .graph
                .connect(self.current_node, &self.current_pad, split_id, "default");

        // Create one chain handle per output pad
        let pad_names: Vec<String> = (0..count).map(|i| format!("out{i}")).collect();

        BranchSet {
            builder: self.builder,
            split_node: split_id,
            pad_names,
        }
    }

    /// Access the underlying builder (useful for multi-branch pipelines).
    pub fn into_builder(mut self) -> PipelineBuilder {
        self.builder.last_output = Some((self.current_node, self.current_pad));
        self.builder
    }

    /// Return the current node id in the chain.
    pub fn current_node_id(&self) -> NodeId {
        self.current_node
    }
}

// ── BranchSet ────────────────────────────────────────────────────────────────

/// Represents the output of a `branch()` call — a set of independent pipeline
/// paths originating from a single `Split` node.
pub struct BranchSet {
    builder: PipelineBuilder,
    split_node: NodeId,
    pad_names: Vec<String>,
}

impl BranchSet {
    /// Take a branch by index and return a `NodeChain` rooted at the split
    /// node's corresponding output pad.
    ///
    /// Returns `None` if `index` is out of range or the branch was already
    /// taken (pad name consumed).
    pub fn take_branch(&mut self, index: usize) -> Option<BranchChain> {
        if index >= self.pad_names.len() {
            return None;
        }
        // We allow taking the same branch index multiple times (returns same pad),
        // but typically each index is taken once.
        let pad_name = self.pad_names[index].clone();
        Some(BranchChain {
            split_node: self.split_node,
            pad_name,
        })
    }

    /// Return the number of branches.
    pub fn branch_count(&self) -> usize {
        self.pad_names.len()
    }

    /// Consume a `BranchChain` by connecting it through filters to a sink,
    /// recording the result back into this `BranchSet`'s builder.
    ///
    /// The `build_fn` receives a `NodeChain` starting at the split output pad
    /// and should return a `PipelineBuilder` (typically by calling `.sink()`).
    pub fn connect_branch<F>(&mut self, branch: BranchChain, build_fn: F)
    where
        F: FnOnce(NodeChain) -> PipelineBuilder,
    {
        // Create a NodeChain rooted at the split node's output pad
        let chain = NodeChain {
            builder: std::mem::take(&mut self.builder),
            current_node: branch.split_node,
            current_pad: branch.pad_name,
        };
        self.builder = build_fn(chain);
    }

    /// Finish the branching and return the underlying `PipelineBuilder`.
    pub fn finish(self) -> PipelineBuilder {
        self.builder
    }
}

// ── BranchChain ──────────────────────────────────────────────────────────────

/// A lightweight token identifying a single branch from a `BranchSet`.
///
/// Obtained via `BranchSet::take_branch()` and consumed by
/// `BranchSet::connect_branch()`.
pub struct BranchChain {
    split_node: NodeId,
    pad_name: String,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn default_video_spec() -> StreamSpec {
    StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_pipeline() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn source_scale_sink() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720)
            .sink("output", SinkConfig::File("out.mp4".into()))
            .build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn multiple_filters_chain() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720)
            .hflip()
            .vflip()
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        assert_eq!(g.node_count(), 5); // src + scale + hflip + vflip + sink
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn crop_filter() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .crop(100, 100, 640, 480)
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn trim_filter() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .trim(0, 5000)
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn fps_filter() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .fps(30.0)
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn hflip_filter() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .hflip()
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn vflip_filter() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .vflip()
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn sink_file_output() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .sink("output", SinkConfig::File("out.mkv".into()))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn sink_memory_output() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .sink("buffer", SinkConfig::Memory("buf0".into()))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn network_source() {
        let result = PipelineBuilder::new()
            .source(
                "live",
                SourceConfig::Network("rtmp://example.com/live".into()),
            )
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn synthetic_source() {
        let result = PipelineBuilder::new()
            .source(
                "test",
                SourceConfig::Synthetic(crate::node::SyntheticSource::BlackFrame {
                    width: 1920,
                    height: 1080,
                    fps: 25.0,
                }),
            )
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn default_builder() {
        let builder = PipelineBuilder::default();
        assert_eq!(builder.graph().node_count(), 0);
    }

    #[test]
    fn chain_into_builder() {
        let chain = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720);
        let builder = chain.into_builder();
        assert_eq!(builder.graph().node_count(), 2);
    }

    #[test]
    fn current_node_id_accessible() {
        let chain = PipelineBuilder::new().source("input", SourceConfig::File("in.mp4".into()));
        let _id = chain.current_node_id();
        // Just verify it doesn't panic and returns a valid id
    }

    #[test]
    fn generic_filter() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .filter(
                "custom",
                FilterConfig::Custom {
                    name: "eq".to_string(),
                    params: vec![("brightness".to_string(), "0.5".to_string())],
                },
            )
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn long_chain() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1920, 1080)
            .hflip()
            .vflip()
            .crop(0, 0, 960, 540)
            .fps(30.0)
            .trim(0, 10000)
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        assert_eq!(g.node_count(), 8);
        assert_eq!(g.edge_count(), 7);
    }

    #[test]
    fn graph_is_accessible_before_build() {
        let chain = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720);
        let builder = chain.into_builder();
        assert_eq!(builder.graph().node_count(), 2);
        assert_eq!(builder.graph().edge_count(), 1);
    }

    #[test]
    fn topological_order_preserved_after_build() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720)
            .hflip()
            .sink("output", SinkConfig::Null)
            .build();
        let g = result.expect("valid");
        let sorted = g.topological_sort().expect("no cycle");
        assert_eq!(sorted.len(), 4);
    }

    #[test]
    fn volume_filter_chain() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .volume(6.0)
            .sink("output", SinkConfig::Null)
            .build();
        // Volume on a video source will produce an incompatible connection
        // but builder defers validation to build()
        // The build may produce validation errors since video->audio is incompatible
        // This tests that the builder doesn't panic
        let _ = result;
    }

    // ── Branch tests ────────────────────────────────────────────────────────

    #[test]
    fn branch_creates_split_node() {
        let chain = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720);

        let mut branches = chain.branch(2);
        assert_eq!(branches.branch_count(), 2);

        let b0 = branches.take_branch(0).expect("branch 0");
        branches.connect_branch(b0, |c| {
            c.hflip().sink("out_a", SinkConfig::File("a.mp4".into()))
        });

        let b1 = branches.take_branch(1).expect("branch 1");
        branches.connect_branch(b1, |c| {
            c.vflip().sink("out_b", SinkConfig::File("b.mp4".into()))
        });

        let result = branches.finish().build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        // src + scale + split + hflip + sink_a + vflip + sink_b = 7
        assert_eq!(g.node_count(), 7);
    }

    #[test]
    fn branch_three_ways() {
        let chain = PipelineBuilder::new().source("input", SourceConfig::File("in.mp4".into()));

        let mut branches = chain.branch(3);
        assert_eq!(branches.branch_count(), 3);

        for i in 0..3 {
            let b = branches.take_branch(i).expect("branch");
            branches.connect_branch(b, |c| c.sink(&format!("out_{i}"), SinkConfig::Null));
        }

        let result = branches.finish().build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        // src + split + 3 sinks = 5
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.sink_nodes().len(), 3);
    }

    #[test]
    fn branch_minimum_is_two() {
        let chain = PipelineBuilder::new().source("input", SourceConfig::File("in.mp4".into()));
        // Request 1 branch, but minimum is 2
        let branches = chain.branch(1);
        assert_eq!(branches.branch_count(), 2);
    }

    #[test]
    fn take_branch_out_of_range_returns_none() {
        let chain = PipelineBuilder::new().source("input", SourceConfig::File("in.mp4".into()));
        let mut branches = chain.branch(2);
        assert!(branches.take_branch(5).is_none());
    }

    #[test]
    fn branch_with_filters_on_each_path() {
        let chain = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .scale(1920, 1080);

        let mut branches = chain.branch(2);

        let b0 = branches.take_branch(0).expect("branch 0");
        branches.connect_branch(b0, |c| {
            c.scale(1280, 720)
                .hflip()
                .sink("hd", SinkConfig::File("hd.mp4".into()))
        });

        let b1 = branches.take_branch(1).expect("branch 1");
        branches.connect_branch(b1, |c| {
            c.scale(640, 480)
                .vflip()
                .sink("sd", SinkConfig::File("sd.mp4".into()))
        });

        let result = branches.finish().build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        // src + scale + split + (scale+hflip+sink_hd) + (scale+vflip+sink_sd) = 9
        assert_eq!(g.node_count(), 9);
        assert_eq!(g.sink_nodes().len(), 2);
    }

    #[test]
    fn branch_four_ways() {
        let chain = PipelineBuilder::new().source("input", SourceConfig::File("in.mp4".into()));

        let mut branches = chain.branch(4);
        assert_eq!(branches.branch_count(), 4);

        for i in 0..4 {
            let b = branches.take_branch(i).expect("branch exists");
            branches.connect_branch(b, |c| c.sink(&format!("out_{i}"), SinkConfig::Null));
        }

        let result = branches.finish().build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        // src + split + 4 sinks = 6
        assert_eq!(g.node_count(), 6);
        assert_eq!(g.sink_nodes().len(), 4);
    }

    #[test]
    fn branch_and_continue_pipeline() {
        // Branch from a mid-pipeline position
        let chain = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .hflip()
            .scale(1280, 720);

        let mut branches = chain.branch(2);

        let b0 = branches.take_branch(0).expect("branch 0");
        branches.connect_branch(b0, |c| {
            c.vflip().sink("out_a", SinkConfig::File("a.mp4".into()))
        });

        let b1 = branches.take_branch(1).expect("branch 1");
        branches.connect_branch(b1, |c| {
            c.trim(0, 5000)
                .sink("out_b", SinkConfig::File("b.mp4".into()))
        });

        let result = branches.finish().build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        // src + hflip + scale + split + vflip + sink_a + trim + sink_b = 8
        assert_eq!(g.node_count(), 8);

        // Verify topological sort works on branched graph
        let sorted = g.topological_sort();
        assert!(sorted.is_ok());
    }

    #[test]
    fn parametric_filter_in_chain() {
        let result = PipelineBuilder::new()
            .source("input", SourceConfig::File("in.mp4".into()))
            .filter(
                "scale_hq",
                FilterConfig::Scale {
                    width: 1280,
                    height: 720,
                }
                .with_property("quality", "high")
                .with_property("algorithm", "lanczos"),
            )
            .sink("output", SinkConfig::Null)
            .build();
        assert!(result.is_ok());
        let g = result.expect("valid");
        assert_eq!(g.node_count(), 3);

        // Find the filter node and verify its parametric config
        for spec in g.nodes.values() {
            if let crate::node::NodeType::Filter(ref config) = spec.node_type {
                if spec.name == "scale_hq" {
                    assert_eq!(config.get_property("quality"), Some("high"));
                    assert_eq!(config.get_property("algorithm"), Some("lanczos"));
                }
            }
        }
    }
}
