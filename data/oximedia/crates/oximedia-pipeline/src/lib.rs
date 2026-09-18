//! Declarative media processing pipeline DSL for OxiMedia.
//!
//! This crate provides a typed filter graph, node composition, and execution
//! planning for building media processing pipelines.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig, StreamSpec, FrameFormat};
//!
//! let graph = PipelineBuilder::new()
//!     .source("input", SourceConfig::File("video.mkv".into()))
//!     .scale(1280, 720)
//!     .hflip()
//!     .sink("output", SinkConfig::File("out.mkv".into()))
//!     .build()
//!     .expect("pipeline should validate");
//!
//! assert!(graph.node_count() >= 3);
//! ```

pub mod backpressure;
pub mod builder;
pub mod composition;
pub mod conditional;
pub mod diff;
pub mod dot;
pub mod dot_export;
pub mod dynamic_reconfig;
pub mod execution_plan;
pub mod format_negotiation;
pub mod graph;
pub mod hardware_validator;
pub mod memory_pool;
pub mod metrics;
pub mod node;
pub mod optimizer;
pub mod pipeline_debugger;
pub mod pipeline_profiler;
pub mod preset_library;
pub mod profiler;
pub mod replay;
#[cfg(feature = "serde")]
pub mod serialization;
pub mod simd_scheduler;
pub mod templates;
#[cfg(test)]
pub mod topo_tests;
pub mod validation;
pub mod zero_copy;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur while constructing, validating, or planning a pipeline.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PipelineError {
    /// The pipeline graph contains a cycle and cannot be topologically sorted.
    ///
    /// The `path` field carries the ordered sequence of node names (or IDs as
    /// strings) that form the cycle, e.g. `["A", "B", "C", "A"]`.
    #[error("cycle detected in pipeline graph: {}", path.join(" → "))]
    CycleDetected {
        /// Ordered list of node names forming the detected cycle.
        ///
        /// The last element repeats the first to make the cycle explicit:
        /// `["scale", "hflip", "scale"]`.
        path: Vec<String>,
    },

    /// A referenced node was not found in the graph.
    #[error("node not found: {0}")]
    NodeNotFound(String),

    /// A referenced pad was not found on the specified node.
    #[error("pad not found: node={node}, pad={pad}")]
    PadNotFound {
        /// The node name or id.
        node: String,
        /// The pad name.
        pad: String,
    },

    /// Two connected streams have incompatible types (e.g. video to audio).
    #[error("incompatible streams")]
    IncompatibleStreams,

    /// A generic validation error with a descriptive message.
    #[error("validation error: {0}")]
    ValidationError(String),

    /// An error during pipeline construction.
    #[error("build error: {0}")]
    BuildError(String),

    /// An error from the pipeline profiler.
    #[error("profiler error: {0}")]
    ProfilerError(String),

    /// An error from the pipeline metrics collector.
    #[error("metrics error: {0}")]
    MetricsError(String),

    /// An error produced by a pipeline template builder.
    #[error("template error: {0}")]
    TemplateError(String),
}

impl PipelineError {
    /// Construct a `CycleDetected` error from a list of node names forming the cycle.
    pub fn cycle(path: Vec<String>) -> Self {
        PipelineError::CycleDetected { path }
    }
}

// ── Re-exports ───────────────────────────────────────────────────────────────

pub use builder::{NodeChain, PipelineBuilder};
pub use conditional::{
    ConditionalBranch, PipelineCondition, PipelineContext, PipelineDsl, PipelineOp,
};
pub use dot::{DotExportOptions, DotExporter};
pub use execution_plan::{
    ExecutionPlan, ExecutionPlanner, ExecutionStage, NodeState, PipelineCheckpoint,
    PipelineOptimizer, ResourceEstimate,
};
pub use graph::{Edge, PipelineGraph};
pub use metrics::{BufferStats, LatencyStats, NodeMetrics, PipelineMetrics, ThroughputStats};
pub use node::{
    ConditionOp, FilterConfig, FrameFormat, IfNode, NodeId, NodeSpec, NodeType, PadId, SinkConfig,
    SourceConfig, StreamKind, StreamSpec, SyntheticSource,
};
pub use profiler::{NodeProfilingSummary, NodeTimingSample, PipelineProfiler, ProfilingReport};
#[cfg(feature = "serde")]
pub use serialization::{PipelineDeserializer, PipelineSerializer, SerializationError};
pub use validation::{PipelineValidator, ValidationError, ValidationReport, ValidationWarning};
