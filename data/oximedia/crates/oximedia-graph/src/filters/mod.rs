//! Filter implementations for the filter graph.
//!
//! This module provides various filter implementations that can be used
//! in a filter graph pipeline. Filters are organized by media type:
//!
//! - [`video`] - Video processing filters
//! - [`audio`] - Audio processing filters
//!
//! # Example
//!
//! ```
//! use oximedia_graph::filters::video::{PassthroughFilter, NullSink};
//! use oximedia_graph::node::NodeId;
//!
//! // Create a simple passthrough filter
//! let filter = PassthroughFilter::new(NodeId(0), "passthrough");
//!
//! // Create a null sink for benchmarking
//! let sink = NullSink::new(NodeId(0), "null_sink");
//! ```

pub mod audio;
pub mod video;

// Re-export commonly used filters
pub use audio::AudioPassthrough;
pub use video::{MergeFilter, NullSink, PassthroughFilter, RateLimitFilter, SplitFilter};
