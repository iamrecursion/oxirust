//! Cue point support for seeking optimization.
//!
//! Provides cue point generation and optimization for efficient seeking.

#![forbid(unsafe_code)]

pub mod generator;
pub mod optimizer;

pub use generator::{CueGenerator, CueGeneratorConfig, CuePoint, CueSeeker};
pub use optimizer::{CueOptimizer, CueStats, OptimizerConfig, PlacementStrategy};
