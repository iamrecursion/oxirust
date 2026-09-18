//! Entropy coding optimization module.
//!
//! Context modeling optimization for CABAC/CAVLC.

pub mod context;

pub use context::{ContextModel, ContextOptimizer, EntropyStats};
