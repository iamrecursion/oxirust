//! Multi-pass video analysis and optimization.
//!
//! Analyzes entire video sequences before stabilization to determine optimal
//! parameters and strategies.

pub mod analyze;
pub mod optimize;
pub mod progressive;

pub use analyze::{MultipassAnalysis, MultipassAnalyzer};
pub use optimize::GlobalOptimizer;
pub use progressive::{ProgressiveAnalysisConfig, ProgressiveAnalyzer};
