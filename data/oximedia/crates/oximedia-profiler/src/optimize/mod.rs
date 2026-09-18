//! Optimization suggestion modules.

pub mod analyze;
pub mod suggest;

pub use analyze::{AnalysisResult, CodeAnalyzer};
pub use suggest::{OptimizationSuggester, Suggestion};
