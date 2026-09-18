//! Lazy evaluation for large EinsumGraphs.
//!
//! Provides deferred tensor computation with memoization and memory-optimal
//! execution ordering via topological analysis.

pub mod executor;
pub mod graph;
pub mod plan;
pub mod tensor;

pub use executor::{LazyExecutor, LazyStats};
pub use graph::LazyEinsumGraph;
pub use plan::{EvaluationPlan, NodeMemoryEstimate};
pub use tensor::LazyTensor;
