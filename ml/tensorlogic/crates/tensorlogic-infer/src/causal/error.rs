//! Error type for causal inference computations.
//!
//! Defines [`CausalError`] and its `Display` / `Error` trait implementations.

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during causal inference computations.
#[derive(Debug)]
pub enum CausalError {
    /// A named node was not found in the graph.
    NodeNotFound(String),
    /// A cycle was detected, violating the DAG requirement.
    CycleDetected,
    /// Sample dimension does not match the number of variables.
    DimensionMismatch,
    /// Not enough data for the requested computation.
    InsufficientData,
    /// No directed causal path exists between the specified nodes.
    NoCausalPath,
    /// A numerical computation failed (e.g. division by zero).
    NumericalError(String),
}

impl std::fmt::Display for CausalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CausalError::NodeNotFound(name) => {
                write!(f, "causal: node '{}' not found in graph", name)
            }
            CausalError::CycleDetected => {
                write!(f, "causal: cycle detected — graph is not a DAG")
            }
            CausalError::DimensionMismatch => {
                write!(
                    f,
                    "causal: sample dimension does not match number of variables"
                )
            }
            CausalError::InsufficientData => {
                write!(f, "causal: insufficient data for the requested computation")
            }
            CausalError::NoCausalPath => {
                write!(
                    f,
                    "causal: no directed causal path between the specified nodes"
                )
            }
            CausalError::NumericalError(msg) => {
                write!(f, "causal: numerical error — {}", msg)
            }
        }
    }
}

impl std::error::Error for CausalError {}
