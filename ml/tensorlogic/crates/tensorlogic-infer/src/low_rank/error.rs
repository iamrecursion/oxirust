//! Error types for low-rank approximation.

/// Errors that can arise during low-rank approximation.
#[derive(Debug, thiserror::Error)]
pub enum LowRankError {
    #[error("SVD failed after {iterations} iterations: {reason}")]
    SvdFailed { iterations: usize, reason: String },

    #[error("Rank {rank} exceeds min(rows={rows}, cols={cols})")]
    RankExceedsDimensions {
        rank: usize,
        rows: usize,
        cols: usize,
    },

    #[error("Approximation error {actual:.6} exceeds threshold {threshold:.6}")]
    ErrorExceedsThreshold { actual: f64, threshold: f64 },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}
