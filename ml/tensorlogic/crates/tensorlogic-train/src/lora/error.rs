//! LoRA-specific error types.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoraError {
    #[error("invalid rank: {0} (must be >= 1 and <= min(d, k))")]
    InvalidRank(usize),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("merge error: {0}")]
    MergeError(String),

    #[error("frozen weights: {0}")]
    FrozenWeights(String),
}

pub type LoraResult<T> = Result<T, LoraError>;
