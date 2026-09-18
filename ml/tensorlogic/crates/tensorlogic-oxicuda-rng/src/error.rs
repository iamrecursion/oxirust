//! Error types for the tensorlogic-oxicuda-rng crate.

/// Errors that can arise from RNG operations.
#[derive(Debug, thiserror::Error)]
pub enum RngError {
    /// The output buffer was empty; nothing to fill.
    #[error("empty output buffer")]
    EmptyBuffer,

    /// A caller-supplied parameter was invalid (e.g. negative std-dev, p outside \[0,1\]).
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    /// A GPU-side error (driver, memory allocation, kernel launch, or stream failure).
    #[error("GPU RNG error: {0}")]
    GpuError(String),
}
