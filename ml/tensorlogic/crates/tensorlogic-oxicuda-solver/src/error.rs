//! Error types for the OxiCUDA solver wrapper.

/// All errors that can be returned by linear solver operations.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    /// The matrix provided was not square (required for LU and Cholesky).
    #[error("matrix is not square: {rows}x{cols}")]
    NotSquare {
        /// Number of rows in the non-square matrix.
        rows: usize,
        /// Number of columns in the non-square matrix.
        cols: usize,
    },

    /// The matrix is singular or numerically near-singular; no unique solution exists.
    #[error("singular or near-singular matrix")]
    Singular,

    /// The conjugate gradient iteration exhausted its budget without achieving the target residual.
    #[error("CG did not converge in {max_iter} iterations (residual: {residual:.2e})")]
    DidNotConverge {
        /// Maximum number of iterations that were allowed before giving up.
        max_iter: usize,
        /// Final residual norm at termination.
        residual: f32,
    },

    /// An error propagated from the OxiCUDA GPU solver subsystem.
    #[error("GPU solver error: {0}")]
    GpuError(String),

    /// The dimensions of the supplied arrays are inconsistent with the requested operation.
    #[error("dimension mismatch: {0}")]
    DimMismatch(String),
}
