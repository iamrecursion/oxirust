//! Prelude for linear algebra operations.
//!
//! Import this module to get access to all commonly used BLAS and LAPACK functions.
//!
//! # Example
//!
//! ```rust,ignore
//! use scirs2_core::linalg::prelude::*;
//!
//! // Now you can use dot_ndarray, matmul, solve_ndarray, etc.
//! ```

// Re-export all BLAS operations
pub use super::blas::*;

// Re-export all LAPACK operations
pub use super::lapack::*;
