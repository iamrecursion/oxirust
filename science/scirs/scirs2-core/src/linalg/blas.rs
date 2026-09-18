//! BLAS (Basic Linear Algebra Subprograms) operations.
//!
//! Re-exports from OxiBLAS for SciRS2 POLICY compliance.

// Re-export BLAS Level 1 operations (vector-vector)
pub use oxiblas_ndarray::blas::{
    asum_ndarray, axpy_ndarray, dot_ndarray, nrm2_ndarray, scal_ndarray,
};

// Re-export BLAS Level 2 operations (matrix-vector)
pub use oxiblas_ndarray::blas::{gemv_ndarray, matvec, matvec_t, Transpose};

// Re-export BLAS Level 3 operations (matrix-matrix)
pub use oxiblas_ndarray::blas::{gemm_ndarray, matmul, matmul_c, matmul_into};

// Re-export matrix norms
pub use oxiblas_ndarray::blas::{frobenius_norm, norm_1, norm_inf, norm_max};

// NOTE: OxiBLAS handles parallelism internally via the "parallel" feature
// No separate _par versions are needed
