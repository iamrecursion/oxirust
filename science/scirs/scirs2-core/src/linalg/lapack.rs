//! LAPACK (Linear Algebra PACKage) operations.
//!
//! Re-exports from OxiBLAS for SciRS2 POLICY compliance.

// Re-export decomposition result types
pub use oxiblas_ndarray::lapack::{
    CholeskyResult, GeneralEvdResult, LuResult, QrResult, RandomizedSvdResult, SchurResult,
    SvdResult, SymEvdResult,
};

// Re-export error types
pub use oxiblas_ndarray::lapack::{LapackError, LapackResult};

// Re-export eigenvalue type for general EVD (used for conversion)
pub use oxiblas_lapack::evd::Eigenvalue;

// Re-export LU decomposition
pub use oxiblas_ndarray::lapack::lu_ndarray;

// Re-export QR decomposition
pub use oxiblas_ndarray::lapack::qr_ndarray;

// Re-export SVD decomposition
pub use oxiblas_ndarray::lapack::{svd_ndarray, svd_truncated};

// Re-export Randomized SVD
pub use oxiblas_ndarray::lapack::{rsvd_ndarray, rsvd_power_ndarray};

// Re-export Cholesky decomposition
pub use oxiblas_ndarray::lapack::cholesky_ndarray;

// Re-export Eigenvalue decomposition
pub use oxiblas_ndarray::lapack::{eig_ndarray, eig_symmetric, eigvals_ndarray, eigvals_symmetric};

// Re-export Schur decomposition
pub use oxiblas_ndarray::lapack::schur_ndarray;

// Re-export linear solvers
pub use oxiblas_ndarray::lapack::{lstsq_ndarray, solve_multiple_ndarray, solve_ndarray};

// Re-export tridiagonal solvers
pub use oxiblas_ndarray::lapack::{
    tridiag_solve_multiple_ndarray, tridiag_solve_ndarray, tridiag_solve_spd_ndarray,
};

// Re-export matrix operations
pub use oxiblas_ndarray::lapack::{
    cond_ndarray, det_ndarray, inv_ndarray, pinv_ndarray, rank_ndarray,
};

// Re-export low-rank approximation
pub use oxiblas_ndarray::lapack::low_rank_approx_ndarray;
