//! Banded / tridiagonal linear solvers.
//!
//! Provides the Thomas algorithm (Gaussian elimination specialised for
//! tridiagonal systems) in both f32 and f64 variants.
//!
//! # Thomas Algorithm
//!
//! A tridiagonal system `A · x = rhs` is characterised by three vectors:
//! - `sub[i]`  – sub-diagonal (element at row i, column i-1); `sub[0]` is unused.
//! - `diag[i]` – main diagonal (element at row i, column i).
//! - `sup[i]`  – super-diagonal (element at row i, column i+1); `sup[n-1]` is unused.
//!
//! The Thomas algorithm runs in O(n) time, making it far more efficient than
//! general LU decomposition for tridiagonal problems.

use scirs2_core::numeric::Float;

use crate::error::SolverError;

// ---------------------------------------------------------------------------
// Generic Thomas algorithm implementation
// ---------------------------------------------------------------------------

/// Generic Thomas (tridiagonal LU) solver for any Float scalar type.
///
/// # Errors
/// Returns [`SolverError::DimMismatch`] if the four input slices do not all
/// have the same length n, or [`SolverError::Singular`] if a zero pivot is
/// encountered during the forward sweep.
fn solve_tridiagonal_impl<T>(
    sub: &[T],
    diag: &[T],
    sup: &[T],
    rhs: &[T],
) -> Result<Vec<T>, SolverError>
where
    T: Float,
{
    let n = diag.len();

    // Validate that all slices are consistent with n.
    if sub.len() != n {
        return Err(SolverError::DimMismatch(format!(
            "sub-diagonal has length {} but diag has length {n}",
            sub.len()
        )));
    }
    if sup.len() != n {
        return Err(SolverError::DimMismatch(format!(
            "super-diagonal has length {} but diag has length {n}",
            sup.len()
        )));
    }
    if rhs.len() != n {
        return Err(SolverError::DimMismatch(format!(
            "rhs has length {} but diag has length {n}",
            rhs.len()
        )));
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let eps = T::epsilon();

    // Thomas forward sweep.
    // We maintain two working arrays:
    //   w[i]: the modified diagonal pivot at step i
    //   g[i]: the modified right-hand side at step i
    let mut w: Vec<T> = vec![T::zero(); n];
    let mut g: Vec<T> = vec![T::zero(); n];

    w[0] = diag[0];
    g[0] = rhs[0];

    for i in 1..n {
        if w[i - 1].abs() < eps {
            return Err(SolverError::Singular);
        }
        let m = sub[i] / w[i - 1];
        w[i] = diag[i] - m * sup[i - 1];
        g[i] = rhs[i] - m * g[i - 1];
    }

    // Back substitution.
    let mut x: Vec<T> = vec![T::zero(); n];

    if w[n - 1].abs() < eps {
        return Err(SolverError::Singular);
    }
    x[n - 1] = g[n - 1] / w[n - 1];

    for i in (0..n - 1).rev() {
        if w[i].abs() < eps {
            return Err(SolverError::Singular);
        }
        x[i] = (g[i] - sup[i] * x[i + 1]) / w[i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Public f32 and f64 entry points
// ---------------------------------------------------------------------------

/// Solve the tridiagonal system `A · x = rhs` using the Thomas algorithm (f32 variant).
///
/// `A` is defined by three vectors:
/// - `sub`  – sub-diagonal of length n; `sub[0]` is ignored.
/// - `diag` – main diagonal of length n.
/// - `sup`  – super-diagonal of length n; `sup[n-1]` is ignored.
///
/// All four slices must have the same length n.
///
/// The algorithm runs in O(n) time and O(n) space.
///
/// # Errors
/// - [`SolverError::DimMismatch`] if the slices have inconsistent lengths.
/// - [`SolverError::Singular`] if a zero pivot is encountered.
pub fn solve_tridiagonal(
    sub: &[f32],
    diag: &[f32],
    sup: &[f32],
    rhs: &[f32],
) -> Result<Vec<f32>, SolverError> {
    solve_tridiagonal_impl(sub, diag, sup, rhs)
}

/// Solve the tridiagonal system `A · x = rhs` using the Thomas algorithm (f64 variant).
///
/// See [`solve_tridiagonal`] for full documentation.
///
/// # Errors
/// - [`SolverError::DimMismatch`] if the slices have inconsistent lengths.
/// - [`SolverError::Singular`] if a zero pivot is encountered.
pub fn solve_tridiagonal_f64(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, SolverError> {
    solve_tridiagonal_impl(sub, diag, sup, rhs)
}

// ---------------------------------------------------------------------------
// Internal tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn tridiagonal_identity_3x3() {
        // A = I, rhs = [1,2,3] → x = [1,2,3]
        let sub = vec![0f32, 0., 0.];
        let diag = vec![1f32, 1., 1.];
        let sup = vec![0f32, 0., 0.];
        let rhs = vec![1f32, 2., 3.];
        let x = solve_tridiagonal(&sub, &diag, &sup, &rhs).unwrap();
        for (xi, ri) in x.iter().zip(rhs.iter()) {
            assert!((xi - ri).abs() < 1e-5, "expected {ri} got {xi}");
        }
    }

    #[test]
    fn tridiagonal_singular_returns_error() {
        // diag = [1, 0, 1]: zero pivot at index 1 (or after elimination)
        let sub = vec![0f32, 0., 0.];
        let diag = vec![1f32, 0., 1.];
        let sup = vec![0f32, 0., 0.];
        let rhs = vec![1f32, 1., 1.];
        assert!(matches!(
            solve_tridiagonal(&sub, &diag, &sup, &rhs),
            Err(SolverError::Singular)
        ));
    }
}
