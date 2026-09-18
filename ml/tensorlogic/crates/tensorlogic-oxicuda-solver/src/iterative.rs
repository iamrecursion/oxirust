//! Iterative solvers: Conjugate Gradient (CG) and Preconditioned CG (PCG).
//!
//! The CPU implementation follows the textbook Hestenes–Stiefel CG algorithm
//! and the left-preconditioned variant for PCG.
//!
//! Supported preconditioners (see [`Precond`]):
//! - [`Precond::Jacobi`]: diagonal scaling M^{-1} v = v ./ diag(A).
//! - [`Precond::IncompleteCholesky`]: IC(0) — for dense SPD matrices this is
//!   the exact Cholesky L, applied via forward + backward substitution.
//!
//! The GPU path is a stub that will be connected to the `oxicuda-solver`
//! sparse-PCG kernel in a future round.

use scirs2_core::numeric::Float;

use crate::error::SolverError;

// ---------------------------------------------------------------------------
// Internal helpers (generic, inline, no allocation)
// ---------------------------------------------------------------------------

/// Dot product ⟨u, v⟩ for any Float scalar.
#[inline]
fn dot_generic<T: Float>(u: &[T], v: &[T]) -> T {
    u.iter()
        .zip(v.iter())
        .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
}

/// Accumulate `dest += scale * src` in-place (generic AXPY).
#[inline]
fn axpy_generic<T: Float>(dest: &mut [T], scale: T, src: &[T]) {
    dest.iter_mut()
        .zip(src.iter())
        .for_each(|(d, &s)| *d = *d + scale * s);
}

/// Dense matrix–vector multiply y = A · x  (A is n×n row-major, x and y are length n).
#[inline]
fn mat_vec_n_generic<T: Float>(a: &[T], n: usize, x: &[T], y: &mut [T]) {
    for i in 0..n {
        let row = &a[i * n..(i + 1) * n];
        y[i] = dot_generic(row, x);
    }
}

// ---------------------------------------------------------------------------
// Preconditioner definition
// ---------------------------------------------------------------------------

/// Preconditioner variant for the PCG solver.
///
/// The preconditioner approximates the inverse of the system matrix A,
/// transforming the iterative system into one with better spectral properties.
#[derive(Copy, Clone)]
pub enum Precond {
    /// Jacobi (diagonal) preconditioner: `M^{-1} v[i] = v[i] / A[i,i]`.
    ///
    /// Effective for diagonally dominant systems.  If a diagonal entry is
    /// near zero (below machine epsilon) it is treated as 1 to avoid
    /// division by zero.
    Jacobi,

    /// Incomplete Cholesky IC(0) preconditioner.
    ///
    /// For a dense SPD matrix, IC(0) with no fill-in positions is equivalent
    /// to the exact Cholesky factor L (Banachiewicz decomposition).  The
    /// preconditioner solves `L · Lᵀ · z = r` via forward + backward
    /// substitution.
    ///
    /// This preconditioner is well-suited to well-conditioned dense SPD systems
    /// where storing the full L factor is acceptable.
    IncompleteCholesky,
}

// ---------------------------------------------------------------------------
// Preconditioner application — generic
// ---------------------------------------------------------------------------

/// Apply the Jacobi preconditioner: z[i] = r[i] / max(|A[i,i]|, ε).
fn apply_jacobi<T: Float>(a: &[T], n: usize, r: &[T], z: &mut [T]) {
    let eps = T::epsilon();
    for i in 0..n {
        let d = a[i * n + i];
        let d_safe = if d.abs() < eps { T::one() } else { d };
        z[i] = r[i] / d_safe;
    }
}

/// Compute the lower-triangular Cholesky factor L for a dense n×n SPD matrix.
///
/// Returns the factor as a flat `Vec<T>` in row-major layout (n×n), with only
/// the lower-triangular entries filled.
///
/// Returns [`SolverError::Singular`] if A is not positive-definite.
fn compute_cholesky_factor<T>(a: &[T], n: usize) -> Result<Vec<T>, SolverError>
where
    T: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    let mut l: Vec<T> = vec![T::zero(); n * n];

    for i in 0..n {
        // diagonal: L[i,i] = sqrt(A[i,i] - sum_{j<i} L[i,j]^2)
        let mut diag = a[i * n + i];
        for j in 0..i {
            let l_ij = l[i * n + j];
            diag -= l_ij * l_ij;
        }
        if diag <= T::zero() {
            return Err(SolverError::Singular);
        }
        l[i * n + i] = diag.sqrt();

        let l_ii = l[i * n + i];
        let l_ii_inv = T::one() / l_ii;

        // sub-diagonal: L[i,k] = (A[i,k] - sum_{j<k} L[i,j]*L[k,j]) / L[k,k]  for k < i
        for k in (i + 1)..n {
            let mut val = a[k * n + i]; // A is symmetric; use lower triangle
            for j in 0..i {
                val -= l[k * n + j] * l[i * n + j];
            }
            l[k * n + i] = val * l_ii_inv;
        }
    }

    Ok(l)
}

/// Apply the IC(0) preconditioner: solve `L · Lᵀ · z = r`.
///
/// Requires a pre-computed lower-triangular factor `l` (row-major, n×n).
fn apply_ichol<T>(l: &[T], n: usize, r: &[T], z: &mut [T])
where
    T: Float,
{
    // forward substitution: L · y = r
    let mut y: Vec<T> = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = r[i];
        for j in 0..i {
            sum = sum - l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i];
    }

    // backward substitution: Lᵀ · z = y
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum = sum - l[j * n + i] * z[j];
        }
        z[i] = sum / l[i * n + i];
    }
}

// ---------------------------------------------------------------------------
// Generic CG core
// ---------------------------------------------------------------------------

/// Generic CG solve for any Float scalar type.
///
/// Implements the Hestenes–Stiefel CG algorithm.
///
/// # Errors
/// - [`SolverError::DimMismatch`] if array lengths are inconsistent.
/// - [`SolverError::Singular`] if the denominator `p · Ap` collapses.
/// - [`SolverError::DidNotConverge`] if `max_iter` is exhausted.
pub(crate) fn cg_core<T>(
    a: &[T],
    n: usize,
    b: &[T],
    max_iter: usize,
    tol: T,
) -> Result<Vec<T>, SolverError>
where
    T: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if a.len() != n * n {
        return Err(SolverError::DimMismatch(format!(
            "A has {} elements but n={n} implies {n}×{n}={}",
            a.len(),
            n * n
        )));
    }
    if b.len() != n {
        return Err(SolverError::DimMismatch(format!(
            "b has length {} but n={n}",
            b.len()
        )));
    }
    if n == 0 {
        return Ok(vec![]);
    }

    let tol_sq = tol * tol;

    let mut x: Vec<T> = vec![T::zero(); n];
    let mut r: Vec<T> = b.to_vec();
    let mut p: Vec<T> = r.clone();

    let mut r_dot_r = dot_generic(&r, &r);

    if r_dot_r < tol_sq {
        return Ok(x);
    }

    let mut ap: Vec<T> = vec![T::zero(); n];

    // Threshold for detecting a near-zero denominator p·Ap, which signals that
    // the matrix is near-singular or not SPD.  We use min_positive_value (the
    // smallest normal float) rather than epsilon*large_constant to avoid false
    // positives on legitimately small-scale problems.
    let near_zero = T::min_positive_value();

    for _iter in 0..max_iter {
        mat_vec_n_generic(a, n, &p, &mut ap);

        let p_dot_ap = dot_generic(&p, &ap);
        if p_dot_ap.abs() < near_zero {
            return Err(SolverError::Singular);
        }
        let alpha = r_dot_r / p_dot_ap;

        axpy_generic(&mut x, alpha, &p);
        axpy_generic(&mut r, -alpha, &ap);

        let r_dot_r_new = dot_generic(&r, &r);

        if r_dot_r_new < tol_sq {
            return Ok(x);
        }

        let beta = r_dot_r_new / r_dot_r;

        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        r_dot_r = r_dot_r_new;
    }

    let residual = r_dot_r.sqrt();
    // Convert residual to f32 for the error type (best-effort)
    let residual_f32 = residual.to_f64().unwrap_or(f64::INFINITY) as f32;
    Err(SolverError::DidNotConverge {
        max_iter,
        residual: residual_f32,
    })
}

// ---------------------------------------------------------------------------
// Generic PCG core
// ---------------------------------------------------------------------------

/// Generic preconditioned CG solve for any Float scalar type.
///
/// Implements the left-preconditioned Hestenes–Stiefel PCG algorithm.
///
/// # Errors
/// - [`SolverError::DimMismatch`] if array lengths are inconsistent.
/// - [`SolverError::Singular`] if the denominator collapses or IC(0) fails.
/// - [`SolverError::DidNotConverge`] if `max_iter` is exhausted.
fn pcg_core<T>(
    a: &[T],
    n: usize,
    b: &[T],
    precond: &Precond,
    max_iter: usize,
    tol: T,
) -> Result<Vec<T>, SolverError>
where
    T: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if a.len() != n * n {
        return Err(SolverError::DimMismatch(format!(
            "A has {} elements but n={n} implies {n}×{n}={}",
            a.len(),
            n * n
        )));
    }
    if b.len() != n {
        return Err(SolverError::DimMismatch(format!(
            "b has length {} but n={n}",
            b.len()
        )));
    }
    if n == 0 {
        return Ok(vec![]);
    }

    // Pre-compute Cholesky factor for IC(0) if needed.
    let ichol_l: Option<Vec<T>> = match precond {
        Precond::IncompleteCholesky => Some(compute_cholesky_factor(a, n)?),
        Precond::Jacobi => None,
    };

    let tol_sq = tol * tol;

    // x = 0, r = b - A*0 = b
    let mut x: Vec<T> = vec![T::zero(); n];
    let mut r: Vec<T> = b.to_vec();

    // z = M^{-1} r
    let mut z: Vec<T> = vec![T::zero(); n];
    match precond {
        Precond::Jacobi => apply_jacobi(a, n, &r, &mut z),
        Precond::IncompleteCholesky => {
            let l = ichol_l
                .as_deref()
                .ok_or_else(|| SolverError::DimMismatch("IC(0) factor missing".to_string()))?;
            apply_ichol(l, n, &r, &mut z);
        }
    }

    let mut p: Vec<T> = z.clone();
    let mut rho = dot_generic(&r, &z);

    // Early exit if already converged
    if dot_generic(&r, &r) < tol_sq {
        return Ok(x);
    }

    let mut ap: Vec<T> = vec![T::zero(); n];

    // Same near-zero guard as in cg_core — catch degenerate denominators.
    let near_zero = T::min_positive_value();

    for _iter in 0..max_iter {
        mat_vec_n_generic(a, n, &p, &mut ap);

        let p_dot_ap = dot_generic(&p, &ap);
        if p_dot_ap.abs() < near_zero {
            return Err(SolverError::Singular);
        }
        let alpha = rho / p_dot_ap;

        axpy_generic(&mut x, alpha, &p);
        axpy_generic(&mut r, -alpha, &ap);

        // Check convergence on residual norm
        if dot_generic(&r, &r) < tol_sq {
            return Ok(x);
        }

        // z = M^{-1} r
        match precond {
            Precond::Jacobi => apply_jacobi(a, n, &r, &mut z),
            Precond::IncompleteCholesky => {
                let l = ichol_l
                    .as_deref()
                    .ok_or_else(|| SolverError::DimMismatch("IC(0) factor missing".to_string()))?;
                apply_ichol(l, n, &r, &mut z);
            }
        }

        let rho_new = dot_generic(&r, &z);
        let beta = rho_new / rho;

        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rho = rho_new;
    }

    let r_norm = dot_generic(&r, &r).sqrt();
    let residual_f32 = r_norm.to_f64().unwrap_or(f64::INFINITY) as f32;
    Err(SolverError::DidNotConverge {
        max_iter,
        residual: residual_f32,
    })
}

// ---------------------------------------------------------------------------
// Public CPU entry points — f32
// ---------------------------------------------------------------------------

/// Solve the n×n symmetric positive-definite system `A · x = b` using the
/// Conjugate Gradient (CG) method (f32 variant).
///
/// # Parameters
/// - `a`        – Row-major n×n coefficient matrix (must be SPD).
/// - `n`        – System dimension.
/// - `b`        – Right-hand side vector (length n).
/// - `max_iter` – Maximum number of CG iterations.
/// - `tol`      – Convergence tolerance on the Euclidean residual norm ‖r‖₂.
///
/// # Errors
/// - [`SolverError::DimMismatch`] if array lengths are inconsistent.
/// - [`SolverError::DidNotConverge`] if `max_iter` is exhausted.
pub fn cg_solve_cpu(
    a: &[f32],
    n: usize,
    b: &[f32],
    max_iter: usize,
    tol: f32,
) -> Result<Vec<f32>, SolverError> {
    cg_core::<f32>(a, n, b, max_iter, tol)
}

/// Solve the n×n SPD system `A · x = b` using Preconditioned CG (f32 variant).
///
/// # Parameters
/// - `a`        – Row-major n×n SPD coefficient matrix.
/// - `n`        – System dimension.
/// - `b`        – Right-hand side vector (length n).
/// - `precond`  – Preconditioner to use (see [`Precond`]).
/// - `max_iter` – Maximum number of PCG iterations.
/// - `tol`      – Convergence tolerance on the Euclidean residual norm ‖r‖₂.
///
/// # Errors
/// - [`SolverError::DimMismatch`] if array lengths are inconsistent.
/// - [`SolverError::Singular`] if the preconditioner encounters a non-positive pivot.
/// - [`SolverError::DidNotConverge`] if `max_iter` is exhausted.
pub fn pcg_solve(
    a: &[f32],
    n: usize,
    b: &[f32],
    precond: Precond,
    max_iter: usize,
    tol: f32,
) -> Result<Vec<f32>, SolverError> {
    pcg_core::<f32>(a, n, b, &precond, max_iter, tol)
}

// ---------------------------------------------------------------------------
// Public CPU entry points — f64
// ---------------------------------------------------------------------------

/// Solve the n×n SPD system `A · x = b` using Conjugate Gradient (f64 variant).
///
/// Delegates to `cg_core::<f64>` internally.
///
/// # Errors
/// See [`cg_solve_cpu`] for the full list of error conditions.
pub fn cg_solve_cpu_f64(
    a: &[f64],
    n: usize,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, SolverError> {
    cg_core::<f64>(a, n, b, max_iter, tol)
}

/// Solve the n×n SPD system `A · x = b` using Preconditioned CG (f64 variant).
///
/// Delegates to `pcg_core::<f64>` internally.
///
/// # Errors
/// See [`pcg_solve`] for the full list of error conditions.
pub fn pcg_solve_f64(
    a: &[f64],
    n: usize,
    b: &[f64],
    precond: Precond,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, SolverError> {
    pcg_core::<f64>(a, n, b, &precond, max_iter, tol)
}

// ---------------------------------------------------------------------------
// GPU stub
// ---------------------------------------------------------------------------

/// GPU conjugate gradient solve.  Stub — wired in Round 6.
#[cfg(feature = "gpu")]
pub(crate) fn cg_solve_gpu(
    _a: &[f32],
    _n: usize,
    _b: &[f32],
    _max_iter: usize,
    _tol: f32,
) -> Result<Vec<f32>, SolverError> {
    Err(SolverError::GpuError(
        "GPU CG solver requires CUDA runtime (wired in Round 6)".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Internal tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn max_abs_err(u: &[f32], v: &[f32]) -> f32 {
        u.iter()
            .zip(v.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn cg_identity_3x3() {
        let a = vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.];
        let b = vec![3f32, -1., 2.];
        let x = cg_solve_cpu(&a, 3, &b, 50, 1e-6).unwrap();
        assert!(max_abs_err(&x, &b) < 1e-4, "x={x:?}");
    }

    #[test]
    fn cg_2x2_spd() {
        // A = [[4,2],[2,3]], b=[6,5], x=[1,1]
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let x = cg_solve_cpu(&a, 2, &b, 100, 1e-6).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }

    #[test]
    fn cg_diagonal_4x4() {
        // Diagonal SPD: diag(1,2,3,4), b=[1,2,3,4] → x=[1,1,1,1]
        #[rustfmt::skip]
        let a = vec![
            1f32, 0., 0., 0.,
            0., 2., 0., 0.,
            0., 0., 3., 0.,
            0., 0., 0., 4.,
        ];
        let b = vec![1f32, 2., 3., 4.];
        let x = cg_solve_cpu(&a, 4, &b, 20, 1e-6).unwrap();
        let expected = vec![1f32, 1., 1., 1.];
        assert!(max_abs_err(&x, &expected) < 1e-4, "x={x:?}");
    }

    #[test]
    fn cg_max_iter_exceeded_returns_error() {
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let result = cg_solve_cpu(&a, 2, &b, 0, 1e-6);
        assert!(
            matches!(result, Err(SolverError::DidNotConverge { .. })),
            "expected DidNotConverge, got {result:?}"
        );
    }

    #[test]
    fn cg_empty_system() {
        let result = cg_solve_cpu(&[], 0, &[], 10, 1e-6).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn pcg_jacobi_2x2_spd() {
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let x = pcg_solve(&a, 2, &b, Precond::Jacobi, 100, 1e-6).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }

    #[test]
    fn pcg_ichol_2x2_spd() {
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let x = pcg_solve(&a, 2, &b, Precond::IncompleteCholesky, 100, 1e-6).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }
}
