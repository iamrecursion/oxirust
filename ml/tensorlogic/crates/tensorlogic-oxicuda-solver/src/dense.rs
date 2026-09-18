//! Dense linear solvers: LU (with partial pivoting), Cholesky, and QR least-squares.
//!
//! All routines operate on row-major float slices.  No external linear-algebra
//! crates are required: the CPU paths implement Doolittle LU, Cholesky-Banachiewicz,
//! and modified Gram-Schmidt QR in plain Rust, which keeps the dependency surface
//! minimal and avoids the ndarray import that the SciRS2 policy forbids here.
//!
//! Generic cores (`lu_core`, `cholesky_core`, `qr_core`) work for any
//! `scirs2_core::numeric::Float` scalar type.  The public `f32` entry points
//! simply delegate to those cores, and the `f64` variants do the same.
//!
//! # GPU dispatch
//!
//! When compiled with `--features gpu` and `gpu_available()` returns `true`, each
//! public entry point calls the corresponding OxiCUDA GPU kernel.  The GPU paths
//! are scaffolded in Round 5; the full wiring is completed in Round 6.

use scirs2_core::numeric::Float;

use crate::error::SolverError;

// ---------------------------------------------------------------------------
// Small helpers (dot product, matrix–vector multiply)
// ---------------------------------------------------------------------------

/// Dot product of two equal-length `f32` slices (used by the test-only `mat_vec` helper).
#[cfg(test)]
#[inline]
fn dot(u: &[f32], v: &[f32]) -> f32 {
    u.iter()
        .zip(v.iter())
        .fold(0.0f32, |acc, (&a, &b)| acc + a * b)
}

/// Row-major matrix–vector multiply: y = A * x, where A is m×n.
/// Only used in tests for residual verification.
#[cfg(test)]
#[inline]
fn mat_vec(a: &[f32], m: usize, n: usize, x: &[f32], y: &mut [f32]) {
    for (i, y_i) in y.iter_mut().enumerate().take(m) {
        let row_start = i * n;
        *y_i = dot(&a[row_start..row_start + n], &x[..n]);
    }
}

// ---------------------------------------------------------------------------
// Generic helper: convert a numeric literal to T
// ---------------------------------------------------------------------------

/// Convert a `u64` constant to `T`, returning `SolverError::DimMismatch` on failure.
///
/// This is used internally to write numeric literals inside generic functions.
#[inline]
fn cast_val<T: Float>(v: u64, ctx: &'static str) -> Result<T, SolverError> {
    T::from(v).ok_or_else(|| SolverError::DimMismatch(format!("numeric cast failed: {ctx}")))
}

// ---------------------------------------------------------------------------
// Generic LU core (Doolittle decomposition with partial pivoting)
// ---------------------------------------------------------------------------

/// Generic LU solve for any Float scalar type.
///
/// Implements Doolittle factorisation with partial (row-wise) pivoting.
///
/// # Errors
/// Returns [`SolverError::NotSquare`], [`SolverError::DimMismatch`], or
/// [`SolverError::Singular`] as appropriate.
pub(crate) fn lu_core<T>(a: &[T], n: usize, b: &[T]) -> Result<Vec<T>, SolverError>
where
    T: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if a.len() != n * n {
        let rows = a.len() / n.max(1);
        return Err(SolverError::NotSquare { rows, cols: n });
    }
    if b.len() != n {
        return Err(SolverError::DimMismatch(format!(
            "rhs has length {} but matrix is {n}x{n}",
            b.len()
        )));
    }
    if n == 0 {
        return Ok(vec![]);
    }

    let eps_scale: T = cast_val(64, "eps_scale")?;
    let threshold = T::epsilon() * eps_scale;

    let mut lu: Vec<T> = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // find pivot row (row with largest absolute value in column k from row k onward)
        let mut pivot_row = k;
        let mut pivot_abs = lu[k * n + k].abs();
        for r in (k + 1)..n {
            let val_abs = lu[r * n + k].abs();
            if val_abs > pivot_abs {
                pivot_abs = val_abs;
                pivot_row = r;
            }
        }

        if pivot_abs < threshold {
            return Err(SolverError::Singular);
        }

        if pivot_row != k {
            for j in 0..n {
                lu.swap(k * n + j, pivot_row * n + j);
            }
            perm.swap(k, pivot_row);
        }

        let one: T = cast_val(1, "one")?;
        let pivot_inv = one / lu[k * n + k];
        for i in (k + 1)..n {
            lu[i * n + k] *= pivot_inv;
            let mult = lu[i * n + k];
            for j in (k + 1)..n {
                let kj_val = lu[k * n + j];
                lu[i * n + j] -= mult * kj_val;
            }
        }
    }

    // apply permutation to b
    let mut pb: Vec<T> = (0..n).map(|i| b[perm[i]]).collect();

    // forward substitution: L · y = Pb (unit lower triangular)
    for i in 1..n {
        for j in 0..i {
            let lij = lu[i * n + j];
            let yj = pb[j];
            pb[i] -= lij * yj;
        }
    }

    // back substitution: U · x = y
    let mut x = pb;
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let uij = lu[i * n + j];
            let xj = x[j];
            x[i] -= uij * xj;
        }
        let u_ii = lu[i * n + i];
        if u_ii.abs() < threshold {
            return Err(SolverError::Singular);
        }
        let u_ii_inv = {
            let one: T = cast_val(1, "u_ii_inv")?;
            one / u_ii
        };
        x[i] *= u_ii_inv;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Generic Cholesky core (Banachiewicz decomposition, SPD matrices only)
// ---------------------------------------------------------------------------

/// Generic Cholesky solve for any Float scalar type.
///
/// Computes the Cholesky–Banachiewicz decomposition `A = L · Lᵀ` and solves
/// by forward and backward substitution.
///
/// # Errors
/// Returns [`SolverError::Singular`] if A is not positive-definite.
pub(crate) fn cholesky_core<T>(a: &[T], n: usize, b: &[T]) -> Result<Vec<T>, SolverError>
where
    T: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if a.len() != n * n {
        let rows = a.len() / n.max(1);
        return Err(SolverError::NotSquare { rows, cols: n });
    }
    if b.len() != n {
        return Err(SolverError::DimMismatch(format!(
            "rhs has length {} but matrix is {n}x{n}",
            b.len()
        )));
    }
    if n == 0 {
        return Ok(vec![]);
    }

    let mut l: Vec<T> = vec![T::zero(); n * n];

    for j in 0..n {
        let mut diag = a[j * n + j];
        for k in 0..j {
            let l_jk = l[j * n + k];
            diag -= l_jk * l_jk;
        }
        if diag <= T::zero() {
            return Err(SolverError::Singular);
        }
        l[j * n + j] = diag.sqrt();

        let l_jj = l[j * n + j];
        let one: T = cast_val(1, "chol_one")?;
        let l_jj_inv = one / l_jj;
        for i in (j + 1)..n {
            let mut val = a[i * n + j];
            for k in 0..j {
                let lik = l[i * n + k];
                let ljk = l[j * n + k];
                val -= lik * ljk;
            }
            l[i * n + j] = val * l_jj_inv;
        }
    }

    // forward substitution: L · y = b
    let mut y: Vec<T> = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        let one: T = cast_val(1, "chol_fwd")?;
        y[i] = sum * (one / l[i * n + i]);
    }

    // backward substitution: Lᵀ · x = y
    let mut x: Vec<T> = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        let one: T = cast_val(1, "chol_bwd")?;
        x[i] = sum * (one / l[i * n + i]);
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Generic QR core (modified Gram-Schmidt orthogonalisation)
// ---------------------------------------------------------------------------

/// Generic QR least-squares solve for any Float scalar type.
///
/// Computes the minimum-norm least-squares solution to the (possibly
/// overdetermined) m×n system `A · x ≈ b` using modified Gram-Schmidt QR.
///
/// # Errors
/// Returns [`SolverError::Singular`] when the matrix is rank-deficient.
pub(crate) fn qr_core<T>(a: &[T], m: usize, n: usize, b: &[T]) -> Result<Vec<T>, SolverError>
where
    T: Float
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign,
{
    if a.len() != m * n {
        return Err(SolverError::DimMismatch(format!(
            "A has {} elements but m={m}, n={n} implies {}",
            a.len(),
            m * n
        )));
    }
    if b.len() != m {
        return Err(SolverError::DimMismatch(format!(
            "b has length {} but A has {m} rows",
            b.len()
        )));
    }
    if n == 0 || m == 0 {
        return Ok(vec![T::zero(); n]);
    }

    let eps_scale: T = cast_val(64, "qr_eps_scale")?;
    let threshold = T::epsilon() * eps_scale;

    let k = m.min(n);

    let mut q_cols: Vec<Vec<T>> = Vec::with_capacity(k);
    let mut r: Vec<T> = vec![T::zero(); k * n];

    let mut a_cols: Vec<Vec<T>> = (0..n)
        .map(|j| (0..m).map(|i| a[i * n + j]).collect())
        .collect();

    let one: T = cast_val(1, "qr_one")?;

    for j in 0..k {
        let norm_sq = a_cols[j].iter().fold(T::zero(), |acc, &v| acc + v * v);
        let norm_j = norm_sq.sqrt();
        if norm_j < threshold {
            return Err(SolverError::Singular);
        }
        r[j * n + j] = norm_j;

        let inv_norm = one / norm_j;
        let q_j: Vec<T> = a_cols[j].iter().map(|&v| v * inv_norm).collect();

        for jj in (j + 1)..n {
            let r_j_jj = q_j
                .iter()
                .zip(a_cols[jj].iter())
                .fold(T::zero(), |acc, (&qi, &av)| acc + qi * av);
            r[j * n + jj] = r_j_jj;
            for l in 0..m {
                let subtract = q_j[l] * r_j_jj;
                a_cols[jj][l] -= subtract;
            }
        }

        q_cols.push(q_j);
    }

    // Qᵀ · b
    let mut qtb: Vec<T> = vec![T::zero(); k];
    for (i, q_i) in q_cols.iter().enumerate() {
        qtb[i] = q_i
            .iter()
            .zip(b.iter())
            .fold(T::zero(), |acc, (&qi, &bi)| acc + qi * bi);
    }

    // back-substitution on the k×k upper-triangular leading block of R
    let mut x: Vec<T> = vec![T::zero(); n];
    for i in (0..k).rev() {
        let mut sum = qtb[i];
        for j in (i + 1)..k {
            sum -= r[i * n + j] * x[j];
        }
        let r_ii = r[i * n + i];
        if r_ii.abs() < threshold {
            return Err(SolverError::Singular);
        }
        x[i] = sum * (one / r_ii);
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// CPU LU solve — f32 and f64 entry points
// ---------------------------------------------------------------------------

/// Solve the n×n square system `A · x = b` using LU factorisation with partial
/// (row-wise) pivoting (f32 variant).
///
/// # Algorithm
/// Doolittle factorisation with partial pivoting, then forward and backward
/// substitution.
///
/// # Errors
/// Returns [`SolverError::NotSquare`] if `a.len() != n*n` or `b.len() != n`,
/// and [`SolverError::Singular`] if a pivot is below machine epsilon.
pub(crate) fn solve_lu_cpu(a: &[f32], n: usize, b: &[f32]) -> Result<Vec<f32>, SolverError> {
    lu_core::<f32>(a, n, b)
}

/// f64 variant of the LU solver.
///
/// Delegates to [`lu_core::<f64>`].
pub(crate) fn solve_lu_cpu_f64(a: &[f64], n: usize, b: &[f64]) -> Result<Vec<f64>, SolverError> {
    lu_core::<f64>(a, n, b)
}

// ---------------------------------------------------------------------------
// CPU Cholesky solve — f32 and f64 entry points
// ---------------------------------------------------------------------------

/// Solve the n×n SPD system `A · x = b` using Cholesky–Banachiewicz (f32 variant).
///
/// # Errors
/// Returns [`SolverError::Singular`] if A is not positive-definite.
pub(crate) fn solve_cholesky_cpu(a: &[f32], n: usize, b: &[f32]) -> Result<Vec<f32>, SolverError> {
    cholesky_core::<f32>(a, n, b)
}

/// f64 variant of the Cholesky solver.
///
/// Delegates to [`cholesky_core::<f64>`].
pub(crate) fn solve_cholesky_cpu_f64(
    a: &[f64],
    n: usize,
    b: &[f64],
) -> Result<Vec<f64>, SolverError> {
    cholesky_core::<f64>(a, n, b)
}

// ---------------------------------------------------------------------------
// CPU QR least-squares — f32 and f64 entry points
// ---------------------------------------------------------------------------

/// Compute the minimum-norm least-squares solution to the (possibly
/// overdetermined) m×n system `A · x ≈ b` using modified Gram-Schmidt QR (f32 variant).
///
/// # Errors
/// Returns [`SolverError::Singular`] when A is rank-deficient.
pub(crate) fn solve_qr_lstsq_cpu(
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
) -> Result<Vec<f32>, SolverError> {
    qr_core::<f32>(a, m, n, b)
}

/// f64 variant of the QR least-squares solver.
///
/// Delegates to [`qr_core::<f64>`].
pub(crate) fn solve_qr_lstsq_cpu_f64(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
) -> Result<Vec<f64>, SolverError> {
    qr_core::<f64>(a, m, n, b)
}

// ---------------------------------------------------------------------------
// GPU stubs (Round 6 wiring)
// ---------------------------------------------------------------------------

/// GPU-accelerated LU solve.  Stub: returns a [`SolverError::GpuError`] until
/// the full OxiCUDA solver API is wired in Round 6.
#[cfg(feature = "gpu")]
pub(crate) fn solve_lu_gpu(_a: &[f32], _n: usize, _b: &[f32]) -> Result<Vec<f32>, SolverError> {
    Err(SolverError::GpuError(
        "GPU LU solver requires CUDA runtime (wired in Round 6)".to_string(),
    ))
}

/// GPU-accelerated Cholesky solve.  Stub: see [`solve_lu_gpu`].
#[cfg(feature = "gpu")]
pub(crate) fn solve_cholesky_gpu(
    _a: &[f32],
    _n: usize,
    _b: &[f32],
) -> Result<Vec<f32>, SolverError> {
    Err(SolverError::GpuError(
        "GPU Cholesky solver requires CUDA runtime (wired in Round 6)".to_string(),
    ))
}

/// GPU-accelerated QR least-squares.  Stub: see [`solve_lu_gpu`].
#[cfg(feature = "gpu")]
pub(crate) fn solve_qr_lstsq_gpu(
    _a: &[f32],
    _m: usize,
    _n: usize,
    _b: &[f32],
) -> Result<Vec<f32>, SolverError> {
    Err(SolverError::GpuError(
        "GPU QR solver requires CUDA runtime (wired in Round 6)".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Internal tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Helper: max absolute error between two equal-length slices.
    fn max_abs_err(u: &[f32], v: &[f32]) -> f32 {
        u.iter()
            .zip(v.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    // --- LU tests ---

    #[test]
    fn lu_identity_3x3() {
        let a = vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.];
        let b = vec![7f32, -3., 5.];
        let x = solve_lu_cpu(&a, 3, &b).unwrap();
        assert!(max_abs_err(&x, &b) < 1e-5, "x={x:?}");
    }

    #[test]
    fn lu_2x2_known_solution() {
        // A = [[3,1],[1,2]], b = [9,8]
        // Solving: 3x+y=9, x+2y=8 → x=2, y=3
        let a = vec![3f32, 1., 1., 2.];
        let b = vec![9f32, 8.];
        let x = solve_lu_cpu(&a, 2, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-4, "x[1]={}", x[1]);
        // Also verify residual: A*x - b should be ~0
        let ax0 = 3.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 2.0 * x[1];
        assert!((ax0 - 9.0).abs() < 1e-4, "ax0={ax0}");
        assert!((ax1 - 8.0).abs() < 1e-4, "ax1={ax1}");
    }

    #[test]
    fn lu_singular_returns_error() {
        let a = vec![1f32, 2., 2., 4.]; // rank 1
        let b = vec![1f32, 2.];
        assert!(matches!(
            solve_lu_cpu(&a, 2, &b),
            Err(SolverError::Singular)
        ));
    }

    #[test]
    fn lu_4x4_random_ish() {
        // Hand-constructed non-singular 4×4
        #[rustfmt::skip]
        let a = vec![
             4f32, 3., 2., 1.,
             3., 4., 3., 2.,
             2., 3., 4., 3.,
             1., 2., 3., 4.,
        ];
        let b = vec![10f32, 12., 12., 10.];
        let x = solve_lu_cpu(&a, 4, &b).unwrap();
        // Verify: A * x ≈ b
        let mut ax = vec![0.0f32; 4];
        mat_vec(&a, 4, 4, &x, &mut ax);
        assert!(max_abs_err(&ax, &b) < 1e-4, "residual ax={ax:?} vs b={b:?}");
    }

    // --- Cholesky tests ---

    #[test]
    fn cholesky_2x2_spd() {
        // A = [[4,2],[2,3]], b = [6,5] → x = [1,1]
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let x = solve_cholesky_cpu(&a, 2, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-5, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-5, "x[1]={}", x[1]);
    }

    #[test]
    fn cholesky_identity_3x3() {
        let a = vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.];
        let b = vec![2f32, -1., 4.];
        let x = solve_cholesky_cpu(&a, 3, &b).unwrap();
        assert!(max_abs_err(&x, &b) < 1e-5, "x={x:?}");
    }

    #[test]
    fn cholesky_non_spd_returns_singular() {
        // A = [[1,2],[2,1]] has eigenvalues 3 and -1 → not SPD
        let a = vec![1f32, 2., 2., 1.];
        let b = vec![1f32, 1.];
        assert!(matches!(
            solve_cholesky_cpu(&a, 2, &b),
            Err(SolverError::Singular)
        ));
    }

    // --- QR tests ---

    #[test]
    fn qr_square_2x2() {
        // Same SPD system as cholesky test
        let a = vec![4f32, 2., 2., 3.];
        let b = vec![6f32, 5.];
        let x = solve_qr_lstsq_cpu(&a, 2, 2, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }

    #[test]
    fn qr_overdetermined_3x2() {
        // A (3×2) = [[1,0],[0,1],[1,1]], b = [1,1,2] → x = [1,1] is exact
        let a = vec![1f32, 0., 0., 1., 1., 1.];
        let b = vec![1f32, 1., 2.];
        let x = solve_qr_lstsq_cpu(&a, 3, 2, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
    }
}
