//! Signed singular value decomposition of 3×3 matrices.
//!
//! The elasto-plastic MPM models (snow clamping, Drucker-Prager return mapping,
//! corotated stress) all work in the principal frame of the elastic deformation
//! gradient, so they need a *true* SVD whose factors are orthogonal to machine
//! precision even when the matrix is badly conditioned or rank deficient.
//!
//! [`mat3_svd`] uses the one-sided (Hestenes) Jacobi method: plane rotations are
//! applied to the columns of `M` until they are mutually orthogonal. Unlike an
//! eigen-decomposition of `MᵀM` this never squares the condition number, so the
//! left singular vectors stay orthonormal to a few ulps for any non-zero
//! singular value, and the right factor is a product of rotations by
//! construction. Left vectors of (numerically) zero singular values are
//! completed by Gram-Schmidt against the vectors already in place.

use super::functions::{cross3, dot3, len3, mat3_identity, scale3, sub3};

/// A column pair is treated as orthogonal once `|a_p · a_q| <= TOL |a_p| |a_q|`.
const ORTHOGONALITY_TOL: f64 = 8.0 * f64::EPSILON;

/// Upper bound on Jacobi sweeps. A 3×3 matrix converges quadratically and needs
/// four to seven sweeps; the cap only guards against pathological input.
const MAX_SWEEPS: usize = 64;

/// Column pairs visited by one cyclic sweep.
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Applies the plane rotation `(c, s)` to columns `p` and `q` of `cols`.
#[inline]
fn rotate_columns(cols: &mut [[f64; 3]; 3], p: usize, q: usize, c: f64, s: f64) {
    let (col_p, col_q) = (cols[p], cols[q]);
    cols[p] = std::array::from_fn(|r| c * col_p[r] - s * col_q[r]);
    cols[q] = std::array::from_fn(|r| s * col_p[r] + c * col_q[r]);
}

/// Determinant of the matrix whose columns are `cols` (their triple product).
#[inline]
fn column_determinant(cols: &[[f64; 3]; 3]) -> f64 {
    dot3(cols[0], cross3(cols[1], cols[2]))
}

/// Fills every column of `cols` that `present` flags as missing with a unit
/// vector orthogonal to all columns already in place, and marks it present.
///
/// For each missing column the coordinate axis least aligned with the present
/// columns is orthogonalised against them (two Gram-Schmidt passes). With `k`
/// orthonormal columns in place the squared residuals of the three axes sum to
/// `3 - k >= 1`, so the best residual is at least `1/sqrt(3)` and the
/// normalisation is always well conditioned. A column is marked present as soon
/// as it is written, so a second missing column is also made orthogonal to the
/// first one.
fn complete_orthonormal_columns(cols: &mut [[f64; 3]; 3], present: &mut [bool; 3]) {
    for j in 0..3 {
        if present[j] {
            continue;
        }
        let mut best = [0.0f64; 3];
        let mut best_len = 0.0f64;
        for axis in 0..3 {
            let mut candidate = [0.0f64; 3];
            candidate[axis] = 1.0;
            for _ in 0..2 {
                for k in 0..3 {
                    if present[k] {
                        let overlap = dot3(candidate, cols[k]);
                        candidate = sub3(candidate, scale3(cols[k], overlap));
                    }
                }
            }
            let len = len3(candidate);
            if len > best_len {
                best_len = len;
                best = candidate;
            }
        }
        cols[j] = best.map(|component| component / best_len);
        present[j] = true;
    }
}

/// Result returned for input without a meaningful SVD (NaN, infinity, or entries
/// so large that their squares overflow): identity factors and NaN singular values.
fn non_finite_svd() -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    (mat3_identity(), [f64::NAN; 3], mat3_identity())
}

/// Singular value decomposition `M = U · diag(σ) · Vᵀ` of a 3×3 matrix.
///
/// Returns `(U, sigma, Vᵀ)` in the *signed* (reflection-free) convention of
/// Irving et al. (2004) and Stomakhin et al. (2012), which the plasticity models
/// of the MPM solver rely on:
///
/// * `U` and `V` are proper rotations (orthogonal with determinant `+1`);
/// * `|σ₀| >= |σ₁| >= |σ₂|`, where `σ₀` and `σ₁` are non-negative and `σ₂`
///   carries the sign of `det M`. A negative `σ₂` therefore flags an inverted
///   (reflected) deformation gradient and `det M = σ₀ σ₁ σ₂`.
///
/// The factors are orthonormal to a few ulps for every input, including rank
/// deficient matrices (the left vectors of zero singular values are completed
/// to an orthonormal basis, so `M = 0` gives `U = V = I`). A diagonal input
/// that is already sorted in descending order is returned exactly, with
/// `U = V = I`. If `m` contains a NaN or infinity, or entries beyond about
/// `1e150` whose squares overflow, the result is `(I, [NaN; 3], I)`; entries
/// below about `1e-150` are indistinguishable from zero.
///
/// Reference: A. McAdams et al., "Computing the Singular Value Decomposition of
/// 3x3 matrices with minimal branching and elementary floating point operations"
/// (2011) for the sign convention; Hestenes (1958) for the Jacobi iteration.
pub fn mat3_svd(m: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    if m.iter().flatten().any(|entry| !entry.is_finite()) {
        return non_finite_svd();
    }

    // `a[k]` is column k of the work matrix A = M V, `v[k]` column k of V.
    let mut a: [[f64; 3]; 3] = std::array::from_fn(|k| [m[0][k], m[1][k], m[2][k]]);
    let mut v: [[f64; 3]; 3] = std::array::from_fn(|k| {
        let mut unit = [0.0f64; 3];
        unit[k] = 1.0;
        unit
    });

    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for (p, q) in PAIRS {
            let alpha = dot3(a[p], a[p]);
            let beta = dot3(a[q], a[q]);
            let gamma = dot3(a[p], a[q]);
            if gamma == 0.0 || gamma.abs() <= ORTHOGONALITY_TOL * alpha.sqrt() * beta.sqrt() {
                continue;
            }
            // t = tan(theta) is the smaller root of t² + 2 ζ t - 1 = 0, the
            // rotation that makes the two columns orthogonal.
            let zeta = (beta - alpha) / (2.0 * gamma);
            let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
            let c = 1.0 / t.hypot(1.0);
            let s = c * t;
            rotate_columns(&mut a, p, q, c, s);
            rotate_columns(&mut v, p, q, c, s);
            rotated = true;
        }
        if !rotated {
            break;
        }
    }

    // The columns of A are now mutually orthogonal: |a_k| = σ_k and u_k = a_k / σ_k.
    let norms: [f64; 3] = std::array::from_fn(|k| len3(a[k]));
    if norms.iter().any(|norm| !norm.is_finite()) {
        return non_finite_svd();
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    let mut sigma = [norms[order[0]], norms[order[1]], norms[order[2]]];
    let mut v_cols = [v[order[0]], v[order[1]], v[order[2]]];

    let negligible = sigma[0] * f64::EPSILON;
    let mut u_cols = [[0.0f64; 3]; 3];
    let mut present = [false; 3];
    for k in 0..3 {
        if sigma[k] > negligible {
            u_cols[k] = a[order[k]].map(|component| component / sigma[k]);
            present[k] = true;
        }
    }
    complete_orthonormal_columns(&mut u_cols, &mut present);

    // V is a product of rotations, but sorting may have permuted its columns
    // oddly and the completed U has an arbitrary orientation: restore
    // det U = det V = +1 by moving the sign into the smallest singular value.
    if column_determinant(&v_cols) < 0.0 {
        v_cols[2] = v_cols[2].map(|component| -component);
        sigma[2] = -sigma[2];
    }
    if column_determinant(&u_cols) < 0.0 {
        u_cols[2] = u_cols[2].map(|component| -component);
        sigma[2] = -sigma[2];
    }
    if sigma[2] == 0.0 {
        // Normalise a negative zero produced by the sign moves above.
        sigma[2] = 0.0;
    }

    // U has the u_k as columns; Vᵀ has the v_k as rows.
    let u = std::array::from_fn(|r| std::array::from_fn(|k| u_cols[k][r]));
    (u, sigma, v_cols)
}
