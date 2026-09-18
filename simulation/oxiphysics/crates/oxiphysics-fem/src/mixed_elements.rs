// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Mixed finite elements for incompressible problems.
//!
//! Implements Taylor-Hood P2/P1 elements, Stokes element matrices,
//! pressure stabilization, LBB inf-sup condition checks, penalty methods
//! for near-incompressibility, and MINI element bubble enrichment.

/// A mixed finite element pairing velocity and pressure fields.
///
/// Uses Taylor-Hood P2/P1 elements: quadratic velocity, linear pressure.
#[derive(Debug, Clone)]
pub struct MixedElement {
    /// Number of velocity nodes (P2 element).
    pub n_vel_nodes: usize,
    /// Number of pressure nodes (P1 element).
    pub n_pres_nodes: usize,
    /// Velocity degrees of freedom per node (spatial dimension).
    pub vel_dofs_per_node: usize,
    /// Stabilization parameter for pressure (Brezzi-Pitkäranta).
    pub stab_param: f64,
    /// Penalty parameter for near-incompressibility.
    pub penalty: f64,
}

impl MixedElement {
    /// Create a new mixed element with given parameters.
    ///
    /// # Arguments
    /// * `n_vel_nodes` - number of velocity nodes (e.g. 6 for 2D P2)
    /// * `n_pres_nodes` - number of pressure nodes (e.g. 3 for 2D P1)
    /// * `vel_dofs_per_node` - spatial dimension (2 or 3)
    /// * `stab_param` - Brezzi-Pitkäranta stabilization coefficient
    /// * `penalty` - penalty parameter for incompressibility
    pub fn new(
        n_vel_nodes: usize,
        n_pres_nodes: usize,
        vel_dofs_per_node: usize,
        stab_param: f64,
        penalty: f64,
    ) -> Self {
        Self {
            n_vel_nodes,
            n_pres_nodes,
            vel_dofs_per_node,
            stab_param,
            penalty,
        }
    }

    /// Total velocity degrees of freedom.
    pub fn n_vel_dofs(&self) -> usize {
        self.n_vel_nodes * self.vel_dofs_per_node
    }

    /// Total pressure degrees of freedom.
    pub fn n_pres_dofs(&self) -> usize {
        self.n_pres_nodes
    }
}

/// Brezzi-Pitkäranta pressure stabilization matrix.
///
/// Computes the stabilization term `alpha * h^2 * (grad p, grad q)` for
/// a local element of size `h_elem`. Returns a flat row-major `n_pres x n_pres` matrix.
///
/// # Arguments
/// * `h_elem` - characteristic element size
/// * `alpha` - stabilization coefficient (typically 0.01–0.25)
/// * `n_pres` - number of pressure DOFs
/// * `grad_p` - pressure shape function gradients, shape `[n_pres][dim]`
/// * `weight` - quadrature weight times Jacobian determinant
pub fn pressure_stabilization(
    h_elem: f64,
    alpha: f64,
    n_pres: usize,
    grad_p: &[[f64; 2]],
    weight: f64,
) -> Vec<f64> {
    assert_eq!(grad_p.len(), n_pres);
    let mut s = vec![0.0f64; n_pres * n_pres];
    let coeff = alpha * h_elem * h_elem * weight;
    for i in 0..n_pres {
        for j in 0..n_pres {
            let dot = grad_p[i][0] * grad_p[j][0] + grad_p[i][1] * grad_p[j][1];
            s[i * n_pres + j] += coeff * dot;
        }
    }
    s
}

/// Maximum number of one-sided Jacobi sweeps of [`smallest_singular_value`].
///
/// The method converges quadratically, a handful of sweeps suffice; the cap only
/// guards against a pathological (NaN-contaminated) input.
const SVD_MAX_SWEEPS: usize = 60;

/// Smallest singular value of the `m x n` matrix whose **rows** are `rows`
/// (`m = rows.len()`, every row of length `n`).
///
/// Uses the one-sided Jacobi (Hestenes) method on the rows: plane rotations
/// orthogonalise them pairwise, after which the Euclidean norms of the rows are
/// the singular values.  Unlike the eigenvalues of `B Bᵀ` this keeps the *relative*
/// accuracy of small singular values (forming `B Bᵀ` squares the condition number
/// and leaves a floor of about `1e-8 · σ_max` on the computed `σ_min`).
///
/// A matrix with more rows than columns (`m > n`) is rank deficient, its smallest
/// singular value is `0`.
fn smallest_singular_value(rows: &mut [Vec<f64>], n: usize) -> f64 {
    let m = rows.len();
    if m == 0 || n == 0 || m > n {
        return 0.0;
    }
    for _ in 0..SVD_MAX_SWEEPS {
        let mut rotated = false;
        for i in 0..m {
            for j in (i + 1)..m {
                let (head, tail) = rows.split_at_mut(j);
                let row_i = &mut head[i];
                let row_j = &mut tail[0];
                let alpha: f64 = row_i.iter().map(|v| v * v).sum();
                let beta: f64 = row_j.iter().map(|v| v * v).sum();
                let gamma: f64 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                // Rotation that makes the two rows orthogonal: tan = t with
                // t² + 2ζt − 1 = 0, the root of smaller magnitude.
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for (a, b) in row_i.iter_mut().zip(row_j.iter_mut()) {
                    let (x, y) = (*a, *b);
                    *a = c * x - s * y;
                    *b = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    rows.iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(f64::INFINITY, f64::min)
}

/// Check the LBB (Ladyzhenskaya-Babuška-Brezzi) inf-sup condition.
///
/// Returns the discrete inf-sup constant of the divergence operator `B`
/// (`[n_pres][n_vel_dofs]`, row-major),
///
/// `β_h = min_p max_v (pᵀ B v) / (‖p‖ ‖v‖) = σ_min(B)`,
///
/// the smallest singular value of `B` in Euclidean norms on both spaces.  It is
/// zero when `B` is rank deficient (`n_pres > n_vel_dofs`, or linearly dependent
/// pressure modes such as a checkerboard), and `0.0` is returned when there is no
/// pressure DOF at all.  Note that this is the *algebraic* constant: the mesh
/// independent LBB constant uses the `L²` norm of the pressure and the `H¹` norm of
/// the velocity, i.e. `β² = λ_min(Mₚ⁻¹ B K⁻¹ Bᵀ)` with the pressure mass matrix
/// `Mₚ` and the velocity stiffness `K`.
///
/// For a fully enclosed flow the constant pressure is always in the kernel of
/// `Bᵀ`, so this value is `0` for every such pair; use [`inf_sup_constant`] with
/// `exclude_constant_mode = true` to measure the stability of the remaining
/// pressure modes.
///
/// The singular values are computed by a one-sided Jacobi iteration that keeps
/// the relative accuracy of small values, so a rank-deficient `B` gives a result
/// at rounding level (`≲ 1e-12 · σ_max`), far below any stable pair.
///
/// # Panics
///
/// Panics if `b_matrix.len() != n_pres * n_vel_dofs`.
pub fn inf_sup_check(b_matrix: &[f64], n_pres: usize, n_vel_dofs: usize) -> f64 {
    inf_sup_constant(b_matrix, n_pres, n_vel_dofs, false)
}

/// Discrete inf-sup constant of `B`, optionally restricted to the pressure modes
/// orthogonal to the constant (hydrostatic) mode.
///
/// With `exclude_constant_mode = false` this is [`inf_sup_check`], `σ_min(B)`.
/// With `exclude_constant_mode = true` the infimum runs over the pressures
/// `p ⟂ (1, …, 1)` only, `β_h = min_{p ⟂ 1} ‖Bᵀp‖ / ‖p‖`; in an enclosed flow the
/// constant pressure is not determined by the velocity and must be removed to
/// obtain a meaningful constant.  The restriction is exact (an orthonormal basis of
/// the complement of the constant vector is applied to the rows of `B`), it does
/// not assume that the constant mode lies in the kernel of `Bᵀ`.
///
/// Returns `0.0` when there is no pressure mode to test: no pressure DOF, or a
/// single one while the constant mode is excluded.  A non-finite entry of `B`
/// gives `NaN`.
///
/// # Panics
///
/// Panics if `b_matrix.len() != n_pres * n_vel_dofs`.
pub fn inf_sup_constant(
    b_matrix: &[f64],
    n_pres: usize,
    n_vel_dofs: usize,
    exclude_constant_mode: bool,
) -> f64 {
    assert_eq!(b_matrix.len(), n_pres * n_vel_dofs);
    if b_matrix.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let m = n_pres;
    let n = n_vel_dofs;
    if m == 0 || n == 0 {
        return 0.0;
    }
    let b_rows: Vec<&[f64]> = b_matrix.chunks_exact(n).collect();
    let mut rows: Vec<Vec<f64>> = if exclude_constant_mode {
        // Helmert basis of the complement of (1, …, 1): row k (1 ≤ k < m) is
        // (1, …, 1, −k, 0, …, 0)/√(k(k+1)) with k leading ones.  Applying it to B
        // gives the (m−1) x n matrix H B whose singular values are those of B
        // restricted to p ⟂ 1.
        let mut projected = Vec::with_capacity(m.saturating_sub(1));
        let mut running_sum = vec![0.0_f64; n];
        for (k, row) in b_rows.iter().enumerate().take(m - 1) {
            for (acc, v) in running_sum.iter_mut().zip(row.iter()) {
                *acc += v;
            }
            let scale = 1.0 / (((k + 1) * (k + 2)) as f64).sqrt();
            let next = b_rows[k + 1];
            projected.push(
                running_sum
                    .iter()
                    .zip(next.iter())
                    .map(|(sum, b_next)| (sum - (k + 1) as f64 * b_next) * scale)
                    .collect(),
            );
        }
        projected
    } else {
        b_rows.iter().map(|row| row.to_vec()).collect()
    };
    smallest_singular_value(&mut rows, n)
}

/// Compute local stiffness matrix `K` and divergence matrix `B` for a Stokes element.
///
/// Returns `(K, B)` where:
/// - `K` is `[n_vel_dofs x n_vel_dofs]` (viscous term), row-major flat
/// - `B` is `[n_pres x n_vel_dofs]` (divergence coupling), row-major flat
///
/// # Arguments
/// * `viscosity` - dynamic viscosity
/// * `grad_v` - velocity shape function gradients, shape `[n_vel_nodes][dim=2]`
/// * `grad_p` - pressure shape function gradients, shape `[n_pres_nodes][dim=2]`
/// * `phi_p` - pressure shape function values at quadrature point, shape `[n_pres_nodes]`
/// * `div_v` - divergence of each velocity shape function, shape `[n_vel_nodes]`
/// * `weight` - quadrature weight * Jacobian det
/// * `dim` - spatial dimension (2)
pub fn stokes_element_matrix(
    viscosity: f64,
    grad_v: &[[f64; 2]],
    _grad_p: &[[f64; 2]],
    phi_p: &[f64],
    div_v: &[f64],
    weight: f64,
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_vel_nodes = grad_v.len();
    let n_pres_nodes = phi_p.len();
    let n_vel_dofs = n_vel_nodes * dim;
    let n_pres_dofs = n_pres_nodes;

    // K: viscosity * integral( grad(v_i) : grad(v_j) ) delta_{alpha,beta}
    // For dim=2, K is block-diagonal per component
    let mut k = vec![0.0f64; n_vel_dofs * n_vel_dofs];
    for i in 0..n_vel_nodes {
        for j in 0..n_vel_nodes {
            let dot = grad_v[i][0] * grad_v[j][0] + grad_v[i][1] * grad_v[j][1];
            let val = viscosity * weight * dot;
            // Block diagonal: each spatial component
            for d in 0..dim {
                let row = i * dim + d;
                let col = j * dim + d;
                k[row * n_vel_dofs + col] += val;
            }
        }
    }

    // B: integral( phi_p_i * div(v_j) ) where div(v_j) = sum_d grad_v[j][d]
    // B shape: [n_pres_dofs x n_vel_dofs]
    // For each pressure dof i and velocity node j, component d:
    //   B[i, j*dim+d] = -weight * phi_p[i] * grad_v[j][d]
    let mut b = vec![0.0f64; n_pres_dofs * n_vel_dofs];
    let _ = div_v; // div_v is redundant given grad_v; use grad_v directly
    for i in 0..n_pres_nodes {
        for (j, grad_vj) in grad_v.iter().enumerate().take(n_vel_nodes) {
            for (d, &gvjd) in grad_vj.iter().enumerate().take(dim) {
                let col = j * dim + d;
                b[i * n_vel_dofs + col] += -weight * phi_p[i] * gvjd;
            }
        }
    }

    (k, b)
}

/// Assemble incompressibility penalty term into the stiffness matrix.
///
/// Adds `penalty * B^T B` to the velocity block `K`. This enforces approximate
/// incompressibility without a separate pressure unknown.
///
/// # Arguments
/// * `k` - stiffness matrix (modified in place), size `n_vel_dofs x n_vel_dofs`
/// * `b` - divergence matrix, size `n_pres_dofs x n_vel_dofs`
/// * `penalty` - large penalty parameter
/// * `n_vel_dofs` - velocity DOF count
/// * `n_pres_dofs` - pressure DOF count
pub fn incompressibility_penalty(
    k: &mut [f64],
    b: &[f64],
    penalty: f64,
    n_vel_dofs: usize,
    n_pres_dofs: usize,
) {
    // K += penalty * B^T B
    // B^T is [n_vel_dofs x n_pres_dofs], B is [n_pres_dofs x n_vel_dofs]
    for i in 0..n_vel_dofs {
        for j in 0..n_vel_dofs {
            let mut s = 0.0;
            for p in 0..n_pres_dofs {
                // B^T[i,p] = B[p,i]
                s += b[p * n_vel_dofs + i] * b[p * n_vel_dofs + j];
            }
            k[i * n_vel_dofs + j] += penalty * s;
        }
    }
}

/// Evaluate the MINI element bubble function on a triangle.
///
/// The bubble function for the MINI element is `lambda_1 * lambda_2 * lambda_3`
/// where `lambda_i` are the barycentric coordinates. At interior points this
/// is positive and vanishes on all edges.
///
/// # Arguments
/// * `lam` - barycentric coordinates `[lambda_1, lambda_2, lambda_3]`, must sum to 1
///
/// Returns the bubble function value.
pub fn bubble_function(lam: &[f64; 3]) -> f64 {
    lam[0] * lam[1] * lam[2]
}

/// Evaluate the gradient of the MINI element bubble function.
///
/// Returns `grad(lambda_1 * lambda_2 * lambda_3)` with respect to barycentric
/// coordinates. The physical gradient requires multiplication by the inverse
/// Jacobian of the element mapping.
///
/// # Arguments
/// * `lam` - barycentric coordinates `[lambda_1, lambda_2, lambda_3]`
pub fn bubble_function_grad(lam: &[f64; 3]) -> [f64; 3] {
    [lam[1] * lam[2], lam[0] * lam[2], lam[0] * lam[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MixedElement construction ──────────────────────────────────────────

    #[test]
    fn test_mixed_element_new() {
        let e = MixedElement::new(6, 3, 2, 0.1, 1e6);
        assert_eq!(e.n_vel_nodes, 6);
        assert_eq!(e.n_pres_nodes, 3);
    }

    #[test]
    fn test_mixed_element_dofs() {
        let e = MixedElement::new(6, 3, 2, 0.1, 1e6);
        assert_eq!(e.n_vel_dofs(), 12);
        assert_eq!(e.n_pres_dofs(), 3);
    }

    #[test]
    fn test_mixed_element_3d() {
        let e = MixedElement::new(10, 4, 3, 0.05, 1e8);
        assert_eq!(e.n_vel_dofs(), 30);
        assert_eq!(e.n_pres_dofs(), 4);
    }

    #[test]
    fn test_mixed_element_stab_param() {
        let e = MixedElement::new(6, 3, 2, 0.25, 1e6);
        assert!((e.stab_param - 0.25).abs() < 1e-14);
    }

    #[test]
    fn test_mixed_element_penalty() {
        let e = MixedElement::new(6, 3, 2, 0.1, 1e9);
        assert!((e.penalty - 1e9).abs() < 1.0);
    }

    // ── pressure_stabilization ─────────────────────────────────────────────

    #[test]
    fn test_pressure_stab_size() {
        let grad_p = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]];
        let s = pressure_stabilization(0.1, 0.1, 3, &grad_p, 0.5);
        assert_eq!(s.len(), 9);
    }

    #[test]
    fn test_pressure_stab_symmetry() {
        let grad_p = [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]];
        let s = pressure_stabilization(0.2, 0.1, 3, &grad_p, 1.0);
        for i in 0..3 {
            for j in 0..3 {
                let diff = (s[i * 3 + j] - s[j * 3 + i]).abs();
                assert!(diff < 1e-14, "S not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_pressure_stab_positive_semidefinite() {
        let grad_p = [[1.0, 0.0], [0.0, 1.0]];
        let s = pressure_stabilization(1.0, 1.0, 2, &grad_p, 1.0);
        // Diagonal entries must be non-negative
        assert!(s[0] >= 0.0);
        assert!(s[3] >= 0.0);
    }

    #[test]
    fn test_pressure_stab_zero_alpha() {
        let grad_p = [[1.0, 0.0], [0.0, 1.0]];
        let s = pressure_stabilization(1.0, 0.0, 2, &grad_p, 1.0);
        for &v in &s {
            assert!(v.abs() < 1e-14);
        }
    }

    #[test]
    fn test_pressure_stab_scales_with_h_squared() {
        let grad_p = [[1.0, 0.0]];
        let s1 = pressure_stabilization(1.0, 1.0, 1, &grad_p, 1.0);
        let s2 = pressure_stabilization(2.0, 1.0, 1, &grad_p, 1.0);
        let ratio = s2[0] / s1[0];
        assert!((ratio - 4.0).abs() < 1e-10);
    }

    // ── inf_sup_check ──────────────────────────────────────────────────────

    /// Deterministic pseudo-random numbers in `[-1, 1)` (64-bit LCG).
    fn lcg_unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn test_inf_sup_identity_is_one() {
        let b = vec![1.0, 0.0, 0.0, 1.0];
        assert!((inf_sup_check(&b, 2, 2) - 1.0).abs() < 1e-12);
        // identity-like 3 x 4 divergence operator: all singular values are 1
        let b: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!((inf_sup_check(&b, 3, 4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_inf_sup_1x1() {
        // sigma_min of a 1x1 matrix is its absolute value
        assert!((inf_sup_check(&[3.0], 1, 1) - 3.0).abs() < 1e-12);
        assert!((inf_sup_check(&[-3.0], 1, 1) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_inf_sup_coupled_singular_values_three_and_one() {
        // B = U diag(3, 1) Vᵀ with a 30° rotation U: B Bᵀ has the eigenvalues 9 and 1
        // and non-zero off-diagonal coupling 8 sin(30°)cos(30°).  The Gershgorin
        // lower bound `diag − Σ|offdiag|` of the previous implementation is negative
        // here (3 − 3.46) and reported a stable pair as violating the LBB condition.
        let (c, s) = (30.0_f64.to_radians().cos(), 30.0_f64.to_radians().sin());
        let b = vec![3.0 * c, -s, 0.0, 3.0 * s, c, 0.0];
        let bbt01 = b[0] * b[3] + b[1] * b[4];
        assert!((bbt01 - 8.0 * s * c).abs() < 1e-12, "coupled: {bbt01}");
        let value = inf_sup_check(&b, 2, 3);
        assert!((value - 1.0).abs() < 1e-10, "sigma_min = {value}");

        // the same singular values with a rotated right-singular basis
        let phi = 0.7_f64;
        let (cp, sp) = (phi.cos(), phi.sin());
        let b = vec![3.0 * c, -s * cp, -s * sp, 3.0 * s, c * cp, c * sp];
        let value = inf_sup_check(&b, 2, 3);
        assert!((value - 1.0).abs() < 1e-10, "sigma_min = {value}");
    }

    #[test]
    fn test_inf_sup_rank_deficient_is_at_rounding_level() {
        // Two pressure modes with parallel rows (a checkerboard-like null mode): the
        // pair violates the LBB condition.  The singular value is at rounding level,
        // not at the ~1e-8 floor that the eigenvalues of B Bᵀ would leave.
        let b = vec![0.1, 0.2, 0.3, 0.07, 0.14, 0.21];
        let value = inf_sup_check(&b, 2, 3);
        assert!(value < 1e-12, "sigma_min = {value}");
        // a non-trivial rank-1 3 x 4 matrix
        let u = [0.3, -1.1, 0.7];
        let v = [1.3, 0.2, -0.4, 0.9];
        let b: Vec<f64> = u
            .iter()
            .flat_map(|ui| v.iter().map(move |vj| ui * vj))
            .collect();
        assert!(inf_sup_check(&b, 3, 4) < 1e-12);
    }

    #[test]
    fn test_inf_sup_more_pressure_than_velocity_dofs_is_zero() {
        // 3 pressure modes cannot be controlled by 2 velocity DOFs
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(inf_sup_check(&b, 3, 2), 0.0);
    }

    #[test]
    fn test_inf_sup_degenerate_sizes_and_zero_matrix() {
        // no pressure DOF at all: no stability can be claimed (and no sqrt(f64::MAX))
        assert_eq!(inf_sup_check(&[], 0, 4), 0.0);
        assert_eq!(inf_sup_check(&[], 3, 0), 0.0);
        assert_eq!(inf_sup_check(&[0.0; 6], 2, 3), 0.0);
        assert!(inf_sup_check(&[1.0, f64::NAN, 0.0, 1.0], 2, 2).is_nan());
    }

    #[test]
    fn test_inf_sup_matches_eigenvalues_of_b_bt_and_scales_linearly() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let (m, n) = (4, 7);
        let b: Vec<f64> = (0..m * n).map(|_| lcg_unit(&mut state)).collect();
        let bbt: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                (0..m)
                    .map(|j| (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum())
                    .collect()
            })
            .collect();
        let (eigenvalues, _) = crate::eigen::jacobi_eigen_dense(&bbt, m);
        let expected = eigenvalues[0].max(0.0).sqrt();
        let value = inf_sup_check(&b, m, n);
        assert!(value > 1e-3, "a generic B is well conditioned: {value}");
        assert!(
            (value - expected).abs() < 1e-9,
            "sigma_min = {value}, sqrt(lambda_min(B Bᵀ)) = {expected}"
        );
        let scaled: Vec<f64> = b.iter().map(|v| -2.5 * v).collect();
        assert!((inf_sup_check(&scaled, m, n) - 2.5 * value).abs() < 1e-12);
    }

    #[test]
    fn test_inf_sup_constant_without_exclusion_equals_check() {
        let b = vec![1.0, 2.0, 3.0, 0.5, 1.0, 0.0];
        assert_eq!(inf_sup_constant(&b, 2, 3, false), inf_sup_check(&b, 2, 3));
    }

    #[test]
    fn test_inf_sup_constant_excludes_the_hydrostatic_mode() {
        // Rows r and -r: the constant pressure is in the kernel of Bᵀ (an enclosed
        // flow), so the plain constant is 0 while the remaining mode is controlled
        // with sigma = |(r - (-r))|/sqrt(2) = sqrt(2) |r| = 2.
        let b = vec![1.0, -1.0, 0.0, -1.0, 1.0, 0.0];
        assert!(inf_sup_constant(&b, 2, 3, false) < 1e-12);
        assert!((inf_sup_constant(&b, 2, 3, true) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_inf_sup_constant_exclusion_is_a_projection_not_a_dropped_eigenvalue() {
        // B = diag(1, 2): the constant mode is NOT in the kernel of Bᵀ.  Over
        // p ⟂ (1, 1) the infimum is |Bᵀ(1, -1)|/sqrt(2) = sqrt(5/2); dropping the
        // smallest eigenvalue of B Bᵀ (1, 4) would wrongly give 2.
        let b = vec![1.0, 0.0, 0.0, 2.0];
        assert!((inf_sup_constant(&b, 2, 2, false) - 1.0).abs() < 1e-12);
        assert!((inf_sup_constant(&b, 2, 2, true) - 2.5_f64.sqrt()).abs() < 1e-12);
        // three pressure modes: the restricted infimum is never below sigma_min(B)
        let b = vec![
            1.0, 0.2, 0.0, 0.1, //
            0.0, 1.5, 0.3, 0.0, //
            0.2, 0.0, 2.0, 0.5,
        ];
        let restricted = inf_sup_constant(&b, 3, 4, true);
        assert!(restricted >= inf_sup_constant(&b, 3, 4, false) - 1e-12);
        // direct evaluation for the Helmert vectors h1 = (1,-1,0)/sqrt2 and
        // h2 = (1,1,-2)/sqrt6: the infimum over span{h1, h2} is at most min of both
        let norm_of = |p: [f64; 3]| -> f64 {
            (0..4)
                .map(|j| (0..3).map(|i| p[i] * b[i * 4 + j]).sum::<f64>().powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let h1 = norm_of([1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt(), 0.0]);
        let h2 = norm_of([
            1.0 / 6.0_f64.sqrt(),
            1.0 / 6.0_f64.sqrt(),
            -2.0 / 6.0_f64.sqrt(),
        ]);
        assert!(restricted <= h1.min(h2) + 1e-12);
    }

    #[test]
    fn test_inf_sup_constant_with_a_single_pressure_mode_and_exclusion_is_zero() {
        // only the constant mode exists: nothing is left to test
        assert_eq!(inf_sup_constant(&[1.0, 2.0, 3.0], 1, 3, true), 0.0);
        assert_eq!(inf_sup_constant(&[], 0, 3, true), 0.0);
    }

    // ── stokes_element_matrix ──────────────────────────────────────────────

    #[test]
    fn test_stokes_k_size() {
        let grad_v = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]];
        let grad_p = [[1.0, 0.0], [-1.0, 0.0]];
        let phi_p = [0.5, 0.5];
        let div_v = [1.0, 1.0, -2.0];
        let (k, _b) = stokes_element_matrix(1.0, &grad_v, &grad_p, &phi_p, &div_v, 1.0, 2);
        // n_vel_dofs = 3*2 = 6
        assert_eq!(k.len(), 36);
    }

    #[test]
    fn test_stokes_b_size() {
        let grad_v = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]];
        let grad_p = [[1.0, 0.0], [-1.0, 0.0]];
        let phi_p = [0.5, 0.5];
        let div_v = [1.0, 1.0, -2.0];
        let (_k, b) = stokes_element_matrix(1.0, &grad_v, &grad_p, &phi_p, &div_v, 1.0, 2);
        // n_pres_dofs=2, n_vel_dofs=6 => 12
        assert_eq!(b.len(), 12);
    }

    #[test]
    fn test_stokes_k_symmetry() {
        let grad_v = [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]];
        let grad_p = [[1.0, 0.0]];
        let phi_p = [1.0];
        let div_v = [3.0, 7.0, -0.5];
        let (k, _) = stokes_element_matrix(1.0, &grad_v, &grad_p, &phi_p, &div_v, 1.0, 2);
        let n = 6;
        for i in 0..n {
            for j in 0..n {
                let diff = (k[i * n + j] - k[j * n + i]).abs();
                assert!(diff < 1e-12, "K not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_stokes_viscosity_scaling() {
        let grad_v = [[1.0, 0.0], [0.0, 1.0]];
        let phi_p = [1.0];
        let grad_p = [[0.5, 0.5]];
        let div_v = [1.0, 1.0];
        let (k1, _) = stokes_element_matrix(1.0, &grad_v, &grad_p, &phi_p, &div_v, 1.0, 2);
        let (k2, _) = stokes_element_matrix(2.0, &grad_v, &grad_p, &phi_p, &div_v, 1.0, 2);
        for (a, b) in k1.iter().zip(k2.iter()) {
            assert!((b - 2.0 * a).abs() < 1e-12);
        }
    }

    #[test]
    fn test_stokes_k_positive_diagonal() {
        let grad_v = [[1.0, 0.0], [0.0, 1.0]];
        let phi_p = [1.0];
        let grad_p = [[0.5, 0.5]];
        let div_v = [1.0, 1.0];
        let (k, _) = stokes_element_matrix(1.0, &grad_v, &grad_p, &phi_p, &div_v, 1.0, 2);
        // Diagonal entries should be >= 0
        let n = 4;
        for i in 0..n {
            assert!(k[i * n + i] >= 0.0, "Negative diagonal at {i}");
        }
    }

    // ── incompressibility_penalty ──────────────────────────────────────────

    #[test]
    fn test_penalty_increases_k() {
        let mut k = vec![1.0; 4];
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let k_before = k.clone();
        incompressibility_penalty(&mut k, &b, 1.0, 2, 2);
        // K += penalty * B^T B = I => K[0,0] += 1
        assert!(k[0] >= k_before[0]);
    }

    #[test]
    fn test_penalty_zero_no_change() {
        let mut k = vec![2.0, 0.0, 0.0, 2.0];
        let k_orig = k.clone();
        let b = vec![1.0, 0.0, 0.0, 1.0];
        incompressibility_penalty(&mut k, &b, 0.0, 2, 2);
        for (a, b) in k.iter().zip(k_orig.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn test_penalty_symmetry_preserved() {
        let mut k = vec![1.0, 0.5, 0.5, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 B
        incompressibility_penalty(&mut k, &b, 1.0, 2, 2);
        let diff = (k[1] - k[2]).abs();
        assert!(diff < 1e-12, "Symmetry broken after penalty");
    }

    // ── bubble_function ────────────────────────────────────────────────────

    #[test]
    fn test_bubble_centroid() {
        let lam = [1.0 / 3.0; 3];
        let val = bubble_function(&lam);
        let expected = (1.0 / 3.0f64).powi(3);
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_bubble_vertex_zero() {
        assert!(bubble_function(&[1.0, 0.0, 0.0]).abs() < 1e-14);
        assert!(bubble_function(&[0.0, 1.0, 0.0]).abs() < 1e-14);
        assert!(bubble_function(&[0.0, 0.0, 1.0]).abs() < 1e-14);
    }

    #[test]
    fn test_bubble_positive_interior() {
        let lam = [0.5, 0.3, 0.2];
        let val = bubble_function(&lam);
        assert!(val > 0.0);
    }

    #[test]
    fn test_bubble_grad_centroid() {
        let lam = [1.0 / 3.0; 3];
        let grad = bubble_function_grad(&lam);
        // By symmetry all components equal
        let expected = (1.0 / 3.0f64).powi(2);
        for &g in &grad {
            assert!((g - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_bubble_grad_nonzero_interior() {
        let lam = [0.6, 0.3, 0.1];
        let grad = bubble_function_grad(&lam);
        // gradient components are products of other two barycentric coords
        assert!((grad[0] - lam[1] * lam[2]).abs() < 1e-14);
        assert!((grad[1] - lam[0] * lam[2]).abs() < 1e-14);
        assert!((grad[2] - lam[0] * lam[1]).abs() < 1e-14);
    }

    #[test]
    fn test_bubble_grad_vertex() {
        let lam = [1.0, 0.0, 0.0];
        let grad = bubble_function_grad(&lam);
        // grad[0] = lam[1]*lam[2] = 0
        assert!(grad[0].abs() < 1e-14);
        // grad[1] = lam[0]*lam[2] = 0
        assert!(grad[1].abs() < 1e-14);
        // grad[2] = lam[0]*lam[1] = 0
        assert!(grad[2].abs() < 1e-14);
    }
}
