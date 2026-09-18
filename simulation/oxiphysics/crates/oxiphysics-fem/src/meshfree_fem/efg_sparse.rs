// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Sparse assembly and iterative solution of the 2-D plane-stress EFG system.
//!
//! The dense path ([`efg_assemble_2d`] followed by
//! [`solve_dense`]) stores `ndof²` numbers and eliminates in
//! `O(ndof³)`.  Here
//!
//! * the stiffness matrix is accumulated row by row, so only the couplings between
//!   overlapping supports exist (`O(N m²)` entries for `m` nodes in range);
//! * the essential boundary conditions enter as triplet contributions
//!   ([`BcContribution`]), the same ones the dense
//!   `apply_*_bc` functions add to their arrays;
//! * the penalty and Nitsche systems, which are symmetric positive definite for a
//!   restrained body and a sufficient parameter, are solved with Jacobi-preconditioned
//!   conjugate gradients (with iterative refinement of the true residual); the
//!   Lagrange-multiplier saddle system `[[K, Bᵀ], [B, 0]]` is symmetric indefinite and is
//!   solved with MINRES, falling back to restarted GMRES, on the equilibrated matrix
//!   (the constraint rows hold values `φ ≤ 1` while `K` holds `E`-sized entries).
//!
//! A solve is only accepted if its **true** residual `‖b − A x‖/‖b‖` is small; an
//! unconverged iteration is never returned.  [`efg_analysis_2d`]
//! switches to this path for systems of [`EFG_SPARSE_DOF_THRESHOLD`] DOF and more and
//! falls back to the dense solution if it fails.
//!
//! [`efg_assemble_2d`]: crate::meshfree_fem::efg_assemble_2d
//! [`solve_dense`]: crate::meshfree_fem::solve_dense
//! [`efg_analysis_2d`]: crate::meshfree_fem::efg_analysis_2d

use std::collections::HashMap;

use crate::solvers::{gmres_sparse, minres_sparse, pcg_solve};
use crate::sparse::CsrMatrix;

use super::boundary::{
    BcContribution, lagrange_bc_contribution, nitsche_bc_contribution, penalty_bc_contribution,
};
use super::functions::{Point2, assemble_efg_2d_with};
use super::types::{
    BcEnforcementMethod, EfgMaterial, EfgResult, EssentialBc, KernelType, SupportDomain, TriCell,
};

/// Number of degrees of freedom from which [`efg_analysis_2d`](super::efg_analysis_2d)
/// uses the sparse assembly and the iterative solvers instead of the dense matrix and
/// Gaussian elimination.
pub const EFG_SPARSE_DOF_THRESHOLD: usize = 200;

/// Relative accuracy of each inner Krylov solve of the refinement loop.
const INNER_TOLERANCE: f64 = 1e-10;

/// True relative residual `‖b − A x‖/‖b‖` at which the refinement stops (rounding
/// level: the solution is as accurate as the conditioning of `A` allows).
const TARGET_RESIDUAL: f64 = 1e-14;

/// Largest true relative residual of a solution that is accepted.
const ACCEPT_TOLERANCE: f64 = 1e-10;

/// Most refinement passes (each solves the residual equation) per solve.
const MAX_REFINEMENTS: usize = 6;

/// Restart length of the GMRES fallback.
const GMRES_RESTART: usize = 120;

/// Sparse matrix under assembly: one hash map per row, summing repeated entries.
struct RowAccumulator {
    rows: Vec<HashMap<usize, f64>>,
}

impl RowAccumulator {
    fn new(n: usize) -> Self {
        Self {
            rows: vec![HashMap::new(); n],
        }
    }

    fn add(&mut self, row: usize, col: usize, value: f64) {
        *self.rows[row].entry(col).or_insert(0.0) += value;
    }

    /// Append `extra` empty rows (the multiplier DOF of a saddle system).
    fn grow(&mut self, extra: usize) {
        self.rows.extend((0..extra).map(|_| HashMap::new()));
    }

    fn into_csr(self) -> CsrMatrix {
        let n = self.rows.len();
        let triplets: Vec<(usize, usize, f64)> = self
            .rows
            .into_iter()
            .enumerate()
            .flat_map(|(row, entries)| {
                entries
                    .into_iter()
                    .map(move |(col, value)| (row, col, value))
            })
            .collect();
        CsrMatrix::from_triplets(n, n, &triplets)
    }
}

/// Add a boundary-condition contribution to a system under assembly.
fn apply_contribution(
    system: &mut RowAccumulator,
    force: &mut [f64],
    contribution: &BcContribution,
) {
    for &(row, col, value) in &contribution.stiffness {
        system.add(row, col, value);
    }
    for &(row, value) in &contribution.force {
        force[row] += value;
    }
}

/// Assemble the global stiffness matrix and force vector of a 2-D plane-stress EFG
/// analysis in sparse form.
///
/// The same integration as the dense [`efg_assemble_2d`](super::efg_assemble_2d)
/// (triangular background cells, one-point rule) with the same entries, stored as a
/// [`CsrMatrix`] built from `(2 I + r, 2 J + c, value)` contributions.  `None` if the
/// MLS shape functions are undefined at a cell centroid.
pub fn efg_assemble_2d_sparse(
    domain: &SupportDomain,
    cells: &[TriCell],
    mat: &EfgMaterial,
    ktype: KernelType,
    body_force: Point2,
) -> Option<(CsrMatrix, Vec<f64>)> {
    let mut system = RowAccumulator::new(domain.node_count() * 2);
    let force = assemble_efg_2d_with(domain, cells, mat, ktype, body_force, |row, col, value| {
        system.add(row, col, value);
    })?;
    Some((system.into_csr(), force))
}

/// `‖b − A x‖ / ‖b‖` (`0` for a zero right-hand side that `x` solves exactly).
fn relative_residual(matrix: &CsrMatrix, x: &[f64], rhs: &[f64]) -> f64 {
    let ax = matrix.mul_vec(x);
    let residual: f64 = ax
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (b - a) * (b - a))
        .sum::<f64>()
        .sqrt();
    let scale: f64 = rhs.iter().map(|b| b * b).sum::<f64>().sqrt();
    if scale > 0.0 {
        residual / scale
    } else {
        residual
    }
}

/// Solve `A x = b` by iterative refinement with an inexact inner solver.
///
/// Starting from `x = 0`, every pass solves the residual equation `A d = r`,
/// `r = b − A x`, to about [`INNER_TOLERANCE`] with `solve_correction` and adds `d`
/// to `x`.  The true residual shrinks geometrically until it reaches rounding level;
/// the recurrence-based residual of a Krylov method alone does not (CG and MINRES
/// drift from the true residual in the ill-conditioned systems of penalty and
/// multiplier methods).  Returns `None` unless the true relative residual ends
/// below [`ACCEPT_TOLERANCE`]; an unconverged iterate is never returned.
fn solve_by_refinement<F>(
    matrix: &CsrMatrix,
    rhs: &[f64],
    mut solve_correction: F,
) -> Option<Vec<f64>>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let mut x = vec![0.0_f64; rhs.len()];
    let mut residual = relative_residual(matrix, &x, rhs);
    for _ in 0..MAX_REFINEMENTS {
        if residual <= TARGET_RESIDUAL {
            break;
        }
        let ax = matrix.mul_vec(&x);
        let remainder: Vec<f64> = rhs.iter().zip(ax.iter()).map(|(b, a)| b - a).collect();
        let correction = solve_correction(&remainder);
        let candidate: Vec<f64> = x
            .iter()
            .zip(correction.iter())
            .map(|(a, d)| a + d)
            .collect();
        let candidate_residual = relative_residual(matrix, &candidate, rhs);
        if candidate_residual < residual {
            x = candidate;
            residual = candidate_residual;
        } else {
            break;
        }
    }
    (residual <= ACCEPT_TOLERANCE).then_some(x)
}

/// Jacobi-preconditioned CG (with refinement); `None` for a matrix with a
/// non-positive diagonal or if the true residual does not end below
/// [`ACCEPT_TOLERANCE`].
fn solve_cg(matrix: &CsrMatrix, rhs: &[f64]) -> Option<Vec<f64>> {
    let diagonal = matrix.diagonal_vec();
    if !diagonal.iter().all(|d| d.is_finite() && *d > 0.0) {
        return None;
    }
    let preconditioner: Vec<f64> = diagonal.iter().map(|d| 1.0 / d).collect();
    let max_iterations = 20 * rhs.len() + 1000;
    solve_by_refinement(matrix, rhs, |remainder| {
        pcg_solve(
            matrix,
            remainder,
            &preconditioner,
            max_iterations,
            INNER_TOLERANCE,
        )
        .0
    })
}

/// MINRES, then restarted GMRES, on `S A S y = S b` (`x = S y`) for the diagonal
/// scaling `S = diag(scale)`, both with refinement; `None` unless the scaled true
/// residual ends below [`ACCEPT_TOLERANCE`].
fn solve_krylov_scaled(matrix: &CsrMatrix, rhs: &[f64], scale: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    let mut scaled = matrix.clone();
    for row in 0..scaled.nrows {
        for idx in scaled.row_ptr[row]..scaled.row_ptr[row + 1] {
            scaled.values[idx] *= scale[row] * scale[scaled.col_indices[idx]];
        }
    }
    let scaled_rhs: Vec<f64> = rhs.iter().zip(scale.iter()).map(|(b, s)| b * s).collect();
    let unscale =
        |y: Vec<f64>| -> Vec<f64> { y.iter().zip(scale.iter()).map(|(v, s)| v * s).collect() };

    let max_iterations = 20 * n + 1000;
    if let Some(y) = solve_by_refinement(&scaled, &scaled_rhs, |remainder| {
        minres_sparse(&scaled, remainder, max_iterations, INNER_TOLERANCE).0
    }) {
        return Some(unscale(y));
    }
    let restart = GMRES_RESTART.min(n.max(1));
    let outer = (max_iterations / restart).max(10);
    solve_by_refinement(&scaled, &scaled_rhs, |remainder| {
        gmres_sparse(&scaled, remainder, restart, outer, INNER_TOLERANCE).0
    })
    .map(unscale)
}

/// Solve a symmetric system: conjugate gradients when `A` is positive definite
/// (positive diagonal and CG converges), otherwise Krylov methods on the
/// Jacobi-equilibrated matrix.
fn solve_symmetric(matrix: &CsrMatrix, rhs: &[f64]) -> Option<Vec<f64>> {
    if let Some(x) = solve_cg(matrix, rhs) {
        return Some(x);
    }
    let scale: Vec<f64> = matrix
        .diagonal_vec()
        .iter()
        .map(|d| {
            if d.is_finite() && d.abs() > 1e-300 {
                1.0 / d.abs().sqrt()
            } else {
                1.0
            }
        })
        .collect();
    solve_krylov_scaled(matrix, rhs, &scale)
}

/// Solve the saddle system `[[K, Bᵀ], [B, 0]]` whose first `n_primal` rows form `K`.
///
/// The primal block is scaled to a unit diagonal and the multiplier rows by the
/// root of the mean primal diagonal, which brings the constraint block to the size of
/// the equilibrated `K` (`u` is unchanged by the scaling).
fn solve_saddle(matrix: &CsrMatrix, rhs: &[f64], n_primal: usize) -> Option<Vec<f64>> {
    let diagonal = matrix.diagonal_vec();
    let primal: Vec<f64> = diagonal.iter().take(n_primal).map(|d| d.abs()).collect();
    let positive: Vec<f64> = primal
        .iter()
        .copied()
        .filter(|d| d.is_finite() && *d > 1e-300)
        .collect();
    if positive.is_empty() {
        return None;
    }
    let mean = positive.iter().sum::<f64>() / positive.len() as f64;
    let scale: Vec<f64> = (0..matrix.nrows)
        .map(|i| {
            if i >= n_primal {
                mean.sqrt()
            } else if primal[i].is_finite() && primal[i] > 1e-300 {
                1.0 / primal[i].sqrt()
            } else {
                1.0
            }
        })
        .collect();
    solve_krylov_scaled(matrix, rhs, &scale)
}

/// Run a complete 2-D plane-stress EFG analysis with sparse storage and iterative
/// solvers.
///
/// The arguments and the returned parameters are those of
/// [`efg_analysis_2d`](super::efg_analysis_2d).  `None` is returned if the assembly or
/// the boundary conditions fail, or if the iterative solver does not reach a small
/// true residual (the dense [`efg_analysis_2d_dense`](super::efg_analysis_2d_dense)
/// has no such limitation).
pub fn efg_analysis_2d_sparse(
    domain: &SupportDomain,
    cells: &[TriCell],
    mat: &EfgMaterial,
    ktype: KernelType,
    body_force: Point2,
    bcs: &[EssentialBc],
    bc_method: BcEnforcementMethod,
    penalty: f64,
) -> Option<EfgResult> {
    let ndof = domain.node_count() * 2;
    let mut system = RowAccumulator::new(ndof);
    let mut force =
        assemble_efg_2d_with(domain, cells, mat, ktype, body_force, |row, col, value| {
            system.add(row, col, value);
        })?;
    let displacements = match bc_method {
        BcEnforcementMethod::Penalty => {
            let contribution = penalty_bc_contribution(domain, ktype, bcs, penalty).ok()?;
            apply_contribution(&mut system, &mut force, &contribution);
            solve_symmetric(&system.into_csr(), &force)?
        }
        BcEnforcementMethod::Nitsche => {
            let contribution = nitsche_bc_contribution(domain, ktype, mat, bcs, penalty).ok()?;
            apply_contribution(&mut system, &mut force, &contribution);
            solve_symmetric(&system.into_csr(), &force)?
        }
        BcEnforcementMethod::LagrangeMultiplier => {
            let contribution = lagrange_bc_contribution(domain, ktype, bcs).ok()?;
            system.grow(contribution.extra_dofs);
            force.resize(ndof + contribution.extra_dofs, 0.0);
            apply_contribution(&mut system, &mut force, &contribution);
            let solution = solve_saddle(&system.into_csr(), &force, ndof)?;
            solution[..ndof].to_vec()
        }
    };
    Some(EfgResult {
        displacements,
        num_nodes: domain.node_count(),
    })
}
