// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Sparse assembly of the EFG stiffness matrix.
//!
//! The nodes in range of a quadrature point are gathered with a k-d tree that is
//! built once, so a point costs `O(log n + m)` for the `m` nodes of its support
//! instead of `O(n)`.  The shape functions and their gradients are evaluated on
//! that support set only, with the **analytic** derivative of the moving-least-squares
//! construction (no finite differences), and the `m × m` blocks `wᵩ B_Iᵀ D B_J` are
//! accumulated in a sparse matrix.  The matrix is exactly symmetric to rounding, and
//! its memory is `O(n m)` instead of `O(n²)`.
//!
//! # Derivative of the MLS shape functions
//!
//! With the basis `p_I = [1, (x_I − x)/h, (y_I − y)/h]` shifted to the evaluation
//! point `x` and scaled by the support radius `h` (so `∂p/∂x = [0, −1/h, 0]` and
//! `∂p/∂y = [0, 0, −1/h]` do not depend on the node), the moment matrix
//! `A = Σ w_I p_I p_Iᵀ`, the correction `c = A⁻¹ e₁` and the shape functions
//! `Φ_I = w_I cᵀ p_I`, one differentiates with `∂_k A = Σ [∂_k w_I p_I p_Iᵀ +
//! w_I (∂_k p p_Iᵀ + p_I ∂_k pᵀ)]`, `∂_k c = −A⁻¹ (∂_k A) c` and
//!
//! `∂_k Φ_I = ∂_k w_I (cᵀ p_I) + w_I ((∂_k c)ᵀ p_I + cᵀ ∂_k p)`.
//!
//! The kernel gradient is `∂_k w_I = w'(s_I) ∂_k s_I` with `s_I = |x − x_I|/h`
//! (taken as `0` at `s = 0`, where the radial direction is undefined).

use std::collections::HashMap;

use oxiphysics_core::math::Vec3;
use oxiphysics_core::spatial::KdTree;

use super::{
    BasisLayout, EfgMethod, PIVOT_RELATIVE_TOL, WeightType, basis_layout, plane_stress_d_matrix,
    solve_pivoted, weight_function, weight_function_deriv,
};
use crate::error::{Error, Result};
use crate::sparse::CsrMatrix;

/// Gradients of the MLS shape functions on the support set of one point.
///
/// Only the gradients enter the stiffness matrix; the values `Φ_I` are available from
/// [`mls_shape_functions`](super::mls_shape_functions).
#[derive(Debug, Clone)]
pub(super) struct ShapeSet {
    /// Cloud indices of the nodes in range (ascending).
    pub(super) support: Vec<usize>,
    /// `∂Φ_I/∂x`, aligned with `support`.
    pub(super) dphi_dx: Vec<f64>,
    /// `∂Φ_I/∂y`, aligned with `support`.
    pub(super) dphi_dy: Vec<f64>,
}

/// Shape-function gradients for a basis of `N` terms.
///
/// `basis[I]` is `p_I`, `basis_gradients[k]` the (node independent) derivative
/// `∂_k p`, `weights[I]` and `weight_gradients[I]` the kernel and its gradient.
/// `None` if the moment matrix is singular.
fn gradient_shape_values<const N: usize>(
    weights: &[f64],
    weight_gradients: &[[f64; 2]],
    basis: &[[f64; N]],
    basis_gradients: &[[f64; N]; 2],
) -> Option<Vec<[f64; 2]>> {
    let mut moment = [[0.0_f64; N]; N];
    let mut moment_gradients = [[[0.0_f64; N]; N]; 2];
    for ((&w, dw), p) in weights.iter().zip(weight_gradients).zip(basis) {
        for row in 0..N {
            for col in 0..N {
                let outer = p[row] * p[col];
                moment[row][col] += w * outer;
                for (k, gradient) in moment_gradients.iter_mut().enumerate() {
                    gradient[row][col] += dw[k] * outer
                        + w * (basis_gradients[k][row] * p[col] + p[row] * basis_gradients[k][col]);
                }
            }
        }
    }
    let mut e1 = [0.0_f64; N];
    e1[0] = 1.0;
    let correction = solve_pivoted(moment, e1, PIVOT_RELATIVE_TOL)?;
    let mut correction_gradients = [[0.0_f64; N]; 2];
    for (k, gradient) in correction_gradients.iter_mut().enumerate() {
        let mut rhs = [0.0_f64; N];
        for (row, entry) in rhs.iter_mut().enumerate() {
            *entry = -(0..N)
                .map(|col| moment_gradients[k][row][col] * correction[col])
                .sum::<f64>();
        }
        *gradient = solve_pivoted(moment, rhs, PIVOT_RELATIVE_TOL)?;
    }

    let dot =
        |a: &[f64; N], b: &[f64; N]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };
    let mut gradients = Vec::with_capacity(weights.len());
    for ((&w, dw), p) in weights.iter().zip(weight_gradients).zip(basis) {
        let polynomial = dot(&correction, p);
        let mut gradient = [0.0_f64; 2];
        for (k, entry) in gradient.iter_mut().enumerate() {
            *entry = dw[k] * polynomial
                + w * (dot(&correction_gradients[k], p) + dot(&correction, &basis_gradients[k]));
        }
        gradients.push(gradient);
    }
    Some(gradients)
}

/// Analytic gradients of the MLS shape functions at `(x, y)` on the nodes of
/// `support`.
///
/// `h` is the support radius and `layout` the basis of the **whole cloud** (it must not
/// be derived from the support subset: a collinear support in a 2-D cloud has to stay
/// singular instead of silently switching to a reduced basis).  Returns `None`
/// where the shape functions do not exist (singular moment matrix, `h` not positive and
/// finite, non-finite point).
pub(super) fn shape_set(
    x: f64,
    y: f64,
    support: &[usize],
    node_x: &[f64],
    node_y: &[f64],
    h: f64,
    wtype: WeightType,
    layout: BasisLayout,
) -> Option<ShapeSet> {
    if !x.is_finite() || !y.is_finite() || !h.is_finite() || h <= 0.0 {
        return None;
    }
    let mut local = Vec::with_capacity(support.len());
    let mut weights = Vec::with_capacity(support.len());
    let mut weight_gradients = Vec::with_capacity(support.len());
    for &index in support {
        let lx = (node_x[index] - x) / h;
        let ly = (node_y[index] - y) / h;
        let s = lx.hypot(ly);
        local.push([lx, ly]);
        weights.push(weight_function(s, wtype));
        let dw_ds = weight_function_deriv(s, wtype);
        // ∂s/∂x = −lx/(s h), ∂s/∂y = −ly/(s h); no direction at s = 0
        weight_gradients.push(if s > 0.0 {
            [-dw_ds * lx / (s * h), -dw_ds * ly / (s * h)]
        } else {
            [0.0, 0.0]
        });
    }
    let inv_h = 1.0 / h;
    let gradients = match layout {
        BasisLayout::Plane => {
            let basis: Vec<[f64; 3]> = local.iter().map(|p| [1.0, p[0], p[1]]).collect();
            gradient_shape_values(
                &weights,
                &weight_gradients,
                &basis,
                &[[0.0, -inv_h, 0.0], [0.0, 0.0, -inv_h]],
            )?
        }
        BasisLayout::AlongX => {
            let basis: Vec<[f64; 2]> = local.iter().map(|p| [1.0, p[0]]).collect();
            gradient_shape_values(
                &weights,
                &weight_gradients,
                &basis,
                &[[0.0, -inv_h], [0.0, 0.0]],
            )?
        }
        BasisLayout::AlongY => {
            let basis: Vec<[f64; 2]> = local.iter().map(|p| [1.0, p[1]]).collect();
            gradient_shape_values(
                &weights,
                &weight_gradients,
                &basis,
                &[[0.0, 0.0], [0.0, -inv_h]],
            )?
        }
    };
    Some(ShapeSet {
        support: support.to_vec(),
        dphi_dx: gradients.iter().map(|g| g[0]).collect(),
        dphi_dy: gradients.iter().map(|g| g[1]).collect(),
    })
}

/// Add the `2m × 2m` block `weight · B_Iᵀ D B_J` of one quadrature point to the rows.
///
/// `B_I = [[∂Φ_I/∂x, 0], [0, ∂Φ_I/∂y], [∂Φ_I/∂y, ∂Φ_I/∂x]]` is the strain-displacement
/// matrix of node `I` (engineering shear strain).
fn accumulate_point(
    rows: &mut [HashMap<usize, f64>],
    shape: &ShapeSet,
    d: &[[f64; 3]; 3],
    weight: f64,
) {
    let strain_matrix = |a: usize| -> [[f64; 2]; 3] {
        [
            [shape.dphi_dx[a], 0.0],
            [0.0, shape.dphi_dy[a]],
            [shape.dphi_dy[a], shape.dphi_dx[a]],
        ]
    };
    // D B_J, once per node
    let stressed: Vec<[[f64; 2]; 3]> = (0..shape.support.len())
        .map(|j| {
            let b_j = strain_matrix(j);
            let mut db = [[0.0_f64; 2]; 3];
            for (p, db_row) in db.iter_mut().enumerate() {
                for (c, entry) in db_row.iter_mut().enumerate() {
                    *entry = (0..3).map(|q| d[p][q] * b_j[q][c]).sum();
                }
            }
            db
        })
        .collect();
    for (a, &node_a) in shape.support.iter().enumerate() {
        let b_a = strain_matrix(a);
        for (b, &node_b) in shape.support.iter().enumerate() {
            let block: [[f64; 2]; 2] = [0, 1].map(|r| {
                [0, 1].map(|c| (0..3).map(|p| b_a[p][r] * stressed[b][p][c]).sum::<f64>())
            });
            for (r, block_row) in block.iter().enumerate() {
                for (c, value) in block_row.iter().enumerate() {
                    *rows[2 * node_a + r].entry(2 * node_b + c).or_insert(0.0) += weight * value;
                }
            }
        }
    }
}

/// Sparse EFG stiffness matrix (`2N × 2N`, plane stress, two DOF per node).
///
/// A 2 × 2 Gauss rule is used in every background cell.  The support set of each
/// quadrature point is found with a k-d tree, the shape-function gradients are
/// analytic (see the module documentation) and the matrix is symmetric to rounding.
/// Use this instead of [`efg_stiffness`](super::efg_stiffness) for more than a few
/// hundred nodes: the dense matrix needs `O(N²)` memory.
///
/// # Errors
///
/// [`Error::InvalidInput`] if the node arrays differ in length, a background grid
/// direction has no cells, the influence radius is not positive and finite, or the
/// MLS shape functions do not exist at a quadrature point (the influence radius is
/// too small for the node distribution: every quadrature point needs at least
/// three non-collinear nodes in range).
pub fn efg_stiffness_sparse(efg: &EfgMethod) -> Result<CsrMatrix> {
    if efg.node_x.len() != efg.node_y.len() {
        return Err(Error::InvalidInput(format!(
            "node_x has {} entries but node_y has {}",
            efg.node_x.len(),
            efg.node_y.len()
        )));
    }
    if efg.n_cells_x == 0 || efg.n_cells_y == 0 {
        return Err(Error::InvalidInput(
            "the background grid needs at least one cell in each direction".to_string(),
        ));
    }
    if !efg.influence_radius.is_finite() || efg.influence_radius <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "the influence radius must be positive and finite, got {}",
            efg.influence_radius
        )));
    }
    let dof = 2 * efg.num_nodes();
    let d = plane_stress_d_matrix(efg.young, efg.poisson);
    let radius = efg.influence_radius;
    let layout = basis_layout(&efg.node_x, &efg.node_y, radius);
    let tree = KdTree::build(
        efg.node_x
            .iter()
            .zip(efg.node_y.iter())
            .map(|(&x, &y)| Vec3::new(x, y, 0.0))
            .collect(),
    );

    let [x_min, x_max, y_min, y_max] = efg.domain;
    let hx = (x_max - x_min) / efg.n_cells_x as f64;
    let hy = (y_max - y_min) / efg.n_cells_y as f64;
    // 2-point Gauss rule: points ±1/√3, weight 1
    let gauss = [-1.0_f64 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()];
    let jacobian = 0.25 * hx * hy;

    let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); dof];
    for cell_i in 0..efg.n_cells_x {
        for cell_j in 0..efg.n_cells_y {
            let x_c = x_min + (cell_i as f64 + 0.5) * hx;
            let y_c = y_min + (cell_j as f64 + 0.5) * hy;
            for &xi in &gauss {
                for &eta in &gauss {
                    let xq = x_c + 0.5 * hx * xi;
                    let yq = y_c + 0.5 * hy * eta;
                    let mut support = tree.range_search(Vec3::new(xq, yq, 0.0), radius);
                    support.sort_unstable();
                    let shape = shape_set(
                        xq,
                        yq,
                        &support,
                        &efg.node_x,
                        &efg.node_y,
                        radius,
                        efg.weight_type,
                        layout,
                    )
                    .ok_or_else(|| {
                        Error::InvalidInput(format!(
                            "the MLS shape functions are undefined at the quadrature point \
                             ({xq}, {yq}): the influence radius {radius} is too small for the \
                             node distribution"
                        ))
                    })?;
                    accumulate_point(&mut rows, &shape, &d, jacobian);
                }
            }
        }
    }

    let triplets: Vec<(usize, usize, f64)> = rows
        .into_iter()
        .enumerate()
        .flat_map(|(row, entries)| {
            entries
                .into_iter()
                .map(move |(col, value)| (row, col, value))
        })
        .collect();
    Ok(CsrMatrix::from_triplets(dof, dof, &triplets))
}
