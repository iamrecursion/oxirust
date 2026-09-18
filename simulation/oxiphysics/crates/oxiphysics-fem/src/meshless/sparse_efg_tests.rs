// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the sparse EFG stiffness assembly (F1242): analytic MLS gradients on
//! k-d tree support sets must reproduce the old dense finite-difference matrix and
//! satisfy exact consistency properties that the finite differences only
//! approximated.

use super::sparse_efg::shape_set;
use super::*;
use crate::error::Error;

const ALL_WEIGHTS: [WeightType; 3] = [
    WeightType::CubicSpline,
    WeightType::Gaussian,
    WeightType::QuarticSpline,
];

/// Nodes of a regular `nx x ny` grid over the unit square.
fn unit_square_nodes(nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            xs.push(i as f64 / (nx - 1) as f64);
            ys.push(j as f64 / (ny - 1) as f64);
        }
    }
    (xs, ys)
}

fn unit_square_method(
    nx: usize,
    ny: usize,
    cells: usize,
    radius: f64,
    wtype: WeightType,
) -> EfgMethod {
    let (xs, ys) = unit_square_nodes(nx, ny);
    EfgMethod::new(
        xs,
        ys,
        cells,
        cells,
        [0.0, 1.0, 0.0, 1.0],
        1.0e6,
        0.3,
        radius,
        wtype,
    )
}

/// The dense stiffness of the previous implementation, kept as a reference: shape
/// functions over **all** nodes at three points per quadrature point and a one-sided
/// finite-difference gradient with step `1e-5 · radius`.
fn finite_difference_stiffness(efg: &EfgMethod) -> Vec<f64> {
    let n = efg.num_nodes();
    let dof = 2 * n;
    let mut k = vec![0.0_f64; dof * dof];
    let d = plane_stress_d_matrix(efg.young, efg.poisson);
    let [x_min, x_max, y_min, y_max] = efg.domain;
    let hx = (x_max - x_min) / efg.n_cells_x as f64;
    let hy = (y_max - y_min) / efg.n_cells_y as f64;
    let shape_at = |px: f64, py: f64| -> Vec<f64> {
        mls_shape_functions(
            px,
            py,
            &efg.node_x,
            &efg.node_y,
            efg.influence_radius,
            efg.weight_type,
        )
        .expect("shape functions exist at every quadrature point of the test cloud")
    };
    let gp = [-1.0_f64 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()];
    for ci in 0..efg.n_cells_x {
        for cj in 0..efg.n_cells_y {
            let x_c = x_min + (ci as f64 + 0.5) * hx;
            let y_c = y_min + (cj as f64 + 0.5) * hy;
            for &xi in &gp {
                for &eta in &gp {
                    let xq = x_c + 0.5 * hx * xi;
                    let yq = y_c + 0.5 * hy * eta;
                    let wq = 0.25 * hx * hy;
                    let phi = shape_at(xq, yq);
                    let delta = efg.influence_radius * 1e-5;
                    let phi_px = shape_at(xq + delta, yq);
                    let phi_py = shape_at(xq, yq + delta);
                    for i in 0..n {
                        let di_dx = (phi_px[i] - phi[i]) / delta;
                        let di_dy = (phi_py[i] - phi[i]) / delta;
                        let bi = [[di_dx, 0.0], [0.0, di_dy], [di_dy, di_dx]];
                        for j in 0..n {
                            let dj_dx = (phi_px[j] - phi[j]) / delta;
                            let dj_dy = (phi_py[j] - phi[j]) / delta;
                            let bj = [[dj_dx, 0.0], [0.0, dj_dy], [dj_dy, dj_dx]];
                            for a in 0..2 {
                                for b in 0..2 {
                                    let mut val = 0.0;
                                    for p in 0..3 {
                                        for q in 0..3 {
                                            val += bi[p][a] * d[p][q] * bj[q][b];
                                        }
                                    }
                                    k[(2 * i + a) * dof + (2 * j + b)] += wq * val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    k
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn mat_vec(k: &[f64], x: &[f64]) -> Vec<f64> {
    let dof = x.len();
    (0..dof)
        .map(|row| (0..dof).map(|col| k[row * dof + col] * x[col]).sum())
        .collect()
}

/// `xᵀ K x` of the dense matrix.
fn energy(k: &[f64], x: &[f64]) -> f64 {
    x.iter().zip(mat_vec(k, x)).map(|(a, b)| a * b).sum()
}

#[test]
fn analytic_sparse_stiffness_agrees_with_the_finite_difference_reference() {
    // The acceptance test of the finding: agreement to 1e-4 on 5 x 5 nodes.
    for wtype in ALL_WEIGHTS {
        let efg = unit_square_method(5, 5, 4, 0.6, wtype);
        let reference = finite_difference_stiffness(&efg);
        let dense = efg_stiffness(&efg).expect("support covers every quadrature point");
        assert_eq!(dense.len(), reference.len());
        let scale = max_abs(&reference);
        assert!(scale > 1.0e5, "{wtype:?}: scale {scale}");
        let worst = dense
            .iter()
            .zip(reference.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst < 1e-4 * scale,
            "{wtype:?}: max |K - K_fd| = {worst:e} (max |K| = {scale:e})"
        );
    }
}

#[test]
fn the_dense_wrapper_is_the_sparse_matrix() {
    let efg = unit_square_method(6, 6, 5, 0.5, WeightType::CubicSpline);
    let sparse = efg_stiffness_sparse(&efg).expect("support covers every quadrature point");
    let dense = efg_stiffness(&efg).expect("support covers every quadrature point");
    let dof = 2 * efg.num_nodes();
    assert_eq!((sparse.nrows, sparse.ncols), (dof, dof));
    assert_eq!(dense.len(), dof * dof);
    for row in 0..dof {
        for col in 0..dof {
            assert_eq!(
                dense[row * dof + col],
                sparse.get(row, col),
                "({row}, {col})"
            );
        }
    }
}

#[test]
fn stiffness_is_symmetric_to_rounding() {
    // The finite-difference matrix was symmetric only to ~1e-4; the analytic one is
    // the sum of exactly symmetric blocks.
    for wtype in ALL_WEIGHTS {
        let efg = unit_square_method(5, 5, 4, 0.6, wtype);
        let k = efg_stiffness(&efg).expect("support covers every quadrature point");
        let dof = 2 * efg.num_nodes();
        let scale = max_abs(&k);
        for i in 0..dof {
            for j in (i + 1)..dof {
                let asymmetry = (k[i * dof + j] - k[j * dof + i]).abs();
                assert!(
                    asymmetry < 1e-12 * scale,
                    "{wtype:?}: ({i},{j}): {asymmetry:e}"
                );
            }
        }
    }
}

#[test]
fn rigid_motions_and_constant_strain_are_reproduced_exactly() {
    // MLS with a linear basis reproduces linear fields, so for u_I = u(x_I) the strain
    // at every quadrature point is exact: rigid motions carry no force, and a constant
    // strain ε has the energy uᵀ K u = ε : D ε · area to rounding.
    for wtype in ALL_WEIGHTS {
        let efg = unit_square_method(7, 7, 6, 0.5, wtype);
        let k = efg_stiffness(&efg).expect("support covers every quadrature point");
        let scale = max_abs(&k);
        let d = plane_stress_d_matrix(efg.young, efg.poisson);
        let field = |f: &dyn Fn(f64, f64) -> [f64; 2]| -> Vec<f64> {
            efg.node_x
                .iter()
                .zip(efg.node_y.iter())
                .flat_map(|(&x, &y)| f(x, y))
                .collect()
        };

        let translation = field(&|_, _| [1.0, -2.0]);
        let rotation = field(&|x, y| [-y, x]);
        for (name, u) in [("translation", &translation), ("rotation", &rotation)] {
            let force = mat_vec(&k, u);
            assert!(
                max_abs(&force) < 1e-9 * scale,
                "{wtype:?} {name}: |K u| = {:e}",
                max_abs(&force)
            );
        }

        // ε_xx = 1: energy D11; γ_xy = 1: energy D33; ε_yy = 1: energy D22
        let stretch_x = field(&|x, _| [x, 0.0]);
        let stretch_y = field(&|_, y| [0.0, y]);
        let shear = field(&|_, y| [y, 0.0]);
        for (name, u, expected) in [
            ("ε_xx", &stretch_x, d[0][0]),
            ("ε_yy", &stretch_y, d[1][1]),
            ("γ_xy", &shear, d[2][2]),
        ] {
            let e = energy(&k, u);
            assert!(
                (e - expected).abs() < 1e-9 * expected,
                "{wtype:?} {name}: uᵀKu = {e}, expected {expected}"
            );
        }
        // mixed strain: ε_xx = 1, ε_yy = 1 -> D11 + 2 D12 + D22
        let biaxial = field(&|x, y| [x, y]);
        let expected = d[0][0] + 2.0 * d[0][1] + d[1][1];
        assert!((energy(&k, &biaxial) - expected).abs() < 1e-9 * expected);
    }
}

/// Central finite difference of the value-only shape functions.
fn central_difference(
    xs: &[f64],
    ys: &[f64],
    h: f64,
    wtype: WeightType,
    (x, y): (f64, f64),
    step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let shape = |px: f64, py: f64| {
        mls_shape_functions(px, py, xs, ys, h, wtype).expect("shape functions exist")
    };
    let (xp, xm) = (shape(x + step, y), shape(x - step, y));
    let (yp, ym) = (shape(x, y + step), shape(x, y - step));
    let dx = xp
        .iter()
        .zip(xm.iter())
        .map(|(a, b)| (a - b) / (2.0 * step))
        .collect();
    let dy = yp
        .iter()
        .zip(ym.iter())
        .map(|(a, b)| (a - b) / (2.0 * step))
        .collect();
    (dx, dy)
}

/// A node cloud (`x`, `y`) with the points at which its gradients are probed.
type Probe<'a> = (&'a [f64], &'a [f64], &'a [(f64, f64)]);

#[test]
fn analytic_gradients_match_central_differences_in_every_layout() {
    let h = 0.6;
    let step = 1e-6 * h;
    // (nodes, evaluation points): a 2-D cloud, a cloud along x, a cloud along y
    let (gx, gy) = unit_square_nodes(6, 6);
    let line: Vec<f64> = (0..8).map(|i| i as f64 * 0.15).collect();
    let zeros = vec![0.0; line.len()];
    let clouds: [Probe; 3] = [
        (
            &gx,
            &gy,
            &[(0.37, 0.41), (0.05, 0.93), (0.5, 0.5), (0.81, 0.07)],
        ),
        (&line, &zeros, &[(0.52, 0.02), (0.33, -0.1), (0.91, 0.0)]),
        (&zeros, &line, &[(0.03, 0.52), (-0.1, 0.33), (0.02, 0.91)]),
    ];
    for wtype in ALL_WEIGHTS {
        for (xs, ys, points) in &clouds {
            let layout = basis_layout(xs, ys, h);
            let all: Vec<usize> = (0..xs.len()).collect();
            for &(x, y) in *points {
                let analytic =
                    shape_set(x, y, &all, xs, ys, h, wtype, layout).expect("shape functions exist");
                let (dx, dy) = central_difference(xs, ys, h, wtype, (x, y), step);
                for i in 0..xs.len() {
                    assert!(
                        (analytic.dphi_dx[i] - dx[i]).abs() < 1e-6 * (1.0 + dx[i].abs()),
                        "{wtype:?} {layout:?} ({x},{y}) node {i}: dx {} vs {}",
                        analytic.dphi_dx[i],
                        dx[i]
                    );
                    assert!(
                        (analytic.dphi_dy[i] - dy[i]).abs() < 1e-6 * (1.0 + dy[i].abs()),
                        "{wtype:?} {layout:?} ({x},{y}) node {i}: dy {} vs {}",
                        analytic.dphi_dy[i],
                        dy[i]
                    );
                }
            }
        }
    }
}

#[test]
fn gradients_satisfy_the_completeness_identities() {
    // Σ Φ_I = 1 and Σ Φ_I x_I = x differentiate to Σ ∇Φ_I = 0 and Σ ∇Φ_I x_I = e_x
    // (likewise for y): exact identities of the analytic gradient.
    let h = 0.55;
    let (xs, ys) = unit_square_nodes(6, 6);
    let layout = basis_layout(&xs, &ys, h);
    assert_eq!(layout, BasisLayout::Plane);
    let all: Vec<usize> = (0..xs.len()).collect();
    let weighted = |values: &[f64], coordinates: &[f64]| -> f64 {
        values
            .iter()
            .zip(coordinates.iter())
            .map(|(v, c)| v * c)
            .sum()
    };
    for wtype in ALL_WEIGHTS {
        for &(x, y) in &[(0.31, 0.62), (0.02, 0.5), (0.97, 0.98), (0.5, 0.5)] {
            let s = shape_set(x, y, &all, &xs, &ys, h, wtype, layout).expect("exists");
            let ones = vec![1.0; xs.len()];
            assert!(
                weighted(&s.dphi_dx, &ones).abs() < 1e-10,
                "{wtype:?} ({x},{y})"
            );
            assert!(
                weighted(&s.dphi_dy, &ones).abs() < 1e-10,
                "{wtype:?} ({x},{y})"
            );
            assert!(
                (weighted(&s.dphi_dx, &xs) - 1.0).abs() < 1e-10,
                "{wtype:?} ({x},{y})"
            );
            assert!(
                weighted(&s.dphi_dx, &ys).abs() < 1e-10,
                "{wtype:?} ({x},{y})"
            );
            assert!(
                weighted(&s.dphi_dy, &xs).abs() < 1e-10,
                "{wtype:?} ({x},{y})"
            );
            assert!(
                (weighted(&s.dphi_dy, &ys) - 1.0).abs() < 1e-10,
                "{wtype:?} ({x},{y})"
            );
        }
    }
}

#[test]
fn gradient_at_a_node_is_finite_for_the_cusped_quartic_kernel() {
    // (1 − s)⁴ has a cusp at s = 0 (dw/ds = −4): the gradient there must not be 0/0.
    let (xs, ys) = unit_square_nodes(5, 5);
    let h = 0.6;
    let layout = basis_layout(&xs, &ys, h);
    let all: Vec<usize> = (0..xs.len()).collect();
    for wtype in ALL_WEIGHTS {
        // exactly at node 12 = (0.5, 0.5)
        let s = shape_set(0.5, 0.5, &all, &xs, &ys, h, wtype, layout).expect("exists");
        assert!(
            s.dphi_dx
                .iter()
                .chain(s.dphi_dy.iter())
                .all(|v| v.is_finite()),
            "{wtype:?}"
        );
        // the completeness identities still hold with the symmetric (zero) cusp gradient
        let gx_x: f64 = s.dphi_dx.iter().zip(xs.iter()).map(|(g, x)| g * x).sum();
        assert!((gx_x - 1.0).abs() < 1e-9, "{wtype:?}: {gx_x}");
    }
}

#[test]
fn kd_tree_support_reproduces_the_gradients_of_the_whole_cloud() {
    // Nodes out of range have a zero kernel: evaluating on the support subset must give
    // the gradients of the whole-cloud evaluation, and those are zero elsewhere.
    let (xs, ys) = unit_square_nodes(8, 8);
    let h = 0.35;
    let layout = basis_layout(&xs, &ys, h);
    let all: Vec<usize> = (0..xs.len()).collect();
    for wtype in ALL_WEIGHTS {
        for &(x, y) in &[(0.43, 0.29), (0.01, 0.99), (0.7, 0.7)] {
            let support: Vec<usize> = (0..xs.len())
                .filter(|&i| (xs[i] - x).hypot(ys[i] - y) <= h)
                .collect();
            assert!(support.len() >= 3 && support.len() < xs.len());
            let local = shape_set(x, y, &support, &xs, &ys, h, wtype, layout).expect("exists");
            let full = shape_set(x, y, &all, &xs, &ys, h, wtype, layout).expect("exists");
            assert_eq!(local.support, support);
            for (k, &node) in support.iter().enumerate() {
                assert!(
                    (local.dphi_dx[k] - full.dphi_dx[node]).abs() < 1e-11,
                    "{wtype:?} node {node}"
                );
                assert!(
                    (local.dphi_dy[k] - full.dphi_dy[node]).abs() < 1e-11,
                    "{wtype:?} node {node}"
                );
            }
            let outside: f64 = (0..xs.len())
                .filter(|i| !support.contains(i))
                .map(|i| full.dphi_dx[i].abs() + full.dphi_dy[i].abs())
                .sum();
            assert_eq!(outside, 0.0, "nodes out of range carry no gradient");
        }
    }
}

#[test]
fn the_layout_comes_from_the_cloud_not_from_the_support_subset() {
    // Three collinear nodes are in range of the point, but the cloud is 2-D (a far node
    // off the line): the 3 x 3 moment matrix is singular and the shape functions do not
    // exist, instead of switching to a reduced basis.
    let xs = vec![0.0, 1.0, 2.0, 0.0];
    let ys = vec![0.0, 0.0, 0.0, 10.0];
    let h = 1.5;
    let layout = basis_layout(&xs, &ys, h);
    assert_eq!(layout, BasisLayout::Plane);
    let support = [0usize, 1, 2];
    assert!(
        shape_set(
            1.0,
            0.2,
            &support,
            &xs,
            &ys,
            h,
            WeightType::CubicSpline,
            layout
        )
        .is_none()
    );
    // and the sparse assembly reports it like the value-only evaluation does
    let efg = EfgMethod::new(
        xs,
        ys,
        1,
        1,
        [0.5, 1.5, 0.0, 0.4],
        1e6,
        0.3,
        h,
        WeightType::CubicSpline,
    );
    assert!(matches!(
        efg_stiffness_sparse(&efg),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn one_dimensional_clouds_use_the_reduced_basis_in_the_stiffness() {
    // Nodes along x with y = 0 under a 2-D background grid: compared with the
    // finite-difference reference (same reduced basis).
    let xs: Vec<f64> = (0..9).map(|i| 0.125 * i as f64).collect();
    let ys = vec![0.0; xs.len()];
    let efg = EfgMethod::new(
        xs,
        ys,
        6,
        2,
        [0.0, 1.0, -0.05, 0.05],
        2.0e5,
        0.25,
        0.4,
        WeightType::CubicSpline,
    );
    let sparse = efg_stiffness(&efg).expect("1-D cloud");
    let reference = finite_difference_stiffness(&efg);
    let scale = max_abs(&reference);
    let worst = sparse
        .iter()
        .zip(reference.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst < 1e-4 * scale,
        "max |K - K_fd| = {worst:e} (max |K| = {scale:e})"
    );
}

#[test]
fn sparse_stiffness_of_a_large_cloud_is_banded_and_exact() {
    // 30 x 30 = 900 nodes (1800 DOF): the dense matrix would hold 3.2 million entries,
    // the overlapping support sets couple under a tenth of them.
    let efg = unit_square_method(30, 30, 29, 0.08, WeightType::CubicSpline);
    let k = efg_stiffness_sparse(&efg).expect("support covers every quadrature point");
    let dof = 2 * efg.num_nodes();
    assert_eq!((k.nrows, k.ncols), (dof, dof));
    assert!(
        k.nnz() < dof * dof / 10,
        "nnz = {} of {}",
        k.nnz(),
        dof * dof
    );
    assert!(k.nnz() > 20 * dof, "every DOF couples with its neighbours");

    // constant strain energy and rigid translation, evaluated with the sparse product
    let d = plane_stress_d_matrix(efg.young, efg.poisson);
    let stretch: Vec<f64> = efg.node_x.iter().flat_map(|&x| [x, 0.0]).collect();
    let force = k.mul_vec(&stretch);
    let e: f64 = stretch.iter().zip(force.iter()).map(|(a, b)| a * b).sum();
    assert!(
        (e - d[0][0]).abs() < 1e-9 * d[0][0],
        "uᵀKu = {e}, D11 = {}",
        d[0][0]
    );
    let translation: Vec<f64> = (0..dof)
        .map(|i| if i.is_multiple_of(2) { 1.0 } else { 0.0 })
        .collect();
    let rigid_force = k.mul_vec(&translation);
    let scale = k.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        max_abs(&rigid_force) < 1e-9 * scale,
        "|K t| = {:e}",
        max_abs(&rigid_force)
    );
}

#[test]
fn sparse_assembly_validates_the_method() {
    let base = unit_square_method(3, 3, 2, 0.6, WeightType::CubicSpline);
    assert!(efg_stiffness_sparse(&base).is_ok());
    let mut no_cells = base.clone();
    no_cells.n_cells_y = 0;
    assert!(matches!(
        efg_stiffness_sparse(&no_cells),
        Err(Error::InvalidInput(_))
    ));
    for radius in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
        let mut bad = base.clone();
        bad.influence_radius = radius;
        assert!(
            matches!(efg_stiffness_sparse(&bad), Err(Error::InvalidInput(_))),
            "radius {radius}"
        );
    }
    let mut ragged = base.clone();
    ragged.node_y.pop();
    assert!(matches!(
        efg_stiffness_sparse(&ragged),
        Err(Error::InvalidInput(_))
    ));
    // the corner quadrature points of the 3 x 3 cloud see a single node with radius 0.4
    let mut small = base;
    small.influence_radius = 0.4;
    match efg_stiffness_sparse(&small) {
        Err(Error::InvalidInput(message)) => {
            assert!(message.contains("quadrature point"), "{message}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}
