// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Q4 finite-element solver of the topology
//! optimisers: element matrix against the closed form of `top88.m`, exact
//! patch-test solutions, agreement of the direct and iterative solvers, the
//! node renumbering, and the error paths.

use super::*;
use crate::error::PhysicsError;
use crate::topology_optimization::{FilteringMethods, MaterialParams, MultiLoadTopOpt};

/// The element stiffness matrix of `top88.m` (Andreassen et al., 2011) for
/// unit modulus, in the same interleaved `(u_x, u_y)` ordering.
fn top88_stiffness(nu: f64) -> [[f64; 8]; 8] {
    let a11 = [
        [12.0, 3.0, -6.0, -3.0],
        [3.0, 12.0, 3.0, 0.0],
        [-6.0, 3.0, 12.0, -3.0],
        [-3.0, 0.0, -3.0, 12.0],
    ];
    let a12 = [
        [-6.0, -3.0, 0.0, 3.0],
        [-3.0, -6.0, -3.0, -6.0],
        [0.0, -3.0, -6.0, 3.0],
        [3.0, -6.0, 3.0, -6.0],
    ];
    let b11 = [
        [-4.0, 3.0, -2.0, 9.0],
        [3.0, -4.0, -9.0, 4.0],
        [-2.0, -9.0, -4.0, -3.0],
        [9.0, 4.0, -3.0, -4.0],
    ];
    let b12 = [
        [2.0, -3.0, 4.0, -9.0],
        [-3.0, 2.0, 9.0, -2.0],
        [4.0, 9.0, 2.0, 3.0],
        [-9.0, -2.0, 3.0, 2.0],
    ];
    let scale = 1.0 / (1.0 - nu * nu) / 24.0;
    let mut ke = [[0.0; 8]; 8];
    for i in 0..4 {
        for j in 0..4 {
            ke[i][j] = scale * (a11[i][j] + nu * b11[i][j]);
            ke[i][j + 4] = scale * (a12[i][j] + nu * b12[i][j]);
            ke[i + 4][j] = scale * (a12[j][i] + nu * b12[j][i]);
            ke[i + 4][j + 4] = scale * (a11[i][j] + nu * b11[i][j]);
        }
    }
    ke
}

/// All degrees of freedom of the nodes in the left column.
fn left_edge(model: &Q4Model) -> Vec<usize> {
    let rows = model.grid().ny;
    (0..=rows)
        .flat_map(|row| {
            let node = model.node_index(row, 0);
            [model.dof(node, 0), model.dof(node, 1)]
        })
        .collect()
}

/// Deterministic moduli in `[0.2, 1.0]`.
fn graded_modulus(n_elem: usize) -> Vec<f64> {
    (0..n_elem)
        .map(|e| 0.2 + 0.8 * ((e * 7 + 3) % 11) as f64 / 10.0)
        .collect()
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |peak, v| peak.max(v.abs()))
}

// ---------------------------------------------------------------------------
// Element matrix
// ---------------------------------------------------------------------------

#[test]
fn element_stiffness_equals_the_closed_form_of_top88() {
    for nu in [0.0, 0.3, 0.45] {
        let gauss = q4_element_stiffness(nu);
        let closed = top88_stiffness(nu);
        for (i, (gauss_row, closed_row)) in gauss.iter().zip(closed.iter()).enumerate() {
            for (j, (&got, &want)) in gauss_row.iter().zip(closed_row.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-14,
                    "nu {nu}, entry ({i}, {j}): {got} vs {want}"
                );
            }
        }
    }
    let nu = 0.3;
    let ke = q4_element_stiffness(nu);
    assert!((ke[0][0] - (12.0 - 4.0 * nu) / (24.0 * (1.0 - nu * nu))).abs() < 1e-15);
    assert!((ke[0][1] - (3.0 + 3.0 * nu) / (24.0 * (1.0 - nu * nu))).abs() < 1e-15);
}

#[test]
fn element_stiffness_is_symmetric_with_exactly_the_rigid_body_kernel() {
    let nu = 0.3;
    let ke = q4_element_stiffness(nu);
    for (i, row) in ke.iter().enumerate() {
        for (j, &entry) in row.iter().enumerate() {
            assert!((entry - ke[j][i]).abs() < 1e-15, "entry ({i}, {j})");
        }
    }
    let apply = |u: &[f64; 8]| -> [f64; 8] {
        let mut out = [0.0; 8];
        for (row, entry) in ke.iter().zip(out.iter_mut()) {
            *entry = row.iter().zip(u.iter()).map(|(k, v)| k * v).sum();
        }
        out
    };
    let energy = |u: &[f64; 8]| -> f64 { u.iter().zip(apply(u).iter()).map(|(a, b)| a * b).sum() };

    // Nodes: lower-left, lower-right, upper-right, upper-left.
    let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let mut translation_x = [0.0; 8];
    let mut translation_y = [0.0; 8];
    let mut rotation = [0.0; 8];
    let mut stretch_x = [0.0; 8];
    let mut stretch_y = [0.0; 8];
    let mut shear = [0.0; 8];
    for (node, &(x, y)) in corners.iter().enumerate() {
        translation_x[2 * node] = 1.0;
        translation_y[2 * node + 1] = 1.0;
        rotation[2 * node] = -y;
        rotation[2 * node + 1] = x;
        stretch_x[2 * node] = x;
        stretch_y[2 * node + 1] = y;
        shear[2 * node] = y;
    }
    for mode in [translation_x, translation_y, rotation] {
        assert!(
            apply(&mode).iter().all(|force| force.abs() < 1e-14),
            "rigid-body mode carries force"
        );
    }
    // Uniform strains: the energy is the plane-stress constitutive entry.
    assert!((energy(&stretch_x) - 1.0 / (1.0 - nu * nu)).abs() < 1e-14);
    assert!((energy(&stretch_y) - 1.0 / (1.0 - nu * nu)).abs() < 1e-14);
    assert!((energy(&shear) - 1.0 / (2.0 * (1.0 + nu))).abs() < 1e-14);

    // Exactly three zero eigenvalues (the rigid-body modes), five positive.
    let matrix = nalgebra::SMatrix::<f64, 8, 8>::from_fn(|i, j| ke[i][j]);
    let eigenvalues = matrix.symmetric_eigen().eigenvalues;
    let zero = eigenvalues
        .iter()
        .filter(|value| value.abs() < 1e-12)
        .count();
    let positive = eigenvalues.iter().filter(|&&value| value > 1e-3).count();
    assert_eq!((zero, positive), (3, 5), "eigenvalues {eigenvalues:?}");
}

// ---------------------------------------------------------------------------
// Mesh numbering
// ---------------------------------------------------------------------------

#[test]
fn nodes_and_element_dofs_follow_the_documented_numbering() {
    let model = Q4Model::new(GridSize::new(4, 3), 0.3).unwrap();
    assert_eq!(model.n_nodes(), 20);
    assert_eq!(model.n_dofs(), 40);
    assert_eq!(model.node_index(2, 3), 2 * 5 + 3);
    assert_eq!(model.dof(13, 0), 26);
    assert_eq!(model.dof(13, 1), 27);
    // Element (row 1, col 2): lower-left node 1*5+2 = 7, then 8, 13, 12.
    assert_eq!(model.element_dofs(4 + 2), [14, 15, 16, 17, 26, 27, 24, 25]);
    assert_eq!(model.grid(), GridSize::new(4, 3));
    assert!((model.poisson_ratio() - 0.3).abs() < 1e-15);
    assert_eq!(model.unit_element_stiffness(), &q4_element_stiffness(0.3));
}

#[test]
fn internal_numbering_is_a_permutation_with_a_bandwidth_set_by_the_shorter_side() {
    for (nx, ny) in [(8, 3), (3, 8), (5, 5), (1, 1), (1, 5), (6, 1)] {
        let model = Q4Model::new(GridSize::new(nx, ny), 0.3).unwrap();
        let mut seen = vec![false; model.n_dofs()];
        for &internal in &model.to_internal {
            assert!(
                !seen[internal],
                "{nx}x{ny}: internal DOF {internal} used twice"
            );
            seen[internal] = true;
        }
        assert!(
            seen.iter().all(|&used| used),
            "{nx}x{ny}: not a permutation"
        );
        assert_eq!(model.half_bandwidth, 2 * nx.min(ny) + 5, "{nx}x{ny}");
        // The CSR pattern is symmetric and sorted within every row.
        for row in 0..model.n_dofs() {
            let columns = &model.col_idx[model.row_ptr[row]..model.row_ptr[row + 1]];
            assert!(columns.windows(2).all(|pair| pair[0] < pair[1]));
            for &column in columns {
                let mirror = &model.col_idx[model.row_ptr[column]..model.row_ptr[column + 1]];
                assert!(
                    mirror.binary_search(&row).is_ok(),
                    "{nx}x{ny}: asymmetric pattern"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exact solutions
// ---------------------------------------------------------------------------

#[test]
fn uniaxial_tension_patch_test_reproduces_the_exact_displacement_field() {
    let (nx, ny) = (5, 4);
    let nu = 0.3;
    let young = 2.5;
    let sigma = 0.7;
    let model = Q4Model::new(GridSize::new(nx, ny), nu).unwrap();
    // Consistent nodal forces of the uniform traction `sigma` on the right
    // edge: sigma per node, half of it at the two corners.
    let mut loads = vec![0.0; model.n_dofs()];
    for row in 0..=ny {
        let share = if row == 0 || row == ny { 0.5 } else { 1.0 };
        loads[model.dof(model.node_index(row, nx), 0)] = sigma * share;
    }
    // Symmetry condition on the left edge plus one vertical support.
    let mut fixed: Vec<usize> = (0..=ny)
        .map(|row| model.dof(model.node_index(row, 0), 0))
        .collect();
    fixed.push(model.dof(model.node_index(0, 0), 1));

    for solver in [
        LinearSolver::Direct,
        LinearSolver::Pcg {
            tolerance: 1e-13,
            max_iterations: 10_000,
        },
    ] {
        let solution = model
            .solve(&vec![young; nx * ny], &loads, &fixed, solver)
            .unwrap();
        for row in 0..=ny {
            for col in 0..=nx {
                let node = model.node_index(row, col);
                let expected_x = sigma * col as f64 / young;
                let expected_y = -nu * sigma * row as f64 / young;
                let got_x = solution.displacement[model.dof(node, 0)];
                let got_y = solution.displacement[model.dof(node, 1)];
                assert!(
                    (got_x - expected_x).abs() < 1e-9,
                    "{solver:?} u_x at ({row}, {col}): {got_x} vs {expected_x}"
                );
                assert!(
                    (got_y - expected_y).abs() < 1e-9,
                    "{solver:?} u_y at ({row}, {col}): {got_y} vs {expected_y}"
                );
            }
        }
        // Work of the loads: (sigma ny) * (sigma nx / E).
        let expected_compliance = sigma * sigma * (nx * ny) as f64 / young;
        assert!((solution.compliance - expected_compliance).abs() < 1e-9 * expected_compliance);
        // Every element carries the same energy: twice the strain energy of
        // the uniform field, E * sigma^2 / E^2 per unit area.
        let per_element = sigma * sigma / (young * young);
        for &energy in &solution.element_energy {
            assert!((energy - per_element).abs() < 1e-9 * per_element);
        }
        assert!(solution.relative_residual < 1e-10);
    }
}

#[test]
fn compliance_is_the_sum_of_the_modulus_weighted_element_energies() {
    let model = Q4Model::new(GridSize::new(9, 6), 0.3).unwrap();
    let modulus = graded_modulus(54);
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(6, 9), 1)] = -1.0;
    loads[model.dof(model.node_index(3, 9), 0)] = 0.4;
    let fixed = left_edge(&model);
    let solution = model
        .solve(&modulus, &loads, &fixed, LinearSolver::Direct)
        .unwrap();
    let from_energies: f64 = modulus
        .iter()
        .zip(solution.element_energy.iter())
        .map(|(young, energy)| young * energy)
        .sum();
    assert!(solution.compliance > 0.0);
    assert!(
        (from_energies - solution.compliance).abs() < 1e-9 * solution.compliance,
        "{from_energies} vs {}",
        solution.compliance
    );
    // Displacements vanish at the supports.
    for &dof in &fixed {
        assert_eq!(solution.displacement[dof], 0.0);
    }
}

#[test]
fn compliance_scales_inversely_with_the_modulus_and_quadratically_with_the_load() {
    let model = Q4Model::new(GridSize::new(7, 4), 0.3).unwrap();
    let base = graded_modulus(28);
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(4, 7), 1)] = -1.0;
    let fixed = left_edge(&model);
    let reference = model
        .solve(&base, &loads, &fixed, LinearSolver::Direct)
        .unwrap();

    let stiffer: Vec<f64> = base.iter().map(|young| 4.0 * young).collect();
    let heavier: Vec<f64> = loads.iter().map(|force| 3.0 * force).collect();
    let scaled = model
        .solve(&stiffer, &heavier, &fixed, LinearSolver::Direct)
        .unwrap();
    // c = f^T K^-1 f: factor 3^2 / 4.
    assert!(
        (scaled.compliance - reference.compliance * 9.0 / 4.0).abs() < 1e-10 * scaled.compliance
    );
    // u_e = 3/4 u, so the unit-modulus energy scales with (3/4)^2.
    let peak = max_abs(&reference.element_energy);
    for (a, b) in scaled
        .element_energy
        .iter()
        .zip(reference.element_energy.iter())
    {
        assert!((a - b * 9.0 / 16.0).abs() < 1e-9 * a.abs() + 1e-12 * peak);
    }
    // The reversed load reverses the displacements and keeps the energies.
    let reversed: Vec<f64> = loads.iter().map(|force| -force).collect();
    let flipped = model
        .solve(&base, &reversed, &fixed, LinearSolver::Direct)
        .unwrap();
    for (a, b) in flipped
        .displacement
        .iter()
        .zip(reference.displacement.iter())
    {
        assert!((a + b).abs() < 1e-12 * (1.0 + b.abs()));
    }
    assert!((flipped.compliance - reference.compliance).abs() < 1e-10 * reference.compliance);
}

#[test]
fn transposing_the_structure_leaves_the_compliance_and_the_displacements_unchanged() {
    // 8 x 3 uses the row-fastest renumbering, 3 x 8 the user numbering.
    let (wide_nx, wide_ny) = (8, 3);
    let wide = Q4Model::new(GridSize::new(wide_nx, wide_ny), 0.3).unwrap();
    let tall = Q4Model::new(GridSize::new(wide_ny, wide_nx), 0.3).unwrap();
    let modulus_wide: Vec<f64> = (0..wide_nx * wide_ny)
        .map(|e| {
            let (row, col) = (e / wide_nx, e % wide_nx);
            1.0 + 0.3 * ((row * 5 + col * 3) % 7) as f64
        })
        .collect();
    // Element (row, col) of the wide grid is element (col, row) of the tall one.
    let mut modulus_tall = vec![0.0; wide_nx * wide_ny];
    for row in 0..wide_ny {
        for col in 0..wide_nx {
            modulus_tall[col * wide_ny + row] = modulus_wide[row * wide_nx + col];
        }
    }
    // A force (f_x, f_y) at node (row, col) becomes (f_y, f_x) at node (col, row).
    let mut loads_wide = vec![0.0; wide.n_dofs()];
    let mut loads_tall = vec![0.0; tall.n_dofs()];
    for &(row, col, fx, fy) in &[(3, 8, 0.2, -1.0), (1, 5, 0.7, 0.3), (2, 2, -0.4, 0.5)] {
        loads_wide[wide.dof(wide.node_index(row, col), 0)] += fx;
        loads_wide[wide.dof(wide.node_index(row, col), 1)] += fy;
        loads_tall[tall.dof(tall.node_index(col, row), 0)] += fy;
        loads_tall[tall.dof(tall.node_index(col, row), 1)] += fx;
    }
    // Clamp the left edge of the wide grid, the bottom edge of the tall one.
    let fixed_wide = left_edge(&wide);
    let fixed_tall: Vec<usize> = (0..=wide_ny)
        .flat_map(|col| {
            let node = tall.node_index(0, col);
            [tall.dof(node, 0), tall.dof(node, 1)]
        })
        .collect();
    let a = wide
        .solve(
            &modulus_wide,
            &loads_wide,
            &fixed_wide,
            LinearSolver::Direct,
        )
        .unwrap();
    let b = tall
        .solve(
            &modulus_tall,
            &loads_tall,
            &fixed_tall,
            LinearSolver::Direct,
        )
        .unwrap();
    assert!((a.compliance - b.compliance).abs() < 1e-10 * a.compliance);
    for row in 0..=wide_ny {
        for col in 0..=wide_nx {
            for component in 0..2 {
                let u_wide = a.displacement[wide.dof(wide.node_index(row, col), component)];
                let u_tall = b.displacement[tall.dof(tall.node_index(col, row), 1 - component)];
                assert!(
                    (u_wide - u_tall).abs() < 1e-10 * (1.0 + u_wide.abs()),
                    "node ({row}, {col}) component {component}: {u_wide} vs {u_tall}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Solvers
// ---------------------------------------------------------------------------

#[test]
fn direct_and_conjugate_gradient_solvers_agree_and_warm_starts_converge_at_once() {
    let model = Q4Model::new(GridSize::new(9, 6), 0.3).unwrap();
    let modulus = graded_modulus(54);
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(6, 9), 0)] = 0.3;
    loads[model.dof(model.node_index(6, 9), 1)] = -1.0;
    loads[model.dof(model.node_index(3, 4), 0)] = 0.5;
    loads[model.dof(model.node_index(3, 4), 1)] = 0.2;
    let fixed = left_edge(&model);

    let direct = model
        .solve(&modulus, &loads, &fixed, LinearSolver::Direct)
        .unwrap();
    assert_eq!(direct.iterations, 0);
    assert!(
        direct.relative_residual < 1e-12,
        "{}",
        direct.relative_residual
    );

    let pcg = LinearSolver::Pcg {
        tolerance: 1e-13,
        max_iterations: 20_000,
    };
    let iterative = model.solve(&modulus, &loads, &fixed, pcg).unwrap();
    assert!(iterative.iterations > 0);
    assert!(
        iterative.relative_residual < 1e-11,
        "{}",
        iterative.relative_residual
    );
    let scale = max_abs(&direct.displacement);
    for (a, b) in direct
        .displacement
        .iter()
        .zip(iterative.displacement.iter())
    {
        assert!((a - b).abs() < 1e-9 * scale, "{a} vs {b}");
    }
    assert!((direct.compliance - iterative.compliance).abs() < 1e-9 * direct.compliance);

    // Starting from the converged displacements the iteration stops at once.
    let analysis = model
        .analyse(
            &modulus,
            &fixed,
            LinearSolver::Pcg {
                tolerance: 1e-10,
                max_iterations: 5_000,
            },
        )
        .unwrap();
    let warm = analysis
        .solve_from(&loads, Some(&direct.displacement))
        .unwrap();
    assert_eq!(warm.iterations, 0);
    // A guess of the wrong length is ignored rather than trusted.
    let cold = analysis.solve_from(&loads, Some(&[1.0, 2.0])).unwrap();
    assert!(cold.iterations > 0);
    // The analysis solves further load vectors without reassembly.
    let mut second = vec![0.0; model.n_dofs()];
    second[model.dof(model.node_index(0, 9), 1)] = 1.0;
    let again = analysis.solve(&second).unwrap();
    let reference = model
        .solve(&modulus, &second, &fixed, LinearSolver::Direct)
        .unwrap();
    assert!((again.compliance - reference.compliance).abs() < 1e-7 * reference.compliance);
}

#[test]
fn the_default_solver_is_direct_for_small_models() {
    assert_eq!(LinearSolver::default(), LinearSolver::Auto);
    let model = Q4Model::new(GridSize::new(6, 4), 0.3).unwrap();
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(4, 6), 1)] = -1.0;
    let fixed = left_edge(&model);
    let solution = model
        .solve(&[1.0; 24], &loads, &fixed, LinearSolver::Auto)
        .unwrap();
    assert_eq!(solution.iterations, 0, "Auto must factorise small models");
}

#[test]
fn auto_factorises_models_up_to_300_by_100_and_iterates_beyond() {
    // 300 x 100: 60802 unknowns, half-bandwidth 205, 12.5 million band entries.
    assert!(prefers_direct_factorisation(2 * 301 * 101, 2 * 100 + 5));
    assert!(prefers_direct_factorisation(8, 7));
    // 400 x 400: 321602 unknowns, half-bandwidth 805, 259 million entries.
    assert!(!prefers_direct_factorisation(2 * 401 * 401, 2 * 400 + 5));
    assert!(!prefers_direct_factorisation(usize::MAX, usize::MAX - 1));

    // The model of the large grid really takes the iterative path (assembly
    // and preconditioner only, no solve).
    let grid = GridSize::new(400, 400);
    let model = Q4Model::new(grid, 0.3).unwrap();
    assert_eq!(model.half_bandwidth, 805);
    let fixed: Vec<usize> = (0..=grid.ny)
        .map(|row| model.dof(model.node_index(row, 0), 0))
        .chain([model.dof(model.node_index(0, 0), 1)])
        .collect();
    let analysis = model
        .analyse(&vec![1.0; grid.n_elem()], &fixed, LinearSolver::Auto)
        .unwrap();
    assert!(matches!(analysis.operator, Operator::Iterative { .. }));
    let small = Q4Model::new(GridSize::new(20, 10), 0.3).unwrap();
    let direct = small
        .analyse(&vec![1.0; 200], &[0, 1, 3], LinearSolver::Auto)
        .unwrap();
    assert!(matches!(direct.operator, Operator::Direct(_)));
}

#[test]
fn the_direct_solver_survives_the_stiffness_contrast_of_simp_designs() {
    let (nx, ny) = (12, 6);
    let model = Q4Model::new(GridSize::new(nx, ny), 0.3).unwrap();
    // A solid L (left column and bottom row), the rest void at 1e-9.
    let modulus: Vec<f64> = (0..nx * ny)
        .map(|e| {
            let (row, col) = (e / nx, e % nx);
            if row == 0 || col == 0 { 1.0 } else { 1e-9 }
        })
        .collect();
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(0, nx), 1)] = -1.0;
    let fixed = left_edge(&model);
    let direct = model
        .solve(&modulus, &loads, &fixed, LinearSolver::Direct)
        .unwrap();
    assert!(direct.compliance.is_finite() && direct.compliance > 0.0);
    assert!(
        direct.relative_residual < 1e-8,
        "{}",
        direct.relative_residual
    );

    let iterative = model
        .solve(
            &modulus,
            &loads,
            &fixed,
            LinearSolver::Pcg {
                tolerance: 1e-11,
                max_iterations: 200_000,
            },
        )
        .unwrap();
    assert!(
        (direct.compliance - iterative.compliance).abs() < 1e-6 * direct.compliance,
        "{} vs {}",
        direct.compliance,
        iterative.compliance
    );
}

#[test]
fn zero_loads_give_zero_displacements_without_iterating() {
    let model = Q4Model::new(GridSize::new(4, 3), 0.3).unwrap();
    let fixed = left_edge(&model);
    for solver in [
        LinearSolver::Direct,
        LinearSolver::Pcg {
            tolerance: 1e-10,
            max_iterations: 10,
        },
    ] {
        let solution = model
            .solve(&[1.0; 12], &vec![0.0; model.n_dofs()], &fixed, solver)
            .unwrap();
        assert_eq!(solution.compliance, 0.0);
        assert_eq!(solution.iterations, 0);
        assert!(solution.displacement.iter().all(|&u| u == 0.0));
        assert!(solution.element_energy.iter().all(|&e| e == 0.0));
    }
}

// ---------------------------------------------------------------------------
// Sensitivities through the optimiser interface
// ---------------------------------------------------------------------------

#[test]
fn simp_oracle_gradient_matches_finite_differences_of_the_fe_compliance() {
    let (nx, ny) = (6, 4);
    let grid = GridSize::new(nx, ny);
    let simp = SiMP::new(grid, 0.5, 3.0, 1.5, MaterialParams::default());
    let model = Q4Model::new(grid, simp.material.nu).unwrap();
    let fixed = left_edge(&model);
    let mut tip_down = vec![0.0; model.n_dofs()];
    tip_down[model.dof(model.node_index(ny, nx), 1)] = -1.0;
    let mut tip_side = vec![0.0; model.n_dofs()];
    tip_side[model.dof(model.node_index(2, nx), 0)] = 0.8;
    tip_side[model.dof(model.node_index(2, nx), 1)] = 0.3;
    let optimiser = MultiLoadTopOpt::new(
        simp.clone(),
        FilteringMethods::new(1.5, grid),
        vec![LoadCase::new(tip_down, 0.6), LoadCase::new(tip_side, 0.4)],
    );
    let density: Vec<f64> = (0..nx * ny)
        .map(|e| 0.35 + 0.5 * ((e * 5) % 7) as f64 / 7.0)
        .collect();

    let (compliance, gradient) = optimiser
        .evaluate(
            &density,
            model.simp_oracle(&simp, &fixed, LinearSolver::Direct),
        )
        .unwrap();
    assert!(compliance > 0.0);
    assert!(gradient.iter().all(|&g| g < 0.0), "{gradient:?}");

    let weighted_compliance = |field: &[f64]| -> f64 {
        optimiser
            .evaluate(
                field,
                model.simp_oracle(&simp, &fixed, LinearSolver::Direct),
            )
            .unwrap()
            .0
    };
    let step = 1e-5;
    for e in 0..nx * ny {
        let mut plus = density.clone();
        plus[e] += step;
        let mut minus = density.clone();
        minus[e] -= step;
        let finite_difference =
            (weighted_compliance(&plus) - weighted_compliance(&minus)) / (2.0 * step);
        assert!(
            (finite_difference - gradient[e]).abs() <= 1e-5 * gradient[e].abs().max(1e-2),
            "element {e}: analytic {} vs finite difference {finite_difference}",
            gradient[e]
        );
    }
}

#[test]
fn simp_oracle_reuses_the_factorisation_for_equal_densities_only() {
    let grid = GridSize::new(5, 3);
    let simp = SiMP::new(grid, 0.5, 3.0, 1.5, MaterialParams::default());
    let model = Q4Model::new(grid, simp.material.nu).unwrap();
    let fixed = left_edge(&model);
    let mut oracle = model.simp_oracle(&simp, &fixed, LinearSolver::Direct);
    let mut load = vec![0.0; model.n_dofs()];
    load[model.dof(model.node_index(3, 5), 1)] = -1.0;
    let case = LoadCase::new(load.clone(), 1.0);

    let first_density = [0.5; 15];
    let second_density: Vec<f64> = (0..15).map(|e| 0.3 + 0.04 * e as f64).collect();
    let (c1, e1) = oracle(&first_density, &case).unwrap();
    let (c1_again, _) = oracle(&first_density, &case).unwrap();
    let (c2, e2) = oracle(&second_density, &case).unwrap();
    let (c1_third, e1_third) = oracle(&first_density, &case).unwrap();
    assert_eq!(c1, c1_again);
    assert_eq!(c1, c1_third);
    assert_eq!(e1, e1_third);
    assert!(
        (c1 - c2).abs() > 1e-3 * c1,
        "different densities must give different results"
    );
    assert_ne!(e1, e2);

    // The closure agrees with a direct solve of the same design.
    let modulus: Vec<f64> = second_density
        .iter()
        .map(|&rho| simp.interpolated_stiffness(rho))
        .collect();
    let reference = model
        .solve(&modulus, &load, &fixed, LinearSolver::Direct)
        .unwrap();
    assert!((reference.compliance - c2).abs() < 1e-12 * c2);
    assert_eq!(reference.element_energy, e2);

    // A failing solve is reported and leaves the closure usable.
    let bad_case = LoadCase::new(vec![0.0; 3], 1.0);
    assert!(matches!(
        oracle(&second_density, &bad_case),
        Err(PhysicsError::InvalidInput { .. })
    ));
    let (c2_after, _) = oracle(&second_density, &case).unwrap();
    assert_eq!(c2, c2_after);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

fn invalid_field(error: &PhysicsError) -> &str {
    match error {
        PhysicsError::InvalidInput { field, .. } => field.as_str(),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn invalid_models_are_rejected() {
    assert_eq!(
        invalid_field(&Q4Model::new(GridSize::new(0, 3), 0.3).unwrap_err()),
        "grid"
    );
    assert_eq!(
        invalid_field(&Q4Model::new(GridSize::new(3, 0), 0.3).unwrap_err()),
        "grid"
    );
    for nu in [0.5, 0.7, -1.0, f64::NAN] {
        assert!(
            matches!(
                Q4Model::new(GridSize::new(3, 3), nu),
                Err(PhysicsError::MaterialError { .. })
            ),
            "nu {nu}"
        );
    }
}

#[test]
fn malformed_systems_are_rejected_with_specific_errors() {
    let model = Q4Model::new(GridSize::new(4, 3), 0.3).unwrap();
    let fixed = left_edge(&model);
    let ok_modulus = vec![1.0; 12];

    let short = model.analyse(&[1.0; 11], &fixed, LinearSolver::Direct);
    assert_eq!(invalid_field(&short.unwrap_err()), "modulus");
    let mut negative = ok_modulus.clone();
    negative[5] = -1.0;
    let error = model
        .analyse(&negative, &fixed, LinearSolver::Direct)
        .unwrap_err();
    assert_eq!(invalid_field(&error), "modulus[5]");
    let mut not_finite = ok_modulus.clone();
    not_finite[2] = f64::NAN;
    let error = model
        .analyse(&not_finite, &fixed, LinearSolver::Direct)
        .unwrap_err();
    assert_eq!(invalid_field(&error), "modulus[2]");

    let outside = model.analyse(&ok_modulus, &[0, 1, 2, 40], LinearSolver::Direct);
    assert_eq!(
        outside.unwrap_err(),
        PhysicsError::OutOfBounds { index: 40, max: 40 }
    );

    for solver in [
        LinearSolver::Direct,
        LinearSolver::Pcg {
            tolerance: 1e-8,
            max_iterations: 100,
        },
    ] {
        let error = model.analyse(&ok_modulus, &[0, 1], solver).unwrap_err();
        assert!(
            matches!(error, PhysicsError::FemBoundaryConditionError { .. }),
            "{solver:?}: {error:?}"
        );
        // Repeated entries count once.
        let error = model
            .analyse(&ok_modulus, &[0, 0, 1, 1], solver)
            .unwrap_err();
        assert!(matches!(
            error,
            PhysicsError::FemBoundaryConditionError { .. }
        ));
    }

    for (tolerance, max_iterations) in [(0.0, 10), (-1.0, 10), (f64::NAN, 10), (1e-8, 0)] {
        let error = model
            .analyse(
                &ok_modulus,
                &fixed,
                LinearSolver::Pcg {
                    tolerance,
                    max_iterations,
                },
            )
            .unwrap_err();
        assert_eq!(
            invalid_field(&error),
            "solver",
            "{tolerance} {max_iterations}"
        );
    }

    let analysis = model
        .analyse(&ok_modulus, &fixed, LinearSolver::Direct)
        .unwrap();
    assert_eq!(
        invalid_field(&analysis.solve(&[0.0; 5]).unwrap_err()),
        "loads"
    );
    let mut infinite = vec![0.0; model.n_dofs()];
    infinite[9] = f64::INFINITY;
    assert_eq!(
        invalid_field(&analysis.solve(&infinite).unwrap_err()),
        "loads[9]"
    );
}

#[test]
fn a_support_that_leaves_a_rigid_body_motion_is_reported_as_singular() {
    let model = Q4Model::new(GridSize::new(4, 3), 0.3).unwrap();
    let modulus = [1.0; 12];
    // Node (0, 0) is pinned and node (0, 1) is held in x: it may still move
    // in y, which is a rotation about the pin.
    let pin = [
        model.dof(model.node_index(0, 0), 0),
        model.dof(model.node_index(0, 0), 1),
        model.dof(model.node_index(0, 1), 0),
    ];
    let error = model
        .analyse(&modulus, &pin, LinearSolver::Direct)
        .unwrap_err();
    assert!(
        matches!(error, PhysicsError::NumericalDivergence { .. }),
        "{error:?}"
    );
    // Holding the second node in y instead removes the rotation.
    let stable = [
        model.dof(model.node_index(0, 0), 0),
        model.dof(model.node_index(0, 0), 1),
        model.dof(model.node_index(0, 1), 1),
    ];
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(3, 4), 1)] = -1.0;
    let solution = model
        .solve(&modulus, &loads, &stable, LinearSolver::Direct)
        .unwrap();
    assert!(solution.compliance > 0.0);
}

#[test]
fn conjugate_gradients_report_non_convergence() {
    let model = Q4Model::new(GridSize::new(9, 6), 0.3).unwrap();
    let modulus = graded_modulus(54);
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(6, 9), 1)] = -1.0;
    let fixed = left_edge(&model);
    let error = model
        .solve(
            &modulus,
            &loads,
            &fixed,
            LinearSolver::Pcg {
                tolerance: 1e-14,
                max_iterations: 3,
            },
        )
        .unwrap_err();
    match error {
        PhysicsError::ConvergenceFailed {
            iterations,
            tolerance,
            residual,
        } => {
            assert_eq!(iterations, 3);
            assert_eq!(tolerance, 1e-14);
            assert!(residual > tolerance);
        }
        other => panic!("expected ConvergenceFailed, got {other:?}"),
    }
}
