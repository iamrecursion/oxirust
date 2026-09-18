// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the `O(n R²)` filter neighbourhoods, the Sigmund
//! sensitivity filter, and the finite-element driven SIMP and multi-load
//! optimisations.

use super::*;

// ---------------------------------------------------------------------------
// Filter neighbourhoods
// ---------------------------------------------------------------------------

/// The original construction that scans every pair of elements.
fn brute_force_neighbors(radius: f64, grid: GridSize) -> Vec<Vec<(usize, f64)>> {
    let n = grid.n_elem();
    let mut result = vec![Vec::new(); n];
    for (i, list) in result.iter_mut().enumerate() {
        let (ri, ci) = grid.row_col(i);
        for j in 0..n {
            let (rj, cj) = grid.row_col(j);
            let dist = ((ri as f64 - rj as f64).powi(2) + (ci as f64 - cj as f64).powi(2)).sqrt();
            if dist < radius {
                list.push((j, radius - dist));
            }
        }
    }
    result
}

#[test]
fn neighbour_lists_equal_the_all_pairs_construction_on_a_10_by_7_grid() {
    let grid = GridSize::new(10, 7);
    let filter = FilteringMethods::new(2.5, grid);
    let reference = brute_force_neighbors(2.5, grid);
    for (e, expected) in reference.iter().enumerate() {
        assert_eq!(filter.neighbors(e), expected.as_slice(), "element {e}");
        // Ascending element order, so the lists are also equal after sorting.
        assert!(
            filter
                .neighbors(e)
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0)
        );
    }
    // An interior element of this grid sees the 21 centres within 2.5.
    assert_eq!(filter.neighbors(3 * 10 + 4).len(), 21);
}

#[test]
fn neighbour_lists_equal_the_all_pairs_construction_for_assorted_grids_and_radii() {
    let grids = [
        (10, 7),
        (7, 10),
        (1, 1),
        (1, 6),
        (6, 1),
        (3, 3),
        (12, 5),
        (4, 4),
    ];
    let radii = [
        0.0,
        0.4,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.7,
        20.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
    ];
    for &(nx, ny) in &grids {
        let grid = GridSize::new(nx, ny);
        for &radius in &radii {
            let filter = FilteringMethods::new(radius, grid);
            let reference = brute_force_neighbors(radius, grid);
            for (e, expected) in reference.iter().enumerate() {
                assert_eq!(
                    filter.neighbors(e),
                    expected.as_slice(),
                    "grid {nx}x{ny}, radius {radius}, element {e}"
                );
            }
        }
    }
}

#[test]
fn neighbour_construction_visits_only_the_filter_window() {
    // A 300 x 100 grid has 9e8 element pairs; the window of radius 3 has 25
    // elements, so the lists hold 25 entries in the interior.
    let grid = GridSize::new(300, 100);
    let filter = FilteringMethods::new(3.0, grid);
    assert_eq!(filter.neighbors(50 * 300 + 150).len(), 25);
    assert_eq!(filter.neighbors(0).len(), 9);
    assert_eq!(filter.neighbors(300 * 100 - 1).len(), 9);
    let window = |extent: usize| -> usize {
        (0..extent)
            .map(|i| (i + 2).min(extent - 1) - i.saturating_sub(2) + 1)
            .sum()
    };
    let total: usize = (0..grid.n_elem()).map(|e| filter.neighbors(e).len()).sum();
    assert_eq!(total, window(300) * window(100));
    // Weights are the hat function of the centre distance.
    let (self_index, self_weight) = filter.neighbors(0)[0];
    assert_eq!(self_index, 0);
    assert_eq!(self_weight, 3.0);
}

#[test]
fn elements_outside_the_grid_have_no_neighbours() {
    let grid = GridSize::new(4, 3);
    let filter = FilteringMethods::new(1.5, grid);
    assert!(filter.neighbors(12).is_empty());
    assert!(filter.neighbors(usize::MAX).is_empty());
    assert!(!filter.neighbors(11).is_empty());
}

#[test]
fn density_filter_and_chain_rule_keep_working_on_the_compressed_lists() {
    let grid = GridSize::new(5, 4);
    let filter = FilteringMethods::new(2.0, grid);
    let reference = brute_force_neighbors(2.0, grid);
    let density: Vec<f64> = (0..20).map(|e| 0.1 + 0.04 * e as f64).collect();
    let filtered = filter.density_filter(&density);
    for (e, list) in reference.iter().enumerate() {
        let num: f64 = list.iter().map(|&(f, h)| h * density[f]).sum();
        let den: f64 = list.iter().map(|&(_, h)| h).sum();
        assert!((filtered[e] - num / den).abs() < 1e-14, "element {e}");
    }
    // The adjoint of the (normalised) density filter conserves the total.
    let sensitivity = [1.0; 20];
    let chained = filter.filter_sensitivity_chain(&sensitivity, &density);
    let total: f64 = chained.iter().sum();
    assert!((total - 20.0).abs() < 1e-12, "{total}");
}

// ---------------------------------------------------------------------------
// Sensitivity filter
// ---------------------------------------------------------------------------

#[test]
fn sensitivity_filter_follows_sigmund_on_a_non_uniform_field() {
    let grid = GridSize::new(3, 1);
    let filter = FilteringMethods::new(1.5, grid);
    let out = filter.sensitivity_filter(&[0.1, 1.0, 1.0], &[-1.0, -1.0, -1.0]);
    // Element 0: neighbours 0 (H = 1.5) and 1 (H = 0.5):
    // (1.5 * 0.1 + 0.5 * 1) * (-1) / (0.1 * 2.0) = -3.25; the density-weighted
    // mean of the old implementation gave -1.
    assert!((out[0] + 3.25).abs() < 1e-12, "{}", out[0]);
    assert!((out[1] + 0.82).abs() < 1e-12, "{}", out[1]);
    assert!((out[2] + 1.0).abs() < 1e-12, "{}", out[2]);
}

#[test]
fn sensitivity_filter_matches_a_dense_transcription_of_top88() {
    let (nx, ny) = (6, 5);
    let grid = GridSize::new(nx, ny);
    let radius = 2.3;
    let filter = FilteringMethods::new(radius, grid);
    let n = grid.n_elem();
    let density: Vec<f64> = (0..n)
        .map(|e| 0.02 + 0.97 * ((e * 11 + 5) % 17) as f64 / 16.0)
        .collect();
    let sensitivity: Vec<f64> = (0..n)
        .map(|e| -0.1 - 2.0 * ((e * 7 + 1) % 13) as f64 / 12.0)
        .collect();
    let out = filter.sensitivity_filter(&density, &sensitivity);
    // dc(:) = H * (x(:) .* dc(:)) ./ Hs ./ max(1e-3, x(:))
    for e in 0..n {
        let (re, ce) = (e / nx, e % nx);
        let mut num = 0.0;
        let mut hs = 0.0;
        for f in 0..n {
            let (rf, cf) = (f / nx, f % nx);
            let dist =
                (((re as f64 - rf as f64).powi(2)) + ((ce as f64 - cf as f64).powi(2))).sqrt();
            let h = (radius - dist).max(0.0);
            num += h * density[f] * sensitivity[f];
            hs += h;
        }
        let expected = num / hs / density[e].max(1e-3);
        assert!(
            (out[e] - expected).abs() < 1e-12 * expected.abs().max(1.0),
            "element {e}: {} vs {expected}",
            out[e]
        );
    }
}

#[test]
fn sensitivity_filter_preserves_uniform_fields_and_is_the_identity_for_unit_radius() {
    let grid = GridSize::new(5, 4);
    let filter = FilteringMethods::new(2.0, grid);
    for rho in [0.5, 1.0] {
        let out = filter.sensitivity_filter(&[rho; 20], &[-0.7; 20]);
        assert!(
            out.iter().all(|&v| (v + 0.7).abs() < 1e-12),
            "rho {rho}: {out:?}"
        );
    }
    // Radius 1 reaches only the element itself: filtered = rho dc / max(gamma, rho) = dc.
    let unit = FilteringMethods::new(1.0, grid);
    let density: Vec<f64> = (0..20).map(|e| 0.2 + 0.04 * e as f64).collect();
    let sensitivity: Vec<f64> = (0..20).map(|e| -1.0 - 0.1 * e as f64).collect();
    let out = unit.sensitivity_filter(&density, &sensitivity);
    for (a, b) in out.iter().zip(sensitivity.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn sensitivity_filter_amplifies_low_density_elements_up_to_the_gamma_floor() {
    let grid = GridSize::new(3, 1);
    let filter = FilteringMethods::new(1.5, grid);
    // rho_0 = 0: numerator -0.5 (only the neighbour contributes), denominator
    // max(1e-3, 0) * 2.0.
    let out = filter.sensitivity_filter(&[0.0, 1.0, 1.0], &[-1.0, -1.0, -1.0]);
    assert!((out[0] + 250.0).abs() < 1e-9, "{}", out[0]);
    assert!(out.iter().all(|v| v.is_finite()));
    // Densities below gamma are treated as gamma.
    let tiny = filter.sensitivity_filter(&[1e-6, 1.0, 1.0], &[-1.0, -1.0, -1.0]);
    assert!((tiny[0] - (-(1.5e-6 + 0.5) / (1e-3 * 2.0))).abs() < 1e-9);
}

#[test]
fn sensitivity_filter_degenerate_inputs_return_the_raw_sensitivities() {
    let grid = GridSize::new(3, 2);
    let density = [0.5; 6];
    let sensitivity = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    for radius in [0.0, -1.0, f64::NAN] {
        let filter = FilteringMethods::new(radius, grid);
        assert_eq!(
            filter.sensitivity_filter(&density, &sensitivity),
            sensitivity
        );
    }
    let filter = FilteringMethods::new(1.5, grid);
    assert_eq!(
        filter.sensitivity_filter(&density[..5], &sensitivity),
        sensitivity
    );
    assert_eq!(
        filter.sensitivity_filter(&density, &sensitivity[..4]),
        sensitivity[..4]
    );
}

// ---------------------------------------------------------------------------
// SIMP optimisation with finite-element analysis
// ---------------------------------------------------------------------------

/// Half MBB beam of `top88.m`: load at the top-left corner, symmetry
/// condition on the left edge, roller at the bottom-right corner.
fn mbb_problem(
    nx: usize,
    ny: usize,
    volume: f64,
    radius: f64,
) -> (TopOptSolver, Vec<f64>, Vec<usize>) {
    let grid = GridSize::new(nx, ny);
    let simp = SiMP::new(grid, volume, 3.0, radius, MaterialParams::default());
    let solver = TopOptSolver::new(simp, FilteringMethods::new(radius, grid));
    let model = Q4Model::new(grid, 0.3).unwrap();
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(ny, 0), 1)] = -1.0;
    let mut fixed: Vec<usize> = (0..=ny)
        .map(|row| model.dof(model.node_index(row, 0), 0))
        .collect();
    fixed.push(model.dof(model.node_index(0, nx), 1));
    (solver, loads, fixed)
}

/// Cantilever with the load at the middle of the free end.
fn cantilever_problem(
    nx: usize,
    ny: usize,
    volume: f64,
    radius: f64,
) -> (TopOptSolver, Vec<f64>, Vec<usize>) {
    let grid = GridSize::new(nx, ny);
    let simp = SiMP::new(grid, volume, 3.0, radius, MaterialParams::default());
    let solver = TopOptSolver::new(simp, FilteringMethods::new(radius, grid));
    let model = Q4Model::new(grid, 0.3).unwrap();
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(ny / 2, nx), 1)] = -1.0;
    let fixed: Vec<usize> = (0..=ny)
        .flat_map(|row| {
            let node = model.node_index(row, 0);
            [model.dof(node, 0), model.dof(node, 1)]
        })
        .collect();
    (solver, loads, fixed)
}

/// Compliance history of `top88(60, 20, 0.5, 3, 1.5, 1)`, produced by a
/// line-by-line NumPy/SciPy transcription of `top88.m` (Andreassen, Clausen,
/// Schevenels, Lazarov, Sigmund: Struct. Multidisc. Optim. 43:1-16, 2011).
const TOP88_FIRST_COMPLIANCES: [f64; 5] = [
    1007.0221007162465,
    579.4186935302589,
    412.45632408465735,
    343.71618504679043,
    322.04703108518254,
];

/// Compliance of the last iteration of that run (iteration 94, `change`
/// 0.00996 < 0.01).
const TOP88_FINAL_COMPLIANCE: f64 = 203.19245764008605;

/// `(row, col, density)` of the final design of that run, row 0 being the
/// bottom of the beam.
const TOP88_FINAL_DENSITY_SAMPLES: [(usize, usize, f64); 12] = [
    (0, 0, 1.0),
    (0, 59, 1.0),
    (19, 0, 1.0),
    (19, 59, 0.0),
    (10, 30, 0.978336),
    (5, 5, 0.255670),
    (15, 10, 0.141161),
    (10, 50, 1.0),
    (2, 40, 0.266657),
    (18, 3, 1.0),
    (12, 20, 0.0),
    (7, 45, 0.0),
];

/// Mean density of the 10 x 10 blocks of the final design of that run, the
/// bottom row of blocks first.
const TOP88_FINAL_BLOCK_MEANS: [[f64; 6]; 2] = [
    [0.527464, 0.805854, 0.638443, 0.311780, 0.253016, 0.595365],
    [0.635107, 0.534920, 0.440595, 0.770733, 0.449086, 0.037275],
];

#[test]
fn mbb_beam_reproduces_top88_within_one_percent() {
    let (solver, loads, fixed) = mbb_problem(60, 20, 0.5, 1.5);
    let mut solver = solver.with_tolerance(0.01).with_max_iter(400);
    let result = solver.run(&loads, &fixed).unwrap();

    // The first iteration analyses the uniform start, independent of the update.
    let first = result.compliance_history[0];
    assert!(
        (first - TOP88_FIRST_COMPLIANCES[0]).abs() < 1e-9 * TOP88_FIRST_COMPLIANCES[0],
        "{first}"
    );
    // The next iterations follow top88 closely; the small differences come
    // from top88 meeting the volume constraint only to a relative tolerance
    // of 1e-3 in the Lagrange multiplier, this code meets it exactly.
    for (iteration, &expected) in TOP88_FIRST_COMPLIANCES.iter().enumerate().skip(1) {
        let got = result.compliance_history[iteration];
        assert!(
            (got - expected).abs() < 0.003 * expected,
            "iteration {}: {got} vs {expected}",
            iteration + 1
        );
    }
    assert!(
        result.converged,
        "stopped after {} iterations",
        result.iterations
    );
    assert!(
        (90..=100).contains(&result.iterations),
        "top88 needs 94 iterations, got {}",
        result.iterations
    );
    assert!(
        (result.compliance - TOP88_FINAL_COMPLIANCE).abs() < 0.01 * TOP88_FINAL_COMPLIANCE,
        "final compliance {} vs top88 {TOP88_FINAL_COMPLIANCE}",
        result.compliance
    );
    let last = result.compliance_history[result.compliance_history.len() - 1];
    assert!((last - TOP88_FINAL_COMPLIANCE).abs() < 0.005 * TOP88_FINAL_COMPLIANCE);
    // Exact volume constraint (top88 only meets it to about 1e-3).
    assert!(
        (result.volume_fraction - 0.5).abs() < 1e-9,
        "{}",
        result.volume_fraction
    );
    assert!(result.density.iter().all(|&rho| (0.0..=1.0).contains(&rho)));
    // The design itself agrees with top88 element by element (the largest
    // difference over the whole field is below 1e-3 for the exact-volume
    // variant of the update) and block by block.
    let at = |row: usize, col: usize| result.density[row * 60 + col];
    for &(row, col, expected) in &TOP88_FINAL_DENSITY_SAMPLES {
        assert!(
            (at(row, col) - expected).abs() < 5e-3,
            "element ({row}, {col}): {} vs top88 {expected}",
            at(row, col)
        );
    }
    for (block_row, expected_row) in TOP88_FINAL_BLOCK_MEANS.iter().enumerate() {
        for (block_col, &expected) in expected_row.iter().enumerate() {
            let mut total = 0.0;
            for row in 10 * block_row..10 * (block_row + 1) {
                for col in 10 * block_col..10 * (block_col + 1) {
                    total += at(row, col);
                }
            }
            assert!(
                (total / 100.0 - expected).abs() < 5e-3,
                "block ({block_row}, {block_col}): {} vs top88 {expected}",
                total / 100.0
            );
        }
    }
}

#[test]
fn the_first_iteration_is_the_uniform_design_scaled_by_the_simp_modulus() {
    let (solver, loads, fixed) = mbb_problem(12, 6, 0.5, 1.5);
    let model = Q4Model::new(GridSize::new(12, 6), 0.3).unwrap();
    let unit = model
        .solve(&vec![1.0; 72], &loads, &fixed, LinearSolver::Direct)
        .unwrap();
    let result = solver.with_max_iter(1).run(&loads, &fixed).unwrap();
    // E(0.5) = 1e-9 + 0.5^3 (1 - 1e-9).
    let modulus = 1e-9 + 0.125 * (1.0 - 1e-9);
    let expected = unit.compliance / modulus;
    assert!(
        (result.compliance_history[0] - expected).abs() < 1e-9 * expected,
        "{} vs {expected}",
        result.compliance_history[0]
    );
    assert_eq!(result.iterations, 1);
}

#[test]
fn run_stores_a_consistent_result_in_the_solver() {
    let (solver, loads, fixed) = cantilever_problem(12, 6, 0.4, 1.5);
    let mut solver = solver.with_max_iter(25).with_tolerance(1e-3);
    let result = solver.run(&loads, &fixed).unwrap();

    assert_eq!(solver.simp.density, result.density);
    assert_eq!(solver.compliance_history, result.compliance_history);
    assert_eq!(result.iterations, result.compliance_history.len());
    assert!((result.volume_fraction - volume_fraction(&result.density)).abs() < 1e-15);
    assert!((result.volume_fraction - 0.4).abs() < 1e-9);
    // The reported compliance and displacements belong to the returned design.
    let model = Q4Model::new(solver.simp.grid, 0.3).unwrap();
    let modulus: Vec<f64> = result
        .density
        .iter()
        .map(|&rho| solver.simp.interpolated_stiffness(rho))
        .collect();
    let check = model
        .solve(&modulus, &loads, &fixed, LinearSolver::Direct)
        .unwrap();
    assert!((check.compliance - result.compliance).abs() < 1e-9 * check.compliance);
    for (a, b) in check.displacement.iter().zip(result.displacement.iter()) {
        assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()));
    }
    // Optimisation helps: the final design is much stiffer than the start.
    assert!(result.compliance < 0.5 * result.compliance_history[0]);
    // A converged run reports a change below the tolerance, otherwise not.
    assert_eq!(
        result.converged,
        result.max_density_change < solver.tolerance
    );
}

#[test]
fn run_stops_at_the_iteration_limit_and_reports_it() {
    let (solver, loads, fixed) = mbb_problem(12, 6, 0.5, 1.5);
    let mut solver = solver.with_max_iter(3).with_tolerance(1e-12);
    let result = solver.run(&loads, &fixed).unwrap();
    assert_eq!(result.iterations, 3);
    assert!(!result.converged);
    assert!(result.max_density_change > 1e-12);

    // No iteration at all: the initial design is analysed and returned.
    let (solver, loads, fixed) = mbb_problem(12, 6, 0.5, 1.5);
    let mut solver = solver.with_max_iter(0);
    let result = solver.run(&loads, &fixed).unwrap();
    assert_eq!(result.iterations, 0);
    assert!(!result.converged);
    assert_eq!(result.max_density_change, 0.0);
    assert_eq!(result.density, vec![0.5; 72]);
    assert!(result.compliance_history.is_empty());
    assert!(result.compliance > 0.0);
}

#[test]
fn run_preserves_the_mirror_symmetry_of_a_symmetric_cantilever() {
    let (nx, ny) = (16, 8);
    let (solver, loads, fixed) = cantilever_problem(nx, ny, 0.4, 1.5);
    let mut solver = solver.with_max_iter(40).with_tolerance(0.0);
    let result = solver.run(&loads, &fixed).unwrap();
    assert_eq!(result.iterations, 40);
    for row in 0..ny / 2 {
        for col in 0..nx {
            let below = result.density[row * nx + col];
            let above = result.density[(ny - 1 - row) * nx + col];
            assert!(
                (below - above).abs() < 1e-4,
                "row {row}, col {col}: {below} vs {above}"
            );
        }
    }
    // The design is not trivially uniform.
    let max = result.density.iter().copied().fold(0.0f64, f64::max);
    let min = result.density.iter().copied().fold(1.0f64, f64::min);
    assert!(max > 0.95 && min < 0.05, "[{min}, {max}]");
}

#[test]
fn the_design_does_not_depend_on_the_magnitude_of_the_load() {
    let (solver, loads, fixed) = mbb_problem(20, 8, 0.5, 1.5);
    let mut reference = solver.clone().with_max_iter(15).with_tolerance(0.0);
    let expected = reference.run(&loads, &fixed).unwrap();
    let heavy_loads: Vec<f64> = loads.iter().map(|f| 1000.0 * f).collect();
    let mut heavy = solver.with_max_iter(15).with_tolerance(0.0);
    let result = heavy.run(&heavy_loads, &fixed).unwrap();
    for (a, b) in result.density.iter().zip(expected.density.iter()) {
        assert!((a - b).abs() < 1e-7, "{a} vs {b}");
    }
    // Compliance scales with the square of the load.
    for (a, b) in result
        .compliance_history
        .iter()
        .zip(expected.compliance_history.iter())
    {
        assert!((a / b - 1.0e6).abs() < 1e-6 * 1.0e6, "{a} vs {b}");
    }
}

#[test]
fn run_with_conjugate_gradients_follows_the_direct_run() {
    let (solver, loads, fixed) = mbb_problem(10, 5, 0.5, 1.5);
    let mut direct = solver
        .clone()
        .with_max_iter(5)
        .with_tolerance(0.0)
        .with_linear_solver(LinearSolver::Direct);
    let mut iterative = solver
        .with_max_iter(5)
        .with_tolerance(0.0)
        .with_linear_solver(LinearSolver::Pcg {
            tolerance: 1e-10,
            max_iterations: 100_000,
        });
    let a = direct.run(&loads, &fixed).unwrap();
    let b = iterative.run(&loads, &fixed).unwrap();
    for (x, y) in a.compliance_history.iter().zip(b.compliance_history.iter()) {
        assert!((x - y).abs() < 1e-6 * x, "{x} vs {y}");
    }
    for (x, y) in a.density.iter().zip(b.density.iter()) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

fn invalid_field(error: &PhysicsError) -> &str {
    match error {
        PhysicsError::InvalidInput { field, .. } => field.as_str(),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn run_rejects_unusable_setups_and_leaves_the_solver_untouched() {
    let (solver, loads, fixed) = mbb_problem(8, 4, 0.5, 1.5);
    let untouched = solver.simp.density.clone();

    let mut failing = solver.clone();
    let error = failing.run(&vec![0.0; loads.len()], &fixed).unwrap_err();
    assert_eq!(invalid_field(&error), "loads");
    assert_eq!(failing.simp.density, untouched);
    assert!(failing.compliance_history.is_empty());

    // A load on a supported degree of freedom does no work either.
    let mut on_support = vec![0.0; loads.len()];
    on_support[fixed[0]] = 1.0;
    let error = solver.clone().run(&on_support, &fixed).unwrap_err();
    assert_eq!(invalid_field(&error), "loads");

    let error = solver.clone().run(&loads[..10], &fixed).unwrap_err();
    assert_eq!(invalid_field(&error), "loads");
    let error = solver.clone().run(&loads, &[]).unwrap_err();
    assert!(matches!(
        error,
        PhysicsError::FemBoundaryConditionError { .. }
    ));
    let error = solver.clone().run(&loads, &[1_000_000, 0, 1]).unwrap_err();
    assert!(matches!(error, PhysicsError::OutOfBounds { .. }));

    let broken = |tweak: &dyn Fn(&mut TopOptSolver)| -> PhysicsError {
        let mut copy = solver.clone();
        tweak(&mut copy);
        copy.run(&loads, &fixed).unwrap_err()
    };
    assert_eq!(
        invalid_field(&broken(&|s| s.simp.volume_fraction = 0.0)),
        "simp.volume_fraction"
    );
    assert_eq!(
        invalid_field(&broken(&|s| s.simp.volume_fraction = 1.5)),
        "simp.volume_fraction"
    );
    assert_eq!(
        invalid_field(&broken(&|s| s.simp.penalty = 0.5)),
        "simp.penalty"
    );
    assert_eq!(
        invalid_field(&broken(&|s| s.simp.penalty = f64::NAN)),
        "simp.penalty"
    );
    assert_eq!(
        invalid_field(&broken(&|s| s.simp.material.e_void = 2.0)),
        "simp.material"
    );
    assert_eq!(
        invalid_field(&broken(&|s| s.simp.material.e_void = -1.0)),
        "simp.material"
    );
    assert_eq!(
        invalid_field(&broken(&|s| s.move_limit = 0.0)),
        "move_limit"
    );
    assert_eq!(invalid_field(&broken(&|s| s.damping = f64::NAN)), "damping");
    assert_eq!(invalid_field(&broken(&|s| s.tolerance = -1.0)), "tolerance");
    assert_eq!(
        invalid_field(&broken(&|s| {
            s.simp.density.pop();
        })),
        "simp.density"
    );
    assert_eq!(
        invalid_field(&broken(
            &|s| s.filter = FilteringMethods::new(1.5, GridSize::new(4, 8))
        )),
        "filter"
    );
    assert!(matches!(
        broken(&|s| s.simp.material.nu = 0.6),
        PhysicsError::MaterialError { .. }
    ));
}

// ---------------------------------------------------------------------------
// Multi-load optimisation with finite-element analysis
// ---------------------------------------------------------------------------

fn two_load_cantilever(weights: (f64, f64)) -> (MultiLoadTopOpt, Vec<usize>, Q4Model) {
    let (nx, ny) = (12, 6);
    let grid = GridSize::new(nx, ny);
    let simp = SiMP::new(grid, 0.4, 3.0, 1.5, MaterialParams::default());
    let model = Q4Model::new(grid, simp.material.nu).unwrap();
    let fixed: Vec<usize> = (0..=ny)
        .flat_map(|row| {
            let node = model.node_index(row, 0);
            [model.dof(node, 0), model.dof(node, 1)]
        })
        .collect();
    let mut tip_down = vec![0.0; model.n_dofs()];
    tip_down[model.dof(model.node_index(0, nx), 1)] = -1.0;
    let mut corner_pull = vec![0.0; model.n_dofs()];
    corner_pull[model.dof(model.node_index(ny, nx), 0)] = 1.0;
    let optimiser = MultiLoadTopOpt::new(
        simp,
        FilteringMethods::new(1.5, grid),
        vec![
            LoadCase::new(tip_down, weights.0),
            LoadCase::new(corner_pull, weights.1),
        ],
    );
    (optimiser, fixed, model)
}

#[test]
fn multi_load_evaluation_with_the_fe_solver_is_the_weighted_sum_of_the_load_cases() {
    let (both, fixed, model) = two_load_cantilever((0.7, 0.3));
    let density: Vec<f64> = (0..72)
        .map(|e| 0.3 + 0.5 * ((e * 7 + 2) % 9) as f64 / 8.0)
        .collect();
    let simp = both.simp.clone();
    let (total, gradient) = both
        .evaluate(
            &density,
            model.simp_oracle(&simp, &fixed, LinearSolver::Direct),
        )
        .unwrap();

    let mut single_gradients = Vec::new();
    let mut single_compliance = Vec::new();
    for case in &both.load_cases {
        let mut alone = both.clone();
        alone.load_cases = vec![LoadCase::new(case.loads.clone(), 1.0)];
        let (c, g) = alone
            .evaluate(
                &density,
                model.simp_oracle(&simp, &fixed, LinearSolver::Direct),
            )
            .unwrap();
        single_compliance.push(c);
        single_gradients.push(g);
    }
    assert!(
        (total - (0.7 * single_compliance[0] + 0.3 * single_compliance[1])).abs() < 1e-9 * total
    );
    for (e, &entry) in gradient.iter().enumerate() {
        let expected = 0.7 * single_gradients[0][e] + 0.3 * single_gradients[1][e];
        assert!(
            (entry - expected).abs() < 1e-9 * expected.abs().max(1e-9),
            "element {e}"
        );
    }
    // The two load cases genuinely differ.
    assert!((single_compliance[0] - single_compliance[1]).abs() > 1e-3 * single_compliance[0]);
}

#[test]
fn run_fe_optimises_the_weighted_compliance_and_meets_the_volume() {
    let (mut optimiser, fixed, model) = two_load_cantilever((0.7, 0.3));
    let initial = optimiser.simp.density.clone();
    let density = optimiser.run_fe(30, 1e-6, &fixed).unwrap();

    let history = &optimiser.compliance_history;
    assert_eq!(history.len(), 30);
    assert!(
        history[history.len() - 1] < 0.6 * history[0],
        "{} -> {}",
        history[0],
        history[history.len() - 1]
    );
    assert!((volume_fraction(&density) - 0.4).abs() < 1e-6);
    assert_eq!(optimiser.simp.density, density);
    assert_ne!(density, initial);

    // `run_fe` is `run` with the closure of `Q4Model::simp_oracle`.
    let (mut manual, _, _) = two_load_cantilever((0.7, 0.3));
    let simp = manual.simp.clone();
    let oracle = model.simp_oracle(&simp, &fixed, LinearSolver::Auto);
    let by_hand = manual.run(30, 1e-6, oracle).unwrap();
    assert_eq!(by_hand, density);
}

#[test]
fn run_fe_reports_unusable_models_and_keeps_the_design() {
    let (mut optimiser, _, _) = two_load_cantilever((0.7, 0.3));
    let before = optimiser.simp.density.clone();
    let error = optimiser.run_fe(5, 1e-6, &[0, 1]).unwrap_err();
    assert!(matches!(
        error,
        PhysicsError::FemBoundaryConditionError { .. }
    ));
    assert_eq!(optimiser.simp.density, before);
    optimiser.simp.material.nu = 0.75;
    let error = optimiser.run_fe(5, 1e-6, &[0, 1, 2]).unwrap_err();
    assert!(matches!(error, PhysicsError::MaterialError { .. }));
}
