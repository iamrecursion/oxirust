// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the level-set optimiser: the upwind
//! Hamilton-Jacobi update, the CFL limit, the Sussman reinitialization, the
//! smoothed Heaviside, and the finite-element driven loop.

use super::*;
use crate::error::PhysicsError;
use crate::topology_optimization::{FilteringMethods, SiMP, TopOptSolver};

const N: usize = 40;

fn square_grid() -> GridSize {
    GridSize::new(N, N)
}

fn nodes(grid: GridSize) -> usize {
    (grid.nx + 1) * (grid.ny + 1)
}

/// `radius - distance` from the point `(cx, cy)` at every node.
fn circle_phi(grid: GridSize, radius: f64, cx: f64, cy: f64) -> Vec<f64> {
    let stride = grid.nx + 1;
    (0..nodes(grid))
        .map(|i| {
            let (row, col) = (i / stride, i % stride);
            radius - ((col as f64 - cx).powi(2) + (row as f64 - cy).powi(2)).sqrt()
        })
        .collect()
}

fn level_set(grid: GridSize, dt: f64) -> LevelSetTopOpt {
    LevelSetTopOpt::new(grid, 0.5, dt, MaterialParams::default())
}

/// Position along `row` of the first crossing from positive to non-positive
/// values to the right of `start`, linearly interpolated.
fn crossing(phi: &[f64], grid: GridSize, row: usize, start: usize) -> f64 {
    let stride = grid.nx + 1;
    for col in start..grid.nx {
        let (a, b) = (phi[row * stride + col], phi[row * stride + col + 1]);
        if a > 0.0 && b <= 0.0 {
            return col as f64 + a / (a - b);
        }
    }
    f64::NAN
}

/// Central-difference gradient magnitude at an interior node.
fn gradient_magnitude(phi: &[f64], grid: GridSize, row: usize, col: usize) -> f64 {
    let stride = grid.nx + 1;
    let dx = 0.5 * (phi[row * stride + col + 1] - phi[row * stride + col - 1]);
    let dy = 0.5 * (phi[(row + 1) * stride + col] - phi[(row - 1) * stride + col]);
    dx.hypot(dy)
}

// ---------------------------------------------------------------------------
// Smoothed Heaviside
// ---------------------------------------------------------------------------

#[test]
fn heaviside_is_continuous_monotone_and_spans_void_to_solid() {
    let eps = 1.0;
    assert_eq!(LevelSetTopOpt::heaviside(-1.5, eps), VOID_DENSITY);
    assert_eq!(LevelSetTopOpt::heaviside(1.5, eps), 1.0);
    // Continuous at both ends of the transition band ...
    assert!((LevelSetTopOpt::heaviside(-eps, eps) - VOID_DENSITY).abs() < 1e-12);
    assert!((LevelSetTopOpt::heaviside(eps, eps) - 1.0).abs() < 1e-12);
    // ... and half way at the front.
    let middle = VOID_DENSITY + 0.5 * (1.0 - VOID_DENSITY);
    assert!((LevelSetTopOpt::heaviside(0.0, eps) - middle).abs() < 1e-12);
    // Monotone without jumps; the slope matches the Dirac approximation.
    let samples: Vec<f64> = (0..=3000).map(|i| -1.5 + 0.001 * i as f64).collect();
    let values: Vec<f64> = samples
        .iter()
        .map(|&x| LevelSetTopOpt::heaviside(x, eps))
        .collect();
    for (i, pair) in values.windows(2).enumerate() {
        assert!(pair[1] >= pair[0] - 1e-15, "not monotone at {}", samples[i]);
        assert!(pair[1] - pair[0] < 2e-3, "jump at {}", samples[i]);
    }
    for &x in &[-0.7, -0.2, 0.0, 0.4, 0.9] {
        let h = 1e-6;
        let slope = (LevelSetTopOpt::heaviside(x + h, eps) - LevelSetTopOpt::heaviside(x - h, eps))
            / (2.0 * h);
        assert!(
            (slope - LevelSetTopOpt::dirac(x, eps)).abs() < 1e-6,
            "x = {x}"
        );
    }
}

#[test]
fn element_density_follows_the_sign_of_the_level_set() {
    let grid = GridSize::new(6, 4);
    let mut ls = level_set(grid, 0.1);
    ls.phi = vec![5.0; nodes(grid)];
    assert!(ls.element_density().iter().all(|&rho| rho == 1.0));
    ls.phi = vec![-5.0; nodes(grid)];
    assert!(ls.element_density().iter().all(|&rho| rho == VOID_DENSITY));
    // A vertical front at x = 3: solid on the right.
    let stride = grid.nx + 1;
    ls.phi = (0..nodes(grid))
        .map(|i| (i % stride) as f64 - 3.0)
        .collect();
    let density = ls.element_density();
    for (e, &rho) in density.iter().enumerate() {
        let col = e % grid.nx;
        match col {
            0..=1 => assert_eq!(rho, VOID_DENSITY, "col {col}"),
            4..=5 => assert_eq!(rho, 1.0, "col {col}"),
            _ => assert!(rho > VOID_DENSITY && rho < 1.0, "col {col}: {rho}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Hamilton-Jacobi update
// ---------------------------------------------------------------------------

#[test]
fn a_circular_front_moves_into_the_structure_at_the_requested_speed() {
    let grid = square_grid();
    let mut ls = level_set(grid, 0.2);
    ls.phi = circle_phi(grid, 12.0, 20.0, 20.0);
    let velocity = vec![1.0; nodes(grid)];
    for _ in 0..20 {
        let used = ls.hj_update(&velocity);
        assert_eq!(used, 0.2);
    }
    // The circle shrinks from radius 12 to 12 - 20 * 0.2 = 8.
    let radius = crossing(&ls.phi, grid, 20, 20) - 20.0;
    assert!((radius - 8.0).abs() < 0.05, "radius {radius}");
    // The field stays a smooth, nearly signed-distance function.
    let exact = circle_phi(grid, 8.0, 20.0, 20.0);
    let stride = grid.nx + 1;
    for row in 2..N - 1 {
        for col in 2..N - 1 {
            let distance = ((col as f64 - 20.0).powi(2) + (row as f64 - 20.0).powi(2)).sqrt();
            if distance < 2.0 || exact[row * stride + col].abs() > 3.0 {
                continue;
            }
            let error = (ls.phi[row * stride + col] - exact[row * stride + col]).abs();
            assert!(error < 0.25, "node ({row}, {col}): error {error}");
            let slope = gradient_magnitude(&ls.phi, grid, row, col);
            assert!(
                (0.9..=1.1).contains(&slope),
                "node ({row}, {col}): |grad| {slope}"
            );
        }
    }
    assert!(ls.phi.iter().all(|v| v.is_finite()));
}

#[test]
fn a_circular_front_grows_against_the_gradient_for_negative_speed() {
    let grid = square_grid();
    let mut ls = level_set(grid, 0.2);
    ls.phi = circle_phi(grid, 12.0, 20.0, 20.0);
    let velocity = vec![-1.0; nodes(grid)];
    for _ in 0..20 {
        ls.hj_update(&velocity);
    }
    let radius = crossing(&ls.phi, grid, 20, 20) - 20.0;
    assert!((radius - 16.0).abs() < 0.05, "radius {radius}");
    let exact = circle_phi(grid, 16.0, 20.0, 20.0);
    let stride = grid.nx + 1;
    for row in 2..N - 1 {
        for col in 2..N - 1 {
            let distance = ((col as f64 - 20.0).powi(2) + (row as f64 - 20.0).powi(2)).sqrt();
            if distance < 2.0 || exact[row * stride + col].abs() > 3.0 {
                continue;
            }
            let error = (ls.phi[row * stride + col] - exact[row * stride + col]).abs();
            assert!(error < 0.25, "node ({row}, {col}): error {error}");
            let slope = gradient_magnitude(&ls.phi, grid, row, col);
            assert!(
                (0.9..=1.1).contains(&slope),
                "node ({row}, {col}): |grad| {slope}"
            );
        }
    }
}

#[test]
fn the_update_is_upwind_a_valley_bottom_does_not_deepen() {
    // phi = |x - 10| - 3 on a 20 x 6 grid: material outside |x - 10| > 3. For
    // positive speed the front moves up the gradient, away from the valley
    // bottom, so the minimum must stay at -3. A forward-difference scheme
    // would deepen it by dt per step.
    let grid = GridSize::new(20, 6);
    let mut ls = level_set(grid, 0.2);
    let stride = grid.nx + 1;
    ls.phi = (0..nodes(grid))
        .map(|i| ((i % stride) as f64 - 10.0).abs() - 3.0)
        .collect();
    let velocity = vec![1.0; nodes(grid)];
    for step in 1..=8 {
        ls.hj_update(&velocity);
        let lowest = ls.phi.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(lowest >= -3.0 - 1e-12, "step {step}: minimum {lowest}");
        assert_eq!(
            ls.phi[2 * stride + 10],
            -3.0,
            "step {step}: valley bottom moved"
        );
    }
    // The flanks do move up the gradient: the valley widens.
    assert!(ls.phi[2 * stride + 11] < -2.0 - 0.2 + 1e-12);
    assert!(ls.phi[2 * stride + 14] < 1.0);
}

#[test]
fn the_update_is_upwind_a_ridge_top_does_not_grow_for_negative_speed() {
    // phi = 5 - |x - 10|: for negative speed the front moves down the
    // gradient, away from the ridge, so the maximum must stay at 5.
    let grid = GridSize::new(20, 6);
    let mut ls = level_set(grid, 0.2);
    let stride = grid.nx + 1;
    ls.phi = (0..nodes(grid))
        .map(|i| 5.0 - ((i % stride) as f64 - 10.0).abs())
        .collect();
    let velocity = vec![-1.0; nodes(grid)];
    for step in 1..=8 {
        ls.hj_update(&velocity);
        let highest = ls.phi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(highest <= 5.0 + 1e-12, "step {step}: maximum {highest}");
        assert_eq!(ls.phi[3 * stride + 10], 5.0, "step {step}: ridge moved");
    }
    assert!(
        ls.phi[3 * stride + 5] > 0.0,
        "the flank must have moved outwards"
    );
}

#[test]
fn a_linear_field_shifts_uniformly_including_the_boundary_nodes() {
    // For phi = x - 3 and V = 1 the exact solution is phi - t everywhere; the
    // one-sided boundary differences must reproduce it at the grid edge.
    let grid = GridSize::new(8, 5);
    let mut ls = level_set(grid, 0.25);
    let stride = grid.nx + 1;
    ls.phi = (0..nodes(grid))
        .map(|i| (i % stride) as f64 - 3.0)
        .collect();
    let before = ls.phi.clone();
    ls.hj_update(&vec![1.0; nodes(grid)]);
    for (a, b) in ls.phi.iter().zip(before.iter()) {
        assert!((a - (b - 0.25)).abs() < 1e-14, "{a} vs {}", b - 0.25);
    }
    // The same holds for a slanted field and negative speed.
    ls.phi = (0..nodes(grid))
        .map(|i| 0.3 * (i % stride) as f64 + 0.4 * (i / stride) as f64)
        .collect();
    let before = ls.phi.clone();
    ls.hj_update(&vec![-0.8; nodes(grid)]);
    let slope = 0.3f64.hypot(0.4);
    for (a, b) in ls.phi.iter().zip(before.iter()) {
        assert!((a - (b + 0.25 * 0.8 * slope)).abs() < 1e-13);
    }
}

#[test]
fn the_time_step_is_limited_by_the_cfl_condition() {
    let grid = square_grid();
    let mut limited = level_set(grid, 10.0);
    limited.phi = circle_phi(grid, 12.0, 20.0, 20.0);
    let mut reference = limited.clone();
    reference.dt = 0.5;
    let velocity = vec![1.0; nodes(grid)];

    let used = limited.hj_update(&velocity);
    assert_eq!(used, 0.5);
    assert_eq!(reference.hj_update(&velocity), 0.5);
    assert_eq!(limited.phi, reference.phi);
    // No overshoot: the front moved by at most half a cell.
    let moved = (crossing(&limited.phi, grid, 20, 20) - 20.0 - 12.0).abs();
    assert!(moved < 0.55, "front moved {moved}");

    // The limit scales with the speed; a short enough step is left alone.
    let fast = vec![4.0; nodes(grid)];
    assert_eq!(limited.hj_update(&fast), 0.125);
    limited.dt = 0.01;
    assert_eq!(limited.hj_update(&fast), 0.01);
    // Many steps at the limit stay bounded (an unlimited step of 10 explodes).
    let mut stable = level_set(grid, 10.0);
    stable.phi = circle_phi(grid, 12.0, 20.0, 20.0);
    for _ in 0..60 {
        stable.hj_update(&velocity);
    }
    assert!(stable.phi.iter().all(|v| v.is_finite() && v.abs() < 60.0));
    let radius = crossing(&stable.phi, grid, 20, 20) - 20.0;
    assert!(radius.is_nan() || radius < 0.5, "shrunk to {radius}");
}

#[test]
fn zero_and_missing_velocities_leave_the_level_set_alone() {
    let grid = GridSize::new(6, 6);
    let mut ls = level_set(grid, 0.2);
    let before = ls.phi.clone();
    assert_eq!(ls.hj_update(&vec![0.0; nodes(grid)]), 0.2);
    assert_eq!(ls.phi, before);
    // Entries beyond the end of the velocity slice count as zero.
    let mut partial = vec![0.0; nodes(grid) / 2];
    partial[0] = 1.0;
    ls.hj_update(&partial);
    let stride = grid.nx + 1;
    let half = nodes(grid) / 2;
    for (i, (&now, &then)) in ls.phi.iter().zip(before.iter()).enumerate().skip(half) {
        assert_eq!(now, then, "node {i} ({} {})", i / stride, i % stride);
    }
}

#[test]
fn speeds_beyond_the_last_node_do_not_shorten_the_time_step() {
    let grid = GridSize::new(6, 6);
    let mut ls = level_set(grid, 0.2);
    let mut velocity = vec![1.0; nodes(grid) + 3];
    velocity[nodes(grid) + 1] = 1.0e6;
    assert_eq!(ls.hj_update(&velocity), 0.2);
}

#[test]
fn a_level_set_of_the_wrong_length_is_left_untouched() {
    let grid = GridSize::new(6, 6);
    let mut ls = level_set(grid, 0.2);
    ls.phi.pop();
    let before = ls.phi.clone();
    assert_eq!(ls.hj_update(&vec![1.0; nodes(grid)]), 0.0);
    ls.reinitialize(5);
    assert_eq!(ls.phi, before);
}

// ---------------------------------------------------------------------------
// Reinitialization
// ---------------------------------------------------------------------------

#[test]
fn reinitialization_restores_a_unit_gradient_from_a_scaled_distance_function() {
    let grid = square_grid();
    let mut ls = level_set(grid, 0.2);
    let (cx, cy, radius) = (20.3, 19.6, 12.4);
    ls.phi = circle_phi(grid, radius, cx, cy)
        .into_iter()
        .map(|v| 2.0 * v)
        .collect();
    let initial_crossing = crossing(&ls.phi, grid, 20, 21);
    let stride = grid.nx + 1;
    assert!((gradient_magnitude(&ls.phi, grid, 20, 31) - 2.0).abs() < 0.2);

    ls.reinitialize(100);

    let exact = circle_phi(grid, radius, cx, cy);
    let mut checked = 0;
    for row in 2..N - 1 {
        for col in 2..N - 1 {
            let distance = ((col as f64 - cx).powi(2) + (row as f64 - cy).powi(2)).sqrt();
            if distance < 2.0 || ls.phi[row * stride + col].abs() > 5.0 {
                continue;
            }
            let slope = gradient_magnitude(&ls.phi, grid, row, col);
            assert!(
                (0.9..=1.1).contains(&slope),
                "node ({row}, {col}): |grad| {slope}"
            );
            assert!((ls.phi[row * stride + col] - exact[row * stride + col]).abs() < 0.7);
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} nodes checked");
    // The zero contour stays where it was, to a few hundredths of a cell.
    let after = crossing(&ls.phi, grid, 20, 21);
    assert!(
        (after - initial_crossing).abs() < 0.05,
        "{initial_crossing} -> {after}"
    );
    // The sign of phi is preserved.
    let initial = circle_phi(grid, radius, cx, cy);
    for (before, now) in initial.iter().zip(ls.phi.iter()) {
        if before.abs() > 0.6 {
            assert_eq!(before.signum(), now.signum());
        }
    }
}

#[test]
fn reinitialization_leaves_a_signed_distance_function_nearly_unchanged() {
    let grid = square_grid();
    let mut ls = level_set(grid, 0.2);
    ls.phi = circle_phi(grid, 12.4, 20.3, 19.6);
    let before = ls.phi.clone();
    ls.reinitialize(30);
    let stride = grid.nx + 1;
    for row in 2..N - 1 {
        for col in 2..N - 1 {
            let distance = ((col as f64 - 20.3).powi(2) + (row as f64 - 19.6).powi(2)).sqrt();
            if distance < 2.0 || before[row * stride + col].abs() > 4.0 {
                continue;
            }
            let drift = (ls.phi[row * stride + col] - before[row * stride + col]).abs();
            assert!(drift < 0.2, "node ({row}, {col}) drifted by {drift}");
        }
    }
    let shift = (crossing(&ls.phi, grid, 20, 21) - crossing(&before, grid, 20, 21)).abs();
    assert!(shift < 0.05, "front moved by {shift}");
}

#[test]
fn zero_reinitialization_steps_change_nothing() {
    let grid = GridSize::new(8, 8);
    let mut ls = level_set(grid, 0.2);
    let before = ls.phi.clone();
    ls.reinitialize(0);
    assert_eq!(ls.phi, before);
}

// ---------------------------------------------------------------------------
// Mock optimisation
// ---------------------------------------------------------------------------

#[test]
fn the_mock_run_stays_finite_and_records_its_history() {
    let grid = GridSize::new(10, 10);
    let mut ls = level_set(grid, 0.05);
    ls.run_mock(25);
    assert_eq!(ls.compliance_history.len(), 25);
    assert!(ls.phi.iter().all(|v| v.is_finite()));
    assert!(
        ls.compliance_history
            .iter()
            .all(|c| c.is_finite() && *c > 0.0)
    );
}

// ---------------------------------------------------------------------------
// Finite-element driven optimisation
// ---------------------------------------------------------------------------

/// `distance to the nearest hole centre - radius` for holes on a square
/// lattice of the given spacing (positive outside the holes).
fn hole_array(grid: GridSize, spacing: f64, radius: f64) -> Vec<f64> {
    let stride = grid.nx + 1;
    (0..nodes(grid))
        .map(|i| {
            let (row, col) = ((i / stride) as f64, (i % stride) as f64);
            let cx = ((col - spacing / 2.0) / spacing).round() * spacing + spacing / 2.0;
            let cy = ((row - spacing / 2.0) / spacing).round() * spacing + spacing / 2.0;
            ((col - cx).powi(2) + (row - cy).powi(2)).sqrt() - radius
        })
        .collect()
}

/// Half MBB beam: load at the top-left corner, symmetry on the left edge,
/// roller at the bottom-right corner.
fn mbb(grid: GridSize) -> (Vec<f64>, Vec<usize>) {
    let model = Q4Model::new(grid, 0.3).unwrap();
    let mut loads = vec![0.0; model.n_dofs()];
    loads[model.dof(model.node_index(grid.ny, 0), 1)] = -1.0;
    let mut fixed: Vec<usize> = (0..=grid.ny)
        .map(|row| model.dof(model.node_index(row, 0), 0))
        .collect();
    fixed.push(model.dof(model.node_index(0, grid.nx), 1));
    (loads, fixed)
}

#[test]
fn the_fe_run_reaches_the_volume_target_and_competes_with_simp() {
    let grid = GridSize::new(30, 10);
    let (loads, fixed) = mbb(grid);
    let mut ls = LevelSetTopOpt::new(grid, 0.5, 0.5, MaterialParams::default());
    ls.phi = hole_array(grid, 5.0, 1.2);
    let result = ls.run(&loads, &fixed, 80).unwrap();

    assert_eq!(result.iterations, 80);
    assert_eq!(result.compliance_history.len(), 80);
    assert_eq!(result.volume_history.len(), 80);
    assert_eq!(ls.compliance_history, result.compliance_history);
    assert!(
        result
            .compliance_history
            .iter()
            .all(|c| c.is_finite() && *c > 0.0)
    );
    // Shrinking from the initial 80 % solid takes a few CFL-limited steps;
    // afterwards every Hamilton-Jacobi step restores the target volume, also
    // right after a reinitialization.
    for (iteration, &volume) in result.volume_history.iter().enumerate().skip(5) {
        assert!(
            (volume - 0.5).abs() < 5e-3,
            "iteration {iteration}: volume {volume}"
        );
    }
    assert!(
        (result.volume_fraction - 0.5).abs() < 5e-3,
        "final volume {}",
        result.volume_fraction
    );

    // Two benchmarks of the same 50 % problem. A uniform gray design (E = 0.5
    // everywhere) is twice as compliant as the solid structure; the SIMP
    // optimum of `TopOptSolver::run` (top88: 225.5 for this beam) is what a
    // good layout reaches.
    let model = Q4Model::new(grid, 0.3).unwrap();
    let solid = model
        .solve(
            &vec![1.0; grid.n_elem()],
            &loads,
            &fixed,
            LinearSolver::Direct,
        )
        .unwrap();
    let uniform_gray = solid.compliance / 0.5;
    let simp = SiMP::new(grid, 0.5, 3.0, 1.5, MaterialParams::default());
    let mut simp_solver = TopOptSolver::new(simp, FilteringMethods::new(1.5, grid))
        .with_tolerance(0.01)
        .with_max_iter(300);
    let simp_optimum = simp_solver.run(&loads, &fixed).unwrap();
    let gray = result
        .density
        .iter()
        .filter(|&&rho| rho > 0.1 && rho < 0.9)
        .count();
    assert!(
        result.compliance < 0.95 * uniform_gray,
        "compliance {} vs uniform gray {uniform_gray}",
        result.compliance
    );
    assert!(
        result.compliance < 1.1 * simp_optimum.compliance,
        "compliance {} vs SIMP optimum {}",
        result.compliance,
        simp_optimum.compliance
    );
    // After the volume has settled the compliance neither diverges nor
    // leaves the neighbourhood of the SIMP optimum.
    let worst = result.compliance_history[10..]
        .iter()
        .copied()
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1.25 * simp_optimum.compliance,
        "worst compliance {worst} vs SIMP optimum {}",
        simp_optimum.compliance
    );
    // The reported compliance belongs to the returned density.
    let modulus: Vec<f64> = result
        .density
        .iter()
        .map(|&rho| 1e-9 + rho * (1.0 - 1e-9))
        .collect();
    let check = model
        .solve(&modulus, &loads, &fixed, LinearSolver::Direct)
        .unwrap();
    assert!((check.compliance - result.compliance).abs() < 1e-9 * check.compliance);
    assert_eq!(result.density, ls.element_density());
    // The design is mostly solid or void rather than gray.
    assert!(gray * 2 < grid.n_elem(), "{gray} gray elements");
}

#[test]
fn the_fe_run_follows_the_configured_reinitialization_and_solver() {
    let grid = GridSize::new(16, 8);
    let (loads, fixed) = mbb(grid);
    let make = || {
        let mut ls = LevelSetTopOpt::new(grid, 0.5, 0.4, MaterialParams::default());
        ls.phi = hole_array(grid, 4.0, 1.0);
        ls
    };
    let mut never = make().with_reinitialization(0, 10);
    assert_eq!((never.reinit_interval, never.reinit_steps), (0, 10));
    let a = never.run(&loads, &fixed, 12).unwrap();
    let mut often = make().with_reinitialization(2, 5);
    let b = often.run(&loads, &fixed, 12).unwrap();
    assert_ne!(a.density, b.density, "reinitialization must matter");
    assert!(a.compliance.is_finite() && b.compliance.is_finite());

    // The conjugate-gradient solver gives the same optimisation.
    let mut iterative = make().with_linear_solver(LinearSolver::Pcg {
        tolerance: 1e-10,
        max_iterations: 100_000,
    });
    let mut direct = make().with_linear_solver(LinearSolver::Direct);
    let x = direct.run(&loads, &fixed, 6).unwrap();
    let y = iterative.run(&loads, &fixed, 6).unwrap();
    for (p, q) in x.compliance_history.iter().zip(y.compliance_history.iter()) {
        assert!((p - q).abs() < 1e-6 * p, "{p} vs {q}");
    }
}

#[test]
fn the_fe_run_without_iterations_analyses_the_initial_level_set() {
    let grid = GridSize::new(12, 6);
    let (loads, fixed) = mbb(grid);
    let mut ls = LevelSetTopOpt::new(grid, 0.5, 0.4, MaterialParams::default());
    let before = ls.phi.clone();
    let result = ls.run(&loads, &fixed, 0).unwrap();
    assert_eq!(result.iterations, 0);
    assert!(result.compliance_history.is_empty());
    assert_eq!(ls.phi, before);
    assert!(result.compliance > 0.0);
    assert_eq!(result.density, ls.element_density());
}

#[test]
fn failed_fe_runs_restore_the_level_set_and_the_history() {
    let grid = GridSize::new(12, 6);
    let (loads, fixed) = mbb(grid);
    let mut ls = LevelSetTopOpt::new(grid, 0.5, 0.4, MaterialParams::default());
    ls.phi = hole_array(grid, 4.0, 1.0);
    ls.compliance_history = vec![1.0, 2.0];
    let before = ls.phi.clone();

    let invalid = |error: &PhysicsError| match error {
        PhysicsError::InvalidInput { field, .. } => field.clone(),
        other => panic!("expected InvalidInput, got {other:?}"),
    };
    let error = ls.run(&vec![0.0; loads.len()], &fixed, 4).unwrap_err();
    assert_eq!(invalid(&error), "loads");
    assert_eq!(ls.phi, before);
    assert_eq!(ls.compliance_history, vec![1.0, 2.0]);

    // Too few supports are rejected by the first finite-element solve.
    let error = ls.run(&loads, &fixed[..2], 4).unwrap_err();
    assert!(matches!(
        error,
        PhysicsError::FemBoundaryConditionError { .. }
    ));
    assert_eq!(ls.phi, before);

    let mut bad = ls.clone();
    bad.volume_fraction = 1.0;
    assert_eq!(
        invalid(&bad.run(&loads, &fixed, 2).unwrap_err()),
        "volume_fraction"
    );
    let mut bad = ls.clone();
    bad.volume_fraction = 0.0;
    assert_eq!(
        invalid(&bad.run(&loads, &fixed, 2).unwrap_err()),
        "volume_fraction"
    );
    let mut bad = ls.clone();
    bad.dt = 0.0;
    assert_eq!(invalid(&bad.run(&loads, &fixed, 2).unwrap_err()), "dt");
    let mut bad = ls.clone();
    bad.material.e_void = 5.0;
    assert_eq!(
        invalid(&bad.run(&loads, &fixed, 2).unwrap_err()),
        "material"
    );
    let mut bad = ls.clone();
    bad.phi.pop();
    assert_eq!(invalid(&bad.run(&loads, &fixed, 2).unwrap_err()), "phi");
    let mut bad = ls.clone();
    bad.phi[3] = f64::NAN;
    assert_eq!(invalid(&bad.run(&loads, &fixed, 2).unwrap_err()), "phi");
    let mut bad = ls.clone();
    bad.material.nu = 0.9;
    assert!(matches!(
        bad.run(&loads, &fixed, 2).unwrap_err(),
        PhysicsError::MaterialError { .. }
    ));
}
