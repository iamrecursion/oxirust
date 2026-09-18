// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests of the two-way particle-fluid feedback.
//!
//! The fluid must receive the reaction of the *hydrodynamic* force on a particle
//! (`-hydro_force`), never the reaction of its weight: a settling particle
//! pushes the fluid down with exactly the drag it feels, which at terminal
//! velocity equals its (buoyancy-reduced) weight.

use super::*;

/// Sum of a flat array of 2-D vectors.
fn total(field: &[[f64; 2]]) -> [f64; 2] {
    field
        .iter()
        .fold([0.0, 0.0], |acc, v| [acc[0] + v[0], acc[1] + v[1]])
}

/// Deterministic pseudo-random numbers in `[0, 1)` (no external dependency).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn update_particle_records_the_hydrodynamic_part_separately() {
    let coupling = TwoWayCoupling::new(8, 8, 1.0);
    let mut p = CoupledParticle::new([3.0, 3.0], [0.0, 0.0], 2.0, 0.4, 7);
    assert_eq!(p.hydro_force, [0.0, 0.0]);
    coupling.update_particle(&mut p, [1.0, 2.0], [0.0, -3.0], 0.1);
    assert_eq!(p.force, [1.0, -1.0], "net force = drag + gravity");
    assert_eq!(p.hydro_force, [1.0, 2.0], "hydrodynamic force = drag only");
}

#[test]
fn particle_at_terminal_velocity_pushes_the_fluid_with_its_drag() {
    // A heavy sphere sedimenting in quiescent fluid: Stokes drag balances the
    // weight, so the net force on the particle vanishes but the fluid must
    // still receive the (non-zero) drag reaction.
    let (nx, ny) = (8_usize, 8_usize);
    let coupling = TwoWayCoupling::new(nx, ny, 1.0);
    let mu = 0.5;
    let mass = 1.0;
    let weight = 0.01;
    let gravity = [0.0, -weight];
    let fluid = FluidCell::new(1.0, 0.0, 0.0);
    let mut p = CoupledParticle::new([3.0, 4.0], [0.0, 0.0], mass, 0.5, 0);
    let dt = 0.01;
    for _ in 0..2_000 {
        let drag = DragForce::stokes_drag(&p, &fluid, mu);
        coupling.update_particle(&mut p, drag, gravity, dt);
    }
    let terminal = -weight / (6.0 * PI * mu * p.radius);
    assert!(
        (p.velocity[1] - terminal).abs() < 1e-9 * terminal.abs(),
        "terminal velocity {} vs {terminal}",
        p.velocity[1]
    );
    // Net force on the particle is (numerically) zero ...
    assert!(
        p.force[0].abs() < 1e-12 && p.force[1].abs() < 1e-12,
        "{:?}",
        p.force
    );
    // ... while the drag it exerts on the fluid is -weight (downwards), not zero.
    let grid = total(&coupling.spread_particle_force(&[p.clone()], nx, ny));
    assert!(grid[0].abs() < 1e-12, "fx = {}", grid[0]);
    assert!(
        (grid[1] - (-weight)).abs() < 1e-9 * weight,
        "the fluid must receive -drag = -{weight}, got {}",
        grid[1]
    );
    // The same holds for the momentum source density (cell area 1).
    let source = total(&fluid_momentum_source(&[p], nx, ny, 1.0));
    assert!(
        (source[1] - (-weight)).abs() < 1e-9 * weight,
        "sy = {}",
        source[1]
    );
}

#[test]
fn free_fall_exerts_no_force_on_the_fluid() {
    // Fluid and particle both at rest: the only force is gravity, which the
    // particle carries alone (it used to be reflected into the fluid).
    let coupling = TwoWayCoupling::new(6, 6, 1.0);
    let mut p = CoupledParticle::new([2.0, 2.0], [0.0, 0.0], 3.0, 0.3, 0);
    coupling.update_particle(&mut p, [0.0, 0.0], [0.0, -3.0 * 9.81], 0.01);
    assert!(p.force[1] < 0.0);
    let grid = coupling.spread_particle_force(&[p.clone()], 6, 6);
    assert!(grid.iter().all(|f| f[0] == 0.0 && f[1] == 0.0));
    let source = fluid_momentum_source(&[p], 6, 6, 1.0);
    assert!(source.iter().all(|f| f[0] == 0.0 && f[1] == 0.0));
}

#[test]
fn newton_third_law_holds_over_a_whole_trajectory() {
    // Impulse bookkeeping: the particle gains m v = ∫ (F_hydro + F_gravity) dt;
    // the fluid gains exactly -∫ F_hydro dt, so that the coupling alone conserves
    // the momentum of the particle-fluid system.
    let (nx, ny) = (10_usize, 10_usize);
    let coupling = TwoWayCoupling::new(nx, ny, 1.0);
    let mass = 2.0;
    let gravity = [0.0, -0.05];
    let fluid = FluidCell::new(1.0, 0.02, 0.0);
    let dt = 0.05;
    let mut p = CoupledParticle::new([4.5, 6.25], [0.0, 0.0], mass, 0.4, 0);
    let mut fluid_impulse = [0.0_f64; 2];
    let mut hydro_impulse = [0.0_f64; 2];
    for _ in 0..400 {
        let drag = DragForce::stokes_drag(&p, &fluid, 0.3);
        coupling.update_particle(&mut p, drag, gravity, dt);
        let grid = total(&coupling.spread_particle_force(std::slice::from_ref(&p), nx, ny));
        for k in 0..2 {
            fluid_impulse[k] += grid[k] * dt;
            hydro_impulse[k] += p.hydro_force[k] * dt;
        }
    }
    for k in 0..2 {
        assert!(
            (fluid_impulse[k] + hydro_impulse[k]).abs() < 1e-12,
            "component {k}: fluid {} vs particle {}",
            fluid_impulse[k],
            hydro_impulse[k]
        );
        let gravity_impulse = gravity[k] * dt * 400.0;
        let momentum = mass * p.velocity[k];
        assert!(
            (momentum - (hydro_impulse[k] + gravity_impulse)).abs() < 1e-9,
            "component {k}: m v = {momentum}"
        );
    }
}

#[test]
fn reaction_is_shared_among_the_four_surrounding_nodes() {
    let (nx, ny) = (6_usize, 6_usize);
    let coupling = TwoWayCoupling::new(nx, ny, 1.0);
    let mut p = CoupledParticle::new([2.25, 3.5], [0.0, 0.0], 1.0, 0.2, 0);
    p.hydro_force = [4.0, -8.0];
    let grid = coupling.spread_particle_force(&[p], nx, ny);
    // tx = 0.25, ty = 0.5
    let expected = [
        ((2, 3), 0.375),
        ((3, 3), 0.125),
        ((2, 4), 0.375),
        ((3, 4), 0.125),
    ];
    for ((ix, iy), w) in expected {
        let f = grid[iy * nx + ix];
        assert!(
            (f[0] - (-w * 4.0)).abs() < 1e-14,
            "fx at ({ix},{iy}) = {}",
            f[0]
        );
        assert!(
            (f[1] - (w * 8.0)).abs() < 1e-14,
            "fy at ({ix},{iy}) = {}",
            f[1]
        );
    }
    let touched = grid.iter().filter(|f| f[0] != 0.0 || f[1] != 0.0).count();
    assert_eq!(touched, 4);
}

#[test]
fn momentum_source_density_divides_by_the_cell_area() {
    let (nx, ny) = (5_usize, 5_usize);
    let dx = 2.0;
    let mut p = CoupledParticle::new([4.0, 6.0], [0.0, 0.0], 1.0, 0.2, 0);
    p.hydro_force = [2.0, -6.0];
    let source = fluid_momentum_source(&[p], nx, ny, dx);
    // Position (4, 6) is the node (2, 3) of a grid of spacing 2.
    let k = 3 * nx + 2;
    assert!((source[k][0] - (-2.0 / 4.0)).abs() < 1e-14);
    assert!((source[k][1] - (6.0 / 4.0)).abs() < 1e-14);
    let sum = total(&source);
    assert!((sum[0] * dx * dx - (-2.0)).abs() < 1e-13);
    assert!((sum[1] * dx * dx - 6.0).abs() < 1e-13);
}

#[test]
fn spreading_is_the_adjoint_of_interpolation() {
    // <interp(u, x), F> = - <u, spread(F at x)> for every position, including
    // positions outside the domain that are clamped.
    let (nx, ny) = (7_usize, 5_usize);
    let dx = 0.5;
    let coupling = TwoWayCoupling::new(nx, ny, dx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let field: Vec<(f64, f64)> = (0..nx * ny)
        .map(|_| (lcg(&mut state) - 0.5, lcg(&mut state) - 0.5))
        .collect();
    let positions = [
        [1.37, 0.81],
        [0.0, 0.0],
        [3.0, 2.0],
        [2.9999, 1.0001],
        [-4.0, 1.1],
        [100.0, -3.0],
        [1.75, 100.0],
    ];
    for pos in positions {
        let mut p = CoupledParticle::new(pos, [0.0, 0.0], 1.0, 0.1, 0);
        p.hydro_force = [lcg(&mut state) - 0.5, lcg(&mut state) - 0.5];
        let u = coupling.interpolate_fluid_velocity(&field, pos);
        let lhs = u[0] * p.hydro_force[0] + u[1] * p.hydro_force[1];
        let grid = coupling.spread_particle_force(&[p], nx, ny);
        let rhs: f64 = grid
            .iter()
            .zip(field.iter())
            .map(|(g, f)| g[0] * f.0 + g[1] * f.1)
            .sum();
        assert!(
            (lhs + rhs).abs() < 1e-14,
            "adjoint mismatch at {pos:?}: {lhs} vs {}",
            -rhs
        );
    }
}

#[test]
fn positions_outside_the_domain_deposit_on_the_nearest_boundary_node() {
    let (nx, ny) = (4_usize, 3_usize);
    let coupling = TwoWayCoupling::new(nx, ny, 1.0);
    let mut p = CoupledParticle::new([-5.0, 99.0], [0.0, 0.0], 1.0, 0.1, 0);
    p.hydro_force = [1.0, 1.0];
    let grid = coupling.spread_particle_force(&[p], nx, ny);
    let corner = (ny - 1) * nx;
    assert_eq!(grid[corner], [-1.0, -1.0]);
    assert_eq!(total(&grid), [-1.0, -1.0]);
}

#[test]
fn single_node_directions_do_not_panic() {
    // Grids with one node along an axis used to panic in the interpolation
    // (clamp with min > max).
    for &(nx, ny) in &[(1_usize, 1_usize), (1, 5), (5, 1)] {
        let coupling = TwoWayCoupling::new(nx, ny, 1.0);
        let field: Vec<(f64, f64)> = (0..nx * ny).map(|k| (k as f64, -(k as f64))).collect();
        let u = coupling.interpolate_fluid_velocity(&field, [0.4, 0.0]);
        assert!(u[0].is_finite() && u[1].is_finite());
        if nx * ny == 1 {
            assert_eq!(u, [0.0, 0.0]);
        }
        let mut p = CoupledParticle::new([0.4, 2.2], [0.0, 0.0], 1.0, 0.1, 0);
        p.hydro_force = [3.0, -1.0];
        let grid = coupling.spread_particle_force(&[p], nx, ny);
        let sum = total(&grid);
        assert!((sum[0] + 3.0).abs() < 1e-14 && (sum[1] - 1.0).abs() < 1e-14);
    }
}

#[test]
fn empty_grids_neither_panic_nor_deposit() {
    let coupling = TwoWayCoupling::new(0, 0, 1.0);
    assert_eq!(
        coupling.interpolate_fluid_velocity(&[], [1.0, 1.0]),
        [0.0, 0.0]
    );
    let mut p = CoupledParticle::new([1.0, 1.0], [0.0, 0.0], 1.0, 0.1, 0);
    p.hydro_force = [1.0, 1.0];
    assert!(
        coupling
            .spread_particle_force(&[p.clone()], 0, 3)
            .is_empty()
    );
    assert!(fluid_momentum_source(&[p], 3, 0, 1.0).is_empty());
}
