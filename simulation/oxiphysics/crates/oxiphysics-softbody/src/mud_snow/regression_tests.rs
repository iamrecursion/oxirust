// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Behavioural tests of the MLS-MPM snow solver.
//!
//! `SnowMpm::step` used to be a gravity-only PIC integrator: it never read the
//! Young's modulus, Poisson's ratio, critical compression/stretch or hardening,
//! kept no deformation gradient, and its truncated 3-node stencil did not even
//! conserve mass.  A snow block placed on the floor fell through the domain.
//! The tests below check what the old integrator could not do: rest on the
//! floor, compact plastically and harden under impact, respond to the
//! material parameters, and conserve mass and momentum in the grid transfers.

use super::*;

/// Cells per axis of the test domain.
const CELLS: usize = 12;
/// Edge length of the test domain (m): cell size `h = 0.0625 m`.
const DOMAIN: f64 = 0.75;
const DENSITY: f64 = 400.0;
const TEMPERATURE: f64 = 263.15;
/// Explicit time step (s): the dilatational wave speed is about 20 m/s, so the
/// CFL number `c dt / h` stays near 0.16 (and below 0.3 when hardened).
const DT: f64 = 5.0e-4;

/// A 3 × 3 × 3 cell block (6 × 6 × 6 particles) of snow, `lift_cells` cells
/// above the floor, moving with the vertical velocity `velocity_y`.
///
/// The sticky node layers fill `y <= (thickness - 1) h`, so snow rests on the
/// plane of their innermost row: with `lift_cells = 0` the block sits directly
/// on the floor.
fn snow_block(lift_cells: f64, velocity_y: f64) -> SnowMpm {
    let mut sim = SnowMpm::new(CELLS, DOMAIN);
    let h = sim.cell_size;
    let floor = (sim.boundary_thickness as f64 - 1.0) * h;
    let lo = [4.0 * h, floor + lift_cells * h, 4.0 * h];
    let hi = [7.0 * h, floor + (3.0 + lift_cells) * h, 7.0 * h];
    let added = sim.fill_box(lo, hi, DENSITY, TEMPERATURE);
    assert_eq!(added, 216);
    for p in sim.particles.iter_mut() {
        p.velocity = [0.0, velocity_y, 0.0];
    }
    sim
}

fn run(sim: &mut SnowMpm, duration: f64) {
    let steps = (duration / DT).round() as usize;
    for _ in 0..steps {
        sim.step(DT);
    }
}

fn height(sim: &SnowMpm) -> f64 {
    let ys = sim.particles.iter().map(|p| p.position[1]);
    ys.clone().fold(f64::MIN, f64::max) - ys.fold(f64::MAX, f64::min)
}

fn min_y(sim: &SnowMpm) -> f64 {
    sim.particles
        .iter()
        .map(|p| p.position[1])
        .fold(f64::MAX, f64::min)
}

fn max_speed(sim: &SnowMpm) -> f64 {
    sim.particles
        .iter()
        .map(|p| len3(p.velocity))
        .fold(0.0, f64::max)
}

fn max_hardness(sim: &SnowMpm) -> f64 {
    sim.particles.iter().map(|p| p.hardness).fold(0.0, f64::max)
}

fn min_det_f(sim: &SnowMpm) -> f64 {
    sim.particles
        .iter()
        .map(|p| mat3_det(mat3_from_row_major(p.def_gradient)))
        .fold(f64::MAX, f64::min)
}

fn total_mass(sim: &SnowMpm) -> f64 {
    sim.particles.iter().map(|p| p.mass).sum()
}

fn all_finite(sim: &SnowMpm) -> bool {
    sim.particles.iter().all(|p| {
        p.position
            .iter()
            .chain(p.velocity.iter())
            .chain(p.def_gradient.iter())
            .all(|c| c.is_finite())
    })
}

// ---------------------------------------------------------------------------
// Particle bookkeeping
// ---------------------------------------------------------------------------

/// `fill_box` places 2 × 2 × 2 particles per cell that together represent the
/// volume and the mass of the box.
#[test]
fn fill_box_places_a_two_per_cell_lattice() {
    let mut sim = SnowMpm::new(CELLS, DOMAIN);
    let h = sim.cell_size;
    let (lo, hi) = ([3.0 * h, 3.0 * h, 3.0 * h], [5.0 * h, 6.0 * h, 4.0 * h]);
    let added = sim.fill_box(lo, hi, DENSITY, TEMPERATURE);
    assert_eq!(added, 4 * 6 * 2);
    assert_eq!(sim.particles.len(), added);

    let box_volume = 2.0 * 3.0 * 1.0 * h * h * h;
    let volume: f64 = sim.particles.iter().map(|p| p.volume).sum();
    assert!((volume - box_volume).abs() < 1e-12 * box_volume, "{volume}");
    assert!((total_mass(&sim) - DENSITY * box_volume).abs() < 1e-9);
    for p in &sim.particles {
        assert!((p.volume - h * h * h / 8.0).abs() < 1e-15);
        for d in 0..3 {
            assert!(p.position[d] > lo[d] && p.position[d] < hi[d], "{p:?}");
        }
    }
    // The MPM state is index-aligned with the particles.
    assert_eq!(sim.mpm_particles.len(), added);
}

/// A particle pushed straight into `particles` is adopted at the next step with
/// the grid-consistent volume and mass; a pre-deformed particle starts from its
/// deformation as an elastic state.
#[test]
fn particles_pushed_directly_are_adopted() {
    let mut sim = SnowMpm::new(CELLS, DOMAIN);
    let mut deformed = SnowParticle::new([0.375, 0.3, 0.375], DENSITY, TEMPERATURE);
    deformed.def_gradient = [1.005, 0.0, 0.0, 0.0, 0.995, 0.0, 0.0, 0.0, 1.0];
    sim.particles.push(deformed);
    sim.step(DT);
    assert_eq!(sim.mpm_particles.len(), 1);
    let p = &sim.particles[0];
    assert!((p.volume - sim.particle_volume()).abs() < 1e-18);
    assert!((p.mass - DENSITY * sim.particle_volume()).abs() < 1e-12);
    // The deformation was elastic: nothing was moved into the plastic part.
    let fp = sim.mpm_particles[0].fp;
    assert!(
        (mat3_det(fp) - 1.0).abs() < 1e-6,
        "det Fp = {}",
        mat3_det(fp)
    );
    assert!(p.def_gradient[0] > 1.004, "F00 = {}", p.def_gradient[0]);
}

/// The Lamé parameters follow from Young's modulus and Poisson's ratio and are
/// re-read at every step; the hardening factor of a particle starts at one.
#[test]
fn solver_reads_the_material_parameters() {
    let mut sim = SnowMpm::new(CELLS, DOMAIN);
    sim.young_modulus = 2.0e5;
    sim.poisson_ratio = 0.25;
    let (mu, lambda) = sim.lame_parameters();
    assert!((mu - 2.0e5 / 2.5).abs() < 1e-6, "mu = {mu}");
    assert!(
        (lambda - 2.0e5 * 0.25 / (1.25 * 0.5)).abs() < 1e-6,
        "lambda = {lambda}"
    );

    sim.add_particle(SnowParticle::new([0.375, 0.3, 0.375], DENSITY, TEMPERATURE));
    assert!((sim.mpm_particles[0].mu0 - mu).abs() < 1e-9);
    assert!((sim.mpm_particles[0].lambda0 - lambda).abs() < 1e-9);
    assert_eq!(sim.particles[0].hardness, 1.0);

    // Changing the parameters between steps reaches the particles.
    sim.young_modulus = 4.0e5;
    sim.hardening = 7.0;
    sim.step(DT);
    let (mu, lambda) = sim.lame_parameters();
    assert!((sim.mpm_particles[0].mu0 - mu).abs() < 1e-9);
    assert!((sim.mpm_particles[0].lambda0 - lambda).abs() < 1e-9);
    assert_eq!(sim.mpm_particles[0].hardening, 7.0);
}

#[test]
fn invalid_time_steps_leave_the_state_unchanged() {
    let mut sim = snow_block(1.0, -1.0);
    let before: Vec<([f64; 3], [f64; 3])> = sim
        .particles
        .iter()
        .map(|p| (p.position, p.velocity))
        .collect();
    for dt in [0.0, -DT, f64::NAN, f64::INFINITY] {
        sim.step(dt);
    }
    let after: Vec<([f64; 3], [f64; 3])> = sim
        .particles
        .iter()
        .map(|p| (p.position, p.velocity))
        .collect();
    assert_eq!(before, after);
}

// ---------------------------------------------------------------------------
// Grid transfers
// ---------------------------------------------------------------------------

/// P2G followed by G2P conserves mass and linear momentum and reproduces a
/// uniform velocity field exactly (the quadratic B-spline weights of the three
/// stencil nodes per axis sum to one).
#[test]
fn grid_transfers_conserve_mass_and_momentum() {
    let mut sim = SnowMpm::new(CELLS, DOMAIN);
    let h = sim.cell_size;
    // A few particles at arbitrary sub-cell offsets well inside the domain.
    for (i, offset) in [0.13, 0.5, 0.87, 0.31].iter().enumerate() {
        let position = [
            (4.0 + offset) * h,
            (5.0 + 0.37 * i as f64) * h,
            (6.0 + 1.0 - offset) * h,
        ];
        sim.add_particle(SnowParticle::new(position, DENSITY, TEMPERATURE));
    }
    let velocity = [0.3, -0.2, 0.1];
    for p in sim.particles.iter_mut() {
        p.velocity = velocity;
    }
    let mass = total_mass(&sim);

    sim.clear_grid();
    sim.particle_to_grid();
    let grid_mass: f64 = sim.grid.nodes.iter().map(|n| n.mass).sum();
    assert!(
        (grid_mass - mass).abs() < 1e-12 * mass,
        "{grid_mass} vs {mass}"
    );
    let mut momentum = [0.0; 3];
    for node in &sim.grid.nodes {
        for (total, part) in momentum.iter_mut().zip(node.momentum.iter()) {
            *total += part;
        }
    }
    for (total, v) in momentum.iter().zip(velocity.iter()) {
        assert!(
            (total - mass * v).abs() < 1e-12 * mass,
            "momentum {momentum:?}"
        );
    }

    // Scramble the particle velocities: G2P restores the uniform grid field.
    for p in sim.particles.iter_mut() {
        p.velocity = [9.0; 3];
    }
    for state in sim.mpm_particles.iter_mut() {
        state.velocity = [9.0; 3];
    }
    sim.grid_to_particle();
    for p in &sim.particles {
        for (got, want) in p.velocity.iter().zip(velocity.iter()) {
            assert!((got - want).abs() < 1e-12, "{:?}", p.velocity);
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

/// A block of snow on the floor stays a block: after one second its height has
/// changed by less than 2 % (the finding asked for 10 %), nothing sank into the
/// floor, it is at rest, and — being far inside the elastic range — it has not
/// hardened.  The former gravity-only integrator let the block fall through the
/// domain.
#[test]
fn snow_block_rests_on_the_floor() {
    let mut sim = snow_block(0.0, 0.0);
    let h = sim.cell_size;
    let height0 = height(&sim);
    let mass0 = total_mass(&sim);
    run(&mut sim, 1.0);

    assert!(all_finite(&sim));
    let height1 = height(&sim);
    assert!(
        (height1 - height0).abs() < 0.02 * height0,
        "block height changed from {height0} to {height1}"
    );
    // The floor plane is the innermost sticky node row at y = h: nothing falls below it.
    assert!(min_y(&sim) > h, "min y = {}", min_y(&sim));
    assert!(max_speed(&sim) < 0.05, "max speed = {}", max_speed(&sim));
    // Gravity compresses the block by ~0.5 %, far below the critical strain of
    // 1.9 %: no plastic flow, no hardening.
    assert!(max_hardness(&sim) < 1.01, "hardness {}", max_hardness(&sim));
    assert!(min_det_f(&sim) > 0.99, "min det F = {}", min_det_f(&sim));
    assert!(
        min_det_f(&sim) < 1.0,
        "the block must be compressed a little"
    );
    assert!((total_mass(&sim) - mass0).abs() < 1e-12);
}

/// A block released above the floor lands, is compacted by the impact (a fall of
/// 1.25 cells is ~1.2 m/s, beyond the elastic limit of the snow) and then rests:
/// it neither bounces away nor keeps falling.
#[test]
fn snow_block_released_above_the_floor_lands_and_settles() {
    let mut sim = snow_block(1.0, 0.0);
    let h = sim.cell_size;
    let start_min_y = min_y(&sim);
    run(&mut sim, 1.0);
    assert!(all_finite(&sim));
    // It dropped, but the sticky floor layers stopped it.
    assert!(
        min_y(&sim) < start_min_y - 0.1 * h,
        "block has not landed: min y = {} (start {start_min_y})",
        min_y(&sim)
    );
    assert!(min_y(&sim) > h, "min y = {}", min_y(&sim));
    assert!(max_speed(&sim) < 0.05, "max speed = {}", max_speed(&sim));
    // The impact compacted the lowest particles permanently.
    assert!(max_hardness(&sim) > 1.2, "hardness {}", max_hardness(&sim));
    assert!(min_det_f(&sim) < 0.97, "min det F = {}", min_det_f(&sim));
}

/// Snow hit by the floor at 2 m/s is compacted permanently (the elastic strain
/// is limited to the critical compression, the rest moves into the plastic
/// part) and hardens; it neither bounces off like an elastic body nor blows up.
#[test]
fn snow_compacts_plastically_and_hardens_on_impact() {
    let mut sim = snow_block(0.0, -2.0);
    let height0 = height(&sim);
    run(&mut sim, 0.6);

    assert!(all_finite(&sim));
    let height1 = height(&sim);
    assert!(
        height1 < 0.99 * height0,
        "impact must compact the block: {height0} -> {height1}"
    );
    assert!(height1 > 0.5 * height0, "block collapsed: {height1}");
    assert!(
        max_hardness(&sim) > 1.1,
        "compacted snow must be harder: {}",
        max_hardness(&sim)
    );
    assert!(min_det_f(&sim) < 0.97, "min det F = {}", min_det_f(&sim));
    // Plastic flow dissipates the impact: the block ends up at rest.
    assert!(max_speed(&sim) < 0.5, "max speed = {}", max_speed(&sim));
}

/// With a huge critical compression the snow never yields: the same impact does
/// not harden it (this is the test that the critical strains reach the solver).
#[test]
fn critical_compression_controls_the_plastic_flow() {
    let mut elastic = snow_block(0.0, -2.0);
    elastic.critical_compression = 0.6;
    elastic.critical_stretch = 0.6;
    run(&mut elastic, 0.3);
    assert!(all_finite(&elastic));
    assert!(
        max_hardness(&elastic) < 1.01,
        "no plastic flow expected, hardness {}",
        max_hardness(&elastic)
    );

    let mut snow = snow_block(0.0, -2.0);
    run(&mut snow, 0.3);
    assert!(
        max_hardness(&snow) > 1.1,
        "hardness {}",
        max_hardness(&snow)
    );
}

/// Stiffer snow compresses less in the same impact.
#[test]
fn stiffer_snow_compresses_less() {
    let compression = |young_modulus: f64| {
        let mut sim = snow_block(0.0, -1.0);
        sim.young_modulus = young_modulus;
        let height0 = height(&sim);
        let mut lowest = height0;
        let steps = (0.3 / DT).round() as usize;
        for _ in 0..steps {
            sim.step(DT);
            lowest = lowest.min(height(&sim));
        }
        assert!(all_finite(&sim));
        1.0 - lowest / height0
    };
    let soft = compression(1.4e5);
    let stiff = compression(5.6e5);
    assert!(
        soft > stiff + 0.01,
        "soft snow compressed by {soft}, stiff snow by {stiff}"
    );
}
