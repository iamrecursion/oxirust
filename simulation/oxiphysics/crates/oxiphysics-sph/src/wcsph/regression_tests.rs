// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the WCSPH force terms.
//!
//! Each test pins one physical property that an earlier, sign- or
//! scale-inverted implementation violated:
//!
//! - the Tait speed of sound is `sqrt(K/ρ₀)` at the rest density,
//! - δ-SPH density diffusion moves density from heavy to light particles,
//! - the hydrostatic-corrected variant leaves a Tait-hydrostatic column alone,
//! - the Riemann interface pressure rises for approaching pairs,
//! - the boundary pressure of a wall below the fluid is compressive,
//! - the viscous term of every WCSPH step is a kinematic-ν Laplacian that
//!   dissipates a shear wave at `ν k²`.

use super::*;
use crate::kernel::{CubicSplineKernel, SphKernel, grad};
use crate::neighbor::SpatialHash;
use crate::particle::{ParticleSet, SphParticle};
use oxiphysics_core::math::Vec3;

/// Pressure of the Tait EOS at density `rho` (via the production code path).
fn tait_pressure(rho: f64, params: &WcsphParams) -> f64 {
    let mut ps = ParticleSet::new();
    ps.add_particle(&SphParticle::new(Vec3::zeros(), Vec3::zeros(), 1.0));
    ps.densities[0] = rho;
    compute_pressure_tait(&mut ps, params);
    ps.pressures[0]
}

// ---------------------------------------------------------------------------
// F4735: speed of sound of the Tait equation of state
// ---------------------------------------------------------------------------

#[test]
fn speed_of_sound_at_rest_density_is_sqrt_k_over_rho0() {
    let c0 = 20.0_f64;
    let rho0 = 1000.0_f64;
    for gamma in [1.4, 7.0, 9.0] {
        let params = WcsphParams {
            rest_density: rho0,
            stiffness: rho0 * c0 * c0,
            gamma,
            viscosity: 0.0,
            smoothing_length: 0.1,
        };
        let c = speed_of_sound_tait(rho0, &params);
        assert!(
            (c - c0).abs() < 1e-12,
            "gamma={gamma}: c(rho0)={c}, expected {c0}"
        );
    }
}

#[test]
fn speed_of_sound_squared_is_the_derivative_of_the_tait_pressure() {
    let params = WcsphParams {
        rest_density: 1000.0,
        stiffness: 1000.0 * 30.0 * 30.0,
        gamma: 7.0,
        viscosity: 0.0,
        smoothing_length: 0.1,
    };
    for rho in [900.0, 1000.0, 1015.0, 1100.0] {
        let step = 1.0e-4 * rho;
        let dp_drho = (tait_pressure(rho + step, &params) - tait_pressure(rho - step, &params))
            / (2.0 * step);
        let c = speed_of_sound_tait(rho, &params);
        assert!(
            (c * c - dp_drho).abs() < 1e-6 * dp_drho,
            "rho={rho}: c^2={} vs dp/drho={dp_drho}",
            c * c
        );
    }
}

// ---------------------------------------------------------------------------
// F4739: δ-SPH density diffusion
// ---------------------------------------------------------------------------

/// Eight particles on a line at spacing `h/2`: the first four at `rho_low`, the
/// last four at `rho_high`; every particle lists all others as neighbours (the
/// kernel is zero outside its support).
fn density_step_line(h: f64, rho_low: f64, rho_high: f64) -> (ParticleSet, Vec<Vec<usize>>) {
    let spacing = 0.5 * h;
    let mass = 1000.0 * spacing.powi(3);
    let mut ps = ParticleSet::new();
    for ix in 0..8 {
        ps.add_particle(&SphParticle::new(
            Vec3::new(ix as f64 * spacing, 0.0, 0.0),
            Vec3::zeros(),
            mass,
        ));
        ps.densities[ix] = if ix < 4 { rho_low } else { rho_high };
    }
    let neighbors: Vec<Vec<usize>> = (0..8)
        .map(|i| (0..8).filter(|&j| j != i).collect())
        .collect();
    (ps, neighbors)
}

#[test]
fn fourtakas_diffusion_raises_light_and_lowers_heavy_interface_particles() {
    let h = 0.1;
    let (mut ps, neighbors) = density_step_line(h, 800.0, 1200.0);
    let rho_before = ps.densities.clone();
    let volumes: Vec<f64> = ps
        .masses
        .iter()
        .zip(rho_before.iter())
        .map(|(m, rho)| m / rho)
        .collect();
    apply_fourtakas_diffusion(
        &mut ps,
        &neighbors,
        &CubicSplineKernel,
        h,
        0.1,
        1480.0,
        1.0e-3,
    );
    let d_rho: Vec<f64> = ps
        .densities
        .iter()
        .zip(rho_before.iter())
        .map(|(after, before)| after - before)
        .collect();
    // The particles on both sides of the jump (indices 3 and 4) are the ones
    // that see an opposite-density neighbour inside 2h.
    assert!(
        d_rho[3] > 1.0,
        "light interface particle must gain: {d_rho:?}"
    );
    assert!(
        d_rho[4] < -1.0,
        "heavy interface particle must lose: {d_rho:?}"
    );
    // Particles 0 and 7 sit at q = 2 from the nearest opposite-density
    // particle, outside the kernel support: they are untouched.
    assert!(
        d_rho[0].abs() < 1e-12 && d_rho[7].abs() < 1e-12,
        "{d_rho:?}"
    );
    // The sign follows the side of the jump everywhere.
    for i in 0..4 {
        assert!(
            d_rho[i] >= 0.0,
            "light side must not lose density: {d_rho:?}"
        );
        assert!(
            d_rho[7 - i] <= 0.0,
            "heavy side must not gain density: {d_rho:?}"
        );
    }
    // The jump between the interface particles shrinks.
    assert!(
        ps.densities[4] - ps.densities[3] < rho_before[4] - rho_before[3],
        "density jump must shrink: {:?}",
        ps.densities
    );
    // Mass conservation: the pair terms are antisymmetric in the
    // volume-weighted sum computed with the pre-update densities.
    let net: f64 = volumes.iter().zip(d_rho.iter()).map(|(v, d)| v * d).sum();
    let scale: f64 = volumes
        .iter()
        .zip(d_rho.iter())
        .map(|(v, d)| (v * d).abs())
        .sum();
    assert!(net.abs() < 1e-12 * scale, "net={net}, scale={scale}");
}

#[test]
fn fourtakas_diffusion_relaxes_a_density_step_without_overshoot() {
    let h = 0.1;
    let (mut ps, neighbors) = density_step_line(h, 800.0, 1200.0);
    let spread = |ps: &ParticleSet| {
        let max = ps.densities.iter().copied().fold(f64::MIN, f64::max);
        let min = ps.densities.iter().copied().fold(f64::MAX, f64::min);
        max - min
    };
    let mut previous = spread(&ps);
    for _ in 0..20 {
        apply_fourtakas_diffusion(
            &mut ps,
            &neighbors,
            &CubicSplineKernel,
            h,
            0.1,
            1480.0,
            1.0e-4,
        );
        let now = spread(&ps);
        assert!(now <= previous + 1e-9, "spread grew: {previous} -> {now}");
        previous = now;
    }
    assert!(previous < 400.0 - 20.0, "spread barely changed: {previous}");
    for &rho in &ps.densities {
        assert!(
            (800.0 - 1e-9..=1200.0 + 1e-9).contains(&rho),
            "maximum principle violated: {rho}"
        );
    }
}

#[test]
fn fourtakas_diffusion_agrees_with_the_array_based_delta_sph_formula() {
    let h = 0.1;
    let (ps, neighbors) = density_step_line(h, 800.0, 1200.0);
    let kernel = CubicSplineKernel;
    let positions: Vec<[f64; 3]> = ps.positions.iter().map(|p| [p.x, p.y, p.z]).collect();
    for i in 0..8 {
        let expected = {
            let nbrs = &neighbors[i];
            let grad_ws: Vec<[f64; 3]> = nbrs
                .iter()
                .map(|&j| {
                    grad(
                        &kernel,
                        [
                            positions[i][0] - positions[j][0],
                            positions[i][1] - positions[j][1],
                            positions[i][2] - positions[j][2],
                        ],
                        h,
                    )
                })
                .collect();
            crate::viscosity::delta_sph_density_diffusion(
                i,
                &positions,
                &ps.densities,
                &ps.masses,
                &grad_ws,
                nbrs,
                0.1,
                h,
                1480.0,
            )
        };
        let actual = fourtakas_density_diffusion(
            i,
            &ps.positions,
            &ps.masses,
            &ps.densities,
            &neighbors,
            &kernel,
            h,
            0.1,
            1480.0,
        );
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "particle {i}: {actual} vs {expected}"
        );
    }
}

/// Hydrostatic column of `n` particles at spacing `dx` below the free surface
/// `z = 0` with the exact Tait stratification of a column of fluid with
/// `K = ρ₀ c₀²`, plus a neighbour list within `2h`.
fn hydrostatic_column(
    n: usize,
    dx: f64,
    h: f64,
    params: &WcsphParams,
    g: f64,
) -> (ParticleSet, Vec<Vec<usize>>) {
    let rho0 = params.rest_density;
    let gamma = params.gamma;
    let c_b = params.stiffness / gamma;
    let mut ps = ParticleSet::new();
    for k in 0..n {
        let depth = k as f64 * dx;
        ps.add_particle(&SphParticle::new(
            Vec3::new(0.0, -depth, 0.0),
            Vec3::zeros(),
            rho0 * dx.powi(3),
        ));
        // Exact solution of dp/dz = -ρ(p) g for the Tait law, p(0) = 0.
        let base = 1.0 + (1.0 - 1.0 / gamma) * rho0 * g * depth / c_b;
        ps.densities[k] = rho0 * base.powf(1.0 / (gamma - 1.0));
    }
    let neighbors = SpatialHash::find_all_neighbors(&ps.positions, 2.0 * h);
    (ps, neighbors)
}

#[test]
fn hydrostatic_density_difference_has_the_physical_sign() {
    let reference = HydrostaticReference::new(1000.0, 1.0e7 / 7.0, 7.0, Vec3::new(0.0, -9.81, 0.0));
    let below = reference.density_difference(Vec3::new(0.0, -0.1, 0.0));
    let above = reference.density_difference(Vec3::new(0.0, 0.1, 0.0));
    let sideways = reference.density_difference(Vec3::new(0.1, 0.0, -0.1));
    // Linear hydrostatics: Δρ = ρ₀ g Δz / c₀² with c₀² = K/ρ₀ = 1e4.
    let linear = 1000.0 * 9.81 * 0.1 / 1.0e4;
    assert!(below > 0.0 && above < 0.0, "below={below}, above={above}");
    assert!(
        (below - linear).abs() < 2e-3 * linear,
        "below={below}, linear={linear}"
    );
    assert!(
        (above + linear).abs() < 2e-3 * linear,
        "above={above}, linear={linear}"
    );
    assert!(
        sideways.abs() < 1e-12,
        "no hydrostatic change sideways: {sideways}"
    );
}

#[test]
fn hydrostatic_reference_from_params_uses_the_tait_coefficient() {
    let params = WcsphParams {
        rest_density: 1000.0,
        stiffness: 7.0e6,
        gamma: 7.0,
        viscosity: 0.0,
        smoothing_length: 0.1,
    };
    let reference = HydrostaticReference::from_params(&params, Vec3::new(0.0, 0.0, -9.81));
    assert!((reference.tait_coefficient - 1.0e6).abs() < 1e-6);
    assert_eq!(reference.gamma, 7.0);
    assert_eq!(reference.rest_density, 1000.0);
}

#[test]
fn hydrostatic_correction_leaves_a_hydrostatic_column_surface_and_bottom_alone() {
    let g = 9.81;
    let dx = 0.05;
    let h = 1.3 * dx;
    let params = WcsphParams {
        rest_density: 1000.0,
        stiffness: 1000.0 * 100.0 * 100.0,
        gamma: 7.0,
        viscosity: 0.0,
        smoothing_length: h,
    };
    let n = 12;
    let (ps, neighbors) = hydrostatic_column(n, dx, h, &params, g);
    let reference = HydrostaticReference::from_params(&params, Vec3::new(0.0, -g, 0.0));
    let kernel = CubicSplineKernel;
    let c_s = 100.0;
    for &i in &[0_usize, n - 1] {
        let plain = fourtakas_density_diffusion(
            i,
            &ps.positions,
            &ps.masses,
            &ps.densities,
            &neighbors,
            &kernel,
            h,
            0.1,
            c_s,
        );
        let corrected = fourtakas_hydrostatic_density_diffusion(
            i,
            &ps.positions,
            &ps.masses,
            &ps.densities,
            &neighbors,
            &kernel,
            h,
            0.1,
            c_s,
            &reference,
        );
        // A one-sided kernel at the free surface / floor sees only a density
        // jump of one sign: the plain term pushes the surface density up
        // (the column "leaks" density to the top), the floor density down.
        if i == 0 {
            assert!(
                plain > 0.0,
                "surface: plain diffusion should be positive: {plain}"
            );
        } else {
            assert!(
                plain < 0.0,
                "floor: plain diffusion should be negative: {plain}"
            );
        }
        assert!(
            corrected.abs() < 0.02 * plain.abs(),
            "particle {i}: corrected {corrected} should be <2% of plain {plain}"
        );
    }
}

#[test]
fn hydrostatic_correction_without_gravity_reduces_to_the_plain_term() {
    let h = 0.1;
    let (ps, neighbors) = density_step_line(h, 800.0, 1200.0);
    let reference = HydrostaticReference::new(1000.0, 1.0e6, 7.0, Vec3::zeros());
    for i in 0..8 {
        let plain = fourtakas_density_diffusion(
            i,
            &ps.positions,
            &ps.masses,
            &ps.densities,
            &neighbors,
            &CubicSplineKernel,
            h,
            0.1,
            1480.0,
        );
        let corrected = fourtakas_hydrostatic_density_diffusion(
            i,
            &ps.positions,
            &ps.masses,
            &ps.densities,
            &neighbors,
            &CubicSplineKernel,
            h,
            0.1,
            1480.0,
            &reference,
        );
        assert_eq!(plain, corrected, "particle {i}");
    }
}

#[test]
fn hydrostatic_apply_keeps_a_hydrostatic_column_closer_to_equilibrium_than_plain() {
    let g = 9.81;
    let dx = 0.05;
    let h = 1.3 * dx;
    let params = WcsphParams {
        rest_density: 1000.0,
        stiffness: 1000.0 * 100.0 * 100.0,
        gamma: 7.0,
        viscosity: 0.0,
        smoothing_length: h,
    };
    let (initial, neighbors) = hydrostatic_column(12, dx, h, &params, g);
    let reference = HydrostaticReference::from_params(&params, Vec3::new(0.0, -g, 0.0));
    let mut plain = initial.clone();
    let mut corrected = initial.clone();
    for _ in 0..10 {
        apply_fourtakas_diffusion(
            &mut plain,
            &neighbors,
            &CubicSplineKernel,
            h,
            0.1,
            100.0,
            1e-4,
        );
        apply_fourtakas_hydrostatic_diffusion(
            &mut corrected,
            &neighbors,
            &CubicSplineKernel,
            h,
            0.1,
            100.0,
            1e-4,
            &reference,
        );
    }
    let drift = |ps: &ParticleSet| -> f64 {
        ps.densities
            .iter()
            .zip(initial.densities.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    };
    assert!(drift(&plain) > 0.0);
    assert!(
        drift(&corrected) < 0.05 * drift(&plain),
        "corrected drift {} vs plain drift {}",
        drift(&corrected),
        drift(&plain)
    );
}

// ---------------------------------------------------------------------------
// F4740: Riemann interface pressure
// ---------------------------------------------------------------------------

#[test]
fn riemann_interface_pressure_rises_for_approaching_pairs() {
    let p = 1000.0;
    let z = 1.0e6;
    let at_rest = riemann_interface_pressure(p, p, z, z, 0.0);
    let approaching = riemann_interface_pressure(p, p, z, z, 1.0);
    let receding = riemann_interface_pressure(p, p, z, z, -1.0);
    assert_eq!(at_rest, p);
    // Z_i Z_j / (Z_i + Z_j) = Z/2 for equal impedances.
    assert!((approaching - (p + 0.5 * z)).abs() < 1e-9, "{approaching}");
    assert!((receding - (p - 0.5 * z)).abs() < 1e-9, "{receding}");
    assert!(approaching > p && receding < p);
}

#[test]
fn riemann_interface_pressure_weights_the_pressures_by_the_opposite_impedance() {
    // Z_i = 3, Z_j = 1: p* = (3 p_j + 1 p_i) / 4.
    let p_star = riemann_interface_pressure(0.0, 100.0, 3.0, 1.0, 0.0);
    assert!((p_star - 75.0).abs() < 1e-12, "{p_star}");
    // A rigid wall (Z_i -> ∞) imposes its own velocity and ignores p_i:
    // p* -> p_j + Z_j (v_i - v_j).e_ij.
    let wall = riemann_interface_pressure(5.0e3, 100.0, 1.0e15, 2.0, 3.0);
    assert!((wall - (100.0 + 2.0 * 3.0)).abs() < 1e-8, "{wall}");
    // Vanishing impedances fall back to the mean.
    assert_eq!(riemann_interface_pressure(10.0, 30.0, 0.0, 0.0, 5.0), 20.0);
}

/// Two particles of equal pressure/density at spacing `s` along x, moving toward
/// each other with relative approach speed `approach` (negative: receding).
fn riemann_pair_force(approach: f64) -> (Vec3, Vec3) {
    let params = WcsphParams::default();
    let s: f64 = 0.04;
    let h = 0.05;
    let mass = 1000.0 * s.powi(3);
    let mut ps = ParticleSet::new();
    ps.add_particle(&SphParticle::new(
        Vec3::zeros(),
        Vec3::new(0.5 * approach, 0.0, 0.0),
        mass,
    ));
    ps.add_particle(&SphParticle::new(
        Vec3::new(s, 0.0, 0.0),
        Vec3::new(-0.5 * approach, 0.0, 0.0),
        mass,
    ));
    ps.densities = vec![1000.0; 2];
    ps.pressures = vec![1000.0; 2];
    ps.clear_forces();
    compute_riemann_pressure_force(&mut ps, &[vec![1], vec![0]], &CubicSplineKernel, h, &params);
    (ps.forces[0], ps.forces[1])
}

#[test]
fn riemann_pressure_force_is_stronger_for_approaching_and_weaker_for_receding_pairs() {
    let (rest0, rest1) = riemann_pair_force(0.0);
    let (close0, close1) = riemann_pair_force(0.2);
    let (open0, open1) = riemann_pair_force(-0.2);
    // Particle 0 sits left of particle 1: a repulsive force points to -x.
    assert!(rest0.x < 0.0 && rest1.x > 0.0, "{rest0:?} {rest1:?}");
    assert!(
        close0.x < rest0.x,
        "approaching must repel harder: {close0:?} vs {rest0:?}"
    );
    assert!(
        open0.x > rest0.x && open0.x < 0.0,
        "receding must repel less: {open0:?}"
    );
    // Momentum is conserved pairwise.
    for (a, b) in [(rest0, rest1), (close0, close1), (open0, open1)] {
        assert!((a + b).norm() < 1e-9 * a.norm(), "{a:?} {b:?}");
    }
    // F ∝ p*: p* = 1000 ± (Z/2)·0.2 with Z = ρ c0 = 1000·sqrt(50).
    let half_z = 0.5 * 1000.0 * 50.0_f64.sqrt();
    let expected_close = (1000.0 + half_z * 0.2) / 1000.0;
    let expected_open = (1000.0 - half_z * 0.2) / 1000.0;
    assert!((close0.x / rest0.x - expected_close).abs() < 1e-9);
    assert!((open0.x / rest0.x - expected_open).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// F4741: dynamic boundary pressure
// ---------------------------------------------------------------------------

#[test]
fn boundary_pressure_below_the_fluid_is_compressive() {
    let g = Vec3::new(0.0, -9.81, 0.0);
    let x_f = Vec3::zeros();
    let below = dynamic_boundary_pressure(0.0, 1000.0, x_f, Vec3::new(0.0, -0.5, 0.0), g);
    let above = dynamic_boundary_pressure(0.0, 1000.0, x_f, Vec3::new(0.0, 0.5, 0.0), g);
    let beside = dynamic_boundary_pressure(2.0e3, 1000.0, x_f, Vec3::new(0.3, 0.0, 0.0), g);
    assert!((below - 4905.0).abs() < 1e-9, "wall 0.5 m below: {below}");
    assert!((above + 4905.0).abs() < 1e-9, "wall 0.5 m above: {above}");
    assert!(
        (beside - 2.0e3).abs() < 1e-9,
        "wall at the same height: {beside}"
    );
}

#[test]
fn boundary_pressure_of_an_accelerating_wall_includes_the_wall_acceleration() {
    let g = Vec3::new(0.0, -9.81, 0.0);
    let x_f = Vec3::zeros();
    let x_b = Vec3::new(0.0, -0.5, 0.0);
    // A floor accelerating upward at 2 m/s² loads the fluid with (g + a) h.
    let lifting =
        dynamic_boundary_pressure_moving(0.0, 1000.0, x_f, x_b, g, Vec3::new(0.0, 2.0, 0.0));
    assert!((lifting - 1000.0 * 11.81 * 0.5).abs() < 1e-9, "{lifting}");
    // A wall in free fall exerts no hydrostatic load.
    let falling = dynamic_boundary_pressure_moving(700.0, 1000.0, x_f, x_b, g, g);
    assert!((falling - 700.0).abs() < 1e-9, "{falling}");
    // The resting-wall function is the a_w = 0 special case.
    let resting = dynamic_boundary_pressure(0.0, 1000.0, x_f, x_b, g);
    let resting_moving = dynamic_boundary_pressure_moving(0.0, 1000.0, x_f, x_b, g, Vec3::zeros());
    assert_eq!(resting, resting_moving);
}

#[test]
fn fluid_particle_resting_on_a_wall_is_pushed_away_by_the_extrapolated_pressure() {
    let g = Vec3::new(0.0, -9.81, 0.0);
    let h = 0.1;
    let x_fluid = Vec3::new(0.0, 0.05, 0.0);
    let x_wall = Vec3::zeros();
    let rho = 1000.0;
    // Fluid pressure at the fluid particle, wall pressure extrapolated from it.
    let p_fluid = 0.0;
    let p_wall = dynamic_boundary_pressure(p_fluid, rho, x_fluid, x_wall, g);
    assert!(p_wall > 0.0, "wall below the fluid is compressed: {p_wall}");
    let force = dynamic_boundary_force_contribution(
        p_fluid,
        p_wall,
        rho,
        rho,
        rho * 0.05_f64.powi(3),
        x_fluid - x_wall,
        &CubicSplineKernel,
        h,
    );
    assert!(
        force.y > 0.0,
        "wall must push the fluid particle up: {force:?}"
    );
    assert!(force.x.abs() < 1e-12 && force.z.abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// F4734: viscosity is a kinematic-ν Laplacian, not ν/ρ
// ---------------------------------------------------------------------------

/// A 3-D block of particles carrying the shear wave `v_y = V sin(k x)`.
struct ShearBlock {
    ps: ParticleSet,
    neighbors: Vec<Vec<usize>>,
    params: WcsphParams,
    /// Wave number of the velocity field.
    k: f64,
    /// Indices of the probe particles: the centre line, at least `2h` away
    /// from every face of the block.
    probes: Vec<usize>,
}

fn shear_block(nu: f64) -> ShearBlock {
    let dx = 0.02;
    let h = 1.3 * dx;
    let (nx, ny, nz) = (16_usize, 7_usize, 7_usize);
    let amplitude = 0.1;
    let k = 2.0 * std::f64::consts::PI / (nx as f64 * dx);
    let mass = 1000.0 * dx.powi(3);
    let mut ps = ParticleSet::new();
    let mut probes = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let x = ix as f64 * dx;
                let velocity = Vec3::new(0.0, amplitude * (k * x).sin(), 0.0);
                ps.add_particle(&SphParticle::new(
                    Vec3::new(x, iy as f64 * dx, iz as f64 * dx),
                    velocity,
                    mass,
                ));
                if iy == ny / 2 && iz == nz / 2 && (3..nx - 3).contains(&ix) {
                    probes.push(ps.len() - 1);
                }
            }
        }
    }
    let neighbors = SpatialHash::find_all_neighbors(&ps.positions, 2.0 * h);
    let params = WcsphParams {
        rest_density: 1000.0,
        stiffness: 50_000.0,
        gamma: 7.0,
        viscosity: nu,
        smoothing_length: h,
    };
    ShearBlock {
        ps,
        neighbors,
        params,
        k,
        probes,
    }
}

impl ShearBlock {
    /// Kinetic energy of the probe particles.
    fn probe_energy(&self) -> f64 {
        self.probes
            .iter()
            .map(|&i| 0.5 * self.ps.masses[i] * self.ps.velocities[i].norm_squared())
            .sum()
    }
}

/// Exponential decay rate of the probe kinetic energy over one step of `advance`.
fn probe_energy_decay_rate(
    block: &mut ShearBlock,
    dt: f64,
    advance: impl FnOnce(&mut ShearBlock, f64),
) -> f64 {
    let before = block.probe_energy();
    advance(block, dt);
    let after = block.probe_energy();
    -(after / before).ln() / dt
}

#[test]
fn step_dissipates_a_shear_wave_at_two_nu_k_squared() {
    let nu = 0.01;
    let dt = 1.0e-3;
    let mut block = shear_block(nu);
    assert!(block.probes.len() >= 8);
    let expected = 2.0 * nu * block.k * block.k;
    let rate = probe_energy_decay_rate(&mut block, dt, |b, dt| {
        step(
            &mut b.ps,
            &b.neighbors,
            &CubicSplineKernel,
            &b.params,
            dt,
            Vec3::zeros(),
        );
    });
    assert!(
        (rate - expected).abs() < 0.10 * expected,
        "KE decay rate {rate} vs 2 nu k^2 = {expected}"
    );
}

#[test]
fn step_with_corrections_dissipates_a_shear_wave_at_two_nu_k_squared() {
    let nu = 0.01;
    let dt = 1.0e-3;
    let mut block = shear_block(nu);
    let expected = 2.0 * nu * block.k * block.k;
    let rate = probe_energy_decay_rate(&mut block, dt, |b, dt| {
        let delta_p = 0.02;
        step_with_corrections(
            &mut b.ps,
            &b.neighbors,
            &CubicSplineKernel,
            &b.params,
            dt,
            Vec3::zeros(),
            0.0,
            0.0,
            4,
            delta_p,
        );
    });
    assert!(
        (rate - expected).abs() < 0.10 * expected,
        "KE decay rate {rate} vs 2 nu k^2 = {expected}"
    );
}

#[test]
fn shear_decay_rate_tracks_the_kinematic_viscosity() {
    // The decay rate must follow 2 nu k^2 over a range of viscosities, i.e. the
    // solver applies nu (m^2/s) itself and not nu / rho or nu * rho.
    let dt = 1.0e-3;
    for nu in [0.005, 0.02, 0.05] {
        let mut block = shear_block(nu);
        let expected = 2.0 * nu * block.k * block.k;
        let rate = probe_energy_decay_rate(&mut block, dt, |b, dt| {
            step(
                &mut b.ps,
                &b.neighbors,
                &CubicSplineKernel,
                &b.params,
                dt,
                Vec3::zeros(),
            );
        });
        assert!(
            (rate - expected).abs() < 0.10 * expected,
            "nu={nu}: KE decay rate {rate} vs 2 nu k^2 = {expected}"
        );
    }
}

#[test]
fn riemann_steps_dissipate_a_shear_wave_faster_than_viscosity_alone() {
    let nu = 0.01;
    let dt = 5.0e-4;
    let mut plain_block = shear_block(nu);
    let plain = probe_energy_decay_rate(&mut plain_block, dt, |b, dt| {
        step(
            &mut b.ps,
            &b.neighbors,
            &CubicSplineKernel,
            &b.params,
            dt,
            Vec3::zeros(),
        );
    });
    let mut riemann_block = shear_block(nu);
    let riemann = probe_energy_decay_rate(&mut riemann_block, dt, |b, dt| {
        step_riemann(
            &mut b.ps,
            &b.neighbors,
            &CubicSplineKernel,
            &b.params,
            dt,
            Vec3::zeros(),
        );
    });
    let mut riemann_corrected_block = shear_block(nu);
    let riemann_corrected = probe_energy_decay_rate(&mut riemann_corrected_block, dt, |b, dt| {
        step_riemann_with_corrections(
            &mut b.ps,
            &b.neighbors,
            &CubicSplineKernel,
            &b.params,
            dt,
            Vec3::zeros(),
            0.0,
            0.0,
        );
    });
    let viscous = 2.0 * nu * plain_block.k * plain_block.k;
    assert!(
        (plain - viscous).abs() < 0.10 * viscous,
        "{plain} vs {viscous}"
    );
    // The acoustic Riemann dissipation adds to the physical viscosity; with the
    // anti-dissipative sign the energy would decay slower than `plain`.
    assert!(riemann > 1.5 * plain, "riemann {riemann} vs plain {plain}");
    assert!(
        riemann_corrected > 1.5 * plain,
        "riemann+corrections {riemann_corrected} vs plain {plain}"
    );
}

/// Deterministic pseudo-random numbers in `[-1, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0
}

#[test]
fn viscous_forces_never_add_energy_for_arbitrary_velocity_fields() {
    let mut block = shear_block(0.0);
    let kernel = CubicSplineKernel;
    let h = block.params.smoothing_length;
    compute_density(&mut block.ps, &block.neighbors, &kernel, h);
    let mut state = 0x1234_5678_9abc_def0_u64;
    for v in block.ps.velocities.iter_mut() {
        *v = Vec3::new(
            lcg_unit(&mut state),
            lcg_unit(&mut state),
            lcg_unit(&mut state),
        );
    }
    type ViscousForce = fn(&mut ParticleSet, &[Vec<usize>], &dyn SphKernel, f64, f64);
    let models: [(&str, ViscousForce, f64); 2] = [
        ("laminar", compute_laminar_viscosity, 10.0),
        ("kinematic", compute_viscosity_force, 0.01),
    ];
    for (name, force, coefficient) in models {
        block.ps.clear_forces();
        force(&mut block.ps, &block.neighbors, &kernel, h, coefficient);
        // Power = Σ m_i v_i · a_i with a_i = f_i / ρ_i.
        let power: f64 = (0..block.ps.len())
            .map(|i| {
                block.ps.masses[i] * block.ps.velocities[i].dot(&block.ps.forces[i])
                    / block.ps.densities[i]
            })
            .sum();
        assert!(
            power < 0.0,
            "{name}: viscous power must be negative: {power}"
        );
    }
}

#[test]
fn kinematic_viscosity_force_equals_laminar_viscosity_at_uniform_density() {
    let kernel = CubicSplineKernel;
    let h = 0.05;
    let mut ps = ParticleSet::new();
    let mass = 1000.0 * 0.03_f64.powi(3);
    for (x, vy) in [(0.0, 0.0), (0.03, 0.2), (0.07, -0.1)] {
        ps.add_particle(&SphParticle::new(
            Vec3::new(x, 0.0, 0.0),
            Vec3::new(0.0, vy, 0.0),
            mass,
        ));
    }
    ps.densities = vec![1000.0; 3];
    let neighbors = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let mut kinematic = ps.clone();
    kinematic.clear_forces();
    compute_viscosity_force(&mut kinematic, &neighbors, &kernel, h, 0.01);
    let mut laminar = ps.clone();
    laminar.clear_forces();
    compute_laminar_viscosity(&mut laminar, &neighbors, &kernel, h, 0.01 * 1000.0);
    for i in 0..3 {
        assert!(kinematic.forces[i].norm() > 0.0);
        assert!(
            (kinematic.forces[i] - laminar.forces[i]).norm() <= 1e-12 * laminar.forces[i].norm(),
            "particle {i}: {:?} vs {:?}",
            kinematic.forces[i],
            laminar.forces[i]
        );
    }
}
