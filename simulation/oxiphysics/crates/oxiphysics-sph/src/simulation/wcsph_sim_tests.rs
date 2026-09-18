//! Regression tests for the neighbour-list based [`WcSphSim`] pipeline.
//!
//! The density summation, the symmetric pressure acceleration, the Morris
//! viscosity and the continuum surface force all run over neighbour lists built
//! once per step.  Every test compares the production sums with an independent
//! all-pairs evaluation (plain double loops over `0..n`), checks the Euler step
//! against the same reference, and verifies that the surface force pulls the
//! surface of a convex droplet inwards while leaving its bulk untouched.

use super::types_sim::WcSphSim;
use crate::particle::{cubic_spline_gradient, cubic_spline_kernel};

/// Rest density (kg/m³).
const RHO0: f64 = 1000.0;
/// Numerical speed of sound (m/s).
const C0: f64 = 10.0;
/// Tait exponent.
const GAMMA: f64 = 7.0;
/// Dynamic viscosity (Pa·s).
const MU: f64 = 0.05;
/// Surface tension coefficient of water in air (N/m).
const SIGMA: f64 = 0.0728;
/// Gravity (m/s²).
const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];

/// Deterministic pseudo-random number in `[0, 1)` (64-bit LCG).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

/// Relative comparison with an absolute floor for values near zero.
fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= abs + rel * b.abs()
}

/// Difference vector `a − b` and its length.
fn separation(a: [f64; 3], b: [f64; 3]) -> ([f64; 3], f64) {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
}

/// Random cloud of `n` particles with random masses and velocities in
/// `[0, extent]³`; every ninth particle is a boundary particle.
fn cloud_sim(n: usize, extent: f64, h: f64, sigma: f64, seed: u64) -> WcSphSim {
    let mut state = seed;
    let mut sim = WcSphSim::new(h, RHO0, C0, GAMMA, MU, sigma, GRAVITY);
    let mass0 = RHO0 * extent.powi(3) / n as f64;
    for i in 0..n {
        let position = [
            extent * lcg_unit(&mut state),
            extent * lcg_unit(&mut state),
            extent * lcg_unit(&mut state),
        ];
        let velocity = [
            lcg_unit(&mut state) - 0.5,
            lcg_unit(&mut state) - 0.5,
            lcg_unit(&mut state) - 0.5,
        ];
        let mass = mass0 * (0.8 + 0.4 * lcg_unit(&mut state));
        sim.particles
            .add_raw(position, velocity, mass, MU, i % 9 == 4);
    }
    sim
}

/// Spherical droplet carved out of a cubic lattice of spacing `dx`: all lattice
/// points with `i² + j² + k² ≤ radius_cells²`, centred on the origin.
fn droplet_sim(radius_cells: i32, dx: f64, sigma: f64) -> WcSphSim {
    let mut sim = WcSphSim::new(1.3 * dx, RHO0, C0, GAMMA, 0.0, sigma, [0.0; 3]);
    let mass = RHO0 * dx * dx * dx;
    for i in -radius_cells..=radius_cells {
        for j in -radius_cells..=radius_cells {
            for k in -radius_cells..=radius_cells {
                if i * i + j * j + k * k <= radius_cells * radius_cells {
                    let position = [i as f64 * dx, j as f64 * dx, k as f64 * dx];
                    sim.particles.add_raw(position, [0.0; 3], mass, 0.0, false);
                }
            }
        }
    }
    sim
}

// ---------------------------------------------------------------------------
// Independent all-pairs reference
// ---------------------------------------------------------------------------

/// `ρᵢ = Σⱼ mⱼ W(|xᵢ − xⱼ|, h)` over all particles (the particle itself included).
fn reference_densities(sim: &WcSphSim) -> Vec<f64> {
    let ps = &sim.particles;
    (0..ps.len())
        .map(|i| {
            (0..ps.len())
                .map(|j| {
                    let (_, r) = separation(ps.positions[i], ps.positions[j]);
                    ps.masses[j] * cubic_spline_kernel(r, sim.h)
                })
                .sum()
        })
        .collect()
}

/// Tait pressure `B ((ρ/ρ₀)^γ − 1)` with `B = ρ₀ c₀² / γ`.
fn reference_pressures(sim: &WcSphSim, densities: &[f64]) -> Vec<f64> {
    let b = sim.rho0 * sim.c0 * sim.c0 / sim.gamma;
    densities
        .iter()
        .map(|&rho| b * ((rho / sim.rho0).powf(sim.gamma) - 1.0))
        .collect()
}

/// Gravity, symmetric pressure acceleration and Morris viscosity of every fluid
/// particle, summed over all other particles inside the support radius `2h`.
fn reference_pressure_viscosity_gravity(
    sim: &WcSphSim,
    densities: &[f64],
    pressures: &[f64],
) -> Vec<[f64; 3]> {
    let ps = &sim.particles;
    let n = ps.len();
    let mut acc = vec![[0.0_f64; 3]; n];
    for i in 0..n {
        if ps.is_boundary[i] {
            continue;
        }
        let rho_i = densities[i].max(1e-20);
        let mut a = sim.gravity;
        for j in 0..n {
            if j == i {
                continue;
            }
            let (d, r) = separation(ps.positions[i], ps.positions[j]);
            if r < 1e-14 || r > 2.0 * sim.h {
                continue;
            }
            let rho_j = densities[j].max(1e-20);
            let dw = cubic_spline_gradient(r, sim.h);
            let pressure_factor = -ps.masses[j]
                * (pressures[i] / (rho_i * rho_i) + pressures[j] / (rho_j * rho_j))
                * dw
                / r;
            let viscous_factor = ps.masses[j] * 2.0 * sim.mu / (rho_i * rho_j) * dw / r;
            for (k, a_k) in a.iter_mut().enumerate() {
                *a_k += pressure_factor * d[k]
                    + viscous_factor * (ps.velocities[i][k] - ps.velocities[j][k]);
            }
        }
        acc[i] = a;
    }
    acc
}

/// Brackbill–Kothe–Zemach surface force with the colour function `ρ/ρ₀`:
/// returns the acceleration `σ κᵢ ∇cᵢ / ρᵢ` of every fluid particle together
/// with the curvature `κᵢ` (zero where the colour gradient vanishes).
fn reference_surface_force(sim: &WcSphSim, densities: &[f64]) -> (Vec<[f64; 3]>, Vec<f64>) {
    let ps = &sim.particles;
    let n = ps.len();
    let h = sim.h;
    let colour: Vec<f64> = densities
        .iter()
        .map(|&rho| rho.max(1e-14) / sim.rho0)
        .collect();
    // Separation d = xᵢ − xⱼ and the weight Vⱼ (dW/dr) / r of a pair in the support.
    let pair = |i: usize, j: usize| -> Option<([f64; 3], f64)> {
        let (d, r) = separation(ps.positions[i], ps.positions[j]);
        if j == i || r * r > 4.0 * h * h || r * r < 1e-28 {
            return None;
        }
        let volume_j = ps.masses[j] / densities[j].max(1e-14);
        Some((d, volume_j * cubic_spline_gradient(r, h) / r))
    };
    let mut grad_c = vec![[0.0_f64; 3]; n];
    for i in 0..n {
        for j in 0..n {
            if let Some((d, weight)) = pair(i, j) {
                for (g, d_k) in grad_c[i].iter_mut().zip(d.iter()) {
                    *g += weight * (colour[j] - colour[i]) * d_k;
                }
            }
        }
    }
    let normals: Vec<Option<[f64; 3]>> = grad_c
        .iter()
        .map(|g| {
            let mag2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            (mag2 >= 1e-20).then(|| {
                let mag = mag2.sqrt();
                [g[0] / mag, g[1] / mag, g[2] / mag]
            })
        })
        .collect();
    let mut acc = vec![[0.0_f64; 3]; n];
    let mut curvature = vec![0.0_f64; n];
    for i in 0..n {
        if ps.is_boundary[i] {
            continue;
        }
        let Some(n_i) = normals[i] else {
            continue;
        };
        let mut kappa = 0.0;
        for (j, normal_j) in normals.iter().enumerate() {
            if let (Some((d, weight)), Some(n_j)) = (pair(i, j), normal_j) {
                let dn_dot_d =
                    (n_j[0] - n_i[0]) * d[0] + (n_j[1] - n_i[1]) * d[1] + (n_j[2] - n_i[2]) * d[2];
                kappa -= weight * dn_dot_d;
            }
        }
        curvature[i] = kappa;
        let scale = sim.sigma * kappa / densities[i].max(1e-14);
        for (a_k, g_k) in acc[i].iter_mut().zip(grad_c[i].iter()) {
            *a_k = scale * g_k;
        }
    }
    (acc, curvature)
}

/// Reference acceleration of every particle: gravity, pressure, viscosity and,
/// for a non-zero `sigma`, the surface force.
fn reference_accelerations(sim: &WcSphSim, densities: &[f64], pressures: &[f64]) -> Vec<[f64; 3]> {
    let mut acc = reference_pressure_viscosity_gravity(sim, densities, pressures);
    if sim.sigma != 0.0 {
        let (surface, _) = reference_surface_force(sim, densities);
        for (a, s) in acc.iter_mut().zip(surface.iter()) {
            for (a_k, s_k) in a.iter_mut().zip(s.iter()) {
                *a_k += s_k;
            }
        }
    }
    acc
}

/// Density, pressure and force evaluation through the public step API.
fn evaluate(sim: &mut WcSphSim) {
    sim.density_step();
    sim.pressure_step();
    sim.force_step();
}

/// Component-wise comparison relative to the largest reference component.
fn assert_vectors_close(actual: &[[f64; 3]], expected: &[[f64; 3]], rel: f64, what: &str) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0_f64, |m, c| m.max(c.abs()));
    assert!(scale > 0.0, "{what}: reference is identically zero");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (a[k] - e[k]).abs() <= rel * scale,
                "{what}[{i}][{k}]: {} vs {}",
                a[k],
                e[k]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Neighbour-list sums versus the all-pairs reference
// ---------------------------------------------------------------------------

#[test]
fn density_and_pressure_steps_equal_the_all_pairs_sums() {
    let mut sim = cloud_sim(220, 0.5, 0.1, 0.0, 3);
    sim.density_step();
    let rho = reference_densities(&sim);
    for (i, (&actual, &expected)) in sim.particles.densities.iter().zip(rho.iter()).enumerate() {
        assert!(close(actual, expected, 1.0e-12, 0.0), "rho[{i}]");
    }
    sim.pressure_step();
    let p = reference_pressures(&sim, &rho);
    for (i, (&actual, &expected)) in sim.particles.pressures.iter().zip(p.iter()).enumerate() {
        assert!(close(actual, expected, 1.0e-10, 1.0e-9), "p[{i}]");
    }
}

#[test]
fn force_step_without_surface_tension_equals_the_all_pairs_reference() {
    let mut sim = cloud_sim(220, 0.5, 0.1, 0.0, 5);
    evaluate(&mut sim);
    let rho = reference_densities(&sim);
    let p = reference_pressures(&sim, &rho);
    let expected = reference_accelerations(&sim, &rho, &p);
    assert_vectors_close(&sim.particles.accelerations, &expected, 1.0e-9, "a");
    // Boundary particles are never accelerated.
    for (a, &boundary) in sim
        .particles
        .accelerations
        .iter()
        .zip(sim.particles.is_boundary.iter())
    {
        if boundary {
            assert_eq!(*a, [0.0; 3]);
        }
    }
}

/// Acceleration difference `a(σ) − a(0)` of two otherwise identical simulations:
/// the surface force alone, free of the (much larger) pressure acceleration.
fn surface_force_of(with_tension: &WcSphSim, without_tension: &WcSphSim) -> Vec<[f64; 3]> {
    with_tension
        .particles
        .accelerations
        .iter()
        .zip(without_tension.particles.accelerations.iter())
        .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        .collect()
}

#[test]
fn force_step_with_surface_tension_equals_the_all_pairs_reference() {
    let mut with_tension = cloud_sim(220, 0.5, 0.1, SIGMA, 7);
    let mut without_tension = cloud_sim(220, 0.5, 0.1, 0.0, 7);
    evaluate(&mut with_tension);
    evaluate(&mut without_tension);
    let rho = reference_densities(&with_tension);
    let p = reference_pressures(&with_tension, &rho);

    // The surface force on its own.
    let (reference_surface, _) = reference_surface_force(&with_tension, &rho);
    let surface_scale = reference_surface
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0_f64, |m, c| m.max(c.abs()));
    assert!(
        surface_scale > 1.0e-6,
        "the random cloud has a free surface, so the surface force cannot vanish"
    );
    assert_vectors_close(
        &surface_force_of(&with_tension, &without_tension),
        &reference_surface,
        1.0e-7,
        "surface force",
    );

    // The complete acceleration: gravity, pressure, viscosity and surface force.
    let expected = reference_accelerations(&with_tension, &rho, &p);
    assert_vectors_close(
        &with_tension.particles.accelerations,
        &expected,
        1.0e-9,
        "a",
    );
}

#[test]
fn surface_force_is_linear_in_sigma_and_absent_without_it() {
    let acceleration = |sigma: f64| -> Vec<[f64; 3]> {
        let mut sim = cloud_sim(160, 0.5, 0.1, sigma, 11);
        evaluate(&mut sim);
        sim.particles.accelerations.clone()
    };
    let base = acceleration(0.0);
    let single = acceleration(SIGMA);
    let double = acceleration(2.0 * SIGMA);
    let mut largest = 0.0_f64;
    for ((b, s), d) in base.iter().zip(single.iter()).zip(double.iter()) {
        for k in 0..3 {
            let first = s[k] - b[k];
            let second = d[k] - b[k];
            largest = largest.max(first.abs());
            assert!(
                (second - 2.0 * first).abs() <= 1.0e-9 * (1.0 + first.abs()),
                "surface force is not linear in sigma"
            );
        }
    }
    assert!(largest > 1.0e-6, "the surface force is identically zero");
}

#[test]
fn force_step_with_shared_lists_equals_the_self_contained_force_step() {
    let mut own = cloud_sim(180, 0.5, 0.1, SIGMA, 13);
    let mut shared = own.clone();
    evaluate(&mut own);

    let lists = shared.neighbor_lists();
    shared.density_step_with_neighbors(&lists);
    shared.pressure_step();
    shared.force_step_with_neighbors(&lists);

    assert_eq!(own.particles.densities, shared.particles.densities);
    assert_eq!(own.particles.pressures, shared.particles.pressures);
    assert_eq!(own.particles.accelerations, shared.particles.accelerations);
}

// ---------------------------------------------------------------------------
// Full step
// ---------------------------------------------------------------------------

#[test]
fn step_integrates_the_all_pairs_reference_accelerations() {
    let dt = 1.0e-4;
    let mut sim = cloud_sim(200, 0.5, 0.1, SIGMA, 17);
    let before = sim.particles.clone();
    let rho = reference_densities(&sim);
    let p = reference_pressures(&sim, &rho);
    let acc = reference_accelerations(&sim, &rho, &p);

    sim.step(dt);

    assert!(close(sim.time, dt, 1.0e-15, 0.0));
    assert_eq!(sim.kinetic_energy_history.len(), 1);
    assert_eq!(sim.potential_energy_history.len(), 1);
    for (i, a) in acc.iter().enumerate() {
        if before.is_boundary[i] {
            assert_eq!(sim.particles.positions[i], before.positions[i]);
            assert_eq!(sim.particles.velocities[i], before.velocities[i]);
            continue;
        }
        let v = [0, 1, 2].map(|k| before.velocities[i][k] + a[k] * dt);
        let x = [0, 1, 2].map(|k| before.positions[i][k] + v[k] * dt);
        for (k, (actual, expected)) in sim.particles.velocities[i].iter().zip(v.iter()).enumerate()
        {
            assert!(close(*actual, *expected, 1.0e-9, 1.0e-12), "v[{i}][{k}]");
        }
        for (k, (actual, expected)) in sim.particles.positions[i].iter().zip(x.iter()).enumerate() {
            assert!(close(*actual, *expected, 1.0e-9, 1.0e-12), "x[{i}][{k}]");
        }
    }
}

// ---------------------------------------------------------------------------
// Droplet
// ---------------------------------------------------------------------------

#[test]
fn surface_force_pulls_a_droplet_surface_inwards_and_leaves_its_bulk_alone() {
    let dx = 0.01;
    let cells = 8;
    let radius = cells as f64 * dx;
    let mut sim = droplet_sim(cells, dx, SIGMA);
    let mut plain = droplet_sim(cells, dx, 0.0);
    evaluate(&mut sim);
    evaluate(&mut plain);

    let rho = reference_densities(&sim);
    let (reference_force, curvature) = reference_surface_force(&sim, &rho);

    // The production surface force equals the all-pairs reference, and so does
    // the complete acceleration (the Tait pressure of the free-surface density
    // deficit dominates it).
    assert_vectors_close(
        &surface_force_of(&sim, &plain),
        &reference_force,
        1.0e-7,
        "surface force",
    );
    let p = reference_pressures(&sim, &rho);
    let expected = reference_accelerations(&sim, &rho, &p);
    assert_vectors_close(&sim.particles.accelerations, &expected, 1.0e-9, "a");

    let mut shell = 0_usize;
    let mut shell_inward = 0_usize;
    let mut shell_curvature = 0.0_f64;
    for (i, position) in sim.particles.positions.iter().enumerate() {
        let r = (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt();
        let radial_force = reference_force[i]
            .iter()
            .zip(position.iter())
            .map(|(f, x)| f * x)
            .sum::<f64>()
            / r.max(1e-30);
        if r <= 2.5 * dx {
            // Far from the interface the colour field is uniform: no force.
            assert!(
                reference_force[i].iter().all(|c| c.abs() < 1.0e-9),
                "bulk particle at r = {r} feels the surface force {:?}",
                reference_force[i]
            );
        }
        if r >= radius - 1.5 * dx {
            shell += 1;
            shell_curvature += curvature[i];
            if radial_force < 0.0 {
                shell_inward += 1;
            }
        }
    }
    assert!(shell > 400, "shell has {shell} particles");
    assert!(
        shell_inward as f64 >= 0.9 * shell as f64,
        "only {shell_inward} of {shell} surface particles are pulled inwards"
    );
    // The mean SPH curvature of the interface layer is the sphere's 2 / R.
    let mean_curvature = shell_curvature / shell as f64;
    assert!(
        (mean_curvature - 2.0 / radius).abs() < 0.15 * (2.0 / radius),
        "mean curvature {mean_curvature} vs 2/R = {}",
        2.0 / radius
    );
}
