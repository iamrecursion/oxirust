// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the neighbour-grid SPH kernels (`sph_compute_density`,
//! `sph_surface_normals`, `sph_compute_forces`, `sph_step`,
//! `gpu_sph_density_parallel`).
//!
//! They used to be O(n²) double loops over all particle pairs, with the square
//! root taken before any cutoff, and `gpu_sph_density_parallel` was a serial
//! copy of the density loop.  The references below are those double loops; the
//! grid kernels must reproduce them.

use super::*;

/// Deterministic uniform samples in `[0, 1)` (splitmix64).
struct Samples(u64);

impl Samples {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn config(surface_tension: f64) -> SphConfig {
    SphConfig {
        h: 0.12,
        rest_density: 1000.0,
        pressure_k: 100.0,
        viscosity: 0.5,
        surface_tension,
        gravity: [0.0, -9.81, 0.0],
        dt: 1.0e-3,
    }
}

/// `n` particles of mass 1 in a unit cube shifted by `shift`, with random
/// velocities, densities and pressures.
fn random_particles(n: usize, seed: u64, shift: [f64; 3]) -> Vec<SphParticle> {
    let mut samples = Samples(seed);
    (0..n)
        .map(|_| {
            let x = samples.next_f64();
            let y = samples.next_f64();
            let z = samples.next_f64();
            let mut p = SphParticle::new([x + shift[0], y + shift[1], z + shift[2]], 1.0);
            p.velocity = [
                samples.next_f64() - 0.5,
                samples.next_f64() - 0.5,
                samples.next_f64() - 0.5,
            ];
            p.density = 800.0 + 400.0 * samples.next_f64();
            p.pressure = 500.0 * samples.next_f64();
            p
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let scale = expected.abs().max(actual.abs()).max(1.0e-300);
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "{what}: {actual} vs {expected}"
    );
}

fn assert_vectors_close(actual: &[[f64; 3]], expected: &[[f64; 3]], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        for (axis, (&got, &want)) in a.iter().zip(e).enumerate() {
            assert_close(got, want, 1.0e-12, &format!("{what}[{i}][{axis}]"));
        }
    }
}

// ── O(n²) references ──────────────────────────────────────────────────────

fn brute_densities(particles: &[SphParticle], config: &SphConfig) -> Vec<f64> {
    let n = particles.len();
    let mut densities = vec![0.0f64; n];
    for i in 0..n {
        let mut rho = 0.0;
        for j in 0..n {
            let r_vec = sub3(particles[i].position, particles[j].position);
            let r = length3(r_vec);
            rho += particles[j].mass * SphKernels::poly6(r, config.h);
        }
        densities[i] = rho.max(1e-6);
    }
    densities
}

fn brute_normals(particles: &[SphParticle], config: &SphConfig) -> Vec<[f64; 3]> {
    let mut normals = Vec::with_capacity(particles.len());
    for (i, p_i) in particles.iter().enumerate() {
        let mut colour_gradient = [0.0f64; 3];
        for (j, p_j) in particles.iter().enumerate() {
            if i == j || p_j.density <= 0.0 {
                continue;
            }
            let r_vec = sub3(p_i.position, p_j.position);
            let r = length3(r_vec);
            if r > config.h || r < 1e-15 {
                continue;
            }
            let grad = SphKernels::poly6_grad(r_vec, r, config.h);
            colour_gradient = add3(colour_gradient, scale3(grad, p_j.mass / p_j.density));
        }
        normals.push(scale3(colour_gradient, -config.h));
    }
    normals
}

/// The force densities, with the same terms as `sph_compute_forces`.
fn brute_forces(particles: &[SphParticle], config: &SphConfig) -> Vec<[f64; 3]> {
    let n = particles.len();
    let normals = if config.surface_tension > 0.0 {
        brute_normals(particles, config)
    } else {
        Vec::new()
    };
    let mut forces = vec![[0.0f64; 3]; n];
    for i in 0..n {
        let mut a_pressure = [0.0f64; 3];
        let mut a_tension = [0.0f64; 3];
        let mut f_viscosity = [0.0f64; 3];
        for j in 0..n {
            if i == j {
                continue;
            }
            let r_vec = sub3(particles[i].position, particles[j].position);
            let r = length3(r_vec);
            if r > config.h || r < 1e-15 {
                continue;
            }
            let pressure_factor = particles[i].pressure
                / (particles[i].density * particles[i].density)
                + particles[j].pressure / (particles[j].density * particles[j].density);
            let grad = SphKernels::spiky_grad(r_vec, r, config.h);
            a_pressure = add3(
                a_pressure,
                scale3(grad, -particles[j].mass * pressure_factor),
            );
            let v_diff = sub3(particles[j].velocity, particles[i].velocity);
            let lap = SphKernels::viscosity_laplacian(r, config.h);
            let visc_factor = config.viscosity * particles[j].mass * lap / particles[j].density;
            f_viscosity = add3(f_viscosity, scale3(v_diff, visc_factor));
            if let (Some(&normal_i), Some(&normal_j)) = (normals.get(i), normals.get(j)) {
                a_tension = add3(
                    a_tension,
                    surface_tension_acceleration(
                        &particles[i],
                        &particles[j],
                        normal_i,
                        normal_j,
                        r_vec,
                        r,
                        config,
                    ),
                );
            }
        }
        let density_i = particles[i].density;
        let f_pressure_and_tension = scale3(add3(a_pressure, a_tension), density_i);
        let f_gravity = scale3(config.gravity, density_i);
        forces[i] = add3(add3(f_pressure_and_tension, f_viscosity), f_gravity);
    }
    forces
}

// ── density ───────────────────────────────────────────────────────────────

#[test]
fn density_matches_the_brute_force_sum_on_1000_random_particles() {
    let cfg = config(0.0);
    let mut particles = random_particles(1000, 31, [0.0; 3]);
    let expected = brute_densities(&particles, &cfg);
    sph_compute_density(&mut particles, &cfg);
    for (i, (p, &want)) in particles.iter().zip(&expected).enumerate() {
        assert_close(p.density, want, 1.0e-12, &format!("density[{i}]"));
    }
    // The cloud is dense enough that the check is not decided by the density floor.
    assert!(particles.iter().filter(|p| p.density > 1.0).count() > 900);
}

#[test]
fn gpu_density_kernel_is_the_same_computation_as_sph_compute_density() {
    let cfg = config(0.0);
    let mut reference = random_particles(600, 32, [0.0; 3]);
    let mut parallel = reference.clone();
    sph_compute_density(&mut reference, &cfg);
    gpu_sph_density_parallel(&mut parallel, &cfg);
    let expected = brute_densities(&parallel, &cfg);
    for i in 0..reference.len() {
        assert_eq!(parallel[i].density, reference[i].density, "particle {i}");
        assert_close(
            parallel[i].density,
            expected[i],
            1.0e-12,
            "against brute force",
        );
    }
}

#[test]
fn density_is_exact_for_clouds_far_from_the_origin_and_with_negative_coordinates() {
    let cfg = config(0.0);
    for shift in [
        [-1500.0, 40.0, -3.0e5],
        [-0.5, -0.5, -0.5],
        [2.0e4, 0.0, 0.0],
    ] {
        let mut particles = random_particles(500, 33, shift);
        let expected = brute_densities(&particles, &cfg);
        sph_compute_density(&mut particles, &cfg);
        for (i, (p, &want)) in particles.iter().zip(&expected).enumerate() {
            assert_close(
                p.density,
                want,
                1.0e-12,
                &format!("shift {shift:?}, density[{i}]"),
            );
        }
    }
}

#[test]
fn density_of_widely_separated_particles_is_the_self_contribution() {
    // Radius 1e-3 over a 1e6-wide box: the grid must stay bounded in size.
    let cfg = SphConfig {
        h: 1.0e-3,
        ..config(0.0)
    };
    let mut samples = Samples(34);
    let mut particles: Vec<SphParticle> = (0..300)
        .map(|_| {
            SphParticle::new(
                [
                    1.0e6 * samples.next_f64(),
                    1.0e6 * samples.next_f64(),
                    1.0e6 * samples.next_f64(),
                ],
                2.0e-6,
            )
        })
        .collect();
    sph_compute_density(&mut particles, &cfg);
    let self_density = 2.0e-6 * SphKernels::poly6(0.0, cfg.h);
    assert!(self_density > 1.0);
    for (i, p) in particles.iter().enumerate() {
        assert_close(p.density, self_density, 1.0e-12, &format!("density[{i}]"));
    }
}

#[test]
fn density_of_a_large_cloud_agrees_with_brute_force_on_sampled_particles() {
    // 20 000 particles would need 4e8 pair evaluations with a double loop.
    let cfg = SphConfig {
        h: 0.04,
        ..config(0.0)
    };
    let mut particles = random_particles(20_000, 35, [0.0; 3]);
    for p in &mut particles {
        p.mass = 0.05;
    }
    let snapshot = particles.clone();
    sph_compute_density(&mut particles, &cfg);
    let mut samples = Samples(36);
    for _ in 0..40 {
        let i = (samples.next_f64() * snapshot.len() as f64) as usize;
        let mut rho = 0.0;
        for q in &snapshot {
            let r = length3(sub3(snapshot[i].position, q.position));
            rho += q.mass * SphKernels::poly6(r, cfg.h);
        }
        assert_close(
            particles[i].density,
            rho.max(1e-6),
            1.0e-12,
            &format!("density[{i}]"),
        );
    }
}

// ── surface normals and forces ────────────────────────────────────────────

#[test]
fn surface_normals_match_the_brute_force_sum() {
    let cfg = config(0.2);
    let particles = random_particles(800, 37, [0.0; 3]);
    let normals = sph_surface_normals(&particles, &cfg);
    assert_vectors_close(&normals, &brute_normals(&particles, &cfg), "normal");
    assert!(normals.iter().any(|n| n.iter().any(|&c| c.abs() > 1.0e-3)));
}

#[test]
fn forces_match_the_brute_force_sums_without_surface_tension() {
    let cfg = config(0.0);
    let mut particles = random_particles(1000, 38, [0.0; 3]);
    let expected = brute_forces(&particles, &cfg);
    sph_compute_forces(&mut particles, &cfg);
    let got: Vec<[f64; 3]> = particles.iter().map(|p| p.force).collect();
    assert_vectors_close(&got, &expected, "force");
    assert!(got.iter().any(|f| f.iter().any(|&c| c.abs() > 1.0e3)));
}

#[test]
fn forces_match_the_brute_force_sums_with_surface_tension() {
    let cfg = config(0.3);
    let mut particles = random_particles(1000, 39, [0.0; 3]);
    let expected = brute_forces(&particles, &cfg);
    let mut without = particles.clone();
    sph_compute_forces(&mut particles, &cfg);
    let got: Vec<[f64; 3]> = particles.iter().map(|p| p.force).collect();
    assert_vectors_close(&got, &expected, "force");
    // The surface-tension term is active: it changes the result.
    sph_compute_forces(&mut without, &config(0.0));
    let differs = particles
        .iter()
        .zip(&without)
        .any(|(a, b)| length3(sub3(a.force, b.force)) > 1.0);
    assert!(differs);
}

#[test]
fn forces_are_exact_for_clouds_far_from_the_origin() {
    let cfg = config(0.1);
    let mut particles = random_particles(500, 40, [-7000.0, 12.0, 3.0e4]);
    let expected = brute_forces(&particles, &cfg);
    sph_compute_forces(&mut particles, &cfg);
    let got: Vec<[f64; 3]> = particles.iter().map(|p| p.force).collect();
    assert_vectors_close(&got, &expected, "force");
}

// ── full step ─────────────────────────────────────────────────────────────

#[test]
fn step_matches_the_brute_force_pipeline() {
    let cfg = config(0.1);
    let mut particles = random_particles(700, 41, [0.0; 3]);
    let mut reference = particles.clone();

    // Reference pipeline with the O(n²) loops.
    let densities = brute_densities(&reference, &cfg);
    for (p, d) in reference.iter_mut().zip(densities) {
        p.density = d;
    }
    sph_compute_pressure(&mut reference, &cfg);
    let forces = brute_forces(&reference, &cfg);
    for (p, f) in reference.iter_mut().zip(forces) {
        p.force = f;
    }
    sph_integrate(&mut reference, &cfg);

    sph_step(&mut particles, &cfg);
    for (i, (got, want)) in particles.iter().zip(&reference).enumerate() {
        assert_close(got.density, want.density, 1.0e-12, &format!("density[{i}]"));
        for axis in 0..3 {
            assert_close(
                got.velocity[axis],
                want.velocity[axis],
                1.0e-11,
                &format!("velocity[{i}][{axis}]"),
            );
            assert_close(
                got.position[axis],
                want.position[axis],
                1.0e-11,
                &format!("position[{i}][{axis}]"),
            );
        }
    }
}

#[test]
fn kernels_accept_empty_and_single_particle_slices() {
    let cfg = config(0.1);
    let mut empty: Vec<SphParticle> = Vec::new();
    sph_compute_density(&mut empty, &cfg);
    sph_compute_forces(&mut empty, &cfg);
    sph_step(&mut empty, &cfg);
    gpu_sph_density_parallel(&mut empty, &cfg);
    assert!(sph_surface_normals(&empty, &cfg).is_empty());

    let mut single = vec![SphParticle::new([5.0, -2.0, 1.0], 1.0)];
    sph_compute_density(&mut single, &cfg);
    assert_close(
        single[0].density,
        SphKernels::poly6(0.0, cfg.h),
        1.0e-14,
        "self density",
    );
    sph_compute_pressure(&mut single, &cfg);
    sph_compute_forces(&mut single, &cfg);
    // Only gravity acts on a lone particle.
    let expected = scale3(cfg.gravity, single[0].density);
    assert_vectors_close(&[single[0].force], &[expected], "lone particle force");
    assert_eq!(sph_surface_normals(&single, &cfg), vec![[0.0; 3]]);
}
