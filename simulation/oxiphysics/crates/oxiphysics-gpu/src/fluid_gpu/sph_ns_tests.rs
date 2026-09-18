// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the neighbour-grid SPH kernels of [`SPHGpuKernels`] (checked against
//! O(n²) references), of its Morris viscosity (checked against the analytic
//! Laplacian and decay of a shear wave), and of the viscous
//! [`NavierStokesGpu`] step.

use super::*;
use std::f64::consts::PI;

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

/// `n` random particles in a unit cube shifted by `shift`, with random
/// velocities, densities and pressures (so every kernel is exercised
/// independently of the others).
fn random_sph(n: usize, seed: u64, shift: [f64; 3]) -> SPHGpuKernels {
    let mut samples = Samples(seed);
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            let x = samples.next_f64();
            let y = samples.next_f64();
            let z = samples.next_f64();
            [x + shift[0], y + shift[1], z + shift[2]]
        })
        .collect();
    let mut sph = SPHGpuKernels::new(positions, 0.14, 1000.0, 50.0, 0.8, 1.0);
    for velocity in &mut sph.velocities {
        let vx = samples.next_f64();
        let vy = samples.next_f64();
        let vz = samples.next_f64();
        *velocity = [2.0 * vx - 1.0, 2.0 * vy - 1.0, 2.0 * vz - 1.0];
    }
    for density in &mut sph.densities {
        *density = 600.0 + 800.0 * samples.next_f64();
    }
    for pressure in &mut sph.pressures {
        *pressure = 2000.0 * samples.next_f64();
    }
    sph
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    let scale = expected.abs().max(actual.abs()).max(1.0e-300);
    assert!(
        (actual - expected).abs() <= tolerance * scale,
        "{what}: {actual} vs {expected}"
    );
}

fn assert_vectors_close(actual: &[[f64; 3]], expected: &[[f64; 3]], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        for (axis, (&got, &want)) in a.iter().zip(e).enumerate() {
            assert_close(got, want, tolerance, &format!("{what}[{i}][{axis}]"));
        }
    }
}

// ── O(n²) references (the loops the grid kernels replace) ─────────────────

fn brute_density(sph: &SPHGpuKernels) -> Vec<f64> {
    sph.positions
        .iter()
        .map(|&pos_i| {
            let mut rho = 0.0;
            for &pos_j in &sph.positions {
                let dx = pos_i[0] - pos_j[0];
                let dy = pos_i[1] - pos_j[1];
                let dz = pos_i[2] - pos_j[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                rho += sph.mass * sph.kernel_w(r);
            }
            rho.max(sph.rest_density * 0.01)
        })
        .collect()
}

fn brute_pressure_force(sph: &SPHGpuKernels) -> Vec<[f64; 3]> {
    let n = sph.n_particles();
    let mut forces = vec![[0.0_f64; 3]; n];
    for (i, force_i) in forces.iter_mut().enumerate() {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = sph.positions[i][0] - sph.positions[j][0];
            let dy = sph.positions[i][1] - sph.positions[j][1];
            let dz = sph.positions[i][2] - sph.positions[j][2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-12 {
                continue;
            }
            let grad = sph.kernel_grad_w([dx, dy, dz], r);
            let rhoi = sph.densities[i].max(1e-12);
            let rhoj = sph.densities[j].max(1e-12);
            let coeff =
                -sph.mass * (sph.pressures[i] / (rhoi * rhoi) + sph.pressures[j] / (rhoj * rhoj));
            for (force, g) in force_i.iter_mut().zip(grad) {
                *force += coeff * g;
            }
        }
    }
    forces
}

/// Morris et al. (1997) viscous acceleration over all pairs.
fn brute_viscosity_force(sph: &SPHGpuKernels) -> Vec<[f64; 3]> {
    let n = sph.n_particles();
    let h = sph.smoothing_length;
    let mut accelerations = vec![[0.0_f64; 3]; n];
    for (i, acceleration_i) in accelerations.iter_mut().enumerate() {
        for j in 0..n {
            if i == j {
                continue;
            }
            let r_ij = [
                sph.positions[i][0] - sph.positions[j][0],
                sph.positions[i][1] - sph.positions[j][1],
                sph.positions[i][2] - sph.positions[j][2],
            ];
            let r2 = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2];
            let r = r2.sqrt();
            if r >= h || r < 1e-12 {
                continue;
            }
            let grad = sph.kernel_grad_w(r_ij, r);
            let r_dot_grad = r_ij[0] * grad[0] + r_ij[1] * grad[1] + r_ij[2] * grad[2];
            let coeff = sph.mass * 2.0 * sph.viscosity
                / (sph.densities[i].max(1e-12) * sph.densities[j].max(1e-12))
                * r_dot_grad
                / (r2 + 0.01 * h * h);
            let relative = [
                sph.velocities[i][0] - sph.velocities[j][0],
                sph.velocities[i][1] - sph.velocities[j][1],
                sph.velocities[i][2] - sph.velocities[j][2],
            ];
            for (acceleration, dv) in acceleration_i.iter_mut().zip(relative) {
                *acceleration += coeff * dv;
            }
        }
    }
    accelerations
}

// ── neighbour-grid kernels versus the O(n²) references ────────────────────

#[test]
fn sph_density_matches_the_brute_force_sum_on_1000_random_particles() {
    let mut sph = random_sph(1000, 11, [0.0; 3]);
    let expected = brute_density(&sph);
    sph.compute_density();
    for (i, (&got, &want)) in sph.densities.iter().zip(&expected).enumerate() {
        assert_close(got, want, 1.0e-12, &format!("density[{i}]"));
    }
    // The cloud is dense enough that the test is not decided by the density floor.
    let floor = sph.rest_density * 0.01;
    assert!(sph.densities.iter().filter(|&&d| d > 2.0 * floor).count() > 900);
}

#[test]
fn sph_pressure_force_matches_the_brute_force_sum_on_1000_random_particles() {
    let sph = random_sph(1000, 12, [0.0; 3]);
    let forces = sph.pressure_force();
    assert_vectors_close(
        &forces,
        &brute_pressure_force(&sph),
        1.0e-12,
        "pressure force",
    );
    assert!(forces.iter().any(|f| f.iter().any(|&c| c.abs() > 1.0)));
}

#[test]
fn sph_viscosity_force_matches_the_morris_sum_on_1000_random_particles() {
    let sph = random_sph(1000, 13, [0.0; 3]);
    let forces = sph.viscosity_force();
    assert_vectors_close(
        &forces,
        &brute_viscosity_force(&sph),
        1.0e-12,
        "viscosity force",
    );
    assert!(forces.iter().any(|f| f.iter().any(|&c| c.abs() > 0.1)));
}

#[test]
fn sph_kernels_are_exact_for_clouds_far_from_the_origin() {
    let mut sph = random_sph(700, 14, [-1500.0, 40.0, -3.0e5]);
    let expected_density = brute_density(&sph);
    sph.compute_density();
    for (i, (&got, &want)) in sph.densities.iter().zip(&expected_density).enumerate() {
        assert_close(got, want, 1.0e-12, &format!("density[{i}]"));
    }
    assert_vectors_close(
        &sph.pressure_force(),
        &brute_pressure_force(&sph),
        1.0e-12,
        "pressure force",
    );
    assert_vectors_close(
        &sph.viscosity_force(),
        &brute_viscosity_force(&sph),
        1.0e-12,
        "viscosity force",
    );
}

#[test]
fn sph_kernels_follow_particles_that_move_between_calls() {
    // The grid is rebuilt per call: moving the (public) positions must change
    // the result exactly like a fresh brute-force evaluation.
    let mut sph = random_sph(300, 15, [0.0; 3]);
    sph.compute_density();
    for position in &mut sph.positions {
        position[0] += 0.37;
        position[2] -= 0.21;
    }
    let expected = brute_density(&sph);
    sph.compute_density();
    for (i, (&got, &want)) in sph.densities.iter().zip(&expected).enumerate() {
        assert_close(got, want, 1.0e-12, &format!("density[{i}]"));
    }
}

#[test]
fn sph_density_of_widely_separated_particles_is_the_self_term() {
    // Radius 1e-3 over a 1e6-wide box would need ~1e27 cells at the nominal
    // size; the grid stays bounded and every particle only sees itself.
    let mut samples = Samples(16);
    let positions: Vec<[f64; 3]> = (0..400)
        .map(|_| {
            [
                1.0e6 * samples.next_f64(),
                1.0e6 * samples.next_f64(),
                1.0e6 * samples.next_f64(),
            ]
        })
        .collect();
    let mut sph = SPHGpuKernels::new(positions, 1.0e-3, 1000.0, 1.0, 0.001, 1.0);
    sph.compute_density();
    let self_density = sph.mass * sph.kernel_w(0.0);
    assert!(self_density > 1000.0);
    for (i, &rho) in sph.densities.iter().enumerate() {
        assert_close(rho, self_density, 1.0e-12, &format!("density[{i}]"));
    }
    assert!(sph.pressure_force().iter().all(|f| *f == [0.0; 3]));
    assert!(sph.viscosity_force().iter().all(|f| *f == [0.0; 3]));
}

#[test]
fn sph_kernels_accept_empty_and_single_particle_clouds() {
    let mut empty = SPHGpuKernels::new(Vec::new(), 0.1, 1000.0, 1.0, 0.001, 1.0);
    empty.compute_density();
    assert!(empty.pressure_force().is_empty());
    assert!(empty.viscosity_force().is_empty());
    let mut single = SPHGpuKernels::new(vec![[3.0, 4.0, 5.0]], 0.1, 1000.0, 1.0, 0.001, 1.0);
    single.compute_density();
    assert_close(
        single.densities[0],
        single.mass * single.kernel_w(0.0),
        1.0e-14,
        "density",
    );
    assert_eq!(single.pressure_force(), vec![[0.0; 3]]);
    assert_eq!(single.viscosity_force(), vec![[0.0; 3]]);
}

// ── viscosity: structure of the Morris term ───────────────────────────────

#[test]
fn sph_viscosity_vanishes_for_a_uniform_velocity() {
    let mut sph = random_sph(500, 17, [0.0; 3]);
    for velocity in &mut sph.velocities {
        *velocity = [0.3, -1.2, 2.5];
    }
    assert!(sph.viscosity_force().iter().all(|a| *a == [0.0; 3]));
}

#[test]
fn sph_viscosity_conserves_momentum() {
    let sph = random_sph(800, 18, [0.0; 3]);
    let accelerations = sph.viscosity_force();
    for axis in 0..3 {
        let total: f64 = accelerations.iter().map(|a| a[axis]).sum();
        let magnitude: f64 = accelerations.iter().map(|a| a[axis].abs()).sum();
        assert!(magnitude > 1.0, "the term must be active");
        assert!(
            total.abs() <= 1.0e-12 * magnitude,
            "axis {axis}: net {total} of {magnitude}"
        );
    }
}

#[test]
fn sph_viscosity_only_removes_kinetic_energy() {
    let sph = random_sph(800, 19, [0.0; 3]);
    let accelerations = sph.viscosity_force();
    let power: f64 = sph
        .velocities
        .iter()
        .zip(&accelerations)
        .map(|(v, a)| sph.mass * (v[0] * a[0] + v[1] * a[1] + v[2] * a[2]))
        .sum();
    assert!(power < 0.0, "viscous power {power}");
}

#[test]
fn sph_viscosity_scales_with_dynamic_viscosity_and_inverse_density_squared() {
    let sph = random_sph(300, 20, [0.0; 3]);
    let base = sph.viscosity_force();

    let mut stiffer = sph.clone();
    stiffer.viscosity *= 2.5;
    let doubled: Vec<[f64; 3]> = base.iter().map(|a| a.map(|c| 2.5 * c)).collect();
    assert_vectors_close(&stiffer.viscosity_force(), &doubled, 1.0e-12, "2.5 mu");

    let mut denser = sph.clone();
    for density in &mut denser.densities {
        *density *= 2.0;
    }
    let quartered: Vec<[f64; 3]> = base.iter().map(|a| a.map(|c| 0.25 * c)).collect();
    assert_vectors_close(&denser.viscosity_force(), &quartered, 1.0e-12, "2 rho");
}

// ── viscosity: analytic shear wave ────────────────────────────────────────

const WAVE_DX: f64 = 0.05;
const WAVE_NX: usize = 9;
const WAVE_NY: usize = 40;
const WAVE_NZ: usize = 9;
const WAVE_RHO: f64 = 1000.0;
const WAVE_MU: f64 = 1.0;
/// Wavelength of the shear wave in lattice spacings.
const WAVE_CELLS: f64 = 10.0;

fn wave_number() -> f64 {
    2.0 * PI / (WAVE_CELLS * WAVE_DX)
}

fn wave_index(ix: usize, iy: usize, iz: usize) -> usize {
    ix + WAVE_NX * (iy + WAVE_NY * iz)
}

/// A lattice of particles with `v_x = sin(k y)`, smoothing length `2 dx`.
fn shear_wave_lattice() -> SPHGpuKernels {
    let mut positions = Vec::new();
    for iz in 0..WAVE_NZ {
        for iy in 0..WAVE_NY {
            for ix in 0..WAVE_NX {
                positions.push([
                    ix as f64 * WAVE_DX,
                    iy as f64 * WAVE_DX,
                    iz as f64 * WAVE_DX,
                ]);
            }
        }
    }
    let mass = WAVE_RHO * WAVE_DX.powi(3);
    let mut sph = SPHGpuKernels::new(positions, 2.0 * WAVE_DX, WAVE_RHO, 1.0, WAVE_MU, mass);
    let k = wave_number();
    for (velocity, position) in sph.velocities.iter_mut().zip(&sph.positions) {
        *velocity = [(k * position[1]).sin(), 0.0, 0.0];
    }
    sph.compute_density();
    sph
}

/// Wave amplitude `A` fitted to `v_x = A sin(k y)` on the central column of the
/// lattice, away from every boundary (where the kernel support is truncated).
fn central_amplitude(sph: &SPHGpuKernels) -> f64 {
    let k = wave_number();
    let mut cross = 0.0;
    let mut norm = 0.0;
    for iy in 12..28 {
        let i = wave_index(WAVE_NX / 2, iy, WAVE_NZ / 2);
        let s = (k * sph.positions[i][1]).sin();
        cross += sph.velocities[i][0] * s;
        norm += s * s;
    }
    cross / norm
}

#[test]
fn sph_viscosity_of_a_shear_wave_is_nu_times_the_laplacian() {
    let sph = shear_wave_lattice();
    let k = wave_number();
    let nu = WAVE_MU / WAVE_RHO;
    // The interior densities are the rest density up to the lattice-sum error.
    let i_centre = wave_index(WAVE_NX / 2, WAVE_NY / 2, WAVE_NZ / 2);
    assert_close(
        sph.densities[i_centre],
        WAVE_RHO,
        1.0e-3,
        "interior density",
    );
    let accelerations = sph.viscosity_force();
    let mut checked = 0;
    for iy in 12..28 {
        let i = wave_index(WAVE_NX / 2, iy, WAVE_NZ / 2);
        let vx = sph.velocities[i][0];
        if vx.abs() < 0.3 {
            continue;
        }
        let expected = -nu * k * k * vx;
        let ratio = accelerations[i][0] / expected;
        assert!(
            (ratio - 1.0).abs() < 0.08,
            "iy {iy}: a = {}, -nu k^2 v = {expected}",
            accelerations[i][0]
        );
        // The wave is purely along x: the other components vanish.
        assert!(accelerations[i][1].abs() < 1.0e-9 * expected.abs());
        assert!(accelerations[i][2].abs() < 1.0e-9 * expected.abs());
        checked += 1;
    }
    assert!(checked >= 6, "only {checked} particles checked");
}

#[test]
fn sph_shear_wave_decays_like_exp_minus_nu_k_squared_t() {
    let mut sph = shear_wave_lattice();
    let k = wave_number();
    let nu = WAVE_MU / WAVE_RHO;
    let dt = 0.1;
    let steps = 30;
    let initial = central_amplitude(&sph);
    assert_close(initial, 1.0, 1.0e-12, "initial amplitude");
    for _ in 0..steps {
        let accelerations = sph.viscosity_force();
        for (velocity, acceleration) in sph.velocities.iter_mut().zip(&accelerations) {
            for (v, a) in velocity.iter_mut().zip(acceleration) {
                *v += dt * a;
            }
        }
    }
    let t = dt * steps as f64;
    let expected = (-nu * k * k * t).exp();
    let decayed = central_amplitude(&sph) / initial;
    assert!(expected < 0.7, "the wave must decay noticeably: {expected}");
    assert!(
        (decayed / expected - 1.0).abs() < 0.10,
        "amplitude {decayed} vs exp(-nu k^2 t) = {expected}"
    );
}

// ── Navier-Stokes: implicit viscous diffusion ─────────────────────────────

#[test]
fn ns_kinematic_viscosity_is_dynamic_viscosity_over_density() {
    let mut ns = NavierStokesGpu::new(4, 4, 4, 0.1, 0.002);
    assert_close(
        ns.kinematic_viscosity(),
        2.0e-6,
        1.0e-12,
        "water-like default",
    );
    ns.rest_density = 500.0;
    assert_close(
        ns.kinematic_viscosity(),
        4.0e-6,
        1.0e-12,
        "rest density 500",
    );
}

const NS_NX: usize = 5;
const NS_NY: usize = 33;
const NS_NZ: usize = 5;

/// Sets `u_x = amplitude * sin(pi y / L)` on every cell with `y_index`, or only on
/// the cells touching the x and z faces when `faces_only` is set.
fn set_sine_shear(ns: &mut NavierStokesGpu, amplitude: f64, faces_only: bool) {
    for z in 0..NS_NZ {
        for y in 0..NS_NY {
            for x in 0..NS_NX {
                let on_face = x == 0 || x == NS_NX - 1 || z == 0 || z == NS_NZ - 1;
                if faces_only && !on_face {
                    continue;
                }
                let index = ns.velocity.idx(x, y, z, 0);
                ns.velocity.front[index] = amplitude * (PI * y as f64 / (NS_NY - 1) as f64).sin();
            }
        }
    }
}

#[test]
fn ns_diffuse_decays_a_sine_shear_like_exp_minus_nu_k_squared_t() {
    // u_x = sin(pi y / L) vanishes on the y faces, which the solver holds fixed.
    // The field does not depend on x or z, so the x and z faces are held at the
    // solution of the discrete problem (a time-dependent Dirichlet condition).
    let dx = 1.0 / (NS_NY - 1) as f64;
    let nu = 0.05;
    let mut ns = NavierStokesGpu::new(NS_NX, NS_NY, NS_NZ, dx, nu * 1000.0);
    assert_close(ns.kinematic_viscosity(), nu, 1.0e-14, "nu");
    set_sine_shear(&mut ns, 1.0, false);
    ns.velocity.copy_front_to_back();

    let dt = 0.01;
    let steps = 100;
    let k = PI;
    let lambda_h = (2.0 - 2.0 * (k * dx).cos()) / (dx * dx);
    let mut discrete_amplitude = 1.0;
    for _ in 0..steps {
        // Backward Euler damps the (discrete) eigenmode by 1 / (1 + nu dt lambda).
        discrete_amplitude /= 1.0 + nu * dt * lambda_h;
        set_sine_shear(&mut ns, discrete_amplitude, true);
        ns.diffuse(dt, 80);
    }
    let t = dt * steps as f64;
    let centre = ns.velocity.read(NS_NX / 2, NS_NY / 2, NS_NZ / 2, 0);
    let exact = (-nu * k * k * t).exp();
    assert!(exact < 0.65 && exact > 0.55, "exact decay {exact}");
    // Within 10 % of the continuous decay (here: far better) ...
    assert!(
        (centre / exact - 1.0).abs() < 0.02,
        "u_x(centre) = {centre}, exp(-nu k^2 t) = {exact}"
    );
    // ... and equal to the backward-Euler recurrence of the discrete mode, which
    // shows the Jacobi sweeps have converged.
    assert!(
        (centre - discrete_amplitude).abs() < 1.0e-6,
        "u_x(centre) = {centre}, backward Euler {discrete_amplitude}"
    );
    // The other components stay zero.
    let index = ns.velocity.idx(NS_NX / 2, NS_NY / 2, NS_NZ / 2, 1);
    assert_eq!(ns.velocity.front[index], 0.0);
}

#[test]
fn ns_diffuse_without_viscosity_leaves_the_field_unchanged() {
    let mut ns = NavierStokesGpu::new(NS_NX, NS_NY, NS_NZ, 0.03, 0.0);
    set_sine_shear(&mut ns, 1.0, false);
    let before = ns.velocity.front.clone();
    ns.diffuse(0.01, 20);
    assert_eq!(ns.velocity.front, before);
    let mut viscous = NavierStokesGpu::new(NS_NX, NS_NY, NS_NZ, 0.03, 1.0);
    set_sine_shear(&mut viscous, 1.0, false);
    let before = viscous.velocity.front.clone();
    viscous.diffuse(0.01, 0);
    viscous.diffuse(0.0, 20);
    viscous.diffuse(f64::NAN, 20);
    assert_eq!(viscous.velocity.front, before);
}

#[test]
fn ns_diffuse_keeps_a_uniform_field_and_the_boundary_cells() {
    let mut ns = NavierStokesGpu::new(6, 7, 8, 0.1, 5.0);
    ns.velocity.fill(0.0);
    for cell in ns.velocity.front.chunks_mut(3) {
        cell.copy_from_slice(&[0.5, -1.0, 2.0]);
    }
    ns.diffuse(0.1, 10);
    for cell in ns.velocity.front.chunks(3) {
        for (value, expected) in cell.iter().zip([0.5, -1.0, 2.0]) {
            assert!((value - expected).abs() < 1.0e-12);
        }
    }
}

#[test]
fn ns_diffuse_spreads_a_spike_and_conserves_its_total() {
    let n = 11;
    let mut ns = NavierStokesGpu::new(n, n, n, 0.1, 100.0);
    let centre = ns.velocity.idx(n / 2, n / 2, n / 2, 2);
    ns.velocity.front[centre] = 1.0;
    ns.diffuse(0.002, 400);
    let spike = ns.velocity.front[centre];
    assert!(spike < 1.0 && spike > 0.0, "peak {spike}");
    let neighbour = ns.velocity.idx(n / 2 + 1, n / 2, n / 2, 2);
    assert!(ns.velocity.front[neighbour] > 0.0);
    // The implicit diffusion operator conserves the sum while the blob stays
    // away from the boundary (here ν dt / dx² = 0.02 per step is tiny).
    let total: f64 = ns.velocity.front.iter().skip(2).step_by(3).sum();
    assert!((total - 1.0).abs() < 1.0e-6, "total {total}");
    // The other components are untouched.
    assert!(ns.velocity.front.iter().step_by(3).all(|&v| v == 0.0));
}

#[test]
fn ns_step_damps_a_viscous_flow_more_than_an_inviscid_one() {
    let make = |viscosity: f64| {
        let mut ns = NavierStokesGpu::new(12, 12, 12, 0.1, viscosity);
        let mut samples = Samples(21);
        for z in 1..11 {
            for y in 1..11 {
                for x in 1..11 {
                    for c in 0..3 {
                        let index = ns.velocity.idx(x, y, z, c);
                        ns.velocity.front[index] = samples.next_f64() - 0.5;
                    }
                }
            }
        }
        ns
    };
    let energy = |ns: &NavierStokesGpu| ns.velocity.front.iter().map(|v| v * v).sum::<f64>();
    let mut inviscid = make(0.0);
    let mut viscous = make(20.0);
    inviscid.step(0.01, 20, 20);
    viscous.step(0.01, 20, 20);
    assert!(energy(&viscous) < energy(&inviscid) * 0.95);
}

#[test]
fn pipeline_default_ns_lists_the_viscosity_pass_after_advection() {
    let pipeline = FluidGpuPipeline::default_ns(1024);
    let labels: Vec<&str> = pipeline.passes.iter().map(|p| p.label.as_str()).collect();
    let position = |label: &str| {
        labels
            .iter()
            .position(|&l| l == label)
            .unwrap_or_else(|| panic!("pass {label} missing in {labels:?}"))
    };
    let advect = position("advect_density");
    let viscosity = position("viscosity_jacobi");
    let divergence = position("divergence");
    assert!(advect < viscosity && viscosity < divergence, "{labels:?}");
    assert_eq!(
        pipeline.total_bytes,
        pipeline
            .passes
            .iter()
            .map(|p| p.estimated_bytes)
            .sum::<usize>()
    );
}
