// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Serial and rayon evaluation of the [`ParticleSet`] gather kernels agree bit
//! for bit, whatever the number of worker threads.
//!
//! Every kernel computes the value of a particle from that particle's neighbour
//! list alone, in list order, so no reduction order can differ between the two
//! evaluations.  The tests run each kernel with [`Execution::Serial`] and with
//! [`Execution::Parallel`] inside pools of one, three and four threads on a
//! random cloud of more particles than a task takes, and compare the bit
//! patterns of every output.

use rayon::ThreadPoolBuilder;

use super::density_diffusion::{HydrostaticReference, apply_delta_sph_diffusion};
use super::gather;
use super::*;
use crate::kernel::CubicSplineKernel;
use crate::neighbor::SpatialHash;
use crate::parallel::Execution;
use crate::particle::{ParticleSet, SphParticle};
use oxiphysics_core::math::Vec3;

/// Smoothing length of the cloud (m).
const H: f64 = 0.1;

/// Thread counts the parallel evaluation is repeated with.
const THREADS: [usize; 3] = [1, 3, 4];

/// Deterministic pseudo-random numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn params() -> WcsphParams {
    WcsphParams {
        rest_density: 1000.0,
        stiffness: 2.0e5,
        gamma: 7.0,
        viscosity: 1.0e-3,
        smoothing_length: H,
    }
}

/// 700 particles scattered over a cube of side 0.9 m (about 30 neighbours each)
/// with random velocities, with densities and Tait pressures filled in, and the
/// neighbour lists for the support radius `2h`.
fn random_cloud(seed: u64) -> (ParticleSet, Vec<Vec<usize>>) {
    let mut rng = Lcg(seed);
    let count = 700;
    let side = 0.9;
    let mass = 1000.0 * side * side * side / count as f64;
    let mut ps = ParticleSet::new();
    for _ in 0..count {
        let position = Vec3::new(rng.next() * side, rng.next() * side, rng.next() * side);
        let velocity = Vec3::new(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5);
        ps.add_particle(&SphParticle::new(position, velocity, mass));
    }
    let neighbors = SpatialHash::find_all_neighbors(&ps.positions, 2.0 * H);
    let kernel = CubicSplineKernel;
    compute_density(&mut ps, &neighbors, &kernel, H);
    // A few vanishing densities exercise the skip branches of the kernels.
    for i in (0..count).step_by(97) {
        ps.densities[i] = 0.0;
    }
    compute_pressure_tait(&mut ps, &params());
    (ps, neighbors)
}

fn assert_same_f64(serial: &[f64], parallel: &[f64], what: &str) {
    assert_eq!(serial.len(), parallel.len(), "{what}: length");
    for (i, (s, p)) in serial.iter().zip(parallel).enumerate() {
        assert_eq!(s.to_bits(), p.to_bits(), "{what}: element {i}: {s} vs {p}");
    }
}

fn assert_same_vec3(serial: &[Vec3], parallel: &[Vec3], what: &str) {
    assert_eq!(serial.len(), parallel.len(), "{what}: length");
    for (i, (s, p)) in serial.iter().zip(parallel).enumerate() {
        for k in 0..3 {
            assert_eq!(
                s[k].to_bits(),
                p[k].to_bits(),
                "{what}: element {i} component {k}: {s:?} vs {p:?}"
            );
        }
    }
}

/// Run `parallel` in pools of every thread count of [`THREADS`].
fn in_pools(mut parallel: impl FnMut() + Send) {
    for threads in THREADS {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(&mut parallel);
    }
}

#[test]
fn density_summation_is_bitwise_identical() {
    let (ps, neighbors) = random_cloud(1);
    let kernel = CubicSplineKernel;
    let mut serial = ps.clone();
    gather::density_summation(Execution::Serial, &mut serial, &neighbors, &kernel, H);
    assert!(serial.densities.iter().any(|&d| d > 0.0));
    in_pools(|| {
        let mut parallel = ps.clone();
        gather::density_summation(Execution::Parallel, &mut parallel, &neighbors, &kernel, H);
        assert_same_f64(&serial.densities, &parallel.densities, "density");
    });
}

#[test]
fn density_summation_of_a_short_neighbour_list_leaves_the_rest_untouched() {
    let (ps, neighbors) = random_cloud(2);
    let kernel = CubicSplineKernel;
    let short = &neighbors[..300];
    let mut serial = ps.clone();
    gather::density_summation(Execution::Serial, &mut serial, short, &kernel, H);
    in_pools(|| {
        let mut parallel = ps.clone();
        gather::density_summation(Execution::Parallel, &mut parallel, short, &kernel, H);
        assert_same_f64(&serial.densities, &parallel.densities, "density");
    });
    assert_same_f64(
        &serial.densities[300..],
        &ps.densities[300..],
        "particles without a list keep their density",
    );
}

#[test]
fn shepard_density_is_bitwise_identical() {
    let (ps, neighbors) = random_cloud(3);
    let kernel = CubicSplineKernel;
    let mut serial = ps.clone();
    gather::density_shepard(Execution::Serial, &mut serial, &neighbors, &kernel, H);
    in_pools(|| {
        let mut parallel = ps.clone();
        gather::density_shepard(Execution::Parallel, &mut parallel, &neighbors, &kernel, H);
        assert_same_f64(&serial.densities, &parallel.densities, "shepard density");
    });
}

#[test]
fn pressure_force_is_bitwise_identical() {
    let (ps, neighbors) = random_cloud(4);
    let kernel = CubicSplineKernel;
    let mut serial = ps.clone();
    gather::pressure_force(Execution::Serial, &mut serial, &neighbors, &kernel, H);
    assert!(serial.forces.iter().any(|f| f.norm() > 0.0));
    in_pools(|| {
        let mut parallel = ps.clone();
        gather::pressure_force(Execution::Parallel, &mut parallel, &neighbors, &kernel, H);
        assert_same_vec3(&serial.forces, &parallel.forces, "pressure force");
    });
}

#[test]
fn viscous_forces_are_bitwise_identical() {
    let (ps, neighbors) = random_cloud(5);
    let kernel = CubicSplineKernel;
    let mu: Vec<f64> = ps.densities.iter().map(|&rho| 1.0e-3 * rho).collect();

    let mut serial = ps.clone();
    gather::morris_viscosity(Execution::Serial, &mut serial, &neighbors, &kernel, H, &mu);
    let mut serial_artificial = ps.clone();
    gather::artificial_viscosity(
        Execution::Serial,
        &mut serial_artificial,
        &neighbors,
        &kernel,
        H,
        0.5,
        1.0,
        40.0,
    );
    assert!(serial.forces.iter().any(|f| f.norm() > 0.0));
    assert!(serial_artificial.forces.iter().any(|f| f.norm() > 0.0));
    in_pools(|| {
        let mut parallel = ps.clone();
        gather::morris_viscosity(
            Execution::Parallel,
            &mut parallel,
            &neighbors,
            &kernel,
            H,
            &mu,
        );
        assert_same_vec3(&serial.forces, &parallel.forces, "Morris viscosity");
        let mut parallel_artificial = ps.clone();
        gather::artificial_viscosity(
            Execution::Parallel,
            &mut parallel_artificial,
            &neighbors,
            &kernel,
            H,
            0.5,
            1.0,
            40.0,
        );
        assert_same_vec3(
            &serial_artificial.forces,
            &parallel_artificial.forces,
            "artificial viscosity",
        );
    });
}

#[test]
fn xsph_and_tensile_corrections_are_bitwise_identical() {
    let (ps, neighbors) = random_cloud(6);
    let kernel = CubicSplineKernel;
    let xsph = gather::xsph_velocities(Execution::Serial, &ps, &neighbors, &kernel, H, 0.5);
    let tensile =
        gather::tensile_instability(Execution::Serial, &ps, &neighbors, &kernel, H, 0.05, 0.2, 4);
    assert!(
        tensile.iter().any(|f| f.norm() > 0.0),
        "tension was exercised"
    );
    in_pools(|| {
        let parallel_xsph =
            gather::xsph_velocities(Execution::Parallel, &ps, &neighbors, &kernel, H, 0.5);
        assert_same_vec3(&xsph, &parallel_xsph, "XSPH velocities");
        let parallel_tensile = gather::tensile_instability(
            Execution::Parallel,
            &ps,
            &neighbors,
            &kernel,
            H,
            0.05,
            0.2,
            4,
        );
        assert_same_vec3(&tensile, &parallel_tensile, "tensile correction");
    });
}

#[test]
fn density_corrections_are_bitwise_identical() {
    let (ps, neighbors) = random_cloud(7);
    let kernel = CubicSplineKernel;
    let shepard =
        gather::shepard_corrected_densities(Execution::Serial, &ps, &neighbors, &kernel, H);
    let mls = gather::mls_correction_factors(Execution::Serial, &ps, &neighbors, &kernel, H);
    in_pools(|| {
        let parallel_shepard =
            gather::shepard_corrected_densities(Execution::Parallel, &ps, &neighbors, &kernel, H);
        assert_same_f64(&shepard, &parallel_shepard, "Shepard density");
        let parallel_mls =
            gather::mls_correction_factors(Execution::Parallel, &ps, &neighbors, &kernel, H);
        assert_same_f64(&mls, &parallel_mls, "MLS factors");
    });
}

#[test]
fn riemann_force_and_shifting_are_bitwise_identical() {
    let (ps, neighbors) = random_cloud(8);
    let kernel = CubicSplineKernel;
    let params = params();
    let mut serial = ps.clone();
    gather::riemann_pressure_force(
        Execution::Serial,
        &mut serial,
        &neighbors,
        &kernel,
        H,
        &params,
    );
    let shifts = gather::particle_shifts(Execution::Serial, &ps, &neighbors, &kernel, H, 0.05);
    assert!(serial.forces.iter().any(|f| f.norm() > 0.0));
    in_pools(|| {
        let mut parallel = ps.clone();
        gather::riemann_pressure_force(
            Execution::Parallel,
            &mut parallel,
            &neighbors,
            &kernel,
            H,
            &params,
        );
        assert_same_vec3(&serial.forces, &parallel.forces, "Riemann force");
        let parallel_shifts =
            gather::particle_shifts(Execution::Parallel, &ps, &neighbors, &kernel, H, 0.05);
        assert_same_vec3(&shifts, &parallel_shifts, "particle shifts");
    });
}

#[test]
fn delta_sph_diffusion_is_bitwise_identical() {
    let (ps, neighbors) = random_cloud(9);
    let kernel = CubicSplineKernel;
    let reference = HydrostaticReference::from_params(&params(), Vec3::new(0.0, 0.0, -9.81));
    for hydrostatic in [None, Some(&reference)] {
        let mut serial = ps.clone();
        apply_delta_sph_diffusion(
            Execution::Serial,
            &mut serial,
            &neighbors,
            &kernel,
            H,
            0.1,
            40.0,
            1.0e-4,
            hydrostatic,
        );
        assert!(
            serial
                .densities
                .iter()
                .zip(&ps.densities)
                .any(|(a, b)| a != b),
            "diffusion changed the densities"
        );
        in_pools(|| {
            let mut parallel = ps.clone();
            apply_delta_sph_diffusion(
                Execution::Parallel,
                &mut parallel,
                &neighbors,
                &kernel,
                H,
                0.1,
                40.0,
                1.0e-4,
                hydrostatic,
            );
            assert_same_f64(&serial.densities, &parallel.densities, "δ-SPH density");
        });
    }
}

#[test]
fn public_functions_use_the_rayon_pool_for_large_clouds_with_the_same_result() {
    // 700 particles is above the threshold at which the public entry points hand
    // the loop to the pool, so `compute_density` & co. run in parallel here.
    let (ps, neighbors) = random_cloud(10);
    assert!(Execution::for_count(ps.len()) == Execution::Parallel);
    let kernel = CubicSplineKernel;

    let mut expected = ps.clone();
    gather::density_summation(Execution::Serial, &mut expected, &neighbors, &kernel, H);
    let mut by_public_fn = ps.clone();
    compute_density(&mut by_public_fn, &neighbors, &kernel, H);
    assert_same_f64(
        &expected.densities,
        &by_public_fn.densities,
        "compute_density",
    );

    let mut expected = ps.clone();
    gather::pressure_force(Execution::Serial, &mut expected, &neighbors, &kernel, H);
    let mut by_public_fn = ps.clone();
    compute_pressure_force(&mut by_public_fn, &neighbors, &kernel, H);
    assert_same_vec3(
        &expected.forces,
        &by_public_fn.forces,
        "compute_pressure_force",
    );

    let mut expected = ps.clone();
    let mu = vec![2.0; ps.len()];
    gather::morris_viscosity(
        Execution::Serial,
        &mut expected,
        &neighbors,
        &kernel,
        H,
        &mu,
    );
    let mut by_public_fn = ps.clone();
    compute_laminar_viscosity(&mut by_public_fn, &neighbors, &kernel, H, 2.0);
    assert_same_vec3(
        &expected.forces,
        &by_public_fn.forces,
        "compute_laminar_viscosity",
    );

    let expected = gather::particle_shifts(Execution::Serial, &ps, &neighbors, &kernel, H, 0.05);
    let by_public_fn = compute_particle_shifting(&ps, &neighbors, &kernel, H, 0.05);
    assert_same_vec3(&expected, &by_public_fn, "compute_particle_shifting");
}

#[test]
fn a_full_step_is_bitwise_identical_to_the_serial_step() {
    let (ps, neighbors) = random_cloud(11);
    let kernel = CubicSplineKernel;
    let params = params();
    let gravity = Vec3::new(0.0, 0.0, -9.81);

    // The step functions run the public entry points, which use the rayon pool
    // for this cloud; a serial reference is assembled from the serial kernels in
    // the order of `step`.
    let serial_step = |ps: &mut ParticleSet| {
        gather::density_summation(Execution::Serial, ps, &neighbors, &kernel, H);
        compute_pressure_tait(ps, &params);
        ps.clear_forces();
        gather::pressure_force(Execution::Serial, ps, &neighbors, &kernel, H);
        let mu = vec![params.viscosity * params.rest_density; ps.len()];
        gather::morris_viscosity(Execution::Serial, ps, &neighbors, &kernel, H, &mu);
        let dt = 1.0e-4;
        for i in 0..ps.len() {
            let acc = ps.forces[i] / ps.densities[i].max(1e-14) + gravity;
            ps.velocities[i] += acc * dt;
            let velocity = ps.velocities[i];
            ps.positions[i] += velocity * dt;
        }
    };
    let mut serial = ps.clone();
    serial_step(&mut serial);
    in_pools(|| {
        let mut parallel = ps.clone();
        step(&mut parallel, &neighbors, &kernel, &params, 1.0e-4, gravity);
        assert_same_vec3(&serial.positions, &parallel.positions, "positions");
        assert_same_vec3(&serial.velocities, &parallel.velocities, "velocities");
        assert_same_vec3(&serial.forces, &parallel.forces, "forces");
        assert_same_f64(&serial.densities, &parallel.densities, "densities");
        assert_same_f64(&serial.pressures, &parallel.pressures, "pressures");
    });
}
