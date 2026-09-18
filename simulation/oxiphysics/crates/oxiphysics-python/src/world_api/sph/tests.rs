// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the neighbour-search based SPH step.
//!
//! The step used to loop over all `N²` particle pairs in both the density and
//! the force pass. These tests keep the former all-pairs implementation as a
//! reference and check that the spatial-hash step reproduces it, that the step
//! visits a sparse set of pairs only, and that particles far from each other
//! (or far from the origin) are handled.

use super::*;

/// xorshift64* pseudo-random generator (the crate has no `rand` dependency).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform sample in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        bits as f64 / (1u64 << 53) as f64
    }

    /// Uniform sample in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_unit()
    }
}

/// State of the reference simulation.
#[derive(Clone)]
struct AllPairs {
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
    densities: Vec<f64>,
    pressures: Vec<f64>,
}

impl AllPairs {
    /// The former `PySphSim::step`: densities and forces from `N²` pair loops.
    fn step(&mut self, config: &PySphConfig, dt: f64) {
        let n = self.positions.len();
        let h = config.h;
        let h2 = h * h;
        let m = config.particle_mass;
        let rho0 = config.rest_density;
        let k = config.stiffness;
        let mu = config.viscosity;
        let g = config.gravity;

        let poly6_coeff = 315.0 / (64.0 * std::f64::consts::PI * h.powi(9));
        for i in 0..n {
            let mut rho = 0.0f64;
            for j in 0..n {
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                let dz = self.positions[i][2] - self.positions[j][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 < h2 {
                    let diff = h2 - r2;
                    rho += m * poly6_coeff * diff * diff * diff;
                }
            }
            self.densities[i] = rho.max(1e-3);
            self.pressures[i] = k * (self.densities[i] - rho0);
        }

        let spiky_coeff = -45.0 / (std::f64::consts::PI * h.powi(6));
        let visc_coeff = 45.0 / (std::f64::consts::PI * h.powi(6));
        let mut forces = vec![[0.0f64; 3]; n];
        for (i, force_i) in forces.iter_mut().enumerate() {
            let mut fp = [0.0f64; 3];
            let mut fv = [0.0f64; 3];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = self.positions[i][0] - self.positions[j][0];
                let dy = self.positions[i][1] - self.positions[j][1];
                let dz = self.positions[i][2] - self.positions[j][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= h2 || r2 < 1e-20 {
                    continue;
                }
                let r = r2.sqrt();
                let hr = h - r;
                let pf = -m * (self.pressures[i] + self.pressures[j]) / (2.0 * self.densities[j])
                    * spiky_coeff
                    * hr
                    * hr
                    / r;
                fp[0] += pf * dx;
                fp[1] += pf * dy;
                fp[2] += pf * dz;
                let vf = mu * m / self.densities[j] * visc_coeff * hr;
                fv[0] += vf * (self.velocities[j][0] - self.velocities[i][0]);
                fv[1] += vf * (self.velocities[j][1] - self.velocities[i][1]);
                fv[2] += vf * (self.velocities[j][2] - self.velocities[i][2]);
            }
            let rho_i = self.densities[i];
            force_i[0] = (fp[0] + fv[0]) / rho_i + g[0];
            force_i[1] = (fp[1] + fv[1]) / rho_i + g[1];
            force_i[2] = (fp[2] + fv[2]) / rho_i + g[2];
        }

        for (i, force_i) in forces.iter().enumerate() {
            self.velocities[i][0] += force_i[0] * dt;
            self.velocities[i][1] += force_i[1] * dt;
            self.velocities[i][2] += force_i[2] * dt;
            self.positions[i][0] += self.velocities[i][0] * dt;
            self.positions[i][1] += self.velocities[i][1] * dt;
            self.positions[i][2] += self.velocities[i][2] * dt;
        }
    }
}

/// A simulation and the reference holding the same particles.
fn paired(positions: &[[f64; 3]], velocities: &[[f64; 3]]) -> (PySphSim, AllPairs) {
    let mut sim = PySphSim::new(PySphConfig::water());
    for &pos in positions {
        sim.add_particle(pos);
    }
    sim.velocities_mut().copy_from_slice(velocities);
    let reference = AllPairs {
        positions: positions.to_vec(),
        velocities: velocities.to_vec(),
        densities: vec![PySphConfig::water().rest_density; positions.len()],
        pressures: vec![0.0; positions.len()],
    };
    (sim, reference)
}

/// `|a - b| <= tol (1 + |b|)`.
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// Component-wise [`close`] of two vectors.
fn vectors_close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    a.iter().zip(&b).all(|(&x, &y)| close(x, y, tol))
}

/// A cloud of `count` particles with random velocities in a cube of side
/// `side` centred at the origin, so that coordinates of both signs occur.
fn random_cloud(count: usize, side: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut rng = Rng::new(seed);
    let half = 0.5 * side;
    let mut positions = Vec::with_capacity(count);
    let mut velocities = Vec::with_capacity(count);
    for _ in 0..count {
        positions.push([
            rng.range(-half, half),
            rng.range(-half, half),
            rng.range(-half, half),
        ]);
        velocities.push([
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ]);
    }
    (positions, velocities)
}

#[test]
fn test_sph_step_matches_the_all_pairs_reference_on_500_random_particles() {
    // 500 particles in a cube of side 3 h: every particle has about 70
    // neighbours, so the comparison involves many pairs and a dense cloud.
    let (positions, velocities) = random_cloud(500, 0.3, 0x5EED_0001);
    let (mut sim, mut reference) = paired(&positions, &velocities);
    let config = PySphConfig::water();
    let dt = 1.0e-4;

    for step in 0..3 {
        sim.step(dt);
        reference.step(&config, dt);
        for i in 0..positions.len() {
            let density = sim.density(i).expect("particle exists");
            let pressure = sim.pressure(i).expect("particle exists");
            assert!(
                close(density, reference.densities[i], 1e-12),
                "step {step}, particle {i}: density {density} vs {}",
                reference.densities[i]
            );
            assert!(
                close(pressure, reference.pressures[i], 1e-12),
                "step {step}, particle {i}: pressure {pressure} vs {}",
                reference.pressures[i]
            );
            let v = sim.velocity(i).expect("particle exists");
            let p = sim.position(i).expect("particle exists");
            assert!(
                vectors_close(v, reference.velocities[i], 1e-12),
                "step {step}, particle {i}: velocity {v:?} vs {:?}",
                reference.velocities[i]
            );
            assert!(
                vectors_close(p, reference.positions[i], 1e-12),
                "step {step}, particle {i}: position {p:?} vs {:?}",
                reference.positions[i]
            );
        }
    }
}

#[test]
fn test_sph_cloud_is_dense_enough_for_the_reference_comparison_to_matter() {
    // Guards the comparison above: the pair loops would be trivial for a
    // sparse cloud. The density must exceed the lone-particle value clearly
    // and most particles must have tens of neighbours.
    let (positions, velocities) = random_cloud(500, 0.3, 0x5EED_0001);
    let (mut sim, _) = paired(&positions, &velocities);
    sim.step(1.0e-4);
    let config = PySphConfig::water();
    let lone = config.particle_mass * 315.0 / (64.0 * std::f64::consts::PI * config.h.powi(9))
        * config.h.powi(6);
    let dense = (0..positions.len())
        .filter(|&i| sim.neighbors.row(i).len() >= 30)
        .count();
    assert!(dense > 400, "{dense} particles with >= 30 neighbours");
    let mean_density = (0..positions.len())
        .map(|i| sim.density(i).expect("particle exists"))
        .sum::<f64>()
        / positions.len() as f64;
    assert!(
        mean_density > 5.0 * lone,
        "mean density {mean_density} vs lone-particle density {lone}"
    );
}

#[test]
fn test_sph_isolated_particle_density_is_its_own_kernel_contribution() {
    // `all_neighbors` excludes the particle itself; the density must still
    // contain the self term m W(0) = m * poly6_coeff * h^6.
    let config = PySphConfig::water();
    let mut sim = PySphSim::new(config.clone());
    sim.add_particle([0.3, -0.2, 0.7]);
    sim.step(1.0e-3);
    let expected = config.particle_mass * 315.0 / (64.0 * std::f64::consts::PI * config.h.powi(9))
        * config.h.powi(6);
    let density = sim.density(0).expect("particle exists");
    assert!(
        close(density, expected, 1e-12),
        "density {density} vs {expected}"
    );
    // Only gravity acts on a particle without neighbours.
    let v = sim.velocity(0).expect("particle exists");
    assert!(v[0].abs() < 1e-15 && v[2].abs() < 1e-15, "v = {v:?}");
    assert!(close(v[1], config.gravity[1] * 1.0e-3, 1e-12), "v = {v:?}");
}

#[test]
fn test_sph_pair_at_exactly_the_smoothing_length_does_not_interact() {
    // The hash accepts distances `<= h`, the kernels have support `< h`: a
    // pair at exactly `h` is a neighbour candidate without any contribution.
    let config = PySphConfig {
        h: 0.125,
        ..PySphConfig::water()
    };
    let mut sim = PySphSim::new(config.clone());
    sim.add_particle([0.0, 0.0, 0.0]);
    sim.add_particle([0.125, 0.0, 0.0]);
    sim.step(1.0e-3);
    assert_eq!(
        sim.neighbors.row(0),
        &[0, 1],
        "the hash reports the pair at distance h"
    );
    let lone = config.particle_mass * 315.0 / (64.0 * std::f64::consts::PI * config.h.powi(9))
        * config.h.powi(6);
    for i in 0..2 {
        let density = sim.density(i).expect("particle exists");
        assert!(close(density, lone, 1e-12), "density {density} vs {lone}");
        let v = sim.velocity(i).expect("particle exists");
        assert!(
            v[0].abs() < 1e-15 && v[2].abs() < 1e-15,
            "no pair force, v = {v:?}"
        );
    }
}

#[test]
fn test_sph_step_visits_a_sparse_set_of_pairs() {
    // 20 x 20 x 20 lattice, spacing h/2: about 33 neighbours per interior
    // particle, so the table holds O(N) entries where an all-pairs loop
    // would visit N^2 = 64e6 pairs per pass.
    let mut sim = PySphSim::new(PySphConfig::water());
    sim.add_particle_block([0.0; 3], 20, 20, 20, 0.05);
    let n = sim.particle_count();
    assert_eq!(n, 8000);
    sim.step(1.0e-4);
    let entries: usize = (0..n).map(|i| sim.neighbors.row(i).len()).sum();
    assert!(entries >= n, "every particle is its own neighbour");
    assert!(
        entries < 50 * n,
        "{entries} neighbour entries for {n} particles"
    );
    // An interior particle sees the 27 lattice points with i^2 + j^2 + k^2 < 4
    // (itself included) and, depending on the rounding of `0.05 * i`, up to the
    // 6 axis points at exactly 2 spacings = h.
    let centre = (10 * 20 + 10) * 20 + 10;
    let seen = sim.neighbors.row(centre).len();
    assert!((27..=33).contains(&seen), "{seen} neighbours");
}

#[test]
fn test_sph_particles_far_from_the_origin_and_each_other_are_handled() {
    // The hash is sparse: coordinates far beyond any bounding box, and beyond
    // the `i32` cell range of the hash, cost nothing and do not panic.
    let config = PySphConfig::water();
    let mut sim = PySphSim::new(config.clone());
    sim.add_particle([0.0, -1.0e9, 0.0]);
    sim.add_particle([0.0, 1.0e9, 0.0]);
    sim.add_particle([3.0e12, 0.0, -3.0e12]);
    sim.step(1.0e-3);
    let lone = config.particle_mass * 315.0 / (64.0 * std::f64::consts::PI * config.h.powi(9))
        * config.h.powi(6);
    for i in 0..3 {
        let density = sim.density(i).expect("particle exists");
        assert!(
            close(density, lone, 1e-12),
            "particle {i}: density {density}"
        );
        let v = sim.velocity(i).expect("particle exists");
        assert!(v.iter().all(|c| c.is_finite()), "particle {i}: v = {v:?}");
        assert!(close(v[1], config.gravity[1] * 1.0e-3, 1e-12));
    }
}

#[test]
fn test_sph_clone_continues_identically() {
    // The neighbour table is scratch state: a clone must step like the
    // original.
    let (positions, velocities) = random_cloud(120, 0.25, 0x5EED_0002);
    let (mut sim, _) = paired(&positions, &velocities);
    sim.step(1.0e-4);
    let mut copy = sim.clone();
    sim.step(1.0e-4);
    copy.step(1.0e-4);
    assert_eq!(sim.all_positions(), copy.all_positions());
    assert_eq!(sim.all_velocities(), copy.all_velocities());
}
