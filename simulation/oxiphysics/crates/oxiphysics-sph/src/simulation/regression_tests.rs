//! Regression tests for the Tait stiffness of [`SphSimulation`], the periodic
//! wrapping of positions, and the periodic interactions of [`SphSim`] /
//! [`PeriodicSphSim`] (particles on opposite faces of the box interact through
//! the minimum-image convention).

use super::sph_sim_neighbors::SphSimNeighbors;
use super::types::{SolverType, SphSim, SphSimConfig, SphSimulationParams};
use super::types_sim::{PeriodicSphSim, SphSimulation, wrap_into_box};
use crate::kernel::CubicSplineKernel;
use crate::particle::{ParticleSet, SphParticle};
use crate::wcsph::compute_pressure_tait;
use oxiphysics_core::math::Vec3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic uniform pseudo-random numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// An empty gravity-free simulator with the given kernel radius.
fn empty_sim(kernel_radius: f64) -> SphSim {
    let mut config = SphSimConfig::default_water();
    config.kernel_radius = kernel_radius;
    config.gravity = [0.0; 3];
    SphSim::new(config, 0)
}

/// Append a particle to a simulator (all per-particle arrays stay in sync).
fn push_particle(sim: &mut SphSim, position: [f64; 3], velocity: [f64; 3], mass: f64) {
    sim.positions.push(position);
    sim.velocities.push(velocity);
    sim.masses.push(mass);
    sim.densities.push(0.0);
    sim.pressures.push(0.0);
    sim.forces.push([0.0; 3]);
}

/// `n`³ particles on a cubic lattice of spacing `dx = 1/n` filling the unit box,
/// with `h = h_over_dx · dx` and the lattice mass `ρ₀ dx³`.
fn unit_box_lattice(n: usize, h_over_dx: f64) -> SphSim {
    let dx = 1.0 / n as f64;
    let mut sim = empty_sim(h_over_dx * dx);
    let mass = sim.config.rest_density * dx.powi(3);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                push_particle(
                    &mut sim,
                    [
                        (i as f64 + 0.5) * dx,
                        (j as f64 + 0.5) * dx,
                        (k as f64 + 0.5) * dx,
                    ],
                    [0.0; 3],
                    mass,
                );
            }
        }
    }
    sim
}

/// Kernel-sum density of an infinite cubic lattice with spacing `dx`, found by
/// enumerating lattice offsets (no neighbour lists involved).
fn lattice_density_by_enumeration(dx: f64, h: f64, mass: f64) -> f64 {
    let reach = (2.0 * h / dx).ceil() as i64 + 1;
    let mut sum = 0.0;
    for a in -reach..=reach {
        for b in -reach..=reach {
            for c in -reach..=reach {
                let r = dx * ((a * a + b * b + c * c) as f64).sqrt();
                sum += SphSim::cubic_w(r, h);
            }
        }
    }
    mass * sum
}

/// Largest component magnitude of a set of force vectors.
fn max_force_component(forces: &[[f64; 3]]) -> f64 {
    forces
        .iter()
        .flat_map(|f| f.iter())
        .fold(0.0_f64, |acc, c| acc.max(c.abs()))
}

// ---------------------------------------------------------------------------
// F4636: the Tait stiffness of SphSimulation is K = rho0 c0^2, not K / gamma
// ---------------------------------------------------------------------------

#[test]
fn wcsph_params_stiffness_is_rho0_c0_squared() {
    let params = SphSimulationParams {
        rest_density: 1200.0,
        sound_speed: 80.0,
        smoothing_length: 0.07,
        viscosity: 0.02,
        ..Default::default()
    };
    let sim = SphSimulation::new(params);
    let w = sim.wcsph_params();
    assert!((w.stiffness - 1200.0 * 80.0 * 80.0).abs() < 1e-6 * w.stiffness);
    assert_eq!(w.gamma, 7.0);
    assert_eq!(w.rest_density, 1200.0);
    assert_eq!(w.smoothing_length, 0.07);
    assert_eq!(w.viscosity, 0.02);
}

#[test]
fn tait_pressure_of_simulation_params_uses_b_equal_rho0_c0_squared_over_seven() {
    let sim = SphSimulation::new(SphSimulationParams::default());
    let (rho0, c0) = (sim.params.rest_density, sim.params.sound_speed);
    let mut particles = ParticleSet::new();
    particles.add_particle(&SphParticle::new(Vec3::zeros(), Vec3::zeros(), 1.0));
    particles.densities[0] = 1.01 * rho0;
    compute_pressure_tait(&mut particles, &sim.wcsph_params());
    let expected = rho0 * c0 * c0 / 7.0 * (1.01_f64.powi(7) - 1.0);
    assert!(
        (particles.pressures[0] - expected).abs() < 1e-9 * expected,
        "p(1.01 rho0) = {} but rho0 c0^2 / 7 (1.01^7 - 1) = {expected}",
        particles.pressures[0]
    );
}

#[test]
fn effective_sound_speed_of_simulation_params_is_c0() {
    // dp/drho at rho0 is c0^2: p(rho0 (1 + e)) / (rho0 e) -> c0^2 as e -> 0.
    let sim = SphSimulation::new(SphSimulationParams {
        sound_speed: 100.0,
        ..Default::default()
    });
    let rho0 = sim.params.rest_density;
    let eps = 1e-7;
    let mut particles = ParticleSet::new();
    particles.add_particle(&SphParticle::new(Vec3::zeros(), Vec3::zeros(), 1.0));
    particles.densities[0] = rho0 * (1.0 + eps);
    compute_pressure_tait(&mut particles, &sim.wcsph_params());
    let c_eff = (particles.pressures[0] / (rho0 * eps)).sqrt();
    assert!(
        (c_eff - 100.0).abs() < 1e-3,
        "effective sound speed {c_eff} should be c0 = 100 (the old code gave c0/sqrt(7) = 37.8)"
    );
}

#[test]
fn simulation_step_pressures_follow_the_tait_law_with_rho0_c0_squared_over_seven() {
    let params = SphSimulationParams {
        smoothing_length: 0.06,
        sound_speed: 60.0,
        rest_density: 1000.0,
        viscosity: 0.0,
        solver_type: SolverType::Wcsph,
        gravity: Vec3::zeros(),
        ..Default::default()
    };
    let mut sim = SphSimulation::new(params);
    // A 4x4x4 block squeezed to 0.04 m spacing (rest spacing would be 0.05 m for
    // this mass), so every density exceeds rho0 and the pressures are positive.
    let spacing = 0.04;
    let mass = 1000.0 * 0.05_f64.powi(3);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    0.5 + f64::from(i) * spacing,
                    0.5 + f64::from(j) * spacing,
                    0.5 + f64::from(k) * spacing,
                );
                sim.particles
                    .add_particle(&SphParticle::new(pos, Vec3::zeros(), mass));
            }
        }
    }
    sim.step(1e-6, &CubicSplineKernel);
    let b = 1000.0 * 60.0 * 60.0 / 7.0;
    let mut positive = 0;
    for i in 0..sim.particles.len() {
        let rho = sim.particles.densities[i];
        let expected = b * ((rho / 1000.0).powi(7) - 1.0);
        let got = sim.particles.pressures[i];
        assert!(
            (got - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "particle {i}: pressure {got} but B ((rho/rho0)^7 - 1) = {expected}"
        );
        if got > 1e3 {
            positive += 1;
        }
    }
    assert!(positive > 0, "the squeezed block must be under pressure");
}

// ---------------------------------------------------------------------------
// F4639: periodic wrapping terminates for huge / infinite coordinates
// ---------------------------------------------------------------------------

#[test]
fn wrap_into_box_is_half_open_and_total() {
    assert_eq!(wrap_into_box(1.25, 1.0), 0.25);
    assert_eq!(wrap_into_box(-0.25, 1.0), 0.75);
    assert_eq!(wrap_into_box(1.0, 1.0), 0.0);
    assert_eq!(wrap_into_box(0.0, 1.0), 0.0);
    // rem_euclid(-1e-17, 1) rounds up to exactly 1.0, which is outside [0, 1).
    let tiny = wrap_into_box(-1e-17, 1.0);
    assert!((0.0..1.0).contains(&tiny), "got {tiny}");
    // -0.0 is normalised to +0.0.
    assert!(wrap_into_box(-1e20, 1.0).is_sign_positive());
    // Not wrappable: returned unchanged.
    assert_eq!(wrap_into_box(f64::INFINITY, 1.0), f64::INFINITY);
    assert_eq!(wrap_into_box(f64::NEG_INFINITY, 1.0), f64::NEG_INFINITY);
    assert!(wrap_into_box(f64::NAN, 1.0).is_nan());
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(wrap_into_box(3.5, bad), 3.5, "length {bad} must be skipped");
    }
}

#[test]
fn simulation_apply_pbc_terminates_on_infinite_and_huge_coordinates() {
    let mut sim = SphSimulation::new(SphSimulationParams::default());
    sim.add_particle_raw([f64::INFINITY, 1e20, -1e20], [0.0; 3], 1.0);
    sim.add_particle_raw([f64::NEG_INFINITY, f64::NAN, 2.5], [0.0; 3], 1.0);
    sim.apply_pbc([1.0; 3]);
    let a = sim.particles.positions[0];
    assert_eq!(a.x, f64::INFINITY, "a non-finite coordinate is left alone");
    assert!((0.0..1.0).contains(&a.y), "y = {}", a.y);
    assert!((0.0..1.0).contains(&a.z), "z = {}", a.z);
    let b = sim.particles.positions[1];
    assert_eq!(b.x, f64::NEG_INFINITY);
    assert!(b.y.is_nan());
    assert_eq!(b.z, 0.5);
}

#[test]
fn simulation_apply_pbc_maps_tiny_negative_into_the_half_open_box() {
    let mut sim = SphSimulation::new(SphSimulationParams::default());
    sim.add_particle_raw([-1e-17, -1e-300, -4.0e-17], [0.0; 3], 1.0);
    sim.apply_pbc([1.0, 2.0, 0.3]);
    let p = sim.particles.positions[0];
    assert!((0.0..1.0).contains(&p.x), "x = {}", p.x);
    assert!((0.0..2.0).contains(&p.y), "y = {}", p.y);
    assert!((0.0..0.3).contains(&p.z), "z = {}", p.z);
}

#[test]
fn simulation_apply_pbc_skips_axes_without_a_positive_finite_length() {
    let mut sim = SphSimulation::new(SphSimulationParams::default());
    sim.add_particle_raw([7.5, -3.5, 9.25], [0.0; 3], 1.0);
    sim.apply_pbc([0.0, f64::NAN, -2.0]);
    let p = sim.particles.positions[0];
    assert_eq!((p.x, p.y), (7.5, -3.5));
    assert_eq!(p.z, 9.25);
}

#[test]
fn periodic_sim_apply_pbc_terminates_and_stays_in_the_half_open_box() {
    let mut inner = empty_sim(0.05);
    push_particle(&mut inner, [f64::INFINITY, 1e20, -1e20], [0.0; 3], 1.0);
    push_particle(&mut inner, [-1e-17, 2.75, -0.0], [0.0; 3], 1.0);
    let mut sim = PeriodicSphSim::new(inner, [1.0; 3]);
    sim.apply_pbc();
    let a = sim.inner.positions[0];
    assert_eq!(a[0], f64::INFINITY);
    assert!((0.0..1.0).contains(&a[1]) && (0.0..1.0).contains(&a[2]));
    let b = sim.inner.positions[1];
    assert!(b.iter().all(|c| (0.0..1.0).contains(c)), "{b:?}");
    assert_eq!(b[1], 0.75);
}

// ---------------------------------------------------------------------------
// F4641: PeriodicSphSim interacts across the periodic faces
// ---------------------------------------------------------------------------

#[test]
fn periodic_lattice_has_the_interior_density_on_every_particle() {
    let n = 8;
    let h_over_dx = 1.2;
    let mut sim = unit_box_lattice(n, h_over_dx);
    sim.periodic = Some([1.0; 3]);
    sim.compute_density_sum();
    let dx = 1.0 / n as f64;
    let expected = lattice_density_by_enumeration(dx, h_over_dx * dx, sim.masses[0]);
    for (i, rho) in sim.densities.iter().enumerate() {
        assert!(
            (rho - expected).abs() < 1e-9 * expected,
            "particle {i} (position {:?}): density {rho} but the infinite lattice has {expected}",
            sim.positions[i]
        );
    }
    // Face and corner particles in particular equal the interior one within 1 %.
    let corner = sim.densities[0];
    let interior = sim.densities[(n / 2) * n * n + (n / 2) * n + n / 2];
    assert!((corner - interior).abs() < 0.01 * interior);
}

#[test]
fn open_lattice_density_drops_towards_the_faces() {
    // Negative control: without periodicity the corner particle misses most of
    // its neighbours.
    let n = 8;
    let mut sim = unit_box_lattice(n, 1.2);
    sim.compute_density_sum();
    let corner = sim.densities[0];
    let interior = sim.densities[(n / 2) * n * n + (n / 2) * n + n / 2];
    assert!(
        corner < 0.6 * interior,
        "corner density {corner} should be well below the interior density {interior}"
    );
}

#[test]
fn periodic_sim_wrapper_makes_the_inner_lattice_periodic() {
    let inner = unit_box_lattice(8, 1.2);
    let mut sim = PeriodicSphSim::new(inner, [1.0; 3]);
    assert_eq!(sim.inner.periodic, Some([1.0; 3]));
    assert!(sim.min_image_valid());
    sim.inner.compute_density_sum();
    let corner = sim.inner.densities[0];
    let interior = sim.inner.densities[(4 * 8 + 4) * 8 + 4];
    assert!((corner - interior).abs() < 1e-9 * interior);
}

#[test]
fn periodic_uniform_pressure_exerts_no_net_force_on_any_particle() {
    let n = 8;
    let h_over_dx = 1.2;
    let dx = 1.0 / n as f64;
    let pressure = 5.0e3;
    let mut periodic = unit_box_lattice(n, h_over_dx);
    periodic.periodic = Some([1.0; 3]);
    periodic.compute_density_sum();
    periodic.pressures.fill(pressure);
    periodic.compute_pressure_forces();
    // Magnitude of a single pair force at one lattice spacing.
    let m = periodic.masses[0];
    let rho = periodic.densities[0];
    let typical_pair_force =
        m * m * 2.0 * pressure / (rho * rho) * SphSim::cubic_grad_w(dx, h_over_dx * dx).abs();
    assert!(typical_pair_force > 1.0);
    let periodic_max = max_force_component(&periodic.forces);
    assert!(
        periodic_max < 1e-9 * typical_pair_force,
        "uniform periodic lattice must be in equilibrium, max force {periodic_max} \
         (one pair force is {typical_pair_force})"
    );

    // Negative control: the same uniform pressure on the open lattice pushes the
    // face particles (truncated support).
    let mut open = unit_box_lattice(n, h_over_dx);
    open.compute_density_sum();
    open.pressures.fill(pressure);
    open.compute_pressure_forces();
    let open_max = max_force_component(&open.forces);
    assert!(
        open_max > 0.1 * typical_pair_force,
        "the open lattice must feel face forces: {open_max} (one pair force is {typical_pair_force})"
    );
}

/// Two particles 0.04 apart, with `h = 0.05` and a mass that compresses them.
const PAIR_MASS: f64 = 0.36;

fn pair_sim(a: [f64; 3], b: [f64; 3], va: [f64; 3], vb: [f64; 3]) -> SphSim {
    let mut sim = empty_sim(0.05);
    push_particle(&mut sim, a, va, PAIR_MASS);
    push_particle(&mut sim, b, vb, PAIR_MASS);
    sim
}

#[test]
fn pair_across_a_periodic_face_interacts_like_an_interior_pair() {
    // Interior reference: left particle at x = 0.50, right at x = 0.54.
    let mut interior = pair_sim([0.50, 0.5, 0.5], [0.54, 0.5, 0.5], [0.0; 3], [0.0; 3]);
    interior.step();
    // Across the face: A at x = 0.02 is the *right* partner of B at x = 0.98
    // (B sits at -0.02 in the minimum-image picture).
    let mut face = PeriodicSphSim::new(
        pair_sim([0.02, 0.5, 0.5], [0.98, 0.5, 0.5], [0.0; 3], [0.0; 3]),
        [1.0; 3],
    );
    face.step();

    // Same densities, pressures and velocities as the interior pair
    // (B <-> left, A <-> right).
    for (face_idx, interior_idx) in [(1_usize, 0_usize), (0, 1)] {
        let (rf, ri) = (
            face.inner.densities[face_idx],
            interior.densities[interior_idx],
        );
        assert!((rf - ri).abs() < 1e-9 * ri, "density {rf} vs {ri}");
        for k in 0..3 {
            let (vf, vi) = (
                face.inner.velocities[face_idx][k],
                interior.velocities[interior_idx][k],
            );
            assert!(
                (vf - vi).abs() < 1e-9 * vi.abs().max(1e-12),
                "velocity[{k}] {vf} vs {vi}"
            );
        }
    }
    // The compressed pair repels: A is pushed to +x, B to -x, equally strongly.
    let (va, vb) = (face.inner.velocities[0], face.inner.velocities[1]);
    assert!(va[0] > 1e-6, "A must be pushed away from B, got {}", va[0]);
    assert!(vb[0] < -1e-6, "B must be pushed away from A, got {}", vb[0]);
    assert!((va[0] + vb[0]).abs() < 1e-12 * va[0].abs());
    let momentum = face.total_momentum();
    assert!(momentum.iter().all(|m| m.abs() < 1e-12));

    // The same two particles in an open domain are 0.96 apart: no interaction.
    let mut open = pair_sim([0.02, 0.5, 0.5], [0.98, 0.5, 0.5], [0.0; 3], [0.0; 3]);
    open.step();
    assert!(open.velocities.iter().flatten().all(|v| v.abs() < 1e-15));
}

#[test]
fn viscosity_acts_between_particles_approaching_across_a_periodic_face() {
    let approach = [1.0, 0.0, 0.0];
    let recede = [-1.0, 0.0, 0.0];
    // Interior pair: left (x = 0.50) moves right, right (x = 0.54) moves left.
    let mut interior = pair_sim([0.50, 0.5, 0.5], [0.54, 0.5, 0.5], approach, recede);
    interior.compute_density_sum();
    interior.compute_viscosity_forces();
    // Face pair: B (x = 0.98 = -0.02) moves right, A (x = 0.02) moves left.
    let mut face = pair_sim([0.02, 0.5, 0.5], [0.98, 0.5, 0.5], recede, approach);
    face.periodic = Some([1.0; 3]);
    face.compute_density_sum();
    face.compute_viscosity_forces();

    let (f_left, f_right) = (interior.forces[0], interior.forces[1]);
    assert!(f_left[0] < -1e-9, "viscosity must brake the approach");
    assert!(f_right[0] > 1e-9);
    // B is the left partner, A the right one.
    for k in 0..3 {
        assert!((face.forces[1][k] - f_left[k]).abs() <= 1e-9 * f_left[0].abs());
        assert!((face.forces[0][k] - f_right[k]).abs() <= 1e-9 * f_right[0].abs());
    }
    // Open domain: nothing within the support.
    let mut open = pair_sim([0.02, 0.5, 0.5], [0.98, 0.5, 0.5], recede, approach);
    open.compute_density_sum();
    open.compute_viscosity_forces();
    assert!(open.forces.iter().flatten().all(|f| *f == 0.0));
}

/// Jittered 4x4x4 cluster (spacing 0.04) centred at `centre`, with random
/// velocities; `seed` fixes the cloud so that two copies are identical.
fn cluster_sim(centre: [f64; 3], seed: u64) -> SphSim {
    let mut sim = empty_sim(0.048);
    sim.config.gravity = [0.0, -9.81, 0.0];
    sim.config.dt = 2e-4;
    let mut rng = Lcg(seed);
    let spacing: f64 = 0.04;
    let mass = sim.config.rest_density * spacing.powi(3);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let jitter = |rng: &mut Lcg| rng.range(-0.1, 0.1) * spacing;
                let pos = [
                    centre[0] + (f64::from(i) - 1.5) * spacing + jitter(&mut rng),
                    centre[1] + (f64::from(j) - 1.5) * spacing + jitter(&mut rng),
                    centre[2] + (f64::from(k) - 1.5) * spacing + jitter(&mut rng),
                ];
                let vel = [
                    rng.range(-0.2, 0.2),
                    rng.range(-0.2, 0.2),
                    rng.range(-0.2, 0.2),
                ];
                push_particle(&mut sim, pos, vel, mass);
            }
        }
    }
    sim
}

#[test]
fn periodic_step_with_surface_tension_is_translation_invariant_across_the_face() {
    // The same cloud once in the middle of the box and once straddling the x = 0
    // face.  With periodic interactions the dynamics cannot tell the difference.
    let middle = cluster_sim([0.5, 0.5, 0.5], 12345);
    let straddling = cluster_sim([0.0, 0.5, 0.5], 12345);
    let initial_velocities = middle.velocities.clone();
    let mut a = PeriodicSphSim::new(middle, [1.0; 3]);
    let mut b = PeriodicSphSim::new(straddling, [1.0; 3]);
    assert!(
        b.inner.positions.iter().any(|p| p[0] < 0.0),
        "the cloud must reach across the x = 0 face"
    );
    let steps = 4;
    for _ in 0..steps {
        a.step_with_surface_tension();
        b.step_with_surface_tension();
    }
    for i in 0..a.len() {
        for k in 0..3 {
            let (va, vb) = (a.inner.velocities[i][k], b.inner.velocities[i][k]);
            assert!(
                (va - vb).abs() <= 1e-8 * (1.0 + va.abs()),
                "particle {i} velocity[{k}]: {va} (middle) vs {vb} (straddling)"
            );
        }
        // Positions agree after undoing the x translation of 0.5 (mod 1).
        let shift = [0.5, 0.0, 0.0];
        for (k, &s) in shift.iter().enumerate() {
            let expected = wrap_into_box(a.inner.positions[i][k] - s, 1.0);
            let got = b.inner.positions[i][k];
            let diff = (expected - got).abs();
            assert!(
                diff < 1e-9 || (1.0 - diff).abs() < 1e-9,
                "particle {i} position[{k}]: {expected} vs {got}"
            );
        }
    }
    // The particle interactions were not trivial: they changed the velocities
    // well beyond the ballistic gravity update ...
    let dt = a.inner.config.dt;
    let g = a.inner.config.gravity;
    let mut max_interaction = 0.0_f64;
    for (v, v0) in a.inner.velocities.iter().zip(initial_velocities.iter()) {
        for k in 0..3 {
            let ballistic = v0[k] + g[k] * dt * f64::from(steps);
            max_interaction = max_interaction.max((v[k] - ballistic).abs());
        }
    }
    assert!(
        max_interaction > 1e-3,
        "pair forces should matter, max deviation from ballistic motion {max_interaction}"
    );
}

#[test]
fn periodic_step_conserves_momentum_for_a_cloud_straddling_the_face() {
    // Pressure and viscosity forces are pairwise action-reaction pairs, also for
    // pairs that interact through a periodic face, so only gravity changes the
    // total momentum.
    let cloud = cluster_sim([0.0, 0.5, 0.5], 777);
    let mass_total: f64 = cloud.masses.iter().sum();
    let initial = cloud.linear_momentum();
    let (dt, g) = (cloud.config.dt, cloud.config.gravity);
    let mut sim = PeriodicSphSim::new(cloud, [1.0; 3]);
    let steps = 5;
    for _ in 0..steps {
        sim.step();
    }
    let finalp = sim.total_momentum();
    for k in 0..3 {
        let expected = initial[k] + mass_total * g[k] * dt * f64::from(steps);
        assert!(
            (finalp[k] - expected).abs() < 1e-9 * (1.0 + expected.abs()),
            "momentum[{k}] {} but initial + gravity impulse is {expected}",
            finalp[k]
        );
    }
}

/// All-pairs reference of the periodic neighbour lists: wrap both positions into
/// the box and take the nearest of the three images on each periodic axis.
fn reference_periodic_rows(
    positions: &[[f64; 3]],
    radius: f64,
    box_size: [f64; 3],
) -> Vec<Vec<usize>> {
    let nearest = |a: f64, b: f64, l: f64| -> f64 {
        if l.is_finite() && l > 0.0 {
            let (wa, wb) = (a.rem_euclid(l), b.rem_euclid(l));
            [wa - wb - l, wa - wb, wa - wb + l]
                .into_iter()
                .fold(
                    f64::INFINITY,
                    |best, d| if d.abs() < best.abs() { d } else { best },
                )
        } else {
            a - b
        }
    };
    positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (0..positions.len())
                .filter(|&j| j != i)
                .filter(|&j| {
                    let d2: f64 = (0..3)
                        .map(|k| nearest(p[k], positions[j][k], box_size[k]).powi(2))
                        .sum();
                    d2 <= radius * radius
                })
                .collect()
        })
        .collect()
}

#[test]
fn periodic_neighbour_lists_match_an_all_pairs_minimum_image_search() {
    let mut rng = Lcg(2024);
    // Positions deliberately stick out of the box on all sides.
    let positions: Vec<[f64; 3]> = (0..260)
        .map(|_| {
            [
                rng.range(-0.3, 1.3),
                rng.range(-0.3, 1.3),
                rng.range(-0.3, 1.3),
            ]
        })
        .collect();
    let cases: [([f64; 3], f64); 7] = [
        ([1.0, 1.0, 1.0], 0.05),
        ([0.45, 0.45, 0.45], 0.05),
        // Two cells along an axis: the +-1 stencil would visit a cell twice.
        ([0.25, 1.0, 1.0], 0.05),
        // Shorter than the support radius: a single cell.
        ([0.3, 1.0, 1.0], 0.1),
        // Mixed: open z, periodic x/y.
        ([1.0, 0.8, 0.0], 0.06),
        // Axes without a positive finite length are open.
        ([1.0, f64::NAN, -1.0], 0.06),
        ([0.9, 1.1, 0.7], 0.08),
    ];
    for (box_size, kernel_radius) in cases {
        let radius = 2.0 * kernel_radius;
        let lists = SphSimNeighbors::build_with(&positions, kernel_radius, Some(box_size));
        let reference = reference_periodic_rows(&positions, radius, box_size);
        let mut total = 0;
        for (i, expected) in reference.iter().enumerate() {
            assert_eq!(
                lists.neighbors_of(i),
                expected.as_slice(),
                "box {box_size:?}, h {kernel_radius}: neighbour list of particle {i}"
            );
            total += expected.len();
        }
        assert!(total > 50, "box {box_size:?} has too few pairs ({total})");
    }
}

#[test]
fn periodic_neighbour_lists_ignore_non_finite_positions() {
    let positions = [
        [0.03, 0.5, 0.5],
        [0.95, 0.5, 0.5],
        [f64::NAN, 0.5, 0.5],
        [0.5, f64::INFINITY, 0.5],
    ];
    let lists = SphSimNeighbors::build_with(&positions, 0.05, Some([1.0; 3]));
    assert_eq!(lists.neighbors_of(0), &[1]);
    assert_eq!(lists.neighbors_of(1), &[0]);
    assert!(lists.neighbors_of(2).is_empty());
    assert!(lists.neighbors_of(3).is_empty());
}

#[test]
fn refresh_neighbors_follows_the_periodic_box() {
    let mut sim = pair_sim([0.02, 0.5, 0.5], [0.98, 0.5, 0.5], [0.0; 3], [0.0; 3]);
    assert_eq!(sim.refresh_neighbors().entry_count(), 0, "open domain");
    sim.periodic = Some([1.0; 3]);
    assert_eq!(sim.refresh_neighbors().entry_count(), 2, "periodic box");
    assert_eq!(sim.refresh_neighbors().periodic(), Some([1.0; 3]));
    sim.periodic = Some([5.0; 3]);
    assert_eq!(sim.refresh_neighbors().entry_count(), 0, "box too large");
    sim.periodic = None;
    assert_eq!(sim.refresh_neighbors().entry_count(), 0, "open again");
    assert!(sim.refresh_neighbors().periodic().is_none());
}

#[test]
fn open_neighbour_cache_is_not_current_for_a_periodic_query() {
    let positions = [[0.1, 0.2, 0.3], [0.15, 0.2, 0.3]];
    let open = SphSimNeighbors::build(&positions, 0.05);
    assert!(open.is_current(&positions, 0.05));
    assert!(open.is_current_with(&positions, 0.05, None));
    assert!(!open.is_current_with(&positions, 0.05, Some([1.0; 3])));
    let periodic = SphSimNeighbors::build_with(&positions, 0.05, Some([1.0; 3]));
    assert!(periodic.is_current_with(&positions, 0.05, Some([1.0; 3])));
    assert!(!periodic.is_current(&positions, 0.05));
    assert!(!periodic.is_current_with(&positions, 0.05, Some([2.0; 3])));
}

#[test]
fn periodic_sim_honours_a_changed_box_size_on_the_next_step() {
    let inner = pair_sim([0.02, 0.5, 0.5], [0.98, 0.5, 0.5], [0.0; 3], [0.0; 3]);
    let mut sim = PeriodicSphSim::new(inner, [1.0; 3]);
    sim.step();
    assert!(sim.inner.velocities[0][0] > 1e-6, "pair interacts in L = 1");

    // Enlarge the box: the pair is now 0.96 apart, out of range.
    let mut sim = PeriodicSphSim::new(
        pair_sim([0.02, 0.5, 0.5], [0.98, 0.5, 0.5], [0.0; 3], [0.0; 3]),
        [1.0; 3],
    );
    sim.box_size = [5.0; 3];
    sim.step();
    assert_eq!(sim.inner.periodic, Some([5.0; 3]));
    assert!(
        sim.inner
            .velocities
            .iter()
            .flatten()
            .all(|v| v.abs() < 1e-15)
    );
}

#[test]
fn min_image_valid_requires_four_smoothing_lengths_per_periodic_axis() {
    let make = |h: f64, box_size: [f64; 3]| PeriodicSphSim::new(empty_sim(h), box_size);
    assert!(make(0.1, [1.0; 3]).min_image_valid());
    assert!(
        make(0.1, [0.4, 0.5, 0.6]).min_image_valid(),
        "L = 4h is allowed"
    );
    assert!(!make(0.1, [1.0, 0.39, 1.0]).min_image_valid());
    assert!(!make(0.1, [0.3; 3]).min_image_valid());
    // Open axes (no positive finite length) do not count.
    assert!(make(0.1, [1.0, 0.0, f64::NAN]).min_image_valid());
}
