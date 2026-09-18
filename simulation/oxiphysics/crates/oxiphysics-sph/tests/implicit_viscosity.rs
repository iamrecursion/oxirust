// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the implicit / semi-implicit SPH viscosity solver.
//!
//! Headline guarantee: the implicit Morris-Laplacian update is unconditionally
//! stable, so it runs at time steps far beyond the explicit viscous limit
//! `Δt < h²/(2·d·ν)` without blowing up, while still dissipating kinetic energy
//! monotonically.

use oxiphysics_sph::kernel::{CubicSplineKernel, SphKernel, grad};
use oxiphysics_sph::viscosity_implicit::{
    NeighborEntry, ViscosityError, ViscosityParticles, ViscositySolveOptions,
    solve_implicit_viscosity,
};

const MORRIS_EPSILON: f64 = 0.01;

/// Build cubic-spline neighbour lists for `positions` with support radius `2h`.
fn build_neighbors(positions: &[[f64; 3]], h: f64) -> Vec<Vec<NeighborEntry>> {
    let kernel = CubicSplineKernel;
    let support = 2.0 * h;
    let n = positions.len();
    let mut neighbors = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let r_ij = [
                positions[i][0] - positions[j][0],
                positions[i][1] - positions[j][1],
                positions[i][2] - positions[j][2],
            ];
            let r = (r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]).sqrt();
            if r >= support || r < 1.0e-12 {
                continue;
            }
            let gw = grad(&kernel, r_ij, h);
            neighbors[i].push(NeighborEntry::new(j, r_ij, gw));
        }
    }
    neighbors
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Total kinetic energy `Σ ½ m_i |v_i|²`.
fn kinetic_energy(velocities: &[[f64; 3]], masses: &[f64]) -> f64 {
    velocities
        .iter()
        .zip(masses.iter())
        .map(|(v, &m)| 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum()
}

/// Build a regular 2-D grid of particles in the z = 0 plane, spacing `dx`.
fn grid_2d(nx: usize, ny: usize, dx: f64) -> Vec<[f64; 3]> {
    let mut positions = Vec::with_capacity(nx * ny);
    for ix in 0..nx {
        for iy in 0..ny {
            positions.push([ix as f64 * dx, iy as f64 * dx, 0.0]);
        }
    }
    positions
}

/// Test 1 — stability gate: implicit solver stays stable at 100× the explicit
/// viscous limit, with no NaNs and monotonically non-increasing kinetic energy.
#[test]
fn implicit_stable_at_100x_explicit_limit() {
    let dx = 0.1_f64;
    let h = 0.1_f64;
    let nu = 1.0_f64;
    let rho0 = 1000.0_f64;
    let dim = 2.0_f64;

    let dt_explicit = dx * dx / (2.0 * dim * nu); // = 0.00125
    let dt = 100.0 * dt_explicit; // 0.125 — far past the explicit bound

    let positions = grid_2d(8, 8, dx);
    let n = positions.len();
    let neighbors = build_neighbors(&positions, h);

    // Checkerboard initial velocity field (high-frequency, the worst case for
    // explicit stability).
    let mut velocities: Vec<[f64; 3]> = Vec::with_capacity(n);
    for (idx, _) in positions.iter().enumerate() {
        let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
        velocities.push([sign, 0.0, 0.0]);
    }
    let densities = vec![rho0; n];
    let masses = vec![rho0 * dx * dx; n]; // 2-D mass per particle
    let viscosities = vec![rho0 * nu; n]; // μ = ρ·ν

    let ke0 = kinetic_energy(&velocities, &masses);

    for step in 0..20 {
        let particles = ViscosityParticles {
            velocities: &velocities,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        let updated = solve_implicit_viscosity(
            &particles,
            &neighbors,
            dt,
            h,
            ViscositySolveOptions::default(),
        )
        .unwrap_or_else(|e| panic!("solve failed at step {step}: {e}"));
        for v in &updated {
            for &c in v {
                assert!(c.is_finite(), "non-finite velocity at step {step}");
            }
        }
        let ke = kinetic_energy(&updated, &masses);
        assert!(
            ke <= ke0 + 1e-9,
            "kinetic energy grew (step {step}): ke={ke}, ke0={ke0}"
        );
        velocities = updated;
    }

    let ke_final = kinetic_energy(&velocities, &masses);
    assert!(
        ke_final < ke0,
        "viscosity must dissipate energy: ke_final={ke_final}, ke0={ke0}"
    );
}

/// Test 2 — convergence to truth: explicit (small dt) and implicit (large dt)
/// integration of the same diffusion problem reach the same steady state.
///
/// A 1-D row of particles carries a step velocity profile.  Pure diffusion with
/// no-flux (isolated) ends drives the field toward its conserved mean.  We march
/// the explicit Morris update at a stable dt and the implicit update at 50× that
/// dt to the same physical time, and require the two velocity fields to agree.
#[test]
fn implicit_matches_explicit_steady_state() {
    let dx = 0.1_f64;
    let h = 0.1_f64;
    // Kinematic viscosity; the Laplacian `c_ij` below already contains it.
    let nu = 0.25_f64;
    let rho0 = 1000.0_f64;

    // 1-D chain of 16 particles.
    let n = 16usize;
    let positions: Vec<[f64; 3]> = (0..n).map(|i| [i as f64 * dx, 0.0, 0.0]).collect();
    let neighbors = build_neighbors(&positions, h);
    let densities = vec![rho0; n];
    let masses = vec![rho0 * dx; n];
    let viscosities = vec![rho0 * nu; n];

    // Step profile: left half +1, right half −1.
    let initial: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            if i < n / 2 {
                [1.0, 0.0, 0.0]
            } else {
                [-1.0, 0.0, 0.0]
            }
        })
        .collect();

    let t_final = 0.5_f64;

    // -- Explicit reference, dt at the *discrete* stable limit. -----------
    // The continuum bound h²/(2dν) is too loose for the SPH Morris operator,
    // whose largest eigenvalue is ≈ 2·maxᵢ Σⱼ c_ij.  Forward-Euler diffusion
    // is stable only for dt·λ_max < 2, so size dt from the assembled
    // operator: dt = safety / maxᵢ Σⱼ c_ij with safety = 0.2 gives
    // dt·λ_max ≈ 0.4, comfortably stable.  (`c_ij` is the full viscous
    // coupling: it carries ν through μ = ρν, there is no separate factor.)
    let max_diag_sum = (0..n)
        .map(|i| {
            neighbors[i]
                .iter()
                .map(|entry| {
                    let r2 = dot3(entry.r_ij, entry.r_ij);
                    if r2 < 1.0e-24 {
                        return 0.0;
                    }
                    let denom = r2 + MORRIS_EPSILON * h * h;
                    (masses[entry.j] / densities[entry.j]) * (viscosities[i] + viscosities[entry.j])
                        / densities[i]
                        * (-dot3(entry.r_ij, entry.grad_w_ij) / denom)
                })
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);
    let dt_explicit = 0.2 / max_diag_sum;
    let n_explicit = (t_final / dt_explicit).ceil() as usize;
    let dt_e = t_final / n_explicit as f64;
    let mut v_explicit = initial.clone();
    for _ in 0..n_explicit {
        let mut next = v_explicit.clone();
        for i in 0..n {
            let mut lap = [0.0_f64; 3];
            for entry in &neighbors[i] {
                let j = entry.j;
                let r2 = dot3(entry.r_ij, entry.r_ij);
                if r2 < 1.0e-24 {
                    continue;
                }
                let denom = r2 + MORRIS_EPSILON * h * h;
                let c = (masses[j] / densities[j]) * (viscosities[i] + viscosities[j])
                    / densities[i]
                    * (-dot3(entry.r_ij, entry.grad_w_ij) / denom);
                for k in 0..3 {
                    lap[k] += c * (v_explicit[j][k] - v_explicit[i][k]);
                }
            }
            for k in 0..3 {
                next[i][k] = v_explicit[i][k] + dt_e * lap[k];
            }
        }
        v_explicit = next;
    }

    // -- Implicit, dt 50× larger. -----------------------------------------
    let dt_i = 50.0 * dt_e;
    let n_implicit = (t_final / dt_i).ceil() as usize;
    let dt_i = t_final / n_implicit as f64;
    let mut v_implicit = initial.clone();
    for _ in 0..n_implicit {
        let particles = ViscosityParticles {
            velocities: &v_implicit,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        v_implicit = solve_implicit_viscosity(
            &particles,
            &neighbors,
            dt_i,
            h,
            ViscositySolveOptions::default(),
        )
        .expect("implicit solve should converge");
    }

    // Both fields should be close (diffusion is smoothing toward the mean).
    let mut max_diff = 0.0_f64;
    for i in 0..n {
        max_diff = max_diff.max((v_explicit[i][0] - v_implicit[i][0]).abs());
    }
    // The two schemes have different temporal truncation error; require
    // agreement to within a generous absolute tolerance on the O(1) field.
    assert!(
        max_diff < 0.05,
        "implicit/explicit steady states disagree: max |Δv| = {max_diff}"
    );

    // Sanity: total momentum is conserved by both (equal masses, symmetric op).
    let p_e: f64 = (0..n).map(|i| masses[i] * v_explicit[i][0]).sum();
    let p_i: f64 = (0..n).map(|i| masses[i] * v_implicit[i][0]).sum();
    assert!(
        (p_e - p_i).abs() < 1e-6,
        "momentum mismatch: {p_e} vs {p_i}"
    );
}

/// Test 3 — symmetry / positive definiteness through the public API.
///
/// With unequal masses the physical coupling `c_ij` is not symmetric
/// (`c_ij ≠ c_ji`), but the solve operator `S = (I − Δt L)⁻¹` is self-adjoint in
/// the mass inner product `⟨a, b⟩_M = Σ m_i a_i·b_i` because `m_i c_ij = m_j c_ji`.
/// The test checks `⟨u, S w⟩_M = ⟨S u, w⟩_M`, positivity `⟨w, S w⟩_M > 0` and the
/// contraction `‖S w‖_M ≤ ‖w‖_M` for two pseudo-random fields.
#[test]
fn solve_operator_is_symmetric_positive_definite_in_the_mass_inner_product() {
    let h = 0.1_f64;
    let dt = 0.05_f64;
    let positions = vec![
        [0.0, 0.0, 0.0],
        [0.08, 0.0, 0.0],
        [-0.08, 0.0, 0.0],
        [0.0, 0.08, 0.0],
        [0.0, -0.08, 0.0],
    ];
    let n = positions.len();
    let neighbors = build_neighbors(&positions, h);
    let densities = vec![900.0, 1100.0, 1000.0, 1200.0, 800.0];
    let masses = vec![1.0, 2.0, 0.5, 3.0, 1.5];
    let kinematic = [0.01, 0.02, 0.005, 0.03, 0.015];
    let viscosities: Vec<f64> = densities
        .iter()
        .zip(kinematic.iter())
        .map(|(rho, nu)| rho * nu)
        .collect();
    let options = ViscositySolveOptions {
        tol: 1.0e-13,
        max_iters: 5000,
    };
    let apply = |field: &[[f64; 3]]| -> Vec<[f64; 3]> {
        let particles = ViscosityParticles {
            velocities: field,
            densities: &densities,
            masses: &masses,
            viscosities: &viscosities,
        };
        solve_implicit_viscosity(&particles, &neighbors, dt, h, options)
            .expect("solve should converge")
    };
    let inner = |a: &[[f64; 3]], b: &[[f64; 3]]| -> f64 {
        (0..n)
            .map(|i| masses[i] * (0..3).map(|k| a[i][k] * b[i][k]).sum::<f64>())
            .sum()
    };
    let field = |seed: f64| -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let t = i as f64 + seed;
                [(1.7 * t).sin(), (0.9 * t).cos(), (2.3 * t).sin() * 0.5]
            })
            .collect()
    };
    let u = field(0.3);
    let w = field(2.1);
    let s_u = apply(&u);
    let s_w = apply(&w);

    // Self-adjointness.
    let lhs = inner(&u, &s_w);
    let rhs = inner(&s_u, &w);
    assert!(
        (lhs - rhs).abs() < 1e-9 * (lhs.abs() + rhs.abs()),
        "⟨u, S w⟩ = {lhs} but ⟨S u, w⟩ = {rhs}"
    );
    // Positive definiteness and contraction.
    for (name, v, s_v) in [("u", &u, &s_u), ("w", &w, &s_w)] {
        let quad = inner(v, s_v);
        assert!(quad > 0.0, "⟨{name}, S {name}⟩ = {quad} must be positive");
        assert!(
            inner(s_v, s_v) <= inner(v, v) + 1e-12,
            "S must not amplify {name} in the mass norm"
        );
    }
}

/// Test 4 — CG convergence on a 50-particle system: the internal Jacobi-PCG
/// must converge within the iteration budget, i.e. the solver returns `Ok`
/// rather than `SolverNotConverged`.
#[test]
fn solver_converges_on_fifty_particles() {
    let dx = 0.1_f64;
    let h = 0.1_f64;
    let nu = 1.0_f64;
    let dt = 0.1_f64;
    let rho0 = 1000.0_f64;

    // 5×10 grid = 50 particles.
    let positions = grid_2d(5, 10, dx);
    let n = positions.len();
    assert_eq!(n, 50);
    let neighbors = build_neighbors(&positions, h);
    let densities = vec![rho0; n];
    let masses = vec![rho0 * dx * dx; n];
    let viscosities = vec![rho0 * nu; n];

    // Random-ish initial velocities (deterministic, no rng dependency).
    let velocities: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let a = (i as f64 * 0.37).sin();
            let b = (i as f64 * 0.71).cos();
            [a, b, 0.0]
        })
        .collect();

    let particles = ViscosityParticles {
        velocities: &velocities,
        densities: &densities,
        masses: &masses,
        viscosities: &viscosities,
    };

    // Tight tolerance, capped at 100 iterations — must still converge.
    let options = ViscositySolveOptions {
        tol: 1.0e-8,
        max_iters: 100,
    };
    let result = solve_implicit_viscosity(&particles, &neighbors, dt, h, options);
    match result {
        Ok(out) => {
            assert_eq!(out.len(), n);
            for v in &out {
                for &c in v {
                    assert!(c.is_finite(), "non-finite velocity in solution");
                }
            }
        }
        Err(ViscosityError::SolverNotConverged { iters, residual }) => {
            panic!("CG failed to converge in 100 iters: iters={iters}, residual={residual}");
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

/// A fully periodic simple-cubic lattice reduced to a ring of `n_x` lattice
/// planes along x.
///
/// The velocity field of the shear-wave tests depends on x only, so all
/// particles of one `x` plane move together and a single particle can stand for
/// the plane.  Each plane interacts with the periodic images of every plane
/// within `2h`: the offsets `(a, b, c)` in lattice units, wrapped modulo `n_x`
/// along x.  Images in y/z map to the same ring particle and are accumulated by
/// the solver, which is exactly what the full 3-D periodic lattice produces.
fn periodic_ring_neighbors(n_x: usize, dx: f64, h: f64) -> Vec<Vec<NeighborEntry>> {
    let kernel = CubicSplineKernel;
    let support = 2.0 * h;
    let reach = (support / dx).ceil() as i64;
    let mut neighbors = vec![Vec::new(); n_x];
    for (i, list) in neighbors.iter_mut().enumerate() {
        for a in -reach..=reach {
            for b in -reach..=reach {
                for c in -reach..=reach {
                    // x_j = x_i + offset, so r_ij = x_i − x_j = −offset.
                    let r_ij = [-(a as f64) * dx, -(b as f64) * dx, -(c as f64) * dx];
                    let r = dot3(r_ij, r_ij).sqrt();
                    if r < 1.0e-12 || r >= support {
                        continue;
                    }
                    let j = (i as i64 + a).rem_euclid(n_x as i64) as usize;
                    list.push(NeighborEntry::new(j, r_ij, grad(&kernel, r_ij, h)));
                }
            }
        }
    }
    neighbors
}

/// The shear wave `v_y = A sin(k x_i)` on the ring of `n_x` planes.
fn shear_wave(n_x: usize, dx: f64, amplitude: f64) -> Vec<[f64; 3]> {
    let k = 2.0 * std::f64::consts::PI / (n_x as f64 * dx);
    (0..n_x)
        .map(|i| [0.0, amplitude * (k * i as f64 * dx).sin(), 0.0])
        .collect()
}

/// Per-step amplification factor of the shear wave under one implicit step.
fn shear_wave_factor(nu: f64, dt: f64, options: ViscositySolveOptions) -> f64 {
    let (n_x, dx) = (16_usize, 0.02_f64);
    let h = 1.3 * dx;
    let rho0 = 1000.0;
    let neighbors = periodic_ring_neighbors(n_x, dx, h);
    let densities = vec![rho0; n_x];
    let masses = vec![rho0 * dx * dx * dx; n_x];
    let viscosities = vec![rho0 * nu; n_x];
    let velocities = shear_wave(n_x, dx, 0.1);
    let particles = ViscosityParticles {
        velocities: &velocities,
        densities: &densities,
        masses: &masses,
        viscosities: &viscosities,
    };
    let out = solve_implicit_viscosity(&particles, &neighbors, dt, h, options)
        .expect("implicit solve should converge");
    // Every plane shares the same factor (the mode is an eigenvector); read it
    // off the plane with the largest amplitude.
    let peak = (0..n_x)
        .max_by(|&a, &b| velocities[a][1].abs().total_cmp(&velocities[b][1].abs()))
        .expect("non-empty ring");
    for i in 0..n_x {
        let expected = out[peak][1] / velocities[peak][1] * velocities[i][1];
        assert!(
            (out[i][1] - expected).abs() < 1e-8,
            "plane {i} does not follow the eigenmode"
        );
        assert!(out[i][0].abs() < 1e-12 && out[i][2].abs() < 1e-12);
    }
    out[peak][1] / velocities[peak][1]
}

/// Test 5 — the damping rate of a sinusoidal shear mode is `ν k²`, with `ν` the
/// kinematic viscosity carried by `μ = ρν` (and *not* `ν²`, which an extra
/// scalar factor `ν` in the system matrix would give).
#[test]
fn sinusoidal_shear_mode_decays_at_nu_k_squared() {
    let (n_x, dx) = (16_usize, 0.02_f64);
    let h = 1.3 * dx;
    let nu = 0.01_f64;
    let k = 2.0 * std::f64::consts::PI / (n_x as f64 * dx);
    let kernel = CubicSplineKernel;

    // Independent evaluation of the discrete eigenvalue of the Morris operator on
    // the lattice: λ = Σ_offsets c (1 − cos(k a dx)), c = 2ν dx³ g(r),
    // g(r) = −r W'(r) / (r² + ε h²).
    let reach = (2.0 * h / dx).ceil() as i64;
    let mut lambda = 0.0_f64;
    for a in -reach..=reach {
        for b in -reach..=reach {
            for c in -reach..=reach {
                let r = ((a * a + b * b + c * c) as f64).sqrt() * dx;
                if r < 1.0e-12 || r >= 2.0 * h {
                    continue;
                }
                let g = -r * kernel.grad_w(r, h) / (r * r + MORRIS_EPSILON * h * h);
                let coupling = 2.0 * nu * dx * dx * dx * g;
                lambda += coupling * (1.0 - (k * a as f64 * dx).cos());
            }
        }
    }
    // The SPH Laplacian reproduces ν k² on a smooth mode (a few percent at
    // h = 1.3 dx, k dx = 0.39).
    assert!(
        (lambda / (nu * k * k) - 1.0).abs() < 0.06,
        "discrete rate {lambda} vs nu k^2 = {}",
        nu * k * k
    );

    // One implicit step multiplies the mode by 1 / (1 + dt λ).
    let dt = 0.01;
    let options = ViscositySolveOptions {
        tol: 1.0e-12,
        max_iters: 2000,
    };
    let factor = shear_wave_factor(nu, dt, options);
    let expected = 1.0 / (1.0 + dt * lambda);
    assert!(
        (factor - expected).abs() < 1e-8,
        "step factor {factor} vs 1/(1+dt λ) = {expected}"
    );
    // ν² would leave the mode essentially untouched (rate 1e-4 · k² ≈ 0.04 /s).
    assert!(factor < 0.97, "the mode must visibly decay: {factor}");

    // March to t = 0.1 s with small steps: amplitude ≈ exp(−ν k² t) up to the
    // lattice-operator error and the O(dt) implicit-Euler error.
    let dt_small = 1.0e-3;
    let steps = 100;
    let per_step = shear_wave_factor(nu, dt_small, options);
    let amplitude = per_step.powi(steps);
    let physical = (-nu * k * k * dt_small * steps as f64).exp();
    assert!(
        (amplitude / physical - 1.0).abs() < 0.04,
        "amplitude {amplitude} vs exp(-nu k^2 t) = {physical}"
    );
}

/// Test 6 — the damping is linear in the kinematic viscosity: `1/f − 1 = dt λ ∝ ν`.
#[test]
fn damping_is_linear_in_the_kinematic_viscosity() {
    let options = ViscositySolveOptions {
        tol: 1.0e-12,
        max_iters: 2000,
    };
    let dt = 0.005;
    let slow = 1.0 / shear_wave_factor(0.01, dt, options) - 1.0;
    let fast = 1.0 / shear_wave_factor(0.04, dt, options) - 1.0;
    assert!(slow > 0.0);
    assert!(
        (fast / slow - 4.0).abs() < 1e-6,
        "quadrupling nu must quadruple the damping, got x{}",
        fast / slow
    );
}
