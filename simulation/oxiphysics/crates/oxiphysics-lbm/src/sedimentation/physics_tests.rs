// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Physics regression tests for the sedimentation module: the mixture
//! viscosity of [`SediLbm`], the conservative hindered settling of
//! [`Sedimentation3D`] and the lattice-unit momentum-exchange coupling of
//! [`LbmDemSolver`].

use super::functions::{
    EY9, W9, column_face_fluxes, feq_d2q9, godunov_batch_flux, mixture_viscosity,
};
use super::types::*;

const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// SediLbm: mixture viscosity, body forces
// ---------------------------------------------------------------------------

/// A denser suspension is more viscous: the relaxation frequency must fall
/// with the solid fraction and `(1/omega_mix - 1/2) / (1/omega - 1/2)` must be
/// the Krieger-Dougherty relative viscosity.
#[test]
fn mixture_omega_decreases_with_particle_loading() {
    let omega = 1.7;
    let sim = SediLbm::new(4, 4, omega, 1.0, 2.5, 0.0);
    let omega_clean = sim.mixture_omega(0.0);
    assert!((omega_clean - omega).abs() < 1e-14);

    let mut previous = omega_clean;
    for k in 1..=10 {
        let phi = 0.06 * f64::from(k);
        let current = sim.mixture_omega(phi);
        assert!(
            current > 0.0 && current < previous,
            "omega({phi}) = {current} must be below {previous}"
        );
        previous = current;
    }

    let omega_loaded = sim.mixture_omega(0.3);
    assert!(omega_loaded < omega_clean);
    let relative_viscosity = (1.0 / omega_loaded - 0.5) / (1.0 / omega - 0.5);
    assert!(relative_viscosity > 1.0);
    assert!(
        (relative_viscosity - mixture_viscosity(0.3, 0.64)).abs() < 1e-12,
        "relative viscosity {relative_viscosity}"
    );
}

/// Amplitude of the `sin(2 pi y / ny)` mode of `u_x`.
fn shear_mode_amplitude(sim: &SediLbm) -> f64 {
    let k = 2.0 * PI / sim.ny as f64;
    let mut acc = 0.0;
    for (y, row) in sim.u.iter().enumerate() {
        for cell in row {
            acc += cell[0] * (k * y as f64).sin();
        }
    }
    2.0 * acc / (sim.nx * sim.ny) as f64
}

/// Exponential decay rate per step of a sinusoidal shear wave in a uniformly
/// loaded suspension (no gravity).
fn shear_decay_rate(phi: f64, steps: usize) -> f64 {
    let (nx, ny, omega) = (4, 32, 1.5);
    let mut sim = SediLbm::new(nx, ny, omega, 1.0, 2.5, 0.0);
    sim.set_uniform_phi(phi);
    let k = 2.0 * PI / ny as f64;
    for y in 0..ny {
        let ux = 0.02 * (k * y as f64).sin();
        for x in 0..nx {
            sim.u[y][x] = [ux, 0.0];
            sim.f[y][x] = feq_d2q9(1.0, ux, 0.0).to_vec();
        }
    }
    let a0 = shear_mode_amplitude(&sim);
    for _ in 0..steps {
        sim.step();
    }
    let a1 = shear_mode_amplitude(&sim);
    (a0 / a1).ln() / steps as f64
}

/// End-to-end check of the viscosity direction: the shear wave decays faster
/// in the loaded suspension, by the Krieger-Dougherty factor.
#[test]
fn shear_wave_decays_faster_in_denser_suspension() {
    let rate_clean = shear_decay_rate(0.0, 100);
    let rate_loaded = shear_decay_rate(0.3, 100);

    let nu_clean = (1.0 / 1.5 - 0.5) / 3.0;
    let k = 2.0 * PI / 32.0;
    let expected_clean = nu_clean * k * k;
    assert!(
        (rate_clean - expected_clean).abs() < 0.08 * expected_clean,
        "clean rate {rate_clean}, expected nu k^2 = {expected_clean}"
    );

    let factor = mixture_viscosity(0.3, 0.64);
    assert!(rate_loaded > rate_clean);
    assert!(
        (rate_loaded / rate_clean - factor).abs() < 0.05 * factor,
        "rate ratio {} vs viscosity factor {factor}",
        rate_loaded / rate_clean
    );
}

/// Collision with a spatially varying solid fraction (which defeats the
/// cached relaxation frequency on every cell) matches the per-cell formula.
#[test]
fn collide_with_varying_phi_matches_per_cell_reference() {
    let (nx, ny, g_latt) = (6, 4, 2.0e-4);
    let mut sim = SediLbm::new(nx, ny, 1.4, 1.0, 2.5, g_latt);
    for y in 0..ny {
        for x in 0..nx {
            sim.phi[y][x] = match (x + y) % 3 {
                0 => 0.0,
                1 => 0.2,
                _ => 0.45,
            };
            let ux = 0.01 * x as f64 - 0.02;
            let uy = 0.005 * y as f64;
            sim.u[y][x] = [ux, uy];
            sim.rho[y][x] = 1.0 + 0.01 * (x * y) as f64;
            // Deliberately off-equilibrium so that the relaxation matters.
            sim.f[y][x] = feq_d2q9(sim.rho[y][x], 0.5 * ux, 2.0 * uy).to_vec();
        }
    }
    let before = sim.clone();
    sim.collide();
    for y in 0..ny {
        for x in 0..nx {
            let omega = before.mixture_omega(before.phi[y][x]);
            let feq = feq_d2q9(before.rho[y][x], before.u[y][x][0], before.u[y][x][1]);
            let buoy = (2.5 - 1.0) * before.phi[y][x] * g_latt;
            for q in 0..9 {
                let force = 3.0 * W9[q] * EY9[q] as f64 * (-buoy);
                let expected = before.f[y][x][q] * (1.0 - omega) + omega * feq[q] + force;
                assert!(
                    (sim.f[y][x][q] - expected).abs() < 1e-14,
                    "cell ({x}, {y}) direction {q}"
                );
            }
        }
    }
}

/// The external body force field injects exactly `F` of momentum per step.
#[test]
fn body_force_injects_exact_momentum() {
    let mut sim = SediLbm::new(8, 8, 1.3, 1.0, 2.5, 0.0);
    sim.body_force[3][5] = [2.0e-4, -3.0e-4];
    sim.step();
    let mut momentum = [0.0f64; 2];
    for (rho_row, u_row) in sim.rho.iter().zip(sim.u.iter()) {
        for (rho, u) in rho_row.iter().zip(u_row.iter()) {
            momentum[0] += rho * u[0];
            momentum[1] += rho * u[1];
        }
    }
    assert!((momentum[0] - 2.0e-4).abs() < 1e-14, "px = {}", momentum[0]);
    assert!((momentum[1] + 3.0e-4).abs() < 1e-14, "py = {}", momentum[1]);

    sim.clear_body_force();
    assert!(sim.body_force.iter().flatten().all(|f| *f == [0.0, 0.0]));
}

// ---------------------------------------------------------------------------
// Sedimentation3D: conservative hindered settling
// ---------------------------------------------------------------------------

/// A single-column tank (`nx = nz = 1`, so the flat index equals `y`).
fn column(ny: usize, dt: f64, dz: f64) -> Sedimentation3D {
    Sedimentation3D::new([1, ny, 1], 1.0e-4, 2500.0, 1000.0, 1.0e-3, dt, dz)
}

/// Height (m) where the linearly interpolated profile of a single column
/// falls through `threshold`, searching down from the top of the tank.
fn interface_height(sim: &Sedimentation3D, threshold: f64) -> f64 {
    let ny = sim.dims[1];
    let top = (0..ny)
        .rev()
        .find(|&j| sim.phi[j] >= threshold)
        .expect("the suspension must not be empty");
    let phi_top = sim.phi[top];
    let phi_above = if top + 1 < ny { sim.phi[top + 1] } else { 0.0 };
    let fraction = (phi_top - threshold) / (phi_top - phi_above);
    (top as f64 + 0.5 + fraction) * sim.dz
}

/// The sum of the volume fractions is conserved to round-off, `phi` stays in
/// `[0, 0.64]`, the closed bottom never loses solids, and the top loses them.
#[test]
fn sedimentation3d_conserves_solids_and_stays_bounded() {
    let (nx, ny, nz) = (3usize, 40usize, 2usize);
    let mut sim = Sedimentation3D::new([nx, ny, nz], 1.0e-4, 2500.0, 1000.0, 1.0e-3, 0.05, 1.0e-3);
    for z in 0..nz {
        for y in 0..ny {
            let bump = (-((y as f64 - 20.0) / 6.0).powi(2)).exp();
            for x in 0..nx {
                sim.phi[z * ny * nx + y * nx + x] = 0.02 + 0.4 * bump;
            }
        }
    }
    let total0: f64 = sim.phi.iter().sum();
    let bottom0 = sim.phi[0];
    let top0 = sim.phi[(ny - 1) * nx];
    let mut bottom_previous = bottom0;
    for step in 0..100 {
        sim.step();
        for &phi in &sim.phi {
            assert!(
                (-1e-15..=0.64 + 1e-12).contains(&phi),
                "step {step}: phi = {phi} outside [0, 0.64]"
            );
        }
        // Nothing can leave through the closed bottom.
        assert!(sim.phi[0] >= bottom_previous - 1e-15, "bottom lost solids");
        bottom_previous = sim.phi[0];
    }
    let total1: f64 = sim.phi.iter().sum();
    assert!(
        (total1 - total0).abs() < 1e-12 * total0,
        "solids not conserved: {total0} -> {total1}"
    );
    assert!(sim.phi[0] > bottom0, "bottom cell must collect solids");
    assert!(sim.phi[(ny - 1) * nx] < top0, "top cell must lose solids");
}

/// The top interface of a uniform suspension descends at the hindered
/// settling velocity `v_t (1 - phi)^n`, and the plateau below it is untouched.
#[test]
fn sedimentation3d_top_interface_descends_at_hindered_velocity() {
    let (ny, dz, dt) = (200usize, 1.0e-3, 0.05);
    let mut sim = column(ny, dt, dz);
    let phi0: f64 = 0.1;
    let vt = sim.vt_single();
    assert!(vt * dt / dz < 0.5, "CFL number {} too large", vt * dt / dz);
    let expected = vt * (1.0 - phi0).powf(sim.rz_exp);

    let steps = 150;
    for _ in 0..steps {
        sim.step();
    }
    let elapsed = steps as f64 * dt;
    let height = interface_height(&sim, 0.5 * phi0);
    let measured = (ny as f64 * dz - height) / elapsed;
    assert!(
        (measured - expected).abs() < 0.05 * expected,
        "interface speed {measured}, expected {expected}"
    );
    // Well below the front the suspension keeps its initial concentration.
    assert!(
        (sim.phi[100] - phi0).abs() < 1e-12,
        "phi = {}",
        sim.phi[100]
    );
    // The sediment collects at the bottom, the clear liquid above the front.
    assert!(sim.phi[0] > phi0);
    assert!(sim.phi[ny - 1] < 1e-6, "phi_top = {}", sim.phi[ny - 1]);
}

/// A dense column compacts into a bed that never exceeds the maximum packing.
#[test]
fn sedimentation3d_never_exceeds_maximum_packing() {
    let mut sim = column(30, 0.05, 1.0e-3);
    sim.phi.fill(0.6);
    let total0: f64 = sim.phi.iter().sum();
    for step in 0..400 {
        sim.step();
        for &phi in &sim.phi {
            assert!(
                (-1e-15..=0.64 + 1e-12).contains(&phi),
                "step {step}: phi = {phi}"
            );
        }
    }
    let total1: f64 = sim.phi.iter().sum();
    assert!((total1 - total0).abs() < 1e-12 * total0);
    assert!(
        (sim.phi[0] - 0.64).abs() < 1e-9,
        "bottom cell should be a packed bed: {}",
        sim.phi[0]
    );
}

/// Light particles (`rho_p < rho_f`) cream: they rise to the closed top, and
/// the donor-cell scheme must follow the upward motion without going negative.
#[test]
fn sedimentation3d_creaming_rises_to_closed_top() {
    let mut sim = Sedimentation3D::new([1, 60, 1], 1.0e-4, 900.0, 1000.0, 1.0e-3, 0.05, 1.0e-3);
    assert!(sim.is_creaming());
    assert!(sim.vt_single() < 0.0);
    let total0: f64 = sim.phi.iter().sum();
    let (bottom0, top0) = (sim.phi[0], sim.phi[59]);
    for step in 0..200 {
        sim.step();
        for &phi in &sim.phi {
            assert!(
                (-1e-15..=0.64 + 1e-12).contains(&phi),
                "step {step}: phi = {phi}"
            );
        }
    }
    let total1: f64 = sim.phi.iter().sum();
    assert!((total1 - total0).abs() < 1e-12 * total0);
    assert!(sim.phi[59] > top0, "cream must collect at the top");
    assert!(sim.phi[0] < bottom0, "the bottom must be depleted");
}

/// With a single cell per column both faces are closed walls: nothing moves.
#[test]
fn sedimentation3d_single_cell_column_is_closed() {
    let mut sim = column(1, 0.05, 1.0e-3);
    for _ in 0..10 {
        sim.step();
    }
    assert_eq!(sim.phi, vec![0.1]);
}

/// A non-positive cell height or time step is rejected without producing NaN.
#[test]
fn sedimentation3d_non_positive_spacing_leaves_field_unchanged() {
    let mut zero_dz = column(10, 0.05, 0.0);
    zero_dz.step();
    assert!(zero_dz.phi.iter().all(|&p| p == 0.1));
    let mut negative_dt = column(10, -0.05, 1.0e-3);
    negative_dt.step();
    assert!(negative_dt.phi.iter().all(|&p| p == 0.1));
}

/// Kynch batch flux `q(phi) = phi (1 - phi)^n` with `v_t = 1`.
fn batch_flux(phi: f64, n: i32) -> f64 {
    phi * (1.0 - phi).powi(n)
}

/// The Godunov flux reduces to the donor-cell flux for dilute states, takes
/// the receiving cell's flux for dense states and the maximum flux across the
/// peak of the batch flux.
#[test]
fn godunov_batch_flux_covers_all_regimes() {
    let n = 4;
    let peak = 1.0 / f64::from(n + 1);
    let q_peak = batch_flux(peak, n);
    let q = |phi: f64| batch_flux(phi, n);
    let flux = |from: f64, to: f64| godunov_batch_flux(from, to, q(from), q(to), peak, q_peak);

    let same = |a: f64, b: f64| (a - b).abs() < 1e-15;

    // Equal states carry their own flux.
    for phi in [0.05, 0.2, 0.5] {
        assert!(same(flux(phi, phi), q(phi)));
    }
    // Dilute (both below the peak): the flux of the cell the solids come from.
    assert!(same(flux(0.05, 0.15), q(0.05)));
    assert!(same(flux(0.15, 0.05), q(0.15)));
    // Dense (both above the peak): the flux of the receiving cell.
    assert!(same(flux(0.3, 0.5), q(0.5)));
    assert!(same(flux(0.5, 0.3), q(0.3)));
    // Dense source over a dilute receiver: the maximum batch flux.
    assert!(same(flux(0.4, 0.1), q_peak));
    // Dilute source over a dense receiver: the smaller of the two fluxes.
    assert!(same(flux(0.1, 0.4), q(0.4).min(q(0.1))));
    assert!(q_peak >= q(0.1) && q_peak >= q(0.4));
}

/// Face fluxes of a settling column: closed walls, a packed cell that cannot
/// accept more, an empty source cell, and the Godunov flux in between.
#[test]
fn column_face_fluxes_apply_walls_and_limiters() {
    let n = 4;
    let peak = 1.0 / f64::from(n + 1);
    let peak_pair = (peak, batch_flux(peak, n));
    let phi: [f64; 4] = [0.64, 0.30, 0.10, 0.0];
    let vs: Vec<f64> = phi.iter().map(|p| (1.0 - p).powi(n)).collect();
    let mut flux = [9.0; 5];
    column_face_fluxes(&phi, &vs, 0.1, peak_pair, true, &mut flux);
    assert_eq!(flux[0], 0.0, "closed bottom");
    assert_eq!(flux[4], 0.0, "closed top");
    assert_eq!(flux[1], 0.0, "the packed bed cannot accept more solids");
    assert!(
        (flux[2] - batch_flux(0.10, n)).abs() < 1e-15,
        "dilute source feeding a denser cell: {}",
        flux[2]
    );
    assert_eq!(flux[3], 0.0, "an empty source cell sends nothing");

    // A source cell can never give away more than it holds in one step.
    let phi = [0.0, 0.2];
    let vs = [1.0, 1.0];
    let mut flux = [0.0; 3];
    column_face_fluxes(&phi, &vs, 100.0, (0.2, 0.1), true, &mut flux);
    assert!((flux[1] - 0.2 / 100.0).abs() < 1e-15, "flux {}", flux[1]);
}

/// Creaming is the mirror image of settling: reversing the column and the
/// sign of the velocity reverses and negates the face fluxes.
#[test]
fn column_face_fluxes_creaming_mirrors_settling() {
    let n = 4;
    let peak = 1.0 / f64::from(n + 1);
    let peak_pair = (peak, batch_flux(peak, n));
    // Deterministic pseudo-random column with dilute, dense and packed cells.
    let mut seed = 12345u64;
    let phi: Vec<f64> = (0..16)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.64 * ((seed >> 33) as f64 / (1u64 << 31) as f64)
        })
        .collect();
    let vs: Vec<f64> = phi.iter().map(|p| (1.0 - p).powi(n)).collect();
    let mut settling = vec![0.0; 17];
    column_face_fluxes(&phi, &vs, 0.4, peak_pair, true, &mut settling);

    let phi_rev: Vec<f64> = phi.iter().rev().copied().collect();
    let vs_rev: Vec<f64> = vs.iter().rev().map(|v| -v).collect();
    let mut creaming = vec![0.0; 17];
    column_face_fluxes(&phi_rev, &vs_rev, 0.4, peak_pair, false, &mut creaming);
    for j in 0..=16 {
        assert!(
            (creaming[16 - j] + settling[j]).abs() < 1e-15,
            "face {j}: {} vs {}",
            creaming[16 - j],
            settling[j]
        );
    }
    assert!(settling.iter().any(|&f| f > 0.0));
}

/// Dense suspensions (above the batch-flux peak `1 / (n + 1)`) settle without
/// oscillations: the profile between bed and front is monotone, the top
/// interface keeps the Kynch shock speed `v_t (1 - phi_0)^n`, the bed grows
/// at `phi_0 v_s / (phi_max - phi_0)` and the solids are conserved.  A plain
/// donor-cell flux saw-tooths between 0 and 0.64 here.
#[test]
fn sedimentation3d_dense_suspensions_settle_monotonically() {
    let (ny, dz, dt, steps) = (150usize, 1.0e-3, 0.05, 300);
    for phi0 in [0.1, 0.25, 0.3, 0.4] {
        let mut sim = column(ny, dt, dz);
        sim.phi.fill(phi0);
        let total0: f64 = sim.phi.iter().sum();
        for _ in 0..steps {
            sim.step();
        }
        let elapsed = steps as f64 * dt;
        let total1: f64 = sim.phi.iter().sum();
        assert!(
            (total1 - total0).abs() < 1e-12 * total0,
            "phi0 = {phi0}: solids not conserved"
        );
        // Bed (bottom), uniform suspension, clear liquid (top): non-increasing.
        for j in 1..ny {
            assert!(
                sim.phi[j] <= sim.phi[j - 1] + 1e-12,
                "phi0 = {phi0}: profile oscillates at cell {j}: {} > {}",
                sim.phi[j],
                sim.phi[j - 1]
            );
        }
        assert!(
            sim.phi
                .iter()
                .all(|&p| (-1e-15..=0.64 + 1e-12).contains(&p))
        );

        let v_front = sim.vt_single() * (1.0 - phi0).powf(sim.rz_exp);
        let front = (ny as f64 * dz - interface_height(&sim, 0.5 * phi0)) / elapsed;
        assert!(
            (front - v_front).abs() < 0.05 * v_front,
            "phi0 = {phi0}: front speed {front} vs {v_front}"
        );

        // Bed top: where the profile falls through the mean of bed and suspension.
        let threshold = 0.5 * (0.64 + phi0);
        let first_below = (0..ny)
            .find(|&j| sim.phi[j] < threshold)
            .expect("the suspension must lie above the bed");
        let bed_top = (first_below as f64 - 0.5
            + (sim.phi[first_below - 1] - threshold)
                / (sim.phi[first_below - 1] - sim.phi[first_below]))
            * dz;
        let v_bed = phi0 * v_front / (0.64 - phi0);
        assert!(
            (bed_top / elapsed - v_bed).abs() < 0.12 * v_bed,
            "phi0 = {phi0}: bed growth {} vs {v_bed}",
            bed_top / elapsed
        );
    }
}

/// Dense creaming: the clear-liquid interface rises from the bottom at the
/// hindered velocity and the cream layer packs against the closed top, with a
/// monotone (non-decreasing) profile.
#[test]
fn sedimentation3d_dense_creaming_rises_monotonically() {
    let (ny, dz, dt, steps) = (100usize, 1.0e-3, 0.5, 400);
    let mut sim = Sedimentation3D::new([1, ny, 1], 1.0e-4, 900.0, 1000.0, 1.0e-3, dt, dz);
    let phi0 = 0.3;
    sim.phi.fill(phi0);
    let total0: f64 = sim.phi.iter().sum();
    for _ in 0..steps {
        sim.step();
    }
    let elapsed = steps as f64 * dt;
    let total1: f64 = sim.phi.iter().sum();
    assert!((total1 - total0).abs() < 1e-12 * total0);
    for j in 1..ny {
        assert!(
            sim.phi[j] >= sim.phi[j - 1] - 1e-12,
            "profile oscillates at cell {j}: {} < {}",
            sim.phi[j],
            sim.phi[j - 1]
        );
    }
    assert!(
        sim.phi
            .iter()
            .all(|&p| (-1e-15..=0.64 + 1e-12).contains(&p))
    );
    assert!(
        (sim.phi[ny - 1] - 0.64).abs() < 1e-9,
        "cream layer not packed"
    );

    // Clear-liquid interface: first cell (from the bottom) above half the load.
    let threshold = 0.5 * phi0;
    let first_above = (0..ny)
        .find(|&j| sim.phi[j] >= threshold)
        .expect("the suspension must be present");
    let below = if first_above == 0 {
        0.0
    } else {
        sim.phi[first_above - 1]
    };
    let height =
        (first_above as f64 - 0.5 + (threshold - below) / (sim.phi[first_above] - below)) * dz;
    let v_expected = sim.vt_single().abs() * (1.0 - phi0).powf(sim.rz_exp);
    assert!(
        (height / elapsed - v_expected).abs() < 0.05 * v_expected,
        "interface rise {} vs {v_expected}",
        height / elapsed
    );
}

// ---------------------------------------------------------------------------
// LbmDemSolver: lattice units and momentum-exchange coupling
// ---------------------------------------------------------------------------

/// A periodic `n x n` solver for a fluid of density `rho_fluid` and dynamic
/// viscosity `mu`; the time step follows from `nu_phys = nu_lat dx^2 / dt`.
fn make_solver(
    n: usize,
    omega: f64,
    mu: f64,
    rho_fluid: f64,
    rho_solid: f64,
    dx: f64,
) -> LbmDemSolver {
    let nu_lat = (1.0 / omega - 0.5) / 3.0;
    let nu_phys = mu / rho_fluid;
    let dt = nu_lat * dx * dx / nu_phys;
    LbmDemSolver::new(n, n, omega, rho_fluid, rho_solid, dx, dt)
}

/// Total momentum (kg m/s) of the fluid and the particles.
fn total_momentum(solver: &LbmDemSolver) -> [f64; 2] {
    // mass unit rho dx^3 times velocity unit dx / dt
    let unit = solver.rho_fluid * solver.dx.powi(4) / solver.dt;
    let mut momentum = [0.0f64; 2];
    for (rho_row, u_row) in solver.lbm.rho.iter().zip(solver.lbm.u.iter()) {
        for (rho, u) in rho_row.iter().zip(u_row.iter()) {
            momentum[0] += rho * u[0] * unit;
            momentum[1] += rho * u[1] * unit;
        }
    }
    for p in &solver.particles {
        momentum[0] += p.mass() * p.velocity[0];
        momentum[1] += p.mass() * p.velocity[1];
    }
    momentum
}

/// The fluid runs in lattice units: unit reference density, density ratio for
/// the solid, no SI density or gravity fed into the collision.
#[test]
fn lbm_dem_solver_runs_in_lattice_units() {
    let mut solver = make_solver(12, 1.25, 1.0e-3, 1000.0, 2500.0, 1.0e-3);
    assert_eq!(solver.lbm.rho_fluid, 1.0);
    assert_eq!(solver.lbm.rho_solid, 2.5);
    assert_eq!(solver.lbm.gravity, 0.0);
    solver.add_particle(DemParticle::new([6.5e-3, 6.5e-3, 0.0], 1.0e-5, 2500.0, 0));
    for _ in 0..5 {
        solver.step();
    }
    for &rho in solver.lbm.rho.iter().flatten() {
        assert!((rho - 1.0).abs() < 1e-2, "lattice density {rho}");
    }
    for u in solver.lbm.u.iter().flatten() {
        assert!(u[0].abs() < 1e-2 && u[1].abs() < 1e-2, "u = {u:?}");
    }
}

/// `hydro_force` converts the lattice fluid velocity and viscosity to SI:
/// `F = 6 pi mu r (u_f - v_p)` with the physical `mu`.
#[test]
fn lbm_dem_hydro_force_uses_si_viscosity_and_velocity() {
    let (mu, dx, radius) = (2.0e-3, 1.0e-3, 2.0e-5);
    let mut solver = make_solver(8, 1.25, mu, 1000.0, 2500.0, dx);
    for u in solver.lbm.u.iter_mut().flatten() {
        *u = [0.01, -0.02];
    }
    let mut p = DemParticle::new([4.5e-3, 4.5e-3, 0.0], radius, 2500.0, 0);
    p.velocity = [1.0e-4, 5.0e-5, 0.0];
    let force = solver.hydro_force(&p);
    let to_si = dx / solver.dt;
    let expected = [
        6.0 * PI * mu * radius * (0.01 * to_si - 1.0e-4),
        6.0 * PI * mu * radius * (-0.02 * to_si - 5.0e-5),
    ];
    for k in 0..2 {
        assert!(
            (force[k] - expected[k]).abs() < 1e-9 * expected[k].abs(),
            "component {k}: {} vs {}",
            force[k],
            expected[k]
        );
    }
}

/// Without gravity the drag exchanges momentum between the particle and the
/// fluid, and the total momentum stays constant to round-off.
#[test]
fn lbm_dem_momentum_exchange_conserves_total_momentum() {
    let dx = 1.0e-3;
    let mut solver = make_solver(16, 1.25, 1.0e-3, 1000.0, 2500.0, dx);
    solver.set_gravity([0.0; 3]);
    let mut p = DemParticle::new([8.3 * dx, 8.6 * dx, 0.0], 5.0e-5, 2500.0, 0);
    p.velocity = [2.0e-4, -1.0e-4, 0.0];
    solver.add_particle(p);

    let before = total_momentum(&solver);
    assert!(before[0] > 0.0);
    for step in 0..40 {
        solver.step();
        let now = total_momentum(&solver);
        for k in 0..2 {
            assert!(
                (now[k] - before[k]).abs() < 1e-8 * before[0].abs(),
                "step {step}, component {k}: {} vs {}",
                now[k],
                before[k]
            );
        }
    }
    // The particle really handed its momentum to the fluid.
    let particle = &solver.particles[0];
    let particle_px = particle.mass() * particle.velocity[0];
    assert!(
        particle_px < 0.1 * before[0],
        "particle still carries {particle_px} of {}",
        before[0]
    );
}

/// With gravity on, the total momentum grows by exactly the particle's
/// buoyancy-reduced weight times time: the weight reaches the fluid once, via
/// the drag reaction, not a second time through the mixture buoyancy.
#[test]
fn lbm_dem_weight_is_transmitted_to_the_fluid_exactly_once() {
    let (dx, rho_f, rho_s) = (1.0e-3, 1000.0, 2500.0);
    let mut solver = make_solver(24, 1.25, 1.0e-3, rho_f, rho_s, dx);
    let radius = 4.0e-6;
    solver.add_particle(DemParticle::new(
        [12.5 * dx, 18.5 * dx, 0.0],
        radius,
        rho_s,
        0,
    ));
    let mass = solver.particles[0].mass();
    let steps = 60;
    for _ in 0..steps {
        solver.step();
    }
    let momentum = total_momentum(&solver);
    let impulse = steps as f64 * solver.dt * mass * (-9.81) * (1.0 - rho_f / rho_s);
    assert!(
        (momentum[1] - impulse).abs() < 1e-6 * impulse.abs(),
        "momentum {} vs weight impulse {impulse}",
        momentum[1]
    );
    assert!(momentum[0].abs() < 1e-6 * impulse.abs());
}

/// A single small particle settles at the Stokes velocity
/// `2 (rho_s - rho_f) g r^2 / (9 mu)`: the slip velocity against the host
/// cell reproduces it, and the particle velocity stays within 10 %.
#[test]
fn lbm_dem_single_particle_reaches_stokes_velocity() {
    let (dx, mu, rho_f, rho_s) = (1.0e-3, 1.0e-3, 1000.0, 2500.0);
    let mut solver = make_solver(32, 1.25, mu, rho_f, rho_s, dx);
    let radius = 4.0e-6; // r / dx = 0.004: a point particle on the lattice
    solver.add_particle(DemParticle::new(
        [16.5 * dx, 24.5 * dx, 0.0],
        radius,
        rho_s,
        0,
    ));
    let v_stokes = 2.0 * (rho_s - rho_f) * 9.81 * radius * radius / (9.0 * mu);
    assert!(
        v_stokes * solver.dt / dx < 0.02,
        "lattice Mach number too high"
    );

    let steps = 200;
    for _ in 0..steps - 1 {
        solver.step();
    }
    let (ix, iy) = solver.grid_cell(solver.particles[0].position);
    let fluid_before = solver.lbm.u[iy][ix][1] * dx / solver.dt;
    solver.step();

    let particle = &solver.particles[0];
    let slip = fluid_before - particle.velocity[1];
    assert!(
        (slip - v_stokes).abs() < 0.01 * v_stokes,
        "slip {slip} vs Stokes {v_stokes}"
    );
    assert!(
        (-particle.velocity[1] - v_stokes).abs() < 0.10 * v_stokes,
        "settling speed {} vs Stokes {v_stokes}",
        -particle.velocity[1]
    );
    assert!(particle.position[1] < 24.5 * dx, "particle must have sunk");
    assert!(
        particle.position[1] > 20.0 * dx,
        "particle stayed in the tank"
    );
}

/// The physical gravity vector drives the particle and `set_gravity` keeps
/// the derived lattice value in sync.
#[test]
fn lbm_dem_gravity_vector_drives_the_particle() {
    let (dx, mu, rho_f, rho_s) = (1.0e-3, 1.0e-3, 1000.0, 2500.0);
    let radius = 4.0e-6;
    let v_stokes_per_g = 2.0 * (rho_s - rho_f) * radius * radius / (9.0 * mu);

    let mut solver = make_solver(16, 1.25, mu, rho_f, rho_s, dx);
    solver.set_gravity([2.0, 0.0, 0.0]);
    let to_lattice = solver.dt * solver.dt / dx;
    assert!((solver.gravity_latt[0] - 2.0 * to_lattice).abs() < 1e-18);
    assert_eq!(solver.gravity_latt[1], 0.0);
    solver.add_particle(DemParticle::new(
        [8.5 * dx, 8.5 * dx, 0.0],
        radius,
        rho_s,
        0,
    ));
    solver.step();
    let p = &solver.particles[0];
    assert!(
        (p.velocity[0] - 2.0 * v_stokes_per_g).abs() < 0.01 * 2.0 * v_stokes_per_g,
        "vx = {}",
        p.velocity[0]
    );
    assert!(p.velocity[1].abs() < 1e-6 * p.velocity[0].abs());

    // Without gravity nothing moves.
    let mut still = make_solver(16, 1.25, mu, rho_f, rho_s, dx);
    still.set_gravity([0.0; 3]);
    still.add_particle(DemParticle::new(
        [8.5 * dx, 8.5 * dx, 0.0],
        radius,
        rho_s,
        0,
    ));
    for _ in 0..10 {
        still.step();
    }
    for k in 0..3 {
        assert!(still.particles[0].velocity[k].abs() < 1e-12, "v[{k}]");
    }
    assert!(
        still
            .lbm
            .u
            .iter()
            .flatten()
            .all(|u| u[0].abs() < 1e-14 && u[1].abs() < 1e-14)
    );
}

/// The reaction force is deposited only in the host cell of the particle, with
/// the sign opposite to the drag and the lattice scaling `dt^2 / (rho dx^4)`.
#[test]
fn lbm_dem_reaction_force_is_deposited_in_the_host_cell() {
    let (dx, mu, rho_f, rho_s) = (1.0e-3, 1.0e-3, 1000.0, 2500.0);
    let mut solver = make_solver(12, 1.25, mu, rho_f, rho_s, dx);
    solver.set_gravity([0.0; 3]);
    let mut p = DemParticle::new([5.5 * dx, 6.5 * dx, 0.0], 1.0e-5, rho_s, 0);
    p.velocity = [1.0e-4, 0.0, 0.0];
    let before = p.velocity[0];
    solver.add_particle(p);
    solver.step();

    // Impulse the particle lost over the step, converted to lattice force density.
    let lost = solver.particles[0].mass() * (before - solver.particles[0].velocity[0]) / solver.dt;
    let expected = lost * solver.dt * solver.dt / (rho_f * dx.powi(4));
    let force = solver.lbm.body_force[6][5];
    assert!(expected > 0.0);
    assert!(
        (force[0] - expected).abs() < 1e-9 * expected,
        "fx = {}, expected {expected}",
        force[0]
    );
    assert!(force[1].abs() < 1e-12 * expected);
    for (y, row) in solver.lbm.body_force.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            if (x, y) != (5, 6) {
                assert_eq!(*cell, [0.0, 0.0], "cell ({x}, {y})");
            }
        }
    }
}

/// For a particle whose response time is much longer than the time step the
/// implicit drag used by `step` agrees with the explicit Stokes law
/// `hydro_force`: `v+ = v + dt F / m` up to `O((dt / tau_p)^2)`.
#[test]
fn lbm_dem_step_matches_explicit_drag_for_slow_particles() {
    let (dx, mu, rho_f, rho_s) = (1.0e-3, 1.0e-3, 1000.0, 2.5e6);
    let mut solver = make_solver(16, 1.25, mu, rho_f, rho_s, dx);
    solver.set_gravity([0.0; 3]);
    let radius = 4.2e-4;
    let mut p = DemParticle::new([8.5 * dx, 8.5 * dx, 0.0], radius, rho_s, 0);
    p.velocity = [1.0e-3, 0.0, 0.0];
    let force = solver.hydro_force(&p);
    let response_time = p.mass() / (6.0 * PI * mu * radius);
    assert!(
        solver.dt / response_time < 2.0e-3,
        "particle is not slow enough"
    );
    let expected = p.velocity[0] + solver.dt * force[0] / p.mass();
    assert!(force[0] < 0.0, "the drag must oppose the motion");

    solver.add_particle(p);
    solver.step();
    let v_new = solver.particles[0].velocity[0];
    assert!(
        (v_new - expected).abs() < 1.0e-5 * 1.0e-3,
        "implicit {v_new} vs explicit {expected}"
    );
    assert!(v_new < 1.0e-3 && v_new > 0.99e-3);
}
