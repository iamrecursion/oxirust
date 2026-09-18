// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Lorentz-force curl and the vector-distribution
//! (Dellar 2002) MHD-LBM solver.

use super::*;

// ---------------------------------------------------------------------------
// LorentzForce (F2598)
// ---------------------------------------------------------------------------

/// Evaluate `compute` on the five-point stencil of a field given as a function `b(x, y)`.
fn lorentz_on_field(
    lf: &LorentzForce,
    b: impl Fn(f64, f64) -> [f64; 3],
    x: f64,
    y: f64,
) -> ([f64; 3], [f64; 3]) {
    let dx = lf.dx;
    let force = lf.compute(
        b(x, y),
        b(x + dx, y),
        b(x - dx, y),
        b(x, y + dx),
        b(x, y - dx),
    );
    let current = lf.current_density(b(x + dx, y), b(x - dx, y), b(x, y + dx), b(x, y - dx));
    (force, current)
}

#[test]
fn transverse_field_gradient_gives_tension_force() {
    // B = (0, x, 0): J = (0, 0, 1/mu), F = J x B = (-By/mu, 0, 0).
    let mu = 2.0;
    let lf = LorentzForce::new(mu, 0.1);
    let x0 = 0.7;
    let (force, current) = lorentz_on_field(&lf, |x, _| [0.0, x, 0.0], x0, 0.3);
    assert!(current[0].abs() < 1e-12 && current[1].abs() < 1e-12);
    assert!((current[2] - 1.0 / mu).abs() < 1e-12, "Jz = {}", current[2]);
    assert!((force[0] + x0 / mu).abs() < 1e-12, "Fx = {}", force[0]);
    assert!(force[1].abs() < 1e-12 && force[2].abs() < 1e-12);
}

#[test]
fn out_of_plane_field_gradient_gives_magnetic_pressure_force() {
    // B = (0, 0, y): J = (1/mu, 0, 0), F = (0, -Bz/mu, 0) = -grad(Bz^2/(2 mu)).
    let mu = 2.0;
    let lf = LorentzForce::new(mu, 0.1);
    let y0 = 0.4;
    let (force, current) = lorentz_on_field(&lf, |_, y| [0.0, 0.0, y], 0.5, y0);
    assert!((current[0] - 1.0 / mu).abs() < 1e-12, "Jx = {}", current[0]);
    assert!(current[1].abs() < 1e-12 && current[2].abs() < 1e-12);
    assert!(force[0].abs() < 1e-12, "Fx = {}", force[0]);
    assert!((force[1] + y0 / mu).abs() < 1e-12, "Fy = {}", force[1]);
    assert!(force[2].abs() < 1e-12);
}

#[test]
fn in_plane_current_density_has_all_curl_components() {
    // Bz = a x + b y, By = c x, Bx = d y  =>  mu J = (b, -a, c - d).
    let (a, b, c, d) = (0.3, -0.7, 1.1, 0.4);
    let mu = 1.5;
    let lf = LorentzForce::new(mu, 0.05);
    let (_, current) = lorentz_on_field(&lf, |x, y| [d * y, c * x, a * x + b * y], 0.2, 0.9);
    assert!((current[0] - b / mu).abs() < 1e-12, "Jx = {}", current[0]);
    assert!((current[1] + a / mu).abs() < 1e-12, "Jy = {}", current[1]);
    assert!(
        (current[2] - (c - d) / mu).abs() < 1e-12,
        "Jz = {}",
        current[2]
    );
}

#[test]
fn compute_is_current_density_cross_center_field() {
    let lf = LorentzForce::new(MU_0, 0.01);
    let center = [0.3, -0.2, 0.5];
    let xp = [0.31, -0.1, 0.52];
    let xm = [0.28, -0.3, 0.47];
    let yp = [0.36, -0.2, 0.58];
    let ym = [0.25, -0.2, 0.41];
    let j = lf.current_density(xp, xm, yp, ym);
    let force = lf.compute(center, xp, xm, yp, ym);
    let expected = [
        j[1] * center[2] - j[2] * center[1],
        j[2] * center[0] - j[0] * center[2],
        j[0] * center[1] - j[1] * center[0],
    ];
    for c in 0..3 {
        assert!(
            (force[c] - expected[c]).abs() <= 1e-12 * expected[c].abs().max(1.0),
            "component {c}: {} vs {}",
            force[c],
            expected[c]
        );
    }
    // The Lorentz force is perpendicular to B.
    let dot: f64 = (0..3).map(|c| force[c] * center[c]).sum();
    assert!(dot.abs() <= 1e-9 * force.iter().map(|v| v.abs()).sum::<f64>().max(1.0));
}

#[test]
fn uniform_field_exerts_no_force_in_three_components() {
    let lf = LorentzForce::new(MU_0, 0.1);
    let b = [0.2, -0.4, 0.9];
    let force = lf.compute(b, b, b, b, b);
    assert!(force.iter().all(|c| c.abs() < 1e-15));
}

// ---------------------------------------------------------------------------
// Equilibria (F2601)
// ---------------------------------------------------------------------------

fn moments(f: &[f64; D2Q9_Q]) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let mut m0 = 0.0;
    let mut m1 = [0.0; 2];
    let mut m2 = [[0.0; 2]; 2];
    for (q, &fq) in f.iter().enumerate() {
        let c = [D2Q9_EX[q], D2Q9_EY[q]];
        m0 += fq;
        for (a, m1_a) in m1.iter_mut().enumerate() {
            *m1_a += c[a] * fq;
            for (b, m2_ab) in m2[a].iter_mut().enumerate() {
                *m2_ab += c[a] * c[b] * fq;
            }
        }
    }
    (m0, m1, m2)
}

#[test]
fn mhd_fluid_equilibrium_moments_include_the_maxwell_stress() {
    let (rho, ux, uy, bx, by) = (1.3, 0.04, -0.03, 0.12, -0.07);
    let feq = mhd_fluid_equilibrium(rho, ux, uy, bx, by);
    let (m0, m1, m2) = moments(&feq);
    assert!((m0 - rho).abs() < 1e-14);
    assert!((m1[0] - rho * ux).abs() < 1e-14);
    assert!((m1[1] - rho * uy).abs() < 1e-14);
    let u = [ux, uy];
    let b = [bx, by];
    let b2 = bx * bx + by * by;
    for a in 0..2 {
        for c in 0..2 {
            let delta = if a == c { 1.0 } else { 0.0 };
            let expected = rho * CS2 * delta + rho * u[a] * u[c] + 0.5 * b2 * delta - b[a] * b[c];
            assert!(
                (m2[a][c] - expected).abs() < 1e-14,
                "Pi[{a}][{c}] = {} vs {expected}",
                m2[a][c]
            );
        }
    }
}

#[test]
fn mhd_fluid_equilibrium_without_field_is_the_hydrodynamic_one() {
    let a = mhd_fluid_equilibrium(1.1, 0.05, 0.02, 0.0, 0.0);
    let b = fluid_equilibrium(1.1, 0.05, 0.02);
    for q in 0..D2Q9_Q {
        assert!((a[q] - b[q]).abs() < 1e-15);
    }
}

#[test]
fn magnetic_equilibrium_recovers_field_and_induction_flux() {
    let (bx, by, ux, uy) = (0.11, -0.06, 0.03, 0.05);
    let (gx, gy) = magnetic_equilibrium(bx, by, ux, uy);
    let (gx0, gx1, _) = moments(&gx);
    let (gy0, gy1, _) = moments(&gy);
    assert!((gx0 - bx).abs() < 1e-15, "sum gx = {gx0}");
    assert!((gy0 - by).abs() < 1e-15, "sum gy = {gy0}");
    // Flux of B_beta along alpha is u_alpha B_beta - B_alpha u_beta.
    let flux = |alpha: usize, beta: usize| {
        let u = [ux, uy];
        let b = [bx, by];
        u[alpha] * b[beta] - b[alpha] * u[beta]
    };
    assert!((gx1[0] - flux(0, 0)).abs() < 1e-15);
    assert!(
        (gx1[1] - flux(1, 0)).abs() < 1e-15,
        "y-flux of Bx = {}",
        gx1[1]
    );
    assert!(
        (gy1[0] - flux(0, 1)).abs() < 1e-15,
        "x-flux of By = {}",
        gy1[0]
    );
    assert!((gy1[1] - flux(1, 1)).abs() < 1e-15);
    // The flux of B_y along x is -(flux of B_x along y): both are +-(u x B)_z.
    assert!((gx1[1] + gy1[0]).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// MhdLbm dynamics (F2601)
// ---------------------------------------------------------------------------

#[test]
fn uniform_magnetised_state_is_an_exact_steady_solution() {
    let mut lbm = MhdLbm::new(8, 6, 0.8, 0.9);
    for _ in 0..100 {
        lbm.advance();
    }
    for idx in 0..8 * 6 {
        assert!((lbm.bx[idx] - 0.1).abs() < 1e-14, "Bx = {}", lbm.bx[idx]);
        assert!(lbm.by[idx].abs() < 1e-14, "By = {}", lbm.by[idx]);
        assert!(lbm.ux[idx].abs() < 1e-14 && lbm.uy[idx].abs() < 1e-14);
        assert!((lbm.rho[idx] - 1.0).abs() < 1e-14);
    }
    assert_eq!(lbm.step, 100);
}

#[test]
fn uniform_field_with_uniform_flow_is_advected_without_change() {
    let mut lbm = MhdLbm::new(6, 6, 0.9, 0.7);
    for j in 0..6 {
        for i in 0..6 {
            lbm.set_cell(i, j, 1.0, [0.05, -0.02], [0.08, 0.03])
                .unwrap();
        }
    }
    for _ in 0..60 {
        lbm.advance();
    }
    for idx in 0..36 {
        assert!((lbm.bx[idx] - 0.08).abs() < 1e-13, "Bx = {}", lbm.bx[idx]);
        assert!((lbm.by[idx] - 0.03).abs() < 1e-13, "By = {}", lbm.by[idx]);
        assert!((lbm.ux[idx] - 0.05).abs() < 1e-13, "ux = {}", lbm.ux[idx]);
        assert!((lbm.uy[idx] + 0.02).abs() < 1e-13, "uy = {}", lbm.uy[idx]);
    }
}

#[test]
fn magnetic_pressure_gradient_accelerates_the_fluid() {
    // Bx(y) = B0 sin(2 pi y / ny), u = 0, rho = 1, f at equilibrium.  After one
    // step the momentum is the central difference of the magnetic pressure
    // gradient: rho u_y = -(Bx(y+1)^2 - Bx(y-1)^2)/4.  The old scalar-g model
    // never coupled B to the fluid, so u stayed exactly zero.
    let (nx, ny) = (4, 16);
    let mut lbm = MhdLbm::new(nx, ny, 1.0, 1.0);
    let bx_at = |j: usize| 0.1 * (2.0 * PI * j as f64 / ny as f64).sin();
    for j in 0..ny {
        for i in 0..nx {
            lbm.set_cell(i, j, 1.0, [0.0, 0.0], [bx_at(j), 0.0])
                .unwrap();
        }
    }
    lbm.advance();
    let mut max_force = 0.0_f64;
    for j in 0..ny {
        let expected = -0.25 * (bx_at((j + 1) % ny).powi(2) - bx_at((j + ny - 1) % ny).powi(2));
        max_force = max_force.max(expected.abs());
        for i in 0..nx {
            let idx = j * nx + i;
            let jy = lbm.rho[idx] * lbm.uy[idx];
            assert!(
                (jy - expected).abs() < 1e-14,
                "j={j}: rho*uy = {jy}, expected {expected}"
            );
            assert!((lbm.rho[idx] * lbm.ux[idx]).abs() < 1e-15);
        }
    }
    assert!(max_force > 1e-4, "test must exercise a non-trivial force");
}

#[test]
fn velocity_gradient_along_the_field_induces_a_transverse_component() {
    // B = (B0, 0) and u_y(x) = A sin(kx) give Lambda = u_x B_y - u_y B_x = -u_y B0,
    // hence dB_y/dt = -dLambda/dx = B0 du_y/dx (field-line bending).  With all
    // populations at equilibrium one step yields exactly the central difference
    // B_y(x) = B0 (u_y(x+1) - u_y(x-1))/2 = B0 A sin(k) cos(kx).
    let (nx, ny) = (32, 4);
    let k = 2.0 * PI / nx as f64;
    let (b0, amp) = (0.1, 1e-3);
    let mut lbm = MhdLbm::new(nx, ny, 1.0, 1.0);
    for j in 0..ny {
        for i in 0..nx {
            let uy = amp * (k * i as f64).sin();
            lbm.set_cell(i, j, 1.0, [0.0, uy], [b0, 0.0]).unwrap();
        }
    }
    lbm.advance();
    for j in 0..ny {
        for i in 0..nx {
            let expected = b0 * amp * k.sin() * (k * i as f64).cos();
            let by = lbm.by[j * nx + i];
            assert!(
                (by - expected).abs() < 1e-14,
                "i={i} j={j}: By = {by}, expected {expected}"
            );
        }
    }
    assert!(
        b0 * amp * k.sin() > 1e-5,
        "the induced component must be non-trivial"
    );
}

#[test]
fn periodic_run_conserves_mass_momentum_and_magnetic_flux() {
    let (nx, ny) = (12, 10);
    let mut lbm = MhdLbm::new(nx, ny, 0.8, 0.7);
    for j in 0..ny {
        for i in 0..nx {
            let (x, y) = (i as f64 / nx as f64, j as f64 / ny as f64);
            let rho = 1.0 + 0.02 * (2.0 * PI * x).sin();
            let u = [
                0.03 * (2.0 * PI * y).cos(),
                0.02 * (2.0 * PI * (x + y)).sin(),
            ];
            let b = [
                0.1 + 0.03 * (2.0 * PI * y).sin(),
                0.04 * (2.0 * PI * x).cos(),
            ];
            lbm.set_cell(i, j, rho, u, b).unwrap();
        }
    }
    let mass0 = lbm.total_mass();
    let p0 = lbm.total_momentum();
    let flux0 = lbm.total_magnetic_flux();
    let energy0 = lbm.kinetic_energy() + lbm.magnetic_energy();
    for _ in 0..80 {
        lbm.advance();
    }
    let p1 = lbm.total_momentum();
    let flux1 = lbm.total_magnetic_flux();
    assert!((lbm.total_mass() - mass0).abs() < 1e-11, "mass drift");
    assert!(
        (p1[0] - p0[0]).abs() < 1e-11 && (p1[1] - p0[1]).abs() < 1e-11,
        "momentum drift"
    );
    assert!(
        (flux1[0] - flux0[0]).abs() < 1e-11 && (flux1[1] - flux0[1]).abs() < 1e-11,
        "flux drift"
    );
    // Viscosity and resistivity dissipate energy; the run must stay bounded and regular.
    let energy1 = lbm.kinetic_energy() + lbm.magnetic_energy();
    assert!(
        energy1.is_finite() && energy1 < 1.5 * energy0,
        "energy {energy0} -> {energy1}"
    );
    assert!(lbm.rho.iter().all(|r| r.is_finite() && *r > 0.5));
}

/// Amplitude of the `sin(kx)` mode of `u_y` averaged over all rows.
fn sine_mode_amplitude(lbm: &MhdLbm, k: f64) -> f64 {
    let mut acc = 0.0;
    for j in 0..lbm.ny {
        for i in 0..lbm.nx {
            acc += lbm.uy[j * lbm.nx + i] * (k * i as f64).sin();
        }
    }
    2.0 * acc / (lbm.nx * lbm.ny) as f64
}

/// Measure the angular frequency of a transverse Alfvén wave from the spacing of
/// the first two zero crossings of the mode amplitude (linearly interpolated).
fn measured_alfven_frequency(rho0: f64, b0: f64, tau: f64) -> f64 {
    let (nx, ny) = (64, 4);
    let k = 2.0 * PI / nx as f64;
    let mut lbm = MhdLbm::new(nx, ny, tau, tau);
    for j in 0..ny {
        for i in 0..nx {
            let uy = 1e-3 * (k * i as f64).sin();
            lbm.set_cell(i, j, rho0, [0.0, uy], [b0, 0.0]).unwrap();
        }
    }
    let mut crossings: Vec<f64> = Vec::new();
    let mut previous = sine_mode_amplitude(&lbm, k);
    for t in 1..=4000 {
        lbm.advance();
        let current = sine_mode_amplitude(&lbm, k);
        if previous * current < 0.0 {
            crossings.push((t - 1) as f64 + previous / (previous - current));
            if crossings.len() == 2 {
                break;
            }
        }
        previous = current;
    }
    assert_eq!(crossings.len(), 2, "wave did not oscillate: {crossings:?}");
    PI / (crossings[1] - crossings[0])
}

#[test]
fn alfven_wave_travels_at_b_over_sqrt_rho() {
    let k = 2.0 * PI / 64.0;
    for &(rho0, b0) in &[(1.0, 0.1), (2.0, 0.1), (1.0, 0.05)] {
        let measured = measured_alfven_frequency(rho0, b0, 0.6);
        let expected = b0 / f64::sqrt(rho0) * k;
        let error = (measured / expected - 1.0).abs();
        assert!(
            error < 0.02,
            "rho={rho0} B0={b0}: omega = {measured}, expected {expected} (error {error})"
        );
    }
}

#[test]
fn magnetic_diffusion_decays_at_the_rate_eta_k_squared() {
    let (nx, ny) = (4, 32);
    let k = 2.0 * PI / ny as f64;
    let b0 = 0.01;
    for &tau_g in &[0.6, 0.8, 1.2] {
        let mut lbm = MhdLbm::new(nx, ny, 0.8, tau_g);
        for j in 0..ny {
            for i in 0..nx {
                lbm.set_cell(i, j, 1.0, [0.0, 0.0], [b0 * (k * j as f64).sin(), 0.0])
                    .unwrap();
            }
        }
        let steps = 200;
        for _ in 0..steps {
            lbm.advance();
        }
        let mut acc = 0.0;
        for j in 0..ny {
            for i in 0..nx {
                acc += lbm.bx[j * nx + i] * (k * j as f64).sin();
            }
        }
        let amplitude = 2.0 * acc / (nx * ny) as f64;
        let eta = lbm.magnetic_diffusivity();
        assert!((eta - CS2 * (tau_g - 0.5)).abs() < 1e-15);
        let expected = b0 * (-eta * k * k * steps as f64).exp();
        assert!(
            (amplitude / expected - 1.0).abs() < 0.02,
            "tau_g={tau_g}: amplitude {amplitude}, expected {expected}"
        );
    }
}

#[test]
fn transport_coefficients_map_to_relaxation_times() {
    let lbm = MhdLbm::from_transport_coefficients(8, 8, 0.05, 0.02);
    assert!((lbm.tau_f - (0.05 * 3.0 + 0.5)).abs() < 1e-14);
    assert!((lbm.tau_g - (0.02 * 3.0 + 0.5)).abs() < 1e-14);
    assert!((lbm.viscosity() - 0.05).abs() < 1e-14);
    assert!((lbm.magnetic_diffusivity() - 0.02).abs() < 1e-14);
}

#[test]
fn perturb_velocity_keeps_the_field_and_the_magnetic_stress() {
    let mut lbm = MhdLbm::new(5, 5, 1.0, 1.0);
    lbm.set_uniform_magnetic_field(0.09, -0.05);
    lbm.perturb_velocity(2, 3, 0.04, -0.02);
    let idx = 3 * 5 + 2;
    assert!((lbm.bx[idx] - 0.09).abs() < 1e-15 && (lbm.by[idx] + 0.05).abs() < 1e-15);
    let (m0, m1, m2) = moments(&lbm.f[idx]);
    assert!((m0 - 1.0).abs() < 1e-14);
    assert!((m1[0] - 0.04).abs() < 1e-14 && (m1[1] + 0.02).abs() < 1e-14);
    // Pi_xx - Pi_yy = rho (ux^2 - uy^2) - (Bx^2 - By^2): the stress of the local field.
    let expected = (0.04_f64.powi(2) - 0.02_f64.powi(2)) - (0.09_f64.powi(2) - 0.05_f64.powi(2));
    assert!((m2[0][0] - m2[1][1] - expected).abs() < 1e-14);
    let (g0x, _, _) = moments(&lbm.gx[idx]);
    let (g0y, _, _) = moments(&lbm.gy[idx]);
    assert!((g0x - 0.09).abs() < 1e-15 && (g0y + 0.05).abs() < 1e-15);
}

#[test]
fn set_cell_rejects_indices_outside_the_grid() {
    let mut lbm = MhdLbm::new(4, 3, 1.0, 1.0);
    assert!(lbm.set_cell(3, 2, 1.0, [0.0; 2], [0.1, 0.0]).is_ok());
    assert!(lbm.set_cell(4, 0, 1.0, [0.0; 2], [0.1, 0.0]).is_err());
    assert!(lbm.set_cell(0, 3, 1.0, [0.0; 2], [0.1, 0.0]).is_err());
}

#[test]
fn alfven_speed_diagnostic_uses_the_local_state() {
    let mut lbm = MhdLbm::new(4, 4, 1.0, 1.0);
    lbm.set_cell(1, 2, 4.0, [0.0, 0.0], [0.3, 0.4]).unwrap();
    assert!((lbm.alfven_speed(1, 2) - 0.5 / 2.0).abs() < 1e-14);
    assert!((lbm.alfven_speed(0, 0) - 0.1).abs() < 1e-14);
}
