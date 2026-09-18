// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the rotating-frame, Rossby-wave and tidal solvers.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// RotatingFrameLbm: vorticity and enstrophy
// ─────────────────────────────────────────────────────────────────────────────

/// Fill the whole grid with the equilibrium of the velocity `velocity(ix, iy)`.
fn set_rotating_velocity(
    lbm: &mut RotatingFrameLbm,
    velocity: impl Fn(usize, usize) -> (f64, f64),
) {
    for ix in 0..lbm.nx {
        for iy in 0..lbm.ny {
            let (ux, uy) = velocity(ix, iy);
            lbm.f[ix][iy] = feq(1.0, ux, uy);
        }
    }
}

#[test]
fn vorticity_of_solid_body_rotation_is_twice_the_angular_velocity() {
    let omega_rot = 0.004;
    let (nx, ny) = (32, 24);
    let mut lbm = RotatingFrameLbm::new(nx, ny, 1.0, 0.0);
    let (xc, yc) = (0.5 * nx as f64, 0.5 * ny as f64);
    set_rotating_velocity(&mut lbm, |ix, iy| {
        (-omega_rot * (iy as f64 - yc), omega_rot * (ix as f64 - xc))
    });
    let mut interior_enstrophy = 0.0;
    for ix in 1..nx - 1 {
        for iy in 1..ny - 1 {
            let vort = lbm.vorticity_at(ix, iy);
            assert!(
                (vort - 2.0 * omega_rot).abs() < 1e-10,
                "vorticity at ({ix}, {iy}) is {vort}, expected {}",
                2.0 * omega_rot
            );
            interior_enstrophy += 0.5 * vort * vort;
        }
    }
    let mean = interior_enstrophy / ((nx - 2) * (ny - 2)) as f64;
    let expected = 2.0 * omega_rot * omega_rot;
    assert!(
        (mean - expected).abs() < 1e-12,
        "interior enstrophy {mean}, expected 2 Omega^2 = {expected}"
    );
}

#[test]
fn enstrophy_of_an_x_velocity_shear_mode_matches_the_finite_difference_value() {
    // u_x = U sin(2 pi iy / ny): zeta = -(u_x(iy+1) - u_x(iy-1)) / 2
    //                                = -U sin(k) cos(k iy), k = 2 pi / ny,
    // so <zeta^2 / 2> = U^2 sin^2(k) / 4.  The old stencil returned 0 here.
    let (nx, ny) = (12, 20);
    let amplitude = 0.03;
    let k = 2.0 * PI / ny as f64;
    let mut lbm = RotatingFrameLbm::new(nx, ny, 1.0, 0.0);
    set_rotating_velocity(&mut lbm, |_, iy| (amplitude * (k * iy as f64).sin(), 0.0));
    let expected = amplitude * amplitude * k.sin().powi(2) / 4.0;
    let ens = lbm.enstrophy();
    assert!(expected > 1e-7, "the reference value must be non-trivial");
    assert!(
        (ens - expected).abs() < 1e-14 * expected.max(1.0),
        "enstrophy {ens}, expected {expected}"
    );
}

#[test]
fn enstrophy_of_a_y_velocity_shear_mode_matches_the_finite_difference_value() {
    // u_y = V sin(2 pi ix / nx): zeta = (u_y(ix+1) - u_y(ix-1)) / 2 along x.
    let (nx, ny) = (20, 12);
    let amplitude = 0.02;
    let k = 2.0 * PI / nx as f64;
    let mut lbm = RotatingFrameLbm::new(nx, ny, 1.0, 0.0);
    set_rotating_velocity(&mut lbm, |ix, _| (0.0, amplitude * (k * ix as f64).sin()));
    let expected = amplitude * amplitude * k.sin().powi(2) / 4.0;
    let ens = lbm.enstrophy();
    assert!(
        (ens - expected).abs() < 1e-14 * expected.max(1.0),
        "enstrophy {ens}, expected {expected}"
    );
    // Vorticity of this mode at ix = 0: (sin(k) - sin(-k)) / 2 * V = V sin(k).
    let vort = lbm.vorticity_at(0, 3);
    assert!((vort - amplitude * k.sin()).abs() < 1e-14, "zeta = {vort}");
}

#[test]
fn irrotational_flow_has_no_enstrophy() {
    // u = grad(phi) with phi = sin(2 pi ix / nx) sin(2 pi iy / ny) is curl free
    // in the continuum; with periodic central differences the discrete curl of
    // a discrete gradient field also vanishes identically.
    let (nx, ny) = (16, 10);
    let kx = 2.0 * PI / nx as f64;
    let ky = 2.0 * PI / ny as f64;
    let phi = |ix: usize, iy: usize| 0.01 * (kx * ix as f64).sin() * (ky * iy as f64).sin();
    let mut lbm = RotatingFrameLbm::new(nx, ny, 1.0, 0.0);
    set_rotating_velocity(&mut lbm, |ix, iy| {
        let ixp = (ix + 1) % nx;
        let ixm = (ix + nx - 1) % nx;
        let iyp = (iy + 1) % ny;
        let iym = (iy + ny - 1) % ny;
        (
            0.5 * (phi(ixp, iy) - phi(ixm, iy)),
            0.5 * (phi(ix, iyp) - phi(ix, iym)),
        )
    });
    assert!(lbm.enstrophy() < 1e-26, "enstrophy {}", lbm.enstrophy());
}

// ─────────────────────────────────────────────────────────────────────────────
// RossbyWaveSim: Coriolis parameter and stream function
// ─────────────────────────────────────────────────────────────────────────────

/// Set the equilibrium velocity field of `sim` from `velocity(ix, iy)`.
fn set_rossby_velocity(sim: &mut RossbyWaveSim, velocity: impl Fn(usize, usize) -> (f64, f64)) {
    for ix in 0..sim.nx {
        for iy in 0..sim.ny {
            let (ux, uy) = velocity(ix, iy);
            sim.f[ix][iy] = feq(1.0, ux, uy);
        }
    }
    sim.update_streamfunction();
}

#[test]
fn coriolis_parameter_equals_f0_at_the_centre_row() {
    let (f0, beta, dy) = (1.0e-4, 2.0e-6, 3.0);
    let rw = RossbyWaveSim::new(10, 16, 1.0, f0, beta, dy);
    assert_eq!(rw.coriolis_at(8), f0, "f0 is the value at the centre row");
    for offset in 1..=7usize {
        let north = rw.coriolis_at(8 + offset);
        let south = rw.coriolis_at(8 - offset);
        assert!(
            (north + south - 2.0 * f0).abs() < 1e-18,
            "f must be antisymmetric about the centre: {north} {south}"
        );
    }
    let expected_bottom = f0 + beta * (-8.0 * dy);
    assert!((rw.coriolis_at(0) - expected_bottom).abs() < 1e-18);
    assert!(rw.coriolis_at(0) < f0 && rw.coriolis_at(15) > f0);
}

#[test]
fn streamfunction_recovers_an_x_dependent_mode() {
    // psi = psi0 sin(2 pi m ix / nx):  u_x = 0,  u_y = d psi / dx.
    let (nx, ny) = (24, 16);
    let psi0 = 0.05;
    for mode in [1usize, 2, 5] {
        let k = 2.0 * PI * mode as f64 / nx as f64;
        let mut rw = RossbyWaveSim::new(nx, ny, 1.0, 0.0, 0.0, 1.0);
        set_rossby_velocity(&mut rw, |ix, _| (0.0, psi0 * k * (k * ix as f64).cos()));
        for ix in 0..nx {
            for iy in 0..ny {
                let expected = psi0 * (k * ix as f64).sin();
                assert!(
                    (rw.psi[ix][iy] - expected).abs() < 1e-10,
                    "mode {mode}: psi[{ix}][{iy}] = {}, expected {expected}",
                    rw.psi[ix][iy]
                );
            }
        }
    }
}

#[test]
fn streamfunction_recovers_a_y_dependent_mode() {
    // psi = psi0 cos(2 pi m iy / ny):  u_x = -d psi / dy,  u_y = 0.
    let (nx, ny) = (12, 20);
    let psi0 = 0.03;
    for mode in [1usize, 3] {
        let k = 2.0 * PI * mode as f64 / ny as f64;
        let mut rw = RossbyWaveSim::new(nx, ny, 1.0, 0.0, 0.0, 1.0);
        set_rossby_velocity(&mut rw, |_, iy| (psi0 * k * (k * iy as f64).sin(), 0.0));
        for ix in 0..nx {
            for iy in 0..ny {
                let expected = psi0 * (k * iy as f64).cos();
                assert!(
                    (rw.psi[ix][iy] - expected).abs() < 1e-10,
                    "mode {mode}: psi[{ix}][{iy}] = {}, expected {expected}",
                    rw.psi[ix][iy]
                );
            }
        }
    }
}

#[test]
fn streamfunction_recovers_an_oblique_mode_on_a_rectangular_grid() {
    // psi = psi0 sin(kx x + ky y + 0.7) with one cycle along x and two along y.
    let (nx, ny) = (18, 14);
    let psi0 = 0.04;
    let kx = 2.0 * PI / nx as f64;
    let ky = 2.0 * PI * 2.0 / ny as f64;
    let phase = |ix: usize, iy: usize| kx * ix as f64 + ky * iy as f64 + 0.7;
    let mut rw = RossbyWaveSim::new(nx, ny, 1.0, 0.0, 0.0, 1.0);
    set_rossby_velocity(&mut rw, |ix, iy| {
        let c = psi0 * phase(ix, iy).cos();
        (-ky * c, kx * c)
    });
    for ix in 0..nx {
        for iy in 0..ny {
            let expected = psi0 * phase(ix, iy).sin();
            assert!(
                (rw.psi[ix][iy] - expected).abs() < 1e-10,
                "psi[{ix}][{iy}] = {}, expected {expected}",
                rw.psi[ix][iy]
            );
        }
    }
}

#[test]
fn streamfunction_ignores_the_divergent_part_of_the_velocity() {
    // u = grad(phi) is a potential flow: zero vorticity, hence psi = 0.
    let (nx, ny) = (16, 12);
    let kx = 2.0 * PI / nx as f64;
    let ky = 2.0 * PI / ny as f64;
    let mut rw = RossbyWaveSim::new(nx, ny, 1.0, 0.0, 0.0, 1.0);
    set_rossby_velocity(&mut rw, |ix, iy| {
        let phase_x = kx * ix as f64;
        let phase_y = ky * iy as f64;
        (
            0.01 * kx * phase_x.cos() * phase_y.sin(),
            0.01 * ky * phase_x.sin() * phase_y.cos(),
        )
    });
    let max_abs = rw
        .psi
        .iter()
        .flatten()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    assert!(max_abs < 1e-12, "potential flow has psi = {max_abs}");
}

#[test]
fn wave_ic_streamfunction_matches_the_analytic_amplitude() {
    // On a square grid wave_ic is the stream-function field
    // psi = A n / (2 pi) sin(2 pi (kx ix + ky iy) / n).
    let n = 16usize;
    let amplitude = 0.01;
    let mut rw = RossbyWaveSim::new(n, n, 1.0, 1.0e-4, 1.0e-5, 1.0);
    rw.wave_ic(1.0, 2.0, amplitude);
    let psi0 = amplitude * n as f64 / (2.0 * PI);
    let mut max_psi = 0.0f64;
    for ix in 0..n {
        for iy in 0..n {
            let theta = 2.0 * PI * (ix as f64 + 2.0 * iy as f64) / n as f64;
            let expected = psi0 * theta.sin();
            max_psi = max_psi.max(expected.abs());
            assert!(
                (rw.psi[ix][iy] - expected).abs() < 1e-10,
                "psi[{ix}][{iy}] = {}, expected {expected}",
                rw.psi[ix][iy]
            );
        }
    }
    assert!(max_psi > 0.02, "the wave must have a non-trivial amplitude");
}

#[test]
fn streamfunction_stays_finite_and_zero_mean_while_stepping() {
    let mut rw = RossbyWaveSim::new(16, 16, 1.2, 1.0e-4, 1.0e-5, 1.0);
    rw.wave_ic(1.0, 1.0, 0.01);
    for _ in 0..10 {
        rw.step();
        let mean = rw.psi.iter().flatten().sum::<f64>() / 256.0;
        assert!(mean.abs() < 1e-12, "psi mean {mean}");
        assert!(rw.psi.iter().flatten().all(|value| value.is_finite()));
    }
    let max_abs = rw
        .psi
        .iter()
        .flatten()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    assert!(max_abs > 1e-3, "psi must carry the wave: {max_abs}");
}

// ─────────────────────────────────────────────────────────────────────────────
// TidalSimulation: depth-dependent bottom drag
// ─────────────────────────────────────────────────────────────────────────────

/// Speed `|u|` of the flow at cell `(ix, iy)`.
fn tidal_speed(ts: &TidalSimulation, ix: usize, iy: usize) -> f64 {
    let (_, ux, uy) = macroscopic(&ts.f[ix][iy]);
    (ux * ux + uy * uy).sqrt()
}

/// A tidal simulation with the uniform depth `depth` and strong quadratic
/// drag (`Cd = 0.5`), so that the depth has a visible effect.
fn tidal_with_uniform_depth(depth: f64) -> TidalSimulation {
    let mut ts = TidalSimulation::new(8, 4, 1.2, 0.02, 0.01, 0.5);
    for column in ts.depth.iter_mut() {
        column.iter_mut().for_each(|cell| *cell = depth);
    }
    ts
}

#[test]
fn shallower_water_brakes_the_tidal_flow_more() {
    let speed_after = |depth: f64| {
        let mut ts = tidal_with_uniform_depth(depth);
        for _ in 0..150 {
            ts.step();
        }
        tidal_speed(&ts, 3, 2)
    };
    let shallow = speed_after(0.25);
    let medium = speed_after(1.0);
    let deep = speed_after(4.0);
    assert!(
        shallow < 0.8 * medium && medium < 0.9 * deep,
        "speeds for depths 0.25 / 1 / 4: {shallow} {medium} {deep}"
    );
    // Deep water approaches the frictionless tide a sin(w t) = 0.02 sin(1.5).
    let frictionless = 0.02 * (0.01f64 * 150.0).sin();
    assert!(
        deep < frictionless && deep > 0.9 * frictionless,
        "{deep} vs {frictionless}"
    );
}

#[test]
fn shelf_and_uniform_bathymetry_give_different_velocities() {
    let mut shelf = TidalSimulation::new(8, 4, 1.2, 0.02, 0.01, 0.5);
    shelf.shelf_bathymetry(0.25, 2.0, 4);
    let mut uniform = TidalSimulation::new(8, 4, 1.2, 0.02, 0.01, 0.5);
    uniform.shelf_bathymetry(2.0, 2.0, 4);
    for _ in 0..150 {
        shelf.step();
        uniform.step();
    }
    let on_shelf = tidal_speed(&shelf, 1, 2);
    let reference = tidal_speed(&uniform, 1, 2);
    assert!(
        on_shelf < 0.85 * reference,
        "the shelf must slow the flow: {on_shelf} vs uniform {reference}"
    );
    // Within the shelf run the shallow side is slower than the deep side.
    assert!(tidal_speed(&shelf, 1, 2) < tidal_speed(&shelf, 6, 2));
}

#[test]
fn drag_impulse_is_inversely_proportional_to_the_depth() {
    // One step from a uniform flow with the tidal force switched off: the
    // momentum lost to the drag is Cd |u|^2 / H per unit depth.
    let loss = |depth: f64| {
        let mut ts = TidalSimulation::new(6, 5, 1.0, 0.0, 0.01, 0.01);
        for column in ts.depth.iter_mut() {
            column.iter_mut().for_each(|cell| *cell = depth);
        }
        let u0 = 0.03;
        for column in ts.f.iter_mut() {
            column.iter_mut().for_each(|cell| *cell = feq(1.0, u0, 0.0));
        }
        ts.step();
        u0 - tidal_speed(&ts, 2, 2)
    };
    let (shallow, deep) = (loss(0.5), loss(1.0));
    assert!(shallow > 1e-6, "the drag must act: {shallow}");
    assert!(
        (shallow / deep - 2.0).abs() < 0.02,
        "halving the depth must double the drag: {shallow} vs {deep}"
    );
    // Crank-Nicolson value: u1 = u0 (1 - r/2) / (1 + r/2) with r = Cd |u0| / H.
    let (u0, cd, depth) = (0.03, 0.01, 0.5);
    let r = cd * u0 / depth;
    let expected = u0 * (1.0 - 0.5 * r) / (1.0 + 0.5 * r);
    assert!(
        (u0 - shallow - expected).abs() < 1e-12,
        "{} vs {expected}",
        u0 - shallow
    );
}

#[test]
fn dry_and_invalid_depths_neither_blow_up_nor_reverse_the_flow() {
    for depth in [0.0, -3.0, f64::NAN, 1e-9] {
        let mut ts = tidal_with_uniform_depth(1.0);
        for column in ts.depth.iter_mut() {
            column[1] = depth;
        }
        for _ in 0..100 {
            ts.step();
        }
        for ix in 0..ts.nx {
            for iy in 0..ts.ny {
                let speed = tidal_speed(&ts, ix, iy);
                assert!(
                    speed.is_finite() && speed < 0.05,
                    "depth {depth}: speed {speed} at ({ix}, {iy})"
                );
            }
        }
        // The dry row is braked to (nearly) rest, far below the open water.
        assert!(
            tidal_speed(&ts, 3, 1) < 0.5 * tidal_speed(&ts, 3, 3),
            "depth {depth}"
        );
    }
}

#[test]
fn tidal_forcing_alone_follows_the_harmonic_acceleration() {
    // Frictionless (Cd = 0) uniform tide: u(t) = a sin(w t) for a spatially
    // uniform body force.
    let (a, w) = (0.02, 0.01);
    let mut ts = TidalSimulation::new(6, 5, 1.0, a, w, 0.0);
    for step in 1..=200 {
        ts.step();
        let t = step as f64;
        // Impulse of the explicitly sampled force: sum of a w cos(w s), s < t.
        let expected: f64 = (0..step).map(|s| a * w * (w * s as f64).cos()).sum();
        let speed = tidal_speed(&ts, 2, 2);
        assert!(
            (speed - expected).abs() < 1e-12,
            "t = {t}: {speed} vs {expected}"
        );
    }
}
