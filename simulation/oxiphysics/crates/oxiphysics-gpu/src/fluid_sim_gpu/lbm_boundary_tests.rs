// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the Zou-He inlet and outlet of [`LbmD2Q9`].
//!
//! The periodic streaming step refills the populations that point into the
//! domain at the inlet and outlet column with whatever leaves through the
//! opposite wall, so a boundary condition that is only written once (by the
//! setters) is gone after the first step.  These tests check that the conditions
//! are re-imposed by every step: analytically for single cells, and on a
//! channel flow driven only by the inlet velocity and the outlet density.

use super::*;

const NX: usize = 40;
const NY: usize = 12;
const TAU: f64 = 0.8;
const U_IN: f64 = 0.05;
const RHO_OUT: f64 = 1.0;

/// A channel with no-slip walls on the top and bottom row, a Zou-He velocity
/// inlet on the left and a Zou-He pressure outlet on the right.
fn channel() -> LbmD2Q9 {
    let mut lbm = LbmD2Q9::new(NX, NY, TAU);
    lbm.set_inlet_velocity(U_IN, 0.0);
    lbm.set_outlet_pressure(RHO_OUT);
    // The walls are set after the open boundaries, so the corners are solid.
    for x in 0..NX {
        lbm.cell_type[x] = LbmCellType::Solid;
        lbm.cell_type[(NY - 1) * NX + x] = LbmCellType::Solid;
    }
    lbm
}

fn equilibrium_populations(rho: f64, ux: f64, uy: f64) -> [f64; D2Q9_Q] {
    let mut f = [0.0; D2Q9_Q];
    for (q, value) in f.iter_mut().enumerate() {
        *value = LbmD2Q9::f_equilibrium(rho, ux, uy, q);
    }
    f
}

fn column_flux(lbm: &LbmD2Q9, x: usize) -> f64 {
    (1..NY - 1)
        .map(|y| {
            let [ux, _] = lbm.velocity(x, y);
            lbm.density(x, y) * ux
        })
        .sum()
}

#[test]
fn setters_record_the_imposed_state() {
    let fresh = LbmD2Q9::new(8, 6, 0.9);
    assert_eq!(fresh.inlet_velocity, None);
    assert_eq!(fresh.outlet_density, None);
    let mut lbm = LbmD2Q9::new(8, 6, 0.9);
    lbm.set_inlet_velocity(0.04, -0.01);
    lbm.set_outlet_pressure(1.02);
    assert_eq!(lbm.inlet_velocity, Some([0.04, -0.01]));
    assert_eq!(lbm.outlet_density, Some(1.02));
    assert!((0..6).all(|y| lbm.cell_type[y * 8] == LbmCellType::Inlet));
    assert!((0..6).all(|y| lbm.cell_type[y * 8 + 7] == LbmCellType::Outlet));
}

#[test]
fn zou_he_inlet_leaves_the_equilibrium_state_unchanged_and_repairs_streamed_populations() {
    let (rho, ux, uy) = (1.03, 0.07, 0.02);
    let equilibrium = equilibrium_populations(rho, ux, uy);
    let mut lbm = LbmD2Q9::new(6, 5, 0.8);
    lbm.set_inlet_velocity(ux, uy);
    for (q, &value) in equilibrium.iter().enumerate() {
        lbm.set_f(0, 2, q, value);
    }
    // What the periodic streaming wrote into the populations that point into the
    // domain is arbitrary and must be replaced.
    lbm.set_f(0, 2, 1, 0.3);
    lbm.set_f(0, 2, 5, -0.2);
    lbm.set_f(0, 2, 8, 0.9);
    lbm.apply_open_boundaries();
    for (q, &value) in equilibrium.iter().enumerate() {
        assert!(
            (lbm.get_f(0, 2, q) - value).abs() < 1.0e-13,
            "population {q}: {} vs {value}",
            lbm.get_f(0, 2, q)
        );
    }
    let [got_ux, got_uy] = lbm.velocity(0, 2);
    assert!((got_ux - ux).abs() < 1.0e-13 && (got_uy - uy).abs() < 1.0e-13);
}

#[test]
fn zou_he_inlet_imposes_the_velocity_whatever_the_streamed_populations_are() {
    let mut lbm = LbmD2Q9::new(6, 5, 0.8);
    lbm.set_inlet_velocity(0.06, 0.015);
    let garbage = [0.4, 0.11, 0.07, 0.13, 0.05, 0.02, 0.03, 0.04, 0.06];
    for (q, &value) in garbage.iter().enumerate() {
        lbm.set_f(0, 3, q, value);
    }
    lbm.apply_open_boundaries();
    let [ux, uy] = lbm.velocity(0, 3);
    assert!((ux - 0.06).abs() < 1.0e-13, "ux = {ux}");
    assert!((uy - 0.015).abs() < 1.0e-13, "uy = {uy}");
}

#[test]
fn zou_he_outlet_leaves_the_equilibrium_state_unchanged_and_repairs_streamed_populations() {
    let (rho, ux) = (1.02, 0.05);
    let equilibrium = equilibrium_populations(rho, ux, 0.0);
    let mut lbm = LbmD2Q9::new(6, 5, 0.8);
    lbm.set_outlet_pressure(rho);
    for (q, &value) in equilibrium.iter().enumerate() {
        lbm.set_f(5, 2, q, value);
    }
    lbm.set_f(5, 2, 3, 0.8);
    lbm.set_f(5, 2, 6, -0.3);
    lbm.set_f(5, 2, 7, 0.01);
    lbm.apply_open_boundaries();
    for (q, &value) in equilibrium.iter().enumerate() {
        assert!(
            (lbm.get_f(5, 2, q) - value).abs() < 1.0e-13,
            "population {q}: {} vs {value}",
            lbm.get_f(5, 2, q)
        );
    }
}

#[test]
fn zou_he_outlet_imposes_the_density_and_zero_transverse_velocity() {
    let mut lbm = LbmD2Q9::new(6, 5, 0.8);
    lbm.set_outlet_pressure(1.04);
    let garbage = [0.4, 0.11, 0.07, 0.13, 0.05, 0.02, 0.03, 0.04, 0.06];
    for (q, &value) in garbage.iter().enumerate() {
        lbm.set_f(5, 1, q, value);
    }
    lbm.apply_open_boundaries();
    assert!((lbm.density(5, 1) - 1.04).abs() < 1.0e-13);
    assert!(lbm.velocity(5, 1)[1].abs() < 1.0e-13);
}

#[test]
fn open_boundary_cells_without_an_imposed_state_are_left_alone() {
    let mut lbm = LbmD2Q9::new(6, 5, 0.8);
    lbm.cell_type[0] = LbmCellType::Inlet;
    lbm.cell_type[5] = LbmCellType::Outlet;
    for q in 0..D2Q9_Q {
        lbm.set_f(0, 0, q, 0.1 + 0.01 * q as f64);
        lbm.set_f(5, 0, q, 0.2 + 0.01 * q as f64);
    }
    let before = lbm.f.clone();
    lbm.apply_open_boundaries();
    assert_eq!(lbm.f, before);
}

#[test]
fn degenerate_imposed_values_are_ignored() {
    for (inlet, outlet) in [
        (Some([1.0, 0.0]), None),
        (None, Some(0.0)),
        (None, Some(-1.0)),
        (None, Some(f64::NAN)),
    ] {
        let mut lbm = LbmD2Q9::new(6, 5, 0.8);
        lbm.set_inlet_velocity(0.05, 0.0);
        lbm.set_outlet_pressure(1.0);
        lbm.inlet_velocity = inlet;
        lbm.outlet_density = outlet;
        let before = lbm.f.clone();
        lbm.apply_open_boundaries();
        assert_eq!(lbm.f, before, "inlet {inlet:?}, outlet {outlet:?}");
        assert!(lbm.f.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn every_step_re_imposes_the_inlet_velocity_and_the_outlet_density() {
    let mut lbm = channel();
    for step in 1..=60 {
        lbm.step();
        for y in 1..NY - 1 {
            let [ux, uy] = lbm.velocity(0, y);
            assert!(
                (ux - U_IN).abs() < 1.0e-12 && uy.abs() < 1.0e-12,
                "step {step}, inlet cell {y}: ({ux}, {uy})"
            );
            assert!(
                (lbm.density(NX - 1, y) - RHO_OUT).abs() < 1.0e-12,
                "step {step}, outlet cell {y}: rho = {}",
                lbm.density(NX - 1, y)
            );
            assert!(lbm.velocity(NX - 1, y)[1].abs() < 1.0e-12);
        }
    }
}

#[test]
fn driven_channel_reaches_the_inlet_speed_and_a_poiseuille_profile() {
    let mut lbm = channel();
    for _ in 0..2000 {
        lbm.step();
    }
    assert!(lbm.f.iter().all(|v| v.is_finite()));
    let mid = NX / 2;
    // Mean streamwise velocity over the fluid cells of the middle cross-section:
    // the flow keeps entering at the inlet speed.  (The density, and with it the
    // mean velocity, rises a few per cent towards the inlet as the pressure drop
    // drives the flow.)
    let mean_ux: f64 = (1..NY - 1).map(|y| lbm.velocity(mid, y)[0]).sum::<f64>() / (NY - 2) as f64;
    assert!(
        (mean_ux / U_IN - 1.0).abs() < 0.10,
        "mean ux {mean_ux} vs inlet {U_IN}"
    );
    // A channel with no-slip walls develops the parabolic profile whose centre
    // velocity is 1.5 times the mean.
    let centre_ux = lbm.velocity(mid, NY / 2)[0];
    let ratio = centre_ux / mean_ux;
    assert!(ratio > 1.3 && ratio < 1.6, "centre/mean = {ratio}");
    // Mass is conserved along the steady channel.
    let inlet_flux = column_flux(&lbm, 0);
    for x in [1, 5, mid, NX - 5, NX - 1] {
        let flux = column_flux(&lbm, x);
        assert!(
            (flux / inlet_flux - 1.0).abs() < 1.0e-3,
            "flux at {x}: {flux} vs inlet {inlet_flux}"
        );
    }
    // The pressure falls from the inlet to the outlet.
    let density = |x: usize| lbm.density(x, NY / 2);
    assert!(density(2) > density(mid) && density(mid) > density(NX - 2));
    assert!((density(NX - 1) - RHO_OUT).abs() < 1.0e-12);
}
