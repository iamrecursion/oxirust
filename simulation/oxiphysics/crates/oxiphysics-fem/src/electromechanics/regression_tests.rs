// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The strain coefficients `d33`, `d31` of the isotropic-skeleton piezoelectric models.
//!
//! `d33 = e33 / C33` and `d31 = e31 / C33` are the coefficients of a laterally clamped
//! bar; the strain of a stress-free body is `d = s e`, which couples `e31` into `d33`
//! through the Poisson contraction.

use super::*;
use crate::error::Error;

fn assert_close(actual: f64, expected: f64, relative: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs(),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn closed_forms_match_the_general_six_by_six_inversion() {
    for (e_mod, nu) in [
        (67.0e9, 0.31),
        (120.0e9, 0.0),
        (50.0e9, 0.45),
        (30.0e9, 0.25),
    ] {
        let coupling = PiezoCoupling::new(e_mod, nu, 15.1, -5.2, 12.3);
        let material = PiezoMaterial::isotropic_with_piezo(e_mod, nu, 1200.0, 15.1, -5.2, 12.3);
        let d = material.d_matrix().expect("valid isotropic material");
        assert_close(coupling.d33(), d[2][2], 1e-9, "d33");
        assert_close(coupling.d31(), d[0][2], 1e-9, "d31");
    }
}

#[test]
fn d33_follows_the_poisson_corrected_formula() {
    let (e_mod, nu, e33, e31) = (67.0e9, 0.31, 15.1, -5.2);
    let coupling = PiezoCoupling::new(e_mod, nu, e33, e31, 12.3);
    assert_close(coupling.d33(), (e33 - 2.0 * nu * e31) / e_mod, 1e-14, "d33");
    assert_close(
        coupling.d31(),
        ((1.0 - nu) * e31 - nu * e33) / e_mod,
        1e-14,
        "d31",
    );
    // The laterally clamped shortcut e33 / C33 is about 40 % too small.
    let c33 = e_mod * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    assert!((coupling.d33() - e33 / c33).abs() > 0.3 * coupling.d33());
    // d31 is negative for e31 < 0, e33 > 0.
    assert!(coupling.d31() < 0.0);
}

#[test]
fn degenerate_young_modulus_gives_zero_coefficients() {
    let coupling = PiezoCoupling::new(0.0, 0.3, 15.0, -5.0, 10.0);
    assert_eq!(coupling.d33(), 0.0);
    assert_eq!(coupling.d31(), 0.0);
}

#[test]
fn sensor_charge_uses_the_corrected_coefficient() {
    let sensor = PiezoSensor::new(67.0e9, 0.31, 15.1, -5.2, 12.3, 1e-4, 1e-3, 1200.0, 1e6);
    let d33 = sensor.material.d33();
    assert_close(
        sensor.short_circuit_charge(2.0e6),
        d33 * 2.0e6 * 1e-4,
        1e-14,
        "Q",
    );
}

#[test]
fn actuator_free_strain_comes_from_the_stiffness_and_coupling_matrices() {
    let material = PiezoMaterial::isotropic_with_piezo(67.0e9, 0.31, 1200.0, 15.1, -5.2, 12.3);
    let d = material.d_matrix().expect("valid isotropic material");
    let actuator = PiezoActuator {
        length: 2.0e-3,
        area: 1e-4,
        material,
        voltage: 150.0,
    };
    let strain = actuator.free_strain().expect("valid actuator");
    assert_close(strain, d[2][2] * 150.0 / 2.0e-3, 1e-12, "free strain");
    assert_close(
        actuator.tip_displacement().expect("valid actuator"),
        strain * 2.0e-3,
        1e-12,
        "tip displacement",
    );

    // The stress-free state: sigma = c epsilon - e^T E vanishes for epsilon = d E.
    let field = 150.0 / 2.0e-3;
    let mut sigma = [0.0_f64; 6];
    for (j, sigma_j) in sigma.iter_mut().enumerate() {
        let strain_row: f64 = (0..6)
            .map(|i| actuator.material.elastic_stiffness[j][i] * d[i][2] * field)
            .sum();
        *sigma_j = strain_row - actuator.material.piezo_coupling[j][2] * field;
    }
    let scale = 15.1 * field;
    for (j, s) in sigma.iter().enumerate() {
        assert!(s.abs() < 1e-9 * scale, "sigma[{j}] = {s}");
    }
}

#[test]
fn actuator_with_invalid_data_reports_an_error() {
    let material = PiezoMaterial::isotropic_with_piezo(67.0e9, 0.31, 1200.0, 15.1, -5.2, 12.3);
    let mut actuator = PiezoActuator {
        length: 0.0,
        area: 1e-4,
        material,
        voltage: 10.0,
    };
    assert!(matches!(
        actuator.free_strain(),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        actuator.tip_displacement(),
        Err(Error::InvalidInput(_))
    ));

    actuator.length = 1e-3;
    actuator.material.elastic_stiffness = [[0.0; 6]; 6];
    assert!(matches!(actuator.free_strain(), Err(Error::SingularSystem)));

    // An incompressible skeleton has an infinite stiffness: not a finite matrix.
    let incompressible = PiezoMaterial::isotropic_with_piezo(67.0e9, 0.5, 1200.0, 15.1, -5.2, 12.3);
    assert!(matches!(
        incompressible.d_matrix(),
        Err(Error::InvalidInput(_))
    ));
}
