// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! The piezoelectric strain coefficients `d = s^E e` and everything derived from them.
//!
//! Before the fix the free strain used the constants `d33 = 400 pC/N`, `d31 = -175 pC/N`
//! for every material and `d33 = e33 / c33` (the coefficient of a laterally clamped bar,
//! 142 pC/N for PZT-5A) elsewhere.

use super::*;
use crate::error::Error;

/// `d33`, `d31` of a transversely isotropic ceramic from the closed-form inverse of the
/// normal-stress block: `delta = c33 (c11 + c12) - 2 c13^2`,
/// `d33 = ((c11 + c12) e33 - 2 c13 e31) / delta`, `d31 = (c33 e31 - c13 e33) / delta`.
fn closed_form_d(mat: &PiezoMaterial) -> (f64, f64) {
    let c = &mat.c_matrix;
    let (c11, c12, c13, c33) = (c[0][0], c[0][1], c[0][2], c[2][2]);
    let (e31, e33) = (mat.e_matrix[0][2], mat.e_matrix[2][2]);
    let delta = c33 * (c11 + c12) - 2.0 * c13 * c13;
    (
        ((c11 + c12) * e33 - 2.0 * c13 * e31) / delta,
        (c33 * e31 - c13 * e33) / delta,
    )
}

fn assert_close(actual: f64, expected: f64, relative: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs(),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn pzt5a_strain_coefficients_match_the_handbook_within_five_percent() {
    let d = PiezoMaterial::pzt5a().d_matrix().expect("PZT-5A");
    assert_close(d[2][2], 374e-12, 0.05, "d33");
    assert_close(d[0][2], -171e-12, 0.05, "d31");
    assert_close(d[1][2], d[0][2], 1e-12, "d32 = d31 by symmetry");
    // The poled ceramic has no shear response to a field along the poling axis.
    for row in &d[3..] {
        assert!(row[2].abs() < 1e-22, "{row:?}");
    }
}

#[test]
fn pzt5h_strain_coefficients_match_the_handbook_within_five_percent() {
    let d = PiezoMaterial::pzt5h().d_matrix().expect("PZT-5H");
    assert_close(d[2][2], 593e-12, 0.05, "d33");
    assert_close(d[0][2], -274e-12, 0.05, "d31");
}

#[test]
fn lu_route_agrees_with_the_closed_form_inverse() {
    for mat in [PiezoMaterial::pzt5a(), PiezoMaterial::pzt5h()] {
        let d = mat.d_matrix().expect("valid material");
        let (d33, d31) = closed_form_d(&mat);
        assert_close(d[2][2], d33, 1e-12, "d33");
        assert_close(d[0][2], d31, 1e-12, "d31");
        // The clamped-bar shortcut e33 / c33 is far off.
        let clamped = mat.e_matrix[2][2] / mat.c_matrix[2][2];
        assert!(
            (clamped - d33).abs() > 0.3 * d33,
            "e33/c33 = {clamped} vs {d33}"
        );
    }
}

#[test]
fn compliance_is_the_inverse_of_the_stiffness() {
    let mat = PiezoMaterial::pzt5a();
    let s = mat.compliance_matrix().expect("PZT-5A");
    for (i, c_row) in mat.c_matrix.iter().enumerate() {
        for j in 0..6 {
            let product: f64 = c_row.iter().zip(&s).map(|(c_ik, s_k)| c_ik * s_k[j]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (product - expected).abs() < 1e-12,
                "(c s)[{i}][{j}] = {product}"
            );
        }
    }
    // Handbook compliances of PZT-5A (1e-12 / Pa): s11 = 16.4, s33 = 18.8 within 5 %.
    assert_close(s[0][0], 16.4e-12, 0.05, "s11");
    assert_close(s[2][2], 18.8e-12, 0.05, "s33");
}

#[test]
fn free_strain_leaves_the_body_stress_free_for_any_field_direction() {
    for mat in [PiezoMaterial::pzt5a(), PiezoMaterial::pzt5h()] {
        for e_field in [
            [0.0, 0.0, 2.0e5],
            [1.0e4, -2.0e4, 3.0e5],
            [5.0e5, 0.0, 0.0],
            [0.0, 4.0e5, -1.0e5],
        ] {
            let strain = mat.free_strain(&e_field).expect("valid material");
            let sigma = mat.mechanical_stress(&strain, &e_field);
            let scale =
                e_field.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * mat.e_matrix[2][2].abs();
            for (i, s) in sigma.iter().enumerate() {
                assert!(
                    s.abs() < 1e-9 * scale,
                    "{}: sigma[{i}] = {s} for E = {e_field:?}",
                    mat.name
                );
            }
        }
    }
}

#[test]
fn actuator_free_strain_follows_the_material() {
    let (thickness, voltage) = (0.5e-3, 100.0);
    let field = voltage / thickness;
    let mut strains = Vec::new();
    for mat in [PiezoMaterial::pzt5a(), PiezoMaterial::pzt5h()] {
        let d = mat.d_matrix().expect("valid material");
        let actuator = PiezoActuator::new(mat, 1e-4, thickness);
        let strain = actuator.free_strain(voltage).expect("valid actuator");
        assert_close(strain[2], d[2][2] * field, 1e-12, "epsilon_3");
        assert_close(strain[0], d[0][2] * field, 1e-12, "epsilon_1");
        assert_close(strain[1], d[1][2] * field, 1e-12, "epsilon_2");
        assert!(strain[0] < 0.0 && strain[2] > 0.0);
        strains.push(strain[2]);
    }
    // PZT-5H is the more active ceramic; the old constants made both identical.
    assert!(strains[1] > 1.4 * strains[0], "{strains:?}");
}

#[test]
fn free_strain_scales_with_the_material_constants() {
    let base = PiezoMaterial::pzt5a();
    let d = base.d_matrix().expect("valid material");

    // Doubling the coupling coefficients doubles d.
    let mut strong = PiezoMaterial::pzt5a();
    for row in strong.e_matrix.iter_mut() {
        for value in row.iter_mut() {
            *value *= 2.0;
        }
    }
    let d_strong = strong.d_matrix().expect("valid material");
    assert_close(d_strong[2][2], 2.0 * d[2][2], 1e-12, "d33 with 2e");
    assert_close(d_strong[0][2], 2.0 * d[0][2], 1e-12, "d31 with 2e");

    // Doubling the stiffness halves d.
    let mut stiff = PiezoMaterial::pzt5a();
    for row in stiff.c_matrix.iter_mut() {
        for value in row.iter_mut() {
            *value *= 2.0;
        }
    }
    let d_stiff = stiff.d_matrix().expect("valid material");
    assert_close(d_stiff[2][2], 0.5 * d[2][2], 1e-12, "d33 with 2c");
}

#[test]
fn element_induced_strain_and_free_displacement_use_the_same_coefficients() {
    let mat = PiezoMaterial::pzt5a();
    let (d33, d31) = closed_form_d(&mat);
    let element = PiezoElement::new(mat, 1e-9);

    let strain = element.induced_strain(50.0, 1e-3).expect("valid element");
    assert_close(strain[2], d33 * 50.0 / 1e-3, 1e-12, "epsilon_3");
    assert_close(strain[0], d31 * 50.0 / 1e-3, 1e-12, "epsilon_1");

    assert_close(
        element
            .compute_effective_piezo_charge()
            .expect("valid element"),
        d33,
        1e-12,
        "d33",
    );
    assert_close(
        element.compute_free_displacement(120.0).expect("valid"),
        d33 * 120.0,
        1e-12,
        "free displacement",
    );
}

#[test]
fn blocking_force_equals_the_force_of_the_axially_clamped_free_lateral_state() {
    // Constitutive check, independent of the compliance matrix: with sigma_1 = sigma_2 = 0
    // the lateral strains are epsilon_1 = epsilon_2 = e31 E / (c11 + c12), and the axial
    // stress needed to keep epsilon_3 = 0 is
    //   sigma_3 = 2 c13 e31 E / (c11 + c12) - e33 E.
    let mat = PiezoMaterial::pzt5a();
    let c = mat.c_matrix;
    let (e31, e33) = (mat.e_matrix[0][2], mat.e_matrix[2][2]);
    let (area, thickness, voltage) = (1e-4, 0.5e-3, 100.0);
    let field = voltage / thickness;
    let lateral = e31 * field / (c[0][0] + c[0][1]);
    let strain = [lateral, lateral, 0.0, 0.0, 0.0, 0.0];
    let sigma = mat.mechanical_stress(&strain, &[0.0, 0.0, field]);
    assert!(
        sigma[0].abs() < 1e-9 * e33 * field && sigma[1].abs() < 1e-9 * e33 * field,
        "the lateral faces must be stress free: {sigma:?}"
    );
    let expected = -sigma[2] * area;

    let actuator = PiezoActuator::new(mat, area, thickness);
    let force = actuator.blocking_force(voltage).expect("valid actuator");
    assert!(force > 0.0);
    assert_close(force, expected, 1e-10, "blocking force");
    // The former c33-based formula was off by a factor of about 2.
    let old_formula = actuator.free_strain(voltage).expect("valid")[2] * c[2][2] * area;
    assert!(old_formula > 1.5 * force, "{old_formula} vs {force}");
}

#[test]
fn invalid_actuators_and_materials_are_errors() {
    for thickness in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        let actuator = PiezoActuator::new(PiezoMaterial::pzt5a(), 1e-4, thickness);
        assert!(
            matches!(actuator.free_strain(10.0), Err(Error::InvalidInput(_))),
            "free_strain with thickness {thickness}"
        );
        assert!(
            matches!(actuator.blocking_force(10.0), Err(Error::InvalidInput(_))),
            "blocking_force with thickness {thickness}"
        );
        let element = PiezoElement::new(PiezoMaterial::pzt5a(), 1e-9);
        assert!(matches!(
            element.induced_strain(10.0, thickness),
            Err(Error::InvalidInput(_))
        ));
    }

    let mut singular = PiezoMaterial::pzt5a();
    singular.c_matrix[3] = [0.0; 6];
    assert!(matches!(singular.d_matrix(), Err(Error::SingularSystem)));
    assert!(matches!(
        singular.compliance_matrix(),
        Err(Error::SingularSystem)
    ));
    assert!(matches!(
        PiezoActuator::new(singular, 1e-4, 1e-3).free_strain(1.0),
        Err(Error::SingularSystem)
    ));

    let mut not_finite = PiezoMaterial::pzt5a();
    not_finite.c_matrix[0][0] = f64::NAN;
    assert!(matches!(not_finite.d_matrix(), Err(Error::InvalidInput(_))));
    let mut bad_coupling = PiezoMaterial::pzt5a();
    bad_coupling.e_matrix[2][2] = f64::INFINITY;
    assert!(matches!(
        bad_coupling.d_matrix(),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn e_form_and_handbook_d_form_describe_the_same_ceramic() {
    let derived = PiezoMaterial::pzt5a().d_matrix().expect("PZT-5A");
    let handbook = PiezoMaterialDForm::pzt5a_d_form();
    // `PiezoMaterialDForm::d_matrix` is [3][6]: d33 = [2][2], d31 = [2][0].
    assert_close(derived[2][2], handbook.d_matrix[2][2], 0.01, "d33");
    assert_close(derived[0][2], handbook.d_matrix[2][0], 0.01, "d31");
}

#[test]
fn compliance_and_d_helpers_work_on_raw_matrices() {
    let mat = PiezoMaterial::pzt5a();
    let s = elastic_compliance_matrix(&mat.c_matrix).expect("invertible");
    let d = piezo_d_matrix(&mat.c_matrix, &mat.e_matrix).expect("invertible");
    // d = s e
    for (j, d_row) in d.iter().enumerate() {
        for (k, &value) in d_row.iter().enumerate() {
            let expected: f64 = (0..6).map(|i| s[j][i] * mat.e_matrix[i][k]).sum();
            assert_close(value, expected, 1e-10, "d = s e");
        }
    }
}
