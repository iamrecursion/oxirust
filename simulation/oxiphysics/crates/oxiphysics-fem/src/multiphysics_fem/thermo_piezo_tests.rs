// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the consistent thermal load vector of [`ThermoElasticFem`] and of the PZT-5A
//! piezoelectric coupling.

use super::*;

/// Kuhn (Freudenthal) decomposition of the unit cube into six tetrahedra that all share
/// the main diagonal 0-7.  Node `i` sits at `(i & 1, (i >> 1) & 1, (i >> 2) & 1)`.
fn unit_cube_six_tets() -> (Vec<[f64; 3]>, Vec<[usize; 4]>) {
    let nodes: Vec<[f64; 3]> = (0..8usize)
        .map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
        .collect();
    let tets = vec![
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ];
    (nodes, tets)
}

fn thermo_elastic_cube() -> ThermoElasticFem {
    let (nodes, tets) = unit_cube_six_tets();
    ThermoElasticFem::new(2e11, 0.3, 12e-6, 293.0, 50.0, 3.8e6, 8)
        .with_mesh(nodes, tets)
        .expect("valid cube mesh")
}

/// Bulk modulus `K = E / (3 (1 - 2 nu))` of the test material.
fn bulk_modulus(fem: &ThermoElasticFem) -> f64 {
    fem.young_modulus / (3.0 * (1.0 - 2.0 * fem.poisson_ratio))
}

#[test]
fn test_thermal_load_vector_requires_mesh() {
    let fem = ThermoElasticFem::new(2e11, 0.3, 12e-6, 293.0, 50.0, 3.8e6, 10);
    assert!(
        matches!(fem.thermal_load_vector(), Err(Error::InvalidInput(_))),
        "a model without a mesh has no consistent thermal load"
    );
}

#[test]
fn test_thermal_load_vector_single_tet_matches_hand_calculation() {
    // Corner tetrahedron: V = 1/6 and grad N = (-1,-1,-1), e_x, e_y, e_z, so the nodal
    // force is f_a = V * sigma_th * grad N_a with sigma_th = 3 K alpha dT.
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut fem = ThermoElasticFem::new(2e11, 0.3, 12e-6, 293.0, 50.0, 3.8e6, 4)
        .with_mesh(nodes, vec![[0, 1, 2, 3]])
        .expect("valid mesh");
    fem.set_temperatures(&[393.0; 4]);
    let sigma_th = 3.0 * bulk_modulus(&fem) * fem.thermal_expansion * 100.0;
    let f = fem.thermal_load_vector().expect("valid thermal load");
    assert_eq!(f.len(), 12);
    let s6 = sigma_th / 6.0;
    let expected = [-s6, -s6, -s6, s6, 0.0, 0.0, 0.0, s6, 0.0, 0.0, 0.0, s6];
    for (i, (fi, ei)) in f.iter().zip(expected.iter()).enumerate() {
        assert!(
            (fi - ei).abs() < 1e-6 * sigma_th,
            "f[{i}] = {fi}, expected {ei}"
        );
    }
}

#[test]
fn test_thermal_load_vector_uniform_heating_free_cube() {
    let (nodes, tets) = unit_cube_six_tets();
    let mut fem = thermo_elastic_cube();
    let delta_t = 100.0;
    fem.set_temperatures(&[fem.reference_temperature + delta_t; 8]);
    let f = fem.thermal_load_vector().expect("valid thermal load");
    assert_eq!(f.len(), 24);

    // The six tetrahedra tile the unit cube exactly.
    let volume: f64 = tets
        .iter()
        .map(|t| {
            let p: Vec<Vec3> = t
                .iter()
                .map(|&i| Vec3::new(nodes[i][0], nodes[i][1], nodes[i][2]))
                .collect();
            LinearTetrahedron::volume(&[p[0], p[1], p[2], p[3]])
        })
        .sum();
    assert!((volume - 1.0).abs() < 1e-14, "cube volume {volume}");

    let stress_scale = 3.0 * bulk_modulus(&fem) * fem.thermal_expansion * delta_t;
    for axis in 0..3 {
        // A free body must carry no net thermal force ...
        let resultant: f64 = (0..8).map(|n| f[3 * n + axis]).sum();
        assert!(
            resultant.abs() < 1e-9 * stress_scale,
            "net force along axis {axis}: {resultant}"
        );
        // ... and the first moment reproduces 3 K alpha dT V (the virial of the
        // isotropic eigenstress).
        let moment: f64 = (0..8).map(|n| f[3 * n + axis] * nodes[n][axis]).sum();
        assert!(
            (moment - stress_scale * volume).abs() < 1e-9 * stress_scale,
            "first moment along axis {axis}: {moment}, expected {}",
            stress_scale * volume
        );
    }
    // The off-diagonal moments of an isotropic eigenstress vanish.
    let cross: f64 = (0..8).map(|n| f[3 * n] * nodes[n][1]).sum();
    assert!(cross.abs() < 1e-9 * stress_scale, "cross moment {cross}");
}

#[test]
fn test_thermal_load_vector_linear_temperature_field() {
    // T - T_ref = 50 x on the unit cube: the mean temperature rise is 25 K, so the first
    // moment along x is 3 K alpha * 25, while the resultant still vanishes.
    let (nodes, _) = unit_cube_six_tets();
    let mut fem = thermo_elastic_cube();
    let temps: Vec<f64> = nodes
        .iter()
        .map(|p| fem.reference_temperature + 50.0 * p[0])
        .collect();
    fem.set_temperatures(&temps);
    let f = fem.thermal_load_vector().expect("valid thermal load");
    let stress_scale = 3.0 * bulk_modulus(&fem) * fem.thermal_expansion;
    for axis in 0..3 {
        let resultant: f64 = (0..8).map(|n| f[3 * n + axis]).sum();
        assert!(
            resultant.abs() < 1e-9 * stress_scale * 50.0,
            "net force along axis {axis}: {resultant}"
        );
    }
    let moment_x: f64 = (0..8).map(|n| f[3 * n] * nodes[n][0]).sum();
    assert!(
        (moment_x - stress_scale * 25.0).abs() < 1e-9 * stress_scale * 25.0,
        "first moment {moment_x}, expected {}",
        stress_scale * 25.0
    );
}

#[test]
fn test_thermal_load_vector_unheated_body_has_no_load() {
    let fem = thermo_elastic_cube();
    let f = fem.thermal_load_vector().expect("valid thermal load");
    assert!(
        f.iter().all(|v| v.abs() < 1e-12),
        "T = T_ref must be load-free"
    );
}

#[test]
fn test_thermal_load_vector_is_reproducible() {
    let mut fem = thermo_elastic_cube();
    let temps: Vec<f64> = (0..8).map(|i| 293.0 + 13.7 * i as f64).collect();
    fem.set_temperatures(&temps);
    let a = fem.thermal_load_vector().expect("valid thermal load");
    let b = fem.thermal_load_vector().expect("valid thermal load");
    assert_eq!(a, b, "assembly must be bit-for-bit reproducible");
}

#[test]
fn test_with_mesh_rejects_inconsistent_meshes() {
    let (nodes, tets) = unit_cube_six_tets();
    let base = ThermoElasticFem::new(2e11, 0.3, 12e-6, 293.0, 50.0, 3.8e6, 8);
    // Wrong node count.
    assert!(matches!(
        base.clone().with_mesh(nodes[..7].to_vec(), tets.clone()),
        Err(Error::InvalidInput(_))
    ));
    // Tetrahedron referencing a node that does not exist.
    assert!(matches!(
        base.clone().with_mesh(nodes.clone(), vec![[0, 1, 3, 8]]),
        Err(Error::InvalidInput(_))
    ));
    // Non-finite coordinate.
    let mut bad_nodes = nodes.clone();
    bad_nodes[3][1] = f64::NAN;
    assert!(matches!(
        base.with_mesh(bad_nodes, tets),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn test_thermal_load_vector_rejects_degenerate_and_inconsistent_state() {
    // Four coplanar nodes: zero volume.
    let flat = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let fem = ThermoElasticFem::new(2e11, 0.3, 12e-6, 293.0, 50.0, 3.8e6, 4)
        .with_mesh(flat, vec![[0, 1, 2, 3]])
        .expect("indices are valid");
    assert!(matches!(
        fem.thermal_load_vector(),
        Err(Error::InvalidInput(_))
    ));

    // A temperature field that no longer matches the mesh.
    let mut fem = thermo_elastic_cube();
    fem.temperatures.pop();
    assert!(matches!(
        fem.thermal_load_vector(),
        Err(Error::InvalidInput(_))
    ));

    // A non-finite temperature.
    let mut fem = thermo_elastic_cube();
    fem.temperatures[2] = f64::INFINITY;
    assert!(matches!(
        fem.thermal_load_vector(),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn test_piezoelectric_compute_stress() {
    let fem = make_pzt5a(30, 10);
    let strain = [1e-4, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e_field = [0.0, 0.0, 0.0];
    let sigma = fem.compute_stress(&strain, &e_field);
    // C11 e11 and C12 e11 = C21 e11 on the first two normals, C13 e11 on the third.
    assert!((sigma[0] - 121e9 * 1e-4).abs() < 1e-3);
    assert!((sigma[1] - 75.4e9 * 1e-4).abs() < 1e-3);
    assert!((sigma[2] - 75.2e9 * 1e-4).abs() < 1e-3);
    assert!(sigma[3..].iter().all(|s| s.abs() < 1e-12));
}

#[test]
fn test_piezoelectric_field_along_poling_axis() {
    // sigma = -e^T E: a field E3 along the poling axis gives sigma_11 = sigma_22 = -e31 E3
    // and sigma_33 = -e33 E3, with e31 = -5.4 C/m^2 and e33 = 15.8 C/m^2.
    let fem = make_pzt5a(30, 10);
    let sigma = fem.compute_stress(&[0.0; 6], &[0.0, 0.0, 1.0e3]);
    let expected = [5.4e3, 5.4e3, -15.8e3, 0.0, 0.0, 0.0];
    for (i, (s, e)) in sigma.iter().zip(expected.iter()).enumerate() {
        assert!((s - e).abs() < 1e-9, "sigma[{i}] = {s}, expected {e}");
    }
}

#[test]
fn test_piezoelectric_transverse_fields_excite_shear_only() {
    // e15 couples E1 to the 13 shear and E2 to the 23 shear (Voigt order 11,22,33,23,13,12).
    let fem = make_pzt5a(30, 10);
    let sigma_e1 = fem.compute_stress(&[0.0; 6], &[1.0e3, 0.0, 0.0]);
    let expected_e1 = [0.0, 0.0, 0.0, 0.0, -12.3e3, 0.0];
    let sigma_e2 = fem.compute_stress(&[0.0; 6], &[0.0, 1.0e3, 0.0]);
    let expected_e2 = [0.0, 0.0, 0.0, -12.3e3, 0.0, 0.0];
    for i in 0..6 {
        assert!(
            (sigma_e1[i] - expected_e1[i]).abs() < 1e-9,
            "E1: sigma[{i}] = {}",
            sigma_e1[i]
        );
        assert!(
            (sigma_e2[i] - expected_e2[i]).abs() < 1e-9,
            "E2: sigma[{i}] = {}",
            sigma_e2[i]
        );
    }
}

#[test]
fn test_piezoelectric_electric_displacement() {
    let fem = make_pzt5a(30, 10);
    let kappa_33 = 1700.0 * 8.854e-12;
    let kappa_11 = 1650.0 * 8.854e-12;
    // Field only: D = kappa E.
    let d_field = fem.compute_electric_displacement(&[0.0; 6], &[0.0, 0.0, 1000.0]);
    assert!((d_field[2] - kappa_33 * 1000.0).abs() < 1e-18);
    assert!(d_field[0].abs() < 1e-18 && d_field[1].abs() < 1e-18);
    let d_field_x = fem.compute_electric_displacement(&[0.0; 6], &[1000.0, 0.0, 0.0]);
    assert!((d_field_x[0] - kappa_11 * 1000.0).abs() < 1e-18);

    // Strain only (direct effect): D3 = e31 (e11 + e22) + e33 e33.
    let strain = [1e-4, 1e-4, 2e-4, 0.0, 0.0, 0.0];
    let d_strain = fem.compute_electric_displacement(&strain, &[0.0; 3]);
    let expected_d3 = -5.4 * 2e-4 + 15.8 * 2e-4;
    assert!(
        (d_strain[2] - expected_d3).abs() < 1e-12,
        "D3 = {}",
        d_strain[2]
    );

    // Shear strains (direct effect): D1 = e15 gamma_13, D2 = e15 gamma_23.
    let shear = [0.0, 0.0, 0.0, 3e-4, 5e-4, 0.0];
    let d_shear = fem.compute_electric_displacement(&shear, &[0.0; 3]);
    assert!(
        (d_shear[0] - 12.3 * 5e-4).abs() < 1e-12,
        "D1 = {}",
        d_shear[0]
    );
    assert!(
        (d_shear[1] - 12.3 * 3e-4).abs() < 1e-12,
        "D2 = {}",
        d_shear[1]
    );
    assert!(d_shear[2].abs() < 1e-12);
}

#[test]
fn test_piezoelectric_capacitance() {
    let fem = make_pzt5a(30, 10);
    let c = fem.capacitance_scalar(0.01, 0.001);
    // Electrodes normal to the poling axis: C = kappa_33 A / d.
    let expected = 1700.0 * 8.854e-12 * 0.01 / 0.001;
    assert!((c - expected).abs() < 1e-9 * expected, "C = {c}");
    assert_eq!(fem.capacitance_scalar(0.01, 0.0), 0.0);
}
