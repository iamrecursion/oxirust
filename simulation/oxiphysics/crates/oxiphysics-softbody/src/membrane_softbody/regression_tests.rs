// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the membrane element (CST with Green-Lagrange strain),
//! the Mooney-Rivlin balloon inflation law and the membrane mesh dynamics.

use super::*;

/// Rodrigues rotation of `v` about `axis` by `angle`.
fn rotate(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let k = vec_normalize(axis);
    let (sin_a, cos_a) = angle.sin_cos();
    vec_add(
        vec_add(vec_scale(v, cos_a), vec_scale(vec_cross(k, v), sin_a)),
        vec_scale(k, vec_dot(k, v) * (1.0 - cos_a)),
    )
}

/// Generic (non axis-aligned) rest triangle in 3-D.
fn generic_rest() -> Vec<[f64; 3]> {
    vec![[0.10, 0.20, 0.05], [1.30, 0.10, 0.20], [0.40, 1.10, -0.10]]
}

/// A deformed copy of [`generic_rest`] with roughly 10 % strains and shear.
fn generic_deformed() -> Vec<[f64; 3]> {
    vec![[0.12, 0.18, 0.06], [1.42, 0.05, 0.30], [0.35, 1.25, -0.05]]
}

fn element(rest: &[[f64; 3]], tension_only: bool) -> MembraneTriangle {
    let mut tri = MembraneTriangle::new([0, 1, 2], 2.0e-3, 5.0e6, 0.3, rest);
    tri.tension_only = tension_only;
    tri
}

fn element_forces(tri: &MembraneTriangle, current: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0; 3]; current.len()];
    tri.apply_forces(current, &mut forces);
    forces
}

fn max_abs(forces: &[[f64; 3]]) -> f64 {
    forces
        .iter()
        .flat_map(|f| f.iter())
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

// ---------------------------------------------------------------------------
// Mooney-Rivlin balloon inflation (F4192)
// ---------------------------------------------------------------------------

#[test]
fn inflation_pressure_vanishes_at_unit_stretch() {
    let p = spherical_inflation_pressure(0.1, 0.002, 2.0e5, 7.0e4, 1.0);
    assert!(p.abs() < 1e-9, "p(1) must be 0, got {p}");
}

#[test]
fn inflation_pressure_matches_stress_and_laplace_derivation() {
    // Independent route: equibiaxial Cauchy stress of a Mooney-Rivlin solid,
    // then Laplace's law p = 2 sigma t / r with the deformed t and r.
    let (r0, t0, c1, c2) = (0.15, 0.003, 3.0e5, 1.2e5);
    for &lam in &[1.05_f64, 1.2, 1.383, 1.7, 2.4, 3.0] {
        let sigma = 2.0 * (lam * lam - lam.powi(-4)) * (c1 + c2 * lam * lam);
        let t = balloon_current_thickness(t0, lam);
        let r = balloon_current_radius(r0, lam);
        let expected = 2.0 * sigma * t / r;
        let p = spherical_inflation_pressure(r0, t0, c1, c2, lam);
        assert!(
            (p - expected).abs() <= 1e-12 * expected.abs(),
            "lambda {lam}: {p} vs {expected}"
        );
    }
}

#[test]
fn neo_hookean_balloon_has_limit_point_at_seven_to_one_sixth() {
    let (r0, t0, c1) = (0.1, 0.001, 1.0e5);
    let pressure = |lam: f64| spherical_inflation_pressure(r0, t0, c1, 0.0, lam);
    // Scan the stretch range [1, 3] at 1e-4 resolution for the pressure maximum.
    let mut best_lam = 1.0;
    let mut best_p = f64::NEG_INFINITY;
    for step in 0..=20_000_i32 {
        let lam = 1.0 + f64::from(step) * 1.0e-4;
        let p = pressure(lam);
        if p > best_p {
            best_p = p;
            best_lam = lam;
        }
    }
    let expected = 7.0_f64.powf(1.0 / 6.0);
    assert!(
        (best_lam - expected).abs() < 1e-3,
        "pressure maximum at {best_lam}, expected {expected}"
    );
    assert!(pressure(1.383) > pressure(1.2));
    assert!(pressure(1.383) > pressure(2.0));
    // Closed form of the peak pressure: 4 C1 t0/R0 (7^-1/6 - 7^-7/6).
    let peak = 4.0 * c1 * t0 / r0 * (expected.recip() - expected.powi(-7));
    assert!((best_p - peak).abs() < 1e-6 * peak, "{best_p} vs {peak}");
}

#[test]
fn mooney_rivlin_second_term_scales_with_stretch_squared() {
    // For C1 = 0 the pressure is 4 C2 t0/R0 (lambda - lambda^-5): monotonic, no limit point.
    let (r0, t0, c2) = (0.1, 0.001, 8.0e4);
    let mut previous = 0.0;
    for step in 1..=400_i32 {
        let lam = 1.0 + f64::from(step) * 0.005;
        let p = spherical_inflation_pressure(r0, t0, 0.0, c2, lam);
        let expected = 4.0 * t0 / r0 * c2 * (lam - lam.powi(-5));
        assert!((p - expected).abs() <= 1e-12 * expected.abs());
        assert!(p > previous, "C1 = 0 pressure must increase with stretch");
        previous = p;
    }
}

#[test]
fn deflated_balloon_has_negative_pressure() {
    let p = spherical_inflation_pressure(0.1, 0.001, 1.0e5, 2.0e4, 0.8);
    assert!(p < 0.0, "a compressed shell needs suction, got {p}");
}

// ---------------------------------------------------------------------------
// Tension-field helpers
// ---------------------------------------------------------------------------

#[test]
fn principal_values_of_known_tensor() {
    let (hi, lo) = principal_values([[5.0, 2.0], [2.0, 2.0]]);
    // Eigenvalues of [[5, 2], [2, 2]] are 6 and 1.
    assert!((hi - 6.0).abs() < 1e-12 && (lo - 1.0).abs() < 1e-12);
}

#[test]
fn tensile_part_keeps_tension_and_drops_compression() {
    // Pure tension is unchanged.
    let tension = [[4.0, 1.0], [1.0, 3.0]];
    assert_eq!(tensile_part(tension), tension);
    // Pure compression vanishes.
    assert_eq!(tensile_part([[-4.0, 1.0], [1.0, -3.0]]), [[0.0; 2]; 2]);
    // Mixed state: S = R diag(3, -2) R^T keeps only 3 n n^T along its own direction.
    let phi = 0.7_f64;
    let (s, c) = phi.sin_cos();
    let mixed = [
        [3.0 * c * c - 2.0 * s * s, 5.0 * c * s],
        [5.0 * c * s, 3.0 * s * s - 2.0 * c * c],
    ];
    let relaxed = tensile_part(mixed);
    let expected = [[3.0 * c * c, 3.0 * c * s], [3.0 * c * s, 3.0 * s * s]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (relaxed[i][j] - expected[i][j]).abs() < 1e-12,
                "entry ({i},{j}): {} vs {}",
                relaxed[i][j],
                expected[i][j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// CST element (F4191)
// ---------------------------------------------------------------------------

#[test]
fn rest_configuration_has_identity_deformation_gradient() {
    let rest = generic_rest();
    let tri = element(&rest, false);
    let (f, _) = tri
        .deformation_gradient(&rest)
        .expect("non-degenerate triangle");
    let c = MembraneTriangle::right_cauchy_green(&f);
    assert!((c[0][0] - 1.0).abs() < 1e-12);
    assert!((c[1][1] - 1.0).abs() < 1e-12);
    assert!(c[0][1].abs() < 1e-12);
    assert!(tri.strain_energy(&rest).abs() < 1e-15);
}

#[test]
fn rest_and_rigid_motion_produce_no_force() {
    let rest = generic_rest();
    for &tension_only in &[false, true] {
        let tri = element(&rest, tension_only);
        assert!(max_abs(&element_forces(&tri, &rest)) < 1e-6);
        let moved: Vec<[f64; 3]> = rest
            .iter()
            .map(|&p| vec_add(rotate(p, [0.3, -1.0, 0.5], 1.1), [4.0, -2.0, 7.0]))
            .collect();
        let forces = element_forces(&tri, &moved);
        assert!(
            max_abs(&forces) < 1e-6,
            "rigid motion must be force free, got {forces:?}"
        );
        assert!(tri.strain_energy(&moved).abs() < 1e-12);
    }
}

#[test]
fn element_forces_are_momentum_and_torque_free() {
    let rest = generic_rest();
    let current = generic_deformed();
    for &tension_only in &[false, true] {
        let mut tri = element(&rest, tension_only);
        tri.set_prestress(2.0e4, -1.0e4);
        let forces = element_forces(&tri, &current);
        let scale = max_abs(&forces);
        assert!(scale > 1.0, "the deformed element must carry load: {scale}");
        let net = forces.iter().fold([0.0; 3], |acc, &f| vec_add(acc, f));
        assert!(
            vec_len(net) < 1e-9 * scale,
            "net force {net:?} (scale {scale})"
        );
        let torque = current
            .iter()
            .zip(&forces)
            .fold([0.0; 3], |acc, (&x, &f)| vec_add(acc, vec_cross(x, f)));
        assert!(
            vec_len(torque) < 1e-9 * scale * 2.0,
            "net torque {torque:?} (scale {scale})"
        );
    }
}

#[test]
fn element_forces_are_negative_gradient_of_strain_energy() {
    // Finite-difference check of f = -dW/dx on a generic 3-D deformation,
    // with pre-stress, without tension-only relaxation (which has no potential).
    let rest = generic_rest();
    let current = generic_deformed();
    let mut tri = element(&rest, false);
    tri.set_prestress(1.5e5, -6.0e4);
    let forces = element_forces(&tri, &current);
    let scale = max_abs(&forces);
    assert!(scale > 10.0, "force scale {scale}");
    let h = 1.0e-6;
    for node in 0..3 {
        for axis in 0..3 {
            let mut plus = current.clone();
            let mut minus = current.clone();
            plus[node][axis] += h;
            minus[node][axis] -= h;
            let gradient = (tri.strain_energy(&plus) - tri.strain_energy(&minus)) / (2.0 * h);
            assert!(
                (forces[node][axis] + gradient).abs() < 1e-6 * scale,
                "node {node} axis {axis}: force {} vs -dW/dx {}",
                forces[node][axis],
                -gradient
            );
        }
    }
}

#[test]
fn planar_stretch_has_no_normal_force_component() {
    // Element in the XZ plane (y = 0), stretched and sheared in plane.
    let rest = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let current = vec![[0.0, 0.0, 0.0], [1.25, 0.0, 0.10], [0.05, 0.0, 0.85]];
    for &tension_only in &[false, true] {
        let tri = element(&rest, tension_only);
        let forces = element_forces(&tri, &current);
        assert!(
            max_abs(&forces) > 1.0,
            "in-plane stretch must load the element"
        );
        for force in &forces {
            assert_eq!(force[1], 0.0, "no out-of-plane force for planar strain");
        }
    }
    // The fully elastic element does negative work against the displacement.
    let tri = element(&rest, false);
    let forces = element_forces(&tri, &current);
    let work: f64 = forces
        .iter()
        .zip(current.iter().zip(&rest))
        .map(|(f, (x, x0))| vec_dot(*f, vec_sub(*x, *x0)))
        .sum();
    assert!(
        work < 0.0,
        "restoring forces oppose the displacement, W = {work}"
    );
}

#[test]
fn equibiaxial_stretch_matches_hydrostatic_edge_tractions() {
    // Right triangle with legs L stretched by lambda in both directions: the
    // Cauchy stress is the hydrostatic S = E e / (1 - nu) with e = (lambda^2 - 1)/2.
    let (leg, lam) = (2.0_f64, 1.1_f64);
    let rest = vec![[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [0.0, leg, 0.0]];
    let current = vec![
        [0.0, 0.0, 0.0],
        [lam * leg, 0.0, 0.0],
        [0.0, lam * leg, 0.0],
    ];
    let tri = element(&rest, true);
    let strain = 0.5 * (lam * lam - 1.0);
    let sigma = tri.youngs_modulus * strain / (1.0 - tri.poissons_ratio);
    let tau = tri.thickness * sigma;
    // Internal force on a node = tau/2 * (length of the opposite edge) * (its outward normal).
    let half = 0.5 * tau * lam * leg;
    let forces = element_forces(&tri, &current);
    let expected = [[half, half, 0.0], [-half, 0.0, 0.0], [0.0, -half, 0.0]];
    for node in 0..3 {
        for axis in 0..3 {
            assert!(
                (forces[node][axis] - expected[node][axis]).abs() < 1e-9 * half,
                "node {node} axis {axis}: {} vs {}",
                forces[node][axis],
                expected[node][axis]
            );
        }
    }
    assert!((tri.membrane_tension(&current) - tau).abs() < 1e-9 * tau);
    let (s_max, s_min) = tri.principal_stresses(&current);
    assert!((s_max - sigma).abs() < 1e-6 * sigma && (s_min - sigma).abs() < 1e-6 * sigma);
    assert_eq!(tri.element_wrinkling_state(&current), WrinklingState::Taut);
}

#[test]
fn prestressed_rest_element_pulls_its_nodes_inwards() {
    let leg = 2.0_f64;
    let rest = vec![[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [0.0, leg, 0.0]];
    let mut tri = element(&rest, true);
    tri.set_prestress(1.0e5, 1.0e5);
    let forces = element_forces(&tri, &rest);
    // tau = t * sigma0 = 200 N/m; node 0 force = tau/2 * (L sqrt(2)) * (1, 1)/sqrt(2).
    let half = 0.5 * 200.0 * leg;
    assert!((forces[0][0] - half).abs() < 1e-9 && (forces[0][1] - half).abs() < 1e-9);
    assert!((forces[1][0] + half).abs() < 1e-9 && forces[1][1].abs() < 1e-9);
    assert!(forces[2][0].abs() < 1e-9 && (forces[2][1] + half).abs() < 1e-9);
    assert!((tri.membrane_tension(&rest) - 200.0).abs() < 1e-9);
}

#[test]
fn tension_only_element_ignores_uniform_compression() {
    let leg = 2.0_f64;
    let rest = vec![[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [0.0, leg, 0.0]];
    let squeezed = vec![
        [0.0, 0.0, 0.0],
        [0.5 * leg, 0.0, 0.0],
        [0.0, 0.5 * leg, 0.0],
    ];
    let tri = element(&rest, true);
    assert_eq!(max_abs(&element_forces(&tri, &squeezed)), 0.0);
    assert_eq!(tri.membrane_tension(&squeezed), 0.0);
    assert_eq!(
        tri.element_wrinkling_state(&squeezed),
        WrinklingState::Slack
    );
    // The fully elastic element resists the same compression.
    let elastic = element(&rest, false);
    assert!(max_abs(&element_forces(&elastic, &squeezed)) > 1.0);
    assert!(elastic.membrane_tension(&squeezed) < 0.0);
}

#[test]
fn tension_only_element_keeps_only_the_tensile_direction() {
    // nu = 0: stretch x by 1.1 and squeeze y to 0.8. Only S_xx survives.
    let leg = 2.0_f64;
    let rest = vec![[0.0, 0.0, 0.0], [leg, 0.0, 0.0], [0.0, leg, 0.0]];
    let current = vec![
        [0.0, 0.0, 0.0],
        [1.1 * leg, 0.0, 0.0],
        [0.0, 0.8 * leg, 0.0],
    ];
    let mut tri = MembraneTriangle::new([0, 1, 2], 2.0e-3, 5.0e6, 0.0, &rest);
    tri.tension_only = true;
    let s_xx = 5.0e6 * 0.5 * (1.1 * 1.1 - 1.0);
    // P_xx = lambda_x S_xx; node 1 force = -(L t / 2) P_xx, node 0 the opposite.
    let expected = 0.5 * leg * 2.0e-3 * 1.1 * s_xx;
    let forces = element_forces(&tri, &current);
    assert!((forces[0][0] - expected).abs() < 1e-9 * expected);
    assert!((forces[1][0] + expected).abs() < 1e-9 * expected);
    assert_eq!(
        forces[2],
        [0.0, 0.0, 0.0],
        "the squeezed direction carries nothing"
    );
    assert_eq!(forces[0][1], 0.0);
    assert_eq!(
        tri.element_wrinkling_state(&current),
        WrinklingState::Wrinkled
    );
    // Without the tension-only relaxation the squeezed direction pushes back.
    tri.tension_only = false;
    let elastic = element_forces(&tri, &current);
    assert!(
        elastic[2][1].abs() > 100.0,
        "elastic y force {:?}",
        elastic[2]
    );
}

#[test]
fn degenerate_rest_triangle_carries_no_force() {
    let collinear = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let tri = element(&collinear, false);
    let stretched = vec![[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [2.0, 0.0, 0.3]];
    assert!(tri.deformation_gradient(&collinear).is_none());
    assert_eq!(max_abs(&element_forces(&tri, &stretched)), 0.0);
    assert_eq!(tri.strain_energy(&stretched), 0.0);
    assert_eq!(tri.membrane_tension(&stretched), 0.0);
    assert_eq!(tri.principal_stresses(&stretched), (0.0, 0.0));
}

#[test]
fn out_of_plane_lift_gives_cubic_restoring_force() {
    // Lifting node 2 of the flat triangle by w along the normal direction gives the
    // exact Green strain E = diag(0, w^2 / 2), hence S_11 = (lambda_bar / 2 + mu) w^2
    // and an out-of-plane force f_y = -A0 t S_11 w that is exactly cubic in w.
    let rest = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    for &tension_only in &[false, true] {
        let tri = element(&rest, tension_only);
        let lift = |w: f64| vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, w, 1.0]];
        let force_y = |w: f64| element_forces(&tri, &lift(w))[2][1];
        let (lambda_bar, mu) = tri.plane_stress_moduli();
        let volume = tri.ref_area * tri.thickness;
        for &w in &[0.01_f64, 0.02, 0.1] {
            let expected = -volume * (0.5 * lambda_bar + mu) * w.powi(3);
            let actual = force_y(w);
            assert!(
                (actual - expected).abs() < 1e-9 * expected.abs(),
                "w = {w}: {actual} vs {expected}"
            );
        }
        assert!((force_y(0.02) / force_y(0.01) - 8.0).abs() < 1e-9);
    }
}

// ---------------------------------------------------------------------------
// Membrane mesh (F4194)
// ---------------------------------------------------------------------------

fn pinned_square(nodes_per_side: usize, node_mass: f64) -> MembraneMesh {
    let mut mesh = MembraneMesh::create_rectangular(
        nodes_per_side,
        nodes_per_side,
        1.0,
        1.0,
        node_mass,
        1.0e-3,
        1.0e6,
        0.3,
    );
    mesh.fix_boundary(nodes_per_side, nodes_per_side);
    mesh.gravity = [0.0; 3];
    mesh
}

#[test]
fn membrane_forces_pull_a_displaced_node_back() {
    let mut mesh = pinned_square(3, 1.0);
    // Displace the centre node in plane.
    mesh.nodes[4].position[0] += 0.1;
    mesh.clear_forces();
    mesh.apply_membrane_forces();
    assert!(
        mesh.nodes[4].force[0] < -1.0,
        "restoring force expected, got {:?}",
        mesh.nodes[4].force
    );
    for (idx, node) in mesh.nodes.iter().enumerate() {
        if node.is_fixed() {
            assert_eq!(
                node.force, [0.0; 3],
                "fixed node {idx} must receive no force"
            );
        }
    }
}

#[test]
fn unloaded_flat_membrane_stays_flat() {
    let mut mesh = pinned_square(6, 0.05);
    mesh.simulate(1.0e-3, 200);
    for node in &mesh.nodes {
        assert!(node.position[1].abs() < 1e-12, "{:?}", node.position);
    }
}

#[test]
fn step_applies_membrane_elasticity() {
    // Same mesh, same push: with the elastic forces the stretched node returns,
    // without them (stiffness 0) it would drift away.
    let mut stiff = pinned_square(5, 0.05);
    let mut limp = pinned_square(5, 0.05);
    for triangle in &mut limp.triangles {
        triangle.youngs_modulus = 0.0;
    }
    for mesh in [&mut stiff, &mut limp] {
        mesh.nodes[12].position[0] += 0.05;
        mesh.nodes[12].prev_position[0] += 0.05;
    }
    stiff.simulate(1.0e-3, 5);
    limp.simulate(1.0e-3, 5);
    let stiff_shift = stiff.nodes[12].position[0] - 0.5;
    let limp_shift = limp.nodes[12].position[0] - 0.5;
    assert!(
        stiff_shift < limp_shift - 1e-6,
        "elasticity must pull the node back: {stiff_shift} vs {limp_shift}"
    );
}

#[test]
fn pressurised_pinned_membrane_settles_to_bounded_equilibrium() {
    let n = 11;
    let mut mesh = pinned_square(n, 0.05);
    mesh.pressure = 100.0;
    let dt = 1.0e-3;
    let deflection = |m: &MembraneMesh| {
        m.nodes
            .iter()
            .map(|node| node.position[1].abs())
            .fold(0.0_f64, f64::max)
    };
    mesh.simulate(dt, 19_000);
    let before = deflection(&mesh);
    mesh.simulate(dt, 1_000);
    let after = deflection(&mesh);
    assert!(after.is_finite());
    // Bounded bulge towards -y (the winding normal of the mesh). Hencky's circular
    // membrane of radius 0.5 m would deflect by 0.6 a (p a / E t)^(1/3) = 0.11 m; the
    // 1 m square is slightly more compliant.
    assert!(after > 0.08 && after < 0.2, "max deflection {after}");
    assert!(
        (after - before).abs() < 1e-6,
        "deflection still changing: {before} -> {after}"
    );
    let centre = &mesh.nodes[(n / 2) * n + n / 2];
    assert!(centre.position[1] < 0.0 && (centre.position[1].abs() - after).abs() < 1e-9);
    // At rest: negligible velocity and a vanishing force residual on every free node.
    let max_speed = mesh
        .nodes
        .iter()
        .map(|node| vec_len(node.velocity))
        .fold(0.0_f64, f64::max);
    assert!(max_speed < 1e-8, "residual speed {max_speed}");
    mesh.clear_forces();
    mesh.apply_pressure();
    mesh.apply_membrane_forces();
    let node_load = 100.0 * (1.0 / (n as f64 - 1.0)).powi(2);
    let residual = mesh
        .nodes
        .iter()
        .filter(|node| !node.is_fixed())
        .map(|node| vec_len(node.force))
        .fold(0.0_f64, f64::max);
    assert!(
        residual < 1e-6 * node_load,
        "force residual {residual} vs nodal load {node_load}"
    );
}
