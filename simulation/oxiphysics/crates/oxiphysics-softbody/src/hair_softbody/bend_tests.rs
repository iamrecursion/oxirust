// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the bend force of [`DiscreteElasticRod`].
//!
//! The bend force used to be applied along the curvature *binormal*, the unit
//! normal of the plane in which the rod bends. The gradient of the turning
//! angle lies in that plane, so the old force was orthogonal to it: it was
//! zero for a straight rod, could not straighten a bent one and pushed a
//! planar rod out of its plane. The force is now the exact negative gradient
//! of [`DiscreteElasticRod::bend_energy`].

use super::*;

const FD_STEP: f64 = 1.0e-6;

/// Rod with three segments whose two interior vertices are clearly bent (0.55
/// and 0.79 rad), out of any common plane, with non-zero rest angles.
fn bent_rod() -> DiscreteElasticRod {
    let mut rod = DiscreteElasticRod::new_straight(
        [0.0, 0.0, 0.0],
        [0.0, -0.9, 0.0],
        3,
        0.01,
        100.0,
        0.5,
        0.01,
        0.1,
    );
    rod.positions = vec![
        [0.0, 0.0, 0.0],
        [0.05, -0.3, 0.02],
        [0.22, -0.55, -0.06],
        [0.30, -0.93, 0.12],
    ];
    rod.rest_curvatures = vec![0.15, 0.35];
    rod
}

/// Planar V-shaped rod in the x-y plane: two segments meeting at 90 degrees.
fn planar_v_rod() -> DiscreteElasticRod {
    let mut rod = DiscreteElasticRod::new_straight(
        [0.0, 0.0, 0.0],
        [0.6, 0.0, 0.0],
        2,
        0.01,
        100.0,
        0.5,
        0.01,
        0.1,
    );
    rod.positions = vec![[0.0, 0.0, 0.0], [0.3, -0.3, 0.0], [0.6, 0.0, 0.0]];
    rod
}

fn bend_forces(rod: &DiscreteElasticRod) -> Vec<[f64; 3]> {
    let mut forces = vec![zero3(); rod.num_vertices()];
    rod.accumulate_bend_forces(&mut forces);
    forces
}

#[test]
fn bend_forces_are_the_negative_gradient_of_the_bend_energy() {
    let rod = bent_rod();
    assert!(rod.curvature(1) > 0.5 && rod.curvature(2) > 0.7);
    let forces = bend_forces(&rod);
    let scale = forces
        .iter()
        .flat_map(|f| f.iter())
        .fold(0.0_f64, |acc, c| acc.max(c.abs()));
    assert!(scale > 1.0e-2, "the bent rod must be loaded, got {scale}");
    for (vertex, force) in forces.iter().enumerate() {
        for (axis, analytic) in force.iter().enumerate() {
            let mut plus = rod.clone();
            plus.positions[vertex][axis] += FD_STEP;
            let mut minus = rod.clone();
            minus.positions[vertex][axis] -= FD_STEP;
            let gradient = (plus.bend_energy() - minus.bend_energy()) / (2.0 * FD_STEP);
            assert!(
                (analytic + gradient).abs() <= 1.0e-5 * scale,
                "vertex {vertex} axis {axis}: analytic {analytic}, finite-difference {}",
                -gradient
            );
        }
    }
}

#[test]
fn straight_rod_feels_no_bend_force() {
    let rod = DiscreteElasticRod::new_straight(
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        5,
        0.01,
        100.0,
        0.5,
        0.01,
        0.1,
    );
    for force in bend_forces(&rod) {
        assert_eq!(force, [0.0; 3]);
    }
}

#[test]
fn bend_force_acts_in_the_bending_plane_and_flattens_the_bend() {
    let rod = planar_v_rod();
    let forces = bend_forces(&rod);
    // The old force was along the binormal (z) and nothing else.
    for (vertex, force) in forces.iter().enumerate() {
        assert!(
            force[2].abs() < 1.0e-12,
            "vertex {vertex}: force {force:?} leaves the bending plane"
        );
    }
    // The apex is pushed up towards the chord, the free ends down and outwards.
    assert!(forces[1][1] > 0.0, "apex force {:?}", forces[1]);
    assert!(forces[0][1] < 0.0 && forces[0][0] < 0.0, "{:?}", forces[0]);
    assert!(forces[2][1] < 0.0 && forces[2][0] > 0.0, "{:?}", forces[2]);
    // Equal and opposite: no net force.
    let net = forces.iter().fold(zero3(), |acc, f| add3(acc, *f));
    assert!(len3(net) < 1.0e-12, "net force {net:?}");
    // The force of the apex is k_b * theta / l_bar times the gradient: with a
    // 90 degree bend, l_bar = 0.3 and |e| = 0.3 * sqrt(2) the closed form is
    // 2 * k_b * (pi/2) / l_bar * (1 / |e|) * cos(45 deg) along +y.
    let length = 0.3 * 2.0_f64.sqrt();
    let expected = 2.0 * 0.5 * (PI / 2.0) / 0.3 / length * (PI / 4.0).cos();
    assert!(
        (forces[1][1] - expected).abs() < 1.0e-9 * expected,
        "apex force {} should be {expected}",
        forces[1][1]
    );
    // A small step along the force lowers the bend energy.
    let mut relaxed = rod.clone();
    for (position, force) in relaxed.positions.iter_mut().zip(forces.iter()) {
        *position = add3(*position, scale3(*force, 1.0e-3));
    }
    assert!(relaxed.bend_energy() < rod.bend_energy());
}

#[test]
fn bend_forces_conserve_linear_and_angular_momentum() {
    let rod = bent_rod();
    let forces = bend_forces(&rod);
    let net = forces.iter().fold(zero3(), |acc, f| add3(acc, *f));
    let torque = rod
        .positions
        .iter()
        .zip(forces.iter())
        .fold(zero3(), |acc, (x, f)| add3(acc, cross3(*x, *f)));
    assert!(len3(net) < 1.0e-12, "net force {net:?}");
    assert!(len3(torque) < 1.0e-12, "net torque {torque:?}");
}

#[test]
fn rest_curvature_sets_the_preferred_angle() {
    // At the rest angle the bend force vanishes; below it the rod is pushed
    // towards a larger angle and above it towards a smaller one.
    let mut rod = planar_v_rod();
    let theta = rod.curvature(1);
    rod.rest_curvatures = vec![theta];
    for force in bend_forces(&rod) {
        assert!(len3(force) < 1.0e-12, "{force:?}");
    }
    rod.rest_curvatures = vec![theta + 0.3];
    let towards_larger = bend_forces(&rod);
    rod.rest_curvatures = vec![theta - 0.3];
    let towards_smaller = bend_forces(&rod);
    assert!(towards_larger[1][1] < 0.0, "{:?}", towards_larger[1]);
    assert!(towards_smaller[1][1] > 0.0, "{:?}", towards_smaller[1]);
}

#[test]
fn planar_rod_straightens_without_leaving_its_plane() {
    let mut rod = planar_v_rod();
    rod.damping = 5.0;
    let before = rod.bend_energy();
    for _ in 0..4000 {
        rod.step(5.0e-4, [0.0; 3]);
    }
    let after = rod.bend_energy();
    assert!(
        after < 0.05 * before,
        "the rod should relax towards straight: {before} -> {after}"
    );
    for position in &rod.positions {
        assert!(position[2].abs() < 1.0e-12, "left the plane: {position:?}");
    }
    assert!(rod.curvature(1) < 0.5 * PI / 2.0);
}

#[test]
fn degenerate_vertices_are_skipped() {
    // Folded back on itself (turning angle pi) and coincident vertices.
    let mut rod = planar_v_rod();
    rod.positions = vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]];
    for force in bend_forces(&rod) {
        assert!(force.iter().all(|c| c.is_finite()));
    }
    rod.positions = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.1, 0.0]];
    for force in bend_forces(&rod) {
        assert!(force.iter().all(|c| c.is_finite()));
    }
}
