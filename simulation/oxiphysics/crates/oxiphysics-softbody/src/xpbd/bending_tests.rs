// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for [`XpbdBendingConstraint`]: signed interior dihedral
//! angle, exact four-particle gradient and momentum-conserving projection.
//!
//! The old constraint measured the unsigned `acos` of the face-normal angle
//! (a cusp at the flat state) and moved only the two wing particles along a
//! fixed-sign direction, so one of the two fold directions of a flat hinge was
//! deepened instead of restored and the correction injected momentum.

use super::{XpbdBendingConstraint, XpbdBody};
use oxiphysics_core::math::{Vec3, quat_from_euler};
use std::f64::consts::{PI, TAU};

/// XPBD sub-step length used by the tests.
const DT: f64 = 1.0 / 60.0;

/// A hinge along the x axis: `p0` at the origin, `p1 = (1, 0, 0)`, the wing
/// `p2` at height 0.8 above the edge in the `+z` direction and the wing `p3` at
/// height 0.9, folded by `fold` radians from the flat position (`-z`) about the
/// edge.  The wings sit at different positions along the edge, so the hinge is
/// not symmetric.
///
/// The normal `(p1 - p0) x (p2 - p0)` of the first triangle points to `-y`, so
/// a positive `fold` folds the wings towards that side and the interior dihedral
/// angle is `pi - fold`; a negative `fold` gives `pi + |fold|`.
fn hinge(fold: f64) -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.3, 0.0, 0.8),
        Vec3::new(0.65, -0.9 * fold.sin(), -0.9 * fold.cos()),
    ]
}

/// An asymmetric, non-planar hinge with no special alignment to any axis.
fn generic_hinge() -> Vec<Vec3> {
    vec![
        Vec3::new(0.10, -0.20, 0.05),
        Vec3::new(1.15, 0.30, -0.10),
        Vec3::new(0.40, 0.25, 0.95),
        Vec3::new(0.75, -0.85, -0.55),
    ]
}

/// Mirror image across the plane `y = 0`; it maps the interior angle `theta`
/// to `2 pi - theta`.
fn mirror(v: &Vec3) -> Vec3 {
    Vec3::new(v.x, -v.y, v.z)
}

fn mirrored(positions: &[Vec3]) -> Vec<Vec3> {
    positions.iter().map(mirror).collect()
}

fn constraint_for(positions: &[Vec3], compliance: f64) -> XpbdBendingConstraint {
    XpbdBendingConstraint::new(0, 1, 2, 3, positions, compliance)
}

/// `sum_i m_i (after_i - before_i)`.
fn linear_impulse(before: &[Vec3], after: &[Vec3], masses: &[f64]) -> Vec3 {
    before
        .iter()
        .zip(after)
        .zip(masses)
        .fold(Vec3::zeros(), |sum, ((b, a), m)| sum + (a - b) * *m)
}

/// `sum_i before_i x m_i (after_i - before_i)`.
fn angular_impulse(before: &[Vec3], after: &[Vec3], masses: &[f64]) -> Vec3 {
    before
        .iter()
        .zip(after)
        .zip(masses)
        .fold(Vec3::zeros(), |sum, ((b, a), m)| {
            sum + b.cross(&((a - b) * *m))
        })
}

#[test]
fn flat_hinge_has_the_interior_angle_pi_and_no_violation() {
    let positions = hinge(0.0);
    let constraint = constraint_for(&positions, 0.0);
    assert!((constraint.rest_angle - PI).abs() < 1e-14);
    assert!((constraint.current_angle(&positions) - PI).abs() < 1e-14);
    assert!(constraint.evaluate(&positions).abs() < 1e-14);
}

#[test]
fn interior_angle_is_pi_minus_the_fold_towards_the_first_normal() {
    let constraint = constraint_for(&hinge(0.0), 0.0);
    for degrees in [10.0_f64, 45.0, 90.0, 135.0, 170.0] {
        let fold = degrees.to_radians();
        let towards = constraint.current_angle(&hinge(fold));
        let away = constraint.current_angle(&hinge(-fold));
        assert!(
            (towards - (PI - fold)).abs() < 1e-12,
            "{degrees} deg towards: {towards}"
        );
        assert!(
            (away - (PI + fold)).abs() < 1e-12,
            "{degrees} deg away: {away}"
        );
        assert!((towards + away - TAU).abs() < 1e-12);
    }
}

#[test]
fn angle_and_gradients_are_rigid_motion_covariant() {
    let positions = generic_hinge();
    let constraint = constraint_for(&positions, 0.0);
    let rotation = quat_from_euler(0.7, -0.4, 1.9);
    let shift = Vec3::new(3.0, -2.0, 5.0);
    let moved: Vec<Vec3> = positions.iter().map(|p| rotation * *p + shift).collect();
    assert!(
        (constraint.current_angle(&moved) - constraint.current_angle(&positions)).abs() < 1e-12
    );
    let reference = constraint
        .angle_gradients(&positions)
        .expect("the generic hinge is not degenerate");
    let rotated = constraint
        .angle_gradients(&moved)
        .expect("the moved hinge is not degenerate");
    for (before, after) in reference.iter().zip(&rotated) {
        assert!((rotation * *before - after).norm() < 1e-12);
    }
}

#[test]
fn angle_gradients_match_central_differences() {
    for positions in [generic_hinge(), mirrored(&generic_hinge())] {
        let constraint = constraint_for(&positions, 0.0);
        let analytic = constraint
            .angle_gradients(&positions)
            .expect("the generic hinge is not degenerate");
        let step = 1e-6;
        for (particle, gradient) in analytic.iter().enumerate() {
            for (axis, analytic_component) in gradient.iter().enumerate() {
                let mut plus = positions.clone();
                let mut minus = positions.clone();
                plus[particle][axis] += step;
                minus[particle][axis] -= step;
                let numeric = (constraint.current_angle(&plus) - constraint.current_angle(&minus))
                    / (2.0 * step);
                assert!(
                    (numeric - analytic_component).abs() < 1e-7,
                    "particle {particle} axis {axis}: analytic {analytic_component}, numeric {numeric}"
                );
            }
        }
    }
}

#[test]
fn angle_gradients_sum_to_zero_and_carry_no_torque() {
    for positions in [generic_hinge(), mirrored(&generic_hinge()), hinge(0.4)] {
        let constraint = constraint_for(&positions, 0.0);
        let gradients = constraint
            .angle_gradients(&positions)
            .expect("the hinge is not degenerate");
        let force = gradients.iter().fold(Vec3::zeros(), |sum, g| sum + *g);
        let torque = positions
            .iter()
            .zip(&gradients)
            .fold(Vec3::zeros(), |sum, (x, g)| sum + x.cross(g));
        assert!(force.norm() < 1e-12, "net gradient {force:?}");
        assert!(torque.norm() < 1e-12, "net torque {torque:?}");
    }
}

#[test]
fn a_flat_hinge_bent_either_way_is_restored_to_flat() {
    let inv_masses = [1.0; 4];
    let mut restored = Vec::new();
    for fold in [10.0_f64.to_radians(), -10.0_f64.to_radians()] {
        let mut constraint = constraint_for(&hinge(0.0), 0.0);
        let mut positions = hinge(fold);
        let before = constraint.evaluate(&positions);
        assert!(
            (before.abs() - fold.abs()).abs() < 1e-12,
            "violation {before} for a {fold} rad fold"
        );
        constraint.solve(&mut positions, &inv_masses, DT);
        let after = constraint.evaluate(&positions);
        assert!(
            after.abs() < 0.05 * before.abs(),
            "one projection must remove most of the violation: {before} -> {after}"
        );
        for _ in 0..20 {
            constraint.reset_lambda();
            constraint.solve(&mut positions, &inv_masses, DT);
        }
        assert!(constraint.evaluate(&positions).abs() < 1e-12);
        assert!((constraint.current_angle(&positions) - PI).abs() < 1e-12);
        restored.push(positions);
    }
    // Neither fold direction is preferred: the two results are mirror images.
    for (positive, negative) in restored[0].iter().zip(&restored[1]) {
        assert!((mirror(positive) - negative).norm() < 1e-12);
    }
}

#[test]
fn the_restoring_step_points_the_right_way_on_both_sides_of_flat() {
    let inv_masses = [1.0; 4];
    for fold in [0.3_f64, -0.3] {
        let mut constraint = constraint_for(&hinge(0.0), 0.0);
        let mut positions = hinge(fold);
        let angle_before = constraint.current_angle(&positions);
        constraint.solve(&mut positions, &inv_masses, DT);
        let angle_after = constraint.current_angle(&positions);
        // theta = pi - fold: folding towards the first normal lowers it, so the
        // restoring step must raise it, and vice versa.
        if fold > 0.0 {
            assert!(
                angle_after > angle_before,
                "{angle_before} -> {angle_after}"
            );
        } else {
            assert!(
                angle_after < angle_before,
                "{angle_before} -> {angle_after}"
            );
        }
        assert!((angle_after - PI).abs() < (angle_before - PI).abs());
    }
}

#[test]
fn a_folded_rest_angle_is_restored_from_both_sides() {
    let rest_fold = 60.0_f64.to_radians();
    let rest_positions = hinge(rest_fold);
    let inv_masses = [1.0; 4];
    for offset in [15.0_f64.to_radians(), -15.0_f64.to_radians()] {
        let mut constraint = constraint_for(&rest_positions, 0.0);
        assert!((constraint.rest_angle - (PI - rest_fold)).abs() < 1e-12);
        let mut positions = hinge(rest_fold + offset);
        let before = constraint.evaluate(&positions);
        // More fold lowers theta, so the violation is minus the extra fold.
        assert!((before + offset).abs() < 1e-12, "violation {before}");
        constraint.solve(&mut positions, &inv_masses, DT);
        assert!(constraint.evaluate(&positions).abs() < 0.05 * before.abs());
        for _ in 0..20 {
            constraint.reset_lambda();
            constraint.solve(&mut positions, &inv_masses, DT);
        }
        assert!(constraint.evaluate(&positions).abs() < 1e-12);
        assert!((constraint.current_angle(&positions) - constraint.rest_angle).abs() < 1e-12);
    }
}

#[test]
fn evaluate_and_solve_take_the_shortest_way_round() {
    // theta = 2 pi - 0.1: a hinge folded almost completely onto the other
    // side.  A rest angle of 0.1 is only 0.2 rad away through 2 pi == 0.
    let positions = hinge(-(PI - 0.1));
    let mut constraint = constraint_for(&hinge(0.0), 0.0);
    constraint.rest_angle = 0.1;
    let theta = constraint.current_angle(&positions);
    assert!((theta - (TAU - 0.1)).abs() < 1e-12, "theta {theta}");
    let violation = constraint.evaluate(&positions);
    assert!(
        (violation + 0.2).abs() < 1e-12,
        "the wrapped violation is -0.2 rad, got {violation}"
    );
    let mut moved = positions.clone();
    constraint.solve(&mut moved, &[1.0; 4], DT);
    // The projection carries the angle up through 2 pi == 0 to 0.1; going the
    // long way round would leave a violation of about 6 rad.
    let residual = constraint.evaluate(&moved);
    assert!(residual.abs() < 0.02, "residual {residual}");
}

#[test]
fn projection_conserves_linear_and_angular_momentum_with_unequal_masses() {
    let masses = [1.0, 2.0, 0.5, 3.0];
    let inv_masses = masses.map(|mass| 1.0 / mass);
    for before in [generic_hinge(), mirrored(&generic_hinge())] {
        let mut constraint = constraint_for(&before, 1e-3);
        constraint.rest_angle = PI;
        let mut after = before.clone();
        constraint.solve(&mut after, &inv_masses, DT);
        let moved = before
            .iter()
            .zip(&after)
            .map(|(b, a)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(moved > 1e-3, "the projection must act (moved {moved})");
        let linear = linear_impulse(&before, &after, &masses);
        let angular = angular_impulse(&before, &after, &masses);
        assert!(linear.norm() < 1e-13, "linear impulse {linear:?}");
        assert!(angular.norm() < 1e-13, "angular impulse {angular:?}");
    }
}

#[test]
fn pinned_shared_edge_leaves_only_the_wings_moving() {
    let inv_masses = [0.0, 0.0, 1.0, 2.0];
    for fold in [0.3_f64, -0.3] {
        let mut constraint = constraint_for(&hinge(0.0), 0.0);
        let start = hinge(fold);
        let mut positions = start.clone();
        for _ in 0..30 {
            constraint.reset_lambda();
            constraint.solve(&mut positions, &inv_masses, DT);
        }
        assert_eq!(positions[0], start[0]);
        assert_eq!(positions[1], start[1]);
        assert_ne!(positions[2], start[2]);
        assert_ne!(positions[3], start[3]);
        assert!(constraint.evaluate(&positions).abs() < 1e-10);
    }
}

#[test]
fn pinned_wings_let_the_shared_edge_do_the_work() {
    // Both wings are fixed: the old constraint, which never moved the shared
    // edge, could do nothing here.
    let inv_masses = [1.0, 2.0, 0.0, 0.0];
    for fold in [0.2_f64, -0.2] {
        let mut constraint = constraint_for(&hinge(0.0), 0.0);
        let start = hinge(fold);
        let mut positions = start.clone();
        for _ in 0..100 {
            constraint.reset_lambda();
            constraint.solve(&mut positions, &inv_masses, DT);
        }
        assert_eq!(positions[2], start[2]);
        assert_eq!(positions[3], start[3]);
        assert_ne!(positions[0], start[0]);
        assert_ne!(positions[1], start[1]);
        assert!(
            constraint.evaluate(&positions).abs() < 1e-9,
            "residual {}",
            constraint.evaluate(&positions)
        );
    }
}

#[test]
fn a_hinge_of_pinned_particles_does_not_move() {
    let mut constraint = constraint_for(&hinge(0.0), 0.0);
    let start = hinge(0.4);
    let mut positions = start.clone();
    constraint.solve(&mut positions, &[0.0; 4], DT);
    assert_eq!(positions, start);
    assert_eq!(constraint.lambda, 0.0);
}

#[test]
fn compliance_follows_the_xpbd_update_rule() {
    let compliance = 1e-4;
    let inv_masses = [1.0, 2.0, 0.5, 1.5];
    let alpha_tilde = compliance / (DT * DT);
    let mut constraint = constraint_for(&hinge(0.0), compliance);
    let mut positions = hinge(0.3);
    let mut expected_lambda = 0.0;
    for _ in 0..3 {
        let violation = constraint.evaluate(&positions);
        let gradients = constraint
            .angle_gradients(&positions)
            .expect("the hinge is not degenerate");
        let weighted: f64 = gradients
            .iter()
            .zip(&inv_masses)
            .map(|(g, w)| w * g.norm_squared())
            .sum();
        expected_lambda += (-violation - alpha_tilde * expected_lambda) / (weighted + alpha_tilde);
        constraint.solve(&mut positions, &inv_masses, DT);
        assert!(
            (constraint.lambda - expected_lambda).abs() < 1e-12 * expected_lambda.abs().max(1.0),
            "lambda {} expected {expected_lambda}",
            constraint.lambda
        );
    }
    constraint.reset_lambda();
    assert_eq!(constraint.lambda, 0.0);
    // A softer constraint leaves a larger residual than a rigid one.
    let rigid_residual = {
        let mut rigid = constraint_for(&hinge(0.0), 0.0);
        let mut moved = hinge(0.3);
        rigid.solve(&mut moved, &inv_masses, DT);
        rigid.evaluate(&moved).abs()
    };
    let soft_residual = {
        let mut soft = constraint_for(&hinge(0.0), compliance);
        let mut moved = hinge(0.3);
        soft.solve(&mut moved, &inv_masses, DT);
        soft.evaluate(&moved).abs()
    };
    assert!(soft_residual > rigid_residual);
}

#[test]
fn degenerate_hinges_have_no_angle_and_are_left_alone() {
    let zero_edge = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.3, 0.0, 0.8),
        Vec3::new(0.3, 0.0, -0.8),
    ];
    let mut wing_on_the_hinge_line = hinge(0.3);
    wing_on_the_hinge_line[2] = Vec3::new(0.4, 0.0, 0.0);
    let mut not_finite = hinge(0.3);
    not_finite[3].x = f64::NAN;
    for start in [zero_edge, wing_on_the_hinge_line, not_finite] {
        let mut constraint = constraint_for(&start, 0.0);
        assert_eq!(constraint.rest_angle, PI, "a degenerate rest hinge is flat");
        constraint.rest_angle = 1.0;
        assert_eq!(constraint.current_angle(&start), 1.0);
        assert_eq!(constraint.evaluate(&start), 0.0);
        assert!(constraint.angle_gradients(&start).is_none());
        let mut positions = start.clone();
        constraint.solve(&mut positions, &[1.0; 4], DT);
        for (after, before) in positions.iter().zip(&start) {
            assert!(
                after
                    .iter()
                    .zip(before.iter())
                    .all(|(a, b)| a == b || (a.is_nan() && b.is_nan())),
                "a degenerate hinge must not move"
            );
        }
        assert_eq!(constraint.lambda, 0.0);
    }
}

#[test]
fn free_hinge_keeps_its_centre_of_mass_and_momentum_in_a_full_step() {
    let masses = [1.0, 2.0, 0.5, 3.0];
    let mut body = XpbdBody::new(4);
    for (position, mass) in hinge(0.4).into_iter().zip(masses) {
        body.add_particle(position, mass);
    }
    body.add_bending_constraint(0, 1, 2, 3, 0.0);
    body.bending_constraints[0].rest_angle = PI;
    let violation_before = body.bending_constraints[0].evaluate(&body.positions);
    let centre_before = body.centre_of_mass();
    body.step(DT, Vec3::zeros(), 4, 4);
    let centre_after = body.centre_of_mass();
    assert!(
        (centre_after - centre_before).norm() < 1e-12,
        "the centre of mass drifted by {:?}",
        centre_after - centre_before
    );
    let momentum = body
        .velocities
        .iter()
        .zip(&masses)
        .fold(Vec3::zeros(), |sum, (v, m)| sum + *v * *m);
    assert!(momentum.norm() < 1e-9, "net momentum {momentum:?}");
    let violation_after = body.bending_constraints[0].evaluate(&body.positions);
    assert!(
        violation_after.abs() < 1e-3 * violation_before.abs(),
        "the hinge must be flattened: {violation_before} -> {violation_after}"
    );
}
