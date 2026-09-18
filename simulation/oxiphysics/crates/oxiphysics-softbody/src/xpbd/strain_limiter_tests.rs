// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the strain limiter that [`XpbdBody::step`] and
//! [`XpbdBody::step_with_stats`] run once per substep.

use super::{StrainLimitConfig, XpbdBody, XpbdStrainLimiter};
use crate::Error;
use oxiphysics_core::math::Vec3;
use std::collections::BTreeSet;

/// Frame time (s) of the tests.
const FRAME: f64 = 1.0 / 60.0;

/// Tolerance for results that are exact up to rounding.
const ROUNDING: f64 = 1e-12;

fn limits(max_stretch: f64, min_stretch: f64, iterations: u32) -> StrainLimitConfig {
    StrainLimitConfig {
        max_stretch,
        min_stretch,
        iterations,
    }
}

/// Right triangle in the XZ plane with unit legs: particle 0 at the origin,
/// 1 at `(1, 0, 0)` and 2 at `(0, 0, 1)`; unit masses, except that the origin
/// is pinned when `pin_origin` is set.
fn unit_triangle(pin_origin: bool) -> XpbdBody {
    let mut body = XpbdBody::new(3);
    body.add_particle(Vec3::zeros(), if pin_origin { 0.0 } else { 1.0 });
    body.add_particle(Vec3::new(1.0, 0.0, 0.0), 1.0);
    body.add_particle(Vec3::new(0.0, 0.0, 1.0), 1.0);
    body
}

/// Length of the edge from particle `a` to particle `b`.
fn edge_length(body: &XpbdBody, a: usize, b: usize) -> f64 {
    (body.positions[b] - body.positions[a]).norm()
}

/// The limiter of a body on which a test has just enabled one.
fn limiter_of(body: &XpbdBody) -> &XpbdStrainLimiter {
    body.strain_limiter
        .as_ref()
        .expect("the strain limiter is enabled")
}

/// A 3 x 3 patch of cloth in the XZ plane (spacing 0.2 m, unit masses) whose
/// first row is pinned, with a distance constraint of the given compliance on
/// every edge of its triangulation.  Returns the body and the 8 triangles.
fn cloth_patch(compliance: f64) -> (XpbdBody, Vec<[usize; 3]>) {
    const SIDE: usize = 3;
    let mut body = XpbdBody::new(SIDE * SIDE);
    for row in 0..SIDE {
        for column in 0..SIDE {
            let mass = if row == 0 { 0.0 } else { 1.0 };
            body.add_particle(Vec3::new(column as f64 * 0.2, 0.0, row as f64 * 0.2), mass);
        }
    }
    let mut triangles = Vec::new();
    for row in 0..SIDE - 1 {
        for column in 0..SIDE - 1 {
            let v00 = row * SIDE + column;
            let v10 = v00 + 1;
            let v01 = v00 + SIDE;
            let v11 = v01 + 1;
            triangles.push([v00, v10, v01]);
            triangles.push([v10, v11, v01]);
        }
    }
    let mut edges = BTreeSet::new();
    for &[a, b, c] in &triangles {
        for (p, q) in [(a, b), (b, c), (c, a)] {
            edges.insert((p.min(q), p.max(q)));
        }
    }
    for (a, b) in edges {
        body.add_distance_constraint(a, b, compliance);
    }
    (body, triangles)
}

#[test]
fn strain_limiting_is_off_by_default() {
    let mut body = unit_triangle(false);
    assert!(body.strain_limiter.is_none());
    body.positions[1] = Vec3::new(1.5, 0.0, 0.0);
    body.step(FRAME, Vec3::zeros(), 2, 2);
    assert_eq!(
        body.positions[1],
        Vec3::new(1.5, 0.0, 0.0),
        "plain XPBD leaves a free stretched triangle alone"
    );
    assert_eq!(body.max_velocity(), 0.0);
}

#[test]
fn step_clamps_a_stretched_triangle_to_max_stretch() {
    let mut body = unit_triangle(false);
    body.enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    body.positions[1] = Vec3::new(1.5, 0.0, 0.0);
    let centre = body.centre_of_mass();
    body.step(FRAME, Vec3::zeros(), 1, 0);
    let stretched_leg = body.positions[1] - body.positions[0];
    let other_leg = body.positions[2] - body.positions[0];
    assert!(
        (stretched_leg - Vec3::new(1.05, 0.0, 0.0)).norm() < ROUNDING,
        "stretched leg {stretched_leg:?}"
    );
    assert!(
        (other_leg - Vec3::new(0.0, 0.0, 1.0)).norm() < ROUNDING,
        "unstrained leg {other_leg:?}"
    );
    assert!(
        (body.centre_of_mass() - centre).norm() < ROUNDING,
        "a free triangle is limited without net momentum"
    );
}

#[test]
fn step_lifts_a_compressed_triangle_to_min_stretch() {
    let mut body = unit_triangle(false);
    body.enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    body.positions[1] = Vec3::new(0.5, 0.0, 0.0);
    body.step(FRAME, Vec3::zeros(), 1, 0);
    let leg = edge_length(&body, 0, 1);
    assert!((leg - 0.95).abs() < ROUNDING, "compressed leg {leg}");
    let other = edge_length(&body, 0, 2);
    assert!((other - 1.0).abs() < ROUNDING, "unstrained leg {other}");
}

#[test]
fn limiter_runs_before_the_velocity_update() {
    let mut body = unit_triangle(true);
    body.enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    body.positions[1] = Vec3::new(1.5, 0.0, 0.0);
    body.step(FRAME, Vec3::zeros(), 1, 1);
    // The pinned origin stays, and the corner that was not strained stays too.
    assert_eq!(body.positions[0], Vec3::zeros());
    assert!((body.positions[1] - Vec3::new(1.05, 0.0, 0.0)).norm() < ROUNDING);
    assert!((body.positions[2] - Vec3::new(0.0, 0.0, 1.0)).norm() < ROUNDING);
    // The stretched corner was pulled back by 0.45 m within the substep, so its
    // velocity carries the correction: (x - x_previous) / h.
    assert_eq!(body.prev_positions[1], Vec3::new(1.5, 0.0, 0.0));
    let expected = (body.positions[1] - body.prev_positions[1]) / FRAME;
    assert!((body.velocities[1] - expected).norm() < 1e-9);
    assert!(
        (body.velocities[1].x + 0.45 / FRAME).abs() < 1e-9,
        "velocity {:?}",
        body.velocities[1]
    );
    assert_eq!(body.velocities[0], Vec3::zeros());
    assert!(body.velocities[2].norm() < 1e-9);
}

#[test]
fn limiter_runs_after_the_constraint_projections() {
    // The distance constraint insists on a leg of 1.5 although the triangle may
    // only stretch to 1.05: whoever runs last decides.
    let build = || {
        let mut body = unit_triangle(true);
        body.add_distance_constraint(0, 1, 0.0);
        body.distance_constraints[0].rest_length = 1.5;
        body
    };
    let mut plain = build();
    plain.step(FRAME, Vec3::zeros(), 1, 8);
    let plain_leg = edge_length(&plain, 0, 1);
    assert!(
        (plain_leg - 1.5).abs() < 1e-9,
        "constraint alone: {plain_leg}"
    );

    let mut limited = build();
    limited
        .enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    limited.step(FRAME, Vec3::zeros(), 1, 8);
    let limited_leg = edge_length(&limited, 0, 1);
    assert!(
        (limited_leg - 1.05).abs() < ROUNDING,
        "the limiter has the last word: {limited_leg}"
    );
    assert_eq!(limited.positions[0], Vec3::zeros(), "the pinned origin");
}

#[test]
fn enabling_with_a_missing_particle_is_rejected_and_changes_nothing() {
    let mut body = unit_triangle(false);
    let error = body
        .enable_strain_limiting(vec![[0, 1, 2], [1, 2, 3]], limits(1.05, 0.95, 1))
        .unwrap_err();
    assert!(
        matches!(
            &error,
            Error::InvalidParameter(message)
                if message.contains("triangle 1") && message.contains("particle 3")
        ),
        "unexpected error: {error}"
    );
    assert!(body.strain_limiter.is_none());

    // A limiter that is already enabled survives a rejected replacement.
    body.enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    assert!(
        body.enable_strain_limiting(vec![[0, 1, 7]], limits(1.2, 0.8, 2))
            .is_err()
    );
    let limiter = limiter_of(&body);
    assert_eq!(limiter.triangles(), &[[0, 1, 2]]);
    assert_eq!(limiter.config().iterations, 1);
}

#[test]
fn limiter_constructor_needs_a_rest_position_per_referenced_particle() {
    let rest = [Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0)];
    let error = XpbdStrainLimiter::new(vec![[0, 1, 2]], &rest, limits(1.05, 0.95, 1)).unwrap_err();
    assert!(matches!(error, Error::InvalidParameter(_)), "{error}");
    // Triangles are not required to use every particle.
    let limiter = XpbdStrainLimiter::new(Vec::new(), &rest, limits(1.05, 0.95, 1)).unwrap();
    assert!(limiter.triangles().is_empty());
    assert_eq!(limiter.rest_positions().len(), 2);
}

#[test]
fn rest_shape_is_the_positions_at_enable_time() {
    let mut body = unit_triangle(false);
    body.positions[2] = Vec3::new(0.0, 0.0, 2.0);
    body.enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    assert_eq!(
        limiter_of(&body).rest_positions(),
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    );
    // The body starts out in its rest shape, so there is nothing to limit.
    let before = body.positions.clone();
    body.step(FRAME, Vec3::zeros(), 3, 2);
    assert_eq!(body.positions, before);
}

#[test]
fn explicit_rest_shape_is_limited_against() {
    // Twice the size of the rest triangle: both legs are stretched to 2.
    let mut body = XpbdBody::new(3);
    body.add_particle(Vec3::zeros(), 1.0);
    body.add_particle(Vec3::new(2.0, 0.0, 0.0), 1.0);
    body.add_particle(Vec3::new(0.0, 0.0, 2.0), 1.0);
    let rest = [
        Vec3::zeros(),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    body.strain_limiter =
        Some(XpbdStrainLimiter::new(vec![[0, 1, 2]], &rest, limits(1.05, 0.95, 1)).unwrap());
    body.step(FRAME, Vec3::zeros(), 1, 0);
    for leg in [edge_length(&body, 0, 1), edge_length(&body, 0, 2)] {
        assert!((leg - 1.05).abs() < ROUNDING, "leg {leg}");
    }
}

#[test]
fn apply_does_nothing_without_enough_particles() {
    let body = unit_triangle(false);
    let mut limiter =
        XpbdStrainLimiter::new(vec![[0, 1, 2]], &body.positions, limits(1.05, 0.95, 1)).unwrap();

    let mut short = vec![Vec3::zeros(), Vec3::new(1.5, 0.0, 0.0)];
    let before = short.clone();
    assert_eq!(limiter.apply(&mut short, &[1.0, 1.0, 1.0]), 0);
    assert_eq!(short, before, "too few positions");

    let mut stretched = vec![
        Vec3::zeros(),
        Vec3::new(1.5, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let before = stretched.clone();
    assert_eq!(limiter.apply(&mut stretched, &[1.0, 1.0]), 0);
    assert_eq!(stretched, before, "too few inverse masses");

    assert_eq!(limiter.apply(&mut stretched, &[1.0; 3]), 1);
    let leg = (stretched[1] - stretched[0]).norm();
    assert!(
        (leg - 1.05).abs() < ROUNDING,
        "all particles present: {leg}"
    );
}

#[test]
fn apply_counts_triangle_projections() {
    let (body, triangles) = cloth_patch(0.0);
    assert_eq!(triangles.len(), 8);
    let mut positions = body.positions.clone();
    let masses = body.inv_masses.clone();

    let mut three_sweeps =
        XpbdStrainLimiter::new(triangles.clone(), &positions, limits(1.05, 0.95, 3)).unwrap();
    assert_eq!(three_sweeps.apply(&mut positions, &masses), 24);

    let mut no_sweeps =
        XpbdStrainLimiter::new(triangles, &positions, limits(1.05, 0.95, 0)).unwrap();
    positions[8] = Vec3::new(0.9, 0.0, 0.9);
    let before = positions.clone();
    assert_eq!(no_sweeps.apply(&mut positions, &masses), 0);
    assert_eq!(positions, before, "no sweeps, no movement");

    let mut no_triangles =
        XpbdStrainLimiter::new(Vec::new(), &positions, limits(1.05, 0.95, 3)).unwrap();
    assert_eq!(no_triangles.apply(&mut positions, &masses), 0);
    assert_eq!(positions, before, "no triangles, no movement");
}

#[test]
fn set_config_replaces_the_limits() {
    let mut body = unit_triangle(false);
    body.enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    if let Some(limiter) = body.strain_limiter.as_mut() {
        limiter.set_config(limits(1.2, 0.8, 1));
    }
    assert!((limiter_of(&body).config().max_stretch - 1.2).abs() < ROUNDING);
    body.positions[1] = Vec3::new(1.5, 0.0, 0.0);
    body.step(FRAME, Vec3::zeros(), 1, 0);
    let leg = edge_length(&body, 0, 1);
    assert!((leg - 1.2).abs() < ROUNDING, "leg {leg}");
}

#[test]
fn particles_added_after_enabling_are_stepped_but_not_limited() {
    let mut body = unit_triangle(false);
    body.enable_strain_limiting(vec![[0, 1, 2]], limits(1.05, 0.95, 1))
        .unwrap();
    let extra = body.add_particle(Vec3::new(5.0, 0.0, 0.0), 1.0);
    body.positions[1] = Vec3::new(1.5, 0.0, 0.0);
    body.step(FRAME, Vec3::new(0.0, -9.81, 0.0), 1, 1);
    assert!(body.positions[extra].y < 0.0, "the new particle falls");
    let leg = edge_length(&body, 0, 1);
    assert!(
        (leg - 1.05).abs() < ROUNDING,
        "the triangle is still limited: {leg}"
    );
}

#[test]
fn step_with_stats_runs_the_same_limiter_pass_as_step() {
    let gravity = Vec3::new(0.0, -980.0, 0.0);
    let sweeps = 3;
    let (substeps, iterations) = (4, 2);
    let (mut stepped, triangles) = cloth_patch(1e-4);
    let (mut counted, _) = cloth_patch(1e-4);
    let (mut unlimited, _) = cloth_patch(1e-4);
    stepped
        .enable_strain_limiting(triangles.clone(), limits(1.05, 0.95, sweeps))
        .unwrap();
    counted
        .enable_strain_limiting(triangles.clone(), limits(1.05, 0.95, sweeps))
        .unwrap();
    let distance_constraints = stepped.distance_constraints.len();
    for _ in 0..30 {
        stepped.step(FRAME, gravity, substeps, iterations);
        unlimited.step(FRAME, gravity, substeps, iterations);
        let stats = counted.step_with_stats(FRAME, gravity, substeps, iterations);
        assert_eq!(
            stats.total_projections,
            substeps * (iterations * distance_constraints + sweeps as usize * triangles.len())
        );
    }
    assert_eq!(stepped.positions, counted.positions);
    assert_eq!(stepped.velocities, counted.velocities);
    let deviation = stepped
        .positions
        .iter()
        .zip(&unlimited.positions)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        deviation > 1e-3,
        "the limiter must have changed the hang, deviation {deviation}"
    );
}

#[test]
fn disabling_strain_limiting_restores_plain_xpbd() {
    let gravity = Vec3::new(0.0, -980.0, 0.0);
    let (mut plain, triangles) = cloth_patch(1e-4);
    let (mut toggled, _) = cloth_patch(1e-4);
    toggled
        .enable_strain_limiting(triangles, limits(1.05, 0.95, 4))
        .unwrap();
    toggled.disable_strain_limiting();
    assert!(toggled.strain_limiter.is_none());
    for _ in 0..30 {
        plain.step(FRAME, gravity, 4, 2);
        toggled.step(FRAME, gravity, 4, 2);
    }
    assert_eq!(plain.positions, toggled.positions);
    assert_eq!(plain.velocities, toggled.velocities);
}

#[test]
fn limiter_that_has_nothing_to_limit_leaves_the_trajectory_untouched() {
    // Rigid constraints keep this patch within a fraction of a percent of its
    // rest shape under ordinary gravity: the limiter never has to act, and the
    // conversion of the positions to and from its own layout is exact.
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let (mut plain, triangles) = cloth_patch(0.0);
    let (mut limited, _) = cloth_patch(0.0);
    limited
        .enable_strain_limiting(triangles, limits(1.05, 0.95, 4))
        .unwrap();
    for _ in 0..60 {
        plain.step(FRAME, gravity, 4, 4);
        limited.step(FRAME, gravity, 4, 4);
    }
    assert!(
        plain.positions.iter().any(|p| p.y < -1e-4),
        "the patch sags"
    );
    assert_eq!(plain.positions, limited.positions);
    assert_eq!(plain.velocities, limited.velocities);
}
