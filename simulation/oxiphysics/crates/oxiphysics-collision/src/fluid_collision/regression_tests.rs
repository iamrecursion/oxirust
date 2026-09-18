// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the sphere centre of buoyancy, the lift direction and
//! the underwater contact friction law.
//!
//! * `buoyancy_sphere` measured the centroid of the submerged spherical cap from
//!   the free surface although the cap formula gives its distance from the sphere
//!   centre; the centre of buoyancy was only right for the half-submerged sphere.
//! * `lift_force` pointed down for a positive lift coefficient (and along `+z` for
//!   a vertical relative velocity).
//! * `underwater_contact_friction` returned the Coulomb force for every viscous
//!   coefficient, so the documented viscous term had no effect.

use super::*;

/// Tolerance for the closed-form centre-of-buoyancy values.
const COB_TOL: f64 = 1e-12;

/// Height of the centroid of the part of a sphere (centre height `cy`, radius
/// `r`) that lies below the free surface `free_surface`, by midpoint integration
/// of the horizontal disc slices: an independent check of the cap formula.
fn submerged_centroid_by_slicing(cy: f64, r: f64, free_surface: f64) -> f64 {
    let bottom = cy - r;
    let top = free_surface.min(cy + r);
    let slices = 200_000_u32;
    let dy = (top - bottom) / f64::from(slices);
    let mut moment = 0.0;
    let mut volume = 0.0;
    for i in 0..slices {
        let y = bottom + (f64::from(i) + 0.5) * dy;
        let area = PI * (r * r - (y - cy) * (y - cy)).max(0.0);
        moment += y * area * dy;
        volume += area * dy;
    }
    moment / volume
}

fn sphere_with_depth(depth: f64, r: f64) -> RigidBodyFluidState {
    // The free surface is at y = 0, so the centre is `depth` below it.
    RigidBodyFluidState::new([0.3, -depth, -0.7], [0.0; 3], 1.0, r)
}

fn cob_y(depth: f64, r: f64) -> f64 {
    let fluid = FluidProperties::water();
    buoyancy_sphere(&sphere_with_depth(depth, r), &fluid, 0.0, 9.81).centre_of_buoyancy[1]
}

// ── buoyancy_sphere: centre of buoyancy ──────────────────────────────────────

#[test]
fn half_submerged_sphere_has_its_centre_of_buoyancy_three_eighths_below_the_centre() {
    let r = 1.0;
    let fluid = FluidProperties::water();
    let body = sphere_with_depth(0.0, r);
    let result = buoyancy_sphere(&body, &fluid, 0.0, 9.81);
    // The centroid of a solid hemisphere lies 3r/8 from the centre of its flat face.
    assert!(
        (result.centre_of_buoyancy[1] - (0.0 - 3.0 * r / 8.0)).abs() < COB_TOL,
        "cob = {:?}",
        result.centre_of_buoyancy
    );
    // The centre of buoyancy lies on the vertical through the sphere centre.
    assert_eq!(result.centre_of_buoyancy[0], body.position[0]);
    assert_eq!(result.centre_of_buoyancy[2], body.position[2]);
}

#[test]
fn centre_of_buoyancy_is_continuous_with_the_fully_submerged_branch() {
    let r = 0.8;
    // Just below the depth r at which the sphere is fully submerged.
    let almost = cob_y(r - 1e-9, r);
    assert!(
        (almost - (-(r - 1e-9))).abs() < 1e-8,
        "cob {almost} is not at the sphere centre {}",
        -(r - 1e-9)
    );
    // At and beyond that depth the centre of buoyancy is the sphere centre.
    assert_eq!(cob_y(r, r), -r);
    assert_eq!(cob_y(r + 3.0, r), -(r + 3.0));
}

#[test]
fn centre_of_buoyancy_of_a_barely_wetted_sphere_is_its_bottom_point() {
    let r = 1.5;
    let depth = -r + 1e-9;
    let cob = cob_y(depth, r);
    let bottom = -depth - r;
    assert!(
        (cob - bottom).abs() < 1e-8,
        "cob {cob} is not at the bottom point {bottom}"
    );
    // Not below the sphere, as the old formula (cy - 2r) put it.
    assert!(cob >= bottom - 1e-12);
}

#[test]
fn centre_of_buoyancy_matches_the_numerically_integrated_centroid() {
    for &(r, depth) in &[
        (1.3, -1.0),
        (1.3, -0.4),
        (1.3, 0.0),
        (1.3, 0.5),
        (1.3, 1.1),
        (0.25, 0.05),
        (4.0, -3.5),
    ] {
        let cy = -depth;
        let expected = submerged_centroid_by_slicing(cy, r, 0.0);
        let cob = cob_y(depth, r);
        assert!(
            (cob - expected).abs() < 1e-7 * r,
            "r = {r}, depth = {depth}: cob {cob} vs integrated centroid {expected}"
        );
    }
}

#[test]
fn centre_of_buoyancy_rises_monotonically_with_the_depth() {
    // A fixed sphere whose depth grows because the free surface rises from just
    // above the bottom point to the top point: the centre of buoyancy climbs
    // from the bottom point to the centre and never leaves the sphere's axis
    // range. (A free surface exactly at the bottom point wets nothing: there is
    // no submerged volume to take a centroid of.)
    let r = 1.0;
    let cy = 2.0;
    let fluid = FluidProperties::water();
    let body = RigidBodyFluidState::new([0.0, cy, 0.0], [0.0; 3], 1.0, r);
    let steps = 400;
    let mut previous = f64::NEG_INFINITY;
    for step in 0..=steps {
        let free_surface = cy - r + 1e-9 + 2.0 * r * f64::from(step) / f64::from(steps);
        let cob = buoyancy_sphere(&body, &fluid, free_surface, 9.81).centre_of_buoyancy[1];
        assert!(
            cob >= previous - 1e-12,
            "free surface {free_surface}: cob {cob} fell below {previous}"
        );
        assert!(cob >= cy - r - 1e-12, "cob {cob} below the bottom point");
        assert!(cob <= cy + 1e-12, "cob {cob} above the sphere centre");
        assert!(
            cob <= free_surface + 1e-12,
            "cob {cob} above the free surface {free_surface}"
        );
        previous = cob;
    }
    // From the bottom point to the centre.
    assert!((previous - cy).abs() < 1e-9);
}

#[test]
fn buoyancy_force_and_volume_are_those_of_the_spherical_cap() {
    let r = 1.2;
    let fluid = FluidProperties::water();
    let g = 9.81;
    let depth = 0.35;
    let body = sphere_with_depth(depth, r);
    let result = buoyancy_sphere(&body, &fluid, 0.0, g);
    let h = r + depth;
    let cap = PI * h * h * (r - h / 3.0);
    assert!((result.submerged_volume - cap).abs() < 1e-12);
    assert!((result.force[1] - fluid.density * g * cap).abs() < 1e-9);
    assert_eq!(result.force[0], 0.0);
    assert_eq!(result.force[2], 0.0);
}

// ── lift_force ────────────────────────────────────────────────────────────────

fn dynamic_pressure_force(fluid: &FluidProperties, cl: f64, area: f64, speed_sq: f64) -> f64 {
    0.5 * fluid.density * cl * area * speed_sq
}

#[test]
fn positive_lift_coefficient_lifts_a_body_in_horizontal_flow() {
    let fluid = FluidProperties::water();
    let f = lift_force([1.0, 0.0, 0.0], [0.0; 3], &fluid, 1.0, 0.5);
    let expected = dynamic_pressure_force(&fluid, 1.0, 0.5, 1.0);
    assert!(f[1] > 0.0, "lift must point up, got {f:?}");
    assert!(
        (f[1] - expected).abs() < 1e-9,
        "f = {f:?}, expected {expected}"
    );
    assert!(f[0].abs() < 1e-12 && f[2].abs() < 1e-12, "f = {f:?}");
}

#[test]
fn negative_lift_coefficient_presses_the_body_down() {
    let fluid = FluidProperties::water();
    let f = lift_force([-2.0, 0.0, 0.0], [0.0; 3], &fluid, -0.5, 0.2);
    let expected = dynamic_pressure_force(&fluid, -0.5, 0.2, 4.0);
    assert!(f[1] < 0.0, "downforce expected, got {f:?}");
    assert!(
        (f[1] - expected).abs() < 1e-9,
        "f = {f:?}, expected {expected}"
    );
}

#[test]
fn lift_vanishes_for_a_vertical_relative_velocity() {
    let fluid = FluidProperties::water();
    // The old direction fell back to +z here and produced a spurious force.
    assert_eq!(
        lift_force([0.0, 1.0, 0.0], [0.0; 3], &fluid, 1.0, 1.0),
        [0.0; 3]
    );
    assert_eq!(
        lift_force([0.0, -3.0, 0.0], [0.0; 3], &fluid, 1.0, 1.0),
        [0.0; 3]
    );
    // A horizontal flow that cancels the horizontal body velocity leaves a
    // vertical relative velocity.
    assert_eq!(
        lift_force([2.0, 1.0, 0.0], [2.0, 0.0, 0.0], &fluid, 1.0, 1.0),
        [0.0; 3]
    );
    // No relative motion at all.
    assert_eq!(
        lift_force([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], &fluid, 1.0, 1.0),
        [0.0; 3]
    );
}

#[test]
fn lift_is_perpendicular_to_the_relative_velocity_in_the_vertical_plane() {
    let fluid = FluidProperties::water();
    let area = 0.8;
    let cl = 0.6;

    // v = (3, 4, 0): the lift direction is (-0.8, 0.6, 0).
    let v = [3.0, 4.0, 0.0];
    let f = lift_force(v, [0.0; 3], &fluid, cl, area);
    let magnitude = dynamic_pressure_force(&fluid, cl, area, 25.0);
    assert!((f[0] - (-0.8) * magnitude).abs() < 1e-9, "f = {f:?}");
    assert!((f[1] - 0.6 * magnitude).abs() < 1e-9, "f = {f:?}");
    assert!(f[2].abs() < 1e-12, "f = {f:?}");
    assert!(dot3(f, v).abs() < 1e-9, "lift must be perpendicular to v");

    // A fully three-dimensional case: v = (1, 2, 2), |v|^2 = 9. The lift direction
    // is up |v|^2 - v (v . up) = (-2, 5, -4) / sqrt(45).
    let v = [1.0, 2.0, 2.0];
    let f = lift_force(v, [0.0; 3], &fluid, cl, area);
    let magnitude = dynamic_pressure_force(&fluid, cl, area, 9.0);
    let norm = 45.0_f64.sqrt();
    assert!((f[0] - (-2.0 / norm) * magnitude).abs() < 1e-9, "f = {f:?}");
    assert!((f[1] - (5.0 / norm) * magnitude).abs() < 1e-9, "f = {f:?}");
    assert!((f[2] - (-4.0 / norm) * magnitude).abs() < 1e-9, "f = {f:?}");
    assert!(dot3(f, v).abs() < 1e-9, "lift must be perpendicular to v");
    // The force lies in the vertical plane through v: its normal is v x up.
    let plane_normal = cross3(v, [0.0, 1.0, 0.0]);
    assert!(dot3(f, plane_normal).abs() < 1e-9);
    assert!(len3(f) > 0.0 && (len3(f) - magnitude).abs() < 1e-9);
}

#[test]
fn lift_uses_the_velocity_relative_to_the_fluid() {
    let fluid = FluidProperties::water();
    // A body at rest in a stream flowing along -x sees v_rel = (2, 0, 0).
    let f = lift_force([0.0; 3], [-2.0, 0.0, 0.0], &fluid, 1.0, 0.5);
    let expected = dynamic_pressure_force(&fluid, 1.0, 0.5, 4.0);
    assert!((f[1] - expected).abs() < 1e-9, "f = {f:?}");
    // Moving with the stream: no relative velocity, no lift.
    assert_eq!(
        lift_force([5.0, 0.0, 1.0], [5.0, 0.0, 1.0], &fluid, 1.0, 0.5),
        [0.0; 3]
    );
}

// ── underwater_contact_friction ───────────────────────────────────────────────

fn norm_of(f: &UnderwaterFrictionResult) -> f64 {
    len3(f.friction_force)
}

#[test]
fn viscous_term_limits_the_friction_below_the_coulomb_limit() {
    // Coulomb limit mu |F_n| = 0.5 * 100 = 50 N; viscous force c |v_t| = 1 N.
    let r = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.0, 1.0);
    assert!((norm_of(&r) - 1.0).abs() < 1e-12, "{r:?}");
    assert!((r.friction_force[0] + 1.0).abs() < 1e-12, "{r:?}");
    assert!(!r.is_sliding, "{r:?}");
}

#[test]
fn friction_saturates_at_the_coulomb_limit_for_a_large_viscous_coefficient() {
    let r = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.0, 1000.0);
    assert!((norm_of(&r) - 50.0).abs() < 1e-12, "{r:?}");
    assert!((r.friction_force[0] + 50.0).abs() < 1e-12, "{r:?}");
    assert!(r.is_sliding, "{r:?}");
}

#[test]
fn the_viscous_coefficient_changes_the_friction_force() {
    let mut previous = -1.0;
    for c in [0.5, 1.0, 5.0, 20.0, 49.0] {
        let r = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.0, c);
        assert!((norm_of(&r) - c).abs() < 1e-12, "c = {c}: {r:?}");
        assert!(norm_of(&r) > previous, "force must grow with c");
        assert!(!r.is_sliding, "c = {c}: {r:?}");
        previous = norm_of(&r);
    }
    // Once c |v_t| reaches the Coulomb limit the force stays there.
    for c in [50.0, 51.0, 80.0, 1e6] {
        let r = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.0, c);
        assert!((norm_of(&r) - 50.0).abs() < 1e-12, "c = {c}: {r:?}");
        assert!(r.is_sliding, "c = {c}: {r:?}");
    }
}

#[test]
fn friction_ramps_up_linearly_with_the_sliding_speed_and_then_saturates() {
    // c = 10 N s/m, limit 50 N: the ramp ends at |v_t| = 5 m/s.
    for speed in [0.1, 1.0, 2.5, 4.9] {
        let r = underwater_contact_friction(100.0, [0.0, 0.0, speed], 0.5, 0.0, 10.0);
        assert!(
            (norm_of(&r) - 10.0 * speed).abs() < 1e-9,
            "speed {speed}: {r:?}"
        );
        assert!(!r.is_sliding);
    }
    for speed in [5.0, 6.0, 100.0] {
        let r = underwater_contact_friction(100.0, [0.0, 0.0, speed], 0.5, 0.0, 10.0);
        assert!((norm_of(&r) - 50.0).abs() < 1e-9, "speed {speed}: {r:?}");
        assert!(r.is_sliding);
    }
}

#[test]
fn friction_opposes_the_tangential_velocity() {
    // c |v_t| = 5 N < 50 N: the force is -c v_t.
    let v = [3.0, 0.0, 4.0];
    let r = underwater_contact_friction(100.0, v, 0.5, 0.0, 1.0);
    assert!((r.friction_force[0] + 3.0).abs() < 1e-12, "{r:?}");
    assert!(r.friction_force[1].abs() < 1e-12, "{r:?}");
    assert!((r.friction_force[2] + 4.0).abs() < 1e-12, "{r:?}");
}

#[test]
fn lubrication_lowers_the_coulomb_limit() {
    let dry = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.0, 1e6);
    let wet = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.5, 1e6);
    assert!((norm_of(&dry) - 50.0).abs() < 1e-12);
    assert!((norm_of(&wet) - 25.0).abs() < 1e-12, "{wet:?}");
    assert!(wet.is_sliding);
}

#[test]
fn zero_or_negative_coefficients_never_accelerate_the_contact() {
    // c = 0: the viscous ramp is flat, no friction force.
    let r = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.0, 0.0);
    assert_eq!(r.friction_force, [0.0; 3]);
    assert!(!r.is_sliding);
    // Negative coefficients are clamped to zero instead of reversing the force.
    let r = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], 0.5, 0.0, -5.0);
    assert_eq!(r.friction_force, [0.0; 3]);
    let r = underwater_contact_friction(100.0, [1.0, 0.0, 0.0], -0.5, 0.0, 10.0);
    assert_eq!(r.friction_force, [0.0; 3]);
}

#[test]
fn no_tangential_motion_means_no_friction_and_no_sliding() {
    let r = underwater_contact_friction(100.0, [0.0; 3], 0.5, 0.3, 10.0);
    assert_eq!(r.friction_force, [0.0; 3]);
    assert!(!r.is_sliding);
    let r = underwater_contact_friction(100.0, [1e-13, 0.0, 0.0], 0.5, 0.3, 1e6);
    assert_eq!(r.friction_force, [0.0; 3]);
    assert!(!r.is_sliding);
}
