// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Cartesian → Kepler conversion and the Lambert
//! solver of [`super`].
//!
//! The conversion is checked by round trips through [`kepler_to_cartesian`] and
//! by asserting the documented conventions for circular and equatorial orbits
//! directly. Lambert transfers are verified against numerical propagation with
//! the module's own RK4 integrator.

use super::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::TAU;

/// Largest tolerated position change of a state → elements → state round trip \[m\].
const POSITION_TOL: f64 = 1e-6;

/// Largest tolerated velocity change of a state → elements → state round trip \[m/s\].
const VELOCITY_TOL: f64 = 1e-9;

/// Smallest difference between two angles modulo 2π.
fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(TAU);
    d.min(TAU - d)
}

/// Convert `elem` to a state and back, require the recovered elements to be in
/// range and to regenerate the same state, and return them.
fn assert_round_trip(label: &str, elem: &KeplerElements) -> KeplerElements {
    let (pos, vel) = kepler_to_cartesian(elem, MU_EARTH);
    let back = cartesian_to_kepler(pos, vel, MU_EARTH);
    assert!(
        (0.0..=PI).contains(&back.inclination),
        "{label}: inclination out of range in {back:?}"
    );
    for angle in [back.raan, back.arg_periapsis, back.true_anomaly] {
        assert!(
            (0.0..TAU).contains(&angle),
            "{label}: angle {angle} outside [0, 2π) in {back:?}"
        );
    }
    let (pos2, vel2) = kepler_to_cartesian(&back, MU_EARTH);
    let position_error = norm3(sub3(pos2, pos));
    let velocity_error = norm3(sub3(vel2, vel));
    assert!(
        position_error < POSITION_TOL,
        "{label}: position differs by {position_error} m after the round trip ({back:?})"
    );
    assert!(
        velocity_error < VELOCITY_TOL,
        "{label}: velocity differs by {velocity_error} m/s after the round trip ({back:?})"
    );
    back
}

// ---- circular orbits ------------------------------------------------------

#[test]
fn circular_inclined_orbit_returns_its_argument_of_latitude() {
    // Before the fix every circular state mapped to ν = π/2.
    let elem = KeplerElements::new(7e6, 0.0, 0.5, 0.0, 0.0, 1.0);
    let back = assert_round_trip("circular inclined", &elem);
    assert!(angle_diff(back.true_anomaly, 1.0) < 1e-9, "{back:?}");
    assert_eq!(back.arg_periapsis, 0.0);
    assert!(angle_diff(back.raan, 0.0) < 1e-12, "{back:?}");
    assert!((back.inclination - 0.5).abs() < 1e-12, "{back:?}");
}

#[test]
fn circular_inclined_orbit_folds_periapsis_into_true_anomaly() {
    // For e = 0 only Ω and the argument of latitude ω + ν are observable.
    let elem = KeplerElements::new(7e6, 0.0, 0.5, 0.3, 0.2, 1.0);
    let back = assert_round_trip("circular inclined, Ω ω ≠ 0", &elem);
    assert_eq!(back.arg_periapsis, 0.0);
    assert!(angle_diff(back.raan, 0.3) < 1e-9, "{back:?}");
    assert!(angle_diff(back.true_anomaly, 1.2) < 1e-9, "{back:?}");
}

#[test]
fn circular_inclined_orbit_covers_the_whole_circle() {
    for step in 0..24 {
        let nu = TAU * f64::from(step) / 24.0 + 0.05;
        let elem = KeplerElements::new(7.2e6, 0.0, 1.1, 2.0, 0.0, nu);
        let back = assert_round_trip("circular inclined sweep", &elem);
        assert!(
            angle_diff(back.true_anomaly, nu) < 1e-9,
            "ν = {nu} came back as {}",
            back.true_anomaly
        );
    }
}

#[test]
fn circular_equatorial_orbit_returns_its_true_longitude() {
    let elem = KeplerElements::new(7e6, 0.0, 0.0, 0.0, 0.0, 2.0);
    let back = assert_round_trip("circular equatorial", &elem);
    assert!(angle_diff(back.true_anomaly, 2.0) < 1e-9, "{back:?}");
    assert_eq!(back.raan, 0.0);
    assert_eq!(back.arg_periapsis, 0.0);
    assert!(back.inclination < 1e-12, "{back:?}");
}

#[test]
fn circular_equatorial_orbit_folds_every_angle_into_the_true_longitude() {
    // With i = 0 the rotation is Rz(Ω + ω): the longitude is Ω + ω + ν.
    let elem = KeplerElements::new(7e6, 0.0, 0.0, 0.9, 0.4, 2.0);
    let back = assert_round_trip("circular equatorial, Ω ω ≠ 0", &elem);
    assert_eq!(back.raan, 0.0);
    assert_eq!(back.arg_periapsis, 0.0);
    assert!(angle_diff(back.true_anomaly, 3.3) < 1e-9, "{back:?}");
}

#[test]
fn retrograde_circular_equatorial_orbit_measures_angles_clockwise() {
    let radius = 7e6;
    let speed = circular_speed(radius, MU_EARTH);
    // Clockwise (retrograde) circular state, z exactly 0: at +x moving -y.
    let at_x = cartesian_to_kepler([radius, 0.0, 0.0], [0.0, -speed, 0.0], MU_EARTH);
    assert!((at_x.inclination - PI).abs() < 1e-12, "{at_x:?}");
    assert_eq!(at_x.raan, 0.0);
    assert_eq!(at_x.arg_periapsis, 0.0);
    assert!(angle_diff(at_x.true_anomaly, 0.0) < 1e-12, "{at_x:?}");
    // A quarter turn clockwise from +x lies on -y: ν = π/2.
    let at_minus_y = cartesian_to_kepler([0.0, -radius, 0.0], [-speed, 0.0, 0.0], MU_EARTH);
    assert!(
        angle_diff(at_minus_y.true_anomaly, 0.5 * PI) < 1e-12,
        "{at_minus_y:?}"
    );
    // +y is three quarter turns clockwise from +x: ν = 3π/2.
    let at_y = cartesian_to_kepler([0.0, radius, 0.0], [speed, 0.0, 0.0], MU_EARTH);
    assert!(angle_diff(at_y.true_anomaly, 1.5 * PI) < 1e-12, "{at_y:?}");
    for (pos, vel) in [
        ([radius, 0.0, 0.0], [0.0, -speed, 0.0]),
        ([0.0, -radius, 0.0], [-speed, 0.0, 0.0]),
        ([0.0, radius, 0.0], [speed, 0.0, 0.0]),
    ] {
        let back = cartesian_to_kepler(pos, vel, MU_EARTH);
        let (pos2, vel2) = kepler_to_cartesian(&back, MU_EARTH);
        assert!(norm3(sub3(pos2, pos)) < POSITION_TOL, "{back:?}");
        assert!(norm3(sub3(vel2, vel)) < VELOCITY_TOL, "{back:?}");
    }
}

#[test]
fn retrograde_circular_orbit_from_the_element_conversion_round_trips() {
    // sin(π) ≠ 0, so the node vector of this state is round-off, not zero.
    let elem = KeplerElements::new(7e6, 0.0, PI, 0.0, 0.0, 2.0);
    let back = assert_round_trip("circular i = π", &elem);
    assert!((back.inclination - PI).abs() < 1e-12, "{back:?}");
    assert_eq!(back.raan, 0.0);
    assert_eq!(back.arg_periapsis, 0.0);
    assert!(angle_diff(back.true_anomaly, 2.0) < 1e-9, "{back:?}");
}

#[test]
fn retrograde_inclined_circular_orbit_round_trips() {
    let elem = KeplerElements::new(7e6, 0.0, PI - 0.3, 1.0, 0.0, 2.5);
    let back = assert_round_trip("circular retrograde inclined", &elem);
    assert!((back.inclination - (PI - 0.3)).abs() < 1e-12, "{back:?}");
    assert!(angle_diff(back.raan, 1.0) < 1e-9, "{back:?}");
    assert!(angle_diff(back.true_anomaly, 2.5) < 1e-9, "{back:?}");
}

// ---- equatorial elliptic orbits -------------------------------------------

#[test]
fn equatorial_elliptic_orbit_keeps_its_longitude_of_periapsis() {
    // Before the fix ω was forced to 0 while ν was still measured from the
    // eccentricity vector, so the position came back rotated by ω.
    let elem = KeplerElements::new(8e6, 0.1, 0.0, 0.0, 0.7, 0.3);
    let back = assert_round_trip("equatorial elliptic", &elem);
    assert!(angle_diff(back.arg_periapsis, 0.7) < 1e-9, "{back:?}");
    assert!(angle_diff(back.true_anomaly, 0.3) < 1e-9, "{back:?}");
    assert!((back.eccentricity - 0.1).abs() < 1e-12, "{back:?}");
    assert!((back.semi_major_axis - 8e6).abs() < 1e-3, "{back:?}");
    assert_eq!(back.raan, 0.0);
}

#[test]
fn equatorial_elliptic_orbit_folds_raan_into_the_periapsis_longitude() {
    let elem = KeplerElements::new(8e6, 0.25, 0.0, 0.5, 0.7, 4.0);
    let back = assert_round_trip("equatorial elliptic, Ω ≠ 0", &elem);
    assert_eq!(back.raan, 0.0);
    assert!(angle_diff(back.arg_periapsis, 1.2) < 1e-9, "{back:?}");
    assert!(angle_diff(back.true_anomaly, 4.0) < 1e-9, "{back:?}");
}

#[test]
fn retrograde_equatorial_elliptic_orbit_round_trips() {
    // Mirror a prograde equatorial state in the x axis, so z stays exactly 0
    // and h_z < 0: the angles are then measured clockwise from +x.
    let prograde = KeplerElements::new(8e6, 0.2, 0.0, 0.0, 0.7, 0.3);
    let (pos, vel) = kepler_to_cartesian(&prograde, MU_EARTH);
    let pos = [pos[0], -pos[1], 0.0];
    let vel = [vel[0], -vel[1], 0.0];
    let back = cartesian_to_kepler(pos, vel, MU_EARTH);
    assert!((back.inclination - PI).abs() < 1e-12, "{back:?}");
    assert_eq!(back.raan, 0.0);
    assert!(angle_diff(back.arg_periapsis, 0.7) < 1e-9, "{back:?}");
    assert!(angle_diff(back.true_anomaly, 0.3) < 1e-9, "{back:?}");
    let (pos2, vel2) = kepler_to_cartesian(&back, MU_EARTH);
    assert!(norm3(sub3(pos2, pos)) < POSITION_TOL, "{back:?}");
    assert!(norm3(sub3(vel2, vel)) < VELOCITY_TOL, "{back:?}");
}

#[test]
fn retrograde_equatorial_orbit_from_the_element_conversion_round_trips() {
    // For i = π, Ω = 0.5: the periapsis lies at Ω - ω counter-clockwise from +x,
    // i.e. at ω - Ω clockwise, and ν is unchanged.
    let elem = KeplerElements::new(8e6, 0.2, PI, 0.5, 0.7, 0.3);
    let back = assert_round_trip("equatorial retrograde elliptic", &elem);
    assert_eq!(back.raan, 0.0);
    assert!(angle_diff(back.arg_periapsis, 0.2) < 1e-9, "{back:?}");
    assert!(angle_diff(back.true_anomaly, 0.3) < 1e-9, "{back:?}");
}

// ---- general orbits -------------------------------------------------------

#[test]
fn inclined_elliptic_orbits_recover_every_element() {
    for step in 0..12 {
        let nu = TAU * f64::from(step) / 12.0 + 0.1;
        let elem = KeplerElements::new(9e6, 0.3, 1.0, 2.0, 3.0, nu);
        let back = assert_round_trip("inclined elliptic", &elem);
        assert!((back.semi_major_axis - 9e6).abs() < 1e-3, "{back:?}");
        assert!((back.eccentricity - 0.3).abs() < 1e-12, "{back:?}");
        assert!((back.inclination - 1.0).abs() < 1e-12, "{back:?}");
        assert!(angle_diff(back.raan, 2.0) < 1e-10, "{back:?}");
        assert!(angle_diff(back.arg_periapsis, 3.0) < 1e-10, "{back:?}");
        assert!(angle_diff(back.true_anomaly, nu) < 1e-10, "{back:?}");
    }
}

#[test]
fn hyperbolic_orbits_round_trip() {
    for (inclination, raan) in [(0.0, 0.0), (0.4, 1.0), (PI, 0.0)] {
        let elem = KeplerElements::new(-2e7, 1.5, inclination, raan, 0.8, 0.5);
        let back = assert_round_trip("hyperbolic", &elem);
        assert!(back.semi_major_axis < 0.0, "{back:?}");
        assert!((back.eccentricity - 1.5).abs() < 1e-10, "{back:?}");
    }
}

#[test]
fn nearly_circular_orbits_stay_accurate_just_above_the_circular_threshold() {
    for eccentricity in [1e-8, 1e-10, 1e-11] {
        let elem = KeplerElements::new(7e6, eccentricity, 0.9, 1.3, 0.6, 2.2);
        let back = assert_round_trip("nearly circular", &elem);
        // ω and ν are individually ill-conditioned here, but ω + ν is not.
        assert!(
            angle_diff(back.arg_periapsis + back.true_anomaly, 0.6 + 2.2) < 1e-4,
            "e = {eccentricity}: {back:?}"
        );
        assert!(angle_diff(back.raan, 1.3) < 1e-6, "{back:?}");
    }
}

#[test]
fn tiny_inclinations_are_not_flushed_to_zero() {
    // acos(h_z / |h|) returns exactly 0 (or π) for |i| below about 1e-8.
    for inclination in [1e-9, PI - 1e-9] {
        let elem = KeplerElements::new(8e6, 0.2, inclination, 0.4, 0.5, 0.6);
        let back = assert_round_trip("tiny inclination", &elem);
        assert!(
            (back.inclination - inclination).abs() < 1e-12,
            "i = {inclination} came back as {}",
            back.inclination
        );
    }
}

#[test]
fn random_states_round_trip_with_in_range_finite_elements() {
    let mut rng = StdRng::seed_from_u64(0x000b_17a1);
    let eccentricities = [0.0, 1e-15, 1e-11, 1e-8, 0.05, 0.4, 0.8];
    let inclinations = [0.0, 1e-10, 0.3, 1.2, PI / 2.0, 2.5, PI - 1e-10, PI];
    for case in 0..2000 {
        let semi_major_axis = 6.7e6 + 3.6e7 * rng.random::<f64>();
        let eccentricity = eccentricities[case % eccentricities.len()];
        let inclination = inclinations[(case / eccentricities.len()) % inclinations.len()];
        let elem = KeplerElements::new(
            semi_major_axis,
            eccentricity,
            inclination,
            TAU * rng.random::<f64>(),
            TAU * rng.random::<f64>(),
            TAU * rng.random::<f64>(),
        );
        let back = assert_round_trip(&format!("random case {case}"), &elem);
        assert!(
            (back.semi_major_axis - semi_major_axis).abs() < 1e-3,
            "case {case}: {back:?}"
        );
        assert!(
            (back.eccentricity - eccentricity).abs() < 1e-9,
            "case {case}: {back:?}"
        );
    }
}

// ---- Lambert's problem ----------------------------------------------------

/// Number of RK4 sub-steps used to propagate a Lambert departure state.
const LAMBERT_SUBSTEPS: usize = 10_000;

/// Propagate a two-body state (no J2) over `tof` seconds.
fn propagate(pos: [f64; 3], vel: [f64; 3], tof: f64) -> ([f64; 3], [f64; 3]) {
    let dt = tof / LAMBERT_SUBSTEPS as f64;
    let (mut pos, mut vel) = (pos, vel);
    for _ in 0..LAMBERT_SUBSTEPS {
        (pos, vel) = propagate_rk4(pos, vel, dt, MU_EARTH, 0.0, R_EARTH_EQ);
    }
    (pos, vel)
}

/// Solve Lambert's problem, fly the departure state for `tof` and require the
/// body to arrive at `r2` with the arrival velocity of the solution.
fn assert_lambert_arrives(
    label: &str,
    r1: [f64; 3],
    r2: [f64; 3],
    tof: f64,
    prograde: bool,
) -> LambertSolution {
    let solution = lambert_problem(r1, r2, tof, MU_EARTH, prograde)
        .unwrap_or_else(|| panic!("{label}: no transfer found"));
    let (r_end, v_end) = propagate(r1, solution.v1, tof);
    let miss = norm3(sub3(r_end, r2));
    assert!(miss < 1.0, "{label}: arrival misses r2 by {miss} m");
    assert!(
        miss / norm3(r2) < 1e-6,
        "{label}: relative miss {} exceeds 1e-6",
        miss / norm3(r2)
    );
    let velocity_error = norm3(sub3(v_end, solution.v2));
    assert!(
        velocity_error < 1e-3,
        "{label}: arrival velocity is off by {velocity_error} m/s"
    );
    // The requested sense of rotation: angular momentum along +z (prograde) or -z.
    let h = cross3(r1, solution.v1);
    if prograde {
        assert!(h[2] > 0.0, "{label}: prograde transfer has h = {h:?}");
    } else {
        assert!(h[2] < 0.0, "{label}: retrograde transfer has h = {h:?}");
    }
    // The semi-major axis follows from the departure energy.
    let energy = 0.5 * dot3(solution.v1, solution.v1) - MU_EARTH / norm3(r1);
    let expected_axis = -MU_EARTH / (2.0 * energy);
    assert!(
        (solution.semi_major_axis - expected_axis).abs() < 1e-6 * expected_axis.abs(),
        "{label}: a = {} but the energy gives {expected_axis}",
        solution.semi_major_axis
    );
    solution
}

/// Point at radius `radius` and polar angle `angle` in the xy plane.
fn in_plane(radius: f64, angle: f64) -> [f64; 3] {
    [radius * angle.cos(), radius * angle.sin(), 0.0]
}

#[test]
fn lambert_quarter_orbit_transfer_reaches_the_target() {
    let solution = assert_lambert_arrives(
        "90 degrees",
        [7e6, 0.0, 0.0],
        [0.0, 7e6, 0.0],
        1_500.0,
        true,
    );
    assert!(solution.semi_major_axis > 0.0, "{solution:?}");
}

#[test]
fn lambert_135_degree_transfer_reaches_the_target() {
    assert_lambert_arrives(
        "135 degrees",
        [7e6, 0.0, 0.0],
        in_plane(7.4e6, 135.0_f64.to_radians()),
        2_300.0,
        true,
    );
}

#[test]
fn lambert_270_degree_long_way_transfer_reaches_the_target() {
    // r1 × r2 points along -z, so the prograde transfer is the long way round.
    assert_lambert_arrives(
        "270 degrees prograde",
        [7e6, 0.0, 0.0],
        [0.0, -7e6, 0.0],
        4_000.0,
        true,
    );
}

#[test]
fn lambert_retrograde_transfers_reach_the_target() {
    // Retrograde short way (90 degrees clockwise) and long way (270 degrees).
    assert_lambert_arrives(
        "90 degrees retrograde",
        [7e6, 0.0, 0.0],
        [0.0, -7e6, 0.0],
        1_500.0,
        false,
    );
    assert_lambert_arrives(
        "270 degrees retrograde",
        [7e6, 0.0, 0.0],
        [0.0, 7e6, 0.0],
        4_000.0,
        false,
    );
}

#[test]
fn lambert_unequal_radii_transfer_reaches_the_target() {
    assert_lambert_arrives(
        "unequal radii",
        [7e6, 0.0, 0.0],
        in_plane(1.1e7, 100.0_f64.to_radians()),
        3_200.0,
        true,
    );
}

#[test]
fn lambert_transfer_out_of_the_xy_plane_reaches_the_target() {
    let solution = assert_lambert_arrives(
        "inclined plane",
        [6.8e6, 5.0e5, 3.0e5],
        [-2.0e6, 7.0e6, 2.5e6],
        2_500.0,
        true,
    );
    // The velocity stays in the plane spanned by r1 and r2.
    let normal = cross3([6.8e6, 5.0e5, 3.0e5], [-2.0e6, 7.0e6, 2.5e6]);
    assert!(
        dot3(solution.v1, normal).abs() < 1e-9 * norm3(solution.v1) * norm3(normal),
        "v1 leaves the transfer plane: {solution:?}"
    );
}

#[test]
fn lambert_hyperbolic_transfer_has_a_negative_semi_major_axis() {
    let solution = assert_lambert_arrives(
        "hyperbolic",
        [7e6, 0.0, 0.0],
        [2.0e7, 2.5e7, 0.0],
        1_500.0,
        true,
    );
    assert!(solution.semi_major_axis < 0.0, "{solution:?}");
}

#[test]
fn lambert_problem_returns_the_shared_izzo_solution() {
    let r1 = [7e6, 0.0, 0.0];
    let r2 = in_plane(7.4e6, 135.0_f64.to_radians());
    for prograde in [true, false] {
        let shared = lambert_solver(r1, r2, 2_300.0, MU_EARTH, prograde)
            .expect("the shared solver handles a regular transfer");
        let local = lambert_problem(r1, r2, 2_300.0, MU_EARTH, prograde)
            .expect("the wrapper handles a regular transfer");
        assert_eq!(local.v1, shared.v1);
        assert_eq!(local.v2, shared.v2);
        assert_eq!(local.semi_major_axis, shared.semi_major_axis);
    }
}

#[test]
fn lambert_without_a_transfer_plane_returns_none() {
    // Antipodal positions (the old test's "half-period" case): the plane of
    // the transfer is undefined, so there is no unique solution.
    let tof = orbital_period(7e6, MU_EARTH) / 2.0;
    assert!(lambert_problem([7e6, 0.0, 0.0], [-7e6, 0.0, 0.0], tof, MU_EARTH, true).is_none());
    // Same direction, different radius.
    assert!(lambert_problem([7e6, 0.0, 0.0], [9e6, 0.0, 0.0], 1_000.0, MU_EARTH, true).is_none());
}

#[test]
fn lambert_rejects_ill_posed_inputs() {
    let (r1, r2) = ([7e6, 0.0, 0.0], [0.0, 7e6, 0.0]);
    assert!(lambert_problem(r1, r2, 0.0, MU_EARTH, true).is_none());
    assert!(lambert_problem(r1, r2, -10.0, MU_EARTH, true).is_none());
    assert!(lambert_problem(r1, r2, f64::NAN, MU_EARTH, true).is_none());
    assert!(lambert_problem(r1, r2, 1_500.0, 0.0, true).is_none());
    assert!(lambert_problem([0.0; 3], r2, 1_500.0, MU_EARTH, true).is_none());
    assert!(lambert_problem(r1, [f64::NAN, 0.0, 0.0], 1_500.0, MU_EARTH, true).is_none());
}
