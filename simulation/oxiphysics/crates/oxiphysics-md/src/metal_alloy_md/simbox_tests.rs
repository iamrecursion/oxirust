//! Regression tests for the simulation box (`SimBox`) and the fallible
//! integrator of [`AlloySystem`]: degenerate boxes, huge and non-finite
//! coordinates, and blow-up detection.

use super::types::*;

#[test]
fn try_new_rejects_non_finite_and_non_positive_edges() {
    let valid = SimBox::try_new(3.0, 4.0, 5.0).unwrap();
    assert_eq!(valid.lengths, [3.0, 4.0, 5.0]);
    assert_eq!(valid.xy, 0.0);
    for (edges, axis) in [
        ([0.0, 4.0, 5.0], "x"),
        ([3.0, -1.0, 5.0], "y"),
        ([3.0, 4.0, f64::NAN], "z"),
        ([f64::INFINITY, 4.0, 5.0], "x"),
        ([3.0, f64::NEG_INFINITY, 5.0], "y"),
    ] {
        let error = SimBox::try_new(edges[0], edges[1], edges[2])
            .expect_err("degenerate box accepted")
            .to_string();
        assert!(
            error.contains(&format!("along {axis}")),
            "{edges:?}: message `{error}` does not name axis {axis}"
        );
    }
}

#[test]
fn validate_checks_the_tilt_and_the_edges_of_an_existing_box() {
    assert!(SimBox::cubic(3.6).with_xy_tilt(1.2).validate().is_ok());
    assert!(
        SimBox::cubic(3.6)
            .with_xy_tilt(f64::NAN)
            .validate()
            .is_err()
    );
    assert!(
        SimBox::cubic(3.6)
            .with_xy_tilt(f64::INFINITY)
            .validate()
            .is_err()
    );
    let mut sim_box = SimBox::cubic(3.6);
    sim_box.lengths[2] = 0.0;
    assert!(sim_box.validate().is_err());
}

#[test]
fn wrapping_huge_and_non_finite_coordinates_returns_immediately_and_exactly() {
    // The loops this replaces subtracted one period at a time and never ended for
    // these coordinates (`1e300 - 3.6 == 1e300`).
    let len = 3.6;
    let sim_box = SimBox::cubic(len);
    let wrapped = sim_box.wrap([1.0e300, f64::INFINITY, -1.0e300]);
    // The reduction is the exact remainder of the double.
    assert_eq!(wrapped[0], 1.0e300_f64 % len);
    assert!(
        wrapped[1].is_infinite(),
        "a non-finite coordinate is left alone"
    );
    assert_eq!(wrapped[2], (-1.0e300_f64).rem_euclid(len));
    for value in [wrapped[0], wrapped[2]] {
        assert!((0.0..len).contains(&value), "{value} outside [0, {len})");
    }
    // Far beyond the direct form's range, yet still exact.
    assert_eq!(sim_box.wrap([1.0e12, 0.0, 0.0])[0], 1.0e12_f64 % len);
    // Within the direct form's range the two agree.
    let direct = sim_box.wrap([1.0e5, 0.0, 0.0])[0];
    assert!((direct - 1.0e5_f64 % len).abs() < 1.0e-9, "{direct}");
    // A coordinate that is an exact multiple of the period maps to zero, never to
    // the upper edge and never to a negative zero.
    let lattice_point = SimBox::cubic(10.0).wrap([1.0e300, -1.0e17, 0.0]);
    assert!(lattice_point.iter().all(|&c| (0.0..10.0).contains(&c)));
}

#[test]
fn minimum_image_of_huge_displacements_is_exact_and_bounded() {
    let len = 10.0;
    let sim_box = SimBox::cubic(len);
    for d in [1.0e300, -1.0e300, 1.0e12 + 3.0, -1.0e17 - 7.0] {
        let image = sim_box.min_image([d, 0.0, 0.0])[0];
        assert!(image.abs() <= 0.5 * len, "d = {d:e}: {image}");
        // congruent to d modulo the period
        let residue = ((d % len) - image).abs();
        assert!(
            residue < 1.0e-9 || (residue - len).abs() < 1.0e-9,
            "d = {d:e}: image {image} is not congruent to the displacement"
        );
    }
    // The tilt of a box with a huge y displacement stays finite.
    let tilted = SimBox::cubic(len).with_xy_tilt(2.0);
    let image = tilted.min_image([1.0, 1.0e300, 0.0]);
    assert!(image.iter().all(|c| c.is_finite()), "{image:?}");
    let untilted = SimBox::cubic(len).min_image([1.0, 1.0e300, 0.0]);
    assert!(untilted.iter().all(|c| c.is_finite()), "{untilted:?}");
}

/// A healthy copper dimer 2.4 Å apart in a 20 Å box.
fn dimer() -> AlloySystem {
    let mut sys = AlloySystem::new(
        vec![EamParams::copper()],
        SimBox::cubic(20.0),
        MixingRule::Geometric,
        0.001,
    );
    sys.add_atom([5.0, 5.0, 5.0], 0);
    sys.add_atom([7.4, 5.0, 5.0], 0);
    sys
}

#[test]
fn healthy_systems_step_and_run() {
    let mut sys = dimer();
    sys.run(25).unwrap();
    assert!(
        sys.atoms
            .iter()
            .all(|a| a.position.iter().all(|c| c.is_finite()))
    );
    sys.step().unwrap();
}

#[test]
fn step_reports_a_non_finite_state_instead_of_propagating_it() {
    // A NaN velocity.
    let mut sys = dimer();
    sys.compute_forces();
    sys.atoms[1].velocity[2] = f64::NAN;
    let error = sys.step().unwrap_err().to_string();
    assert!(
        error.contains("atom 1") && error.contains("velocity"),
        "{error}"
    );

    // A non-finite position.
    let mut sys = dimer();
    sys.compute_forces();
    sys.atoms[0].position[1] = f64::INFINITY;
    let error = sys.step().unwrap_err().to_string();
    assert!(
        error.contains("atom 0") && error.contains("position"),
        "{error}"
    );

    // A velocity so large that the drift overflows to infinity.
    let mut sys = dimer();
    sys.compute_forces();
    sys.dt = 1.0e10;
    sys.atoms[1].velocity = [1.0e308, 0.0, 0.0];
    let error = sys.step().unwrap_err().to_string();
    assert!(
        error.contains("atom 1") && error.contains("after the position update"),
        "{error}"
    );

    // A non-finite force that was never cleared.
    let mut sys = dimer();
    sys.atoms[0].force = [f64::NAN, 0.0, 0.0];
    let error = sys.step().unwrap_err().to_string();
    assert!(
        error.contains("atom 0") && error.contains("force"),
        "{error}"
    );

    // A non-finite timestep and a degenerate box.
    let mut sys = dimer();
    sys.dt = f64::NAN;
    assert!(sys.step().is_err());
    let mut sys = dimer();
    sys.sim_box.lengths[0] = 0.0;
    assert!(sys.step().is_err());
    assert!(sys.run(3).is_err());
}

#[test]
fn run_stops_at_the_first_failing_step() {
    let mut sys = dimer();
    sys.compute_forces();
    sys.atoms[1].velocity[0] = f64::INFINITY;
    assert!(sys.run(100).is_err());
    // The failed run did not silently keep integrating a poisoned state.
    assert!(sys.atoms[1].velocity[0].is_infinite());
}
