//! Regression tests: limits that come from public fields and setters must not
//! make a solver panic.
//!
//! `f64::clamp` panics when its lower bound exceeds the upper one or when a
//! bound is NaN.  Motor torque limits, anti-windup limits, impulse bounds and
//! compliance ranges are all plain `pub` data (or unvalidated constructor
//! arguments), so a negative `max_force`, an inverted `[min, max]` pair or a NaN
//! used to bring down the whole simulation step.  A symmetric limit is now read
//! as its magnitude (NaN counts as zero) and a pair of bounds in the wrong order
//! is swapped (a NaN bound leaves its side unbounded).

use oxiphysics_constraints::pgs_solver::{
    clamp_friction_lambda, solve_single_constraint, solve_single_constraint_sor,
};
use oxiphysics_constraints::pid_motor::{PidController, PidMotorJoint};
use oxiphysics_constraints::six_dof_constraint::{
    PositionDriveConfig, VelocityDriveConfig, clamp_accumulated_impulse,
};
use oxiphysics_constraints::tgs_solver::{
    AdaptiveTgsSolver, BlockConstraint6, JacobianRow6, MassMatrix6, SoftParams,
};
use oxiphysics_constraints::traits::types::{
    AngularMotorData, AngularMotorMode, ComplianceRange, LagrangeMultiplier,
};
use oxiphysics_core::math::Vec3;

const DT: f64 = 1.0 / 60.0;

// ── Symmetric limits (negative or NaN magnitudes) ────────────────────────────

#[test]
fn pid_integral_clamp_is_read_as_a_magnitude() {
    // |clamp| = 2: a unit integral gain turns an error of 10 over 1 s into 2.
    let mut pid = PidController::new(0.0, 1.0, 0.0).with_integral_clamp(-2.0);
    assert_eq!(pid.update(10.0, 1.0), 2.0);
    assert_eq!(pid.integral, 2.0);
    assert_eq!(pid.update(-30.0, 1.0), -2.0);

    // A NaN clamp holds the integral term at zero.
    let mut pid = PidController::new(0.5, 1.0, 0.0).with_integral_clamp(f64::NAN);
    assert_eq!(pid.update(10.0, 1.0), 5.0);
    assert_eq!(pid.integral, 0.0);

    // The default is no clamping at all.
    let mut pid = PidController::new(0.0, 1.0, 0.0);
    assert_eq!(pid.update(10.0, 1.0), 10.0);
}

#[test]
fn pid_motor_torque_limit_is_read_as_a_magnitude() {
    let mut motor = PidMotorJoint::new(100.0, 0.0, 0.0, -3.0);
    motor.set_target(1.0);
    assert_eq!(motor.update(DT), 3.0);
    motor.set_target(-1.0);
    assert_eq!(motor.update(DT), -3.0);

    let mut motor = PidMotorJoint::new(100.0, 0.0, 0.0, f64::NAN);
    motor.set_target(1.0);
    assert_eq!(motor.update(DT), 0.0);

    // The same limit also bounds the simulated joint.
    let mut motor = PidMotorJoint::new(100.0, 0.0, 0.0, -3.0);
    motor.set_target(1.0);
    motor.simulate_step(0.0, 1.0, DT);
    assert!((motor.current_velocity - 3.0 * DT).abs() < 1e-12);
}

#[test]
fn angular_motor_torque_limit_is_read_as_a_magnitude() {
    for (max_torque, expected) in [(-5.0, 5.0), (5.0, 5.0), (f64::NAN, 0.0)] {
        for mode in [AngularMotorMode::Velocity, AngularMotorMode::Position] {
            let mut motor = AngularMotorData::new(max_torque);
            motor.mode = mode;
            motor.target_velocity = 10.0;
            motor.target_angle = 10.0;
            let torque = motor.compute_torque(0.0, 0.0, DT);
            assert_eq!(torque, expected, "max_torque {max_torque}, mode {mode:?}");
            assert_eq!(motor.current_torque, expected);
        }
    }
    let mut off = AngularMotorData::new(-1.0);
    assert_eq!(off.compute_torque(1.0, 1.0, DT), 0.0);
}

#[test]
fn velocity_drive_impulse_limit_is_read_as_a_magnitude() {
    // Unclamped impulse: effective_mass * kp * (target - actual) = 2.
    let drive = VelocityDriveConfig::new(2.0, -10.0, 1.0);
    assert_eq!(drive.compute_impulse(1.0, 0.0, 0.1), 1.0);
    assert_eq!(drive.compute_impulse(1.0, 4.0, 0.1), -1.0);
    // A negative time step does not invert the limit either.
    let drive = VelocityDriveConfig::new_linear(2.0, 10.0);
    assert_eq!(drive.compute_impulse(1.0, 0.0, -0.1), 1.0);
    // NaN limits (from NaN force or time step) permit no impulse.
    assert_eq!(drive.compute_impulse(1.0, 0.0, f64::NAN), 0.0);
    let drive = VelocityDriveConfig::new(2.0, f64::NAN, 1.0);
    assert_eq!(drive.compute_impulse(1.0, 0.0, 0.1), 0.0);
    // The ordinary case is unchanged.
    let drive = VelocityDriveConfig::new(2.0, 100.0, 1.0);
    assert_eq!(drive.compute_impulse(1.0, 0.0, 0.1), 2.0);
}

#[test]
fn position_drive_force_limit_is_read_as_a_magnitude() {
    // Unclamped force: kp * (target - x) - kd * v = 10.
    let drive = PositionDriveConfig::new(1.0, 10.0, 0.0, -4.0);
    assert_eq!(drive.compute_force(0.0, 0.0), 4.0);
    assert_eq!(drive.compute_force(2.0, 0.0), -4.0);
    let drive = PositionDriveConfig::new(1.0, 10.0, 0.0, f64::NAN);
    assert_eq!(drive.compute_force(0.0, 0.0), 0.0);
    let drive = PositionDriveConfig::new(1.0, 10.0, 0.0, 40.0);
    assert_eq!(drive.compute_force(0.0, 0.0), 10.0);
}

#[test]
fn friction_impulse_limit_is_read_as_a_magnitude() {
    // The Coulomb limit is mu * lambda_n = -0.5 * 2 = -1 -> 1.
    assert_eq!(clamp_friction_lambda(2.0, -0.5, 10.0), 1.0);
    assert_eq!(clamp_friction_lambda(2.0, -0.5, -10.0), -1.0);
    assert_eq!(clamp_friction_lambda(2.0, f64::NAN, 10.0), 0.0);
    assert_eq!(clamp_friction_lambda(f64::INFINITY, 0.0, 10.0), 0.0);
    assert_eq!(clamp_friction_lambda(2.0, 0.5, 0.25), 0.25);
    // A tension in the normal direction leaves no friction at all.
    assert_eq!(clamp_friction_lambda(-2.0, 0.5, 10.0), 0.0);
}

// ── Two-sided bounds (inverted or NaN) ───────────────────────────────────────

#[test]
fn accumulated_impulse_bounds_may_be_given_in_either_order() {
    // The accumulated impulse 0.5 + 10 is clamped to [-1, 1].
    assert_eq!(clamp_accumulated_impulse(0.5, 10.0, 1.0, -1.0), (1.0, 0.5));
    assert_eq!(clamp_accumulated_impulse(0.5, 10.0, -1.0, 1.0), (1.0, 0.5));
    assert_eq!(
        clamp_accumulated_impulse(0.5, -10.0, 1.0, -1.0),
        (-1.0, -1.5)
    );
    // A NaN bound leaves its side unbounded.
    assert_eq!(
        clamp_accumulated_impulse(0.0, -100.0, f64::NAN, 2.0),
        (-100.0, -100.0)
    );
    assert_eq!(
        clamp_accumulated_impulse(0.0, 100.0, f64::NAN, 2.0),
        (2.0, 2.0)
    );
    assert_eq!(
        clamp_accumulated_impulse(0.0, 100.0, -2.0, f64::NAN),
        (100.0, 100.0)
    );
    // Ordinary bounds behave as before.
    assert_eq!(clamp_accumulated_impulse(0.5, 0.25, 0.0, 1.0), (0.75, 0.25));
}

#[test]
fn single_row_updates_tolerate_inverted_and_nan_bounds() {
    // delta = eff * (-jv + bias) = 3.
    assert_eq!(solve_single_constraint(-3.0, 1.0, 0.0, 0.0, 5.0, 0.0), 3.0);
    assert_eq!(
        solve_single_constraint(-3.0, 1.0, 0.0, 0.0, f64::NAN, 2.0),
        2.0
    );
    assert_eq!(
        solve_single_constraint(-3.0, 1.0, 0.0, 0.0, 0.0, f64::NAN),
        3.0
    );
    assert_eq!(solve_single_constraint(-3.0, 1.0, 0.0, 0.0, 0.0, 5.0), 3.0);
    // SOR scales the update before the bounds apply: 1.5 * 3 = 4.5.
    assert_eq!(
        solve_single_constraint_sor(-3.0, 1.0, 0.0, 0.0, 5.0, 0.0, 1.5),
        4.5
    );
    // A lower bound of 5 with no upper bound lifts the relaxed update 4.5 to 5.
    assert_eq!(
        solve_single_constraint_sor(-3.0, 1.0, 0.0, 0.0, 5.0, f64::NAN, 1.5),
        5.0
    );
}

#[test]
fn compliance_range_clamps_without_panicking() {
    let ordinary = ComplianceRange::new(1e-6, 1e-2);
    assert_eq!(ordinary.clamp(1.0), 1e-2);
    assert_eq!(ordinary.clamp(0.0), 1e-6);
    // The fields are public, so an inverted range can exist in a release build
    // (the constructor only asserts in debug builds).
    let inverted = ComplianceRange { min: 5.0, max: 1.0 };
    assert_eq!(inverted.clamp(3.0), 3.0);
    assert_eq!(inverted.clamp(10.0), 5.0);
    assert_eq!(inverted.clamp(0.1), 1.0);
    let open_below = ComplianceRange {
        min: f64::NAN,
        max: 2.0,
    };
    assert_eq!(open_below.clamp(7.0), 2.0);
    assert_eq!(open_below.clamp(-7.0), -7.0);
}

#[test]
fn lagrange_multiplier_projects_onto_unordered_bounds() {
    let mut inverted = LagrangeMultiplier::with_bounds(1.0, -1.0);
    assert_eq!(inverted.apply_delta(5.0), 1.0);
    assert_eq!(inverted.lambda, 1.0);
    assert_eq!(inverted.apply_delta(-10.0), -2.0);
    assert_eq!(inverted.lambda, -1.0);
    inverted.warm_start(100.0, 0.5);
    assert_eq!(inverted.lambda, 1.0);

    let mut open_below = LagrangeMultiplier::with_bounds(f64::NAN, 2.0);
    assert_eq!(open_below.apply_delta(-100.0), -100.0);
    assert_eq!(open_below.apply_delta(500.0), 102.0);
    assert_eq!(open_below.lambda, 2.0);

    let mut open_above = LagrangeMultiplier::with_bounds(0.0, f64::NAN);
    open_above.warm_start(40.0, 0.5);
    assert_eq!(open_above.lambda, 20.0);
    open_above.warm_start(-40.0, 0.5);
    assert_eq!(open_above.lambda, 0.0);

    // Ordinary bounds behave as before.
    let mut unilateral = LagrangeMultiplier::unilateral();
    assert_eq!(unilateral.apply_delta(-3.0), 0.0);
    assert_eq!(unilateral.apply_delta(3.0), 3.0);
}

fn block_row(lambda_min: f64, lambda_max: f64) -> BlockConstraint6 {
    let jacobian = JacobianRow6 {
        j_a: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        j_b: [0.0; 6],
    };
    BlockConstraint6::new(
        jacobian,
        MassMatrix6::new(1.0, [1.0; 3]),
        MassMatrix6::static_body(),
        0.0,
        lambda_min,
        lambda_max,
    )
}

#[test]
fn block_rows_tolerate_inverted_and_nan_bounds() {
    // Body A moves at -2 m/s along the row: the impulse that stops it is +2.
    let moving = [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let rest = [0.0; 6];
    for (lambda_min, lambda_max) in [(5.0, 0.0), (0.0, 5.0), (0.0, f64::NAN), (f64::NAN, 5.0)] {
        let mut row = block_row(lambda_min, lambda_max);
        let (delta, impulse_a, _) = row.solve_iteration(&moving, &rest);
        assert_eq!(delta, 2.0, "bounds ({lambda_min}, {lambda_max})");
        assert_eq!(row.lambda, 2.0);
        assert_eq!(impulse_a[0], 2.0);

        let mut row = block_row(lambda_min, lambda_max);
        let (delta, _, _) = row.solve_iteration_soft(&moving, &rest, 0.0, &SoftParams::rigid());
        assert_eq!(delta, 2.0, "soft, bounds ({lambda_min}, {lambda_max})");
    }
    // The bounds still bind: [0, 1] only lets an impulse of 1 through.
    let mut row = block_row(1.0, 0.0);
    let (delta, _, _) = row.solve_iteration(&moving, &rest);
    assert_eq!(delta, 1.0);
}

#[test]
fn adaptive_substep_range_may_be_inverted_or_empty() {
    let mut solver = AdaptiveTgsSolver::new(4, Vec3::zeros(), 0.25);
    // 100 m/s over 1 s with 0.25 m per substep wants 400 substeps.
    solver.min_substeps = 8;
    solver.max_substeps = 2;
    assert_eq!(solver.compute_substeps(100.0, 1.0), 8);
    solver.min_substeps = 0;
    solver.max_substeps = 0;
    assert_eq!(solver.compute_substeps(100.0, 1.0), 1);
    solver.min_substeps = 3;
    solver.max_substeps = 64;
    assert_eq!(solver.compute_substeps(100.0, 1.0), 64);
    assert_eq!(solver.compute_substeps(0.0, 1.0), 3);
    assert_eq!(solver.compute_substeps(1.0, 1.0), 4);
}
