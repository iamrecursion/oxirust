// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the audited vehicle dynamics defects F4927 (critical
//! and characteristic speed), F4929 (nonlinear bicycle tire force sign),
//! F4930 (double-track yaw balance), F4931 (J-turn peak lateral acceleration),
//! F4932 / F4928 (steady-state yaw rate and lateral acceleration) and F4937
//! (limited-slip differential torque split).
//!
//! The reference numbers were obtained independently of this crate from the
//! closed-form bicycle model expressions (Gillespie, *Fundamentals of Vehicle
//! Dynamics*, ch. 6) and a separate numerical integration.

use super::*;

/// Relative comparison with a readable failure message.
fn assert_close(actual: f64, expected: f64, rel_tol: f64) {
    assert!(
        (actual - expected).abs() <= rel_tol * expected.abs(),
        "actual {actual} differs from expected {expected} by more than the relative tolerance {rel_tol}"
    );
}

/// Default sedan with overridden axle cornering stiffnesses.
fn sedan(cf: f64, cr: f64) -> LinearBicycleModel {
    LinearBicycleModel {
        cf,
        cr,
        ..LinearBicycleModel::new()
    }
}

/// Understeer-gradient view of the default sedan (cf = 80 kN/rad, cr = 90 kN/rad).
fn understeering_sedan() -> UndersteerGradient {
    UndersteerGradient::new(1500.0, 1.1, 1.6, 80_000.0, 90_000.0)
}

/// Understeer-gradient view of an oversteering sedan (cf = 150 kN/rad, cr = 60 kN/rad).
fn oversteering_sedan() -> UndersteerGradient {
    UndersteerGradient::new(1500.0, 1.1, 1.6, 150_000.0, 60_000.0)
}

/// Determinant of the 2x2 system matrix of the linear bicycle model.
///
/// The trace is always negative, so the yaw dynamics are stable if and only
/// if this determinant is positive.
fn system_determinant(model: &LinearBicycleModel, vx: f64) -> f64 {
    let (a11, a12, a21, a22, _, _) = model.stability_derivatives(vx);
    a11 * a22 - a12 * a21
}

// ─── F4927: critical / characteristic speed ─────────────────────────────────

#[test]
fn f4927_characteristic_speed_of_understeering_sedan() {
    let model = LinearBicycleModel::new();
    // K = 8720 N / 80000 N/rad - 5995 N / 90000 N/rad = 0.042389 rad/g.
    assert_close(model.understeer_gradient(), 0.042_388_9, 1e-6);
    // V_char = sqrt(g L / K) = 25.0 m/s; the inverted g / K formula gave 15.2 m/s.
    assert_close(model.characteristic_speed(), 24.997_143, 1e-6);
    assert_eq!(model.critical_speed(), f64::INFINITY);
}

#[test]
fn f4927_characteristic_speed_doubles_the_ackermann_steer_angle() {
    let ug = understeering_sedan();
    let model = LinearBicycleModel::new();
    let v_char = ug.characteristic_speed();
    let radius = 150.0;
    assert_close(
        ug.ackermann_steer(v_char, radius),
        2.0 * ug.kinematic_steer(radius),
        1e-12,
    );
    // The steady-state yaw-rate gain peaks at V_char with the value V_char / (2 L).
    let gain = |v: f64| model.steady_state_yaw_rate(v, 1.0);
    assert_close(gain(v_char), v_char / (2.0 * 2.7), 1e-12);
    assert!(gain(v_char) > gain(0.9 * v_char));
    assert!(gain(v_char) > gain(1.1 * v_char));
}

#[test]
fn f4927_oversteer_critical_speed_is_where_the_yaw_dynamics_lose_stability() {
    let model = sedan(150_000.0, 60_000.0);
    assert!(model.understeer_gradient() < 0.0);
    let v_crit = model.critical_speed();
    assert_close(v_crit, 25.177_63, 1e-6);
    // Closed form from det(A) = 0: V^2 = cf cr L^2 / (m (cf lf - cr lr)).
    let l = model.lf + model.lr;
    let closed_form =
        model.cf * model.cr * l * l / (model.mass * (model.cf * model.lf - model.cr * model.lr));
    assert_close(v_crit * v_crit, closed_form, 1e-9);
    assert!(system_determinant(&model, 0.99 * v_crit) > 0.0);
    assert!(system_determinant(&model, 1.01 * v_crit) < 0.0);
    assert_eq!(model.characteristic_speed(), f64::INFINITY);
}

#[test]
fn f4927_understeering_model_is_stable_at_every_speed() {
    let model = LinearBicycleModel::new();
    assert!(model.understeer_gradient() > 0.0);
    assert_eq!(model.critical_speed(), f64::INFINITY);
    for vx in [2.0, 10.0, 25.0, 60.0, 150.0] {
        assert!(
            system_determinant(&model, vx) > 0.0,
            "yaw dynamics unstable at {vx} m/s"
        );
    }
}

#[test]
fn f4927_understeer_gradient_struct_speeds_match_linear_model() {
    let ug = understeering_sedan();
    let model = LinearBicycleModel::new();
    assert_close(
        ug.characteristic_speed(),
        model.characteristic_speed(),
        1e-12,
    );
    assert_close(ug.characteristic_speed(), 24.997_143, 1e-6);
    assert!(ug.is_understeering());
    assert_eq!(ug.critical_speed(), f64::INFINITY);

    let ug = oversteering_sedan();
    let model = sedan(150_000.0, 60_000.0);
    assert!(ug.is_oversteering());
    assert_close(ug.critical_speed(), model.critical_speed(), 1e-12);
    assert_close(ug.critical_speed(), 25.177_63, 1e-6);
    assert_eq!(ug.characteristic_speed(), f64::INFINITY);
}

// ─── F4932 / F4928: steady-state yaw rate and lateral acceleration ──────────

#[test]
fn f4932_steady_state_yaw_rate_matches_integrated_step_response() {
    let mut model = LinearBicycleModel::new();
    let (vx, delta) = (20.0, 0.02);
    for _ in 0..10_000 {
        model.step(1.0e-3, vx, delta);
    }
    let expected = model.steady_state_yaw_rate(vx, delta);
    // r = V δ / (L + K V² / g) = 0.4 / (2.7 + 0.042389 · 400 / 9.81) = 0.090326 rad/s;
    // the old formula multiplied the understeer term by L and gave 0.0543 rad/s.
    assert_close(expected, 0.090_326_2, 1e-6);
    assert_close(model.state.r, expected, 1.0e-2);
}

#[test]
fn f4928_steady_state_yaw_rate_matches_twenty_second_step_response() {
    let mut model = LinearBicycleModel::new();
    for _ in 0..20_000 {
        model.step(1.0e-3, 20.0, 0.05);
    }
    // r = V δ / (L + K V² / g) = 1 / 4.4284 = 0.22582 rad/s; the old formula gave 0.13575 rad/s.
    let expected = model.steady_state_yaw_rate(20.0, 0.05);
    assert_close(expected, 0.225_815, 1e-5);
    assert_close(model.state.r, expected, 1.0e-2);
}

#[test]
fn f4932_lateral_acceleration_is_speed_times_steady_state_yaw_rate() {
    let ug = understeering_sedan();
    let model = LinearBicycleModel::new();
    for (vx, delta) in [(10.0, 0.01), (20.0, 0.05), (30.0, -0.02)] {
        assert_close(
            ug.lateral_accel(vx, delta),
            vx * model.steady_state_yaw_rate(vx, delta),
            1e-12,
        );
    }
    // A left steer at 20 m/s used to give -249 m/s² here.
    assert_close(ug.lateral_accel(20.0, 0.05), 4.516_309, 1e-6);
}

#[test]
fn f4932_lateral_accel_inverts_ackermann_steer() {
    for ug in [understeering_sedan(), oversteering_sedan()] {
        for (speed, radius) in [(8.0, 40.0), (20.0, 100.0), (22.0, 250.0), (15.0, -80.0)] {
            let delta = ug.ackermann_steer(speed, radius);
            assert_close(ug.lateral_accel(speed, delta), speed * speed / radius, 1e-9);
        }
    }
}

#[test]
fn f4932_lateral_accel_is_zero_without_steer_or_steady_state() {
    let ug = understeering_sedan();
    assert_eq!(ug.lateral_accel(20.0, 0.0), 0.0);
    // At the critical speed of an oversteering vehicle no finite steady-state turn exists.
    let ug = oversteering_sedan();
    assert_eq!(ug.lateral_accel(ug.critical_speed(), 0.05), 0.0);
}

#[test]
fn f4932_neutral_steer_lateral_accel_is_kinematic() {
    // Equal load-proportional stiffness gives K = 0: a_y = V² δ / L.
    let ug = UndersteerGradient::new(1500.0, 1.1, 1.6, 8_720.0 * 19.0, 5_995.0 * 19.0);
    assert!(ug.understeer_gradient().abs() < 1e-12);
    assert_close(ug.lateral_accel(20.0, 0.027), 400.0 * 0.027 / 2.7, 1e-9);
}

#[test]
fn lateral_acceleration_matches_axle_force_sum() {
    let mut model = LinearBicycleModel::new();
    model.state.v_y = 0.3;
    model.state.r = 0.1;
    let (vx, delta) = (20.0, 0.03);
    // m·a_y = F_yf + F_yr with F_yf = cf (δ - (v_y + lf r)/V), F_yr = -cr (v_y - lr r)/V.
    let fy_front = model.cf * (delta - (0.3 + model.lf * 0.1) / vx);
    let fy_rear = -model.cr * (0.3 - model.lr * 0.1) / vx;
    assert_close(
        model.lateral_acceleration(vx, delta),
        (fy_front + fy_rear) / model.mass,
        1e-12,
    );
}

// ─── F4929: nonlinear bicycle model tire force sign ─────────────────────────

#[test]
fn f4929_left_steer_produces_positive_yaw_rate() {
    let mut model = NonlinearBicycleModel::new();
    for _ in 0..1_000 {
        model.step(1.0e-3, 20.0, 0.02, 0.0);
    }
    // Identical load-proportional tires are neutral steer: r → V δ / L = 0.14815 rad/s.
    assert_close(model.r, 20.0 * 0.02 / 2.7, 1.0e-2);
    // Side slip at the CG is negative for a left turn at this speed (rear axle needs slip angle).
    assert_close(model.vy, -0.091_208, 1.0e-3);
}

#[test]
fn f4929_steering_response_is_antisymmetric() {
    let mut left = NonlinearBicycleModel::new();
    let mut right = NonlinearBicycleModel::new();
    for _ in 0..500 {
        left.step(1.0e-3, 20.0, 0.02, 0.0);
        right.step(1.0e-3, 20.0, -0.02, 0.0);
    }
    assert!(left.r > 0.0 && right.r < 0.0);
    assert_close(left.r, -right.r, 1e-9);
    assert_close(left.vy, -right.vy, 1e-9);
}

#[test]
fn f4929_lateral_velocity_perturbation_decays() {
    let mut model = NonlinearBicycleModel::new();
    model.vy = 0.5;
    let mut vy_after_half_second = 0.0;
    for step in 0..2_000 {
        model.step(1.0e-3, 20.0, 0.0, 0.0);
        if step == 499 {
            vy_after_half_second = model.vy;
        }
    }
    // The wrong tire force sign amplified the perturbation instead.
    assert!(
        vy_after_half_second.abs() < 0.02 * 0.5,
        "{vy_after_half_second}"
    );
    assert!(model.vy.abs() < 1.0e-6, "vy = {}", model.vy);
    assert!(model.r.abs() < 1.0e-6, "r = {}", model.r);
}

#[test]
fn f4929_small_steer_matches_linearised_bicycle_model() {
    let mut nonlinear = NonlinearBicycleModel::new();
    // The linearised cornering stiffness of a Pacejka tire is B·C·D per newton of vertical load.
    let cf =
        nonlinear.tire_f.b * nonlinear.tire_f.c * nonlinear.tire_f.d * nonlinear.front_load(0.0);
    let cr =
        nonlinear.tire_r.b * nonlinear.tire_r.c * nonlinear.tire_r.d * nonlinear.rear_load(0.0);
    let mut linear = LinearBicycleModel {
        mass: nonlinear.mass,
        iz: nonlinear.iz,
        lf: nonlinear.lf,
        lr: nonlinear.lr,
        cf,
        cr,
        state: BicycleState::default(),
    };
    let (vx, delta) = (20.0, 0.005);
    for _ in 0..5_000 {
        nonlinear.step(1.0e-3, vx, delta, 0.0);
        linear.step(1.0e-3, vx, delta);
    }
    let r_steady = linear.steady_state_yaw_rate(vx, delta);
    assert!(r_steady > 0.0);
    assert_close(nonlinear.r, r_steady, 0.05);
    assert_close(nonlinear.r, linear.state.r, 0.05);
}

// ─── F4930: DoubleTrackModel::step ──────────────────────────────────────────

#[test]
fn f4930_torque_vectoring_moment_is_applied_once() {
    let mut model = DoubleTrackModel::new();
    model.set_tv_torque(100.0);
    let dt = 0.01;
    model.step(dt, 0.0, 0.0, 0.0, &[0.0; 4]);
    // dr = tv_torque / Iz exactly; it used to be added twice.
    assert!((model.state.r - 100.0 / model.iz * dt).abs() < 1e-15);
    assert_eq!(model.state.vy, 0.0);
    assert_eq!(model.state.ax_drive, 0.0);
}

#[test]
fn f4930_asymmetric_rear_torque_yaws_the_vehicle() {
    let dt = 1.0e-3;
    let torque = 500.0;
    let mut right_drive = DoubleTrackModel::new();
    right_drive.step(dt, 20.0, 0.0, 0.0, &[0.0, 0.0, 0.0, torque]);
    let fx = torque / right_drive.wheel_radius;
    let expected_r = fx * right_drive.track_r / 2.0 / right_drive.iz * dt;
    // More drive force on the right wheel turns the car to the left (positive yaw).
    assert!(right_drive.state.r > 0.0);
    assert!((right_drive.state.r - expected_r).abs() < 1e-15);
    assert!((right_drive.state.ax_drive - fx / right_drive.mass).abs() < 1e-12);

    let mut left_drive = DoubleTrackModel::new();
    left_drive.step(dt, 20.0, 0.0, 0.0, &[0.0, 0.0, torque, 0.0]);
    assert!((left_drive.state.r + expected_r).abs() < 1e-15);

    let mut symmetric = DoubleTrackModel::new();
    symmetric.step(dt, 20.0, 0.0, 0.0, &[torque; 4]);
    assert!(symmetric.state.r.abs() < 1e-15);
    assert!(symmetric.state.ax_drive > 0.0);
}

#[test]
fn f4930_front_and_rear_torque_differences_add_up() {
    let mut model = DoubleTrackModel::new();
    model.track_f = 1.6;
    model.track_r = 1.4;
    model.wheel_radius = 0.3;
    model.step(1.0e-3, 0.0, 0.0, 0.0, &[100.0, 300.0, 50.0, 250.0]);
    // Mz = [(300 - 100)·1.6 + (250 - 50)·1.4] / (2·0.3) = 1000 N·m.
    assert!((model.state.r - 1000.0 / model.iz * 1.0e-3).abs() < 1e-12);
    // ΣFx / m = (100 + 300 + 50 + 250) / 0.3 / 1500.
    assert!((model.state.ax_drive - 700.0 / 0.3 / 1500.0).abs() < 1e-12);
}

#[test]
fn f4930_braking_torque_decelerates_without_yaw() {
    let mut model = DoubleTrackModel::new();
    model.step(1.0e-3, 20.0, 0.0, 0.0, &[-200.0; 4]);
    let expected = -4.0 * 200.0 / model.wheel_radius / model.mass;
    assert!((model.state.ax_drive - expected).abs() < 1e-12);
    assert!(expected < -1.0);
    assert!(model.state.r.abs() < 1e-15);
}

#[test]
fn f4930_non_positive_wheel_radius_does_not_produce_non_finite_state() {
    let mut model = DoubleTrackModel::new();
    model.wheel_radius = 0.0;
    model.step(1.0e-3, 20.0, 0.0, 0.0, &[10.0, 20.0, 30.0, 40.0]);
    assert!(model.state.r.is_finite());
    assert!(model.state.ax_drive.is_finite());
    assert!(model.state.ax_drive > 0.0);
}

#[test]
fn f4930_lateral_acceleration_sign_selects_the_loaded_side() {
    let vx = 20.0;
    // Left turn: a_y = V·r > 0 loads the right wheels.
    let mut left_turn = DoubleTrackModel::new();
    left_turn.state.r = 0.1;
    left_turn.step(1.0e-3, vx, 0.0, 0.0, &[0.0; 4]);
    assert!(left_turn.state.fy[1].abs() > left_turn.state.fy[0].abs());
    assert!(left_turn.state.fy[3].abs() > left_turn.state.fy[2].abs());

    // Right turn: a_y < 0 loads the left wheels (the old |vy·r| always loaded the right side).
    let mut right_turn = DoubleTrackModel::new();
    right_turn.state.r = -0.1;
    right_turn.step(1.0e-3, vx, 0.0, 0.0, &[0.0; 4]);
    assert!(right_turn.state.fy[0].abs() > right_turn.state.fy[1].abs());
    assert!(right_turn.state.fy[2].abs() > right_turn.state.fy[3].abs());

    // The two manoeuvres are mirror images of each other.
    assert_close(right_turn.state.fy[0], -left_turn.state.fy[1], 1e-9);
    assert_close(right_turn.state.fy[1], -left_turn.state.fy[0], 1e-9);
    assert_close(right_turn.state.fy[2], -left_turn.state.fy[3], 1e-9);
    assert_close(right_turn.state.fy[3], -left_turn.state.fy[2], 1e-9);
}

#[test]
fn f4930_steady_cornering_matches_equivalent_bicycle_model() {
    let mut model = DoubleTrackModel::new();
    let wheel_load = model.mass * 9.81 / 4.0;
    let wb = model.lf + model.lr;
    // Per-axle stiffness implied by the load-scaled tire model without load transfer.
    let cf = 2.0 * model.cf_tire * (model.mass * 9.81 * model.lr / wb / 2.0) / wheel_load;
    let cr = 2.0 * model.cr_tire * (model.mass * 9.81 * model.lf / wb / 2.0) / wheel_load;
    let bicycle = LinearBicycleModel {
        mass: model.mass,
        iz: model.iz,
        lf: model.lf,
        lr: model.lr,
        cf,
        cr,
        state: BicycleState::default(),
    };
    let (vx, delta) = (20.0, 0.005);
    for _ in 0..5_000 {
        model.step(1.0e-3, vx, delta, 0.0, &[0.0; 4]);
    }
    let expected = bicycle.steady_state_yaw_rate(vx, delta);
    assert!(expected > 0.0);
    assert_close(model.state.r, expected, 0.02);
}

// ─── F4931: RolloverDynamics::j_turn_peak_ay ────────────────────────────────

/// J-turn peak lateral acceleration of the default sedan (1500 kg, lf 1.1 m, lr 1.6 m, Iz 2500).
fn j_turn(rollover: &RolloverDynamics, v0: f64, delta: f64, cf: f64, cr: f64) -> f64 {
    rollover.j_turn_peak_ay(v0, delta, cf, cr, 1500.0, 1.1, 1.6, 2500.0)
}

#[test]
fn f4931_j_turn_without_steer_input_gives_zero() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    assert_eq!(j_turn(&rollover, 20.0, 0.0, 80_000.0, 90_000.0), 0.0);
    assert_eq!(j_turn(&rollover, 35.0, 0.0, 150_000.0, 60_000.0), 0.0);
}

#[test]
fn f4931_j_turn_peak_grows_with_steer_input() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    let small = j_turn(&rollover, 20.0, 0.01, 80_000.0, 90_000.0);
    let medium = j_turn(&rollover, 20.0, 0.02, 80_000.0, 90_000.0);
    let large = j_turn(&rollover, 20.0, 0.04, 80_000.0, 90_000.0);
    assert_close(small, 0.909_475, 1e-5);
    assert_close(medium, 1.818_95, 1e-5);
    assert_close(large, 3.637_90, 1e-5);
    assert!(small < medium && medium < large);
    assert!(large < rollover.rollover_threshold_lat_accel());
    // The bicycle model is linear, so the response is proportional to the steer input.
    assert_close(medium, 2.0 * small, 1e-9);
    assert_close(large, 2.0 * medium, 1e-9);
}

#[test]
fn f4931_j_turn_peak_responds_to_tire_stiffness() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    let base = j_turn(&rollover, 20.0, 0.02, 80_000.0, 90_000.0);
    let stiffer_rear = j_turn(&rollover, 20.0, 0.02, 80_000.0, 140_000.0);
    let softer_front = j_turn(&rollover, 20.0, 0.02, 50_000.0, 90_000.0);
    assert_close(stiffer_rear, 1.491_617, 1e-5);
    assert_close(softer_front, 1.152_791, 1e-5);
    assert!((base - stiffer_rear).abs() > 0.1);
    assert!((base - softer_front).abs() > 0.1);
}

#[test]
fn f4931_j_turn_peak_is_not_below_the_steady_state_lateral_acceleration() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    let model = LinearBicycleModel::new();
    let (v0, delta) = (20.0, 0.02);
    let steady_state = v0 * model.steady_state_yaw_rate(v0, delta);
    let peak = j_turn(&rollover, v0, delta, model.cf, model.cr);
    assert_close(steady_state, 1.806_524, 1e-5);
    assert!(peak >= steady_state * (1.0 - 1e-6), "peak {peak}");
    // The default sedan is well damped: only a small overshoot.
    assert!(peak <= 1.05 * steady_state, "peak {peak}");
}

#[test]
fn f4931_j_turn_peak_is_limited_to_the_rollover_threshold() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    let peak = j_turn(&rollover, 20.0, 0.5, 80_000.0, 90_000.0);
    assert_eq!(peak, rollover.rollover_threshold_lat_accel());
}

#[test]
fn f4931_j_turn_of_oversteering_vehicle_diverges_above_critical_speed() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    let critical = sedan(150_000.0, 60_000.0).critical_speed();
    assert_close(critical, 25.177_63, 1e-6);
    let below = j_turn(&rollover, 15.0, 0.005, 150_000.0, 60_000.0);
    let above = j_turn(&rollover, 35.0, 0.005, 150_000.0, 60_000.0);
    assert_close(below, 0.645_905, 1e-5);
    assert!(below < rollover.rollover_threshold_lat_accel());
    assert_eq!(above, rollover.rollover_threshold_lat_accel());
}

#[test]
fn f4931_j_turn_is_symmetric_in_steer_direction() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    let left = j_turn(&rollover, 20.0, 0.02, 80_000.0, 90_000.0);
    let right = j_turn(&rollover, 20.0, -0.02, 80_000.0, 90_000.0);
    assert!(left > 0.0);
    assert_close(left, right, 1e-12);
}

#[test]
fn f4931_j_turn_below_minimum_speed_returns_zero() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    assert_eq!(j_turn(&rollover, 0.05, 0.02, 80_000.0, 90_000.0), 0.0);
    assert_eq!(j_turn(&rollover, 0.0, 0.02, 80_000.0, 90_000.0), 0.0);
    assert_eq!(j_turn(&rollover, f64::NAN, 0.02, 80_000.0, 90_000.0), 0.0);
}

#[test]
fn f4931_j_turn_with_non_finite_steer_is_not_reported_as_a_peak() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    assert_eq!(j_turn(&rollover, 20.0, f64::NAN, 80_000.0, 90_000.0), 0.0);
    // An unbounded steer input is clipped by the rollover limit instead.
    assert_eq!(
        j_turn(&rollover, 20.0, f64::INFINITY, 80_000.0, 90_000.0),
        rollover.rollover_threshold_lat_accel()
    );
}

#[test]
fn f4931_j_turn_of_stiff_low_speed_case_is_integrated_stably() {
    let rollover = RolloverDynamics::new(1.5, 0.5, 1500.0);
    // With a fixed 1 ms explicit Euler step this configuration is numerically unstable
    // (|λ|·dt > 2) and would blow up to the rollover cap. The physical response peaks at
    // the instantaneous front axle force cf·δ/m and then decays.
    let peak = j_turn(&rollover, 0.5, 0.01, 1.0e6, 1.0e6);
    assert_close(peak, 1.0e6 * 0.01 / 1500.0, 1e-9);
    assert!(peak < rollover.rollover_threshold_lat_accel());
}

// ─── F4937: limited-slip differential ───────────────────────────────────────

#[test]
fn f4937_lsd_moves_torque_to_the_slower_wheel() {
    let mut lsd = WheelSpinControl::lsd(3.0);
    lsd.wheel_speeds = [10.0, 15.0];
    let t = lsd.distribute(200.0, 0.0);
    assert!((t[0] - 150.0).abs() < 1e-9, "slower left wheel: {}", t[0]);
    assert!((t[1] - 50.0).abs() < 1e-9, "faster right wheel: {}", t[1]);
    assert!((t[0] / t[1] - 3.0).abs() < 1e-9);

    // Mirrored: the left wheel is faster, so the right wheel gets the torque.
    lsd.wheel_speeds = [15.0, 10.0];
    let t = lsd.distribute(200.0, 0.0);
    assert!((t[0] - 50.0).abs() < 1e-9);
    assert!((t[1] - 150.0).abs() < 1e-9);
}

#[test]
fn f4937_lsd_keeps_negative_input_torque_negative() {
    let mut lsd = WheelSpinControl::lsd(3.0);
    lsd.wheel_speeds = [10.0, 15.0];
    let t = lsd.distribute(-200.0, 0.0);
    // The locking torque still acts from the faster to the slower wheel, so the faster
    // wheel gets the larger braking torque; neither wheel changes sign.
    assert!((t[0] + 50.0).abs() < 1e-9, "left: {}", t[0]);
    assert!((t[1] + 150.0).abs() < 1e-9, "right: {}", t[1]);
    assert!(t[0] <= 0.0 && t[1] <= 0.0);
    assert!((t[1] / t[0] - 3.0).abs() < 1e-9);
}

#[test]
fn f4937_lsd_transfer_is_proportional_below_the_bias_limit() {
    let mut lsd = WheelSpinControl::lsd(3.0);
    lsd.wheel_speeds = [10.0, 10.04];
    // 500 N·m·s/rad · 0.04 rad/s = 20 N·m, well below the 50 N·m bias limit.
    let t = lsd.distribute(200.0, 0.0);
    assert!((t[0] - 120.0).abs() < 1e-9);
    assert!((t[1] - 80.0).abs() < 1e-9);

    lsd.lsd_lock_gain = 1_000.0;
    let t = lsd.distribute(200.0, 0.0);
    assert!((t[0] - 140.0).abs() < 1e-9);
    assert!((t[1] - 60.0).abs() < 1e-9);

    lsd.lsd_lock_gain = 0.0;
    let t = lsd.distribute(200.0, 0.0);
    assert!((t[0] - 100.0).abs() < 1e-9 && (t[1] - 100.0).abs() < 1e-9);
}

#[test]
fn f4937_lsd_equal_speeds_split_the_torque_evenly() {
    let mut lsd = WheelSpinControl::lsd(4.0);
    lsd.wheel_speeds = [30.0, 30.0];
    let t = lsd.distribute(240.0, 0.0);
    assert!((t[0] - 120.0).abs() < 1e-12 && (t[1] - 120.0).abs() < 1e-12);
    assert!(lsd.speed_error() < 1e-12);
}

#[test]
fn f4937_lsd_conserves_torque_and_respects_the_bias_ratio() {
    for bias in [1.5, 2.0, 3.0, 5.0] {
        for input in [-300.0, -40.0, 40.0, 300.0] {
            for speeds in [[10.0, 20.0], [20.0, 10.0]] {
                let mut lsd = WheelSpinControl::lsd(bias);
                lsd.wheel_speeds = speeds;
                let t = lsd.distribute(input, 0.0);
                assert!((t[0] + t[1] - input).abs() < 1e-9);
                let (high, low) = (t[0].abs().max(t[1].abs()), t[0].abs().min(t[1].abs()));
                assert!(
                    high <= bias * low + 1e-9,
                    "bias {bias}, input {input}: {t:?}"
                );
                assert!(t[0] * input >= 0.0 && t[1] * input >= 0.0);
            }
        }
    }
}

#[test]
fn f4937_lsd_bias_below_one_behaves_like_an_open_differential() {
    for bias in [1.0, 0.5, -1.0, f64::NAN] {
        let mut lsd = WheelSpinControl::lsd(bias);
        lsd.wheel_speeds = [10.0, 15.0];
        let t = lsd.distribute(200.0, 0.0);
        assert!((t[0] - 100.0).abs() < 1e-12 && (t[1] - 100.0).abs() < 1e-12);
    }
}

#[test]
fn f4937_default_lock_gain_is_500() {
    assert_eq!(WheelSpinControl::open().lsd_lock_gain, 500.0);
    assert_eq!(WheelSpinControl::lsd(2.0).lsd_lock_gain, 500.0);
    assert_eq!(
        WheelSpinControl::torque_vectoring(300.0).lsd_lock_gain,
        500.0
    );
}
