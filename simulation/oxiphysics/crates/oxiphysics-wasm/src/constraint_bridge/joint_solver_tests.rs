// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the engine-coupled `WasmConstraintSolver`.

use super::{WORLD_BODY, WasmConstraintSolver, WasmJointConfig, WasmJointHandle, WasmMotorTarget};
use crate::engine::WasmPhysicsEngine;
use crate::error::Error;

const DT: f64 = 1.0 / 60.0;

fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    // v' = v + 2w (u x v) + 2 u x (u x v)
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let c1 = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let c2 = [
        u[1] * c1[2] - u[2] * c1[1],
        u[2] * c1[0] - u[0] * c1[2],
        u[0] * c1[1] - u[1] * c1[0],
    ];
    [
        v[0] + 2.0 * (w * c1[0] + c2[0]),
        v[1] + 2.0 * (w * c1[1] + c2[1]),
        v[2] + 2.0 * (w * c1[2] + c2[2]),
    ]
}

fn anchor_world(engine: &WasmPhysicsEngine, body: u32, local: [f64; 3]) -> [f64; 3] {
    let p = engine.get_position(body);
    let r = rotate(engine.get_rotation(body), local);
    [p[0] + r[0], p[1] + r[1], p[2] + r[2]]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Dynamic sphere body (the collider gives it rotational inertia).
fn sphere_body(engine: &mut WasmPhysicsEngine, mass: f64, pos: [f64; 3]) -> u32 {
    let body = engine
        .add_dynamic_body(mass, pos[0], pos[1], pos[2])
        .expect("add body");
    engine.add_sphere_collider(body, 0.1).expect("add collider");
    body
}

#[test]
fn ball_joint_pendulum_keeps_anchor_distance() {
    let mut engine = WasmPhysicsEngine::new(0.0, -9.81, 0.0);
    let bob = sphere_body(&mut engine, 1.0, [1.0, 0.0, 0.0]);
    let mut solver = WasmConstraintSolver::new(10, true);
    solver.add_joint(WasmJointConfig {
        joint_type: "ball".to_string(),
        body_a: WORLD_BODY,
        body_b: u64::from(bob),
        anchor_a: [0.0; 3],
        anchor_b: [-1.0, 0.0, 0.0],
        ..Default::default()
    });
    let mut lowest = 0.0_f64;
    let mut far_side = 0.0_f64;
    let mut worst = 0.0_f64;
    for _ in 0..600 {
        solver.step(&mut engine, DT).expect("step");
        let pivot = anchor_world(&engine, bob, [-1.0, 0.0, 0.0]);
        worst = worst.max(distance(pivot, [0.0; 3]));
        lowest = lowest.min(engine.get_position(bob)[1]);
        far_side = far_side.min(engine.get_position(bob)[0]);
    }
    assert!(worst < 1e-6, "anchor separation {worst}");
    // Released horizontally it swings through the bottom to the far side.
    assert!(far_side < -0.9, "swing reaches x = {far_side}");
    assert!(
        lowest < -0.9,
        "pendulum must swing down, lowest y = {lowest}"
    );
    let state = solver
        .get_joint_state(WasmJointHandle::new(1))
        .expect("state");
    assert!(state.constraint_force > 0.0);
}

#[test]
fn revolute_position_motor_converges() {
    let mut engine = WasmPhysicsEngine::new(0.0, 0.0, 0.0);
    let body = engine
        .add_dynamic_body(2.0, 0.0, 0.0, 0.0)
        .expect("add body");
    engine
        .add_box_collider(body, 0.5, 0.1, 0.2)
        .expect("add collider");
    let mut solver = WasmConstraintSolver::new(20, true);
    let mut cfg = WasmJointConfig::revolute(WORLD_BODY, u64::from(body), [0.0; 3], [0.0, 1.0, 0.0]);
    cfg.motor_max_force = 50.0;
    let h = solver.add_joint(cfg);
    solver
        .set_motor_target(h, WasmMotorTarget::TargetPosition(0.5))
        .expect("set target");
    for _ in 0..240 {
        solver.step(&mut engine, DT).expect("step");
    }
    let state = solver.get_joint_state(h).expect("state");
    assert!(
        (state.position - 0.5).abs() < 1e-3,
        "angle {} should reach 0.5",
        state.position
    );
    // The body really rotated about +Y by the reported angle.
    let q = engine.get_rotation(body);
    let angle = 2.0 * q[1].atan2(q[3]);
    assert!((angle - 0.5).abs() < 2e-3, "body angle {angle}");
    assert!(q[0].abs() < 1e-6 && q[2].abs() < 1e-6, "axis stays aligned");
    assert!(distance(engine.get_position(body), [0.0; 3]) < 1e-6);
}

#[test]
fn position_target_does_not_overwrite_velocity_target() {
    let mut solver = WasmConstraintSolver::new(10, false);
    let h = solver.add_joint(WasmJointConfig::revolute(0, 1, [0.0; 3], [0.0, 1.0, 0.0]));
    solver
        .set_motor_target(h, WasmMotorTarget::TargetVelocity(3.0))
        .expect("velocity");
    solver
        .set_motor_target(h, WasmMotorTarget::TargetPosition(0.25))
        .expect("position");
    let cfg = solver.get_joint_config(h).expect("config");
    assert!(cfg.motor_enabled);
    assert!((cfg.motor_target_velocity - 3.0).abs() < 1e-15);
    assert_eq!(cfg.motor_target_position, Some(0.25));
    solver
        .set_motor_target(h, WasmMotorTarget::TargetVelocity(-1.0))
        .expect("velocity again");
    let cfg = solver.get_joint_config(h).expect("config");
    assert_eq!(cfg.motor_target_position, None);
    assert!(matches!(
        solver.set_motor_target(WasmJointHandle::new(77), WasmMotorTarget::MaxTorque(1.0)),
        Err(Error::ConstraintNotFound(77))
    ));
    assert!(
        solver
            .set_motor_target(h, WasmMotorTarget::MaxTorque(-1.0))
            .is_err()
    );
}

#[test]
fn revolute_velocity_motor_respects_torque_cap() {
    let mut engine = WasmPhysicsEngine::new(0.0, 0.0, 0.0);
    let body = engine.add_dynamic_body(1.0, 0.0, 0.0, 0.0).expect("body");
    engine.add_sphere_collider(body, 0.5).expect("collider");
    let mut solver = WasmConstraintSolver::new(10, true);
    let mut cfg = WasmJointConfig::revolute(WORLD_BODY, u64::from(body), [0.0; 3], [0.0, 0.0, 1.0]);
    cfg.motor_enabled = true;
    cfg.motor_target_velocity = 2.0;
    cfg.motor_max_force = 0.1; // I = 0.4 * 1 * 0.25 = 0.1 -> at most 1 rad/s^2
    let h = solver.add_joint(cfg);
    solver.step(&mut engine, DT).expect("step");
    // The engine's default angular damping (0.05 1/s) acts after the solve.
    let damping = 1.0 - 0.05 * DT;
    let w = engine.get_angular_velocity(body)[2];
    assert!(
        (w - DT * damping).abs() < 1e-9,
        "one capped step must give dw = tau/I*dt, got {w}"
    );
    for _ in 0..300 {
        solver.step(&mut engine, DT).expect("step");
    }
    let w = engine.get_angular_velocity(body)[2];
    assert!(
        (w - 2.0 * damping).abs() < 1e-6,
        "motor must reach 2 rad/s, got {w}"
    );
    let state = solver.get_joint_state(h).expect("state");
    assert!((state.velocity - 2.0).abs() < 1e-6);
}

#[test]
fn revolute_limit_stops_motor() {
    let mut engine = WasmPhysicsEngine::new(0.0, 0.0, 0.0);
    let body = engine.add_dynamic_body(1.0, 0.0, 0.0, 0.0).expect("body");
    engine.add_sphere_collider(body, 0.5).expect("collider");
    let mut solver = WasmConstraintSolver::new(20, true);
    let mut cfg = WasmJointConfig::revolute(WORLD_BODY, u64::from(body), [0.0; 3], [0.0, 1.0, 0.0]);
    cfg.motor_enabled = true;
    cfg.motor_target_velocity = 3.0;
    cfg.motor_max_force = 5.0;
    cfg.limits_enabled = true;
    cfg.lower_limit = -0.3;
    cfg.upper_limit = 0.3;
    let h = solver.add_joint(cfg);
    let mut max_angle = f64::NEG_INFINITY;
    for _ in 0..240 {
        solver.step(&mut engine, DT).expect("step");
        let state = solver.get_joint_state(h).expect("state");
        max_angle = max_angle.max(state.position);
    }
    let state = solver.get_joint_state(h).expect("state");
    assert!(state.limit_active);
    assert!(max_angle < 0.3 + 0.06, "angle overshoot {max_angle}");
    assert!((state.position - 0.3).abs() < 0.06, "rests at the limit");
}

#[test]
fn prismatic_joint_constrains_to_axis_and_limit() {
    let mut engine = WasmPhysicsEngine::new(0.0, 0.0, 0.0);
    let body = engine.add_dynamic_body(1.0, 0.0, 0.0, 0.0).expect("body");
    engine
        .add_box_collider(body, 0.2, 0.2, 0.2)
        .expect("collider");
    engine.set_velocity(body, 1.0, 1.0, 0.5).expect("velocity");
    engine
        .set_angular_velocity(body, 0.3, 0.0, 0.0)
        .expect("spin");
    let mut solver = WasmConstraintSolver::new(20, true);
    let mut cfg =
        WasmJointConfig::prismatic(WORLD_BODY, u64::from(body), [0.0; 3], [1.0, 0.0, 0.0]);
    cfg.limits_enabled = true;
    cfg.lower_limit = -1.0;
    cfg.upper_limit = 0.5;
    let h = solver.add_joint(cfg);
    for _ in 0..120 {
        solver.step(&mut engine, DT).expect("step");
    }
    let p = engine.get_position(body);
    assert!(
        p[1].abs() < 1e-3 && p[2].abs() < 1e-3,
        "off-axis drift {p:?}"
    );
    assert!(
        p[0] > 0.4 && p[0] < 0.52,
        "slides to the upper limit, x = {}",
        p[0]
    );
    let q = engine.get_rotation(body);
    assert!((q[3].abs() - 1.0).abs() < 1e-4, "rotation locked: {q:?}");
    let state = solver.get_joint_state(h).expect("state");
    assert!((state.position - p[0]).abs() < 1e-9);
}

#[test]
fn spring_joint_settles_at_rest_length() {
    let mut engine = WasmPhysicsEngine::new(0.0, 0.0, 0.0);
    let body = sphere_body(&mut engine, 1.0, [2.0, 0.0, 0.0]);
    let mut solver = WasmConstraintSolver::new(10, true);
    let mut cfg =
        WasmJointConfig::spring(WORLD_BODY, u64::from(body), [0.0; 3], [0.0; 3], 100.0, 5.0);
    cfg.spring_rest_length = 1.0;
    let h = solver.add_joint(cfg);
    let mut min_len = f64::INFINITY;
    for _ in 0..600 {
        solver.step(&mut engine, DT).expect("step");
        min_len = min_len.min(engine.get_position(body)[0]);
    }
    // Under-damped (zeta = 0.25): it overshoots and then settles.
    assert!(min_len < 1.0, "spring must overshoot, min length {min_len}");
    let x = engine.get_position(body)[0];
    assert!((x - 1.0).abs() < 1e-3, "settles at rest length, x = {x}");
    let state = solver.get_joint_state(h).expect("state");
    assert!((state.position - x).abs() < 1e-9);
}

#[test]
fn fixed_joint_holds_body_against_gravity() {
    let mut engine = WasmPhysicsEngine::new(0.0, -9.81, 0.0);
    let body = engine.add_dynamic_body(3.0, 0.0, 1.0, 0.0).expect("body");
    engine
        .add_box_collider(body, 0.5, 0.2, 0.2)
        .expect("collider");
    let mut solver = WasmConstraintSolver::new(10, true);
    // Weld at an off-centre anchor: gravity also produces a torque.
    solver.add_joint(WasmJointConfig {
        joint_type: "fixed".to_string(),
        body_a: WORLD_BODY,
        body_b: u64::from(body),
        anchor_a: [-0.5, 1.0, 0.0],
        anchor_b: [-0.5, 0.0, 0.0],
        ..Default::default()
    });
    for _ in 0..300 {
        solver.step(&mut engine, DT).expect("step");
    }
    let p = engine.get_position(body);
    assert!(distance(p, [0.0, 1.0, 0.0]) < 1e-3, "position drift {p:?}");
    let q = engine.get_rotation(body);
    assert!((q[3].abs() - 1.0).abs() < 1e-5, "orientation drift {q:?}");
}

#[test]
fn chain_reports_actual_sweep_count_and_converges() {
    let mut engine = WasmPhysicsEngine::new(0.0, -9.81, 0.0);
    let a = sphere_body(&mut engine, 1.0, [1.0, 0.0, 0.0]);
    let b = sphere_body(&mut engine, 1.0, [2.0, 0.0, 0.0]);
    let mut solver = WasmConstraintSolver::new(50, false);
    solver.add_joint(WasmJointConfig::ball(WORLD_BODY, u64::from(a), [0.0; 3]));
    let mut link = WasmJointConfig::ball(u64::from(a), u64::from(b), [0.5, 0.0, 0.0]);
    link.anchor_b = [-0.5, 0.0, 0.0];
    solver.add_joint(link);
    // With a world anchor at the origin the first joint's anchor on `a` is
    // its local origin offset: place it at the pivot.
    solver.joints[0].1.anchor_b = [-1.0, 0.0, 0.0];
    let used = solver.solve(&mut engine, DT).expect("solve");
    assert_eq!(used, solver.iterations_used);
    assert!(used > 1, "a coupled chain needs several sweeps");
    assert!(used < 50, "converged before the cap, used {used}");
    // Zero sweeps allowed -> nothing solved.
    let mut idle = WasmConstraintSolver::new(0, false);
    idle.add_joint(WasmJointConfig::ball(WORLD_BODY, u64::from(a), [0.0; 3]));
    assert_eq!(idle.solve(&mut engine, DT).expect("solve"), 0);
}

#[test]
fn warm_start_reduces_sweeps_in_steady_state() {
    let run = |warm: bool| -> u32 {
        let mut engine = WasmPhysicsEngine::new(0.0, -9.81, 0.0);
        let a = sphere_body(&mut engine, 1.0, [0.0, -1.0, 0.0]);
        let b = sphere_body(&mut engine, 1.0, [0.0, -2.0, 0.0]);
        let mut solver = WasmConstraintSolver::new(100, warm);
        let mut j0 = WasmJointConfig::ball(WORLD_BODY, u64::from(a), [0.0; 3]);
        j0.anchor_b = [0.0, 1.0, 0.0];
        solver.add_joint(j0);
        let mut j1 = WasmJointConfig::ball(u64::from(a), u64::from(b), [0.0, -0.5, 0.0]);
        j1.anchor_b = [0.0, 0.5, 0.0];
        solver.add_joint(j1);
        let mut used = 0;
        for _ in 0..30 {
            used = solver.step(&mut engine, DT).expect("step");
        }
        used
    };
    let cold = run(false);
    let warm = run(true);
    assert!(warm < cold, "warm start {warm} sweeps vs cold {cold}");
}

#[test]
fn invalid_joints_are_rejected_without_touching_bodies() {
    let mut engine = WasmPhysicsEngine::new(0.0, 0.0, 0.0);
    let body = sphere_body(&mut engine, 1.0, [0.0; 3]);
    engine.set_velocity(body, 1.0, 0.0, 0.0).expect("velocity");

    let mut solver = WasmConstraintSolver::new(10, false);
    solver.add_joint(WasmJointConfig::ball(WORLD_BODY, u64::from(body), [0.0; 3]));
    solver.add_joint(WasmJointConfig {
        joint_type: "hinge".to_string(),
        body_a: WORLD_BODY,
        body_b: u64::from(body),
        ..Default::default()
    });
    assert!(matches!(
        solver.solve(&mut engine, DT),
        Err(Error::InvalidParameter { .. })
    ));
    assert_eq!(engine.get_velocity(body), [1.0, 0.0, 0.0]);

    let mut missing = WasmConstraintSolver::new(10, false);
    missing.add_joint(WasmJointConfig::ball(WORLD_BODY, 4242, [0.0; 3]));
    assert!(matches!(
        missing.solve(&mut engine, DT),
        Err(Error::BodyNotFound(4242))
    ));

    let mut same = WasmConstraintSolver::new(10, false);
    same.add_joint(WasmJointConfig::ball(
        u64::from(body),
        u64::from(body),
        [0.0; 3],
    ));
    assert!(same.solve(&mut engine, DT).is_err());

    let mut zero_axis = WasmConstraintSolver::new(10, false);
    zero_axis.add_joint(WasmJointConfig::revolute(
        WORLD_BODY,
        u64::from(body),
        [0.0; 3],
        [0.0; 3],
    ));
    assert!(zero_axis.solve(&mut engine, DT).is_err());

    assert!(matches!(
        solver.solve(&mut engine, 0.0),
        Err(Error::InvalidTimeStep(_))
    ));
}
