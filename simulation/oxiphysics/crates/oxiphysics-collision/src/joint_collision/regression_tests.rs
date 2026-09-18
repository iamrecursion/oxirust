// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for [`JointLimitSolver`].
//!
//! The old `solve` never changed a constraint's position or velocity, so every
//! iteration recomputed the same penalty impulse: the result grew linearly with
//! the iteration count, the accumulated impulse was stored and added again on the
//! next frame (unbounded growth), `compliance` was ignored and every joint
//! between the same two bodies shared one warm-start key.
//!
//! All numbers below follow from the one-dimensional velocity problem
//! `s * v >= bias` with `bias = erp * penetration / dt` and the inverse effective
//! mass `K = im_a + im_b`: the converged impulse is `(bias - s * v0) / K`.

use super::*;

const DT: f64 = 0.01;
/// Limits are `[-1, 1]` in every test; the rigid default has `erp = 0.2`, so a
/// penetration of `p` asks for the velocity `0.2 * p / 0.01 = 20 p`.
const RIGID_BIAS_PER_PENETRATION: f64 = 20.0;

fn limit(
    body_a: usize,
    body_b: usize,
    position: f64,
    velocity: f64,
    stop: StopType,
) -> JointLimitConstraint {
    let mut c = JointLimitConstraint::new(
        body_a,
        body_b,
        JointType::Prismatic,
        JointLimits::new(-1.0, 1.0),
        stop,
        [1.0, 0.0, 0.0],
    );
    c.position = position;
    c.velocity = velocity;
    c
}

/// `+1` below the lower limit, `-1` above the upper one.
fn side(c: &JointLimitConstraint) -> f64 {
    if c.position < c.limits.lower {
        1.0
    } else {
        -1.0
    }
}

fn solver_with(iterations: usize, constraints: Vec<JointLimitConstraint>) -> JointLimitSolver {
    let mut s = JointLimitSolver::new(iterations);
    for c in constraints {
        s.add_constraint(c);
    }
    s
}

fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: expected {expected}, got {actual}"
    );
}

#[test]
fn total_impulse_does_not_depend_on_the_iteration_count() {
    // Penetration 0.5 -> bias 10; the joint moves further out at +3, i.e.
    // s * v = -3, so the impulse is (10 + 3) / 2 = 6.5.
    for iterations in [1, 4, 40] {
        let mut s = solver_with(iterations, vec![limit(0, 1, 1.5, 3.0, StopType::Hard)]);
        let total = s.solve(&[1.0, 1.0], DT);
        assert_close(
            total,
            6.5,
            1e-12,
            &format!("total with {iterations} iterations"),
        );
        assert_close(
            s.constraints[0].warm_impulse,
            6.5,
            1e-12,
            "accumulated impulse",
        );
    }
}

#[test]
fn solved_velocity_reaches_the_bias_on_both_sides() {
    for (position, velocity) in [(1.5, 3.0), (-1.5, -3.0), (1.25, 0.0), (-1.25, 7.0)] {
        let mut s = solver_with(8, vec![limit(0, 1, position, velocity, StopType::Hard)]);
        s.solve(&[1.0, 1.0], DT);
        let c = &s.constraints[0];
        let penetration = c.limits.penetration(position);
        let bias = RIGID_BIAS_PER_PENETRATION * penetration;
        assert!(
            side(c) * c.velocity >= bias - 1e-9,
            "position {position}: s*v = {} below the bias {bias}",
            side(c) * c.velocity
        );
    }
    // A joint that is already moving back faster than the bias keeps its speed,
    // an equality is only reached when the limit has to push.
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, 3.0, StopType::Hard)]);
    s.solve(&[1.0, 1.0], DT);
    assert_close(-s.constraints[0].velocity, 10.0, 1e-9, "upper limit s*v");
    let mut s = solver_with(4, vec![limit(0, 1, -1.5, -3.0, StopType::Hard)]);
    s.solve(&[1.0, 1.0], DT);
    assert_close(s.constraints[0].velocity, 10.0, 1e-9, "lower limit s*v");
}

#[test]
fn a_joint_leaving_its_limit_fast_is_not_pulled_back() {
    // s * v = 50 already exceeds the bias 10: a limit only pushes.
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, -50.0, StopType::Hard)]);
    let total = s.solve(&[1.0, 1.0], DT);
    assert_eq!(total, 0.0);
    assert_eq!(s.constraints[0].velocity, -50.0);
    assert_eq!(s.constraints[0].warm_impulse, 0.0);
    assert!(s.warm_start.is_empty());
}

#[test]
fn a_resting_limit_does_not_accumulate_impulse_from_frame_to_frame() {
    // The old solver stored its (iteration-scaled) total and added it again the
    // next frame. Fresh constraints each frame:
    let mut s = JointLimitSolver::new(4);
    for frame in 0..12 {
        s.clear_constraints();
        s.add_constraint(limit(0, 1, 1.5, 3.0, StopType::Hard));
        let total = s.solve(&[1.0, 1.0], DT);
        assert_close(total, 6.5, 1e-12, &format!("frame {frame}"));
        assert_eq!(s.warm_start.len(), 1);
    }
    // The same constraint object, its velocity refreshed by the integrator:
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, 3.0, StopType::Hard)]);
    for frame in 0..12 {
        s.constraints[0].velocity = 3.0;
        let total = s.solve(&[1.0, 1.0], DT);
        assert_close(total, 6.5, 1e-12, &format!("persistent frame {frame}"));
    }
}

#[test]
fn limits_between_the_same_bodies_keep_separate_warm_impulses() {
    let frame = || {
        vec![
            // Lower limit: bias 10, impulse (10 + 3) / 2 = 6.5.
            limit(0, 1, -1.5, -3.0, StopType::Hard),
            // Upper limit: bias 4, impulse 4 / 2 = 2.
            limit(0, 1, 1.2, 0.0, StopType::Hard),
        ]
    };
    let mut s = solver_with(4, frame());
    let first = s.solve(&[1.0, 1.0], DT);
    assert_close(first, 8.5, 1e-12, "first frame");
    assert_eq!(s.warm_start.len(), 2, "one cache entry per limit");

    s.clear_constraints();
    for c in frame() {
        s.add_constraint(c);
    }
    s.apply_warm_starts();
    assert_close(
        s.constraints[0].warm_impulse,
        6.5,
        1e-12,
        "lower limit warm start",
    );
    assert_close(
        s.constraints[1].warm_impulse,
        2.0,
        1e-12,
        "upper limit warm start",
    );
    let second = s.solve(&[1.0, 1.0], DT);
    assert_close(second, 8.5, 1e-12, "second frame");
}

#[test]
fn a_cached_impulse_is_dropped_when_the_joint_switches_limits() {
    let mut s = solver_with(4, vec![limit(0, 1, -1.5, -3.0, StopType::Hard)]);
    s.solve(&[1.0, 1.0], DT);
    assert_eq!(s.warm_start.len(), 1);

    // Same body pair and slot, now at the upper limit.
    s.clear_constraints();
    s.add_constraint(limit(0, 1, 1.5, 3.0, StopType::Hard));
    s.apply_warm_starts();
    assert_eq!(s.constraints[0].warm_impulse, 0.0);

    // Back on the lower side the cached impulse is found again.
    s.clear_constraints();
    s.add_constraint(limit(0, 1, -1.5, -3.0, StopType::Hard));
    s.apply_warm_starts();
    assert_close(s.constraints[0].warm_impulse, 6.5, 1e-12, "lower side");
}

#[test]
fn inactive_constraints_clear_their_warm_impulse_and_the_cache() {
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, 3.0, StopType::Hard)]);
    s.solve(&[1.0, 1.0], DT);
    assert_eq!(s.warm_start.len(), 1);

    // The joint comes back inside its limits.
    s.constraints[0].position = 0.0;
    s.constraints[0].warm_impulse = 5.0;
    let total = s.solve(&[1.0, 1.0], DT);
    assert_eq!(total, 0.0);
    assert_eq!(s.constraints[0].warm_impulse, 0.0);
    assert!(s.warm_start.is_empty());
}

#[test]
fn the_solver_wide_compliance_sets_the_correction_speed() {
    // erp = 1: the whole penetration (0.5) is corrected in one step, bias 50.
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, 0.0, StopType::Hard)]);
    s.compliance = JointComplianceParams::new(1.0, 0.0);
    let total = s.solve(&[1.0, 1.0], DT);
    assert_close(-s.constraints[0].velocity, 50.0, 1e-9, "s*v");
    assert_close(total, 25.0, 1e-12, "impulse");

    // erp = 0 asks for no position correction: only the approach is stopped.
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, 3.0, StopType::Hard)]);
    s.compliance = JointComplianceParams::new(0.0, 0.0);
    s.solve(&[1.0, 1.0], DT);
    assert_close(s.constraints[0].velocity, 0.0, 1e-9, "velocity stopped");
}

#[test]
fn constraint_force_mixing_softens_the_limit() {
    // erp 0.5, cfm 0.25: bias = 25, K = 2, K + cfm = 2.25 and the limit holds
    // s * v + cfm * lambda = bias.
    let totals: Vec<f64> = [1, 30]
        .iter()
        .map(|&iterations| {
            let mut s = solver_with(iterations, vec![limit(0, 1, 1.5, 0.0, StopType::Hard)]);
            s.compliance = JointComplianceParams::new(0.5, 0.25);
            let total = s.solve(&[1.0, 1.0], DT);
            let c = &s.constraints[0];
            let sv = side(c) * c.velocity;
            assert_close(sv + 0.25 * total, 25.0, 1e-9, "softened limit relation");
            assert!(sv < 25.0, "a soft limit gives way: s*v = {sv}");
            total
        })
        .collect();
    assert_close(totals[0], 25.0 / 2.25, 1e-12, "impulse");
    assert_close(totals[0], totals[1], 1e-12, "iteration independence");
}

#[test]
fn soft_stops_are_implicit_spring_dampers() {
    // stiffness 400 N/m, damping 4 N s/m, h = 0.01, K = 0.5 + 1.5 = 2.
    // Backward Euler: lambda = h (k (pen - h v+) - c v+) with v+ = K lambda
    //   => lambda = h k pen / (1 + h (h k + c) K).
    let (stiffness, damping, penetration) = (400.0, 4.0, 0.5);
    let expected = DT * stiffness * penetration / (1.0 + DT * (DT * stiffness + damping) * 2.0);
    for compliance in [
        JointComplianceParams::rigid(),
        JointComplianceParams::new(1.0, 5.0),
    ] {
        let mut s = solver_with(
            6,
            vec![limit(0, 1, 1.5, 0.0, StopType::Soft { stiffness, damping })],
        );
        s.compliance = compliance;
        let total = s.solve(&[0.5, 1.5], DT);
        assert_close(total, expected, 1e-12, "spring-damper impulse");
        assert_close(
            -s.constraints[0].velocity,
            2.0 * expected,
            1e-12,
            "velocity",
        );
    }
    // The damper resists the approach: a faster approach (s * v < 0) needs more.
    let mut s = solver_with(
        6,
        vec![limit(0, 1, 1.5, 2.0, StopType::Soft { stiffness, damping })],
    );
    let faster = s.solve(&[0.5, 1.5], DT);
    assert!(faster > expected, "{faster} <= {expected}");
}

#[test]
fn stiffer_soft_stops_push_harder_and_approach_the_rigid_limit() {
    let impulse = |stiffness: f64| {
        let mut s = solver_with(
            4,
            vec![limit(
                0,
                1,
                1.5,
                0.0,
                StopType::Soft {
                    stiffness,
                    damping: 0.0,
                },
            )],
        );
        s.solve(&[1.0, 1.0], DT)
    };
    let (weak, medium, strong, huge) = (impulse(10.0), impulse(100.0), impulse(1e4), impulse(1e9));
    assert!(weak > 0.0 && weak < medium && medium < strong && strong < huge);
    // k -> infinity is the limit with erp = 1: bias 50, impulse 25.
    assert_close(huge, 25.0, 1e-3, "stiff limit");
}

#[test]
fn disabled_limits_produce_no_impulse() {
    let mut c = limit(0, 1, 5.0, 3.0, StopType::Hard);
    c.limits.enabled = false;
    let mut s = solver_with(4, vec![c]);
    let total = s.solve(&[1.0, 1.0], DT);
    assert_eq!(total, 0.0);
    assert_eq!(s.constraints[0].velocity, 3.0);
    assert!(s.warm_start.is_empty());
}

#[test]
fn static_bodies_cannot_push_a_joint_back() {
    // Both bodies static, and bodies outside the mass table (also static).
    for (a, b, masses) in [(0, 1, vec![0.0, 0.0]), (5, 6, vec![1.0, 1.0])] {
        let mut s = solver_with(4, vec![limit(a, b, 1.5, 3.0, StopType::Hard)]);
        let total = s.solve(&masses, DT);
        assert_eq!(total, 0.0);
        assert_eq!(s.constraints[0].velocity, 3.0);
        assert_eq!(s.constraints[0].warm_impulse, 0.0);
    }
    // One dynamic body (inverse mass 4): impulse (10 + 3) / 4.
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, 3.0, StopType::Hard)]);
    let total = s.solve(&[0.0, 4.0], DT);
    assert_close(total, 3.25, 1e-12, "impulse");
    assert_close(-s.constraints[0].velocity, 10.0, 1e-9, "s*v");
}

#[test]
fn independent_constraints_are_each_solved() {
    let mut s = solver_with(
        3,
        vec![
            limit(0, 1, 1.5, 3.0, StopType::Hard),
            limit(2, 3, -1.2, 0.0, StopType::Hard),
            limit(4, 5, 0.0, 9.0, StopType::Hard),
        ],
    );
    let total = s.solve(&[1.0; 6], DT);
    assert_close(total, 6.5 + 2.0, 1e-12, "sum of the active impulses");
    assert_close(-s.constraints[0].velocity, 10.0, 1e-9, "first");
    assert_close(s.constraints[1].velocity, 4.0, 1e-9, "second");
    assert_eq!(s.constraints[2].velocity, 9.0, "inside its limits");
    assert_eq!(s.constraints[2].warm_impulse, 0.0);
    assert_eq!(s.warm_start.len(), 2);
}

#[test]
fn the_warm_start_is_applied_before_iterating() {
    let mut s = solver_with(4, vec![limit(0, 1, 1.5, 3.0, StopType::Hard)]);
    s.solve(&[1.0, 1.0], DT);

    // With no iterations left only the cached impulse acts: v = 3 - 2 * 6.5.
    s.iterations = 0;
    s.clear_constraints();
    s.add_constraint(limit(0, 1, 1.5, 3.0, StopType::Hard));
    let total = s.solve(&[1.0, 1.0], DT);
    assert_close(total, 6.5, 1e-12, "warm impulse");
    assert_close(
        s.constraints[0].velocity,
        -10.0,
        1e-9,
        "warm-started velocity",
    );
}

#[test]
fn bad_warm_starts_are_corrected_or_discarded() {
    let fresh = || limit(0, 1, 1.5, 3.0, StopType::Hard);
    let mut s = solver_with(4, vec![fresh()]);
    s.solve(&[1.0, 1.0], DT);

    // Ten times too much impulse is taken back to the converged value.
    s.warm_start.scale(10.0);
    s.clear_constraints();
    s.add_constraint(fresh());
    let total = s.solve(&[1.0, 1.0], DT);
    assert_close(total, 6.5, 1e-12, "over-applied warm start");
    assert_close(-s.constraints[0].velocity, 10.0, 1e-9, "s*v");

    // A non-finite cache entry is ignored instead of poisoning the state.
    s.warm_start.scale(f64::NAN);
    s.clear_constraints();
    s.add_constraint(fresh());
    let total = s.solve(&[1.0, 1.0], DT);
    assert_close(total, 6.5, 1e-12, "non-finite warm start");
    assert!(s.constraints[0].velocity.is_finite());
}

#[test]
fn a_zero_time_step_stays_finite() {
    for stop in [
        StopType::Hard,
        StopType::Soft {
            stiffness: 100.0,
            damping: 1.0,
        },
    ] {
        let mut s = solver_with(4, vec![limit(0, 1, 1.5, 3.0, stop)]);
        let total = s.solve(&[1.0, 1.0], 0.0);
        // No position correction without a time step; the approach is stopped.
        assert!(total.is_finite());
        assert_close(total, 1.5, 1e-12, "impulse that stops the approach");
        assert_close(s.constraints[0].velocity, 0.0, 1e-9, "velocity");
    }
}

// ---------------------------------------------------------------------------
// Penetration is a magnitude, `side` names the limit, impulses are force * dt
// ---------------------------------------------------------------------------

#[test]
fn penetration_is_a_non_negative_magnitude_and_side_names_the_violated_limit() {
    let limits = JointLimits::new(-1.0, 1.0);
    assert_close(
        limits.penetration(-1.5),
        0.5,
        1e-15,
        "below the lower limit",
    );
    assert_close(limits.penetration(1.5), 0.5, 1e-15, "above the upper limit");
    assert_eq!(limits.penetration(0.25), 0.0);
    assert_eq!(limits.penetration(-1.0), 0.0);
    assert_eq!(limits.penetration(1.0), 0.0);

    assert_eq!(limits.side(-1.5), 1.0, "below the lower limit: push up");
    assert_eq!(limits.side(1.5), -1.0, "above the upper limit: push down");
    assert_eq!(limits.side(0.25), 0.0);
    assert_eq!(limits.side(-1.0), 0.0, "on the limit is inside it");
    assert_eq!(limits.side(1.0), 0.0, "on the limit is inside it");
    assert_eq!(limits.side(f64::NAN), 0.0);
    // The side and the magnitude together are the signed violation.
    for value in [-3.0, -1.2, 0.0, 1.7, 4.0] {
        let signed = limits.side(value) * limits.penetration(value);
        assert_close(
            value + signed,
            limits.clamp(value),
            1e-15,
            "value moved by the signed penetration lands in the limits",
        );
    }
}

/// Soft stop with the given stiffness and damping on limits `[-1, 1]`.
fn soft_limit(position: f64, velocity: f64, stiffness: f64, damping: f64) -> JointLimitConstraint {
    limit(
        0,
        1,
        position,
        velocity,
        StopType::Soft { stiffness, damping },
    )
}

#[test]
fn compute_impulse_is_the_limit_force_times_the_time_step() {
    // 0.4 past the upper limit with k = 2000 N/m: F = 800 N, J = F dt.
    let c = soft_limit(1.4, 0.0, 2000.0, 0.0);
    assert_close(c.compute_impulse(0.005), -4.0, 1e-12, "upper limit, N s");
    // The impulse is linear in dt and has nothing to do with any mass.
    assert_close(c.compute_impulse(0.01), -8.0, 1e-12, "doubled step");
    assert_eq!(c.compute_impulse(0.0), 0.0);
    // 0.25 below the lower limit: F = 500 N pushes up.
    let c = soft_limit(-1.25, 0.0, 2000.0, 0.0);
    assert_close(c.compute_impulse(0.005), 2.5, 1e-12, "lower limit, N s");
    // It is exactly the stop's force times dt, signed by the side.
    for (position, velocity) in [(1.4, 0.7), (1.4, -0.7), (-1.25, 0.7), (-1.25, -0.7)] {
        let c = soft_limit(position, velocity, 300.0, 12.0);
        let side = c.limits.side(position);
        let force = c
            .stop_type
            .constraint_force(c.penetration_depth(), side * velocity);
        assert_close(
            c.compute_impulse(0.004),
            side * force * 0.004,
            1e-15,
            "impulse = side * force * dt",
        );
    }
}

#[test]
fn the_impulse_points_back_into_the_limits_on_both_sides() {
    for stop in [
        StopType::Hard,
        StopType::Soft {
            stiffness: 500.0,
            damping: 5.0,
        },
    ] {
        let below = limit(0, 1, -1.3, -1.0, stop);
        let above = limit(0, 1, 1.3, 1.0, stop);
        assert!(below.compute_impulse(0.01) > 0.0, "{stop:?}: pushes up");
        assert!(above.compute_impulse(0.01) < 0.0, "{stop:?}: pushes down");
        // Mirror images give opposite impulses of the same size.
        assert_close(
            below.compute_impulse(0.01),
            -above.compute_impulse(0.01),
            1e-6,
            "mirror symmetry",
        );
    }
    // The hard stop is the 1e9 N/m penalty: 1e9 * 0.3 * 0.01.
    let hard = limit(0, 1, 1.3, 0.0, StopType::Hard);
    assert_close(hard.compute_impulse(0.01), -3.0e6, 1e-3, "hard stop");
}

#[test]
fn the_damper_resists_the_approach_at_the_upper_limit_too() {
    // k pen = 100 * 0.5 = 50 N; the damper adds c * speed into the limit.
    let (k, c) = (100.0, 10.0);
    let dt = 0.01;
    // Upper limit: moving up is moving deeper.
    assert_close(
        soft_limit(1.5, 2.0, k, c).compute_impulse(dt),
        -(50.0 + 20.0) * dt,
        1e-12,
        "deeper at the upper limit",
    );
    assert_close(
        soft_limit(1.5, -2.0, k, c).compute_impulse(dt),
        -50.0 * dt,
        1e-12,
        "leaving the upper limit",
    );
    // Lower limit: moving down is moving deeper.
    assert_close(
        soft_limit(-1.5, -2.0, k, c).compute_impulse(dt),
        (50.0 + 20.0) * dt,
        1e-12,
        "deeper at the lower limit",
    );
    assert_close(
        soft_limit(-1.5, 2.0, k, c).compute_impulse(dt),
        50.0 * dt,
        1e-12,
        "leaving the lower limit",
    );
}

#[test]
fn no_impulse_inside_the_limits_or_with_disabled_limits() {
    let inside = soft_limit(0.3, 5.0, 1e4, 10.0);
    assert_eq!(inside.compute_impulse(0.01), 0.0);
    assert_eq!(soft_limit(1.0, 5.0, 1e4, 10.0).compute_impulse(0.01), 0.0);
    let mut disabled = soft_limit(2.0, 5.0, 1e4, 10.0);
    disabled.limits.enabled = false;
    assert_eq!(disabled.compute_impulse(0.01), 0.0);
}

fn prismatic(translation: f64, velocity: f64) -> PrismaticJointLimit {
    let mut joint = PrismaticJointLimit::new(
        0,
        1,
        [1.0, 0.0, 0.0],
        -0.5,
        0.5,
        StopType::Hard,
        JointComplianceParams::rigid(),
    );
    joint.translation = translation;
    joint.velocity = velocity;
    joint
}

#[test]
fn prismatic_impulse_stops_a_joint_that_moves_deeper_into_the_upper_limit() {
    // 0.3 past the upper limit: bias = 0.2 * 0.3 / 0.01 = 6, row = -1, K = 2.
    let at_rest = prismatic(0.8, 0.0).solve_impulse(1.0, 1.0, DT);
    assert_close(at_rest, 3.0, 1e-12, "at rest");
    // Moving deeper at +10: the velocity condition -v >= 6 needs (6 + 10) / 2.
    // (The old (bias - v) / K gave nothing at all here.)
    let deeper = prismatic(0.8, 10.0).solve_impulse(1.0, 1.0, DT);
    assert_close(deeper, 8.0, 1e-12, "moving deeper");
    // Leaving the limit faster than the bias needs no push.
    assert_eq!(prismatic(0.8, -10.0).solve_impulse(1.0, 1.0, DT), 0.0);
    // Leaving slower than the bias still gets the missing part.
    assert_close(
        prismatic(0.8, -2.0).solve_impulse(1.0, 1.0, DT),
        2.0,
        1e-12,
        "leaving slowly",
    );
}

#[test]
fn prismatic_impulse_is_mirror_symmetric_between_the_limits() {
    for velocity in [-10.0, -2.0, 0.0, 2.0, 10.0] {
        let upper = prismatic(0.8, velocity).solve_impulse(1.0, 3.0, DT);
        let lower = prismatic(-0.8, -velocity).solve_impulse(1.0, 3.0, DT);
        assert_close(upper, lower, 1e-12, &format!("velocity {velocity}"));
    }
    assert_eq!(prismatic(0.1, 7.0).solve_impulse(1.0, 1.0, DT), 0.0);
    let mut disabled = prismatic(0.8, 7.0);
    disabled.limits.enabled = false;
    assert_eq!(disabled.solve_impulse(1.0, 1.0, DT), 0.0);
    assert_eq!(
        disabled.constraint_row(),
        -1.0,
        "the row names the side only"
    );
}

fn revolute(angle: f64, ang_velocity: f64, restitution: f64) -> RevoluteJointLimit {
    let mut joint = RevoluteJointLimit::new(
        0,
        1,
        [0.0, 0.0, 1.0],
        -1.0,
        1.0,
        StopType::Hard,
        JointComplianceParams::new(0.5, 0.0),
        restitution,
    );
    joint.angle = angle;
    joint.ang_velocity = ang_velocity;
    joint
}

#[test]
fn revolute_impulse_is_mirror_symmetric_between_the_limits() {
    // 0.4 past a limit: bias = 0.5 * 0.4 / 0.01 = 20, K = 2.
    for (velocity, restitution, expected) in [
        // Moving deeper at 3 rad/s; restitution 0.5 adds 0.5 * 3 to the target.
        (3.0, 0.5, (20.0 + 1.5 + 3.0) / 2.0),
        (3.0, 0.0, (20.0 + 3.0) / 2.0),
        // At rest.
        (0.0, 0.5, 20.0 / 2.0),
        // Leaving at 3 rad/s: no restitution, the bias is reached sooner.
        (-3.0, 0.5, (20.0 - 3.0) / 2.0),
        // Leaving faster than the bias: no push.
        (-30.0, 0.5, 0.0),
    ] {
        let upper = revolute(1.4, velocity, restitution).solve_impulse(1.0, 1.0, DT);
        let lower = revolute(-1.4, -velocity, restitution).solve_impulse(1.0, 1.0, DT);
        assert_close(upper, expected, 1e-12, &format!("upper, v = {velocity}"));
        assert_close(lower, expected, 1e-12, &format!("lower, v = {velocity}"));
    }
    assert_eq!(revolute(0.2, 5.0, 0.5).solve_impulse(1.0, 1.0, DT), 0.0);
    let mut disabled = revolute(1.4, 5.0, 0.5);
    disabled.limits.enabled = false;
    assert_eq!(disabled.solve_impulse(1.0, 1.0, DT), 0.0);
}

// ---------------------------------------------------------------------------
// The friction bound is an impulse: force * dt
// ---------------------------------------------------------------------------

fn friction(coefficient: f64, normal_force: f64) -> JointFrictionConstraint {
    let mut fric = JointFrictionConstraint::new(0, 1, [0.0, 0.0, 1.0], coefficient);
    fric.normal_force = normal_force;
    fric
}

#[test]
fn the_friction_bound_is_the_coulomb_force_times_the_time_step() {
    let fric = friction(0.5, 10.0);
    assert_close(fric.max_friction_impulse(0.01), 0.05, 1e-15, "mu N dt");
    assert_close(
        fric.max_friction_impulse(0.02),
        0.10,
        1e-15,
        "twice the step",
    );
    // The direction of the normal force does not matter.
    assert_close(
        friction(0.5, -10.0).max_friction_impulse(0.01),
        0.05,
        1e-15,
        "negative normal force",
    );
    // Nothing can be transmitted without a coefficient, a force or time.
    assert_eq!(friction(-0.5, 10.0).max_friction_impulse(0.01), 0.0);
    assert_eq!(friction(0.5, 0.0).max_friction_impulse(0.01), 0.0);
    assert_eq!(fric.max_friction_impulse(0.0), 0.0);
    assert_eq!(fric.max_friction_impulse(-0.01), 0.0);
    assert_eq!(fric.max_friction_impulse(f64::NAN), 0.0);
}

#[test]
fn a_fast_joint_is_decelerated_by_the_coulomb_impulse_not_stopped() {
    let mut fric = friction(0.5, 10.0);
    fric.relative_velocity = 1.0e6;
    assert_close(fric.compute_impulse(1.0, 0.01), -0.05, 1e-15, "clamped");
    fric.relative_velocity = -1.0e6;
    assert_close(fric.compute_impulse(1.0, 0.01), 0.05, 1e-15, "clamped");
    // A unit-mass joint loses mu N / m * dt = 0.05 m/s of speed per step.
    let mut fric = friction(0.5, 100.0);
    fric.relative_velocity = 2.0;
    let impulse = fric.compute_impulse(1.0, 0.01);
    assert_close(impulse, -0.5, 1e-15, "dv = mu N dt inv_mass");
    // A slow joint is brought to rest without overshooting.
    fric.relative_velocity = 0.2;
    assert_close(fric.compute_impulse(1.0, 0.01), -0.2, 1e-15, "sticks");
    // No time, no friction impulse.
    fric.relative_velocity = 2.0;
    assert_eq!(fric.compute_impulse(1.0, 0.0), 0.0);
}

#[test]
fn the_friction_warm_start_is_clamped_to_the_impulse_bound() {
    let mut fric = friction(0.1, 50.0);
    let bound = 0.1 * 50.0 * 0.01;
    fric.warm_impulse = 1000.0;
    assert_close(fric.warm_start_impulse(0.01), bound, 1e-15, "above");
    fric.warm_impulse = -1000.0;
    assert_close(fric.warm_start_impulse(0.01), -bound, 1e-15, "below");
    fric.warm_impulse = 0.01;
    assert_close(fric.warm_start_impulse(0.01), 0.01, 1e-15, "within");
}

#[test]
fn the_articulated_stop_damper_resists_the_approach_at_the_upper_limit_too() {
    let stop_at = |position: f64, velocity: f64| {
        let mut stop = ArticulatedBodyJointStop::new(
            0,
            2.0,
            JointLimits::new(-1.0, 1.0),
            StopType::Soft {
                stiffness: 100.0,
                damping: 10.0,
            },
        );
        stop.position = position;
        stop.velocity = velocity;
        stop
    };
    // k * penetration = 50 N, the damper adds 10 * 2 = 20 N for an approach at
    // 2 per second; the helper divides the impulse by the inertia 2.
    assert_close(
        stop_at(1.5, 2.0).compute_impulse(DT),
        70.0 * DT / 2.0,
        1e-12,
        "approach at the upper limit",
    );
    assert_close(
        stop_at(1.5, -2.0).compute_impulse(DT),
        50.0 * DT / 2.0,
        1e-12,
        "leaving the upper limit",
    );
    assert_close(
        stop_at(-1.5, -2.0).compute_impulse(DT),
        70.0 * DT / 2.0,
        1e-12,
        "approach at the lower limit",
    );
    assert_close(
        stop_at(-1.5, 2.0).compute_impulse(DT),
        50.0 * DT / 2.0,
        1e-12,
        "leaving the lower limit",
    );
}
