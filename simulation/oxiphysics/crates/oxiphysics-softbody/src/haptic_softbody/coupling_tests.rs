// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the force returned by [`HapticRenderingLoop::tick`].
//!
//! [`VirtualCoupling::coupling_force`] is the force *on the simulated tool*.
//! The loop used to add it, unnegated, to the force commanded to the device
//! (positive feedback: the harder the user pulled away from the tool, the
//! harder the device was pushed away), never advanced the tool, evaluated the
//! contact model twice and limited only the contact part to the device range.

use super::*;

const DT: f64 = 1.0e-3;
/// Surface y = 0 with the outward normal +y; a cursor above it is not in contact.
const SURFACE_POINT: [f64; 3] = [0.0, -1.0, 0.0];
const SURFACE_NORMAL: [f64; 3] = [0.0, 1.0, 0.0];

fn loop_with(coupling_k: f64, coupling_b: f64, tool_mass: f64) -> HapticRenderingLoop {
    HapticRenderingLoop::new(1000.0, 10.0, coupling_k, coupling_b, tool_mass, DT)
}

#[test]
fn coupling_reaction_pulls_the_device_towards_the_tool() {
    // Device held 0.1 m away from a tool that starts at the origin; an
    // undamped coupling (K = 50 N/m, 5 N initially, below the device limit).
    let mut haptic = loop_with(50.0, 0.0, 0.05);
    let device = [0.1, 0.0, 0.0];
    let first = haptic.tick(device, [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
    assert!(
        norm3(sub3(first, [-5.0, 0.0, 0.0])) < 1e-12,
        "the device must be pulled back towards the tool with K * 0.1 = 5 N, got {first:?}"
    );
    // The undamped tool reaches the device after a quarter period
    // (pi / (2 * sqrt(K / m)) = 49.7 ms): until then the force keeps pointing
    // from the device towards the tool and decays as the tool catches up.
    let mut previous = first[0].abs();
    let mut previous_tool = haptic.coupling.tool_position[0];
    assert!(
        previous_tool > 0.0,
        "the first tick must already move the tool"
    );
    for tick in 1..45 {
        let f = haptic.tick(device, [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
        assert!(
            f[0] < 0.0,
            "tick {tick}: force {f:?} must point towards the tool"
        );
        assert!(f[1].abs() < 1e-12 && f[2].abs() < 1e-12);
        assert!(f[0].abs() < previous, "tick {tick}: force must decay");
        let tool = haptic.coupling.tool_position[0];
        assert!(
            tool > previous_tool && tool < device[0],
            "tick {tick}: tool at {tool}"
        );
        previous = f[0].abs();
        previous_tool = tool;
    }
}

#[test]
fn tick_advances_the_virtual_tool_under_the_coupling_force() {
    let mut haptic = loop_with(50.0, 0.0, 0.05);
    haptic.tick([0.1, 0.0, 0.0], [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
    // a = K * 0.1 / m = 100 m/s^2: v = a dt, x = v dt (semi-implicit Euler).
    assert!((haptic.coupling.tool_velocity[0] - 0.1).abs() < 1e-15);
    assert!((haptic.coupling.tool_position[0] - 1.0e-4).abs() < 1e-15);
    assert_eq!(haptic.coupling.tool_velocity[1], 0.0);
}

#[test]
fn a_tool_that_caught_up_no_longer_pulls_the_device() {
    // Critically damped coupling: B = 2 sqrt(K m).
    let (k, m) = (50.0, 0.05);
    let mut haptic = loop_with(k, 2.0 * (k * m).sqrt(), m);
    let device = [0.1, 0.0, 0.0];
    let mut last = [0.0; 3];
    for _ in 0..1000 {
        last = haptic.tick(device, [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
    }
    assert!(
        (haptic.coupling.tool_position[0] - 0.1).abs() < 1e-6,
        "the tool must reach the device, at {:?}",
        haptic.coupling.tool_position
    );
    assert!(
        norm3(last) < 1e-3,
        "the pull must have decayed, got {last:?}"
    );
}

#[test]
fn damper_reaction_resists_the_device_velocity() {
    // No spring: the damper acts on the velocity difference alone.
    let mut haptic = loop_with(0.0, 4.0, 0.05);
    let f = haptic.tick(
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        SURFACE_POINT,
        SURFACE_NORMAL,
    );
    assert!(
        norm3(sub3(f, [-0.8, 0.0, 0.0])) < 1e-12,
        "B * v = 0.8 N must oppose the device motion, got {f:?}"
    );
}

#[test]
fn total_force_is_limited_to_the_device_range() {
    // K = 500 N/m and 0.1 m of separation would be 50 N.
    let mut haptic = loop_with(500.0, 0.0, 0.05);
    let f = haptic.tick([0.1, 0.0, 0.0], [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
    assert!(
        (norm3(f) - PHANTOM_MAX_FORCE_N).abs() < 1e-12,
        "|f| = {} must be saturated at {PHANTOM_MAX_FORCE_N}",
        norm3(f)
    );
    assert!(
        f[0] < 0.0 && f[1].abs() < 1e-12 && f[2].abs() < 1e-12,
        "{f:?}"
    );

    // Contact and coupling together: the direction is kept when saturating.
    let mut haptic = loop_with(500.0, 0.0, 0.05);
    let device = [0.02, -0.004, 0.0];
    let contact_point = [0.02, 0.0, 0.0];
    let f = haptic.tick(device, [0.0; 3], contact_point, SURFACE_NORMAL);
    // Contact: 1000 * 0.004 = 4 N along +y. Reaction: -500 * device.
    let unsaturated = [-10.0, 2.0 + 4.0, 0.0];
    let scale = PHANTOM_MAX_FORCE_N / norm3(unsaturated);
    assert!(
        norm3(sub3(f, scale3(unsaturated, scale))) < 1e-9,
        "expected the saturated resultant, got {f:?}"
    );
    // A lower limit on the model is honoured as well.
    let mut haptic = loop_with(500.0, 0.0, 0.05);
    haptic.contact_model.max_force = 3.0;
    let f = haptic.tick([0.1, 0.0, 0.0], [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
    assert!((norm3(f) - 3.0).abs() < 1e-12, "{f:?}");
}

#[test]
fn surface_contact_force_pushes_the_device_out_of_the_surface() {
    let mut haptic = loop_with(50.0, 0.0, 0.05);
    let device = [0.0, -0.002, 0.0];
    // The tool already sits at the device, so the coupling is relaxed.
    haptic.coupling.tool_position = device;
    let f = haptic.tick(device, [0.0; 3], [0.0; 3], SURFACE_NORMAL);
    // k_s * penetration = 1000 * 0.002 = 2 N out of the surface.
    assert!(norm3(sub3(f, [0.0, 2.0, 0.0])) < 1e-12, "f = {f:?}");
    assert_eq!(haptic.cursor.feedback_force, f);
    assert!(haptic.proxy.in_contact);
}

#[test]
fn contact_and_coupling_reactions_superpose_below_the_limit() {
    let mut haptic = loop_with(20.0, 0.0, 0.05);
    let device = [0.05, -0.002, 0.0];
    let f = haptic.tick(device, [0.0; 3], [0.05, 0.0, 0.0], SURFACE_NORMAL);
    // Contact: 1000 * 0.002 = 2 N along +y. Coupling on the tool: 20 * device;
    // the user feels its negative: (-1.0, +0.04, 0).
    assert!(
        norm3(sub3(f, [-1.0, 2.04, 0.0])) < 1e-12,
        "f = {f:?}, expected the sum of contact and coupling reaction"
    );
    // feedback_force keeps only the contact component, evaluated once.
    assert!(norm3(sub3(haptic.cursor.feedback_force, [0.0, 2.0, 0.0])) < 1e-12);
}

#[test]
fn a_massless_tool_follows_the_device_exactly() {
    let mut haptic = loop_with(500.0, 5.0, 0.0);
    let device = [0.03, -0.01, 0.02];
    let velocity = [0.1, 0.0, -0.2];
    let first = haptic.tick(device, velocity, SURFACE_POINT, SURFACE_NORMAL);
    assert!(first.iter().all(|c| c.is_finite()));
    assert_eq!(haptic.coupling.tool_position, device);
    assert_eq!(haptic.coupling.tool_velocity, velocity);
    // With no mass there is nothing left to pull the device back.
    let second = haptic.tick(device, velocity, SURFACE_POINT, SURFACE_NORMAL);
    assert!(norm3(second) < 1e-12, "second tick force {second:?}");

    let mut vc = VirtualCoupling::new(100.0, 1.0);
    vc.integrate_tool(device, velocity, -1.0, DT);
    assert_eq!(vc.tool_position, device);
    vc.integrate_tool([1.0, 1.0, 1.0], [0.0; 3], f64::NAN, DT);
    assert_eq!(vc.tool_position, [1.0, 1.0, 1.0]);
}

#[test]
fn a_heavier_tool_lags_behind_the_device_longer() {
    let device = [0.1, 0.0, 0.0];
    let mut light = loop_with(50.0, 3.0, 0.01);
    let mut heavy = loop_with(50.0, 3.0, 1.0);
    for _ in 0..30 {
        light.tick(device, [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
        heavy.tick(device, [0.0; 3], SURFACE_POINT, SURFACE_NORMAL);
    }
    assert!(
        light.coupling.tool_position[0] > heavy.coupling.tool_position[0],
        "light {:?} should be ahead of heavy {:?}",
        light.coupling.tool_position,
        heavy.coupling.tool_position
    );
}
