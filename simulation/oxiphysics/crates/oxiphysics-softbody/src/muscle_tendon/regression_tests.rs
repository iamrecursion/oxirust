// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the dynamics of [`MusculotendonUnit`].
//!
//! `MusculotendonUnit::step` used to re-solve the static equilibrium only; the
//! fibre velocity was never updated, so the force-velocity curve, the damping,
//! and the power and work of [`MusculotendonAnalysis`] were identically zero.
//! These tests drive a unit with moving boundary conditions (activation build-up
//! and a ramp of the musculotendon length) and check the fibre velocity, the
//! force balance between fibre and tendon, and the resulting power and work.

use super::*;

const DT: f64 = 0.001;
/// Imposed speed of the musculotendon length in the ramp tests (m/s).
const RAMP_SPEED: f64 = 0.05;

/// A fully excited unit that has settled into its isometric state at
/// `l_mt = 0.30` m (fibre 0.10 m, tendon slack length 0.20 m).
fn settled_unit(pennation: f64) -> MusculotendonUnit {
    let mut unit = MusculotendonUnit::new("test", 1000.0, 0.10, 0.20, pennation);
    unit.activation.set_excitation(1.0);
    unit.activation.activation = 1.0;
    for _ in 0..400 {
        unit.step(DT);
    }
    unit
}

fn force_imbalance(unit: &MusculotendonUnit) -> f64 {
    (unit.projected_fiber_force() - unit.tendon_force()).abs()
}

/// Ramp the musculotendon length by `delta` over 100 steps and return the fibre
/// velocity after every step.
fn ramp(unit: &mut MusculotendonUnit, delta: f64) -> Vec<f64> {
    let l0 = unit.l_mt;
    (0..100)
        .map(|k| {
            unit.set_length(l0 + delta * (k + 1) as f64 / 100.0);
            unit.step(DT);
            assert!(
                force_imbalance(unit) < 1e-6,
                "step {k}: fibre and tendon force differ by {} N",
                force_imbalance(unit)
            );
            unit.muscle.v_fiber
        })
        .collect()
}

/// The tendon is slack and the muscle is excited: the fibre has to shorten to
/// stretch the tendon, i.e. a negative fibre velocity that follows from the
/// motion of the fibre (`v = Δl / dt`).
#[test]
fn contraction_from_slack_shortens_the_fibre_with_negative_velocity() {
    let mut unit = MusculotendonUnit::new("fresh", 1000.0, 0.10, 0.20, 0.0);
    unit.activation.set_excitation(1.0);
    let mut previous_length = unit.muscle.l_fiber;
    assert_eq!(unit.muscle.v_fiber, 0.0);
    for k in 0..10 {
        unit.step(DT);
        let v = unit.muscle.v_fiber;
        assert!(
            v < -0.1,
            "step {k}: v_fiber = {v} should be clearly negative"
        );
        assert!(
            unit.muscle.l_fiber < previous_length,
            "step {k}: the fibre must shorten"
        );
        let expected = (unit.muscle.l_fiber - previous_length) / DT;
        assert!(
            (v - expected).abs() < 1e-12,
            "step {k}: v_fiber {v} is not the fibre displacement rate {expected}"
        );
        assert!(force_imbalance(&unit) < 1e-6, "step {k}");
        previous_length = unit.muscle.l_fiber;
    }
    // The tendon carries a growing force while the fibre keeps shortening.
    assert!(unit.tendon_force() > 100.0, "{}", unit.tendon_force());
}

/// At constant musculotendon length the unit settles into the static
/// equilibrium with a vanishing fibre velocity.
#[test]
fn isometric_unit_settles_to_the_static_equilibrium() {
    for pennation in [0.0, 0.3] {
        let unit = settled_unit(pennation);
        assert!(
            unit.muscle.v_fiber.abs() < 1e-5,
            "pennation {pennation}: v_fiber = {}",
            unit.muscle.v_fiber
        );
        let mut reference = MusculotendonUnit::new("reference", 1000.0, 0.10, 0.20, pennation);
        reference.activation.activation = 1.0;
        reference.set_length(unit.l_mt);
        reference.solve_equilibrium();
        assert!(
            (unit.muscle.l_fiber - reference.muscle.l_fiber).abs() < 1e-6,
            "pennation {pennation}: dynamic {} vs static {}",
            unit.muscle.l_fiber,
            reference.muscle.l_fiber
        );
        assert!(force_imbalance(&unit) < 1e-6);
        assert!(unit.tendon_force() > 800.0, "{}", unit.tendon_force());
    }
}

/// Stretching the unit gives a positive fibre velocity close to the imposed
/// speed (the tendon is much stiffer than the fibre's force-velocity slope),
/// shortening a negative one.  The velocity must never flip its sign during a
/// monotone ramp: an explicit velocity update from the previous equilibrium
/// oscillates for the 1 ms step used here.
#[test]
fn ramp_of_the_musculotendon_length_gives_fibre_velocity_of_the_same_sign() {
    let up = ramp(&mut settled_unit(0.0), RAMP_SPEED * 0.1);
    assert!(
        up.iter().all(|v| *v > 0.0),
        "lengthening ramp: fibre velocity changed sign: {up:?}"
    );
    assert!(
        up.windows(2).all(|w| w[1] >= w[0] - 1e-9),
        "lengthening ramp: fibre velocity must rise smoothly: {up:?}"
    );
    let v_end = up[up.len() - 1];
    assert!(
        v_end > 0.9 * RAMP_SPEED && v_end < 1.01 * RAMP_SPEED,
        "fibre velocity {v_end} should approach the imposed {RAMP_SPEED} m/s"
    );

    let down = ramp(&mut settled_unit(0.0), -RAMP_SPEED * 0.1);
    assert!(
        down.iter().all(|v| *v < 0.0),
        "shortening ramp: fibre velocity changed sign: {down:?}"
    );
    let v_end = down[down.len() - 1];
    assert!(
        v_end < -0.9 * RAMP_SPEED && v_end > -1.01 * RAMP_SPEED,
        "fibre velocity {v_end} should approach the imposed -{RAMP_SPEED} m/s"
    );
}

/// Force-velocity effect: at the same musculotendon length a lengthening fibre
/// carries more force than the static equilibrium (eccentric branch of the
/// force-velocity curve), a shortening fibre less (concentric branch).
#[test]
fn force_velocity_effect_is_active_during_the_ramp() {
    let static_force_at = |l_mt: f64| {
        let mut reference = settled_unit(0.0);
        reference.set_length(l_mt);
        reference.solve_equilibrium();
        reference.tendon_force()
    };

    let mut stretched = settled_unit(0.0);
    ramp(&mut stretched, RAMP_SPEED * 0.1);
    let static_stretched = static_force_at(stretched.l_mt);
    assert!(
        stretched.tendon_force() > static_stretched + 50.0,
        "stretched {} vs static {static_stretched}",
        stretched.tendon_force()
    );

    let mut shortened = settled_unit(0.0);
    ramp(&mut shortened, -RAMP_SPEED * 0.1);
    let static_shortened = static_force_at(shortened.l_mt);
    assert!(
        shortened.tendon_force() < static_shortened - 50.0,
        "shortened {} vs static {static_shortened}",
        shortened.tendon_force()
    );
}

/// Power and work of the analysis helper were identically zero.
#[test]
fn analysis_reports_power_and_work_during_the_ramp() {
    let moment_arm = 0.05;
    let mut analysis = MusculotendonAnalysis::new(vec![settled_unit(0.0)], vec![moment_arm]);
    analysis.set_excitation(0, 1.0);
    assert_eq!(analysis.total_work(), 0.0);

    let l0 = analysis.units[0].l_mt;
    for k in 0..100 {
        analysis.set_mt_length(0, l0 + RAMP_SPEED * 0.1 * (k + 1) as f64 / 100.0);
        analysis.step(DT);
    }
    let power = analysis.powers()[0];
    assert!(
        power > 20.0,
        "force (~1.1 kN) times fibre speed (~5 cm/s) is ~50 W, got {power}"
    );
    assert!(analysis.joint_power() > 20.0, "{}", analysis.joint_power());
    let work = analysis.total_work();
    // F ≈ 1.0..1.1 kN over a fibre stretch of about 4 mm: roughly 4 J.
    assert!(work > 2.0 && work < 8.0, "total work {work} J");

    // Shortening ramp: the fibre power (F · v) is negative.
    let mut shortening = MusculotendonAnalysis::new(vec![settled_unit(0.0)], vec![moment_arm]);
    shortening.set_excitation(0, 1.0);
    for k in 0..100 {
        shortening.set_mt_length(0, l0 - RAMP_SPEED * 0.1 * (k + 1) as f64 / 100.0);
        shortening.step(DT);
    }
    assert!(shortening.powers()[0] < -20.0, "{}", shortening.powers()[0]);
    assert!(shortening.total_work() > 2.0);
}

/// With a very long step the velocity term vanishes and the step reproduces
/// the static equilibrium.
#[test]
fn very_long_step_reaches_the_static_equilibrium() {
    for pennation in [0.0, 0.4] {
        // Excitation equal to the activation: the activation stays at 0.7.
        let mut stepped = MusculotendonUnit::new("long", 1000.0, 0.10, 0.20, pennation);
        stepped.activation.activation = 0.7;
        stepped.activation.set_excitation(0.7);
        stepped.step(1.0e4);
        assert!((stepped.activation.activation - 0.7).abs() < 1e-12);

        let mut reference = MusculotendonUnit::new("ref", 1000.0, 0.10, 0.20, pennation);
        reference.activation.activation = 0.7;
        reference.solve_equilibrium();
        assert!(
            (stepped.muscle.l_fiber - reference.muscle.l_fiber).abs() < 1e-6,
            "pennation {pennation}: {} vs {}",
            stepped.muscle.l_fiber,
            reference.muscle.l_fiber
        );
        assert!(stepped.muscle.v_fiber.abs() < 1e-6);
    }
}

/// `dt = 0` falls back to the static solve and never divides by zero.
#[test]
fn zero_time_step_solves_the_static_equilibrium() {
    let mut unit = MusculotendonUnit::new("zero", 1000.0, 0.10, 0.20, 0.0);
    unit.activation.activation = 0.8;
    unit.activation.set_excitation(0.8);
    unit.step(0.0);
    assert!(unit.muscle.l_fiber.is_finite() && unit.muscle.v_fiber == 0.0);
    assert!(force_imbalance(&unit) < 1e-5);
    assert!(
        unit.muscle.l_fiber < 0.10,
        "an active unit shortens its fibre"
    );
}

/// The static solve describes a state without fibre motion.
#[test]
fn static_equilibrium_resets_the_fibre_velocity() {
    let mut unit = MusculotendonUnit::new("static", 1000.0, 0.10, 0.20, 0.0);
    unit.activation.activation = 0.5;
    unit.muscle.set_state(0.10, 0.4);
    unit.solve_equilibrium();
    assert_eq!(unit.muscle.v_fiber, 0.0);
    assert!(force_imbalance(&unit) < 1e-5);
}

/// `total_force_at` evaluates an explicit state and agrees with the
/// state-based `total_force` and with the sum of its three components.
#[test]
fn total_force_at_matches_the_stored_state_force() {
    let mut muscle = HillMuscleModel::new(900.0, 0.12);
    for (l, v, a) in [
        (0.12, 0.0, 1.0),
        (0.10, -0.4, 0.6),
        (0.15, 0.7, 0.9),
        (0.09, 1.5, 0.3),
    ] {
        let explicit = muscle.total_force_at(a, l, v);
        muscle.set_state(l, v);
        let stored = muscle.total_force(a);
        assert!((explicit - stored).abs() < 1e-9, "{explicit} vs {stored}");
        let parts = muscle.active_force(a) + muscle.passive_force() + muscle.damping_force();
        assert!((explicit - parts).abs() < 1e-9, "{explicit} vs {parts}");
    }
    // Reading the state through the explicit API does not disturb it.
    muscle.set_state(0.11, 0.2);
    let _ = muscle.total_force_at(1.0, 0.2, -0.5);
    assert_eq!((muscle.l_fiber, muscle.v_fiber), (0.11, 0.2));
}
