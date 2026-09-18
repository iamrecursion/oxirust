//! Boundary-condition tests for [`oximedia_graphics::keyframe::Easing`].
//!
//! These tests pin the behaviour of the extended easing curves at their
//! domain edges and in their interiors:
//!
//! - every curve anchors exactly at `apply(0.0) == 0.0` and `apply(1.0) == 1.0`;
//! - the input `t` is **clamped** to `[0, 1]` (the implementation calls
//!   `t.clamp(0.0, 1.0)` first), so out-of-range inputs alias the nearest
//!   endpoint rather than extrapolating;
//! - the "smooth" families (Quad / Cubic / Quart / Sine) plus `CubicBezier`
//!   are monotone non-decreasing and stay inside `[0, 1]`;
//! - the `Back` family deliberately under-/over-shoots `[0, 1]`;
//! - the three `Elastic` variants oscillate outside `[0, 1]`;
//! - `Spring` and `CubicBezier` are clamped to `[0, 1]` internally and never
//!   leave the unit interval;
//! - the `Bounce` family stays inside `[0, 1]`.

use oximedia_graphics::keyframe::Easing;

/// Absolute tolerance for floating-point anchor / range comparisons.
const EPS: f64 = 1e-6;

/// All 24 [`Easing`] variants (22 unit + `CubicBezier` + `Spring`).
///
/// `Easing` is `Clone` but **not** `Copy`, so callers must iterate by
/// reference (`for e in all_variants().iter()`).
fn all_variants() -> Vec<Easing> {
    vec![
        Easing::Linear,
        Easing::EaseInQuad,
        Easing::EaseOutQuad,
        Easing::EaseInOutQuad,
        Easing::EaseInCubic,
        Easing::EaseOutCubic,
        Easing::EaseInOutCubic,
        Easing::EaseInQuart,
        Easing::EaseOutQuart,
        Easing::EaseInOutQuart,
        Easing::EaseInSine,
        Easing::EaseOutSine,
        Easing::EaseInOutSine,
        Easing::EaseInElastic,
        Easing::EaseOutElastic,
        Easing::EaseInOutElastic,
        Easing::EaseInBack,
        Easing::EaseOutBack,
        Easing::EaseInOutBack,
        Easing::EaseInBounce,
        Easing::EaseOutBounce,
        Easing::EaseInOutBounce,
        Easing::CubicBezier(0.25, 0.1, 0.25, 1.0),
        Easing::Spring {
            mass: 1.0,
            stiffness: 100.0,
            damping: 10.0,
        },
    ]
}

/// The monotone-non-decreasing, in-range family (no overshoot, no oscillation):
/// `Linear`, every `Quad`/`Cubic`/`Quart`/`Sine` variant, and the standard
/// CSS ease `CubicBezier(0.25, 0.1, 0.25, 1.0)`.
fn monotone_variants() -> Vec<Easing> {
    vec![
        Easing::Linear,
        Easing::EaseInQuad,
        Easing::EaseOutQuad,
        Easing::EaseInOutQuad,
        Easing::EaseInCubic,
        Easing::EaseOutCubic,
        Easing::EaseInOutCubic,
        Easing::EaseInQuart,
        Easing::EaseOutQuart,
        Easing::EaseInOutQuart,
        Easing::EaseInSine,
        Easing::EaseOutSine,
        Easing::EaseInOutSine,
        Easing::CubicBezier(0.25, 0.1, 0.25, 1.0),
    ]
}

#[test]
fn all_variants_anchor_at_zero_and_one() {
    for e in all_variants().iter() {
        assert!(
            e.apply(0.0).abs() < EPS,
            "{e:?}@0 expected ~0.0, got {}",
            e.apply(0.0)
        );
        assert!(
            (e.apply(1.0) - 1.0).abs() < EPS,
            "{e:?}@1 expected ~1.0, got {}",
            e.apply(1.0)
        );
    }
}

#[test]
fn clamp_below_zero_equals_zero_input() {
    // `apply` clamps `t` to [0, 1] before evaluating, so any negative input
    // must be byte-identical to `apply(0.0)` (NOT extrapolated below it).
    for e in all_variants().iter() {
        let at_zero = e.apply(0.0);
        assert_eq!(e.apply(-0.1), at_zero, "{e:?}: apply(-0.1) != apply(0.0)");
        assert_eq!(
            e.apply(-1000.0),
            at_zero,
            "{e:?}: apply(-1000.0) != apply(0.0)"
        );
    }
}

#[test]
fn clamp_above_one_equals_one_input() {
    // Symmetric to the lower bound: inputs above 1 alias `apply(1.0)`.
    for e in all_variants().iter() {
        let at_one = e.apply(1.0);
        assert_eq!(e.apply(1.1), at_one, "{e:?}: apply(1.1) != apply(1.0)");
        assert_eq!(e.apply(1e9), at_one, "{e:?}: apply(1e9) != apply(1.0)");
    }
}

#[test]
fn monotone_variants_nondecreasing() {
    for e in monotone_variants().iter() {
        let mut prev = e.apply(0.0);
        for k in 0..=100 {
            let t = f64::from(k) / 100.0;
            let v = e.apply(t);
            assert!(
                v >= prev - EPS,
                "{e:?} not non-decreasing at t={t}: {v} < {prev}",
            );
            assert!(
                (-EPS..=1.0 + EPS).contains(&v),
                "{e:?} out of [0,1] at t={t}: {v}",
            );
            prev = v;
        }
    }
}

#[test]
fn back_variants_overshoot() {
    // EaseInBack retracts below 0 early; EaseOutBack overshoots above 1 late;
    // EaseInOutBack does both (under early, over late).
    assert!(
        Easing::EaseInBack.apply(0.4) < -1e-3,
        "EaseInBack@0.4 should undershoot <0, got {}",
        Easing::EaseInBack.apply(0.4)
    );
    assert!(
        Easing::EaseOutBack.apply(0.6) > 1.0 + 1e-3,
        "EaseOutBack@0.6 should overshoot >1, got {}",
        Easing::EaseOutBack.apply(0.6)
    );
    assert!(
        Easing::EaseInOutBack.apply(0.2) < -1e-3,
        "EaseInOutBack@0.2 should undershoot <0, got {}",
        Easing::EaseInOutBack.apply(0.2)
    );
    assert!(
        Easing::EaseInOutBack.apply(0.8) > 1.0 + 1e-3,
        "EaseInOutBack@0.8 should overshoot >1, got {}",
        Easing::EaseInOutBack.apply(0.8)
    );
}

#[test]
fn elastic_variants_overshoot() {
    // Sample the open interval (0, 1) — the endpoints are pinned to 0/1 by the
    // helper short-circuits, so the oscillation only shows in the interior.
    let mut out_above = false;
    let mut in_below = false;
    let mut inout_above = false;
    let mut inout_below = false;

    for k in 1..100 {
        let t = f64::from(k) / 100.0;

        let out = Easing::EaseOutElastic.apply(t);
        if out > 1.0 + 1e-3 {
            out_above = true;
        }

        let ein = Easing::EaseInElastic.apply(t);
        if ein < -1e-3 {
            in_below = true;
        }

        let inout = Easing::EaseInOutElastic.apply(t);
        if inout > 1.0 + 1e-3 {
            inout_above = true;
        }
        if inout < -1e-3 {
            inout_below = true;
        }
    }

    assert!(
        out_above,
        "EaseOutElastic should exceed 1.0 somewhere in (0,1)"
    );
    assert!(
        in_below,
        "EaseInElastic should fall below 0.0 somewhere in (0,1)"
    );
    assert!(
        inout_above && inout_below,
        "EaseInOutElastic should exceed both bounds (above={inout_above}, below={inout_below})"
    );
}

#[test]
fn spring_and_bezier_stay_clamped() {
    // Both internally `.clamp(0.0, 1.0)` their output, so no sample may leave
    // the unit interval (within float tolerance).
    let clamped = [
        Easing::Spring {
            mass: 1.0,
            stiffness: 100.0,
            damping: 10.0,
        },
        Easing::CubicBezier(0.25, 0.1, 0.25, 1.0),
    ];
    for e in clamped.iter() {
        for k in 0..=100 {
            let t = f64::from(k) / 100.0;
            let v = e.apply(t);
            assert!(
                (-EPS..=1.0 + EPS).contains(&v),
                "{e:?} out of [0,1] at t={t}: {v}",
            );
        }
    }
}

#[test]
fn bounce_variants_in_range_and_anchored() {
    // The bounce curves decelerate via successive parabolic arcs that never
    // exceed [0, 1]; reassert the anchors here too.
    let bounces = [
        Easing::EaseInBounce,
        Easing::EaseOutBounce,
        Easing::EaseInOutBounce,
    ];
    for e in bounces.iter() {
        assert!(e.apply(0.0).abs() < EPS, "{e:?}@0 expected ~0.0");
        assert!((e.apply(1.0) - 1.0).abs() < EPS, "{e:?}@1 expected ~1.0");
        for k in 0..=100 {
            let t = f64::from(k) / 100.0;
            let v = e.apply(t);
            assert!(
                (-EPS..=1.0 + EPS).contains(&v),
                "{e:?} out of [0,1] at t={t}: {v}",
            );
        }
    }
}
