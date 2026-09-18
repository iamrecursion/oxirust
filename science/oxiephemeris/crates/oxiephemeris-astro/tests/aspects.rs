//! Tests for `oxiephemeris_astro::aspects`: exact angle wrapping, orb
//! boundaries, and the applying/separating truth table (Wave B
//! `aspects-ayanamsha` track).

use oxiephemeris_astro::aspects::{find_aspect, Aspect, AspectHit, AspectKind, OrbPolicy};
use proptest::prelude::*;

/// Simple string-error result type so `#[test]` functions can use `?`
/// instead of `.unwrap()`/`.expect()` (denied workspace-wide).
type TestResult = Result<(), String>;

const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;

/// Degrees to radians, for readable test literals.
const fn deg(x: f64) -> f64 {
    x * DEG_TO_RAD
}

/// `find_aspect`, converting a `None` result into a `TestResult` error
/// instead of panicking.
fn must_find(
    lon1_rad: f64,
    speed1: f64,
    lon2_rad: f64,
    speed2: f64,
    policy: &OrbPolicy,
    context: &str,
) -> Result<AspectHit, String> {
    find_aspect(lon1_rad, speed1, lon2_rad, speed2, policy)
        .ok_or_else(|| format!("{context}: expected a match, got None"))
}

// ---------------------------------------------------------------------
// Exact wrap cases
// ---------------------------------------------------------------------

#[test]
fn conjunction_wraps_359_vs_1_degree() -> TestResult {
    // 359 degrees and 1 degree are 2 degrees apart, not 358 -- the raw
    // difference (358 deg) must wrap to -2 deg (or equivalently the
    // separation must reduce to 2 deg), not be read as a near-opposite
    // separation.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(359.0), 0.0, deg(1.0), 0.0, &policy, "359 vs 1 deg")?;
    if hit.aspect.kind != AspectKind::Conjunction {
        return Err(format!("expected conjunction, got {:?}", hit.aspect.kind));
    }
    if (hit.offset_rad.abs() - deg(2.0)).abs() >= 1e-9 {
        return Err(format!("offset {} != +-2 deg", hit.offset_rad));
    }
    Ok(())
}

#[test]
fn conjunction_wraps_179_vs_negative_179_degree() -> TestResult {
    // 179 deg and -179 deg (= 181 deg) are 2 degrees apart (179 -> 180
    // -> 181), not 358: exercises the wrap with a body given as a
    // *negative* longitude.
    let policy = OrbPolicy::default();
    let hit = must_find(
        deg(179.0),
        0.0,
        deg(-179.0),
        0.0,
        &policy,
        "179 vs -179 deg",
    )?;
    if hit.aspect.kind != AspectKind::Conjunction {
        return Err(format!("expected conjunction, got {:?}", hit.aspect.kind));
    }
    if (hit.offset_rad.abs() - deg(2.0)).abs() >= 1e-9 {
        return Err(format!("offset {} != +-2 deg", hit.offset_rad));
    }
    Ok(())
}

#[test]
fn opposition_still_matches_across_the_wrap_boundary() -> TestResult {
    // 1 degree and 182 degrees: raw difference -181 deg wraps to +179
    // deg, separation 179 deg -- just inside the default 8 deg
    // opposition orb (offset 1 deg).
    let policy = OrbPolicy::default();
    let hit = must_find(deg(1.0), 0.0, deg(182.0), 0.0, &policy, "1 vs 182 deg")?;
    if hit.aspect.kind != AspectKind::Opposition {
        return Err(format!("expected opposition, got {:?}", hit.aspect.kind));
    }
    if (hit.offset_rad.abs() - deg(1.0)).abs() >= 1e-9 {
        return Err(format!("offset {} != +-1 deg", hit.offset_rad));
    }
    Ok(())
}

#[test]
fn aspect_just_inside_orb_matches() -> TestResult {
    // Square exact = 90 deg, default orb = 7 deg. 96.9 deg separation:
    // offset 6.9 deg, inside orb.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(96.9), 0.0, deg(0.0), 0.0, &policy, "96.9 deg")?;
    if hit.aspect.kind != AspectKind::Square {
        return Err(format!("expected square, got {:?}", hit.aspect.kind));
    }
    if (hit.offset_rad.abs() - deg(6.9)).abs() >= 1e-9 {
        return Err(format!("offset {} != +-6.9 deg", hit.offset_rad));
    }
    Ok(())
}

#[test]
fn aspect_just_outside_orb_matches_nothing() {
    // 97.1 deg separation: 7.1 deg past square (orb 7), and far from
    // every other aspect's exact angle within its own orb, so no
    // aspect should match at all.
    let policy = OrbPolicy::default();
    let hit = find_aspect(deg(97.1), 0.0, deg(0.0), 0.0, &policy);
    assert!(hit.is_none(), "expected no match, got {hit:?}");
}

#[test]
fn all_aspect_consts_have_the_documented_exact_angles() {
    let expected_deg = [
        (AspectKind::Conjunction, 0.0),
        (AspectKind::Semisextile, 30.0),
        (AspectKind::Semisquare, 45.0),
        (AspectKind::Sextile, 60.0),
        (AspectKind::Quintile, 72.0),
        (AspectKind::Square, 90.0),
        (AspectKind::Trine, 120.0),
        (AspectKind::Sesquiquadrate, 135.0),
        (AspectKind::Biquintile, 144.0),
        (AspectKind::Quincunx, 150.0),
        (AspectKind::Opposition, 180.0),
    ];
    for (kind, expected) in expected_deg {
        let aspect = Aspect::for_kind(kind);
        assert_eq!(aspect.kind, kind);
        assert!(
            (aspect.exact_angle_rad - deg(expected)).abs() < 1e-12,
            "{kind:?}: {} != {expected} deg",
            aspect.exact_angle_rad / DEG_TO_RAD
        );
    }
}

// ---------------------------------------------------------------------
// Applying / separating truth table (hand-computed)
// ---------------------------------------------------------------------
//
// Rule under test (see `aspects.rs` module docs for the derivation):
// `applying = offset * (speed1 - speed2) < 0.0`, i.e. the orb
// `|offset|` is shrinking.
//
// All cases below are near conjunction (exact = 0) unless noted, with
// lon1 always the larger (or equal) raw longitude so `offset > 0`;
// hand-computed via `d(gap)/dt` where `gap = lon1 - lon2`:

#[test]
fn applying_direct_direct_gap_closing() -> TestResult {
    // lon1=2 deg (speed 0.3 deg/day, direct), lon2=0 deg (speed 1.0
    // deg/day, direct, faster): gap(t) = 2 + (0.3 - 1.0) t = 2 - 0.7 t
    // -> shrinking -> applying.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(2.0), deg(0.3), deg(0.0), deg(1.0), &policy, "case")?;
    if hit.aspect.kind != AspectKind::Conjunction {
        return Err(format!("expected conjunction, got {:?}", hit.aspect.kind));
    }
    if !hit.applying {
        return Err("gap 2 - 0.7t shrinks: must be applying".into());
    }
    Ok(())
}

#[test]
fn separating_direct_direct_gap_opening() -> TestResult {
    // lon1=10 deg (speed 1.0, direct), lon2=5 deg (speed 0.5, direct,
    // slower): gap(t) = 5 + (1.0 - 0.5) t = 5 + 0.5 t -> growing ->
    // separating.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(10.0), deg(1.0), deg(5.0), deg(0.5), &policy, "case")?;
    if hit.aspect.kind != AspectKind::Conjunction {
        return Err(format!("expected conjunction, got {:?}", hit.aspect.kind));
    }
    if hit.applying {
        return Err("gap 5 + 0.5t grows: must be separating".into());
    }
    Ok(())
}

#[test]
fn separating_is_swap_invariant_while_offset_flips() -> TestResult {
    // The exact same physical configuration as
    // `separating_direct_direct_gap_opening`, with the two bodies'
    // labels swapped: `applying` must be unchanged (separating is a
    // physical fact, not a labeling artifact), while `offset_rad` must
    // flip sign.
    let policy = OrbPolicy::default();
    let original = must_find(deg(10.0), deg(1.0), deg(5.0), deg(0.5), &policy, "original")?;
    let swapped = must_find(deg(5.0), deg(0.5), deg(10.0), deg(1.0), &policy, "swapped")?;
    if original.applying != swapped.applying {
        return Err("applying must be swap-invariant".into());
    }
    if original.applying {
        return Err("expected separating".into());
    }
    if (original.offset_rad + swapped.offset_rad).abs() >= 1e-9 {
        return Err(format!(
            "offsets {} and {} do not cancel",
            original.offset_rad, swapped.offset_rad
        ));
    }
    Ok(())
}

#[test]
fn applying_retrograde_body_versus_direct_body() -> TestResult {
    // lon1=2 deg (speed -0.5 deg/day, RETROGRADE), lon2=0 deg (speed
    // 1.0 deg/day, direct): gap(t) = 2 + (-0.5 - 1.0) t = 2 - 1.5 t ->
    // shrinking fast -> applying. Retrograde motion needs no special
    // case: it is just a negative speed in the same linear rule.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(2.0), deg(-0.5), deg(0.0), deg(1.0), &policy, "case")?;
    if !hit.applying {
        return Err("gap 2 - 1.5t shrinks: must be applying".into());
    }
    Ok(())
}

#[test]
fn separating_both_bodies_retrograde() -> TestResult {
    // lon1=2 deg (speed -0.3, retrograde), lon2=0 deg (speed -1.0,
    // retrograde and faster): gap(t) = 2 + (-0.3 - (-1.0)) t
    // = 2 + 0.7 t -> growing -> separating, even though both bodies
    // move backward in absolute terms.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(2.0), deg(-0.3), deg(0.0), deg(-1.0), &policy, "case")?;
    if hit.applying {
        return Err("gap 2 + 0.7t grows: must be separating".into());
    }
    Ok(())
}

#[test]
fn applying_both_bodies_retrograde() -> TestResult {
    // lon1=2 deg (speed -1.0, retrograde and faster), lon2=0 deg
    // (speed -0.3, retrograde): gap(t) = 2 + (-1.0 - (-0.3)) t
    // = 2 - 0.7 t -> shrinking -> applying.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(2.0), deg(-1.0), deg(0.0), deg(-0.3), &policy, "case")?;
    if !hit.applying {
        return Err("gap 2 - 0.7t shrinks: must be applying".into());
    }
    Ok(())
}

#[test]
fn applying_toward_opposition() -> TestResult {
    // lon1=175 deg (speed 1.0 deg/day, direct), lon2=0 deg (speed 0):
    // gap(t) = 175 + t, increasing toward the 180 deg exact
    // opposition (offset starts at -5 deg and shrinks in magnitude)
    // -> applying.
    let policy = OrbPolicy::default();
    let hit = must_find(deg(175.0), deg(1.0), deg(0.0), 0.0, &policy, "case")?;
    if hit.aspect.kind != AspectKind::Opposition {
        return Err(format!("expected opposition, got {:?}", hit.aspect.kind));
    }
    if !hit.applying {
        return Err("gap approaching 180 deg from below: applying".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Proptest: symmetry and orb invariants
// ---------------------------------------------------------------------

proptest! {
    /// Swapping the two bodies flips `offset_rad`'s sign and leaves the
    /// matched aspect (and `applying`) unchanged; `|offset_rad|` never
    /// exceeds the configured orb.
    #[test]
    fn find_aspect_symmetric_under_swap(
        lon1 in -20.0f64..20.0,
        lon2 in -20.0f64..20.0,
        speed1 in -3.0f64..3.0,
        speed2 in -3.0f64..3.0,
    ) {
        let policy = OrbPolicy::default();
        let direct = find_aspect(lon1, speed1, lon2, speed2, &policy);
        let swapped = find_aspect(lon2, speed2, lon1, speed1, &policy);

        match (direct, swapped) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                prop_assert_eq!(a.aspect.kind, b.aspect.kind);
                // Skip the (measure-zero) instant right at the exact
                // aspect angle or at the raw +-pi branch cut, where
                // the sign of a near-zero offset is not numerically
                // stable.
                prop_assume!(a.offset_rad.abs() > 1e-6);
                prop_assert!(
                    (a.offset_rad + b.offset_rad).abs() < 1e-9,
                    "offsets {} and {} do not cancel",
                    a.offset_rad,
                    b.offset_rad
                );
                prop_assert_eq!(a.applying, b.applying);
                let orb = policy.orb_rad(a.aspect.kind);
                prop_assert!(a.offset_rad.abs() <= orb + 1e-12);
                prop_assert!(b.offset_rad.abs() <= orb + 1e-12);
            }
            (a, b) => {
                prop_assert!(false, "asymmetric match: {a:?} vs {b:?}");
            }
        }
    }

    /// Whenever `find_aspect` returns a hit, `|offset_rad|` is at most
    /// the policy's orb for that aspect kind.
    #[test]
    fn offset_never_exceeds_orb(
        lon1 in -50.0f64..50.0,
        lon2 in -50.0f64..50.0,
        speed1 in -5.0f64..5.0,
        speed2 in -5.0f64..5.0,
    ) {
        let policy = OrbPolicy::default();
        if let Some(hit) = find_aspect(lon1, speed1, lon2, speed2, &policy) {
            let orb = policy.orb_rad(hit.aspect.kind);
            prop_assert!(hit.offset_rad.abs() <= orb + 1e-12);
        }
    }
}
