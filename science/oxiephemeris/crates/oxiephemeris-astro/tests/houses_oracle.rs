//! Independent brute-force oracle for the house systems.
//!
//! Every cusp returned by [`oxiephemeris_astro::houses::cusps`] is
//! verified against its *defining condition*, computed from scratch with
//! spherical trigonometry that does not reuse the implementation's
//! closed forms or iterations:
//!
//! - **Placidus** — from the returned longitude, derive `(α, δ)` via the
//!   equatorial unit vector, the semi-diurnal arc
//!   `SDA = acos(−tan φ tan δ)` and the hour angle `H = θ − α`, and
//!   assert the trisection ratio of the point's own semi-arc.
//! - **Koch** — assert the *rising condition*: at the shifted sidereal
//!   time the cusp sits on the horizon (altitude 0) in the east.
//! - **Regiomontanus / Campanus** — rebuild the house circle through the
//!   north/south points of the horizon and the equator (resp. prime
//!   vertical) division point with rotation matrices, and assert the
//!   cusp lies in its plane (normal dot product `< 1e-10`).
//! - **Porphyry / Equal / Whole Sign** — assert the arithmetic
//!   definitions.
//!
//! Plus, for all quadrant systems: cusp 1/10 bit-identical to the
//! ascendant/MC, opposite cusps antipodal, cusps strictly increasing
//! once around the zodiac (winding number one), the documented
//! Placidus/Koch polar-circle errors, and proptest randomization over
//! `(θ, φ, ε)`.

use core::f64::consts::{FRAC_PI_2, FRAC_PI_6, PI};
use oxiephemeris_astro::angles::{ascendant, mc};
use oxiephemeris_astro::houses::{cusps, HouseSystem, HousesError};
use oxiephemeris_bodies::math::{cross, dot, norm, r2, r3, Mat3, Vec3};
use oxiephemeris_core::angle::normalize_pm_pi;
use proptest::prelude::*;

const TWO_PI: f64 = 2.0 * PI;
const DEG: f64 = PI / 180.0;
const EPS_TEST: f64 = 23.4367 * DEG;

/// Grid latitudes (degrees). All below the 66.5633° polar circle of
/// `EPS_TEST`; 66° deliberately stresses the Placidus/Koch solvers.
const PHIS_DEG: [f64; 7] = [-60.0, -35.0, 0.0, 20.0, 35.0, 51.5, 66.0];

/// Grid sidereal times (degrees), mixing exact quadrant boundaries with
/// offset values.
const THETAS_DEG: [f64; 8] = [0.0, 45.0, 93.7, 138.7, 180.0, 228.7, 270.0, 317.3];

const QUADRANT_SYSTEMS: [HouseSystem; 5] = [
    HouseSystem::Placidus,
    HouseSystem::Koch,
    HouseSystem::Porphyry,
    HouseSystem::Regiomontanus,
    HouseSystem::Campanus,
];

const ALL_SYSTEMS: [HouseSystem; 7] = [
    HouseSystem::Placidus,
    HouseSystem::Koch,
    HouseSystem::WholeSign,
    HouseSystem::Equal,
    HouseSystem::Porphyry,
    HouseSystem::Regiomontanus,
    HouseSystem::Campanus,
];

fn wheel(system: HouseSystem, theta: f64, phi: f64, eps: f64) -> [f64; 12] {
    match cusps(system, theta, phi, eps) {
        Ok(c) => c,
        Err(e) => panic!("cusps({system:?}) failed at theta={theta}, phi={phi}: {e}"),
    }
}

fn ok_angle<E: std::fmt::Display>(r: Result<f64, E>) -> f64 {
    match r {
        Ok(v) => v,
        Err(e) => panic!("unexpected angles error: {e}"),
    }
}

/// Smallest angular distance between two directions, radians.
fn wrap_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(TWO_PI);
    if d > PI {
        TWO_PI - d
    } else {
        d
    }
}

/// Equatorial unit vector of the ecliptic point of longitude `lambda`.
fn ecl_unit(lambda: f64, eps: f64) -> Vec3 {
    [
        lambda.cos(),
        lambda.sin() * eps.cos(),
        lambda.sin() * eps.sin(),
    ]
}

/// Zenith direction (equatorial frame).
fn zenith(theta: f64, phi: f64) -> Vec3 {
    [phi.cos() * theta.cos(), phi.cos() * theta.sin(), phi.sin()]
}

/// East point of the horizon (equatorial frame).
fn east_dir(theta: f64) -> Vec3 {
    [-theta.sin(), theta.cos(), 0.0]
}

/// Right ascension and declination of an ecliptic longitude — via the
/// Cartesian vector, not the implementation's `tan δ = tan ε sin α`.
fn ra_dec(lambda: f64, eps: f64) -> (f64, f64) {
    let p = ecl_unit(lambda, eps);
    let ra = p[1].atan2(p[0]);
    let dec = p[2].atan2((p[0] * p[0] + p[1] * p[1]).sqrt());
    (ra, dec)
}

/// Semi-diurnal arc of declination `dec` at latitude `phi`, if the
/// point rises and sets at all.
fn semi_diurnal_arc(phi: f64, dec: f64) -> Option<f64> {
    let cos_h0 = -phi.tan() * dec.tan();
    if cos_h0.abs() <= 1.0 {
        Some(cos_h0.acos())
    } else {
        None
    }
}

/// Equatorial → horizon (south, east, zenith) rotation.
fn eq_to_hor(theta: f64, phi: f64) -> Mat3 {
    r2(FRAC_PI_2 - phi).mul(&r3(theta))
}

/// Invariants shared by every quadrant system: cusp 1 / cusp 10 are the
/// ascendant / MC *bit-identically*, and opposite cusps are antipodal.
fn check_quadrant_invariants(c: &[f64; 12], theta: f64, phi: f64, eps: f64, tag: HouseSystem) {
    let asc = ok_angle(ascendant(theta, phi, eps));
    assert_eq!(c[0].to_bits(), asc.to_bits(), "{tag:?}: cusp 1 != ASC");
    assert_eq!(
        c[9].to_bits(),
        mc(theta, eps).to_bits(),
        "{tag:?}: cusp 10 != MC"
    );
    for (i, (lower, upper)) in c[..6].iter().zip(&c[6..]).enumerate() {
        assert!(
            wrap_diff(*upper, lower + PI) < 1e-12,
            "{tag:?}: cusp {} not opposite cusp {}",
            i + 7,
            i + 1
        );
    }
}

/// Cusps must run strictly forward around the zodiac exactly once.
fn check_strictly_increasing(c: &[f64; 12], tag: HouseSystem) {
    let mut total = 0.0;
    for (i, (current, next)) in c.iter().zip(c.iter().cycle().skip(1)).take(12).enumerate() {
        let inc = (next - current).rem_euclid(TWO_PI);
        assert!(
            inc > 1e-9 && inc < TWO_PI - 1e-9,
            "{tag:?}: cusp {} -> {} increment {inc:e} not strictly forward",
            i + 1,
            (i + 1) % 12 + 1
        );
        total += inc;
    }
    assert!(
        (total - TWO_PI).abs() < 1e-9,
        "{tag:?}: winding {total} != 2 pi"
    );
}

/// Placidus oracle: the defining semi-arc trisection, from scratch.
/// Above the horizon `H = −f·SDA` (cusps 11, 12); below it
/// `H = −(SDA + f·SNA)` with `SNA = π − SDA` (cusps 2, 3).
fn check_placidus(c: &[f64; 12], theta: f64, phi: f64, eps: f64) {
    let cases: [(usize, f64, bool); 4] = [
        (10, 1.0 / 3.0, true), // cusp 11
        (11, 2.0 / 3.0, true), // cusp 12
        (1, 1.0 / 3.0, false), // cusp 2
        (2, 2.0 / 3.0, false), // cusp 3
    ];
    for (idx, frac, diurnal) in cases {
        let (ra, dec) = ra_dec(c[idx], eps);
        let Some(sda) = semi_diurnal_arc(phi, dec) else {
            panic!("Placidus cusp {} circumpolar at phi={phi}", idx + 1);
        };
        let hour_angle = normalize_pm_pi(theta - ra);
        let target = if diurnal {
            -frac * sda
        } else {
            -(sda + frac * (PI - sda))
        };
        let residual = normalize_pm_pi(hour_angle - target).abs();
        assert!(
            residual < 1e-9,
            "Placidus cusp {} defining ratio off by {residual:e} \
             (phi={phi}, theta={theta})",
            idx + 1
        );
    }
}

/// Koch oracle: each intermediate cusp must be *rising* — on the
/// horizon, in the east — when the meridian has turned by k/3 of the MC
/// degree's semi-diurnal arc.
fn check_koch(c: &[f64; 12], theta: f64, phi: f64, eps: f64) {
    let (_, dec_mc) = ra_dec(c[9], eps);
    let Some(h0) = semi_diurnal_arc(phi, dec_mc) else {
        panic!("Koch: MC degree circumpolar at phi={phi}");
    };
    let cases: [(usize, f64); 5] = [
        (10, -2.0), // cusp 11 rises 2/3 H0 of sidereal turn before birth
        (11, -1.0), // cusp 12
        (0, 0.0),   // cusp 1: the ascendant itself
        (1, 1.0),   // cusp 2
        (2, 2.0),   // cusp 3
    ];
    for (idx, k) in cases {
        let shifted = theta + k * h0 / 3.0;
        let p = ecl_unit(c[idx], eps);
        let altitude_sin = dot(p, zenith(shifted, phi));
        assert!(
            altitude_sin.abs() < 1e-9,
            "Koch cusp {} not on horizon at shifted time (phi={phi}, theta={theta}): {altitude_sin:e}",
            idx + 1
        );
        assert!(
            dot(p, east_dir(shifted)) > 0.0,
            "Koch cusp {} not rising in the east (phi={phi}, theta={theta})",
            idx + 1
        );
    }
    // The ladder closes: one more third-step past cusp 3 reaches the IC.
    let ic_rising = ok_angle(ascendant(theta + h0, phi, eps));
    assert!(
        wrap_diff(ic_rising, c[3]) < 1e-9,
        "Koch ladder does not close on the IC (phi={phi}, theta={theta})"
    );
}

/// Regiomontanus oracle: the house circle through the horizon's
/// north/south points (the x-axis of the horizon frame) and the equator
/// point 30°·k east of the meridian must contain the cusp.
fn check_regiomontanus(c: &[f64; 12], theta: f64, phi: f64, eps: f64, check_side: bool) {
    let m = eq_to_hor(theta, phi);
    let cases: [(usize, i32); 6] = [(9, 0), (10, 1), (11, 2), (0, 3), (1, 4), (2, 5)];
    for (idx, k) in cases {
        let rho = theta + f64::from(k) * FRAC_PI_6;
        let division = m.apply([rho.cos(), rho.sin(), 0.0]);
        let normal = cross([1.0, 0.0, 0.0], division);
        let normal_len = norm(normal);
        assert!(normal_len > 1e-3, "degenerate house circle k={k}");
        let p = m.apply(ecl_unit(c[idx], eps));
        let off_plane = dot(p, normal).abs() / normal_len;
        assert!(
            off_plane < 1e-10,
            "Regiomontanus cusp {} off its house circle by {off_plane:e} \
             (phi={phi}, theta={theta})",
            idx + 1
        );
        if check_side {
            assert!(
                dot(p, division) > 0.0,
                "Regiomontanus cusp {} on the wrong branch (phi={phi}, theta={theta})",
                idx + 1
            );
        }
    }
}

/// Campanus oracle: same construction, but the division points sit on
/// the prime vertical, 30°·k from the zenith toward the east — written
/// directly in horizon-frame coordinates `(0, sin a, cos a)`.
fn check_campanus(c: &[f64; 12], theta: f64, phi: f64, eps: f64, check_side: bool) {
    let m = eq_to_hor(theta, phi);
    let cases: [(usize, i32); 6] = [(9, 0), (10, 1), (11, 2), (0, 3), (1, 4), (2, 5)];
    for (idx, k) in cases {
        let a = f64::from(k) * FRAC_PI_6;
        let division = [0.0, a.sin(), a.cos()];
        let normal = cross([1.0, 0.0, 0.0], division);
        let p = m.apply(ecl_unit(c[idx], eps));
        let off_plane = dot(p, normal).abs();
        assert!(
            off_plane < 1e-10,
            "Campanus cusp {} off its house circle by {off_plane:e} \
             (phi={phi}, theta={theta})",
            idx + 1
        );
        if check_side {
            assert!(
                dot(p, division) > 0.0,
                "Campanus cusp {} on the wrong branch (phi={phi}, theta={theta})",
                idx + 1
            );
        }
    }
}

/// Porphyry oracle: trisection of the zodiacal ASC–MC quadrant arcs.
fn check_porphyry(c: &[f64; 12], theta: f64, phi: f64, eps: f64) {
    let asc = ok_angle(ascendant(theta, phi, eps));
    let mc_val = mc(theta, eps);
    let quadrant = (asc - mc_val).rem_euclid(TWO_PI);
    assert!(
        wrap_diff(c[10], mc_val + quadrant / 3.0) < 1e-12,
        "Porphyry cusp 11"
    );
    assert!(
        wrap_diff(c[11], mc_val + 2.0 * quadrant / 3.0) < 1e-12,
        "Porphyry cusp 12"
    );
    let below = (mc_val + PI - asc).rem_euclid(TWO_PI);
    assert!(
        wrap_diff(c[1], asc + below / 3.0) < 1e-12,
        "Porphyry cusp 2"
    );
    assert!(
        wrap_diff(c[2], asc + 2.0 * below / 3.0) < 1e-12,
        "Porphyry cusp 3"
    );
}

/// Equal-house oracle: twelve exact 30° steps from the ascendant.
fn check_equal(c: &[f64; 12], theta: f64, phi: f64, eps: f64) {
    let asc = ok_angle(ascendant(theta, phi, eps));
    for (i, cusp) in c.iter().enumerate() {
        let expected = asc + f64::from(i32::try_from(i).unwrap_or(0)) * FRAC_PI_6;
        assert!(
            wrap_diff(*cusp, expected) < 1e-12,
            "Equal cusp {} off its 30-degree step",
            i + 1
        );
    }
}

/// Whole-sign oracle: cusp 1 is 0° of the sign containing the
/// ascendant, then 30° steps.
fn check_whole_sign(c: &[f64; 12], theta: f64, phi: f64, eps: f64) {
    let asc = ok_angle(ascendant(theta, phi, eps));
    let offset = (asc - c[0]).rem_euclid(TWO_PI);
    assert!(
        offset < FRAC_PI_6 + 1e-12,
        "WholeSign: ASC not inside its cusp-1 sign (offset {offset})"
    );
    let sign_index = (c[0] / FRAC_PI_6).round();
    assert!(
        (c[0] - sign_index * FRAC_PI_6).abs() < 1e-9,
        "WholeSign: cusp 1 not a sign boundary"
    );
    for (i, cusp) in c.iter().enumerate() {
        let expected = c[0] + f64::from(i32::try_from(i).unwrap_or(0)) * FRAC_PI_6;
        assert!(
            wrap_diff(*cusp, expected) < 1e-12,
            "WholeSign cusp {} off its 30-degree step",
            i + 1
        );
    }
}

#[test]
fn oracle_grid_quadrant_systems() {
    for &phi_deg in &PHIS_DEG {
        for &theta_deg in &THETAS_DEG {
            let (theta, phi) = (theta_deg * DEG, phi_deg * DEG);
            for system in QUADRANT_SYSTEMS {
                let c = wheel(system, theta, phi, EPS_TEST);
                check_quadrant_invariants(&c, theta, phi, EPS_TEST, system);
                check_strictly_increasing(&c, system);
                match system {
                    HouseSystem::Placidus => check_placidus(&c, theta, phi, EPS_TEST),
                    HouseSystem::Koch => check_koch(&c, theta, phi, EPS_TEST),
                    HouseSystem::Porphyry => check_porphyry(&c, theta, phi, EPS_TEST),
                    HouseSystem::Regiomontanus => {
                        check_regiomontanus(&c, theta, phi, EPS_TEST, true);
                    }
                    HouseSystem::Campanus => check_campanus(&c, theta, phi, EPS_TEST, true),
                    // `#[non_exhaustive]`-safe fallback: `system` only ever
                    // comes from `QUADRANT_SYSTEMS` above, so every arm
                    // reaching here (WholeSign, Equal, or any future
                    // variant) is a test bug, not a real case.
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn oracle_grid_fixed_step_systems() {
    for &phi_deg in &PHIS_DEG {
        for &theta_deg in &THETAS_DEG {
            let (theta, phi) = (theta_deg * DEG, phi_deg * DEG);
            check_equal(
                &wheel(HouseSystem::Equal, theta, phi, EPS_TEST),
                theta,
                phi,
                EPS_TEST,
            );
            check_whole_sign(
                &wheel(HouseSystem::WholeSign, theta, phi, EPS_TEST),
                theta,
                phi,
                EPS_TEST,
            );
        }
    }
}

#[test]
fn polar_circle_and_pole_errors() {
    let phi67 = 67.0 * DEG;
    let phi80 = 80.0 * DEG;
    for &theta_deg in &THETAS_DEG {
        let theta = theta_deg * DEG;
        // Placidus/Koch: hard error above the polar circle, both signs.
        for system in [HouseSystem::Placidus, HouseSystem::Koch] {
            assert_eq!(
                cusps(system, theta, phi67, EPS_TEST),
                Err(HousesError::Undefined)
            );
            assert_eq!(
                cusps(system, theta, -phi67, EPS_TEST),
                Err(HousesError::Undefined)
            );
        }
        // The other systems stay defined at high latitude; the geometric
        // oracle must still hold there (branch choice not asserted).
        check_regiomontanus(
            &wheel(HouseSystem::Regiomontanus, theta, phi80, EPS_TEST),
            theta,
            phi80,
            EPS_TEST,
            false,
        );
        check_campanus(
            &wheel(HouseSystem::Campanus, theta, phi80, EPS_TEST),
            theta,
            phi80,
            EPS_TEST,
            false,
        );
        check_porphyry(
            &wheel(HouseSystem::Porphyry, theta, phi80, EPS_TEST),
            theta,
            phi80,
            EPS_TEST,
        );
        check_equal(
            &wheel(HouseSystem::Equal, theta, phi80, EPS_TEST),
            theta,
            phi80,
            EPS_TEST,
        );
        check_whole_sign(
            &wheel(HouseSystem::WholeSign, theta, phi80, EPS_TEST),
            theta,
            phi80,
            EPS_TEST,
        );
    }
    // Exact poles: every system refuses.
    for system in ALL_SYSTEMS {
        assert_eq!(
            cusps(system, 1.0, FRAC_PI_2, EPS_TEST),
            Err(HousesError::PolarLatitude)
        );
        assert_eq!(
            cusps(system, 1.0, -FRAC_PI_2, EPS_TEST),
            Err(HousesError::PolarLatitude)
        );
    }
    // Ecliptic-horizon coincidence at the polar circle: Undefined.
    assert_eq!(
        cusps(
            HouseSystem::Equal,
            3.0 * FRAC_PI_2,
            FRAC_PI_2 - EPS_TEST,
            EPS_TEST
        ),
        Err(HousesError::Undefined)
    );
    // Non-finite inputs.
    assert_eq!(
        cusps(HouseSystem::Placidus, f64::NAN, 0.5, EPS_TEST),
        Err(HousesError::NonFiniteInput)
    );
    assert_eq!(
        cusps(HouseSystem::Campanus, 1.0, f64::INFINITY, EPS_TEST),
        Err(HousesError::NonFiniteInput)
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Random charts in the safe latitude band: every quadrant system
    /// must satisfy the shared invariants and its own oracle.
    #[test]
    fn proptest_quadrant_invariants(
        theta in 0.0..TWO_PI,
        phi_deg in -60.0..60.0_f64,
        eps_deg in 23.0..23.7_f64,
    ) {
        let phi = phi_deg * DEG;
        let eps = eps_deg * DEG;
        let asc = ascendant(theta, phi, eps)
            .map_err(|e| TestCaseError::fail(format!("asc: {e}")))?;
        let mc_val = mc(theta, eps);
        for system in QUADRANT_SYSTEMS {
            let c = cusps(system, theta, phi, eps)
                .map_err(|e| TestCaseError::fail(format!("{system:?}: {e}")))?;
            prop_assert_eq!(c[0].to_bits(), asc.to_bits());
            prop_assert_eq!(c[9].to_bits(), mc_val.to_bits());
            for (lower, upper) in c[..6].iter().zip(&c[6..]) {
                prop_assert!(wrap_diff(*upper, lower + PI) < 1e-12);
            }
            let mut total = 0.0;
            for (current, next) in c.iter().zip(c.iter().cycle().skip(1)).take(12) {
                let inc = (next - current).rem_euclid(TWO_PI);
                prop_assert!(inc > 1e-9, "{:?}: non-increasing cusp pair", system);
                total += inc;
            }
            prop_assert!((total - TWO_PI).abs() < 1e-9, "{:?}: winding {}", system, total);
        }
    }

    /// Random charts: Equal / Whole Sign are exact 30° ladders anchored
    /// to the ascendant / its sign.
    #[test]
    fn proptest_fixed_step_systems(
        theta in 0.0..TWO_PI,
        phi_deg in -60.0..60.0_f64,
        eps_deg in 23.0..23.7_f64,
    ) {
        let phi = phi_deg * DEG;
        let eps = eps_deg * DEG;
        let asc = ascendant(theta, phi, eps)
            .map_err(|e| TestCaseError::fail(format!("asc: {e}")))?;
        let equal = cusps(HouseSystem::Equal, theta, phi, eps)
            .map_err(|e| TestCaseError::fail(format!("equal: {e}")))?;
        prop_assert_eq!(equal[0].to_bits(), asc.to_bits());
        let whole = cusps(HouseSystem::WholeSign, theta, phi, eps)
            .map_err(|e| TestCaseError::fail(format!("whole: {e}")))?;
        prop_assert!((asc - whole[0]).rem_euclid(TWO_PI) < FRAC_PI_6 + 1e-12);
        for c in [&equal, &whole] {
            for (current, next) in c.iter().zip(c.iter().cycle().skip(1)).take(12) {
                let inc = (next - current).rem_euclid(TWO_PI);
                prop_assert!((inc - FRAC_PI_6).abs() < 1e-12);
            }
        }
    }
}
