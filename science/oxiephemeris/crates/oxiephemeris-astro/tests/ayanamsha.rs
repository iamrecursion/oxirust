//! Tests for `oxiephemeris_astro::ayanamsha` (Wave B `aspects-ayanamsha`
//! track).
//!
//! # On the choice of "published figure" and tolerance
//!
//! This crate's mandate forbids calibrating against Swiss-Ephemeris-
//! derived numbers (even indirectly, via a third-party site that
//! itself ran SE) — the clean-room boundary treats *any* SE output as
//! an SE artifact, not just its source code. Many popular "ayanamsha
//! value tables" found online turn out, on inspection, to be exactly
//! that (several explicitly say so). So rather than hand-picking a
//! single decimal figure from such a table, these tests cross-check
//! this crate's anchor-plus-`p_A` model against the two things that
//! *are* independently and repeatedly published, free of that taint:
//!
//! 1. The **historical anchor decrees themselves** (the 1956 Lahiri
//!    decree, the Fagan/Bradley SVP definition at B1950.0, the KP
//!    table's 22 deg 21' 50" at 1900.0, Raman's own 21 deg 11' 29"
//!    for 1912 from *Hindu Predictive Astrology* ch. X) — see the
//!    citations in `ayanamsha.rs`.
//! 2. The **near-universally cited precession rate**, "~50.29
//!    arcsec/year" (stated in this crate's task brief too, and
//!    repeated across independent astronomy/astrology references) —
//!    used here as a simple, independent linear extrapolation from the
//!    anchor, computed from scratch in each test (not by calling this
//!    crate's own `p_A`), which this crate's fuller nonlinear P03
//!    model should reproduce to a couple of arcseconds even over the
//!    century-long spans tested (the P03 `t^2` term contributes
//!    ~1.3 arcsec per century away from J2000; confirmed below).
//! 3. The **published inter-system relationships** (KP sits ~6
//!    arcminutes below Lahiri; Raman ~1.4-1.5 deg below Lahiri;
//!    Fagan/Bradley ~53 arcminutes above), stated across independent
//!    astrology references — these are anchor-transposition guards:
//!    a swapped or misattributed anchor value (e.g. a Lahiri-1900
//!    figure planted on Raman) breaks them immediately.
//!
//! Where an independent *non-numeric-table* figure is available (e.g.
//! Cyril Fagan's own published estimate, "about 24 deg 44 min for
//! 1 January 2000" per secondary discussion of his papers), it is used
//! too, at the (coarser, arcminute-level) precision that source
//! actually states — the residual is computed and asserted, not hidden.

use oxiephemeris_astro::ayanamsha::{
    ayanamsha_rad, general_precession_iau2006_rad, sidereal_from_tropical, Ayanamsha,
};

const RAD_TO_ARCSEC: f64 = 180.0 * 3600.0 / core::f64::consts::PI;
const RAD_TO_DEG: f64 = 180.0 / core::f64::consts::PI;
const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;

/// Julian Date (TT) of J2000.0.
const JD_J2000: f64 = 2_451_545.0;
const DAYS_PER_CENTURY: f64 = 36_525.0;

fn centuries(jd_tt: f64) -> f64 {
    (jd_tt - JD_J2000) / DAYS_PER_CENTURY
}

/// JD (0h UT, treated as TT) for a Gregorian civil date, via the
/// standard Fliegel & Van Flandern (1968) integer Julian Day Number
/// algorithm (used only to build test epochs — the crate itself never
/// hardcodes calendar arithmetic here beyond these test-local JDs).
///
/// `jdn` stays in the low millions for every date used in this file,
/// far inside `f64`'s exact-integer range (`2^52`), so the `i64 -> f64`
/// cast below loses no precision in practice.
#[allow(clippy::cast_precision_loss)]
fn jd_civil(y: i64, m: i64, d: i64) -> f64 {
    let a = (14 - m).div_euclid(12);
    let y2 = y + 4800 - a;
    let m2 = m + 12 * a - 3;
    let jdn = d + (153 * m2 + 2).div_euclid(5) + 365 * y2 + y2.div_euclid(4) - y2.div_euclid(100)
        + y2.div_euclid(400)
        - 32045;
    jdn as f64 - 0.5
}

const JD_LAHIRI_ANCHOR: f64 = 2_435_553.5; // 1956-03-21 0h
const JD_B1950_0: f64 = 2_433_282.423_5;

fn deg_to_rad_dms(d: f64, m: f64, s: f64) -> f64 {
    (d + m / 60.0 + s / 3600.0) * DEG_TO_RAD
}

// ---------------------------------------------------------------------
// Anchor-exactness checks (validate JD/century wiring, not external
// data): both anchors, evaluated at their own defining epoch, must
// reproduce the defining value to numerical precision.
// ---------------------------------------------------------------------

#[test]
fn lahiri_reproduces_its_own_1956_decree_value() {
    let t0 = centuries(JD_LAHIRI_ANCHOR);
    let expected = deg_to_rad_dms(23.0, 15.0, 0.0);
    let got = ayanamsha_rad(Ayanamsha::Lahiri, t0);
    assert!(
        (got - expected).abs() < 1e-9,
        "residual {} arcsec",
        (got - expected) * RAD_TO_ARCSEC
    );
}

#[test]
fn fagan_bradley_reproduces_its_own_b1950_svp_definition() {
    let t0 = centuries(JD_B1950_0);
    let expected = deg_to_rad_dms(24.0, 2.0, 31.36);
    let got = ayanamsha_rad(Ayanamsha::FaganBradley, t0);
    assert!(
        (got - expected).abs() < 1e-9,
        "residual {} arcsec",
        (got - expected) * RAD_TO_ARCSEC
    );
}

#[test]
fn krishnamurti_reproduces_its_own_1900_anchor() {
    // KP table anchor: 22 deg 21' 50" at 1900 Jan 1, 0h (the epoch JD
    // is recomputed here via Fliegel & Van Flandern, independently of
    // the crate's own JD constant, so a wrong epoch wiring also fails).
    let t0 = centuries(jd_civil(1900, 1, 1));
    let expected = deg_to_rad_dms(22.0, 21.0, 50.0);
    let got = ayanamsha_rad(Ayanamsha::Krishnamurti, t0);
    assert!(
        (got - expected).abs() < 1e-9,
        "residual {} arcsec",
        (got - expected) * RAD_TO_ARCSEC
    );
}

#[test]
fn raman_reproduces_its_own_1912_table_value() {
    // Raman's own published "Ayanamsa for 1912 ... 21 11 29" (Hindu
    // Predictive Astrology, ch. X), pinned by the crate to 1912 Jan 1,
    // 0h; epoch JD recomputed independently as above.
    let t0 = centuries(jd_civil(1912, 1, 1));
    let expected = deg_to_rad_dms(21.0, 11.0, 29.0);
    let got = ayanamsha_rad(Ayanamsha::Raman, t0);
    assert!(
        (got - expected).abs() < 1e-9,
        "residual {} arcsec",
        (got - expected) * RAD_TO_ARCSEC
    );
}

// ---------------------------------------------------------------------
// Spot checks at J2000 and 1950 (task-required epochs)
// ---------------------------------------------------------------------

#[test]
fn fagan_bradley_spot_check_at_1950_01_01() {
    // The B1950.0 anchor (1949-12-31 ~22:09 UT) and 1950-01-01 0h UT
    // are ~1.8 hours apart, so this is essentially the anchor check
    // again but at the *civil* date the task asks for.
    let jd = jd_civil(1950, 1, 1);
    let t = centuries(jd);
    let published = deg_to_rad_dms(24.0, 2.0, 31.36); // the SVP definition itself
    let got = ayanamsha_rad(Ayanamsha::FaganBradley, t);
    let residual_arcsec = (got - published) * RAD_TO_ARCSEC;
    assert!(
        residual_arcsec.abs() < 2.0,
        "Fagan/Bradley @ 1950-01-01: got {} deg, published-anchor {} deg, residual {residual_arcsec} arcsec",
        got * RAD_TO_DEG,
        published * RAD_TO_DEG
    );
}

#[test]
fn fagan_bradley_spot_check_near_j2000() {
    // Cyril Fagan's own published estimate for 1 Jan 2000 is "about
    // 24 deg 44 min" (arcminute precision only, per secondary
    // discussion of the Fagan/Bradley siderealist literature).
    let jd = jd_civil(2000, 1, 1);
    let t = centuries(jd);
    let published_deg = 24.0 + 44.0 / 60.0;
    let got = ayanamsha_rad(Ayanamsha::FaganBradley, t);
    let residual_arcsec = (got * RAD_TO_DEG - published_deg) * 3600.0;
    // The cited figure is only stated to the nearest arcminute, so we
    // allow the full +-1 arcmin rounding window on each side (120
    // arcsec), not this crate's usual sub-arcsecond target.
    assert!(
        residual_arcsec.abs() < 120.0,
        "Fagan/Bradley @ J2000: got {} deg, published ~24d44m, residual {residual_arcsec} arcsec",
        got * RAD_TO_DEG
    );
}

#[test]
fn lahiri_spot_check_at_1950_01_01_via_independent_linear_rate() {
    // Independent check: the 1956 decree value, extrapolated to
    // 1950-01-01 via the widely-published (and task-cited) constant
    // rate of 50.29 arcsec/year -- NOT via this crate's own p_A, so
    // this is a genuine cross-check of the P03 propagation, not a
    // tautology.
    let jd_1950 = jd_civil(1950, 1, 1);
    let days = jd_1950 - JD_LAHIRI_ANCHOR;
    let years = days / 365.25;
    let published_rate_deg_per_year = 50.29 / 3600.0;
    let anchor_deg = 23.0 + 15.0 / 60.0;
    let expected_deg = anchor_deg + years * published_rate_deg_per_year;

    let t = centuries(jd_1950);
    let got = ayanamsha_rad(Ayanamsha::Lahiri, t);
    let residual_arcsec = (got * RAD_TO_DEG - expected_deg) * 3600.0;
    assert!(
        residual_arcsec.abs() < 2.0,
        "Lahiri @ 1950-01-01: got {} deg, linear-rate estimate {} deg, residual {residual_arcsec} arcsec",
        got * RAD_TO_DEG,
        expected_deg
    );
}

#[test]
fn lahiri_spot_check_near_j2000_via_independent_linear_rate() {
    // Same independent cross-check, extrapolated ~43.8 years forward
    // instead of ~6.2 years back. The P03 t^2 term becomes slightly
    // more visible over the longer span (~0.3 arcsec here) but is
    // still far inside the 2 arcsec tolerance.
    let jd_2000 = jd_civil(2000, 1, 1);
    let days = jd_2000 - JD_LAHIRI_ANCHOR;
    let years = days / 365.25;
    let published_rate_deg_per_year = 50.29 / 3600.0;
    let anchor_deg = 23.0 + 15.0 / 60.0;
    let expected_deg = anchor_deg + years * published_rate_deg_per_year;

    let t = centuries(jd_2000);
    let got = ayanamsha_rad(Ayanamsha::Lahiri, t);
    let residual_arcsec = (got * RAD_TO_DEG - expected_deg) * 3600.0;
    assert!(
        residual_arcsec.abs() < 2.0,
        "Lahiri @ 2000-01-01: got {} deg (~23 deg 51', matching the widely cited figure), \
         linear-rate estimate {} deg, residual {residual_arcsec} arcsec",
        got * RAD_TO_DEG,
        expected_deg
    );
    // Sanity check against the crate brief's own hint ("~23 deg 51'"):
    assert!((got * RAD_TO_DEG - 23.85).abs() < 1.0 / 60.0 * 5.0);
}

#[test]
fn krishnamurti_spot_check_near_j2000_via_independent_linear_rate() {
    // The KP anchor (22 deg 21' 50" @ 1900.0) extrapolated to
    // 2000-01-01 with the generic ~50.29"/yr published rate, computed
    // from scratch — a cross-check of the p_A propagation over a full
    // century (P03 t^2 term ~1.3 arcsec here). Deliberately NOT
    // Krishnamurti's own 50.2388"/yr table rate: the ~5 arcsec/century
    // divergence between that rate and p_A is a documented model
    // choice (see the module docs' "Honesty about accuracy"), and
    // this tolerance is set below it plus the t^2 term combined so a
    // wrong anchor (arcminutes) can never hide inside it.
    let jd_2000 = jd_civil(2000, 1, 1);
    let jd_1900 = jd_civil(1900, 1, 1);
    let years = (jd_2000 - jd_1900) / 365.25;
    let anchor_deg = 22.0 + 21.0 / 60.0 + 50.0 / 3600.0;
    let expected_deg = anchor_deg + years * 50.29 / 3600.0;
    let got = ayanamsha_rad(Ayanamsha::Krishnamurti, centuries(jd_2000));
    let residual_arcsec = (got * RAD_TO_DEG - expected_deg) * 3600.0;
    assert!(
        residual_arcsec.abs() < 3.0,
        "Krishnamurti @ 2000-01-01: got {} deg (~23 deg 45', the widely cited KP figure), \
         linear-rate estimate {} deg, residual {residual_arcsec} arcsec",
        got * RAD_TO_DEG,
        expected_deg
    );
    // The widely cited KP value near J2000 is ~23 deg 45-46'.
    assert!((got * RAD_TO_DEG - (23.0 + 45.5 / 60.0)).abs() < 5.0 / 60.0);
}

#[test]
fn raman_spot_check_near_j2000_via_independent_linear_rate() {
    // Raman's 1912 anchor extrapolated to 2000-01-01 with the generic
    // ~50.29"/yr rate (88-year span; P03 t^2 term ~1.1 arcsec). As
    // above, Raman's own ~50 1/3"/yr table rate diverges from p_A by
    // only ~4 arcsec over this span — far below the arcminute scale
    // that an anchor transposition would produce.
    let jd_2000 = jd_civil(2000, 1, 1);
    let jd_1912 = jd_civil(1912, 1, 1);
    let years = (jd_2000 - jd_1912) / 365.25;
    let anchor_deg = 21.0 + 11.0 / 60.0 + 29.0 / 3600.0;
    let expected_deg = anchor_deg + years * 50.29 / 3600.0;
    let got = ayanamsha_rad(Ayanamsha::Raman, centuries(jd_2000));
    let residual_arcsec = (got * RAD_TO_DEG - expected_deg) * 3600.0;
    assert!(
        residual_arcsec.abs() < 3.0,
        "Raman @ 2000-01-01: got {} deg (~22 deg 25'), \
         linear-rate estimate {} deg, residual {residual_arcsec} arcsec",
        got * RAD_TO_DEG,
        expected_deg
    );
}

// ---------------------------------------------------------------------
// Published inter-system relationships (anchor-transposition guards)
// ---------------------------------------------------------------------

#[test]
fn published_inter_system_relationships_hold() {
    // Independently published, anchor-free relationships between the
    // systems (see the test-module docs, point 3):
    //   - Krishnamurti sits ~6' below Lahiri (the classic KP-vs-Lahiri
    //     offset, stated across the KP literature);
    //   - Raman sits ~1.4-1.5 deg below Lahiri ("about 1.5 deg less
    //     than Lahiri"; zero year ~397 CE vs Lahiri's ~285 CE);
    //   - Fagan/Bradley sits ~53' above Lahiri.
    // In this crate's model every variant propagates with the same
    // p_A, so each pairwise difference is time-independent; checking
    // at several epochs additionally guards the per-variant epoch
    // wiring. These bands are intentionally a few arcminutes wide
    // (the honest fidelity of the historical definitions) — but a
    // transposed anchor is off by *degrees* or tens of arcminutes and
    // cannot pass. In particular the erroneous Lahiri-style value
    // this file's crate once carried for Raman (22 deg 27' 37" @
    // 1900) would make raman_minus_lahiri ~ -0.005 deg, far outside
    // the band, and would violate raman < krishnamurti.
    for (y, m, d) in [
        (1900, 1, 1),
        (1912, 1, 1),
        (1956, 3, 21),
        (2000, 1, 1),
        (2026, 1, 1),
    ] {
        let t = centuries(jd_civil(y, m, d));
        let lahiri = ayanamsha_rad(Ayanamsha::Lahiri, t);
        let kp = ayanamsha_rad(Ayanamsha::Krishnamurti, t);
        let raman = ayanamsha_rad(Ayanamsha::Raman, t);
        let fb = ayanamsha_rad(Ayanamsha::FaganBradley, t);

        assert!(
            raman < kp && kp < lahiri && lahiri < fb,
            "{y}-{m:02}-{d:02}: ordering violated: raman {} < kp {} < lahiri {} < fb {} (deg)",
            raman * RAD_TO_DEG,
            kp * RAD_TO_DEG,
            lahiri * RAD_TO_DEG,
            fb * RAD_TO_DEG
        );

        let kp_minus_lahiri_arcmin = (kp - lahiri) * RAD_TO_DEG * 60.0;
        assert!(
            (-8.0..=-4.0).contains(&kp_minus_lahiri_arcmin),
            "{y}-{m:02}-{d:02}: kp - lahiri = {kp_minus_lahiri_arcmin} arcmin, expected ~ -6'"
        );

        let raman_minus_lahiri_deg = (raman - lahiri) * RAD_TO_DEG;
        assert!(
            (-1.6..=-1.3).contains(&raman_minus_lahiri_deg),
            "{y}-{m:02}-{d:02}: raman - lahiri = {raman_minus_lahiri_deg} deg, \
             expected ~ -1.44 deg (published: 'about 1.5 deg less than Lahiri')"
        );

        let fb_minus_lahiri_arcmin = (fb - lahiri) * RAD_TO_DEG * 60.0;
        assert!(
            (45.0..=60.0).contains(&fb_minus_lahiri_arcmin),
            "{y}-{m:02}-{d:02}: fb - lahiri = {fb_minus_lahiri_arcmin} arcmin, expected ~ +53'"
        );
    }
}

// ---------------------------------------------------------------------
// Monotonic increase, ~50.3 arcsec/year
// ---------------------------------------------------------------------

#[test]
fn ayanamsha_grows_monotonically_at_about_50_point_3_arcsec_per_year() {
    // Deliberately tests the *unwrapped* `p_A` quantity
    // (`general_precession_iau2006_rad`), not `ayanamsha_rad`: any
    // sidereal-zodiac value is an angle, taken modulo a full turn
    // (`wrap_0_2pi`), so it necessarily has a discontinuity somewhere
    // (by design -- 359.99 degrees and 0.01 degrees are adjacent, not
    // far apart). `J2000Zero`'s anchor sits exactly at that
    // discontinuity (value 0 at t=0), so sweeping `t` through 0 would
    // spuriously "jump" from just under 360 degrees to just over 0.
    // The underlying precession quantity itself, before wrapping, has
    // no such discontinuity and is the right thing to check for
    // monotonicity and rate.
    let mut previous = general_precession_iau2006_rad(-2.0);
    let years_per_step = 1.0;
    let centuries_per_step = years_per_step / 100.0;
    let mut t = -2.0 + centuries_per_step;
    let mut rates_arcsec_per_year = Vec::new();
    while t <= 2.0 {
        let current = general_precession_iau2006_rad(t);
        assert!(current > previous, "not monotonic at t={t}");
        rates_arcsec_per_year.push((current - previous) * RAD_TO_ARCSEC / years_per_step);
        previous = current;
        t += centuries_per_step;
    }
    assert!(
        rates_arcsec_per_year.len() > 300,
        "expected ~400 yearly samples, got {}",
        rates_arcsec_per_year.len()
    );
    for rate in rates_arcsec_per_year {
        assert!(
            (49.5..=51.1).contains(&rate),
            "yearly rate {rate} arcsec/yr outside the expected ~50.3 arcsec/yr band \
             (P03 predicts ~50.27-50.31 arcsec/yr over +-2 centuries from J2000)"
        );
    }
}

// ---------------------------------------------------------------------
// Custom anchor round-trip
// ---------------------------------------------------------------------

#[test]
fn custom_anchor_round_trips_at_its_own_epoch() {
    let t0_jd_tt = 2_460_000.5; // an arbitrary modern epoch
    let value_at_t0_rad = 24.5 * DEG_TO_RAD;
    let kind = Ayanamsha::Custom {
        t0_jd_tt,
        value_at_t0_rad,
    };
    let t0_centuries = centuries(t0_jd_tt);
    let got = ayanamsha_rad(kind, t0_centuries);
    assert!(
        (got - value_at_t0_rad).abs() < 1e-9,
        "residual {} arcsec",
        (got - value_at_t0_rad) * RAD_TO_ARCSEC
    );
}

#[test]
fn custom_anchor_propagates_consistently_forward_and_back() {
    let t0_jd_tt = 2_400_000.5;
    let value_at_t0_rad = 10.0 * DEG_TO_RAD;
    let kind = Ayanamsha::Custom {
        t0_jd_tt,
        value_at_t0_rad,
    };
    let t0_centuries = centuries(t0_jd_tt);

    // Forward by 1 century, then treat that value as a *new* Custom
    // anchor and propagate back to t0: must recover the original
    // value_at_t0_rad.
    let t1_centuries = t0_centuries + 1.0;
    let value_at_t1 = ayanamsha_rad(kind, t1_centuries);
    let round_trip_kind = Ayanamsha::Custom {
        t0_jd_tt: t1_centuries * DAYS_PER_CENTURY + JD_J2000,
        value_at_t0_rad: value_at_t1,
    };
    let back = ayanamsha_rad(round_trip_kind, t0_centuries);
    assert!(
        (back - value_at_t0_rad).abs() < 1e-9,
        "round trip residual {} arcsec",
        (back - value_at_t0_rad) * RAD_TO_ARCSEC
    );
}

// ---------------------------------------------------------------------
// sidereal_from_tropical sanity
// ---------------------------------------------------------------------

#[test]
fn sidereal_from_tropical_subtracts_the_ayanamsha() {
    let t = centuries(jd_civil(2024, 6, 1));
    let tropical = 100.0 * DEG_TO_RAD;
    let aya = ayanamsha_rad(Ayanamsha::Lahiri, t);
    let sidereal = sidereal_from_tropical(tropical, Ayanamsha::Lahiri, t);
    let expected = (tropical - aya).rem_euclid(2.0 * core::f64::consts::PI);
    assert!((sidereal - expected).abs() < 1e-12);
    // Sidereal Lahiri longitude for a body at tropical 100 deg in 2024
    // should be a bit under 76 deg (100 - ~24.2).
    assert!(sidereal * RAD_TO_DEG > 75.0 && sidereal * RAD_TO_DEG < 77.0);
}

#[test]
fn all_kinds_produce_finite_wrapped_values() {
    let t = centuries(jd_civil(1975, 5, 5));
    for kind in [
        Ayanamsha::FaganBradley,
        Ayanamsha::Lahiri,
        Ayanamsha::Krishnamurti,
        Ayanamsha::Raman,
        Ayanamsha::J2000Zero,
    ] {
        let v = ayanamsha_rad(kind, t);
        assert!(v.is_finite());
        assert!(
            (0.0..core::f64::consts::PI * 2.0).contains(&v),
            "{kind:?}: {v}"
        );
    }
}

#[test]
fn j2000_zero_is_exactly_zero_at_j2000() {
    let v = ayanamsha_rad(Ayanamsha::J2000Zero, 0.0);
    // Exact by construction: anchor value 0.0 plus p_A(0.0) - p_A(0.0).
    assert!(v.abs() < 1e-15, "expected exactly 0.0, got {v}");
}
