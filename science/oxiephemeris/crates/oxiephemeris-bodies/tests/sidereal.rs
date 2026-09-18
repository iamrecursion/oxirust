//! Tests for `oxiephemeris_bodies::sidereal`: ERA/GMST/EE/GAST
//! properties, plus a JPL Horizons sidereal-time oracle.
//!
//! # The Horizons oracle (quantity 7, `L_Ap_Sid_Time`)
//!
//! The fixtures under `tests/fixtures/horizons_sidereal/` are raw
//! responses of the public JPL Horizons API in OBSERVER mode with
//! `QUANTITIES='7'` (local apparent sidereal time) for a **geodetic site
//! at longitude 0, latitude 0, altitude 0** (`CENTER='coord@399'`,
//! `SITE_COORD='0,0,0'`), at UTC epochs spread over 1975–2020 (three
//! gated, one informational — see below) — see the fixture `README.md`
//! for full provenance. Quantity 7 depends
//! only on the site and the clock (the `COMMAND` body is irrelevant);
//! at longitude 0 the polar-motion longitude correction of USNO
//! Circular 179 eq. 2.16 vanishes identically (its factor `tan φ` is 0
//! at the equator and the correction is 0 at λ = 0 anyway), so
//! **LAST = GAST** exactly.
//!
//! Horizons expresses apparent places — and hence apparent sidereal
//! time — on its EOP-corrected IAU76/80 frame, whose RA origin is offset
//! from the IAU 2006/2000A true-of-date frame by the documented constant
//! −52.93 mas equinox tie (derivation and citations in
//! `tests/horizons_spotcheck.rs`, which validated the same constant
//! against ten Horizons RA fixtures over 1965–2026). A sidereal time is
//! an RA (of the meridian), so the same tie applies:
//! `LAST(Horizons) = GAST(IAU 2006/2000A) + tie`, with the arcsec →
//! seconds-of-time conversion 1s = 15″. Both the raw and the tied
//! residuals are always printed; the gate is on the tied residual,
//! < 1 ms of time (≈ 15 mas), robust to EOP noise but far tighter than
//! any sign/frame/timescale error (the tie alone is 3.53 ms).
//!
//! # The ungated 1975 epoch (pre-VLBI UT1-series disagreement)
//!
//! A fourth fixture, 1975-06-20, is compared and printed but **not
//! gated**. Unlike the TT-driven RA spotchecks, LAST depends directly on
//! UT1, and in the pre-VLBI era independent UT1 realizations genuinely
//! disagree at the millisecond level: at MJD 42583 the modern IERS
//! series themselves differ by 0.32 ms (`finals2000A.all` +0.2239539 s
//! vs EOP 20 C04 +0.2236377 s, the latter with a 1.9 ms formal error),
//! and the UT1 implied by Horizons' LAST (its EOP file descends from
//! JPL's own historical Earth-orientation solution) differs from both by
//! ~1.0–1.3 ms. The observed tied residual (~+1.3 ms) is therefore
//! EOP-series disagreement, not a model error — the measurement itself
//! is only good to a couple of ms there. See the fixture `README.md`
//! ("Why the early gated epoch is 1987, not 1975") for the full
//! investigation. The earliest *gated* epoch, 1987-04-10, is in the
//! VLBI era, where `finals2000A.all` and C04 agree to 0.007 ms.

use std::error::Error;
use std::path::PathBuf;

use oxiephemeris_bodies::frames::{mean_obliquity_iau2006, nutation_iau2000a_truncated};
use oxiephemeris_bodies::sidereal::{equation_of_equinoxes, era, gast_iau2006, gmst_iau2006};
use oxiephemeris_core::angle::{normalize_pm_pi, AS2R, TWO_PI};
use oxiephemeris_core::time::{julday, utc_to_tt, Calendar, JulianDate, J2000_JD, SECONDS_PER_DAY};

type TestResult = Result<(), Box<dyn Error>>;

/// ERA rate excess over one revolution per UT1 day (IERS TN36 eq. 5.15:
/// the full rate is `1 + this` = 1.00273781191135448 rev/day; the excess
/// form keeps the literal exactly representable).
const ERA_RATE_EXCESS_REV_PER_DAY: f64 = 0.002_737_811_911_354_48;

/// ERA at J2000.0 UT1 in revolutions (IERS TN36 eq. 5.15).
const ERA_AT_J2000_REV: f64 = 0.779_057_273_264_0;

#[test]
fn era_at_j2000_matches_literal_formula() {
    // Tu = 0 exactly: ERA = 2π · 0.7790572732640 (mod 2π).
    let got = era(JulianDate::from_f64(J2000_JD));
    let want = TWO_PI * ERA_AT_J2000_REV;
    assert!(
        (got - want).abs() < 1e-13,
        "ERA(J2000.0) = {got}, want {want}"
    );
}

#[test]
fn era_is_split_invariant() {
    // The same instant under different (hi, lo) splits must give the
    // same angle. The day fractions are dyadic (exactly representable)
    // so every pair below denotes the *identical* real number — with a
    // non-dyadic fraction like 0.7 the splits would legitimately differ
    // by the one-part rounding remainder (~20 µs, the very thing the
    // two-part JD exists to avoid).
    let splits = [
        JulianDate::new(2_451_545.0, 0.75),
        JulianDate::new(2_451_545.75, 0.0),
        JulianDate::new(2_451_546.0, -0.25),
        JulianDate::new(2_451_545.5, 0.25),
    ];
    let reference = era(splits[0]);
    for jd in &splits[1..] {
        let got = era(*jd);
        assert!(
            (got - reference).abs() < 1e-13,
            "ERA split-dependent: {got} vs {reference} at {jd:?}"
        );
    }
}

#[test]
fn era_advances_at_the_published_rate() {
    // ERA(jd + 1 d) − ERA(jd) ≡ 2π · 1.00273781191135448 (mod 2π),
    // i.e. 2π · 0.00273781191135448 after the whole revolution drops.
    // Sampled over 1900–2100 (JD 2415020.5 … 2488069.5) with a
    // non-trivial day fraction so the fractional-part paths are hit.
    let expected = TWO_PI * ERA_RATE_EXCESS_REV_PER_DAY;
    let start = JulianDate::new(2_415_020.5, 0.437);
    let mut worst = 0.0_f64;
    for k in 0..731 {
        let jd = start.add_days(f64::from(k) * 100.0); // 731 samples, 200 y
        let diff = normalize_pm_pi(era(jd.add_days(1.0)) - era(jd) - expected);
        worst = worst.max(diff.abs());
    }
    assert!(
        worst < 1e-12,
        "ERA daily-advance error {worst} rad >= 1e-12 rad"
    );
}

/// The GMST−ERA polynomial of IERS TN36 eq. (5.32), arcseconds,
/// recomputed independently of the library.
fn gmst_poly_arcsec(t: f64) -> f64 {
    0.014_506
        + t * (4_612.156_534
            + t * (1.391_581_7
                + t * (-0.000_000_44 + t * (-0.000_029_956 + t * (-0.000_000_036_8)))))
}

#[test]
fn gmst_minus_era_is_the_polynomial() {
    // GMST − ERA must equal the TN36 eq. 5.32 polynomial exactly (up to
    // the 2π reduction), independently at several t and several UT1
    // epochs (the polynomial depends only on t).
    let epochs = [
        JulianDate::from_f64(2_442_583.5),
        JulianDate::new(2_451_545.0, 0.25),
        JulianDate::from_f64(2_488_069.5),
    ];
    for &t in &[-1.0, -0.5, -0.25, 0.0, 0.37, 1.0] {
        for jd in epochs {
            let diff = normalize_pm_pi(gmst_iau2006(jd, t) - era(jd) - gmst_poly_arcsec(t) * AS2R);
            assert!(
                diff.abs() < 1e-12,
                "GMST − ERA − poly = {diff} rad at t = {t}"
            );
        }
    }
}

#[test]
fn equation_of_equinoxes_bounds() {
    // The leading term Δψ cos εA reaches ±16″ ≈ ±1.05 s of time, so the
    // equation of the equinoxes stays below 1.3 seconds of time
    // (= 19.5″); the complementary terms are below ~3 mas.
    let mut worst_ee_time_sec = 0.0_f64;
    let mut worst_comp_arcsec = 0.0_f64;
    for k in 0..=2000 {
        let t = -1.0 + f64::from(k) * 0.001; // t ∈ [−1, +1], 2001 samples
        let ee = equation_of_equinoxes(t);
        let nut = nutation_iau2000a_truncated(t);
        let leading = nut.dpsi_rad * mean_obliquity_iau2006(t).cos();
        worst_ee_time_sec = worst_ee_time_sec.max((ee / AS2R / 15.0).abs());
        worst_comp_arcsec = worst_comp_arcsec.max(((ee - leading) / AS2R).abs());
    }
    assert!(
        worst_ee_time_sec < 1.3,
        "|EE| reaches {worst_ee_time_sec} s of time (>= 1.3 s) over 1900–2100"
    );
    // Complementary terms are milliarcsecond-scale: < 5 mas.
    assert!(
        worst_comp_arcsec < 0.005,
        "|EE − Δψ cos εA| reaches {worst_comp_arcsec}\" (>= 5 mas)"
    );
    // ... and they are not accidentally zero (the Ω term alone has a
    // 2.64 mas amplitude, so the maximum over 200 years exceeds 2 mas).
    assert!(
        worst_comp_arcsec > 0.002,
        "complementary terms suspiciously small: {worst_comp_arcsec}\""
    );
}

#[test]
fn gast_is_gmst_plus_ee_normalized() {
    for &t in &[-0.5, 0.0, 0.19] {
        let jd = JulianDate::new(2_451_545.0, 0.6 + t * 36_525.0);
        let diff =
            normalize_pm_pi(gast_iau2006(jd, t) - gmst_iau2006(jd, t) - equation_of_equinoxes(t));
        assert!(diff.abs() < 1e-13, "GAST − GMST − EE = {diff} rad");
        let gast = gast_iau2006(jd, t);
        assert!((0.0..TWO_PI).contains(&gast), "GAST {gast} not in [0, 2π)");
    }
}

// ---------------------------------------------------------------------
// Horizons oracle
// ---------------------------------------------------------------------

/// ICRS RA-origin offset from the J2000.0 mean equinox, arcsec: frame
/// bias `dα₀` (IERS TN36 §5.5.4, eq. 5.21; Chapront, Chapront-Touzé &
/// Francou 2002, A&A 387, 700). Duplicated from
/// `tests/horizons_spotcheck.rs` (same constant pair, same derivation —
/// see that file's module docs; test binaries cannot share code without
/// a helper crate, so the two literals are kept in sync by hand).
const FRAME_BIAS_DALPHA0_ARCSEC: f64 = -0.014_60;

/// GCRS frame-bias pole offset `ξ₀`, arcsec (IERS TN36 eq. 5.21).
/// Duplicated from `tests/horizons_spotcheck.rs` — see above.
const FRAME_BIAS_XI0_ARCSEC: f64 = -0.016_617_0;

/// RA-origin tie of the Horizons apparent frame (EOP-corrected IAU76/80)
/// relative to the IAU 2006/2000A true-of-date frame, radians:
/// `dα₀ + ξ₀ / tan ε₀ = −0.05293″` (−52.93 mas; full first-order
/// derivation in `tests/horizons_spotcheck.rs`, matching the "−53 mas"
/// note in Horizons' own output footers). Sidereal time is the RA of
/// the meridian, so `LAST(Horizons) = GAST(ours) + tie`.
fn horizons_equinox_tie_rad() -> f64 {
    let eps0 = mean_obliquity_iau2006(0.0);
    (FRAME_BIAS_DALPHA0_ARCSEC + FRAME_BIAS_XI0_ARCSEC / eps0.tan()) * AS2R
}

/// Gate on the frame-tied residual, milliseconds of time (≈ 15 mas).
const TIED_RESIDUAL_GATE_MS: f64 = 1.0;

/// One oracle epoch: a 00:00:00 UTC instant, its `UT1−UTC`, and the
/// committed raw Horizons response carrying the expected LAST.
///
/// `dut1_seconds` values are read from `data/iers/finals2000A.all`
/// (IERS combined EOP series, workspace snapshot of 2026-07-05;
/// column 59–68, the Bulletin-A-style UT1−UTC in seconds). The exact
/// source lines are quoted in `tests/fixtures/horizons_sidereal/`
/// `README.md`; the values are hardcoded here on purpose so the test
/// does not silently track a refreshed EOP file.
struct OracleEpoch {
    /// Fixture file name under `tests/fixtures/horizons_sidereal/`.
    fixture: &'static str,
    /// Date token as Horizons prints it, for row cross-checking.
    horizons_date: &'static str,
    /// UTC calendar date (proleptic Gregorian), at 00:00:00 UTC.
    year: i32,
    month: u8,
    day: u8,
    /// UT1 − UTC in seconds at that 0h-UTC instant.
    dut1_seconds: f64,
    /// Whether the < 1 ms gate applies. `false` only for the pre-VLBI
    /// 1975 epoch, where the UT1 realizations themselves disagree at the
    /// millisecond level (see the module docs) — printed, never gated.
    gated: bool,
}

const ORACLE_EPOCHS: [OracleEpoch; 4] = [
    OracleEpoch {
        fixture: "1975-06-20.txt",
        horizons_date: "1975-Jun-20 00:00",
        year: 1975,
        month: 6,
        day: 20,
        // finals2000A.all: "75 620 42583.00 I … I 0.2239539 0.0007530 …"
        // (EOP 20 C04 says 0.2236377 ± 0.0019 for the same instant: the
        // pre-VLBI series disagree at the ms level, hence not gated.)
        dut1_seconds: 0.223_953_9,
        gated: false,
    },
    OracleEpoch {
        fixture: "1987-04-10.txt",
        horizons_date: "1987-Apr-10 00:00",
        year: 1987,
        month: 4,
        day: 10,
        // finals2000A.all: "87 410 46895.00 I … I-0.2910912 0.0000149 …"
        dut1_seconds: -0.291_091_2,
        gated: true,
    },
    OracleEpoch {
        fixture: "1998-11-10.txt",
        horizons_date: "1998-Nov-10 00:00",
        year: 1998,
        month: 11,
        day: 10,
        // finals2000A.all: "981110 51127.00 I … I-0.2191202 0.0000094 …"
        dut1_seconds: -0.219_120_2,
        gated: true,
    },
    OracleEpoch {
        fixture: "2019-03-05.txt",
        horizons_date: "2019-Mar-05 00:00",
        year: 2019,
        month: 3,
        day: 5,
        // finals2000A.all: "19 3 5 58547.00 I … I-0.0913344 0.0000057 …"
        dut1_seconds: -0.091_334_4,
        gated: true,
    },
];

/// Extracts the first data row's `L_Ap_Sid_Time` from a raw Horizons
/// quantity-7 CSV response, cross-checking the row's date token, and
/// returns it in seconds of time.
fn parse_last_seconds(raw: &str, expected_date: &str) -> Result<f64, Box<dyn Error>> {
    let soe = raw
        .find("$$SOE")
        .ok_or("fixture has no $$SOE data-block marker")?;
    let eoe = raw.find("$$EOE").ok_or("fixture has no $$EOE marker")?;
    let block = raw.get(soe + 5..eoe).ok_or("malformed $$SOE/$$EOE block")?;
    let row = block
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .ok_or("empty $$SOE/$$EOE block")?;
    // CSV row shape: "1975-Jun-20 00:00, ,m, 17 50 20.7529,"
    let mut fields = row.split(',').map(str::trim);
    let date = fields.next().ok_or("row has no date field")?;
    if date != expected_date {
        return Err(format!("fixture row date {date:?} != expected {expected_date:?}").into());
    }
    let sid = fields
        .nth(2)
        .ok_or("row has no L_Ap_Sid_Time field (expected 4th CSV field)")?;
    let mut hms = sid.split_whitespace();
    let bad = || -> Box<dyn Error> { format!("malformed L_Ap_Sid_Time field {sid:?}").into() };
    let hours: u8 = hms.next().ok_or_else(bad)?.parse()?;
    let minutes: u8 = hms.next().ok_or_else(bad)?.parse()?;
    let seconds: f64 = hms.next().ok_or_else(bad)?.parse()?;
    if hms.next().is_some() || hours > 23 || minutes > 59 || !(0.0..60.0).contains(&seconds) {
        return Err(bad());
    }
    Ok(f64::from(hours) * 3600.0 + f64::from(minutes) * 60.0 + seconds)
}

/// Residual `a − b` on the sidereal-day circle, wrapped to
/// `[−43200, +43200)` seconds so a wrap at 0h/24h cannot masquerade as a
/// huge error.
fn wrap_residual_seconds(a: f64, b: f64) -> f64 {
    normalize_pm_pi((a - b) / SECONDS_PER_DAY * TWO_PI) / TWO_PI * SECONDS_PER_DAY
}

#[test]
// `jd_utc` / `jd_ut1` / `jd_tt` are the standard, unambiguous names of
// the three time scales involved; renaming them would hurt clarity.
#[allow(clippy::similar_names)]
fn horizons_last_oracle_three_gated_epochs() -> TestResult {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("horizons_sidereal");
    if !dir.is_dir() {
        println!(
            "note: skipping test — fixture directory not found at {}",
            dir.display()
        );
        return Ok(());
    }
    let tie_rad = horizons_equinox_tie_rad();
    println!(
        "Horizons LAST oracle (lon 0 ⇒ LAST = GAST); equinox tie {:.5}\" = {:.4} ms of time",
        tie_rad / AS2R,
        tie_rad / AS2R / 15.0 * 1000.0
    );
    println!(
        "{:<14} {:>14} {:>14} {:>12} {:>12}  gate",
        "epoch (UTC)", "horizons (s)", "ours (s)", "raw (ms)", "tied (ms)"
    );

    let mut failures = Vec::new();
    for epoch in &ORACLE_EPOCHS {
        let raw = std::fs::read_to_string(dir.join(epoch.fixture))
            .map_err(|e| format!("{}: {e}", epoch.fixture))?;
        let horizons_last_sec = parse_last_seconds(&raw, epoch.horizons_date)
            .map_err(|e| format!("{}: {e}", epoch.fixture))?;

        // UTC 00:00:00 → TT via the core leap-second table; UT1 = UTC + dUT1.
        let jd_utc = julday(Calendar::Gregorian, epoch.year, epoch.month, epoch.day, 0.0)
            .map_err(|e| format!("{}: julday: {e}", epoch.fixture))?;
        let jd_tt = utc_to_tt(jd_utc).map_err(|e| format!("{}: utc_to_tt: {e}", epoch.fixture))?;
        let t_tt = jd_tt.diff_days(JulianDate::from_f64(J2000_JD)) / 36_525.0;
        let jd_ut1 = jd_utc.add_seconds(epoch.dut1_seconds);

        let gast = gast_iau2006(jd_ut1, t_tt);
        let ours_sec = gast / TWO_PI * SECONDS_PER_DAY;
        let tied_sec = (gast + tie_rad) / TWO_PI * SECONDS_PER_DAY;
        let raw_residual_ms = wrap_residual_seconds(ours_sec, horizons_last_sec) * 1000.0;
        let tied_residual_ms = wrap_residual_seconds(tied_sec, horizons_last_sec) * 1000.0;
        println!(
            "{:<14} {:>14.4} {:>14.4} {:>12.4} {:>12.4}  {}",
            epoch.horizons_date.split(' ').next().unwrap_or("?"),
            horizons_last_sec,
            ours_sec,
            raw_residual_ms,
            tied_residual_ms,
            if epoch.gated {
                "< 1 ms"
            } else {
                "none (pre-VLBI UT1-series disagreement; see module docs)"
            }
        );
        if !epoch.gated {
            continue;
        }
        if !tied_residual_ms.is_finite() || tied_residual_ms.abs() >= TIED_RESIDUAL_GATE_MS {
            failures.push(format!(
                "{}: |tied residual| = {:.4} ms >= {TIED_RESIDUAL_GATE_MS} ms \
                 (raw, without the -52.93 mas equinox tie: {:.4} ms)",
                epoch.fixture,
                tied_residual_ms.abs(),
                raw_residual_ms
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Horizons sidereal-time gate violations:\n{}",
        failures.join("\n")
    );
    Ok(())
}
