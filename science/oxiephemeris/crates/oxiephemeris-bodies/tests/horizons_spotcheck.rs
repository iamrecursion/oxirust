//! JPL Horizons spot-check of the full apparent-place pipeline
//! (CLAUDE.md phase 1 step 9; design.md exit criterion: < 0.01″).
//!
//! The ten fixtures under `tests/fixtures/horizons/` are raw responses of
//! the public JPL Horizons API (geocentric observer `500@399`, quantities
//! 1, 2, 20, 31, `ANG_FORMAT=DEG`, `EXTRA_PREC=YES`, epochs requested and
//! returned in **TT** — see the fixture `README.md` for full provenance).
//! `manifest.json` is the machine-readable per-body extract of the first
//! CSV row of each raw response.
//!
//! What is compared (semantics quoted from the Horizons column-meaning
//! footer embedded in every raw fixture):
//!
//! * **Astrometric RA/Dec** (quantity 1): ICRF axes, down-leg light-time
//!   only — matched by [`Frame::Icrs`] with aberration and deflection off.
//!   Hard gate: true angular separation < 0.01″ per body.
//! * **Apparent RA/Dec** (quantity 2, airless): true equator and equinox
//!   of date, light-time + solar deflection + stellar aberration +
//!   precession + nutation, on the EOP-corrected IAU76/80 frame model —
//!   matched by [`Frame::TrueOfDate`] (IAU 2006/2000A-truncated here) with
//!   aberration and deflection on, **after applying the documented equinox
//!   frame tie between the two frame realizations** (see below). Hard
//!   gate: frame-tied separation < 0.01″ per body. The raw (untied)
//!   separation is also printed so the ≈ −53 mas systematic stays visible
//!   and reported, never silently absorbed.
//!
//! # The Horizons apparent-frame equinox tie (why it exists, how large)
//!
//! Horizons quantity 2 is *not* expressed in the IAU 2006/2000A
//! true-of-date frame. Its own column-meaning footer (identical in all ten
//! raw fixtures) says the frame x-axis is the "Earth equinox of-date
//! (x-axis, EOP-corrected IAU76/80)" and adds verbatim: *"Note: equinox
//! (RA origin) is offset -53 mas from the of-date frame defined by the
//! IAU06/00a P & N system."* Comparing an IAU 2006/2000A pipeline against
//! quantity 2 therefore requires the RA-origin tie; without it the
//! comparison fails by the documented 53 mas for **any** correct
//! implementation.
//!
//! The tie is derivable from published constants (no fitting). Horizons
//! applies the EOP-corrected IAU76/80 chain directly to ICRF/DE vectors
//! without the GCRS frame-bias rotation. The IERS celestial-pole offsets
//! restore the true **pole** (they absorb the bias pole components), but a
//! correction applied as a nutation-in-longitude increment `δψ` moves the
//! pole by `δψ sin ε` while moving the equinox along the equator by
//! `δψ cos ε`; the bias RA-rotation `dα₀` is never applied at all. To
//! first order the two realized frames then share a pole but their RA
//! origins differ by (frame-bias angles, IERS TN36 eq. 5.21):
//!
//! ```text
//! RA(Horizons) = RA(IAU 2006/2000A) + dα₀ + ξ₀ / tan ε₀
//!              = RA(IAU 2006/2000A) − 0.05293″
//! ```
//!
//! with `dα₀ = −0.01460″` (ICRS RA origin vs. J2000 mean equinox,
//! TN36 §5.5.4 / Chapront, Chapront-Touzé & Francou 2002, A&A 387, 700),
//! `ξ₀ = −0.0166170″` (TN36 eq. 5.21), `ε₀ = 84381.406″` (IAU 2006,
//! Capitaine, Wallace & Chapront 2003) — matching the "−53 mas" Horizons
//! itself documents. The observed per-body raw offsets here are −50.7 to
//! −52.8 mas, i.e. the tie closes them to ≤ 3 mas.
//!
//! **EOP-coverage caveat**: the tie above presumes Horizons' pole is
//! EOP-corrected, which holds only inside the EOP table of the Horizons
//! server (each raw fixture header prints it: `EOP coverage : DATA-BASED
//! 1962-JAN-20 TO 2026-JUL-03`). Beyond that date the corrections are
//! frozen and Horizons' frame drifts away at the IAU76 precession-rate
//! error (≈ −0.3″/cy in `δψ`); fixture epochs must therefore lie inside
//! the data-based EOP span. The original Uranus (2033) and Pluto (2040)
//! fixtures violated this and were re-fetched at in-coverage epochs
//! (2013 / 1996) — see the fixture `README.md`.
//! * **delta** (quantity 20): light-time-aberrated range in AU — matched
//!   by [`BodyPosition::r_au`] (same retarded-epoch convention, see
//!   `apparent.rs`). Hard gate: |difference| < 1e-6 AU.
//! * **`ObsEcLon`/`ObsEcLat`** (quantity 31): *apparent* ecliptic-OF-DATE
//!   longitude/latitude on the **IAU76/80** ecliptic — compared against
//!   [`Frame::EclipticTrueOfDate`] and only **printed**, never asserted,
//!   in phase 1: the IAU 1980 mean obliquity differs from the IAU 2006
//!   value by ≈ 42 mas at J2000 (Capitaine, Wallace & Chapront 2003, A&A
//!   412, 567, vs. Lieske et al. 1977, A&A 58, 1) and the Horizons values
//!   carry only 1e-7 deg (0.36 mas) of print precision.
//!
//! Body mapping: Horizons `COMMAND` codes 10/301/199/299 are the Sun,
//! Moon, Mercury and Venus body centers; codes 4–9 are the Mars–Pluto
//! **planetary-system barycenters**. The pipeline's [`Target::Mars`] ..
//! [`Target::Pluto`] denote exactly those DE system-barycenter series
//! (Park et al. 2021, AJ 161, 105, §4 — the DE files integrate and store
//! the system barycenters for Mars through Pluto), so the comparison is
//! apples-to-apples by construction.
//!
//! Ephemeris caveat: Horizons served **DE441**, this test evaluates
//! **DE440**. Park et al. 2021 (§5) describe both as sharing the same
//! data fits; over 1965–2026 the state differences for these targets are
//! sub-km, i.e. ≪ 1 mas and ≪ 1e-8 AU here.
//!
//! Both the DE440 binary (`data/de440/`, not committed) and the fixture
//! directory are checked at runtime; the test skips gracefully (prints a
//! note, returns `Ok`) when either is absent.

use std::error::Error;
use std::path::PathBuf;

use serde::Deserialize;

use oxiephemeris_bodies::apparent::{apparent, Center, Frame, Options, Target};
use oxiephemeris_bodies::frames::{mean_obliquity_iau2006, NutationModel};
use oxiephemeris_bodies::math::{cross, dot, norm, spherical_to_cartesian};
use oxiephemeris_core::angle::{normalize_pm_pi, AS2R, DEG2RAD};
use oxiephemeris_core::time::{julday, Calendar, JulianDate};
use oxiephemeris_de::DeFile;

type TestResult = Result<(), Box<dyn Error>>;

/// Hard gate for the astrometric (ICRF, light-time-only) comparison, in
/// arcseconds (design.md phase 1 exit criterion).
const ASTROMETRIC_TOL_ARCSEC: f64 = 0.01;

/// Hard gate for the apparent (true-of-date, airless) comparison after the
/// equinox frame tie, in arcseconds (design.md phase 1 exit criterion).
const APPARENT_TOL_ARCSEC: f64 = 0.01;

/// ICRS RA-origin offset from the J2000.0 mean equinox, arcsec: the frame
/// bias `dα₀` (IERS TN36 §5.5.4, eq. 5.21; Chapront, Chapront-Touzé &
/// Francou 2002, A&A 387, 700).
const FRAME_BIAS_DALPHA0_ARCSEC: f64 = -0.014_60;

/// GCRS frame-bias pole offset `ξ₀`, arcsec (IERS TN36 eq. 5.21).
const FRAME_BIAS_XI0_ARCSEC: f64 = -0.016_617_0;

/// RA-origin tie of the Horizons apparent frame (EOP-corrected IAU76/80)
/// relative to the IAU 2006/2000A true-of-date frame, in radians:
///
/// ```text
/// RA(Horizons) = RA(IAU 2006/2000A) + (dα₀ + ξ₀ / tan ε₀)
/// ```
///
/// = −0.05293″ = −52.93 mas (module docs give the first-order derivation;
/// Horizons' own footer documents the same value rounded, "−53 mas").
/// `ε₀` is the IAU 2006 mean obliquity at J2000.0 taken from the library
/// itself so the same constant is used on both sides.
fn horizons_equinox_tie_rad() -> f64 {
    let eps0 = mean_obliquity_iau2006(0.0);
    (FRAME_BIAS_DALPHA0_ARCSEC + FRAME_BIAS_XI0_ARCSEC / eps0.tan()) * AS2R
}

/// Hard gate for the range comparison against Horizons quantity 20, AU.
const DELTA_TOL_AU: f64 = 1e-6;

/// Manifest-integrity gate: independently recomputed JD(TT) of
/// `epoch_iso` must match the manifest `jd` to below a millisecond.
const JD_CROSS_CHECK_TOL_DAYS: f64 = 1e-8;

/// One entry of `manifest.json` (fields not needed by this test are left
/// undeclared; `serde` ignores unknown JSON fields by default).
#[derive(Debug, Deserialize)]
struct Fixture {
    body: String,
    command: String,
    epoch_iso: String,
    time_scale: String,
    jd: f64,
    astrometric_ra_deg: f64,
    astrometric_dec_deg: f64,
    apparent_ra_deg: f64,
    apparent_dec_deg: f64,
    delta_au: f64,
    obs_ecl_lon_deg: f64,
    obs_ecl_lat_deg: f64,
    source_file: String,
}

/// Per-body comparison results, kept so the full table always prints
/// before any gate is asserted.
///
/// `apparent_raw_sep_arcsec` is the direct separation against Horizons
/// quantity 2 (dominated by the documented ≈ −53 mas equinox tie, printed
/// so the systematic stays reported); `apparent_tied_sep_arcsec` is the
/// separation after rotating our RA origin onto the Horizons realization
/// by [`horizons_equinox_tie_rad`] — this is the gated quantity.
/// `apparent_dra_cosdec_arcsec` / `apparent_ddec_arcsec` decompose the
/// **raw** residual into an along-equator and a declination component: a
/// pure RA-origin (equinox) rotation between the two frame models shows up
/// as a uniform `dRA·cosδ` with near-zero `dDec`.
struct Row {
    body: String,
    astrometric_sep_arcsec: f64,
    apparent_raw_sep_arcsec: f64,
    apparent_tied_sep_arcsec: f64,
    apparent_dra_cosdec_arcsec: f64,
    apparent_ddec_arcsec: f64,
    delta_diff_au: f64,
    ecl_dlon_arcsec: f64,
    ecl_dlat_arcsec: f64,
}

/// Loads the DE440 binary, or `None` (with a note) when absent.
fn de440_bytes() -> Option<Vec<u8>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("de440")
        .join("linux_p1550p2650.440");
    if let Ok(bytes) = std::fs::read(&path) {
        Some(bytes)
    } else {
        println!(
            "note: skipping test — DE440 file not found at {}",
            path.display()
        );
        None
    }
}

/// Loads and parses `manifest.json`, or `Ok(None)` (with a note) when the
/// fixture directory is absent. A present-but-unparsable manifest is an
/// error, not a skip.
fn load_manifest() -> Result<Option<Vec<Fixture>>, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("horizons")
        .join("manifest.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        println!(
            "note: skipping test — Horizons manifest not found at {}",
            path.display()
        );
        return Ok(None);
    };
    let fixtures: Vec<Fixture> = serde_json::from_str(&text)
        .map_err(|e| format!("manifest.json is present but unparsable: {e}"))?;
    Ok(Some(fixtures))
}

/// Horizons `COMMAND` code → pipeline target. Codes 4–9 are the
/// planetary-system barycenters, which is precisely what the DE-backed
/// [`Target::Mars`] .. [`Target::Pluto`] evaluate (see module docs).
fn target_for_command(command: &str) -> Option<Target> {
    match command {
        "10" => Some(Target::Sun),
        "301" => Some(Target::Moon),
        "199" => Some(Target::Mercury),
        "299" => Some(Target::Venus),
        "4" => Some(Target::Mars),
        "5" => Some(Target::Jupiter),
        "6" => Some(Target::Saturn),
        "7" => Some(Target::Uranus),
        "8" => Some(Target::Neptune),
        "9" => Some(Target::Pluto),
        _ => None,
    }
}

/// Parses `YYYY-MM-DDTHH:MM:SS` into `(year, month, day, hours)`.
/// Hand-rolled (project policy: no chrono); all fixture epochs are CE
/// Gregorian dates with four-digit years.
fn parse_iso_epoch(iso: &str) -> Result<(i32, u8, u8, f64), Box<dyn Error>> {
    let bad = || -> Box<dyn Error> { format!("malformed epoch_iso {iso:?}").into() };
    let (date, time) = iso.split_once('T').ok_or_else(bad)?;
    let mut date_parts = date.split('-');
    let year: i32 = date_parts.next().ok_or_else(bad)?.parse()?;
    let month: u8 = date_parts.next().ok_or_else(bad)?.parse()?;
    let day: u8 = date_parts.next().ok_or_else(bad)?.parse()?;
    if date_parts.next().is_some() {
        return Err(bad());
    }
    let mut time_parts = time.split(':');
    let hour: u8 = time_parts.next().ok_or_else(bad)?.parse()?;
    let minute: u8 = time_parts.next().ok_or_else(bad)?.parse()?;
    let second: f64 = time_parts.next().ok_or_else(bad)?.parse()?;
    if time_parts.next().is_some() {
        return Err(bad());
    }
    let hours = f64::from(hour) + f64::from(minute) / 60.0 + second / 3600.0;
    Ok((year, month, day, hours))
}

/// True angular separation between two spherical directions, arcseconds.
/// Uses `atan2(|a×b|, a·b)` on unit vectors, which is accurate for tiny
/// angles and immune to RA wrap at 0/360°.
fn separation_arcsec(lon1_rad: f64, lat1_rad: f64, lon2_rad: f64, lat2_rad: f64) -> f64 {
    let a = spherical_to_cartesian(lon1_rad, lat1_rad, 1.0);
    let b = spherical_to_cartesian(lon2_rad, lat2_rad, 1.0);
    norm(cross(a, b)).atan2(dot(a, b)) / AS2R
}

/// Recomputes JD(TT) of `epoch_iso` and cross-checks the manifest `jd`
/// against it, returning the two-part epoch to drive the pipeline with.
fn epoch_from_fixture(fix: &Fixture) -> Result<JulianDate, Box<dyn Error>> {
    if fix.time_scale != "TT" {
        return Err(format!(
            "{}: manifest time_scale {:?} is not TT",
            fix.body, fix.time_scale
        )
        .into());
    }
    let (year, month, day, hours) = parse_iso_epoch(&fix.epoch_iso)?;
    let jd_tt = julday(Calendar::Gregorian, year, month, day, hours)
        .map_err(|e| format!("{}: julday({}): {e}", fix.body, fix.epoch_iso))?;
    let jd_err_days = (jd_tt.value() - fix.jd).abs();
    if jd_err_days > JD_CROSS_CHECK_TOL_DAYS {
        return Err(format!(
            "{}: manifest jd {} disagrees with julday({}) = {} by {jd_err_days} d",
            fix.body,
            fix.jd,
            fix.epoch_iso,
            jd_tt.value()
        )
        .into());
    }
    Ok(jd_tt)
}

/// Runs the three pipeline variants for one fixture body and returns the
/// comparison row.
fn evaluate_body(de: &DeFile<'_>, fix: &Fixture) -> Result<Row, Box<dyn Error>> {
    let target = target_for_command(&fix.command).ok_or_else(|| {
        format!(
            "{} ({}): unknown Horizons COMMAND code {:?}",
            fix.body, fix.source_file, fix.command
        )
    })?;
    let jd_tt = epoch_from_fixture(fix)?;

    // Quantity 1: ICRF, light-time only (no aberration, no deflection).
    let astrometric = apparent(
        de,
        target,
        jd_tt,
        Options::new(
            Center::Geocentric,
            Frame::Icrs,
            false,
            false,
            true,
            false,
            NutationModel::Iau2000a,
        ),
    )?;
    // Quantity 2: airless apparent, true equator and equinox of date
    // (full IAU 2000A_R06 nutation — the pipeline default, stated
    // explicitly so the oracle configuration is self-documenting).
    let apparent_tod = apparent(
        de,
        target,
        jd_tt,
        Options::new(
            Center::Geocentric,
            Frame::TrueOfDate,
            true,
            true,
            true,
            false,
            NutationModel::Iau2000a,
        ),
    )?;
    // Quantity 31: apparent ecliptic-of-date (informational only).
    let apparent_ecl = apparent(
        de,
        target,
        jd_tt,
        Options::new(
            Center::Geocentric,
            Frame::EclipticTrueOfDate,
            true,
            true,
            true,
            false,
            NutationModel::Iau2000a,
        ),
    )?;

    let fix_app_ra_rad = fix.apparent_ra_deg * DEG2RAD;
    let fix_app_dec_rad = fix.apparent_dec_deg * DEG2RAD;
    // Our apparent RA expressed on the Horizons RA origin (a pure rotation
    // about the shared true-of-date pole changes RA only, exactly).
    let tied_lon_rad = apparent_tod.lon_rad + horizons_equinox_tie_rad();
    Ok(Row {
        body: fix.body.clone(),
        astrometric_sep_arcsec: separation_arcsec(
            astrometric.lon_rad,
            astrometric.lat_rad,
            fix.astrometric_ra_deg * DEG2RAD,
            fix.astrometric_dec_deg * DEG2RAD,
        ),
        apparent_raw_sep_arcsec: separation_arcsec(
            apparent_tod.lon_rad,
            apparent_tod.lat_rad,
            fix_app_ra_rad,
            fix_app_dec_rad,
        ),
        apparent_tied_sep_arcsec: separation_arcsec(
            tied_lon_rad,
            apparent_tod.lat_rad,
            fix_app_ra_rad,
            fix_app_dec_rad,
        ),
        apparent_dra_cosdec_arcsec: normalize_pm_pi(apparent_tod.lon_rad - fix_app_ra_rad)
            * fix_app_dec_rad.cos()
            / AS2R,
        apparent_ddec_arcsec: (apparent_tod.lat_rad - fix_app_dec_rad) / AS2R,
        delta_diff_au: astrometric.r_au - fix.delta_au,
        ecl_dlon_arcsec: normalize_pm_pi(apparent_ecl.lon_rad - fix.obs_ecl_lon_deg * DEG2RAD)
            / AS2R,
        ecl_dlat_arcsec: (apparent_ecl.lat_rad - fix.obs_ecl_lat_deg * DEG2RAD) / AS2R,
    })
}

/// Prints the always-emitted per-body comparison table.
fn print_table(rows: &[Row]) {
    println!("Horizons geocentric spot-check, OxiEphemeris on DE440 vs Horizons on DE441");
    println!(
        "{:<18} {:>11} {:>11} {:>11} {:>12} {:>11} {:>12} {:>11} {:>11}",
        "body",
        "astro(\")",
        "appRaw(\")",
        "appTied(\")",
        "appDRAcosD",
        "appDDec(\")",
        "d_delta(AU)",
        "eclDlon(\")",
        "eclDlat(\")"
    );
    for row in rows {
        println!(
            "{:<18} {:>11.6} {:>11.6} {:>11.6} {:>12.4} {:>11.4} {:>12.3e} {:>11.4} {:>11.4}",
            row.body,
            row.astrometric_sep_arcsec,
            row.apparent_raw_sep_arcsec,
            row.apparent_tied_sep_arcsec,
            row.apparent_dra_cosdec_arcsec,
            row.apparent_ddec_arcsec,
            row.delta_diff_au,
            row.ecl_dlon_arcsec,
            row.ecl_dlat_arcsec
        );
    }
    println!(
        "note: appRaw is the direct separation against Horizons quantity 2 and \
         carries the documented RA-origin (equinox) offset between the \
         EOP-corrected IAU76/80 frame Horizons uses and the IAU 2006/2000A \
         frame used here (Horizons footer: -53 mas; derived from IERS TN36 \
         eq. 5.21 frame-bias angles: dalpha_0 + xi_0/tan(eps_0) = -52.93 mas). \
         appTied applies that published tie to our RA origin and is the gated \
         quantity; appDRAcosD/appDDec decompose the RAW residual (ours minus \
         Horizons) along and across the equator so the systematic stays \
         visible and reported."
    );
    println!(
        "note: eclDlon/eclDlat are informational only in phase 1 — Horizons \
         quantity 31 is the apparent IAU76/80 ecliptic-OF-DATE frame (its mean \
         obliquity differs from our IAU 2006 value by ~42 mas at J2000, plus a \
         possible apparent-vs-astrometric convention difference), so small \
         systematic deltas are expected and not asserted."
    );
}

/// The phase 1 exit-criterion test: ten Horizons fixtures, astrometric
/// and (frame-tied) apparent gates at < 0.01″, range gate at < 1e-6 AU.
#[test]
fn horizons_spotcheck_ten_bodies() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let Some(fixtures) = load_manifest()? else {
        return Ok(());
    };
    if fixtures.len() != 10 {
        return Err(format!(
            "expected 10 fixtures in manifest.json, found {}",
            fixtures.len()
        )
        .into());
    }
    let de = DeFile::parse(&bytes)?;

    let mut rows = Vec::with_capacity(fixtures.len());
    for fix in &fixtures {
        rows.push(evaluate_body(&de, fix)?);
    }

    // The table always prints, pass or fail (nextest shows the captured
    // stdout of failing tests).
    print_table(&rows);

    let mut failures = Vec::new();
    for row in &rows {
        // The explicit `is_finite` guard also traps a NaN separation.
        if !row.astrometric_sep_arcsec.is_finite()
            || row.astrometric_sep_arcsec >= ASTROMETRIC_TOL_ARCSEC
        {
            failures.push(format!(
                "{}: astrometric separation {:.6}\" >= {ASTROMETRIC_TOL_ARCSEC}\"",
                row.body, row.astrometric_sep_arcsec
            ));
        }
        if !row.apparent_tied_sep_arcsec.is_finite()
            || row.apparent_tied_sep_arcsec >= APPARENT_TOL_ARCSEC
        {
            failures.push(format!(
                "{}: frame-tied apparent separation {:.6}\" >= {APPARENT_TOL_ARCSEC}\" \
                 (raw, without the documented -52.93 mas equinox tie: {:.6}\")",
                row.body, row.apparent_tied_sep_arcsec, row.apparent_raw_sep_arcsec
            ));
        }
        // No separate raw-separation gate is needed: the tie is a fixed
        // 52.93 mas rotation, so passing the tied gate pins the raw
        // residual to 52.93 ± 10 mas by the triangle inequality — a real
        // pipeline error can never hide behind the tie.
        if !row.delta_diff_au.is_finite() || row.delta_diff_au.abs() >= DELTA_TOL_AU {
            failures.push(format!(
                "{}: |r_au - delta_au| = {:.3e} AU >= {DELTA_TOL_AU} AU",
                row.body,
                row.delta_diff_au.abs()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Horizons spot-check gate violations:\n{}",
        failures.join("\n")
    );
    Ok(())
}
