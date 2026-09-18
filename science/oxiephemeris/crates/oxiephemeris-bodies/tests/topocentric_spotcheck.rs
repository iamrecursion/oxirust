//! JPL Horizons **topocentric** spot-check of the apparent-place pipeline
//! with a ground-station observer (phase 2 Wave A; Moon parallax is the
//! sharp end-to-end test of the station geometry).
//!
//! The four fixtures under `tests/fixtures/horizons_topo/` are raw
//! responses of the public JPL Horizons API for an **observer site at
//! 139.7414° E, 35.6581° N, 40 m (Tokyo, airless)**, quantities 1, 2, 20,
//! epochs requested and returned in TT — see that directory's `README.md`
//! for provenance, the per-epoch IERS EOP values, and the WGS84
//! cross-check against Horizons' own cylindrical site coordinates.
//!
//! Comparisons and gates (per fixture):
//!
//! * **Astrometric RA/Dec** (quantity 1, ICRF, down-leg light time from
//!   the station) vs [`Frame::Icrs`] with aberration and deflection off:
//!   separation < 0.01″.
//! * **Apparent RA/Dec** (quantity 2, airless, EOP-corrected IAU76/80
//!   true-of-date) vs [`Frame::TrueOfDate`] with aberration + deflection,
//!   **after the documented −52.93 mas RA-origin (equinox) tie** between
//!   the Horizons frame realization and the IAU 2006/2000A one — the tie
//!   constants are duplicated from `horizons_spotcheck.rs`, which derives
//!   them from the IERS TN36 eq. (5.21) frame-bias angles (see the module
//!   docs there for the full derivation and the EOP-coverage caveat).
//!   Gate: tied separation < 0.01″; the raw (untied) separation is always
//!   printed so the systematic stays visible, never silently absorbed.
//! * **delta** (quantity 20, topocentric range, light-time aberrated) vs
//!   [`BodyPosition::r_au`]: |difference| < 5 × 10⁻¹¹ AU (≈ 7.5 m — the
//!   station position itself only enters at the ≈ 6378 km level, so this
//!   is a ≈ 10⁻⁶-relative test of the station chain).
//!
//! The EOP (UT1−UTC, `x_p`, `y_p`) for each epoch are **hardcoded** below
//! with the `data/iers/finals2000A.all` source line quoted in a comment,
//! and cross-checked against the fixture manifest so the two records
//! cannot drift apart.
//!
//! Ephemeris caveat: Horizons served DE441, this test evaluates DE440 —
//! sub-mas / sub-km for these targets and epochs (see the geocentric
//! spot-check module docs). Both the DE440 binary and the fixture
//! directory are located at runtime; the tests skip gracefully (print a
//! note, return `Ok`) when either is absent.

use std::error::Error;
use std::path::PathBuf;

use serde::Deserialize;

use oxiephemeris_bodies::apparent::{
    apparent, apparent_topocentric, BodiesError, Center, Frame, Options, Target,
};
use oxiephemeris_bodies::frames::{mean_obliquity_iau2006, NutationModel};
use oxiephemeris_bodies::math::{cross, dot, norm, scale, spherical_to_cartesian, sub};
use oxiephemeris_bodies::topocentric::{station_gcrs_state_m, Eop, Observer, TopocentricObserver};
use oxiephemeris_core::angle::{normalize_pm_pi, AS2R, DEG2RAD};
use oxiephemeris_core::time::{julday, Calendar, JulianDate};
use oxiephemeris_de::DeFile;

type TestResult = Result<(), Box<dyn Error>>;

/// Hard gate for the astrometric (ICRF, light-time-only) comparison, in
/// arcseconds — same criterion as the geocentric spot-check.
const ASTROMETRIC_TOL_ARCSEC: f64 = 0.01;

/// Hard gate for the apparent (true-of-date, airless) comparison after
/// the equinox frame tie, in arcseconds.
const APPARENT_TOL_ARCSEC: f64 = 0.01;

/// Hard gate for the topocentric range vs Horizons quantity 20, AU
/// (5 × 10⁻¹¹ AU ≈ 7.5 m).
const DELTA_TOL_AU: f64 = 5e-11;

/// Manifest-integrity gate: recomputed JD(TT) of `epoch_iso` must match
/// the manifest `jd` to below a millisecond.
const JD_CROSS_CHECK_TOL_DAYS: f64 = 1e-8;

/// ICRS RA-origin offset from the J2000.0 mean equinox, arcsec — the
/// frame bias `dα₀` (IERS TN36 §5.5.4, eq. 5.21). **Duplicated from
/// `tests/horizons_spotcheck.rs`**, which documents the full derivation
/// of the Horizons equinox tie; keep the two copies in sync.
const FRAME_BIAS_DALPHA0_ARCSEC: f64 = -0.014_60;

/// GCRS frame-bias pole offset `ξ₀`, arcsec (IERS TN36 eq. 5.21).
/// Duplicated from `tests/horizons_spotcheck.rs` — see above.
const FRAME_BIAS_XI0_ARCSEC: f64 = -0.016_617_0;

/// RA-origin tie of the Horizons apparent frame (EOP-corrected IAU76/80)
/// relative to the IAU 2006/2000A true-of-date frame, radians:
/// `RA(Horizons) = RA(IAU 2006/2000A) + (dα₀ + ξ₀ / tan ε₀)` = −52.93 mas
/// (identical function in `tests/horizons_spotcheck.rs`, where the module
/// docs give the first-order derivation; Horizons' own footer documents
/// "−53 mas").
fn horizons_equinox_tie_rad() -> f64 {
    let eps0 = mean_obliquity_iau2006(0.0);
    (FRAME_BIAS_DALPHA0_ARCSEC + FRAME_BIAS_XI0_ARCSEC / eps0.tan()) * AS2R
}

/// The fixture observer site: 139.7414° E, 35.6581° N, 40 m (Tokyo),
/// WGS84 — exactly the `SITE_COORD` of the Horizons requests.
const SITE: Observer = Observer {
    latitude_deg: 35.6581,
    longitude_deg: 139.7414,
    height_m: 40.0,
};

/// Per-epoch IERS Earth-orientation parameters, hardcoded from
/// `data/iers/finals2000A.all` (IERS combined series; all four rows are
/// final values, flag `I`). Each entry quotes its source line verbatim
/// (trailing padding trimmed); the same values live in the fixture
/// `manifest.json` and `README.md`, and the test cross-checks the two.
/// The rows are tabulated at 0h UTC of the epoch's calendar date,
/// ≈ 1 minute after the 00:00-TT epoch — see the fixture README for why
/// no interpolation is warranted (≲ 2 µs of UT1 over that minute).
const EOP_BY_SOURCE: [(&str, Eop); 4] = [
    // "20 115 58863.00 I  0.059576 0.000031  0.293792 0.000018  I-0.1799645 0.0000022  0.7463 0.0017  I"
    (
        "moon_2020.txt",
        Eop::new(-0.179_964_5, 0.059_576, 0.293_792),
    ),
    // "87 410 46895.00 I  0.086991 0.000262  0.210887 0.000501  I-0.2910912 0.0000149  1.7858 0.0102  I"
    (
        "moon_1987.txt",
        Eop::new(-0.291_091_2, 0.086_991, 0.210_887),
    ),
    // " 3 827 52878.00 I  0.260148 0.000064  0.412496 0.000080  I-0.3492927 0.0000067 -0.0703 0.0055  I"
    (
        "mars_2003.txt",
        Eop::new(-0.349_292_7, 0.260_148, 0.412_496),
    ),
    // "15 4 1 57113.00 I  0.013822 0.000026  0.396474 0.000048  I-0.5750634 0.0000072  1.3823 0.0050  I"
    (
        "venus_2015.txt",
        Eop::new(-0.575_063_4, 0.013_822, 0.396_474),
    ),
];

/// One entry of `manifest.json` (unused JSON fields are ignored by
/// `serde` by default).
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
    site_east_lon_deg: f64,
    site_lat_deg: f64,
    site_height_m: f64,
    dut1_s: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
    source_file: String,
}

/// Per-fixture comparison results, kept so the full table always prints
/// before any gate is asserted. Column semantics match the geocentric
/// spot-check: `apparent_raw_sep_arcsec` carries the documented ≈ −53 mas
/// equinox systematic, `apparent_tied_sep_arcsec` is the gated quantity,
/// and the `dra·cosδ` / `ddec` pair decomposes the **raw** residual.
struct Row {
    label: String,
    astrometric_sep_arcsec: f64,
    apparent_raw_sep_arcsec: f64,
    apparent_tied_sep_arcsec: f64,
    apparent_dra_cosdec_arcsec: f64,
    apparent_ddec_arcsec: f64,
    delta_diff_au: f64,
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
        .join("horizons_topo")
        .join("manifest.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        println!(
            "note: skipping test — topocentric Horizons manifest not found at {}",
            path.display()
        );
        return Ok(None);
    };
    let fixtures: Vec<Fixture> = serde_json::from_str(&text)
        .map_err(|e| format!("manifest.json is present but unparsable: {e}"))?;
    Ok(Some(fixtures))
}

/// Horizons `COMMAND` code → pipeline target (301 = Moon body center,
/// 4 = Mars **system barycenter** — exactly what `Target::Mars`
/// evaluates on a DE file — 299 = Venus body center).
fn target_for_command(command: &str) -> Option<Target> {
    match command {
        "301" => Some(Target::Moon),
        "4" => Some(Target::Mars),
        "299" => Some(Target::Venus),
        _ => None,
    }
}

/// The hardcoded EOP entry for a fixture source file.
fn eop_for_source(source_file: &str) -> Option<Eop> {
    EOP_BY_SOURCE
        .iter()
        .find(|(name, _)| *name == source_file)
        .map(|(_, eop)| *eop)
}

/// Parses `YYYY-MM-DDTHH:MM:SS` into `(year, month, day, hours)`
/// (hand-rolled; project policy: no chrono).
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

/// True angular separation between two spherical directions, arcseconds
/// (`atan2(|a×b|, a·b)` on unit vectors — accurate for tiny angles,
/// immune to RA wrap).
fn separation_arcsec(lon1_rad: f64, lat1_rad: f64, lon2_rad: f64, lat2_rad: f64) -> f64 {
    let a = spherical_to_cartesian(lon1_rad, lat1_rad, 1.0);
    let b = spherical_to_cartesian(lon2_rad, lat2_rad, 1.0);
    norm(cross(a, b)).atan2(dot(a, b)) / AS2R
}

/// Angle between two vectors in radians.
fn angle_between_rad(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

/// Validates the fixture's site + EOP against the hardcoded constants and
/// returns the ready-to-use observer bundle.
fn observer_for_fixture(fix: &Fixture) -> Result<TopocentricObserver, Box<dyn Error>> {
    if (fix.site_east_lon_deg - SITE.longitude_deg).abs() > 1e-12
        || (fix.site_lat_deg - SITE.latitude_deg).abs() > 1e-12
        || (fix.site_height_m - SITE.height_m).abs() > 1e-9
    {
        return Err(format!("{}: manifest site differs from the test SITE", fix.body).into());
    }
    let eop = eop_for_source(&fix.source_file)
        .ok_or_else(|| format!("{}: no hardcoded EOP for {}", fix.body, fix.source_file))?;
    // Integrity: the hardcoded EOP and the manifest record must agree
    // exactly (both are transcriptions of the same finals2000A.all line).
    if (eop.dut1_s - fix.dut1_s).abs() > 0.0
        || (eop.xp_arcsec - fix.xp_arcsec).abs() > 0.0
        || (eop.yp_arcsec - fix.yp_arcsec).abs() > 0.0
    {
        return Err(format!(
            "{}: hardcoded EOP disagrees with manifest for {}",
            fix.body, fix.source_file
        )
        .into());
    }
    Ok(TopocentricObserver::new(SITE, eop))
}

/// Recomputes JD(TT) of `epoch_iso` and cross-checks the manifest `jd`.
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

/// Runs the two pipeline variants for one fixture and returns the row.
fn evaluate_fixture(de: &DeFile<'_>, fix: &Fixture) -> Result<Row, Box<dyn Error>> {
    let target = target_for_command(&fix.command).ok_or_else(|| {
        format!(
            "{} ({}): unknown Horizons COMMAND code {:?}",
            fix.body, fix.source_file, fix.command
        )
    })?;
    let jd_tt = epoch_from_fixture(fix)?;
    let topo = observer_for_fixture(fix)?;

    // Quantity 1: ICRF, down-leg light time from the station only.
    let astrometric = apparent_topocentric(
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
        &topo,
    )?;
    // Quantity 2: airless apparent, true equator and equinox of date
    // (station light time + deflection + aberration incl. the diurnal
    // term from the station velocity; full IAU 2000A_R06 nutation — the
    // pipeline default, stated explicitly so the oracle configuration is
    // self-documenting).
    let apparent_tod = apparent_topocentric(
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
        &topo,
    )?;

    let fix_app_ra_rad = fix.apparent_ra_deg * DEG2RAD;
    let fix_app_dec_rad = fix.apparent_dec_deg * DEG2RAD;
    let tied_lon_rad = apparent_tod.lon_rad + horizons_equinox_tie_rad();
    Ok(Row {
        label: format!("{} {}", fix.body, fix.epoch_iso),
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
    })
}

/// Prints the always-emitted per-fixture comparison table (raw **and**
/// tied residuals for every case, per the gate policy).
fn print_table(rows: &[Row]) {
    println!(
        "Horizons topocentric spot-check (site 139.7414 E, 35.6581 N, 40 m), \
         OxiEphemeris on DE440 vs Horizons on DE441"
    );
    println!(
        "{:<32} {:>11} {:>11} {:>11} {:>12} {:>11} {:>12}",
        "fixture",
        "astro(\")",
        "appRaw(\")",
        "appTied(\")",
        "appDRAcosD",
        "appDDec(\")",
        "d_delta(AU)"
    );
    for row in rows {
        println!(
            "{:<32} {:>11.6} {:>11.6} {:>11.6} {:>12.4} {:>11.4} {:>12.3e}",
            row.label,
            row.astrometric_sep_arcsec,
            row.apparent_raw_sep_arcsec,
            row.apparent_tied_sep_arcsec,
            row.apparent_dra_cosdec_arcsec,
            row.apparent_ddec_arcsec,
            row.delta_diff_au
        );
    }
    println!(
        "note: appRaw carries the documented RA-origin (equinox) offset between \
         Horizons' EOP-corrected IAU76/80 frame and the IAU 2006/2000A frame \
         used here (-52.93 mas, IERS TN36 eq. 5.21 frame-bias angles; see \
         tests/horizons_spotcheck.rs for the derivation). appTied applies that \
         published tie and is the gated quantity; appDRAcosD/appDDec decompose \
         the RAW residual (ours minus Horizons)."
    );
}

/// The topocentric exit-criterion test: four fixtures, astrometric and
/// frame-tied apparent gates at < 0.01″, range gate at < 5e-11 AU.
#[test]
fn horizons_topocentric_spotcheck() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let Some(fixtures) = load_manifest()? else {
        return Ok(());
    };
    if fixtures.len() != 4 {
        return Err(format!(
            "expected 4 fixtures in manifest.json, found {}",
            fixtures.len()
        )
        .into());
    }
    let de = DeFile::parse(&bytes)?;

    let mut rows = Vec::with_capacity(fixtures.len());
    for fix in &fixtures {
        rows.push(evaluate_fixture(&de, fix)?);
    }
    print_table(&rows);

    let mut failures = Vec::new();
    for row in &rows {
        if !row.astrometric_sep_arcsec.is_finite()
            || row.astrometric_sep_arcsec >= ASTROMETRIC_TOL_ARCSEC
        {
            failures.push(format!(
                "{}: astrometric separation {:.6}\" >= {ASTROMETRIC_TOL_ARCSEC}\"",
                row.label, row.astrometric_sep_arcsec
            ));
        }
        if !row.apparent_tied_sep_arcsec.is_finite()
            || row.apparent_tied_sep_arcsec >= APPARENT_TOL_ARCSEC
        {
            failures.push(format!(
                "{}: frame-tied apparent separation {:.6}\" >= {APPARENT_TOL_ARCSEC}\" \
                 (raw, without the documented -52.93 mas equinox tie: {:.6}\")",
                row.label, row.apparent_tied_sep_arcsec, row.apparent_raw_sep_arcsec
            ));
        }
        if !row.delta_diff_au.is_finite() || row.delta_diff_au.abs() >= DELTA_TOL_AU {
            failures.push(format!(
                "{}: |r_au - delta_au| = {:.3e} AU >= {DELTA_TOL_AU} AU",
                row.label,
                row.delta_diff_au.abs()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "topocentric spot-check gate violations:\n{}",
        failures.join("\n")
    );
    Ok(())
}

/// Moon parallax sanity: the topocentric-vs-geocentric direction shift
/// must sit in the 0.15°–1.02° band for the fixture geometry and satisfy
/// the exact sine-rule identity of the geocenter–station–Moon triangle,
/// `sin p = ρ sin z / d_topo` (angle at the Moon vs angle at the
/// geocenter), where `ρ` is the station's geocentric distance, `z` the
/// geocentric zenith distance of the Moon at the station and `d_topo`
/// the topocentric range. Light-time differences between the geocentric
/// and topocentric runs break the identity only at the ≈ 10⁻⁷-rad level.
#[test]
fn moon_topocentric_parallax_is_consistent() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let Some(fixtures) = load_manifest()? else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let astrometric_opts = Options::new(
        Center::Geocentric,
        Frame::Icrs,
        false,
        false,
        true,
        false,
        NutationModel::Iau2000a,
    );

    let mut checked = 0_usize;
    for fix in fixtures.iter().filter(|f| f.command == "301") {
        let jd_tt = epoch_from_fixture(fix)?;
        let topo = observer_for_fixture(fix)?;

        let geo = apparent(&de, Target::Moon, jd_tt, astrometric_opts)?;
        let top = apparent_topocentric(&de, Target::Moon, jd_tt, astrometric_opts, &topo)?;
        let (station_m, _) = station_gcrs_state_m(&topo.site, &topo.eop, jd_tt)?;
        let station_au = scale(station_m, 1.0 / (de.au_km() * 1_000.0));

        let parallax_rad = angle_between_rad(geo.position_au, top.position_au);
        let parallax_deg = parallax_rad / DEG2RAD;
        println!(
            "{}: parallax {parallax_deg:.5} deg (geocentric zenith distance {:.2} deg, \
             topocentric range {:.8} AU)",
            fix.epoch_iso,
            angle_between_rad(station_au, geo.position_au) / DEG2RAD,
            top.r_au
        );
        assert!(
            (0.15..=1.02).contains(&parallax_deg),
            "{}: Moon parallax {parallax_deg} deg outside the fixture-geometry band",
            fix.epoch_iso
        );

        // Sine rule in the geocenter–station–Moon triangle.
        let zenith_rad = angle_between_rad(station_au, geo.position_au);
        let predicted_sin_p = norm(station_au) * zenith_rad.sin() / top.r_au;
        let residual = (parallax_rad.sin() - predicted_sin_p).abs();
        println!(
            "{}: sine-rule residual |sin p - rho sin z / d| = {residual:.3e}",
            fix.epoch_iso
        );
        assert!(
            residual < 5e-7,
            "{}: sine-rule residual {residual:.3e} >= 5e-7",
            fix.epoch_iso
        );

        // The station offset really is what the parallax responds to:
        // geo − top = station + [moon(t−τ_geo) − moon(t−τ_topo)]. The
        // second (retarded-epoch) term is bounded by the Moon's
        // *barycentric* speed (≈ 31 km/s, dominated by the Earth's
        // heliocentric motion) times |Δτ| ≤ ρ/c ≈ 21 ms, i.e. ≤ 0.7 km;
        // observed ≈ 0.14 km here. Everything beyond that must close.
        let apparent_shift = sub(geo.position_au, top.position_au);
        let shift_error_km = norm(sub(apparent_shift, station_au)) * de.au_km();
        println!(
            "{}: closure |(geo - top) - station| = {shift_error_km:.4} km \
             (retarded-epoch bound ~0.7 km)",
            fix.epoch_iso
        );
        assert!(
            shift_error_km < 1.0,
            "{}: position closure error {shift_error_km} km",
            fix.epoch_iso
        );
        checked += 1;
    }
    assert!(checked == 2, "expected 2 Moon fixtures, found {checked}");
    Ok(())
}

/// A topocentric observer with a non-geocentric center is a typed error,
/// and the pre-1972 leap-second limit surfaces as `BodiesError::Time`.
#[test]
fn topocentric_center_and_epoch_validation() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let topo = TopocentricObserver::new(SITE, Eop::default());
    let jd_tt = julday(Calendar::Gregorian, 2020, 1, 15, 0.0)?;

    for center in [Center::Heliocentric, Center::Barycentric] {
        let mut opts = Options::default();
        opts.center = center;
        let got = apparent_topocentric(&de, Target::Moon, jd_tt, opts, &topo);
        assert!(
            matches!(got, Err(BodiesError::ObserverNotGeocentric)),
            "{center:?}: expected ObserverNotGeocentric, got {got:?}"
        );
    }

    let pre_1972 = julday(Calendar::Gregorian, 1960, 1, 1, 0.0)?;
    let got = apparent_topocentric(&de, Target::Moon, pre_1972, Options::default(), &topo);
    assert!(
        matches!(got, Err(BodiesError::Time(_))),
        "expected BodiesError::Time for a 1960 epoch, got {got:?}"
    );
    Ok(())
}
