//! Integration tests for the `oxieph` binary: spawns the compiled binary
//! (`env!("CARGO_BIN_EXE_oxieph")`) via `std::process::Command`, exactly as
//! CLAUDE.md's test plan requires (no new dev-dependencies: the
//! `oxiephemeris-core`/`-de`/`-bodies` crates used below are already
//! regular dependencies of `oxiephemeris-cli`).

use std::path::PathBuf;
use std::process::Command;

use oxiephemeris_bodies::apparent::apparent;
use oxiephemeris_bodies::{Center, Frame, Options};
use oxiephemeris_core::angle::RAD2DEG;
use oxiephemeris_core::time::{julday, revjul, Calendar, JulianDate};
use oxiephemeris_de::DeFile;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_oxieph"))
}

/// Path to the DE440 file, resolved at *compile* time relative to this
/// crate's manifest directory (not a hardcoded absolute path: it is
/// derived from `CARGO_MANIFEST_DIR`, the standard way to locate
/// workspace-relative fixtures from an integration test).
fn de440_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440")
}

fn run(args: &[&str]) -> (bool, String, String) {
    let output = match Command::new(bin()).args(args).output() {
        Ok(o) => o,
        Err(e) => panic!("failed to spawn oxieph: {e}"),
    };
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn extract_value<'a>(stdout: &'a str, key: &str) -> Option<&'a str> {
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            if let Some(rest) = rest.trim_start().strip_prefix('=') {
                return Some(rest.trim());
            }
        }
    }
    None
}

#[test]
fn convert_date_to_jd_matches_core_julday() -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&["convert", "--date", "2000-01-01T12:00:00"]);
    assert!(ok, "stderr: {stderr}");
    let jd_str = extract_value(&stdout, "jd").ok_or("missing jd= line")?;
    let jd_cli: f64 = jd_str.parse()?;

    let expected = julday(Calendar::Gregorian, 2000, 1, 1, 12.0)?;
    assert!(
        (jd_cli - expected.value()).abs() < 1e-9,
        "cli={jd_cli} expected={}",
        expected.value()
    );
    Ok(())
}

/// Parses a printed `HH:MM:SS.ffffff` time-of-day into seconds since
/// midnight, for tolerance comparisons (the printed `jd` is truncated to 9
/// decimal digits, i.e. ~86 microseconds, so a round trip through the
/// printed text is only exact to about that tolerance).
fn hms_to_seconds(time: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let mut parts = time.split(':');
    let h: f64 = parts.next().ok_or("missing hour")?.parse()?;
    let m: f64 = parts.next().ok_or("missing minute")?.parse()?;
    let s: f64 = parts.next().ok_or("missing second")?.parse()?;
    Ok(h * 3600.0 + m * 60.0 + s)
}

#[test]
fn convert_round_trips_date_jd_date() -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&["convert", "--date", "2024-03-15T06:30:00"]);
    assert!(ok, "stderr: {stderr}");
    let jd_str = extract_value(&stdout, "jd").ok_or("missing jd")?;

    let (ok, stdout, stderr) = run(&["convert", "--jd", jd_str]);
    assert!(ok, "stderr: {stderr}");
    let date = extract_value(&stdout, "date").ok_or("missing date")?;
    let time = extract_value(&stdout, "time").ok_or("missing time")?;
    assert_eq!(date, "2024-03-15");
    let printed_seconds = hms_to_seconds(time)?;
    let expected_seconds = 6.0 * 3600.0 + 30.0 * 60.0;
    assert!(
        (printed_seconds - expected_seconds).abs() < 1e-3,
        "time={time}"
    );
    Ok(())
}

#[test]
fn convert_bce_date_prints_astronomical_year() -> Result<(), Box<dyn std::error::Error>> {
    // Year -1 (astronomical) == 2 BCE.
    let (ok, stdout, stderr) = run(&["convert", "--date=-0001-06-15"]);
    assert!(ok, "stderr: {stderr}");
    let jd_str = extract_value(&stdout, "jd").ok_or("missing jd")?;
    let jd_cli: f64 = jd_str.parse()?;

    let expected = julday(Calendar::Gregorian, -1, 6, 15, 0.0)?;
    assert!((jd_cli - expected.value()).abs() < 1e-9);

    let (ok, stdout, stderr) = run(&["convert", "--jd", jd_str]);
    assert!(ok, "stderr: {stderr}");
    let date = extract_value(&stdout, "date").ok_or("missing date")?;
    assert_eq!(date, "-0001-06-15");
    Ok(())
}

#[test]
fn convert_jd_accepts_fractional_and_negative() -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&["convert", "--jd", "-100.25"]);
    assert!(ok, "stderr: {stderr}");

    let (date, hours) = revjul(JulianDate::from_f64(-100.25), Calendar::Gregorian)?;
    let expected_date = format!("{:04}-{:02}-{:02}", date.year, date.month, date.day);
    let printed_date = extract_value(&stdout, "date").ok_or("missing date")?;
    assert_eq!(printed_date, expected_date);
    assert!((hours - 6.0).abs() < 1e-6, "hours={hours}");
    Ok(())
}

#[test]
fn convert_rejects_both_jd_and_date() {
    let (ok, _stdout, stderr) = run(&["convert", "--jd", "0", "--date", "2000-01-01"]);
    assert!(!ok);
    assert!(stderr.contains("exactly one"), "stderr={stderr}");
}

#[test]
fn convert_rejects_neither_jd_nor_date() {
    let (ok, _stdout, stderr) = run(&["convert"]);
    assert!(!ok);
    assert!(stderr.contains("specify one of"), "stderr={stderr}");
}

#[test]
fn convert_date_to_jd_json_matches_direct_library_computation(
) -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&["convert", "--date", "2024-03-15T06:30:15.5", "--json"]);
    assert!(ok, "stderr: {stderr}");
    let json: serde_json::Value = serde_json::from_str(stdout.trim())?;

    assert_eq!(json["year"], 2024);
    assert_eq!(json["month"], 3);
    assert_eq!(json["day"], 15);
    assert_eq!(json["hour"], 6);
    assert_eq!(json["minute"], 30);
    let second = json["second"].as_f64().ok_or("second missing")?;
    assert!((second - 15.5).abs() < 1e-9, "second={second}");
    assert_eq!(json["calendar"], "gregorian");
    assert_eq!(json["scale"], "utc");

    let jd = json["jd"].as_f64().ok_or("jd missing")?;
    let expected = julday(Calendar::Gregorian, 2024, 3, 15, 6.5 + 15.5 / 3600.0)?;
    assert!(
        (jd - expected.value()).abs() < 1e-12,
        "jd={jd} expected={}",
        expected.value()
    );
    Ok(())
}

#[test]
fn convert_jd_to_date_json_matches_direct_library_computation(
) -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&["convert", "--jd", "2451545.25", "--json"]);
    assert!(ok, "stderr: {stderr}");
    let json: serde_json::Value = serde_json::from_str(stdout.trim())?;

    // Bit-exact comparison (not `==`, which clippy's `float_cmp` flags):
    // this JD is exactly representable, so the round trip through JSON
    // must reproduce the identical `f64` bit pattern.
    assert_eq!(
        json["jd"].as_f64().ok_or("jd missing")?.to_bits(),
        2_451_545.25_f64.to_bits()
    );
    assert_eq!(json["calendar"], "gregorian");
    assert_eq!(json["scale"], "utc");

    let (date, hours) = revjul(JulianDate::from_f64(2_451_545.25), Calendar::Gregorian)?;
    assert_eq!(json["year"], date.year);
    assert_eq!(json["month"], u64::from(date.month));
    assert_eq!(json["day"], u64::from(date.day));

    let hour = json["hour"].as_f64().ok_or("hour missing")?;
    let minute = json["minute"].as_f64().ok_or("minute missing")?;
    let second = json["second"].as_f64().ok_or("second missing")?;
    let printed_seconds = hour * 3600.0 + minute * 60.0 + second;
    assert!(
        (printed_seconds - hours * 3600.0).abs() < 1e-6,
        "printed={printed_seconds} expected={}",
        hours * 3600.0
    );
    Ok(())
}

/// Full-precision round trip: `convert --date ... --json` -> extract `jd`
/// -> `convert --jd <jd> --json` -> the calendar fields must match the
/// original input exactly.
///
/// The epoch is midnight (`hours = 0`) so `julday`'s two-part split has an
/// exactly-zero `lo` and collapsing to a single `f64` for the JSON `jd`
/// field loses nothing; a fractional time of day would instead be subject
/// to the single-`f64`-JD precision limit that `oxiephemeris_core`'s
/// `JulianDate` module docs describe (~40 microseconds for contemporary
/// dates) — a documented property of the format, not a bug, and orthogonal
/// to what this test checks (that the JSON schema itself round-trips
/// losslessly).
#[test]
fn convert_json_round_trips_date_jd_date_at_full_precision(
) -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&["convert", "--date", "2024-03-15T00:00:00", "--json"]);
    assert!(ok, "stderr: {stderr}");
    let forward: serde_json::Value = serde_json::from_str(stdout.trim())?;
    let jd = forward["jd"].as_f64().ok_or("jd missing")?;

    // Round-trip the *exact* printed jd text (serde_json prints the
    // shortest string that round-trips to the same f64), not a re-formatted
    // `f64` (which could differ if we let Rust's default `Display` re-round
    // it), so this really exercises the JSON schema end-to-end.
    let jd_text = forward["jd"].to_string();
    let (ok, stdout, stderr) = run(&["convert", "--jd", &jd_text, "--json"]);
    assert!(ok, "stderr: {stderr}");
    let back: serde_json::Value = serde_json::from_str(stdout.trim())?;

    // Bit-exact comparisons (not `==`, which clippy's `float_cmp` flags):
    // this is the full-precision round trip the test exists to check.
    assert_eq!(
        back["jd"].as_f64().ok_or("jd missing")?.to_bits(),
        jd.to_bits()
    );
    assert_eq!(back["year"], 2024);
    assert_eq!(back["month"], 3);
    assert_eq!(back["day"], 15);
    assert_eq!(back["hour"], 0);
    assert_eq!(back["minute"], 0);
    assert_eq!(
        back["second"].as_f64().ok_or("second missing")?.to_bits(),
        0.0_f64.to_bits()
    );
    Ok(())
}

/// A `--date` input with a non-zero UTC offset must emit JSON calendar
/// fields describing the UTC-normalized instant (the one `jd` encodes),
/// not an echo of the raw local-time input: `2024-01-01T00:30:00+09:00`
/// is 2023-12-31 15:30:00 UTC, the previous calendar day.
#[test]
fn convert_date_to_jd_json_normalizes_utc_offset() -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&["convert", "--date", "2024-01-01T00:30:00+09:00", "--json"]);
    assert!(ok, "stderr: {stderr}");
    let json: serde_json::Value = serde_json::from_str(stdout.trim())?;

    assert_eq!(json["year"], 2023);
    assert_eq!(json["month"], 12);
    assert_eq!(json["day"], 31);
    assert_eq!(json["hour"], 15);
    assert_eq!(json["minute"], 30);
    let second = json["second"].as_f64().ok_or("second missing")?;
    assert!(second.abs() < 1e-6, "second={second}");

    // The jd must be the same one the CLI computes internally
    // (julday of the local reading with the offset folded into hours).
    let jd = json["jd"].as_f64().ok_or("jd missing")?;
    let expected = julday(Calendar::Gregorian, 2024, 1, 1, 0.5 - 9.0)?;
    assert!(
        (jd - expected.value()).abs() < 1e-12,
        "jd={jd} expected={}",
        expected.value()
    );
    // ... and identical to the direct previous-day-UTC computation.
    let same_instant = julday(Calendar::Gregorian, 2023, 12, 31, 15.5)?;
    assert!(
        (jd - same_instant.value()).abs() < 1e-12,
        "jd={jd} same_instant={}",
        same_instant.value()
    );
    Ok(())
}

#[test]
fn convert_rejects_calendar_invalid_date_february_30() {
    let (ok, _stdout, stderr) = run(&["convert", "--date", "2026-02-30"]);
    assert!(!ok);
    assert!(stderr.contains("2026-02-30"), "stderr={stderr}");
    assert!(stderr.contains("28 days"), "stderr={stderr}");
    assert!(
        !stderr.contains("invalid input to a conversion"),
        "stderr={stderr}"
    );
}

#[test]
fn convert_rejects_calendar_invalid_date_feb_29_non_leap_year() {
    // 2026 is not a leap year (not divisible by 4).
    let (ok, _stdout, stderr) = run(&["convert", "--date", "2026-02-29"]);
    assert!(!ok);
    assert!(stderr.contains("2026-02-29"), "stderr={stderr}");
    assert!(stderr.contains("not a leap year"), "stderr={stderr}");
}

#[test]
fn convert_accepts_feb_29_in_a_leap_year() {
    let (ok, _stdout, stderr) = run(&["convert", "--date", "2024-02-29"]);
    assert!(ok, "stderr: {stderr}");
}

#[test]
fn convert_rejects_nan_jd_with_a_friendly_message() {
    let (ok, _stdout, stderr) = run(&["convert", "--jd", "NaN"]);
    assert!(!ok);
    assert!(stderr.contains("--jd"), "stderr={stderr}");
    assert!(
        !stderr.contains("invalid input to a conversion"),
        "stderr={stderr}"
    );
}

#[test]
fn convert_rejects_infinite_jd_with_a_friendly_message() {
    let (ok, _stdout, stderr) = run(&["convert", "--jd", "inf"]);
    assert!(!ok);
    assert!(stderr.contains("--jd"), "stderr={stderr}");
    assert!(
        !stderr.contains("invalid input to a conversion"),
        "stderr={stderr}"
    );
}

#[test]
fn pos_rejects_calendar_invalid_date_with_a_friendly_message() {
    let (ok, _stdout, stderr) = run(&["pos", "mars", "2026-02-30T00:00:00"]);
    assert!(!ok);
    assert!(stderr.contains("2026-02-30"), "stderr={stderr}");
    assert!(
        !stderr.contains("invalid input to a conversion"),
        "stderr={stderr}"
    );
}

#[test]
fn pos_unknown_body_is_a_friendly_error() {
    let (ok, _stdout, stderr) = run(&["pos", "earth", "2000-01-01T00:00:00"]);
    assert!(!ok);
    assert!(stderr.contains("unknown body"), "stderr={stderr}");
}

#[test]
fn pos_missing_de_file_is_a_friendly_error() {
    let (ok, _stdout, stderr) = run(&[
        "pos",
        "sun",
        "2000-01-01T12:00:00",
        "--scale",
        "tt",
        "--de",
        "/definitely/does/not/exist.440",
    ]);
    assert!(!ok);
    assert!(stderr.contains("fetch_de440.sh"), "stderr={stderr}");
    assert!(stderr.contains("OXIEPH_DE"), "stderr={stderr}");
    // The repo-independent direct-download hint must survive too, so a
    // `cargo install` user (no repo) still learns how to obtain DE440.
    assert!(stderr.contains("ssd.jpl.nasa.gov"), "stderr={stderr}");
}

#[test]
fn pos_sun_json_matches_direct_library_computation() -> Result<(), Box<dyn std::error::Error>> {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!(
            "note: skipping (DE440 fixture not present at {}); run scripts/fetch_de440.sh",
            de_path.display()
        );
        return Ok(());
    }

    let de_path_str = de_path.to_string_lossy().into_owned();
    let (ok, stdout, stderr) = run(&[
        "pos",
        "sun",
        "2000-01-01T12:00:00",
        "--scale",
        "tt",
        "--de",
        &de_path_str,
        "--json",
    ]);
    assert!(ok, "stderr: {stderr}");

    let json: serde_json::Value = serde_json::from_str(stdout.trim())?;
    assert_eq!(json["body"], "Sun");
    assert_eq!(json["frame"], "ecliptic");
    assert_eq!(json["center"], "geocentric");

    // Sanity range from CLAUDE.md's test plan.
    let lon_deg = json["lon_deg"].as_f64().ok_or("lon_deg missing")?;
    let lat_deg = json["lat_deg"].as_f64().ok_or("lat_deg missing")?;
    assert!((279.0..282.0).contains(&lon_deg), "lon_deg={lon_deg}");
    assert!(lat_deg.abs() < 0.01, "lat_deg={lat_deg}");

    // Independent computation through the library, bypassing the CLI.
    let bytes = std::fs::read(&de_path)?;
    let de = DeFile::parse(&bytes)?;
    let jd_tt = julday(Calendar::Gregorian, 2000, 1, 1, 12.0)?; // --scale tt: no leap seconds
                                                                // Must mirror the CLI's own Options (unexposed knobs at defaults).
    let mut opts = Options::default();
    opts.center = Center::Geocentric;
    opts.frame = Frame::EclipticTrueOfDate;
    opts.aberration = true;
    opts.deflection = true;
    opts.with_speed = true;
    let place = apparent(&de, oxiephemeris_bodies::Target::Sun, jd_tt, opts)?;
    let expected_lon_deg = place.lon_rad * RAD2DEG;
    let expected_lat_deg = place.lat_rad * RAD2DEG;
    let expected_distance_au = place.r_au;
    let expected_light_time = place.light_time_days;

    assert!(
        (lon_deg - expected_lon_deg).abs() < 1e-9,
        "cli={lon_deg} lib={expected_lon_deg}"
    );
    assert!(
        (lat_deg - expected_lat_deg).abs() < 1e-9,
        "cli={lat_deg} lib={expected_lat_deg}"
    );
    let distance_au = json["distance_au"].as_f64().ok_or("distance_au missing")?;
    assert!((distance_au - expected_distance_au).abs() < 1e-9);
    let light_time_days = json["light_time_days"]
        .as_f64()
        .ok_or("light_time_days missing")?;
    assert!((light_time_days - expected_light_time).abs() < 1e-9);

    let rates = place.rates.ok_or("expected rates")?;
    let lon_speed = json["lon_speed_deg_per_day"]
        .as_f64()
        .ok_or("lon_speed missing")?;
    let lat_speed = json["lat_speed_deg_per_day"]
        .as_f64()
        .ok_or("lat_speed missing")?;
    assert!((lon_speed - rates.lon_rad_per_day * RAD2DEG).abs() < 1e-9);
    assert!((lat_speed - rates.lat_rad_per_day * RAD2DEG).abs() < 1e-9);
    // ra_deg/dec_deg must be absent for the ecliptic frame.
    assert!(json.get("ra_deg").is_none());
    assert!(json.get("dec_deg").is_none());
    Ok(())
}

#[test]
fn pos_equatorial_frame_emits_ra_dec_not_lon_lat() -> Result<(), Box<dyn std::error::Error>> {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!(
            "note: skipping (DE440 fixture not present at {}); run scripts/fetch_de440.sh",
            de_path.display()
        );
        return Ok(());
    }
    let de_path_str = de_path.to_string_lossy().into_owned();
    let (ok, stdout, stderr) = run(&[
        "pos",
        "mars",
        "2000-01-01T12:00:00",
        "--scale",
        "tt",
        "--frame",
        "equatorial",
        "--de",
        &de_path_str,
        "--json",
    ]);
    assert!(ok, "stderr: {stderr}");
    let json: serde_json::Value = serde_json::from_str(stdout.trim())?;
    assert!(json.get("ra_deg").is_some());
    assert!(json.get("dec_deg").is_some());
    assert!(json.get("lon_deg").is_none());
    assert!(json.get("lat_deg").is_none());
    Ok(())
}

// ---------------------------------------------------------------------
// `houses` / `chart` / `pos --all` (Wave B additions)
// ---------------------------------------------------------------------

/// Recomputes the `houses` chain through the libraries, bypassing the
/// CLI: UTC -> TT (leap seconds), GAST, LAST, true obliquity, cusps.
fn expected_houses(
    year: i32,
    month: u8,
    day: u8,
    hours: f64,
    east_lon_deg: f64,
    lat_deg: f64,
    system: oxiephemeris_astro::houses::HouseSystem,
) -> Result<(f64, [f64; 12]), Box<dyn std::error::Error>> {
    let jd_utc = julday(Calendar::Gregorian, year, month, day, hours)?;
    let jd_tt = oxiephemeris_core::time::leap::utc_to_tt(jd_utc)?;
    let t_tt = ((jd_tt.hi - 2_451_545.0) + jd_tt.lo) / 36_525.0;
    let gast = oxiephemeris_bodies::sidereal::gast_iau2006(jd_utc, t_tt);
    let theta = oxiephemeris_astro::angles::local_sidereal_time(gast, east_lon_deg.to_radians());
    let nut = oxiephemeris_bodies::frames::nutation_iau2000a(t_tt);
    let eps = oxiephemeris_bodies::frames::mean_obliquity_iau2006(t_tt) + nut.deps_rad;
    let cusps = oxiephemeris_astro::houses::cusps(system, theta, lat_deg.to_radians(), eps)
        .map_err(|e| format!("cusps failed: {e}"))?;
    let mut cusps_deg = [0.0_f64; 12];
    for (slot, rad) in cusps_deg.iter_mut().zip(cusps) {
        *slot = rad * RAD2DEG;
    }
    Ok((jd_tt.value(), cusps_deg))
}

/// Pre-1972 epochs (before the leap-second table) must work for
/// `houses`/`chart` via the documented ΔT-model fallback — natal charts
/// routinely need them. The TT chain: `TT = UT1 + ΔT(1965) ≈ + 35.9 s`.
#[test]
fn houses_pre_1972_epoch_works_via_delta_t_fallback() -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&[
        "houses",
        "--system",
        "equal",
        "--lat",
        "35.6895",
        "--lon",
        "139.6917",
        "--json",
        "1965-03-15T10:30:00",
    ]);
    assert!(ok, "pre-1972 houses failed: {stderr}");
    let json: serde_json::Value = serde_json::from_str(stdout.trim())?;
    // TT − UTC = ΔT(1965.2) ≈ 35.9 s (Espenak–Meeus); check the epoch
    // chain really used the ΔT fallback, not garbage.
    let jd_utc = julday(Calendar::Gregorian, 1965, 3, 15, 10.5)?;
    let jd_tt = json["jd_tt"].as_f64().ok_or("jd_tt missing")?;
    let delta_t_s = (jd_tt - jd_utc.value()) * 86_400.0;
    assert!(
        (30.0..45.0).contains(&delta_t_s),
        "TT-UT at 1965 should be ~36 s, got {delta_t_s}"
    );
    let cusps = json["cusps_deg"].as_array().ok_or("cusps_deg missing")?;
    assert_eq!(cusps.len(), 12);
    Ok(())
}

#[test]
fn houses_json_matches_direct_astro_computation() -> Result<(), Box<dyn std::error::Error>> {
    let (ok, stdout, stderr) = run(&[
        "houses",
        "--system",
        "placidus",
        "--lat",
        "43.06",
        "--lon",
        "141.35",
        "--json",
        "1999-03-30T20:15:00",
    ]);
    assert!(ok, "stderr: {stderr}");
    let json: serde_json::Value = serde_json::from_str(stdout.trim())?;
    assert_eq!(json["system"], "placidus");

    let (expected_jd_tt, expected_cusps) = expected_houses(
        1999,
        3,
        30,
        20.25,
        141.35,
        43.06,
        oxiephemeris_astro::houses::HouseSystem::Placidus,
    )?;
    let jd_tt = json["jd_tt"].as_f64().ok_or("jd_tt missing")?;
    assert!((jd_tt - expected_jd_tt).abs() < 1e-9, "jd_tt {jd_tt}");
    let cusps = json["cusps_deg"].as_array().ok_or("cusps_deg missing")?;
    assert_eq!(cusps.len(), 12);
    for (i, (got, want)) in cusps.iter().zip(expected_cusps).enumerate() {
        let got = got.as_f64().ok_or("cusp not a number")?;
        assert!(
            (got - want).abs() < 1e-9,
            "cusp {}: cli={got} lib={want}",
            i + 1
        );
    }
    // Quadrant-system invariants surface in the JSON too.
    let asc = json["ascendant_deg"].as_f64().ok_or("ascendant missing")?;
    let mc = json["mc_deg"].as_f64().ok_or("mc missing")?;
    assert!((asc - expected_cusps[0]).abs() < 1e-9);
    assert!((mc - expected_cusps[9]).abs() < 1e-9);
    Ok(())
}

#[test]
fn houses_sidereal_shifts_longitudes_but_not_sidereal_time(
) -> Result<(), Box<dyn std::error::Error>> {
    let base = [
        "houses",
        "--system",
        "equal",
        "--lat",
        "35.6581",
        "--lon",
        "139.7414",
        "--json",
        "1999-03-30T20:15:00",
    ];
    let (ok, stdout, stderr) = run(&base);
    assert!(ok, "stderr: {stderr}");
    let tropical: serde_json::Value = serde_json::from_str(stdout.trim())?;

    let mut sidereal_args = base.to_vec();
    sidereal_args.extend_from_slice(&["--sidereal", "lahiri"]);
    let (ok, stdout, stderr) = run(&sidereal_args);
    assert!(ok, "stderr: {stderr}");
    let sidereal: serde_json::Value = serde_json::from_str(stdout.trim())?;

    assert_eq!(sidereal["sidereal"], "lahiri");
    let ayanamsha = sidereal["ayanamsha_deg"]
        .as_f64()
        .ok_or("ayanamsha_deg missing")?;
    // Lahiri is ~23.85 deg around 1999 (published 23 deg 51').
    assert!((23.7..24.0).contains(&ayanamsha), "ayanamsha={ayanamsha}");

    // The longitude shift is the *true-equinox* ayanamsha (mean ayanamsha
    // + nutation-in-longitude dpsi), not the mean ayanamsha alone: the
    // cusps/angles `houses` reports are true-equinox-of-date quantities
    // (see `oxiephemeris-cli`'s `astro_epoch::sidereal_offset_true_equinox_rad`
    // and the identical convention in `oxiephemeris-compat`'s
    // `houses_ex`/`sidereal_flag_shifts_cusps_by_the_ayanamsha` test).
    // Compute dpsi independently (not via the CLI) to keep this a real
    // check rather than a tautology.
    let jd_tt = sidereal["jd_tt"].as_f64().ok_or("jd_tt missing")?;
    let t_tt = (jd_tt - 2_451_545.0) / 36_525.0;
    let dpsi_deg = oxiephemeris_bodies::frames::nutation_iau2000a(t_tt).dpsi_rad * RAD2DEG;
    let expected_shift = ayanamsha + dpsi_deg;

    let trop_cusp1 = tropical["cusps_deg"][0].as_f64().ok_or("cusp1")?;
    let sid_cusp1 = sidereal["cusps_deg"][0].as_f64().ok_or("cusp1")?;
    let shift = (trop_cusp1 - sid_cusp1).rem_euclid(360.0);
    assert!(
        (shift - expected_shift).abs() < 1e-9,
        "shift={shift} expected(ayanamsha+dpsi)={expected_shift} (ayanamsha={ayanamsha} \
         dpsi={dpsi_deg})"
    );

    // Sidereal *time* quantities are unaffected by the zodiac choice.
    assert_eq!(tropical["gast_deg"], sidereal["gast_deg"]);
    assert_eq!(tropical["last_deg"], sidereal["last_deg"]);
    // The tropical run must not carry the sidereal keys at all.
    assert!(tropical.get("sidereal").is_none());
    assert!(tropical.get("ayanamsha_deg").is_none());
    Ok(())
}

/// Cross-checks `oxieph houses --sidereal` against
/// `oxiephemeris-compat`'s `Context::houses_ex(SEFLG_SIDEREAL)` at the
/// same epoch/site/ayanamsha -- concrete proof that the CLI's
/// true-equinox sidereal-offset fix (`astro_epoch::
/// sidereal_offset_true_equinox_rad`) now agrees with `compat`'s
/// independently-implemented `houses_ex` shift
/// (`ayanamsha_rad(..) + nut.dpsi_rad`, `oxiephemeris-compat/src/
/// houses.rs`), not just a re-check of the CLI's own formula against
/// itself.
///
/// Uses a **pre-1972** epoch deliberately: both `oxieph` (via its
/// documented ΔT-model fallback, `astro_epoch::resolve_chart_epoch`'s
/// module docs) and `oxiephemeris-compat` (which always uses the
/// Espenak-Meeus ΔT model, `oxiephemeris-compat::context`'s module
/// docs) then compute `TT` from the *identical* polynomial, so any
/// residual is attributable to the sidereal-offset formula itself, not
/// to two different time chains happening to agree.
///
/// Uses `--system placidus` (a quadrant system) rather than `equal`:
/// for `equal`, cusp 10 is `Ascendant + 270 deg` by definition (not the
/// independently-computed MC — see
/// `oxiephemeris_astro::houses::cusps`'s `HouseSystem::Equal` arm), so
/// the CLI's `mc_deg` (`== cusps_deg[9]`) and `compat`'s `mc()` (always
/// the independent MC) legitimately disagree there by a couple of
/// degrees -- a pre-existing, sidereal-unrelated quirk of the `equal`
/// system, not a bug this task is about. Quadrant systems don't have
/// that ambiguity: cusp 10 is the MC by construction in both stacks
/// (same underlying `oxiephemeris_astro::houses::cusps` call).
#[test]
fn houses_sidereal_matches_compat_houses_ex() -> Result<(), Box<dyn std::error::Error>> {
    let lat_deg = 35.6895;
    let lon_deg = 139.6917;
    let (ok, stdout, stderr) = run(&[
        "houses",
        "--system",
        "placidus",
        "--lat",
        "35.6895",
        "--lon",
        "139.6917",
        "--sidereal",
        "lahiri",
        "--json",
        "1965-03-15T10:30:00",
    ]);
    assert!(ok, "stderr: {stderr}");
    let cli_json: serde_json::Value = serde_json::from_str(stdout.trim())?;
    let cli_asc_deg = cli_json["ascendant_deg"].as_f64().ok_or("ascendant_deg")?;
    let cli_mc_deg = cli_json["mc_deg"].as_f64().ok_or("mc_deg")?;
    let cli_cusp1_deg = cli_json["cusps_deg"][0].as_f64().ok_or("cusp1")?;

    // Same epoch, computed the same way `expected_houses` above does
    // (bypassing the CLI), fed as a raw UT `f64` to `compat::houses_ex`.
    let jd_ut = julday(Calendar::Gregorian, 1965, 3, 15, 10.5)?.value();
    let mut ctx = oxiephemeris_compat::Context::new();
    ctx.set_sid_mode(oxiephemeris_compat::SiderealMode::Lahiri);
    let compat_houses = ctx
        .houses_ex(
            jd_ut,
            oxiephemeris_compat::SEFLG_SIDEREAL,
            lat_deg,
            lon_deg,
            'P',
        )
        .map_err(|e| format!("compat houses_ex failed: {e}"))?;

    let asc_diff = (cli_asc_deg - compat_houses.asc()).abs();
    let mc_diff = (cli_mc_deg - compat_houses.mc()).abs();
    let cusp1_diff = (cli_cusp1_deg - compat_houses.cusps[0]).abs();
    assert!(
        asc_diff < 1e-6,
        "ascendant: cli={cli_asc_deg} compat={} diff={asc_diff} deg",
        compat_houses.asc()
    );
    assert!(
        mc_diff < 1e-6,
        "mc: cli={cli_mc_deg} compat={} diff={mc_diff} deg",
        compat_houses.mc()
    );
    assert!(
        cusp1_diff < 1e-6,
        "cusp1: cli={cli_cusp1_deg} compat={} diff={cusp1_diff} deg",
        compat_houses.cusps[0]
    );
    // cusp1 must equal the Ascendant on both sides (quadrant-system
    // invariant), sanity-checking this isn't a vacuous pass.
    assert!((cli_asc_deg - cli_cusp1_deg).abs() < 1e-12);
    Ok(())
}

#[test]
fn houses_polar_placidus_error_is_friendly() {
    let (ok, _stdout, stderr) = run(&[
        "houses",
        "--system",
        "placidus",
        "--lat",
        "78.0",
        "--lon",
        "15.0",
        "2020-06-21T12:00:00",
    ]);
    assert!(!ok, "polar Placidus must fail");
    assert!(
        stderr.contains("undefined") && stderr.contains("whole-sign"),
        "unhelpful error: {stderr}"
    );
}

#[test]
fn houses_out_of_range_longitude_is_rejected() {
    let (ok, _stdout, stderr) = run(&[
        "houses",
        "--system",
        "placidus",
        "--lat",
        "35.68",
        "--lon",
        "999",
        "2026-07-05T12:00:00Z",
    ]);
    assert!(!ok, "out-of-range --lon must fail");
    assert!(
        stderr.contains("--lon") && stderr.contains("-180") && stderr.contains("180"),
        "unhelpful error: {stderr}"
    );
}

#[test]
fn houses_boundary_longitudes_are_accepted() {
    for lon in ["-180", "180", "0"] {
        let (ok, _stdout, stderr) = run(&[
            "houses",
            "--system",
            "placidus",
            "--lat",
            "35.68",
            "--lon",
            lon,
            "2026-07-05T12:00:00Z",
        ]);
        assert!(ok, "boundary lon {lon} should be accepted: {stderr}");
    }
}

#[test]
fn chart_out_of_range_longitude_is_rejected() {
    // The longitude check fires inside `resolve_chart_epoch`, the very
    // first statement of `chart::run`, before any DE-file access — so
    // this must fail the same way regardless of whether a DE440 fixture
    // is present in this checkout.
    let (ok, _stdout, stderr) = run(&[
        "chart",
        "--lat",
        "35.68",
        "--lon",
        "-500",
        "2026-07-05T12:00:00Z",
    ]);
    assert!(!ok, "out-of-range --lon must fail for chart too");
    assert!(
        stderr.contains("--lon") && stderr.contains("-180") && stderr.contains("180"),
        "unhelpful error: {stderr}"
    );
}

#[test]
fn pos_all_matches_single_body_runs() -> Result<(), Box<dyn std::error::Error>> {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!("note: skipping (DE440 fixture not present)");
        return Ok(());
    }
    let de_path_str = de_path.to_string_lossy().into_owned();
    let epoch = "2000-01-01T12:00:00";

    let (ok, stdout, stderr) = run(&[
        "pos",
        "--all",
        epoch,
        "--scale",
        "tt",
        "--de",
        &de_path_str,
        "--json",
    ]);
    assert!(ok, "stderr: {stderr}");
    let all: serde_json::Value = serde_json::from_str(stdout.trim())?;
    let bodies = all["bodies"].as_array().ok_or("bodies missing")?;
    assert_eq!(bodies.len(), 10);
    let names: Vec<&str> = bodies
        .iter()
        .map(|b| b["body"].as_str().unwrap_or("?"))
        .collect();
    assert_eq!(
        names,
        [
            "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
            "Pluto"
        ]
    );

    // The Sun entry must be identical to a single-body run.
    let (ok, stdout, stderr) = run(&[
        "pos",
        "sun",
        epoch,
        "--scale",
        "tt",
        "--de",
        &de_path_str,
        "--json",
    ]);
    assert!(ok, "stderr: {stderr}");
    let single: serde_json::Value = serde_json::from_str(stdout.trim())?;
    for key in ["lon_deg", "lat_deg", "distance_au", "light_time_days"] {
        let a = bodies[0][key].as_f64().ok_or("missing in --all")?;
        let s = single[key].as_f64().ok_or("missing in single")?;
        assert!((a - s).abs() < 1e-12, "{key}: all={a} single={s}");
    }
    assert_eq!(all["jd_tt"], single["jd_tt"]);
    assert_eq!(all["frame"], single["frame"]);
    Ok(())
}

/// The full aspect-name vocabulary `chart --json` may emit.
const KNOWN_ASPECTS: [&str; 11] = [
    "conjunction",
    "semisextile",
    "semisquare",
    "sextile",
    "quintile",
    "square",
    "trine",
    "sesquiquadrate",
    "biquintile",
    "quincunx",
    "opposition",
];

#[test]
fn chart_json_is_internally_consistent() -> Result<(), Box<dyn std::error::Error>> {
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!("note: skipping (DE440 fixture not present)");
        return Ok(());
    }
    let de_path_str = de_path.to_string_lossy().into_owned();
    let (ok, stdout, stderr) = run(&[
        "chart",
        "1999-03-30T20:15:00",
        "--lat",
        "43.06",
        "--lon",
        "141.35",
        "--de",
        &de_path_str,
        "--json",
    ]);
    assert!(ok, "stderr: {stderr}");
    let chart: serde_json::Value = serde_json::from_str(stdout.trim())?;

    for key in ["jd_tt", "bodies", "houses", "angles", "nodes", "aspects"] {
        assert!(chart.get(key).is_some(), "missing key {key}");
    }
    let bodies = chart["bodies"].as_array().ok_or("bodies")?;
    assert_eq!(bodies.len(), 10);
    for body in bodies {
        let lon = body["lon_deg"].as_f64().ok_or("lon_deg")?;
        assert!((0.0..360.0).contains(&lon), "lon out of range: {lon}");
        assert!(body["distance_au"].as_f64().ok_or("distance_au")? > 0.0);
    }

    // Houses/angles must agree with a direct `houses` run.
    let cusp1 = chart["houses"]["cusps_deg"][0].as_f64().ok_or("cusp1")?;
    let asc = chart["angles"]["ascendant_deg"].as_f64().ok_or("asc")?;
    assert!((cusp1 - asc).abs() < 1e-12);
    let (_, expected_cusps) = expected_houses(
        1999,
        3,
        30,
        20.25,
        141.35,
        43.06,
        oxiephemeris_astro::houses::HouseSystem::Placidus,
    )?;
    assert!((cusp1 - expected_cusps[0]).abs() < 1e-9);

    // The true node oscillates within +-1.7 deg of the mean node.
    let mean_node = chart["nodes"]["mean_node_deg"].as_f64().ok_or("mean")?;
    let true_node = chart["nodes"]["true_node_deg"].as_f64().ok_or("true")?;
    let node_diff = (true_node - mean_node + 180.0).rem_euclid(360.0) - 180.0;
    assert!(node_diff.abs() < 1.8, "true-mean node = {node_diff} deg");

    // Aspect entries carry a known aspect name and an in-orb offset.
    let aspects = chart["aspects"].as_array().ok_or("aspects")?;
    assert!(!aspects.is_empty(), "1999-03-31 chart has known aspects");
    for hit in aspects {
        let name = hit["aspect"].as_str().ok_or("aspect name")?;
        assert!(KNOWN_ASPECTS.contains(&name), "unknown aspect {name}");
        let offset = hit["offset_deg"].as_f64().ok_or("offset")?;
        assert!(
            offset.abs() <= 10.0,
            "offset beyond any default orb: {offset}"
        );
        assert!(hit["applying"].is_boolean());
    }
    Ok(())
}

/// Regression test for the "chart bodies never sidereal-shifted" bug
/// (the module doc's `--sidereal ... shifts every other longitude this
/// command reports (bodies, house cusps, angles, nodes)` claim was, pre
/// -fix, false for `bodies`: they were pushed into `BodyJson` straight
/// from the apparent-place pipeline with no ayanamsha subtraction at
/// all). Confirms each body's longitude now shifts by exactly the
/// true-equinox ayanamsha (mean ayanamsha + `Δψ`, matching `houses`'
/// convention), while the aspect table -- built from the *unshifted*
/// tropical longitudes/speeds by design (module doc's "Aspects"
/// section) -- stays byte-for-byte identical between the tropical and
/// sidereal runs.
#[test]
fn chart_sidereal_shifts_body_longitudes_but_not_aspects() -> Result<(), Box<dyn std::error::Error>>
{
    let de_path = de440_path();
    if !de_path.exists() {
        eprintln!("note: skipping (DE440 fixture not present)");
        return Ok(());
    }
    let de_path_str = de_path.to_string_lossy().into_owned();
    let base = [
        "chart",
        "1999-03-30T20:15:00",
        "--lat",
        "43.06",
        "--lon",
        "141.35",
        "--de",
        &de_path_str,
        "--json",
    ];
    let (ok, stdout, stderr) = run(&base);
    assert!(ok, "stderr: {stderr}");
    let tropical: serde_json::Value = serde_json::from_str(stdout.trim())?;

    let mut sidereal_args = base.to_vec();
    sidereal_args.extend_from_slice(&["--sidereal", "lahiri"]);
    let (ok, stdout, stderr) = run(&sidereal_args);
    assert!(ok, "stderr: {stderr}");
    let sidereal: serde_json::Value = serde_json::from_str(stdout.trim())?;

    let ayanamsha = sidereal["ayanamsha_deg"].as_f64().ok_or("ayanamsha_deg")?;
    let jd_tt = sidereal["jd_tt"].as_f64().ok_or("jd_tt")?;
    let t_tt = (jd_tt - 2_451_545.0) / 36_525.0;
    let dpsi_deg = oxiephemeris_bodies::frames::nutation_iau2000a(t_tt).dpsi_rad * RAD2DEG;
    let expected_shift = ayanamsha + dpsi_deg;

    // Independently recomputes the true-equinox ayanamsha's drift rate
    // (central difference, matching `astro_epoch::
    // sidereal_offset_true_equinox_rate_rad_per_day`'s own formula) so
    // the speed-correction check below is a real numeric comparison,
    // not a loose magnitude bound. This is *not* simply "general
    // precession / century": at this specific epoch the instantaneous
    // nutation rate partially cancels the precession rate (see
    // `astro_epoch::tests::rate_is_close_to_mean_precession_rate`'s
    // doc), so the combined rate can be much smaller than the ~0.14
    // arcsec/day precession-only figure.
    let rate_step_days = 0.01_f64;
    let dt_c = rate_step_days / 36_525.0;
    let offset_at = |t: f64| {
        oxiephemeris_astro::ayanamsha::ayanamsha_rad(
            oxiephemeris_astro::ayanamsha::Ayanamsha::Lahiri,
            t,
        ) + oxiephemeris_bodies::frames::nutation_iau2000a(t).dpsi_rad
    };
    let plus = offset_at(t_tt + dt_c);
    let minus = offset_at(t_tt - dt_c);
    let expected_rate_deg_per_day =
        oxiephemeris_core::angle::normalize_pm_pi(plus - minus) / (2.0 * rate_step_days) * RAD2DEG;

    let trop_bodies = tropical["bodies"].as_array().ok_or("bodies")?;
    let sid_bodies = sidereal["bodies"].as_array().ok_or("bodies")?;
    assert_eq!(trop_bodies.len(), sid_bodies.len());
    for (trop, sid) in trop_bodies.iter().zip(sid_bodies) {
        assert_eq!(trop["body"], sid["body"]);
        let trop_lon = trop["lon_deg"].as_f64().ok_or("lon_deg")?;
        let sid_lon = sid["lon_deg"].as_f64().ok_or("lon_deg")?;
        let shift = (trop_lon - sid_lon).rem_euclid(360.0);
        assert!(
            (shift - expected_shift).abs() < 1e-9,
            "{}: shift={shift} expected(ayanamsha+dpsi)={expected_shift}",
            trop["body"]
        );
        let trop_speed = trop["lon_speed_deg_per_day"]
            .as_f64()
            .ok_or("lon_speed_deg_per_day")?;
        let sid_speed = sid["lon_speed_deg_per_day"]
            .as_f64()
            .ok_or("lon_speed_deg_per_day")?;
        assert!(
            (trop_speed - sid_speed - expected_rate_deg_per_day).abs() < 1e-9,
            "{}: speed trop={trop_speed} sid={sid_speed} expected_rate={expected_rate_deg_per_day}",
            trop["body"]
        );
    }

    // Aspects intentionally use unshifted tropical longitudes/speeds
    // (module doc's "Aspects" section: the ayanamsha cancels in the
    // pairwise difference) -- the fix for the bodies bug above must not
    // have touched this.
    assert_eq!(tropical["aspects"], sidereal["aspects"]);
    Ok(())
}
