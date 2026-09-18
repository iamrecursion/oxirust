//! Identity tests for the SE-compat surface: every `Context` entry point
//! must reproduce the underlying `oxiephemeris-bodies`/`-astro` call it
//! documents itself as wrapping, unit-converted per the SE convention
//! (degrees, AU, deg/day).
//!
//! The DE440 binary (`data/de440/linux_p1550p2650.440`, ~100 MB, fetched
//! by `scripts/fetch_de440.sh`) is not committed; every test needing it
//! skips cleanly when it is absent (same pattern as the other crates'
//! DE-dependent tests).
#![allow(clippy::similar_names)] // jd_ut / jd_tt are the domain-standard names

use std::path::PathBuf;

use oxiephemeris_astro::ayanamsha::{ayanamsha_rad, Ayanamsha};
use oxiephemeris_astro::nodes::{mean_node, true_node_of_date};
use oxiephemeris_bodies::apparent::{
    apparent, apparent_topocentric, Center, Frame, Options, Target,
};
use oxiephemeris_bodies::frames::{nutation_iau2000a, precession_bias_matrix};
use oxiephemeris_bodies::topocentric::{Eop, Observer, TopocentricObserver};
use oxiephemeris_compat::flags::{
    SEFLG_EQUATORIAL, SEFLG_HELCTR, SEFLG_ICRS, SEFLG_J2000, SEFLG_NONUT, SEFLG_SIDEREAL,
    SEFLG_SPEED, SEFLG_TOPOCTR, SEFLG_XYZ,
};
use oxiephemeris_compat::{Context, SiderealMode, SE_MEAN_NODE, SE_MOON, SE_SUN, SE_TRUE_NODE};
use oxiephemeris_core::angle::{normalize_0_two_pi, RAD2DEG};
use oxiephemeris_core::time::{JulianDate, J2000_JD};
use oxiephemeris_de::DeFile;

/// A TT test epoch well inside DE440 and after 1972 (2023-02-25).
const JD_TT: f64 = 2_460_000.5;

/// Julian centuries TT since J2000.0 of [`JD_TT`].
fn t_tt() -> f64 {
    (JD_TT - J2000_JD) / 36_525.0
}

fn de440_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440")
}

/// Loads DE440, or `None` (skip) when the file is not fetched.
fn load_de440() -> Option<Vec<u8>> {
    let path = de440_path();
    if !path.exists() {
        eprintln!(
            "skipping: {} not present (run scripts/fetch_de440.sh)",
            path.display()
        );
        return None;
    }
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(e) => panic!("failed to read {}: {e}", path.display()),
    }
}

fn parse(bytes: &[u8]) -> DeFile<'_> {
    match DeFile::parse(bytes) {
        Ok(de) => de,
        Err(e) => panic!("DE440 parse failed: {e}"),
    }
}

fn calc(ctx: &Context<'_>, ipl: i32, iflag: u32) -> [f64; 6] {
    match ctx.calc(JD_TT, ipl, iflag) {
        Ok(xx) => xx,
        Err(e) => panic!("calc({ipl}, {iflag:#x}) failed: {e}"),
    }
}

fn place(de: &DeFile<'_>, target: Target, opts: Options) -> oxiephemeris_bodies::BodyPosition {
    match apparent(de, target, JulianDate::from_f64(JD_TT), opts) {
        Ok(p) => p,
        Err(e) => panic!("apparent failed: {e}"),
    }
}

const DEFAULT_OPTS: Options = Options::new(
    Center::Geocentric,
    Frame::EclipticTrueOfDate,
    true,
    true,
    true,
    false,
    oxiephemeris_bodies::NutationModel::Iau2000a,
);

#[test]
fn default_calc_matches_apparent_ecliptic_true_of_date() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de.clone());
    for (ipl, target) in [
        (SE_SUN, Target::Sun),
        (SE_MOON, Target::Moon),
        (4, Target::Mars),
    ] {
        let xx = calc(&ctx, ipl, SEFLG_SPEED);
        let mut opts = DEFAULT_OPTS;
        opts.with_speed = true;
        let expected = place(&de, target, opts);
        assert!(
            (xx[0] - expected.lon_rad * RAD2DEG).abs() < 1e-12,
            "lon ipl={ipl}"
        );
        assert!(
            (xx[1] - expected.lat_rad * RAD2DEG).abs() < 1e-12,
            "lat ipl={ipl}"
        );
        assert!((xx[2] - expected.r_au).abs() < 1e-14, "dist ipl={ipl}");
        let Some(rates) = expected.rates else {
            panic!("with_speed must produce rates")
        };
        assert!((xx[3] - rates.lon_rad_per_day * RAD2DEG).abs() < 1e-12);
        assert!((xx[4] - rates.lat_rad_per_day * RAD2DEG).abs() < 1e-12);
        assert!((xx[5] - rates.r_au_per_day).abs() < 1e-14);
    }
}

#[test]
fn frame_flags_route_to_the_documented_frames() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de.clone());
    for (iflag, frame) in [
        (SEFLG_EQUATORIAL, Frame::TrueOfDate),
        (SEFLG_NONUT, Frame::EclipticMeanOfDate),
        (SEFLG_EQUATORIAL | SEFLG_NONUT, Frame::MeanOfDate),
        (SEFLG_EQUATORIAL | SEFLG_J2000 | SEFLG_ICRS, Frame::Icrs),
    ] {
        let xx = calc(&ctx, SE_MOON, iflag);
        let mut opts = DEFAULT_OPTS;
        opts.frame = frame;
        let expected = place(&de, Target::Moon, opts);
        assert!(
            (xx[0] - expected.lon_rad * RAD2DEG).abs() < 1e-12,
            "iflag={iflag:#x}"
        );
        assert!((xx[1] - expected.lat_rad * RAD2DEG).abs() < 1e-12);
    }
}

#[test]
fn helctr_flag_matches_heliocentric_center() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de.clone());
    let xx = calc(&ctx, 4, SEFLG_HELCTR);
    let mut opts = DEFAULT_OPTS;
    opts.center = Center::Heliocentric;
    let expected = place(&de, Target::Mars, opts);
    assert!((xx[0] - expected.lon_rad * RAD2DEG).abs() < 1e-12);
    assert!((xx[2] - expected.r_au).abs() < 1e-14);
}

#[test]
fn j2000_flag_is_the_frame_bias_rotation_at_t_zero() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de.clone());
    let xx = calc(&ctx, SE_MOON, SEFLG_J2000 | SEFLG_EQUATORIAL);
    let mut icrs_opts = DEFAULT_OPTS;
    icrs_opts.frame = Frame::Icrs;
    let icrs = place(&de, Target::Moon, icrs_opts);
    let v = precession_bias_matrix(0.0).apply(icrs.position_au);
    let lon = normalize_0_two_pi(v[1].atan2(v[0])) * RAD2DEG;
    let lat = v[2].atan2((v[0] * v[0] + v[1] * v[1]).sqrt()) * RAD2DEG;
    assert!((xx[0] - lon).abs() < 1e-12, "J2000 RA: {} vs {lon}", xx[0]);
    assert!((xx[1] - lat).abs() < 1e-12, "J2000 Dec");
    // ... and it differs from raw ICRS by the ~17 mas frame bias, no more.
    let sep_mas = ((xx[0] - icrs.lon_rad * RAD2DEG).abs()) * 3.6e6;
    assert!(
        sep_mas < 40.0,
        "bias-sized shift expected, got {sep_mas} mas of RA"
    );
}

#[test]
fn icrs_alone_is_of_date_with_the_bias_omitted() {
    // SEFLG_ICRS without SEFLG_J2000 (see its flag docs): the of-date
    // frame with the frame-bias factor stripped — it must differ from
    // the plain of-date output by a bias-sized rotation only, never by
    // the precession span.
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de);
    let of_date = calc(&ctx, SE_MOON, SEFLG_EQUATORIAL);
    let no_bias = calc(&ctx, SE_MOON, SEFLG_EQUATORIAL | SEFLG_ICRS);
    let sep_mas = (no_bias[0] - of_date[0]).abs() * 3.6e6;
    assert!(
        sep_mas > 0.01 && sep_mas < 40.0,
        "bias-sized shift expected, got {sep_mas} mas of RA"
    );
    assert!((no_bias[1] - of_date[1]).abs() * 3.6e6 < 40.0, "Dec");
}

#[test]
fn xyz_flag_returns_the_position_vector() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de.clone());
    let xx = calc(&ctx, SE_SUN, SEFLG_XYZ);
    let mut opts = DEFAULT_OPTS;
    opts.frame = Frame::EclipticTrueOfDate;
    let expected = place(&de, Target::Sun, opts);
    for (i, expected_component) in expected.position_au.iter().enumerate() {
        assert!((xx[i] - expected_component).abs() < 1e-14, "component {i}");
    }
    assert!(xx[3].abs() < 1e-30 && xx[4].abs() < 1e-30 && xx[5].abs() < 1e-30);
}

#[test]
fn topoctr_flag_matches_apparent_topocentric() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let mut ctx = Context::with_ephemeris(de.clone());
    // SE parameter order: lon, lat, alt.
    ctx.set_topo(139.7414, 35.6581, 40.0);
    let xx = calc(&ctx, SE_MOON, SEFLG_TOPOCTR);
    let topo = TopocentricObserver::new(
        Observer {
            latitude_deg: 35.6581,
            longitude_deg: 139.7414,
            height_m: 40.0,
        },
        Eop::default(),
    );
    let expected = match apparent_topocentric(
        &de,
        Target::Moon,
        JulianDate::from_f64(JD_TT),
        DEFAULT_OPTS,
        &topo,
    ) {
        Ok(p) => p,
        Err(e) => panic!("apparent_topocentric failed: {e}"),
    };
    assert!((xx[0] - expected.lon_rad * RAD2DEG).abs() < 1e-12);
    assert!((xx[2] - expected.r_au).abs() < 1e-14);
}

#[test]
fn sidereal_flag_subtracts_the_ayanamsha() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let mut ctx = Context::with_ephemeris(de.clone());
    let tropical = calc(&ctx, SE_SUN, 0);
    // The subtracted offset is the ayanamsha referred to the true
    // equinox of date (mean ayanamsha + Δψ) — the SE-matching convention
    // documented at `context::calc::sidereal_offset_rad`.
    let dpsi = nutation_iau2000a(t_tt()).dpsi_rad;
    // Default mode: Fagan/Bradley (SE-documented default).
    let sid_default = calc(&ctx, SE_SUN, SEFLG_SIDEREAL);
    let fb = (ayanamsha_rad(Ayanamsha::FaganBradley, t_tt()) + dpsi) * RAD2DEG;
    let diff = (tropical[0] - sid_default[0]).rem_euclid(360.0);
    assert!((diff - fb).abs() < 1e-10, "FB shift {diff} vs {fb}");

    ctx.set_sid_mode(SiderealMode::Lahiri);
    let sid_lahiri = calc(&ctx, SE_SUN, SEFLG_SIDEREAL);
    let lahiri = (ayanamsha_rad(Ayanamsha::Lahiri, t_tt()) + dpsi) * RAD2DEG;
    let diff = (tropical[0] - sid_lahiri[0]).rem_euclid(360.0);
    assert!(
        (diff - lahiri).abs() < 1e-10,
        "Lahiri shift {diff} vs {lahiri}"
    );
}

#[test]
fn node_bodies_match_astro_plus_nutation() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de.clone());
    let dpsi_deg = nutation_iau2000a(t_tt()).dpsi_rad * RAD2DEG;

    let xx = calc(&ctx, SE_TRUE_NODE, 0);
    let oscu = match true_node_of_date(&de, JulianDate::from_f64(JD_TT)) {
        Ok(el) => el,
        Err(e) => panic!("true_node_of_date failed: {e}"),
    };
    let expected = (oscu.node_lon_rad * RAD2DEG + dpsi_deg).rem_euclid(360.0);
    assert!(
        (xx[0] - expected).abs() < 1e-10,
        "true node {} vs {expected}",
        xx[0]
    );
    assert!(
        xx[1].abs() < 1e-30 && xx[2].abs() < 1e-30,
        "lat/dist must be 0"
    );

    // NONUT: the raw mean-equinox-of-date longitude.
    let xx_nonut = calc(&ctx, SE_TRUE_NODE, SEFLG_NONUT);
    assert!((xx_nonut[0] - oscu.node_lon_rad * RAD2DEG).abs() < 1e-10);
}

#[test]
fn mean_node_needs_no_ephemeris() {
    let ctx = Context::new();
    let xx = match ctx.calc(JD_TT, SE_MEAN_NODE, SEFLG_NONUT | SEFLG_SPEED) {
        Ok(xx) => xx,
        Err(e) => panic!("mean node without DE must work: {e}"),
    };
    let expected = mean_node(t_tt()) * RAD2DEG;
    assert!((xx[0] - expected).abs() < 1e-10);
    // The node regresses ~0.053 deg/day.
    assert!(
        (-0.06..-0.045).contains(&xx[3]),
        "mean-node speed out of range: {}",
        xx[3]
    );
}

#[test]
fn calc_ut_is_calc_after_the_delta_t_shift() {
    let Some(bytes) = load_de440() else { return };
    let de = parse(&bytes);
    let ctx = Context::with_ephemeris(de);
    let jd_ut = 2_451_545.0; // 2000-01-01 12:00 UT
    let dt_s = oxiephemeris_core::time::delta_t_seconds(2000.0);
    let via_ut = match ctx.calc_ut(jd_ut, SE_SUN, 0) {
        Ok(xx) => xx,
        Err(e) => panic!("calc_ut failed: {e}"),
    };
    let via_tt = match ctx.calc(jd_ut + dt_s / 86_400.0, SE_SUN, 0) {
        Ok(xx) => xx,
        Err(e) => panic!("calc failed: {e}"),
    };
    // decimal_year granularity (day-level) vs the exact 2000.0 makes ΔT
    // differ by well under a millisecond here; the Sun moves ~0.04 mas/ms.
    assert!(
        (via_ut[0] - via_tt[0]).abs() < 1e-7,
        "{} vs {}",
        via_ut[0],
        via_tt[0]
    );
}

#[test]
fn error_paths_are_typed() {
    let ctx = Context::new();
    // Chiron: unsupported body.
    assert!(matches!(
        ctx.calc(JD_TT, 15, 0),
        Err(oxiephemeris_compat::CompatError::UnsupportedBody(15))
    ));
    // Direct body without an ephemeris.
    assert!(matches!(
        ctx.calc(JD_TT, SE_SUN, 0),
        Err(oxiephemeris_compat::CompatError::NoEphemeris)
    ));
    // Topocentric without set_topo.
    assert!(matches!(
        ctx.calc(JD_TT, SE_SUN, SEFLG_TOPOCTR),
        Err(oxiephemeris_compat::CompatError::TopoNotSet)
    ));
    // Sidereal + equatorial conflict.
    assert!(matches!(
        ctx.calc(JD_TT, SE_SUN, SEFLG_SIDEREAL | SEFLG_EQUATORIAL),
        Err(oxiephemeris_compat::CompatError::ConflictingFlags(_))
    ));
    // Nodes reject equatorial output.
    assert!(matches!(
        ctx.calc(JD_TT, SE_MEAN_NODE, SEFLG_EQUATORIAL),
        Err(oxiephemeris_compat::CompatError::ConflictingFlags(_))
    ));
}

#[test]
fn sidtime_matches_gast_through_the_delta_t_chain() {
    let ctx = Context::new();
    let jd_ut = 2_451_545.0;
    let hours = match ctx.sidtime(jd_ut) {
        Ok(h) => h,
        Err(e) => panic!("sidtime failed: {e}"),
    };
    // Independent recomputation through the same documented chain.
    let dt_s = oxiephemeris_core::time::delta_t_seconds(2000.0);
    let jd_tt = JulianDate::from_f64(jd_ut).add_seconds(dt_s);
    let t = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / 36_525.0;
    let gast = oxiephemeris_bodies::sidereal::gast_iau2006(JulianDate::from_f64(jd_ut), t);
    let expected_hours = gast * 12.0 / core::f64::consts::PI;
    // Sub-millisecond agreement (ΔT decimal-year granularity only).
    assert!(
        (hours - expected_hours).abs() * 3600.0 < 1e-3,
        "{hours} vs {expected_hours}"
    );
}
