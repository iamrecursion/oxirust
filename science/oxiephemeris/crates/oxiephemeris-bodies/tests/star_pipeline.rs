//! Physics tests for the apparent-star pipeline: each classical effect
//! is isolated and checked against its textbook magnitude, plus a real
//! catalog end-to-end case (Polaris).
//!
//! The DE440 binary (`data/de440/linux_p1550p2650.440`) is required for
//! the Earth/Sun states; every test skips cleanly when it is absent.

use std::path::PathBuf;

use oxiephemeris_bodies::star::{apparent_star, CatalogStar, HIPPARCOS_EPOCH_JD_TT};
use oxiephemeris_bodies::{Center, Frame, NutationModel, Options};
use oxiephemeris_core::angle::{AS2R, DEG2RAD, MAS2R, RAD2DEG};
use oxiephemeris_core::time::JulianDate;
use oxiephemeris_de::DeFile;

/// ICRS position of the north ecliptic pole: RA 18h, Dec (90° − ε₀)
/// with ε₀ = 84381.406″ (IAU 2006).
const NEP_RA_DEG: f64 = 270.0;
const NEP_DEC_DEG: f64 = 90.0 - 84_381.406 / 3600.0;

/// The constant of aberration, arcsec (κ = 20.49552″; e.g. USNO
/// Circular 179 §3.2's v/c scale for the Earth's orbital speed).
const ABERRATION_CONSTANT_AS: f64 = 20.495_52;

fn de440() -> Option<Vec<u8>> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/de440/linux_p1550p2650.440");
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return None;
    }
    match std::fs::read(&path) {
        Ok(b) => Some(b),
        Err(e) => panic!("read failed: {e}"),
    }
}

fn parse(bytes: &[u8]) -> DeFile<'_> {
    match DeFile::parse(bytes) {
        Ok(d) => d,
        Err(e) => panic!("parse failed: {e}"),
    }
}

fn fixed_star(ra_deg: f64, dec_deg: f64) -> CatalogStar {
    CatalogStar::new(ra_deg, dec_deg, 0.0, 0.0, 0.0, 0.0, HIPPARCOS_EPOCH_JD_TT)
}

fn opts(frame: Frame, aberration: bool, deflection: bool) -> Options {
    Options::new(
        Center::Geocentric,
        frame,
        aberration,
        deflection,
        true,
        false,
        NutationModel::Iau2000a,
    )
}

fn place(de: &DeFile<'_>, star: &CatalogStar, jd: f64, o: Options) -> (f64, f64) {
    match apparent_star(de, star, JulianDate::from_f64(jd), o) {
        Ok(p) => (p.lon_rad, p.lat_rad),
        Err(e) => panic!("apparent_star failed: {e}"),
    }
}

/// Angular separation of two (lon, lat) directions, radians.
fn separation(a: (f64, f64), b: (f64, f64)) -> f64 {
    let ca = [a.1.cos() * a.0.cos(), a.1.cos() * a.0.sin(), a.1.sin()];
    let cb = [b.1.cos() * b.0.cos(), b.1.cos() * b.0.sin(), b.1.sin()];
    let d = ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2];
    d.clamp(-1.0, 1.0).acos()
}

#[test]
fn identity_without_effects_returns_the_catalog_direction() {
    let Some(bytes) = de440() else { return };
    let de = parse(&bytes);
    let star = fixed_star(30.0, 40.0);
    let (lon, lat) = place(&de, &star, 2_460_000.5, opts(Frame::Icrs, false, false));
    // The only residual is the annual parallax of the floored-parallax
    // distance (~1 AU / 2e11 AU ≈ 5e-12 rad ≈ 1 µas).
    assert!((lon - 30.0 * DEG2RAD).abs() < 2e-11, "lon residual {lon}");
    assert!((lat - 40.0 * DEG2RAD).abs() < 2e-11, "lat residual {lat}");
}

#[test]
fn annual_aberration_at_the_ecliptic_pole_is_the_aberration_constant() {
    let Some(bytes) = de440() else { return };
    let de = parse(&bytes);
    let star = fixed_star(NEP_RA_DEG, NEP_DEC_DEG);
    // At the ecliptic pole the Earth's orbital velocity is always
    // (nearly) perpendicular to the line of sight, so the aberrational
    // displacement is κ·(1 ± e) with e = 0.0167 the orbital eccentricity
    // (plus ≲ 0.02″ of barycentric-vs-heliocentric velocity difference).
    for season_jd in [2_460_000.5, 2_460_091.5, 2_460_183.5, 2_460_274.5] {
        let with = place(&de, &star, season_jd, opts(Frame::Icrs, true, false));
        let without = place(&de, &star, season_jd, opts(Frame::Icrs, false, false));
        let sep_as = separation(with, without) / AS2R;
        let (lo, hi) = (
            ABERRATION_CONSTANT_AS * (1.0 - 0.0167) - 0.03,
            ABERRATION_CONSTANT_AS * (1.0 + 0.0167) + 0.03,
        );
        assert!(
            (lo..hi).contains(&sep_as),
            "aberration at jd {season_jd}: {sep_as}\" not in [{lo}, {hi}]"
        );
    }
}

#[test]
fn annual_parallax_displacement_matches_the_catalog_parallax() {
    let Some(bytes) = de440() else { return };
    let de = parse(&bytes);
    let mut near = fixed_star(NEP_RA_DEG, NEP_DEC_DEG);
    near.parallax_mas = 500.0;
    let far = fixed_star(NEP_RA_DEG, NEP_DEC_DEG);
    for jd in [2_459_900.5, 2_460_000.5, 2_460_100.5] {
        let p_near = place(&de, &near, jd, opts(Frame::Icrs, false, false));
        let p_far = place(&de, &far, jd, opts(Frame::Icrs, false, false));
        let sep_mas = separation(p_near, p_far) / MAS2R;
        // At the ecliptic pole the parallactic displacement is π times
        // the observer's barycentric distance projected on the ecliptic:
        // within a few percent of 500 mas (orbit eccentricity + the
        // Sun-vs-barycenter offset of up to ~0.01 AU).
        assert!(
            (475.0..525.0).contains(&sep_mas),
            "parallax displacement {sep_mas} mas at jd {jd}"
        );
    }
}

#[test]
fn proper_motion_advances_the_position_linearly() {
    let Some(bytes) = de440() else { return };
    let de = parse(&bytes);
    // Identical stars except for the declination proper motion, so the
    // (equal) parallactic displacement cancels in the difference.
    let mut star = fixed_star(120.0, 0.0);
    star.pm_dec_mas_yr = 1000.0;
    star.parallax_mas = 50.0;
    let mut base_star = fixed_star(120.0, 0.0);
    base_star.parallax_mas = 50.0;
    // 20 Julian years after the catalog epoch: +20″ of declination.
    let jd = HIPPARCOS_EPOCH_JD_TT + 20.0 * 365.25;
    let moved = place(&de, &star, jd, opts(Frame::Icrs, false, false));
    let base = place(&de, &base_star, jd, opts(Frame::Icrs, false, false));
    let ddec_as = (moved.1 - base.1) / AS2R;
    assert!(
        (ddec_as - 20.0).abs() < 0.02,
        "20 yr of 1\"/yr proper motion moved {ddec_as}\""
    );
    let dra_as = (moved.0 - base.0) / AS2R;
    assert!(
        dra_as.abs() < 0.02,
        "RA should be unchanged, moved {dra_as}\""
    );
}

#[test]
fn polaris_apparent_place_2026_is_sane() {
    let Some(bytes) = de440() else { return };
    let de = parse(&bytes);
    // HIP 11767 (Polaris), Hipparcos catalog (ESA 1997, hip_main.dat):
    // ICRS epoch J1991.25, RA 37.94614689°, Dec +89.26413805°,
    // μα* = 44.22 mas/yr, μδ = −11.74 mas/yr, π = 7.56 mas.
    let polaris = CatalogStar::new(
        37.946_146_89,
        89.264_138_05,
        44.22,
        -11.74,
        7.56,
        0.0,
        HIPPARCOS_EPOCH_JD_TT,
    );
    // 2026-07-05 TT, apparent (true equator and equinox of date).
    let jd = 2_461_226.5;
    let (_, lat) = place(&de, &polaris, jd, opts(Frame::TrueOfDate, true, true));
    let pole_distance_deg = 90.0 - lat * RAD2DEG;
    // Polaris' north-polar distance: 0.736° at J2000, shrinking toward
    // its ~2102 minimum of ~0.45°; the apparent value in 2026 sits near
    // 0.66° (±nutation/aberration wobbles of ≲ 0.01°). The band below
    // is wide enough for those wobbles yet tight enough to catch any
    // sign/ordering error in precession, proper motion or frame
    // rotation (each of which shifts the result by ≥ 0.07°).
    assert!(
        (0.60..0.73).contains(&pole_distance_deg),
        "Polaris pole distance {pole_distance_deg}° in 2026"
    );
    // And it must differ visibly from the ICRS (J2000-axes) value.
    let (_, lat_icrs) = place(&de, &polaris, jd, opts(Frame::Icrs, true, true));
    let icrs_pole_distance_deg = 90.0 - lat_icrs * RAD2DEG;
    assert!(
        (icrs_pole_distance_deg - 0.736).abs() < 0.02,
        "ICRS pole distance {icrs_pole_distance_deg}°"
    );
}

#[test]
fn speeds_reflect_the_diurnal_and_annual_rates() {
    let Some(bytes) = de440() else { return };
    let de = parse(&bytes);
    let star = fixed_star(200.0, -30.0);
    let mut o = opts(Frame::TrueOfDate, true, true);
    o.with_speed = true;
    let p = match apparent_star(&de, &star, JulianDate::from_f64(2_460_000.5), o) {
        Ok(p) => p,
        Err(e) => panic!("apparent_star failed: {e}"),
    };
    let Some(rates) = p.rates else {
        panic!("with_speed must produce rates")
    };
    // A fixed star's apparent place moves by aberration (≤ ~0.35″/day
    // rate of change) plus precession (~0.13″/day in RA): well below
    // 3″/day, well above zero.
    let total_as_day =
        ((rates.lon_rad_per_day * p.lat_rad.cos()).powi(2) + rates.lat_rad_per_day.powi(2)).sqrt()
            / AS2R;
    assert!(
        (0.01..3.0).contains(&total_as_day),
        "apparent drift {total_as_day}\"/day"
    );
}
