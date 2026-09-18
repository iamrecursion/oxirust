//! Integration tests for the apparent-place pipeline
//! (`oxiephemeris_bodies::apparent`), verified against a real DE440 file.
//!
//! The DE440 binary (`data/de440/linux_p1550p2650.440`, ≈ 100 MB) is not
//! committed; every test skips gracefully (prints a note, returns `Ok`)
//! when it is absent.
//!
//! Expected magnitudes cited in the individual tests:
//! * light time Sun ≈ 8.3 min = 0.0057–0.0059 d (1 AU / 173.1446 AU/d,
//!   Kaplan et al. 1989 step (o));
//! * annual aberration ≈ 20.5″ (κ = 20.49552″, IAU 1976 aberration
//!   constant);
//! * solar deflection ≈ 4 mas at 90° elongation (Kaplan et al. 1989,
//!   eq. 15 with g1 = 0.004 07″), rising steeply toward conjunction;
//! * precession: general precession in longitude p ≈ 50.29″/yr
//!   (Capitaine et al. 2003, P03), so a near-ecliptic body's direction
//!   changes by ≈ 50.3″/yr between ICRS and true-of-date, while the
//!   celestial pole moves by p·sin(ε) ≈ 20.0″/yr.

use std::error::Error;
use std::path::PathBuf;

use oxiephemeris_bodies::apparent::{apparent, Center, Frame, Options, Target};
use oxiephemeris_bodies::frames::gcrs_to_true_of_date;
use oxiephemeris_bodies::math::{cross, dot, norm, scale, sub, Vec3};
use oxiephemeris_bodies::NutationModel;
use oxiephemeris_core::angle::RAD2DEG;
use oxiephemeris_core::time::{julday, tdb_minus_tt_seconds, Calendar, JulianDate, J2000_JD};
use oxiephemeris_de::{Body, DeFile};

type TestResult = Result<(), Box<dyn Error>>;

/// Radians per arcsecond.
const AS: f64 = std::f64::consts::PI / (180.0 * 3600.0);

/// Loads the DE440 file, or `None` (with a note) when it is not on disk.
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

/// Angle between two vectors in radians, accurate down to ~1e-16 rad.
fn angle_between(a: Vec3, b: Vec3) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

/// Unit vector (test helper; callers pass non-zero vectors).
fn unit(v: Vec3) -> Result<Vec3, Box<dyn Error>> {
    let n = norm(v);
    if n > 0.0 {
        Ok(scale(v, 1.0 / n))
    } else {
        Err("zero vector".into())
    }
}

/// Geocentric ICRS geometric options (no aberration, no deflection;
/// everything else — including the nutation model — pipeline default).
fn geometric_opts() -> Options {
    Options::new(
        Center::Geocentric,
        Frame::Icrs,
        false,
        false,
        true,
        false,
        NutationModel::default(),
    )
}

fn gregorian_jd(year: i32, month: u8, day: u8) -> Result<JulianDate, Box<dyn Error>> {
    julday(Calendar::Gregorian, year, month, day, 0.0)
        .map_err(|e| format!("julday({year}-{month}-{day}): {e}").into())
}

#[test]
fn sun_light_time_and_moon_distance_are_physical() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let epochs = [
        JulianDate::from_f64(J2000_JD),
        gregorian_jd(2010, 1, 1)?,
        gregorian_jd(1990, 6, 15)?,
        gregorian_jd(2024, 10, 7)?,
    ];
    for jd in epochs {
        // Sun: light time ~ 8.3 min; Earth-Sun distance 0.983-1.017 AU
        // => tau in [0.00568, 0.00588] d; test band [0.0056, 0.0060].
        let sun = apparent(&de, Target::Sun, jd, Options::default())?;
        assert!(
            sun.light_time_days > 0.0056 && sun.light_time_days < 0.0060,
            "Sun light time {} d out of band at jd {}",
            sun.light_time_days,
            jd.value()
        );
        // Light time and range must be mutually consistent by construction.
        let c_au_day = oxiephemeris_bodies::apparent::c_au_per_day(&de);
        assert!((sun.light_time_days - sun.r_au / c_au_day).abs() < 1e-15);

        // Moon: geocentric distance between perigee ~356 500 km
        // (0.002383 AU) and apogee ~406 700 km (0.002719 AU).
        let moon = apparent(&de, Target::Moon, jd, Options::default())?;
        assert!(
            moon.r_au > 0.00238 && moon.r_au < 0.00272,
            "Moon distance {} AU out of band at jd {}",
            moon.r_au,
            jd.value()
        );
    }
    Ok(())
}

#[test]
fn annual_aberration_magnitude_is_about_20_arcsec() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    // For the Sun the observer velocity is (nearly) perpendicular to the
    // line of sight at every epoch, so the aberration shift is close to
    // the aberration constant kappa = 20.49552 arcsec, modulated a few
    // percent by the orbital eccentricity. Band: [19, 21.5] arcsec.
    for (y, m, d) in [(2000, 1, 1), (2005, 4, 15), (2013, 7, 1), (1995, 10, 20)] {
        let jd = gregorian_jd(y, m, d)?;
        let geometric = apparent(&de, Target::Sun, jd, geometric_opts())?;
        let mut aberrated_opts = geometric_opts();
        aberrated_opts.aberration = true;
        let aberrated = apparent(&de, Target::Sun, jd, aberrated_opts)?;
        let shift = angle_between(geometric.position_au, aberrated.position_au);
        assert!(
            shift > 19.0 * AS && shift < 21.5 * AS,
            "aberration shift {} arcsec out of band at {y}-{m}-{d}",
            shift / AS
        );
        // Disabled aberration must reproduce the geometric direction.
        let again = apparent(&de, Target::Sun, jd, geometric_opts())?;
        assert!(angle_between(geometric.position_au, again.position_au) < 1e-15);
    }
    Ok(())
}

#[test]
fn solar_deflection_magnitude_and_sun_exemption() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;

    // Scan Saturn's elongation over ~14 months to find quadrature-ish
    // (elongation nearest 90 deg) and conjunction-ish (minimum elongation)
    // geometry. Saturn's synodic period is 378 d, so both occur in-scan.
    let start = gregorian_jd(2005, 1, 1)?;
    let mut best_quad: (f64, JulianDate) = (f64::INFINITY, start);
    let mut best_conj: (f64, JulianDate) = (f64::INFINITY, start);
    for day in 0..430_i32 {
        let jd = start.add_days(f64::from(day));
        let sun = apparent(&de, Target::Sun, jd, geometric_opts())?;
        let sat = apparent(&de, Target::Saturn, jd, geometric_opts())?;
        let elong = angle_between(sun.position_au, sat.position_au);
        let quad_dist = (elong - 90.0_f64.to_radians()).abs();
        if quad_dist < best_quad.0 {
            best_quad = (quad_dist, jd);
        }
        if elong < best_conj.0 {
            best_conj = (elong, jd);
        }
    }

    let deflection_shift = |jd: JulianDate| -> Result<f64, Box<dyn Error>> {
        let off = apparent(&de, Target::Saturn, jd, geometric_opts())?;
        let mut on_opts = geometric_opts();
        on_opts.deflection = true;
        on_opts.light_time = true;
        let on = apparent(&de, Target::Saturn, jd, on_opts)?;
        Ok(angle_between(off.position_au, on.position_au))
    };

    // Quadrature: ~4 mas expected (Kaplan et al. 1989 eq. 15, tan(psi/2)
    // with psi ~ 84 deg for Saturn => ~3.7 mas). Band [2, 8] mas.
    let quad_shift = deflection_shift(best_quad.1)?;
    assert!(
        quad_shift > 2.0e-3 * AS && quad_shift < 8.0e-3 * AS,
        "deflection at quadrature {} mas out of [2, 8] band",
        quad_shift / AS * 1e3
    );

    // Near superior conjunction the deflection rises steeply; > 1 mas is
    // a very conservative floor (already exceeded at quadrature).
    let conj_shift = deflection_shift(best_conj.1)?;
    assert!(
        conj_shift > 1.0e-3 * AS,
        "deflection near conjunction {} mas not > 1 mas",
        conj_shift / AS * 1e3
    );
    assert!(
        conj_shift > quad_shift,
        "deflection must grow toward conjunction"
    );

    // The Sun itself is exempt from deflection: on == off exactly.
    let jd = best_quad.1;
    let sun_off = apparent(&de, Target::Sun, jd, geometric_opts())?;
    let mut sun_on_opts = geometric_opts();
    sun_on_opts.deflection = true;
    sun_on_opts.light_time = true;
    let sun_on = apparent(&de, Target::Sun, jd, sun_on_opts)?;
    assert!(angle_between(sun_off.position_au, sun_on.position_au) < 1e-15);
    Ok(())
}

#[test]
fn icrs_vs_true_of_date_is_consistent_with_precession() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    for (y, m, d) in [(2010, 1, 1), (1990, 1, 1)] {
        let jd = gregorian_jd(y, m, d)?;
        let years = (jd.value() - J2000_JD) / 365.25;
        let t = (jd.value() - J2000_JD) / 36_525.0;

        // (a) A near-ecliptic body direction (the Sun) rotates by the full
        // general precession in longitude, ~50.29 arcsec/yr (P03), because
        // the precession rotation axis (the ecliptic pole) is ~90 deg away.
        // Nutation adds <= ~17 arcsec; band: +/-20 percent.
        let icrs = apparent(&de, Target::Sun, jd, geometric_opts())?;
        let mut tod_opts = geometric_opts();
        tod_opts.frame = Frame::TrueOfDate;
        let tod = apparent(&de, Target::Sun, jd, tod_opts)?;
        let body_shift = angle_between(icrs.position_au, tod.position_au);
        let expected_body = 50.29 * years.abs() * AS;
        assert!(
            body_shift > 0.8 * expected_body && body_shift < 1.2 * expected_body,
            "Sun ICRS vs true-of-date shift {} arcsec, expected ~{} arcsec at {y}",
            body_shift / AS,
            expected_body / AS
        );

        // (b) The celestial pole itself moves by p*sin(eps) ~ 20.0
        // arcsec/yr (the "20 arcsec per year" precession-of-the-pole
        // figure); check the frame rotation directly on the ICRS pole.
        let pole = [0.0, 0.0, 1.0];
        let pole_shift = angle_between(pole, gcrs_to_true_of_date(t).apply(pole));
        let expected_pole = 20.0 * years.abs() * AS;
        assert!(
            pole_shift > 0.8 * expected_pole && pole_shift < 1.2 * expected_pole,
            "pole shift {} arcsec, expected ~{} arcsec at {y}",
            pole_shift / AS,
            expected_pole / AS
        );
    }
    Ok(())
}

#[test]
fn daily_speeds_are_physical() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let mut opts = Options::default();
    opts.frame = Frame::EclipticTrueOfDate;
    opts.with_speed = true;
    for (y, m, d) in [(2000, 1, 1), (2008, 6, 1), (1993, 3, 21)] {
        let jd = gregorian_jd(y, m, d)?;

        // Moon: apparent ecliptic longitude rate 11.8 - 15.4 deg/day.
        let moon = apparent(&de, Target::Moon, jd, opts)?;
        let Some(rates) = moon.rates else {
            return Err("with_speed set but Moon rates missing".into());
        };
        let moon_rate = rates.lon_rad_per_day * RAD2DEG;
        assert!(
            moon_rate > 11.5 && moon_rate < 15.5,
            "Moon lon rate {moon_rate} deg/day out of band at {y}-{m}-{d}"
        );

        // Sun: apparent ecliptic longitude rate 0.953 - 1.020 deg/day.
        let sun = apparent(&de, Target::Sun, jd, opts)?;
        let Some(rates) = sun.rates else {
            return Err("with_speed set but Sun rates missing".into());
        };
        let sun_rate = rates.lon_rad_per_day * RAD2DEG;
        assert!(
            sun_rate > 0.95 && sun_rate < 1.03,
            "Sun lon rate {sun_rate} deg/day out of band at {y}-{m}-{d}"
        );
    }
    Ok(())
}

#[test]
fn geometric_icrs_direction_matches_light_time_reconstruction() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let au_km = de.au_km();
    let clight_km_s = de.constant("CLIGHT").ok_or("DE header lacks CLIGHT")?;
    let c_au_day = clight_km_s * 86_400.0 / au_km;

    for target in [Target::Moon, Target::Venus, Target::Mars, Target::Jupiter] {
        for (y, m, d) in [(2003, 8, 27), (2015, 2, 11)] {
            let jd_tt = gregorian_jd(y, m, d)?;
            let out = apparent(&de, target, jd_tt, geometric_opts())?;

            // Independent reconstruction directly from state_km, iterated
            // to a tighter tolerance (1e-14 d) than the pipeline's 1e-12.
            let jd_tdb = jd_tt.add_seconds(tdb_minus_tt_seconds(jd_tt));
            let earth = de.state_km(Body::Earth, (jd_tdb.hi, jd_tdb.lo))?;
            let earth_pos = [earth[0] / au_km, earth[1] / au_km, earth[2] / au_km];
            let mut tau = 0.0_f64;
            let mut u = [0.0_f64; 3];
            for _ in 0..30 {
                let retarded = jd_tdb.add_days(-tau);
                let b = de.state_km(target.body(), (retarded.hi, retarded.lo))?;
                u = sub([b[0] / au_km, b[1] / au_km, b[2] / au_km], earth_pos);
                let tau_next = norm(u) / c_au_day;
                let converged = (tau_next - tau).abs() < 1e-14;
                tau = tau_next;
                if converged {
                    break;
                }
            }

            let expected = unit(u)?;
            let got = unit(out.position_au)?;
            let diff = angle_between(expected, got);
            assert!(
                diff < 1e-12,
                "{target:?} geometric direction differs by {diff} rad at {y}-{m}-{d}"
            );
            assert!(
                (out.r_au - norm(u)).abs() < 1e-10,
                "{target:?} range differs: {} vs {}",
                out.r_au,
                norm(u)
            );
        }
    }
    Ok(())
}

#[test]
fn two_part_epoch_split_is_honored() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let jd = 2_455_197.5; // 2010-01-01 00:00 TT
    let mut opts = Options::default();
    opts.frame = Frame::EclipticTrueOfDate;
    opts.with_speed = true;
    for target in [Target::Moon, Target::Mercury] {
        // Same instant, three different (hi, lo) splits. The third one is
        // built directly from the public fields, bypassing the two-sum
        // canonicalization of JulianDate::new, so a pipeline that ignored
        // `lo` would be off by 0.25 day (~3.3 deg of Moon longitude).
        let canonical = apparent(&de, target, JulianDate::new(jd, 0.0), opts)?;
        let split = apparent(&de, target, JulianDate::new(jd - 0.25, 0.25), opts)?;
        let raw = apparent(
            &de,
            target,
            JulianDate {
                hi: jd - 0.25,
                lo: 0.25,
            },
            opts,
        )?;
        for other in [&split, &raw] {
            let diff = angle_between(canonical.position_au, other.position_au);
            assert!(
                diff < 1e-12,
                "{target:?} two-part split changes direction by {diff} rad"
            );
            assert!((canonical.r_au - other.r_au).abs() < 1e-12);
        }
    }
    Ok(())
}

#[test]
fn earth_target_from_geocenter_is_an_error() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let jd = JulianDate::from_f64(J2000_JD);

    let geo = apparent(&de, Target::Earth, jd, Options::default());
    assert!(matches!(
        geo,
        Err(oxiephemeris_bodies::BodiesError::TargetIsCenter)
    ));
    let mut helio_opts = Options::default();
    helio_opts.center = Center::Heliocentric;
    let helio_sun = apparent(&de, Target::Sun, jd, helio_opts);
    assert!(matches!(
        helio_sun,
        Err(oxiephemeris_bodies::BodiesError::TargetIsCenter)
    ));

    // Earth is a perfectly good heliocentric target, and its distance is
    // ~1 AU.
    let helio_earth = apparent(&de, Target::Earth, jd, helio_opts)?;
    assert!(helio_earth.r_au > 0.97 && helio_earth.r_au < 1.03);
    Ok(())
}

#[test]
fn heliocentric_and_barycentric_ignore_observer_effects() -> TestResult {
    let Some(bytes) = de440_bytes() else {
        return Ok(());
    };
    let de = DeFile::parse(&bytes)?;
    let jd = gregorian_jd(2012, 5, 5)?;
    for center in [Center::Heliocentric, Center::Barycentric] {
        let mut with_effects_opts = geometric_opts();
        with_effects_opts.center = center;
        with_effects_opts.aberration = true;
        with_effects_opts.deflection = true;
        with_effects_opts.light_time = true;
        let with_effects = apparent(&de, Target::Mars, jd, with_effects_opts)?;
        let mut without_opts = geometric_opts();
        without_opts.center = center;
        let without = apparent(&de, Target::Mars, jd, without_opts)?;
        // Astrometric-style semantics: flags must be no-ops off-geocenter.
        let diff = angle_between(with_effects.position_au, without.position_au);
        assert!(diff < 1e-15, "{center:?} applied observer effects");
    }
    Ok(())
}
