//! Apparent places of catalog stars: space motion, parallax, solar
//! gravitational deflection, annual (+ optional diurnal) aberration, and
//! the frame rotations of the planetary pipeline.
//!
//! # Method (clean-room sources)
//!
//! Kaplan et al. (1989), AJ 97, 1197, §IV ("The computation for stars
//! parallels that for planets…"), and USNO Circular 179 (Kaplan 2005),
//! §7.2: the catalog position is turned into a barycentric position
//! *vector* at the catalog epoch (the inverse-parallax distance along
//! the catalog direction), a space-motion *velocity* vector is built
//! from the proper-motion components, the parallax and the radial
//! velocity (ESA 1997, *The Hipparcos and Tycho Catalogues*, vol. 1,
//! §1.5.5, eq. 1.5.69 — the standard rectangular space-motion
//! construction), and the epoch-of-date barycentric vector
//!
//! ```text
//! R(t) = R(t₀) + (t − t₀) · Ṙ
//! ```
//!
//! is then observed exactly like a solar-system body: the observer's
//! barycentric position is subtracted (annual parallax), the direction
//! is deflected by the Sun and aberrated by the observer's velocity
//! (the same [`crate::apparent`] routines, Kaplan et al. eqs. 10–17),
//! and rotated into the requested output frame. Light-time to the star
//! is not iterated: the radial-velocity term of the space motion *is*
//! the first-order light-time (perspective) effect, and higher orders
//! are far below a microarcsecond for real stars.
//!
//! # Parallax floor
//!
//! A zero catalog parallax would put the star at infinite distance and
//! make the position vector undefined. Following the same practice as
//! the USNO reductions (Circular 179 §7.2 note), parallaxes below
//! [`MIN_PARALLAX_MAS`] are replaced by it — the star is treated as
//! "at" ≈ 206 kAU × 10⁶; the direction error committed for a true-zero
//! parallax star is below 10 µas of annual parallax.

use libm::{cos, sin};
use oxiephemeris_core::angle::{normalize_0_two_pi, DEG2RAD, MAS2R};
use oxiephemeris_core::time::{tdb_minus_tt_seconds, JulianDate, J2000_JD};
use oxiephemeris_de::{Body, DeFile};

use crate::apparent::{
    aberrate, c_au_per_day, deflect_by_sun, frame_rotation, state_au, unit, BodiesError,
    BodyPosition, Center, Options, SphericalRates, DAYS_PER_CENTURY,
};
use crate::math::{add, norm, scale, sub, Vec3};
use crate::topocentric::{station_gcrs_state_m, TopocentricObserver};

/// Parallax floor, milliarcseconds (see the module docs).
pub const MIN_PARALLAX_MAS: f64 = 1e-3;

/// Days per Julian year (IAU): proper motions are per Julian year.
const DAYS_PER_JULIAN_YEAR: f64 = 365.25;

/// Seconds per day, for the radial-velocity unit conversion.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// The Hipparcos catalog epoch J1991.25 as a TT Julian date
/// (`J2000.0 − 8.75 · 365.25 d`), for convenience when building
/// [`CatalogStar`] values from Hipparcos data.
pub const HIPPARCOS_EPOCH_JD_TT: f64 = 2_448_349.062_5;

/// A catalog star: ICRS position at a catalog epoch plus its space
/// motion — the five (plus radial velocity) astrometric parameters of a
/// modern catalog, in catalog units.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogStar {
    /// ICRS right ascension at [`Self::epoch_jd_tt`], degrees.
    pub ra_deg: f64,
    /// ICRS declination at [`Self::epoch_jd_tt`], degrees.
    pub dec_deg: f64,
    /// Proper motion in right ascension, `μ_α* = μ_α cos δ`,
    /// milliarcseconds per Julian year.
    pub pm_ra_cosdec_mas_yr: f64,
    /// Proper motion in declination, milliarcseconds per Julian year.
    pub pm_dec_mas_yr: f64,
    /// Trigonometric parallax, milliarcseconds (floored at
    /// [`MIN_PARALLAX_MAS`]).
    pub parallax_mas: f64,
    /// Radial velocity, km/s, positive receding. Enters only through
    /// the space-motion (perspective) term; `0.0` is a fine default for
    /// stars without a measured value (the effect is
    /// ≲ 1 mas/century² except for extreme nearby high-µ stars).
    pub rv_km_s: f64,
    /// Catalog epoch as a TT Julian date (Hipparcos:
    /// [`HIPPARCOS_EPOCH_JD_TT`]).
    pub epoch_jd_tt: f64,
}

impl CatalogStar {
    /// Builds a [`CatalogStar`] from its astrometric parameters (this
    /// struct is `#[non_exhaustive]`, so struct-literal syntax is
    /// unavailable outside this crate); an epoch/angle field defaulting
    /// to `0.0` would look like a real position rather than "unset", so
    /// this crate provides a constructor instead of `Default`.
    #[must_use]
    #[allow(clippy::too_many_arguments)] // mirrors the struct's own fields 1:1
    pub const fn new(
        ra_deg: f64,
        dec_deg: f64,
        pm_ra_cosdec_mas_yr: f64,
        pm_dec_mas_yr: f64,
        parallax_mas: f64,
        rv_km_s: f64,
        epoch_jd_tt: f64,
    ) -> Self {
        Self {
            ra_deg,
            dec_deg,
            pm_ra_cosdec_mas_yr,
            pm_dec_mas_yr,
            parallax_mas,
            rv_km_s,
            epoch_jd_tt,
        }
    }

    /// Barycentric position (AU) and space-motion velocity (AU/day) at
    /// the catalog epoch, ICRS axes (ESA 1997, vol. 1, §1.5.5).
    fn barycentric_state_au(&self) -> (Vec3, Vec3) {
        let ra = self.ra_deg * DEG2RAD;
        let dec = self.dec_deg * DEG2RAD;
        let (sin_ra, cos_ra) = (sin(ra), cos(ra));
        let (sin_dec, cos_dec) = (sin(dec), cos(dec));
        let u0: Vec3 = [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec];
        // Local triad: p̂ toward +α, q̂ toward +δ.
        let p_hat: Vec3 = [-sin_ra, cos_ra, 0.0];
        let q_hat: Vec3 = [-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec];

        let plx_rad = self.parallax_mas.max(MIN_PARALLAX_MAS) * MAS2R;
        // Distance in AU: 1 AU subtends the parallax at the star, so
        // d = 1/π with π in radians.
        let dist_au = 1.0 / plx_rad;
        let r0 = scale(u0, dist_au);

        // Tangential motion: μ (rad/yr) at distance d gives μ·d AU/yr;
        // radial motion: km/s → AU/day → per-day directly below.
        let mu_p = self.pm_ra_cosdec_mas_yr * MAS2R / DAYS_PER_JULIAN_YEAR; // rad/day
        let mu_q = self.pm_dec_mas_yr * MAS2R / DAYS_PER_JULIAN_YEAR;
        let tangential = add(scale(p_hat, mu_p * dist_au), scale(q_hat, mu_q * dist_au));
        let v = tangential; // AU/day so far
        (r0, v)
    }

    /// Adds the radial term to the space motion (needs the AU in km
    /// from the ephemeris header for the km/s → AU/day conversion).
    fn velocity_with_radial(&self, v_tangential: Vec3, u0: Vec3, au_km: f64) -> Vec3 {
        let vr_au_day = self.rv_km_s * SECONDS_PER_DAY / au_km;
        add(v_tangential, scale(u0, vr_au_day))
    }
}

/// Apparent place of a catalog star for a geocentric observer.
///
/// The pipeline (module docs): space motion to the epoch of date,
/// observer-barycentric parallax, solar deflection
/// ([`Options::deflection`]), relativistic annual aberration
/// ([`Options::aberration`]), frame rotation ([`Options::frame`], with
/// [`Options::nutation`]). [`Options::with_speed`] adds daily rates by
/// the same central differences and step sizes as the planetary
/// pipeline (±0.01 d geocentric, ±0.001 d topocentric — the diurnal
/// aberration is a 1-day-period signal the coarser step would
/// under-resolve).
///
/// The returned [`BodyPosition::r_au`] is the star's barycentric
/// distance (inverse catalog parallax — enormous for far stars) and
/// [`BodyPosition::light_time_days`] the corresponding straight-line
/// light time; both are catalog-derived bookkeeping, not measurements.
///
/// # Errors
///
/// [`BodiesError::ObserverNotGeocentric`] unless
/// `opts.center == Center::Geocentric` (a heliocentric/barycentric
/// "apparent star" mixes conventions; compute from the catalog vector
/// directly if that is really wanted); [`BodiesError::De`] if the
/// ephemeris cannot supply the Earth/Sun states at the epoch.
pub fn apparent_star(
    de: &DeFile<'_>,
    star: &CatalogStar,
    jd_tt: JulianDate,
    opts: Options,
) -> Result<BodyPosition, BodiesError> {
    star_impl(de, star, jd_tt, opts, None)
}

/// Topocentric apparent place of a catalog star: as [`apparent_star`],
/// with the observation point moved to the ground station (diurnal
/// parallax is sub-µas for stars, but the diurnal aberration,
/// ≈ 0.32″ · cos φ, is not).
///
/// # Errors
///
/// As [`apparent_star`], plus [`BodiesError::Time`] for epochs before
/// the 1972 leap-second table (station rotation needs UT1).
pub fn apparent_star_topocentric(
    de: &DeFile<'_>,
    star: &CatalogStar,
    jd_tt: JulianDate,
    opts: Options,
    topo: &TopocentricObserver,
) -> Result<BodyPosition, BodiesError> {
    star_impl(de, star, jd_tt, opts, Some(topo))
}

fn star_impl(
    de: &DeFile<'_>,
    star: &CatalogStar,
    jd_tt: JulianDate,
    opts: Options,
    topo: Option<&TopocentricObserver>,
) -> Result<BodyPosition, BodiesError> {
    let mut out = star_once(de, star, jd_tt, opts, topo)?;
    if opts.with_speed {
        let h = if topo.is_some() {
            crate::apparent::TOPO_SPEED_STEP_DAYS
        } else {
            crate::apparent::SPEED_STEP_DAYS
        };
        let plus = star_once(de, star, jd_tt.add_days(h), opts, topo)?;
        let minus = star_once(de, star, jd_tt.add_days(-h), opts, topo)?;
        let wrap = |d: f64| oxiephemeris_core::angle::normalize_pm_pi(d);
        out.rates = Some(SphericalRates {
            lon_rad_per_day: wrap(plus.lon_rad - minus.lon_rad) / (2.0 * h),
            lat_rad_per_day: (plus.lat_rad - minus.lat_rad) / (2.0 * h),
            r_au_per_day: (plus.r_au - minus.r_au) / (2.0 * h),
        });
    }
    Ok(out)
}

fn star_once(
    de: &DeFile<'_>,
    star: &CatalogStar,
    jd_tt: JulianDate,
    opts: Options,
    topo: Option<&TopocentricObserver>,
) -> Result<BodyPosition, BodiesError> {
    if opts.center != Center::Geocentric {
        return Err(BodiesError::ObserverNotGeocentric);
    }
    let au_km = de.au_km();
    let c_au_day = c_au_per_day(de);
    let jd_tdb = jd_tt.add_seconds(tdb_minus_tt_seconds(jd_tt));

    // Space motion to the epoch of date (Julian days since the catalog
    // epoch; the TT-vs-TDB distinction is nanoseconds here).
    let (r0, v_tan) = star.barycentric_state_au();
    let u0 = unit(r0);
    let v = star.velocity_with_radial(v_tan, u0, au_km);
    let dt_days = (jd_tt.hi - star.epoch_jd_tt) + jd_tt.lo;
    let r_star = add(r0, scale(v, dt_days));

    // Observer barycentric state (geocenter, optionally + station).
    let (mut obs_pos, mut obs_vel) = state_au(de, Body::Earth, jd_tdb, au_km)?;
    if let Some(t) = topo {
        let (st_pos_m, st_vel_m_s) = station_gcrs_state_m(&t.site, &t.eop, jd_tt)?;
        let m_per_au = au_km * 1_000.0;
        obs_pos = add(obs_pos, scale(st_pos_m, 1.0 / m_per_au));
        obs_vel = add(obs_vel, scale(st_vel_m_s, SECONDS_PER_DAY / m_per_au));
    }

    // Annual (+ diurnal) parallax: the observer-to-star vector.
    let p_vec = sub(r_star, obs_pos);
    let dist_au = norm(p_vec);
    let mut p_hat = unit(p_vec);

    if opts.deflection {
        p_hat = deflect_by_sun(de, jd_tdb, au_km, c_au_day, obs_pos, p_vec, p_hat)?;
    }
    if opts.aberration {
        p_hat = aberrate(p_hat, scale(obs_vel, 1.0 / c_au_day));
    }

    let t_centuries = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_CENTURY;
    let rot = frame_rotation(opts.frame, t_centuries, opts.nutation);
    let p_out = rot.apply(p_hat);

    let lon_rad = normalize_0_two_pi(libm::atan2(p_out[1], p_out[0]));
    let lat_rad = libm::atan2(
        p_out[2],
        libm::sqrt(p_out[0] * p_out[0] + p_out[1] * p_out[1]),
    );
    Ok(BodyPosition {
        jd_tt,
        position_au: scale(p_out, dist_au),
        lon_rad,
        lat_rad,
        r_au: dist_au,
        light_time_days: dist_au / c_au_day,
        rates: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{CatalogStar, HIPPARCOS_EPOCH_JD_TT, MIN_PARALLAX_MAS};
    use crate::math::{dot, norm};

    fn fixed_star(ra_deg: f64, dec_deg: f64) -> CatalogStar {
        CatalogStar {
            ra_deg,
            dec_deg,
            pm_ra_cosdec_mas_yr: 0.0,
            pm_dec_mas_yr: 0.0,
            parallax_mas: 0.0,
            rv_km_s: 0.0,
            epoch_jd_tt: HIPPARCOS_EPOCH_JD_TT,
        }
    }

    #[test]
    fn zero_parallax_is_floored_not_infinite() {
        let (r0, _) = fixed_star(10.0, 20.0).barycentric_state_au();
        let d = norm(r0);
        assert!(d.is_finite());
        // 1 mas floor ... wait, MIN_PARALLAX_MAS = 1e-3 mas = 1 µas:
        // distance = 206264.8" / 1e-6 " = 2.06e11 AU.
        let expected = 1.0 / (MIN_PARALLAX_MAS * oxiephemeris_core::angle::MAS2R);
        assert!((d - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn space_motion_direction_matches_triad() {
        // A star on the equator at RA 0: +μα* moves toward +y, +μδ
        // toward +z.
        let mut star = fixed_star(0.0, 0.0);
        star.pm_ra_cosdec_mas_yr = 1000.0;
        star.parallax_mas = 100.0;
        let (r0, v) = star.barycentric_state_au();
        assert!(v[1] > 0.0 && v[0].abs() < 1e-18 && v[2].abs() < 1e-18);
        // Tangential: v ⟂ r.
        assert!(dot(r0, v).abs() / (norm(r0) * norm(v)) < 1e-15);
    }
}
