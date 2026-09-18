//! Topocentric observers: WGS84 geodetic site → ITRS Cartesian, Earth
//! orientation parameters, the polar-motion matrix `W(t)` of the IERS
//! Conventions, and the station's GCRS state for the apparent-place
//! pipeline (used by [`crate::apparent::apparent_topocentric`]).
//!
//! # Clean-room provenance
//!
//! * IERS Conventions (2010), IERS Technical Note 36 ("TN36"), ch. 5:
//!   the ITRS → GCRS decomposition `[GCRS] = Q(t) R(t) W(t) [ITRS]`
//!   (eq. 5.1), the polar-motion matrix
//!   `W(t) = R3(−s′) · R2(x_p) · R1(y_p)` (eq. 5.3, §5.4.1), the TIO
//!   locator approximation `s′ = −47 µas · t` (eq. 5.13, after Lambert &
//!   Bizouard 2002), the Earth-rotation matrix `R(t) = R3(−ERA)`
//!   (eq. 5.5) with its equinox-based form using GAST (§5.4.3), and the
//!   ERA rate `1.00273781191135448` rev/UT1-day (eq. 5.14).
//! * NIMA TR8350.2 (3rd ed., 2000), "Department of Defense World Geodetic
//!   System 1984": the WGS84 ellipsoid parameters (Table 3.1) and derived
//!   geometric constants (Table 3.3).
//! * IOGP Publication 373-7-2 ("EPSG Guidance Note 7-2", *Coordinate
//!   Conversions and Transformations including Formulas*), EPSG operation
//!   method 9602 "Geographic/geocentric conversions": the closed-form
//!   geodetic → geocentric-Cartesian equations (also Heiskanen & Moritz
//!   1967, *Physical Geodesy*, §5.3). The worked example of that note is
//!   a unit test below.
//!
//! # Frame chain (equinox-based, TN36 eq. 5.1 with §5.4.3)
//!
//! ```text
//! r_TIRS = W(t) · r_ITRS                      (TN36 eq. 5.3)
//! r_true = R3(−GAST) · r_TIRS                 (TN36 eq. 5.5 with GAST, §5.4.3)
//! r_GCRS = (N(t) · PB(t))ᵀ · r_true           (transpose of Circ. 179 eq. 5.3)
//! ```
//!
//! with `N · PB` the crate's [`crate::frames::gcrs_to_true_of_date`] and
//! GAST from [`crate::sidereal::gast_iau2006`]; rotations are orthonormal,
//! so the transpose is the inverse.
//!
//! # Time chain
//!
//! The caller supplies the epoch in TT. Internally: TT → UTC through the
//! embedded leap-second table ([`oxiephemeris_core::time::tt_to_utc`];
//! epochs before 1972 are an error), then `UT1 = UTC + dUT1` with `dUT1`
//! from [`Eop`]. GAST takes the UT1 epoch for the fast term and Julian
//! centuries TT for the precession-nutation terms; the frame matrices take
//! centuries TT (TN36 eq. 5.2).

use libm::{cos, sin, sqrt};
use oxiephemeris_core::angle::{AS2R, DEG2RAD, TWO_PI, UAS2R};
use oxiephemeris_core::time::{tt_to_utc, JulianDate, J2000_JD};

use crate::apparent::BodiesError;
use crate::frames::gcrs_to_true_of_date;
use crate::math::{cross, r1, r2, r3, Mat3, Vec3};
use crate::sidereal::gast_iau2006;

/// WGS84 semi-major axis in meters (NIMA TR8350.2, Table 3.1).
pub const WGS84_A_M: f64 = 6_378_137.0;

/// WGS84 reciprocal flattening `1/f` (NIMA TR8350.2, Table 3.1).
pub const WGS84_INV_F: f64 = 298.257_223_563;

/// WGS84 flattening `f` (derived from [`WGS84_INV_F`]).
pub const WGS84_F: f64 = 1.0 / WGS84_INV_F;

/// WGS84 first eccentricity squared `e² = f(2 − f)` (derived; TR8350.2
/// Table 3.3 lists e² = 6.694 379 990 14 × 10⁻³).
pub const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

/// WGS84 semi-minor axis `b = a(1 − f)` in meters (derived; TR8350.2
/// Table 3.3 lists b = 6 356 752.3142 m).
pub const WGS84_B_M: f64 = WGS84_A_M * (1.0 - WGS84_F);

/// TIO-locator rate: `s′ = −47 µas · t` with `t` in Julian centuries TT
/// (IERS TN36 eq. 5.13, using the mean Chandlerian/annual wobble
/// amplitudes of Lambert & Bizouard 2002).
const S_PRIME_UAS_PER_CENTURY: f64 = -47.0;

/// Earth rotation rate in rad/s: the time-derivative of the ERA of IERS
/// TN36 eq. (5.14), `ERA(Tu) = 2π (0.7790572732640 +
/// 1.00273781191135448 · Tu)` with `Tu` in UT1 days, i.e.
/// `ω = 2π · 1.00273781191135448 / 86400 ≈ 7.292 115 146 706 98 × 10⁻⁵`.
/// Formally rad per UT1-second; the UT1-vs-SI second rate difference
/// (|d(UT1−UTC)/dt| ≲ 4 ms/day, relative 5 × 10⁻⁸) shifts the station
/// velocity by ≲ 2 × 10⁻⁵ m/s — irrelevant to aberration (≪ 1 µas).
/// (Written as `1 + excess` so the full published precision of the rate
/// survives the `f64` literal.)
const EARTH_ROTATION_RAD_PER_S: f64 = TWO_PI * (1.0 + 0.002_737_811_911_354_48) / 86_400.0;

/// Days per Julian century (IERS TN36 eq. 5.2).
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// A ground station in WGS84 geodetic coordinates.
///
/// The ellipsoid is WGS84 (`a` = [`WGS84_A_M`], `1/f` = [`WGS84_INV_F`];
/// NIMA TR8350.2 Table 3.1) — the same ellipsoid JPL Horizons uses for
/// its geodetic observer sites (its response headers print `Center radii:
/// 6378.137, 6378.137, 6356.752 km`), so site definitions interoperate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observer {
    /// Geodetic latitude in degrees, positive north.
    pub latitude_deg: f64,
    /// Geodetic longitude in degrees, positive **east** (IERS/IAU
    /// convention; Circ. 179 eq. 2.15 measures longitude east positive).
    pub longitude_deg: f64,
    /// Height above the WGS84 ellipsoid in meters.
    pub height_m: f64,
}

impl Observer {
    /// Geocentric (ITRS) Cartesian position of the station in meters.
    ///
    /// Standard closed form (EPSG Guidance Note 7-2, operation method
    /// 9602 "Geographic/geocentric conversions"; equivalently Heiskanen &
    /// Moritz 1967, §5.3), with `ν` the prime-vertical radius of
    /// curvature:
    ///
    /// ```text
    /// ν = a / √(1 − e² sin²φ)
    /// x = (ν + h) cos φ cos λ
    /// y = (ν + h) cos φ sin λ
    /// z = (ν (1 − e²) + h) sin φ
    /// ```
    ///
    /// The worked example of the guidance note (φ = 53°48′33.82″ N,
    /// λ = 2°07′46.38″ E, h = 73.0 m → X = 3 771 793.968 m,
    /// Y = 140 253.342 m, Z = 5 124 304.349 m) is reproduced in the unit
    /// tests, together with an independent parametric-latitude form.
    #[must_use]
    pub fn itrs_position_m(&self) -> Vec3 {
        let phi = self.latitude_deg * DEG2RAD;
        let lam = self.longitude_deg * DEG2RAD;
        let (sin_phi, cos_phi) = (sin(phi), cos(phi));
        let (sin_lam, cos_lam) = (sin(lam), cos(lam));
        let nu = WGS84_A_M / sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi);
        [
            (nu + self.height_m) * cos_phi * cos_lam,
            (nu + self.height_m) * cos_phi * sin_lam,
            (nu * (1.0 - WGS84_E2) + self.height_m) * sin_phi,
        ]
    }
}

/// Earth-orientation parameters at the epoch of observation, from IERS
/// data (e.g. the `finals2000A.all` combined series: columns `UT1−UTC`,
/// `PM-x`, `PM-y`).
///
/// # Accuracy impact of `Default` (all zeros)
///
/// Leaving the EOP at zero degrades only the *station-dependent* effects:
///
/// * `dut1_s = 0`: |UT1 − UTC| is kept below 0.9 s by leap seconds, so
///   the station longitude is mis-rotated by up to 0.9 s of Earth
///   rotation ≈ 13.5″ ≈ **420 m** of station displacement (equator).
///   For the Moon (distance ≈ 3.8 × 10⁵ km) that is up to ≈ **0.22″** of
///   topocentric parallax error; for Mars at 0.5 AU, ≈ 1 mas.
/// * `xp/yp = 0`: polar motion is ≲ 0.3″, i.e. the pole wanders ≲ 10 m —
///   ≈ **5 mas** on the Moon, sub-mas on planets.
///
/// The diurnal/semi-diurnal ocean-tide and libration EOP terms that TN36
/// eq. (5.11) and §5.5.3 would add to the IERS tabulations (≲ 0.5 mas in
/// the pole, ≲ 0.05 ms in UT1, i.e. centimeters of station position) are
/// always neglected here.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Eop {
    /// `UT1 − UTC` in seconds (IERS Bulletin A / `finals2000A.all`).
    pub dut1_s: f64,
    /// Polar motion `x_p` in arcseconds.
    pub xp_arcsec: f64,
    /// Polar motion `y_p` in arcseconds.
    pub yp_arcsec: f64,
}

impl Eop {
    /// Builds an [`Eop`] from its three components (this struct is
    /// `#[non_exhaustive]`, so struct-literal syntax is unavailable
    /// outside this crate, including in `const` contexts where
    /// `..Eop::default()` cannot be used since `Default::default` is not
    /// `const`).
    #[must_use]
    pub const fn new(dut1_s: f64, xp_arcsec: f64, yp_arcsec: f64) -> Self {
        Self {
            dut1_s,
            xp_arcsec,
            yp_arcsec,
        }
    }
}

/// A topocentric observation configuration: a geodetic [`Observer`] plus
/// the [`Eop`] at the epoch. Passed to
/// [`crate::apparent::apparent_topocentric`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopocentricObserver {
    /// The ground station.
    pub site: Observer,
    /// Earth-orientation parameters at the epoch ([`Eop::default`] = all
    /// zeros is permitted; see [`Eop`] for the accuracy impact).
    pub eop: Eop,
}

impl TopocentricObserver {
    /// Builds a [`TopocentricObserver`] from a ground station and its
    /// Earth-orientation parameters (this struct is `#[non_exhaustive]`,
    /// so struct-literal syntax is unavailable outside this crate).
    #[must_use]
    pub const fn new(site: Observer, eop: Eop) -> Self {
        Self { site, eop }
    }
}

/// The polar-motion matrix `W(t)` of IERS TN36 eq. (5.3), §5.4.1:
///
/// ```text
/// W(t) = R3(−s′) · R2(x_p) · R1(y_p)
/// ```
///
/// transforming ITRS vectors into the Terrestrial Intermediate Reference
/// System (TIRS), whose z-axis is the CIP: TN36 eq. (5.1) applies `W(t)`
/// first in `[GCRS] = Q(t) R(t) W(t) [ITRS]`. `x_p`, `y_p` are the "polar
/// coordinates" of the CIP in the ITRS and `s′` the TIO locator, whose
/// integral (TN36 eq. 5.4) is approximated by `s′ = −47 µas · t` (TN36
/// eq. 5.13); `t` is in Julian centuries TT since J2000.0 (TN36 eq. 5.2).
///
/// With the TN36 §5.4 rotation convention implemented by
/// [`crate::math::r1`], [`crate::math::r2`] and [`crate::math::r3`],
/// the CIP unit vector in the ITRS is `W(t)ᵀ ẑ ≈ (x_p, −y_p, 1)` to first
/// order (checked in the unit tests).
#[must_use]
pub fn polar_motion_matrix(eop: &Eop, t_centuries_tt: f64) -> Mat3 {
    let s_prime_rad = S_PRIME_UAS_PER_CENTURY * t_centuries_tt * UAS2R;
    r3(-s_prime_rad)
        .mul(&r2(eop.xp_arcsec * AS2R))
        .mul(&r1(eop.yp_arcsec * AS2R))
}

/// GCRS position (meters) and velocity (meters per second) of a ground
/// station at the TT epoch `jd_tt`.
///
/// # Method
///
/// Position: the module-level frame chain — `W(t)` (TN36 eq. 5.3), then
/// `R3(−GAST)` (TN36 eq. 5.5 in its equinox-based form, §5.4.3), then the
/// transpose of the crate's `N · PB` true-of-date chain.
///
/// Velocity: the only non-negligible time dependence is the spin
/// `R3(−GAST)`, so `v_TIRS = ω × r_TIRS` with `ω = ω ẑ`,
/// `ω = 2π · 1.00273781191135448 / 86400` rad/s — the time-derivative of
/// the ERA of TN36 eq. (5.14) (GAST differs from ERA by the slowly varying
/// equation of the origins) — rotated through the same chain. Neglected
/// rates and their aberration impact:
///
/// * `Ṅ`, `ṖB` (precession-nutation, ≈ 50″/yr ≈ 7.7 × 10⁻¹² rad/s):
///   |Δv| ≲ 5 × 10⁻⁵ m/s → ≈ 0.03 µas;
/// * `Ẇ` (polar motion, ∼ mas/day): |Δv| ≲ 4 × 10⁻⁷ m/s → ≪ 0.001 µas;
/// * GAST-vs-ERA rate difference (equation of the origins drift):
///   ≲ 10⁻¹⁴ rad/s, likewise sub-µas.
///
/// Both are therefore "sub-microarcsecond" for any conceivable use of the
/// returned velocity (diurnal aberration is ≈ 0.32″ · cos φ).
///
/// # Time scales
///
/// TT → UTC uses the embedded leap-second table, then
/// `UT1 = UTC + dut1_s`. The frame matrices and GAST polynomial take
/// Julian centuries TT (TN36 eq. 5.2).
///
/// # Errors
///
/// [`BodiesError::Time`] if the epoch precedes the 1972 start of the
/// leap-second table (or is non-finite).
pub fn station_gcrs_state_m(
    site: &Observer,
    eop: &Eop,
    jd_tt: JulianDate,
) -> Result<(Vec3, Vec3), BodiesError> {
    let jd_ut1 = tt_to_utc(jd_tt)?.add_seconds(eop.dut1_s);
    let t = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_CENTURY;

    let r_itrs = site.itrs_position_m();
    let r_tirs = polar_motion_matrix(eop, t).apply(r_itrs);
    // ω × r in the TIRS (ω along the CIP by construction of the TIRS).
    let omega = [0.0, 0.0, EARTH_ROTATION_RAD_PER_S];
    let v_tirs = cross(omega, r_tirs);

    let spin = r3(-gast_iau2006(jd_ut1, t));
    // Rotations are orthonormal: transpose = inverse.
    let true_to_gcrs = gcrs_to_true_of_date(t).transpose();
    let r_gcrs = true_to_gcrs.apply(spin.apply(r_tirs));
    let v_gcrs = true_to_gcrs.apply(spin.apply(v_tirs));
    Ok((r_gcrs, v_gcrs))
}

#[cfg(test)]
mod tests {
    use super::{
        polar_motion_matrix, station_gcrs_state_m, Eop, Observer, EARTH_ROTATION_RAD_PER_S,
        WGS84_A_M, WGS84_B_M, WGS84_E2, WGS84_F,
    };
    use crate::apparent::BodiesError;
    use crate::math::{dot, norm, sub};
    use libm::{atan, cos, fabs, sin, sqrt, tan};
    use oxiephemeris_core::angle::{AS2R, DEG2RAD};
    use oxiephemeris_core::time::JulianDate;

    /// Derived WGS84 constants match NIMA TR8350.2 Table 3.3 to the
    /// precision printed there (b = 6 356 752.3142 m,
    /// e² = 0.006 694 379 990 14).
    #[test]
    fn wgs84_derived_constants_match_tr8350_2_table_3_3() {
        assert!(fabs(WGS84_B_M - 6_356_752.314_2) < 5e-5);
        assert!(fabs(WGS84_E2 - 0.006_694_379_990_14) < 5e-15);
    }

    /// Equatorial sea-level stations sit at one semi-major axis from the
    /// spin axis, in the equatorial plane.
    #[test]
    fn itrs_equator_sea_level() {
        let p = Observer {
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            height_m: 0.0,
        }
        .itrs_position_m();
        assert!(fabs(p[0] - WGS84_A_M) < 1e-9);
        assert!(fabs(p[1]) < 1e-9);
        assert!(fabs(p[2]) < 1e-9);

        let q = Observer {
            latitude_deg: 0.0,
            longitude_deg: 90.0,
            height_m: 0.0,
        }
        .itrs_position_m();
        assert!(fabs(q[0]) < 1e-6); // cos(pi/2) rounding at a ~ 6.4e6 m scale
        assert!(fabs(q[1] - WGS84_A_M) < 1e-9);
    }

    /// Poles sit at one semi-minor axis on the spin axis.
    #[test]
    fn itrs_poles() {
        for (lat, sign) in [(90.0, 1.0), (-90.0, -1.0)] {
            let p = Observer {
                latitude_deg: lat,
                longitude_deg: 139.0,
                height_m: 0.0,
            }
            .itrs_position_m();
            assert!(fabs(p[2] - sign * WGS84_B_M) < 1e-6);
            assert!(sqrt(p[0] * p[0] + p[1] * p[1]) < 1e-6);
        }
    }

    /// Worked example of EPSG Guidance Note 7-2 (IOGP Publication
    /// 373-7-2), operation method 9602 "Geographic/geocentric
    /// conversions", on WGS84: φ = 53°48′33.82″ N, λ = 2°07′46.38″ E,
    /// h = 73.0 m → X = 3 771 793.968 m, Y = 140 253.342 m,
    /// Z = 5 124 304.349 m (printed to the millimeter in the note).
    #[test]
    fn itrs_matches_epsg_guidance_note_worked_example() {
        let p = Observer {
            latitude_deg: 53.0 + 48.0 / 60.0 + 33.82 / 3600.0,
            longitude_deg: 2.0 + 7.0 / 60.0 + 46.38 / 3600.0,
            height_m: 73.0,
        }
        .itrs_position_m();
        assert!(fabs(p[0] - 3_771_793.968) < 5e-4, "x = {}", p[0]);
        assert!(fabs(p[1] - 140_253.342) < 5e-4, "y = {}", p[1]);
        assert!(fabs(p[2] - 5_124_304.349) < 5e-4, "z = {}", p[2]);
    }

    /// Independent closed form via the parametric (reduced) latitude
    /// `tan u = (1 − f) tan φ`: a sea-level point is
    /// `(a cos u cos λ, a cos u sin λ, b sin u)` and the height is added
    /// along the geodetic normal `(cos φ cos λ, cos φ sin λ, sin φ)`
    /// (Heiskanen & Moritz 1967, §5.3). Must agree with the
    /// prime-vertical form to well below a micrometer.
    #[test]
    fn itrs_agrees_with_parametric_latitude_form() {
        for (lat, lon, h) in [
            (35.6581, 139.7414, 40.0),
            (-70.0, -68.3, 2835.0),
            (10.0, 250.0, -30.0),
        ] {
            let p = Observer {
                latitude_deg: lat,
                longitude_deg: lon,
                height_m: h,
            }
            .itrs_position_m();
            let phi = lat * DEG2RAD;
            let lam = lon * DEG2RAD;
            let u = atan((1.0 - WGS84_F) * tan(phi));
            let q = [
                WGS84_A_M * cos(u) * cos(lam) + h * cos(phi) * cos(lam),
                WGS84_A_M * cos(u) * sin(lam) + h * cos(phi) * sin(lam),
                WGS84_B_M * sin(u) + h * sin(phi),
            ];
            assert!(
                norm(sub(p, q)) < 1e-7,
                "lat {lat}: |Δ| = {}",
                norm(sub(p, q))
            );
        }
    }

    /// `W(t)` is a proper rotation and maps the CIP onto the TIRS z-axis:
    /// the CIP unit vector in the ITRS is `W(t)ᵀ ẑ ≈ (x_p, −y_p, 1)`
    /// (TN36 §5.4.1 sign convention).
    #[test]
    fn polar_motion_matrix_convention() {
        let eop = Eop {
            dut1_s: 0.0,
            xp_arcsec: 0.25,
            yp_arcsec: 0.40,
        };
        let w = polar_motion_matrix(&eop, 0.2);
        assert!(fabs(w.det() - 1.0) < 1e-14);
        let cip_in_itrs = w.transpose().apply([0.0, 0.0, 1.0]);
        // Small-angle: second-order terms are ~ (2e-6 rad)^2 ~ 4e-12.
        assert!(fabs(cip_in_itrs[0] - eop.xp_arcsec * AS2R) < 1e-11);
        assert!(fabs(cip_in_itrs[1] + eop.yp_arcsec * AS2R) < 1e-11);
        assert!(cip_in_itrs[2] > 1.0 - 1e-11);
    }

    /// With zero pole coordinates, `W(t) = R3(−s′)` — a pure spin by the
    /// TIO locator `s′ = −47 µas · t` (TN36 eq. 5.13): at `t = 1` the
    /// x-axis moves by exactly that angle and `ẑ` is untouched.
    #[test]
    fn polar_motion_matrix_tio_locator_only() {
        let w = polar_motion_matrix(&Eop::default(), 1.0);
        let z = w.apply([0.0, 0.0, 1.0]);
        assert!(fabs(z[2] - 1.0) < 1e-15 && fabs(z[0]) < 1e-15 && fabs(z[1]) < 1e-15);
        let x = w.apply([1.0, 0.0, 0.0]);
        let s_prime_rad = -47.0e-6 * AS2R;
        // R3(−s′) x̂ = (cos s′, +sin s′, 0).
        assert!(fabs(x[1] - sin(s_prime_rad)) < 1e-18);
    }

    /// Station GCRS state: the rotation chain preserves the geocentric
    /// distance, the velocity is `ω × r` (perpendicular to `r`, magnitude
    /// `ω · ρ_xy`), and the numbers land where a 35.66°-latitude station
    /// should (spin-axis distance 5 188 237.7 m → 378.3 m/s).
    #[test]
    fn station_state_geometry() {
        let site = Observer {
            latitude_deg: 35.6581,
            longitude_deg: 139.7414,
            height_m: 40.0,
        };
        let eop = Eop {
            dut1_s: -0.18,
            xp_arcsec: 0.06,
            yp_arcsec: 0.29,
        };
        let jd_tt = JulianDate::new(2_458_863.5, 0.0);
        let (r, v) = match station_gcrs_state_m(&site, &eop, jd_tt) {
            Ok(state) => state,
            Err(e) => panic!("station state failed: {e}"),
        };
        let r_itrs = site.itrs_position_m();
        assert!(fabs(norm(r) - norm(r_itrs)) < 1e-6);
        // v ⟂ r and |v| = ω · (distance from the spin axis). The spin
        // axis is the CIP, i.e. the TIRS z-axis; project in TIRS terms
        // via the invariant |v| / ω.
        assert!(fabs(dot(r, v)) / (norm(r) * norm(v)) < 1e-12);
        let rho_xy = norm(v) / EARTH_ROTATION_RAD_PER_S;
        assert!(fabs(rho_xy - 5_188_237.7) < 15.0, "rho_xy = {rho_xy}");
        assert!(fabs(norm(v) - 378.33) < 0.01, "|v| = {}", norm(v));
    }

    /// Epochs before the 1972 leap-second table are a typed error.
    #[test]
    fn station_state_pre_1972_is_time_error() {
        let site = Observer {
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            height_m: 0.0,
        };
        let jd_tt = JulianDate::new(2_430_000.5, 0.0); // 1941
        match station_gcrs_state_m(&site, &Eop::default(), jd_tt) {
            Err(BodiesError::Time(_)) => {}
            other => panic!("expected BodiesError::Time, got {other:?}"),
        }
    }
}
