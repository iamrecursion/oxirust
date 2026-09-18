//! Apparent-place pipeline for solar-system bodies: light-time iteration,
//! solar gravitational light deflection, relativistic annual aberration,
//! and rotation into the requested output frame, evaluated on a JPL DE
//! ephemeris.
//!
//! # Clean-room provenance
//!
//! The algorithm follows the proper-place procedure that USNO Circular 179
//! (G. H. Kaplan 2005, "The IAU Resolutions on Astronomical Reference
//! Systems, Time Scales, and Earth Rotation Models", §1.3 and §5.4)
//! summarizes and — for the detailed formulas — defers to the published
//! paper it cites (Circ. 179, §5.4, main text following eq. 5.3):
//!
//! * G. H. Kaplan, J. A. Hughes, P. K. Seidelmann, C. A. Smith &
//!   B. D. Yallop (1989), "Mean and apparent place computations in the new
//!   IAU system. III. Apparent, topocentric, and astrometric places of
//!   planets and stars", AJ 97, 1197 ("Kaplan et al. 1989" below):
//!   light-time eqs. (6)–(9), solar deflection eqs. (10)–(15), annual
//!   aberration eqs. (16)–(17).
//!
//! Order of operations (Circ. 179 §1.3): light-time (BCRS vectors) →
//! gravitational deflection (BCRS) → aberration (BCRS → GCRS) → the
//! equinox-based rotation chain `r_true = N P B r_GCRS` (Circ. 179
//! eq. 5.3), here evaluated with the IAU 2006/2000A-truncated machinery of
//! [`crate::frames`].
//!
//! # Time scales
//!
//! The pipeline is driven by a TT epoch. The DE ephemeris argument is TDB
//! (`T_eph` ≡ TDB for JPL exports, Park et al. 2021, AJ 161, 105 §2), so the
//! epoch is converted once with the Fairhead–Bretagnon series
//! ([`oxiephemeris_core::time::tdb_minus_tt_seconds`]) and kept as a
//! two-part [`JulianDate`] throughout. Frame matrices take Julian
//! centuries TT from J2000.0 (see [`crate::frames`] for why TT vs. TDB is
//! irrelevant there).
//!
//! # Constants
//!
//! The speed of light `CLIGHT` (km/s), the astronomical unit `AU` (km) and
//! the heliocentric gravitational parameter `GMS` (AU³/day²) are read from
//! the DE file header so the pipeline is self-consistent with the
//! ephemeris it evaluates (the DE constants are the values the ephemeris
//! was integrated with). Defined fallback constants are used only when a
//! header lacks them; see [`c_au_per_day`] and [`gm_sun_au3_day2`].

use libm::{fabs, sqrt};
use oxiephemeris_core::angle::{normalize_0_two_pi, normalize_pm_pi};
use oxiephemeris_core::time::{tdb_minus_tt_seconds, JulianDate, J2000_JD, SECONDS_PER_DAY};
use oxiephemeris_core::CoreError;
use oxiephemeris_de::{Body, DeError, DeFile};

use crate::frames::{
    gcrs_to_true_of_date_with, mean_of_date_to_ecliptic, precession_bias_matrix,
    true_of_date_to_ecliptic_with, NutationModel,
};
use crate::math::{add, cartesian_to_spherical, dot, norm, scale, sub, Mat3, Vec3};
use crate::topocentric::{station_gcrs_state_m, TopocentricObserver};

/// Light-time convergence tolerance in days (`|Δτ| < 1e-12 d` ≈ 86 ns).
///
/// Kaplan et al. (1989), §III(e), require 2×10⁻⁸ d for 1 mas on the fastest
/// target (the Moon, 0.5″/s); 10⁻¹² d keeps the light-time contribution to
/// the direction below 10⁻¹² rad. The iteration contracts by a factor of
/// order `v/c` ≈ 10⁻⁴ per round, so 2–4 rounds suffice.
const LIGHT_TIME_TOL_DAYS: f64 = 1e-12;

/// Hard cap on light-time rounds (converges in 2–4; the cap only guards
/// against a hypothetical non-finite state and can never bind physically).
const LIGHT_TIME_MAX_ITER: usize = 20;

/// Central-difference step for daily speeds, in days. See [`apparent`] for
/// the truncation-error estimate. (`pub(crate)`: the star pipeline uses
/// the same steps.)
pub(crate) const SPEED_STEP_DAYS: f64 = 0.01;

/// Central-difference step for **topocentric** daily speeds, in days. The
/// topocentric signal contains the diurnal-parallax oscillation (angular
/// frequency `ω = 2π/day`, amplitude up to the Moon's ≈ 0.95° horizontal
/// parallax), so the central-difference truncation error `(h²/6)·ω³·A`
/// would be ≈ 0.1°/day at `h = 0.05 d` — the step must resolve the
/// diurnal period. At `h = 0.001 d` the truncation error drops to
/// ≈ 4×10⁻⁵ °/day while position round-off (≈ 10⁻¹¹ rad over `2h`)
/// contributes only ≈ 5×10⁻⁹ rad/day.
pub(crate) const TOPO_SPEED_STEP_DAYS: f64 = 0.001;

/// Days per Julian century (IERS TN36 eq. 5.2).
pub(crate) const DAYS_PER_CENTURY: f64 = 36_525.0;

/// Fallback speed of light, km/s (defining constant; IAU 1976 System of
/// Astronomical Constants, same value as SI). Used only if the DE header
/// lacks `CLIGHT`.
const FALLBACK_CLIGHT_KM_S: f64 = 299_792.458;

/// Fallback heliocentric gravitational parameter `k²` in AU³/day², with
/// `k = 0.017 202 098 95` the Gaussian gravitational constant (IAU 1976).
/// Kaplan et al. (1989), step (o), use exactly this value for the
/// deflection factor. Used only if the DE header lacks a sane `GMS`.
const FALLBACK_GMS_AU3_DAY2: f64 = 0.017_202_098_95 * 0.017_202_098_95;

/// Skip threshold for the deflection denominator `g2 = 1 + q̂·ê`
/// (Kaplan et al. 1989, eq. 13). `g2 → 0` only for a body on the far-side
/// extension of the Sun–observer line; `g2 < 1e-8` corresponds to a body
/// within ≈ 0.008° of the anti-solar direction of the Sun's center as seen
/// from the Sun — deep inside the ≈ 0.27°-radius solar disk, hence
/// occulted. Kaplan et al.: "For these bodies or the Sun itself, the
/// deflection can be considered to be zero."
const DEFLECTION_MIN_DENOM: f64 = 1e-8;

/// Errors of the apparent-place pipeline. No function in this module
/// panics; every failure is reported through this type.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodiesError {
    /// The underlying DE evaluation failed (epoch out of range, truncated
    /// file, …).
    De(DeError),
    /// The requested target coincides with the observation center (for
    /// example [`Target::Earth`] with [`Center::Geocentric`], or
    /// [`Target::Sun`] with [`Center::Heliocentric`]): the direction is
    /// undefined. Also returned defensively if the observer–target
    /// separation evaluates to zero.
    TargetIsCenter,
    /// A topocentric observer was supplied together with a
    /// non-geocentric [`Center`]: a ground station is an offset from the
    /// geocenter, so [`apparent_topocentric`] is only defined for
    /// [`Center::Geocentric`].
    ObserverNotGeocentric,
    /// A time-scale conversion for the topocentric observer failed (the
    /// TT → UTC leg needs the leap-second table, which starts at
    /// 1972-01-01 UTC; see [`oxiephemeris_core::time::tt_to_utc`]).
    Time(CoreError),
}

impl core::fmt::Display for BodiesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::De(e) => write!(f, "ephemeris evaluation failed: {e}"),
            Self::TargetIsCenter => f.write_str("target coincides with the observation center"),
            Self::ObserverNotGeocentric => {
                f.write_str("a topocentric observer requires the geocentric center")
            }
            Self::Time(e) => write!(f, "time-scale conversion for the observer failed: {e}"),
        }
    }
}

impl core::error::Error for BodiesError {}

impl From<DeError> for BodiesError {
    fn from(e: DeError) -> Self {
        Self::De(e)
    }
}

impl From<CoreError> for BodiesError {
    fn from(e: CoreError) -> Self {
        Self::Time(e)
    }
}

/// Observable solar-system targets.
///
/// # DE barycenter caveat
///
/// The JPL DE files store, for Mars through Pluto, the states of the
/// **planetary-system barycenters**, not the planet centers (Park et al.
/// 2021, AJ 161, 105, §4; `asc2eph.f` pointer table). [`Target::Mars`]
/// through [`Target::Pluto`] therefore denote those system barycenters.
/// The offset is negligible for Mars, tens–hundreds of km for the giant
/// planets, but ≈ 2000 km for Pluto (Charon), i.e. up to ≈ 0.1″
/// geocentric. Swiss-Ephemeris-style outputs computed from DE files share
/// this convention.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    /// The Sun (center of mass).
    Sun,
    /// The Moon (center of mass; DE geocentric series re-based to the SSB).
    Moon,
    /// Mercury.
    Mercury,
    /// Venus.
    Venus,
    /// The Earth (center of mass). Only meaningful for
    /// [`Center::Heliocentric`] / [`Center::Barycentric`];
    /// [`Center::Geocentric`] returns [`BodiesError::TargetIsCenter`].
    Earth,
    /// Mars **system barycenter** (see the type-level caveat).
    Mars,
    /// Jupiter **system barycenter** (see the type-level caveat).
    Jupiter,
    /// Saturn **system barycenter** (see the type-level caveat).
    Saturn,
    /// Uranus **system barycenter** (see the type-level caveat).
    Uranus,
    /// Neptune **system barycenter** (see the type-level caveat).
    Neptune,
    /// Pluto **system barycenter** (see the type-level caveat).
    Pluto,
}

impl Target {
    /// The DE ephemeris body this target maps to.
    #[must_use]
    pub const fn body(self) -> Body {
        match self {
            Self::Sun => Body::Sun,
            Self::Moon => Body::Moon,
            Self::Mercury => Body::Mercury,
            Self::Venus => Body::Venus,
            Self::Earth => Body::Earth,
            Self::Mars => Body::Mars,
            Self::Jupiter => Body::Jupiter,
            Self::Saturn => Body::Saturn,
            Self::Uranus => Body::Uranus,
            Self::Neptune => Body::Neptune,
            Self::Pluto => Body::Pluto,
        }
    }
}

/// Observation center.
///
/// # SE-like semantics
///
/// For [`Center::Heliocentric`] and [`Center::Barycentric`] the output is
/// the **light-time-corrected geometric** (astrometric-style) position:
/// annual aberration and solar gravitational deflection are observer
/// effects of a geocentric observer and are **not** applied, regardless of
/// [`Options::aberration`] / [`Options::deflection`] (Swiss Ephemeris
/// behaves the same way for its `HELCTR`/`BARYCTR` flags).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Center {
    /// Observer at the Earth's center of mass.
    Geocentric,
    /// Observer at the Sun's center of mass (astrometric-style output).
    Heliocentric,
    /// Observer at the solar-system barycenter (astrometric-style output).
    Barycentric,
}

impl Center {
    /// The DE ephemeris body whose state is the observer state.
    #[must_use]
    pub const fn observer_body(self) -> Body {
        match self {
            Self::Geocentric => Body::Earth,
            Self::Heliocentric => Body::Sun,
            Self::Barycentric => Body::Ssb,
        }
    }
}

/// Output reference frame for the direction (and the position vector).
///
/// The rotations are the equinox-based chain of Circ. 179 eq. (5.3),
/// `r_true = N P B r_GCRS`, built from [`crate::frames`]; `t` is Julian
/// centuries TT from J2000.0 of the requested epoch.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Frame {
    /// GCRS/ICRS axes (no rotation). `lon` is ICRS right ascension.
    Icrs,
    /// Mean equator and equinox of date (frame bias + IAU 2006 precession,
    /// [`precession_bias_matrix`]). `lon` is right ascension from the mean
    /// equinox of date.
    MeanOfDate,
    /// True equator and equinox of date ([`gcrs_to_true_of_date_with`],
    /// nutation series per [`Options::nutation`]).
    /// `lon` is apparent right ascension.
    TrueOfDate,
    /// Ecliptic and **mean** equinox of date
    /// (`R1(eps_A) · PB`, [`mean_of_date_to_ecliptic`]).
    EclipticMeanOfDate,
    /// Ecliptic and **true** equinox of date
    /// (`R1(eps_A + deps) · N · PB`, [`true_of_date_to_ecliptic_with`],
    /// nutation series per [`Options::nutation`]).
    /// Longitude is measured from the **true equinox of date** — this is
    /// the Swiss-Ephemeris-style *apparent longitude* used by the default
    /// SE output.
    EclipticTrueOfDate,
}

/// Pipeline options. `Default` is a geocentric ICRS apparent place:
/// aberration and deflection on, full IAU `2000A_R06` nutation, no
/// speeds.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // independent physical-effect toggles, not a state machine
pub struct Options {
    /// Observation center (see [`Center`] for the astrometric-style
    /// semantics of the non-geocentric centers).
    pub center: Center,
    /// Output frame (see [`Frame`]).
    pub frame: Frame,
    /// Apply relativistic annual aberration (Kaplan et al. 1989,
    /// eqs. 16–17). Only effective for [`Center::Geocentric`].
    pub aberration: bool,
    /// Apply solar gravitational light deflection (Kaplan et al. 1989,
    /// eqs. 10–14). Only effective for [`Center::Geocentric`], and skipped
    /// for [`Target::Sun`].
    pub deflection: bool,
    /// Apply the light-time retardation (Kaplan et al. 1989, eqs. 6–9):
    /// the body is evaluated at the retarded epoch `t − τ`. With `false`
    /// the body is evaluated at the observation epoch itself (`τ = 0`),
    /// giving the *geometric* ("true") position — the Swiss-Ephemeris
    /// `SEFLG_TRUEPOS` convention. Geometric output is normally combined
    /// with [`Options::aberration`]`/`[`Options::deflection`] `= false`
    /// (both are light-path effects), but the pipeline does not enforce
    /// that. Default: `true`.
    pub light_time: bool,
    /// Also compute daily rates of the spherical coordinates by central
    /// finite differences (see [`apparent`]).
    pub with_speed: bool,
    /// Nutation series for the true-of-date output frames
    /// ([`Frame::TrueOfDate`], [`Frame::EclipticTrueOfDate`]); the other
    /// frames involve no nutation. Default: the **full** IAU `2000A_R06`
    /// series; [`NutationModel::Iau2000aTruncated`] is a ≈ 15× cheaper
    /// alternative that agrees with it to < 0.7 mas in `dpsi` / < 0.4 mas
    /// in `deps` over 1995–2050.
    pub nutation: NutationModel,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            center: Center::Geocentric,
            frame: Frame::Icrs,
            aberration: true,
            deflection: true,
            light_time: true,
            with_speed: false,
            nutation: NutationModel::Iau2000a,
        }
    }
}

impl Options {
    /// Builds an [`Options`] from every field explicitly (this struct is
    /// `#[non_exhaustive]`, so struct-literal syntax — including
    /// functional-update syntax — is unavailable outside this crate;
    /// callers there should use this constructor, or [`Options::default`]
    /// followed by direct field assignment for a partial override, since
    /// the fields remain `pub`).
    #[must_use]
    #[allow(clippy::fn_params_excessive_bools)] // mirrors the struct's own fields 1:1
    pub const fn new(
        center: Center,
        frame: Frame,
        aberration: bool,
        deflection: bool,
        light_time: bool,
        with_speed: bool,
        nutation: NutationModel,
    ) -> Self {
        Self {
            center,
            frame,
            aberration,
            deflection,
            light_time,
            with_speed,
            nutation,
        }
    }
}

/// Daily rates of the spherical coordinates (per day), from central finite
/// differences of the full pipeline.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SphericalRates {
    /// Longitude (or right ascension) rate, rad/day, wrap-corrected.
    pub lon_rad_per_day: f64,
    /// Latitude (or declination) rate, rad/day.
    pub lat_rad_per_day: f64,
    /// Distance rate, AU/day.
    pub r_au_per_day: f64,
}

/// Result of the apparent-place pipeline.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyPosition {
    /// The TT epoch the position was evaluated for.
    pub jd_tt: JulianDate,
    /// Position vector in AU, in the requested [`Frame`]: the apparent
    /// unit direction scaled by [`Self::r_au`].
    pub position_au: Vec3,
    /// Longitude in radians, normalized to `[0, 2π)`. For the equatorial
    /// frames ([`Frame::Icrs`], [`Frame::MeanOfDate`],
    /// [`Frame::TrueOfDate`]) this is the right ascension.
    pub lon_rad: f64,
    /// Latitude (declination for equatorial frames) in radians.
    pub lat_rad: f64,
    /// Distance in AU: `|u|` at the converged light time, i.e. the
    /// observer-to-body distance at the retarded epoch. This matches the
    /// JPL Horizons "delta" convention and the "true distance" of the
    /// Astronomical Almanac (Kaplan et al. 1989, step (m); the geometric
    /// distance of their eq. 6 differs by the body's motion during τ).
    pub r_au: f64,
    /// Converged one-way light time τ in days (`= r_au / c`) — except
    /// with [`Options::light_time`] off, where the output is geometric
    /// and this field is `0.0` (no retardation was applied; `r_au` is
    /// then the geometric distance). The Shapiro
    /// gravitational time delay is neglected: it is at most a few tenths
    /// of a millisecond of light time and moves the angular coordinates
    /// only at the sub-microarcsecond level for solar-system geometry
    /// (Kaplan et al. 1989, §III(e), discussion of Murray 1981 / Yallop's
    /// formula).
    pub light_time_days: f64,
    /// Daily rates, present when [`Options::with_speed`] is set.
    pub rates: Option<SphericalRates>,
}

/// Speed of light in AU/day, from the DE header (`CLIGHT` in km/s and the
/// header `AU` in km), so that the light-time geometry uses the constants
/// the ephemeris was built with: `c' = CLIGHT · 86400 / AU`.
///
/// Sanity value: 173.144 633 AU/day (Kaplan et al. 1989, step (o)).
/// Falls back to the private `FALLBACK_CLIGHT_KM_S` if the header lacks
/// `CLIGHT`.
#[must_use]
pub fn c_au_per_day(de: &DeFile<'_>) -> f64 {
    let clight_km_s = match de.constant("CLIGHT") {
        Some(c) if c.is_finite() && c > 0.0 => c,
        _ => FALLBACK_CLIGHT_KM_S,
    };
    clight_km_s * 86_400.0 / de.au_km()
}

/// Heliocentric gravitational parameter GM☉ in AU³/day², from the DE
/// header constant `GMS`.
///
/// DE headers carry the body GMs in AU³/day² (the units of the
/// integration; `asc2eph.f` header GROUP 1041); the value is ≈ k² =
/// 2.959 122×10⁻⁴, which is the magnitude check applied here (accepted
/// range `1e-4..1e-3`). If `GMS` is missing or out of range, falls back
/// to the IAU 1976 `k²` (the private `FALLBACK_GMS_AU3_DAY2`) — the DE440 value
/// differs from `k²` only in the 10th significant digit (Park et al.
/// 2021, Table 3: GM☉ = 132 712 440 041.279 419 km³/s²), which is far
/// below the accuracy that the ≈ 2×10⁻⁸-rad deflection needs.
#[must_use]
pub fn gm_sun_au3_day2(de: &DeFile<'_>) -> f64 {
    match de.constant("GMS") {
        Some(g) if (1e-4..1e-3).contains(&g) => g,
        _ => FALLBACK_GMS_AU3_DAY2,
    }
}

/// Position and velocity of a DE body in AU and AU/day.
pub(crate) fn state_au(
    de: &DeFile<'_>,
    body: Body,
    jd_tdb: JulianDate,
    au_km: f64,
) -> Result<(Vec3, Vec3), DeError> {
    let s = de.state_km(body, (jd_tdb.hi, jd_tdb.lo))?;
    Ok((
        [s[0] / au_km, s[1] / au_km, s[2] / au_km],
        [s[3] / au_km, s[4] / au_km, s[5] / au_km],
    ))
}

/// `v / |v|`. The caller guarantees `|v| > 0`.
pub(crate) fn unit(v: Vec3) -> Vec3 {
    scale(v, 1.0 / norm(v))
}

/// Solar gravitational light deflection, Kaplan et al. (1989), §III(f),
/// eqs. (10)–(14):
///
/// ```text
/// g1 = 2 GM☉ / (c'² |E'|)         (eq. 13; ≈ 2×10⁻⁸ at |E'| = 1 AU)
/// g2 = 1 + q̂ · ê                  (eq. 13)
/// p1 = p̂ + (g1/g2) [ (p̂·q̂) ê − (ê·p̂) q̂ ]   (unit part of eq. 14)
/// ```
///
/// where `p̂` is the observed (geocentric, light-time-corrected) unit
/// direction, `q̂` the heliocentric unit direction of the body at the
/// retarded epoch (`q = u₂ − S(t)`, eq. 11), and `ê` the heliocentric
/// unit direction of the observer (eq. 12). Using the Sun's barycentric
/// position at the observation epoch `t` for both `q` and `ê` is the
/// ≤ 0.1 mas approximation discussed by Kaplan et al. after eq. (11).
/// The maximum limb-grazing deflection is 1.75″; at 90° elongation it is
/// ≈ 4 mas (their eq. 15 with g1 = 0.004 07″).
///
/// Degenerate geometry (`g2 <` [`DEFLECTION_MIN_DENOM`], body occulted by
/// the Sun) is a documented no-op: "the deflection can be considered to
/// be zero" (Kaplan et al. 1989, §III(f)).
pub(crate) fn deflect_by_sun(
    de: &DeFile<'_>,
    jd_tdb: JulianDate,
    au_km: f64,
    c_au_day: f64,
    obs_pos_au: Vec3,
    u_au: Vec3,
    p_hat: Vec3,
) -> Result<Vec3, BodiesError> {
    let (sun_pos, _) = state_au(de, Body::Sun, jd_tdb, au_km)?;
    let e_vec = sub(obs_pos_au, sun_pos);
    let q_vec = sub(add(obs_pos_au, u_au), sun_pos);
    let e_dist = norm(e_vec);
    let q_dist = norm(q_vec);
    if e_dist <= 0.0 || q_dist <= 0.0 {
        // Observer or body at the Sun's center: no deflection defined.
        return Ok(p_hat);
    }
    let e_hat = scale(e_vec, 1.0 / e_dist);
    let q_hat = scale(q_vec, 1.0 / q_dist);
    let grav = 2.0 * gm_sun_au3_day2(de) / (c_au_day * c_au_day * e_dist);
    let denom = 1.0 + dot(q_hat, e_hat);
    if denom < DEFLECTION_MIN_DENOM {
        return Ok(p_hat);
    }
    let coeff = grav / denom;
    let bend = sub(
        scale(e_hat, dot(p_hat, q_hat)),
        scale(q_hat, dot(e_hat, p_hat)),
    );
    Ok(unit(add(p_hat, scale(bend, coeff))))
}

/// Relativistic annual aberration, Kaplan et al. (1989), §III(g),
/// eqs. (16)–(17), after Murray (1981). With `V = Ė(t)/c'`
/// (dimensionless observer barycentric velocity), `β = |V|`,
/// `γ⁻¹ = √(1 − β²)`, and `p̂` the (deflected) unit direction:
///
/// ```text
/// f1 = p̂ · V                                (β cos D of eq. 16)
/// f2 = 1 + f1 / (1 + γ⁻¹)                   (eq. 16, divided by τ)
/// p' = (γ⁻¹ p̂ + f2 V) / (1 + f1)            (eq. 17, divided by |u₄|)
/// ```
///
/// renormalized to a unit vector. The relativistic terms are of order
/// 1 mas; the classical limit is `p̂ + V` (their eq. 18).
pub(crate) fn aberrate(p_hat: Vec3, v_over_c: Vec3) -> Vec3 {
    let beta_sq = dot(v_over_c, v_over_c);
    let gamma_inv = sqrt(1.0 - beta_sq);
    let f1 = dot(p_hat, v_over_c);
    let f2 = 1.0 + f1 / (1.0 + gamma_inv);
    unit(scale(
        add(scale(p_hat, gamma_inv), scale(v_over_c, f2)),
        1.0 / (1.0 + f1),
    ))
}

/// GCRS → requested-frame rotation at `t` Julian centuries TT from
/// J2000.0, with the nutation series (true-of-date branches only)
/// selected by `nutation`. See [`Frame`] for the composition of each
/// branch.
pub(crate) fn frame_rotation(frame: Frame, t: f64, nutation: NutationModel) -> Mat3 {
    match frame {
        Frame::Icrs => Mat3::IDENTITY,
        Frame::MeanOfDate => precession_bias_matrix(t),
        Frame::TrueOfDate => gcrs_to_true_of_date_with(t, nutation),
        Frame::EclipticMeanOfDate => mean_of_date_to_ecliptic(t).mul(&precession_bias_matrix(t)),
        Frame::EclipticTrueOfDate => {
            true_of_date_to_ecliptic_with(t, nutation).mul(&gcrs_to_true_of_date_with(t, nutation))
        }
    }
}

/// One full pipeline evaluation without rates. With `topo` set (only
/// reachable through [`apparent_topocentric`], which enforces the
/// geocentric center), the observation point is the ground station: its
/// GCRS offset is added to the Earth's barycentric state, so the
/// light-time iteration, the deflection geometry and the aberration
/// velocity are all evaluated for the station.
fn place_once(
    de: &DeFile<'_>,
    target: Target,
    jd_tt: JulianDate,
    opts: Options,
    topo: Option<&TopocentricObserver>,
) -> Result<BodyPosition, BodiesError> {
    let observer = opts.center.observer_body();
    let body = target.body();
    if body == observer {
        return Err(BodiesError::TargetIsCenter);
    }

    // (a) TT → TDB once; the DE argument is TDB (Park et al. 2021, §2).
    let jd_tdb = jd_tt.add_seconds(tdb_minus_tt_seconds(jd_tt));
    let au_km = de.au_km();
    let c_au_day = c_au_per_day(de);

    // (b) Observer barycentric state at the observation epoch.
    let (mut obs_pos, mut obs_vel) = state_au(de, observer, jd_tdb, au_km)?;
    if let Some(topo) = topo {
        // Station BCRS state = Earth barycentric state + station GCRS
        // state: GCRS axes are the ICRS axes of the DE vectors by
        // construction (kinematically non-rotating, Circ. 179 §1.2). The
        // GCRS-vs-BCRS relativistic scale difference (∼ 1.5 × 10⁻⁸) is
        // ≤ 10 cm at one Earth radius and is neglected.
        let (r_m, v_m_s) = station_gcrs_state_m(&topo.site, &topo.eop, jd_tt)?;
        let meters_per_au = au_km * 1_000.0;
        obs_pos = add(obs_pos, scale(r_m, 1.0 / meters_per_au));
        obs_vel = add(obs_vel, scale(v_m_s, SECONDS_PER_DAY / meters_per_au));
    }

    // (c) Light-time iteration, Kaplan et al. (1989) eqs. (6)–(9):
    // u = P_body(t − τ) − P_obs(t), τ' = |u| / c', repeated until |Δτ|
    // < LIGHT_TIME_TOL_DAYS. The final τ equals |u|/c' for the last u
    // exactly, so `light_time_days` and `r_au` are mutually consistent.
    // With `Options::light_time` off, a single τ = 0 evaluation gives
    // the geometric position instead.
    let mut tau = 0.0_f64;
    let mut u = [0.0_f64; 3];
    let mut dist = 0.0_f64;
    let max_iter = if opts.light_time {
        LIGHT_TIME_MAX_ITER
    } else {
        1
    };
    for _ in 0..max_iter {
        let retarded = jd_tdb.add_days(-tau);
        let (body_pos, _) = state_au(de, body, retarded, au_km)?;
        u = sub(body_pos, obs_pos);
        dist = norm(u);
        if !opts.light_time {
            break; // geometric: τ stays 0, no retardation
        }
        let tau_next = dist / c_au_day;
        let converged = fabs(tau_next - tau) < LIGHT_TIME_TOL_DAYS;
        tau = tau_next;
        if converged {
            break;
        }
    }
    if dist <= 0.0 || !dist.is_finite() {
        return Err(BodiesError::TargetIsCenter);
    }
    let mut p_hat = scale(u, 1.0 / dist);

    // (d) Solar gravitational deflection — geocentric observer only, and
    // never for the Sun itself (Kaplan et al. 1989, §III(f)).
    if opts.deflection && opts.center == Center::Geocentric && target != Target::Sun {
        p_hat = deflect_by_sun(de, jd_tdb, au_km, c_au_day, obs_pos, u, p_hat)?;
    }

    // (e) Annual aberration — geocentric observer only (heliocentric /
    // barycentric outputs stay astrometric-style, see `Center`).
    if opts.aberration && opts.center == Center::Geocentric {
        p_hat = aberrate(p_hat, scale(obs_vel, 1.0 / c_au_day));
    }

    // (f) Frame rotation, with t in Julian centuries TT from J2000.0.
    let t = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_CENTURY;
    let p_frame = frame_rotation(opts.frame, t, opts.nutation).apply(p_hat);

    // (g) Spherical coordinates, longitude normalized to [0, 2π).
    let (lon_raw, lat_rad, _) = cartesian_to_spherical(p_frame);
    Ok(BodyPosition {
        jd_tt,
        position_au: scale(p_frame, dist),
        lon_rad: normalize_0_two_pi(lon_raw),
        lat_rad,
        r_au: dist,
        light_time_days: tau,
        rates: None,
    })
}

/// Computes the apparent (or astrometric-style, see [`Center`]) place of
/// `target` at the TT epoch `jd_tt`, evaluated on the DE ephemeris `de`.
///
/// Pipeline (Circ. 179 §1.3; Kaplan et al. 1989 §III):
/// TT → TDB, observer barycentric state, light-time iteration, optional
/// solar deflection, optional annual aberration, frame rotation, spherical
/// output. See [`BodyPosition`] for the output conventions.
///
/// # Speeds
///
/// With [`Options::with_speed`], the daily rates of `(lon, lat, r)` are
/// central finite differences of the **full pipeline** at
/// `jd_tt ± 0.01 d` (longitude differences wrap-corrected through
/// [`normalize_pm_pi`]). Swiss Ephemeris likewise outputs daily speeds;
/// analytic differentiation through the light-time/deflection/aberration
/// chain is not warranted here. The truncation error of a central
/// difference is `(h²/6)·max|λ‴|`; for the fastest target (the Moon,
/// dominant longitude oscillation ≈ 0.11 rad at the anomalistic
/// frequency 2π/27.55 d⁻¹) this is ≈ (0.01²/6)·0.11·(0.228)³ ≈ 2×10⁻⁸
/// rad/day ≈ 0.004″/day, and orders of magnitude smaller for the
/// planets; position round-off (≈ 10⁻¹¹ rad over the 0.02 d span)
/// contributes ≈ 5×10⁻¹⁰ rad/day.
///
/// # Errors
///
/// * [`BodiesError::TargetIsCenter`] — `target` coincides with the center
///   (e.g. [`Target::Earth`] from [`Center::Geocentric`]).
/// * [`BodiesError::De`] — the DE file cannot be evaluated at the epoch
///   (out of span, truncated, …); with speeds, the epoch must also be
///   ≥ 0.01 d inside the file span.
pub fn apparent(
    de: &DeFile<'_>,
    target: Target,
    jd_tt: JulianDate,
    opts: Options,
) -> Result<BodyPosition, BodiesError> {
    apparent_impl(de, target, jd_tt, opts, None)
}

/// Computes the topocentric apparent place of `target` for a ground
/// station: identical pipeline to [`apparent`], but the observation point
/// is displaced from the geocenter to the station. The light-time
/// iteration, the solar-deflection geometry and the aberration velocity
/// are all evaluated from the station's barycentric state (Earth
/// barycentric state from the DE file + station GCRS state from
/// [`station_gcrs_state_m`]), so both the topocentric parallax (up to
/// ≈ 1° for the Moon) and the diurnal aberration (≈ 0.32″ · cos φ) come
/// out of the one pipeline rather than as bolted-on corrections.
///
/// # Design note (why a separate entry point, not an `Options` field)
///
/// `Options` is a `Copy + Eq` value type; an observer field would have to
/// be a borrowed `&TopocentricObserver` (adding a lifetime to `Options`)
/// or an owned copy of site + EOP per call. A dedicated entry point adds
/// the capability while keeping geocentric behavior via [`apparent`]
/// bit-identical. `opts.center` must be [`Center::Geocentric`];
/// everything else in [`Options`] keeps its meaning.
///
/// # Nutation-model caveat
///
/// The station's ITRS → GCRS chain ([`station_gcrs_state_m`]) always uses
/// the **default** (full) nutation series inside
/// [`crate::frames::gcrs_to_true_of_date`], independent of
/// [`Options::nutation`], which only selects the series of the *output*
/// frame rotation. A model mismatch there displaces the station by at
/// most ≈ 0.7 mas × R⊕ ≈ 2 cm (≲ 10 µas on the Moon, less for planets) —
/// negligible against every gate in this crate.
///
/// # Speeds caveat
///
/// [`Options::with_speed`] here uses a **finer** central-difference step
/// than [`apparent`]: a topocentric direction carries a ≈ 1-day-period
/// parallax oscillation, so the truncation error `(h²/6)·max|λ‴|` at
/// the geocentric ±0.01 d step would still be ≈ 7×10⁻⁵ rad/day for the
/// Moon. The ±0.001 d topocentric step brings it to
/// ≈ (0.001²/6) · 0.017 · (2π)³ ≈ 7×10⁻⁷ rad/day (≈ 0.15″/day), and
/// ≈ 40× less for a planet at 30″ of parallax.
///
/// # Errors
///
/// All of [`apparent`]'s errors, plus:
///
/// * [`BodiesError::ObserverNotGeocentric`] — `opts.center` is not
///   [`Center::Geocentric`].
/// * [`BodiesError::Time`] — the epoch precedes the 1972 start of the
///   leap-second table (TT → UTC → UT1 is required for Earth rotation).
pub fn apparent_topocentric(
    de: &DeFile<'_>,
    target: Target,
    jd_tt: JulianDate,
    opts: Options,
    topo: &TopocentricObserver,
) -> Result<BodyPosition, BodiesError> {
    if opts.center != Center::Geocentric {
        return Err(BodiesError::ObserverNotGeocentric);
    }
    apparent_impl(de, target, jd_tt, opts, Some(topo))
}

/// Shared implementation of [`apparent`] / [`apparent_topocentric`].
fn apparent_impl(
    de: &DeFile<'_>,
    target: Target,
    jd_tt: JulianDate,
    opts: Options,
    topo: Option<&TopocentricObserver>,
) -> Result<BodyPosition, BodiesError> {
    let mut out = place_once(de, target, jd_tt, opts, topo)?;
    if opts.with_speed {
        let step = if topo.is_some() {
            TOPO_SPEED_STEP_DAYS
        } else {
            SPEED_STEP_DAYS
        };
        let plus = place_once(de, target, jd_tt.add_days(step), opts, topo)?;
        let minus = place_once(de, target, jd_tt.add_days(-step), opts, topo)?;
        let span = 2.0 * step;
        out.rates = Some(SphericalRates {
            lon_rad_per_day: normalize_pm_pi(plus.lon_rad - minus.lon_rad) / span,
            lat_rad_per_day: (plus.lat_rad - minus.lat_rad) / span,
            r_au_per_day: (plus.r_au - minus.r_au) / span,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        aberrate, frame_rotation, unit, BodiesError, Center, Frame, NutationModel, Options, Target,
    };
    use crate::math::{cross, dot, norm, Mat3};
    use libm::atan2;
    use oxiephemeris_de::{Body, DeError};

    /// Angle between two unit vectors, accurate for tiny angles.
    fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
        atan2(norm(cross(a, b)), dot(a, b))
    }

    #[test]
    fn target_body_mapping_matches_de_bodies() {
        assert!(Target::Sun.body() == Body::Sun);
        assert!(Target::Moon.body() == Body::Moon);
        assert!(Target::Mercury.body() == Body::Mercury);
        assert!(Target::Venus.body() == Body::Venus);
        assert!(Target::Earth.body() == Body::Earth);
        assert!(Target::Mars.body() == Body::Mars);
        assert!(Target::Jupiter.body() == Body::Jupiter);
        assert!(Target::Saturn.body() == Body::Saturn);
        assert!(Target::Uranus.body() == Body::Uranus);
        assert!(Target::Neptune.body() == Body::Neptune);
        assert!(Target::Pluto.body() == Body::Pluto);
    }

    #[test]
    fn center_observer_bodies() {
        assert!(Center::Geocentric.observer_body() == Body::Earth);
        assert!(Center::Heliocentric.observer_body() == Body::Sun);
        assert!(Center::Barycentric.observer_body() == Body::Ssb);
    }

    #[test]
    fn default_options_are_geocentric_icrs_apparent() {
        let opts = Options::default();
        assert!(opts.center == Center::Geocentric);
        assert!(opts.frame == Frame::Icrs);
        assert!(opts.aberration && opts.deflection && !opts.with_speed);
        // Wave A integration: the FULL IAU 2000A_R06 series is the
        // pipeline default; the truncated series is the opt-in fast path.
        assert!(opts.nutation == NutationModel::Iau2000a);
        assert!(opts.nutation == NutationModel::default());
    }

    #[test]
    fn aberration_is_identity_for_zero_velocity() {
        let p = unit([0.3, -0.4, 0.5]);
        let q = aberrate(p, [0.0, 0.0, 0.0]);
        assert!(angle_between(p, q) < 1e-15);
    }

    #[test]
    fn aberration_matches_classical_limit_for_perpendicular_velocity() {
        // V perpendicular to p, |V| = 1e-4 (Earth-like): the deflection
        // angle must be atan(|V|) + O(V^2) ~ 1e-4 rad ~ 20.6 arcsec.
        let p = [1.0, 0.0, 0.0];
        let v = [0.0, 1e-4, 0.0];
        let q = aberrate(p, v);
        let angle = angle_between(p, q);
        assert!((angle - 1e-4).abs() < 1e-11);
        // Aberration pushes the apparent direction toward +V.
        assert!(q[1] > 0.0);
        // Output is a unit vector.
        assert!((norm(q) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn icrs_frame_rotation_is_identity() {
        for model in [NutationModel::Iau2000a, NutationModel::Iau2000aTruncated] {
            let m = frame_rotation(Frame::Icrs, 0.123, model);
            assert!(m == Mat3::IDENTITY);
        }
    }

    #[test]
    fn frame_rotations_are_proper_for_all_frames() {
        for frame in [
            Frame::Icrs,
            Frame::MeanOfDate,
            Frame::TrueOfDate,
            Frame::EclipticMeanOfDate,
            Frame::EclipticTrueOfDate,
        ] {
            for model in [NutationModel::Iau2000a, NutationModel::Iau2000aTruncated] {
                let m = frame_rotation(frame, 0.25, model);
                assert!((m.det() - 1.0).abs() < 1e-12);
            }
        }
    }

    /// The nutation selector must actually route: the two models differ
    /// (so the wiring is live) but only at the sub-mas level (< 0.7 mas
    /// ≈ 3.4e-9 in matrix elements; generous 1e-8 ceiling).
    #[test]
    fn nutation_selector_routes_the_true_of_date_rotation() {
        for frame in [Frame::TrueOfDate, Frame::EclipticTrueOfDate] {
            let t = 0.25; // 2025.0
            let full = frame_rotation(frame, t, NutationModel::Iau2000a);
            let trunc = frame_rotation(frame, t, NutationModel::Iau2000aTruncated);
            let mut worst = 0.0_f64;
            for (row_f, row_t) in full.0.iter().zip(trunc.0.iter()) {
                for (a, b) in row_f.iter().zip(row_t.iter()) {
                    worst = worst.max((a - b).abs());
                }
            }
            assert!(worst > 1e-13, "{frame:?}: selector not routed ({worst:e})");
            assert!(worst < 1e-8, "{frame:?}: models too far apart ({worst:e})");
        }
        // Nutation-free frames must be model-independent.
        for frame in [Frame::Icrs, Frame::MeanOfDate, Frame::EclipticMeanOfDate] {
            let full = frame_rotation(frame, 0.25, NutationModel::Iau2000a);
            let trunc = frame_rotation(frame, 0.25, NutationModel::Iau2000aTruncated);
            assert!(full == trunc, "{frame:?} depends on the nutation model");
        }
    }

    #[test]
    fn error_display_is_informative() {
        extern crate alloc;
        use alloc::string::ToString;
        let e = BodiesError::TargetIsCenter;
        assert!(e.to_string().contains("center"));
        let e = BodiesError::from(DeError::EpochOutOfRange);
        assert!(e.to_string().contains("ephemeris"));
        let e = BodiesError::ObserverNotGeocentric;
        assert!(e.to_string().contains("geocentric"));
        let e = BodiesError::from(oxiephemeris_core::CoreError::DateOutOfRange);
        assert!(e.to_string().contains("time-scale"));
    }
}
