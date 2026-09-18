//! Chart angles: local sidereal time, RAMC, Midheaven (MC), Ascendant,
//! Vertex, and East Point (equatorial ascendant).
//!
//! # Inputs and conventions
//!
//! Every function works in **radians** and returns ecliptic longitudes
//! normalized to `[0, 2π)`:
//!
//! - `theta` — the local apparent sidereal time (LAST), which equals the
//!   right ascension of the midheaven (RAMC). Callers compute it as
//!   GAST + east longitude (USNO Circular 179, Kaplan 2005, eq. 2.15);
//!   use [`oxiephemeris_bodies::sidereal::gast_iau2006`] and the
//!   [`local_sidereal_time`] helper below.
//! - `phi` — the observer's geodetic latitude, positive north.
//! - `eps` — the **true** obliquity of date, `ε = ε_A + Δε` from
//!   [`oxiephemeris_bodies::frames::mean_obliquity_iau2006`] plus the
//!   nutation in obliquity. Required range `0 ≤ eps < π/2`.
//!
//! Non-finite inputs propagate as NaN results (the underlying `atan2`
//! and normalization propagate NaN); no function here panics.
//!
//! # Geometry (the derivation used throughout)
//!
//! In equatorial coordinates of date (x toward the equinox, z toward the
//! north celestial pole), an ecliptic point of longitude `λ` is
//!
//! ```text
//! p(λ) = (cos λ, sin λ cos ε, sin λ sin ε),
//! ```
//!
//! the north ecliptic pole is `N_e = (0, −sin ε, cos ε)`, the zenith of
//! the observer is `Z = (cos φ cos θ, cos φ sin θ, sin φ)`, the east
//! point of the horizon is `E = (−sin θ, cos θ, 0)`, and the north point
//! of the horizon is `N_h = (−sin φ cos θ, −sin φ sin θ, cos φ)`.
//! The intersection of the ecliptic (pole `N_e`) with any great circle of
//! pole `P` is the pair of directions `±(N_e × P)`; expressing the cross
//! product in ecliptic coordinates gives each closed form below, and the
//! sign is fixed by evaluating a non-degenerate special case. The test
//! suite re-derives every angle through explicit rotation matrices
//! ([`oxiephemeris_bodies::math`]) as an independent cross-check.
//!
//! # Clean-room sources
//!
//! - Wikipedia, "Ascendant" (accessed 2026-07-06): the closed form
//!   `λ_Asc = arctan(−cos θ / (sin θ cos ε + tan φ sin ε))` with quadrant
//!   rules, reproduced here in quadrant-correct `atan2` form.
//! - Urania Trust, "The Astronomy of Houses" (accessed 2026-07-06):
//!   definitions of the prime vertical (poles at the north/south points
//!   of the horizon) and of the Vertex as the ecliptic / prime-vertical
//!   intersection in the west.
//! - Kaplan (2005), USNO Circular 179, eq. 2.15: local sidereal time.
//! - R. W. Holden, *The Elements of House Division* (Fowler, 1977):
//!   standard mathematical treatment of the angles (background
//!   reference; all equations here are re-derived from the vector
//!   construction above).
//!
//! No Swiss Ephemeris or SOFA/ERFA source code was consulted.

use libm::{atan2, cos, sin, sqrt};
use oxiephemeris_core::angle::normalize_0_two_pi;

/// Errors of the chart-angle functions.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnglesError {
    /// The ecliptic coincides with the reference great circle (the
    /// horizon for [`ascendant`], the prime vertical for [`vertex`]), so
    /// the intersection direction is undefined. This happens exactly
    /// when the two circle poles are (anti-)parallel: for the horizon at
    /// `|φ| = π/2 − ε` with `θ = ±π/2` (the ecliptic *is* the horizon),
    /// for the prime vertical at `|φ| = ε` with `θ = ±π/2`. Detected
    /// when the sine of the angle between the poles falls below `1e-12`.
    EclipticCoincidesWithReferenceCircle,
}

impl core::fmt::Display for AnglesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EclipticCoincidesWithReferenceCircle => f.write_str(
                "the ecliptic coincides with the reference great circle; \
                 the intersection point is undefined",
            ),
        }
    }
}

impl core::error::Error for AnglesError {}

/// Degeneracy threshold: minimum value of `|N_e × P|` (the sine of the
/// angle between the ecliptic pole and the reference-circle pole) below
/// which the intersection is reported as undefined.
const COINCIDENCE_TOL: f64 = 1e-12;

/// Local apparent sidereal time from Greenwich apparent sidereal time
/// and the observer's **east** longitude, radians in `[0, 2π)`:
/// `LAST = GAST + λ_east` (USNO Circular 179, eq. 2.15).
///
/// The result is the `theta` argument of every other function in this
/// module and of [`crate::houses::cusps`].
#[must_use]
pub fn local_sidereal_time(gast_rad: f64, east_longitude_rad: f64) -> f64 {
    normalize_0_two_pi(gast_rad + east_longitude_rad)
}

/// Right ascension of the midheaven (RAMC), radians in `[0, 2π)`.
///
/// The RAMC *is* the local apparent sidereal time; this function only
/// normalizes it. Provided so call sites can state which quantity they
/// mean.
#[must_use]
pub fn ramc(theta: f64) -> f64 {
    normalize_0_two_pi(theta)
}

/// Midheaven (MC): the ecliptic point on the upper meridian, radians in
/// `[0, 2π)`.
///
/// The upper meridian is the half-plane of right ascension `α = θ`; the
/// ecliptic point with that right ascension satisfies
/// `tan α = tan λ cos ε`, inverted quadrant-correctly as
///
/// ```text
/// λ_MC = atan2(sin θ, cos θ cos ε),
/// ```
///
/// which is continuous in `θ` (for `0 ≤ ε < π/2` the two `atan2`
/// arguments never vanish simultaneously) and reproduces
/// `RA(λ_MC) = θ` exactly (verified in `tests/angles.rs`). The MC is
/// independent of latitude; beyond the polar circles the point can stand
/// below the horizon, but it remains the meridian crossing by
/// convention. For `ε = 0` this degenerates continuously to
/// `λ_MC = θ`.
///
/// # Examples
///
/// ```
/// use oxiephemeris_astro::angles::mc;
/// let eps = 23.4367_f64.to_radians();
/// // RAMC = 0: the equinox culminates, and the MC is 0° Aries.
/// assert!(mc(0.0, eps).abs() < 1e-15);
/// ```
#[must_use]
pub fn mc(theta: f64, eps: f64) -> f64 {
    normalize_0_two_pi(atan2(sin(theta), cos(theta) * cos(eps)))
}

/// Ascendant: the ecliptic point on the eastern horizon, radians in
/// `[0, 2π)`.
///
/// # Derivation
///
/// The horizon great circle has pole `Z` (the zenith). In ecliptic
/// coordinates the intersection direction `N_e × Z` has components
///
/// ```text
/// x_e = −(cos φ sin θ cos ε + sin φ sin ε),   y_e = cos φ cos θ,
/// ```
///
/// so `λ_Asc = atan2(cos φ cos θ, −(cos φ sin θ cos ε + sin φ sin ε))`.
/// This is the quadrant-correct form of the published closed formula
/// `λ_Asc = arctan(−cos θ / (sin θ cos ε + tan φ sin ε))` (Wikipedia,
/// "Ascendant", accessed 2026-07-06); keeping the common factor `cos φ`
/// instead of dividing it out avoids the `tan φ` singularity at the
/// poles. The sign choice picks the *eastern* intersection: the dot
/// product of `N_e × Z` with the east direction `E` is
/// `cos φ cos ε + sin φ sin ε sin θ > 0` whenever `|φ| < π/2 − ε`
/// (below the polar circles).
///
/// # Degenerate and high-latitude behavior
///
/// - `|φ| < π/2 − ε`: the returned point is always east (rising).
/// - polar circle `≤ |φ| <` pole: the formula stays continuous wherever
///   defined, but the returned intersection can lie in the *western*
///   half of the horizon (the well-known polar behavior of the
///   ascendant); no attempt is made to flip it.
/// - `|φ| = π/2 − ε` with `θ = ±π/2`: the ecliptic coincides with the
///   horizon and [`AnglesError::EclipticCoincidesWithReferenceCircle`]
///   is returned.
/// - `|φ| = π/2` (with `ε > 0`): the continuous limit `π` (north pole)
///   or `0` (south pole) is returned — "rising" loses its meaning there.
/// - `ε = 0`: continuous limit `λ_Asc = θ + π/2` (the ecliptic equals
///   the equator and the equator point of RA `θ + π/2` rises), for any
///   `|φ| < π/2`.
///
/// # Errors
///
/// [`AnglesError::EclipticCoincidesWithReferenceCircle`] when
/// `|N_e × Z| < 1e-12`, i.e. the horizon and the ecliptic coincide.
///
/// # Examples
///
/// ```
/// use oxiephemeris_astro::angles::ascendant;
/// use core::f64::consts::FRAC_PI_2;
/// let eps = 23.4367_f64.to_radians();
/// // At the equator with RAMC = 270° the equinox is exactly rising.
/// let asc = ascendant(3.0 * FRAC_PI_2, 0.0, eps);
/// assert!(matches!(asc, Ok(a) if a.abs() < 1e-15));
/// ```
pub fn ascendant(theta: f64, phi: f64, eps: f64) -> Result<f64, AnglesError> {
    let cos_phi = cos(phi);
    let y = cos_phi * cos(theta);
    let x = -(cos_phi * sin(theta) * cos(eps) + sin(phi) * sin(eps));
    intersection_longitude(y, x)
}

/// Vertex: the ecliptic / prime-vertical intersection in the west,
/// radians in `[0, 2π)`.
///
/// # Derivation
///
/// The prime vertical (the great circle through the zenith and the east
/// and west points of the horizon) has its poles at the north and south
/// points of the horizon (Urania Trust, "The Astronomy of Houses",
/// accessed 2026-07-06), so its pole vector is
/// `N_h = (−sin φ cos θ, −sin φ sin θ, cos φ)`. Observing that
/// `N_h = Z(π/2 − φ, θ + π)` — the zenith of a fictitious observer at
/// co-latitude with the meridian flipped — the intersection
/// `N_e × N_h` follows from the ascendant derivation by substitution:
///
/// ```text
/// λ_Vtx = atan2(−sin φ cos θ, sin φ sin θ cos ε − cos φ sin ε).
/// ```
///
/// The Vertex is the **western** of the two antipodal intersections
/// (the eastern one is the Antivertex / "electric ascendant"), in either
/// hemisphere. The east component of the `N_e × N_h` branch is
///
/// ```text
/// (N_e × N_h) · E = cos φ sin ε sin θ − sin φ cos ε,
/// ```
///
/// so that branch is western for `φ > ε` but *eastern* for `φ < −ε`;
/// when the component is positive the antipode is returned instead,
/// making the result the western intersection always.
///
/// # Degenerate and low-latitude behavior
///
/// - `|φ| > ε`: the sign of the east component cannot change (it would
///   have to pass through a zenith crossing of the ecliptic, impossible
///   outside the tropics), so the returned western point is also
///   continuous in `θ`.
/// - `|φ| ≤ ε` (tropics): the ecliptic crosses the zenith twice per
///   sidereal day and the western intersection jumps by `π` there — the
///   function stays "the western intersection" and is therefore
///   discontinuous at those instants (at the exact crossing, where the
///   east component vanishes, the `N_e × N_h` branch is kept). At
///   `φ = 0` the prime vertical is the celestial equator and the vertex
///   is the equinox currently west (`λ = 0` for `sin θ > 0`, `λ = π`
///   for `sin θ < 0`).
/// - `|φ| = ε` with `θ = ±π/2`: the ecliptic coincides with the prime
///   vertical; an error is returned.
///
/// # Errors
///
/// [`AnglesError::EclipticCoincidesWithReferenceCircle`] when
/// `|N_e × N_h| < 1e-12`, i.e. the prime vertical and the ecliptic
/// coincide.
pub fn vertex(theta: f64, phi: f64, eps: f64) -> Result<f64, AnglesError> {
    let sin_phi = sin(phi);
    let mut y = -sin_phi * cos(theta);
    let mut x = sin_phi * sin(theta) * cos(eps) - cos(phi) * sin(eps);
    // East component of this branch: positive means it is the
    // Antivertex, and the western antipode must be returned.
    if cos(phi) * sin(eps) * sin(theta) - sin_phi * cos(eps) > 0.0 {
        y = -y;
        x = -x;
    }
    intersection_longitude(y, x)
}

/// East Point (equatorial ascendant): the ecliptic point whose right
/// ascension is `θ + π/2`, radians in `[0, 2π)`.
///
/// # Derivation
///
/// At latitude zero the horizon is the great circle of right ascension
/// `θ ± π/2` (it contains both celestial poles), so the ecliptic degree
/// rising there — the equatorial ascendant — is the ecliptic point with
/// `α = θ + π/2`. Substituting `θ + π/2` into the [`mc`] form (or `φ = 0`
/// into the [`ascendant`] form — both give the same expression):
///
/// ```text
/// λ_EP = atan2(cos θ, −sin θ cos ε).
/// ```
///
/// The East Point is latitude-independent and defined for every `θ` and
/// `0 ≤ ε < π/2` (for `ε = 0` it degenerates continuously to
/// `θ + π/2`).
#[must_use]
pub fn east_point(theta: f64, eps: f64) -> f64 {
    normalize_0_two_pi(atan2(cos(theta), -sin(theta) * cos(eps)))
}

/// Shared tail of the plane-intersection formulas: `(y, x)` are the
/// ecliptic-frame components of `N_e × P` for the reference-circle pole
/// `P`; their norm is the sine of the angle between the ecliptic and the
/// reference circle, and vanishing norm means the circles coincide.
fn intersection_longitude(y: f64, x: f64) -> Result<f64, AnglesError> {
    if sqrt(x * x + y * y) < COINCIDENCE_TOL {
        return Err(AnglesError::EclipticCoincidesWithReferenceCircle);
    }
    Ok(normalize_0_two_pi(atan2(y, x)))
}
