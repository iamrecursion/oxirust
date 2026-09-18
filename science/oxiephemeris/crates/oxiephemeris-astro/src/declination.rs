//! Ecliptic-to-equatorial declination, antiscia, and the declination
//! aspects (parallel / contraparallel, out-of-bounds).
//!
//! # Declination
//!
//! Given an ecliptic longitude `lambda` and latitude `beta` (radians) and
//! the true obliquity of date `epsilon`, the equatorial declination is
//!
//! ```text
//! sin(delta) = cos(epsilon) sin(beta) + sin(epsilon) cos(beta) sin(lambda)
//! ```
//!
//! — the z-component of the ecliptic unit vector after the frame rotation
//! `R1(-epsilon)` from the ecliptic to the equator (IERS TN36 §5.4 sign
//! convention). See [`declination`].
//!
//! # Antiscia
//!
//! The **antiscion** of a point is its mirror image across the solstitial
//! (0° Cancer / 0° Capricorn) axis: `antiscia(lambda) = pi - lambda`. Two
//! points in antiscia share the same declination on the same side of the
//! equator. The **contra-antiscion** mirrors across the equinoctial (0°
//! Aries / 0° Libra) axis: `contra_antiscia(lambda) = -lambda`; such
//! points have equal and opposite declination. See [`antiscia`] /
//! [`contra_antiscia`].
//!
//! # Declination aspects
//!
//! Two bodies are in **parallel** when their declinations are equal
//! (same side) within an orb, and in **contraparallel** when they are
//! equal in magnitude but opposite in side. A body is **out of bounds**
//! when `|delta|` exceeds the obliquity — beyond the Sun's extreme
//! declination. See [`declination_aspect`] and [`is_out_of_bounds`].
//!
//! # Clean-room provenance
//!
//! The declination formula is the elementary ecliptic→equatorial
//! rotation; antiscia and the declination aspects are classical
//! definitions. No Swiss Ephemeris or SOFA/ERFA source was consulted.

use oxiephemeris_core::angle::normalize_0_two_pi;

/// The equatorial declination (radians, in `[-pi/2, pi/2]`) of an
/// ecliptic position `(lon_rad, lat_rad)` under true obliquity
/// `obliquity_rad`.
#[must_use]
pub fn declination(lon_rad: f64, lat_rad: f64, obliquity_rad: f64) -> f64 {
    let sin_lon = libm::sin(lon_rad);
    let cos_beta = libm::cos(lat_rad);
    let sin_beta = libm::sin(lat_rad);
    let sin_eps = libm::sin(obliquity_rad);
    let cos_eps = libm::cos(obliquity_rad);
    // Clamp guards the poles against a rounding overshoot that would make
    // `asin` return NaN (a NaN input still propagates, since `clamp`
    // leaves NaN untouched).
    let sin_dec = (cos_eps * sin_beta + sin_eps * cos_beta * sin_lon).clamp(-1.0, 1.0);
    libm::asin(sin_dec)
}

/// The antiscion of `lon_rad`: its reflection across the solstitial axis,
/// `pi - lon`, wrapped to `[0, 2*pi)`.
#[must_use]
pub fn antiscia(lon_rad: f64) -> f64 {
    normalize_0_two_pi(core::f64::consts::PI - lon_rad)
}

/// The contra-antiscion of `lon_rad`: its reflection across the
/// equinoctial axis, `-lon`, wrapped to `[0, 2*pi)`.
#[must_use]
pub fn contra_antiscia(lon_rad: f64) -> f64 {
    normalize_0_two_pi(-lon_rad)
}

/// A declination relationship between two bodies.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeclinationAspect {
    /// Equal declinations on the same side of the equator.
    Parallel,
    /// Equal-magnitude declinations on opposite sides.
    Contraparallel,
}

impl DeclinationAspect {
    /// Both declination aspects, in canonical order.
    pub const ALL: [Self; 2] = [Self::Parallel, Self::Contraparallel];

    /// A short, stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Parallel => "parallel",
            Self::Contraparallel => "contraparallel",
        }
    }
}

/// Classifies the declination relationship between `dec1_rad` and
/// `dec2_rad` within `orb_rad`, or `None` if neither holds.
///
/// Parallel (same-side match) is tested first; a pair straddling the
/// equator at near-zero declination could satisfy both tests, and the
/// same-side reading is the conventional one.
#[must_use]
pub fn declination_aspect(dec1_rad: f64, dec2_rad: f64, orb_rad: f64) -> Option<DeclinationAspect> {
    let orb = libm::fabs(orb_rad);
    if libm::fabs(dec1_rad - dec2_rad) <= orb {
        Some(DeclinationAspect::Parallel)
    } else if libm::fabs(dec1_rad + dec2_rad) <= orb {
        Some(DeclinationAspect::Contraparallel)
    } else {
        None
    }
}

/// Whether a body at declination `dec_rad` is "out of bounds" — beyond
/// the obliquity `obliquity_rad` in absolute declination.
#[must_use]
pub fn is_out_of_bounds(dec_rad: f64, obliquity_rad: f64) -> bool {
    libm::fabs(dec_rad) > libm::fabs(obliquity_rad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    /// A mean obliquity for tests (IAU value near J2000), radians.
    const EPS: f64 = 0.409_092_6; // ~23.4366 deg

    #[test]
    fn declination_on_ecliptic_matches_closed_form() {
        // For beta = 0 the general formula reduces to the textbook
        // identity sin(delta) = sin(eps) sin(lambda).
        for lon_deg in [0.0, 30.0, 90.0, 200.0, 235.5, 315.0] {
            let lon = deg(lon_deg);
            let dec = declination(lon, 0.0, EPS);
            let expected = libm::asin(libm::sin(EPS) * libm::sin(lon));
            assert!(
                libm::fabs(dec - expected) < 1e-12,
                "lon {lon_deg}: {dec} vs {expected}"
            );
        }
    }

    #[test]
    fn declination_extremes_at_solstices() {
        // 90 deg (0 Cancer) -> +eps; 270 deg (0 Capricorn) -> -eps.
        assert!(libm::fabs(declination(deg(90.0), 0.0, EPS) - EPS) < 1e-12);
        assert!(libm::fabs(declination(deg(270.0), 0.0, EPS) + EPS) < 1e-12);
        // Equinoxes -> 0.
        assert!(libm::fabs(declination(deg(0.0), 0.0, EPS)) < 1e-12);
        assert!(libm::fabs(declination(deg(180.0), 0.0, EPS)) < 1e-12);
    }

    #[test]
    fn latitude_can_push_out_of_bounds() {
        // A body at 0 Cancer with +5 deg ecliptic latitude exceeds the
        // obliquity in declination.
        let dec = declination(deg(90.0), deg(5.0), EPS);
        assert!(dec > EPS);
        assert!(is_out_of_bounds(dec, EPS));
        // On the ecliptic it is exactly at the bound, not beyond.
        assert!(!is_out_of_bounds(declination(deg(90.0), 0.0, EPS), EPS));
    }

    #[test]
    fn antiscia_pairs() {
        // 10 Aries (10 deg) <-> 20 Virgo (170 deg).
        assert!(libm::fabs(antiscia(deg(10.0)) - deg(170.0)) < 1e-12);
        // Antiscia is an involution.
        assert!(libm::fabs(antiscia(antiscia(deg(10.0))) - deg(10.0)) < 1e-12);
        // Antiscia points share declination on the ecliptic.
        let d1 = declination(deg(10.0), 0.0, EPS);
        let d2 = declination(antiscia(deg(10.0)), 0.0, EPS);
        assert!(libm::fabs(d1 - d2) < 1e-12);
    }

    #[test]
    fn contra_antiscia_pairs() {
        // 10 Aries <-> 20 Pisces (350 deg).
        assert!(libm::fabs(contra_antiscia(deg(10.0)) - deg(350.0)) < 1e-12);
        // Contra-antiscia points have opposite declination on the ecliptic.
        let d1 = declination(deg(10.0), 0.0, EPS);
        let d2 = declination(contra_antiscia(deg(10.0)), 0.0, EPS);
        assert!(libm::fabs(d1 + d2) < 1e-12);
    }

    #[test]
    fn parallel_and_contraparallel() {
        let orb = deg(1.0);
        assert_eq!(
            declination_aspect(deg(20.0), deg(20.5), orb),
            Some(DeclinationAspect::Parallel)
        );
        assert_eq!(
            declination_aspect(deg(20.0), deg(-20.5), orb),
            Some(DeclinationAspect::Contraparallel)
        );
        assert_eq!(declination_aspect(deg(20.0), deg(10.0), orb), None);
    }
}
