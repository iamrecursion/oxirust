//! SE-documented body-number constants and the mapping onto this crate's
//! internal targets.
//!
//! # Provenance
//!
//! Numeric values from the Swiss Ephemeris programmer's documentation
//! (Astrodienst, <https://www.astro.com/swisseph/swephprg.htm>, §3.2
//! "Planets and stars", consulted 2026-07-06) — documentation only, per
//! this crate's clean-room policy (see [`crate::flags`]'s provenance
//! note, which applies identically here).
//!
//! # Supported set
//!
//! [`SE_SUN`]`..=`[`SE_PLUTO`] (0–9), [`SE_MEAN_NODE`], [`SE_TRUE_NODE`],
//! [`SE_MEAN_APOG`], [`SE_OSCU_APOG`] (10–13), and [`SE_EARTH`] (14,
//! meaningful only for [`Center::Heliocentric`]/[`Center::Barycentric`],
//! exactly like [`oxiephemeris_bodies::apparent::Target::Earth`]).
//! Everything else — Chiron (SE body 15) and the other minor planets,
//! the asteroid-number range (`SE_AST_OFFSET`-based), fixed stars, and
//! fictitious bodies — is **not implemented** through this numeric-body
//! interface; the internal resolver returns
//! [`crate::CompatError::UnsupportedBody`] for every one of them.

use oxiephemeris_bodies::apparent::{Center, Target};

use crate::CompatError;

/// The Sun.
pub const SE_SUN: i32 = 0;
/// The Moon.
pub const SE_MOON: i32 = 1;
/// Mercury.
pub const SE_MERCURY: i32 = 2;
/// Venus.
pub const SE_VENUS: i32 = 3;
/// Mars.
pub const SE_MARS: i32 = 4;
/// Jupiter.
pub const SE_JUPITER: i32 = 5;
/// Saturn.
pub const SE_SATURN: i32 = 6;
/// Uranus.
pub const SE_URANUS: i32 = 7;
/// Neptune.
pub const SE_NEPTUNE: i32 = 8;
/// Pluto.
pub const SE_PLUTO: i32 = 9;
/// Mean lunar ascending node (see [`crate::context::calc`] for the
/// "no radius/latitude convention" honesty note shared by all four node
/// and apogee bodies).
pub const SE_MEAN_NODE: i32 = 10;
/// True (osculating) lunar ascending node.
pub const SE_TRUE_NODE: i32 = 11;
/// Mean lunar apogee ("mean Black Moon Lilith").
pub const SE_MEAN_APOG: i32 = 12;
/// Osculating lunar apogee ("true Black Moon Lilith").
pub const SE_OSCU_APOG: i32 = 13;
/// The Earth. Only defined for a non-geocentric [`Center`]; see the
/// module docs.
pub const SE_EARTH: i32 = 14;

/// A resolved calculation target: either a direct DE-ephemeris body or
/// one of the four lunar node/apogee variants (which route through
/// [`oxiephemeris_astro::nodes`] instead of
/// [`oxiephemeris_bodies::apparent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResolvedBody {
    /// A direct ephemeris target (Sun..Pluto, Earth).
    Direct(Target),
    /// Mean node, mean apogee, true node, or osculating apogee — see
    /// [`NodeKind`].
    Node(NodeKind),
}

/// Which of the four SE node/apogee bodies was requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum NodeKind {
    MeanNode,
    TrueNode,
    MeanApogee,
    OscuApogee,
}

/// Resolves an SE body number to a [`ResolvedBody`].
///
/// # Errors
///
/// [`CompatError::UnsupportedBody`] for any number outside the
/// documented supported set (see the module docs).
pub(crate) fn resolve(body: i32) -> Result<ResolvedBody, CompatError> {
    match body {
        SE_SUN => Ok(ResolvedBody::Direct(Target::Sun)),
        SE_MOON => Ok(ResolvedBody::Direct(Target::Moon)),
        SE_MERCURY => Ok(ResolvedBody::Direct(Target::Mercury)),
        SE_VENUS => Ok(ResolvedBody::Direct(Target::Venus)),
        SE_MARS => Ok(ResolvedBody::Direct(Target::Mars)),
        SE_JUPITER => Ok(ResolvedBody::Direct(Target::Jupiter)),
        SE_SATURN => Ok(ResolvedBody::Direct(Target::Saturn)),
        SE_URANUS => Ok(ResolvedBody::Direct(Target::Uranus)),
        SE_NEPTUNE => Ok(ResolvedBody::Direct(Target::Neptune)),
        SE_PLUTO => Ok(ResolvedBody::Direct(Target::Pluto)),
        SE_MEAN_NODE => Ok(ResolvedBody::Node(NodeKind::MeanNode)),
        SE_TRUE_NODE => Ok(ResolvedBody::Node(NodeKind::TrueNode)),
        SE_MEAN_APOG => Ok(ResolvedBody::Node(NodeKind::MeanApogee)),
        SE_OSCU_APOG => Ok(ResolvedBody::Node(NodeKind::OscuApogee)),
        SE_EARTH => Ok(ResolvedBody::Direct(Target::Earth)),
        other => Err(CompatError::UnsupportedBody(other)),
    }
}

/// Validates that `center` is geocentric, for the node/apogee bodies
/// (which this crate treats as inherently geocentric ecliptic-of-date
/// quantities — see [`crate::context::calc`]).
pub(crate) fn require_geocentric(center: Center) -> Result<(), CompatError> {
    if center == Center::Geocentric {
        Ok(())
    } else {
        Err(CompatError::ConflictingFlags(
            "SE_MEAN_NODE / SE_TRUE_NODE / SE_MEAN_APOG / SE_OSCU_APOG are always geocentric in \
             this crate: SEFLG_HELCTR / SEFLG_BARYCTR are not supported for these targets",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve, NodeKind, ResolvedBody};
    use crate::CompatError;
    use oxiephemeris_bodies::apparent::Target;

    #[test]
    fn documented_numeric_values() {
        assert_eq!(super::SE_SUN, 0);
        assert_eq!(super::SE_MOON, 1);
        assert_eq!(super::SE_MERCURY, 2);
        assert_eq!(super::SE_VENUS, 3);
        assert_eq!(super::SE_MARS, 4);
        assert_eq!(super::SE_JUPITER, 5);
        assert_eq!(super::SE_SATURN, 6);
        assert_eq!(super::SE_URANUS, 7);
        assert_eq!(super::SE_NEPTUNE, 8);
        assert_eq!(super::SE_PLUTO, 9);
        assert_eq!(super::SE_MEAN_NODE, 10);
        assert_eq!(super::SE_TRUE_NODE, 11);
        assert_eq!(super::SE_MEAN_APOG, 12);
        assert_eq!(super::SE_OSCU_APOG, 13);
        assert_eq!(super::SE_EARTH, 14);
    }

    #[test]
    fn resolves_every_supported_body() {
        assert_eq!(
            resolve(super::SE_SUN),
            Ok(ResolvedBody::Direct(Target::Sun))
        );
        assert_eq!(
            resolve(super::SE_PLUTO),
            Ok(ResolvedBody::Direct(Target::Pluto))
        );
        assert_eq!(
            resolve(super::SE_EARTH),
            Ok(ResolvedBody::Direct(Target::Earth))
        );
        assert_eq!(
            resolve(super::SE_MEAN_NODE),
            Ok(ResolvedBody::Node(NodeKind::MeanNode))
        );
        assert_eq!(
            resolve(super::SE_TRUE_NODE),
            Ok(ResolvedBody::Node(NodeKind::TrueNode))
        );
        assert_eq!(
            resolve(super::SE_MEAN_APOG),
            Ok(ResolvedBody::Node(NodeKind::MeanApogee))
        );
        assert_eq!(
            resolve(super::SE_OSCU_APOG),
            Ok(ResolvedBody::Node(NodeKind::OscuApogee))
        );
    }

    #[test]
    fn rejects_chiron_and_asteroids_and_negatives() {
        for unsupported in [-1, 15, 16, 20, 10_000, 10_001] {
            assert_eq!(
                resolve(unsupported),
                Err(CompatError::UnsupportedBody(unsupported))
            );
        }
    }
}
