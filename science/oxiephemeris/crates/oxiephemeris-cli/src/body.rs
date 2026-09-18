//! Case-insensitive body-name lookup for the `pos` subcommand.
//!
//! Supported names: `sun`, `moon`, `mercury`, `venus`, `mars`, `jupiter`,
//! `saturn`, `uranus`, `neptune`, `pluto`.
//! Earth is intentionally not offered: every [`Center`](oxiephemeris_bodies::Center)
//! choice has the Earth as either the observer itself or unreachable
//! without a second body's ephemeris in the same slot, so it is out of
//! scope for the `pos` command (see `apparent`'s `Target::Earth` doc
//! comment).

use oxiephemeris_bodies::Target;

/// Every body `pos --all` and `chart` compute, in this fixed order (also
/// the order `pos --all --json`'s `bodies` array and `chart --json`'s
/// `bodies`/aspect-pair enumeration use): Sun, Moon, Mercury, Venus,
/// Mars, Jupiter, Saturn, Uranus, Neptune, Pluto — the same set
/// [`parse_body`] accepts, Earth excluded for the reasons on this
/// module's doc comment. This order is part of the stable JSON schemas
/// that reference it.
pub const ALL_BODIES: [(Target, &str); 10] = [
    (Target::Sun, "Sun"),
    (Target::Moon, "Moon"),
    (Target::Mercury, "Mercury"),
    (Target::Venus, "Venus"),
    (Target::Mars, "Mars"),
    (Target::Jupiter, "Jupiter"),
    (Target::Saturn, "Saturn"),
    (Target::Uranus, "Uranus"),
    (Target::Neptune, "Neptune"),
    (Target::Pluto, "Pluto"),
];

/// Resolves a case-insensitive body name to a [`Target`] and its canonical
/// display name (title case).
///
/// # Errors
///
/// Returns the offending input, wrapped for the caller to format a
/// friendly message listing the supported names.
pub fn parse_body(name: &str) -> Result<(Target, &'static str), String> {
    match name.to_ascii_lowercase().as_str() {
        "sun" => Ok((Target::Sun, "Sun")),
        "moon" => Ok((Target::Moon, "Moon")),
        "mercury" => Ok((Target::Mercury, "Mercury")),
        "venus" => Ok((Target::Venus, "Venus")),
        "mars" => Ok((Target::Mars, "Mars")),
        "jupiter" => Ok((Target::Jupiter, "Jupiter")),
        "saturn" => Ok((Target::Saturn, "Saturn")),
        "uranus" => Ok((Target::Uranus, "Uranus")),
        "neptune" => Ok((Target::Neptune, "Neptune")),
        "pluto" => Ok((Target::Pluto, "Pluto")),
        other => Err(format!(
            "unknown body '{other}'; supported: sun, moon, mercury, venus, mars, \
             jupiter, saturn, uranus, neptune, pluto"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_body;
    use oxiephemeris_bodies::Target;

    #[test]
    fn case_insensitive() {
        assert_eq!(parse_body("MaRs").map(|(t, _)| t), Ok(Target::Mars));
        assert_eq!(parse_body("PLUTO").map(|(t, _)| t), Ok(Target::Pluto));
    }

    #[test]
    fn unknown_name_is_an_error() {
        assert!(parse_body("earth").is_err());
        assert!(parse_body("ceres").is_err());
    }
}
