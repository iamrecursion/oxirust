//! Chart sect (day/night) and the hermetic Arabic Parts (Lots).
//!
//! # Sect
//!
//! A chart is **diurnal** ("of the day") when the Sun is above the
//! horizon, and **nocturnal** otherwise. With the Ascendant longitude
//! known, a degree of the ecliptic is above the horizon exactly when it
//! lies in the upper hemisphere — the semicircle from the Descendant up
//! over the Midheaven to the Ascendant. Numbering the houses
//! counter-clockwise from the Ascendant, that upper hemisphere is houses
//! 7-12, i.e. ecliptic longitudes for which `wrap(lon - asc)` lies in
//! `[pi, 2*pi)`. See [`sect`].
//!
//! # Lots
//!
//! A **Lot** (Arabic Part) is a point defined by projecting the arc
//! between two chart factors from a third. The archetypal example is the
//! **Lot of Fortune**, whose formula reverses between day and night
//! charts (Paulus Alexandrinus, 4th c.; codified by Bonatti and later
//! authors):
//!
//! ```text
//! diurnal:   Fortune = Asc + (Moon - Sun)
//! nocturnal: Fortune = Asc + (Sun  - Moon)
//! ```
//!
//! The **Lot of Spirit** is Fortune's sect mirror (the Sun/Moon roles
//! swapped), and every classical lot follows the same `Asc + (A - B)`
//! shape with a day/night swap of `A` and `B` — captured by the general
//! [`lot`] helper.
//!
//! All longitudes are radians and every result is wrapped to
//! `[0, 2*pi)`.
//!
//! # Clean-room provenance
//!
//! The sect rule and the Fortune/Spirit formulae are classical
//! definitions of Hellenistic astrology; they are elementary modular
//! arithmetic on the circle, not sourced from any ephemeris
//! implementation. No Swiss Ephemeris or SOFA/ERFA source was consulted.

use oxiephemeris_core::angle::normalize_0_two_pi;

/// Whether a chart is of the day or of the night.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sect {
    /// The Sun is above the horizon.
    Diurnal,
    /// The Sun is below the horizon.
    Nocturnal,
}

impl Sect {
    /// Both sects, in canonical order.
    pub const ALL: [Self; 2] = [Self::Diurnal, Self::Nocturnal];

    /// A short, stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Diurnal => "diurnal",
            Self::Nocturnal => "nocturnal",
        }
    }

    /// `true` for a daytime (diurnal) chart.
    #[must_use]
    pub const fn is_diurnal(self) -> bool {
        matches!(self, Self::Diurnal)
    }
}

/// The sect of a chart from the Sun's and the Ascendant's ecliptic
/// longitudes (radians).
///
/// The Sun is above the horizon — a diurnal chart — exactly when
/// `wrap(sun_lon - asc_lon)` is in `[pi, 2*pi)` (houses 7-12). At the two
/// horizon crossings the classification is a boundary convention: the
/// Ascendant itself (`wrap == 0`) is treated as nocturnal (the Sun has
/// not yet risen) and the Descendant (`wrap == pi`) as diurnal.
#[must_use]
pub fn sect(sun_lon: f64, asc_lon: f64) -> Sect {
    if normalize_0_two_pi(sun_lon - asc_lon) >= core::f64::consts::PI {
        Sect::Diurnal
    } else {
        Sect::Nocturnal
    }
}

/// A general sect-sensitive lot: `Asc + (a - b)` by day, `Asc + (b - a)`
/// by night.
///
/// Passing the diurnal `(a, b)` ordering lets the day/night reversal be
/// handled here once. For the Lot of Fortune, `a = Moon`, `b = Sun`; for
/// the Lot of Spirit, `a = Sun`, `b = Moon`.
#[must_use]
pub fn lot(asc_lon: f64, a_lon: f64, b_lon: f64, sect: Sect) -> f64 {
    let arc = match sect {
        Sect::Diurnal => a_lon - b_lon,
        Sect::Nocturnal => b_lon - a_lon,
    };
    normalize_0_two_pi(asc_lon + arc)
}

/// The Lot (Part) of Fortune.
///
/// `Asc + (Moon - Sun)` by day, `Asc + (Sun - Moon)` by night.
#[must_use]
pub fn part_of_fortune(asc_lon: f64, sun_lon: f64, moon_lon: f64, sect: Sect) -> f64 {
    lot(asc_lon, moon_lon, sun_lon, sect)
}

/// The Lot (Part) of Spirit — Fortune's sect mirror.
///
/// `Asc + (Sun - Moon)` by day, `Asc + (Moon - Sun)` by night.
#[must_use]
pub fn part_of_spirit(asc_lon: f64, sun_lon: f64, moon_lon: f64, sect: Sect) -> f64 {
    lot(asc_lon, sun_lon, moon_lon, sect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    fn approx(a: f64, b: f64) -> bool {
        libm::fabs(normalize_0_two_pi(a) - normalize_0_two_pi(b)) < 1e-9
    }

    #[test]
    fn sect_upper_hemisphere_is_diurnal() {
        // Sun exactly at MC-ish (asc + 270 deg -> wrap 270 in [180,360)).
        assert_eq!(sect(deg(270.0), deg(0.0)), Sect::Diurnal);
        // Sun just below the eastern horizon (house 1): asc + 10 deg.
        assert_eq!(sect(deg(10.0), deg(0.0)), Sect::Nocturnal);
        // Sun just above the eastern horizon (house 12): asc - 10 deg.
        assert_eq!(sect(deg(-10.0), deg(0.0)), Sect::Diurnal);
    }

    #[test]
    fn sect_boundary_conventions() {
        // Ascendant itself -> not yet risen -> nocturnal.
        assert_eq!(sect(deg(30.0), deg(30.0)), Sect::Nocturnal);
        // Descendant -> setting, still above -> diurnal.
        assert_eq!(sect(deg(210.0), deg(30.0)), Sect::Diurnal);
    }

    #[test]
    fn fortune_and_spirit_swap_between_day_and_night() {
        let asc = deg(30.0);
        let sun = deg(235.0);
        let moon = deg(285.0);
        // Day Fortune == Night Spirit, and vice versa.
        let day_fortune = part_of_fortune(asc, sun, moon, Sect::Diurnal);
        let night_spirit = part_of_spirit(asc, sun, moon, Sect::Nocturnal);
        assert!(approx(day_fortune, night_spirit));
        let night_fortune = part_of_fortune(asc, sun, moon, Sect::Nocturnal);
        let day_spirit = part_of_spirit(asc, sun, moon, Sect::Diurnal);
        assert!(approx(night_fortune, day_spirit));
    }

    #[test]
    fn fortune_formula_matches_definition() {
        let asc = deg(30.0);
        let sun = deg(235.0);
        let moon = deg(285.0);
        // Diurnal: Asc + Moon - Sun = 30 + 285 - 235 = 80 deg.
        assert!(approx(
            part_of_fortune(asc, sun, moon, Sect::Diurnal),
            deg(80.0)
        ));
        // Nocturnal: Asc + Sun - Moon = 30 + 235 - 285 = -20 -> 340 deg.
        assert!(approx(
            part_of_fortune(asc, sun, moon, Sect::Nocturnal),
            deg(340.0)
        ));
    }

    #[test]
    fn results_are_normalized() {
        let f = part_of_fortune(deg(350.0), deg(20.0), deg(200.0), Sect::Diurnal);
        assert!((0.0..2.0 * PI).contains(&f));
    }
}
