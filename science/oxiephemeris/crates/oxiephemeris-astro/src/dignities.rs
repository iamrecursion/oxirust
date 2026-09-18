//! Essential dignities and the element/modality distribution of a chart.
//!
//! # Essential dignity
//!
//! A planet's *essential dignity* measures how well placed it is by
//! zodiacal position alone, through five classical tiers — **domicile**
//! (rulership), **exaltation**, **triplicity**, **term** (bound), and
//! **face** (decan) — together with the two debilities **detriment**
//! (opposite the domicile) and **fall** (opposite the exaltation). A
//! planet with no dignity at all in its place is **peregrine**.
//!
//! The point weights returned in [`EssentialDignity::score`] follow the
//! table popularised by William Lilly, *Christian Astrology* (1647),
//! I.104: domicile +5, exaltation +4, triplicity +3, term +2, face +1,
//! detriment −5, fall −4, peregrine −5.
//!
//! ## Tables
//!
//! - **Domicile / detriment**: the standard sign rulerships. Two schemes
//!   are offered ([`RulershipScheme`]): the seven-planet *traditional*
//!   scheme, and a *modern* scheme assigning Scorpio to Pluto, Aquarius
//!   to Uranus, and Pisces to Neptune. Term, triplicity, and face use
//!   the seven classical planets only, irrespective of scheme.
//! - **Exaltation / fall**: the seven classical exaltations with their
//!   degrees (Sun Aries 19°, Moon Taurus 3°, Mercury Virgo 15°, Venus
//!   Pisces 27°, Mars Capricorn 28°, Jupiter Cancer 15°, Saturn Libra
//!   21°). Ptolemy, *Tetrabiblos* I.19.
//! - **Triplicity**: the Dorothean day/night/participating rulers of the
//!   four elements (Dorotheus of Sidon, *Carmen Astrologicum* I.1);
//!   scoring uses the sect-appropriate ruler.
//! - **Terms (bounds)**: the Egyptian terms as tabulated by Ptolemy,
//!   *Tetrabiblos* I.20-21 — five unequal Saturn/Jupiter/Mars/Venus/
//!   Mercury divisions of each sign.
//! - **Faces (decans)**: the Chaldean decan sequence, the seven planets
//!   in descending order of orbital period beginning with Mars at 0°
//!   Aries, one planet per 10°.
//!
//! # Distribution
//!
//! [`distribution`] tallies how a set of placed bodies falls across the
//! four elements and three modalities — the "balance" of a chart.
//!
//! # Clean-room provenance
//!
//! Every table above is transcribed from the cited classical/printed
//! sources; no Swiss Ephemeris or SOFA/ERFA source was consulted.

use crate::parts::Sect;
use crate::zodiac::{Element, Modality, Sign, SignPosition};

/// A dignity-bearing body: the seven classical planets plus the three
/// modern (trans-Saturnian) planets. The Sun and Moon are included as
/// "planets" in the astrological sense.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Planet {
    /// The Sun.
    Sun,
    /// The Moon.
    Moon,
    /// Mercury.
    Mercury,
    /// Venus.
    Venus,
    /// Mars.
    Mars,
    /// Jupiter.
    Jupiter,
    /// Saturn.
    Saturn,
    /// Uranus (modern).
    Uranus,
    /// Neptune (modern).
    Neptune,
    /// Pluto (modern).
    Pluto,
}

impl Planet {
    /// All ten planets, in the conventional chart order (luminaries
    /// first, then Mercury outward to Pluto). This is the same order as
    /// the CLI's body list, so external consumers can zip the two.
    pub const ALL: [Self; 10] = [
        Self::Sun,
        Self::Moon,
        Self::Mercury,
        Self::Venus,
        Self::Mars,
        Self::Jupiter,
        Self::Saturn,
        Self::Uranus,
        Self::Neptune,
        Self::Pluto,
    ];

    /// A short, stable capitalized name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sun => "Sun",
            Self::Moon => "Moon",
            Self::Mercury => "Mercury",
            Self::Venus => "Venus",
            Self::Mars => "Mars",
            Self::Jupiter => "Jupiter",
            Self::Saturn => "Saturn",
            Self::Uranus => "Uranus",
            Self::Neptune => "Neptune",
            Self::Pluto => "Pluto",
        }
    }
}

/// Which set of sign rulerships to use for domicile and detriment.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RulershipScheme {
    /// The classical seven-planet scheme (Mars rules Aries and Scorpio,
    /// Saturn rules Capricorn and Aquarius, Jupiter rules Sagittarius and
    /// Pisces).
    Traditional,
    /// The modern scheme: Scorpio → Pluto, Aquarius → Uranus, Pisces →
    /// Neptune; all other signs as traditional.
    Modern,
}

/// Traditional domicile ruler of each sign, indexed by [`Sign::index`].
const TRADITIONAL_RULERS: [Planet; 12] = [
    Planet::Mars,    // Aries
    Planet::Venus,   // Taurus
    Planet::Mercury, // Gemini
    Planet::Moon,    // Cancer
    Planet::Sun,     // Leo
    Planet::Mercury, // Virgo
    Planet::Venus,   // Libra
    Planet::Mars,    // Scorpio
    Planet::Jupiter, // Sagittarius
    Planet::Saturn,  // Capricorn
    Planet::Saturn,  // Aquarius
    Planet::Jupiter, // Pisces
];

/// The domicile ruler of `sign` under the given `scheme`.
#[must_use]
pub const fn domicile_ruler(sign: Sign, scheme: RulershipScheme) -> Planet {
    match scheme {
        RulershipScheme::Traditional => TRADITIONAL_RULERS[sign.index()],
        RulershipScheme::Modern => match sign {
            Sign::Scorpio => Planet::Pluto,
            Sign::Aquarius => Planet::Uranus,
            Sign::Pisces => Planet::Neptune,
            other => TRADITIONAL_RULERS[other.index()],
        },
    }
}

/// The sign directly opposite `sign` (180° away).
#[must_use]
const fn opposite(sign: Sign) -> Sign {
    Sign::from_index(sign.index() + 6)
}

/// The classical exaltation of `sign`, if any: the exalted planet and the
/// degree (within the sign, `0..30`) of greatest exaltation.
#[must_use]
pub const fn sign_exaltation(sign: Sign) -> Option<(Planet, f64)> {
    match sign {
        Sign::Aries => Some((Planet::Sun, 19.0)),
        Sign::Taurus => Some((Planet::Moon, 3.0)),
        Sign::Cancer => Some((Planet::Jupiter, 15.0)),
        Sign::Virgo => Some((Planet::Mercury, 15.0)),
        Sign::Libra => Some((Planet::Saturn, 21.0)),
        Sign::Capricorn => Some((Planet::Mars, 28.0)),
        Sign::Pisces => Some((Planet::Venus, 27.0)),
        _ => None,
    }
}

/// The Dorothean triplicity rulers of an element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriplicityRulers {
    /// Ruler by day.
    pub day: Planet,
    /// Ruler by night.
    pub night: Planet,
    /// Participating (co-)ruler, used by both sects.
    pub participating: Planet,
}

impl TriplicityRulers {
    /// The ruler for a given sect (day or night).
    #[must_use]
    pub const fn for_sect(self, sect: Sect) -> Planet {
        match sect {
            Sect::Diurnal => self.day,
            Sect::Nocturnal => self.night,
        }
    }
}

/// The Dorothean triplicity rulers of `sign`'s element.
#[must_use]
pub const fn triplicity_rulers(sign: Sign) -> TriplicityRulers {
    match sign.element() {
        Element::Fire => TriplicityRulers {
            day: Planet::Sun,
            night: Planet::Jupiter,
            participating: Planet::Saturn,
        },
        Element::Earth => TriplicityRulers {
            day: Planet::Venus,
            night: Planet::Moon,
            participating: Planet::Mars,
        },
        Element::Air => TriplicityRulers {
            day: Planet::Saturn,
            night: Planet::Mercury,
            participating: Planet::Jupiter,
        },
        Element::Water => TriplicityRulers {
            day: Planet::Mars,
            night: Planet::Mars,
            participating: Planet::Moon,
        },
    }
}

/// One Egyptian term: its ruler and the upper degree bound (within the
/// sign) at which the term ends.
type Term = (Planet, f64);

/// The Egyptian terms (bounds), five per sign, indexed by
/// [`Sign::index`]. Each entry's `f64` is the exclusive upper bound in
/// degrees within the sign; the five bounds always end at `30.0`.
const EGYPTIAN_TERMS: [[Term; 5]; 12] = [
    // Aries
    [
        (Planet::Jupiter, 6.0),
        (Planet::Venus, 12.0),
        (Planet::Mercury, 20.0),
        (Planet::Mars, 25.0),
        (Planet::Saturn, 30.0),
    ],
    // Taurus
    [
        (Planet::Venus, 8.0),
        (Planet::Mercury, 14.0),
        (Planet::Jupiter, 22.0),
        (Planet::Saturn, 27.0),
        (Planet::Mars, 30.0),
    ],
    // Gemini
    [
        (Planet::Mercury, 6.0),
        (Planet::Jupiter, 12.0),
        (Planet::Venus, 17.0),
        (Planet::Mars, 24.0),
        (Planet::Saturn, 30.0),
    ],
    // Cancer
    [
        (Planet::Mars, 7.0),
        (Planet::Venus, 13.0),
        (Planet::Mercury, 19.0),
        (Planet::Jupiter, 26.0),
        (Planet::Saturn, 30.0),
    ],
    // Leo
    [
        (Planet::Jupiter, 6.0),
        (Planet::Venus, 11.0),
        (Planet::Saturn, 18.0),
        (Planet::Mercury, 24.0),
        (Planet::Mars, 30.0),
    ],
    // Virgo
    [
        (Planet::Mercury, 7.0),
        (Planet::Venus, 17.0),
        (Planet::Jupiter, 21.0),
        (Planet::Mars, 28.0),
        (Planet::Saturn, 30.0),
    ],
    // Libra
    [
        (Planet::Saturn, 6.0),
        (Planet::Mercury, 14.0),
        (Planet::Jupiter, 21.0),
        (Planet::Venus, 28.0),
        (Planet::Mars, 30.0),
    ],
    // Scorpio
    [
        (Planet::Mars, 7.0),
        (Planet::Venus, 11.0),
        (Planet::Mercury, 19.0),
        (Planet::Jupiter, 24.0),
        (Planet::Saturn, 30.0),
    ],
    // Sagittarius
    [
        (Planet::Jupiter, 12.0),
        (Planet::Venus, 17.0),
        (Planet::Mercury, 21.0),
        (Planet::Saturn, 26.0),
        (Planet::Mars, 30.0),
    ],
    // Capricorn
    [
        (Planet::Mercury, 7.0),
        (Planet::Jupiter, 14.0),
        (Planet::Venus, 22.0),
        (Planet::Saturn, 26.0),
        (Planet::Mars, 30.0),
    ],
    // Aquarius
    [
        (Planet::Mercury, 7.0),
        (Planet::Venus, 13.0),
        (Planet::Jupiter, 20.0),
        (Planet::Mars, 25.0),
        (Planet::Saturn, 30.0),
    ],
    // Pisces
    [
        (Planet::Venus, 12.0),
        (Planet::Jupiter, 16.0),
        (Planet::Mercury, 19.0),
        (Planet::Mars, 28.0),
        (Planet::Saturn, 30.0),
    ],
];

/// The Egyptian-term ruler at `degrees_in_sign` (`0..30`) of `sign`.
#[must_use]
pub fn term_ruler(sign: Sign, degrees_in_sign: f64) -> Planet {
    let terms = &EGYPTIAN_TERMS[sign.index()];
    for (planet, upper) in terms {
        if degrees_in_sign < *upper {
            return *planet;
        }
    }
    // `degrees_in_sign` is always < 30 = the last bound; the loop above
    // returns for any in-range input. This fallback keeps the function
    // total for a degenerate (>=30) input without a panic.
    terms[4].0
}

/// The seven classical planets in Chaldean order (descending mean
/// orbital period), the cycle underlying the decan/face assignment.
const CHALDEAN_ORDER: [Planet; 7] = [
    Planet::Mars,
    Planet::Sun,
    Planet::Venus,
    Planet::Mercury,
    Planet::Moon,
    Planet::Saturn,
    Planet::Jupiter,
];

/// The face (decan) ruler at `degrees_in_sign` (`0..30`) of `sign`.
///
/// Faces are 10° each; the ruler cycles through the Chaldean order
/// (`CHALDEAN_ORDER`) starting from Mars at 0° Aries.
#[must_use]
// `within` is clamped to `[0, 30)`, so `floor(within / 10)` is 0, 1, or
// 2 — a small non-negative integer that casts losslessly.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn face_ruler(sign: Sign, degrees_in_sign: f64) -> Planet {
    let within = if degrees_in_sign < 0.0 {
        0.0
    } else if degrees_in_sign >= 30.0 {
        29.999
    } else {
        degrees_in_sign
    };
    let decan_in_sign = libm::floor(within / 10.0) as usize; // 0, 1, or 2
    let decan = sign.index() * 3 + decan_in_sign;
    CHALDEAN_ORDER[decan % 7]
}

/// Lilly point weights for each dignity tier.
const SCORE_DOMICILE: i32 = 5;
const SCORE_EXALTATION: i32 = 4;
const SCORE_TRIPLICITY: i32 = 3;
const SCORE_TERM: i32 = 2;
const SCORE_FACE: i32 = 1;
const SCORE_DETRIMENT: i32 = -5;
const SCORE_FALL: i32 = -4;
const SCORE_PEREGRINE: i32 = -5;

/// The essential dignity of one planet at one ecliptic longitude.
///
/// Each boolean flags whether that tier applies to the queried planet;
/// [`EssentialDignity::score`] is the Lilly point sum (see the module
/// doc). `peregrine` is set only when the planet has none of the five
/// dignities *and* neither debility, and contributes the peregrine
/// penalty (`SCORE_PEREGRINE`).
// The seven dignity/debility flags plus `peregrine` are the natural
// public shape of a dignity report; grouping them into sub-structs would
// only obscure the one-to-one mapping to the classical tiers.
#[allow(clippy::struct_excessive_bools)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EssentialDignity {
    /// Planet rules the sign (domicile).
    pub domicile: bool,
    /// Planet is exalted in the sign.
    pub exaltation: bool,
    /// Planet is the sect's triplicity ruler of the sign's element.
    pub triplicity: bool,
    /// Planet rules the Egyptian term the longitude falls in.
    pub term: bool,
    /// Planet rules the face (decan) the longitude falls in.
    pub face: bool,
    /// Planet is in detriment (sign opposite its domicile).
    pub detriment: bool,
    /// Planet is in fall (sign opposite its exaltation).
    pub fall: bool,
    /// Planet has no dignity and no debility here.
    pub peregrine: bool,
    /// Lilly point sum over all applicable tiers.
    pub score: i32,
}

/// Computes the essential dignity of `planet` at ecliptic longitude
/// `lon_rad`, for a chart of the given `sect` and rulership `scheme`.
///
/// Triplicity, term, and face always use the seven classical planets;
/// only domicile and detriment consult `scheme`. A modern planet
/// (Uranus/Neptune/Pluto) therefore earns triplicity/term/face dignity
/// only under [`RulershipScheme::Modern`] domicile, and never any
/// exaltation (they have no classical exaltation).
#[must_use]
pub fn essential_dignity(
    planet: Planet,
    lon_rad: f64,
    sect: Sect,
    scheme: RulershipScheme,
) -> EssentialDignity {
    let pos = SignPosition::of(lon_rad);
    let sign = pos.sign;
    let deg = pos.degrees_in_sign;

    let domicile = domicile_ruler(sign, scheme) == planet;
    let detriment = domicile_ruler(opposite(sign), scheme) == planet;

    let exaltation = matches!(sign_exaltation(sign), Some((p, _)) if p == planet);
    let fall = matches!(sign_exaltation(opposite(sign)), Some((p, _)) if p == planet);

    let triplicity = triplicity_rulers(sign).for_sect(sect) == planet;
    let term = term_ruler(sign, deg) == planet;
    let face = face_ruler(sign, deg) == planet;

    let has_dignity = domicile || exaltation || triplicity || term || face;
    let peregrine = !has_dignity && !detriment && !fall;

    let mut score = 0;
    if domicile {
        score += SCORE_DOMICILE;
    }
    if exaltation {
        score += SCORE_EXALTATION;
    }
    if triplicity {
        score += SCORE_TRIPLICITY;
    }
    if term {
        score += SCORE_TERM;
    }
    if face {
        score += SCORE_FACE;
    }
    if detriment {
        score += SCORE_DETRIMENT;
    }
    if fall {
        score += SCORE_FALL;
    }
    if peregrine {
        score += SCORE_PEREGRINE;
    }

    EssentialDignity {
        domicile,
        exaltation,
        triplicity,
        term,
        face,
        detriment,
        fall,
        peregrine,
        score,
    }
}

/// The element/modality balance of a set of placed bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Distribution {
    /// Count of bodies in Fire, Earth, Air, Water signs — indexed as
    /// [`Element::ALL`].
    pub elements: [usize; 4],
    /// Count of bodies in Cardinal, Fixed, Mutable signs — indexed as
    /// [`Modality::ALL`].
    pub modalities: [usize; 3],
}

impl Distribution {
    /// The count for a given element.
    #[must_use]
    pub const fn element(&self, element: Element) -> usize {
        match element {
            Element::Fire => self.elements[0],
            Element::Earth => self.elements[1],
            Element::Air => self.elements[2],
            Element::Water => self.elements[3],
        }
    }

    /// The count for a given modality.
    #[must_use]
    pub const fn modality(&self, modality: Modality) -> usize {
        match modality {
            Modality::Cardinal => self.modalities[0],
            Modality::Fixed => self.modalities[1],
            Modality::Mutable => self.modalities[2],
        }
    }
}

/// Tallies the element and modality distribution of a set of signs (one
/// per placed body).
#[must_use]
pub fn distribution(signs: &[Sign]) -> Distribution {
    let mut elements = [0_usize; 4];
    let mut modalities = [0_usize; 3];
    for sign in signs {
        let e = match sign.element() {
            Element::Fire => 0,
            Element::Earth => 1,
            Element::Air => 2,
            Element::Water => 3,
        };
        elements[e] += 1;
        let m = match sign.modality() {
            Modality::Cardinal => 0,
            Modality::Fixed => 1,
            Modality::Mutable => 2,
        };
        modalities[m] += 1;
    }
    Distribution {
        elements,
        modalities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    #[test]
    fn domicile_traditional_and_modern() {
        assert_eq!(
            domicile_ruler(Sign::Leo, RulershipScheme::Traditional),
            Planet::Sun
        );
        assert_eq!(
            domicile_ruler(Sign::Scorpio, RulershipScheme::Traditional),
            Planet::Mars
        );
        assert_eq!(
            domicile_ruler(Sign::Scorpio, RulershipScheme::Modern),
            Planet::Pluto
        );
        assert_eq!(
            domicile_ruler(Sign::Aquarius, RulershipScheme::Modern),
            Planet::Uranus
        );
    }

    #[test]
    fn exaltation_table_spot_checks() {
        assert_eq!(sign_exaltation(Sign::Aries), Some((Planet::Sun, 19.0)));
        assert_eq!(sign_exaltation(Sign::Cancer), Some((Planet::Jupiter, 15.0)));
        assert_eq!(sign_exaltation(Sign::Gemini), None);
    }

    #[test]
    fn sun_in_leo_is_in_domicile() {
        // Sun at 15 Leo (135 deg), day chart.
        let d = essential_dignity(
            Planet::Sun,
            deg(135.0),
            Sect::Diurnal,
            RulershipScheme::Traditional,
        );
        assert!(d.domicile);
        assert!(!d.peregrine);
        assert!(d.score >= SCORE_DOMICILE);
    }

    #[test]
    fn sun_exalted_at_19_aries() {
        let d = essential_dignity(
            Planet::Sun,
            deg(19.0),
            Sect::Diurnal,
            RulershipScheme::Traditional,
        );
        assert!(d.exaltation);
        // Also the day triplicity ruler of Fire.
        assert!(d.triplicity);
    }

    #[test]
    fn saturn_in_aries_is_in_fall() {
        // Saturn's exaltation is Libra; opposite is Aries -> fall.
        let d = essential_dignity(
            Planet::Saturn,
            deg(5.0),
            Sect::Nocturnal,
            RulershipScheme::Traditional,
        );
        assert!(d.fall);
    }

    #[test]
    fn mars_in_libra_is_in_detriment() {
        // Mars rules Aries; opposite Libra -> detriment.
        let d = essential_dignity(
            Planet::Mars,
            deg(190.0),
            Sect::Diurnal,
            RulershipScheme::Traditional,
        );
        assert!(d.detriment);
    }

    #[test]
    fn egyptian_terms_boundaries() {
        // Aries: Jupiter 0-6, Venus 6-12, Mercury 12-20, Mars 20-25,
        // Saturn 25-30.
        assert_eq!(term_ruler(Sign::Aries, 0.0), Planet::Jupiter);
        assert_eq!(term_ruler(Sign::Aries, 5.9), Planet::Jupiter);
        assert_eq!(term_ruler(Sign::Aries, 6.0), Planet::Venus);
        assert_eq!(term_ruler(Sign::Aries, 19.9), Planet::Mercury);
        assert_eq!(term_ruler(Sign::Aries, 20.0), Planet::Mars);
        assert_eq!(term_ruler(Sign::Aries, 29.0), Planet::Saturn);
    }

    #[test]
    fn faces_follow_chaldean_cycle() {
        // Aries: Mars, Sun, Venus.
        assert_eq!(face_ruler(Sign::Aries, 5.0), Planet::Mars);
        assert_eq!(face_ruler(Sign::Aries, 15.0), Planet::Sun);
        assert_eq!(face_ruler(Sign::Aries, 25.0), Planet::Venus);
        // Taurus: Mercury, Moon, Saturn.
        assert_eq!(face_ruler(Sign::Taurus, 5.0), Planet::Mercury);
        // Gemini first face is Jupiter.
        assert_eq!(face_ruler(Sign::Gemini, 5.0), Planet::Jupiter);
        // Leo faces: Saturn, Jupiter, Mars.
        assert_eq!(face_ruler(Sign::Leo, 5.0), Planet::Saturn);
        assert_eq!(face_ruler(Sign::Leo, 25.0), Planet::Mars);
    }

    #[test]
    fn peregrine_when_no_dignity() {
        // Moon in Sagittarius ~ 5 deg, night chart: not ruler (Jupiter),
        // not exalted, Fire night triplicity ruler is Jupiter, term
        // (Sag 0-12 = Jupiter), face (Sag 0-10 = Mercury). Moon has none
        // -> peregrine.
        let d = essential_dignity(
            Planet::Moon,
            deg(245.0),
            Sect::Nocturnal,
            RulershipScheme::Traditional,
        );
        assert!(!d.domicile && !d.exaltation && !d.triplicity && !d.term && !d.face);
        assert!(!d.detriment && !d.fall);
        assert!(d.peregrine);
        assert_eq!(d.score, SCORE_PEREGRINE);
    }

    #[test]
    fn distribution_counts_sum_to_input() {
        let signs = [Sign::Aries, Sign::Aries, Sign::Cancer, Sign::Libra];
        let dist = distribution(&signs);
        let elem_total: usize = dist.elements.iter().sum();
        let mod_total: usize = dist.modalities.iter().sum();
        assert_eq!(elem_total, 4);
        assert_eq!(mod_total, 4);
        assert_eq!(dist.element(Element::Fire), 2);
        assert_eq!(dist.element(Element::Water), 1);
        assert_eq!(dist.modality(Modality::Cardinal), 4);
    }
}
