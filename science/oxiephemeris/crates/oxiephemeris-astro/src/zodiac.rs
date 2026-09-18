//! The tropical zodiac: the twelve 30-degree signs, their classical
//! element (triplicity) and modality (quadruplicity), and the
//! decomposition of an ecliptic longitude into a sign plus
//! degrees/arcminutes/arcseconds within that sign.
//!
//! # Model
//!
//! The zodiac divides the ecliptic into twelve equal 30-degree arcs
//! beginning at the vernal equinox (0 degrees of Aries). A longitude
//! `lambda` (radians, measured along the ecliptic from the equinox in
//! the sense of increasing right ascension) falls in sign number
//! `floor(lambda / 30 deg)`, counting Aries = 0. Whether `lambda` is a
//! *tropical* (equinox-anchored) or *sidereal* (star-anchored) longitude
//! is the caller's choice: this module performs the same 30-degree
//! partition either way — for the sidereal zodiac the caller simply
//! subtracts the ayanamsha first (see [`crate::ayanamsha`]).
//!
//! The four classical elements (Fire, Earth, Air, Water) and the three
//! modalities (Cardinal, Fixed, Mutable) are the standard triplicity and
//! quadruplicity groupings: element repeats every four signs
//! (Aries/Leo/Sagittarius = Fire, ...) and modality every three
//! (Aries/Cancer/Libra/Capricorn = Cardinal, ...).
//!
//! # Clean-room provenance
//!
//! The 30-degree sign division and the element/modality groupings are
//! elementary, millennia-old definitions of Western astrology (attested
//! e.g. in Ptolemy, *Tetrabiblos* I.11-13); they are plain arithmetic on
//! the circle, not sourced from any ephemeris implementation. No Swiss
//! Ephemeris or SOFA/ERFA source was consulted.

use oxiephemeris_core::angle::{normalize_0_two_pi, RAD2DEG};

/// One-twelfth of the circle, in radians: the angular width of a sign.
const SIGN_WIDTH_RAD: f64 = core::f64::consts::PI / 6.0;

/// The classical element (triplicity) of a sign.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    /// Fire: Aries, Leo, Sagittarius.
    Fire,
    /// Earth: Taurus, Virgo, Capricorn.
    Earth,
    /// Air: Gemini, Libra, Aquarius.
    Air,
    /// Water: Cancer, Scorpio, Pisces.
    Water,
}

impl Element {
    /// A short, stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fire => "fire",
            Self::Earth => "earth",
            Self::Air => "air",
            Self::Water => "water",
        }
    }

    /// All four elements, in canonical order.
    pub const ALL: [Self; 4] = [Self::Fire, Self::Earth, Self::Air, Self::Water];
}

/// The modality (quadruplicity) of a sign.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Cardinal: Aries, Cancer, Libra, Capricorn.
    Cardinal,
    /// Fixed: Taurus, Leo, Scorpio, Aquarius.
    Fixed,
    /// Mutable: Gemini, Virgo, Sagittarius, Pisces.
    Mutable,
}

impl Modality {
    /// A short, stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cardinal => "cardinal",
            Self::Fixed => "fixed",
            Self::Mutable => "mutable",
        }
    }

    /// All three modalities, in canonical order.
    pub const ALL: [Self; 3] = [Self::Cardinal, Self::Fixed, Self::Mutable];
}

/// One of the twelve zodiac signs, in ecliptic order from the vernal
/// equinox.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    /// 0-30 degrees. Cardinal Fire.
    Aries,
    /// 30-60 degrees. Fixed Earth.
    Taurus,
    /// 60-90 degrees. Mutable Air.
    Gemini,
    /// 90-120 degrees. Cardinal Water.
    Cancer,
    /// 120-150 degrees. Fixed Fire.
    Leo,
    /// 150-180 degrees. Mutable Earth.
    Virgo,
    /// 180-210 degrees. Cardinal Air.
    Libra,
    /// 210-240 degrees. Fixed Water.
    Scorpio,
    /// 240-270 degrees. Mutable Fire.
    Sagittarius,
    /// 270-300 degrees. Cardinal Earth.
    Capricorn,
    /// 300-330 degrees. Fixed Air.
    Aquarius,
    /// 330-360 degrees. Mutable Water.
    Pisces,
}

impl Sign {
    /// All twelve signs, in ecliptic order (Aries first).
    pub const ALL: [Self; 12] = [
        Self::Aries,
        Self::Taurus,
        Self::Gemini,
        Self::Cancer,
        Self::Leo,
        Self::Virgo,
        Self::Libra,
        Self::Scorpio,
        Self::Sagittarius,
        Self::Capricorn,
        Self::Aquarius,
        Self::Pisces,
    ];

    /// The sign containing ecliptic longitude `lon_rad`.
    ///
    /// The longitude is first wrapped to `[0, 2*pi)`, so any real input
    /// is accepted. A non-finite input propagates through
    /// [`normalize_0_two_pi`] and would land on `Aries` via the final
    /// clamp; callers that care should reject non-finite longitudes
    /// upstream.
    #[must_use]
    // `floor(x / 30 deg)` for `x` in `[0, 360 deg)` is a non-negative
    // integer in `0..=11`, so the cast neither truncates meaningfully nor
    // loses a sign.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn of(lon_rad: f64) -> Self {
        let idx = libm::floor(normalize_0_two_pi(lon_rad) / SIGN_WIDTH_RAD) as usize;
        // `normalize_0_two_pi` maps into `[0, 2*pi)`, so `idx` is in
        // `0..=11`; the `min` guards the half-ulp corner where the
        // division rounds up to exactly 12.
        Self::ALL[if idx > 11 { 11 } else { idx }]
    }

    /// The 0-based index of this sign (Aries = 0, ..., Pisces = 11).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Aries => 0,
            Self::Taurus => 1,
            Self::Gemini => 2,
            Self::Cancer => 3,
            Self::Leo => 4,
            Self::Virgo => 5,
            Self::Libra => 6,
            Self::Scorpio => 7,
            Self::Sagittarius => 8,
            Self::Capricorn => 9,
            Self::Aquarius => 10,
            Self::Pisces => 11,
        }
    }

    /// The sign for a 0-based index taken modulo 12 (so `12` wraps to
    /// `Aries`, `-1`-style callers should add 12 first).
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        Self::ALL[index % 12]
    }

    /// The English name, capitalized (e.g. `"Scorpio"`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Aries => "Aries",
            Self::Taurus => "Taurus",
            Self::Gemini => "Gemini",
            Self::Cancer => "Cancer",
            Self::Leo => "Leo",
            Self::Virgo => "Virgo",
            Self::Libra => "Libra",
            Self::Scorpio => "Scorpio",
            Self::Sagittarius => "Sagittarius",
            Self::Capricorn => "Capricorn",
            Self::Aquarius => "Aquarius",
            Self::Pisces => "Pisces",
        }
    }

    /// The Unicode astrological symbol for the sign (e.g. Scorpio =
    /// `"\u{264F}"`).
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Aries => "\u{2648}",
            Self::Taurus => "\u{2649}",
            Self::Gemini => "\u{264A}",
            Self::Cancer => "\u{264B}",
            Self::Leo => "\u{264C}",
            Self::Virgo => "\u{264D}",
            Self::Libra => "\u{264E}",
            Self::Scorpio => "\u{264F}",
            Self::Sagittarius => "\u{2650}",
            Self::Capricorn => "\u{2651}",
            Self::Aquarius => "\u{2652}",
            Self::Pisces => "\u{2653}",
        }
    }

    /// The classical element (triplicity) of the sign.
    #[must_use]
    pub const fn element(self) -> Element {
        // Element repeats every four signs, in the order Fire, Earth,
        // Air, Water starting from Aries.
        match self.index() % 4 {
            0 => Element::Fire,
            1 => Element::Earth,
            2 => Element::Air,
            _ => Element::Water,
        }
    }

    /// The modality (quadruplicity) of the sign.
    #[must_use]
    pub const fn modality(self) -> Modality {
        // Modality repeats every three signs, in the order Cardinal,
        // Fixed, Mutable starting from Aries.
        match self.index() % 3 {
            0 => Modality::Cardinal,
            1 => Modality::Fixed,
            _ => Modality::Mutable,
        }
    }
}

/// A longitude decomposed into its sign and the position within that
/// sign.
///
/// `degrees_in_sign` is always in `[0, 30)` degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignPosition {
    /// The sign the longitude falls in.
    pub sign: Sign,
    /// Position within the sign, degrees in `[0, 30)`.
    pub degrees_in_sign: f64,
}

impl SignPosition {
    /// Decomposes ecliptic longitude `lon_rad` into a sign plus the
    /// position within it.
    #[must_use]
    // `sign.index()` is in `0..=11`, exactly representable in `f64`.
    #[allow(clippy::cast_precision_loss)]
    pub fn of(lon_rad: f64) -> Self {
        let lon_deg = normalize_0_two_pi(lon_rad) * RAD2DEG;
        let sign = Sign::of(lon_rad);
        // `lon_deg` is in `[0, 360)`; subtracting the sign's base keeps
        // the remainder in `[0, 30)` (the boundary 30 only appears from
        // float rounding at a sign edge, which `Sign::of`'s own wrap
        // already places in the next sign).
        let degrees_in_sign = lon_deg - 30.0 * (sign.index() as f64);
        Self {
            sign,
            degrees_in_sign,
        }
    }

    /// The position within the sign as integer degrees, integer
    /// arcminutes, and floating-point arcseconds — the conventional
    /// `dd° mm' ss"` astrological notation.
    ///
    /// Degrees are in `0..=29`, arcminutes in `0..=59`, arcseconds in
    /// `[0, 60)`.
    #[must_use]
    // `deg` is in `0..=29` and `arcmin` in `0..=59` (both floored from a
    // non-negative `[0, 30)`-degree quantity), so neither cast truncates
    // meaningfully nor loses a sign.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn dms(self) -> (u32, u32, f64) {
        let total_arcsec = self.degrees_in_sign * 3600.0;
        let deg = libm::floor(total_arcsec / 3600.0);
        let rem_after_deg = total_arcsec - deg * 3600.0;
        let arcmin = libm::floor(rem_after_deg / 60.0);
        let arcsec = rem_after_deg - arcmin * 60.0;
        (deg as u32, arcmin as u32, arcsec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// Degrees to radians for readable test literals.
    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // loop index 0..12
    fn sign_of_each_30_degree_block() {
        for (i, sign) in Sign::ALL.iter().enumerate() {
            // Mid-sign sample must land in the expected sign.
            let mid = deg(30.0 * (i as f64) + 15.0);
            assert_eq!(Sign::of(mid), *sign, "mid of sign {i}");
        }
    }

    #[test]
    fn sign_boundaries_belong_to_the_upper_sign() {
        // Exactly 30 degrees is 0 Taurus, not 30 Aries.
        assert_eq!(Sign::of(deg(0.0)), Sign::Aries);
        assert_eq!(Sign::of(deg(30.0)), Sign::Taurus);
        assert_eq!(Sign::of(deg(330.0)), Sign::Pisces);
        // Just below a boundary stays in the lower sign.
        assert_eq!(Sign::of(deg(29.999_999)), Sign::Aries);
    }

    #[test]
    fn sign_of_wraps_negative_and_over_full_turn() {
        assert_eq!(Sign::of(deg(-15.0)), Sign::Pisces);
        assert_eq!(Sign::of(deg(360.0 + 45.0)), Sign::Taurus);
    }

    #[test]
    fn element_and_modality_groupings() {
        assert_eq!(Sign::Aries.element(), Element::Fire);
        assert_eq!(Sign::Cancer.element(), Element::Water);
        assert_eq!(Sign::Aquarius.element(), Element::Air);
        assert_eq!(Sign::Capricorn.element(), Element::Earth);
        assert_eq!(Sign::Aries.modality(), Modality::Cardinal);
        assert_eq!(Sign::Taurus.modality(), Modality::Fixed);
        assert_eq!(Sign::Gemini.modality(), Modality::Mutable);
        // Every element appears exactly three times, every modality four.
        for element in Element::ALL {
            let count = Sign::ALL.iter().filter(|s| s.element() == element).count();
            assert_eq!(count, 3, "element {}", element.name());
        }
        for modality in Modality::ALL {
            let count = Sign::ALL
                .iter()
                .filter(|s| s.modality() == modality)
                .count();
            assert_eq!(count, 4, "modality {}", modality.name());
        }
    }

    #[test]
    fn index_round_trips() {
        for sign in Sign::ALL {
            assert_eq!(Sign::from_index(sign.index()), sign);
        }
    }

    #[test]
    fn sign_position_dms_scorpio_example() {
        // 235.509050876 deg -> 25 deg 30' 32.583..." Scorpio
        let pos = SignPosition::of(deg(235.509_050_876));
        assert_eq!(pos.sign, Sign::Scorpio);
        let (d, m, s) = pos.dms();
        assert_eq!(d, 25);
        assert_eq!(m, 30);
        // 0.509050876 * 60 = 30.5430..., .5430 * 60 = 32.58...
        assert!((s - 32.583).abs() < 0.01, "arcsec {s} not near 32.583");
    }

    #[test]
    fn degrees_in_sign_always_in_range() {
        let mut x = 0.0;
        while x < 360.0 {
            let pos = SignPosition::of(deg(x));
            assert!(
                (0.0..30.0).contains(&pos.degrees_in_sign),
                "deg_in_sign {} out of range at {x}",
                pos.degrees_in_sign
            );
            x += 0.37;
        }
    }
}
