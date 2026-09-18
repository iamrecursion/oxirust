//! The property datatypes of [XSL 1.1 §5.11].
//!
//! Every datatype that a formatting property can take is represented here: the
//! scalar ones ([`Length`], [`Color`], [`Measure`]) and the *compound* ones
//! ([`Space`], [`Keep`], [`LengthRange`], [`LengthConditional`],
//! [`LengthBpIp`]) whose components are addressed in a stylesheet by appending
//! `.component` to the property name.
//!
//! [XSL 1.1 §5.11]: https://www.w3.org/TR/xsl11/#datatype

use core::fmt;

/// A unit of measure, as defined by [XSL 1.1 §5.9.13].
///
/// `em` is the only *relative* unit; it is resolved against the font size in
/// force for the formatting object being refined.
///
/// [XSL 1.1 §5.9.13]: https://www.w3.org/TR/xsl11/#d0e6212
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unit {
    /// Centimetres.
    Cm,
    /// Millimetres.
    Mm,
    /// Inches, defined as `2.54cm`.
    In,
    /// Points, defined as `1/72in`.
    Pt,
    /// Picas, defined as `12pt`.
    Pc,
    /// Pixels; device dependent, see [`Length::from_unit_with`].
    Px,
    /// Ems; relative to the current font size.
    Em,
}

impl Unit {
    /// Parses a unit name as written in a stylesheet.
    ///
    /// ```
    /// use oxixml_fo::property::Unit;
    ///
    /// assert_eq!(Unit::parse("mm"), Some(Unit::Mm));
    /// assert_eq!(Unit::parse("furlong"), None);
    /// ```
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "cm" => Self::Cm,
            "mm" => Self::Mm,
            "in" => Self::In,
            "pt" => Self::Pt,
            "pc" => Self::Pc,
            "px" => Self::Px,
            "em" => Self::Em,
            _ => return None,
        })
    }

    /// The unit name as written in a stylesheet.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cm => "cm",
            Self::Mm => "mm",
            Self::In => "in",
            Self::Pt => "pt",
            Self::Pc => "pc",
            Self::Px => "px",
            Self::Em => "em",
        }
    }

    /// Whether this unit is relative and therefore needs a context to resolve.
    #[must_use]
    pub const fn is_relative(self) -> bool {
        matches!(self, Self::Em)
    }
}

/// An absolute length, stored in points (`1pt = 1/72in`).
///
/// Lengths are computed values: relative units have already been resolved, so a
/// `Length` can be compared and added without a context.
///
/// ```
/// use oxixml_fo::property::{Length, Unit};
///
/// let inch = Length::from_unit(1.0, Unit::In);
/// assert!((inch.points() - 72.0).abs() < 1e-9);
/// assert_eq!(inch + Length::from_points(6.0), Length::from_points(78.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Length(f64);

impl Length {
    /// The zero length.
    pub const ZERO: Self = Self(0.0);

    /// Creates a length from a measurement in points.
    #[must_use]
    pub const fn from_points(points: f64) -> Self {
        Self(points)
    }

    /// Creates a length from `value` expressed in `unit`, assuming the default
    /// resolution of 72 pixels per inch and a font size of 12pt.
    ///
    /// Use [`Length::from_unit_with`] to supply the real values.
    #[must_use]
    pub fn from_unit(value: f64, unit: Unit) -> Self {
        Self::from_unit_with(value, unit, 12.0, 72.0)
    }

    /// Creates a length from `value` expressed in `unit`.
    ///
    /// `font_size` (in points) resolves the `em` unit and `pixels_per_inch`
    /// resolves the `px` unit; XSL 1.1 §5.9.13.1 leaves the latter to the
    /// implementation and explicitly permits treating `px` as a fixed
    /// conversion factor.
    #[must_use]
    pub fn from_unit_with(value: f64, unit: Unit, font_size: f64, pixels_per_inch: f64) -> Self {
        let points = match unit {
            Unit::Cm => value * 72.0 / 2.54,
            Unit::Mm => value * 72.0 / 25.4,
            Unit::In => value * 72.0,
            Unit::Pt => value,
            Unit::Pc => value * 12.0,
            Unit::Px => {
                if pixels_per_inch == 0.0 {
                    value
                } else {
                    value * 72.0 / pixels_per_inch
                }
            }
            Unit::Em => value * font_size,
        };
        Self(points)
    }

    /// The length in points.
    #[must_use]
    pub const fn points(self) -> f64 {
        self.0
    }

    /// The length in millipoints, the integral unit used by renderers.
    #[must_use]
    pub fn millipoints(self) -> i64 {
        (self.0 * 1000.0).round() as i64
    }

    /// Returns the smaller of two lengths.
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        if other.0 < self.0 { other } else { self }
    }

    /// Returns the larger of two lengths.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        if other.0 > self.0 { other } else { self }
    }
}

impl core::ops::Add for Length {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Length {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl core::ops::Neg for Length {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl core::ops::Mul<f64> for Length {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

impl core::iter::Sum for Length {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|l| l.0).sum())
    }
}

impl fmt::Display for Length {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}pt", self.0)
    }
}

/// A length that may still be relative to something the formatter has not
/// determined yet.
///
/// A percentage whose reference value is unknown at refinement time (the width
/// of a containing block, say) is *preserved* rather than guessed at; the
/// downstream formatter resolves it once the area geometry is known.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Measure {
    /// An absolute length.
    Length(Length),
    /// A percentage of the property-specific reference value, `100.0` meaning
    /// 100%.
    Percent(f64),
    /// The `auto` keyword.
    #[default]
    Auto,
}

impl Measure {
    /// A zero-length measure.
    pub const ZERO: Self = Self::Length(Length::ZERO);

    /// Returns the absolute length, if this measure is one.
    #[must_use]
    pub const fn length(self) -> Option<Length> {
        match self {
            Self::Length(length) => Some(length),
            _ => None,
        }
    }

    /// Resolves this measure against `reference`, returning `None` when the
    /// measure is `auto`.
    #[must_use]
    pub fn resolve(self, reference: Length) -> Option<Length> {
        match self {
            Self::Length(length) => Some(length),
            Self::Percent(pct) => Some(Length::from_points(reference.points() * pct / 100.0)),
            Self::Auto => None,
        }
    }
}

impl fmt::Display for Measure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(length) => write!(f, "{length}"),
            Self::Percent(pct) => write!(f, "{pct}%"),
            Self::Auto => f.write_str("auto"),
        }
    }
}

/// The `.conditionality` component of a `<space>` or `<length-conditional>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Conditionality {
    /// The space or border is discarded at a leading or trailing edge.
    #[default]
    Discard,
    /// The space or border is retained at a leading or trailing edge.
    Retain,
}

impl Conditionality {
    /// Parses the keyword form.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "discard" => Some(Self::Discard),
            "retain" => Some(Self::Retain),
            _ => None,
        }
    }

    /// The keyword as written in a stylesheet.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Discard => "discard",
            Self::Retain => "retain",
        }
    }
}

/// The `.precedence` component of a `<space>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precedence {
    /// The `force` keyword: this space is never discarded by the space
    /// resolution of XSL 1.1 §4.3.
    Force,
    /// A numeric precedence; the initial value is `0`.
    Value(i32),
}

impl Precedence {
    /// The initial value, `0`.
    pub const INITIAL: Self = Self::Value(0);
}

impl Default for Precedence {
    fn default() -> Self {
        Self::INITIAL
    }
}

/// The `<space>` compound datatype of XSL 1.1 §5.11.
///
/// Its five components are addressed in a stylesheet as `space-before.minimum`,
/// `space-before.optimum`, and so on.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Space {
    /// The `.minimum` component.
    pub minimum: Measure,
    /// The `.optimum` component.
    pub optimum: Measure,
    /// The `.maximum` component.
    pub maximum: Measure,
    /// The `.precedence` component.
    pub precedence: Precedence,
    /// The `.conditionality` component.
    pub conditionality: Conditionality,
}

impl Space {
    /// The initial value of the `space-*` properties: all lengths zero,
    /// precedence `0`, conditionality `discard`.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            minimum: Measure::ZERO,
            optimum: Measure::ZERO,
            maximum: Measure::ZERO,
            precedence: Precedence::INITIAL,
            conditionality: Conditionality::Discard,
        }
    }

    /// Builds the short form: every `<length>` component takes `measure` and the
    /// remaining components keep their initial values (XSL 1.1 §5.11).
    #[must_use]
    pub fn uniform(measure: Measure) -> Self {
        Self {
            minimum: measure,
            optimum: measure,
            maximum: measure,
            precedence: Precedence::INITIAL,
            conditionality: Conditionality::Discard,
        }
    }

    /// Clamps `minimum` and `maximum` around `optimum`, as XSL 1.1 §5.11
    /// requires when the specified range is inconsistent.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        if let (Measure::Length(min), Measure::Length(opt)) = (self.minimum, self.optimum)
            && min > opt
        {
            self.minimum = Measure::Length(opt);
        }
        if let (Measure::Length(max), Measure::Length(opt)) = (self.maximum, self.optimum)
            && max < opt
        {
            self.maximum = Measure::Length(opt);
        }
        self
    }
}

/// The `<length-range>` compound datatype of XSL 1.1 §5.11.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LengthRange {
    /// The `.minimum` component.
    pub minimum: Measure,
    /// The `.optimum` component.
    pub optimum: Measure,
    /// The `.maximum` component.
    pub maximum: Measure,
}

impl LengthRange {
    /// A range whose three components are all `auto`.
    #[must_use]
    pub fn auto() -> Self {
        Self::default()
    }

    /// Builds the short form: every component takes `measure`.
    #[must_use]
    pub fn uniform(measure: Measure) -> Self {
        Self {
            minimum: measure,
            optimum: measure,
            maximum: measure,
        }
    }

    /// Clamps `minimum` and `maximum` around `optimum`.
    #[must_use]
    pub fn normalized(mut self) -> Self {
        if let (Measure::Length(min), Measure::Length(opt)) = (self.minimum, self.optimum)
            && min > opt
        {
            self.minimum = Measure::Length(opt);
        }
        if let (Measure::Length(max), Measure::Length(opt)) = (self.maximum, self.optimum)
            && max < opt
        {
            self.maximum = Measure::Length(opt);
        }
        self
    }
}

/// The `<length-conditional>` compound datatype of XSL 1.1 §5.11.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LengthConditional {
    /// The `.length` component.
    pub length: Measure,
    /// The `.conditionality` component.
    pub conditionality: Conditionality,
}

impl LengthConditional {
    /// Builds the short form: `.length` takes `measure`, `.conditionality`
    /// keeps its initial value of `discard`.
    #[must_use]
    pub fn new(measure: Measure) -> Self {
        Self {
            length: measure,
            conditionality: Conditionality::Discard,
        }
    }
}

/// The `<length-bp-ip-direction>` compound datatype of XSL 1.1 §5.11.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LengthBpIp {
    /// The `.block-progression-direction` component.
    pub block_progression_direction: Measure,
    /// The `.inline-progression-direction` component.
    pub inline_progression_direction: Measure,
}

impl LengthBpIp {
    /// Builds the short form: both components take `measure`.
    #[must_use]
    pub fn uniform(measure: Measure) -> Self {
        Self {
            block_progression_direction: measure,
            inline_progression_direction: measure,
        }
    }
}

/// The value of one component of the `<keep>` compound datatype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepValue {
    /// `auto`: no keep constraint.
    #[default]
    Auto,
    /// `always`: an infinitely strong keep constraint.
    Always,
    /// A numeric keep strength.
    Strength(i32),
}

impl KeepValue {
    /// Parses a keep component value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "always" => Some(Self::Always),
            other => other.parse::<i32>().ok().map(Self::Strength),
        }
    }
}

/// The `<keep>` compound datatype of XSL 1.1 §5.11.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Keep {
    /// The `.within-line` component.
    pub within_line: KeepValue,
    /// The `.within-column` component.
    pub within_column: KeepValue,
    /// The `.within-page` component.
    pub within_page: KeepValue,
}

impl Keep {
    /// Builds the short form: every component takes `value` (XSL 1.1 §5.11).
    #[must_use]
    pub fn uniform(value: KeepValue) -> Self {
        Self {
            within_line: value,
            within_column: value,
            within_page: value,
        }
    }
}

/// The `<shape>` datatype, `rect(<top> <right> <bottom> <left>)`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Shape {
    /// Offset from the top edge of the content rectangle.
    pub top: Measure,
    /// Offset from the right edge of the content rectangle.
    pub right: Measure,
    /// Offset from the bottom edge of the content rectangle.
    pub bottom: Measure,
    /// Offset from the left edge of the content rectangle.
    pub left: Measure,
}

/// A colour, as produced by the `<color>` datatype and the colour functions of
/// XSL 1.1 §5.10.2.
#[derive(Debug, Clone, PartialEq)]
pub enum Color {
    /// An sRGB colour.
    Rgb {
        /// Red component.
        red: u8,
        /// Green component.
        green: u8,
        /// Blue component.
        blue: u8,
    },
    /// An ICC colour with its sRGB fallback, from `rgb-icc()`.
    RgbIcc {
        /// Red component of the sRGB fallback.
        red: u8,
        /// Green component of the sRGB fallback.
        green: u8,
        /// Blue component of the sRGB fallback.
        blue: u8,
        /// Name of the colour profile declared with `fo:color-profile`.
        profile: String,
        /// Profile-specific colour values.
        components: Vec<f64>,
    },
    /// A system colour, from `system-color()`.
    System(String),
    /// The `transparent` keyword.
    Transparent,
}

impl Color {
    /// The `black` colour.
    pub const BLACK: Self = Self::Rgb {
        red: 0,
        green: 0,
        blue: 0,
    };

    /// Builds an sRGB colour from three channel values.
    #[must_use]
    pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
        Self::Rgb { red, green, blue }
    }

    /// Resolves one of the sixteen colour keywords listed in XSL 1.1 §5.11.
    ///
    /// ```
    /// use oxixml_fo::property::Color;
    ///
    /// assert_eq!(Color::from_keyword("navy"), Some(Color::rgb(0x00, 0x00, 0x80)));
    /// assert_eq!(Color::from_keyword("chartreuse"), None);
    /// ```
    #[must_use]
    pub fn from_keyword(name: &str) -> Option<Self> {
        let (red, green, blue) = match name {
            "aqua" => (0x00, 0xff, 0xff),
            "black" => (0x00, 0x00, 0x00),
            "blue" => (0x00, 0x00, 0xff),
            "fuchsia" => (0xff, 0x00, 0xff),
            "gray" => (0x80, 0x80, 0x80),
            "green" => (0x00, 0x80, 0x00),
            "lime" => (0x00, 0xff, 0x00),
            "maroon" => (0x80, 0x00, 0x00),
            "navy" => (0x00, 0x00, 0x80),
            "olive" => (0x80, 0x80, 0x00),
            "purple" => (0x80, 0x00, 0x80),
            "red" => (0xff, 0x00, 0x00),
            "silver" => (0xc0, 0xc0, 0xc0),
            "teal" => (0x00, 0x80, 0x80),
            "white" => (0xff, 0xff, 0xff),
            "yellow" => (0xff, 0xff, 0x00),
            "transparent" => return Some(Self::Transparent),
            _ => return None,
        };
        Some(Self::Rgb { red, green, blue })
    }

    /// Parses the `#rgb` and `#rrggbb` hexadecimal notations.
    ///
    /// ```
    /// use oxixml_fo::property::Color;
    ///
    /// assert_eq!(Color::from_hex("#0f8"), Some(Color::rgb(0x00, 0xff, 0x88)));
    /// assert_eq!(Color::from_hex("#00ff88"), Some(Color::rgb(0x00, 0xff, 0x88)));
    /// ```
    #[must_use]
    pub fn from_hex(text: &str) -> Option<Self> {
        let digits = text.strip_prefix('#')?;
        if !digits.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let nibble = |c: char| c.to_digit(16).map(|d| d as u8);
        let mut chars = digits.chars();
        match digits.len() {
            3 => {
                let red = nibble(chars.next()?)?;
                let green = nibble(chars.next()?)?;
                let blue = nibble(chars.next()?)?;
                Some(Self::Rgb {
                    red: red * 17,
                    green: green * 17,
                    blue: blue * 17,
                })
            }
            6 => {
                let mut byte = || -> Option<u8> {
                    let high = nibble(chars.next()?)?;
                    let low = nibble(chars.next()?)?;
                    Some(high * 16 + low)
                };
                let red = byte()?;
                let green = byte()?;
                let blue = byte()?;
                Some(Self::Rgb { red, green, blue })
            }
            _ => None,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rgb { red, green, blue }
            | Self::RgbIcc {
                red, green, blue, ..
            } => write!(f, "#{red:02x}{green:02x}{blue:02x}"),
            Self::System(name) => write!(f, "system-color({name})"),
            Self::Transparent => f.write_str("transparent"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_conversions_match_the_recommendation() {
        assert!((Length::from_unit(1.0, Unit::In).points() - 72.0).abs() < 1e-9);
        assert!((Length::from_unit(2.54, Unit::Cm).points() - 72.0).abs() < 1e-9);
        assert!((Length::from_unit(25.4, Unit::Mm).points() - 72.0).abs() < 1e-9);
        assert!((Length::from_unit(1.0, Unit::Pc).points() - 12.0).abs() < 1e-9);
        assert!((Length::from_unit_with(2.0, Unit::Em, 10.0, 72.0).points() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn space_normalization_clamps_to_optimum() {
        let space = Space {
            minimum: Measure::Length(Length::from_points(10.0)),
            optimum: Measure::Length(Length::from_points(5.0)),
            maximum: Measure::Length(Length::from_points(1.0)),
            ..Space::zero()
        }
        .normalized();
        assert_eq!(space.minimum, Measure::Length(Length::from_points(5.0)));
        assert_eq!(space.maximum, Measure::Length(Length::from_points(5.0)));
    }

    #[test]
    fn hex_colors_round_trip() {
        assert_eq!(Color::from_hex("#fff"), Some(Color::rgb(255, 255, 255)));
        assert_eq!(Color::from_hex("#zzz"), None);
        assert_eq!(Color::from_hex("fff"), None);
    }
}
