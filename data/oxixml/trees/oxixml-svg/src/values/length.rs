//! `<length>` and `<angle>` values.
//!
//! [`Length`] models SVG's `<length>` microsyntax — a number with an
//! optional unit suffix (`10`, `12.5px`, `50%`, `2em`, …) — and [`Angle`]
//! models `<angle>` (`45`, `45deg`, `50grad`, `1.5rad`, `0.25turn`). Neither
//! type resolves its unit on its own: [`Length::resolve`] turns a `Length`
//! into an absolute number of user units given a [`LengthContext`] (the
//! font size, root font size, and viewport dimensions that the relative and
//! percentage units are defined against) and a [`PercentBasis`] (which
//! reference dimension a `%` value uses, since that differs by attribute or
//! property).
//!
//! This module also parses the list microsyntaxes built out of `<length>`
//! and plain numbers: [`parse_length_list`] (`stroke-width`-style lists),
//! [`parse_dash_array`] (`stroke-dasharray`), and
//! [`parse_number_optional_number`] (the `stdDeviation`/`baseFrequency`/
//! `order`/`radius`/`kernelUnitLength` idiom).
//!
//! # Case and serialization
//!
//! Unit suffixes are parsed case-insensitively (`10PX`, `10Px`, and `10px`
//! all parse to the same [`Length`]), matching the general CSS
//! `<dimension>` token rule, even though the historical SVG 1.1
//! presentation-attribute grammar only ever spells them lowercase. `Length`
//! and `Angle` always *serialize* the canonical lowercase spelling through
//! their `Display` implementations (`Length::new(10.0,
//! LengthUnit::Q).to_string()` is `"10q"`, never `"10Q"`), so
//! round-tripping through parse then serialize normalizes case rather than
//! preserving the input's casing.
//!
//! # Examples
//!
//! ```
//! use oxixml_svg::values::length::{Length, LengthContext, LengthUnit, PercentBasis};
//!
//! let length = Length::parse("12.5px").expect("valid length");
//! assert_eq!(length, Length::new(12.5, LengthUnit::Px));
//! assert_eq!(length.to_string(), "12.5px");
//!
//! let half = Length::parse("50%").expect("valid percentage");
//! let ctx = LengthContext::default();
//! assert_eq!(half.resolve(&ctx, PercentBasis::Width), 50.0);
//! ```

use std::fmt;

use crate::error::ValueError;
use crate::values::lex::Cursor;

/// The unit suffix of an SVG `<length>` value.
///
/// Covers the absolute, font-relative, viewport-relative, and percentage
/// units SVG 1.1 and SVG 2 / CSS Values and Units Level 4 define for
/// geometry attributes. Parsed case-insensitively; always serialized in
/// lowercase (see the [module documentation](self) for why).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LengthUnit {
    /// No unit suffix: a plain number in the user coordinate system (a
    /// "user unit").
    #[default]
    None,
    /// Pixels (`px`), CSS's canonical absolute unit; equal to a user unit
    /// unless an outer viewport transform says otherwise.
    Px,
    /// Font-relative: multiples of the current font size (`em`).
    Em,
    /// Font-relative: multiples of the font's x-height (`ex`).
    Ex,
    /// Font-relative: multiples of the advance width of the font's `0`
    /// glyph (`ch`).
    Ch,
    /// Font-relative: multiples of the root element's font size (`rem`).
    Rem,
    /// Viewport-relative: percent of the viewport width (`vw`).
    Vw,
    /// Viewport-relative: percent of the viewport height (`vh`).
    Vh,
    /// Viewport-relative: percent of the smaller viewport dimension
    /// (`vmin`).
    Vmin,
    /// Viewport-relative: percent of the larger viewport dimension
    /// (`vmax`).
    Vmax,
    /// Absolute: quarter-millimeters (`Q`), `1Q = 1/4mm`.
    Q,
    /// Absolute: inches (`in`), `1in = 96px`.
    In,
    /// Absolute: centimeters (`cm`).
    Cm,
    /// Absolute: millimeters (`mm`).
    Mm,
    /// Absolute: points (`pt`), `1pt = 1/72in`.
    Pt,
    /// Absolute: picas (`pc`), `1pc = 1/6in = 16px`.
    Pc,
    /// A percentage (`%`) of an applicable reference value; see
    /// [`PercentBasis`].
    Percent,
}

impl LengthUnit {
    /// The canonical lowercase unit suffix text: `""` for [`Self::None`],
    /// `"%"` for [`Self::Percent`], otherwise the unit name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Px => "px",
            Self::Em => "em",
            Self::Ex => "ex",
            Self::Ch => "ch",
            Self::Rem => "rem",
            Self::Vw => "vw",
            Self::Vh => "vh",
            Self::Vmin => "vmin",
            Self::Vmax => "vmax",
            Self::Q => "q",
            Self::In => "in",
            Self::Cm => "cm",
            Self::Mm => "mm",
            Self::Pt => "pt",
            Self::Pc => "pc",
            Self::Percent => "%",
        }
    }

    /// Looks up the unit a keyword (already scanned by
    /// [`Cursor::keyword`]) names, ignoring ASCII case. Returns `None` for
    /// any string that is not one of the sixteen named-suffix units (the
    /// `%` sign is handled separately by [`Length::scan`], not through
    /// this lookup).
    fn from_keyword(word: &str) -> Option<Self> {
        if !word.is_ascii() || word.len() > 4 {
            return None;
        }
        let mut buf = [0u8; 4];
        for (dst, src) in buf.iter_mut().zip(word.bytes()) {
            *dst = src.to_ascii_lowercase();
        }
        match &buf[..word.len()] {
            b"px" => Some(Self::Px),
            b"em" => Some(Self::Em),
            b"ex" => Some(Self::Ex),
            b"ch" => Some(Self::Ch),
            b"rem" => Some(Self::Rem),
            b"vw" => Some(Self::Vw),
            b"vh" => Some(Self::Vh),
            b"vmin" => Some(Self::Vmin),
            b"vmax" => Some(Self::Vmax),
            b"q" => Some(Self::Q),
            b"in" => Some(Self::In),
            b"cm" => Some(Self::Cm),
            b"mm" => Some(Self::Mm),
            b"pt" => Some(Self::Pt),
            b"pc" => Some(Self::Pc),
            _ => None,
        }
    }
}

impl fmt::Display for LengthUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An SVG `<length>`: a number together with the unit it was written in,
/// unresolved against any particular font or viewport context.
///
/// See the [module documentation](self) for parsing/serialization case
/// rules and [`Length::resolve`] for turning a `Length` into an absolute
/// number of user units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Length {
    /// The numeric magnitude, in the units named by [`Self::unit`].
    pub value: f64,
    /// The unit suffix that followed the number, or [`LengthUnit::None`]
    /// when the value carried no suffix.
    pub unit: LengthUnit,
}

impl Length {
    /// Creates a length from a value and a unit.
    #[must_use]
    pub const fn new(value: f64, unit: LengthUnit) -> Self {
        Self { value, unit }
    }

    /// Parses a complete attribute value as one `<length>`, allowing
    /// surrounding white space.
    ///
    /// # Errors
    ///
    /// Returns [`ValueError::Empty`] for an empty or white-space-only
    /// value, [`ValueError::InvalidNumber`] or [`ValueError::UnexpectedEnd`]
    /// for a malformed number, [`ValueError::UnknownKeyword`] for a suffix
    /// that is not one of the units [`LengthUnit`] names, and
    /// [`ValueError::TrailingData`] for well-formed content followed by
    /// more text.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let mut cursor = Cursor::new(input);
        cursor.skip_wsp();
        if cursor.is_at_end() {
            return Err(ValueError::Empty);
        }
        let length = Self::scan(&mut cursor)?;
        cursor.done()?;
        Ok(length)
    }

    /// Scans one `<length>` starting at the cursor's current position,
    /// leaving the cursor positioned immediately after the optional unit
    /// suffix.
    ///
    /// This is the primitive [`parse_length_list`], [`parse_dash_array`],
    /// and other list/composite parsers drive to read one element without
    /// requiring it to be the entire input.
    ///
    /// # Errors
    ///
    /// As [`Length::parse`], except that trailing content after the length
    /// is left for the caller rather than rejected.
    pub fn scan(cursor: &mut Cursor<'_>) -> Result<Self, ValueError> {
        let value = cursor.number()?;
        if cursor.peek() == Some('%') {
            cursor.bump();
            return Ok(Self {
                value,
                unit: LengthUnit::Percent,
            });
        }
        let start = cursor.pos();
        let word = cursor.keyword();
        if word.is_empty() {
            return Ok(Self {
                value,
                unit: LengthUnit::None,
            });
        }
        match LengthUnit::from_keyword(word) {
            Some(unit) => Ok(Self { value, unit }),
            None => Err(ValueError::UnknownKeyword {
                offset: start,
                keyword: word.into(),
            }),
        }
    }

    /// Resolves this length to an absolute number of user units.
    ///
    /// [`LengthUnit::None`] and [`LengthUnit::Px`] pass `value` through
    /// unchanged. Font-relative units multiply by `ctx.font_size` (`em`) or
    /// `ctx.root_font_size` (`rem`); `ex`/`ch` use `ctx.x_height`/
    /// `ctx.ch_advance` when set, falling back to half the font size
    /// (`0.5em`) per CSS when not. Viewport-relative units (`vw`, `vh`,
    /// `vmin`, `vmax`) and [`LengthUnit::Percent`] resolve against
    /// `ctx.viewport_width`/`ctx.viewport_height`, with `Percent` further
    /// choosing which dimension (or their normalized diagonal) via `basis`.
    /// Absolute units (`in`, `cm`, `mm`, `pt`, `pc`, `Q`) convert through
    /// the fixed CSS pixel ratios (`1in = 96px`, etc.), independent of
    /// `ctx`.
    #[must_use]
    pub fn resolve(&self, ctx: &LengthContext, basis: PercentBasis) -> f64 {
        match self.unit {
            LengthUnit::None | LengthUnit::Px => self.value,
            LengthUnit::Em => self.value * ctx.font_size,
            LengthUnit::Ex => self.value * ctx.x_height.unwrap_or(ctx.font_size * 0.5),
            LengthUnit::Ch => self.value * ctx.ch_advance.unwrap_or(ctx.font_size * 0.5),
            LengthUnit::Rem => self.value * ctx.root_font_size,
            LengthUnit::Vw => self.value * ctx.viewport_width / 100.0,
            LengthUnit::Vh => self.value * ctx.viewport_height / 100.0,
            LengthUnit::Vmin => self.value * ctx.viewport_width.min(ctx.viewport_height) / 100.0,
            LengthUnit::Vmax => self.value * ctx.viewport_width.max(ctx.viewport_height) / 100.0,
            // The `96.0 / divisor` ratio is computed first and then
            // applied as a single multiplication (rather than the
            // left-to-right `value * 96.0 / divisor`) because that
            // evaluation order keeps round-number conversions exact: e.g.
            // `25.4 * 96.0 / 25.4` rounds to `95.99999999999999` in f64
            // arithmetic, while `25.4 * (96.0 / 25.4)` is exactly `96.0`.
            LengthUnit::Q => self.value * (96.0 / 101.6),
            LengthUnit::In => self.value * 96.0,
            LengthUnit::Cm => self.value * (96.0 / 2.54),
            LengthUnit::Mm => self.value * (96.0 / 25.4),
            LengthUnit::Pt => self.value * (96.0 / 72.0),
            LengthUnit::Pc => self.value * 16.0,
            LengthUnit::Percent => {
                let reference = match basis {
                    PercentBasis::Width => ctx.viewport_width,
                    PercentBasis::Height => ctx.viewport_height,
                    PercentBasis::Diagonal => {
                        (ctx.viewport_width.powi(2) + ctx.viewport_height.powi(2)).sqrt()
                            / std::f64::consts::SQRT_2
                    }
                };
                self.value / 100.0 * reference
            }
        }
    }
}

impl fmt::Display for Length {
    /// Formats the length as `<number><unit>`, e.g. `12.5px` or `50%`.
    ///
    /// Round-trips for finite values: `Length::parse(&length.to_string())`
    /// reproduces the original value and unit. A non-finite magnitude
    /// (reachable by parsing an overflowing literal such as `"1e999"`,
    /// which the shared number scanner saturates to an infinity) formats
    /// as `inf`/`-inf`/`NaN`, none of which is valid `<number>` syntax, so
    /// non-finite lengths are outside this round-trip guarantee.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.value, self.unit)
    }
}

/// The unit suffix of an SVG `<angle>` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AngleUnit {
    /// No unit suffix: a bare number. SVG presentation-attribute contexts
    /// (such as the `rotate()`/`skewX()`/`skewY()` transform functions)
    /// interpret this as degrees.
    #[default]
    None,
    /// Degrees (`deg`), 360 per full turn.
    Deg,
    /// Gradians (`grad`), 400 per full turn.
    Grad,
    /// Radians (`rad`), 2π per full turn.
    Rad,
    /// Turns (`turn`), one full rotation.
    Turn,
}

impl AngleUnit {
    /// The canonical lowercase unit suffix text: `""` for [`Self::None`],
    /// otherwise the unit name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Deg => "deg",
            Self::Grad => "grad",
            Self::Rad => "rad",
            Self::Turn => "turn",
        }
    }

    /// Looks up the unit a keyword (already scanned by
    /// [`Cursor::keyword`]) names, ignoring ASCII case.
    fn from_keyword(word: &str) -> Option<Self> {
        if !word.is_ascii() || word.len() > 4 {
            return None;
        }
        let mut buf = [0u8; 4];
        for (dst, src) in buf.iter_mut().zip(word.bytes()) {
            *dst = src.to_ascii_lowercase();
        }
        match &buf[..word.len()] {
            b"deg" => Some(Self::Deg),
            b"grad" => Some(Self::Grad),
            b"rad" => Some(Self::Rad),
            b"turn" => Some(Self::Turn),
            _ => None,
        }
    }
}

impl fmt::Display for AngleUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An SVG `<angle>`: a number together with the unit it was written in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    /// The numeric magnitude, in the units named by [`Self::unit`].
    pub value: f64,
    /// The unit suffix that followed the number, or [`AngleUnit::None`]
    /// when the value carried no suffix (interpreted as degrees).
    pub unit: AngleUnit,
}

impl Angle {
    /// Creates an angle from a value and a unit.
    #[must_use]
    pub const fn new(value: f64, unit: AngleUnit) -> Self {
        Self { value, unit }
    }

    /// Parses a complete attribute value as one `<angle>`, allowing
    /// surrounding white space.
    ///
    /// # Errors
    ///
    /// As [`Length::parse`]: [`ValueError::Empty`] for empty input,
    /// [`ValueError::UnknownKeyword`] for an unrecognized suffix,
    /// [`ValueError::TrailingData`] for extra content, and the number
    /// errors for a malformed magnitude.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let mut cursor = Cursor::new(input);
        cursor.skip_wsp();
        if cursor.is_at_end() {
            return Err(ValueError::Empty);
        }
        let angle = Self::scan(&mut cursor)?;
        cursor.done()?;
        Ok(angle)
    }

    /// Scans one `<angle>` starting at the cursor's current position,
    /// leaving the cursor positioned immediately after the optional unit
    /// suffix. The [`Length::scan`] counterpart for embedding an angle
    /// inside a larger microsyntax.
    ///
    /// # Errors
    ///
    /// As [`Angle::parse`], except that trailing content after the angle is
    /// left for the caller rather than rejected.
    pub fn scan(cursor: &mut Cursor<'_>) -> Result<Self, ValueError> {
        let value = cursor.number()?;
        let start = cursor.pos();
        let word = cursor.keyword();
        if word.is_empty() {
            return Ok(Self {
                value,
                unit: AngleUnit::None,
            });
        }
        match AngleUnit::from_keyword(word) {
            Some(unit) => Ok(Self { value, unit }),
            None => Err(ValueError::UnknownKeyword {
                offset: start,
                keyword: word.into(),
            }),
        }
    }

    /// Converts this angle to degrees.
    #[must_use]
    pub fn to_degrees(&self) -> f64 {
        match self.unit {
            AngleUnit::None | AngleUnit::Deg => self.value,
            AngleUnit::Grad => self.value * 360.0 / 400.0,
            AngleUnit::Rad => self.value.to_degrees(),
            AngleUnit::Turn => self.value * 360.0,
        }
    }

    /// Converts this angle to radians.
    #[must_use]
    pub fn to_radians(&self) -> f64 {
        match self.unit {
            AngleUnit::Rad => self.value,
            AngleUnit::None | AngleUnit::Deg => self.value.to_radians(),
            AngleUnit::Grad => (self.value * 360.0 / 400.0).to_radians(),
            AngleUnit::Turn => (self.value * 360.0).to_radians(),
        }
    }
}

impl fmt::Display for Angle {
    /// Formats the angle as `<number><unit>`, e.g. `45deg` or `0.25turn`,
    /// with the same finite-value-only round-trip guarantee as
    /// [`Length`]'s `Display` implementation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.value, self.unit)
    }
}

/// The reference value a [`Length`] percentage resolves against.
///
/// SVG resolves geometric percentages against the width, height, or
/// diagonal of the current viewport, depending on which attribute or
/// property carries the percentage: horizontal quantities (`x`, `width`,
/// …) use [`Self::Width`], vertical quantities (`y`, `height`, …) use
/// [`Self::Height`], and quantities that are not inherently horizontal or
/// vertical (`stroke-width`, `r`, …) use [`Self::Diagonal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PercentBasis {
    /// Resolve against the viewport width.
    Width,
    /// Resolve against the viewport height.
    Height,
    /// Resolve against `sqrt(width^2 + height^2) / sqrt(2)`, the
    /// normalized diagonal SVG defines for percentages that are not
    /// inherently horizontal or vertical.
    Diagonal,
}

/// The font and viewport context a [`Length`] percentage or relative unit
/// resolves against.
///
/// Construct with [`LengthContext::default`] for the CSS initial values (a
/// 16px font and a 100×100 user-unit viewport) and adjust individual
/// fields directly, or chain the `with_*` builder methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthContext {
    /// The current font size, in user units, that `em` and (absent a more
    /// specific override) `ex`/`ch` resolve against.
    pub font_size: f64,
    /// The font's x-height, in user units, that `ex` resolves against.
    /// `None` falls back to half the font size (`0.5em`), the CSS default
    /// for fonts whose actual x-height is not known.
    pub x_height: Option<f64>,
    /// The font's advance width of the `0` glyph, in user units, that `ch`
    /// resolves against. `None` falls back to the same half-the-font-size
    /// default as `ex`.
    pub ch_advance: Option<f64>,
    /// The root element's font size, in user units, that `rem` resolves
    /// against.
    pub root_font_size: f64,
    /// The viewport width, in user units, that `vw`, `vmin`/`vmax`, and
    /// [`PercentBasis::Width`] resolve against.
    pub viewport_width: f64,
    /// The viewport height, in user units, that `vh`, `vmin`/`vmax`, and
    /// [`PercentBasis::Height`] resolve against.
    pub viewport_height: f64,
}

impl Default for LengthContext {
    /// The CSS initial values: a 16px font (used for both the current and
    /// root font size) and a 100×100 user-unit viewport, with `ex`/`ch`
    /// left to their half-em fallback.
    fn default() -> Self {
        Self {
            font_size: 16.0,
            x_height: None,
            ch_advance: None,
            root_font_size: 16.0,
            viewport_width: 100.0,
            viewport_height: 100.0,
        }
    }
}

impl LengthContext {
    /// Sets the font size, in user units.
    #[must_use]
    pub const fn with_font_size(mut self, font_size: f64) -> Self {
        self.font_size = font_size;
        self
    }

    /// Sets the font's x-height, in user units, overriding the half-em
    /// fallback `ex` otherwise resolves against.
    #[must_use]
    pub const fn with_x_height(mut self, x_height: f64) -> Self {
        self.x_height = Some(x_height);
        self
    }

    /// Sets the font's `ch` advance width, in user units, overriding the
    /// half-em fallback `ch` otherwise resolves against.
    #[must_use]
    pub const fn with_ch_advance(mut self, ch_advance: f64) -> Self {
        self.ch_advance = Some(ch_advance);
        self
    }

    /// Sets the root element's font size, in user units, that `rem`
    /// resolves against.
    #[must_use]
    pub const fn with_root_font_size(mut self, root_font_size: f64) -> Self {
        self.root_font_size = root_font_size;
        self
    }

    /// Sets the viewport dimensions, in user units, that viewport-relative
    /// units and percentages resolve against.
    #[must_use]
    pub const fn with_viewport(mut self, width: f64, height: f64) -> Self {
        self.viewport_width = width;
        self.viewport_height = height;
        self
    }
}

/// Skips one `comma-wsp` separator like [`Cursor::skip_comma_wsp`], but
/// additionally reports whether the separator included a literal comma (as
/// opposed to bare white space). Returns `(separator_consumed,
/// comma_seen)`. Used where a *dangling* comma (one with nothing valid
/// after it) must be told apart from unremarkable trailing white space.
fn skip_separator(cursor: &mut Cursor<'_>) -> (bool, bool) {
    let start = cursor.pos();
    cursor.skip_wsp();
    let had_comma = cursor.eat(',');
    if had_comma {
        cursor.skip_wsp();
    }
    (cursor.pos() != start, had_comma)
}

/// Parses a `comma-wsp`-separated list of `<length>` values, such as a
/// `stroke-width`-family attribute that accepts more than one value.
///
/// An empty (or white-space-only) value yields an empty list; callers whose
/// grammar requires at least one value must check for that themselves. A
/// separator is optional immediately before a value that begins with `+`,
/// `-`, or `.` — the same "the next value's own leading character can act
/// as the separator" rule
/// [`crate::values::lex::parse_number_list`] uses for plain numbers — and,
/// like that function, **a single trailing separator with nothing after it
/// is accepted silently** (`parse_length_list("1px,")` is `Ok(vec![Length::new(1.0,
/// LengthUnit::Px)])`), matching the shared list grammar's established
/// leniency rather than treating a dangling separator as an error. See
/// [`parse_number_optional_number`] for a list-shaped production that
/// instead rejects a dangling separator, and the module's points sibling
/// (`crate::values::points::parse_points`) for a grammar with a real
/// error-recovery rule that intentionally does not share this leniency.
///
/// # Errors
///
/// [`ValueError::UnknownKeyword`]/number errors from the offending
/// [`Length`] element, or [`ValueError::TrailingData`] when a value ends
/// with **no** separator at all before unrecognizable content. When at
/// least some white space (or a comma) *was* consumed but what follows
/// still is not a valid `<length>`, that failed attempt's own error
/// (typically [`ValueError::InvalidNumber`]) is reported instead, since a
/// consumed separator commits the parser to expecting another element.
pub fn parse_length_list(input: &str) -> Result<Vec<Length>, ValueError> {
    let mut cursor = Cursor::new(input);
    let mut lengths = Vec::new();
    cursor.skip_wsp();
    if cursor.is_at_end() {
        return Ok(lengths);
    }
    loop {
        lengths.push(Length::scan(&mut cursor)?);
        let separated = cursor.skip_comma_wsp();
        if cursor.is_at_end() {
            return Ok(lengths);
        }
        if !separated && !matches!(cursor.peek(), Some('+' | '-' | '.' | '0'..='9')) {
            return Err(ValueError::TrailingData {
                offset: cursor.pos(),
            });
        }
    }
}

/// Parses a `stroke-dasharray` dash list: one or more `<length>` or
/// `<percentage>` values, none of which may be negative.
///
/// This parses only the `dashes` production (a negative-rejecting
/// [`parse_length_list`]); the full `stroke-dasharray` grammar is `"none" |
/// dashes`, and the `none` keyword alternative is the caller's
/// responsibility to check for **before** calling this function —
/// `parse_dash_array("none")` fails with [`ValueError::InvalidNumber`]
/// rather than being special-cased here. Follows `parse_length_list`'s
/// separator leniency: a single trailing separator with nothing after it is
/// accepted.
///
/// # Errors
///
/// As [`parse_length_list`], plus [`ValueError::OutOfRange`] (pointing at
/// the offending value's own offset) for a negative dash length.
pub fn parse_dash_array(input: &str) -> Result<Vec<Length>, ValueError> {
    let mut cursor = Cursor::new(input);
    let mut lengths = Vec::new();
    cursor.skip_wsp();
    if cursor.is_at_end() {
        return Ok(lengths);
    }
    loop {
        let start = cursor.pos();
        let length = Length::scan(&mut cursor)?;
        if length.value < 0.0 {
            return Err(ValueError::OutOfRange {
                offset: start,
                what: "dash array values must not be negative",
            });
        }
        lengths.push(length);
        let separated = cursor.skip_comma_wsp();
        if cursor.is_at_end() {
            return Ok(lengths);
        }
        if !separated && !matches!(cursor.peek(), Some('+' | '-' | '.' | '0'..='9')) {
            return Err(ValueError::TrailingData {
                offset: cursor.pos(),
            });
        }
    }
}

/// Parses the `<number> [<number>]` idiom shared by `stdDeviation`,
/// `baseFrequency`, `order`, `radius`, and `kernelUnitLength`: one required
/// number, optionally followed by a second that supplies a separate value
/// for the second axis (the first is shared by both axes when only one
/// number is given).
///
/// Unlike [`parse_length_list`], this is a closed one-or-two-value
/// production rather than an open-ended list, so — deliberately diverging
/// from `parse_length_list`'s leniency — a separator with nothing valid
/// after it (`"5,"`) is rejected as [`ValueError::UnexpectedEnd`] rather
/// than treated as a complete one-number value.
///
/// # Examples
///
/// ```
/// use oxixml_svg::values::length::parse_number_optional_number;
///
/// assert_eq!(parse_number_optional_number("5").expect("single value"), (5.0, None));
/// assert_eq!(
///     parse_number_optional_number("5,3").expect("comma-separated pair"),
///     (5.0, Some(3.0))
/// );
/// assert_eq!(
///     parse_number_optional_number("5 3").expect("space-separated pair"),
///     (5.0, Some(3.0))
/// );
/// ```
///
/// # Errors
///
/// [`ValueError::Empty`] for empty input, [`ValueError::UnexpectedEnd`] for
/// a dangling trailing comma, and [`ValueError::TrailingData`] for content
/// glued on with no separator at all that is not the start of a second
/// number. When at least some white space (or a comma) was consumed before
/// unparsable content, the number scanner's own error is reported instead
/// (see [`parse_length_list`] for the same rule spelled out in more
/// detail), and any malformed first or second number surfaces that
/// number's own error regardless of separators.
pub fn parse_number_optional_number(input: &str) -> Result<(f64, Option<f64>), ValueError> {
    let mut cursor = Cursor::new(input);
    cursor.skip_wsp();
    if cursor.is_at_end() {
        return Err(ValueError::Empty);
    }
    let first = cursor.number()?;
    let (separated, had_comma) = skip_separator(&mut cursor);
    if cursor.is_at_end() {
        return if had_comma {
            Err(ValueError::UnexpectedEnd {
                offset: cursor.pos(),
            })
        } else {
            Ok((first, None))
        };
    }
    if !separated && !matches!(cursor.peek(), Some('+' | '-' | '.' | '0'..='9')) {
        return Err(ValueError::TrailingData {
            offset: cursor.pos(),
        });
    }
    let second = cursor.number()?;
    cursor.done()?;
    Ok((first, Some(second)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_LENGTH_UNITS: [LengthUnit; 17] = [
        LengthUnit::None,
        LengthUnit::Px,
        LengthUnit::Em,
        LengthUnit::Ex,
        LengthUnit::Ch,
        LengthUnit::Rem,
        LengthUnit::Vw,
        LengthUnit::Vh,
        LengthUnit::Vmin,
        LengthUnit::Vmax,
        LengthUnit::Q,
        LengthUnit::In,
        LengthUnit::Cm,
        LengthUnit::Mm,
        LengthUnit::Pt,
        LengthUnit::Pc,
        LengthUnit::Percent,
    ];

    #[test]
    fn length_unit_as_str_is_lowercase_for_every_variant() {
        for unit in ALL_LENGTH_UNITS {
            let text = unit.as_str();
            assert!(
                text.chars().all(|c| !c.is_ascii_uppercase()),
                "{unit:?} serializes as {text:?}, which contains an uppercase letter"
            );
        }
        assert_eq!(LengthUnit::None.as_str(), "");
        assert_eq!(LengthUnit::Percent.as_str(), "%");
        assert_eq!(LengthUnit::Q.as_str(), "q");
    }

    #[test]
    fn length_unit_display_matches_as_str() {
        for unit in ALL_LENGTH_UNITS {
            assert_eq!(unit.to_string(), unit.as_str());
        }
    }

    #[test]
    fn length_parses_every_named_unit_case_insensitively() {
        let cases: [(&str, LengthUnit); 16] = [
            ("px", LengthUnit::Px),
            ("em", LengthUnit::Em),
            ("ex", LengthUnit::Ex),
            ("ch", LengthUnit::Ch),
            ("rem", LengthUnit::Rem),
            ("vw", LengthUnit::Vw),
            ("vh", LengthUnit::Vh),
            ("vmin", LengthUnit::Vmin),
            ("vmax", LengthUnit::Vmax),
            ("q", LengthUnit::Q),
            ("in", LengthUnit::In),
            ("cm", LengthUnit::Cm),
            ("mm", LengthUnit::Mm),
            ("pt", LengthUnit::Pt),
            ("pc", LengthUnit::Pc),
            ("%", LengthUnit::Percent),
        ];
        for (suffix, unit) in cases {
            for spelling in [
                suffix.to_string(),
                suffix.to_ascii_uppercase(),
                titlecase(suffix),
            ] {
                let text = format!("10{spelling}");
                let length = Length::parse(&text)
                    .unwrap_or_else(|e| panic!("{text:?} should parse as a length, got {e}"));
                assert_eq!(length, Length::new(10.0, unit), "parsing {text:?}");
            }
        }
        // A bare number has no unit at all.
        assert_eq!(
            Length::parse("10").expect("unitless length"),
            Length::new(10.0, LengthUnit::None)
        );
    }

    fn titlecase(s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
            None => String::new(),
        }
    }

    #[test]
    fn length_parse_rejects_empty_value() {
        assert_eq!(Length::parse(""), Err(ValueError::Empty));
        assert_eq!(Length::parse("   "), Err(ValueError::Empty));
    }

    #[test]
    fn length_parse_rejects_unknown_unit_with_offset() {
        assert_eq!(
            Length::parse("10furlongs"),
            Err(ValueError::UnknownKeyword {
                offset: 2,
                keyword: "furlongs".into(),
            })
        );
    }

    #[test]
    fn length_parse_rejects_trailing_data() {
        assert_eq!(
            Length::parse("10px extra"),
            Err(ValueError::TrailingData { offset: 5 })
        );
        assert_eq!(
            Length::parse("50%eter"),
            Err(ValueError::TrailingData { offset: 3 })
        );
    }

    #[test]
    fn length_display_round_trips_for_finite_values() {
        for (value, unit) in [
            (0.0, LengthUnit::None),
            (12.5, LengthUnit::Px),
            (-3.0, LengthUnit::Em),
            (0.5, LengthUnit::Percent),
            (100.0, LengthUnit::Vmax),
            (1.0, LengthUnit::Q),
        ] {
            let length = Length::new(value, unit);
            let text = length.to_string();
            assert_eq!(Length::parse(&text).expect("round-trip parse"), length);
        }
        // Rust's `{}` for f64 drops the decimal point for whole numbers.
        assert_eq!(Length::new(10.0, LengthUnit::Px).to_string(), "10px");
    }

    #[test]
    fn length_non_finite_value_is_out_of_the_round_trip_contract() {
        let length = Length::parse("1e999").expect("overflowing literal saturates, not errors");
        assert_eq!(length.value, f64::INFINITY);
        assert_eq!(length.unit, LengthUnit::None);
        assert_eq!(length.to_string(), "inf");
        // `"inf"` is not valid `<number>` syntax, so it does not re-parse:
        // non-finite lengths are documented as outside the round-trip
        // guarantee, not silently broken.
        assert!(Length::parse("inf").is_err());
    }

    #[test]
    fn angle_unit_display_matches_as_str() {
        for unit in [
            AngleUnit::None,
            AngleUnit::Deg,
            AngleUnit::Grad,
            AngleUnit::Rad,
            AngleUnit::Turn,
        ] {
            assert_eq!(unit.to_string(), unit.as_str());
        }
        assert_eq!(AngleUnit::None.as_str(), "");
    }

    #[test]
    fn angle_parses_bare_number_as_unitless() {
        assert_eq!(
            Angle::parse("45").expect("bare angle"),
            Angle::new(45.0, AngleUnit::None)
        );
    }

    #[test]
    fn angle_parses_every_unit_case_insensitively() {
        for (suffix, unit) in [
            ("deg", AngleUnit::Deg),
            ("grad", AngleUnit::Grad),
            ("rad", AngleUnit::Rad),
            ("turn", AngleUnit::Turn),
        ] {
            let text = format!("30{}", suffix.to_ascii_uppercase());
            assert_eq!(
                Angle::parse(&text).unwrap_or_else(|e| panic!("{text:?}: {e}")),
                Angle::new(30.0, unit)
            );
        }
    }

    #[test]
    fn angle_parse_rejects_unknown_unit() {
        assert_eq!(
            Angle::parse("45degrees"),
            Err(ValueError::UnknownKeyword {
                offset: 2,
                keyword: "degrees".into(),
            })
        );
    }

    #[test]
    fn angle_to_degrees_matches_known_conversions() {
        assert_eq!(Angle::new(45.0, AngleUnit::None).to_degrees(), 45.0);
        assert_eq!(Angle::new(45.0, AngleUnit::Deg).to_degrees(), 45.0);
        assert_eq!(Angle::new(100.0, AngleUnit::Grad).to_degrees(), 90.0);
        assert_eq!(Angle::new(0.25, AngleUnit::Turn).to_degrees(), 90.0);
        let from_radians = Angle::new(std::f64::consts::PI, AngleUnit::Rad).to_degrees();
        assert!((from_radians - 180.0).abs() < 1e-9, "{from_radians}");
    }

    #[test]
    fn angle_to_radians_matches_known_conversions() {
        let two_pi = std::f64::consts::TAU;
        assert!((Angle::new(1.0, AngleUnit::Turn).to_radians() - two_pi).abs() < 1e-9);
        assert!(
            (Angle::new(180.0, AngleUnit::Deg).to_radians() - std::f64::consts::PI).abs() < 1e-9
        );
        assert_eq!(Angle::new(1.5, AngleUnit::Rad).to_radians(), 1.5);
    }

    #[test]
    fn angle_display_round_trips() {
        for (value, unit) in [
            (45.0, AngleUnit::None),
            (45.0, AngleUnit::Deg),
            (100.0, AngleUnit::Grad),
            (1.5, AngleUnit::Rad),
            (0.25, AngleUnit::Turn),
        ] {
            let angle = Angle::new(value, unit);
            assert_eq!(Angle::parse(&angle.to_string()).expect("round-trip"), angle);
        }
    }

    #[test]
    fn length_context_default_matches_css_initial_values() {
        let ctx = LengthContext::default();
        assert_eq!(ctx.font_size, 16.0);
        assert_eq!(ctx.root_font_size, 16.0);
        assert_eq!(ctx.viewport_width, 100.0);
        assert_eq!(ctx.viewport_height, 100.0);
        assert_eq!(ctx.x_height, None);
        assert_eq!(ctx.ch_advance, None);
    }

    #[test]
    fn length_context_builder_methods_update_fields() {
        let ctx = LengthContext::default()
            .with_font_size(20.0)
            .with_root_font_size(24.0)
            .with_x_height(12.0)
            .with_ch_advance(9.0)
            .with_viewport(320.0, 240.0);
        assert_eq!(ctx.font_size, 20.0);
        assert_eq!(ctx.root_font_size, 24.0);
        assert_eq!(ctx.x_height, Some(12.0));
        assert_eq!(ctx.ch_advance, Some(9.0));
        assert_eq!(ctx.viewport_width, 320.0);
        assert_eq!(ctx.viewport_height, 240.0);
    }

    #[test]
    fn resolve_passes_through_unitless_and_px() {
        let ctx = LengthContext::default();
        assert_eq!(
            Length::new(42.0, LengthUnit::None).resolve(&ctx, PercentBasis::Width),
            42.0
        );
        assert_eq!(
            Length::new(42.0, LengthUnit::Px).resolve(&ctx, PercentBasis::Width),
            42.0
        );
    }

    #[test]
    fn resolve_font_relative_units_use_font_size() {
        let ctx = LengthContext::default().with_font_size(20.0);
        assert_eq!(
            Length::new(2.0, LengthUnit::Em).resolve(&ctx, PercentBasis::Width),
            40.0
        );
    }

    #[test]
    fn resolve_ex_and_ch_fall_back_to_half_em_by_default() {
        let ctx = LengthContext::default().with_font_size(20.0);
        assert_eq!(
            Length::new(1.0, LengthUnit::Ex).resolve(&ctx, PercentBasis::Width),
            10.0
        );
        assert_eq!(
            Length::new(1.0, LengthUnit::Ch).resolve(&ctx, PercentBasis::Width),
            10.0
        );
    }

    #[test]
    fn resolve_ex_and_ch_use_explicit_context_values_when_set() {
        let ctx = LengthContext::default()
            .with_font_size(20.0)
            .with_x_height(11.0)
            .with_ch_advance(9.5);
        assert_eq!(
            Length::new(2.0, LengthUnit::Ex).resolve(&ctx, PercentBasis::Width),
            22.0
        );
        assert_eq!(
            Length::new(2.0, LengthUnit::Ch).resolve(&ctx, PercentBasis::Width),
            19.0
        );
    }

    #[test]
    fn resolve_rem_uses_root_font_size_not_current_font_size() {
        let ctx = LengthContext::default()
            .with_font_size(10.0)
            .with_root_font_size(16.0);
        assert_eq!(
            Length::new(2.0, LengthUnit::Rem).resolve(&ctx, PercentBasis::Width),
            32.0
        );
    }

    #[test]
    fn resolve_viewport_relative_units() {
        let ctx = LengthContext::default().with_viewport(200.0, 50.0);
        assert_eq!(
            Length::new(50.0, LengthUnit::Vw).resolve(&ctx, PercentBasis::Width),
            100.0
        );
        assert_eq!(
            Length::new(50.0, LengthUnit::Vh).resolve(&ctx, PercentBasis::Width),
            25.0
        );
        assert_eq!(
            Length::new(100.0, LengthUnit::Vmin).resolve(&ctx, PercentBasis::Width),
            50.0
        );
        assert_eq!(
            Length::new(100.0, LengthUnit::Vmax).resolve(&ctx, PercentBasis::Width),
            200.0
        );
    }

    #[test]
    fn resolve_absolute_units_match_css_pixel_conversions() {
        let ctx = LengthContext::default();
        assert_eq!(
            Length::new(1.0, LengthUnit::In).resolve(&ctx, PercentBasis::Width),
            96.0
        );
        assert_eq!(
            Length::new(2.54, LengthUnit::Cm).resolve(&ctx, PercentBasis::Width),
            96.0
        );
        assert_eq!(
            Length::new(25.4, LengthUnit::Mm).resolve(&ctx, PercentBasis::Width),
            96.0
        );
        assert_eq!(
            Length::new(101.6, LengthUnit::Q).resolve(&ctx, PercentBasis::Width),
            96.0
        );
        assert_eq!(
            Length::new(72.0, LengthUnit::Pt).resolve(&ctx, PercentBasis::Width),
            96.0
        );
        assert_eq!(
            Length::new(1.0, LengthUnit::Pc).resolve(&ctx, PercentBasis::Width),
            16.0
        );
        assert_eq!(
            Length::new(6.0, LengthUnit::Pc).resolve(&ctx, PercentBasis::Width),
            96.0
        );
    }

    #[test]
    fn resolve_percent_uses_the_requested_basis() {
        let ctx = LengthContext::default().with_viewport(200.0, 100.0);
        let half = Length::new(50.0, LengthUnit::Percent);
        assert_eq!(half.resolve(&ctx, PercentBasis::Width), 100.0);
        assert_eq!(half.resolve(&ctx, PercentBasis::Height), 50.0);
        let expected_diagonal =
            (200_f64.powi(2) + 100_f64.powi(2)).sqrt() / std::f64::consts::SQRT_2 * 0.5;
        assert_eq!(
            half.resolve(&ctx, PercentBasis::Diagonal),
            expected_diagonal
        );
    }

    #[test]
    fn resolve_percent_bases_agree_for_a_square_viewport() {
        // For the default 100x100 viewport, sqrt(100^2+100^2)/sqrt(2) is
        // exactly 100, so all three bases must agree; this is the test
        // that would catch a dropped or misplaced `/ sqrt(2)`.
        let ctx = LengthContext::default();
        let half = Length::new(50.0, LengthUnit::Percent);
        let width = half.resolve(&ctx, PercentBasis::Width);
        let height = half.resolve(&ctx, PercentBasis::Height);
        let diagonal = half.resolve(&ctx, PercentBasis::Diagonal);
        assert_eq!(width, 50.0);
        assert_eq!(height, 50.0);
        assert_eq!(diagonal, 50.0);
    }

    #[test]
    fn parse_length_list_accepts_mixed_separators() {
        assert_eq!(
            parse_length_list("10px, 20% 1in,2em").expect("valid list"),
            vec![
                Length::new(10.0, LengthUnit::Px),
                Length::new(20.0, LengthUnit::Percent),
                Length::new(1.0, LengthUnit::In),
                Length::new(2.0, LengthUnit::Em),
            ]
        );
    }

    #[test]
    fn parse_length_list_empty_value_is_empty_list() {
        assert_eq!(parse_length_list("").expect("empty list"), Vec::new());
        assert_eq!(parse_length_list("   ").expect("empty list"), Vec::new());
    }

    #[test]
    fn parse_length_list_accepts_a_single_trailing_separator() {
        // Documents and pins the deliberate leniency shared with
        // `crate::values::lex::parse_number_list`.
        assert_eq!(
            parse_length_list("10px,").expect("trailing comma accepted"),
            vec![Length::new(10.0, LengthUnit::Px)]
        );
    }

    #[test]
    fn parse_length_list_rejects_unknown_unit_mid_list() {
        assert_eq!(
            parse_length_list("10px, 5bogus"),
            Err(ValueError::UnknownKeyword {
                offset: 7,
                keyword: "bogus".into(),
            })
        );
    }

    #[test]
    fn parse_length_list_rejects_trailing_garbage_glued_with_no_separator() {
        // No white space or comma at all before the garbage: the list loop
        // never gets a chance to attempt another element, so this is
        // reported as trailing content after the value already scanned.
        assert_eq!(
            parse_length_list("10px;"),
            Err(ValueError::TrailingData { offset: 4 })
        );
    }

    #[test]
    fn parse_length_list_garbage_after_white_space_is_an_invalid_next_element() {
        // A space *was* consumed as a separator, so the loop instead
        // attempts to scan another list element starting at the garbage —
        // the same rule `crate::values::lex::parse_number_list` follows —
        // and reports whatever error that failed attempt produces rather
        // than `TrailingData`.
        assert_eq!(
            parse_length_list("10px x"),
            Err(ValueError::InvalidNumber { offset: 5 })
        );
    }

    #[test]
    fn parse_length_list_rejects_a_unit_keyword_glued_after_percent() {
        // `%` is a fixed one-character unit: `Length::scan` never attempts
        // a `keyword()` lookup after consuming it, so a following letter is
        // never absorbed as if it were part of the percent sign. It is
        // reported as trailing content, exactly as a letter glued after any
        // other unit would be.
        assert_eq!(
            parse_length_list("50%px 1"),
            Err(ValueError::TrailingData { offset: 3 })
        );
    }

    #[test]
    fn parse_dash_array_accepts_lengths_and_percentages() {
        assert_eq!(
            parse_dash_array("5,3,2").expect("valid dash array"),
            vec![
                Length::new(5.0, LengthUnit::None),
                Length::new(3.0, LengthUnit::None),
                Length::new(2.0, LengthUnit::None),
            ]
        );
        assert_eq!(
            parse_dash_array("5px 10%").expect("lengths and percentages mix"),
            vec![
                Length::new(5.0, LengthUnit::Px),
                Length::new(10.0, LengthUnit::Percent),
            ]
        );
    }

    #[test]
    fn parse_dash_array_rejects_negative_values_with_offset() {
        assert_eq!(
            parse_dash_array("5,-3,2"),
            Err(ValueError::OutOfRange {
                offset: 2,
                what: "dash array values must not be negative",
            })
        );
    }

    #[test]
    fn parse_dash_array_does_not_special_case_the_none_keyword() {
        // `"none"` is part of the outer `stroke-dasharray` grammar, not the
        // `dashes` production this function parses; callers must check for
        // it themselves before calling in.
        assert_eq!(
            parse_dash_array("none"),
            Err(ValueError::InvalidNumber { offset: 0 })
        );
    }

    #[test]
    fn parse_number_optional_number_single_value() {
        assert_eq!(
            parse_number_optional_number("5").expect("single value"),
            (5.0, None)
        );
        assert_eq!(
            parse_number_optional_number(" 5 ").expect("padded single value"),
            (5.0, None)
        );
    }

    #[test]
    fn parse_number_optional_number_two_values() {
        assert_eq!(
            parse_number_optional_number("5,3").expect("comma-separated pair"),
            (5.0, Some(3.0))
        );
        assert_eq!(
            parse_number_optional_number("5 3").expect("space-separated pair"),
            (5.0, Some(3.0))
        );
        assert_eq!(
            parse_number_optional_number("5-3").expect("sign-glued pair"),
            (5.0, Some(-3.0))
        );
    }

    #[test]
    fn parse_number_optional_number_rejects_dangling_trailing_comma() {
        assert_eq!(
            parse_number_optional_number("5,"),
            Err(ValueError::UnexpectedEnd { offset: 2 })
        );
    }

    #[test]
    fn parse_number_optional_number_rejects_trailing_garbage_glued_with_no_separator() {
        assert_eq!(
            parse_number_optional_number("5x"),
            Err(ValueError::TrailingData { offset: 1 })
        );
    }

    #[test]
    fn parse_number_optional_number_garbage_after_white_space_is_an_invalid_second_number() {
        // As with `parse_length_list`: a space was consumed as a
        // separator, so the function commits to reading a second number
        // and reports the number scanner's own error rather than
        // `TrailingData`.
        assert_eq!(
            parse_number_optional_number("5 x"),
            Err(ValueError::InvalidNumber { offset: 2 })
        );
    }

    #[test]
    fn parse_number_optional_number_rejects_a_third_value() {
        assert_eq!(
            parse_number_optional_number("5,3,1"),
            Err(ValueError::TrailingData { offset: 3 })
        );
    }

    #[test]
    fn parse_number_optional_number_rejects_empty_value() {
        assert_eq!(parse_number_optional_number(""), Err(ValueError::Empty));
        assert_eq!(parse_number_optional_number("  "), Err(ValueError::Empty));
    }
}
