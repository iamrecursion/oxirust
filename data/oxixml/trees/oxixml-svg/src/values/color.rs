//! The SVG/CSS `<color>` microsyntax: named keywords, hexadecimal notation, and
//! the `rgb()`/`hsl()` functional notations.
//!
//! Every accepted syntax is resolved eagerly to an 8-bit-per-channel sRGB
//! [`Color`]. This is a deliberate, documented scope decision:
//!
//! # What is accepted
//! - The 147 SVG 1.1 / CSS extended named colors, plus `rebeccapurple` (CSS
//!   Color 4) and `transparent`, looked up case-insensitively (148 + 1 = 149
//!   keywords total, including the American/British `gray`/`grey` spelling
//!   pairs the SVG 1.1 table itself carries).
//! - Hex notation: `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`. Any other digit
//!   count — anything but 3, 4, 6, or 8 — is a grammar error, never silently
//!   padded or truncated.
//! - `rgb()`/`rgba()` (the two names are accepted interchangeably, matching
//!   CSS Color 4's unification of them into aliases of one function): the
//!   legacy comma syntax with integer or percentage channels (never mixed
//!   within one call) and an optional comma-separated alpha, and the modern
//!   space syntax with an optional `/ <alpha>`.
//! - `hsl()`/`hsla()` (also accepted interchangeably): legacy comma and
//!   modern space+slash syntax. The hue accepts a bare number (degrees) or an
//!   explicit `deg`/`grad`/`rad`/`turn` angle unit; saturation and lightness
//!   must be percentages (CSS Color 4 does not permit bare numbers there).
//!   Conversion to RGB follows the CSS Color 3/4 `hsl-to-rgb` algorithm
//!   verbatim.
//! - Out-of-range numeric values clamp per CSS (`rgb(300 0 0)` clamps to
//!   `#ff0000`, it does not error) — only grammar violations are rejected.
//! - Every numeric component (`rgb()`/`hsl()` channels and the hue's number)
//!   is scanned with [`crate::values::lex::Cursor::number`], which implements
//!   the SVG `<number>` grammar — a **lenient superset** of CSS's own
//!   `<number>` token: it additionally accepts a bare trailing decimal point
//!   (`5.`) with no following digits, which strict CSS does not. This is the
//!   shared, crate-wide lexer (see the spec catalog §3.1, which explicitly
//!   sanctions "implement one lenient superset and document the divergence"
//!   over hand-rolling a second, CSS-exact tokenizer for this one context).
//!
//! # What is deliberately out of scope for this release
//! - `hwb()`, `lab()`, `lch()`, `oklab()`, `oklch()`, and `color()` are
//!   rejected with [`ValueError::UnknownKeyword`]. **Note for whoever reviews
//!   this cut**: the spec catalog (§3.4) records a "Tier 1: parse as a typed,
//!   unresolved variant, do not implement the color-space math" decision for
//!   these functions, which presupposes a `Color` type with per-family
//!   variants. This module's `Color` is the concrete
//!   `{ r: u8, g: u8, b: u8, a: u8 }` struct the crate-level implementation
//!   plan mandates, which has no unresolved-variant slot to hold an
//!   un-converted `oklch()` triple in — the catalog's premise does not hold
//!   for this type, so its conclusion does not transfer here. The
//!   implementation plan's own "Key design decisions" section independently
//!   and unambiguously cuts the same three families (grouping `hwb` with
//!   `lab`/`oklch`) for 0.1.1, so that is the rule this module follows. `hwb()`
//!   is cheap to add later (it reduces to the same hue machinery `hsl()`
//!   already uses, mixed with white/black), but is not implemented now so
//!   that the cut is one uniform, reversible line rather than a partial one.
//! - The CSS Color 4 `none` per-channel keyword (e.g. `rgb(none 0 0)`, used
//!   for animation interpolation) is not accepted.
//! - The SVG 1.1 `<color> <icccolor>` trailing-annotation form is not
//!   accepted; a value such as `"blue icc-color(...)"` parses the `<color>`
//!   part and then reports the remainder as
//!   [`ValueError::TrailingData`]. This form appears in real-world SVG 1.1
//!   content, so callers that need to tolerate it should strip a trailing
//!   `icc-color(...)` annotation themselves before calling [`Color::parse`].
//! - The CSS-wide keyword `inherit` is not part of the `<color>` grammar
//!   itself (it is a per-property cascade behavior); callers implementing
//!   the cascade must special-case the literal string `"inherit"` before
//!   calling into this module.

use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use crate::error::ValueError;
use crate::values::lex::Cursor;

/// An sRGB color with 8-bit red, green, blue, and alpha channels.
///
/// `Color` is the fully-resolved result of parsing any accepted `<color>`
/// syntax (named keyword, hex notation, or a `rgb()`/`hsl()` function): all
/// syntaxes converge on this one concrete representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Color {
    /// Opaque black, `#000000`.
    pub const BLACK: Self = Self::rgb(0x00, 0x00, 0x00);

    /// Opaque white, `#ffffff`.
    pub const WHITE: Self = Self::rgb(0xFF, 0xFF, 0xFF);

    /// Fully transparent black, the resolved value of the `transparent`
    /// keyword.
    pub const TRANSPARENT: Self = Self::rgba(0x00, 0x00, 0x00, 0x00);

    /// Creates a fully opaque color from its red, green, and blue channels.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Creates a color from its red, green, blue, and alpha channels.
    #[must_use]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// The red channel, `0..=255`.
    #[must_use]
    pub const fn r(&self) -> u8 {
        self.r
    }

    /// The green channel, `0..=255`.
    #[must_use]
    pub const fn g(&self) -> u8 {
        self.g
    }

    /// The blue channel, `0..=255`.
    #[must_use]
    pub const fn b(&self) -> u8 {
        self.b
    }

    /// The alpha channel, `0..=255` (255 is fully opaque).
    #[must_use]
    pub const fn a(&self) -> u8 {
        self.a
    }

    /// Looks up one of the 149 recognized color keywords (the 147 SVG 1.1
    /// names, `rebeccapurple`, and `transparent`) case-insensitively.
    ///
    /// This performs no allocation: the comparison walks both byte slices
    /// directly rather than building a lowercased copy of `name`.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        named_color(name)
    }

    /// Parses a complete SVG/CSS `<color>` value.
    ///
    /// Accepts a named keyword, `#`-prefixed hex notation, or a
    /// `rgb()`/`rgba()`/`hsl()`/`hsla()` function, with optional surrounding
    /// white space. See the [module documentation](self) for the exact
    /// grammar accepted and the syntaxes this release does not implement.
    ///
    /// # Errors
    /// Returns [`ValueError`] with a byte offset into `input` if the value is
    /// empty, uses a syntax this module does not recognize, or otherwise
    /// violates the grammar (numeric *ranges* are clamped rather than
    /// rejected; only shape violations are errors).
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let mut cursor = Cursor::new(input);
        cursor.skip_wsp();
        if cursor.is_at_end() {
            return Err(ValueError::Empty);
        }
        let color = parse_from_cursor(&mut cursor)?;
        cursor.done()?;
        Ok(color)
    }

    /// Formats this color as `#rrggbb`, or `#rrggbbaa` when the alpha channel
    /// is not fully opaque (255).
    #[must_use]
    pub fn to_hex_string(&self) -> String {
        if self.a == 0xFF {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("#{:02x}{:02x}{:02x}{:02x}", self.r, self.g, self.b, self.a)
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex_string())
    }
}

impl FromStr for Color {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Parses one `<color>` value starting at the cursor's current position,
/// consuming exactly the characters that belong to it and leaving the
/// remainder (e.g. a `<paint>` fallback, or an `icc-color(...)` annotation)
/// for the caller. This is the composition point [`crate::values::paint`]
/// uses; [`Color::parse`] is a thin wrapper that additionally requires the
/// whole input to be consumed.
pub(crate) fn parse_from_cursor(cursor: &mut Cursor<'_>) -> Result<Color, ValueError> {
    match cursor.peek() {
        Some('#') => parse_hex(cursor),
        Some(c) if c.is_ascii_alphabetic() => {
            let start = cursor.pos();
            let word = cursor.keyword();
            if cursor.peek() == Some('(') {
                cursor.bump();
                parse_function(cursor, word, start)
            } else {
                named_color(word).ok_or_else(|| ValueError::UnknownKeyword {
                    offset: start,
                    keyword: word.into(),
                })
            }
        }
        Some(found) => Err(ValueError::UnexpectedChar {
            offset: cursor.pos(),
            found,
        }),
        None => Err(ValueError::UnexpectedEnd {
            offset: cursor.pos(),
        }),
    }
}

/// Parses a `<number>` optionally followed by `%`, without judging whether a
/// percentage was required at this position. Returns the raw numeric value
/// together with whether a trailing `%` was consumed. Shared with
/// [`crate::values::paint::parse_opacity`], which is why this is visible
/// beyond this module.
pub(crate) fn number_or_percent(cursor: &mut Cursor<'_>) -> Result<(f64, bool), ValueError> {
    let value = cursor.number()?;
    if cursor.peek() == Some('%') {
        cursor.bump();
        Ok((value, true))
    } else {
        Ok((value, false))
    }
}

fn parse_hex(cursor: &mut Cursor<'_>) -> Result<Color, ValueError> {
    let hash_offset = cursor.pos();
    cursor.bump(); // the '#' itself
    let digits_start = cursor.pos();
    while matches!(cursor.peek(), Some(c) if c.is_ascii_hexdigit()) {
        cursor.bump();
    }
    let digits = &cursor.input().as_bytes()[digits_start..cursor.pos()];
    match digits.len() {
        3 => Ok(Color::rgb(
            expand_nibble(digits[0]),
            expand_nibble(digits[1]),
            expand_nibble(digits[2]),
        )),
        4 => Ok(Color::rgba(
            expand_nibble(digits[0]),
            expand_nibble(digits[1]),
            expand_nibble(digits[2]),
            expand_nibble(digits[3]),
        )),
        6 => Ok(Color::rgb(
            hex_byte(digits[0], digits[1]),
            hex_byte(digits[2], digits[3]),
            hex_byte(digits[4], digits[5]),
        )),
        8 => Ok(Color::rgba(
            hex_byte(digits[0], digits[1]),
            hex_byte(digits[2], digits[3]),
            hex_byte(digits[4], digits[5]),
            hex_byte(digits[6], digits[7]),
        )),
        _ => Err(ValueError::InvalidNumber {
            offset: hash_offset,
        }),
    }
}

/// Converts one already-validated ASCII hex digit byte to its numeric value.
fn hex_nibble(byte: u8) -> u8 {
    (byte as char)
        .to_digit(16)
        .expect("byte was validated as an ASCII hex digit while scanning") as u8
}

/// Expands a single hex digit `x` to the doubled byte `0xXX` (`x * 17`), the
/// `#RGB`/`#RGBA` shorthand expansion rule.
fn expand_nibble(byte: u8) -> u8 {
    hex_nibble(byte) * 17
}

/// Combines two already-validated ASCII hex digit bytes into one channel
/// value.
fn hex_byte(hi: u8, lo: u8) -> u8 {
    hex_nibble(hi) * 16 + hex_nibble(lo)
}

fn parse_function(
    cursor: &mut Cursor<'_>,
    name: &str,
    name_start: usize,
) -> Result<Color, ValueError> {
    if name.eq_ignore_ascii_case("rgb") || name.eq_ignore_ascii_case("rgba") {
        parse_rgb_body(cursor)
    } else if name.eq_ignore_ascii_case("hsl") || name.eq_ignore_ascii_case("hsla") {
        parse_hsl_body(cursor)
    } else {
        // Covers hwb()/lab()/lch()/oklab()/oklch()/color() (documented cut,
        // see the module doc) and any other unrecognized function name.
        Err(ValueError::UnknownKeyword {
            offset: name_start,
            keyword: name.into(),
        })
    }
}

/// Parses one `rgb()` channel. Once the first channel has established
/// whether the call uses integers or percentages (`want_percent`), every
/// later channel must match: mixing is a grammar error, not a clamp.
fn parse_rgb_channel(
    cursor: &mut Cursor<'_>,
    want_percent: Option<bool>,
) -> Result<(f64, bool), ValueError> {
    let value = cursor.number()?;
    let after_number = cursor.pos();
    let is_percent = cursor.peek() == Some('%');
    if let Some(want) = want_percent {
        if want && !is_percent {
            return Err(match cursor.peek() {
                Some(found) => ValueError::UnexpectedChar {
                    offset: after_number,
                    found,
                },
                None => ValueError::UnexpectedEnd {
                    offset: after_number,
                },
            });
        }
        if !want && is_percent {
            return Err(ValueError::UnexpectedChar {
                offset: after_number,
                found: '%',
            });
        }
    }
    if is_percent {
        cursor.bump();
    }
    Ok((value, is_percent))
}

/// Scales a channel value (a plain number in `0..=255`, or a percentage of
/// `255`) to `u8`, clamping out-of-range magnitudes per CSS.
fn channel_to_u8(value: f64, is_percent: bool) -> u8 {
    let scaled = if is_percent {
        value / 100.0 * 255.0
    } else {
        value
    };
    scaled.clamp(0.0, 255.0).round() as u8
}

/// Normalizes an alpha value (a plain number in `0..=1`, or a percentage) to
/// `u8`, clamping out-of-range magnitudes per CSS.
fn alpha_to_u8(value: f64, is_percent: bool) -> u8 {
    let normalized = if is_percent { value / 100.0 } else { value };
    (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Skips white space, requiring at least one character to be consumed. Used
/// between the space-separated channels of the modern `rgb()`/`hsl()`
/// syntaxes, where (unlike around punctuation) the separator is mandatory.
fn require_wsp(cursor: &mut Cursor<'_>) -> Result<(), ValueError> {
    let before = cursor.pos();
    cursor.skip_wsp();
    if cursor.pos() == before {
        Err(match cursor.peek() {
            Some(found) => ValueError::UnexpectedChar {
                offset: before,
                found,
            },
            None => ValueError::UnexpectedEnd { offset: before },
        })
    } else {
        Ok(())
    }
}

fn parse_rgb_body(cursor: &mut Cursor<'_>) -> Result<Color, ValueError> {
    cursor.skip_wsp();
    let (v1, p1) = parse_rgb_channel(cursor, None)?;
    let gap_before = cursor.pos();
    cursor.skip_wsp();
    let had_wsp = cursor.pos() != gap_before;
    let (r, g, b, alpha) = if cursor.eat(',') {
        // Legacy comma syntax: `rgb(R, G, B[, A])`.
        cursor.skip_wsp();
        let (v2, _) = parse_rgb_channel(cursor, Some(p1))?;
        cursor.skip_wsp();
        cursor.expect_char(',')?;
        cursor.skip_wsp();
        let (v3, _) = parse_rgb_channel(cursor, Some(p1))?;
        cursor.skip_wsp();
        let alpha = if cursor.eat(',') {
            cursor.skip_wsp();
            let (av, ap) = number_or_percent(cursor)?;
            cursor.skip_wsp();
            alpha_to_u8(av, ap)
        } else {
            0xFF
        };
        (
            channel_to_u8(v1, p1),
            channel_to_u8(v2, p1),
            channel_to_u8(v3, p1),
            alpha,
        )
    } else {
        // Modern space syntax: `rgb(R G B [/ A])`. The gap already skipped
        // above must have been white space (there is no comma alternative
        // once we know we are not in the legacy branch).
        if !had_wsp {
            return Err(match cursor.peek() {
                Some(found) => ValueError::UnexpectedChar {
                    offset: cursor.pos(),
                    found,
                },
                None => ValueError::UnexpectedEnd {
                    offset: cursor.pos(),
                },
            });
        }
        let (v2, _) = parse_rgb_channel(cursor, Some(p1))?;
        require_wsp(cursor)?;
        let (v3, _) = parse_rgb_channel(cursor, Some(p1))?;
        cursor.skip_wsp();
        let alpha = if cursor.eat('/') {
            cursor.skip_wsp();
            let (av, ap) = number_or_percent(cursor)?;
            cursor.skip_wsp();
            alpha_to_u8(av, ap)
        } else {
            0xFF
        };
        (
            channel_to_u8(v1, p1),
            channel_to_u8(v2, p1),
            channel_to_u8(v3, p1),
            alpha,
        )
    };
    cursor.expect_char(')')?;
    Ok(Color::rgba(r, g, b, alpha))
}

fn parse_hsl_body(cursor: &mut Cursor<'_>) -> Result<Color, ValueError> {
    cursor.skip_wsp();
    let hue = parse_hue(cursor)?;
    let gap_before = cursor.pos();
    cursor.skip_wsp();
    let had_wsp = cursor.pos() != gap_before;
    let (sat, light, alpha) = if cursor.eat(',') {
        // Legacy comma syntax: `hsl(H, S%, L%[, A])`.
        cursor.skip_wsp();
        let sat = parse_percentage_required(cursor)?;
        cursor.skip_wsp();
        cursor.expect_char(',')?;
        cursor.skip_wsp();
        let light = parse_percentage_required(cursor)?;
        cursor.skip_wsp();
        let alpha = if cursor.eat(',') {
            cursor.skip_wsp();
            let (av, ap) = number_or_percent(cursor)?;
            cursor.skip_wsp();
            alpha_to_u8(av, ap)
        } else {
            0xFF
        };
        (sat, light, alpha)
    } else {
        // Modern space syntax: `hsl(H S% L% [/ A])`.
        if !had_wsp {
            return Err(match cursor.peek() {
                Some(found) => ValueError::UnexpectedChar {
                    offset: cursor.pos(),
                    found,
                },
                None => ValueError::UnexpectedEnd {
                    offset: cursor.pos(),
                },
            });
        }
        let sat = parse_percentage_required(cursor)?;
        require_wsp(cursor)?;
        let light = parse_percentage_required(cursor)?;
        cursor.skip_wsp();
        let alpha = if cursor.eat('/') {
            cursor.skip_wsp();
            let (av, ap) = number_or_percent(cursor)?;
            cursor.skip_wsp();
            alpha_to_u8(av, ap)
        } else {
            0xFF
        };
        (sat, light, alpha)
    };
    cursor.expect_char(')')?;
    Ok(hsl_to_color(hue, sat, light, alpha))
}

/// Parses a `<hue>`: a bare number (degrees) or a number followed by a
/// `deg`/`grad`/`rad`/`turn` angle unit, returning degrees.
fn parse_hue(cursor: &mut Cursor<'_>) -> Result<f64, ValueError> {
    let value = cursor.number()?;
    let unit_start = cursor.pos();
    let mut probe = cursor.clone();
    let word = probe.keyword();
    if word.is_empty() {
        return Ok(value);
    }
    let degrees = if word.eq_ignore_ascii_case("deg") {
        value
    } else if word.eq_ignore_ascii_case("grad") {
        value * 0.9
    } else if word.eq_ignore_ascii_case("rad") {
        value.to_degrees()
    } else if word.eq_ignore_ascii_case("turn") {
        value * 360.0
    } else {
        return Err(ValueError::UnknownKeyword {
            offset: unit_start,
            keyword: word.into(),
        });
    };
    *cursor = probe;
    Ok(degrees)
}

/// Parses a percentage, erroring (rather than accepting a bare number) if no
/// `%` follows — `hsl()`'s saturation and lightness are percentage-only.
fn parse_percentage_required(cursor: &mut Cursor<'_>) -> Result<f64, ValueError> {
    let value = cursor.number()?;
    if cursor.peek() == Some('%') {
        cursor.bump();
        Ok(value)
    } else {
        Err(match cursor.peek() {
            Some(found) => ValueError::UnexpectedChar {
                offset: cursor.pos(),
                found,
            },
            None => ValueError::UnexpectedEnd {
                offset: cursor.pos(),
            },
        })
    }
}

fn hsl_to_color(hue_deg: f64, sat_pct: f64, light_pct: f64, alpha: u8) -> Color {
    let sat = sat_pct.clamp(0.0, 100.0) / 100.0;
    let light = light_pct.clamp(0.0, 100.0) / 100.0;
    let (r, g, b) = hsl_to_rgb(hue_deg, sat, light);
    Color::rgba(unit_to_u8(r), unit_to_u8(g), unit_to_u8(b), alpha)
}

fn unit_to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The CSS Color Module Level 3/4 `hsl-to-rgb` algorithm, transcribed
/// verbatim (`hue_deg` in degrees, `sat`/`light` normalized to `0.0..=1.0`).
fn hsl_to_rgb(hue_deg: f64, sat: f64, light: f64) -> (f64, f64, f64) {
    let h = (hue_deg / 360.0).rem_euclid(1.0);
    let m2 = if light <= 0.5 {
        light * (sat + 1.0)
    } else {
        light + sat - light * sat
    };
    let m1 = light * 2.0 - m2;
    (
        hue_to_rgb(m1, m2, h + 1.0 / 3.0),
        hue_to_rgb(m1, m2, h),
        hue_to_rgb(m1, m2, h - 1.0 / 3.0),
    )
}

/// The `hue-to-rgb` helper the CSS spec's `hsl-to-rgb` algorithm calls three
/// times (once per channel, with `h` offset by a third of a turn each time).
fn hue_to_rgb(m1: f64, m2: f64, h: f64) -> f64 {
    let mut h = h;
    if h < 0.0 {
        h += 1.0;
    }
    if h > 1.0 {
        h -= 1.0;
    }
    if h * 6.0 < 1.0 {
        m1 + (m2 - m1) * h * 6.0
    } else if h * 2.0 < 1.0 {
        m2
    } else if h * 3.0 < 2.0 {
        m1 + (m2 - m1) * (2.0 / 3.0 - h) * 6.0
    } else {
        m1
    }
}

type NamedEntry = (&'static str, Color);

/// The 149 recognized color keywords (147 SVG 1.1 names + `rebeccapurple` +
/// `transparent`), sorted strictly ascending by name so [`named_color`] can
/// binary-search it. **Do not hand-edit this list without re-verifying sort
/// order**: the SVG 1.1 spec's own prose table lists `gray, grey, green,
/// greenyellow` in that order (grouping the British spelling next to the
/// American one), which is *not* the byte-wise ascending order a binary
/// search requires (`green` sorts before `grey`) — this table was generated
/// from a script that sorted the (name, color) pairs directly, not
/// transcribed by eye, for exactly that reason.
static NAMED_COLORS: &[NamedEntry] = &[
    ("aliceblue", Color::rgb(0xF0, 0xF8, 0xFF)),
    ("antiquewhite", Color::rgb(0xFA, 0xEB, 0xD7)),
    ("aqua", Color::rgb(0x00, 0xFF, 0xFF)),
    ("aquamarine", Color::rgb(0x7F, 0xFF, 0xD4)),
    ("azure", Color::rgb(0xF0, 0xFF, 0xFF)),
    ("beige", Color::rgb(0xF5, 0xF5, 0xDC)),
    ("bisque", Color::rgb(0xFF, 0xE4, 0xC4)),
    ("black", Color::rgb(0x00, 0x00, 0x00)),
    ("blanchedalmond", Color::rgb(0xFF, 0xEB, 0xCD)),
    ("blue", Color::rgb(0x00, 0x00, 0xFF)),
    ("blueviolet", Color::rgb(0x8A, 0x2B, 0xE2)),
    ("brown", Color::rgb(0xA5, 0x2A, 0x2A)),
    ("burlywood", Color::rgb(0xDE, 0xB8, 0x87)),
    ("cadetblue", Color::rgb(0x5F, 0x9E, 0xA0)),
    ("chartreuse", Color::rgb(0x7F, 0xFF, 0x00)),
    ("chocolate", Color::rgb(0xD2, 0x69, 0x1E)),
    ("coral", Color::rgb(0xFF, 0x7F, 0x50)),
    ("cornflowerblue", Color::rgb(0x64, 0x95, 0xED)),
    ("cornsilk", Color::rgb(0xFF, 0xF8, 0xDC)),
    ("crimson", Color::rgb(0xDC, 0x14, 0x3C)),
    ("cyan", Color::rgb(0x00, 0xFF, 0xFF)),
    ("darkblue", Color::rgb(0x00, 0x00, 0x8B)),
    ("darkcyan", Color::rgb(0x00, 0x8B, 0x8B)),
    ("darkgoldenrod", Color::rgb(0xB8, 0x86, 0x0B)),
    ("darkgray", Color::rgb(0xA9, 0xA9, 0xA9)),
    ("darkgreen", Color::rgb(0x00, 0x64, 0x00)),
    ("darkgrey", Color::rgb(0xA9, 0xA9, 0xA9)),
    ("darkkhaki", Color::rgb(0xBD, 0xB7, 0x6B)),
    ("darkmagenta", Color::rgb(0x8B, 0x00, 0x8B)),
    ("darkolivegreen", Color::rgb(0x55, 0x6B, 0x2F)),
    ("darkorange", Color::rgb(0xFF, 0x8C, 0x00)),
    ("darkorchid", Color::rgb(0x99, 0x32, 0xCC)),
    ("darkred", Color::rgb(0x8B, 0x00, 0x00)),
    ("darksalmon", Color::rgb(0xE9, 0x96, 0x7A)),
    ("darkseagreen", Color::rgb(0x8F, 0xBC, 0x8F)),
    ("darkslateblue", Color::rgb(0x48, 0x3D, 0x8B)),
    ("darkslategray", Color::rgb(0x2F, 0x4F, 0x4F)),
    ("darkslategrey", Color::rgb(0x2F, 0x4F, 0x4F)),
    ("darkturquoise", Color::rgb(0x00, 0xCE, 0xD1)),
    ("darkviolet", Color::rgb(0x94, 0x00, 0xD3)),
    ("deeppink", Color::rgb(0xFF, 0x14, 0x93)),
    ("deepskyblue", Color::rgb(0x00, 0xBF, 0xFF)),
    ("dimgray", Color::rgb(0x69, 0x69, 0x69)),
    ("dimgrey", Color::rgb(0x69, 0x69, 0x69)),
    ("dodgerblue", Color::rgb(0x1E, 0x90, 0xFF)),
    ("firebrick", Color::rgb(0xB2, 0x22, 0x22)),
    ("floralwhite", Color::rgb(0xFF, 0xFA, 0xF0)),
    ("forestgreen", Color::rgb(0x22, 0x8B, 0x22)),
    ("fuchsia", Color::rgb(0xFF, 0x00, 0xFF)),
    ("gainsboro", Color::rgb(0xDC, 0xDC, 0xDC)),
    ("ghostwhite", Color::rgb(0xF8, 0xF8, 0xFF)),
    ("gold", Color::rgb(0xFF, 0xD7, 0x00)),
    ("goldenrod", Color::rgb(0xDA, 0xA5, 0x20)),
    ("gray", Color::rgb(0x80, 0x80, 0x80)),
    ("green", Color::rgb(0x00, 0x80, 0x00)),
    ("greenyellow", Color::rgb(0xAD, 0xFF, 0x2F)),
    ("grey", Color::rgb(0x80, 0x80, 0x80)),
    ("honeydew", Color::rgb(0xF0, 0xFF, 0xF0)),
    ("hotpink", Color::rgb(0xFF, 0x69, 0xB4)),
    ("indianred", Color::rgb(0xCD, 0x5C, 0x5C)),
    ("indigo", Color::rgb(0x4B, 0x00, 0x82)),
    ("ivory", Color::rgb(0xFF, 0xFF, 0xF0)),
    ("khaki", Color::rgb(0xF0, 0xE6, 0x8C)),
    ("lavender", Color::rgb(0xE6, 0xE6, 0xFA)),
    ("lavenderblush", Color::rgb(0xFF, 0xF0, 0xF5)),
    ("lawngreen", Color::rgb(0x7C, 0xFC, 0x00)),
    ("lemonchiffon", Color::rgb(0xFF, 0xFA, 0xCD)),
    ("lightblue", Color::rgb(0xAD, 0xD8, 0xE6)),
    ("lightcoral", Color::rgb(0xF0, 0x80, 0x80)),
    ("lightcyan", Color::rgb(0xE0, 0xFF, 0xFF)),
    ("lightgoldenrodyellow", Color::rgb(0xFA, 0xFA, 0xD2)),
    ("lightgray", Color::rgb(0xD3, 0xD3, 0xD3)),
    ("lightgreen", Color::rgb(0x90, 0xEE, 0x90)),
    ("lightgrey", Color::rgb(0xD3, 0xD3, 0xD3)),
    ("lightpink", Color::rgb(0xFF, 0xB6, 0xC1)),
    ("lightsalmon", Color::rgb(0xFF, 0xA0, 0x7A)),
    ("lightseagreen", Color::rgb(0x20, 0xB2, 0xAA)),
    ("lightskyblue", Color::rgb(0x87, 0xCE, 0xFA)),
    ("lightslategray", Color::rgb(0x77, 0x88, 0x99)),
    ("lightslategrey", Color::rgb(0x77, 0x88, 0x99)),
    ("lightsteelblue", Color::rgb(0xB0, 0xC4, 0xDE)),
    ("lightyellow", Color::rgb(0xFF, 0xFF, 0xE0)),
    ("lime", Color::rgb(0x00, 0xFF, 0x00)),
    ("limegreen", Color::rgb(0x32, 0xCD, 0x32)),
    ("linen", Color::rgb(0xFA, 0xF0, 0xE6)),
    ("magenta", Color::rgb(0xFF, 0x00, 0xFF)),
    ("maroon", Color::rgb(0x80, 0x00, 0x00)),
    ("mediumaquamarine", Color::rgb(0x66, 0xCD, 0xAA)),
    ("mediumblue", Color::rgb(0x00, 0x00, 0xCD)),
    ("mediumorchid", Color::rgb(0xBA, 0x55, 0xD3)),
    ("mediumpurple", Color::rgb(0x93, 0x70, 0xDB)),
    ("mediumseagreen", Color::rgb(0x3C, 0xB3, 0x71)),
    ("mediumslateblue", Color::rgb(0x7B, 0x68, 0xEE)),
    ("mediumspringgreen", Color::rgb(0x00, 0xFA, 0x9A)),
    ("mediumturquoise", Color::rgb(0x48, 0xD1, 0xCC)),
    ("mediumvioletred", Color::rgb(0xC7, 0x15, 0x85)),
    ("midnightblue", Color::rgb(0x19, 0x19, 0x70)),
    ("mintcream", Color::rgb(0xF5, 0xFF, 0xFA)),
    ("mistyrose", Color::rgb(0xFF, 0xE4, 0xE1)),
    ("moccasin", Color::rgb(0xFF, 0xE4, 0xB5)),
    ("navajowhite", Color::rgb(0xFF, 0xDE, 0xAD)),
    ("navy", Color::rgb(0x00, 0x00, 0x80)),
    ("oldlace", Color::rgb(0xFD, 0xF5, 0xE6)),
    ("olive", Color::rgb(0x80, 0x80, 0x00)),
    ("olivedrab", Color::rgb(0x6B, 0x8E, 0x23)),
    ("orange", Color::rgb(0xFF, 0xA5, 0x00)),
    ("orangered", Color::rgb(0xFF, 0x45, 0x00)),
    ("orchid", Color::rgb(0xDA, 0x70, 0xD6)),
    ("palegoldenrod", Color::rgb(0xEE, 0xE8, 0xAA)),
    ("palegreen", Color::rgb(0x98, 0xFB, 0x98)),
    ("paleturquoise", Color::rgb(0xAF, 0xEE, 0xEE)),
    ("palevioletred", Color::rgb(0xDB, 0x70, 0x93)),
    ("papayawhip", Color::rgb(0xFF, 0xEF, 0xD5)),
    ("peachpuff", Color::rgb(0xFF, 0xDA, 0xB9)),
    ("peru", Color::rgb(0xCD, 0x85, 0x3F)),
    ("pink", Color::rgb(0xFF, 0xC0, 0xCB)),
    ("plum", Color::rgb(0xDD, 0xA0, 0xDD)),
    ("powderblue", Color::rgb(0xB0, 0xE0, 0xE6)),
    ("purple", Color::rgb(0x80, 0x00, 0x80)),
    ("rebeccapurple", Color::rgb(0x66, 0x33, 0x99)),
    ("red", Color::rgb(0xFF, 0x00, 0x00)),
    ("rosybrown", Color::rgb(0xBC, 0x8F, 0x8F)),
    ("royalblue", Color::rgb(0x41, 0x69, 0xE1)),
    ("saddlebrown", Color::rgb(0x8B, 0x45, 0x13)),
    ("salmon", Color::rgb(0xFA, 0x80, 0x72)),
    ("sandybrown", Color::rgb(0xF4, 0xA4, 0x60)),
    ("seagreen", Color::rgb(0x2E, 0x8B, 0x57)),
    ("seashell", Color::rgb(0xFF, 0xF5, 0xEE)),
    ("sienna", Color::rgb(0xA0, 0x52, 0x2D)),
    ("silver", Color::rgb(0xC0, 0xC0, 0xC0)),
    ("skyblue", Color::rgb(0x87, 0xCE, 0xEB)),
    ("slateblue", Color::rgb(0x6A, 0x5A, 0xCD)),
    ("slategray", Color::rgb(0x70, 0x80, 0x90)),
    ("slategrey", Color::rgb(0x70, 0x80, 0x90)),
    ("snow", Color::rgb(0xFF, 0xFA, 0xFA)),
    ("springgreen", Color::rgb(0x00, 0xFF, 0x7F)),
    ("steelblue", Color::rgb(0x46, 0x82, 0xB4)),
    ("tan", Color::rgb(0xD2, 0xB4, 0x8C)),
    ("teal", Color::rgb(0x00, 0x80, 0x80)),
    ("thistle", Color::rgb(0xD8, 0xBF, 0xD8)),
    ("tomato", Color::rgb(0xFF, 0x63, 0x47)),
    ("transparent", Color::rgba(0x00, 0x00, 0x00, 0x00)),
    ("turquoise", Color::rgb(0x40, 0xE0, 0xD0)),
    ("violet", Color::rgb(0xEE, 0x82, 0xEE)),
    ("wheat", Color::rgb(0xF5, 0xDE, 0xB3)),
    ("white", Color::rgb(0xFF, 0xFF, 0xFF)),
    ("whitesmoke", Color::rgb(0xF5, 0xF5, 0xF5)),
    ("yellow", Color::rgb(0xFF, 0xFF, 0x00)),
    ("yellowgreen", Color::rgb(0x9A, 0xCD, 0x32)),
];

fn named_color(name: &str) -> Option<Color> {
    NAMED_COLORS
        .binary_search_by(|(candidate, _)| cmp_ascii_ignore_case(candidate, name))
        .ok()
        .map(|index| NAMED_COLORS[index].1)
}

/// Byte-wise ASCII case-insensitive comparison, used to binary-search
/// [`NAMED_COLORS`] without allocating a lowercased copy of either side.
fn cmp_ascii_ignore_case(a: &str, b: &str) -> Ordering {
    a.bytes()
        .map(|c| c.to_ascii_lowercase())
        .cmp(b.bytes().map(|c| c.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- named-color table integrity ----

    #[test]
    fn named_color_table_has_exactly_149_entries() {
        assert_eq!(NAMED_COLORS.len(), 149);
    }

    #[test]
    fn named_color_table_is_strictly_ascending() {
        assert!(
            NAMED_COLORS.windows(2).all(|w| w[0].0 < w[1].0),
            "table must be sorted for binary_search_by to work"
        );
    }

    #[test]
    fn every_table_entry_is_findable_by_name() {
        for &(name, color) in NAMED_COLORS {
            assert_eq!(
                named_color(name),
                Some(color),
                "entry {name:?} must be findable via its own key"
            );
            let upper: String = name.to_uppercase();
            assert_eq!(
                named_color(&upper),
                Some(color),
                "entry {name:?} must be findable case-insensitively"
            );
        }
    }

    #[test]
    fn rebeccapurple_and_transparent_sort_where_expected() {
        // A wrong splice point here would silently break binary search for
        // neighboring entries without every_table_entry_is_findable_by_name
        // necessarily catching it if the splice still happened to preserve
        // local order; assert the exact neighbors directly too.
        let names: Vec<&str> = NAMED_COLORS.iter().map(|(n, _)| *n).collect();
        let reb = names
            .iter()
            .position(|&n| n == "rebeccapurple")
            .expect("present");
        assert_eq!(names[reb - 1], "purple");
        assert_eq!(names[reb + 1], "red");
        let transparent = names
            .iter()
            .position(|&n| n == "transparent")
            .expect("present");
        assert_eq!(names[transparent - 1], "tomato");
        assert_eq!(names[transparent + 1], "turquoise");
    }

    #[test]
    fn gray_and_grey_spellings_are_aliases_with_the_same_value() {
        for (american, british) in [
            ("gray", "grey"),
            ("darkgray", "darkgrey"),
            ("dimgray", "dimgrey"),
            ("lightgray", "lightgrey"),
            ("lightslategray", "lightslategrey"),
            ("slategray", "slategrey"),
            ("darkslategray", "darkslategrey"),
        ] {
            assert_eq!(Color::named(american), Color::named(british));
        }
    }

    #[test]
    fn aqua_cyan_and_fuchsia_magenta_are_synonym_pairs() {
        assert_eq!(Color::named("aqua"), Color::named("cyan"));
        assert_eq!(Color::named("fuchsia"), Color::named("magenta"));
    }

    #[test]
    fn named_lookup_is_case_insensitive() {
        assert_eq!(Color::named("Red"), Some(Color::rgb(0xFF, 0, 0)));
        assert_eq!(
            Color::named("REBECCAPURPLE"),
            Some(Color::rgb(0x66, 0x33, 0x99))
        );
        assert_eq!(Color::named("ToMaTo"), Some(Color::rgb(0xFF, 0x63, 0x47)));
    }

    #[test]
    fn unknown_name_is_none() {
        assert_eq!(Color::named("notacolor"), None);
        assert_eq!(Color::named(""), None);
    }

    #[test]
    fn transparent_keyword_resolves_to_zero_alpha_black() {
        assert_eq!(Color::named("transparent"), Some(Color::TRANSPARENT));
        assert_eq!(Color::TRANSPARENT.a(), 0);
    }

    // ---- hex notation ----

    #[test]
    fn hex_forms_3_4_6_8_are_all_accepted() {
        assert_eq!(Color::parse("#abc"), Ok(Color::rgb(0xAA, 0xBB, 0xCC)));
        assert_eq!(
            Color::parse("#abc1"),
            Ok(Color::rgba(0xAA, 0xBB, 0xCC, 0x11))
        );
        assert_eq!(Color::parse("#aabbcc"), Ok(Color::rgb(0xAA, 0xBB, 0xCC)));
        assert_eq!(
            Color::parse("#aabbccdd"),
            Ok(Color::rgba(0xAA, 0xBB, 0xCC, 0xDD))
        );
    }

    #[test]
    fn hex_is_case_insensitive() {
        assert_eq!(Color::parse("#ABC"), Color::parse("#abc"));
        assert_eq!(Color::parse("#FF00FF"), Ok(Color::rgb(0xFF, 0x00, 0xFF)));
    }

    #[test]
    fn hex_wrong_digit_counts_are_rejected_not_padded_or_truncated() {
        // Every digit string here is composed entirely of valid hex
        // characters, so the length is the only thing under test — a bad
        // trailing *non*-hex character (like the 'g' in "#abcdefg") would
        // instead be a shorter valid color followed by `TrailingData`,
        // which is a different, also-correct outcome this test is not
        // about.
        for bad in ["#", "#a", "#ab", "#abcde", "#1234567", "#123456789"] {
            let result = Color::parse(bad);
            assert!(
                matches!(result, Err(ValueError::InvalidNumber { offset: 0 })),
                "{bad:?} should be InvalidNumber at offset 0, got {result:?}"
            );
        }
    }

    #[test]
    fn hex_with_a_non_hex_trailing_character_is_trailing_data_not_invalid_number() {
        // "#abcdef" is a complete, valid 6-digit color; the "g" after it is
        // simply unparsed trailing content, not part of a malformed hex run.
        assert_eq!(
            Color::parse("#abcdefg"),
            Err(ValueError::TrailingData { offset: 7 })
        );
    }

    #[test]
    fn hex_five_digit_string_is_rejected() {
        // Explicit regression for the exact case named in the spec catalog's
        // nasty-edge-case list (§9.3 #17).
        assert_eq!(
            Color::parse("#12345"),
            Err(ValueError::InvalidNumber { offset: 0 })
        );
    }

    // ---- rgb()/rgba() ----

    #[test]
    fn rgb_legacy_integer_syntax() {
        assert_eq!(Color::parse("rgb(255,0,0)"), Ok(Color::rgb(255, 0, 0)));
        assert_eq!(Color::parse("rgb(255, 0, 0)"), Ok(Color::rgb(255, 0, 0)));
        // The alpha component of rgb()/rgba() is always a number in 0..=1 or
        // a percentage — never a 0-255 integer, even in the "legacy" comma
        // form. A bare `128` here is a valid alpha *number*, which clamps to
        // 1.0 (opaque), matching CSS's clamp-don't-error rule.
        assert_eq!(
            Color::parse("rgba(0,128,255,0.5)"),
            Ok(Color::rgba(0, 128, 255, 128))
        );
        assert_eq!(
            Color::parse("rgba(0,128,255,128)"),
            Ok(Color::rgba(0, 128, 255, 255))
        );
    }

    #[test]
    fn rgb_legacy_percentage_syntax() {
        assert_eq!(Color::parse("rgb(100%,0%,0%)"), Ok(Color::rgb(255, 0, 0)));
        assert_eq!(
            Color::parse("rgb(50%,50%,50%)"),
            Ok(Color::rgb(128, 128, 128))
        );
    }

    #[test]
    fn rgb_legacy_mixing_int_and_percent_is_rejected() {
        // Spec catalog §9.3 #18.
        let result = Color::parse("rgb(100%, 128, 0)");
        assert!(
            matches!(result, Err(ValueError::UnexpectedChar { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn rgb_modern_space_syntax_with_slash_alpha() {
        // Spec catalog §9.3 #18.
        assert_eq!(
            Color::parse("rgb(255 128 0 / 50%)"),
            Ok(Color::rgba(255, 128, 0, 128))
        );
        assert_eq!(Color::parse("rgb(255 128 0)"), Ok(Color::rgb(255, 128, 0)));
    }

    #[test]
    fn rgb_modern_syntax_rejects_a_zero_width_separator() {
        // There is no number that could immediately follow another number
        // with zero separator without simply extending it, so exercise the
        // boundary via a percent sign followed directly by a digit.
        let result = Color::parse("rgb(50%0% 0%)");
        assert!(result.is_err());
    }

    #[test]
    fn rgb_does_not_mix_legacy_and_modern_separators() {
        assert!(Color::parse("rgb(255,0,0/50%)").is_err());
        assert!(Color::parse("rgb(255 0 0, 50%)").is_err());
    }

    #[test]
    fn rgb_out_of_range_components_clamp_rather_than_error() {
        assert_eq!(Color::parse("rgb(300,0,0)"), Ok(Color::rgb(255, 0, 0)));
        assert_eq!(Color::parse("rgb(-10,0,0)"), Ok(Color::rgb(0, 0, 0)));
        assert_eq!(
            Color::parse("rgb(0,0,0,150%)"),
            Ok(Color::rgba(0, 0, 0, 255))
        );
    }

    #[test]
    fn rgb_and_rgba_names_are_interchangeable() {
        assert_eq!(Color::parse("rgb(1,2,3,4)"), Color::parse("rgba(1,2,3,4)"));
        assert_eq!(Color::parse("rgba(1,2,3)"), Color::parse("rgb(1,2,3)"));
    }

    #[test]
    fn rgb_wrong_channel_count_is_rejected() {
        assert!(Color::parse("rgb(1,2)").is_err());
        assert!(Color::parse("rgb(1,2,3,4,5)").is_err());
        assert!(Color::parse("rgb(1 2)").is_err());
    }

    #[test]
    fn rgb_unclosed_paren_is_rejected() {
        let result = Color::parse("rgb(1,2,3");
        assert!(matches!(
            result,
            Err(ValueError::UnexpectedEnd { .. }) | Err(ValueError::UnexpectedChar { .. })
        ));
    }

    // ---- hsl()/hsla() ----

    #[test]
    fn hsl_known_primary_conversions() {
        assert_eq!(Color::parse("hsl(0,100%,50%)"), Ok(Color::rgb(255, 0, 0)));
        assert_eq!(Color::parse("hsl(120,100%,50%)"), Ok(Color::rgb(0, 255, 0)));
        assert_eq!(Color::parse("hsl(240,100%,50%)"), Ok(Color::rgb(0, 0, 255)));
    }

    #[test]
    fn hsl_lightness_extremes_and_zero_saturation() {
        assert_eq!(
            Color::parse("hsl(0,0%,100%)"),
            Ok(Color::rgb(255, 255, 255))
        );
        assert_eq!(Color::parse("hsl(0,0%,0%)"), Ok(Color::rgb(0, 0, 0)));
        assert_eq!(Color::parse("hsl(0,0%,50%)"), Ok(Color::rgb(128, 128, 128)));
    }

    #[test]
    fn hsl_modern_space_syntax() {
        assert_eq!(
            Color::parse("hsl(120 100% 50% / 50%)"),
            Ok(Color::rgba(0, 255, 0, 128))
        );
        assert_eq!(
            Color::parse("hsla(240 100% 50%)"),
            Ok(Color::rgb(0, 0, 255))
        );
    }

    #[test]
    fn hsl_hue_accepts_all_angle_units() {
        let deg = Color::parse("hsl(120,100%,50%)").expect("valid");
        assert_eq!(Color::parse("hsl(120deg,100%,50%)"), Ok(deg));
        assert_eq!(Color::parse("hsl(133.333grad,100%,50%)"), Ok(deg));
        let rad = std::f64::consts::PI * 2.0 / 3.0; // 120 degrees in radians
        assert_eq!(Color::parse(&format!("hsl({rad}rad,100%,50%)")), Ok(deg));
        assert_eq!(Color::parse("hsl(0.3333turn,100%,50%)"), Ok(deg));
    }

    #[test]
    fn hsl_negative_and_overflowing_hue_wrap() {
        assert_eq!(
            Color::parse("hsl(-240,100%,50%)"),
            Color::parse("hsl(120,100%,50%)")
        );
        assert_eq!(
            Color::parse("hsl(480,100%,50%)"),
            Color::parse("hsl(120,100%,50%)")
        );
    }

    #[test]
    fn hsl_saturation_and_lightness_require_percent_sign() {
        let result = Color::parse("hsl(0,100,50%)");
        assert!(matches!(result, Err(ValueError::UnexpectedChar { .. })));
    }

    #[test]
    fn hsl_out_of_range_percentages_clamp() {
        assert_eq!(Color::parse("hsl(0,150%,50%)"), Ok(Color::rgb(255, 0, 0)));
        assert_eq!(
            Color::parse("hsl(0,-50%,50%)"),
            Ok(Color::rgb(128, 128, 128))
        );
    }

    #[test]
    fn hsl_unknown_hue_unit_is_rejected() {
        let result = Color::parse("hsl(10xyz,100%,50%)");
        assert!(matches!(result, Err(ValueError::UnknownKeyword { .. })));
    }

    // ---- deliberately unimplemented functions ----

    #[test]
    fn hwb_lab_lch_oklab_oklch_color_are_rejected_as_unknown() {
        for (input, name) in [
            ("hwb(120 10% 10%)", "hwb"),
            ("lab(50% 40 30)", "lab"),
            ("lch(50% 40 30)", "lch"),
            ("oklab(0.5 0.1 0.1)", "oklab"),
            ("oklch(0.5 0.1 30)", "oklch"),
            ("color(display-p3 1 0 0)", "color"),
        ] {
            let result = Color::parse(input);
            assert_eq!(
                result,
                Err(ValueError::UnknownKeyword {
                    offset: 0,
                    keyword: name.into(),
                }),
                "input {input:?}"
            );
        }
    }

    #[test]
    fn arbitrary_unknown_function_name_is_rejected() {
        assert!(matches!(
            Color::parse("notacolor(1,2,3)"),
            Err(ValueError::UnknownKeyword { .. })
        ));
    }

    // ---- general grammar / error handling ----

    #[test]
    fn empty_and_whitespace_only_values_are_empty_errors() {
        assert_eq!(Color::parse(""), Err(ValueError::Empty));
        assert_eq!(Color::parse("   "), Err(ValueError::Empty));
    }

    #[test]
    fn surrounding_whitespace_is_tolerated() {
        assert_eq!(Color::parse("  red  "), Ok(Color::rgb(255, 0, 0)));
        assert_eq!(Color::parse("  #fff  "), Ok(Color::WHITE));
    }

    #[test]
    fn trailing_data_after_a_complete_value_is_reported() {
        // Exercises the icc-color-annotation case documented as unsupported
        // in the module doc: it must fail predictably, not silently drop
        // data or panic.
        let result = Color::parse("blue icc-color(foo 0 0 0)");
        assert!(
            matches!(result, Err(ValueError::TrailingData { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn a_bare_word_that_is_not_a_color_is_unknown_keyword() {
        assert_eq!(
            Color::parse("notacolor"),
            Err(ValueError::UnknownKeyword {
                offset: 0,
                keyword: "notacolor".into(),
            })
        );
    }

    #[test]
    fn a_leading_character_that_cannot_start_any_color_is_unexpected() {
        assert_eq!(
            Color::parse("!red"),
            Err(ValueError::UnexpectedChar {
                offset: 0,
                found: '!'
            })
        );
    }

    #[test]
    fn function_name_with_intervening_whitespace_before_the_paren_is_rejected() {
        // CSS function tokens have no whitespace between the name and `(`;
        // "rgb (...)" is not a function call, and "rgb" alone is not a
        // named color either.
        let result = Color::parse("rgb (255,0,0)");
        assert!(matches!(result, Err(ValueError::UnknownKeyword { .. })));
    }

    // ---- Display / round-trip ----

    #[test]
    fn to_hex_string_omits_alpha_when_opaque() {
        assert_eq!(Color::rgb(0x11, 0x22, 0x33).to_hex_string(), "#112233");
        assert_eq!(
            Color::rgba(0x11, 0x22, 0x33, 0x44).to_hex_string(),
            "#11223344"
        );
    }

    #[test]
    fn display_matches_to_hex_string() {
        let c = Color::rgba(1, 2, 3, 4);
        assert_eq!(c.to_string(), c.to_hex_string());
    }

    #[test]
    fn round_trip_through_display_and_parse_preserves_the_value() {
        for c in [
            Color::BLACK,
            Color::WHITE,
            Color::TRANSPARENT,
            Color::rgb(0x12, 0x34, 0x56),
            Color::rgba(0xAB, 0xCD, 0xEF, 0x01),
            Color::named("rebeccapurple").expect("present"),
        ] {
            let text = c.to_string();
            assert_eq!(Color::parse(&text), Ok(c), "round-trip of {text:?}");
        }
    }

    #[test]
    fn from_str_matches_parse() {
        assert_eq!("red".parse::<Color>(), Color::parse("red"));
        assert_eq!("#zzz".parse::<Color>(), Color::parse("#zzz"));
    }

    // ---- parse_from_cursor composition (used by paint.rs) ----

    #[test]
    fn parse_from_cursor_leaves_trailing_input_for_the_caller() {
        let mut cursor = Cursor::new("red rest");
        let color = parse_from_cursor(&mut cursor).expect("valid color");
        assert_eq!(color, Color::rgb(255, 0, 0));
        assert_eq!(cursor.rest(), " rest");
    }
}
