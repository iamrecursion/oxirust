//! SMIL clock values and animation timing specifiers.
//!
//! This module covers the microsyntaxes specification catalog §3.13
//! transcribes: the `Clock-value` grammar shared by `dur`, `min`, `max`,
//! and every `begin`/`end` offset, `repeatCount`, and the `begin-value` /
//! `end-value` timing-specifier grammar that `begin` and `end` lists are
//! built from.
//!
//! # Clock values
//!
//! ```text
//! Clock-value         ::= Full-clock-value | Partial-clock-value | Timecount-value
//! Full-clock-value    ::= Hours ":" Minutes ":" Seconds ("." Fraction)?
//! Partial-clock-value ::= Minutes ":" Seconds ("." Fraction)?
//! Timecount-value     ::= Timecount ("." Fraction)? Metric?
//! Metric              ::= "h" | "min" | "s" | "ms"
//! ```
//!
//! [`ClockValue::parse`] accepts all three forms (`"01:30:05.5"`,
//! `"30:05"`, `"3.5s"`, `"90min"`, `"250ms"`, and a bare `"5"` — which the
//! grammar defines as seconds). `Hours` and `Minutes` accept any number of
//! leading digits (not just two), and `Minutes`/`Seconds` are *not*
//! range-checked against 0–59, matching the grammar's own tolerance
//! ("conventionally 2-digit but grammar doesn't hard-require it"). A
//! leading sign is **never** accepted here — only the timing grammar's
//! `offset-value` is signed (see [`SignedClockValue`]), never a bare clock
//! value.
//!
//! [`ClockValue`] stores the value's total seconds (as an `f64`, via
//! [`ClockValue::as_seconds`]) plus just enough information to
//! [`Display`](fmt::Display) it back out reasonably — which of the three
//! forms it was written in, and (for the timecount form) which metric.
//! This is **canonical re-emission, not exact round-tripping**: `"90min"`
//! re-renders as `"90min"`, but the partial form `"1:2"` (`Partial-clock-value`,
//! one minute two seconds) re-renders in full `Full-clock-value` form as
//! `"0:01:02"` — no crate policy requires byte-for-byte fidelity, only
//! that re-parsing the rendered text yields an equal value
//! (`ClockValue::parse(&v.to_string())?.as_seconds() == v.as_seconds()`,
//! up to ordinary floating-point unit-conversion rounding — converting
//! `250ms` to seconds and back to milliseconds for display is not
//! guaranteed to be bit-exact, since neither `1000.0` nor `0.001` is a
//! power of two).
//!
//! # `repeatCount` and `dur`
//!
//! [`RepeatCount::parse`] accepts `"indefinite"` or a plain `<number>`
//! (reusing [`crate::values::lex::parse_number`]) that must be strictly
//! greater than zero. [`ClockDuration::parse`] (also reachable as the free
//! function [`parse_repeat_dur`], matching the name this module's mission
//! used) accepts a [`ClockValue`], `"indefinite"`, or `"media"`.
//! `"media"` is not in the mission that commissioned this module, but is
//! in the grammar the catalog transcribes for `dur` (`Clock-value |
//! "media" | "indefinite"`); it is included here as a deliberate,
//! documented deviation toward matching the specification.
//!
//! # Timing specifiers
//!
//! ```text
//! begin-value-list     ::= begin-value (wsp* ";" wsp* begin-value-list)?
//! begin-value          ::= offset-value | syncbase-value | event-value | repeat-value
//!                         | accessKey-value | wallclock-sync-value | "indefinite"
//! offset-value         ::= ("+"|"-")? Clock-value
//! syncbase-value       ::= Id-value "." ("begin"|"end") (("+"|"-") Clock-value)?
//! event-value          ::= (Id-value ".")? event-ref (("+"|"-") Clock-value)?
//! repeat-value         ::= (Id-value ".")? "repeat(" integer ")" (("+"|"-") Clock-value)?
//! accessKey-value      ::= "accessKey(" character ")" (("+"|"-") Clock-value)?
//! wallclock-sync-value ::= "wallclock(" wallclock-value ")"
//! ```
//!
//! [`TimingSpecifier::parse`] parses one item of this grammar (`end` reuses
//! the identical `begin-value-list` grammar); [`parse_timing_list`] parses
//! a whole semicolon-separated `begin`/`end` attribute value, trimming
//! white space around each item.
//!
//! ## Disambiguating `id.begin` / `id.end` / `id.repeat(n)` / `id.event`
//!
//! All four forms share the shape `id "." suffix`. This module splits on
//! the **last** `.` in the token (XML ids may themselves legitimately
//! contain `.`), then checks whether `suffix` is exactly `"begin"`,
//! exactly `"end"`, or a `"repeat(" integer ")"` call — each optionally
//! followed by a signed offset with no space before its sign. If none of
//! those match, the whole thing is an event-value: `suffix` is the event
//! name. This is the one place the grammar is genuinely ambiguous when
//! *both* an id and an event name contain dots (`"a.b.c.d"` could be
//! `id="a.b.c"` on event `"d"`, or `id="a.b"` on event `"c.d"`, or...); this
//! module always resolves it as "the last dot separates id from suffix,"
//! documented here rather than silently guessed at per call site.
//! `"indefinite"` always wins over an event coincidentally named
//! `"indefinite"`, since it is checked first and matched as a whole token.
//!
//! Event names ([`TimingSpecifier::Event::event`]) are validated only
//! loosely: any run of ASCII letters, digits, and `_` is accepted (no
//! fixed list of DOM event names is enforced), matching what the mission
//! that commissioned this module asked for. This does **not** include `-`,
//! so a hyphenated custom event name is not representable — a deliberate
//! simplification that also keeps `"id.end-2s"` unambiguous (`end` then a
//! `-2s` offset, not an event literally named `"end-2s"`).
//!
//! [`TimingSpecifier::Wallclock`] preserves its parenthesized payload
//! verbatim as text, without attempting to parse it as an ISO 8601
//! date-time — wall-clock synchronization needs a real calendar/timezone
//! implementation this crate does not have yet, so that parsing is
//! deferred to a future release; the raw payload is still there for a
//! caller that wants to parse it itself.
//!
//! Backslash escapes inside an id (SMIL nominally allows escaping a
//! literal `.` in an id with `\.`) are **not** supported; an id is always
//! read as a plain run of non-`.` characters.

use crate::error::ValueError;
use crate::values::lex::{is_wsp, parse_number};
use std::fmt;

/// The metric suffix (if any) a [`ClockValue`] was written with in its
/// `Timecount-value` form. `None` is the bare, unit-less form (`"5"`),
/// which the grammar defines as seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Metric {
    /// No metric suffix: a bare number, meaning seconds.
    None,
    /// `h`: hours.
    Hours,
    /// `min`: minutes.
    Minutes,
    /// `s`: seconds, written explicitly.
    Seconds,
    /// `ms`: milliseconds.
    Milliseconds,
}

impl Metric {
    /// The number of seconds one unit of this metric represents.
    const fn scale(self) -> f64 {
        match self {
            Self::None | Self::Seconds => 1.0,
            Self::Milliseconds => 0.001,
            Self::Minutes => 60.0,
            Self::Hours => 3600.0,
        }
    }

    /// The literal suffix text this metric renders as.
    const fn suffix(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Hours => "h",
            Self::Minutes => "min",
            Self::Seconds => "s",
            Self::Milliseconds => "ms",
        }
    }
}

/// Which of the three `Clock-value` forms a [`ClockValue`] was parsed from,
/// kept only so [`Display`](fmt::Display) can re-render it reasonably.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockForm {
    /// `Full-clock-value` or `Partial-clock-value`: `hh:mm:ss(.frac)?` or
    /// `mm:ss(.frac)?`. Always re-rendered in full `h:mm:ss(.frac)?` form.
    Clock,
    /// `Timecount-value`, with the metric it was written with.
    Timecount(Metric),
}

/// A parsed SMIL clock value: a non-negative duration in seconds.
///
/// See the [module documentation](self) for the grammar this accepts and
/// this type's canonical-re-emission `Display` policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockValue {
    seconds: f64,
    form: ClockForm,
}

impl ClockValue {
    /// Parses one clock value: `hh:mm:ss(.frac)?`, `mm:ss(.frac)?`, or a
    /// timecount with an optional `h`/`min`/`s`/`ms` metric (bare, with no
    /// metric, means seconds). Surrounding white space is allowed. Never
    /// accepts a leading sign — see [`SignedClockValue::parse`] for that.
    ///
    /// ```
    /// use oxixml_svg::values::clock::ClockValue;
    ///
    /// assert_eq!(ClockValue::parse("3.5s").expect("valid").as_seconds(), 3.5);
    /// assert_eq!(ClockValue::parse("90min").expect("valid").as_seconds(), 5400.0);
    /// assert_eq!(ClockValue::parse("250ms").expect("valid").as_seconds(), 0.25);
    /// assert_eq!(ClockValue::parse("5").expect("bare number is seconds").as_seconds(), 5.0);
    /// assert_eq!(ClockValue::parse("01:30:05").expect("full form").as_seconds(), 5405.0);
    /// assert!(ClockValue::parse("-5s").is_err(), "a bare clock value is never signed");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ValueError`] if `input` (after trimming white space) is
    /// empty, does not match any of the three forms, or has trailing
    /// content after a complete value.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let leading_trim = input.len() - input.trim_start_matches(is_wsp).len();
        let trimmed = input.trim_matches(is_wsp);
        if trimmed.is_empty() {
            return Err(ValueError::Empty);
        }
        if trimmed.contains(':') {
            let seconds = parse_clock_form(trimmed, leading_trim)?;
            Ok(Self {
                seconds,
                form: ClockForm::Clock,
            })
        } else {
            let (seconds, metric) = parse_timecount_form(trimmed, leading_trim)?;
            Ok(Self {
                seconds,
                form: ClockForm::Timecount(metric),
            })
        }
    }

    /// Creates a clock value directly from a number of seconds, rendering
    /// it in explicit `"<seconds>s"` timecount form.
    #[must_use]
    pub const fn from_seconds(seconds: f64) -> Self {
        Self {
            seconds,
            form: ClockForm::Timecount(Metric::Seconds),
        }
    }

    /// The value in seconds.
    #[must_use]
    pub const fn as_seconds(&self) -> f64 {
        self.seconds
    }
}

impl fmt::Display for ClockValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.form {
            ClockForm::Clock => {
                let total_seconds = self.seconds.max(0.0);
                let whole_minutes = (total_seconds / 60.0).floor();
                // Kept in `[0, 60)`: rendered with plain `{}` `f64` Display
                // (shortest round-trip form) rather than a fixed decimal
                // precision, so this cannot introduce its own rounding
                // artifact the way truncating to N decimal places could.
                let seconds_component = total_seconds - whole_minutes * 60.0;
                let whole_minutes = whole_minutes as u64;
                let hours = whole_minutes / 60;
                let minutes = whole_minutes % 60;
                write!(f, "{hours}:{minutes:02}:")?;
                if seconds_component < 10.0 {
                    write!(f, "0{seconds_component}")
                } else {
                    write!(f, "{seconds_component}")
                }
            }
            ClockForm::Timecount(metric) => {
                let value = self.seconds / metric.scale();
                write!(f, "{value}{}", metric.suffix())
            }
        }
    }
}

/// Parses `s` (already validated as a non-empty run of ASCII digits, or
/// digits, a `.`, and more digits) as `f64`, rejecting a result that
/// overflowed to infinity.
///
/// `str::parse::<f64>` never errors on overflow, it only saturates (a long
/// enough all-digit run parses to `f64::INFINITY`), so without this check an
/// overlong `Hours`/`Minutes`/`Timecount` digit run would silently produce a
/// [`ClockValue`] whose `Display` renders the bare string `"inf"` — not a
/// valid SMIL clock-value token under any grammar production, and one that
/// fails to re-parse.
fn parse_finite_digits(s: &str, offset: usize) -> Result<f64, ValueError> {
    let value: f64 = s
        .parse()
        .expect("validated as an ASCII digit run, or digits '.' digits");
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ValueError::OutOfRange {
            offset,
            what: "clock value is too large to represent as a finite number of seconds",
        })
    }
}

/// Parses the digit run `s` as a plain non-negative integer, requiring
/// every byte to be an ASCII digit. Used for `Hours`, `Minutes`, and
/// `Timecount`, none of which accept a sign, a fraction, or an exponent.
fn parse_plain_digits(s: &str, offset: usize) -> Result<f64, ValueError> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ValueError::InvalidNumber { offset });
    }
    parse_finite_digits(s, offset)
}

/// Parses `s` as `Seconds ("." Fraction)?` or `Timecount ("." Fraction)?`:
/// a plain digit run, optionally followed by a `.` and *another*
/// non-empty plain digit run (the grammar's `Fraction` production requires
/// at least one digit — `"5."` is not a valid clock-value fraction, unlike
/// the more permissive SVG `<number>` grammar in [`crate::values::lex`]).
fn parse_digits_with_optional_fraction(s: &str, offset: usize) -> Result<f64, ValueError> {
    match s.split_once('.') {
        None => parse_plain_digits(s, offset),
        Some((int_part, frac_part)) => {
            if int_part.is_empty() || !int_part.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ValueError::InvalidNumber { offset });
            }
            if frac_part.is_empty() || !frac_part.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ValueError::InvalidNumber {
                    offset: offset + int_part.len() + 1,
                });
            }
            parse_finite_digits(s, offset)
        }
    }
}

/// Parses the `Full-clock-value | Partial-clock-value` forms of `text`
/// (which is known to contain at least one `:`), returning the total
/// number of seconds. `offset` is the byte offset of `text` within the
/// original input the caller was given, so every error this returns is
/// already relative to that original input.
fn parse_clock_form(text: &str, offset: usize) -> Result<f64, ValueError> {
    let first_colon = text.find(':').expect("caller checked text contains ':'");
    let after_first = &text[first_colon + 1..];
    match after_first.find(':') {
        None => {
            // Partial-clock-value: "mm:ss(.frac)?"
            let minutes = parse_plain_digits(&text[..first_colon], offset)?;
            let seconds =
                parse_digits_with_optional_fraction(after_first, offset + first_colon + 1)?;
            Ok(minutes * 60.0 + seconds)
        }
        Some(second_colon_rel) => {
            let second_colon = first_colon + 1 + second_colon_rel;
            let minutes_str = &text[first_colon + 1..second_colon];
            let seconds_str = &text[second_colon + 1..];
            if seconds_str.contains(':') {
                return Err(ValueError::TrailingData {
                    offset: offset + second_colon + 1,
                });
            }
            let hours = parse_plain_digits(&text[..first_colon], offset)?;
            let minutes = parse_plain_digits(minutes_str, offset + first_colon + 1)?;
            let seconds =
                parse_digits_with_optional_fraction(seconds_str, offset + second_colon + 1)?;
            Ok(hours * 3600.0 + minutes * 60.0 + seconds)
        }
    }
}

/// Parses the `Timecount-value` form of `text` (known to contain no `:`),
/// returning the total number of seconds and the [`Metric`] `text` was
/// written with. `offset` is `text`'s byte offset within the original
/// input, exactly as in [`parse_clock_form`].
fn parse_timecount_form(text: &str, offset: usize) -> Result<(f64, Metric), ValueError> {
    let digit_len = text.bytes().take_while(u8::is_ascii_digit).count();
    if digit_len == 0 {
        return Err(ValueError::InvalidNumber { offset });
    }
    let mut number_len = digit_len;
    if text.as_bytes().get(digit_len) == Some(&b'.') {
        let frac_len = text.as_bytes()[digit_len + 1..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if frac_len == 0 {
            return Err(ValueError::InvalidNumber {
                offset: offset + digit_len,
            });
        }
        number_len = digit_len + 1 + frac_len;
    }
    let value = parse_finite_digits(&text[..number_len], offset)?;
    let suffix = &text[number_len..];
    let (metric, consumed) = if suffix.starts_with("ms") {
        (Metric::Milliseconds, 2)
    } else if suffix.starts_with("min") {
        (Metric::Minutes, 3)
    } else if suffix.starts_with('h') {
        (Metric::Hours, 1)
    } else if suffix.starts_with('s') {
        (Metric::Seconds, 1)
    } else if suffix.is_empty() {
        (Metric::None, 0)
    } else {
        return Err(ValueError::TrailingData {
            offset: offset + number_len,
        });
    };
    if consumed != suffix.len() {
        return Err(ValueError::TrailingData {
            offset: offset + number_len + consumed,
        });
    }
    Ok((value * metric.scale(), metric))
}

/// Shifts every offset a [`ValueError`] carries by `by` bytes. Used
/// throughout this module to re-report an inner parser's error (which is
/// relative to the sub-slice it was handed) relative to a larger input a
/// caller is assembling that sub-slice from.
fn shift_offset(error: ValueError, by: usize) -> ValueError {
    match error {
        ValueError::UnexpectedEnd { offset } => ValueError::UnexpectedEnd {
            offset: offset + by,
        },
        ValueError::UnexpectedChar { offset, found } => ValueError::UnexpectedChar {
            offset: offset + by,
            found,
        },
        ValueError::InvalidNumber { offset } => ValueError::InvalidNumber {
            offset: offset + by,
        },
        ValueError::OutOfRange { offset, what } => ValueError::OutOfRange {
            offset: offset + by,
            what,
        },
        ValueError::UnknownKeyword { offset, keyword } => ValueError::UnknownKeyword {
            offset: offset + by,
            keyword,
        },
        ValueError::Empty => ValueError::Empty,
        ValueError::TrailingData { offset } => ValueError::TrailingData {
            offset: offset + by,
        },
    }
}

/// A [`ClockValue`] together with an optional sign: the timing grammar's
/// `offset-value ::= ("+"|"-")? Clock-value`, the only place a signed
/// clock value appears. [`ClockValue`] itself is always non-negative; a
/// leading sign is rejected there and modeled here instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedClockValue {
    magnitude: ClockValue,
    negative: bool,
}

impl SignedClockValue {
    /// Parses an optionally-signed clock value: an optional leading `+` or
    /// `-`, then a [`ClockValue`]. Surrounding white space is allowed.
    ///
    /// ```
    /// use oxixml_svg::values::clock::SignedClockValue;
    ///
    /// assert_eq!(SignedClockValue::parse("5s").expect("valid").as_seconds(), 5.0);
    /// assert_eq!(SignedClockValue::parse("+5s").expect("valid").as_seconds(), 5.0);
    /// assert_eq!(SignedClockValue::parse("-5s").expect("valid").as_seconds(), -5.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ValueError`] under the same conditions as
    /// [`ClockValue::parse`] applied to the text after the sign.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let leading_trim = input.len() - input.trim_start_matches(is_wsp).len();
        let trimmed = input.trim_matches(is_wsp);
        if trimmed.is_empty() {
            return Err(ValueError::Empty);
        }
        let (negative, rest) = match trimmed.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
        };
        let sign_len = trimmed.len() - rest.len();
        let magnitude = ClockValue::parse(rest)
            .map_err(|error| shift_offset(error, leading_trim + sign_len))?;
        Ok(Self {
            magnitude,
            negative,
        })
    }

    /// The signed value in seconds: negative when the value carried a
    /// leading `-`.
    #[must_use]
    pub const fn as_seconds(&self) -> f64 {
        if self.negative {
            -self.magnitude.seconds
        } else {
            self.magnitude.seconds
        }
    }

    /// The magnitude, ignoring the sign.
    #[must_use]
    pub const fn magnitude(&self) -> ClockValue {
        self.magnitude
    }

    /// True when the value carried a leading `-`.
    #[must_use]
    pub const fn is_negative(&self) -> bool {
        self.negative
    }
}

impl fmt::Display for SignedClockValue {
    /// Renders the sign only when negative (`"5s"`, not `"+5s"`) — see the
    /// [module documentation](self)'s canonical-re-emission policy.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negative {
            write!(f, "-{}", self.magnitude)
        } else {
            write!(f, "{}", self.magnitude)
        }
    }
}

/// A `repeatCount` attribute's value: how many times a timed element
/// repeats, or that it repeats forever.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepeatCount {
    /// A finite repeat count, always strictly greater than zero — see
    /// [`RepeatCount::parse`].
    Count(f64),
    /// `"indefinite"`: repeats without end.
    Indefinite,
}

impl RepeatCount {
    /// Parses a `repeatCount` value: `"indefinite"`, or a plain
    /// [`crate::values::lex::parse_number`] value that must be strictly
    /// greater than zero (a fractional repeat count, e.g. `"2.5"`, is
    /// valid SMIL: the element repeats twice, then plays half again).
    ///
    /// # Errors
    ///
    /// Returns a [`ValueError`] if `input` is not `"indefinite"` and does
    /// not parse as an SVG `<number>`, parses to a value that is zero or
    /// negative, or overflows to infinity (see [`Self::Count`]'s
    /// documentation).
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let trimmed = input.trim_matches(is_wsp);
        if trimmed == "indefinite" {
            return Ok(Self::Indefinite);
        }
        let value = parse_number(input)?;
        let leading_trim = input.len() - input.trim_start_matches(is_wsp).len();
        if !value.is_finite() {
            // `parse_number` documents that an overlong digit run saturates
            // to infinity rather than erroring; left unchecked, that would
            // produce a `RepeatCount` whose `Display` renders the bare
            // string `"inf"`, which is neither `"indefinite"` nor a valid
            // `<number>` and so cannot be re-parsed.
            return Err(ValueError::OutOfRange {
                offset: leading_trim,
                what: "repeatCount is too large to represent as a finite number",
            });
        }
        if value <= 0.0 {
            return Err(ValueError::OutOfRange {
                offset: leading_trim,
                what: "repeatCount must be greater than zero",
            });
        }
        Ok(Self::Count(value))
    }
}

impl fmt::Display for RepeatCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count(value) => write!(f, "{value}"),
            Self::Indefinite => f.write_str("indefinite"),
        }
    }
}

/// A `dur` attribute's value: a specific [`ClockValue`], `"media"` (the
/// referenced media's own intrinsic duration), or `"indefinite"`.
///
/// See the [module documentation](self) for why `Media` is present even
/// though it is not in the exact shape this module's originating mission
/// described.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockDuration {
    /// A specific duration.
    Clock(ClockValue),
    /// `"media"`: the intrinsic duration of the referenced media.
    Media,
    /// `"indefinite"`: no natural duration; lasts until explicitly ended.
    Indefinite,
}

impl ClockDuration {
    /// Parses a `dur` value: `"indefinite"`, `"media"`, or a
    /// [`ClockValue`].
    ///
    /// # Errors
    ///
    /// Returns a [`ValueError`] under the same conditions as
    /// [`ClockValue::parse`] when `input` is neither `"indefinite"` nor
    /// `"media"`.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let trimmed = input.trim_matches(is_wsp);
        match trimmed {
            "indefinite" => Ok(Self::Indefinite),
            "media" => Ok(Self::Media),
            _ => Ok(Self::Clock(ClockValue::parse(input)?)),
        }
    }
}

impl fmt::Display for ClockDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(clock) => write!(f, "{clock}"),
            Self::Media => f.write_str("media"),
            Self::Indefinite => f.write_str("indefinite"),
        }
    }
}

/// Parses a `dur` attribute value. A free function under the name this
/// module's originating mission specified; identical to
/// [`ClockDuration::parse`].
///
/// # Errors
///
/// See [`ClockDuration::parse`].
pub fn parse_repeat_dur(input: &str) -> Result<ClockDuration, ValueError> {
    ClockDuration::parse(input)
}

/// Which edge of a referenced element's timeline a
/// [`TimingSpecifier::Syncbase`] synchronizes with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncbaseEdge {
    /// Synchronizes with the referenced element's begin.
    Begin,
    /// Synchronizes with the referenced element's end.
    End,
}

impl fmt::Display for SyncbaseEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Begin => "begin",
            Self::End => "end",
        })
    }
}

/// One item of a `begin` or `end` attribute's semicolon-separated value
/// list — the `begin-value`/`end-value` grammar specification catalog
/// §3.13 gives, which the two attributes share structurally.
///
/// See the [module documentation](self) for how the `id.begin` /
/// `id.end` / `id.repeat(n)` / `id.event` forms are told apart.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TimingSpecifier {
    /// A plain, optionally signed offset from the document's (or
    /// containing time container's) begin: `"5s"`, `"+5s"`, `"-2s"`.
    Offset(SignedClockValue),
    /// A reference to another element's begin or end, optionally offset:
    /// `"otherElement.begin"`, `"otherElement.end+2s"`.
    Syncbase {
        /// The referenced element's id.
        id: Box<str>,
        /// Which of the referenced element's instance times this
        /// synchronizes with.
        edge: SyncbaseEdge,
        /// An offset applied after the synchronization point, if any.
        offset: Option<SignedClockValue>,
    },
    /// An event-based time, optionally scoped to a specific element and
    /// optionally offset: `"click"`, `"otherElement.load"`, `"click+2s"`.
    Event {
        /// The element the event must occur on, or [`None`] for the
        /// element the timed attribute itself belongs to.
        id: Option<Box<str>>,
        /// The event name, validated only loosely — see the [module
        /// documentation](self).
        event: Box<str>,
        /// An offset applied after the event fires, if any.
        offset: Option<SignedClockValue>,
    },
    /// A repeat event of a (possibly implicit) element's timed animation:
    /// `"repeat(2)"`, `"otherElement.repeat(3)+1s"`.
    Repeat {
        /// The element whose repeat event this is, or [`None`] for the
        /// element the timed attribute itself belongs to.
        id: Option<Box<str>>,
        /// Which repeat iteration triggers this time.
        iteration: u32,
        /// An offset applied after the repeat fires, if any.
        offset: Option<SignedClockValue>,
    },
    /// An access-key time: `"accessKey(a)"`, optionally offset.
    AccessKey {
        /// The access key character.
        key: char,
        /// An offset applied after the key is pressed, if any.
        offset: Option<SignedClockValue>,
    },
    /// A wall-clock time, `"wallclock(...)"`: the parenthesized payload,
    /// preserved verbatim rather than parsed as an ISO 8601 date-time —
    /// see the [module documentation](self) for why that is deferred.
    Wallclock(Box<str>),
    /// `"indefinite"`: unresolved until something else (e.g. a
    /// `beginElement()` call) resolves it.
    Indefinite,
}

/// Strips `prefix` from the start of `text` and finds the first `)` in
/// what remains, returning `(payload, remainder-after-the-close-paren)`.
/// None of `wallclock(...)`, `accessKey(...)`, or `repeat(...)`'s payloads
/// can themselves contain a `)`, so a first-match, non-nesting search is
/// exact for all three.
fn strip_call<'a>(text: &'a str, prefix: &str) -> Option<(&'a str, &'a str)> {
    let rest = text.strip_prefix(prefix)?;
    let close = rest.find(')')?;
    Some((&rest[..close], &rest[close + 1..]))
}

/// Matches `text` against exactly `keyword`, optionally followed
/// immediately by a sign (the start of a signed-offset suffix). Returns
/// the text after `keyword` (empty, or starting with `+`/`-`) on a match;
/// `None` if `text` does not start with `keyword`, or continues with
/// anything else (so `"beginner"` is not mistaken for `"begin"` plus
/// garbage — it falls through to being read as a plain event name).
fn match_keyword_with_optional_signed_suffix<'a>(text: &'a str, keyword: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(keyword)?;
    if rest.is_empty() || rest.starts_with(['+', '-']) {
        Some(rest)
    } else {
        None
    }
}

/// Splits `s` into a leading event-name run (ASCII letters, digits, and
/// `_` — see the [module documentation](self) for why `-` is excluded)
/// and whatever follows it.
fn split_event_and_offset(s: &str) -> (&str, &str) {
    let name_len = s
        .bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
        .count();
    s.split_at(name_len)
}

/// Parses an optional signed-offset suffix (the text after a recognized
/// keyword, id-separator, or function call): empty means no offset;
/// otherwise `rest` must start with `+` or `-`, which
/// [`SignedClockValue::parse`] itself requires. `position` is `rest`'s
/// byte offset within the original input.
fn parse_trailing_offset(
    rest: &str,
    position: usize,
) -> Result<Option<SignedClockValue>, ValueError> {
    if rest.is_empty() {
        return Ok(None);
    }
    if rest.starts_with(['+', '-']) {
        let signed =
            SignedClockValue::parse(rest).map_err(|error| shift_offset(error, position))?;
        Ok(Some(signed))
    } else {
        let found = rest
            .chars()
            .next()
            .expect("rest is non-empty, checked above");
        Err(ValueError::UnexpectedChar {
            offset: position,
            found,
        })
    }
}

/// Parses `payload` (the text between `repeat(` and `)`) as the iteration
/// number a [`TimingSpecifier::Repeat`] fires on.
fn parse_repeat_iteration(payload: &str, offset: usize) -> Result<u32, ValueError> {
    if payload.is_empty() || !payload.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ValueError::InvalidNumber { offset });
    }
    payload.parse::<u32>().map_err(|_| ValueError::OutOfRange {
        offset,
        what: "repeat() iteration must fit in a 32-bit unsigned integer",
    })
}

impl TimingSpecifier {
    /// Parses one item of a `begin` or `end` attribute's value list — not
    /// the whole semicolon-separated list; see [`parse_timing_list`] for
    /// that. Surrounding white space is allowed.
    ///
    /// ```
    /// use oxixml_svg::values::clock::{SyncbaseEdge, TimingSpecifier};
    ///
    /// assert_eq!(TimingSpecifier::parse("indefinite"), Ok(TimingSpecifier::Indefinite));
    /// assert!(matches!(
    ///     TimingSpecifier::parse("click").expect("valid"),
    ///     TimingSpecifier::Event { id: None, .. }
    /// ));
    /// let syncbase = TimingSpecifier::parse("box1.end+2s").expect("valid");
    /// match syncbase {
    ///     TimingSpecifier::Syncbase { id, edge, offset } => {
    ///         assert_eq!(&*id, "box1");
    ///         assert_eq!(edge, SyncbaseEdge::End);
    ///         assert_eq!(offset.expect("has an offset").as_seconds(), 2.0);
    ///     }
    ///     other => panic!("expected a Syncbase, got {other:?}"),
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ValueError`] if `input` (after trimming white space) is
    /// empty or does not match any alternative of the `begin-value`
    /// grammar.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let leading = input.len() - input.trim_start_matches(is_wsp).len();
        let trimmed = input.trim_matches(is_wsp);
        if trimmed.is_empty() {
            return Err(ValueError::Empty);
        }
        if trimmed == "indefinite" {
            return Ok(Self::Indefinite);
        }
        if let Some((payload, remainder)) = strip_call(trimmed, "wallclock(") {
            if !remainder.is_empty() {
                return Err(ValueError::TrailingData {
                    offset: leading + (trimmed.len() - remainder.len()),
                });
            }
            if payload.is_empty() {
                return Err(ValueError::OutOfRange {
                    offset: leading + "wallclock(".len(),
                    what: "wallclock() requires a date-time payload",
                });
            }
            return Ok(Self::Wallclock(payload.into()));
        }
        if let Some((payload, remainder)) = strip_call(trimmed, "accessKey(") {
            let mut chars = payload.chars();
            let key_offset = leading + "accessKey(".len();
            let key = chars.next().ok_or(ValueError::OutOfRange {
                offset: key_offset,
                what: "accessKey() requires exactly one character",
            })?;
            if chars.next().is_some() {
                return Err(ValueError::OutOfRange {
                    offset: key_offset,
                    what: "accessKey() requires exactly one character",
                });
            }
            let offset_position = leading + (trimmed.len() - remainder.len());
            let offset = parse_trailing_offset(remainder, offset_position)?;
            return Ok(Self::AccessKey { key, offset });
        }
        if let Some((payload, remainder)) = strip_call(trimmed, "repeat(") {
            let iteration = parse_repeat_iteration(payload, leading + "repeat(".len())?;
            let offset_position = leading + (trimmed.len() - remainder.len());
            let offset = parse_trailing_offset(remainder, offset_position)?;
            return Ok(Self::Repeat {
                id: None,
                iteration,
                offset,
            });
        }
        if let Some(last_dot) = trimmed.rfind('.') {
            let id_part = &trimmed[..last_dot];
            let suffix = &trimmed[last_dot + 1..];
            let suffix_offset = leading + last_dot + 1;
            if id_part.is_empty() {
                return Err(ValueError::OutOfRange {
                    offset: leading,
                    what: "a syncbase, repeat, or event reference needs a non-empty id before the '.'",
                });
            }
            if let Some(rest) = match_keyword_with_optional_signed_suffix(suffix, "begin") {
                let offset = parse_trailing_offset(rest, suffix_offset + "begin".len())?;
                return Ok(Self::Syncbase {
                    id: id_part.into(),
                    edge: SyncbaseEdge::Begin,
                    offset,
                });
            }
            if let Some(rest) = match_keyword_with_optional_signed_suffix(suffix, "end") {
                let offset = parse_trailing_offset(rest, suffix_offset + "end".len())?;
                return Ok(Self::Syncbase {
                    id: id_part.into(),
                    edge: SyncbaseEdge::End,
                    offset,
                });
            }
            if let Some((payload, remainder)) = strip_call(suffix, "repeat(") {
                let iteration = parse_repeat_iteration(payload, suffix_offset + "repeat(".len())?;
                let offset_position = suffix_offset + (suffix.len() - remainder.len());
                let offset = parse_trailing_offset(remainder, offset_position)?;
                return Ok(Self::Repeat {
                    id: Some(id_part.into()),
                    iteration,
                    offset,
                });
            }
            let (event_name, rest) = split_event_and_offset(suffix);
            if event_name.is_empty() {
                return Err(ValueError::OutOfRange {
                    offset: suffix_offset,
                    what: "event name must not be empty",
                });
            }
            let offset = parse_trailing_offset(rest, suffix_offset + event_name.len())?;
            return Ok(Self::Event {
                id: Some(id_part.into()),
                event: event_name.into(),
                offset,
            });
        }
        if trimmed.starts_with(|c: char| c.is_ascii_digit() || c == '+' || c == '-') {
            let signed =
                SignedClockValue::parse(trimmed).map_err(|error| shift_offset(error, leading))?;
            return Ok(Self::Offset(signed));
        }
        let (event_name, rest) = split_event_and_offset(trimmed);
        if event_name.is_empty() {
            return Err(ValueError::UnexpectedChar {
                offset: leading,
                found: trimmed
                    .chars()
                    .next()
                    .expect("trimmed is non-empty, checked above"),
            });
        }
        let offset = parse_trailing_offset(rest, leading + event_name.len())?;
        Ok(Self::Event {
            id: None,
            event: event_name.into(),
            offset,
        })
    }
}

/// Writes `offset` as a signed suffix (`"+5s"` or `"-2s"`) when present, or
/// nothing at all when `None`. Unlike a top-level [`SignedClockValue`]
/// [`Display`](fmt::Display), the sign is **never** omitted for a positive
/// value here, because an offset suffix's sign is not optional in the
/// grammar it appears in (`(("+"|"-") Clock-value)?` — the whole suffix is
/// optional, but a sign is mandatory whenever the suffix is present).
fn write_offset_suffix(
    f: &mut fmt::Formatter<'_>,
    offset: Option<&SignedClockValue>,
) -> fmt::Result {
    match offset {
        Some(offset) if offset.is_negative() => write!(f, "{offset}"),
        Some(offset) => write!(f, "+{offset}"),
        None => Ok(()),
    }
}

impl fmt::Display for TimingSpecifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Offset(signed) => write!(f, "{signed}"),
            Self::Syncbase { id, edge, offset } => {
                write!(f, "{id}.{edge}")?;
                write_offset_suffix(f, offset.as_ref())
            }
            Self::Event { id, event, offset } => {
                if let Some(id) = id {
                    write!(f, "{id}.")?;
                }
                f.write_str(event)?;
                write_offset_suffix(f, offset.as_ref())
            }
            Self::Repeat {
                id,
                iteration,
                offset,
            } => {
                if let Some(id) = id {
                    write!(f, "{id}.")?;
                }
                write!(f, "repeat({iteration})")?;
                write_offset_suffix(f, offset.as_ref())
            }
            Self::AccessKey { key, offset } => {
                write!(f, "accessKey({key})")?;
                write_offset_suffix(f, offset.as_ref())
            }
            Self::Wallclock(payload) => write!(f, "wallclock({payload})"),
            Self::Indefinite => f.write_str("indefinite"),
        }
    }
}

/// Parses a whole `begin` or `end` attribute value: a semicolon-separated
/// list of [`TimingSpecifier`]s (`end` reuses the identical grammar `begin`
/// uses). White space around each item is trimmed before that item is
/// parsed. The list must have at least one item — an empty (or
/// white-space-only) item, including the whole input being blank, is an
/// error, matching the grammar (`begin-value-list` has no "empty list"
/// production, unlike, say, [`crate::style::attr::parse_style_attribute`]'s
/// declaration list).
///
/// ```
/// use oxixml_svg::values::clock::{TimingSpecifier, parse_timing_list};
///
/// let specifiers = parse_timing_list("0s; click; box1.end+2s").expect("valid list");
/// assert_eq!(specifiers.len(), 3);
/// assert!(matches!(specifiers[0], TimingSpecifier::Offset(_)));
/// assert!(matches!(specifiers[1], TimingSpecifier::Event { id: None, .. }));
/// assert!(matches!(specifiers[2], TimingSpecifier::Syncbase { .. }));
/// ```
///
/// # Errors
///
/// Returns the first item's [`ValueError`] (with an offset into the whole
/// `input`, not just that item) if any item is empty or fails to parse.
pub fn parse_timing_list(input: &str) -> Result<Vec<TimingSpecifier>, ValueError> {
    let mut specifiers = Vec::new();
    let mut offset = 0usize;
    for item in input.split(';') {
        if item.trim_matches(is_wsp).is_empty() {
            return Err(ValueError::Empty);
        }
        let specifier =
            TimingSpecifier::parse(item).map_err(|error| shift_offset(error, offset))?;
        specifiers.push(specifier);
        offset += item.len() + 1;
    }
    Ok(specifiers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seconds(input: &str) -> f64 {
        ClockValue::parse(input)
            .unwrap_or_else(|e| panic!("{input:?} should parse as a clock value: {e}"))
            .as_seconds()
    }

    #[test]
    fn clock_value_parses_every_grammar_form() {
        let cases = [
            ("0", 0.0),
            ("5", 5.0),
            (" 5s ", 5.0),
            ("3.5s", 3.5),
            ("90min", 5400.0),
            ("250ms", 0.25),
            ("2h", 7200.0),
            ("01:30:05", 5405.0),
            ("1:30:05.5", 5405.5),
            ("90:00", 5400.0),
            ("2:3:4", 7384.0),
            ("1:2", 62.0),
        ];
        for (input, expected) in cases {
            assert_eq!(seconds(input), expected, "input {input:?}");
        }
    }

    #[test]
    fn clock_value_rejects_an_overlong_digit_run_that_would_overflow_to_infinity() {
        // `str::parse::<f64>` never errors on overflow, it only saturates: an
        // unusually long digit run must be rejected explicitly here, or the
        // parsed value would be `f64::INFINITY`, whose `Display` renders the
        // bare string `"inf"` -- not a valid clock-value token under any
        // grammar production, and one `ClockValue::parse` itself rejects.
        let overlong = "9".repeat(400);
        assert!(
            overlong.parse::<f64>().is_ok_and(f64::is_infinite),
            "test premise: an all-9s run this long must overflow f64"
        );

        // Timecount-value: the bare digit run itself overflows.
        assert!(matches!(
            ClockValue::parse(&overlong),
            Err(ValueError::OutOfRange { .. })
        ));
        // Timecount-value with an explicit metric.
        assert!(matches!(
            ClockValue::parse(&format!("{overlong}s")),
            Err(ValueError::OutOfRange { .. })
        ));
        // Full-clock-value: the Hours component overflows.
        assert!(matches!(
            ClockValue::parse(&format!("{overlong}:00:00")),
            Err(ValueError::OutOfRange { .. })
        ));
        // Full-clock-value: the Seconds component's integer part overflows
        // even though a fraction follows it.
        assert!(matches!(
            ClockValue::parse(&format!("00:00:{overlong}.5")),
            Err(ValueError::OutOfRange { .. })
        ));
    }

    #[test]
    fn clock_value_rejects_a_leading_sign() {
        assert!(
            ClockValue::parse("-5s").is_err(),
            "bare clock values are never signed"
        );
        assert!(
            ClockValue::parse("+5s").is_err(),
            "bare clock values are never signed"
        );
    }

    #[test]
    fn clock_value_rejects_empty_or_blank_input() {
        assert_eq!(ClockValue::parse(""), Err(ValueError::Empty));
        assert_eq!(ClockValue::parse("   "), Err(ValueError::Empty));
    }

    #[test]
    fn clock_value_rejects_trailing_garbage_and_unknown_metrics() {
        assert_eq!(
            ClockValue::parse("5sx"),
            Err(ValueError::TrailingData { offset: 2 })
        );
        assert!(
            ClockValue::parse("5m").is_err(),
            "'m' alone is not a metric"
        );
        assert!(ClockValue::parse("5x").is_err());
    }

    #[test]
    fn clock_value_rejects_a_missing_clock_form_component() {
        assert_eq!(
            ClockValue::parse("1:30:"),
            Err(ValueError::InvalidNumber { offset: 5 }),
            "seconds component is missing"
        );
        assert!(
            ClockValue::parse(":30:00").is_err(),
            "hours component is missing"
        );
        assert!(
            ClockValue::parse("1::00").is_err(),
            "minutes component is missing"
        );
    }

    #[test]
    fn clock_value_rejects_an_empty_fraction() {
        assert_eq!(
            ClockValue::parse("1:30:05."),
            Err(ValueError::InvalidNumber { offset: 8 }),
            "a '.' with no digits after it is not a valid Fraction"
        );
        assert!(
            ClockValue::parse("5.").is_err(),
            "same rule for the timecount form"
        );
    }

    #[test]
    fn clock_value_rejects_too_many_colon_segments() {
        assert_eq!(
            ClockValue::parse("1:2:3:4"),
            Err(ValueError::TrailingData { offset: 4 })
        );
    }

    #[test]
    fn clock_value_display_round_trips_through_reparse() {
        for input in ["0", "5s", "90min", "2h", "01:30:00", "1:2", "3.5s"] {
            let original = ClockValue::parse(input).expect("valid clock value");
            let rendered = original.to_string();
            let reparsed = ClockValue::parse(&rendered)
                .unwrap_or_else(|e| panic!("re-parsing {rendered:?} (from {input:?}) failed: {e}"));
            assert!(
                (reparsed.as_seconds() - original.as_seconds()).abs() < 1e-9,
                "input {input:?} rendered as {rendered:?}, which reparsed to {} instead of {}",
                reparsed.as_seconds(),
                original.as_seconds()
            );
        }
    }

    #[test]
    fn clock_value_from_seconds_displays_in_explicit_seconds_form() {
        assert_eq!(ClockValue::from_seconds(5.0).to_string(), "5s");
        assert_eq!(ClockValue::from_seconds(5.0).as_seconds(), 5.0);
    }

    #[test]
    fn signed_clock_value_handles_sign_and_magnitude() {
        assert_eq!(
            SignedClockValue::parse("5s").expect("valid").as_seconds(),
            5.0
        );
        assert_eq!(
            SignedClockValue::parse("+5s").expect("valid").as_seconds(),
            5.0
        );
        assert_eq!(
            SignedClockValue::parse("-5s").expect("valid").as_seconds(),
            -5.0
        );
        assert!(!SignedClockValue::parse("5s").expect("valid").is_negative());
        assert!(SignedClockValue::parse("-5s").expect("valid").is_negative());
        assert_eq!(
            SignedClockValue::parse("-5s")
                .expect("valid")
                .magnitude()
                .as_seconds(),
            5.0,
            "magnitude() never carries the sign"
        );
    }

    #[test]
    fn signed_clock_value_display_omits_a_positive_sign_but_keeps_a_negative_one() {
        assert_eq!(
            SignedClockValue::parse("5s").expect("valid").to_string(),
            "5s"
        );
        assert_eq!(
            SignedClockValue::parse("+5s").expect("valid").to_string(),
            "5s"
        );
        assert_eq!(
            SignedClockValue::parse("-5s").expect("valid").to_string(),
            "-5s"
        );
    }

    #[test]
    fn signed_clock_value_shifts_inner_errors_past_the_sign_byte() {
        assert_eq!(
            SignedClockValue::parse("-5sx"),
            Err(ValueError::TrailingData { offset: 3 }),
            "the inner ClockValue error offset (2, for '5sx') must shift by the 1-byte '-'"
        );
    }

    #[test]
    fn repeat_count_parses_indefinite_and_positive_numbers() {
        assert_eq!(
            RepeatCount::parse("indefinite"),
            Ok(RepeatCount::Indefinite)
        );
        assert_eq!(RepeatCount::parse("3"), Ok(RepeatCount::Count(3.0)));
        assert_eq!(RepeatCount::parse("2.5"), Ok(RepeatCount::Count(2.5)));
    }

    #[test]
    fn repeat_count_rejects_zero_and_negative_values() {
        assert!(matches!(
            RepeatCount::parse("0"),
            Err(ValueError::OutOfRange { .. })
        ));
        assert!(matches!(
            RepeatCount::parse("-1"),
            Err(ValueError::OutOfRange { .. })
        ));
    }

    #[test]
    fn repeat_count_rejects_an_overlong_digit_run_that_would_overflow_to_infinity() {
        // Same hazard as ClockValue: `parse_number` saturates an overlong
        // digit run to `f64::INFINITY` rather than erroring, which would
        // otherwise slip past the `value <= 0.0` check (infinity is not
        // `<= 0.0`) and produce a `RepeatCount` whose `Display` renders the
        // bare string `"inf"`, which cannot be re-parsed as either
        // `"indefinite"` or a valid `<number>`.
        let overlong = "9".repeat(400);
        assert!(matches!(
            RepeatCount::parse(&overlong),
            Err(ValueError::OutOfRange { .. })
        ));
    }

    #[test]
    fn repeat_count_display() {
        assert_eq!(RepeatCount::Count(3.0).to_string(), "3");
        assert_eq!(RepeatCount::Indefinite.to_string(), "indefinite");
    }

    #[test]
    fn clock_duration_parses_all_three_forms() {
        assert_eq!(
            ClockDuration::parse("indefinite"),
            Ok(ClockDuration::Indefinite)
        );
        assert_eq!(ClockDuration::parse("media"), Ok(ClockDuration::Media));
        assert_eq!(
            ClockDuration::parse("5s").expect("valid"),
            ClockDuration::Clock(ClockValue::parse("5s").expect("valid"))
        );
    }

    #[test]
    fn parse_repeat_dur_matches_clock_duration_parse() {
        for input in ["indefinite", "media", "5s", ""] {
            assert_eq!(
                parse_repeat_dur(input),
                ClockDuration::parse(input),
                "input {input:?}"
            );
        }
    }

    #[test]
    fn clock_duration_display() {
        assert_eq!(ClockDuration::Media.to_string(), "media");
        assert_eq!(ClockDuration::Indefinite.to_string(), "indefinite");
        assert_eq!(
            ClockDuration::Clock(ClockValue::from_seconds(5.0)).to_string(),
            "5s"
        );
    }

    #[test]
    fn timing_specifier_parses_offset_forms() {
        for (input, expected) in [("5s", 5.0), ("+5s", 5.0), ("-2s", -2.0)] {
            match TimingSpecifier::parse(input).unwrap_or_else(|e| panic!("{input:?}: {e}")) {
                TimingSpecifier::Offset(signed) => {
                    assert_eq!(signed.as_seconds(), expected, "input {input:?}");
                }
                other => panic!("input {input:?}: expected Offset, got {other:?}"),
            }
        }
    }

    #[test]
    fn timing_specifier_indefinite_wins_over_an_event_named_indefinite() {
        assert_eq!(
            TimingSpecifier::parse("indefinite"),
            Ok(TimingSpecifier::Indefinite)
        );
    }

    #[test]
    fn timing_specifier_parses_syncbase_forms_including_the_hyphen_offset_trap() {
        let cases = [
            ("box1.begin", SyncbaseEdge::Begin, None),
            ("box1.end", SyncbaseEdge::End, None),
            ("box1.begin+5s", SyncbaseEdge::Begin, Some(5.0)),
            // The classic trap: `keyword()`-style scanning that includes
            // hyphens would misread "end-2s" as one token. This must
            // parse as edge `End` with a `-2s` offset.
            ("box1.end-2s", SyncbaseEdge::End, Some(-2.0)),
        ];
        for (input, expected_edge, expected_offset) in cases {
            match TimingSpecifier::parse(input).unwrap_or_else(|e| panic!("{input:?}: {e}")) {
                TimingSpecifier::Syncbase { id, edge, offset } => {
                    assert_eq!(&*id, "box1", "input {input:?}");
                    assert_eq!(edge, expected_edge, "input {input:?}");
                    assert_eq!(
                        offset.map(|o| o.as_seconds()),
                        expected_offset,
                        "input {input:?}"
                    );
                }
                other => panic!("input {input:?}: expected Syncbase, got {other:?}"),
            }
        }
    }

    #[test]
    fn timing_specifier_parses_event_forms_bare_and_scoped() {
        assert_eq!(
            TimingSpecifier::parse("click"),
            Ok(TimingSpecifier::Event {
                id: None,
                event: "click".into(),
                offset: None,
            })
        );
        assert_eq!(
            TimingSpecifier::parse("box1.load"),
            Ok(TimingSpecifier::Event {
                id: Some("box1".into()),
                event: "load".into(),
                offset: None,
            })
        );
        match TimingSpecifier::parse("click+2s").expect("valid") {
            TimingSpecifier::Event {
                id: None,
                event,
                offset,
            } => {
                assert_eq!(&*event, "click");
                assert_eq!(offset.expect("has an offset").as_seconds(), 2.0);
            }
            other => panic!("expected Event, got {other:?}"),
        }
    }

    #[test]
    fn timing_specifier_parses_repeat_forms_bare_and_scoped() {
        assert_eq!(
            TimingSpecifier::parse("repeat(2)"),
            Ok(TimingSpecifier::Repeat {
                id: None,
                iteration: 2,
                offset: None,
            })
        );
        assert_eq!(
            TimingSpecifier::parse("box1.repeat(4)"),
            Ok(TimingSpecifier::Repeat {
                id: Some("box1".into()),
                iteration: 4,
                offset: None,
            })
        );
        match TimingSpecifier::parse("repeat(3)+1s").expect("valid") {
            TimingSpecifier::Repeat {
                id: None,
                iteration: 3,
                offset,
            } => {
                assert_eq!(offset.expect("has an offset").as_seconds(), 1.0);
            }
            other => panic!("expected Repeat, got {other:?}"),
        }
    }

    #[test]
    fn timing_specifier_a_bare_repeat_suffix_without_parens_is_an_event_not_a_repeat() {
        assert_eq!(
            TimingSpecifier::parse("box1.repeat"),
            Ok(TimingSpecifier::Event {
                id: Some("box1".into()),
                event: "repeat".into(),
                offset: None,
            }),
            "no '(' means this is the event named \"repeat\", not a repeat-value"
        );
    }

    #[test]
    fn timing_specifier_splits_a_dotted_id_at_the_last_dot() {
        assert_eq!(
            TimingSpecifier::parse("a.b.begin"),
            Ok(TimingSpecifier::Syncbase {
                id: "a.b".into(),
                edge: SyncbaseEdge::Begin,
                offset: None,
            })
        );
    }

    #[test]
    fn timing_specifier_parses_access_key_forms() {
        assert_eq!(
            TimingSpecifier::parse("accessKey(a)"),
            Ok(TimingSpecifier::AccessKey {
                key: 'a',
                offset: None,
            })
        );
        assert_eq!(
            TimingSpecifier::parse("accessKey(\u{e9})"),
            Ok(TimingSpecifier::AccessKey {
                key: '\u{e9}',
                offset: None,
            }),
            "a single non-ASCII character is still exactly one character"
        );
        match TimingSpecifier::parse("accessKey(b)+1s").expect("valid") {
            TimingSpecifier::AccessKey { key: 'b', offset } => {
                assert_eq!(offset.expect("has an offset").as_seconds(), 1.0);
            }
            other => panic!("expected AccessKey, got {other:?}"),
        }
    }

    #[test]
    fn timing_specifier_parses_wallclock_preserving_the_raw_payload() {
        assert_eq!(
            TimingSpecifier::parse("wallclock(20040101T000000Z)"),
            Ok(TimingSpecifier::Wallclock("20040101T000000Z".into()))
        );
    }

    #[test]
    fn timing_specifier_rejects_empty_input() {
        assert_eq!(TimingSpecifier::parse(""), Err(ValueError::Empty));
        assert_eq!(TimingSpecifier::parse("   "), Err(ValueError::Empty));
    }

    #[test]
    fn timing_specifier_rejects_malformed_wallclock_and_access_key_payloads() {
        assert!(matches!(
            TimingSpecifier::parse("wallclock()"),
            Err(ValueError::OutOfRange { .. })
        ));
        assert!(matches!(
            TimingSpecifier::parse("wallclock(x)y"),
            Err(ValueError::TrailingData { .. })
        ));
        assert!(matches!(
            TimingSpecifier::parse("accessKey()"),
            Err(ValueError::OutOfRange { .. })
        ));
        assert!(matches!(
            TimingSpecifier::parse("accessKey(ab)"),
            Err(ValueError::OutOfRange { .. })
        ));
    }

    #[test]
    fn timing_specifier_rejects_an_empty_id_or_empty_suffix_around_the_dot() {
        assert!(
            TimingSpecifier::parse(".begin").is_err(),
            "empty id before the dot"
        );
        assert!(
            TimingSpecifier::parse("box1.").is_err(),
            "empty suffix after the dot"
        );
    }

    #[test]
    fn timing_specifier_rejects_bad_repeat_iterations() {
        assert!(matches!(
            TimingSpecifier::parse("repeat(x)"),
            Err(ValueError::InvalidNumber { .. })
        ));
        assert!(matches!(
            TimingSpecifier::parse("repeat(99999999999)"),
            Err(ValueError::OutOfRange { .. })
        ));
    }

    #[test]
    fn timing_specifier_rejects_a_token_starting_with_an_unusable_character() {
        assert!(matches!(
            TimingSpecifier::parse("@foo"),
            Err(ValueError::UnexpectedChar { found: '@', .. })
        ));
    }

    #[test]
    fn timing_specifier_display_round_trips_through_reparse() {
        for input in [
            "indefinite",
            "5s",
            "-2s",
            "box1.begin",
            "box1.end-2s",
            "click",
            "click+2s",
            "box1.load",
            "repeat(2)",
            "box1.repeat(4)+1s",
            "accessKey(a)",
            "wallclock(20040101T000000Z)",
        ] {
            let original =
                TimingSpecifier::parse(input).unwrap_or_else(|e| panic!("{input:?}: {e}"));
            let rendered = original.to_string();
            let reparsed = TimingSpecifier::parse(&rendered)
                .unwrap_or_else(|e| panic!("re-parsing {rendered:?} (from {input:?}) failed: {e}"));
            assert_eq!(
                reparsed, original,
                "input {input:?} rendered as {rendered:?}"
            );
        }
    }

    #[test]
    fn parse_timing_list_parses_a_realistic_multi_item_list() {
        let specifiers = parse_timing_list("0s; click; box1.end+2s").expect("valid list");
        assert_eq!(specifiers.len(), 3);
        assert!(matches!(specifiers[0], TimingSpecifier::Offset(_)));
        assert!(matches!(
            specifiers[1],
            TimingSpecifier::Event { id: None, .. }
        ));
        assert!(matches!(specifiers[2], TimingSpecifier::Syncbase { .. }));
    }

    #[test]
    fn parse_timing_list_trims_white_space_around_each_item() {
        let specifiers = parse_timing_list(" click ; load ").expect("valid list");
        assert_eq!(specifiers.len(), 2);
        assert_eq!(
            specifiers[0],
            TimingSpecifier::Event {
                id: None,
                event: "click".into(),
                offset: None,
            }
        );
    }

    #[test]
    fn parse_timing_list_rejects_an_empty_list_or_a_blank_item() {
        assert_eq!(parse_timing_list(""), Err(ValueError::Empty));
        assert_eq!(parse_timing_list("click;;load"), Err(ValueError::Empty));
    }

    #[test]
    fn parse_timing_list_reports_an_error_offset_into_the_whole_list_not_just_the_item() {
        // "click; bad@id; load": the second item is " bad@id" (offset 6
        // within the whole list); within that item, '@' is at relative
        // offset 4, so the shifted offset must be 10.
        assert_eq!(
            parse_timing_list("click; bad@id; load"),
            Err(ValueError::UnexpectedChar {
                offset: 10,
                found: '@',
            })
        );
    }

    #[test]
    fn shift_offset_shifts_every_offset_carrying_variant_and_leaves_empty_alone() {
        assert_eq!(
            shift_offset(ValueError::UnexpectedEnd { offset: 1 }, 10),
            ValueError::UnexpectedEnd { offset: 11 }
        );
        assert_eq!(
            shift_offset(
                ValueError::UnexpectedChar {
                    offset: 1,
                    found: 'x',
                },
                10
            ),
            ValueError::UnexpectedChar {
                offset: 11,
                found: 'x',
            }
        );
        assert_eq!(
            shift_offset(ValueError::InvalidNumber { offset: 1 }, 10),
            ValueError::InvalidNumber { offset: 11 }
        );
        assert_eq!(
            shift_offset(
                ValueError::OutOfRange {
                    offset: 1,
                    what: "reason",
                },
                10
            ),
            ValueError::OutOfRange {
                offset: 11,
                what: "reason",
            }
        );
        assert_eq!(
            shift_offset(
                ValueError::UnknownKeyword {
                    offset: 1,
                    keyword: "kw".into(),
                },
                10
            ),
            ValueError::UnknownKeyword {
                offset: 11,
                keyword: "kw".into(),
            }
        );
        assert_eq!(shift_offset(ValueError::Empty, 10), ValueError::Empty);
        assert_eq!(
            shift_offset(ValueError::TrailingData { offset: 1 }, 10),
            ValueError::TrailingData { offset: 11 }
        );
    }
}
