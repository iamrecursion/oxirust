//! [`fn:format-dateTime`](https://www.w3.org/TR/xpath-functions-31/#func-format-dateTime),
//! `fn:format-date` and `fn:format-time`, and the picture-string language of
//! [F&O 3.1 §9.8](https://www.w3.org/TR/xpath-functions-31/#formatting-dates-and-times).
//!
//! A picture is a sequence of literal substrings and *variable markers* enclosed in square
//! brackets (`[[` and `]]` escape a literal bracket). A marker is a one-letter component
//! specifier, then up to two presentation modifiers, then an optional width modifier introduced
//! by a comma:
//!
//! ```text
//! [Y0001]-[M01]-[D01]T[H01]:[m01]:[s01][Z0:00]     2002-12-31T09:00:00+05:00
//! [D1] [MNn] [Y]                                    31 December 2002
//! [FNn], [D1o] [MNn] [Y]                            Tuesday, 31st December 2002
//! [H01]:[m01] [Pn]                                  09:00 am
//! ```
//!
//! The numeric presentation modifiers are those of
//! [`fn:format-integer`](fn@crate::datetime::format_integer), so `[D1]`, `[D01]`, `[Di]`, `[Dw]`
//! and `[D1o]` all work; `n`, `N` and `Nn` request the component *by name*, which this
//! implementation supplies in English for months, weekdays, the am/pm marker and the era.
//!
//! The language and calendar arguments are accepted and validated; a language other than
//! English or a calendar other than the ISO 8601 (`AD`) calendar is *not* an error, and the
//! fallback English/ISO representation is used, as §9.8.4.8 requires.
//!
//! `$place` selects from a small database of civil timezones — see the module source's
//! `CIVIL_ZONES`, whose contents §9.8.4.6 leaves to the implementation. Naming one of them
//! adjusts the value to the offset in force there (§9.8.4.3), and naming a country narrows the
//! search for the name `[ZN]` writes. A timezone the database cannot name is written as its
//! numeric offset instead.

use crate::context::FnContext;
use crate::datetime::format_integer::{IntegerPicture, digit_family_zero, ordinal_suffix};
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicData, AtomicValue};
use oxixml_xsd::{Decimal, TimezoneOffset};

mod timezone;

use timezone::{Place, apply_place};

/// Which of the three date formatting functions is being applied, and therefore which
/// components the value makes available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    DateTime,
    Date,
    Time,
}

impl Shape {
    const fn has_date(self) -> bool {
        matches!(self, Self::DateTime | Self::Date)
    }

    const fn has_time(self) -> bool {
        matches!(self, Self::DateTime | Self::Time)
    }

    const fn function(self) -> &'static str {
        match self {
            Self::DateTime => "fn:format-dateTime",
            Self::Date => "fn:format-date",
            Self::Time => "fn:format-time",
        }
    }
}

/// The components a value makes available to the formatter.
#[derive(Debug, Clone, Copy)]
struct Components {
    year: i64,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: Decimal,
    /// Minutes east of UTC, or `None` for a value with no timezone.
    timezone: Option<i32>,
    /// What `[ZN]` should call the timezone, when `$place` let the formatter name it.
    zone_name: Option<&'static str>,
    shape: Shape,
}

fn syntax_error(reason: impl std::fmt::Display) -> FnError {
    FnError::new(
        FnErrorCode::Fofd1340,
        format!("invalid date/time picture: {reason}"),
    )
}

fn unavailable(specifier: char, shape: Shape) -> FnError {
    FnError::new(
        FnErrorCode::Fofd1350,
        format!(
            "the component `{specifier}` is not available to {}",
            shape.function()
        ),
    )
}

/// One piece of a parsed picture string.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Literal(String),
    Marker(Marker),
}

/// A parsed variable marker.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Marker {
    specifier: char,
    presentation: String,
    second_modifier: Option<char>,
    min_width: Option<usize>,
    max_width: Option<usize>,
}

fn parse_picture(picture: &str) -> FnResult<Vec<Piece>> {
    let characters: Vec<char> = picture.chars().collect();
    let mut pieces = Vec::new();
    let mut literal = String::new();
    let mut index = 0;
    while index < characters.len() {
        match characters[index] {
            '[' if characters.get(index + 1) == Some(&'[') => {
                literal.push('[');
                index += 2;
            }
            ']' if characters.get(index + 1) == Some(&']') => {
                literal.push(']');
                index += 2;
            }
            ']' => return Err(syntax_error("an unmatched `]`")),
            '[' => {
                let end = characters[index + 1..]
                    .iter()
                    .position(|character| *character == ']')
                    .ok_or_else(|| syntax_error("an unterminated variable marker"))?
                    + index
                    + 1;
                if !literal.is_empty() {
                    pieces.push(Piece::Literal(std::mem::take(&mut literal)));
                }
                let body: String = characters[index + 1..end].iter().collect();
                pieces.push(Piece::Marker(parse_marker(&body)?));
                index = end + 1;
            }
            other => {
                literal.push(other);
                index += 1;
            }
        }
    }
    if !literal.is_empty() {
        pieces.push(Piece::Literal(literal));
    }
    Ok(pieces)
}

fn parse_marker(body: &str) -> FnResult<Marker> {
    let body: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    let mut characters = body.chars();
    let specifier = characters
        .next()
        .ok_or_else(|| syntax_error("an empty variable marker"))?;
    let rest: String = characters.collect();

    let (presentation, width) = match rest.rfind(',') {
        Some(index) => (rest[..index].to_owned(), Some(rest[index + 1..].to_owned())),
        None => (rest, None),
    };

    let mut presentation = presentation;
    let mut second_modifier = None;
    if presentation.chars().count() > 1 {
        let last = presentation.chars().next_back().unwrap_or(' ');
        if matches!(last, 'a' | 't' | 'c' | 'o') {
            second_modifier = Some(last);
            presentation.truncate(presentation.len() - last.len_utf8());
        }
    }

    let (min_width, max_width) = match width {
        None => (None, None),
        Some(width) => parse_width(&width)?,
    };

    Ok(Marker {
        specifier,
        presentation,
        second_modifier,
        min_width,
        max_width,
    })
}

fn parse_width(width: &str) -> FnResult<(Option<usize>, Option<usize>)> {
    let (minimum, maximum) = match width.split_once('-') {
        Some((minimum, maximum)) => (minimum, Some(maximum)),
        None => (width, None),
    };
    let minimum = parse_width_part(minimum)?;
    let maximum = match maximum {
        None => None,
        Some(maximum) => parse_width_part(maximum)?,
    };
    if minimum == Some(0) {
        return Err(syntax_error("a minimum width of zero"));
    }
    if maximum == Some(0) {
        return Err(syntax_error("a maximum width of zero"));
    }
    if let (Some(minimum), Some(maximum)) = (minimum, maximum) {
        if maximum < minimum {
            return Err(syntax_error("a maximum width below the minimum width"));
        }
    }
    Ok((minimum, maximum))
}

fn parse_width_part(part: &str) -> FnResult<Option<usize>> {
    if part == "*" {
        return Ok(None);
    }
    part.parse::<usize>()
        .map(Some)
        .map_err(|_| syntax_error(format!("`{part}` is not a width")))
}

/// [`fn:format-dateTime`](https://www.w3.org/TR/xpath-functions-31/#func-format-dateTime).
///
/// # Errors
///
/// [`FnErrorCode::Fofd1340`] for an invalid picture, [`FnErrorCode::Fofd1350`] for a component
/// the value does not have, and [`FnErrorCode::Xpty0004`] when the value is not an
/// `xs:dateTime`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{datetime::format_date_time, FnContext};
/// use oxixml_infoset::{AtomicType, AtomicValue};
///
/// let context = FnContext::new();
/// let value = AtomicValue::parse(AtomicType::DateTime, "2002-12-31T19:20:00-05:00")?;
///
/// assert_eq!(
///     format_date_time(&context, &value, "[Y0001]-[M01]-[D01]T[H01]:[m01]:[s01][Z]", None, None, None)?,
///     "2002-12-31T19:20:00-05:00"
/// );
/// assert_eq!(
///     format_date_time(&context, &value, "[h].[m01][Pn] on [FNn], [D1o] [MNn]", None, None, None)?,
///     "7.20pm on Tuesday, 31st December"
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn format_date_time(
    context: &FnContext,
    value: &AtomicValue,
    picture: &str,
    language: Option<&str>,
    calendar: Option<&str>,
    place: Option<&str>,
) -> FnResult<String> {
    let AtomicData::DateTime(instant) = value.data() else {
        return Err(wrong_type(value, "an xs:dateTime", "fn:format-dateTime"));
    };
    let components = Components {
        year: instant.year(),
        month: instant.month(),
        day: instant.day(),
        hour: instant.hour(),
        minute: instant.minute(),
        second: instant.second(),
        timezone: instant.timezone_offset().map(offset_minutes),
        zone_name: None,
        shape: Shape::DateTime,
    };
    format_components(context, components, picture, language, calendar, place)
}

/// [`fn:format-date`](https://www.w3.org/TR/xpath-functions-31/#func-format-date).
///
/// # Errors
///
/// See [`format_date_time`].
pub fn format_date(
    context: &FnContext,
    value: &AtomicValue,
    picture: &str,
    language: Option<&str>,
    calendar: Option<&str>,
    place: Option<&str>,
) -> FnResult<String> {
    let AtomicData::Date(date) = value.data() else {
        return Err(wrong_type(value, "an xs:date", "fn:format-date"));
    };
    let components = Components {
        year: date.year(),
        month: date.month(),
        day: date.day(),
        hour: 0,
        minute: 0,
        second: Decimal::from(0_i64),
        timezone: date.timezone_offset().map(offset_minutes),
        zone_name: None,
        shape: Shape::Date,
    };
    format_components(context, components, picture, language, calendar, place)
}

/// [`fn:format-time`](https://www.w3.org/TR/xpath-functions-31/#func-format-time).
///
/// # Errors
///
/// See [`format_date_time`].
pub fn format_time(
    context: &FnContext,
    value: &AtomicValue,
    picture: &str,
    language: Option<&str>,
    calendar: Option<&str>,
    place: Option<&str>,
) -> FnResult<String> {
    let AtomicData::Time(time) = value.data() else {
        return Err(wrong_type(value, "an xs:time", "fn:format-time"));
    };
    let components = Components {
        year: 0,
        month: 1,
        day: 1,
        hour: time.hour(),
        minute: time.minute(),
        second: time.second(),
        timezone: time.timezone_offset().map(offset_minutes),
        zone_name: None,
        shape: Shape::Time,
    };
    format_components(context, components, picture, language, calendar, place)
}

fn wrong_type(value: &AtomicValue, expected: &str, what: &str) -> FnError {
    FnError::new(
        FnErrorCode::Xpty0004,
        format!("{what} expects {expected}, not {}", value.atomic_type()),
    )
}

fn format_components(
    context: &FnContext,
    components: Components,
    picture: &str,
    language: Option<&str>,
    calendar: Option<&str>,
    place: Option<&str>,
) -> FnResult<String> {
    let _ = context;
    if let Some(calendar) = calendar {
        validate_calendar(calendar)?;
    }
    let components = apply_place(components, Place::parse(place));
    // F&O 3.1 §9.8.4.7 and §9.8.4.8: when the requested language or calendar is one the
    // processor cannot use, it falls back on the one it has — and must *say so*, by prefixing
    // the result with `[Language: xx]` or `[Calendar: AD]`. Silently answering in English
    // would leave the caller unable to tell the two cases apart (`format-date-en151`, `-en152`).
    let mut out = String::new();
    if let Some(language) = language {
        if !is_supported_language(language) {
            out.push_str("[Language: en]");
        }
    }
    if let Some(calendar) = calendar {
        if !is_supported_calendar(calendar) {
            out.push_str("[Calendar: AD]");
        }
    }
    for piece in parse_picture(picture)? {
        match piece {
            Piece::Literal(literal) => out.push_str(&literal),
            Piece::Marker(marker) => out.push_str(&render(&marker, components)?),
        }
    }
    Ok(out)
}

/// Whether the names this module can spell are the ones `language` asks for.
///
/// The month, weekday, era and am/pm names here are English, so only English — with or without
/// a region subtag — is served without a fallback note.
fn is_supported_language(language: &str) -> bool {
    let primary = language
        .split(['-', '_'])
        .next()
        .unwrap_or(language)
        .trim()
        .to_ascii_lowercase();
    primary.is_empty() || primary == "en"
}

/// Whether `calendar` names the one calendar this module computes in.
///
/// `AD` and `ISO` are the same proleptic Gregorian reckoning the XDM date/time types use; every
/// other calendar of §9.8.4.3 would need its own era and month arithmetic.
fn is_supported_calendar(calendar: &str) -> bool {
    let local = match calendar.strip_prefix("Q{") {
        Some(rest) => rest.split_once('}').map_or(calendar, |(_, local)| local),
        None => calendar
            .split_once(':')
            .map_or(calendar, |(_, local)| local),
    };
    matches!(local.trim(), "" | "AD" | "ISO")
}

/// The calendars F&O 3.1 §9.8.4.3 names. A calendar *in no namespace* that is not one of these
/// does not exist, and naming it is `FOFD1340`; one in a namespace is implementation-defined,
/// and this implementation defines it as "ignored, the Gregorian calendar is used".
const KNOWN_CALENDARS: &[&str] = &[
    "AD", "AH", "AME", "AM", "AP", "AS", "BE", "CB", "CE", "CL", "CS", "EE", "FE", "ISO", "JE",
    "KE", "KY", "ME", "MS", "NS", "OS", "RS", "SE", "SH", "SS", "TE", "VE", "VS",
];

fn validate_calendar(calendar: &str) -> FnResult<()> {
    let invalid = || {
        FnError::new(
            FnErrorCode::Fofd1340,
            format!("`{calendar}` is not a valid calendar name"),
        )
    };
    // A calendar is a QName or an EQName; anything with a space or a bracket is a syntax error.
    if calendar.trim().is_empty() || calendar.contains(char::is_whitespace) {
        return Err(invalid());
    }
    // `Q{uri}local`: a non-empty URI makes the calendar implementation-defined, and an *empty*
    // one is the same name space as an unprefixed lexical QName.
    let (namespaced, local) = match calendar.strip_prefix("Q{") {
        Some(rest) => match rest.split_once('}') {
            Some((uri, local)) => (!uri.is_empty(), local),
            None => return Err(invalid()),
        },
        None => match calendar.split_once(':') {
            Some((prefix, local)) => {
                if !oxixml_qname::is_ncname(prefix) {
                    return Err(invalid());
                }
                (true, local)
            }
            None => (false, calendar),
        },
    };
    if !oxixml_qname::is_ncname(local) {
        return Err(invalid());
    }
    if !namespaced && !KNOWN_CALENDARS.contains(&local) {
        return Err(invalid());
    }
    Ok(())
}

fn render(marker: &Marker, components: Components) -> FnResult<String> {
    let presentation = if marker.presentation.is_empty() {
        default_presentation(marker.specifier).to_owned()
    } else {
        marker.presentation.clone()
    };
    // The `[f]` picture is validated by `FractionPicture`, which reads it the other way round.
    if marker.specifier != 'f' {
        validate_digit_pattern(&presentation)?;
    }

    match marker.specifier {
        'Y' => {
            require_date(marker.specifier, components)?;
            let modulus = year_modulus(marker, &presentation);
            let year = components.year.unsigned_abs() % modulus;
            Ok(numeric(
                marker,
                &presentation,
                i64::try_from(year).unwrap_or(0),
            ))
        }
        'M' => {
            require_date(marker.specifier, components)?;
            named_or_numeric(
                marker,
                &presentation,
                i64::from(components.month),
                MONTH_NAMES
                    .get(usize::from(components.month).saturating_sub(1))
                    .copied(),
                MONTH_ABBREVIATIONS
                    .get(usize::from(components.month).saturating_sub(1))
                    .copied(),
            )
        }
        'D' => {
            require_date(marker.specifier, components)?;
            Ok(numeric(marker, &presentation, i64::from(components.day)))
        }
        'd' => {
            require_date(marker.specifier, components)?;
            let day = day_in_year(components.year, components.month, components.day);
            Ok(numeric(marker, &presentation, day))
        }
        'F' => {
            require_date(marker.specifier, components)?;
            let weekday = iso_weekday(components.year, components.month, components.day);
            named_or_numeric(
                marker,
                &presentation,
                weekday,
                WEEKDAY_NAMES
                    .get(usize::try_from(weekday).unwrap_or(1).saturating_sub(1))
                    .copied(),
                WEEKDAY_ABBREVIATIONS
                    .get(usize::try_from(weekday).unwrap_or(1).saturating_sub(1))
                    .copied(),
            )
        }
        'W' => {
            require_date(marker.specifier, components)?;
            let week = iso_week_of_year(components.year, components.month, components.day);
            Ok(numeric(marker, &presentation, week))
        }
        'w' => {
            require_date(marker.specifier, components)?;
            let week = week_in_month(components.year, components.month, components.day);
            Ok(numeric(marker, &presentation, week))
        }
        'H' => {
            require_time(marker.specifier, components)?;
            Ok(numeric(marker, &presentation, i64::from(components.hour)))
        }
        'h' => {
            require_time(marker.specifier, components)?;
            let hour = match components.hour % 12 {
                0 => 12,
                other => i64::from(other),
            };
            Ok(numeric(marker, &presentation, hour))
        }
        'P' => {
            require_time(marker.specifier, components)?;
            let marker_text = if components.hour < 12 { "am" } else { "pm" };
            Ok(apply_name_case(marker_text, &presentation, marker))
        }
        'm' => {
            require_time(marker.specifier, components)?;
            Ok(numeric(marker, &presentation, i64::from(components.minute)))
        }
        's' => {
            require_time(marker.specifier, components)?;
            let seconds = whole_seconds(components.second);
            Ok(numeric(marker, &presentation, seconds))
        }
        'f' => {
            require_time(marker.specifier, components)?;
            fractional_seconds(marker, &presentation, components.second)
        }
        'Z' => Ok(timezone(marker, &presentation, components, false)),
        'z' => Ok(timezone(marker, &presentation, components, true)),
        'C' => Ok(apply_name_case("ISO", &presentation, marker)),
        'E' => {
            require_date(marker.specifier, components)?;
            let era = if components.year > 0 { "AD" } else { "BC" };
            Ok(apply_name_case(era, &presentation, marker))
        }
        other => Err(syntax_error(format!(
            "`{other}` is not a component specifier"
        ))),
    }
}

fn require_date(specifier: char, components: Components) -> FnResult<()> {
    if components.shape.has_date() {
        Ok(())
    } else {
        Err(unavailable(specifier, components.shape))
    }
}

fn require_time(specifier: char, components: Components) -> FnResult<()> {
    if components.shape.has_time() {
        Ok(())
    } else {
        Err(unavailable(specifier, components.shape))
    }
}

const fn default_presentation(specifier: char) -> &'static str {
    match specifier {
        'F' | 'P' | 'C' | 'E' => "n",
        'm' | 's' => "01",
        'Z' | 'z' => "01:01",
        _ => "1",
    }
}

const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const WEEKDAY_NAMES: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

/// The conventional English abbreviations §9.8.4.4 asks a name to be shortened to when the
/// picture's maximum width will not hold it.
const MONTH_ABBREVIATIONS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// The conventional English weekday abbreviations — see [`MONTH_ABBREVIATIONS`].
const WEEKDAY_ABBREVIATIONS: [&str; 7] = ["Mon", "Tues", "Weds", "Thurs", "Fri", "Sat", "Sun"];

fn is_name_presentation(presentation: &str) -> bool {
    matches!(presentation, "n" | "N" | "Nn")
}

fn named_or_numeric(
    marker: &Marker,
    presentation: &str,
    value: i64,
    name: Option<&str>,
    abbreviation: Option<&str>,
) -> FnResult<String> {
    match (is_name_presentation(presentation), name) {
        (true, Some(name)) => {
            // F&O 3.1 §9.8.4.4: a name too long for the maximum width is abbreviated, and
            // "abbreviation should follow any convention for the language in question" — so
            // `[FNn,3-4]` writes `Weds`, not the truncation `Wedn` (`format-date-en117`,
            // `-en118`). A conventional abbreviation that is *still* too long is then truncated
            // like any other name, which is how `[FNn,3-3]` reaches `Wed`.
            let chosen = match (marker.max_width, abbreviation) {
                (Some(maximum), Some(short)) if name.chars().count() > maximum => short,
                _ => name,
            };
            Ok(apply_name_case(chosen, presentation, marker))
        }
        _ => Ok(numeric(marker, presentation, value)),
    }
}

fn apply_name_case(name: &str, presentation: &str, marker: &Marker) -> String {
    let cased = match presentation {
        "N" => name.to_uppercase(),
        "Nn" => title_case(name),
        _ => name.to_lowercase(),
    };
    // F&O 3.1 §9.8.4.4: the maximum width truncates a name that no conventional abbreviation has
    // already shortened enough. The *minimum* width does not pad it: the padding the section
    // describes is the numbering sequence's own — leading zeros, or the trailing space
    // [`pad_non_numeric`] gives a Roman numeral — and a name is neither. `[PNn,4-8]` on `09:30`
    // is therefore `Am`, not `Am  ` (`date-064`), while `[FNn,3-3]` is still `Wed`.
    match marker.max_width {
        Some(maximum) if cased.chars().count() > maximum => {
            cased.chars().take(maximum).collect::<String>()
        }
        _ => cased,
    }
}

fn title_case(value: &str) -> String {
    let mut characters = value.chars();
    match characters.next() {
        None => String::new(),
        Some(first) => {
            let rest: String = characters.flat_map(char::to_lowercase).collect();
            format!("{}{rest}", first.to_uppercase())
        }
    }
}

/// Renders an integer-valued component through `fn:format-integer`, having widened the
/// presentation modifier to satisfy the minimum width.
/// Checks a decimal-digit-pattern: F&O 3.1 §4.6.1 lets the optional digit signs precede the
/// mandatory ones, never follow them, and forbids mixing two digit families. A picture that is
/// not a digit pattern at all — `I`, `a`, `w`, `N` — is left to [`IntegerPicture`].
fn validate_digit_pattern(presentation: &str) -> FnResult<()> {
    let mut zero = None;
    let mut mandatory = 0_usize;
    for character in presentation.chars() {
        if let Some(family) = digit_family_zero(character) {
            if *zero.get_or_insert(family) != family {
                return Err(syntax_error("a picture may not mix two digit families"));
            }
            mandatory += 1;
        } else if character == '#' && mandatory > 0 {
            return Err(syntax_error(
                "an optional digit sign may not follow a mandatory one",
            ));
        }
    }
    Ok(())
}

fn numeric(marker: &Marker, presentation: &str, value: i64) -> String {
    let token = widen(presentation, marker.min_width);
    let ordinal = marker.second_modifier == Some('o');
    let rendered = match IntegerPicture::from_primary(&token, ordinal) {
        Ok(parsed) => parsed.format(value),
        // An unusable presentation modifier falls back on the default, which is never an error.
        Err(_) => {
            if ordinal {
                format!("{}{}", value, ordinal_suffix(value.unsigned_abs()))
            } else {
                value.to_string()
            }
        }
    };
    pad_non_numeric(&rendered, presentation, marker.min_width)
}

/// Applies the minimum width to a component written with a *non-numeric* numbering sequence.
///
/// [F&O 3.1 §9.8.4.4](https://www.w3.org/TR/xpath-functions-31/#date-time-examples) pads a
/// decimal digit pattern on the left with the zero digit — which [`widen`] does by adding
/// mandatory digit signs — and everything else *on the right with spaces*. `[Yi,4-4]` on the
/// year 1004 is `miv ` (`format-dateTime-006a`), and `[Yi,3-3]` on the same year, whose maximum
/// width has already reduced it to 4, is `iv ` (`format-dateTime-006`).
fn pad_non_numeric(rendered: &str, presentation: &str, min_width: Option<usize>) -> String {
    let Some(minimum) = min_width else {
        return rendered.to_owned();
    };
    if presentation.chars().any(|c| digit_family_zero(c).is_some()) {
        return rendered.to_owned();
    }
    let width = rendered.chars().count();
    if width >= minimum {
        return rendered.to_owned();
    }
    format!("{rendered}{}", " ".repeat(minimum - width))
}

/// Increases the number of mandatory digit signs in a decimal digit pattern to `min_width`,
/// first by promoting optional digit signs and then by prepending mandatory ones.
fn widen(presentation: &str, min_width: Option<usize>) -> String {
    let Some(minimum) = min_width else {
        return presentation.to_owned();
    };
    let zero = presentation
        .chars()
        .find_map(digit_family_zero)
        .unwrap_or('0');
    let is_mandatory = |character: char| digit_family_zero(character).is_some();
    let mandatory = presentation.chars().filter(|c| is_mandatory(*c)).count();
    if mandatory == 0 || mandatory >= minimum {
        if mandatory == 0 {
            return presentation.to_owned();
        }
        return presentation.to_owned();
    }
    let mut characters: Vec<char> = presentation.chars().collect();
    let mut needed = minimum - mandatory;
    for character in characters.iter_mut().rev() {
        if needed == 0 {
            break;
        }
        if *character == '#' {
            *character = zero;
            needed -= 1;
        }
    }
    let mut token: String = characters.into_iter().collect();
    for _ in 0..needed {
        token.insert(0, zero);
    }
    token
}

fn whole_seconds(second: Decimal) -> i64 {
    let rendered = second.to_string();
    rendered
        .split('.')
        .next()
        .and_then(|integer| integer.parse::<i64>().ok())
        .unwrap_or(0)
}

fn fractional_digits(second: Decimal) -> String {
    let rendered = second.to_string();
    match rendered.split_once('.') {
        Some((_, fraction)) => fraction.trim_end_matches('0').to_owned(),
        None => String::new(),
    }
}

/// The `[f]` picture, which F&O 3.1 §9.8.4.2 reads *left to right*: the fractional digits are
/// written from the decimal point outwards, so the mandatory digit signs come first and the
/// optional `#` signs after them — the opposite of an integer picture.
struct FractionPicture {
    /// The zero of the picture's digit family.
    zero: char,
    /// How many digits must be written.
    mandatory: usize,
    /// How many further digits may be written.
    optional: usize,
    /// Where a grouping separator falls, as a count of digits from the decimal point, with the
    /// separator itself.
    separators: Vec<(usize, char)>,
}

impl FractionPicture {
    /// # Errors
    ///
    /// [`FnErrorCode::Fofd1340`] when a mandatory digit sign follows an optional one, or when
    /// the picture mixes two digit families — both of which make the picture unreadable rather
    /// than merely unusual.
    fn parse(presentation: &str) -> FnResult<Self> {
        let mut zero = None;
        let mut mandatory = 0_usize;
        let mut optional = 0_usize;
        let mut separators = Vec::new();
        for character in presentation.chars() {
            if let Some(family) = digit_family_zero(character) {
                if optional > 0 {
                    return Err(syntax_error(
                        "a mandatory digit sign may not follow an optional one in `[f]`",
                    ));
                }
                if *zero.get_or_insert(family) != family {
                    return Err(syntax_error("a picture may not mix two digit families"));
                }
                mandatory += 1;
            } else if character == '#' {
                optional += 1;
            } else {
                separators.push((mandatory + optional, character));
            }
        }
        Ok(Self {
            zero: zero.unwrap_or('0'),
            mandatory,
            optional,
            separators,
        })
    }
}

fn fractional_seconds(marker: &Marker, presentation: &str, second: Decimal) -> FnResult<String> {
    let picture = FractionPicture::parse(presentation)?;
    if picture.mandatory == 0 && picture.optional == 0 {
        return Ok(String::new());
    }

    let minimum = marker
        .min_width
        .unwrap_or(picture.mandatory)
        .max(picture.mandatory);
    let unbounded = marker.max_width.is_none() && picture.mandatory == 1 && picture.optional == 0;
    let maximum = if unbounded {
        usize::MAX
    } else {
        marker
            .max_width
            .unwrap_or(picture.mandatory + picture.optional)
            .max(minimum)
    };

    let mut digits = fractional_digits(second);
    while digits.chars().count() < minimum {
        digits.push('0');
    }
    let mut digits: Vec<char> = digits.chars().take(maximum).collect();
    // F&O 3.1 §9.8.4.2: once the value has been truncated to the maximum width, a trailing zero
    // is dropped as long as the minimum width still allows it. `[f,*-2]` on `.006` therefore
    // truncates to `00` and then drops one zero, giving `0` (`format-time-023u`), while
    // `[f,2-2]` keeps both (`-023s`).
    while digits.len() > minimum && digits.last() == Some(&'0') {
        digits.pop();
    }
    let base = u32::from(picture.zero);
    let mut out = String::with_capacity(digits.len() + picture.separators.len());
    for (index, digit) in digits.iter().enumerate() {
        for (position, separator) in &picture.separators {
            if *position == index {
                out.push(*separator);
            }
        }
        let offset = u32::from(*digit) - u32::from('0');
        out.push(char::from_u32(base + offset).unwrap_or(*digit));
    }
    Ok(out)
}

fn timezone(
    marker: &Marker,
    presentation: &str,
    components: Components,
    gmt_prefix: bool,
) -> String {
    let Some(minutes) = components.timezone else {
        // A value without a timezone produces no output, except for military letters.
        return if presentation == "Z" {
            "J".to_owned()
        } else {
            String::new()
        };
    };

    if presentation == "Z" {
        return military_letter(minutes).map_or_else(
            || render_offset(minutes, &OffsetPicture::parse("01:01")),
            String::from,
        );
    }
    if presentation == "N" {
        // §9.8.4.6: `[ZN]` asks for the timezone *by name*. The names come from the small
        // database `$place` selects from — see [`CIVIL_ZONES`] — and when it holds none for
        // this timezone the offset is written out instead, as the specification's fallback.
        return components.zone_name.map_or_else(
            || render_offset(minutes, &OffsetPicture::parse("01:01")),
            str::to_owned,
        );
    }

    let traditional = marker.second_modifier == Some('t');
    let body = if traditional && minutes == 0 {
        "Z".to_owned()
    } else {
        let picture = OffsetPicture::parse(presentation);
        let full = render_offset(minutes, &picture);
        // F&O 3.1 §9.8.4.2, and XSLT 2.0 Erratum E29 before it: the full representation of an
        // offset is its sign, its hours, and — only when the offset is *not* a whole number of
        // hours — a colon and its minutes. A maximum width the full `±hh:mm` will not fit
        // therefore drops the `:00` of a whole-hour offset and keeps the `:30` of a half-hour
        // one: `[z,2-2]` writes `GMT-14` and `GMT-13:30` (`format-date-017`), where the widths
        // that do hold it — `[z,6-6]`, `[z,5-6]`, `[z,2-6]` — keep writing `GMT-14:00`
        // (`format-time-016` … `-018`).
        match marker.max_width {
            Some(maximum) if full.chars().count() > maximum && minutes % 60 == 0 => {
                render_offset(minutes, &picture.with_optional_minutes())
            }
            _ => full,
        }
    };
    if gmt_prefix && body != "Z" {
        format!("GMT{body}")
    } else {
        body
    }
}

/// How a `[Z]`/`[z]` presentation modifier lays the offset out.
///
/// F&O 3.1 §9.8.4.2 reads the modifier as a picture of the offset's digits: the digits before an
/// (optional) separator are the hours and those after it the minutes, so `01:01` is `+05:30`,
/// `00~00` is `-14~00`, `0:01` is `-9:30` and `999` is `-930`. The digit *family* is the
/// picture's, which is how `[Z٠٠:٠٠]` writes `+٠٥:٣٠`.
struct OffsetPicture {
    /// The zero of the picture's digit family.
    zero: char,
    /// The least number of hour digits.
    hour_width: usize,
    /// The separator between hours and minutes, empty when the picture had none.
    separator: String,
    /// The least number of minute digits, or `None` when the minutes are written only when the
    /// offset is not a whole number of hours.
    minute_width: Option<usize>,
}

impl OffsetPicture {
    fn parse(presentation: &str) -> Self {
        let is_digit = |character: char| digit_family_zero(character).is_some();
        let zero = presentation
            .chars()
            .find_map(digit_family_zero)
            .unwrap_or('0');
        let hours: String = presentation.chars().take_while(|c| is_digit(*c)).collect();
        let rest: String = presentation.chars().skip_while(|c| is_digit(*c)).collect();
        let separator: String = rest.chars().take_while(|c| !is_digit(*c)).collect();
        let minutes: String = rest.chars().skip_while(|c| !is_digit(*c)).collect();
        if separator.is_empty() {
            let digits = hours.chars().count().max(1);
            // Three or four digits are `hmm`/`hhmm`, run together; one or two are whole hours,
            // with `:mm` appended only when the offset needs it.
            return if digits >= 3 {
                Self {
                    zero,
                    hour_width: digits - 2,
                    separator: String::new(),
                    minute_width: Some(2),
                }
            } else {
                Self {
                    zero,
                    hour_width: digits,
                    separator: ":".to_owned(),
                    minute_width: None,
                }
            };
        }
        Self {
            zero,
            hour_width: hours.chars().count().max(1),
            separator,
            minute_width: Some(minutes.chars().count().max(1)),
        }
    }

    /// The same picture with the minutes written only when the offset needs them, which is what
    /// a maximum width too small for the full `±hh:mm` form asks for.
    fn with_optional_minutes(self) -> Self {
        Self {
            minute_width: None,
            ..self
        }
    }

    /// Writes `value` in this picture's digit family, padded to `width` digits.
    fn digits(&self, value: u32, width: usize) -> String {
        let base = u32::from(self.zero);
        let decimal = format!("{value:0width$}");
        decimal
            .chars()
            .map(|digit| {
                let offset = u32::from(digit) - u32::from('0');
                char::from_u32(base + offset).unwrap_or(digit)
            })
            .collect()
    }
}

fn render_offset(minutes: i32, picture: &OffsetPicture) -> String {
    let sign = if minutes < 0 { '-' } else { '+' };
    let magnitude = minutes.unsigned_abs();
    let hours = picture.digits(magnitude / 60, picture.hour_width);
    let remainder = magnitude % 60;
    match picture.minute_width {
        None if remainder == 0 => format!("{sign}{hours}"),
        None => format!(
            "{sign}{hours}{}{}",
            picture.separator,
            picture.digits(remainder, 2)
        ),
        Some(width) => format!(
            "{sign}{hours}{}{}",
            picture.separator,
            picture.digits(remainder, width)
        ),
    }
}

fn offset_minutes(offset: TimezoneOffset) -> i32 {
    // TimezoneOffset renders as `Z` or `±hh:mm`; parsing that back is the only public route to
    // its magnitude.
    let rendered = offset.to_string();
    if rendered == "Z" {
        return 0;
    }
    let sign = if rendered.starts_with('-') { -1 } else { 1 };
    let body = rendered.trim_start_matches(['+', '-']);
    let (hours, minutes) = body.split_once(':').unwrap_or((body, "0"));
    let hours: i32 = hours.parse().unwrap_or(0);
    let minutes: i32 = minutes.parse().unwrap_or(0);
    sign * (hours * 60 + minutes)
}

fn military_letter(minutes: i32) -> Option<&'static str> {
    if minutes % 60 != 0 {
        return None;
    }
    const POSITIVE: [&str; 13] = [
        "Z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    ];
    const NEGATIVE: [&str; 13] = [
        "Z", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
    ];
    let hours = minutes / 60;
    if hours >= 0 {
        POSITIVE.get(usize::try_from(hours).ok()?).copied()
    } else {
        NEGATIVE.get(usize::try_from(-hours).ok()?).copied()
    }
}

/// Days since 1970-01-01 in the proleptic Gregorian calendar
/// ([Howard Hinnant's `days_from_civil`](https://howardhinnant.github.io/date_algorithms.html)).
fn days_from_civil(year: i64, month: u8, day: u8) -> i64 {
    let month = i64::from(month);
    let day = i64::from(day);
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The ISO 8601 day of the week: 1 is Monday, 7 is Sunday.
fn iso_weekday(year: i64, month: u8, day: u8) -> i64 {
    let days = days_from_civil(year, month, day);
    // 1970-01-01 was a Thursday, which is ISO day 4.
    (days + 3).rem_euclid(7) + 1
}

fn day_in_year(year: i64, month: u8, day: u8) -> i64 {
    days_from_civil(year, month, day) - days_from_civil(year, 1, 1) + 1
}

/// The ISO 8601 week number: week 1 is the week containing the first Thursday of the year.
fn iso_week_of_year(year: i64, month: u8, day: u8) -> i64 {
    let weekday = iso_weekday(year, month, day);
    let thursday = days_from_civil(year, month, day) - (weekday - 1) + 3;
    let thursday_year = civil_year_of(thursday);
    let january_first = days_from_civil(thursday_year, 1, 1);
    (thursday - january_first) / 7 + 1
}

/// The calendar year containing the given day number.
fn civil_year_of(days: i64) -> i64 {
    let mut year = 1970 + days / 366;
    while days_from_civil(year + 1, 1, 1) <= days {
        year += 1;
    }
    while days_from_civil(year, 1, 1) > days {
        year -= 1;
    }
    year
}

/// The week within the month, numbered the way the ISO calendar numbers weeks within a year.
///
/// [F&O 3.1 §9.8.4.2](https://www.w3.org/TR/xpath-functions-31/#date-time-examples) gives the
/// `w` component the ISO rule scaled down to a month: weeks run Monday to Sunday, and week one
/// of a month is the week that contains that month's first Thursday. A date in the days before
/// that week therefore belongs to the *last* week of the preceding month — `2006-09-01` is a
/// Friday in the week of 28 August, and `format-date-011`/`format-dateTime-011` want `5` for it,
/// not `1`.
fn week_in_month(year: i64, month: u8, day: u8) -> i64 {
    let mut year = year;
    let mut month = month;
    let date = days_from_civil(year, month, day);
    loop {
        let start = first_week_start(year, month);
        if date >= start {
            return (date - start) / 7 + 1;
        }
        // Before week one: the date is in the closing week of the previous month.
        if month == 1 {
            year -= 1;
            month = 12;
        } else {
            month -= 1;
        }
    }
}

/// The Monday of the week that contains the first Thursday of `year`-`month`.
fn first_week_start(year: i64, month: u8) -> i64 {
    let first = days_from_civil(year, month, 1);
    // `iso_weekday` numbers Monday 1 … Sunday 7, so this is the Monday of the first day's week.
    let monday = first - (iso_weekday(year, month, 1) - 1);
    // That week holds the month's first Thursday only when the month starts on or before it.
    if monday + 3 >= first {
        monday
    } else {
        monday + 7
    }
}

fn year_modulus(marker: &Marker, presentation: &str) -> u64 {
    if let Some(maximum) = marker.max_width {
        return 10_u64
            .checked_pow(u32::try_from(maximum).unwrap_or(0))
            .unwrap_or(u64::MAX);
    }
    let digit_signs = presentation
        .chars()
        .filter(|c| digit_family_zero(*c).is_some() || *c == '#')
        .count();
    if digit_signs >= 2 {
        return 10_u64
            .checked_pow(u32::try_from(digit_signs).unwrap_or(0))
            .unwrap_or(u64::MAX);
    }
    u64::MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::AtomicType;

    fn parse(atomic_type: AtomicType, lexical: &str) -> AtomicValue {
        AtomicValue::parse(atomic_type, lexical).expect("a valid lexical form")
    }

    fn formatted(lexical: &str, picture: &str) -> String {
        let context = FnContext::new();
        format_date_time(
            &context,
            &parse(AtomicType::DateTime, lexical),
            picture,
            None,
            None,
            None,
        )
        .expect("a valid picture")
    }

    #[test]
    fn examples_from_the_specification() {
        let value = "2002-12-31T19:20:00-05:00";
        assert_eq!(
            formatted(value, "[Y0001]-[M01]-[D01]T[H01]:[m01]:[s01][Z]"),
            "2002-12-31T19:20:00-05:00"
        );
        assert_eq!(formatted(value, "[M]-[D]-[Y]"), "12-31-2002");
        assert_eq!(formatted(value, "[D]/[M]/[Y]"), "31/12/2002");
        assert_eq!(formatted(value, "[D1] [MNn] [Y]"), "31 December 2002");
        assert_eq!(formatted(value, "[D1o] [MNn] [Y]"), "31st December 2002");
        assert_eq!(formatted(value, "[MNn] [D], [Y]"), "December 31, 2002");
        assert_eq!(
            formatted(value, "[FNn], [D1o] [MNn] [Y]"),
            "Tuesday, 31st December 2002"
        );
        assert_eq!(formatted(value, "[H01]:[m01]"), "19:20");
        assert_eq!(formatted(value, "[h]:[m01] [Pn]"), "7:20 pm");
        assert_eq!(formatted(value, "[Y]"), "2002");
        assert_eq!(formatted(value, "[Y01]"), "02");
    }

    #[test]
    fn literal_text_and_escaped_brackets() {
        let value = "2002-12-31T19:20:00-05:00";
        assert_eq!(formatted(value, "on [D1] [MNn]"), "on 31 December");
        assert_eq!(formatted(value, "[[[Y]]]"), "[2002]");
        assert_eq!(formatted(value, "no markers"), "no markers");
    }

    #[test]
    fn width_modifiers_pad_and_truncate() {
        let value = "2002-01-05T09:07:00Z";
        assert_eq!(formatted(value, "[D,2]"), "05");
        assert_eq!(formatted(value, "[M,2]"), "01");
        assert_eq!(formatted(value, "[MNn,3-3]"), "Jan");
        assert_eq!(formatted(value, "[FNn,3-3]"), "Sat");
        assert_eq!(formatted(value, "[H,2]:[m,2]"), "09:07");
    }

    #[test]
    fn numbering_sequences_are_those_of_format_integer() {
        let value = "2002-12-31T19:20:00-05:00";
        assert_eq!(formatted(value, "[MI]"), "XII");
        assert_eq!(formatted(value, "[Mi]"), "xii");
        assert_eq!(formatted(value, "[Dw]"), "thirty-one");
        assert_eq!(formatted(value, "[DWw]"), "Thirty-One");
        assert_eq!(formatted(value, "[Ma]"), "l");
    }

    #[test]
    fn fractional_seconds() {
        let value = "2002-12-31T19:20:03.25Z";
        assert_eq!(formatted(value, "[s].[f]"), "03.25");
        assert_eq!(formatted(value, "[s].[f001]"), "03.250");
        assert_eq!(formatted(value, "[s].[f01]"), "03.25");
        assert_eq!(formatted(value, "[s].[f1,1-1]"), "03.2");
        assert_eq!(formatted("2002-12-31T19:20:03Z", "[s].[f001]"), "03.000");
    }

    #[test]
    fn timezone_presentations() {
        assert_eq!(formatted("2002-12-31T19:20:00-05:00", "[Z]"), "-05:00");
        assert_eq!(formatted("2002-12-31T19:20:00-05:00", "[Z0]"), "-5");
        assert_eq!(formatted("2002-12-31T19:20:00+05:30", "[Z0]"), "+5:30");
        assert_eq!(formatted("2002-12-31T19:20:00+00:00", "[Z0]"), "+0");
        assert_eq!(formatted("2002-12-31T19:20:00-05:00", "[Z0000]"), "-0500");
        assert_eq!(formatted("2002-12-31T19:20:00+00:00", "[Z0000t]"), "Z");
        assert_eq!(formatted("2002-12-31T19:20:00-05:00", "[ZZ]"), "R");
        assert_eq!(formatted("2002-12-31T19:20:00Z", "[ZZ]"), "Z");
        assert_eq!(formatted("2002-12-31T19:20:00-05:00", "[z]"), "GMT-05:00");
        // A request for the timezone *name* (`[ZN]`) has no name database to draw on, so F&O 3.1
        // §9.8.4.2's fallback to the numeric offset applies.
        assert_eq!(formatted("2002-12-31T19:20:00-05:00", "[ZN]"), "-05:00");
        assert_eq!(formatted("2002-12-31T19:20:00+05:30", "[ZN]"), "+05:30");
        // A value with no timezone produces nothing at all.
        assert_eq!(formatted("2002-12-31T19:20:00", "[H01][Z]"), "19");
        assert_eq!(formatted("2002-12-31T19:20:00", "[ZZ]"), "J");
    }

    /// F&O 3.1 §9.8.4.2: a maximum width too small for the full `±hh:mm` drops the minutes of a
    /// whole-hour offset and keeps those of any other — `format-date-017` over the whole range of
    /// half-hour offsets, against `format-time-016`/`-017`/`-018`, which give the width room for
    /// the full form and must keep it.
    #[test]
    fn a_maximum_width_shortens_a_whole_hour_timezone_offset() {
        let at = |offset: &str| format!("0985-03-01T09:15:06.456{offset}");
        for (offset, narrow) in [
            ("-14:00", "GMT-14"),
            ("-13:30", "GMT-13:30"),
            ("-05:00", "GMT-05"),
            ("-00:30", "GMT-00:30"),
            ("Z", "GMT+00"),
            ("+05:30", "GMT+05:30"),
            ("+14:00", "GMT+14"),
        ] {
            assert_eq!(formatted(&at(offset), "[z,2-2]"), narrow, "for {offset}");
            // The widths that hold `±hh:mm` write it in full, however wide the minimum.
            let full = format!("GMT{}", if offset == "Z" { "+00:00" } else { offset });
            for picture in ["[z,6-6]", "[z,5-6]", "[z,2-6]", "[z]"] {
                assert_eq!(
                    formatted(&at(offset), picture),
                    full,
                    "{picture} for {offset}"
                );
            }
        }
        // The same rule without the `GMT` prefix, and with an explicit digit family.
        assert_eq!(formatted(&at("-05:00"), "[Z,2-2]"), "-05");
        assert_eq!(formatted(&at("-05:30"), "[Z,2-2]"), "-05:30");
        assert_eq!(formatted(&at("+00:00"), "[Z,1-1]"), "+00");
    }

    /// XSLT 3.0 §9.8: the `[P]` component names the half of the day, cased by the presentation
    /// modifier and shortened — never padded — by the width (`date-064`, `format-time-021`,
    /// `-022`).
    #[test]
    fn the_half_of_the_day_is_cased_and_shortened_but_never_padded() {
        for (value, morning) in [
            ("2002-12-31T00:00:00Z", true),
            ("2002-12-31T00:05:00Z", true),
            ("2002-12-31T09:30:00Z", true),
            ("2002-12-31T12:00:00Z", false),
            ("2002-12-31T15:15:00Z", false),
        ] {
            let (lower, title, upper, letter) = if morning {
                ("am", "Am", "AM", "A")
            } else {
                ("pm", "Pm", "PM", "P")
            };
            assert_eq!(formatted(value, "[Pn]"), lower);
            assert_eq!(formatted(value, "[PNn]"), title);
            assert_eq!(formatted(value, "[PN]"), upper);
            // A minimum width wider than the name leaves it alone rather than padding it.
            for picture in ["[PNn,4-8]", "[PNn,4-4]", "[PNn,3-3]", "[PNn,2-2]"] {
                assert_eq!(formatted(value, picture), title, "{picture} for {value}");
            }
            assert_eq!(formatted(value, "[PN,2-2]"), upper);
            assert_eq!(formatted(value, "[Pn,2-2]"), lower);
            assert_eq!(formatted(value, "[PNn,1-1]"), letter);
        }
        // A month or weekday name is not padded either, and is still abbreviated and truncated.
        assert_eq!(formatted("2002-05-04T00:00:00Z", "[MNn,6-6]"), "May");
        assert_eq!(formatted("2002-05-04T00:00:00Z", "[FNn,9-9]"), "Saturday");
        assert_eq!(formatted("2002-05-04T00:00:00Z", "[FNn,3-3]"), "Sat");
    }

    #[test]
    fn calendar_day_and_week_components() {
        assert_eq!(formatted("2002-12-31T00:00:00Z", "[d]"), "365");
        assert_eq!(formatted("2004-12-31T00:00:00Z", "[d]"), "366");
        assert_eq!(formatted("2002-01-01T00:00:00Z", "[W]"), "1");
        assert_eq!(formatted("2005-01-01T00:00:00Z", "[W]"), "53");
        assert_eq!(formatted("2002-12-31T00:00:00Z", "[F1]"), "2");
        // `[w]` numbers weeks within the month the way ISO numbers them within a year: week one
        // is the Monday-to-Sunday week holding the month's first Thursday. December 2002 begins
        // on a Sunday, so 1 December is still in November's closing week — its fourth — and 31
        // December, a Tuesday, opens December's fifth week.
        assert_eq!(formatted("2002-12-01T00:00:00Z", "[w]"), "4");
        assert_eq!(formatted("2002-12-31T00:00:00Z", "[w]"), "5");
    }

    #[test]
    fn eras_and_calendars() {
        assert_eq!(formatted("2002-12-31T00:00:00Z", "[E]"), "ad");
        assert_eq!(formatted("2002-12-31T00:00:00Z", "[EN]"), "AD");
        assert_eq!(formatted("2002-12-31T00:00:00Z", "[CNn]"), "Iso");
    }

    #[test]
    fn components_unavailable_to_the_value_raise_fofd1350() {
        let context = FnContext::new();
        let time = parse(AtomicType::Time, "19:20:00");
        let error =
            format_time(&context, &time, "[Y]", None, None, None).expect_err("a time has no year");
        assert!(error.is(FnErrorCode::Fofd1350));

        let date = parse(AtomicType::Date, "2002-12-31");
        let error = format_date(&context, &date, "[H01]", None, None, None)
            .expect_err("a date has no hour");
        assert!(error.is(FnErrorCode::Fofd1350));
    }

    #[test]
    fn invalid_pictures_raise_fofd1340() {
        let context = FnContext::new();
        let value = parse(AtomicType::DateTime, "2002-12-31T19:20:00Z");
        // `V` and `v` are CLDR zone specifiers, not F&O 3.1 ones, so they are rejected like any
        // other unknown component specifier rather than emitting a zone name.
        for picture in ["[Y", "]", "[]", "[Q]", "[V]", "[v]", "[Y,0]", "[Y,3-2]"] {
            let error =
                format_date_time(&context, &value, picture, None, None, None).expect_err(picture);
            assert!(error.is(FnErrorCode::Fofd1340), "{picture}: {error}");
        }
    }

    #[test]
    fn format_date_and_format_time_accept_their_own_types() -> FnResult<()> {
        let context = FnContext::new();
        assert_eq!(
            format_date(
                &context,
                &parse(AtomicType::Date, "2002-12-31"),
                "[D]-[M]-[Y]",
                None,
                None,
                None
            )?,
            "31-12-2002"
        );
        assert_eq!(
            format_time(
                &context,
                &parse(AtomicType::Time, "19:20:00"),
                "[H01]:[m01]",
                None,
                None,
                None
            )?,
            "19:20"
        );
        Ok(())
    }

    /// §9.8.4.4: a name that will not fit the maximum width is shortened by the language's
    /// convention where there is one — `Wednesday` to `Weds`, not to `Wedn` (`format-date-en116`
    /// through `-en118`, `-en104` through `-en106`).
    #[test]
    fn names_too_long_for_the_picture_use_a_conventional_abbreviation() -> FnResult<()> {
        let context = FnContext::new();
        let weekdays = |picture: &str| -> FnResult<Vec<String>> {
            // 2003-12-08 was a Monday.
            (8..15)
                .map(|day| {
                    format_date(
                        &context,
                        &parse(AtomicType::Date, &format!("2003-12-{day:02}")),
                        picture,
                        None,
                        None,
                        None,
                    )
                })
                .collect()
        };
        assert_eq!(
            weekdays("[FNn,3-3]")?,
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        );
        assert_eq!(
            weekdays("[FNn,3-4]")?,
            ["Mon", "Tues", "Weds", "Thur", "Fri", "Sat", "Sun"]
        );
        assert_eq!(
            weekdays("[FNn,3-5]")?,
            ["Mon", "Tues", "Weds", "Thurs", "Fri", "Sat", "Sun"]
        );
        assert_eq!(
            weekdays("[FN,3-3]")?,
            ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        );
        // A name the width does hold is left whole.
        assert_eq!(
            weekdays("[FNn]")?,
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        );
        let month = |picture: &str, month: u8| -> FnResult<String> {
            format_date(
                &context,
                &parse(AtomicType::Date, &format!("2003-{month:02}-01")),
                picture,
                None,
                None,
                None,
            )
        };
        assert_eq!(month("[MNn,3-3]", 9)?, "Sep");
        assert_eq!(month("[MN,3-3]", 12)?, "DEC");
        assert_eq!(month("[MNn]", 9)?, "September");
        Ok(())
    }
}
