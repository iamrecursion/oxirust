//! Lexical-representation parsing for the date/time datatypes, following the
//! grammar productions of [XML Schema Part 2 §D.3](https://www.w3.org/TR/xmlschema11-2/#dateTime-lexical-mapping).

use super::{
    DateTimeOverflowError, ParseDateTimeError, ParseDateTimeErrorKind, TimezoneOffset,
    days_in_month,
};
use crate::{Date, DateTime, Decimal, GDay, GMonth, GMonthDay, GYear, GYearMonth, Time};
use std::str::FromStr;

// [16]   dateTimeLexicalRep ::= yearFrag '-' monthFrag '-' dayFrag 'T' ((hourFrag ':' minuteFrag ':' secondFrag) | endOfDayFrag) timezoneFrag?
pub(super) fn date_time_lexical_rep(input: &str) -> Result<(DateTime, &str), ParseDateTimeError> {
    let (year, input) = year_frag(input)?;
    let input = expect_char(input, '-', "The year and month must be separated by '-'")?;
    let (month, input) = month_frag(input)?;
    let input = expect_char(input, '-', "The month and day must be separated by '-'")?;
    let (day, input) = day_frag(input)?;
    let input = expect_char(input, 'T', "The date and time must be separated by 'T'")?;
    let (hour, input) = hour_frag(input)?;
    let input = expect_char(input, ':', "The hours and minutes must be separated by ':'")?;
    let (minute, input) = minute_frag(input)?;
    let input = expect_char(
        input,
        ':',
        "The minutes and seconds must be separated by ':'",
    )?;
    let (second, input) = second_frag(input)?;
    // [17] endOfDayFrag ::= '24:00:00' ('.' '0'+)? — hour 24 is only legal when *both*
    // the minute and the second are zero, so the guard must be a disjunction, not a conjunction.
    if hour == 24 && (minute != 0 || second != Decimal::from(0)) {
        return Err(ParseDateTimeError::msg(
            "Times are not allowed to be after 24:00:00",
        ));
    }
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    validate_day_of_month(Some(year), month, day)?;
    Ok((
        DateTime::new(year, month, day, hour, minute, second, timezone_offset)?,
        input,
    ))
}

// [17]   timeLexicalRep ::= ((hourFrag ':' minuteFrag ':' secondFrag) | endOfDayFrag) timezoneFrag?
pub(super) fn time_lexical_rep(input: &str) -> Result<(Time, &str), ParseDateTimeError> {
    let (hour, input) = hour_frag(input)?;
    let input = expect_char(input, ':', "The hours and minutes must be separated by ':'")?;
    let (minute, input) = minute_frag(input)?;
    let input = expect_char(
        input,
        ':',
        "The minutes and seconds must be separated by ':'",
    )?;
    let (second, input) = second_frag(input)?;
    // [17] endOfDayFrag ::= '24:00:00' ('.' '0'+)? — hour 24 is only legal when *both*
    // the minute and the second are zero, so the guard must be a disjunction, not a conjunction.
    if hour == 24 && (minute != 0 || second != Decimal::from(0)) {
        return Err(ParseDateTimeError::msg(
            "Times are not allowed to be after 24:00:00",
        ));
    }
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    Ok((Time::new(hour, minute, second, timezone_offset)?, input))
}

// [18]   dateLexicalRep ::= yearFrag '-' monthFrag '-' dayFrag timezoneFrag?
pub(super) fn date_lexical_rep(input: &str) -> Result<(Date, &str), ParseDateTimeError> {
    let (year, input) = year_frag(input)?;
    let input = expect_char(input, '-', "The year and month must be separated by '-'")?;
    let (month, input) = month_frag(input)?;
    let input = expect_char(input, '-', "The month and day must be separated by '-'")?;
    let (day, input) = day_frag(input)?;
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    validate_day_of_month(Some(year), month, day)?;
    Ok((Date::new(year, month, day, timezone_offset)?, input))
}

// [19]   gYearMonthLexicalRep ::= yearFrag '-' monthFrag timezoneFrag?
pub(super) fn g_year_month_lexical_rep(
    input: &str,
) -> Result<(GYearMonth, &str), ParseDateTimeError> {
    let (year, input) = year_frag(input)?;
    let input = expect_char(input, '-', "The year and month must be separated by '-'")?;
    let (month, input) = month_frag(input)?;
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    Ok((GYearMonth::new(year, month, timezone_offset)?, input))
}

// [20]   gYearLexicalRep ::= yearFrag timezoneFrag?
pub(super) fn g_year_lexical_rep(input: &str) -> Result<(GYear, &str), ParseDateTimeError> {
    let (year, input) = year_frag(input)?;
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    Ok((GYear::new(year, timezone_offset)?, input))
}

// [21]   gMonthDayLexicalRep ::= '--' monthFrag '-' dayFrag timezoneFrag?
pub(super) fn g_month_day_lexical_rep(
    input: &str,
) -> Result<(GMonthDay, &str), ParseDateTimeError> {
    let input = expect_char(input, '-', "gMonthDay values must start with '--'")?;
    let input = expect_char(input, '-', "gMonthDay values must start with '--'")?;
    let (month, input) = month_frag(input)?;
    let input = expect_char(input, '-', "The month and day must be separated by '-'")?;
    let (day, input) = day_frag(input)?;
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    validate_day_of_month(None, month, day)?;
    Ok((GMonthDay::new(month, day, timezone_offset)?, input))
}

// [22]   gDayLexicalRep ::= '---' dayFrag timezoneFrag?
pub(super) fn g_day_lexical_rep(input: &str) -> Result<(GDay, &str), ParseDateTimeError> {
    let input = expect_char(input, '-', "gDay values must start with '---'")?;
    let input = expect_char(input, '-', "gDay values must start with '---'")?;
    let input = expect_char(input, '-', "gDay values must start with '---'")?;
    let (day, input) = day_frag(input)?;
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    Ok((GDay::new(day, timezone_offset)?, input))
}

// [23]   gMonthLexicalRep ::= '--' monthFrag timezoneFrag?
//
// The original XSD 1.0 Recommendation instead had `'--' monthFrag ('--')? timezoneFrag?`
// — a redundant trailing '--' placeholder for the day field that `gMonth` shares no room
// for. A later erratum (W3C Bug 6901) dropped it from the grammar, and this parser follows
// the modern, erratum'd production: a trailing '--' (`msData/datatypes/gMonth002.xml`:
// `--03--`; the W3C XML Schema Test Suite's own `gMonth_*` facet tests under
// `Facets/gMonth/`) is a lexical error, exactly like the Test Suite's own
// `expected validity="invalid"` facet-schema tests (e.g. `gMonth_maxExclusive002_208`,
// itself filed against the same bug) require. `Display` already only ever emits this
// modern form.
pub(super) fn g_month_lexical_rep(input: &str) -> Result<(GMonth, &str), ParseDateTimeError> {
    let input = expect_char(input, '-', "gMonth values must start with '--'")?;
    let input = expect_char(input, '-', "gMonth values must start with '--'")?;
    let (month, input) = month_frag(input)?;
    let (timezone_offset, input) = optional_end(input, timezone_frag)?;
    Ok((GMonth::new(month, timezone_offset)?, input))
}

// [56]   yearFrag ::= '-'? (([1-9] digit digit digit+)) | ('0' digit digit digit))
fn year_frag(input: &str) -> Result<(i64, &str), ParseDateTimeError> {
    let (sign, input) = if let Some(left) = input.strip_prefix('-') {
        (-1, left)
    } else {
        (1, input)
    };
    let (number_str, input) = integer_prefix(input);
    if number_str.len() < 4 {
        return Err(ParseDateTimeError::msg(
            "The year should be encoded on 4 digits",
        ));
    }
    if number_str.len() > 4 && number_str.starts_with('0') {
        return Err(ParseDateTimeError::msg(
            "The years value must not start with 0 if it can be encoded in at least 4 digits",
        ));
    }
    // Production [56] places no limit on the digit count, so a syntactically valid
    // `yearFrag` can still exceed what this implementation's `i64`-keyed internal
    // representation can hold. That is an *overflow* of the value space, not a
    // lexical error: it must be distinguishable via `ParseDateTimeError::is_overflow`
    // (matching XPath FODT0001) rather than folded into the generic lexical-error
    // path (which would otherwise surface as FORG0001 to a downstream consumer).
    let number = i64::from_str(number_str).map_err(|_| DateTimeOverflowError)?;
    Ok((sign * number, input))
}

// [57]   monthFrag ::= ('0' [1-9]) | ('1' [0-2])
fn month_frag(input: &str) -> Result<(u8, &str), ParseDateTimeError> {
    let (number_str, input) = integer_prefix(input);
    if number_str.len() != 2 {
        return Err(ParseDateTimeError::msg(
            "Month must be encoded with two digits",
        ));
    }
    let number =
        u8::from_str(number_str).map_err(|_| ParseDateTimeError::msg("Integer overflow"))?;
    if !(1..=12).contains(&number) {
        return Err(ParseDateTimeError::msg("Month must be between 01 and 12"));
    }
    Ok((number, input))
}

// [58]   dayFrag ::= ('0' [1-9]) | ([12] digit) | ('3' [01])
fn day_frag(input: &str) -> Result<(u8, &str), ParseDateTimeError> {
    let (number_str, input) = integer_prefix(input);
    if number_str.len() != 2 {
        return Err(ParseDateTimeError::msg(
            "Day must be encoded with two digits",
        ));
    }
    let number =
        u8::from_str(number_str).map_err(|_| ParseDateTimeError::msg("Integer overflow"))?;
    if !(1..=31).contains(&number) {
        return Err(ParseDateTimeError::msg("Day must be between 01 and 31"));
    }
    Ok((number, input))
}

// [59]   hourFrag ::= ([01] digit) | ('2' [0-3])
// We also allow 24 for ease of parsing.
fn hour_frag(input: &str) -> Result<(u8, &str), ParseDateTimeError> {
    let (number_str, input) = integer_prefix(input);
    if number_str.len() != 2 {
        return Err(ParseDateTimeError::msg(
            "Hours must be encoded with two digits",
        ));
    }
    let number =
        u8::from_str(number_str).map_err(|_| ParseDateTimeError::msg("Integer overflow"))?;
    if !(0..=24).contains(&number) {
        return Err(ParseDateTimeError::msg("Hours must be between 00 and 24"));
    }
    Ok((number, input))
}

// [60]   minuteFrag ::= [0-5] digit
fn minute_frag(input: &str) -> Result<(u8, &str), ParseDateTimeError> {
    let (number_str, input) = integer_prefix(input);
    if number_str.len() != 2 {
        return Err(ParseDateTimeError::msg(
            "Minutes must be encoded with two digits",
        ));
    }
    let number =
        u8::from_str(number_str).map_err(|_| ParseDateTimeError::msg("Integer overflow"))?;
    if !(0..=59).contains(&number) {
        return Err(ParseDateTimeError::msg("Minutes must be between 00 and 59"));
    }
    Ok((number, input))
}

// [61]   secondFrag ::= ([0-5] digit) ('.' digit+)?
fn second_frag(input: &str) -> Result<(Decimal, &str), ParseDateTimeError> {
    let (number_str, input) = decimal_prefix(input);
    let (before_dot_str, _) = number_str.split_once('.').unwrap_or((number_str, ""));
    if before_dot_str.len() != 2 {
        return Err(ParseDateTimeError::msg(
            "Seconds must be encoded with two digits",
        ));
    }
    let number = Decimal::from_str(number_str)
        .map_err(|_| ParseDateTimeError::msg("The second precision is too large"))?;
    if number < Decimal::from(0) || number >= Decimal::from(60) {
        return Err(ParseDateTimeError::msg("Seconds must be between 00 and 60"));
    }
    if number_str.ends_with('.') {
        return Err(ParseDateTimeError::msg(
            "Seconds are not allowed to end with a dot",
        ));
    }
    Ok((number, input))
}

// [63]   timezoneFrag ::= 'Z' | ('+' | '-') (('0' digit | '1' [0-3]) ':' minuteFrag | '14:00')
fn timezone_frag(input: &str) -> Result<(TimezoneOffset, &str), ParseDateTimeError> {
    if let Some(left) = input.strip_prefix('Z') {
        return Ok((TimezoneOffset::UTC, left));
    }
    let (sign, input) = if let Some(left) = input.strip_prefix('-') {
        (-1, left)
    } else if let Some(left) = input.strip_prefix('+') {
        (1, left)
    } else {
        (1, input)
    };

    let (hour_str, input) = integer_prefix(input);
    if hour_str.len() != 2 {
        return Err(ParseDateTimeError::msg(
            "The timezone hours must be encoded with two digits",
        ));
    }
    let hours = i16::from_str(hour_str).map_err(|_| ParseDateTimeError::msg("Integer overflow"))?;

    let input = expect_char(
        input,
        ':',
        "The timezone hours and minutes must be separated by ':'",
    )?;
    let (minutes, input) = minute_frag(input)?;

    if hours > 13 && !(hours == 14 && minutes == 0) {
        return Err(ParseDateTimeError::msg(
            "The timezone hours must be between 00 and 13",
        ));
    }

    Ok((
        TimezoneOffset::new(sign * (hours * 60 + i16::from(minutes)))
            .map_err(|e| ParseDateTimeError(ParseDateTimeErrorKind::InvalidTimezone(e)))?,
        input,
    ))
}

fn expect_char<'a>(
    input: &'a str,
    constant: char,
    error_message: &'static str,
) -> Result<&'a str, ParseDateTimeError> {
    if let Some(left) = input.strip_prefix(constant) {
        Ok(left)
    } else {
        Err(ParseDateTimeError::msg(error_message))
    }
}

fn integer_prefix(input: &str) -> (&str, &str) {
    let mut end = input.len();
    for (i, c) in input.char_indices() {
        if !c.is_ascii_digit() {
            end = i;
            break;
        }
    }
    input.split_at(end)
}

fn decimal_prefix(input: &str) -> (&str, &str) {
    let mut end = input.len();
    let mut dot_seen = false;
    for (i, c) in input.char_indices() {
        if c.is_ascii_digit() {
            // Ok.
        } else if c == '.' && !dot_seen {
            dot_seen = true;
        } else {
            end = i;
            break;
        }
    }
    input.split_at(end)
}

fn optional_end<T>(
    input: &str,
    parse: impl FnOnce(&str) -> Result<(T, &str), ParseDateTimeError>,
) -> Result<(Option<T>, &str), ParseDateTimeError> {
    Ok(if input.is_empty() {
        (None, input)
    } else {
        let (result, input) = parse(input)?;
        (Some(result), input)
    })
}

fn validate_day_of_month(year: Option<i64>, month: u8, day: u8) -> Result<(), ParseDateTimeError> {
    // Constraint: Day-of-month Values.
    if day > days_in_month(year, month) {
        return Err(ParseDateTimeError(
            ParseDateTimeErrorKind::InvalidDayOfMonth { day, month },
        ));
    }
    Ok(())
}
