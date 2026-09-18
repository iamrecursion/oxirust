//! Hand-rolled ISO 8601 date/time parser (no `chrono`, no external
//! dependencies).
//!
//! # Grammar
//!
//! ```text
//! [+-]YYYY[YY]-MM-DD[THH:MM[:SS[.fraction]]][Z|(+|-)HH:MM|(+|-)HHMM]
//! ```
//!
//! - The year is 4, 5, or 6 ASCII digits, astronomical year numbering
//!   (year `0000` = 1 BCE, year `-0001` = 2 BCE, matching
//!   [`oxiephemeris_core::time::calendar`]). A leading `-` selects a
//!   negative year for BCE dates; `+` is accepted but not required for
//!   non-negative years. Years beyond the plain 4-digit range (the ISO 8601
//!   "expanded representation") use 5 or 6 digits.
//! - `T`/`t` introduces the time of day; minutes are mandatory once a time
//!   is present, seconds (with an optional fractional part) are optional.
//! - The UTC offset (`Z` for zero, or a signed `HH:MM`/`HHMM`/`HH`) is
//!   optional and only meaningful when a time of day is present. When
//!   omitted the clock reading is taken as-is (zero offset).
//!
//! This module only validates *format* (digit counts, separators, and the
//! coarse ranges month 1-12 / day 1-31 / hour 0-23 / minute 0-59 /
//! second 0-60). Calendar-specific day-of-month validity (e.g. rejecting
//! February 30, honoring leap years) is *not* checked here, because it
//! depends on which calendar (`--cal`) the caller selects for a given date
//! string — a choice this module has no visibility into. Callers run
//! [`crate::calendar::check_calendar_day`] right after [`parse`] (and before
//! [`oxiephemeris_core::time::calendar::julday`]) to catch that case with a
//! friendly, calendar-aware message instead of
//! `oxiephemeris_core`'s generic `CoreError::InvalidInput`;
//! `julday`/`CalendarDate::new` still re-check it independently as a
//! defense-in-depth backstop.

use core::fmt;

/// A parsed ISO 8601 timestamp, still in "local clock reading" form: the
/// UTC offset has not yet been applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Iso8601 {
    /// Astronomical year number (may be zero or negative).
    pub year: i32,
    /// Month, 1-12.
    pub month: u8,
    /// Day of month, 1-31 (format-valid only; not calendar-checked here).
    pub day: u8,
    /// Hour of day, 0-23.
    pub hour: u8,
    /// Minute, 0-59.
    pub minute: u8,
    /// Second, including any fractional part, in `[0, 61)` (61 tolerates a
    /// leap-second reading `:60` plus a fraction; the caller's calendar
    /// arithmetic folds any overflow into the next minute).
    pub second: f64,
    /// UTC offset in minutes: `local = UTC + offset_minutes`. Zero if `Z`
    /// or no offset/time was given.
    pub offset_minutes: i32,
}

impl Iso8601 {
    /// The day-of-day reading in hours, with the UTC offset already
    /// subtracted (`local - offset = UTC`), suitable to pass directly as
    /// the `hours` argument of
    /// [`julday`](oxiephemeris_core::time::calendar::julday): out-of-range
    /// values (negative, or `>= 24`) are folded into adjacent calendar days
    /// by that function.
    #[must_use]
    pub fn utc_hours(self) -> f64 {
        f64::from(self.hour) + f64::from(self.minute) / 60.0 + self.second / 3600.0
            - f64::from(self.offset_minutes) / 60.0
    }
}

/// Errors from [`parse`]. Each variant names the first grammar element that
/// failed to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iso8601Error {
    /// The input string was empty.
    Empty,
    /// The input contains a non-ASCII byte.
    NonAscii,
    /// The year field was not 4-6 ASCII digits.
    BadYear,
    /// A `-` date separator was expected but not found.
    BadDateSeparator,
    /// The month was not two digits in `01..=12`.
    BadMonth,
    /// The day was not two digits in `01..=31`.
    BadDay,
    /// A `:` time separator was expected but not found.
    BadTimeSeparator,
    /// The hour was not two digits in `00..=23`.
    BadHour,
    /// The minute was not two digits in `00..=59`.
    BadMinute,
    /// The second (and optional fraction) was not valid.
    BadSecond,
    /// The `Z`/`+HH:MM`/`-HH:MM` UTC offset was not valid.
    BadOffset,
    /// Extra characters followed a complete, valid timestamp.
    TrailingGarbage,
}

impl fmt::Display for Iso8601Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Empty => "empty date/time string",
            Self::NonAscii => "date/time string contains non-ASCII bytes",
            Self::BadYear => "year must be 4-6 digits (optionally signed)",
            Self::BadDateSeparator => "expected '-' between date components",
            Self::BadMonth => "month must be two digits, 01-12",
            Self::BadDay => "day must be two digits, 01-31",
            Self::BadTimeSeparator => "expected ':' between time components",
            Self::BadHour => "hour must be two digits, 00-23",
            Self::BadMinute => "minute must be two digits, 00-59",
            Self::BadSecond => "second must be two digits (00-60), optionally with a fraction",
            Self::BadOffset => "UTC offset must be 'Z' or (+|-)HH[:]MM",
            Self::TrailingGarbage => "unexpected trailing characters",
        };
        f.write_str(msg)
    }
}

impl core::error::Error for Iso8601Error {}

/// Reads exactly two ASCII digits at `*pos`, advancing `*pos` by 2.
fn two_digits(bytes: &[u8], pos: &mut usize, err: Iso8601Error) -> Result<u8, Iso8601Error> {
    if *pos + 2 > bytes.len() || !bytes[*pos].is_ascii_digit() || !bytes[*pos + 1].is_ascii_digit()
    {
        return Err(err);
    }
    let value = 10 * (bytes[*pos] - b'0') + (bytes[*pos + 1] - b'0');
    *pos += 2;
    Ok(value)
}

/// Parses `[+-]YYYY[YY]-MM-DD`, returning `(year, month, day)` and leaving
/// `*i` positioned just past the day field.
fn parse_date(s: &str, i: &mut usize) -> Result<(i32, u8, u8), Iso8601Error> {
    let bytes = s.as_bytes();
    let sign: i32 = match bytes[*i] {
        b'+' => {
            *i += 1;
            1
        }
        b'-' => {
            *i += 1;
            -1
        }
        _ => 1,
    };

    let year_start = *i;
    while *i < bytes.len() && bytes[*i].is_ascii_digit() && (*i - year_start) < 6 {
        *i += 1;
    }
    let year_len = *i - year_start;
    if !(4..=6).contains(&year_len) {
        return Err(Iso8601Error::BadYear);
    }
    let year_mag: i32 = s[year_start..*i]
        .parse()
        .map_err(|_| Iso8601Error::BadYear)?;
    let year = sign * year_mag;

    if *i >= bytes.len() || bytes[*i] != b'-' {
        return Err(Iso8601Error::BadDateSeparator);
    }
    *i += 1;
    let month = two_digits(bytes, i, Iso8601Error::BadMonth)?;
    if !(1..=12).contains(&month) {
        return Err(Iso8601Error::BadMonth);
    }

    if *i >= bytes.len() || bytes[*i] != b'-' {
        return Err(Iso8601Error::BadDateSeparator);
    }
    *i += 1;
    let day = two_digits(bytes, i, Iso8601Error::BadDay)?;
    if !(1..=31).contains(&day) {
        return Err(Iso8601Error::BadDay);
    }

    Ok((year, month, day))
}

/// Parses the optional fractional-seconds tail (`.` + digits) at `*i`,
/// returning the fraction in `[0, 1)`, or `0.0` if there is none.
fn parse_fraction(bytes: &[u8], i: &mut usize) -> Result<f64, Iso8601Error> {
    if *i >= bytes.len() || bytes[*i] != b'.' {
        return Ok(0.0);
    }
    *i += 1;
    let frac_start = *i;
    while *i < bytes.len() && bytes[*i].is_ascii_digit() {
        *i += 1;
    }
    if *i == frac_start {
        return Err(Iso8601Error::BadSecond);
    }
    let mut frac = 0.0f64;
    let mut scale = 0.1f64;
    for &b in &bytes[frac_start..*i] {
        frac += f64::from(b - b'0') * scale;
        scale *= 0.1;
    }
    Ok(frac)
}

/// Parses the optional `Z`/`+HH:MM`/`-HHMM` UTC offset at `*i`, in minutes.
fn parse_offset(bytes: &[u8], i: &mut usize) -> Result<i32, Iso8601Error> {
    if *i >= bytes.len() {
        return Ok(0);
    }
    match bytes[*i] {
        b'Z' | b'z' => {
            *i += 1;
            Ok(0)
        }
        b'+' | b'-' => {
            let osign: i32 = if bytes[*i] == b'-' { -1 } else { 1 };
            *i += 1;
            let oh = two_digits(bytes, i, Iso8601Error::BadOffset)?;
            if *i < bytes.len() && bytes[*i] == b':' {
                *i += 1;
            }
            let om = two_digits(bytes, i, Iso8601Error::BadOffset)?;
            if oh > 23 || om > 59 {
                return Err(Iso8601Error::BadOffset);
            }
            Ok(osign * (i32::from(oh) * 60 + i32::from(om)))
        }
        _ => Err(Iso8601Error::TrailingGarbage),
    }
}

/// Parses `THH:MM[:SS[.fraction]][offset]`, returning
/// `(hour, minute, second, offset_minutes)`. Assumes `bytes[*i]` is `T`/`t`.
fn parse_time(bytes: &[u8], i: &mut usize) -> Result<(u8, u8, f64, i32), Iso8601Error> {
    *i += 1; // consume 'T'/'t'

    let hour = two_digits(bytes, i, Iso8601Error::BadHour)?;
    if hour > 23 {
        return Err(Iso8601Error::BadHour);
    }
    if *i >= bytes.len() || bytes[*i] != b':' {
        return Err(Iso8601Error::BadTimeSeparator);
    }
    *i += 1;

    let minute = two_digits(bytes, i, Iso8601Error::BadMinute)?;
    if minute > 59 {
        return Err(Iso8601Error::BadMinute);
    }

    let mut second = 0.0f64;
    if *i < bytes.len() && bytes[*i] == b':' {
        *i += 1;
        let sec_int = two_digits(bytes, i, Iso8601Error::BadSecond)?;
        if sec_int > 60 {
            return Err(Iso8601Error::BadSecond);
        }
        second = f64::from(sec_int) + parse_fraction(bytes, i)?;
    }

    let offset_minutes = parse_offset(bytes, i)?;
    Ok((hour, minute, second, offset_minutes))
}

/// Parses an ISO 8601 date/time string per the grammar documented at the
/// top of this module.
///
/// # Errors
///
/// Returns the specific [`Iso8601Error`] variant naming the first grammar
/// element that failed to parse.
pub fn parse(s: &str) -> Result<Iso8601, Iso8601Error> {
    if s.is_empty() {
        return Err(Iso8601Error::Empty);
    }
    if !s.is_ascii() {
        return Err(Iso8601Error::NonAscii);
    }
    let bytes = s.as_bytes();
    let mut i = 0usize;

    let (year, month, day) = parse_date(s, &mut i)?;

    let (hour, minute, second, offset_minutes) =
        if i < bytes.len() && (bytes[i] == b'T' || bytes[i] == b't') {
            parse_time(bytes, &mut i)?
        } else {
            (0, 0, 0.0, 0)
        };

    if i != bytes.len() {
        return Err(Iso8601Error::TrailingGarbage);
    }

    Ok(Iso8601 {
        year,
        month,
        day,
        hour,
        minute,
        second,
        offset_minutes,
    })
}

#[cfg(test)]
mod tests {
    use super::{parse, Iso8601Error};

    #[test]
    fn date_only() -> Result<(), Iso8601Error> {
        let d = parse("2024-03-15")?;
        assert_eq!(d.year, 2024);
        assert_eq!(d.month, 3);
        assert_eq!(d.day, 15);
        assert_eq!(d.hour, 0);
        assert_eq!(d.minute, 0);
        assert!((d.second - 0.0).abs() < 1e-15);
        assert_eq!(d.offset_minutes, 0);
        Ok(())
    }

    #[test]
    fn seconds_and_fraction() -> Result<(), Iso8601Error> {
        let d = parse("2024-03-15T12:30:45.123456")?;
        assert_eq!(d.hour, 12);
        assert_eq!(d.minute, 30);
        assert!((d.second - 45.123_456).abs() < 1e-9);
        assert_eq!(d.offset_minutes, 0);
        Ok(())
    }

    #[test]
    fn zulu_offset() -> Result<(), Iso8601Error> {
        let d = parse("2024-03-15T12:30:45Z")?;
        assert_eq!(d.offset_minutes, 0);
        let d = parse("2024-03-15T12:30:45z")?;
        assert_eq!(d.offset_minutes, 0);
        Ok(())
    }

    #[test]
    fn positive_offset_with_colon() -> Result<(), Iso8601Error> {
        let d = parse("2024-03-15T12:30:45+09:00")?;
        assert_eq!(d.offset_minutes, 9 * 60);
        Ok(())
    }

    #[test]
    fn negative_offset_without_colon() -> Result<(), Iso8601Error> {
        let d = parse("2024-03-15T12:30:45-0530")?;
        assert_eq!(d.offset_minutes, -(5 * 60 + 30));
        Ok(())
    }

    #[test]
    fn negative_year_is_bce_astronomical() -> Result<(), Iso8601Error> {
        // Year -1 == astronomical year -1 == 2 BCE.
        let d = parse("-0001-06-15")?;
        assert_eq!(d.year, -1);
        Ok(())
    }

    #[test]
    fn expanded_six_digit_year() -> Result<(), Iso8601Error> {
        let d = parse("+012000-01-01")?;
        assert_eq!(d.year, 12000);
        let d = parse("-012000-01-01")?;
        assert_eq!(d.year, -12000);
        Ok(())
    }

    #[test]
    fn utc_hours_applies_offset() -> Result<(), Iso8601Error> {
        let d = parse("2024-03-15T12:00:00+09:00")?;
        // 12:00 local at UTC+9 is 03:00 UTC.
        assert!((d.utc_hours() - 3.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn bad_month_rejected() {
        assert_eq!(parse("2024-13-01"), Err(Iso8601Error::BadMonth));
    }

    #[test]
    fn bad_day_rejected() {
        assert_eq!(parse("2024-01-32"), Err(Iso8601Error::BadDay));
    }

    #[test]
    fn bad_offset_rejected() {
        assert_eq!(
            parse("2024-03-15T12:00:00+25:00"),
            Err(Iso8601Error::BadOffset)
        );
    }

    #[test]
    fn garbage_rejected() {
        assert!(parse("not-a-date").is_err());
        assert_eq!(parse(""), Err(Iso8601Error::Empty));
    }

    #[test]
    fn trailing_garbage_rejected() {
        assert_eq!(
            parse("2024-03-15T12:00:00Zextra"),
            Err(Iso8601Error::TrailingGarbage)
        );
    }
}
