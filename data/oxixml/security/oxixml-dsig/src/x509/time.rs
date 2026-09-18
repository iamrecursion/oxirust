//! Instants, and the two ways X.509 spells one.
//!
//! [RFC 5280 §4.1.2.5](https://www.rfc-editor.org/rfc/rfc5280#section-4.1.2.5)
//! gives a certificate's `Validity` two encodings and pins both of them to UTC
//! with the seconds always present:
//!
//! * `UTCTime`, `YYMMDDHHMMSSZ`, used for every date through 2049, with `YY`
//!   from 50 to 99 meaning the 1900s and 00 to 49 the 2000s;
//! * `GeneralizedTime`, `YYYYMMDDHHMMSSZ`, used for 2050 and later, with
//!   fractional seconds forbidden.
//!
//! Anything else — a missing `Z`, a numeric offset, an omitted seconds field, a
//! fraction — is refused rather than guessed at. A certificate whose validity
//! cannot be read is a certificate whose validity cannot be enforced.
//!
//! Both encodings become a [`Timestamp`], a count of seconds from the Unix
//! epoch, which is also the type a caller passes to say what "now" means. The
//! conversion is pure integer arithmetic on the proleptic Gregorian calendar
//! and knows nothing about leap seconds, which is the right model here: X.509
//! validity is compared, never subtracted.

use std::fmt;

/// An instant, as a count of seconds from the Unix epoch, 1970-01-01T00:00:00Z.
///
/// This is both what a certificate's `notBefore` and `notAfter` decode to and
/// what the caller hands [`PathValidator`](super::PathValidator) to say what
/// time validation should be performed at. **The library never reads a clock**:
/// supplying the instant is the caller's job, which keeps path validation
/// deterministic, testable, and usable for "was this signature valid when it
/// was made?" as well as for "is it valid now?".
///
/// # Example
///
/// ```
/// use oxixml_dsig::Timestamp;
///
/// let midnight = Timestamp::from_unix_seconds(1_767_225_600);
/// assert_eq!(midnight.to_string(), "2026-01-01T00:00:00Z");
/// assert_eq!(midnight.unix_seconds(), 1_767_225_600);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp {
    seconds: i64,
}

impl Timestamp {
    /// The instant `seconds` after the Unix epoch, or before it when negative.
    #[must_use]
    pub const fn from_unix_seconds(seconds: i64) -> Self {
        Self { seconds }
    }

    /// The instant as seconds from the Unix epoch.
    #[must_use]
    pub const fn unix_seconds(self) -> i64 {
        self.seconds
    }

    /// The instant as a `(year, month, day, hour, minute, second)` tuple in UTC,
    /// with the month and day one-based.
    #[must_use]
    pub const fn to_utc(self) -> (i64, u32, u32, u32, u32, u32) {
        let days = self.seconds.div_euclid(SECONDS_PER_DAY);
        let remainder = self.seconds.rem_euclid(SECONDS_PER_DAY);
        let (year, month, day) = civil_from_days(days);
        // `remainder` is in `0..86_400`, so each of these fits in a `u32`.
        let hour = (remainder / 3_600) as u32;
        let minute = ((remainder % 3_600) / 60) as u32;
        let second = (remainder % 60) as u32;
        (year, month, day, hour, minute, second)
    }
}

impl fmt::Display for Timestamp {
    /// Writes the instant in the `YYYY-MM-DDTHH:MM:SSZ` form of
    /// [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html), which
    /// is what an error message about an expired certificate should say.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (year, month, day, hour, minute, second) = self.to_utc();
        write!(
            f,
            "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
        )
    }
}

/// Seconds in a day, the only unit conversion this module needs.
const SECONDS_PER_DAY: i64 = 86_400;

/// Days from the Unix epoch to 0000-03-01, the shifted origin the civil-date
/// conversions below count from.
const DAYS_TO_SHIFTED_EPOCH: i64 = 719_468;

/// Days in one 400-year Gregorian era.
const DAYS_PER_ERA: i64 = 146_097;

/// Parses an X.509 `UTCTime`: exactly `YYMMDDHHMMSSZ`.
pub(crate) fn parse_utc_time(text: &[u8]) -> Option<Timestamp> {
    let [year_high, year_low, rest @ ..] = text else {
        return None;
    };
    let two_digit_year = i64::from(digit(*year_high)?) * 10 + i64::from(digit(*year_low)?);
    // RFC 5280 §4.1.2.5.1: 50 and above is the twentieth century, below it the
    // twenty-first. The encoding therefore cannot express 2050 or later, which
    // is why `GeneralizedTime` exists.
    let year = if two_digit_year >= 50 {
        1900 + two_digit_year
    } else {
        2000 + two_digit_year
    };
    parse_month_onwards(year, rest)
}

/// Parses an X.509 `GeneralizedTime`: exactly `YYYYMMDDHHMMSSZ`.
pub(crate) fn parse_generalized_time(text: &[u8]) -> Option<Timestamp> {
    let [a, b, c, d, rest @ ..] = text else {
        return None;
    };
    let year = i64::from(digit(*a)?) * 1_000
        + i64::from(digit(*b)?) * 100
        + i64::from(digit(*c)?) * 10
        + i64::from(digit(*d)?);
    parse_month_onwards(year, rest)
}

/// Parses the `MMDDHHMMSSZ` both encodings share, given the year the digits in
/// front of it produced.
fn parse_month_onwards(year: i64, text: &[u8]) -> Option<Timestamp> {
    let [m1, m2, d1, d2, h1, h2, min1, min2, s1, s2, b'Z'] = text else {
        return None;
    };
    let month = u32::from(digit(*m1)?) * 10 + u32::from(digit(*m2)?);
    let day = u32::from(digit(*d1)?) * 10 + u32::from(digit(*d2)?);
    let hour = i64::from(digit(*h1)?) * 10 + i64::from(digit(*h2)?);
    let minute = i64::from(digit(*min1)?) * 10 + i64::from(digit(*min2)?);
    let second = i64::from(digit(*s1)?) * 10 + i64::from(digit(*s2)?);
    if month == 0 || month > 12 || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let seconds = days
        .checked_mul(SECONDS_PER_DAY)?
        .checked_add(hour * 3_600 + minute * 60 + second)?;
    Some(Timestamp::from_unix_seconds(seconds))
}

/// The value of an ASCII decimal digit.
const fn digit(byte: u8) -> Option<u8> {
    if byte.is_ascii_digit() {
        Some(byte - b'0')
    } else {
        None
    }
}

/// Whether a proleptic Gregorian year is a leap year.
const fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// How many days a month has, which is the check that keeps 31 February from
/// silently becoming 3 March.
const fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days from the Unix epoch to a civil date, by Howard Hinnant's `days_from_civil`.
///
/// The trick is to shift the year so that it starts on 1 March: the leap day
/// then lands at the *end* of the year, and the day-of-year becomes a closed
/// formula with no table and no branch.
const fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let month = month as i64;
    let day = day as i64;
    let shifted_year = if month <= 2 { year - 1 } else { year };
    let era = if shifted_year >= 0 {
        shifted_year
    } else {
        shifted_year - 399
    } / 400;
    let year_of_era = shifted_year - era * 400;
    let month_of_shifted_year = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_of_shifted_year + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * DAYS_PER_ERA + day_of_era - DAYS_TO_SHIFTED_EPOCH
}

/// The inverse of [`days_from_civil`], by the same construction.
const fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + DAYS_TO_SHIFTED_EPOCH;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / DAYS_PER_ERA;
    let day_of_era = shifted - era * DAYS_PER_ERA;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_of_shifted_year = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_of_shifted_year + 2) / 5 + 1;
    let month = if month_of_shifted_year < 10 {
        month_of_shifted_year + 3
    } else {
        month_of_shifted_year - 9
    };
    let year = year_of_era + era * 400 + if month <= 2 { 1 } else { 0 };
    // `month` is in `1..=12` and `day` in `1..=31`, so both fit in a `u32`.
    (year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_encodings_agree_on_the_same_instant() {
        let utc = parse_utc_time(b"260101000000Z").expect("a well-formed UTCTime");
        let generalized =
            parse_generalized_time(b"20260101000000Z").expect("a well-formed GeneralizedTime");
        assert_eq!(utc, generalized);
        assert_eq!(utc.unix_seconds(), 1_767_225_600);
        assert_eq!(utc.to_string(), "2026-01-01T00:00:00Z");
    }

    /// The 1950–2049 window `UTCTime` can express, at both ends.
    #[test]
    fn two_digit_years_pivot_at_fifty() {
        let nineties = parse_utc_time(b"990101000000Z").expect("1999");
        assert_eq!(nineties.to_string(), "1999-01-01T00:00:00Z");
        let fifty = parse_utc_time(b"500101000000Z").expect("1950");
        assert_eq!(fifty.to_string(), "1950-01-01T00:00:00Z");
        let forty_nine = parse_utc_time(b"490101000000Z").expect("2049");
        assert_eq!(forty_nine.to_string(), "2049-01-01T00:00:00Z");
    }

    #[test]
    fn the_epoch_and_the_dates_around_it_round_trip() {
        for (seconds, rendered) in [
            (0_i64, "1970-01-01T00:00:00Z"),
            (-1, "1969-12-31T23:59:59Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (4_102_444_799, "2099-12-31T23:59:59Z"),
            (-2_208_988_800, "1900-01-01T00:00:00Z"),
        ] {
            let timestamp = Timestamp::from_unix_seconds(seconds);
            assert_eq!(timestamp.to_string(), rendered);
            let (year, month, day, hour, minute, second) = timestamp.to_utc();
            assert_eq!(
                days_from_civil(year, month, day) * SECONDS_PER_DAY
                    + i64::from(hour) * 3_600
                    + i64::from(minute) * 60
                    + i64::from(second),
                seconds
            );
        }
    }

    /// Every calendar shape the encodings forbid. A date that cannot be read
    /// must not become a date that happens to parse: 31 February would decode
    /// as 3 March and quietly extend a certificate's life.
    #[test]
    fn malformed_and_impossible_dates_are_refused() {
        for text in [
            b"2601010000Z".as_slice(),       // seconds omitted
            b"260101000000".as_slice(),      // no Z
            b"260101000000+0100".as_slice(), // an offset, which RFC 5280 forbids
            b"260101000000.5Z".as_slice(),   // a fraction
            b"26010100000Z".as_slice(),      // one digit short
            b"2601010000000Z".as_slice(),    // one digit long
            b"2600010000000Z".as_slice(),
            b"26xx01000000Z".as_slice(), // not digits
            b"260001000000Z".as_slice(), // month 0
            b"261301000000Z".as_slice(), // month 13
            b"260100000000Z".as_slice(), // day 0
            b"260132000000Z".as_slice(), // day 32
            b"260229000000Z".as_slice(), // 2026 is not a leap year
            b"260101240000Z".as_slice(), // hour 24
            b"260101006000Z".as_slice(), // minute 60
            b"260101000060Z".as_slice(), // second 60
            b"".as_slice(),
        ] {
            assert_eq!(
                parse_utc_time(text),
                None,
                "{:?}",
                String::from_utf8_lossy(text)
            );
        }
        assert_eq!(parse_generalized_time(b"20260229000000Z"), None);
        assert_eq!(parse_generalized_time(b"260101000000Z"), None);
        // …but the leap day of a leap year is a real date.
        assert!(parse_generalized_time(b"20240229000000Z").is_some());
        assert!(parse_utc_time(b"000229000000Z").is_some());
    }

    /// 1900 is not a leap year and 2000 is: the two cases the `%100` and `%400`
    /// rules exist for.
    #[test]
    fn the_century_leap_year_rules_are_applied() {
        assert!(parse_generalized_time(b"19000229000000Z").is_none());
        assert!(parse_generalized_time(b"20000229000000Z").is_some());
    }
}
