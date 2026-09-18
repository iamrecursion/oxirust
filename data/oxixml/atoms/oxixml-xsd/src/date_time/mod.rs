//! The XML Schema date/time datatypes built on the seven-property model.
//!
//! This module gathers the shared machinery — the internal [`Timestamp`]
//! representation, the calendar algorithms of [Appendix E of XML Schema
//! Part 2](https://www.w3.org/TR/xmlschema11-2/#vp-dt-arithmetic), the public
//! [`TimezoneOffset`] type and the error types — while the concrete datatypes
//! live in the [`main_types`] and [`gregorian`] submodules and the lexical
//! grammar lives in [`parser`].

use crate::{DayTimeDuration, Decimal, Duration};
use std::cmp::{Ordering, min};
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use std::time::Duration as StdDuration;

mod gregorian;
mod main_types;
mod parser;
#[cfg(test)]
mod tests;

pub use self::gregorian::{GDay, GMonth, GMonthDay, GYear, GYearMonth};
pub use self::main_types::{Date, DateTime, Time};

/// The largest year this implementation admits in a date, `dateTime`, `gYear` or
/// `gYearMonth` value: `4294967295`, the magnitude an unsigned 32-bit integer can hold.
///
/// XML Schema itself sets no upper bound. [Production `yearFrag`](https://www.w3.org/TR/xmlschema11-2/#dt-yearFrag)
/// is "four digits *at least*", and [XSD 1.1 Part 2 §3.3.7](https://www.w3.org/TR/xmlschema11-2/#dateTime)
/// says of the `dateTime` value space that "the year is an unbounded integer", while
/// [F&O 3.1 §18.3](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes) requires
/// only that an implementation support years `1` through `9999` and leaves everything past
/// that *implementation-defined*. This is that definition: every year whose magnitude fits
/// [`u32`] is supported, and reaching past it is an **overflow** —
/// [`ParseDateTimeError::is_overflow`] answers `true`, which an XPath layer reports as
/// [`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001) rather than as the
/// lexical `FORG0001`.
///
/// # Where the range is enforced
///
/// Wherever a year *component* is supplied or recomputed: every lexical form, every
/// year-taking constructor, the year–month duration arithmetic of [Appendix
/// E](https://www.w3.org/TR/xmlschema11-2/#vp-dt-arithmetic) — which derives a new year from an
/// old one — and every operation on a [`Date`], [`GYear`] or [`GYearMonth`], each of which
/// rebuilds itself from its year on the way back from the [`DateTime`] it borrows.
///
/// One path is deliberately outside it: a [`DateTime`] moved by a *day–time* duration stays on
/// the timeline and never recomputes a year, so it is bounded by the internal encoding — a
/// [`Decimal`] count of seconds, reaching some three orders of magnitude further — rather than
/// by this range. A large enough `xs:dayTimeDuration` can therefore carry a `dateTime` a little
/// past the boundary instead of overflowing at it. Nothing *reads* such a value back in: it is
/// past the end of the lexical space, and no constructor will accept it again.
///
/// ```
/// use oxixml_xsd::{Date, MAX_YEAR};
/// use std::str::FromStr;
///
/// assert_eq!(MAX_YEAR, 4_294_967_295);
/// assert!(Date::from_str("4294967295-01-02").is_ok());
///
/// let past_the_end = Date::from_str("4294967296-01-02").unwrap_err();
/// assert!(past_the_end.is_overflow());
/// ```
pub const MAX_YEAR: i64 = u32::MAX as i64;

/// The smallest year this implementation admits, `-4294967295` — the negation of
/// [`MAX_YEAR`]; see it for the reasoning and for what "unsupported" means.
///
/// The range is symmetric because XSD 1.1 admits the year zero (XSD 1.0 did not), so the
/// negative years are the mirror image of the positive ones rather than being shifted by one.
pub const MIN_YEAR: i64 = -MAX_YEAR;

/// Whether `year` lies within the supported range, [`MIN_YEAR`]`..=`[`MAX_YEAR`].
#[inline]
const fn year_is_supported(year: i64) -> bool {
    MIN_YEAR <= year && year <= MAX_YEAR
}

/// A timezone offset with respect to UTC.
///
/// It is encoded as a number of minutes between -PT14H and PT14H.
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct TimezoneOffset {
    offset: i16, // in minutes with respect to UTC
}

impl TimezoneOffset {
    /// The maximal timezone offset, `+14:00`.
    pub const MAX: Self = Self { offset: 14 * 60 };
    /// The minimal timezone offset, `-14:00`.
    pub const MIN: Self = Self { offset: -14 * 60 };
    /// The UTC (`Z`) timezone offset.
    pub const UTC: Self = Self { offset: 0 };

    /// Builds a timezone offset from a number of minutes with respect to UTC.
    ///
    /// Returns an error if the offset is outside the `-14:00`..=`+14:00` range.
    #[inline]
    pub fn new(offset_in_minutes: i16) -> Result<Self, InvalidTimezoneError> {
        let value = Self {
            offset: offset_in_minutes,
        };
        if Self::MIN <= value && value <= Self::MAX {
            Ok(value)
        } else {
            Err(InvalidTimezoneError {
                offset_in_minutes: offset_in_minutes.into(),
            })
        }
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 2]) -> Self {
        Self {
            offset: i16::from_be_bytes(bytes),
        }
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 2] {
        self.offset.to_be_bytes()
    }
}

impl TryFrom<DayTimeDuration> for TimezoneOffset {
    type Error = InvalidTimezoneError;

    #[inline]
    fn try_from(value: DayTimeDuration) -> Result<Self, Self::Error> {
        let offset_in_minutes = value.minutes() + value.hours() * 60;
        let result = Self::new(
            offset_in_minutes
                .try_into()
                .map_err(|_| Self::Error { offset_in_minutes })?,
        )?;
        if DayTimeDuration::from(result) == value {
            Ok(result)
        } else {
            // The value is not an integral number of minutes or has overflowed.
            Err(Self::Error { offset_in_minutes })
        }
    }
}

impl TryFrom<Duration> for TimezoneOffset {
    type Error = InvalidTimezoneError;

    #[inline]
    fn try_from(value: Duration) -> Result<Self, Self::Error> {
        DayTimeDuration::try_from(value)
            .map_err(|_| Self::Error {
                offset_in_minutes: 0,
            })?
            .try_into()
    }
}

impl From<TimezoneOffset> for DayTimeDuration {
    #[inline]
    fn from(value: TimezoneOffset) -> Self {
        Self::new(i64::from(value.offset) * 60)
    }
}

impl From<TimezoneOffset> for Duration {
    #[inline]
    fn from(value: TimezoneOffset) -> Self {
        DayTimeDuration::from(value).into()
    }
}

impl fmt::Display for TimezoneOffset {
    /// Formats the canonical representation: `Z` for UTC, otherwise `±HH:MM`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.offset {
            0 => f.write_str("Z"),
            offset if offset < 0 => write!(f, "-{:02}:{:02}", -offset / 60, -offset % 60),
            offset => write!(f, "+{:02}:{:02}", offset / 60, offset % 60),
        }
    }
}

/// [The Date/time Seven-property model](https://www.w3.org/TR/xmlschema11-2/#dt-dt-7PropMod)
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
struct DateTimeSevenPropertyModel {
    year: Option<i64>,
    month: Option<u8>,
    day: Option<u8>,
    hour: Option<u8>,
    minute: Option<u8>,
    second: Option<Decimal>,
    timezone_offset: Option<TimezoneOffset>,
}

/// The internal representation shared by every date/time datatype: a number of
/// seconds on the Gregorian timeline together with an optional timezone offset.
#[derive(Debug, Clone, Copy)]
struct Timestamp {
    value: Decimal,
    timezone_offset: Option<TimezoneOffset>,
}

impl PartialEq for Timestamp {
    /// [`op:dateTime-equal`](https://www.w3.org/TR/xpath-functions-31/#func-dateTime-equal)
    /// with the *default* implicit timezone, UTC.
    ///
    /// A timezone-less timestamp keeps its literal reading in [`Timestamp::value`] — exactly
    /// what a UTC implicit timezone maps it to — so normalisation degenerates into a plain
    /// value comparison. Use [`Timestamp::eq_with_implicit_timezone`] to supply a different
    /// implicit timezone.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl Eq for Timestamp {}

impl PartialOrd for Timestamp {
    /// [`op:dateTime-less-than`/`-greater-than`](https://www.w3.org/TR/xpath-functions-31/#func-dateTime-less-than)
    /// with the *default* implicit timezone, UTC.
    ///
    /// Unlike the XSD 1.1 order relation (available as [`Timestamp::xsd_partial_cmp`]), this
    /// is a **total** order: applying the implicit timezone makes every pair comparable.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl Hash for Timestamp {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Timestamp {
    const MAX: Self = Self {
        value: Decimal::MAX,
        timezone_offset: Some(TimezoneOffset::MAX),
    };
    const MIN: Self = Self {
        value: Decimal::MIN,
        timezone_offset: Some(TimezoneOffset::MIN),
    };

    /// The single seam every date/time constructor passes through, and therefore the one
    /// place [`MAX_YEAR`] is enforced: a year outside the supported range is an overflow of
    /// the value space, not a lexical error, so it reaches an XPath caller as `FODT0001`.
    #[inline]
    fn new(props: &DateTimeSevenPropertyModel) -> Result<Self, DateTimeOverflowError> {
        if let Some(year) = props.year
            && !year_is_supported(year)
        {
            return Err(DateTimeOverflowError);
        }
        Ok(Self {
            timezone_offset: props.timezone_offset,
            value: time_on_timeline(props).ok_or(DateTimeOverflowError)?,
        })
    }

    #[inline]
    fn checked_now() -> Option<Self> {
        Self::new(&date_time_plus_duration(
            since_unix_epoch()?,
            &DateTimeSevenPropertyModel {
                year: Some(1970),
                month: Some(1),
                day: Some(1),
                hour: Some(0),
                minute: Some(0),
                second: Some(Decimal::default()),
                timezone_offset: Some(TimezoneOffset::UTC),
            },
        )?)
        .ok()
    }

    #[inline]
    fn from_be_bytes(bytes: [u8; 18]) -> Self {
        Self {
            value: Decimal::from_be_bytes(
                bytes[0..16].try_into().expect("16 bytes for the timestamp"),
            ),
            timezone_offset: if bytes[16..18] == [u8::MAX; 2] {
                None
            } else {
                Some(TimezoneOffset::from_be_bytes(
                    bytes[16..18].try_into().expect("2 bytes for the timezone"),
                ))
            },
        }
    }

    #[inline]
    #[must_use]
    fn year_month_day(&self) -> (i64, u8, u8) {
        let mut days = (self.value.as_i128()
            + i128::from(self.timezone_offset.unwrap_or(TimezoneOffset::UTC).offset) * 60)
            .div_euclid(86400)
            + 366;

        // Make days positive.
        let shift = if days < 0 {
            let shift = days / 146_097 - 1;
            days -= shift * 146_097;
            shift * 400
        } else {
            0
        };

        let year_mul_400 = days / 146_097;
        days -= year_mul_400 * 146_097;

        days -= 1;
        let year_mul_100 = days / 36524;
        days -= year_mul_100 * 36524;

        days += 1;
        let year_mul_4 = days / 1461;
        days -= year_mul_4 * 1461;

        days -= 1;
        let year_mod_4 = days / 365;
        days -= year_mod_4 * 365;

        let year =
            (400 * year_mul_400 + 100 * year_mul_100 + 4 * year_mul_4 + year_mod_4 + shift) as i64;

        let is_leap_year = (year_mul_100 == 0 || year_mul_4 != 0) && year_mod_4 == 0;
        days += i128::from(is_leap_year);

        let mut month = 0;
        for month_i in 1..=12 {
            let days_in_month = i128::from(days_in_month(Some(year), month_i));
            if days_in_month > days {
                month = month_i;
                break;
            }
            days -= days_in_month;
        }
        let day = days as u8 + 1;

        (year, month, day)
    }

    #[inline]
    #[must_use]
    fn year(&self) -> i64 {
        let (year, _, _) = self.year_month_day();
        year
    }

    #[inline]
    #[must_use]
    fn month(&self) -> u8 {
        let (_, month, _) = self.year_month_day();
        month
    }

    #[inline]
    #[must_use]
    fn day(&self) -> u8 {
        let (_, _, day) = self.year_month_day();
        day
    }

    #[inline]
    #[must_use]
    fn hour(&self) -> u8 {
        (((self.value.as_i128()
            + i128::from(self.timezone_offset.unwrap_or(TimezoneOffset::UTC).offset) * 60)
            .rem_euclid(86400))
            / 3600) as u8
    }

    #[inline]
    #[must_use]
    fn minute(&self) -> u8 {
        (((self.value.as_i128()
            + i128::from(self.timezone_offset.unwrap_or(TimezoneOffset::UTC).offset) * 60)
            .rem_euclid(3600))
            / 60) as u8
    }

    #[inline]
    #[must_use]
    fn second(&self) -> Decimal {
        self.value
            .checked_rem_euclid(60)
            .and_then(Decimal::checked_abs)
            .expect("A remainder by 60 and its absolute value are always defined")
    }

    #[inline]
    #[must_use]
    const fn timezone_offset(&self) -> Option<TimezoneOffset> {
        self.timezone_offset
    }

    #[inline]
    #[must_use]
    fn checked_add_seconds(&self, seconds: impl Into<Decimal>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_add(seconds.into())?,
            timezone_offset: self.timezone_offset,
        })
    }

    /// Replaces a missing timezone offset with `implicit_timezone`, keeping the literal
    /// reading of the value — the
    /// [implicit timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone)
    /// substitution mandated by F&O whenever a timezone-less value meets a timezoned one.
    ///
    /// Returns [`None`] only when the shift would push the value outside [`Decimal`]'s range.
    #[inline]
    #[must_use]
    fn checked_with_implicit_timezone(self, implicit_timezone: TimezoneOffset) -> Option<Self> {
        Some(match self.timezone_offset {
            Some(_) => self,
            None => Self {
                value: self
                    .value
                    .checked_sub(i64::from(implicit_timezone.offset) * 60)?,
                timezone_offset: Some(implicit_timezone),
            },
        })
    }

    /// [`checked_with_implicit_timezone`](Self::checked_with_implicit_timezone), saturating on
    /// overflow so that comparisons stay total at the very edges of the timeline.
    #[inline]
    #[must_use]
    fn with_implicit_timezone(self, implicit_timezone: TimezoneOffset) -> Self {
        self.checked_with_implicit_timezone(implicit_timezone)
            .unwrap_or(self)
    }

    /// [`op:dateTime-equal`](https://www.w3.org/TR/xpath-functions-31/#func-dateTime-equal)
    /// under an explicit implicit timezone.
    #[inline]
    #[must_use]
    fn eq_with_implicit_timezone(self, other: Self, implicit_timezone: TimezoneOffset) -> bool {
        self.with_implicit_timezone(implicit_timezone)
            .value
            .eq(&other.with_implicit_timezone(implicit_timezone).value)
    }

    /// [`op:dateTime-less-than`](https://www.w3.org/TR/xpath-functions-31/#func-dateTime-less-than)
    /// under an explicit implicit timezone. Always determinate.
    #[inline]
    #[must_use]
    fn cmp_with_implicit_timezone(
        self,
        other: Self,
        implicit_timezone: TimezoneOffset,
    ) -> Ordering {
        self.with_implicit_timezone(implicit_timezone)
            .value
            .cmp(&other.with_implicit_timezone(implicit_timezone).value)
    }

    /// The [XSD 1.1 order relation](https://www.w3.org/TR/xmlschema11-2/#dateTime-order): a
    /// *partial* order in which a timezoned and a timezone-less value are indeterminate
    /// whenever the ±14:00 window of the latter straddles the former.
    ///
    /// This is what XSD facet checking (`minInclusive`, `maxExclusive`, …) is defined in terms
    /// of; F&O comparison — the default [`PartialOrd`] impl — instead applies the implicit
    /// timezone and is always determinate.
    #[inline]
    #[must_use]
    fn xsd_partial_cmp(self, other: Self) -> Option<Ordering> {
        match (self.timezone_offset, other.timezone_offset) {
            (Some(_), Some(_)) | (None, None) => self.value.partial_cmp(&other.value),
            (Some(_), None) => {
                let plus_result = self
                    .value
                    .partial_cmp(&(other.value.checked_add(14 * 3600)?));
                let minus_result = self
                    .value
                    .partial_cmp(&(other.value.checked_sub(14 * 3600)?));
                if plus_result == minus_result {
                    plus_result
                } else {
                    None
                }
            }
            (None, Some(_)) => {
                let plus_result = self.value.checked_add(14 * 3600)?.partial_cmp(&other.value);
                let minus_result = self.value.checked_sub(14 * 3600)?.partial_cmp(&other.value);
                if plus_result == minus_result {
                    plus_result
                } else {
                    None
                }
            }
        }
    }

    /// The XSD 1.1 identity-style equality: a timezoned value is never equal to a
    /// timezone-less one, no matter the implicit timezone.
    #[inline]
    #[must_use]
    fn xsd_eq(self, other: Self) -> bool {
        self.xsd_partial_cmp(other) == Some(Ordering::Equal)
    }

    #[inline]
    #[must_use]
    fn checked_sub(&self, rhs: Self) -> Option<DayTimeDuration> {
        self.checked_sub_with_implicit_timezone(rhs, TimezoneOffset::UTC)
    }

    /// [`op:subtract-dateTimes`](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dateTimes):
    /// "If either `$arg1` or `$arg2` do not contain an explicit timezone then […] the implicit
    /// timezone provided by the dynamic evaluation context is assumed."
    #[inline]
    #[must_use]
    fn checked_sub_with_implicit_timezone(
        &self,
        rhs: Self,
        implicit_timezone: TimezoneOffset,
    ) -> Option<DayTimeDuration> {
        Some(DayTimeDuration::new(
            self.checked_with_implicit_timezone(implicit_timezone)?
                .value
                .checked_sub(rhs.checked_with_implicit_timezone(implicit_timezone)?.value)?,
        ))
    }

    #[inline]
    #[must_use]
    fn checked_sub_seconds(&self, seconds: Decimal) -> Option<Self> {
        Some(Self {
            value: self.value.checked_sub(seconds)?,
            timezone_offset: self.timezone_offset,
        })
    }

    #[inline]
    #[must_use]
    fn adjust(&self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(if let Some(from_timezone) = self.timezone_offset {
            if let Some(to_timezone) = timezone_offset {
                Self {
                    value: self.value, // We keep the timestamp.
                    timezone_offset: Some(to_timezone),
                }
            } else {
                Self {
                    value: self
                        .value
                        .checked_add(i64::from(from_timezone.offset) * 60)?, /* We keep the literal value */
                    timezone_offset: None,
                }
            }
        } else if let Some(to_timezone) = timezone_offset {
            Self {
                value: self.value.checked_sub(i64::from(to_timezone.offset) * 60)?, /* We keep the literal value */
                timezone_offset: Some(to_timezone),
            }
        } else {
            Self {
                value: self.value,
                timezone_offset: None,
            }
        })
    }

    #[inline]
    #[must_use]
    fn to_be_bytes(self) -> [u8; 18] {
        let mut bytes = [0; 18];
        bytes[0..16].copy_from_slice(&self.value.to_be_bytes());
        bytes[16..18].copy_from_slice(&match &self.timezone_offset {
            Some(timezone_offset) => timezone_offset.to_be_bytes(),
            None => [u8::MAX; 2],
        });
        bytes
    }

    #[inline]
    #[must_use]
    fn is_identical_with(self, other: Self) -> bool {
        self.value == other.value && self.timezone_offset == other.timezone_offset
    }
}

/// Generates, for a date/time datatype wrapping a [`Timestamp`] in a `timestamp` field, the
/// public comparison API that lets a caller supply the
/// [implicit timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone) of its
/// dynamic evaluation context, plus the XSD 1.1 partial order.
macro_rules! implicit_timezone_comparisons {
    ($ty:ident, $lexical:literal) => {
        impl $ty {
            #[doc = concat!("[`op:", $lexical, "-equal`](https://www.w3.org/TR/xpath-functions-31/#func-", $lexical, "-equal) using `implicit_timezone`.")]
            ///
            /// The [`PartialEq`] impl is the same comparison with the most common implicit
            /// timezone, [`TimezoneOffset::UTC`]; pass the dynamic context's own implicit
            /// timezone here when it differs.
            #[inline]
            #[must_use]
            pub fn equal_with_implicit_timezone(
                self,
                other: Self,
                implicit_timezone: $crate::TimezoneOffset,
            ) -> bool {
                self.timestamp
                    .eq_with_implicit_timezone(other.timestamp, implicit_timezone)
            }

            #[doc = concat!("Orders two `xs:", $lexical, "` values using `implicit_timezone`.")]
            ///
            /// Unlike [`Self::xsd_partial_cmp`] the result is always determinate, as F&O
            /// requires: a value without an explicit timezone is read in the implicit one.
            #[inline]
            #[must_use]
            pub fn compare_with_implicit_timezone(
                self,
                other: Self,
                implicit_timezone: $crate::TimezoneOffset,
            ) -> ::std::cmp::Ordering {
                self.timestamp
                    .cmp_with_implicit_timezone(other.timestamp, implicit_timezone)
            }

            /// The [XSD 1.1 order relation](https://www.w3.org/TR/xmlschema11-2/#dateTime-order),
            /// a *partial* order: comparing a timezoned value with a timezone-less one yields
            /// [`None`] whenever the latter's ±14:00 window straddles the former.
            ///
            /// This is the relation XSD facet checking (`minInclusive`, `maxExclusive`, …) is
            /// defined in terms of. The [`PartialOrd`] impl deliberately implements the
            /// *F&O* relation instead, which applies an implicit timezone and never returns
            /// [`None`].
            #[inline]
            #[must_use]
            pub fn xsd_partial_cmp(self, other: Self) -> Option<::std::cmp::Ordering> {
                self.timestamp.xsd_partial_cmp(other.timestamp)
            }

            /// XSD 1.1 equality: a value carrying a timezone is never equal to one without,
            /// whatever the implicit timezone. [`PartialEq`] implements the F&O relation.
            #[inline]
            #[must_use]
            pub fn xsd_eq(self, other: Self) -> bool {
                self.timestamp.xsd_eq(other.timestamp)
            }
        }
    };
}

/// Generates the implicit-timezone-aware subtraction for the three datatypes F&O gives an
/// `op:subtract-*` function to: `xs:dateTime`, `xs:date` and `xs:time`.
macro_rules! implicit_timezone_subtraction {
    ($ty:ident, $lexical:literal) => {
        impl $ty {
            #[doc = concat!("[`op:subtract-", $lexical, "s`](https://www.w3.org/TR/xpath-functions-31/#func-subtract-", $lexical, "s) using `implicit_timezone`.")]
            ///
            /// Whenever exactly one operand carries an explicit timezone, F&O 3.1 §8.4
            /// requires the other to be read in the implicit timezone of the dynamic
            /// evaluation context rather than raising an error.
            /// [`Self::checked_sub`] is this function with [`TimezoneOffset::UTC`], the
            /// most common implicit timezone.
            ///
            /// Returns [`None`] only on [`crate::Decimal`] overflow.
            #[inline]
            #[must_use]
            pub fn checked_sub_with_implicit_timezone(
                self,
                rhs: impl Into<Self>,
                implicit_timezone: $crate::TimezoneOffset,
            ) -> Option<$crate::DayTimeDuration> {
                self.timestamp
                    .checked_sub_with_implicit_timezone(rhs.into().timestamp, implicit_timezone)
            }
        }
    };
}

pub(super) use {implicit_timezone_comparisons, implicit_timezone_subtraction};

/// A source of the current time, as an elapsed [`StdDuration`] since the Unix epoch
/// (`1970-01-01T00:00:00Z`).
///
/// See [`set_now_provider`].
pub type NowProvider = fn() -> StdDuration;

static NOW_PROVIDER: OnceLock<NowProvider> = OnceLock::new();

/// Registers the clock backing [`DateTime::now`], [`Date::now`] and [`Time::now`], replacing
/// the platform clock.
///
/// This exists because [`std::time::SystemTime::now`] is not implemented on every target
/// Rust can build for. On `wasm32-unknown-unknown` in particular there is no host clock at
/// all and calling it aborts the module with an `unreachable` trap, which would make the whole
/// XPath/XQuery stack unusable there (`FnContext` timestamps every evaluation). On such a
/// target, register a provider once during start-up — typically one that reads the host's
/// clock through the JavaScript `Date.now()` binding of your choice:
///
/// ```
/// use oxixml_xsd::{DateTime, set_now_provider};
/// use std::time::Duration;
///
/// # fn host_milliseconds_since_epoch() -> f64 { 946_684_800_000.0 }
/// set_now_provider(|| Duration::from_millis(host_milliseconds_since_epoch() as u64))?;
/// assert!(DateTime::checked_now().is_some());
/// # Ok::<_, oxixml_xsd::NowProviderAlreadySetError>(())
/// ```
///
/// Deliberately dependency-free: no `js-sys`, no `wasm-bindgen`, no `unsafe` `extern` hook —
/// the provider is an ordinary safe `fn` pointer, so the choice of host binding stays with the
/// application.
///
/// # Errors
///
/// Returns [`NowProviderAlreadySetError`] if a provider was already registered, or if the
/// clock was already read once through the platform fallback. The provider can only be set
/// once so that a single evaluation never observes two different clocks.
#[inline]
pub fn set_now_provider(provider: NowProvider) -> Result<(), NowProviderAlreadySetError> {
    NOW_PROVIDER
        .set(provider)
        .map_err(|_| NowProviderAlreadySetError)
}

/// Error returned by [`set_now_provider`] when a provider has already been registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NowProviderAlreadySetError;

impl fmt::Display for NowProviderAlreadySetError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("A current-time provider has already been registered")
    }
}

impl Error for NowProviderAlreadySetError {}

/// The current time as an XSD [`Duration`] since the Unix epoch, or [`None`] when no clock is
/// available on this target and no [`NowProvider`] was registered.
fn since_unix_epoch() -> Option<Duration> {
    let elapsed = match NOW_PROVIDER.get() {
        Some(provider) => provider(),
        None => platform_since_unix_epoch()?,
    };
    elapsed.try_into().ok()
}

/// `wasm32-unknown-unknown` (and any other target without a host clock) has no
/// [`std::time::SystemTime`] implementation: calling it traps. Report "no clock" instead, so
/// that a registered [`NowProvider`] is the only way to obtain the current time there.
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
#[inline]
fn platform_since_unix_epoch() -> Option<StdDuration> {
    None
}

#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
#[inline]
fn platform_since_unix_epoch() -> Option<StdDuration> {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .ok()
}

/// The panic message shared by the infallible `now()` constructors.
const NO_CLOCK: &str = "No clock is available on this target: register one with oxixml_xsd::set_now_provider(), \
     or use the checked_now() constructors";

/// The [normalizeMonth](https://www.w3.org/TR/xmlschema11-2/#f-dt-normMo) function:
/// `yr := yr + fQuotient(mo, 1, 13)` and `mo := modulo(mo, 1, 13)`.
///
/// The two auxiliary functions of [Appendix E](https://www.w3.org/TR/xmlschema11-2/#adding-durations-to-dateTimes)
/// are *floored*: `fQuotient(a, b)` is `floor(a div b)` and `modulo(a, b)` is
/// `a - fQuotient(a, b) * b`, which for a positive `b` never goes negative.
/// Rust's `/` and `%` truncate towards zero instead, and the two disagree for
/// every `mo` outside `1..=12` — the range the caller *reaches* whenever a
/// duration carries a negative month count. `mo = 0`, January minus one month,
/// is the case that hides best: truncation answers `(yr, 2)`, a perfectly
/// well-formed February that no later normalization step has any reason to
/// correct, which would make `xs:date("1997-01-01") -
/// xs:yearMonthDuration("P1M")` yield `1997-02-01` rather than `1996-12-01`.
/// `div_euclid`/`rem_euclid` are exactly the floored pair for a positive
/// divisor, and agree with `/` and `%` over `1..=12`, so nothing inside the
/// ordinary range moves.
fn normalize_month(yr: i64, mo: i64) -> Option<(i64, u8)> {
    let shifted = mo.checked_sub(1)?;
    let yr = yr.checked_add(shifted.checked_div_euclid(12)?)?;
    let mo = u8::try_from(shifted.checked_rem_euclid(12)?.checked_add(1)?).ok()?;
    Some((yr, mo))
}

/// The [normalizeDa](https://www.w3.org/TR/xmlschema11-2/#f-dt-normDa) function.
fn normalize_day(yr: i64, mo: i64, mut da: i64) -> Option<(i64, u8, u8)> {
    let (mut yr, mut mo) = normalize_month(yr, mo)?;
    loop {
        if da <= 0 {
            let (yr2, mo2) = normalize_month(yr, i64::from(mo).checked_sub(1)?)?;
            yr = yr2;
            mo = mo2;
            da = da.checked_add(days_in_month(Some(yr), mo).into())?;
        } else if da > days_in_month(Some(yr), mo).into() {
            da = da.checked_sub(days_in_month(Some(yr), mo).into())?;
            let (yr2, mo2) = normalize_month(yr, i64::from(mo).checked_add(1)?)?;
            yr = yr2;
            mo = mo2;
        } else {
            return Some((yr, mo, u8::try_from(da).ok()?));
        }
    }
}

/// The [normalizeMinute](https://www.w3.org/TR/xmlschema11-2/#f-dt-normMi) function.
///
/// Floored throughout, for the reason given on [`normalize_month`]: a negative
/// day-time component would otherwise leave a negative hour or minute behind,
/// which the closing `u8::try_from` could only report as an overflow.
fn normalize_minute(yr: i64, mo: i64, da: i64, hr: i64, mi: i64) -> Option<(i64, u8, u8, u8, u8)> {
    let hr = hr.checked_add(mi.checked_div_euclid(60)?)?;
    let mi = mi.checked_rem_euclid(60)?;
    let da = da.checked_add(hr.checked_div_euclid(24)?)?;
    let hr = hr.checked_rem_euclid(24)?;
    let (yr, mo, da) = normalize_day(yr, mo, da)?;
    Some((yr, mo, da, u8::try_from(hr).ok()?, u8::try_from(mi).ok()?))
}

/// The [normalizeSecond](https://www.w3.org/TR/xmlschema11-2/#f-dt-normSe) function.
///
/// The seconds carry a fraction, so the floored quotient and remainder are
/// taken over the *integral* part and the fraction is added back afterwards:
/// with `se = whole + frac` and `0 <= frac < 1`, `whole mod 60` is at most 59,
/// so `whole mod 60 + frac` stays below 60 and the two agree with
/// `fQuotient(se, 60)` and `modulo(se, 60)` exactly. Dividing the `Decimal`
/// itself would instead round at its last representable fractional digit.
fn normalize_second(
    yr: i64,
    mo: i64,
    da: i64,
    hr: i64,
    mi: i64,
    se: Decimal,
) -> Option<(i64, u8, u8, u8, u8, Decimal)> {
    let whole = se.checked_floor()?;
    let fraction = se.checked_sub(whole)?;
    let whole = whole.as_i128();
    let mi = mi.checked_add(i64::try_from(whole.checked_div_euclid(60)?).ok()?)?;
    let se = Decimal::try_from(whole.checked_rem_euclid(60)?)
        .ok()?
        .checked_add(fraction)?;
    let (yr, mo, da, hr, mi) = normalize_minute(yr, mo, da, hr, mi)?;
    Some((yr, mo, da, hr, mi, se))
}

/// The [daysInMonth](https://www.w3.org/TR/xmlschema11-2/#f-daysInMonth) function.
///
/// `y = None` means "no year is known", which is how [`super::GMonthDay`]
/// validates its day-of-month: per [XSD 1.1 §3.3.15's constraint on the
/// gMonthDay value
/// space](https://www.w3.org/TR/xmlschema11-2/#gMonthDay), a `--MM-DD` value
/// is legal as long as `DD` occurs in `MM` in *some* year, so February is
/// given its leap-year length (29) here rather than 28 — `--02-29` is a
/// legal `xs:gMonthDay` (confirmed by the W3C XSD Test Suite,
/// `msData/datatypes/gMonthDay004.xml`, `expected validity="valid"`) even
/// though `--02-30` never occurs in any year and stays rejected.
fn days_in_month(y: Option<i64>, m: u8) -> u8 {
    match m {
        2 => {
            if let Some(y) = y {
                if y % 4 != 0 || (y % 100 == 0 && y % 400 != 0) {
                    28
                } else {
                    29
                }
            } else {
                29
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// The [dateTimePlusDuration](https://www.w3.org/TR/xmlschema11-2/#vp-dt-dateTimePlusDuration) function.
fn date_time_plus_duration(
    du: Duration,
    dt: &DateTimeSevenPropertyModel,
) -> Option<DateTimeSevenPropertyModel> {
    let yr = dt.year.unwrap_or(1);
    let mo = dt.month.unwrap_or(1);
    let da = dt.day.unwrap_or(1);
    let hr = dt.hour.unwrap_or(0);
    let mi = dt.minute.unwrap_or(0);
    let se = dt.second.unwrap_or_default();
    let mo = i64::from(mo).checked_add(du.all_months())?;
    let (yr, mo) = normalize_month(yr, mo)?;
    let da = min(da, days_in_month(Some(yr), mo));
    let se = se.checked_add(du.all_seconds())?;
    let (yr, mo, da, hr, mi, se) =
        normalize_second(yr, mo.into(), da.into(), hr.into(), mi.into(), se)?;

    Some(DateTimeSevenPropertyModel {
        year: dt.year.map(|_| yr),
        month: dt.month.map(|_| mo),
        day: dt.day.map(|_| da),
        hour: dt.hour.map(|_| hr),
        minute: dt.minute.map(|_| mi),
        second: dt.second.map(|_| se),
        timezone_offset: dt.timezone_offset,
    })
}

/// The [timeOnTimeline](https://www.w3.org/TR/xmlschema11-2/#vp-dt-timeOnTimeline) function.
fn time_on_timeline(props: &DateTimeSevenPropertyModel) -> Option<Decimal> {
    let yr = props.year.map_or(1971, |y| y - 1);
    let mo = props.month.unwrap_or(12);
    let da = props
        .day
        .map_or_else(|| days_in_month(Some(yr + 1), mo) - 1, |d| d - 1);
    let hr = props.hour.unwrap_or(0);
    let mi = i128::from(props.minute.unwrap_or(0))
        - i128::from(props.timezone_offset.unwrap_or(TimezoneOffset::UTC).offset);
    let se = props.second.unwrap_or_default();

    Decimal::try_from(
        31_536_000 * i128::from(yr)
            + 86400 * i128::from(yr.div_euclid(400) - yr.div_euclid(100) + yr.div_euclid(4))
            + 86400
                * (1..mo)
                    .map(|m| i128::from(days_in_month(Some(yr + 1), m)))
                    .sum::<i128>()
            + 86400 * i128::from(da)
            + 3600 * i128::from(hr)
            + 60 * mi,
    )
    .ok()?
    .checked_add(se)
}

/// Runs a lexical-representation parser and ensures the whole input was consumed.
fn ensure_complete<T>(
    input: &str,
    parse: impl FnOnce(&str) -> Result<(T, &str), ParseDateTimeError>,
) -> Result<T, ParseDateTimeError> {
    let (result, left) = parse(input)?;
    if !left.is_empty() {
        return Err(ParseDateTimeError::msg("Unrecognized value suffix"));
    }
    Ok(result)
}

/// A parsing error for the date/time datatypes.
#[derive(Debug, Clone)]
pub struct ParseDateTimeError(ParseDateTimeErrorKind);

#[derive(Debug, Clone)]
enum ParseDateTimeErrorKind {
    InvalidDayOfMonth { day: u8, month: u8 },
    Overflow(DateTimeOverflowError),
    InvalidTimezone(InvalidTimezoneError),
    Message(&'static str),
}

impl ParseDateTimeError {
    const fn msg(message: &'static str) -> Self {
        Self(ParseDateTimeErrorKind::Message(message))
    }
}

impl fmt::Display for ParseDateTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ParseDateTimeErrorKind::InvalidDayOfMonth { day, month } => {
                write!(f, "{day} is not a valid day of {month}")
            }
            ParseDateTimeErrorKind::Overflow(error) => error.fmt(f),
            ParseDateTimeErrorKind::InvalidTimezone(error) => error.fmt(f),
            ParseDateTimeErrorKind::Message(message) => f.write_str(message),
        }
    }
}

impl Error for ParseDateTimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.0 {
            ParseDateTimeErrorKind::Overflow(error) => Some(error),
            ParseDateTimeErrorKind::InvalidTimezone(error) => Some(error),
            ParseDateTimeErrorKind::InvalidDayOfMonth { .. }
            | ParseDateTimeErrorKind::Message(_) => None,
        }
    }
}

impl ParseDateTimeError {
    /// Returns `true` if this error is an **overflow** of the date/time value
    /// space — for example a syntactically well-formed year (production
    /// `yearFrag` places no limit on the digit count) that this implementation
    /// does not support, because it is outside [`MIN_YEAR`]`..=`[`MAX_YEAR`]
    /// or does not even fit an [`i64`] — as opposed to a **lexical** (grammar)
    /// error such as a missing separator or an out-of-range month.
    ///
    /// This lets callers distinguish the two failure modes additively,
    /// without matching on the (private) error internals: an XPath
    /// implementation should raise
    /// [`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)
    /// for an overflow, rather than the generic invalid-lexical-value error
    /// ([`FORG0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFORG0001))
    /// that a plain lexical mismatch deserves.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xsd::DateTime;
    /// use std::str::FromStr;
    ///
    /// // A year with far more digits than `i64` can hold: overflow, not garbage.
    /// let huge_year = DateTime::from_str("999999999999999999999-01-01T00:00:00Z")
    ///     .unwrap_err();
    /// assert!(huge_year.is_overflow());
    ///
    /// // A year that fits `i64` but is past `MAX_YEAR` is still an overflow,
    /// // not a lexical error.
    /// let unsupported = DateTime::from_str("9999999999999-01-01T00:00:00Z").unwrap_err();
    /// assert!(unsupported.is_overflow());
    ///
    /// // A genuinely malformed lexical form is not an overflow.
    /// let garbage = DateTime::from_str("not-a-date").unwrap_err();
    /// assert!(!garbage.is_overflow());
    /// let bad_month = DateTime::from_str("2020-13-01T00:00:00Z").unwrap_err();
    /// assert!(!bad_month.is_overflow());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_overflow(&self) -> bool {
        matches!(self.0, ParseDateTimeErrorKind::Overflow(_))
    }
}

/// An overflow during [`DateTime`]-related operations.
///
/// Matches XPath [`FODT0001` error](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001).
#[derive(Debug, Clone, Copy)]
pub struct DateTimeOverflowError;

impl fmt::Display for DateTimeOverflowError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("overflow during xsd:dateTime computation")
    }
}

impl Error for DateTimeOverflowError {}

impl From<DateTimeOverflowError> for ParseDateTimeError {
    #[inline]
    fn from(error: DateTimeOverflowError) -> Self {
        Self(ParseDateTimeErrorKind::Overflow(error))
    }
}

/// The value provided as timezone is not valid.
///
/// Matches XPath [`FODT0003` error](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0003).
#[derive(Debug, Clone, Copy)]
pub struct InvalidTimezoneError {
    offset_in_minutes: i64,
}

impl fmt::Display for InvalidTimezoneError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid timezone offset {}:{}",
            self.offset_in_minutes / 60,
            self.offset_in_minutes.abs() % 60
        )
    }
}

impl Error for InvalidTimezoneError {}
