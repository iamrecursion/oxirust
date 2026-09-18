//! The XML Schema gregorian datatypes: `gYearMonth`, `gYear`, `gMonthDay`,
//! `gMonth` and `gDay`.

use super::parser::{
    g_day_lexical_rep, g_month_day_lexical_rep, g_month_lexical_rep, g_year_lexical_rep,
    g_year_month_lexical_rep,
};
use super::{
    DateTimeOverflowError, DateTimeSevenPropertyModel, ParseDateTimeError, Timestamp,
    TimezoneOffset, ensure_complete, implicit_timezone_comparisons,
};
use crate::{Date, DateTime, DayTimeDuration, Decimal};
use std::fmt;
use std::str::FromStr;

/// [XML Schema `gYearMonth` datatype](https://www.w3.org/TR/xmlschema11-2/#gYearMonth)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`], when combined with the day-time 31T00:00:00, and an
/// optional timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct GYearMonth {
    timestamp: Timestamp,
}

impl GYearMonth {
    /// The largest `gYearMonth` the internal timeline encoding can hold.
    ///
    /// Its year lies past [`MAX_YEAR`](crate::MAX_YEAR), so it is a saturation sentinel
    /// rather than a value any constructor or lexical form can produce.
    pub const MAX: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(170_141_183_460_469_216_800),
            timezone_offset: Some(TimezoneOffset::MAX),
        },
    };
    /// The smallest `gYearMonth` the internal timeline encoding can hold; see [`Self::MAX`].
    pub const MIN: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(-170_141_183_460_466_970_400),
            timezone_offset: Some(TimezoneOffset::MIN),
        },
    };

    #[inline]
    pub(crate) fn new(
        year: i64,
        month: u8,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: Some(year),
                month: Some(month),
                day: None,
                hour: None,
                minute: None,
                second: None,
                timezone_offset,
            })?,
        })
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 18]) -> Self {
        Self {
            timestamp: Timestamp::from_be_bytes(bytes),
        }
    }

    /// The gregorian year.
    #[inline]
    #[must_use]
    pub fn year(self) -> i64 {
        self.timestamp.year()
    }

    /// The gregorian month, between 1 and 12.
    #[inline]
    #[must_use]
    pub fn month(self) -> u8 {
        self.timestamp.month()
    }

    /// The timezone offset expressed as a [`DayTimeDuration`], if any.
    #[inline]
    #[must_use]
    pub fn timezone(self) -> Option<DayTimeDuration> {
        Some(self.timezone_offset()?.into())
    }

    /// The timezone offset of this value, if any.
    #[inline]
    #[must_use]
    pub fn timezone_offset(self) -> Option<TimezoneOffset> {
        self.timestamp.timezone_offset()
    }

    /// [fn:adjust-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-dateTime-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(Self {
            timestamp: self.timestamp.adjust(timezone_offset)?,
        })
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl TryFrom<DateTime> for GYearMonth {
    type Error = DateTimeOverflowError;

    #[inline]
    fn try_from(date_time: DateTime) -> Result<Self, Self::Error> {
        Self::new(
            date_time.year(),
            date_time.month(),
            date_time.timezone_offset(),
        )
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
///
/// `gYearMonth` always encodes its month using that month's *last* calendar day (see the
/// type-level doc), so casting a `date` whose own day is earlier in the month moves the
/// encoded instant forward within it. That re-derivation can leave the representable range —
/// most plainly for the [`Date::MAX`]/[`Date::MIN`] sentinels, whose years lie past
/// [`MAX_YEAR`](crate::MAX_YEAR). Rather than making this cast fallible (unlike this file's
/// other `date`/`dateTime` casts, it mirrors an upstream API that returns `Self` directly), it
/// saturates towards the end of the range the source lies at, keeping `date`'s exact timezone:
/// the result still reports the correct [`year`](Self::year) and [`month`](Self::month).
impl From<Date> for GYearMonth {
    #[inline]
    fn from(date: Date) -> Self {
        Self::new(date.year(), date.month(), date.timezone_offset()).unwrap_or(Self {
            timestamp: Timestamp {
                value: if date.year() < 0 {
                    Self::MIN.timestamp.value
                } else {
                    Self::MAX.timestamp.value
                },
                timezone_offset: date.timezone_offset(),
            },
        })
    }
}

impl FromStr for GYearMonth {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, g_year_month_lexical_rep)
    }
}

impl fmt::Display for GYearMonth {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let year = self.year();
        if year < 0 {
            f.write_str("-")?;
        }
        write!(f, "{:04}-{:02}", year.abs(), self.month())?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

/// [XML Schema `gYear` datatype](https://www.w3.org/TR/xmlschema11-2/#gYear)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`], when combined with the month-day-time 12-31T00:00:00, and
/// an optional timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct GYear {
    timestamp: Timestamp,
}

impl GYear {
    /// The largest `gYear` the internal timeline encoding can hold.
    ///
    /// Its year lies past [`MAX_YEAR`](crate::MAX_YEAR), so it is a saturation sentinel
    /// rather than a value any constructor or lexical form can produce.
    pub const MAX: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(170_141_183_460_461_440_800),
            timezone_offset: Some(TimezoneOffset::MAX),
        },
    };
    /// The smallest `gYear` the internal timeline encoding can hold; see [`Self::MAX`].
    pub const MIN: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(-170_141_183_460_461_700_000),
            timezone_offset: Some(TimezoneOffset::MIN),
        },
    };

    #[inline]
    pub(crate) fn new(
        year: i64,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: Some(year),
                month: None,
                day: None,
                hour: None,
                minute: None,
                second: None,
                timezone_offset,
            })?,
        })
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 18]) -> Self {
        Self {
            timestamp: Timestamp::from_be_bytes(bytes),
        }
    }

    /// The gregorian year.
    #[inline]
    #[must_use]
    pub fn year(self) -> i64 {
        self.timestamp.year()
    }

    /// The timezone offset expressed as a [`DayTimeDuration`], if any.
    #[inline]
    #[must_use]
    pub fn timezone(self) -> Option<DayTimeDuration> {
        Some(self.timezone_offset()?.into())
    }

    /// The timezone offset of this value, if any.
    #[inline]
    #[must_use]
    pub fn timezone_offset(self) -> Option<TimezoneOffset> {
        self.timestamp.timezone_offset()
    }

    /// [fn:adjust-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-dateTime-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(Self {
            timestamp: self.timestamp.adjust(timezone_offset)?,
        })
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl TryFrom<DateTime> for GYear {
    type Error = DateTimeOverflowError;

    #[inline]
    fn try_from(date_time: DateTime) -> Result<Self, Self::Error> {
        Self::new(date_time.year(), date_time.timezone_offset())
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl TryFrom<Date> for GYear {
    type Error = DateTimeOverflowError;

    #[inline]
    fn try_from(date: Date) -> Result<Self, Self::Error> {
        Self::new(date.year(), date.timezone_offset())
    }
}

impl TryFrom<GYearMonth> for GYear {
    type Error = DateTimeOverflowError;

    #[inline]
    fn try_from(year_month: GYearMonth) -> Result<Self, Self::Error> {
        Self::new(year_month.year(), year_month.timezone_offset())
    }
}

impl FromStr for GYear {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, g_year_lexical_rep)
    }
}

impl fmt::Display for GYear {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let year = self.year();
        if year < 0 {
            f.write_str("-")?;
        }
        write!(f, "{:04}", year.abs())?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

/// [XML Schema `gMonthDay` datatype](https://www.w3.org/TR/xmlschema11-2/#gMonthDay)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`], when combined with the year 1972 and the time
/// 31T00:00:00, and an optional timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct GMonthDay {
    timestamp: Timestamp,
}

impl GMonthDay {
    #[inline]
    pub(crate) fn new(
        month: u8,
        day: u8,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: None,
                month: Some(month),
                day: Some(day),
                hour: None,
                minute: None,
                second: None,
                timezone_offset,
            })?,
        })
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 18]) -> Self {
        Self {
            timestamp: Timestamp::from_be_bytes(bytes),
        }
    }

    /// The gregorian month, between 1 and 12.
    #[inline]
    #[must_use]
    pub fn month(&self) -> u8 {
        self.timestamp.month()
    }

    /// The gregorian day of month, between 1 and 31.
    #[inline]
    #[must_use]
    pub fn day(&self) -> u8 {
        self.timestamp.day()
    }

    /// The timezone offset expressed as a [`DayTimeDuration`], if any.
    #[inline]
    #[must_use]
    pub fn timezone(&self) -> Option<DayTimeDuration> {
        Some(self.timezone_offset()?.into())
    }

    /// The timezone offset of this value, if any.
    #[inline]
    #[must_use]
    pub fn timezone_offset(&self) -> Option<TimezoneOffset> {
        self.timestamp.timezone_offset()
    }

    /// [fn:adjust-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-dateTime-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(&self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(Self {
            timestamp: self.timestamp.adjust(timezone_offset)?,
        })
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl From<DateTime> for GMonthDay {
    #[inline]
    fn from(date_time: DateTime) -> Self {
        Self::new(
            date_time.month(),
            date_time.day(),
            date_time.timezone_offset(),
        )
        .expect("Casting from xsd:dateTime to xsd:gMonthDay can't fail")
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl From<Date> for GMonthDay {
    #[inline]
    fn from(date: Date) -> Self {
        Self::new(date.month(), date.day(), date.timezone_offset())
            .expect("Casting from xsd:date to xsd:gMonthDay can't fail")
    }
}

impl FromStr for GMonthDay {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, g_month_day_lexical_rep)
    }
}

impl fmt::Display for GMonthDay {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "--{:02}-{:02}", self.month(), self.day())?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

/// [XML Schema `gMonth` datatype](https://www.w3.org/TR/xmlschema11-2/#gMonth)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`], when combined with the year 1972 and the day-time
/// 31T00:00:00, and an optional timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct GMonth {
    timestamp: Timestamp,
}

impl GMonth {
    #[inline]
    pub(crate) fn new(
        month: u8,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: None,
                month: Some(month),
                day: None,
                hour: None,
                minute: None,
                second: None,
                timezone_offset,
            })?,
        })
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 18]) -> Self {
        Self {
            timestamp: Timestamp::from_be_bytes(bytes),
        }
    }

    /// The gregorian month, between 1 and 12.
    #[inline]
    #[must_use]
    pub fn month(&self) -> u8 {
        self.timestamp.month()
    }

    /// The timezone offset expressed as a [`DayTimeDuration`], if any.
    #[inline]
    #[must_use]
    pub fn timezone(&self) -> Option<DayTimeDuration> {
        Some(self.timezone_offset()?.into())
    }

    /// The timezone offset of this value, if any.
    #[inline]
    #[must_use]
    pub fn timezone_offset(&self) -> Option<TimezoneOffset> {
        self.timestamp.timezone_offset()
    }

    /// [fn:adjust-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-dateTime-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(&self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(Self {
            timestamp: self.timestamp.adjust(timezone_offset)?,
        })
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl From<DateTime> for GMonth {
    #[inline]
    fn from(date_time: DateTime) -> Self {
        Self::new(date_time.month(), date_time.timezone_offset())
            .expect("Casting from xsd:dateTime to xsd:gMonth can't fail")
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl From<Date> for GMonth {
    #[inline]
    fn from(date: Date) -> Self {
        Self::new(date.month(), date.timezone_offset())
            .expect("Casting from xsd:date to xsd:gMonth can't fail")
    }
}

impl From<GYearMonth> for GMonth {
    #[inline]
    fn from(year_month: GYearMonth) -> Self {
        Self::new(year_month.month(), year_month.timezone_offset())
            .expect("Casting from xsd:gYearMonth to xsd:gMonth can't fail")
    }
}

impl From<GMonthDay> for GMonth {
    #[inline]
    fn from(month_day: GMonthDay) -> Self {
        Self::new(month_day.month(), month_day.timezone_offset())
            .expect("Casting from xsd:gMonthDay to xsd:gMonth can't fail")
    }
}

impl FromStr for GMonth {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, g_month_lexical_rep)
    }
}

impl fmt::Display for GMonth {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "--{:02}", self.month())?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

/// [XML Schema `gDay` datatype](https://www.w3.org/TR/xmlschema11-2/#gDay)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`], when combined with the year-month 1972-12 and the time
/// 00:00:00, and an optional timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct GDay {
    timestamp: Timestamp,
}

impl GDay {
    #[inline]
    pub(crate) fn new(
        day: u8,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: None,
                month: None,
                day: Some(day),
                hour: None,
                minute: None,
                second: None,
                timezone_offset,
            })?,
        })
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 18]) -> Self {
        Self {
            timestamp: Timestamp::from_be_bytes(bytes),
        }
    }

    /// The gregorian day of month, between 1 and 31.
    #[inline]
    #[must_use]
    pub fn day(&self) -> u8 {
        self.timestamp.day()
    }

    /// The timezone offset expressed as a [`DayTimeDuration`], if any.
    #[inline]
    #[must_use]
    pub fn timezone(&self) -> Option<DayTimeDuration> {
        Some(self.timezone_offset()?.into())
    }

    /// The timezone offset of this value, if any.
    #[inline]
    #[must_use]
    pub fn timezone_offset(&self) -> Option<TimezoneOffset> {
        self.timestamp.timezone_offset()
    }

    /// [fn:adjust-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-dateTime-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(&self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(Self {
            timestamp: self.timestamp.adjust(timezone_offset)?,
        })
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl From<DateTime> for GDay {
    #[inline]
    fn from(date_time: DateTime) -> Self {
        Self::new(date_time.day(), date_time.timezone_offset())
            .expect("Casting from xsd:dateTime to xsd:gDay can't fail")
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl From<Date> for GDay {
    #[inline]
    fn from(date: Date) -> Self {
        Self::new(date.day(), date.timezone_offset())
            .expect("Casting from xsd:date to xsd:gDay can't fail")
    }
}

impl From<GMonthDay> for GDay {
    #[inline]
    fn from(month_day: GMonthDay) -> Self {
        Self::new(month_day.day(), month_day.timezone_offset())
            .expect("Casting from xsd:gMonthDay to xsd:gDay can't fail")
    }
}

impl FromStr for GDay {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, g_day_lexical_rep)
    }
}

impl fmt::Display for GDay {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "---{:02}", self.day())?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

implicit_timezone_comparisons!(GYearMonth, "gYearMonth");
implicit_timezone_comparisons!(GYear, "gYear");
implicit_timezone_comparisons!(GMonthDay, "gMonthDay");
implicit_timezone_comparisons!(GMonth, "gMonth");
implicit_timezone_comparisons!(GDay, "gDay");
