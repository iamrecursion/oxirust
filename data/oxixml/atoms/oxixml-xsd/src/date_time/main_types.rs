//! The XML Schema `dateTime`, `time` and `date` datatypes.

use super::parser::{date_lexical_rep, date_time_lexical_rep, time_lexical_rep};
use super::{
    DateTimeOverflowError, DateTimeSevenPropertyModel, NO_CLOCK, ParseDateTimeError, Timestamp,
    TimezoneOffset, date_time_plus_duration, ensure_complete, implicit_timezone_comparisons,
    implicit_timezone_subtraction,
};
use crate::{DayTimeDuration, Decimal, Duration, YearMonthDuration};
use std::fmt;
use std::str::FromStr;

/// [XML Schema `dateTime` datatype](https://www.w3.org/TR/xmlschema11-2/#dateTime)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`] and an optional timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct DateTime {
    timestamp: Timestamp,
}

impl DateTime {
    /// The largest `dateTime` the internal timeline encoding can hold.
    ///
    /// Its year lies past [`MAX_YEAR`](crate::MAX_YEAR), so it is a saturation sentinel
    /// rather than a value any constructor or lexical form can produce.
    pub const MAX: Self = Self {
        timestamp: Timestamp::MAX,
    };
    /// The smallest `dateTime` the internal timeline encoding can hold; see [`Self::MAX`].
    pub const MIN: Self = Self {
        timestamp: Timestamp::MIN,
    };

    #[inline]
    pub(crate) fn new(
        year: i64,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: Decimal,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: Some(year),
                month: Some(month),
                day: Some(day),
                hour: Some(hour),
                minute: Some(minute),
                second: Some(second),
                timezone_offset,
            })?,
        })
    }

    /// [fn:current-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-current-dateTime)
    ///
    /// # Panics
    ///
    /// Panics when no clock is available on the target and no provider was registered with
    /// [`set_now_provider`](crate::set_now_provider) — see [`Self::checked_now`].
    #[inline]
    #[must_use]
    pub fn now() -> Self {
        Self::checked_now().expect(NO_CLOCK)
    }

    /// [fn:current-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-current-dateTime),
    /// returning [`None`] instead of panicking when the target has no clock.
    ///
    /// `wasm32-unknown-unknown` has no host clock at all, so
    /// [`std::time::SystemTime::now`] traps there; use
    /// [`set_now_provider`](crate::set_now_provider) to supply one.
    ///
    /// A [`None`] here is ambiguous between "no clock is available" and "the clock reading
    /// overflows the internal representation" -- the latter is only reachable through a
    /// [`NowProvider`](crate::NowProvider) reporting a time far outside this representation's
    /// multi-trillion-year range, but a misbehaving one gives no signal to tell the two apart.
    #[inline]
    #[must_use]
    pub fn checked_now() -> Option<Self> {
        Some(Self {
            timestamp: Timestamp::checked_now()?,
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

    /// [fn:year-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-year-from-dateTime)
    #[inline]
    #[must_use]
    pub fn year(self) -> i64 {
        self.timestamp.year()
    }

    /// [fn:month-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-month-from-dateTime)
    #[inline]
    #[must_use]
    pub fn month(self) -> u8 {
        self.timestamp.month()
    }

    /// [fn:day-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-day-from-dateTime)
    #[inline]
    #[must_use]
    pub fn day(self) -> u8 {
        self.timestamp.day()
    }

    /// [fn:hour-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-hours-from-dateTime)
    #[inline]
    #[must_use]
    pub fn hour(self) -> u8 {
        self.timestamp.hour()
    }

    /// [fn:minute-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-minutes-from-dateTime)
    #[inline]
    #[must_use]
    pub fn minute(self) -> u8 {
        self.timestamp.minute()
    }

    /// [fn:second-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-seconds-from-dateTime)
    #[inline]
    #[must_use]
    pub fn second(self) -> Decimal {
        self.timestamp.second()
    }

    /// [fn:timezone-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-timezone-from-dateTime)
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

    #[inline]
    fn properties(self) -> DateTimeSevenPropertyModel {
        DateTimeSevenPropertyModel {
            year: Some(self.year()),
            month: Some(self.month()),
            day: Some(self.day()),
            hour: Some(self.hour()),
            minute: Some(self.minute()),
            second: Some(self.second()),
            timezone_offset: self.timezone_offset(),
        }
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// [op:subtract-dateTimes](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dateTimes)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<DayTimeDuration> {
        self.timestamp.checked_sub(rhs.into().timestamp)
    }

    /// [op:add-yearMonthDuration-to-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-add-yearMonthDuration-to-dateTime)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_year_month_duration(
        self,
        rhs: impl Into<YearMonthDuration>,
    ) -> Option<Self> {
        self.checked_add_duration(Duration::from(rhs.into()))
    }

    /// [op:add-dayTimeDuration-to-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDuration-to-dateTime)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_day_time_duration(self, rhs: impl Into<Duration>) -> Option<Self> {
        let rhs = rhs.into();
        Some(Self {
            timestamp: self.timestamp.checked_add_seconds(rhs.all_seconds())?,
        })
    }

    /// [op:add-yearMonthDuration-to-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-add-yearMonthDuration-to-dateTime) and [op:add-dayTimeDuration-to-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDuration-to-dateTime)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_duration(self, rhs: impl Into<Duration>) -> Option<Self> {
        let rhs = rhs.into();
        if let Ok(rhs) = DayTimeDuration::try_from(rhs) {
            self.checked_add_day_time_duration(rhs)
        } else {
            Some(Self {
                timestamp: Timestamp::new(&date_time_plus_duration(rhs, &self.properties())?)
                    .ok()?,
            })
        }
    }

    /// [op:subtract-yearMonthDuration-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-subtract-yearMonthDuration-from-dateTime)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_year_month_duration(
        self,
        rhs: impl Into<YearMonthDuration>,
    ) -> Option<Self> {
        self.checked_sub_duration(Duration::from(rhs.into()))
    }

    /// [op:subtract-dayTimeDuration-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDuration-from-dateTime)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_day_time_duration(self, rhs: impl Into<DayTimeDuration>) -> Option<Self> {
        let rhs = rhs.into();
        Some(Self {
            timestamp: self.timestamp.checked_sub_seconds(rhs.as_seconds())?,
        })
    }

    /// [op:subtract-yearMonthDuration-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-subtract-yearMonthDuration-from-dateTime) and [op:subtract-dayTimeDuration-from-dateTime](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDuration-from-dateTime)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_duration(self, rhs: impl Into<Duration>) -> Option<Self> {
        let rhs = rhs.into();
        if let Ok(rhs) = DayTimeDuration::try_from(rhs) {
            self.checked_sub_day_time_duration(rhs)
        } else {
            Some(Self {
                timestamp: Timestamp::new(&date_time_plus_duration(
                    rhs.checked_neg()?,
                    &self.properties(),
                )?)
                .ok()?,
            })
        }
    }

    /// [fn:adjust-dateTime-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-dateTime-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(Self {
            timestamp: self.timestamp.adjust(timezone_offset)?,
        })
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    ///
    /// Unlike [`PartialEq`], this distinguishes values with different timezone
    /// offsets even when they denote the same instant.
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl TryFrom<Date> for DateTime {
    type Error = DateTimeOverflowError;

    #[inline]
    fn try_from(date: Date) -> Result<Self, Self::Error> {
        Self::new(
            date.year(),
            date.month(),
            date.day(),
            0,
            0,
            Decimal::default(),
            date.timezone_offset(),
        )
    }
}

impl FromStr for DateTime {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, date_time_lexical_rep)
    }
}

impl fmt::Display for DateTime {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let year = self.year();
        if year < 0 {
            f.write_str("-")?;
        }
        let second = self.second();
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{}{}",
            year.abs(),
            self.month(),
            self.day(),
            self.hour(),
            self.minute(),
            if Decimal::from(-10) < second && second < Decimal::from(10) {
                "0"
            } else {
                ""
            },
            second
        )?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

/// [XML Schema `time` datatype](https://www.w3.org/TR/xmlschema11-2/#time)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`], when combined with the date 1972-12-31, and an optional
/// timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct Time {
    timestamp: Timestamp,
}

impl Time {
    #[cfg(test)]
    pub(crate) const MAX: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(62_230_255_200),
            timezone_offset: Some(TimezoneOffset::MIN),
        },
    };
    #[cfg(test)]
    pub(crate) const MIN: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(62_230_154_400),
            timezone_offset: Some(TimezoneOffset::MAX),
        },
    };

    #[inline]
    pub(crate) fn new(
        mut hour: u8,
        minute: u8,
        second: Decimal,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        if hour == 24 && minute == 0 && second == Decimal::default() {
            hour = 0;
        }
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: None,
                month: None,
                day: None,
                hour: Some(hour),
                minute: Some(minute),
                second: Some(second),
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

    /// [fn:current-time](https://www.w3.org/TR/xpath-functions-31/#func-current-time)
    ///
    /// # Panics
    ///
    /// Panics when no clock is available on the target and no provider was registered with
    /// [`set_now_provider`](crate::set_now_provider) — see [`Self::checked_now`].
    #[inline]
    #[must_use]
    pub fn now() -> Self {
        Self::checked_now().expect(NO_CLOCK)
    }

    /// [fn:current-time](https://www.w3.org/TR/xpath-functions-31/#func-current-time),
    /// returning [`None`] instead of panicking when the target has no clock.
    ///
    /// A [`None`] here is ambiguous between "no clock is available" and "the clock reading
    /// overflows the internal representation" -- the latter is only reachable through a
    /// [`NowProvider`](crate::NowProvider) reporting a time far outside this representation's
    /// multi-trillion-year range, but a misbehaving one gives no signal to tell the two apart.
    #[inline]
    #[must_use]
    pub fn checked_now() -> Option<Self> {
        Some(Self {
            timestamp: Timestamp::checked_now()?,
        })
    }

    /// [fn:hour-from-time](https://www.w3.org/TR/xpath-functions-31/#func-hours-from-time)
    #[inline]
    #[must_use]
    pub fn hour(self) -> u8 {
        self.timestamp.hour()
    }

    /// [fn:minute-from-time](https://www.w3.org/TR/xpath-functions-31/#func-minutes-from-time)
    #[inline]
    #[must_use]
    pub fn minute(self) -> u8 {
        self.timestamp.minute()
    }

    /// [fn:second-from-time](https://www.w3.org/TR/xpath-functions-31/#func-seconds-from-time)
    #[inline]
    #[must_use]
    pub fn second(self) -> Decimal {
        self.timestamp.second()
    }

    /// [fn:timezone-from-time](https://www.w3.org/TR/xpath-functions-31/#func-timezone-from-time)
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

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// [op:subtract-times](https://www.w3.org/TR/xpath-functions-31/#func-subtract-times)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<DayTimeDuration> {
        self.timestamp.checked_sub(rhs.into().timestamp)
    }

    /// [op:add-dayTimeDuration-to-time](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDuration-to-time)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_day_time_duration(self, rhs: impl Into<DayTimeDuration>) -> Option<Self> {
        self.checked_add_duration(Duration::from(rhs.into()))
    }

    /// [op:add-dayTimeDuration-to-time](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDuration-to-time)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_duration(self, rhs: impl Into<Duration>) -> Option<Self> {
        Some(
            DateTime::new(
                1972,
                12,
                31,
                self.hour(),
                self.minute(),
                self.second(),
                self.timezone_offset(),
            )
            .ok()?
            .checked_add_duration(rhs)?
            .into(),
        )
    }

    /// [op:subtract-dayTimeDuration-from-time](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDuration-from-time)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_day_time_duration(self, rhs: impl Into<DayTimeDuration>) -> Option<Self> {
        self.checked_sub_duration(Duration::from(rhs.into()))
    }

    /// [op:subtract-dayTimeDuration-from-time](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDuration-from-time)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_duration(self, rhs: impl Into<Duration>) -> Option<Self> {
        Some(
            DateTime::new(
                1972,
                12,
                31,
                self.hour(),
                self.minute(),
                self.second(),
                self.timezone_offset(),
            )
            .ok()?
            .checked_sub_duration(rhs)?
            .into(),
        )
    }

    /// [fn:adjust-time-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-time-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        Some(
            DateTime::new(
                1972,
                12,
                31,
                self.hour(),
                self.minute(),
                self.second(),
                self.timezone_offset(),
            )
            .ok()?
            .adjust(timezone_offset)?
            .into(),
        )
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl From<DateTime> for Time {
    #[inline]
    fn from(date_time: DateTime) -> Self {
        Self::new(
            date_time.hour(),
            date_time.minute(),
            date_time.second(),
            date_time.timezone_offset(),
        )
        .expect("Casting from xsd:dateTime to xsd:time can't fail")
    }
}

impl FromStr for Time {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, time_lexical_rep)
    }
}

impl fmt::Display for Time {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let second = self.second();
        write!(
            f,
            "{:02}:{:02}:{}{}",
            self.hour(),
            self.minute(),
            if Decimal::from(-10) < second && second < Decimal::from(10) {
                "0"
            } else {
                ""
            },
            second
        )?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

/// [XML Schema `date` datatype](https://www.w3.org/TR/xmlschema11-2/#date)
///
/// It encodes the value using a number of seconds from the Gregorian calendar era
/// using a [`Decimal`], when combined with the time 00:00:00, and an optional
/// timezone offset in minutes.
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct Date {
    timestamp: Timestamp,
}

impl Date {
    /// The largest `date` the internal timeline encoding can hold.
    ///
    /// Its year lies past [`MAX_YEAR`](crate::MAX_YEAR), so it is a saturation sentinel
    /// rather than a value any constructor or lexical form can produce.
    pub const MAX: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(170_141_183_460_469_216_800),
            timezone_offset: Some(TimezoneOffset::MAX),
        },
    };
    /// The smallest `date` the internal timeline encoding can hold; see [`Self::MAX`].
    pub const MIN: Self = Self {
        timestamp: Timestamp {
            value: Decimal::new_from_i128_unchecked(-170_141_183_460_469_216_800),
            timezone_offset: Some(TimezoneOffset::MIN),
        },
    };

    #[inline]
    pub(crate) fn new(
        year: i64,
        month: u8,
        day: u8,
        timezone_offset: Option<TimezoneOffset>,
    ) -> Result<Self, DateTimeOverflowError> {
        Ok(Self {
            timestamp: Timestamp::new(&DateTimeSevenPropertyModel {
                year: Some(year),
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

    /// [fn:current-date](https://www.w3.org/TR/xpath-functions-31/#func-current-date)
    ///
    /// # Panics
    ///
    /// Panics when no clock is available on the target and no provider was registered with
    /// [`set_now_provider`](crate::set_now_provider) — see [`Self::checked_now`].
    #[inline]
    #[must_use]
    pub fn now() -> Self {
        Self::checked_now().expect(NO_CLOCK)
    }

    /// [fn:current-date](https://www.w3.org/TR/xpath-functions-31/#func-current-date),
    /// returning [`None`] instead of panicking when the target has no clock.
    ///
    /// A [`None`] here has the same ambiguity as [`DateTime::checked_now`]'s: "no clock is
    /// available" versus "the clock reading overflows the internal representation". The final
    /// `.ok()` also discards any [`Date`]-specific conversion failure -- dropping a
    /// [`DateTime`]'s time-of-day to derive a [`Date`] can, in principle, fail near either end
    /// of the representable range -- but that is not reachable through this constructor today:
    /// [`set_now_provider`](crate::set_now_provider) takes an unsigned duration *since* the
    /// epoch, so every reading it can produce is already at or after 1970, nowhere near the
    /// multi-trillion-year-distant boundary such a failure would require.
    #[inline]
    #[must_use]
    pub fn checked_now() -> Option<Self> {
        DateTime::checked_now()?.try_into().ok()
    }

    /// [fn:year-from-date](https://www.w3.org/TR/xpath-functions-31/#func-year-from-date)
    #[inline]
    #[must_use]
    pub fn year(self) -> i64 {
        self.timestamp.year()
    }

    /// [fn:month-from-date](https://www.w3.org/TR/xpath-functions-31/#func-month-from-date)
    #[inline]
    #[must_use]
    pub fn month(self) -> u8 {
        self.timestamp.month()
    }

    /// [fn:day-from-date](https://www.w3.org/TR/xpath-functions-31/#func-day-from-date)
    #[inline]
    #[must_use]
    pub fn day(self) -> u8 {
        self.timestamp.day()
    }

    /// [fn:timezone-from-date](https://www.w3.org/TR/xpath-functions-31/#func-timezone-from-date)
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

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 18] {
        self.timestamp.to_be_bytes()
    }

    /// [op:subtract-dates](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dates)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<DayTimeDuration> {
        self.timestamp.checked_sub(rhs.into().timestamp)
    }

    /// [op:add-yearMonthDuration-to-date](https://www.w3.org/TR/xpath-functions-31/#func-add-yearMonthDuration-to-date)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_year_month_duration(
        self,
        rhs: impl Into<YearMonthDuration>,
    ) -> Option<Self> {
        self.checked_add_duration(Duration::from(rhs.into()))
    }

    /// [op:add-dayTimeDuration-to-date](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDuration-to-date)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_day_time_duration(self, rhs: impl Into<DayTimeDuration>) -> Option<Self> {
        self.checked_add_duration(Duration::from(rhs.into()))
    }

    /// [op:add-yearMonthDuration-to-date](https://www.w3.org/TR/xpath-functions-31/#func-add-yearMonthDuration-to-date) and [op:add-dayTimeDuration-to-date](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDuration-to-date)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_add_duration(self, rhs: impl Into<Duration>) -> Option<Self> {
        DateTime::try_from(self)
            .ok()?
            .checked_add_duration(rhs)?
            .try_into()
            .ok()
    }

    /// [op:subtract-yearMonthDuration-from-date](https://www.w3.org/TR/xpath-functions-31/#func-subtract-yearMonthDuration-from-date)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_year_month_duration(
        self,
        rhs: impl Into<YearMonthDuration>,
    ) -> Option<Self> {
        self.checked_sub_duration(Duration::from(rhs.into()))
    }

    /// [op:subtract-dayTimeDuration-from-date](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDuration-from-date)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_day_time_duration(self, rhs: impl Into<DayTimeDuration>) -> Option<Self> {
        self.checked_sub_duration(Duration::from(rhs.into()))
    }

    /// [op:subtract-yearMonthDuration-from-date](https://www.w3.org/TR/xpath-functions-31/#func-subtract-yearMonthDuration-from-date) and [op:subtract-dayTimeDuration-from-date](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDuration-from-date)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn checked_sub_duration(self, rhs: impl Into<Duration>) -> Option<Self> {
        DateTime::try_from(self)
            .ok()?
            .checked_sub_duration(rhs)?
            .try_into()
            .ok()
    }

    /// [fn:adjust-date-to-timezone](https://www.w3.org/TR/xpath-functions-31/#func-adjust-date-to-timezone)
    ///
    /// Returns `None` in case of overflow ([`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001)).
    #[inline]
    #[must_use]
    pub fn adjust(self, timezone_offset: Option<TimezoneOffset>) -> Option<Self> {
        DateTime::new(
            self.year(),
            self.month(),
            self.day(),
            0,
            0,
            Decimal::default(),
            self.timezone_offset(),
        )
        .ok()?
        .adjust(timezone_offset)?
        .try_into()
        .ok()
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self.timestamp.is_identical_with(other.timestamp)
    }
}

/// Conversion according to [XPath cast rules](https://www.w3.org/TR/xpath-functions-31/#casting-to-datetimes).
impl TryFrom<DateTime> for Date {
    type Error = DateTimeOverflowError;

    #[inline]
    fn try_from(date_time: DateTime) -> Result<Self, Self::Error> {
        Self::new(
            date_time.year(),
            date_time.month(),
            date_time.day(),
            date_time.timezone_offset(),
        )
    }
}

impl FromStr for Date {
    type Err = ParseDateTimeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        ensure_complete(input, date_lexical_rep)
    }
}

impl fmt::Display for Date {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let year = self.year();
        if year < 0 {
            f.write_str("-")?;
        }
        write!(f, "{:04}-{:02}-{:02}", year.abs(), self.month(), self.day())?;
        if let Some(timezone_offset) = self.timezone_offset() {
            write!(f, "{timezone_offset}")?;
        }
        Ok(())
    }
}

implicit_timezone_comparisons!(DateTime, "dateTime");
implicit_timezone_subtraction!(DateTime, "dateTime");
implicit_timezone_comparisons!(Time, "time");
implicit_timezone_subtraction!(Time, "time");
implicit_timezone_comparisons!(Date, "date");
implicit_timezone_subtraction!(Date, "date");
