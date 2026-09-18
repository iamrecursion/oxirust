//! The XML Schema `duration`, `yearMonthDuration` and `dayTimeDuration` datatypes.

use crate::{DateTime, Decimal};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::str::FromStr;
use std::time::Duration as StdDuration;

/// [XML Schema `duration` datatype](https://www.w3.org/TR/xmlschema11-2/#duration)
///
/// It stores the duration using a pair of a [`YearMonthDuration`] and a [`DayTimeDuration`].
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash, Default)]
pub struct Duration {
    year_month: YearMonthDuration,
    day_time: DayTimeDuration,
}

impl Duration {
    /// The largest representable `duration` value.
    pub const MAX: Self = Self {
        year_month: YearMonthDuration::MAX,
        day_time: DayTimeDuration::MAX,
    };
    /// The smallest representable `duration` value.
    pub const MIN: Self = Self {
        year_month: YearMonthDuration::MIN,
        day_time: DayTimeDuration::MIN,
    };

    /// Builds a duration from a number of months and a number of seconds.
    ///
    /// Returns an error if the two components have opposite signs.
    #[inline]
    pub fn new(
        months: impl Into<i64>,
        seconds: impl Into<Decimal>,
    ) -> Result<Self, OppositeSignInDurationComponentsError> {
        Self::construct(
            YearMonthDuration::new(months),
            DayTimeDuration::new(seconds),
        )
    }

    #[inline]
    fn construct(
        year_month: YearMonthDuration,
        day_time: DayTimeDuration,
    ) -> Result<Self, OppositeSignInDurationComponentsError> {
        if (year_month > YearMonthDuration::default() && day_time < DayTimeDuration::default())
            || (year_month < YearMonthDuration::default() && day_time > DayTimeDuration::default())
        {
            return Err(OppositeSignInDurationComponentsError);
        }
        Ok(Self {
            year_month,
            day_time,
        })
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 24]) -> Self {
        Self {
            year_month: YearMonthDuration::from_be_bytes(
                bytes[0..8].try_into().expect("8 bytes for year_month"),
            ),
            day_time: DayTimeDuration::from_be_bytes(
                bytes[8..24].try_into().expect("16 bytes for day_time"),
            ),
        }
    }

    /// [fn:years-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-years-from-duration)
    #[inline]
    #[must_use]
    pub fn years(self) -> i64 {
        self.year_month.years()
    }

    /// [fn:months-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-months-from-duration)
    #[inline]
    #[must_use]
    pub fn months(self) -> i64 {
        self.year_month.months()
    }

    /// [fn:days-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-days-from-duration)
    #[inline]
    #[must_use]
    pub fn days(self) -> i64 {
        self.day_time.days()
    }

    /// [fn:hours-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-hours-from-duration)
    #[inline]
    #[must_use]
    pub fn hours(self) -> i64 {
        self.day_time.hours()
    }

    /// [fn:minutes-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-minutes-from-duration)
    #[inline]
    #[must_use]
    pub fn minutes(self) -> i64 {
        self.day_time.minutes()
    }

    /// [fn:seconds-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-seconds-from-duration)
    #[inline]
    #[must_use]
    pub fn seconds(self) -> Decimal {
        self.day_time.seconds()
    }

    #[inline]
    #[must_use]
    pub(crate) const fn all_months(self) -> i64 {
        self.year_month.all_months()
    }

    #[inline]
    #[must_use]
    pub(crate) const fn all_seconds(self) -> Decimal {
        self.day_time.as_seconds()
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        bytes[0..8].copy_from_slice(&self.year_month.to_be_bytes());
        bytes[8..24].copy_from_slice(&self.day_time.to_be_bytes());
        bytes
    }

    /// [op:add-yearMonthDurations](https://www.w3.org/TR/xpath-functions-31/#func-add-yearMonthDurations) and [op:add-dayTimeDurations](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDurations)
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        Self::construct(
            self.year_month.checked_add(rhs.year_month)?,
            self.day_time.checked_add(rhs.day_time)?,
        )
        .ok()
    }

    /// [op:subtract-yearMonthDurations](https://www.w3.org/TR/xpath-functions-31/#func-subtract-yearMonthDurations) and [op:subtract-dayTimeDurations](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDurations)
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        Self::construct(
            self.year_month.checked_sub(rhs.year_month)?,
            self.day_time.checked_sub(rhs.day_time)?,
        )
        .ok()
    }

    /// Unary negation.
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            year_month: self.year_month.checked_neg()?,
            day_time: self.day_time.checked_neg()?,
        })
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self == other
    }
}

impl TryFrom<StdDuration> for Duration {
    type Error = DurationOverflowError;

    #[inline]
    fn try_from(value: StdDuration) -> Result<Self, Self::Error> {
        Ok(DayTimeDuration::try_from(value)?.into())
    }
}

impl FromStr for Duration {
    type Err = ParseDurationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let parts = ensure_complete(input, duration_parts)?;
        if parts.year_month.is_none() && parts.day_time.is_none() {
            return Err(Self::Err::msg("Empty duration"));
        }
        Ok(Self::new(
            parts.year_month.unwrap_or(0),
            parts.day_time.unwrap_or_default(),
        )?)
    }
}

impl fmt::Display for Duration {
    /// Formats the canonical representation, e.g. `P1Y2M`, `-PT1.5S` or `PT0S`
    /// for the zero duration.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let months_total = self.year_month.months;
        let seconds_total = self.day_time.seconds;
        let zero = Decimal::from(0);

        if (months_total < 0 && seconds_total > zero) || (months_total > 0 && seconds_total < zero)
        {
            // Not able to format a duration with a partially negative value.
            return Err(fmt::Error);
        }
        if months_total < 0 || seconds_total < zero {
            f.write_str("-")?;
        }
        f.write_str("P")?;

        if months_total == 0 && seconds_total == zero {
            return f.write_str("T0S");
        }

        {
            let years = months_total / 12;
            let months = months_total % 12;

            if years != 0 {
                if months == 0 {
                    write!(f, "{}Y", years.abs())?;
                } else {
                    write!(f, "{}Y{}M", years.abs(), months.abs())?;
                }
            } else if months != 0 || seconds_total == zero {
                write!(f, "{}M", months.abs())?;
            }
        }

        {
            let seconds_int = seconds_total.as_i128();
            let days = seconds_int / 86400;
            let hours = (seconds_int % 86400) / 3600;
            let minutes = (seconds_int % 3600) / 60;
            let fraction = seconds_total
                .checked_sub(
                    Decimal::try_from(days * 86400 + hours * 3600 + minutes * 60)
                        .map_err(|_| fmt::Error)?,
                )
                .ok_or(fmt::Error)?;

            if days != 0 {
                write!(f, "{}D", days.abs())?;
            }

            if hours != 0 || minutes != 0 || fraction != zero {
                f.write_str("T")?;
                if hours != 0 {
                    write!(f, "{}H", hours.abs())?;
                }
                if minutes != 0 {
                    write!(f, "{}M", minutes.abs())?;
                }
                if fraction != zero {
                    write!(f, "{}S", fraction.checked_abs().ok_or(fmt::Error)?)?;
                }
            }
        }
        Ok(())
    }
}

impl PartialOrd for Duration {
    /// Partial order per the XSD rules: two durations are comparable only if they
    /// compare identically against the four reference [`DateTime`]s chosen so that
    /// the outcome is determinate, and incomparable (returns `None`) otherwise —
    /// e.g. `P1M` versus `P30D`.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let first = DateTime::new(1969, 9, 1, 0, 0, Decimal::from(0), None).ok()?;
        let first_result = first
            .checked_add_duration(*self)?
            .partial_cmp(&first.checked_add_duration(*other)?);
        let second = DateTime::new(1697, 2, 1, 0, 0, Decimal::from(0), None).ok()?;
        let second_result = second
            .checked_add_duration(*self)?
            .partial_cmp(&second.checked_add_duration(*other)?);
        let third = DateTime::new(1903, 3, 1, 0, 0, Decimal::from(0), None).ok()?;
        let third_result = third
            .checked_add_duration(*self)?
            .partial_cmp(&third.checked_add_duration(*other)?);
        let fourth = DateTime::new(1903, 7, 1, 0, 0, Decimal::from(0), None).ok()?;
        let fourth_result = fourth
            .checked_add_duration(*self)?
            .partial_cmp(&fourth.checked_add_duration(*other)?);
        if first_result == second_result
            && second_result == third_result
            && third_result == fourth_result
        {
            first_result
        } else {
            None
        }
    }
}

/// [XML Schema `yearMonthDuration` datatype](https://www.w3.org/TR/xmlschema11-2/#yearMonthDuration)
///
/// It stores the duration as a number of months encoded using a [`i64`].
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy, Hash, Default)]
pub struct YearMonthDuration {
    months: i64,
}

impl YearMonthDuration {
    /// The largest representable `yearMonthDuration` value.
    pub const MAX: Self = Self { months: i64::MAX };
    /// The smallest representable `yearMonthDuration` value.
    pub const MIN: Self = Self { months: i64::MIN };

    /// Builds a `yearMonthDuration` from a number of months.
    #[inline]
    pub fn new(months: impl Into<i64>) -> Self {
        Self {
            months: months.into(),
        }
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self {
            months: i64::from_be_bytes(bytes),
        }
    }

    /// [fn:years-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-years-from-duration)
    #[inline]
    #[must_use]
    pub fn years(self) -> i64 {
        self.months / 12
    }

    /// [fn:months-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-months-from-duration)
    #[inline]
    #[must_use]
    pub fn months(self) -> i64 {
        self.months % 12
    }

    #[inline]
    #[must_use]
    pub(crate) const fn all_months(self) -> i64 {
        self.months
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 8] {
        self.months.to_be_bytes()
    }

    /// [op:add-yearMonthDurations](https://www.w3.org/TR/xpath-functions-31/#func-add-yearMonthDurations)
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        Some(Self {
            months: self.months.checked_add(rhs.months)?,
        })
    }

    /// [op:subtract-yearMonthDurations](https://www.w3.org/TR/xpath-functions-31/#func-subtract-yearMonthDurations)
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        Some(Self {
            months: self.months.checked_sub(rhs.months)?,
        })
    }

    /// Unary negation.
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            months: self.months.checked_neg()?,
        })
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self == other
    }
}

impl From<YearMonthDuration> for Duration {
    #[inline]
    fn from(value: YearMonthDuration) -> Self {
        Self {
            year_month: value,
            day_time: DayTimeDuration::default(),
        }
    }
}

impl TryFrom<Duration> for YearMonthDuration {
    type Error = DurationOverflowError;

    #[inline]
    fn try_from(value: Duration) -> Result<Self, Self::Error> {
        if value.day_time == DayTimeDuration::default() {
            Ok(value.year_month)
        } else {
            Err(DurationOverflowError)
        }
    }
}

impl FromStr for YearMonthDuration {
    type Err = ParseDurationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let parts = ensure_complete(input, duration_parts)?;
        if parts.day_time.is_some() {
            return Err(Self::Err::msg(
                "There must not be any day or time component in a yearMonthDuration",
            ));
        }
        Ok(Self::new(
            parts
                .year_month
                .ok_or(Self::Err::msg("No year and month values found"))?,
        ))
    }
}

impl fmt::Display for YearMonthDuration {
    /// Formats the canonical representation, using `P0M` for the zero duration.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.months == 0 {
            f.write_str("P0M")
        } else {
            Duration::from(*self).fmt(f)
        }
    }
}

impl PartialEq<Duration> for YearMonthDuration {
    #[inline]
    fn eq(&self, other: &Duration) -> bool {
        Duration::from(*self).eq(other)
    }
}

impl PartialEq<YearMonthDuration> for Duration {
    #[inline]
    fn eq(&self, other: &YearMonthDuration) -> bool {
        self.eq(&Self::from(*other))
    }
}

impl PartialOrd<Duration> for YearMonthDuration {
    #[inline]
    fn partial_cmp(&self, other: &Duration) -> Option<Ordering> {
        Duration::from(*self).partial_cmp(other)
    }
}

impl PartialOrd<YearMonthDuration> for Duration {
    #[inline]
    fn partial_cmp(&self, other: &YearMonthDuration) -> Option<Ordering> {
        self.partial_cmp(&Self::from(*other))
    }
}

/// [XML Schema `dayTimeDuration` datatype](https://www.w3.org/TR/xmlschema11-2/#dayTimeDuration)
///
/// It stores the duration as a number of seconds encoded using a [`Decimal`].
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy, Hash, Default)]
pub struct DayTimeDuration {
    seconds: Decimal,
}

impl DayTimeDuration {
    /// The largest representable `dayTimeDuration` value.
    pub const MAX: Self = Self {
        seconds: Decimal::MAX,
    };
    /// The smallest representable `dayTimeDuration` value.
    pub const MIN: Self = Self {
        seconds: Decimal::MIN,
    };

    /// Builds a `dayTimeDuration` from a number of seconds.
    #[inline]
    pub fn new(seconds: impl Into<Decimal>) -> Self {
        Self {
            seconds: seconds.into(),
        }
    }

    /// Reads the value from its big-endian in-memory encoding.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 16]) -> Self {
        Self {
            seconds: Decimal::from_be_bytes(bytes),
        }
    }

    /// [fn:days-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-days-from-duration)
    #[inline]
    #[must_use]
    pub fn days(self) -> i64 {
        (self.seconds.as_i128() / 86400) as i64
    }

    /// [fn:hours-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-hours-from-duration)
    #[inline]
    #[must_use]
    pub fn hours(self) -> i64 {
        ((self.seconds.as_i128() % 86400) / 3600) as i64
    }

    /// [fn:minutes-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-minutes-from-duration)
    #[inline]
    #[must_use]
    pub fn minutes(self) -> i64 {
        ((self.seconds.as_i128() % 3600) / 60) as i64
    }

    /// [fn:seconds-from-duration](https://www.w3.org/TR/xpath-functions-31/#func-seconds-from-duration)
    #[inline]
    #[must_use]
    pub fn seconds(self) -> Decimal {
        self.seconds
            .checked_rem(60)
            .expect("A remainder by 60 is always defined")
    }

    /// The duration in seconds.
    #[inline]
    #[must_use]
    pub const fn as_seconds(self) -> Decimal {
        self.seconds
    }

    /// Returns the big-endian in-memory encoding of the value.
    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 16] {
        self.seconds.to_be_bytes()
    }

    /// [op:add-dayTimeDurations](https://www.w3.org/TR/xpath-functions-31/#func-add-dayTimeDurations)
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        Some(Self {
            seconds: self.seconds.checked_add(rhs.seconds)?,
        })
    }

    /// [op:subtract-dayTimeDurations](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dayTimeDurations)
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        Some(Self {
            seconds: self.seconds.checked_sub(rhs.seconds)?,
        })
    }

    /// Unary negation.
    ///
    /// Returns `None` in case of overflow ([`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002)).
    #[inline]
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            seconds: self.seconds.checked_neg()?,
        })
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self == other
    }
}

impl From<DayTimeDuration> for Duration {
    #[inline]
    fn from(value: DayTimeDuration) -> Self {
        Self {
            year_month: YearMonthDuration::default(),
            day_time: value,
        }
    }
}

impl TryFrom<Duration> for DayTimeDuration {
    type Error = DurationOverflowError;

    #[inline]
    fn try_from(value: Duration) -> Result<Self, Self::Error> {
        if value.year_month == YearMonthDuration::default() {
            Ok(value.day_time)
        } else {
            Err(DurationOverflowError)
        }
    }
}

impl TryFrom<StdDuration> for DayTimeDuration {
    type Error = DurationOverflowError;

    #[inline]
    fn try_from(value: StdDuration) -> Result<Self, Self::Error> {
        Ok(Self {
            seconds: Decimal::new(
                i128::try_from(value.as_nanos()).map_err(|_| DurationOverflowError)?,
                9,
            )
            .map_err(|_| DurationOverflowError)?,
        })
    }
}

impl TryFrom<DayTimeDuration> for StdDuration {
    type Error = DurationOverflowError;

    #[inline]
    fn try_from(value: DayTimeDuration) -> Result<Self, Self::Error> {
        if value.seconds.is_negative() {
            return Err(DurationOverflowError);
        }
        let secs = value.seconds.checked_floor().ok_or(DurationOverflowError)?;
        let nanos = value
            .seconds
            .checked_sub(secs)
            .ok_or(DurationOverflowError)?
            .checked_mul(1_000_000_000)
            .ok_or(DurationOverflowError)?
            .checked_floor()
            .ok_or(DurationOverflowError)?;
        Ok(Self::new(
            secs.as_i128()
                .try_into()
                .map_err(|_| DurationOverflowError)?,
            nanos
                .as_i128()
                .try_into()
                .map_err(|_| DurationOverflowError)?,
        ))
    }
}

impl FromStr for DayTimeDuration {
    type Err = ParseDurationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let parts = ensure_complete(input, duration_parts)?;
        if parts.year_month.is_some() {
            return Err(Self::Err::msg(
                "There must not be any year or month component in a dayTimeDuration",
            ));
        }
        Ok(Self::new(
            parts
                .day_time
                .ok_or(Self::Err::msg("No day or time values found"))?,
        ))
    }
}

impl fmt::Display for DayTimeDuration {
    /// Formats the canonical representation, delegating to [`Duration`].
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Duration::from(*self).fmt(f)
    }
}

impl PartialEq<Duration> for DayTimeDuration {
    #[inline]
    fn eq(&self, other: &Duration) -> bool {
        Duration::from(*self).eq(other)
    }
}

impl PartialEq<DayTimeDuration> for Duration {
    #[inline]
    fn eq(&self, other: &DayTimeDuration) -> bool {
        self.eq(&Self::from(*other))
    }
}

impl PartialEq<YearMonthDuration> for DayTimeDuration {
    #[inline]
    fn eq(&self, other: &YearMonthDuration) -> bool {
        Duration::from(*self).eq(&Duration::from(*other))
    }
}

impl PartialEq<DayTimeDuration> for YearMonthDuration {
    #[inline]
    fn eq(&self, other: &DayTimeDuration) -> bool {
        Duration::from(*self).eq(&Duration::from(*other))
    }
}

impl PartialOrd<Duration> for DayTimeDuration {
    #[inline]
    fn partial_cmp(&self, other: &Duration) -> Option<Ordering> {
        Duration::from(*self).partial_cmp(other)
    }
}

impl PartialOrd<DayTimeDuration> for Duration {
    #[inline]
    fn partial_cmp(&self, other: &DayTimeDuration) -> Option<Ordering> {
        self.partial_cmp(&Self::from(*other))
    }
}

impl PartialOrd<YearMonthDuration> for DayTimeDuration {
    #[inline]
    fn partial_cmp(&self, other: &YearMonthDuration) -> Option<Ordering> {
        Duration::from(*self).partial_cmp(&Duration::from(*other))
    }
}

impl PartialOrd<DayTimeDuration> for YearMonthDuration {
    #[inline]
    fn partial_cmp(&self, other: &DayTimeDuration) -> Option<Ordering> {
        Duration::from(*self).partial_cmp(&Duration::from(*other))
    }
}

// [6]   duYearFrag      ::= unsignedNoDecimalPtNumeral 'Y'
// [7]   duMonthFrag     ::= unsignedNoDecimalPtNumeral 'M'
// [8]   duDayFrag       ::= unsignedNoDecimalPtNumeral 'D'
// [9]   duHourFrag      ::= unsignedNoDecimalPtNumeral 'H'
// [10]  duMinuteFrag    ::= unsignedNoDecimalPtNumeral 'M'
// [11]  duSecondFrag    ::= (unsignedNoDecimalPtNumeral | unsignedDecimalPtNumeral) 'S'
// [12]  duYearMonthFrag ::= (duYearFrag duMonthFrag?) | duMonthFrag
// [13]  duTimeFrag      ::= 'T' ((duHourFrag duMinuteFrag? duSecondFrag?) | (duMinuteFrag duSecondFrag?) | duSecondFrag)
// [14]  duDayTimeFrag   ::= (duDayFrag duTimeFrag?) | duTimeFrag
// [15]  durationLexicalRep ::= '-'? 'P' ((duYearMonthFrag duDayTimeFrag?) | duDayTimeFrag)
struct DurationParts {
    year_month: Option<i64>,
    day_time: Option<Decimal>,
}

fn duration_parts(input: &str) -> Result<(DurationParts, &str), ParseDurationError> {
    // States.
    const START: u32 = 0;
    const AFTER_YEAR: u32 = 1;
    const AFTER_MONTH: u32 = 2;
    const AFTER_DAY: u32 = 3;
    const AFTER_T: u32 = 4;
    const AFTER_HOUR: u32 = 5;
    const AFTER_MINUTE: u32 = 6;
    const AFTER_SECOND: u32 = 7;

    let (is_negative, input) = if let Some(left) = input.strip_prefix('-') {
        (true, left)
    } else {
        (false, input)
    };
    let mut input = expect_char(input, 'P', "Durations must start with 'P'")?;
    let mut state = START;
    let mut year_month: Option<i64> = None;
    let mut day_time: Option<Decimal> = None;
    while !input.is_empty() {
        if let Some(left) = input.strip_prefix('T') {
            if state >= AFTER_T {
                return Err(ParseDurationError::msg("Duplicated time separator 'T'"));
            }
            state = AFTER_T;
            input = left;
        } else {
            let (number_str, left) = decimal_prefix(input);
            match left.chars().next() {
                Some('Y') if state < AFTER_YEAR => {
                    require_component_digits(number_str)?;
                    if number_str.contains('.') {
                        return Err(ParseDurationError::msg(
                            "Decimal numbers are not allowed for years",
                        ));
                    }
                    year_month = Some(
                        year_month
                            .unwrap_or_default()
                            .checked_add(
                                apply_i64_neg(
                                    i64::from_str(number_str).map_err(|_| OVERFLOW_ERROR)?,
                                    is_negative,
                                )?
                                .checked_mul(12)
                                .ok_or(OVERFLOW_ERROR)?,
                            )
                            .ok_or(OVERFLOW_ERROR)?,
                    );
                    state = AFTER_YEAR;
                }
                Some('M') if state < AFTER_MONTH => {
                    require_component_digits(number_str)?;
                    if number_str.contains('.') {
                        return Err(ParseDurationError::msg(
                            "Decimal numbers are not allowed for months",
                        ));
                    }
                    year_month = Some(
                        year_month
                            .unwrap_or_default()
                            .checked_add(apply_i64_neg(
                                i64::from_str(number_str).map_err(|_| OVERFLOW_ERROR)?,
                                is_negative,
                            )?)
                            .ok_or(OVERFLOW_ERROR)?,
                    );
                    state = AFTER_MONTH;
                }
                Some('D') if state < AFTER_DAY => {
                    require_component_digits(number_str)?;
                    if number_str.contains('.') {
                        return Err(ParseDurationError::msg(
                            "Decimal numbers are not allowed for days",
                        ));
                    }
                    day_time = Some(
                        day_time
                            .unwrap_or_default()
                            .checked_add(
                                apply_decimal_neg(
                                    Decimal::from_str(number_str).map_err(|_| OVERFLOW_ERROR)?,
                                    is_negative,
                                )?
                                .checked_mul(86400)
                                .ok_or(OVERFLOW_ERROR)?,
                            )
                            .ok_or(OVERFLOW_ERROR)?,
                    );
                    state = AFTER_DAY;
                }
                Some('H') if state == AFTER_T => {
                    require_component_digits(number_str)?;
                    if number_str.contains('.') {
                        return Err(ParseDurationError::msg(
                            "Decimal numbers are not allowed for hours",
                        ));
                    }
                    day_time = Some(
                        day_time
                            .unwrap_or_default()
                            .checked_add(
                                apply_decimal_neg(
                                    Decimal::from_str(number_str).map_err(|_| OVERFLOW_ERROR)?,
                                    is_negative,
                                )?
                                .checked_mul(3600)
                                .ok_or(OVERFLOW_ERROR)?,
                            )
                            .ok_or(OVERFLOW_ERROR)?,
                    );
                    state = AFTER_HOUR;
                }
                Some('M') if (AFTER_T..AFTER_MINUTE).contains(&state) => {
                    require_component_digits(number_str)?;
                    if number_str.contains('.') {
                        return Err(ParseDurationError::msg(
                            "Decimal numbers are not allowed for minutes",
                        ));
                    }
                    day_time = Some(
                        day_time
                            .unwrap_or_default()
                            .checked_add(
                                apply_decimal_neg(
                                    Decimal::from_str(number_str).map_err(|_| OVERFLOW_ERROR)?,
                                    is_negative,
                                )?
                                .checked_mul(60)
                                .ok_or(OVERFLOW_ERROR)?,
                            )
                            .ok_or(OVERFLOW_ERROR)?,
                    );
                    state = AFTER_MINUTE;
                }
                Some('S') if (AFTER_T..AFTER_SECOND).contains(&state) => {
                    require_component_digits(number_str)?;
                    require_digits_around_decimal_point(number_str)?;
                    day_time = Some(
                        day_time
                            .unwrap_or_default()
                            .checked_add(apply_decimal_neg(
                                Decimal::from_str(number_str).map_err(|_| OVERFLOW_ERROR)?,
                                is_negative,
                            )?)
                            .ok_or(OVERFLOW_ERROR)?,
                    );
                    state = AFTER_SECOND;
                }
                Some(_) => return Err(ParseDurationError::msg("Unexpected type character")),
                None => {
                    return Err(ParseDurationError::msg(
                        "Numbers in durations must be followed by a type character",
                    ));
                }
            }
            input = &left[1..];
        }
    }

    // [13] duTimeFrag ::= 'T' ((duHourFrag duMinuteFrag? duSecondFrag?) | (duMinuteFrag
    // duSecondFrag?) | duSecondFrag) — the 'T' designator is not optional trailing noise: at
    // least one of the hour, minute or second components must follow it. Staying in `AFTER_T`
    // when the input runs out means the lexical representation ended on a dangling 'T'
    // (`P1YT`, `P1Y1M1DT`, `P1DT`, …), which is a lexical error, not a duration without a
    // time part.
    if state == AFTER_T {
        return Err(ParseDurationError::msg(
            "The time separator 'T' must be followed by at least one hour, minute or second component",
        ));
    }

    Ok((
        DurationParts {
            year_month,
            day_time,
        },
        input,
    ))
}

/// Rejects a component (`Y`, `M`, `D`, `H`, `M` or `S`) whose numeral is missing entirely —
/// e.g. `PD` or the bare-dot form `PT.S` — before it reaches [`i64::from_str`] or
/// [`Decimal::from_str`].
///
/// Without this guard, [`i64::from_str("")`](i64::from_str) and
/// [`Decimal::from_str("")`](Decimal::from_str) both fail, and that failure was
/// indistinguishable from a genuine magnitude overflow (both were mapped to
/// [`OVERFLOW_ERROR`]) — a missing digit sequence is a lexical error, not an overflow, and
/// must not make [`ParseDurationError::is_overflow`] answer `true`.
fn require_component_digits(number_str: &str) -> Result<(), ParseDurationError> {
    if number_str.is_empty() || number_str == "." {
        return Err(ParseDurationError::msg(
            "A digit sequence is expected before the component letter",
        ));
    }
    Ok(())
}

/// Rejects a seconds component whose numeral begins or ends with the decimal
/// point — `PT.5S`, `PT30.S` — which [`Decimal::from_str`] itself accepts,
/// since `.5` and `30.` are both legal `xs:decimal` lexical forms.
///
/// The `xs:duration` lexical space is narrower than `xs:decimal`'s here. XSD
/// 1.1 Part 2 §3.3.6.2 states the seconds fraction twice, and the two
/// statements disagree: the `duSecondFrag` BNF permits an omitted integer or
/// fraction part, while the regular expression given for the same production in
/// the same section requires a digit on both sides. The XML Schema Test Suite
/// resolves it against the BNF — `saxonData/Simple/simple086` declares
/// `PT12H30M12.5S` valid and both `PT12H30M12.S` ("trailing '.' not allowed")
/// and `PT12H30M.5S` ("leading '.' not allowed") invalid, noting "the BNF is
/// believed to be in error" — and QT3 agrees independently, in four separate
/// cases that all expect `FORG0001`: `prod/CastExpr` `CastAs688`
/// (`xs:duration('PT.5S')`), `CastAs689` (`xs:dayTimeDuration('PT10M.5S')`),
/// `CastAs690` (`xs:duration('PT30.S')`), and `CastAs691`
/// (`xs:dayTimeDuration('PT10M30.S')`). XSD 1.0's grammar, which has no such
/// ambiguity, requires the digits too.
///
/// Only the seconds component can carry a fraction at all; every other
/// component rejects a `.` outright.
fn require_digits_around_decimal_point(number_str: &str) -> Result<(), ParseDurationError> {
    if number_str.starts_with('.') || number_str.ends_with('.') {
        return Err(ParseDurationError::msg(
            "The seconds component must have a digit on both sides of the decimal point",
        ));
    }
    Ok(())
}

fn apply_i64_neg(value: i64, is_negative: bool) -> Result<i64, ParseDurationError> {
    if is_negative {
        value.checked_neg().ok_or(OVERFLOW_ERROR)
    } else {
        Ok(value)
    }
}

fn apply_decimal_neg(value: Decimal, is_negative: bool) -> Result<Decimal, ParseDurationError> {
    if is_negative {
        value.checked_neg().ok_or(OVERFLOW_ERROR)
    } else {
        Ok(value)
    }
}

fn ensure_complete<T>(
    input: &str,
    parse: impl FnOnce(&str) -> Result<(T, &str), ParseDurationError>,
) -> Result<T, ParseDurationError> {
    let (result, left) = parse(input)?;
    if !left.is_empty() {
        return Err(ParseDurationError::msg("Unrecognized value suffix"));
    }
    Ok(result)
}

fn expect_char<'a>(
    input: &'a str,
    constant: char,
    error_message: &'static str,
) -> Result<&'a str, ParseDurationError> {
    if let Some(left) = input.strip_prefix(constant) {
        Ok(left)
    } else {
        Err(ParseDurationError::msg(error_message))
    }
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

/// An error raised while parsing a [`Duration`], [`YearMonthDuration`] or
/// [`DayTimeDuration`].
#[derive(Debug, Clone)]
pub struct ParseDurationError {
    kind: ParseDurationErrorKind,
}

/// The internal classification behind [`ParseDurationError`].
///
/// Kept private — [`ParseDurationError::is_overflow`] is the additive, stable way to read it
/// — so new variants can be added later without being a breaking change.
#[derive(Debug, Clone, Copy)]
enum ParseDurationErrorKind {
    /// A component's magnitude (or the accumulated year-month/day-time total) does not fit
    /// the internal representation.
    Overflow,
    /// Any other lexical (grammar) violation.
    Message(&'static str),
}

const OVERFLOW_ERROR: ParseDurationError = ParseDurationError {
    kind: ParseDurationErrorKind::Overflow,
};

impl ParseDurationError {
    const fn msg(msg: &'static str) -> Self {
        Self {
            kind: ParseDurationErrorKind::Message(msg),
        }
    }

    /// Returns `true` if this error is an **overflow** of the internal duration
    /// representation — a component, or the accumulated year-month/day-time total, too
    /// large to hold — as opposed to a **lexical** (grammar) error such as a missing digit
    /// sequence or a misplaced decimal point.
    ///
    /// This lets callers distinguish the two failure modes additively, without matching on
    /// the (private) error internals: an XPath implementation should raise
    /// [`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002) for an overflow,
    /// rather than the generic invalid-lexical-value error that a plain grammar mismatch
    /// deserves.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xsd::Duration;
    /// use std::str::FromStr;
    ///
    /// // 18 nines of years fits `i64`, but overflows once multiplied out to months.
    /// let huge = Duration::from_str("P999999999999999999Y").unwrap_err();
    /// assert!(huge.is_overflow());
    ///
    /// // A genuinely malformed lexical form is not an overflow.
    /// let garbage = Duration::from_str("not-a-duration").unwrap_err();
    /// assert!(!garbage.is_overflow());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_overflow(&self) -> bool {
        matches!(self.kind, ParseDurationErrorKind::Overflow)
    }
}

impl fmt::Display for ParseDurationError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self.kind {
            ParseDurationErrorKind::Overflow => "Overflow error",
            ParseDurationErrorKind::Message(msg) => msg,
        })
    }
}

impl Error for ParseDurationError {}

/// An overflow during [`Duration`]-related operations.
///
/// Matches XPath [`FODT0002` error](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002).
#[derive(Debug, Clone, Copy)]
pub struct DurationOverflowError;

impl fmt::Display for DurationOverflowError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("overflow during xsd:duration computation")
    }
}

impl Error for DurationOverflowError {}

/// The year-month and the day-time components of a [`Duration`] have an opposite sign.
#[derive(Debug, Clone, Copy)]
pub struct OppositeSignInDurationComponentsError;

impl fmt::Display for OppositeSignInDurationComponentsError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "The xsd:yearMonthDuration and xsd:dayTimeDuration components of a xsd:duration can't have opposite sign",
        )
    }
}

impl Error for OppositeSignInDurationComponentsError {}

impl From<OppositeSignInDurationComponentsError> for ParseDurationError {
    #[inline]
    fn from(_: OppositeSignInDurationComponentsError) -> Self {
        Self::msg(
            "The xsd:yearMonthDuration and xsd:dayTimeDuration components of a xsd:duration can't have opposite sign",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn from_str() -> Result<(), ParseDurationError> {
        let min = Duration::new(i64::MIN, Decimal::MIN)?;
        let max = Duration::new(i64::MAX, Decimal::MAX)?;

        assert_eq!(YearMonthDuration::from_str("P1Y")?.to_string(), "P1Y");
        assert_eq!(Duration::from_str("P1Y")?.to_string(), "P1Y");
        assert_eq!(YearMonthDuration::from_str("P1M")?.to_string(), "P1M");
        assert_eq!(Duration::from_str("P1M")?.to_string(), "P1M");
        assert_eq!(DayTimeDuration::from_str("P1D")?.to_string(), "P1D");
        assert_eq!(Duration::from_str("P1D")?.to_string(), "P1D");
        assert_eq!(DayTimeDuration::from_str("PT1H")?.to_string(), "PT1H");
        assert_eq!(Duration::from_str("PT1H")?.to_string(), "PT1H");
        assert_eq!(DayTimeDuration::from_str("PT1M")?.to_string(), "PT1M");
        assert_eq!(Duration::from_str("PT1M")?.to_string(), "PT1M");
        assert_eq!(DayTimeDuration::from_str("PT1.1S")?.to_string(), "PT1.1S");
        assert_eq!(Duration::from_str("PT1.1S")?.to_string(), "PT1.1S");
        assert_eq!(YearMonthDuration::from_str("-P1Y")?.to_string(), "-P1Y");
        assert_eq!(Duration::from_str("-P1Y")?.to_string(), "-P1Y");
        assert_eq!(YearMonthDuration::from_str("-P1M")?.to_string(), "-P1M");
        assert_eq!(Duration::from_str("-P1M")?.to_string(), "-P1M");
        assert_eq!(DayTimeDuration::from_str("-P1D")?.to_string(), "-P1D");
        assert_eq!(Duration::from_str("-P1D")?.to_string(), "-P1D");
        assert_eq!(DayTimeDuration::from_str("-PT1H")?.to_string(), "-PT1H");
        assert_eq!(Duration::from_str("-PT1H")?.to_string(), "-PT1H");
        assert_eq!(DayTimeDuration::from_str("-PT1M")?.to_string(), "-PT1M");
        assert_eq!(Duration::from_str("-PT1M")?.to_string(), "-PT1M");
        assert_eq!(DayTimeDuration::from_str("-PT1S")?.to_string(), "-PT1S");
        assert_eq!(Duration::from_str("-PT1S")?.to_string(), "-PT1S");
        assert_eq!(DayTimeDuration::from_str("-PT1.1S")?.to_string(), "-PT1.1S");
        assert_eq!(Duration::from_str("-PT1.1S")?.to_string(), "-PT1.1S");
        assert_eq!(Duration::from_str(&max.to_string())?, max);
        assert_eq!(Duration::from_str(&min.to_string())?, min);
        assert_eq!(Duration::from_str("PT0H")?.to_string(), "PT0S");
        assert_eq!(Duration::from_str("-PT0H")?.to_string(), "PT0S");
        assert_eq!(YearMonthDuration::from_str("P0Y")?.to_string(), "P0M");
        assert_eq!(DayTimeDuration::from_str("PT0H")?.to_string(), "PT0S");
        Ok(())
    }

    /// The seconds component needs a digit on each side of its decimal point,
    /// even though `xs:decimal` itself accepts `12.` and `.5`.
    ///
    /// XSTS `saxonData/Simple/simple086` states the rule against `xs:duration`
    /// directly: `simple086.v01.xml` (`PT12H30M12.5S`) is valid,
    /// `simple086.n01.xml` (`PT12H30M12.S`, "trailing '.' not allowed") and
    /// `simple086.n02.xml` (`PT12H30M.5S`, "leading '.' not allowed") are not.
    /// QT3 `prod/CastExpr` `CastAs688`-`CastAs691` repeat it for both
    /// `xs:duration` and `xs:dayTimeDuration`, all four expecting `FORG0001`.
    #[test]
    fn seconds_fraction_needs_digits_on_both_sides() -> Result<(), ParseDurationError> {
        // simple086.v01.
        assert_eq!(
            Duration::from_str("PT12H30M12.5S")?.to_string(),
            "PT12H30M12.5S"
        );

        for lexical in [
            // simple086.n01, and QT3 CastAs690 (`PT30.S`) / CastAs691.
            "PT12H30M12.S",
            "PT30.S",
            "PT10M30.S",
            // simple086.n02, and QT3 CastAs688 (`PT.5S`) / CastAs689.
            "PT12H30M.5S",
            "PT.5S",
            "PT10M.5S",
            // A lone dot was already rejected, as the missing digit sequence
            // it is; it must stay a lexical error rather than an overflow.
            "PT.S",
        ] {
            let error = Duration::from_str(lexical)
                .expect_err("the seconds fraction must have digits on both sides");
            assert!(
                !error.is_overflow(),
                "{lexical}: a lexical error, not an overflow"
            );
            assert!(DayTimeDuration::from_str(lexical).is_err(), "{lexical}");
        }
        Ok(())
    }

    #[test]
    fn from_std() -> Result<(), DurationOverflowError> {
        assert_eq!(
            Duration::try_from(StdDuration::new(10, 10))?.to_string(),
            "PT10.00000001S"
        );
        Ok(())
    }

    #[test]
    fn to_std() -> Result<(), Box<dyn Error>> {
        let duration = StdDuration::try_from(DayTimeDuration::from_str("PT10.00000001S")?)?;
        assert_eq!(duration.as_secs(), 10);
        assert_eq!(duration.subsec_nanos(), 10);
        Ok(())
    }

    #[test]
    fn to_be_bytes() {
        assert_eq!(
            Duration::from_be_bytes(Duration::MIN.to_be_bytes()),
            Duration::MIN
        );
        assert_eq!(
            Duration::from_be_bytes(Duration::MAX.to_be_bytes()),
            Duration::MAX
        );
        assert_eq!(
            YearMonthDuration::from_be_bytes(YearMonthDuration::MIN.to_be_bytes()),
            YearMonthDuration::MIN
        );
        assert_eq!(
            YearMonthDuration::from_be_bytes(YearMonthDuration::MAX.to_be_bytes()),
            YearMonthDuration::MAX
        );
        assert_eq!(
            DayTimeDuration::from_be_bytes(DayTimeDuration::MIN.to_be_bytes()),
            DayTimeDuration::MIN
        );
        assert_eq!(
            DayTimeDuration::from_be_bytes(DayTimeDuration::MAX.to_be_bytes()),
            DayTimeDuration::MAX
        );
    }

    #[test]
    fn equals() -> Result<(), ParseDurationError> {
        assert_eq!(
            YearMonthDuration::from_str("P1Y")?,
            YearMonthDuration::from_str("P12M")?
        );
        assert_eq!(
            YearMonthDuration::from_str("P1Y")?,
            Duration::from_str("P12M")?
        );
        assert_eq!(
            Duration::from_str("P1Y")?,
            YearMonthDuration::from_str("P12M")?
        );
        assert_eq!(Duration::from_str("P1Y")?, Duration::from_str("P12M")?);
        assert_eq!(
            DayTimeDuration::from_str("PT24H")?,
            DayTimeDuration::from_str("P1D")?
        );
        assert_eq!(
            DayTimeDuration::from_str("PT24H")?,
            Duration::from_str("P1D")?
        );
        assert_eq!(
            Duration::from_str("PT24H")?,
            DayTimeDuration::from_str("P1D")?
        );
        assert_eq!(Duration::from_str("PT24H")?, Duration::from_str("P1D")?);
        assert_ne!(Duration::from_str("P1Y")?, Duration::from_str("P365D")?);
        assert_eq!(Duration::from_str("P0Y")?, Duration::from_str("P0D")?);
        assert_ne!(Duration::from_str("P1Y")?, Duration::from_str("P365D")?);
        assert_eq!(Duration::from_str("P2Y")?, Duration::from_str("P24M")?);
        assert_eq!(Duration::from_str("P10D")?, Duration::from_str("PT240H")?);
        assert_eq!(
            Duration::from_str("P2Y0M0DT0H0M0S")?,
            Duration::from_str("P24M")?
        );
        assert_eq!(
            Duration::from_str("P0Y0M10D")?,
            Duration::from_str("PT240H")?
        );
        assert_ne!(Duration::from_str("P1M")?, Duration::from_str("P30D")?);
        Ok(())
    }

    #[test]
    fn cmp() -> Result<(), ParseDurationError> {
        assert!(Duration::from_str("P1Y1D")? < Duration::from_str("P13MT25H")?);
        assert!(YearMonthDuration::from_str("P1Y")? < YearMonthDuration::from_str("P13M")?);
        assert!(Duration::from_str("P1Y")? < YearMonthDuration::from_str("P13M")?);
        assert!(YearMonthDuration::from_str("P1Y")? < Duration::from_str("P13M")?);
        assert!(DayTimeDuration::from_str("P1D")? < DayTimeDuration::from_str("PT25H")?);
        assert!(DayTimeDuration::from_str("PT1H")? < DayTimeDuration::from_str("PT61M")?);
        assert!(DayTimeDuration::from_str("PT1M")? < DayTimeDuration::from_str("PT61S")?);
        assert!(Duration::from_str("PT1H")? < DayTimeDuration::from_str("PT61M")?);
        assert!(DayTimeDuration::from_str("PT1H")? < Duration::from_str("PT61M")?);
        assert!(YearMonthDuration::from_str("P1M")? < DayTimeDuration::from_str("P40D")?);
        assert!(DayTimeDuration::from_str("P25D")? < YearMonthDuration::from_str("P1M")?);
        // P1M versus P30D is indeterminate.
        assert_eq!(
            Duration::from_str("P1M")?.partial_cmp(&Duration::from_str("P30D")?),
            None
        );
        Ok(())
    }

    #[test]
    fn years() -> Result<(), ParseDurationError> {
        assert_eq!(Duration::from_str("P20Y15M")?.years(), 21);
        assert_eq!(Duration::from_str("-P15M")?.years(), -1);
        assert_eq!(Duration::from_str("-P2DT15H")?.years(), 0);
        Ok(())
    }

    #[test]
    fn months() -> Result<(), ParseDurationError> {
        assert_eq!(Duration::from_str("P20Y15M")?.months(), 3);
        assert_eq!(Duration::from_str("-P20Y18M")?.months(), -6);
        assert_eq!(Duration::from_str("-P2DT15H0M0S")?.months(), 0);
        Ok(())
    }

    #[test]
    fn days() -> Result<(), ParseDurationError> {
        assert_eq!(Duration::from_str("P3DT10H")?.days(), 3);
        assert_eq!(Duration::from_str("P3DT55H")?.days(), 5);
        assert_eq!(Duration::from_str("P3Y5M")?.days(), 0);
        Ok(())
    }

    #[test]
    fn hours() -> Result<(), ParseDurationError> {
        assert_eq!(Duration::from_str("P3DT10H")?.hours(), 10);
        assert_eq!(Duration::from_str("P3DT12H32M12S")?.hours(), 12);
        assert_eq!(Duration::from_str("PT123H")?.hours(), 3);
        assert_eq!(Duration::from_str("-P3DT10H")?.hours(), -10);
        Ok(())
    }

    #[test]
    fn minutes() -> Result<(), ParseDurationError> {
        assert_eq!(Duration::from_str("P3DT10H")?.minutes(), 0);
        assert_eq!(Duration::from_str("-P5DT12H30M")?.minutes(), -30);
        Ok(())
    }

    #[test]
    fn seconds() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            Duration::from_str("P3DT10H12.5S")?.seconds(),
            Decimal::from_str("12.5")?
        );
        assert_eq!(
            Duration::from_str("-PT256S")?.seconds(),
            Decimal::from_str("-16.0")?
        );
        Ok(())
    }

    #[test]
    fn add() -> Result<(), ParseDurationError> {
        assert_eq!(
            Duration::from_str("P2Y11M")?.checked_add(Duration::from_str("P3Y3M")?),
            Some(Duration::from_str("P6Y2M")?)
        );
        assert_eq!(
            Duration::from_str("P2DT12H5M")?.checked_add(Duration::from_str("P5DT12H")?),
            Some(Duration::from_str("P8DT5M")?)
        );
        assert_eq!(
            Duration::from_str("P1M2D")?.checked_add(Duration::from_str("-P3D")?),
            None
        );
        assert_eq!(
            Duration::from_str("P1M2D")?.checked_add(Duration::from_str("-P2M")?),
            None
        );
        Ok(())
    }

    #[test]
    fn sub() -> Result<(), ParseDurationError> {
        assert_eq!(
            Duration::from_str("P2Y11M")?.checked_sub(Duration::from_str("P3Y3M")?),
            Some(Duration::from_str("-P4M")?)
        );
        assert_eq!(
            Duration::from_str("P2DT12H")?.checked_sub(Duration::from_str("P1DT10H30M")?),
            Some(Duration::from_str("P1DT1H30M")?)
        );
        assert_eq!(
            Duration::from_str("P1M2D")?.checked_sub(Duration::from_str("P3D")?),
            None
        );
        assert_eq!(
            Duration::from_str("P1M2D")?.checked_sub(Duration::from_str("P2M")?),
            None
        );
        Ok(())
    }

    #[test]
    fn minimally_conformant() -> Result<(), ParseDurationError> {
        // Fractional-second duration values to milliseconds.
        assert_eq!(Duration::from_str("PT0.001S")?.to_string(), "PT0.001S");
        assert_eq!(Duration::from_str("-PT0.001S")?.to_string(), "-PT0.001S");

        // Months in the range −119999 to 119999 and seconds in the range −31622400 to 31622400.
        assert_eq!(
            Duration::from_str("P119999MT31622400S")?.to_string(),
            "P9999Y11M366D"
        );
        assert_eq!(
            Duration::from_str("-P119999MT31622400S")?.to_string(),
            "-P9999Y11M366D"
        );
        Ok(())
    }

    /// A component whose magnitude does not fit the internal representation must be
    /// distinguishable, via [`ParseDurationError::is_overflow`], from a genuinely malformed
    /// lexical form -- so that a downstream XPath/XQuery consumer can raise
    /// [`FODT0002`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0002) instead of the
    /// generic invalid-lexical-value error.
    #[test]
    fn overflow_is_distinguishable_from_lexical_errors() {
        // 18 nines of years fits `i64` (`i64::MAX` has 19 digits), but overflows once
        // multiplied out to months: a genuine arithmetic overflow in the checked
        // accumulation, not a digit-parsing failure.
        assert!(
            Duration::from_str("P999999999999999999Y")
                .expect_err("18 nines of years overflows once multiplied out to months")
                .is_overflow()
        );
        assert!(
            Duration::from_str("-P999999999999999999Y")
                .expect_err("18 nines of years overflows once multiplied out to months")
                .is_overflow()
        );
        assert!(
            YearMonthDuration::from_str("P999999999999999999Y")
                .expect_err("18 nines of years overflows once multiplied out to months")
                .is_overflow()
        );

        // So many digits that even the initial `i64::from_str` overflows.
        assert!(
            Duration::from_str("P99999999999999999999999999Y")
                .expect_err("28 nines overflows i64 outright")
                .is_overflow()
        );
        // Same, but accumulated one component at a time on the day-time side.
        assert!(
            DayTimeDuration::from_str("P99999999999999999999999999D")
                .expect_err("28 nines overflows the internal Decimal representation")
                .is_overflow()
        );

        // A misplaced decimal point is a lexical error, not an overflow. Before this fix it
        // was misreported as "Overflow error" purely because `i64::from_str("1.5")` also
        // fails, conflating the two failure modes.
        assert!(
            !Duration::from_str("P1.5Y")
                .expect_err("decimals are not allowed for years")
                .is_overflow()
        );
        assert!(
            !Duration::from_str("P1Y1.5M")
                .expect_err("decimals are not allowed for months")
                .is_overflow()
        );

        // A missing digit sequence is a lexical error, not an overflow -- it used to be
        // misreported as "Overflow error" too, since `i64::from_str("")` and
        // `Decimal::from_str("")` also fail.
        assert!(
            !Duration::from_str("PD")
                .expect_err("no digits before 'D'")
                .is_overflow()
        );
        assert!(
            !Duration::from_str("PT.S")
                .expect_err("a bare '.' is not a digit sequence")
                .is_overflow()
        );
        assert!(
            !Duration::from_str("PTS")
                .expect_err("no digits before 'S'")
                .is_overflow()
        );
        assert!(
            !Duration::from_str("PTM")
                .expect_err("no digits before the minutes 'M'")
                .is_overflow()
        );

        assert!(
            !Duration::from_str("not-a-duration")
                .expect_err("not a duration at all")
                .is_overflow()
        );
        assert!(
            !Duration::from_str("P1Y2X")
                .expect_err("'X' is not a component letter")
                .is_overflow()
        );
    }

    /// Audits the checked arithmetic (`checked_add`/`checked_sub`) at the type boundaries for
    /// silent saturation: overflow must always surface as `None`, never as a silently clamped
    /// or wrapped-around `Some` value.
    #[test]
    fn checked_arithmetic_overflows_cleanly_rather_than_saturating() {
        assert_eq!(
            YearMonthDuration::MAX.checked_add(YearMonthDuration::new(1_i64)),
            None
        );
        assert_eq!(
            YearMonthDuration::MIN.checked_sub(YearMonthDuration::new(1_i64)),
            None
        );
        assert_eq!(YearMonthDuration::MIN.checked_neg(), None);
        assert_eq!(
            DayTimeDuration::MAX.checked_add(DayTimeDuration::new(Decimal::from(1))),
            None
        );
        assert_eq!(
            DayTimeDuration::MIN.checked_sub(DayTimeDuration::new(Decimal::from(1))),
            None
        );
        assert_eq!(DayTimeDuration::MIN.checked_neg(), None);
        assert_eq!(
            Duration::MAX.checked_add(Duration::from(YearMonthDuration::new(1_i64))),
            None
        );
        assert_eq!(
            Duration::MIN.checked_sub(Duration::from(YearMonthDuration::new(1_i64))),
            None
        );
    }
    /// Regression test for the `duTimeFrag` production
    /// ([XSD 1.1 Part 2 §D.3](https://www.w3.org/TR/xmlschema11-2/#duration-lexical-space)
    /// production [13]): `'T'` must be followed by at least one of the hour, minute or second
    /// components. A dangling `T` used to be silently swallowed, so `P1YT` parsed as `P1Y` and
    /// the XSD validator accepted malformed `xsd:duration` literals.
    #[test]
    fn dangling_time_separator_is_a_lexical_error() {
        for lexical in ["P1YT", "P1Y1M1DT", "P1DT", "P1MT", "PT", "-P1YT", "P1Y1MT"] {
            let error = Duration::from_str(lexical)
                .expect_err("a 'T' with no time component is a lexical error");
            assert!(
                !error.is_overflow(),
                "{lexical} must fail as a lexical error, not an overflow"
            );
        }
        assert!(DayTimeDuration::from_str("P1DT").is_err());
        assert!(YearMonthDuration::from_str("P1YT").is_err());

        // The well-formed neighbours keep parsing.
        assert!(Duration::from_str("P1Y").is_ok());
        assert!(Duration::from_str("P1Y1M1D").is_ok());
        assert!(Duration::from_str("P1DT1H").is_ok());
        assert!(Duration::from_str("PT1S").is_ok());
        assert!(Duration::from_str("PT1M").is_ok());
    }
}
