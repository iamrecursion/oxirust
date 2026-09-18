//! The component-extraction, timezone-adjustment and "current instant" functions of
//! [F&O 3.1 §9.2–§9.8](https://www.w3.org/TR/xpath-functions-31/#component-extraction).
//!
//! Every extractor is total on the type its signature names and raises `XPTY0004` on anything
//! else; the duration extractors accept all three duration types, because
//! `xs:yearMonthDuration` simply has a zero day/time part and vice versa.

use crate::context::FnContext;
use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicData, AtomicType, AtomicValue};
use oxixml_xsd::{Date, DateTime, DayTimeDuration, Decimal, Duration, Time, TimezoneOffset};

fn wrong_type(value: &AtomicValue, expected: &str, what: &str) -> FnError {
    FnError::new(
        FnErrorCode::Xpty0004,
        format!("{what} expects {expected}, not {}", value.atomic_type()),
    )
}

/// Views any of the three duration types as an unrestricted [`Duration`], which is what the six
/// `*-from-duration` accessors are defined over.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the value is not a duration.
pub fn as_duration(value: &AtomicValue, what: &str) -> FnResult<Duration> {
    match value.data() {
        AtomicData::Duration(duration) => Ok(*duration),
        AtomicData::YearMonthDuration(duration) => Ok(Duration::from(*duration)),
        AtomicData::DayTimeDuration(duration) => Ok(Duration::from(*duration)),
        _ => Err(wrong_type(value, "a duration", what)),
    }
}

fn as_date_time(value: &AtomicValue, what: &str) -> FnResult<DateTime> {
    match value.data() {
        AtomicData::DateTime(value) => Ok(*value),
        _ => Err(wrong_type(value, "an xs:dateTime", what)),
    }
}

fn as_date(value: &AtomicValue, what: &str) -> FnResult<Date> {
    match value.data() {
        AtomicData::Date(value) => Ok(*value),
        _ => Err(wrong_type(value, "an xs:date", what)),
    }
}

fn as_time(value: &AtomicValue, what: &str) -> FnResult<Time> {
    match value.data() {
        AtomicData::Time(value) => Ok(*value),
        _ => Err(wrong_type(value, "an xs:time", what)),
    }
}

macro_rules! duration_component {
    ($( $name:ident => $accessor:ident, $function:literal ),+ $(,)?) => {
        $(
            #[doc = concat!("[`fn:", $function, "`](https://www.w3.org/TR/xpath-functions-31/#func-", $function, ").")]
            ///
            /// # Errors
            ///
            /// [`FnErrorCode::Xpty0004`] when the argument is not a duration.
            pub fn $name(value: &AtomicValue) -> FnResult<i64> {
                Ok(as_duration(value, concat!("fn:", $function))?.$accessor())
            }
        )+
    };
}

duration_component! {
    years_from_duration => years, "years-from-duration",
    months_from_duration => months, "months-from-duration",
    days_from_duration => days, "days-from-duration",
    hours_from_duration => hours, "hours-from-duration",
    minutes_from_duration => minutes, "minutes-from-duration",
}

/// [`fn:seconds-from-duration`](https://www.w3.org/TR/xpath-functions-31/#func-seconds-from-duration),
/// which is the one duration component that is an `xs:decimal`.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not a duration.
pub fn seconds_from_duration(value: &AtomicValue) -> FnResult<Decimal> {
    Ok(as_duration(value, "fn:seconds-from-duration")?.seconds())
}

macro_rules! date_time_component {
    ($( $name:ident => $accessor:ident, $function:literal ),+ $(,)?) => {
        $(
            #[doc = concat!("[`fn:", $function, "`](https://www.w3.org/TR/xpath-functions-31/#func-", $function, ").")]
            ///
            /// # Errors
            ///
            /// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:dateTime`.
            pub fn $name(value: &AtomicValue) -> FnResult<i64> {
                Ok(i64::from(as_date_time(value, concat!("fn:", $function))?.$accessor()))
            }
        )+
    };
}

date_time_component! {
    month_from_date_time => month, "month-from-dateTime",
    day_from_date_time => day, "day-from-dateTime",
    hours_from_date_time => hour, "hours-from-dateTime",
    minutes_from_date_time => minute, "minutes-from-dateTime",
}

/// [`fn:year-from-dateTime`](https://www.w3.org/TR/xpath-functions-31/#func-year-from-dateTime).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:dateTime`.
pub fn year_from_date_time(value: &AtomicValue) -> FnResult<i64> {
    Ok(as_date_time(value, "fn:year-from-dateTime")?.year())
}

/// [`fn:seconds-from-dateTime`](https://www.w3.org/TR/xpath-functions-31/#func-seconds-from-dateTime).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:dateTime`.
pub fn seconds_from_date_time(value: &AtomicValue) -> FnResult<Decimal> {
    Ok(as_date_time(value, "fn:seconds-from-dateTime")?.second())
}

/// [`fn:timezone-from-dateTime`](https://www.w3.org/TR/xpath-functions-31/#func-timezone-from-dateTime).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:dateTime`.
pub fn timezone_from_date_time(value: &AtomicValue) -> FnResult<Option<DayTimeDuration>> {
    Ok(as_date_time(value, "fn:timezone-from-dateTime")?.timezone())
}

/// [`fn:year-from-date`](https://www.w3.org/TR/xpath-functions-31/#func-year-from-date).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:date`.
pub fn year_from_date(value: &AtomicValue) -> FnResult<i64> {
    Ok(as_date(value, "fn:year-from-date")?.year())
}

/// [`fn:month-from-date`](https://www.w3.org/TR/xpath-functions-31/#func-month-from-date).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:date`.
pub fn month_from_date(value: &AtomicValue) -> FnResult<i64> {
    Ok(i64::from(as_date(value, "fn:month-from-date")?.month()))
}

/// [`fn:day-from-date`](https://www.w3.org/TR/xpath-functions-31/#func-day-from-date).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:date`.
pub fn day_from_date(value: &AtomicValue) -> FnResult<i64> {
    Ok(i64::from(as_date(value, "fn:day-from-date")?.day()))
}

/// [`fn:timezone-from-date`](https://www.w3.org/TR/xpath-functions-31/#func-timezone-from-date).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:date`.
pub fn timezone_from_date(value: &AtomicValue) -> FnResult<Option<DayTimeDuration>> {
    Ok(as_date(value, "fn:timezone-from-date")?.timezone())
}

/// [`fn:hours-from-time`](https://www.w3.org/TR/xpath-functions-31/#func-hours-from-time).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:time`.
pub fn hours_from_time(value: &AtomicValue) -> FnResult<i64> {
    Ok(i64::from(as_time(value, "fn:hours-from-time")?.hour()))
}

/// [`fn:minutes-from-time`](https://www.w3.org/TR/xpath-functions-31/#func-minutes-from-time).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:time`.
pub fn minutes_from_time(value: &AtomicValue) -> FnResult<i64> {
    Ok(i64::from(as_time(value, "fn:minutes-from-time")?.minute()))
}

/// [`fn:seconds-from-time`](https://www.w3.org/TR/xpath-functions-31/#func-seconds-from-time).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:time`.
pub fn seconds_from_time(value: &AtomicValue) -> FnResult<Decimal> {
    Ok(as_time(value, "fn:seconds-from-time")?.second())
}

/// [`fn:timezone-from-time`](https://www.w3.org/TR/xpath-functions-31/#func-timezone-from-time).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the argument is not an `xs:time`.
pub fn timezone_from_time(value: &AtomicValue) -> FnResult<Option<DayTimeDuration>> {
    Ok(as_time(value, "fn:timezone-from-time")?.timezone())
}

/// [`fn:current-dateTime`](https://www.w3.org/TR/xpath-functions-31/#func-current-dateTime),
/// which is stable within one evaluation because the context captures it once.
#[must_use]
pub fn current_date_time(context: &FnContext) -> AtomicValue {
    AtomicValue::date_time(context.current_date_time())
}

/// [`fn:current-date`](https://www.w3.org/TR/xpath-functions-31/#func-current-date).
///
/// # Errors
///
/// [`FnErrorCode::Fodt0001`] when the current instant cannot be narrowed to a date.
pub fn current_date(context: &FnContext) -> FnResult<AtomicValue> {
    let date = Date::try_from(context.current_date_time()).map_err(|error| {
        FnError::new(FnErrorCode::Fodt0001, format!("fn:current-date: {error}"))
    })?;
    Ok(AtomicValue::date(date))
}

/// [`fn:current-time`](https://www.w3.org/TR/xpath-functions-31/#func-current-time).
#[must_use]
pub fn current_time(context: &FnContext) -> AtomicValue {
    AtomicValue::time(Time::from(context.current_date_time()))
}

/// [`fn:implicit-timezone`](https://www.w3.org/TR/xpath-functions-31/#func-implicit-timezone).
#[must_use]
pub fn implicit_timezone(context: &FnContext) -> AtomicValue {
    AtomicValue::day_time_duration(DayTimeDuration::from(context.implicit_timezone()))
}

/// What an `$timezone` argument to the three `fn:adjust-*-to-timezone` functions asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimezoneArgument {
    /// The one-argument form: adjust to the context's implicit timezone.
    Implicit,
    /// The two-argument form with the empty sequence: strip the timezone.
    Remove,
    /// The two-argument form with a duration: adjust to that offset.
    Offset(TimezoneOffset),
}

impl TimezoneArgument {
    /// Interprets a `$timezone` argument supplied as an optional `xs:dayTimeDuration`.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fodt0003`] when the duration is not a whole number of minutes within
    /// ±14:00, and [`FnErrorCode::Xpty0004`] when it is not an `xs:dayTimeDuration`.
    pub fn from_optional_duration(duration: Option<&AtomicValue>) -> FnResult<Self> {
        let Some(duration) = duration else {
            return Ok(Self::Remove);
        };
        let AtomicData::DayTimeDuration(duration) = duration.data() else {
            return Err(wrong_type(
                duration,
                "an xs:dayTimeDuration",
                "the $timezone argument",
            ));
        };
        let offset = TimezoneOffset::try_from(*duration).map_err(|error| {
            FnError::new(FnErrorCode::Fodt0003, format!("invalid timezone: {error}"))
        })?;
        Ok(Self::Offset(offset))
    }

    /// The offset to adjust to, resolving [`TimezoneArgument::Implicit`] against `context`.
    #[must_use]
    pub fn resolve(self, context: &FnContext) -> Option<TimezoneOffset> {
        match self {
            Self::Implicit => Some(context.implicit_timezone()),
            Self::Remove => None,
            Self::Offset(offset) => Some(offset),
        }
    }
}

fn adjust_overflow(what: &str) -> FnError {
    FnError::new(
        FnErrorCode::Fodt0001,
        format!("{what} overflowed while adjusting the timezone"),
    )
}

/// [`fn:adjust-dateTime-to-timezone`](https://www.w3.org/TR/xpath-functions-31/#func-adjust-dateTime-to-timezone).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-`xs:dateTime` argument, [`FnErrorCode::Fodt0001`] on
/// overflow.
///
/// # Examples
///
/// ```
/// use oxixml_fn::datetime::{adjust_date_time_to_timezone, TimezoneArgument};
/// use oxixml_fn::FnContext;
/// use oxixml_infoset::{AtomicType, AtomicValue};
/// use oxixml_xsd::TimezoneOffset;
///
/// let context = FnContext::new().with_implicit_timezone(TimezoneOffset::new(-300)?);
/// let value = AtomicValue::parse(AtomicType::DateTime, "2002-03-07T10:00:00")?;
///
/// // The one-argument form attaches the implicit timezone without shifting the clock.
/// assert_eq!(
///     adjust_date_time_to_timezone(&context, &value, TimezoneArgument::Implicit)?.string_value(),
///     "2002-03-07T10:00:00-05:00"
/// );
///
/// // Adjusting a zoned value to another zone shifts it.
/// let zoned = AtomicValue::parse(AtomicType::DateTime, "2002-03-07T10:00:00-07:00")?;
/// assert_eq!(
///     adjust_date_time_to_timezone(&context, &zoned, TimezoneArgument::Implicit)?.string_value(),
///     "2002-03-07T12:00:00-05:00"
/// );
///
/// // The empty $timezone strips the timezone.
/// assert_eq!(
///     adjust_date_time_to_timezone(&context, &zoned, TimezoneArgument::Remove)?.string_value(),
///     "2002-03-07T10:00:00"
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn adjust_date_time_to_timezone(
    context: &FnContext,
    value: &AtomicValue,
    timezone: TimezoneArgument,
) -> FnResult<AtomicValue> {
    let value = as_date_time(value, "fn:adjust-dateTime-to-timezone")?;
    let adjusted = value
        .adjust(timezone.resolve(context))
        .ok_or_else(|| adjust_overflow("fn:adjust-dateTime-to-timezone"))?;
    Ok(AtomicValue::date_time(adjusted))
}

/// [`fn:adjust-date-to-timezone`](https://www.w3.org/TR/xpath-functions-31/#func-adjust-date-to-timezone).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-`xs:date` argument, [`FnErrorCode::Fodt0001`] on overflow.
pub fn adjust_date_to_timezone(
    context: &FnContext,
    value: &AtomicValue,
    timezone: TimezoneArgument,
) -> FnResult<AtomicValue> {
    let value = as_date(value, "fn:adjust-date-to-timezone")?;
    let adjusted = value
        .adjust(timezone.resolve(context))
        .ok_or_else(|| adjust_overflow("fn:adjust-date-to-timezone"))?;
    Ok(AtomicValue::date(adjusted))
}

/// [`fn:adjust-time-to-timezone`](https://www.w3.org/TR/xpath-functions-31/#func-adjust-time-to-timezone).
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for a non-`xs:time` argument, [`FnErrorCode::Fodt0001`] on overflow.
pub fn adjust_time_to_timezone(
    context: &FnContext,
    value: &AtomicValue,
    timezone: TimezoneArgument,
) -> FnResult<AtomicValue> {
    let value = as_time(value, "fn:adjust-time-to-timezone")?;
    let adjusted = value
        .adjust(timezone.resolve(context))
        .ok_or_else(|| adjust_overflow("fn:adjust-time-to-timezone"))?;
    Ok(AtomicValue::time(adjusted))
}

/// [`fn:dateTime`](https://www.w3.org/TR/xpath-functions-31/#func-dateTime): builds an
/// `xs:dateTime` from an `xs:date` and an `xs:time`.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] for the wrong argument types, [`FnErrorCode::Forg0008`] when both
/// carry a timezone and the two differ.
pub fn date_time(date: &AtomicValue, time: &AtomicValue) -> FnResult<AtomicValue> {
    let date = as_date(date, "fn:dateTime")?;
    let time = as_time(time, "fn:dateTime")?;
    Ok(AtomicValue::date_time(
        crate::numeric::date_time_from_parts(date, time)?,
    ))
}

/// The `xs:dayTimeDuration` an optional timezone offset corresponds to, which is what the
/// `fn:timezone-from-*` functions return.
#[must_use]
pub fn timezone_value(timezone: Option<DayTimeDuration>) -> Option<AtomicValue> {
    timezone.map(AtomicValue::day_time_duration)
}

/// The `xs:integer` wrapper the integer-valued component accessors' results are given.
#[must_use]
pub fn integer_value(value: i64) -> AtomicValue {
    AtomicValue::integer(value)
}

/// The `xs:decimal` wrapper the seconds accessors' results are given.
#[must_use]
pub fn decimal_value(value: Decimal) -> AtomicValue {
    AtomicValue::new(AtomicType::Decimal, AtomicData::Decimal(value))
        .unwrap_or_else(|_| AtomicValue::decimal(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn parse(atomic_type: AtomicType, lexical: &str) -> AtomicValue {
        AtomicValue::parse(atomic_type, lexical).expect("a valid lexical form")
    }

    #[test]
    fn duration_components_from_the_specification() -> FnResult<()> {
        let positive = parse(AtomicType::YearMonthDuration, "P20Y15M");
        assert_eq!(years_from_duration(&positive)?, 21);
        assert_eq!(months_from_duration(&positive)?, 3);

        let negative = parse(AtomicType::YearMonthDuration, "-P15M");
        assert_eq!(years_from_duration(&negative)?, -1);
        assert_eq!(months_from_duration(&negative)?, -3);

        let day_time = parse(AtomicType::DayTimeDuration, "P3DT10H");
        assert_eq!(days_from_duration(&day_time)?, 3);
        assert_eq!(hours_from_duration(&day_time)?, 10);
        assert_eq!(days_from_duration(&positive)?, 0);

        let full = parse(AtomicType::DayTimeDuration, "P3DT10H12.5S");
        assert_eq!(seconds_from_duration(&full)?.to_string(), "12.5");
        assert_eq!(minutes_from_duration(&full)?, 0);
        Ok(())
    }

    #[test]
    fn date_time_components_from_the_specification() -> FnResult<()> {
        let value = parse(AtomicType::DateTime, "1999-05-31T13:20:00-05:00");
        assert_eq!(year_from_date_time(&value)?, 1999);
        assert_eq!(month_from_date_time(&value)?, 5);
        assert_eq!(day_from_date_time(&value)?, 31);
        assert_eq!(hours_from_date_time(&value)?, 13);
        assert_eq!(minutes_from_date_time(&value)?, 20);
        assert_eq!(seconds_from_date_time(&value)?.to_string(), "0");
        assert_eq!(
            timezone_from_date_time(&value)?.map(|zone| zone.to_string()),
            Some("-PT5H".to_owned())
        );

        let without = parse(AtomicType::DateTime, "1999-05-31T13:20:00");
        assert_eq!(timezone_from_date_time(&without)?, None);
        Ok(())
    }

    #[test]
    fn date_and_time_components() -> FnResult<()> {
        let date = parse(AtomicType::Date, "1999-05-31-05:00");
        assert_eq!(year_from_date(&date)?, 1999);
        assert_eq!(month_from_date(&date)?, 5);
        assert_eq!(day_from_date(&date)?, 31);
        assert!(timezone_from_date(&date)?.is_some());

        let time = parse(AtomicType::Time, "13:20:10.5");
        assert_eq!(hours_from_time(&time)?, 13);
        assert_eq!(minutes_from_time(&time)?, 20);
        assert_eq!(seconds_from_time(&time)?.to_string(), "10.5");
        assert_eq!(timezone_from_time(&time)?, None);
        Ok(())
    }

    #[test]
    fn extractors_reject_the_wrong_type() {
        let value = AtomicValue::string("2020-01-01");
        assert!(year_from_date_time(&value).is_err());
        assert!(days_from_duration(&value).is_err());
    }

    #[test]
    fn adjust_examples_from_the_specification() -> Result<(), Box<dyn std::error::Error>> {
        let context = FnContext::new().with_implicit_timezone(TimezoneOffset::new(-300)?);
        let naive = parse(AtomicType::DateTime, "2002-03-07T10:00:00");
        assert_eq!(
            adjust_date_time_to_timezone(&context, &naive, TimezoneArgument::Implicit)?
                .string_value(),
            "2002-03-07T10:00:00-05:00"
        );

        let zoned = parse(AtomicType::DateTime, "2002-03-07T10:00:00-07:00");
        assert_eq!(
            adjust_date_time_to_timezone(&context, &zoned, TimezoneArgument::Implicit)?
                .string_value(),
            "2002-03-07T12:00:00-05:00"
        );
        assert_eq!(
            adjust_date_time_to_timezone(&context, &zoned, TimezoneArgument::Remove)?
                .string_value(),
            "2002-03-07T10:00:00"
        );

        let ten_hours = TimezoneArgument::Offset(TimezoneOffset::new(600)?);
        assert_eq!(
            adjust_date_time_to_timezone(&context, &zoned, ten_hours)?.string_value(),
            "2002-03-08T03:00:00+10:00"
        );
        Ok(())
    }

    #[test]
    fn adjust_date_and_time() -> Result<(), Box<dyn std::error::Error>> {
        let context = FnContext::new().with_implicit_timezone(TimezoneOffset::new(-300)?);
        let date = parse(AtomicType::Date, "2002-03-07");
        assert_eq!(
            adjust_date_to_timezone(&context, &date, TimezoneArgument::Implicit)?.string_value(),
            "2002-03-07-05:00"
        );

        let time = parse(AtomicType::Time, "10:00:00");
        assert_eq!(
            adjust_time_to_timezone(&context, &time, TimezoneArgument::Implicit)?.string_value(),
            "10:00:00-05:00"
        );
        let zoned_time = parse(AtomicType::Time, "10:00:00-07:00");
        assert_eq!(
            adjust_time_to_timezone(
                &context,
                &zoned_time,
                TimezoneArgument::Offset(TimezoneOffset::new(600)?)
            )?
            .string_value(),
            "03:00:00+10:00"
        );
        Ok(())
    }

    #[test]
    fn the_timezone_argument_is_validated() -> Result<(), Box<dyn std::error::Error>> {
        let too_large = AtomicValue::day_time_duration(DayTimeDuration::from_str("PT15H")?);
        let error =
            TimezoneArgument::from_optional_duration(Some(&too_large)).expect_err("beyond ±14:00");
        assert!(error.is(FnErrorCode::Fodt0003));

        let five_hours = AtomicValue::day_time_duration(DayTimeDuration::from_str("-PT5H")?);
        assert_eq!(
            TimezoneArgument::from_optional_duration(Some(&five_hours))?,
            TimezoneArgument::Offset(TimezoneOffset::new(-300)?)
        );
        assert_eq!(
            TimezoneArgument::from_optional_duration(None)?,
            TimezoneArgument::Remove
        );
        Ok(())
    }

    #[test]
    fn the_current_instant_comes_from_the_context() -> Result<(), Box<dyn std::error::Error>> {
        let pinned = DateTime::from_str("2004-05-12T18:17:15.125Z")?;
        let context = FnContext::new().with_current_date_time(pinned);
        assert_eq!(
            current_date_time(&context).string_value(),
            "2004-05-12T18:17:15.125Z"
        );
        assert_eq!(current_date(&context)?.string_value(), "2004-05-12Z");
        assert_eq!(current_time(&context).string_value(), "18:17:15.125Z");
        assert_eq!(implicit_timezone(&context).string_value(), "PT0S");
        Ok(())
    }

    #[test]
    fn fn_date_time_combines_a_date_and_a_time() -> FnResult<()> {
        let combined = date_time(
            &parse(AtomicType::Date, "1999-12-31"),
            &parse(AtomicType::Time, "12:00:00"),
        )?;
        assert_eq!(combined.string_value(), "1999-12-31T12:00:00");
        Ok(())
    }
}
