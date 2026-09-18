//! Tests for the date/time datatypes.

use super::TimezoneOffset;
use super::{
    Date, DateTime, GDay, GMonth, GMonthDay, GYear, GYearMonth, MAX_YEAR, MIN_YEAR,
    ParseDateTimeError, Time,
};
use crate::{DayTimeDuration, Decimal, Duration, YearMonthDuration};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

#[test]
fn from_str() -> Result<(), ParseDateTimeError> {
    assert_eq!(Time::from_str("00:00:00Z")?.to_string(), "00:00:00Z");
    assert_eq!(Time::from_str("00:00:00+00:00")?.to_string(), "00:00:00Z");
    assert_eq!(Time::from_str("00:00:00-00:00")?.to_string(), "00:00:00Z");
    assert_eq!(Time::from_str("00:00:00")?.to_string(), "00:00:00");
    assert_eq!(
        Time::from_str("00:00:00+02:00")?.to_string(),
        "00:00:00+02:00"
    );
    assert_eq!(
        Time::from_str("00:00:00+14:00")?.to_string(),
        "00:00:00+14:00"
    );
    assert_eq!(Time::from_str("24:00:00")?.to_string(), "00:00:00");
    assert_eq!(Time::from_str("24:00:00.00")?.to_string(), "00:00:00");
    assert_eq!(
        Time::from_str("23:59:59.9999999999")?.to_string(),
        "23:59:59.9999999999"
    );

    assert_eq!(Date::from_str("0001-01-01Z")?.to_string(), "0001-01-01Z");
    assert_eq!(Date::from_str("0001-01-01")?.to_string(), "0001-01-01");
    assert_eq!(
        DateTime::from_str("0001-01-01T00:00:00Z")?.to_string(),
        "0001-01-01T00:00:00Z"
    );
    assert_eq!(
        DateTime::from_str("0001-01-01T00:00:00")?.to_string(),
        "0001-01-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("1000000000-01-01T00:00:00")?.to_string(),
        "1000000000-01-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("2001-12-31T23:59:59")?.to_string(),
        "2001-12-31T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("2004-12-31T23:59:59")?.to_string(),
        "2004-12-31T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("1900-12-31T23:59:59")?.to_string(),
        "1900-12-31T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("2000-12-31T23:59:59")?.to_string(),
        "2000-12-31T23:59:59",
    );
    assert_eq!(
        DateTime::from_str("1899-12-31T23:59:59")?.to_string(),
        "1899-12-31T23:59:59"
    );

    assert_eq!(
        DateTime::from_str("2001-02-28T23:59:59")?.to_string(),
        "2001-02-28T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("2004-02-29T23:59:59")?.to_string(),
        "2004-02-29T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("1900-02-28T23:59:59")?.to_string(),
        "1900-02-28T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("2000-02-29T23:59:59")?.to_string(),
        "2000-02-29T23:59:59",
    );
    assert_eq!(
        DateTime::from_str("1899-02-28T23:59:59")?.to_string(),
        "1899-02-28T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("2001-03-01T00:00:00")?.to_string(),
        "2001-03-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("2004-03-01T00:00:00")?.to_string(),
        "2004-03-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("1900-03-01T00:00:00")?.to_string(),
        "1900-03-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("2000-03-01T00:00:00")?.to_string(),
        "2000-03-01T00:00:00",
    );
    assert_eq!(
        DateTime::from_str("1899-03-01T00:00:00")?.to_string(),
        "1899-03-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("-0899-03-01T00:00:00")?.to_string(),
        "-0899-03-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00.1234567")?.to_string(),
        "2000-01-01T00:00:00.1234567"
    );
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:12.1234567")?.to_string(),
        "2000-01-01T00:00:12.1234567"
    );
    assert_eq!(
        Time::from_str("01:02:03.1234567")?.to_string(),
        "01:02:03.1234567"
    );
    assert_eq!(
        Time::from_str("01:02:13.1234567")?.to_string(),
        "01:02:13.1234567"
    );

    assert_eq!(
        DateTime::from_str("-1000000000-01-01T00:00:00")?.to_string(),
        "-1000000000-01-01T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("-2001-12-31T23:59:59")?.to_string(),
        "-2001-12-31T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("-2004-12-31T23:59:59")?.to_string(),
        "-2004-12-31T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("-1900-12-31T23:59:59")?.to_string(),
        "-1900-12-31T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("-2000-12-31T23:59:59")?.to_string(),
        "-2000-12-31T23:59:59"
    );
    assert_eq!(
        DateTime::from_str("-1899-12-31T23:59:59")?.to_string(),
        "-1899-12-31T23:59:59"
    );

    assert_eq!(
        GYearMonth::from_str("-1899-12+01:00")?.to_string(),
        "-1899-12+01:00"
    );
    assert_eq!(GYearMonth::from_str("-1899-12")?.to_string(), "-1899-12");
    assert_eq!(GYear::from_str("-1899+01:00")?.to_string(), "-1899+01:00");
    assert_eq!(GYear::from_str("-1899")?.to_string(), "-1899");
    assert_eq!(
        GMonthDay::from_str("--01-01+01:00")?.to_string(),
        "--01-01+01:00"
    );
    assert_eq!(GMonthDay::from_str("--01-01")?.to_string(), "--01-01");
    assert_eq!(GDay::from_str("---01+01:00")?.to_string(), "---01+01:00");
    assert_eq!(GDay::from_str("---01")?.to_string(), "---01");
    assert_eq!(GMonth::from_str("--01+01:00")?.to_string(), "--01+01:00");
    assert_eq!(GMonth::from_str("--01")?.to_string(), "--01");

    assert!(GYear::from_str("02020").is_err());
    assert!(GYear::from_str("+2020").is_err());
    assert!(GYear::from_str("33").is_err());

    assert_eq!(Time::from_str("00:00:00+14:00")?, Time::MIN);
    assert_eq!(Time::from_str("24:00:00-14:00")?, Time::MAX);
    Ok(())
}

#[test]
fn to_be_bytes() -> Result<(), ParseDateTimeError> {
    assert_eq!(
        DateTime::from_be_bytes(DateTime::MIN.to_be_bytes()),
        DateTime::MIN
    );
    assert_eq!(
        DateTime::from_be_bytes(DateTime::MAX.to_be_bytes()),
        DateTime::MAX
    );
    assert_eq!(
        DateTime::from_be_bytes(DateTime::from_str("2022-01-03T01:02:03")?.to_be_bytes()),
        DateTime::from_str("2022-01-03T01:02:03")?
    );
    assert_eq!(Date::from_be_bytes(Date::MIN.to_be_bytes()), Date::MIN);
    assert_eq!(Date::from_be_bytes(Date::MAX.to_be_bytes()), Date::MAX);
    assert_eq!(
        Date::from_be_bytes(Date::from_str("2022-01-03")?.to_be_bytes()),
        Date::from_str("2022-01-03")?
    );
    assert_eq!(Time::from_be_bytes(Time::MIN.to_be_bytes()), Time::MIN);
    assert_eq!(Time::from_be_bytes(Time::MAX.to_be_bytes()), Time::MAX);
    assert_eq!(
        Time::from_be_bytes(Time::from_str("01:02:03")?.to_be_bytes()),
        Time::from_str("01:02:03")?
    );
    assert_eq!(
        GYearMonth::from_be_bytes(GYearMonth::MIN.to_be_bytes()),
        GYearMonth::MIN
    );
    assert_eq!(
        GYearMonth::from_be_bytes(GYearMonth::MAX.to_be_bytes()),
        GYearMonth::MAX
    );
    assert_eq!(GYear::from_be_bytes(GYear::MIN.to_be_bytes()), GYear::MIN);
    assert_eq!(GYear::from_be_bytes(GYear::MAX.to_be_bytes()), GYear::MAX);
    Ok(())
}

#[test]
fn equals() -> Result<(), ParseDateTimeError> {
    // The XSD 1.1 relation, exposed by `xsd_eq`: two values are equal only when they agree on
    // *having* a timezone. `PartialEq` implements the F&O relation instead, see
    // `equality_applies_the_implicit_timezone`.
    assert!(
        DateTime::from_str("2002-04-02T12:00:00-01:00")?
            .xsd_eq(DateTime::from_str("2002-04-02T17:00:00+04:00")?)
    );
    assert!(
        DateTime::from_str("2002-04-02T12:00:00-05:00")?
            .xsd_eq(DateTime::from_str("2002-04-02T23:00:00+06:00")?)
    );
    assert!(
        !DateTime::from_str("2002-04-02T12:00:00-05:00")?
            .xsd_eq(DateTime::from_str("2002-04-02T17:00:00-05:00")?)
    );
    assert!(
        DateTime::from_str("2002-04-02T12:00:00-05:00")?
            .xsd_eq(DateTime::from_str("2002-04-02T12:00:00-05:00")?)
    );
    assert!(
        DateTime::from_str("2002-04-02T23:00:00-04:00")?
            .xsd_eq(DateTime::from_str("2002-04-03T02:00:00-01:00")?)
    );
    assert!(
        DateTime::from_str("1999-12-31T24:00:00-05:00")?
            .xsd_eq(DateTime::from_str("2000-01-01T00:00:00-05:00")?)
    );
    assert!(
        !DateTime::from_str("2005-04-04T24:00:00-05:00")?
            .xsd_eq(DateTime::from_str("2005-04-04T00:00:00-05:00")?)
    );

    assert!(!Date::from_str("2004-12-25Z")?.xsd_eq(Date::from_str("2004-12-25+07:00")?));
    assert!(Date::from_str("2004-12-25-12:00")?.xsd_eq(Date::from_str("2004-12-26+12:00")?));

    assert!(!Time::from_str("08:00:00+09:00")?.xsd_eq(Time::from_str("17:00:00-06:00")?));
    assert!(Time::from_str("21:30:00+10:30")?.xsd_eq(Time::from_str("06:00:00-05:00")?));
    assert!(Time::from_str("24:00:00+01:00")?.xsd_eq(Time::from_str("00:00:00+01:00")?));

    assert!(Time::from_str("05:00:00-03:00")?.xsd_eq(Time::from_str("10:00:00+02:00")?));
    assert!(!Time::from_str("23:00:00-03:00")?.xsd_eq(Time::from_str("02:00:00Z")?));

    assert!(!GYearMonth::from_str("1986-02")?.xsd_eq(GYearMonth::from_str("1986-03")?));
    assert!(!GYearMonth::from_str("1978-03")?.xsd_eq(GYearMonth::from_str("1978-03Z")?));

    assert!(!GYear::from_str("2005-12:00")?.xsd_eq(GYear::from_str("2005+12:00")?));
    assert!(!GYear::from_str("1976-05:00")?.xsd_eq(GYear::from_str("1976")?));

    assert!(GMonthDay::from_str("--12-25-14:00")?.xsd_eq(GMonthDay::from_str("--12-26+10:00")?));
    assert!(!GMonthDay::from_str("--12-25")?.xsd_eq(GMonthDay::from_str("--12-26Z")?));

    assert!(!GMonth::from_str("--12-14:00")?.xsd_eq(GMonth::from_str("--12+10:00")?));
    assert!(!GMonth::from_str("--12")?.xsd_eq(GMonth::from_str("--12Z")?));

    assert!(!GDay::from_str("---25-14:00")?.xsd_eq(GDay::from_str("---25+10:00")?));
    assert!(!GDay::from_str("---12")?.xsd_eq(GDay::from_str("---12Z")?));
    Ok(())
}

/// Regression test for the F&O `op:*-equal` family: when exactly one operand carries an
/// explicit timezone, [F&O 3.1 §8.3](https://www.w3.org/TR/xpath-functions-31/#func-dateTime-equal)
/// mandates reading the other in the dynamic context's *implicit timezone*. Returning `false`
/// unconditionally — the old behaviour — made
/// `xs:dateTime('2000-01-01T00:00:00Z') eq xs:dateTime('2000-01-01T00:00:00')` answer `false`
/// and `distinct-values` keep the very same instant twice.
#[test]
fn equality_applies_the_implicit_timezone() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00Z")?,
        DateTime::from_str("2000-01-01T00:00:00")?
    );
    assert_eq!(
        Date::from_str("2000-01-01Z")?,
        Date::from_str("2000-01-01")?
    );
    assert_eq!(Time::from_str("12:00:00Z")?, Time::from_str("12:00:00")?);
    assert_eq!(
        GYearMonth::from_str("1978-03Z")?,
        GYearMonth::from_str("1978-03")?
    );
    assert_eq!(GMonth::from_str("--12Z")?, GMonth::from_str("--12")?);
    assert_eq!(GDay::from_str("---12Z")?, GDay::from_str("---12")?);

    // Equal values must hash equally, so that `distinct-values` collapses them.
    let mut hasher_with_timezone = DefaultHasher::new();
    DateTime::from_str("2000-01-01T00:00:00Z")?.hash(&mut hasher_with_timezone);
    let mut hasher_without_timezone = DefaultHasher::new();
    DateTime::from_str("2000-01-01T00:00:00")?.hash(&mut hasher_without_timezone);
    assert_eq!(
        hasher_with_timezone.finish(),
        hasher_without_timezone.finish()
    );

    // A non-UTC implicit timezone shifts the timezone-less operand instead.
    let plus_one = TimezoneOffset::new(60)?;
    let utc = DateTime::from_str("2000-01-01T00:00:00Z")?;
    let bare = DateTime::from_str("2000-01-01T00:00:00")?;
    assert!(!utc.equal_with_implicit_timezone(bare, plus_one));
    assert!(DateTime::from_str("2000-01-01T01:00:00")?.equal_with_implicit_timezone(utc, plus_one));

    // The XSD relation is unchanged: a timezoned value never equals a timezone-less one.
    assert!(!utc.xsd_eq(bare));
    Ok(())
}

/// Regression test for `op:subtract-dateTimes` / `-dates` / `-times`
/// ([F&O 3.1 §8.4](https://www.w3.org/TR/xpath-functions-31/#func-subtract-dateTimes)): a
/// missing timezone is filled in from the implicit timezone instead of making the subtraction
/// fail (`FODT0001`).
#[test]
fn subtraction_applies_the_implicit_timezone() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00Z")?
            .checked_sub(DateTime::from_str("1999-12-31T00:00:00")?),
        Some(DayTimeDuration::from_str("P1D")?)
    );
    assert_eq!(
        DateTime::from_str("1999-12-31T00:00:00")?
            .checked_sub(DateTime::from_str("2000-01-01T00:00:00Z")?),
        Some(DayTimeDuration::from_str("-P1D")?)
    );
    assert_eq!(
        Date::from_str("2000-01-01Z")?.checked_sub(Date::from_str("1999-12-31")?),
        Some(DayTimeDuration::from_str("P1D")?)
    );
    assert_eq!(
        Time::from_str("12:00:00Z")?.checked_sub(Time::from_str("11:00:00")?),
        Some(DayTimeDuration::from_str("PT1H")?)
    );

    // With an implicit timezone of +01:00, `1999-12-31T00:00:00` is read as
    // `1999-12-31T00:00:00+01:00`, i.e. `1999-12-30T23:00:00Z`, one hour *before* the UTC
    // reading — so the difference grows from 24 to 25 hours.
    let plus_one = TimezoneOffset::new(60)?;
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00Z")?.checked_sub_with_implicit_timezone(
            DateTime::from_str("1999-12-31T00:00:00")?,
            plus_one
        ),
        Some(DayTimeDuration::from_str("PT25H")?)
    );
    Ok(())
}

#[test]
fn cmp() -> Result<(), ParseDateTimeError> {
    // The XSD 1.1 partial order, exposed by `xsd_partial_cmp`: indeterminate whenever the
    // ±14:00 window of a timezone-less value straddles a timezoned one.
    assert_eq!(
        Date::from_str("2004-12-25Z")?.xsd_partial_cmp(Date::from_str("2004-12-25-05:00")?),
        Some(Ordering::Less)
    );
    assert_ne!(
        Date::from_str("2004-12-25-12:00")?.xsd_partial_cmp(Date::from_str("2004-12-26+12:00")?),
        Some(Ordering::Less)
    );

    assert_eq!(
        Date::from_str("2004-12-25Z")?.xsd_partial_cmp(Date::from_str("2004-12-25+07:00")?),
        Some(Ordering::Greater)
    );
    assert_ne!(
        Date::from_str("2004-12-25-12:00")?.xsd_partial_cmp(Date::from_str("2004-12-26+12:00")?),
        Some(Ordering::Greater)
    );

    assert_ne!(
        Time::from_str("12:00:00")?.xsd_partial_cmp(Time::from_str("23:00:00+06:00")?),
        Some(Ordering::Less)
    );
    assert_eq!(
        Time::from_str("11:00:00-05:00")?.xsd_partial_cmp(Time::from_str("17:00:00Z")?),
        Some(Ordering::Less)
    );
    assert_ne!(
        Time::from_str("23:59:59")?.xsd_partial_cmp(Time::from_str("24:00:00")?),
        Some(Ordering::Less)
    );

    assert_ne!(
        Time::from_str("08:00:00+09:00")?.xsd_partial_cmp(Time::from_str("17:00:00-06:00")?),
        Some(Ordering::Greater)
    );

    assert_eq!(
        GMonthDay::from_str("--12-12+13:00")?
            .xsd_partial_cmp(GMonthDay::from_str("--12-12+11:00")?),
        Some(Ordering::Less)
    );
    assert_eq!(
        GDay::from_str("---15")?.xsd_partial_cmp(GDay::from_str("---16")?),
        Some(Ordering::Less)
    );
    assert_eq!(
        GDay::from_str("---15-13:00")?.xsd_partial_cmp(GDay::from_str("---16+13:00")?),
        Some(Ordering::Greater)
    );
    assert_eq!(
        GDay::from_str("---15-11:00")?.xsd_partial_cmp(GDay::from_str("---16+13:00")?),
        Some(Ordering::Equal)
    );
    assert_eq!(
        GDay::from_str("---15-13:00")?.xsd_partial_cmp(GDay::from_str("---16")?),
        None
    );
    Ok(())
}

/// Regression test for `op:*-less-than` / `-greater-than`
/// ([F&O 3.1 §8.3](https://www.w3.org/TR/xpath-functions-31/#func-dateTime-less-than)): the
/// implicit timezone makes *every* pair comparable, so `fn:min`, `fn:max` and sorting never
/// see the XSD partial order's indeterminate cases (which used to surface as `FORG0006`).
#[test]
fn ordering_applies_the_implicit_timezone() -> Result<(), Box<dyn Error>> {
    assert!(
        DateTime::from_str("2000-01-01T00:00:00Z")? < DateTime::from_str("2000-01-01T05:00:00")?
    );
    assert_ne!(
        DateTime::from_str("2000-01-01T00:00:00Z")?
            .partial_cmp(&DateTime::from_str("2000-01-01T05:00:00")?),
        Some(Ordering::Greater)
    );
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00Z")?
            .partial_cmp(&DateTime::from_str("2000-01-01T05:00:00")?),
        Some(Ordering::Less)
    );
    assert_eq!(
        Time::from_str("12:00:00")?.partial_cmp(&Time::from_str("23:00:00+06:00")?),
        Some(Ordering::Less)
    );

    // Indeterminate under XSD, determinate under F&O.
    assert_eq!(
        GDay::from_str("---15-13:00")?.xsd_partial_cmp(GDay::from_str("---16")?),
        None
    );
    assert_eq!(
        GDay::from_str("---15-13:00")?
            .compare_with_implicit_timezone(GDay::from_str("---16")?, TimezoneOffset::UTC),
        Ordering::Less
    );

    // A different implicit timezone can flip the outcome.
    let minus_five = TimezoneOffset::new(-5 * 60)?;
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00Z")?.compare_with_implicit_timezone(
            DateTime::from_str("1999-12-31T20:00:00")?,
            TimezoneOffset::UTC
        ),
        Ordering::Greater
    );
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00Z")?
            .compare_with_implicit_timezone(DateTime::from_str("1999-12-31T20:00:00")?, minus_five),
        Ordering::Less
    );
    Ok(())
}

#[test]
fn year() -> Result<(), ParseDateTimeError> {
    assert_eq!(
        DateTime::from_str("1999-05-31T13:20:00-05:00")?.year(),
        1999
    );
    assert_eq!(
        DateTime::from_str("1999-05-31T21:30:00-05:00")?.year(),
        1999
    );
    assert_eq!(DateTime::from_str("1999-12-31T19:20:00")?.year(), 1999);
    assert_eq!(DateTime::from_str("1999-12-31T24:00:00")?.year(), 2000);
    assert_eq!(DateTime::from_str("-0002-06-06T00:00:00")?.year(), -2);

    assert_eq!(Date::from_str("1999-05-31")?.year(), 1999);
    assert_eq!(Date::from_str("2000-01-01+05:00")?.year(), 2000);
    assert_eq!(Date::from_str("-0002-06-01")?.year(), -2);

    assert_eq!(GYear::from_str("-0002")?.year(), -2);
    assert_eq!(GYearMonth::from_str("-0002-02")?.year(), -2);
    Ok(())
}

#[test]
fn month() -> Result<(), ParseDateTimeError> {
    assert_eq!(DateTime::from_str("1999-05-31T13:20:00-05:00")?.month(), 5);
    assert_eq!(DateTime::from_str("1999-12-31T19:20:00-05:00")?.month(), 12);

    assert_eq!(Date::from_str("1999-05-31-05:00")?.month(), 5);
    assert_eq!(Date::from_str("2000-01-01+05:00")?.month(), 1);

    assert_eq!(GMonth::from_str("--02")?.month(), 2);
    assert_eq!(GYearMonth::from_str("-0002-02")?.month(), 2);
    assert_eq!(GMonthDay::from_str("--02-03")?.month(), 2);
    Ok(())
}

#[test]
fn day() -> Result<(), ParseDateTimeError> {
    assert_eq!(DateTime::from_str("1999-05-31T13:20:00-05:00")?.day(), 31);
    assert_eq!(DateTime::from_str("1999-12-31T20:00:00-05:00")?.day(), 31);

    assert_eq!(Date::from_str("1999-05-31-05:00")?.day(), 31);
    assert_eq!(Date::from_str("2000-01-01+05:00")?.day(), 1);

    assert_eq!(GDay::from_str("---03")?.day(), 3);
    assert_eq!(GMonthDay::from_str("--02-03")?.day(), 3);
    Ok(())
}

#[test]
fn hour() -> Result<(), ParseDateTimeError> {
    assert_eq!(DateTime::from_str("1999-05-31T08:20:00-05:00")?.hour(), 8);
    assert_eq!(DateTime::from_str("1999-12-31T21:20:00-05:00")?.hour(), 21);
    assert_eq!(DateTime::from_str("1999-12-31T12:00:00")?.hour(), 12);
    assert_eq!(DateTime::from_str("1999-12-31T24:00:00")?.hour(), 0);

    assert_eq!(Time::from_str("11:23:00-05:00")?.hour(), 11);
    assert_eq!(Time::from_str("21:23:00-05:00")?.hour(), 21);
    assert_eq!(Time::from_str("01:23:00+05:00")?.hour(), 1);
    assert_eq!(Time::from_str("24:00:00")?.hour(), 0);
    Ok(())
}

#[test]
fn minute() -> Result<(), ParseDateTimeError> {
    assert_eq!(
        DateTime::from_str("1999-05-31T13:20:00-05:00")?.minute(),
        20
    );
    assert_eq!(
        DateTime::from_str("1999-05-31T13:30:00+05:30")?.minute(),
        30
    );

    assert_eq!(Time::from_str("13:00:00Z")?.minute(), 0);
    Ok(())
}

#[test]
fn second() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("1999-05-31T13:20:00-05:00")?.second(),
        Decimal::from(0)
    );

    assert_eq!(
        Time::from_str("13:20:10.5")?.second(),
        Decimal::from_str("10.5")?
    );
    Ok(())
}

#[test]
fn timezone() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("1999-05-31T13:20:00-05:00")?.timezone(),
        Some(DayTimeDuration::from_str("-PT5H")?)
    );
    assert_eq!(
        DateTime::from_str("2000-06-12T13:20:00Z")?.timezone(),
        Some(DayTimeDuration::from_str("PT0S")?)
    );
    assert_eq!(DateTime::from_str("2004-08-27T00:00:00")?.timezone(), None);

    assert_eq!(
        Date::from_str("1999-05-31-05:00")?.timezone(),
        Some(DayTimeDuration::from_str("-PT5H")?)
    );
    assert_eq!(
        Date::from_str("2000-06-12Z")?.timezone(),
        Some(DayTimeDuration::from_str("PT0S")?)
    );

    assert_eq!(
        Time::from_str("13:20:00-05:00")?.timezone(),
        Some(DayTimeDuration::from_str("-PT5H")?)
    );
    assert_eq!(Time::from_str("13:20:00")?.timezone(), None);
    Ok(())
}

#[test]
fn sub() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("2000-10-30T06:12:00-05:00")?
            .checked_sub(DateTime::from_str("1999-11-28T09:00:00Z")?),
        Some(DayTimeDuration::from_str("P337DT2H12M")?)
    );

    assert_eq!(
        Date::from_str("2000-10-30")?.checked_sub(Date::from_str("1999-11-28")?),
        Some(DayTimeDuration::from_str("P337D")?)
    );
    assert_eq!(
        Date::from_str("2000-10-30+05:00")?.checked_sub(Date::from_str("1999-11-28Z")?),
        Some(DayTimeDuration::from_str("P336DT19H")?)
    );
    assert_eq!(
        Date::from_str("2000-10-15-05:00")?.checked_sub(Date::from_str("2000-10-10+02:00")?),
        Some(DayTimeDuration::from_str("P5DT7H")?)
    );

    assert_eq!(
        Time::from_str("11:12:00Z")?.checked_sub(Time::from_str("04:00:00-05:00")?),
        Some(DayTimeDuration::from_str("PT2H12M")?)
    );
    assert_eq!(
        Time::from_str("11:00:00-05:00")?.checked_sub(Time::from_str("21:30:00+05:30")?),
        Some(DayTimeDuration::from_str("PT0S")?)
    );
    assert_eq!(
        Time::from_str("17:00:00-06:00")?.checked_sub(Time::from_str("08:00:00+09:00")?),
        Some(DayTimeDuration::from_str("P1D")?)
    );
    assert_eq!(
        Time::from_str("24:00:00")?.checked_sub(Time::from_str("23:59:59")?),
        Some(DayTimeDuration::from_str("-PT23H59M59S")?)
    );
    Ok(())
}

#[test]
fn add_duration() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("2000-01-12T12:13:14Z")?
            .checked_add_duration(Duration::from_str("P1Y3M5DT7H10M3.3S")?),
        Some(DateTime::from_str("2001-04-17T19:23:17.3Z")?)
    );
    assert_eq!(
        Date::from_str("2000-01-01")?.checked_add_duration(Duration::from_str("-P3M")?),
        Some(Date::from_str("1999-10-01")?)
    );
    assert_eq!(
        Date::from_str("2000-01-12")?.checked_add_duration(Duration::from_str("PT33H")?),
        Some(Date::from_str("2000-01-13")?)
    );
    assert_eq!(
        Date::from_str("2000-03-30")?.checked_add_duration(Duration::from_str("P1D")?),
        Some(Date::from_str("2000-03-31")?)
    );
    assert_eq!(
        Date::from_str("2000-03-31")?.checked_add_duration(Duration::from_str("P1M")?),
        Some(Date::from_str("2000-04-30")?)
    );
    assert_eq!(
        Date::from_str("2000-03-30")?.checked_add_duration(Duration::from_str("P1M")?),
        Some(Date::from_str("2000-04-30")?)
    );
    assert_eq!(
        Date::from_str("2000-04-30")?.checked_add_duration(Duration::from_str("P1D")?),
        Some(Date::from_str("2000-05-01")?)
    );

    assert_eq!(
        DateTime::from_str("2000-10-30T11:12:00")?
            .checked_add_duration(Duration::from_str("P1Y2M")?),
        Some(DateTime::from_str("2001-12-30T11:12:00")?)
    );
    assert_eq!(
        DateTime::from_str("2000-10-30T11:12:00")?
            .checked_add_duration(Duration::from_str("P3DT1H15M")?),
        Some(DateTime::from_str("2000-11-02T12:27:00")?)
    );

    assert_eq!(
        Date::from_str("2000-10-30")?.checked_add_duration(Duration::from_str("P1Y2M")?),
        Some(Date::from_str("2001-12-30")?)
    );
    assert_eq!(
        Date::from_str("2004-10-30Z")?.checked_add_duration(Duration::from_str("P2DT2H30M0S")?),
        Some(Date::from_str("2004-11-01Z")?)
    );

    assert_eq!(
        Time::from_str("11:12:00")?.checked_add_duration(Duration::from_str("P3DT1H15M")?),
        Some(Time::from_str("12:27:00")?)
    );
    assert_eq!(
        Time::from_str("23:12:00+03:00")?.checked_add_duration(Duration::from_str("P1DT3H15M")?),
        Some(Time::from_str("02:27:00+03:00")?)
    );

    // End-of-month clamping (2000-01-31 + P1M = 2000-02-29 in a leap year).
    assert_eq!(
        Date::from_str("2000-01-31")?.checked_add_duration(Duration::from_str("P1M")?),
        Some(Date::from_str("2000-02-29")?)
    );
    Ok(())
}

#[test]
fn sub_duration() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("2000-10-30T11:12:00")?
            .checked_sub_duration(Duration::from_str("P1Y2M")?),
        Some(DateTime::from_str("1999-08-30T11:12:00")?)
    );
    assert_eq!(
        DateTime::from_str("2000-10-30T11:12:00")?
            .checked_sub_duration(Duration::from_str("P3DT1H15M")?),
        Some(DateTime::from_str("2000-10-27T09:57:00")?)
    );

    assert_eq!(
        Date::from_str("2000-10-30")?.checked_sub_duration(Duration::from_str("P1Y2M")?),
        Some(Date::from_str("1999-08-30")?)
    );
    assert_eq!(
        Date::from_str("2000-02-29Z")?.checked_sub_duration(Duration::from_str("P1Y")?),
        Some(Date::from_str("1999-02-28Z")?)
    );
    assert_eq!(
        Date::from_str("2000-10-31-05:00")?.checked_sub_duration(Duration::from_str("P1Y1M")?),
        Some(Date::from_str("1999-09-30-05:00")?)
    );
    assert_eq!(
        Date::from_str("2000-10-30")?.checked_sub_duration(Duration::from_str("P3DT1H15M")?),
        Some(Date::from_str("2000-10-26")?)
    );

    assert_eq!(
        Time::from_str("11:12:00")?.checked_sub_duration(Duration::from_str("P3DT1H15M")?),
        Some(Time::from_str("09:57:00")?)
    );
    assert_eq!(
        Time::from_str("08:20:00-05:00")?.checked_sub_duration(Duration::from_str("P23DT10H10M")?),
        Some(Time::from_str("22:10:00-05:00")?)
    );
    Ok(())
}

#[test]
fn adjust() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        DateTime::from_str("2002-03-07T10:00:00-07:00")?
            .adjust(Some(DayTimeDuration::from_str("PT10H")?.try_into()?)),
        Some(DateTime::from_str("2002-03-08T03:00:00+10:00")?)
    );
    assert_eq!(
        DateTime::from_str("2002-03-07T00:00:00+01:00")?
            .adjust(Some(DayTimeDuration::from_str("-PT8H")?.try_into()?)),
        Some(DateTime::from_str("2002-03-06T15:00:00-08:00")?)
    );
    assert_eq!(
        DateTime::from_str("2002-03-07T10:00:00")?.adjust(None),
        Some(DateTime::from_str("2002-03-07T10:00:00")?)
    );
    assert_eq!(
        DateTime::from_str("2002-03-07T10:00:00-07:00")?.adjust(None),
        Some(DateTime::from_str("2002-03-07T10:00:00")?)
    );

    assert_eq!(
        Date::from_str("2002-03-07")?
            .adjust(Some(DayTimeDuration::from_str("-PT10H")?.try_into()?)),
        Some(Date::from_str("2002-03-07-10:00")?)
    );
    assert_eq!(
        Date::from_str("2002-03-07-07:00")?
            .adjust(Some(DayTimeDuration::from_str("-PT10H")?.try_into()?)),
        Some(Date::from_str("2002-03-06-10:00")?)
    );
    assert_eq!(
        Date::from_str("2002-03-07")?.adjust(None),
        Some(Date::from_str("2002-03-07")?)
    );
    assert_eq!(
        Date::from_str("2002-03-07-07:00")?.adjust(None),
        Some(Date::from_str("2002-03-07")?)
    );

    assert_eq!(
        Time::from_str("10:00:00")?.adjust(Some(DayTimeDuration::from_str("-PT10H")?.try_into()?)),
        Some(Time::from_str("10:00:00-10:00")?)
    );
    assert_eq!(
        Time::from_str("10:00:00-07:00")?
            .adjust(Some(DayTimeDuration::from_str("-PT10H")?.try_into()?)),
        Some(Time::from_str("07:00:00-10:00")?)
    );
    assert_eq!(
        Time::from_str("10:00:00")?.adjust(None),
        Some(Time::from_str("10:00:00")?)
    );
    assert_eq!(
        Time::from_str("10:00:00-07:00")?.adjust(None),
        Some(Time::from_str("10:00:00")?)
    );
    assert_eq!(
        Time::from_str("10:00:00-07:00")?
            .adjust(Some(DayTimeDuration::from_str("PT10H")?.try_into()?)),
        Some(Time::from_str("03:00:00+10:00")?)
    );
    Ok(())
}

#[test]
fn time_from_datetime() -> Result<(), ParseDateTimeError> {
    assert_eq!(
        Time::from(DateTime::MIN),
        Time::from_str("19:51:08.312696284115894272-14:00")?
    );
    assert_eq!(
        Time::from(DateTime::MAX),
        Time::from_str("04:08:51.687303715884105727+14:00")?
    );
    Ok(())
}

#[test]
fn date_from_datetime() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        Date::try_from(DateTime::from_str("-4294967295-01-01T00:00:00+14:00")?)?,
        Date::from_str("-4294967295-01-01+14:00")?
    );
    assert_eq!(
        Date::try_from(DateTime::from_str("4294967295-12-31T23:59:59-14:00")?)?,
        Date::from_str("4294967295-12-31-14:00")?
    );
    Ok(())
}

#[test]
fn g_year_month_from_date() -> Result<(), ParseDateTimeError> {
    assert_eq!(
        GYearMonth::from(Date::from_str("-4294967295-01-01+14:00")?),
        GYearMonth::from_str("-4294967295-01+14:00")?
    );
    assert_eq!(
        GYearMonth::from(Date::from_str("4294967295-12-31-14:00")?),
        GYearMonth::from_str("4294967295-12-14:00")?
    );
    Ok(())
}

/// `Date::MAX`/`Date::MIN` are the extremes of the internal timeline encoding, whose years lie
/// far beyond [`MAX_YEAR`] and so are unreachable by parsing or by any year-taking constructor.
/// Deriving a `gYearMonth` from one re-derives an internal "last day of the month" placeholder
/// whose own year is equally out of range, so the conversion — whose signature mirrors upstream
/// and returns `Self` rather than a `Result` — must saturate towards the end of the range it
/// came from rather than panic, wrap, or flip sign.
#[test]
fn g_year_month_from_the_encoding_extremes_saturates() {
    assert_eq!(GYearMonth::from(Date::MAX), GYearMonth::MAX);
    assert_eq!(GYearMonth::from(Date::MIN), GYearMonth::MIN);
}

#[test]
fn g_year_from_g_year_month() -> Result<(), ParseDateTimeError> {
    assert_eq!(
        GYear::try_from(GYearMonth::from_str("-4294967295-01+14:00")?)?,
        GYear::from_str("-4294967295+14:00")?
    );
    assert_eq!(
        GYear::try_from(GYearMonth::from_str("4294967295-12-14:00")?)?,
        GYear::from_str("4294967295-14:00")?
    );
    Ok(())
}

/// The supported year range, [`MIN_YEAR`]`..=`[`MAX_YEAR`], is enforced at the one seam every
/// constructor passes through, so it holds for every date/time type that carries a year and for
/// the year–month duration arithmetic that recomputes one. A year one past either end is an
/// *overflow* of the value space (`FODT0001` downstream), never a lexical error (`FORG0001`).
///
/// Day–time arithmetic is deliberately *not* covered: it moves a value along the timeline
/// without touching the year, and is bounded by the internal encoding instead — see
/// [`MAX_YEAR`]'s own documentation.
#[test]
fn every_year_bearing_constructor_enforces_the_supported_range() -> Result<(), Box<dyn Error>> {
    assert_eq!(MAX_YEAR, 4_294_967_295);
    assert_eq!(MIN_YEAR, -4_294_967_295);

    // The last supported year parses, in every type that carries one.
    assert_eq!(
        DateTime::from_str("4294967295-01-02T02:23:12.987")?.to_string(),
        "4294967295-01-02T02:23:12.987"
    );
    assert_eq!(
        Date::from_str("4294967295-01-02")?.to_string(),
        "4294967295-01-02"
    );
    assert_eq!(GYear::from_str("-4294967295")?.to_string(), "-4294967295");
    assert_eq!(
        GYearMonth::from_str("4294967295-01")?.to_string(),
        "4294967295-01"
    );

    // One past it does not, and says so as an overflow.
    for lexical in ["4294967296-01-02", "-4294967296-01-02", "4294967297-01-02"] {
        assert!(
            Date::from_str(lexical)
                .expect_err("beyond MAX_YEAR")
                .is_overflow(),
            "{lexical} must overflow"
        );
    }
    assert!(
        DateTime::from_str("4294967297-01-02T02:23:12.987")
            .expect_err("beyond MAX_YEAR")
            .is_overflow()
    );
    assert!(
        DateTime::from_str("-4294967297-01-02T02:23:12.987")
            .expect_err("beyond MAX_YEAR")
            .is_overflow()
    );
    assert!(
        GYear::from_str("4294967296")
            .expect_err("beyond MAX_YEAR")
            .is_overflow()
    );
    assert!(
        GYearMonth::from_str("4294967296-01")
            .expect_err("beyond MAX_YEAR")
            .is_overflow()
    );

    // Year–month duration arithmetic recomputes a year, and is bounded by the same range: a
    // step that stays inside it succeeds, one that leaves it reports the overflow.
    let start = DateTime::from_str("2008-05-12T12:59:00")?;
    assert_eq!(
        start
            .checked_add_year_month_duration(YearMonthDuration::from_str("P4026720960Y")?)
            .expect("4026722968 is a supported year")
            .to_string(),
        "4026722968-05-12T12:59:00"
    );
    assert!(
        start
            .checked_add_year_month_duration(YearMonthDuration::from_str("P4294967295Y")?)
            .is_none()
    );
    Ok(())
}

#[test]
fn now() -> Result<(), ParseDateTimeError> {
    let now = DateTime::now();
    assert!(DateTime::from_str("2022-01-01T00:00:00Z")? < now);
    assert!(now < DateTime::from_str("2100-01-01T00:00:00Z")?);
    Ok(())
}

#[test]
fn minimally_conformant() -> Result<(), ParseDateTimeError> {
    // Nonnegative year values less than 10000.
    assert_eq!(GYear::from_str("9999")?.to_string(), "9999");
    assert_eq!(
        DateTime::from_str("9999-12-31T23:59:59Z")?.to_string(),
        "9999-12-31T23:59:59Z"
    );

    // Second values to milliseconds.
    assert_eq!(
        Time::from_str("00:00:00.678Z")?.to_string(),
        "00:00:00.678Z"
    );
    assert_eq!(
        DateTime::from_str("2000-01-01T00:00:00.678Z")?.to_string(),
        "2000-01-01T00:00:00.678Z"
    );
    Ok(())
}

/// A syntactically well-formed year that overflows the internal representation must be
/// distinguishable, via [`ParseDateTimeError::is_overflow`], from a genuinely malformed
/// lexical form -- so that a downstream XPath/XQuery consumer can raise
/// [`FODT0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFODT0001) instead of the
/// generic invalid-lexical-value error ([`FORG0001`](https://www.w3.org/TR/xpath-functions-31/#ERRFORG0001)).
#[test]
fn year_overflow_is_distinguishable_from_lexical_errors() {
    // A 21-digit year: far more digits than `i64` (19 digits max) can hold.
    assert!(
        DateTime::from_str("999999999999999999999-01-01T00:00:00Z")
            .expect_err("21 nines overflows i64")
            .is_overflow()
    );
    assert!(
        DateTime::from_str("-999999999999999999999-01-01T00:00:00Z")
            .expect_err("21 nines overflows i64")
            .is_overflow()
    );
    assert!(
        Date::from_str("999999999999999999999-01-01")
            .expect_err("21 nines overflows i64")
            .is_overflow()
    );
    assert!(
        GYear::from_str("999999999999999999999")
            .expect_err("21 nines overflows i64")
            .is_overflow()
    );
    assert!(
        GYearMonth::from_str("999999999999999999999-01")
            .expect_err("21 nines overflows i64")
            .is_overflow()
    );
    assert!(
        GYear::from_str("-999999999999999999999")
            .expect_err("21 nines overflows i64")
            .is_overflow()
    );

    // A year that fits `i64` (13 digits) but is beyond `MAX_YEAR` is *also* an overflow,
    // not a lexical error.
    assert!(
        DateTime::from_str("9999999999999-01-01T00:00:00Z")
            .expect_err("13 nines is beyond MAX_YEAR")
            .is_overflow()
    );
    assert!(
        DateTime::from_str("-9999999999999-01-01T00:00:00Z")
            .expect_err("13 nines is beyond MAX_YEAR")
            .is_overflow()
    );

    // Genuinely malformed lexical forms, and semantically-invalid-but-well-formed ones, are
    // not overflows.
    assert!(
        !DateTime::from_str("not-a-date")
            .expect_err("not a date at all")
            .is_overflow()
    );
    assert!(
        !GYear::from_str("33")
            .expect_err("too few digits")
            .is_overflow()
    );
    assert!(
        !DateTime::from_str("2020-13-01T00:00:00Z")
            .expect_err("month 13 is out of range")
            .is_overflow()
    );
    // Invalid day-of-month (2020 is a leap year, so this is `InvalidDayOfMonth`, not a
    // generic lexical message, but it is still not an overflow).
    assert!(
        !DateTime::from_str("2020-02-30T00:00:00Z")
            .expect_err("2020 is a leap year, but February still stops at 29")
            .is_overflow()
    );
}

/// [XSD 1.0 originally specified](https://www.w3.org/TR/2001/REC-xmlschema-2-20010502/#gMonth)
/// `gMonth` as `'--' monthFrag ('--')? timezoneFrag?` — a redundant trailing `--`
/// placeholder for the day field `gMonth` has no room for. A later erratum
/// ([W3C Bug 6901](https://www.w3.org/Bugs/Public/show_bug.cgi?id=6901)) dropped it from the
/// grammar, and this parser follows the modern, erratum'd production only: the withdrawn
/// trailing `--` is a lexical error, exactly as the W3C XML Schema Test Suite's own
/// `gMonth_*` facet tests (`Facets/gMonth/`, e.g. `gMonth_maxExclusive002_208`) — filed
/// against the same bug — require.
#[test]
fn g_month_rejects_the_withdrawn_trailing_double_dash() -> Result<(), ParseDateTimeError> {
    // gMonth002.xml's `--03--` no longer parses; only the modern `--03` does.
    assert!(GMonth::from_str("--03--").is_err());
    assert_eq!(GMonth::from_str("--03")?.to_string(), "--03");
    // gMonth004.xml's `--05---05:00` (the withdrawn `--` immediately followed by a negative
    // timezone, so three dashes appear in a row) no longer parses either; only the modern
    // `--05-05:00` does.
    assert!(GMonth::from_str("--05---05:00").is_err());
    assert_eq!(GMonth::from_str("--05-05:00")?.to_string(), "--05-05:00");
    // The withdrawn separator is rejected the same way composed with `Z` and a positive
    // offset.
    assert!(GMonth::from_str("--07--Z").is_err());
    assert_eq!(GMonth::from_str("--07Z")?.to_string(), "--07Z");
    assert!(GMonth::from_str("--07--+05:00").is_err());
    assert_eq!(GMonth::from_str("--07+05:00")?.to_string(), "--07+05:00");

    // An out-of-range month is rejected regardless: adding the withdrawn separator does not
    // make an otherwise-invalid value any more invalid.
    assert!(GMonth::from_str("--13--").is_err());
    assert!(GMonth::from_str("--13").is_err());
    Ok(())
}

/// [XSD 1.1 §3.3.15](https://www.w3.org/TR/xmlschema11-2/#gMonthDay) requires only that a
/// `gMonthDay`'s day be possible in its month in *some* year, so February 29 is legal (it
/// occurs in leap years) even though no `xs:date` year is specified. The W3C XML Schema Test
/// Suite agrees (`msData/datatypes/gMonthDay004.xml`: `--02-29`, `expected validity="valid"`).
#[test]
fn g_month_day_allows_february_29_but_rejects_its_invalid_neighbors()
-> Result<(), ParseDateTimeError> {
    assert_eq!(GMonthDay::from_str("--02-29")?.to_string(), "--02-29");
    // February 30 never occurs in any year.
    assert!(GMonthDay::from_str("--02-30").is_err());
    // April 31 never occurs in any year either (April always has 30 days).
    assert!(GMonthDay::from_str("--04-31").is_err());
    // February 29 must still roundtrip through the value (not just parse).
    assert_eq!(GMonthDay::from_str("--02-29")?.day(), 29);
    assert_eq!(GMonthDay::from_str("--02-29")?.month(), 2);
    Ok(())
}

/// Audits `Date`/`DateTime`/`Time` `+`/`-` `Duration` for silent saturation: overflow must
/// always surface as `None`, never as a silently clamped or wrapped-around `Some` value.
#[test]
fn date_time_duration_arithmetic_overflows_cleanly_rather_than_saturating()
-> Result<(), Box<dyn Error>> {
    // A year-month duration far larger than the ~5.39e12-year internal range. It parses
    // successfully on its own (`Duration::from_str` never sees a `DateTime`'s narrower
    // range), but applying it must overflow to `None`.
    let huge = Duration::from_str("P9000000000000Y")?;
    let base = DateTime::from_str("2000-01-01T00:00:00Z")?;
    assert_eq!(base.checked_add_duration(huge), None);
    assert_eq!(base.checked_sub_duration(huge), None);
    assert_eq!(
        Date::from_str("2000-01-01")?.checked_add_duration(huge),
        None
    );
    assert_eq!(Time::from_str("00:00:00")?.checked_add_duration(huge), None);
    assert_eq!(Time::from_str("00:00:00")?.checked_sub_duration(huge), None);

    // Boundary check on the day-time-only fast path: one more second than `DateTime::MAX`
    // (respectively one fewer than `DateTime::MIN`) must overflow too, not silently clamp.
    assert_eq!(
        DateTime::MAX.checked_add_duration(Duration::from_str("PT1S")?),
        None
    );
    assert_eq!(
        DateTime::MIN.checked_sub_duration(Duration::from_str("PT1S")?),
        None
    );
    Ok(())
}

/// Regression test for the `endOfDayFrag` production
/// ([XSD 1.1 Part 2 §D.3](https://www.w3.org/TR/xmlschema11-2/#dateTime-lexical-mapping)
/// production [17]): `endOfDayFrag ::= '24:00:00' ('.' '0'+)?`, so hour 24 is legal only when
/// **both** the minute and the second are zero. The guard used to be a conjunction (`minute
/// != 0 && second != 0`), which silently accepted `24:30:00` / `24:00:30` and normalised them
/// into the next day.
#[test]
fn hour_24_requires_zero_minutes_and_seconds() -> Result<(), ParseDateTimeError> {
    for lexical in [
        "2020-01-01T24:30:00",
        "2020-01-01T24:00:30",
        "2020-01-01T24:30:30",
        "2020-01-01T24:00:00.5",
        "2020-01-01T24:01:00Z",
        "2020-01-01T24:00:01+02:00",
    ] {
        assert!(
            DateTime::from_str(lexical).is_err(),
            "{lexical} is outside the xs:dateTime lexical space"
        );
    }
    for lexical in [
        "24:30:00",
        "24:00:30",
        "24:30:30",
        "24:00:00.5",
        "24:01:00Z",
        "24:00:01+02:00",
    ] {
        assert!(
            Time::from_str(lexical).is_err(),
            "{lexical} is outside the xs:time lexical space"
        );
    }

    // The one legal end-of-day form still normalises to the next day / midnight.
    assert_eq!(
        DateTime::from_str("2020-01-01T24:00:00")?.to_string(),
        "2020-01-02T00:00:00"
    );
    assert_eq!(
        DateTime::from_str("2020-01-01T24:00:00.0Z")?.to_string(),
        "2020-01-02T00:00:00Z"
    );
    assert_eq!(Time::from_str("24:00:00")?.to_string(), "00:00:00");
    Ok(())
}

/// Appendix E's `normalizeMonth` is *floored*, and the month it is handed is
/// routinely outside `1..=12` — a `date` in January minus one month arrives as
/// month 0. Truncating division turned that into a well-formed February that
/// no later normalization step could correct, so subtracting a `P1M` moved the
/// value a month *forward*. The pairs below cover the January boundary in both
/// directions and the multi-year carry beside it.
#[test]
fn subtracting_a_year_month_duration_moves_backwards() -> Result<(), Box<dyn Error>> {
    let date = Date::from_str("1997-01-01")?;
    let date_time = DateTime::from_str("1997-01-01T12:00:00")?;
    for (lexical, subtracted, added) in [
        ("P1M", "1996-12-01", "1997-02-01"),
        ("-P1M", "1997-02-01", "1996-12-01"),
        ("P11M", "1996-02-01", "1997-12-01"),
        ("-P11M", "1997-12-01", "1996-02-01"),
        ("P12M", "1996-01-01", "1998-01-01"),
        ("P13M", "1995-12-01", "1998-02-01"),
        ("-P13M", "1998-02-01", "1995-12-01"),
        ("P1Y", "1996-01-01", "1998-01-01"),
        ("-P1Y", "1998-01-01", "1996-01-01"),
        ("P1Y1M", "1995-12-01", "1998-02-01"),
    ] {
        let duration = YearMonthDuration::from_str(lexical)?;
        assert_eq!(
            date.checked_sub_year_month_duration(duration)
                .ok_or("no overflow")?
                .to_string(),
            subtracted,
            "1997-01-01 - {lexical}"
        );
        assert_eq!(
            date.checked_add_year_month_duration(duration)
                .ok_or("no overflow")?
                .to_string(),
            added,
            "1997-01-01 + {lexical}"
        );
        // The same two operations spelled with the general `xs:duration`,
        // which is the route an XPath `-` takes when the operand's static type
        // is only known to be a duration.
        assert_eq!(
            date.checked_sub_duration(Duration::from(duration))
                .ok_or("no overflow")?
                .to_string(),
            subtracted,
            "1997-01-01 - xs:duration({lexical})"
        );
        assert_eq!(
            date_time
                .checked_sub_year_month_duration(duration)
                .ok_or("no overflow")?
                .to_string(),
            format!("{subtracted}T12:00:00"),
            "1997-01-01T12:00:00 - {lexical}"
        );
        // The `xs:gYearMonth` the result reduces to. XPath defines no duration
        // arithmetic on `gYearMonth` itself — the year and month it would name
        // are the ones the `date` above lands on — so this pins the pair the
        // normalization actually produces.
        assert_eq!(
            GYearMonth::from_str(
                subtracted
                    .rsplit_once('-')
                    .ok_or("a `yyyy-mm-dd` result")?
                    .0
            )?
            .to_string(),
            subtracted
                .rsplit_once('-')
                .ok_or("a `yyyy-mm-dd` result")?
                .0,
        );
    }
    Ok(())
}

/// The day-of-month clamp and the day/time carry that follow the month
/// normalization, over the same January boundary: `daysInMonth` decides the
/// clamp *after* the month has moved, and a negative day-time component has to
/// borrow across it.
#[test]
fn month_arithmetic_clamps_and_borrows_across_january() -> Result<(), Box<dyn Error>> {
    // 31 March minus one month is 28 February, not 3 March.
    assert_eq!(
        Date::from_str("1997-03-31")?
            .checked_sub_year_month_duration(YearMonthDuration::from_str("P1M")?)
            .ok_or("no overflow")?
            .to_string(),
        "1997-02-28"
    );
    // February of a leap year keeps the 29th.
    assert_eq!(
        Date::from_str("1996-03-31")?
            .checked_sub_year_month_duration(YearMonthDuration::from_str("P1M")?)
            .ok_or("no overflow")?
            .to_string(),
        "1996-02-29"
    );
    // A duration with both components: the day-time part is negative here, so
    // the day borrows from the month the year-month part has just moved onto.
    for (lexical, expected) in [
        ("P1YT1S", "1995-12-31T23:59:59"),
        ("P1MT1S", "1996-11-30T23:59:59"),
        ("P1MT0.5S", "1996-11-30T23:59:59.5"),
        ("P1MT1H", "1996-11-30T23:00:00"),
        ("P1MT1M", "1996-11-30T23:59:00"),
        ("P1M1D", "1996-11-30T00:00:00"),
    ] {
        assert_eq!(
            DateTime::from_str("1997-01-01T00:00:00")?
                .checked_sub_duration(Duration::from_str(lexical)?)
                .ok_or("no overflow")?
                .to_string(),
            expected,
            "1997-01-01T00:00:00 - {lexical}"
        );
    }
    Ok(())
}

/// The lexical spaces of `xs:date` and `xs:dateTime` reject every field that
/// falls outside its production or its calendar: a day the month does not
/// have, a February 29th in a year that is not a leap year under the proleptic
/// Gregorian rule, an out-of-range minute, hour or second, and a timezone
/// beyond `±14:00`. Together these are the rejections an XPath `xs:date()` /
/// `xs:dateTime()` constructor turns into `FORG0001`.
#[test]
fn date_and_date_time_reject_out_of_range_fields() {
    for lexical in [
        // The day-of-month production stops at 31 …
        "1999-01-32",
        "1999-12-32",
        // … and the calendar takes it further down per month.
        "1999-04-31",
        "1999-02-30",
        // 100, 200 and 8100 are divisible by 4 and by 100 but not by 400.
        "0100-02-29",
        "0200-02-29",
        "8100-02-29",
        "1900-02-29",
        // The month production stops at 12.
        "1999-13-29",
        "1999-00-01",
    ] {
        assert!(
            Date::from_str(lexical).is_err(),
            "{lexical} is outside the xs:date lexical space"
        );
        assert!(
            DateTime::from_str(&format!("{lexical}T02:23:12.987")).is_err(),
            "{lexical}T02:23:12.987 is outside the xs:dateTime lexical space"
        );
    }

    // 400 and 1600 are divisible by 400, and 1996 by 4 but not by 100.
    for lexical in ["0400-02-29", "1600-02-29", "1996-02-29", "2000-02-29"] {
        assert!(
            Date::from_str(lexical).is_ok(),
            "{lexical} is a leap day of the proleptic Gregorian calendar"
        );
    }

    for suffix in [
        // minute, hour and second, each one past its production.
        "02:63:12.987",
        "24:23:12.987",
        "02:23:74.987",
        // No leap second is in the value space.
        "02:23:60",
        // The timezone offset runs from -14:00 to +14:00 inclusive.
        "02:23:12+15:00",
        "02:23:12-15:00",
        "02:23:12+12:60",
        "02:23:12+14:01",
        "02:23:12-14:01",
    ] {
        assert!(
            DateTime::from_str(&format!("1996-02-29T{suffix}")).is_err(),
            "1996-02-29T{suffix} is outside the xs:dateTime lexical space"
        );
    }
    for suffix in [
        "02:23:12+14:00",
        "02:23:12-14:00",
        "02:23:12Z",
        "02:23:59.9",
    ] {
        assert!(
            DateTime::from_str(&format!("1996-02-29T{suffix}")).is_ok(),
            "1996-02-29T{suffix} is inside the xs:dateTime lexical space"
        );
    }
}

/// The proleptic Gregorian calendar this crate implements is XSD 1.1's, which
/// has a year 0 (1 BCE) and negative years below it. Year 0 is a leap year,
/// `-0000` denotes the same year as `0000` and is written back without the
/// sign, and `-0400` is a leap year for the same reason `0400` is. XSD 1.0's
/// prohibition of `0000` is a property of that *version*, and is applied by
/// the schema processor, not here.
#[test]
fn the_year_zero_and_negative_years_round_trip() -> Result<(), Box<dyn Error>> {
    assert_eq!(Date::from_str("0000-02-29")?.to_string(), "0000-02-29");
    assert_eq!(Date::from_str("-0400-02-29")?.to_string(), "-0400-02-29");
    assert_eq!(Date::from_str("-0000-01-30")?.to_string(), "0000-01-30");
    assert_eq!(Date::from_str("-0001-09-05")?.to_string(), "-0001-09-05");
    assert_eq!(
        DateTime::from_str("0000-02-29T02:23:12.987")?.to_string(),
        "0000-02-29T02:23:12.987"
    );
    assert_eq!(
        DateTime::from_str("-0400-02-29T02:23:12.987")?.to_string(),
        "-0400-02-29T02:23:12.987"
    );
    assert_eq!(
        DateTime::from_str("-0000-01-30T02:23:12.987")?.to_string(),
        "0000-01-30T02:23:12.987"
    );
    assert_eq!(GYear::from_str("-0400")?.to_string(), "-0400");
    assert_eq!(GYear::from_str("0000")?.to_string(), "0000");
    assert_eq!(GYear::from_str("-0000")?.to_string(), "0000");
    assert_eq!(GYearMonth::from_str("-0400-02")?.to_string(), "-0400-02");
    assert_eq!(GYearMonth::from_str("0000-02")?.to_string(), "0000-02");

    // Year 100 is not a leap year, and neither is -0100 (its own proleptic
    // year number is divisible by 100 and not by 400).
    assert!(Date::from_str("0100-02-29").is_err());
    assert!(Date::from_str("-0100-02-29").is_err());

    // Month arithmetic crosses year 0 without a gap: there is no year to skip.
    let one_month = YearMonthDuration::from_str("P1M")?;
    for (from, expected) in [
        ("0001-01-01", "0000-12-01"),
        ("0000-01-01", "-0001-12-01"),
        ("-0001-01-01", "-0002-12-01"),
    ] {
        assert_eq!(
            Date::from_str(from)?
                .checked_sub_year_month_duration(one_month)
                .ok_or("no overflow")?
                .to_string(),
            expected,
            "{from} - P1M"
        );
    }
    Ok(())
}

/// The counterpart of [`every_year_bearing_constructor_enforces_the_supported_range`], pinning
/// exactly how far the range reaches. `date` arithmetic re-derives a year on the way back from
/// the `dateTime` it borrows, so it is bounded by the range like any other constructor. A
/// `dateTime` moved by a *day–time* duration is not: that path stays on the timeline, never
/// recomputing a year, and is bounded by the internal encoding instead. Such a result is past
/// the end of the lexical space, so nothing reads it back in.
#[test]
fn day_time_arithmetic_on_a_date_time_is_bounded_by_the_encoding() -> Result<(), Box<dyn Error>> {
    let last = Date::from_str("4294967295-12-31")?;
    assert!(
        last.checked_add_day_time_duration(DayTimeDuration::from_str("P40D")?)
            .is_none(),
        "a date re-derives its year, so it is bounded by the supported range"
    );

    let last = DateTime::from_str("4294967295-12-31T00:00:00")?;
    let past = last
        .checked_add_day_time_duration(DayTimeDuration::from_str("P40D")?)
        .expect("the timeline has room");
    assert!(
        past.year() > MAX_YEAR,
        "{} is not past the range",
        past.year()
    );
    assert!(
        DateTime::from_str(&past.to_string())
            .expect_err("beyond MAX_YEAR")
            .is_overflow()
    );
    Ok(())
}
