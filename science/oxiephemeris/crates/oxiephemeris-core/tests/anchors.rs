//! Table-driven anchor tests for the calendar ↔ JD conversions.
//!
//! Anchor values from standard references (Explanatory Supplement 3rd ed.,
//! ch. 15; USNO MJD definition): they must hold **exactly** (bit-for-bit).

use oxiephemeris_core::time::{is_leap_year, julday, revjul, Calendar, JulianDate};
use oxiephemeris_core::CoreError;

/// (calendar, year, month, day, hours, expected JD)
const ANCHORS: &[(Calendar, i32, u8, u8, f64, f64)] = &[
    // Origin of the Julian Date scale: Julian −4712-01-01 12:00.
    (Calendar::Julian, -4712, 1, 1, 12.0, 0.0),
    // J2000.0: Gregorian 2000-01-01 12:00.
    (Calendar::Gregorian, 2000, 1, 1, 12.0, 2_451_545.0),
    // First day of the Gregorian reform.
    (Calendar::Gregorian, 1582, 10, 15, 0.0, 2_299_160.5),
    // Last Julian day before the reform (continuity: Oct 4 -> Oct 15).
    (Calendar::Julian, 1582, 10, 4, 0.0, 2_299_159.5),
    // MJD epoch: Gregorian 1858-11-17 00:00 (MJD 0).
    (Calendar::Gregorian, 1858, 11, 17, 0.0, 2_400_000.5),
];

#[test]
fn anchors_forward_exact() -> Result<(), CoreError> {
    for &(cal, year, month, day, hours, jd) in ANCHORS {
        let got = julday(cal, year, month, day, hours)?;
        assert_eq!(
            got.value().to_bits(),
            jd.to_bits(),
            "julday({cal:?}, {year}, {month}, {day}, {hours}) = {} != {jd}",
            got.value()
        );
    }
    Ok(())
}

#[test]
fn anchors_reverse_exact() -> Result<(), CoreError> {
    for &(cal, year, month, day, hours, jd) in ANCHORS {
        let (date, h) = revjul(JulianDate::from_f64(jd), cal)?;
        assert_eq!(date.year, year, "revjul({jd}, {cal:?}) year");
        assert_eq!(date.month, month, "revjul({jd}, {cal:?}) month");
        assert_eq!(date.day, day, "revjul({jd}, {cal:?}) day");
        assert_eq!(
            h.to_bits(),
            hours.to_bits(),
            "revjul({jd}, {cal:?}) hours {h}"
        );
    }
    Ok(())
}

#[test]
fn reform_continuity_julian_oct_4_plus_one_is_gregorian_oct_15() -> Result<(), CoreError> {
    let last_julian = julday(Calendar::Julian, 1582, 10, 4, 0.0)?;
    let first_gregorian = julday(Calendar::Gregorian, 1582, 10, 15, 0.0)?;
    assert_eq!(
        first_gregorian.diff_days(last_julian).to_bits(),
        1.0f64.to_bits()
    );
    Ok(())
}

#[test]
fn julday_split_convention() -> Result<(), CoreError> {
    // hi = the .5-aligned day boundary, lo = the day fraction.
    let jd = julday(Calendar::Gregorian, 2000, 1, 1, 12.0)?;
    assert_eq!(jd.hi.to_bits(), 2_451_544.5f64.to_bits());
    assert_eq!(jd.lo.to_bits(), 0.5f64.to_bits());
    Ok(())
}

#[test]
fn hours_outside_a_day_fold_into_the_date() -> Result<(), CoreError> {
    // 36h on Jan 1 == 12h on Jan 2; -12h on Jan 1 == 12h on Dec 31.
    let a = julday(Calendar::Gregorian, 2000, 1, 1, 36.0)?;
    let b = julday(Calendar::Gregorian, 2000, 1, 2, 12.0)?;
    assert_eq!(a.hi.to_bits(), b.hi.to_bits());
    assert_eq!(a.lo.to_bits(), b.lo.to_bits());
    let c = julday(Calendar::Gregorian, 2000, 1, 1, -12.0)?;
    let d = julday(Calendar::Gregorian, 1999, 12, 31, 12.0)?;
    assert_eq!(c.hi.to_bits(), d.hi.to_bits());
    assert_eq!(c.lo.to_bits(), d.lo.to_bits());
    Ok(())
}

#[test]
fn year_zero_is_leap_in_both_calendars() -> Result<(), CoreError> {
    // Astronomical year 0 (= 1 BCE) is a leap year in both calendars.
    assert!(is_leap_year(Calendar::Gregorian, 0));
    assert!(is_leap_year(Calendar::Julian, 0));
    julday(Calendar::Gregorian, 0, 2, 29, 0.0)?;
    julday(Calendar::Julian, 0, 2, 29, 0.0)?;
    Ok(())
}

#[test]
fn bce_century_leap_rule_differs_between_calendars() {
    // -1000 fails the Gregorian 100/400 rule but is a Julian leap year.
    assert!(!is_leap_year(Calendar::Gregorian, -1000));
    assert!(is_leap_year(Calendar::Julian, -1000));
    assert_eq!(
        julday(Calendar::Gregorian, -1000, 2, 29, 0.0),
        Err(CoreError::InvalidInput)
    );
    assert!(julday(Calendar::Julian, -1000, 2, 29, 0.0).is_ok());
    // Positive-year spot checks of the same rule.
    assert!(is_leap_year(Calendar::Gregorian, 2000));
    assert!(!is_leap_year(Calendar::Gregorian, 1900));
    assert!(is_leap_year(Calendar::Gregorian, -400));
    assert!(is_leap_year(Calendar::Julian, 1900));
}

#[test]
fn invalid_inputs_are_rejected() {
    assert_eq!(
        julday(Calendar::Gregorian, 2000, 0, 1, 0.0),
        Err(CoreError::InvalidInput)
    );
    assert_eq!(
        julday(Calendar::Gregorian, 2000, 13, 1, 0.0),
        Err(CoreError::InvalidInput)
    );
    assert_eq!(
        julday(Calendar::Gregorian, 2000, 1, 0, 0.0),
        Err(CoreError::InvalidInput)
    );
    assert_eq!(
        julday(Calendar::Gregorian, 2000, 1, 32, 0.0),
        Err(CoreError::InvalidInput)
    );
    assert_eq!(
        julday(Calendar::Gregorian, 1900, 2, 29, 0.0),
        Err(CoreError::InvalidInput)
    );
    assert_eq!(
        julday(Calendar::Gregorian, 2000, 1, 1, f64::NAN),
        Err(CoreError::InvalidInput)
    );
    assert_eq!(
        julday(Calendar::Gregorian, 2000, 1, 1, f64::INFINITY),
        Err(CoreError::InvalidInput)
    );
}

#[test]
fn revjul_range_guards() {
    assert_eq!(
        revjul(JulianDate::from_f64(f64::NAN), Calendar::Gregorian),
        Err(CoreError::InvalidInput)
    );
    assert_eq!(
        revjul(JulianDate::from_f64(2e12), Calendar::Gregorian),
        Err(CoreError::DateOutOfRange)
    );
    assert_eq!(
        revjul(JulianDate::from_f64(-2e12), Calendar::Julian),
        Err(CoreError::DateOutOfRange)
    );
}
