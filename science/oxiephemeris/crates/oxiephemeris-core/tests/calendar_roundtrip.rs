//! Property tests for the calendar ↔ JD round trips (CLAUDE.md step 2).

use oxiephemeris_core::time::{days_in_month, julday, revjul, Calendar, JulianDate};
use oxiephemeris_core::CoreError;
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

/// Adapt a `CoreError` to a proptest failure (no unwrap policy).
fn ok<T>(result: Result<T, CoreError>) -> Result<T, TestCaseError> {
    result.map_err(|e| TestCaseError::fail(format!("unexpected core error: {e:?}")))
}

fn calendar_strategy() -> impl Strategy<Value = Calendar> {
    prop_oneof![Just(Calendar::Gregorian), Just(Calendar::Julian)]
}

proptest! {
    /// julday -> revjul returns (y, m, d) exactly and hours to < 1e-6 h,
    /// for both calendars over [-3000, 3000] CE.
    #[test]
    fn julday_revjul_round_trip(
        cal in calendar_strategy(),
        year in -3000i32..=3000,
        month in 1u8..=12,
        day in 1u8..=31,
        hours in 0.0f64..24.0,
    ) {
        let dim = ok(days_in_month(cal, year, month))?;
        prop_assume!(day <= dim);
        let jd = ok(julday(cal, year, month, day, hours))?;
        let (date, h) = ok(revjul(jd, cal))?;
        prop_assert_eq!(date.year, year);
        prop_assert_eq!(date.month, month);
        prop_assert_eq!(date.day, day);
        prop_assert!((h - hours).abs() < 1e-6, "hours {} != {}", h, hours);
    }

    /// jd -> revjul -> julday returns the same jd (< 1e-9 days) for
    /// jd in [0, 3e6] snapped to quarter days.
    #[test]
    fn jd_revjul_julday_round_trip_quarter_days(
        cal in calendar_strategy(),
        quarter in 0i32..=12_000_000,
    ) {
        let jd = JulianDate::from_f64(f64::from(quarter) * 0.25);
        let (date, hours) = ok(revjul(jd, cal))?;
        let jd2 = ok(julday(cal, date.year, date.month, date.day, hours))?;
        prop_assert!(
            jd2.diff_days(jd).abs() < 1e-9,
            "jd {} -> {:?} {}h -> {}",
            jd.value(),
            date,
            hours,
            jd2.value()
        );
    }

    /// julday of consecutive civil days differs by exactly 1.0.
    #[test]
    fn consecutive_days_differ_by_exactly_one(
        cal in calendar_strategy(),
        year in -3000i32..=3000,
        month in 1u8..=12,
        day in 1u8..=31,
    ) {
        let dim = ok(days_in_month(cal, year, month))?;
        prop_assume!(day <= dim);
        let (next_y, next_m, next_d) = if day < dim {
            (year, month, day + 1)
        } else if month < 12 {
            (year, month + 1, 1)
        } else {
            (year + 1, 1, 1)
        };
        let jd0 = ok(julday(cal, year, month, day, 0.0))?;
        let jd1 = ok(julday(cal, next_y, next_m, next_d, 0.0))?;
        prop_assert_eq!(jd1.diff_days(jd0).to_bits(), 1.0f64.to_bits());
    }
}
