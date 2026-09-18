//! Leap-second table self-consistency and TAI/TT/UTC conversion tests.

use oxiephemeris_core::time::{
    julday, tai_minus_utc, tai_to_tt, tai_to_utc, tt_to_tai, tt_to_utc, utc_to_tai, utc_to_tt,
    Calendar, JulianDate, LEAP_SECONDS, SECONDS_PER_DAY, TT_MINUS_TAI_SECONDS, UTC_LEAP_START_JD,
};
use oxiephemeris_core::CoreError;

/// Documented calendar date (year, month; always day 1, 0h UTC) of each
/// `LEAP_SECONDS` row, in table order. Update together with the table
/// (see the update procedure in `time::leap`).
const LEAP_DATES: [(i32, u8); 28] = [
    (1972, 1),
    (1972, 7),
    (1973, 1),
    (1974, 1),
    (1975, 1),
    (1976, 1),
    (1977, 1),
    (1978, 1),
    (1979, 1),
    (1980, 1),
    (1981, 7),
    (1982, 7),
    (1983, 7),
    (1985, 7),
    (1988, 1),
    (1990, 1),
    (1991, 1),
    (1992, 7),
    (1993, 7),
    (1994, 7),
    (1996, 1),
    (1997, 7),
    (1999, 1),
    (2006, 1),
    (2009, 1),
    (2012, 7),
    (2015, 7),
    (2017, 1),
];

/// JD at 0h UTC of 2017-01-01, the most recent leap-second boundary.
const BOUNDARY_2017: f64 = 2_457_754.5;

#[test]
fn table_jds_match_julday_of_documented_dates() -> Result<(), CoreError> {
    assert_eq!(LEAP_SECONDS.len(), LEAP_DATES.len());
    for (&(jd, _), &(year, month)) in LEAP_SECONDS.iter().zip(LEAP_DATES.iter()) {
        let expected = julday(Calendar::Gregorian, year, month, 1, 0.0)?;
        assert_eq!(
            jd.to_bits(),
            expected.value().to_bits(),
            "table row for {year}-{month:02}-01: {jd} != {}",
            expected.value()
        );
    }
    Ok(())
}

#[test]
fn table_is_strictly_ascending_with_unit_steps() {
    for pair in LEAP_SECONDS.windows(2) {
        assert!(
            pair[0].0 < pair[1].0,
            "table not ascending at {}",
            pair[1].0
        );
        assert!(
            (pair[1].1 - pair[0].1 - 1.0).abs() < 1e-12,
            "offset step != 1 s at {}",
            pair[1].0
        );
    }
    assert!((LEAP_SECONDS[0].1 - 10.0).abs() < 1e-12);
    assert!((LEAP_SECONDS[LEAP_SECONDS.len() - 1].1 - 37.0).abs() < 1e-12);
    assert_eq!(LEAP_SECONDS[0].0.to_bits(), UTC_LEAP_START_JD.to_bits());
}

#[test]
fn offset_lookup_spot_checks() -> Result<(), CoreError> {
    // 1972-01-01 exactly: first row applies.
    let start = JulianDate::from_f64(UTC_LEAP_START_JD);
    assert!((tai_minus_utc(start)? - 10.0).abs() < 1e-12);
    // Just before 1972: out of range.
    assert_eq!(
        tai_minus_utc(JulianDate::from_f64(UTC_LEAP_START_JD - 1.0)),
        Err(CoreError::DateOutOfRange)
    );
    // 1985-06-30 -> 22 s; 1985-07-01 -> 23 s.
    let before = julday(Calendar::Gregorian, 1985, 6, 30, 0.0)?;
    let after = julday(Calendar::Gregorian, 1985, 7, 1, 0.0)?;
    assert!((tai_minus_utc(before)? - 22.0).abs() < 1e-12);
    assert!((tai_minus_utc(after)? - 23.0).abs() < 1e-12);
    // Any date after the last announced leap second: 37 s.
    let recent = julday(Calendar::Gregorian, 2026, 7, 5, 12.0)?;
    assert!((tai_minus_utc(recent)? - 37.0).abs() < 1e-12);
    // NaN is invalid input.
    assert_eq!(
        tai_minus_utc(JulianDate::from_f64(f64::NAN)),
        Err(CoreError::InvalidInput)
    );
    Ok(())
}

#[test]
fn round_trips_away_from_boundaries() -> Result<(), CoreError> {
    let epochs = [
        julday(Calendar::Gregorian, 1975, 3, 10, 6.25)?,
        julday(Calendar::Gregorian, 1988, 8, 1, 18.0)?,
        julday(Calendar::Gregorian, 2000, 1, 1, 12.0)?,
        julday(Calendar::Gregorian, 2020, 2, 29, 23.0)?,
    ];
    for utc in epochs {
        let tai = utc_to_tai(utc)?;
        let back = tai_to_utc(tai)?;
        assert!(
            (back.diff_days(utc) * SECONDS_PER_DAY).abs() < 1e-9,
            "utc->tai->utc drift at {}",
            utc.value()
        );
        let tt = utc_to_tt(utc)?;
        let back2 = tt_to_utc(tt)?;
        assert!(
            (back2.diff_days(utc) * SECONDS_PER_DAY).abs() < 1e-9,
            "utc->tt->utc drift at {}",
            utc.value()
        );
        // TT - TAI is the 32.184 s constant.
        let dt = tt.diff_days(tai) * SECONDS_PER_DAY;
        assert!((dt - TT_MINUS_TAI_SECONDS).abs() < 1e-9);
        // tai_to_tt / tt_to_tai are inverses.
        let tt2 = tai_to_tt(tai)?;
        let tai2 = tt_to_tai(tt2)?;
        assert!((tai2.diff_days(tai) * SECONDS_PER_DAY).abs() < 1e-12);
    }
    // Known value: TT - UTC = 64.184 s at J2000.
    let utc = julday(Calendar::Gregorian, 2000, 1, 1, 12.0)?;
    let tt = utc_to_tt(utc)?;
    assert!((tt.diff_days(utc) * SECONDS_PER_DAY - 64.184).abs() < 1e-9);
    Ok(())
}

#[test]
fn behavior_at_the_2017_leap_boundary() -> Result<(), CoreError> {
    let boundary = JulianDate::from_f64(BOUNDARY_2017);
    // One second of UTC before the boundary: old offset (36 s).
    let utc_before = boundary.add_seconds(-1.0);
    let tai_before = utc_to_tai(utc_before)?;
    assert!((tai_before.diff_days(utc_before) * SECONDS_PER_DAY - 36.0).abs() < 1e-9);
    // At the boundary: new offset (37 s).
    let tai_at = utc_to_tai(boundary)?;
    assert!((tai_at.diff_days(boundary) * SECONDS_PER_DAY - 37.0).abs() < 1e-9);
    // Round trips just outside the inserted second are exact.
    for utc_offset in [-2.0, -0.5, 0.0, 0.25, 1.5] {
        let utc = boundary.add_seconds(utc_offset);
        let tai = utc_to_tai(utc)?;
        let back = tai_to_utc(tai)?;
        assert!(
            (back.diff_days(utc) * SECONDS_PER_DAY).abs() < 1e-9,
            "round trip at boundary{utc_offset:+}s"
        );
    }
    // A TAI instant *inside* the inserted second (UTC 23:59:60.5 cannot be
    // represented): mapped forward with the pre-jump offset, landing in the
    // first second of 2017-01-01 UTC. Documented convention.
    let tai_in_gap = boundary.add_seconds(36.5);
    let utc_gap = tai_to_utc(tai_in_gap)?;
    let past_boundary = utc_gap.diff_days(boundary) * SECONDS_PER_DAY;
    assert!(
        (0.0..1.0).contains(&past_boundary),
        "in-gap TAI must map into [boundary, boundary + 1 s), got {past_boundary:+}s"
    );
    // And the round trip is intentionally NOT the identity inside the gap.
    let tai_back = utc_to_tai(utc_gap)?;
    let gap_error = tai_back.diff_days(tai_in_gap) * SECONDS_PER_DAY;
    assert!(
        (gap_error - 1.0).abs() < 1e-9,
        "expected the 1 s ambiguity, got {gap_error}"
    );
    Ok(())
}

#[test]
fn tai_before_table_start_is_out_of_range() {
    // TAI a few seconds after 1972-01-01T00:00:00 TAI still corresponds to
    // a UTC before the table start -> DateOutOfRange.
    let tai = JulianDate::from_f64(UTC_LEAP_START_JD).add_seconds(5.0);
    assert_eq!(tai_to_utc(tai), Err(CoreError::DateOutOfRange));
}
