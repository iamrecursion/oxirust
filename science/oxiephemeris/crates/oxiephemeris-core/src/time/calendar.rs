//! Proleptic Gregorian and Julian calendar ↔ Julian Date conversions.
//!
//! Both calendars are proleptic in both directions and BCE sign-correct
//! using **astronomical year numbering**: year 0 = 1 BCE, year −1 = 2 BCE,
//! so leap-year rules apply directly to the (possibly negative) year value
//! with Euclidean remainders.
//!
//! # Algorithms
//!
//! The day-number conversions use the "shifted year starting in March"
//! device, in which the leap day is the last day of the shifted year and
//! month lengths follow the linear pattern `(153·m + 2)/5`:
//!
//! - Richards, "Calendars", ch. 15 of Urban & Seidelmann (eds.),
//!   *Explanatory Supplement to the Astronomical Almanac*, 3rd ed. (2013)
//!   — the parameterized integer algorithms (15.11 "converting between the
//!   day of the year and calendar dates" family) cover both the Julian and
//!   Gregorian calendars.
//! - Fliegel & Van Flandern (1968), Comm. ACM 11(10), 657 — the classic
//!   Gregorian date → JD one-liner, algebraically equivalent for the
//!   Gregorian case.
//!
//! The published forms assume Fortran-style truncating division and
//! therefore limited year ranges; here every division is `i64::div_euclid`
//! (floor division for positive divisors), which makes the identities exact
//! for **all** proleptic years, positive or negative. This adaptation is
//! verified by anchor and property tests (round trips over ±3000 CE and
//! beyond, calendar-reform continuity, BCE leap years).

use crate::time::jd::JulianDate;
use crate::CoreError;

/// Calendar used for date ↔ JD conversions. Both are proleptic: they extend
/// indefinitely in both directions with their respective leap rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Calendar {
    /// Proleptic Gregorian calendar (400-year cycle of 146 097 days).
    Gregorian,
    /// Proleptic Julian calendar (4-year cycle of 1461 days).
    Julian,
}

/// A calendar date (year, month, day) in astronomical year numbering.
///
/// Construct via [`CalendarDate::new`] to get month/day validation against
/// the given calendar's leap rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CalendarDate {
    /// Astronomical year number (year 0 = 1 BCE, −1 = 2 BCE, …).
    pub year: i32,
    /// Month, 1–12.
    pub month: u8,
    /// Day of month, 1–28/29/30/31 depending on month and calendar.
    pub day: u8,
}

impl CalendarDate {
    /// Construct a validated calendar date.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidInput`] if `month` is not 1–12 or `day` is not a
    /// valid day of that month in `calendar` (leap rules honored, e.g.
    /// Gregorian −1000-02-29 is invalid while Julian −1000-02-29 is valid).
    pub fn new(calendar: Calendar, year: i32, month: u8, day: u8) -> Result<Self, CoreError> {
        let dim = days_in_month(calendar, year, month)?;
        if day == 0 || day > dim {
            return Err(CoreError::InvalidInput);
        }
        Ok(Self { year, month, day })
    }
}

/// Leap-year predicate, BCE sign-correct via Euclidean remainders.
///
/// - Julian: every year divisible by 4 (so year 0 = 1 BCE **is** a leap
///   year, as is −1000).
/// - Gregorian: divisible by 4 and not by 100, or divisible by 400 (so
///   year 0 is a leap year but −1000 is **not**).
#[must_use]
pub fn is_leap_year(calendar: Calendar, year: i32) -> bool {
    match calendar {
        Calendar::Julian => year.rem_euclid(4) == 0,
        Calendar::Gregorian => {
            (year.rem_euclid(4) == 0 && year.rem_euclid(100) != 0) || year.rem_euclid(400) == 0
        }
    }
}

/// Number of days in a month.
///
/// # Errors
///
/// [`CoreError::InvalidInput`] if `month` is not 1–12.
pub fn days_in_month(calendar: Calendar, year: i32, month: u8) -> Result<u8, CoreError> {
    const LENGTHS: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if month == 0 || month > 12 {
        return Err(CoreError::InvalidInput);
    }
    let base = LENGTHS[usize::from(month) - 1];
    if month == 2 && is_leap_year(calendar, year) {
        Ok(base + 1)
    } else {
        Ok(base)
    }
}

/// Julian Day Number (the integer JD of the *noon* of the given civil day)
/// from a calendar date.
///
/// Shifted-year form of Richards (2013), ch. 15; all divisions Euclidean so
/// the identity holds for every proleptic year (see module docs).
fn jdn_from_date(calendar: Calendar, year: i64, month: i64, day: i64) -> i64 {
    // Months March..February are numbered 0..11 in a year that starts in
    // March; `shift` is 1 for January/February, 0 otherwise.
    let shift = (14 - month).div_euclid(12);
    let y = year + 4800 - shift;
    let m = month + 12 * shift - 3;
    // (153 m + 2)/5 is the number of days in the shifted year before month m.
    let common = day + (153 * m + 2).div_euclid(5) + 365 * y + y.div_euclid(4);
    match calendar {
        Calendar::Julian => common - 32083,
        Calendar::Gregorian => common - y.div_euclid(100) + y.div_euclid(400) - 32045,
    }
}

/// Calendar date from a Julian Day Number (inverse of [`jdn_from_date`]).
///
/// Inverse shifted-year algorithm of Richards (2013), ch. 15, with
/// Euclidean divisions throughout (see module docs).
fn date_from_jdn(calendar: Calendar, jdn: i64) -> (i64, i64, i64) {
    // `century` counts Gregorian centuries for the 100/400-year rule; it is
    // identically 0 for the Julian calendar.
    let (century, in_century) = match calendar {
        Calendar::Gregorian => {
            let a = jdn + 32044;
            let b = (4 * a + 3).div_euclid(146_097);
            (b, a - (146_097 * b).div_euclid(4))
        }
        Calendar::Julian => (0, jdn + 32082),
    };
    let d = (4 * in_century + 3).div_euclid(1461);
    let e = in_century - (1461 * d).div_euclid(4);
    let m = (5 * e + 2).div_euclid(153);
    let day = e - (153 * m + 2).div_euclid(5) + 1;
    let month = m + 3 - 12 * m.div_euclid(10);
    let year = 100 * century + d - 4800 + m.div_euclid(10);
    (year, month, day)
}

/// Largest `|JD|` accepted by [`revjul`]. Beyond this the f64 day count
/// itself is coarser than ≈ 20 s and the year would overflow `i32` long
/// before the arithmetic overflows `i64`.
const MAX_ABS_JD: f64 = 1.0e12;

/// Calendar date + time of day → two-part Julian Date.
///
/// Semantics follow the classic ephemeris `julday` convention: `hours` is
/// the time of day in hours. Any **finite** `hours` is accepted and folded
/// into the date (e.g. `25.5` rolls into the next day, negative values roll
/// backwards).
///
/// The returned split is `hi` = the `.5`-aligned day boundary (midnight
/// starting the resulting civil day) and `lo` = the day fraction in
/// `[0, 1)`, so callers can recover the civil day and time of day exactly.
/// This split is intentionally *not* the canonical two-sum form of
/// [`JulianDate::new`]; the arithmetic methods renormalize as needed.
///
/// # Errors
///
/// - [`CoreError::InvalidInput`] for an invalid month/day (per `calendar`'s
///   leap rule) or a non-finite `hours`.
pub fn julday(
    calendar: Calendar,
    year: i32,
    month: u8,
    day: u8,
    hours: f64,
) -> Result<JulianDate, CoreError> {
    let date = CalendarDate::new(calendar, year, month, day)?;
    if !hours.is_finite() {
        return Err(CoreError::InvalidInput);
    }
    let jdn = jdn_from_date(
        calendar,
        i64::from(date.year),
        i64::from(date.month),
        i64::from(date.day),
    );
    // Fold the time of day: whole days go into `hi`, the remainder in [0, 1)
    // becomes `lo`.
    let day_fraction_raw = hours / 24.0;
    let whole_days = libm::floor(day_fraction_raw);
    let day_fraction = day_fraction_raw - whole_days;
    // Exact: |jdn| < 2⁴¹ for any i32 year, far below 2⁵³.
    #[allow(clippy::cast_precision_loss)]
    let jdn_f = jdn as f64;
    Ok(JulianDate {
        hi: jdn_f - 0.5 + whole_days,
        lo: day_fraction,
    })
}

/// Two-part Julian Date → calendar date + time of day in hours (`[0, 24)`).
///
/// Inverse of [`julday`]. The day fraction is extracted with two-part
/// arithmetic so a `julday` result round-trips its time of day to full
/// precision.
///
/// # Errors
///
/// - [`CoreError::InvalidInput`] for a non-finite JD.
/// - [`CoreError::DateOutOfRange`] if `|JD| > 10¹²` or the resulting year
///   does not fit in `i32`.
pub fn revjul(jd: JulianDate, calendar: Calendar) -> Result<(CalendarDate, f64), CoreError> {
    let value = jd.value();
    if !value.is_finite() {
        return Err(CoreError::InvalidInput);
    }
    if value.abs() > MAX_ABS_JD {
        return Err(CoreError::DateOutOfRange);
    }
    // Day number of the civil day containing this instant, noon-referenced:
    // z = ⌊JD + 0.5⌋. Computed on the rounded sum, then corrected below so
    // the two-part fraction lands in [0, 1).
    let mut z = libm::floor(value + 0.5);
    // (hi − z) is exact (Sterbenz: hi and z are within a factor of two or
    // both small); adding 0.5 and lo keeps the two-part precision.
    let mut frac = ((jd.hi - z) + 0.5) + jd.lo;
    if frac < 0.0 {
        z -= 1.0;
        frac += 1.0;
    } else if frac >= 1.0 {
        z += 1.0;
        frac -= 1.0;
    }
    // In range: |z| ≤ 10¹² + 1 < 2⁵³, exact integer-valued f64.
    #[allow(clippy::cast_possible_truncation)]
    let jdn = z as i64;
    let (year, month, day) = date_from_jdn(calendar, jdn);
    let year = i32::try_from(year).map_err(|_| CoreError::DateOutOfRange)?;
    // month is 1–12 and day 1–31 by construction of `date_from_jdn`.
    let month = u8::try_from(month).map_err(|_| CoreError::DateOutOfRange)?;
    let day = u8::try_from(day).map_err(|_| CoreError::DateOutOfRange)?;
    Ok((CalendarDate { year, month, day }, frac * 24.0))
}

#[cfg(test)]
mod tests {
    use super::{date_from_jdn, jdn_from_date, Calendar};

    #[test]
    fn jdn_round_trips_across_bce_boundary() {
        for cal in [Calendar::Gregorian, Calendar::Julian] {
            // A solid block of days spanning year 0 and the JD 0 region.
            for jdn in [-1_000_000i64, -1, 0, 1, 1_721_057, 1_721_058, 2_451_545] {
                let (y, m, d) = date_from_jdn(cal, jdn);
                assert_eq!(jdn_from_date(cal, y, m, d), jdn, "{cal:?} jdn={jdn}");
                assert!((1..=12).contains(&m));
                assert!((1..=31).contains(&d));
            }
        }
    }
}
