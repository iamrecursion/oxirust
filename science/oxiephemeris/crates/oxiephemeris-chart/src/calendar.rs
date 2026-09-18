//! Calendar selection and the pre-`julday` validation guards.
//!
//! [`oxiephemeris_core`]'s `julday` re-checks day validity as a
//! defense-in-depth backstop, but its generic `CoreError::InvalidInput`
//! cannot say *why*. These guards run first so the facade can return a
//! calendar-aware message naming the actual month length.

use oxiephemeris_core::time::{days_in_month, is_leap_year, Calendar};

use crate::error::ChartError;

/// Which proleptic calendar a date string is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalendarKind {
    /// Proleptic Gregorian calendar (the default).
    Gregorian,
    /// Proleptic Julian calendar.
    Julian,
}

impl CalendarKind {
    /// Parses a calendar name (`"gregorian"` / `"julian"`,
    /// case-insensitive). Defaults to Gregorian for an empty string.
    ///
    /// # Errors
    ///
    /// [`ChartError::BadRequest`] for any other value.
    pub fn parse(name: &str) -> Result<Self, ChartError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "gregorian" => Ok(Self::Gregorian),
            "julian" => Ok(Self::Julian),
            other => Err(ChartError::BadRequest(format!(
                "unknown calendar '{other}'; expected 'gregorian' or 'julian'"
            ))),
        }
    }

    /// Maps to the [`oxiephemeris_core`] calendar.
    #[must_use]
    pub const fn to_core(self) -> Calendar {
        match self {
            Self::Gregorian => Calendar::Gregorian,
            Self::Julian => Calendar::Julian,
        }
    }

    /// The capitalized calendar name for messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gregorian => "Gregorian",
            Self::Julian => "Julian",
        }
    }
}

/// Validates that `day` is a real day-of-month for `(calendar, year,
/// month)`, beyond the coarse `1..=31` check the ISO 8601 parser performs.
///
/// `month` is assumed already `1..=12`; `days_in_month` cannot fail, but
/// `31` is used as an inert fallback rather than panicking if it somehow
/// did.
///
/// # Errors
///
/// [`ChartError::InvalidCalendarDay`] if `day` exceeds the actual length
/// of `month` in `(calendar, year)`.
pub fn check_calendar_day(
    calendar: CalendarKind,
    year: i32,
    month: u8,
    day: u8,
) -> Result<(), ChartError> {
    let core = calendar.to_core();
    let dim = days_in_month(core, year, month).unwrap_or(31);
    if day == 0 || day > dim {
        Err(ChartError::InvalidCalendarDay {
            calendar_name: calendar.name(),
            year,
            month,
            day,
            days_in_month: dim,
            is_leap: is_leap_year(core, year),
        })
    } else {
        Ok(())
    }
}

/// Validates that a geographic longitude is finite and within `[-180,
/// 180]` degrees, catching typos (e.g. `999`) before they produce a
/// technically-computable but unintended chart.
///
/// # Errors
///
/// [`ChartError::InvalidLongitude`] if `lon` is `NaN`, infinite, or out of
/// range.
pub fn check_longitude(lon: f64) -> Result<(), ChartError> {
    if (-180.0..=180.0).contains(&lon) {
        Ok(())
    } else {
        Err(ChartError::InvalidLongitude(lon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_calendar_names() {
        assert_eq!(
            CalendarKind::parse("gregorian").ok(),
            Some(CalendarKind::Gregorian)
        );
        assert_eq!(
            CalendarKind::parse("JULIAN").ok(),
            Some(CalendarKind::Julian)
        );
        assert_eq!(CalendarKind::parse("").ok(), Some(CalendarKind::Gregorian));
        assert!(CalendarKind::parse("mayan").is_err());
    }

    #[test]
    fn rejects_impossible_days() {
        assert!(check_calendar_day(CalendarKind::Gregorian, 2026, 2, 30).is_err());
        assert!(check_calendar_day(CalendarKind::Gregorian, 2025, 2, 29).is_err());
        assert!(check_calendar_day(CalendarKind::Gregorian, 2024, 2, 29).is_ok());
        assert!(check_calendar_day(CalendarKind::Gregorian, 2026, 4, 31).is_err());
        assert!(check_calendar_day(CalendarKind::Gregorian, 2026, 1, 0).is_err());
    }

    #[test]
    fn longitude_range() {
        assert!(check_longitude(100.0).is_ok());
        assert!(check_longitude(-180.0).is_ok());
        assert!(check_longitude(999.0).is_err());
        assert!(check_longitude(f64::NAN).is_err());
    }
}
