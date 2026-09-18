//! Aggregate CLI error type: every fallible path in `oxieph` reports
//! through this instead of panicking (no `unwrap`/`expect` anywhere).

use std::fmt;
use std::path::PathBuf;

use oxiephemeris_bodies::BodiesError;
use oxiephemeris_core::time::{days_in_month, is_leap_year, Calendar};
use oxiephemeris_core::CoreError;
use oxiephemeris_de::DeError;

use crate::iso8601::Iso8601Error;

/// English month names, indexed `[month - 1]`, for friendly messages.
const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Errors surfaced by the `oxieph` binary. [`fmt::Display`] renders a
/// message suitable to print after `error: ` on stderr.
#[derive(Debug)]
pub enum CliError {
    /// A malformed CLI argument combination (e.g. neither or both of
    /// `--jd`/`--date` given to `convert`).
    Arg(String),
    /// The ISO 8601 date/time string failed to parse.
    Iso8601(Iso8601Error),
    /// A calendar/time-scale conversion failed
    /// ([`oxiephemeris_core`]-level).
    Core(CoreError),
    /// The parsed year/month/day is not a real calendar date (e.g.
    /// `2026-02-30`, or February 29 in a non-leap year). Caught by
    /// [`check_calendar_day`] at the CLI layer, before the date ever reaches
    /// `oxiephemeris_core`, so the message can name the actual month length
    /// instead of `oxiephemeris_core`'s generic
    /// [`CoreError::InvalidInput`].
    InvalidCalendarDay {
        /// Calendar name for the message, e.g. `"Gregorian"`.
        calendar_name: &'static str,
        /// Astronomical year number.
        year: i32,
        /// Month, 1-12.
        month: u8,
        /// The out-of-range day that was given.
        day: u8,
        /// The actual number of days in `month` for `(calendar, year)`.
        days_in_month: u8,
        /// Whether `year` is a leap year in the given calendar (used to
        /// phrase the February case specially).
        is_leap: bool,
    },
    /// `--jd` was given a non-finite value (`NaN` or `+-Infinity`); caught
    /// at the CLI layer before it reaches `oxiephemeris_core`.
    NonFiniteJd(f64),
    /// `--lon` was given a value outside the canonical geodetic range
    /// `[-180, 180]` degrees (or a non-finite value). Longitude is
    /// mathematically periodic (unlike latitude, which has a real polar
    /// singularity — see `oxiephemeris_astro::houses::HousesError::PolarLatitude`),
    /// so this is a user-facing input-sanity guard, not a domain necessity:
    /// it catches likely typos (e.g. `999`) before they silently produce a
    /// technically-computable but almost-certainly-unintended chart.
    InvalidLongitude(f64),
    /// `--scale utc` (the default) was requested for an epoch before the
    /// leap-second table starts (1972-01-01 UTC).
    PreLeapTableUtc,
    /// The DE ephemeris file could not be read from `path`.
    DeFileMissing {
        /// The path that was attempted.
        path: PathBuf,
        /// The underlying I/O failure.
        source: std::io::Error,
    },
    /// The DE file was read but failed to parse or evaluate.
    De(DeError),
    /// The apparent-place pipeline failed (e.g. target coincides with the
    /// observation center).
    Bodies(BodiesError),
    /// `--json` output could not be serialized (should not happen for the
    /// fixed schema this binary emits).
    Json(serde_json::Error),
    /// RDF (`--format turtle|ntriples`) output failed: an invalid base or
    /// chart IRI, or a write error.
    Rdf(oxiephemeris_rdf::RdfError),
    /// A failure inside the `oxiephemeris-chart` compute/serialize facade
    /// (chart assembly, comparison builders, or JSON/RDF rendering).
    Chart(oxiephemeris_chart::ChartError),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Carries an already-composed, ready-to-print message (built
            // with call-site-specific context); see its doc comment.
            Self::Arg(msg) => write!(f, "{msg}"),
            Self::Iso8601(e) => write!(f, "invalid date/time: {e}"),
            Self::Core(e) => write!(f, "{e}"),
            Self::InvalidCalendarDay {
                calendar_name,
                year,
                month,
                day,
                days_in_month,
                is_leap,
            } => {
                let month_name = MONTH_NAMES
                    .get(usize::from(*month).saturating_sub(1))
                    .copied()
                    .unwrap_or("<unknown month>");
                if *month == 2 && !*is_leap {
                    write!(
                        f,
                        "{year:04}-{month:02}-{day:02} is not a valid {calendar_name} \
                         calendar date: {year} is not a leap year, so February has only \
                         {days_in_month} days"
                    )
                } else {
                    write!(
                        f,
                        "{year:04}-{month:02}-{day:02} is not a valid {calendar_name} \
                         calendar date: {month_name} {year} has only {days_in_month} days"
                    )
                }
            }
            Self::NonFiniteJd(value) => write!(
                f,
                "--jd {value} is not a finite number; a Julian Date must be finite \
                 (NaN and +-Infinity are not valid)"
            ),
            Self::InvalidLongitude(value) => write!(
                f,
                "--lon {value} is out of range: geodetic longitude must be finite and between \
                 -180 and 180 degrees (inclusive), positive east"
            ),
            Self::PreLeapTableUtc => write!(
                f,
                "date is before 1972-01-01 UTC (the leap-second table has no data there); \
                 pass --scale tt to interpret the given clock reading directly as \
                 Terrestrial Time"
            ),
            Self::DeFileMissing { path, source } => write!(
                f,
                "could not read the DE ephemeris file at '{}': {source}\n\
                 resolution order: --de PATH, then $OXIEPH_DE, then \
                 ./data/de440/linux_p1550p2650.440 relative to the current directory\n\
                 to fetch DE440 (public domain, ~114 MB) without a repo checkout, run:\n\
                 curl -fsSL --create-dirs -o data/de440/linux_p1550p2650.440 \
                 https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440\n\
                 then re-run, or point --de/$OXIEPH_DE at wherever you saved it; from a \
                 repo checkout, scripts/fetch_de440.sh downloads it for you",
                path.display()
            ),
            Self::De(e) => write!(f, "ephemeris evaluation failed: {e}"),
            Self::Bodies(e) => write!(f, "apparent-place computation failed: {e}"),
            Self::Json(e) => write!(f, "failed to serialize JSON output: {e}"),
            Self::Rdf(e) => write!(f, "failed to emit RDF output: {e}"),
            // The facade already composes a fully user-facing message.
            Self::Chart(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Chart(e) => Some(e),
            _ => None,
        }
    }
}

impl From<Iso8601Error> for CliError {
    fn from(e: Iso8601Error) -> Self {
        Self::Iso8601(e)
    }
}

impl From<CoreError> for CliError {
    fn from(e: CoreError) -> Self {
        Self::Core(e)
    }
}

impl From<DeError> for CliError {
    fn from(e: DeError) -> Self {
        Self::De(e)
    }
}

impl From<BodiesError> for CliError {
    fn from(e: BodiesError) -> Self {
        Self::Bodies(e)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<oxiephemeris_rdf::RdfError> for CliError {
    fn from(e: oxiephemeris_rdf::RdfError) -> Self {
        Self::Rdf(e)
    }
}

impl From<oxiephemeris_chart::ChartError> for CliError {
    fn from(e: oxiephemeris_chart::ChartError) -> Self {
        Self::Chart(e)
    }
}

/// Validates that `day` is a real day-of-month for `(calendar, year,
/// month)`, beyond the coarse `1..=31` format check
/// [`iso8601::parse`](crate::iso8601::parse) performs. Called by `convert`
/// and `pos` right before handing the date to
/// [`oxiephemeris_core::time::julday`], so calendar-invalid dates
/// (`2026-02-30`, February 29 in a non-leap year) get a specific, friendly
/// [`CliError::InvalidCalendarDay`] instead of `oxiephemeris_core`'s generic
/// [`CoreError::InvalidInput`].
///
/// `month` is assumed already known to be `1..=12` (guaranteed by
/// `iso8601::parse`), so [`days_in_month`] cannot actually fail here; `31`
/// is used as an inert fallback rather than panicking if it somehow did.
///
/// # Errors
///
/// [`CliError::InvalidCalendarDay`] if `day` exceeds the actual length of
/// `month` in `(calendar, year)`.
pub fn check_calendar_day(
    calendar: Calendar,
    calendar_name: &'static str,
    year: i32,
    month: u8,
    day: u8,
) -> Result<(), CliError> {
    let dim = days_in_month(calendar, year, month).unwrap_or(31);
    if day == 0 || day > dim {
        Err(CliError::InvalidCalendarDay {
            calendar_name,
            year,
            month,
            day,
            days_in_month: dim,
            is_leap: is_leap_year(calendar, year),
        })
    } else {
        Ok(())
    }
}

/// Validates that a `--jd` value is finite. Called before the value ever
/// reaches [`oxiephemeris_core::time::revjul`], so `NaN`/`+-Infinity` get a
/// specific, friendly [`CliError::NonFiniteJd`] instead of
/// `oxiephemeris_core`'s generic [`CoreError::InvalidInput`].
///
/// # Errors
///
/// [`CliError::NonFiniteJd`] if `jd` is `NaN` or infinite.
pub fn check_finite_jd(jd: f64) -> Result<(), CliError> {
    if jd.is_finite() {
        Ok(())
    } else {
        Err(CliError::NonFiniteJd(jd))
    }
}

/// Validates that a `--lon` value is finite and within the canonical
/// geodetic range `[-180, 180]` degrees. Called before the value reaches
/// [`crate::astro_epoch::resolve_chart_epoch`] (`houses`/`chart`'s shared
/// epoch resolver), so an out-of-range longitude (e.g. a typo like `999`)
/// gets a specific, friendly [`CliError::InvalidLongitude`] instead of
/// silently succeeding.
///
/// # Errors
///
/// [`CliError::InvalidLongitude`] if `lon` is `NaN`, `+-Infinity`, or
/// outside `[-180, 180]`.
pub fn check_longitude(lon: f64) -> Result<(), CliError> {
    if (-180.0..=180.0).contains(&lon) {
        Ok(())
    } else {
        Err(CliError::InvalidLongitude(lon))
    }
}

#[cfg(test)]
mod tests {
    use super::{check_calendar_day, check_finite_jd, check_longitude, CliError};
    use oxiephemeris_core::time::Calendar;

    /// Returns the error message, or fails the test (without `unwrap`/
    /// `expect`) if `result` was unexpectedly `Ok`.
    fn message_of(result: Result<(), CliError>, ctx: &str) -> String {
        match result {
            Ok(()) => panic!("{ctx}: expected an error, got Ok"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn check_calendar_day_accepts_valid_dates() {
        assert!(check_calendar_day(Calendar::Gregorian, "Gregorian", 2024, 2, 29).is_ok());
        assert!(check_calendar_day(Calendar::Gregorian, "Gregorian", 2026, 1, 31).is_ok());
        assert!(check_calendar_day(Calendar::Julian, "Julian", -1000, 2, 29).is_ok());
    }

    #[test]
    fn check_calendar_day_rejects_february_30() {
        let msg = message_of(
            check_calendar_day(Calendar::Gregorian, "Gregorian", 2026, 2, 30),
            "2026-02-30",
        );
        assert!(msg.contains("2026-02-30"), "msg={msg}");
        assert!(msg.contains("28 days"), "msg={msg}");
        assert!(
            !msg.contains("invalid input to a conversion"),
            "must not fall back to the generic core message: {msg}"
        );
    }

    #[test]
    fn check_calendar_day_rejects_feb_29_in_non_leap_year() {
        let msg = message_of(
            check_calendar_day(Calendar::Gregorian, "Gregorian", 2026, 2, 29),
            "2026-02-29",
        );
        assert!(msg.contains("not a leap year"), "msg={msg}");
        assert!(msg.contains("2026-02-29"), "msg={msg}");
    }

    #[test]
    fn check_calendar_day_accepts_feb_29_in_leap_year() {
        assert!(check_calendar_day(Calendar::Gregorian, "Gregorian", 2024, 2, 29).is_ok());
    }

    #[test]
    fn check_calendar_day_julian_leap_rule_differs_from_gregorian() {
        // -1000 (1001 BCE) is a Julian leap year (div by 4) but not a
        // Gregorian one (div by 4, not by 100, and not by 400).
        assert!(check_calendar_day(Calendar::Julian, "Julian", -1000, 2, 29).is_ok());
        assert!(check_calendar_day(Calendar::Gregorian, "Gregorian", -1000, 2, 29).is_err());
    }

    #[test]
    fn check_calendar_day_rejects_day_zero() {
        assert!(check_calendar_day(Calendar::Gregorian, "Gregorian", 2026, 1, 0).is_err());
    }

    #[test]
    fn check_finite_jd_accepts_finite_values() {
        assert!(check_finite_jd(2_451_545.0).is_ok());
        assert!(check_finite_jd(-100.25).is_ok());
        assert!(check_finite_jd(0.0).is_ok());
    }

    #[test]
    fn check_finite_jd_rejects_nan_and_infinity() {
        assert!(check_finite_jd(f64::NAN).is_err());
        assert!(check_finite_jd(f64::INFINITY).is_err());
        assert!(check_finite_jd(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn non_finite_jd_message_is_friendly_not_generic() {
        let msg = message_of(check_finite_jd(f64::NAN), "NaN");
        assert!(msg.contains("--jd"), "msg={msg}");
        assert!(
            !msg.contains("invalid input to a conversion"),
            "must not fall back to the generic core message: {msg}"
        );
    }

    #[test]
    fn check_longitude_accepts_in_range_values() {
        assert!(check_longitude(-180.0).is_ok());
        assert!(check_longitude(-179.999).is_ok());
        assert!(check_longitude(0.0).is_ok());
        assert!(check_longitude(15.0).is_ok());
        assert!(check_longitude(179.999).is_ok());
        assert!(check_longitude(180.0).is_ok());
    }

    #[test]
    fn check_longitude_rejects_out_of_range_values() {
        assert!(check_longitude(180.000_001).is_err());
        assert!(check_longitude(-180.000_001).is_err());
        assert!(check_longitude(999.0).is_err());
        assert!(check_longitude(-500.0).is_err());
    }

    #[test]
    fn check_longitude_rejects_nan_and_infinity() {
        assert!(check_longitude(f64::NAN).is_err());
        assert!(check_longitude(f64::INFINITY).is_err());
        assert!(check_longitude(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn invalid_longitude_message_is_friendly() {
        let msg = message_of(check_longitude(999.0), "999");
        assert!(msg.contains("--lon"), "msg={msg}");
        assert!(
            !msg.contains("invalid input to a conversion"),
            "must not fall back to the generic core message: {msg}"
        );
    }
}
