//! The chart facade's error type. Nothing here panics.
//!
//! `ChartError` is the union of every failure the compute layer can
//! produce, decoupled from any front end. The CLI wraps it in its own
//! `CliError`, the Python binding maps it to `PyValueError`, and the WASM
//! binding maps it to a `JsError` — but the messages and the fault lines
//! are defined once, here.

use core::fmt;

use oxiephemeris_bodies::BodiesError;
use oxiephemeris_core::CoreError;
use oxiephemeris_de::DeError;

use crate::iso8601::Iso8601Error;

/// A failure of the chart-computation facade.
#[non_exhaustive]
#[derive(Debug)]
pub enum ChartError {
    /// The ISO 8601 date/time string failed to parse.
    Iso8601(Iso8601Error),
    /// A calendar/time-scale conversion failed at the
    /// [`oxiephemeris_core`] level.
    Core(CoreError),
    /// The parsed year/month/day is not a real calendar date (e.g.
    /// `2026-02-30`, or February 29 in a non-leap year). Caught before
    /// the date reaches `oxiephemeris_core`, so the message can name the
    /// actual month length.
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
        /// Whether `year` is a leap year (phrases the February case).
        is_leap: bool,
    },
    /// A geographic longitude outside `[-180, 180]` degrees (or
    /// non-finite). A user-facing sanity guard against typos, not a domain
    /// necessity — longitude is periodic.
    InvalidLongitude(f64),
    /// The DE ephemeris bytes failed to parse or evaluate.
    De(DeError),
    /// The apparent-place pipeline failed (e.g. a target coincides with
    /// the observation center).
    Bodies(BodiesError),
    /// A house-cusp system (or a related chart angle, e.g. the Vertex) is
    /// undefined for the given inputs. The message is composed at the call
    /// site so it can name the specific system/angle.
    ChartGeometry(String),
    /// The lunar-node/apogee computation failed.
    Node(oxiephemeris_astro::nodes::NodeError),
    /// An unknown or malformed request field (e.g. an unrecognised house
    /// system or ayanamsha name).
    BadRequest(String),
    /// RDF serialization failed (an invalid base/chart IRI, or a write
    /// error).
    Rdf(oxiephemeris_rdf::RdfError),
    /// JSON serialization failed (should not happen for the fixed schema).
    Json(serde_json::Error),
}

impl fmt::Display for ChartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
                if *month == 2 {
                    write!(
                        f,
                        "{day} is not a valid day for {calendar_name} {year}-02: \
                         February has {days_in_month} days in {} {year}",
                        if *is_leap { "leap year" } else { "common year" }
                    )
                } else {
                    write!(
                        f,
                        "{day} is not a valid day for {calendar_name} \
                         {year}-{month:02}: that month has {days_in_month} days"
                    )
                }
            }
            Self::InvalidLongitude(lon) => write!(
                f,
                "longitude {lon} is out of range: it must be a finite value in \
                 [-180, 180] degrees (positive east)"
            ),
            Self::De(e) => write!(f, "ephemeris evaluation failed: {e}"),
            Self::Bodies(e) => write!(f, "apparent-place computation failed: {e}"),
            // Both carry an already-composed, ready-to-print message.
            Self::ChartGeometry(msg) | Self::BadRequest(msg) => write!(f, "{msg}"),
            Self::Node(e) => write!(f, "lunar node/apogee computation failed: {e}"),
            Self::Rdf(e) => write!(f, "failed to emit RDF: {e}"),
            Self::Json(e) => write!(f, "failed to serialize JSON: {e}"),
        }
    }
}

impl std::error::Error for ChartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Iso8601(e) => Some(e),
            Self::Core(e) => Some(e),
            Self::De(e) => Some(e),
            Self::Bodies(e) => Some(e),
            Self::Node(e) => Some(e),
            Self::Rdf(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<Iso8601Error> for ChartError {
    fn from(e: Iso8601Error) -> Self {
        Self::Iso8601(e)
    }
}

impl From<CoreError> for ChartError {
    fn from(e: CoreError) -> Self {
        Self::Core(e)
    }
}

impl From<DeError> for ChartError {
    fn from(e: DeError) -> Self {
        Self::De(e)
    }
}

impl From<BodiesError> for ChartError {
    fn from(e: BodiesError) -> Self {
        Self::Bodies(e)
    }
}

impl From<oxiephemeris_astro::nodes::NodeError> for ChartError {
    fn from(e: oxiephemeris_astro::nodes::NodeError) -> Self {
        Self::Node(e)
    }
}

impl From<oxiephemeris_rdf::RdfError> for ChartError {
    fn from(e: oxiephemeris_rdf::RdfError) -> Self {
        Self::Rdf(e)
    }
}

impl From<serde_json::Error> for ChartError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
