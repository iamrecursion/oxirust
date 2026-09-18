//! Shared UTC-epoch chart-geometry context, re-exported from the
//! `oxiephemeris-chart` facade.
//!
//! The compute — Julian centuries TT, Greenwich/local apparent sidereal
//! time, the true obliquity of date, and the pre-1972 ΔT fallback — lives
//! once in [`oxiephemeris_chart::epoch`] and is shared by the CLI, the
//! Python binding, and the WASM binding. This module only adapts the
//! facade's [`CalendarKind`](oxiephemeris_chart::CalendarKind)-typed
//! resolver to the CLI's [`CalArg`] and preserves the CLI's
//! `--lon`-naming longitude-range message.

pub use oxiephemeris_chart::epoch::ChartEpoch;

use crate::convert::CalArg;
use crate::errors::{check_longitude, CliError};

/// Resolves a UTC `date_str` (in calendar `cal`) plus `dut1_s`
/// (`UT1 = UTC + dut1_s`, seconds) and `east_longitude_deg` into a
/// [`ChartEpoch`], via the `oxiephemeris-chart` facade.
///
/// The CLI's own [`check_longitude`] runs first so an out-of-range `--lon`
/// gets the `--lon`-naming message the CLI has always emitted (the facade's
/// equivalent guard phrases it without the flag name); every other failure
/// is the facade's.
///
/// # Errors
///
/// [`CliError::InvalidLongitude`] for an out-of-range longitude, otherwise
/// [`CliError::Chart`] wrapping the facade's epoch-resolution error.
pub fn resolve_chart_epoch(
    date_str: &str,
    cal: CalArg,
    east_longitude_deg: f64,
    dut1_s: f64,
) -> Result<ChartEpoch, CliError> {
    check_longitude(east_longitude_deg)?;
    Ok(oxiephemeris_chart::resolve_chart_epoch(
        date_str,
        cal.to_calendar_kind(),
        east_longitude_deg,
        dut1_s,
    )?)
}
