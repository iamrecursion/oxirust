//! UTC-epoch chart-geometry context: Julian centuries TT,
//! Greenwich/local apparent sidereal time, and the true obliquity of date.
//!
//! # `UT1`
//!
//! `UT1 = UTC + dut1` (`dut1` defaults to `0` seconds): without a supplied
//! `dUT1` from IERS Earth-orientation data, `UT1` is approximated by `UTC`
//! itself, differing from true `UT1` by at most `0.9` s. Pass `dut1` (from
//! e.g. the IERS `finals2000A.all` series) for sub-arcsecond sidereal-time
//! accuracy.
//!
//! # Nutation model
//!
//! The full IAU `2000A_R06` series ([`nutation_iau2000a`]) is used
//! unconditionally.

use oxiephemeris_astro::angles::local_sidereal_time;
use oxiephemeris_astro::ayanamsha::{ayanamsha_rad, Ayanamsha};
use oxiephemeris_bodies::frames::{mean_obliquity_iau2006, nutation_iau2000a};
use oxiephemeris_bodies::sidereal::gast_iau2006;
use oxiephemeris_core::angle::normalize_pm_pi;
use oxiephemeris_core::time::{
    delta_t_seconds, is_leap_year, julday, leap::utc_to_tt, revjul, Calendar, JulianDate, J2000_JD,
};

use crate::calendar::{check_calendar_day, check_longitude, CalendarKind};
use crate::error::ChartError;
use crate::iso8601;

/// Days per Julian century (IERS TN36 eq. 5.2).
const DAYS_PER_JULIAN_CENTURY: f64 = 36_525.0;

/// Central-difference step, days, for
/// [`sidereal_offset_true_equinox_rate_rad_per_day`].
const SPEED_STEP_DAYS: f64 = 0.01;

/// ΔT in seconds at a (UT-ish) Julian Date, via the Espenak–Meeus
/// polynomial on the Gregorian decimal year.
fn delta_t_of(jd: JulianDate) -> Result<f64, ChartError> {
    let (date, _) = revjul(jd, Calendar::Gregorian)?;
    let jan1 = julday(Calendar::Gregorian, date.year, 1, 1, 0.0)?;
    let days_in_year = if is_leap_year(Calendar::Gregorian, date.year) {
        366.0
    } else {
        365.0
    };
    let day_of_year = jd.value() - jan1.value();
    Ok(delta_t_seconds(
        f64::from(date.year) + day_of_year / days_in_year,
    ))
}

/// Chart-geometry context resolved from a UTC epoch, a `dUT1` offset, and
/// an observer's east longitude.
#[derive(Debug, Clone, Copy)]
pub struct ChartEpoch {
    /// The epoch as a two-part TT Julian Date.
    pub jd_tt: JulianDate,
    /// The same epoch as the UTC Julian Date the input string denotes
    /// (pre-1972 this is really UT1). Kept so the RDF layer can render the
    /// epoch as an `xsd:dateTimeStamp` with calendar-correct rollover.
    pub jd_utc: JulianDate,
    /// Julian centuries TT since J2000.0.
    pub t_tt: f64,
    /// Greenwich apparent sidereal time, radians in `[0, 2*pi)`.
    pub gast_rad: f64,
    /// Local apparent sidereal time (`GAST` + east longitude), radians in
    /// `[0, 2*pi)` — the `theta`/RAMC argument of the angle/house
    /// functions.
    pub last_rad: f64,
    /// True obliquity of date (`eps_A + d_eps`), radians.
    pub eps_true_rad: f64,
    /// Nutation in longitude `d_psi`, radians.
    pub dpsi_rad: f64,
}

/// Resolves a UTC `date_str` (in calendar `cal`), a `dut1_s` offset
/// (`UT1 = UTC + dut1_s`, seconds), and `east_longitude_deg` into a
/// [`ChartEpoch`].
///
/// # Time scales before 1972
///
/// From 1972-01-01 the reading is proper UTC and TT comes from the
/// leap-second table. Before that the reading is interpreted as **UT1**
/// and TT comes from the Espenak–Meeus ΔT polynomial.
///
/// # Errors
///
/// Propagates [`ChartError::Iso8601`], [`ChartError::InvalidCalendarDay`],
/// [`ChartError::InvalidLongitude`], and [`ChartError::Core`].
#[allow(clippy::similar_names)] // jd_utc/jd_ut1/jd_tt are three time scales
pub fn resolve_chart_epoch(
    date_str: &str,
    cal: CalendarKind,
    east_longitude_deg: f64,
    dut1_s: f64,
) -> Result<ChartEpoch, ChartError> {
    let parsed = iso8601::parse(date_str)?;
    check_calendar_day(cal, parsed.year, parsed.month, parsed.day)?;
    check_longitude(east_longitude_deg)?;
    let hours = parsed.utc_hours();
    let jd_utc = julday(cal.to_core(), parsed.year, parsed.month, parsed.day, hours)?;
    let jd_tt = match utc_to_tt(jd_utc) {
        Ok(jd) => jd,
        Err(_) => jd_utc.add_seconds(delta_t_of(jd_utc)?),
    };
    let jd_ut1 = jd_utc.add_seconds(dut1_s);

    let t_tt = ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_JULIAN_CENTURY;
    let gast_rad = gast_iau2006(jd_ut1, t_tt);
    let last_rad = local_sidereal_time(gast_rad, east_longitude_deg.to_radians());
    let nut = nutation_iau2000a(t_tt);
    let eps_true_rad = mean_obliquity_iau2006(t_tt) + nut.deps_rad;

    Ok(ChartEpoch {
        jd_tt,
        jd_utc,
        t_tt,
        gast_rad,
        last_rad,
        eps_true_rad,
        dpsi_rad: nut.dpsi_rad,
    })
}

/// Microseconds in a day, and the largest time-of-day value
/// (`23:59:59.999999`).
const MICROS_PER_DAY: i64 = 86_400_000_000;
const MAX_TIME_OF_DAY_MICROS: i64 = MICROS_PER_DAY - 1;

/// Renders a UTC Julian Date as an ISO 8601 `xsd:dateTimeStamp` lexical
/// form, e.g. `"1970-01-01T00:00:00.000000Z"`.
///
/// Going through [`revjul`] normalizes every accepted spelling to one
/// canonical UTC form and keeps the calendar correct when a UTC offset
/// crossed midnight. The time of day is split in integer microseconds;
/// a value a few nanoseconds below 24 h is clamped to `23:59:59.999999`
/// (error ≤ 1 µs; the exact epoch is published separately at full `f64`
/// precision).
///
/// # Errors
///
/// Propagates [`revjul`]'s [`ChartError::Core`].
#[allow(clippy::cast_possible_truncation)] // clamped to < 2^37 µs below
pub fn epoch_utc_iso8601(jd_utc: JulianDate) -> Result<String, ChartError> {
    let (date, hours) = revjul(jd_utc, Calendar::Gregorian)?;
    let micros_f = (hours * 3_600_000_000.0).round();
    let micros = if micros_f.is_finite() {
        (micros_f as i64).clamp(0, MAX_TIME_OF_DAY_MICROS)
    } else {
        0
    };
    let hour = micros / 3_600_000_000;
    let after_hour = micros % 3_600_000_000;
    let minute = after_hour / 60_000_000;
    let after_minute = after_hour % 60_000_000;
    let second = after_minute / 1_000_000;
    let fraction = after_minute % 1_000_000;
    Ok(format!(
        "{:04}-{:02}-{:02}T{hour:02}:{minute:02}:{second:02}.{fraction:06}Z",
        date.year, date.month, date.day
    ))
}

/// The sidereal offset to subtract from a **true-equinox-of-date**
/// tropical longitude to get its sidereal counterpart: the mean-equinox
/// [`ayanamsha_rad`] plus the nutation in longitude `dpsi_rad`.
///
/// Subtracting `Δψ` along with the mean ayanamsha makes the sidereal
/// longitude free of the equinox's nutation wobble — a star-anchored
/// zodiac must not nutate.
#[must_use]
pub fn sidereal_offset_true_equinox_rad(kind: Ayanamsha, t_tt: f64, dpsi_rad: f64) -> f64 {
    ayanamsha_rad(kind, t_tt) + dpsi_rad
}

/// Drift rate of [`sidereal_offset_true_equinox_rad`], rad/day: a central
/// difference over `t_tt` using a small day-scale step, re-evaluating the
/// nutation at each sample point.
#[must_use]
pub fn sidereal_offset_true_equinox_rate_rad_per_day(kind: Ayanamsha, t_tt: f64) -> f64 {
    let dt_c = SPEED_STEP_DAYS / DAYS_PER_JULIAN_CENTURY;
    let plus_t = t_tt + dt_c;
    let minus_t = t_tt - dt_c;
    let plus = sidereal_offset_true_equinox_rad(kind, plus_t, nutation_iau2000a(plus_t).dpsi_rad);
    let minus =
        sidereal_offset_true_equinox_rad(kind, minus_t, nutation_iau2000a(minus_t).dpsi_rad);
    normalize_pm_pi(plus - minus) / (2.0 * SPEED_STEP_DAYS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn true_equinox_offset_matches_two_step_derivation() {
        let t_tt = 0.05;
        let kind = Ayanamsha::Lahiri;
        let dpsi_rad = nutation_iau2000a(t_tt).dpsi_rad;
        let mean_equinox_lon_rad = 1.234_567_f64;
        let true_equinox_lon_rad = mean_equinox_lon_rad + dpsi_rad;
        let sidereal_via_two_step =
            true_equinox_lon_rad - sidereal_offset_true_equinox_rad(kind, t_tt, dpsi_rad);
        let sidereal_via_mean_ayanamsha_alone = mean_equinox_lon_rad - ayanamsha_rad(kind, t_tt);
        assert!((sidereal_via_two_step - sidereal_via_mean_ayanamsha_alone).abs() < 1e-15);
    }

    #[test]
    fn epoch_string_is_canonical_utc() {
        let Ok(epoch) =
            resolve_chart_epoch("1970-01-01T00:00:00Z", CalendarKind::Gregorian, 0.0, 0.0)
        else {
            panic!("epoch must resolve");
        };
        let Ok(text) = epoch_utc_iso8601(epoch.jd_utc) else {
            panic!("epoch must render");
        };
        assert_eq!(text, "1970-01-01T00:00:00.000000Z");
    }

    #[test]
    fn epoch_string_rolls_the_calendar_across_midnight() {
        let Ok(epoch) = resolve_chart_epoch(
            "1970-01-01T05:00:00+09:00",
            CalendarKind::Gregorian,
            0.0,
            0.0,
        ) else {
            panic!("epoch must resolve");
        };
        let Ok(text) = epoch_utc_iso8601(epoch.jd_utc) else {
            panic!("epoch must render");
        };
        assert_eq!(text, "1969-12-31T20:00:00.000000Z");
    }

    #[test]
    fn rate_is_close_to_mean_precession_rate() {
        use oxiephemeris_core::angle::RAD2DEG;
        let expected_deg_day = 50.29 / 365.25 / 3600.0;
        let rate =
            sidereal_offset_true_equinox_rate_rad_per_day(Ayanamsha::J2000Zero, 0.0) * RAD2DEG;
        assert!((rate - expected_deg_day).abs() < 0.02 / 3600.0);
    }
}
