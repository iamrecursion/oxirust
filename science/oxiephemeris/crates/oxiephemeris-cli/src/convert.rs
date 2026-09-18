//! `oxieph convert`: Julian Date <-> proleptic calendar date/time.
//! Pure calendar arithmetic — no leap seconds, no time-scale conversion
//! (that is `pos`'s `--scale`).
//!
//! # JSON schema (`--json`; stable, documented here)
//!
//! Symmetric with `pos --json` ([`crate::pos`]): full `f64` precision via
//! `serde_json`, one object per invocation, printed as a single line.
//!
//! `date` -> `jd` (i.e. `--date` was given):
//!
//! ```text
//! {
//!   "year": <i32>, "month": <u8>, "day": <u8>,
//!   "hour": <u32>, "minute": <u32>, "second": <f64>,
//!   "jd": <f64>,
//!   "calendar": "gregorian" | "julian",
//!   "scale": "utc"
//! }
//! ```
//!
//! `jd` -> `date` (i.e. `--jd` was given):
//!
//! ```text
//! {
//!   "jd": <f64>,
//!   "calendar": "gregorian" | "julian",
//!   "scale": "utc",
//!   "year": <i32>, "month": <u8>, "day": <u8>,
//!   "hour": <u32>, "minute": <u32>, "second": <f64>
//! }
//! ```
//!
//! `year`/`month`/`day`/`hour`/`minute`/`second` are the calendar date and
//! time-of-day (the latter split the same way as the human-readable
//! `time = HH:MM:SS.ffffff` line, but at full precision rather than 9
//! decimal digits). In **both** directions every field of one JSON object
//! describes the same UTC instant as its `jd`: for `date` -> `jd` an input
//! carrying a non-zero UTC offset (e.g. `+09:00`) is normalized to UTC
//! first (in exact minute arithmetic, including the one-minute carry of a
//! `:60` leap-second reading), and the emitted fields reflect that
//! normalized date/time, not the raw input echo.
//! `calendar` is `--cal`, lowercased. `scale` is always `"utc"`: unlike
//! `pos`, `convert` performs pure calendar arithmetic with no leap-second
//! table and no `--scale` flag, so the field is a constant included only
//! for schema symmetry with `pos --json` (which does support `"tt"`).
//! `jd` is the two-part [`JulianDate`] collapsed to one `f64` (via
//! [`JulianDate::value`]), exactly as `pos`'s `jd_tt` field is.

use clap::{Args, ValueEnum};
use oxiephemeris_core::time::{julday, revjul, Calendar, JulianDate};
use serde::Serialize;

use crate::errors::{check_calendar_day, check_finite_jd, CliError};
use crate::iso8601;

/// Calendar selector shared by `convert` and used to interpret `--date`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CalArg {
    /// Proleptic Gregorian calendar (default).
    Gregorian,
    /// Proleptic Julian calendar.
    Julian,
}

impl CalArg {
    /// Maps to the `oxiephemeris_core` calendar. `pub(crate)`: shared with
    /// `houses`/`chart`, which also accept `--cal`.
    pub(crate) const fn to_calendar(self) -> Calendar {
        match self {
            Self::Gregorian => Calendar::Gregorian,
            Self::Julian => Calendar::Julian,
        }
    }

    /// Calendar name for human-readable and friendly-error output.
    /// `pub(crate)`: shared with `houses`/`chart`.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Gregorian => "Gregorian",
            Self::Julian => "Julian",
        }
    }

    /// Maps to the `oxiephemeris_chart` facade calendar kind. `pub(crate)`:
    /// shared with `houses`/`chart` and the two-chart commands, which resolve
    /// their epochs through the facade.
    pub(crate) const fn to_calendar_kind(self) -> oxiephemeris_chart::CalendarKind {
        match self {
            Self::Gregorian => oxiephemeris_chart::CalendarKind::Gregorian,
            Self::Julian => oxiephemeris_chart::CalendarKind::Julian,
        }
    }

    /// JSON `"calendar"` field value: same spelling, lowercased.
    const fn json_name(self) -> &'static str {
        match self {
            Self::Gregorian => "gregorian",
            Self::Julian => "julian",
        }
    }
}

impl std::fmt::Display for CalArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// `oxieph convert` arguments.
#[derive(Debug, Args)]
pub struct ConvertArgs {
    /// Julian Date to convert to a calendar date (mutually exclusive with
    /// `--date`); fractional and negative values are accepted, e.g.
    /// `--jd=-100.5` or `--jd -100.5`.
    #[arg(long, allow_hyphen_values = true)]
    pub jd: Option<f64>,
    /// ISO 8601 date/time to convert to a Julian Date (mutually exclusive
    /// with `--jd`). Negative (BCE) years are accepted, e.g.
    /// `--date=-0001-06-15` or `--date -0001-06-15`.
    #[arg(long, allow_hyphen_values = true)]
    pub date: Option<String>,
    /// Calendar to interpret/print the date in.
    #[arg(long, value_enum, default_value_t = CalArg::Gregorian)]
    pub cal: CalArg,
    /// Emit the stable JSON schema documented on this module instead of
    /// human-readable text.
    #[arg(long)]
    pub json: bool,
}

/// The stable `--json` schema for the `--date` (date -> JD) direction; see
/// the module doc.
#[derive(Debug, Serialize)]
struct DateToJdJson {
    year: i32,
    month: u8,
    day: u8,
    hour: u32,
    minute: u32,
    second: f64,
    jd: f64,
    calendar: &'static str,
    scale: &'static str,
}

/// The stable `--json` schema for the `--jd` (JD -> date) direction; see
/// the module doc.
#[derive(Debug, Serialize)]
struct JdToDateJson {
    jd: f64,
    calendar: &'static str,
    scale: &'static str,
    year: i32,
    month: u8,
    day: u8,
    hour: u32,
    minute: u32,
    second: f64,
}

/// Formats an astronomical year with a 4-digit magnitude for `|year| <
/// 10000` (matching ISO 8601 basic representation, with an explicit sign
/// only when negative) and a signed 6-digit expanded form beyond that.
fn format_year(year: i32) -> String {
    if (0..10_000).contains(&year) {
        format!("{year:04}")
    } else if (-9_999..0).contains(&year) {
        format!("-{:04}", -year)
    } else if year >= 10_000 {
        format!("+{year:06}")
    } else {
        // Negate in i64: `-i32::MIN` overflows i32, and a saturating
        // negation would silently print a magnitude off by one.
        format!("-{:06}", -i64::from(year))
    }
}

/// Splits an hour-of-day value in `[0, 24)` into `(hour, minute, second)`.
fn split_hms(hours: f64) -> (u32, u32, f64) {
    let total_seconds = hours * 3600.0;
    let hour = (total_seconds / 3600.0).floor();
    let rem = total_seconds - hour * 3600.0;
    let minute = (rem / 60.0).floor();
    let second = rem - minute * 60.0;
    // hours is in [0, 24) by the `revjul` contract, so both casts are
    // in-range; clamp defensively against a `23.99999...` rounding to 24.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let h = (hour as u32).min(23);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let m = (minute as u32).min(59);
    (h, m, second)
}

/// Normalizes a parsed local ISO 8601 reading to UTC calendar fields for
/// the `date` -> `jd` JSON output, in exact arithmetic — the offset is
/// whole minutes and a `:60` leap-second reading carries one whole minute,
/// so the time of day never round-trips through a floating-point day
/// fraction (which carries just enough noise to flip an exact minute
/// boundary into `mm-1:59.999...`). The day shift is applied to the
/// integer noon JD, where [`revjul`] is exact.
fn normalize_utc_fields(
    parsed: &iso8601::Iso8601,
    cal: CalArg,
) -> Result<(oxiephemeris_core::time::CalendarDate, u32, u32, f64), CliError> {
    // 60.0 <= second < 61.0, so the subtraction is exact (Sterbenz lemma).
    let (leap_carry_min, second) = if parsed.second >= 60.0 {
        (1_i32, parsed.second - 60.0)
    } else {
        (0_i32, parsed.second)
    };
    let total_minutes = i32::from(parsed.hour) * 60 + i32::from(parsed.minute) + leap_carry_min
        - parsed.offset_minutes;
    let day_shift = total_minutes.div_euclid(24 * 60);
    let tod_minutes = total_minutes.rem_euclid(24 * 60);
    // rem_euclid(1440) is in [0, 1440), so both casts are lossless.
    #[allow(clippy::cast_sign_loss)]
    let (hour, minute) = ((tod_minutes / 60) as u32, (tod_minutes % 60) as u32);
    let noon = julday(
        cal.to_calendar(),
        parsed.year,
        parsed.month,
        parsed.day,
        12.0,
    )?;
    let (date, _noon_hours) = revjul(noon.add_days(f64::from(day_shift)), cal.to_calendar())?;
    Ok((date, hour, minute, second))
}

/// Runs `oxieph convert`.
///
/// # Errors
///
/// [`CliError::Arg`] if `--jd`/`--date` are both or neither given;
/// [`CliError::NonFiniteJd`] if `--jd` is `NaN`/infinite;
/// [`CliError::InvalidCalendarDay`] if `--date` names a day that does not
/// exist in `--cal` (e.g. `2026-02-30`); [`CliError::Iso8601`] or
/// [`CliError::Core`] propagated from parsing; [`CliError::Json`] if
/// `--json` serialization fails (should not happen for this fixed schema).
pub fn run(args: &ConvertArgs) -> Result<(), CliError> {
    match (args.jd, args.date.as_deref()) {
        (Some(_), Some(_)) => Err(CliError::Arg(
            "specify exactly one of --jd or --date, not both".to_owned(),
        )),
        (None, None) => Err(CliError::Arg("specify one of --jd or --date".to_owned())),
        (Some(jd_value), None) => jd_to_calendar(jd_value, args.cal, args.json),
        (None, Some(date_str)) => date_to_jd(date_str, args.cal, args.json),
    }
}

fn jd_to_calendar(jd_value: f64, cal: CalArg, json: bool) -> Result<(), CliError> {
    check_finite_jd(jd_value)?;
    let jd = JulianDate::from_f64(jd_value);
    let (date, hours) = revjul(jd, cal.to_calendar())?;
    let (h, m, s) = split_hms(hours);
    if json {
        let out = JdToDateJson {
            jd: jd_value,
            calendar: cal.json_name(),
            scale: "utc",
            year: date.year,
            month: date.month,
            day: date.day,
            hour: h,
            minute: m,
            second: s,
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("jd = {jd_value:.9} ({cal})");
        println!(
            "date = {}-{:02}-{:02}",
            format_year(date.year),
            date.month,
            date.day
        );
        println!("time = {h:02}:{m:02}:{s:09.6}");
    }
    Ok(())
}

fn date_to_jd(date_str: &str, cal: CalArg, json: bool) -> Result<(), CliError> {
    let parsed = iso8601::parse(date_str)?;
    check_calendar_day(
        cal.to_calendar(),
        cal.name(),
        parsed.year,
        parsed.month,
        parsed.day,
    )?;
    let hours = parsed.utc_hours();
    let jd = julday(
        cal.to_calendar(),
        parsed.year,
        parsed.month,
        parsed.day,
        hours,
    )?;
    if json {
        // Normalize the parsed local reading to UTC instead of echoing it:
        // with a non-zero UTC offset (or a `:60` leap-second reading) the
        // raw input fields would describe a different instant than
        // `jd`/`"scale":"utc"` do.
        let (date, h, m, s) = normalize_utc_fields(&parsed, cal)?;
        let out = DateToJdJson {
            year: date.year,
            month: date.month,
            day: date.day,
            hour: h,
            minute: m,
            second: s,
            jd: jd.value(),
            calendar: cal.json_name(),
            scale: "utc",
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("input = {date_str} ({cal})");
        println!("jd = {:.9}", jd.value());
    }
    Ok(())
}
