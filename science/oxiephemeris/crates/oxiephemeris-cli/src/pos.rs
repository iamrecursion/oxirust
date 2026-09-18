//! `oxieph pos`: apparent (or astrometric-style) place of a solar-system
//! body.
//!
//! # JSON schema (`--json`; stable, documented here)
//!
//! ```text
//! {
//!   "body": "Mars",
//!   "jd_tt": 2451545.0,
//!   "frame": "ecliptic" | "equatorial" | "j2000",
//!   "center": "geocentric" | "heliocentric" | "barycentric",
//!   // ecliptic frame:
//!   "lon_deg": <f64>, "lat_deg": <f64>,
//!   "lon_speed_deg_per_day": <f64>, "lat_speed_deg_per_day": <f64>,
//!   // equatorial / j2000 frame (mutually exclusive with the above):
//!   "ra_deg": <f64>, "dec_deg": <f64>,
//!   "ra_speed_deg_per_day": <f64>, "dec_speed_deg_per_day": <f64>,
//!   "distance_au": <f64>,
//!   "light_time_days": <f64>
//! }
//! ```
//!
//! Exactly one of the `{lon,lat}` or `{ra,dec}` pairs (with their matching
//! `*_speed_deg_per_day` fields) is present, selected by `frame`. Numbers
//! are serialized at full `f64` precision (`serde_json`'s round-trip
//! shortest representation).
//!
//! # `pos --all` JSON schema (`--json`; stable, documented here)
//!
//! `--all` computes every body in [`crate::body::ALL_BODIES`] (Sun,
//! Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto,
//! in that fixed order) at one epoch, under one set of `--frame`/
//! `--center`/`--no-aberration`/`--no-deflection`/`--scale` options, as a
//! single JSON object wrapping a `bodies` array of per-body entries that
//! share the single-body schema above (minus the fields hoisted to the
//! wrapper):
//!
//! ```text
//! {
//!   "jd_tt": <f64>,
//!   "frame": "ecliptic" | "equatorial" | "j2000",
//!   "center": "geocentric" | "heliocentric" | "barycentric",
//!   "bodies": [
//!     {
//!       "body": "Sun",
//!       // ecliptic frame:
//!       "lon_deg": <f64>, "lat_deg": <f64>,
//!       "lon_speed_deg_per_day": <f64>, "lat_speed_deg_per_day": <f64>,
//!       // equatorial / j2000 frame (mutually exclusive with the above):
//!       "ra_deg": <f64>, "dec_deg": <f64>,
//!       "ra_speed_deg_per_day": <f64>, "dec_speed_deg_per_day": <f64>,
//!       "distance_au": <f64>,
//!       "light_time_days": <f64>
//!     },
//!     ... (10 entries total, one per `ALL_BODIES` in its listed order)
//!   ]
//! }
//! ```
//!
//! `--all` takes the epoch as its sole positional argument (no body
//! name): `oxieph pos --all 2026-07-05T12:00:00Z`.

use clap::{Args, ValueEnum};
use serde::Serialize;

use oxiephemeris_bodies::apparent::apparent;
use oxiephemeris_bodies::{Center, Frame, Options};
use oxiephemeris_core::angle::RAD2DEG;
use oxiephemeris_core::time::{julday, leap::utc_to_tt, Calendar};
use oxiephemeris_de::DeFile;

use crate::body::{parse_body, ALL_BODIES};
use crate::de_locate::read_de_bytes;
use crate::errors::{check_calendar_day, CliError};
use crate::iso8601;

/// Output reference frame selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FrameArg {
    /// Apparent geocentric ecliptic-of-date (true equinox); SE-style
    /// default.
    Ecliptic,
    /// Apparent right ascension/declination of date (true equator and
    /// equinox).
    Equatorial,
    /// Astrometric-style ICRS right ascension/declination (aberration and
    /// deflection are still applied unless `--no-aberration`/
    /// `--no-deflection` are given; only the *frame* is J2000/ICRS).
    J2000,
}

impl FrameArg {
    const fn to_frame(self) -> Frame {
        match self {
            Self::Ecliptic => Frame::EclipticTrueOfDate,
            Self::Equatorial => Frame::TrueOfDate,
            Self::J2000 => Frame::Icrs,
        }
    }

    /// `true` for the two equatorial-style frames (right ascension /
    /// declination output), `false` for ecliptic longitude/latitude.
    const fn is_equatorial(self) -> bool {
        !matches!(self, Self::Ecliptic)
    }

    const fn json_name(self) -> &'static str {
        match self {
            Self::Ecliptic => "ecliptic",
            Self::Equatorial => "equatorial",
            Self::J2000 => "j2000",
        }
    }
}

/// Observation center selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CenterArg {
    /// Observer at the Earth's center of mass (default).
    Geo,
    /// Observer at the Sun's center of mass (astrometric-style output).
    Helio,
    /// Observer at the solar-system barycenter (astrometric-style output).
    Bary,
}

impl CenterArg {
    const fn to_center(self) -> Center {
        match self {
            Self::Geo => Center::Geocentric,
            Self::Helio => Center::Heliocentric,
            Self::Bary => Center::Barycentric,
        }
    }

    const fn json_name(self) -> &'static str {
        match self {
            Self::Geo => "geocentric",
            Self::Helio => "heliocentric",
            Self::Bary => "barycentric",
        }
    }
}

/// Time-scale selector for the epoch argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ScaleArg {
    /// Interpret the epoch as UTC (default), converting to TT via the
    /// leap-second table (requires 1972 or later).
    Utc,
    /// Interpret the epoch as Terrestrial Time directly, bypassing leap
    /// seconds (needed before 1972).
    Tt,
}

/// `oxieph pos` arguments.
///
/// # `body`/`date` positional handling (`--all`)
///
/// Both positionals are declared `Option` so that `--all` can be invoked
/// with a single positional value (the epoch) and no body name — clap
/// fills declared-optional positionals strictly left-to-right, so one
/// token given lands in `body`. [`run`] resolves the two into "which
/// token is the date" itself (see its implementation) and reports a
/// friendly [`CliError::Arg`] for every other combination (neither given,
/// or both given together with `--all`). The single-body path (`--all`
/// absent, both positionals given) is unaffected: its behavior and
/// output are byte-identical to when `body`/`date` were plain required
/// `String`s.
#[derive(Debug, Args)]
#[allow(clippy::struct_excessive_bools)] // clap argument struct: each bool is a CLI switch
pub struct PosArgs {
    /// Body name, case-insensitive: sun, moon, mercury, venus, mars,
    /// jupiter, saturn, uranus, neptune, pluto. Omit when `--all` is
    /// given.
    pub body: Option<String>,
    /// ISO 8601 epoch (see the `iso8601` module for the accepted grammar).
    /// Negative (BCE) years are accepted, e.g. `-0001-06-15T00:00:00`.
    /// With `--all`, pass the epoch as the sole positional (it is parsed
    /// out of `body` — see the struct doc).
    #[arg(allow_hyphen_values = true)]
    pub date: Option<String>,
    /// Compute every supported body (see [`crate::body::ALL_BODIES`])
    /// instead of a single one; mutually exclusive with the `body`
    /// positional (pass only the epoch).
    #[arg(long)]
    pub all: bool,
    /// Output reference frame.
    #[arg(long, value_enum, default_value_t = FrameArg::Ecliptic)]
    pub frame: FrameArg,
    /// Disable annual aberration (only affects `--center geo`).
    #[arg(long)]
    pub no_aberration: bool,
    /// Disable solar gravitational light deflection (only affects
    /// `--center geo`).
    #[arg(long)]
    pub no_deflection: bool,
    /// Observation center.
    #[arg(long, value_enum, default_value_t = CenterArg::Geo)]
    pub center: CenterArg,
    /// Emit the stable JSON schema documented on this module instead of
    /// human-readable text.
    #[arg(long)]
    pub json: bool,
    /// Explicit path to a classic-binary DE file (overrides `$OXIEPH_DE`
    /// and the default `data/de440/linux_p1550p2650.440`).
    #[arg(long)]
    pub de: Option<std::path::PathBuf>,
    /// Time scale of `date`.
    #[arg(long, value_enum, default_value_t = ScaleArg::Utc)]
    pub scale: ScaleArg,
}

/// The stable `--json` schema. Exactly one of the ecliptic or equatorial
/// field pairs is populated, per [`FrameArg::is_equatorial`].
#[derive(Debug, Serialize)]
struct PosJson {
    body: &'static str,
    jd_tt: f64,
    frame: &'static str,
    center: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    lon_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lat_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lon_speed_deg_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lat_speed_deg_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ra_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dec_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ra_speed_deg_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dec_speed_deg_per_day: Option<f64>,
    distance_au: f64,
    light_time_days: f64,
}

/// Resolves the `body`/`date` positionals (see [`PosArgs`]'s struct doc)
/// for the single-body path: both must be present.
///
/// # Errors
///
/// [`CliError::Arg`] if either positional is missing.
fn single_body_args(args: &PosArgs) -> Result<(&str, &str), CliError> {
    match (args.body.as_deref(), args.date.as_deref()) {
        (Some(body), Some(date)) => Ok((body, date)),
        _ => Err(CliError::Arg(
            "a body name and a date are required (or pass --all with just a date)".to_owned(),
        )),
    }
}

/// Runs `oxieph pos`.
///
/// # Errors
///
/// Propagates argument, date-parsing, time-scale, DE-file, and
/// apparent-place errors as [`CliError`] (see the variant docs).
pub fn run(args: &PosArgs) -> Result<(), CliError> {
    if args.all {
        return run_all(args);
    }
    let (body_str, date_str) = single_body_args(args)?;
    let (target, body_name) = parse_body(body_str).map_err(CliError::Arg)?;

    let parsed = iso8601::parse(date_str)?;
    check_calendar_day(
        Calendar::Gregorian,
        "Gregorian",
        parsed.year,
        parsed.month,
        parsed.day,
    )?;
    let hours = parsed.utc_hours();
    let jd_raw = julday(
        Calendar::Gregorian,
        parsed.year,
        parsed.month,
        parsed.day,
        hours,
    )?;
    let jd_tt = match args.scale {
        ScaleArg::Utc => utc_to_tt(jd_raw).map_err(|_| CliError::PreLeapTableUtc)?,
        ScaleArg::Tt => jd_raw,
    };

    let bytes = read_de_bytes(args.de.as_deref())?;
    let de = DeFile::parse(&bytes)?;

    // Knobs the CLI does not expose (e.g. the nutation model) stay at
    // the pipeline defaults (full IAU 2000A_R06 nutation).
    let mut opts = Options::default();
    opts.center = args.center.to_center();
    opts.frame = args.frame.to_frame();
    opts.aberration = !args.no_aberration;
    opts.deflection = !args.no_deflection;
    opts.with_speed = true;

    let place = apparent(&de, target, jd_tt, opts)?;
    let rates = place.rates;
    let lon_deg = place.lon_rad * RAD2DEG;
    let lat_deg = place.lat_rad * RAD2DEG;
    let lon_speed = rates.map(|r| r.lon_rad_per_day * RAD2DEG);
    let lat_speed = rates.map(|r| r.lat_rad_per_day * RAD2DEG);

    if args.json {
        let equatorial = args.frame.is_equatorial();
        let out = PosJson {
            body: body_name,
            jd_tt: jd_tt.value(),
            frame: args.frame.json_name(),
            center: args.center.json_name(),
            lon_deg: (!equatorial).then_some(lon_deg),
            lat_deg: (!equatorial).then_some(lat_deg),
            lon_speed_deg_per_day: (!equatorial).then_some(lon_speed).flatten(),
            lat_speed_deg_per_day: (!equatorial).then_some(lat_speed).flatten(),
            ra_deg: equatorial.then_some(lon_deg),
            dec_deg: equatorial.then_some(lat_deg),
            ra_speed_deg_per_day: equatorial.then_some(lon_speed).flatten(),
            dec_speed_deg_per_day: equatorial.then_some(lat_speed).flatten(),
            distance_au: place.r_au,
            light_time_days: place.light_time_days,
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        print_human(
            body_name,
            args,
            jd_tt.value(),
            lon_deg,
            lat_deg,
            lon_speed,
            lat_speed,
            place.r_au,
            rates.map(|r| r.r_au_per_day),
            place.light_time_days,
        );
    }
    Ok(())
}

/// Prints the human-readable `pos` report.
#[allow(clippy::too_many_arguments)]
fn print_human(
    body_name: &str,
    args: &PosArgs,
    jd_tt: f64,
    a_deg: f64,
    b_deg: f64,
    a_speed: Option<f64>,
    b_speed: Option<f64>,
    distance_au: f64,
    r_speed: Option<f64>,
    light_time_days: f64,
) {
    let (a_label, b_label) = if args.frame.is_equatorial() {
        ("ra", "dec")
    } else {
        ("lon", "lat")
    };
    println!("body = {body_name}");
    println!("jd_tt = {jd_tt:.9}");
    println!("frame = {}", args.frame.json_name());
    println!("center = {}", args.center.json_name());
    println!(
        "{a_label} = {a_deg:.9} deg  (speed {:.9} deg/day)",
        a_speed.unwrap_or(0.0)
    );
    println!(
        "{b_label} = {b_deg:.9} deg  (speed {:.9} deg/day)",
        b_speed.unwrap_or(0.0)
    );
    println!(
        "distance_au = {distance_au:.12}  (speed {:.12} au/day)",
        r_speed.unwrap_or(0.0)
    );
    println!("light_time_days = {light_time_days:.9}");
}

/// One `pos --all --json` `bodies[]` entry: identical field set to
/// [`PosJson`] minus the epoch/frame/center fields hoisted to the
/// wrapper object.
#[derive(Debug, Serialize)]
struct PosAllEntryJson {
    body: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    lon_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lat_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lon_speed_deg_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lat_speed_deg_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ra_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dec_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ra_speed_deg_per_day: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dec_speed_deg_per_day: Option<f64>,
    distance_au: f64,
    light_time_days: f64,
}

/// The stable `pos --all --json` schema; see the module doc.
#[derive(Debug, Serialize)]
struct PosAllJson {
    jd_tt: f64,
    frame: &'static str,
    center: &'static str,
    bodies: Vec<PosAllEntryJson>,
}

/// Runs `oxieph pos --all`.
///
/// # Errors
///
/// Same as [`run`], plus [`CliError::Arg`] if the epoch positional is
/// missing, or if both `body` and `date` positionals were given together
/// with `--all` (ambiguous: `--all` takes only an epoch).
fn run_all(args: &PosArgs) -> Result<(), CliError> {
    if args.body.is_some() && args.date.is_some() {
        return Err(CliError::Arg(
            "--all takes only a date argument, not a body name".to_owned(),
        ));
    }
    let date_str = args
        .body
        .as_deref()
        .or(args.date.as_deref())
        .ok_or_else(|| CliError::Arg("a date is required".to_owned()))?;

    let parsed = iso8601::parse(date_str)?;
    check_calendar_day(
        Calendar::Gregorian,
        "Gregorian",
        parsed.year,
        parsed.month,
        parsed.day,
    )?;
    let hours = parsed.utc_hours();
    let jd_raw = julday(
        Calendar::Gregorian,
        parsed.year,
        parsed.month,
        parsed.day,
        hours,
    )?;
    let jd_tt = match args.scale {
        ScaleArg::Utc => utc_to_tt(jd_raw).map_err(|_| CliError::PreLeapTableUtc)?,
        ScaleArg::Tt => jd_raw,
    };

    let bytes = read_de_bytes(args.de.as_deref())?;
    let de = DeFile::parse(&bytes)?;

    let mut opts = Options::default();
    opts.center = args.center.to_center();
    opts.frame = args.frame.to_frame();
    opts.aberration = !args.no_aberration;
    opts.deflection = !args.no_deflection;
    opts.with_speed = true;
    let equatorial = args.frame.is_equatorial();

    let mut places = Vec::with_capacity(ALL_BODIES.len());
    for (target, name) in ALL_BODIES {
        places.push((name, apparent(&de, target, jd_tt, opts)?));
    }

    if args.json {
        let bodies = places
            .iter()
            .map(|(name, place)| {
                let rates = place.rates;
                let lon_deg = place.lon_rad * RAD2DEG;
                let lat_deg = place.lat_rad * RAD2DEG;
                let lon_speed = rates.map(|r| r.lon_rad_per_day * RAD2DEG);
                let lat_speed = rates.map(|r| r.lat_rad_per_day * RAD2DEG);
                PosAllEntryJson {
                    body: name,
                    lon_deg: (!equatorial).then_some(lon_deg),
                    lat_deg: (!equatorial).then_some(lat_deg),
                    lon_speed_deg_per_day: (!equatorial).then_some(lon_speed).flatten(),
                    lat_speed_deg_per_day: (!equatorial).then_some(lat_speed).flatten(),
                    ra_deg: equatorial.then_some(lon_deg),
                    dec_deg: equatorial.then_some(lat_deg),
                    ra_speed_deg_per_day: equatorial.then_some(lon_speed).flatten(),
                    dec_speed_deg_per_day: equatorial.then_some(lat_speed).flatten(),
                    distance_au: place.r_au,
                    light_time_days: place.light_time_days,
                }
            })
            .collect();
        let out = PosAllJson {
            jd_tt: jd_tt.value(),
            frame: args.frame.json_name(),
            center: args.center.json_name(),
            bodies,
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("jd_tt = {:.9}", jd_tt.value());
        println!("frame = {}", args.frame.json_name());
        println!("center = {}", args.center.json_name());
        let (a_label, b_label) = if equatorial {
            ("ra_deg", "dec_deg")
        } else {
            ("lon_deg", "lat_deg")
        };
        println!(
            "{:<9}{a_label:>16}{b_label:>16}{:>16}{:>17}",
            "body", "distance_au", "light_time_d"
        );
        for (name, place) in &places {
            let a_deg = place.lon_rad * RAD2DEG;
            let b_deg = place.lat_rad * RAD2DEG;
            let distance_au = place.r_au;
            let light_time_days = place.light_time_days;
            println!(
                "{name:<9}{a_deg:>16.9}{b_deg:>16.9}{distance_au:>16.9}{light_time_days:>17.9}"
            );
        }
    }
    Ok(())
}
