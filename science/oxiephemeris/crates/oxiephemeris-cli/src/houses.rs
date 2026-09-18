//! `oxieph houses`: house cusps and chart angles (Ascendant, Midheaven,
//! Vertex, East Point) for a geographic site and UTC epoch (design.md
//! §3.7 deliverable 2).
//!
//! # Pipeline
//!
//! `UTC -> TT` (leap seconds) and `UT1 = UTC + --dut1` (`--dut1` defaults
//! to `0`; see [`crate::astro_epoch`] for the accuracy note this implies).
//! `GAST` via [`oxiephemeris_bodies::sidereal::gast_iau2006`]; local
//! apparent sidereal time `LAST = GAST +` (east longitude,
//! [`oxiephemeris_astro::angles::local_sidereal_time`]) is the `theta`
//! every `oxiephemeris_astro` angle/house function takes. The true
//! obliquity of date is the IAU 2006 mean obliquity plus the full IAU
//! `2000A_R06` nutation in obliquity. `--alt` is accepted for CLI-surface
//! symmetry with a future topocentric refinement but is **not** used by
//! the pure-spherical-trigonometry cusp/angle formulas (no
//! dip-of-horizon term is modeled), so it does not change the output.
//!
//! `--sidereal <ayanamsha>` subtracts that ayanamsha's value from every
//! longitude this command reports (all 12 cusps, the Ascendant, MC,
//! Vertex and East Point) — see [`oxiephemeris_astro::ayanamsha`]. The
//! cusps/angles are true-equinox-of-date quantities (they derive from
//! `LAST`/the true obliquity, both nutation-inclusive), so the quantity
//! subtracted is the ayanamsha referred to the *true* equinox of date:
//! the mean-equinox ayanamsha plus the nutation in longitude `Δψ` (see
//! [`oxiephemeris_chart::epoch::sidereal_offset_true_equinox_rad`]),
//! matching `oxiephemeris-compat`'s `houses_ex` convention. The reported
//! `ayanamsha_deg` JSON/text field itself stays the plain mean-equinox
//! value (matching Swiss Ephemeris' `swe_get_ayanamsa`) — it is the
//! *shift applied to longitudes* that additionally carries `Δψ`, not
//! that field.
//!
//! # JSON schema (`--json`; stable, documented here)
//!
//! ```text
//! {
//!   "system": "placidus" | "koch" | "whole-sign" | "equal" | "porphyry"
//!             | "regiomontanus" | "campanus",
//!   "jd_tt": <f64>,
//!   "gast_deg": <f64>,
//!   "last_deg": <f64>,
//!   "obliquity_true_deg": <f64>,
//!   // "sidereal" and "ayanamsha_deg" are present together, and only
//!   // when --sidereal was given (omitted entirely for tropical output):
//!   "sidereal": "fagan-bradley" | "lahiri" | "krishnamurti" | "raman",
//!   "ayanamsha_deg": <f64>,
//!   "cusps_deg": [<f64>; 12],
//!   "ascendant_deg": <f64>,
//!   "mc_deg": <f64>,
//!   "vertex_deg": <f64>,
//!   "east_point_deg": <f64>
//! }
//! ```
//!
//! `cusps_deg[0]` is cusp 1 (bit-identical to `ascendant_deg` in radians
//! before the degree conversion), `cusps_deg[9]` is cusp 10
//! (bit-identical to `mc_deg`), matching
//! [`oxiephemeris_astro::houses::cusps`]'s indexing. `ayanamsha_deg` is
//! present exactly when `sidereal` is non-null. Numbers are full-`f64`
//! precision as in `pos`/`convert`.

use clap::Args;
use serde::Serialize;

use oxiephemeris_astro::ayanamsha::ayanamsha_rad;
use oxiephemeris_core::angle::RAD2DEG;

pub use oxiephemeris_chart::houses::{compute_houses, HousesResult};

use crate::astro_args::{HouseSystemArg, SiderealArg};
use crate::astro_epoch::{resolve_chart_epoch, ChartEpoch};
use crate::convert::CalArg;
use crate::errors::CliError;

/// `oxieph houses` arguments.
#[derive(Debug, Args)]
pub struct HousesArgs {
    /// House system.
    #[arg(long, value_enum)]
    pub system: HouseSystemArg,
    /// Observer's geodetic latitude, degrees, positive north.
    #[arg(long, allow_hyphen_values = true)]
    pub lat: f64,
    /// Observer's geodetic longitude, degrees, positive **east**.
    #[arg(long, allow_hyphen_values = true)]
    pub lon: f64,
    /// Height above the ellipsoid, meters. Accepted for CLI-surface
    /// symmetry; see the module doc — it does not affect this command's
    /// output.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub alt: f64,
    /// `UT1 - UTC`, seconds; `UT1 = UTC + dut1`. Defaults to `0` (UT1
    /// approximated by UTC; see the module doc).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub dut1: f64,
    /// ISO 8601 UTC epoch (see the `iso8601` module for the accepted
    /// grammar).
    #[arg(allow_hyphen_values = true)]
    pub date: String,
    /// Calendar `date` is written in.
    #[arg(long, value_enum, default_value_t = CalArg::Gregorian)]
    pub cal: CalArg,
    /// Sidereal (ayanamsha-shifted) longitudes instead of tropical.
    #[arg(long, value_enum)]
    pub sidereal: Option<SiderealArg>,
    /// Emit the stable JSON schema documented on this module instead of
    /// human-readable text.
    #[arg(long)]
    pub json: bool,
}

/// The stable `--json` schema; see the module doc.
#[derive(Debug, Serialize)]
struct HousesJson {
    system: &'static str,
    jd_tt: f64,
    gast_deg: f64,
    last_deg: f64,
    obliquity_true_deg: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    sidereal: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ayanamsha_deg: Option<f64>,
    cusps_deg: [f64; 12],
    ascendant_deg: f64,
    mc_deg: f64,
    vertex_deg: f64,
    east_point_deg: f64,
}

/// Runs `oxieph houses`.
///
/// # Errors
///
/// Propagates epoch-resolution errors (see
/// [`crate::astro_epoch::resolve_chart_epoch`]) and the facade's
/// house-geometry errors as [`CliError::Chart`] (see [`compute_houses`]).
pub fn run(args: &HousesArgs) -> Result<(), CliError> {
    let epoch = resolve_chart_epoch(&args.date, args.cal, args.lon, args.dut1)?;
    let phi_rad = args.lat.to_radians();
    let sidereal_kind = args.sidereal.map(SiderealArg::to_ayanamsha);

    let result = compute_houses(
        args.system.to_house_system(),
        &epoch,
        phi_rad,
        sidereal_kind,
    )?;
    let ayanamsha_deg = sidereal_kind.map(|kind| ayanamsha_rad(kind, epoch.t_tt) * RAD2DEG);

    if args.json {
        let mut cusps_deg = [0.0_f64; 12];
        for (slot, rad) in cusps_deg.iter_mut().zip(result.cusps_rad) {
            *slot = rad * RAD2DEG;
        }
        let out = HousesJson {
            system: args.system.json_name(),
            jd_tt: epoch.jd_tt.value(),
            gast_deg: epoch.gast_rad * RAD2DEG,
            last_deg: epoch.last_rad * RAD2DEG,
            obliquity_true_deg: epoch.eps_true_rad * RAD2DEG,
            sidereal: args.sidereal.map(SiderealArg::json_name),
            ayanamsha_deg,
            cusps_deg,
            ascendant_deg: result.ascendant_rad * RAD2DEG,
            mc_deg: result.mc_rad * RAD2DEG,
            vertex_deg: result.vertex_rad * RAD2DEG,
            east_point_deg: result.east_point_rad * RAD2DEG,
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        print_human(args, &epoch, &result, ayanamsha_deg);
    }
    Ok(())
}

/// Prints the human-readable `houses` report.
fn print_human(
    args: &HousesArgs,
    epoch: &ChartEpoch,
    result: &HousesResult,
    ayanamsha_deg: Option<f64>,
) {
    println!("system = {}", args.system.name());
    println!("jd_tt = {:.9}", epoch.jd_tt.value());
    println!("gast = {:.9} deg", epoch.gast_rad * RAD2DEG);
    println!("last = {:.9} deg", epoch.last_rad * RAD2DEG);
    println!("obliquity_true = {:.9} deg", epoch.eps_true_rad * RAD2DEG);
    match (args.sidereal, ayanamsha_deg) {
        (Some(kind), Some(deg)) => {
            println!("sidereal = {} (ayanamsha = {deg:.9} deg)", kind.json_name());
        }
        _ => println!("sidereal = none (tropical)"),
    }
    for (index, cusp_rad) in result.cusps_rad.iter().enumerate() {
        println!("cusp {:>2} = {:.9} deg", index + 1, *cusp_rad * RAD2DEG);
    }
    println!("ascendant = {:.9} deg", result.ascendant_rad * RAD2DEG);
    println!("mc = {:.9} deg", result.mc_rad * RAD2DEG);
    println!("vertex = {:.9} deg", result.vertex_rad * RAD2DEG);
    println!("east_point = {:.9} deg", result.east_point_rad * RAD2DEG);
}
