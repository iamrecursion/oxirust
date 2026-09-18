//! `oxieph progress`: a secondary-progressed chart ("a day for a year")
//! and its aspects back to the natal chart.
//!
//! # Method
//!
//! Secondary progression maps each **year** of life onto one **day** of
//! ephemeris motion after birth. With `n` years elapsed between the natal
//! epoch and the target date, the progressed planetary positions are
//! those of the sky `n` days after birth:
//!
//! ```text
//! n           = (target_tt - natal_tt) / tropical_year_days
//! progressed  = natal_tt + n days
//! ```
//!
//! The tropical year length [`TROPICAL_YEAR_DAYS`] is used for the
//! year → day conversion. Only planetary positions are progressed here;
//! progressed house angles (which need a progressed sidereal time and a
//! choice of key) are out of scope. Tropical only.
//!
//! # JSON (`--json`)
//!
//! ```text
//! { "jd_tt_natal": <f64>, "jd_tt_target": <f64>, "jd_tt_progressed": <f64>,
//!   "elapsed_years": <f64>, "natal": [PointJson...],
//!   "progressed": [PointJson...], "aspects": [CrossAspectJson...] }
//! ```

use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use oxiephemeris_chart::comparison_to_rdf;
use oxiephemeris_de::DeFile;

use crate::astro_args::HouseSystemArg;
use crate::astro_epoch::resolve_chart_epoch;
use crate::chartcommon::{
    cross_aspects_json, natal_points, person_input, planet_points_at, points_json,
    print_cross_aspects, print_positions, CrossAspectJson, PointJson,
};
use crate::convert::CalArg;
use crate::de_locate::read_de_bytes;
use crate::errors::CliError;
use crate::rdf_args::{resolve_format, OutputFormat};

/// Days in a mean tropical year, for the year → day progression step.
const TROPICAL_YEAR_DAYS: f64 = 365.242_19;

/// Seconds per day, for the progressed-instant offset.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// `oxieph progress` arguments.
#[derive(Debug, Args)]
pub struct ProgressArgs {
    /// Natal: ISO 8601 UTC epoch.
    #[arg(allow_hyphen_values = true)]
    pub date: String,
    /// Natal: geodetic latitude, degrees north.
    #[arg(long, allow_hyphen_values = true)]
    pub lat: f64,
    /// Natal: geodetic longitude, degrees east.
    #[arg(long, allow_hyphen_values = true)]
    pub lon: f64,
    /// Target date to progress to: ISO 8601 UTC epoch.
    #[arg(long, allow_hyphen_values = true)]
    pub target: String,
    /// House system (for the natal Ascendant/Midheaven).
    #[arg(long, value_enum, default_value_t = HouseSystemArg::Placidus)]
    pub system: HouseSystemArg,
    /// `UT1 - UTC`, seconds, applied to both epochs.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub dut1: f64,
    /// Calendar the dates are written in.
    #[arg(long, value_enum, default_value_t = CalArg::Gregorian)]
    pub cal: CalArg,
    /// Output format. Defaults to human-readable text.
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,
    /// Emit JSON instead of text. Superseded by `--format json`.
    #[arg(long)]
    pub json: bool,
    /// Base IRI for the emitted resources (RDF formats only).
    #[arg(long)]
    pub base_iri: Option<String>,
    /// Explicit path to a classic-binary DE file.
    #[arg(long)]
    pub de: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct ProgressJson {
    jd_tt_natal: f64,
    jd_tt_target: f64,
    jd_tt_progressed: f64,
    elapsed_years: f64,
    natal: Vec<PointJson>,
    progressed: Vec<PointJson>,
    aspects: Vec<CrossAspectJson>,
}

/// Runs `oxieph progress`.
///
/// # Errors
///
/// Propagates epoch-resolution, DE-file, apparent-place, and
/// house-geometry errors.
pub fn run(args: &ProgressArgs) -> Result<(), CliError> {
    let bytes = read_de_bytes(args.de.as_deref())?;
    let de = DeFile::parse(&bytes)?;
    let system = args.system.to_house_system();
    let fmt = resolve_format(args.format, args.json);

    if let Some(rdf_format) = fmt.as_rdf_format() {
        let comparison = oxiephemeris_chart::progression(
            &de,
            &person_input(&args.date, args.cal, args.lat, args.lon),
            &args.target,
            args.cal.to_calendar_kind(),
            system,
            args.dut1,
        )?;
        println!(
            "{}",
            comparison_to_rdf(&comparison, args.base_iri.as_deref(), rdf_format)?
        );
        return Ok(());
    }

    let natal_epoch = resolve_chart_epoch(&args.date, args.cal, args.lon, args.dut1)?;
    let target_epoch = resolve_chart_epoch(&args.target, args.cal, args.lon, args.dut1)?;
    let elapsed_days = target_epoch.jd_tt.value() - natal_epoch.jd_tt.value();
    let elapsed_years = elapsed_days / TROPICAL_YEAR_DAYS;
    // Progressed instant: natal + elapsed_years *days*.
    let progressed_jd = natal_epoch
        .jd_tt
        .add_seconds(elapsed_years * SECONDS_PER_DAY);
    let natal = natal_points(&de, &natal_epoch, args.lat.to_radians(), system)?;
    let progressed = planet_points_at(&de, progressed_jd)?;

    if matches!(fmt, OutputFormat::Json) {
        let out = ProgressJson {
            jd_tt_natal: natal_epoch.jd_tt.value(),
            jd_tt_target: target_epoch.jd_tt.value(),
            jd_tt_progressed: progressed_jd.value(),
            elapsed_years,
            natal: points_json(&natal),
            progressed: points_json(&progressed),
            aspects: cross_aspects_json(&progressed, &natal),
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("jd_tt_natal      = {:.9}", natal_epoch.jd_tt.value());
        println!("jd_tt_target     = {:.9}", target_epoch.jd_tt.value());
        println!("jd_tt_progressed = {:.9}", progressed_jd.value());
        println!("elapsed_years    = {elapsed_years:.4}");
        println!();
        print_positions("natal", &natal);
        println!();
        print_positions("progressed", &progressed);
        println!();
        println!("-- progressed -> natal aspects --");
        print_cross_aspects(&progressed, &natal, "p", "n");
    }
    Ok(())
}
