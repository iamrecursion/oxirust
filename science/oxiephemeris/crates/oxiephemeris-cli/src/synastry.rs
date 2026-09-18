//! `oxieph synastry`: cross-aspects between two natal charts.
//!
//! Each person's twelve points (ten planets + Ascendant + Midheaven,
//! [`crate::chartcommon::natal_points`]) are cross-aspected against the
//! other's with the default orb policy. Tropical only (cross-aspects are
//! sidereal-invariant; see [`crate::chartcommon`]).
//!
//! # JSON (`--json`)
//!
//! ```text
//! { "jd_tt_a": <f64>, "jd_tt_b": <f64>,
//!   "chart_a": [PointJson...], "chart_b": [PointJson...],
//!   "cross_aspects": [CrossAspectJson...] }
//! ```

use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use oxiephemeris_chart::comparison_to_rdf;
use oxiephemeris_de::DeFile;

use crate::astro_args::HouseSystemArg;
use crate::astro_epoch::resolve_chart_epoch;
use crate::chartcommon::{
    cross_aspects_json, natal_points, person_input, points_json, print_cross_aspects,
    print_positions, CrossAspectJson, PointJson,
};
use crate::convert::CalArg;
use crate::de_locate::read_de_bytes;
use crate::errors::CliError;
use crate::rdf_args::{resolve_format, OutputFormat};

/// `oxieph synastry` arguments.
#[derive(Debug, Args)]
pub struct SynastryArgs {
    /// Person A: ISO 8601 UTC epoch.
    #[arg(long, allow_hyphen_values = true)]
    pub date_a: String,
    /// Person A: geodetic latitude, degrees north.
    #[arg(long, allow_hyphen_values = true)]
    pub lat_a: f64,
    /// Person A: geodetic longitude, degrees east.
    #[arg(long, allow_hyphen_values = true)]
    pub lon_a: f64,
    /// Person B: ISO 8601 UTC epoch.
    #[arg(long, allow_hyphen_values = true)]
    pub date_b: String,
    /// Person B: geodetic latitude, degrees north.
    #[arg(long, allow_hyphen_values = true)]
    pub lat_b: f64,
    /// Person B: geodetic longitude, degrees east.
    #[arg(long, allow_hyphen_values = true)]
    pub lon_b: f64,
    /// House system (for the Ascendant/Midheaven of both charts).
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
struct SynastryJson {
    jd_tt_a: f64,
    jd_tt_b: f64,
    chart_a: Vec<PointJson>,
    chart_b: Vec<PointJson>,
    cross_aspects: Vec<CrossAspectJson>,
}

/// Runs `oxieph synastry`.
///
/// # Errors
///
/// Propagates epoch-resolution, DE-file, apparent-place, and
/// house-geometry errors.
pub fn run(args: &SynastryArgs) -> Result<(), CliError> {
    let bytes = read_de_bytes(args.de.as_deref())?;
    let de = DeFile::parse(&bytes)?;
    let system = args.system.to_house_system();
    let fmt = resolve_format(args.format, args.json);

    if let Some(rdf_format) = fmt.as_rdf_format() {
        let comparison = oxiephemeris_chart::synastry(
            &de,
            &person_input(&args.date_a, args.cal, args.lat_a, args.lon_a),
            &person_input(&args.date_b, args.cal, args.lat_b, args.lon_b),
            system,
            args.dut1,
        )?;
        println!(
            "{}",
            comparison_to_rdf(&comparison, args.base_iri.as_deref(), rdf_format)?
        );
        return Ok(());
    }

    let epoch_a = resolve_chart_epoch(&args.date_a, args.cal, args.lon_a, args.dut1)?;
    let epoch_b = resolve_chart_epoch(&args.date_b, args.cal, args.lon_b, args.dut1)?;
    let points_a = natal_points(&de, &epoch_a, args.lat_a.to_radians(), system)?;
    let points_b = natal_points(&de, &epoch_b, args.lat_b.to_radians(), system)?;

    if matches!(fmt, OutputFormat::Json) {
        let out = SynastryJson {
            jd_tt_a: epoch_a.jd_tt.value(),
            jd_tt_b: epoch_b.jd_tt.value(),
            chart_a: points_json(&points_a),
            chart_b: points_json(&points_b),
            cross_aspects: cross_aspects_json(&points_a, &points_b),
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("jd_tt_a = {:.9}", epoch_a.jd_tt.value());
        println!("jd_tt_b = {:.9}", epoch_b.jd_tt.value());
        println!();
        print_positions("chart A", &points_a);
        println!();
        print_positions("chart B", &points_b);
        println!();
        println!("-- cross-aspects (A -> B) --");
        print_cross_aspects(&points_a, &points_b, "A", "B");
    }
    Ok(())
}
