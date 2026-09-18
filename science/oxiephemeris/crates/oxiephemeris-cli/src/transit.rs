//! `oxieph transit`: aspects of the transiting planets at a given date to
//! a natal chart.
//!
//! The ten transiting planets ([`crate::chartcommon::transit_points`]) at
//! `--transit` are cross-aspected against the twelve natal points (ten
//! planets + Ascendant + Midheaven) of the birth data. The relative speed
//! is carried entirely by the moving transiting side (the natal points
//! are fixed), so applying/separating reads conventionally. Tropical only.
//!
//! # JSON (`--json`)
//!
//! ```text
//! { "jd_tt_natal": <f64>, "jd_tt_transit": <f64>,
//!   "natal": [PointJson...], "transiting": [PointJson...],
//!   "aspects": [CrossAspectJson...] }
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
    print_positions, transit_points, CrossAspectJson, PointJson,
};
use crate::convert::CalArg;
use crate::de_locate::read_de_bytes;
use crate::errors::CliError;
use crate::rdf_args::{resolve_format, OutputFormat};

/// `oxieph transit` arguments.
#[derive(Debug, Args)]
pub struct TransitArgs {
    /// Natal: ISO 8601 UTC epoch.
    #[arg(allow_hyphen_values = true)]
    pub date: String,
    /// Natal: geodetic latitude, degrees north.
    #[arg(long, allow_hyphen_values = true)]
    pub lat: f64,
    /// Natal: geodetic longitude, degrees east.
    #[arg(long, allow_hyphen_values = true)]
    pub lon: f64,
    /// Transiting instant: ISO 8601 UTC epoch.
    #[arg(long, allow_hyphen_values = true)]
    pub transit: String,
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
struct TransitJson {
    jd_tt_natal: f64,
    jd_tt_transit: f64,
    natal: Vec<PointJson>,
    transiting: Vec<PointJson>,
    aspects: Vec<CrossAspectJson>,
}

/// Runs `oxieph transit`.
///
/// # Errors
///
/// Propagates epoch-resolution, DE-file, apparent-place, and
/// house-geometry errors.
pub fn run(args: &TransitArgs) -> Result<(), CliError> {
    let bytes = read_de_bytes(args.de.as_deref())?;
    let de = DeFile::parse(&bytes)?;
    let system = args.system.to_house_system();
    let fmt = resolve_format(args.format, args.json);

    if let Some(rdf_format) = fmt.as_rdf_format() {
        let comparison = oxiephemeris_chart::transit(
            &de,
            &person_input(&args.date, args.cal, args.lat, args.lon),
            &args.transit,
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
    // The transiting instant needs no geographic longitude (geocentric
    // planets); pass 0 for the (unused) sidereal-time longitude.
    let transit_epoch = resolve_chart_epoch(&args.transit, args.cal, 0.0, args.dut1)?;
    let natal = natal_points(&de, &natal_epoch, args.lat.to_radians(), system)?;
    let transiting = transit_points(&de, &transit_epoch)?;

    if matches!(fmt, OutputFormat::Json) {
        let out = TransitJson {
            jd_tt_natal: natal_epoch.jd_tt.value(),
            jd_tt_transit: transit_epoch.jd_tt.value(),
            natal: points_json(&natal),
            transiting: points_json(&transiting),
            aspects: cross_aspects_json(&transiting, &natal),
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("jd_tt_natal   = {:.9}", natal_epoch.jd_tt.value());
        println!("jd_tt_transit = {:.9}", transit_epoch.jd_tt.value());
        println!();
        print_positions("natal", &natal);
        println!();
        print_positions("transiting", &transiting);
        println!();
        println!("-- transit -> natal aspects --");
        print_cross_aspects(&transiting, &natal, "t", "n");
    }
    Ok(())
}
