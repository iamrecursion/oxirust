//! `oxieph composite`: the midpoint composite chart of two natal charts.
//!
//! Each composite point is the near (short-arc) midpoint of the two
//! like-named natal points (ten planets + Ascendant + Midheaven,
//! [`crate::chartcommon::natal_points`]); its speed is the mean of the
//! two natal speeds (a convenience for the retrograde tag and the
//! applying/separating classification of the composite's internal
//! aspects). Tropical only.
//!
//! # JSON (`--json`)
//!
//! ```text
//! { "jd_tt_a": <f64>, "jd_tt_b": <f64>,
//!   "composite": [PointJson...], "aspects": [AspectJson...] }
//! ```

use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use oxiephemeris_astro::aspects::{find_aspect, OrbPolicy};
use oxiephemeris_astro::midpoints::midpoint;
use oxiephemeris_chart::composite_to_rdf;
use oxiephemeris_core::angle::RAD2DEG;
use oxiephemeris_de::DeFile;

use crate::astro_args::HouseSystemArg;
use crate::astro_epoch::resolve_chart_epoch;
use crate::chartcommon::{
    natal_points, person_input, points_json, print_positions, ChartPoint, PointJson,
};
use crate::convert::CalArg;
use crate::de_locate::read_de_bytes;
use crate::errors::CliError;
use crate::rdf_args::{resolve_format, OutputFormat};

/// `oxieph composite` arguments.
#[derive(Debug, Args)]
pub struct CompositeArgs {
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
    /// House system (for each chart's Ascendant/Midheaven).
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
struct AspectJson {
    body1: &'static str,
    body2: &'static str,
    aspect: &'static str,
    exact_angle_deg: f64,
    offset_deg: f64,
    applying: bool,
}

#[derive(Debug, Serialize)]
struct CompositeJson {
    jd_tt_a: f64,
    jd_tt_b: f64,
    composite: Vec<PointJson>,
    aspects: Vec<AspectJson>,
}

/// Runs `oxieph composite`.
///
/// # Errors
///
/// Propagates epoch-resolution, DE-file, apparent-place, and
/// house-geometry errors.
pub fn run(args: &CompositeArgs) -> Result<(), CliError> {
    let bytes = read_de_bytes(args.de.as_deref())?;
    let de = DeFile::parse(&bytes)?;
    let system = args.system.to_house_system();
    let fmt = resolve_format(args.format, args.json);

    if let Some(rdf_format) = fmt.as_rdf_format() {
        // The two source charts' IRIs are minted under `base_iri` inside the
        // builder, so the serializer must be handed the same base.
        let comp = oxiephemeris_chart::composite(
            &de,
            &person_input(&args.date_a, args.cal, args.lat_a, args.lon_a),
            &person_input(&args.date_b, args.cal, args.lat_b, args.lon_b),
            system,
            args.dut1,
            args.base_iri.as_deref(),
        )?;
        println!(
            "{}",
            composite_to_rdf(&comp, args.base_iri.as_deref(), rdf_format)?
        );
        return Ok(());
    }

    let epoch_a = resolve_chart_epoch(&args.date_a, args.cal, args.lon_a, args.dut1)?;
    let epoch_b = resolve_chart_epoch(&args.date_b, args.cal, args.lon_b, args.dut1)?;
    let points_a = natal_points(&de, &epoch_a, args.lat_a.to_radians(), system)?;
    let points_b = natal_points(&de, &epoch_b, args.lat_b.to_radians(), system)?;

    // `natal_points` returns the same points in the same order for both
    // charts, so zipping pairs them by identity.
    let composite: Vec<ChartPoint> = points_a
        .iter()
        .zip(points_b.iter())
        .map(|(a, b)| ChartPoint {
            name: a.name,
            lon_rad: midpoint(a.lon_rad, b.lon_rad),
            speed_rad_per_day: f64::midpoint(a.speed_rad_per_day, b.speed_rad_per_day),
        })
        .collect();

    let aspects = internal_aspects(&composite);

    if matches!(fmt, OutputFormat::Json) {
        let out = CompositeJson {
            jd_tt_a: epoch_a.jd_tt.value(),
            jd_tt_b: epoch_b.jd_tt.value(),
            composite: points_json(&composite),
            aspects,
        };
        println!("{}", serde_json::to_string(&out)?);
    } else {
        println!("jd_tt_a = {:.9}", epoch_a.jd_tt.value());
        println!("jd_tt_b = {:.9}", epoch_b.jd_tt.value());
        println!();
        print_positions("composite (midpoint)", &composite);
        println!();
        println!("-- composite aspects --");
        if aspects.is_empty() {
            println!("(none within the default orbs)");
        }
        for a in &aspects {
            let applying = if a.applying { "applying" } else { "separating" };
            println!(
                "{:<8} {:<14} {:<8} offset = {:>9.4} deg ({applying})",
                a.body1, a.aspect, a.body2, a.offset_deg,
            );
        }
    }
    Ok(())
}

/// The internal aspects among a chart's points (each unordered pair
/// `i < j` tested once).
fn internal_aspects(points: &[ChartPoint]) -> Vec<AspectJson> {
    let policy = OrbPolicy::default();
    let mut aspects = Vec::new();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let a = &points[i];
            let b = &points[j];
            if let Some(hit) = find_aspect(
                a.lon_rad,
                a.speed_rad_per_day,
                b.lon_rad,
                b.speed_rad_per_day,
                &policy,
            ) {
                aspects.push(AspectJson {
                    body1: a.name,
                    body2: b.name,
                    aspect: hit.aspect.kind.name(),
                    exact_angle_deg: hit.aspect.exact_angle_rad * RAD2DEG,
                    offset_deg: hit.offset_rad * RAD2DEG,
                    applying: hit.applying,
                });
            }
        }
    }
    aspects
}
