//! `oxieph chart`: a full chart in one call — `pos --all` plus house
//! cusps/angles plus lunar nodes plus an aspect table, enriched with the
//! astrological presentation layer: zodiac sign/degree notation, retrograde
//! flags, house occupancy, essential dignities, the element/modality
//! distribution, the Lots of Fortune/Spirit, and equatorial declinations.
//!
//! # Pipeline
//!
//! The entire chart is computed by [`oxiephemeris_chart::natal_chart`] from
//! a [`NatalRequest`] built out of the CLI arguments; this command only
//! renders the returned [`ChartResource`] as text, JSON
//! ([`oxiephemeris_chart::chart_to_json`]), or RDF
//! ([`oxiephemeris_chart::chart_to_rdf`]). The compute — epoch resolution,
//! bodies, houses, nodes, Lots, dignities, distribution, and aspects — is
//! the facade's single source of truth, shared with the Python and WASM
//! bindings.
//!
//! [`ChartResource`]: oxiephemeris_rdf::model::ChartResource
//!
//! # Sign / house / dignity conventions
//!
//! Sign notation and house occupancy use the *displayed* longitude (so
//! they follow `--sidereal` when given). **Essential dignities** are a
//! tropical technique and are always computed from the tropical longitude,
//! irrespective of `--sidereal`; the rulership scheme is selectable with
//! `--rulership`. **Declinations** are true equatorial quantities and are
//! independent of the zodiac choice.
//!
//! # JSON schema (`--json`)
//!
//! The stable JSON schema is [`oxiephemeris_chart::json::ChartJson`], shared
//! verbatim with the Python and WASM bindings.

use std::path::PathBuf;

use clap::{Args, ValueEnum};

use oxiephemeris_astro::dignities::RulershipScheme;
use oxiephemeris_de::DeFile;
use oxiephemeris_rdf::model as rdf_model;

use oxiephemeris_chart::{chart_to_json, chart_to_rdf, natal_chart, NatalRequest};

use crate::astro_args::{HouseSystemArg, SiderealArg};
use crate::chart_render::format_sign;
use crate::convert::CalArg;
use crate::de_locate::read_de_bytes;
use crate::errors::{check_longitude, CliError};
use crate::rdf_args::{resolve_format, OutputFormat};

/// Which rulership scheme the dignity tables use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RulershipArg {
    /// Classical seven-planet rulerships (the dignity tables' origin).
    Traditional,
    /// Modern scheme (Scorpio → Pluto, Aquarius → Uranus, Pisces →
    /// Neptune).
    Modern,
}

impl RulershipArg {
    const fn to_scheme(self) -> RulershipScheme {
        match self {
            Self::Traditional => RulershipScheme::Traditional,
            Self::Modern => RulershipScheme::Modern,
        }
    }
}

/// `oxieph chart` arguments.
#[derive(Debug, Args)]
pub struct ChartArgs {
    /// ISO 8601 UTC epoch (see the `iso8601` module for the accepted
    /// grammar).
    #[arg(allow_hyphen_values = true)]
    pub date: String,
    /// Observer's geodetic latitude, degrees, positive north.
    #[arg(long, allow_hyphen_values = true)]
    pub lat: f64,
    /// Observer's geodetic longitude, degrees, positive **east**.
    #[arg(long, allow_hyphen_values = true)]
    pub lon: f64,
    /// Height above the ellipsoid, meters. Accepted for CLI-surface
    /// symmetry; `chart` is geocentric and does not use it.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub alt: f64,
    /// House system.
    #[arg(long, value_enum, default_value_t = HouseSystemArg::Placidus)]
    pub system: HouseSystemArg,
    /// Sidereal (ayanamsha-shifted) longitudes instead of tropical.
    #[arg(long, value_enum)]
    pub sidereal: Option<SiderealArg>,
    /// Rulership scheme for essential dignities.
    #[arg(long, value_enum, default_value_t = RulershipArg::Traditional)]
    pub rulership: RulershipArg,
    /// `UT1 - UTC`, seconds; `UT1 = UTC + dut1`. Defaults to `0`.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub dut1: f64,
    /// Calendar `date` is written in.
    #[arg(long, value_enum, default_value_t = CalArg::Gregorian)]
    pub cal: CalArg,
    /// Output format. Defaults to human-readable text.
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,
    /// Emit the stable JSON schema instead of text. Superseded by
    /// `--format json`; kept for compatibility.
    #[arg(long)]
    pub json: bool,
    /// Base IRI for the emitted chart resource (RDF formats only).
    /// Defaults to the published instance namespace. The *vocabulary*
    /// namespaces are fixed and unaffected.
    #[arg(long)]
    pub base_iri: Option<String>,
    /// Use this exact IRI for the chart resource instead of minting one
    /// from its defining inputs (RDF formats only).
    #[arg(long)]
    pub chart_iri: Option<String>,
    /// Explicit path to a classic-binary DE file (overrides `$OXIEPH_DE`
    /// and the default `data/de440/linux_p1550p2650.440`).
    #[arg(long)]
    pub de: Option<PathBuf>,
}

impl ChartArgs {
    /// Builds the facade [`NatalRequest`] this command computes from.
    fn to_request(&self) -> NatalRequest {
        NatalRequest {
            date: self.date.clone(),
            cal: self.cal.to_calendar_kind(),
            lat_deg: self.lat,
            lon_deg: self.lon,
            alt_m: self.alt,
            dut1_s: self.dut1,
            system: self.system.to_house_system(),
            sidereal: self.sidereal.map(SiderealArg::to_ayanamsha),
            rulership: self.rulership.to_scheme(),
        }
    }
}

/// Runs `oxieph chart`.
///
/// # Errors
///
/// [`CliError::InvalidLongitude`] for an out-of-range `--lon`,
/// [`CliError::DeFileMissing`]/[`CliError::De`] for the ephemeris, and
/// [`CliError::Chart`] wrapping any facade compute/serialize failure.
pub fn run(args: &ChartArgs) -> Result<(), CliError> {
    // Fires before any DE-file access so an out-of-range longitude fails
    // the same way regardless of whether a DE fixture is present, and keeps
    // the CLI's `--lon`-naming message.
    check_longitude(args.lon)?;

    let bytes = read_de_bytes(args.de.as_deref())?;
    let de = DeFile::parse(&bytes)?;

    let chart = natal_chart(&de, &args.to_request())?;

    match resolve_format(args.format, args.json) {
        OutputFormat::Json => println!("{}", chart_to_json(&chart)?),
        OutputFormat::Text => print_human(args, &chart),
        format => {
            let Some(rdf_format) = format.as_rdf_format() else {
                // `resolve_format` only yields Text/Json/Turtle/NTriples,
                // and both non-RDF arms are handled above.
                return Err(CliError::Arg(
                    "chart called with an unsupported output format".to_owned(),
                ));
            };
            println!(
                "{}",
                chart_to_rdf(
                    &chart,
                    args.base_iri.as_deref(),
                    args.chart_iri.as_deref(),
                    rdf_format,
                )?
            );
        }
    }
    Ok(())
}

/// Finds an angle longitude (degrees) by kind, defaulting to `0.0`.
fn angle_deg(angles: &[rdf_model::AngleRecord], kind: rdf_model::AngleKind) -> f64 {
    angles
        .iter()
        .find(|a| a.kind == kind)
        .map_or(0.0, |a| a.lon_deg)
}

/// Finds a node longitude (degrees) by kind, defaulting to `0.0`.
fn node_deg(nodes: &[rdf_model::NodeRecord], kind: rdf_model::NodeKind) -> f64 {
    nodes
        .iter()
        .find(|n| n.kind == kind)
        .map_or(0.0, |n| n.lon_deg)
}

/// Finds a lot longitude (degrees) by kind, defaulting to `0.0`.
fn lot_deg(lots: &[rdf_model::LotRecord], kind: rdf_model::LotKind) -> f64 {
    lots.iter()
        .find(|l| l.kind == kind)
        .map_or(0.0, |l| l.lon_deg)
}

/// The lower-case display label of a dignity tier.
const fn tier_label(tier: rdf_model::DignityTier) -> &'static str {
    match tier {
        rdf_model::DignityTier::Domicile => "domicile",
        rdf_model::DignityTier::Exaltation => "exaltation",
        rdf_model::DignityTier::Triplicity => "triplicity",
        rdf_model::DignityTier::Term => "term",
        rdf_model::DignityTier::Face => "face",
        rdf_model::DignityTier::Detriment => "detriment",
        rdf_model::DignityTier::Fall => "fall",
        rdf_model::DignityTier::Peregrine => "peregrine",
    }
}

/// Prints the human-readable `chart` report from the computed model.
#[allow(clippy::too_many_lines)] // linear section-by-section transcription
fn print_human(args: &ChartArgs, chart: &rdf_model::ChartResource) {
    println!("jd_tt = {:.9}", chart.jd_tt);
    match (args.sidereal, chart.zodiac) {
        (Some(kind), rdf_model::Zodiac::Sidereal { degrees, .. }) => {
            println!(
                "sidereal = {} (ayanamsha = {degrees:.6} deg)",
                kind.json_name()
            );
        }
        _ => println!("sidereal = none (tropical)"),
    }
    println!("sect = {}", chart.sect.name());
    println!();

    println!("-- bodies --");
    for b in &chart.bodies {
        let tag = b.motion.tag();
        let tag_col = if tag.is_empty() { " " } else { tag };
        println!(
            "{:<8} {:<20} {tag_col:<1} house {:>2}   dec = {:>+8.4} deg   dist_au = {:>12.9}",
            b.name,
            format_sign(b.lon_deg.to_radians()),
            b.house,
            b.declination_deg,
            b.distance_au,
        );
    }
    println!();

    println!("-- houses ({}) --", args.system.name());
    for cusp in &chart.cusps {
        println!(
            "cusp {:>2} = {:>13.6} deg   {}",
            cusp.number,
            cusp.lon_deg,
            format_sign(cusp.lon_deg.to_radians()),
        );
    }
    println!();

    println!("-- angles --");
    print_angle(
        "ascendant",
        angle_deg(&chart.angles, rdf_model::AngleKind::Ascendant),
    );
    print_angle(
        "mc",
        angle_deg(&chart.angles, rdf_model::AngleKind::Midheaven),
    );
    print_angle(
        "vertex",
        angle_deg(&chart.angles, rdf_model::AngleKind::Vertex),
    );
    print_angle(
        "east_point",
        angle_deg(&chart.angles, rdf_model::AngleKind::EastPoint),
    );
    println!();

    println!("-- nodes & lots --");
    print_angle(
        "mean_node",
        node_deg(&chart.nodes, rdf_model::NodeKind::MeanNode),
    );
    print_angle(
        "mean_apogee",
        node_deg(&chart.nodes, rdf_model::NodeKind::MeanApogee),
    );
    print_angle(
        "true_node",
        node_deg(&chart.nodes, rdf_model::NodeKind::TrueNode),
    );
    print_angle(
        "true_apogee",
        node_deg(&chart.nodes, rdf_model::NodeKind::TrueApogee),
    );
    print_angle("fortune", lot_deg(&chart.lots, rdf_model::LotKind::Fortune));
    print_angle("spirit", lot_deg(&chart.lots, rdf_model::LotKind::Spirit));
    println!();

    let dist = &chart.distribution;
    println!("-- distribution --");
    println!(
        "elements  : fire {}  earth {}  air {}  water {}",
        dist.elements[0], dist.elements[1], dist.elements[2], dist.elements[3],
    );
    println!(
        "modalities: cardinal {}  fixed {}  mutable {}",
        dist.modalities[0], dist.modalities[1], dist.modalities[2],
    );
    println!();

    println!("-- dignities ({}) --", chart.rulership);
    for d in &chart.dignities {
        let tags: Vec<&str> = d.tiers.iter().map(|t| tier_label(*t)).collect();
        println!(
            "{:<8} score {:>+3}   {}",
            d.planet.name(),
            d.score,
            tags.join(", ")
        );
    }
    println!();

    println!("-- aspects --");
    if chart.aspects.is_empty() {
        println!("(none within the default orbs)");
    }
    for hit in &chart.aspects {
        let kind = hit.kind.name();
        let applying = if hit.applying {
            "applying"
        } else {
            "separating"
        };
        println!(
            "{:<8} {kind:<14} {:<8} offset = {:>9.4} deg ({applying})",
            hit.body1, hit.body2, hit.offset_deg,
        );
    }
}

/// Prints one named angle/point with its degree and sign notation.
fn print_angle(name: &str, lon_deg: f64) {
    println!(
        "{name:<11} = {lon_deg:>13.6} deg   {}",
        format_sign(lon_deg.to_radians()),
    );
}
