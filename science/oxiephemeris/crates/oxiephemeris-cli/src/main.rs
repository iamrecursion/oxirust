//! `oxieph` — CLI for `OxiEphemeris`: Julian Date/calendar conversion,
//! solar-system body positions, house cusps/chart angles, and full natal
//! charts.
//!
//! Subcommands:
//! - `convert` — Julian Date <-> proleptic calendar date/time (no time
//!   scale involved; see `convert`).
//! - `pos` — apparent (or astrometric-style) place of a solar-system body
//!   on a JPL DE ephemeris, or (`--all`) every supported body at once
//!   (see `pos`).
//! - `houses` — house cusps and chart angles (Ascendant, MC, Vertex, East
//!   Point) for a geographic site and UTC epoch (see `houses`).
//! - `chart` — `pos --all` plus `houses` plus lunar nodes plus an aspect
//!   table, in one call (see `chart`).
//!
//! No `chrono`: ISO 8601 parsing is hand-rolled in `iso8601`.

mod astro_args;
mod astro_epoch;
mod body;
mod chart;
mod chart_render;
mod chartcommon;
mod composite;
mod convert;
mod de_locate;
mod errors;
mod houses;
mod iso8601;
mod pos;
mod progress;
mod rdf_args;
mod synastry;
mod transit;
mod vocab;

use clap::{Parser, Subcommand};

use chart::ChartArgs;
use composite::CompositeArgs;
use convert::ConvertArgs;
use errors::CliError;
use houses::HousesArgs;
use pos::PosArgs;
use progress::ProgressArgs;
use synastry::SynastryArgs;
use transit::TransitArgs;
use vocab::VocabArgs;

#[derive(Debug, Parser)]
#[command(
    name = "oxieph",
    version,
    about = "OxiEphemeris CLI: Julian Date/calendar conversions, apparent-place computations, \
             house cusps, and full charts"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Convert between a Julian Date and a proleptic calendar date/time.
    Convert(ConvertArgs),
    /// Compute the apparent (or astrometric-style) place of a body (or,
    /// with `--all`, every supported body).
    Pos(PosArgs),
    /// Compute house cusps and chart angles for a geographic site and
    /// UTC epoch.
    Houses(HousesArgs),
    /// Compute a full chart: bodies, houses, angles, nodes and aspects.
    Chart(ChartArgs),
    /// Cross-aspect two natal charts (synastry).
    Synastry(SynastryArgs),
    /// Aspects of transiting bodies at a date to a natal chart.
    Transit(TransitArgs),
    /// Secondary-progressed chart (a day for a year) and its aspects to
    /// the natal chart.
    Progress(ProgressArgs),
    /// Midpoint composite chart of two natal charts.
    Composite(CompositeArgs),
    /// Emit the astrology ontology and SKOS concept schemes as RDF.
    Vocab(VocabArgs),
}

fn run() -> Result<(), CliError> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Convert(args) => convert::run(args),
        Command::Pos(args) => pos::run(args),
        Command::Houses(args) => houses::run(args),
        Command::Chart(args) => chart::run(args),
        Command::Synastry(args) => synastry::run(args),
        Command::Transit(args) => transit::run(args),
        Command::Progress(args) => progress::run(args),
        Command::Composite(args) => composite::run(args),
        Command::Vocab(args) => vocab::run(args),
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
