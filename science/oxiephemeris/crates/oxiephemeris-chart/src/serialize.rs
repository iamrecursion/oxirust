//! Rendering computed resources to JSON and RDF (Turtle / N-Triples)
//! strings.
//!
//! IRIs are minted here, from each resource's own fields, so the compute
//! layer stays free of IRI/base concerns. A natal chart's IRI comes from
//! its [`oxiephemeris_rdf::ChartKey`]; a comparison's two sides are minted
//! from their own keys (so a side equals the standalone chart of the same
//! birth) and the comparison from a [`oxiephemeris_rdf::ComparisonKey`].

use oxiephemeris_rdf::model as m;
use oxiephemeris_rdf::oxrdf::NamedNode;
use oxiephemeris_rdf::{
    chart_graph, comparison_graph, composite_graph, to_ntriples_string, to_turtle_string, ChartKey,
    ComparisonKey, IriMinter,
};

use crate::error::ChartError;
use crate::json::{chart_json, comparison_json, composite_json};

/// Which RDF serialization to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfFormat {
    /// RDF 1.1 Turtle.
    Turtle,
    /// RDF 1.1 N-Triples.
    NTriples,
}

/// Builds a minter over an optional base IRI (default = the published
/// instance namespace).
fn minter(base_iri: Option<&str>) -> Result<IriMinter, ChartError> {
    match base_iri {
        Some(base) => Ok(IriMinter::new(base)?),
        None => Ok(IriMinter::default()),
    }
}

/// The zodiac name a chart's `ChartKey` uses.
fn zodiac_name(zodiac: m::Zodiac) -> &'static str {
    match zodiac {
        m::Zodiac::Tropical => "tropical",
        m::Zodiac::Sidereal { ayanamsha, .. } => ayanamsha.name().unwrap_or("sidereal-custom"),
    }
}

/// The `ChartKey` of a full natal [`m::ChartResource`].
fn chart_key(chart: &m::ChartResource) -> ChartKey<'_> {
    let (lat_deg, lon_deg) = chart
        .observer
        .map_or((0.0, 0.0), |o| (o.lat_deg, o.lon_deg));
    ChartKey {
        kind: chart.kind.name(),
        jd_tt: chart.jd_tt,
        lat_deg,
        lon_deg,
        house_system: chart.house_system.name(),
        zodiac: zodiac_name(chart.zodiac),
    }
}

/// The chart IRI of one comparison side, or `None` if the side has no
/// epoch (it cannot then be a standalone chart).
fn side_iri(minter: &IriMinter, side: &m::ChartSide) -> Option<NamedNode> {
    let jd_tt = side.jd_tt?;
    let (lat_deg, lon_deg) = side.observer.map_or((0.0, 0.0), |o| (o.lat_deg, o.lon_deg));
    let key = ChartKey {
        kind: side.kind.name(),
        jd_tt,
        lat_deg,
        lon_deg,
        house_system: side.house_system.name(),
        zodiac: "tropical",
    };
    Some(minter.chart(&key))
}

/// The natal chart as its stable JSON string.
///
/// # Errors
///
/// [`ChartError::Json`] on serialization failure.
pub fn chart_to_json(chart: &m::ChartResource) -> Result<String, ChartError> {
    Ok(serde_json::to_string(&chart_json(chart))?)
}

/// The natal chart as an RDF string in `format`, under `base_iri`.
///
/// `chart_iri` overrides the minted IRI when given.
///
/// # Errors
///
/// [`ChartError::Rdf`] for an invalid base/chart IRI or a write failure.
pub fn chart_to_rdf(
    chart: &m::ChartResource,
    base_iri: Option<&str>,
    chart_iri: Option<&str>,
    format: RdfFormat,
) -> Result<String, ChartError> {
    let minter = minter(base_iri)?;
    let iri = match chart_iri {
        Some(explicit) => IriMinter::chart_explicit(explicit)?,
        None => minter.chart(&chart_key(chart)),
    };
    let graph = chart_graph(chart, &iri, &minter);
    Ok(render(&graph, format)?)
}

/// The comparison as its JSON string.
///
/// # Errors
///
/// [`ChartError::Json`] on serialization failure.
pub fn comparison_to_json(cmp: &m::ChartComparison) -> Result<String, ChartError> {
    Ok(serde_json::to_string(&comparison_json(cmp))?)
}

/// The comparison as an RDF string in `format`, under `base_iri`.
///
/// # Errors
///
/// [`ChartError::Rdf`] for an invalid base IRI, a write failure, or a side
/// that lacks an epoch (and so cannot be minted).
pub fn comparison_to_rdf(
    cmp: &m::ChartComparison,
    base_iri: Option<&str>,
    format: RdfFormat,
) -> Result<String, ChartError> {
    let minter = minter(base_iri)?;
    let (Some(a_iri), Some(b_iri)) = (
        side_iri(&minter, &cmp.chart_a),
        side_iri(&minter, &cmp.chart_b),
    ) else {
        return Err(ChartError::BadRequest(
            "both comparison sides must have an epoch to be serialized as RDF".to_owned(),
        ));
    };
    let key = ComparisonKey {
        kind: cmp.kind.name(),
        chart_a: a_iri.as_str(),
        chart_b: b_iri.as_str(),
    };
    let comparison_iri = minter.comparison(&key);
    let graph = comparison_graph(cmp, &comparison_iri, &a_iri, &b_iri, &minter);
    Ok(render(&graph, format)?)
}

/// The composite as its JSON string.
///
/// # Errors
///
/// [`ChartError::Json`] on serialization failure.
pub fn composite_to_json(comp: &m::CompositeChartResource) -> Result<String, ChartError> {
    Ok(serde_json::to_string(&composite_json(comp))?)
}

/// The composite as an RDF string in `format`, under `base_iri`.
///
/// The composite's IRI is derived from its two source charts' IRIs (in
/// `source_charts`), so `base_iri` must match the base those were minted
/// under (the compute layer uses the default base unless told otherwise).
///
/// # Errors
///
/// [`ChartError::Rdf`]/[`ChartError::BadRequest`] for an invalid base or
/// missing sources.
pub fn composite_to_rdf(
    comp: &m::CompositeChartResource,
    base_iri: Option<&str>,
    format: RdfFormat,
) -> Result<String, ChartError> {
    let minter = minter(base_iri)?;
    let [a, b] = comp.source_charts.as_slice() else {
        return Err(ChartError::BadRequest(
            "a composite must have exactly two source charts".to_owned(),
        ));
    };
    let key = ComparisonKey {
        kind: "composite",
        chart_a: a,
        chart_b: b,
    };
    let chart_iri = minter.chart_from_slug(&key.slug());
    let graph = composite_graph(comp, &chart_iri, &minter);
    Ok(render(&graph, format)?)
}

/// Serializes a graph in the chosen format.
fn render(
    graph: &oxiephemeris_rdf::oxrdf::Graph,
    format: RdfFormat,
) -> Result<String, oxiephemeris_rdf::RdfError> {
    match format {
        RdfFormat::Turtle => to_turtle_string(graph),
        RdfFormat::NTriples => to_ntriples_string(graph),
    }
}
