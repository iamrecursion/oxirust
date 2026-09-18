//! `OxiEphemeris` Linked Open Data layer: RDF graphs, a SKOS vocabulary,
//! and PROV-O provenance for the astrological chart data computed by
//! [`oxiephemeris_astro`].
#![forbid(unsafe_code)]

/// The RDF term model this crate's graphs are built from, re-exported so
/// downstream crates can name `Graph`/`NamedNode` without taking their own
/// dependency on it (and risking a version skew with the one these graphs
/// are made of).
///
/// The underlying crate is [`oxixml-model`](https://crates.io/crates/oxixml-model);
/// the local name is kept as `oxrdf` because that is the name the RDF
/// literature — and every `use` path in this crate — spells it. These are
/// **not** the same types as the `oxrdf` crate's: code that mixes
/// `oxiephemeris_rdf::oxrdf::*` with terms built by `oxrdf` itself (for
/// instance via `oxigraph::model`) has to convert between the two.
pub use oxrdf;

mod builder;
pub mod chart;
pub mod comparison;
pub mod error;
pub mod iri;
pub mod labels;
pub mod model;
pub mod ontology;
pub mod prov;
pub mod serialize;
pub mod skos;
pub mod vocab;

pub use chart::chart_graph;
pub use comparison::{comparison_graph, composite_graph};
pub use error::RdfError;
pub use iri::{ChartKey, ComparisonKey, IriMinter};
pub use model::{ChartComparison, ChartResource, CompositeChartResource};
pub use ontology::ontology_graph;
pub use serialize::{to_ntriples_string, to_turtle_string, write_ntriples, write_turtle};
pub use skos::concept_scheme_graph;
