//! `OxiEphemeris` chart-computation facade: the UI-free layer that turns a
//! birth (or comparison) request plus DE ephemeris bytes into a
//! [`oxiephemeris_rdf::model::ChartResource`] / [`ChartComparison`] /
//! [`CompositeChartResource`], plus their stable JSON and RDF renderings.
//!
//! This is the single source of truth the CLI, the Python binding, and the
//! WASM binding all share, so a natal chart computed through any of them is
//! bit-identical.
//!
//! [`ChartComparison`]: oxiephemeris_rdf::model::ChartComparison
//! [`CompositeChartResource`]: oxiephemeris_rdf::model::CompositeChartResource
#![forbid(unsafe_code)]

pub mod body;
pub mod calendar;
pub mod comparison;
pub mod epoch;
pub mod error;
pub mod houses;
pub mod iso8601;
pub mod json;
pub mod natal;
pub mod points;
pub mod positions;
pub mod request;
pub mod serialize;

pub use comparison::{composite, progression, synastry, transit, PersonInput};
pub use natal::natal_chart;
pub use request::{NatalRequest, NatalSpec};
pub use serialize::{
    chart_to_json, chart_to_rdf, comparison_to_json, comparison_to_rdf, composite_to_json,
    composite_to_rdf, RdfFormat,
};

pub use calendar::CalendarKind;
pub use epoch::{resolve_chart_epoch, ChartEpoch};
pub use error::ChartError;
pub use houses::{compute_houses, HousesResult};
pub use points::{natal_points, transit_points, ChartPoint};
pub use positions::{compute_bodies, BodyPlace};
