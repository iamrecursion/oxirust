//! Faceted search and aggregation.

pub mod aggregation;
pub mod aggregator;
pub mod search;

pub use aggregation::Facets;
pub use search::FacetedSearch;
