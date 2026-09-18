//! SPARQL query generation from TLExpr logic expressions.
//!
//! Split into submodules by `splitrs`; public API preserved via re-exports.

pub mod functions;
pub mod functions_2;
pub mod functions_3;
pub mod sparqlgenconfig_traits;
pub mod sparqlgenerror_traits;
pub mod sparqlquery_traits;
pub mod types;

pub use functions::*;
pub use functions_2::*;
pub use types::*;
