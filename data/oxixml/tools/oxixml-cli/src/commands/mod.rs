//! One module per `oxixml` subcommand, each exposing a `run(args, stdin, stdout, stderr) -> u8`
//! entry point that [`crate::run_status`] dispatches to by name.

pub(crate) mod canon;
pub(crate) mod convert;
mod doc;
pub(crate) mod fmt;
pub(crate) mod hdt;
pub(crate) mod lint;
pub(crate) mod query;
mod rdf_format;
pub(crate) mod sparql;
pub(crate) mod svg;
pub(crate) mod transform;
pub(crate) mod validate;
