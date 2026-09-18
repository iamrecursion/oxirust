//! `OxiEphemeris` SPARQL store and HTTP endpoint.
//!
//! This crate turns the RDF/SKOS vocabulary and chart graphs built by
//! [`oxiephemeris_rdf`] into a queryable, dereferenceable Linked Open Data
//! service:
//!
//! * [`FileBackedStore`] — an in-memory oxigraph store with pure-Rust,
//!   whole-file N-Quads persistence (no `RocksDB`, per the COOLJAPAN pure-Rust
//!   policy). See [`store`] for the honest trade-offs.
//! * [`endpoint::handle`] — the SPARQL 1.1 Protocol handler as a *pure
//!   function*, so the whole HTTP surface is testable without binding a
//!   socket.
//! * [`LodError`] — one error type mapping cleanly onto HTTP status codes.
//!
//! The `oxieph-sparqld` binary wires [`endpoint::handle`] behind an oxhttp
//! server. Because it serves the `/ns/oxiephemeris/…` graphs as Turtle, the
//! published `oxa:`/`oxc:`/`oxs:` IRIs become dereferenceable when the
//! service is hosted at `cooljapan.tech`.
#![forbid(unsafe_code)]

pub mod endpoint;
pub mod error;
pub mod store;

pub use endpoint::handle;
pub use error::LodError;
pub use store::FileBackedStore;
