//! `oxixml-turtle` — streaming parsers and serializers for the RDF Turtle family of
//! syntaxes: [N-Triples](https://www.w3.org/TR/n-triples/),
//! [N-Quads](https://www.w3.org/TR/n-quads/), [Turtle](https://www.w3.org/TR/turtle/),
//! [TriG](https://www.w3.org/TR/trig/) and [Notation3](https://w3c.github.io/N3/spec/).
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing. This crate is the `std`-only,
//! dependency-light replacement for [`oxttl`](https://crates.io/crates/oxttl), mirroring its
//! public API so that code written against `oxttl` ports over by changing only the `use` path.
//!
//! Every format exposes a builder-style parser (`*Parser::new()`) with:
//!
//! * `for_slice` / `for_reader` iterators that yield `Result<Triple, TurtleParseError>` (or
//!   `Quad` for the quad formats),
//! * a low-level incremental push API (`LowLevel*Parser`) that consumes bytes and produces
//!   parse events, exposing [`prefixes`](turtle::LowLevelTurtleParser::prefixes) and
//!   [`base_iri`](turtle::LowLevelTurtleParser::base_iri) accessors mid-parse,
//! * syntax errors carrying precise [`TextPosition`] line/column/offset ranges.
//!
//! Serializers offer the mirrored `for_writer` and low-level forms.
//!
//! # Features
//!
//! * `rdf-12` — RDF 1.2 / RDF-star: quoted/triple terms and directional language-tagged
//!   strings, forwarded to [`oxixml-model`](oxixml_model).
//! * `async-tokio` — asynchronous parsing and serialization on top of `tokio`'s `AsyncRead`
//!   and `AsyncWrite`, mirroring `oxttl`'s feature of the same name: every parser gains
//!   `for_tokio_async_reader` and every serializer gains `for_tokio_async_writer`. The async
//!   wrappers feed the very same incremental cores as the blocking ones, so parsing behaviour
//!   is identical. Off by default, keeping the default build third-party-free.
//!
//! # Example: parsing Turtle
//!
//! ```
//! use oxixml_turtle::TurtleParser;
//! use oxixml_model::vocab::rdf;
//! use oxixml_model::NamedNode;
//!
//! let data = br#"@base <http://example.com/> .
//! @prefix schema: <http://schema.org/> .
//! <foo> a schema:Person ;
//!     schema:name "Foo" ."#;
//!
//! let schema_person = NamedNode::new("http://schema.org/Person")?;
//! let mut count = 0;
//! for triple in TurtleParser::new().for_slice(data) {
//!     let triple = triple?;
//!     if triple.predicate == rdf::TYPE {
//!         assert!(triple.object == schema_person);
//!     }
//!     count += 1;
//! }
//! assert_eq!(count, 2);
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

mod chunker;
mod lexer;
mod line_formats;
pub mod n3;
pub mod nquads;
pub mod ntriples;
mod terse;
mod toolkit;
pub mod trig;
pub mod turtle;

pub use crate::n3::N3Parser;
pub use crate::nquads::{NQuadsParser, NQuadsSerializer};
pub use crate::ntriples::{NTriplesParser, NTriplesSerializer};
pub use crate::toolkit::{TextPosition, TurtleParseError, TurtleSyntaxError};
pub use crate::trig::{TriGParser, TriGSerializer};
pub use crate::turtle::{TurtleParser, TurtleSerializer};

pub(crate) const MIN_BUFFER_SIZE: usize = 4096;
/// Represents a reasonable default max buffer size in bytes for parsing
/// rdf data
pub(crate) const DEFAULT_MAX_BUFFER_SIZE: usize = 4096 * 4096;
pub(crate) const MIN_PARALLEL_CHUNK_SIZE: usize = 16384;
