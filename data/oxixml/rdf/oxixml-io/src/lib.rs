//! `oxixml-io` — unified RDF format detection, parsing, and serialization.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust
//! foundation for all XML and RDF processing. This crate is the dependency-light replacement for
//! [`oxrdfio`](https://crates.io/crates/oxrdfio): it mirrors that crate's public API — the
//! [`RdfFormat`] enum, the [`RdfParser`]/[`RdfSerializer`] builders, and the unified
//! [`RdfParseError`]/[`RdfSyntaxError`] error types — so that code written against `oxrdfio` ports
//! over by changing only the `use` paths, plus one **OxiXML extension**:
//! [`RdfFormat::TriX`].
//!
//! It wires together every RDF syntax this workspace implements behind one API:
//!
//! | Format | Backend crate | Datasets? |
//! |---|---|---|
//! | [`RdfFormat::NTriples`] | [`oxixml-turtle`](oxixml_turtle) | no |
//! | [`RdfFormat::NQuads`] | [`oxixml-turtle`](oxixml_turtle) | yes |
//! | [`RdfFormat::Turtle`] | [`oxixml-turtle`](oxixml_turtle) | no |
//! | [`RdfFormat::TriG`] | [`oxixml-turtle`](oxixml_turtle) | yes |
//! | [`RdfFormat::N3`] | [`oxixml-turtle`](oxixml_turtle) (parses; serializes as Turtle) | no |
//! | [`RdfFormat::RdfXml`] | [`oxixml-rdfxml`](oxixml_rdfxml) | no |
//! | [`RdfFormat::JsonLd`] | [`oxixml-jsonld`](oxixml_jsonld) | yes |
//! | [`RdfFormat::TriX`] | [`oxixml-trix`](oxixml_trix) | yes |
//!
//! # Features
//!
//! * `rdf-12` — RDF 1.2 / RDF-star, forwarded to [`oxixml-model`](oxixml_model) and
//!   [`oxixml-turtle`](oxixml_turtle): the Turtle-family formats read and write triple terms and
//!   directional language-tagged strings, and blank nodes nested inside a triple term take part in
//!   [`rename_blank_nodes`](RdfParser::rename_blank_nodes). The RDF/XML, TriX and JSON-LD
//!   backends have no syntax for a triple term and report an error instead of dropping it.
//!
//! # Parsing
//!
//! [`RdfParser::from_format`] builds a parser; [`for_reader`](RdfParser::for_reader) and
//! [`for_slice`](RdfParser::for_slice) turn it into an iterator of
//! [`Quad`](oxixml_model::Quad)s. Formats limited to a single graph (N-Triples, Turtle, RDF/XML)
//! place their triples in the [default graph](oxixml_model::GraphName::DefaultGraph), or in
//! [`with_default_graph`](RdfParser::with_default_graph)'s target when set.
//!
//! # Serializing
//!
//! [`RdfSerializer::from_format`] builds a serializer; [`for_writer`](RdfSerializer::for_writer)
//! turns it into a [`WriterQuadSerializer`] that accepts [`serialize_quad`] and
//! [`serialize_triple`] calls, finished with [`finish`](WriterQuadSerializer::finish).
//!
//! [`serialize_quad`]: WriterQuadSerializer::serialize_quad
//! [`serialize_triple`]: WriterQuadSerializer::serialize_triple
//!
//! # Format detection
//!
//! [`RdfFormat::from_extension`] and [`RdfFormat::from_media_type`] recognize file extensions
//! (`"ttl"`, `"nq"`, ...) and IANA media types (`"text/turtle"`, `"application/ld+json"`, ...,
//! including `;charset=`/`;profile=` parameters), the same way `oxrdfio` does, plus TriX's
//! `"trix"` extension and `"application/trix+xml"` media type.
//!
//! # Example: format detection, cross-format conversion, and blank-node renaming
//!
//! ```
//! use oxixml_io::{RdfFormat, RdfParser, RdfSerializer};
//!
//! // Detect the format from a file extension, then parse.
//! let format = RdfFormat::from_extension("ttl").expect("known extension");
//! let turtle = r#"@prefix schema: <http://schema.org/> .
//! _:alice a schema:Person ;
//!     schema:name "Alice" ;
//!     schema:knows _:bob .
//! _:bob a schema:Person ;
//!     schema:name "Bob" ."#;
//!
//! let quads = RdfParser::from_format(format)
//!     .rename_blank_nodes()
//!     .for_reader(turtle.as_bytes())
//!     .collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(quads.len(), 5);
//!
//! // Re-serialize the same quads as N-Quads (a dataset format, so every quad round-trips).
//! let mut serializer = RdfSerializer::from_format(RdfFormat::NQuads).for_writer(Vec::new());
//! for quad in &quads {
//!     serializer.serialize_quad(quad)?;
//! }
//! let nquads = String::from_utf8(serializer.finish()?)?;
//! assert_eq!(nquads.lines().count(), 5);
//! # Result::<(), Box<dyn std::error::Error>>::Ok(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

mod document;
mod error;
mod format;
mod mapper;
mod parser;
mod serializer;

pub use document::LoadedDocument;
pub use error::{RdfParseError, RdfSyntaxError, TextPosition};
pub use format::RdfFormat;
pub use oxixml_jsonld::{JsonLdProfile, JsonLdProfileSet};
#[cfg(feature = "async-tokio")]
pub use parser::TokioAsyncReaderQuadParser;
pub use parser::{PrefixesIter, RdfParser, ReaderQuadParser, SliceQuadParser};
#[cfg(feature = "async-tokio")]
pub use serializer::TokioAsyncWriterQuadSerializer;
pub use serializer::{RdfSerializer, WriterQuadSerializer};
