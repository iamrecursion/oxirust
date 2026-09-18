//! `oxixml-rdfxml` — a streaming [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) parser and
//! serializer.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust
//! foundation for all XML and RDF processing. This crate is the dependency-light replacement for
//! [`oxrdfxml`](https://crates.io/crates/oxrdfxml): it mirrors that crate's public API, so code
//! written against `oxrdfxml` ports over by changing only the `use` paths. The XML layer is
//! [`oxixml-xml`](oxixml_xml) rather than `quick-xml`, and the data model is
//! [`oxixml-model`](oxixml_model) rather than `oxrdf`.
//!
//! # Parsing
//!
//! [`RdfXmlParser`] is a builder: [`with_base_iri`](RdfXmlParser::with_base_iri) sets the base for
//! relative IRI references (overridden by any `xml:base` in the document) and
//! [`lenient`](RdfXmlParser::lenient) skips IRI, language-tag and `rdf:ID`-uniqueness validation.
//! [`for_slice`](RdfXmlParser::for_slice) and [`for_reader`](RdfXmlParser::for_reader) then produce
//! iterators of [`Triple`](oxixml_model::Triple)s.
//!
//! The full RDF/XML 1.1 grammar is supported: an optional `rdf:RDF` document element, node and
//! property elements, `rdf:about` / `rdf:ID` / `rdf:nodeID` / `rdf:resource`, property attributes,
//! `rdf:parseType` of `Literal`, `Resource` and `Collection`, containers (`rdf:li` becoming
//! `rdf:_1`, `rdf:_2`, …), reification through `rdf:ID` on property elements, `xml:lang` and
//! `xml:base` scoping, `rdf:datatype`, internal-DTD entity declarations, and rejection of the
//! terms the specification withdrew (`rdf:aboutEach`, `rdf:aboutEachPrefix`).
//!
//! # Serializing
//!
//! [`RdfXmlSerializer`] writes striped RDF/XML: consecutive triples sharing a subject are grouped
//! into one `rdf:Description` — or into a typed node element when the group starts with an
//! `rdf:type`. Sorting the input by subject therefore yields the most compact output.
//!
//! # Features
//!
//! * `async-tokio` — asynchronous parsing and serialization on top of `tokio`'s `AsyncRead` and
//!   `AsyncWrite`, mirroring `oxrdfxml`'s feature of the same name:
#![cfg_attr(
    feature = "async-tokio",
    doc = "  [`RdfXmlParser::for_tokio_async_reader`] and [`RdfXmlSerializer::for_tokio_async_writer`]."
)]
#![cfg_attr(
    not(feature = "async-tokio"),
    doc = "  `RdfXmlParser::for_tokio_async_reader` and `RdfXmlSerializer::for_tokio_async_writer`."
)]
//!   Both drive the very same event-at-a-time state machines as the blocking paths (through
//!   [`oxixml-xml`](oxixml_xml)'s own `async-tokio` feature), so behaviour and output are
//!   identical. Off by default, keeping the default build third-party-free.
//!
//! # Example
//!
//! ```
//! use oxixml_model::vocab::rdf;
//! use oxixml_model::NamedNode;
//! use oxixml_rdfxml::{RdfXmlParser, RdfXmlSerializer};
//!
//! let file = r#"<?xml version="1.0"?>
//! <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
//!  <schema:Person rdf:about="http://example.com/foo" schema:name="Foo" />
//! </rdf:RDF>"#;
//!
//! let triples = RdfXmlParser::new()
//!     .for_slice(file)
//!     .collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(triples.len(), 2);
//! let person = NamedNode::new("http://schema.org/Person")?;
//! assert!(triples.iter().any(|t| t.predicate == rdf::TYPE && t.object == person));
//!
//! let mut serializer = RdfXmlSerializer::new()
//!     .with_prefix("schema", "http://schema.org/")?
//!     .for_writer(Vec::new());
//! for triple in &triples {
//!     serializer.serialize_triple(triple)?;
//! }
//! let out = String::from_utf8(serializer.finish()?)?;
//! assert!(out.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```

#![deny(missing_docs)]

mod error;
mod parser;
mod serializer;
mod utils;

pub use crate::error::{RdfXmlParseError, RdfXmlSyntaxError};
#[cfg(feature = "async-tokio")]
pub use crate::parser::TokioAsyncReaderRdfXmlParser;
pub use crate::parser::{RdfXmlParser, RdfXmlPrefixesIter, ReaderRdfXmlParser, SliceRdfXmlParser};
#[cfg(feature = "async-tokio")]
pub use crate::serializer::TokioAsyncWriterRdfXmlSerializer;
pub use crate::serializer::{RdfXmlSerializer, WriterRdfXmlSerializer};
