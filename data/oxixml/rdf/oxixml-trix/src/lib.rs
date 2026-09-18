//! `oxixml-trix` — [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) (Triples in XML) parser
//! and serializer, producing and consuming [`oxixml-model`](oxixml_model) [`Quad`](oxixml_model::Quad)s.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! TriX is a simple XML syntax for RDF datasets: a `<TriX>` document element in the
//! `http://www.w3.org/2004/03/trix/trix-1/` namespace contains zero or more `<graph>`
//! elements, each optionally named by a leading `<uri>` or `<id>` child (an unnamed
//! `<graph>` holds the default graph), followed by any number of `<triple>` elements, each
//! containing exactly 3 term children in subject/predicate/object order:
//!
//! * `<uri>` — an IRI (a [`NamedNode`](oxixml_model::NamedNode)),
//! * `<id>` — a document-scoped blank node ([`BlankNode`](oxixml_model::BlankNode)),
//! * `<plainLiteral>` (optionally with an `xml:lang` attribute) — a simple or
//!   language-tagged [`Literal`](oxixml_model::Literal),
//! * `<typedLiteral datatype="...">` — a datatyped [`Literal`](oxixml_model::Literal).
//!
//! The parser ([`TriXParser`]) is a streaming, pull-based [`Iterator`] over
//! [`Quad`](oxixml_model::Quad)s built on [`oxixml-xml`](oxixml_xml)'s namespace-aware
//! [`NsReader`](oxixml_xml::reader::NsReader); the serializer ([`TriXSerializer`]) writes
//! quads back out, grouping consecutive same-graph quads under one `<graph>` element.
//!
//! # Example
//!
//! ```
//! use oxixml_model::{Literal, NamedNode, Quad};
//! use oxixml_trix::{TriXParser, TriXSerializer};
//!
//! let quad = Quad::new(
//!     NamedNode::new("http://example.com/s")?,
//!     NamedNode::new("http://example.com/p")?,
//!     Literal::new_simple_literal("o"),
//!     NamedNode::new("http://example.com/g")?,
//! );
//!
//! let mut serializer = TriXSerializer::new().for_writer(Vec::new());
//! serializer.serialize_quad(&quad)?;
//! let bytes = serializer.finish()?;
//!
//! let parsed = TriXParser::new()
//!     .for_slice(&bytes)
//!     .collect::<Result<Vec<_>, _>>()?;
//! assert_eq!(parsed, vec![quad]);
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```

#![deny(missing_docs)]

mod error;
mod parser;
mod serializer;

pub use crate::error::{TriXParseError, TriXSyntaxError};
#[cfg(feature = "async-tokio")]
pub use crate::parser::TokioAsyncReaderTriXParser;
pub use crate::parser::{ReaderTriXParser, SliceTriXParser, TRIX_NAMESPACE, TriXParser};
#[cfg(feature = "async-tokio")]
pub use crate::serializer::TokioAsyncWriterTriXSerializer;
pub use crate::serializer::{TriXSerializer, WriterTriXSerializer};
