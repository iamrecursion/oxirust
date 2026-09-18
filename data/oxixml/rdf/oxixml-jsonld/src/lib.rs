//! `oxixml-jsonld` — a complete [JSON-LD 1.1](https://www.w3.org/TR/json-ld11/) processor, and the
//! Pure-Rust replacement for [`oxjsonld`](https://crates.io/crates/oxjsonld).
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust
//! foundation for all XML and RDF processing.
//!
//! # Two layers
//!
//! * **Streaming RDF interface** — [`JsonLdParser`] turns a JSON-LD document into a stream of RDF
//!   [`Quad`](oxixml_model::Quad)s (the *toRdf* pipeline), and [`JsonLdSerializer`] turns a
//!   dataset back into JSON-LD (the *fromRdf* pipeline). This mirrors the `oxjsonld` API so that
//!   code written against it ports over by changing only the `use` path.
//! * **Full JSON-LD 1.1 API** — [`expand`], [`compact`], [`flatten`], [`to_rdf`], and [`from_rdf`]
//!   implement the [JSON-LD 1.1 API](https://www.w3.org/TR/json-ld11-api/) algorithms over an
//!   internal, `serde`-free [`JsonValue`] tree (parsed via [`oxixml_json`] events): context
//!   processing (term definitions, `@vocab`, `@base`, protected terms, scoped contexts, remote
//!   contexts via a [`DocumentLoader`] hook), expansion, compaction, flattening, and RDF
//!   (de)serialization including `@json` literals, `@direction` handling, and a generalized-RDF
//!   flag.
//!
//! # No I/O by default
//!
//! The default [`DocumentLoader`] ([`NoLoader`]) performs **no** network or file access: a document
//! that references a remote `@context` fails cleanly unless the caller supplies a loader (directly,
//! or via [`ClosureLoader`], or through
//! [`with_load_document_callback`](SliceJsonLdParser::with_load_document_callback) on a parser).
//! Nothing in this crate opens a socket or a file on its own.
//!
//! # Conformance
//!
//! The crate passes **every** entry it runs of the W3C
//! [JSON-LD 1.1 API](https://w3c.github.io/json-ld-api/tests/) and
//! [JSON-LD 1.1 Streaming](https://w3c.github.io/json-ld-streaming/tests/) test suites:
//! expansion 376/376, compaction 244/244, flattening 55/55, toRdf 455/455, fromRdf 52/52, and the
//! streaming toRdf profile 471/471. The only entries not run are those targeting JSON-LD **1.0**
//! (whose expectations contradict 1.1), those whose input is an HTML document, and a single
//! generalized-RDF expectation with a blank node in predicate position, which
//! [`Quad`](oxixml_model::Quad) — like the RDF 1.1 abstract syntax — cannot represent.
//!
//! The harness lives in `tests/jsonld_suite.rs` and finds the suites through the
//! `OXIXML_TESTSUITE_DIR` environment variable; it reports "skipped" and passes when unset.
//!
//! # Serializer output, against `oxjsonld`'s
//!
//! [`JsonLdSerializer`] writes the same document shape `oxjsonld` writes. With neither
//! [`with_base_iri`](JsonLdSerializer::with_base_iri) nor
//! [`with_prefix`](JsonLdSerializer::with_prefix) set, the output is a bare array of expanded node
//! objects. As soon as either is set, an `@context` is written and the nodes move into a `@graph`:
//! `{"@context": {…}, "@graph": […]}`. Inside that context `@base` comes first, then one entry per
//! prefix in name order, with the empty prefix name written as `@vocab`. Property keys stay
//! expanded IRIs in both processors: declaring a prefix declares a term, it does not shorten the
//! keys. Node identifiers are made relative to `@base`; `@type` is not, because expansion resolves
//! a type against `@vocab` rather than `@base`.
//!
//! One difference is forced by the algorithm this crate implements. `oxjsonld` writes a stream of
//! quads straight out, so it emits `rdf:type` as an ordinary property key
//! (`"…#type": [{"@id": …}]`). This crate runs the specified *fromRdf* algorithm over a node map,
//! and that algorithm folds `rdf:type` into the `@type` keyword — `"@type": ["…"]` — unless
//! [`JsonLdOptions::use_rdf_type`] is set. Both forms expand to exactly the same RDF, and the node
//! map is what makes `@list` reconstruction and named-graph nesting possible in the first place.
//!
//! # Framing
//!
//! [JSON-LD Framing](https://www.w3.org/TR/json-ld11-framing/) is **not** implemented; it is a
//! documented gap. Expansion, compaction, flattening, toRdf, and fromRdf are all supported, as is
//! the streaming profile's key-order validation. Framing-specific keywords (`@embed`, `@explicit`,
//! `@default`, `@omitDefault`, `@requireAll`) are consequently not recognised, and
//! [`JsonLdProfile::Frame`] / [`JsonLdProfile::Framed`] exist only as profile identifiers.
//!
//! # Features
//!
//! * `async-tokio` — asynchronous parsing and serialization on top of `tokio`'s `AsyncRead` and
//!   `AsyncWrite`, mirroring `oxjsonld`'s feature of the same name:
#![cfg_attr(
    feature = "async-tokio",
    doc = "  [`JsonLdParser::for_tokio_async_reader`] and [`JsonLdSerializer::for_tokio_async_writer`]."
)]
#![cfg_attr(
    not(feature = "async-tokio"),
    doc = "  `JsonLdParser::for_tokio_async_reader` and `JsonLdSerializer::for_tokio_async_writer`."
)]
//!   JSON-LD expansion and *fromRdf* both need the whole document in hand before they can produce
//!   anything, so — exactly as in the blocking paths — the async wrappers await only the I/O and
//!   then run the very same cores. Off by default, keeping the default build third-party-free.
//!
//! # Example
//!
//! ```
//! use oxixml_jsonld::{JsonLdParser};
//! use oxixml_model::vocab::rdf;
//! use oxixml_model::{NamedNode, Term};
//!
//! let file = r#"{
//!     "@context": {"name": "http://schema.org/name"},
//!     "@id": "http://example.com/foo",
//!     "@type": "http://schema.org/Person",
//!     "name": "Foo"
//! }"#;
//!
//! let person = NamedNode::new("http://schema.org/Person")?;
//! let mut has_type = false;
//! for quad in JsonLdParser::new().for_slice(file.as_bytes()) {
//!     let quad = quad?;
//!     if quad.predicate == rdf::TYPE && quad.object == Term::from(person.clone()) {
//!         has_type = true;
//!     }
//! }
//! assert!(has_type);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]

mod api;
mod compaction;
mod context;
mod error;
mod expansion;
mod flattening;
mod from_rdf;
mod inverse;
mod loader;
mod node_map;
mod parser;
mod profile;
mod streaming;
mod to_rdf;
mod value;

pub use api::{JsonLdOptions, compact, expand, flatten, from_rdf, to_rdf};
/// A position within a JSON-LD document.
///
/// Its `line` and `column` are **0-based** (the first character of a document is line `0`,
/// column `0`) and its byte `offset` is 0-based too; only its [`Display`](std::fmt::Display)
/// impl switches to the human-facing 1-based convention.
pub use error::JsonTextPosition as TextPosition;
pub use error::{JsonLdErrorCode, JsonLdParseError, JsonLdSyntaxError, JsonTextPosition};
#[cfg(feature = "async-tokio")]
pub use from_rdf::TokioAsyncWriterJsonLdSerializer;
pub use from_rdf::{JsonLdSerializer, WriterJsonLdSerializer};
pub use loader::{
    ClosureLoader, DocumentLoader, LoadDocumentError, LoadDocumentOptions, NoLoader, RemoteDocument,
};

#[cfg(feature = "async-tokio")]
pub use parser::TokioAsyncReaderJsonLdParser;
pub use parser::{JsonLdParser, JsonLdPrefixesIter, ReaderJsonLdParser, SliceJsonLdParser};
pub use profile::{JsonLdProcessingMode, JsonLdProfile, JsonLdProfileBagIter, JsonLdProfileSet};
pub use to_rdf::{RdfDirection, canonical_double, canonical_json, ecmascript_number};
pub use value::{DEFAULT_MAX_NESTING_DEPTH, JsonObject, JsonValue};

/// The `oxjsonld` name for [`LoadDocumentOptions`].
pub type JsonLdLoadDocumentOptions = LoadDocumentOptions;
/// The `oxjsonld` name for [`RemoteDocument`].
pub type JsonLdRemoteDocument = RemoteDocument;
