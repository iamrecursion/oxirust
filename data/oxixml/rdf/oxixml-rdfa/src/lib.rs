//! `oxixml-rdfa` — an [RDFa 1.1](https://www.w3.org/TR/rdfa-core/) extractor for HTML, XHTML and
//! XML.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust
//! foundation for all XML and RDF processing. This crate is what turns the structured data authors
//! embed in ordinary web pages into [`oxixml_model`] triples: it walks an [`oxixml_dom`] tree
//! (produced by [`oxixml_html`] for `text/html`, or by [`oxixml_xml`] for XHTML and XML) applying
//! the fourteen processing steps of [RDFa Core 1.1
//! §7.5](https://www.w3.org/TR/rdfa-core/#s_sequence), and hands back a graph.
//!
//! # What is implemented
//!
//! * **The whole RDFa Core 1.1 processing model** — the evaluation context (base, parent subject,
//!   parent object, incomplete triples, list mappings, language, IRI mappings, term mappings,
//!   default vocabulary) and every step of the sequence: `@vocab` with its `rdfa:usesVocabulary`
//!   triple, `@prefix` and the legacy `xmlns:` mappings, `@typeof`, `@about`, `@resource`,
//!   `@href`, `@src`, `@property`, `@rel`, `@rev`, `@content`, `@datatype`, `@inlist`, chaining
//!   through incomplete triples, hanging `@rel`/`@rev` completion, typed literals,
//!   `rdf:XMLLiteral`, plain literals with `xml:lang`, and RDF collections.
//! * **The HTML+RDFa 1.1 extensions** — `@datetime` and the `<time>` element with XSD datatype
//!   inference, `rdf:HTML` literals, the `<head>`/`<body>` subject rule, the rule that drops bare
//!   terms from `@rel`/`@rev` when `@property` is on the same element, and
//!   [property copying](https://www.w3.org/TR/html-rdfa/#property-copying) through `rdfa:copy`
//!   and `rdfa:Pattern`.
//! * **The initial contexts** — the published
//!   [RDFa Core](http://www.w3.org/2011/rdfa-context/rdfa-1.1),
//!   [XHTML](http://www.w3.org/2011/rdfa-context/xhtml-rdfa-1.1) and
//!   [HTML](http://www.w3.org/2011/rdfa-context/html-rdfa-1.1) tables, transcribed into
//!   [`context`].
//! * **`@role`** — the [Role Attribute 1.0](https://www.w3.org/TR/role-attribute/) triples, whose
//!   terms always resolve against the XHTML vocabulary.
//!
//! # Example
//!
//! ```
//! use oxixml_rdfa::RdfaParser;
//!
//! let html = r##"<!DOCTYPE html>
//! <html prefix="schema: http://schema.org/">
//!   <body>
//!     <div about="#me" typeof="schema:Person">
//!       <span property="schema:name">Natsume Sōseki</span>
//!       <a property="schema:url" href="https://example.com/soseki">home page</a>
//!     </div>
//!   </body>
//! </html>"##;
//!
//! let triples = RdfaParser::new()
//!     .with_base_iri("https://example.com/about")?
//!     .for_slice(html)
//!     .collect::<Result<Vec<_>, _>>()?;
//!
//! assert_eq!(triples.len(), 3);
//! assert!(triples.iter().any(|triple| {
//!     triple.subject.to_string() == "<https://example.com/about#me>"
//!         && triple.predicate.as_str() == "http://schema.org/name"
//!         && triple.object.to_string() == "\"Natsume Sōseki\""
//! }));
//! # Result::<_, Box<dyn std::error::Error>>::Ok(())
//! ```
//!
//! # Host languages
//!
//! RDFa Core leaves several decisions to the language that embeds it, so extraction always happens
//! *as* some host language — see [`HostLanguage`] for the table of what each one changes.
//! [`RdfaParser::for_slice`] parses HTML for [`HostLanguage::Html5`] and XML for every other
//! variant; [`RdfaParser::for_document`] takes a tree you parsed yourself, which is the way to
//! feed it a `text/html` document whose parse errors you want to inspect first.
//!
//! # Conformance
//!
//! `tests/rdfa_suite.rs` runs the [W3C RDFa Test Suite](https://rdfa.info/test-suite/) by
//! generating each input document from the suite's templates exactly as the suite's own harness
//! does, extracting a graph, and evaluating the case's SPARQL `ASK` against it. Against the
//! `rdfa1.1` cases the result is:
//!
//! | host language | result |
//! |---|---|
//! | `html5` | **170 / 170** |
//! | `xhtml1` | **181 / 181** |
//! | `xhtml5` | **177 / 177** |
//! | `xml` | **126 / 126** |
//!
//! The `rdfa1.0`, `html4`, `svg` and `*-invalid` rows of the manifest are counted as skips: this
//! crate implements RDFa 1.1 only, and neither the HTML 4 nor the SVG host language. Point
//! `OXIXML_RDFA_SUITE_DIR` at a checkout of <https://github.com/rdfa/rdfa-website> to run the
//! harness; without the variable it skips silently. `tests/inline.rs` covers the same ground with
//! self-contained cases that need no suite at all.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod attrs;
pub mod context;
mod curie;
mod error;
mod eval;
mod host;
mod literal;
mod parser;

pub use crate::error::{RdfaParseError, RdfaSyntaxError};
pub use crate::host::HostLanguage;
pub use crate::parser::{RdfaParser, RdfaTriples};
