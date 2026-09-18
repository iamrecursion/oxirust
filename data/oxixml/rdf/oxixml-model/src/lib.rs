//! `oxixml-model` — the RDF 1.1 / 1.2 data model with RDF-star support.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust
//! foundation for all XML and RDF processing. This crate is the `std`-only, dependency-light
//! replacement for [`oxrdf`](https://crates.io/crates/oxrdf), mirroring its public API so that
//! code written against `oxrdf` ports over by changing only the `use` path.
//!
//! # Relationship to `oxrdf`
//!
//! The owned/borrowed split follows `oxrdf` exactly, which is what makes the port mechanical:
//!
//! * Mutators take borrows — [`Graph::insert`], [`Graph::remove`], [`Graph::contains`],
//!   [`Dataset::insert`], [`Dataset::remove`], [`Dataset::contains`] and the
//!   [`GraphViewMut`](dataset::GraphViewMut) equivalents all accept `impl Into<TripleRef<'_>>` /
//!   `impl Into<QuadRef<'_>>`, so an owned value is passed as `&triple`.
//! * Readers hand out borrows — [`Graph::iter`] yields [`TripleRef`], [`Dataset::iter`] yields
//!   [`QuadRef`], and the pattern queries yield [`TripleRef`] / [`TermRef`] / [`NamedNodeRef`] /
//!   [`NamedOrBlankNodeRef`]. Iterating allocates nothing; call `into_owned()` when an owned
//!   term is wanted.
//! * [`vocab`] constants are `Copy` `NamedNodeRef<'static>` values, and [`Literal::datatype`]
//!   returns a [`NamedNodeRef`].
//! * The `&str` constructors ([`NamedNode::new`], [`Literal::new_typed_literal`], …) are bound by
//!   [`IntoOxString`], which accepts every input `oxrdf`'s `impl Into<String>` accepts —
//!   including a `&str` that does not live for `'static`.
//!
//! Three deviations remain, none of which changes a term's meaning:
//!
//! * **Source-compatibility.** [`Term`] additionally implements `PartialEq` against
//!   [`NamedNode`], [`BlankNode`] and [`Literal`], which `oxrdf` does not. `term == named_node`
//!   therefore needs no conversion, but the inverse costs something: in `term == x.into()` the
//!   target of `into()` is now ambiguous, so such a comparison must name it —
//!   `term == Term::from(x)`.
//! * **String type.** The owned accessors ([`NamedNode::into_string`], [`Literal::into_value`], …)
//!   return [`OxString`] rather than [`String`]. `OxString` derefs to `str` and converts into a
//!   [`String`], so only an explicit `let s: String = …` binding needs an added `.into()`.
//! * **Supersets.** Canonicalization adds the bounded
//!   [`try_canonicalize`](Dataset::try_canonicalize) family on top of `oxrdf`'s unbounded
//!   [`canonicalize`](Dataset::canonicalize). `oxrdf`'s methods that are `#[deprecated]` there
//!   (`is_plain`, `LiteralRef::destruct`) are not provided at all.
//!
//! # Terms
//!
//! The building blocks of RDF are modelled as owned/borrowed pairs:
//!
//! * [`NamedNode`] / [`NamedNodeRef`] — IRIs, validated through
//!   [`oxixml-iri`](oxixml_iri).
//! * [`BlankNode`] / [`BlankNodeRef`] — blank nodes, with a random-id [`BlankNode::default`].
//! * [`Literal`] / [`LiteralRef`] — simple, language-tagged, and datatyped literals, with
//!   datatypes drawn from [`oxixml-xsd`](oxixml_xsd) and language tags validated through
//!   [`oxixml-langtag`](oxixml_langtag).
//! * [`Variable`] / [`VariableRef`] — SPARQL variables.
//! * the unions [`Term`], [`NamedOrBlankNode`], [`GraphName`] and the statements [`Triple`] /
//!   [`Quad`] (plus their `*Ref` borrows).
//!
//! Every term implements [`Display`](std::fmt::Display) as its canonical N-Triples / N-Quads
//! serialization, and [`FromStr`](std::str::FromStr) for the reverse direction.
//!
//! # Containers
//!
//! [`Graph`] and [`Dataset`] are in-memory, string-interning collections offering
//! insert/remove/contains/iteration and rich triple-pattern queries, plus blank-node
//! canonicalization.
//!
//! # Features
//!
#![cfg_attr(
    feature = "rdf-12",
    doc = "* `rdf-12` — RDF 1.2 / RDF-star: embedded [`Term::Triple`] terms and directional \
language-tagged strings ([`BaseDirection`])."
)]
#![cfg_attr(
    not(feature = "rdf-12"),
    doc = "* `rdf-12` — RDF 1.2 / RDF-star: embedded `Term::Triple` terms and directional \
language-tagged strings (`BaseDirection`)."
)]
//! * `rdfc-10` — the standard [RDFC-1.0](https://www.w3.org/TR/rdf-canon/) canonicalization
//!   algorithm (SHA-256 / SHA-384, provided by `oxixml-support`).
//!
//! # Example
//!
//! ```
//! use oxixml_model::{Graph, NamedNodeRef, TripleRef};
//!
//! let mut graph = Graph::new();
//! let ex = NamedNodeRef::new("http://example.com/foo")?;
//! let triple = TripleRef::new(ex, ex, ex);
//! graph.insert(triple);
//!
//! assert!(graph.contains(triple));
//! assert_eq!(
//!     graph.to_string(),
//!     "<http://example.com/foo> <http://example.com/foo> <http://example.com/foo> .\n"
//! );
//! # Result::<_, oxixml_model::IriParseError>::Ok(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

mod blank_node;
pub mod dataset;
pub mod graph;
mod interning;
mod literal;
mod named_node;
mod parser;
mod string;
mod triple;
mod variable;
pub mod vocab;

pub use crate::blank_node::{BlankNode, BlankNodeIdParseError, BlankNodeRef};
pub use crate::dataset::Dataset;
pub use crate::graph::Graph;
#[cfg(feature = "rdf-12")]
#[cfg_attr(docsrs, doc(cfg(feature = "rdf-12")))]
pub use crate::literal::BaseDirection;
pub use crate::literal::{Literal, LiteralRef, print_quoted_str};
pub use crate::named_node::{NamedNode, NamedNodeRef};
pub use crate::parser::TermParseError;
pub use crate::string::{IntoOxString, OxStr, OxString};
pub use crate::triple::{
    GraphName, GraphNameRef, NamedOrBlankNode, NamedOrBlankNodeRef, Quad, QuadRef, Term, TermRef,
    Triple, TripleRef, TryFromTermError,
};
pub use crate::variable::{Variable, VariableNameParseError, VariableRef};
pub use oxixml_iri::IriParseError;
pub use oxixml_langtag::LanguageTagParseError;
