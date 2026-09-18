//! `oxixml-sparql-syntax` — SPARQL 1.1 query and update syntax to algebra.
//!
//! A Pure-Rust, dependency-light replacement for `spargebra`: it parses the full
//! [SPARQL 1.1 Query](https://www.w3.org/TR/sparql11-query/) and
//! [SPARQL 1.1 Update](https://www.w3.org/TR/sparql11-update/) grammars with a hand-written
//! recursive-descent parser, and translates them into a SPARQL algebra
//! ([`crate::algebra`]) that round-trips back to SPARQL text and to the
//! [SSE](https://jena.apache.org/documentation/notes/sse.html) s-expression form.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN Pure-Rust
//! foundation for all XML and RDF processing. RDF terms are reused from [`oxixml_model`], exactly
//! like `spargebra` reuses `oxrdf`.
//!
//! # Cargo features
//!
//! * `sparql-12` — [SPARQL 1.2](https://www.w3.org/TR/sparql12-query/) / SPARQL-star: triple terms
//!   (`<<( ?s ?p ?o )>>`), reified triples (`<< ?s ?p ?o ~ ?r >>`), reifiers and annotation blocks
//!   (`{| … |}`), the `VERSION` declaration, directional language-tagged strings (`"x"@en--ltr`),
//!   the `TRIPLE`/`SUBJECT`/`PREDICATE`/`OBJECT`/`isTRIPLE`/`LANGDIR`/`STRLANGDIR`/`hasLANG`/
//!   `hasLANGDIR` built-ins, and the generalised
//!   [codepoint escapes](https://www.w3.org/TR/sparql12-query/#codepointEscape) of §19.2. It turns
//!   [`oxixml_model`]'s `rdf-12` feature on.
//!
//! # Relation to `spargebra`'s printed form
//!
//! The algebra is `spargebra`-shaped and the printed forms ([`Display`](std::fmt::Display) and
//! [`Query::to_sse`]) are *semantically* equivalent to `spargebra`'s, but they are deliberately
//! **not** byte-for-byte identical. The intentional divergences are:
//!
//! * `ASK` keeps its where clause as-is instead of wrapping it in a projection of every in-scope
//!   variable: `ASK { ?s ?p ?o }` prints as `ASK WHERE { ?s ?p ?o . }` and gives the SSE
//!   `(ask (bgp (triple ?s ?p ?o)))`, where `spargebra` gives
//!   `(ask (project (?o ?p ?s) (bgp (triple ?s ?p ?o))))`. An `ASK` returns a boolean, so the
//!   projection is unobservable.
//! * A `SELECT` whose aggregates are all projected is printed back in its flat
//!   `SELECT (COUNT(*) AS ?c) … GROUP BY …` form rather than as a nested sub-`SELECT`.
//! * Parser-generated names (for `[]`, collections, path expansion and aggregates) are
//!   deterministic for a given input instead of random, so that printing a parsed query is
//!   reproducible. They are still guaranteed never to collide with a name written in the query.
//!
//! What *is* guaranteed is the internal round-trip: parsing the printed form of a query yields an
//! equivalent algebra.
//!
//! # Example
//!
//! ```
//! use oxixml_sparql_syntax::SparqlParser;
//!
//! let query = SparqlParser::new().parse_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o . }")?;
//! assert_eq!(query.to_string(), "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }");
//! assert_eq!(query.to_sse(), "(project (?s ?p ?o) (bgp (triple ?s ?p ?o)))");
//! # Ok::<_, oxixml_sparql_syntax::SparqlSyntaxError>(())
//! ```

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod algebra;
mod error;
mod parser;
pub mod query;
pub mod term;
pub mod update;
pub mod vocab;

pub use crate::error::{SparqlSyntaxError, TextPosition};
pub use crate::query::Query;
pub use crate::update::Update;

use crate::term::NamedNode;
use oxixml_iri::{Iri, IriParseError};
use oxixml_model::OxString;
use std::collections::{HashMap, HashSet};

/// A SPARQL parser.
///
/// ```
/// use oxixml_sparql_syntax::SparqlParser;
///
/// let query_str = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
/// let query = SparqlParser::new().parse_query(query_str)?;
/// assert_eq!(query.to_string(), query_str);
/// # Ok::<_, oxixml_sparql_syntax::SparqlSyntaxError>(())
/// ```
#[must_use]
#[derive(Clone, Default)]
pub struct SparqlParser {
    base_iri: Option<Iri<OxString>>,
    prefixes: HashMap<String, String>,
    custom_aggregate_functions: HashSet<NamedNode>,
}

impl SparqlParser {
    /// Builds a new parser with no base IRI and no predefined prefixes.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Provides an IRI used to resolve the relative IRIs in the parsed operation.
    ///
    /// ```
    /// use oxixml_sparql_syntax::SparqlParser;
    ///
    /// let query = SparqlParser::new()
    ///     .with_base_iri("http://example.com/")?
    ///     .parse_query("SELECT * WHERE { <s> <p> <o> }")?;
    /// assert_eq!(
    ///     query.to_string(),
    ///     "BASE <http://example.com/>\nSELECT * WHERE { <http://example.com/s> <http://example.com/p> <http://example.com/o> . }"
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_base_iri(mut self, base_iri: &str) -> Result<Self, IriParseError> {
        self.base_iri = Some(Iri::parse(OxString::new_owned(base_iri))?);
        Ok(self)
    }

    /// Sets a prefix used during parsing.
    ///
    /// ```
    /// use oxixml_sparql_syntax::SparqlParser;
    ///
    /// let query = SparqlParser::new()
    ///     .with_prefix("ex", "http://example.com/")?
    ///     .parse_query("SELECT * WHERE { ex:s ex:p ex:o }")?;
    /// assert_eq!(
    ///     query.to_string(),
    ///     "SELECT * WHERE { <http://example.com/s> <http://example.com/p> <http://example.com/o> . }"
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: &str,
        prefix_iri: &str,
    ) -> Result<Self, IriParseError> {
        let iri = Iri::parse(prefix_iri.to_owned())?;
        self.prefixes
            .insert(prefix_name.to_owned(), iri.into_inner());
        Ok(self)
    }

    /// Adds a new function to be parsed as a custom aggregate function and not as a regular
    /// custom function.
    ///
    /// ```
    /// use oxixml_sparql_syntax::SparqlParser;
    /// use oxixml_sparql_syntax::term::NamedNode;
    ///
    /// let query = SparqlParser::new()
    ///     .with_custom_aggregate_function(NamedNode::new("http://example.com/concat")?)
    ///     .parse_query(
    ///         "PREFIX ex: <http://example.com/> SELECT (ex:concat(?o) AS ?c) WHERE { ex:s ex:p ?o }",
    ///     )?;
    /// // The variable holding the aggregate result is generated by the parser: its name is built
    /// // from a prefix that cannot occur in the parsed text, so it never clashes with a variable
    /// // written in the query.
    /// assert!(
    ///     query
    ///         .to_sse()
    ///         .contains("(group () (((<http://example.com/concat> ?o) ?")
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_custom_aggregate_function(mut self, name: impl Into<NamedNode>) -> Self {
        self.custom_aggregate_functions.insert(name.into());
        self
    }

    /// Parses the given query string using the already set options.
    ///
    /// ```
    /// use oxixml_sparql_syntax::SparqlParser;
    ///
    /// let query_str = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
    /// let query = SparqlParser::new().parse_query(query_str)?;
    /// assert_eq!(query.to_string(), query_str);
    /// # Ok::<_, oxixml_sparql_syntax::SparqlSyntaxError>(())
    /// ```
    pub fn parse_query(&self, query: &str) -> Result<Query, SparqlSyntaxError> {
        parser::parse_query(
            query,
            self.base_iri.clone(),
            self.prefixes.clone(),
            self.custom_aggregate_functions.clone(),
        )
    }

    /// Parses the given update string using the already set options.
    ///
    /// ```
    /// use oxixml_sparql_syntax::SparqlParser;
    ///
    /// let update_str = "CLEAR ALL ;";
    /// let update = SparqlParser::new().parse_update(update_str)?;
    /// assert_eq!(update.to_string().trim(), update_str);
    /// # Ok::<_, oxixml_sparql_syntax::SparqlSyntaxError>(())
    /// ```
    pub fn parse_update(&self, update: &str) -> Result<Update, SparqlSyntaxError> {
        parser::parse_update(
            update,
            self.base_iri.clone(),
            self.prefixes.clone(),
            self.custom_aggregate_functions.clone(),
        )
    }
}
