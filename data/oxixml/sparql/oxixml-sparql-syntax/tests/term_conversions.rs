//! Direct tests of the public `Term`/`Triple` to `TermPattern`/`TriplePattern` conversions in
//! [`oxixml_sparql_syntax::term`].
//!
//! No SPARQL query text exercises these: every internal parser call site only ever builds a
//! `TermPattern` from a `Variable`/`BlankNode`/`NamedNode`/`Literal`/`TriplePattern` it just
//! parsed, never from a bare `Term` or `Triple` — the conversions here are public API surface for
//! external callers (for example, code that already holds a `Term` from an RDF store and wants to
//! use it as a query pattern), so they need their own, direct tests.

use oxixml_sparql_syntax::term::{NamedNode, Term, TermPattern};

fn iri(s: &'static str) -> NamedNode {
    NamedNode::new(s).unwrap_or_else(|e| panic!("failed to parse IRI {s:?}: {e}"))
}

#[test]
fn named_node_term_always_converts() {
    let pattern = TermPattern::try_from(Term::from(iri("http://example.com/s")))
        .expect("a NamedNode term always has a lossless TermPattern");
    assert_eq!(pattern, TermPattern::from(iri("http://example.com/s")));
}

#[test]
fn blank_node_term_always_converts() {
    use oxixml_sparql_syntax::term::BlankNode;

    let bnode = BlankNode::default();
    let pattern = TermPattern::try_from(Term::from(bnode.clone()))
        .expect("a BlankNode term always has a lossless TermPattern");
    assert_eq!(pattern, TermPattern::from(bnode));
}

#[test]
fn literal_term_always_converts() {
    use oxixml_sparql_syntax::term::Literal;

    let literal = Literal::new_simple_literal("hello");
    let pattern = TermPattern::try_from(Term::from(literal.clone()))
        .expect("a Literal term always has a lossless TermPattern");
    assert_eq!(pattern, TermPattern::from(literal));
}

/// The case this conversion was made fallible for: a triple term, which only has a lossless
/// `TermPattern` when this crate's own `sparql-12` feature is on (see the doc comment on the
/// `TryFrom` impl). Under `sparql-12` — this test's own configuration — it always succeeds; the
/// crate cannot express the feature-mismatched case (this crate's `sparql-12` off, but
/// `oxixml-model`'s `rdf-12` on regardless through Cargo feature unification with some other
/// crate in the build) as an ordinary `#[cfg(feature = "...")]` test, since there is no feature of
/// *this* crate that names that combination — it is exercised instead by building this crate
/// directly with `--no-default-features --features oxixml-model/rdf-12` (see the crate's
/// contribution notes / the release checklist for that exact command).
#[cfg(feature = "sparql-12")]
#[test]
fn triple_term_converts_when_sparql_12_is_enabled() {
    use oxixml_sparql_syntax::term::{NamedOrBlankNode, Triple};

    let triple = Triple::new(
        iri("http://example.com/s"),
        iri("http://example.com/p"),
        iri("http://example.com/o"),
    );
    let pattern = TermPattern::try_from(Term::from(triple))
        .expect("a Triple term has a lossless TermPattern when sparql-12 is enabled");
    let TermPattern::Triple(triple_pattern) = pattern else {
        panic!("expected a TermPattern::Triple");
    };
    assert_eq!(
        triple_pattern.subject,
        TermPattern::from(NamedOrBlankNode::from(iri("http://example.com/s")))
    );
}
