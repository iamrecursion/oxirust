//! A small, reusable index over a W3C-style `mf:` Turtle test manifest.
//!
//! [`rdf_tests`](crate::rdf_tests), [`rdf_canon`](crate::rdf_canon) and [`sparql`](crate::sparql)
//! all consume the same manifest shape — the [test-manifest vocabulary][mf] used across the
//! `rdf-tests` and SPARQL suites — so the Turtle parsing and subject/predicate/object indexing
//! live here once instead of three times.
//!
//! This is deliberately not a general RDF store: it keeps every triple in memory indexed by the
//! lexical form of its subject, which is all a manifest driver ever needs (a manifest describes
//! a fixed, small set of files, never a graph large enough to want real indexing).
//!
//! [mf]: http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#

use oxixml_model::{NamedOrBlankNode, Term};
use oxixml_turtle::TurtleParser;
use std::collections::HashMap;

/// `rdf:type`.
pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
/// `rdf:first`.
pub const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
/// `rdf:rest`.
pub const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
/// `rdf:nil`.
pub const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
/// The `mf:` (test-manifest) namespace.
pub const MF: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#";
/// The `qt:` (test-query) namespace, used by the SPARQL manifests' `qt:query`/`qt:data`.
pub const QT: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-query#";

/// The stable key a subject is indexed under: an IRI verbatim, or `_:label` for a blank node.
#[must_use]
pub fn subject_key(subject: &NamedOrBlankNode) -> String {
    match subject {
        NamedOrBlankNode::NamedNode(node) => node.as_str().to_owned(),
        NamedOrBlankNode::BlankNode(node) => format!("_:{}", node.as_str()),
    }
}

/// The IRI of `term`, or `None` when it is not a named node.
#[must_use]
pub fn term_iri(term: &Term) -> Option<&str> {
    match term {
        Term::NamedNode(node) => Some(node.as_str()),
        _ => None,
    }
}

/// The lexical value of `term`, or `None` when it is not a literal.
#[must_use]
pub fn term_literal(term: &Term) -> Option<&str> {
    match term {
        Term::Literal(literal) => Some(literal.value()),
        _ => None,
    }
}

/// A key usable to look a blank- or named-node term up in [`ManifestIndex`] as a subject, or
/// `None` for a literal or (under the `rdf-12` feature of [`oxixml_model`]) a triple term —
/// neither is ever a manifest subject.
///
/// The match is intentionally left non-exhaustive-proof (`_ => None`) rather than cfg-gated on
/// `rdf-12`, so that the RDF 1.2 shape of [`Term`] is a silent behavioural no-op here instead of
/// a build break: a triple term is not a manifest subject either way.
#[must_use]
pub fn term_key(term: &Term) -> Option<String> {
    match term {
        Term::NamedNode(node) => Some(node.as_str().to_owned()),
        Term::BlankNode(node) => Some(format!("_:{}", node.as_str())),
        _ => None,
    }
}

/// A Turtle document indexed as `subject key -> predicate IRI -> objects (encounter order)`.
#[derive(Debug, Default, Clone)]
pub struct ManifestIndex {
    statements: HashMap<String, HashMap<String, Vec<Term>>>,
}

impl ManifestIndex {
    /// Parses `text` as Turtle against `base_iri`, indexing every triple by subject.
    ///
    /// Manifests are conventionally parsed against a synthetic `http://.../invalid/` base so
    /// that every relative reference in them resolves to a predictable, stripped-prefix path —
    /// see [`strip_base`].
    pub fn parse(text: &[u8], base_iri: &str) -> Result<Self, String> {
        let parser = TurtleParser::new()
            .with_base_iri(base_iri)
            .map_err(|error| format!("invalid base IRI {base_iri}: {error}"))?;
        let mut statements: HashMap<String, HashMap<String, Vec<Term>>> = HashMap::new();
        for triple in parser.for_slice(text) {
            let triple = triple.map_err(|error| error.to_string())?;
            statements
                .entry(subject_key(&triple.subject))
                .or_default()
                .entry(triple.predicate.as_str().to_owned())
                .or_default()
                .push(triple.object);
        }
        Ok(Self { statements })
    }

    /// Every object recorded for `subject`/`predicate`, in encounter order; empty when there are
    /// none.
    #[must_use]
    pub fn objects(&self, subject: &str, predicate: &str) -> &[Term] {
        self.statements
            .get(subject)
            .and_then(|predicates| predicates.get(predicate))
            .map_or(&[], Vec::as_slice)
    }

    /// The first recorded object for `subject`/`predicate`.
    #[must_use]
    pub fn first_object(&self, subject: &str, predicate: &str) -> Option<&Term> {
        self.objects(subject, predicate).first()
    }

    /// The IRI of the first `subject`/`predicate` object, when it is a named node.
    #[must_use]
    pub fn first_iri(&self, subject: &str, predicate: &str) -> Option<&str> {
        self.first_object(subject, predicate).and_then(term_iri)
    }

    /// The lexical value of the first `subject`/`predicate` object, when it is a literal.
    #[must_use]
    pub fn first_literal(&self, subject: &str, predicate: &str) -> Option<&str> {
        self.first_object(subject, predicate).and_then(term_literal)
    }

    /// The `rdf:type` values of `subject` that are named nodes.
    pub fn types(&self, subject: &str) -> impl Iterator<Item = &str> {
        self.objects(subject, RDF_TYPE).iter().filter_map(term_iri)
    }

    /// Every subject the index has at least one statement for.
    pub fn subjects(&self) -> impl Iterator<Item = &str> {
        self.statements.keys().map(String::as_str)
    }

    /// Whether the index has no statements at all (an empty or unparseable manifest).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty()
    }

    /// Walks an `rdf:List` starting at `head`, returning its members in order.
    ///
    /// Stops at `rdf:nil`, at a node the index has no `rdf:first`/`rdf:rest` statements for, or
    /// after ten thousand members — whichever comes first — so a malformed or cyclic list
    /// (which a manifest is never supposed to contain) cannot hang the caller.
    #[must_use]
    pub fn list_members(&self, head: &Term) -> Vec<Term> {
        const GUARD: usize = 10_000;
        let mut members = Vec::new();
        let mut current = head.clone();
        for _ in 0..GUARD {
            if matches!(&current, Term::NamedNode(node) if node.as_str() == RDF_NIL) {
                break;
            }
            let Some(key) = term_key(&current) else {
                break;
            };
            if let Some(first) = self.first_object(&key, RDF_FIRST) {
                members.push(first.clone());
            }
            match self.first_object(&key, RDF_REST) {
                Some(rest) => current = rest.clone(),
                None => break,
            }
        }
        members
    }
}

/// Strips `base` off the front of `iri`, recovering the manifest-relative reference that
/// resolved to it. Returns `None` when `iri` is not under `base` at all (a cross-suite
/// `mf:include` that points somewhere this driver does not follow, typically).
#[must_use]
pub fn strip_base<'a>(iri: &'a str, base: &str) -> Option<&'a str> {
    iri.strip_prefix(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "http://oxixml.invalid/manifest-test/";

    #[test]
    fn indexes_a_small_manifest_by_subject_and_predicate() {
        let text = br#"
            @prefix mf: <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> .
            @prefix rdft: <http://www.w3.org/ns/rdftest#> .

            <#t1> a rdft:TestTurtlePositiveSyntax ;
                mf:name "t1" ;
                mf:action <t1.ttl> .
        "#;
        let index = ManifestIndex::parse(text, BASE).expect("valid Turtle");
        let subject = format!("{BASE}#t1");
        assert_eq!(
            index.types(&subject).collect::<Vec<_>>(),
            vec!["http://www.w3.org/ns/rdftest#TestTurtlePositiveSyntax"]
        );
        assert_eq!(
            index.first_literal(&subject, &format!("{MF}name")),
            Some("t1")
        );
        assert_eq!(
            index.first_iri(&subject, &format!("{MF}action")),
            Some(format!("{BASE}t1.ttl").as_str())
        );
    }

    #[test]
    fn walks_an_rdf_list() {
        let text = br#"
            @prefix mf: <http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#> .
            <#m> mf:include ( <a.ttl> <b.ttl> <c.ttl> ) .
        "#;
        let index = ManifestIndex::parse(text, BASE).expect("valid Turtle");
        let subject = format!("{BASE}#m");
        let head = index
            .first_object(&subject, &format!("{MF}include"))
            .expect("mf:include is present")
            .clone();
        let members = index.list_members(&head);
        let iris: Vec<&str> = members.iter().filter_map(term_iri).collect();
        assert_eq!(
            iris,
            [
                format!("{BASE}a.ttl"),
                format!("{BASE}b.ttl"),
                format!("{BASE}c.ttl")
            ]
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_list_and_missing_subject_are_handled() {
        let index = ManifestIndex::parse(b"", BASE).expect("empty Turtle parses");
        assert!(index.is_empty());
        assert!(index.objects("nowhere", "nothing").is_empty());
        assert_eq!(index.first_iri("nowhere", "nothing"), None);
    }

    #[test]
    fn strip_base_recovers_a_relative_reference() {
        assert_eq!(strip_base(&format!("{BASE}x/y.ttl"), BASE), Some("x/y.ttl"));
        assert_eq!(strip_base("http://elsewhere/z.ttl", BASE), None);
    }
}
