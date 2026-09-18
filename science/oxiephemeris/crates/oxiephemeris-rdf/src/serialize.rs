//! Turtle and N-Triples serialization.
//!
//! # Deterministic byte output
//!
//! An [`oxrdf::Graph`] is a *set*: its iteration order is unspecified and
//! in practice hash-dependent. Serializing it directly would give a
//! different byte stream on every run, which would make published
//! documents diff-noisy and defeat content-addressing. Every writer here
//! therefore **sorts the triples** by their canonical N-Triples rendering
//! (subject, then predicate, then object) before emitting them.
//!
//! The result: the same [`crate::ChartResource`] always produces the same
//! bytes, on any machine, in any order the graph happened to be built.

use std::io::Write;

use oxrdf::{Graph, Triple, TripleRef};
use oxttl::{NTriplesSerializer, TurtleSerializer};

use crate::error::RdfError;
use crate::vocab::PREFIXES;

/// The graph's triples in a stable, canonical order.
///
/// Sorting is on the N-Triples rendering of each term, which is injective
/// (two distinct terms never render alike), so the order is a total order
/// and depends only on the graph's content.
#[must_use]
pub fn sorted_triples(graph: &Graph) -> Vec<Triple> {
    // `Graph::iter` yields borrows into the graph; this function hands
    // back an owned, independently-usable `Vec`, so each is cloned out.
    let mut triples: Vec<Triple> = graph.iter().map(TripleRef::into_owned).collect();
    triples.sort_by(|a, b| {
        (
            a.subject.to_string(),
            a.predicate.to_string(),
            a.object.to_string(),
        )
            .cmp(&(
                b.subject.to_string(),
                b.predicate.to_string(),
                b.object.to_string(),
            ))
    });
    triples
}

/// Writes `graph` as Turtle, with this crate's prefix bindings.
///
/// # Errors
///
/// [`RdfError::InvalidPrefix`] if a prefix binding is rejected (a bug in
/// [`PREFIXES`]; the tests pin that it cannot happen), or
/// [`RdfError::Io`] if writing fails.
pub fn write_turtle<W: Write>(graph: &Graph, writer: W) -> Result<(), RdfError> {
    let mut serializer = TurtleSerializer::new();
    for (prefix, namespace) in PREFIXES {
        serializer = serializer
            .with_prefix(*prefix, *namespace)
            .map_err(|_| RdfError::InvalidPrefix((*prefix).to_owned()))?;
    }
    let mut writer = serializer.for_writer(writer);
    for triple in sorted_triples(graph) {
        writer.serialize_triple(&triple)?;
    }
    writer.finish()?;
    Ok(())
}

/// Writes `graph` as N-Triples.
///
/// # Errors
///
/// [`RdfError::Io`] if writing fails.
pub fn write_ntriples<W: Write>(graph: &Graph, writer: W) -> Result<(), RdfError> {
    let mut writer = NTriplesSerializer::new().for_writer(writer);
    for triple in sorted_triples(graph) {
        writer.serialize_triple(&triple)?;
    }
    // N-Triples has no trailing state, so `finish` just hands the writer
    // back rather than returning a `Result`.
    let _ = writer.finish();
    Ok(())
}

/// Serializes `graph` to a Turtle `String`.
///
/// # Errors
///
/// As [`write_turtle`]; additionally [`RdfError::Io`] if the produced
/// bytes are not valid UTF-8, which cannot happen for Turtle.
pub fn to_turtle_string(graph: &Graph) -> Result<String, RdfError> {
    let mut buffer = Vec::new();
    write_turtle(graph, &mut buffer)?;
    String::from_utf8(buffer).map_err(|e| {
        RdfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

/// Serializes `graph` to an N-Triples `String`.
///
/// # Errors
///
/// As [`write_ntriples`].
pub fn to_ntriples_string(graph: &Graph) -> Result<String, RdfError> {
    let mut buffer = Vec::new();
    write_ntriples(graph, &mut buffer)?;
    String::from_utf8(buffer).map_err(|e| {
        RdfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skos::concept_scheme_graph;
    use oxrdf::NamedNode;
    use oxttl::TurtleParser;

    fn parse_turtle(text: &str) -> Graph {
        let mut graph = Graph::new();
        for triple in TurtleParser::new().for_slice(text.as_bytes()) {
            let Ok(triple) = triple else {
                panic!("emitted Turtle must re-parse");
            };
            graph.insert(&triple);
        }
        graph
    }

    #[test]
    fn all_prefixes_are_accepted_by_the_serializer() {
        let mut serializer = TurtleSerializer::new();
        for (prefix, namespace) in PREFIXES {
            let Ok(next) = serializer.with_prefix(*prefix, *namespace) else {
                panic!("prefix {prefix} -> {namespace} was rejected");
            };
            serializer = next;
        }
    }

    #[test]
    fn turtle_output_is_byte_identical_across_runs() {
        let Ok(a) = to_turtle_string(&concept_scheme_graph()) else {
            panic!("serialization must succeed");
        };
        let Ok(b) = to_turtle_string(&concept_scheme_graph()) else {
            panic!("serialization must succeed");
        };
        assert_eq!(a, b, "Turtle output must be deterministic");
    }

    #[test]
    fn turtle_round_trips_the_whole_vocabulary() {
        let original = concept_scheme_graph();
        let Ok(text) = to_turtle_string(&original) else {
            panic!("serialization must succeed");
        };
        let reparsed = parse_turtle(&text);
        assert_eq!(reparsed.len(), original.len(), "triple count changed");
        for triple in &original {
            assert!(reparsed.contains(triple), "lost triple {triple}");
        }
    }

    #[test]
    fn ntriples_round_trips_and_is_deterministic() {
        let original = concept_scheme_graph();
        let Ok(a) = to_ntriples_string(&original) else {
            panic!("serialization must succeed");
        };
        let Ok(b) = to_ntriples_string(&original) else {
            panic!("serialization must succeed");
        };
        assert_eq!(a, b);
        // N-Triples is a subset of Turtle, so the Turtle parser reads it.
        let reparsed = parse_turtle(&a);
        assert_eq!(reparsed.len(), original.len());
    }

    #[test]
    fn turtle_uses_the_registered_prefixes() {
        let Ok(text) = to_turtle_string(&concept_scheme_graph()) else {
            panic!("serialization must succeed");
        };
        assert!(text.contains("@prefix oxa:"), "oxa prefix missing");
        assert!(text.contains("@prefix skos:"), "skos prefix missing");
        assert!(text.contains("@prefix wd:"), "wd prefix missing");
        assert!(text.contains("@prefix sign:"), "sign prefix missing");
    }

    /// With one prefix per concept scheme, no concept IRI needs its inner
    /// slash escaped — `sign:Scorpio`, not `oxc:sign\/Scorpio`.
    #[test]
    fn concept_iris_are_abbreviated_without_escaping() {
        let Ok(text) = to_turtle_string(&concept_scheme_graph()) else {
            panic!("serialization must succeed");
        };
        assert!(
            text.contains("sign:Scorpio"),
            "sign:Scorpio not abbreviated"
        );
        assert!(text.contains("planet:Mars"), "planet:Mars not abbreviated");
        assert!(
            text.contains("aspect:trine"),
            "aspect:trine not abbreviated"
        );
        assert!(
            !text.contains("\\/"),
            "no local name should need an escaped slash"
        );
    }

    #[test]
    fn sorting_is_total_and_content_only() {
        let mut g1 = Graph::new();
        let mut g2 = Graph::new();
        let s = NamedNode::new_unchecked("https://example.org/s");
        let p1 = NamedNode::new_unchecked("https://example.org/a");
        let p2 = NamedNode::new_unchecked("https://example.org/b");
        let o = NamedNode::new_unchecked("https://example.org/o");
        // Insert in opposite orders.
        g1.insert(&Triple::new(s.clone(), p1.clone(), o.clone()));
        g1.insert(&Triple::new(s.clone(), p2.clone(), o.clone()));
        g2.insert(&Triple::new(s.clone(), p2, o.clone()));
        g2.insert(&Triple::new(s, p1, o));
        assert_eq!(sorted_triples(&g1), sorted_triples(&g2));
    }
}
