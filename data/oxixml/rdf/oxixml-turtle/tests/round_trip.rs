//! Parse → serialize → parse round-trip coverage for every syntax this crate implements.
//!
//! The inline `test_write` unit tests only assert that a serializer produces one fixed byte
//! string; nothing feeds that output back through the matching parser. These tests close that
//! gap with the awkward literals a fixed-string assertion never covers: control characters,
//! embedded quotes and backslashes, numeric edge cases, non-ASCII IRIs and astral-plane text.

use oxixml_model::vocab::xsd;
use oxixml_model::{
    BlankNode, GraphName, Literal, NamedNode, NamedOrBlankNode, Quad, Term, Triple,
};
use oxixml_turtle::nquads::{NQuadsParser, NQuadsSerializer};
use oxixml_turtle::ntriples::{NTriplesParser, NTriplesSerializer};
use oxixml_turtle::trig::{TriGParser, TriGSerializer};
use oxixml_turtle::turtle::{TurtleParser, TurtleSerializer};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn subject(iri: &str) -> Result<NamedOrBlankNode, Box<dyn std::error::Error>> {
    Ok(NamedNode::new(iri.to_owned())?.into())
}

/// Every object term worth round-tripping: the awkward literals a fixed-string assertion in a
/// unit test never exercises.
fn awkward_objects() -> Result<Vec<Term>, Box<dyn std::error::Error>> {
    Ok(vec![
        Literal::new_simple_literal("line1\nline2\ttab").into(),
        Literal::new_simple_literal("carriage\rreturn").into(),
        Literal::new_simple_literal("a \u{0} nul and a \u{7} bell").into(),
        Literal::new_simple_literal("embedded \"double\" quotes").into(),
        Literal::new_simple_literal("embedded 'single' quotes").into(),
        Literal::new_simple_literal(r"a\backslash\and\more").into(),
        Literal::new_simple_literal(r#""""triple quoted""""#).into(),
        Literal::new_simple_literal("").into(),
        Literal::new_simple_literal("   leading and trailing   ").into(),
        Literal::new_simple_literal("日本語のニックネーム").into(),
        Literal::new_simple_literal("astral: \u{1_0000}\u{1_F600}").into(),
        Literal::new_language_tagged_literal("héllo", "fr-be")?.into(),
        Literal::new_typed_literal("0042", xsd::INTEGER).into(),
        Literal::new_typed_literal("-0", xsd::INTEGER).into(),
        Literal::new_typed_literal("NaN", xsd::DOUBLE).into(),
        Literal::new_typed_literal("INF", xsd::DOUBLE).into(),
        Literal::new_typed_literal("-INF", xsd::DOUBLE).into(),
        Literal::new_typed_literal("1.0E-300", xsd::DOUBLE).into(),
        Literal::new_typed_literal("true", xsd::BOOLEAN).into(),
        Literal::new_typed_literal("0.0000001", xsd::DECIMAL).into(),
        NamedNode::new("http://example.com/héllo/日本語")?.into(),
        NamedNode::new("http://example.com/with%20escape?q=1&r=2#frag")?.into(),
        BlankNode::new("b0")?.into(),
    ])
}

fn awkward_triples() -> Result<Vec<Triple>, Box<dyn std::error::Error>> {
    let predicate = NamedNode::new("http://example.com/predicate")?;
    let mut triples = Vec::new();
    for (index, object) in awkward_objects()?.into_iter().enumerate() {
        triples.push(Triple::new(
            subject(&format!("http://example.com/s{index}"))?,
            predicate.clone(),
            object,
        ));
    }
    // A blank node in subject position too.
    triples.push(Triple::new(
        NamedOrBlankNode::from(BlankNode::new("bsubject")?),
        predicate,
        Literal::new_simple_literal("blank subject"),
    ));
    Ok(triples)
}

/// Blank-node labels are re-generated on parsing, so compare statements with every blank node
/// erased to a single placeholder: this checks the *lexical* round-trip, not blank-node identity.
fn normalized(quad: &Quad) -> String {
    fn term(term: &Term) -> String {
        match term {
            Term::BlankNode(_) => "_:B".to_owned(),
            other => other.to_string(),
        }
    }
    format!(
        "{} {} {} {}",
        match &quad.subject {
            NamedOrBlankNode::BlankNode(_) => "_:B".to_owned(),
            other => other.to_string(),
        },
        quad.predicate,
        term(&quad.object),
        match &quad.graph_name {
            GraphName::BlankNode(_) => "_:B".to_owned(),
            other => other.to_string(),
        }
    )
}

fn normalized_all(quads: &[Quad]) -> Vec<String> {
    let mut normalized: Vec<String> = quads.iter().map(normalized).collect();
    normalized.sort_unstable();
    normalized
}

#[test]
fn turtle_round_trips_awkward_terms() -> TestResult {
    let triples = awkward_triples()?;
    let mut serializer = TurtleSerializer::new().for_writer(Vec::new());
    for triple in &triples {
        serializer.serialize_triple(triple)?;
    }
    let document = serializer.finish()?;

    let parsed = TurtleParser::new()
        .for_slice(&document)
        .collect::<Result<Vec<_>, _>>()?;
    let expected: Vec<Quad> = triples
        .into_iter()
        .map(|triple| triple.in_graph(GraphName::DefaultGraph))
        .collect();
    let parsed: Vec<Quad> = parsed
        .into_iter()
        .map(|triple| triple.in_graph(GraphName::DefaultGraph))
        .collect();
    assert_eq!(normalized_all(&parsed), normalized_all(&expected));
    Ok(())
}

#[test]
fn ntriples_round_trips_awkward_terms() -> TestResult {
    let triples = awkward_triples()?;
    let mut serializer = NTriplesSerializer::new().for_writer(Vec::new());
    for triple in &triples {
        serializer.serialize_triple(triple)?;
    }
    let document = serializer.finish();

    let parsed = NTriplesParser::new()
        .for_slice(&document)
        .collect::<Result<Vec<_>, _>>()?;
    let expected: Vec<Quad> = triples
        .into_iter()
        .map(|triple| triple.in_graph(GraphName::DefaultGraph))
        .collect();
    let parsed: Vec<Quad> = parsed
        .into_iter()
        .map(|triple| triple.in_graph(GraphName::DefaultGraph))
        .collect();
    assert_eq!(normalized_all(&parsed), normalized_all(&expected));
    Ok(())
}

fn awkward_quads() -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let graphs = [
        GraphName::DefaultGraph,
        GraphName::from(NamedNode::new("http://example.com/g1")?),
        GraphName::from(NamedNode::new("http://example.com/graph/日本語")?),
    ];
    let mut quads = Vec::new();
    for (index, triple) in awkward_triples()?.into_iter().enumerate() {
        let graph = graphs[index % graphs.len()].clone();
        quads.push(triple.in_graph(graph));
    }
    Ok(quads)
}

#[test]
fn trig_round_trips_awkward_terms_across_interleaved_graphs() -> TestResult {
    let quads = awkward_quads()?;
    let mut serializer = TriGSerializer::new().for_writer(Vec::new());
    for quad in &quads {
        serializer.serialize_quad(quad)?;
    }
    let document = serializer.finish()?;

    let parsed = TriGParser::new()
        .for_slice(&document)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(normalized_all(&parsed), normalized_all(&quads));
    Ok(())
}

#[test]
fn nquads_round_trips_awkward_terms_across_interleaved_graphs() -> TestResult {
    let quads = awkward_quads()?;
    let mut serializer = NQuadsSerializer::new().for_writer(Vec::new());
    for quad in &quads {
        serializer.serialize_quad(quad)?;
    }
    let document = serializer.finish();

    let parsed = NQuadsParser::new()
        .for_slice(&document)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(normalized_all(&parsed), normalized_all(&quads));
    Ok(())
}

/// Serializing an empty document and parsing it back must yield nothing, in every syntax.
#[test]
fn empty_documents_round_trip() -> TestResult {
    assert!(
        TurtleParser::new()
            .for_slice(&TurtleSerializer::new().for_writer(Vec::new()).finish()?)
            .collect::<Result<Vec<_>, _>>()?
            .is_empty()
    );
    assert!(
        TriGParser::new()
            .for_slice(&TriGSerializer::new().for_writer(Vec::new()).finish()?)
            .collect::<Result<Vec<_>, _>>()?
            .is_empty()
    );
    assert!(
        NTriplesParser::new()
            .for_slice(&NTriplesSerializer::new().for_writer(Vec::new()).finish())
            .collect::<Result<Vec<_>, _>>()?
            .is_empty()
    );
    assert!(
        NQuadsParser::new()
            .for_slice(&NQuadsSerializer::new().for_writer(Vec::new()).finish())
            .collect::<Result<Vec<_>, _>>()?
            .is_empty()
    );
    Ok(())
}

/// Re-serializing the re-parsed statements reproduces the document verbatim, except for the
/// freshly-generated blank-node labels.
#[test]
fn a_second_round_trip_reproduces_every_non_blank_line() -> TestResult {
    let serialize = |triples: &[Triple]| -> Result<String, Box<dyn std::error::Error>> {
        let mut serializer = NTriplesSerializer::new().for_writer(Vec::new());
        for triple in triples {
            serializer.serialize_triple(triple)?;
        }
        Ok(String::from_utf8(serializer.finish())?)
    };
    let first = serialize(&awkward_triples()?)?;
    let reparsed = NTriplesParser::new()
        .for_slice(first.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    let second = serialize(&reparsed)?;

    let without_blanks = |document: &str| -> Vec<String> {
        let mut lines: Vec<String> = document
            .lines()
            .filter(|line| !line.contains("_:"))
            .map(str::to_owned)
            .collect();
        lines.sort_unstable();
        lines
    };
    assert_eq!(first.lines().count(), second.lines().count());
    assert_eq!(without_blanks(&first), without_blanks(&second));
    assert!(!without_blanks(&first).is_empty());
    Ok(())
}
