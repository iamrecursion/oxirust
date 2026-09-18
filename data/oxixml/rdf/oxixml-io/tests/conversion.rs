//! Integration tests: format detection matrices, cross-format conversion round-trips over a
//! mixed corpus, blank-node renaming, and base IRI injection.

use oxixml_io::{RdfFormat, RdfParser, RdfSerializer};
use oxixml_jsonld::JsonLdProfileSet;
use oxixml_model::{GraphName, Literal, NamedNode, Quad};
use std::collections::BTreeSet;
use std::error::Error;
use std::io::ErrorKind;

type TestResult = Result<(), Box<dyn Error>>;

fn json_ld() -> RdfFormat {
    RdfFormat::JsonLd {
        profile: JsonLdProfileSet::empty(),
    }
}

// ---------------------------------------------------------------------------------------------
// Format detection matrices
// ---------------------------------------------------------------------------------------------

#[test]
fn extension_detection_matrix() {
    let cases = [
        ("nt", RdfFormat::NTriples),
        ("txt", RdfFormat::NTriples),
        ("nq", RdfFormat::NQuads),
        ("ttl", RdfFormat::Turtle),
        ("trig", RdfFormat::TriG),
        ("n3", RdfFormat::N3),
        ("rdf", RdfFormat::RdfXml),
        ("xml", RdfFormat::RdfXml),
        ("trix", RdfFormat::TriX),
        ("json", json_ld()),
        ("jsonld", json_ld()),
    ];
    for (extension, expected) in cases {
        assert_eq!(
            RdfFormat::from_extension(extension),
            Some(expected),
            "extension {extension:?}"
        );
        // Extension matching is case-insensitive.
        assert_eq!(
            RdfFormat::from_extension(&extension.to_ascii_uppercase()),
            Some(expected),
            "uppercased extension {extension:?}"
        );
    }
    assert_eq!(RdfFormat::from_extension("bogus"), None);
}

#[test]
fn media_type_detection_matrix() {
    let cases = [
        ("text/turtle", RdfFormat::Turtle),
        ("text/turtle; charset=utf-8", RdfFormat::Turtle),
        ("text/turtle;charset=UTF-8", RdfFormat::Turtle),
        ("application/x-turtle", RdfFormat::Turtle),
        ("application/n-triples", RdfFormat::NTriples),
        ("application/n-quads", RdfFormat::NQuads),
        ("application/trig", RdfFormat::TriG),
        ("text/n3", RdfFormat::N3),
        ("application/rdf+xml", RdfFormat::RdfXml),
        ("application/xml", RdfFormat::RdfXml),
        ("application/trix+xml", RdfFormat::TriX),
        ("application/x-trix+xml", RdfFormat::TriX),
        ("application/ld+json", json_ld()),
        ("application/ld+json; charset=utf-8", json_ld()),
    ];
    for (media_type, expected) in cases {
        assert_eq!(
            RdfFormat::from_media_type(media_type),
            Some(expected),
            "media type {media_type:?}"
        );
    }
    // An unsupported charset parameter must be rejected outright, matching `oxrdfio`.
    assert_eq!(
        RdfFormat::from_media_type("text/turtle; charset=latin1"),
        None
    );
    assert_eq!(RdfFormat::from_media_type("text/csv"), None);
    assert_eq!(RdfFormat::from_media_type("not-a-media-type"), None);
}

#[test]
fn every_format_round_trips_its_own_metadata() {
    let formats = [
        RdfFormat::N3,
        RdfFormat::NQuads,
        RdfFormat::NTriples,
        RdfFormat::RdfXml,
        RdfFormat::TriG,
        RdfFormat::TriX,
        RdfFormat::Turtle,
        json_ld(),
    ];
    let mut extensions = BTreeSet::new();
    for format in formats {
        assert_eq!(
            RdfFormat::from_extension(format.file_extension()),
            Some(format),
            "{format} extension does not round-trip"
        );
        // Every format's canonical extension is recognized (no accidental collisions hiding a
        // format behind another one's detection).
        extensions.insert(format.file_extension());
    }
    assert_eq!(extensions.len(), formats.len());
}

// ---------------------------------------------------------------------------------------------
// Cross-format conversion round-trips over a mixed corpus
// ---------------------------------------------------------------------------------------------

/// A small dataset spanning the default graph and one named graph, mixing a plain, a
/// language-tagged, and a datatyped literal.
fn corpus() -> Vec<Quad> {
    let iri = |local: &str| NamedNode::new_unchecked(format!("http://example.com/{local}"));
    vec![
        Quad::new(
            iri("alice"),
            iri("name"),
            Literal::new_simple_literal("Alice"),
            GraphName::DefaultGraph,
        ),
        Quad::new(
            iri("alice"),
            iri("knows"),
            iri("bob"),
            GraphName::DefaultGraph,
        ),
        Quad::new(
            iri("bob"),
            iri("name"),
            Literal::new_language_tagged_literal_unchecked("Bob", "en"),
            iri("graph1"),
        ),
        Quad::new(
            iri("bob"),
            iri("age"),
            Literal::new_typed_literal("42", oxixml_model::vocab::xsd::INTEGER),
            iri("graph1"),
        ),
    ]
}

/// Canonical N-Quads-shaped strings, used as an order-independent, format-independent point of
/// comparison between two quad sets (every `Quad`'s [`Display`](std::fmt::Display) is N-Quads
/// compatible, regardless of which format actually produced it).
fn canonical(quads: &[Quad]) -> BTreeSet<String> {
    quads.iter().map(ToString::to_string).collect()
}

#[test]
fn round_trip_over_dataset_formats() -> TestResult {
    let quads = corpus();
    for format in [
        RdfFormat::NQuads,
        RdfFormat::TriG,
        RdfFormat::TriX,
        json_ld(),
    ] {
        assert!(
            format.supports_datasets(),
            "{format} should support datasets"
        );

        let mut serializer = RdfSerializer::from_format(format).for_writer(Vec::new());
        for quad in &quads {
            serializer.serialize_quad(quad)?;
        }
        let bytes = serializer.finish()?;

        let round_tripped = RdfParser::from_format(format)
            .for_reader(bytes.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("{format} round-trip failed: {error}"))?;

        assert_eq!(
            canonical(&round_tripped),
            canonical(&quads),
            "{format} did not round-trip the corpus"
        );
    }
    Ok(())
}

#[test]
fn round_trip_default_graph_over_graph_only_formats() -> TestResult {
    // Graph-only formats can only carry the default-graph slice of the corpus.
    let default_graph_quads: Vec<Quad> = corpus()
        .into_iter()
        .filter(|quad| quad.graph_name.is_default_graph())
        .collect();
    assert_eq!(default_graph_quads.len(), 2);

    for format in [RdfFormat::NTriples, RdfFormat::Turtle, RdfFormat::RdfXml] {
        assert!(
            !format.supports_datasets(),
            "{format} should not support datasets"
        );

        let mut serializer = RdfSerializer::from_format(format).for_writer(Vec::new());
        for quad in &default_graph_quads {
            serializer.serialize_quad(quad)?;
        }
        let bytes = serializer.finish()?;

        let round_tripped = RdfParser::from_format(format)
            .for_reader(bytes.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("{format} round-trip failed: {error}"))?;

        assert_eq!(
            canonical(&round_tripped),
            canonical(&default_graph_quads),
            "{format} did not round-trip the default-graph corpus"
        );
    }
    Ok(())
}

/// A quad outside the default graph handed to a graph-only serializer must fail gracefully with
/// an [`ErrorKind::InvalidInput`] I/O error, matching `oxrdfio`'s `DefaultGraphOnly` behavior —
/// never a panic and never silent data loss.
#[test]
fn graph_only_formats_reject_named_graphs_gracefully() -> TestResult {
    let quad = Quad::new(
        NamedNode::new_unchecked("http://example.com/s"),
        NamedNode::new_unchecked("http://example.com/p"),
        NamedNode::new_unchecked("http://example.com/o"),
        NamedNode::new_unchecked("http://example.com/g"),
    );
    for format in [RdfFormat::NTriples, RdfFormat::Turtle, RdfFormat::RdfXml] {
        let mut serializer = RdfSerializer::from_format(format).for_writer(Vec::new());
        let error = serializer
            .serialize_quad(&quad)
            .expect_err(&format!("{format} must reject a named-graph quad"));
        assert_eq!(error.kind(), ErrorKind::InvalidInput, "{format}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Blank-node renaming
// ---------------------------------------------------------------------------------------------

#[test]
fn rename_blank_nodes_produces_fresh_but_internally_consistent_ids() -> TestResult {
    for format in [RdfFormat::NTriples, RdfFormat::Turtle, RdfFormat::TriG] {
        let file = "_:a <http://example.com/p> _:a .";
        let first = RdfParser::from_format(format)
            .rename_blank_nodes()
            .for_reader(file.as_bytes())
            .collect::<Result<Vec<_>, _>>()?;
        let second = RdfParser::from_format(format)
            .rename_blank_nodes()
            .for_reader(file.as_bytes())
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(first.len(), 1, "{format}");
        // Fresh across independent parses.
        assert_ne!(first, second, "{format}");
        // But the same source blank node maps to the same fresh id within one parse.
        assert_eq!(
            first[0].subject.to_string(),
            first[0].object.to_string(),
            "{format}"
        );
        // And it is no longer the original id.
        assert_ne!(first[0].subject.to_string(), "_:a", "{format}");
    }
    Ok(())
}

#[test]
fn without_rename_blank_node_ids_are_preserved() -> TestResult {
    let file = "_:a <http://example.com/p> <http://example.com/o> .";
    let quads = RdfParser::from_format(RdfFormat::NTriples)
        .for_reader(file.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads[0].subject.to_string(), "_:a");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Base IRI injection
// ---------------------------------------------------------------------------------------------

#[test]
fn base_iri_resolves_relative_iris_where_supported() -> TestResult {
    for format in [RdfFormat::Turtle, RdfFormat::TriG, RdfFormat::N3] {
        let file = "</s> </p> </o> .";
        let quads = RdfParser::from_format(format)
            .with_base_iri("http://example.com")?
            .for_reader(file.as_bytes())
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(quads.len(), 1, "{format}");
        assert_eq!(
            quads[0].subject.to_string(),
            "<http://example.com/s>",
            "{format}"
        );
    }
    Ok(())
}

#[test]
fn base_iri_resolves_relative_iris_in_rdf_xml() -> TestResult {
    let file = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
 <rdf:Description rdf:about="foo">
   <rdf:type rdf:resource="http://schema.org/Person" />
 </rdf:Description>
</rdf:RDF>"#;
    let quads = RdfParser::from_format(RdfFormat::RdfXml)
        .with_base_iri("http://example.com/")?
        .for_reader(file.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads.len(), 1);
    assert_eq!(quads[0].subject.to_string(), "<http://example.com/foo>");
    Ok(())
}

/// N-Triples, N-Quads, and TriX have no notion of a base IRI: `with_base_iri` must be a
/// silent, error-free no-op for them, matching `oxrdfio`.
#[test]
fn base_iri_is_a_silent_no_op_where_unsupported() -> TestResult {
    for format in [RdfFormat::NTriples, RdfFormat::NQuads, RdfFormat::TriX] {
        let _ = RdfParser::from_format(format).with_base_iri("not even a valid IRI at all")?;
    }
    Ok(())
}

/// Regression test for the format-agnostic parallel bulk-load helpers.
#[test]
fn split_for_parallel_parsing_covers_every_quad() -> Result<(), Box<dyn std::error::Error>> {
    let mut document = String::new();
    for i in 0..5_000 {
        document.push_str(&format!(
            "<http://example.com/s{i}> <http://example.com/p> <http://example.com/o{i}> .\n"
        ));
    }

    for target_parallelism in [1usize, 2, 4, 8] {
        let parsers = RdfParser::from_format(RdfFormat::NTriples)
            .split_slice_for_parallel_parsing(&document, target_parallelism);
        assert!(!parsers.is_empty());
        let quads = parsers
            .into_iter()
            .flatten()
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(quads.len(), 5_000, "parallelism {target_parallelism}");
    }

    // A format that cannot be split still yields exactly one, complete parser.
    let turtle = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .";
    let parsers =
        RdfParser::from_format(RdfFormat::Turtle).split_slice_for_parallel_parsing(turtle, 4);
    assert_eq!(parsers.len(), 1);
    assert_eq!(
        parsers
            .into_iter()
            .flatten()
            .collect::<Result<Vec<_>, _>>()?
            .len(),
        1
    );

    // And the file-based variant, over a temporary file.
    let path = std::env::temp_dir().join("oxixml-io-split-parallel-test.nq");
    std::fs::write(&path, document.as_bytes())?;
    let quads = RdfParser::from_format(RdfFormat::NQuads)
        .split_file_for_parallel_parsing(&path, 4)?
        .into_iter()
        .flatten()
        .collect::<Result<Vec<_>, _>>()?;
    std::fs::remove_file(&path)?;
    assert_eq!(quads.len(), 5_000);
    Ok(())
}
