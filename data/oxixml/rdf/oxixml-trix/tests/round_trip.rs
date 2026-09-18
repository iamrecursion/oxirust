//! Round-trip and cross-format interoperability tests.

use std::collections::HashSet;

use oxixml_model::dataset::CanonicalizationAlgorithm;
use oxixml_model::{BlankNode, Dataset, GraphName, Literal, NamedNode, Quad};
use oxixml_trix::{TriXParser, TriXSerializer};
use oxixml_turtle::NQuadsParser;

/// Serializes `quads`, re-parses the result, and returns the parsed quads together with the
/// serialized bytes (for debugging test failures).
fn round_trip(quads: &[Quad]) -> (Vec<Quad>, Vec<u8>) {
    let mut serializer = TriXSerializer::new().for_writer(Vec::new());
    for quad in quads {
        serializer
            .serialize_quad(quad)
            .expect("serialization must not fail for a Vec<u8> sink");
    }
    let bytes = serializer.finish().expect("finish must not fail");
    let parsed = TriXParser::new()
        .for_slice(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| {
            panic!(
                "re-parsing the serialized document failed: {e}\n---\n{}",
                String::from_utf8_lossy(&bytes)
            )
        });
    (parsed, bytes)
}

fn as_set(quads: &[Quad]) -> HashSet<Quad> {
    quads.iter().cloned().collect()
}

#[test]
fn round_trip_every_term_kind() -> Result<(), Box<dyn std::error::Error>> {
    let quads = vec![
        Quad::new(
            NamedNode::new("http://example.com/s1")?,
            NamedNode::new("http://example.com/p1")?,
            NamedNode::new("http://example.com/o1")?,
            NamedNode::new("http://example.com/g1")?,
        ),
        Quad::new(
            BlankNode::new_unchecked("b1"),
            NamedNode::new("http://example.com/p2")?,
            Literal::new_simple_literal("plain value"),
            NamedNode::new("http://example.com/g1")?,
        ),
        Quad::new(
            NamedNode::new("http://example.com/s3")?,
            NamedNode::new("http://example.com/p3")?,
            Literal::new_language_tagged_literal("hello", "en-us")?,
            GraphName::DefaultGraph,
        ),
        Quad::new(
            NamedNode::new("http://example.com/s4")?,
            NamedNode::new("http://example.com/p4")?,
            Literal::new_typed_literal(
                "1.5",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")?,
            ),
            BlankNode::new_unchecked("g2"),
        ),
        Quad::new(
            NamedNode::new("http://example.com/s5")?,
            NamedNode::new("http://example.com/p5")?,
            BlankNode::new_unchecked("b1"),
            NamedNode::new("http://example.com/g1")?,
        ),
    ];
    let (parsed, bytes) = round_trip(&quads);
    assert_eq!(
        as_set(&parsed),
        as_set(&quads),
        "round-tripped quads differ; serialized document was:\n{}",
        String::from_utf8_lossy(&bytes),
    );
    // The two quads sharing `b1` and graph `g1` must still share it after the round trip.
    let b1_subject: oxixml_model::Term = parsed
        .iter()
        .find(|q| q.predicate.as_str() == "http://example.com/p2")
        .expect("p2 quad")
        .subject
        .clone()
        .into();
    let b1_object = parsed
        .iter()
        .find(|q| q.predicate.as_str() == "http://example.com/p5")
        .expect("p5 quad")
        .object
        .clone();
    assert_eq!(b1_subject, b1_object);
    Ok(())
}

#[test]
fn round_trip_unicode_literal() -> Result<(), Box<dyn std::error::Error>> {
    let quads = vec![Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_simple_literal("日本語 emoji \u{1F600} caf\u{e9} <weird> & \"quoted\""),
        GraphName::DefaultGraph,
    )];
    let (parsed, _) = round_trip(&quads);
    assert_eq!(parsed, quads);
    Ok(())
}

/// Regression test: an embedded `\r` in TriX text content used to be written as a raw,
/// unescaped byte. XML mandates that a conformant parser silently rewrite that to `\n` on
/// re-parse (end-of-line handling, XML 1.0 §2.11), so the value did not round-trip. The fix
/// applies to all three kinds of text content TriX writes through `write_text_element` (`<uri>`,
/// `<id>`, and literal values), exercised here together; only a *literal* value has no
/// additional value-level grammar to violate, so only [`round_trip_carriage_return_in_literal`]
/// checks a full round trip. A `\r` is not a legal `NCName` or IRI character to begin with, so a
/// `<uri>`/`<id>` carrying one is correctly rejected by `NCName`/IRI validation on re-parse no
/// matter how well the XML layer preserved it -- `new_unchecked` is used precisely to get such a
/// value past construction and prove the *serializer* still escapes it correctly at the XML
/// level (checked here as literal `&#13;` bytes in the output).
#[test]
fn carriage_return_in_uri_id_and_literal_is_escaped() -> Result<(), Box<dyn std::error::Error>> {
    let quad = Quad::new(
        BlankNode::new_unchecked("b\r1"),
        NamedNode::new_unchecked("http://example.com/p\r1"),
        Literal::new_simple_literal("val\rue"),
        GraphName::DefaultGraph,
    );
    let mut serializer = TriXSerializer::new().for_writer(Vec::new());
    serializer
        .serialize_quad(&quad)
        .expect("a carriage return must be escaped, not rejected");
    let bytes = serializer.finish().expect("finish must not fail");
    let text = String::from_utf8_lossy(&bytes);
    // All three element kinds preserve the carriage return as a character reference rather than
    // a raw byte that end-of-line handling would silently turn into `\n`.
    assert_eq!(text.matches("&#13;").count(), 3, "{text}");
    Ok(())
}

/// Same defect as above, for a literal value in isolation: unlike a `<uri>`/`<id>`, whose value
/// is independently constrained by IRI/`NCName` grammar, a literal's value has no grammar of its
/// own, so (unlike [`carriage_return_in_uri_id_and_literal_is_escaped`]) a full
/// write-then-reparse round trip is the right check here.
#[test]
fn round_trip_carriage_return_in_literal() -> Result<(), Box<dyn std::error::Error>> {
    let quads = vec![Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_simple_literal("val\rue"),
        GraphName::DefaultGraph,
    )];
    let (parsed, bytes) = round_trip(&quads);
    assert_eq!(
        parsed,
        quads,
        "serialized document was:\n{}",
        String::from_utf8_lossy(&bytes)
    );
    Ok(())
}

/// Regression test: code points the XML `Char` production forbids (here, NUL) must be rejected
/// rather than written as non-well-formed XML, for every one of the three kinds of text content
/// TriX writes (`<uri>`, `<id>`, and literal values).
#[test]
fn serialize_rejects_illegal_xml_char() -> Result<(), Box<dyn std::error::Error>> {
    let bad_uri = Quad::new(
        NamedNode::new_unchecked("http://example.com/s\u{0}"),
        NamedNode::new("http://example.com/p")?,
        NamedNode::new("http://example.com/o")?,
        GraphName::DefaultGraph,
    );
    let bad_id = Quad::new(
        BlankNode::new_unchecked("b\u{0}"),
        NamedNode::new("http://example.com/p")?,
        NamedNode::new("http://example.com/o")?,
        GraphName::DefaultGraph,
    );
    let bad_literal = Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_simple_literal("\u{0}"),
        GraphName::DefaultGraph,
    );
    for quad in [bad_uri, bad_id, bad_literal] {
        let mut serializer = oxixml_trix::TriXSerializer::new().for_writer(Vec::new());
        serializer
            .serialize_quad(&quad)
            .expect_err("NUL must be rejected");
    }
    Ok(())
}

/// Same defect as [`carriage_return_in_uri_id_and_literal_is_escaped`], for the two *attribute*
/// values `write_text_element` writes rather than its element text: a `<plainLiteral>`'s
/// `xml:lang` and a `<typedLiteral>`'s `datatype`. These used to go through `oxixml_xml`'s own
/// attribute escaping, which (like element text through `BytesText::new`) never escapes
/// tab/CR/LF. As with `<uri>`/`<id>`, a bare `\r` is not a legal BCP 47 language tag or IRI
/// character to begin with, so `_unchecked` constructors are needed to get one past
/// construction, and a full round trip is not the right check; this asserts the byte-level fix
/// instead.
#[test]
fn carriage_return_in_lang_tag_and_datatype_is_escaped() -> Result<(), Box<dyn std::error::Error>> {
    let mut serializer = TriXSerializer::new().for_writer(Vec::new());
    serializer.serialize_quad(&Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_language_tagged_literal_unchecked("hello", "en\r"),
        GraphName::DefaultGraph,
    ))?;
    serializer.serialize_quad(&Quad::new(
        NamedNode::new("http://example.com/s2")?,
        NamedNode::new("http://example.com/p2")?,
        Literal::new_typed_literal("1", NamedNode::new_unchecked("http://example.com/dt\r")),
        GraphName::DefaultGraph,
    ))?;
    let bytes = serializer.finish()?;
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        !text.contains('\r'),
        "a raw carriage return leaked through: {text}"
    );
    assert_eq!(
        text.matches("&#13;").count(),
        2,
        "expected one escaped carriage return per attribute: {text}"
    );
    Ok(())
}

/// Regression test: code points the XML `Char` production forbids (here, NUL) must be rejected
/// for a `<plainLiteral>`'s `xml:lang` and a `<typedLiteral>`'s `datatype` too, the same as for
/// `<uri>`/`<id>`/literal text in [`serialize_rejects_illegal_xml_char`].
#[test]
fn serialize_rejects_illegal_xml_char_in_lang_tag_and_datatype()
-> Result<(), Box<dyn std::error::Error>> {
    let bad_lang = Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_language_tagged_literal_unchecked("hello", "e\u{0}n"),
        GraphName::DefaultGraph,
    );
    let bad_datatype = Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_typed_literal("1", NamedNode::new_unchecked("http://example.com/dt\u{0}")),
        GraphName::DefaultGraph,
    );
    for quad in [bad_lang, bad_datatype] {
        let mut serializer = TriXSerializer::new().for_writer(Vec::new());
        serializer
            .serialize_quad(&quad)
            .expect_err("NUL must be rejected");
    }
    Ok(())
}

#[test]
fn round_trip_empty_string_values() -> Result<(), Box<dyn std::error::Error>> {
    let quads = vec![Quad::new(
        NamedNode::new("http://example.com/s")?,
        NamedNode::new("http://example.com/p")?,
        Literal::new_simple_literal(""),
        GraphName::DefaultGraph,
    )];
    let (parsed, _) = round_trip(&quads);
    assert_eq!(parsed, quads);
    Ok(())
}

#[test]
fn round_trip_empty_dataset() {
    let (parsed, bytes) = round_trip(&[]);
    assert!(parsed.is_empty());
    assert!(String::from_utf8(bytes).expect("utf-8").contains("TriX"));
}

#[test]
fn round_trip_interleaved_graphs() -> Result<(), Box<dyn std::error::Error>> {
    // Two graphs whose triples are interleaved: the serializer must still produce
    // well-formed, semantically equivalent TriX (by emitting more than one
    // `<graph>` element for the graph that gets interrupted).
    let g1 = NamedNode::new("http://example.com/g1")?;
    let g2 = NamedNode::new("http://example.com/g2")?;
    let quads = vec![
        Quad::new(
            NamedNode::new("http://example.com/s1")?,
            NamedNode::new("http://example.com/p")?,
            NamedNode::new("http://example.com/o1")?,
            g1.clone(),
        ),
        Quad::new(
            NamedNode::new("http://example.com/s2")?,
            NamedNode::new("http://example.com/p")?,
            NamedNode::new("http://example.com/o2")?,
            g2.clone(),
        ),
        Quad::new(
            NamedNode::new("http://example.com/s3")?,
            NamedNode::new("http://example.com/p")?,
            NamedNode::new("http://example.com/o3")?,
            g1,
        ),
    ];
    let (parsed, bytes) = round_trip(&quads);
    assert_eq!(
        as_set(&parsed),
        as_set(&quads),
        "serialized document was:\n{}",
        String::from_utf8_lossy(&bytes)
    );
    Ok(())
}

/// Parses the same dataset as both TriX and N-Quads, canonicalizes both resulting
/// [`Dataset`]s, and checks that they are isomorphic — this is the interop fixture requested
/// for this crate: confirm equal quad sets (up to blank node renaming) across formats.
#[test]
fn interop_with_nquads_is_dataset_isomorphic() -> Result<(), Box<dyn std::error::Error>> {
    let trix = r#"<?xml version="1.0" encoding="utf-8"?>
<TriX xmlns="http://www.w3.org/2004/03/trix/trix-1/">
  <graph>
    <uri>http://example.com/graph1</uri>
    <triple>
      <id>a</id>
      <uri>http://example.com/knows</uri>
      <id>b</id>
    </triple>
    <triple>
      <id>b</id>
      <uri>http://example.com/name</uri>
      <plainLiteral xml:lang="en">Bob</plainLiteral>
    </triple>
    <triple>
      <uri>http://example.com/alice</uri>
      <uri>http://example.com/age</uri>
      <typedLiteral datatype="http://www.w3.org/2001/XMLSchema#integer">42</typedLiteral>
    </triple>
  </graph>
  <graph>
    <triple>
      <uri>http://example.com/alice</uri>
      <uri>http://example.com/knows</uri>
      <id>a</id>
    </triple>
  </graph>
</TriX>"#;

    let nquads = r#"_:a <http://example.com/knows> _:b <http://example.com/graph1> .
_:b <http://example.com/name> "Bob"@en <http://example.com/graph1> .
<http://example.com/alice> <http://example.com/age> "42"^^<http://www.w3.org/2001/XMLSchema#integer> <http://example.com/graph1> .
<http://example.com/alice> <http://example.com/knows> _:a .
"#;

    let mut from_trix = Dataset::new();
    for quad in TriXParser::new().for_slice(trix) {
        from_trix.insert(&quad?);
    }

    let mut from_nquads = Dataset::new();
    for quad in NQuadsParser::new().for_slice(nquads) {
        from_nquads.insert(&quad?);
    }

    assert_eq!(from_trix.len(), from_nquads.len());
    from_trix.canonicalize(CanonicalizationAlgorithm::Unstable);
    from_nquads.canonicalize(CanonicalizationAlgorithm::Unstable);
    assert_eq!(from_trix, from_nquads);
    Ok(())
}
