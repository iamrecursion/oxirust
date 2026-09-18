//! Round-trip tests for the `async-tokio` feature of `oxixml-rdfxml`.

#![cfg(feature = "async-tokio")]

use oxixml_model::vocab::rdf;
use oxixml_model::{Literal, NamedNode, Triple};
use oxixml_rdfxml::{RdfXmlParser, RdfXmlSerializer};

const FILE: &str = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/" xml:base="http://example.com/">
 <rdf:Description rdf:about="foo">
   <rdf:type rdf:resource="http://schema.org/Person" />
   <schema:name>Foo</schema:name>
 </rdf:Description>
</rdf:RDF>"#;

#[tokio::test]
async fn async_parser_matches_blocking_parser() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = RdfXmlParser::new().for_tokio_async_reader(FILE.as_bytes());
    assert_eq!(parser.prefixes().collect::<Vec<_>>(), []);
    assert!(parser.base_iri().is_none());

    let mut parsed = vec![parser.next().await.ok_or("a first triple")??];

    // The mid-parse accessors must be live on the async wrapper too, exactly as on the
    // blocking one: they reflect the bindings currently in scope, so they are only populated
    // while the parser sits inside the document element.
    assert_eq!(
        parser.prefixes().collect::<Vec<_>>(),
        [
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("schema", "http://schema.org/")
        ]
    );
    assert_eq!(parser.base_iri(), Some("http://example.com/"));
    assert!(parser.buffer_position() > 0);

    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }

    let expected = RdfXmlParser::new()
        .for_reader(FILE.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(parsed, expected);
    assert_eq!(parsed.len(), 2);
    Ok(())
}

#[tokio::test]
async fn async_serializer_matches_blocking_serializer() -> Result<(), Box<dyn std::error::Error>> {
    let triples = [
        Triple::new(
            NamedNode::new("http://example.com/me")?,
            rdf::TYPE,
            NamedNode::new("http://schema.org/Person")?,
        ),
        Triple::new(
            NamedNode::new("http://example.com/me")?,
            NamedNode::new("http://schema.org/name")?,
            Literal::new_simple_literal("Me"),
        ),
    ];

    let mut serializer = RdfXmlSerializer::new()
        .with_prefix("schema", "http://schema.org/")?
        .for_tokio_async_writer(Vec::new());
    for triple in &triples {
        serializer.serialize_triple(triple).await?;
    }
    let asynchronous = serializer.finish().await?;

    let mut blocking = RdfXmlSerializer::new()
        .with_prefix("schema", "http://schema.org/")?
        .for_writer(Vec::new());
    for triple in &triples {
        blocking.serialize_triple(triple)?;
    }
    assert_eq!(asynchronous, blocking.finish()?);

    // And the bytes must parse back to the triples we wrote.
    let mut parser = RdfXmlParser::new().for_tokio_async_reader(asynchronous.as_slice());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    parsed.sort_by_key(|t| t.predicate.to_string());
    let mut expected = triples.to_vec();
    expected.sort_by_key(|t| t.predicate.to_string());
    assert_eq!(parsed, expected);
    Ok(())
}

/// A reader that yields a single byte per poll, proving the async parser really feeds the
/// incremental core rather than slurping the whole input.
#[tokio::test]
async fn async_parser_handles_chunked_reads() -> Result<(), Box<dyn std::error::Error>> {
    struct OneByteAtATime<'a> {
        data: &'a [u8],
    }

    impl tokio::io::AsyncRead for OneByteAtATime<'_> {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            if let Some((first, rest)) = self.data.split_first() {
                buf.put_slice(std::slice::from_ref(first));
                self.data = rest;
            }
            std::task::Poll::Ready(Ok(()))
        }
    }

    let mut parser = RdfXmlParser::new().for_tokio_async_reader(OneByteAtATime {
        data: FILE.as_bytes(),
    });
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }

    let expected = RdfXmlParser::new()
        .for_slice(FILE)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(parsed, expected);
    Ok(())
}
