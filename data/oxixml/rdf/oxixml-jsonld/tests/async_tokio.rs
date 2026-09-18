//! Round-trip tests for the `async-tokio` feature of `oxixml-jsonld`.

#![cfg(feature = "async-tokio")]

use oxixml_jsonld::{JsonLdParser, JsonLdSerializer};
use oxixml_model::vocab::rdf;
use oxixml_model::{GraphName, Literal, NamedNode, Quad};

const FILE: &str = r#"{
    "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
    "@type": "schema:Person",
    "@id": "foo",
    "schema:name": "Foo"
}"#;

#[tokio::test]
async fn async_parser_matches_blocking_parser() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = JsonLdParser::new().for_tokio_async_reader(FILE.as_bytes());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }

    let expected = JsonLdParser::new()
        .for_reader(FILE.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(parsed, expected);
    assert_eq!(parsed.len(), 2);

    // The mid-parse accessors must be live on the async wrapper too.
    assert_eq!(
        parser.prefixes().collect::<Vec<_>>(),
        [("schema", "http://schema.org/")]
    );
    assert_eq!(parser.base_iri(), Some("http://example.com/"));
    Ok(())
}

#[tokio::test]
async fn async_serializer_matches_blocking_serializer() -> Result<(), Box<dyn std::error::Error>> {
    let quads = [
        Quad::new(
            NamedNode::new("http://example.com/me")?,
            rdf::TYPE,
            NamedNode::new("http://schema.org/Person")?,
            GraphName::DefaultGraph,
        ),
        Quad::new(
            NamedNode::new("http://example.com/me")?,
            NamedNode::new("http://schema.org/name")?,
            Literal::new_simple_literal("Me"),
            GraphName::DefaultGraph,
        ),
    ];

    let mut serializer = JsonLdSerializer::new().for_tokio_async_writer(Vec::new());
    for quad in &quads {
        serializer.serialize_quad(quad).await?;
    }
    let asynchronous = serializer.finish().await?;

    let mut blocking = JsonLdSerializer::new().for_writer(Vec::new());
    for quad in &quads {
        blocking.serialize_quad(quad)?;
    }
    assert_eq!(asynchronous, blocking.finish()?);

    // The `@context` envelope must come out of the asynchronous path identically.
    let mut serializer = JsonLdSerializer::new()
        .with_base_iri("http://example.com/")?
        .with_prefix("schema", "http://schema.org/")?
        .for_tokio_async_writer(Vec::new());
    for quad in &quads {
        serializer.serialize_quad(quad).await?;
    }
    let with_context = serializer.finish().await?;

    let mut blocking = JsonLdSerializer::new()
        .with_base_iri("http://example.com/")?
        .with_prefix("schema", "http://schema.org/")?
        .for_writer(Vec::new());
    for quad in &quads {
        blocking.serialize_quad(quad)?;
    }
    assert_eq!(with_context, blocking.finish()?);
    assert!(String::from_utf8(with_context)?.starts_with(
        r#"{"@context":{"@base":"http://example.com/","schema":"http://schema.org/"},"@graph":["#
    ));

    // And the bytes must parse back to the quads we wrote.
    let mut parser = JsonLdParser::new().for_tokio_async_reader(asynchronous.as_slice());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    parsed.sort_by_key(|q| q.predicate.to_string());
    let mut expected = quads.to_vec();
    expected.sort_by_key(|q| q.predicate.to_string());
    assert_eq!(parsed, expected);
    Ok(())
}

/// A reader that yields a single byte per poll: the async wrapper must still drain the whole
/// document before expanding it.
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

    let mut parser = JsonLdParser::new().for_tokio_async_reader(OneByteAtATime {
        data: FILE.as_bytes(),
    });
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }

    let expected = JsonLdParser::new()
        .for_slice(FILE)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(parsed, expected);
    Ok(())
}
