//! Round-trip tests for the `async-tokio` feature of `oxixml-turtle`.
//!
//! Each format is serialized through its `for_tokio_async_writer` serializer and read back
//! through its `for_tokio_async_reader` parser, asserting that the asynchronous wrappers agree
//! with the blocking ones both in output bytes and in parsed terms.

#![cfg(feature = "async-tokio")]

use oxixml_model::vocab::rdf;
use oxixml_model::{NamedNode, Quad, Triple};
use oxixml_turtle::n3::{N3Parser, N3Quad, N3Term};
use oxixml_turtle::{
    NQuadsParser, NQuadsSerializer, NTriplesParser, NTriplesSerializer, TriGParser, TriGSerializer,
    TurtleParser, TurtleSerializer,
};

fn person() -> Result<NamedNode, Box<dyn std::error::Error>> {
    Ok(NamedNode::new("http://schema.org/Person")?)
}

fn sample_triple() -> Result<Triple, Box<dyn std::error::Error>> {
    Ok(Triple::new(
        NamedNode::new("http://example.com/me")?,
        rdf::TYPE,
        person()?,
    ))
}

fn sample_quad() -> Result<Quad, Box<dyn std::error::Error>> {
    Ok(Quad::new(
        NamedNode::new("http://example.com/me")?,
        rdf::TYPE,
        person()?,
        NamedNode::new("http://example.com/g")?,
    ))
}

#[tokio::test]
async fn ntriples_async_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let triple = sample_triple()?;
    let mut serializer = NTriplesSerializer::new().for_tokio_async_writer(Vec::new());
    serializer.serialize_triple(&triple).await?;
    let buffer = serializer.finish();

    // The asynchronous serializer must produce exactly the blocking serializer's bytes.
    let mut blocking = NTriplesSerializer::new().for_writer(Vec::new());
    blocking.serialize_triple(&triple)?;
    assert_eq!(buffer, blocking.finish());

    let mut parser = NTriplesParser::new().for_tokio_async_reader(buffer.as_slice());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    assert_eq!(parsed, vec![triple]);
    Ok(())
}

#[tokio::test]
async fn nquads_async_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let quad = sample_quad()?;
    let mut serializer = NQuadsSerializer::new().for_tokio_async_writer(Vec::new());
    serializer.serialize_quad(&quad).await?;
    let buffer = serializer.finish();

    let mut blocking = NQuadsSerializer::new().for_writer(Vec::new());
    blocking.serialize_quad(&quad)?;
    assert_eq!(buffer, blocking.finish());

    let mut parser = NQuadsParser::new().for_tokio_async_reader(buffer.as_slice());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    assert_eq!(parsed, vec![quad]);
    Ok(())
}

#[tokio::test]
async fn turtle_async_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let triple = sample_triple()?;
    let mut serializer = TurtleSerializer::new()
        .with_prefix("schema", "http://schema.org/")?
        .for_tokio_async_writer(Vec::new());
    serializer.serialize_triple(&triple).await?;
    let buffer = serializer.finish().await?;
    assert!(String::from_utf8(buffer.clone())?.contains("@prefix schema:"));

    let mut parser = TurtleParser::new().for_tokio_async_reader(buffer.as_slice());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    assert_eq!(parsed, vec![triple]);
    Ok(())
}

#[tokio::test]
async fn trig_async_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let quad = sample_quad()?;
    let mut serializer = TriGSerializer::new().for_tokio_async_writer(Vec::new());
    serializer.serialize_quad(&quad).await?;
    let buffer = serializer.finish().await?;

    let mut parser = TriGParser::new().for_tokio_async_reader(buffer.as_slice());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    assert_eq!(parsed, vec![quad]);
    Ok(())
}

#[tokio::test]
async fn n3_async_parses_and_exposes_prefixes() -> Result<(), Box<dyn std::error::Error>> {
    let file = r#"@base <http://example.com/> .
@prefix schema: <http://schema.org/> .
<me> a schema:Person ."#;

    let mut parser = N3Parser::new().for_tokio_async_reader(file.as_bytes());
    assert_eq!(parser.prefixes().collect::<Vec<_>>(), []);
    assert!(parser.base_iri().is_none());

    let mut parsed: Vec<N3Quad> = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].object, N3Term::NamedNode(person()?));
    assert_eq!(
        parser.prefixes().collect::<Vec<_>>(),
        [("schema", "http://schema.org/")]
    );
    assert_eq!(parser.base_iri(), Some("http://example.com/"));
    Ok(())
}

/// The asynchronous parser must feed the incremental core chunk by chunk, so a reader that
/// hands out one byte at a time has to produce exactly the same triples as a bulk read.
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

    let file = r#"@prefix schema: <http://schema.org/> .
<http://example.com/me> a schema:Person ;
    schema:name "Me" ."#;

    let mut parser = TurtleParser::new().for_tokio_async_reader(OneByteAtATime {
        data: file.as_bytes(),
    });
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }

    let expected = TurtleParser::new()
        .for_slice(file)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(parsed, expected);
    Ok(())
}
