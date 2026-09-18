//! Round-trip tests for the `async-tokio` feature of `oxixml-io`.
//!
//! Every [`RdfFormat`] the umbrella supports is serialized asynchronously and read back
//! asynchronously, and both halves are compared against the blocking API.

#![cfg(feature = "async-tokio")]

use oxixml_io::{JsonLdProfileSet, RdfFormat, RdfParser, RdfSerializer};
use oxixml_model::{GraphName, NamedNode, Quad};

/// Every format the umbrella can both write and read back.
fn formats() -> Vec<RdfFormat> {
    vec![
        RdfFormat::JsonLd {
            profile: JsonLdProfileSet::empty(),
        },
        RdfFormat::NQuads,
        RdfFormat::NTriples,
        RdfFormat::RdfXml,
        RdfFormat::TriG,
        RdfFormat::TriX,
        RdfFormat::Turtle,
    ]
}

/// A quad in the default graph, serializable by every format above.
fn quad() -> Quad {
    Quad::new(
        NamedNode::new_unchecked("http://example.com/s"),
        NamedNode::new_unchecked("http://example.com/p"),
        NamedNode::new_unchecked("http://example.com/o"),
        GraphName::DefaultGraph,
    )
}

async fn serialize_async(format: RdfFormat) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut serializer = RdfSerializer::from_format(format).for_tokio_async_writer(Vec::new());
    serializer.serialize_quad(&quad()).await?;
    Ok(serializer.finish().await?)
}

fn serialize_blocking(format: RdfFormat) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut serializer = RdfSerializer::from_format(format).for_writer(Vec::new());
    serializer.serialize_quad(&quad())?;
    Ok(serializer.finish()?)
}

#[tokio::test]
async fn async_serializer_matches_blocking() -> Result<(), Box<dyn std::error::Error>> {
    for format in formats() {
        assert_eq!(
            serialize_async(format).await?,
            serialize_blocking(format)?,
            "serialized bytes differ for {format:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn async_round_trip_every_format() -> Result<(), Box<dyn std::error::Error>> {
    for format in formats() {
        let bytes = serialize_async(format).await?;

        let mut parser = RdfParser::from_format(format).for_tokio_async_reader(bytes.as_slice());
        let mut parsed = Vec::new();
        while let Some(next) = parser.next().await {
            parsed.push(next?);
        }

        let expected = RdfParser::from_format(format)
            .for_reader(bytes.as_slice())
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(parsed, expected, "parsed quads differ for {format:?}");
        assert_eq!(parsed, vec![quad()], "round trip lost data for {format:?}");
    }
    Ok(())
}

#[tokio::test]
async fn async_parser_exposes_prefixes_and_base_iri() -> Result<(), Box<dyn std::error::Error>> {
    let file = r#"@base <http://example.com/> .
@prefix schema: <http://schema.org/> .
<foo> a schema:Person ;
    schema:name "Foo" ."#;

    let mut parser =
        RdfParser::from_format(RdfFormat::Turtle).for_tokio_async_reader(file.as_bytes());
    assert!(parser.prefixes().next().is_none());
    assert!(parser.base_iri().is_none());

    parser.next().await.ok_or("a first quad")??;

    assert_eq!(
        parser.prefixes().collect::<Vec<_>>(),
        [("schema", "http://schema.org/")]
    );
    assert_eq!(parser.base_iri(), Some("http://example.com/"));
    Ok(())
}

/// N-Triples cannot carry a graph name; the async serializer must reject a named-graph quad
/// exactly as the blocking one does.
#[tokio::test]
async fn async_serializer_rejects_named_graph_for_graph_formats() {
    let named = Quad::new(
        NamedNode::new_unchecked("http://example.com/s"),
        NamedNode::new_unchecked("http://example.com/p"),
        NamedNode::new_unchecked("http://example.com/o"),
        NamedNode::new_unchecked("http://example.com/g"),
    );
    let mut serializer =
        RdfSerializer::from_format(RdfFormat::NTriples).for_tokio_async_writer(Vec::new());
    let error = serializer
        .serialize_quad(&named)
        .await
        .expect_err("a named graph must be rejected");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

/// A reader that yields a single byte per poll, exercising the incremental refill path of every
/// streaming backend behind the umbrella.
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

    for format in formats() {
        let bytes = serialize_blocking(format)?;
        let mut parser =
            RdfParser::from_format(format).for_tokio_async_reader(OneByteAtATime { data: &bytes });
        let mut parsed = Vec::new();
        while let Some(next) = parser.next().await {
            parsed.push(next?);
        }
        assert_eq!(parsed, vec![quad()], "chunked read failed for {format:?}");
    }
    Ok(())
}
