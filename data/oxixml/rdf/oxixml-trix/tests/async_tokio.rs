//! Round-trip tests for the `async-tokio` feature of `oxixml-trix`.

#![cfg(feature = "async-tokio")]

use oxixml_model::{GraphName, NamedNode, Quad};
use oxixml_trix::{TriXParser, TriXSerializer};

fn quads() -> Vec<Quad> {
    vec![
        Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/o"),
            GraphName::DefaultGraph,
        ),
        Quad::new(
            NamedNode::new_unchecked("http://example.com/s2"),
            NamedNode::new_unchecked("http://example.com/p2"),
            NamedNode::new_unchecked("http://example.com/o2"),
            NamedNode::new_unchecked("http://example.com/g"),
        ),
    ]
}

#[tokio::test]
async fn async_serializer_matches_blocking() -> Result<(), Box<dyn std::error::Error>> {
    let mut asynchronous = TriXSerializer::new().for_tokio_async_writer(Vec::new());
    for quad in &quads() {
        asynchronous.serialize_quad(quad).await?;
    }
    let asynchronous = asynchronous.finish().await?;

    let mut blocking = TriXSerializer::new().for_writer(Vec::new());
    for quad in &quads() {
        blocking.serialize_quad(quad)?;
    }
    assert_eq!(asynchronous, blocking.finish()?);
    Ok(())
}

#[tokio::test]
async fn async_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let mut serializer = TriXSerializer::new().for_tokio_async_writer(Vec::new());
    for quad in &quads() {
        serializer.serialize_quad(quad).await?;
    }
    let bytes = serializer.finish().await?;

    let mut parser = TriXParser::new().for_tokio_async_reader(bytes.as_slice());
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    assert_eq!(parsed, quads());

    // And the blocking parser over the same bytes must agree exactly.
    let expected = TriXParser::new()
        .for_reader(bytes.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(parsed, expected);
    Ok(())
}

/// A reader that yields one byte per poll: the async parser drains it fully before parsing, so
/// the result must be unchanged.
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

    let mut serializer = TriXSerializer::new().for_writer(Vec::new());
    for quad in &quads() {
        serializer.serialize_quad(quad)?;
    }
    let bytes = serializer.finish()?;

    let mut parser = TriXParser::new().for_tokio_async_reader(OneByteAtATime { data: &bytes });
    let mut parsed = Vec::new();
    while let Some(next) = parser.next().await {
        parsed.push(next?);
    }
    assert_eq!(parsed, quads());
    Ok(())
}

#[tokio::test]
async fn async_parser_reports_syntax_errors() {
    let mut parser = TriXParser::new().for_tokio_async_reader(b"<not-trix/>".as_slice());
    let error = parser
        .next()
        .await
        .expect("an error")
        .expect_err("a syntax error");
    assert!(!error.to_string().is_empty());
}
