//! Round-trip tests for the `async-tokio` feature of `oxixml-json`.

#![cfg(feature = "async-tokio")]

use oxixml_json::{
    JsonEvent, JsonParseError, ReaderJsonParser, TokioAsyncReaderJsonParser,
    TokioAsyncWriterJsonSerializer, WriterJsonSerializer,
};

const DOCUMENT: &str = r#"{"head":{"vars":["a","b"]},"results":{"bindings":[{"a":{"type":"literal","value":"x"}}]},"n":12345.678e2,"t":true,"z":null}"#;

async fn collect_async(
    data: &[u8],
) -> Result<Vec<JsonEvent<'static>>, Box<dyn std::error::Error + Send + Sync>> {
    let mut parser = TokioAsyncReaderJsonParser::new(data);
    let mut events = Vec::new();
    loop {
        let event = parser.parse_next().await?;
        let is_eof = event == JsonEvent::Eof;
        events.push(event);
        if is_eof {
            return Ok(events);
        }
    }
}

fn collect_blocking(
    data: &[u8],
) -> Result<Vec<JsonEvent<'static>>, Box<dyn std::error::Error + Send + Sync>> {
    let mut parser = ReaderJsonParser::new(data);
    let mut events = Vec::new();
    loop {
        let event = parser.parse_next()?;
        let is_eof = event == JsonEvent::Eof;
        events.push(event);
        if is_eof {
            return Ok(events);
        }
    }
}

#[tokio::test]
async fn async_parser_matches_blocking_parser()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    assert_eq!(
        collect_async(DOCUMENT.as_bytes()).await?,
        collect_blocking(DOCUMENT.as_bytes())?
    );
    Ok(())
}

/// A reader that yields one byte per poll: the async parser must refill its buffer repeatedly
/// and still produce exactly the same event stream.
#[tokio::test]
async fn async_parser_handles_chunked_reads() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
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

    let mut parser = TokioAsyncReaderJsonParser::new(OneByteAtATime {
        data: DOCUMENT.as_bytes(),
    });
    let mut events = Vec::new();
    loop {
        let event = parser.parse_next().await?;
        let is_eof = event == JsonEvent::Eof;
        events.push(event);
        if is_eof {
            break;
        }
    }
    assert_eq!(events, collect_blocking(DOCUMENT.as_bytes())?);
    Ok(())
}

#[tokio::test]
async fn async_serializer_matches_blocking_serializer()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let events = collect_blocking(DOCUMENT.as_bytes())?;

    let mut asynchronous = TokioAsyncWriterJsonSerializer::new(Vec::new());
    for event in &events {
        if *event == JsonEvent::Eof {
            break;
        }
        asynchronous.serialize_event(event.clone()).await?;
    }
    let asynchronous = asynchronous.finish()?;

    let mut blocking = WriterJsonSerializer::new(Vec::new());
    for event in &events {
        if *event == JsonEvent::Eof {
            break;
        }
        blocking.serialize_event(event.clone())?;
    }
    assert_eq!(asynchronous, blocking.finish()?);

    // And the bytes must re-parse to the same events.
    assert_eq!(collect_async(&asynchronous).await?, events);
    Ok(())
}

#[tokio::test]
async fn async_parser_enforces_max_buffer_size() {
    let input = format!(r#"{{"k": "{}"}}"#, "x".repeat(10_000));
    let mut parser = TokioAsyncReaderJsonParser::new(input.as_bytes()).with_max_buffer_size(64);
    assert_eq!(
        parser.parse_next().await.expect("start"),
        JsonEvent::StartObject
    );
    assert_eq!(
        parser.parse_next().await.expect("key"),
        JsonEvent::ObjectKey("k".into())
    );
    let error = parser
        .parse_next()
        .await
        .expect_err("buffer should overflow");
    assert!(
        matches!(&error, JsonParseError::Io(e) if e.kind() == std::io::ErrorKind::OutOfMemory),
        "unexpected error: {error}"
    );
}
