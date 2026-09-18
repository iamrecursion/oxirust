//! Integration tests for the `async-tokio` [`Reader`]/[`NsReader`] methods.
//!
//! Mirrors `tests/reader.rs`'s one-byte-at-a-time chunk-boundary corpus (the
//! `OneByteReader`/`collect_one_byte` pair), but drives the reader through
//! [`tokio::io::AsyncBufRead`] instead, yielding a single byte per
//! `poll_fill_buf`. This exercises the same split-sequence handling in the
//! shared `impl_buffered_source!` macro (see `src/reader/buffered_reader.rs`)
//! that the synchronous `BufRead` path exercises, and asserts the async event
//! stream is byte-for-byte identical to both the zero-copy borrowed reader and
//! the synchronous one-byte-chunked reader.

use std::pin::Pin;
use std::task::{Context, Poll};

use oxixml_xml::errors::Error;
use oxixml_xml::events::{BytesEnd, BytesStart, Event};
use oxixml_xml::reader::Reader;
use tokio::io::{self, AsyncBufRead, AsyncRead, ReadBuf};

/// An [`AsyncBufRead`] that hands out at most one byte at a time from
/// `poll_fill_buf`, forcing every markup construct across a chunk boundary.
struct OneByteAsyncReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> OneByteAsyncReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl AsyncRead for OneByteAsyncReader<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let end = (this.pos + 1).min(this.data.len());
        let chunk = &this.data[this.pos..end];
        let n = chunk.len().min(out.remaining());
        out.put_slice(&chunk[..n]);
        this.pos += n;
        Poll::Ready(Ok(()))
    }
}

impl AsyncBufRead for OneByteAsyncReader<'_> {
    fn poll_fill_buf(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<&[u8]>> {
        let this = self.get_mut();
        let end = (this.pos + 1).min(this.data.len());
        Poll::Ready(Ok(&this.data[this.pos..end]))
    }

    fn consume(self: Pin<&mut Self>, amt: usize) {
        let this = self.get_mut();
        this.pos = (this.pos + amt).min(this.data.len());
    }
}

/// Collects all events from a borrowed (`&str`) reader into owned events.
fn collect_borrowed(xml: &str) -> Result<Vec<Event<'static>>, Error> {
    let mut reader = Reader::from_str(xml);
    let mut events = Vec::new();
    loop {
        let event = reader.read_event()?;
        let is_eof = matches!(event, Event::Eof);
        events.push(event.into_owned());
        if is_eof {
            break;
        }
    }
    Ok(events)
}

/// Collects all events from an async reader that yields one byte at a time
/// per `poll_fill_buf`.
async fn collect_one_byte_async(xml: &str) -> Result<Vec<Event<'static>>, Error> {
    let mut reader = Reader::from_reader(OneByteAsyncReader::new(xml.as_bytes()));
    let mut buf = Vec::new();
    let mut events = Vec::new();
    loop {
        let event = reader.read_event_into_async(&mut buf).await?;
        let is_eof = matches!(event, Event::Eof);
        events.push(event.into_owned());
        if is_eof {
            break;
        }
        buf.clear();
    }
    Ok(events)
}

/// The same document corpus as `tests/reader.rs`'s
/// `one_byte_chunks_match_borrowed`, plus a DOCTYPE-with-internal-subset
/// document, so both chunk sources are exercised identically.
fn corpus() -> Vec<&'static str> {
    vec![
        "<?xml version=\"1.0\"?><root a=\"1\" b='2'><child>text</child><!--c--></root>",
        "<a>&amp;&lt;&#48;</a>",
        "<![CDATA[ raw ]] ]]> data ]]><x/>",
        "<!-- comment with -> inside --><y/>",
        "<?target data with ? and > ?><z/>",
        "<a>line1\r\nline2</a>",
        "<!DOCTYPE r [ <!ENTITY e \"a > b\"> <!ATTLIST r a CDATA #IMPLIED> ]><r/>",
    ]
}

#[tokio::test]
async fn one_byte_async_chunks_match_borrowed() -> Result<(), Error> {
    for doc in corpus() {
        let borrowed = collect_borrowed(doc)?;
        let chunked = collect_one_byte_async(doc).await?;
        assert_eq!(borrowed, chunked, "mismatch for doc: {doc:?}");
    }
    Ok(())
}

#[tokio::test]
async fn read_event_into_async_basic_start_end() -> Result<(), Error> {
    let mut reader = Reader::from_reader("<a></a>".as_bytes());
    let mut buf = Vec::new();
    assert_eq!(
        reader.read_event_into_async(&mut buf).await?,
        Event::Start(BytesStart::new("a"))
    );
    buf.clear();
    assert_eq!(
        reader.read_event_into_async(&mut buf).await?,
        Event::End(BytesEnd::new("a"))
    );
    buf.clear();
    assert_eq!(reader.read_event_into_async(&mut buf).await?, Event::Eof);
    Ok(())
}

#[tokio::test]
async fn read_to_end_into_async_skips_nested_content() -> Result<(), Error> {
    let mut reader = Reader::from_reader("<a><inner/></a><tail/>".as_bytes());
    let mut buf = Vec::new();
    assert_eq!(
        reader.read_event_into_async(&mut buf).await?,
        Event::Start(BytesStart::new("a"))
    );
    buf.clear();
    let end = BytesStart::new("a").to_end().into_owned();
    reader.read_to_end_into_async(end.name(), &mut buf).await?;
    buf.clear();
    assert_eq!(
        reader.read_event_into_async(&mut buf).await?,
        Event::Empty(BytesStart::new("tail"))
    );
    Ok(())
}
