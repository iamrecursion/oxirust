//! [`ReaderJsonParser`]: parses a JSON document from a [`std::io::Read`] source.

use crate::error::JsonParseError;
use crate::event::JsonEvent;
use crate::low_level::{LowLevelJsonParser, LowLevelJsonParserResult};
use std::cmp::min;
use std::io::{self, Read};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncReadExt};

/// The smallest chunk grown onto the internal buffer per [`Read::read`] call.
const MIN_BUFFER_SIZE: usize = 4096;
/// The default ceiling on the internal buffer's size (16 MiB), overridable with
/// [`ReaderJsonParser::with_max_buffer_size`].
const MAX_BUFFER_SIZE: usize = 4096 * 4096;

/// Parses a JSON document from a [`std::io::Read`] implementation.
///
/// # A note on borrowing
///
/// Unlike [`crate::SliceJsonParser`], `ReaderJsonParser` always returns **owned**
/// (`JsonEvent<'static>`) events rather than events borrowing from an internal buffer. This is a
/// deliberate, safety-motivated departure from the upstream `json-event-parser` crate's
/// `ReaderJsonParser`, which returns `JsonEvent<'_>` borrowing from a self-owned buffer by relying
/// on an `unsafe` lifetime-erasure workaround for a
/// [known NLL limitation](https://github.com/rust-lang/rust/issues/70255) (returning a borrow
/// derived from `&mut self` on one path of a loop while mutating `self` on another). OxiXML
/// forbids `unsafe_code` workspace-wide, so `ReaderJsonParser` instead converts each event to an
/// owned value at the point it is produced — a small, one-time allocation per string/number
/// containing a borrowed value, and no allocation at all for the common structural events
/// (`StartObject`, `EndArray`, `Boolean`, `Null`, ...).
///
/// # Example
///
/// ```
/// use oxixml_json::{JsonEvent, ReaderJsonParser};
///
/// let mut reader = ReaderJsonParser::new(b"{\"foo\": 1}".as_slice());
/// assert_eq!(reader.parse_next()?, JsonEvent::StartObject);
/// assert_eq!(reader.parse_next()?, JsonEvent::ObjectKey("foo".into()));
/// assert_eq!(reader.parse_next()?, JsonEvent::Number("1".into()));
/// assert_eq!(reader.parse_next()?, JsonEvent::EndObject);
/// assert_eq!(reader.parse_next()?, JsonEvent::Eof);
/// # Ok::<(), oxixml_json::JsonParseError>(())
/// ```
pub struct ReaderJsonParser<R: Read> {
    input_buffer: Vec<u8>,
    input_buffer_start: usize,
    input_buffer_end: usize,
    max_buffer_size: usize,
    is_ending: bool,
    read: R,
    parser: LowLevelJsonParser,
}

impl<R: Read> ReaderJsonParser<R> {
    /// Creates a new parser reading from `read`.
    #[inline]
    #[must_use]
    pub const fn new(read: R) -> Self {
        Self {
            input_buffer: Vec::new(),
            input_buffer_start: 0,
            input_buffer_end: 0,
            max_buffer_size: MAX_BUFFER_SIZE,
            is_ending: false,
            read,
            parser: LowLevelJsonParser::new(),
        }
    }

    /// Sets the maximum size, in bytes, that the internal buffer is allowed to grow to.
    ///
    /// A single JSON string or number literal that (including any escapes) is longer than this
    /// cannot be parsed; [`parse_next`](Self::parse_next) then fails with an
    /// [`io::ErrorKind::OutOfMemory`] error.
    #[inline]
    #[must_use]
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Sets the maximum allowed nesting depth of arrays and objects.
    ///
    /// See [`LowLevelJsonParser::with_max_stack_size`].
    #[inline]
    #[must_use]
    pub fn with_max_stack_size(mut self, size: usize) -> Self {
        self.parser = self.parser.with_max_stack_size(size);
        self
    }

    /// Reads the next event, pulling more bytes from the underlying reader as needed.
    #[inline]
    pub fn parse_next(&mut self) -> Result<JsonEvent<'static>, JsonParseError> {
        loop {
            let owned_event = {
                let slice = &self.input_buffer[self.input_buffer_start..self.input_buffer_end];
                let LowLevelJsonParserResult {
                    event,
                    consumed_bytes,
                } = self.parser.parse_next(slice, self.is_ending);
                self.input_buffer_start += consumed_bytes;
                // Converting to an owned event here ends any borrow of `self.input_buffer`
                // before we go on to mutate it below.
                event.map(|result| result.map(JsonEvent::into_owned))
            };
            if let Some(event) = owned_event {
                return Ok(event?);
            }
            self.fill_buffer()?;
        }
    }

    /// Compacts already-consumed bytes out of the front of the buffer, grows it if useful, and
    /// pulls in one more [`Read::read`] chunk.
    fn fill_buffer(&mut self) -> io::Result<()> {
        if self.input_buffer_start > 0 {
            self.input_buffer
                .copy_within(self.input_buffer_start..self.input_buffer_end, 0);
            self.input_buffer_end -= self.input_buffer_start;
            self.input_buffer_start = 0;
        }
        if self.input_buffer.len() >= self.max_buffer_size {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!(
                    "Reached the buffer maximal size of {} bytes",
                    self.max_buffer_size
                ),
            ));
        }
        let min_end = min(
            self.input_buffer_end + MIN_BUFFER_SIZE,
            self.max_buffer_size,
        );
        if self.input_buffer.len() < min_end {
            self.input_buffer.resize(min_end, 0);
        }
        if self.input_buffer.len() < self.input_buffer.capacity() {
            // Use all the capacity we already paid for, to avoid needless reallocation later.
            self.input_buffer.resize(self.input_buffer.capacity(), 0);
        }
        let read = self
            .read
            .read(&mut self.input_buffer[self.input_buffer_end..])?;
        self.input_buffer_end += read;
        self.is_ending = read == 0;
        Ok(())
    }
}

/// Parses a JSON document from a [`tokio::io::AsyncRead`] implementation.
///
/// The asynchronous counterpart of [`ReaderJsonParser`], mirroring
/// `json-event-parser`'s `TokioAsyncReaderJsonParser`. It drives the very same
/// [`LowLevelJsonParser`] core over the very same growable buffer; only the refill is awaited,
/// so parsing behaviour and errors are identical to the blocking parser's.
///
/// Like [`ReaderJsonParser`], it yields **owned** (`JsonEvent<'static>`) events rather than
/// events borrowing from the internal buffer, so that no `unsafe` lifetime erasure is needed.
///
/// # Example
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), oxixml_json::JsonParseError> {
/// use oxixml_json::{JsonEvent, TokioAsyncReaderJsonParser};
///
/// let mut reader = TokioAsyncReaderJsonParser::new(b"{\"foo\": 1}".as_slice());
/// assert_eq!(reader.parse_next().await?, JsonEvent::StartObject);
/// assert_eq!(reader.parse_next().await?, JsonEvent::ObjectKey("foo".into()));
/// assert_eq!(reader.parse_next().await?, JsonEvent::Number("1".into()));
/// assert_eq!(reader.parse_next().await?, JsonEvent::EndObject);
/// assert_eq!(reader.parse_next().await?, JsonEvent::Eof);
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncReaderJsonParser<R: AsyncRead + Unpin> {
    input_buffer: Vec<u8>,
    input_buffer_start: usize,
    input_buffer_end: usize,
    max_buffer_size: usize,
    is_ending: bool,
    read: R,
    parser: LowLevelJsonParser,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderJsonParser<R> {
    /// Creates a new parser reading from `read`.
    #[inline]
    #[must_use]
    pub const fn new(read: R) -> Self {
        Self {
            input_buffer: Vec::new(),
            input_buffer_start: 0,
            input_buffer_end: 0,
            max_buffer_size: MAX_BUFFER_SIZE,
            is_ending: false,
            read,
            parser: LowLevelJsonParser::new(),
        }
    }

    /// Sets the maximum size, in bytes, that the internal buffer is allowed to grow to.
    ///
    /// See [`ReaderJsonParser::with_max_buffer_size`].
    #[inline]
    #[must_use]
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Sets the maximum allowed nesting depth of arrays and objects.
    ///
    /// See [`LowLevelJsonParser::with_max_stack_size`].
    #[inline]
    #[must_use]
    pub fn with_max_stack_size(mut self, size: usize) -> Self {
        self.parser = self.parser.with_max_stack_size(size);
        self
    }

    /// Reads the next event, awaiting more bytes from the underlying reader as needed.
    ///
    /// # Errors
    ///
    /// Returns a [`JsonParseError::Io`] if the underlying reader fails or the buffer would have
    /// to grow past its maximal size, and a [`JsonParseError::Syntax`] on malformed JSON.
    #[inline]
    pub async fn parse_next(&mut self) -> Result<JsonEvent<'static>, JsonParseError> {
        loop {
            let owned_event = {
                let slice = &self.input_buffer[self.input_buffer_start..self.input_buffer_end];
                let LowLevelJsonParserResult {
                    event,
                    consumed_bytes,
                } = self.parser.parse_next(slice, self.is_ending);
                self.input_buffer_start += consumed_bytes;
                // Converting to an owned event here ends any borrow of `self.input_buffer`
                // before we go on to mutate it below.
                event.map(|result| result.map(JsonEvent::into_owned))
            };
            if let Some(event) = owned_event {
                return Ok(event?);
            }
            self.fill_buffer().await?;
        }
    }

    /// Compacts already-consumed bytes out of the front of the buffer, grows it if useful, and
    /// awaits one more [`AsyncReadExt::read`] chunk.
    async fn fill_buffer(&mut self) -> io::Result<()> {
        if self.input_buffer_start > 0 {
            self.input_buffer
                .copy_within(self.input_buffer_start..self.input_buffer_end, 0);
            self.input_buffer_end -= self.input_buffer_start;
            self.input_buffer_start = 0;
        }
        if self.input_buffer.len() >= self.max_buffer_size {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!(
                    "Reached the buffer maximal size of {} bytes",
                    self.max_buffer_size
                ),
            ));
        }
        let min_end = min(
            self.input_buffer_end + MIN_BUFFER_SIZE,
            self.max_buffer_size,
        );
        if self.input_buffer.len() < min_end {
            self.input_buffer.resize(min_end, 0);
        }
        if self.input_buffer.len() < self.input_buffer.capacity() {
            // Use all the capacity we already paid for, to avoid needless reallocation later.
            self.input_buffer.resize(self.input_buffer.capacity(), 0);
        }
        let read = self
            .read
            .read(&mut self.input_buffer[self.input_buffer_end..])
            .await?;
        self.input_buffer_end += read;
        self.is_ending = read == 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_from_a_slice_reader() -> Result<(), JsonParseError> {
        let mut reader = ReaderJsonParser::new(br#"{"a": [1, 2, 3]}"#.as_slice());
        assert_eq!(reader.parse_next()?, JsonEvent::StartObject);
        assert_eq!(reader.parse_next()?, JsonEvent::ObjectKey("a".into()));
        assert_eq!(reader.parse_next()?, JsonEvent::StartArray);
        assert_eq!(reader.parse_next()?, JsonEvent::Number("1".into()));
        assert_eq!(reader.parse_next()?, JsonEvent::Number("2".into()));
        assert_eq!(reader.parse_next()?, JsonEvent::Number("3".into()));
        assert_eq!(reader.parse_next()?, JsonEvent::EndArray);
        assert_eq!(reader.parse_next()?, JsonEvent::EndObject);
        assert_eq!(reader.parse_next()?, JsonEvent::Eof);
        Ok(())
    }

    /// A [`Read`] impl that only ever yields one byte per call, to exercise the buffer-growth and
    /// incremental-feeding logic regardless of the platform's pipe buffering.
    struct OneByteAtATime<'a>(&'a [u8]);

    impl Read for OneByteAtATime<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.0[0];
            self.0 = &self.0[1..];
            Ok(1)
        }
    }

    #[test]
    fn parses_with_a_slow_trickling_reader() -> Result<(), JsonParseError> {
        let input = br#"{"key": "a fairly long string value", "n": 12345.678e2}"#;
        let mut reader = ReaderJsonParser::new(OneByteAtATime(input));
        assert_eq!(reader.parse_next()?, JsonEvent::StartObject);
        assert_eq!(reader.parse_next()?, JsonEvent::ObjectKey("key".into()));
        assert_eq!(
            reader.parse_next()?,
            JsonEvent::String("a fairly long string value".into())
        );
        assert_eq!(reader.parse_next()?, JsonEvent::ObjectKey("n".into()));
        assert_eq!(
            reader.parse_next()?,
            JsonEvent::Number("12345.678e2".into())
        );
        assert_eq!(reader.parse_next()?, JsonEvent::EndObject);
        assert_eq!(reader.parse_next()?, JsonEvent::Eof);
        Ok(())
    }

    #[test]
    fn max_buffer_size_is_enforced() -> Result<(), JsonParseError> {
        let input = format!(r#"{{"k": "{}"}}"#, "x".repeat(10_000));
        let mut reader = ReaderJsonParser::new(input.as_bytes()).with_max_buffer_size(64);
        assert_eq!(reader.parse_next()?, JsonEvent::StartObject);
        assert_eq!(reader.parse_next()?, JsonEvent::ObjectKey("k".into()));
        let err = reader.parse_next().expect_err("buffer should overflow");
        assert!(matches!(err, JsonParseError::Io(e) if e.kind() == io::ErrorKind::OutOfMemory));
        Ok(())
    }

    #[test]
    fn max_stack_size_is_configurable() -> Result<(), JsonParseError> {
        let mut reader = ReaderJsonParser::new(b"[[[1]]]".as_slice()).with_max_stack_size(1);
        assert_eq!(reader.parse_next()?, JsonEvent::StartArray);
        assert_eq!(reader.parse_next()?, JsonEvent::StartArray);
        assert!(reader.parse_next().is_err());
        Ok(())
    }

    #[test]
    fn truncated_document_is_a_syntax_error() -> Result<(), JsonParseError> {
        let mut reader = ReaderJsonParser::new(br#"{"a":"#.as_slice());
        assert_eq!(reader.parse_next()?, JsonEvent::StartObject);
        assert_eq!(reader.parse_next()?, JsonEvent::ObjectKey("a".into()));
        let err = reader.parse_next().expect_err("truncated document");
        assert!(matches!(err, JsonParseError::Syntax(_)));
        Ok(())
    }
}
