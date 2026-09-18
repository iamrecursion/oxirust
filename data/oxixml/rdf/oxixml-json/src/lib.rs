//! `oxixml-json` — Streaming JSON event parser and writer (replaces `json-event-parser`).
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! This crate turns a JSON document into (or back from) a flat stream of [`JsonEvent`]s —
//! `StartObject`, `ObjectKey`, `String`, `Number`, ..., `EndObject` — rather than building an
//! in-memory tree. That makes it a good fit for building JSON-LD, SPARQL-results-JSON, or any
//! other JSON-based format processor on top without paying for a DOM you are only going to walk
//! once.
//!
//! Conformance targets [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) exactly: every number
//! form the grammar allows (and only those), every string escape including `\uXXXX` surrogate
//! pairs, rejection of raw control characters and lone surrogates in strings, and UTF-8
//! validation of the input. Duplicate object keys are *not* rejected here — RFC 8259 leaves that
//! choice to the application, so every `ObjectKey`/value pair is reported as written.
//!
//! # Choosing an entry point
//!
//! * [`SliceJsonParser`] — the whole document is already in memory as a `&[u8]`. Zero-copy:
//!   strings/numbers that need no escape processing borrow directly from the input.
//! * [`ReaderJsonParser`] — the document comes from a [`std::io::Read`] (a file, a socket, ...).
//!   Always returns owned events (see that type's docs for why).
//! * [`LowLevelJsonParser`] — you are driving your own buffer (e.g. from a non-blocking or
//!   `async` I/O loop) and want to feed bytes in on your own schedule.
//! * [`WriterJsonSerializer`] / [`LowLevelJsonSerializer`] — the reverse direction: turn a
//!   [`JsonEvent`] stream back into bytes.
//!
//! # Example
//!
//! ```
//! use oxixml_json::{JsonEvent, SliceJsonParser, WriterJsonSerializer};
//!
//! // Parse.
//! let mut parser = SliceJsonParser::new(br#"{"answer": 42, "ok": true}"#);
//! assert_eq!(parser.parse_next()?, JsonEvent::StartObject);
//! assert_eq!(parser.parse_next()?, JsonEvent::ObjectKey("answer".into()));
//! assert_eq!(parser.parse_next()?, JsonEvent::Number("42".into()));
//! assert_eq!(parser.parse_next()?, JsonEvent::ObjectKey("ok".into()));
//! assert_eq!(parser.parse_next()?, JsonEvent::Boolean(true));
//! assert_eq!(parser.parse_next()?, JsonEvent::EndObject);
//! assert_eq!(parser.parse_next()?, JsonEvent::Eof);
//!
//! // And write the same events back out.
//! let mut writer = WriterJsonSerializer::new(Vec::new());
//! writer.serialize_event(JsonEvent::StartObject)?;
//! writer.serialize_event(JsonEvent::ObjectKey("answer".into()))?;
//! writer.serialize_event(JsonEvent::Number("42".into()))?;
//! writer.serialize_event(JsonEvent::ObjectKey("ok".into()))?;
//! writer.serialize_event(JsonEvent::Boolean(true))?;
//! writer.serialize_event(JsonEvent::EndObject)?;
//! assert_eq!(writer.finish()?, br#"{"answer":42,"ok":true}"#);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

mod error;
mod event;
mod low_level;
mod reader;
mod slice;
mod writer;

pub use error::{JsonParseError, JsonSyntaxError};
pub use event::JsonEvent;
pub use low_level::{LowLevelJsonParser, LowLevelJsonParserResult};
pub use reader::ReaderJsonParser;
#[cfg(feature = "async-tokio")]
pub use reader::TokioAsyncReaderJsonParser;
pub use slice::SliceJsonParser;
#[cfg(feature = "async-tokio")]
pub use writer::TokioAsyncWriterJsonSerializer;
pub use writer::{LowLevelJsonSerializer, WriterJsonSerializer};

/// A position within a JSON document: re-exported from [`oxixml_support::pos`] so callers of
/// [`JsonSyntaxError::location`] never need to depend on `oxixml-support` directly.
///
/// Its `line` and `column` are **0-based** (the first character of a document is line `0`,
/// column `0`) and its byte `offset` is 0-based too; only its [`Display`](std::fmt::Display)
/// impl switches to the human-facing 1-based convention.
pub use oxixml_support::pos::TextPosition;
