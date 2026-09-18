//! JSON serialization: [`WriterJsonSerializer`] and the underlying [`LowLevelJsonSerializer`].

use crate::event::JsonEvent;
use std::io::{Error, ErrorKind, Result, Write};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncWrite, AsyncWriteExt};

/// A JSON streaming writer writing to a [`Write`] implementation.
///
/// Feed it the same [`JsonEvent`] stream a parser would produce; it rejects (with an
/// [`io::Error`](std::io::Error) of kind [`InvalidInput`](std::io::ErrorKind::InvalidInput)) any
/// sequence that would not itself be valid JSON, e.g. writing an [`JsonEvent::ObjectKey`] outside
/// an object, or [`finish`](Self::finish)ing with an array left open.
///
/// # Example
///
/// ```
/// use oxixml_json::{JsonEvent, WriterJsonSerializer};
///
/// let mut writer = WriterJsonSerializer::new(Vec::new());
/// writer.serialize_event(JsonEvent::StartObject)?;
/// writer.serialize_event(JsonEvent::ObjectKey("foo".into()))?;
/// writer.serialize_event(JsonEvent::Number("1".into()))?;
/// writer.serialize_event(JsonEvent::EndObject)?;
///
/// assert_eq!(writer.finish()?.as_slice(), b"{\"foo\":1}");
/// # std::io::Result::Ok(())
/// ```
pub struct WriterJsonSerializer<W: Write> {
    write: W,
    writer: LowLevelJsonSerializer,
}

impl<W: Write> WriterJsonSerializer<W> {
    /// Creates a new serializer writing to `write`.
    #[inline]
    #[must_use]
    pub const fn new(write: W) -> Self {
        Self {
            write,
            writer: LowLevelJsonSerializer::new(),
        }
    }

    /// Writes one event.
    #[inline]
    pub fn serialize_event(&mut self, event: JsonEvent<'_>) -> Result<()> {
        self.writer.serialize_event(event, &mut self.write)
    }

    /// Finishes serialization, checking that exactly one balanced JSON value was written, and
    /// returns the underlying writer.
    #[inline]
    pub fn finish(self) -> Result<W> {
        self.writer.validate_eof()?;
        Ok(self.write)
    }
}

/// A JSON streaming writer writing to a [`tokio::io::AsyncWrite`] implementation.
///
/// The asynchronous counterpart of [`WriterJsonSerializer`], mirroring
/// `json-event-parser`'s `TokioAsyncWriterJsonSerializer`. Each event is rendered by the very
/// same [`LowLevelJsonSerializer`] into a scratch buffer, which is then flushed to the sink, so
/// the bytes produced and the well-formedness checks applied are identical to the blocking
/// serializer's.
///
/// # Example
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> std::io::Result<()> {
/// use oxixml_json::{JsonEvent, TokioAsyncWriterJsonSerializer};
///
/// let mut writer = TokioAsyncWriterJsonSerializer::new(Vec::new());
/// writer.serialize_event(JsonEvent::StartObject).await?;
/// writer.serialize_event(JsonEvent::ObjectKey("foo".into())).await?;
/// writer.serialize_event(JsonEvent::Number("1".into())).await?;
/// writer.serialize_event(JsonEvent::EndObject).await?;
///
/// assert_eq!(writer.finish()?.as_slice(), b"{\"foo\":1}");
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "async-tokio")]
pub struct TokioAsyncWriterJsonSerializer<W: AsyncWrite + Unpin> {
    write: W,
    writer: LowLevelJsonSerializer,
    buffer: Vec<u8>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterJsonSerializer<W> {
    /// Creates a new serializer writing to `write`.
    #[inline]
    #[must_use]
    pub const fn new(write: W) -> Self {
        Self {
            write,
            writer: LowLevelJsonSerializer::new(),
            buffer: Vec::new(),
        }
    }

    /// Writes one event.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] of kind [`ErrorKind::InvalidInput`] if the event would not produce
    /// valid JSON at this point, or whatever the underlying sink fails with.
    #[inline]
    pub async fn serialize_event(&mut self, event: JsonEvent<'_>) -> Result<()> {
        self.writer.serialize_event(event, &mut self.buffer)?;
        self.write.write_all(&self.buffer).await?;
        self.buffer.clear();
        Ok(())
    }

    /// Finishes serialization, checking that exactly one balanced JSON value was written, and
    /// returns the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] of kind [`ErrorKind::InvalidInput`] if a value was left unbalanced.
    #[inline]
    pub fn finish(self) -> Result<W> {
        self.writer.validate_eof()?;
        Ok(self.write)
    }
}

/// A low-level JSON streaming writer, generic over where its output bytes go.
///
/// You probably want [`WriterJsonSerializer`] instead, which owns its [`Write`] target;
/// `LowLevelJsonSerializer` is useful when the destination is not always the same `Write`
/// implementation from call to call (for instance, alternating between a socket and an in-memory
/// staging buffer).
///
/// # Example
///
/// ```
/// use oxixml_json::{JsonEvent, LowLevelJsonSerializer};
///
/// let mut writer = LowLevelJsonSerializer::new();
/// let mut output = Vec::new();
/// writer.serialize_event(JsonEvent::StartObject, &mut output)?;
/// writer.serialize_event(JsonEvent::ObjectKey("foo".into()), &mut output)?;
/// writer.serialize_event(JsonEvent::Number("1".into()), &mut output)?;
/// writer.serialize_event(JsonEvent::EndObject, &mut output)?;
///
/// assert_eq!(output.as_slice(), b"{\"foo\":1}");
/// # std::io::Result::Ok(())
/// ```
#[derive(Default)]
pub struct LowLevelJsonSerializer {
    state_stack: Vec<JsonState>,
    element_written: bool,
}

impl LowLevelJsonSerializer {
    /// Creates a new serializer at the start of a document.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state_stack: Vec::new(),
            element_written: false,
        }
    }

    /// Writes one event to `write`.
    pub fn serialize_event(&mut self, event: JsonEvent<'_>, mut write: impl Write) -> Result<()> {
        match event {
            JsonEvent::String(s) => {
                self.before_value(&mut write)?;
                write_escaped_json_string(&s, write)
            }
            JsonEvent::Number(number) => {
                self.before_value(&mut write)?;
                write.write_all(number.as_bytes())
            }
            JsonEvent::Boolean(b) => {
                self.before_value(&mut write)?;
                write.write_all(if b { b"true" } else { b"false" })
            }
            JsonEvent::Null => {
                self.before_value(&mut write)?;
                write.write_all(b"null")
            }
            JsonEvent::StartArray => {
                self.before_value(&mut write)?;
                self.state_stack.push(JsonState::OpenArray);
                write.write_all(b"[")
            }
            JsonEvent::EndArray => match self.state_stack.pop() {
                Some(JsonState::OpenArray | JsonState::ContinuationArray) => write.write_all(b"]"),
                Some(other) => {
                    self.state_stack.push(other);
                    Err(Error::new(
                        ErrorKind::InvalidInput,
                        "Closing a not opened array",
                    ))
                }
                None => Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Closing a not opened array",
                )),
            },
            JsonEvent::StartObject => {
                self.before_value(&mut write)?;
                self.state_stack.push(JsonState::OpenObject);
                write.write_all(b"{")
            }
            JsonEvent::EndObject => match self.state_stack.pop() {
                Some(JsonState::OpenObject | JsonState::ContinuationObject) => {
                    write.write_all(b"}")
                }
                Some(other) => {
                    self.state_stack.push(other);
                    Err(Error::new(
                        ErrorKind::InvalidInput,
                        "Closing a not opened object",
                    ))
                }
                None => Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Closing a not opened object",
                )),
            },
            JsonEvent::ObjectKey(key) => {
                match self.state_stack.pop() {
                    Some(JsonState::OpenObject) => {}
                    Some(JsonState::ContinuationObject) => write.write_all(b",")?,
                    _ => {
                        return Err(Error::new(
                            ErrorKind::InvalidInput,
                            "Trying to write an object key outside of an object",
                        ));
                    }
                }
                self.state_stack.push(JsonState::ContinuationObject);
                self.state_stack.push(JsonState::ObjectValue);
                write_escaped_json_string(&key, &mut write)?;
                write.write_all(b":")
            }
            JsonEvent::Eof => Err(Error::new(
                ErrorKind::InvalidInput,
                "Eof is not allowed in a JSON writer: just stop calling serialize_event",
            )),
        }
    }

    /// Emits whatever separator is needed (nothing, `,`, or an error) before writing a value, and
    /// tracks that a root value now exists.
    fn before_value(&mut self, mut write: impl Write) -> Result<()> {
        match self.state_stack.pop() {
            Some(JsonState::OpenArray) => {
                self.state_stack.push(JsonState::ContinuationArray);
                Ok(())
            }
            Some(JsonState::ContinuationArray) => {
                self.state_stack.push(JsonState::ContinuationArray);
                write.write_all(b",")
            }
            Some(state @ (JsonState::OpenObject | JsonState::ContinuationObject)) => {
                self.state_stack.push(state);
                Err(Error::new(
                    ErrorKind::InvalidInput,
                    "An object key was expected, a value was given instead",
                ))
            }
            Some(JsonState::ObjectValue) => Ok(()),
            None => {
                if self.element_written {
                    Err(Error::new(
                        ErrorKind::InvalidInput,
                        "A root JSON value has already been written",
                    ))
                } else {
                    self.element_written = true;
                    Ok(())
                }
            }
        }
    }

    /// Checks that the document is complete: exactly one root value written, and every array or
    /// object opened has since been closed.
    fn validate_eof(&self) -> Result<()> {
        if !self.state_stack.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "The written JSON is not balanced: an object or an array has not been closed",
            ));
        }
        if !self.element_written {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "A JSON document cannot be empty",
            ));
        }
        Ok(())
    }
}

/// The state of an in-progress array or object being serialized.
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
enum JsonState {
    /// Just wrote `[`; the next value needs no leading comma.
    OpenArray,
    /// At least one array element has been written; the next value needs a leading comma.
    ContinuationArray,
    /// Just wrote `{`; the next key needs no leading comma.
    OpenObject,
    /// At least one member has been written; the next key needs a leading comma.
    ContinuationObject,
    /// A key was just written; exactly one value must follow before anything else.
    ObjectValue,
}

/// Writes `s` as a JSON string literal, escaping exactly the characters RFC 8259 requires
/// (`"`, `\`, and control characters `U+0000`-`U+001F`) plus the shorthand escapes for the most
/// common control characters (matching common JSON serializer conventions for readability).
fn write_escaped_json_string(s: &str, mut write: impl Write) -> Result<()> {
    write.write_all(b"\"")?;
    let mut escape_buffer = [b'\\', b'u', 0, 0, 0, 0];
    for c in s.chars() {
        match c {
            '\\' => write.write_all(b"\\\\"),
            '"' => write.write_all(b"\\\""),
            '\u{8}' => write.write_all(b"\\b"),
            '\u{C}' => write.write_all(b"\\f"),
            '\n' => write.write_all(b"\\n"),
            '\r' => write.write_all(b"\\r"),
            '\t' => write.write_all(b"\\t"),
            c if c < '\u{20}' => {
                let mut value = c as u8;
                for slot in escape_buffer.iter_mut().skip(2).rev() {
                    let digit = value % 16;
                    *slot = if digit < 10 {
                        b'0' + digit
                    } else {
                        b'a' + digit - 10
                    };
                    value /= 16;
                }
                write.write_all(&escape_buffer)
            }
            c => {
                let mut utf8_buffer = [0u8; 4];
                write.write_all(c.encode_utf8(&mut utf8_buffer).as_bytes())
            }
        }?;
    }
    write.write_all(b"\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_round_trip() -> Result<()> {
        let events = [
            JsonEvent::StartArray,
            JsonEvent::StartObject,
            JsonEvent::ObjectKey("f\to".into()),
            JsonEvent::String("\0\x08\x0c\n\r\t\"\\".into()),
            JsonEvent::EndObject,
            JsonEvent::Number("112".into()),
            JsonEvent::Boolean(true),
            JsonEvent::Boolean(false),
            JsonEvent::Null,
            JsonEvent::EndArray,
        ];
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        for event in events {
            serializer.serialize_event(event)?;
        }
        assert_eq!(
            serializer.finish()?,
            b"[{\"f\\to\":\"\\u0000\\b\\f\\n\\r\\t\\\"\\\\\"},112,true,false,null]"
        );
        Ok(())
    }

    #[test]
    fn non_ascii_is_written_as_utf8_not_escaped() -> Result<()> {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        serializer.serialize_event(JsonEvent::String("caf\u{e9} \u{1F600}".into()))?;
        assert_eq!(serializer.finish()?, "\"café 😀\"".as_bytes());
        Ok(())
    }

    #[test]
    fn closing_unopened_array_errors() {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        assert!(serializer.serialize_event(JsonEvent::EndArray).is_err());
    }

    #[test]
    fn closing_unopened_object_errors() {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        assert!(serializer.serialize_event(JsonEvent::EndObject).is_err());
    }

    #[test]
    fn object_key_outside_object_errors() {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        assert!(
            serializer
                .serialize_event(JsonEvent::ObjectKey("k".into()))
                .is_err()
        );
    }

    #[test]
    fn value_instead_of_key_errors() -> Result<()> {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        serializer.serialize_event(JsonEvent::StartObject)?;
        assert!(serializer.serialize_event(JsonEvent::Null).is_err());
        Ok(())
    }

    #[test]
    fn eof_event_is_rejected() {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        assert!(serializer.serialize_event(JsonEvent::Eof).is_err());
    }

    #[test]
    fn second_root_value_errors() -> Result<()> {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        serializer.serialize_event(JsonEvent::Null)?;
        assert!(serializer.serialize_event(JsonEvent::Null).is_err());
        Ok(())
    }

    #[test]
    fn finish_rejects_unbalanced_document() -> Result<()> {
        let mut serializer = WriterJsonSerializer::new(Vec::new());
        serializer.serialize_event(JsonEvent::StartArray)?;
        assert!(serializer.finish().is_err());
        Ok(())
    }

    #[test]
    fn finish_rejects_empty_document() {
        let serializer = WriterJsonSerializer::new(Vec::new());
        assert!(serializer.finish().is_err());
    }

    #[test]
    fn low_level_serializer_is_default_constructible() -> Result<()> {
        let mut writer: LowLevelJsonSerializer = LowLevelJsonSerializer::default();
        let mut out = Vec::new();
        writer.serialize_event(JsonEvent::Null, &mut out)?;
        assert_eq!(out, b"null");
        Ok(())
    }
}
