//! [`SliceJsonParser`]: parses a JSON document already fully in memory.

use crate::error::JsonSyntaxError;
use crate::event::JsonEvent;
use crate::low_level::{LowLevelJsonParser, LowLevelJsonParserResult};

/// Parses a JSON document that is already fully available as a `&[u8]`.
///
/// Because the whole document is available up front, `SliceJsonParser` can borrow
/// [`JsonEvent::String`]/[`JsonEvent::Number`] content directly from the input slice whenever no
/// escape processing is needed, without allocating.
///
/// # Example
///
/// ```
/// use oxixml_json::{JsonEvent, SliceJsonParser};
///
/// let mut parser = SliceJsonParser::new(br#"{"foo": 1}"#);
/// assert_eq!(parser.parse_next()?, JsonEvent::StartObject);
/// assert_eq!(parser.parse_next()?, JsonEvent::ObjectKey("foo".into()));
/// assert_eq!(parser.parse_next()?, JsonEvent::Number("1".into()));
/// assert_eq!(parser.parse_next()?, JsonEvent::EndObject);
/// assert_eq!(parser.parse_next()?, JsonEvent::Eof);
/// # Ok::<(), oxixml_json::JsonSyntaxError>(())
/// ```
pub struct SliceJsonParser<'a> {
    input_buffer: &'a [u8],
    parser: LowLevelJsonParser,
}

impl<'a> SliceJsonParser<'a> {
    /// Creates a new parser over the whole of `buffer`.
    #[inline]
    #[must_use]
    pub const fn new(buffer: &'a [u8]) -> Self {
        Self {
            input_buffer: buffer,
            parser: LowLevelJsonParser::new(),
        }
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

    /// Reads the next event.
    ///
    /// Once [`JsonEvent::Eof`] (or an error) has been returned, further calls keep returning the
    /// same terminal result rather than reading past the end of the buffer.
    #[inline]
    pub fn parse_next(&mut self) -> Result<JsonEvent<'a>, JsonSyntaxError> {
        loop {
            let LowLevelJsonParserResult {
                event,
                consumed_bytes,
            } = self.parser.parse_next(self.input_buffer, true);
            self.input_buffer = &self.input_buffer[consumed_bytes..];
            if let Some(event) = event {
                return event;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn parses_a_flat_document() -> Result<(), JsonSyntaxError> {
        let mut parser = SliceJsonParser::new(br#"{"foo": 1, "bar": [true, null, "x"]}"#);
        assert_eq!(parser.parse_next()?, JsonEvent::StartObject);
        assert_eq!(parser.parse_next()?, JsonEvent::ObjectKey("foo".into()));
        assert_eq!(parser.parse_next()?, JsonEvent::Number("1".into()));
        assert_eq!(parser.parse_next()?, JsonEvent::ObjectKey("bar".into()));
        assert_eq!(parser.parse_next()?, JsonEvent::StartArray);
        assert_eq!(parser.parse_next()?, JsonEvent::Boolean(true));
        assert_eq!(parser.parse_next()?, JsonEvent::Null);
        assert_eq!(parser.parse_next()?, JsonEvent::String("x".into()));
        assert_eq!(parser.parse_next()?, JsonEvent::EndArray);
        assert_eq!(parser.parse_next()?, JsonEvent::EndObject);
        assert_eq!(parser.parse_next()?, JsonEvent::Eof);
        Ok(())
    }

    #[test]
    fn zero_copy_strings_borrow_from_input() -> Result<(), JsonSyntaxError> {
        let input = br#""plain string, no escapes""#;
        let mut parser = SliceJsonParser::new(input);
        match parser.parse_next()? {
            JsonEvent::String(Cow::Borrowed(s)) => {
                assert_eq!(s, "plain string, no escapes");
            }
            other => panic!("expected a borrowed string, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn escaped_strings_are_owned() -> Result<(), JsonSyntaxError> {
        let mut parser = SliceJsonParser::new(br#""a\tb""#);
        match parser.parse_next()? {
            JsonEvent::String(Cow::Owned(s)) => assert_eq!(s, "a\tb"),
            other => panic!("expected an owned string, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn syntax_error_is_reported() {
        let mut parser = SliceJsonParser::new(b"{,}");
        assert_eq!(parser.parse_next(), Ok(JsonEvent::StartObject));
        assert!(parser.parse_next().is_err());
    }

    #[test]
    fn max_stack_size_is_configurable() {
        let mut parser = SliceJsonParser::new(b"[[[1]]]").with_max_stack_size(1);
        assert_eq!(parser.parse_next(), Ok(JsonEvent::StartArray));
        assert_eq!(parser.parse_next(), Ok(JsonEvent::StartArray));
        assert!(parser.parse_next().is_err());
    }
}
