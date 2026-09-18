//! The core, allocation-frugal JSON parsing engine: [`LowLevelJsonParser`].

mod lexer;

use crate::error::JsonSyntaxError;
use crate::event::JsonEvent;
use lexer::{JsonLexer, JsonToken};

/// The default maximum number of nested arrays/objects a [`LowLevelJsonParser`] will accept
/// before failing with an error, unless overridden with
/// [`with_max_stack_size`](LowLevelJsonParser::with_max_stack_size).
const DEFAULT_MAX_STACK_SIZE: usize = 65_536;

/// A low-level JSON parser acting directly on caller-provided buffers.
///
/// This is the engine underneath [`crate::SliceJsonParser`] and [`crate::ReaderJsonParser`]; most
/// applications should use one of those instead. Use [`LowLevelJsonParser`] directly only when
/// neither `SliceJsonParser` (whole document in memory) nor `ReaderJsonParser` (a
/// [`std::io::Read`] source) fits — for instance, when driving the parser from a non-blocking
/// I/O loop or an `async` runtime.
///
/// It allocates nothing beyond a small stack tracking how deeply arrays and objects are nested
/// (bounded by [`with_max_stack_size`](Self::with_max_stack_size)) plus, occasionally, an owned
/// [`String`] when a JSON string literal contains an escape sequence.
///
/// # Feeding input
///
/// Each call to [`parse_next`](Self::parse_next) must be given every byte that has not yet been
/// consumed (as reported by the previous call's
/// [`consumed_bytes`](LowLevelJsonParserResult::consumed_bytes)), and `is_ending` must be `true`
/// exactly once all remaining input is already in `input_buffer` (no more bytes will ever
/// arrive).
///
/// ```
/// use oxixml_json::{JsonEvent, LowLevelJsonParser, LowLevelJsonParserResult};
///
/// let mut parser = LowLevelJsonParser::new();
///
/// // The `{` is a complete token by itself: it is consumed and reported immediately.
/// let LowLevelJsonParserResult { consumed_bytes, event } = parser.parse_next(b"{\"fo", false);
/// assert_eq!(consumed_bytes, 1);
/// assert_eq!(event, Some(Ok(JsonEvent::StartObject)));
///
/// // The in-progress string `"fo` cannot be completed yet: no more bytes are consumed and no
/// // event is produced, so the caller must supply more input (unconsumed bytes included again).
/// let LowLevelJsonParserResult { consumed_bytes, event } = parser.parse_next(b"\"fo", false);
/// assert_eq!(consumed_bytes, 0);
/// assert!(event.is_none());
///
/// // Once the closing quote arrives the key can be completed and reported; the following `:`
/// // is left unconsumed for the next call.
/// let LowLevelJsonParserResult { consumed_bytes, event } = parser.parse_next(b"\"foo\":1}", true);
/// assert_eq!(consumed_bytes, 5); // the string token `"foo"` itself
/// assert_eq!(event, Some(Ok(JsonEvent::ObjectKey("foo".into()))));
/// ```
pub struct LowLevelJsonParser {
    lexer: JsonLexer,
    state_stack: Vec<JsonState>,
    max_state_stack_size: usize,
    element_read: bool,
    buffered_event: Option<JsonEvent<'static>>,
}

impl LowLevelJsonParser {
    /// Creates a new parser positioned at the start of a document.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            lexer: JsonLexer::new(),
            state_stack: Vec::new(),
            max_state_stack_size: DEFAULT_MAX_STACK_SIZE,
            element_read: false,
            buffered_event: None,
        }
    }

    /// Sets the maximum allowed nesting depth of arrays and objects. Once this many container
    /// opens are pending, a further `[` or `{` fails with a syntax error rather than growing the
    /// internal stack without bound.
    #[inline]
    #[must_use]
    pub fn with_max_stack_size(mut self, size: usize) -> Self {
        self.max_state_stack_size = size;
        self
    }

    /// Reads the next event out of `input_buffer`.
    ///
    /// `input_buffer` must start at the position implied by every previous call's
    /// `consumed_bytes` (i.e. it must contain no byte that has already been consumed). Set
    /// `is_ending` to `true` once `input_buffer` holds all the remaining input there will ever
    /// be.
    #[inline]
    pub fn parse_next<'a>(
        &mut self,
        input_buffer: &'a [u8],
        is_ending: bool,
    ) -> LowLevelJsonParserResult<'a> {
        if let Some(event) = self.buffered_event.take() {
            return LowLevelJsonParserResult {
                consumed_bytes: 0,
                event: Some(Ok(event)),
            };
        }
        let start_offset = self.lexer.position().offset;
        while let Some(token) = self.lexer.read_next_token(
            &input_buffer[self.relative_offset(start_offset)..],
            is_ending,
        ) {
            let consumed_bytes = self.relative_offset(start_offset);
            match token {
                Ok(token) => {
                    let (event, error) = self.apply_new_token(token);
                    if let Some(message) = error {
                        let error = self.lexer.error_at_last_token(message);
                        self.buffered_event = event.map(JsonEvent::into_owned);
                        return LowLevelJsonParserResult {
                            consumed_bytes,
                            event: Some(Err(error)),
                        };
                    }
                    if let Some(event) = event {
                        return LowLevelJsonParserResult {
                            consumed_bytes,
                            event: Some(Ok(event)),
                        };
                    }
                }
                Err(error) => {
                    return LowLevelJsonParserResult {
                        consumed_bytes,
                        event: Some(Err(error)),
                    };
                }
            }
        }
        let consumed_bytes = self.relative_offset(start_offset);
        LowLevelJsonParserResult {
            consumed_bytes,
            event: if is_ending {
                self.buffered_event = Some(JsonEvent::Eof);
                let pos = self.lexer.position();
                Some(Err(JsonSyntaxError::new(
                    pos..pos,
                    "Unexpected end of file",
                )))
            } else {
                None
            },
        }
    }

    /// Bytes consumed by the lexer since `start_offset`, as a `usize` suitable for slicing.
    #[inline]
    fn relative_offset(&self, start_offset: u64) -> usize {
        let delta = self.lexer.position().offset - start_offset;
        usize::try_from(delta).unwrap_or(usize::MAX)
    }

    /// Advances the state machine with a newly lexed `token`, returning the [`JsonEvent`] it
    /// produces (if any) and an error message (if the token was not valid at this point in the
    /// grammar). An error and an event can be returned together: the event is still meaningful
    /// (e.g. `EndObject` despite a trailing comma) and is replayed after the error is reported.
    #[inline]
    fn apply_new_token<'a>(
        &mut self,
        token: JsonToken<'a>,
    ) -> (Option<JsonEvent<'a>>, Option<String>) {
        match self.state_stack.pop() {
            Some(JsonState::ObjectKeyOrEnd) => {
                if token == JsonToken::ClosingCurlyBracket {
                    (Some(JsonEvent::EndObject), None)
                } else {
                    if let Err(e) = self.push_state_stack(JsonState::ObjectKey) {
                        return (None, Some(e));
                    }
                    self.apply_new_token(token)
                }
            }
            Some(JsonState::ObjectKey) => {
                if token == JsonToken::ClosingCurlyBracket {
                    return (
                        Some(JsonEvent::EndObject),
                        Some("Trailing commas are not allowed".into()),
                    );
                }
                if let Err(e) = self.push_state_stack(JsonState::ObjectColon) {
                    return (None, Some(e));
                }
                if let JsonToken::String(key) = token {
                    (Some(JsonEvent::ObjectKey(key)), None)
                } else {
                    (None, Some("Object keys must be strings".into()))
                }
            }
            Some(JsonState::ObjectColon) => {
                if let Err(e) = self.push_state_stack(JsonState::ObjectValue) {
                    return (None, Some(e));
                }
                if token == JsonToken::Colon {
                    (None, None)
                } else {
                    let (event, _) = self.apply_new_token(token);
                    (
                        event,
                        Some("Object keys must be followed by a colon ':'".into()),
                    )
                }
            }
            Some(JsonState::ObjectValue) => {
                if let Err(e) = self.push_state_stack(JsonState::ObjectCommaOrEnd) {
                    return (None, Some(e));
                }
                self.apply_new_token_for_value(token)
            }
            Some(JsonState::ObjectCommaOrEnd) => match token {
                JsonToken::Comma => (None, self.push_state_stack(JsonState::ObjectKey).err()),
                JsonToken::ClosingCurlyBracket => (Some(JsonEvent::EndObject), None),
                _ => (
                    None,
                    Some(
                        "Object values must be followed by a comma to add a new value or a curly bracket to end the object"
                            .into(),
                    ),
                ),
            },
            Some(JsonState::ArrayValueOrEnd) => {
                if token == JsonToken::ClosingSquareBracket {
                    return (Some(JsonEvent::EndArray), None);
                }
                if let Err(e) = self.push_state_stack(JsonState::ArrayValue) {
                    return (None, Some(e));
                }
                self.apply_new_token(token)
            }
            Some(JsonState::ArrayValue) => {
                if token == JsonToken::ClosingSquareBracket {
                    return (
                        Some(JsonEvent::EndArray),
                        Some("Trailing commas are not allowed".into()),
                    );
                }
                if let Err(e) = self.push_state_stack(JsonState::ArrayCommaOrEnd) {
                    return (None, Some(e));
                }
                self.apply_new_token_for_value(token)
            }
            Some(JsonState::ArrayCommaOrEnd) => match token {
                JsonToken::Comma => (None, self.push_state_stack(JsonState::ArrayValue).err()),
                JsonToken::ClosingSquareBracket => (Some(JsonEvent::EndArray), None),
                _ => {
                    // We already have an error; still push a state so the resulting stream stays
                    // internally consistent for a caller that keeps reading after the error.
                    let _ = self.push_state_stack(JsonState::ArrayValue);
                    let (event, _) = self.apply_new_token(token);
                    (
                        event,
                        Some(
                            "Array values must be followed by a comma to add a new value or a squared bracket to end the array"
                                .into(),
                        ),
                    )
                }
            },
            None => {
                if self.element_read {
                    if token == JsonToken::Eof {
                        (Some(JsonEvent::Eof), None)
                    } else {
                        (None, Some("The JSON already contains one root element".into()))
                    }
                } else {
                    self.element_read = true;
                    self.apply_new_token_for_value(token)
                }
            }
        }
    }

    /// Interprets `token` as the start of a JSON *value* (as opposed to a structural token like a
    /// comma that only makes sense in specific grammar positions).
    #[inline]
    fn apply_new_token_for_value<'a>(
        &mut self,
        token: JsonToken<'a>,
    ) -> (Option<JsonEvent<'a>>, Option<String>) {
        match token {
            JsonToken::OpeningSquareBracket => (
                Some(JsonEvent::StartArray),
                self.push_state_stack(JsonState::ArrayValueOrEnd).err(),
            ),
            JsonToken::ClosingSquareBracket => (
                None,
                Some("Unexpected closing square bracket, no array to close".into()),
            ),
            JsonToken::OpeningCurlyBracket => (
                Some(JsonEvent::StartObject),
                self.push_state_stack(JsonState::ObjectKeyOrEnd).err(),
            ),
            JsonToken::ClosingCurlyBracket => (
                None,
                Some("Unexpected closing curly bracket, no object to close".into()),
            ),
            JsonToken::Comma => (None, Some("Unexpected comma, no values to separate".into())),
            JsonToken::Colon => (None, Some("Unexpected colon, no key to follow".into())),
            JsonToken::String(string) => (Some(JsonEvent::String(string)), None),
            JsonToken::Number(number) => (Some(JsonEvent::Number(number)), None),
            JsonToken::True => (Some(JsonEvent::Boolean(true)), None),
            JsonToken::False => (Some(JsonEvent::Boolean(false)), None),
            JsonToken::Null => (Some(JsonEvent::Null), None),
            JsonToken::Eof => (
                Some(JsonEvent::Eof),
                Some("Unexpected end of file, a value was expected".into()),
            ),
        }
    }

    #[inline]
    fn push_state_stack(&mut self, state: JsonState) -> Result<(), String> {
        self.check_stack_size()?;
        self.state_stack.push(state);
        Ok(())
    }

    #[inline]
    fn check_stack_size(&self) -> Result<(), String> {
        if self.state_stack.len() > self.max_state_stack_size {
            Err(format!(
                "Max stack size of {} reached on an object or array opening",
                self.max_state_stack_size
            ))
        } else {
            Ok(())
        }
    }
}

impl Default for LowLevelJsonParser {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// The state of an in-progress array or object, tracked in [`LowLevelJsonParser`]'s stack.
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
enum JsonState {
    /// Just inside `{`: the next token is either the first key or `}`.
    ObjectKeyOrEnd,
    /// A key is expected next (after a comma).
    ObjectKey,
    /// A `:` is expected next.
    ObjectColon,
    /// A value is expected next.
    ObjectValue,
    /// A `,` or `}` is expected next.
    ObjectCommaOrEnd,
    /// A value is expected next (inside an array).
    ArrayValue,
    /// Just inside `[`: the next token is either the first value or `]`.
    ArrayValueOrEnd,
    /// A `,` or `]` is expected next.
    ArrayCommaOrEnd,
}

/// The result of a single [`LowLevelJsonParser::parse_next`] call.
#[derive(Debug)]
pub struct LowLevelJsonParserResult<'a> {
    /// How many bytes were consumed from the `input_buffer` passed to
    /// [`parse_next`](LowLevelJsonParser::parse_next); the caller should not pass these bytes
    /// again.
    pub consumed_bytes: usize,
    /// A newly available event, if any. `None` means "not enough input yet" (only possible when
    /// `is_ending` was `false`).
    pub event: Option<Result<JsonEvent<'a>, JsonSyntaxError>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn parse_all(input: &[u8]) -> Vec<Result<String, String>> {
        let mut parser = LowLevelJsonParser::new();
        let mut offset = 0usize;
        let mut out = Vec::new();
        loop {
            let LowLevelJsonParserResult {
                consumed_bytes,
                event,
            } = parser.parse_next(&input[offset..], true);
            offset += consumed_bytes;
            match event {
                None => panic!("low-level parser asked for more data despite is_ending = true"),
                Some(Ok(JsonEvent::Eof)) => {
                    out.push(Ok("Eof".to_string()));
                    break;
                }
                Some(Ok(event)) => out.push(Ok(format!("{event:?}"))),
                Some(Err(e)) => {
                    out.push(Err(e.message().to_string()));
                    break;
                }
            }
        }
        out
    }

    #[test]
    fn flat_object() {
        let events = parse_all(br#"{"a":1,"b":true}"#);
        let expected = [
            format!("{:?}", JsonEvent::StartObject),
            format!("{:?}", JsonEvent::ObjectKey(Cow::Borrowed("a"))),
            format!("{:?}", JsonEvent::Number(Cow::Borrowed("1"))),
            format!("{:?}", JsonEvent::ObjectKey(Cow::Borrowed("b"))),
            format!("{:?}", JsonEvent::Boolean(true)),
            format!("{:?}", JsonEvent::EndObject),
            "Eof".to_string(),
        ];
        let ok: Vec<String> = events.into_iter().map(|r| r.expect("no errors")).collect();
        assert_eq!(ok, expected);
    }

    #[test]
    fn nested_array_and_object() {
        let events = parse_all(br#"[1,{"x":[true,null]},"s"]"#);
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(events.last(), Some(&Ok("Eof".to_string())));
    }

    #[test]
    fn trailing_comma_in_object_errors() {
        let events = parse_all(br#"{"a":1,}"#);
        assert!(matches!(events.last(), Some(Err(_))));
    }

    #[test]
    fn trailing_comma_in_array_errors() {
        let events = parse_all(b"[1,2,]");
        assert!(matches!(events.last(), Some(Err(_))));
    }

    #[test]
    fn two_root_values_errors() {
        let events = parse_all(b"1 2");
        assert!(matches!(events.last(), Some(Err(_))));
    }

    #[test]
    fn object_key_must_be_string() {
        let events = parse_all(b"{1:2}");
        assert!(matches!(events.last(), Some(Err(_))));
    }

    #[test]
    fn missing_colon_errors() {
        let events = parse_all(br#"{"a" 1}"#);
        assert!(matches!(events.last(), Some(Err(_))));
    }

    #[test]
    fn unbalanced_close_errors() {
        assert!(matches!(parse_all(b"]").last(), Some(Err(_))));
        assert!(matches!(parse_all(b"}").last(), Some(Err(_))));
    }

    #[test]
    fn truncated_input_errors_at_eof() {
        let events = parse_all(br#"{"a":"#);
        assert!(matches!(events.last(), Some(Err(_))));
    }

    #[test]
    fn max_stack_size_is_enforced() {
        let mut parser = LowLevelJsonParser::new().with_max_stack_size(1);
        let input = b"[[[1]]]";
        let mut offset = 0usize;
        let mut saw_error = false;
        loop {
            let LowLevelJsonParserResult {
                consumed_bytes,
                event,
            } = parser.parse_next(&input[offset..], true);
            offset += consumed_bytes;
            match event {
                Some(Ok(JsonEvent::Eof)) | None => break,
                Some(Err(_)) => {
                    saw_error = true;
                    break;
                }
                Some(Ok(_)) => {}
            }
        }
        assert!(saw_error, "expected a max-stack-size error");
    }

    #[test]
    fn incremental_byte_at_a_time_feeding_reaches_eof() {
        let input = br#"{"key": [1, 2.5, true, null, "x\ny"]}"#;
        let mut parser = LowLevelJsonParser::new();
        let mut consumed_total = 0usize;
        let mut fed_len = 0usize;
        let mut events = Vec::new();
        loop {
            fed_len = (fed_len + 1).min(input.len());
            let is_ending = fed_len == input.len();
            let LowLevelJsonParserResult {
                consumed_bytes,
                event,
            } = parser.parse_next(&input[consumed_total..fed_len], is_ending);
            consumed_total += consumed_bytes;
            if let Some(event) = event {
                let event = event.expect("no errors expected");
                let is_eof = event == JsonEvent::Eof;
                events.push(event);
                if is_eof {
                    break;
                }
            }
        }
        assert_eq!(events.last(), Some(&JsonEvent::Eof));
        assert!(events.contains(&JsonEvent::StartObject));
        assert!(events.contains(&JsonEvent::StartArray));
    }
}
