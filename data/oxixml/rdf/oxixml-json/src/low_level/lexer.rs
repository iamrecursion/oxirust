//! Byte-level JSON tokenizer: turns raw bytes into [`JsonToken`]s with exact
//! [`TextPosition`](oxixml_support::pos::TextPosition) tracking.

use crate::error::JsonSyntaxError;
use oxixml_support::pos::{LineColTracker, TextPosition};
use std::borrow::Cow;
use std::ops::Range;
use std::str;

/// A lexical token of the JSON grammar (RFC 8259 §2), before it has been assembled into a
/// [`crate::JsonEvent`] by the state machine in [`super::LowLevelJsonParser`].
#[derive(Eq, PartialEq, Clone, Debug)]
pub(crate) enum JsonToken<'a> {
    /// `[`
    OpeningSquareBracket,
    /// `]`
    ClosingSquareBracket,
    /// `{`
    OpeningCurlyBracket,
    /// `}`
    ClosingCurlyBracket,
    /// `,`
    Comma,
    /// `:`
    Colon,
    /// A decoded string literal, escapes already resolved.
    String(Cow<'a, str>),
    /// A raw (unparsed) number literal.
    Number(Cow<'a, str>),
    /// The literal `true`.
    True,
    /// The literal `false`.
    False,
    /// The literal `null`.
    Null,
    /// End of input: no more tokens follow.
    Eof,
}

/// Tokenizes a byte stream into [`JsonToken`]s, tracking an exact
/// [`TextPosition`](oxixml_support::pos::TextPosition) as it goes.
///
/// The lexer is fed successive, possibly-overlapping-in-content-but-not-in-position slices: each
/// call to [`JsonLexer::read_next_token`] receives every byte that has not yet been *committed*
/// (i.e. every byte starting exactly at [`JsonLexer::position`]). When a token cannot be
/// completed because the slice runs out before its end is found, the call returns `None` and
/// commits nothing for that (incomplete) token — whitespace already skipped in the same call is
/// still committed, so a retry with a longer slice resumes exactly where it left off.
pub(crate) struct JsonLexer {
    tracker: LineColTracker,
    /// The position at which the token currently being tokenized starts (i.e. the position right
    /// after the most recently skipped run of whitespace). Used to build error ranges for
    /// state-machine-level errors that span "the token we just read".
    token_start: TextPosition,
    /// `true` until the very first token has been considered, used to strip a leading UTF-8 BOM.
    is_start: bool,
}

impl JsonLexer {
    /// Creates a lexer positioned at the start of a document.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self {
            tracker: LineColTracker::new(),
            token_start: TextPosition::start(),
            is_start: true,
        }
    }

    /// The current position: the position of the next byte that has not yet been committed.
    #[inline]
    pub(crate) const fn position(&self) -> TextPosition {
        self.tracker.position()
    }

    /// Builds a [`JsonSyntaxError`] spanning from the start of the most recently returned token
    /// to the current position. Used for state-machine-level errors (e.g. "expected a comma"),
    /// which are about the shape of the token stream rather than a specific byte range.
    #[inline]
    pub(crate) fn error_at_last_token(&self, message: impl Into<String>) -> JsonSyntaxError {
        JsonSyntaxError::new(self.token_start..self.position(), message)
    }

    /// Builds a [`JsonSyntaxError`] for a byte range measured from the current position
    /// (`0` == [`JsonLexer::position`]) within `input_buffer`.
    #[inline]
    fn error_at(
        &self,
        input_buffer: &[u8],
        range: Range<usize>,
        message: impl Into<String>,
    ) -> JsonSyntaxError {
        let start = self.position_at(input_buffer, range.start);
        let end = self.position_at(input_buffer, range.end);
        JsonSyntaxError::new(start..end, message)
    }

    /// The [`TextPosition`] of the byte at local offset `byte_offset` within `input_buffer`,
    /// where offset `0` corresponds to [`JsonLexer::position`]. Does not mutate the lexer.
    #[inline]
    fn position_at(&self, input_buffer: &[u8], byte_offset: usize) -> TextPosition {
        let mut probe = self.tracker.clone();
        let end = byte_offset.min(input_buffer.len());
        probe.advance(&input_buffer[..end]);
        probe.position()
    }

    /// Reads the next token from `input_buffer`, which must contain every byte starting at
    /// [`JsonLexer::position`] that is currently available.
    ///
    /// Returns `None` when `input_buffer` does not contain enough bytes to determine the next
    /// token and `is_ending` is `false` (more input may still arrive). `is_ending` must be `true`
    /// once `input_buffer` holds all remaining input.
    pub(crate) fn read_next_token<'a>(
        &mut self,
        mut input_buffer: &'a [u8],
        is_ending: bool,
    ) -> Option<Result<JsonToken<'a>, JsonSyntaxError>> {
        if self.is_start {
            if input_buffer.len() < 3 && !is_ending {
                return None;
            }
            self.is_start = false;
            if input_buffer.starts_with(&[0xEF, 0xBB, 0xBF]) {
                self.tracker.advance(&input_buffer[..3]);
                input_buffer = &input_buffer[3..];
            }
        }

        // Skip whitespace (space, tab, LF, CR, CRLF), committing every confirmed byte.
        let mut i = 0usize;
        let mut need_more = false;
        loop {
            match input_buffer.get(i) {
                Some(b' ' | b'\t' | b'\n') => i += 1,
                Some(b'\r') => {
                    i += 1;
                    match input_buffer.get(i) {
                        Some(b'\n') => i += 1,
                        Some(_) => {}
                        None if !is_ending => {
                            // Need one more byte to know whether this is a lone CR or a CRLF;
                            // do not commit the pending CR yet.
                            i -= 1;
                            need_more = true;
                            break;
                        }
                        None => {}
                    }
                }
                _ => break,
            }
        }
        self.tracker.advance(&input_buffer[..i]);
        if need_more {
            return None;
        }
        input_buffer = &input_buffer[i..];
        self.token_start = self.tracker.position();

        if is_ending && input_buffer.is_empty() {
            return Some(Ok(JsonToken::Eof));
        }

        match *input_buffer.first()? {
            b'{' => {
                self.tracker.advance(&input_buffer[..1]);
                Some(Ok(JsonToken::OpeningCurlyBracket))
            }
            b'}' => {
                self.tracker.advance(&input_buffer[..1]);
                Some(Ok(JsonToken::ClosingCurlyBracket))
            }
            b'[' => {
                self.tracker.advance(&input_buffer[..1]);
                Some(Ok(JsonToken::OpeningSquareBracket))
            }
            b']' => {
                self.tracker.advance(&input_buffer[..1]);
                Some(Ok(JsonToken::ClosingSquareBracket))
            }
            b',' => {
                self.tracker.advance(&input_buffer[..1]);
                Some(Ok(JsonToken::Comma))
            }
            b':' => {
                self.tracker.advance(&input_buffer[..1]);
                Some(Ok(JsonToken::Colon))
            }
            b'"' => self.read_string(input_buffer),
            b't' => self.read_constant(input_buffer, is_ending, "true", JsonToken::True),
            b'f' => self.read_constant(input_buffer, is_ending, "false", JsonToken::False),
            b'n' => self.read_constant(input_buffer, is_ending, "null", JsonToken::Null),
            b'-' | b'0'..=b'9' => self.read_number(input_buffer, is_ending),
            c => {
                let message = if c.is_ascii() {
                    format!("Unexpected char: '{}'", char::from(c))
                } else {
                    format!("Unexpected byte: \\x{c:X}")
                };
                let error = self.error_at(input_buffer, 0..1, message);
                self.tracker.advance(&input_buffer[..1]);
                Some(Err(error))
            }
        }
    }

    /// Reads a `"`-delimited string, starting at `input_buffer[0] == b'"'`.
    #[inline]
    fn read_string<'a>(
        &mut self,
        input_buffer: &'a [u8],
    ) -> Option<Result<JsonToken<'a>, JsonSyntaxError>> {
        let mut error: Option<JsonSyntaxError> = None;
        // Accumulates decoded content once an escape forces us off the zero-copy fast path:
        // (decoded-so-far, "raw bytes up to this offset have been folded in").
        let mut string: Option<(String, usize)> = None;
        let mut next = 1usize;
        loop {
            match *input_buffer.get(next)? {
                b'"' => {
                    let result = if let Some(err) = error.clone() {
                        Err(err)
                    } else if let Some((mut owned, read_until)) = string {
                        if read_until < next {
                            let (chunk, decode_error) =
                                self.decode_utf8(input_buffer, read_until..next);
                            error = error.or(decode_error);
                            owned.push_str(&chunk);
                        }
                        if let Some(err) = error.clone() {
                            Err(err)
                        } else {
                            Ok(JsonToken::String(Cow::Owned(owned)))
                        }
                    } else {
                        let (content, decode_error) = self.decode_utf8(input_buffer, 1..next);
                        if let Some(err) = decode_error {
                            Err(err)
                        } else {
                            Ok(JsonToken::String(content))
                        }
                    };
                    self.tracker.advance(&input_buffer[..=next]);
                    return Some(result);
                }
                b'\\' => {
                    if string.is_none() {
                        string = Some((String::new(), 1));
                    }
                    // Safe: we just ensured `string` is `Some`.
                    let (owned, read_until) = string
                        .as_mut()
                        .expect("string was just initialized to Some above");
                    if *read_until < next {
                        let (chunk, decode_error) =
                            self.decode_utf8(input_buffer, *read_until..next);
                        error = error.or(decode_error);
                        owned.push_str(&chunk);
                    }
                    next += 1;
                    match *input_buffer.get(next)? {
                        b'"' => {
                            owned.push('"');
                            next += 1;
                        }
                        b'\\' => {
                            owned.push('\\');
                            next += 1;
                        }
                        b'/' => {
                            owned.push('/');
                            next += 1;
                        }
                        b'b' => {
                            owned.push('\u{8}');
                            next += 1;
                        }
                        b'f' => {
                            owned.push('\u{C}');
                            next += 1;
                        }
                        b'n' => {
                            owned.push('\n');
                            next += 1;
                        }
                        b'r' => {
                            owned.push('\r');
                            next += 1;
                        }
                        b't' => {
                            owned.push('\t');
                            next += 1;
                        }
                        b'u' => {
                            next += 1;
                            let code_point = self.read_unicode_escape(
                                input_buffer,
                                &mut next,
                                &mut error,
                                owned,
                            )?;
                            if let Some(c) = code_point {
                                owned.push(c);
                            }
                        }
                        c => {
                            next += 1;
                            if error.is_none() {
                                error = Some(self.error_at(
                                    input_buffer,
                                    (next - 2)..next,
                                    format!("'\\{}' is not a valid escape sequence", char::from(c)),
                                ));
                            }
                            owned.push(char::REPLACEMENT_CHARACTER);
                        }
                    }
                    *read_until = next;
                }
                c @ 0..=0x1F => {
                    if error.is_none() {
                        error = Some(self.error_at(
                            input_buffer,
                            next..next + 1,
                            format!("'{}' is not allowed in JSON strings", char::from(c)),
                        ));
                    }
                    next += 1;
                }
                _ => next += 1,
            }
        }
    }

    /// Reads a `\uXXXX` escape (and, for a high surrogate, the following `\uXXXX` low surrogate),
    /// starting right after the `u`. Returns `None` if more input is needed.
    ///
    /// On success, returns `Some(Some(char))` for a decodable code point, or `Some(None)` if the
    /// escape was malformed in a way that still consumed a fixed number of bytes (in which case
    /// `error` has been set and the caller should push a replacement character itself, matching
    /// the existing behavior of the `\u` branch below for lone low surrogates).
    #[inline]
    fn read_unicode_escape(
        &self,
        input_buffer: &[u8],
        next: &mut usize,
        error: &mut Option<JsonSyntaxError>,
        owned: &mut String,
    ) -> Option<Option<char>> {
        let val = input_buffer.get(*next..*next + 4)?;
        *next += 4;
        let code_point = match read_hex4(val) {
            Ok(cp) => cp,
            Err(e) => {
                if error.is_none() {
                    *error = Some(self.error_at(input_buffer, (*next - 4)..*next, e));
                }
                owned.push(char::REPLACEMENT_CHARACTER);
                return Some(None);
            }
        };
        if let Some(c) = char::from_u32(code_point) {
            return Some(Some(c));
        }
        // Not a scalar value on its own: must be a surrogate half.
        let high_surrogate = code_point;
        if !(0xD800..=0xDBFF).contains(&high_surrogate) {
            if error.is_none() {
                *error = Some(self.error_at(
                    input_buffer,
                    (*next - 6)..*next,
                    format!("\\u{high_surrogate:X} is not a valid high surrogate"),
                ));
            }
            owned.push(char::REPLACEMENT_CHARACTER);
            return Some(None);
        }
        let low_escape = input_buffer.get(*next..*next + 6)?;
        *next += 6;
        if !low_escape.starts_with(b"\\u") {
            if error.is_none() {
                *error = Some(self.error_at(
                    input_buffer,
                    (*next - 6)..*next,
                    format!(
                        "\\u{high_surrogate:X} is a high surrogate and should be followed by a low surrogate \\uXXXX"
                    ),
                ));
            }
            owned.push(char::REPLACEMENT_CHARACTER);
            return Some(None);
        }
        let low_surrogate = match read_hex4(&low_escape[2..]) {
            Ok(cp) => cp,
            Err(e) => {
                if error.is_none() {
                    *error = Some(self.error_at(input_buffer, (*next - 6)..*next, e));
                }
                owned.push(char::REPLACEMENT_CHARACTER);
                return Some(None);
            }
        };
        if !(0xDC00..=0xDFFF).contains(&low_surrogate) {
            if error.is_none() {
                *error = Some(self.error_at(
                    input_buffer,
                    (*next - 6)..*next,
                    format!("\\u{low_surrogate:X} is not a valid low surrogate"),
                ));
            }
            owned.push(char::REPLACEMENT_CHARACTER);
            return Some(None);
        }
        let code_point = 0x10000 + ((high_surrogate & 0x03FF) << 10) + (low_surrogate & 0x03FF);
        if let Some(c) = char::from_u32(code_point) {
            Some(Some(c))
        } else {
            if error.is_none() {
                *error = Some(self.error_at(
                    input_buffer,
                    (*next - 12)..*next,
                    format!(
                        "\\u{high_surrogate:X}\\u{low_surrogate:X} is an invalid surrogate pair"
                    ),
                ));
            }
            owned.push(char::REPLACEMENT_CHARACTER);
            Some(None)
        }
    }

    /// Decodes `input_buffer[range]` as UTF-8, reporting an error (and lossily decoding) on
    /// invalid bytes rather than failing outright, so the lexer can keep scanning for the
    /// string's end and report an accurate token span.
    #[inline]
    fn decode_utf8<'a>(
        &self,
        input_buffer: &'a [u8],
        range: Range<usize>,
    ) -> (Cow<'a, str>, Option<JsonSyntaxError>) {
        let slice = &input_buffer[range.clone()];
        match str::from_utf8(slice) {
            Ok(s) => (Cow::Borrowed(s), None),
            Err(e) => {
                let bad = range.start + e.valid_up_to();
                (
                    Cow::Owned(String::from_utf8_lossy(slice).into_owned()),
                    Some(self.error_at(input_buffer, bad..bad + 1, format!("Invalid UTF-8: {e}"))),
                )
            }
        }
    }

    /// Reads one of the keyword literals `true`, `false` or `null`.
    #[inline]
    fn read_constant<'a>(
        &mut self,
        input_buffer: &'a [u8],
        is_ending: bool,
        expected: &str,
        value: JsonToken<'static>,
    ) -> Option<Result<JsonToken<'a>, JsonSyntaxError>> {
        if input_buffer.get(..expected.len())? == expected.as_bytes() {
            self.tracker.advance(&input_buffer[..expected.len()]);
            return Some(Ok(value));
        }
        let ascii_chars = input_buffer
            .iter()
            .take_while(|c| c.is_ascii_alphabetic())
            .count();
        if ascii_chars == input_buffer.len() && !is_ending {
            return None; // A longer keyword-shaped token might still match once more data arrives.
        }
        let read = ascii_chars.max(1);
        let error = self.error_at(input_buffer, 0..read, format!("{expected} expected"));
        self.tracker.advance(&input_buffer[..read]);
        Some(Err(error))
    }

    /// Reads a JSON number literal (RFC 8259 §6).
    #[inline]
    fn read_number<'a>(
        &mut self,
        input_buffer: &'a [u8],
        is_ending: bool,
    ) -> Option<Result<JsonToken<'a>, JsonSyntaxError>> {
        let mut next = 0usize;
        if *input_buffer.get(next)? == b'-' {
            next += 1;
        }
        match *input_buffer.get(next)? {
            b'0' => next += 1,
            b'1'..=b'9' => {
                next += 1;
                next += read_digits(&input_buffer[next..], is_ending)?;
            }
            c => {
                next += 1;
                let error = self.error_at(
                    input_buffer,
                    (next - 1)..next,
                    format!("A number is not allowed to start with '{}'", char::from(c)),
                );
                self.tracker.advance(&input_buffer[..next]);
                return Some(Err(error));
            }
        }

        // Fractional part.
        if peek_after(input_buffer, next, is_ending)? == Some(b'.') {
            next += 1;
            let c = *input_buffer.get(next)?;
            next += 1;
            if !c.is_ascii_digit() {
                let error = self.error_at(
                    input_buffer,
                    (next - 1)..next,
                    format!(
                        "A number fractional part must start with a digit and not '{}'",
                        char::from(c)
                    ),
                );
                self.tracker.advance(&input_buffer[..next]);
                return Some(Err(error));
            }
            next += read_digits(&input_buffer[next..], is_ending)?;
        }

        // Exponent part.
        if matches!(
            peek_after(input_buffer, next, is_ending)?,
            Some(b'e' | b'E')
        ) {
            next += 1;
            match *input_buffer.get(next)? {
                b'-' | b'+' => {
                    next += 1;
                    let c = *input_buffer.get(next)?;
                    next += 1;
                    if !c.is_ascii_digit() {
                        let error = self.error_at(
                            input_buffer,
                            (next - 1)..next,
                            format!(
                                "A number exponential part must contain at least a digit, '{}' found",
                                char::from(c)
                            ),
                        );
                        self.tracker.advance(&input_buffer[..next]);
                        return Some(Err(error));
                    }
                }
                b'0'..=b'9' => next += 1,
                c => {
                    next += 1;
                    let error = self.error_at(
                        input_buffer,
                        (next - 1)..next,
                        format!(
                            "A number exponential part must start with +, - or a digit, '{}' found",
                            char::from(c)
                        ),
                    );
                    self.tracker.advance(&input_buffer[..next]);
                    return Some(Err(error));
                }
            }
            next += read_digits(&input_buffer[next..], is_ending)?;
        }

        self.tracker.advance(&input_buffer[..next]);
        let text = str::from_utf8(&input_buffer[..next])
            .expect("JSON number literals only ever contain ASCII bytes");
        Some(Ok(JsonToken::Number(Cow::Borrowed(text))))
    }
}

impl Default for JsonLexer {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Peeks at the byte right after local offset `next`, treating "buffer exhausted at the document
/// end" as an explicit `Some(None)` (no such byte, no more will ever come) and "buffer exhausted
/// but more input may arrive" as `None` (ask the caller to wait).
#[inline]
fn peek_after(input_buffer: &[u8], next: usize, is_ending: bool) -> Option<Option<u8>> {
    match input_buffer.get(next) {
        Some(c) => Some(Some(*c)),
        None if is_ending => Some(None),
        None => None,
    }
}

/// Counts a run of ASCII digits at the start of `input_buffer`. Returns `None` if the run reaches
/// the end of `input_buffer` and more input might still extend it.
#[inline]
fn read_digits(input_buffer: &[u8], is_ending: bool) -> Option<usize> {
    let count = input_buffer
        .iter()
        .take_while(|c| c.is_ascii_digit())
        .count();
    if count == input_buffer.len() && !is_ending {
        return None;
    }
    Some(count)
}

/// Parses exactly 4 hex digits (a `\uXXXX` payload) into a code point value.
#[inline]
fn read_hex4(input: &[u8]) -> Result<u32, String> {
    let mut value = 0u32;
    for &c in input {
        let digit = match c {
            b'0'..=b'9' => u32::from(c - b'0'),
            b'a'..=b'f' => u32::from(c - b'a') + 10,
            b'A'..=b'F' => u32::from(c - b'A') + 10,
            _ => {
                return Err(format!(
                    "Unexpected character in a unicode escape: '{}'",
                    char::from(c)
                ));
            }
        };
        value = value * 16 + digit;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(input: &[u8]) -> Vec<JsonToken<'static>> {
        let mut lexer = JsonLexer::new();
        let mut out = Vec::new();
        let mut offset = 0usize;
        loop {
            match lexer.read_next_token(&input[offset..], true) {
                Some(Ok(JsonToken::Eof)) => {
                    out.push(JsonToken::Eof);
                    break;
                }
                Some(Ok(tok)) => {
                    offset = usize::try_from(lexer.position().offset).expect("fits in usize");
                    out.push(owned_token(tok));
                }
                Some(Err(e)) => panic!("unexpected lex error: {e}"),
                None => panic!("unexpected request for more data"),
            }
        }
        out
    }

    fn owned_token(t: JsonToken<'_>) -> JsonToken<'static> {
        match t {
            JsonToken::String(s) => JsonToken::String(Cow::Owned(s.into_owned())),
            JsonToken::Number(n) => JsonToken::Number(Cow::Owned(n.into_owned())),
            JsonToken::OpeningSquareBracket => JsonToken::OpeningSquareBracket,
            JsonToken::ClosingSquareBracket => JsonToken::ClosingSquareBracket,
            JsonToken::OpeningCurlyBracket => JsonToken::OpeningCurlyBracket,
            JsonToken::ClosingCurlyBracket => JsonToken::ClosingCurlyBracket,
            JsonToken::Comma => JsonToken::Comma,
            JsonToken::Colon => JsonToken::Colon,
            JsonToken::True => JsonToken::True,
            JsonToken::False => JsonToken::False,
            JsonToken::Null => JsonToken::Null,
            JsonToken::Eof => JsonToken::Eof,
        }
    }

    #[test]
    fn punctuation_and_keywords() {
        assert_eq!(
            tokens(b"[{,:}] true false null"),
            vec![
                JsonToken::OpeningSquareBracket,
                JsonToken::OpeningCurlyBracket,
                JsonToken::Comma,
                JsonToken::Colon,
                JsonToken::ClosingCurlyBracket,
                JsonToken::ClosingSquareBracket,
                JsonToken::True,
                JsonToken::False,
                JsonToken::Null,
                JsonToken::Eof,
            ]
        );
    }

    #[test]
    fn plain_string() {
        assert_eq!(
            tokens(br#""hello world""#),
            vec![
                JsonToken::String(Cow::Owned("hello world".to_string())),
                JsonToken::Eof
            ]
        );
    }

    #[test]
    fn string_escapes() {
        assert_eq!(
            tokens(br#""a\n\t\"\\\/b""#),
            vec![
                JsonToken::String(Cow::Owned("a\n\t\"\\/b".to_string())),
                JsonToken::Eof
            ]
        );
    }

    #[test]
    fn string_surrogate_pair_escape() {
        // U+1F600 (grinning face) written as a UTF-16 surrogate pair escape:
        // `"😀"`.
        assert_eq!(
            tokens(b"\"\\uD83D\\uDE00\""),
            vec![
                JsonToken::String(Cow::Owned("\u{1F600}".to_string())),
                JsonToken::Eof
            ]
        );
    }

    #[test]
    fn raw_utf8_string_content() {
        // A literal (non-escaped) multi-byte character in the source bytes.
        assert_eq!(
            tokens("\"caf\u{e9}\"".as_bytes()),
            vec![
                JsonToken::String(Cow::Owned("caf\u{e9}".to_string())),
                JsonToken::Eof
            ]
        );
    }

    #[test]
    fn numbers() {
        for (input, expected) in [
            ("0", "0"),
            ("-0", "-0"),
            ("123", "123"),
            ("-123.456", "-123.456"),
            ("1e10", "1e10"),
            ("1E+10", "1E+10"),
            ("1.5e-10", "1.5e-10"),
        ] {
            assert_eq!(
                tokens(input.as_bytes()),
                vec![
                    JsonToken::Number(Cow::Owned(expected.to_string())),
                    JsonToken::Eof
                ],
                "input {input:?}"
            );
        }
    }

    #[test]
    fn leading_zero_number_is_two_tokens() {
        // `01` is `0` immediately followed by `1`: not a valid single number literal.
        assert_eq!(
            tokens(b"01"),
            vec![
                JsonToken::Number(Cow::Owned("0".to_string())),
                JsonToken::Number(Cow::Owned("1".to_string())),
                JsonToken::Eof,
            ]
        );
    }

    #[test]
    fn bom_is_stripped_once() {
        assert_eq!(
            tokens(b"\xEF\xBB\xBF{}"),
            vec![
                JsonToken::OpeningCurlyBracket,
                JsonToken::ClosingCurlyBracket,
                JsonToken::Eof
            ]
        );
    }

    #[test]
    fn crlf_and_cr_and_lf_all_skip() {
        assert_eq!(
            tokens(b"{\r\n\t,\r \n}"),
            vec![
                JsonToken::OpeningCurlyBracket,
                JsonToken::Comma,
                JsonToken::ClosingCurlyBracket,
                JsonToken::Eof
            ]
        );
    }

    #[test]
    fn incremental_feeding_matches_whole_buffer() {
        let input = br#"{"key": [1, 2.5, true, null, "x\ny"]}"#;
        let mut whole = JsonLexer::new();
        let mut whole_tokens = Vec::new();
        let mut offset = 0usize;
        loop {
            match whole.read_next_token(&input[offset..], true) {
                Some(Ok(JsonToken::Eof)) => break,
                Some(Ok(tok)) => {
                    offset = usize::try_from(whole.position().offset).expect("fits");
                    whole_tokens.push(owned_token(tok));
                }
                _ => panic!("unexpected"),
            }
        }

        // Feed one byte at a time; only the last call sets is_ending. `consumed` must be
        // resynchronized to the lexer's own position after *every* call, not just successful
        // ones: whitespace can be committed even when a call returns `None` (waiting for the
        // token that follows it to become available).
        let mut fed = JsonLexer::new();
        let mut fed_tokens = Vec::new();
        let mut consumed = 0usize;
        let mut fed_len = 0usize;
        while fed_tokens.len() < whole_tokens.len() {
            fed_len = (fed_len + 1).max(consumed).min(input.len());
            let is_ending = fed_len >= input.len();
            let slice = &input[consumed..fed_len];
            if let Some(result) = fed.read_next_token(slice, is_ending) {
                let tok = result.expect("no lex errors expected");
                fed_tokens.push(owned_token(tok));
            }
            consumed = usize::try_from(fed.position().offset).expect("fits");
        }
        assert_eq!(whole_tokens, fed_tokens);
    }

    #[test]
    fn control_character_in_string_is_rejected() {
        let mut lexer = JsonLexer::new();
        let result = lexer.read_next_token(b"\"a\x01b\"", true);
        assert!(matches!(result, Some(Err(_))));
    }

    #[test]
    fn lone_high_surrogate_is_rejected() {
        // `\uD800` (a high surrogate) followed by `A` ('A'), which is not in the low
        // surrogate range: this is fully self-contained, so the lexer can reject it outright
        // without waiting for more input.
        let mut lexer = JsonLexer::new();
        let result = lexer.read_next_token(b"\"\\uD800\\u0041\"", true);
        assert!(matches!(result, Some(Err(_))));
    }

    #[test]
    fn error_position_accounts_for_prior_multibyte_content() {
        let mut lexer = JsonLexer::new();
        let input = "\"\u{e9}\" #".as_bytes(); // `"é" #`
        let first = lexer.read_next_token(input, true);
        assert!(matches!(first, Some(Ok(JsonToken::String(_)))));

        let consumed = usize::try_from(lexer.position().offset).expect("fits in usize");
        let second = lexer.read_next_token(&input[consumed..], true);
        let err = match second {
            Some(Err(e)) => e,
            other => panic!("expected error, got {other:?}"),
        };
        let loc = err.location();
        // 0-based columns: `"é"` occupies columns 0..=2 (quote, é, quote); the space is column 3;
        // `#` is column 4 (rendered as "line 1 column 5").
        assert_eq!(loc.start.line, 0);
        assert_eq!(loc.start.column, 4);
        assert!(err.to_string().contains("line 1 column 5"), "{err}");
        assert_eq!(loc.start.offset, 5); // 4 bytes for `"é"` + 1 byte for the space
    }
}
