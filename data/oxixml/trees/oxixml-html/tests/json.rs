//! A minimal JSON reader, used only by the optional html5lib tokenizer
//! harness.
//!
//! The html5lib tokenizer tests are distributed as JSON, and this workspace's
//! foundation crates may not depend on an external JSON library, so the harness
//! brings its own. It is deliberately just enough to read those files: objects,
//! arrays, strings with `\uXXXX` escapes (including the unpaired surrogates the
//! suite uses on purpose), numbers, and the three literals.

#![allow(dead_code)]

use std::collections::BTreeMap;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, kept as an `f64`.
    Number(f64),
    /// A string. Unpaired surrogates become U+FFFD, which is what a Rust
    /// `String` can hold.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, with keys in sorted order.
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// The string behind a [`Json::String`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// The elements of a [`Json::Array`].
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Looks a key up in a [`Json::Object`].
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Object(map) => map.get(key),
            _ => None,
        }
    }

    /// The entries of a [`Json::Object`].
    pub fn entries(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }
}

/// Parses a whole JSON document.
///
/// # Errors
///
/// Returns a message describing the first syntax error.
pub fn parse(input: &str) -> Result<Json, String> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        position: 0,
    };
    parser.skip_whitespace();
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.position != parser.bytes.len() {
        return Err(format!("trailing input at byte {}", parser.position));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Parser<'_> {
    fn skip_whitespace(&mut self) {
        while matches!(
            self.bytes.get(self.position),
            Some(b' ' | b'\t' | b'\n' | b'\r')
        ) {
            self.position += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.peek() == Some(byte) {
            self.position += 1;
            Ok(())
        } else {
            Err(format!(
                "expected {:?} at byte {}",
                byte as char, self.position
            ))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(_) => self.number(),
            None => Err("unexpected end of input".to_owned()),
        }
    }

    fn literal(&mut self, text: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.position..].starts_with(text.as_bytes()) {
            self.position += text.len();
            Ok(value)
        } else {
            Err(format!("expected {text} at byte {}", self.position))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.position;
        while matches!(
            self.peek(),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.position += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|error| error.to_string())?;
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|error| error.to_string())
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.position += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_whitespace();
            items.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(format!("expected , or ] at byte {}", self.position)),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.position += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value()?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    return Ok(Json::Object(map));
                }
                _ => return Err(format!("expected , or }} at byte {}", self.position)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| "unterminated string".to_owned())?;
            match byte {
                b'"' => {
                    self.position += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.position += 1;
                    let escape = self
                        .peek()
                        .ok_or_else(|| "unterminated escape".to_owned())?;
                    self.position += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        other => return Err(format!("bad escape \\{}", other as char)),
                    }
                }
                _ => {
                    // Copy one whole UTF-8 sequence.
                    let start = self.position;
                    self.position += 1;
                    while matches!(self.peek(), Some(b) if (b & 0xc0) == 0x80) {
                        self.position += 1;
                    }
                    out.push_str(
                        std::str::from_utf8(&self.bytes[start..self.position])
                            .map_err(|error| error.to_string())?,
                    );
                }
            }
        }
    }

    /// Reads a `\uXXXX` escape, joining a surrogate pair when one follows and
    /// substituting U+FFFD for an unpaired surrogate.
    fn unicode_escape(&mut self) -> Result<char, String> {
        let first = self.hex4()?;
        if (0xd800..0xdc00).contains(&first) {
            if self.bytes[self.position..].starts_with(b"\\u") {
                let save = self.position;
                self.position += 2;
                let second = self.hex4()?;
                if (0xdc00..0xe000).contains(&second) {
                    let combined = 0x1_0000 + ((first - 0xd800) << 10) + (second - 0xdc00);
                    return char::from_u32(combined).ok_or_else(|| "bad surrogate pair".to_owned());
                }
                self.position = save;
            }
            return Ok('\u{fffd}');
        }
        if (0xdc00..0xe000).contains(&first) {
            return Ok('\u{fffd}');
        }
        char::from_u32(first).ok_or_else(|| "bad code point".to_owned())
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let end = self.position + 4;
        if end > self.bytes.len() {
            return Err("truncated \\u escape".to_owned());
        }
        let text = std::str::from_utf8(&self.bytes[self.position..end])
            .map_err(|error| error.to_string())?;
        let value = u32::from_str_radix(text, 16).map_err(|error| error.to_string())?;
        self.position = end;
        Ok(value)
    }
}

#[test]
fn reads_the_shapes_the_suite_uses() -> Result<(), String> {
    let value = parse(r#"{"tests":[{"input":"aA\ud800","output":[["Character","x"]]}]}"#)?;
    let tests = value.get("tests").and_then(Json::as_array).expect("tests");
    assert_eq!(tests.len(), 1);
    assert_eq!(
        tests[0].get("input").and_then(Json::as_str),
        Some("aA\u{fffd}")
    );
    assert_eq!(
        parse("[1, 2.5, -3e2]")?.as_array().map(<[Json]>::len),
        Some(3)
    );
    assert_eq!(parse("  null ")?, Json::Null);
    Ok(())
}
