//! A conforming JSON scanner and parser for the JSON functions of
//! [F&O 3.1 §17.5](https://www.w3.org/TR/xpath-functions-31/#json-functions).
//!
//! The grammar is [RFC 7159](https://www.rfc-editor.org/rfc/rfc7159): any value may appear at the
//! top level, numbers are `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?`, strings admit exactly
//! the eight backslash escapes plus `\uXXXX`, and an unescaped control character is an error.
//!
//! The scanner is deliberately *not* a general-purpose JSON reader: it exists to produce the
//! [`JsonValue`] tree that `fn:parse-json`, `fn:json-doc` and `fn:json-to-xml` share, with the
//! string handling those functions' `escape`, `fallback` and `duplicates` options prescribe.
//!
//! Arrays and objects are scanned by recursive descent, so how deeply they may nest is bounded —
//! see [`parse`] and [`JsonOptions::max_depth`](crate::json::JsonOptions::max_depth).

use crate::error::{FnError, FnErrorCode, FnResult};

/// The `duplicates` option of the JSON parsing functions,
/// [F&O 3.1 §17.5.1](https://www.w3.org/TR/xpath-functions-31/#func-parse-json).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Duplicates {
    /// `reject` — a repeated key raises `FOJS0003`.
    Reject,
    /// `use-first` — the first entry with a given key wins.
    UseFirst,
    /// `use-last` — the last entry with a given key wins.
    UseLast,
    /// `retain` — every entry is kept, which only `fn:json-to-xml` can represent.
    Retain,
}

impl Duplicates {
    /// Reads the option's lexical value.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fojs0005`] for a value outside the four the specification defines.
    pub fn parse(value: &str) -> FnResult<Self> {
        match value {
            "reject" => Ok(Self::Reject),
            "use-first" => Ok(Self::UseFirst),
            "use-last" => Ok(Self::UseLast),
            "retain" => Ok(Self::Retain),
            other => Err(FnError::new(
                FnErrorCode::Fojs0005,
                format!("`{other}` is not a valid value for the `duplicates` option"),
            )),
        }
    }
}

/// The `fallback` option: what an unescapable character is replaced by.
///
/// It is a *function* in the specification, and the data model has no item for one, so an
/// evaluator that owns function items passes its own closure through this borrow.
pub type Fallback<'a> = &'a dyn Fn(&str) -> FnResult<String>;

/// How a JSON string is turned into an XDM string: the `escape` and `fallback` options.
#[derive(Default)]
pub struct StringPolicy<'a> {
    /// The `escape` option: keep characters that JSON escapes in their escaped form.
    pub escape: bool,
    /// The `fallback` option: what an unescapable character is replaced by when `escape` is false.
    pub fallback: Option<Fallback<'a>>,
}

impl std::fmt::Debug for StringPolicy<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringPolicy")
            .field("escape", &self.escape)
            .field("fallback", &self.fallback.is_some())
            .finish()
    }
}

/// The tree a JSON text parses to.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, carrying both its double value and the lexical form it was written in.
    Number(f64, String),
    /// A string, already processed according to the [`StringPolicy`] in force.
    String(String),
    /// An array.
    Array(Vec<JsonValue>),
    /// An object, whose entries are in document order after the `duplicates` policy was applied.
    Object(Vec<(String, JsonValue)>),
}

/// Parses `input` as a complete JSON text, refusing to nest more than `max_depth` levels deep.
///
/// An array or object is scanned by re-entering the value production, so every level of nesting
/// costs a native stack frame. [RFC 8259 §9](https://www.rfc-editor.org/rfc/rfc8259#section-9)
/// expressly allows a parser to set a limit on nesting depth; without one, a few tens of
/// kilobytes of `[[[[…]]]]` overflows the stack, which is a process abort (`SIGABRT`) rather
/// than a catchable Rust panic — so a host that parses externally supplied JSON text could not
/// defend against it. Past the ceiling the text is rejected with `FOJS0001` instead.
///
/// # Errors
///
/// [`FnErrorCode::Fojs0001`] for any syntax error, and for a text nesting deeper than
/// `max_depth`; [`FnErrorCode::Fojs0003`] for a duplicate key under `duplicates=reject`.
pub fn parse(
    input: &str,
    duplicates: Duplicates,
    policy: &StringPolicy<'_>,
    max_depth: usize,
) -> FnResult<JsonValue> {
    let mut parser = Parser {
        chars: input.chars().collect(),
        position: 0,
        duplicates,
        policy,
        max_depth,
    };
    parser.skip_whitespace();
    let value = parser.value(0)?;
    parser.skip_whitespace();
    if parser.position < parser.chars.len() {
        return Err(syntax("unexpected content after the top-level JSON value"));
    }
    Ok(value)
}

fn syntax(what: impl Into<String>) -> FnError {
    FnError::new(FnErrorCode::Fojs0001, what.into())
}

struct Parser<'a> {
    chars: Vec<char>,
    position: usize,
    duplicates: Duplicates,
    policy: &'a StringPolicy<'a>,
    /// How many arrays and objects may be open at once — see [`parse`].
    max_depth: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.position).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let next = self.peek();
        if next.is_some() {
            self.position += 1;
        }
        next
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.position += 1;
        }
    }

    fn expect(&mut self, wanted: char) -> FnResult<()> {
        if self.peek() == Some(wanted) {
            self.position += 1;
            Ok(())
        } else {
            Err(syntax(format!("expected `{wanted}`")))
        }
    }

    /// Scans one value, `depth` structures deep inside the text.
    fn value(&mut self, depth: usize) -> FnResult<JsonValue> {
        match self.peek() {
            Some('{') => self.object(depth),
            Some('[') => self.array(depth),
            Some('"') => Ok(JsonValue::String(self.string()?)),
            Some('t') => self.keyword("true").map(|()| JsonValue::Bool(true)),
            Some('f') => self.keyword("false").map(|()| JsonValue::Bool(false)),
            Some('n') => self.keyword("null").map(|()| JsonValue::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(syntax(format!("`{c}` cannot start a JSON value"))),
            None => Err(syntax("the JSON text is empty")),
        }
    }

    fn keyword(&mut self, word: &str) -> FnResult<()> {
        for wanted in word.chars() {
            if self.bump() != Some(wanted) {
                return Err(syntax(format!("expected the keyword `{word}`")));
            }
        }
        Ok(())
    }

    /// Refuses to open a structure that would nest past [`Parser::max_depth`].
    ///
    /// `depth` is how many arrays and objects already enclose this one, so the structure about
    /// to be scanned sits at level `depth + 1`.
    fn enter(&self, depth: usize) -> FnResult<usize> {
        if depth >= self.max_depth {
            return Err(syntax(format!(
                "the JSON text nests more than {} levels deep, which is this parser's configurable limit",
                self.max_depth
            )));
        }
        Ok(depth + 1)
    }

    fn array(&mut self, depth: usize) -> FnResult<JsonValue> {
        let inner = self.enter(depth)?;
        self.expect('[')?;
        let mut members = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(']') {
            self.position += 1;
            return Ok(JsonValue::Array(members));
        }
        loop {
            self.skip_whitespace();
            members.push(self.value(inner)?);
            self.skip_whitespace();
            match self.bump() {
                Some(',') => {}
                Some(']') => return Ok(JsonValue::Array(members)),
                _ => return Err(syntax("expected `,` or `]` in a JSON array")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> FnResult<JsonValue> {
        let inner = self.enter(depth)?;
        self.expect('{')?;
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.position += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some('"') {
                return Err(syntax("a JSON object key must be a string"));
            }
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(':')
                .map_err(|_| syntax("expected `:` after a JSON object key"))?;
            self.skip_whitespace();
            let value = self.value(inner)?;
            self.record(&mut entries, key, value)?;
            self.skip_whitespace();
            match self.bump() {
                Some(',') => {}
                Some('}') => return Ok(JsonValue::Object(entries)),
                _ => return Err(syntax("expected `,` or `}` in a JSON object")),
            }
        }
    }

    /// Applies the `duplicates` policy as each entry arrives.
    fn record(
        &self,
        entries: &mut Vec<(String, JsonValue)>,
        key: String,
        value: JsonValue,
    ) -> FnResult<()> {
        let existing = entries.iter().position(|(seen, _)| *seen == key);
        match (self.duplicates, existing) {
            (_, None) | (Duplicates::Retain, _) => entries.push((key, value)),
            (Duplicates::Reject, Some(_)) => {
                return Err(FnError::new(
                    FnErrorCode::Fojs0003,
                    format!("the key `{key}` appears more than once in a JSON object"),
                ));
            }
            (Duplicates::UseFirst, Some(_)) => {}
            (Duplicates::UseLast, Some(index)) => entries[index] = (key, value),
        }
        Ok(())
    }

    fn number(&mut self) -> FnResult<JsonValue> {
        let start = self.position;
        if self.peek() == Some('-') {
            self.position += 1;
        }
        match self.peek() {
            Some('0') => self.position += 1,
            Some(c) if c.is_ascii_digit() => {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.position += 1;
                }
            }
            _ => return Err(syntax("a JSON number needs at least one digit")),
        }
        if self.peek() == Some('.') {
            self.position += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(syntax("a JSON fraction needs at least one digit"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.position += 1;
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.position += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.position += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(syntax("a JSON exponent needs at least one digit"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.position += 1;
            }
        }
        // A number may only be followed by structure, whitespace or the end of the text.
        if self
            .peek()
            .is_some_and(|c| c.is_alphanumeric() || c == '.' || c == '+' || c == '-')
        {
            return Err(syntax("trailing characters after a JSON number"));
        }
        let lexical: String = self.chars[start..self.position].iter().collect();
        let value = lexical
            .parse::<f64>()
            .map_err(|_| syntax(format!("`{lexical}` is not a JSON number")))?;
        Ok(JsonValue::Number(value, lexical))
    }

    /// Scans a JSON string literal and applies the [`StringPolicy`].
    fn string(&mut self) -> FnResult<String> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            let Some(next) = self.bump() else {
                return Err(syntax("an unterminated JSON string"));
            };
            match next {
                '"' => return Ok(out),
                '\\' => self.escape_sequence(&mut out)?,
                c if (c as u32) < 0x20 => {
                    return Err(syntax(
                        "an unescaped control character in a JSON string literal",
                    ));
                }
                c => self.emit(&mut out, c as u32, None)?,
            }
        }
    }

    /// Reads one backslash escape and appends its effect to `out`.
    fn escape_sequence(&mut self, out: &mut String) -> FnResult<()> {
        let Some(marker) = self.bump() else {
            return Err(syntax("a JSON string ends in a backslash"));
        };
        let simple = match marker {
            '"' => Some('"'),
            '\\' => Some('\\'),
            '/' => Some('/'),
            'b' => Some('\u{8}'),
            'f' => Some('\u{c}'),
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            'u' => None,
            other => {
                return Err(syntax(format!("`\\{other}` is not a JSON escape sequence")));
            }
        };
        if let Some(character) = simple {
            let written = format!("\\{marker}");
            return self.emit(out, character as u32, Some(&written));
        }
        let high = self.hex4()?;
        if (0xD800..0xDC00).contains(&high) {
            // A high surrogate: a following low surrogate completes a supplementary character.
            if self.peek() == Some('\\') && self.chars.get(self.position + 1) == Some(&'u') {
                let saved = self.position;
                self.position += 2;
                let low = self.hex4()?;
                if (0xDC00..0xE000).contains(&low) {
                    let combined = 0x1_0000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                    let written = format!("\\u{high:04X}\\u{low:04X}");
                    return self.emit(out, combined, Some(&written));
                }
                self.position = saved;
            }
        }
        let written = format!("\\u{high:04X}");
        self.emit(out, high, Some(&written))
    }

    fn hex4(&mut self) -> FnResult<u32> {
        let mut value = 0_u32;
        for _ in 0..4 {
            let Some(digit) = self.bump().and_then(|c| c.to_digit(16)) else {
                return Err(syntax("`\\u` must be followed by four hexadecimal digits"));
            };
            value = value * 16 + digit;
        }
        Ok(value)
    }

    /// Appends one source character to the string being built, honouring `escape` and `fallback`.
    ///
    /// `written` is the escape sequence the character was spelled with, which the `fallback`
    /// function receives and which `escape=true` reproduces.
    fn emit(&self, out: &mut String, codepoint: u32, written: Option<&str>) -> FnResult<()> {
        if self.policy.escape {
            out.push_str(&escaped_form(codepoint));
            return Ok(());
        }
        match char::from_u32(codepoint).filter(|c| is_xml_character(*c)) {
            Some(character) => out.push(character),
            None => {
                let sequence = match written {
                    Some(written) => written.to_owned(),
                    None => format!("\\u{codepoint:04X}"),
                };
                match self.policy.fallback {
                    Some(fallback) => out.push_str(&fallback(&sequence)?),
                    None => out.push('\u{FFFD}'),
                }
            }
        }
        Ok(())
    }
}

/// The form `escape=true` gives a character: an escape sequence for what JSON or XML cannot
/// spell directly, and the character itself otherwise.
fn escaped_form(codepoint: u32) -> String {
    match codepoint {
        // The quotation mark is *not* re-escaped: it needs an escape only inside a JSON literal,
        // and what is being built here is an XDM string. W3C bug #28995 settled this, and
        // `json-to-xml-049` pins it — `"\\\/\"\r\t "` becomes `\\/"\r\t `.
        0x5C => "\\\\".to_owned(),
        0x08 => "\\b".to_owned(),
        0x0C => "\\f".to_owned(),
        0x0A => "\\n".to_owned(),
        0x0D => "\\r".to_owned(),
        0x09 => "\\t".to_owned(),
        _ => match char::from_u32(codepoint).filter(|c| is_xml_character(*c)) {
            Some(character) => character.to_string(),
            // Unpaired surrogates and the characters XML forbids keep their `\uXXXX` spelling.
            None => format!("\\u{codepoint:04X}"),
        },
    }
}

/// Whether `character` is one XML 1.0 permits in character data.
#[must_use]
pub fn is_xml_character(character: char) -> bool {
    matches!(character,
        '\u{9}' | '\u{A}' | '\u{D}'
        | '\u{20}'..='\u{D7FF}'
        | '\u{E000}'..='\u{FFFD}'
        | '\u{10000}'..='\u{10FFFF}')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::JsonOptions;

    fn value(input: &str) -> FnResult<JsonValue> {
        parse(
            input,
            Duplicates::UseFirst,
            &StringPolicy::default(),
            JsonOptions::DEFAULT_MAX_DEPTH,
        )
    }

    #[test]
    fn the_four_scalar_forms_parse() -> FnResult<()> {
        assert_eq!(value("null")?, JsonValue::Null);
        assert_eq!(value("true")?, JsonValue::Bool(true));
        assert_eq!(value("93.7")?, JsonValue::Number(93.7, "93.7".to_owned()));
        assert_eq!(value("\"a\"")?, JsonValue::String("a".to_owned()));
        Ok(())
    }

    #[test]
    fn structures_nest() -> FnResult<()> {
        let parsed = value("{\"a\":[1,{}]}")?;
        let JsonValue::Object(entries) = parsed else {
            panic!("an object");
        };
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "a");
        Ok(())
    }

    #[test]
    fn malformed_texts_raise_fojs0001() {
        for bad in [
            "",
            "[",
            "[1,]",
            "{x:1}",
            "'a'",
            "[01]",
            "[.3]",
            "[+23]",
            "[1.234f0]",
            "[1,2,3,]",
            "{\"a\":=13}",
            "{}}",
            "[\"\\s\"]",
            "[\"\\u2\"]",
            "[\"\\x20\"]",
        ] {
            let error = value(bad).expect_err(bad);
            assert!(error.is(FnErrorCode::Fojs0001), "{bad}");
        }
    }

    #[test]
    fn surrogate_pairs_combine_and_lone_surrogates_fall_back() -> FnResult<()> {
        assert_eq!(
            value("\"\\uD834\\uDD1E\"")?,
            JsonValue::String("\u{1D11E}".to_owned())
        );
        assert_eq!(
            value("\"\\uDEAD\"")?,
            JsonValue::String("\u{FFFD}".to_owned())
        );
        Ok(())
    }

    #[test]
    fn the_escape_option_keeps_special_characters_escaped() -> FnResult<()> {
        let policy = StringPolicy {
            escape: true,
            fallback: None,
        };
        let depth = JsonOptions::DEFAULT_MAX_DEPTH;
        assert_eq!(
            parse("[\"\\r\"]", Duplicates::UseFirst, &policy, depth)?,
            JsonValue::Array(vec![JsonValue::String("\\r".to_owned())])
        );
        assert_eq!(
            parse("[\"\\u0025\"]", Duplicates::UseFirst, &policy, depth)?,
            JsonValue::Array(vec![JsonValue::String("%".to_owned())])
        );
        Ok(())
    }

    #[test]
    fn duplicate_keys_follow_the_option() -> FnResult<()> {
        let source = "{\"a\":1,\"b\":2,\"a\":3}";
        let policy = StringPolicy::default();
        let depth = JsonOptions::DEFAULT_MAX_DEPTH;
        let JsonValue::Object(first) = parse(source, Duplicates::UseFirst, &policy, depth)? else {
            panic!("an object");
        };
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].1, JsonValue::Number(1.0, "1".to_owned()));
        let JsonValue::Object(last) = parse(source, Duplicates::UseLast, &policy, depth)? else {
            panic!("an object");
        };
        assert_eq!(last[0].1, JsonValue::Number(3.0, "3".to_owned()));
        let JsonValue::Object(all) = parse(source, Duplicates::Retain, &policy, depth)? else {
            panic!("an object");
        };
        assert_eq!(all.len(), 3);
        assert!(parse(source, Duplicates::Reject, &policy, depth).is_err());
        Ok(())
    }

    /// A deeply nested text used to recurse once per level and abort the process with a stack
    /// overflow; it must now come back as a bounded `FOJS0001`.
    #[test]
    fn nesting_past_the_ceiling_is_rejected_rather_than_overflowing_the_stack() -> FnResult<()> {
        let limit = JsonOptions::DEFAULT_MAX_DEPTH;
        let at_the_limit = format!("{}{}", "[".repeat(limit), "]".repeat(limit));
        assert!(value(&at_the_limit).is_ok());

        for over in [limit + 1, 30_000, 200_000] {
            let source = format!("{}{}", "[".repeat(over), "]".repeat(over));
            let error = value(&source).expect_err("past the nesting ceiling");
            assert!(error.is(FnErrorCode::Fojs0001), "{over} levels");
            assert!(error.to_string().contains(&limit.to_string()), "{over}");
        }

        // Objects nest through the same production, and the two interleave.
        let objects = format!("{}1{}", "{\"a\":".repeat(30_000), "}".repeat(30_000));
        assert!(
            value(&objects)
                .expect_err("past the ceiling")
                .is(FnErrorCode::Fojs0001)
        );
        let mixed = format!("{}1{}", "[{\"a\":".repeat(20_000), "}]".repeat(20_000));
        assert!(
            value(&mixed)
                .expect_err("past the ceiling")
                .is(FnErrorCode::Fojs0001)
        );
        Ok(())
    }

    /// The ceiling travels with the options, so a caller may lower or raise it.
    #[test]
    fn the_ceiling_is_configurable() -> FnResult<()> {
        let policy = StringPolicy::default();
        assert!(parse("[[[]]]", Duplicates::UseFirst, &policy, 3).is_ok());
        assert!(parse("[[[]]]", Duplicates::UseFirst, &policy, 2).is_err());
        assert!(parse("1", Duplicates::UseFirst, &policy, 0).is_ok());
        assert!(parse("[]", Duplicates::UseFirst, &policy, 0).is_err());
        Ok(())
    }
}
