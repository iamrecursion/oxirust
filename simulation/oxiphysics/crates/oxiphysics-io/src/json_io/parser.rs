// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Recursive-descent JSON text parser behind [`JsonValue::parse`] and
//! [`JsonStreamParser`](super::JsonStreamParser).
//!
//! Every function takes a `&str` and returns the parsed value together with
//! the unparsed remainder.  The input is treated as untrusted:
//!
//! * arrays and objects may nest at most [`MAX_PARSE_DEPTH`] levels, so a
//!   pathological document cannot exhaust the call stack;
//! * error messages quote the offending input by whole characters, never by
//!   byte offsets, so multi-byte UTF-8 text cannot trigger a char-boundary
//!   panic;
//! * `\uXXXX` escapes need exactly four hexadecimal digits, and UTF-16
//!   surrogate pairs are combined into one scalar value.

use std::str::Chars;

use super::{JsonValue, MAX_PARSE_DEPTH};

/// Skip JSON insignificant whitespace (space, tab, line feed, carriage return).
fn skip_whitespace(s: &str) -> &str {
    s.trim_start_matches(|c: char| c.is_ascii_whitespace())
}

/// The first `n` characters of `s`, for quoting input in error messages.
///
/// Counting characters (not bytes) keeps the cut on a char boundary.
fn snippet(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Account for entering one more array/object; fails past [`MAX_PARSE_DEPTH`].
///
/// `depth` is the number of containers already entered.
fn enter_container(depth: usize) -> Result<usize, String> {
    if depth >= MAX_PARSE_DEPTH {
        Err(format!("JSON nesting exceeds {MAX_PARSE_DEPTH} levels"))
    } else {
        Ok(depth + 1)
    }
}

/// Parse one JSON value from the start of `s`.
///
/// `depth` is the number of arrays/objects that enclose this value; callers
/// start at `0`.
pub(super) fn parse_value(s: &str, depth: usize) -> Result<(JsonValue, &str), String> {
    let s = skip_whitespace(s);
    let Some(first) = s.chars().next() else {
        return Err("Unexpected end of input".to_string());
    };
    match first {
        'n' => parse_null(s),
        't' | 'f' => parse_bool(s),
        '"' => parse_string(s),
        '[' => parse_array(s, depth),
        '{' => parse_object(s, depth),
        '-' | '0'..='9' => parse_number(s),
        c => Err(format!("Unexpected character: '{c}'")),
    }
}

fn parse_null(s: &str) -> Result<(JsonValue, &str), String> {
    s.strip_prefix("null")
        .map(|rest| (JsonValue::Null, rest))
        .ok_or_else(|| format!("Expected 'null', got: {:?}", snippet(s, 6)))
}

fn parse_bool(s: &str) -> Result<(JsonValue, &str), String> {
    if let Some(rest) = s.strip_prefix("true") {
        return Ok((JsonValue::Bool(true), rest));
    }
    if let Some(rest) = s.strip_prefix("false") {
        return Ok((JsonValue::Bool(false), rest));
    }
    Err(format!("Expected boolean, got: {:?}", snippet(s, 6)))
}

fn parse_number(s: &str) -> Result<(JsonValue, &str), String> {
    let end = s
        .find(|c: char| !matches!(c, '-' | '+' | '.' | 'e' | 'E' | '0'..='9'))
        .unwrap_or(s.len());
    let num_str = &s[..end];
    let n: f64 = num_str
        .parse()
        .map_err(|e| format!("Number parse error '{}': {}", num_str, e))?;
    Ok((JsonValue::Number(n), &s[end..]))
}

/// Read exactly four hexadecimal digits of a `\uXXXX` escape.
///
/// A short escape, a sign character or any non-hex digit is an error; nothing
/// is padded or guessed.
fn parse_hex4(chars: &mut Chars<'_>) -> Result<u32, String> {
    let mut code = 0u32;
    for _ in 0..4 {
        let c = chars
            .next()
            .ok_or_else(|| "Unterminated \\u escape: expected 4 hex digits".to_string())?;
        let digit = c
            .to_digit(16)
            .ok_or_else(|| format!("Invalid hex digit {c:?} in \\u escape"))?;
        code = code * 16 + digit;
    }
    Ok(code)
}

/// Turn the code unit of a `\uXXXX` escape into a `char`.
///
/// A high surrogate (`D800..=DBFF`) must be followed by a `\uDC00..=\uDFFF`
/// escape; the pair is combined into one supplementary-plane character.  A
/// low surrogate on its own is rejected.
fn decode_unicode_escape(code: u32, chars: &mut Chars<'_>) -> Result<char, String> {
    match code {
        0xD800..=0xDBFF => {
            if chars.next() != Some('\\') || chars.next() != Some('u') {
                return Err(format!(
                    "Unpaired high surrogate \\u{code:04X}: expected a following \\u escape"
                ));
            }
            let low = parse_hex4(chars)?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err(format!(
                    "Invalid low surrogate \\u{low:04X} after high surrogate \\u{code:04X}"
                ));
            }
            let combined = 0x1_0000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            char::from_u32(combined)
                .ok_or_else(|| format!("Invalid unicode codepoint U+{combined:04X}"))
        }
        0xDC00..=0xDFFF => Err(format!("Unpaired low surrogate \\u{code:04X}")),
        _ => char::from_u32(code).ok_or_else(|| format!("Invalid unicode codepoint U+{code:04X}")),
    }
}

/// Parse a string literal and return its decoded contents.
///
/// `s` must start with the opening quote.
fn parse_string_contents(s: &str) -> Result<(String, &str), String> {
    let Some(body) = s.strip_prefix('"') else {
        return Err(format!("Expected string, got: {:?}", snippet(s, 5)));
    };
    let mut result = String::new();
    let mut chars = body.chars();
    loop {
        match chars.next() {
            None => return Err("Unterminated string".to_string()),
            Some('"') => return Ok((result, chars.as_str())),
            Some('\\') => match chars.next() {
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('/') => result.push('/'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('b') => result.push('\x08'),
                Some('f') => result.push('\x0C'),
                Some('u') => {
                    let code = parse_hex4(&mut chars)?;
                    result.push(decode_unicode_escape(code, &mut chars)?);
                }
                Some(c) => return Err(format!("Invalid escape \\{c}")),
                None => return Err("Unterminated escape".to_string()),
            },
            Some(c) => result.push(c),
        }
    }
}

fn parse_string(s: &str) -> Result<(JsonValue, &str), String> {
    parse_string_contents(s).map(|(text, rest)| (JsonValue::Str(text), rest))
}

fn parse_array(s: &str, depth: usize) -> Result<(JsonValue, &str), String> {
    let depth = enter_container(depth)?;
    // `s` starts with the ASCII character '['.
    let mut s = skip_whitespace(&s[1..]);
    let mut items = Vec::new();
    if let Some(rest) = s.strip_prefix(']') {
        return Ok((JsonValue::Array(items), rest));
    }
    loop {
        let (val, rest) = parse_value(s, depth)?;
        items.push(val);
        s = skip_whitespace(rest);
        if let Some(rest) = s.strip_prefix(']') {
            return Ok((JsonValue::Array(items), rest));
        }
        if let Some(rest) = s.strip_prefix(',') {
            s = skip_whitespace(rest);
        } else {
            return Err(format!(
                "Expected ',' or ']' in array, got: {:?}",
                snippet(s, 5)
            ));
        }
    }
}

fn parse_object(s: &str, depth: usize) -> Result<(JsonValue, &str), String> {
    let depth = enter_container(depth)?;
    // `s` starts with the ASCII character '{'.
    let mut s = skip_whitespace(&s[1..]);
    let mut pairs = Vec::new();
    if let Some(rest) = s.strip_prefix('}') {
        return Ok((JsonValue::Object(pairs), rest));
    }
    loop {
        s = skip_whitespace(s);
        if !s.starts_with('"') {
            return Err(format!(
                "Expected string key in object, got: {:?}",
                snippet(s, 5)
            ));
        }
        let (key, rest) = parse_string_contents(s)?;
        s = skip_whitespace(rest);
        let colon_rest = s
            .strip_prefix(':')
            .ok_or_else(|| format!("Expected ':' after key, got: {:?}", snippet(s, 5)))?;
        s = skip_whitespace(colon_rest);
        let (val, rest) = parse_value(s, depth)?;
        pairs.push((key, val));
        s = skip_whitespace(rest);
        if let Some(rest) = s.strip_prefix('}') {
            return Ok((JsonValue::Object(pairs), rest));
        }
        if let Some(rest) = s.strip_prefix(',') {
            s = skip_whitespace(rest);
        } else {
            return Err(format!(
                "Expected ',' or '}}' in object, got: {:?}",
                snippet(s, 5)
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_io::JsonStreamParser;

    /// `n` nested empty-array brackets: `[[[...]]]`.
    fn nested_arrays(n: usize) -> String {
        format!("{}{}", "[".repeat(n), "]".repeat(n))
    }

    #[test]
    fn json_deep_nesting_returns_err() {
        // An unterminated opening run, a balanced run and a run of objects, each
        // far deeper than the call stack could take if the parser recursed
        // without a limit.
        let open_only = "[".repeat(100_000);
        let err = JsonValue::parse(&open_only).expect_err("100k open brackets");
        assert!(err.contains("nesting"), "unexpected message: {err}");

        let balanced = nested_arrays(100_000);
        let err = JsonValue::parse(&balanced).expect_err("100k balanced brackets");
        assert!(err.contains("nesting"), "unexpected message: {err}");

        let objects = "{\"a\":".repeat(100_000);
        let err = JsonValue::parse(&objects).expect_err("100k nested objects");
        assert!(err.contains("nesting"), "unexpected message: {err}");

        let mixed = "[{\"a\":".repeat(50_000);
        assert!(JsonValue::parse(&mixed).is_err());
    }

    #[test]
    fn json_nesting_limit_is_exact() {
        let at_limit = nested_arrays(MAX_PARSE_DEPTH);
        let mut value = JsonValue::parse(&at_limit).expect("exactly MAX_PARSE_DEPTH levels");
        for _ in 1..MAX_PARSE_DEPTH {
            match value {
                JsonValue::Array(mut items) => {
                    assert_eq!(items.len(), 1);
                    value = items.remove(0);
                }
                other => panic!("expected a nested array, got {other:?}"),
            }
        }
        assert_eq!(value, JsonValue::Array(Vec::new()));

        let past_limit = nested_arrays(MAX_PARSE_DEPTH + 1);
        assert!(JsonValue::parse(&past_limit).is_err());

        // Objects count towards the same budget.
        let obj_at_limit = format!(
            "{}1{}",
            "{\"k\":".repeat(MAX_PARSE_DEPTH),
            "}".repeat(MAX_PARSE_DEPTH)
        );
        assert!(JsonValue::parse(&obj_at_limit).is_ok());
        let obj_past_limit = format!(
            "{}1{}",
            "{\"k\":".repeat(MAX_PARSE_DEPTH + 1),
            "}".repeat(MAX_PARSE_DEPTH + 1)
        );
        assert!(JsonValue::parse(&obj_past_limit).is_err());
    }

    #[test]
    fn json_wide_documents_are_not_limited_by_depth() {
        let wide = format!("[{}]", vec!["[1,2,3]"; 10_000].join(","));
        let parsed = JsonValue::parse(&wide).expect("wide but shallow document");
        assert_eq!(parsed.len(), 10_000);
    }

    #[test]
    fn json_stream_parser_survives_deep_nesting() {
        let mut parser = JsonStreamParser::new();
        let produced = parser.feed(&"{\"a\":".repeat(100_000));
        assert_eq!(produced, 0);
        assert_eq!(parser.result_count(), 0);
    }

    #[test]
    fn json_multibyte_error_no_panic() {
        // Each of these used to slice the error snippet at a fixed byte offset
        // that falls inside a multi-byte character.
        let inputs = [
            "[1 日日]",
            "{ééé}",
            "{\"a\" 日日日}",
            "{\"a\":1 日日}",
            "nul日",
            "nu日日日日",
            "tru日日日",
            "fals日日日",
            "[日]",
            "日",
            "{日:1}",
            "[\"a\" é]",
            "{\"é\":é}",
        ];
        for input in inputs {
            let err = JsonValue::parse(input).expect_err(input);
            assert!(!err.is_empty(), "empty message for {input:?}");
        }
    }

    #[test]
    fn json_multibyte_error_messages_quote_whole_characters() {
        let err = JsonValue::parse("[1 日日]").expect_err("missing comma");
        assert!(
            err.contains("日日"),
            "snippet should keep both characters: {err}"
        );

        let err = JsonValue::parse("{ééé}").expect_err("key is not a string");
        assert!(
            err.contains("ééé"),
            "snippet should keep all characters: {err}"
        );

        let err = JsonValue::parse("日本語").expect_err("not JSON");
        assert!(err.contains('日'), "the real character is reported: {err}");
    }

    #[test]
    fn json_multibyte_sweep_never_panics() {
        let doc = r#"{"é日":[1, -2.5e3, "x\u00e9\uD83D\uDE00", true, null, {"k": "日本語"}], "z": false}"#;
        assert!(JsonValue::parse(doc).is_ok());
        // Every prefix and suffix cut at a char boundary.
        for (i, _) in doc.char_indices() {
            let _ = JsonValue::parse(&doc[..i]);
            let _ = JsonValue::parse(&doc[i..]);
        }
        // Every single character replaced by, or followed by, multi-byte text.
        let chars: Vec<char> = doc.chars().collect();
        for i in 0..chars.len() {
            for filler in ['日', 'é', '😀'] {
                let replaced: String = chars
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| if j == i { filler } else { c })
                    .collect();
                let _ = JsonValue::parse(&replaced);
                let mut inserted = chars.clone();
                inserted.insert(i, filler);
                let _ = JsonValue::parse(&inserted.into_iter().collect::<String>());
            }
        }
    }

    #[test]
    fn json_surrogate_pair_emoji() {
        let parsed = JsonValue::parse("\"\\uD83D\\uDE00\"").expect("surrogate pair");
        assert_eq!(parsed, JsonValue::Str("\u{1F600}".to_string()));
        assert_eq!(parsed.as_str().map(|s| s.chars().count()), Some(1));

        // Hex digits are case-insensitive.
        let lower = JsonValue::parse("\"\\ud83d\\ude00\"").expect("lowercase pair");
        assert_eq!(lower, parsed);

        // Surrounding text, a second pair and the first/last supplementary
        // code points.
        let text = JsonValue::parse("\"a\\uD83D\\uDE00b\\uD834\\uDD1Ec\"").expect("mixed text");
        assert_eq!(text, JsonValue::Str("a\u{1F600}b\u{1D11E}c".to_string()));
        let first = JsonValue::parse("\"\\uD800\\uDC00\"").expect("U+10000");
        assert_eq!(first, JsonValue::Str("\u{10000}".to_string()));
        let last = JsonValue::parse("\"\\uDBFF\\uDFFF\"").expect("U+10FFFF");
        assert_eq!(last, JsonValue::Str("\u{10FFFF}".to_string()));

        // Object keys use the same string decoder.
        let object = JsonValue::parse("{\"\\uD83D\\uDE00\": 1}").expect("emoji key");
        assert_eq!(
            object.get("\u{1F600}").and_then(JsonValue::as_f64),
            Some(1.0)
        );

        // A written emoji (raw UTF-8) round-trips through the serializer.
        let written = JsonValue::Str("\u{1F600} \u{65E5}".to_string());
        let reparsed = JsonValue::parse(&written.to_json_string()).expect("round trip");
        assert_eq!(reparsed, written);
    }

    #[test]
    fn json_unpaired_surrogates_are_errors() {
        for input in [
            "\"\\uD83D\"",        // high surrogate at the end of the string
            "\"\\uD83Dx\"",       // followed by a plain character
            "\"\\uD83D\\n\"",     // followed by a different escape
            "\"\\uD83D\\u0041\"", // followed by a non-surrogate escape
            "\"\\uD83D\\uD83D\"", // followed by another high surrogate
            "\"\\uDE00\"",        // lone low surrogate
            "\"\\uDE00\\uD83D\"", // reversed pair
            "\"\\uD83D",          // truncated input
            "\"\\uD83D\\u",       // truncated second escape
        ] {
            assert!(
                JsonValue::parse(input).is_err(),
                "{input:?} must be rejected"
            );
        }
    }

    #[test]
    fn json_unicode_escape_requires_four_hex_digits() {
        for input in [
            "\"\\u\"",     // no digits
            "\"\\u12\"",   // short escape: the closing quote is not a hex digit
            "\"\\u123",    // input ends inside the escape (no zero padding)
            "\"\\u+041\"", // sign accepted by integer parsing, not by JSON
            "\"\\u-041\"",
            "\"\\u 041\"",
            "\"\\u12G4\"",
            "\"\\u0x41\"",
            "\"\\u日日日日\"", // multi-byte characters in the digit positions
            "\"\\u00é9\"",
        ] {
            assert!(
                JsonValue::parse(input).is_err(),
                "{input:?} must be rejected"
            );
        }
    }

    #[test]
    fn json_bmp_unicode_escapes_still_decode() {
        let parsed = JsonValue::parse("\"\\u0041\\u00e9\\u65E5\\u0000\"").expect("BMP escapes");
        assert_eq!(parsed, JsonValue::Str("A\u{e9}\u{65E5}\0".to_string()));
        let boundary = JsonValue::parse("\"\\uD7FF\\uE000\"").expect("edges of the surrogate gap");
        assert_eq!(boundary, JsonValue::Str("\u{D7FF}\u{E000}".to_string()));
    }

    #[test]
    fn json_unexpected_multibyte_character_is_reported_whole() {
        let err = JsonValue::parse("日").expect_err("not a JSON value");
        assert_eq!(err, "Unexpected character: '日'");
    }

    #[test]
    fn json_error_paths_report_the_remaining_text() {
        let err = JsonValue::parse("[1 2]").expect_err("missing comma");
        assert!(err.contains("Expected ',' or ']'"), "{err}");
        let err = JsonValue::parse("{\"a\" 1}").expect_err("missing colon");
        assert!(err.contains("Expected ':'"), "{err}");
        let err = JsonValue::parse("{\"a\":1 \"b\"}").expect_err("missing comma");
        assert!(err.contains("Expected ',' or '}'"), "{err}");
        let err = JsonValue::parse("{1:2}").expect_err("numeric key");
        assert!(err.contains("Expected string key"), "{err}");
        assert!(JsonValue::parse("[1,]").is_err());
        assert!(JsonValue::parse("").is_err());
    }
}
