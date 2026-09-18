//! Shared lexing utilities for the SVG attribute microsyntaxes.
//!
//! Every microsyntax parser in [`crate::values`] drives a [`Cursor`] over the
//! attribute value it was handed. The cursor implements the shared terminals
//! of the SVG grammars — white space, `comma-wsp` separators, `<number>`, and
//! the single-digit flags used by elliptical arcs — so the individual parsers
//! only encode their own productions.

use crate::error::ValueError;

/// True for the four characters the SVG grammars treat as white space
/// (space, tab, carriage return, and line feed).
#[must_use]
pub const fn is_wsp(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

/// A scanning cursor over an SVG attribute value.
///
/// The cursor tracks a 0-based byte offset into the input; every error it
/// produces carries the offset at which scanning failed.
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Creates a cursor positioned at the start of `input`.
    #[must_use]
    pub const fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    /// The full input the cursor scans.
    #[must_use]
    pub const fn input(&self) -> &'a str {
        self.input
    }

    /// The current 0-based byte offset.
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// The not-yet-consumed remainder of the input.
    #[must_use]
    pub fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    /// True once the whole input has been consumed.
    #[must_use]
    pub fn is_at_end(&self) -> bool {
        self.pos >= self.input.len()
    }

    /// The next character, without consuming it.
    #[must_use]
    pub fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    /// Consumes and returns the next character.
    pub fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Consumes the next character if it equals `expected`.
    pub fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    /// Consumes the literal `text` if the input continues with it
    /// (case-sensitive comparison).
    pub fn eat_str(&mut self, text: &str) -> bool {
        if self.rest().starts_with(text) {
            self.pos += text.len();
            true
        } else {
            false
        }
    }

    /// Consumes the next character, requiring it to equal `expected`.
    pub fn expect_char(&mut self, expected: char) -> Result<(), ValueError> {
        match self.peek() {
            Some(c) if c == expected => {
                self.pos += c.len_utf8();
                Ok(())
            }
            Some(found) => Err(ValueError::UnexpectedChar {
                offset: self.pos,
                found,
            }),
            None => Err(ValueError::UnexpectedEnd { offset: self.pos }),
        }
    }

    /// Skips any run of SVG white space.
    pub fn skip_wsp(&mut self) {
        while let Some(c) = self.peek() {
            if is_wsp(c) {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    /// Skips one `comma-wsp` separator: white space with at most one comma
    /// inside it. Returns true when at least one character was consumed.
    pub fn skip_comma_wsp(&mut self) -> bool {
        let start = self.pos;
        self.skip_wsp();
        if self.eat(',') {
            self.skip_wsp();
        }
        self.pos != start
    }

    /// Scans one SVG `<number>`.
    ///
    /// The accepted grammar is the one shared by path data, transform lists,
    /// and the basic data types chapter: an optional sign, then either a
    /// digit sequence, a digit sequence with a trailing point, a fractional
    /// constant such as `.5` or `1.25`, and an optional exponent whose `e` is
    /// only consumed when digits follow it (so `1em` scans as `1` and leaves
    /// `em` in place). Values whose magnitude overflows an `f64` saturate to
    /// an infinity, as `f64` parsing defines.
    pub fn number(&mut self) -> Result<f64, ValueError> {
        let bytes = self.input.as_bytes();
        let start = self.pos;
        let mut i = self.pos;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let integer_digits = scan_digits(bytes, &mut i);
        let mut fraction_digits = false;
        if i < bytes.len() && bytes[i] == b'.' {
            let mut j = i + 1;
            fraction_digits = scan_digits(bytes, &mut j);
            if fraction_digits || integer_digits {
                i = j;
            }
        }
        if !integer_digits && !fraction_digits {
            return if self.is_at_end() {
                Err(ValueError::UnexpectedEnd { offset: start })
            } else {
                Err(ValueError::InvalidNumber { offset: start })
            };
        }
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            let mut j = i + 1;
            if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                j += 1;
            }
            if scan_digits(bytes, &mut j) {
                i = j;
            }
        }
        let text = &self.input[start..i];
        let value: f64 = text
            .parse()
            .map_err(|_| ValueError::InvalidNumber { offset: start })?;
        self.pos = i;
        Ok(value)
    }

    /// Scans one arc flag: a bare `0` or `1` digit.
    pub fn flag(&mut self) -> Result<bool, ValueError> {
        match self.peek() {
            Some('0') => {
                self.pos += 1;
                Ok(false)
            }
            Some('1') => {
                self.pos += 1;
                Ok(true)
            }
            Some(found) => Err(ValueError::UnexpectedChar {
                offset: self.pos,
                found,
            }),
            None => Err(ValueError::UnexpectedEnd { offset: self.pos }),
        }
    }

    /// Scans a keyword: an ASCII letter followed by ASCII letters, digits,
    /// and hyphens. Returns the empty string (without advancing) when the
    /// input does not start with a letter.
    pub fn keyword(&mut self) -> &'a str {
        let rest = self.rest();
        if !rest.starts_with(|c: char| c.is_ascii_alphabetic()) {
            return "";
        }
        let len = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
            .unwrap_or(rest.len());
        let keyword = &rest[..len];
        self.pos += len;
        keyword
    }

    /// Skips trailing white space and requires the input to end there.
    pub fn done(&mut self) -> Result<(), ValueError> {
        self.skip_wsp();
        if self.is_at_end() {
            Ok(())
        } else {
            Err(ValueError::TrailingData { offset: self.pos })
        }
    }
}

fn scan_digits(bytes: &[u8], i: &mut usize) -> bool {
    let start = *i;
    while *i < bytes.len() && bytes[*i].is_ascii_digit() {
        *i += 1;
    }
    *i != start
}

/// Parses a complete attribute value as one SVG `<number>`, allowing
/// surrounding white space.
pub fn parse_number(input: &str) -> Result<f64, ValueError> {
    let mut cursor = Cursor::new(input);
    cursor.skip_wsp();
    if cursor.is_at_end() {
        return Err(ValueError::Empty);
    }
    let value = cursor.number()?;
    cursor.done()?;
    Ok(value)
}

/// Parses a complete attribute value as a `comma-wsp`-separated list of SVG
/// `<number>`s. An empty (or white-space-only) value yields an empty list;
/// callers whose grammar requires content impose that themselves.
pub fn parse_number_list(input: &str) -> Result<Vec<f64>, ValueError> {
    let mut cursor = Cursor::new(input);
    let mut numbers = Vec::new();
    cursor.skip_wsp();
    if cursor.is_at_end() {
        return Ok(numbers);
    }
    loop {
        numbers.push(cursor.number()?);
        let separated = cursor.skip_comma_wsp();
        if cursor.is_at_end() {
            return Ok(numbers);
        }
        if !separated && !matches!(cursor.peek(), Some('+' | '-' | '.' | '0'..='9')) {
            return Err(ValueError::TrailingData {
                offset: cursor.pos(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_cover_the_svg_grammar() {
        for (text, expected) in [
            ("0", 0.0),
            ("15", 15.0),
            ("-4", -4.0),
            ("+4", 4.0),
            (".5", 0.5),
            ("-.5", -0.5),
            ("1.", 1.0),
            ("1.25", 1.25),
            ("1e2", 100.0),
            ("1E2", 100.0),
            ("1e+2", 100.0),
            ("-1e-2", -0.01),
            (".5e2", 50.0),
            ("1.e2", 100.0),
        ] {
            let mut cursor = Cursor::new(text);
            let value = cursor.number().expect("grammar-valid number");
            assert_eq!(value, expected, "parsing {text:?}");
            assert!(cursor.is_at_end(), "whole input consumed for {text:?}");
        }
    }

    #[test]
    fn exponent_is_left_when_not_followed_by_digits() {
        let mut cursor = Cursor::new("1em");
        assert_eq!(cursor.number().expect("leading 1 is a number"), 1.0);
        assert_eq!(cursor.rest(), "em");
    }

    #[test]
    fn second_point_starts_a_new_number() {
        let mut cursor = Cursor::new("1.5.5");
        assert_eq!(cursor.number().expect("first number"), 1.5);
        assert_eq!(cursor.number().expect("second number"), 0.5);
        assert!(cursor.is_at_end());
    }

    #[test]
    fn malformed_numbers_are_rejected_with_offsets() {
        let mut cursor = Cursor::new("x");
        assert_eq!(
            cursor.number(),
            Err(ValueError::InvalidNumber { offset: 0 })
        );
        let mut sign_only = Cursor::new("+");
        assert_eq!(
            sign_only.number(),
            Err(ValueError::InvalidNumber { offset: 0 })
        );
        let mut point_only = Cursor::new(".");
        assert_eq!(
            point_only.number(),
            Err(ValueError::InvalidNumber { offset: 0 })
        );
        let mut empty = Cursor::new("");
        assert_eq!(empty.number(), Err(ValueError::UnexpectedEnd { offset: 0 }));
    }

    #[test]
    fn comma_wsp_allows_at_most_one_comma() {
        let mut cursor = Cursor::new(" , ,");
        assert!(cursor.skip_comma_wsp());
        assert_eq!(cursor.rest(), ",");
    }

    #[test]
    fn flags_are_single_digits() {
        let mut cursor = Cursor::new("011");
        assert_eq!(cursor.flag(), Ok(false));
        assert_eq!(cursor.flag(), Ok(true));
        assert_eq!(cursor.flag(), Ok(true));
        assert!(cursor.is_at_end());
    }

    #[test]
    fn keywords_stop_at_non_name_characters() {
        let mut cursor = Cursor::new("meetOrSlice rest");
        assert_eq!(cursor.keyword(), "meetOrSlice");
        assert_eq!(cursor.rest(), " rest");
        let mut none = Cursor::new("1px");
        assert_eq!(none.keyword(), "");
        assert_eq!(none.pos(), 0);
    }

    #[test]
    fn number_lists_accept_all_separator_styles() {
        assert_eq!(
            parse_number_list("1 2,3 ,4, 5").expect("valid list"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert_eq!(
            parse_number_list("1-2.3.4").expect("signs and points separate"),
            vec![1.0, -2.3, 0.4]
        );
        assert_eq!(
            parse_number_list("  ").expect("empty list"),
            Vec::<f64>::new()
        );
        assert_eq!(
            parse_number_list("1,,2"),
            Err(ValueError::InvalidNumber { offset: 2 })
        );
    }

    #[test]
    fn whole_value_parsing_flags_trailing_data() {
        assert_eq!(parse_number(" 42 ").expect("padded number"), 42.0);
        assert_eq!(parse_number(""), Err(ValueError::Empty));
        assert_eq!(
            parse_number("42 x"),
            Err(ValueError::TrailingData { offset: 3 })
        );
    }
}
