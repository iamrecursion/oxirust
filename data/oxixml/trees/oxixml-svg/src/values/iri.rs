//! `<funciri>` (`url(...)`) values and `href`/`xlink:href` reference handling.
//!
//! # `<funciri>` grammar
//! ```text
//! funciri ::= "url(" wsp* iri wsp* ")"
//! iri     ::= unquoted-iri-chars | "'" iri-chars-no-squote "'" | '"' iri-chars-no-dquote '"'
//! ```
//! [`FuncIri`] accepts the unquoted form, both single- and double-quoted
//! forms, and white space around the inner reference (`url( #id )`). The
//! `url(` keyword is matched case-insensitively (`URL(#a)` is accepted), per
//! CSS's general ASCII-case-insensitive keyword matching.
//!
//! This module implements the grammar exactly as transcribed above, which —
//! unlike the full CSS `url()` token — has **no backslash-escape mechanism**:
//! a quoted reference runs until the next occurrence of its own quote
//! character, full stop. This is deliberately simple and matches what SVG's
//! own `<funciri>` production (rather than CSS's more general `url()` token)
//! actually specifies; a reference that must itself contain an unescaped
//! copy of both `'` and `"` cannot be represented without ambiguity by this
//! grammar and [`FuncIri`]'s [`Display`](fmt::Display) impl falls back to
//! double-quoting it as a best effort in that pathological case (fragment
//! identifiers, being XML names, cannot contain quote characters at all, so
//! this only matters for hand-crafted or malformed absolute IRI references).
//!
//! # Scope note: fragment-only references
//! The overwhelmingly common case in SVG is a fragment-only reference
//! (`url(#gradient)`, `href="#gradient"`) pointing at another element in the
//! same document. [`FuncIri::reference`] returns the raw captured text
//! between the delimiters verbatim (including a leading `#`, if present, or
//! an absolute/relative IRI if that is what was authored); this module does
//! not resolve or validate absolute/relative external references (against a
//! document base URI or `xml:base`) — that is a larger, separate concern for
//! the typed document layer. [`local_reference`] is the narrow helper for
//! the fragment-only case.

use std::fmt;

use crate::error::ValueError;
use crate::values::lex::{Cursor, is_wsp};

/// A parsed `<funciri>` value: `url(...)`, holding the raw reference text
/// found between the delimiters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncIri {
    reference: Box<str>,
}

impl FuncIri {
    /// Creates a `FuncIri` directly from a reference string, without parsing.
    /// Useful for constructing values programmatically (e.g. from a writer
    /// or builder API) rather than from parsed attribute text.
    #[must_use]
    pub fn new(reference: impl Into<Box<str>>) -> Self {
        Self {
            reference: reference.into(),
        }
    }

    /// Parses a complete `<funciri>` value: `url(...)`, with optional
    /// surrounding white space, in any of the unquoted/single-quoted/
    /// double-quoted forms.
    ///
    /// # Errors
    /// Returns [`ValueError`] if `input` is empty, does not start with (a
    /// case-insensitive) `url(`, has an unterminated quoted reference, or is
    /// missing its closing `)`.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let mut cursor = Cursor::new(input);
        cursor.skip_wsp();
        if cursor.is_at_end() {
            return Err(ValueError::Empty);
        }
        let iri = parse_from_cursor(&mut cursor)?;
        cursor.done()?;
        Ok(iri)
    }

    /// The raw reference text found between `url(`'s delimiters (quotes
    /// stripped, if any were present), verbatim.
    #[must_use]
    pub fn reference(&self) -> &str {
        &self.reference
    }

    /// Consumes this value, returning the owned reference text.
    #[must_use]
    pub fn into_reference(self) -> Box<str> {
        self.reference
    }
}

impl fmt::Display for FuncIri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_url(f, &self.reference)
    }
}

/// Parses one `<funciri>` value starting at the cursor's current position,
/// leaving anything after the closing `)` (such as a `<paint>` fallback) for
/// the caller. This is the composition point [`crate::values::paint`] uses.
pub(crate) fn parse_from_cursor(cursor: &mut Cursor<'_>) -> Result<FuncIri, ValueError> {
    let start = cursor.pos();
    if !eat_url_prefix(cursor) {
        return Err(match cursor.peek() {
            None => ValueError::UnexpectedEnd { offset: start },
            Some(_) => {
                let mut probe = cursor.clone();
                let word = probe.keyword();
                if word.is_empty() {
                    ValueError::UnexpectedChar {
                        offset: start,
                        found: cursor
                            .peek()
                            .expect("peek() was Some in the outer match arm"),
                    }
                } else {
                    ValueError::UnknownKeyword {
                        offset: start,
                        keyword: word.into(),
                    }
                }
            }
        });
    }
    cursor.skip_wsp();
    let reference: Box<str> = match cursor.peek() {
        Some('"') => {
            cursor.bump();
            scan_quoted(cursor, '"')?.into()
        }
        Some('\'') => {
            cursor.bump();
            scan_quoted(cursor, '\'')?.into()
        }
        _ => scan_unquoted(cursor).into(),
    };
    cursor.skip_wsp();
    cursor.expect_char(')')?;
    Ok(FuncIri { reference })
}

/// Reports whether the cursor's remaining input begins with a
/// case-insensitive `url(`, without consuming anything. Used by
/// [`crate::values::paint`] to decide whether a `<paint>` value is a
/// `<funciri>` (with a possible fallback) or a plain `<color>`.
pub(crate) fn looks_like_url(cursor: &Cursor<'_>) -> bool {
    let rest = cursor.rest();
    rest.len() >= 4 && rest.as_bytes()[..4].eq_ignore_ascii_case(b"url(")
}

fn eat_url_prefix(cursor: &mut Cursor<'_>) -> bool {
    if looks_like_url(cursor) {
        for _ in 0..4 {
            cursor.bump();
        }
        true
    } else {
        false
    }
}

/// Scans unquoted reference text up to (not including) the next white space
/// character or `)`.
fn scan_unquoted<'a>(cursor: &mut Cursor<'a>) -> &'a str {
    let start = cursor.pos();
    while let Some(c) = cursor.peek() {
        if is_wsp(c) || c == ')' {
            break;
        }
        cursor.bump();
    }
    &cursor.input()[start..cursor.pos()]
}

/// Scans quoted reference text up to (not including) the next occurrence of
/// `quote`, consuming that closing quote. There is no escape mechanism: any
/// other character, including the *other* quote character, is literal
/// content (this is what makes `url("it's fine")` and `url('say "hi"')`
/// work — the "nested quotes" case).
fn scan_quoted<'a>(cursor: &mut Cursor<'a>, quote: char) -> Result<&'a str, ValueError> {
    let start = cursor.pos();
    while let Some(c) = cursor.peek() {
        if c == quote {
            let content = &cursor.input()[start..cursor.pos()];
            cursor.bump();
            return Ok(content);
        }
        cursor.bump();
    }
    Err(ValueError::UnexpectedEnd {
        offset: cursor.pos(),
    })
}

/// Writes `url(...)`, quoting `reference` only when its content requires it
/// (contains white space, `)`, or a quote character).
pub(crate) fn write_url(f: &mut fmt::Formatter<'_>, reference: &str) -> fmt::Result {
    f.write_str("url(")?;
    if needs_quoting(reference) {
        // Prefer double quotes; fall back to single quotes only when the
        // reference contains a double quote but no single quote. A
        // reference containing both is a pathological case this grammar
        // cannot represent losslessly (see the module doc); double-quoting
        // it anyway is the least-surprising best effort.
        let quote = if reference.contains('"') && !reference.contains('\'') {
            '\''
        } else {
            '"'
        };
        write!(f, "{quote}{reference}{quote}")?;
    } else {
        f.write_str(reference)?;
    }
    f.write_str(")")
}

fn needs_quoting(reference: &str) -> bool {
    reference
        .chars()
        .any(|c| is_wsp(c) || c == '\'' || c == '"' || c == ')')
}

/// Extracts the local (same-document) identifier from a fragment-only
/// reference, `"#id"` -> `"id"`. Returns [`None`] if `href` does not begin
/// with `#` (an absolute or relative external reference, which this helper
/// does not resolve) — it does not validate that what follows `#` is a
/// well-formed XML name.
#[must_use]
pub fn local_reference(href: &str) -> Option<&str> {
    href.strip_prefix('#')
}

/// Picks the effective reference between an SVG2 `href` and a legacy
/// `xlink:href`, per SVG2's precedence rule: a bare `href` wins when both
/// are present on the same element.
#[must_use]
pub fn effective_href<'a>(href: Option<&'a str>, xlink_href: Option<&'a str>) -> Option<&'a str> {
    href.or(xlink_href)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FuncIri parsing ----

    #[test]
    fn plain_fragment_reference() {
        let iri = FuncIri::parse("url(#gradient1)").expect("valid");
        assert_eq!(iri.reference(), "#gradient1");
    }

    #[test]
    fn url_keyword_is_case_insensitive() {
        assert_eq!(FuncIri::parse("URL(#a)").expect("valid").reference(), "#a");
        assert_eq!(FuncIri::parse("UrL(#a)").expect("valid").reference(), "#a");
    }

    #[test]
    fn empty_url_is_accepted_with_an_empty_reference() {
        let iri = FuncIri::parse("url()").expect("empty url() is valid");
        assert_eq!(iri.reference(), "");
    }

    #[test]
    fn double_and_single_quoted_forms() {
        assert_eq!(
            FuncIri::parse(r##"url("#a")"##).expect("valid").reference(),
            "#a"
        );
        assert_eq!(
            FuncIri::parse("url('#a')").expect("valid").reference(),
            "#a"
        );
    }

    #[test]
    fn nested_quotes_inside_the_opposite_quote_style() {
        // A single-quoted reference may freely contain unescaped double
        // quotes (and vice versa) since scanning only watches for the quote
        // character that opened the reference — there is no escape
        // mechanism in this grammar (see the module doc), so a doubled copy
        // of the *same* quote character is not an escape sequence, it simply
        // closes the string early.
        assert_eq!(
            FuncIri::parse(r#"url('it is "quoted" here')"#)
                .expect("valid")
                .reference(),
            r#"it is "quoted" here"#
        );
        assert_eq!(
            FuncIri::parse(r#"url("say 'hi'")"#)
                .expect("valid")
                .reference(),
            "say 'hi'"
        );
    }

    #[test]
    fn doubled_same_quote_closes_the_reference_early_no_escaping() {
        // Documents the no-escape-mechanism behavior explicitly: the second
        // `'` in "it''s" closes the reference right after "it", so the
        // parser next expects `)` and instead finds the `'` that opens "'s'"
        // — a grammar error, not a successful parse of "it's".
        let result = FuncIri::parse("url('it''s')");
        assert_eq!(
            result,
            Err(ValueError::UnexpectedChar {
                offset: 8,
                found: '\''
            })
        );
    }

    #[test]
    fn whitespace_inside_url_parens_is_allowed() {
        let iri = FuncIri::parse("url( #a )").expect("valid");
        assert_eq!(iri.reference(), "#a");
        let quoted = FuncIri::parse("url(  \"#a\"  )").expect("valid");
        assert_eq!(quoted.reference(), "#a");
    }

    #[test]
    fn unclosed_paren_reports_the_end_of_input_offset() {
        let input = "url(#id";
        let result = FuncIri::parse(input);
        assert_eq!(
            result,
            Err(ValueError::UnexpectedEnd {
                offset: input.len()
            })
        );
    }

    #[test]
    fn unterminated_quote_is_unexpected_end() {
        let result = FuncIri::parse(r##"url("#a)"##);
        assert!(
            matches!(result, Err(ValueError::UnexpectedEnd { .. })),
            "got {result:?}"
        );
    }

    #[test]
    fn missing_url_prefix_is_rejected() {
        assert!(matches!(
            FuncIri::parse("#a"),
            Err(ValueError::UnexpectedChar {
                offset: 0,
                found: '#'
            })
        ));
        assert!(matches!(
            FuncIri::parse("something(#a)"),
            Err(ValueError::UnknownKeyword { .. })
        ));
    }

    #[test]
    fn empty_input_is_empty_error() {
        assert_eq!(FuncIri::parse(""), Err(ValueError::Empty));
        assert_eq!(FuncIri::parse("   "), Err(ValueError::Empty));
    }

    #[test]
    fn trailing_data_after_a_complete_funciri_is_reported() {
        let result = FuncIri::parse("url(#a) red");
        assert!(
            matches!(result, Err(ValueError::TrailingData { .. })),
            "got {result:?}"
        );
    }

    // ---- cursor-based composition (what paint.rs relies on) ----

    #[test]
    fn parse_from_cursor_leaves_a_paint_fallback_untouched() {
        for (input, rest) in [
            ("url(#g) red", " red"),
            ("url(#g) none", " none"),
            ("url(#g)", ""),
            ("url( '#g' ) currentColor", " currentColor"),
        ] {
            let mut cursor = Cursor::new(input);
            let iri = parse_from_cursor(&mut cursor).expect("valid funciri prefix");
            assert_eq!(iri.reference(), "#g", "input {input:?}");
            assert_eq!(cursor.rest(), rest, "input {input:?}");
        }
    }

    // ---- Display / round-trip ----

    #[test]
    fn display_omits_quotes_when_not_needed() {
        assert_eq!(FuncIri::new("#a").to_string(), "url(#a)");
        assert_eq!(FuncIri::new("").to_string(), "url()");
    }

    #[test]
    fn display_quotes_only_when_content_requires_it() {
        assert_eq!(FuncIri::new("has space").to_string(), r#"url("has space")"#);
        assert_eq!(FuncIri::new("has)paren").to_string(), r#"url("has)paren")"#);
        assert_eq!(
            FuncIri::new("has\"double").to_string(),
            "url('has\"double')"
        );
    }

    #[test]
    fn round_trip_through_display_and_parse_preserves_the_reference() {
        for reference in ["#a", "", "#with-dash", "has space", "has\"double"] {
            let iri = FuncIri::new(reference);
            let text = iri.to_string();
            let reparsed = FuncIri::parse(&text)
                .unwrap_or_else(|e| panic!("re-parsing {text:?} (from {reference:?}) failed: {e}"));
            assert_eq!(reparsed, iri, "round-trip of {reference:?} via {text:?}");
        }
    }

    #[test]
    fn into_reference_moves_the_owned_string_out() {
        let iri = FuncIri::new("#a");
        assert_eq!(iri.into_reference(), Box::<str>::from("#a"));
    }

    // ---- local_reference / effective_href ----

    #[test]
    fn local_reference_strips_a_leading_hash() {
        assert_eq!(local_reference("#foo"), Some("foo"));
        assert_eq!(local_reference("#"), Some(""));
    }

    #[test]
    fn local_reference_is_none_for_non_fragment_hrefs() {
        assert_eq!(local_reference("foo"), None);
        assert_eq!(local_reference(""), None);
        assert_eq!(local_reference("other.svg#frag"), None);
    }

    #[test]
    fn effective_href_prefers_href_over_xlink_href() {
        assert_eq!(effective_href(Some("a"), Some("b")), Some("a"));
        assert_eq!(effective_href(Some("a"), None), Some("a"));
        assert_eq!(effective_href(None, Some("b")), Some("b"));
        assert_eq!(effective_href(None, None), None);
    }
}
