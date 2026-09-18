//! The `style` attribute: a CSS-error-tolerant declaration-list parser.
//!
//! SVG's `style` attribute (specification catalog §3.15) holds a small,
//! self-contained CSS fragment:
//!
//! ```text
//! style-attribute  ::= declaration-list
//! declaration-list ::= wsp* (declaration (";" wsp* declaration)* ";"? )? wsp*
//! declaration      ::= property wsp* ":" wsp* value ("!" wsp* "important")?
//! ```
//!
//! [`parse_style_attribute`] parses this grammar the way real CSS user
//! agents do: per-declaration error recovery. A declaration that does not
//! match `property : value ["!" "important"]` is dropped on its own,
//! without discarding the declarations before or after it, and scanning
//! resumes at the next top-level `;`. [`parse_style_attribute_strict`]
//! shares the same scanner but stops at, and reports, the first malformed
//! declaration instead of skipping it. [`serialize_declarations`] renders a
//! declaration list back into `style`-attribute text.
//!
//! # What "top-level" means here
//!
//! The three punctuation marks the grammar above cares about — `;` between
//! declarations, `:` between a property and its value, and `!` before
//! `important` — only count when they appear at the *top level* of the
//! value: not inside a `/* ... */` comment, not inside a `'...'`/`"..."`
//! quoted string, and not nested inside a parenthesized function argument
//! list such as `url(...)`. That is what lets `background:
//! url(data:image/png;base64,...)` keep its embedded `;` and `:` as part of
//! one declaration's value, and what lets `content: "a;b"` keep its
//! embedded `;` as part of a quoted string. Every offset [`ValueError`]
//! reports is a 0-based byte offset into the original `input` string passed
//! to the parser — comments are stripped from the *returned* declaration
//! text, but never change where an error is reported, because this module
//! locates every structural character by scanning the untouched input
//! directly.
//!
//! # Case handling
//!
//! Property names are case-normalized to ASCII lowercase (matching CSS's
//! own case-insensitive property names): `"FILL: red"` and `"fill: red"`
//! both produce [`Declaration::name`] `"fill"`. Values are never
//! case-folded; `"red"` and `"RED"` remain distinct strings, since a
//! value's case sensitivity depends on the property (colors are
//! case-insensitive, `url()` targets and font names are not), which is a
//! cascade-layer concern, not this module's.
//!
//! # Known simplifications
//!
//! This is a lexical-recovery parser, not a full CSS tokenizer, so a few
//! corners are deliberately simplified:
//!
//! * A quoted string or a parenthesized function argument list that is
//!   never closed consumes the remainder of the attribute value as part of
//!   the declaration it started in, rather than recovering at the next line
//!   break the way the CSS Syntax specification's "bad string" and
//!   unterminated-function recovery rules do. SVG attribute values are
//!   single XML attribute strings (no literal newlines survive XML
//!   attribute-value normalization), so there is no reliable recovery point
//!   to fall back to.
//! * A backslash only escapes the following character inside a quoted
//!   string (so `"a\"b"` is one string, not two). Backslash escapes outside
//!   quotes — a rare corner of CSS identifier syntax — are not recognized.
//! * Neither parser validates that a property name is a property this
//!   crate (or SVG) actually defines, nor that a value is well-formed for
//!   its property. [`crate::style::property::PropertyId`] and the
//!   cascade layer built on top of this module own that; a [`Declaration`]
//!   is exactly what its name says, an unvalidated name/value pair.

use crate::error::ValueError;
use crate::values::lex::is_wsp;

/// One `property: value` declaration from a style attribute or a `<style>`
/// rule body, with an optional trailing `!important` marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    /// The property name, case-normalized to ASCII lowercase (e.g.
    /// `"fill"`, `"stroke-width"`).
    pub name: Box<str>,
    /// The value, exactly as written apart from comment removal and
    /// leading/trailing white space trimming. Case is preserved.
    pub value: Box<str>,
    /// Whether the declaration carried a trailing `!important` marker.
    pub important: bool,
}

impl Declaration {
    /// Creates a declaration from an already-known name, value, and
    /// importance, without parsing or validating any of them.
    ///
    /// This is a plain constructor for callers that already have the three
    /// fields in hand (for instance [`crate::style::css`] building
    /// declarations for a stylesheet rule); [`parse_style_attribute`] is
    /// the entry point for parsing raw `style`-attribute text.
    #[must_use]
    pub fn new(name: impl Into<Box<str>>, value: impl Into<Box<str>>, important: bool) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            important,
        }
    }
}

/// One occurrence of `;`, `:`, or `!` found at the top level of a style
/// value (see the [module documentation](self)).
#[derive(Debug, Clone, Copy)]
struct Marker {
    offset: usize,
    ch: char,
}

/// Scans `text` once and records the byte offset of every top-level `;`,
/// `:`, and `!` — the only characters either parser needs to locate.
/// Offsets are ordered ascending, since they are discovered in a single
/// left-to-right pass.
fn top_level_markers(text: &str) -> Vec<Marker> {
    let mut markers = Vec::new();
    let mut chars = text.char_indices().peekable();
    let mut quote: Option<char> = None;
    let mut in_comment = false;
    let mut paren_depth: u32 = 0;
    while let Some((offset, ch)) = chars.next() {
        if in_comment {
            if ch == '*' && chars.peek().map(|&(_, c)| c) == Some('/') {
                chars.next();
                in_comment = false;
            }
            continue;
        }
        if let Some(quote_char) = quote {
            if ch == '\\' {
                // A backslash escapes the following character; consume it
                // unconditionally so an escaped quote does not end the
                // string early.
                chars.next();
                continue;
            }
            if ch == quote_char {
                quote = None;
            }
            continue;
        }
        if ch == '/' && chars.peek().map(|&(_, c)| c) == Some('*') {
            chars.next();
            in_comment = true;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        if ch == '(' {
            paren_depth += 1;
            continue;
        }
        if ch == ')' {
            paren_depth = paren_depth.saturating_sub(1);
            continue;
        }
        if paren_depth == 0 && matches!(ch, ';' | ':' | '!') {
            markers.push(Marker { offset, ch });
        }
    }
    markers
}

/// Splits `input` into the byte ranges between top-level `;` separators
/// (using markers already found by [`top_level_markers`]), covering the
/// whole input. A trailing `;`, or `input` containing no `;` at all, both
/// fall out of this naturally: the former yields one empty final span, the
/// latter yields one span covering everything.
fn top_level_spans(input: &str, markers: &[Marker]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for marker in markers.iter().filter(|marker| marker.ch == ';') {
        spans.push((start, marker.offset));
        start = marker.offset + 1;
    }
    spans.push((start, input.len()));
    spans
}

/// Trims leading and trailing SVG/CSS white space (space, tab, CR, LF) from
/// `text`, matching [`is_wsp`] rather than Rust's broader Unicode notion of
/// white space.
fn trim_wsp(text: &str) -> &str {
    text.trim_matches(is_wsp)
}

/// Copies `raw` with every `/* ... */` comment removed, leaving quoted
/// string contents untouched — a comment-opening sequence inside a quoted
/// string is left as literal text, matching CSS's own rule that comments do
/// not nest inside strings.
fn strip_comments(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.char_indices().peekable();
    let mut quote: Option<char> = None;
    while let Some((_, ch)) = chars.next() {
        if let Some(quote_char) = quote {
            out.push(ch);
            if ch == '\\' {
                if let Some(&(_, escaped)) = chars.peek() {
                    out.push(escaped);
                    chars.next();
                }
                continue;
            }
            if ch == quote_char {
                quote = None;
            }
            continue;
        }
        if ch == '/' && chars.peek().map(|&(_, c)| c) == Some('*') {
            chars.next(); // consume the '*' that opens the comment
            let mut previous_was_star = false;
            for (_, comment_char) in chars.by_ref() {
                if previous_was_star && comment_char == '/' {
                    break;
                }
                previous_was_star = comment_char == '*';
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
        }
        out.push(ch);
    }
    out
}

/// The outcome of attempting to parse one top-level, semicolon-delimited
/// span of a style value as a single declaration.
enum SpanOutcome {
    /// The span was empty, or only white space and comments: the standard
    /// CSS tolerance for stray or doubled `;` separators. This contributes
    /// no declaration and is never an error, in either parser.
    Empty,
    /// The span parsed as a well-formed declaration.
    Declaration(Declaration),
    /// The span has real content but does not match
    /// `property wsp* ":" wsp* value ("!" wsp* "important")?`.
    Malformed(ValueError),
}

/// Parses the input byte range `span_start..span_end` of `input` — one
/// top-level `;`-delimited span — as a single declaration, using
/// `markers` (already computed over the whole `input`) to find the
/// top-level `:` and `!` inside it.
fn parse_span(input: &str, span_start: usize, span_end: usize, markers: &[Marker]) -> SpanOutcome {
    let raw = &input[span_start..span_end];
    if trim_wsp(&strip_comments(raw)).is_empty() {
        return SpanOutcome::Empty;
    }

    let Some(colon) = markers
        .iter()
        .find(|marker| marker.ch == ':' && marker.offset >= span_start && marker.offset < span_end)
    else {
        return SpanOutcome::Malformed(ValueError::UnexpectedEnd { offset: span_end });
    };

    let name = trim_wsp(&strip_comments(&input[span_start..colon.offset])).to_ascii_lowercase();
    if name.is_empty() {
        return SpanOutcome::Malformed(ValueError::OutOfRange {
            offset: colon.offset,
            what: "declaration has an empty property name",
        });
    }

    let value_start = colon.offset + 1;
    let bang = markers.iter().find(|marker| {
        marker.ch == '!' && marker.offset >= value_start && marker.offset < span_end
    });
    let (value_end, important) = match bang {
        Some(bang_marker) => {
            let after_bang = strip_comments(&input[bang_marker.offset + 1..span_end]);
            let after_bang = trim_wsp(&after_bang);
            if !after_bang.eq_ignore_ascii_case("important") {
                let keyword = after_bang.split_whitespace().next().unwrap_or("");
                return SpanOutcome::Malformed(ValueError::UnknownKeyword {
                    offset: bang_marker.offset + 1,
                    keyword: keyword.into(),
                });
            }
            (bang_marker.offset, true)
        }
        None => (span_end, false),
    };

    let value = trim_wsp(&strip_comments(&input[value_start..value_end])).to_string();
    if value.is_empty() {
        return SpanOutcome::Malformed(ValueError::OutOfRange {
            offset: value_start,
            what: "declaration has an empty value",
        });
    }

    SpanOutcome::Declaration(Declaration::new(name, value, important))
}

/// Parses a `style` attribute value into its declarations, tolerating CSS
/// syntax errors the way a browser does: a declaration that does not match
/// `property wsp* ":" wsp* value ("!" wsp* "important")?` is dropped, and
/// scanning resumes at the next top-level `;`. Declarations before and
/// after the malformed one are unaffected. An empty (or white-space- and
/// comment-only) value yields an empty vector, never an error.
///
/// See the [module documentation](self) for exactly what "top-level" means
/// for the `;`/`:`/`!` this grammar depends on, and for the case-folding
/// rule applied to property names.
///
/// ```
/// use oxixml_svg::style::attr::parse_style_attribute;
///
/// let declarations = parse_style_attribute("fill: red; STROKE-WIDTH : 2 !important ;;");
/// assert_eq!(declarations.len(), 2);
/// assert_eq!(&*declarations[0].name, "fill");
/// assert_eq!(&*declarations[0].value, "red");
/// assert!(!declarations[0].important);
/// assert_eq!(&*declarations[1].name, "stroke-width");
/// assert!(declarations[1].important);
/// ```
#[must_use]
pub fn parse_style_attribute(input: &str) -> Vec<Declaration> {
    let markers = top_level_markers(input);
    let spans = top_level_spans(input, &markers);
    let mut declarations = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        if let SpanOutcome::Declaration(declaration) = parse_span(input, start, end, &markers) {
            declarations.push(declaration);
        }
    }
    declarations
}

/// Parses a `style` attribute value the same way [`parse_style_attribute`]
/// does, except that the first declaration which fails to parse stops the
/// whole parse and is reported as a [`ValueError`] carrying a byte offset
/// into `input`, instead of being skipped.
///
/// An empty (or white-space- and comment-only) value is not an error here
/// either — the grammar's declaration list is optional — so this returns
/// `Ok(Vec::new())` for it, matching [`parse_style_attribute`].
///
/// # Errors
///
/// Returns the [`ValueError`] for the first declaration that is missing a
/// top-level `:`, has an empty property name, has an empty value, or has a
/// malformed `!`-suffix that is not (case-insensitively) `important`.
pub fn parse_style_attribute_strict(input: &str) -> Result<Vec<Declaration>, ValueError> {
    let markers = top_level_markers(input);
    let spans = top_level_spans(input, &markers);
    let mut declarations = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match parse_span(input, start, end, &markers) {
            SpanOutcome::Declaration(declaration) => declarations.push(declaration),
            SpanOutcome::Empty => {}
            SpanOutcome::Malformed(error) => return Err(error),
        }
    }
    Ok(declarations)
}

/// Renders a declaration list back into `style`-attribute text, one
/// `name: value` pair per declaration (`!important` appended when set),
/// separated by `"; "`.
///
/// This is not guaranteed to reproduce the exact text a declaration list
/// was parsed from (original white space and comments are not preserved),
/// but re-parsing its output with [`parse_style_attribute`] always yields
/// declarations equal to the ones serialized.
///
/// ```
/// use oxixml_svg::style::attr::{Declaration, serialize_declarations};
///
/// let declarations = vec![
///     Declaration::new("fill", "red", false),
///     Declaration::new("stroke", "blue", true),
/// ];
/// assert_eq!(serialize_declarations(&declarations), "fill: red; stroke: blue !important");
/// ```
#[must_use]
pub fn serialize_declarations(declarations: &[Declaration]) -> String {
    let mut out = String::new();
    for (index, declaration) in declarations.iter().enumerate() {
        if index > 0 {
            out.push_str("; ");
        }
        out.push_str(&declaration.name);
        out.push_str(": ");
        out.push_str(&declaration.value);
        if declaration.important {
            out.push_str(" !important");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_simple_declaration_list() {
        let declarations = parse_style_attribute("fill:red; stroke-width:2");
        assert_eq!(declarations.len(), 2);
        assert_eq!(&*declarations[0].name, "fill");
        assert_eq!(&*declarations[0].value, "red");
        assert!(!declarations[0].important);
        assert_eq!(&*declarations[1].name, "stroke-width");
        assert_eq!(&*declarations[1].value, "2");
        assert!(!declarations[1].important);
    }

    #[test]
    fn trailing_semicolon_and_surrounding_white_space_are_tolerated() {
        let declarations = parse_style_attribute("  fill : red ;  ");
        assert_eq!(declarations.len(), 1);
        assert_eq!(&*declarations[0].name, "fill");
        assert_eq!(&*declarations[0].value, "red");
    }

    #[test]
    fn empty_declarations_between_semicolons_are_ignored() {
        let declarations = parse_style_attribute("fill:red;;;stroke:blue");
        assert_eq!(declarations.len(), 2);
        assert_eq!(&*declarations[0].name, "fill");
        assert_eq!(&*declarations[1].name, "stroke");
    }

    #[test]
    fn empty_or_blank_input_yields_no_declarations_and_is_not_an_error() {
        assert_eq!(parse_style_attribute(""), Vec::new());
        assert_eq!(parse_style_attribute("   "), Vec::new());
        assert_eq!(parse_style_attribute("/* just a comment */"), Vec::new());
        assert_eq!(
            parse_style_attribute_strict("").expect("blank value is not an error"),
            Vec::new()
        );
        assert_eq!(
            parse_style_attribute_strict("  ;;  ").expect("blank declarations are not an error"),
            Vec::new()
        );
    }

    #[test]
    fn property_names_are_lowercased_but_values_are_not() {
        let declarations = parse_style_attribute("FILL:RED");
        assert_eq!(declarations.len(), 1);
        assert_eq!(&*declarations[0].name, "fill");
        assert_eq!(
            &*declarations[0].value, "RED",
            "value case must be preserved"
        );
    }

    #[test]
    fn comments_are_stripped_between_and_within_declarations() {
        let declarations =
            parse_style_attribute("fill/*name*/: /* value */ red; /* skip */ stroke:blue");
        assert_eq!(declarations.len(), 2);
        assert_eq!(&*declarations[0].name, "fill");
        assert_eq!(&*declarations[0].value, "red");
        assert_eq!(&*declarations[1].name, "stroke");
        assert_eq!(&*declarations[1].value, "blue");
    }

    #[test]
    fn semicolon_inside_a_quoted_string_does_not_split_the_declaration() {
        let declarations = parse_style_attribute(r#"content: "a;b"; color: red"#);
        assert_eq!(declarations.len(), 2);
        assert_eq!(&*declarations[0].name, "content");
        assert_eq!(&*declarations[0].value, r#""a;b""#);
        assert_eq!(&*declarations[1].name, "color");
        assert_eq!(&*declarations[1].value, "red");
    }

    #[test]
    fn quoted_semicolon_survives_a_backslash_escaped_quote() {
        // The raw text is: content: "a\"b"; color:red — the backslash
        // escapes the quote, so the string only ends at the *second*
        // unescaped quote, and the `;` right after it is the real
        // top-level separator.
        let quoted_value = format!("{q}a{esc}{q}b{q}", q = '"', esc = '\\');
        let input = format!("content: {quoted_value}; color:red");
        let declarations = parse_style_attribute(&input);
        assert_eq!(declarations.len(), 2, "input was {input:?}");
        assert_eq!(&*declarations[0].name, "content");
        assert_eq!(&*declarations[0].value, quoted_value);
        assert_eq!(&*declarations[1].name, "color");
        assert_eq!(&*declarations[1].value, "red");
    }

    #[test]
    fn semicolon_and_colon_inside_parentheses_do_not_split_or_separate() {
        let declarations =
            parse_style_attribute("background: url(data:image/png;base64,iVBORw==); color: red");
        assert_eq!(declarations.len(), 2);
        assert_eq!(&*declarations[0].name, "background");
        assert_eq!(
            &*declarations[0].value,
            "url(data:image/png;base64,iVBORw==)"
        );
        assert_eq!(&*declarations[1].name, "color");
        assert_eq!(&*declarations[1].value, "red");
    }

    #[test]
    fn important_is_recognized_with_and_without_surrounding_white_space_and_any_case() {
        for text in [
            "fill:red!important",
            "fill:red !important",
            "fill:red  !important",
            "fill:red !IMPORTANT",
            "fill:red !ImPoRtAnT",
            "fill:red !/*c*/important",
        ] {
            let declarations = parse_style_attribute(text);
            assert_eq!(declarations.len(), 1, "input was {text:?}");
            assert_eq!(&*declarations[0].name, "fill");
            assert_eq!(&*declarations[0].value, "red", "input was {text:?}");
            assert!(declarations[0].important, "input was {text:?}");
        }
    }

    #[test]
    fn a_declaration_without_a_colon_is_dropped_in_the_tolerant_parser() {
        let declarations = parse_style_attribute("fill red; stroke:blue");
        assert_eq!(declarations.len(), 1);
        assert_eq!(&*declarations[0].name, "stroke");
    }

    #[test]
    fn strict_mode_reports_a_missing_colon_at_the_end_of_the_declaration() {
        let error = parse_style_attribute_strict("fill red; stroke:blue")
            .expect_err("missing colon is malformed");
        assert_eq!(error, ValueError::UnexpectedEnd { offset: 8 });
    }

    #[test]
    fn strict_mode_reports_an_empty_property_name_at_the_colon() {
        let error = parse_style_attribute_strict(": red").expect_err("empty name is malformed");
        assert_eq!(
            error,
            ValueError::OutOfRange {
                offset: 0,
                what: "declaration has an empty property name",
            }
        );
    }

    #[test]
    fn strict_mode_reports_an_empty_value_right_after_the_colon() {
        let error = parse_style_attribute_strict("fill: ;").expect_err("empty value is malformed");
        assert_eq!(
            error,
            ValueError::OutOfRange {
                offset: 5,
                what: "declaration has an empty value",
            }
        );
    }

    #[test]
    fn strict_mode_reports_an_unknown_bang_keyword_right_after_the_bang() {
        let error = parse_style_attribute_strict("fill:red !imprint")
            .expect_err("bad bang keyword is malformed");
        assert_eq!(
            error,
            ValueError::UnknownKeyword {
                offset: 10,
                keyword: "imprint".into(),
            }
        );
    }

    #[test]
    fn strict_mode_stops_at_the_first_malformed_declaration_only() {
        // The first declaration is fine, the second is missing its colon,
        // and a would-be third declaration is never reached.
        let error = parse_style_attribute_strict("fill:red; oops; stroke:blue")
            .expect_err("second declaration is malformed");
        assert!(matches!(error, ValueError::UnexpectedEnd { .. }));
    }

    #[test]
    fn tolerant_parser_skips_every_malformed_declaration_and_keeps_the_rest() {
        let declarations =
            parse_style_attribute("fill:red; oops; stroke:blue; also bad; opacity:0.5");
        let names: Vec<&str> = declarations.iter().map(|d| &*d.name).collect();
        assert_eq!(names, vec!["fill", "stroke", "opacity"]);
    }

    #[test]
    fn a_realistic_mixed_declaration_list_parses_completely() {
        let declarations = parse_style_attribute(
            "fill:red; /* comment */ stroke : blue !important ;; opacity:0.5",
        );
        assert_eq!(declarations.len(), 3);
        assert_eq!(&*declarations[0].name, "fill");
        assert_eq!(&*declarations[0].value, "red");
        assert_eq!(&*declarations[1].name, "stroke");
        assert_eq!(&*declarations[1].value, "blue");
        assert!(declarations[1].important);
        assert_eq!(&*declarations[2].name, "opacity");
        assert_eq!(&*declarations[2].value, "0.5");
    }

    #[test]
    fn serialize_declarations_matches_the_documented_format() {
        let declarations = vec![
            Declaration::new("fill", "red", false),
            Declaration::new("stroke", "blue", true),
        ];
        assert_eq!(
            serialize_declarations(&declarations),
            "fill: red; stroke: blue !important"
        );
    }

    #[test]
    fn serialize_then_parse_round_trips_to_equal_declarations() {
        let original = vec![
            Declaration::new("fill", "red", false),
            Declaration::new("stroke-width", "2", false),
            Declaration::new("opacity", "0.5", true),
        ];
        let reparsed = parse_style_attribute(&serialize_declarations(&original));
        assert_eq!(reparsed, original);
    }

    #[test]
    fn serialize_declarations_of_an_empty_list_is_an_empty_string() {
        assert_eq!(serialize_declarations(&[]), "");
    }

    #[test]
    fn declaration_new_stores_its_fields_as_given() {
        let declaration = Declaration::new("fill", "red", true);
        assert_eq!(&*declaration.name, "fill");
        assert_eq!(&*declaration.value, "red");
        assert!(declaration.important);
    }

    #[test]
    fn unterminated_quotes_comments_and_parentheses_do_not_panic() {
        // These are all malformed in one way or another; the only
        // requirement here is that scanning them terminates cleanly.
        for text in [
            "fill: 'unterminated",
            "fill: \"unterminated",
            "fill: /* unterminated comment",
            "fill: url(unterminated",
            "fill",
            ":",
            ";",
            "!",
            "!!!",
            ";;;;;;",
        ] {
            let _ = parse_style_attribute(text);
            let _ = parse_style_attribute_strict(text);
        }
    }
}
