//! A minimal [XPointer](https://www.w3.org/TR/xptr-framework/) evaluator, just deep enough for
//! `xi:include`'s `xpointer` attribute.
//!
//! Full support is:
//!
//! * the **shorthand form** — a bare `NCName`, selecting the element whose `ID`-typed attribute
//!   has that value (see
//!   [`Document::get_element_by_id`](oxixml_dom::Document::get_element_by_id)); and
//! * the **`element()` scheme** of
//!   [XPointer element() Scheme](https://www.w3.org/TR/xptr-element/), both the shorthand-ID
//!   form (`element(chapter1)`) and the child-sequence form, either rooted at an `ID`
//!   (`element(chapter1/1/2)`) or at the document (`element(/1/2)`).
//!
//! On top of that, the `xmlns()` scheme is recognized (and, since this crate does not resolve
//! `QName`s inside `xpointer()` expressions, ignored) and the `xpointer()` scheme understands the
//! single, extremely common expression `xpointer(id('name'))` — an XPath `id()` call selecting
//! an element by `ID`, equivalent to the shorthand pointer `name`. Any other `xpointer()`
//! expression, and any other scheme name entirely, causes that *one pointer part* to fail; per
//! the [XPointer Framework](https://www.w3.org/TR/xptr-framework/#POprocessingptr) processing
//! model this only matters if every part in the whole `xpointer` attribute fails, at which point
//! [`evaluate`] returns `None` and the caller treats the include as a resource error (triggering
//! `xi:fallback`), exactly as an XPointer error the specification does not require full support
//! for.

use oxixml_dom::{Document, NodeId, NodeKind};
use oxixml_qname::is_ncname;

/// Evaluates an `xpointer` attribute value against `doc`, returning the single element it
/// selects, or `None` when no part of the pointer could be evaluated.
///
/// `None` is not a Rust error: per the XPointer Framework, a pointer that names nothing is a
/// *resource error*, which the caller turns into `xi:fallback` processing.
#[must_use]
pub(crate) fn evaluate(doc: &Document, pointer: &str) -> Option<NodeId> {
    let trimmed = pointer.trim();
    if is_ncname(trimmed) {
        return doc.get_element_by_id(trimmed);
    }
    for part in parse_scheme_parts(trimmed)? {
        let found = match part.name {
            "element" => resolve_element_scheme(doc, part.data),
            "xpointer" => resolve_minimal_xpointer_scheme(doc, part.data),
            // `xmlns(prefix=uri)` only ever matters for resolving QNames inside a later
            // `xpointer()` part, which this crate does not implement; skip to the next part.
            "xmlns" => None,
            // An unrecognized scheme is not an error by itself — the Framework has the
            // processor move on to the next pointer part.
            _ => None,
        };
        if found.is_some() {
            return found;
        }
    }
    None
}

/// One `SchemeName(SchemeData)` part of a scheme-based pointer.
struct SchemePart<'a> {
    name: &'a str,
    data: &'a str,
}

/// Splits a scheme-based `xpointer` attribute value into its pointer parts, per the
/// [XPointer Framework](https://www.w3.org/TR/xptr-framework/#syntax) `PointerPart*` grammar:
/// whitespace-separated `SchemeName '(' SchemeData ')'` groups, where `SchemeData` may itself
/// contain balanced, escaped parentheses (`^(`, `^)`, `^^`).
///
/// Returns `None` when the value is not well-formed as a sequence of scheme-based pointer parts
/// at all (for instance, an unbalanced parenthesis) — from the caller's point of view this is
/// just one more way for the whole pointer to fail to select anything.
fn parse_scheme_parts(input: &str) -> Option<Vec<SchemePart<'_>>> {
    let mut parts = Vec::new();
    let mut rest = input;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let open = rest.find('(')?;
        let name = &rest[..open];
        if name.is_empty() || !is_ncname(name) {
            return None;
        }
        let (data, tail) = scan_balanced(&rest[open + 1..])?;
        parts.push(SchemePart { name, data });
        rest = tail;
    }
    if parts.is_empty() { None } else { Some(parts) }
}

/// Scans `text` (which starts right after an already-consumed opening `(`) for the matching
/// closing `)`, treating nested, unescaped parentheses as part of the data and `^(`/`^)`/`^^` as
/// escaped literals. Returns the data slice (escapes left untouched — callers that care about
/// escaped content unescape it themselves) and the remainder of `text` after the closing `)`.
fn scan_balanced(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    let mut depth = 1_usize;
    let mut i = 0_usize;
    let mut escaped = false;
    while i < bytes.len() {
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        match bytes[i] {
            b'^' => escaped = true,
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&text[..i], &text[i + 1..]));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// The parsed data of an `element(...)` scheme part: an optional `ID`, followed by an optional
/// child sequence of 1-based, element-counting-only indices.
struct ElementPointer<'a> {
    id: Option<&'a str>,
    sequence: Vec<usize>,
}

/// Parses `element()` scheme data per
/// [XPointer element() Scheme §3](https://www.w3.org/TR/xptr-element/#scheme):
///
/// ```text
/// SchemeData    ::= NCName ChildSequence? | ChildSequence
/// ChildSequence ::= ('/' [1-9] [0-9]*)+
/// ```
fn parse_element_data(data: &str) -> Option<ElementPointer<'_>> {
    if data.is_empty() {
        return None;
    }
    let (id_part, sequence_part) = match data.find('/') {
        Some(idx) => (&data[..idx], &data[idx..]),
        None => (data, ""),
    };
    let id = match id_part {
        "" => None,
        _ if is_ncname(id_part) => Some(id_part),
        _ => return None,
    };
    let mut sequence = Vec::new();
    for step in sequence_part.split('/').skip(1) {
        if step.is_empty() || (step.len() > 1 && step.starts_with('0')) {
            return None;
        }
        let index: usize = step.parse().ok()?;
        if index == 0 {
            return None;
        }
        sequence.push(index);
    }
    if id.is_none() && sequence.is_empty() {
        return None;
    }
    Some(ElementPointer { id, sequence })
}

/// Resolves one `element(...)` scheme part.
fn resolve_element_scheme(doc: &Document, data: &str) -> Option<NodeId> {
    let pointer = parse_element_data(data)?;
    let mut current = match pointer.id {
        Some(id) => doc.get_element_by_id(id)?,
        None => doc.root(),
    };
    for step in pointer.sequence {
        current = doc
            .children(current)
            .filter(|&child| doc.kind(child) == Some(NodeKind::Element))
            .nth(step - 1)?;
    }
    (doc.kind(current) == Some(NodeKind::Element)).then_some(current)
}

/// Resolves the one `xpointer()` expression form this crate understands:
/// `xpointer(id('name'))` or `xpointer(id("name"))`, an XPath `id()` call naming an `ID`.
fn resolve_minimal_xpointer_scheme(doc: &Document, data: &str) -> Option<NodeId> {
    let trimmed = data.trim();
    let inner = trimmed.strip_prefix("id(")?.strip_suffix(')')?;
    let name = parse_xpath_literal(inner.trim())?;
    doc.get_element_by_id(name)
}

/// Parses an XPath 1.0 `Literal` (`'...'` or `"..."`, no escape sequences), returning the
/// content between the matching quotes.
fn parse_xpath_literal(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let quote = bytes[0];
    if (quote == b'\'' || quote == b'"') && bytes[bytes.len() - 1] == quote {
        Some(&text[1..text.len() - 1])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    fn doc(xml: &str) -> Document {
        build::parse_str(xml).expect("well-formed fixture")
    }

    #[test]
    fn shorthand_pointer_selects_by_id() {
        let d = doc(r#"<r><a xml:id="x"/><b xml:id="y"/></r>"#);
        let target = evaluate(&d, "y").expect("resolved");
        assert_eq!(d.element(target).map(|e| e.local_name()), Some("b"));
    }

    #[test]
    fn element_scheme_shorthand_id_matches_bare_shorthand() {
        let d = doc(r#"<r><a xml:id="x"/></r>"#);
        assert_eq!(evaluate(&d, "element(x)"), evaluate(&d, "x"));
    }

    #[test]
    fn element_scheme_child_sequence_from_root() {
        let d = doc(r#"<r><a/><b><c/><d/></b></r>"#);
        let target = evaluate(&d, "element(/1/2/2)").expect("resolved");
        assert_eq!(d.element(target).map(|e| e.local_name()), Some("d"));
    }

    #[test]
    fn element_scheme_child_sequence_from_id() {
        let d = doc(r#"<r><a xml:id="mid"><x/><y/></a></r>"#);
        let target = evaluate(&d, "element(mid/2)").expect("resolved");
        assert_eq!(d.element(target).map(|e| e.local_name()), Some("y"));
    }

    #[test]
    fn element_scheme_out_of_range_index_fails() {
        let d = doc(r#"<r><a/></r>"#);
        assert!(evaluate(&d, "element(/1/9)").is_none());
    }

    #[test]
    fn element_scheme_rejects_leading_zero_and_zero_index() {
        assert!(parse_element_data("/01").is_none());
        assert!(parse_element_data("/0").is_none());
    }

    #[test]
    fn xmlns_scheme_is_skipped_and_xpointer_id_still_resolves() {
        let d = doc(r#"<r><a xml:id="x"/></r>"#);
        let target = evaluate(&d, "xmlns(x=urn:example) xpointer(id('x'))").expect("resolved");
        assert_eq!(d.element(target).map(|e| e.local_name()), Some("a"));
    }

    #[test]
    fn unrecognized_scheme_is_skipped_in_favor_of_a_later_part() {
        let d = doc(r#"<r><a xml:id="x"/></r>"#);
        let target = evaluate(&d, "unsupported(whatever) element(x)").expect("resolved");
        assert_eq!(d.element(target).map(|e| e.local_name()), Some("a"));
    }

    #[test]
    fn malformed_scheme_syntax_yields_none() {
        let d = doc("<r/>");
        assert!(evaluate(&d, "element(").is_none());
        assert!(evaluate(&d, "()").is_none());
    }

    #[test]
    fn unresolvable_pointer_yields_none() {
        let d = doc("<r/>");
        assert!(evaluate(&d, "missing").is_none());
        assert!(evaluate(&d, "element(missing)").is_none());
    }
}
