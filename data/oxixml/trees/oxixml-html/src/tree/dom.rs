//! Making HTML content representable in an XML DOM.
//!
//! HTML permits names and character data that XML forbids; [`oxixml_dom`]
//! enforces the XML rules. The functions here rewrite the few constructs that
//! would otherwise be rejected, and return [`Cow::Borrowed`] when nothing had
//! to change so that the caller can tell whether to record a parse error. See
//! the crate-level documentation for the full list of rewrites.

use std::borrow::Cow;

use oxixml_dom::is_xml_char;
use oxixml_qname::{is_name, is_ncname, is_qname};

/// Returns `true` when `ch` may start an XML `NCName`.
fn is_name_start(ch: char) -> bool {
    let mut buffer = [0_u8; 4];
    is_ncname(ch.encode_utf8(&mut buffer))
}

/// Returns `true` when `ch` may continue an XML `NCName`.
fn is_name_continue(ch: char) -> bool {
    let mut buffer = [0_u8; 4];
    let mut probe = String::with_capacity(5);
    probe.push('a');
    probe.push_str(ch.encode_utf8(&mut buffer));
    is_ncname(&probe)
}

/// Rewrites a name so that it matches the XML `NCName` production.
fn sanitize(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 1);
    for ch in name.chars() {
        if out.is_empty() {
            if is_name_start(ch) {
                out.push(ch);
            } else if is_name_continue(ch) {
                out.push('_');
                out.push(ch);
            } else {
                out.push('_');
            }
        } else if is_name_continue(ch) {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}

/// Makes an element name usable with `Document::create_element_ns`.
///
/// A qualified name is left alone, except that the two prefixes XML reserves
/// (`xml` and `xmlns`) are folded into the local name, because the DOM refuses
/// to bind them to the HTML, MathML or SVG namespace.
pub(crate) fn element_name(name: &str) -> Cow<'_, str> {
    if is_qname(name) && !name.starts_with("xml:") && !name.starts_with("xmlns:") {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(sanitize(name))
    }
}

/// Makes an attribute name usable with `Document::set_attribute`, which accepts
/// any XML `Name` (colons included).
pub(crate) fn attribute_name(name: &str) -> Cow<'_, str> {
    if is_name(name) {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(sanitize(name))
    }
}

/// Makes a DOCTYPE name usable with `Document::create_document_type`, which
/// requires an XML `Name`. The nameless `<!DOCTYPE>` of HTML becomes `_`.
pub(crate) fn doctype_name(name: &str) -> Cow<'_, str> {
    if !name.is_empty() && is_name(name) {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(sanitize(name))
    }
}

/// Drops the characters that the XML `Char` production forbids.
pub(crate) fn character_data(text: &str) -> Cow<'_, str> {
    if text.chars().all(is_xml_char) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(text.chars().filter(|&ch| is_xml_char(ch)).collect())
    }
}

/// Makes comment data usable with `Document::create_comment`: XML comments may
/// contain neither `--` nor a trailing `-`.
pub(crate) fn comment_data(text: &str) -> Cow<'_, str> {
    let clean = character_data(text);
    if !clean.contains("--") && !clean.ends_with('-') {
        return clean;
    }
    let mut out = String::with_capacity(clean.len() + 2);
    let mut previous_was_dash = false;
    for ch in clean.chars() {
        if ch == '-' && previous_was_dash {
            out.push(' ');
        }
        previous_was_dash = ch == '-';
        out.push(ch);
    }
    if out.ends_with('-') {
        out.push(' ');
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::{attribute_name, character_data, comment_data, element_name};

    #[test]
    fn valid_names_are_borrowed_unchanged() {
        assert_eq!(element_name("div"), "div");
        assert_eq!(element_name("a:b"), "a:b");
        assert_eq!(attribute_name("xml:lang"), "xml:lang");
        assert_eq!(attribute_name("data-x"), "data-x");
    }

    #[test]
    fn invalid_names_are_rewritten() {
        assert_eq!(element_name("a\"b"), "a_b");
        assert_eq!(element_name("xml:a"), "xml_a");
        assert_eq!(element_name("a:b:c"), "a_b_c");
        assert_eq!(attribute_name("0"), "_0");
        assert_eq!(attribute_name("\""), "_");
    }

    #[test]
    fn doctype_names_are_made_representable() {
        assert_eq!(super::doctype_name("html"), "html");
        assert_eq!(super::doctype_name(""), "_");
        assert_eq!(super::doctype_name("..."), "_...");
    }

    #[test]
    fn character_data_drops_forbidden_characters() {
        assert_eq!(character_data("ok"), "ok");
        assert_eq!(character_data("a\u{1}b"), "ab");
        assert_eq!(character_data("tab\there"), "tab\there");
    }

    #[test]
    fn comments_lose_double_hyphens() {
        assert_eq!(comment_data("plain"), "plain");
        assert_eq!(comment_data("a--b"), "a- -b");
        assert_eq!(comment_data("a---b"), "a- - -b");
        assert_eq!(comment_data("trailing-"), "trailing- ");
    }
}
