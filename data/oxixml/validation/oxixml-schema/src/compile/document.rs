//! The schema-representation rules that are stated over a *whole* schema
//! document rather than over one element of the vocabulary.
//!
//! [`crate::represent`] checks each element against the schema for schemas —
//! its attributes, their types and the order of its children. Three rules do
//! not fit that shape and are collected here instead, because each of them
//! needs either the document element's own attributes or a walk that knows
//! which subtrees are exempt:
//!
//! * a *namespace name* is never the empty string, so neither
//!   `targetNamespace=""` nor `<xs:import namespace="">` names anything
//!   ([§3.15.2][schema], `msData/schema/schZ014_a`, `schZ014_b`);
//! * every element of the vocabulary has element-only content, so character
//!   data is allowed only inside `xs:documentation` and `xs:appinfo`, whose
//!   content the Recommendation leaves open (`msData/notations/notatG001`,
//!   `notatG003` — and `notatG005`, whose notation contains only whitespace
//!   and is valid);
//! * the `xml:lang` of an `xs:documentation` is an `xs:language`
//!   (`msData/annotations/annotF001`, `annotF003`).
//!
//! A fourth rule belongs to the notation declaration itself and is checked
//! here for the same reason — it is about the *absence* of attributes, which
//! the per-element table cannot express: XSD 1.0 Errata E1-16 made `public`
//! optional, leaving the requirement that a notation carry at least one of
//! `public` and `system` (`msData/errata10/errC004` is the notation with only
//! a `system`, valid; `msData/notations/notatB001`, `notatG006` and
//! `sunData/Notation/publicId/publicId00201m1` have neither and are not).
//!
//! [schema]: https://www.w3.org/TR/xmlschema-1/#Schemas

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::error::SchemaError;
use crate::names::XML_NS;
use crate::parse::{attr, child_elements, node_path, xsd_name};
use crate::value::collapse;

/// Checks the document-wide representation rules of one schema document.
///
/// # Errors
///
/// Returns the first `src-…` violation found, in document order.
pub(crate) fn check(doc: &Document, root: NodeId) -> Result<(), SchemaError> {
    check_namespace_names(doc, root)?;
    check_content(doc, root)
}

/// A namespace name is an absolute URI reference, and never the empty string:
/// writing `targetNamespace=""` or `<xs:import namespace="">` says "the
/// namespace whose name is nothing", which no element or attribute can ever be
/// in. XML Schema follows [Namespaces in XML][ns] here — an empty `xmlns`
/// *undeclares* the default namespace rather than naming one — so the two
/// attributes are in error rather than meaning "no namespace".
///
/// [ns]: https://www.w3.org/TR/xml-names/#ns-decl
fn check_namespace_names(doc: &Document, root: NodeId) -> Result<(), SchemaError> {
    if let Some(text) = attr(doc, root, "targetNamespace")
        && collapse(text).is_empty()
    {
        return Err(SchemaError::new(
            "src-schema",
            "`targetNamespace` may not be the empty string, which is not a namespace name",
        )
        .at(node_path(doc, root)));
    }
    for child in child_elements(doc, root) {
        if xsd_name(doc, child) != Some("import") {
            continue;
        }
        if let Some(text) = attr(doc, child, "namespace")
            && collapse(text).is_empty()
        {
            return Err(SchemaError::new(
                "src-import",
                "the `namespace` of an `xs:import` may not be the empty string, which is not a \
                 namespace name",
            )
            .at(node_path(doc, child)));
        }
    }
    Ok(())
}

/// Walks the element tree, checking the rules that apply to every element of
/// the vocabulary.
///
/// The walk uses an explicit stack: the document's nesting has already been
/// bounded by [`SchemaSet::MAX_DEPTH`](crate::SchemaSet::MAX_DEPTH), but there
/// is no reason for a second recursion when one loop does.
fn check_content(doc: &Document, root: NodeId) -> Result<(), SchemaError> {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let Some(name) = xsd_name(doc, node) else {
            continue;
        };
        // `xs:appinfo` and `xs:documentation` carry arbitrary content — that
        // is what they are for — so neither they nor anything below them is
        // checked. Their own `xml:lang` still is.
        if matches!(name, "appinfo" | "documentation") {
            check_language(doc, node)?;
            continue;
        }
        if name == "notation" {
            check_notation(doc, node)?;
        }
        for child in doc.children(node) {
            match doc.kind(child) {
                Some(NodeKind::Element) => stack.push(child),
                Some(NodeKind::Text | NodeKind::CdataSection) => {
                    let blank = doc
                        .character_data(child)
                        .is_none_or(|data| data.trim().is_empty());
                    if !blank {
                        return Err(SchemaError::new(
                            "src-element",
                            format!(
                                "`xs:{name}` has element-only content, so the character data \
                                     inside it is not allowed"
                            ),
                        )
                        .at(node_path(doc, node)));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// XSD 1.0 Errata E1-16 and XSD 1.1 §3.14.1: `public` is optional, but a
/// notation declaration that names neither a public nor a system identifier
/// identifies no notation at all.
fn check_notation(doc: &Document, node: NodeId) -> Result<(), SchemaError> {
    if attr(doc, node, "public").is_none() && attr(doc, node, "system").is_none() {
        return Err(SchemaError::new(
            "src-notation",
            "an `xs:notation` needs a `public` or a `system` identifier",
        )
        .at(node_path(doc, node)));
    }
    Ok(())
}

/// The schema for schemas gives `xs:documentation` and `xs:appinfo` an
/// `xml:lang` attribute of type `xs:language`, whose lexical space is
/// `[a-zA-Z]{1,8}(-[a-zA-Z0-9]{1,8})*` — so neither the empty string nor a
/// blank one is a language.
fn check_language(doc: &Document, node: NodeId) -> Result<(), SchemaError> {
    let Some(raw) = doc.get_attribute_ns(node, Some(XML_NS), "lang") else {
        return Ok(());
    };
    // `xs:language` derives from `xs:token`, so the value is collapsed first.
    if is_language(&collapse(raw)) {
        return Ok(());
    }
    Err(SchemaError::new(
        "src-attribute-value",
        format!("`{raw}` is not an `xs:language` value"),
    )
    .at(node_path(doc, node)))
}

/// Whether `value` matches the lexical space of `xs:language`.
fn is_language(value: &str) -> bool {
    let mut parts = value.split('-');
    let Some(first) = parts.next() else {
        return false;
    };
    if first.is_empty() || first.len() > 8 || !first.bytes().all(|byte| byte.is_ascii_alphabetic())
    {
        return false;
    }
    parts.all(|part| {
        !part.is_empty() && part.len() <= 8 && part.bytes().all(|byte| byte.is_ascii_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_lexical_space() {
        assert!(is_language("en"));
        assert!(is_language("en-GB"));
        assert!(is_language("x-klingon-1"));
        assert!(!is_language(""));
        assert!(!is_language("en-"));
        assert!(!is_language("-en"));
        assert!(!is_language("abcdefghi"));
        assert!(!is_language("1234"));
    }
}
