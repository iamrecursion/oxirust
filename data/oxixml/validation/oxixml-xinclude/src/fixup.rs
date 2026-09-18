//! Base URI and language fixup for grafted content.
//!
//! [XInclude 1.0 §4.5.5–4.5.6](https://www.w3.org/TR/xinclude/#base_uri_fixup) requires that,
//! once a `xi:include` element is replaced by its
//! [top-level included items](https://www.w3.org/TR/xinclude/#dt-top-level-included-items),
//! each of those top-level items carries an explicit `xml:base` (and, separately, `xml:lang`)
//! attribute whenever its own effective value — the one it had *before* being grafted — differs
//! from the value in effect at its new position (the `xi:include` element's parent, its
//! *include parent*). This is what keeps every relative URI reference, and every
//! language-sensitive default, inside the grafted subtree resolving exactly as it did in the
//! document it was acquired from.
//!
//! Only the *top-level* items need the added attribute: the acquired subtree's own internal
//! `xml:base`/`xml:lang` attributes (including ones this same fixup already added while
//! processing its own nested `xi:include` elements) already give every descendant the correct
//! effective value by ordinary XML Base / `xml:lang` inheritance.

use oxixml_dom::{Document, NodeId, XML_NAMESPACE};
use oxixml_iri::{Iri, IriRef};

use crate::error::XIncludeError;

/// Computes the effective base URI in effect at `node`: `document_base` overridden by every
/// `xml:base` attribute found on `node`'s ancestor-or-self chain, applied from the document root
/// down to `node` itself (an `xml:base` on `node` applies to `node`'s own attributes, including
/// its own `href` when `node` is itself an `xi:include` element).
///
/// # Errors
///
/// Returns [`XIncludeError::InvalidXmlBase`] when an `xml:base` value along the chain is not a
/// valid IRI reference, or does not resolve against the base accumulated so far.
pub(crate) fn effective_base(
    doc: &Document,
    node: NodeId,
    document_base: &Iri<String>,
) -> Result<Iri<String>, XIncludeError> {
    let mut chain: Vec<NodeId> = doc.ancestors(node).collect();
    chain.reverse();
    chain.push(node);

    let mut base = document_base.clone();
    for id in chain {
        let Some(raw) = doc.get_attribute_ns(id, Some(XML_NAMESPACE), "base") else {
            continue;
        };
        let iri_ref = IriRef::<String>::parse(raw.to_owned()).map_err(|source| {
            XIncludeError::InvalidXmlBase {
                value: raw.to_owned(),
                source,
            }
        })?;
        base = base
            .resolve(&iri_ref)
            .map_err(|source| XIncludeError::InvalidXmlBase {
                value: raw.to_owned(),
                source,
            })?;
    }
    Ok(base)
}

/// Computes the effective language in scope at `node`: the normalized value of the nearest
/// `xml:lang` attribute on `node`'s ancestor-or-self chain, or `None` when none is found. An
/// empty `xml:lang=""` explicitly clears the language (per
/// [XML 1.0 §2.12](https://www.w3.org/TR/xml/#sec-lang-tag)), which this represents as `None`
/// too, matching the *language* infoset property XInclude's fixup algorithm is defined over.
pub(crate) fn effective_lang(doc: &Document, node: NodeId) -> Option<String> {
    let mut chain: Vec<NodeId> = doc.ancestors(node).collect();
    chain.reverse();
    chain.push(node);

    let mut lang: Option<String> = None;
    for id in chain {
        if let Some(raw) = doc.get_attribute_ns(id, Some(XML_NAMESPACE), "lang") {
            lang = if raw.is_empty() {
                None
            } else {
                Some(raw.to_owned())
            };
        }
    }
    lang
}

/// Returns `true` when two effective languages differ under
/// [RFC 3066](https://www.rfc-editor.org/rfc/rfc3066) case-insensitive tag comparison — the
/// comparison [XInclude 1.0 §4.5.6](https://www.w3.org/TR/xinclude/#language_fixup) specifies.
pub(crate) fn lang_differs(acquired: Option<&str>, target: Option<&str>) -> bool {
    match (acquired, target) {
        (None, None) => false,
        (Some(a), Some(b)) => !a.eq_ignore_ascii_case(b),
        _ => true,
    }
}

/// Adds (or replaces) the `xml:base` attribute on a freshly grafted top-level item.
pub(crate) fn apply_base_fixup(
    doc: &mut Document,
    node: NodeId,
    base: &Iri<String>,
) -> Result<(), XIncludeError> {
    doc.set_attribute_ns(node, Some(XML_NAMESPACE), "xml:base", base.as_str())?;
    Ok(())
}

/// Adds (or replaces) the `xml:lang` attribute on a freshly grafted top-level item. `lang` is
/// written verbatim, including the empty string for "no language in scope".
pub(crate) fn apply_lang_fixup(
    doc: &mut Document,
    node: NodeId,
    lang: Option<&str>,
) -> Result<(), XIncludeError> {
    doc.set_attribute_ns(node, Some(XML_NAMESPACE), "xml:lang", lang.unwrap_or(""))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    fn base(s: &str) -> Iri<String> {
        Iri::parse(s.to_owned()).expect("valid test base")
    }

    #[test]
    fn effective_base_defaults_to_the_document_base() {
        let d = build::parse_str("<r/>").expect("well-formed");
        let root = d.document_element().expect("root");
        let b = effective_base(&d, root, &base("http://example.com/a/b.xml")).expect("resolved");
        assert_eq!(b.as_str(), "http://example.com/a/b.xml");
    }

    #[test]
    fn effective_base_accumulates_xml_base_down_the_tree() {
        let d = build::parse_str(r#"<r xml:base="sub/"><c xml:base="deep.xml"/></r>"#)
            .expect("well-formed");
        let root = d.document_element().expect("root");
        let child = d.children(root).next().expect("child");
        let root_base = effective_base(&d, root, &base("http://example.com/a/b.xml")).expect("ok");
        assert_eq!(root_base.as_str(), "http://example.com/a/sub/");
        let child_base =
            effective_base(&d, child, &base("http://example.com/a/b.xml")).expect("ok");
        assert_eq!(child_base.as_str(), "http://example.com/a/sub/deep.xml");
    }

    #[test]
    fn effective_lang_is_inherited_and_can_be_cleared() {
        let d = build::parse_str(r#"<r xml:lang="en"><a/><b xml:lang=""/><c xml:lang="fr"/></r>"#)
            .expect("well-formed");
        let root = d.document_element().expect("root");
        let mut children = d.children(root);
        let a = children.next().expect("a");
        let b = children.next().expect("b");
        let c = children.next().expect("c");
        assert_eq!(effective_lang(&d, root).as_deref(), Some("en"));
        assert_eq!(effective_lang(&d, a).as_deref(), Some("en"));
        assert_eq!(effective_lang(&d, b), None);
        assert_eq!(effective_lang(&d, c).as_deref(), Some("fr"));
    }

    #[test]
    fn lang_differs_is_case_insensitive() {
        assert!(!lang_differs(Some("EN"), Some("en")));
        assert!(lang_differs(Some("en"), Some("fr")));
        assert!(!lang_differs(None, None));
        assert!(lang_differs(Some("en"), None));
    }
}
