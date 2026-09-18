//! Shared helpers for the conformance tests.
//!
//! # Why a DTD shim is needed
//!
//! [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel) canonicalizes
//! the tree an XML processor produces *after* it has applied the declarations of
//! the document type declaration: "the attribute value normalization and entity
//! reference resolution MUST be performed in accordance with the behaviors of a
//! validating XML processor. As well, nodes for default attributes … are created
//! in each element." Several of the worked examples in the Recommendations rely
//! on exactly that — `<!ATTLIST e9 attr CDATA "default">` adds an attribute,
//! `<!ATTLIST normNames attr NMTOKENS>` changes how a value is normalized, and
//! `<!ENTITY ent2 SYSTEM "world.txt">` pulls in an external file.
//!
//! `oxixml-dom` is a non-validating builder: it resolves internal entities and
//! normalizes attribute values as CDATA, but it does not read a DTD's attribute
//! declarations and it cannot fetch external entities. Canonicalization is not
//! the layer that should do either. The helpers below therefore stand in for the
//! validating processor, applying by hand the exact declarations each example
//! writes in its internal subset, so that the tree handed to the canonicalizer
//! is the one the Recommendation describes.

#![allow(dead_code)]

use oxixml_dom::{Document, NodeId, NodeKind, XML_NAMESPACE, build};

/// Parses a document, panicking with the parser's message on failure.
pub fn parse(xml: &str) -> Document {
    build::parse_str(xml).expect("the test input is well-formed XML")
}

/// Parses a document whose element nesting may exceed the reader's default
/// depth limit, raising that limit to `depth`.
pub fn parse_nested(xml: &str, depth: usize) -> Document {
    let options = build::BuilderOptions::new().with_max_element_depth(depth);
    build::parse_str_with(xml, &options).expect("the test input is well-formed XML")
}

/// Returns the canonical output as a string, for readable assertion failures.
pub fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("canonical XML is always valid UTF-8")
}

/// Returns every element with the given qualified name, in document order.
pub fn elements_named(document: &Document, name: &str) -> Vec<NodeId> {
    document
        .get_elements_by_tag_name(document.root(), name)
        .collect()
}

/// Returns the single element with the given qualified name.
pub fn element_named(document: &Document, name: &str) -> NodeId {
    let mut found = elements_named(document, name);
    assert_eq!(found.len(), 1, "expected exactly one <{name}> element");
    found.remove(0)
}

/// Applies an `<!ATTLIST element attribute … "value">` default declaration:
/// every matching element that does not specify the attribute gets it.
pub fn apply_attribute_default(
    document: &mut Document,
    element_name: &str,
    attribute_name: &str,
    value: &str,
) {
    let namespace = xml_namespace_of(attribute_name);
    for element in elements_named(document, element_name) {
        if document
            .get_attribute_node(element, attribute_name)
            .is_none()
        {
            document
                .set_attribute_ns(element, namespace, attribute_name, value)
                .expect("the default attribute name and value are valid");
        }
    }
}

/// Applies the tokenized-type attribute value normalization of
/// [XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize): for an attribute
/// that is not of type `CDATA`, leading and trailing `#x20` characters are
/// discarded and runs of `#x20` collapse to one.
///
/// Only spaces count — the tab, carriage return and line feed that survive in
/// the value came from character references and are left alone, which is what
/// makes the `normNames` and `normId` examples interesting.
pub fn apply_tokenized_type(document: &mut Document, element_name: &str, attribute_name: &str) {
    for element in elements_named(document, element_name) {
        let Some(attribute) = document.get_attribute_node(element, attribute_name) else {
            continue;
        };
        let Some(value) = document.attr(attribute).map(|data| data.value().to_owned()) else {
            continue;
        };
        let normalized = value
            .split(' ')
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        document
            .set_node_value(attribute, normalized)
            .expect("a normalized attribute value is still valid");
    }
}

/// Replaces every unexpanded `&name;` reference with the replacement text an
/// external entity would have supplied.
pub fn expand_entity_reference(document: &mut Document, name: &str, replacement: &str) {
    let references: Vec<NodeId> = (0..document.len())
        .map(|index| NodeId::from_index(index as u32))
        .filter(|node| {
            document.kind(*node) == Some(NodeKind::EntityReference)
                && document.entity_reference(*node) == Some(name)
        })
        .collect();
    for reference in references {
        let parent = document
            .parent(reference)
            .expect("an entity reference in the tree has a parent");
        let text = document
            .create_text_node(replacement)
            .expect("the replacement text is valid character data");
        document
            .replace_child(parent, text, reference)
            .expect("an entity reference can be replaced by text");
    }
}

/// The namespace name a qualified attribute name resolves to, for the handful of
/// prefixes the examples use.
fn xml_namespace_of(attribute_name: &str) -> Option<&'static str> {
    match attribute_name.split_once(':') {
        Some(("xml", _)) => Some(XML_NAMESPACE),
        Some((prefix, _)) => panic!("the test helper does not know the prefix `{prefix}`"),
        None => None,
    }
}
