//! Parsing the **XML syntax** of a RELAX NG schema into the
//! [`Element`](crate::tree::Element) data model, applying rules
//! [4.1 (annotations)](https://relaxng.org/spec-20011203.html#IDAJZR) and
//! [4.2 (whitespace)](https://relaxng.org/spec-20011203.html#IDAOZR) on the way.
//!
//! Those two rules are done during construction rather than as separate passes
//! because they are exactly the difference between "the XML document" and "the
//! RELAX NG data model": foreign elements and attributes disappear, comments and
//! processing instructions are not part of the data model at all (so the
//! character data around them coalesces into a single string), and whitespace-only
//! strings are dropped everywhere except inside `value` and `param`.

use std::rc::Rc;

use oxixml_dom::{Document, NodeId, NodeKind, XMLNS_NAMESPACE};

use crate::error::{Position, SchemaError, grammar, syntax};
use crate::tree::{Attribute, Child, Context, Element, NamespaceMap, RNG_NS, trim_whitespace};

/// The XML namespace name, used to spot `xml:base`.
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";

/// The greatest depth of nested elements [`convert`] will follow before reporting a
/// [`SchemaError`] instead of exhausting the native call stack.
///
/// The XML reader underlying [`parse_str`] already bounds raw element nesting on its own — but
/// at a considerably larger depth chosen for plain DOM trees generally, and `convert` pays
/// several stack frames of its own per level (a cloned namespace map, an attribute list, an
/// owned name) for every level of RELAX NG pattern nesting, which costs measurably more native
/// stack per level than the reader's own tree walk. A schema nested deep enough to still pass
/// the reader's cap (thousands of `zeroOrMore` elements each wrapping the next, say) would
/// otherwise still exhaust the stack here, well short of the reader's own limit. This mirrors
/// [`crate::compact::parse`]'s `MAX_PARSE_NESTING_DEPTH`, chosen for the same reason and given
/// the same value so that neither syntax is more exploitable than the other.
const MAX_ELEMENT_DEPTH: usize = 64;

/// Parses a well-formed XML document into the RELAX NG data model.
///
/// `base` is the base URI of the document itself; `xml:base` attributes refine it
/// as the tree is walked.
pub(crate) fn parse_document(
    document: &Document,
    base: Option<&str>,
) -> Result<Element, SchemaError> {
    let root = document
        .document_element()
        .ok_or_else(|| syntax("the schema document has no document element"))?;
    let context = Context::new(
        base.map(Rc::from),
        Rc::new(document_namespace_map(document, root)),
    );
    convert(document, root, &context, 0)?.ok_or_else(|| {
        grammar("the document element of a schema must be in the RELAX NG namespace")
    })
}

/// Parses a schema held in a string.
pub(crate) fn parse_str(text: &str, base: Option<&str>) -> Result<Element, SchemaError> {
    let options = oxixml_dom::BuilderOptions::new()
        .with_comments(false)
        .with_processing_instructions(false)
        .with_cdata_as_text(true)
        .with_well_formedness_checks(true);
    let document = oxixml_dom::build::parse_str_with(text, &options)
        .map_err(|error| syntax(format!("the schema is not well-formed XML: {error}")))?;
    parse_document(&document, base)
}

/// Builds the namespace map in force at `element`, walking up from the root.
fn document_namespace_map(document: &Document, element: NodeId) -> NamespaceMap {
    let mut chain = vec![element];
    chain.extend(document.ancestors(element));
    let mut map = NamespaceMap::initial();
    for node in chain.into_iter().rev() {
        if document.kind(node) != Some(NodeKind::Element) {
            continue;
        }
        apply_declarations(document, node, &mut map);
    }
    map
}

fn apply_declarations(document: &Document, element: NodeId, map: &mut NamespaceMap) {
    for (prefix, namespace) in document.namespace_declarations(element) {
        match prefix {
            Some(prefix) => map.bind(prefix, namespace),
            None => map.set_default(namespace),
        }
    }
}

/// Converts one DOM element (and its subtree) into the data model, or returns
/// `Ok(None)` when the element is foreign and rule 4.1 removes it.
///
/// `depth` is how many elements this call is nested inside, checked against
/// [`MAX_ELEMENT_DEPTH`] before doing anything else, so that a schema nested deep enough to
/// exhaust the native call stack is rejected with a clean [`SchemaError`] before it can.
fn convert(
    document: &Document,
    node: NodeId,
    parent_context: &Context,
    depth: usize,
) -> Result<Option<Element>, SchemaError> {
    if depth > MAX_ELEMENT_DEPTH {
        return Err(syntax(format!(
            "the schema nests elements more than {MAX_ELEMENT_DEPTH} levels deep"
        )));
    }
    let Some(data) = document.element(node) else {
        return Ok(None);
    };
    if data.namespace_uri() != Some(RNG_NS) {
        return Ok(None);
    }

    let mut namespaces = (*parent_context.namespaces).clone();
    apply_declarations(document, node, &mut namespaces);
    let mut base = parent_context.base.clone();

    let mut attributes = Vec::new();
    for attribute in document.attributes(node) {
        let Some(attribute) = document.attr(attribute) else {
            continue;
        };
        let namespace = attribute.namespace_uri();
        if namespace == Some(XMLNS_NAMESPACE) || attribute.qualified_name() == "xmlns" {
            continue;
        }
        if namespace == Some(XML_NS) && attribute.local_name() == "base" {
            base = Some(resolve_base(base.as_deref(), attribute.value()));
            continue;
        }
        match namespace {
            // Rule 4.1: foreign attributes are removed.
            Some(namespace) if namespace != RNG_NS => continue,
            Some(_) => attributes.push(Attribute {
                namespace: RNG_NS.to_owned(),
                local: attribute.local_name().to_owned(),
                value: attribute.value().to_owned(),
            }),
            None => attributes.push(Attribute::new(
                attribute.local_name(),
                attribute.value().to_owned(),
            )),
        }
    }

    let name = data.local_name().to_owned();
    // Rule 4.2: leading and trailing whitespace is removed from the value of each
    // `name`, `type` and `combine` attribute.
    for attribute in &mut attributes {
        if attribute.namespace.is_empty()
            && matches!(attribute.local.as_str(), "name" | "type" | "combine")
        {
            attribute.value = trim_whitespace(&attribute.value).to_owned();
        }
    }

    let context = Context::new(base, Rc::new(namespaces));
    let mut element = Element::new(name, context.clone(), Position::UNKNOWN);
    element.attributes = attributes;

    // Comments and processing instructions are not part of the data model, so the
    // character data around them forms a single string.
    let text_only = matches!(element.name.as_str(), "value" | "param" | "name");
    let keeps_whitespace = matches!(element.name.as_str(), "value" | "param");
    let mut pending = String::new();
    for child in document.children(node) {
        match document.kind(child) {
            Some(NodeKind::Text | NodeKind::CdataSection) => {
                if let Some(text) = document.character_data(child) {
                    pending.push_str(text);
                }
            }
            Some(NodeKind::Element) => {
                // §3: only elements that cannot have string children may have
                // foreign child elements, so `value`, `param` and `name` may not
                // have any child element at all.
                if text_only {
                    return Err(grammar(format!(
                        "the `{}` element must not have child elements",
                        element.name
                    )));
                }
                flush_text(&mut element, &mut pending, keeps_whitespace);
                if let Some(child) = convert(document, child, &context, depth + 1)? {
                    element.children.push(Child::Element(child));
                }
            }
            Some(NodeKind::EntityReference) => {
                // An unexpanded entity reference cannot appear in the infoset the
                // data model is defined over; keep its name so the grammar check
                // reports the offending content.
                if let Some(name) = document.entity_reference(child) {
                    pending.push('&');
                    pending.push_str(name);
                    pending.push(';');
                }
            }
            _ => {}
        }
    }
    flush_text(&mut element, &mut pending, keeps_whitespace);

    // Rule 4.2: leading and trailing whitespace is removed from the content of
    // each `name` element.
    if element.name == "name" {
        let text = trim_whitespace(&element.text()).to_owned();
        element.children.clear();
        if !text.is_empty() {
            element.children.push(Child::Text(text));
        }
    }

    Ok(Some(element))
}

fn flush_text(element: &mut Element, pending: &mut String, keeps_whitespace: bool) {
    if pending.is_empty() {
        return;
    }
    let text = std::mem::take(pending);
    // Rule 4.2: for each element other than `value` and `param`, each child that
    // is a string containing only whitespace characters is removed.
    if keeps_whitespace || !crate::tree::is_whitespace(&text) {
        element.children.push(Child::Text(text));
    }
}

/// Resolves an `xml:base` value against the base URI in force.
fn resolve_base(current: Option<&str>, value: &str) -> Rc<str> {
    let escaped = crate::uri::xlink_escape(value);
    let resolved = current
        .and_then(|current| oxixml_iri::Iri::parse(current).ok())
        .and_then(|base| {
            let reference = oxixml_iri::IriRef::parse(escaped.as_str()).ok()?;
            base.resolve(&reference).ok()
        });
    match resolved {
        Some(resolved) => Rc::from(resolved.as_str()),
        None => Rc::from(escaped.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_elements_and_attributes_are_removed() -> Result<(), SchemaError> {
        let element = parse_str(
            r#"<element name="foo" xmlns="http://relaxng.org/ns/structure/1.0"
                        xmlns:a="urn:a" a:note="x">
                 <a:doc>ignored</a:doc>
                 <empty/>
               </element>"#,
            None,
        )?;
        assert_eq!(element.name, "element");
        assert_eq!(element.attributes.len(), 1);
        assert_eq!(element.attribute("name"), Some("foo"));
        let children: Vec<&str> = element.child_elements().map(|c| c.name.as_str()).collect();
        assert_eq!(children, ["empty"]);
        Ok(())
    }

    #[test]
    fn whitespace_only_children_are_dropped_except_in_value() -> Result<(), SchemaError> {
        let element = parse_str(
            "<element name=\"foo\" xmlns=\"http://relaxng.org/ns/structure/1.0\">\n  <value>  </value>\n</element>",
            None,
        )?;
        let value = element
            .child_elements()
            .next()
            .expect("the value element survives");
        assert_eq!(value.text(), "  ");
        assert!(!element.has_text_child());
        Ok(())
    }

    #[test]
    fn name_content_and_attributes_are_trimmed() -> Result<(), SchemaError> {
        let element = parse_str(
            "<element xmlns=\"http://relaxng.org/ns/structure/1.0\"><name>  foo  </name><empty/></element>",
            None,
        )?;
        let name = element
            .child_elements()
            .next()
            .expect("the name element is first");
        assert_eq!(name.text(), "foo");
        Ok(())
    }

    #[test]
    fn comments_do_not_split_character_data() -> Result<(), SchemaError> {
        let element = parse_str(
            "<value xmlns=\"http://relaxng.org/ns/structure/1.0\">a<!--c-->b</value>",
            None,
        )?;
        assert_eq!(element.text(), "ab");
        assert_eq!(element.children.len(), 1);
        Ok(())
    }

    #[test]
    fn a_foreign_document_element_is_rejected() {
        assert!(parse_str("<foo/>", None).is_err());
    }
}
