//! Serializing an element's content into an `rdf:XMLLiteral` or an `rdf:HTML` literal.
//!
//! [RDFa Core 1.1 §7.5 step 11](https://www.w3.org/TR/rdfa-core/#s_sequence) says that an
//! `rdf:XMLLiteral` is "a string created by serializing to text, all nodes that are descendants of
//! the current element, i.e., not including the element itself", in the format of [Exclusive XML
//! Canonicalization](https://www.w3.org/TR/xml-exc-c14n/), and adds a normative note requiring the
//! in-scope namespace declarations to be reproduced on each serialized top-level element.
//! [HTML+RDFa 1.1 §3.4](https://www.w3.org/TR/html-rdfa/#invalid-xmlliteral-values) restates that
//! requirement and extends it to the prefixes declared with `@prefix`, and
//! [§3.1](https://www.w3.org/TR/html-rdfa/#additional-rdfa-processing-rules) adds the parallel
//! `rdf:HTML` literal, serialized with the HTML rules instead.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use oxixml_dom::{Document, NodeId, NodeKind, XMLNS_NAMESPACE};

/// Serializes the children of `element` as an `rdf:XMLLiteral`.
///
/// `declarations` are the namespace declarations to reproduce on every top-level element, as
/// `(prefix, namespace name)` pairs with `None` as the prefix standing for the default namespace.
/// A declaration the element itself makes always wins over an inherited one.
///
/// Returns `None` instead of a value if `element`'s descendants are nested more than `max_depth`
/// levels deep (counting `element`'s direct children as depth `0`): [`write_xml_node`] and
/// [`write_xml_element`] recurse into each other with one native stack frame per level, so an
/// unbounded subtree would otherwise turn a modestly-sized document into a process-aborting stack
/// overflow. This is the same class of risk [`Processor`](crate::eval::Processor) itself guards
/// against while walking the document, and is bounded by the same `max_depth`, but is tracked
/// independently: an `rdf:XMLLiteral` element's descendants are a fresh subtree, unrelated to how
/// deep that element itself sits in the outer document.
pub(crate) fn xml_literal(
    document: &Document,
    element: NodeId,
    declarations: &BTreeMap<Option<String>, String>,
    max_depth: usize,
) -> Option<String> {
    let mut out = String::new();
    for child in document.children(element) {
        write_xml_node(document, child, Some(declarations), &mut out, 0, max_depth)?;
    }
    Some(out)
}

/// Serializes the children of `element` as an `rdf:HTML` literal: the HTML fragment serialization,
/// with no namespace fixup at all.
///
/// Returns `None` on excessive nesting; see [`xml_literal`], whose depth bound this mirrors.
pub(crate) fn html_literal(
    document: &Document,
    element: NodeId,
    max_depth: usize,
) -> Option<String> {
    let mut out = String::new();
    for child in document.children(element) {
        write_html_node(document, child, &mut out, 0, max_depth)?;
    }
    Some(out)
}

/// The namespace declarations in scope at `node`, innermost first, as an RDFa processor needs them
/// for an `rdf:XMLLiteral`.
///
/// Both spellings are recognised: real `xmlns` attributes in the XML namespace (what an XML or
/// XHTML parse produces) and the plain `xmlns:prefix` attributes an HTML parse leaves behind,
/// since `text/html` puts no attribute in the `xmlns` namespace.
pub(crate) fn in_scope_declarations(
    document: &Document,
    node: NodeId,
) -> BTreeMap<Option<String>, String> {
    let mut declarations = BTreeMap::new();
    let mut chain: Vec<NodeId> = document.ancestors(node).collect();
    chain.reverse();
    chain.push(node);
    for ancestor in chain {
        if document.kind(ancestor) != Some(NodeKind::Element) {
            continue;
        }
        for attribute in document.attributes(ancestor) {
            let Some(data) = document.attr(attribute) else {
                continue;
            };
            let qualified = data.qualified_name();
            let prefix = if data.namespace_uri() == Some(XMLNS_NAMESPACE) {
                if data.prefix() == Some("xmlns") {
                    Some(data.local_name().to_owned())
                } else {
                    None
                }
            } else if let Some(prefix) = qualified.strip_prefix("xmlns:") {
                Some(prefix.to_owned())
            } else if qualified == "xmlns" {
                None
            } else {
                continue;
            };
            declarations.insert(prefix, data.value().to_owned());
        }
    }
    declarations
}

/// Writes one node of an `rdf:XMLLiteral`.
///
/// `depth` is `node`'s own nesting level (see [`xml_literal`]); returns `None` without writing
/// anything once it reaches `max_depth`, which the caller must propagate outward rather than
/// paper over, since the literal built so far is incomplete.
fn write_xml_node(
    document: &Document,
    node: NodeId,
    declarations: Option<&BTreeMap<Option<String>, String>>,
    out: &mut String,
    depth: usize,
    max_depth: usize,
) -> Option<()> {
    if depth >= max_depth {
        return None;
    }
    match document.kind(node) {
        Some(NodeKind::Text | NodeKind::CdataSection) => {
            if let Some(text) = document.character_data(node) {
                escape_text(text, out);
            }
        }
        Some(NodeKind::Comment) => {
            if let Some(text) = document.character_data(node) {
                let _ = write!(out, "<!--{text}-->");
            }
        }
        Some(NodeKind::ProcessingInstruction) => {
            if let Some(data) = document.processing_instruction(node) {
                let _ = write!(out, "<?{} {}?>", data.target(), data.data());
            }
        }
        Some(NodeKind::Element) => {
            write_xml_element(document, node, declarations, out, depth, max_depth)?;
        }
        _ => {}
    }
    Some(())
}

fn write_xml_element(
    document: &Document,
    element: NodeId,
    declarations: Option<&BTreeMap<Option<String>, String>>,
    out: &mut String,
    depth: usize,
    max_depth: usize,
) -> Option<()> {
    let Some(data) = document.element(element) else {
        return Some(());
    };
    let name = data.qualified_name();
    out.push('<');
    out.push_str(&name);

    let mut own: BTreeMap<Option<String>, String> = BTreeMap::new();
    for attribute in document.attributes(element) {
        let Some(attr) = document.attr(attribute) else {
            continue;
        };
        let qualified = attr.qualified_name();
        if qualified == "xmlns" {
            own.insert(None, attr.value().to_owned());
            continue;
        }
        if let Some(prefix) = qualified.strip_prefix("xmlns:") {
            own.insert(Some(prefix.to_owned()), attr.value().to_owned());
            continue;
        }
        out.push(' ');
        out.push_str(&qualified);
        out.push_str("=\"");
        escape_attribute(attr.value(), out);
        out.push('"');
    }

    if let Some(inherited) = declarations {
        let mut merged = inherited.clone();
        // The element's own declarations take precedence over the inherited ones.
        for (prefix, namespace) in &own {
            merged.insert(prefix.clone(), namespace.clone());
        }
        // A top-level element that is itself in a namespace must keep it.
        if let Some(namespace) = data.namespace_uri() {
            merged
                .entry(data.prefix().map(str::to_owned))
                .or_insert_with(|| namespace.to_owned());
        }
        write_declarations(&merged, out);
    } else {
        write_declarations(&own, out);
    }

    out.push('>');
    for child in document.children(element) {
        write_xml_node(document, child, None, out, depth + 1, max_depth)?;
    }
    let _ = write!(out, "</{name}>");
    Some(())
}

fn write_declarations(declarations: &BTreeMap<Option<String>, String>, out: &mut String) {
    if let Some(namespace) = declarations.get(&None) {
        out.push_str(" xmlns=\"");
        escape_attribute(namespace, out);
        out.push('"');
    }
    for (prefix, namespace) in declarations {
        let Some(prefix) = prefix else { continue };
        let _ = write!(out, " xmlns:{prefix}=\"");
        escape_attribute(namespace, out);
        out.push('"');
    }
}

/// Writes one node of an `rdf:HTML` literal; see [`write_xml_node`], whose depth handling this
/// mirrors.
fn write_html_node(
    document: &Document,
    node: NodeId,
    out: &mut String,
    depth: usize,
    max_depth: usize,
) -> Option<()> {
    if depth >= max_depth {
        return None;
    }
    match document.kind(node) {
        Some(NodeKind::Text | NodeKind::CdataSection) => {
            if let Some(text) = document.character_data(node) {
                escape_text(text, out);
            }
        }
        Some(NodeKind::Comment) => {
            if let Some(text) = document.character_data(node) {
                let _ = write!(out, "<!--{text}-->");
            }
        }
        Some(NodeKind::Element) => {
            let Some(data) = document.element(node) else {
                return Some(());
            };
            let name = data.local_name().to_ascii_lowercase();
            out.push('<');
            out.push_str(&name);
            for attribute in document.attributes(node) {
                if let Some(attr) = document.attr(attribute) {
                    out.push(' ');
                    out.push_str(&attr.qualified_name());
                    out.push_str("=\"");
                    escape_attribute(attr.value(), out);
                    out.push('"');
                }
            }
            out.push('>');
            if oxixml_html::is_void_element(&name) {
                return Some(());
            }
            for child in document.children(node) {
                write_html_node(document, child, out, depth + 1, max_depth)?;
            }
            let _ = write!(out, "</{name}>");
        }
        _ => {}
    }
    Some(())
}

fn escape_text(text: &str, out: &mut String) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#xD;"),
            _ => out.push(ch),
        }
    }
}

fn escape_attribute(value: &str, out: &mut String) {
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' => out.push_str("&quot;"),
            '\t' => out.push_str("&#x9;"),
            '\n' => out.push_str("&#xA;"),
            '\r' => out.push_str("&#xD;"),
            _ => out.push(ch),
        }
    }
}
