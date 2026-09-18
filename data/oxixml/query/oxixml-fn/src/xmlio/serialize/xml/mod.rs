//! The `xml`, `html` and `xhtml` output methods of
//! [Serialization 3.1 §5](https://www.w3.org/TR/xslt-xquery-serialization-31/#xml-output),
//! [§7](https://www.w3.org/TR/xslt-xquery-serialization-31/#html-output) and
//! [§8](https://www.w3.org/TR/xslt-xquery-serialization-31/#xhtml-output).
//!
//! The three share every structural rule — the same element, attribute, namespace, comment and
//! CDATA handling — and differ in the places §7 and §8 name.
//!
//! `html` writes HTML rather than XML: an element the HTML vocabulary knows to be empty is
//! written with a start tag alone, every other element gets an explicit end tag rather than a
//! self-closing one, a boolean attribute is minimized, a processing instruction is closed by `>`
//! alone, markup inside `script` and `style` is not escaped, an `&` that starts what a script
//! would read as a variable reference is left alone, a URI-valued attribute is percent-encoded,
//! and no XML declaration is written.
//!
//! `xhtml` writes XML — the declaration, the escaping, `?>` and CDATA sections are the `xml`
//! method's — with the three concessions §8 makes to HTML readers: an element of the XHTML
//! vocabulary whose content model is empty is minimized to `<br />`, with the space before the
//! slash that an HTML reader needs; every other empty element is written `<div></div>` rather
//! than self-closed, which an HTML reader would not recognize; and the vocabularies HTML 5
//! embeds without a prefix — XHTML, SVG and MathML — are written under a default namespace
//! declaration. It shares §7's document type declaration and its content-type `meta`.

mod escape;
mod html;
#[cfg(test)]
mod tests;

use super::params::{OutputMethod, SerializationParameters};
use super::{DepthGuard, normalized_items};
use crate::error::{FnError, FnErrorCode, FnResult};
pub(super) use escape::is_restricted_c0;
use escape::{AttributeStyle, escape_attribute, escape_text, write_cdata};
use html::{
    HTML_MEDIA_TYPE, XHTML_NAMESPACE, is_content_type_meta, is_html_element, is_html_empty_element,
    is_html_raw_text_element, is_unprefixed_namespace, is_xhtml_element, is_xhtml_empty_element,
};
use oxixml_infoset::{Item, NodeKind, NodeRef, Sequence, XdmNode};
use oxixml_qname::XML_NAMESPACE;
use std::fmt::Write as _;

pub(super) fn serialize_xml(
    sequence: &Sequence,
    parameters: &SerializationParameters,
) -> FnResult<String> {
    let items = normalized_items(sequence, parameters)?;
    let html_method = parameters.method() == OutputMethod::Html;
    if !html_method
        && (parameters.standalone().is_some() || parameters.doctype_system().is_some())
        && !is_well_formed_document(&items)
    {
        return Err(not_well_formed(parameters));
    }
    let mut out = String::new();
    // Serialization 3.1 §7: HTML has no XML declaration, so the `html` method never writes one
    // however `omit-xml-declaration` is set.
    if !parameters.omit_xml_declaration() && !html_method {
        let _ = write!(
            out,
            "<?xml version=\"{}\" encoding=\"{}\"",
            parameters.version(),
            parameters.encoding()
        );
        if let Some(standalone) = parameters.standalone() {
            let _ = write!(
                out,
                " standalone=\"{}\"",
                if standalone { "yes" } else { "no" }
            );
        }
        out.push_str("?>");
        if parameters.indent() {
            out.push('\n');
        }
    }
    let doctype_system = parameters.doctype_system();
    let doctype_public = parameters.doctype_public();
    let xhtml_method = parameters.method() == OutputMethod::Xhtml;
    // Serialization 3.1 §7.2 and §8.2: with an effective HTML version of 5 and no
    // `doctype-system`, both methods write the HTML5 document type declaration. The `html`
    // method writes `<!DOCTYPE root PUBLIC "…">` when `doctype-public` stands alone, so a
    // `doctype-public` displaces the HTML5 declaration there; the `xhtml` method follows §5.1.4
    // instead, where a public identifier without a system identifier is no declaration at all, so
    // the HTML5 one still stands (QT3 `Serialization-html-27` against `Serialization-xhtml-27`).
    let html5_doctype = (html_method || xhtml_method)
        && doctype_system.is_none()
        && (xhtml_method || doctype_public.is_none())
        && parameters.is_html5()
        // Only a *document*, whose outermost element is `html`, gets a document type
        // declaration: an HTML fragment such as a bare `body` must not (QT3 `serialize-html-003`,
        // `Serialization-xhtml-23`, `-24`).
        && document_element_name(&items).eq_ignore_ascii_case("html");
    if html5_doctype {
        out.push_str("<!DOCTYPE html>");
        out.push('\n');
    }
    // Serialization 3.1 §5.1.4 gives the `xml` method a document type declaration only when
    // `doctype-system` is specified — `doctype-public` alone is ignored, because an external
    // identifier needs a system identifier. §7.2 lifts that for the `html` method, whose
    // `<!DOCTYPE html PUBLIC "…">` is complete without one (QT3 `Serialization-html-25`, `-27`).
    if doctype_system.is_some() || (html_method && doctype_public.is_some()) {
        let root = document_element_name(&items);
        match (&doctype_public, &doctype_system) {
            (Some(public), Some(system)) => {
                let _ = write!(out, "<!DOCTYPE {root} PUBLIC \"{public}\" \"{system}\">");
            }
            (Some(public), None) => {
                let _ = write!(out, "<!DOCTYPE {root} PUBLIC \"{public}\">");
            }
            (None, Some(system)) => {
                let _ = write!(out, "<!DOCTYPE {root} SYSTEM \"{system}\">");
            }
            (None, None) => {}
        }
        if parameters.indent() {
            out.push('\n');
        }
    }

    let separator = parameters.item_separator();
    let mut previous_atomic = false;
    for (index, item) in items.iter().enumerate() {
        let atomic = matches!(item, Item::Atomic(_));
        if index > 0 {
            match &separator {
                Some(separator) => out.push_str(separator),
                // Serialization 3.1 §2: with no `item-separator` only *adjacent strings* are
                // joined by a single space, and an atomic value is the only item that becomes
                // one — a node keeps its neighbours' text hard against it.
                None if atomic && previous_atomic => out.push(' '),
                None => {}
            }
        }
        previous_atomic = atomic;
        match item {
            Item::Node(node) => write_node(&mut out, node, parameters, WriteContext::TOP)?,
            Item::Atomic(value) => {
                escape_text(&mut out, &value.string_value(), parameters, false);
            }
            // `normalized_items` has already replaced every array by its members and refused
            // every map and function item, so nothing else can reach this point.
            Item::Map(_) | Item::Array(_) | Item::Function(_) => {
                return Err(super::unserializable_item(parameters.method()));
            }
        }
    }
    Ok(out)
}

/// `SEPM0004` — `doctype-system`, or a `standalone` other than `omit`, was asked for while the
/// value being serialized is not a well-formed XML document.
fn not_well_formed(parameters: &SerializationParameters) -> FnError {
    let asked_for = if parameters.doctype_system().is_some() {
        "doctype-system"
    } else {
        "standalone"
    };
    FnError::new(
        FnErrorCode::Sepm0004,
        format!(
            "`{asked_for}` needs a well-formed document — exactly one element and no text of its own"
        ),
    )
}

/// Whether the normalized value would be a well-formed XML *document*: exactly one element, and
/// no character data outside it.
///
/// [Serialization 3.1 §5.1.1](https://www.w3.org/TR/xslt-xquery-serialization-31/#xml-output)
/// makes both `doctype-system` and a `standalone` other than `omit` an `SEPM0004` when this does
/// not hold, because neither an internal subset nor a standalone declaration means anything for
/// a fragment (QT3 `Serialization-026`, `-028`). Whitespace, comments and processing
/// instructions are the `Misc` that XML itself admits around the document element, so they do not
/// spoil it.
fn is_well_formed_document(items: &[Item]) -> bool {
    let mut elements = 0_usize;
    for item in items {
        match item {
            Item::Atomic(value) => {
                if !value.string_value().trim().is_empty() {
                    return false;
                }
            }
            Item::Node(node) => match node.node_kind() {
                NodeKind::Element => elements += 1,
                NodeKind::Document => {
                    for child in node.children() {
                        match child.node_kind() {
                            NodeKind::Element => elements += 1,
                            NodeKind::Text if !child.string_value().trim().is_empty() => {
                                return false;
                            }
                            _ => {}
                        }
                    }
                }
                NodeKind::Text if !node.string_value().trim().is_empty() => return false,
                _ => {}
            },
            _ => return false,
        }
    }
    elements == 1
}

fn document_element_name(items: &[Item]) -> String {
    items
        .iter()
        .find_map(|item| {
            let node = item.as_node()?;
            match node.node_kind() {
                NodeKind::Element => Some(qualified_name(node, &[])),
                NodeKind::Document => node
                    .children()
                    .into_iter()
                    .find(|child| child.node_kind() == NodeKind::Element)
                    .map(|element| qualified_name(&element, &[])),
                _ => None,
            }
        })
        .unwrap_or_else(|| "root".to_owned())
}

/// The in-scope `(prefix, namespace)` bindings of a node, with `None` for the default namespace.
type Bindings = [(Option<String>, String)];

/// Everything the writer inherits from the node's parent: where it sits, what its ancestors
/// declared, and which of the two rules that switch escaping and indentation off are in force.
#[derive(Debug, Clone, Copy)]
struct WriteContext<'a> {
    /// The indentation depth, which stands still inside an element that is not indented.
    depth: usize,
    /// The namespace bindings the ancestors already declared, so that none is written twice.
    inherited: &'a Bindings,
    /// Whether this node is inside a `script` or `style` element, whose markup the `html` method
    /// writes without escaping anything.
    raw_text: bool,
    /// Whether the parent decided to indent its children — that is, whether a newline and this
    /// node's indentation go before it.
    indent_here: bool,
    /// Whether `xml:space="preserve"`, or a `suppress-indentation` element, is in scope: the
    /// serializer may then neither add nor remove whitespace, however element-only the content is
    /// (QT3 `method-xml/K2-Serialization-40`, `-41`, `-42`).
    preserve_space: bool,
}

impl WriteContext<'_> {
    /// The context an item of the value being serialized starts from: outermost, with nothing in
    /// scope and no indentation before it.
    const TOP: Self = Self {
        depth: 0,
        inherited: &[],
        raw_text: false,
        indent_here: false,
        preserve_space: false,
    };
}

/// Writes one node with the `xml` output method's rules, from the outermost level — which is how
/// the `adaptive` method of §11 writes a node it meets.
pub(super) fn write_top_level_node(
    out: &mut String,
    node: &NodeRef,
    parameters: &SerializationParameters,
) -> FnResult<()> {
    write_node(out, node, parameters, WriteContext::TOP)
}

/// Escapes an attribute value with none of the `html` method's rules in force, which is how the
/// `adaptive` method writes the value of an attribute node.
pub(super) fn escape_plain_attribute(
    out: &mut String,
    value: &str,
    parameters: &SerializationParameters,
) {
    escape_attribute(out, value, parameters, AttributeStyle::PLAIN);
}

fn write_node(
    out: &mut String,
    node: &NodeRef,
    parameters: &SerializationParameters,
    context: WriteContext<'_>,
) -> FnResult<()> {
    let _guard = DepthGuard::enter(parameters.max_depth())?;
    let kind = node.node_kind();
    if context.indent_here && kind != NodeKind::Document {
        out.push('\n');
        for _ in 0..context.depth {
            out.push_str("  ");
        }
    }
    match kind {
        NodeKind::Document => {
            for child in node.children() {
                write_node(out, &child, parameters, context)?;
            }
            Ok(())
        }
        NodeKind::Element => write_element(out, node, parameters, context),
        NodeKind::Text => {
            escape_text(out, &node.string_value(), parameters, context.raw_text);
            Ok(())
        }
        NodeKind::Comment => {
            let _ = write!(out, "<!--{}-->", node.string_value());
            Ok(())
        }
        NodeKind::ProcessingInstruction => {
            let target = node
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default();
            let content = node.string_value();
            // Serialization 3.1 §7.6: the `html` output method closes a processing instruction
            // with `>` alone, because HTML has no `?>` terminator (QT3 `Serialization-html-48`,
            // `-58`, `-59`).
            let close = if parameters.method() == OutputMethod::Html {
                ">"
            } else {
                "?>"
            };
            if content.is_empty() {
                let _ = write!(out, "<?{target}{close}");
            } else {
                let _ = write!(out, "<?{target} {content}{close}");
            }
            Ok(())
        }
        // Serialization 3.1 §2: a parentless attribute or namespace node has no serialization
        // under the `xml` and `html` output methods.
        NodeKind::Attribute | NodeKind::Namespace => {
            Err(super::unserializable_node(parameters.method(), kind))
        }
    }
}

/// The `xml:space` an element sets, if it sets one: `Some(true)` for `preserve`, `Some(false)`
/// for `default`, which turns the rule off again for the subtree.
fn declared_space_preservation(element: &NodeRef) -> Option<bool> {
    element
        .attributes()
        .into_iter()
        .find(|attribute| {
            attribute.node_name().is_some_and(|name| {
                name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "space"
            })
        })
        .map(|attribute| attribute.string_value().trim() == "preserve")
}

fn write_element(
    out: &mut String,
    element: &NodeRef,
    parameters: &SerializationParameters,
    context: WriteContext<'_>,
) -> FnResult<()> {
    let bindings = element_bindings(element);
    let html = is_html_element(element, parameters);
    let xhtml = is_xhtml_element(element, parameters);
    let xhtml_method = parameters.method() == OutputMethod::Xhtml;
    let local = element
        .node_name()
        .map_or_else(String::new, |name| name.local_name().to_owned());
    let element_namespace = element
        .node_name()
        .and_then(|name| name.namespace().map(str::to_owned));
    // Serialization 3.1 §7.1: an element the `html` method writes as an HTML element carries no
    // prefix — HTML's own syntax has none — so an element in the XHTML namespace is written under
    // a default namespace declaration instead of under the prefix the tree happens to hold (QT3
    // `Serialization-html-19c`). §8 extends that to the three vocabularies an HTML 5 document
    // holds without a prefix, so that the `xhtml` method writes `<svg xmlns="…">` for an element
    // the tree spells `<svg:svg>` (`Serialization-xhtml-50`, `-51`, `-52`).
    let defaulted_namespace = element_namespace.as_deref().filter(|namespace| {
        if xhtml_method {
            is_unprefixed_namespace(namespace)
        } else {
            html && *namespace == XHTML_NAMESPACE
        }
    });
    let name = if defaulted_namespace.is_some() {
        local.clone()
    } else {
        qualified_name(element, &bindings)
    };

    let _ = write!(out, "<{name}");

    // An element whose *name* is in no namespace may not be written under a default-namespace
    // declaration: it would be read back in that namespace. The default binding it happens to
    // carry in scope is therefore suppressed, and undeclared with `xmlns=""` when an enclosing
    // element declared one — the namespace *undeclaration* of
    // [Serialization 3.1 §2](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam).
    // Only the default namespace can be undeclared in XML 1.0, so no `xmlns:p=""` is written.
    let in_no_namespace = element
        .node_name()
        .is_none_or(|name| name.namespace().is_none());
    let mut effective: Vec<(Option<String>, String)> = Vec::with_capacity(bindings.len() + 1);
    for (prefix, namespace) in &bindings {
        if prefix.is_none() && in_no_namespace {
            continue;
        }
        if drops_binding(element, namespace, defaulted_namespace, xhtml_method) {
            // Written below as the default namespace, on each element that is in it.
            continue;
        }
        effective.push((prefix.clone(), namespace.clone()));
        if prefix.as_deref() == Some("xml") {
            continue;
        }
        if context
            .inherited
            .iter()
            .any(|(bound, uri)| bound == prefix && uri == namespace)
        {
            continue;
        }
        match prefix {
            Some(prefix) => {
                let _ = write!(out, " xmlns:{prefix}=\"");
            }
            None => out.push_str(" xmlns=\""),
        }
        escape_attribute(out, namespace, parameters, AttributeStyle::PLAIN);
        out.push('"');
    }
    if let Some(namespace) = defaulted_namespace {
        if !context
            .inherited
            .iter()
            .any(|(prefix, uri)| prefix.is_none() && uri == namespace)
        {
            out.push_str(" xmlns=\"");
            escape_attribute(out, namespace, parameters, AttributeStyle::PLAIN);
            out.push('"');
        }
        effective.push((None, namespace.to_owned()));
    }
    if in_no_namespace {
        let inherits_default = context
            .inherited
            .iter()
            .any(|(prefix, uri)| prefix.is_none() && !uri.is_empty());
        if inherits_default {
            out.push_str(" xmlns=\"\"");
        }
        effective.push((None, String::new()));
    }
    write_undeclarations(
        out,
        parameters,
        &bindings,
        context.inherited,
        &mut effective,
    );

    for attribute in element.attributes() {
        let attribute_name = qualified_name(&attribute, &bindings);
        let attribute_local = attribute
            .node_name()
            .map_or_else(String::new, |name| name.local_name().to_owned());
        let value = attribute.string_value();
        // Serialization 3.1 §7.3: HTML's boolean attributes are written in minimized form —
        // the name alone, with no value — and an attribute counts as one exactly when its value
        // is its own name, compared without regard to case (QT3 `Serialization-html-12`, `-13`).
        if html && attribute_name.eq_ignore_ascii_case(&value) {
            let _ = write!(out, " {attribute_name}");
            continue;
        }
        let _ = write!(out, " {attribute_name}=\"");
        escape_attribute(
            out,
            &value,
            parameters,
            AttributeStyle::attribute(html, context.raw_text, &local, &attribute_local),
        );
        out.push('"');
    }

    // Serialization 3.1 §7.4: unless `include-content-type` says otherwise, the HTML and XHTML
    // output methods add a `meta` element declaring the media type and character encoding as the
    // first child of `head` — and *replace* one the source already carries (QT3
    // `Serialization-html-33`, `-34`, `-36`, `-37`, `-39`, `-40`,
    // `Serialization-xhtml-33`…`-40`). An HTML name is matched without regard to case; an XHTML
    // name is an XML name, so it is matched as it stands.
    let head = if xhtml {
        local == "head"
    } else {
        html && local.eq_ignore_ascii_case("head")
    };
    let content_type = (head && parameters.include_content_type()).then(|| {
        // XHTML is XML, so the element the method adds closes itself — with the space before the
        // slash HTML readers need, and with no separate end tag, which QT3
        // `Serialization-xhtml-34` and `-37` pin by allowing at most one `meta` in the result.
        let close = if xhtml { " />" } else { ">" };
        format!(
            "<meta http-equiv=\"Content-Type\" content=\"{HTML_MEDIA_TYPE}; charset={}\"{close}",
            parameters.encoding()
        )
    });

    // Serialization 3.1 §7.1: an HTML element whose content model is empty is written with a
    // start tag alone and never self-closed, and every *other* HTML element keeps its end tag
    // even when it has no children — `<spin></spin>`, not `<spin/>` (QT3 `Serialization-html-7`,
    // `-8`, `-23`, `-24`, `-45`).
    let empty_html_element = html && is_html_empty_element(&local);
    // Serialization 3.1 §8: XHTML minimizes the same vocabulary — but to `<br />`, self-closed
    // with a space, because XHTML is XML and a start tag alone would swallow the rest of the
    // document. Every *other* empty element is written with an explicit end tag, `<div></div>`,
    // which is what an HTML reader recognizes (QT3 `Serialization-xhtml-1`, `-1a`, `-2`, `-4`,
    // `-23`, `-24`, `-51`).
    let empty_xhtml_element = xhtml && is_xhtml_empty_element(&local, parameters);
    let children = element.children();
    if children.is_empty() {
        match &content_type {
            Some(meta) => {
                let _ = write!(out, ">{meta}</{name}>");
            }
            None if empty_html_element => out.push('>'),
            None if empty_xhtml_element => out.push_str(" />"),
            None if html || xhtml_method => {
                let _ = write!(out, "></{name}>");
            }
            None => out.push_str("/>"),
        }
        return Ok(());
    }
    out.push('>');
    if let Some(meta) = &content_type {
        out.push_str(meta);
    }
    // Serialization 3.1 §5.1.1 and §7.3: indentation may only be added where doing so cannot
    // change what the content means — inside an element that holds elements and nothing that
    // reads as text. Mixed content is therefore left alone (QT3 `Serialization-html-49`…`-54`,
    // `Serialization-xhtml-61`), and so is content that holds no element at all, such as a lone
    // processing instruction (`Serialization-html-58`, `-59`).
    // Serialization 3.1 §7: HTML element names are case-insensitive, and the `html` method matches
    // the names a parameter lists that way throughout (QT3 `Serialization-html-56`, `-57`).
    let case_blind = parameters.method() == OutputMethod::Html;
    let preserve_space = declared_space_preservation(element).unwrap_or(context.preserve_space)
        || parameters.suppresses_indentation(element, case_blind);
    let indentable = parameters.indent()
        && !preserve_space
        && children
            .iter()
            .all(|child| child.node_kind() != NodeKind::Text)
        && children
            .iter()
            .any(|child| child.node_kind() == NodeKind::Element);
    // Serialization 3.1 §7.1.4: HTML has no CDATA sections, so `cdata-section-elements` reaches
    // only the elements the `html` method is *not* writing as HTML — the XML islands (QT3
    // `Serialization-html-18`, `-19a`, `-19b`, `-19c`).
    let cdata = !html && parameters.writes_cdata_sections(element, case_blind);
    let replaces_content_type = content_type.is_some();
    // Serialization 3.1 §7.5: no escaping is performed on the content of `script` and `style`,
    // and the rule reaches every descendant, not only the direct children.
    let raw_text = context.raw_text || (html && is_html_raw_text_element(&local));
    let child_context = WriteContext {
        depth: if indentable {
            context.depth + 1
        } else {
            context.depth
        },
        inherited: &effective,
        raw_text,
        indent_here: indentable,
        preserve_space,
    };
    for child in &children {
        if cdata && child.node_kind() == NodeKind::Text {
            write_cdata(out, &child.string_value(), parameters);
            continue;
        }
        // The `meta` just written *replaces* the one the source declared, rather than joining it.
        if replaces_content_type && is_content_type_meta(child) {
            continue;
        }
        write_node(out, child, parameters, child_context)?;
    }
    if indentable {
        out.push('\n');
        for _ in 0..context.depth {
            out.push_str("  ");
        }
    }
    if !empty_html_element {
        let _ = write!(out, "</{name}>");
    }
    Ok(())
}

/// Whether a namespace binding an element holds is left out of its start tag because the method
/// writes that namespace as a *default* declaration instead.
///
/// For the `html` method that is the one namespace the element itself is being written under. The
/// `xhtml` method of Serialization 3.1 §8 goes further: XHTML, SVG and MathML are written without
/// a prefix wherever they occur, so a prefix bound to one of them is dropped rather than carried
/// through the document — which is what `Serialization-xhtml-51` asks for when it wants
/// `<svg xmlns="…/svg">` and not `<svg xmlns:h="…/xhtml" xmlns="…/svg">`. The prefix is kept when
/// an *attribute* of this element still needs it, because an attribute name has no default
/// namespace to fall back on.
fn drops_binding(
    element: &NodeRef,
    namespace: &str,
    defaulted: Option<&str>,
    xhtml_method: bool,
) -> bool {
    if !xhtml_method {
        return defaulted == Some(namespace);
    }
    if !is_unprefixed_namespace(namespace) {
        return false;
    }
    !element.attributes().iter().any(|attribute| {
        attribute
            .node_name()
            .is_some_and(|name| name.namespace() == Some(namespace))
    })
}

/// Writes the XML 1.1 namespace *undeclarations* `undeclare-prefixes` asks for.
///
/// [XML 1.1 §6](https://www.w3.org/TR/xml-names11/#nsc-NSDeclared) lets `xmlns:p=""` take a
/// prefix out of scope, and Serialization 3.1 §5.1.1 has the `xml` output method write one for
/// every prefix an ancestor declared that this element no longer has in scope — but only when the
/// output is XML 1.1, because XML 1.0 has no such declaration (QT3
/// `method-xml/K2-Serialization-20` against `-21`).
fn write_undeclarations(
    out: &mut String,
    parameters: &SerializationParameters,
    bindings: &Bindings,
    inherited: &Bindings,
    effective: &mut Vec<(Option<String>, String)>,
) {
    if !parameters.undeclare_prefixes() || !parameters.version().trim().starts_with("1.1") {
        return;
    }
    for (prefix, namespace) in inherited {
        let Some(prefix) = prefix else { continue };
        if prefix == "xml" || namespace.is_empty() {
            continue;
        }
        if bindings
            .iter()
            .any(|(bound, _)| bound.as_deref() == Some(prefix.as_str()))
        {
            continue;
        }
        let _ = write!(out, " xmlns:{prefix}=\"\"");
        effective.push((Some(prefix.clone()), String::new()));
    }
}

pub(super) fn element_bindings(element: &NodeRef) -> Vec<(Option<String>, String)> {
    element
        .namespaces()
        .into_iter()
        .map(|node| {
            (
                node.node_name().map(|name| name.local_name().to_owned()),
                node.string_value(),
            )
        })
        .collect()
}

fn qualified_name(node: &NodeRef, bindings: &Bindings) -> String {
    let Some(name) = node.node_name() else {
        return String::new();
    };
    let Some(namespace) = name.namespace() else {
        return name.local_name().to_owned();
    };
    if namespace == XML_NAMESPACE {
        return format!("xml:{}", name.local_name());
    }
    let prefix = bindings
        .iter()
        .find(|(prefix, uri)| uri == namespace && prefix.is_some())
        .and_then(|(prefix, _)| prefix.clone())
        .or_else(|| {
            node.parent().and_then(|parent| {
                element_bindings(&parent)
                    .into_iter()
                    .find(|(prefix, uri)| uri == namespace && prefix.is_some())
                    .and_then(|(prefix, _)| prefix)
            })
        });
    match prefix {
        Some(prefix) => format!("{prefix}:{}", name.local_name()),
        None => name.local_name().to_owned(),
    }
}
