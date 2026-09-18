//! Writing a parsed tree back out: as HTML, and as the html5lib tree dump.

use std::fmt::Write as _;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::ns::{
    HTML_NAMESPACE, MATHML_NAMESPACE, SVG_NAMESPACE, XLINK_NAMESPACE, XML_NAMESPACE,
    XMLNS_NAMESPACE, is_void_element,
};
use crate::parse::ParseOutput;

/// Options for [`serialize_node`].
#[derive(Debug, Clone, Default)]
pub struct SerializeOptions {
    /// Whether to write the node's own tags as well as its children.
    ///
    /// The HTML standard's algorithm is *fragment* serialization: it writes the
    /// children of a node, not the node itself, which is what `innerHTML`
    /// returns. Setting this to `true` gives `outerHTML` instead.
    pub include_self: bool,
}

impl SerializeOptions {
    /// Fragment serialization: the children only. This is the default.
    #[must_use]
    pub fn inner() -> Self {
        Self {
            include_self: false,
        }
    }

    /// Serialization including the node's own tags.
    #[must_use]
    pub fn outer() -> Self {
        Self { include_self: true }
    }
}

/// Serializes a parse result as HTML.
///
/// This is the "HTML fragment serialization algorithm" of
/// [HTML § 13.3](https://html.spec.whatwg.org/multipage/syntax.html#serialising-html-fragments):
/// void elements get no end tag, the raw-text elements get no escaping, and
/// `&`, `<`, `>` and U+00A0 are escaped in text.
///
/// # Examples
///
/// ```
/// use oxixml_html::{parse_document, serialize};
///
/// let parsed = parse_document("<p>a & b<br>c");
/// assert_eq!(serialize(&parsed), "<html><head></head><body><p>a &amp; b<br>c</p></body></html>");
/// ```
#[must_use]
pub fn serialize(parsed: &ParseOutput) -> String {
    let mut out = String::new();
    write_tree(
        &mut out,
        parsed.document(),
        parsed.root(),
        Some(parsed),
        false,
    );
    out
}

/// Serializes one node of a document as HTML.
#[must_use]
pub fn serialize_node(document: &Document, node: NodeId, options: &SerializeOptions) -> String {
    let mut out = String::new();
    write_tree(&mut out, document, node, None, options.include_self);
    out
}

/// One entry of the serializer's explicit work-list.
enum Step {
    /// Write this node (and schedule whatever it contains).
    Node(NodeId),
    /// Write this end tag, once everything it encloses has been written.
    Close(String),
}

/// Writes `node` — its children only, or the node itself as well when
/// `include_self` — walking the tree with an explicit work-list.
///
/// Deliberately iterative: one native stack frame per DOM level would abort the
/// process with a stack overflow on a deeply nested (but otherwise perfectly
/// ordinary) document that parsed just fine.
fn write_tree(
    out: &mut String,
    document: &Document,
    node: NodeId,
    parsed: Option<&ParseOutput>,
    include_self: bool,
) {
    let mut stack: Vec<Step> = Vec::new();
    if include_self {
        stack.push(Step::Node(node));
    } else {
        push_children(&mut stack, document, node);
    }

    while let Some(step) = stack.pop() {
        match step {
            Step::Close(tag_name) => {
                let _ = write!(out, "</{tag_name}>");
            }
            Step::Node(node) => write_node(out, document, node, parsed, &mut stack),
        }
    }
}

/// Schedules the children of `node`, in reverse so that popping yields them in
/// document order.
fn push_children(stack: &mut Vec<Step>, document: &Document, node: NodeId) {
    let children: Vec<NodeId> = document.children(node).collect();
    for child in children.into_iter().rev() {
        stack.push(Step::Node(child));
    }
}

/// The HTML spelling of a node's name or data, when it had to be rewritten to
/// fit an XML DOM.
fn original(parsed: Option<&ParseOutput>, node: NodeId) -> Option<&str> {
    parsed.and_then(|parsed| parsed.original_html(node))
}

fn write_node(
    out: &mut String,
    document: &Document,
    node: NodeId,
    parsed: Option<&ParseOutput>,
    stack: &mut Vec<Step>,
) {
    match document.kind(node) {
        Some(NodeKind::Element) => write_element(out, document, node, parsed, stack),
        Some(NodeKind::Text) | Some(NodeKind::CdataSection) => {
            let text = document.character_data(node).unwrap_or_default();
            if parent_is_raw_text(document, node) {
                out.push_str(text);
            } else {
                escape_text(out, text);
            }
        }
        Some(NodeKind::Comment) => {
            let data = original(parsed, node)
                .unwrap_or_else(|| document.character_data(node).unwrap_or_default());
            let _ = write!(out, "<!--{data}-->");
        }
        Some(NodeKind::DocumentType) => {
            if let Some(data) = document.doctype_data(node) {
                let name = original(parsed, node).unwrap_or_else(|| data.name());
                let _ = write!(out, "<!DOCTYPE {name}>");
            }
        }
        Some(NodeKind::ProcessingInstruction) => {
            if let Some(data) = document.processing_instruction(node) {
                let _ = write!(out, "<?{} {}?>", data.target(), data.data());
            }
        }
        _ => push_children(stack, document, node),
    }
}

fn write_element(
    out: &mut String,
    document: &Document,
    node: NodeId,
    parsed: Option<&ParseOutput>,
    stack: &mut Vec<Step>,
) {
    let Some(element) = document.element(node) else {
        return;
    };
    let name = element.local_name().to_owned();
    let is_html = element.namespace_uri() == Some(HTML_NAMESPACE);
    let tag_name = match original(parsed, node) {
        Some(original) => original.to_owned(),
        None if is_html => name.clone(),
        None => element.qualified_name(),
    };
    out.push('<');
    out.push_str(&tag_name);
    for attribute in document.attributes(node).collect::<Vec<_>>() {
        let Some(data) = document.attr(attribute) else {
            continue;
        };
        out.push(' ');
        match original(parsed, attribute) {
            Some(original) => out.push_str(original),
            None => out.push_str(&serialized_attribute_name(data)),
        }
        out.push_str("=\"");
        escape_attribute(out, data.value());
        out.push('"');
    }
    out.push('>');
    if is_html && is_void_element(&name) {
        return;
    }

    // HTML § 13.3 step 2: a `template` element's content lives in a separate
    // document fragment, not in its child list, so that is what gets written.
    let content = if is_html && name == "template" {
        parsed
            .and_then(|parsed| parsed.template_contents(node))
            .unwrap_or(node)
    } else {
        node
    };

    // HTML § 13.3: re-insert the U+000A that tree construction swallows right
    // after a `<pre>`, `<textarea>` or `<listing>` start tag, so that
    // serialize → parse → serialize is a fixed point.
    if is_html && matches!(name.as_str(), "pre" | "textarea" | "listing") {
        let starts_with_newline = document
            .children(content)
            .next()
            .filter(|&child| document.kind(child) == Some(NodeKind::Text))
            .and_then(|child| document.character_data(child))
            .is_some_and(|text| text.starts_with('\n'));
        if starts_with_newline {
            out.push('\n');
        }
    }

    stack.push(Step::Close(tag_name));
    push_children(stack, document, content);
}

/// The name an attribute is written with: the qualified name, except that the
/// namespaced attributes of foreign content are written with the prefix the
/// HTML serialization algorithm prescribes.
fn serialized_attribute_name(data: &oxixml_dom::AttrData) -> String {
    match data.namespace_uri() {
        Some(XML_NAMESPACE) => format!("xml:{}", data.local_name()),
        Some(XLINK_NAMESPACE) => format!("xlink:{}", data.local_name()),
        Some(XMLNS_NAMESPACE) => {
            if data.local_name() == "xmlns" {
                "xmlns".to_owned()
            } else {
                format!("xmlns:{}", data.local_name())
            }
        }
        _ => data.qualified_name(),
    }
}

fn parent_is_raw_text(document: &Document, node: NodeId) -> bool {
    document
        .parent(node)
        .and_then(|parent| document.element(parent))
        .is_some_and(|element| {
            element.namespace_uri() == Some(HTML_NAMESPACE)
                && crate::ns::is_raw_text_element(element.local_name())
        })
}

fn escape_text(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '\u{a0}' => out.push_str("&nbsp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
    }
}

fn escape_attribute(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '\u{a0}' => out.push_str("&nbsp;"),
            '"' => out.push_str("&quot;"),
            other => out.push(other),
        }
    }
}

// --------------------------------------------------------------- tree dump

/// Formats a parse result in the html5lib tree dump format.
///
/// This is the format the HTML standard's own test suite uses, and it is the
/// most useful thing to print when a tree is not the shape you expected: one
/// line per node, two spaces of indentation per level, with a `| ` gutter.
///
/// # Examples
///
/// ```
/// use oxixml_html::{dump_tree, parse_fragment};
///
/// let parsed = parse_fragment("<b>bold<i>both</b>italic", "body");
/// assert_eq!(
///     dump_tree(&parsed),
///     "\
/// | <b>
/// |   \"bold\"
/// |   <i>
/// |     \"both\"
/// | <i>
/// |   \"italic\"
/// "
/// );
/// ```
#[must_use]
pub fn dump_tree(parsed: &ParseOutput) -> String {
    let mut out = String::new();
    let mut stack: Vec<DumpEntry> = Vec::new();
    push_dump_children(&mut stack, parsed.document(), parsed.root(), 1);
    while let Some((node, depth)) = stack.pop() {
        dump_node(&mut out, parsed, node, depth, &mut stack);
    }
    out
}

/// One entry of the tree dump's explicit work-list: a node still waiting to
/// be dumped, together with the indentation depth it belongs at.
type DumpEntry = (NodeId, usize);

/// Schedules the children of `node` to be dumped at `depth`, in reverse so
/// that popping the stack (last-in, first-out) yields them in document
/// order.
fn push_dump_children(stack: &mut Vec<DumpEntry>, document: &Document, node: NodeId, depth: usize) {
    let children: Vec<NodeId> = document.children(node).collect();
    for child in children.into_iter().rev() {
        stack.push((child, depth));
    }
}

/// Writes the `| ` gutter and `depth - 1` levels of two-space indentation.
fn indent(out: &mut String, depth: usize) {
    out.push_str("| ");
    for _ in 1..depth {
        out.push_str("  ");
    }
}

/// Dumps one node, scheduling whatever it contains onto `stack` rather than
/// recursing into it directly — see [`write_tree`]'s documentation for why:
/// one native stack frame per DOM level would abort the process with a stack
/// overflow on a deeply nested (but otherwise perfectly ordinary) document
/// that parsed just fine.
fn dump_node(
    out: &mut String,
    parsed: &ParseOutput,
    node: NodeId,
    depth: usize,
    stack: &mut Vec<DumpEntry>,
) {
    let document = parsed.document();
    match document.kind(node) {
        Some(NodeKind::Element) => dump_element(out, parsed, node, depth, stack),
        Some(NodeKind::Text) | Some(NodeKind::CdataSection) => {
            indent(out, depth);
            let _ = writeln!(
                out,
                "\"{}\"",
                document.character_data(node).unwrap_or_default()
            );
        }
        Some(NodeKind::Comment) => {
            indent(out, depth);
            let data = parsed
                .original_html(node)
                .unwrap_or_else(|| document.character_data(node).unwrap_or_default());
            let _ = writeln!(out, "<!-- {data} -->");
        }
        Some(NodeKind::DocumentType) => {
            indent(out, depth);
            if let Some(data) = document.doctype_data(node) {
                let name = parsed.original_html(node).unwrap_or_else(|| data.name());
                match (data.public_id(), data.system_id()) {
                    (None, None) => {
                        let _ = writeln!(out, "<!DOCTYPE {name}>");
                    }
                    (public, system) => {
                        let _ = writeln!(
                            out,
                            "<!DOCTYPE {name} \"{}\" \"{}\">",
                            public.unwrap_or(""),
                            system.unwrap_or("")
                        );
                    }
                }
            }
        }
        Some(NodeKind::ProcessingInstruction) => {
            indent(out, depth);
            if let Some(data) = document.processing_instruction(node) {
                let _ = writeln!(out, "<?{} {}?>", data.target(), data.data());
            }
        }
        _ => {}
    }
}

fn dump_element(
    out: &mut String,
    parsed: &ParseOutput,
    node: NodeId,
    depth: usize,
    stack: &mut Vec<DumpEntry>,
) {
    let document = parsed.document();
    let Some(element) = document.element(node) else {
        return;
    };
    indent(out, depth);
    let prefix = match element.namespace_uri() {
        Some(MATHML_NAMESPACE) => "math ",
        Some(SVG_NAMESPACE) => "svg ",
        _ => "",
    };
    let name = parsed
        .original_html(node)
        .map_or_else(|| element.qualified_name(), str::to_owned);
    let _ = writeln!(out, "<{prefix}{name}>");

    let mut attributes: Vec<String> = document
        .attributes(node)
        .filter_map(|attribute| document.attr(attribute))
        .zip(document.attributes(node))
        .map(|(data, id)| {
            let name = match parsed.original_html(id) {
                Some(original) => original.to_owned(),
                None => match data.namespace_uri() {
                    Some(XML_NAMESPACE) => format!("xml {}", data.local_name()),
                    Some(XLINK_NAMESPACE) => format!("xlink {}", data.local_name()),
                    Some(XMLNS_NAMESPACE) => format!("xmlns {}", data.local_name()),
                    _ => data.qualified_name(),
                },
            };
            format!("{name}=\"{}\"", data.value())
        })
        .collect();
    attributes.sort();
    for attribute in attributes {
        indent(out, depth + 1);
        let _ = writeln!(out, "{attribute}");
    }

    // Schedule the node's own children first (so they sort under everything
    // the template content below pushes — a stack pops last-in first) and
    // the template content's children last (so they are popped, and
    // therefore dumped, before the node's own children). That reproduces
    // the order the equivalent recursive walk would produce (content, then
    // the node's own children) without ever recursing into either.
    push_dump_children(stack, document, node, depth + 1);
    if let Some(content) = parsed.template_contents(node) {
        indent(out, depth + 1);
        out.push_str("content\n");
        push_dump_children(stack, document, content, depth + 2);
    }
}
