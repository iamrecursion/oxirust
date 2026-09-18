//! The XPath 1.0 data model: four value types and the conversions between them.
//!
//! [XPath §1](https://www.w3.org/TR/1999/REC-xpath-19991116/#section-Introduction)
//! defines exactly four types — node-set, boolean, number and string — and
//! [§4.2 / §3.4](https://www.w3.org/TR/1999/REC-xpath-19991116/#booleans) fixes
//! how each converts to the others. Getting those conversions right is most of
//! what makes a predicate behave: `[@a]` is a node-set turned into a boolean,
//! `[1]` is a number compared against `position()`, and `[@a='x']` is the
//! existential comparison of §3.4, which is true when *any* node in the set
//! compares equal.
//!
//! # Namespace nodes
//!
//! XPath models an element's in-scope namespace bindings as *namespace nodes*
//! that live on the element, and this module represents them explicitly
//! ([`XNode::Namespace`]) so the `namespace::` axis and `count(namespace::*)`
//! behave. They sort immediately after their element and before its attributes,
//! which is the order [§5](https://www.w3.org/TR/1999/REC-xpath-19991116/#data-model)
//! describes.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use oxixml_dom::{Document, NodeId, NodeKind, XML_NAMESPACE, XMLNS_NAMESPACE};

/// A node in the XPath data model.
///
/// Everything except a namespace node is an ordinary DOM node; namespace nodes
/// do not exist in the DOM (a `xmlns:*` declaration is an attribute there) and
/// are synthesised on demand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum XNode {
    /// A node that exists in the DOM arena.
    Tree(NodeId),
    /// A synthesised namespace node.
    Namespace(NamespaceNode),
}

/// A namespace node: a binding, plus the element it is in scope on.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NamespaceNode {
    /// The element the node hangs off.
    pub(crate) element: NodeId,
    /// The prefix, empty for the default namespace.
    pub(crate) prefix: String,
    /// The namespace name.
    pub(crate) uri: String,
}

impl XNode {
    /// The DOM node, for everything but a namespace node.
    pub(crate) const fn as_tree(&self) -> Option<NodeId> {
        match self {
            Self::Tree(id) => Some(*id),
            Self::Namespace(_) => None,
        }
    }

    /// The sort key that puts nodes in document order.
    ///
    /// The DOM already numbers a node's attributes immediately after it and
    /// before its children, so a namespace node only has to claim the slot
    /// between the element and its first attribute: it borrows the element's
    /// number and breaks the tie with a rank of 1 against the element's 0.
    fn sort_key(&self, document: &Document) -> (u32, u8, &str) {
        match self {
            Self::Tree(id) => (document.document_order(*id).unwrap_or(u32::MAX), 0, ""),
            Self::Namespace(namespace) => (
                document
                    .document_order(namespace.element)
                    .unwrap_or(u32::MAX),
                1,
                namespace.prefix.as_str(),
            ),
        }
    }

    /// The XPath *string-value* of the node —
    /// [§5](https://www.w3.org/TR/1999/REC-xpath-19991116/#data-model).
    pub(crate) fn string_value(&self, document: &Document) -> String {
        match self {
            Self::Namespace(namespace) => namespace.uri.clone(),
            Self::Tree(id) => match document.kind(*id) {
                Some(NodeKind::Attr) => document
                    .attr(*id)
                    .map(|attribute| attribute.value().to_owned())
                    .unwrap_or_default(),
                Some(NodeKind::Text | NodeKind::CdataSection | NodeKind::Comment) => {
                    document.character_data(*id).unwrap_or_default().to_owned()
                }
                Some(NodeKind::ProcessingInstruction) => document
                    .processing_instruction(*id)
                    .map(|pi| pi.data().to_owned())
                    .unwrap_or_default(),
                _ => document.text_content(*id).unwrap_or_default(),
            },
        }
    }

    /// The XPath *expanded name*: `(namespace, local)`, or [`None`] for the
    /// node types that have none.
    pub(crate) fn expanded_name(&self, document: &Document) -> Option<(Option<String>, String)> {
        match self {
            Self::Namespace(namespace) => Some((None, namespace.prefix.clone())),
            Self::Tree(id) => match document.kind(*id) {
                Some(NodeKind::Element) => document.element(*id).map(|element| {
                    (
                        element.namespace_uri().map(str::to_owned),
                        element.local_name().to_owned(),
                    )
                }),
                Some(NodeKind::Attr) => document.attr(*id).map(|attribute| {
                    (
                        attribute.namespace_uri().map(str::to_owned),
                        attribute.local_name().to_owned(),
                    )
                }),
                Some(NodeKind::ProcessingInstruction) => document
                    .processing_instruction(*id)
                    .map(|pi| (None, pi.target().to_owned())),
                _ => None,
            },
        }
    }
}

/// Whether a DOM attribute is really a namespace declaration.
///
/// `oxixml-dom` keeps `xmlns` and `xmlns:*` as ordinary attributes in the
/// `http://www.w3.org/2000/xmlns/` namespace, which is what the DOM
/// Recommendation prescribes. XPath does not have them on the `attribute::`
/// axis at all, so they have to be filtered out wherever that axis is walked.
pub(crate) fn is_namespace_declaration(document: &Document, attribute: NodeId) -> bool {
    document.attr(attribute).is_some_and(|data| {
        data.namespace_uri() == Some(XMLNS_NAMESPACE)
            || (data.prefix().is_none() && data.local_name() == "xmlns")
    })
}

/// The namespace nodes in scope on `element`, in prefix order.
///
/// Every element has a node for the `xml` prefix, which
/// [XML Names §3](https://www.w3.org/TR/xml-names/#ns-decl) binds implicitly.
/// A declaration of `xmlns=""` *undeclares* the default namespace, so it
/// produces no node.
pub(crate) fn namespace_nodes(document: &Document, element: NodeId) -> Vec<NamespaceNode> {
    let mut bindings: Vec<(String, String)> = vec![("xml".to_owned(), XML_NAMESPACE.to_owned())];
    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.insert("xml".to_owned());

    let mut current = Some(element);
    while let Some(node) = current {
        if document.kind(node) == Some(NodeKind::Element) {
            for attribute in document.attributes(node) {
                let Some(data) = document.attr(attribute) else {
                    continue;
                };
                // `xmlns:p="…"` binds the prefix `p`; `xmlns="…"` binds the
                // default namespace, which XPath models as the empty prefix.
                let prefix = match (data.prefix(), data.local_name()) {
                    (Some("xmlns"), local) => local.to_owned(),
                    (None, "xmlns") => String::new(),
                    _ => continue,
                };
                if !seen.insert(prefix.clone()) {
                    continue;
                }
                if data.value().is_empty() {
                    // An undeclaration: the binding disappears, it does not
                    // become a node bound to the empty string.
                    continue;
                }
                bindings.push((prefix, data.value().to_owned()));
            }
        }
        current = document.parent(node);
    }

    bindings.sort();
    bindings
        .into_iter()
        .map(|(prefix, uri)| NamespaceNode {
            element,
            prefix,
            uri,
        })
        .collect()
}

/// An XPath node-set: a set of nodes, kept sorted in document order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NodeSet {
    nodes: Vec<XNode>,
}

impl NodeSet {
    /// An empty set.
    pub(crate) const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// A set holding one node.
    pub(crate) fn singleton(node: XNode) -> Self {
        Self { nodes: vec![node] }
    }

    /// Builds a set from an iterator, sorting and de-duplicating.
    pub(crate) fn from_nodes(document: &Document, nodes: impl IntoIterator<Item = XNode>) -> Self {
        let mut set = Self {
            nodes: nodes.into_iter().collect(),
        };
        set.normalize(document);
        set
    }

    /// The nodes, in document order.
    pub(crate) fn nodes(&self) -> &[XNode] {
        &self.nodes
    }

    /// Consumes the set and returns its nodes.
    pub(crate) fn into_nodes(self) -> Vec<XNode> {
        self.nodes
    }

    /// The number of nodes.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the set is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The first node in document order.
    pub(crate) fn first(&self) -> Option<&XNode> {
        self.nodes.first()
    }

    /// Adds a node without re-sorting; call [`normalize`](Self::normalize) after.
    pub(crate) fn push(&mut self, node: XNode) {
        self.nodes.push(node);
    }

    /// Sorts into document order and removes duplicates.
    pub(crate) fn normalize(&mut self, document: &Document) {
        self.nodes
            .sort_by(|a, b| compare_document_order(document, a, b));
        self.nodes.dedup();
    }

    /// The union of two sets.
    pub(crate) fn union(mut self, document: &Document, other: Self) -> Self {
        self.nodes.extend(other.nodes);
        self.normalize(document);
        self
    }

    /// The string-value of the set: that of its first node, or `""` when empty.
    pub(crate) fn string_value(&self, document: &Document) -> String {
        self.nodes
            .first()
            .map(|node| node.string_value(document))
            .unwrap_or_default()
    }
}

/// Compares two XPath nodes in document order.
pub(crate) fn compare_document_order(document: &Document, a: &XNode, b: &XNode) -> Ordering {
    a.sort_key(document).cmp(&b.sort_key(document))
}

/// An XPath value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    /// A node-set.
    NodeSet(NodeSet),
    /// A boolean.
    Boolean(bool),
    /// A number, always an IEEE 754 double.
    Number(f64),
    /// A string.
    Str(String),
}

impl Value {
    /// The `boolean()` conversion —
    /// [XPath §4.3](https://www.w3.org/TR/1999/REC-xpath-19991116/#function-boolean).
    pub(crate) fn to_boolean(&self) -> bool {
        match self {
            Self::Boolean(value) => *value,
            Self::Number(value) => *value != 0.0 && !value.is_nan(),
            Self::Str(text) => !text.is_empty(),
            Self::NodeSet(set) => !set.is_empty(),
        }
    }

    /// The `string()` conversion —
    /// [XPath §4.2](https://www.w3.org/TR/1999/REC-xpath-19991116/#function-string).
    pub(crate) fn to_xpath_string(&self, document: &Document) -> String {
        match self {
            Self::Str(text) => text.clone(),
            Self::Boolean(value) => if *value { "true" } else { "false" }.to_owned(),
            Self::Number(value) => number_to_string(*value),
            Self::NodeSet(set) => set.string_value(document),
        }
    }

    /// The `number()` conversion —
    /// [XPath §4.4](https://www.w3.org/TR/1999/REC-xpath-19991116/#function-number).
    pub(crate) fn to_number(&self, document: &Document) -> f64 {
        match self {
            Self::Number(value) => *value,
            Self::Boolean(value) => {
                if *value {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Str(text) => string_to_number(text),
            Self::NodeSet(_) => string_to_number(&self.to_xpath_string(document)),
        }
    }
}

/// The XPath number-to-string conversion.
///
/// XPath §4.2 asks for a decimal expansion with no exponent. This defers to the
/// ECMAScript `Number::toString` algorithm in [`oxixml_text::num`], which agrees
/// with XPath on every value whose magnitude lies in `[1e-6, 1e21)` — the whole
/// range a signature transform can plausibly produce — and differs only by
/// using exponent notation outside it.
fn number_to_string(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    if value == 0.0 {
        // XPath has no negative zero in its lexical space.
        return "0".to_owned();
    }
    oxixml_text::num::ecmascript_double(value)
}

/// The XPath string-to-number conversion: optional whitespace, an optional
/// minus sign, digits with an optional fractional part, optional whitespace.
/// Anything else is `NaN` — in particular an exponent, which XPath does not
/// accept.
fn string_to_number(text: &str) -> f64 {
    let trimmed = text.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\r' | '\n'));
    let body = trimmed.strip_prefix('-').unwrap_or(trimmed);
    if body.is_empty() {
        return f64::NAN;
    }
    let mut seen_digit = false;
    let mut seen_dot = false;
    for character in body.chars() {
        match character {
            '0'..='9' => seen_digit = true,
            '.' if !seen_dot => seen_dot = true,
            _ => return f64::NAN,
        }
    }
    if !seen_digit {
        return f64::NAN;
    }
    trimmed.parse::<f64>().unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    #[test]
    fn conversions_follow_the_recommendation() {
        let document = build::parse_str("<a/>").expect("well-formed input");
        assert!(Value::Number(1.0).to_boolean());
        assert!(!Value::Number(0.0).to_boolean());
        assert!(!Value::Number(f64::NAN).to_boolean());
        assert!(!Value::Str(String::new()).to_boolean());
        assert!(Value::Str("0".to_owned()).to_boolean());
        assert!(!Value::NodeSet(NodeSet::new()).to_boolean());

        assert_eq!(Value::Boolean(true).to_xpath_string(&document), "true");
        assert_eq!(Value::Number(1.0).to_xpath_string(&document), "1");
        assert_eq!(Value::Number(1.5).to_xpath_string(&document), "1.5");
        assert_eq!(Value::Number(-0.0).to_xpath_string(&document), "0");
        assert_eq!(Value::Number(f64::NAN).to_xpath_string(&document), "NaN");
        assert_eq!(
            Value::Number(f64::INFINITY).to_xpath_string(&document),
            "Infinity"
        );

        assert_eq!(Value::Str("  12 ".to_owned()).to_number(&document), 12.0);
        assert_eq!(Value::Str("-1.5".to_owned()).to_number(&document), -1.5);
        assert!(Value::Str("1e3".to_owned()).to_number(&document).is_nan());
        assert!(Value::Str("abc".to_owned()).to_number(&document).is_nan());
        assert!(Value::Str(String::new()).to_number(&document).is_nan());
    }

    #[test]
    fn namespace_nodes_include_the_implicit_xml_prefix() {
        let document =
            build::parse_str(r#"<a xmlns:p="urn:p"><b xmlns=""/></a>"#).expect("well-formed input");
        let a = document.document_element().expect("a document element");
        let nodes = namespace_nodes(&document, a);
        let prefixes: Vec<&str> = nodes.iter().map(|node| node.prefix.as_str()).collect();
        assert_eq!(prefixes, ["p", "xml"]);
    }

    #[test]
    fn an_undeclaration_removes_the_default_namespace() {
        let document =
            build::parse_str(r#"<a xmlns="urn:a"><b xmlns=""/></a>"#).expect("well-formed input");
        let a = document.document_element().expect("a document element");
        let b = document.children(a).next().expect("a child");
        let outer: Vec<String> = namespace_nodes(&document, a)
            .into_iter()
            .map(|node| node.prefix)
            .collect();
        assert_eq!(outer, ["", "xml"]);
        let inner: Vec<String> = namespace_nodes(&document, b)
            .into_iter()
            .map(|node| node.prefix)
            .collect();
        assert_eq!(inner, ["xml"]);
    }

    #[test]
    fn namespace_declarations_are_not_attributes() {
        let document =
            build::parse_str(r#"<a xmlns:p="urn:p" xmlns="urn:d" x="1"/>"#).expect("well-formed");
        let a = document.document_element().expect("a document element");
        let real: Vec<NodeId> = document
            .attributes(a)
            .filter(|attribute| !is_namespace_declaration(&document, *attribute))
            .collect();
        assert_eq!(real.len(), 1);
    }
}
