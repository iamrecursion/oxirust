//! In-scope namespace bindings and the namespace axis.
//!
//! `oxixml-dom` keeps namespace declarations as ordinary attribute nodes in the
//! `http://www.w3.org/2000/xmlns/` namespace, exactly as DOM Level 3 Core does.
//! Canonicalization instead needs XPath's *namespace axis*: for every element,
//! the set of prefix → namespace-name bindings that are in scope there,
//! including the ones inherited from ancestors and the permanent binding of the
//! `xml` prefix.
//!
//! [`Bindings`] is that axis. It is built once for the topmost element being
//! visited and then extended cheaply while walking down the tree.

use std::collections::{BTreeMap, BTreeSet};

use oxixml_dom::{Document, NodeId, XML_NAMESPACE, XMLNS_NAMESPACE};

use crate::error::{C14nError, C14nResult};

/// The prefix used to key the default namespace declaration, `xmlns="…"`.
pub(crate) const DEFAULT_PREFIX: &str = "";

/// The `xml` prefix, which is bound to [`XML_NAMESPACE`] everywhere and is never
/// rendered by any of the algorithms.
pub(crate) const XML_PREFIX: &str = "xml";

/// The in-scope namespace bindings of one element: the XPath namespace axis.
///
/// The map is keyed by prefix, with `""` standing for the default namespace.
/// A prefix that has been *undeclared* (`xmlns:p=""`, or `xmlns=""`) is absent
/// from the map, because XPath has no namespace node for it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Bindings {
    map: BTreeMap<String, String>,
}

impl Bindings {
    /// The bindings in force outside any element: only the `xml` prefix.
    pub(crate) fn root() -> Self {
        let mut map = BTreeMap::new();
        map.insert(XML_PREFIX.to_owned(), XML_NAMESPACE.to_owned());
        Self { map }
    }

    /// Builds the bindings in force *at* `element`, walking the real document
    /// ancestors — used to seed the traversal at an apex node whose ancestors
    /// are not part of the subset.
    pub(crate) fn at(document: &Document, element: NodeId) -> Self {
        let mut chain: Vec<NodeId> = document.ancestors(element).collect();
        chain.reverse();
        chain.push(element);
        let mut bindings = Self::root();
        for node in chain {
            bindings.declare_from(document, node);
        }
        bindings
    }

    /// Returns the bindings that are in force inside `element`, i.e. `self`
    /// updated with the namespace declarations written on `element` itself.
    pub(crate) fn child(&self, document: &Document, element: NodeId) -> Self {
        let mut next = self.clone();
        next.declare_from(document, element);
        next
    }

    fn declare_from(&mut self, document: &Document, element: NodeId) {
        for attribute in document.attributes(element) {
            let Some(data) = document.attr(attribute) else {
                continue;
            };
            let Some(prefix) =
                declared_prefix(data.namespace_uri(), data.prefix(), data.local_name())
            else {
                continue;
            };
            if data.value().is_empty() {
                self.map.remove(prefix);
            } else {
                self.map.insert(prefix.to_owned(), data.value().to_owned());
            }
        }
    }

    /// Returns the namespace name bound to `prefix`, if any.
    pub(crate) fn get(&self, prefix: &str) -> Option<&str> {
        self.map.get(prefix).map(String::as_str)
    }

    /// Returns the namespace name of the default namespace, or `""` when there
    /// is none in scope.
    pub(crate) fn default_namespace(&self) -> &str {
        self.get(DEFAULT_PREFIX).unwrap_or("")
    }

    /// Iterates over the namespace axis in ascending prefix order, which is the
    /// order Canonical XML renders namespace nodes in (the default namespace
    /// node has no local name and is therefore least).
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map
            .iter()
            .map(|(prefix, namespace)| (prefix.as_str(), namespace.as_str()))
    }
}

/// Returns the prefix a namespace-declaration attribute declares, or `None`
/// when the attribute is an ordinary one.
///
/// `xmlns="…"` declares the default namespace and yields `Some("")`;
/// `xmlns:p="…"` yields `Some("p")`.
pub(crate) fn declared_prefix<'a>(
    namespace: Option<&str>,
    prefix: Option<&'a str>,
    local_name: &'a str,
) -> Option<&'a str> {
    if namespace != Some(XMLNS_NAMESPACE) {
        return None;
    }
    match prefix {
        Some("xmlns") => Some(local_name),
        None if local_name == "xmlns" => Some(DEFAULT_PREFIX),
        _ => None,
    }
}

/// Returns `true` when `attribute` is a namespace declaration rather than an
/// ordinary attribute.
pub(crate) fn is_namespace_declaration(document: &Document, attribute: NodeId) -> bool {
    document.attr(attribute).is_some_and(|data| {
        declared_prefix(data.namespace_uri(), data.prefix(), data.local_name()).is_some()
    })
}

/// Returns the set of prefixes on the namespace axis of `element`, used to seed
/// an explicit namespace-node selection in [`NodeSet`](crate::NodeSet).
pub(crate) fn in_scope_prefixes(document: &Document, element: NodeId) -> BTreeSet<String> {
    Bindings::at(document, element)
        .iter()
        .map(|(prefix, _)| prefix.to_owned())
        .collect()
}

/// Rejects a relative namespace URI.
///
/// [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel): "implementations
/// of XML canonicalization MUST report an operation failure on documents
/// containing relative namespace URIs". A namespace name is relative when it is
/// not empty and does not begin with an RFC 3986 `scheme` followed by a colon.
pub(crate) fn check_absolute(prefix: &str, namespace: &str) -> C14nResult<()> {
    if namespace.is_empty() || is_absolute_uri(namespace) {
        Ok(())
    } else {
        Err(C14nError::RelativeNamespaceUri {
            prefix: prefix.to_owned(),
            namespace: namespace.to_owned(),
        })
    }
}

/// Returns `true` when `value` starts with an RFC 3986 `scheme ":"`.
fn is_absolute_uri(value: &str) -> bool {
    let mut chars = value.char_indices();
    match chars.next() {
        Some((_, first)) if first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    for (_, ch) in chars {
        if ch == ':' {
            return true;
        }
        if !(ch.is_ascii_alphanumeric() || ch == '+' || ch == '-' || ch == '.') {
            return false;
        }
    }
    false
}

/// Renders a namespace node's qualified name: `xmlns` or `xmlns:prefix`.
pub(crate) fn namespace_node_name(prefix: &str) -> String {
    if prefix.is_empty() {
        "xmlns".to_owned()
    } else {
        format!("xmlns:{prefix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    #[test]
    fn absolute_uris_need_a_scheme() {
        assert!(is_absolute_uri("http://example.com"));
        assert!(is_absolute_uri("urn:x"));
        assert!(is_absolute_uri("a:b"));
        assert!(is_absolute_uri("foo+bar-baz.qux:"));
        assert!(!is_absolute_uri("../relative"));
        assert!(!is_absolute_uri("relative"));
        assert!(!is_absolute_uri("9nope:x"));
        assert!(!is_absolute_uri("/absolute/path"));
    }

    #[test]
    fn undeclaration_removes_the_namespace_node() -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str("<a xmlns='urn:a'><b xmlns=''/></a>")?;
        let a = doc.document_element().ok_or("no document element")?;
        let b = doc.children(a).next().ok_or("no child")?;
        assert_eq!(Bindings::at(&doc, a).default_namespace(), "urn:a");
        assert_eq!(Bindings::at(&doc, b).default_namespace(), "");
        Ok(())
    }

    #[test]
    fn the_xml_prefix_is_always_bound() {
        assert_eq!(Bindings::root().get(XML_PREFIX), Some(XML_NAMESPACE));
    }
}
