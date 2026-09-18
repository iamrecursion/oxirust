//! The mutable element tree that the simplification rules of
//! [§4](https://relaxng.org/spec-20011203.html#simplification) operate on.
//!
//! RELAX NG defines simplification "at the data model level": the schema is
//! first parsed into the data model of
//! [§2](https://relaxng.org/spec-20011203.html#data-model) — elements with a
//! name, a context (base URI plus namespace map), a set of attributes and a
//! sequence of children that are elements or strings — and the transformation
//! rules then rewrite that tree. This module is exactly that data model,
//! specialized to the fact that after rule 4.1 every remaining element is in the
//! RELAX NG namespace, so only its local name matters.
//!
//! The compact-syntax parser produces the very same tree, which is why both
//! syntaxes share the whole downstream pipeline.

use std::rc::Rc;

use crate::error::Position;

/// The RELAX NG structure namespace.
pub(crate) const RNG_NS: &str = "http://relaxng.org/ns/structure/1.0";

/// The namespace of namespace-declaration attributes, per the XML Infoset.
pub(crate) const INFOSET_XMLNS_NS: &str = "http://www.w3.org/2000/xmlns";

/// A namespace map: the prefix bindings in scope at some element, plus the
/// default namespace.
///
/// This is the part of an element's *context* that QName resolution (rule 4.10)
/// and the `QName`/`NOTATION` datatypes of the W3C XML Schema library need.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NamespaceMap {
    bindings: Vec<(String, String)>,
    default_namespace: Option<String>,
}

impl NamespaceMap {
    /// The map in force at the top of every XML document: `xml` is bound
    /// implicitly, nothing else is.
    pub(crate) fn initial() -> Self {
        Self {
            bindings: vec![("xml".to_owned(), oxixml_qname::XML_NAMESPACE.to_owned())],
            default_namespace: None,
        }
    }

    /// Binds `prefix` to `namespace`, replacing any previous binding.
    ///
    /// Binding to the empty string *removes* the binding, as `xmlns:p=""` does
    /// in XML 1.1 (and as the compact syntax's `inherit` handling requires).
    pub(crate) fn bind(&mut self, prefix: &str, namespace: &str) {
        self.bindings.retain(|(p, _)| p != prefix);
        if !namespace.is_empty() {
            self.bindings
                .push((prefix.to_owned(), namespace.to_owned()));
        }
    }

    /// Sets the default namespace, or clears it when `namespace` is empty.
    pub(crate) fn set_default(&mut self, namespace: &str) {
        self.default_namespace = if namespace.is_empty() {
            None
        } else {
            Some(namespace.to_owned())
        };
    }

    /// Returns the namespace bound to `prefix`, if any.
    pub(crate) fn lookup(&self, prefix: &str) -> Option<&str> {
        self.bindings
            .iter()
            .find(|(p, _)| p == prefix)
            .map(|(_, ns)| ns.as_str())
    }

    /// Returns the default namespace, if one is in scope.
    pub(crate) fn default_namespace(&self) -> Option<&str> {
        self.default_namespace.as_deref()
    }

    /// Returns a copy of this map whose default namespace is `namespace`
    /// (or absent when `namespace` is empty) — the `context(u, cx)` operation of
    /// [§6.2.8](https://relaxng.org/spec-20011203.html#IDA0YZR).
    pub(crate) fn with_default(&self, namespace: &str) -> Self {
        let mut copy = self.clone();
        copy.set_default(namespace);
        copy
    }
}

/// The context of an element: its base URI and its namespace map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Context {
    /// The base URI in force, when one is known.
    pub base: Option<Rc<str>>,
    /// The namespace map in force.
    pub namespaces: Rc<NamespaceMap>,
}

impl Context {
    pub(crate) fn new(base: Option<Rc<str>>, namespaces: Rc<NamespaceMap>) -> Self {
        Self { base, namespaces }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            base: None,
            namespaces: Rc::new(NamespaceMap::initial()),
        }
    }
}

/// An attribute of a schema element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Attribute {
    /// The namespace name, or the empty string when the attribute is unprefixed.
    pub namespace: String,
    /// The local name.
    pub local: String,
    /// The normalized value.
    pub value: String,
}

impl Attribute {
    pub(crate) fn new(local: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            namespace: String::new(),
            local: local.into(),
            value: value.into(),
        }
    }
}

/// A child of a schema element: either a nested element or a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Child {
    /// A nested element.
    Element(Element),
    /// A maximal sequence of characters.
    Text(String),
}

/// An element of the schema, in the RELAX NG namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Element {
    /// The local name; the namespace is always [`RNG_NS`].
    pub name: String,
    /// The attributes, in document order.
    pub attributes: Vec<Attribute>,
    /// The children, in document order.
    pub children: Vec<Child>,
    /// The context (base URI and namespace map) of this element.
    pub context: Context,
    /// Where the element started in its source, for diagnostics.
    pub position: Position,
}

impl Element {
    /// Builds an element with no attributes and no children.
    pub(crate) fn new(name: impl Into<String>, context: Context, position: Position) -> Self {
        Self {
            name: name.into(),
            attributes: Vec::new(),
            children: Vec::new(),
            context,
            position,
        }
    }

    /// Builds an element that inherits `template`'s context and position.
    pub(crate) fn like(template: &Self, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: Vec::new(),
            children: Vec::new(),
            context: template.context.clone(),
            position: template.position,
        }
    }

    /// Returns the value of the unprefixed attribute `local`, if present.
    pub(crate) fn attribute(&self, local: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|attribute| attribute.namespace.is_empty() && attribute.local == local)
            .map(|attribute| attribute.value.as_str())
    }

    /// Sets (or adds) the unprefixed attribute `local`.
    pub(crate) fn set_attribute(&mut self, local: &str, value: impl Into<String>) {
        let value = value.into();
        if let Some(existing) = self
            .attributes
            .iter_mut()
            .find(|attribute| attribute.namespace.is_empty() && attribute.local == local)
        {
            existing.value = value;
        } else {
            self.attributes.push(Attribute::new(local, value));
        }
    }

    /// Removes the unprefixed attribute `local`, returning its value.
    pub(crate) fn remove_attribute(&mut self, local: &str) -> Option<String> {
        let index = self
            .attributes
            .iter()
            .position(|attribute| attribute.namespace.is_empty() && attribute.local == local)?;
        Some(self.attributes.remove(index).value)
    }

    /// Iterates over the child elements, skipping strings.
    pub(crate) fn child_elements(&self) -> impl Iterator<Item = &Self> {
        self.children.iter().filter_map(|child| match child {
            Child::Element(element) => Some(element),
            Child::Text(_) => None,
        })
    }

    /// Iterates mutably over the child elements, skipping strings.
    pub(crate) fn child_elements_mut(&mut self) -> impl Iterator<Item = &mut Self> {
        self.children.iter_mut().filter_map(|child| match child {
            Child::Element(element) => Some(element),
            Child::Text(_) => None,
        })
    }

    /// Returns the child elements as a vector, consuming the children and
    /// discarding any strings.
    pub(crate) fn take_child_elements(&mut self) -> Vec<Self> {
        std::mem::take(&mut self.children)
            .into_iter()
            .filter_map(|child| match child {
                Child::Element(element) => Some(element),
                Child::Text(_) => None,
            })
            .collect()
    }

    /// Replaces the children with the given elements.
    pub(crate) fn set_child_elements(&mut self, elements: Vec<Self>) {
        self.children = elements.into_iter().map(Child::Element).collect();
    }

    /// Concatenates every string child, ignoring nested elements.
    pub(crate) fn text(&self) -> String {
        let mut out = String::new();
        for child in &self.children {
            if let Child::Text(text) = child {
                out.push_str(text);
            }
        }
        out
    }

    /// Returns `true` when the element has at least one string child.
    #[cfg(test)]
    pub(crate) fn has_text_child(&self) -> bool {
        self.children
            .iter()
            .any(|child| matches!(child, Child::Text(_)))
    }

    /// Returns `true` when this element or any descendant has the local name
    /// `name`.
    pub(crate) fn has_descendant_or_self(&self, name: &str) -> bool {
        self.name == name
            || self
                .child_elements()
                .any(|c| c.has_descendant_or_self(name))
    }

    /// Returns `true` when any descendant (not this element) has local name
    /// `name`.
    pub(crate) fn has_descendant(&self, name: &str) -> bool {
        self.child_elements()
            .any(|c| c.has_descendant_or_self(name))
    }
}

/// Returns `true` when `s` consists solely of XML whitespace (`#x20`, `#x9`,
/// `#xD`, `#xA`), including when it is empty.
#[must_use]
pub(crate) fn is_whitespace(s: &str) -> bool {
    s.bytes().all(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
}

/// Removes leading and trailing XML whitespace.
#[must_use]
pub(crate) fn trim_whitespace(s: &str) -> &str {
    s.trim_matches(|c| matches!(c, ' ' | '\t' | '\r' | '\n'))
}

/// The `normalizeWhiteSpace` function of
/// [§6.2.9](https://relaxng.org/spec-20011203.html#IDAK1ZR): trims, then
/// collapses each internal run of whitespace to a single space.
#[must_use]
pub(crate) fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for c in trim_whitespace(s).chars() {
        if matches!(c, ' ' | '\t' | '\r' | '\n') {
            pending_space = true;
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(c);
        }
    }
    out
}

/// The `split` function of
/// [§6.2.10](https://relaxng.org/spec-20011203.html#IDA43ZR): the whitespace
/// delimited, non-empty tokens of `s`.
pub(crate) fn split_tokens(s: &str) -> Vec<&str> {
    s.split([' ', '\t', '\r', '\n'])
        .filter(|token| !token.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_helpers() {
        assert!(is_whitespace(""));
        assert!(is_whitespace(" \t\r\n"));
        assert!(!is_whitespace(" a "));
        assert_eq!(trim_whitespace("\n foo \t"), "foo");
        assert_eq!(normalize_whitespace(" a \n b  c "), "a b c");
        assert_eq!(split_tokens("  a  b\nc "), vec!["a", "b", "c"]);
        assert!(split_tokens("   ").is_empty());
    }

    #[test]
    fn namespace_map_binding() {
        let mut map = NamespaceMap::initial();
        assert_eq!(map.lookup("xml"), Some(oxixml_qname::XML_NAMESPACE));
        map.bind("p", "urn:example");
        assert_eq!(map.lookup("p"), Some("urn:example"));
        map.bind("p", "");
        assert_eq!(map.lookup("p"), None);
        map.set_default("urn:default");
        assert_eq!(map.default_namespace(), Some("urn:default"));
        assert_eq!(map.with_default("").default_namespace(), None);
    }

    #[test]
    fn element_attribute_access() {
        let mut element = Element::new("element", Context::default(), Position::UNKNOWN);
        element.set_attribute("name", "foo");
        assert_eq!(element.attribute("name"), Some("foo"));
        element.set_attribute("name", "bar");
        assert_eq!(element.attribute("name"), Some("bar"));
        assert_eq!(element.remove_attribute("name").as_deref(), Some("bar"));
        assert_eq!(element.attribute("name"), None);
    }
}
