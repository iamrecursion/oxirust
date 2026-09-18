//! The list of resolved attributes passed to
//! [`ContentHandler::start_element`](crate::ContentHandler::start_element).
//!
//! Mirrors `org.xml.sax.Attributes`, translated to an owned, indexable Rust
//! collection (rather than an interface queried by index).

use std::slice::Iter;

/// A single attribute reported by [`start_element`](crate::ContentHandler::start_element).
///
/// The value has already been normalized per the [XML specification's attribute
/// value normalization rules]: `\t`, `\r` and `\n` are replaced by a single space
/// (`0x20`) and the five predefined entities (`&lt;`, `&gt;`, `&amp;`, `&apos;`,
/// `&quot;`) are resolved.
///
/// [XML specification's attribute value normalization rules]: https://www.w3.org/TR/xml11/#AVNormalize
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    /// The resolved namespace URI, or `None` if the attribute name has no prefix
    /// (an unprefixed attribute never inherits the default namespace) or if
    /// namespace processing is disabled.
    pub uri: Option<String>,
    /// The local (unprefixed) part of the attribute name.
    pub local_name: String,
    /// The attribute name as written in the source (`prefix:local` or `local`).
    pub qname: String,
    /// The normalized attribute value.
    pub value: String,
    /// The declared attribute type (`"CDATA"`, `"ID"`, `"NMTOKEN"`, …). Because
    /// this reader does not perform DTD-based validation, this is always
    /// `"CDATA"`, matching the SAX2 convention for parsers that do not read an
    /// attribute-list declaration for the attribute.
    pub attr_type: String,
}

impl Attribute {
    /// The attribute type reported when no DTD-driven type information is
    /// available.
    pub(crate) const CDATA: &str = "CDATA";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// An ordered, indexable list of an element's [`Attribute`]s, in the order they
/// appeared in the source.
///
/// Namespace declaration attributes (`xmlns` and `xmlns:*`) are omitted unless
/// [`Feature::NamespacePrefixes`](crate::Feature::NamespacePrefixes) is enabled,
/// mirroring the SAX2 `namespace-prefixes` feature.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Attributes(Vec<Attribute>);

impl Attributes {
    /// Creates an empty attribute list.
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn push(&mut self, attribute: Attribute) {
        self.0.push(attribute);
    }

    /// The number of attributes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if there are no attributes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the attribute at `index`, if any.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Attribute> {
        self.0.get(index)
    }

    /// Iterates over the attributes in document order.
    pub fn iter(&self) -> Iter<'_, Attribute> {
        self.0.iter()
    }

    /// Finds the index of the attribute with the given raw qualified name
    /// (`prefix:local` or `local`), if any.
    #[must_use]
    pub fn index_by_qname(&self, qname: &str) -> Option<usize> {
        self.0.iter().position(|a| a.qname == qname)
    }

    /// Finds the index of the attribute with the given resolved namespace URI and
    /// local name, if any.
    #[must_use]
    pub fn index_by_name(&self, uri: Option<&str>, local_name: &str) -> Option<usize> {
        self.0
            .iter()
            .position(|a| a.uri.as_deref() == uri && a.local_name == local_name)
    }

    /// Returns the value of the attribute with the given raw qualified name, if any.
    #[must_use]
    pub fn value_by_qname(&self, qname: &str) -> Option<&str> {
        self.index_by_qname(qname).map(|i| self.0[i].value.as_str())
    }

    /// Returns the value of the attribute with the given resolved namespace URI
    /// and local name, if any.
    #[must_use]
    pub fn value_by_name(&self, uri: Option<&str>, local_name: &str) -> Option<&str> {
        self.index_by_name(uri, local_name)
            .map(|i| self.0[i].value.as_str())
    }
}

impl<'a> IntoIterator for &'a Attributes {
    type Item = &'a Attribute;
    type IntoIter = Iter<'a, Attribute>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attr(qname: &str, local_name: &str, uri: Option<&str>, value: &str) -> Attribute {
        Attribute {
            uri: uri.map(str::to_string),
            local_name: local_name.to_string(),
            qname: qname.to_string(),
            value: value.to_string(),
            attr_type: Attribute::CDATA.to_string(),
        }
    }

    #[test]
    fn empty_list() {
        let attributes = Attributes::new();
        assert_eq!(attributes.len(), 0);
        assert!(attributes.is_empty());
        assert_eq!(attributes.get(0), None);
    }

    #[test]
    fn lookup_by_qname_and_resolved_name() {
        let mut attributes = Attributes::new();
        attributes.push(attr("id", "id", None, "42"));
        attributes.push(attr("x:href", "href", Some("urn:x"), "target"));

        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes.value_by_qname("id"), Some("42"));
        assert_eq!(attributes.value_by_qname("missing"), None);
        assert_eq!(
            attributes.value_by_name(Some("urn:x"), "href"),
            Some("target")
        );
        assert_eq!(attributes.value_by_name(None, "href"), None);
        assert_eq!(attributes.index_by_qname("x:href"), Some(1));
    }

    #[test]
    fn iteration_preserves_document_order() {
        let mut attributes = Attributes::new();
        attributes.push(attr("a", "a", None, "1"));
        attributes.push(attr("b", "b", None, "2"));
        let names: Vec<&str> = attributes.iter().map(|a| a.qname.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);

        let names: Vec<&str> = (&attributes)
            .into_iter()
            .map(|a| a.qname.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b"]);
    }
}
