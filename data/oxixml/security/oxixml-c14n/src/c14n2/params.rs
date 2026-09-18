//! The parameters of Canonical XML 2.0.
//!
//! [Canonical XML 2.0 §2.2](https://www.w3.org/TR/xml-c14n2/#sec-Parameters)
//! replaces the family of algorithms of the earlier versions with a *single*
//! algorithm plus four parameters. When the algorithm is named from an XML
//! Signature, each parameter becomes a child element of
//! `<ds:CanonicalizationMethod>` in the `http://www.w3.org/2010/xml-c14n2`
//! namespace, which is what [`C14n2Parameters::from_method_element`] reads.
//!
//! | parameter | values | default |
//! |---|---|---|
//! | `IgnoreComments` | `true`, `false` | `true` |
//! | `TrimTextNodes` | `true`, `false` | `true` |
//! | `PrefixRewrite` | `none`, `sequential` | `none` |
//! | `QNameAware` | a set of element and attribute names | empty |

use std::collections::BTreeSet;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::error::{C14nError, C14nResult};

/// `http://www.w3.org/2010/xml-c14n2`, the namespace of the parameter elements
/// and the algorithm identifier of Canonical XML 2.0.
pub const C14N2_NAMESPACE: &str = "http://www.w3.org/2010/xml-c14n2";

/// What to do with namespace prefixes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefixRewrite {
    /// Leave the prefixes of the input document unchanged.
    #[default]
    None,
    /// Rewrite every prefix to `n0`, `n1`, `n2`, … with one rewritten prefix per
    /// namespace name, so that the canonical form no longer depends on the
    /// prefixes the author happened to pick. The `xml` prefix is never
    /// rewritten.
    Sequential,
}

impl PrefixRewrite {
    /// Parses the element content of a `<c14n2:PrefixRewrite>` element.
    ///
    /// # Errors
    ///
    /// Returns [`C14nError::InvalidParameter`] for anything but `none` and
    /// `sequential`.
    pub fn parse(value: &str) -> C14nResult<Self> {
        match value.trim() {
            "none" => Ok(Self::None),
            "sequential" => Ok(Self::Sequential),
            other => Err(C14nError::InvalidParameter {
                parameter: "PrefixRewrite",
                value: other.to_owned(),
            }),
        }
    }

    /// Returns the spelling used in a `<c14n2:PrefixRewrite>` element.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sequential => "sequential",
        }
    }
}

/// The set of nodes "whose entire content must be processed as `QName`-valued".
///
/// See [`C14n2Parameters`] and the
/// [Canonical XML 2.0 §2.5.1](https://www.w3.org/TR/xml-c14n2/#sec-Namespace-Concepts)
/// definition of *visible utilization* for what this buys.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QNameAware {
    elements: BTreeSet<(String, String)>,
    xpath_elements: BTreeSet<(String, String)>,
    qualified_attributes: BTreeSet<(String, String)>,
    unqualified_attributes: BTreeSet<(String, String, String)>,
}

impl QNameAware {
    /// Returns the empty set — the default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when nothing is `QName`-aware.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
            && self.xpath_elements.is_empty()
            && self.qualified_attributes.is_empty()
            && self.unqualified_attributes.is_empty()
    }

    /// Declares that the text content of this element is a `QName`.
    pub fn add_element(&mut self, namespace: &str, name: &str) -> &mut Self {
        self.elements
            .insert((namespace.to_owned(), name.to_owned()));
        self
    }

    /// Declares that the text content of this element is an XPath 1.0
    /// expression whose prefixes are visibly utilized.
    pub fn add_xpath_element(&mut self, namespace: &str, name: &str) -> &mut Self {
        self.xpath_elements
            .insert((namespace.to_owned(), name.to_owned()));
        self
    }

    /// Declares that the value of this namespace-qualified attribute is a
    /// `QName`.
    pub fn add_qualified_attribute(&mut self, namespace: &str, name: &str) -> &mut Self {
        self.qualified_attributes
            .insert((namespace.to_owned(), name.to_owned()));
        self
    }

    /// Declares that the value of this unqualified attribute, on elements with
    /// the given name, is a `QName`.
    pub fn add_unqualified_attribute(
        &mut self,
        parent_namespace: &str,
        parent_name: &str,
        name: &str,
    ) -> &mut Self {
        self.unqualified_attributes.insert((
            parent_namespace.to_owned(),
            parent_name.to_owned(),
            name.to_owned(),
        ));
        self
    }

    /// Returns `true` when this element's content is a `QName`.
    #[must_use]
    pub fn is_qname_element(&self, namespace: &str, name: &str) -> bool {
        self.contains(&self.elements, namespace, name)
    }

    /// Returns `true` when this element's content is an XPath expression.
    #[must_use]
    pub fn is_xpath_element(&self, namespace: &str, name: &str) -> bool {
        self.contains(&self.xpath_elements, namespace, name)
    }

    /// Returns `true` when this qualified attribute's value is a `QName`.
    #[must_use]
    pub fn is_qname_qualified_attribute(&self, namespace: &str, name: &str) -> bool {
        self.contains(&self.qualified_attributes, namespace, name)
    }

    /// Returns `true` when this unqualified attribute's value is a `QName` on an
    /// element with the given name.
    #[must_use]
    pub fn is_qname_unqualified_attribute(
        &self,
        parent_namespace: &str,
        parent_name: &str,
        name: &str,
    ) -> bool {
        self.unqualified_attributes
            .iter()
            .any(|(ns, parent, attr)| {
                ns == parent_namespace && parent == parent_name && attr == name
            })
    }

    fn contains(&self, set: &BTreeSet<(String, String)>, namespace: &str, name: &str) -> bool {
        set.iter()
            .any(|(ns, local)| ns == namespace && local == name)
    }
}

/// The complete parameter set of one Canonical XML 2.0 run.
///
/// # Example
///
/// ```
/// use oxixml_c14n::{C14n2Parameters, PrefixRewrite};
///
/// let parameters = C14n2Parameters::new()
///     .with_prefix_rewrite(PrefixRewrite::Sequential)
///     .with_trim_text_nodes(false);
/// assert!(parameters.ignore_comments());
/// assert!(!parameters.trim_text_nodes());
/// assert_eq!(parameters.prefix_rewrite(), PrefixRewrite::Sequential);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C14n2Parameters {
    ignore_comments: bool,
    trim_text_nodes: bool,
    prefix_rewrite: PrefixRewrite,
    qname_aware: QNameAware,
}

impl Default for C14n2Parameters {
    fn default() -> Self {
        Self {
            ignore_comments: true,
            trim_text_nodes: true,
            prefix_rewrite: PrefixRewrite::None,
            qname_aware: QNameAware::new(),
        }
    }
}

impl C14n2Parameters {
    /// Returns the defaults of
    /// [§2.2](https://www.w3.org/TR/xml-c14n2/#sec-Parameters).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether comment nodes are dropped.
    #[must_use]
    pub const fn with_ignore_comments(mut self, ignore: bool) -> Self {
        self.ignore_comments = ignore;
        self
    }

    /// Sets whether the leading and trailing white space of every text node is
    /// removed.
    #[must_use]
    pub const fn with_trim_text_nodes(mut self, trim: bool) -> Self {
        self.trim_text_nodes = trim;
        self
    }

    /// Sets the prefix rewriting mode.
    #[must_use]
    pub const fn with_prefix_rewrite(mut self, rewrite: PrefixRewrite) -> Self {
        self.prefix_rewrite = rewrite;
        self
    }

    /// Sets the `QNameAware` node set.
    #[must_use]
    pub fn with_qname_aware(mut self, qname_aware: QNameAware) -> Self {
        self.qname_aware = qname_aware;
        self
    }

    /// Returns `true` when comments are dropped.
    #[must_use]
    pub const fn ignore_comments(&self) -> bool {
        self.ignore_comments
    }

    /// Returns `true` when text nodes are trimmed.
    #[must_use]
    pub const fn trim_text_nodes(&self) -> bool {
        self.trim_text_nodes
    }

    /// Returns the prefix rewriting mode.
    #[must_use]
    pub const fn prefix_rewrite(&self) -> PrefixRewrite {
        self.prefix_rewrite
    }

    /// Returns the `QNameAware` node set.
    #[must_use]
    pub const fn qname_aware(&self) -> &QNameAware {
        &self.qname_aware
    }

    /// Reads the parameters out of a `<ds:CanonicalizationMethod>` element.
    ///
    /// Only the children in the `http://www.w3.org/2010/xml-c14n2` namespace are
    /// looked at, and they may appear in any order; anything else is ignored, as
    /// are the `Algorithm` attribute and the element's own name.
    ///
    /// # Errors
    ///
    /// Returns [`C14nError::NotAnElement`] when `method` is not an element, and
    /// [`C14nError::InvalidParameter`] when a parameter carries a value the
    /// Recommendation does not define.
    pub fn from_method_element(document: &Document, method: NodeId) -> C14nResult<Self> {
        match document.kind(method) {
            Some(NodeKind::Element) => {}
            Some(kind) => return Err(C14nError::NotAnElement { node: method, kind }),
            None => return Err(C14nError::UnknownNode { node: method }),
        }

        let mut parameters = Self::new();
        for child in document.children(method) {
            let Some(element) = document.element(child) else {
                continue;
            };
            if element.namespace_uri() != Some(C14N2_NAMESPACE) {
                continue;
            }
            let text = document.text_content(child).unwrap_or_default();
            match element.local_name() {
                "IgnoreComments" => {
                    parameters.ignore_comments = parse_boolean("IgnoreComments", &text)?;
                }
                "TrimTextNodes" => {
                    parameters.trim_text_nodes = parse_boolean("TrimTextNodes", &text)?;
                }
                "PrefixRewrite" => {
                    parameters.prefix_rewrite = PrefixRewrite::parse(&text)?;
                }
                "QNameAware" => {
                    parameters.qname_aware = parse_qname_aware(document, child);
                }
                _ => {}
            }
        }
        Ok(parameters)
    }
}

fn parse_boolean(parameter: &'static str, value: &str) -> C14nResult<bool> {
    match value.trim() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        other => Err(C14nError::InvalidParameter {
            parameter,
            value: other.to_owned(),
        }),
    }
}

fn parse_qname_aware(document: &Document, element: NodeId) -> QNameAware {
    let mut aware = QNameAware::new();
    for child in document.children(element) {
        let Some(data) = document.element(child) else {
            continue;
        };
        if data.namespace_uri() != Some(C14N2_NAMESPACE) {
            continue;
        }
        let name = document.get_attribute(child, "Name").unwrap_or_default();
        let namespace = document.get_attribute(child, "NS").unwrap_or_default();
        match data.local_name() {
            "Element" => {
                aware.add_element(namespace, name);
            }
            "XPathElement" => {
                aware.add_xpath_element(namespace, name);
            }
            "QualifiedAttr" => {
                aware.add_qualified_attribute(namespace, name);
            }
            "UnqualifiedAttr" => {
                let parent_name = document
                    .get_attribute(child, "ParentName")
                    .unwrap_or_default();
                let parent_namespace = document
                    .get_attribute(child, "ParentNS")
                    .unwrap_or_default();
                aware.add_unqualified_attribute(parent_namespace, parent_name, name);
            }
            _ => {}
        }
    }
    aware
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    #[test]
    fn reads_a_canonicalization_method_element() -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str(
            r#"<dsig:CanonicalizationMethod
                 xmlns:dsig="http://www.w3.org/2000/09/xmldsig#"
                 xmlns:c14n2="http://www.w3.org/2010/xml-c14n2"
                 Algorithm="http://www.w3.org/2010/xml-c14n2">
                 <c14n2:IgnoreComments>false</c14n2:IgnoreComments>
                 <c14n2:TrimTextNodes>true</c14n2:TrimTextNodes>
                 <c14n2:PrefixRewrite>sequential</c14n2:PrefixRewrite>
                 <c14n2:QNameAware>
                   <c14n2:QualifiedAttr Name="type" NS="http://www.w3.org/2001/XMLSchema-instance"/>
                   <c14n2:Element Name="bar" NS="http://a"/>
                   <c14n2:XPathElement Name="IncludedXPath" NS="http://www.w3.org/2010/xmldsig2#"/>
                   <c14n2:UnqualifiedAttr Name="ref" ParentName="thing" ParentNS="urn:p"/>
                 </c14n2:QNameAware>
               </dsig:CanonicalizationMethod>"#,
        )?;
        let method = doc.document_element().ok_or("no document element")?;
        let parameters = C14n2Parameters::from_method_element(&doc, method)?;

        assert!(!parameters.ignore_comments());
        assert!(parameters.trim_text_nodes());
        assert_eq!(parameters.prefix_rewrite(), PrefixRewrite::Sequential);

        let aware = parameters.qname_aware();
        assert!(aware.is_qname_element("http://a", "bar"));
        assert!(aware.is_xpath_element("http://www.w3.org/2010/xmldsig2#", "IncludedXPath"));
        assert!(
            aware.is_qname_qualified_attribute("http://www.w3.org/2001/XMLSchema-instance", "type")
        );
        assert!(aware.is_qname_unqualified_attribute("urn:p", "thing", "ref"));
        assert!(!aware.is_qname_element("http://b", "bar"));
        Ok(())
    }

    #[test]
    fn an_empty_method_element_yields_the_defaults() -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str(
            r#"<CanonicalizationMethod Algorithm="http://www.w3.org/2010/xml-c14n2"/>"#,
        )?;
        let method = doc.document_element().ok_or("no document element")?;
        let parameters = C14n2Parameters::from_method_element(&doc, method)?;
        assert_eq!(parameters, C14n2Parameters::new());
        assert!(parameters.qname_aware().is_empty());
        Ok(())
    }

    #[test]
    fn bad_values_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str(
            r#"<m xmlns:c="http://www.w3.org/2010/xml-c14n2"><c:PrefixRewrite>random</c:PrefixRewrite></m>"#,
        )?;
        let method = doc.document_element().ok_or("no document element")?;
        let error = C14n2Parameters::from_method_element(&doc, method)
            .expect_err("`random` is not a PrefixRewrite value");
        assert_eq!(
            error,
            C14nError::InvalidParameter {
                parameter: "PrefixRewrite",
                value: "random".to_owned()
            }
        );
        Ok(())
    }
}
