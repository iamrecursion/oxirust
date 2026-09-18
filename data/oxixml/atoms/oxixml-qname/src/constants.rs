//! Well-known namespace names fixed by [Namespaces in XML 1.0/1.1](https://www.w3.org/TR/xml-names/).

/// The namespace name implicitly bound to the reserved `xml` prefix in every namespace-aware
/// XML document.
///
/// Per [Namespaces in XML §3](https://www.w3.org/TR/xml-names/#ns-decl): "the prefix `xml` is
/// by definition bound to the namespace name `http://www.w3.org/XML/1998/namespace`. It MAY,
/// but need not, be declared, and MUST NOT be bound to any other namespace name. Other prefixes
/// MUST NOT be bound to this namespace name, and it MUST NOT be declared as the default
/// namespace."
///
/// # Examples
///
/// ```
/// use oxixml_qname::{NameContext, NamespaceStack, NamespacesVersion, XML_NAMESPACE};
///
/// let namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
/// // `xml:lang` resolves even though nobody declared it.
/// let expanded = namespaces.resolve_qname("xml:lang", NameContext::Attribute)?;
/// assert_eq!(expanded.namespace(), Some(XML_NAMESPACE));
/// # Ok::<(), oxixml_qname::ResolveQNameError>(())
/// ```
pub const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";

/// The namespace name implicitly bound to the reserved `xmlns` prefix.
///
/// This namespace name exists only so that namespace *declaration* attributes
/// (`xmlns:prefix="..."` / `xmlns="..."`) have a namespace name of their own for APIs that
/// expose attributes uniformly (DOM, SAX2). The `xmlns` prefix itself MUST NOT be declared, and
/// no other prefix (nor the default namespace) may be bound to this namespace name.
///
/// # Examples
///
/// ```
/// use oxixml_qname::{NamespaceStack, NamespacesVersion, XMLNS_NAMESPACE};
///
/// let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);
/// // Nothing may bind to the `xmlns` namespace name, not even a fresh prefix.
/// assert!(namespaces.declare(Some("x"), XMLNS_NAMESPACE).is_err());
/// ```
pub const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";
