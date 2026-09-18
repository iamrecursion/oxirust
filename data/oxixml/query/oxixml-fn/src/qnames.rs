//! The `xs:QName` functions of
//! [F&O 3.1 §11](https://www.w3.org/TR/xpath-functions-31/#QName-funcs): construction
//! (`fn:resolve-QName`, `fn:QName`), decomposition (`fn:prefix-from-QName`,
//! `fn:local-name-from-QName`, `fn:namespace-uri-from-QName`) and namespace lookup
//! (`fn:namespace-uri-for-prefix`, `fn:in-scope-prefixes`).

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicType, AtomicValue, NodeKind, NodeRef, QNameValue, XdmNode};
use oxixml_qname::{XML_NAMESPACE, is_ncname};

/// The in-scope namespace bindings of an element node, as `(prefix, namespace)` pairs with
/// `None` standing for the default namespace.
///
/// The `xml` prefix is always in scope, as Namespaces in XML requires.
#[must_use]
pub fn in_scope_namespaces(element: &NodeRef) -> Vec<(Option<String>, String)> {
    let mut bindings: Vec<(Option<String>, String)> = element
        .namespaces()
        .into_iter()
        .map(|node| {
            let prefix = node.node_name().map(|name| name.local_name().to_owned());
            (prefix, node.string_value())
        })
        .collect();
    if !bindings
        .iter()
        .any(|(prefix, _)| prefix.as_deref() == Some("xml"))
    {
        bindings.push((Some("xml".to_owned()), XML_NAMESPACE.to_owned()));
    }
    bindings
}

/// [`fn:in-scope-prefixes`](https://www.w3.org/TR/xpath-functions-31/#func-in-scope-prefixes):
/// the prefixes bound on an element, with the zero-length string standing for the default
/// namespace.
///
/// # Examples
///
/// ```
/// use oxixml_fn::qnames::in_scope_prefixes;
/// use oxixml_infoset::Tree;
///
/// let tree = Tree::parse_str("<a xmlns='urn:d' xmlns:p='urn:p'/>")?;
/// let root = tree.document_element().expect("a document element");
/// let mut prefixes = in_scope_prefixes(&root);
/// prefixes.sort();
/// assert_eq!(prefixes, vec!["".to_owned(), "p".to_owned(), "xml".to_owned()]);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn in_scope_prefixes(element: &NodeRef) -> Vec<String> {
    in_scope_namespaces(element)
        .into_iter()
        .map(|(prefix, _)| prefix.unwrap_or_default())
        .collect()
}

/// [`fn:namespace-uri-for-prefix`](https://www.w3.org/TR/xpath-functions-31/#func-namespace-uri-for-prefix).
///
/// A zero-length or absent prefix selects the default namespace; a prefix that is not bound
/// yields the empty sequence.
#[must_use]
pub fn namespace_uri_for_prefix(prefix: Option<&str>, element: &NodeRef) -> Option<String> {
    let wanted = prefix.filter(|prefix| !prefix.is_empty());
    in_scope_namespaces(element)
        .into_iter()
        .find(|(bound, _)| bound.as_deref() == wanted)
        .map(|(_, namespace)| namespace)
        .filter(|namespace| !namespace.is_empty())
}

/// [`fn:resolve-QName`](https://www.w3.org/TR/xpath-functions-31/#func-resolve-QName): resolves
/// a lexical `QName` against the in-scope namespaces of `element`.
///
/// # Errors
///
/// [`FnErrorCode::Foca0002`] when the string does not match the `QName` production — F&O 3.1
/// §11.1.1 makes this the *function's* code, not the cast code `FORG0001` — and
/// [`FnErrorCode::Fons0004`] when its prefix is not bound.
///
/// # Examples
///
/// ```
/// use oxixml_fn::qnames::resolve_qname;
/// use oxixml_infoset::Tree;
///
/// let tree = Tree::parse_str("<a xmlns:p='urn:p'/>")?;
/// let root = tree.document_element().expect("a document element");
///
/// let resolved = resolve_qname("p:name", &root)?;
/// assert_eq!(resolved.prefix(), Some("p"));
/// assert_eq!(resolved.namespace(), Some("urn:p"));
/// assert_eq!(resolved.local_name(), "name");
///
/// // An unbound prefix is FONS0004.
/// assert!(resolve_qname("q:name", &root).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn resolve_qname(lexical: &str, element: &NodeRef) -> FnResult<QNameValue> {
    let collapsed = lexical.trim();
    let (prefix, local) = match collapsed.split_once(':') {
        Some((prefix, local)) => (Some(prefix), local),
        None => (None, collapsed),
    };
    if !is_ncname(local) || prefix.is_some_and(|prefix| !is_ncname(prefix)) {
        return Err(FnError::new(
            FnErrorCode::Foca0002,
            format!("`{lexical}` does not match the QName production"),
        ));
    }
    let namespace = namespace_uri_for_prefix(prefix, element);
    if prefix.is_some() && namespace.is_none() {
        return Err(FnError::new(
            FnErrorCode::Fons0004,
            format!(
                "the prefix `{}` is not bound on the supplied element",
                prefix.unwrap_or_default()
            ),
        ));
    }
    Ok(QNameValue::new(prefix, namespace, local))
}

/// [`fn:QName`](https://www.w3.org/TR/xpath-functions-31/#func-QName): builds an `xs:QName` from
/// a namespace URI and a lexical `QName`.
///
/// # Errors
///
/// [`FnErrorCode::Foca0002`] when the second argument is not a lexical `QName`, and the same
/// code when a prefixed name is combined with an absent namespace — F&O 3.1 §11.1.2 gives
/// `fn:QName` `FOCA0002` for both, rather than the cast code `FORG0001`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::qnames::qname;
///
/// let value = qname(Some("http://www.example.com/example"), "person")?;
/// assert_eq!(value.namespace(), Some("http://www.example.com/example"));
/// assert_eq!(value.local_name(), "person");
/// assert_eq!(value.prefix(), None);
///
/// let prefixed = qname(Some("http://www.example.com/example"), "ht:person")?;
/// assert_eq!(prefixed.prefix(), Some("ht"));
///
/// // A prefix with no namespace is an error.
/// assert!(qname(None, "ht:person").is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn qname(namespace: Option<&str>, lexical: &str) -> FnResult<QNameValue> {
    let namespace = namespace.filter(|namespace| !namespace.is_empty());
    let (prefix, local) = match lexical.split_once(':') {
        Some((prefix, local)) => (Some(prefix), local),
        None => (None, lexical),
    };
    if !is_ncname(local) || prefix.is_some_and(|prefix| !is_ncname(prefix)) {
        return Err(FnError::new(
            FnErrorCode::Foca0002,
            format!("`{lexical}` does not match the QName production"),
        ));
    }
    if prefix.is_some() && namespace.is_none() {
        return Err(FnError::new(
            FnErrorCode::Foca0002,
            "fn:QName cannot combine a prefix with an absent namespace",
        ));
    }
    Ok(QNameValue::new(prefix, namespace, local))
}

/// [`fn:prefix-from-QName`](https://www.w3.org/TR/xpath-functions-31/#func-prefix-from-QName).
#[must_use]
pub fn prefix_from_qname(value: &QNameValue) -> Option<String> {
    value.prefix().map(str::to_owned)
}

/// [`fn:local-name-from-QName`](https://www.w3.org/TR/xpath-functions-31/#func-local-name-from-QName).
#[must_use]
pub fn local_name_from_qname(value: &QNameValue) -> String {
    value.local_name().to_owned()
}

/// [`fn:namespace-uri-from-QName`](https://www.w3.org/TR/xpath-functions-31/#func-namespace-uri-from-QName),
/// which answers the zero-length `xs:anyURI` for a name in no namespace.
#[must_use]
pub fn namespace_uri_from_qname(value: &QNameValue) -> String {
    value.namespace().unwrap_or_default().to_owned()
}

/// The `xs:QName` atomic value wrapper the constructors' results are given.
#[must_use]
pub fn qname_value(value: QNameValue) -> AtomicValue {
    AtomicValue::qname(value)
}

/// The `xs:NCName` atomic value `fn:prefix-from-QName` and `fn:local-name-from-QName` return.
///
/// # Errors
///
/// [`FnErrorCode::Forg0001`] when the string is not an `NCName`.
pub fn ncname_value(value: &str) -> FnResult<AtomicValue> {
    Ok(AtomicValue::parse(AtomicType::Ncname, value)?)
}

/// Extracts the `xs:QName` payload of an atomic value.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the value is not an `xs:QName` or `xs:NOTATION`.
pub fn as_qname(value: &AtomicValue, what: &str) -> FnResult<QNameValue> {
    value.as_qname().cloned().ok_or_else(|| {
        FnError::new(
            FnErrorCode::Xpty0004,
            format!("{what} expects an xs:QName, not {}", value.atomic_type()),
        )
    })
}

/// Checks that a node is an element, which the three namespace-aware functions require.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the node is not an element.
pub fn require_element(node: &NodeRef, what: &str) -> FnResult<()> {
    if node.node_kind() == NodeKind::Element {
        Ok(())
    } else {
        Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("{what} expects an element node, not a {}", node.node_kind()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    fn element(xml: &str) -> Result<NodeRef, Box<dyn std::error::Error>> {
        let tree = Tree::parse_str(xml)?;
        Ok(tree.document_element().ok_or("no document element")?)
    }

    #[test]
    fn in_scope_prefixes_include_xml() -> Result<(), Box<dyn std::error::Error>> {
        let root = element("<a xmlns='urn:d' xmlns:p='urn:p'/>")?;
        let mut prefixes = in_scope_prefixes(&root);
        prefixes.sort();
        assert_eq!(
            prefixes,
            vec!["".to_owned(), "p".to_owned(), "xml".to_owned()]
        );
        Ok(())
    }

    #[test]
    fn namespace_uri_for_prefix_examples() -> Result<(), Box<dyn std::error::Error>> {
        let root = element("<a xmlns='urn:d' xmlns:p='urn:p'/>")?;
        assert_eq!(
            namespace_uri_for_prefix(Some("p"), &root),
            Some("urn:p".to_owned())
        );
        assert_eq!(
            namespace_uri_for_prefix(None, &root),
            Some("urn:d".to_owned())
        );
        assert_eq!(
            namespace_uri_for_prefix(Some(""), &root),
            Some("urn:d".to_owned())
        );
        assert_eq!(namespace_uri_for_prefix(Some("q"), &root), None);
        assert_eq!(
            namespace_uri_for_prefix(Some("xml"), &root),
            Some(XML_NAMESPACE.to_owned())
        );
        Ok(())
    }

    #[test]
    fn resolve_qname_examples_from_the_specification() -> Result<(), Box<dyn std::error::Error>> {
        let root = element("<a xmlns:p='urn:p' xmlns='urn:d'/>")?;
        let resolved = resolve_qname("p:name", &root)?;
        assert_eq!(resolved.prefix(), Some("p"));
        assert_eq!(resolved.namespace(), Some("urn:p"));
        assert_eq!(resolved.local_name(), "name");

        let unprefixed = resolve_qname("name", &root)?;
        assert_eq!(unprefixed.namespace(), Some("urn:d"));
        assert_eq!(unprefixed.prefix(), None);

        assert!(
            resolve_qname("q:name", &root)
                .expect_err("unbound prefix")
                .is(FnErrorCode::Fons0004)
        );
        assert!(
            resolve_qname("1bad", &root)
                .expect_err("not a QName")
                .is(FnErrorCode::Foca0002)
        );
        Ok(())
    }

    #[test]
    fn qname_examples_from_the_specification() -> FnResult<()> {
        let value = qname(Some("http://www.example.com/example"), "person")?;
        assert_eq!(value.namespace(), Some("http://www.example.com/example"));
        assert_eq!(value.local_name(), "person");
        assert_eq!(value.prefix(), None);
        assert_eq!(value.lexical_form(), "person");

        let prefixed = qname(Some("http://www.example.com/example"), "ht:person")?;
        assert_eq!(prefixed.prefix(), Some("ht"));
        assert_eq!(prefixed.lexical_form(), "ht:person");

        assert!(qname(None, "ht:person").is_err());
        assert!(qname(Some("urn:x"), "not a name").is_err());
        assert!(qname(None, "person").is_ok());
        Ok(())
    }

    #[test]
    fn decomposition_functions() -> FnResult<()> {
        let value = qname(Some("http://www.example.com/example"), "person")?;
        assert_eq!(prefix_from_qname(&value), None);
        assert_eq!(local_name_from_qname(&value), "person");
        assert_eq!(
            namespace_uri_from_qname(&value),
            "http://www.example.com/example"
        );

        let bare = qname(None, "person")?;
        assert_eq!(namespace_uri_from_qname(&bare), "");
        Ok(())
    }

    #[test]
    fn non_elements_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a/>")?;
        assert!(require_element(&tree.root(), "fn:resolve-QName").is_err());
        assert!(require_element(&element("<a/>")?, "fn:resolve-QName").is_ok());
        Ok(())
    }
}
