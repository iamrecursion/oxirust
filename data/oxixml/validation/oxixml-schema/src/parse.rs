//! Reading a *schema document*: the DOM helpers and the attribute-value
//! grammars that [XML Schema Part 1 §3.2](https://www.w3.org/TR/xmlschema-1/)
//! defines for the schema vocabulary itself.

use oxixml_dom::{Document, NodeId, NodeKind};
use oxixml_qname::{ExpandedName, is_ncname};

use crate::components::{DerivationSet, MaxOccurs, NamespaceConstraint, ProcessContents, Wildcard};
use crate::error::SchemaError;
use crate::names::{XSD_NS, XsdVersion};
use crate::value::{NamespaceContext, collapse};

/// The child elements of `node`, in document order.
#[must_use]
pub(crate) fn child_elements(doc: &Document, node: NodeId) -> Vec<NodeId> {
    doc.children(node)
        .filter(|child| doc.kind(*child) == Some(NodeKind::Element))
        .collect()
}

/// The local name of `node` when it is in the XML Schema namespace.
#[must_use]
pub(crate) fn xsd_name(doc: &Document, node: NodeId) -> Option<&str> {
    let element = doc.element(node)?;
    (element.namespace_uri() == Some(XSD_NS)).then(|| element.local_name())
}

/// The value of an unqualified attribute of `node`.
#[must_use]
pub(crate) fn attr<'a>(doc: &'a Document, node: NodeId, name: &str) -> Option<&'a str> {
    doc.get_attribute_ns(node, None, name)
}

/// The `whiteSpace="collapse"` form of an attribute value.
#[must_use]
pub(crate) fn attr_token(doc: &Document, node: NodeId, name: &str) -> Option<String> {
    attr(doc, node, name).map(collapse)
}

/// The namespace bindings in scope at `node`.
#[must_use]
pub(crate) fn namespace_context(doc: &Document, node: NodeId) -> NamespaceContext {
    let mut ancestors: Vec<NodeId> = doc.ancestors(node).collect();
    ancestors.reverse();
    let mut context = NamespaceContext::new();
    for ancestor in ancestors.into_iter().chain(std::iter::once(node)) {
        for (prefix, namespace) in doc.namespace_declarations(ancestor) {
            context.declare(prefix, namespace);
        }
    }
    context
}

/// The `xpathDefaultNamespace` in force on `node`, already resolved.
///
/// XSD 1.1 §3.13.1 lets the attribute appear on `xs:schema`, on the two
/// identity-constraint paths and on `xs:assert`/`xs:assertion`/
/// `xs:alternative`; the innermost wins, and the three `##` tokens stand for
/// the target namespace, for the default namespace of the schema document, and
/// for "no namespace" respectively. Absent everywhere, an unprefixed name in an
/// assertion is in no namespace.
pub(crate) fn xpath_default_namespace(
    doc: &Document,
    node: NodeId,
    target_namespace: Option<&str>,
) -> Option<String> {
    let mut current = Some(node);
    while let Some(id) = current {
        if let Some(text) = attr_token(doc, id, "xpathDefaultNamespace") {
            return match text.as_str() {
                "##local" => None,
                "##targetNamespace" => target_namespace.map(str::to_owned),
                "##defaultNamespace" => {
                    namespace_context(doc, node).lookup(None).map(str::to_owned)
                }
                "" => None,
                other => Some(other.to_owned()),
            };
        }
        current = doc.parent(id);
    }
    None
}

/// A simplified XPath to `node`, for diagnostics.
///
/// The sibling index is computed without allocating, because this runs once per
/// reported error on documents that may have very wide element lists.
#[must_use]
pub(crate) fn node_path(doc: &Document, node: NodeId) -> String {
    let mut steps: Vec<String> = Vec::new();
    let mut current = Some(node);
    while let Some(id) = current {
        match doc.kind(id) {
            Some(NodeKind::Element) => {
                let Some(element) = doc.element(id) else {
                    break;
                };
                let namespace = element.namespace_uri();
                let local = element.local_name();
                let index = match doc.parent(id) {
                    Some(parent) => {
                        let mut position = 1;
                        for sibling in doc.children(parent) {
                            if sibling == id {
                                break;
                            }
                            if let Some(other) = doc.element(sibling)
                                && other.namespace_uri() == namespace
                                && other.local_name() == local
                            {
                                position += 1;
                            }
                        }
                        position
                    }
                    None => 1,
                };
                steps.push(format!("{}[{index}]", element.qualified_name()));
            }
            Some(NodeKind::Attr) => {
                if let Some(data) = doc.attr(id) {
                    steps.push(format!("@{}", data.qualified_name()));
                }
                current = doc.owner_element(id);
                continue;
            }
            _ => {}
        }
        current = doc.parent(id);
    }
    steps.reverse();
    if steps.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", steps.join("/"))
    }
}

/// Resolves a `QName`-valued attribute in the namespace context of `node`.
///
/// # Errors
///
/// Returns `src-qname` when the value is not a `QName` or its prefix is not
/// declared.
pub(crate) fn resolve_qname(
    doc: &Document,
    node: NodeId,
    value: &str,
) -> Result<ExpandedName, SchemaError> {
    let value = collapse(value);
    let context = namespace_context(doc, node);
    match value.split_once(':') {
        Some((prefix, local)) => {
            if !is_ncname(prefix) || !is_ncname(local) {
                return Err(SchemaError::new(
                    "src-qname",
                    format!("`{value}` is not a QName"),
                ));
            }
            let namespace = context.lookup(Some(prefix)).ok_or_else(|| {
                SchemaError::new(
                    "src-qname",
                    format!("the namespace prefix `{prefix}` is not declared"),
                )
            })?;
            Ok(ExpandedName::new(Some(namespace), local))
        }
        None => {
            if !is_ncname(&value) {
                return Err(SchemaError::new(
                    "src-qname",
                    format!("`{value}` is not a QName"),
                ));
            }
            Ok(ExpandedName::new(context.lookup(None), value))
        }
    }
}

/// Parses `minOccurs`, whose type is `xs:nonNegativeInteger` — unbounded, so a
/// value beyond [`u32`] is legal and saturates.
pub(crate) fn min_occurs(doc: &Document, node: NodeId) -> Result<u32, SchemaError> {
    match attr_token(doc, node, "minOccurs") {
        Some(text) => nonnegative_integer(&text, "minOccurs")
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        None => Ok(1),
    }
}

/// Parses `maxOccurs`, which is `unbounded` or an `xs:nonNegativeInteger`.
pub(crate) fn max_occurs(doc: &Document, node: NodeId) -> Result<MaxOccurs, SchemaError> {
    match attr_token(doc, node, "maxOccurs") {
        Some(text) if text == "unbounded" => Ok(MaxOccurs::Unbounded),
        Some(text) => Ok(match nonnegative_integer(&text, "maxOccurs")? {
            value if value > u128::from(u32::MAX) => MaxOccurs::Unbounded,
            value => MaxOccurs::Bounded(u32::try_from(value).unwrap_or(u32::MAX)),
        }),
        None => Ok(MaxOccurs::Bounded(1)),
    }
}

fn nonnegative_integer(text: &str, what: &str) -> Result<u128, SchemaError> {
    let digits = text.strip_prefix('+').unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(SchemaError::new(
            "src-particle",
            format!("`{text}` is not a valid `{what}`"),
        ));
    }
    // Values wider than 128 bits are meaningless here and saturate.
    Ok(digits.parse::<u128>().unwrap_or(u128::MAX))
}

/// Parses a `boolean`-valued schema attribute.
pub(crate) fn bool_attr(
    doc: &Document,
    node: NodeId,
    name: &str,
    default: bool,
) -> Result<bool, SchemaError> {
    match attr_token(doc, node, name).as_deref() {
        None => Ok(default),
        Some("true" | "1") => Ok(true),
        Some("false" | "0") => Ok(false),
        Some(other) => Err(SchemaError::new(
            "src-attribute",
            format!("`{other}` is not a boolean value for `{name}`"),
        )),
    }
}

/// Parses a `block`/`final`/`blockDefault`/`finalDefault` value.
pub(crate) fn derivation_set(
    text: &str,
    allowed: &[&str],
    what: &str,
) -> Result<DerivationSet, SchemaError> {
    let text = collapse(text);
    if text == "#all" {
        return Ok(DerivationSet::all());
    }
    let mut set = DerivationSet::empty();
    for token in text.split(' ').filter(|token| !token.is_empty()) {
        if !allowed.contains(&token) {
            return Err(SchemaError::new(
                "src-simple-type",
                format!("`{token}` is not allowed in `{what}`"),
            ));
        }
        match token {
            "extension" => set.extension = true,
            "restriction" => set.restriction = true,
            "list" => set.list = true,
            "union" => set.union = true,
            "substitution" => set.substitution = true,
            _ => {}
        }
    }
    Ok(set)
}

/// Parses an `xs:any`/`xs:anyAttribute` wildcard.
pub(crate) fn wildcard(
    doc: &Document,
    node: NodeId,
    target_namespace: Option<&str>,
    version: XsdVersion,
) -> Result<Wildcard, SchemaError> {
    let process_contents = match attr_token(doc, node, "processContents") {
        Some(text) => ProcessContents::parse(&text).ok_or_else(|| {
            SchemaError::new(
                "src-wildcard",
                format!("`{text}` is not a `processContents` value"),
            )
        })?,
        None => ProcessContents::Strict,
    };
    // XSD 1.1 §3.10.2 (*Wildcard*, XML representation): `namespace` and
    // `notNamespace` are alternatives — writing both is a schema-representation
    // error (`saxonData/Wild/wild007`, `wild008`).
    if version.is_1_1()
        && attr_token(doc, node, "namespace").is_some()
        && attr_token(doc, node, "notNamespace").is_some()
    {
        return Err(SchemaError::new(
            "src-wildcard",
            "`namespace` and `notNamespace` may not both be present on a wildcard",
        )
        .at(node_path(doc, node)));
    }
    let namespace = if let Some(text) = attr_token(doc, node, "namespace") {
        namespace_constraint(&text, target_namespace, false)?
    } else if version.is_1_1()
        && let Some(text) = attr_token(doc, node, "notNamespace")
    {
        namespace_constraint(&text, target_namespace, true)?
    } else {
        NamespaceConstraint::Any
    };
    // XSD 1.1 §3.10.2: `notQName` is a list of QNames plus, for an element
    // wildcard, the two tokens `##defined` and `##definedSibling`; an attribute
    // wildcard may write `##defined` alone.
    let mut disallowed = crate::components::DisallowedNames::default();
    if version.is_1_1()
        && let Some(text) = attr(doc, node, "notQName")
    {
        let attribute_wildcard = xsd_name(doc, node) == Some("anyAttribute");
        let context = namespace_context(doc, node);
        for token in text.split_whitespace() {
            match token {
                "##defined" => disallowed.defined = true,
                "##definedSibling" if !attribute_wildcard => disallowed.defined_sibling = true,
                "##definedSibling" => {
                    return Err(SchemaError::new(
                        "src-wildcard",
                        "`##definedSibling` may only appear on an element wildcard",
                    )
                    .at(node_path(doc, node)));
                }
                other => {
                    let (prefix, local) = match other.split_once(':') {
                        Some((prefix, local)) => (Some(prefix), local),
                        None => (None, other),
                    };
                    if !crate::value::is_xml_ncname(local)
                        || prefix.is_some_and(|prefix| !crate::value::is_xml_ncname(prefix))
                    {
                        return Err(SchemaError::new(
                            "src-wildcard",
                            format!("`{other}` is not a QName"),
                        )
                        .at(node_path(doc, node)));
                    }
                    // Datatypes §3.2.18: "If the host language does not
                    // specify otherwise, unqualified names are bound to the
                    // **default namespace**. Note: The default treatment of
                    // unqualified names parallels that specified in
                    // [Namespaces in XML] for element names (as opposed to
                    // that specified for attribute names)." XSD does not
                    // specify otherwise for `notQName`, so an unprefixed
                    // entry picks up whatever `xmlns` is in force on the
                    // wildcard (`ibmData/instance_invalid/S3_4_6/
                    // s3_4_6ii03`), exactly as `type` or `base` would.
                    let namespace = match prefix {
                        Some(prefix) => match context.lookup(Some(prefix)) {
                            Some(namespace) => Some(namespace.to_owned()),
                            None => {
                                return Err(SchemaError::new(
                                    "src-wildcard",
                                    format!("the prefix `{prefix}` of `{other}` is not bound"),
                                )
                                .at(node_path(doc, node)));
                            }
                        },
                        None => context.lookup(None).map(str::to_owned),
                    };
                    let name = ExpandedName::new(namespace, local);
                    if !disallowed.names.contains(&name) {
                        disallowed.names.push(name);
                    }
                }
            }
        }
    }
    // XSD 1.1 §3.10.2: every QName in `notQName` must name something the
    // wildcard's namespace constraint could have matched — excluding a name
    // that is out of scope anyway says nothing and is an error
    // (`saxonData/Wild/wild031` … `wild035`).
    for name in &disallowed.names {
        if !namespace.allows(name.namespace()) {
            return Err(SchemaError::new(
                "src-wildcard",
                format!("`{name}` is not in a namespace this wildcard matches"),
            )
            .at(node_path(doc, node)));
        }
    }
    Ok(Wildcard {
        namespace,
        process_contents,
        disallowed,
    })
}

fn namespace_constraint(
    text: &str,
    target_namespace: Option<&str>,
    negated: bool,
) -> Result<NamespaceConstraint, SchemaError> {
    if !negated {
        if text == "##any" {
            return Ok(NamespaceConstraint::Any);
        }
        if text == "##other" {
            let mut excluded = std::collections::BTreeSet::new();
            excluded.insert(target_namespace.map(str::to_owned));
            excluded.insert(None);
            return Ok(NamespaceConstraint::Not(excluded));
        }
    }
    let mut set = std::collections::BTreeSet::new();
    for token in text.split(' ').filter(|token| !token.is_empty()) {
        match token {
            "##targetNamespace" => {
                set.insert(target_namespace.map(str::to_owned));
            }
            "##local" => {
                set.insert(None);
            }
            "##any" | "##other" => {
                return Err(SchemaError::new(
                    "src-wildcard",
                    format!("`{token}` may not appear in a namespace list"),
                ));
            }
            other => {
                set.insert(Some(other.to_owned()));
            }
        }
    }
    Ok(if negated {
        NamespaceConstraint::Not(set)
    } else {
        NamespaceConstraint::Enumeration(set)
    })
}

/// Returns the children of `node` with any leading `xs:annotation` removed.
#[must_use]
pub(crate) fn content_children(doc: &Document, node: NodeId) -> Vec<NodeId> {
    child_elements(doc, node)
        .into_iter()
        .filter(|child| xsd_name(doc, *child) != Some("annotation"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(text: &str) -> Document {
        oxixml_dom::build::parse_str(text).expect("well-formed XML")
    }

    #[test]
    fn qnames_resolve_in_scope() -> Result<(), SchemaError> {
        let doc = document(r#"<a xmlns:p="urn:x" xmlns="urn:d"><b/></a>"#);
        let root = doc.document_element().expect("a root");
        let child = child_elements(&doc, root)[0];
        assert_eq!(
            resolve_qname(&doc, child, "p:t")?,
            ExpandedName::new(Some("urn:x"), "t")
        );
        assert_eq!(
            resolve_qname(&doc, child, "t")?,
            ExpandedName::new(Some("urn:d"), "t")
        );
        assert!(resolve_qname(&doc, child, "q:t").is_err());
        Ok(())
    }

    #[test]
    fn occurrence_attributes() -> Result<(), SchemaError> {
        let doc = document(r#"<a minOccurs="0" maxOccurs="unbounded"/>"#);
        let root = doc.document_element().expect("a root");
        assert_eq!(min_occurs(&doc, root)?, 0);
        assert_eq!(max_occurs(&doc, root)?, MaxOccurs::Unbounded);
        Ok(())
    }

    #[test]
    fn derivation_sets_parse() -> Result<(), SchemaError> {
        let set = derivation_set(
            "extension restriction",
            &["extension", "restriction"],
            "block",
        )?;
        assert!(set.extension && set.restriction);
        assert_eq!(
            derivation_set("#all", &["extension"], "block")?,
            DerivationSet::all()
        );
        assert!(derivation_set("list", &["extension"], "block").is_err());
        Ok(())
    }
}
