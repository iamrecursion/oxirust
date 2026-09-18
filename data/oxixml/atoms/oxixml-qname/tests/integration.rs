//! Black-box integration tests exercising `oxixml-qname`'s public API as an external consumer
//! would: through the crate root only, mirroring the scenarios called out in the crate's
//! design brief (name-validation boundaries, namespace-stack scoping/shadowing/undeclare rules,
//! reserved-prefix enforcement, element-vs-attribute default-namespace resolution, and
//! `ExpandedName` identity across differing prefixes).

use std::collections::HashSet;
use std::error::Error;

use oxixml_qname::{
    ExpandedName, NameContext, NamespaceDeclError, NamespaceStack, NamespacesVersion, QName,
    ResolveQNameError, XML_NAMESPACE, XMLNS_NAMESPACE, is_name, is_ncname, is_nmtoken, is_qname,
};

#[test]
fn name_validation_boundaries() {
    // Colon rules: `Name` allows colons anywhere, `NCName`/`QName` restrict them tightly.
    assert!(is_name("a:b:c"));
    assert!(!is_ncname("a:b"));
    assert!(is_qname("a:b"));
    assert!(!is_qname("a:b:c"));
    assert!(!is_qname(":b"));
    assert!(!is_qname("a:"));

    // A digit can never start a Name/NCName, but is fine as a bare Nmtoken or mid-name.
    assert!(!is_name("9lives"));
    assert!(!is_ncname("9lives"));
    assert!(is_nmtoken("9lives"));
    assert!(is_name("a9"));

    // Combining characters (e.g. U+0301 COMBINING ACUTE ACCENT) are NameChar but not
    // NameStartChar.
    assert!(is_name("a\u{0301}"));
    assert!(!is_name("\u{0301}a"));
    assert!(is_nmtoken("\u{0301}")); // Nmtoken has no start-char restriction

    // CJK Unified Ideographs (5th-edition NameStartChar range #x3001-#xD7FF) and astral-plane
    // ideographs (#x10000-#xEFFFF) are accepted as NCName start characters.
    assert!(is_ncname("日本語"));
    assert!(is_ncname("\u{20000}")); // an astral CJK Extension B ideograph
}

#[test]
fn namespace_stack_scoping_shadowing_and_undeclare_rules() -> Result<(), NamespaceDeclError> {
    let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_1);

    namespaces.push_scope();
    namespaces.declare(Some("p"), "urn:outer")?;
    assert_eq!(namespaces.resolve(Some("p")), Some("urn:outer"));

    // Shadowing: an inner scope's declaration for the same prefix wins while it is open...
    namespaces.push_scope();
    namespaces.declare(Some("p"), "urn:inner")?;
    assert_eq!(namespaces.resolve(Some("p")), Some("urn:inner"));

    // ...and undeclaring it with an empty URI (a Namespaces 1.1 relaxation) removes it from
    // scope entirely, rather than falling back to the outer binding.
    namespaces.declare(Some("p"), "")?;
    assert_eq!(namespaces.resolve(Some("p")), None);
    namespaces.pop_scope();

    // Popping restores the outer scope's binding.
    assert_eq!(namespaces.resolve(Some("p")), Some("urn:outer"));
    namespaces.pop_scope();
    assert_eq!(namespaces.resolve(Some("p")), None);

    // The same empty-URI undeclare is a hard error under Namespaces 1.0.
    let mut v10 = NamespaceStack::new(NamespacesVersion::V1_0);
    v10.push_scope();
    v10.declare(Some("p"), "urn:outer")?;
    assert_eq!(
        v10.declare(Some("p"), ""),
        Err(NamespaceDeclError::EmptyUriRequiresNamespaces11)
    );
    // ...but undeclaring the *default* namespace with an empty URI is always fine.
    assert!(v10.declare(None, "").is_ok());

    Ok(())
}

#[test]
fn reserved_prefix_enforcement() {
    let mut namespaces = NamespaceStack::default();
    namespaces.push_scope();

    // `xmlns` can never be declared as a prefix.
    assert_eq!(
        namespaces.declare(Some("xmlns"), "urn:whatever"),
        Err(NamespaceDeclError::ReservedXmlnsPrefix)
    );
    // `xml` may only ever be (redundantly) bound to its one fixed namespace name.
    assert_eq!(
        namespaces.declare(Some("xml"), "urn:wrong"),
        Err(NamespaceDeclError::XmlPrefixMisbound)
    );
    assert!(namespaces.declare(Some("xml"), XML_NAMESPACE).is_ok());
    // No other prefix (nor the default namespace) may bind to the XML or XMLNS namespace names.
    assert_eq!(
        namespaces.declare(Some("x"), XML_NAMESPACE),
        Err(NamespaceDeclError::XmlNamespaceReserved)
    );
    assert_eq!(
        namespaces.declare(None, XML_NAMESPACE),
        Err(NamespaceDeclError::XmlNamespaceReserved)
    );
    assert_eq!(
        namespaces.declare(Some("x"), XMLNS_NAMESPACE),
        Err(NamespaceDeclError::XmlnsNamespaceReserved)
    );
    // And these two are always resolvable regardless of any declaration.
    assert_eq!(namespaces.resolve(Some("xml")), Some(XML_NAMESPACE));
    assert_eq!(namespaces.resolve(Some("xmlns")), Some(XMLNS_NAMESPACE));
}

#[test]
fn element_vs_attribute_default_namespace_resolution() -> Result<(), Box<dyn Error>> {
    let mut namespaces = NamespaceStack::default();
    namespaces.push_scope();
    namespaces.declare(None, "urn:example:default")?;
    namespaces.declare(Some("dc"), "urn:example:dc")?;

    // An unprefixed *element* name inherits the default namespace.
    let element = namespaces.resolve_qname("title", NameContext::Element)?;
    assert_eq!(element.namespace(), Some("urn:example:default"));
    assert_eq!(element.local_name(), "title");

    // An unprefixed *attribute* name never does — this is the one rule the context parameter
    // exists to encode.
    let attribute = namespaces.resolve_qname("id", NameContext::Attribute)?;
    assert_eq!(attribute.namespace(), None);
    assert_eq!(attribute.local_name(), "id");

    // Prefixed names resolve identically regardless of context.
    let prefixed_element = namespaces.resolve_qname("dc:title", NameContext::Element)?;
    let prefixed_attribute = namespaces.resolve_qname("dc:title", NameContext::Attribute)?;
    assert_eq!(prefixed_element, prefixed_attribute);
    assert_eq!(prefixed_element.namespace(), Some("urn:example:dc"));

    // An unbound prefix is always an error, in either context.
    assert!(matches!(
        namespaces.resolve_qname("nope:x", NameContext::Attribute),
        Err(ResolveQNameError::UnboundPrefix(p)) if p == "nope"
    ));

    Ok(())
}

#[test]
fn expanded_name_identity_is_independent_of_prefix() -> Result<(), Box<dyn Error>> {
    // Two different prefixes bound to the same namespace name must resolve to the *same*
    // ExpandedName identity — this is the entire reason ExpandedName exists.
    let mut namespaces = NamespaceStack::default();
    namespaces.push_scope();
    namespaces.declare(Some("a"), "urn:shared")?;
    namespaces.declare(Some("b"), "urn:shared")?;

    let via_a = namespaces.resolve_qname("a:widget", NameContext::Element)?;
    let via_b = namespaces.resolve_qname("b:widget", NameContext::Element)?;
    assert_eq!(via_a, via_b);

    let mut seen = HashSet::new();
    seen.insert(via_a);
    assert!(seen.contains(&via_b));
    assert_eq!(seen.len(), 1);

    // But a different local name, or a different namespace, is a genuinely different identity.
    let different_local = namespaces.resolve_qname("a:gadget", NameContext::Element)?;
    assert_ne!(different_local, via_b);
    assert!(!seen.contains(&different_local));

    namespaces.declare(Some("c"), "urn:other")?;
    let different_namespace = namespaces.resolve_qname("c:widget", NameContext::Element)?;
    assert_ne!(different_namespace, via_b);

    Ok(())
}

#[test]
fn qname_and_expanded_name_round_trip_through_a_streaming_style_walk() -> Result<(), Box<dyn Error>>
{
    // A small simulation of what a streaming parser does: push a scope per element, declare
    // whatever `xmlns*` attributes it saw, resolve the element and its attributes, then pop.
    let mut namespaces = NamespaceStack::new(NamespacesVersion::V1_0);

    // <book xmlns="urn:example:book" xmlns:dc="http://purl.org/dc/elements/1.1/">
    namespaces.push_scope();
    namespaces.declare(None, "urn:example:book")?;
    namespaces.declare(Some("dc"), "http://purl.org/dc/elements/1.1/")?;
    let book = namespaces.resolve_qname("book", NameContext::Element)?;
    assert_eq!(book, ExpandedName::new(Some("urn:example:book"), "book"));

    //   <dc:title id="t1">Structure and Interpretation</dc:title>
    namespaces.push_scope();
    let title_qname: QName<&str> = QName::parse("dc:title")?;
    assert_eq!(title_qname.prefix(), Some("dc"));
    let title = namespaces.resolve_qname(title_qname.as_str(), NameContext::Element)?;
    assert_eq!(
        title,
        ExpandedName::new(Some("http://purl.org/dc/elements/1.1/"), "title")
    );
    let id_attr = namespaces.resolve_qname("id", NameContext::Attribute)?;
    assert_eq!(id_attr.namespace(), None); // unprefixed attribute: no namespace, even here
    namespaces.pop_scope(); // </dc:title>

    namespaces.pop_scope(); // </book>
    assert_eq!(namespaces.resolve(None), None);

    Ok(())
}
