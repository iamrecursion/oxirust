//! Parsing: `xml:base` resolution/scoping, `group` scoping, and error paths.

use oxixml_catalog::{Catalog, CatalogError};

#[test]
fn relative_uri_resolves_against_supplied_base() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <system systemId="urn:example:foo" uri="../dtds/foo.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/cat/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system("urn:example:foo").as_deref(),
        Some("http://example.org/dtds/foo.dtd")
    );
    Ok(())
}

#[test]
fn xml_base_on_root_scopes_all_entries() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"
             xml:base="http://example.org/other/">
      <system systemId="urn:example:foo" uri="foo.dtd"/>
    </catalog>
    "#;
    // The supplied base is deliberately different from the root's xml:base,
    // to confirm the attribute wins.
    let catalog = Catalog::parse(xml, "http://example.org/cat/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system("urn:example:foo").as_deref(),
        Some("http://example.org/other/foo.dtd")
    );
    Ok(())
}

#[test]
fn xml_base_on_group_scopes_only_its_children() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <system systemId="urn:example:outside" uri="outside.dtd"/>
      <group xml:base="http://example.org/scoped/">
        <system systemId="urn:example:inside" uri="inside.dtd"/>
      </group>
      <system systemId="urn:example:after" uri="after.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/base/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system("urn:example:outside").as_deref(),
        Some("http://example.org/base/outside.dtd")
    );
    assert_eq!(
        catalog.resolve_system("urn:example:inside").as_deref(),
        Some("http://example.org/scoped/inside.dtd")
    );
    assert_eq!(
        catalog.resolve_system("urn:example:after").as_deref(),
        Some("http://example.org/base/after.dtd")
    );
    Ok(())
}

#[test]
fn xml_base_on_entry_overrides_enclosing_group_base() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <group xml:base="http://example.org/group/">
        <system systemId="urn:example:a" uri="a.dtd"/>
        <system systemId="urn:example:b" uri="b.dtd" xml:base="http://example.org/entry/"/>
      </group>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system("urn:example:a").as_deref(),
        Some("http://example.org/group/a.dtd")
    );
    assert_eq!(
        catalog.resolve_system("urn:example:b").as_deref(),
        Some("http://example.org/entry/b.dtd")
    );
    Ok(())
}

#[test]
fn nested_group_base_and_prefer_compose() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"
             xml:base="http://example.org/root/" prefer="system">
      <group xml:base="inner/" prefer="public">
        <public publicId="-//Example//DTD Foo//EN" uri="foo.dtd"/>
      </group>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/unused/catalog.xml")?;
    // prefer="public" on the group makes the public entry eligible even
    // though a system_id is supplied and no system entry matches.
    assert_eq!(
        catalog
            .resolve_public("-//Example//DTD Foo//EN", Some("urn:example:whatever"))
            .as_deref(),
        Some("http://example.org/root/inner/foo.dtd")
    );
    Ok(())
}

#[test]
fn foreign_namespace_and_unknown_elements_are_ignored() -> Result<(), CatalogError> {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" xmlns:x="urn:example:ext">
      <x:extension>
        <system systemId="urn:example:should-be-ignored" uri="ignored.dtd"/>
      </x:extension>
      <unknownEntry systemId="urn:example:also-ignored" uri="ignored2.dtd"/>
      <system systemId="urn:example:real" uri="real.dtd"/>
    </catalog>
    "#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system("urn:example:should-be-ignored"),
        None
    );
    assert_eq!(catalog.resolve_system("urn:example:also-ignored"), None);
    assert_eq!(
        catalog.resolve_system("urn:example:real").as_deref(),
        Some("http://example.org/real.dtd")
    );
    Ok(())
}

#[test]
fn empty_catalog_parses_and_matches_nothing() -> Result<(), CatalogError> {
    let xml = r#"<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"/>"#;
    let catalog = Catalog::parse(xml, "http://example.org/catalog.xml")?;
    assert_eq!(catalog.resolve_system("urn:anything"), None);
    assert_eq!(catalog.resolve_uri("urn:anything"), None);
    assert_eq!(catalog.resolve_public("urn:anything", None), None);
    Ok(())
}

#[test]
fn missing_root_element_is_an_error() {
    let xml = r#"<notACatalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"/>"#;
    let err = Catalog::parse(xml, "http://example.org/catalog.xml").expect_err("expected an error");
    assert!(matches!(err, CatalogError::MissingRootElement));
}

#[test]
fn wrong_namespace_root_is_an_error() {
    let xml = r#"<catalog xmlns="urn:example:not-the-catalog-ns"/>"#;
    let err = Catalog::parse(xml, "http://example.org/catalog.xml").expect_err("expected an error");
    assert!(matches!(err, CatalogError::MissingRootElement));
}

#[test]
fn missing_required_attribute_is_an_error() {
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <system uri="foo.dtd"/>
    </catalog>
    "#;
    let err = Catalog::parse(xml, "http://example.org/catalog.xml").expect_err("expected an error");
    match err {
        CatalogError::MissingAttribute { element, attribute } => {
            assert_eq!(element, "system");
            assert_eq!(attribute, "systemId");
        }
        other => panic!("expected MissingAttribute, got {other:?}"),
    }
}

#[test]
fn invalid_prefer_value_is_an_error() {
    let xml = r#"<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" prefer="maybe"/>"#;
    let err = Catalog::parse(xml, "http://example.org/catalog.xml").expect_err("expected an error");
    match err {
        CatalogError::InvalidPreferValue { value } => assert_eq!(value, "maybe"),
        other => panic!("expected InvalidPreferValue, got {other:?}"),
    }
}

#[test]
fn invalid_base_uri_is_an_error() {
    let xml = r#"<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"/>"#;
    let err = Catalog::parse(xml, "not a valid iri").expect_err("expected an error");
    assert!(matches!(err, CatalogError::InvalidIri { .. }));
}

/// `depth` nested `<group>` elements around a `<system>` entry.
fn nested_groups(depth: usize) -> String {
    let mut xml = String::from(r#"<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">"#);
    for _ in 0..depth {
        xml.push_str("<group>");
    }
    xml.push_str(r#"<system systemId="urn:example:foo" uri="foo.dtd"/>"#);
    for _ in 0..depth {
        xml.push_str("</group>");
    }
    xml.push_str("</catalog>");
    xml
}

#[test]
fn deeply_nested_group_elements_are_a_clean_error_not_a_stack_overflow() {
    // `parse_children`/`dispatch` recurse once per `group` nesting level with no depth guard of
    // their own; nesting `group` elements deep enough used to exhaust the native call stack (a
    // genuine, uncatchable process abort) well before any other limit in this crate kicked in.
    let xml = nested_groups(oxixml_catalog::MAX_GROUP_DEPTH + 1);
    let error =
        Catalog::parse(&xml, "http://example.org/catalog.xml").expect_err("must be rejected");
    assert!(
        matches!(
            error,
            CatalogError::GroupNestingTooDeep { limit } if limit == oxixml_catalog::MAX_GROUP_DEPTH
        ),
        "{error:?}"
    );
}

#[test]
fn nested_group_elements_at_the_depth_cap_still_parse() -> Result<(), CatalogError> {
    // The guard must not reject a chain that lands exactly on the cap — only one deeper.
    let xml = nested_groups(oxixml_catalog::MAX_GROUP_DEPTH);
    let catalog = Catalog::parse(&xml, "http://example.org/catalog.xml")?;
    assert_eq!(
        catalog.resolve_system("urn:example:foo").as_deref(),
        Some("http://example.org/foo.dtd")
    );
    Ok(())
}

#[test]
fn unclosed_entry_element_is_an_error() {
    // A non-empty `<system>` with a nested element is unusual but must still
    // be handled: the parser should either accept and skip it, or the
    // document is genuinely malformed. Here it's deliberately unterminated.
    let xml = r#"
    <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
      <system systemId="urn:example:foo" uri="foo.dtd">
    </catalog>
    "#;
    assert!(Catalog::parse(xml, "http://example.org/catalog.xml").is_err());
}
