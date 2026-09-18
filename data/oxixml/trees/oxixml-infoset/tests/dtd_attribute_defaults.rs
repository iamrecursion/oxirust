//! Attributes supplied by the document type declaration.
//!
//! [XML 1.0 §3.3.2](https://www.w3.org/TR/xml/#sec-attr-defaults) has a processor add an
//! attribute to an element whenever the element's attribute-list declaration gives that attribute
//! a default value and the start tag did not write one. Reading the internal subset is not
//! optional — [XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types) requires even a
//! non-validating processor to do it — so this happens for every tree this crate builds.
//!
//! When the attribute so supplied is a namespace declaration,
//! [Namespaces in XML §3](https://www.w3.org/TR/xml-names/#ns-decl) makes it a declaration like
//! any other: "a namespace declaration may be supplied by the attribute-list declaration in the
//! DTD", so the prefix it binds is in scope on the element.

use oxixml_infoset::{Tree, XdmNode};

/// The prefixes in scope on `node`, sorted, with the default-namespace binding spelled `""`.
fn in_scope_prefixes(node: &oxixml_infoset::NodeRef) -> Vec<String> {
    let mut prefixes: Vec<String> = node
        .namespaces()
        .iter()
        .map(|namespace| {
            namespace
                .node_name()
                .map_or_else(String::new, |name| name.local_name().to_owned())
        })
        .collect();
    prefixes.sort();
    prefixes
}

/// The attributes of `node` as `(name, value)` pairs, with the name spelled the way
/// [`oxixml_qname::ExpandedName`] displays it.
fn attributes(node: &oxixml_infoset::NodeRef) -> Vec<(String, String)> {
    node.attributes()
        .iter()
        .map(|attribute| {
            (
                attribute
                    .node_name()
                    .map_or_else(String::new, |name| name.to_string()),
                attribute.string_value(),
            )
        })
        .collect()
}

#[test]
fn a_defaulted_namespace_declaration_binds_its_prefix() -> Result<(), oxixml_infoset::XdmError> {
    // The document writes one namespace declaration and the DTD supplies the other.
    let tree = Tree::parse_str(
        r#"<!DOCTYPE svg[
             <!ELEMENT svg EMPTY>
             <!ATTLIST svg
                       xmlns CDATA #IMPLIED
                       xmlns:xlink CDATA #FIXED "http://www.w3.org/1999/xlink">
           ]>
           <svg xmlns="http://www.w3.org/2000/svg"/>"#,
    )?;
    let svg = tree.document_element().expect("a document element");
    assert_eq!(in_scope_prefixes(&svg), ["", "xlink", "xml"]);

    let xlink = svg
        .namespaces()
        .into_iter()
        .find(|namespace| {
            namespace
                .node_name()
                .is_some_and(|name| name.local_name() == "xlink")
        })
        .expect("the defaulted binding");
    assert_eq!(xlink.string_value(), "http://www.w3.org/1999/xlink");

    // A namespace declaration is not also an attribute node.
    assert!(
        attributes(&svg).is_empty(),
        "{:?}",
        attributes(&svg)
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn a_defaulted_default_namespace_declaration_binds_unprefixed_names()
-> Result<(), oxixml_infoset::XdmError> {
    let tree = Tree::parse_str(r#"<!DOCTYPE r [<!ATTLIST r xmlns CDATA "urn:d">]><r><c/></r>"#)?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(
        root.node_name().map(|name| name.to_string()).as_deref(),
        Some("{urn:d}r")
    );
    // The declaration is on `r`, so it is in scope on the child too.
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(
        child.node_name().map(|name| name.to_string()).as_deref(),
        Some("{urn:d}c")
    );
    Ok(())
}

#[test]
fn an_ordinary_attribute_default_is_supplied() -> Result<(), oxixml_infoset::XdmError> {
    let tree = Tree::parse_str(
        r#"<!DOCTYPE r [<!ATTLIST r kind CDATA "plain" size CDATA #FIXED "7">]><r/>"#,
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(
        attributes(&root),
        [
            ("kind".to_owned(), "plain".to_owned()),
            ("size".to_owned(), "7".to_owned())
        ]
    );
    Ok(())
}

#[test]
fn required_and_implied_declarations_supply_nothing() -> Result<(), oxixml_infoset::XdmError> {
    let tree =
        Tree::parse_str(r#"<!DOCTYPE r [<!ATTLIST r a CDATA #IMPLIED b CDATA #REQUIRED>]><r/>"#)?;
    let root = tree.document_element().expect("a document element");
    assert!(attributes(&root).is_empty(), "{:?}", attributes(&root));
    Ok(())
}

#[test]
fn a_written_attribute_overrides_its_default() -> Result<(), oxixml_infoset::XdmError> {
    let tree = Tree::parse_str(
        r#"<!DOCTYPE r [
             <!ATTLIST r a CDATA "declared" xmlns:p CDATA #FIXED "urn:declared">
           ]><r a="written" xmlns:p="urn:written"/>"#,
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(attributes(&root), [("a".to_owned(), "written".to_owned())]);

    let p = root
        .namespaces()
        .into_iter()
        .find(|namespace| {
            namespace
                .node_name()
                .is_some_and(|name| name.local_name() == "p")
        })
        .expect("the written binding");
    assert_eq!(p.string_value(), "urn:written");
    Ok(())
}

#[test]
fn one_declaration_may_define_several_attributes() -> Result<(), oxixml_infoset::XdmError> {
    // `AttType` has two parenthesized forms, and the `|` and white space inside them must not be
    // mistaken for the start of the next `AttDef`. Only `b` and `d` have a default value.
    let tree = Tree::parse_str(
        r#"<!DOCTYPE e [
             <!ATTLIST e a CDATA #IMPLIED b (x | y) 'x' c NOTATION (n) #REQUIRED d CDATA #FIXED "v">
             <!NOTATION n SYSTEM "n.exe">
           ]><e/>"#,
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(
        attributes(&root),
        [
            ("b".to_owned(), "x".to_owned()),
            ("d".to_owned(), "v".to_owned())
        ]
    );
    Ok(())
}

#[test]
fn the_element_name_is_matched_as_written() -> Result<(), oxixml_infoset::XdmError> {
    // An attribute-list declaration names an element by the qualified name the document writes,
    // which is what Namespaces in XML §7 leaves it as: two prefixes bound to the same namespace
    // are two different declarations.
    let tree = Tree::parse_str(
        r#"<!DOCTYPE p:e [<!ATTLIST p:e a CDATA "yes">]>
           <p:e xmlns:p="urn:p"><q:e xmlns:q="urn:p"/></p:e>"#,
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(attributes(&root), [("a".to_owned(), "yes".to_owned())]);
    let other = root.children().first().cloned().expect("one child");
    assert!(attributes(&other).is_empty(), "{:?}", attributes(&other));
    Ok(())
}

#[test]
fn a_default_value_is_normalized_like_a_written_one() -> Result<(), oxixml_infoset::XdmError> {
    let tree = Tree::parse_str(
        "<!DOCTYPE r [\n\
           <!ENTITY e \"in\">\n\
           <!ATTLIST r wrapped CDATA \"one\ntwo\"\n\
                       referenced CDATA \"[&e;]\"\n\
                       escaped CDATA \"a&#10;b\"\n\
                       tokens NMTOKENS \"  one   two  \">\n\
         ]><r/>",
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(
        attributes(&root),
        [
            // A literal line end in the declaration is white space, so it becomes a space.
            ("wrapped".to_owned(), "one two".to_owned()),
            ("referenced".to_owned(), "[in]".to_owned()),
            // A character reference denotes the character itself and survives normalization.
            ("escaped".to_owned(), "a\nb".to_owned()),
            // A tokenized type has its value trimmed and its runs of space collapsed.
            ("tokens".to_owned(), "one two".to_owned()),
        ]
    );
    Ok(())
}

#[test]
fn defaults_apply_to_every_element_of_the_declared_name() -> Result<(), oxixml_infoset::XdmError> {
    let tree =
        Tree::parse_str(r#"<!DOCTYPE r [<!ATTLIST c n CDATA "d">]><r><c/><c n="w"/><d/></r>"#)?;
    let root = tree.document_element().expect("a document element");
    let children = root.children();
    assert_eq!(attributes(&children[0]), [("n".to_owned(), "d".to_owned())]);
    assert_eq!(attributes(&children[1]), [("n".to_owned(), "w".to_owned())]);
    assert!(attributes(&children[2]).is_empty());
    Ok(())
}

#[test]
fn the_first_declaration_of_an_attribute_binds() -> Result<(), oxixml_infoset::XdmError> {
    let tree = Tree::parse_str(
        r#"<!DOCTYPE r [<!ATTLIST r a CDATA "first"><!ATTLIST r a CDATA "second">]><r/>"#,
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(attributes(&root), [("a".to_owned(), "first".to_owned())]);
    Ok(())
}

#[test]
fn an_element_from_an_entity_is_defaulted_like_any_other() -> Result<(), oxixml_infoset::XdmError> {
    // The replacement text of an entity is parsed as a source of its own, but the elements it
    // yields are elements of the document, so the document's declarations apply to them.
    let tree = Tree::parse_str(
        r#"<!DOCTYPE r [<!ENTITY e "<c/>"><!ATTLIST c n CDATA "d" xmlns:p CDATA "urn:p">]><r>&e;</r>"#,
    )?;
    let c = tree
        .document_element()
        .and_then(|root| root.children().first().cloned())
        .expect("the spliced element");
    assert_eq!(attributes(&c), [("n".to_owned(), "d".to_owned())]);
    assert_eq!(in_scope_prefixes(&c), ["p", "xml"]);
    Ok(())
}

#[test]
fn the_query_test_suites_own_document_reports_the_prefixes_it_expects()
-> Result<(), oxixml_infoset::XdmError> {
    // `fn:in-scope-prefixes` over the document element of this file must answer
    // `("", "xml", "xlink")`. The file is found through `OXIXML_XML_SUITES_DIR`, the same
    // variable the conformance tests use; without it there is nothing to check and the test
    // succeeds silently, exactly as they do.
    let Some(root) = std::env::var_os("OXIXML_XML_SUITES_DIR") else {
        return Ok(());
    };
    let path = std::path::Path::new(&root)
        .join("qt3tests")
        .join("fn")
        .join("in-scope-prefixes")
        .join("NamespaceSuppliedInternally.xml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(());
    };
    let tree = Tree::parse_str(&text)?;
    let svg = tree.document_element().expect("a document element");
    assert_eq!(in_scope_prefixes(&svg), ["", "xlink", "xml"]);
    Ok(())
}
