//! The unparsed entities and notations a document's DTD declares, as the data model carries them.
//!
//! [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode) gives the document node
//! an `unparsed-entities` accessor: a map from entity name to its system identifier (as an
//! absolute URI) and public identifier. `fn:unparsed-entity-uri` and
//! `fn:unparsed-entity-public-id` read exactly that, and the notation an `NDATA` clause names is
//! kept beside it so that a `NOTATION`-typed value can be checked against a declaration.

use oxixml_infoset::{ExpandedName, NodeKind, ParseOptions, Tree, TreeBuilder, XdmError, XdmNode};

const SUBSET: &str = r#"<!DOCTYPE catalog [
    <!NOTATION png SYSTEM "image/png">
    <!NOTATION gif PUBLIC "-//x//NOTATION GIF//EN">
    <!ENTITY logo SYSTEM "img/logo.png" NDATA png>
    <!ENTITY badge PUBLIC "-//x//BADGE//EN" "img/badge.gif" NDATA gif>
    <!ENTITY plain "just text">
]>"#;

fn catalog() -> Result<Tree, XdmError> {
    Tree::parse_str_with(
        &format!("{SUBSET}<catalog>&plain;</catalog>"),
        ParseOptions::default().document_uri("http://example.com/a/b.xml"),
    )
}

#[test]
fn unparsed_entities_reach_the_tree_with_their_identifiers() -> Result<(), XdmError> {
    let tree = catalog()?;
    assert_eq!(tree.unparsed_entity_names(), ["badge", "logo"]);

    let logo = tree.unparsed_entity("logo").expect("`logo` is declared");
    assert_eq!(logo.system_id, "img/logo.png");
    assert_eq!(
        logo.uri.as_deref(),
        Some("http://example.com/a/img/logo.png"),
        "resolved against the document URI"
    );
    assert_eq!(logo.public_id, None);
    assert_eq!(logo.notation, "png");

    let badge = tree.unparsed_entity("badge").expect("`badge` is declared");
    assert_eq!(badge.public_id.as_deref(), Some("-//x//BADGE//EN"));
    assert_eq!(badge.system_id, "img/badge.gif");
    assert_eq!(
        badge.uri.as_deref(),
        Some("http://example.com/a/img/badge.gif")
    );
    assert_eq!(badge.notation, "gif");

    // A parsed internal entity is not one of them; it is expanded into the tree instead.
    assert!(tree.unparsed_entity("plain").is_none());
    assert_eq!(
        tree.document_element().expect("root").string_value(),
        "just text"
    );
    Ok(())
}

#[test]
fn notations_reach_the_tree_with_their_identifiers() -> Result<(), XdmError> {
    let tree = catalog()?;
    assert_eq!(tree.notation_names(), ["gif", "png"]);

    let png = tree.notation("png").expect("`png` is declared");
    assert_eq!(png.public_id, None);
    assert_eq!(png.system_id.as_deref(), Some("image/png"));
    assert_eq!(png.uri.as_deref(), Some("http://example.com/a/image/png"));

    // The `PublicID` form of the declaration carries no system identifier, and therefore no URI.
    let gif = tree.notation("gif").expect("`gif` is declared");
    assert_eq!(gif.public_id.as_deref(), Some("-//x//NOTATION GIF//EN"));
    assert_eq!(gif.system_id, None);
    assert_eq!(gif.uri, None);
    Ok(())
}

#[test]
fn a_relative_system_id_has_no_absolute_uri_without_a_document_uri() -> Result<(), XdmError> {
    let tree = Tree::parse_str(&format!("{SUBSET}<catalog/>"))?;
    let logo = tree.unparsed_entity("logo").expect("`logo` is declared");
    assert_eq!(logo.system_id, "img/logo.png");
    assert_eq!(logo.uri, None, "there is nothing to resolve against");

    // An identifier that is already absolute needs no base URI.
    let tree = Tree::parse_str(
        r#"<!DOCTYPE r [<!ENTITY e SYSTEM "http://example.org/e.bin" NDATA n>]><r/>"#,
    )?;
    let entity = tree.unparsed_entity("e").expect("`e` is declared");
    assert_eq!(entity.uri.as_deref(), Some("http://example.org/e.bin"));
    Ok(())
}

#[test]
fn the_declarations_are_a_property_of_the_document_node() -> Result<(), XdmError> {
    let tree = catalog()?;
    let root = tree.root();
    assert_eq!(root.node_kind(), NodeKind::Document);
    assert_eq!(
        root.unparsed_entity("logo")
            .map(|entity| entity.notation.clone()),
        Some("png".to_owned())
    );
    assert_eq!(root.unparsed_entity_names(), ["badge", "logo"]);
    assert_eq!(root.notation_names(), ["gif", "png"]);

    // Anywhere else in the tree the property does not apply, which is how a caller tells
    // "no such entity" from "not rooted at a document node".
    let element = tree.document_element().expect("a document element");
    assert!(element.unparsed_entity("logo").is_none());
    assert!(element.unparsed_entity_names().is_empty());
    assert!(element.notation("png").is_none());
    assert!(element.notation_names().is_empty());
    // …and climbing to the root is what makes the question answerable again.
    assert!(element.root().unparsed_entity("logo").is_some());
    Ok(())
}

#[test]
fn a_document_without_a_dtd_declares_nothing() -> Result<(), XdmError> {
    let tree = Tree::parse_str("<r/>")?;
    assert!(tree.unparsed_entity_names().is_empty());
    assert!(tree.notation_names().is_empty());
    assert!(tree.unparsed_entity("anything").is_none());
    assert!(tree.notation("anything").is_none());
    assert!(tree.root().unparsed_entity("anything").is_none());
    Ok(())
}

#[test]
fn a_builder_can_declare_them_directly() -> Result<(), XdmError> {
    let mut builder = TreeBuilder::with_document_uri("http://example.com/d.xml");
    builder.declare_notation("png", None, Some("image/png"))?;
    builder.declare_unparsed_entity("logo", Some("-//x//LOGO//EN"), "logo.png", "png")?;
    // The first declaration of a name binds, as it does for parsed entities.
    builder.declare_unparsed_entity("logo", None, "other.png", "png")?;
    builder.start_element(ExpandedName::new(None::<&str>, "r"))?;
    builder.end_element()?;
    let tree = builder.build()?;

    let logo = tree.unparsed_entity("logo").expect("`logo` is declared");
    assert_eq!(logo.system_id, "logo.png");
    assert_eq!(logo.public_id.as_deref(), Some("-//x//LOGO//EN"));
    assert_eq!(logo.uri.as_deref(), Some("http://example.com/logo.png"));
    assert_eq!(
        tree.notation("png").and_then(|png| png.system_id.clone()),
        Some("image/png".to_owned())
    );

    // A declaration has to name what it declares.
    let mut builder = TreeBuilder::new();
    assert!(builder.declare_notation("", None, None).is_err());
    assert!(builder.declare_unparsed_entity("", None, "s", "n").is_err());
    Ok(())
}

#[test]
fn a_reference_to_an_unparsed_entity_is_still_rejected() {
    // XML 1.0 WFC: Parsed Entity — an unparsed entity may only be named by an attribute of type
    // ENTITY or ENTITIES, never referenced from content.
    assert!(Tree::parse_str(&format!("{SUBSET}<catalog>&logo;</catalog>")).is_err());
}
