//! Trees whose root is an element rather than a document node.
//!
//! [XDM 3.1 §6](https://www.w3.org/TR/xpath-datamodel-31/#Node) requires only that every node
//! belong to a tree with a unique root; a document node is one possible root among several. The
//! case that needs the other one is an XSD 1.1 assertion ([XML Schema 1.1 Part 1
//! §3.13.4](https://www.w3.org/TR/xmlschema11-1/#cvc-assertion)), which is evaluated against a
//! data model instance whose root *is* the element being asserted, so that a descendant search
//! cannot climb out of it.

use oxixml_infoset::{
    Axis, ExpandedName, NodeKind, Sequence, Tree, TreeBuilder, XdmError, XdmNode, axis_nodes,
    compare_document_order, deep_equal,
};
use std::cmp::Ordering;

/// The namespace the `xml` prefix is permanently bound to.
const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";

/// `<x a="1"><y>text</y><!--c--></x>`, rooted at `x` itself.
fn element_rooted_sample() -> Result<Tree, XdmError> {
    let mut builder = TreeBuilder::new_element_rooted();
    builder.start_element(ExpandedName::new(None::<&str>, "x"))?;
    builder.attribute(ExpandedName::new(None::<&str>, "a"), "1")?;
    builder.start_element(ExpandedName::new(None::<&str>, "y"))?;
    builder.text("text")?;
    builder.end_element()?;
    builder.comment("c")?;
    builder.end_element()?;
    builder.build()
}

#[test]
fn the_root_is_the_element_and_has_no_parent() -> Result<(), XdmError> {
    let tree = element_rooted_sample()?;
    let root = tree.root();
    assert_eq!(root.node_kind(), NodeKind::Element);
    assert_eq!(
        root.node_name().map(|name| name.to_string()).as_deref(),
        Some("x")
    );
    assert_eq!(root.parent(), None);
    assert_eq!(root.root(), root, "fn:root(.) is the element itself");
    // `document_element` reports the same node: the tree *is* its element.
    assert_eq!(tree.document_element(), Some(root));
    Ok(())
}

#[test]
fn every_descendant_roots_at_the_element() -> Result<(), XdmError> {
    let tree = element_rooted_sample()?;
    let root = tree.root();
    for node in tree.nodes_in_document_order() {
        assert_eq!(node.root(), root, "{node:?}");
    }
    // Nothing above the element is reachable on the reverse axes.
    let y = root.children().first().cloned().expect("the `y` child");
    assert_eq!(axis_nodes(&y, Axis::Ancestor), vec![root.clone()]);
    assert_eq!(
        axis_nodes(&y, Axis::AncestorOrSelf),
        vec![y.clone(), root.clone()]
    );
    let text = y.children().first().cloned().expect("the text node");
    assert_eq!(axis_nodes(&text, Axis::Ancestor), vec![y, root]);
    Ok(())
}

#[test]
fn the_usual_data_model_invariants_still_hold() -> Result<(), XdmError> {
    let tree = element_rooted_sample()?;
    let nodes = tree.nodes_in_document_order();
    assert_eq!(nodes.len(), tree.len());
    assert!(!tree.is_empty());
    for pair in nodes.windows(2) {
        assert_eq!(
            compare_document_order(&pair[0], &pair[1]),
            Ordering::Less,
            "document order is not strictly increasing at {:?}",
            pair[0]
        );
    }
    // Parent is the inverse of children/attributes/namespaces, everywhere but at the root.
    for node in &nodes {
        for child in node.children() {
            assert_eq!(child.parent().as_ref(), Some(node));
        }
        for attribute in node.attributes() {
            assert_eq!(attribute.parent().as_ref(), Some(node));
        }
        for namespace in node.namespaces() {
            assert_eq!(namespace.parent().as_ref(), Some(node));
        }
    }
    assert_eq!(
        nodes.iter().filter(|node| node.parent().is_none()).count(),
        1,
        "exactly one node has no parent"
    );
    // The implicit `xml` binding is materialized on every element, as it is anywhere else.
    let root = tree.root();
    assert_eq!(root.namespaces().len(), 1);
    assert_eq!(root.string_value(), "text");
    Ok(())
}

#[test]
fn namespaces_declared_on_the_root_are_in_scope_below_it() -> Result<(), XdmError> {
    let mut builder = TreeBuilder::new_element_rooted();
    builder.start_element(ExpandedName::new(Some("urn:p"), "x"))?;
    builder.declare_namespace(Some("p"), "urn:p")?;
    builder.start_element(ExpandedName::new(Some("urn:p"), "y"))?;
    builder.end_element()?;
    builder.end_element()?;
    let tree = builder.build()?;

    let prefixes = |node: &oxixml_infoset::NodeRef| -> Vec<String> {
        node.namespaces()
            .iter()
            .filter_map(XdmNode::node_name)
            .map(|name| name.local_name().to_owned())
            .collect()
    };
    let root = tree.root();
    assert_eq!(prefixes(&root), ["p", "xml"]);
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(prefixes(&child), ["p", "xml"]);
    Ok(())
}

#[test]
fn an_element_rooted_tree_has_no_document_uri_but_does_have_a_base_uri() -> Result<(), XdmError> {
    let mut builder = TreeBuilder::element_rooted_with_base_uri("http://example.com/a/b.xml");
    builder.start_element(ExpandedName::new(None::<&str>, "x"))?;
    builder.start_element(ExpandedName::new(None::<&str>, "y"))?;
    builder.attribute(ExpandedName::new(Some(XML_NAMESPACE), "base"), "c/")?;
    builder.end_element()?;
    builder.end_element()?;
    let tree = builder.build()?;

    let root = tree.root();
    assert_eq!(tree.document_uri(), None, "there is no document node");
    assert_eq!(root.document_uri(), None);
    assert_eq!(
        root.base_uri().as_deref(),
        Some("http://example.com/a/b.xml")
    );
    let child = root.children().first().cloned().expect("one child");
    assert_eq!(child.base_uri().as_deref(), Some("http://example.com/a/c/"));
    Ok(())
}

#[test]
fn it_is_deep_equal_to_the_same_element_inside_a_document() -> Result<(), XdmError> {
    let parsed = Tree::parse_str(r#"<doc><x a="1"><y>text</y><!--c--></x></doc>"#)?;
    let inside = parsed
        .document_element()
        .and_then(|doc| doc.children().first().cloned())
        .expect("the `x` element");
    let alone = element_rooted_sample()?.root();
    assert!(deep_equal(&Sequence::from(inside), &Sequence::from(alone))?);
    Ok(())
}

// ---------------------------------------------------------------------------
// What an element-rooted builder refuses
// ---------------------------------------------------------------------------

#[test]
fn a_tree_with_no_element_is_refused_rather_than_built_empty() {
    // A store with no nodes at all would answer every accessor with a default, so this must be
    // an error and not a `Tree`.
    assert!(TreeBuilder::new_element_rooted().build().is_err());
}

#[test]
fn nothing_may_sit_beside_the_root_element() -> Result<(), XdmError> {
    // A comment or a processing instruction at the top level would need a node above the root.
    let mut builder = TreeBuilder::new_element_rooted();
    assert!(builder.comment("beside").is_err());
    assert!(builder.processing_instruction("t", "beside").is_err());
    assert!(builder.text("beside").is_err(), "no element is open");

    // …but both are fine inside the element.
    builder.start_element(ExpandedName::new(None::<&str>, "x"))?;
    assert!(builder.comment("inside").is_ok());
    assert!(builder.processing_instruction("t", "inside").is_ok());
    builder.end_element()?;
    assert!(builder.clone().build().is_ok());

    // A second element at the top level would be a second root.
    assert!(
        builder
            .start_element(ExpandedName::new(None::<&str>, "z"))
            .is_err(),
        "the one document element was already seen"
    );
    Ok(())
}

#[test]
fn dtd_declarations_need_a_document_node() {
    let mut builder = TreeBuilder::new_element_rooted();
    assert!(
        builder
            .declare_unparsed_entity("logo", None, "logo.png", "png")
            .is_err()
    );
    assert!(
        builder
            .declare_notation("png", None, Some("image/png"))
            .is_err()
    );
}

#[test]
fn a_document_rooted_tree_is_unchanged() -> Result<(), XdmError> {
    // The tripwire for the additive API: the default builder still puts a document node on top.
    let mut builder = TreeBuilder::new();
    builder.comment("before")?;
    builder.start_element(ExpandedName::new(None::<&str>, "x"))?;
    builder.end_element()?;
    builder.comment("after")?;
    let tree = builder.build()?;

    let root = tree.root();
    assert_eq!(root.node_kind(), NodeKind::Document);
    assert_eq!(root.parent(), None);
    assert_eq!(root.children().len(), 3);
    let element = tree.document_element().expect("a document element");
    assert_eq!(element.node_kind(), NodeKind::Element);
    assert_eq!(element.parent(), Some(root));
    Ok(())
}
