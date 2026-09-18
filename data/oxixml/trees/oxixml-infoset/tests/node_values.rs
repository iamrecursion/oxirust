//! Names, string values and typed values, kind by kind, per XDM 3.1 §6.

mod common;

use common::{find, sample_tree};
use oxixml_infoset::{
    AtomicType, ErrorCode, ExpandedName, Item, NodeKind, Sequence, Tree, TreeBuilder, XdmError,
    XdmNode,
};

fn one_atomic(sequence: &Sequence) -> (AtomicType, String) {
    assert_eq!(sequence.len(), 1, "expected a singleton typed value");
    let value = sequence
        .first()
        .and_then(Item::as_atomic)
        .expect("an atomic item");
    (value.atomic_type(), value.string_value())
}

#[test]
fn node_names_follow_the_specification_for_each_kind() -> Result<(), XdmError> {
    let tree = sample_tree()?;

    assert_eq!(tree.root().node_name(), None, "a document node has no name");
    assert_eq!(
        find(&tree, "r").node_name(),
        Some(ExpandedName::new(None::<&str>, "r"))
    );
    assert_eq!(
        find(&tree, "@{urn:p}a2").node_name(),
        Some(ExpandedName::new(Some("urn:p"), "a2"))
    );
    assert_eq!(find(&tree, "text(t1)").node_name(), None);
    assert_eq!(find(&tree, "comment(mid)").node_name(), None);
    assert_eq!(
        find(&tree, "pi(pi)").node_name(),
        Some(ExpandedName::new(None::<&str>, "pi")),
        "a processing instruction's name is its target, in no namespace"
    );

    let namespace = find(&tree, "c1")
        .namespaces()
        .first()
        .cloned()
        .expect("a namespace node");
    assert_eq!(
        namespace.node_name(),
        Some(ExpandedName::new(None::<&str>, "p")),
        "a namespace node's name is its prefix, in no namespace"
    );
    Ok(())
}

#[test]
fn string_values_follow_the_specification_for_each_kind() -> Result<(), XdmError> {
    let tree = sample_tree()?;

    assert_eq!(
        tree.root().string_value(),
        "t1xt2",
        "a document's string value concatenates its descendant text nodes"
    );
    assert_eq!(find(&tree, "r").string_value(), "t1xt2");
    assert_eq!(find(&tree, "c1").string_value(), "x");
    assert_eq!(find(&tree, "c2").string_value(), "");
    assert_eq!(find(&tree, "@a1").string_value(), "v1");
    assert_eq!(find(&tree, "text(t1)").string_value(), "t1");
    assert_eq!(find(&tree, "comment(mid)").string_value(), "mid");
    assert_eq!(
        find(&tree, "pi(pi)").string_value(),
        "data",
        "a processing instruction's string value excludes its target"
    );

    let namespace = find(&tree, "c1")
        .namespaces()
        .first()
        .cloned()
        .expect("a namespace node");
    assert_eq!(namespace.string_value(), "urn:p");
    Ok(())
}

#[test]
fn typed_values_of_an_unvalidated_tree() -> Result<(), XdmError> {
    let tree = sample_tree()?;

    for (label, expected_type, expected_value) in [
        ("/", AtomicType::UntypedAtomic, "t1xt2"),
        ("r", AtomicType::UntypedAtomic, "t1xt2"),
        ("@a1", AtomicType::UntypedAtomic, "v1"),
        ("text(t1)", AtomicType::UntypedAtomic, "t1"),
        ("comment(mid)", AtomicType::String, "mid"),
        ("pi(pi)", AtomicType::String, "data"),
    ] {
        let node = find(&tree, label);
        assert_eq!(
            one_atomic(&node.typed_value()?),
            (expected_type, expected_value.to_owned()),
            "typed value of {label}"
        );
    }

    let namespace = find(&tree, "c1")
        .namespaces()
        .first()
        .cloned()
        .expect("a namespace node");
    assert_eq!(
        one_atomic(&namespace.typed_value()?),
        (AtomicType::String, "urn:p".to_owned())
    );
    Ok(())
}

#[test]
fn atomizing_a_node_is_atomizing_its_typed_value() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    let element = find(&tree, "c1");
    let atomized = Sequence::from(element.clone()).atomize()?;
    assert_eq!(atomized, element.typed_value()?);
    assert_eq!(one_atomic(&atomized).1, "x");
    Ok(())
}

#[test]
fn schema_annotations_replace_the_untyped_defaults() -> Result<(), XdmError> {
    use oxixml_infoset::{Annotation, AtomicValue};

    let mut builder = TreeBuilder::new();
    builder.start_annotated_element(
        ExpandedName::new(None::<&str>, "measurement"),
        Annotation {
            type_name: Some(AtomicType::Double.expanded_name()),
            typed_value: Some(Sequence::from(AtomicValue::double(2.5))),
            nilled: Some(false),
            ..Annotation::default()
        },
    )?;
    builder.annotated_attribute(
        ExpandedName::new(None::<&str>, "ref"),
        "m1 m2",
        Annotation {
            type_name: Some(AtomicType::Idref.expanded_name()),
            typed_value: Some(Sequence::from_items([
                Item::from(AtomicValue::parse(AtomicType::Idref, "m1")?),
                Item::from(AtomicValue::parse(AtomicType::Idref, "m2")?),
            ])),
            is_idref: true,
            ..Annotation::default()
        },
    )?;
    builder.text("2.5")?;
    builder.end_element()?;
    let tree = builder.build()?;

    let element = tree.document_element().expect("a document element");
    assert_eq!(
        element.type_name(),
        Some(AtomicType::Double.expanded_name())
    );
    assert_eq!(one_atomic(&element.typed_value()?).0, AtomicType::Double);
    assert_eq!(element.nilled(), Some(false));
    assert_eq!(
        element.string_value(),
        "2.5",
        "the string value still comes from the text content"
    );

    let attribute = element
        .attributes()
        .first()
        .cloned()
        .expect("one attribute");
    assert!(attribute.is_idref());
    assert!(!attribute.is_id());
    assert_eq!(
        attribute.typed_value()?.len(),
        2,
        "an IDREFS attribute atomizes to a sequence"
    );
    assert_eq!(attribute.nilled(), None, "only elements can be nilled");
    Ok(())
}

#[test]
fn a_store_may_report_that_a_node_has_no_typed_value() {
    // The default tree never raises FOTY0012, so demonstrate that the error code exists and is
    // what the trait contract names, keeping the constant honest for schema-aware backends.
    let error = XdmError::new(ErrorCode::Foty0012, "element-only content");
    assert_eq!(error.code(), ErrorCode::Foty0012);
    assert!(error.to_string().starts_with("[FOTY0012]"));
}

#[test]
fn base_and_document_uris() -> Result<(), XdmError> {
    use oxixml_infoset::ParseOptions;

    let tree = Tree::parse_str_with(
        r#"<a xml:base="sub/"><b xml:base="deep/"><c/></b></a>"#,
        ParseOptions::default().document_uri("http://example.com/root/doc.xml"),
    )?;

    assert_eq!(tree.document_uri(), Some("http://example.com/root/doc.xml"));
    assert_eq!(
        tree.root().document_uri().as_deref(),
        Some("http://example.com/root/doc.xml")
    );
    let a = tree.document_element().expect("a document element");
    assert_eq!(
        a.document_uri(),
        None,
        "only the document node carries a document URI"
    );
    assert_eq!(
        a.base_uri().as_deref(),
        Some("http://example.com/root/sub/")
    );

    let b = a.children().first().cloned().expect("one child");
    assert_eq!(
        b.base_uri().as_deref(),
        Some("http://example.com/root/sub/deep/")
    );
    let c = b.children().first().cloned().expect("one grandchild");
    assert_eq!(
        c.base_uri().as_deref(),
        Some("http://example.com/root/sub/deep/"),
        "an element with no xml:base inherits"
    );
    Ok(())
}

#[test]
fn is_id_recognizes_xml_id_without_a_dtd() -> Result<(), XdmError> {
    let tree = Tree::parse_str(r#"<a xml:id="x" id="y"/>"#)?;
    let attributes = tree
        .document_element()
        .expect("a document element")
        .attributes();
    assert_eq!(attributes.len(), 2);
    assert!(attributes[0].is_id());
    assert!(
        !attributes[1].is_id(),
        "a plain `id` attribute is only an ID when a DTD or schema says so"
    );
    assert!(!attributes[0].is_idref());
    Ok(())
}

#[test]
fn kinds_and_parents_are_consistent_across_the_whole_tree() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    for node in tree.nodes_in_document_order() {
        match node.node_kind() {
            NodeKind::Document => assert!(node.parent().is_none()),
            NodeKind::Attribute | NodeKind::Namespace => {
                let parent = node.parent().expect("an owner element");
                assert_eq!(parent.node_kind(), NodeKind::Element);
                assert!(
                    !parent.children().contains(&node),
                    "attribute and namespace nodes are not children"
                );
                let owned = if node.node_kind() == NodeKind::Attribute {
                    parent.attributes()
                } else {
                    parent.namespaces()
                };
                assert!(owned.contains(&node));
            }
            _ => {
                let parent = node.parent().expect("a parent");
                assert!(parent.children().contains(&node));
            }
        }
        assert_eq!(node.root(), tree.root());
    }
    Ok(())
}

#[test]
fn no_two_adjacent_text_nodes_ever_appear() -> Result<(), XdmError> {
    let tree = Tree::parse_str("<a>one<![CDATA[two]]>&amp;three<b/>four</a>")?;
    let a = tree.document_element().expect("a document element");
    let kinds: Vec<NodeKind> = a.children().iter().map(XdmNode::node_kind).collect();
    assert_eq!(kinds, [NodeKind::Text, NodeKind::Element, NodeKind::Text]);
    assert_eq!(a.string_value(), "onetwo&threefour");
    Ok(())
}
