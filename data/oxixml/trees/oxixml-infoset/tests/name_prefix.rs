//! The prefix a name was *written* with has to survive into the node.
//!
//! [XDM 3.1 §5.4](https://www.w3.org/TR/xpath-datamodel-31/#dm-node-name) makes a node's name a
//! *triple* — prefix, namespace name, local name — and keeps the prefix so that `fn:name`,
//! `fn:node-name` and `fn:prefix-from-QName` can report the spelling the document used, even
//! though the prefix takes no part in equality. These tests pin that end of the contract for
//! both ways a tree is built here (parsing and [`TreeBuilder`]) so that the function library
//! above can rely on the data being there.

use oxixml_infoset::{
    ExpandedName, QNameValue, Tree, TreeBuilder, XSD_NAMESPACE, XdmError, XdmNode,
};

fn prefix_of(node: &impl XdmNode) -> Option<String> {
    node.node_name()
        .and_then(|name| name.prefix().map(str::to_owned))
}

#[test]
fn a_parsed_element_and_attribute_keep_their_prefix() -> Result<(), XdmError> {
    let tree = Tree::parse_str(
        r#"<pre:e xmlns:pre="urn:p" xmlns="urn:d" pre:a="1" b="2"><plain/></pre:e>"#,
    )?;
    let element = tree.document_element().expect("a document element");
    assert_eq!(prefix_of(&element).as_deref(), Some("pre"));
    assert_eq!(
        element.node_name().map(|name| name.local_name().to_owned()),
        Some("e".to_owned())
    );

    let attributes = element.attributes();
    assert_eq!(prefix_of(&attributes[0]).as_deref(), Some("pre"));
    assert_eq!(
        prefix_of(&attributes[1]),
        None,
        "an unprefixed attribute has no prefix to retain"
    );

    // A name spelled without a prefix keeps none, even when it is in the default namespace.
    let child = element.children().first().cloned().expect("one child");
    assert_eq!(prefix_of(&child), None);
    assert_eq!(
        child
            .node_name()
            .and_then(|n| n.namespace().map(str::to_owned))
            .as_deref(),
        Some("urn:d")
    );
    Ok(())
}

#[test]
fn a_built_element_and_attribute_keep_their_prefix() -> Result<(), XdmError> {
    let mut builder = TreeBuilder::new();
    builder.start_element(ExpandedName::new(Some("urn:p"), "e").with_prefix("pre"))?;
    builder.declare_namespace(Some("pre"), "urn:p")?;
    builder.attribute(
        ExpandedName::new(Some("urn:p"), "a").with_prefix("pre"),
        "1",
    )?;
    builder.end_element()?;
    let tree = builder.build()?;

    let element = tree.document_element().expect("a document element");
    assert_eq!(prefix_of(&element).as_deref(), Some("pre"));
    assert_eq!(
        prefix_of(&element.attributes()[0]).as_deref(),
        Some("pre"),
        "the builder must not drop an attribute's prefix"
    );
    Ok(())
}

#[test]
fn the_prefix_reaches_the_qname_a_node_name_atomizes_to() -> Result<(), XdmError> {
    // This is the step `fn:node-name` takes: the node's name becomes an `xs:QName` value, whose
    // third component is the retained prefix and whose lexical form therefore shows it.
    let tree = Tree::parse_str(r#"<pre:e xmlns:pre="urn:p" pre:a="1"/>"#)?;
    let element = tree.document_element().expect("a document element");
    let name = QNameValue::from(element.node_name().expect("an element has a name"));
    assert_eq!(name.prefix(), Some("pre"));
    assert_eq!(name.lexical_form(), "pre:e");
    assert_eq!(name.namespace(), Some("urn:p"));

    let attribute = QNameValue::from(
        element.attributes()[0]
            .node_name()
            .expect("an attribute has a name"),
    );
    assert_eq!(attribute.lexical_form(), "pre:a");

    // The prefix is an annotation, never part of the value.
    assert_eq!(name, QNameValue::new(Some("other"), Some("urn:p"), "e"));
    assert_ne!(name, QNameValue::new(Some("pre"), Some(XSD_NAMESPACE), "e"));
    Ok(())
}

#[test]
fn the_prefix_survives_every_hop_of_the_tree_builder() -> Result<(), XdmError> {
    // Deeply nested, so the materialization work-list (which is not a simple recursion) is
    // exercised rather than the single-element path.
    let depth = 40;
    let mut builder = TreeBuilder::new();
    for index in 0..depth {
        builder.start_element(
            ExpandedName::new(Some("urn:p"), format!("e{index}")).with_prefix("pre"),
        )?;
        if index == 0 {
            builder.declare_namespace(Some("pre"), "urn:p")?;
        }
        builder.attribute(
            ExpandedName::new(Some("urn:p"), format!("a{index}")).with_prefix("pre"),
            "v",
        )?;
    }
    for _ in 0..depth {
        builder.end_element()?;
    }
    let tree = builder.build()?;

    let mut node = tree.document_element().expect("a document element");
    for index in 0..depth {
        assert_eq!(prefix_of(&node).as_deref(), Some("pre"), "level {index}");
        assert_eq!(
            prefix_of(&node.attributes()[0]).as_deref(),
            Some("pre"),
            "level {index}"
        );
        if let Some(child) = node.children().first().cloned() {
            node = child;
        }
    }
    Ok(())
}
