//! The `(line, column)` a node began at, which parsing records only when asked to.
//!
//! Nothing in the data model has this property; a host language whose diagnostics must name a
//! place in the file the user wrote does. The coordinates are those of the character that opens
//! the node's markup, so a diagnostic points at the construct rather than past it.

use oxixml_infoset::{NodeKind, NodeRef, ParseOptions, SourcePosition, Tree, XdmError, XdmNode};

/// Parses `xml` with source positions recorded.
fn located(xml: &str) -> Result<Tree, XdmError> {
    Tree::parse_str_with(xml, ParseOptions::default().record_source_positions(true))
}

/// `(line, column)` of `node`, for terser assertions.
fn at(node: &NodeRef) -> Option<(u32, u32)> {
    node.source_position()
        .map(|position| (position.line, position.column))
}

#[test]
fn nothing_is_recorded_unless_it_is_asked_for() -> Result<(), XdmError> {
    let tree = Tree::parse_str("<a>\n  <b/>\n</a>")?;
    let a = tree.document_element().expect("a document element");
    assert_eq!(at(&a), None);
    for child in a.children() {
        assert_eq!(at(&child), None);
    }
    Ok(())
}

#[test]
fn an_element_is_located_at_its_start_tag() -> Result<(), XdmError> {
    let tree = located(
        "<?xml version=\"1.0\"?>\n\
         <root>\n\
         \x20 <first/>\n\
         \x20 <second>\n\
         \x20   <deep/>\n\
         \x20 </second>\n\
         </root>",
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(at(&root), Some((2, 1)));

    let elements: Vec<NodeRef> = root
        .children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .collect();
    assert_eq!(at(&elements[0]), Some((3, 3)));
    assert_eq!(at(&elements[1]), Some((4, 3)));

    let deep = elements[1]
        .children()
        .into_iter()
        .find(|child| child.node_kind() == NodeKind::Element)
        .expect("the nested element");
    assert_eq!(at(&deep), Some((5, 5)));
    Ok(())
}

#[test]
fn text_comments_and_processing_instructions_are_located_too() -> Result<(), XdmError> {
    let tree = located("<a>text\n<!--note-->\n<?target data?></a>")?;
    let a = tree.document_element().expect("a document element");
    let children = a.children();
    assert_eq!(children[0].node_kind(), NodeKind::Text);
    // The text node begins immediately after the start tag.
    assert_eq!(at(&children[0]), Some((1, 4)));
    assert_eq!(children[1].node_kind(), NodeKind::Comment);
    assert_eq!(at(&children[1]), Some((2, 1)));
    assert_eq!(children[3].node_kind(), NodeKind::ProcessingInstruction);
    assert_eq!(at(&children[3]), Some((3, 1)));
    Ok(())
}

#[test]
fn a_text_node_is_located_at_its_first_character() -> Result<(), XdmError> {
    // Character data reaches the parser in several pieces — plain text, a reference, a CDATA
    // section — that merge into one node. The node belongs where the first piece was.
    let tree = located("<a>one &amp; two<![CDATA[ three]]></a>")?;
    let text = tree
        .document_element()
        .and_then(|a| a.children().first().cloned())
        .expect("one text node");
    assert_eq!(text.string_value(), "one & two three");
    assert_eq!(at(&text), Some((1, 4)));
    Ok(())
}

#[test]
fn columns_count_characters_rather_than_bytes() -> Result<(), XdmError> {
    let tree = located("<a>\u{e9}\u{1f600}<b/></a>")?;
    let b = tree
        .document_element()
        .and_then(|a| a.children().get(1).cloned())
        .expect("the element after the text");
    assert_eq!(at(&b), Some((1, 6)));
    Ok(())
}

#[test]
fn attribute_namespace_and_document_nodes_have_no_position() -> Result<(), XdmError> {
    let tree = located("<a xmlns:p=\"urn:p\" id=\"x\"/>")?;
    assert_eq!(at(&tree.root()), None);
    let a = tree.document_element().expect("a document element");
    assert_eq!(at(&a), Some((1, 1)));
    for attribute in a.attributes() {
        assert_eq!(at(&attribute), None);
    }
    for namespace in a.namespaces() {
        assert_eq!(at(&namespace), None);
    }
    Ok(())
}

#[test]
fn a_node_that_came_from_an_entity_has_no_position() -> Result<(), XdmError> {
    // The replacement text of an entity is parsed as a source of its own, so an offset into it is
    // not an offset into the document. Reporting one would be a plausible-looking wrong answer.
    let tree = located(
        "<!DOCTYPE r [<!ENTITY e \"<b/>\">]>\n\
         <r>\n\
         \x20 &e;\n\
         \x20 <c/>\n\
         </r>",
    )?;
    let root = tree.document_element().expect("a document element");
    let elements: Vec<NodeRef> = root
        .children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .collect();
    assert_eq!(
        elements[0]
            .node_name()
            .map(|name| name.to_string())
            .as_deref(),
        Some("b")
    );
    assert_eq!(at(&elements[0]), None, "the spliced element");
    // The document's own markup is unaffected by what precedes it.
    assert_eq!(at(&elements[1]), Some((4, 3)));
    Ok(())
}

#[test]
fn every_line_end_of_the_source_starts_a_line() -> Result<(), XdmError> {
    for (name, xml) in [
        ("line feed", "<a>\n<b/></a>"),
        ("carriage return and line feed", "<a>\r\n<b/></a>"),
        ("carriage return", "<a>\r<b/></a>"),
    ] {
        let tree = located(xml)?;
        let b = tree
            .document_element()
            .and_then(|a| a.children().get(1).cloned())
            .expect("the element on the second line");
        assert_eq!(at(&b), Some((2, 1)), "{name}");
    }
    Ok(())
}

#[test]
fn a_recorded_position_survives_whitespace_stripping() -> Result<(), XdmError> {
    // Stripping drops nodes, which renumbers the arena; the surviving nodes must still carry
    // their own coordinates rather than a neighbour's.
    let tree = Tree::parse_str_with(
        "<a>\n  <b/>\n  <c>text</c>\n</a>",
        ParseOptions::default()
            .record_source_positions(true)
            .strip_boundary_whitespace(true),
    )?;
    let a = tree.document_element().expect("a document element");
    let children = a.children();
    assert_eq!(children.len(), 2, "the boundary whitespace is gone");
    assert_eq!(at(&children[0]), Some((2, 3)));
    assert_eq!(at(&children[1]), Some((3, 3)));
    let text = children[1].children().first().cloned().expect("its text");
    assert_eq!(at(&text), Some((3, 6)));
    Ok(())
}

#[test]
fn a_built_tree_has_no_positions() -> Result<(), XdmError> {
    use oxixml_infoset::{ExpandedName, TreeBuilder};

    let mut builder = TreeBuilder::new();
    builder.start_element(ExpandedName::new(None::<&str>, "a"))?;
    builder.text("x")?;
    builder.end_element()?;
    let tree = builder.build()?;
    let a = tree.document_element().expect("a document element");
    assert_eq!(at(&a), None);
    assert_eq!(at(&a.children()[0]), None);
    Ok(())
}

#[test]
fn the_position_type_is_ordered_by_line_then_column() {
    let first = SourcePosition { line: 1, column: 9 };
    let second = SourcePosition { line: 2, column: 1 };
    assert!(first < second);
    assert_eq!(first, SourcePosition { line: 1, column: 9 });
}
