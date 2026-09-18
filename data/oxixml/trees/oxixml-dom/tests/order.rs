//! Document order and the navigation axes.
//!
//! Document order is the order of
//! [XPath 1.0 §5](https://www.w3.org/TR/xpath-10/#data-model): the root first,
//! then every node in the order its first character appears in the serialized
//! document, with an element's attributes placed after the element and before
//! its children.

use std::cmp::Ordering;

use oxixml_dom::{Document, NodeId, NodeKind, build};

const SAMPLE: &str = r#"<r a="1" b="2"><x><y/>text</x><!--c--><z><?pi d?></z></r>"#;

fn labels(doc: &Document, nodes: impl IntoIterator<Item = NodeId>) -> Vec<String> {
    nodes.into_iter().map(|node| label(doc, node)).collect()
}

fn label(doc: &Document, node: NodeId) -> String {
    match doc.kind(node) {
        Some(NodeKind::Document) => "#document".to_owned(),
        Some(NodeKind::Element) => doc.node_name(node).unwrap_or_default(),
        Some(NodeKind::Attr) => format!("@{}", doc.node_name(node).unwrap_or_default()),
        Some(NodeKind::Text) => format!("'{}'", doc.character_data(node).unwrap_or_default()),
        Some(NodeKind::Comment) => {
            format!("<!--{}-->", doc.character_data(node).unwrap_or_default())
        }
        Some(NodeKind::ProcessingInstruction) => {
            format!("<?{}?>", doc.node_name(node).unwrap_or_default())
        }
        Some(NodeKind::CdataSection) => "#cdata".to_owned(),
        Some(other) => other.name().to_owned(),
        None => "?".to_owned(),
    }
}

#[test]
fn document_order_places_attributes_before_children() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(SAMPLE)?;
    assert_eq!(
        labels(&doc, doc.nodes_in_document_order()),
        [
            "#document",
            "r",
            "@a",
            "@b",
            "x",
            "y",
            "'text'",
            "<!--c-->",
            "z",
            "<?pi?>",
        ]
    );
    Ok(())
}

#[test]
fn document_order_is_a_total_order() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(SAMPLE)?;
    let ordered = doc.nodes_in_document_order();
    for window in ordered.windows(2) {
        let [first, second] = window else { continue };
        assert_eq!(
            doc.compare_document_order(*first, *second),
            Some(Ordering::Less),
            "{} must precede {}",
            label(&doc, *first),
            label(&doc, *second)
        );
        assert_eq!(
            doc.compare_document_order(*second, *first),
            Some(Ordering::Greater)
        );
    }
    for node in &ordered {
        assert_eq!(
            doc.compare_document_order(*node, *node),
            Some(Ordering::Equal)
        );
    }
    assert_eq!(
        doc.compare_document_order(doc.root(), NodeId::from_index(9_999)),
        None
    );
    Ok(())
}

#[test]
fn document_order_is_recomputed_after_a_mutation() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = build::parse_str("<r><a/><b/></r>")?;
    let root = doc.document_element().expect("a document element");
    let a = doc.first_child(root).expect("a");
    let b = doc.last_child(root).expect("b");
    assert_eq!(
        doc.compare_document_order(a, b),
        Some(Ordering::Less),
        "a precedes b to begin with"
    );

    doc.remove_child(root, a)?;
    doc.append_child(root, a)?;
    assert_eq!(
        doc.compare_document_order(a, b),
        Some(Ordering::Greater),
        "after the move, b precedes a"
    );
    assert_eq!(doc.to_xml(), "<r><b/><a/></r>");
    Ok(())
}

#[test]
fn detached_nodes_keep_a_consistent_relative_order() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = build::parse_str("<r><a><inner/></a></r>")?;
    let root = doc.document_element().expect("a document element");
    let a = doc.first_child(root).expect("a");
    let inner = doc.first_child(a).expect("inner");
    doc.remove_child(root, a)?;

    let root_order = doc.document_order(root).expect("root has an order");
    let a_order = doc.document_order(a).expect("a has an order");
    let inner_order = doc.document_order(inner).expect("inner has an order");
    assert!(root_order < a_order, "the live tree comes first");
    assert!(
        a_order < inner_order,
        "a detached subtree keeps its own document order"
    );
    Ok(())
}

#[test]
fn the_following_axis_skips_descendants() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(SAMPLE)?;
    let x = doc
        .get_elements_by_tag_name(doc.root(), "x")
        .next()
        .expect("the x element");
    assert_eq!(labels(&doc, doc.following(x)), ["<!--c-->", "z", "<?pi?>"]);

    let y = doc
        .get_elements_by_tag_name(doc.root(), "y")
        .next()
        .expect("the y element");
    assert_eq!(
        labels(&doc, doc.following(y)),
        ["'text'", "<!--c-->", "z", "<?pi?>"]
    );
    Ok(())
}

#[test]
fn the_preceding_axis_skips_ancestors() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(SAMPLE)?;
    let z = doc
        .get_elements_by_tag_name(doc.root(), "z")
        .next()
        .expect("the z element");
    // In reverse document order, and without the ancestors `r` and `#document`.
    assert_eq!(
        labels(&doc, doc.preceding(z)),
        ["<!--c-->", "'text'", "y", "x"]
    );

    let x = doc
        .get_elements_by_tag_name(doc.root(), "x")
        .next()
        .expect("the x element");
    assert_eq!(labels(&doc, doc.preceding(x)), Vec::<String>::new());
    Ok(())
}

#[test]
fn the_axes_of_an_attribute_start_at_its_element() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(SAMPLE)?;
    let root = doc.document_element().expect("a document element");
    let attribute = doc.get_attribute_node(root, "a").expect("the a attribute");

    assert_eq!(doc.parent(attribute), None, "an attribute has no parent");
    assert_eq!(doc.owner_element(attribute), Some(root));
    assert_eq!(labels(&doc, doc.ancestors(attribute)), ["r", "#document"]);
    assert!(doc.is_ancestor_of(root, attribute));
    assert_eq!(
        labels(&doc, doc.following(attribute)),
        ["x", "y", "'text'", "<!--c-->", "z", "<?pi?>"]
    );
    Ok(())
}

#[test]
fn ancestors_descendants_and_siblings_agree_with_document_order()
-> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(SAMPLE)?;
    let root = doc.document_element().expect("a document element");

    assert_eq!(
        labels(&doc, doc.descendants(root)),
        ["x", "y", "'text'", "<!--c-->", "z", "<?pi?>"]
    );
    assert_eq!(
        labels(&doc, doc.traverse(root)),
        ["r", "x", "y", "'text'", "<!--c-->", "z", "<?pi?>"]
    );

    let y = doc
        .get_elements_by_tag_name(doc.root(), "y")
        .next()
        .expect("the y element");
    assert_eq!(labels(&doc, doc.ancestors(y)), ["x", "r", "#document"]);
    assert_eq!(doc.tree_root(y), Some(doc.root()));
    assert_eq!(labels(&doc, doc.following_siblings(y)), ["'text'"]);
    assert_eq!(
        labels(&doc, doc.preceding_siblings(y)),
        Vec::<String>::new()
    );

    let x = doc
        .get_elements_by_tag_name(doc.root(), "x")
        .next()
        .expect("the x element");
    assert_eq!(labels(&doc, doc.following_siblings(x)), ["<!--c-->", "z"]);

    let z = doc
        .get_elements_by_tag_name(doc.root(), "z")
        .next()
        .expect("the z element");
    assert_eq!(labels(&doc, doc.preceding_siblings(z)), ["<!--c-->", "x"]);
    Ok(())
}

#[test]
fn document_order_matches_a_serialization_walk() -> Result<(), Box<dyn std::error::Error>> {
    // Independent check: for every pair of *element* nodes, the one whose start
    // tag appears earlier in the serialized document must come earlier in
    // document order.
    let xml = r#"<r><a><b/><c><d/></c></a><e/><f><g><h/></g></f></r>"#;
    let doc = build::parse_str(xml)?;

    let elements: Vec<NodeId> = doc
        .descendants(doc.root())
        .filter(|&node| doc.kind(node) == Some(NodeKind::Element))
        .collect();
    let mut by_offset: Vec<(usize, NodeId)> = elements
        .iter()
        .map(|&node| {
            let name = doc.node_name(node).unwrap_or_default();
            let offset = xml.find(&format!("<{name}")).expect("the start tag");
            (offset, node)
        })
        .collect();
    by_offset.sort_by_key(|(offset, _)| *offset);

    for window in by_offset.windows(2) {
        let [(_, first), (_, second)] = window else {
            continue;
        };
        assert_eq!(
            doc.compare_document_order(*first, *second),
            Some(Ordering::Less),
            "{} vs {}",
            label(&doc, *first),
            label(&doc, *second)
        );
    }
    Ok(())
}

#[test]
fn deep_trees_have_a_correct_order() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = Document::new();
    let mut parent = doc.root();
    let mut chain = Vec::new();
    for _ in 0..1_000 {
        let element = doc.create_element("n")?;
        doc.append_child(parent, element)?;
        chain.push(element);
        parent = element;
    }
    for window in chain.windows(2) {
        let [first, second] = window else { continue };
        assert_eq!(
            doc.compare_document_order(*first, *second),
            Some(Ordering::Less)
        );
    }
    assert_eq!(doc.ancestors(parent).count(), 1_000);
    Ok(())
}
