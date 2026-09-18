//! The thirteen XPath axes over a known tree: exact node sequences, exact order.

mod common;

use common::{find, labels, sample_tree};
use oxixml_infoset::{
    Axis, NodeKind, NodeRef, Tree, XdmError, XdmNode, axis_nodes, compare_document_order,
    is_ancestor_of,
};
use std::cmp::Ordering;

fn axis(node: &NodeRef, axis: Axis) -> Vec<String> {
    labels(&axis_nodes(node, axis))
}

#[test]
fn the_document_is_laid_out_in_the_expected_order() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    assert_eq!(
        labels(&tree.nodes_in_document_order()),
        [
            "/",
            "comment(before)",
            "r",
            "ns:p",
            "ns:xml",
            "@a1",
            "@{urn:p}a2",
            "text(t1)",
            "c1",
            "ns:p",
            "ns:xml",
            "text(x)",
            "comment(mid)",
            "c2",
            "ns:p",
            "ns:xml",
            "g1",
            "ns:p",
            "ns:xml",
            "g2",
            "ns:p",
            "ns:xml",
            "pi(pi)",
            "text(t2)",
            "comment(after)",
        ],
        "an element precedes its namespace nodes, which precede its attributes, \
         which precede its children"
    );
    assert_eq!(tree.len(), 25);
    Ok(())
}

#[test]
fn every_axis_from_an_element() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    let c2 = find(&tree, "c2");

    assert_eq!(axis(&c2, Axis::SelfAxis), ["c2"]);
    assert_eq!(axis(&c2, Axis::Child), ["g1", "g2"]);
    assert_eq!(axis(&c2, Axis::Descendant), ["g1", "g2"]);
    assert_eq!(axis(&c2, Axis::DescendantOrSelf), ["c2", "g1", "g2"]);
    assert_eq!(axis(&c2, Axis::Parent), ["r"]);
    assert_eq!(axis(&c2, Axis::Ancestor), ["r", "/"]);
    assert_eq!(axis(&c2, Axis::AncestorOrSelf), ["c2", "r", "/"]);
    assert_eq!(axis(&c2, Axis::FollowingSibling), ["pi(pi)", "text(t2)"]);
    assert_eq!(
        axis(&c2, Axis::PrecedingSibling),
        ["comment(mid)", "c1", "text(t1)"],
        "reverse document order, nearest sibling first"
    );
    assert_eq!(
        axis(&c2, Axis::Following),
        ["pi(pi)", "text(t2)", "comment(after)"],
        "descendants are excluded; attribute and namespace nodes never appear"
    );
    assert_eq!(
        axis(&c2, Axis::Preceding),
        [
            "comment(mid)",
            "text(x)",
            "c1",
            "text(t1)",
            "comment(before)"
        ],
        "reverse document order; ancestors are excluded"
    );
    assert_eq!(axis(&c2, Axis::Attribute), Vec::<String>::new());
    assert_eq!(axis(&c2, Axis::Namespace), ["ns:p", "ns:xml"]);
    Ok(())
}

#[test]
fn every_axis_from_an_attribute() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    let a1 = find(&tree, "@a1");

    assert_eq!(axis(&a1, Axis::SelfAxis), ["@a1"]);
    assert_eq!(axis(&a1, Axis::Child), Vec::<String>::new());
    assert_eq!(axis(&a1, Axis::Descendant), Vec::<String>::new());
    assert_eq!(axis(&a1, Axis::DescendantOrSelf), ["@a1"]);
    assert_eq!(
        axis(&a1, Axis::Parent),
        ["r"],
        "an attribute has a parent even though it is not its parent's child"
    );
    assert_eq!(axis(&a1, Axis::Ancestor), ["r", "/"]);
    assert_eq!(axis(&a1, Axis::AncestorOrSelf), ["@a1", "r", "/"]);
    assert_eq!(
        axis(&a1, Axis::FollowingSibling),
        Vec::<String>::new(),
        "the sibling axes are empty for an attribute, even with a second attribute present"
    );
    assert_eq!(axis(&a1, Axis::PrecedingSibling), Vec::<String>::new());
    assert_eq!(
        axis(&a1, Axis::Following),
        [
            "text(t1)",
            "c1",
            "text(x)",
            "comment(mid)",
            "c2",
            "g1",
            "g2",
            "pi(pi)",
            "text(t2)",
            "comment(after)",
        ],
        "the owner element's content follows its attributes in document order"
    );
    assert_eq!(axis(&a1, Axis::Preceding), ["comment(before)"]);
    assert_eq!(axis(&a1, Axis::Attribute), Vec::<String>::new());
    assert_eq!(axis(&a1, Axis::Namespace), Vec::<String>::new());
    Ok(())
}

#[test]
fn every_axis_from_a_namespace_node() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    let namespaces = find(&tree, "c1").namespaces();
    let prefix_p = namespaces
        .first()
        .cloned()
        .expect("c1 has a namespace node for `p`");
    assert_eq!(prefix_p.node_kind(), NodeKind::Namespace);
    assert_eq!(prefix_p.string_value(), "urn:p");

    assert_eq!(axis(&prefix_p, Axis::Parent), ["c1"]);
    assert_eq!(axis(&prefix_p, Axis::Ancestor), ["c1", "r", "/"]);
    assert_eq!(
        axis(&prefix_p, Axis::FollowingSibling),
        Vec::<String>::new()
    );
    assert_eq!(
        axis(&prefix_p, Axis::Following),
        [
            "text(x)",
            "comment(mid)",
            "c2",
            "g1",
            "g2",
            "pi(pi)",
            "text(t2)",
            "comment(after)",
        ]
    );
    assert_eq!(
        axis(&prefix_p, Axis::Preceding),
        ["text(t1)", "comment(before)"]
    );
    Ok(())
}

#[test]
fn every_axis_from_the_document_node() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    let root = tree.root();

    assert_eq!(axis(&root, Axis::SelfAxis), ["/"]);
    assert_eq!(
        axis(&root, Axis::Child),
        ["comment(before)", "r", "comment(after)"]
    );
    assert_eq!(axis(&root, Axis::Parent), Vec::<String>::new());
    assert_eq!(axis(&root, Axis::Ancestor), Vec::<String>::new());
    assert_eq!(axis(&root, Axis::AncestorOrSelf), ["/"]);
    assert_eq!(axis(&root, Axis::Following), Vec::<String>::new());
    assert_eq!(axis(&root, Axis::Preceding), Vec::<String>::new());
    assert_eq!(axis(&root, Axis::Attribute), Vec::<String>::new());
    assert_eq!(axis(&root, Axis::Namespace), Vec::<String>::new());
    assert_eq!(
        axis(&root, Axis::Descendant).len(),
        12,
        "every node but the document itself and the attribute/namespace nodes"
    );
    Ok(())
}

#[test]
fn forward_axes_ascend_and_reverse_axes_descend_from_every_node() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    for node in tree.nodes_in_document_order() {
        for &axis in Axis::ALL {
            let result = axis_nodes(&node, axis);
            let wanted = if axis.is_forward() {
                Ordering::Less
            } else {
                Ordering::Greater
            };
            for pair in result.windows(2) {
                assert_eq!(
                    compare_document_order(&pair[0], &pair[1]),
                    wanted,
                    "{axis} from {:?} is not in its own order",
                    node
                );
            }
            let mut seen = result.clone();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), result.len(), "{axis} yielded a duplicate node");
        }
    }
    Ok(())
}

/// The specification's own definition of the `following` axis, computed by brute force over the
/// whole tree, so that the incremental sibling walk in the crate has something independent to
/// agree with.
fn following_by_definition(tree: &Tree, node: &NodeRef) -> Vec<NodeRef> {
    tree.nodes_in_document_order()
        .into_iter()
        .filter(|candidate| {
            !matches!(
                candidate.node_kind(),
                NodeKind::Attribute | NodeKind::Namespace
            ) && compare_document_order(candidate, node) == Ordering::Greater
                && !is_ancestor_of(node, candidate)
        })
        .collect()
}

/// The same for the `preceding` axis, which is reverse document order.
fn preceding_by_definition(tree: &Tree, node: &NodeRef) -> Vec<NodeRef> {
    let mut result: Vec<NodeRef> = tree
        .nodes_in_document_order()
        .into_iter()
        .filter(|candidate| {
            !matches!(
                candidate.node_kind(),
                NodeKind::Attribute | NodeKind::Namespace
            ) && compare_document_order(candidate, node) == Ordering::Less
                && !is_ancestor_of(candidate, node)
        })
        .collect();
    result.reverse();
    result
}

#[test]
fn following_and_preceding_match_their_definitions_everywhere() -> Result<(), XdmError> {
    let tree = sample_tree()?;
    for node in tree.nodes_in_document_order() {
        assert_eq!(
            axis_nodes(&node, Axis::Following),
            following_by_definition(&tree, &node),
            "following from {node:?}"
        );
        assert_eq!(
            axis_nodes(&node, Axis::Preceding),
            preceding_by_definition(&tree, &node),
            "preceding from {node:?}"
        );
    }
    Ok(())
}

#[test]
fn the_five_axis_partitions_cover_the_tree_exactly_once() -> Result<(), XdmError> {
    // XPath guarantees that ancestor, descendant, preceding, following and self partition all
    // the nodes of a tree that are neither attributes nor namespace nodes.
    let tree = sample_tree()?;
    let total = tree
        .nodes_in_document_order()
        .into_iter()
        .filter(|node| !matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace))
        .count();

    for node in tree.nodes_in_document_order() {
        if matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace) {
            continue;
        }
        let mut partition: Vec<NodeRef> = Vec::new();
        for &axis in &[
            Axis::SelfAxis,
            Axis::Ancestor,
            Axis::Descendant,
            Axis::Following,
            Axis::Preceding,
        ] {
            partition.extend(axis_nodes(&node, axis));
        }
        let before = partition.len();
        partition.sort();
        partition.dedup();
        assert_eq!(
            before,
            partition.len(),
            "the partitions overlap at {node:?}"
        );
        assert_eq!(
            partition.len(),
            total,
            "the partitions do not cover the tree at {node:?}"
        );
    }
    Ok(())
}

#[test]
fn axes_work_over_a_builder_made_tree_too() -> Result<(), XdmError> {
    use oxixml_infoset::{ExpandedName, TreeBuilder};

    let mut builder = TreeBuilder::new();
    builder.start_element(ExpandedName::new(None::<&str>, "a"))?;
    builder.start_element(ExpandedName::new(None::<&str>, "b"))?;
    builder.end_element()?;
    builder.start_element(ExpandedName::new(None::<&str>, "c"))?;
    builder.end_element()?;
    builder.end_element()?;
    let tree = builder.build()?;

    let a = tree.document_element().expect("a document element");
    assert_eq!(axis(&a, Axis::Child), ["b", "c"]);
    let b = find(&tree, "b");
    assert_eq!(axis(&b, Axis::FollowingSibling), ["c"]);
    assert_eq!(axis(&b, Axis::Following), ["c"]);
    assert_eq!(axis(&b, Axis::Preceding), Vec::<String>::new());
    Ok(())
}
