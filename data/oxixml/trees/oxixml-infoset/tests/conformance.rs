//! Data-model invariants checked over the W3C XML Conformance Test Suite.
//!
//! The suite is discovered through the `OXIXML_XML_SUITES_DIR` environment variable, which must
//! point at a directory containing an `xmlts*` subdirectory. When the variable is unset — the
//! normal case on a machine without the suites — every test here silently succeeds without
//! asserting anything.
//!
//! What is checked is not "does this document parse the way the suite says it should" (that is
//! [`oxixml_xml`]'s job, and this crate deliberately does no DTD processing, so many valid
//! documents are legitimately rejected here). It is the stronger, structural claim that *every*
//! tree this crate does build satisfies the XDM's own axioms.

use oxixml_infoset::{Axis, NodeKind, NodeRef, Tree, XdmNode, axis_nodes, compare_document_order};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

/// The root of the `xmlts*` suite, or `None` when the suites are not configured.
fn suite_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let entries = fs::read_dir(&root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("xmlts"))
        {
            return Some(path);
        }
    }
    None
}

/// Collects up to `limit` `*.xml` files under `directory`, in a deterministic order.
fn xml_files(directory: &Path, limit: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        let mut children: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
        children.sort();
        for child in children {
            if child.is_dir() {
                stack.push(child);
            } else if child.extension().and_then(|ext| ext.to_str()) == Some("xml") {
                found.push(child);
                if found.len() >= limit {
                    found.sort();
                    return found;
                }
            }
        }
    }
    found.sort();
    found
}

/// Every structural invariant the data model promises, checked over one tree.
fn assert_invariants(tree: &Tree, source: &Path) {
    let nodes = tree.nodes_in_document_order();
    assert!(!nodes.is_empty(), "{}: an empty tree", source.display());
    assert_eq!(
        nodes.len(),
        tree.len(),
        "{}: node count disagrees",
        source.display()
    );

    for pair in nodes.windows(2) {
        assert_eq!(
            compare_document_order(&pair[0], &pair[1]),
            Ordering::Less,
            "{}: document order is not strictly increasing",
            source.display()
        );
    }

    assert_eq!(nodes[0].node_kind(), NodeKind::Document);
    assert!(nodes[0].parent().is_none());

    let mut document_elements = 0_usize;
    for node in &nodes {
        assert_eq!(node.root(), tree.root(), "{}: root", source.display());

        match node.node_kind() {
            NodeKind::Document => assert!(node.parent().is_none()),
            NodeKind::Attribute | NodeKind::Namespace => {
                let parent = node.parent().expect("an owner element");
                assert_eq!(parent.node_kind(), NodeKind::Element);
                let owned = if node.node_kind() == NodeKind::Attribute {
                    parent.attributes()
                } else {
                    parent.namespaces()
                };
                assert!(owned.contains(node), "{}: ownership", source.display());
                assert!(
                    !parent.children().contains(node),
                    "{}: an attribute or namespace node is not a child",
                    source.display()
                );
                assert!(axis_nodes(node, Axis::FollowingSibling).is_empty());
                assert!(axis_nodes(node, Axis::PrecedingSibling).is_empty());
            }
            kind => {
                let parent = node.parent().expect("a parent");
                assert!(
                    parent.children().contains(node),
                    "{}: child linkage",
                    source.display()
                );
                if kind == NodeKind::Element && parent.node_kind() == NodeKind::Document {
                    document_elements += 1;
                }
            }
        }

        // No two adjacent text nodes, ever.
        let children = node.children();
        for pair in children.windows(2) {
            assert!(
                pair[0].node_kind() != NodeKind::Text || pair[1].node_kind() != NodeKind::Text,
                "{}: adjacent text nodes",
                source.display()
            );
        }

        // Every element's string value is the concatenation of its descendant text nodes.
        if matches!(node.node_kind(), NodeKind::Element | NodeKind::Document) {
            let concatenated: String = axis_nodes(node, Axis::Descendant)
                .iter()
                .filter(|descendant| descendant.node_kind() == NodeKind::Text)
                .map(XdmNode::string_value)
                .collect();
            assert_eq!(
                node.string_value(),
                concatenated,
                "{}: string value",
                source.display()
            );
        }
    }
    assert_eq!(
        document_elements,
        1,
        "{}: exactly one document element",
        source.display()
    );
}

/// Checks the ordering and de-duplication guarantee of every axis from a handful of nodes.
fn assert_axis_orders(tree: &Tree, source: &Path) {
    let nodes = tree.nodes_in_document_order();
    // Sample rather than sweep: the invariant is per-node, and the suite has thousands of files.
    let step = (nodes.len() / 8).max(1);
    for node in nodes.iter().step_by(step) {
        for &axis in Axis::ALL {
            let result: Vec<NodeRef> = axis_nodes(node, axis);
            let wanted = if axis.is_forward() {
                Ordering::Less
            } else {
                Ordering::Greater
            };
            for pair in result.windows(2) {
                assert_eq!(
                    compare_document_order(&pair[0], &pair[1]),
                    wanted,
                    "{}: {axis} is out of order",
                    source.display()
                );
            }
            let mut unique = result.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(
                unique.len(),
                result.len(),
                "{}: {axis} yielded a duplicate",
                source.display()
            );
        }
    }
}

#[test]
fn every_tree_built_from_the_w3c_suite_satisfies_the_data_model_axioms() {
    let Some(root) = suite_root() else {
        // The suites are not configured on this machine; nothing to check.
        return;
    };

    let files = xml_files(&root, 4000);
    assert!(
        !files.is_empty(),
        "OXIXML_XML_SUITES_DIR points at {} but no .xml files were found",
        root.display()
    );

    let mut parsed = 0_usize;
    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            // Non-UTF-8 documents are outside this crate's remit: the tree builder takes `&str`.
            continue;
        };
        // Documents whose content depends on a DTD (entity declarations, defaulted attributes)
        // are legitimately rejected here, so a parse failure is never itself a test failure.
        if let Ok(tree) = Tree::parse_str(&text) {
            assert_invariants(&tree, file);
            assert_axis_orders(&tree, file);
            parsed += 1;
        }
    }

    assert!(
        parsed > 1000,
        "only {parsed} of {} suite documents produced a tree, which suggests the parser, \
         not the corpus, is at fault",
        files.len()
    );
}

#[test]
fn round_tripping_document_order_is_stable_across_trees() {
    let Some(root) = suite_root() else {
        return;
    };
    let files = xml_files(&root, 200);
    let mut compared = 0_usize;
    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        let (Ok(first), Ok(second)) = (Tree::parse_str(&text), Tree::parse_str(&text)) else {
            continue;
        };
        // Two trees built from the same text have identical shapes but are never the same nodes,
        // and their document-order keys never interleave.
        assert_eq!(first.len(), second.len(), "{}", file.display());
        let left = first.root();
        let right = second.root();
        assert_ne!(left, right, "distinct trees have distinct identities");
        let ordering = compare_document_order(&left, &right);
        assert_ne!(ordering, Ordering::Equal);
        for node in second.nodes_in_document_order() {
            assert_eq!(
                compare_document_order(&left, &node),
                ordering,
                "{}: the whole of one tree precedes or follows the whole of the other",
                file.display()
            );
        }
        compared += 1;
    }
    assert!(compared > 10, "only {compared} documents were compared");
}
