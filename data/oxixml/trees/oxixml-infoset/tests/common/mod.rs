//! Shared fixtures for the integration tests: one known tree and a readable node labeller.

// Each integration test binary compiles this module separately and uses only part of it.
#![allow(dead_code)]

use oxixml_infoset::{NodeKind, NodeRef, Tree, XdmError, XdmNode};

/// The document every axis and accessor test navigates.
///
/// It is written without insignificant white space so that the node sequence is completely
/// determined, and it exercises every node kind: comments beside and inside the document
/// element, two attributes (one namespaced), a namespace declaration, text, nested elements and
/// a processing instruction.
pub const SAMPLE: &str = concat!(
    "<!--before-->",
    r#"<r xmlns:p="urn:p" a1="v1" p:a2="v2">"#,
    "t1",
    "<c1>x</c1>",
    "<!--mid-->",
    "<c2><g1/><g2/></c2>",
    "<?pi data?>",
    "t2",
    "</r>",
    "<!--after-->",
);

/// Parses [`SAMPLE`].
pub fn sample_tree() -> Result<Tree, XdmError> {
    Tree::parse_str(SAMPLE)
}

/// A short, unambiguous label for a node, used to spell out expected axis results.
pub fn label(node: &NodeRef) -> String {
    let name = node
        .node_name()
        .map_or_else(String::new, |name| name.to_string());
    match node.node_kind() {
        NodeKind::Document => "/".to_owned(),
        NodeKind::Element => name,
        NodeKind::Attribute => format!("@{name}"),
        NodeKind::Namespace => {
            if name.is_empty() {
                "ns:#default".to_owned()
            } else {
                format!("ns:{name}")
            }
        }
        NodeKind::Text => format!("text({})", node.string_value()),
        NodeKind::Comment => format!("comment({})", node.string_value()),
        NodeKind::ProcessingInstruction => format!("pi({name})"),
    }
}

/// Labels a whole node list.
pub fn labels(nodes: &[NodeRef]) -> Vec<String> {
    nodes.iter().map(label).collect()
}

/// Finds the single node with the given label, failing loudly if there is not exactly one.
pub fn find(tree: &Tree, wanted: &str) -> NodeRef {
    let matches: Vec<NodeRef> = tree
        .nodes_in_document_order()
        .into_iter()
        .filter(|node| label(node) == wanted)
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one node labelled `{wanted}`, found {}",
        matches.len()
    );
    matches.into_iter().next().unwrap_or_else(|| {
        unreachable!("the length was just asserted to be one");
    })
}
