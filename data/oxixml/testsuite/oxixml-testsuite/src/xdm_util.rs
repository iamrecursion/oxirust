//! `NodeRef` navigation helpers shared by the [`crate::qt3`] and [`crate::xslt30`] catalog
//! drivers.
//!
//! Both catalogues are read through [`oxixml_infoset::Tree`] rather than `oxixml_dom` — the
//! engines under test consume XDM trees for their own inputs, so reading the catalogue itself
//! through the same tree representation means the harness needs only one XML tree type, not two.

use oxixml_infoset::{ExpandedName, NodeKind, NodeRef, XdmNode};

/// The element children of `node`, in document order.
#[must_use]
pub fn element_children(node: &NodeRef) -> Vec<NodeRef> {
    node.children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .collect()
}

/// The element children of `node` whose local name is `local`, ignoring namespace.
pub fn children_named<'a>(node: &'a NodeRef, local: &'a str) -> impl Iterator<Item = NodeRef> + 'a {
    element_children(node)
        .into_iter()
        .filter(move |child| local_name(child) == local)
}

/// The first element child of `node` named `local`, ignoring namespace.
#[must_use]
pub fn find_child(node: &NodeRef, local: &str) -> Option<NodeRef> {
    children_named(node, local).next()
}

/// `node`'s local name (the empty string for a node with none, e.g. text).
#[must_use]
pub fn local_name(node: &NodeRef) -> String {
    node.node_name()
        .map(|name| name.local_name().to_owned())
        .unwrap_or_default()
}

/// The value of the unprefixed attribute `local` on `node`.
#[must_use]
pub fn attribute(node: &NodeRef, local: &str) -> Option<String> {
    node.attributes().into_iter().find_map(|attribute| {
        (attribute.node_name()? == ExpandedName::new(None::<&str>, local))
            .then(|| attribute.string_value())
    })
}

/// The concatenated *direct* text-node children of `node` — not its whole string value, so a
/// nested element's text never leaks into a catalogue assertion's expected value.
#[must_use]
pub fn text_of(node: &NodeRef) -> String {
    node.children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Text)
        .map(|child| child.string_value())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn navigation_helpers_agree_with_the_data_model() {
        let tree = Tree::parse_str(r#"<a n="1"><b/>text<c/></a>"#).expect("well-formed fixture");
        let element = tree.document_element().expect("a document element");
        assert_eq!(local_name(&element), "a");
        assert_eq!(attribute(&element, "n").as_deref(), Some("1"));
        assert_eq!(element_children(&element).len(), 2);
        assert_eq!(text_of(&element), "text");
        assert!(find_child(&element, "b").is_some());
        assert!(find_child(&element, "missing").is_none());
        assert_eq!(children_named(&element, "b").count(), 1);
    }
}
