//! Document order: comparison, sorting, de-duplication, and the ancestor/descendant relation.

use crate::node::{NodeRef, XdmNode};
use std::cmp::Ordering;
use std::collections::HashSet;

/// Compares two nodes in [document order](https://www.w3.org/TR/xpath-datamodel-31/#document-order).
///
/// Nodes in the same tree are ordered by the traversal the specification prescribes: an element
/// precedes its namespace nodes, which precede its attribute nodes, which precede its children.
/// Nodes in *different* trees are ordered by their trees' identifiers, which is stable within a
/// process — exactly the freedom
/// [XDM 3.1 §2.4](https://www.w3.org/TR/xpath-datamodel-31/#document-order) grants.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use oxixml_infoset::{compare_document_order, Tree, XdmNode};
///
/// let tree = Tree::parse_str(r#"<a x="1"><b/></a>"#)?;
/// let a = tree.document_element().expect("a document element");
/// let attribute = a.attributes().first().cloned().expect("one attribute");
/// let b = a.children().first().cloned().expect("one child");
///
/// assert_eq!(compare_document_order(&a, &attribute), Ordering::Less);
/// assert_eq!(compare_document_order(&attribute, &b), Ordering::Less);
/// assert_eq!(compare_document_order(&a, &a), Ordering::Equal);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn compare_document_order(left: &NodeRef, right: &NodeRef) -> Ordering {
    left.document_order_key().cmp(&right.document_order_key())
}

/// Sorts `nodes` into document order and removes duplicates, which is the normalization every
/// path expression and set operator applies to its result.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{sort_document_order, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b/><c/></a>")?;
/// let a = tree.document_element().expect("a document element");
/// let children = a.children();
///
/// let mut jumbled = vec![children[1].clone(), children[0].clone(), children[1].clone()];
/// sort_document_order(&mut jumbled);
/// assert_eq!(jumbled, children);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
pub fn sort_document_order(nodes: &mut Vec<NodeRef>) {
    nodes.sort_by(compare_document_order);
    nodes.dedup();
}

/// Removes duplicate nodes, keeping the first occurrence of each, **without** reordering.
///
/// Use this when a reverse-order axis result must keep its order; use
/// [`sort_document_order`] when the result must end up in document order.
///
/// Runs in time linear in the length of `nodes`: node identity is a hash key
/// ([`NodeRef`] is `Hash + Eq`), so a large node-set — the reverse-axis results this exists for
/// are routinely document-sized — never degrades into a quadratic scan.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{deduplicate_nodes, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b/><c/></a>")?;
/// let children = tree.document_element().expect("a document element").children();
///
/// let mut reversed = vec![children[1].clone(), children[0].clone(), children[1].clone()];
/// deduplicate_nodes(&mut reversed);
/// assert_eq!(reversed, vec![children[1].clone(), children[0].clone()]);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
pub fn deduplicate_nodes(nodes: &mut Vec<NodeRef>) {
    let mut seen: HashSet<NodeRef> = HashSet::with_capacity(nodes.len());
    nodes.retain(|node| seen.insert(node.clone()));
}

/// Whether `ancestor` is on `node`'s `ancestor` axis.
///
/// An attribute or namespace node's owner element *is* its ancestor, even though the node is not
/// its owner's child.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{is_ancestor_of, Tree, XdmNode};
///
/// let tree = Tree::parse_str(r#"<a x="1"><b/></a>"#)?;
/// let a = tree.document_element().expect("a document element");
/// let attribute = a.attributes().first().cloned().expect("one attribute");
/// let b = a.children().first().cloned().expect("one child");
///
/// assert!(is_ancestor_of(&a, &b));
/// assert!(is_ancestor_of(&a, &attribute));
/// assert!(is_ancestor_of(&tree.root(), &b));
/// assert!(!is_ancestor_of(&a, &a), "the relation is strict");
/// assert!(!is_ancestor_of(&b, &attribute));
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn is_ancestor_of(ancestor: &NodeRef, node: &NodeRef) -> bool {
    let mut current = node.parent();
    while let Some(candidate) = current {
        if &candidate == ancestor {
            return true;
        }
        current = candidate.parent();
    }
    false
}

/// Whether `descendant` is on `node`'s `descendant` axis — the inverse of [`is_ancestor_of`].
#[must_use]
pub fn is_descendant_of(descendant: &NodeRef, node: &NodeRef) -> bool {
    is_ancestor_of(node, descendant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Tree;

    /// Enough nodes that a de-duplication which compared every candidate against every node it
    /// had already kept would take minutes rather than milliseconds.
    const MANY: usize = 20_000;

    fn wide_tree() -> Tree {
        let mut xml = String::from("<root>");
        for _ in 0..MANY {
            xml.push_str("<c/>");
        }
        xml.push_str("</root>");
        Tree::parse_str(&xml).expect("a well-formed document")
    }

    #[test]
    fn duplicates_are_dropped_without_reordering() {
        let tree = Tree::parse_str("<a><b/><c/><d/></a>").expect("a well-formed document");
        let children = tree
            .document_element()
            .expect("a document element")
            .children();

        let mut nodes = vec![
            children[2].clone(),
            children[0].clone(),
            children[2].clone(),
            children[1].clone(),
            children[0].clone(),
        ];
        deduplicate_nodes(&mut nodes);
        assert_eq!(
            nodes,
            vec![
                children[2].clone(),
                children[0].clone(),
                children[1].clone()
            ]
        );
    }

    #[test]
    fn a_large_node_set_is_deduplicated_in_linear_time() {
        let tree = wide_tree();
        let children = tree
            .document_element()
            .expect("a document element")
            .children();
        assert_eq!(children.len(), MANY);

        // Every node distinct: the worst case for a scan of what has been kept so far.
        let mut nodes = children.clone();
        deduplicate_nodes(&mut nodes);
        assert_eq!(nodes, children);

        // Every node twice, in a reverse-axis order that must be preserved.
        let mut nodes: Vec<NodeRef> = children.iter().rev().cloned().collect();
        nodes.extend(children.iter().rev().cloned());
        deduplicate_nodes(&mut nodes);
        assert_eq!(nodes.len(), MANY);
        assert_eq!(nodes.first(), children.last());
    }
}
