//! The thirteen XPath axes, computed over any [`XdmNode`].

use crate::node::{NodeKind, NodeRef, XdmNode};
use std::fmt;

/// One of the thirteen [XPath axes](https://www.w3.org/TR/xpath-31/#axes).
///
/// # Direction
///
/// Eight axes are *forward* — their result is in document order — and five are *reverse*, with
/// their result in reverse document order. [`Axis::is_forward`] reports which, and
/// [`axis_nodes`] always returns the result already in the axis's own order, so a positional
/// predicate can index into it directly.
///
/// # Principal node kind
///
/// Each axis has a principal node kind, which is what an unprefixed name test on that axis
/// matches: `attribute` for the `attribute` axis, `namespace` for the `namespace` axis, and
/// `element` for the other eleven.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{Axis, NodeKind};
///
/// assert!(Axis::Descendant.is_forward());
/// assert!(!Axis::Ancestor.is_forward());
/// assert_eq!(Axis::Attribute.principal_node_kind(), NodeKind::Attribute);
/// assert_eq!(Axis::Child.principal_node_kind(), NodeKind::Element);
///
/// assert_eq!(Axis::from_name("following-sibling"), Some(Axis::FollowingSibling));
/// assert_eq!(Axis::SelfAxis.name(), "self");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    /// `self` — the context node itself. Spelled `SelfAxis` because `Self` is a Rust keyword.
    SelfAxis,
    /// `child` — the node's children, excluding attribute and namespace nodes.
    Child,
    /// `descendant` — the children, their children, and so on.
    Descendant,
    /// `descendant-or-self`.
    DescendantOrSelf,
    /// `parent` — at most one node.
    Parent,
    /// `ancestor` — the parent chain, nearest first.
    Ancestor,
    /// `ancestor-or-self`.
    AncestorOrSelf,
    /// `following-sibling` — later children of the same parent.
    FollowingSibling,
    /// `preceding-sibling` — earlier children of the same parent, nearest first.
    PrecedingSibling,
    /// `following` — everything after the node in document order that is not one of its
    /// descendants, excluding attribute and namespace nodes.
    Following,
    /// `preceding` — everything before the node in document order that is not one of its
    /// ancestors, excluding attribute and namespace nodes.
    Preceding,
    /// `attribute` — an element's attributes.
    Attribute,
    /// `namespace` — an element's in-scope namespace bindings.
    Namespace,
}

impl Axis {
    /// Every axis, in the order the XPath specification lists them.
    pub const ALL: &'static [Self] = &[
        Self::SelfAxis,
        Self::Child,
        Self::Descendant,
        Self::DescendantOrSelf,
        Self::Parent,
        Self::Ancestor,
        Self::AncestorOrSelf,
        Self::FollowingSibling,
        Self::PrecedingSibling,
        Self::Following,
        Self::Preceding,
        Self::Attribute,
        Self::Namespace,
    ];

    /// The axis's name as it is spelled in an XPath step.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SelfAxis => "self",
            Self::Child => "child",
            Self::Descendant => "descendant",
            Self::DescendantOrSelf => "descendant-or-self",
            Self::Parent => "parent",
            Self::Ancestor => "ancestor",
            Self::AncestorOrSelf => "ancestor-or-self",
            Self::FollowingSibling => "following-sibling",
            Self::PrecedingSibling => "preceding-sibling",
            Self::Following => "following",
            Self::Preceding => "preceding",
            Self::Attribute => "attribute",
            Self::Namespace => "namespace",
        }
    }

    /// Looks an axis up by its XPath name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|axis| axis.name() == name)
    }

    /// Whether the axis is a forward axis, whose result is in document order.
    #[must_use]
    pub const fn is_forward(self) -> bool {
        !matches!(
            self,
            Self::Parent
                | Self::Ancestor
                | Self::AncestorOrSelf
                | Self::PrecedingSibling
                | Self::Preceding
        )
    }

    /// The axis's principal node kind — what an unprefixed name test on it matches.
    #[must_use]
    pub const fn principal_node_kind(self) -> NodeKind {
        match self {
            Self::Attribute => NodeKind::Attribute,
            Self::Namespace => NodeKind::Namespace,
            _ => NodeKind::Element,
        }
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Enumerates `axis` from `node`, in the axis's own order.
///
/// The result is in document order for a forward axis and in reverse document order for a
/// reverse axis, and never contains the same node twice.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{axis_nodes, Axis, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b><c/></b><d/></a>")?;
/// let b = tree
///     .document_element()
///     .expect("a document element")
///     .children()
///     .first()
///     .cloned()
///     .expect("the <b> element");
///
/// let names = |axis| {
///     axis_nodes(&b, axis)
///         .iter()
///         .map(|node| node.node_name().map_or_else(|| "/".to_owned(), |n| n.to_string()))
///         .collect::<Vec<_>>()
/// };
///
/// assert_eq!(names(Axis::Child), ["c"]);
/// assert_eq!(names(Axis::Ancestor), ["a", "/"], "nearest ancestor first");
/// assert_eq!(names(Axis::Following), ["d"]);
/// assert_eq!(names(Axis::DescendantOrSelf), ["b", "c"]);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn axis_nodes(node: &NodeRef, axis: Axis) -> Vec<NodeRef> {
    match axis {
        Axis::SelfAxis => vec![node.clone()],
        Axis::Child => node.children(),
        Axis::Descendant => descendants(node),
        Axis::DescendantOrSelf => descendant_or_self(node),
        Axis::Parent => node.parent().into_iter().collect(),
        Axis::Ancestor => ancestors(node),
        Axis::AncestorOrSelf => ancestor_or_self(node),
        Axis::FollowingSibling => following_siblings(node),
        Axis::PrecedingSibling => preceding_siblings(node),
        Axis::Following => following(node),
        Axis::Preceding => preceding(node),
        Axis::Attribute => node.attributes(),
        Axis::Namespace => node.namespaces(),
    }
}

/// The `descendant` axis: every node reachable through `child`, in document order.
#[must_use]
pub fn descendants(node: &NodeRef) -> Vec<NodeRef> {
    let mut out = Vec::new();
    push_descendants(node, &mut out);
    out
}

/// Appends every descendant of `node` to `out` in document order.
///
/// Deliberately iterative: recursing once per tree level would overflow the
/// stack — an uncatchable process abort — on a document that is merely deeply
/// nested, without any exotic syntax being involved.
fn push_descendants(node: &NodeRef, out: &mut Vec<NodeRef>) {
    // Each entry is the not-yet-visited tail of one level's child list, kept in
    // reverse so that `pop` yields the next node in document order.
    let mut stack: Vec<Vec<NodeRef>> = Vec::new();
    let mut pending: Vec<NodeRef> = node.children();
    pending.reverse();
    stack.push(pending);

    while let Some(level) = stack.last_mut() {
        let Some(child) = level.pop() else {
            stack.pop();
            continue;
        };
        out.push(child.clone());
        let mut grandchildren = child.children();
        if !grandchildren.is_empty() {
            grandchildren.reverse();
            stack.push(grandchildren);
        }
    }
}

/// The `descendant-or-self` axis, in document order.
#[must_use]
pub fn descendant_or_self(node: &NodeRef) -> Vec<NodeRef> {
    let mut out = vec![node.clone()];
    push_descendants(node, &mut out);
    out
}

/// The `ancestor` axis: the parent chain, **nearest ancestor first** (reverse document order).
#[must_use]
pub fn ancestors(node: &NodeRef) -> Vec<NodeRef> {
    let mut out = Vec::new();
    let mut current = node.parent();
    while let Some(ancestor) = current {
        current = ancestor.parent();
        out.push(ancestor);
    }
    out
}

/// The `ancestor-or-self` axis, in reverse document order.
#[must_use]
pub fn ancestor_or_self(node: &NodeRef) -> Vec<NodeRef> {
    let mut out = vec![node.clone()];
    out.extend(ancestors(node));
    out
}

/// The `following-sibling` axis, in document order.
///
/// Attribute and namespace nodes have no siblings: the axis is empty for them, even when their
/// owner element has several attributes.
#[must_use]
pub fn following_siblings(node: &NodeRef) -> Vec<NodeRef> {
    siblings(node, true)
}

/// The `preceding-sibling` axis, **nearest sibling first** (reverse document order).
#[must_use]
pub fn preceding_siblings(node: &NodeRef) -> Vec<NodeRef> {
    siblings(node, false)
}

fn siblings(node: &NodeRef, forward: bool) -> Vec<NodeRef> {
    if matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace) {
        return Vec::new();
    }
    let Some(parent) = node.parent() else {
        return Vec::new();
    };
    let children = parent.children();
    let Some(position) = children.iter().position(|child| child == node) else {
        return Vec::new();
    };
    if forward {
        children.into_iter().skip(position + 1).collect()
    } else {
        children.into_iter().take(position).rev().collect()
    }
}

/// The `following` axis: every node of the tree that comes after `node` in document order and is
/// not one of its descendants, excluding attribute and namespace nodes.
///
/// For an attribute or namespace node this starts with the content of its owner element, since
/// the owner's children follow the attribute in document order without being its descendants.
#[must_use]
pub fn following(node: &NodeRef) -> Vec<NodeRef> {
    let mut out = Vec::new();
    let mut current = node.clone();
    if matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace) {
        let Some(owner) = node.parent() else {
            return out;
        };
        for child in owner.children() {
            out.extend(descendant_or_self(&child));
        }
        current = owner;
    }
    loop {
        for sibling in following_siblings(&current) {
            out.extend(descendant_or_self(&sibling));
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
    out
}

/// The `preceding` axis: every node of the tree that comes before `node` in document order and
/// is not one of its ancestors, excluding attribute and namespace nodes.
///
/// The result is in reverse document order, so the first element is the node immediately
/// preceding `node`.
#[must_use]
pub fn preceding(node: &NodeRef) -> Vec<NodeRef> {
    let mut out = Vec::new();
    let mut current = if matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace) {
        match node.parent() {
            Some(owner) => owner,
            None => return out,
        }
    } else {
        node.clone()
    };
    loop {
        for sibling in preceding_siblings(&current) {
            let mut subtree = descendant_or_self(&sibling);
            subtree.reverse();
            out.extend(subtree);
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn names_round_trip_and_are_unique() {
        let mut seen = HashSet::new();
        for &axis in Axis::ALL {
            assert!(seen.insert(axis.name()), "duplicate axis name {axis}");
            assert_eq!(Axis::from_name(axis.name()), Some(axis));
            assert_eq!(axis.to_string(), axis.name());
        }
        assert_eq!(seen.len(), 13);
        assert_eq!(Axis::from_name("nope"), None);
    }

    #[test]
    fn exactly_five_axes_are_reverse_axes() {
        let reverse: Vec<_> = Axis::ALL
            .iter()
            .copied()
            .filter(|axis| !axis.is_forward())
            .collect();
        assert_eq!(
            reverse,
            [
                Axis::Parent,
                Axis::Ancestor,
                Axis::AncestorOrSelf,
                Axis::PrecedingSibling,
                Axis::Preceding
            ]
        );
    }

    #[test]
    fn principal_node_kinds() {
        assert_eq!(Axis::Attribute.principal_node_kind(), NodeKind::Attribute);
        assert_eq!(Axis::Namespace.principal_node_kind(), NodeKind::Namespace);
        for &axis in Axis::ALL {
            if !matches!(axis, Axis::Attribute | Axis::Namespace) {
                assert_eq!(axis.principal_node_kind(), NodeKind::Element, "{axis}");
            }
        }
    }
}
