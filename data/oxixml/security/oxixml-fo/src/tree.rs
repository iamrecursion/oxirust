//! The formatting object tree.
//!
//! An [`FoTree`] is an arena of [`FoNode`]s addressed by [`NodeId`]: there are
//! no reference cycles, no reference counting, and a node identifier stays
//! valid for the lifetime of the tree. Each formatting object carries both the
//! *specified* property text taken from the source document and the *refined*
//! values computed from it.

use core::fmt;

use crate::fo::FoElement;
use crate::property::{PropertyId, PropertyKey, Value};

/// A handle to a node of an [`FoTree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// The index of this node in the arena.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// An element that is not in the XSL-FO namespace.
///
/// These appear as the content of `fo:instream-foreign-object` (SVG, MathML, …)
/// and as extension children of `fo:declarations`. They are kept verbatim so a
/// renderer can hand them to the right processor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignElement {
    /// The namespace URI, or `None` for an element in no namespace.
    pub namespace: Option<String>,
    /// The local part of the element name.
    pub local_name: String,
    /// The attributes, as `(namespace, local name, value)` triples.
    pub attributes: Vec<(Option<String>, String, String)>,
}

/// What a node of an [`FoTree`] is.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// A formatting object.
    Fo(FoElement),
    /// Character data.
    Text(String),
    /// An element outside the XSL-FO namespace.
    Foreign(ForeignElement),
}

/// The refined values of the properties that apply to one formatting object.
///
/// Entries are sorted by [`PropertyId`], so lookups are a binary search.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedProperties {
    entries: Vec<(PropertyId, Value)>,
}

impl ResolvedProperties {
    /// Builds a property set from `(id, value)` pairs, which need not be
    /// sorted. Later pairs win over earlier ones.
    #[must_use]
    pub fn from_pairs(mut pairs: Vec<(PropertyId, Value)>) -> Self {
        pairs.sort_by_key(|(id, _)| *id);
        pairs.dedup_by_key(|(id, _)| *id);
        Self { entries: pairs }
    }

    /// The refined value of `property`, if it applies to this object.
    #[must_use]
    pub fn get(&self, property: PropertyId) -> Option<&Value> {
        self.entries
            .binary_search_by_key(&property, |(id, _)| *id)
            .ok()
            .map(|index| &self.entries[index].1)
    }

    /// The refined value of `property`, or [`Value::Absent`].
    #[must_use]
    pub fn value(&self, property: PropertyId) -> &Value {
        static ABSENT: Value = Value::Absent;
        self.get(property).unwrap_or(&ABSENT)
    }

    /// Inserts or replaces a value.
    pub fn set(&mut self, property: PropertyId, value: Value) {
        match self.entries.binary_search_by_key(&property, |(id, _)| *id) {
            Ok(index) => self.entries[index].1 = value,
            Err(index) => self.entries.insert(index, (property, value)),
        }
    }

    /// Iterates over every refined property, in identifier order.
    pub fn iter(&self) -> impl Iterator<Item = (PropertyId, &Value)> {
        self.entries.iter().map(|(id, value)| (*id, value))
    }

    /// The number of refined properties.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no property has been refined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<'a> IntoIterator for &'a ResolvedProperties {
    type Item = (PropertyId, &'a Value);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (PropertyId, Value)>,
        fn(&'a (PropertyId, Value)) -> (PropertyId, &'a Value),
    >;

    fn into_iter(self) -> Self::IntoIter {
        fn project(entry: &(PropertyId, Value)) -> (PropertyId, &Value) {
            (entry.0, &entry.1)
        }
        self.entries.iter().map(project)
    }
}

/// One node of an [`FoTree`].
#[derive(Debug, Clone)]
pub struct FoNode {
    pub(crate) kind: NodeKind,
    pub(crate) parent: Option<NodeId>,
    pub(crate) children: Vec<NodeId>,
    pub(crate) specified: Vec<(PropertyKey, String)>,
    pub(crate) foreign_attributes: Vec<(Option<String>, String, String)>,
    pub(crate) properties: ResolvedProperties,
}

impl FoNode {
    /// What this node is.
    #[must_use]
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    /// The formatting object, if this node is one.
    #[must_use]
    pub fn element(&self) -> Option<FoElement> {
        match self.kind {
            NodeKind::Fo(element) => Some(element),
            _ => None,
        }
    }

    /// The character data, if this node is a text node.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The foreign element, if this node is one.
    #[must_use]
    pub fn foreign(&self) -> Option<&ForeignElement> {
        match &self.kind {
            NodeKind::Foreign(element) => Some(element),
            _ => None,
        }
    }

    /// The node's parent, or `None` for the root.
    #[must_use]
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// The node's children, in document order.
    #[must_use]
    pub fn children(&self) -> &[NodeId] {
        &self.children
    }

    /// The property specifications written on this object in the source
    /// document, in document order and before shorthand expansion.
    #[must_use]
    pub fn specified(&self) -> &[(PropertyKey, String)] {
        &self.specified
    }

    /// Whether the stylesheet named `property` on this object, either directly
    /// or through one of its `.component` spellings.
    #[must_use]
    pub fn specifies(&self, property: PropertyId) -> bool {
        self.specified.iter().any(|(key, _)| key.id == property)
    }

    /// Attributes that are not in the XSL-FO property vocabulary, as
    /// `(namespace, local name, value)` triples.
    ///
    /// `fo:instream-foreign-object` in particular passes these through to the
    /// processor of the foreign content.
    #[must_use]
    pub fn foreign_attributes(&self) -> &[(Option<String>, String, String)] {
        &self.foreign_attributes
    }

    /// The refined property values of this object.
    #[must_use]
    pub fn properties(&self) -> &ResolvedProperties {
        &self.properties
    }
}

/// A refined formatting object tree.
#[derive(Debug, Clone)]
pub struct FoTree {
    pub(crate) nodes: Vec<FoNode>,
    pub(crate) root: NodeId,
    pub(crate) element_order: Vec<NodeId>,
}

impl FoTree {
    /// The `fo:root` node.
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Borrows a node.
    #[must_use]
    pub fn node(&self, id: NodeId) -> &FoNode {
        &self.nodes[id.index()]
    }

    /// The number of nodes in the tree, including text and foreign nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The formatting object at `id`, if the node is one.
    #[must_use]
    pub fn element(&self, id: NodeId) -> Option<FoElement> {
        self.node(id).element()
    }

    /// The children of `id`, in document order.
    #[must_use]
    pub fn children(&self, id: NodeId) -> &[NodeId] {
        self.node(id).children()
    }

    /// The parent of `id`.
    #[must_use]
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.node(id).parent
    }

    /// The refined properties of `id`.
    #[must_use]
    pub fn properties(&self, id: NodeId) -> &ResolvedProperties {
        &self.node(id).properties
    }

    /// Every formatting object node, in document order.
    ///
    /// Text and foreign nodes are skipped; this is the order refinement runs
    /// in, so a node's ancestors always precede it.
    #[must_use]
    pub fn elements(&self) -> &[NodeId] {
        &self.element_order
    }

    /// The ancestors of `id`, nearest first.
    pub fn ancestors(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        let mut current = self.node(id).parent;
        core::iter::from_fn(move || {
            let next = current?;
            current = self.node(next).parent;
            Some(next)
        })
    }

    /// The nearest ancestor that is one of `elements`.
    #[must_use]
    pub fn nearest_ancestor(&self, id: NodeId, elements: &[FoElement]) -> Option<NodeId> {
        self.ancestors(id)
            .find(|ancestor| matches!(self.element(*ancestor), Some(e) if elements.contains(&e)))
    }

    /// Every node below `id`, in document order, excluding `id` itself.
    pub fn descendants(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        let mut stack: Vec<NodeId> = self.children(id).iter().rev().copied().collect();
        core::iter::from_fn(move || {
            let next = stack.pop()?;
            stack.extend(self.children(next).iter().rev().copied());
            Some(next)
        })
    }

    /// The concatenated character data below `id`.
    #[must_use]
    pub fn text_content(&self, id: NodeId) -> String {
        let mut out = String::new();
        for descendant in self.descendants(id) {
            if let Some(text) = self.node(descendant).text() {
                out.push_str(text);
            }
        }
        out
    }

    /// Renders the tree as an indented outline, for diagnostics and tests.
    #[must_use]
    pub fn outline(&self) -> String {
        let mut out = String::new();
        self.write_outline(&mut out);
        out
    }

    /// Depth-first, but as an explicit stack rather than a recursive call: the
    /// nesting of a source document is chosen by whoever wrote it, not by this
    /// crate, and a recursive walk turns a few kilobytes of nested `<fo:block>`
    /// elements into a stack overflow — which aborts the process outright,
    /// rather than this diagnostics helper ever failing gracefully. Keeping the
    /// pending work on the heap makes the depth a matter of memory, not of
    /// stack.
    fn write_outline(&self, out: &mut String) {
        let mut stack: Vec<(NodeId, usize)> = vec![(self.root, 0)];
        while let Some((id, depth)) = stack.pop() {
            let node = self.node(id);
            for _ in 0..depth {
                out.push_str("  ");
            }
            match &node.kind {
                NodeKind::Fo(element) => out.push_str(&element.qualified_name()),
                NodeKind::Text(text) => {
                    out.push('"');
                    out.push_str(text.trim());
                    out.push('"');
                }
                NodeKind::Foreign(element) => {
                    out.push('{');
                    out.push_str(&element.local_name);
                    out.push('}');
                }
            }
            out.push('\n');
            // Children are pushed in reverse so the first one is popped — and
            // hence rendered — first, reproducing document order depth-first.
            stack.extend(node.children.iter().rev().map(|child| (*child, depth + 1)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `outline()` used to walk the tree with one recursive call per node,
    /// which turned a deeply nested source document into a stack overflow — an
    /// abort that a `Result` cannot express. Build a tree far deeper than a
    /// recursive walk could survive and confirm it still renders correctly.
    #[test]
    fn outline_survives_a_chain_deeper_than_a_recursive_walk_would() {
        // The rendered outline indents every line by its depth, so the output
        // size grows with the square of the chain length; a few thousand
        // levels is already far beyond anything a real stylesheet produces
        // and keeps the test itself fast.
        const DEPTH: usize = 4_000;
        let mut nodes = Vec::with_capacity(DEPTH);
        let mut parent = None;
        for index in 0..DEPTH {
            let id = NodeId::from_index(index);
            nodes.push(FoNode {
                kind: NodeKind::Fo(FoElement::Block),
                parent,
                children: Vec::new(),
                specified: Vec::new(),
                foreign_attributes: Vec::new(),
                properties: ResolvedProperties::default(),
            });
            if let Some(parent_id) = parent {
                nodes[parent_id.index()].children.push(id);
            }
            parent = Some(id);
        }
        let tree = FoTree {
            nodes,
            root: NodeId::from_index(0),
            element_order: Vec::new(),
        };

        let outline = tree.outline();
        assert_eq!(outline.lines().count(), DEPTH);
        assert_eq!(outline.lines().next(), Some("fo:block"));
        assert_eq!(
            outline.lines().last(),
            Some(format!("{}fo:block", "  ".repeat(DEPTH - 1)).as_str())
        );
    }
}
