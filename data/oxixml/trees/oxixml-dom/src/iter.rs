//! Lazy iterators over the tree.
//!
//! Every iterator borrows the [`Document`] it walks, so it always observes the
//! tree as it is at the moment each item is produced — the "live-ish" behaviour
//! of the DOM `NodeList`, without the memory-management contortions a truly live
//! list would need. Because the borrow is shared, the document cannot be mutated
//! while an iterator is alive; collect into a `Vec<NodeId>` first if a mutation
//! has to happen during the walk.

use crate::document::Document;
use crate::node::{ElementData, NodeId, NodeKind};

/// Iterator over the child nodes of a node, in document order.
///
/// Created by [`Document::children`].
#[derive(Debug, Clone)]
pub struct Children<'a> {
    doc: &'a Document,
    next: Option<NodeId>,
    next_back: Option<NodeId>,
    done: bool,
}

impl<'a> Children<'a> {
    pub(crate) fn new(doc: &'a Document, parent: NodeId) -> Self {
        let (first, last) = doc
            .get(parent)
            .map_or((None, None), |node| (node.first_child, node.last_child));
        Self {
            doc,
            next: first,
            next_back: last,
            done: first.is_none(),
        }
    }
}

impl Iterator for Children<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        if self.done {
            return None;
        }
        let current = self.next?;
        if Some(current) == self.next_back {
            self.done = true;
        } else {
            self.next = self.doc.get(current).and_then(|node| node.next_sibling);
        }
        Some(current)
    }
}

impl DoubleEndedIterator for Children<'_> {
    fn next_back(&mut self) -> Option<NodeId> {
        if self.done {
            return None;
        }
        let current = self.next_back?;
        if Some(current) == self.next {
            self.done = true;
        } else {
            self.next_back = self.doc.get(current).and_then(|node| node.prev_sibling);
        }
        Some(current)
    }
}

/// Iterator over the attribute nodes of an element, in the order they were
/// added.
///
/// Created by [`Document::attributes`].
#[derive(Debug, Clone)]
pub struct Attributes<'a> {
    inner: std::slice::Iter<'a, NodeId>,
}

/// The attribute list of a node that has none.
const NO_ATTRIBUTES: &[NodeId] = &[];

impl<'a> Attributes<'a> {
    pub(crate) fn new(doc: &'a Document, element: NodeId) -> Self {
        Self {
            inner: doc
                .element(element)
                .map_or(NO_ATTRIBUTES, ElementData::attribute_ids)
                .iter(),
        }
    }
}

impl Iterator for Attributes<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        self.inner.next().copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for Attributes<'_> {
    fn next_back(&mut self) -> Option<NodeId> {
        self.inner.next_back().copied()
    }
}

impl ExactSizeIterator for Attributes<'_> {}

/// Iterator over the ancestors of a node, from its parent up to the document
/// node.
///
/// Created by [`Document::ancestors`]. For an attribute node the walk starts at
/// its owner element.
#[derive(Debug, Clone)]
pub struct Ancestors<'a> {
    doc: &'a Document,
    next: Option<NodeId>,
}

impl<'a> Ancestors<'a> {
    pub(crate) const fn new(doc: &'a Document, start: Option<NodeId>) -> Self {
        Self { doc, next: start }
    }
}

impl Iterator for Ancestors<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let current = self.next?;
        self.next = self.doc.parent(current);
        Some(current)
    }
}

/// Iterator over the following or preceding siblings of a node.
///
/// Created by [`Document::following_siblings`] and
/// [`Document::preceding_siblings`].
#[derive(Debug, Clone)]
pub struct Siblings<'a> {
    doc: &'a Document,
    next: Option<NodeId>,
    forward: bool,
}

impl<'a> Siblings<'a> {
    pub(crate) const fn new(doc: &'a Document, start: Option<NodeId>, forward: bool) -> Self {
        Self {
            doc,
            next: start,
            forward,
        }
    }
}

impl Iterator for Siblings<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let current = self.next?;
        self.next = if self.forward {
            self.doc.next_sibling(current)
        } else {
            self.doc.previous_sibling(current)
        };
        Some(current)
    }
}

/// Iterator over a subtree in document order.
///
/// Created by [`Document::descendants`] (which excludes the subtree root) and by
/// [`Document::traverse`] (which includes it). Attribute nodes are not part of
/// the child tree and are therefore not visited.
#[derive(Debug, Clone)]
pub struct Descendants<'a> {
    doc: &'a Document,
    root: NodeId,
    next: Option<NodeId>,
}

impl<'a> Descendants<'a> {
    pub(crate) const fn new(doc: &'a Document, root: NodeId, next: Option<NodeId>) -> Self {
        Self { doc, root, next }
    }
}

impl Iterator for Descendants<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let current = self.next?;
        self.next = next_in_subtree(self.doc, current, self.root);
        Some(current)
    }
}

/// Iterator over the XPath `following` axis: every node that starts after the
/// context node ends, excluding its descendants and attribute nodes.
///
/// Created by [`Document::following`].
#[derive(Debug, Clone)]
pub struct Following<'a> {
    doc: &'a Document,
    next: Option<NodeId>,
}

impl<'a> Following<'a> {
    pub(crate) fn new(doc: &'a Document, origin: NodeId) -> Self {
        // An attribute has no descendants, so the children of its owner element
        // *are* after it in document order and belong to the axis; for every
        // other node the whole subtree is skipped.
        let next = match doc.owner_element(origin) {
            Some(element) => next_in_document(doc, element),
            None => next_skipping_subtree(doc, origin),
        };
        Self { doc, next }
    }
}

impl Iterator for Following<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let current = self.next?;
        self.next = next_in_document(self.doc, current);
        Some(current)
    }
}

/// Iterator over the XPath `preceding` axis: every node that ends before the
/// context node starts, excluding its ancestors and attribute nodes.
///
/// The nodes are produced in reverse document order, which is the order the
/// `preceding` axis is defined to have.
///
/// Created by [`Document::preceding`].
#[derive(Debug, Clone)]
pub struct Preceding<'a> {
    doc: &'a Document,
    origin: NodeId,
    next: Option<NodeId>,
}

impl<'a> Preceding<'a> {
    pub(crate) fn new(doc: &'a Document, origin: NodeId) -> Self {
        let start = doc.owner_element(origin).unwrap_or(origin);
        let mut iter = Self {
            doc,
            origin: start,
            next: previous_in_document(doc, start),
        };
        iter.skip_ancestors();
        iter
    }

    fn skip_ancestors(&mut self) {
        while let Some(candidate) = self.next {
            if self.doc.is_ancestor_of(candidate, self.origin) {
                self.next = previous_in_document(self.doc, candidate);
            } else {
                break;
            }
        }
    }
}

impl Iterator for Preceding<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        let current = self.next?;
        self.next = previous_in_document(self.doc, current);
        self.skip_ancestors();
        Some(current)
    }
}

/// Iterator over the elements of a subtree whose name matches a pattern.
///
/// Created by [`Document::get_elements_by_tag_name`] and
/// [`Document::get_elements_by_tag_name_ns`].
#[derive(Debug, Clone)]
pub struct Elements<'a> {
    inner: Descendants<'a>,
    filter: ElementFilter<'a>,
}

impl<'a> Elements<'a> {
    pub(crate) const fn new(inner: Descendants<'a>, filter: ElementFilter<'a>) -> Self {
        Self { inner, filter }
    }
}

impl Iterator for Elements<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        loop {
            let candidate = self.inner.next()?;
            if self.inner.doc.kind(candidate) != Some(NodeKind::Element) {
                continue;
            }
            if self.filter.matches(self.inner.doc, candidate) {
                return Some(candidate);
            }
        }
    }
}

/// How [`Elements`] decides whether an element matches.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ElementFilter<'a> {
    /// Match on the qualified name; `"*"` matches every element.
    QualifiedName(&'a str),
    /// Match on the expanded name; `None` for the namespace means "no
    /// namespace", `Some("*")` and a local name of `"*"` are wildcards.
    ExpandedName {
        namespace: Option<&'a str>,
        local: &'a str,
    },
}

impl ElementFilter<'_> {
    fn matches(&self, doc: &Document, id: NodeId) -> bool {
        let Some(data) = doc.element(id) else {
            return false;
        };
        match self {
            Self::QualifiedName("*") => true,
            Self::QualifiedName(name) => data.qualified_name() == *name,
            Self::ExpandedName { namespace, local } => {
                let namespace_matches = match namespace {
                    Some("*") => true,
                    Some(ns) => data.namespace_uri() == Some(*ns),
                    None => data.namespace_uri().is_none(),
                };
                namespace_matches && (*local == "*" || data.local_name() == *local)
            }
        }
    }
}

/// Returns the next node of a pre-order walk restricted to the subtree rooted at
/// `root`.
fn next_in_subtree(doc: &Document, current: NodeId, root: NodeId) -> Option<NodeId> {
    if let Some(child) = doc.first_child(current) {
        return Some(child);
    }
    let mut cursor = current;
    loop {
        if cursor == root {
            return None;
        }
        if let Some(sibling) = doc.next_sibling(cursor) {
            return Some(sibling);
        }
        cursor = doc.parent(cursor)?;
    }
}

/// Returns the next node of an unrestricted pre-order walk of the whole tree.
fn next_in_document(doc: &Document, current: NodeId) -> Option<NodeId> {
    if let Some(child) = doc.first_child(current) {
        return Some(child);
    }
    next_skipping_subtree(doc, current)
}

/// Returns the next node in document order that is not inside the subtree rooted
/// at `current`.
fn next_skipping_subtree(doc: &Document, current: NodeId) -> Option<NodeId> {
    let mut cursor = current;
    loop {
        if let Some(sibling) = doc.next_sibling(cursor) {
            return Some(sibling);
        }
        cursor = doc.parent(cursor)?;
    }
}

/// Returns the previous node of a pre-order walk of the whole tree.
fn previous_in_document(doc: &Document, current: NodeId) -> Option<NodeId> {
    match doc.previous_sibling(current) {
        Some(sibling) => Some(deepest_last_descendant(doc, sibling)),
        None => doc.parent(current),
    }
}

fn deepest_last_descendant(doc: &Document, from: NodeId) -> NodeId {
    let mut cursor = from;
    while let Some(last) = doc.last_child(cursor) {
        cursor = last;
    }
    cursor
}
