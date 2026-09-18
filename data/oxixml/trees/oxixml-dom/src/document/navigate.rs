//! Read-only navigation of the tree.

use crate::document::Document;
use crate::iter::{Ancestors, Attributes, Children, Descendants, Following, Preceding, Siblings};
use crate::node::{NodeContent, NodeId, NodeKind};

impl Document {
    /// Returns the parent of a node.
    ///
    /// Attribute nodes are not children of their element, so this returns `None`
    /// for them — use [`Self::owner_element`] instead. The document node has no
    /// parent either.
    #[must_use]
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.get(id)?.parent
    }

    /// Returns the element an attribute node belongs to, or `None` for any other
    /// kind of node and for a detached attribute.
    #[must_use]
    pub fn owner_element(&self, id: NodeId) -> Option<NodeId> {
        match &self.get(id)?.content {
            NodeContent::Attr(data) => data.owner,
            _ => None,
        }
    }

    /// Returns the first child of a node.
    #[must_use]
    pub fn first_child(&self, id: NodeId) -> Option<NodeId> {
        self.get(id)?.first_child
    }

    /// Returns the last child of a node.
    #[must_use]
    pub fn last_child(&self, id: NodeId) -> Option<NodeId> {
        self.get(id)?.last_child
    }

    /// Returns the next sibling of a node.
    #[must_use]
    pub fn next_sibling(&self, id: NodeId) -> Option<NodeId> {
        self.get(id)?.next_sibling
    }

    /// Returns the previous sibling of a node.
    #[must_use]
    pub fn previous_sibling(&self, id: NodeId) -> Option<NodeId> {
        self.get(id)?.prev_sibling
    }

    /// Returns `true` when the node has at least one child.
    #[must_use]
    pub fn has_child_nodes(&self, id: NodeId) -> bool {
        self.first_child(id).is_some()
    }

    /// Returns `true` when the element has at least one attribute.
    #[must_use]
    pub fn has_attributes(&self, id: NodeId) -> bool {
        self.element(id)
            .is_some_and(|data| !data.attributes.is_empty())
    }

    /// Iterates over the children of a node in document order.
    ///
    /// An unknown identifier, or a node that cannot have children, yields an
    /// empty iterator.
    #[must_use]
    pub fn children(&self, id: NodeId) -> Children<'_> {
        Children::new(self, id)
    }

    /// Iterates over the attribute nodes of an element in the order they were
    /// added. Namespace declarations are ordinary attributes in the
    /// `http://www.w3.org/2000/xmlns/` namespace and are included.
    #[must_use]
    pub fn attributes(&self, id: NodeId) -> Attributes<'_> {
        Attributes::new(self, id)
    }

    /// Iterates over the ancestors of a node, nearest first, ending at the
    /// document node.
    ///
    /// For an attribute node the walk starts at its owner element.
    #[must_use]
    pub fn ancestors(&self, id: NodeId) -> Ancestors<'_> {
        let start = match self.get(id) {
            Some(node) => match &node.content {
                NodeContent::Attr(data) => data.owner,
                _ => node.parent,
            },
            None => None,
        };
        Ancestors::new(self, start)
    }

    /// Iterates over the following siblings of a node.
    #[must_use]
    pub fn following_siblings(&self, id: NodeId) -> Siblings<'_> {
        Siblings::new(self, self.next_sibling(id), true)
    }

    /// Iterates over the preceding siblings of a node, nearest first.
    #[must_use]
    pub fn preceding_siblings(&self, id: NodeId) -> Siblings<'_> {
        Siblings::new(self, self.previous_sibling(id), false)
    }

    /// Iterates over the subtree rooted at `id` in document order, *excluding*
    /// `id` itself. Attribute nodes are not visited.
    #[must_use]
    pub fn descendants(&self, id: NodeId) -> Descendants<'_> {
        Descendants::new(self, id, self.first_child(id))
    }

    /// Iterates over the subtree rooted at `id` in document order, *including*
    /// `id` itself. Attribute nodes are not visited.
    #[must_use]
    pub fn traverse(&self, id: NodeId) -> Descendants<'_> {
        let start = if self.contains(id) { Some(id) } else { None };
        Descendants::new(self, id, start)
    }

    /// Iterates over the XPath `following` axis of a node: every node that
    /// starts after the context node ends, excluding its descendants and
    /// attribute nodes, in document order.
    #[must_use]
    pub fn following(&self, id: NodeId) -> Following<'_> {
        Following::new(self, id)
    }

    /// Iterates over the XPath `preceding` axis of a node: every node that ends
    /// before the context node starts, excluding its ancestors and attribute
    /// nodes, in reverse document order.
    #[must_use]
    pub fn preceding(&self, id: NodeId) -> Preceding<'_> {
        Preceding::new(self, id)
    }

    /// Returns `true` when `candidate` is an ancestor of `id`.
    ///
    /// A node is not its own ancestor. For an attribute node the walk starts at
    /// its owner element, so the owner element and everything above it are
    /// ancestors of the attribute.
    #[must_use]
    pub fn is_ancestor_of(&self, candidate: NodeId, id: NodeId) -> bool {
        self.ancestors(id).any(|ancestor| ancestor == candidate)
    }

    /// Returns the root of the tree the node belongs to: the document node for a
    /// node that is in the document, or the root of a detached subtree.
    #[must_use]
    pub fn tree_root(&self, id: NodeId) -> Option<NodeId> {
        if !self.contains(id) {
            return None;
        }
        Some(self.ancestors(id).last().unwrap_or(id))
    }

    /// Returns the document element — the single element child of the document
    /// node.
    #[must_use]
    pub fn document_element(&self) -> Option<NodeId> {
        self.children(self.root())
            .find(|&child| self.kind(child) == Some(NodeKind::Element))
    }

    /// Returns the document type declaration node, if the document has one.
    #[must_use]
    pub fn doctype(&self) -> Option<NodeId> {
        self.children(self.root())
            .find(|&child| self.kind(child) == Some(NodeKind::DocumentType))
    }

    /// Returns the position of `child` in the child list of its parent, counting
    /// from zero.
    #[must_use]
    pub fn child_index(&self, child: NodeId) -> Option<usize> {
        let parent = self.parent(child)?;
        self.children(parent)
            .position(|candidate| candidate == child)
    }

    /// Returns the DOM `textContent` of a node: the concatenation of the
    /// character data of every descendant text and CDATA node, in document
    /// order.
    ///
    /// Comment and processing-instruction *children* contribute nothing, matching
    /// the table in
    /// [DOM Level 3 Core `Node.textContent`](https://www.w3.org/TR/DOM-Level-3-Core/core.html#Node3-textContent):
    /// a text, CDATA, comment or processing-instruction node yields its own
    /// `nodeValue`, an attribute node yields its value, and the document and
    /// document type nodes yield `null` — represented here by `None`.
    #[must_use]
    pub fn text_content(&self, id: NodeId) -> Option<String> {
        let node = self.get(id)?;
        match &node.content {
            NodeContent::Text(text)
            | NodeContent::CdataSection(text)
            | NodeContent::Comment(text) => Some(text.clone()),
            NodeContent::Attr(data) => Some(data.value.clone()),
            NodeContent::ProcessingInstruction(data) => Some(data.data.clone()),
            NodeContent::Document | NodeContent::DocumentType(_) => None,
            NodeContent::DocumentFragment
            | NodeContent::Element(_)
            | NodeContent::EntityReference(_) => {
                let mut out = String::new();
                for descendant in self.descendants(id) {
                    if let Some(node) = self.get(descendant) {
                        match &node.content {
                            NodeContent::Text(text) | NodeContent::CdataSection(text) => {
                                out.push_str(text);
                            }
                            _ => {}
                        }
                    }
                }
                Some(out)
            }
        }
    }
}
