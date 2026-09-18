//! Structural mutation: the DOM Core child-list operations, cloning, importing
//! and normalization.
//!
//! Every operation performs the well-formedness checks of
//! [DOM Level 3 Core §1.4](https://www.w3.org/TR/DOM-Level-3-Core/core.html#ID-1590626202)
//! *before* touching the arena, so a rejected call leaves the tree exactly as it
//! was.

use crate::document::Document;
use crate::error::{DomError, DomResult, HierarchyRequest, NotFound};
use crate::names::{check_cdata, check_chars, check_comment, check_pi_data};
use crate::node::{NodeContent, NodeId, NodeKind};

impl Document {
    /// Returns `true` when a node of kind `child` is allowed in the child list of
    /// a node of kind `parent`, ignoring the additional constraints the document
    /// node places on the number of its element and document type children.
    #[must_use]
    pub const fn can_contain(parent: NodeKind, child: NodeKind) -> bool {
        use NodeKind::{
            Attr, CdataSection, Comment, Document, DocumentFragment, DocumentType, Element,
            EntityReference, ProcessingInstruction, Text,
        };
        match parent {
            Document => matches!(
                child,
                Element | ProcessingInstruction | Comment | DocumentType
            ),
            DocumentFragment | Element | EntityReference => matches!(
                child,
                Element | Text | CdataSection | Comment | ProcessingInstruction | EntityReference
            ),
            Attr | DocumentType | Text | CdataSection | Comment | ProcessingInstruction => false,
        }
    }

    /// Appends `child` to the end of the child list of `parent`.
    ///
    /// When `child` is a [`DocumentFragment`](NodeKind::DocumentFragment) its
    /// children are moved instead, and the fragment itself is left empty. When
    /// `child` already has a parent it is first removed from it.
    ///
    /// Returns `child`, mirroring DOM `Node.appendChild`.
    ///
    /// # Errors
    ///
    /// * [`DomError::NotFound`] — either identifier is unknown.
    /// * [`DomError::HierarchyRequest`] — the kinds are incompatible, the
    ///   insertion would create a cycle, or a document-node constraint (at most
    ///   one document element, at most one document type, document type first)
    ///   would be violated.
    pub fn append_child(&mut self, parent: NodeId, child: NodeId) -> DomResult<NodeId> {
        self.insert_before(parent, child, None)
    }

    /// Inserts `child` into the child list of `parent` immediately before
    /// `reference`, or at the end when `reference` is `None`.
    ///
    /// Returns `child`, mirroring DOM `Node.insertBefore`.
    ///
    /// # Errors
    ///
    /// In addition to the errors of [`Self::append_child`], returns
    /// [`DomError::NotFound`] when `reference` is not a child of `parent`.
    pub fn insert_before(
        &mut self,
        parent: NodeId,
        child: NodeId,
        reference: Option<NodeId>,
    ) -> DomResult<NodeId> {
        self.check_insertion(parent, child, reference, None)?;

        // WHATWG DOM "pre-insert", step 3: "If referenceChild is node, then set
        // referenceChild to node's next sibling." Without this, `insertBefore(n,
        // n)` — a common no-op idiom — would link `n` before itself and turn the
        // child list into an endless self-referential cycle.
        let reference = if reference == Some(child) {
            self.next_sibling(child)
        } else {
            reference
        };

        if self.node(child)?.kind() == NodeKind::DocumentFragment {
            let moved: Vec<NodeId> = self.children(child).collect();
            for node in moved {
                self.unlink(node);
                self.link_before(parent, node, reference);
            }
            return Ok(child);
        }

        self.unlink(child);
        self.link_before(parent, child, reference);
        Ok(child)
    }

    /// Removes `child` from the child list of `parent`.
    ///
    /// The node is only detached: it stays in the arena, keeps its own subtree,
    /// and the returned identifier can be inserted somewhere else.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] when either identifier is unknown or when
    /// `child` is not a child of `parent`.
    pub fn remove_child(&mut self, parent: NodeId, child: NodeId) -> DomResult<NodeId> {
        self.node(parent)?;
        if self.node(child)?.parent != Some(parent) {
            return Err(DomError::NotFound(NotFound::NotAChild { parent, child }));
        }
        self.unlink(child);
        Ok(child)
    }

    /// Replaces `old_child` with `new_child` in the child list of `parent`.
    ///
    /// Returns `old_child`, which is left detached, mirroring DOM
    /// `Node.replaceChild`.
    ///
    /// # Errors
    ///
    /// The union of the errors of [`Self::insert_before`] and
    /// [`Self::remove_child`].
    pub fn replace_child(
        &mut self,
        parent: NodeId,
        new_child: NodeId,
        old_child: NodeId,
    ) -> DomResult<NodeId> {
        if self.node(old_child)?.parent != Some(parent) {
            return Err(DomError::NotFound(NotFound::NotAChild {
                parent,
                child: old_child,
            }));
        }
        self.check_insertion(parent, new_child, Some(old_child), Some(old_child))?;

        // WHATWG DOM "replace", step 6: "If referenceChild is node, then set
        // referenceChild to node's next sibling." The reference is `old_child`'s
        // next sibling, which *is* `new_child` whenever the two are adjacent;
        // linking `new_child` before itself would corrupt the child list into a
        // cycle.
        let reference = match self.next_sibling(old_child) {
            Some(reference) if reference == new_child => self.next_sibling(new_child),
            other => other,
        };
        self.unlink(old_child);

        if self.node(new_child)?.kind() == NodeKind::DocumentFragment {
            let moved: Vec<NodeId> = self.children(new_child).collect();
            for node in moved {
                self.unlink(node);
                self.link_before(parent, node, reference);
            }
        } else {
            self.unlink(new_child);
            self.link_before(parent, new_child, reference);
        }
        Ok(old_child)
    }

    /// Detaches a node from whatever holds it: the child list of its parent, or
    /// the attribute list of its owner element.
    ///
    /// Detaching a node that is already detached is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] when the identifier is unknown, and
    /// [`DomError::HierarchyRequest`] when asked to detach the document node.
    pub fn detach(&mut self, id: NodeId) -> DomResult<()> {
        let node = self.node(id)?;
        if node.kind() == NodeKind::Document {
            return Err(DomError::HierarchyRequest(
                HierarchyRequest::KindNotAllowed {
                    parent: NodeKind::Document,
                    child: NodeKind::Document,
                },
            ));
        }
        if let NodeContent::Attr(data) = &node.content {
            if let Some(owner) = data.owner {
                self.remove_attribute_node(owner, id)?;
                return Ok(());
            }
        }
        self.unlink(id);
        Ok(())
    }

    /// Replaces the DOM `textContent` of a node.
    ///
    /// Every child is removed and, when `text` is not empty, a single text node
    /// holding it is appended. On a text, CDATA, comment, processing-instruction
    /// or attribute node the value itself is replaced. On the document and
    /// document type nodes the call has no effect, as required by
    /// [DOM Level 3 Core](https://www.w3.org/TR/DOM-Level-3-Core/core.html#Node3-textContent).
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] for an unknown identifier and
    /// [`DomError::InvalidCharacter`] when the text cannot be represented in the
    /// target node.
    pub fn set_text_content(&mut self, id: NodeId, text: impl Into<String>) -> DomResult<()> {
        let text = text.into();
        match self.node(id)?.kind() {
            NodeKind::Document | NodeKind::DocumentType => Ok(()),
            NodeKind::Text
            | NodeKind::CdataSection
            | NodeKind::Comment
            | NodeKind::ProcessingInstruction
            | NodeKind::Attr => self.set_node_value(id, text),
            NodeKind::Element | NodeKind::DocumentFragment | NodeKind::EntityReference => {
                let children: Vec<NodeId> = self.children(id).collect();
                for child in children {
                    self.unlink(child);
                }
                if !text.is_empty() {
                    let node = self.create_text_node(text)?;
                    self.link_before(id, node, None);
                }
                Ok(())
            }
        }
    }

    /// Merges adjacent text nodes and drops empty ones throughout the subtree
    /// rooted at `id`, as DOM `Node.normalize` prescribes.
    ///
    /// CDATA sections are *not* merged into their text neighbours: DOM Level 3
    /// only does that when the `cdata-sections` configuration parameter is
    /// turned off, and this crate always keeps CDATA sections distinct so that a
    /// document round-trips byte-for-byte.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] when the identifier is unknown.
    pub fn normalize(&mut self, id: NodeId) -> DomResult<()> {
        self.node(id)?;
        let mut stack = vec![id];
        while let Some(current) = stack.pop() {
            let children: Vec<NodeId> = self.children(current).collect();
            let mut merge_target: Option<NodeId> = None;
            for child in children {
                match self.node(child)?.kind() {
                    NodeKind::Text => {
                        let text = self.character_data(child).unwrap_or_default().to_owned();
                        match merge_target {
                            Some(target) => {
                                if let NodeContent::Text(existing) =
                                    &mut self.node_mut(target)?.content
                                {
                                    existing.push_str(&text);
                                }
                                self.unlink(child);
                            }
                            None if text.is_empty() => self.unlink(child),
                            None => merge_target = Some(child),
                        }
                    }
                    _ => {
                        if let Some(target) = merge_target.take() {
                            if self.character_data(target).is_some_and(str::is_empty) {
                                self.unlink(target);
                            }
                        }
                        stack.push(child);
                    }
                }
            }
            if let Some(target) = merge_target {
                if self.character_data(target).is_some_and(str::is_empty) {
                    self.unlink(target);
                }
            }
        }
        Ok(())
    }

    /// Creates a copy of a node inside the same document.
    ///
    /// The copy is detached. An element copy always carries a copy of every
    /// attribute, including the namespace declarations; when `deep` is `true` the
    /// whole subtree is copied as well. This is DOM `Node.cloneNode`.
    ///
    /// # Errors
    ///
    /// Returns [`DomError::NotFound`] when the identifier is unknown.
    pub fn clone_node(&mut self, id: NodeId, deep: bool) -> DomResult<NodeId> {
        let root = self.clone_shallow(id)?;
        if deep {
            let mut stack = vec![(id, root)];
            while let Some((source, target)) = stack.pop() {
                let children: Vec<NodeId> = self.children(source).collect();
                for child in children {
                    let copy = self.clone_shallow(child)?;
                    self.link_before(target, copy, None);
                    stack.push((child, copy));
                }
            }
        }
        Ok(root)
    }

    /// Copies a node from another document into this one, as DOM
    /// `Document.importNode` does.
    ///
    /// The copy is detached and belongs to this document. `deep` has the same
    /// meaning as in [`Self::clone_node`].
    ///
    /// # Errors
    ///
    /// * [`DomError::NotFound`] — `id` is unknown to `source`.
    /// * [`DomError::WrongDocument`] — `source` *is* this document; use
    ///   [`Self::clone_node`] instead.
    /// * [`DomError::NotSupported`] — the node is a document node, which cannot
    ///   be imported.
    pub fn import_node(&mut self, source: &Self, id: NodeId, deep: bool) -> DomResult<NodeId> {
        if std::ptr::eq(self, source) {
            return Err(DomError::WrongDocument {
                detail: "importNode was called with the destination document as its source",
            });
        }
        let node = source
            .get(id)
            .ok_or(DomError::NotFound(NotFound::Node(id)))?;
        if node.kind() == NodeKind::Document {
            return Err(DomError::NotSupported {
                detail: "a document node cannot be imported",
            });
        }

        let root = self.import_shallow(source, id)?;
        if deep {
            let mut stack = vec![(id, root)];
            while let Some((from, to)) = stack.pop() {
                let children: Vec<NodeId> = source.children(from).collect();
                for child in children {
                    let copy = self.import_shallow(source, child)?;
                    self.link_before(to, copy, None);
                    stack.push((child, copy));
                }
            }
        }
        Ok(root)
    }

    // -------------------------------------------------------------- internals

    fn clone_shallow(&mut self, id: NodeId) -> DomResult<NodeId> {
        let content = self.node(id)?.content.clone();
        let copy = match content {
            NodeContent::Element(mut data) => {
                data.attributes.clear();
                self.allocate(NodeContent::Element(data))
            }
            NodeContent::Attr(mut data) => {
                data.owner = None;
                self.allocate(NodeContent::Attr(data))
            }
            other => self.allocate(other),
        }?;
        let attributes: Vec<NodeId> = self.attributes(id).collect();
        for attribute in attributes {
            let cloned = self.clone_shallow(attribute)?;
            self.attach_attribute(copy, cloned);
        }
        Ok(copy)
    }

    fn import_shallow(&mut self, source: &Self, id: NodeId) -> DomResult<NodeId> {
        let content = source
            .get(id)
            .ok_or(DomError::NotFound(NotFound::Node(id)))?
            .content
            .clone();
        self.check_imported_content(&content)?;
        let copy = match content {
            NodeContent::Element(mut data) => {
                data.attributes.clear();
                self.allocate(NodeContent::Element(data))
            }
            NodeContent::Attr(mut data) => {
                data.owner = None;
                self.allocate(NodeContent::Attr(data))
            }
            NodeContent::Document => {
                return Err(DomError::NotSupported {
                    detail: "a document node cannot be imported",
                });
            }
            other => self.allocate(other),
        }?;
        let attributes: Vec<NodeId> = source.attributes(id).collect();
        for attribute in attributes {
            let imported = self.import_shallow(source, attribute)?;
            self.attach_attribute(copy, imported);
        }
        Ok(copy)
    }

    /// Holds imported character data to *this* document's `Char` production.
    ///
    /// The source document may be of a different XML version — a text node of an
    /// XML 1.1 document may hold a `RestrictedChar` that an XML 1.0 document
    /// could never write back out — so importing runs the same checks the node
    /// factories do, rather than copying the content unexamined.
    fn check_imported_content(&self, content: &NodeContent) -> DomResult<()> {
        let version = self.char_version();
        match content {
            NodeContent::Text(text) => check_chars(text, version),
            NodeContent::Attr(data) => check_chars(&data.value, version),
            NodeContent::CdataSection(text) => check_cdata(text, version),
            NodeContent::Comment(text) => check_comment(text, version),
            NodeContent::ProcessingInstruction(data) => check_pi_data(&data.data, version),
            _ => Ok(()),
        }
    }

    /// Validates an insertion. `ignored` names a node that the caller is about to
    /// remove, so it does not count towards the document-node constraints.
    fn check_insertion(
        &self,
        parent: NodeId,
        child: NodeId,
        reference: Option<NodeId>,
        ignored: Option<NodeId>,
    ) -> DomResult<()> {
        let parent_node = self.node(parent)?;
        let child_node = self.node(child)?;
        let parent_kind = parent_node.kind();
        let child_kind = child_node.kind();

        if child_kind == NodeKind::Attr {
            return Err(DomError::HierarchyRequest(
                HierarchyRequest::AttributeIsNotAChild,
            ));
        }
        if !child_kind.can_be_child() {
            return Err(DomError::HierarchyRequest(
                HierarchyRequest::KindNotAllowed {
                    parent: parent_kind,
                    child: child_kind,
                },
            ));
        }
        // A node with no children of its own cannot be an ancestor of anything,
        // so the (O(depth)) ancestor walk is skipped for the overwhelmingly
        // common case of appending a freshly created leaf — otherwise building a
        // deeply nested tree one node at a time is quadratic in its depth.
        let child_may_be_an_ancestor = child_node.first_child.is_some();
        if child == parent || (child_may_be_an_ancestor && self.is_ancestor_of(child, parent)) {
            return Err(DomError::HierarchyRequest(
                HierarchyRequest::WouldCreateCycle { parent, child },
            ));
        }
        if let Some(reference) = reference {
            if self.node(reference)?.parent != Some(parent) {
                return Err(DomError::NotFound(NotFound::NotAChild {
                    parent,
                    child: reference,
                }));
            }
        }

        if child_kind == NodeKind::DocumentFragment {
            return self.check_fragment(parent, parent_kind, child, reference, ignored);
        }
        self.check_child_kind(parent, parent_kind, child_kind, reference, ignored)
    }

    /// Validates the insertion of a document fragment, whose children are moved
    /// into `parent` as a group.
    ///
    /// [`Self::check_child_kind`] only ever sees the tree as it stands *before*
    /// the insertion, so checking each child of the fragment against it in
    /// isolation would let a fragment holding two elements past the document
    /// node's "at most one document element" constraint: neither element sees
    /// the other. What the fragment itself contributes is therefore counted
    /// across the loop. Only elements have to be counted this way — a document
    /// type declaration is not among the children [`Self::can_contain`] allows
    /// inside a fragment in the first place.
    fn check_fragment(
        &self,
        parent: NodeId,
        parent_kind: NodeKind,
        fragment: NodeId,
        reference: Option<NodeId>,
        ignored: Option<NodeId>,
    ) -> DomResult<()> {
        let mut seen_element = false;
        for grandchild in self.children(fragment) {
            let child_kind = self.node(grandchild)?.kind();
            self.check_child_kind(parent, parent_kind, child_kind, reference, ignored)?;
            if parent_kind != NodeKind::Document || child_kind != NodeKind::Element {
                continue;
            }
            if seen_element {
                return Err(DomError::HierarchyRequest(
                    HierarchyRequest::MultipleDocumentElements,
                ));
            }
            seen_element = true;
        }
        Ok(())
    }

    fn check_child_kind(
        &self,
        parent: NodeId,
        parent_kind: NodeKind,
        child_kind: NodeKind,
        reference: Option<NodeId>,
        ignored: Option<NodeId>,
    ) -> DomResult<()> {
        if !Self::can_contain(parent_kind, child_kind) {
            return Err(DomError::HierarchyRequest(
                HierarchyRequest::KindNotAllowed {
                    parent: parent_kind,
                    child: child_kind,
                },
            ));
        }
        if parent_kind != NodeKind::Document {
            return Ok(());
        }

        let existing_element = self
            .children(parent)
            .find(|&id| Some(id) != ignored && self.kind(id) == Some(NodeKind::Element));
        let existing_doctype = self
            .children(parent)
            .find(|&id| Some(id) != ignored && self.kind(id) == Some(NodeKind::DocumentType));

        match child_kind {
            NodeKind::Element => {
                if existing_element.is_some() {
                    return Err(DomError::HierarchyRequest(
                        HierarchyRequest::MultipleDocumentElements,
                    ));
                }
                // The document element may not be placed before the document type
                // declaration, i.e. the reference child must lie strictly after it.
                if let (Some(doctype), Some(reference)) = (existing_doctype, reference) {
                    if self.follows_or_equals(doctype, reference) {
                        return Err(DomError::HierarchyRequest(
                            HierarchyRequest::DoctypeAfterDocumentElement,
                        ));
                    }
                }
            }
            NodeKind::DocumentType => {
                if existing_doctype.is_some() {
                    return Err(DomError::HierarchyRequest(
                        HierarchyRequest::MultipleDoctypes,
                    ));
                }
                if let Some(element) = existing_element {
                    // The declaration must go in at or before the document element.
                    let ok = reference
                        .is_some_and(|reference| self.follows_or_equals(element, reference));
                    if !ok {
                        return Err(DomError::HierarchyRequest(
                            HierarchyRequest::DoctypeAfterDocumentElement,
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Returns `true` when `a` is `b` or a following sibling of `b`.
    fn follows_or_equals(&self, a: NodeId, b: NodeId) -> bool {
        let mut cursor = Some(b);
        while let Some(current) = cursor {
            if current == a {
                return true;
            }
            cursor = self.next_sibling(current);
        }
        false
    }

    /// Removes a node from the child list of its parent, without any checks.
    pub(crate) fn unlink(&mut self, child: NodeId) {
        let Some(node) = self.get(child) else { return };
        let (parent, previous, next) = (node.parent, node.prev_sibling, node.next_sibling);

        if let Some(parent) = parent {
            if let Some(parent_node) = self.get_mut(parent) {
                if parent_node.first_child == Some(child) {
                    parent_node.first_child = next;
                }
                if parent_node.last_child == Some(child) {
                    parent_node.last_child = previous;
                }
            }
        }
        if let Some(previous) = previous {
            if let Some(node) = self.get_mut(previous) {
                node.next_sibling = next;
            }
        }
        if let Some(next) = next {
            if let Some(node) = self.get_mut(next) {
                node.prev_sibling = previous;
            }
        }
        if let Some(node) = self.get_mut(child) {
            node.parent = None;
            node.prev_sibling = None;
            node.next_sibling = None;
        }
        self.invalidate_order();
    }

    /// Links `child` into the child list of `parent`, immediately before
    /// `reference` (or at the end), without any checks. `child` must already be
    /// unlinked.
    pub(crate) fn link_before(&mut self, parent: NodeId, child: NodeId, reference: Option<NodeId>) {
        // Defence in depth: a node can never be inserted before itself. Callers
        // apply the WHATWG pre-insert adjustment first; because `child` is
        // already unlinked here it has no next sibling left to fall back to, so
        // the adjustment degenerates to "append at the end".
        let reference = if reference == Some(child) {
            None
        } else {
            reference
        };

        let previous = match reference {
            Some(reference) => self.previous_sibling(reference),
            None => self.last_child(parent),
        };

        if let Some(node) = self.get_mut(child) {
            node.parent = Some(parent);
            node.prev_sibling = previous;
            node.next_sibling = reference;
        }
        match previous {
            Some(previous) => {
                if let Some(node) = self.get_mut(previous) {
                    node.next_sibling = Some(child);
                }
            }
            None => {
                if let Some(node) = self.get_mut(parent) {
                    node.first_child = Some(child);
                }
            }
        }
        match reference {
            Some(reference) => {
                if let Some(node) = self.get_mut(reference) {
                    node.prev_sibling = Some(child);
                }
            }
            None => {
                if let Some(node) = self.get_mut(parent) {
                    node.last_child = Some(child);
                }
            }
        }
        self.invalidate_order();
    }
}
