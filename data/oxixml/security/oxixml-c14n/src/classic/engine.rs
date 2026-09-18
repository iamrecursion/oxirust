//! The depth-first traversal shared by Canonical XML 1.0, 1.1 and Exclusive.
//!
//! [Canonical XML §2.3](https://www.w3.org/TR/xml-c14n#ProcessingModel): "The
//! XPath node-set is converted into an octet stream, the canonical form, by
//! generating the representative UCS characters for each node in the node-set in
//! ascending document order, then encoding the result in UTF-8." A node that is
//! *not* in the node-set still has its namespace axis, attribute axis and
//! children processed — that is what makes a document subset work.

use std::collections::BTreeMap;
use std::rc::Rc;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::error::{C14nError, C14nResult};
use crate::escape::{escape_text, write_comment, write_processing_instruction};
use crate::namespaces::Bindings;
use crate::nodeset::NodeSet;

use super::Variant;

/// Where a child of the document node sits relative to the document element,
/// which decides the line feed [Canonical XML §2.3][spec] renders around
/// top-level processing instructions and comments.
///
/// [spec]: https://www.w3.org/TR/xml-c14n#ProcessingModel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootPosition {
    /// A child of the document node preceding the document element.
    Prologue,
    /// A child of the document node following the document element.
    Epilogue,
    /// Anything inside the document element.
    Inside,
}

/// The traversal state inherited by an element from its parent.
#[derive(Debug, Clone)]
pub(crate) struct Frame {
    /// The namespace axis in force inside the parent element.
    pub(crate) bindings: Bindings,
    /// The nearest ancestor element that is in the node-set, together with its
    /// own namespace axis — the "nearest ancestor element … in the node-set"
    /// that the inclusive namespace rules keep referring to.
    pub(crate) visible: Option<(NodeId, Bindings)>,
    /// Exclusive C14N only: the `ns_rendered` dictionary of prefixes already
    /// rendered by an output ancestor.
    pub(crate) rendered: BTreeMap<String, String>,
}

impl Frame {
    fn root() -> Self {
        Self {
            bindings: Bindings::root(),
            visible: None,
            rendered: BTreeMap::new(),
        }
    }
}

/// One item of work on the explicit traversal stack.
///
/// The walk is depth-first, but it is *not* written as a recursive function:
/// the depth of an XML document is chosen by whoever wrote the document, and an
/// XML Signature verifier canonicalizes before it has authenticated anything.
/// A recursive walk turns a few kilobytes of `<a><a><a>…` into a stack overflow,
/// which aborts the process instead of returning a [`C14nError`]. Keeping the
/// pending work on the heap makes the depth a matter of memory, not of stack.
enum Task {
    /// Render a node and schedule whatever it contains.
    Node {
        node: NodeId,
        frame: Rc<Frame>,
        position: RootPosition,
    },
    /// Write the end tag of an element whose children are all done.
    EndTag { name: String },
}

/// Renders a document subset into its canonical form.
pub(crate) struct Engine<'a> {
    pub(crate) document: &'a Document,
    pub(crate) subset: &'a NodeSet,
    pub(crate) variant: &'a Variant,
    pub(crate) with_comments: bool,
    pub(crate) out: String,
}

impl<'a> Engine<'a> {
    pub(crate) fn new(
        document: &'a Document,
        subset: &'a NodeSet,
        variant: &'a Variant,
        with_comments: bool,
    ) -> Self {
        Self {
            document,
            subset,
            variant,
            with_comments,
            out: String::new(),
        }
    }

    /// Walks the whole document and returns the canonical octets.
    pub(crate) fn run(mut self) -> C14nResult<Vec<u8>> {
        let root = self.document.root();
        let document_element = self.document.document_element();
        let frame = Rc::new(Frame::root());

        let mut pending: Vec<Task> = Vec::new();
        let mut seen_document_element = false;
        for child in self.document.children(root).collect::<Vec<_>>() {
            let position = if Some(child) == document_element {
                seen_document_element = true;
                RootPosition::Inside
            } else if seen_document_element {
                RootPosition::Epilogue
            } else {
                RootPosition::Prologue
            };
            pending.push(Task::Node {
                node: child,
                frame: Rc::clone(&frame),
                position,
            });
        }

        // The stack is popped from the end, so the first child has to go on last.
        let mut stack: Vec<Task> = pending.into_iter().rev().collect();
        while let Some(task) = stack.pop() {
            match task {
                Task::Node {
                    node,
                    frame,
                    position,
                } => self.process(node, &frame, position, &mut stack)?,
                Task::EndTag { name } => {
                    self.out.push_str("</");
                    self.out.push_str(&name);
                    self.out.push('>');
                }
            }
        }
        Ok(self.out.into_bytes())
    }

    /// Schedules the children of `parent`, in document order.
    fn push_children(
        &self,
        parent: NodeId,
        frame: &Rc<Frame>,
        position: RootPosition,
        stack: &mut Vec<Task>,
    ) {
        for child in self
            .document
            .children(parent)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            stack.push(Task::Node {
                node: child,
                frame: Rc::clone(frame),
                position,
            });
        }
    }

    fn process(
        &mut self,
        node: NodeId,
        frame: &Rc<Frame>,
        position: RootPosition,
        stack: &mut Vec<Task>,
    ) -> C14nResult<()> {
        let Some(kind) = self.document.kind(node) else {
            return Err(C14nError::UnknownNode { node });
        };
        match kind {
            NodeKind::Element => self.process_element(node, frame, stack),
            NodeKind::Text | NodeKind::CdataSection => {
                if self.subset.contains(node) {
                    if let Some(text) = self.document.character_data(node) {
                        escape_text(&mut self.out, text);
                    }
                }
                Ok(())
            }
            NodeKind::Comment => {
                if self.with_comments && self.subset.contains(node) {
                    if let Some(text) = self.document.character_data(node) {
                        if position == RootPosition::Epilogue {
                            self.out.push('\n');
                        }
                        write_comment(&mut self.out, text);
                        if position == RootPosition::Prologue {
                            self.out.push('\n');
                        }
                    }
                }
                Ok(())
            }
            NodeKind::ProcessingInstruction => {
                if self.subset.contains(node) {
                    if let Some(data) = self.document.processing_instruction(node) {
                        if position == RootPosition::Epilogue {
                            self.out.push('\n');
                        }
                        write_processing_instruction(&mut self.out, data.target(), data.data());
                        if position == RootPosition::Prologue {
                            self.out.push('\n');
                        }
                    }
                }
                Ok(())
            }
            NodeKind::EntityReference => {
                let name = self
                    .document
                    .entity_reference(node)
                    .unwrap_or_default()
                    .to_owned();
                if self.document.first_child(node).is_none() {
                    return Err(C14nError::UnexpandedEntityReference { name });
                }
                self.push_children(node, frame, position, stack);
                Ok(())
            }
            // The document type declaration contributes nothing, and neither a
            // second document node nor a detached fragment can be reached here.
            NodeKind::DocumentType
            | NodeKind::Document
            | NodeKind::DocumentFragment
            | NodeKind::Attr => Ok(()),
        }
    }

    fn process_element(
        &mut self,
        element: NodeId,
        frame: &Rc<Frame>,
        stack: &mut Vec<Task>,
    ) -> C14nResult<()> {
        let bindings = frame.bindings.child(self.document, element);
        let in_subset = self.subset.contains(element);
        let mut rendered = frame.rendered.clone();

        let name = self
            .document
            .element(element)
            .map(|data| data.qualified_name())
            .ok_or(C14nError::NotAnElement {
                node: element,
                kind: self.document.kind(element).unwrap_or(NodeKind::Document),
            })?;

        if in_subset {
            self.out.push('<');
            self.out.push_str(&name);
        }
        self.render_namespace_axis(element, &bindings, frame, &mut rendered)?;
        self.render_attribute_axis(element)?;
        if in_subset {
            self.out.push('>');
        }

        let visible = if in_subset {
            Some((element, bindings.clone()))
        } else {
            frame.visible.clone()
        };
        let child_frame = Rc::new(Frame {
            bindings,
            visible,
            rendered,
        });
        if in_subset {
            stack.push(Task::EndTag { name });
        }
        self.push_children(element, &child_frame, RootPosition::Inside, stack);
        Ok(())
    }
}
