//! The Canonical XML 2.0 traversal.
//!
//! The shape follows [§2.3][model] and [§2.5.3][ns] of the specification: walk
//! each apex subtree depth-first, and at every element compute the list of
//! visibly utilized namespace declarations, rewrite their prefixes if asked to,
//! drop the ones an output ancestor already wrote, sort what is left by prefix,
//! and emit it before the attributes.
//!
//! [model]: https://www.w3.org/TR/xml-c14n2/#sec-Processing-Model
//! [ns]: https://www.w3.org/TR/xml-c14n2/#sec-Namespace-Processing-Algorithm

use std::collections::BTreeMap;
use std::rc::Rc;

use oxixml_dom::{Document, NodeId, NodeKind, XML_NAMESPACE};

use crate::error::{C14nError, C14nResult};
use crate::escape::{escape_text, write_attribute, write_comment, write_processing_instruction};
use crate::namespaces::{
    Bindings, DEFAULT_PREFIX, XML_PREFIX, is_namespace_declaration, namespace_node_name,
};

use super::qname::{qname_prefix, rewrite_qname, rewrite_xpath_prefixes, xpath_prefixes};
use super::{C14n2Parameters, C14n2Subset, PrefixRewrite};

/// Where a node sits relative to the document element, for the line feeds
/// rendered around top-level processing instructions and comments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    Prologue,
    Epilogue,
    Inside,
}

/// What an element inherits from its parent.
#[derive(Debug, Clone)]
struct Frame {
    /// The namespace bindings in force inside the parent element.
    bindings: Bindings,
    /// The prefix → namespace name pairs already written by an output ancestor.
    rendered: BTreeMap<String, String>,
    /// Whether an ancestor set `xml:space="preserve"`.
    preserve_space: bool,
}

/// One item of work on the explicit traversal stack.
///
/// As in the classic engine, the depth-first walk keeps its pending work on the
/// heap rather than on the call stack: the nesting depth of the input is chosen
/// by whoever wrote the document, and a recursive walk would turn a few
/// kilobytes of `<a><a><a>…` into a stack overflow — an abort, not a
/// [`C14nError`] a caller could handle.
enum Task {
    /// Render a node and schedule whatever it contains.
    Node {
        node: NodeId,
        frame: Rc<Frame>,
        position: Position,
    },
    /// Write one coalesced run of character data.
    ///
    /// The text is kept raw and only trimmed and prefix-rewritten when the task
    /// is popped, so that `QName` rewriting sees exactly the prefix assignments
    /// that were in force at that point of the document order.
    Text {
        text: String,
        frame: Rc<Frame>,
        element: NodeId,
        rewrite_qname_content: bool,
        rewrite_xpath_content: bool,
    },
    /// Write the end tag of an element whose children are all done.
    EndTag { name: String },
}

/// One namespace declaration that survived filtering.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Declaration {
    prefix: String,
    namespace: String,
}

pub(super) struct Engine<'a> {
    document: &'a Document,
    subset: &'a C14n2Subset,
    parameters: &'a C14n2Parameters,
    out: String,
    /// Namespace name → rewritten prefix, shared by the whole canonicalization
    /// so that "once a namespace URI has been assigned a prefix, it always gets
    /// that prefix everywhere in the document".
    assigned: BTreeMap<String, String>,
    counter: u32,
}

impl<'a> Engine<'a> {
    pub(super) fn new(
        document: &'a Document,
        subset: &'a C14n2Subset,
        parameters: &'a C14n2Parameters,
    ) -> Self {
        Self {
            document,
            subset,
            parameters,
            out: String::new(),
            assigned: BTreeMap::new(),
            counter: 0,
        }
    }

    pub(super) fn run(mut self) -> C14nResult<Vec<u8>> {
        let mut stack: Vec<Task> = Vec::new();
        if self.subset.is_whole_document() {
            self.run_document(&mut stack)?;
        } else {
            for apex in self.subset.apex_nodes(self.document) {
                self.run_apex(apex, &mut stack)?;
            }
        }
        Ok(self.out.into_bytes())
    }

    /// Drains the pending work, deepest-first, until the subtree is written.
    fn drive(&mut self, stack: &mut Vec<Task>) -> C14nResult<()> {
        while let Some(task) = stack.pop() {
            match task {
                Task::Node {
                    node,
                    frame,
                    position,
                } => self.process(node, &frame, position, stack)?,
                Task::Text {
                    text,
                    frame,
                    element,
                    rewrite_qname_content,
                    rewrite_xpath_content,
                } => self.write_text(
                    &text,
                    &frame,
                    element,
                    rewrite_qname_content,
                    rewrite_xpath_content,
                ),
                Task::EndTag { name } => {
                    self.out.push_str("</");
                    self.out.push_str(&name);
                    self.out.push('>');
                }
            }
        }
        Ok(())
    }

    fn run_document(&mut self, stack: &mut Vec<Task>) -> C14nResult<()> {
        let root = self.document.root();
        let document_element = self.document.document_element();
        let frame = Rc::new(Frame {
            bindings: Bindings::root(),
            rendered: self.initial_rendered(),
            preserve_space: false,
        });

        let mut pending: Vec<Task> = Vec::new();
        let mut seen = false;
        for child in self.document.children(root).collect::<Vec<_>>() {
            let position = if Some(child) == document_element {
                seen = true;
                Position::Inside
            } else if seen {
                Position::Epilogue
            } else {
                Position::Prologue
            };
            pending.push(Task::Node {
                node: child,
                frame: Rc::clone(&frame),
                position,
            });
        }
        stack.extend(pending.into_iter().rev());
        self.drive(stack)
    }

    fn run_apex(&mut self, apex: NodeId, stack: &mut Vec<Task>) -> C14nResult<()> {
        match self.document.kind(apex) {
            Some(NodeKind::Element) => {}
            Some(kind) => return Err(C14nError::NotAnElement { node: apex, kind }),
            None => return Err(C14nError::UnknownNode { node: apex }),
        }
        if self.subset.is_excluded_element(apex) {
            return Ok(());
        }
        let bindings = match self.document.parent(apex) {
            Some(parent) => Bindings::at(self.document, parent),
            None => Bindings::root(),
        };
        let frame = Rc::new(Frame {
            bindings,
            rendered: self.initial_rendered(),
            preserve_space: self.inherited_space_preserve(apex),
        });
        self.process_element(apex, &frame, stack)?;
        self.drive(stack)
    }

    /// The output context an apex starts from.
    ///
    /// Without prefix rewriting the default namespace starts out bound to the
    /// empty string, so an element in no namespace does not emit a redundant
    /// `xmlns=""`. With rewriting there is no default namespace at all — every
    /// namespace name, the empty one included, gets an `nN` prefix.
    fn initial_rendered(&self) -> BTreeMap<String, String> {
        let mut rendered = BTreeMap::new();
        if self.parameters.prefix_rewrite() == PrefixRewrite::None {
            rendered.insert(DEFAULT_PREFIX.to_owned(), String::new());
        }
        rendered
    }

    /// Looks up the `xml:space` value in force above `element`.
    fn inherited_space_preserve(&self, element: NodeId) -> bool {
        for ancestor in self.document.ancestors(element) {
            if let Some(value) = self.xml_space(ancestor) {
                return value;
            }
        }
        false
    }

    /// Returns the `xml:space` setting written on `element`, if any.
    fn xml_space(&self, element: NodeId) -> Option<bool> {
        self.document.attributes(element).find_map(|attribute| {
            let data = self.document.attr(attribute)?;
            if data.namespace_uri() != Some(XML_NAMESPACE) || data.local_name() != "space" {
                return None;
            }
            match data.value() {
                "preserve" => Some(true),
                "default" => Some(false),
                _ => None,
            }
        })
    }

    fn process(
        &mut self,
        node: NodeId,
        frame: &Rc<Frame>,
        position: Position,
        stack: &mut Vec<Task>,
    ) -> C14nResult<()> {
        let Some(kind) = self.document.kind(node) else {
            return Err(C14nError::UnknownNode { node });
        };
        match kind {
            NodeKind::Element => {
                if self.subset.is_excluded_element(node) {
                    return Ok(());
                }
                self.process_element(node, frame, stack)
            }
            NodeKind::Comment => {
                if self.parameters.ignore_comments() {
                    return Ok(());
                }
                if let Some(text) = self.document.character_data(node) {
                    if position == Position::Epilogue {
                        self.out.push('\n');
                    }
                    write_comment(&mut self.out, text);
                    if position == Position::Prologue {
                        self.out.push('\n');
                    }
                }
                Ok(())
            }
            NodeKind::ProcessingInstruction => {
                if let Some(data) = self.document.processing_instruction(node) {
                    if position == Position::Epilogue {
                        self.out.push('\n');
                    }
                    write_processing_instruction(&mut self.out, data.target(), data.data());
                    if position == Position::Prologue {
                        self.out.push('\n');
                    }
                }
                Ok(())
            }
            NodeKind::EntityReference => Err(C14nError::UnexpandedEntityReference {
                name: self
                    .document
                    .entity_reference(node)
                    .unwrap_or_default()
                    .to_owned(),
            }),
            // Text nodes are handled by `schedule_children`, which has to see
            // adjacent runs of them at once in order to trim them correctly.
            NodeKind::Text
            | NodeKind::CdataSection
            | NodeKind::DocumentType
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
        let utilized = self.visibly_utilized(element, &bindings);
        if self.parameters.prefix_rewrite() == PrefixRewrite::Sequential {
            self.assign_prefixes(&utilized);
        }

        let declarations = self.filter_declarations(&utilized, &frame.rendered);
        let mut rendered = frame.rendered.clone();
        for declaration in &declarations {
            rendered.insert(declaration.prefix.clone(), declaration.namespace.clone());
        }

        let name = self.element_name(element)?;
        self.out.push('<');
        self.out.push_str(&name);
        for declaration in &declarations {
            write_attribute(
                &mut self.out,
                &namespace_node_name(&declaration.prefix),
                &declaration.namespace,
            );
        }
        self.write_attributes(element)?;
        self.out.push('>');

        let child_frame = Rc::new(Frame {
            bindings,
            rendered,
            preserve_space: self.xml_space(element).unwrap_or(frame.preserve_space),
        });
        stack.push(Task::EndTag { name });
        self.schedule_children(element, &child_frame, stack);
        Ok(())
    }

    /// Step 1 of §2.5.3: the prefixes this element visibly utilizes, mapped to
    /// the namespace names they are bound to. The `xml` prefix is removed, as
    /// Canonical XML 2.0 never emits its declaration.
    fn visibly_utilized(&self, element: NodeId, bindings: &Bindings) -> BTreeMap<String, String> {
        let mut utilized: BTreeMap<String, String> = BTreeMap::new();
        let Some(data) = self.document.element(element) else {
            return utilized;
        };
        let element_namespace = data.namespace_uri().unwrap_or_default().to_owned();
        let element_local = data.local_name().to_owned();
        utilized.insert(
            data.prefix().unwrap_or(DEFAULT_PREFIX).to_owned(),
            element_namespace.clone(),
        );

        for attribute in self.document.attributes(element) {
            if self.subset.is_excluded_attribute(attribute)
                || is_namespace_declaration(self.document, attribute)
            {
                continue;
            }
            let Some(attribute_data) = self.document.attr(attribute) else {
                continue;
            };
            if let Some(prefix) = attribute_data.prefix() {
                utilized.insert(
                    prefix.to_owned(),
                    attribute_data
                        .namespace_uri()
                        .unwrap_or_default()
                        .to_owned(),
                );
            }
            let aware = self.parameters.qname_aware();
            let is_qname_valued = match attribute_data.namespace_uri() {
                Some(namespace) => {
                    aware.is_qname_qualified_attribute(namespace, attribute_data.local_name())
                }
                None => aware.is_qname_unqualified_attribute(
                    &element_namespace,
                    &element_local,
                    attribute_data.local_name(),
                ),
            };
            if is_qname_valued {
                self.utilize_prefix(
                    &mut utilized,
                    bindings,
                    qname_prefix(attribute_data.value()),
                );
            }
        }

        let aware = self.parameters.qname_aware();
        if aware.is_qname_element(&element_namespace, &element_local) {
            let content = self.document.text_content(element).unwrap_or_default();
            self.utilize_prefix(&mut utilized, bindings, qname_prefix(&content));
        }
        if aware.is_xpath_element(&element_namespace, &element_local) {
            let content = self.document.text_content(element).unwrap_or_default();
            for prefix in xpath_prefixes(&content) {
                self.utilize_prefix(&mut utilized, bindings, &prefix);
            }
        }

        utilized.remove(XML_PREFIX);
        utilized
    }

    /// Records one prefix found in content, provided it actually resolves.
    fn utilize_prefix(
        &self,
        utilized: &mut BTreeMap<String, String>,
        bindings: &Bindings,
        prefix: &str,
    ) {
        if prefix == XML_PREFIX {
            return;
        }
        match bindings.get(prefix) {
            Some(namespace) => {
                utilized.insert(prefix.to_owned(), namespace.to_owned());
            }
            None if prefix == DEFAULT_PREFIX => {
                utilized.insert(DEFAULT_PREFIX.to_owned(), String::new());
            }
            // A prefix that is not bound has no declaration to emit; the
            // specification leaves this case to the application.
            None => {}
        }
    }

    /// Step 2 of §2.5.3: hand out `nN` prefixes to the namespace names this
    /// element utilizes, in ascending order of namespace name.
    fn assign_prefixes(&mut self, utilized: &BTreeMap<String, String>) {
        let mut namespaces: Vec<&String> = utilized.values().collect();
        namespaces.sort();
        namespaces.dedup();
        for namespace in namespaces {
            if !self.assigned.contains_key(namespace) {
                self.assigned
                    .insert(namespace.clone(), format!("n{}", self.counter));
                self.counter += 1;
            }
        }
    }

    /// Steps 3 and 4 of §2.5.3: drop the declarations an output ancestor already
    /// wrote and sort the rest by (rewritten) prefix.
    fn filter_declarations(
        &self,
        utilized: &BTreeMap<String, String>,
        rendered: &BTreeMap<String, String>,
    ) -> Vec<Declaration> {
        let mut declarations: Vec<Declaration> = Vec::new();
        for (prefix, namespace) in utilized {
            let output_prefix = self.output_prefix(prefix, namespace);
            if rendered.get(&output_prefix).map(String::as_str) == Some(namespace.as_str()) {
                continue;
            }
            if declarations
                .iter()
                .any(|existing| existing.prefix == output_prefix)
            {
                continue;
            }
            declarations.push(Declaration {
                prefix: output_prefix,
                namespace: namespace.clone(),
            });
        }
        declarations.sort();
        declarations
    }

    /// The prefix a namespace name is written with in the output.
    fn output_prefix(&self, prefix: &str, namespace: &str) -> String {
        match self.parameters.prefix_rewrite() {
            PrefixRewrite::None => prefix.to_owned(),
            PrefixRewrite::Sequential => self
                .assigned
                .get(namespace)
                .cloned()
                .unwrap_or_else(|| prefix.to_owned()),
        }
    }

    fn element_name(&self, element: NodeId) -> C14nResult<String> {
        let data = self
            .document
            .element(element)
            .ok_or(C14nError::NotAnElement {
                node: element,
                kind: self.document.kind(element).unwrap_or(NodeKind::Document),
            })?;
        match self.parameters.prefix_rewrite() {
            PrefixRewrite::None => Ok(data.qualified_name()),
            PrefixRewrite::Sequential => {
                let namespace = data.namespace_uri().unwrap_or_default();
                match self.assigned.get(namespace) {
                    Some(prefix) => Ok(format!("{prefix}:{}", data.local_name())),
                    None => Ok(data.qualified_name()),
                }
            }
        }
    }

    /// §2.6: sort the attributes by namespace name and local name, rewrite the
    /// prefixes, and rewrite `QName`-valued content.
    fn write_attributes(&mut self, element: NodeId) -> C14nResult<()> {
        let (element_namespace, element_local) = match self.document.element(element) {
            Some(data) => (
                data.namespace_uri().unwrap_or_default().to_owned(),
                data.local_name().to_owned(),
            ),
            None => (String::new(), String::new()),
        };

        let mut attributes: Vec<(String, String, String, String)> = Vec::new();
        for attribute in self.document.attributes(element) {
            if self.subset.is_excluded_attribute(attribute)
                || is_namespace_declaration(self.document, attribute)
            {
                continue;
            }
            let Some(data) = self.document.attr(attribute) else {
                continue;
            };
            let namespace = data.namespace_uri().unwrap_or_default().to_owned();
            let local = data.local_name().to_owned();

            let name = match (data.prefix(), self.parameters.prefix_rewrite()) {
                (None, _) => local.clone(),
                (Some(XML_PREFIX), _) | (Some(_), PrefixRewrite::None) => data.qualified_name(),
                (Some(prefix), PrefixRewrite::Sequential) => match self.assigned.get(&namespace) {
                    Some(rewritten) => format!("{rewritten}:{local}"),
                    None => format!("{prefix}:{local}"),
                },
            };

            let aware = self.parameters.qname_aware();
            let is_qname_valued = if namespace.is_empty() {
                aware.is_qname_unqualified_attribute(&element_namespace, &element_local, &local)
            } else {
                aware.is_qname_qualified_attribute(&namespace, &local)
            };
            let value = if is_qname_valued
                && self.parameters.prefix_rewrite() == PrefixRewrite::Sequential
            {
                self.rewrite_content_qname(data.value(), element)
            } else {
                data.value().to_owned()
            };

            attributes.push((namespace, local, name, value));
        }

        attributes.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        for (_, _, name, value) in &attributes {
            write_attribute(&mut self.out, name, value);
        }
        Ok(())
    }

    /// Rewrites the prefix of a `QName` sitting in content, using the bindings
    /// in force at `element`.
    fn rewrite_content_qname(&self, value: &str, element: NodeId) -> String {
        let bindings = Bindings::at(self.document, element);
        rewrite_qname(value, |prefix| {
            let namespace = if prefix == DEFAULT_PREFIX {
                bindings.default_namespace()
            } else {
                bindings.get(prefix)?
            };
            self.assigned.get(namespace).cloned()
        })
    }

    /// Schedules the children of an element, coalescing adjacent text nodes so
    /// that `TrimTextNodes` and `QName` rewriting see the whole run at once.
    fn schedule_children(&mut self, element: NodeId, frame: &Rc<Frame>, stack: &mut Vec<Task>) {
        let (namespace, local) = match self.document.element(element) {
            Some(data) => (
                data.namespace_uri().unwrap_or_default().to_owned(),
                data.local_name().to_owned(),
            ),
            None => (String::new(), String::new()),
        };
        let aware = self.parameters.qname_aware();
        let rewrite_qname_content = self.parameters.prefix_rewrite() == PrefixRewrite::Sequential
            && aware.is_qname_element(&namespace, &local);
        let rewrite_xpath_content = self.parameters.prefix_rewrite() == PrefixRewrite::Sequential
            && aware.is_xpath_element(&namespace, &local);

        let mut tasks: Vec<Task> = Vec::new();
        let mut pending = String::new();
        let flush = |pending: &mut String, tasks: &mut Vec<Task>| {
            if pending.is_empty() {
                return;
            }
            tasks.push(Task::Text {
                text: std::mem::take(pending),
                frame: Rc::clone(frame),
                element,
                rewrite_qname_content,
                rewrite_xpath_content,
            });
        };
        for child in self.document.children(element).collect::<Vec<_>>() {
            match self.document.kind(child) {
                Some(NodeKind::Text | NodeKind::CdataSection) => {
                    if let Some(text) = self.document.character_data(child) {
                        pending.push_str(text);
                    }
                }
                _ => {
                    flush(&mut pending, &mut tasks);
                    tasks.push(Task::Node {
                        node: child,
                        frame: Rc::clone(frame),
                        position: Position::Inside,
                    });
                }
            }
        }
        flush(&mut pending, &mut tasks);
        stack.extend(tasks.into_iter().rev());
    }

    /// Writes one coalesced run of character data, trimming and rewriting it
    /// with the prefix assignments in force at this point of the document.
    fn write_text(
        &mut self,
        text: &str,
        frame: &Frame,
        element: NodeId,
        rewrite_qname_content: bool,
        rewrite_xpath_content: bool,
    ) {
        let trimmed = if self.parameters.trim_text_nodes() && !frame.preserve_space {
            text.trim_matches(is_xml_whitespace)
        } else {
            text
        };
        if trimmed.is_empty() {
            return;
        }
        let rewritten = if rewrite_qname_content {
            self.rewrite_content_qname(trimmed, element)
        } else if rewrite_xpath_content {
            let bindings = Bindings::at(self.document, element);
            rewrite_xpath_prefixes(trimmed, |prefix| {
                let namespace = bindings.get(prefix)?;
                self.assigned.get(namespace).cloned()
            })
        } else {
            trimmed.to_owned()
        };
        escape_text(&mut self.out, &rewritten);
    }
}

/// The `S` production of [XML 1.0 §2.3](https://www.w3.org/TR/xml/#NT-S).
const fn is_xml_whitespace(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\r' | '\n')
}
