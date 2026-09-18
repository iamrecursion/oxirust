//! Serialization of a [`Document`] back to XML text.
//!
//! The serializer is a non-recursive walk over the arena, so arbitrarily deep
//! documents are safe. It performs the *namespace fixup* of
//! [DOM Level 3 Core Appendix B.1](https://www.w3.org/TR/DOM-Level-3-Core/namespaces-algorithms.html):
//! every element and attribute whose namespace is not already bound to its
//! prefix in the enclosing scope gets a declaration emitted for it, and an
//! attribute that carries a namespace but no prefix gets a generated one. A
//! document built by [`crate::build`] already carries all its declarations as
//! attributes, so nothing is added to it and the output is a faithful copy.

use std::fmt;
use std::io;

use oxixml_xml::XmlVersion;

use crate::document::Document;
use crate::error::SerializeError;
use crate::names::XMLNS_NAMESPACE;
use crate::node::{NodeContent, NodeId, NodeKind};

/// Whether, and how, the XML declaration is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XmlDeclaration {
    /// Do not write a declaration at all.
    #[default]
    Omit,
    /// Write a declaration built from [`Document::xml_version`],
    /// [`Document::encoding`] and [`Document::standalone`], defaulting the
    /// version to `1.0`.
    Include,
}

/// Options controlling [`Document::write_xml`] and its siblings.
///
/// # Example
///
/// ```
/// use oxixml_dom::{SerializeOptions, XmlDeclaration, build};
///
/// let doc = build::parse_str("<a><b/><b>text</b></a>")?;
/// let options = SerializeOptions::pretty()
///     .with_xml_declaration(XmlDeclaration::Include);
/// assert_eq!(
///     doc.to_xml_with(&options),
///     "<?xml version=\"1.0\"?>\n<a>\n  <b/>\n  <b>text</b>\n</a>",
/// );
/// # Ok::<(), oxixml_dom::BuildError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializeOptions {
    indent: Option<String>,
    newline: String,
    declaration: XmlDeclaration,
    fold_empty_elements: bool,
    namespace_fixup: bool,
}

impl Default for SerializeOptions {
    fn default() -> Self {
        Self {
            indent: None,
            newline: "\n".to_owned(),
            declaration: XmlDeclaration::Omit,
            fold_empty_elements: true,
            namespace_fixup: true,
        }
    }
}

impl SerializeOptions {
    /// Returns the compact defaults: no indentation, no XML declaration, empty
    /// elements folded to `<a/>` and namespace fixup enabled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns options that indent the output with two spaces per level.
    ///
    /// Only element-only content is indented: an element that has at least one
    /// text, CDATA or entity-reference child is written on a single line, so no
    /// white space is ever added to or removed from mixed content.
    #[must_use]
    pub fn pretty() -> Self {
        Self::default().with_indent(Some("  "))
    }

    /// Sets the indentation string, or disables indentation with `None`.
    #[must_use]
    pub fn with_indent(mut self, indent: Option<&str>) -> Self {
        self.indent = indent.map(str::to_owned);
        self
    }

    /// Sets the line separator used when indenting. The default is `"\n"`.
    #[must_use]
    pub fn with_newline(mut self, newline: &str) -> Self {
        self.newline = newline.to_owned();
        self
    }

    /// Selects whether an XML declaration is written.
    #[must_use]
    pub const fn with_xml_declaration(mut self, declaration: XmlDeclaration) -> Self {
        self.declaration = declaration;
        self
    }

    /// Selects whether a childless element is written as `<a/>` (the default) or
    /// as `<a></a>`.
    #[must_use]
    pub const fn with_fold_empty_elements(mut self, fold: bool) -> Self {
        self.fold_empty_elements = fold;
        self
    }

    /// Selects whether missing namespace declarations are synthesized.
    ///
    /// Turning this off writes the attribute list verbatim, which is what a
    /// caller who maintains the declarations itself wants.
    #[must_use]
    pub const fn with_namespace_fixup(mut self, fixup: bool) -> Self {
        self.namespace_fixup = fixup;
        self
    }

    const fn is_pretty(&self) -> bool {
        self.indent.is_some()
    }
}

/// A destination for serialized text.
trait Sink {
    fn push(&mut self, text: &str) -> Result<(), SerializeError>;
}

struct FmtSink<W>(W);

impl<W: fmt::Write> Sink for FmtSink<W> {
    fn push(&mut self, text: &str) -> Result<(), SerializeError> {
        self.0.write_str(text).map_err(SerializeError::Fmt)
    }
}

struct IoSink<W>(W);

impl<W: io::Write> Sink for IoSink<W> {
    fn push(&mut self, text: &str) -> Result<(), SerializeError> {
        self.0
            .write_all(text.as_bytes())
            .map_err(SerializeError::Io)
    }
}

impl Document {
    /// Serializes the whole document with the default options and returns it as
    /// a `String`.
    #[must_use]
    pub fn to_xml(&self) -> String {
        self.to_xml_with(&SerializeOptions::default())
    }

    /// Serializes the whole document with the given options and returns it as a
    /// `String`.
    #[must_use]
    pub fn to_xml_with(&self, options: &SerializeOptions) -> String {
        let mut out = String::new();
        // Writing into a `String` cannot fail, and the node identifier is the
        // document node, which always exists.
        let _ = self.write_xml(&mut out, options);
        out
    }

    /// Serializes the subtree rooted at `id` with the given options and returns
    /// it as a `String`. Returns an empty string for an unknown identifier.
    #[must_use]
    pub fn node_to_xml(&self, id: NodeId, options: &SerializeOptions) -> String {
        let mut out = String::new();
        let _ = self.write_node(id, &mut out, options);
        out
    }

    /// Writes the whole document to a [`std::fmt::Write`] sink.
    ///
    /// # Errors
    ///
    /// Returns [`SerializeError::Fmt`] when the sink fails.
    pub fn write_xml<W: fmt::Write>(
        &self,
        writer: W,
        options: &SerializeOptions,
    ) -> Result<(), SerializeError> {
        self.serialize(self.root(), &mut FmtSink(writer), options)
    }

    /// Writes the whole document to a [`std::io::Write`] sink, encoded as UTF-8.
    ///
    /// # Errors
    ///
    /// Returns [`SerializeError::Io`] when the sink fails.
    pub fn write_xml_io<W: io::Write>(
        &self,
        writer: W,
        options: &SerializeOptions,
    ) -> Result<(), SerializeError> {
        self.serialize(self.root(), &mut IoSink(writer), options)
    }

    /// Writes the subtree rooted at `id` to a [`std::fmt::Write`] sink.
    ///
    /// # Errors
    ///
    /// Returns [`SerializeError::UnknownNode`] when the identifier is unknown
    /// and [`SerializeError::Fmt`] when the sink fails.
    pub fn write_node<W: fmt::Write>(
        &self,
        id: NodeId,
        writer: W,
        options: &SerializeOptions,
    ) -> Result<(), SerializeError> {
        self.serialize(id, &mut FmtSink(writer), options)
    }

    /// Writes the subtree rooted at `id` to a [`std::io::Write`] sink, encoded
    /// as UTF-8.
    ///
    /// # Errors
    ///
    /// Returns [`SerializeError::UnknownNode`] when the identifier is unknown
    /// and [`SerializeError::Io`] when the sink fails.
    pub fn write_node_io<W: io::Write>(
        &self,
        id: NodeId,
        writer: W,
        options: &SerializeOptions,
    ) -> Result<(), SerializeError> {
        self.serialize(id, &mut IoSink(writer), options)
    }

    fn serialize<S: Sink>(
        &self,
        root: NodeId,
        sink: &mut S,
        options: &SerializeOptions,
    ) -> Result<(), SerializeError> {
        if !self.contains(root) {
            return Err(SerializeError::UnknownNode(root));
        }
        let mut state = Serializer {
            doc: self,
            options,
            scopes: NamespaceScopes::new(),
            generated: 0,
            wrote_anything: false,
        };
        // Seed the scope stack with the declarations that are in force at the
        // subtree root, so that serializing a fragment does not re-declare them.
        state.seed_scopes(root);
        state.run(root, sink)
    }
}

impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_xml(f, &SerializeOptions::default())
            .map_err(|_| fmt::Error)
    }
}

/// One step of the iterative walk.
enum Step {
    Open {
        id: NodeId,
        depth: usize,
        indented: bool,
    },
    Close {
        id: NodeId,
        depth: usize,
        indented: bool,
    },
}

struct Serializer<'a> {
    doc: &'a Document,
    options: &'a SerializeOptions,
    scopes: NamespaceScopes,
    generated: usize,
    wrote_anything: bool,
}

impl Serializer<'_> {
    fn seed_scopes(&mut self, root: NodeId) {
        let mut chain: Vec<NodeId> = self.doc.ancestors(root).collect();
        chain.reverse();
        for element in chain {
            self.scopes.push();
            for (prefix, namespace) in self.doc.namespace_declarations(element) {
                self.scopes
                    .declare(prefix.map(str::to_owned), namespace.to_owned());
            }
        }
    }

    fn run<S: Sink>(&mut self, root: NodeId, sink: &mut S) -> Result<(), SerializeError> {
        // [XML 1.1 §2.8](https://www.w3.org/TR/xml11/#sec-prolog-dtd) requires a
        // 1.1 document to *begin* with a declaration naming the version. It is
        // therefore not optional here: without it the output would re-parse as
        // XML 1.0, where the character references this serializer writes for the
        // restricted characters are not references at all.
        let required = matches!(self.doc.char_version(), XmlVersion::Explicit1_1);
        if root == self.doc.root()
            && (required || self.options.declaration == XmlDeclaration::Include)
        {
            self.write_declaration(sink)?;
        }

        let mut stack = vec![Step::Open {
            id: root,
            depth: 0,
            indented: false,
        }];
        while let Some(step) = stack.pop() {
            match step {
                Step::Open {
                    id,
                    depth,
                    indented,
                } => {
                    self.open(id, depth, indented, &mut stack, sink)?;
                }
                Step::Close {
                    id,
                    depth,
                    indented,
                } => {
                    if indented {
                        self.write_break(depth, sink)?;
                    }
                    self.scopes.pop();
                    sink.push("</")?;
                    if let Some(data) = self.doc.element(id) {
                        sink.push(&data.qualified_name())?;
                    }
                    sink.push(">")?;
                }
            }
        }
        Ok(())
    }

    fn write_declaration<S: Sink>(&mut self, sink: &mut S) -> Result<(), SerializeError> {
        sink.push("<?xml version=\"")?;
        sink.push(
            self.doc
                .xml_version()
                .unwrap_or(match self.doc.char_version() {
                    XmlVersion::Explicit1_1 => "1.1",
                    _ => "1.0",
                }),
        )?;
        sink.push("\"")?;
        if let Some(encoding) = self.doc.encoding() {
            sink.push(" encoding=\"")?;
            sink.push(encoding)?;
            sink.push("\"")?;
        }
        if let Some(standalone) = self.doc.standalone() {
            sink.push(" standalone=\"")?;
            sink.push(if standalone { "yes" } else { "no" })?;
            sink.push("\"")?;
        }
        sink.push("?>")?;
        // The newline is written here rather than left to `write_break` so that
        // a declaration is followed by one even when indentation is off, and by
        // exactly one when it is on.
        sink.push(&self.options.newline)?;
        Ok(())
    }

    fn write_break<S: Sink>(&mut self, depth: usize, sink: &mut S) -> Result<(), SerializeError> {
        let Some(indent) = self.options.indent.as_deref() else {
            return Ok(());
        };
        if self.wrote_anything {
            sink.push(&self.options.newline)?;
        }
        for _ in 0..depth {
            sink.push(indent)?;
        }
        Ok(())
    }

    fn open<S: Sink>(
        &mut self,
        id: NodeId,
        depth: usize,
        indented: bool,
        stack: &mut Vec<Step>,
        sink: &mut S,
    ) -> Result<(), SerializeError> {
        let Some(node) = self.doc.get(id) else {
            return Err(SerializeError::UnknownNode(id));
        };
        if indented {
            self.write_break(depth, sink)?;
        }

        match &node.content {
            NodeContent::Document | NodeContent::DocumentFragment => {
                let children: Vec<NodeId> = self.doc.children(id).collect();
                let indent_children = self.options.is_pretty();
                for child in children.into_iter().rev() {
                    stack.push(Step::Open {
                        id: child,
                        depth,
                        indented: indent_children,
                    });
                }
                return Ok(());
            }
            NodeContent::Text(text) => {
                write_escaped_text(sink, text, self.doc.char_version())?;
            }
            NodeContent::CdataSection(text) => {
                sink.push("<![CDATA[")?;
                sink.push(text)?;
                sink.push("]]>")?;
            }
            NodeContent::Comment(text) => {
                sink.push("<!--")?;
                sink.push(text)?;
                sink.push("-->")?;
            }
            NodeContent::ProcessingInstruction(data) => {
                sink.push("<?")?;
                sink.push(data.target())?;
                if !data.data().is_empty() {
                    sink.push(" ")?;
                    sink.push(data.data())?;
                }
                sink.push("?>")?;
            }
            NodeContent::EntityReference(name) => {
                sink.push("&")?;
                sink.push(name)?;
                sink.push(";")?;
            }
            NodeContent::DocumentType(data) => {
                sink.push("<!DOCTYPE ")?;
                sink.push(data.name())?;
                match (data.public_id(), data.system_id()) {
                    (Some(public), Some(system)) => {
                        sink.push(" PUBLIC \"")?;
                        sink.push(public)?;
                        sink.push("\" \"")?;
                        sink.push(system)?;
                        sink.push("\"")?;
                    }
                    (None, Some(system)) => {
                        sink.push(" SYSTEM \"")?;
                        sink.push(system)?;
                        sink.push("\"")?;
                    }
                    (Some(public), None) => {
                        sink.push(" PUBLIC \"")?;
                        sink.push(public)?;
                        sink.push("\"")?;
                    }
                    (None, None) => {}
                }
                if let Some(subset) = data.internal_subset() {
                    sink.push(" [")?;
                    sink.push(subset)?;
                    sink.push("]")?;
                }
                sink.push(">")?;
            }
            NodeContent::Attr(_) => {
                // A lone attribute node is serialized as `name="value"`, which is
                // what a caller asking for one explicitly expects.
                if let Some(data) = self.doc.attr(id) {
                    sink.push(&data.qualified_name())?;
                    sink.push("=\"")?;
                    write_escaped_attribute(sink, data.value(), self.doc.char_version())?;
                    sink.push("\"")?;
                }
            }
            NodeContent::Element(_) => {
                return self.open_element(id, depth, stack, sink);
            }
        }
        self.wrote_anything = true;
        Ok(())
    }

    fn open_element<S: Sink>(
        &mut self,
        id: NodeId,
        depth: usize,
        stack: &mut Vec<Step>,
        sink: &mut S,
    ) -> Result<(), SerializeError> {
        self.scopes.push();
        for (prefix, namespace) in self.doc.namespace_declarations(id) {
            self.scopes
                .declare(prefix.map(str::to_owned), namespace.to_owned());
        }
        // A declaration added through the DOM Level 1 `setAttribute` API carries
        // no `namespaceURI`, so `namespace_declarations` — which filters on the
        // XMLNS namespace — cannot see it. It is still written out verbatim
        // below, so the fixup has to honour it or it would emit a second,
        // duplicate `xmlns:…` attribute and produce non-well-formed output.
        let declaration_ids: Vec<NodeId> = self.doc.attributes(id).collect();
        for attribute in declaration_ids {
            let Some(data) = self.doc.attr(attribute) else {
                continue;
            };
            if data.namespace_uri() == Some(XMLNS_NAMESPACE) {
                continue;
            }
            let qualified_name = data.qualified_name();
            let prefix = if qualified_name == "xmlns" {
                None
            } else if let Some(prefix) = qualified_name.strip_prefix("xmlns:") {
                Some(prefix.to_owned())
            } else {
                continue;
            };
            self.scopes.declare(prefix, data.value().to_owned());
        }

        let mut attributes: Vec<(String, String)> = Vec::new();
        let mut generated: Vec<(Option<String>, String)> = Vec::new();

        let element_name = {
            let Some(data) = self.doc.element(id) else {
                return Err(SerializeError::UnknownNode(id));
            };
            let prefix = data.prefix().map(str::to_owned);
            let namespace = data.namespace_uri().map(str::to_owned);
            if self.options.namespace_fixup
                && self.scopes.resolve(prefix.as_deref()) != namespace.as_deref()
            {
                generated.push((prefix.clone(), namespace.clone().unwrap_or_default()));
                self.scopes
                    .declare(prefix.clone(), namespace.unwrap_or_default());
            }
            data.qualified_name()
        };

        let attribute_ids: Vec<NodeId> = self.doc.attributes(id).collect();
        for attribute in attribute_ids {
            let Some(data) = self.doc.attr(attribute) else {
                continue;
            };
            let value = data.value().to_owned();
            if data.namespace_uri() == Some(XMLNS_NAMESPACE) {
                attributes.push((data.qualified_name(), value));
                continue;
            }
            let Some(namespace) = data.namespace_uri().map(str::to_owned) else {
                attributes.push((data.qualified_name(), value));
                continue;
            };
            let local = data.local_name().to_owned();
            let declared_prefix = data.prefix().map(str::to_owned);

            if !self.options.namespace_fixup {
                attributes.push((
                    crate::node::qualify(declared_prefix.as_deref(), &local),
                    value,
                ));
                continue;
            }

            // DOM Level 3 Core Appendix B.1: an attribute whose declared prefix
            // is already bound to a *different* namespace must be given a freshly
            // generated prefix. Re-binding the declared prefix instead would
            // change the namespace of the element and of every other name using
            // that prefix (and duplicate `xmlns:p` when the element declares it).
            let prefix = match declared_prefix {
                Some(prefix) if self.scopes.resolve(Some(&prefix)) == Some(&namespace) => prefix,
                Some(_) | None => match self.scopes.prefix_for(&namespace) {
                    Some(existing) => existing.to_owned(),
                    None => {
                        let prefix = self.next_generated_prefix();
                        generated.push((Some(prefix.clone()), namespace.clone()));
                        self.scopes.declare(Some(prefix.clone()), namespace);
                        prefix
                    }
                },
            };
            attributes.push((crate::node::qualify(Some(&prefix), &local), value));
        }

        sink.push("<")?;
        sink.push(&element_name)?;
        for (prefix, namespace) in &generated {
            sink.push(" ")?;
            match prefix {
                Some(prefix) => {
                    sink.push("xmlns:")?;
                    sink.push(prefix)?;
                }
                None => sink.push("xmlns")?,
            }
            sink.push("=\"")?;
            write_escaped_attribute(sink, namespace, self.doc.char_version())?;
            sink.push("\"")?;
        }
        for (name, value) in &attributes {
            sink.push(" ")?;
            sink.push(name)?;
            sink.push("=\"")?;
            write_escaped_attribute(sink, value, self.doc.char_version())?;
            sink.push("\"")?;
        }

        let children: Vec<NodeId> = self.doc.children(id).collect();
        if children.is_empty() && self.options.fold_empty_elements {
            sink.push("/>")?;
            self.scopes.pop();
            self.wrote_anything = true;
            return Ok(());
        }
        sink.push(">")?;
        self.wrote_anything = true;

        let indent_children = self.options.is_pretty()
            && !children.is_empty()
            && children.iter().all(|&child| {
                !matches!(
                    self.doc.kind(child),
                    Some(NodeKind::Text | NodeKind::CdataSection | NodeKind::EntityReference)
                )
            });

        stack.push(Step::Close {
            id,
            depth,
            indented: indent_children,
        });
        for child in children.into_iter().rev() {
            stack.push(Step::Open {
                id: child,
                depth: depth + 1,
                indented: indent_children,
            });
        }
        Ok(())
    }

    fn next_generated_prefix(&mut self) -> String {
        loop {
            self.generated += 1;
            let candidate = format!("ns{}", self.generated);
            if self.scopes.resolve(Some(&candidate)).is_none() {
                return candidate;
            }
        }
    }
}

/// The stack of namespace declarations in force while serializing.
struct NamespaceScopes {
    bindings: Vec<(Option<String>, String)>,
    levels: Vec<usize>,
}

impl NamespaceScopes {
    /// The stack starts with the two bindings that
    /// [Namespaces in XML §3](https://www.w3.org/TR/xml-names/#ns-decl) declares
    /// to be permanently in force, so that `xml:` and `xmlns:` names never
    /// provoke a synthesized declaration.
    fn new() -> Self {
        Self {
            bindings: vec![
                (
                    Some("xml".to_owned()),
                    crate::names::XML_NAMESPACE.to_owned(),
                ),
                (Some("xmlns".to_owned()), XMLNS_NAMESPACE.to_owned()),
            ],
            levels: Vec::new(),
        }
    }

    fn push(&mut self) {
        self.levels.push(self.bindings.len());
    }

    fn pop(&mut self) {
        if let Some(mark) = self.levels.pop() {
            self.bindings.truncate(mark);
        }
    }

    fn declare(&mut self, prefix: Option<String>, namespace: String) {
        self.bindings.push((prefix, namespace));
    }

    /// Resolves a prefix to the namespace it is bound to, treating an empty
    /// binding (`xmlns=""`) as "no namespace".
    fn resolve(&self, prefix: Option<&str>) -> Option<&str> {
        self.bindings
            .iter()
            .rev()
            .find(|(declared, _)| declared.as_deref() == prefix)
            .and_then(|(_, namespace)| {
                if namespace.is_empty() {
                    None
                } else {
                    Some(namespace.as_str())
                }
            })
    }

    /// Returns a non-default prefix currently bound to `namespace`, if any.
    fn prefix_for(&self, namespace: &str) -> Option<&str> {
        for (declared, declared_namespace) in self.bindings.iter().rev() {
            let Some(prefix) = declared.as_deref() else {
                continue;
            };
            if declared_namespace == namespace && self.resolve(Some(prefix)) == Some(namespace) {
                return Some(prefix);
            }
        }
        None
    }
}

fn write_escaped_text<S: Sink>(
    sink: &mut S,
    text: &str,
    version: XmlVersion,
) -> Result<(), SerializeError> {
    let mut last = 0;
    for (index, ch) in text.char_indices() {
        let replacement = match ch {
            '&' => "&amp;",
            '<' => "&lt;",
            '>' => "&gt;",
            '\r' => "&#13;",
            _ if needs_numeric_reference(ch, version) => {
                sink.push(&text[last..index])?;
                write_numeric_reference(sink, ch)?;
                last = index + ch.len_utf8();
                continue;
            }
            _ => continue,
        };
        sink.push(&text[last..index])?;
        sink.push(replacement)?;
        last = index + ch.len_utf8();
    }
    sink.push(&text[last..])
}

fn write_escaped_attribute<S: Sink>(
    sink: &mut S,
    value: &str,
    version: XmlVersion,
) -> Result<(), SerializeError> {
    let mut last = 0;
    for (index, ch) in value.char_indices() {
        let replacement = match ch {
            '&' => "&amp;",
            '<' => "&lt;",
            '>' => "&gt;",
            '"' => "&quot;",
            '\t' => "&#9;",
            '\n' => "&#10;",
            '\r' => "&#13;",
            _ if needs_numeric_reference(ch, version) => {
                sink.push(&value[last..index])?;
                write_numeric_reference(sink, ch)?;
                last = index + ch.len_utf8();
                continue;
            }
            _ => continue,
        };
        sink.push(&value[last..index])?;
        sink.push(replacement)?;
        last = index + ch.len_utf8();
    }
    sink.push(&value[last..])
}

/// Whether `ch` has to be written as a numeric character reference for the
/// result to be a well-formed document of `version`.
///
/// Under XML 1.0 nothing beyond the delimiters and `#xD` needs one. Under XML
/// 1.1 the `RestrictedChar`s do — writing one literally is a fatal error — and
/// so do `#x85` and `#x2028`, which are not restricted but *are* line ends:
/// left literal they would come back from a re-parse as `#xA`.
const fn needs_numeric_reference(ch: char, version: XmlVersion) -> bool {
    matches!(version, XmlVersion::Explicit1_1)
        && (crate::names::is_xml11_restricted_char(ch) || matches!(ch, '\u{85}' | '\u{2028}'))
}

fn write_numeric_reference<S: Sink>(sink: &mut S, ch: char) -> Result<(), SerializeError> {
    sink.push("&#x")?;
    sink.push(&format!("{:X}", u32::from(ch)))?;
    sink.push(";")
}
