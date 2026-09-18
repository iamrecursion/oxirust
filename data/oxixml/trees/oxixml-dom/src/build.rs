//! Building a [`Document`] from an XML event stream.
//!
//! [`parse_str`] and [`parse_reader`] drive an [`oxixml_xml::reader::NsReader`]
//! and turn its events into a tree; [`DocumentBuilder`] exposes the same state
//! machine for callers that already have an event source of their own (a SAX
//! handler, an XSLT result-tree fragment, a hand-written generator).
//!
//! # Namespaces
//!
//! Prefixes are resolved with the semantics of `NsReader`, which are those of
//! [Namespaces in XML 1.0](https://www.w3.org/TR/xml-names/): an unprefixed
//! *element* name takes the default namespace, an unprefixed *attribute* name
//! never does, and the `xml` and `xmlns` prefixes are permanently bound. The
//! `xmlns` and `xmlns:*` attributes themselves are kept in the tree as ordinary
//! attribute nodes in the `http://www.w3.org/2000/xmlns/` namespace, so
//! serializing the document reproduces its declarations verbatim.
//!
//! # Entities
//!
//! The reader reports every `&…;` as its own event. Character references and the
//! five predefined entities are always replaced by the characters they stand
//! for. What happens to the rest is decided by [`EntityPolicy`]: an entity
//! declared in the internal subset is replaced by its replacement text, with the
//! character and general entity references *inside* that text resolved in turn,
//! following the two-phase rules of
//! [XML 1.0 §4.4](https://www.w3.org/TR/xml/#entproc): a character reference in
//! an `EntityValue` is resolved where the entity is *declared*, a general entity
//! reference where it is *used*. A recursive entity is rejected with
//! [`BuildError::EntityExpansion`] rather than followed, as is one whose
//! expansion would exceed the depth, count or size limits that bound a
//! "billion laughs" declaration. An entity whose replacement text contains
//! markup cannot be spliced into the tree as character data, so — like an
//! undeclared one — it is kept as an
//! [`EntityReference`](crate::NodeKind::EntityReference) node.
//!
//! # Example
//!
//! ```
//! use oxixml_dom::{NodeKind, build};
//!
//! let doc = build::parse_str(
//!     r#"<?xml version="1.0"?>
//!        <r:root xmlns:r="urn:example:r" xml:id="top">
//!          <r:item>a &amp; b</r:item>
//!          <![CDATA[<raw>]]>
//!        </r:root>"#,
//! )?;
//!
//! let root = doc.document_element().expect("a document element");
//! assert_eq!(doc.element(root).map(|e| e.namespace_uri()), Some(Some("urn:example:r")));
//! assert_eq!(doc.get_element_by_id("top"), Some(root));
//!
//! let item = doc
//!     .get_elements_by_tag_name_ns(root, Some("urn:example:r"), "item")
//!     .next()
//!     .expect("one item");
//! assert_eq!(doc.text_content(item).as_deref(), Some("a & b"));
//! # Ok::<(), oxixml_dom::BuildError>(())
//! ```

use std::collections::HashMap;
use std::io::BufRead;

use oxixml_xml::XmlVersion;
use oxixml_xml::encoding::EncodingError;
use oxixml_xml::escape::EscapeError;
use oxixml_xml::events::Event;
use oxixml_xml::events::attributes::Attribute;
use oxixml_xml::name::ResolveResult;
use oxixml_xml::reader::{DEFAULT_MAX_ELEMENT_DEPTH, NsReader};

use crate::document::Document;
use crate::dtd::{Declaration, parse_doctype, scan_internal_subset};
use crate::entity::{ExpandError, Expander, Limit, MAX_ENTITY_DEPTH};
use crate::error::{BuildError, DomError};
use crate::external::{ExternalEntity, ExternalEntityResolver};
use crate::names::{XMLNS_NAMESPACE, first_restricted_char};
use crate::node::NodeId;

/// The maximum depth to which an external parsed general entity may reference
/// further external entities.
///
/// Each level is a separate retrieval, so this bounds both the recursion and the
/// number of files one document can pull in through a single reference.
pub const MAX_EXTERNAL_ENTITY_DEPTH: usize = 16;

/// What to do with a general entity reference that is not one of the five
/// predefined ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EntityPolicy {
    /// Replace the reference when the entity was declared in the internal DTD
    /// subset, and keep it as an
    /// [`EntityReference`](crate::NodeKind::EntityReference) node otherwise.
    #[default]
    Resolve,
    /// Always keep the reference as an
    /// [`EntityReference`](crate::NodeKind::EntityReference) node, even when the
    /// entity is declared.
    Keep,
    /// Fail with [`BuildError::UnknownEntity`] when the entity cannot be
    /// resolved.
    Strict,
}

/// Options controlling how an event stream is turned into a tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuilderOptions {
    entity_policy: EntityPolicy,
    keep_comments: bool,
    keep_processing_instructions: bool,
    cdata_as_text: bool,
    check_well_formedness: bool,
    xml_version: Option<XmlVersion>,
    max_element_depth: usize,
}

impl Default for BuilderOptions {
    fn default() -> Self {
        Self {
            entity_policy: EntityPolicy::Resolve,
            keep_comments: true,
            keep_processing_instructions: true,
            cdata_as_text: false,
            check_well_formedness: false,
            xml_version: None,
            max_element_depth: DEFAULT_MAX_ELEMENT_DEPTH,
        }
    }
}

impl BuilderOptions {
    /// Returns the defaults: comments, processing instructions and CDATA
    /// sections are all preserved, and entities are resolved when they can be.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects what happens to a general entity reference.
    #[must_use]
    pub const fn with_entity_policy(mut self, policy: EntityPolicy) -> Self {
        self.entity_policy = policy;
        self
    }

    /// Selects whether comment nodes are kept.
    #[must_use]
    pub const fn with_comments(mut self, keep: bool) -> Self {
        self.keep_comments = keep;
        self
    }

    /// Selects whether processing-instruction nodes are kept.
    #[must_use]
    pub const fn with_processing_instructions(mut self, keep: bool) -> Self {
        self.keep_processing_instructions = keep;
        self
    }

    /// Selects whether a CDATA section becomes a
    /// [`CdataSection`](crate::NodeKind::CdataSection) node (the default) or is
    /// merged into the surrounding text, which is what the DOM Level 3
    /// `cdata-sections=false` configuration parameter does.
    #[must_use]
    pub const fn with_cdata_as_text(mut self, as_text: bool) -> Self {
        self.cdata_as_text = as_text;
        self
    }

    /// Turns on the full well-formedness checking of
    /// [`oxixml_xml::reader::Config::check_well_formedness`], which makes the
    /// reader a conforming non-validating XML processor instead of the
    /// deliberately permissive `quick-xml` compatible one.
    #[must_use]
    pub const fn with_well_formedness_checks(mut self, check: bool) -> Self {
        self.check_well_formedness = check;
        self
    }

    /// Forces the XML version used for attribute-value normalization instead of
    /// taking it from the XML declaration.
    #[must_use]
    pub const fn with_xml_version(mut self, version: Option<XmlVersion>) -> Self {
        self.xml_version = version;
        self
    }

    /// Sets the maximum element nesting depth the underlying
    /// [`NsReader`] accepts, which defaults to
    /// [`DEFAULT_MAX_ELEMENT_DEPTH`]. Raise it to build trees from documents
    /// nested deeper than that; the tree builder itself is bounded only by
    /// memory.
    #[must_use]
    pub const fn with_max_element_depth(mut self, depth: usize) -> Self {
        self.max_element_depth = depth;
        self
    }

    /// Returns the configured maximum element nesting depth.
    #[must_use]
    pub const fn max_element_depth(&self) -> usize {
        self.max_element_depth
    }
}

/// Parses a whole document held in memory.
///
/// # Errors
///
/// Returns [`BuildError::Xml`] when the text is not well-formed XML, and the
/// other [`BuildError`] variants when the event stream cannot be turned into a
/// tree.
pub fn parse_str(text: &str) -> Result<Document, BuildError> {
    parse_str_with(text, &BuilderOptions::default())
}

/// Parses a whole document held in memory, with explicit options.
///
/// # Errors
///
/// See [`parse_str`].
pub fn parse_str_with(text: &str, options: &BuilderOptions) -> Result<Document, BuildError> {
    parse_str_inner(text, options, None)
}

/// Parses a whole document held in memory, retrieving external parsed general
/// entities through `resolver`.
///
/// Without a resolver — which is what every other entry point uses — an `&name;`
/// declared as `<!ENTITY name SYSTEM "…">` is left in the tree as an
/// [`EntityReference`](crate::NodeKind::EntityReference) node, since a
/// non-validating processor need not retrieve it. With one, the entity's content
/// is parsed and spliced into the tree in place of the reference; see
/// [`crate::external`].
///
/// # Errors
///
/// See [`parse_str`]. A resolver that fails is reported as
/// [`BuildError::EntityExpansion`]; a resolver that returns `Ok(None)` leaves
/// the reference alone.
pub fn parse_str_with_resolver(
    text: &str,
    options: &BuilderOptions,
    resolver: &mut dyn ExternalEntityResolver,
) -> Result<Document, BuildError> {
    parse_str_inner(text, options, Some(resolver))
}

fn parse_str_inner(
    text: &str,
    options: &BuilderOptions,
    resolver: Option<&mut dyn ExternalEntityResolver>,
) -> Result<Document, BuildError> {
    let mut reader = NsReader::from_str(text);
    reader.config_mut().check_well_formedness = options.check_well_formedness;
    reader.config_mut().max_element_depth = options.max_element_depth;
    let mut builder = StreamBuilder::new(options.clone(), resolver);
    loop {
        let event = reader.read_event()?;
        if !builder.handle(&reader, event)? {
            break;
        }
    }
    builder.finish()
}

/// Parses a whole document from a buffered UTF-8 reader.
///
/// # Errors
///
/// See [`parse_str`]; I/O failures are reported as [`BuildError::Xml`].
pub fn parse_reader<R: BufRead>(reader: R) -> Result<Document, BuildError> {
    parse_reader_with(reader, &BuilderOptions::default())
}

/// Parses a whole document from a buffered UTF-8 reader, with explicit options.
///
/// # Errors
///
/// See [`parse_str`].
pub fn parse_reader_with<R: BufRead>(
    reader: R,
    options: &BuilderOptions,
) -> Result<Document, BuildError> {
    parse_reader_inner(reader, options, None)
}

/// Parses a whole document from a buffered UTF-8 reader, retrieving external
/// parsed general entities through `resolver`.
///
/// # Errors
///
/// See [`parse_str_with_resolver`].
pub fn parse_reader_with_resolver<R: BufRead>(
    reader: R,
    options: &BuilderOptions,
    resolver: &mut dyn ExternalEntityResolver,
) -> Result<Document, BuildError> {
    parse_reader_inner(reader, options, Some(resolver))
}

fn parse_reader_inner<R: BufRead>(
    reader: R,
    options: &BuilderOptions,
    resolver: Option<&mut dyn ExternalEntityResolver>,
) -> Result<Document, BuildError> {
    let mut reader = NsReader::from_reader(reader);
    reader.config_mut().check_well_formedness = options.check_well_formedness;
    reader.config_mut().max_element_depth = options.max_element_depth;
    let mut builder = StreamBuilder::new(options.clone(), resolver);
    let mut buffer = Vec::new();
    loop {
        let event = reader.read_event_into(&mut buffer)?;
        if !builder.handle(&reader, event)? {
            break;
        }
        buffer.clear();
    }
    builder.finish()
}

/// The state machine that turns a stream of parser events into a tree.
///
/// `DocumentBuilder` is what [`parse_str`] drives internally; it is public so
/// that a caller with a different event source can build the same tree. Text is
/// coalesced: consecutive [`push_text`](Self::push_text) calls contribute to a
/// single [`Text`](crate::NodeKind::Text) node, which is flushed as soon as any
/// other kind of node is pushed.
///
/// # Example
///
/// ```
/// use oxixml_dom::build::DocumentBuilder;
///
/// let mut builder = DocumentBuilder::new();
/// builder.push_element_start(Some("urn:example"), "e:greeting")?;
/// builder.push_attribute(Some("http://www.w3.org/2000/xmlns/"), "xmlns:e", "urn:example")?;
/// builder.push_text("hello");
/// builder.push_element_end()?;
/// let doc = builder.finish()?;
///
/// assert_eq!(doc.to_xml(), "<e:greeting xmlns:e=\"urn:example\">hello</e:greeting>");
/// # Ok::<(), oxixml_dom::BuildError>(())
/// ```
#[derive(Debug)]
pub struct DocumentBuilder {
    doc: Document,
    stack: Vec<NodeId>,
    pending_text: String,
    options: BuilderOptions,
}

impl Default for DocumentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentBuilder {
    /// Creates a builder with the default options.
    #[must_use]
    pub fn new() -> Self {
        Self::with_options(BuilderOptions::default())
    }

    /// Creates a builder with explicit options.
    #[must_use]
    pub fn with_options(options: BuilderOptions) -> Self {
        let mut doc = Document::new();
        if let Some(version) = options.xml_version {
            // A forced version governs the `Char` production from the very first
            // node, even before — or without — an XML declaration.
            doc.set_char_version(version);
        }
        Self {
            doc,
            stack: Vec::new(),
            pending_text: String::new(),
            options,
        }
    }

    /// Returns the document as it currently stands.
    #[must_use]
    pub const fn document(&self) -> &Document {
        &self.doc
    }

    /// Returns the element currently being built, if any.
    #[must_use]
    pub fn current_element(&self) -> Option<NodeId> {
        self.stack.last().copied()
    }

    /// Records the contents of the XML declaration.
    ///
    /// The declared version normally selects the `Char` production the document
    /// applies; a version forced through
    /// [`BuilderOptions::with_xml_version`] outranks it, so that the declaration
    /// is still reproduced verbatim on serialization while the caller's choice
    /// governs validation.
    pub fn set_declaration(
        &mut self,
        version: Option<&str>,
        encoding: Option<&str>,
        standalone: Option<bool>,
    ) {
        self.doc.set_xml_version(version);
        if let Some(forced) = self.options.xml_version {
            self.doc.set_char_version(forced);
        }
        self.doc.set_encoding(encoding);
        self.doc.set_standalone(standalone);
    }

    /// Appends a document type declaration node to the document.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when the name is not an XML `Name` or when
    /// the document already has a document type declaration.
    pub fn push_doctype(
        &mut self,
        name: &str,
        public_id: Option<&str>,
        system_id: Option<&str>,
        internal_subset: Option<&str>,
    ) -> Result<NodeId, BuildError> {
        self.flush_text()?;
        let node = self.doc.create_document_type(name, public_id, system_id)?;
        if let Some(subset) = internal_subset {
            self.doc.set_internal_subset(node, Some(subset))?;
        }
        let parent = self.parent();
        self.doc.append_child(parent, node)?;
        Ok(node)
    }

    /// Opens an element and makes it the current one.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when the name is invalid or when the element
    /// may not appear where it would be placed (for example a second document
    /// element).
    pub fn push_element_start(
        &mut self,
        namespace: Option<&str>,
        qualified_name: &str,
    ) -> Result<NodeId, BuildError> {
        self.flush_text()?;
        let node = self.doc.create_element_ns(namespace, qualified_name)?;
        let parent = self.parent();
        self.doc.append_child(parent, node)?;
        self.stack.push(node);
        Ok(node)
    }

    /// Adds an attribute to the current element.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when there is no current element or when the
    /// name or value is invalid, and [`BuildError::DuplicateAttribute`] when the
    /// element already carries an attribute with the same expanded name
    /// (Namespaces in XML 1.0 §6.3).
    pub fn push_attribute(
        &mut self,
        namespace: Option<&str>,
        qualified_name: &str,
        value: &str,
    ) -> Result<NodeId, BuildError> {
        let element = self
            .current_element()
            .ok_or(BuildError::Dom(DomError::InvalidState {
                detail: "an attribute was pushed while no element was open",
            }))?;
        let local = local_name_of(qualified_name);
        if self
            .doc
            .get_attribute_node_ns(element, namespace, local)
            .is_some()
        {
            // `set_attribute_ns` would silently overwrite the earlier
            // attribute; a *parsed* document must be rejected instead.
            return Err(BuildError::DuplicateAttribute {
                qualified_name: qualified_name.to_owned(),
            });
        }
        Ok(self
            .doc
            .set_attribute_ns(element, namespace, qualified_name, value)?)
    }

    /// Closes the current element.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when no element is open.
    pub fn push_element_end(&mut self) -> Result<(), BuildError> {
        self.flush_text()?;
        if self.stack.pop().is_none() {
            return Err(BuildError::Dom(DomError::InvalidState {
                detail: "an end tag was pushed while no element was open",
            }));
        }
        Ok(())
    }

    /// Appends character data to the pending text node.
    pub fn push_text(&mut self, text: &str) {
        self.pending_text.push_str(text);
    }

    /// Appends a CDATA section.
    ///
    /// With [`BuilderOptions::with_cdata_as_text`] enabled the content is merged
    /// into the surrounding text instead.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when the content cannot be represented.
    pub fn push_cdata(&mut self, text: &str) -> Result<(), BuildError> {
        if self.options.cdata_as_text {
            self.pending_text.push_str(text);
            return Ok(());
        }
        self.flush_text()?;
        let node = self.doc.create_cdata_section(text)?;
        self.append(node)
    }

    /// Appends a comment node, unless comments have been turned off.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when the content cannot be represented in a
    /// comment.
    pub fn push_comment(&mut self, text: &str) -> Result<(), BuildError> {
        if !self.options.keep_comments {
            return Ok(());
        }
        self.flush_text()?;
        let node = self.doc.create_comment(text)?;
        self.append(node)
    }

    /// Appends a processing instruction, unless they have been turned off.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when the target or the data is invalid.
    pub fn push_processing_instruction(
        &mut self,
        target: &str,
        data: &str,
    ) -> Result<(), BuildError> {
        if !self.options.keep_processing_instructions {
            return Ok(());
        }
        self.flush_text()?;
        let node = self.doc.create_processing_instruction(target, data)?;
        self.append(node)
    }

    /// Appends an unexpanded entity reference node.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when the name is not an XML `Name`.
    pub fn push_entity_reference(&mut self, name: &str) -> Result<(), BuildError> {
        self.flush_text()?;
        let node = self.doc.create_entity_reference(name)?;
        self.append(node)
    }

    /// Registers an `<!ATTLIST … ID …>` declaration, so that
    /// [`Document::get_element_by_id`] honours it.
    pub fn declare_id_attribute(&mut self, element_name: &str, attribute_name: &str) {
        self.doc
            .declare_id_attribute(Some(element_name), attribute_name);
    }

    /// Finishes the build and returns the document.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Dom`] when an element is still open.
    pub fn finish(mut self) -> Result<Document, BuildError> {
        self.flush_text()?;
        if !self.stack.is_empty() {
            return Err(BuildError::Dom(DomError::InvalidState {
                detail: "the document ended while an element was still open",
            }));
        }
        Ok(self.doc)
    }

    fn parent(&self) -> NodeId {
        self.stack
            .last()
            .copied()
            .unwrap_or_else(|| self.doc.root())
    }

    fn append(&mut self, node: NodeId) -> Result<(), BuildError> {
        let parent = self.parent();
        self.doc.append_child(parent, node)?;
        Ok(())
    }

    /// Flushes the pending character data into a text node.
    ///
    /// White space outside the document element is not character data at all —
    /// it is `Misc` in the grammar of
    /// [XML 1.0 §2.1](https://www.w3.org/TR/xml/#sec-well-formed) — so it is
    /// dropped rather than rejected by the document node's child constraints.
    /// *Non*-white-space characters there are a well-formedness violation and
    /// are reported as [`BuildError::ContentOutsideDocumentElement`] rather than
    /// silently discarded.
    fn flush_text(&mut self) -> Result<(), BuildError> {
        if self.pending_text.is_empty() {
            return Ok(());
        }
        let text = std::mem::take(&mut self.pending_text);
        if self.stack.is_empty() {
            if text.trim_matches(is_xml_whitespace).is_empty() {
                return Ok(());
            }
            return Err(BuildError::ContentOutsideDocumentElement { text });
        }
        let node = self.doc.create_text_node(text)?;
        let parent = self.parent();
        self.doc.append_child(parent, node)?;
        Ok(())
    }
}

/// Drives a [`DocumentBuilder`] from [`oxixml_xml`] events.
struct StreamBuilder<'resolver> {
    builder: DocumentBuilder,
    entities: HashMap<String, String>,
    /// `name -> (public id, system id)` for every external *parsed* general
    /// entity declared in the internal subset.
    external_entities: HashMap<String, (Option<String>, String)>,
    /// Supplied by the caller to retrieve those entities; without one they stay
    /// unresolved, which is what a non-validating processor is allowed to do.
    resolver: Option<&'resolver mut dyn ExternalEntityResolver>,
    /// The external entities currently being spliced, innermost last: an entity
    /// that reaches itself again is a well-formedness violation.
    open_external: Vec<String>,
    expander: Expander,
    version: XmlVersion,
    saw_document_element: bool,
}

impl<'resolver> StreamBuilder<'resolver> {
    fn new(
        options: BuilderOptions,
        resolver: Option<&'resolver mut dyn ExternalEntityResolver>,
    ) -> Self {
        let version = options.xml_version.unwrap_or(XmlVersion::Implicit1_0);
        let mut expander = Expander::default();
        expander.set_version(version);
        Self {
            builder: DocumentBuilder::with_options(options),
            entities: HashMap::new(),
            external_entities: HashMap::new(),
            resolver,
            open_external: Vec::new(),
            expander,
            version,
            saw_document_element: false,
        }
    }

    fn finish(self) -> Result<Document, BuildError> {
        if !self.saw_document_element {
            return Err(BuildError::NoDocumentElement);
        }
        self.builder.finish()
    }

    /// Handles one event. Returns `false` once the end of the stream is reached.
    fn handle<R>(&mut self, reader: &NsReader<R>, event: Event<'_>) -> Result<bool, BuildError> {
        match event {
            Event::Eof => return Ok(false),
            Event::Decl(decl) => {
                let version = decode_owned(&decl.version()?)?;
                let encoding = match decl.encoding() {
                    Some(encoding) => {
                        Some(decode_owned(&encoding.map_err(oxixml_xml::Error::from)?)?)
                    }
                    None => None,
                };
                let standalone = match decl.standalone() {
                    Some(value) => {
                        let value = decode_owned(&value.map_err(oxixml_xml::Error::from)?)?;
                        Some(value.eq_ignore_ascii_case("yes"))
                    }
                    None => None,
                };
                if self.builder.options.xml_version.is_none() {
                    self.version = decl.xml_version()?;
                    self.expander.set_version(self.version);
                }
                self.builder
                    .set_declaration(Some(&version), encoding.as_deref(), standalone);
            }
            Event::DocType(text) => {
                let body = decode_text(&text, self.version)?;
                // The prolog is part of the `document` production too, so a
                // restricted character written literally in the internal subset
                // is fatal there as well — and would otherwise reach a text node
                // or an attribute value through an entity's replacement text.
                self.reject_restricted(&body)?;
                let doctype = parse_doctype(&body);
                if let Some(subset) = doctype.internal_subset.as_deref() {
                    for declaration in scan_internal_subset(subset, self.version) {
                        match declaration {
                            Declaration::Entity { name, value } => {
                                // XML 1.0 §4.2: when an entity is declared more
                                // than once, the *first* declaration binds.
                                self.entities.entry(name).or_insert(value);
                            }
                            Declaration::ExternalEntity {
                                name,
                                public_id,
                                system_id,
                            } => {
                                self.external_entities
                                    .entry(name)
                                    .or_insert((public_id, system_id));
                            }
                            Declaration::IdAttribute { element, attribute } => {
                                self.builder.declare_id_attribute(&element, &attribute);
                            }
                        }
                    }
                }
                self.builder.push_doctype(
                    &doctype.name,
                    doctype.public_id.as_deref(),
                    doctype.system_id.as_deref(),
                    doctype.internal_subset.as_deref(),
                )?;
            }
            Event::Start(start) => {
                self.start_element(reader, &start)?;
            }
            Event::Empty(start) => {
                self.start_element(reader, &start)?;
                self.builder.push_element_end()?;
            }
            Event::End(_) => self.builder.push_element_end()?,
            Event::Text(text) => {
                let text = decode_text(&text, self.version)?;
                self.reject_restricted(&text)?;
                self.builder.push_text(&text);
            }
            Event::CData(cdata) => {
                let text = normalize_line_endings(
                    &cdata.decode().map_err(oxixml_xml::Error::from)?,
                    self.version,
                );
                self.reject_restricted(&text)?;
                self.builder.push_cdata(&text)?;
            }
            Event::Comment(comment) => {
                let text = decode_text(&comment, self.version)?;
                self.reject_restricted(&text)?;
                self.builder.push_comment(&text)?;
            }
            Event::PI(pi) => {
                let target = decode_owned(pi.target())?;
                let data = normalize_line_endings(&decode_owned(pi.content())?, self.version);
                self.reject_restricted(&data)?;
                self.builder
                    .push_processing_instruction(&target, data.trim_start())?;
            }
            Event::GeneralRef(reference) => {
                if let Some(character) = reference.resolve_char_ref()? {
                    let mut buffer = [0_u8; 4];
                    self.builder.push_text(character.encode_utf8(&mut buffer));
                    return Ok(true);
                }
                let name = decode_owned(&reference)?;
                self.push_entity(&name)?;
            }
        }
        Ok(true)
    }

    /// Retrieves an external parsed general entity and splices its content into
    /// the tree in place of the reference. Returns whether it did.
    ///
    /// The replacement text is parsed as `content`
    /// ([XML 1.0 §4.3.2](https://www.w3.org/TR/xml/#wf-entities)): it may hold
    /// markup, several top-level elements or none at all. Its element and
    /// attribute names are resolved in the namespace scope of the *reference*,
    /// as [Namespaces in XML](https://www.w3.org/TR/xml-names/) requires, which
    /// is arranged by re-declaring the bindings in force here around it.
    fn splice_external_entity(&mut self, name: &str) -> Result<bool, BuildError> {
        if self.resolver.is_none() {
            return Ok(false);
        }
        if self.open_external.iter().any(|open| open == name) {
            return Err(BuildError::EntityExpansion {
                name: name.to_owned(),
                detail: format!(
                    "the external general entity `&{name};` refers to itself, which XML 1.0 forbids"
                ),
            });
        }
        if self.open_external.len() >= MAX_EXTERNAL_ENTITY_DEPTH {
            return Err(BuildError::EntityExpansion {
                name: name.to_owned(),
                detail: format!(
                    "expanding the external general entity `&{name};` would exceed the maximum \
                     external entity nesting depth of {MAX_EXTERNAL_ENTITY_DEPTH}"
                ),
            });
        }
        let Some((public_id, system_id)) = self.external_entities.get(name) else {
            return Ok(false);
        };
        let request = ExternalEntity {
            name,
            public_id: public_id.as_deref(),
            system_id,
        };
        let resolved = match self.resolver.as_mut() {
            Some(resolver) => resolver.resolve(&request),
            None => return Ok(false),
        };
        let text = match resolved {
            Ok(Some(text)) => text,
            Ok(None) => return Ok(false),
            Err(error) => {
                return Err(BuildError::EntityExpansion {
                    name: name.to_owned(),
                    detail: error.to_string(),
                });
            }
        };
        self.expander
            .charge_external_entity(name, text.len())
            .map_err(|error| BuildError::EntityExpansion {
                name: error.name().to_owned(),
                detail: error.to_string(),
            })?;

        // A byte-order mark is an encoding signature rather than content
        // ([XML 1.0 §4.3.3](https://www.w3.org/TR/xml/#charencoding)), and a
        // `TextDecl` may only appear at the very start of an external entity and
        // is not part of its content either.
        let content = strip_text_declaration(text.strip_prefix('\u{FEFF}').unwrap_or(&text));
        let source = self.wrap_in_scope(content);
        self.open_external.push(name.to_owned());
        let result = self.splice_content(&source);
        self.open_external.pop();
        result.map(|()| true)
    }

    /// Wraps `content` in an element carrying every namespace binding in force
    /// at the current position, so that the sub-parse resolves prefixes exactly
    /// as the surrounding document would.
    fn wrap_in_scope(&self, content: &str) -> String {
        let mut out = String::with_capacity(content.len() + 64);
        out.push_str("<e");
        let mut declared: Vec<Option<String>> = Vec::new();
        let document = self.builder.document();
        let mut chain: Vec<NodeId> = match self.builder.current_element() {
            Some(element) => std::iter::once(element)
                .chain(document.ancestors(element))
                .collect(),
            None => Vec::new(),
        };
        // Innermost first, so the first binding seen for a prefix is the one in
        // force; an outer redeclaration of the same prefix is then skipped.
        chain.dedup();
        for element in chain {
            for (prefix, namespace) in document.namespace_declarations(element) {
                let key = prefix.map(str::to_owned);
                if declared.contains(&key) {
                    continue;
                }
                out.push(' ');
                match &key {
                    Some(prefix) => {
                        out.push_str("xmlns:");
                        out.push_str(prefix);
                    }
                    None => out.push_str("xmlns"),
                }
                out.push_str("=\"");
                push_attribute_escaped(&mut out, namespace);
                out.push('"');
                declared.push(key);
            }
        }
        out.push('>');
        out.push_str(content);
        out.push_str("</e>");
        out
    }

    /// Parses `source` — an element wrapping an entity's content — and appends
    /// everything inside the wrapper to the tree at the current position.
    fn splice_content(&mut self, source: &str) -> Result<(), BuildError> {
        /// Reported for entity text that closes the wrapper this method put
        /// around it, which no `content` production can do.
        const UNBALANCED: BuildError = BuildError::Dom(DomError::InvalidState {
            detail: "the replacement text of an external entity is not balanced content",
        });

        let mut reader = NsReader::from_str(source);
        reader.config_mut().check_well_formedness = self.builder.options.check_well_formedness;
        reader.config_mut().max_element_depth = self.builder.options.max_element_depth;
        // `0` while the wrapper has not been opened (or after it closed), `1`
        // inside it, deeper for each element of the entity's own content.
        let mut depth = 0_usize;
        let mut closed = false;
        loop {
            let event = reader.read_event()?;
            if closed && !matches!(event, Event::Eof) {
                return Err(UNBALANCED);
            }
            match event {
                Event::Eof => break,
                // The wrapper itself is not content: only what it contains is.
                Event::Start(_) if depth == 0 => depth = 1,
                Event::End(_) if depth == 1 => {
                    depth = 0;
                    closed = true;
                }
                Event::Start(_) | Event::Empty(_) => {
                    if !matches!(event, Event::Empty(_)) {
                        depth += 1;
                    }
                    self.handle(&reader, event)?;
                }
                Event::End(_) => {
                    depth = depth.checked_sub(1).ok_or(UNBALANCED)?;
                    self.handle(&reader, event)?;
                }
                // A declaration or a document type declaration inside an
                // entity is not content and must not touch the document's own.
                Event::Decl(_) | Event::DocType(_) => {}
                other if depth == 0 => {
                    // Character data before the wrapper opened cannot happen:
                    // the wrapper is the first thing written.
                    let _ = other;
                    return Err(UNBALANCED);
                }
                other => {
                    self.handle(&reader, other)?;
                }
            }
        }
        if !closed {
            return Err(BuildError::Dom(DomError::InvalidState {
                detail: "an external entity ended while an element was still open",
            }));
        }
        Ok(())
    }

    /// Rejects a `RestrictedChar` that appears *literally* in the document.
    ///
    /// XML 1.1 subtracts those characters from the `document` production —
    /// `document ::= ( prolog element Misc* ) - ( Char* RestrictedChar Char* )`
    /// — so they may only be named by a character reference. The distinction is
    /// one of provenance, which is visible here and nowhere later: by the time
    /// the character data is coalesced into a text node, a `#x7` written as
    /// `&#x7;` and one written literally look the same.
    ///
    /// Under XML 1.0 there are no restricted characters, so this never fires.
    fn reject_restricted(&self, text: &str) -> Result<(), BuildError> {
        match first_restricted_char(text, self.version) {
            Some(character) => Err(BuildError::RestrictedCharacter { character }),
            None => Ok(()),
        }
    }

    fn push_entity(&mut self, name: &str) -> Result<(), BuildError> {
        if let Some(text) = resolve_predefined(name) {
            self.builder.push_text(text);
            return Ok(());
        }
        if self.builder.options.entity_policy == EntityPolicy::Keep {
            return self.builder.push_entity_reference(name);
        }
        if !self.entities.contains_key(name)
            && self.external_entities.contains_key(name)
            && self.splice_external_entity(name)?
        {
            return Ok(());
        }
        match self.expander.expand(name, &self.entities) {
            Ok(text) => {
                self.builder.push_text(&text);
                Ok(())
            }
            // An entity this crate cannot turn into character data — undeclared
            // (possibly external, or declared in an external subset that is never
            // read) or holding markup — keeps its reference in the tree, so that
            // serializing the document writes it back out unchanged.
            Err(ExpandError::Undeclared { .. }) => match self.builder.options.entity_policy {
                EntityPolicy::Strict => Err(BuildError::UnknownEntity {
                    name: name.to_owned(),
                }),
                EntityPolicy::Keep | EntityPolicy::Resolve => {
                    self.builder.push_entity_reference(name)
                }
            },
            Err(error @ ExpandError::Markup { .. }) => match self.builder.options.entity_policy {
                EntityPolicy::Strict => Err(BuildError::EntityExpansion {
                    name: error.name().to_owned(),
                    detail: error.to_string(),
                }),
                EntityPolicy::Keep | EntityPolicy::Resolve => {
                    self.builder.push_entity_reference(name)
                }
            },
            // Recursion is a well-formedness violation and an exhausted budget
            // means the document is trying to expand without bound: neither may
            // be papered over by keeping the reference.
            Err(error) => Err(BuildError::EntityExpansion {
                name: error.name().to_owned(),
                detail: error.to_string(),
            }),
        }
    }

    fn start_element<R>(
        &mut self,
        reader: &NsReader<R>,
        start: &oxixml_xml::events::BytesStart<'_>,
    ) -> Result<(), BuildError> {
        let raw_name = start.name();
        let qualified_name = decode_owned(raw_name.0)?;
        let namespace = resolved_namespace(reader.resolve_element(raw_name).0)?;

        let mut attributes: Vec<(Option<String>, String, String)> = Vec::new();
        for attribute in start.attributes() {
            let attribute: Attribute<'_> = attribute.map_err(oxixml_xml::Error::from)?;
            let name = decode_owned(attribute.key.0)?;
            // The *raw* value is the only place an attribute's provenance is
            // still visible: `&#x7;` is five characters here and one character
            // after normalization. A literal restricted character is a fatal
            // error under XML 1.1 exactly as it is in content.
            if let Ok(raw) = std::str::from_utf8(&attribute.value) {
                self.reject_restricted(raw)?;
            }
            let entities = &self.entities;
            let expander = &mut self.expander;
            // The budget is charged from inside the resolver because
            // `normalized_value_with` walks the nested references itself: it is
            // the only point at which each occurrence — including the ones it
            // reaches recursively — is visible. Refusing to resolve is how a
            // spent budget is signalled, which surfaces as an `Escape` error and
            // is turned back into the right diagnosis below.
            let mut over_budget: Option<String> = None;
            let value = attribute
                .normalized_value_with(self.version, MAX_ENTITY_DEPTH, |entity| {
                    if let Some(text) = resolve_predefined(entity) {
                        return Some(text);
                    }
                    let text = entities.get(entity)?;
                    if over_budget.is_some()
                        || !expander.charge_attribute_expansion(entity, text.len())
                    {
                        over_budget.get_or_insert_with(|| entity.to_owned());
                        return None;
                    }
                    Some(text.as_str())
                })
                .map_err(|error| match error {
                    // The expansion ran past `MAX_ENTITY_DEPTH`, which only a
                    // recursive or pathologically nested declaration does; it is
                    // the same condition `push_entity` reports for content, and
                    // is not an *unknown* entity.
                    oxixml_xml::Error::Escape(EscapeError::TooManyNestedEntities) => {
                        let raw = String::from_utf8_lossy(&attribute.value);
                        let entity = first_general_entity(&raw).unwrap_or_else(|| name.clone());
                        BuildError::EntityExpansion {
                            detail: format!(
                                "expanding the general entity `&{entity};` would exceed {}",
                                Limit::Depth
                            ),
                            name: entity,
                        }
                    }
                    oxixml_xml::Error::Escape(_) => match over_budget.take() {
                        Some(entity) => BuildError::EntityExpansion {
                            detail: format!(
                                "expanding the general entity `&{entity};` would exceed the \
                                 limits on entity expansion"
                            ),
                            name: entity,
                        },
                        None => {
                            let raw = String::from_utf8_lossy(&attribute.value);
                            BuildError::UnknownEntity {
                                name: first_unresolvable_entity(&raw, entities)
                                    .unwrap_or_else(|| name.clone()),
                            }
                        }
                    },
                    other => BuildError::Xml(other),
                })?
                .into_owned();
            let namespace = if attribute.key.as_namespace_binding().is_some() {
                Some(XMLNS_NAMESPACE.to_owned())
            } else {
                resolved_namespace(reader.resolve_attribute(attribute.key).0)?
            };
            attributes.push((namespace, name, value));
        }

        self.builder
            .push_element_start(namespace.as_deref(), &qualified_name)?;
        if self.builder.stack.len() == 1 {
            self.saw_document_element = true;
        }
        for (namespace, name, value) in attributes {
            self.builder
                .push_attribute(namespace.as_deref(), &name, &value)?;
        }
        Ok(())
    }
}

fn resolved_namespace(result: ResolveResult<'_>) -> Result<Option<String>, BuildError> {
    match result {
        ResolveResult::Unbound => Ok(None),
        ResolveResult::Bound(namespace) => Ok(Some(decode_owned(namespace.into_inner())?)),
        ResolveResult::Unknown(prefix) => Err(BuildError::UnboundPrefix {
            prefix: String::from_utf8_lossy(&prefix).into_owned(),
        }),
    }
}

fn decode_owned(bytes: &[u8]) -> Result<String, BuildError> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(text.to_owned()),
        Err(error) => Err(BuildError::Xml(oxixml_xml::Error::from(
            EncodingError::Utf8(error),
        ))),
    }
}

fn decode_text(
    text: &oxixml_xml::events::BytesText<'_>,
    version: XmlVersion,
) -> Result<String, BuildError> {
    Ok(normalize_line_endings(
        &text.decode().map_err(oxixml_xml::Error::from)?,
        version,
    ))
}

/// Returns whether `ch` is one of the four XML 1.0 `S` (white space) characters.
const fn is_xml_whitespace(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\r' | '\n')
}

/// Returns the local part of `qualified_name` (everything after the first `:`).
fn local_name_of(qualified_name: &str) -> &str {
    match qualified_name.split_once(':') {
        Some((_, local)) => local,
        None => qualified_name,
    }
}

/// Applies the [XML 1.0 §2.11](https://www.w3.org/TR/xml/#sec-line-ends)
/// end-of-line normalization: both the two-character sequence `#xD #xA` and any
/// `#xD` not followed by `#xA` are translated to a single `#xA`.
///
/// [XML 1.1 §2.11](https://www.w3.org/TR/xml11/#sec-line-ends) extends the set:
/// `#xD #x85`, a lone `#x85` (NEL) and `#x2028` (LINE SEPARATOR) become `#xA`
/// as well. Neither version touches a character *reference* to one of those code
/// points — normalization happens to the entity's characters before parsing, and
/// a reference is not yet a character at that point.
///
/// This is a parsing-time obligation ("all line breaks MUST have been normalized
/// on input"), so every character-data-bearing event has to pass through here —
/// otherwise the DOM disagrees with the Infoset/XDM tree built from the same
/// bytes, and a later serialization escapes the stray `#xD` permanently.
fn normalize_line_endings(text: &str, version: XmlVersion) -> String {
    let xml11 = matches!(version, XmlVersion::Explicit1_1);
    let breaks_a_line = |ch: char| ch == '\r' || (xml11 && matches!(ch, '\u{85}' | '\u{2028}'));
    if !text.contains(breaks_a_line) {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\r' {
            // `#xD #xA` and, in XML 1.1, `#xD #x85` are single line ends.
            if chars.peek() == Some(&'\n') || (xml11 && chars.peek() == Some(&'\u{85}')) {
                let _ = chars.next();
            }
            out.push('\n');
        } else if breaks_a_line(ch) {
            out.push('\n');
        } else {
            out.push(ch);
        }
    }
    out
}

/// Removes the `TextDecl` an external parsed entity may begin with
/// ([XML 1.0 §4.3.1](https://www.w3.org/TR/xml/#sec-TextDecl)), which declares
/// the entity's own version and encoding and is not part of its content.
fn strip_text_declaration(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("<?xml") else {
        return text;
    };
    // `<?xmlfoo?>` is a processing instruction, not a text declaration.
    if !rest.starts_with([' ', '\t', '\r', '\n']) {
        return text;
    }
    match rest.find("?>") {
        Some(end) => &rest[end + "?>".len()..],
        None => text,
    }
}

/// Appends `value` to `out` escaped for use inside a double-quoted attribute.
fn push_attribute_escaped(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '"' => out.push_str("&quot;"),
            other => out.push(other),
        }
    }
}

fn resolve_predefined(name: &str) -> Option<&'static str> {
    match name {
        "lt" => Some("<"),
        "gt" => Some(">"),
        "amp" => Some("&"),
        "apos" => Some("'"),
        "quot" => Some("\""),
        _ => None,
    }
}

/// Returns the name of the first general entity reference in `raw` that neither
/// is predefined nor was declared in the internal subset.
fn first_unresolvable_entity(raw: &str, entities: &HashMap<String, String>) -> Option<String> {
    general_entities(raw).find(|name| !entities.contains_key(name.as_str()))
}

/// Returns the name of the first general entity reference in `raw` that is not
/// one of the five predefined ones.
fn first_general_entity(raw: &str) -> Option<String> {
    general_entities(raw).next()
}

/// Iterates over the names of the general entity references in `raw`, skipping
/// character references and the five predefined entities.
fn general_entities(raw: &str) -> impl Iterator<Item = String> + '_ {
    let mut rest = raw;
    std::iter::from_fn(move || {
        loop {
            let start = rest.find('&')?;
            rest = &rest[start + 1..];
            let end = rest.find(';')?;
            let name = &rest[..end];
            rest = &rest[end + 1..];
            if !name.starts_with('#') && resolve_predefined(name).is_none() {
                return Some(name.to_owned());
            }
        }
    })
}
