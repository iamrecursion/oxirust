//! Node identifiers, node kinds and the per-kind payloads stored in the arena.
//!
//! A [`Document`](crate::Document) owns a single `Vec<NodeData>`; every node in
//! the tree — including the attribute nodes, which are *not* children — lives in
//! that vector and is addressed by a [`NodeId`]. Because the links between nodes
//! are plain indices rather than reference-counted pointers, a DOM tree built
//! with this crate contains no reference cycles, needs no interior mutability for
//! its structure, and can be cloned, dropped and moved in bulk.

use std::fmt;

use oxixml_qname::ExpandedName;

/// A handle to a node inside one particular [`Document`](crate::Document).
///
/// `NodeId`s are only meaningful together with the document that produced them.
/// Passing an identifier obtained from one document to another one yields
/// [`DomError::NotFound`](crate::DomError::NotFound) (or, in the worst case, a
/// silently wrong but memory-safe answer — the arena is index-based, so there is
/// no way to violate memory safety).
///
/// The document node itself is always [`NodeId::DOCUMENT`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    /// The identifier of the document node, which every document creates first.
    pub const DOCUMENT: Self = Self(0);

    /// Creates an identifier from a raw arena index.
    ///
    /// The index is not validated here; every `Document` method that accepts a
    /// `NodeId` checks it and reports
    /// [`DomError::NotFound`](crate::DomError::NotFound) for an out-of-range one.
    #[must_use]
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }

    /// Returns the raw arena index behind this identifier.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the raw arena index behind this identifier as a `u32`.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The kind of a node, corresponding one-to-one with the DOM Core `nodeType`
/// constants of
/// [DOM Level 3 Core §1.4](https://www.w3.org/TR/DOM-Level-3-Core/core.html#ID-1950641247).
///
/// The `Entity` and `Notation` kinds are deliberately absent: this crate models
/// the *document* tree, and the entity/notation declarations of a DTD are kept
/// with the [`DocumentType`](NodeKind::DocumentType) node's internal subset
/// instead of being reified as nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeKind {
    /// The root of the tree; `nodeType` 9.
    Document,
    /// A `<!DOCTYPE …>` declaration; `nodeType` 10.
    DocumentType,
    /// An element; `nodeType` 1.
    Element,
    /// An attribute; `nodeType` 2. Attributes are not children of their element.
    Attr,
    /// Character data; `nodeType` 3.
    Text,
    /// A `<![CDATA[…]]>` section; `nodeType` 4.
    CdataSection,
    /// A `<!-- … -->` comment; `nodeType` 8.
    Comment,
    /// A `<?target data?>` processing instruction; `nodeType` 7.
    ProcessingInstruction,
    /// An unexpanded `&name;` general entity reference; `nodeType` 5.
    EntityReference,
    /// A lightweight container used to move several nodes at once; `nodeType` 11.
    DocumentFragment,
}

impl NodeKind {
    /// Returns the DOM Core `nodeType` constant for this kind.
    #[must_use]
    pub const fn node_type(self) -> u16 {
        match self {
            Self::Element => 1,
            Self::Attr => 2,
            Self::Text => 3,
            Self::CdataSection => 4,
            Self::EntityReference => 5,
            Self::ProcessingInstruction => 7,
            Self::Comment => 8,
            Self::Document => 9,
            Self::DocumentType => 10,
            Self::DocumentFragment => 11,
        }
    }

    /// Returns the human-readable name of this kind, as used in the DOM
    /// specification's prose (`"element"`, `"CDATA section"`, …).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::DocumentType => "document type",
            Self::Element => "element",
            Self::Attr => "attribute",
            Self::Text => "text",
            Self::CdataSection => "CDATA section",
            Self::Comment => "comment",
            Self::ProcessingInstruction => "processing instruction",
            Self::EntityReference => "entity reference",
            Self::DocumentFragment => "document fragment",
        }
    }

    /// Returns `true` for the kinds whose value is character data, i.e.
    /// [`Text`](Self::Text), [`CdataSection`](Self::CdataSection) and
    /// [`Comment`](Self::Comment).
    #[must_use]
    pub const fn is_character_data(self) -> bool {
        matches!(self, Self::Text | Self::CdataSection | Self::Comment)
    }

    /// Returns `true` when a node of this kind may appear in the child list of
    /// another node at all.
    #[must_use]
    pub const fn can_be_child(self) -> bool {
        !matches!(self, Self::Document | Self::Attr)
    }
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The payload of an [`Element`](NodeKind::Element) node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementData {
    pub(crate) name: ExpandedName,
    pub(crate) prefix: Option<String>,
    pub(crate) attributes: Vec<NodeId>,
}

impl ElementData {
    /// Returns the expanded (namespace name plus local name) name of the element.
    #[must_use]
    pub const fn name(&self) -> &ExpandedName {
        &self.name
    }

    /// Returns the namespace name of the element, or `None` when the element is
    /// in no namespace.
    #[must_use]
    pub fn namespace_uri(&self) -> Option<&str> {
        self.name.namespace()
    }

    /// Returns the local part of the element name.
    #[must_use]
    pub fn local_name(&self) -> &str {
        self.name.local_name()
    }

    /// Returns the namespace prefix the element was spelled with, or `None` when
    /// it was unprefixed.
    #[must_use]
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    /// Returns the qualified name (`prefix:local`, or just `local`) — the DOM
    /// `tagName` of the element.
    #[must_use]
    pub fn qualified_name(&self) -> String {
        qualify(self.prefix.as_deref(), self.name.local_name())
    }

    /// Returns the identifiers of the element's attribute nodes, in the order in
    /// which they were added.
    #[must_use]
    pub fn attribute_ids(&self) -> &[NodeId] {
        &self.attributes
    }
}

/// The payload of an [`Attr`](NodeKind::Attr) node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrData {
    pub(crate) name: ExpandedName,
    pub(crate) prefix: Option<String>,
    pub(crate) value: String,
    pub(crate) specified: bool,
    pub(crate) is_id: bool,
    pub(crate) owner: Option<NodeId>,
}

impl AttrData {
    /// Returns the expanded (namespace name plus local name) name of the
    /// attribute.
    ///
    /// Unprefixed attributes never pick up the default namespace, so their
    /// namespace name is `None` — see
    /// [Namespaces in XML §6.2](https://www.w3.org/TR/xml-names/#defaulting).
    #[must_use]
    pub const fn name(&self) -> &ExpandedName {
        &self.name
    }

    /// Returns the namespace name of the attribute, or `None` when it is in no
    /// namespace.
    #[must_use]
    pub fn namespace_uri(&self) -> Option<&str> {
        self.name.namespace()
    }

    /// Returns the local part of the attribute name.
    #[must_use]
    pub fn local_name(&self) -> &str {
        self.name.local_name()
    }

    /// Returns the namespace prefix the attribute was spelled with.
    #[must_use]
    pub fn prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    /// Returns the qualified name (`prefix:local`, or just `local`) — the DOM
    /// `name` of the attribute.
    #[must_use]
    pub fn qualified_name(&self) -> String {
        qualify(self.prefix.as_deref(), self.name.local_name())
    }

    /// Returns the normalized attribute value.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the DOM `specified` flag: `true` when the attribute was explicitly
    /// given in the document (or set through the API), `false` when it was
    /// supplied from a DTD default.
    #[must_use]
    pub const fn specified(&self) -> bool {
        self.specified
    }

    /// Returns the DOM Level 3 `isId` flag, which marks the attribute as being of
    /// type `ID` and therefore eligible for
    /// [`Document::get_element_by_id`](crate::Document::get_element_by_id).
    #[must_use]
    pub const fn is_id(&self) -> bool {
        self.is_id
    }

    /// Returns the element this attribute belongs to, or `None` when the
    /// attribute node is detached.
    #[must_use]
    pub const fn owner_element(&self) -> Option<NodeId> {
        self.owner
    }
}

/// The payload of a [`DocumentType`](NodeKind::DocumentType) node.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DocumentTypeData {
    pub(crate) name: String,
    pub(crate) public_id: Option<String>,
    pub(crate) system_id: Option<String>,
    pub(crate) internal_subset: Option<String>,
}

impl DocumentTypeData {
    /// Returns the name declared immediately after `<!DOCTYPE`, which must equal
    /// the qualified name of the document element.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the public identifier of the external subset, if any.
    #[must_use]
    pub fn public_id(&self) -> Option<&str> {
        self.public_id.as_deref()
    }

    /// Returns the system identifier of the external subset, if any.
    #[must_use]
    pub fn system_id(&self) -> Option<&str> {
        self.system_id.as_deref()
    }

    /// Returns the internal subset — everything between the `[` and the `]` of
    /// the document type declaration — verbatim, without the brackets.
    #[must_use]
    pub fn internal_subset(&self) -> Option<&str> {
        self.internal_subset.as_deref()
    }
}

/// The payload of a
/// [`ProcessingInstruction`](NodeKind::ProcessingInstruction) node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingInstructionData {
    pub(crate) target: String,
    pub(crate) data: String,
}

impl ProcessingInstructionData {
    /// Returns the target — the `Name` immediately after `<?`.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Returns the instruction data: everything after the target and the white
    /// space that separates it, with the trailing `?>` removed.
    #[must_use]
    pub fn data(&self) -> &str {
        &self.data
    }
}

/// The per-kind payload of a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NodeContent {
    Document,
    DocumentType(DocumentTypeData),
    Element(ElementData),
    Attr(AttrData),
    Text(String),
    CdataSection(String),
    Comment(String),
    ProcessingInstruction(ProcessingInstructionData),
    EntityReference(String),
    DocumentFragment,
}

impl NodeContent {
    pub(crate) const fn kind(&self) -> NodeKind {
        match self {
            Self::Document => NodeKind::Document,
            Self::DocumentType(_) => NodeKind::DocumentType,
            Self::Element(_) => NodeKind::Element,
            Self::Attr(_) => NodeKind::Attr,
            Self::Text(_) => NodeKind::Text,
            Self::CdataSection(_) => NodeKind::CdataSection,
            Self::Comment(_) => NodeKind::Comment,
            Self::ProcessingInstruction(_) => NodeKind::ProcessingInstruction,
            Self::EntityReference(_) => NodeKind::EntityReference,
            Self::DocumentFragment => NodeKind::DocumentFragment,
        }
    }
}

/// One arena slot: the structural links plus the payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeData {
    pub(crate) parent: Option<NodeId>,
    pub(crate) first_child: Option<NodeId>,
    pub(crate) last_child: Option<NodeId>,
    pub(crate) prev_sibling: Option<NodeId>,
    pub(crate) next_sibling: Option<NodeId>,
    pub(crate) content: NodeContent,
}

impl NodeData {
    pub(crate) const fn new(content: NodeContent) -> Self {
        Self {
            parent: None,
            first_child: None,
            last_child: None,
            prev_sibling: None,
            next_sibling: None,
            content,
        }
    }

    pub(crate) const fn kind(&self) -> NodeKind {
        self.content.kind()
    }
}

/// Joins an optional prefix and a local name into a qualified name.
pub(crate) fn qualify(prefix: Option<&str>, local: &str) -> String {
    match prefix {
        Some(prefix) if !prefix.is_empty() => {
            let mut out = String::with_capacity(prefix.len() + 1 + local.len());
            out.push_str(prefix);
            out.push(':');
            out.push_str(local);
            out
        }
        _ => local.to_owned(),
    }
}
