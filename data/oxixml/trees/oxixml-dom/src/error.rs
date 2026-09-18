//! Error types for the DOM Core operations, the event-stream builder and the
//! serializer.
//!
//! [`DomError`] mirrors the exception vocabulary of
//! [DOM Level 3 Core §1.4](https://www.w3.org/TR/DOM-Level-3-Core/core.html#ID-17189187):
//! every variant carries the same meaning as the `DOMException` code of the same
//! name, and [`DomError::code_name`] returns the specification's constant name so
//! that host bindings can map the Rust error onto the exception they must raise.

use std::fmt;
use std::io;

use crate::node::{NodeId, NodeKind};

/// Result alias for the DOM Core mutation and lookup operations.
pub type DomResult<T> = Result<T, DomError>;

/// A DOM Core exception.
///
/// The variants are named after the `DOMException` codes of
/// [DOM Level 3 Core §1.4](https://www.w3.org/TR/DOM-Level-3-Core/core.html#ID-17189187);
/// each one carries enough context to explain *why* the operation was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DomError {
    /// `INDEX_SIZE_ERR` — an index or size is negative, or greater than the
    /// allowed value.
    IndexSize {
        /// The offending index.
        index: usize,
        /// The number of items that were available.
        length: usize,
    },
    /// `HIERARCHY_REQUEST_ERR` — a node was inserted somewhere it does not belong.
    HierarchyRequest(HierarchyRequest),
    /// `WRONG_DOCUMENT_ERR` — a node was used in a different document than the one
    /// that created it.
    WrongDocument {
        /// Why the document was considered the wrong one.
        detail: &'static str,
    },
    /// `INVALID_CHARACTER_ERR` — a name or a piece of character data contains a
    /// character that the corresponding XML production forbids.
    InvalidCharacter {
        /// The name of the XML production that was violated (`Name`, `NCName`,
        /// `Char`, `Comment`, `PI`, …).
        production: &'static str,
        /// The offending value.
        value: String,
    },
    /// `NOT_FOUND_ERR` — a referenced node does not exist in the context in which
    /// it was used.
    NotFound(NotFound),
    /// `NOT_SUPPORTED_ERR` — the implementation does not support the requested
    /// type of object or operation.
    NotSupported {
        /// What exactly is not supported.
        detail: &'static str,
    },
    /// `INUSE_ATTRIBUTE_ERR` — an attempt was made to add an attribute that is
    /// already in use elsewhere.
    InUseAttribute {
        /// The qualified name of the attribute.
        name: String,
        /// The element that already owns the attribute node.
        owner: NodeId,
    },
    /// `INVALID_STATE_ERR` — an attempt was made to use an object that is not, or
    /// is no longer, usable.
    InvalidState {
        /// What made the state invalid.
        detail: &'static str,
    },
    /// `INVALID_MODIFICATION_ERR` — an attempt was made to modify the type of the
    /// underlying object.
    InvalidModification {
        /// What made the modification invalid.
        detail: &'static str,
    },
    /// `NAMESPACE_ERR` — an attempt was made to create or change an object in a
    /// way that is incorrect with regard to
    /// [Namespaces in XML](https://www.w3.org/TR/xml-names/).
    Namespace(NamespaceViolation),
}

impl DomError {
    /// Returns the `DOMException` constant name that corresponds to this error,
    /// e.g. `"HIERARCHY_REQUEST_ERR"`.
    #[must_use]
    pub const fn code_name(&self) -> &'static str {
        match self {
            Self::IndexSize { .. } => "INDEX_SIZE_ERR",
            Self::HierarchyRequest(_) => "HIERARCHY_REQUEST_ERR",
            Self::WrongDocument { .. } => "WRONG_DOCUMENT_ERR",
            Self::InvalidCharacter { .. } => "INVALID_CHARACTER_ERR",
            Self::NotFound(_) => "NOT_FOUND_ERR",
            Self::NotSupported { .. } => "NOT_SUPPORTED_ERR",
            Self::InUseAttribute { .. } => "INUSE_ATTRIBUTE_ERR",
            Self::InvalidState { .. } => "INVALID_STATE_ERR",
            Self::InvalidModification { .. } => "INVALID_MODIFICATION_ERR",
            Self::Namespace(_) => "NAMESPACE_ERR",
        }
    }

    /// Returns the numeric `DOMException` code that corresponds to this error.
    #[must_use]
    pub const fn code(&self) -> u16 {
        match self {
            Self::IndexSize { .. } => 1,
            Self::HierarchyRequest(_) => 3,
            Self::WrongDocument { .. } => 4,
            Self::InvalidCharacter { .. } => 5,
            Self::NotFound(_) => 8,
            Self::NotSupported { .. } => 9,
            Self::InUseAttribute { .. } => 10,
            Self::InvalidState { .. } => 11,
            Self::InvalidModification { .. } => 13,
            Self::Namespace(_) => 14,
        }
    }
}

impl fmt::Display for DomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.code_name())?;
        match self {
            Self::IndexSize { index, length } => {
                write!(f, "index {index} is out of range for {length} item(s)")
            }
            Self::HierarchyRequest(reason) => reason.fmt(f),
            Self::WrongDocument { detail } => f.write_str(detail),
            Self::InvalidCharacter { production, value } => {
                write!(
                    f,
                    "{value:?} does not match the XML `{production}` production"
                )
            }
            Self::NotFound(reason) => reason.fmt(f),
            Self::NotSupported { detail } | Self::InvalidState { detail } => f.write_str(detail),
            Self::InvalidModification { detail } => f.write_str(detail),
            Self::InUseAttribute { name, owner } => {
                write!(f, "attribute {name:?} is already owned by node #{owner}")
            }
            Self::Namespace(reason) => reason.fmt(f),
        }
    }
}

impl std::error::Error for DomError {}

/// The precise reason a `HIERARCHY_REQUEST_ERR` was raised.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HierarchyRequest {
    /// A node of the given kind may never be a child of a node of the other kind.
    KindNotAllowed {
        /// The kind of the intended parent.
        parent: NodeKind,
        /// The kind of the node that was to be inserted.
        child: NodeKind,
    },
    /// The insertion would have made a node its own ancestor.
    WouldCreateCycle {
        /// The intended parent.
        parent: NodeId,
        /// The node that was to be inserted.
        child: NodeId,
    },
    /// A [`Document`](crate::Document) may have at most one element child (the
    /// document element).
    MultipleDocumentElements,
    /// A [`Document`](crate::Document) may have at most one
    /// [`DocumentType`](crate::NodeKind::DocumentType) child.
    MultipleDoctypes,
    /// The document type declaration must precede the document element.
    DoctypeAfterDocumentElement,
    /// An [`Attr`](crate::NodeKind::Attr) node is never part of the child tree; it
    /// belongs to the attribute list of its owner element.
    AttributeIsNotAChild,
}

impl fmt::Display for HierarchyRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KindNotAllowed { parent, child } => write!(
                f,
                "a {} node cannot be a child of a {} node",
                child.name(),
                parent.name()
            ),
            Self::WouldCreateCycle { parent, child } => write!(
                f,
                "inserting node #{child} into node #{parent} would create a cycle"
            ),
            Self::MultipleDocumentElements => {
                f.write_str("a document may have only one document element")
            }
            Self::MultipleDoctypes => {
                f.write_str("a document may have only one document type declaration")
            }
            Self::DoctypeAfterDocumentElement => {
                f.write_str("the document type declaration must precede the document element")
            }
            Self::AttributeIsNotAChild => {
                f.write_str("an attribute node cannot be inserted into the child list")
            }
        }
    }
}

/// The precise reason a `NOT_FOUND_ERR` was raised.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NotFound {
    /// The identifier does not name a node of this document.
    Node(NodeId),
    /// The node exists, but it is not a child of the node it was looked up in.
    NotAChild {
        /// The node the child was looked up in.
        parent: NodeId,
        /// The node that was not found among its children.
        child: NodeId,
    },
    /// The node exists, but it is not an attribute of the given element.
    NotAnAttribute {
        /// The element the attribute was looked up on.
        element: NodeId,
        /// The node that was not found among its attributes.
        attribute: NodeId,
    },
    /// No attribute with the given name exists on the element.
    Attribute {
        /// The qualified (or expanded) name that was looked up.
        name: String,
    },
    /// The node has no parent, so it cannot be removed or replaced.
    Detached(NodeId),
}

impl fmt::Display for NotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node(id) => write!(f, "no node #{id} in this document"),
            Self::NotAChild { parent, child } => {
                write!(f, "node #{child} is not a child of node #{parent}")
            }
            Self::NotAnAttribute { element, attribute } => {
                write!(
                    f,
                    "node #{attribute} is not an attribute of node #{element}"
                )
            }
            Self::Attribute { name } => write!(f, "no attribute named {name:?}"),
            Self::Detached(id) => write!(f, "node #{id} has no parent"),
        }
    }
}

/// The precise reason a `NAMESPACE_ERR` was raised.
///
/// The constraints are those of
/// [Namespaces in XML 1.0 §3](https://www.w3.org/TR/xml-names/#ns-qualnames) as
/// restated by
/// [DOM Level 3 Core §1.3.3](https://www.w3.org/TR/DOM-Level-3-Core/core.html#Namespaces-Considerations).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NamespaceViolation {
    /// The qualified name is not a well-formed `QName`.
    MalformedQualifiedName {
        /// The offending qualified name.
        qualified_name: String,
    },
    /// The qualified name carries a prefix but the namespace name is absent or
    /// empty.
    PrefixWithoutNamespace {
        /// The offending prefix.
        prefix: String,
    },
    /// The prefix `xml` may only be bound to
    /// `http://www.w3.org/XML/1998/namespace`.
    ReservedPrefixXml {
        /// The namespace name the prefix was bound to.
        namespace: String,
    },
    /// The prefix `xmlns`, and the unprefixed attribute name `xmlns`, may only be
    /// bound to `http://www.w3.org/2000/xmlns/`.
    ReservedPrefixXmlns {
        /// The namespace name the prefix was bound to.
        namespace: String,
    },
    /// `http://www.w3.org/2000/xmlns/` may only be used by the `xmlns` prefix and
    /// by the unprefixed attribute name `xmlns`.
    ReservedNamespaceXmlns {
        /// The qualified name that tried to use the namespace.
        qualified_name: String,
    },
    /// An element name may not use the `xmlns` prefix at all.
    XmlnsPrefixOnElement,
}

impl fmt::Display for NamespaceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedQualifiedName { qualified_name } => {
                write!(f, "{qualified_name:?} is not a well-formed QName")
            }
            Self::PrefixWithoutNamespace { prefix } => write!(
                f,
                "the prefix {prefix:?} cannot be used without a namespace name"
            ),
            Self::ReservedPrefixXml { namespace } => write!(
                f,
                "the prefix `xml` cannot be bound to {namespace:?}; only to \
                 `http://www.w3.org/XML/1998/namespace`"
            ),
            Self::ReservedPrefixXmlns { namespace } => write!(
                f,
                "the name `xmlns` cannot be bound to {namespace:?}; only to \
                 `http://www.w3.org/2000/xmlns/`"
            ),
            Self::ReservedNamespaceXmlns { qualified_name } => write!(
                f,
                "{qualified_name:?} cannot be bound to `http://www.w3.org/2000/xmlns/`"
            ),
            Self::XmlnsPrefixOnElement => {
                f.write_str("an element name cannot use the `xmlns` prefix")
            }
        }
    }
}

/// An error raised while building a [`Document`](crate::Document) from an XML
/// event stream.
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// The underlying [`oxixml_xml`] reader reported a malformed document.
    Xml(oxixml_xml::Error),
    /// The event stream describes a tree the DOM refuses to build.
    Dom(DomError),
    /// A namespace prefix used in the document is not bound in scope.
    UnboundPrefix {
        /// The offending prefix. Empty for an unbound default namespace.
        prefix: String,
    },
    /// An entity reference could not be resolved and
    /// [`EntityPolicy::Strict`](crate::EntityPolicy::Strict) was selected.
    UnknownEntity {
        /// The name of the entity, without the leading `&` and trailing `;`.
        name: String,
    },
    /// A declared entity could not be expanded: it refers to itself, which
    /// [XML 1.0](https://www.w3.org/TR/xml/#norecursion) forbids, or its
    /// expansion would exceed the depth, count or size limits that bound how far
    /// a declaration is allowed to multiply its own replacement text.
    EntityExpansion {
        /// The name of the entity, without the leading `&` and trailing `;`.
        name: String,
        /// What made the expansion impossible.
        detail: String,
    },
    /// The stream ended without ever producing a document element.
    NoDocumentElement,
    /// Two attributes of the same element share one expanded name, which
    /// [Namespaces in XML 1.0 §6.3](https://www.w3.org/TR/xml-names/#uniqAttrs)
    /// forbids (typically two prefixes bound to the same namespace URI).
    DuplicateAttribute {
        /// The qualified name of the offending attribute.
        qualified_name: String,
    },
    /// Non-white-space character data appeared outside the document element,
    /// which [XML 1.0 §2.1](https://www.w3.org/TR/xml/#sec-well-formed) forbids
    /// (`document ::= prolog element Misc*`).
    ContentOutsideDocumentElement {
        /// The offending character data.
        text: String,
    },
    /// A character of the XML 1.1 `RestrictedChar` production appeared
    /// literally. [XML 1.1 §2.1](https://www.w3.org/TR/xml11/#sec-well-formed)
    /// subtracts those characters from the `document` production, so they can
    /// only be written as a character reference.
    RestrictedCharacter {
        /// The offending character.
        character: char,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(err) => write!(f, "XML parse error: {err}"),
            Self::Dom(err) => write!(f, "DOM error: {err}"),
            Self::UnboundPrefix { prefix } if prefix.is_empty() => {
                f.write_str("the default namespace is not declared")
            }
            Self::UnboundPrefix { prefix } => {
                write!(f, "the namespace prefix {prefix:?} is not declared")
            }
            Self::UnknownEntity { name } => write!(f, "unknown entity `&{name};`"),
            Self::EntityExpansion { detail, .. } => f.write_str(detail),
            Self::NoDocumentElement => f.write_str("the document has no document element"),
            Self::DuplicateAttribute { qualified_name } => write!(
                f,
                "the attribute {qualified_name:?} repeats an expanded name already present on the same element"
            ),
            Self::ContentOutsideDocumentElement { text } => {
                write!(f, "character data outside the document element: {text:?}")
            }
            Self::RestrictedCharacter { character } => write!(
                f,
                "the restricted character U+{:04X} may only be written as a character reference \
                 in an XML 1.1 document",
                u32::from(*character)
            ),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Xml(err) => Some(err),
            Self::Dom(err) => Some(err),
            _ => None,
        }
    }
}

impl From<oxixml_xml::Error> for BuildError {
    fn from(err: oxixml_xml::Error) -> Self {
        Self::Xml(err)
    }
}

impl From<DomError> for BuildError {
    fn from(err: DomError) -> Self {
        Self::Dom(err)
    }
}

/// An error raised while serializing a [`Document`](crate::Document) back to XML.
#[derive(Debug)]
#[non_exhaustive]
pub enum SerializeError {
    /// The [`std::io::Write`] sink failed.
    Io(io::Error),
    /// The [`std::fmt::Write`] sink failed.
    Fmt(fmt::Error),
    /// The node identifier passed to the serializer does not exist.
    UnknownNode(NodeId),
}

impl fmt::Display for SerializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Fmt(err) => write!(f, "formatting error: {err}"),
            Self::UnknownNode(id) => write!(f, "no node #{id} in this document"),
        }
    }
}

impl std::error::Error for SerializeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Fmt(err) => Some(err),
            Self::UnknownNode(_) => None,
        }
    }
}

impl From<io::Error> for SerializeError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<fmt::Error> for SerializeError {
    fn from(err: fmt::Error) -> Self {
        Self::Fmt(err)
    }
}
