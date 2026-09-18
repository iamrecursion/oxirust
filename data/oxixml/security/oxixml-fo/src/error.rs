//! Errors reported while building and refining an XSL-FO tree.

use core::fmt;

use crate::fo::FoElement;

/// Result alias for the fallible operations of this crate.
pub type Result<T> = core::result::Result<T, FoError>;

/// Position of an error inside the source document, in bytes from its start.
///
/// The offset is the one reported by the underlying [`oxixml_xml`] reader for the
/// event that was being processed, so it points at the *end* of the offending
/// markup rather than at its first character.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SourcePosition {
    /// Byte offset from the beginning of the document.
    pub offset: usize,
}

impl SourcePosition {
    /// Creates a position at `offset` bytes from the start of the document.
    #[must_use]
    pub const fn new(offset: usize) -> Self {
        Self { offset }
    }
}

impl fmt::Display for SourcePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "byte {}", self.offset)
    }
}

/// An error raised while parsing, building or refining a formatting-object tree.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FoError {
    /// The underlying XML is not well-formed, or could not be decoded.
    Xml {
        /// Human-readable rendering of the [`oxixml_xml`] error.
        message: String,
        /// Where the reader stopped.
        position: SourcePosition,
    },
    /// An element in the XSL-FO namespace is not a formatting object defined by
    /// XSL 1.1.
    UnknownFormattingObject {
        /// The local name that was encountered.
        name: String,
        /// Where the element started.
        position: SourcePosition,
    },
    /// The document element is not `fo:root`.
    NotAnFoDocument {
        /// The expanded name of the document element, for diagnostics.
        found: String,
    },
    /// A formatting object appears where the content model of its parent does not
    /// allow it.
    InvalidChild {
        /// The parent formatting object.
        parent: FoElement,
        /// The rejected child.
        child: FoElement,
        /// Where the child started.
        position: SourcePosition,
    },
    /// Character data appears where the content model of the parent forbids it.
    InvalidText {
        /// The formatting object whose content model was violated.
        parent: FoElement,
        /// Where the text was found.
        position: SourcePosition,
    },
    /// A required child is missing, or a child occurs too often.
    ContentModel {
        /// The formatting object whose content model was violated.
        parent: FoElement,
        /// What the content model requires.
        expected: String,
        /// Where the parent element ended.
        position: SourcePosition,
    },
    /// A property was specified on a formatting object that it does not apply to.
    ///
    /// XSL 1.1 §7 lists, for every formatting object, the properties that apply
    /// to it; specifying any other property is an error.
    InapplicableProperty {
        /// The formatting object.
        fo: FoElement,
        /// The offending property name, including any `.component` suffix.
        property: String,
        /// Where the element started.
        position: SourcePosition,
    },
    /// An attribute in the XSL-FO namespace-less attribute space is not a known
    /// property name.
    UnknownProperty {
        /// The attribute name as written.
        property: String,
        /// Where the element started.
        position: SourcePosition,
    },
    /// A property value expression could not be tokenized or parsed.
    Expression {
        /// The property whose value failed to parse.
        property: String,
        /// What went wrong.
        message: String,
    },
    /// A property value is syntactically valid but not permitted for the
    /// property.
    InvalidPropertyValue {
        /// The property whose value was rejected.
        property: String,
        /// The value as written.
        value: String,
        /// Why it was rejected.
        message: String,
    },
    /// A compound property was given an unknown `.component` suffix.
    UnknownComponent {
        /// The compound property.
        property: String,
        /// The rejected component name.
        component: String,
    },
}

impl fmt::Display for FoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml { message, position } => write!(f, "XML error at {position}: {message}"),
            Self::UnknownFormattingObject { name, position } => {
                write!(f, "unknown formatting object `fo:{name}` at {position}")
            }
            Self::NotAnFoDocument { found } => {
                write!(f, "document element is `{found}`, expected `fo:root`")
            }
            Self::InvalidChild {
                parent,
                child,
                position,
            } => write!(
                f,
                "`{}` is not allowed as a child of `{}` at {position}",
                child.qualified_name(),
                parent.qualified_name()
            ),
            Self::InvalidText { parent, position } => write!(
                f,
                "character data is not allowed in `{}` at {position}",
                parent.qualified_name()
            ),
            Self::ContentModel {
                parent,
                expected,
                position,
            } => write!(
                f,
                "content of `{}` at {position} does not match its content model: {expected}",
                parent.qualified_name()
            ),
            Self::InapplicableProperty {
                fo,
                property,
                position,
            } => write!(
                f,
                "the `{property}` property does not apply to `{}` at {position}",
                fo.qualified_name()
            ),
            Self::UnknownProperty { property, position } => {
                write!(f, "unknown property `{property}` at {position}")
            }
            Self::Expression { property, message } => {
                write!(f, "invalid expression for `{property}`: {message}")
            }
            Self::InvalidPropertyValue {
                property,
                value,
                message,
            } => write!(f, "invalid value `{value}` for `{property}`: {message}"),
            Self::UnknownComponent {
                property,
                component,
            } => write!(f, "`{property}` has no `.{component}` component"),
        }
    }
}

impl std::error::Error for FoError {}

impl FoError {
    /// Builds an [`FoError::Xml`] from an [`oxixml_xml`] error and a reader offset.
    pub(crate) fn xml(error: &oxixml_xml::Error, offset: usize) -> Self {
        Self::Xml {
            message: error.to_string(),
            position: SourcePosition::new(offset),
        }
    }

    /// Builds an [`FoError::Expression`].
    pub(crate) fn expression(property: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Expression {
            property: property.into(),
            message: message.into(),
        }
    }

    /// Builds an [`FoError::InvalidPropertyValue`].
    pub(crate) fn invalid_value(
        property: impl Into<String>,
        value: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::InvalidPropertyValue {
            property: property.into(),
            value: value.into(),
            message: message.into(),
        }
    }
}
