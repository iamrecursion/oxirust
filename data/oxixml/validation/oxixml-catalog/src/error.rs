//! Error type returned while parsing an OASIS XML Catalog document.

use std::fmt;

use oxixml_iri::IriParseError;
use oxixml_xml::errors::Error as XmlError;

/// An error raised while parsing a [`crate::Catalog`] document.
///
/// Errors raised while lazily loading a `nextCatalog`- or `delegate*`-referenced
/// catalog (via a [`crate::CatalogLoader`]) are **not** reported through this
/// type: per the OASIS XML Catalogs recommendation, a catalog that cannot be
/// loaded or parsed is simply treated as contributing no matches, so that
/// resolution can fall back to whatever catalog entries remain.
#[derive(Debug)]
pub enum CatalogError {
    /// The document could not be tokenized as well-formed XML.
    Xml(XmlError),
    /// The `base` URI supplied to [`crate::Catalog::parse`], or the value of an
    /// `xml:base`, `uri`, or `catalog` attribute, is not a valid IRI reference.
    InvalidIri {
        /// The offending textual value.
        value: String,
        /// The underlying parse error.
        source: IriParseError,
    },
    /// The document's root element is missing, or is not a `catalog` element in
    /// the `urn:oasis:names:tc:entity:xmlns:xml:catalog` namespace.
    MissingRootElement,
    /// The document ended before a start tag was closed.
    UnexpectedEof {
        /// The local name of the element whose end tag was never found.
        element: String,
    },
    /// A required attribute was missing from a catalog entry element.
    MissingAttribute {
        /// The local name of the element.
        element: String,
        /// The local name of the missing attribute.
        attribute: String,
    },
    /// A `prefer` attribute held a value other than `system` or `public`.
    InvalidPreferValue {
        /// The offending value.
        value: String,
    },
    /// `group` elements were nested deeper than
    /// [`MAX_GROUP_DEPTH`](crate::MAX_GROUP_DEPTH).
    ///
    /// Nested `group` scopes are parsed with one native call frame per level, so a document
    /// nesting them deeply enough would otherwise exhaust the call stack — a genuine, uncatchable
    /// process abort — well before hitting any other structural limit in this crate.
    GroupNestingTooDeep {
        /// The depth guard that was exceeded.
        limit: usize,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Xml(e) => write!(f, "malformed catalog XML: {e}"),
            Self::InvalidIri { value, source } => {
                write!(f, "invalid IRI reference {value:?}: {source}")
            }
            Self::MissingRootElement => write!(
                f,
                "the document has no `catalog` root element in the \
                 `urn:oasis:names:tc:entity:xmlns:xml:catalog` namespace"
            ),
            Self::UnexpectedEof { element } => {
                write!(f, "document ended inside an unclosed `<{element}>` element")
            }
            Self::MissingAttribute { element, attribute } => {
                write!(
                    f,
                    "`<{element}>` is missing its required `{attribute}` attribute"
                )
            }
            Self::InvalidPreferValue { value } => write!(
                f,
                "invalid `prefer` value {value:?}: expected `system` or `public`"
            ),
            Self::GroupNestingTooDeep { limit } => write!(
                f,
                "`group` elements are nested more than {limit} levels deep (this limit is not \
                 configurable)"
            ),
        }
    }
}

impl std::error::Error for CatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Xml(e) => Some(e),
            Self::InvalidIri { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<XmlError> for CatalogError {
    fn from(error: XmlError) -> Self {
        Self::Xml(error)
    }
}
