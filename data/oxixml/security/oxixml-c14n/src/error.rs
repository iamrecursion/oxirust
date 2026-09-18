//! The error type reported by every canonicalization entry point.

use std::error::Error;
use std::fmt;

use oxixml_dom::{NodeId, NodeKind};

/// The result of a canonicalization.
pub type C14nResult<T> = Result<T, C14nError>;

/// Everything that can go wrong while canonicalizing a document subset.
///
/// Canonicalization itself never fails on a well-formed, namespace-well-formed
/// document: every variant here reports either an input that the
/// Recommendations explicitly require an implementation to reject (a relative
/// namespace URI), an input that is not a proper document tree (an unbound
/// prefix, a dangling node identifier), or an unusable parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum C14nError {
    /// A node identifier does not belong to the document being canonicalized.
    UnknownNode {
        /// The offending identifier.
        node: NodeId,
    },
    /// A node identifier was expected to denote an element and did not.
    NotAnElement {
        /// The offending identifier.
        node: NodeId,
        /// What the node actually is.
        kind: NodeKind,
    },
    /// A namespace declaration binds a prefix to a relative URI reference.
    ///
    /// [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel) requires
    /// implementations to "report an operation failure on documents containing
    /// relative namespace URIs".
    RelativeNamespaceUri {
        /// The declared prefix, or the empty string for the default namespace.
        prefix: String,
        /// The relative URI reference that was found.
        namespace: String,
    },
    /// A qualified name uses a prefix that is not bound at that point of the
    /// tree, so the document is not namespace-well-formed.
    UnboundPrefix {
        /// The prefix that could not be resolved.
        prefix: String,
    },
    /// An unexpanded general entity reference node was found in the subset.
    ///
    /// The data model of [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel)
    /// is the one produced *after* character and parsed entity references have
    /// been replaced, so a canonicalizer cannot see one.
    UnexpandedEntityReference {
        /// The name of the entity.
        name: String,
    },
    /// The algorithm URI handed to [`CanonicalizationAlgorithm::from_uri`] is
    /// not one of the identifiers defined by the XML Security specifications.
    ///
    /// [`CanonicalizationAlgorithm::from_uri`]: crate::CanonicalizationAlgorithm::from_uri
    UnknownAlgorithm {
        /// The URI that was offered.
        uri: String,
    },
    /// A canonicalization parameter could not be understood.
    InvalidParameter {
        /// The name of the parameter.
        parameter: &'static str,
        /// The value that was rejected.
        value: String,
    },
    /// The document subset handed to Canonical XML 2.0 is not expressible in
    /// its restricted input model.
    InvalidSubset {
        /// What is wrong with it.
        detail: &'static str,
    },
}

impl fmt::Display for C14nError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode { node } => {
                write!(f, "node {node} does not belong to this document")
            }
            Self::NotAnElement { node, kind } => {
                write!(f, "node {node} is a {kind}, not an element")
            }
            Self::RelativeNamespaceUri { prefix, namespace } => {
                if prefix.is_empty() {
                    write!(f, "the default namespace `{namespace}` is a relative URI")
                } else {
                    write!(
                        f,
                        "the namespace `{namespace}` bound to the prefix `{prefix}` is a relative URI"
                    )
                }
            }
            Self::UnboundPrefix { prefix } => {
                write!(f, "the namespace prefix `{prefix}` is not bound")
            }
            Self::UnexpandedEntityReference { name } => {
                write!(f, "the general entity reference `&{name};` is unexpanded")
            }
            Self::UnknownAlgorithm { uri } => {
                write!(f, "`{uri}` is not a known canonicalization algorithm")
            }
            Self::InvalidParameter { parameter, value } => {
                write!(f, "`{value}` is not a valid value for `{parameter}`")
            }
            Self::InvalidSubset { detail } => f.write_str(detail),
        }
    }
}

impl Error for C14nError {}
