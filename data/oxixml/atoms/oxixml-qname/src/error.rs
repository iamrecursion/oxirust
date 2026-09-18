//! Error types returned by [`crate::QName`] parsing, [`crate::NamespaceStack::declare`], and
//! [`crate::NamespaceStack::resolve_qname`].

use std::error::Error;
use std::fmt;

/// An error raised by [`crate::QName::parse`] when the input does not match the
/// [`QName` production](https://www.w3.org/TR/xml-names/#NT-QName).
///
/// The [`fmt::Display`] implementation names the specific reason: an empty string, an invalid
/// prefix, or an invalid local part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QNameParseError {
    kind: QNameParseErrorKind,
}

impl QNameParseError {
    pub(crate) fn empty() -> Self {
        Self {
            kind: QNameParseErrorKind::Empty,
        }
    }

    pub(crate) fn invalid_prefix() -> Self {
        Self {
            kind: QNameParseErrorKind::InvalidPrefix,
        }
    }

    pub(crate) fn invalid_local_part() -> Self {
        Self {
            kind: QNameParseErrorKind::InvalidLocalPart,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QNameParseErrorKind {
    Empty,
    InvalidPrefix,
    InvalidLocalPart,
}

impl fmt::Display for QNameParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            QNameParseErrorKind::Empty => write!(f, "a QName must not be empty"),
            QNameParseErrorKind::InvalidPrefix => {
                write!(f, "the QName prefix is not a valid NCName")
            }
            QNameParseErrorKind::InvalidLocalPart => {
                write!(f, "the QName local part is not a valid NCName")
            }
        }
    }
}

impl Error for QNameParseError {}

/// An error raised by [`crate::NamespaceStack::declare`] when a `prefix -> uri` binding would
/// violate [Namespaces in XML §3](https://www.w3.org/TR/xml-names/#ns-decl).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespaceDeclError {
    /// `prefix` is `Some(_)` but is not a syntactically valid `NCName`.
    InvalidPrefixSyntax,
    /// The `xmlns` prefix was passed as `prefix`. `xmlns` is reserved to write namespace
    /// declarations themselves and must never be declared as a namespace prefix.
    ReservedXmlnsPrefix,
    /// The `xml` prefix was passed as `prefix` together with a `uri` other than
    /// [`crate::XML_NAMESPACE`]. The `xml` prefix is implicitly bound to that namespace name
    /// and must not be bound to any other (including the empty string, i.e. it can never be
    /// undeclared).
    XmlPrefixMisbound,
    /// [`crate::XML_NAMESPACE`] was passed as `uri` together with a `prefix` other than
    /// `Some("xml")` (including the default namespace, `prefix = None`). Only the `xml` prefix
    /// may ever be bound to this namespace name.
    XmlNamespaceReserved,
    /// [`crate::XMLNS_NAMESPACE`] was passed as `uri`. No prefix, and no default namespace, may
    /// ever be bound to this namespace name.
    XmlnsNamespaceReserved,
    /// `uri` was the empty string for a non-default `prefix`, but `self.version()` is
    /// [`crate::NamespacesVersion::V1_0`]. Undeclaring a *prefixed* binding with an empty URI
    /// is a [Namespaces in XML 1.1](https://www.w3.org/TR/xml-names11/#dt-NSName) relaxation;
    /// it is a syntax error under Namespaces in XML 1.0. (Undeclaring the *default* namespace
    /// with `xmlns=""` is legal under both versions.)
    EmptyUriRequiresNamespaces11,
}

impl fmt::Display for NamespaceDeclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrefixSyntax => write!(f, "the namespace prefix is not a valid NCName"),
            Self::ReservedXmlnsPrefix => {
                write!(f, "the 'xmlns' prefix must not be declared")
            }
            Self::XmlPrefixMisbound => write!(
                f,
                "the 'xml' prefix must only be bound to the XML namespace name"
            ),
            Self::XmlNamespaceReserved => write!(
                f,
                "the XML namespace name must only be bound to the 'xml' prefix"
            ),
            Self::XmlnsNamespaceReserved => write!(
                f,
                "the XMLNS namespace name must not be bound to any prefix"
            ),
            Self::EmptyUriRequiresNamespaces11 => write!(
                f,
                "undeclaring a prefixed namespace binding with an empty URI requires Namespaces in XML 1.1"
            ),
        }
    }
}

impl Error for NamespaceDeclError {}

/// An error raised by [`crate::NamespaceStack::resolve_qname`] (and
/// [`crate::NamespaceStack::resolve_qname_ref`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveQNameError {
    /// The input did not match the `QName` grammar; see the wrapped [`QNameParseError`].
    InvalidQName(QNameParseError),
    /// The `QName` has a prefix that is not currently bound to any namespace name.
    UnboundPrefix(String),
}

impl fmt::Display for ResolveQNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQName(e) => write!(f, "invalid QName: {e}"),
            Self::UnboundPrefix(prefix) => {
                write!(f, "prefix '{prefix}' is not bound to any namespace")
            }
        }
    }
}

impl Error for ResolveQNameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidQName(e) => Some(e),
            Self::UnboundPrefix(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qname_parse_error_variants_have_non_empty_display() -> Result<(), Box<dyn Error>> {
        for error in [
            QNameParseError::empty(),
            QNameParseError::invalid_prefix(),
            QNameParseError::invalid_local_part(),
        ] {
            assert!(!error.to_string().is_empty());
            let _: &dyn Error = &error;
        }
        Ok(())
    }

    #[test]
    fn namespace_decl_error_variants_have_non_empty_display() {
        let variants = [
            NamespaceDeclError::InvalidPrefixSyntax,
            NamespaceDeclError::ReservedXmlnsPrefix,
            NamespaceDeclError::XmlPrefixMisbound,
            NamespaceDeclError::XmlNamespaceReserved,
            NamespaceDeclError::XmlnsNamespaceReserved,
            NamespaceDeclError::EmptyUriRequiresNamespaces11,
        ];
        for error in variants {
            assert!(!error.to_string().is_empty());
            let _: &dyn Error = &error;
        }
    }

    #[test]
    fn resolve_qname_error_exposes_source_only_for_invalid_qname() {
        let invalid = ResolveQNameError::InvalidQName(QNameParseError::empty());
        assert!(Error::source(&invalid).is_some());
        assert!(!invalid.to_string().is_empty());

        let unbound = ResolveQNameError::UnboundPrefix("p".to_owned());
        assert!(Error::source(&unbound).is_none());
        assert!(unbound.to_string().contains('p'));
    }
}
