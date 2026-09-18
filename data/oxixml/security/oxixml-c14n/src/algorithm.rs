//! The canonicalization algorithms, keyed by their W3C identifier URI.

use std::fmt;
use std::str::FromStr;

use oxixml_dom::Document;

use crate::classic::{InclusiveNamespaces, Variant};
use crate::error::{C14nError, C14nResult};
use crate::nodeset::NodeSet;

/// `http://www.w3.org/TR/2001/REC-xml-c14n-20010315`
pub const CANONICAL_XML_1_0: &str = "http://www.w3.org/TR/2001/REC-xml-c14n-20010315";
/// `http://www.w3.org/TR/2001/REC-xml-c14n-20010315#WithComments`
pub const CANONICAL_XML_1_0_WITH_COMMENTS: &str =
    "http://www.w3.org/TR/2001/REC-xml-c14n-20010315#WithComments";
/// `http://www.w3.org/2006/12/xml-c14n11`
pub const CANONICAL_XML_1_1: &str = "http://www.w3.org/2006/12/xml-c14n11";
/// `http://www.w3.org/2006/12/xml-c14n11#WithComments`
pub const CANONICAL_XML_1_1_WITH_COMMENTS: &str =
    "http://www.w3.org/2006/12/xml-c14n11#WithComments";
/// `http://www.w3.org/2001/10/xml-exc-c14n#`
pub const EXCLUSIVE_XML_C14N_1_0: &str = "http://www.w3.org/2001/10/xml-exc-c14n#";
/// `http://www.w3.org/2001/10/xml-exc-c14n#WithComments`
pub const EXCLUSIVE_XML_C14N_1_0_WITH_COMMENTS: &str =
    "http://www.w3.org/2001/10/xml-exc-c14n#WithComments";
/// `http://www.w3.org/2010/xml-c14n2`
pub const CANONICAL_XML_2_0: &str = "http://www.w3.org/2010/xml-c14n2";

/// One of the canonicalization algorithms an XML Signature may name in
/// `<ds:CanonicalizationMethod Algorithm="…">` or
/// `<ds:Transform Algorithm="…">`.
///
/// The enum exists so that `oxixml-dsig` can turn an algorithm URI straight into
/// something runnable, without having to know how the algorithms differ.
///
/// # Example
///
/// ```
/// use oxixml_c14n::{CanonicalizationAlgorithm, NodeSet};
/// use oxixml_dom::build;
///
/// let algorithm = CanonicalizationAlgorithm::from_uri(
///     "http://www.w3.org/2001/10/xml-exc-c14n#",
/// )?;
/// assert!(!algorithm.includes_comments());
/// assert_eq!(algorithm.uri(), "http://www.w3.org/2001/10/xml-exc-c14n#");
///
/// let doc = build::parse_str("<a xmlns:unused='urn:x'><b/></a>")?;
/// let canonical = algorithm.apply(&doc, &NodeSet::document(&doc))?;
/// assert_eq!(canonical, b"<a><b></b></a>");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum CanonicalizationAlgorithm {
    /// Canonical XML 1.0, comments omitted.
    Canonical1_0,
    /// Canonical XML 1.0, comments retained.
    Canonical1_0WithComments,
    /// Canonical XML 1.1, comments omitted.
    Canonical1_1,
    /// Canonical XML 1.1, comments retained.
    Canonical1_1WithComments,
    /// Exclusive XML Canonicalization 1.0, comments omitted.
    Exclusive1_0,
    /// Exclusive XML Canonicalization 1.0, comments retained.
    Exclusive1_0WithComments,
    /// Canonical XML 2.0, with its default parameters.
    ///
    /// Canonical XML 2.0 carries its options in child elements of
    /// `<ds:CanonicalizationMethod>` rather than in the algorithm URI; use
    /// [`crate::c14n_2_0`] with an explicit [`crate::C14n2Parameters`] to set
    /// them.
    Canonical2_0,
}

impl CanonicalizationAlgorithm {
    /// Every algorithm this crate implements, in a stable order.
    pub const ALL: [Self; 7] = [
        Self::Canonical1_0,
        Self::Canonical1_0WithComments,
        Self::Canonical1_1,
        Self::Canonical1_1WithComments,
        Self::Exclusive1_0,
        Self::Exclusive1_0WithComments,
        Self::Canonical2_0,
    ];

    /// Looks an algorithm up by its identifier URI.
    ///
    /// # Errors
    ///
    /// Returns [`C14nError::UnknownAlgorithm`] for any other URI.
    pub fn from_uri(uri: &str) -> C14nResult<Self> {
        match uri {
            CANONICAL_XML_1_0 => Ok(Self::Canonical1_0),
            CANONICAL_XML_1_0_WITH_COMMENTS => Ok(Self::Canonical1_0WithComments),
            CANONICAL_XML_1_1 => Ok(Self::Canonical1_1),
            CANONICAL_XML_1_1_WITH_COMMENTS => Ok(Self::Canonical1_1WithComments),
            EXCLUSIVE_XML_C14N_1_0 => Ok(Self::Exclusive1_0),
            EXCLUSIVE_XML_C14N_1_0_WITH_COMMENTS => Ok(Self::Exclusive1_0WithComments),
            CANONICAL_XML_2_0 => Ok(Self::Canonical2_0),
            other => Err(C14nError::UnknownAlgorithm {
                uri: other.to_owned(),
            }),
        }
    }

    /// Returns the identifier URI of this algorithm.
    #[must_use]
    pub const fn uri(self) -> &'static str {
        match self {
            Self::Canonical1_0 => CANONICAL_XML_1_0,
            Self::Canonical1_0WithComments => CANONICAL_XML_1_0_WITH_COMMENTS,
            Self::Canonical1_1 => CANONICAL_XML_1_1,
            Self::Canonical1_1WithComments => CANONICAL_XML_1_1_WITH_COMMENTS,
            Self::Exclusive1_0 => EXCLUSIVE_XML_C14N_1_0,
            Self::Exclusive1_0WithComments => EXCLUSIVE_XML_C14N_1_0_WITH_COMMENTS,
            Self::Canonical2_0 => CANONICAL_XML_2_0,
        }
    }

    /// Returns `true` when comment nodes survive canonicalization.
    #[must_use]
    pub const fn includes_comments(self) -> bool {
        matches!(
            self,
            Self::Canonical1_0WithComments
                | Self::Canonical1_1WithComments
                | Self::Exclusive1_0WithComments
        )
    }

    /// Returns `true` for the two exclusive variants, the ones that accept an
    /// `InclusiveNamespaces` `PrefixList`.
    #[must_use]
    pub const fn is_exclusive(self) -> bool {
        matches!(self, Self::Exclusive1_0 | Self::Exclusive1_0WithComments)
    }

    /// Returns the same algorithm with comments turned on or off, where such a
    /// variant exists.
    ///
    /// Canonical XML 2.0 controls comments through its `IgnoreComments`
    /// parameter, so it is returned unchanged.
    #[must_use]
    pub const fn with_comments(self, comments: bool) -> Self {
        match (self, comments) {
            (Self::Canonical1_0 | Self::Canonical1_0WithComments, true) => {
                Self::Canonical1_0WithComments
            }
            (Self::Canonical1_0 | Self::Canonical1_0WithComments, false) => Self::Canonical1_0,
            (Self::Canonical1_1 | Self::Canonical1_1WithComments, true) => {
                Self::Canonical1_1WithComments
            }
            (Self::Canonical1_1 | Self::Canonical1_1WithComments, false) => Self::Canonical1_1,
            (Self::Exclusive1_0 | Self::Exclusive1_0WithComments, true) => {
                Self::Exclusive1_0WithComments
            }
            (Self::Exclusive1_0 | Self::Exclusive1_0WithComments, false) => Self::Exclusive1_0,
            (Self::Canonical2_0, _) => Self::Canonical2_0,
        }
    }

    /// Canonicalizes `subset` with this algorithm and an empty
    /// `InclusiveNamespaces` `PrefixList`.
    ///
    /// # Errors
    ///
    /// See [`C14nError`].
    pub fn apply(self, document: &Document, subset: &NodeSet) -> C14nResult<Vec<u8>> {
        self.apply_with(document, subset, &InclusiveNamespaces::none())
    }

    /// Canonicalizes `subset`, passing an `InclusiveNamespaces` `PrefixList` to
    /// the exclusive variants (other algorithms ignore it).
    ///
    /// # Errors
    ///
    /// See [`C14nError`].
    pub fn apply_with(
        self,
        document: &Document,
        subset: &NodeSet,
        inclusive_prefixes: &InclusiveNamespaces,
    ) -> C14nResult<Vec<u8>> {
        if self == Self::Canonical2_0 {
            return crate::c14n2::canonicalize(
                document,
                &crate::c14n2::C14n2Subset::from_node_set(document, subset),
                &crate::c14n2::C14n2Parameters::new(),
            );
        }
        let variant = match self {
            Self::Canonical1_0 | Self::Canonical1_0WithComments => Variant::Inclusive10,
            Self::Canonical1_1 | Self::Canonical1_1WithComments => Variant::Inclusive11,
            _ => Variant::Exclusive(inclusive_prefixes.clone()),
        };
        crate::classic::canonicalize(document, subset, &variant, self.includes_comments())
    }
}

impl fmt::Display for CanonicalizationAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.uri())
    }
}

impl FromStr for CanonicalizationAlgorithm {
    type Err = C14nError;

    fn from_str(uri: &str) -> Result<Self, Self::Err> {
        Self::from_uri(uri)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_algorithm_round_trips_through_its_uri() -> C14nResult<()> {
        for algorithm in CanonicalizationAlgorithm::ALL {
            assert_eq!(
                CanonicalizationAlgorithm::from_uri(algorithm.uri())?,
                algorithm
            );
        }
        Ok(())
    }

    #[test]
    fn unknown_uris_are_rejected() {
        let error = CanonicalizationAlgorithm::from_uri("urn:nope")
            .expect_err("`urn:nope` is not an algorithm identifier");
        assert_eq!(
            error,
            C14nError::UnknownAlgorithm {
                uri: "urn:nope".to_owned()
            }
        );
    }

    #[test]
    fn comment_variants_pair_up() {
        for algorithm in CanonicalizationAlgorithm::ALL {
            let with = algorithm.with_comments(true);
            let without = algorithm.with_comments(false);
            if algorithm == CanonicalizationAlgorithm::Canonical2_0 {
                assert_eq!(with, without);
            } else {
                assert!(with.includes_comments());
                assert!(!without.includes_comments());
                assert_eq!(with.is_exclusive(), algorithm.is_exclusive());
            }
        }
    }
}
