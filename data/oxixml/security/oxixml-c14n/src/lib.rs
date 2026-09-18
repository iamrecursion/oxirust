//! `oxixml-c14n` — Canonical XML 1.0, 1.1, 2.0 and Exclusive XML Canonicalization.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace —
//! the COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! Two XML documents can carry exactly the same information and still differ
//! byte for byte: attributes reorder, empty elements collapse, entities expand,
//! namespace declarations move. *Canonicalization* is the function that maps all
//! of those spellings onto one, so that a digital signature can be computed over
//! meaning rather than over bytes. This crate implements every canonicalization
//! algorithm the XML Security stack uses:
//!
//! | algorithm | identifier URI | entry point |
//! |---|---|---|
//! | [Canonical XML 1.0](https://www.w3.org/TR/xml-c14n) | `…/REC-xml-c14n-20010315`(`#WithComments`) | [`c14n_1_0`] |
//! | [Canonical XML 1.1](https://www.w3.org/TR/xml-c14n11/) | `…/2006/12/xml-c14n11`(`#WithComments`) | [`c14n_1_1`] |
//! | [Exclusive C14N 1.0](https://www.w3.org/TR/xml-exc-c14n/) | `…/2001/10/xml-exc-c14n#`(`WithComments`) | [`exclusive_c14n`] |
//! | [Canonical XML 2.0](https://www.w3.org/TR/xml-c14n2/) | `…/2010/xml-c14n2` | [`c14n_2_0`] |
//!
//! All of them read an [`oxixml_dom::Document`] and write UTF-8 octets, with no
//! byte order mark, no XML declaration and no document type declaration.
//!
//! # Document subsets
//!
//! The input is not a subtree but a [`NodeSet`] — an XPath node-set, exactly as
//! [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel) specifies, so
//! that an XPath transform in an `<ds:Signature>` can select an arbitrary subset
//! of the document. Omitted nodes still shape the output of the nodes that
//! survive: their namespace declarations stay in scope, and (for the inclusive
//! algorithms) their `xml:*` attributes are re-imported into the orphaned
//! descendants. Canonical XML 2.0 uses its own, deliberately narrower input
//! model, [`C14n2Subset`].
//!
//! # What the caller has to do first
//!
//! [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel) canonicalizes
//! the tree an XML processor produces, and it lists four things that processor
//! must already have done: normalize line breaks to `#xA` *before* parsing,
//! normalize attribute values, replace CDATA sections with their character
//! content, and resolve character and parsed entity references — the last two
//! "in accordance with the behaviors of a validating XML processor", including
//! creating nodes for attributes defaulted from an `<!ATTLIST>` declaration.
//!
//! None of that can be done here. Line breaks in particular *must* be normalized
//! before parsing, because once a `#xD` is in the tree there is no way to tell a
//! literal carriage return (which should have become `#xA`) from one written as
//! `&#xD;` (which must survive as `&#xD;`). CDATA sections and character
//! references are handled by [`oxixml_dom::build`]; DTD-driven attribute
//! defaulting and tokenized-type normalization are the job of a validating
//! builder, and a subset that needs them must apply them before calling in.
//!
//! # Choosing an algorithm
//!
//! * **Inclusive** (1.0, 1.1) reproduces the whole inherited namespace context
//!   at the apex of the subset. Signing `<Foo>` inside one envelope and
//!   verifying it inside another therefore fails.
//! * **Exclusive** renders a namespace declaration only where it is *visibly
//!   utilized*, which makes an extracted subdocument canonicalize identically
//!   however it is re-enveloped. Prefixes that only appear inside content (a
//!   `QName` in an attribute value, an XPath expression) must be listed in the
//!   `InclusiveNamespaces` [`PrefixList`](InclusiveNamespaces) or they will be
//!   lost.
//! * **2.0** keeps exclusive namespace handling and closes the
//!   `QName`-in-content hole with its `QNameAware` parameter. It is implemented
//!   here as a [staged feature](c14n2); see that module for the known
//!   deviations.
//!
//! # Example
//!
//! ```
//! use oxixml_c14n::{CanonicalizationAlgorithm, InclusiveNamespaces, NodeSet};
//! use oxixml_c14n::{c14n_1_0, c14n_1_1, exclusive_c14n};
//! use oxixml_dom::build;
//!
//! let doc = build::parse_str(
//!     r#"<?xml version="1.0"?>
//!        <env:Envelope xmlns:env="urn:env" xmlns:unused="urn:unused" xml:lang="ja">
//!          <p:Payload xmlns:p="urn:payload" b="2" a="1">   text   </p:Payload>
//!          <!-- a comment -->
//!        </env:Envelope>"#,
//! )?;
//!
//! // The whole document, Canonical XML 1.0 without comments: the XML
//! // declaration is gone, the attributes are sorted, and nothing else moved.
//! let whole = c14n_1_0(&doc, &NodeSet::document(&doc), false)?;
//! assert!(whole.starts_with(br#"<env:Envelope xmlns:env="urn:env" xmlns:unused="urn:unused""#));
//! assert!(!whole.windows(9).any(|window| window == b"a comment"));
//!
//! // Just the payload. Inclusive canonicalization drags the whole namespace
//! // context and `xml:lang` along …
//! let payload = doc
//!     .get_elements_by_tag_name_ns(doc.root(), Some("urn:payload"), "Payload")
//!     .next()
//!     .expect("the payload element");
//! let subset = NodeSet::subtree(&doc, payload);
//! assert_eq!(
//!     c14n_1_1(&doc, &subset, false)?,
//!     br#"<p:Payload xmlns:env="urn:env" xmlns:p="urn:payload" xmlns:unused="urn:unused" a="1" b="2" xml:lang="ja">   text   </p:Payload>"#,
//! );
//!
//! // … while exclusive canonicalization keeps only what the element uses.
//! assert_eq!(
//!     exclusive_c14n(&doc, &subset, &InclusiveNamespaces::none(), false)?,
//!     br#"<p:Payload xmlns:p="urn:payload" a="1" b="2">   text   </p:Payload>"#,
//! );
//!
//! // Signature code normally arrives with a URI rather than a function.
//! let algorithm = CanonicalizationAlgorithm::from_uri("http://www.w3.org/2006/12/xml-c14n11")?;
//! assert_eq!(algorithm.apply(&doc, &subset)?, c14n_1_1(&doc, &subset, false)?);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod algorithm;
pub mod c14n2;
mod classic;
mod error;
mod escape;
mod namespaces;
mod nodeset;
mod uri;

use oxixml_dom::{Document, NodeId};

pub use algorithm::{
    CANONICAL_XML_1_0, CANONICAL_XML_1_0_WITH_COMMENTS, CANONICAL_XML_1_1,
    CANONICAL_XML_1_1_WITH_COMMENTS, CANONICAL_XML_2_0, CanonicalizationAlgorithm,
    EXCLUSIVE_XML_C14N_1_0, EXCLUSIVE_XML_C14N_1_0_WITH_COMMENTS,
};
pub use c14n2::{C14N2_NAMESPACE, C14n2Parameters, C14n2Subset, PrefixRewrite, QNameAware};
pub use classic::InclusiveNamespaces;
pub use error::{C14nError, C14nResult};
pub use nodeset::NodeSet;
pub use uri::join_uri_references;

use classic::Variant;

/// Canonicalizes a document subset with
/// [Canonical XML 1.0](https://www.w3.org/TR/xml-c14n).
///
/// `with_comments` selects between the two algorithm identifiers: `false` is
/// [`CANONICAL_XML_1_0`], `true` is [`CANONICAL_XML_1_0_WITH_COMMENTS`].
///
/// # Errors
///
/// See [`C14nError`]; in particular a relative namespace URI is an operation
/// failure, as the Recommendation requires.
pub fn c14n_1_0(document: &Document, subset: &NodeSet, with_comments: bool) -> C14nResult<Vec<u8>> {
    classic::canonicalize(document, subset, &Variant::Inclusive10, with_comments)
}

/// Canonicalizes a document subset with
/// [Canonical XML 1.1](https://www.w3.org/TR/xml-c14n11/).
///
/// Version 1.1 differs from 1.0 only in document subsets: `xml:id` is no longer
/// inherited into an orphaned element, and `xml:base` is *fixed up* — the base
/// URIs of the dropped ancestors are folded into the surviving element's own
/// `xml:base` — so that relative references keep resolving to the same place.
///
/// # Errors
///
/// See [`C14nError`].
pub fn c14n_1_1(document: &Document, subset: &NodeSet, with_comments: bool) -> C14nResult<Vec<u8>> {
    classic::canonicalize(document, subset, &Variant::Inclusive11, with_comments)
}

/// Canonicalizes a document subset with
/// [Exclusive XML Canonicalization 1.0](https://www.w3.org/TR/xml-exc-c14n/).
///
/// `inclusive_prefixes` is the `InclusiveNamespaces` `PrefixList` parameter:
/// prefixes named there are rendered by the inclusive rules even when the
/// element does not visibly utilize them, which is how a prefix that only occurs
/// inside content survives.
///
/// # Errors
///
/// See [`C14nError`].
pub fn exclusive_c14n(
    document: &Document,
    subset: &NodeSet,
    inclusive_prefixes: &InclusiveNamespaces,
    with_comments: bool,
) -> C14nResult<Vec<u8>> {
    classic::canonicalize(
        document,
        subset,
        &Variant::Exclusive(inclusive_prefixes.clone()),
        with_comments,
    )
}

/// Canonicalizes a document subset with
/// [Canonical XML 2.0](https://www.w3.org/TR/xml-c14n2/).
///
/// See the [`c14n2`] module for what "staged" means here.
///
/// # Errors
///
/// See [`C14nError`].
pub fn c14n_2_0(
    document: &Document,
    subset: &C14n2Subset,
    parameters: &C14n2Parameters,
) -> C14nResult<Vec<u8>> {
    c14n2::canonicalize(document, subset, parameters)
}

/// Canonicalizes a whole document with Canonical XML 1.0.
///
/// Shorthand for [`c14n_1_0`] over [`NodeSet::document`].
///
/// # Errors
///
/// See [`C14nError`].
pub fn c14n_1_0_document(document: &Document, with_comments: bool) -> C14nResult<Vec<u8>> {
    c14n_1_0(document, &NodeSet::document(document), with_comments)
}

/// Canonicalizes one element and everything below it with Canonical XML 1.0.
///
/// Shorthand for [`c14n_1_0`] over [`NodeSet::subtree`]. Because the document
/// node is not in the subset, `element` is an orphan and the `xml:*`
/// inheritance rules of a document subset apply to it.
///
/// # Errors
///
/// See [`C14nError`].
pub fn c14n_1_0_subtree(
    document: &Document,
    element: NodeId,
    with_comments: bool,
) -> C14nResult<Vec<u8>> {
    c14n_1_0(
        document,
        &NodeSet::subtree(document, element),
        with_comments,
    )
}

/// Canonicalizes one element and everything below it with Exclusive C14N 1.0.
///
/// This is the shape almost every XML Signature uses: an enveloped element,
/// canonicalized so that re-enveloping it cannot change the digest.
///
/// # Errors
///
/// See [`C14nError`].
pub fn exclusive_c14n_subtree(
    document: &Document,
    element: NodeId,
    inclusive_prefixes: &InclusiveNamespaces,
    with_comments: bool,
) -> C14nResult<Vec<u8>> {
    exclusive_c14n(
        document,
        &NodeSet::subtree(document, element),
        inclusive_prefixes,
        with_comments,
    )
}
