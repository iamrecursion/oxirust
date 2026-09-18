//! `oxixml-catalog` — OASIS [XML Catalogs 1.1](https://www.oasis-open.org/committees/entity/spec-2001-08-06.html)
//! resolution.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace —
//! the COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! A [`Catalog`] parses one OASIS XML Catalog document and answers the three
//! lookups the specification defines:
//!
//! * [`Catalog::resolve_system`] — map a `SYSTEM` external identifier to a
//!   local (or otherwise preferred) URI.
//! * [`Catalog::resolve_public`] — map a `PUBLIC` identifier (optionally
//!   alongside its `SYSTEM` identifier, per the `prefer` search strategy) to
//!   a URI.
//! * [`Catalog::resolve_uri`] — map an arbitrary URI reference (for example
//!   an XML Schema `schemaLocation`, or an XInclude `href`) to a replacement.
//!
//! Every element the OASIS Committee Specification 1.1 defines is
//! implemented: `public`, `system`, `uri`, `rewriteSystem`, `rewriteURI`,
//! `systemSuffix`, `uriSuffix`, `delegatePublic`, `delegateSystem`,
//! `delegateURI`, `nextCatalog`, and `group`/`catalog`'s `prefer` and
//! `xml:base` scoping — including longest-match tie-breaking for the
//! rewrite/suffix/delegate families, delegate-chain restart semantics, and
//! loop detection across the whole resolution attempt.
//!
//! # Loading other catalogs
//!
//! A catalog document can reference *other* catalog documents (`nextCatalog`,
//! `delegatePublic`, `delegateSystem`, `delegateURI`). This crate never reads
//! the filesystem or the network to fetch them: every `resolve_*_with` method
//! takes an `Option<&dyn `[`CatalogLoader`]`>`, so callers stay in full
//! control of *what* may be dereferenced — the plain `resolve_*` methods
//! (equivalent to passing `None`) only ever consult the entries already
//! parsed into the [`Catalog`] itself.
//!
//! # Example
//!
//! ```
//! use oxixml_catalog::Catalog;
//!
//! let xml = r#"
//! <catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog" prefer="public">
//!   <public publicId="-//OASIS//DTD DocBook XML V4.5//EN" uri="docbook/docbookx.dtd"/>
//!   <rewriteSystem systemIdStartString="http://example.com/dtds/"
//!                  rewritePrefix="local-dtds/"/>
//!   <uriSuffix uriSuffix=".xsd" uri="schemas/fallback.xsd"/>
//! </catalog>
//! "#;
//!
//! let catalog = Catalog::parse(xml, "file:///etc/xml/catalog.xml")?;
//!
//! // A direct `public` match.
//! assert_eq!(
//!     catalog.resolve_public("-//OASIS//DTD DocBook XML V4.5//EN", None).as_deref(),
//!     Some("file:///etc/xml/docbook/docbookx.dtd")
//! );
//!
//! // `rewriteSystem` splices the rewritten prefix onto the unmatched suffix.
//! assert_eq!(
//!     catalog.resolve_system("http://example.com/dtds/foo.dtd").as_deref(),
//!     Some("file:///etc/xml/local-dtds/foo.dtd")
//! );
//!
//! // `uriSuffix` matches the end of a URI reference.
//! assert_eq!(
//!     catalog.resolve_uri("http://elsewhere.example/anything/at/all.xsd").as_deref(),
//!     Some("file:///etc/xml/schemas/fallback.xsd")
//! );
//!
//! // No entry matches this one: `None`, not an error.
//! assert_eq!(catalog.resolve_system("urn:unknown"), None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod error;
mod model;
mod normalize;
mod parser;
mod resolve;

pub use error::CatalogError;
pub use model::Prefer;
pub use parser::MAX_GROUP_DEPTH;
pub use resolve::{CatalogLoader, MAX_DELEGATE_DEPTH};

use model::CatalogFile;
use oxixml_iri::Iri;

/// A parsed OASIS XML Catalog document, ready to answer
/// [`resolve_system`](Self::resolve_system), [`resolve_public`](Self::resolve_public)
/// and [`resolve_uri`](Self::resolve_uri) lookups.
///
/// See the [crate-level documentation](crate) for an overview and example.
#[derive(Debug, Clone)]
pub struct Catalog {
    file: CatalogFile,
    default_prefer: Prefer,
}

impl Catalog {
    /// Parses `xml` as a catalog document, resolving every relative
    /// `xml:base`, `uri`, and `catalog` reference against `base`.
    ///
    /// The `prefer` search strategy used where neither the root `catalog`
    /// element nor any enclosing `group` declares its own `prefer` attribute
    /// defaults to [`Prefer::Public`] — the most common default among
    /// existing XML Catalog implementations. Use
    /// [`parse_with_prefer`](Self::parse_with_prefer) to choose
    /// [`Prefer::System`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] if `xml` is not well-formed, `base` is not a
    /// valid IRI, the document's root element is not a `catalog` element in
    /// the `urn:oasis:names:tc:entity:xmlns:xml:catalog` namespace, a
    /// required attribute is missing from some entry, or an `xml:base`,
    /// `uri`, `catalog`, or `rewritePrefix` attribute is not a valid IRI
    /// reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_catalog::Catalog;
    ///
    /// let catalog = Catalog::parse(
    ///     r#"<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog"/>"#,
    ///     "file:///etc/xml/catalog.xml",
    /// )?;
    /// assert_eq!(catalog.resolve_uri("urn:whatever"), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn parse(xml: &str, base: &str) -> Result<Self, CatalogError> {
        Self::parse_with_prefer(xml, base, Prefer::Public)
    }

    /// Like [`parse`](Self::parse), but with an explicit default `prefer`
    /// search strategy instead of assuming [`Prefer::Public`].
    ///
    /// # Errors
    ///
    /// See [`parse`](Self::parse).
    pub fn parse_with_prefer(
        xml: &str,
        base: &str,
        default_prefer: Prefer,
    ) -> Result<Self, CatalogError> {
        let base_iri = Iri::parse(base.to_owned()).map_err(|source| CatalogError::InvalidIri {
            value: base.to_owned(),
            source,
        })?;
        let file = parser::parse_document(xml, base_iri, default_prefer)?;
        Ok(Self {
            file,
            default_prefer,
        })
    }

    /// Resolves a `SYSTEM` external identifier using only the entries of this
    /// catalog (equivalent to [`resolve_system_with`](Self::resolve_system_with)
    /// with `loader: None`) — `nextCatalog` and `delegateSystem` entries are
    /// present in the document but cannot be followed without a loader, so
    /// they simply contribute no match.
    ///
    /// Returns `None` if no entry matches; this is normal catalog behavior,
    /// not an error.
    #[must_use]
    pub fn resolve_system(&self, system_id: &str) -> Option<String> {
        self.resolve_system_with(system_id, None)
    }

    /// Resolves a `SYSTEM` external identifier (OASIS XML Catalogs 1.1
    /// §7.1.2), following `nextCatalog` and `delegateSystem` entries through
    /// `loader` when one is supplied.
    ///
    /// A chain of `delegateSystem` restarts longer than
    /// [`MAX_DELEGATE_DEPTH`] contributes no match, the same as a catalog
    /// `loader` cannot reach at all — this bounds how far a misbehaving
    /// `loader` that keeps delegating to new, never-before-seen catalog URIs
    /// can recurse.
    #[must_use]
    pub fn resolve_system_with(
        &self,
        system_id: &str,
        loader: Option<&dyn CatalogLoader>,
    ) -> Option<String> {
        resolve::resolve_external(
            &self.file,
            Some(system_id),
            None,
            loader,
            self.default_prefer,
        )
    }

    /// Resolves a `PUBLIC` identifier, optionally alongside the external
    /// identifier's `SYSTEM` identifier, using only the entries of this
    /// catalog. Equivalent to [`resolve_public_with`](Self::resolve_public_with)
    /// with `loader: None`.
    ///
    /// Returns `None` if no entry matches; this is normal catalog behavior,
    /// not an error.
    #[must_use]
    pub fn resolve_public(&self, public_id: &str, system_id: Option<&str>) -> Option<String> {
        self.resolve_public_with(public_id, system_id, None)
    }

    /// Resolves a `PUBLIC` identifier (OASIS XML Catalogs 1.1 §7.1.1),
    /// following `nextCatalog` and `delegatePublic`/`delegateSystem` entries
    /// through `loader` when one is supplied.
    ///
    /// `system_id`, when present, is tried first via the same `system`,
    /// `rewriteSystem`, `systemSuffix`, and `delegateSystem` lookups as
    /// [`resolve_system_with`](Self::resolve_system_with); the `public_id`
    /// path is only consulted where the `prefer` scope in effect is
    /// [`Prefer::Public`], or `system_id` is `None`.
    ///
    /// As with [`resolve_system_with`](Self::resolve_system_with), a
    /// delegate chain longer than [`MAX_DELEGATE_DEPTH`] contributes no
    /// match rather than recursing without bound.
    #[must_use]
    pub fn resolve_public_with(
        &self,
        public_id: &str,
        system_id: Option<&str>,
        loader: Option<&dyn CatalogLoader>,
    ) -> Option<String> {
        resolve::resolve_external(
            &self.file,
            system_id,
            Some(public_id),
            loader,
            self.default_prefer,
        )
    }

    /// Resolves a URI reference using only the entries of this catalog.
    /// Equivalent to [`resolve_uri_with`](Self::resolve_uri_with) with
    /// `loader: None`.
    ///
    /// Returns `None` if no entry matches; this is normal catalog behavior,
    /// not an error.
    #[must_use]
    pub fn resolve_uri(&self, uri: &str) -> Option<String> {
        self.resolve_uri_with(uri, None)
    }

    /// Resolves a URI reference (OASIS XML Catalogs 1.1 §7.2), following
    /// `nextCatalog` and `delegateURI` entries through `loader` when one is
    /// supplied.
    ///
    /// As with [`resolve_system_with`](Self::resolve_system_with), a
    /// delegate chain longer than [`MAX_DELEGATE_DEPTH`] contributes no
    /// match rather than recursing without bound.
    #[must_use]
    pub fn resolve_uri_with(
        &self,
        uri: &str,
        loader: Option<&dyn CatalogLoader>,
    ) -> Option<String> {
        resolve::resolve_uri(&self.file, uri, loader, self.default_prefer)
    }
}
