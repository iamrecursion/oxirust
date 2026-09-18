//! Pluggable resource fetching.
//!
//! [XInclude 1.0](https://www.w3.org/TR/xinclude/) is defined over an abstract notion of
//! "retrieving a resource"; this crate never touches the filesystem or the network itself.
//! Instead, [`process_document`](crate::process_document) takes a `&dyn `[`ResourceLoader`],
//! and every `xi:include` resource fetch — for both `parse="xml"` and `parse="text"` — goes
//! through it. The default, [`NullLoader`], resolves nothing: embedding this crate in a service
//! that must never make an outbound request (SSRF-safe by construction) is exactly a matter of
//! passing `&NullLoader` (or simply not implementing anything more capable).

use std::collections::HashMap;
use std::fmt;

/// Why a [`ResourceLoader`] could not produce the bytes of a resource.
///
/// Every variant is a *resource error* in the vocabulary of
/// [XInclude 1.0 §4.4](https://www.w3.org/TR/xinclude/#fallback_element): the processor reacts
/// to it by running `xi:fallback` processing, never by failing the whole document.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoadError {
    /// No resource exists at the given location.
    NotFound,
    /// The resource exists but the loader refused to fetch it (for example, a scheme or host
    /// that a security policy does not allow).
    PermissionDenied,
    /// Any other failure, described by the loader itself (a network error, a timeout, …).
    Other(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("resource not found"),
            Self::PermissionDenied => f.write_str("permission denied"),
            Self::Other(detail) => f.write_str(detail),
        }
    }
}

impl std::error::Error for LoadError {}

/// Fetches the raw bytes an `xi:include` element refers to.
///
/// Implementations receive the fully resolved, absolute IRI (never a relative reference — the
/// processor always resolves `href` against the base URI in scope before calling this) together
/// with the advisory `accept` / `accept-language` values taken verbatim from the `xi:include`
/// element's attributes, for loaders that speak HTTP and want to send them as request headers.
/// A loader that does not care about content negotiation is free to ignore both.
///
/// The same method is used for both `parse="xml"` and `parse="text"` inclusion; the caller
/// decides what to do with the bytes.
pub trait ResourceLoader {
    /// Fetches the resource named by `iri`.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] when the resource cannot be produced. This is always treated as a
    /// *resource error* (see [`LoadError`]'s documentation) — it never aborts the whole
    /// document, only triggers `xi:fallback` processing for this one `xi:include`.
    fn load(
        &self,
        iri: &str,
        accept: Option<&str>,
        accept_language: Option<&str>,
    ) -> Result<Vec<u8>, LoadError>;
}

impl<T: ResourceLoader + ?Sized> ResourceLoader for &T {
    fn load(
        &self,
        iri: &str,
        accept: Option<&str>,
        accept_language: Option<&str>,
    ) -> Result<Vec<u8>, LoadError> {
        (**self).load(iri, accept, accept_language)
    }
}

/// A [`ResourceLoader`] that resolves nothing.
///
/// This is the safe default: it never reaches outside the process, so embedding
/// `oxixml-xinclude` in a context that must not perform filesystem or network I/O is simply a
/// matter of using it (or any other loader that only serves resources from a fixed, trusted
/// set, such as [`MapLoader`]).
///
/// # Example
///
/// ```
/// use oxixml_xinclude::{LoadError, NullLoader, ResourceLoader};
///
/// let loader = NullLoader;
/// assert_eq!(loader.load("https://example.com/x.xml", None, None), Err(LoadError::NotFound));
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NullLoader;

impl ResourceLoader for NullLoader {
    fn load(
        &self,
        _iri: &str,
        _accept: Option<&str>,
        _accept_language: Option<&str>,
    ) -> Result<Vec<u8>, LoadError> {
        Err(LoadError::NotFound)
    }
}

/// A [`ResourceLoader`] backed by a fixed, in-memory table of absolute-IRI-to-bytes mappings.
///
/// Useful for tests and for embedding a small, closed set of trusted resources (a bundled
/// schema catalog, for instance) without touching the filesystem.
///
/// # Example
///
/// ```
/// use oxixml_xinclude::{LoadError, MapLoader, ResourceLoader};
///
/// let loader = MapLoader::new().with("https://example.com/x.xml", b"<r>hi</r>".to_vec());
/// assert_eq!(
///     loader.load("https://example.com/x.xml", None, None),
///     Ok(b"<r>hi</r>".to_vec())
/// );
/// assert_eq!(
///     loader.load("https://example.com/missing.xml", None, None),
///     Err(LoadError::NotFound)
/// );
/// ```
#[derive(Debug, Clone, Default)]
pub struct MapLoader {
    resources: HashMap<String, Vec<u8>>,
}

impl MapLoader {
    /// Creates an empty loader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a resource, keyed by its exact absolute IRI string, consuming and returning
    /// `self` for chaining.
    #[must_use]
    pub fn with(mut self, iri: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        self.resources.insert(iri.into(), bytes.into());
        self
    }

    /// Registers a resource whose content is UTF-8 text, consuming and returning `self` for
    /// chaining.
    #[must_use]
    pub fn with_text(self, iri: impl Into<String>, text: impl AsRef<str>) -> Self {
        self.with(iri, text.as_ref().as_bytes().to_vec())
    }

    /// Registers a resource in place.
    pub fn insert(&mut self, iri: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        self.resources.insert(iri.into(), bytes.into());
    }
}

impl ResourceLoader for MapLoader {
    fn load(
        &self,
        iri: &str,
        _accept: Option<&str>,
        _accept_language: Option<&str>,
    ) -> Result<Vec<u8>, LoadError> {
        self.resources.get(iri).cloned().ok_or(LoadError::NotFound)
    }
}
