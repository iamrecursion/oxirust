//! Document loading: the [`DocumentLoader`] hook used to resolve remote `@context` (and remote
//! document) references.
//!
//! The default implementation ([`NoLoader`]) performs **no** network access whatsoever — the
//! foundation stays pure and side-effect-free. Callers that need to dereference remote contexts
//! supply their own loader, either by implementing [`DocumentLoader`] or, more conveniently, by
//! wrapping a closure with [`ClosureLoader`]. This keeps all I/O policy in the caller's hands.

use crate::error::JsonLdErrorCode;
use crate::profile::JsonLdProfileSet;
use std::error::Error;
use std::fmt::{self, Formatter};

/// Options passed to a [`DocumentLoader`] when resolving a reference.
#[derive(Debug, Default, Clone)]
pub struct LoadDocumentOptions {
    /// One or more IRIs to use in the request as a `profile` parameter, per the JSON-LD API's
    /// `LoadDocumentOptions.requestProfile`.
    pub request_profile: JsonLdProfileSet,
}

/// A document (or context) returned by a [`DocumentLoader`].
///
/// `document_url` is the *final* URL after any redirects, which JSON-LD uses as the base for
/// resolving relative references inside the loaded document.
#[derive(Debug, Clone)]
pub struct RemoteDocument {
    /// The retrieved document, as raw bytes; it is parsed as JSON by the processor.
    pub document: Vec<u8>,
    /// The final URL of the loaded document (after redirects).
    pub document_url: String,
}

impl RemoteDocument {
    /// Builds a [`RemoteDocument`] for `document` retrieved from `document_url`.
    #[must_use]
    pub fn new(document: impl Into<Vec<u8>>, document_url: impl Into<String>) -> Self {
        Self {
            document: document.into(),
            document_url: document_url.into(),
        }
    }
}

/// An error raised while loading a remote document or context.
#[derive(Debug)]
pub struct LoadDocumentError {
    message: String,
    code: JsonLdErrorCode,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl LoadDocumentError {
    /// Builds a loading error with the given message and JSON-LD error code.
    #[must_use]
    pub fn new(message: impl Into<String>, code: JsonLdErrorCode) -> Self {
        Self {
            message: message.into(),
            code,
            source: None,
        }
    }

    /// Attaches an underlying error as the [`source`](Error::source).
    #[must_use]
    pub fn with_source(mut self, source: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// The JSON-LD error code to surface for this failure.
    #[must_use]
    pub fn code(&self) -> JsonLdErrorCode {
        self.code
    }
}

impl fmt::Display for LoadDocumentError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl Error for LoadDocumentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source.as_ref() as &(dyn Error + 'static))
    }
}

/// A hook to resolve remote `@context` / document references (`DocumentLoader` in the JSON-LD
/// API).
///
/// Implementations are responsible for fetching the resource at `url` (from the network, a cache,
/// a bundled resource map, ...), parsing it as JSON, and returning it as a [`RemoteDocument`].
/// They perform whatever I/O policy the caller wants; the JSON-LD algorithms themselves never do
/// any I/O.
pub trait DocumentLoader {
    /// Loads the document identified by `url`.
    fn load(
        &self,
        url: &str,
        options: &LoadDocumentOptions,
    ) -> Result<RemoteDocument, LoadDocumentError>;
}

/// The default loader: refuses every request without touching the network.
///
/// Because the foundation must be side-effect-free by default, any document that references a
/// remote context fails with [`JsonLdErrorCode::LoadingRemoteContextFailed`] unless the caller
/// supplies their own [`DocumentLoader`].
#[derive(Debug, Default, Clone, Copy)]
pub struct NoLoader;

impl DocumentLoader for NoLoader {
    fn load(
        &self,
        url: &str,
        _options: &LoadDocumentOptions,
    ) -> Result<RemoteDocument, LoadDocumentError> {
        Err(LoadDocumentError::new(
            format!("no document loader is configured; cannot load <{url}>"),
            JsonLdErrorCode::LoadingRemoteContextFailed,
        ))
    }
}

/// A [`DocumentLoader`] backed by a closure.
///
/// The closure receives the requested `url` and the [`LoadDocumentOptions`] and returns a
/// [`RemoteDocument`]. This is the convenient way for a caller to plug in loading behaviour
/// without defining a new type.
///
/// ```
/// use oxixml_jsonld::{ClosureLoader, DocumentLoader, LoadDocumentOptions, RemoteDocument};
///
/// let loader = ClosureLoader::new(|url: &str, _opts: &LoadDocumentOptions| {
///     Ok(RemoteDocument::new(
///         br#"{"@context": {"ex": "http://example.com/"}}"#.to_vec(),
///         url,
///     ))
/// });
/// let doc = loader.load("http://example.com/ctx.jsonld", &LoadDocumentOptions::default())?;
/// assert_eq!(doc.document_url, "http://example.com/ctx.jsonld");
/// # Ok::<(), oxixml_jsonld::LoadDocumentError>(())
/// ```
pub struct ClosureLoader<F>(F);

impl<F> ClosureLoader<F>
where
    F: Fn(&str, &LoadDocumentOptions) -> Result<RemoteDocument, LoadDocumentError>,
{
    /// Wraps `closure` as a [`DocumentLoader`].
    pub fn new(closure: F) -> Self {
        Self(closure)
    }
}

impl<F> DocumentLoader for ClosureLoader<F>
where
    F: Fn(&str, &LoadDocumentOptions) -> Result<RemoteDocument, LoadDocumentError>,
{
    fn load(
        &self,
        url: &str,
        options: &LoadDocumentOptions,
    ) -> Result<RemoteDocument, LoadDocumentError> {
        (self.0)(url, options)
    }
}
