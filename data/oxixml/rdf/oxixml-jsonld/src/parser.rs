//! The streaming [`JsonLdParser`]: the `oxjsonld`-compatible entry point that yields RDF [`Quad`]s.
//!
//! The parser materialises the document into a [`JsonValue`] tree, expands it, and deserialises it
//! to RDF (the full JSON-LD 1.1 *toRdf* pipeline), then yields the resulting quads one by one. The
//! whole document is buffered: JSON-LD's context rules make a genuinely incremental toRdf mapping
//! impossible in general, and the [streaming profile](crate::JsonLdProfile::Streaming) is honoured
//! by *validating* the document's key order rather than by changing the output.

use crate::api::{JsonLdOptions, document_prefixes, to_rdf};
use crate::error::{JsonLdErrorCode, JsonLdParseError, JsonLdSyntaxError};
use crate::loader::{
    ClosureLoader, DocumentLoader, LoadDocumentError, LoadDocumentOptions, NoLoader, RemoteDocument,
};
use crate::profile::{JsonLdProcessingMode, JsonLdProfileSet};
use crate::to_rdf::RdfDirection;
use crate::value::{DEFAULT_MAX_NESTING_DEPTH, JsonValue};
use oxixml_iri::{Iri, IriParseError};
use oxixml_model::Quad;
use std::error::Error;
use std::io::Read;
use std::marker::PhantomData;
use std::slice;
use std::vec;
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncReadExt};

/// A [JSON-LD](https://www.w3.org/TR/json-ld/) parser producing RDF [`Quad`]s.
///
/// # Example
///
/// ```
/// use oxixml_jsonld::JsonLdParser;
/// use oxixml_model::vocab::rdf;
/// use oxixml_model::{NamedNode, Term};
///
/// let file = r#"{
///     "@context": {"schema": "http://schema.org/"},
///     "@graph": [
///         {"@type": "schema:Person", "@id": "http://example.com/foo", "schema:name": "Foo"},
///         {"@type": "schema:Person", "schema:name": "Bar"}
///     ]
/// }"#;
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in JsonLdParser::new().for_slice(file.as_bytes()) {
///     let quad = quad?;
///     if quad.predicate == rdf::TYPE && quad.object == Term::from(schema_person.clone()) {
///         count += 1;
///     }
/// }
/// assert_eq!(count, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
#[must_use]
pub struct JsonLdParser {
    processing_mode: JsonLdProcessingMode,
    lenient: bool,
    profile: JsonLdProfileSet,
    base: Option<String>,
    rdf_direction: RdfDirection,
    produce_generalized_rdf: bool,
    max_nesting_depth: usize,
}

impl Default for JsonLdParser {
    fn default() -> Self {
        Self {
            processing_mode: JsonLdProcessingMode::default(),
            lenient: false,
            profile: JsonLdProfileSet::empty(),
            base: None,
            rdf_direction: RdfDirection::default(),
            produce_generalized_rdf: false,
            max_nesting_depth: DEFAULT_MAX_NESTING_DEPTH,
        }
    }
}

impl JsonLdParser {
    /// Builds a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Skips some validations to make parsing faster (may emit broken RDF on invalid input).
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Assumes the given profile(s) during parsing.
    ///
    /// [`JsonLdProfile::Streaming`](crate::JsonLdProfile::Streaming) additionally enforces the
    /// streaming document form's key-order constraints.
    pub fn with_profile(mut self, profile: impl Into<JsonLdProfileSet>) -> Self {
        self.profile = profile.into();
        self
    }

    /// Sets the maximum JSON nesting depth accepted from the input document.
    ///
    /// Every JSON-LD algorithm walks the document tree recursively, so an unbounded nesting depth
    /// would let a modestly sized untrusted document abort the process with a stack overflow.
    /// Defaults to [`DEFAULT_MAX_NESTING_DEPTH`].
    pub fn with_max_nesting_depth(mut self, max_nesting_depth: usize) -> Self {
        self.max_nesting_depth = max_nesting_depth;
        self
    }

    /// Sets the processing mode.
    pub fn with_processing_mode(mut self, processing_mode: JsonLdProcessingMode) -> Self {
        self.processing_mode = processing_mode;
        self
    }

    /// Selects how `@direction` is represented in the produced RDF.
    pub fn with_rdf_direction(mut self, rdf_direction: RdfDirection) -> Self {
        self.rdf_direction = rdf_direction;
        self
    }

    /// Emits generalized RDF.
    ///
    /// A blank node in predicate position cannot be represented by [`Quad`], so such statements are
    /// still dropped; the flag is accepted for API compatibility and governs the rest of the
    /// generalized-RDF behaviour.
    pub fn with_generalized_rdf(mut self) -> Self {
        self.produce_generalized_rdf = true;
        self
    }

    /// Sets the base IRI used when expanding the document.
    ///
    /// # Errors
    ///
    /// Returns [`IriParseError`] if `base_iri` is not a valid IRI.
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        let base_iri = base_iri.into();
        Iri::parse(base_iri.as_str())?;
        self.base = Some(base_iri);
        Ok(self)
    }

    /// Parses a JSON-LD document from a [`Read`] source.
    pub fn for_reader<R: Read>(self, mut reader: R) -> ReaderJsonLdParser<R> {
        let mut buffer = Vec::new();
        let read_result = reader.read_to_end(&mut buffer);
        ReaderJsonLdParser {
            inner: InnerParser::new(self, read_result.map(|_| buffer)),
            reader: PhantomData,
        }
    }

    /// Parses a JSON-LD document from a [`AsyncRead`] source.
    ///
    /// This is the asynchronous counterpart of [`for_reader`](Self::for_reader), and takes the
    /// same (non-`async`) shape so that the umbrella crates can forward to it uniformly. JSON-LD
    /// expansion needs the whole document in memory before any quad can be emitted, so the input
    /// is drained on the first [`next`](TokioAsyncReaderJsonLdParser::next) call — awaiting only
    /// the I/O — and then handed to the very same expansion/`toRdf` core the blocking path uses.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderJsonLdParser<R> {
        TokioAsyncReaderJsonLdParser {
            config: Some(self),
            inner: None,
            reader,
            loader: None,
        }
    }

    /// Parses a JSON-LD document from an in-memory slice.
    pub fn for_slice(self, slice: &(impl AsRef<[u8]> + ?Sized)) -> SliceJsonLdParser<'_> {
        SliceJsonLdParser {
            inner: InnerParser::new(self, Ok(slice.as_ref().to_vec())),
            slice: PhantomData,
        }
    }

    fn options<'a>(&'a self, loader: &'a dyn DocumentLoader) -> JsonLdOptions<'a> {
        JsonLdOptions {
            base: self.base.clone(),
            processing_mode: self.processing_mode,
            lenient: self.lenient,
            produce_generalized_rdf: self.produce_generalized_rdf,
            rdf_direction: self.rdf_direction,
            profile: self.profile,
            max_nesting_depth: self.max_nesting_depth,
            loader,
            ..JsonLdOptions::new()
        }
    }
}

/// The state both parser flavours share.
struct InnerParser {
    config: JsonLdParser,
    buffer: std::io::Result<Vec<u8>>,
    loader: Option<Box<dyn DocumentLoader + Send>>,
    quads: Option<vec::IntoIter<Quad>>,
    prefixes: Vec<(String, String)>,
    base_iri: Option<String>,
}

impl InnerParser {
    fn new(config: JsonLdParser, buffer: std::io::Result<Vec<u8>>) -> Self {
        let base_iri = config.base.clone();
        Self {
            config,
            buffer,
            loader: None,
            quads: None,
            prefixes: Vec::new(),
            base_iri,
        }
    }

    fn next_quad(&mut self) -> Option<Result<Quad, JsonLdParseError>> {
        if self.quads.is_none() {
            let bytes = match &self.buffer {
                Ok(bytes) => bytes.clone(),
                Err(e) => {
                    let error = std::io::Error::new(e.kind(), e.to_string());
                    self.quads = Some(Vec::new().into_iter());
                    return Some(Err(JsonLdParseError::Io(error)));
                }
            };
            match self.run(&bytes) {
                Ok(quads) => self.quads = Some(quads.into_iter()),
                Err(e) => {
                    self.quads = Some(Vec::new().into_iter());
                    return Some(Err(e));
                }
            }
        }
        self.quads.as_mut()?.next().map(Ok)
    }

    fn run(&mut self, bytes: &[u8]) -> Result<Vec<Quad>, JsonLdParseError> {
        let document =
            JsonValue::from_slice_with_max_nesting_depth(bytes, self.config.max_nesting_depth)
                .map_err(JsonLdSyntaxError::from)?;
        let default_loader = NoLoader;
        let loader = self.loader.as_deref().unwrap_or(&default_loader);
        let options = self.config.options(loader);
        let (prefixes, base_iri) = document_prefixes(&document, &options);
        self.prefixes = prefixes;
        if base_iri.is_some() {
            self.base_iri = base_iri;
        }
        to_rdf(document, &options).map_err(JsonLdParseError::from)
    }
}

/// Wraps a caller-supplied closure whose error type is the boxed one `oxjsonld` uses.
///
/// The `Send` bound is what keeps the parsers themselves `Send`; `oxjsonld` puts the same bound on
/// its own `LoadDocumentCallback`.
fn boxed_error_loader<F>(callback: F) -> impl DocumentLoader + Send + 'static
where
    F: Fn(&str, &LoadDocumentOptions) -> Result<RemoteDocument, Box<dyn Error + Send + Sync>>
        + Send
        + 'static,
{
    ClosureLoader::new(move |url: &str, options: &LoadDocumentOptions| {
        callback(url, options).map_err(|e| {
            LoadDocumentError::new(e.to_string(), JsonLdErrorCode::LoadingRemoteContextFailed)
                .with_source(e)
        })
    })
}

/// The iterator returned by [`JsonLdParser::for_reader`].
#[must_use]
pub struct ReaderJsonLdParser<R: Read> {
    inner: InnerParser,
    reader: PhantomData<R>,
}

/// The parser returned by [`JsonLdParser::for_tokio_async_reader`].
///
/// The document is drained from the reader lazily, on the first [`next`](Self::next) call, so
/// that construction can stay non-`async` like every other `for_tokio_async_reader` in the
/// workspace. Any loader installed with [`with_load_document_callback`](Self::with_load_document_callback)
/// or [`with_loader`](Self::with_loader) before that first call is honoured.
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncReaderJsonLdParser<R: AsyncRead + Unpin> {
    /// The pending configuration, taken when the document is drained. `None` afterwards.
    config: Option<JsonLdParser>,
    /// The running parse state, created once the document has been drained.
    inner: Option<InnerParser>,
    reader: R,
    loader: Option<Box<dyn DocumentLoader + Send>>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderJsonLdParser<R> {
    /// Reads the next quad or returns `None` if the document is finished.
    pub async fn next(&mut self) -> Option<Result<Quad, JsonLdParseError>> {
        if let Some(config) = self.config.take() {
            let mut buffer = Vec::new();
            let read_result = self.reader.read_to_end(&mut buffer).await;
            let mut inner = InnerParser::new(config, read_result.map(|_| buffer));
            inner.loader = self.loader.take();
            self.inner = Some(inner);
        }
        self.inner.as_mut()?.next_quad()
    }

    /// Registers a closure resolving remote contexts and documents.
    ///
    /// The default behaviour performs no I/O at all: without a callback, a document that
    /// references a remote `@context` fails cleanly.
    pub fn with_load_document_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, &LoadDocumentOptions) -> Result<RemoteDocument, Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    {
        self.set_loader(Box::new(boxed_error_loader(callback)));
        self
    }

    /// Installs a custom [`DocumentLoader`].
    ///
    /// The loader must be `Send` so that the parser itself stays `Send`, as `oxjsonld`'s is.
    pub fn with_loader(mut self, loader: impl DocumentLoader + Send + 'static) -> Self {
        self.set_loader(Box::new(loader));
        self
    }

    /// Stores `loader` wherever the parse state currently lives.
    fn set_loader(&mut self, loader: Box<dyn DocumentLoader + Send>) {
        if let Some(inner) = self.inner.as_mut() {
            inner.loader = Some(loader);
        } else {
            self.loader = Some(loader);
        }
    }

    /// The IRI prefixes declared by the document's top-level `@context`.
    ///
    /// The list is empty until the first quad has been produced, at which point the whole
    /// document has been processed and it is complete.
    pub fn prefixes(&self) -> JsonLdPrefixesIter<'_> {
        JsonLdPrefixesIter {
            prefixes: match self.inner.as_ref() {
                Some(inner) => inner.prefixes.iter(),
                None => [].iter(),
            },
        }
    }

    /// The base IRI in effect: the one given to the parser, then the document's `@base` once the
    /// document has been processed.
    #[must_use]
    pub fn base_iri(&self) -> Option<&str> {
        self.inner.as_ref()?.base_iri.as_deref()
    }
}

/// The iterator returned by [`JsonLdParser::for_slice`].
#[must_use]
pub struct SliceJsonLdParser<'a> {
    inner: InnerParser,
    slice: PhantomData<&'a [u8]>,
}

impl<R: Read> ReaderJsonLdParser<R> {
    /// Registers a closure resolving remote contexts and documents.
    ///
    /// The default behaviour performs no I/O at all: without a callback, a document that
    /// references a remote `@context` fails cleanly.
    pub fn with_load_document_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, &LoadDocumentOptions) -> Result<RemoteDocument, Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    {
        self.inner.loader = Some(Box::new(boxed_error_loader(callback)));
        self
    }

    /// Installs a custom [`DocumentLoader`].
    ///
    /// The loader must be `Send` so that the parser itself stays `Send`, as `oxjsonld`'s is.
    pub fn with_loader(mut self, loader: impl DocumentLoader + Send + 'static) -> Self {
        self.inner.loader = Some(Box::new(loader));
        self
    }

    /// The IRI prefixes declared by the document's top-level `@context`.
    ///
    /// The list is empty until the first quad has been produced, at which point the whole
    /// document has been processed and it is complete.
    pub fn prefixes(&self) -> JsonLdPrefixesIter<'_> {
        JsonLdPrefixesIter {
            prefixes: self.inner.prefixes.iter(),
        }
    }

    /// The base IRI in effect: the one given to the parser, then the document's `@base` once the
    /// document has been processed.
    #[must_use]
    pub fn base_iri(&self) -> Option<&str> {
        self.inner.base_iri.as_deref()
    }
}

impl SliceJsonLdParser<'_> {
    /// Registers a closure resolving remote contexts and documents.
    ///
    /// The default behaviour performs no I/O at all: without a callback, a document that
    /// references a remote `@context` fails cleanly.
    pub fn with_load_document_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, &LoadDocumentOptions) -> Result<RemoteDocument, Box<dyn Error + Send + Sync>>
            + Send
            + 'static,
    {
        self.inner.loader = Some(Box::new(boxed_error_loader(callback)));
        self
    }

    /// Installs a custom [`DocumentLoader`].
    ///
    /// The loader must be `Send` so that the parser itself stays `Send`, as `oxjsonld`'s is.
    pub fn with_loader(mut self, loader: impl DocumentLoader + Send + 'static) -> Self {
        self.inner.loader = Some(Box::new(loader));
        self
    }

    /// The IRI prefixes declared by the document's top-level `@context`.
    ///
    /// The list is empty until the first quad has been produced, at which point the whole
    /// document has been processed and it is complete.
    pub fn prefixes(&self) -> JsonLdPrefixesIter<'_> {
        JsonLdPrefixesIter {
            prefixes: self.inner.prefixes.iter(),
        }
    }

    /// The base IRI in effect: the one given to the parser, then the document's `@base` once the
    /// document has been processed.
    #[must_use]
    pub fn base_iri(&self) -> Option<&str> {
        self.inner.base_iri.as_deref()
    }
}

/// Iterator over the prefixes declared by a parsed document.
///
/// See [`SliceJsonLdParser::prefixes`].
pub struct JsonLdPrefixesIter<'a> {
    prefixes: slice::Iter<'a, (String, String)>,
}

impl<'a> Iterator for JsonLdPrefixesIter<'a> {
    type Item = (&'a str, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        let (prefix, iri) = self.prefixes.next()?;
        Some((prefix.as_str(), iri.as_str()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.prefixes.size_hint()
    }
}

impl<R: Read> Iterator for ReaderJsonLdParser<R> {
    type Item = Result<Quad, JsonLdParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_quad()
    }
}

impl Iterator for SliceJsonLdParser<'_> {
    type Item = Result<Quad, JsonLdParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_quad()
    }
}
