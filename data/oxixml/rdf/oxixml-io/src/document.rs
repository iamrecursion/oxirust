//! [`LoadedDocument`]: the payload returned by a caller-supplied remote-document loader.

use crate::RdfFormat;

/// A remote document fetched to support parsing.
///
/// This is mostly used to retrieve JSON-LD remote `@context` documents through
/// [`RdfParser::for_reader`](crate::RdfParser::for_reader) /
/// [`RdfParser::for_slice`](crate::RdfParser::for_slice) followed by
/// [`with_document_loader`](crate::ReaderQuadParser::with_document_loader). `oxixml-io` performs
/// **no** I/O of its own: the caller's closure is entirely responsible for fetching `content`,
/// whether from the network, a local cache, or a bundled resource map.
#[derive(Debug, Clone)]
pub struct LoadedDocument {
    /// The final URL of the remote document, after any redirections or normalizations.
    pub url: String,
    /// The raw bytes of the document.
    pub content: Vec<u8>,
    /// The format of the document.
    ///
    /// Only [`RdfFormat::JsonLd`] is currently accepted: a JSON-LD remote `@context` must itself
    /// be JSON-LD.
    pub format: RdfFormat,
}

impl LoadedDocument {
    /// Builds a [`LoadedDocument`].
    #[must_use]
    pub fn new(url: impl Into<String>, content: impl Into<Vec<u8>>, format: RdfFormat) -> Self {
        Self {
            url: url.into(),
            content: content.into(),
            format,
        }
    }
}
