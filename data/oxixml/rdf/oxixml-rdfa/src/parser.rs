//! The public parser: a builder, and the iterators it produces.

use std::io::Read;
use std::vec;

use oxixml_dom::{Document, build};
use oxixml_iri::Iri;
use oxixml_model::Triple;

use crate::error::{RdfaParseError, RdfaSyntaxError};
use crate::eval::Processor;
use crate::host::HostLanguage;

/// The base IRI used when the caller supplies none.
///
/// RDFa resolves every relative reference — including the empty one that `@about=""` produces —
/// against the base, so there has to be one. A document parsed out of a byte slice has no address
/// of its own, so this stands in for it and makes the failure mode obvious in the output.
const DEFAULT_BASE: &str = "http://example.com/";

/// A builder for RDFa extraction.
///
/// ```
/// use oxixml_rdfa::RdfaParser;
///
/// let html = r##"<!DOCTYPE html>
/// <html>
///   <body>
///     <p about="http://example.com/photo.jpg" property="http://purl.org/dc/terms/creator">
///       Mark Birbeck
///     </p>
///   </body>
/// </html>"##;
///
/// let triples = RdfaParser::new()
///     .with_base_iri("http://example.com/doc")?
///     .for_slice(html)
///     .collect::<Result<Vec<_>, _>>()?;
/// assert_eq!(triples.len(), 1);
/// assert_eq!(triples[0].subject.to_string(), "<http://example.com/photo.jpg>");
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct RdfaParser {
    base_iri: Option<Iri<String>>,
    host: HostLanguage,
    max_depth: usize,
}

impl Default for RdfaParser {
    fn default() -> Self {
        Self {
            base_iri: None,
            host: HostLanguage::default(),
            max_depth: Self::DEFAULT_MAX_DEPTH,
        }
    }
}

impl RdfaParser {
    /// The default ceiling on how many elements deep [`for_document`](Self::for_document) (and
    /// the parsing entry points built on it) walks the document tree, and — independently,
    /// restarting the count at each one — how many elements deep it walks an
    /// `rdf:XMLLiteral`/`rdf:HTML` literal's descendants.
    ///
    /// Extraction recurses with one native stack frame per level of document nesting, so an
    /// unbounded document would otherwise turn an untrusted, modestly-sized input into a
    /// process-aborting stack overflow. See [`with_max_depth`](Self::with_max_depth) to change it.
    pub const DEFAULT_MAX_DEPTH: usize = 256;

    /// Builds a parser with no base IRI and the [`HostLanguage::Html5`] processing rules.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the IRI relative references are resolved against.
    ///
    /// A `<base href="…">` element, and — outside `text/html` — an `@xml:base` attribute, override
    /// it while processing.
    ///
    /// # Errors
    ///
    /// Returns [`RdfaSyntaxError`] when `base_iri` is not an absolute IRI.
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, RdfaSyntaxError> {
        let base_iri = base_iri.into();
        self.base_iri = Some(Iri::parse(base_iri.clone()).map_err(|error| {
            RdfaSyntaxError::msg(format!("invalid base IRI {base_iri}: {error}"))
        })?);
        Ok(self)
    }

    /// Selects the host language, which decides how the base and the natural language are spelled,
    /// which initial contexts are loaded, and whether the HTML+RDFa extensions apply.
    pub fn with_host_language(mut self, host: HostLanguage) -> Self {
        self.host = host;
        self
    }

    /// The host language this parser will use.
    pub const fn host_language(&self) -> HostLanguage {
        self.host
    }

    /// Sets how many elements deep extraction walks the document tree (and, independently, an
    /// `rdf:XMLLiteral`/`rdf:HTML` literal's descendants) before giving up rather than recursing
    /// further. Defaults to [`DEFAULT_MAX_DEPTH`](Self::DEFAULT_MAX_DEPTH).
    ///
    /// A document nested deeper than this fails extraction with [`RdfaSyntaxError`] instead of
    /// being processed, so raise it only for a source you trust to also raise the process stack
    /// size accordingly. The default assumes a normal process main-thread stack; conversely,
    /// *lower* it when calling into this crate from a thread you know has a smaller one (a
    /// worker pool with a small configured `stack_size`, for instance), since the default is not
    /// calibrated for that.
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// The nesting-depth limit this parser will enforce; see
    /// [`with_max_depth`](Self::with_max_depth).
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    fn base(&self) -> Iri<String> {
        self.base_iri
            .clone()
            .unwrap_or_else(|| Iri::parse_unchecked(DEFAULT_BASE.to_owned()))
    }

    /// Extracts RDFa from an already-parsed document tree.
    ///
    /// This is the entry point to use when the document came from somewhere else — an
    /// [`oxixml_html`] parse whose parse errors you want to inspect, an
    /// [`oxixml_dom::build::parse_str`] of XHTML, or a tree you built yourself.
    ///
    /// ```
    /// use oxixml_dom::build;
    /// use oxixml_rdfa::{HostLanguage, RdfaParser};
    ///
    /// let document = build::parse_str(
    ///     r#"<root xmlns:dc="http://purl.org/dc/terms/">
    ///          <item about="urn:example:1" property="dc:title">Kokoro</item>
    ///        </root>"#,
    /// )?;
    /// let triples: Vec<_> = RdfaParser::new()
    ///     .with_host_language(HostLanguage::Xml)
    ///     .for_document(&document)
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// assert_eq!(triples.len(), 1);
    /// assert_eq!(triples[0].predicate.as_str(), "http://purl.org/dc/terms/title");
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_document(&self, document: &Document) -> RdfaTriples {
        match Processor::extract(document, self.host, self.base(), self.max_depth) {
            Ok(triples) => RdfaTriples {
                error: None,
                inner: triples.into_iter(),
            },
            Err(error) => RdfaTriples::failed(error.into()),
        }
    }

    /// Parses `data` and extracts RDFa from it.
    ///
    /// The markup parser follows the host language: [`HostLanguage::Html5`] uses
    /// [`oxixml_html::parse_document`], and every other variant uses the XML parser, since XHTML
    /// and XHTML5 are XML.
    pub fn for_slice(&self, data: impl AsRef<str>) -> RdfaTriples {
        match self.parse(data.as_ref()) {
            Ok(document) => self.for_document(&document),
            Err(error) => RdfaTriples::failed(error),
        }
    }

    /// Reads `reader` to the end, parses it, and extracts RDFa from it.
    ///
    /// # Errors
    ///
    /// The returned iterator yields a single [`RdfaParseError`] when reading or parsing fails.
    pub fn for_reader(&self, mut reader: impl Read) -> RdfaTriples {
        let mut data = String::new();
        match reader.read_to_string(&mut data) {
            Ok(_) => self.for_slice(data),
            Err(error) => RdfaTriples::failed(error.into()),
        }
    }

    fn parse(&self, data: &str) -> Result<Document, RdfaParseError> {
        if self.host == HostLanguage::Html5 {
            return Ok(oxixml_html::parse_document(data).into_document());
        }
        build::parse_str(data).map_err(|error| RdfaSyntaxError::msg(error.to_string()).into())
    }
}

/// The triples of one document, in the order the processing model generated them and with
/// duplicates removed.
///
/// The whole graph is materialized before the first item is yielded: RDFa is not a streaming
/// syntax — a `@rel` high in the tree can only be completed by a subject found much later, and
/// property copying is defined on the finished graph.
#[derive(Debug)]
pub struct RdfaTriples {
    /// A failure that happened before any triple could be produced; yielded once, first.
    error: Option<RdfaParseError>,
    inner: vec::IntoIter<Triple>,
}

impl RdfaTriples {
    fn failed(error: RdfaParseError) -> Self {
        Self {
            error: Some(error),
            inner: Vec::new().into_iter(),
        }
    }
}

impl Iterator for RdfaTriples {
    type Item = Result<Triple, RdfaParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.error.take() {
            return Some(Err(error));
        }
        self.inner.next().map(Ok)
    }
}
