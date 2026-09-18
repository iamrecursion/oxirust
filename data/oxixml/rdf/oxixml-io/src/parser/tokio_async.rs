//! [`TokioAsyncReaderQuadParser`]: parses an RDF document from a [`AsyncRead`] implementation.
//!
//! The asynchronous mirror of [`super::reader`]: identical dispatch over the same per-format
//! parsers, differing only in that every read is awaited. It exists behind the `async-tokio`
//! feature and mirrors `oxrdfio`'s `TokioAsyncReaderQuadParser`.

use super::RdfParserKind;
use super::prefixes::{PrefixesIter, PrefixesIterKind};
use crate::RdfFormat;
use crate::document::LoadedDocument;
use crate::error::RdfParseError;
use crate::mapper::QuadMapper;
use oxixml_jsonld::{RemoteDocument, TokioAsyncReaderJsonLdParser};
use oxixml_model::Quad;
use oxixml_rdfxml::TokioAsyncReaderRdfXmlParser;
use oxixml_trix::TokioAsyncReaderTriXParser;
use oxixml_turtle::n3::TokioAsyncReaderN3Parser;
use oxixml_turtle::nquads::TokioAsyncReaderNQuadsParser;
use oxixml_turtle::ntriples::TokioAsyncReaderNTriplesParser;
use oxixml_turtle::trig::TokioAsyncReaderTriGParser;
use oxixml_turtle::turtle::TokioAsyncReaderTurtleParser;
use std::error::Error;
use std::panic::{RefUnwindSafe, UnwindSafe};
use tokio::io::AsyncRead;

/// Builds a [`TokioAsyncReaderQuadParser`] from an already-configured [`RdfParserKind`].
pub(super) fn build<R: AsyncRead + Unpin>(
    kind: RdfParserKind,
    reader: R,
    mapper: QuadMapper,
) -> TokioAsyncReaderQuadParser<R> {
    TokioAsyncReaderQuadParser {
        inner: match kind {
            RdfParserKind::JsonLd(p, _) => {
                TokioAsyncReaderQuadParserKind::JsonLd(p.for_tokio_async_reader(reader))
            }
            RdfParserKind::N3(p) => {
                TokioAsyncReaderQuadParserKind::N3(p.for_tokio_async_reader(reader))
            }
            RdfParserKind::NQuads(p) => {
                TokioAsyncReaderQuadParserKind::NQuads(p.for_tokio_async_reader(reader))
            }
            RdfParserKind::NTriples(p) => {
                TokioAsyncReaderQuadParserKind::NTriples(p.for_tokio_async_reader(reader))
            }
            RdfParserKind::RdfXml(p) => {
                TokioAsyncReaderQuadParserKind::RdfXml(p.for_tokio_async_reader(reader))
            }
            RdfParserKind::TriG(p) => {
                TokioAsyncReaderQuadParserKind::TriG(p.for_tokio_async_reader(reader))
            }
            RdfParserKind::TriX(p) => {
                TokioAsyncReaderQuadParserKind::TriX(p.for_tokio_async_reader(reader))
            }
            RdfParserKind::Turtle(p) => {
                TokioAsyncReaderQuadParserKind::Turtle(p.for_tokio_async_reader(reader))
            }
        },
        mapper,
    }
}

/// Parses an RDF document from a [`AsyncRead`] implementation, yielding [`Quad`]s.
///
/// Can be built using
/// [`RdfParser::for_tokio_async_reader`](crate::RdfParser::for_tokio_async_reader).
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), oxixml_io::RdfParseError> {
/// use oxixml_io::{RdfFormat, RdfParser};
///
/// let file = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .";
///
/// let mut parser =
///     RdfParser::from_format(RdfFormat::NTriples).for_tokio_async_reader(file.as_bytes());
/// if let Some(quad) = parser.next().await {
///     assert_eq!(quad?.subject.to_string(), "<http://example.com/s>");
/// }
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct TokioAsyncReaderQuadParser<R: AsyncRead + Unpin> {
    inner: TokioAsyncReaderQuadParserKind<R>,
    mapper: QuadMapper,
}

enum TokioAsyncReaderQuadParserKind<R: AsyncRead + Unpin> {
    JsonLd(TokioAsyncReaderJsonLdParser<R>),
    N3(TokioAsyncReaderN3Parser<R>),
    NQuads(TokioAsyncReaderNQuadsParser<R>),
    NTriples(TokioAsyncReaderNTriplesParser<R>),
    RdfXml(TokioAsyncReaderRdfXmlParser<R>),
    TriG(TokioAsyncReaderTriGParser<R>),
    TriX(TokioAsyncReaderTriXParser<R>),
    Turtle(TokioAsyncReaderTurtleParser<R>),
}

impl<R: AsyncRead + Unpin> TokioAsyncReaderQuadParser<R> {
    /// Reads the next quad or returns `None` if the document is finished.
    pub async fn next(&mut self) -> Option<Result<Quad, RdfParseError>> {
        Some(match &mut self.inner {
            TokioAsyncReaderQuadParserKind::JsonLd(parser) => match parser.next().await? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            TokioAsyncReaderQuadParserKind::N3(parser) => match parser.next().await? {
                Ok(quad) => self.mapper.map_n3_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            TokioAsyncReaderQuadParserKind::NQuads(parser) => match parser.next().await? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            TokioAsyncReaderQuadParserKind::NTriples(parser) => match parser.next().await? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
            TokioAsyncReaderQuadParserKind::RdfXml(parser) => match parser.next().await? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
            TokioAsyncReaderQuadParserKind::TriG(parser) => match parser.next().await? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            TokioAsyncReaderQuadParserKind::TriX(parser) => match parser.next().await? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            TokioAsyncReaderQuadParserKind::Turtle(parser) => match parser.next().await? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
        })
    }

    /// The `(prefix name, prefix IRI)` pairs declared so far by the parse.
    ///
    /// Empty at the start, and grows as prefix declarations are read. Always empty for N-Quads,
    /// N-Triples, and TriX, none of which has a notion of a namespace prefix.
    #[must_use]
    pub fn prefixes(&self) -> PrefixesIter<'_> {
        PrefixesIter {
            inner: match &self.inner {
                TokioAsyncReaderQuadParserKind::JsonLd(p) => PrefixesIterKind::JsonLd(p.prefixes()),
                TokioAsyncReaderQuadParserKind::N3(p) => PrefixesIterKind::N3(p.prefixes()),
                TokioAsyncReaderQuadParserKind::RdfXml(p) => PrefixesIterKind::RdfXml(p.prefixes()),
                TokioAsyncReaderQuadParserKind::TriG(p) => PrefixesIterKind::TriG(p.prefixes()),
                TokioAsyncReaderQuadParserKind::Turtle(p) => PrefixesIterKind::Turtle(p.prefixes()),
                TokioAsyncReaderQuadParserKind::NQuads(_)
                | TokioAsyncReaderQuadParserKind::NTriples(_)
                | TokioAsyncReaderQuadParserKind::TriX(_) => PrefixesIterKind::None,
            },
        }
    }

    /// The base IRI in effect at the current step of the parse.
    ///
    /// `None` before any `@base`/`xml:base` has been read, and always `None` for N-Quads,
    /// N-Triples, and TriX, none of which has a notion of a base IRI.
    #[must_use]
    pub fn base_iri(&self) -> Option<&str> {
        match &self.inner {
            TokioAsyncReaderQuadParserKind::JsonLd(p) => p.base_iri(),
            TokioAsyncReaderQuadParserKind::N3(p) => p.base_iri(),
            TokioAsyncReaderQuadParserKind::RdfXml(p) => p.base_iri(),
            TokioAsyncReaderQuadParserKind::TriG(p) => p.base_iri(),
            TokioAsyncReaderQuadParserKind::Turtle(p) => p.base_iri(),
            TokioAsyncReaderQuadParserKind::NQuads(_)
            | TokioAsyncReaderQuadParserKind::NTriples(_)
            | TokioAsyncReaderQuadParserKind::TriX(_) => None,
        }
    }

    /// Registers a callback to resolve remote documents during parsing.
    ///
    /// Behaves exactly as
    /// [`ReaderQuadParser::with_document_loader`](crate::ReaderQuadParser::with_document_loader):
    /// it is only used to load JSON-LD remote `@context`s, and is a no-op for every other format.
    /// The callback itself is synchronous, since resolution happens inside the JSON-LD context
    /// algorithms rather than on the parser's I/O path.
    pub fn with_document_loader(
        mut self,
        loader: impl Fn(&str) -> Result<LoadedDocument, Box<dyn Error + Send + Sync>>
        + Send
        + Sync
        + UnwindSafe
        + RefUnwindSafe
        + 'static,
    ) -> Self {
        self.inner = match self.inner {
            TokioAsyncReaderQuadParserKind::JsonLd(p) => TokioAsyncReaderQuadParserKind::JsonLd(
                p.with_load_document_callback(move |iri, _options| {
                    let response = loader(iri)?;
                    if !matches!(response.format, RdfFormat::JsonLd { .. }) {
                        return Err(format!(
                            "the JSON-LD context format must be JSON-LD, {} found",
                            response.format
                        )
                        .into());
                    }
                    Ok(RemoteDocument::new(response.content, response.url))
                }),
            ),
            other => other,
        };
        self
    }
}
