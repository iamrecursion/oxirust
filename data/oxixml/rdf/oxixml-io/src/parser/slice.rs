//! [`SliceQuadParser`]: parses an RDF document from an in-memory byte slice.

use super::RdfParserKind;
use super::prefixes::{PrefixesIter, PrefixesIterKind};
use crate::RdfFormat;
use crate::document::LoadedDocument;
use crate::error::RdfParseError;
use crate::mapper::QuadMapper;
use oxixml_jsonld::{RemoteDocument, SliceJsonLdParser};
use oxixml_rdfxml::SliceRdfXmlParser;
use oxixml_trix::SliceTriXParser;
use oxixml_turtle::n3::SliceN3Parser;
use oxixml_turtle::nquads::SliceNQuadsParser;
use oxixml_turtle::ntriples::SliceNTriplesParser;
use oxixml_turtle::trig::SliceTriGParser;
use oxixml_turtle::turtle::SliceTurtleParser;
use std::error::Error;
use std::panic::{RefUnwindSafe, UnwindSafe};

/// Builds a [`SliceQuadParser`] from an already-configured [`RdfParserKind`].
pub(super) fn build<'a>(
    kind: RdfParserKind,
    slice: &'a [u8],
    mapper: QuadMapper,
) -> SliceQuadParser<'a> {
    SliceQuadParser {
        inner: match kind {
            RdfParserKind::JsonLd(p, _) => SliceQuadParserKind::JsonLd(p.for_slice(slice)),
            RdfParserKind::N3(p) => SliceQuadParserKind::N3(p.for_slice(slice)),
            RdfParserKind::NQuads(p) => SliceQuadParserKind::NQuads(p.for_slice(slice)),
            RdfParserKind::NTriples(p) => SliceQuadParserKind::NTriples(p.for_slice(slice)),
            RdfParserKind::RdfXml(p) => SliceQuadParserKind::RdfXml(p.for_slice(slice)),
            RdfParserKind::TriG(p) => SliceQuadParserKind::TriG(p.for_slice(slice)),
            RdfParserKind::TriX(p) => SliceQuadParserKind::TriX(p.for_slice(slice)),
            RdfParserKind::Turtle(p) => SliceQuadParserKind::Turtle(p.for_slice(slice)),
        },
        mapper,
    }
}

/// Wraps one already-built N-Triples chunk parser (see
/// [`RdfParser::split_slice_for_parallel_parsing`](crate::RdfParser::split_slice_for_parallel_parsing)).
pub(super) fn from_ntriples(
    parser: SliceNTriplesParser<'_>,
    mapper: QuadMapper,
) -> SliceQuadParser<'_> {
    SliceQuadParser {
        inner: SliceQuadParserKind::NTriples(parser),
        mapper,
    }
}

/// Wraps one already-built N-Quads chunk parser.
pub(super) fn from_nquads(
    parser: SliceNQuadsParser<'_>,
    mapper: QuadMapper,
) -> SliceQuadParser<'_> {
    SliceQuadParser {
        inner: SliceQuadParserKind::NQuads(parser),
        mapper,
    }
}

/// Parses an RDF document from a byte slice, yielding [`Quad`](oxixml_model::Quad)s.
///
/// Can be built using [`RdfParser::for_slice`](crate::RdfParser::for_slice).
///
/// Unlike `oxrdfio`'s equivalent, this iterator's `Item` is `Result<Quad, `[`RdfParseError`]`>`
/// rather than a syntax-only error: two of the four backends (JSON-LD, TriX) already report their
/// own unified parse-error type for slice parsing, so unifying on [`RdfParseError`] here avoids a
/// lossy, effectively-dead-code conversion for those two formats.
///
/// ```
/// use oxixml_io::{RdfFormat, RdfParser};
///
/// let file = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .";
///
/// let quads = RdfParser::from_format(RdfFormat::NTriples)
///     .for_slice(file)
///     .collect::<Result<Vec<_>, _>>()?;
///
/// assert_eq!(quads.len(), 1);
/// assert_eq!(quads[0].subject.to_string(), "<http://example.com/s>");
/// # Result::<(), oxixml_io::RdfParseError>::Ok(())
/// ```
#[must_use]
pub struct SliceQuadParser<'a> {
    inner: SliceQuadParserKind<'a>,
    mapper: QuadMapper,
}

enum SliceQuadParserKind<'a> {
    JsonLd(SliceJsonLdParser<'a>),
    N3(SliceN3Parser<'a>),
    NQuads(SliceNQuadsParser<'a>),
    NTriples(SliceNTriplesParser<'a>),
    RdfXml(SliceRdfXmlParser<'a>),
    TriG(SliceTriGParser<'a>),
    TriX(SliceTriXParser<'a>),
    Turtle(SliceTurtleParser<'a>),
}

impl Iterator for SliceQuadParser<'_> {
    type Item = Result<oxixml_model::Quad, RdfParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(match &mut self.inner {
            SliceQuadParserKind::JsonLd(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            SliceQuadParserKind::N3(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_n3_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            SliceQuadParserKind::NQuads(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            SliceQuadParserKind::NTriples(parser) => match parser.next()? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
            SliceQuadParserKind::RdfXml(parser) => match parser.next()? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
            SliceQuadParserKind::TriG(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            SliceQuadParserKind::TriX(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            SliceQuadParserKind::Turtle(parser) => match parser.next()? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
        })
    }
}

impl SliceQuadParser<'_> {
    /// The `(prefix name, prefix IRI)` pairs declared so far by the parse.
    ///
    /// See [`ReaderQuadParser::prefixes`](crate::ReaderQuadParser::prefixes).
    #[must_use]
    pub fn prefixes(&self) -> PrefixesIter<'_> {
        PrefixesIter {
            inner: match &self.inner {
                SliceQuadParserKind::JsonLd(p) => PrefixesIterKind::JsonLd(p.prefixes()),
                SliceQuadParserKind::N3(p) => PrefixesIterKind::N3(p.prefixes()),
                SliceQuadParserKind::RdfXml(p) => PrefixesIterKind::RdfXml(p.prefixes()),
                SliceQuadParserKind::TriG(p) => PrefixesIterKind::TriG(p.prefixes()),
                SliceQuadParserKind::Turtle(p) => PrefixesIterKind::Turtle(p.prefixes()),
                SliceQuadParserKind::NQuads(_)
                | SliceQuadParserKind::NTriples(_)
                | SliceQuadParserKind::TriX(_) => PrefixesIterKind::None,
            },
        }
    }

    /// The base IRI in effect at the current step of the parse.
    ///
    /// See [`ReaderQuadParser::base_iri`](crate::ReaderQuadParser::base_iri).
    #[must_use]
    pub fn base_iri(&self) -> Option<&str> {
        match &self.inner {
            SliceQuadParserKind::JsonLd(p) => p.base_iri(),
            SliceQuadParserKind::N3(p) => p.base_iri(),
            SliceQuadParserKind::RdfXml(p) => p.base_iri(),
            SliceQuadParserKind::TriG(p) => p.base_iri(),
            SliceQuadParserKind::Turtle(p) => p.base_iri(),
            SliceQuadParserKind::NQuads(_)
            | SliceQuadParserKind::NTriples(_)
            | SliceQuadParserKind::TriX(_) => None,
        }
    }

    /// Registers a callback to resolve remote documents during parsing.
    ///
    /// See [`ReaderQuadParser::with_document_loader`](crate::ReaderQuadParser::with_document_loader).
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
            SliceQuadParserKind::JsonLd(p) => {
                SliceQuadParserKind::JsonLd(p.with_load_document_callback(move |iri, _options| {
                    let response = loader(iri)?;
                    if !matches!(response.format, RdfFormat::JsonLd { .. }) {
                        return Err(format!(
                            "the JSON-LD context format must be JSON-LD, {} found",
                            response.format
                        )
                        .into());
                    }
                    Ok(RemoteDocument::new(response.content, response.url))
                }))
            }
            other => other,
        };
        self
    }
}
