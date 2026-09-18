//! [`ReaderQuadParser`]: parses an RDF document from a [`Read`] implementation.

use super::RdfParserKind;
use super::prefixes::{PrefixesIter, PrefixesIterKind};
use crate::RdfFormat;
use crate::document::LoadedDocument;
use crate::error::RdfParseError;
use crate::mapper::QuadMapper;
use oxixml_jsonld::{ReaderJsonLdParser, RemoteDocument};
use oxixml_model::Quad;
use oxixml_rdfxml::ReaderRdfXmlParser;
use oxixml_trix::ReaderTriXParser;
use oxixml_turtle::n3::ReaderN3Parser;
use oxixml_turtle::nquads::ReaderNQuadsParser;
use oxixml_turtle::ntriples::ReaderNTriplesParser;
use oxixml_turtle::trig::ReaderTriGParser;
use oxixml_turtle::turtle::ReaderTurtleParser;
use std::error::Error;
use std::io::Read;
use std::panic::{RefUnwindSafe, UnwindSafe};

/// Builds a [`ReaderQuadParser`] from an already-configured [`RdfParserKind`].
pub(super) fn build<R: Read>(
    kind: RdfParserKind,
    reader: R,
    mapper: QuadMapper,
) -> ReaderQuadParser<R> {
    ReaderQuadParser {
        inner: match kind {
            RdfParserKind::JsonLd(p, _) => ReaderQuadParserKind::JsonLd(p.for_reader(reader)),
            RdfParserKind::N3(p) => ReaderQuadParserKind::N3(p.for_reader(reader)),
            RdfParserKind::NQuads(p) => ReaderQuadParserKind::NQuads(p.for_reader(reader)),
            RdfParserKind::NTriples(p) => ReaderQuadParserKind::NTriples(p.for_reader(reader)),
            RdfParserKind::RdfXml(p) => ReaderQuadParserKind::RdfXml(p.for_reader(reader)),
            RdfParserKind::TriG(p) => ReaderQuadParserKind::TriG(p.for_reader(reader)),
            RdfParserKind::TriX(p) => ReaderQuadParserKind::TriX(p.for_reader(reader)),
            RdfParserKind::Turtle(p) => ReaderQuadParserKind::Turtle(p.for_reader(reader)),
        },
        mapper,
    }
}

/// Wraps one already-built N-Triples chunk parser (see
/// [`RdfParser::split_file_for_parallel_parsing`](crate::RdfParser::split_file_for_parallel_parsing)).
pub(super) fn from_ntriples<R: Read>(
    parser: ReaderNTriplesParser<R>,
    mapper: QuadMapper,
) -> ReaderQuadParser<R> {
    ReaderQuadParser {
        inner: ReaderQuadParserKind::NTriples(parser),
        mapper,
    }
}

/// Wraps one already-built N-Quads chunk parser.
pub(super) fn from_nquads<R: Read>(
    parser: ReaderNQuadsParser<R>,
    mapper: QuadMapper,
) -> ReaderQuadParser<R> {
    ReaderQuadParser {
        inner: ReaderQuadParserKind::NQuads(parser),
        mapper,
    }
}

/// Parses an RDF document from a [`Read`] implementation, yielding [`Quad`]s.
///
/// Can be built using [`RdfParser::for_reader`](crate::RdfParser::for_reader). Reads are
/// buffered internally.
///
/// ```
/// use oxixml_io::{RdfFormat, RdfParser};
///
/// let file = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .";
///
/// let quads = RdfParser::from_format(RdfFormat::NTriples)
///     .for_reader(file.as_bytes())
///     .collect::<Result<Vec<_>, _>>()?;
///
/// assert_eq!(quads.len(), 1);
/// assert_eq!(quads[0].subject.to_string(), "<http://example.com/s>");
/// # Result::<(), oxixml_io::RdfParseError>::Ok(())
/// ```
#[must_use]
pub struct ReaderQuadParser<R: Read> {
    inner: ReaderQuadParserKind<R>,
    mapper: QuadMapper,
}

enum ReaderQuadParserKind<R: Read> {
    JsonLd(ReaderJsonLdParser<R>),
    N3(ReaderN3Parser<R>),
    NQuads(ReaderNQuadsParser<R>),
    NTriples(ReaderNTriplesParser<R>),
    RdfXml(ReaderRdfXmlParser<R>),
    TriG(ReaderTriGParser<R>),
    TriX(ReaderTriXParser<R>),
    Turtle(ReaderTurtleParser<R>),
}

impl<R: Read> Iterator for ReaderQuadParser<R> {
    type Item = Result<Quad, RdfParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(match &mut self.inner {
            ReaderQuadParserKind::JsonLd(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            ReaderQuadParserKind::N3(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_n3_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            ReaderQuadParserKind::NQuads(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            ReaderQuadParserKind::NTriples(parser) => match parser.next()? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
            ReaderQuadParserKind::RdfXml(parser) => match parser.next()? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
            ReaderQuadParserKind::TriG(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            ReaderQuadParserKind::TriX(parser) => match parser.next()? {
                Ok(quad) => self.mapper.map_quad(quad).map_err(Into::into),
                Err(error) => Err(error.into()),
            },
            ReaderQuadParserKind::Turtle(parser) => match parser.next()? {
                Ok(triple) => Ok(self.mapper.map_triple_to_quad(triple)),
                Err(error) => Err(error.into()),
            },
        })
    }
}

impl<R: Read> ReaderQuadParser<R> {
    /// The `(prefix name, prefix IRI)` pairs declared so far by the parse.
    ///
    /// Empty at the start, and grows as prefix declarations are read. Always empty for N-Quads,
    /// N-Triples, and TriX, none of which has a notion of a namespace prefix.
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = RdfParser::from_format(RdfFormat::Turtle).for_reader(file.as_bytes());
    /// assert!(parser.prefixes().next().is_none()); // No prefix at the beginning.
    ///
    /// parser.next().expect("one quad")?; // We read the first triple.
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[must_use]
    pub fn prefixes(&self) -> PrefixesIter<'_> {
        PrefixesIter {
            inner: match &self.inner {
                ReaderQuadParserKind::JsonLd(p) => PrefixesIterKind::JsonLd(p.prefixes()),
                ReaderQuadParserKind::N3(p) => PrefixesIterKind::N3(p.prefixes()),
                ReaderQuadParserKind::RdfXml(p) => PrefixesIterKind::RdfXml(p.prefixes()),
                ReaderQuadParserKind::TriG(p) => PrefixesIterKind::TriG(p.prefixes()),
                ReaderQuadParserKind::Turtle(p) => PrefixesIterKind::Turtle(p.prefixes()),
                ReaderQuadParserKind::NQuads(_)
                | ReaderQuadParserKind::NTriples(_)
                | ReaderQuadParserKind::TriX(_) => PrefixesIterKind::None,
            },
        }
    }

    /// The base IRI in effect at the current step of the parse.
    ///
    /// `None` before any `@base`/`xml:base` has been read, and always `None` for N-Quads,
    /// N-Triples, and TriX, none of which has a notion of a base IRI.
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// let file = r#"@base <http://example.com/> .
    /// <foo> <http://example.com/p> <http://example.com/o> ."#;
    ///
    /// let mut parser = RdfParser::from_format(RdfFormat::Turtle).for_reader(file.as_bytes());
    /// assert!(parser.base_iri().is_none());
    ///
    /// parser.next().expect("one quad")?;
    /// assert_eq!(parser.base_iri(), Some("http://example.com/"));
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[must_use]
    pub fn base_iri(&self) -> Option<&str> {
        match &self.inner {
            ReaderQuadParserKind::JsonLd(p) => p.base_iri(),
            ReaderQuadParserKind::N3(p) => p.base_iri(),
            ReaderQuadParserKind::RdfXml(p) => p.base_iri(),
            ReaderQuadParserKind::TriG(p) => p.base_iri(),
            ReaderQuadParserKind::Turtle(p) => p.base_iri(),
            ReaderQuadParserKind::NQuads(_)
            | ReaderQuadParserKind::NTriples(_)
            | ReaderQuadParserKind::TriX(_) => None,
        }
    }

    /// Registers a callback to resolve remote documents during parsing.
    ///
    /// Currently only used to load JSON-LD remote `@context`s; a no-op for every other format.
    /// Without a callback, a JSON-LD document that references a remote context fails cleanly
    /// rather than the parser silently reaching out to the network — `oxixml-io` performs no I/O
    /// beyond the `reader` it was given.
    ///
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_io::{JsonLdProfileSet, LoadedDocument, RdfFormat, RdfParser};
    ///
    /// let file = r#"{
    ///     "@context": "file://context.jsonld",
    ///     "@type": "schema:Person",
    ///     "@id": "http://example.com/foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in RdfParser::from_format(RdfFormat::JsonLd { profile: JsonLdProfileSet::empty() })
    ///     .for_reader(file.as_bytes())
    ///     .with_document_loader(|url| {
    ///         assert_eq!(url, "file://context.jsonld");
    ///         Ok(LoadedDocument::new(
    ///             "file://context.jsonld",
    ///             br#"{"@context":{"schema": "http://schema.org/"}}"#.to_vec(),
    ///             RdfFormat::JsonLd { profile: JsonLdProfileSet::empty() },
    ///         ))
    ///     })
    /// {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(1, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
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
            ReaderQuadParserKind::JsonLd(p) => {
                ReaderQuadParserKind::JsonLd(p.with_load_document_callback(move |iri, _options| {
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
