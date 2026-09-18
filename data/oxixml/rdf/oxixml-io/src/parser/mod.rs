//! [`RdfParser`]: a format-agnostic builder that wraps every backend parser
//! ([`oxixml-turtle`](oxixml_turtle), [`oxixml-rdfxml`](oxixml_rdfxml),
//! [`oxixml-jsonld`](oxixml_jsonld), [`oxixml-trix`](oxixml_trix)) behind one API.

mod prefixes;
mod reader;
mod slice;
#[cfg(feature = "async-tokio")]
mod tokio_async;

pub use prefixes::PrefixesIter;
pub use reader::ReaderQuadParser;
pub use slice::SliceQuadParser;
#[cfg(feature = "async-tokio")]
pub use tokio_async::TokioAsyncReaderQuadParser;

use crate::RdfFormat;
use crate::mapper::QuadMapper;
use oxixml_jsonld::{JsonLdParser, JsonLdProfileSet};
use oxixml_model::{GraphName, IriParseError};
use oxixml_rdfxml::RdfXmlParser;
use oxixml_trix::TriXParser;
use oxixml_turtle::n3::N3Parser;
use oxixml_turtle::nquads::NQuadsParser;
use oxixml_turtle::ntriples::NTriplesParser;
use oxixml_turtle::trig::TriGParser;
use oxixml_turtle::turtle::TurtleParser;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Take};
use std::path::Path;
#[cfg(feature = "async-tokio")]
use tokio::io::AsyncRead;

/// A format-agnostic builder for parsing RDF into an iterator of
/// [`Quad`](oxixml_model::Quad)s.
///
/// It currently supports every [`RdfFormat`]:
/// * [JSON-LD](https://www.w3.org/TR/json-ld/) ([`RdfFormat::JsonLd`])
/// * [N3](https://w3c.github.io/N3/spec/) ([`RdfFormat::N3`])
/// * [N-Quads](https://www.w3.org/TR/n-quads/) ([`RdfFormat::NQuads`])
/// * [N-Triples](https://www.w3.org/TR/n-triples/) ([`RdfFormat::NTriples`])
/// * [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) ([`RdfFormat::RdfXml`])
/// * [TriG](https://www.w3.org/TR/trig/) ([`RdfFormat::TriG`])
/// * [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) ([`RdfFormat::TriX`])
/// * [Turtle](https://www.w3.org/TR/turtle/) ([`RdfFormat::Turtle`])
///
/// Formats limited to a single [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph)
/// (N-Triples, Turtle, RDF/XML) always parse into the [default graph](GraphName::DefaultGraph),
/// or into [`with_default_graph`](Self::with_default_graph)'s target when set — see
/// [`RdfFormat::supports_datasets`] for a per-format breakdown.
///
/// Useful options:
/// * [`with_base_iri`](Self::with_base_iri) resolves relative IRIs (ignored by N-Triples,
///   N-Quads, and TriX, none of which have a notion of a base IRI).
/// * [`with_default_graph`](Self::with_default_graph) replaces the default graph name in the
///   returned quads.
/// * [`without_named_graphs`](Self::without_named_graphs) rejects any named graph, restricting
///   the parser to a single graph.
/// * [`rename_blank_nodes`](Self::rename_blank_nodes) renames blank nodes to fresh,
///   collision-free ids — handy when merging several parses together.
/// * [`lenient`](Self::lenient) skips some validations for already-trusted input (ignored by
///   TriX, whose parser has no lenient mode).
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
#[derive(Clone)]
#[must_use]
pub struct RdfParser {
    inner: RdfParserKind,
    default_graph: GraphName,
    without_named_graphs: bool,
    rename_blank_nodes: bool,
}

#[derive(Clone)]
enum RdfParserKind {
    JsonLd(JsonLdParser, JsonLdProfileSet),
    N3(N3Parser),
    NQuads(NQuadsParser),
    NTriples(NTriplesParser),
    RdfXml(RdfXmlParser),
    TriG(TriGParser),
    TriX(TriXParser),
    Turtle(TurtleParser),
}

impl RdfParser {
    /// Builds a parser for the given format.
    #[inline]
    pub fn from_format(format: RdfFormat) -> Self {
        Self {
            inner: match format {
                RdfFormat::JsonLd { profile } => {
                    RdfParserKind::JsonLd(JsonLdParser::new().with_profile(profile), profile)
                }
                RdfFormat::N3 => RdfParserKind::N3(N3Parser::new()),
                RdfFormat::NQuads => RdfParserKind::NQuads(NQuadsParser::new()),
                RdfFormat::NTriples => RdfParserKind::NTriples(NTriplesParser::new()),
                RdfFormat::RdfXml => RdfParserKind::RdfXml(RdfXmlParser::new()),
                RdfFormat::TriG => RdfParserKind::TriG(TriGParser::new()),
                RdfFormat::TriX => RdfParserKind::TriX(TriXParser::new()),
                RdfFormat::Turtle => RdfParserKind::Turtle(TurtleParser::new()),
            },
            default_graph: GraphName::DefaultGraph,
            without_named_graphs: false,
            rename_blank_nodes: false,
        }
    }

    /// The format the parser uses.
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// assert_eq!(
    ///     RdfParser::from_format(RdfFormat::Turtle).format(),
    ///     RdfFormat::Turtle
    /// );
    /// ```
    #[must_use]
    pub fn format(&self) -> RdfFormat {
        match &self.inner {
            RdfParserKind::JsonLd(_, profile) => RdfFormat::JsonLd { profile: *profile },
            RdfParserKind::N3(_) => RdfFormat::N3,
            RdfParserKind::NQuads(_) => RdfFormat::NQuads,
            RdfParserKind::NTriples(_) => RdfFormat::NTriples,
            RdfParserKind::RdfXml(_) => RdfFormat::RdfXml,
            RdfParserKind::TriG(_) => RdfFormat::TriG,
            RdfParserKind::TriX(_) => RdfFormat::TriX,
            RdfParserKind::Turtle(_) => RdfFormat::Turtle,
        }
    }

    /// Provides an IRI that relative IRIs in the file resolve against.
    ///
    /// Silently ignored for N-Triples, N-Quads, and TriX: none of these formats has relative
    /// IRIs, so `oxrdfio` likewise leaves them untouched.
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// let file = "</s> </p> </o> .";
    ///
    /// let quads = RdfParser::from_format(RdfFormat::Turtle)
    ///     .with_base_iri("http://example.com")?
    ///     .for_reader(file.as_bytes())
    ///     .collect::<Result<Vec<_>, _>>()?;
    ///
    /// assert_eq!(quads.len(), 1);
    /// assert_eq!(quads[0].subject.to_string(), "<http://example.com/s>");
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`IriParseError`] if `base_iri` is not a valid absolute IRI.
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        let base_iri = base_iri.into();
        self.inner = match self.inner {
            RdfParserKind::JsonLd(p, f) => RdfParserKind::JsonLd(p.with_base_iri(base_iri)?, f),
            RdfParserKind::N3(p) => RdfParserKind::N3(p.with_base_iri(base_iri)?),
            RdfParserKind::NQuads(p) => RdfParserKind::NQuads(p),
            RdfParserKind::NTriples(p) => RdfParserKind::NTriples(p),
            RdfParserKind::RdfXml(p) => RdfParserKind::RdfXml(p.with_base_iri(base_iri)?),
            RdfParserKind::TriG(p) => RdfParserKind::TriG(p.with_base_iri(base_iri)?),
            RdfParserKind::TriX(p) => RdfParserKind::TriX(p),
            RdfParserKind::Turtle(p) => RdfParserKind::Turtle(p.with_base_iri(base_iri)?),
        };
        Ok(self)
    }

    /// Provides the graph name that should replace the default graph in the returned quads.
    ///
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// let file = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .";
    ///
    /// let quads = RdfParser::from_format(RdfFormat::Turtle)
    ///     .with_default_graph(NamedNode::new("http://example.com/g")?)
    ///     .for_reader(file.as_bytes())
    ///     .collect::<Result<Vec<_>, _>>()?;
    ///
    /// assert_eq!(quads.len(), 1);
    /// assert_eq!(quads[0].graph_name.to_string(), "<http://example.com/g>");
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_default_graph(mut self, default_graph: impl Into<GraphName>) -> Self {
        self.default_graph = default_graph.into();
        self
    }

    /// Makes the parser fail on any named graph, restricting it to a single
    /// [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph) rather than a full
    /// [RDF dataset](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-dataset).
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// let file = "<http://example.com/s> <http://example.com/p> <http://example.com/o> <http://example.com/g> .";
    ///
    /// let parser = RdfParser::from_format(RdfFormat::NQuads).without_named_graphs();
    /// assert!(parser.for_reader(file.as_bytes()).next().expect("one quad").is_err());
    /// ```
    #[inline]
    pub fn without_named_graphs(mut self) -> Self {
        self.without_named_graphs = true;
        self
    }

    /// Renames the blank node ids set in the serialization to fresh, collision-free ids.
    ///
    /// Avoids id conflicts when merging separately-parsed graphs together.
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// let file = "_:a <http://example.com/p> <http://example.com/o> .";
    ///
    /// let result1 = RdfParser::from_format(RdfFormat::NQuads)
    ///     .rename_blank_nodes()
    ///     .for_reader(file.as_bytes())
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// let result2 = RdfParser::from_format(RdfFormat::NQuads)
    ///     .rename_blank_nodes()
    ///     .for_reader(file.as_bytes())
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// assert_ne!(result1, result2);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn rename_blank_nodes(mut self) -> Self {
        self.rename_blank_nodes = true;
        self
    }

    /// Assumes the file is already valid, to parse faster.
    ///
    /// Some validations are skipped; a genuinely invalid file may then produce broken RDF instead
    /// of an error. Silently ignored for TriX, whose parser has no lenient mode.
    #[inline]
    pub fn lenient(mut self) -> Self {
        self.inner = match self.inner {
            RdfParserKind::JsonLd(p, f) => RdfParserKind::JsonLd(p.lenient(), f),
            RdfParserKind::N3(p) => RdfParserKind::N3(p.lenient()),
            RdfParserKind::NQuads(p) => RdfParserKind::NQuads(p.lenient()),
            RdfParserKind::NTriples(p) => RdfParserKind::NTriples(p.lenient()),
            RdfParserKind::RdfXml(p) => RdfParserKind::RdfXml(p.lenient()),
            RdfParserKind::TriG(p) => RdfParserKind::TriG(p.lenient()),
            RdfParserKind::TriX(p) => RdfParserKind::TriX(p),
            RdfParserKind::Turtle(p) => RdfParserKind::Turtle(p.lenient()),
        };
        self
    }

    fn mapper(&self) -> QuadMapper {
        QuadMapper {
            default_graph: self.default_graph.clone(),
            without_named_graphs: self.without_named_graphs,
            blank_node_map: self.rename_blank_nodes.then(HashMap::new),
        }
    }

    /// Parses from a [`Read`] implementation and returns an iterator of quads.
    ///
    /// Reads are buffered internally.
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
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderQuadParser<R> {
        let mapper = self.mapper();
        reader::build(self.inner, reader, mapper)
    }

    /// Parses from a [`AsyncRead`] implementation and returns an asynchronous quad parser.
    ///
    /// The asynchronous counterpart of [`for_reader`](Self::for_reader): it dispatches to the
    /// very same per-format parsers, awaiting each read instead of blocking on it.
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
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderQuadParser<R> {
        let mapper = self.mapper();
        tokio_async::build(self.inner, reader, mapper)
    }

    /// Parses from a byte slice and returns an iterator of quads.
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
    pub fn for_slice(self, slice: &(impl AsRef<[u8]> + ?Sized)) -> SliceQuadParser<'_> {
        let mapper = self.mapper();
        slice::build(self.inner, slice.as_ref(), mapper)
    }

    /// Splits `slice` into up to `target_parallelism` independently parseable chunks, one parser
    /// each, so a large document can be loaded on several threads.
    ///
    /// Only the line-based formats ([`RdfFormat::NTriples`] and [`RdfFormat::NQuads`]) can be cut
    /// up this way; every other format yields a single parser covering the whole slice, exactly
    /// like [`for_slice`](Self::for_slice). To pick `target_parallelism` dynamically, use
    /// [`std::thread::available_parallelism`].
    ///
    /// Note that each chunk gets its own blank-node scope, so
    /// [`rename_blank_nodes`](Self::rename_blank_nodes) renames independently per chunk.
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfParser};
    ///
    /// let file = "<http://example.com/s> <http://example.com/p> <http://example.com/o> .";
    ///
    /// let quads = RdfParser::from_format(RdfFormat::NTriples)
    ///     .split_slice_for_parallel_parsing(file, 4)
    ///     .into_iter()
    ///     .flatten()
    ///     .collect::<Result<Vec<_>, _>>()?;
    ///
    /// assert_eq!(quads.len(), 1);
    /// assert_eq!(quads[0].subject.to_string(), "<http://example.com/s>");
    /// # Result::<(), oxixml_io::RdfParseError>::Ok(())
    /// ```
    pub fn split_slice_for_parallel_parsing(
        self,
        slice: &(impl AsRef<[u8]> + ?Sized),
        target_parallelism: usize,
    ) -> Vec<SliceQuadParser<'_>> {
        let mapper = self.mapper();
        match self.inner {
            RdfParserKind::NTriples(p) => p
                .split_slice_for_parallel_parsing(slice, target_parallelism)
                .into_iter()
                .map(|p| slice::from_ntriples(p, mapper.clone()))
                .collect(),
            RdfParserKind::NQuads(p) => p
                .split_slice_for_parallel_parsing(slice, target_parallelism)
                .into_iter()
                .map(|p| slice::from_nquads(p, mapper.clone()))
                .collect(),
            inner => vec![slice::build(inner, slice.as_ref(), mapper)],
        }
    }

    /// Splits the file at `path` into up to `target_parallelism` independently parseable chunks,
    /// one parser each.
    ///
    /// The same restrictions as [`split_slice_for_parallel_parsing`](Self::split_slice_for_parallel_parsing)
    /// apply: only [`RdfFormat::NTriples`] and [`RdfFormat::NQuads`] are actually split.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`](std::io::Error) if the file cannot be opened or its chunk
    /// boundaries cannot be located.
    ///
    /// ```no_run
    /// use oxixml_io::{RdfFormat, RdfParser};
    /// # let path = std::env::temp_dir().join("oxixml-io-parallel-example.nt");
    /// # std::fs::write(&path, "<http://example.com/s> <http://example.com/p> <http://example.com/o> .")?;
    ///
    /// let quads = RdfParser::from_format(RdfFormat::NTriples)
    ///     .split_file_for_parallel_parsing(&path, 4)?
    ///     .into_iter()
    ///     .flatten()
    ///     .collect::<Result<Vec<_>, _>>()?;
    ///
    /// assert_eq!(quads.len(), 1);
    /// # Result::<(), Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn split_file_for_parallel_parsing(
        self,
        path: impl AsRef<Path>,
        target_parallelism: usize,
    ) -> std::io::Result<Vec<ReaderQuadParser<Take<File>>>> {
        let mapper = self.mapper();
        Ok(match self.inner {
            RdfParserKind::NTriples(p) => p
                .split_file_for_parallel_parsing(path, target_parallelism)?
                .into_iter()
                .map(|p| reader::from_ntriples(p, mapper.clone()))
                .collect(),
            RdfParserKind::NQuads(p) => p
                .split_file_for_parallel_parsing(path, target_parallelism)?
                .into_iter()
                .map(|p| reader::from_nquads(p, mapper.clone()))
                .collect(),
            inner => vec![reader::build(
                inner,
                File::open(path)?.take(u64::MAX),
                mapper,
            )],
        })
    }
}

impl From<RdfFormat> for RdfParser {
    #[inline]
    fn from(format: RdfFormat) -> Self {
        Self::from_format(format)
    }
}
