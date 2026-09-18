//! [`RdfSerializer`]: a format-agnostic builder that wraps every backend serializer behind one
//! API.

use crate::RdfFormat;
#[cfg(feature = "async-tokio")]
use oxixml_jsonld::TokioAsyncWriterJsonLdSerializer;
use oxixml_jsonld::{JsonLdSerializer, WriterJsonLdSerializer};
#[cfg(test)]
use oxixml_model::{GraphName, Quad};
use oxixml_model::{GraphNameRef, IriParseError, QuadRef, TripleRef};
#[cfg(feature = "async-tokio")]
use oxixml_rdfxml::TokioAsyncWriterRdfXmlSerializer;
use oxixml_rdfxml::{RdfXmlSerializer, WriterRdfXmlSerializer};
#[cfg(feature = "async-tokio")]
use oxixml_trix::TokioAsyncWriterTriXSerializer;
use oxixml_trix::{TriXSerializer, WriterTriXSerializer};
#[cfg(feature = "async-tokio")]
use oxixml_turtle::nquads::TokioAsyncWriterNQuadsSerializer;
use oxixml_turtle::nquads::{NQuadsSerializer, WriterNQuadsSerializer};
#[cfg(feature = "async-tokio")]
use oxixml_turtle::ntriples::TokioAsyncWriterNTriplesSerializer;
use oxixml_turtle::ntriples::{NTriplesSerializer, WriterNTriplesSerializer};
#[cfg(feature = "async-tokio")]
use oxixml_turtle::trig::TokioAsyncWriterTriGSerializer;
use oxixml_turtle::trig::{TriGSerializer, WriterTriGSerializer};
#[cfg(feature = "async-tokio")]
use oxixml_turtle::turtle::TokioAsyncWriterTurtleSerializer;
use oxixml_turtle::turtle::{TurtleSerializer, WriterTurtleSerializer};
use std::io::{self, Write};
#[cfg(feature = "async-tokio")]
use tokio::io::AsyncWrite;

/// A format-agnostic builder for serializing RDF [`Quad`](oxixml_model::Quad)s (or [`Triple`](oxixml_model::Triple)s).
///
/// It currently supports every [`RdfFormat`]:
/// * [JSON-LD](https://www.w3.org/TR/json-ld/) ([`RdfFormat::JsonLd`])
/// * [N-Quads](https://www.w3.org/TR/n-quads/) ([`RdfFormat::NQuads`])
/// * [N-Triples](https://www.w3.org/TR/n-triples/) ([`RdfFormat::NTriples`])
/// * [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) ([`RdfFormat::RdfXml`])
/// * [TriG](https://www.w3.org/TR/trig/) ([`RdfFormat::TriG`])
/// * [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) ([`RdfFormat::TriX`])
/// * [Turtle](https://www.w3.org/TR/turtle/) ([`RdfFormat::Turtle`])
///
/// [`RdfFormat::N3`] serializes as Turtle: N3 is a strict syntactic superset of Turtle for the
/// triples/quads this crate can express, and `oxixml-turtle` (like `oxttl`) does not provide a
/// distinct N3 writer.
///
/// A quad outside the default graph can only be serialized to a dataset format
/// ([`supports_datasets`](RdfFormat::supports_datasets)); [`serialize_quad`](WriterQuadSerializer::serialize_quad)
/// returns an [`io::ErrorKind::InvalidInput`] error otherwise.
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_io::{RdfFormat, RdfSerializer};
///
/// let mut serializer = RdfSerializer::from_format(RdfFormat::NQuads).for_writer(Vec::new());
/// serializer.serialize_quad(&Quad::new(
///     NamedNode::new("http://example.com/s")?,
///     NamedNode::new("http://example.com/p")?,
///     NamedNode::new("http://example.com/o")?,
///     NamedNode::new("http://example.com/g")?,
/// ))?;
/// assert_eq!(
///     serializer.finish()?,
///     b"<http://example.com/s> <http://example.com/p> <http://example.com/o> <http://example.com/g> .\n"
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct RdfSerializer {
    inner: RdfSerializerKind,
}

enum RdfSerializerKind {
    JsonLd(JsonLdSerializer),
    NQuads(NQuadsSerializer),
    NTriples(NTriplesSerializer),
    RdfXml(RdfXmlSerializer),
    TriG(TriGSerializer),
    TriX(TriXSerializer),
    Turtle(TurtleSerializer),
}

impl RdfSerializer {
    /// Builds a serializer for the given format.
    #[inline]
    pub fn from_format(format: RdfFormat) -> Self {
        Self {
            inner: match format {
                RdfFormat::JsonLd { .. } => RdfSerializerKind::JsonLd(JsonLdSerializer::new()),
                RdfFormat::NQuads => RdfSerializerKind::NQuads(NQuadsSerializer::new()),
                RdfFormat::NTriples => RdfSerializerKind::NTriples(NTriplesSerializer::new()),
                RdfFormat::RdfXml => RdfSerializerKind::RdfXml(RdfXmlSerializer::new()),
                RdfFormat::TriG => RdfSerializerKind::TriG(TriGSerializer::new()),
                RdfFormat::TriX => RdfSerializerKind::TriX(TriXSerializer::new()),
                RdfFormat::Turtle | RdfFormat::N3 => {
                    RdfSerializerKind::Turtle(TurtleSerializer::new())
                }
            },
        }
    }

    /// The format the serializer writes.
    ///
    /// Returns [`RdfFormat::Turtle`] for a serializer built from [`RdfFormat::N3`]: see the
    /// [type-level documentation](Self) for why N3 and Turtle share a writer. Likewise, always
    /// returns an empty [`JsonLdProfileSet`](oxixml_jsonld::JsonLdProfileSet) for JSON-LD: the
    /// underlying `oxixml-jsonld` writer does not track the profile it was built with.
    ///
    /// ```
    /// use oxixml_io::{RdfFormat, RdfSerializer};
    ///
    /// assert_eq!(
    ///     RdfSerializer::from_format(RdfFormat::Turtle).format(),
    ///     RdfFormat::Turtle
    /// );
    /// ```
    #[must_use]
    pub fn format(&self) -> RdfFormat {
        match &self.inner {
            RdfSerializerKind::JsonLd(_) => RdfFormat::JsonLd {
                profile: oxixml_jsonld::JsonLdProfileSet::empty(),
            },
            RdfSerializerKind::NQuads(_) => RdfFormat::NQuads,
            RdfSerializerKind::NTriples(_) => RdfFormat::NTriples,
            RdfSerializerKind::RdfXml(_) => RdfFormat::RdfXml,
            RdfSerializerKind::TriG(_) => RdfFormat::TriG,
            RdfSerializerKind::TriX(_) => RdfFormat::TriX,
            RdfSerializerKind::Turtle(_) => RdfFormat::Turtle,
        }
    }

    /// If the format supports it, registers a namespace prefix.
    ///
    /// A no-op for N-Quads, N-Triples, and TriX, none of which has a notion of a namespace
    /// prefix. JSON-LD declares the prefix in the document's `@context`; its property keys stay
    /// expanded IRIs, as they do in `oxjsonld`.
    ///
    /// This is a deliberate deviation from `oxrdfio`, which drops the prefix for JSON-LD rather
    /// than forwarding it, so a JSON-LD document written here carries a `@context` and a `@graph`
    /// where `oxrdfio` writes a bare array. The same applies to
    /// [`with_base_iri`](Self::with_base_iri). Every other format is byte-identical.
    ///
    /// ```
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::{NamedNode, Triple};
    /// use oxixml_io::{RdfFormat, RdfSerializer};
    ///
    /// let mut serializer = RdfSerializer::from_format(RdfFormat::Turtle)
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_triple(&Triple::new(
    ///     NamedNode::new("http://example.com/s")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://schema.org/Person")?,
    /// ))?;
    /// assert_eq!(
    ///     serializer.finish()?,
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com/s> a schema:Person .\n"
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`IriParseError`] if `prefix_iri` is not a valid IRI.
    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: impl Into<String>,
        prefix_iri: impl Into<String>,
    ) -> Result<Self, IriParseError> {
        self.inner = match self.inner {
            RdfSerializerKind::JsonLd(s) => {
                RdfSerializerKind::JsonLd(s.with_prefix(prefix_name, prefix_iri)?)
            }
            RdfSerializerKind::NQuads(s) => RdfSerializerKind::NQuads(s),
            RdfSerializerKind::NTriples(s) => RdfSerializerKind::NTriples(s),
            RdfSerializerKind::RdfXml(s) => {
                RdfSerializerKind::RdfXml(s.with_prefix(prefix_name, prefix_iri)?)
            }
            RdfSerializerKind::TriG(s) => {
                RdfSerializerKind::TriG(s.with_prefix(prefix_name, prefix_iri)?)
            }
            RdfSerializerKind::TriX(s) => RdfSerializerKind::TriX(s),
            RdfSerializerKind::Turtle(s) => {
                RdfSerializerKind::Turtle(s.with_prefix(prefix_name, prefix_iri)?)
            }
        };
        Ok(self)
    }

    /// If the format supports it, sets a base IRI that written IRIs are made relative to.
    ///
    /// A no-op for N-Quads, N-Triples, and TriX. For JSON-LD the base is written as `@base` in
    /// the document's `@context` and node identifiers are made relative to it — a deliberate
    /// deviation from `oxrdfio`, which drops the base for JSON-LD rather than forwarding it.
    ///
    /// # Errors
    ///
    /// Returns an [`IriParseError`] if `base_iri` is not a valid IRI.
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        let base_iri = base_iri.into();
        self.inner = match self.inner {
            RdfSerializerKind::JsonLd(s) => RdfSerializerKind::JsonLd(s.with_base_iri(base_iri)?),
            RdfSerializerKind::NQuads(s) => RdfSerializerKind::NQuads(s),
            RdfSerializerKind::NTriples(s) => RdfSerializerKind::NTriples(s),
            RdfSerializerKind::RdfXml(s) => RdfSerializerKind::RdfXml(s.with_base_iri(base_iri)?),
            RdfSerializerKind::TriG(s) => RdfSerializerKind::TriG(s.with_base_iri(base_iri)?),
            RdfSerializerKind::TriX(s) => RdfSerializerKind::TriX(s),
            RdfSerializerKind::Turtle(s) => RdfSerializerKind::Turtle(s.with_base_iri(base_iri)?),
        };
        Ok(self)
    }

    /// Serializes to a [`Write`] implementation.
    ///
    /// Do not forget to call [`finish`](WriterQuadSerializer::finish) to write the last bytes of
    /// the file. Writes are unbuffered; consider wrapping `writer` in a [`BufWriter`](io::BufWriter).
    ///
    /// ```
    /// use oxixml_model::{NamedNode, Quad};
    /// use oxixml_io::{RdfFormat, RdfSerializer};
    ///
    /// let mut serializer = RdfSerializer::from_format(RdfFormat::NQuads).for_writer(Vec::new());
    /// serializer.serialize_quad(&Quad::new(
    ///     NamedNode::new("http://example.com/s")?,
    ///     NamedNode::new("http://example.com/p")?,
    ///     NamedNode::new("http://example.com/o")?,
    ///     NamedNode::new("http://example.com/g")?,
    /// ))?;
    /// assert_eq!(
    ///     serializer.finish()?,
    ///     b"<http://example.com/s> <http://example.com/p> <http://example.com/o> <http://example.com/g> .\n"
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_writer<W: Write>(self, writer: W) -> WriterQuadSerializer<W> {
        WriterQuadSerializer {
            inner: match self.inner {
                RdfSerializerKind::JsonLd(s) => {
                    WriterQuadSerializerKind::JsonLd(s.for_writer(writer))
                }
                RdfSerializerKind::NQuads(s) => {
                    WriterQuadSerializerKind::NQuads(s.for_writer(writer))
                }
                RdfSerializerKind::NTriples(s) => {
                    WriterQuadSerializerKind::NTriples(s.for_writer(writer))
                }
                RdfSerializerKind::RdfXml(s) => {
                    WriterQuadSerializerKind::RdfXml(s.for_writer(writer))
                }
                RdfSerializerKind::TriG(s) => WriterQuadSerializerKind::TriG(s.for_writer(writer)),
                RdfSerializerKind::TriX(s) => WriterQuadSerializerKind::TriX(s.for_writer(writer)),
                RdfSerializerKind::Turtle(s) => {
                    WriterQuadSerializerKind::Turtle(s.for_writer(writer))
                }
            },
        }
    }

    /// Serializes to a [`AsyncWrite`] implementation.
    ///
    /// The asynchronous counterpart of [`for_writer`](Self::for_writer): it dispatches to the
    /// very same per-format serializers, awaiting each write instead of blocking on it, and so
    /// produces byte-identical output.
    ///
    /// Do not forget to call [`finish`](TokioAsyncWriterQuadSerializer::finish) to write the
    /// last bytes of the file.
    ///
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> std::io::Result<()> {
    /// use oxixml_io::{RdfFormat, RdfSerializer};
    /// use oxixml_model::{NamedNode, Quad};
    ///
    /// let mut serializer =
    ///     RdfSerializer::from_format(RdfFormat::NQuads).for_tokio_async_writer(Vec::new());
    /// serializer
    ///     .serialize_quad(&Quad::new(
    ///         NamedNode::new_unchecked("http://example.com/s"),
    ///         NamedNode::new_unchecked("http://example.com/p"),
    ///         NamedNode::new_unchecked("http://example.com/o"),
    ///         NamedNode::new_unchecked("http://example.com/g"),
    ///     ))
    ///     .await?;
    /// assert_eq!(
    ///     serializer.finish().await?,
    ///     b"<http://example.com/s> <http://example.com/p> <http://example.com/o> <http://example.com/g> .\n"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterQuadSerializer<W> {
        TokioAsyncWriterQuadSerializer {
            inner: match self.inner {
                RdfSerializerKind::JsonLd(s) => {
                    TokioAsyncWriterQuadSerializerKind::JsonLd(s.for_tokio_async_writer(writer))
                }
                RdfSerializerKind::NQuads(s) => {
                    TokioAsyncWriterQuadSerializerKind::NQuads(s.for_tokio_async_writer(writer))
                }
                RdfSerializerKind::NTriples(s) => {
                    TokioAsyncWriterQuadSerializerKind::NTriples(s.for_tokio_async_writer(writer))
                }
                RdfSerializerKind::RdfXml(s) => {
                    TokioAsyncWriterQuadSerializerKind::RdfXml(s.for_tokio_async_writer(writer))
                }
                RdfSerializerKind::TriG(s) => {
                    TokioAsyncWriterQuadSerializerKind::TriG(s.for_tokio_async_writer(writer))
                }
                RdfSerializerKind::TriX(s) => {
                    TokioAsyncWriterQuadSerializerKind::TriX(s.for_tokio_async_writer(writer))
                }
                RdfSerializerKind::Turtle(s) => {
                    TokioAsyncWriterQuadSerializerKind::Turtle(s.for_tokio_async_writer(writer))
                }
            },
        }
    }
}

/// Serializes [`Quad`](oxixml_model::Quad)s (or [`Triple`](oxixml_model::Triple)s) to a [`AsyncWrite`] implementation.
///
/// Can be built using [`RdfSerializer::for_tokio_async_writer`]. Do not forget to call
/// [`finish`](Self::finish) to write the last bytes of the file.
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterQuadSerializer<W: AsyncWrite + Unpin> {
    inner: TokioAsyncWriterQuadSerializerKind<W>,
}

#[cfg(feature = "async-tokio")]
enum TokioAsyncWriterQuadSerializerKind<W: AsyncWrite + Unpin> {
    JsonLd(TokioAsyncWriterJsonLdSerializer<W>),
    NQuads(TokioAsyncWriterNQuadsSerializer<W>),
    NTriples(TokioAsyncWriterNTriplesSerializer<W>),
    RdfXml(TokioAsyncWriterRdfXmlSerializer<W>),
    TriG(TokioAsyncWriterTriGSerializer<W>),
    TriX(TokioAsyncWriterTriXSerializer<W>),
    Turtle(TokioAsyncWriterTurtleSerializer<W>),
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterQuadSerializer<W> {
    /// Serializes a [`Quad`](oxixml_model::Quad).
    ///
    /// # Errors
    ///
    /// Returns an [`io::ErrorKind::InvalidInput`] error if `quad` is outside the default graph
    /// and the format does not [support datasets](RdfFormat::supports_datasets); otherwise, any
    /// I/O error raised by the underlying writer.
    pub async fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        let quad = quad.into();
        match &mut self.inner {
            TokioAsyncWriterQuadSerializerKind::JsonLd(serializer) => {
                serializer.serialize_quad(quad).await
            }
            TokioAsyncWriterQuadSerializerKind::NQuads(serializer) => {
                serializer.serialize_quad(quad).await
            }
            TokioAsyncWriterQuadSerializerKind::NTriples(serializer) => {
                serializer.serialize_triple(to_triple(quad)?).await
            }
            TokioAsyncWriterQuadSerializerKind::RdfXml(serializer) => {
                serializer.serialize_triple(to_triple(quad)?).await
            }
            TokioAsyncWriterQuadSerializerKind::TriG(serializer) => {
                serializer.serialize_quad(quad).await
            }
            TokioAsyncWriterQuadSerializerKind::TriX(serializer) => {
                serializer.serialize_quad(quad).await
            }
            TokioAsyncWriterQuadSerializerKind::Turtle(serializer) => {
                serializer.serialize_triple(to_triple(quad)?).await
            }
        }
    }

    /// Serializes a [`Triple`](oxixml_model::Triple) into the default graph.
    ///
    /// # Errors
    ///
    /// Returns any I/O error raised by the underlying writer.
    pub async fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
    ) -> io::Result<()> {
        let triple = triple.into();
        match &mut self.inner {
            TokioAsyncWriterQuadSerializerKind::JsonLd(serializer) => {
                serializer.serialize_triple(triple).await
            }
            TokioAsyncWriterQuadSerializerKind::NQuads(serializer) => {
                serializer.serialize_triple(triple).await
            }
            TokioAsyncWriterQuadSerializerKind::NTriples(serializer) => {
                serializer.serialize_triple(triple).await
            }
            TokioAsyncWriterQuadSerializerKind::RdfXml(serializer) => {
                serializer.serialize_triple(triple).await
            }
            TokioAsyncWriterQuadSerializerKind::TriG(serializer) => {
                serializer.serialize_triple(triple).await
            }
            TokioAsyncWriterQuadSerializerKind::TriX(serializer) => {
                serializer
                    .serialize_quad(triple.in_graph(GraphNameRef::DefaultGraph))
                    .await
            }
            TokioAsyncWriterQuadSerializerKind::Turtle(serializer) => {
                serializer.serialize_triple(triple).await
            }
        }
    }

    /// Writes the last bytes of the file.
    ///
    /// This does not flush `writer`.
    ///
    /// # Errors
    ///
    /// Returns any I/O error raised by the underlying writer.
    pub async fn finish(self) -> io::Result<W> {
        Ok(match self.inner {
            TokioAsyncWriterQuadSerializerKind::JsonLd(serializer) => serializer.finish().await?,
            TokioAsyncWriterQuadSerializerKind::NQuads(serializer) => serializer.finish(),
            TokioAsyncWriterQuadSerializerKind::NTriples(serializer) => serializer.finish(),
            TokioAsyncWriterQuadSerializerKind::RdfXml(serializer) => serializer.finish().await?,
            TokioAsyncWriterQuadSerializerKind::TriG(serializer) => serializer.finish().await?,
            TokioAsyncWriterQuadSerializerKind::TriX(serializer) => serializer.finish().await?,
            TokioAsyncWriterQuadSerializerKind::Turtle(serializer) => serializer.finish().await?,
        })
    }
}

impl From<RdfFormat> for RdfSerializer {
    #[inline]
    fn from(format: RdfFormat) -> Self {
        Self::from_format(format)
    }
}

/// Serializes [`Quad`](oxixml_model::Quad)s (or [`Triple`](oxixml_model::Triple)s) to a [`Write`] implementation.
///
/// Can be built using [`RdfSerializer::for_writer`]. Do not forget to call
/// [`finish`](Self::finish) to write the last bytes of the file.
#[must_use]
pub struct WriterQuadSerializer<W: Write> {
    inner: WriterQuadSerializerKind<W>,
}

enum WriterQuadSerializerKind<W: Write> {
    JsonLd(WriterJsonLdSerializer<W>),
    NQuads(WriterNQuadsSerializer<W>),
    NTriples(WriterNTriplesSerializer<W>),
    RdfXml(WriterRdfXmlSerializer<W>),
    TriG(WriterTriGSerializer<W>),
    TriX(WriterTriXSerializer<W>),
    Turtle(WriterTurtleSerializer<W>),
}

impl<W: Write> WriterQuadSerializer<W> {
    /// Serializes a [`Quad`](oxixml_model::Quad).
    ///
    /// # Errors
    ///
    /// Returns an [`io::ErrorKind::InvalidInput`] error if `quad` is outside the default graph
    /// and the format does not [support datasets](RdfFormat::supports_datasets); otherwise, any
    /// I/O error raised by the underlying writer.
    pub fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        let quad = quad.into();
        match &mut self.inner {
            WriterQuadSerializerKind::JsonLd(serializer) => serializer.serialize_quad(quad),
            WriterQuadSerializerKind::NQuads(serializer) => serializer.serialize_quad(quad),
            WriterQuadSerializerKind::NTriples(serializer) => {
                serializer.serialize_triple(to_triple(quad)?)
            }
            WriterQuadSerializerKind::RdfXml(serializer) => {
                serializer.serialize_triple(to_triple(quad)?)
            }
            WriterQuadSerializerKind::TriG(serializer) => serializer.serialize_quad(quad),
            WriterQuadSerializerKind::TriX(serializer) => serializer.serialize_quad(quad),
            WriterQuadSerializerKind::Turtle(serializer) => {
                serializer.serialize_triple(to_triple(quad)?)
            }
        }
    }

    /// Serializes a [`Triple`](oxixml_model::Triple) into the default graph.
    ///
    /// # Errors
    ///
    /// Returns any I/O error raised by the underlying writer.
    pub fn serialize_triple<'a>(&mut self, triple: impl Into<TripleRef<'a>>) -> io::Result<()> {
        let triple = triple.into();
        match &mut self.inner {
            WriterQuadSerializerKind::JsonLd(serializer) => serializer.serialize_triple(triple),
            WriterQuadSerializerKind::NQuads(serializer) => serializer.serialize_triple(triple),
            WriterQuadSerializerKind::NTriples(serializer) => serializer.serialize_triple(triple),
            WriterQuadSerializerKind::RdfXml(serializer) => serializer.serialize_triple(triple),
            WriterQuadSerializerKind::TriG(serializer) => serializer.serialize_triple(triple),
            WriterQuadSerializerKind::TriX(serializer) => {
                serializer.serialize_quad(triple.in_graph(GraphNameRef::DefaultGraph))
            }
            WriterQuadSerializerKind::Turtle(serializer) => serializer.serialize_triple(triple),
        }
    }

    /// Writes the last bytes of the file.
    ///
    /// This does not flush `writer`; wrap it in a [`BufWriter`](io::BufWriter) if buffering is
    /// wanted, and flush that separately.
    ///
    /// # Errors
    ///
    /// Returns any I/O error raised by the underlying writer.
    pub fn finish(self) -> io::Result<W> {
        Ok(match self.inner {
            WriterQuadSerializerKind::JsonLd(serializer) => serializer.finish()?,
            WriterQuadSerializerKind::NQuads(serializer) => serializer.finish(),
            WriterQuadSerializerKind::NTriples(serializer) => serializer.finish(),
            WriterQuadSerializerKind::RdfXml(serializer) => serializer.finish()?,
            WriterQuadSerializerKind::TriG(serializer) => serializer.finish()?,
            WriterQuadSerializerKind::TriX(serializer) => serializer.finish()?,
            WriterQuadSerializerKind::Turtle(serializer) => serializer.finish()?,
        })
    }
}

/// Extracts `quad`'s underlying [`TripleRef`], failing if it is not in the default graph.
///
/// The formats that need this (N-Triples, RDF/XML, Turtle) cannot represent a graph name at all.
fn to_triple<'a>(quad: QuadRef<'a>) -> io::Result<TripleRef<'a>> {
    if quad.graph_name.is_default_graph() {
        Ok(TripleRef {
            subject: quad.subject,
            predicate: quad.predicate,
            object: quad.object,
        })
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "only quads in the default graph can be serialized to a RDF graph format",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_model::NamedNode;

    #[test]
    fn to_triple_rejects_named_graph() {
        let quad = Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/o"),
            NamedNode::new_unchecked("http://example.com/g"),
        );
        let error = to_triple(quad.as_ref()).expect_err("named graph must be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn to_triple_accepts_default_graph() -> io::Result<()> {
        let quad = Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/o"),
            GraphName::DefaultGraph,
        );
        let triple = to_triple(quad.as_ref())?;
        assert_eq!(triple.subject, quad.subject.as_ref());
        Ok(())
    }
}
