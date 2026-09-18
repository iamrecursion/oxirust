//! The [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) streaming parser.

mod internal;
mod state;

use crate::error::{RdfXmlParseError, RdfXmlSyntaxError};
use crate::utils::is_nc_name;
use internal::InternalRdfXmlParser;
use oxixml_iri::{Iri, IriParseError};
use oxixml_model::{OxString, Triple};
use oxixml_xml::escape::unescape_with;
use oxixml_xml::name::{NamespaceBindingsIter, PrefixDeclaration};
use oxixml_xml::{Decoder, NsReader, XmlVersion};
use state::{EntityRegistry, RdfXmlState};
use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{BufReader, Read};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, BufReader as AsyncBufReader};

/// A [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) streaming parser.
///
/// It reads the file in streaming.
/// It does not keep data in memory except a stack for handling nested XML tags, and a set of all
/// seen `rdf:ID`s to detect duplicate ids and fail according to the specification.
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_rdfxml::RdfXmlParser;
///
/// let file = r#"<?xml version="1.0"?>
/// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
///  <rdf:Description rdf:about="http://example.com/foo">
///    <rdf:type rdf:resource="http://schema.org/Person" />
///    <schema:name>Foo</schema:name>
///  </rdf:Description>
///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
/// </rdf:RDF>"#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for triple in RdfXmlParser::new().for_reader(file.as_bytes()) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone)]
#[must_use]
pub struct RdfXmlParser {
    lenient: bool,
    base: Option<Iri<OxString>>,
}

impl RdfXmlParser {
    /// Builds a new [`RdfXmlParser`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assumes the file is valid to make parsing faster.
    ///
    /// It will skip some validations.
    ///
    /// Note that if the file is actually not valid, the parser might emit broken RDF.
    #[inline]
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Sets the base IRI relative IRI references are resolved against.
    ///
    /// It is overridden by any `xml:base` attribute found in the document.
    ///
    /// ```
    /// use oxixml_rdfxml::RdfXmlParser;
    ///
    /// let file = r#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    ///  <rdf:Description rdf:about="foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///  </rdf:Description>
    /// </rdf:RDF>"#;
    ///
    /// let triple = RdfXmlParser::new()
    ///     .with_base_iri("http://example.com/")?
    ///     .for_slice(file)
    ///     .next()
    ///     .expect("one triple")?;
    /// assert_eq!(triple.subject.to_string(), "<http://example.com/foo>");
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an [`IriParseError`] if `base_iri` is not a valid absolute IRI.
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base = Some(Iri::parse(OxString::from(base_iri.into()))?);
        Ok(self)
    }

    /// Parses an RDF/XML file from a [`Read`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_rdfxml::RdfXmlParser;
    ///
    /// let file = r#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for triple in RdfXmlParser::new().for_reader(file.as_bytes()) {
    ///     let triple = triple?;
    ///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderRdfXmlParser<R> {
        ReaderRdfXmlParser {
            results: Vec::new(),
            parser: self.into_internal(BufReader::new(reader)),
            reader_buffer: Vec::new(),
        }
    }

    /// Parses an RDF/XML file from a [`AsyncRead`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_reader`](Self::for_reader): it drives the very
    /// same event-at-a-time state machine, awaiting each XML event instead of blocking on it.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderRdfXmlParser<R> {
        TokioAsyncReaderRdfXmlParser {
            results: Vec::new(),
            parser: self.into_internal(AsyncBufReader::new(reader)),
            reader_buffer: Vec::new(),
        }
    }

    /// Parses an RDF/XML file from a byte slice.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_rdfxml::RdfXmlParser;
    ///
    /// let file = r#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for triple in RdfXmlParser::new().for_slice(file) {
    ///     let triple = triple?;
    ///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_slice(self, slice: &(impl AsRef<[u8]> + ?Sized)) -> SliceRdfXmlParser<'_> {
        SliceRdfXmlParser {
            results: Vec::new(),
            parser: self.into_internal(slice.as_ref()),
        }
    }

    /// Builds the reader-agnostic state machine shared by every entry point.
    fn into_internal<T>(self, reader: T) -> InternalRdfXmlParser<T> {
        let mut reader = NsReader::from_reader(reader);
        reader.config_mut().expand_empty_elements = true;
        InternalRdfXmlParser {
            reader,
            state: vec![RdfXmlState::Doc {
                base_iri: self.base.clone(),
            }],
            custom_entities: EntityRegistry::default(),
            in_literal_depth: 0,
            known_rdf_id: HashSet::new(),
            is_end: false,
            lenient: self.lenient,
            xml_version: XmlVersion::Implicit1_0,
            text_buffer: String::new(),
        }
    }
}

/// Parses an RDF/XML file from a [`Read`] implementation.
///
/// Can be built using [`RdfXmlParser::for_reader`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_rdfxml::RdfXmlParser;
///
/// let file = r#"<?xml version="1.0"?>
/// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
///  <rdf:Description rdf:about="http://example.com/foo">
///    <rdf:type rdf:resource="http://schema.org/Person" />
///    <schema:name>Foo</schema:name>
///  </rdf:Description>
///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
/// </rdf:RDF>"#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for triple in RdfXmlParser::new().for_reader(file.as_bytes()) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct ReaderRdfXmlParser<R: Read> {
    results: Vec<Triple>,
    parser: InternalRdfXmlParser<BufReader<R>>,
    reader_buffer: Vec<u8>,
}

impl<R: Read> Iterator for ReaderRdfXmlParser<R> {
    type Item = Result<Triple, RdfXmlParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(triple) = self.results.pop() {
                return Some(Ok(triple));
            } else if self.parser.is_end {
                return None;
            }
            if let Err(e) = self.parse_step() {
                return Some(Err(e));
            }
        }
    }
}

impl<R: Read> ReaderRdfXmlParser<R> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest
    /// version will be returned).
    ///
    /// ```
    /// use oxixml_rdfxml::RdfXmlParser;
    ///
    /// let file = r#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_reader(file.as_bytes());
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("a triple")?; // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [
    ///         ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ///         ("schema", "http://schema.org/")
    ///     ]
    /// ); // There are now prefixes
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> RdfXmlPrefixesIter<'_> {
        self.parser.prefixes()
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_rdfxml::RdfXmlParser;
    ///
    /// let file = r#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xml:base="http://example.com/">
    ///  <rdf:Description rdf:about="foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///  </rdf:Description>
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_reader(file.as_bytes());
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("a triple")?; // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        Some(self.parser.current_base_iri()?.as_str())
    }

    /// The current byte position in the input data.
    pub fn buffer_position(&self) -> u64 {
        self.parser.reader.buffer_position()
    }

    /// Reads one XML event and feeds it to the state machine.
    fn parse_step(&mut self) -> Result<(), RdfXmlParseError> {
        self.reader_buffer.clear();
        let event = self
            .parser
            .reader
            .read_event_into(&mut self.reader_buffer)?;
        self.parser.parse_event(event, &mut self.results)
    }
}

/// Parses an RDF/XML file from a [`AsyncRead`] implementation.
///
/// Can be built using [`RdfXmlParser::for_tokio_async_reader`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncReaderRdfXmlParser<R: AsyncRead + Unpin> {
    results: Vec<Triple>,
    parser: InternalRdfXmlParser<AsyncBufReader<R>>,
    reader_buffer: Vec<u8>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderRdfXmlParser<R> {
    /// Reads the next triple or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<Triple, RdfXmlParseError>> {
        loop {
            if let Some(triple) = self.results.pop() {
                return Some(Ok(triple));
            } else if self.parser.is_end {
                return None;
            }
            if let Err(e) = self.parse_step().await {
                return Some(Err(e));
            }
        }
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest
    /// version will be returned).
    pub fn prefixes(&self) -> RdfXmlPrefixesIter<'_> {
        self.parser.prefixes()
    }

    /// The base IRI considered at the current step of the parsing.
    pub fn base_iri(&self) -> Option<&str> {
        Some(self.parser.current_base_iri()?.as_str())
    }

    /// The current byte position in the input data.
    pub fn buffer_position(&self) -> u64 {
        self.parser.reader.buffer_position()
    }

    /// Reads one XML event asynchronously and feeds it to the state machine.
    async fn parse_step(&mut self) -> Result<(), RdfXmlParseError> {
        self.reader_buffer.clear();
        let event = self
            .parser
            .reader
            .read_event_into_async(&mut self.reader_buffer)
            .await?;
        self.parser.parse_event(event, &mut self.results)
    }
}

/// Parses an RDF/XML file from a byte slice.
///
/// Can be built using [`RdfXmlParser::for_slice`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_rdfxml::RdfXmlParser;
///
/// let file = r#"<?xml version="1.0"?>
/// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
///  <rdf:Description rdf:about="http://example.com/foo">
///    <rdf:type rdf:resource="http://schema.org/Person" />
///    <schema:name>Foo</schema:name>
///  </rdf:Description>
///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
/// </rdf:RDF>"#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for triple in RdfXmlParser::new().for_slice(file) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct SliceRdfXmlParser<'a> {
    results: Vec<Triple>,
    parser: InternalRdfXmlParser<&'a [u8]>,
}

impl Iterator for SliceRdfXmlParser<'_> {
    type Item = Result<Triple, RdfXmlSyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(triple) = self.results.pop() {
                return Some(Ok(triple));
            } else if self.parser.is_end {
                return None;
            }
            if let Err(RdfXmlParseError::Syntax(e)) = self.parse_step() {
                // I/O errors can't happen
                return Some(Err(e));
            }
        }
    }
}

impl SliceRdfXmlParser<'_> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest
    /// version will be returned).
    ///
    /// ```
    /// use oxixml_rdfxml::RdfXmlParser;
    ///
    /// let file = r#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_slice(file);
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("a triple")?; // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [
    ///         ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ///         ("schema", "http://schema.org/")
    ///     ]
    /// ); // There are now prefixes
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> RdfXmlPrefixesIter<'_> {
        self.parser.prefixes()
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_rdfxml::RdfXmlParser;
    ///
    /// let file = r#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xml:base="http://example.com/">
    ///  <rdf:Description rdf:about="foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///  </rdf:Description>
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_slice(file);
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("a triple")?; // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        Some(self.parser.current_base_iri()?.as_str())
    }

    /// The current byte position in the input data.
    pub fn buffer_position(&self) -> u64 {
        self.parser.reader.buffer_position()
    }

    /// Reads one XML event and feeds it to the state machine.
    fn parse_step(&mut self) -> Result<(), RdfXmlParseError> {
        let event = self.parser.reader.read_event()?;
        self.parser.parse_event(event, &mut self.results)
    }
}

impl<R> InternalRdfXmlParser<R> {
    /// Builds the prefix iterator shared by both public parsers.
    fn prefixes(&self) -> RdfXmlPrefixesIter<'_> {
        RdfXmlPrefixesIter {
            inner: self.reader.resolver().bindings(),
            custom_entities: &self.custom_entities,
            decoder: self.reader.decoder(),
            lenient: self.lenient,
        }
    }
}

/// Iterator on the file prefixes.
///
/// See [`ReaderRdfXmlParser::prefixes`].
pub struct RdfXmlPrefixesIter<'a> {
    inner: NamespaceBindingsIter<'a>,
    custom_entities: &'a EntityRegistry,
    decoder: Decoder,
    lenient: bool,
}

impl<'a> Iterator for RdfXmlPrefixesIter<'a> {
    type Item = (&'a str, &'a str);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (key, value) = self.inner.next()?;
            return Some((
                match key {
                    PrefixDeclaration::Default => "",
                    PrefixDeclaration::Named(name) => {
                        let Ok(Cow::Borrowed(name)) = self.decoder.decode(name) else {
                            continue;
                        };
                        let Ok(Cow::Borrowed(name)) =
                            unescape_with(name, |e| self.custom_entities.resolve(e))
                        else {
                            continue;
                        };
                        if !self.lenient && !is_nc_name(name) {
                            continue; // We don't return invalid prefixes
                        }
                        name
                    }
                },
                {
                    let Ok(Cow::Borrowed(value)) = self.decoder.decode(value.0) else {
                        continue;
                    };
                    let Ok(Cow::Borrowed(value)) =
                        unescape_with(value, |e| self.custom_entities.resolve(e))
                    else {
                        continue;
                    };
                    if !self.lenient && Iri::parse(value).is_err() {
                        continue; // We don't return invalid prefixes
                    }
                    value
                },
            ));
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
