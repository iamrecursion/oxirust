//! A [TriG](https://www.w3.org/TR/trig/) streaming parser implemented by [`TriGParser`]
//! and a serializer implemented by [`TriGSerializer`].

use crate::DEFAULT_MAX_BUFFER_SIZE;
use crate::lexer::N3Lexer;
use crate::terse::TriGRecognizer;
#[cfg(feature = "async-tokio")]
use crate::toolkit::TokioAsyncReaderIterator;
use crate::toolkit::{Parser, ReaderIterator, SliceIterator, TurtleParseError, TurtleSyntaxError};
use oxixml_iri::{Iri, IriParseError};
use oxixml_model::vocab::{rdf, xsd};
use oxixml_model::{
    GraphName, GraphNameRef, LiteralRef, NamedNode, NamedNodeRef, NamedOrBlankNode,
    NamedOrBlankNodeRef, OxString, Quad, QuadRef, TermRef, TripleRef,
};
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::hash_map::Iter;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::{self, Read, Write};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

/// A [TriG](https://www.w3.org/TR/trig/) streaming parser.
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TriGParser;
///
/// let file = r#"@base <http://example.com/> .
/// @prefix schema: <http://schema.org/> .
/// <foo> a schema:Person ;
///     schema:name "Foo" .
/// <bar> a schema:Person ;
///     schema:name "Bar" ."#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in TriGParser::new().for_reader(file.as_bytes()) {
///     let quad = quad?;
///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Clone)]
#[must_use]
pub struct TriGParser {
    lenient: bool,
    base: Option<Iri<OxString>>,
    prefixes: HashMap<OxString, Iri<OxString>>,
    max_buffer_size: usize,
}

impl Default for TriGParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TriGParser {
    /// Builds a new [`TriGParser`].
    #[inline]
    pub fn new() -> Self {
        Self {
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            lenient: false,
            base: None,
            prefixes: HashMap::new(),
        }
    }

    /// Define an upper bound for the internal buffer of the parser in bytes
    ///
    /// This limits the memory consumption of the parser and the maximum size of parsed IRIs and literals.
    ///
    /// The default is set conservatively, use this function to change it (e.g. to [`usize::MAX`] to not set an upper bound).
    #[inline]
    pub fn with_max_buffer_size(mut self, max_buffer_size: usize) -> Self {
        self.max_buffer_size = max_buffer_size;
        self
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
    /// Sets the base IRI used to resolve relative IRIs during parsing.
    ///
    /// # Errors
    ///
    /// Returns an error if the given base IRI is not a valid absolute IRI.
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base = Some(Iri::parse(OxString::from(base_iri.into()))?);
        Ok(self)
    }
    /// Registers a namespace prefix so that later prefixed names resolve against it.
    ///
    /// # Errors
    ///
    /// Returns an error if the given prefix IRI is not a valid IRI.
    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: impl Into<String>,
        prefix_iri: impl Into<String>,
    ) -> Result<Self, IriParseError> {
        self.prefixes.insert(
            OxString::from(prefix_name.into()),
            Iri::parse(OxString::from(prefix_iri.into()))?,
        );
        Ok(self)
    }

    /// Parses a TriG file from a [`Read`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" .
    /// <bar> a schema:Person ;
    ///     schema:name "Bar" ."#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in TriGParser::new().for_reader(file.as_bytes()) {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderTriGParser<R> {
        ReaderTriGParser {
            inner: self.low_level().parser.for_reader(reader),
        }
    }

    /// Parses a TriG file from a [`AsyncRead`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_reader`](Self::for_reader): it feeds the same
    /// incremental parser core, awaiting each chunk instead of blocking on it.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderTriGParser<R> {
        TokioAsyncReaderTriGParser {
            inner: self.low_level().parser.for_tokio_async_reader(reader),
        }
    }

    /// Parses a TriG file from a byte slice.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" .
    /// <bar> a schema:Person ;
    ///     schema:name "Bar" ."#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in TriGParser::new().for_slice(file) {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_slice(self, slice: &(impl AsRef<[u8]> + ?Sized)) -> SliceTriGParser<'_> {
        SliceTriGParser {
            inner: TriGRecognizer::new_parser(
                slice.as_ref(),
                true,
                true,
                self.lenient,
                self.base,
                self.prefixes,
                self.max_buffer_size,
            )
            .into_iter(),
        }
    }

    /// Allows to parse a TriG file by using a low-level API.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file: [&[u8]; 5] = [
    ///     b"@base <http://example.com/>",
    ///     b". @prefix schema: <http://schema.org/> .",
    ///     b"<foo> a schema:Person",
    ///     b" ; schema:name \"Foo\" . <bar>",
    ///     b" a schema:Person ; schema:name \"Bar\" .",
    /// ];
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// let mut parser = TriGParser::new().low_level();
    /// let mut file_chunks = file.iter();
    /// while !parser.is_end() {
    ///     // We feed more data to the parser
    ///     if let Some(chunk) = file_chunks.next() {
    ///         parser.extend_from_slice(chunk);
    ///     } else {
    ///         parser.end(); // It's finished
    ///     }
    ///     // We read as many quads from the parser as possible
    ///     while let Some(quad) = parser.parse_next() {
    ///         let quad = quad?;
    ///         if quad.predicate == rdf::TYPE && quad.object == schema_person {
    ///             count += 1;
    ///         }
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn low_level(self) -> LowLevelTriGParser {
        LowLevelTriGParser {
            parser: TriGRecognizer::new_parser(
                Vec::new(),
                false,
                true,
                self.lenient,
                self.base,
                self.prefixes,
                self.max_buffer_size,
            ),
        }
    }
}

/// Parses a TriG file from a [`Read`] implementation.
///
/// Can be built using [`TriGParser::for_reader`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TriGParser;
///
/// let file = r#"@base <http://example.com/> .
/// @prefix schema: <http://schema.org/> .
/// <foo> a schema:Person ;
///     schema:name "Foo" .
/// <bar> a schema:Person ;
///     schema:name "Bar" ."#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in TriGParser::new().for_reader(file.as_bytes()) {
///     let quad = quad?;
///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct ReaderTriGParser<R: Read> {
    inner: ReaderIterator<R, TriGRecognizer>,
}

impl<R: Read> ReaderTriGParser<R> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TriGParser::new().for_reader(file.as_bytes());
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> TriGPrefixesIter<'_> {
        TriGPrefixesIter {
            inner: self.inner.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TriGParser::new().for_reader(file.as_bytes());
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.inner
            .parser
            .context
            .lexer_options
            .base_iri
            .as_ref()
            .map(Iri::as_str)
    }
}

impl<R: Read> Iterator for ReaderTriGParser<R> {
    type Item = Result<Quad, TurtleParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Parses a TriG file from a [`AsyncRead`] implementation.
///
/// Can be built using [`TriGParser::for_tokio_async_reader`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncReaderTriGParser<R: AsyncRead + Unpin> {
    inner: TokioAsyncReaderIterator<R, TriGRecognizer>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderTriGParser<R> {
    /// Reads the next quad or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<Quad, TurtleParseError>> {
        self.inner.next().await
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest
    /// version will be returned).
    pub fn prefixes(&self) -> TriGPrefixesIter<'_> {
        TriGPrefixesIter {
            inner: self.inner.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    pub fn base_iri(&self) -> Option<&str> {
        self.inner
            .parser
            .context
            .lexer_options
            .base_iri
            .as_ref()
            .map(Iri::as_str)
    }
}

/// Parses a TriG file from a byte slice.
///
/// Can be built using [`TriGParser::for_slice`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TriGParser;
///
/// let file = r#"@base <http://example.com/> .
/// @prefix schema: <http://schema.org/> .
/// <foo> a schema:Person ;
///     schema:name "Foo" .
/// <bar> a schema:Person ;
///     schema:name "Bar" ."#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in TriGParser::new().for_slice(file) {
///     let quad = quad?;
///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct SliceTriGParser<'a> {
    inner: SliceIterator<'a, TriGRecognizer>,
}

impl SliceTriGParser<'_> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TriGParser::new().for_slice(file);
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> TriGPrefixesIter<'_> {
        TriGPrefixesIter {
            inner: self.inner.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TriGParser::new().for_slice(file);
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.inner
            .parser
            .context
            .lexer_options
            .base_iri
            .as_ref()
            .map(Iri::as_str)
    }
}

impl Iterator for SliceTriGParser<'_> {
    type Item = Result<Quad, TurtleSyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Parses a TriG file by using a low-level API.
///
/// Can be built using [`TriGParser::low_level`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TriGParser;
///
/// let file: [&[u8]; 5] = [
///     b"@base <http://example.com/>",
///     b". @prefix schema: <http://schema.org/> .",
///     b"<foo> a schema:Person",
///     b" ; schema:name \"Foo\" . <bar>",
///     b" a schema:Person ; schema:name \"Bar\" .",
/// ];
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// let mut parser = TriGParser::new().low_level();
/// let mut file_chunks = file.iter();
/// while !parser.is_end() {
///     // We feed more data to the parser
///     if let Some(chunk) = file_chunks.next() {
///         parser.extend_from_slice(chunk);
///     } else {
///         parser.end(); // It's finished
///     }
///     // We read as many quads from the parser as possible
///     while let Some(quad) = parser.parse_next() {
///         let quad = quad?;
///         if quad.predicate == rdf::TYPE && quad.object == schema_person {
///             count += 1;
///         }
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
pub struct LowLevelTriGParser {
    parser: Parser<Vec<u8>, TriGRecognizer>,
}

impl LowLevelTriGParser {
    /// Adds some extra bytes to the parser. Should be called when [`parse_next`](Self::parse_next) returns [`None`] and there is still unread data.
    pub fn extend_from_slice(&mut self, other: &[u8]) {
        self.parser.extend_from_slice(other)
    }

    /// Tell the parser that the file is finished.
    ///
    /// This triggers the parsing of the final bytes and might lead [`parse_next`](Self::parse_next) to return some extra values.
    pub fn end(&mut self) {
        self.parser.end()
    }

    /// Returns if the parsing is finished i.e. [`end`](Self::end) has been called and [`parse_next`](Self::parse_next) is always going to return `None`.
    pub fn is_end(&self) -> bool {
        self.parser.is_end()
    }

    /// Attempt to parse a new quad from the already provided data.
    ///
    /// Returns [`None`] if the parsing is finished or more data is required.
    /// If it is the case more data should be fed using [`extend_from_slice`](Self::extend_from_slice).
    pub fn parse_next(&mut self) -> Option<Result<Quad, TurtleSyntaxError>> {
        self.parser.parse_next()
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TriGParser::new().low_level();
    /// parser.extend_from_slice(file.as_bytes());
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.parse_next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> TriGPrefixesIter<'_> {
        TriGPrefixesIter {
            inner: self.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_turtle::TriGParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TriGParser::new().low_level();
    /// parser.extend_from_slice(file.as_bytes());
    /// assert!(parser.base_iri().is_none()); // No base IRI at the beginning
    ///
    /// parser.parse_next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.parser
            .context
            .lexer_options
            .base_iri
            .as_ref()
            .map(Iri::as_str)
    }
}

/// Iterator on the file prefixes.
///
/// See [`LowLevelTriGParser::prefixes`].
pub struct TriGPrefixesIter<'a> {
    inner: Iter<'a, OxString, Iri<OxString>>,
}

impl<'a> Iterator for TriGPrefixesIter<'a> {
    type Item = (&'a str, &'a str);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (key, value) = self.inner.next()?;
        Some((key.as_str(), value.as_str()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// A [TriG](https://www.w3.org/TR/trig/) serializer.
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TriGSerializer;
///
/// let mut serializer = TriGSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .for_writer(Vec::new());
/// serializer.serialize_quad(&Quad::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
///     NamedNode::new("http://example.com")?,
/// ))?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com> {\n\t<http://example.com#me> a schema:Person .\n}\n",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone)]
#[must_use]
pub struct TriGSerializer {
    base_iri: Option<Iri<String>>,
    prefixes: BTreeMap<String, String>,
}

impl TriGSerializer {
    /// Builds a new [`TriGSerializer`].
    #[inline]
    pub fn new() -> Self {
        Self {
            base_iri: None,
            prefixes: BTreeMap::new(),
        }
    }
    /// Registers a namespace prefix so that later prefixed names resolve against it.
    ///
    /// # Errors
    ///
    /// Returns an error if the given prefix IRI is not a valid IRI.
    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: impl Into<String>,
        prefix_iri: impl Into<String>,
    ) -> Result<Self, IriParseError> {
        self.prefixes.insert(
            prefix_name.into(),
            Iri::parse(prefix_iri.into())?.into_inner(),
        );
        Ok(self)
    }

    /// Adds a base IRI to the serialization.
    ///
    /// ```
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::{NamedNode, Quad};
    /// use oxixml_turtle::TriGSerializer;
    ///
    /// let mut serializer = TriGSerializer::new()
    ///     .with_base_iri("http://example.com")?
    ///     .with_prefix("ex", "http://example.com/ns#")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_quad(&Quad::new(
    ///     NamedNode::new("http://example.com/me")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://example.com/ns#Person")?,
    ///     NamedNode::new("http://example.com")?,
    /// ))?;
    /// assert_eq!(
    ///     b"@base <http://example.com> .\n@prefix ex: </ns#> .\n<> {\n\t</me> a ex:Person .\n}\n",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base_iri = Some(Iri::parse(base_iri.into())?);
        Ok(self)
    }

    /// Writes a TriG file to a [`Write`] implementation.
    ///
    /// ```
    /// use oxixml_model::{NamedNode, Quad};
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TriGSerializer;
    ///
    /// let mut serializer = TriGSerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_quad(&Quad::new(
    ///     NamedNode::new("http://example.com#me")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://schema.org/Person")?,
    ///     NamedNode::new("http://example.com")?,
    /// ))?;
    /// assert_eq!(
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com> {\n\t<http://example.com#me> a schema:Person .\n}\n",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_writer<W: Write>(self, writer: W) -> WriterTriGSerializer<W> {
        WriterTriGSerializer {
            writer,
            low_level_writer: self.low_level(),
        }
    }

    /// Writes a TriG file to a [`AsyncWrite`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_writer`](Self::for_writer).
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterTriGSerializer<W> {
        TokioAsyncWriterTriGSerializer {
            writer,
            low_level_writer: self.low_level(),
            buffer: Vec::new(),
        }
    }

    /// Builds a low-level TriG writer.
    ///
    /// ```
    /// use oxixml_model::{NamedNode, Quad};
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TriGSerializer;
    ///
    /// let mut buf = Vec::new();
    /// let mut serializer = TriGSerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .low_level();
    /// serializer.serialize_quad(
    ///     &Quad::new(
    ///         NamedNode::new("http://example.com#me")?,
    ///         rdf::TYPE,
    ///         NamedNode::new("http://schema.org/Person")?,
    ///         NamedNode::new("http://example.com")?,
    ///     ),
    ///     &mut buf,
    /// )?;
    /// serializer.finish(&mut buf)?;
    /// assert_eq!(
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com> {\n\t<http://example.com#me> a schema:Person .\n}\n",
    ///     buf.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn low_level(self) -> LowLevelTriGSerializer {
        // We sort prefixes by decreasing length
        let mut prefixes = self.prefixes.into_iter().collect::<Vec<_>>();
        prefixes.sort_unstable_by_key(|(_, p)| Reverse(p.len()));
        LowLevelTriGSerializer {
            prefixes,
            base_iri: self.base_iri,
            prelude_written: false,
            current_graph_name: GraphName::DefaultGraph,
            current_subject_predicate: None,
        }
    }
}

/// Writes a TriG file to a [`Write`] implementation.
///
/// Can be built using [`TriGSerializer::for_writer`].
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TriGSerializer;
///
/// let mut serializer = TriGSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .for_writer(Vec::new());
/// serializer.serialize_quad(&Quad::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
///     NamedNode::new("http://example.com")?,
/// ))?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com> {\n\t<http://example.com#me> a schema:Person .\n}\n",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct WriterTriGSerializer<W: Write> {
    writer: W,
    low_level_writer: LowLevelTriGSerializer,
}

impl<W: Write> WriterTriGSerializer<W> {
    /// Writes an extra quad.
    pub fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        self.low_level_writer.serialize_quad(quad, &mut self.writer)
    }

    #[doc(hidden)]
    pub fn serialize_triple<'a>(&mut self, triple: impl Into<TripleRef<'a>>) -> io::Result<()> {
        self.low_level_writer
            .serialize_triple(triple, &mut self.writer)
    }

    /// Ends the write process and returns the underlying [`Write`].
    pub fn finish(mut self) -> io::Result<W> {
        self.low_level_writer.finish(&mut self.writer)?;
        Ok(self.writer)
    }
}

/// Writes a TriG file to a [`AsyncWrite`] implementation.
///
/// Can be built using [`TriGSerializer::for_tokio_async_writer`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterTriGSerializer<W: AsyncWrite + Unpin> {
    writer: W,
    low_level_writer: LowLevelTriGSerializer,
    buffer: Vec<u8>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterTriGSerializer<W> {
    /// Writes an extra quad.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn serialize_quad<'a>(&mut self, quad: impl Into<QuadRef<'a>>) -> io::Result<()> {
        self.low_level_writer
            .serialize_quad(quad, &mut self.buffer)?;
        self.writer.write_all(&self.buffer).await?;
        self.buffer.clear();
        Ok(())
    }

    #[doc(hidden)]
    pub async fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
    ) -> io::Result<()> {
        self.low_level_writer
            .serialize_triple(triple, &mut self.buffer)?;
        self.writer.write_all(&self.buffer).await?;
        self.buffer.clear();
        Ok(())
    }

    /// Ends the write process and returns the underlying [`AsyncWrite`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn finish(mut self) -> io::Result<W> {
        self.low_level_writer.finish(&mut self.buffer)?;
        self.writer.write_all(&self.buffer).await?;
        self.buffer.clear();
        Ok(self.writer)
    }
}

/// Writes a TriG file by using a low-level API.
///
/// Can be built using [`TriGSerializer::low_level`].
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TriGSerializer;
///
/// let mut buf = Vec::new();
/// let mut serializer = TriGSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .low_level();
/// serializer.serialize_quad(
///     &Quad::new(
///         NamedNode::new("http://example.com#me")?,
///         rdf::TYPE,
///         NamedNode::new("http://schema.org/Person")?,
///         NamedNode::new("http://example.com")?,
///     ),
///     &mut buf,
/// )?;
/// serializer.finish(&mut buf)?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com> {\n\t<http://example.com#me> a schema:Person .\n}\n",
///     buf.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
pub struct LowLevelTriGSerializer {
    prefixes: Vec<(String, String)>,
    base_iri: Option<Iri<String>>,
    prelude_written: bool,
    current_graph_name: GraphName,
    current_subject_predicate: Option<(NamedOrBlankNode, NamedNode)>,
}

impl LowLevelTriGSerializer {
    /// Writes an extra quad.
    pub fn serialize_quad<'a>(
        &mut self,
        quad: impl Into<QuadRef<'a>>,
        writer: impl Write,
    ) -> io::Result<()> {
        let quad = quad.into();
        self.serialize_quad_parts(
            quad.subject,
            quad.predicate,
            quad.object,
            quad.graph_name,
            writer,
        )
    }

    pub(crate) fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
        writer: impl Write,
    ) -> io::Result<()> {
        let triple = triple.into();
        self.serialize_quad_parts(
            triple.subject,
            triple.predicate,
            triple.object,
            GraphNameRef::DefaultGraph,
            writer,
        )
    }

    fn serialize_quad_parts(
        &mut self,
        subject: NamedOrBlankNodeRef<'_>,
        predicate: NamedNodeRef<'_>,
        object: TermRef<'_>,
        graph_name: GraphNameRef<'_>,
        mut writer: impl Write,
    ) -> io::Result<()> {
        if !self.prelude_written {
            self.prelude_written = true;
            if let Some(base_iri) = &self.base_iri {
                writeln!(writer, "@base <{base_iri}> .")?;
            }
            for (prefix_name, prefix_iri) in &self.prefixes {
                writeln!(
                    writer,
                    "@prefix {prefix_name}: <{}> .",
                    relative_iri(prefix_iri, &self.base_iri)
                )?;
            }
        }
        if graph_name == self.current_graph_name.as_ref() {
            if let Some((current_subject, current_predicate)) =
                self.current_subject_predicate.take()
            {
                if subject == current_subject.as_ref() {
                    if predicate == current_predicate {
                        self.current_subject_predicate = Some((current_subject, current_predicate));
                        write!(writer, " , {}", self.object(object))
                    } else {
                        self.current_subject_predicate =
                            Some((current_subject, predicate.into_owned()));
                        writeln!(writer, " ;")?;
                        if !self.current_graph_name.is_default_graph() {
                            write!(writer, "\t")?;
                        }
                        write!(
                            writer,
                            "\t{} {}",
                            self.predicate(predicate),
                            self.object(object)
                        )
                    }
                } else {
                    self.current_subject_predicate =
                        Some((subject.into_owned(), predicate.into_owned()));
                    writeln!(writer, " .")?;
                    if !self.current_graph_name.is_default_graph() {
                        write!(writer, "\t")?;
                    }
                    write!(
                        writer,
                        "{} {} {}",
                        self.subject(subject),
                        self.predicate(predicate),
                        self.object(object)
                    )
                }
            } else {
                self.current_subject_predicate =
                    Some((subject.into_owned(), predicate.into_owned()));
                if !self.current_graph_name.is_default_graph() {
                    write!(writer, "\t")?;
                }
                write!(
                    writer,
                    "{} {} {}",
                    self.subject(subject),
                    self.predicate(predicate),
                    self.object(object)
                )
            }
        } else {
            if self.current_subject_predicate.is_some() {
                writeln!(writer, " .")?;
            }
            if !self.current_graph_name.is_default_graph() {
                writeln!(writer, "}}")?;
            }
            self.current_graph_name = graph_name.into_owned();
            self.current_subject_predicate = Some((subject.into_owned(), predicate.into_owned()));
            match &self.current_graph_name {
                GraphName::NamedNode(g) => {
                    writeln!(
                        writer,
                        "{} {{",
                        TurtleNamedNode {
                            node: g.as_ref(),
                            prefixes: &self.prefixes,
                            base_iri: &self.base_iri,
                        }
                    )?;
                    write!(writer, "\t")?;
                }
                GraphName::BlankNode(g) => {
                    writeln!(writer, "{g} {{")?;
                    write!(writer, "\t")?;
                }
                GraphName::DefaultGraph => (),
            }

            write!(
                writer,
                "{} {} {}",
                self.subject(subject),
                self.predicate(predicate),
                self.object(object)
            )
        }
    }

    fn subject<'a>(&'a self, node: NamedOrBlankNodeRef<'a>) -> TurtleNamedOrBlankNode<'a> {
        TurtleNamedOrBlankNode {
            node,
            prefixes: &self.prefixes,
            base_iri: &self.base_iri,
        }
    }

    fn predicate<'a>(&'a self, named_node: NamedNodeRef<'a>) -> TurtlePredicate<'a> {
        TurtlePredicate {
            named_node,
            prefixes: &self.prefixes,
            base_iri: &self.base_iri,
        }
    }

    fn object<'a>(&'a self, term: TermRef<'a>) -> TurtleTerm<'a> {
        TurtleTerm {
            term,
            prefixes: &self.prefixes,
            base_iri: &self.base_iri,
        }
    }

    /// Finishes to write the file.
    pub fn finish(&mut self, mut writer: impl Write) -> io::Result<()> {
        if self.current_subject_predicate.is_some() {
            writeln!(writer, " .")?;
        }
        if !self.current_graph_name.is_default_graph() {
            writeln!(writer, "}}")?;
        }
        Ok(())
    }
}

struct TurtleNamedNode<'a> {
    node: NamedNodeRef<'a>,
    prefixes: &'a Vec<(String, String)>,
    base_iri: &'a Option<Iri<String>>,
}

impl fmt::Display for TurtleNamedNode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (prefix_name, prefix_iri) in self.prefixes {
            if let Some(local_name) = self.node.as_str().strip_prefix(prefix_iri) {
                if local_name.is_empty() {
                    return write!(f, "{prefix_name}:");
                } else if let Some(escaped_local_name) = escape_local_name(local_name) {
                    return write!(f, "{prefix_name}:{escaped_local_name}");
                }
            }
        }
        write!(f, "<{}>", relative_iri(self.node.as_str(), self.base_iri))
    }
}

struct TurtlePredicate<'a> {
    named_node: NamedNodeRef<'a>,
    prefixes: &'a Vec<(String, String)>,
    base_iri: &'a Option<Iri<String>>,
}

impl fmt::Display for TurtlePredicate<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.named_node == rdf::TYPE {
            f.write_str("a")
        } else {
            TurtleNamedNode {
                node: self.named_node,
                prefixes: self.prefixes,
                base_iri: self.base_iri,
            }
            .fmt(f)
        }
    }
}

struct TurtleNamedOrBlankNode<'a> {
    node: NamedOrBlankNodeRef<'a>,
    prefixes: &'a Vec<(String, String)>,
    base_iri: &'a Option<Iri<String>>,
}

impl fmt::Display for TurtleNamedOrBlankNode<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.node {
            NamedOrBlankNodeRef::NamedNode(v) => TurtleNamedNode {
                node: v,
                prefixes: self.prefixes,
                base_iri: self.base_iri,
            }
            .fmt(f),
            NamedOrBlankNodeRef::BlankNode(v) => write!(f, "{v}"),
        }
    }
}

struct TurtleTerm<'a> {
    term: TermRef<'a>,
    prefixes: &'a Vec<(String, String)>,
    base_iri: &'a Option<Iri<String>>,
}

impl fmt::Display for TurtleTerm<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.term {
            TermRef::NamedNode(v) => TurtleNamedNode {
                node: v,
                prefixes: self.prefixes,
                base_iri: self.base_iri,
            }
            .fmt(f),
            TermRef::BlankNode(v) => write!(f, "{v}"),
            TermRef::Literal(v) => {
                let value = v.value();
                let datatype = v.datatype();
                let is_plain = {
                    #[cfg(feature = "rdf-12")]
                    {
                        datatype == xsd::STRING
                            || datatype == rdf::LANG_STRING
                            || datatype == rdf::DIR_LANG_STRING
                    }
                    #[cfg(not(feature = "rdf-12"))]
                    {
                        datatype == xsd::STRING || datatype == rdf::LANG_STRING
                    }
                };
                if is_plain {
                    write!(f, "{v}")
                } else {
                    let datatype = v.datatype();
                    if datatype == xsd::BOOLEAN && is_turtle_boolean(value)
                        || datatype == xsd::INTEGER && is_turtle_integer(value)
                        || datatype == xsd::DECIMAL && is_turtle_decimal(value)
                        || datatype == xsd::DOUBLE && is_turtle_double(value)
                    {
                        f.write_str(value)
                    } else {
                        write!(
                            f,
                            "{}^^{}",
                            LiteralRef::new_simple_literal(value),
                            TurtleNamedNode {
                                node: v.datatype(),
                                prefixes: self.prefixes,
                                base_iri: self.base_iri,
                            }
                        )
                    }
                }
            }
            #[cfg(feature = "rdf-12")]
            TermRef::Triple(t) => {
                write!(
                    f,
                    "<<( {} {} {} )>>",
                    TurtleNamedOrBlankNode {
                        node: t.subject.as_ref(),
                        prefixes: self.prefixes,
                        base_iri: self.base_iri,
                    },
                    TurtlePredicate {
                        named_node: t.predicate.as_ref(),
                        prefixes: self.prefixes,
                        base_iri: self.base_iri,
                    },
                    TurtleTerm {
                        term: t.object.as_ref(),
                        prefixes: self.prefixes,
                        base_iri: self.base_iri,
                    }
                )
            }
        }
    }
}

fn relative_iri<'a>(iri: &'a str, base_iri: &Option<Iri<String>>) -> Cow<'a, str> {
    if let Some(base_iri) = base_iri {
        if let Ok(relative) = base_iri.relativize(&Iri::parse_unchecked(iri)) {
            return relative.into_inner().into();
        }
    }
    iri.into()
}

fn is_turtle_boolean(value: &str) -> bool {
    matches!(value, "true" | "false")
}

fn is_turtle_integer(value: &str) -> bool {
    // [19]  INTEGER  ::=  [+-]? [0-9]+
    let mut value = value.as_bytes();
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    !value.is_empty() && value.iter().all(u8::is_ascii_digit)
}

fn is_turtle_decimal(value: &str) -> bool {
    // [20]  DECIMAL  ::=  [+-]? [0-9]* '.' [0-9]+
    let mut value = value.as_bytes();
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    while value.first().is_some_and(u8::is_ascii_digit) {
        value = &value[1..];
    }
    let Some(value) = value.strip_prefix(b".") else {
        return false;
    };
    !value.is_empty() && value.iter().all(u8::is_ascii_digit)
}

fn is_turtle_double(value: &str) -> bool {
    // [21]    DOUBLE    ::=  [+-]? ([0-9]+ '.' [0-9]* EXPONENT | '.' [0-9]+ EXPONENT | [0-9]+ EXPONENT)
    // [154s]  EXPONENT  ::=  [eE] [+-]? [0-9]+
    let mut value = value.as_bytes();
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    let mut with_before = false;
    while value.first().is_some_and(u8::is_ascii_digit) {
        value = &value[1..];
        with_before = true;
    }
    let mut with_after = false;
    if let Some(v) = value.strip_prefix(b".") {
        value = v;
        while value.first().is_some_and(u8::is_ascii_digit) {
            value = &value[1..];
            with_after = true;
        }
    }
    if let Some(v) = value.strip_prefix(b"e") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"E") {
        value = v;
    } else {
        return false;
    }
    if let Some(v) = value.strip_prefix(b"+") {
        value = v;
    } else if let Some(v) = value.strip_prefix(b"-") {
        value = v;
    }
    (with_before || with_after) && !value.is_empty() && value.iter().all(u8::is_ascii_digit)
}

/// Turns an IRI's local part into a `PN_LOCAL`, or returns `None` if it cannot be represented as
/// one at all (the caller then falls back to writing the full `<IRI>` instead of a prefixed
/// name, which is always correct, just less compact).
///
/// ```text
/// [168s]  PN_LOCAL  ::=  (PN_CHARS_U | ':' | [0-9] | PLX) ((PN_CHARS | '.' | ':' | PLX)* (PN_CHARS | ':' | PLX))?
/// ```
///
/// `PLX` is `PERCENT | PN_LOCAL_ESC`; only the `PN_LOCAL_ESC` half is implemented below (via
/// [`can_be_escaped_in_local_name`] and a leading backslash) -- deliberately, not as a missing
/// feature. Every character `PN_LOCAL_ESC` allows escaping (including a literal `%`: it is in
/// `can_be_escaped_in_local_name`'s set) round-trips correctly through a backslash escape, since
/// unescaping a prefixed name only ever drops the backslash and keeps the character, so there is
/// no character this function fails to place *correctly* -- only some it cannot compact at all,
/// which is what the `None` return communicates. The unimplemented `PERCENT` alternative
/// (`'%' HEX HEX`, passed through unescaped rather than backslash-escaped) would only ever apply
/// to a byte that is neither `PN_CHARS`-valid nor in `PN_LOCAL_ESC`'s set -- already forced
/// through the safe `None` fallback above -- or to prettying up a local part that already
/// contains a percent-encoded triplet copied verbatim from the source IRI (`ex:foo%20bar`
/// instead of the equally correct `ex:foo\%20bar` this function produces today).
fn escape_local_name(value: &str) -> Option<String> {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars();
    let first = chars.next()?;
    if N3Lexer::is_possible_pn_chars_u(first) || first == ':' || first.is_ascii_digit() {
        output.push(first);
    } else if can_be_escaped_in_local_name(first) {
        output.push('\\');
        output.push(first);
    } else {
        return None;
    }

    while let Some(c) = chars.next() {
        if N3Lexer::is_possible_pn_chars(c) || c == ':' || (c == '.' && !chars.as_str().is_empty())
        {
            output.push(c);
        } else if can_be_escaped_in_local_name(c) {
            output.push('\\');
            output.push(c);
        } else {
            return None;
        }
    }

    Some(output)
}

fn can_be_escaped_in_local_name(c: char) -> bool {
    matches!(
        c,
        '_' | '~'
            | '.'
            | '-'
            | '!'
            | '$'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | '+'
            | ','
            | ';'
            | '='
            | '/'
            | '?'
            | '#'
            | '@'
            | '%'
    )
}

#[cfg(test)]
#[expect(clippy::panic_in_result_fn)]
mod tests {
    use super::*;
    use oxixml_model::{BlankNode, Literal};
    use std::error::Error;

    #[test]
    fn test_write() -> io::Result<()> {
        let mut serializer = TriGSerializer::new()
            .with_prefix("ex", "http://example.com/")
            .map_err(io::Error::other)?
            .with_prefix("exl", "http://example.com/p/")
            .map_err(io::Error::other)?
            .for_writer(Vec::new());
        serializer.serialize_quad(&Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/p/o."),
            NamedNode::new_unchecked("http://example.com/g"),
        ))?;
        serializer.serialize_quad(&Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/o{o}"),
            NamedNode::new_unchecked("http://example.com/g"),
        ))?;
        serializer.serialize_quad(&Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/"),
            NamedNode::new_unchecked("http://example.com/g"),
        ))?;
        serializer.serialize_quad(&Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            Literal::new_simple_literal("foo"),
            NamedNode::new_unchecked("http://example.com/g"),
        ))?;
        serializer.serialize_quad(&Quad::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p2"),
            Literal::new_language_tagged_literal_unchecked("foo", "en"),
            NamedNode::new_unchecked("http://example.com/g"),
        ))?;
        serializer.serialize_quad(&Quad::new(
            BlankNode::new_unchecked("b"),
            NamedNode::new_unchecked("http://example.com/p2"),
            BlankNode::new_unchecked("b2"),
            NamedNode::new_unchecked("http://example.com/g"),
        ))?;
        serializer.serialize_quad(&Quad::new(
            BlankNode::new_unchecked("b"),
            NamedNode::new_unchecked("http://example.com/p2"),
            Literal::new_typed_literal("true", xsd::BOOLEAN),
            GraphName::DefaultGraph,
        ))?;
        serializer.serialize_quad(&Quad::new(
            BlankNode::new_unchecked("b"),
            NamedNode::new_unchecked("http://example.org/p2"),
            Literal::new_typed_literal("false", xsd::BOOLEAN),
            NamedNode::new_unchecked("http://example.com/g2"),
        ))?;
        assert_eq!(
            String::from_utf8(serializer.finish()?).map_err(io::Error::other)?,
            "@prefix exl: <http://example.com/p/> .\n@prefix ex: <http://example.com/> .\nex:g {\n\tex:s ex:p exl:o\\. , <http://example.com/o{o}> , ex: , \"foo\" ;\n\t\tex:p2 \"foo\"@en .\n\t_:b ex:p2 _:b2 .\n}\n_:b ex:p2 true .\nex:g2 {\n\t_:b <http://example.org/p2> false .\n}\n"
        );
        Ok(())
    }

    /// Regression test: an error recovered from *inside* a `GRAPH { ... }` block used to
    /// unconditionally resynchronize all the way back to the document's top level, forgetting
    /// that the block was still open. The very next (valid) triple inside the same block was
    /// then misparsed as top-level input, and the block's own closing `}` arrived unexpectedly --
    /// turning one genuine error into a cascade of several. Recovery must now stay inside the
    /// block: exactly one error, and every well-formed triple around the bad one -- inside the
    /// block, after it, and after the block itself -- still comes out.
    #[test]
    fn error_inside_a_graph_block_recovers_without_a_spurious_cascade() {
        let input = r#"<http://example.com/g> {
            <http://example.com/s1> <http://example.com/p1> GRAPH .
            <http://example.com/s2> <http://example.com/p2> <http://example.com/o2> .
        }
        <http://example.com/s3> <http://example.com/p3> <http://example.com/o3> ."#;
        let mut quads = Vec::new();
        let mut error_count = 0;
        for item in TriGParser::new().for_slice(input) {
            match item {
                Ok(quad) => quads.push(quad),
                Err(_) => error_count += 1,
            }
        }
        assert_eq!(
            error_count, 1,
            "expected exactly one error, got {error_count}"
        );
        assert_eq!(
            quads,
            vec![
                Quad::new(
                    NamedNode::new_unchecked("http://example.com/s2"),
                    NamedNode::new_unchecked("http://example.com/p2"),
                    NamedNode::new_unchecked("http://example.com/o2"),
                    NamedNode::new_unchecked("http://example.com/g"),
                ),
                Quad::new(
                    NamedNode::new_unchecked("http://example.com/s3"),
                    NamedNode::new_unchecked("http://example.com/p3"),
                    NamedNode::new_unchecked("http://example.com/o3"),
                    GraphName::DefaultGraph,
                ),
            ],
            "{quads:?}"
        );
    }

    /// Regression test: the edge case right next to
    /// [`error_inside_a_graph_block_recovers_without_a_spurious_cascade`] -- recovery resuming
    /// *immediately* into the block's closing `}`, with no well-formed triple in between it and
    /// the `.` that resynchronized on. `Triples` already treats a `}` as "early end" (the same
    /// path an ordinary empty `GRAPH { }` block takes), so this must still produce exactly the
    /// one, original error -- not a second one from misparsing the `}`.
    #[test]
    fn error_immediately_followed_by_the_graph_blocks_closing_brace_is_not_a_cascade() {
        let input = r#"<http://example.com/g> {
            <http://example.com/s1> <http://example.com/p1> GRAPH .
        }
        <http://example.com/s2> <http://example.com/p2> <http://example.com/o2> ."#;
        let mut quads = Vec::new();
        let mut error_count = 0;
        for item in TriGParser::new().for_slice(input) {
            match item {
                Ok(quad) => quads.push(quad),
                Err(_) => error_count += 1,
            }
        }
        assert_eq!(
            error_count, 1,
            "expected exactly one error, got {error_count}"
        );
        assert_eq!(
            quads,
            vec![Quad::new(
                NamedNode::new_unchecked("http://example.com/s2"),
                NamedNode::new_unchecked("http://example.com/p2"),
                NamedNode::new_unchecked("http://example.com/o2"),
                GraphName::DefaultGraph,
            )],
            "{quads:?}"
        );
    }

    /// Regression test: a `GRAPH { ... }` block that errors, resynchronizes on the trailing `.`,
    /// and then the document simply ends without ever supplying the block's closing `}`. This is
    /// a second, independent problem -- the earlier bad token, and a block that (separately) was
    /// truly never closed -- so both must now be reported, since recovery correctly remembers it
    /// is still inside the block. Silently accepting the truncation, as the pre-fix code did, was
    /// itself a symptom of the same lost-track-of-the-block defect
    /// [`error_inside_a_graph_block_recovers_without_a_spurious_cascade`] regression-tests, not a
    /// deliberate leniency worth preserving.
    #[test]
    fn error_then_truncation_inside_a_graph_block_reports_both_problems() {
        let input = "<http://example.com/g> {
            <http://example.com/s1> <http://example.com/p1> GRAPH .
        ";
        let errors: Vec<_> = TriGParser::new()
            .for_slice(input)
            .filter_map(Result::err)
            .collect();
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(
            errors[1].to_string().contains("unexpected end of file"),
            "{errors:?}"
        );
    }

    /// Regression test: `escape_local_name` backslash-escapes a literal `%` in a local part
    /// (rather than passing through the equally valid, unescaped `PERCENT` form); this checks
    /// that choice really does round-trip, for both a percent that is part of what looks like an
    /// already-percent-encoded triplet and one that is not.
    #[test]
    fn prefixed_name_with_a_percent_in_the_local_part_round_trips() -> Result<(), Box<dyn Error>> {
        let write = |iri: NamedNode| -> io::Result<String> {
            let mut serializer = TriGSerializer::new()
                .with_prefix("ex", "http://example.com/")
                .map_err(io::Error::other)?
                .for_writer(Vec::new());
            serializer.serialize_quad(&Quad::new(
                iri,
                NamedNode::new_unchecked("http://example.com/p"),
                NamedNode::new_unchecked("http://example.com/o"),
                GraphName::DefaultGraph,
            ))?;
            Ok(String::from_utf8(serializer.finish()?).expect("utf-8"))
        };

        // A validly-constructed IRI with a genuine percent-encoded triplet: the local part must
        // be written with the `%` backslash-escaped (`ex:foo\%20bar`), and re-parsing that must
        // yield the identical IRI back.
        let iri = NamedNode::new("http://example.com/foo%20bar")?;
        let text = write(iri.clone())?;
        assert!(text.contains(r"ex:foo\%20bar"), "{text}");
        let quads: Vec<_> = TriGParser::new()
            .for_slice(&text)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(quads[0].subject, NamedOrBlankNode::from(iri));

        // A `%` that is not part of a percent-encoded triplet is not a legal IRI byte at all, so
        // this needs `new_unchecked` -- and can never round-trip through re-parsing regardless of
        // how the serializer escapes it, since "100%off" was never a valid IRI to begin with. It
        // still must not corrupt the value at the Turtle-syntax level: written as `ex:100\%off`,
        // not the syntax error `ex:100%off` (a bare, unescaped `%` not followed by two hex
        // digits) would be.
        let text = write(NamedNode::new_unchecked("http://example.com/100%off"))?;
        assert!(text.contains(r"ex:100\%off"), "{text}");
        Ok(())
    }

    /// Regression test: truncated input used to be rejected with the same uninformative
    /// "Unexpected end" message no matter what the parser was in the middle of. Correctness on
    /// invalid input is unchanged (still exactly one error, never a hang or a panic); what
    /// changes is that the message now says what was actually expected.
    #[test]
    fn truncated_input_names_what_was_expected() {
        let cases = [
            ("<http://example.com/s> <http://example.com/p>", "object"),
            (
                "<http://example.com/s> <http://example.com/p> (",
                "collection",
            ),
            (
                "<http://example.com/s> <http://example.com/p> [",
                "property list",
            ),
        ];
        for (input, expected_substring) in cases {
            let errors: Vec<_> = TriGParser::new()
                .for_slice(input)
                .filter_map(Result::err)
                .collect();
            assert_eq!(errors.len(), 1, "{input:?} -> {errors:?}");
            let message = errors[0].to_string();
            assert!(
                message.contains(expected_substring),
                "{input:?} -> {message:?}, expected it to contain {expected_substring:?}"
            );
            assert!(
                !message.contains("Unexpected end"),
                "{input:?} -> {message:?} still uses the generic message"
            );
        }
    }
}
