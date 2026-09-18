//! A [Turtle](https://www.w3.org/TR/turtle/) streaming parser implemented by [`TurtleParser`]
//! and a serializer implemented by [`TurtleSerializer`].

use crate::chunker::get_turtle_slice_chunks;
use crate::terse::TriGRecognizer;
#[cfg(feature = "async-tokio")]
use crate::toolkit::TokioAsyncReaderIterator;
use crate::toolkit::{Parser, ReaderIterator, SliceIterator, TurtleParseError, TurtleSyntaxError};
#[cfg(feature = "async-tokio")]
use crate::trig::TokioAsyncWriterTriGSerializer;
use crate::trig::{LowLevelTriGSerializer, TriGSerializer, WriterTriGSerializer};
use crate::{DEFAULT_MAX_BUFFER_SIZE, MIN_PARALLEL_CHUNK_SIZE};
use oxixml_iri::{Iri, IriParseError};
use oxixml_model::{OxString, Triple, TripleRef};
use std::collections::HashMap;
use std::collections::hash_map::Iter;
use std::io::{self, Read, Write};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncWrite};

/// A [Turtle](https://www.w3.org/TR/turtle/) streaming parser.
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TurtleParser;
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
/// for triple in TurtleParser::new().for_reader(file.as_bytes()) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Clone)]
#[must_use]
pub struct TurtleParser {
    lenient: bool,
    base: Option<Iri<OxString>>,
    prefixes: HashMap<OxString, Iri<OxString>>,
    max_buffer_size: usize,
}

impl Default for TurtleParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TurtleParser {
    /// Builds a new [`TurtleParser`].
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

    /// Parses a Turtle file from a [`Read`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TurtleParser;
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
    /// for triple in TurtleParser::new().for_reader(file.as_bytes()) {
    ///     let triple = triple?;
    ///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderTurtleParser<R> {
        ReaderTurtleParser {
            inner: self.low_level().parser.for_reader(reader),
        }
    }

    /// Parses a Turtle file from a [`AsyncRead`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_reader`](Self::for_reader): it feeds the same
    /// incremental parser core, awaiting each chunk instead of blocking on it.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderTurtleParser<R> {
        TokioAsyncReaderTurtleParser {
            inner: self.low_level().parser.for_tokio_async_reader(reader),
        }
    }

    /// Parses Turtle file from a byte slice.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TurtleParser;
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
    /// for triple in TurtleParser::new().for_slice(file) {
    ///     let triple = triple?;
    ///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_slice(self, slice: &(impl AsRef<[u8]> + ?Sized)) -> SliceTurtleParser<'_> {
        SliceTurtleParser {
            inner: TriGRecognizer::new_parser(
                slice.as_ref(),
                true,
                false,
                self.lenient,
                self.base,
                self.prefixes,
                self.max_buffer_size,
            )
            .into_iter(),
        }
    }

    /// Creates a vector of parsers that may be used to parse a Turtle document slice in parallel.
    /// To dynamically specify target_parallelism, use e.g. [`std::thread::available_parallelism`].
    /// Intended to work on large documents.
    /// Can fail or return wrong results if there are prefixes or base iris that are not defined
    /// at the top of the document, or valid turtle syntax inside literal values.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TurtleParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" .
    /// <bar> a schema:Person ;
    ///     schema:name "Bar" ."#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let readers = TurtleParser::new().split_slice_for_parallel_parsing(file, 2);
    /// let count: usize = std::thread::scope(|scope| {
    ///     let handles: Vec<_> = readers
    ///         .into_iter()
    ///         .map(|reader| {
    ///             let schema_person = schema_person.clone();
    ///             scope.spawn(move || {
    ///         let mut count = 0;
    ///         for triple in reader {
    ///             let triple = triple.expect("oxixml-turtle parser invariant");
    ///             if triple.predicate == rdf::TYPE && triple.object == schema_person {
    ///                 count += 1;
    ///             }
    ///         }
    ///             count
    ///             })
    ///         })
    ///         .collect();
    ///     handles.into_iter().map(|h| h.join().expect("worker thread panicked")).sum()
    /// });
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn split_slice_for_parallel_parsing(
        mut self,
        slice: &(impl AsRef<[u8]> + ?Sized),
        target_parallelism: usize,
    ) -> Vec<SliceTurtleParser<'_>> {
        let slice = slice.as_ref();
        let n_chunks = (slice.len() / MIN_PARALLEL_CHUNK_SIZE).clamp(1, target_parallelism);

        if n_chunks > 1 {
            // Prefixes must be determined before chunks, since determining chunks relies on parser with prefixes determined.
            let mut from_slice_parser = self.clone().for_slice(slice);
            // We don't care about errors: they will be raised when parsing the first chunk anyway
            from_slice_parser.next();
            for (p, iri) in from_slice_parser.prefixes() {
                // Already know this is a valid IRI
                self = self
                    .with_prefix(p, iri)
                    .expect("prefix IRI already validated during parsing");
            }
        }

        get_turtle_slice_chunks(slice, n_chunks, &self)
            .into_iter()
            .map(|(start, end)| self.clone().for_slice(&slice[start..end]))
            .collect()
    }

    /// Allows to parse a Turtle file by using a low-level API.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::NamedNode;
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::TurtleParser;
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
    /// let mut parser = TurtleParser::new().low_level();
    /// let mut file_chunks = file.iter();
    /// while !parser.is_end() {
    ///     // We feed more data to the parser
    ///     if let Some(chunk) = file_chunks.next() {
    ///         parser.extend_from_slice(chunk);
    ///     } else {
    ///         parser.end(); // It's finished
    ///     }
    ///     // We read as many triples from the parser as possible
    ///     while let Some(triple) = parser.parse_next() {
    ///         let triple = triple?;
    ///         if triple.predicate == rdf::TYPE && triple.object == schema_person {
    ///             count += 1;
    ///         }
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn low_level(self) -> LowLevelTurtleParser {
        LowLevelTurtleParser {
            parser: TriGRecognizer::new_parser(
                Vec::new(),
                false,
                false,
                self.lenient,
                self.base,
                self.prefixes,
                self.max_buffer_size,
            ),
        }
    }
}

/// Parses a Turtle file from a [`Read`] implementation.
///
/// Can be built using [`TurtleParser::for_reader`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TurtleParser;
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
/// for triple in TurtleParser::new().for_reader(file.as_bytes()) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct ReaderTurtleParser<R: Read> {
    inner: ReaderIterator<R, TriGRecognizer>,
}

impl<R: Read> ReaderTurtleParser<R> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxixml_turtle::TurtleParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TurtleParser::new().for_reader(file.as_bytes());
    /// assert!(parser.prefixes().collect::<Vec<_>>().is_empty()); // No prefix at the beginning
    ///
    /// parser.next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> TurtlePrefixesIter<'_> {
        TurtlePrefixesIter {
            inner: self.inner.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_turtle::TurtleParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TurtleParser::new().for_reader(file.as_bytes());
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

impl<R: Read> Iterator for ReaderTurtleParser<R> {
    type Item = Result<Triple, TurtleParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.inner.next()?.map(Into::into))
    }
}

/// Parses a Turtle file from a [`AsyncRead`] implementation.
///
/// Can be built using [`TurtleParser::for_tokio_async_reader`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncReaderTurtleParser<R: AsyncRead + Unpin> {
    inner: TokioAsyncReaderIterator<R, TriGRecognizer>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderTurtleParser<R> {
    /// Reads the next triple or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<Triple, TurtleParseError>> {
        Some(self.inner.next().await?.map(Into::into))
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest
    /// version will be returned).
    pub fn prefixes(&self) -> TurtlePrefixesIter<'_> {
        TurtlePrefixesIter {
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

/// Parses a Turtle file from a byte slice.
///
/// Can be built using [`TurtleParser::for_slice`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TurtleParser;
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
/// for triple in TurtleParser::new().for_slice(file) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct SliceTurtleParser<'a> {
    inner: SliceIterator<'a, TriGRecognizer>,
}

impl SliceTurtleParser<'_> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxixml_turtle::TurtleParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TurtleParser::new().for_slice(file);
    /// assert!(parser.prefixes().collect::<Vec<_>>().is_empty()); // No prefix at the beginning
    ///
    /// parser.next().expect("oxixml-turtle parser invariant")?; // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> TurtlePrefixesIter<'_> {
        TurtlePrefixesIter {
            inner: self.inner.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_turtle::TurtleParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TurtleParser::new().for_slice(file);
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

impl Iterator for SliceTurtleParser<'_> {
    type Item = Result<Triple, TurtleSyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.inner.next()?.map(Into::into))
    }
}

/// Parses a Turtle file by using a low-level API.
///
/// Can be built using [`TurtleParser::low_level`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::NamedNode;
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::TurtleParser;
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
/// let mut parser = TurtleParser::new().low_level();
/// let mut file_chunks = file.iter();
/// while !parser.is_end() {
///     // We feed more data to the parser
///     if let Some(chunk) = file_chunks.next() {
///         parser.extend_from_slice(chunk);
///     } else {
///         parser.end(); // It's finished
///     }
///     // We read as many triples from the parser as possible
///     while let Some(triple) = parser.parse_next() {
///         let triple = triple?;
///         if triple.predicate == rdf::TYPE && triple.object == schema_person {
///             count += 1;
///         }
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
pub struct LowLevelTurtleParser {
    parser: Parser<Vec<u8>, TriGRecognizer>,
}

impl LowLevelTurtleParser {
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

    /// Attempt to parse a new triple from the already provided data.
    ///
    /// Returns [`None`] if the parsing is finished or more data is required.
    /// If it is the case more data should be fed using [`extend_from_slice`](Self::extend_from_slice).
    pub fn parse_next(&mut self) -> Option<Result<Triple, TurtleSyntaxError>> {
        Some(self.parser.parse_next()?.map(Into::into))
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxixml_turtle::TurtleParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TurtleParser::new().low_level();
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
    pub fn prefixes(&self) -> TurtlePrefixesIter<'_> {
        TurtlePrefixesIter {
            inner: self.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxixml_turtle::TurtleParser;
    ///
    /// let file = r#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut parser = TurtleParser::new().low_level();
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
/// See [`LowLevelTurtleParser::prefixes`].
pub struct TurtlePrefixesIter<'a> {
    inner: Iter<'a, OxString, Iri<OxString>>,
}

impl<'a> Iterator for TurtlePrefixesIter<'a> {
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

/// A [Turtle](https://www.w3.org/TR/turtle/) serializer.
///
/// ```
/// use oxixml_model::vocab::rdf;
/// use oxixml_model::{NamedNode, Triple};
/// use oxixml_turtle::TurtleSerializer;
///
/// let mut serializer = TurtleSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .for_writer(Vec::new());
/// serializer.serialize_triple(&Triple::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
/// ))?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone)]
#[must_use]
pub struct TurtleSerializer {
    inner: TriGSerializer,
}

impl TurtleSerializer {
    /// Builds a new [`TurtleSerializer`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
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
        self.inner = self.inner.with_prefix(prefix_name, prefix_iri)?;
        Ok(self)
    }

    /// Adds a base IRI to the serialization.
    ///
    /// ```
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::{NamedNode, Triple};
    /// use oxixml_turtle::TurtleSerializer;
    ///
    /// let mut serializer = TurtleSerializer::new()
    ///     .with_base_iri("http://example.com")?
    ///     .with_prefix("ex", "http://example.com/ns#")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_triple(&Triple::new(
    ///     NamedNode::new("http://example.com/me")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://example.com/ns#Person")?,
    /// ))?;
    /// assert_eq!(
    ///     b"@base <http://example.com> .\n@prefix ex: </ns#> .\n</me> a ex:Person .\n",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.inner = self.inner.with_base_iri(base_iri)?;
        Ok(self)
    }

    /// Writes a Turtle file to a [`Write`] implementation.
    ///
    /// ```
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::{NamedNode, Triple};
    /// use oxixml_turtle::TurtleSerializer;
    ///
    /// let mut serializer = TurtleSerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_triple(&Triple::new(
    ///     NamedNode::new("http://example.com#me")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://schema.org/Person")?,
    /// ))?;
    /// assert_eq!(
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_writer<W: Write>(self, writer: W) -> WriterTurtleSerializer<W> {
        WriterTurtleSerializer {
            inner: self.inner.for_writer(writer),
        }
    }

    /// Writes a Turtle file to a [`AsyncWrite`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_writer`](Self::for_writer).
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterTurtleSerializer<W> {
        TokioAsyncWriterTurtleSerializer {
            inner: self.inner.for_tokio_async_writer(writer),
        }
    }

    /// Builds a low-level Turtle writer.
    ///
    /// ```
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::{NamedNode, Triple};
    /// use oxixml_turtle::TurtleSerializer;
    ///
    /// let mut buf = Vec::new();
    /// let mut serializer = TurtleSerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .low_level();
    /// serializer.serialize_triple(
    ///     &Triple::new(
    ///         NamedNode::new("http://example.com#me")?,
    ///         rdf::TYPE,
    ///         NamedNode::new("http://schema.org/Person")?,
    ///     ),
    ///     &mut buf,
    /// )?;
    /// serializer.finish(&mut buf)?;
    /// assert_eq!(
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
    ///     buf.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn low_level(self) -> LowLevelTurtleSerializer {
        LowLevelTurtleSerializer {
            inner: self.inner.low_level(),
        }
    }
}

/// Writes a Turtle file to a [`Write`] implementation.
///
/// Can be built using [`TurtleSerializer::for_writer`].
///
/// ```
/// use oxixml_model::vocab::rdf;
/// use oxixml_model::{NamedNode, Triple};
/// use oxixml_turtle::TurtleSerializer;
///
/// let mut serializer = TurtleSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .for_writer(Vec::new());
/// serializer.serialize_triple(&Triple::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
/// ))?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct WriterTurtleSerializer<W: Write> {
    inner: WriterTriGSerializer<W>,
}

impl<W: Write> WriterTurtleSerializer<W> {
    /// Writes an extra triple.
    pub fn serialize_triple<'a>(&mut self, triple: impl Into<TripleRef<'a>>) -> io::Result<()> {
        self.inner.serialize_triple(triple)
    }

    /// Ends the write process and returns the underlying [`Write`].
    pub fn finish(self) -> io::Result<W> {
        self.inner.finish()
    }
}

/// Writes a Turtle file to a [`AsyncWrite`] implementation.
///
/// Can be built using [`TurtleSerializer::for_tokio_async_writer`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterTurtleSerializer<W: AsyncWrite + Unpin> {
    inner: TokioAsyncWriterTriGSerializer<W>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterTurtleSerializer<W> {
    /// Writes an extra triple.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
    ) -> io::Result<()> {
        self.inner.serialize_triple(triple).await
    }

    /// Ends the write process and returns the underlying [`AsyncWrite`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying writer fails.
    pub async fn finish(self) -> io::Result<W> {
        self.inner.finish().await
    }
}

/// Writes a Turtle file by using a low-level API.
///
/// Can be built using [`TurtleSerializer::low_level`].
///
/// ```
/// use oxixml_model::vocab::rdf;
/// use oxixml_model::{NamedNode, Triple};
/// use oxixml_turtle::TurtleSerializer;
///
/// let mut buf = Vec::new();
/// let mut serializer = TurtleSerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .low_level();
/// serializer.serialize_triple(
///     &Triple::new(
///         NamedNode::new("http://example.com#me")?,
///         rdf::TYPE,
///         NamedNode::new("http://schema.org/Person")?,
///     ),
///     &mut buf,
/// )?;
/// serializer.finish(&mut buf)?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
///     buf.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
pub struct LowLevelTurtleSerializer {
    inner: LowLevelTriGSerializer,
}

impl LowLevelTurtleSerializer {
    /// Writes an extra triple.
    pub fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
        writer: impl Write,
    ) -> io::Result<()> {
        self.inner.serialize_triple(triple, writer)
    }

    /// Finishes to write the file.
    pub fn finish(&mut self, writer: impl Write) -> io::Result<()> {
        self.inner.finish(writer)
    }
}

#[cfg(test)]
#[expect(clippy::panic_in_result_fn)]
mod tests {
    use super::*;
    use oxixml_model::{BlankNode, Literal, NamedNode};

    #[test]
    fn test_write() -> io::Result<()> {
        let mut serializer = TurtleSerializer::new().for_writer(Vec::new());
        serializer.serialize_triple(&Triple::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            NamedNode::new_unchecked("http://example.com/o"),
        ))?;
        serializer.serialize_triple(&Triple::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p"),
            Literal::new_simple_literal("foo"),
        ))?;
        serializer.serialize_triple(&Triple::new(
            NamedNode::new_unchecked("http://example.com/s"),
            NamedNode::new_unchecked("http://example.com/p2"),
            Literal::new_language_tagged_literal_unchecked("foo", "en"),
        ))?;
        serializer.serialize_triple(&Triple::new(
            BlankNode::new_unchecked("b"),
            NamedNode::new_unchecked("http://example.com/p2"),
            BlankNode::new_unchecked("b2"),
        ))?;
        assert_eq!(
            String::from_utf8(serializer.finish()?).map_err(io::Error::other)?,
            "<http://example.com/s> <http://example.com/p> <http://example.com/o> , \"foo\" ;\n\t<http://example.com/p2> \"foo\"@en .\n_:b <http://example.com/p2> _:b2 .\n"
        );
        Ok(())
    }

    /// Regression test: an unrecognized `\x` escape inside an IRIREF (`<...\x...>`) or a long
    /// string (`"""...\x..."""`) used to be reported once, correctly, and then leave the lexer
    /// resuming mid-token: everything after the bad escape up to the next real delimiter was
    /// re-lexed as fresh top-level tokens, each one failing to parse and reporting its own
    /// spurious error on top of the first, genuine one. `TurtleParser` already handled this
    /// correctly for a short string (`"...\x..."`); the fix makes an IRIREF and a long string
    /// consume through to their own closing delimiter the same way, so every case here produces
    /// exactly one error before recovering to the following, well-formed triple.
    #[test]
    fn a_bad_escape_reports_exactly_one_error_and_recovers() {
        let cases = [
            (
                r#"<http://example.com/\zbogus> <http://example.com/p> <http://example.com/o> ."#,
                "an IRIREF",
            ),
            (
                r#"<http://example.com/\z\y\x> <http://example.com/p> <http://example.com/o> ."#,
                "an IRIREF with several bad escapes in a row",
            ),
            (
                "\"\"\"abc\\zdef\"\"\" <http://example.com/p> <http://example.com/o> .",
                "a long string used as a (grammatically invalid, but lexically complete) subject",
            ),
        ];
        for (first_line, description) in cases {
            let input = format!(
                "{first_line}\n<http://example.com/s2> <http://example.com/p2> <http://example.com/o2> ."
            );
            let results: Vec<_> = TurtleParser::new().for_slice(&input).collect();
            let error_count = results.iter().filter(|r| r.is_err()).count();
            assert_eq!(
                error_count, 1,
                "{description}: expected exactly one error, got {results:?}"
            );
            assert!(
                results.iter().any(|r| matches!(r, Ok(triple) if triple.subject.to_string() == "<http://example.com/s2>")),
                "{description}: the following well-formed triple must still parse: {results:?}"
            );
        }
    }
}
