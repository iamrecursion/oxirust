//! A [N-Quads](https://www.w3.org/TR/n-quads/) streaming parser implemented by [`NQuadsParser`]
//! and a serializer implemented by [`NQuadsSerializer`].

use crate::chunker::{get_ntriples_file_chunks, get_ntriples_slice_chunks};
use crate::line_formats::NQuadsRecognizer;
#[cfg(feature = "async-tokio")]
use crate::toolkit::TokioAsyncReaderIterator;
use crate::toolkit::{Parser, ReaderIterator, SliceIterator, TurtleParseError, TurtleSyntaxError};
use crate::{DEFAULT_MAX_BUFFER_SIZE, MIN_PARALLEL_CHUNK_SIZE};
use oxixml_model::{Quad, QuadRef, TripleRef};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Take, Write};
use std::path::Path;
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

/// A [N-Quads](https://www.w3.org/TR/n-quads/) streaming parser.
///
/// Count the number of people:
/// ```
/// use oxixml_model::{NamedNode, vocab::rdf};
/// use oxixml_turtle::NQuadsParser;
///
/// let file = r#"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
/// <http://example.com/foo> <http://schema.org/name> "Foo" .
/// <http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
/// <http://example.com/bar> <http://schema.org/name> "Bar" ."#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in NQuadsParser::new().for_reader(file.as_bytes()) {
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
pub struct NQuadsParser {
    lenient: bool,
    max_buffer_size: usize,
}

impl Default for NQuadsParser {
    fn default() -> Self {
        Self::new()
    }
}

impl NQuadsParser {
    /// Builds a new [`NQuadsParser`].
    #[inline]
    pub fn new() -> Self {
        Self {
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            lenient: false,
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

    /// Parses a N-Quads file from a [`Read`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::{NamedNode, vocab::rdf};
    /// use oxixml_turtle::NQuadsParser;
    ///
    /// let file = r#"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// <http://example.com/foo> <http://schema.org/name> "Foo" .
    /// <http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// <http://example.com/bar> <http://schema.org/name> "Bar" ."#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in NQuadsParser::new().for_reader(file.as_bytes()) {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderNQuadsParser<R> {
        ReaderNQuadsParser {
            inner: self.low_level().parser.for_reader(reader),
        }
    }

    /// Parses a N-Quads file from a [`AsyncRead`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_reader`](Self::for_reader): it feeds the same
    /// incremental parser core, awaiting each chunk instead of blocking on it.
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderNQuadsParser<R> {
        TokioAsyncReaderNQuadsParser {
            inner: self.low_level().parser.for_tokio_async_reader(reader),
        }
    }

    /// Parses a N-Quads file from a byte slice.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::{NamedNode, vocab::rdf};
    /// use oxixml_turtle::NQuadsParser;
    ///
    /// let file = r#"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// <http://example.com/foo> <http://schema.org/name> "Foo" .
    /// <http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// <http://example.com/bar> <http://schema.org/name> "Bar" ."#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in NQuadsParser::new().for_slice(file) {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_slice(self, slice: &(impl AsRef<[u8]> + ?Sized)) -> SliceNQuadsParser<'_> {
        SliceNQuadsParser {
            inner: NQuadsRecognizer::new_parser(
                slice.as_ref(),
                true,
                true,
                self.lenient,
                self.max_buffer_size,
            )
            .into_iter(),
        }
    }

    /// Creates a vector of parsers that may be used to parse an NQuads document slice in parallel.
    /// To dynamically specify target_parallelism, use e.g. [`std::thread::available_parallelism`].
    /// Intended to work on large documents.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::NamedNode;
    /// use oxixml_turtle::NQuadsParser;
    ///
    /// let file = r#"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// <http://example.com/foo> <http://schema.org/name> "Foo" .
    /// <http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// <http://example.com/bar> <http://schema.org/name> "Bar" ."#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let readers = NQuadsParser::new().split_slice_for_parallel_parsing(file, 2);
    /// let count: usize = std::thread::scope(|scope| {
    ///     let handles: Vec<_> = readers
    ///         .into_iter()
    ///         .map(|reader| {
    ///             let schema_person = schema_person.clone();
    ///             scope.spawn(move || {
    ///         let mut count = 0;
    ///         for quad in reader {
    ///             let quad = quad.expect("oxixml-turtle parser invariant");
    ///             if quad.predicate == rdf::TYPE && quad.object == schema_person {
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
        self,
        slice: &(impl AsRef<[u8]> + ?Sized),
        target_parallelism: usize,
    ) -> Vec<SliceNQuadsParser<'_>> {
        let slice = slice.as_ref();
        let n_chunks = (slice.len() / MIN_PARALLEL_CHUNK_SIZE).clamp(1, target_parallelism);
        get_ntriples_slice_chunks(slice, n_chunks)
            .into_iter()
            .map(|(start, end)| self.clone().for_slice(&slice[start..end]))
            .collect()
    }

    /// Creates a vector of parsers that may be used to parse an NQuads file in parallel.
    /// To dynamically specify target_parallelism, use e.g. [`std::thread::available_parallelism`].
    /// Intended to work on large documents.
    ///
    /// Count the number of people:
    /// ```no_run
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_model::NamedNode;
    /// use oxixml_turtle::NTriplesParser;
    /// # let path = std::env::temp_dir().join("oxixml_turtle_parallel_example");
    /// # std::fs::write(&path, r#"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// # <http://example.com/foo> <http://schema.org/name> "Foo" .
    /// # <http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
    /// # <http://example.com/bar> <http://schema.org/name> "Bar" ."#)?;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let readers = NTriplesParser::new().split_file_for_parallel_parsing(&path, 2)?;
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
    pub fn split_file_for_parallel_parsing(
        self,
        path: impl AsRef<Path>,
        target_parallelism: usize,
    ) -> io::Result<Vec<ReaderNQuadsParser<Take<File>>>> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let n_chunks = usize::try_from(
            file_size / u64::try_from(MIN_PARALLEL_CHUNK_SIZE).map_err(io::Error::other)?,
        )
        .map_err(io::Error::other)?
        .clamp(1, target_parallelism);
        get_ntriples_file_chunks(&mut file, file_size, n_chunks)?
            .into_iter()
            .map(|(start, end)| {
                let mut file = File::open(path)?;
                file.seek(SeekFrom::Start(start))?;
                Ok(self.clone().for_reader(file.take(end - start)))
            })
            .collect()
    }

    /// Allows parsing an N-Quads file by using a low-level API.
    ///
    /// Count the number of people:
    /// ```
    /// use oxixml_model::{NamedNode, vocab::rdf};
    /// use oxixml_turtle::NQuadsParser;
    ///
    /// let file: [&[u8]; 4] = [
    ///     b"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .\n",
    ///     b"<http://example.com/foo> <http://schema.org/name> \"Foo\" .\n",
    ///     b"<http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .\n",
    ///     b"<http://example.com/bar> <http://schema.org/name> \"Bar\" .\n"
    /// ];
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// let mut parser = NQuadsParser::new().low_level();
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
    pub fn low_level(self) -> LowLevelNQuadsParser {
        LowLevelNQuadsParser {
            parser: NQuadsRecognizer::new_parser(
                Vec::new(),
                false,
                true,
                self.lenient,
                self.max_buffer_size,
            ),
        }
    }
}

/// Parses a N-Quads file from a [`Read`] implementation.
///
/// Can be built using [`NQuadsParser::for_reader`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::{NamedNode, vocab::rdf};
/// use oxixml_turtle::NQuadsParser;
///
/// let file = r#"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
/// <http://example.com/foo> <http://schema.org/name> "Foo" .
/// <http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
/// <http://example.com/bar> <http://schema.org/name> "Bar" ."#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in NQuadsParser::new().for_reader(file.as_bytes()) {
///     let quad = quad?;
///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct ReaderNQuadsParser<R: Read> {
    inner: ReaderIterator<R, NQuadsRecognizer>,
}

impl<R: Read> Iterator for ReaderNQuadsParser<R> {
    type Item = Result<Quad, TurtleParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Parses a N-Quads file from a [`AsyncRead`] implementation.
///
/// Can be built using [`NQuadsParser::for_tokio_async_reader`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncReaderNQuadsParser<R: AsyncRead + Unpin> {
    inner: TokioAsyncReaderIterator<R, NQuadsRecognizer>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderNQuadsParser<R> {
    /// Reads the next quad or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<Quad, TurtleParseError>> {
        self.inner.next().await
    }
}

/// Parses an N-Quads file from a byte slice.
///
/// Can be built using [`NQuadsParser::for_slice`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::{NamedNode, vocab::rdf};
/// use oxixml_turtle::NQuadsParser;
///
/// let file = r#"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
/// <http://example.com/foo> <http://schema.org/name> "Foo" .
/// <http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .
/// <http://example.com/bar> <http://schema.org/name> "Bar" ."#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in NQuadsParser::new().for_slice(file) {
///     let quad = quad?;
///     if quad.predicate == rdf::TYPE && quad.object == schema_person {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct SliceNQuadsParser<'a> {
    inner: SliceIterator<'a, NQuadsRecognizer>,
}

impl Iterator for SliceNQuadsParser<'_> {
    type Item = Result<Quad, TurtleSyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Parses a N-Quads file by using a low-level API.
///
/// Can be built using [`NQuadsParser::low_level`].
///
/// Count the number of people:
/// ```
/// use oxixml_model::{NamedNode, vocab::rdf};
/// use oxixml_turtle::NQuadsParser;
///
/// let file: [&[u8]; 4] = [
///     b"<http://example.com/foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .\n",
///     b"<http://example.com/foo> <http://schema.org/name> \"Foo\" .\n",
///     b"<http://example.com/bar> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .\n",
///     b"<http://example.com/bar> <http://schema.org/name> \"Bar\" .\n"
/// ];
///
/// let schema_person = NamedNode::new("http://schema.org/Person")?;
/// let mut count = 0;
/// let mut parser = NQuadsParser::new().low_level();
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
pub struct LowLevelNQuadsParser {
    parser: Parser<Vec<u8>, NQuadsRecognizer>,
}

impl LowLevelNQuadsParser {
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
}

/// A [N-Quads](https://www.w3.org/TR/n-quads/) serializer.
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::NQuadsSerializer;
///
/// let mut serializer = NQuadsSerializer::new().for_writer(Vec::new());
/// serializer.serialize_quad(&Quad::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
///     NamedNode::new("http://example.com")?,
/// ))?;
/// assert_eq!(
///     b"<http://example.com#me> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> <http://example.com> .\n",
///     serializer.finish().as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone)]
#[must_use]
#[expect(clippy::empty_structs_with_brackets)]
pub struct NQuadsSerializer {}

impl NQuadsSerializer {
    /// Builds a new [`NQuadsSerializer`].
    #[inline]
    pub fn new() -> Self {
        Self {}
    }

    /// Writes a N-Quads file to a [`Write`] implementation.
    ///
    /// ```
    /// use oxixml_model::{NamedNode, Quad};
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::NQuadsSerializer;
    ///
    /// let mut serializer = NQuadsSerializer::new().for_writer(Vec::new());
    /// serializer.serialize_quad(&Quad::new(
    ///     NamedNode::new("http://example.com#me")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://schema.org/Person")?,
    ///     NamedNode::new("http://example.com")?,
    /// ))?;
    /// assert_eq!(
    ///     b"<http://example.com#me> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> <http://example.com> .\n",
    ///     serializer.finish().as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_writer<W: Write>(self, writer: W) -> WriterNQuadsSerializer<W> {
        WriterNQuadsSerializer {
            writer,
            low_level_writer: self.low_level(),
        }
    }

    /// Writes a N-Quads file to a [`AsyncWrite`] implementation.
    ///
    /// This is the asynchronous counterpart of [`for_writer`](Self::for_writer).
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterNQuadsSerializer<W> {
        TokioAsyncWriterNQuadsSerializer {
            writer,
            low_level_writer: self.low_level(),
            buffer: Vec::new(),
        }
    }

    /// Builds a low-level N-Quads writer.
    ///
    /// ```
    /// use oxixml_model::{NamedNode, Quad};
    /// use oxixml_model::vocab::rdf;
    /// use oxixml_turtle::NQuadsSerializer;
    ///
    /// let mut buf = Vec::new();
    /// let mut serializer = NQuadsSerializer::new().low_level();
    /// serializer.serialize_quad(&Quad::new(
    ///     NamedNode::new("http://example.com#me")?,
    ///     rdf::TYPE,
    ///     NamedNode::new("http://schema.org/Person")?,
    ///     NamedNode::new("http://example.com")?,
    /// ), &mut buf)?;
    /// assert_eq!(
    ///     b"<http://example.com#me> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> <http://example.com> .\n",
    ///     buf.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn low_level(self) -> LowLevelNQuadsSerializer {
        LowLevelNQuadsSerializer {}
    }
}

/// Writes a N-Quads file to a [`Write`] implementation.
///
/// Can be built using [`NQuadsSerializer::for_writer`].
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::NQuadsSerializer;
///
/// let mut serializer = NQuadsSerializer::new().for_writer(Vec::new());
/// serializer.serialize_quad(&Quad::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
///     NamedNode::new("http://example.com")?,
/// ))?;
/// assert_eq!(
///     b"<http://example.com#me> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> <http://example.com> .\n",
///     serializer.finish().as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct WriterNQuadsSerializer<W: Write> {
    writer: W,
    low_level_writer: LowLevelNQuadsSerializer,
}

impl<W: Write> WriterNQuadsSerializer<W> {
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
    pub fn finish(self) -> W {
        self.writer
    }
}

/// Writes a N-Quads file to a [`AsyncWrite`] implementation.
///
/// Can be built using [`NQuadsSerializer::for_tokio_async_writer`].
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterNQuadsSerializer<W: AsyncWrite + Unpin> {
    writer: W,
    low_level_writer: LowLevelNQuadsSerializer,
    buffer: Vec<u8>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterNQuadsSerializer<W> {
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
    pub fn finish(self) -> W {
        self.writer
    }
}

/// Writes a N-Quads file by using a low-level API.
///
/// Can be built using [`NQuadsSerializer::low_level`].
///
/// ```
/// use oxixml_model::{NamedNode, Quad};
/// use oxixml_model::vocab::rdf;
/// use oxixml_turtle::NQuadsSerializer;
///
/// let mut buf = Vec::new();
/// let mut serializer = NQuadsSerializer::new().low_level();
/// serializer.serialize_quad(&Quad::new(
///     NamedNode::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNode::new("http://schema.org/Person")?,
///     NamedNode::new("http://example.com")?,
/// ), &mut buf)?;
/// assert_eq!(
///     b"<http://example.com#me> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> <http://example.com> .\n",
///     buf.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[expect(clippy::empty_structs_with_brackets)]
pub struct LowLevelNQuadsSerializer {}

impl LowLevelNQuadsSerializer {
    /// Writes an extra quad.
    pub fn serialize_quad<'a>(
        &mut self,
        quad: impl Into<QuadRef<'a>>,
        mut writer: impl Write,
    ) -> io::Result<()> {
        writeln!(writer, "{} .", quad.into())
    }

    #[doc(hidden)]
    pub fn serialize_triple<'a>(
        &mut self,
        triple: impl Into<TripleRef<'a>>,
        mut writer: impl Write,
    ) -> io::Result<()> {
        writeln!(writer, "{} .", triple.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    #[test]
    fn test_parse_nquads_above_max_default_size() {
        // Ensure that the parser can be configured to parse files above the default max buffer size
        // and that it does not return an error when parsing such files
        let unbound_parser = NQuadsParser::default().with_max_buffer_size(usize::MAX);
        let bounded_parser = NQuadsParser::default();
        assert_eq!(bounded_parser.max_buffer_size, DEFAULT_MAX_BUFFER_SIZE);

        // Create a literal > 16 MiB (16 MiB + 1 byte)
        let large_literal = "x".repeat(16 * 1024 * 1024 + 1);

        let file =
            format!(r#"<http://example.com/foo> <http://schema.org/name> "{large_literal}" ."#);

        for quad in bounded_parser.for_reader(file.as_bytes()) {
            let err = quad.expect_err("expected out-of-memory error");
            assert!(
                matches!(&err, TurtleParseError::Io(e)
                    if e.kind() == ErrorKind::OutOfMemory
                ),
                "expected out of memory error, got {err:?}"
            );
        }

        for quad in unbound_parser.for_reader(file.as_bytes()) {
            assert!(
                quad.is_ok(),
                "Expected no parsing issue when parsing a large literal with an unbound buffer but got {quad:?}"
            );
        }
    }

    /// Regression test: truncated input used to be rejected with the same uninformative
    /// "Unexpected end" message regardless of what the parser was in the middle of. Each of
    /// these must still be rejected -- correctness on invalid input is unchanged -- but now with
    /// a message naming what was actually expected, not the generic one.
    #[test]
    fn truncated_input_names_what_was_expected() {
        let cases = [
            (
                "<http://example.com/s>",
                "expected a predicate after the subject",
            ),
            (
                "<http://example.com/s> <http://example.com/p>",
                "expected an object after the predicate",
            ),
            (
                r#"<http://example.com/s> <http://example.com/p> "value"^^"#,
                "expected a literal datatype IRI after '^^'",
            ),
        ];
        for (input, expected_substring) in cases {
            let errors: Vec<_> = NQuadsParser::new()
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
