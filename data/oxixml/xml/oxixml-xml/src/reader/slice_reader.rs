//! Implementation of [`Reader`] for reading from a `&[u8]` as underlying byte
//! stream.
//!
//! This implementation supports not using an intermediate buffer as the byte
//! slice itself can be used to borrow from, providing a zero-copy path.

use std::io;

use oxixml_support::bytes::{memchr2, memchr3};

#[cfg(feature = "encoding")]
use crate::encoding::DetectedEncoding;
use crate::errors::{Error, Result};
use crate::events::{BytesText, Event};
use crate::name::QName;
use crate::parser::Parser;
#[cfg(feature = "encoding")]
use crate::reader::EncodingRef;
use crate::reader::{BangType, ReadRefResult, ReadTextResult, Reader, Span, XmlSource};
use crate::utils::is_whitespace;
#[cfg(feature = "encoding")]
use oxixml_encoding::Encoding;

/// This is an implementation for reading from a `&[u8]` as underlying byte stream.
/// This implementation supports not using an intermediate buffer as the byte slice
/// itself can be used to borrow from.
impl<'a> Reader<&'a [u8]> {
    /// Creates an XML reader from a string slice.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &'a str) -> Self {
        // Rust strings are guaranteed to be UTF-8, so lock the encoding
        #[cfg(feature = "encoding")]
        {
            let mut reader = Self::from_reader(s.as_bytes());
            reader.state.encoding = EncodingRef::Explicit(Encoding::Utf8);
            reader
        }

        #[cfg(not(feature = "encoding"))]
        Self::from_reader(s.as_bytes())
    }

    /// Read an event that borrows from the input rather than a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::Event;
    /// use oxixml_xml::reader::Reader;
    ///
    /// let mut reader = Reader::from_str(r#"
    ///     <tag1 att1 = "test">
    ///        <tag2><!--Test comment-->Test</tag2>
    ///        <tag2>Test 2</tag2>
    ///     </tag1>
    /// "#);
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut count = 0;
    /// let mut txt = Vec::new();
    /// loop {
    ///     match reader.read_event().expect("valid event") {
    ///         Event::Start(_) => count += 1,
    ///         Event::Text(e) => txt.push(e.decode().expect("utf-8").into_owned()),
    ///         Event::Eof => break,
    ///         _ => (),
    ///     }
    /// }
    /// assert_eq!(count, 3);
    /// assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
    /// ```
    #[inline]
    pub fn read_event(&mut self) -> Result<Event<'a>> {
        self.read_event_impl(())
    }

    /// Reads until end element is found. This function is supposed to be called
    /// after you already read a [`Start`] event.
    ///
    /// Returns a span that cover content between `>` of an opening tag and `<` of
    /// a closing tag or an empty slice, if [`expand_empty_elements`] is set and
    /// this method was called after reading expanded [`Start`] event.
    ///
    /// Manages nested cases where parent and child elements have the _literally_
    /// same name.
    ///
    /// If a corresponding [`End`] event is not found, an error of type
    /// [`Error::IllFormed`] will be returned.
    ///
    /// The `end` parameter should contain name of the end element _in the reader
    /// encoding_. It is good practice to always get that parameter using
    /// [`BytesStart::to_end()`] method.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::{BytesStart, Event};
    /// use oxixml_xml::reader::Reader;
    ///
    /// let mut reader = Reader::from_str(r#"
    ///     <outer>
    ///         <inner>
    ///             <inner></inner>
    ///             <inner/>
    ///             <outer></outer>
    ///             <outer/>
    ///         </inner>
    ///     </outer>
    /// "#);
    /// reader.config_mut().trim_text(true);
    ///
    /// let start = BytesStart::new("outer");
    /// let end   = start.to_end().into_owned();
    ///
    /// // First, we read a start event...
    /// assert_eq!(reader.read_event().expect("start"), Event::Start(start));
    ///
    /// // ...then, we could skip all events to the corresponding end event.
    /// // This call will correctly handle nested <outer> elements.
    /// reader.read_to_end(end.name()).expect("end");
    ///
    /// // At the end we should get an Eof event, because we ate the whole XML
    /// assert_eq!(reader.read_event().expect("eof"), Event::Eof);
    /// ```
    ///
    /// [`Start`]: Event::Start
    /// [`End`]: Event::End
    /// [`BytesStart::to_end()`]: crate::events::BytesStart::to_end
    /// [`Error::IllFormed`]: crate::errors::Error::IllFormed
    /// [`expand_empty_elements`]: crate::reader::Config::expand_empty_elements
    pub fn read_to_end(&mut self, end: QName) -> Result<Span> {
        Ok(read_to_end!(self, end, (), read_event_impl, {}))
    }

    /// Reads content between start and end tags, including any markup. This
    /// function is supposed to be called after you already read a [`Start`] event.
    ///
    /// Manages nested cases where parent and child elements have the _literally_
    /// same name.
    ///
    /// This method does not unescape read data, instead it returns content
    /// "as is" of the XML document. Any text will be decoded using the XML current
    /// [`decoder()`].
    ///
    /// [`Start`]: Event::Start
    /// [`decoder()`]: Self::decoder()
    pub fn read_text(&mut self, end: QName) -> Result<BytesText<'a>> {
        // self.reader will be changed, so store original reference
        let buffer = self.reader;
        let span = self.read_to_end(end)?;

        let len = span.end - span.start;
        // `span` can only contain indexes up to usize::MAX because it
        // was created from offsets from a single &[u8] slice
        Ok(BytesText::wrap(&buffer[0..len as usize], self.decoder()))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Implementation of `XmlSource` for `&[u8]` reader using a `Self` as buffer
/// that will be borrowed by events. This implementation provides a zero-copy
/// deserialization.
impl<'a> XmlSource<'a, ()> for &'a [u8] {
    #[cfg(not(feature = "encoding"))]
    #[inline]
    fn remove_utf8_bom(&mut self) -> io::Result<()> {
        if self.starts_with(crate::encoding::UTF8_BOM) {
            *self = &self[crate::encoding::UTF8_BOM.len()..];
        }
        Ok(())
    }

    #[cfg(feature = "encoding")]
    #[inline]
    fn detect_encoding(&mut self) -> io::Result<Option<DetectedEncoding>> {
        if let Some(detected) = crate::encoding::detect_encoding(self) {
            *self = &self[detected.bom_len()..];
            return Ok(Some(detected));
        }
        Ok(None)
    }

    #[inline]
    fn read_text(
        &mut self,
        _buf: (),
        position: &mut u64,
        trim_whitespace_only: bool,
    ) -> ReadTextResult<'a, ()> {
        // Search for start of markup or an entity or character reference
        match memchr2(b'<', b'&', self) {
            Some(0) if self[0] == b'<' => ReadTextResult::Markup(()),
            // Do not consume `&` because it may be lone and we would be need to
            // return it as part of Text event
            Some(0) => ReadTextResult::Ref(()),
            Some(i) if self[i] == b'<' => {
                let (bytes, rest) = self.split_at(i);
                *self = rest;
                *position += i as u64;
                if trim_whitespace_only && bytes.iter().all(|b| is_whitespace(*b)) {
                    // Trimming would leave nothing, so report that markup starts
                    // here rather than an empty text run.
                    ReadTextResult::Markup(())
                } else {
                    ReadTextResult::UpToMarkup(bytes)
                }
            }
            Some(i) => {
                let (bytes, rest) = self.split_at(i);
                *self = rest;
                *position += i as u64;
                if trim_whitespace_only && bytes.iter().all(|b| is_whitespace(*b)) {
                    // Trimming would leave nothing, so report that a reference
                    // starts here rather than an empty text run.
                    ReadTextResult::Ref(())
                } else {
                    ReadTextResult::UpToRef(bytes)
                }
            }
            None => {
                let bytes = &self[..];
                *self = &[];
                *position += bytes.len() as u64;
                ReadTextResult::UpToEof(bytes)
            }
        }
    }

    #[inline]
    fn read_ref(&mut self, _buf: (), position: &mut u64) -> ReadRefResult<'a> {
        debug_assert!(
            self.starts_with(b"&"),
            "`read_ref` must be called at `&`:\n{:?}",
            crate::utils::Bytes(self)
        );
        // Search for the end of reference or a start of another reference or a markup
        match memchr3(b';', b'&', b'<', &self[1..]) {
            Some(i) if self[i + 1] == b';' => {
                // +1 for the start `&`
                // +1 for the end `;`
                let end = i + 2;
                let (bytes, rest) = self.split_at(end);
                *self = rest;
                *position += end as u64;

                ReadRefResult::Ref(bytes)
            }
            // Do not consume `&` because it may be lone and we would be need to
            // return it as part of Text event
            Some(i) => {
                let is_amp = self[i + 1] == b'&';
                let (bytes, rest) = self.split_at(i + 1);
                *self = rest;
                *position += i as u64 + 1;

                if is_amp {
                    ReadRefResult::UpToRef(bytes)
                } else {
                    ReadRefResult::UpToMarkup(bytes)
                }
            }
            None => {
                let bytes = &self[..];
                *self = &[];
                *position += bytes.len() as u64;

                ReadRefResult::UpToEof(bytes)
            }
        }
    }

    #[inline]
    fn read_with<P>(&mut self, mut parser: P, _buf: (), position: &mut u64) -> Result<&'a [u8]>
    where
        P: Parser,
    {
        if let Some(i) = parser.feed(self) {
            let used = i + 1; // +1 for `>`
            *position += used as u64;
            let (bytes, rest) = self.split_at(used);
            *self = rest;
            return Ok(bytes);
        }

        *position += self.len() as u64;
        Err(Error::Syntax(parser.eof_error(self)))
    }

    #[inline]
    fn read_bang_element(&mut self, _buf: (), position: &mut u64) -> Result<(BangType, &'a [u8])> {
        // Peeked one bang ('!') before being called, so it's guaranteed to
        // start with it.
        debug_assert!(
            self.starts_with(b"<!"),
            "`read_bang_element` must be called at `<!`:\n{:?}",
            crate::utils::Bytes(self)
        );

        let mut bang_type = BangType::new(self.get(2).copied())?;

        if let Some(i) = bang_type.feed(&[], self) {
            let consumed = i + 1; // +1 for `>`
            *position += consumed as u64;
            let (bytes, rest) = self.split_at(consumed);
            *self = rest;
            return Ok((bang_type, bytes));
        }

        *position += self.len() as u64;
        Err(Error::Syntax(bang_type.to_err()))
    }

    #[inline]
    fn skip_whitespace(&mut self, position: &mut u64) -> io::Result<()> {
        let whitespaces = self
            .iter()
            .position(|b| !is_whitespace(*b))
            .unwrap_or(self.len());
        *position += whitespaces as u64;
        *self = &self[whitespaces..];
        Ok(())
    }

    #[inline]
    fn peek_one(&mut self) -> io::Result<Option<u8>> {
        debug_assert!(
            self.starts_with(b"<"),
            "markup must start from '<':\n{:?}",
            crate::utils::Bytes(self)
        );
        Ok(self.get(1).copied())
    }
}

#[cfg(test)]
mod test {
    use crate::reader::XmlSource;
    use crate::reader::check_tests::check;

    /// Default buffer constructor just pass the byte array from the test
    fn identity<T>(input: T) -> T {
        input
    }

    check!(
        #[test]
        read_event_impl,
        identity,
        0,
        ()
    );
}
