//! Implementation of [`Reader`] for reading from a [`BufRead`] as underlying
//! byte stream.
//!
//! Events read through this path borrow from a caller-provided `Vec<u8>` buffer:
//! bytes are copied out of the underlying reader into that buffer and the events
//! borrow from it, so the same allocation can be reused across reads.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use crate::encoding;
use crate::errors::{Error, Result};
use crate::events::{BytesText, Event};
use crate::name::QName;
use crate::parser::Parser;
use crate::reader::{BangType, ReadRefResult, ReadTextResult, Reader, Span, XmlSource};
use crate::utils::is_whitespace;

/// Generates the body of [`XmlSource`] for a buffered byte source, either
/// synchronous ([`BufRead`]) or, with the `async-tokio` feature, asynchronous
/// (`tokio`'s `AsyncBufRead`).
///
/// This single definition is reused by both the synchronous implementation
/// below and the asynchronous implementation in [`async_tokio`](super::async_tokio),
/// so the byte-level scanning logic (searching for `<`, `&`, `>`, comment/CDATA
/// terminators, …) lives in exactly one place.
///
/// # Parameters
/// - `$lf`: lifetime used on methods that need one only in the async form
///   (the sync trait methods already have it in scope via the `impl` block)
/// - `$reader`: optional field access (e.g. `0`) used to reach the wrapped
///   reader when `Self` is an adapter struct instead of the reader itself
/// - `$async`/`$await`: optional `async`/`.await` tokens, present only for the
///   asynchronous instantiation
macro_rules! impl_buffered_source {
    ($($lf:lifetime, $reader:tt, $async:ident, $await:ident)?) => {
        #[cfg(not(feature = "encoding"))]
        #[inline]
        $($async)? fn remove_utf8_bom(&mut self) -> io::Result<()> {
            loop {
                break match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) => {
                        if n.starts_with(encoding::UTF8_BOM) {
                            self $(.$reader)? .consume(encoding::UTF8_BOM.len());
                        }
                        Ok(())
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => Err(e),
                };
            }
        }

        #[cfg(feature = "encoding")]
        #[inline]
        $($async)? fn detect_encoding(&mut self) -> io::Result<Option<encoding::DetectedEncoding>> {
            loop {
                break match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) => {
                        if let Some(detected) = encoding::detect_encoding(n) {
                            self $(.$reader)? .consume(detected.bom_len());
                            Ok(Some(detected))
                        } else {
                            Ok(None)
                        }
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => Err(e),
                };
            }
        }

        #[inline]
        $($async)? fn read_text $(<$lf>)? (
            &mut self,
            buf: &'b mut Vec<u8>,
            position: &mut u64,
            trim_whitespace_only: bool,
        ) -> ReadTextResult<'b, &'b mut Vec<u8>> {
            let mut read = 0;
            let start = buf.len();
            loop {
                let available = match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) if n.is_empty() => break,
                    Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        *position += read;
                        return ReadTextResult::Err(e);
                    }
                };

                // Search for start of markup or an entity or character reference
                match oxixml_support::bytes::memchr2(b'<', b'&', available) {
                    // Special handling is needed only on the first iteration.
                    // On next iterations we already read something and should emit Text event
                    Some(0) if read == 0 && available[0] == b'<' => return ReadTextResult::Markup(buf),
                    // Do not consume `&` because it may be lone and we would be need to
                    // return it as part of Text event
                    Some(0) if read == 0 => return ReadTextResult::Ref(buf),
                    Some(i) if available[i] == b'<' => {
                        buf.extend_from_slice(&available[..i]);

                        self $(.$reader)? .consume(i);
                        read += i as u64;

                        *position += read;
                        if trim_whitespace_only
                            && buf[start..].iter().all(|b| crate::utils::is_whitespace(*b))
                        {
                            // Trimming would leave nothing, so report that markup
                            // starts here rather than an empty text run.
                            buf.truncate(start);
                            return ReadTextResult::Markup(buf);
                        }
                        return ReadTextResult::UpToMarkup(&buf[start..]);
                    }
                    Some(i) => {
                        buf.extend_from_slice(&available[..i]);

                        self $(.$reader)? .consume(i);
                        read += i as u64;

                        *position += read;
                        if trim_whitespace_only
                            && buf[start..].iter().all(|b| crate::utils::is_whitespace(*b))
                        {
                            // Trimming would leave nothing, so report that a
                            // reference starts here rather than an empty text run.
                            buf.truncate(start);
                            return ReadTextResult::Ref(buf);
                        }
                        return ReadTextResult::UpToRef(&buf[start..]);
                    }
                    None => {
                        buf.extend_from_slice(available);

                        let used = available.len();
                        self $(.$reader)? .consume(used);
                        read += used as u64;
                    }
                }
            }

            *position += read;
            ReadTextResult::UpToEof(&buf[start..])
        }

        #[inline]
        $($async)? fn read_ref $(<$lf>)? (
            &mut self,
            buf: &'b mut Vec<u8>,
            position: &mut u64,
        ) -> ReadRefResult<'b> {
            let mut read = 0;
            let start = buf.len();
            loop {
                let available = match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) if n.is_empty() => break,
                    Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        *position += read;
                        return ReadRefResult::Err(e);
                    }
                };
                // `read_ref` called when the first character is `&`, so we
                // should explicitly skip it at first iteration lest we confuse
                // it with the end
                if read == 0 {
                    debug_assert!(
                        available.starts_with(b"&"),
                        "`read_ref` must be called at `&`:\n{:?}",
                        crate::utils::Bytes(available)
                    );
                    // If that ampersand is lone, then it will be part of text
                    // and we should keep it
                    buf.push(b'&');
                    self $(.$reader)? .consume(1);
                    read += 1;
                    continue;
                }

                match oxixml_support::bytes::memchr3(b';', b'&', b'<', available) {
                    Some(i) if available[i] == b';' => {
                        // +1 -- skip the end `;`
                        let used = i + 1;

                        buf.extend_from_slice(&available[..used]);
                        self $(.$reader)? .consume(used);
                        read += used as u64;

                        *position += read;

                        return ReadRefResult::Ref(&buf[start..]);
                    }
                    // Do not consume `&` because it may be lone and we would be need to
                    // return it as part of Text event
                    Some(i) => {
                        let is_amp = available[i] == b'&';
                        buf.extend_from_slice(&available[..i]);

                        self $(.$reader)? .consume(i);
                        read += i as u64;

                        *position += read;

                        return if is_amp {
                            ReadRefResult::UpToRef(&buf[start..])
                        } else {
                            ReadRefResult::UpToMarkup(&buf[start..])
                        };
                    }
                    None => {
                        buf.extend_from_slice(available);

                        let used = available.len();
                        self $(.$reader)? .consume(used);
                        read += used as u64;
                    }
                }
            }

            *position += read;
            ReadRefResult::UpToEof(&buf[start..])
        }

        #[inline]
        $($async)? fn read_with<$($lf,)? P: Parser>(
            &mut self,
            mut parser: P,
            buf: &'b mut Vec<u8>,
            position: &mut u64,
        ) -> Result<&'b [u8]> {
            let mut read = 1;
            let start = buf.len();
            // '<' was consumed in peek_one(), but not placed in buf
            buf.push(b'<');
            loop {
                let available = match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) if n.is_empty() => break,
                    Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        *position += read;
                        return Err(Error::from(e));
                    }
                };

                if let Some(i) = parser.feed(available) {
                    let used = i + 1; // +1 for `>`
                    buf.extend_from_slice(&available[..used]);

                    self $(.$reader)? .consume(used);
                    read += used as u64;

                    *position += read;
                    return Ok(&buf[start..]);
                }

                // The `>` symbol not yet found, continue reading
                buf.extend_from_slice(available);

                let used = available.len();
                self $(.$reader)? .consume(used);
                read += used as u64;
            }

            *position += read;
            Err(Error::Syntax(parser.eof_error(&buf[start..])))
        }

        #[inline]
        $($async)? fn read_bang_element $(<$lf>)? (
            &mut self,
            buf: &'b mut Vec<u8>,
            position: &mut u64,
        ) -> Result<(BangType, &'b [u8])> {
            // Peeked '<!' before being called, so it's guaranteed to start with it.
            let start = buf.len();
            let mut read = 2;
            // '<' was consumed in peek_one(), but not placed in buf
            buf.push(b'<');
            buf.push(b'!');
            self $(.$reader)? .consume(1);

            let mut bang_type = loop {
                break match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) => BangType::new(n.first().copied())?,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Error::from(e)),
                };
            };

            loop {
                let available = match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) if n.is_empty() => break,
                    Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        *position += read;
                        return Err(Error::from(e));
                    }
                };
                // We only parse from start because we don't want to consider
                // whatever is in the buffer before the bang element
                if let Some(i) = bang_type.feed(&buf[start..], available) {
                    let consumed = i + 1; // +1 for `>`
                    buf.extend_from_slice(&available[..consumed]);

                    self $(.$reader)? .consume(consumed);
                    read += consumed as u64;

                    *position += read;
                    return Ok((bang_type, &buf[start..]));
                }

                // The `>` symbol not yet found, continue reading
                buf.extend_from_slice(available);

                let used = available.len();
                self $(.$reader)? .consume(used);
                read += used as u64;
            }

            *position += read;
            Err(Error::Syntax(bang_type.to_err()))
        }

        #[inline]
        $($async)? fn skip_whitespace(&mut self, position: &mut u64) -> io::Result<()> {
            loop {
                break match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) => {
                        let count = n.iter().position(|b| !is_whitespace(*b)).unwrap_or(n.len());
                        if count > 0 {
                            self $(.$reader)? .consume(count);
                            *position += count as u64;
                            continue;
                        } else {
                            Ok(())
                        }
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => Err(e),
                };
            }
        }

        #[inline]
        $($async)? fn peek_one(&mut self) -> io::Result<Option<u8>> {
            // That method is called only when available buffer starts from '<'
            // We need to consume it
            self $(.$reader)? .consume(1);
            let available = loop {
                break match self $(.$reader)? .fill_buf() $(.$await)? {
                    Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                };
            };
            Ok(available.first().copied())
        }
    };
}

// Made available to the async implementation, which lives in a sibling module
// and reuses the exact same scanning logic via `TokioAdapter`.
#[cfg(feature = "async-tokio")]
pub(super) use impl_buffered_source;

/// Implementation of `XmlSource` for any `BufRead` reader using a user-given
/// `Vec<u8>` as buffer that will be borrowed by events.
impl<'b, R: BufRead> XmlSource<'b, &'b mut Vec<u8>> for R {
    impl_buffered_source!();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// This is an implementation for reading from a [`BufRead`] as underlying byte stream.
impl<R: BufRead> Reader<R> {
    /// Reads the next `Event`.
    ///
    /// This is the main entry point for reading XML `Event`s when the input is a
    /// generic [`BufRead`].
    ///
    /// `Event`s borrow `buf` and can be converted to own their data if needed
    /// (uses `Cow` internally).
    ///
    /// Having the possibility to control the internal buffers gives you some
    /// additional benefits such as reducing the number of allocations by reusing
    /// the same buffer (call `buf.clear()` once you are done with the event).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_xml::events::Event;
    /// use oxixml_xml::reader::Reader;
    ///
    /// let xml = r#"<tag1 att1 = "test">
    ///                 <tag2><!--Test comment-->Test</tag2>
    ///                 <tag2>Test 2</tag2>
    ///              </tag1>"#;
    /// let mut reader = Reader::from_reader(xml.as_bytes());
    /// reader.config_mut().trim_text(true);
    /// let mut count = 0;
    /// let mut buf = Vec::new();
    /// let mut txt = Vec::new();
    /// loop {
    ///     match reader.read_event_into(&mut buf) {
    ///         Ok(Event::Start(_)) => count += 1,
    ///         Ok(Event::Text(e)) => txt.push(e.decode().expect("utf-8").into_owned()),
    ///         Err(e) => panic!("Error at position {}: {:?}", reader.error_position(), e),
    ///         Ok(Event::Eof) => break,
    ///         _ => (),
    ///     }
    ///     buf.clear();
    /// }
    /// assert_eq!(count, 3);
    /// assert_eq!(txt, vec!["Test".to_string(), "Test 2".to_string()]);
    /// ```
    #[inline]
    pub fn read_event_into<'b>(&mut self, buf: &'b mut Vec<u8>) -> Result<Event<'b>> {
        self.read_event_impl(buf)
    }

    /// Reads until end element is found using provided buffer as intermediate
    /// storage for events content. This function is supposed to be called after
    /// you already read a [`Start`] event.
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
    /// [`Start`]: Event::Start
    /// [`End`]: Event::End
    /// [`BytesStart::to_end()`]: crate::events::BytesStart::to_end
    /// [`Error::IllFormed`]: crate::errors::Error::IllFormed
    /// [`expand_empty_elements`]: crate::reader::Config::expand_empty_elements
    pub fn read_to_end_into(&mut self, end: QName, buf: &mut Vec<u8>) -> Result<Span> {
        Ok(read_to_end!(self, end, buf, read_event_impl, {
            buf.clear();
        }))
    }

    /// Reads content between start and end tags, including any markup, using the
    /// provided buffer as intermediate storage for events content. This function
    /// is supposed to be called after you already read a [`Start`] event.
    ///
    /// Manages nested cases where parent and child elements have the _literally_
    /// same name.
    ///
    /// This method does not unescape read data, instead it returns content
    /// "as is" of the XML document.
    ///
    /// [`Start`]: Event::Start
    pub fn read_text_into<'b>(
        &mut self,
        end: QName,
        buf: &'b mut Vec<u8>,
    ) -> Result<BytesText<'b>> {
        let start = buf.len();
        let span = read_to_end!(self, end, buf, read_event_impl, {});

        let len = span.end - span.start;
        // `buf` may contain not more than isize::MAX bytes and because it is
        // not cleared when reading event, length of the returned span should fit into
        // usize (because otherwise we panic at appending to the buffer before that point)
        let end = start + len as usize;

        Ok(BytesText::wrap(&buf[start..end], self.decoder()))
    }
}

impl Reader<BufReader<File>> {
    /// Creates an XML reader from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self::from_reader(reader))
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
        1,
        &mut Vec::new()
    );
}
