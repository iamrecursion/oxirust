//! Asynchronous reading from a [`tokio::io::AsyncBufRead`] source.
//!
//! This is an implementation of [`Reader`] and [`NsReader`] for reading from an
//! [`AsyncBufRead`] as underlying byte stream. This reader fully implements
//! async/await so reading can use non-blocking I/O. It mirrors quick-xml 0.41's
//! `reader::async_tokio` module: the byte-scanning logic is shared with the
//! synchronous [`BufRead`](std::io::BufRead) implementation through the
//! [`impl_buffered_source!`](crate::reader::buffered_reader) macro (reused here
//! via [`TokioAdapter`]), so the sansio parser core in [`state`](super::state)
//! stays the single source of truth for both.

use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{self, AsyncBufRead, AsyncBufReadExt, AsyncRead, ReadBuf};

use crate::encoding;
use crate::errors::{Error, IllFormedError, Result, SyntaxError};
use crate::events::{BytesRef, BytesText, Event};
use crate::name::{QName, ResolveResult};
use crate::parser::{ElementParser, Parser, PiParser};
use crate::reader::buffered_reader::impl_buffered_source;
use crate::reader::{
    BangType, BinaryStream, NsReader, ParseState, ReadRefResult, ReadTextResult, Reader, Span,
};
use crate::utils::is_whitespace;

/// A struct that reads XML asynchronously from an [`AsyncBufRead`].
///
/// Having its own struct lets us implement anything without risk of name
/// conflicts with the synchronous [`XmlSource`](crate::reader::XmlSource)
/// implementation, and sidesteps the impossibility of having `async` methods
/// directly in that trait.
struct TokioAdapter<'a, R>(&'a mut R);

impl<'a, R: AsyncBufRead + Unpin> TokioAdapter<'a, R> {
    impl_buffered_source!('b, 0, async, await);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<R> AsyncRead for BinaryStream<'_, R>
where
    R: AsyncRead + Unpin,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let start = buf.remaining();
        let this = self.get_mut();
        let poll = Pin::new(&mut *this.inner).poll_read(cx, buf);

        // If something was read, update offset
        if let Poll::Ready(Ok(())) = poll {
            let amt = start - buf.remaining();
            *this.offset += amt as u64;
        }
        poll
    }
}

impl<R> AsyncBufRead for BinaryStream<'_, R>
where
    R: AsyncBufRead + Unpin,
{
    #[inline]
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<&[u8]>> {
        Pin::new(&mut *self.get_mut().inner).poll_fill_buf(cx)
    }

    #[inline]
    fn consume(self: Pin<&mut Self>, amt: usize) {
        let this = self.get_mut();
        this.inner.consume(amt);
        *this.offset += amt as u64;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<R: AsyncBufRead + Unpin> Reader<R> {
    /// An asynchronous version of [`read_event_into()`]. Reads the next event
    /// into the given buffer.
    ///
    /// This is the main entry point for reading XML `Event`s when using an
    /// async reader.
    ///
    /// See the documentation of [`read_event_into()`] for more information.
    ///
    /// # Examples
    ///
    /// ```
    /// # use oxixml_xml::events::Event;
    /// # use oxixml_xml::reader::Reader;
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// // This explicitly uses `from_reader("...".as_bytes())` to use a buffered
    /// // reader instead of relying on the zero-copy optimizations for reading
    /// // from byte slices, which provides the sync interface anyway.
    /// let mut reader = Reader::from_reader(
    ///     r#"
    ///     <tag1 att1 = "test">
    ///        <tag2><!--Test comment-->Test</tag2>
    ///        <tag2>Test 2</tag2>
    ///     </tag1>
    /// "#
    ///     .as_bytes(),
    /// );
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut count = 0;
    /// let mut buf = Vec::new();
    /// let mut txt = Vec::new();
    /// loop {
    ///     match reader.read_event_into_async(&mut buf).await {
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
    /// # }
    /// ```
    ///
    /// [`read_event_into()`]: Reader::read_event_into
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn read_event_into_async<'b>(
        &mut self,
        mut buf: &'b mut Vec<u8>,
    ) -> Result<Event<'b>> {
        read_event_impl!(
            self,
            buf,
            TokioAdapter(&mut self.reader),
            read_until_close_async,
            await
        )
    }

    /// An asynchronous version of [`read_to_end_into()`]. Reads asynchronously
    /// until the matching end element is found, using the provided buffer as
    /// intermediate storage for events content. This function is supposed to be
    /// called after you already read a [`Start`] event.
    ///
    /// See the documentation of [`read_to_end_into()`] for more information.
    ///
    /// # Examples
    ///
    /// This example shows how you can skip XML content after you read the start
    /// event.
    ///
    /// ```
    /// # use oxixml_xml::events::{BytesStart, Event};
    /// # use oxixml_xml::reader::Reader;
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let mut reader = Reader::from_reader(
    ///     r#"
    ///     <outer>
    ///         <inner>
    ///             <inner></inner>
    ///             <inner/>
    ///             <outer></outer>
    ///             <outer/>
    ///         </inner>
    ///     </outer>
    /// "#
    ///     .as_bytes(),
    /// );
    /// reader.config_mut().trim_text(true);
    /// let mut buf = Vec::new();
    ///
    /// let start = BytesStart::new("outer");
    /// let end = start.to_end().into_owned();
    ///
    /// // First, we read a start event...
    /// assert_eq!(
    ///     reader.read_event_into_async(&mut buf).await.expect("start"),
    ///     Event::Start(start)
    /// );
    ///
    /// // ...then, we could skip all events to the corresponding end event.
    /// // This call will correctly handle nested <outer> elements.
    /// // Note, however, that this method does not handle namespaces.
    /// reader
    ///     .read_to_end_into_async(end.name(), &mut buf)
    ///     .await
    ///     .expect("end");
    ///
    /// // At the end we should get an Eof event, because we ate the whole XML
    /// assert_eq!(
    ///     reader.read_event_into_async(&mut buf).await.expect("eof"),
    ///     Event::Eof
    /// );
    /// # }
    /// ```
    ///
    /// [`read_to_end_into()`]: Self::read_to_end_into
    /// [`Start`]: Event::Start
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn read_to_end_into_async<'n>(
        &mut self,
        // We should name that lifetime due to https://github.com/rust-lang/rust/issues/63033
        end: QName<'n>,
        buf: &mut Vec<u8>,
    ) -> Result<Span> {
        Ok(read_to_end!(
            self,
            end,
            buf,
            read_event_into_async,
            {
                buf.clear();
            },
            await
        ))
    }

    /// An asynchronous version of [`read_text_into()`]. Reads asynchronously
    /// content between start and end tags, including any markup, using the
    /// provided buffer as intermediate storage for events content. This
    /// function is supposed to be called after you already read a [`Start`]
    /// event.
    ///
    /// See the documentation of [`read_text_into()`] for more information.
    ///
    /// [`read_text_into()`]: Self::read_text_into
    /// [`Start`]: Event::Start
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn read_text_into_async<'n, 'b>(
        &mut self,
        // We should name that lifetime due to https://github.com/rust-lang/rust/issues/63033
        end: QName<'n>,
        buf: &'b mut Vec<u8>,
    ) -> Result<BytesText<'b>> {
        let start = buf.len();
        let span = read_to_end!(self, end, buf, read_event_into_async, {}, await);

        let len = span.end - span.start;
        // `buf` may contain not more than isize::MAX bytes and because it is
        // not cleared when reading event, length of the returned span should fit
        // into usize (because otherwise we panic at appending to the buffer
        // before that point)
        let end = start + len as usize;

        Ok(BytesText::wrap(&buf[start..end], self.decoder()))
    }

    /// Private function to read until `>` is found. This function expects that
    /// it was called just after encounter a `<` symbol.
    async fn read_until_close_async<'b>(&mut self, buf: &'b mut Vec<u8>) -> Result<Event<'b>> {
        read_until_close!(self, buf, TokioAdapter(&mut self.reader), await)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<R: AsyncBufRead + Unpin> NsReader<R> {
    /// An asynchronous version of [`read_event_into()`]. Reads the next event
    /// into the given buffer.
    ///
    /// This method manages namespaces but doesn't resolve them automatically.
    /// You should call [`resolve_element()`] if you want to get a namespace.
    ///
    /// You also can use [`read_resolved_event_into_async()`] instead if you want
    /// to resolve namespaces as soon as you get an event.
    ///
    /// [`read_event_into()`]: NsReader::read_event_into
    /// [`resolve_element()`]: NsReader::resolve_element
    /// [`read_resolved_event_into_async()`]: Self::read_resolved_event_into_async
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn read_event_into_async<'b>(&mut self, buf: &'b mut Vec<u8>) -> Result<Event<'b>> {
        self.pop();
        let event = self.reader.read_event_into_async(buf).await;
        self.process_event(event)
    }

    /// An asynchronous version of [`read_to_end_into()`]. Reads asynchronously
    /// until the matching end element is found, using the provided buffer as
    /// intermediate storage for events content. This function is supposed to be
    /// called after you already read a [`Start`] event.
    ///
    /// See the documentation of [`read_to_end_into()`] for more information.
    ///
    /// [`read_to_end_into()`]: Self::read_to_end_into
    /// [`Start`]: Event::Start
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn read_to_end_into_async<'n>(
        &mut self,
        // We should name that lifetime due to https://github.com/rust-lang/rust/issues/63033
        end: QName<'n>,
        buf: &mut Vec<u8>,
    ) -> Result<Span> {
        // According to https://www.w3.org/TR/xml11/#dt-etag the end name should
        // match the start name literally. See `Config::check_end_names`.
        let result = self.reader.read_to_end_into_async(end, buf).await?;
        // `read_to_end_into_async` consumed the closing tag. Because nobody can
        // access its content anymore, we directly pop the opening tag's
        // namespace scope.
        self.resolver_mut().pop();
        Ok(result)
    }

    /// An asynchronous version of [`read_text_into()`]. Reads asynchronously
    /// content between start and end tags, including any markup, using the
    /// provided buffer as intermediate storage for events content. This
    /// function is supposed to be called after you already read a [`Start`]
    /// event.
    ///
    /// See the documentation of [`read_text_into()`] for more information.
    ///
    /// [`read_text_into()`]: Self::read_text_into
    /// [`Start`]: Event::Start
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn read_text_into_async<'n, 'b>(
        &mut self,
        // We should name that lifetime due to https://github.com/rust-lang/rust/issues/63033
        end: QName<'n>,
        buf: &'b mut Vec<u8>,
    ) -> Result<BytesText<'b>> {
        let result = self.reader.read_text_into_async(end, buf).await?;
        // `read_text_into_async` consumed the closing tag, so pop its namespace
        // scope.
        self.resolver_mut().pop();
        Ok(result)
    }

    /// An asynchronous version of [`read_resolved_event_into()`]. Reads the next
    /// event into the given buffer asynchronously and resolves its namespace
    /// (if applicable).
    ///
    /// Namespace is resolved only for [`Start`], [`Empty`] and [`End`] events.
    /// For all other events the concept of namespace is not defined, so a
    /// [`ResolveResult::Unbound`] is returned.
    ///
    /// If you are not interested in namespaces, you can use
    /// [`read_event_into_async()`] which will not automatically resolve
    /// namespaces for you.
    ///
    /// [`read_resolved_event_into()`]: NsReader::read_resolved_event_into
    /// [`Start`]: Event::Start
    /// [`Empty`]: Event::Empty
    /// [`End`]: Event::End
    /// [`read_event_into_async()`]: Self::read_event_into_async
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn read_resolved_event_into_async<'ns, 'b>(
        // Name 'ns lifetime, because otherwise we get an error
        // "implicit elided lifetime not allowed here" on ResolveResult
        &'ns mut self,
        buf: &'b mut Vec<u8>,
    ) -> Result<(ResolveResult<'ns>, Event<'b>)> {
        let event = self.read_event_into_async(buf).await?;
        Ok(self.resolver().resolve_event(event))
    }
}

#[cfg(test)]
mod test {
    use super::TokioAdapter;
    use crate::reader::check_tests::check;

    check!(
        #[tokio::test]
        read_event_into_async,
        TokioAdapter,
        1,
        &mut Vec::new(),
        async,
        await
    );

    #[test]
    fn test_future_is_send() {
        // This test should just compile, no actual runtime checks are performed here.
        use super::*;
        use tokio::io::BufReader;

        fn check_send<T: Send>(_: T) {}

        let input: Vec<u8> = vec![];
        let mut reading_buf = vec![];
        let mut reader = Reader::from_reader(BufReader::new(input.as_slice()));

        check_send(reader.read_event_into_async(&mut reading_buf));
    }
}
