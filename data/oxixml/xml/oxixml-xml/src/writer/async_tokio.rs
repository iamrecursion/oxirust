//! Asynchronous writing to a [`tokio::io::AsyncWrite`] sink.
//!
//! Mirrors quick-xml 0.41's `writer::async_tokio` module: an async counterpart
//! of every [`Writer::write_event`]/[`ElementWriter`] method that performs I/O,
//! built on the same event-dispatch logic as the synchronous path but writing
//! through [`tokio::io::AsyncWriteExt::write_all`] instead of
//! [`std::io::Write::write_all`].

use std::future::Future;
use std::result::Result as StdResult;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::errors::{Error, Result};
use crate::events::{BytesCData, BytesPI, BytesText, Event};
use crate::{ElementWriter, Writer};

impl<W: AsyncWrite + Unpin> Writer<W> {
    /// Writes the given event to the underlying writer. Async version of
    /// [`Writer::write_event`].
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn write_event_async<'a, E: Into<Event<'a>>>(&mut self, event: E) -> Result<()> {
        let mut next_should_line_break = true;
        let result = match event.into() {
            Event::Start(e) => {
                let result = self.write_wrapped_async(b"<", &e, b">").await;
                if let Some(i) = self.indent.as_mut() {
                    i.grow();
                }
                result
            }
            Event::End(e) => {
                if let Some(i) = self.indent.as_mut() {
                    i.shrink();
                }
                self.write_wrapped_async(b"</", &e, b">").await
            }
            Event::Empty(e) => self.write_wrapped_async(b"<", &e, b"/>").await,
            Event::Text(e) => {
                next_should_line_break = false;
                self.write_async(&e).await
            }
            Event::Comment(e) => self.write_wrapped_async(b"<!--", &e, b"-->").await,
            Event::CData(e) => {
                next_should_line_break = false;
                self.write_async(b"<![CDATA[").await?;
                self.write_async(&e).await?;
                self.write_async(b"]]>").await
            }
            Event::Decl(e) => self.write_wrapped_async(b"<?", &e, b"?>").await,
            Event::PI(e) => self.write_wrapped_async(b"<?", &e, b"?>").await,
            Event::DocType(e) => self.write_wrapped_async(b"<!DOCTYPE ", &e, b">").await,
            Event::GeneralRef(e) => self.write_wrapped_async(b"&", &e, b";").await,
            Event::Eof => Ok(()),
        };
        if let Some(i) = self.indent.as_mut() {
            i.should_line_break = next_should_line_break;
        }
        result
    }

    /// Manually writes a newline and indentation at the proper level. Async
    /// version of [`Writer::write_indent`].
    ///
    /// This method does nothing if `Writer` was not constructed with
    /// [`Writer::new_with_indent`].
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn write_indent_async(&mut self) -> Result<()> {
        if let Some(ref i) = self.indent {
            self.writer.write_all(b"\n").await?;
            self.writer.write_all(i.current()).await?;
        }
        Ok(())
    }

    #[inline]
    async fn write_async(&mut self, value: &[u8]) -> Result<()> {
        self.writer.write_all(value).await.map_err(Error::from)
    }

    #[inline]
    async fn write_wrapped_async(
        &mut self,
        before: &[u8],
        value: &[u8],
        after: &[u8],
    ) -> Result<()> {
        if let Some(ref i) = self.indent {
            if i.should_line_break {
                self.writer.write_all(b"\n").await?;
                self.writer.write_all(i.current()).await?;
            }
        }
        self.write_async(before).await?;
        self.write_async(value).await?;
        self.write_async(after).await?;
        Ok(())
    }
}

impl<'a, W: AsyncWrite + Unpin> ElementWriter<'a, W> {
    /// Writes some text inside the current element. Async version of
    /// [`ElementWriter::write_text_content`].
    ///
    /// # Example
    ///
    /// ```
    /// # use oxixml_xml::events::BytesText;
    /// # use oxixml_xml::writer::Writer;
    /// # use tokio::io::AsyncWriteExt;
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let mut buffer = Vec::new();
    /// let mut tokio_buffer = tokio::io::BufWriter::new(&mut buffer);
    /// let mut writer = Writer::new_with_indent(&mut tokio_buffer, b' ', 4);
    ///
    /// writer
    ///     .create_element("paired")
    ///     .with_attribute(("attr1", "value1"))
    ///     .with_attribute(("attr2", "value2"))
    ///     .write_text_content_async(BytesText::new("text"))
    ///     .await
    ///     .expect("cannot write content");
    ///
    /// tokio_buffer.flush().await.expect("flush failed");
    ///
    /// assert_eq!(
    ///     std::str::from_utf8(&buffer).expect("utf-8"),
    ///     r#"<paired attr1="value1" attr2="value2">text</paired>"#
    /// );
    /// # }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn write_text_content_async(self, text: BytesText<'_>) -> Result<&'a mut Writer<W>> {
        self.writer
            .write_event_async(Event::Start(self.start_tag.borrow()))
            .await?;
        self.writer.write_event_async(Event::Text(text)).await?;
        self.writer
            .write_event_async(Event::End(self.start_tag.to_end()))
            .await?;
        Ok(self.writer)
    }

    /// Writes a CData event `<![CDATA[...]]>` inside the current element. Async
    /// version of [`ElementWriter::write_cdata_content`].
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn write_cdata_content_async(
        self,
        text: BytesCData<'_>,
    ) -> Result<&'a mut Writer<W>> {
        self.writer
            .write_event_async(Event::Start(self.start_tag.borrow()))
            .await?;
        self.writer.write_event_async(Event::CData(text)).await?;
        self.writer
            .write_event_async(Event::End(self.start_tag.to_end()))
            .await?;
        Ok(self.writer)
    }

    /// Writes a processing instruction `<?...?>` inside the current element.
    /// Async version of [`ElementWriter::write_pi_content`].
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn write_pi_content_async(self, pi: BytesPI<'_>) -> Result<&'a mut Writer<W>> {
        self.writer
            .write_event_async(Event::Start(self.start_tag.borrow()))
            .await?;
        self.writer.write_event_async(Event::PI(pi)).await?;
        self.writer
            .write_event_async(Event::End(self.start_tag.to_end()))
            .await?;
        Ok(self.writer)
    }

    /// Writes an empty (self-closing) tag. Async version of
    /// [`ElementWriter::write_empty`].
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn write_empty_async(self) -> Result<&'a mut Writer<W>> {
        self.writer
            .write_event_async(Event::Empty(self.start_tag))
            .await?;
        Ok(self.writer)
    }

    /// Creates a new scope for writing XML inside the current element. Async
    /// version of [`ElementWriter::write_inner_content`].
    ///
    /// # Example
    ///
    /// ```
    /// # use oxixml_xml::errors::Error;
    /// # use oxixml_xml::events::BytesText;
    /// # use oxixml_xml::writer::Writer;
    /// # use tokio::io::AsyncWriteExt;
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let mut buffer = Vec::new();
    /// let mut tokio_buffer = tokio::io::BufWriter::new(&mut buffer);
    /// let mut writer = Writer::new_with_indent(&mut tokio_buffer, b' ', 4);
    ///
    /// writer
    ///     .create_element("outer")
    ///     .with_attributes([("attr1", "value1"), ("attr2", "value2")])
    ///     // We need to provide error type, because it is not named somewhere explicitly
    ///     .write_inner_content_async::<_, _, Error>(|writer| async move {
    ///         let fruits = ["apple", "orange", "banana"];
    ///         for (quant, item) in fruits.iter().enumerate() {
    ///             writer
    ///                 .create_element("fruit")
    ///                 .with_attributes([("quantity", quant.to_string().as_str())])
    ///                 .write_text_content_async(BytesText::new(item))
    ///                 .await?;
    ///         }
    ///         writer
    ///             .create_element("inner")
    ///             .write_inner_content_async(|writer| async move {
    ///                 writer.create_element("empty").write_empty_async().await
    ///             })
    ///             .await?;
    ///
    ///         Ok(writer)
    ///     })
    ///     .await
    ///     .expect("cannot write content");
    ///
    /// tokio_buffer.flush().await.expect("flush failed");
    /// assert_eq!(
    ///     std::str::from_utf8(&buffer).expect("utf-8"),
    ///     r#"<outer attr1="value1" attr2="value2">
    ///     <fruit quantity="0">apple</fruit>
    ///     <fruit quantity="1">orange</fruit>
    ///     <fruit quantity="2">banana</fruit>
    ///     <inner>
    ///         <empty/>
    ///     </inner>
    /// </outer>"#
    /// );
    /// # }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "async-tokio")))]
    pub async fn write_inner_content_async<F, Fut, E>(
        mut self,
        closure: F,
    ) -> StdResult<&'a mut Writer<W>, E>
    where
        F: FnOnce(&'a mut Writer<W>) -> Fut,
        Fut: Future<Output = StdResult<&'a mut Writer<W>, E>>,
        E: From<Error>,
    {
        self.writer
            .write_event_async(Event::Start(self.start_tag.borrow()))
            .await?;
        self.writer = closure(self.writer).await?;
        self.writer
            .write_event_async(Event::End(self.start_tag.to_end()))
            .await?;
        Ok(self.writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::*;

    macro_rules! test {
        ($name: ident, $event: expr, $expected: expr) => {
            #[tokio::test]
            async fn $name() -> Result<()> {
                let mut buffer = Vec::new();
                let mut writer = Writer::new(&mut buffer);

                writer.write_event_async($event).await?;

                assert_eq!(std::str::from_utf8(&buffer).expect("utf-8"), $expected);
                Ok(())
            }
        };
    }

    test!(
        xml_header,
        Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), Some("no"))),
        r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>"#
    );

    test!(empty_tag, Event::Empty(BytesStart::new("tag")), r#"<tag/>"#);

    test!(
        comment,
        Event::Comment(BytesText::new("this is a comment")),
        r#"<!--this is a comment-->"#
    );

    test!(
        cdata,
        Event::CData(BytesCData::new("this is a cdata")),
        r#"<![CDATA[this is a cdata]]>"#
    );

    test!(
        pi,
        Event::PI(BytesPI::new("this is a processing instruction")),
        r#"<?this is a processing instruction?>"#
    );

    test!(
        doctype,
        Event::DocType(BytesText::new("this is a doctype")),
        r#"<!DOCTYPE this is a doctype>"#
    );

    #[tokio::test]
    async fn full_tag() -> Result<()> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new(&mut buffer);

        let start = Event::Start(BytesStart::new("tag"));
        let text = Event::Text(BytesText::new("inner text"));
        let end = Event::End(BytesEnd::new("tag"));
        for i in [start, text, end] {
            writer.write_event_async(i).await?;
        }

        assert_eq!(
            std::str::from_utf8(&buffer).expect("utf-8"),
            r#"<tag>inner text</tag>"#
        );
        Ok(())
    }
}

#[cfg(test)]
mod indentation_async {
    use super::*;
    use crate::events::*;

    #[tokio::test]
    async fn self_closed() -> Result<()> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

        let tag = BytesStart::new("self-closed")
            .with_attributes(vec![("attr1", "value1"), ("attr2", "value2")].into_iter());
        writer.write_event_async(Event::Empty(tag)).await?;

        assert_eq!(
            std::str::from_utf8(&buffer).expect("utf-8"),
            r#"<self-closed attr1="value1" attr2="value2"/>"#
        );
        Ok(())
    }

    #[tokio::test]
    async fn empty_paired() -> Result<()> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

        let start = BytesStart::new("paired")
            .with_attributes(vec![("attr1", "value1"), ("attr2", "value2")].into_iter());
        let end = start.to_end();
        writer
            .write_event_async(Event::Start(start.clone()))
            .await?;
        writer.write_event_async(Event::End(end)).await?;

        assert_eq!(
            std::str::from_utf8(&buffer).expect("utf-8"),
            "<paired attr1=\"value1\" attr2=\"value2\">\n</paired>"
        );
        Ok(())
    }

    #[tokio::test]
    async fn paired_with_inner() -> Result<()> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

        let start = BytesStart::new("paired")
            .with_attributes(vec![("attr1", "value1"), ("attr2", "value2")].into_iter());
        let end = start.to_end();
        let inner = BytesStart::new("inner");

        writer
            .write_event_async(Event::Start(start.clone()))
            .await?;
        writer.write_event_async(Event::Empty(inner)).await?;
        writer.write_event_async(Event::End(end)).await?;

        assert_eq!(
            std::str::from_utf8(&buffer).expect("utf-8"),
            "<paired attr1=\"value1\" attr2=\"value2\">\n    <inner/>\n</paired>"
        );
        Ok(())
    }

    #[tokio::test]
    async fn paired_with_text() -> Result<()> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

        let start = BytesStart::new("paired")
            .with_attributes(vec![("attr1", "value1"), ("attr2", "value2")].into_iter());
        let end = start.to_end();
        let text = BytesText::new("text");

        writer
            .write_event_async(Event::Start(start.clone()))
            .await?;
        writer.write_event_async(Event::Text(text)).await?;
        writer.write_event_async(Event::End(end)).await?;

        assert_eq!(
            std::str::from_utf8(&buffer).expect("utf-8"),
            r#"<paired attr1="value1" attr2="value2">text</paired>"#
        );
        Ok(())
    }

    #[tokio::test]
    async fn mixed_content() -> Result<()> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

        let start = BytesStart::new("paired")
            .with_attributes(vec![("attr1", "value1"), ("attr2", "value2")].into_iter());
        let end = start.to_end();
        let text = BytesText::new("text");
        let inner = BytesStart::new("inner");

        writer
            .write_event_async(Event::Start(start.clone()))
            .await?;
        writer.write_event_async(Event::Text(text)).await?;
        writer.write_event_async(Event::Empty(inner)).await?;
        writer.write_event_async(Event::End(end)).await?;

        assert_eq!(
            std::str::from_utf8(&buffer).expect("utf-8"),
            "<paired attr1=\"value1\" attr2=\"value2\">text<inner/>\n</paired>"
        );
        Ok(())
    }

    #[tokio::test]
    async fn nested() -> Result<()> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b' ', 4);

        let start = BytesStart::new("paired")
            .with_attributes(vec![("attr1", "value1"), ("attr2", "value2")].into_iter());
        let end = start.to_end();
        let inner = BytesStart::new("inner");

        writer
            .write_event_async(Event::Start(start.clone()))
            .await?;
        writer
            .write_event_async(Event::Start(start.clone()))
            .await?;
        writer.write_event_async(Event::Empty(inner)).await?;
        writer.write_event_async(Event::End(end.clone())).await?;
        writer.write_event_async(Event::End(end)).await?;

        assert_eq!(
            std::str::from_utf8(&buffer).expect("utf-8"),
            "<paired attr1=\"value1\" attr2=\"value2\">\n    \
             <paired attr1=\"value1\" attr2=\"value2\">\n        \
             <inner/>\n    </paired>\n</paired>"
        );
        Ok(())
    }
}
