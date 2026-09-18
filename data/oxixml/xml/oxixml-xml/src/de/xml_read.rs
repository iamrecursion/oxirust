//! The [`XmlRead`] abstraction over borrowing and copying XML input sources.

use std::io::BufRead;

use crate::XmlVersion;
use crate::de::event::PayloadEvent;
use crate::encoding::Decoder;
use crate::errors::serialize::DeError;
use crate::events::{BytesStart, Event};
use crate::name::QName;
use crate::reader::NsReader;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts raw reader's event into a payload event.
/// Returns `None`, if event should be skipped.
#[inline(always)]
fn skip_uninterested<'a>(event: Event<'a>) -> Option<PayloadEvent<'a>> {
    let event = match event {
        Event::DocType(e) => PayloadEvent::DocType(e),
        Event::Start(e) => PayloadEvent::Start(e),
        Event::End(e) => PayloadEvent::End(e),
        Event::Eof => PayloadEvent::Eof,

        // Do not trim next text event after Text, CDATA or reference event
        Event::CData(e) => PayloadEvent::CData(e),
        Event::Text(e) => PayloadEvent::Text(e),
        Event::GeneralRef(e) => PayloadEvent::GeneralRef(e),

        _ => return None,
    };
    Some(event)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Trait used by the deserializer for iterating over input. This is manually
/// "specialized" for iterating over `&[u8]`.
///
/// You do not need to implement this trait, it is needed to abstract from
/// [borrowing](SliceReader) and [copying](IoReader) data sources and reuse code in
/// deserializer
pub trait XmlRead<'i> {
    /// Return an input-borrowing event.
    fn next(&mut self) -> Result<PayloadEvent<'i>, DeError>;

    /// Skips until end element is found. Unlike `next()` it will not allocate
    /// when it cannot satisfy the lifetime.
    fn read_to_end(&mut self, name: QName) -> Result<(), DeError>;

    /// Return an XML version of the source.
    fn xml_version(&self) -> XmlVersion;

    /// A copy of the reader's decoder used to decode strings.
    fn decoder(&self) -> Decoder;

    /// Checks if the `start` tag has a [`xsi:nil`] attribute. This method ignores
    /// any errors in attributes.
    ///
    /// [`xsi:nil`]: https://www.w3.org/TR/xmlschema-1/#xsi_nil
    fn has_nil_attr(&self, start: &BytesStart) -> bool;
}

/// XML input source that reads from a std::io input stream.
///
/// You cannot create it, it is created automatically when you call
/// [`Deserializer::from_reader`](crate::de::Deserializer::from_reader)
pub struct IoReader<R: BufRead> {
    pub(crate) reader: NsReader<R>,
    pub(crate) buf: Vec<u8>,
    pub(crate) version: XmlVersion,
}

impl<R: BufRead> IoReader<R> {
    /// Returns the underlying XML reader.
    ///
    /// ```
    /// use serde::Deserialize;
    /// use std::io::Cursor;
    /// use oxixml_xml::de::Deserializer;
    /// use oxixml_xml::NsReader;
    ///
    /// #[derive(Deserialize)]
    /// struct SomeStruct {
    ///     field1: String,
    ///     field2: String,
    /// }
    ///
    /// // Try to deserialize from broken XML
    /// let mut de = Deserializer::from_reader(Cursor::new(
    ///     "<SomeStruct><field1><field2></SomeStruct>"
    /// //   0                           ^= 28        ^= 41
    /// ));
    ///
    /// let err = SomeStruct::deserialize(&mut de);
    /// assert!(err.is_err());
    ///
    /// let reader: &NsReader<Cursor<&str>> = de.get_ref().get_ref();
    ///
    /// assert_eq!(reader.error_position(), 28);
    /// assert_eq!(reader.buffer_position(), 41);
    /// ```
    pub const fn get_ref(&self) -> &NsReader<R> {
        &self.reader
    }
}

impl<'i, R: BufRead> XmlRead<'i> for IoReader<R> {
    fn next(&mut self) -> Result<PayloadEvent<'static>, DeError> {
        loop {
            self.buf.clear();

            let event = self.reader.read_event_into(&mut self.buf)?;
            if let Event::Decl(e) = &event {
                self.version = e.xml_version()?;
            }
            if let Some(event) = skip_uninterested(event) {
                return Ok(event.into_owned());
            }
        }
    }

    fn read_to_end(&mut self, name: QName) -> Result<(), DeError> {
        match self.reader.read_to_end_into(name, &mut self.buf) {
            Err(e) => Err(e.into()),
            Ok(_) => Ok(()),
        }
    }

    #[inline]
    fn xml_version(&self) -> XmlVersion {
        self.version
    }

    #[inline]
    fn decoder(&self) -> Decoder {
        self.reader.decoder()
    }

    fn has_nil_attr(&self, start: &BytesStart) -> bool {
        start.attributes().has_nil(self.reader.resolver())
    }
}

/// XML input source that reads from a slice of bytes and can borrow from it.
///
/// You cannot create it, it is created automatically when you call
/// [`Deserializer::from_str`](crate::de::Deserializer::from_str).
pub struct SliceReader<'de> {
    pub(crate) reader: NsReader<&'de [u8]>,
    pub(crate) version: XmlVersion,
}

impl<'de> SliceReader<'de> {
    /// Returns the underlying XML reader.
    ///
    /// ```
    /// use serde::Deserialize;
    /// use oxixml_xml::de::Deserializer;
    /// use oxixml_xml::NsReader;
    ///
    /// #[derive(Deserialize)]
    /// struct SomeStruct {
    ///     field1: String,
    ///     field2: String,
    /// }
    ///
    /// // Try to deserialize from broken XML
    /// let mut de = Deserializer::from_str(
    ///     "<SomeStruct><field1><field2></SomeStruct>"
    /// //   0                           ^= 28        ^= 41
    /// );
    ///
    /// let err = SomeStruct::deserialize(&mut de);
    /// assert!(err.is_err());
    ///
    /// let reader: &NsReader<&[u8]> = de.get_ref().get_ref();
    ///
    /// assert_eq!(reader.error_position(), 28);
    /// assert_eq!(reader.buffer_position(), 41);
    /// ```
    pub const fn get_ref(&self) -> &NsReader<&'de [u8]> {
        &self.reader
    }
}

impl<'de> XmlRead<'de> for SliceReader<'de> {
    fn next(&mut self) -> Result<PayloadEvent<'de>, DeError> {
        loop {
            let event = self.reader.read_event()?;
            if let Event::Decl(e) = &event {
                self.version = e.xml_version()?;
            }
            if let Some(event) = skip_uninterested(event) {
                return Ok(event);
            }
        }
    }

    fn read_to_end(&mut self, name: QName) -> Result<(), DeError> {
        match self.reader.read_to_end(name) {
            Err(e) => Err(e.into()),
            Ok(_) => Ok(()),
        }
    }

    #[inline]
    fn xml_version(&self) -> XmlVersion {
        self.version
    }

    #[inline]
    fn decoder(&self) -> Decoder {
        self.reader.decoder()
    }

    fn has_nil_attr(&self, start: &BytesStart) -> bool {
        start.attributes().has_nil(self.reader.resolver())
    }
}
