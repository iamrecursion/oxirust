//! Zero-copy XML events used throughout this library.
//!
//! An XML event often represents part of an XML element. For example, the element
//!
//! ```xml
//! <name attr="value">Inner text</name>
//! ```
//!
//! consists of three events: [`Event::Start`], [`Event::Text`] and [`Event::End`].
//! Events can also represent other parts of an XML document, like the XML
//! declaration. See [`Event`] for the list of all possible events.
//!
//! These types mirror quick-xml 0.41 `events`: the [`Event`] enum plus the
//! `Bytes*` payload structs. They occur both when reading and when writing.
//!
//! # Example
//!
//! ```
//! use oxixml_xml::events::{BytesStart, Event};
//! use oxixml_xml::name::QName;
//!
//! let start = BytesStart::new("name").with_attributes([("attr", "value")]);
//! assert_eq!(start.name(), QName(b"name"));
//! let event = Event::Start(start);
//! // Events borrow their payload bytes.
//! assert_eq!(&*event, b"name attr=\"value\"".as_ref());
//! ```

pub mod attributes;

use std::borrow::Cow;
use std::fmt::{self, Debug, Formatter};
use std::iter::FusedIterator;
use std::mem::replace;
use std::ops::Deref;
use std::str::from_utf8;

#[cfg(feature = "encoding")]
use oxixml_encoding::Encoding;

use crate::XmlVersion;
use crate::encoding::{Decoder, EncodingError};
use crate::errors::{Error, IllFormedError};
use crate::escape::{
    EscapeError, escape, minimal_escape, normalize_xml10_eols, normalize_xml11_eols, parse_number,
    partial_escape,
};
use crate::name::{LocalName, QName};
use crate::utils::{self, name_len, trim_xml_end, trim_xml_start, write_cow_string};
use attributes::{AttrError, Attribute, Attributes};

/// Opening tag data ([`Event::Start`]), with optional attributes: `<name attr="value">`.
///
/// The name can be accessed using the [`name`](Self::name) or
/// [`local_name`](Self::local_name) methods, and an iterator over the attributes
/// is returned by [`attributes`](Self::attributes).
///
/// This event dereferences to the tag content between `<` and `>` (or `/>`).
///
/// ```
/// # use oxixml_xml::events::BytesStart;
/// let content = "element a1 = 'val1' a2=\"val2\" ";
/// let event = BytesStart::from_content(content, 7);
/// assert_eq!(event.name().as_ref(), b"element");
/// // deref coercion of &BytesStart to &[u8]
/// assert_eq!(&event as &[u8], content.as_bytes());
/// assert_eq!(event.as_ref(), content.as_bytes());
/// ```
#[derive(Clone, Eq, PartialEq)]
pub struct BytesStart<'a> {
    /// Content of the element, before any UTF-8 conversion.
    pub(crate) buf: Cow<'a, [u8]>,
    /// End of the element name; the name starts at the start of `buf`.
    pub(crate) name_len: usize,
    /// Encoding used for `buf`.
    decoder: Decoder,
}

impl<'a> BytesStart<'a> {
    /// Internal constructor, used by the reader; supplies data in the reader's encoding.
    #[inline]
    #[allow(dead_code)] // used by the reader in a later pipeline stage
    pub(crate) const fn wrap(content: &'a [u8], name_len: usize, decoder: Decoder) -> Self {
        BytesStart {
            buf: Cow::Borrowed(content),
            name_len,
            decoder,
        }
    }

    /// Creates a new `BytesStart` from the given name.
    ///
    /// # Warning
    ///
    /// `name` must be a valid name.
    #[inline]
    pub fn new<C: Into<Cow<'a, str>>>(name: C) -> Self {
        let buf = str_cow_to_bytes(name);
        BytesStart {
            name_len: buf.len(),
            buf,
            decoder: Decoder::utf8(),
        }
    }

    /// Creates a new `BytesStart` from the given content (name + attributes).
    ///
    /// # Warning
    ///
    /// `&content[..name_len]` must be a valid name and the remainder of `content`
    /// must be correctly-formed attributes. Neither is checked.
    #[inline]
    pub fn from_content<C: Into<Cow<'a, str>>>(content: C, name_len: usize) -> Self {
        BytesStart {
            buf: str_cow_to_bytes(content),
            name_len,
            decoder: Decoder::utf8(),
        }
    }

    /// Converts the event into an owned event.
    pub fn into_owned(self) -> BytesStart<'static> {
        BytesStart {
            buf: Cow::Owned(self.buf.into_owned()),
            name_len: self.name_len,
            decoder: self.decoder,
        }
    }

    /// Converts the event into an owned event without taking ownership of it.
    pub fn to_owned(&self) -> BytesStart<'static> {
        BytesStart {
            buf: Cow::Owned(self.buf.clone().into_owned()),
            name_len: self.name_len,
            decoder: self.decoder,
        }
    }

    /// Converts the event into a borrowed event. Most useful when paired with
    /// [`to_end`](Self::to_end).
    pub fn borrow(&self) -> BytesStart<'_> {
        BytesStart {
            buf: Cow::Borrowed(&self.buf),
            name_len: self.name_len,
            decoder: self.decoder,
        }
    }

    /// Creates a new paired closing tag.
    #[inline]
    pub fn to_end(&self) -> BytesEnd<'_> {
        BytesEnd::from(self.name())
    }

    /// Gets the decoder used to decode the bytes of the reader that produced this
    /// event. For manually-constructed events the encoding is UTF-8.
    #[inline]
    pub const fn decoder(&self) -> Decoder {
        self.decoder
    }

    /// Gets the undecoded raw tag name, as present in the input stream.
    #[inline]
    pub fn name(&self) -> QName<'_> {
        QName(&self.buf[..self.name_len])
    }

    /// Gets the undecoded raw local tag name (excluding the namespace prefix), as
    /// present in the input stream.
    #[inline]
    pub fn local_name(&self) -> LocalName<'_> {
        self.name().into()
    }

    /// Edits the name of the `BytesStart` in place.
    ///
    /// # Warning
    ///
    /// `name` must be a valid name.
    pub fn set_name(&mut self, name: &[u8]) -> &mut BytesStart<'a> {
        let bytes = self.buf.to_mut();
        bytes.splice(..self.name_len, name.iter().cloned());
        self.name_len = name.len();
        self
    }
}

/// Attribute-related methods.
impl<'a> BytesStart<'a> {
    /// Consumes `self` and yields a new `BytesStart` with additional attributes
    /// from an iterator. Items must be convertible to [`Attribute`] via `Into`.
    pub fn with_attributes<'b, I>(mut self, attributes: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Attribute<'b>>,
    {
        self.extend_attributes(attributes);
        self
    }

    /// Adds additional attributes to this tag using an iterator. Items must be
    /// convertible to [`Attribute`] via `Into`.
    pub fn extend_attributes<'b, I>(&mut self, attributes: I) -> &mut BytesStart<'a>
    where
        I: IntoIterator,
        I::Item: Into<Attribute<'b>>,
    {
        for attr in attributes {
            self.push_attribute(attr);
        }
        self
    }

    /// Adds an attribute to this element.
    pub fn push_attribute<'b, A>(&mut self, attr: A)
    where
        A: Into<Attribute<'b>>,
    {
        self.buf.to_mut().push(b' ');
        self.push_attr(attr.into());
    }

    /// Removes all attributes from this `BytesStart`.
    pub fn clear_attributes(&mut self) -> &mut BytesStart<'a> {
        self.buf.to_mut().truncate(self.name_len);
        self
    }

    /// Returns an iterator over the attributes of this tag.
    pub fn attributes(&self) -> Attributes<'_> {
        Attributes::wrap(&self.buf, self.name_len, false, self.decoder)
    }

    /// Returns an iterator over the HTML-like attributes of this tag (no mandatory
    /// quotes or `=`).
    pub fn html_attributes(&self) -> Attributes<'_> {
        Attributes::wrap(&self.buf, self.name_len, true, self.decoder)
    }

    /// Gets the undecoded raw string with the attributes of this tag, including
    /// the white space after the tag name if there is any.
    #[inline]
    pub fn attributes_raw(&self) -> &[u8] {
        &self.buf[self.name_len..]
    }

    /// Tries to get an attribute by name.
    pub fn try_get_attribute<N: AsRef<[u8]> + Sized>(
        &'a self,
        attr_name: N,
    ) -> Result<Option<Attribute<'a>>, AttrError> {
        for a in self.attributes().with_checks(false) {
            let a = a?;
            if a.key.as_ref() == attr_name.as_ref() {
                return Ok(Some(a));
            }
        }
        Ok(None)
    }

    /// Adds an attribute to this element (without a leading space).
    pub(crate) fn push_attr<'b>(&mut self, attr: Attribute<'b>) {
        let bytes = self.buf.to_mut();
        bytes.extend_from_slice(attr.key.as_ref());
        bytes.extend_from_slice(b"=\"");
        bytes.extend_from_slice(attr.value.as_ref());
        bytes.push(b'"');
    }

    /// Adds a new line into the existing content of the element. Used by the
    /// [`Writer`](crate::writer::Writer) when laying out attributes across lines.
    pub(crate) fn push_newline(&mut self) {
        self.buf.to_mut().push(b'\n');
    }

    /// Adds indentation bytes into the existing content of the element. Used by
    /// the [`Writer`](crate::writer::Writer) when laying out attributes across lines.
    pub(crate) fn push_indent(&mut self, indent: &[u8]) {
        self.buf.to_mut().extend_from_slice(indent);
    }
}

impl<'a> Debug for BytesStart<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "BytesStart {{ buf: ")?;
        write_cow_string(f, &self.buf)?;
        write!(f, ", name_len: {} }}", self.name_len)
    }
}

impl<'a> Deref for BytesStart<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.buf
    }
}

/// Generates an element name (a non-empty alphanumeric string, so that it is
/// always a syntactically valid `Name`) plus a fuzzer-chosen set of
/// attributes, mirroring quick-xml 0.41's `Arbitrary` impl for `BytesStart`.
#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for BytesStart<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let s = <&str>::arbitrary(u)?;
        if s.is_empty() || !s.chars().all(char::is_alphanumeric) {
            return Err(arbitrary::Error::IncorrectFormat);
        }
        let mut result = Self::new(s);
        result.extend_attributes(Vec::<(&str, &str)>::arbitrary(u)?);
        Ok(result)
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&str as arbitrary::Arbitrary>::size_hint(depth)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Closing tag data ([`Event::End`]): `</name>`.
///
/// The name can be accessed using the [`name`](Self::name) or
/// [`local_name`](Self::local_name) methods. This event dereferences to the tag
/// content between `</` and `>`.
///
/// ```
/// # use oxixml_xml::events::BytesEnd;
/// let event = BytesEnd::new("element");
/// assert_eq!(event.name().as_ref(), b"element");
/// assert_eq!(&event as &[u8], b"element");
/// ```
#[derive(Clone, Eq, PartialEq)]
pub struct BytesEnd<'a> {
    name: Cow<'a, [u8]>,
}

impl<'a> BytesEnd<'a> {
    /// Internal constructor, used by the reader; supplies data in the reader's encoding.
    #[inline]
    pub(crate) const fn wrap(name: Cow<'a, [u8]>) -> Self {
        BytesEnd { name }
    }

    /// Creates a new `BytesEnd` borrowing a slice.
    ///
    /// # Warning
    ///
    /// `name` must be a valid name.
    #[inline]
    pub fn new<C: Into<Cow<'a, str>>>(name: C) -> Self {
        Self::wrap(str_cow_to_bytes(name))
    }

    /// Converts the event into an owned event.
    pub fn into_owned(self) -> BytesEnd<'static> {
        BytesEnd {
            name: Cow::Owned(self.name.into_owned()),
        }
    }

    /// Converts the event into a borrowed event.
    #[inline]
    pub fn borrow(&self) -> BytesEnd<'_> {
        BytesEnd {
            name: Cow::Borrowed(&self.name),
        }
    }

    /// Gets the undecoded raw tag name, as present in the input stream.
    #[inline]
    pub fn name(&self) -> QName<'_> {
        QName(&self.name)
    }

    /// Gets the undecoded raw local tag name (excluding the namespace prefix).
    #[inline]
    pub fn local_name(&self) -> LocalName<'_> {
        self.name().into()
    }
}

impl<'a> Debug for BytesEnd<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "BytesEnd {{ name: ")?;
        write_cow_string(f, &self.name)?;
        write!(f, " }}")
    }
}

impl<'a> Deref for BytesEnd<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.name
    }
}

impl<'a> From<QName<'a>> for BytesEnd<'a> {
    #[inline]
    fn from(name: QName<'a>) -> Self {
        Self::wrap(name.into_inner().into())
    }
}

/// Mirrors quick-xml 0.41's `Arbitrary` impl for `BytesEnd`: any fuzzer-chosen
/// string, without the non-empty/alphanumeric validation [`BytesStart`]'s impl
/// applies (an end tag's name is not independently validated by this type).
#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for BytesEnd<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self::new(<&str>::arbitrary(u)?))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&str as arbitrary::Arbitrary>::size_hint(depth)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Data from various events (most notably [`Event::Text`]).
///
/// This event dereferences to its content. For a comment this is everything
/// between `<!--` and `-->`; for a DTD everything between `<!DOCTYPE` (plus
/// spaces) and the closing `>`.
///
/// ```
/// # use oxixml_xml::events::BytesText;
/// let event = BytesText::new("comment or text ");
/// assert_eq!(&event as &[u8], b"comment or text ");
/// ```
#[derive(Clone, Eq, PartialEq)]
pub struct BytesText<'a> {
    /// Escaped, then encoded, content of the event.
    content: Cow<'a, [u8]>,
    /// Encoding in which `content` is stored.
    decoder: Decoder,
}

impl<'a> BytesText<'a> {
    /// Creates a new `BytesText` from a raw byte sequence in the specified encoding.
    #[inline]
    pub(crate) fn wrap<C: Into<Cow<'a, [u8]>>>(content: C, decoder: Decoder) -> Self {
        Self {
            content: content.into(),
            decoder,
        }
    }

    /// Creates a new `BytesText` from a raw string as it appeared in the XML source.
    ///
    /// # Warning
    ///
    /// `content` is not checked to not contain markup or entity references.
    #[inline]
    pub fn from_escaped<C: Into<Cow<'a, str>>>(content: C) -> Self {
        Self::wrap(str_cow_to_bytes(content), Decoder::utf8())
    }

    /// Creates a new `BytesText` from a string, escaping it using [`escape`].
    ///
    /// [`escape`]: crate::escape::escape
    #[inline]
    pub fn new(content: &'a str) -> Self {
        Self::from_escaped(escape(content))
    }

    /// Ensures that all data is owned to extend the object's lifetime if necessary.
    #[inline]
    pub fn into_owned(self) -> BytesText<'static> {
        BytesText {
            content: self.content.into_owned().into(),
            decoder: self.decoder,
        }
    }

    /// Extracts the inner `Cow` from the `BytesText` event container.
    #[inline]
    pub fn into_inner(self) -> Cow<'a, [u8]> {
        self.content
    }

    /// Converts the event into a borrowed event.
    #[inline]
    pub fn borrow(&self) -> BytesText<'_> {
        BytesText {
            content: Cow::Borrowed(&self.content),
            decoder: self.decoder,
        }
    }

    /// Decodes the content of the event.
    ///
    /// This does not normalize end-of-line characters; usually you need
    /// [`xml_content`](Self::xml_content) instead.
    pub fn decode(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.decode_cow(&self.content)
    }

    /// Decodes the content as XML 1.0 or HTML, normalizing end-of-line characters.
    pub fn xml10_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.content(&self.content, normalize_xml10_eols)
    }

    /// Decodes the content as XML 1.1, normalizing end-of-line characters.
    pub fn xml11_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.content(&self.content, normalize_xml11_eols)
    }

    /// Decodes the content according to the specified XML version.
    #[inline]
    pub fn xml_content(&self, version: XmlVersion) -> Result<Cow<'a, str>, EncodingError> {
        match version {
            XmlVersion::Explicit1_1 => self.xml11_content(),
            _ => self.xml10_content(),
        }
    }

    /// Alias for [`xml10_content`](Self::xml10_content).
    #[inline]
    pub fn html_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.xml10_content()
    }

    /// Removes leading XML white space bytes. Returns `true` if the content is
    /// empty afterwards.
    pub fn inplace_trim_start(&mut self) -> bool {
        self.content = trim_cow(
            replace(&mut self.content, Cow::Borrowed(b"")),
            trim_xml_start,
        );
        self.content.is_empty()
    }

    /// Removes trailing XML white space bytes. Returns `true` if the content is
    /// empty afterwards.
    pub fn inplace_trim_end(&mut self) -> bool {
        self.content = trim_cow(replace(&mut self.content, Cow::Borrowed(b"")), trim_xml_end);
        self.content.is_empty()
    }
}

impl<'a> Debug for BytesText<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "BytesText {{ content: ")?;
        write_cow_string(f, &self.content)?;
        write!(f, " }}")
    }
}

impl<'a> Deref for BytesText<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.content
    }
}

/// Generates alphanumeric-only text content, mirroring quick-xml 0.41's
/// `Arbitrary` impl for `BytesText`: restricting the charset sidesteps having
/// to reason about whether the fuzzer-chosen string round-trips through
/// [`escape`](Self::new)/unescape unchanged.
#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for BytesText<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let s = <&str>::arbitrary(u)?;
        if !s.chars().all(char::is_alphanumeric) {
            return Err(arbitrary::Error::IncorrectFormat);
        }
        Ok(Self::new(s))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&str as arbitrary::Arbitrary>::size_hint(depth)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// CDATA content ([`Event::CData`]) which contains unescaped data. To write it as
/// text, [convert](Self::escape) it to [`BytesText`].
///
/// This event dereferences to its content between `<![CDATA[` and `]]>`.
///
/// ```
/// # use oxixml_xml::events::BytesCData;
/// let event = BytesCData::new(" CDATA section ");
/// assert_eq!(&event as &[u8], b" CDATA section ");
/// ```
#[derive(Clone, Eq, PartialEq)]
pub struct BytesCData<'a> {
    content: Cow<'a, [u8]>,
    /// Encoding in which `content` is stored.
    decoder: Decoder,
}

impl<'a> BytesCData<'a> {
    /// Creates a new `BytesCData` from a byte sequence in the specified encoding.
    #[inline]
    pub(crate) fn wrap<C: Into<Cow<'a, [u8]>>>(content: C, decoder: Decoder) -> Self {
        Self {
            content: content.into(),
            decoder,
        }
    }

    /// Creates a new `BytesCData` from a string.
    ///
    /// # Warning
    ///
    /// `content` must not contain the `]]>` sequence. Use [`BytesCData::escaped`]
    /// to split content that might.
    #[inline]
    pub fn new<C: Into<Cow<'a, str>>>(content: C) -> Self {
        Self::wrap(str_cow_to_bytes(content), Decoder::utf8())
    }

    /// Creates an iterator of `BytesCData` from a string, splitting it into
    /// multiple `CDATA` sections wherever the `]]>` sequence appears (because it
    /// cannot be escaped).
    ///
    /// ```
    /// # use oxixml_xml::events::BytesCData;
    /// let content = "Certain tokens like ]]> can be difficult and <invalid>";
    /// let cdata = BytesCData::escaped(content).collect::<Vec<_>>();
    /// assert_eq!(cdata, &[
    ///     BytesCData::new("Certain tokens like ]]"),
    ///     BytesCData::new("> can be difficult and <invalid>"),
    /// ]);
    /// ```
    #[inline]
    pub const fn escaped(content: &'a str) -> CDataIterator<'a> {
        CDataIterator {
            inner: utils::CDataIterator::new(content),
        }
    }

    /// Ensures that all data is owned to extend the object's lifetime if necessary.
    #[inline]
    pub fn into_owned(self) -> BytesCData<'static> {
        BytesCData {
            content: self.content.into_owned().into(),
            decoder: self.decoder,
        }
    }

    /// Extracts the inner `Cow` from the `BytesCData` event container.
    #[inline]
    pub fn into_inner(self) -> Cow<'a, [u8]> {
        self.content
    }

    /// Converts the event into a borrowed event.
    #[inline]
    pub fn borrow(&self) -> BytesCData<'_> {
        BytesCData {
            content: Cow::Borrowed(&self.content),
            decoder: self.decoder,
        }
    }

    /// Converts this CDATA content into an escaped [`BytesText`] (escaping `<`,
    /// `>`, `&`, `'` and `"`).
    pub fn escape(self) -> Result<BytesText<'a>, EncodingError> {
        let decoded = self.decode()?;
        Ok(BytesText::wrap(
            match escape(decoded) {
                Cow::Borrowed(escaped) => Cow::Borrowed(escaped.as_bytes()),
                Cow::Owned(escaped) => Cow::Owned(escaped.into_bytes()),
            },
            Decoder::utf8(),
        ))
    }

    /// Converts this CDATA content into a partially-escaped [`BytesText`]
    /// (escaping `<`, `>` and `&`).
    pub fn partial_escape(self) -> Result<BytesText<'a>, EncodingError> {
        let decoded = self.decode()?;
        Ok(BytesText::wrap(
            match partial_escape(decoded) {
                Cow::Borrowed(escaped) => Cow::Borrowed(escaped.as_bytes()),
                Cow::Owned(escaped) => Cow::Owned(escaped.into_bytes()),
            },
            Decoder::utf8(),
        ))
    }

    /// Converts this CDATA content into a minimally-escaped [`BytesText`]
    /// (escaping only `<` and `&`).
    pub fn minimal_escape(self) -> Result<BytesText<'a>, EncodingError> {
        let decoded = self.decode()?;
        Ok(BytesText::wrap(
            match minimal_escape(decoded) {
                Cow::Borrowed(escaped) => Cow::Borrowed(escaped.as_bytes()),
                Cow::Owned(escaped) => Cow::Owned(escaped.into_bytes()),
            },
            Decoder::utf8(),
        ))
    }

    /// Decodes the raw byte content of the CDATA section, without XML entity
    /// escaping and without EOL normalization.
    pub fn decode(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.decode_cow(&self.content)
    }

    /// Decodes the CDATA content as XML 1.0 or HTML, normalizing EOLs.
    pub fn xml10_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.content(&self.content, normalize_xml10_eols)
    }

    /// Decodes the CDATA content as XML 1.1, normalizing EOLs.
    pub fn xml11_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.content(&self.content, normalize_xml11_eols)
    }

    /// Decodes the CDATA content according to the specified XML version.
    #[inline]
    pub fn xml_content(&self, version: XmlVersion) -> Result<Cow<'a, str>, EncodingError> {
        match version {
            XmlVersion::Explicit1_1 => self.xml11_content(),
            _ => self.xml10_content(),
        }
    }

    /// Alias for [`xml10_content`](Self::xml10_content).
    #[inline]
    pub fn html_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.xml10_content()
    }
}

impl<'a> Debug for BytesCData<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "BytesCData {{ content: ")?;
        write_cow_string(f, &self.content)?;
        write!(f, " }}")
    }
}

impl<'a> Deref for BytesCData<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.content
    }
}

/// Mirrors quick-xml 0.41's `Arbitrary` impl for `BytesCData`: any
/// fuzzer-chosen string, unvalidated (unlike `BytesCData::new`, this does not
/// require the absence of `]]>`).
#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for BytesCData<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self::new(<&str>::arbitrary(u)?))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&str as arbitrary::Arbitrary>::size_hint(depth)
    }
}

/// Iterator over `CDATA` sections in a string, created by [`BytesCData::escaped`].
#[derive(Debug, Clone)]
pub struct CDataIterator<'a> {
    inner: utils::CDataIterator<'a>,
}

impl<'a> Iterator for CDataIterator<'a> {
    type Item = BytesCData<'a>;

    fn next(&mut self) -> Option<BytesCData<'a>> {
        self.inner
            .next()
            .map(|slice| BytesCData::wrap(slice.as_bytes(), Decoder::utf8()))
    }
}

impl FusedIterator for CDataIterator<'_> {}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// [Processing instructions][pi] (PIs), allowing documents to carry instructions
/// for applications ([`Event::PI`]).
///
/// This event dereferences to its content between `<?` and `?>`.
///
/// ```
/// # use oxixml_xml::events::BytesPI;
/// let instruction = BytesPI::new(r#"xml-stylesheet href="style.css""#);
/// assert_eq!(instruction.target(), b"xml-stylesheet");
/// assert_eq!(instruction.content(), br#" href="style.css""#);
/// ```
///
/// [pi]: https://www.w3.org/TR/xml11/#sec-pi
#[derive(Clone, Eq, PartialEq)]
pub struct BytesPI<'a> {
    content: BytesStart<'a>,
}

impl<'a> BytesPI<'a> {
    /// Creates a new `BytesPI` from a byte sequence in the specified encoding.
    #[inline]
    #[allow(dead_code)] // used by the reader in a later pipeline stage
    pub(crate) const fn wrap(content: &'a [u8], target_len: usize, decoder: Decoder) -> Self {
        Self {
            content: BytesStart::wrap(content, target_len, decoder),
        }
    }

    /// Creates a new `BytesPI` from a string.
    ///
    /// # Warning
    ///
    /// `content` must not contain the `?>` sequence.
    #[inline]
    pub fn new<C: Into<Cow<'a, str>>>(content: C) -> Self {
        let buf = str_cow_to_bytes(content);
        let name_len = name_len(&buf);
        Self {
            content: BytesStart {
                buf,
                name_len,
                decoder: Decoder::utf8(),
            },
        }
    }

    /// Ensures that all data is owned to extend the object's lifetime if necessary.
    #[inline]
    pub fn into_owned(self) -> BytesPI<'static> {
        BytesPI {
            content: self.content.into_owned(),
        }
    }

    /// Extracts the inner `Cow` from the `BytesPI` event container.
    #[inline]
    pub fn into_inner(self) -> Cow<'a, [u8]> {
        self.content.buf
    }

    /// Converts the event into a borrowed event.
    #[inline]
    pub fn borrow(&self) -> BytesPI<'_> {
        BytesPI {
            content: self.content.borrow(),
        }
    }

    /// The target that identifies the application to which the instruction is
    /// directed.
    #[inline]
    pub fn target(&self) -> &[u8] {
        self.content.name().0
    }

    /// The content of the processing instruction: everything after the target
    /// name. The first character is therefore always a space.
    #[inline]
    pub fn content(&self) -> &[u8] {
        self.content.attributes_raw()
    }

    /// A view of the processing instruction's content as a list of key-value pairs.
    ///
    /// ```
    /// # use std::borrow::Cow;
    /// # use oxixml_xml::events::attributes::Attribute;
    /// # use oxixml_xml::events::BytesPI;
    /// # use oxixml_xml::name::QName;
    /// let instruction = BytesPI::new(r#"xml-stylesheet href="style.css""#);
    /// for attr in instruction.attributes() {
    ///     assert_eq!(attr, Ok(Attribute {
    ///         key: QName(b"href"),
    ///         value: Cow::Borrowed(b"style.css"),
    ///     }));
    /// }
    /// ```
    #[inline]
    pub fn attributes(&self) -> Attributes<'_> {
        self.content.attributes()
    }
}

impl<'a> Debug for BytesPI<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "BytesPI {{ content: ")?;
        write_cow_string(f, &self.content.buf)?;
        write!(f, " }}")
    }
}

impl<'a> Deref for BytesPI<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.content
    }
}

/// Mirrors quick-xml 0.41's `Arbitrary` impl for `BytesPI`: any fuzzer-chosen
/// string, unvalidated (unlike `BytesPI::new`, this does not require the
/// absence of `?>`).
#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for BytesPI<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self::new(<&str>::arbitrary(u)?))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&str as arbitrary::Arbitrary>::size_hint(depth)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// An XML declaration ([`Event::Decl`]): `<?xml version="1.0" ...?>`.
///
/// This event dereferences to its content between `<?` and `?>`.
///
/// ```
/// # use oxixml_xml::events::{BytesDecl, BytesStart};
/// let content = "xml version = '1.0' ";
/// let event = BytesDecl::from_start(BytesStart::from_content(content, 3));
/// assert_eq!(&event as &[u8], content.as_bytes());
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BytesDecl<'a> {
    content: BytesStart<'a>,
}

impl<'a> BytesDecl<'a> {
    /// Constructs a new declaration from the mandatory `version` (should be `1.0`
    /// or `1.1`), the optional `encoding` and the optional `standalone` attribute.
    ///
    /// Inputs are not escaped and attribute values are always wrapped in double
    /// quotes.
    pub fn new(
        version: &str,
        encoding: Option<&str>,
        standalone: Option<&str>,
    ) -> BytesDecl<'static> {
        // ' encoding=""'   => 12
        let encoding_attr_len = if let Some(xs) = encoding {
            12 + xs.len()
        } else {
            0
        };
        // ' standalone=""' => 14
        let standalone_attr_len = if let Some(xs) = standalone {
            14 + xs.len()
        } else {
            0
        };
        // 'xml version=""' => 14
        let mut buf = String::with_capacity(14 + encoding_attr_len + standalone_attr_len);

        buf.push_str("xml version=\"");
        buf.push_str(version);

        if let Some(encoding_val) = encoding {
            buf.push_str("\" encoding=\"");
            buf.push_str(encoding_val);
        }

        if let Some(standalone_val) = standalone {
            buf.push_str("\" standalone=\"");
            buf.push_str(standalone_val);
        }
        buf.push('"');

        BytesDecl {
            content: BytesStart::from_content(buf, 3),
        }
    }

    /// Creates a `BytesDecl` from a [`BytesStart`].
    pub const fn from_start(start: BytesStart<'a>) -> Self {
        Self { content: start }
    }

    /// Gets the XML version, excluding quotes.
    ///
    /// The version *must* be the first attribute in the declaration. If it is
    /// missing or not first, [`IllFormedError::MissingDeclVersion`] is returned.
    ///
    /// ```
    /// # use oxixml_xml::errors::{Error, IllFormedError};
    /// # use oxixml_xml::events::{BytesDecl, BytesStart};
    /// let decl = BytesDecl::from_start(BytesStart::from_content(" version='1.1'", 0));
    /// assert_eq!(decl.version().expect("doc example"), b"1.1".as_ref());
    ///
    /// let decl = BytesDecl::from_start(BytesStart::from_content(" encoding='utf-8'", 0));
    /// match decl.version() {
    ///     Err(Error::IllFormed(IllFormedError::MissingDeclVersion(Some(key)))) => assert_eq!(key, "encoding"),
    ///     _ => panic!("expected MissingDeclVersion"),
    /// }
    /// ```
    pub fn version(&self) -> Result<Cow<'_, [u8]>, Error> {
        match self.content.attributes().with_checks(false).next() {
            Some(Ok(a)) if a.key.as_ref() == b"version" => Ok(a.value),
            Some(Ok(a)) => {
                let found = from_utf8(a.key.as_ref())
                    .map_err(|_| IllFormedError::MissingDeclVersion(None))?
                    .to_string();
                Err(Error::IllFormed(IllFormedError::MissingDeclVersion(Some(
                    found,
                ))))
            }
            Some(Err(e)) => Err(e.into()),
            None => Err(Error::IllFormed(IllFormedError::MissingDeclVersion(None))),
        }
    }

    /// Gets the XML encoding, excluding quotes. Returns the first occurrence even
    /// if there are several; the name is not restricted to the grammar.
    pub fn encoding(&self) -> Option<Result<Cow<'_, [u8]>, AttrError>> {
        self.content
            .try_get_attribute("encoding")
            .map(|a| a.map(|a| a.value))
            .transpose()
    }

    /// Gets the XML `standalone` flag, excluding quotes. Returns the first
    /// occurrence even if there are several.
    pub fn standalone(&self) -> Option<Result<Cow<'_, [u8]>, AttrError>> {
        self.content
            .try_get_attribute("standalone")
            .map(|a| a.map(|a| a.value))
            .transpose()
    }

    /// Gets the XML version as a typed enumeration.
    ///
    /// Returns [`IllFormedError::MissingDeclVersion`] if the version is missing or
    /// not first, and [`IllFormedError::UnknownVersion`] if it is neither 1.0 nor 1.1.
    pub fn xml_version(&self) -> Result<XmlVersion, Error> {
        let v = self.version()?;
        match v.as_ref() {
            b"1.0" => Ok(XmlVersion::Explicit1_0),
            b"1.1" => Ok(XmlVersion::Explicit1_1),
            _ => Err(Error::IllFormed(IllFormedError::UnknownVersion)),
        }
    }

    /// Resolves the declared encoding name to an [`Encoding`] using the
    /// [_get an encoding_](https://encoding.spec.whatwg.org/#concept-encoding-get)
    /// algorithm. Returns `None` if the encoding is unknown or absent.
    ///
    /// This is a *context-free* label lookup: a bare `UTF-16` label carries no
    /// endianness of its own (XML 1.0 Appendix F, RFC 2781), so per the WHATWG
    /// algorithm this resolves it to [`Encoding::Utf16Le`] as an arbitrary
    /// default. Do not treat that default as authoritative over a byte-order
    /// mark or the XML 1.0 Appendix F.1 four-byte pattern already observed for
    /// the input — [`Reader`](crate::reader::Reader) does not: it keeps
    /// whichever UTF-16 byte order the bytes themselves established, using
    /// this method only to confirm the encoding *family*, not its byte order.
    #[cfg(feature = "encoding")]
    #[cfg_attr(docsrs, doc(cfg(feature = "encoding")))]
    pub fn encoder(&self) -> Option<Encoding> {
        self.encoding()
            .and_then(|e| e.ok())
            .and_then(|e| std::str::from_utf8(&e).ok().and_then(Encoding::for_label))
    }

    /// Converts the event into an owned event.
    pub fn into_owned(self) -> BytesDecl<'static> {
        BytesDecl {
            content: self.content.into_owned(),
        }
    }

    /// Converts the event into a borrowed event.
    #[inline]
    pub fn borrow(&self) -> BytesDecl<'_> {
        BytesDecl {
            content: self.content.borrow(),
        }
    }
}

impl<'a> Deref for BytesDecl<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.content
    }
}

/// Mirrors quick-xml 0.41's `Arbitrary` impl for `BytesDecl`: fuzzer-chosen
/// `version`, `encoding` and `standalone` values assembled through
/// [`BytesDecl::new`], unvalidated.
#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for BytesDecl<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self::new(
            <&str>::arbitrary(u)?,
            Option::<&str>::arbitrary(u)?,
            Option::<&str>::arbitrary(u)?,
        ))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&str as arbitrary::Arbitrary>::size_hint(depth)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A character or general entity reference ([`Event::GeneralRef`]): `&ref;` or
/// `&#<number>;`.
///
/// This event dereferences to its content between `&` and `;`.
///
/// ```
/// # use oxixml_xml::events::BytesRef;
/// let event = BytesRef::new("entity");
/// assert_eq!(&event as &[u8], b"entity");
/// ```
#[derive(Clone, Eq, PartialEq)]
pub struct BytesRef<'a> {
    content: Cow<'a, [u8]>,
    /// Encoding in which `content` is stored.
    decoder: Decoder,
}

impl<'a> BytesRef<'a> {
    /// Internal constructor, used by the reader; supplies data in the reader's encoding.
    #[inline]
    #[allow(dead_code)] // used by the reader in a later pipeline stage
    pub(crate) const fn wrap(content: &'a [u8], decoder: Decoder) -> Self {
        Self {
            content: Cow::Borrowed(content),
            decoder,
        }
    }

    /// Creates a new `BytesRef` borrowing a slice.
    ///
    /// # Warning
    ///
    /// `name` must be a valid reference body.
    #[inline]
    pub fn new<C: Into<Cow<'a, str>>>(name: C) -> Self {
        Self {
            content: str_cow_to_bytes(name),
            decoder: Decoder::utf8(),
        }
    }

    /// Converts the event into an owned event.
    pub fn into_owned(self) -> BytesRef<'static> {
        BytesRef {
            content: Cow::Owned(self.content.into_owned()),
            decoder: self.decoder,
        }
    }

    /// Extracts the inner `Cow` from the `BytesRef` event container.
    #[inline]
    pub fn into_inner(self) -> Cow<'a, [u8]> {
        self.content
    }

    /// Converts the event into a borrowed event.
    #[inline]
    pub fn borrow(&self) -> BytesRef<'_> {
        BytesRef {
            content: Cow::Borrowed(&self.content),
            decoder: self.decoder,
        }
    }

    /// Decodes the content of the event without EOL normalization.
    pub fn decode(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.decode_cow(&self.content)
    }

    /// Decodes the content as XML 1.0 or HTML, normalizing EOLs.
    pub fn xml10_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.content(&self.content, normalize_xml10_eols)
    }

    /// Decodes the content as XML 1.1, normalizing EOLs.
    pub fn xml11_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.decoder.content(&self.content, normalize_xml11_eols)
    }

    /// Decodes the content according to the specified XML version.
    #[inline]
    pub fn xml_content(&self, version: XmlVersion) -> Result<Cow<'a, str>, EncodingError> {
        match version {
            XmlVersion::Explicit1_1 => self.xml11_content(),
            _ => self.xml10_content(),
        }
    }

    /// Alias for [`xml10_content`](Self::xml10_content).
    #[inline]
    pub fn html_content(&self) -> Result<Cow<'a, str>, EncodingError> {
        self.xml10_content()
    }

    /// Returns `true` if this reference is a character reference (`&#<number>;`).
    ///
    /// ```
    /// # use oxixml_xml::events::BytesRef;
    /// assert_eq!(BytesRef::new("#x30").is_char_ref(), true);
    /// assert_eq!(BytesRef::new("#49" ).is_char_ref(), true);
    /// assert_eq!(BytesRef::new("lt"  ).is_char_ref(), false);
    /// ```
    pub fn is_char_ref(&self) -> bool {
        matches!(self.content.first(), Some(b'#'))
    }

    /// If this reference is a character reference, resolves and returns the
    /// character; otherwise returns `None`.
    ///
    /// ```
    /// # use oxixml_xml::events::BytesRef;
    /// assert_eq!(BytesRef::new("#x30").resolve_char_ref().expect("doc example"), Some('0'));
    /// assert_eq!(BytesRef::new("#49" ).resolve_char_ref().expect("doc example"), Some('1'));
    /// assert_eq!(BytesRef::new("lt"  ).resolve_char_ref().expect("doc example"), None);
    /// ```
    pub fn resolve_char_ref(&self) -> Result<Option<char>, Error> {
        if let Some(num) = self.decode()?.strip_prefix('#') {
            let ch = parse_number(num).map_err(EscapeError::InvalidCharRef)?;
            return Ok(Some(ch));
        }
        Ok(None)
    }
}

impl<'a> Debug for BytesRef<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "BytesRef {{ content: ")?;
        write_cow_string(f, &self.content)?;
        write!(f, " }}")
    }
}

impl<'a> Deref for BytesRef<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.content
    }
}

/// Mirrors quick-xml 0.41's `Arbitrary` impl for `BytesRef`: any fuzzer-chosen
/// string, unvalidated (unlike `BytesRef::new`, this does not require a valid
/// reference body).
#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for BytesRef<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self::new(<&str>::arbitrary(u)?))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        <&str as arbitrary::Arbitrary>::size_hint(depth)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// An event produced while reading (or to be written).
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum Event<'a> {
    /// Start tag (with attributes): `<tag attr="value">`.
    Start(BytesStart<'a>),
    /// End tag: `</tag>`.
    End(BytesEnd<'a>),
    /// Empty element tag (with attributes): `<tag attr="value" />`.
    Empty(BytesStart<'a>),
    /// Escaped character data between tags.
    Text(BytesText<'a>),
    /// Unescaped character data stored in `<![CDATA[...]]>`.
    CData(BytesCData<'a>),
    /// Comment: `<!-- ... -->`.
    Comment(BytesText<'a>),
    /// XML declaration: `<?xml ...?>`.
    Decl(BytesDecl<'a>),
    /// Processing instruction: `<?...?>`.
    PI(BytesPI<'a>),
    /// Document type definition data (DTD) stored in `<!DOCTYPE ...>`.
    DocType(BytesText<'a>),
    /// A general reference `&entity;` in the textual data. Can be either an entity
    /// reference or a character reference.
    GeneralRef(BytesRef<'a>),
    /// End of the XML document.
    Eof,
}

impl<'a> Event<'a> {
    /// Converts the event to an owned version, untied to the lifetime of the
    /// buffer used when reading but incurring a new, separate allocation.
    pub fn into_owned(self) -> Event<'static> {
        match self {
            Event::Start(e) => Event::Start(e.into_owned()),
            Event::End(e) => Event::End(e.into_owned()),
            Event::Empty(e) => Event::Empty(e.into_owned()),
            Event::Text(e) => Event::Text(e.into_owned()),
            Event::Comment(e) => Event::Comment(e.into_owned()),
            Event::CData(e) => Event::CData(e.into_owned()),
            Event::Decl(e) => Event::Decl(e.into_owned()),
            Event::PI(e) => Event::PI(e.into_owned()),
            Event::DocType(e) => Event::DocType(e.into_owned()),
            Event::GeneralRef(e) => Event::GeneralRef(e.into_owned()),
            Event::Eof => Event::Eof,
        }
    }

    /// Converts the event into a borrowed event.
    #[inline]
    pub fn borrow(&self) -> Event<'_> {
        match self {
            Event::Start(e) => Event::Start(e.borrow()),
            Event::End(e) => Event::End(e.borrow()),
            Event::Empty(e) => Event::Empty(e.borrow()),
            Event::Text(e) => Event::Text(e.borrow()),
            Event::Comment(e) => Event::Comment(e.borrow()),
            Event::CData(e) => Event::CData(e.borrow()),
            Event::Decl(e) => Event::Decl(e.borrow()),
            Event::PI(e) => Event::PI(e.borrow()),
            Event::DocType(e) => Event::DocType(e.borrow()),
            Event::GeneralRef(e) => Event::GeneralRef(e.borrow()),
            Event::Eof => Event::Eof,
        }
    }
}

impl<'a> Deref for Event<'a> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match *self {
            Event::Start(ref e) | Event::Empty(ref e) => e,
            Event::End(ref e) => e,
            Event::Text(ref e) => e,
            Event::Decl(ref e) => e,
            Event::PI(ref e) => e,
            Event::CData(ref e) => e,
            Event::Comment(ref e) => e,
            Event::DocType(ref e) => e,
            Event::GeneralRef(ref e) => e,
            Event::Eof => &[],
        }
    }
}

impl<'a> AsRef<Event<'a>> for Event<'a> {
    fn as_ref(&self) -> &Event<'a> {
        self
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[inline]
fn str_cow_to_bytes<'a, C: Into<Cow<'a, str>>>(content: C) -> Cow<'a, [u8]> {
    match content.into() {
        Cow::Borrowed(s) => Cow::Borrowed(s.as_bytes()),
        Cow::Owned(s) => Cow::Owned(s.into_bytes()),
    }
}

fn trim_cow<'a, F>(value: Cow<'a, [u8]>, trim: F) -> Cow<'a, [u8]>
where
    F: FnOnce(&[u8]) -> &[u8],
{
    match value {
        Cow::Borrowed(bytes) => Cow::Borrowed(trim(bytes)),
        Cow::Owned(mut bytes) => {
            let trimmed = trim(&bytes);
            if trimmed.len() != bytes.len() {
                bytes = trimmed.to_vec();
            }
            Cow::Owned(bytes)
        }
    }
}

#[cfg(test)]
mod tests;
