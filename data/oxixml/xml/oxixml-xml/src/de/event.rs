//! Simplified events consumed by the deserializer and the intermediate reader
//! that produces them.

use std::borrow::Cow;
use std::mem::replace;
use std::ops::{Deref, Range};

use crate::de::resolver::{EntityResolver, PredefinedEntityResolver};
use crate::de::xml_read::XmlRead;
use crate::encoding::Decoder;
use crate::errors::serialize::DeError;
use crate::escape::{EscapeError, parse_number};
use crate::events::{BytesCData, BytesEnd, BytesRef, BytesStart, BytesText};
use crate::name::QName;

/// A function to check whether the character is a whitespace (blank, new line, carriage return or tab).
#[inline]
const fn is_non_whitespace(ch: char) -> bool {
    !matches!(ch, ' ' | '\r' | '\n' | '\t')
}

/// Decoded and concatenated content of consequent [`Text`] and [`CData`]
/// events. _Consequent_ means that events should follow each other or be
/// delimited only by (any count of) [`Comment`] or [`PI`] events.
///
/// Internally text is stored in `Cow<str>`. Cloning of text is cheap while it
/// is borrowed and makes copies of data when it is owned.
///
/// [`Text`]: crate::events::Event::Text
/// [`CData`]: crate::events::Event::CData
/// [`Comment`]: crate::events::Event::Comment
/// [`PI`]: crate::events::Event::PI
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Text<'a> {
    /// Untrimmed text after concatenating content of all
    /// [`Text`] and [`CData`] events
    ///
    /// [`Text`]: crate::events::Event::Text
    /// [`CData`]: crate::events::Event::CData
    pub(crate) text: Cow<'a, str>,
    /// A range into `text` which contains data after trimming
    content: Range<usize>,
}

impl<'a> Text<'a> {
    pub(crate) fn new(text: Cow<'a, str>) -> Self {
        let start = text.find(is_non_whitespace).unwrap_or(0);
        let end = text.rfind(is_non_whitespace).map_or(0, |i| i + 1);

        let content = if start >= end { 0..0 } else { start..end };

        Self { text, content }
    }

    /// Returns text without leading and trailing whitespaces as [defined] by XML specification.
    ///
    /// If you want to only check if text contains only whitespaces, use [`is_blank`](Self::is_blank),
    /// which will not allocate.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxixml_xml::de::Text;
    /// #
    /// let text = Text::from("");
    /// assert_eq!(text.trimmed(), "");
    ///
    /// let text = Text::from(" \r\n\t ");
    /// assert_eq!(text.trimmed(), "");
    ///
    /// let text = Text::from("  some useful text  ");
    /// assert_eq!(text.trimmed(), "some useful text");
    /// ```
    ///
    /// [defined]: https://www.w3.org/TR/xml11/#NT-S
    pub fn trimmed(&self) -> Cow<'a, str> {
        match self.text {
            Cow::Borrowed(text) => Cow::Borrowed(&text[self.content.clone()]),
            Cow::Owned(ref text) => Cow::Owned(text[self.content.clone()].to_string()),
        }
    }

    /// Returns `true` if text is empty or contains only whitespaces as [defined] by XML specification.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxixml_xml::de::Text;
    /// #
    /// let text = Text::from("");
    /// assert_eq!(text.is_blank(), true);
    ///
    /// let text = Text::from(" \r\n\t ");
    /// assert_eq!(text.is_blank(), true);
    ///
    /// let text = Text::from("  some useful text  ");
    /// assert_eq!(text.is_blank(), false);
    /// ```
    ///
    /// [defined]: https://www.w3.org/TR/xml11/#NT-S
    pub fn is_blank(&self) -> bool {
        self.content.is_empty()
    }
}

impl<'a> Deref for Text<'a> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.text.deref()
    }
}

impl<'a> From<&'a str> for Text<'a> {
    #[inline]
    fn from(text: &'a str) -> Self {
        Self::new(Cow::Borrowed(text))
    }
}

impl<'a> From<String> for Text<'a> {
    #[inline]
    fn from(text: String) -> Self {
        Self::new(Cow::Owned(text))
    }
}

impl<'a> From<Cow<'a, str>> for Text<'a> {
    #[inline]
    fn from(text: Cow<'a, str>) -> Self {
        Self::new(text)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Simplified event which contains only these variants that used by deserializer
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeEvent<'a> {
    /// Start tag (with attributes) `<tag attr="value">`.
    Start(BytesStart<'a>),
    /// End tag `</tag>`.
    End(BytesEnd<'a>),
    /// Decoded and concatenated content of consequent [`Text`] and [`CData`]
    /// events. _Consequent_ means that events should follow each other or be
    /// delimited only by (any count of) [`Comment`] or [`PI`] events.
    ///
    /// [`Text`]: crate::events::Event::Text
    /// [`CData`]: crate::events::Event::CData
    /// [`Comment`]: crate::events::Event::Comment
    /// [`PI`]: crate::events::Event::PI
    Text(Text<'a>),
    /// End of XML document.
    Eof,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Simplified event which contains only these variants that used by deserializer,
/// but [`Text`] events not yet fully processed.
///
/// [`Text`] events should be trimmed if they does not surrounded by the other
/// [`Text`] or [`CData`] events. This event contains intermediate state of [`Text`]
/// event, where they are trimmed from the start, but not from the end. To trim
/// end spaces we should lookahead by one deserializer event (i. e. skip all
/// comments and processing instructions).
///
/// [`Text`]: crate::events::Event::Text
/// [`CData`]: crate::events::Event::CData
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PayloadEvent<'a> {
    /// Start tag (with attributes) `<tag attr="value">`.
    Start(BytesStart<'a>),
    /// End tag `</tag>`.
    End(BytesEnd<'a>),
    /// Escaped character data between tags.
    Text(BytesText<'a>),
    /// Unescaped character data stored in `<![CDATA[...]]>`.
    CData(BytesCData<'a>),
    /// Document type definition data (DTD) stored in `<!DOCTYPE ...>`.
    DocType(BytesText<'a>),
    /// Reference `&ref;` in the textual data.
    GeneralRef(BytesRef<'a>),
    /// End of XML document.
    Eof,
}

impl<'a> PayloadEvent<'a> {
    /// Ensures that all data is owned to extend the object's lifetime if necessary.
    #[inline]
    pub(crate) fn into_owned(self) -> PayloadEvent<'static> {
        match self {
            PayloadEvent::Start(e) => PayloadEvent::Start(e.into_owned()),
            PayloadEvent::End(e) => PayloadEvent::End(e.into_owned()),
            PayloadEvent::Text(e) => PayloadEvent::Text(e.into_owned()),
            PayloadEvent::CData(e) => PayloadEvent::CData(e.into_owned()),
            PayloadEvent::DocType(e) => PayloadEvent::DocType(e.into_owned()),
            PayloadEvent::GeneralRef(e) => PayloadEvent::GeneralRef(e.into_owned()),
            PayloadEvent::Eof => PayloadEvent::Eof,
        }
    }
}

/// An intermediate reader that consumes [`PayloadEvent`]s and produces final [`DeEvent`]s.
/// [`PayloadEvent::Text`] events, that followed by any event except
/// [`PayloadEvent::Text`] or [`PayloadEvent::CData`], are trimmed from the end.
pub(crate) struct XmlReader<'i, R: XmlRead<'i>, E: EntityResolver = PredefinedEntityResolver> {
    /// A source of low-level XML events
    pub(crate) reader: R,
    /// Intermediate event, that could be returned by the next call to `next()`.
    /// If that is the `Text` event then leading spaces already trimmed, but
    /// trailing spaces is not. Before the event will be returned, trimming of
    /// the spaces could be necessary
    lookahead: Result<PayloadEvent<'i>, DeError>,

    /// Used to resolve unknown entities that would otherwise cause the parser
    /// to return an [`EscapeError::UnrecognizedEntity`] error.
    ///
    /// [`EscapeError::UnrecognizedEntity`]: crate::escape::EscapeError::UnrecognizedEntity
    entity_resolver: E,
}

impl<'i, R: XmlRead<'i>, E: EntityResolver> XmlReader<'i, R, E> {
    pub(crate) fn new(mut reader: R, entity_resolver: E) -> Self {
        // Lookahead by one event immediately, so we do not need to check in the
        // loop if we need lookahead or not
        let lookahead = reader.next();

        Self {
            reader,
            lookahead,
            entity_resolver,
        }
    }

    /// Returns `true` if all events was consumed
    pub(crate) const fn is_empty(&self) -> bool {
        matches!(self.lookahead, Ok(PayloadEvent::Eof))
    }

    /// Read next event and put it in lookahead, return the current lookahead
    #[inline(always)]
    fn next_impl(&mut self) -> Result<PayloadEvent<'i>, DeError> {
        replace(&mut self.lookahead, self.reader.next())
    }

    /// Returns `true` when next event is not a text event in any form.
    #[inline(always)]
    const fn current_event_is_last_text(&self) -> bool {
        // If next event is a text-like event or a DocType (which is
        // metadata and invisible to the data model), we should not
        // trim trailing spaces — there is more content to drain, and
        // any DocType between us and the next text run needs to be
        // absorbed so `read_text` does not later see two consecutive
        // `DeEvent::Text`. Without DocType here, an input like
        // `<a>x<!DOCTYPE y>z</a>` produces two text events for `x`
        // and `z`, tripping `unreachable!()` in `read_text`.
        !matches!(
            self.lookahead,
            Ok(PayloadEvent::Text(_)
                | PayloadEvent::CData(_)
                | PayloadEvent::GeneralRef(_)
                | PayloadEvent::DocType(_))
        )
    }

    /// Read all consequent [`Text`] and [`CData`] events until non-text event
    /// occurs. Content of all events would be appended to `result` and returned
    /// as [`DeEvent::Text`].
    ///
    /// DocType events that fall between text events are absorbed by the
    /// entity resolver and do not break the run — see
    /// [`Self::current_event_is_last_text`] for the rationale.
    ///
    /// [`Text`]: PayloadEvent::Text
    /// [`CData`]: PayloadEvent::CData
    fn drain_text(&mut self, mut result: Cow<'i, str>) -> Result<DeEvent<'i>, DeError> {
        loop {
            if self.current_event_is_last_text() {
                break;
            }

            match self.next_impl()? {
                PayloadEvent::Text(e) => result
                    .to_mut()
                    .push_str(&e.xml_content(self.reader.xml_version())?),
                PayloadEvent::CData(e) => result
                    .to_mut()
                    .push_str(&e.xml_content(self.reader.xml_version())?),
                PayloadEvent::GeneralRef(e) => self.resolve_reference(result.to_mut(), e)?,
                PayloadEvent::DocType(e) => {
                    self.entity_resolver
                        .capture(e)
                        .map_err(|err| DeError::Custom(format!("cannot parse DTD: {}", err)))?;
                }

                // SAFETY: current_event_is_last_text checks that event is Text, CData, GeneralRef, or DocType
                _ => unreachable!(
                    "Only `Text`, `CData`, `GeneralRef` or `DocType` events can come here"
                ),
            }
        }
        Ok(DeEvent::Text(Text::new(result)))
    }

    /// Return an input-borrowing event.
    pub(crate) fn next(&mut self) -> Result<DeEvent<'i>, DeError> {
        loop {
            return match self.next_impl()? {
                PayloadEvent::Start(e) => Ok(DeEvent::Start(e)),
                PayloadEvent::End(e) => Ok(DeEvent::End(e)),
                PayloadEvent::Text(e) => self.drain_text(e.xml_content(self.reader.xml_version())?),
                PayloadEvent::CData(e) => {
                    self.drain_text(e.xml_content(self.reader.xml_version())?)
                }
                PayloadEvent::DocType(e) => {
                    self.entity_resolver
                        .capture(e)
                        .map_err(|err| DeError::Custom(format!("cannot parse DTD: {}", err)))?;
                    continue;
                }
                PayloadEvent::GeneralRef(e) => {
                    let mut text = String::new();
                    self.resolve_reference(&mut text, e)?;
                    self.drain_text(text.into())
                }
                PayloadEvent::Eof => Ok(DeEvent::Eof),
            };
        }
    }

    fn resolve_reference(&mut self, result: &mut String, event: BytesRef) -> Result<(), DeError> {
        let len = event.len();
        let reference = self.decoder().decode(&event)?;

        if let Some(num) = reference.strip_prefix('#') {
            let codepoint = parse_number(num).map_err(EscapeError::InvalidCharRef)?;
            result.push_str(codepoint.encode_utf8(&mut [0u8; 4]));
            return Ok(());
        }
        if let Some(value) = self.entity_resolver.resolve(reference.as_ref()) {
            result.push_str(value);
            return Ok(());
        }
        Err(EscapeError::UnrecognizedEntity(0..len, reference.to_string()).into())
    }

    #[inline]
    pub(crate) fn read_to_end(&mut self, name: QName) -> Result<(), DeError> {
        match self.lookahead {
            // We pre-read event with the same name that is required to be skipped.
            // First call of `read_to_end` will end out pre-read event, the second
            // will consume other events
            Ok(PayloadEvent::Start(ref e)) if e.name() == name => {
                let result1 = self.reader.read_to_end(name);
                let result2 = self.reader.read_to_end(name);

                // In case of error `next_impl` returns `Eof`
                let _ = self.next_impl();
                result1?;
                result2?;
            }
            // We pre-read event with the same name that is required to be skipped.
            // Because this is end event, we already consume the whole tree, so
            // nothing to do, just update lookahead
            Ok(PayloadEvent::End(ref e)) if e.name() == name => {
                let _ = self.next_impl();
            }
            Ok(_) => {
                let result = self.reader.read_to_end(name);

                // In case of error `next_impl` returns `Eof`
                let _ = self.next_impl();
                result?;
            }
            // Read next lookahead event, unpack error from the current lookahead
            Err(_) => {
                self.next_impl()?;
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn decoder(&self) -> Decoder {
        self.reader.decoder()
    }
}
