//! The [`Deserializer`] itself and the free `from_str` / `from_reader` functions.

use std::borrow::Cow;
#[cfg(feature = "overlapped-lists")]
use std::collections::VecDeque;
use std::io::BufRead;
#[cfg(feature = "overlapped-lists")]
use std::num::NonZeroUsize;

use serde::de::{
    self, Deserialize, DeserializeOwned, DeserializeSeed, IntoDeserializer, SeqAccess, Visitor,
};

use crate::XmlVersion;
use crate::de::event::{DeEvent, XmlReader};
use crate::de::map::ElementMapAccess;
use crate::de::resolver::{EntityResolver, PredefinedEntityResolver};
use crate::de::simple_type::SimpleTypeDeserializer;
use crate::de::var;
use crate::de::xml_read::{IoReader, SliceReader, XmlRead};
use crate::errors::Error;
use crate::errors::serialize::DeError;
use crate::name::QName;
use crate::reader::NsReader;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Deserialize an instance of type `T` from a string of XML text.
pub fn from_str<'de, T>(s: &'de str) -> Result<T, DeError>
where
    T: Deserialize<'de>,
{
    let mut de = Deserializer::from_str(s);
    T::deserialize(&mut de)
}

/// Deserialize from a reader. This method will do internal copies of data
/// read from `reader`. If you want have a `&str` input and want to borrow
/// as much as possible, use [`from_str`].
pub fn from_reader<R, T>(reader: R) -> Result<T, DeError>
where
    R: BufRead,
    T: DeserializeOwned,
{
    let mut de = Deserializer::from_reader(reader);
    T::deserialize(&mut de)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A structure that deserializes XML into Rust values.
pub struct Deserializer<'de, R, E: EntityResolver = PredefinedEntityResolver>
where
    R: XmlRead<'de>,
{
    /// An XML reader that streams events into this deserializer
    pub(crate) reader: XmlReader<'de, R, E>,

    /// When deserializing sequences sometimes we have to skip unwanted events.
    /// That events should be stored and then replayed. This is a replay buffer,
    /// that streams events while not empty. When it exhausted, events will
    /// requested from [`Self::reader`].
    #[cfg(feature = "overlapped-lists")]
    pub(crate) read: VecDeque<DeEvent<'de>>,
    /// When deserializing sequences sometimes we have to skip events, because XML
    /// is tolerant to elements order and even if in the XSD order is strictly
    /// specified (using `xs:sequence`) most of XML parsers allows order violations.
    /// That means, that elements, forming a sequence, could be overlapped with
    /// other elements, do not related to that sequence.
    ///
    /// In order to support this, deserializer will scan events and skip unwanted
    /// events, store them here. After call [`Self::start_replay()`] all events
    /// moved from this to [`Self::read`].
    #[cfg(feature = "overlapped-lists")]
    pub(crate) write: VecDeque<DeEvent<'de>>,
    /// Maximum number of events that can be skipped when processing sequences
    /// that occur out-of-order. This field is used to prevent potential
    /// denial-of-service (DoS) attacks which could cause infinite memory
    /// consumption when parsing a very large amount of XML into a sequence field.
    #[cfg(feature = "overlapped-lists")]
    pub(crate) limit: Option<NonZeroUsize>,

    #[cfg(not(feature = "overlapped-lists"))]
    pub(crate) peek: Option<DeEvent<'de>>,

    /// One-event lookahead past [`Self::peek`]. Filled only by [`Self::peek_second()`]
    /// and drained by [`Self::next()`], which shifts it into [`Self::peek`]. It stays
    /// `None` on every code path that does not consult a second event, so ordinary
    /// deserialization behaves exactly as if this slot did not exist.
    #[cfg(not(feature = "overlapped-lists"))]
    pub(crate) peek2: Option<DeEvent<'de>>,

    /// Buffer to store attribute name as a field name exposed to serde consumers
    pub(crate) key_buf: String,
}

impl<'de, R, E> Deserializer<'de, R, E>
where
    R: XmlRead<'de>,
    E: EntityResolver,
{
    /// Create an XML deserializer from one of the possible oxixml_xml input sources.
    ///
    /// Typically it is more convenient to use one of these methods instead:
    ///
    ///  - [`Deserializer::from_str`]
    ///  - [`Deserializer::from_reader`]
    pub(crate) fn new(reader: R, entity_resolver: E) -> Self {
        Self {
            reader: XmlReader::new(reader, entity_resolver),

            #[cfg(feature = "overlapped-lists")]
            read: VecDeque::new(),
            #[cfg(feature = "overlapped-lists")]
            write: VecDeque::new(),
            #[cfg(feature = "overlapped-lists")]
            limit: None,

            #[cfg(not(feature = "overlapped-lists"))]
            peek: None,
            #[cfg(not(feature = "overlapped-lists"))]
            peek2: None,

            key_buf: String::new(),
        }
    }

    /// Returns `true` if all events was consumed.
    pub fn is_empty(&self) -> bool {
        #[cfg(feature = "overlapped-lists")]
        let event = self.read.front();

        #[cfg(not(feature = "overlapped-lists"))]
        let event = self.peek.as_ref();

        match event {
            None | Some(DeEvent::Eof) => self.reader.is_empty(),
            _ => false,
        }
    }

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
    /// let reader: &NsReader<_> = de.get_ref().get_ref();
    ///
    /// assert_eq!(reader.error_position(), 28);
    /// assert_eq!(reader.buffer_position(), 41);
    /// ```
    pub const fn get_ref(&self) -> &R {
        &self.reader.reader
    }

    /// Set the maximum number of events that could be skipped during deserialization
    /// of sequences.
    ///
    /// If `<element>` contains more than specified nested elements, `$text` or
    /// CDATA nodes, then [`DeError::TooManyEvents`] will be returned during
    /// deserialization of sequence field (any type that uses [`deserialize_seq`]
    /// for the deserialization, for example, `Vec<T>`).
    ///
    /// This method can be used to prevent a [DoS] attack and infinite memory
    /// consumption when parsing a very large XML to a sequence field.
    ///
    /// It is strongly recommended to set limit to some value when you parse data
    /// from untrusted sources. You should choose a value that your typical XMLs
    /// can have _between_ different elements that corresponds to the same sequence.
    ///
    /// # Examples
    ///
    /// Let's imagine, that we deserialize such structure:
    /// ```
    /// struct List {
    ///   item: Vec<()>,
    /// }
    /// ```
    ///
    /// The XML that we try to parse look like this:
    /// ```xml
    /// <any-name>
    ///   <item/>
    ///   <!-- Bufferization starts at this point -->
    ///   <another-item>
    ///     <some-element>with text</some-element>
    ///     <yet-another-element/>
    ///   </another-item>
    ///   <!-- Buffer will be emptied at this point; 7 events were buffered -->
    ///   <item/>
    ///   <!-- There is nothing to buffer, because elements follows each other -->
    ///   <item/>
    /// </any-name>
    /// ```
    ///
    /// There, when we deserialize the `item` field, we need to buffer 7 events,
    /// before we can deserialize the second `<item/>`:
    ///
    /// - `<another-item>`
    /// - `<some-element>`
    /// - `$text(with text)`
    /// - `</some-element>`
    /// - `<yet-another-element/>` (virtual start event)
    /// - `<yet-another-element/>` (virtual end event)
    /// - `</another-item>`
    ///
    /// Note, that `<yet-another-element/>` internally represented as 2 events:
    /// one for the start tag and one for the end tag. In the future this can be
    /// eliminated, but for now we use [auto-expanding feature] of a reader,
    /// because this simplifies deserializer code.
    ///
    /// [`deserialize_seq`]: serde::Deserializer::deserialize_seq
    /// [DoS]: https://en.wikipedia.org/wiki/Denial-of-service_attack
    /// [auto-expanding feature]: crate::reader::Config::expand_empty_elements
    #[cfg(feature = "overlapped-lists")]
    #[cfg_attr(docsrs, doc(cfg(feature = "overlapped-lists")))]
    pub fn event_buffer_size(&mut self, limit: Option<NonZeroUsize>) -> &mut Self {
        self.limit = limit;
        self
    }

    #[cfg(feature = "overlapped-lists")]
    pub(crate) fn peek(&mut self) -> Result<&DeEvent<'de>, DeError> {
        if self.read.is_empty() {
            self.read.push_front(self.reader.next()?);
        }
        if let Some(event) = self.read.front() {
            return Ok(event);
        }
        // SAFETY: `self.read` was filled in the code above.
        // NOTE: Can be replaced with `unsafe { std::hint::unreachable_unchecked() }`
        // if unsafe code will be allowed
        unreachable!()
    }
    #[cfg(not(feature = "overlapped-lists"))]
    pub(crate) fn peek(&mut self) -> Result<&DeEvent<'de>, DeError> {
        match &mut self.peek {
            Some(event) => Ok(event),
            empty_peek @ None => Ok(empty_peek.insert(self.reader.next()?)),
        }
    }

    /// Returns the event immediately after the one reported by [`Self::peek()`],
    /// consuming neither. Returns `None` only if the input ends before a second
    /// event exists. Used to resolve element shapes that a single event leaves
    /// ambiguous (a text-only `<tag>text</tag>` versus mixed or nested content).
    #[cfg(feature = "overlapped-lists")]
    pub(crate) fn peek_second(&mut self) -> Result<Option<&DeEvent<'de>>, DeError> {
        while self.read.len() < 2 {
            let event = self.reader.next()?;
            let reached_eof = matches!(event, DeEvent::Eof);
            self.read.push_back(event);
            if reached_eof {
                break;
            }
        }
        Ok(self.read.get(1))
    }
    #[cfg(not(feature = "overlapped-lists"))]
    pub(crate) fn peek_second(&mut self) -> Result<Option<&DeEvent<'de>>, DeError> {
        if self.peek.is_none() {
            self.peek = Some(self.reader.next()?);
        }
        if self.peek2.is_none() {
            self.peek2 = Some(self.reader.next()?);
        }
        Ok(self.peek2.as_ref())
    }

    #[inline]
    pub(crate) fn last_peeked(&self) -> &DeEvent<'de> {
        #[cfg(feature = "overlapped-lists")]
        {
            self.read
                .front()
                .expect("`Deserializer::peek()` should be called")
        }
        #[cfg(not(feature = "overlapped-lists"))]
        {
            self.peek
                .as_ref()
                .expect("`Deserializer::peek()` should be called")
        }
    }

    pub(crate) fn next(&mut self) -> Result<DeEvent<'de>, DeError> {
        // Replay skipped or peeked events
        #[cfg(feature = "overlapped-lists")]
        if let Some(event) = self.read.pop_front() {
            return Ok(event);
        }
        #[cfg(not(feature = "overlapped-lists"))]
        if let Some(e) = self.peek.take() {
            // Keep the lookahead contiguous: the second slot becomes the first.
            self.peek = self.peek2.take();
            return Ok(e);
        }
        self.reader.next()
    }

    pub(crate) fn skip_whitespaces(&mut self) -> Result<(), DeError> {
        loop {
            match self.peek()? {
                DeEvent::Text(e) if e.is_blank() => {
                    self.next()?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    /// Returns the mark after which all events, skipped by [`Self::skip()`] call,
    /// should be replayed after calling [`Self::start_replay()`].
    #[cfg(feature = "overlapped-lists")]
    #[inline]
    #[must_use = "returned checkpoint should be used in `start_replay`"]
    pub(crate) fn skip_checkpoint(&self) -> usize {
        self.write.len()
    }

    /// Extracts XML tree of events from and stores them in the skipped events
    /// buffer from which they can be retrieved later. You MUST call
    /// [`Self::start_replay()`] after calling this to give access to the skipped
    /// events and release internal buffers.
    #[cfg(feature = "overlapped-lists")]
    pub(crate) fn skip(&mut self) -> Result<(), DeError> {
        let event = self.next()?;
        self.skip_event(event)?;
        // Skip all subtree, if we skip a start event
        if let Some(DeEvent::Start(e)) = self.write.back() {
            let end = e.name().as_ref().to_owned();
            let mut depth = 0;
            loop {
                let event = self.next()?;
                match event {
                    DeEvent::Start(ref e) if e.name().as_ref() == end => {
                        self.skip_event(event)?;
                        depth += 1;
                    }
                    DeEvent::End(ref e) if e.name().as_ref() == end => {
                        self.skip_event(event)?;
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    DeEvent::Eof => {
                        self.skip_event(event)?;
                        break;
                    }
                    _ => self.skip_event(event)?,
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "overlapped-lists")]
    #[inline]
    pub(crate) fn skip_event(&mut self, event: DeEvent<'de>) -> Result<(), DeError> {
        if let Some(max) = self.limit {
            if self.write.len() >= max.get() {
                return Err(DeError::TooManyEvents(max));
            }
        }
        self.write.push_back(event);
        Ok(())
    }

    /// Moves buffered events, skipped after given `checkpoint` from [`Self::write`]
    /// skip buffer to [`Self::read`] buffer.
    ///
    /// After calling this method, [`Self::peek()`] and [`Self::next()`] starts
    /// return events that was skipped previously by calling [`Self::skip()`],
    /// and only when all that events will be consumed, the deserializer starts
    /// to drain events from underlying reader.
    ///
    /// This method MUST be called if any number of [`Self::skip()`] was called
    /// after [`Self::new()`] or `start_replay()` or you'll lost events.
    #[cfg(feature = "overlapped-lists")]
    pub(crate) fn start_replay(&mut self, checkpoint: usize) {
        if checkpoint == 0 {
            self.write.append(&mut self.read);
            std::mem::swap(&mut self.read, &mut self.write);
        } else {
            let mut read = self.write.split_off(checkpoint);
            read.append(&mut self.read);
            self.read = read;
        }
    }

    #[inline]
    pub(crate) fn read_string(&mut self) -> Result<Cow<'de, str>, DeError> {
        self.read_string_impl(true)
    }

    /// Consumes consequent [`Text`] and [`CData`] (both a referred below as a _text_)
    /// events, merge them into one string. If there are no such events, returns
    /// an empty string.
    ///
    /// If `allow_start` is `false`, then only text events are consumed, for other
    /// events an error is returned (see table below).
    ///
    /// If `allow_start` is `true`, then two or three events are expected:
    /// - [`DeEvent::Start`];
    /// - _(optional)_ [`DeEvent::Text`] which content is returned;
    /// - [`DeEvent::End`]. If text event was missed, an empty string is returned.
    ///
    /// Corresponding events are consumed.
    ///
    /// # Handling events
    ///
    /// The table below shows how events is handled by this method:
    ///
    /// |Event             |XML                        |Handling
    /// |------------------|---------------------------|----------------------------------------
    /// |[`DeEvent::Start`]|`<tag>...</tag>`           |if `allow_start == true`, result determined by the second table, otherwise emits [`UnexpectedStart("tag")`](DeError::UnexpectedStart)
    /// |[`DeEvent::End`]  |`</any-tag>`               |Normally impossible, the reader guarantees tags are balanced; reachable only when [`Config::allow_unmatched_ends`] is set, in which case emits [`InvalidXml`](DeError::InvalidXml) wrapping [`UnmatchedEndTag`](crate::errors::IllFormedError::UnmatchedEndTag)
    /// |[`DeEvent::Text`] |`text content` or `<![CDATA[cdata content]]>` (probably mixed)|Returns event content unchanged
    /// |[`DeEvent::Eof`]  |                           |Emits [`UnexpectedEof`](DeError::UnexpectedEof)
    ///
    /// [`Config::allow_unmatched_ends`]: crate::reader::Config::allow_unmatched_ends
    ///
    /// Second event, consumed if [`DeEvent::Start`] was received and `allow_start == true`:
    ///
    /// |Event             |XML                        |Handling
    /// |------------------|---------------------------|----------------------------------------------------------------------------------
    /// |[`DeEvent::Start`]|`<any-tag>...</any-tag>`   |Emits [`UnexpectedStart("any-tag")`](DeError::UnexpectedStart)
    /// |[`DeEvent::End`]  |`</tag>`                   |Returns an empty slice. The reader guarantee that tag will match the open one
    /// |[`DeEvent::Text`] |`text content` or `<![CDATA[cdata content]]>` (probably mixed)|Returns event content unchanged, expects the `</tag>` after that
    /// |[`DeEvent::Eof`]  |                           |Emits [`InvalidXml(IllFormed(MissingEndTag))`](DeError::InvalidXml)
    ///
    /// [`Text`]: Event::Text
    /// [`CData`]: Event::CData
    pub(crate) fn read_string_impl(&mut self, allow_start: bool) -> Result<Cow<'de, str>, DeError> {
        match self.next()? {
            // Reached by doc tests only: this file, lines 979 and 996
            DeEvent::Text(e) => Ok(e.text),
            // allow one nested level
            // Reached by trivial::{...}::{field, field_nested, field_tag_after, field_tag_before, nested, tag_after, tag_before, wrapped}
            DeEvent::Start(e) if allow_start => self.read_text(e.name()),
            // Reached by `value_field_rejects_nested_element` in tests/serde-de.rs:
            // a `$value`-typed primitive field whose position in the document holds
            // a nested element instead of text.
            DeEvent::Start(e) => Err(DeError::UnexpectedStart(e.name().as_ref().to_owned())),
            // Only reachable when `Config::allow_unmatched_ends` lets the reader emit
            // an `End` with no matching `Start` anywhere in the document; report it the
            // same way the reader itself would if that option were off.
            DeEvent::End(e) => Err(Error::unmatched_end(e.name(), self.reader.decoder()).into()),
            // Reached by trivial::{empty_doc, only_comment}
            DeEvent::Eof => Err(DeError::UnexpectedEof),
        }
    }
    /// Consumes one [`DeEvent::Text`] event and ensures that it is followed by the
    /// [`DeEvent::End`] event.
    ///
    /// # Parameters
    /// - `name`: name of a tag opened before reading text. The corresponding end tag
    ///   should present in input just after the text
    pub(crate) fn read_text(&mut self, name: QName) -> Result<Cow<'de, str>, DeError> {
        match self.next()? {
            DeEvent::Text(e) => match self.next()? {
                // The matching tag name is guaranteed by the reader
                // Reached by trivial::{...}::{field, wrapped}
                DeEvent::End(_) => Ok(e.text),
                // SAFETY: Cannot be two consequent Text events, they would be merged into one
                DeEvent::Text(_) => unreachable!(),
                // Reached by trivial::{...}::{field_tag_after, tag_after}
                DeEvent::Start(e) => Err(DeError::UnexpectedStart(e.name().as_ref().to_owned())),
                // Reached by struct_::non_closed::elements_child
                DeEvent::Eof => Err(Error::missed_end(name, self.reader.decoder()).into()),
            },
            // We can get End event in case of `<tag></tag>` or `<tag/>` input
            // Return empty text in that case
            // The matching tag name is guaranteed by the reader
            // Reached by {...}::xs_list::empty
            DeEvent::End(_) => Ok("".into()),
            // Reached by trivial::{...}::{field_nested, field_tag_before, nested, tag_before}
            DeEvent::Start(s) => Err(DeError::UnexpectedStart(s.name().as_ref().to_owned())),
            // Reached by struct_::non_closed::elements_child
            DeEvent::Eof => Err(Error::missed_end(name, self.reader.decoder()).into()),
        }
    }

    /// Drops all events until event with [name](BytesEnd::name()) `name` won't be
    /// dropped. This method should be called after [`Self::next()`]
    #[cfg(feature = "overlapped-lists")]
    pub(crate) fn read_to_end(&mut self, name: QName) -> Result<(), DeError> {
        let mut depth = 0;
        loop {
            match self.read.pop_front() {
                Some(DeEvent::Start(e)) if e.name() == name => {
                    depth += 1;
                }
                Some(DeEvent::End(e)) if e.name() == name => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }

                // Drop all other skipped events
                Some(_) => continue,

                // If we do not have skipped events, use effective reading that will
                // not allocate memory for events
                None => {
                    // We should close all opened tags, because we could buffer
                    // Start events, but not the corresponding End events. So we
                    // keep reading events until we exit all nested tags.
                    // `read_to_end()` will return an error if an Eof was encountered
                    // preliminary (in case of malformed XML).
                    //
                    // <tag><tag></tag></tag>
                    // ^^^^^^^^^^             - buffered in `self.read`, when `self.read_to_end()` is called, depth = 2
                    //           ^^^^^^       - read by the first call of `self.reader.read_to_end()`
                    //                 ^^^^^^ - read by the second call of `self.reader.read_to_end()`
                    loop {
                        self.reader.read_to_end(name)?;
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    break;
                }
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "overlapped-lists"))]
    pub(crate) fn read_to_end(&mut self, name: QName) -> Result<(), DeError> {
        // First one might be in self.peek
        match self.next()? {
            DeEvent::Start(e) => self.reader.read_to_end(e.name())?,
            DeEvent::End(e) if e.name() == name => return Ok(()),
            _ => (),
        }
        self.reader.read_to_end(name)
    }

    pub(crate) fn skip_next_tree(&mut self) -> Result<(), DeError> {
        let DeEvent::Start(start) = self.next()? else {
            unreachable!("Only call this if the next event is a start event")
        };
        let name = start.name();
        self.read_to_end(name)
    }

    /// Method for testing Deserializer implementation. Checks that all events was consumed during
    /// deserialization. Panics if the next event will not be [`DeEvent::Eof`].
    #[doc(hidden)]
    #[track_caller]
    pub fn check_eof_reached(&mut self) {
        // Deserializer may not consume trailing spaces, that is normal
        self.skip_whitespaces().expect("cannot skip whitespaces");
        let event = self.peek().expect("cannot peek event");
        assert_eq!(
            *event,
            DeEvent::Eof,
            "the whole XML document should be consumed, expected `Eof`",
        );
    }
}

impl<'de> Deserializer<'de, SliceReader<'de>> {
    /// Create a new deserializer that will borrow data from the specified string.
    ///
    /// Deserializer created with this method will not resolve custom entities.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(source: &'de str) -> Self {
        Self::from_str_with_resolver(source, PredefinedEntityResolver)
    }

    /// Create a new deserializer that will borrow data from the specified preconfigured
    /// reader.
    ///
    /// Deserializer created with this method will not resolve custom entities.
    ///
    /// Note, that config option [`Config::expand_empty_elements`] will be set to `true`.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxixml_xml::de::Deserializer;
    /// # use oxixml_xml::NsReader;
    /// # use serde::Deserialize;
    /// #
    /// #[derive(Deserialize, PartialEq, Debug)]
    /// struct Object<'a> {
    ///     tag: &'a str,
    /// }
    ///
    /// let mut reader = NsReader::from_str("<xml><tag>    test    </tag></xml>");
    ///
    /// let mut de = Deserializer::borrowing(reader.clone());
    /// let obj = Object::deserialize(&mut de).expect("should be Ok");
    /// assert_eq!(obj, Object { tag: "    test    " });
    ///
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut de = Deserializer::borrowing(reader);
    /// let obj = Object::deserialize(&mut de).expect("should be Ok");
    /// assert_eq!(obj, Object { tag: "test" });
    /// ```
    ///
    /// [`Config::expand_empty_elements`]: crate::reader::Config::expand_empty_elements
    #[inline]
    pub fn borrowing(reader: NsReader<&'de [u8]>) -> Self {
        Self::borrowing_with_resolver(reader, PredefinedEntityResolver)
    }
}

impl<'de, E> Deserializer<'de, SliceReader<'de>, E>
where
    E: EntityResolver,
{
    /// Create a new deserializer that will borrow data from the specified string
    /// and use the specified entity resolver.
    pub fn from_str_with_resolver(source: &'de str, entity_resolver: E) -> Self {
        Self::borrowing_with_resolver(NsReader::from_str(source), entity_resolver)
    }

    /// Create a new deserializer that will borrow data from the specified preconfigured
    /// reader and use the specified entity resolver.
    ///
    /// Note, that config option [`Config::expand_empty_elements`] will be set to `true`.
    ///
    /// [`Config::expand_empty_elements`]: crate::reader::Config::expand_empty_elements
    pub fn borrowing_with_resolver(mut reader: NsReader<&'de [u8]>, entity_resolver: E) -> Self {
        let config = reader.config_mut();
        config.expand_empty_elements = true;

        Self::new(
            SliceReader {
                reader,
                version: XmlVersion::Implicit1_0,
            },
            entity_resolver,
        )
    }
}

impl<'de, R> Deserializer<'de, IoReader<R>>
where
    R: BufRead,
{
    /// Create a new deserializer that will copy data from the specified reader
    /// into internal buffer.
    ///
    /// If you already have a string use [`Self::from_str`] instead, because it
    /// will borrow instead of copy. If you have `&[u8]` which is known to represent
    /// UTF-8, you can decode it first before using [`from_str`].
    ///
    /// Deserializer created with this method will not resolve custom entities.
    pub fn from_reader(reader: R) -> Self {
        Self::with_resolver(reader, PredefinedEntityResolver)
    }

    /// Create a new deserializer that will copy data from the specified preconfigured
    /// reader into internal buffer.
    ///
    /// Deserializer created with this method will not resolve custom entities.
    ///
    /// Note, that config option [`Config::expand_empty_elements`] will be set to `true`.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxixml_xml::de::Deserializer;
    /// # use oxixml_xml::NsReader;
    /// # use serde::Deserialize;
    /// #
    /// #[derive(Deserialize, PartialEq, Debug)]
    /// struct Object {
    ///     tag: String,
    /// }
    ///
    /// let mut reader = NsReader::from_str("<xml><tag>    test    </tag></xml>");
    ///
    /// let mut de = Deserializer::buffering(reader.clone());
    /// let obj = Object::deserialize(&mut de).expect("should be Ok");
    /// assert_eq!(obj, Object { tag: "    test    ".to_string() });
    ///
    /// reader.config_mut().trim_text(true);
    ///
    /// let mut de = Deserializer::buffering(reader);
    /// let obj = Object::deserialize(&mut de).expect("should be Ok");
    /// assert_eq!(obj, Object { tag: "test".to_string() });
    /// ```
    ///
    /// [`Config::expand_empty_elements`]: crate::reader::Config::expand_empty_elements
    #[inline]
    pub fn buffering(reader: NsReader<R>) -> Self {
        Self::buffering_with_resolver(reader, PredefinedEntityResolver)
    }
}

impl<'de, R, E> Deserializer<'de, IoReader<R>, E>
where
    R: BufRead,
    E: EntityResolver,
{
    /// Create a new deserializer that will copy data from the specified reader
    /// into internal buffer and use the specified entity resolver.
    ///
    /// If you already have a string use [`Self::from_str`] instead, because it
    /// will borrow instead of copy. If you have `&[u8]` which is known to represent
    /// UTF-8, you can decode it first before using [`from_str`].
    pub fn with_resolver(reader: R, entity_resolver: E) -> Self {
        let mut reader = NsReader::from_reader(reader);
        let config = reader.config_mut();
        config.expand_empty_elements = true;

        Self::new(
            IoReader {
                reader,
                buf: Vec::new(),
                version: XmlVersion::Implicit1_0,
            },
            entity_resolver,
        )
    }

    /// Create new deserializer that will copy data from the specified preconfigured reader
    /// into internal buffer and use the specified entity resolver.
    ///
    /// Note, that config option [`Config::expand_empty_elements`] will be set to `true`.
    ///
    /// [`Config::expand_empty_elements`]: crate::reader::Config::expand_empty_elements
    pub fn buffering_with_resolver(mut reader: NsReader<R>, entity_resolver: E) -> Self {
        let config = reader.config_mut();
        config.expand_empty_elements = true;

        Self::new(
            IoReader {
                reader,
                buf: Vec::new(),
                version: XmlVersion::Implicit1_0,
            },
            entity_resolver,
        )
    }
}

impl<'de, R, E> de::Deserializer<'de> for &mut Deserializer<'de, R, E>
where
    R: XmlRead<'de>,
    E: EntityResolver,
{
    type Error = DeError;

    deserialize_primitives!();

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, DeError>
    where
        V: Visitor<'de>,
    {
        // When document is pretty-printed there could be whitespaces before the root element
        self.skip_whitespaces()?;
        match self.next()? {
            DeEvent::Start(e) => visitor.visit_map(ElementMapAccess::new(self, e, fields)),
            // Normally impossible, the reader guarantees tags are balanced; reachable
            // only when `Config::allow_unmatched_ends` lets a stray closing tag through.
            DeEvent::End(e) => Err(Error::unmatched_end(e.name(), self.reader.decoder()).into()),
            // Deserializer methods are only hints, if deserializer could not satisfy
            // request, it should return the data that it has. It is responsibility
            // of a Visitor to return an error if it does not understand the data
            DeEvent::Text(e) => match e.text {
                Cow::Borrowed(s) => visitor.visit_borrowed_str(s),
                Cow::Owned(s) => visitor.visit_string(s),
            },
            DeEvent::Eof => Err(DeError::UnexpectedEof),
        }
    }

    /// Unit represented in XML as a `xs:element` or text/CDATA content.
    /// Any content inside `xs:element` is ignored and skipped.
    ///
    /// Produces unit struct from any of following inputs:
    /// - any `<tag ...>...</tag>`
    /// - any `<tag .../>`
    /// - any consequent text / CDATA content (can consist of several parts
    ///   delimited by comments and processing instructions)
    ///
    /// # Events handling
    ///
    /// |Event             |XML                        |Handling
    /// |------------------|---------------------------|-------------------------------------------
    /// |[`DeEvent::Start`]|`<tag>...</tag>`           |Calls `visitor.visit_unit()`, consumes all events up to and including corresponding `End` event
    /// |[`DeEvent::End`]  |`</tag>`                   |Normally impossible, the reader guarantees tags are balanced; reachable only when [`Config::allow_unmatched_ends`] is set, in which case emits [`InvalidXml`](DeError::InvalidXml) wrapping [`UnmatchedEndTag`](crate::errors::IllFormedError::UnmatchedEndTag)
    /// |[`DeEvent::Text`] |`text content` or `<![CDATA[cdata content]]>` (probably mixed)|Calls `visitor.visit_unit()`. The content is ignored
    /// |[`DeEvent::Eof`]  |                           |Emits [`UnexpectedEof`](DeError::UnexpectedEof)
    ///
    /// [`Config::allow_unmatched_ends`]: crate::reader::Config::allow_unmatched_ends
    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, DeError>
    where
        V: Visitor<'de>,
    {
        match self.next()? {
            DeEvent::Start(s) => {
                self.read_to_end(s.name())?;
                visitor.visit_unit()
            }
            DeEvent::Text(_) => visitor.visit_unit(),
            // Normally impossible, the reader guarantees tags are balanced; reachable
            // only when `Config::allow_unmatched_ends` lets a stray closing tag through.
            DeEvent::End(e) => Err(Error::unmatched_end(e.name(), self.reader.decoder()).into()),
            DeEvent::Eof => Err(DeError::UnexpectedEof),
        }
    }

    /// Forwards deserialization of the inner type. Always calls [`Visitor::visit_newtype_struct`]
    /// with the same deserializer.
    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, DeError>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, DeError>
    where
        V: Visitor<'de>,
    {
        // When document is pretty-printed there could be whitespaces before the root element
        // which represents the enum variant
        // Checked by `top_level::list_of_enum` test in serde-de-seq
        self.skip_whitespaces()?;
        visitor.visit_enum(var::EnumAccess::new(self))
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, DeError>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(self)
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, DeError>
    where
        V: Visitor<'de>,
    {
        // We cannot use result of `peek()` directly because of borrow checker
        let _ = self.peek()?;
        match self.last_peeked() {
            DeEvent::Text(t) if t.is_empty() => visitor.visit_none(),
            DeEvent::Eof => visitor.visit_none(),
            // if the `xsi:nil` attribute is set to true we got a none value
            DeEvent::Start(start) if self.reader.reader.has_nil_attr(start) => {
                self.skip_next_tree()?;
                visitor.visit_none()
            }
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, DeError>
    where
        V: Visitor<'de>,
    {
        match self.peek()? {
            DeEvent::Text(_) => self.deserialize_str(visitor),
            _ => self.deserialize_map(visitor),
        }
    }
}

/// An accessor to sequence elements forming a value for top-level sequence of XML
/// elements.
///
/// Technically, multiple top-level elements violates XML rule of only one top-level
/// element, but we consider this as several concatenated XML documents.
impl<'de, R, E> SeqAccess<'de> for &mut Deserializer<'de, R, E>
where
    R: XmlRead<'de>,
    E: EntityResolver,
{
    type Error = DeError;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        // When document is pretty-printed there could be whitespaces before, between
        // and after root elements. We cannot defer decision if we need to skip spaces
        // or not: if we have a sequence of type that does not accept blank text, it
        // will need to return something and it can return only error. For example,
        // it can be enum without `$text` variant
        // Checked by `top_level::list_of_enum` test in serde-de-seq
        self.skip_whitespaces()?;
        match self.peek()? {
            DeEvent::Eof => Ok(None),

            // Start(tag), End(tag), Text
            _ => seed.deserialize(&mut **self).map(Some),
        }
    }
}

impl<'de, R, E> IntoDeserializer<'de, DeError> for &mut Deserializer<'de, R, E>
where
    R: XmlRead<'de>,
    E: EntityResolver,
{
    type Deserializer = Self;

    #[inline]
    fn into_deserializer(self) -> Self {
        self
    }
}
