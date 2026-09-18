//! XML attributes module.
//!
//! Provides an iterator ([`Attributes`]) over attribute key/value pairs, the
//! [`Attribute`] value type, the raw [`Attr`] span type, and the [`AttrError`]
//! error, mirroring quick-xml 0.41 `events::attributes`.

use crate::XmlVersion;
use crate::encoding::Decoder;
use crate::errors::Result as XmlResult;
use crate::escape::{escape, resolve_predefined_entity};
use crate::name::{LocalName, Namespace, NamespaceResolver, QName};
use crate::utils::{Bytes, is_whitespace};

use std::collections::HashSet;
use std::fmt::{self, Debug, Display, Formatter};
use std::hash::{BuildHasherDefault, DefaultHasher, Hasher};
use std::iter::FusedIterator;
use std::{borrow::Cow, ops::Range};

/// A key/value XML attribute.
///
/// The `value` field stores raw bytes, possibly containing escape sequences. Most
/// users will want the value via [`normalized_value`](Self::normalized_value) or
/// [`decoded_and_normalized_value`](Self::decoded_and_normalized_value).
#[derive(Clone, Eq, PartialEq)]
pub struct Attribute<'a> {
    /// The key that uniquely identifies the attribute.
    ///
    /// If [`Attributes::with_checks`] is turned off, the key might not be unique.
    pub key: QName<'a>,
    /// The raw value of the attribute.
    pub value: Cow<'a, [u8]>,
}

impl<'a> Attribute<'a> {
    /// Returns the attribute value normalized as per [the XML specification].
    ///
    /// The document **must** be UTF-8 encoded. The characters `\t`, `\r`, `\n` are
    /// replaced with a space (`0x20`), and the five predefined entities are
    /// unescaped. This allocates unless no normalization is required.
    ///
    /// [the XML specification]: https://www.w3.org/TR/xml11/#AVNormalize
    pub fn normalized_value(&self, version: XmlVersion) -> XmlResult<Cow<'a, str>> {
        // `resolve_predefined_entity` returns only non-recursive replacements,
        // so a depth of 1 is enough.
        self.normalized_value_with(version, 1, resolve_predefined_entity)
    }

    /// Returns the attribute value normalized as per [the XML specification], using
    /// a custom entity resolver. The document **must** be UTF-8 encoded.
    ///
    /// # Parameters
    ///
    /// - `depth`: maximum number of nested entities that can be expanded before
    ///   [`EscapeError::TooManyNestedEntities`] is returned;
    /// - `resolve_entity`: resolves named entities (predefined entities are not
    ///   resolved by this function, but you may call [`resolve_predefined_entity`]).
    ///
    /// [the XML specification]: https://www.w3.org/TR/xml11/#AVNormalize
    /// [`EscapeError::TooManyNestedEntities`]: crate::escape::EscapeError::TooManyNestedEntities
    pub fn normalized_value_with<'entity>(
        &self,
        version: XmlVersion,
        depth: usize,
        resolve_entity: impl FnMut(&str) -> Option<&'entity str>,
    ) -> XmlResult<Cow<'a, str>> {
        use crate::encoding::EncodingError;
        use std::str::from_utf8;

        let decoded = match &self.value {
            Cow::Borrowed(bytes) => Cow::Borrowed(from_utf8(bytes).map_err(EncodingError::Utf8)?),
            Cow::Owned(bytes) => {
                Cow::Owned(from_utf8(bytes).map_err(EncodingError::Utf8)?.to_owned())
            }
        };

        match version.normalize_attribute_value(&decoded, depth, resolve_entity)? {
            Cow::Borrowed(_) => Ok(decoded),
            Cow::Owned(s) => Ok(s.into()),
        }
    }

    /// Decodes using the provided decoder and returns the attribute value
    /// normalized as per [the XML specification].
    ///
    /// [the XML specification]: https://www.w3.org/TR/xml11/#AVNormalize
    pub fn decoded_and_normalized_value(
        &self,
        version: XmlVersion,
        decoder: Decoder,
    ) -> XmlResult<Cow<'a, str>> {
        self.decoded_and_normalized_value_with(version, decoder, 1, resolve_predefined_entity)
    }

    /// Decodes using the provided decoder and returns the attribute value
    /// normalized as per [the XML specification], using a custom entity resolver.
    ///
    /// [the XML specification]: https://www.w3.org/TR/xml11/#AVNormalize
    pub fn decoded_and_normalized_value_with<'entity>(
        &self,
        version: XmlVersion,
        decoder: Decoder,
        depth: usize,
        resolve_entity: impl FnMut(&str) -> Option<&'entity str>,
    ) -> XmlResult<Cow<'a, str>> {
        let decoded = match &self.value {
            Cow::Borrowed(bytes) => decoder.decode(bytes)?,
            Cow::Owned(bytes) => decoder.decode(bytes)?.into_owned().into(),
        };

        match version.normalize_attribute_value(&decoded, depth, resolve_entity)? {
            Cow::Borrowed(_) => Ok(decoded),
            Cow::Owned(s) => Ok(s.into()),
        }
    }

    /// Returns the unescaped value.
    ///
    /// Escape sequences such as `&gt;` are replaced with their equivalents. This
    /// allocates if the value contains any escape sequences.
    #[cfg(any(doc, not(feature = "encoding")))]
    #[deprecated = "use `Self::normalized_value()`"]
    pub fn unescape_value(&self) -> XmlResult<Cow<'a, str>> {
        self.normalized_value_with(XmlVersion::Implicit1_0, 1, resolve_predefined_entity)
    }

    /// Decodes using UTF-8, then unescapes the value using a custom entity resolver.
    #[cfg(any(doc, not(feature = "encoding")))]
    #[deprecated = "use `Self::normalized_value_with()`"]
    #[inline]
    pub fn unescape_value_with<'entity>(
        &self,
        resolve_entity: impl FnMut(&str) -> Option<&'entity str>,
    ) -> XmlResult<Cow<'a, str>> {
        self.normalized_value_with(XmlVersion::Implicit1_0, 128, resolve_entity)
    }

    /// Decodes, then unescapes the value.
    #[deprecated = "use `Self::decoded_and_normalized_value()`"]
    pub fn decode_and_unescape_value(&self, decoder: Decoder) -> XmlResult<Cow<'a, str>> {
        self.decoded_and_normalized_value_with(
            XmlVersion::Implicit1_0,
            decoder,
            1,
            resolve_predefined_entity,
        )
    }

    /// Decodes, then unescapes the value using a custom entity resolver.
    #[deprecated = "use `Self::decoded_and_normalized_value_with()`"]
    pub fn decode_and_unescape_value_with<'entity>(
        &self,
        decoder: Decoder,
        resolve_entity: impl FnMut(&str) -> Option<&'entity str>,
    ) -> XmlResult<Cow<'a, str>> {
        self.decoded_and_normalized_value_with(
            XmlVersion::Implicit1_0,
            decoder,
            128,
            resolve_entity,
        )
    }

    /// If the attribute value [represents] a valid boolean, returns `Some`,
    /// otherwise `None`. The valid representations are `"true"`, `"false"`, `"1"`
    /// and `"0"`.
    ///
    /// ```
    /// # use oxixml_xml::events::attributes::Attribute;
    /// assert_eq!(Attribute::from(("attr", "false")).as_bool(), Some(false));
    /// assert_eq!(Attribute::from(("attr", "1")).as_bool(), Some(true));
    /// assert_eq!(Attribute::from(("attr", "not bool")).as_bool(), None);
    /// ```
    ///
    /// [represents]: https://www.w3.org/TR/xmlschema11-2/#boolean
    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self.value.as_ref() {
            b"1" | b"true" => Some(true),
            b"0" | b"false" => Some(false),
            _ => None,
        }
    }
}

impl<'a> Debug for Attribute<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("Attribute")
            .field("key", &Bytes(self.key.as_ref()))
            .field("value", &Bytes(&self.value))
            .finish()
    }
}

impl<'a> From<(&'a [u8], &'a [u8])> for Attribute<'a> {
    /// Creates a new attribute from raw bytes, without transforming key or value.
    fn from(val: (&'a [u8], &'a [u8])) -> Attribute<'a> {
        Attribute {
            key: QName(val.0),
            value: Cow::from(val.1),
        }
    }
}

impl<'a> From<(&'a str, &'a str)> for Attribute<'a> {
    /// Creates a new attribute from text. The key is stored as-is; the value is
    /// escaped.
    fn from(val: (&'a str, &'a str)) -> Attribute<'a> {
        Attribute {
            key: QName(val.0.as_bytes()),
            value: match escape(val.1) {
                Cow::Borrowed(s) => Cow::Borrowed(s.as_bytes()),
                Cow::Owned(s) => Cow::Owned(s.into_bytes()),
            },
        }
    }
}

impl<'a> From<(&'a str, Cow<'a, str>)> for Attribute<'a> {
    /// Creates a new attribute from text. The key is stored as-is; the value is
    /// escaped.
    fn from(val: (&'a str, Cow<'a, str>)) -> Attribute<'a> {
        Attribute {
            key: QName(val.0.as_bytes()),
            value: match escape(val.1) {
                Cow::Borrowed(s) => Cow::Borrowed(s.as_bytes()),
                Cow::Owned(s) => Cow::Owned(s.into_bytes()),
            },
        }
    }
}

impl<'a> From<Attr<&'a [u8]>> for Attribute<'a> {
    #[inline]
    fn from(attr: Attr<&'a [u8]>) -> Self {
        Self {
            key: attr.key(),
            value: Cow::Borrowed(attr.value()),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterator over XML attributes.
///
/// Yields `Result<Attribute, AttrError>`. An `Err` is yielded if an attribute is
/// malformed or duplicated. The duplicate check can be turned off with
/// [`with_checks(false)`](Self::with_checks).
#[derive(Clone)]
pub struct Attributes<'a> {
    /// Slice of the `BytesStart` corresponding to the attributes.
    bytes: &'a [u8],
    /// Iterator state, independent of the actual source of bytes.
    state: IterState,
    /// Encoding used for `bytes`.
    decoder: Decoder,
}

impl<'a> Attributes<'a> {
    /// Internal constructor, used by `BytesStart`; supplies data in the reader's encoding.
    #[inline]
    pub(crate) const fn wrap(buf: &'a [u8], pos: usize, html: bool, decoder: Decoder) -> Self {
        Self {
            bytes: buf,
            state: IterState::new(pos, html),
            decoder,
        }
    }

    /// Creates a new XML-style attribute iterator over `buf`, where attributes are
    /// of the form `name="value"` or `name='value'`. HTML-style attributes cause
    /// an error.
    ///
    /// `pos` is the position in `buf` where the tag name ends and the attributes
    /// begin.
    ///
    /// ```
    /// # use oxixml_xml::events::attributes::{Attribute, Attributes};
    /// let mut iter = Attributes::new("tag-name attr1 = 'value1' attr2='value2' ", 9);
    /// assert_eq!(iter.next(), Some(Ok(Attribute::from(("attr1", "value1")))));
    /// assert_eq!(iter.next(), Some(Ok(Attribute::from(("attr2", "value2")))));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub const fn new(buf: &'a str, pos: usize) -> Self {
        Self::wrap(buf.as_bytes(), pos, false, Decoder::utf8())
    }

    /// Creates a new attribute iterator over `buf`, allowing HTML attribute syntax
    /// (unquoted values and key-only attributes).
    ///
    /// ```
    /// # use oxixml_xml::events::attributes::{Attribute, Attributes};
    /// let mut iter = Attributes::html("tag-name attr1 = value1 attr2 ", 9);
    /// assert_eq!(iter.next(), Some(Ok(Attribute::from(("attr1", "value1")))));
    /// assert_eq!(iter.next(), Some(Ok(Attribute::from(("attr2", "")))));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub const fn html(buf: &'a str, pos: usize) -> Self {
        Self::wrap(buf.as_bytes(), pos, true, Decoder::utf8())
    }

    /// Changes whether attributes are checked for uniqueness (`true` by default).
    pub fn with_checks(&mut self, val: bool) -> &mut Attributes<'a> {
        self.state.check_duplicates = val;
        self
    }

    /// Checks whether the current tag has an [`xsi:nil`] attribute set to a truthy
    /// value, resolving prefixes with `resolver`. Ignores attribute errors.
    ///
    /// ```
    /// # use oxixml_xml::events::BytesStart;
    /// # use oxixml_xml::name::{Namespace, NamespaceResolver, PrefixDeclaration};
    /// let mut resolver = NamespaceResolver::default();
    /// resolver.add(
    ///     PrefixDeclaration::Named(b"xsi"),
    ///     Namespace(b"http://www.w3.org/2001/XMLSchema-instance"),
    /// ).expect("doc example");
    ///
    /// let tag = BytesStart::from_content("e xsi:nil='true'", 1);
    /// assert!(tag.attributes().has_nil(&resolver));
    ///
    /// let tag = BytesStart::from_content("e xsi:nil='false'", 1);
    /// assert!(!tag.attributes().has_nil(&resolver));
    /// ```
    ///
    /// [`xsi:nil`]: https://www.w3.org/TR/xmlschema-1/#xsi_nil
    pub fn has_nil(&mut self, resolver: &NamespaceResolver) -> bool {
        use crate::name::ResolveResult::*;

        self.any(|attr| {
            if let Ok(attr) = attr {
                match resolver.resolve_attribute(attr.key) {
                    (
                        Bound(Namespace(b"http://www.w3.org/2001/XMLSchema-instance")),
                        LocalName(b"nil"),
                    ) => attr.as_bool().unwrap_or_default(),
                    _ => false,
                }
            } else {
                false
            }
        })
    }

    /// Gets the decoder used to decode the bytes of the reader that produced this
    /// iterator. For manually-created iterators the encoding is UTF-8.
    #[inline]
    pub const fn decoder(&self) -> Decoder {
        self.decoder
    }
}

impl<'a> Debug for Attributes<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("Attributes")
            .field("bytes", &Bytes(self.bytes))
            .field("state", &self.state)
            .field("decoder", &self.decoder)
            .finish()
    }
}

impl<'a> Iterator for Attributes<'a> {
    type Item = Result<Attribute<'a>, AttrError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.state.next(self.bytes) {
            None => None,
            Some(Ok(a)) => Some(Ok(a.map(|range| &self.bytes[range]).into())),
            Some(Err(e)) => Some(Err(e)),
        }
    }
}

impl<'a> FusedIterator for Attributes<'a> {}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Errors that can be raised while parsing attributes.
///
/// The recovery positions in the examples show where parsing of the next
/// attribute will be attempted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttrError {
    /// The attribute key was not followed by `=`. Contains the position relative
    /// to the start of the owning tag. Only raised in XML mode.
    ExpectedEq(usize),
    /// No attribute value was found after `=`. Contains the position relative to
    /// the start of the owning tag.
    ExpectedValue(usize),
    /// An attribute value was not quoted. Contains the position relative to the
    /// start of the owning tag. Only raised in XML mode.
    UnquotedValue(usize),
    /// An attribute value was not terminated with a matching quote. Contains the
    /// position and the expected quote byte.
    ExpectedQuote(usize, u8),
    /// An attribute with the same name was already encountered. Contains the
    /// position of the new attribute and the position of the previous one.
    Duplicated(usize, usize),
}

impl Display for AttrError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::ExpectedEq(pos) => write!(
                f,
                r#"position {}: attribute key must be directly followed by `=` or space"#,
                pos
            ),
            Self::ExpectedValue(pos) => write!(
                f,
                r#"position {}: `=` must be followed by an attribute value"#,
                pos
            ),
            Self::UnquotedValue(pos) => write!(
                f,
                r#"position {}: attribute value must be enclosed in `"` or `'`"#,
                pos
            ),
            Self::ExpectedQuote(pos, quote) => write!(
                f,
                r#"position {}: missing closing quote `{}` in attribute value"#,
                pos, *quote as char
            ),
            Self::Duplicated(pos1, pos2) => write!(
                f,
                r#"position {}: duplicated attribute, previous declaration at position {}"#,
                pos1, pos2
            ),
        }
    }
}

impl std::error::Error for AttrError {}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A key/value XML or HTML [attribute] span.
///
/// [attribute]: https://www.w3.org/TR/xml11/#NT-Attribute
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Attr<T> {
    /// Attribute with a value enclosed in double quotes (`"`). Canonical XML style.
    DoubleQ(T, T),
    /// Attribute with a value enclosed in single quotes (`'`). XML style.
    SingleQ(T, T),
    /// Attribute with a value not enclosed in quotes. HTML style; only returned in
    /// HTML mode.
    Unquoted(T, T),
    /// Attribute without a value. HTML style; only returned in HTML mode.
    Empty(T),
}

impl<T> Attr<T> {
    /// Maps an `Attr<T>` to `Attr<U>` by applying `f` to the key and value.
    #[inline]
    pub fn map<U, F>(self, mut f: F) -> Attr<U>
    where
        F: FnMut(T) -> U,
    {
        match self {
            Attr::DoubleQ(key, value) => Attr::DoubleQ(f(key), f(value)),
            Attr::SingleQ(key, value) => Attr::SingleQ(f(key), f(value)),
            Attr::Empty(key) => Attr::Empty(f(key)),
            Attr::Unquoted(key, value) => Attr::Unquoted(f(key), f(value)),
        }
    }
}

impl<'a> Attr<&'a [u8]> {
    /// Returns the key.
    #[inline]
    pub const fn key(&self) -> QName<'a> {
        QName(match self {
            Attr::DoubleQ(key, _) => key,
            Attr::SingleQ(key, _) => key,
            Attr::Empty(key) => key,
            Attr::Unquoted(key, _) => key,
        })
    }
    /// Returns the value. For [`Attr::Empty`] an empty slice is returned.
    #[inline]
    pub const fn value(&self) -> &'a [u8] {
        match self {
            Attr::DoubleQ(_, value) => value,
            Attr::SingleQ(_, value) => value,
            Attr::Empty(_) => &[],
            Attr::Unquoted(_, value) => value,
        }
    }
}

impl<T: AsRef<[u8]>> Debug for Attr<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Attr::DoubleQ(key, value) => f
                .debug_tuple("Attr::DoubleQ")
                .field(&Bytes(key.as_ref()))
                .field(&Bytes(value.as_ref()))
                .finish(),
            Attr::SingleQ(key, value) => f
                .debug_tuple("Attr::SingleQ")
                .field(&Bytes(key.as_ref()))
                .field(&Bytes(value.as_ref()))
                .finish(),
            Attr::Empty(key) => f
                .debug_tuple("Attr::Empty")
                .field(&Bytes(key.as_ref()))
                .finish(),
            Attr::Unquoted(key, value) => f
                .debug_tuple("Attr::Unquoted")
                .field(&Bytes(key.as_ref()))
                .field(&Bytes(value.as_ref()))
                .finish(),
        }
    }
}

/// Unpacks the attribute key and value into a tuple. The value is `None` only for
/// [`Attr::Empty`].
impl<T> From<Attr<T>> for (T, Option<T>) {
    #[inline]
    fn from(attr: Attr<T>) -> Self {
        match attr {
            Attr::DoubleQ(key, value) => (key, Some(value)),
            Attr::SingleQ(key, value) => (key, Some(value)),
            Attr::Empty(key) => (key, None),
            Attr::Unquoted(key, value) => (key, Some(value)),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

type AttrResult = Result<Attr<Range<usize>>, AttrError>;

#[derive(Clone, Copy, Debug)]
enum State {
    /// Iteration finished.
    Done,
    /// The last attribute was parsed successfully; contains the offset from which
    /// the next attribute should be searched.
    Next(usize),
    /// The last attribute returned [`AttrError::UnquotedValue`]; the offset points
    /// to the beginning of the value.
    SkipValue(usize),
    /// The last attribute returned [`AttrError::Duplicated`]; the offset points to
    /// the equals (`=`) sign.
    SkipEqValue(usize),
}

/// Number of attributes a start tag may have before the duplicate-name check
/// switches from a linear scan to a hash pre-filter, keeping the whole tag O(N).
const SMALL_ATTRIBUTE_COUNT: usize = 32;

/// A no-op [`Hasher`] for the `key_hashes` set, whose values are already 64-bit
/// hashes of attribute names.
#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, _: &[u8]) {
        // The set only ever stores `u64` keys, which route through `write_u64`.
        unreachable!("IdentityHasher only supports u64 keys")
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

/// Hashes a single attribute name with a fresh [`DefaultHasher`] (DoS-resistant).
#[inline]
fn hash_name(name: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(name);
    hasher.finish()
}

/// External iterator over the spans of attribute keys and values.
#[derive(Clone, Debug)]
pub(crate) struct IterState {
    /// State that determines what should be done before parsing the next attribute.
    state: State,
    /// If `true`, allows parsing of unquoted values and key-only attributes.
    html: bool,
    /// If `true`, checks for duplicate names.
    check_duplicates: bool,
    /// Ranges of already-parsed attribute names (only if `check_duplicates`).
    keys: Vec<Range<usize>>,
    /// 64-bit hashes of `keys`, used as an O(1) pre-filter once a start tag
    /// declares more than `SMALL_ATTRIBUTE_COUNT` attributes.
    key_hashes: Option<HashSet<u64, BuildHasherDefault<IdentityHasher>>>,
}

impl IterState {
    pub const fn new(offset: usize, html: bool) -> Self {
        Self {
            state: State::Next(offset),
            html,
            check_duplicates: true,
            keys: Vec::new(),
            key_hashes: None,
        }
    }

    /// Recovers from an error made on a previous step, returning the offset from
    /// which parsing should continue, or `None` if there is no input left.
    fn recover(&self, slice: &[u8]) -> Option<usize> {
        match self.state {
            State::Done => None,
            State::Next(offset) => Some(offset),
            State::SkipValue(offset) => self.skip_value(slice, offset),
            State::SkipEqValue(offset) => self.skip_eq_value(slice, offset),
        }
    }

    /// Skips all characters up to the first white space or end-of-input.
    #[inline]
    fn skip_value(&self, slice: &[u8], offset: usize) -> Option<usize> {
        let mut iter = (offset..).zip(slice[offset..].iter());
        iter.find(|&(_, &b)| is_whitespace(b)).map(|(e, _)| e)
    }

    /// Skips an `= value` up to the first white space or end-of-input.
    #[inline]
    fn skip_eq_value(&self, slice: &[u8], offset: usize) -> Option<usize> {
        let mut iter = (offset..).zip(slice[offset..].iter());

        // Skip up to the quote and determine its type.
        let quote = match iter.find(|&(_, &b)| !is_whitespace(b)) {
            Some((_, b'"')) => b'"',
            Some((_, b'\'')) => b'\'',
            Some((offset, _)) => return self.skip_value(slice, offset),
            None => return None,
        };

        match iter.find(|&(_, &b)| b == quote) {
            Some((e, b'"')) => Some(e),
            Some((e, _)) => Some(e),
            None => None,
        }
    }

    /// Checks that the attribute name `key` was not seen earlier in the same tag,
    /// recording it. Uses a linear scan for small tags and a hash pre-filter above
    /// `SMALL_ATTRIBUTE_COUNT`.
    #[inline]
    fn check_for_duplicates(
        &mut self,
        slice: &[u8],
        key: Range<usize>,
    ) -> Result<Range<usize>, AttrError> {
        if self.check_duplicates {
            if self.keys.len() >= SMALL_ATTRIBUTE_COUNT {
                return self.check_for_duplicates_hashed(slice, key);
            }
            if let Some(prev) = self
                .keys
                .iter()
                .find(|r| slice[(*r).clone()] == slice[key.clone()])
            {
                return Err(AttrError::Duplicated(key.start, prev.start));
            }
            self.keys.push(key.clone());
        }
        Ok(key)
    }

    /// Cold path of [`check_for_duplicates`](Self::check_for_duplicates) for tags
    /// with many attributes: an O(1) hash pre-filter keeps iteration O(N).
    #[cold]
    fn check_for_duplicates_hashed(
        &mut self,
        slice: &[u8],
        key: Range<usize>,
    ) -> Result<Range<usize>, AttrError> {
        let keys = &self.keys;
        let key_hashes = self.key_hashes.get_or_insert_with(|| {
            let mut set = HashSet::with_capacity_and_hasher(
                keys.len() * 2,
                BuildHasherDefault::<IdentityHasher>::default(),
            );
            for r in keys {
                set.insert(hash_name(&slice[r.clone()]));
            }
            set
        });
        if !key_hashes.insert(hash_name(&slice[key.clone()])) {
            if let Some(prev) = self
                .keys
                .iter()
                .find(|r| slice[(*r).clone()] == slice[key.clone()])
            {
                return Err(AttrError::Duplicated(key.start, prev.start));
            }
        }
        self.keys.push(key.clone());
        Ok(key)
    }

    #[inline]
    fn key_only(&mut self, slice: &[u8], key: Range<usize>, offset: usize) -> Option<AttrResult> {
        Some(if self.html {
            self.check_for_duplicates(slice, key).map(Attr::Empty)
        } else {
            Err(AttrError::ExpectedEq(offset))
        })
    }

    #[inline]
    fn double_q(&mut self, key: Range<usize>, value: Range<usize>) -> Option<AttrResult> {
        self.state = State::Next(value.end + 1); // +1 for `"`
        Some(Ok(Attr::DoubleQ(key, value)))
    }

    #[inline]
    fn single_q(&mut self, key: Range<usize>, value: Range<usize>) -> Option<AttrResult> {
        self.state = State::Next(value.end + 1); // +1 for `'`
        Some(Ok(Attr::SingleQ(key, value)))
    }

    pub fn next(&mut self, slice: &[u8]) -> Option<AttrResult> {
        let mut iter = match self.recover(slice) {
            Some(offset) => (offset..).zip(slice[offset..].iter()),
            None => return None,
        };

        // Index where the next key starts.
        let start_key = match iter.find(|&(_, &b)| !is_whitespace(b)) {
            Some((s, _)) => s,
            None => {
                self.state = State::Done;
                return None;
            }
        };
        // Span of the key.
        let (key, offset) = match iter.find(|&(_, &b)| b == b'=' || is_whitespace(b)) {
            Some((e, b'=')) => (start_key..e, e),

            Some((e, _)) => match iter.find(|&(_, &b)| !is_whitespace(b)) {
                Some((offset, b'=')) => (start_key..e, offset),
                Some((offset, _)) => {
                    self.state = State::Next(offset);
                    return self.key_only(slice, start_key..e, offset);
                }
                None => {
                    self.state = State::Done;
                    return self.key_only(slice, start_key..e, slice.len());
                }
            },

            None => {
                self.state = State::Done;
                let e = slice.len();
                return self.key_only(slice, start_key..e, e);
            }
        };

        let key = match self.check_for_duplicates(slice, key) {
            Err(e) => {
                self.state = State::SkipEqValue(offset);
                return Some(Err(e));
            }
            Ok(key) => key,
        };

        // Position of the quote and its type.
        let (start_value, quote) = match iter.find(|&(_, &b)| !is_whitespace(b)) {
            Some((s, b'"')) => (s + 1, b'"'),
            Some((s, b'\'')) => (s + 1, b'\''),

            Some((s, _)) if self.html => {
                let end = match iter.find(|&(_, &b)| is_whitespace(b)) {
                    Some((e, _)) => e,
                    None => slice.len(),
                };
                self.state = State::Next(end);
                return Some(Ok(Attr::Unquoted(key, s..end)));
            }
            Some((s, _)) => {
                self.state = State::SkipValue(s);
                return Some(Err(AttrError::UnquotedValue(s)));
            }

            None => {
                self.state = State::Done;
                return Some(Err(AttrError::ExpectedValue(slice.len())));
            }
        };

        match iter.find(|&(_, &b)| b == quote) {
            Some((e, b'"')) => self.double_q(key, start_value..e),
            Some((e, _)) => self.single_q(key, start_value..e),
            None => {
                self.state = State::Done;
                Some(Err(AttrError::ExpectedQuote(slice.len(), quote)))
            }
        }
    }
}

#[cfg(test)]
mod tests;
