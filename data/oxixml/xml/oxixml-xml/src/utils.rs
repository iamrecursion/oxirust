//! Internal helpers shared across the crate.
//!
//! Not an official API, exposed only for integration tests and mirrored from
//! quick-xml 0.41 `utils` module.

use std::borrow::{Borrow, Cow};
use std::fmt::{self, Debug, Formatter};
use std::iter::FusedIterator;
use std::ops::Deref;

use oxixml_support::bytes::{memchr, memchr3};

#[cfg(feature = "serialize")]
use serde::de::{Deserialize, Deserializer, Error, Visitor};
#[cfg(feature = "serialize")]
use serde::ser::{Serialize, Serializer};

/// Writes a `Cow<[u8]>` to the formatter, tagging it as `Owned` or `Borrowed`
/// and rendering the bytes with [`write_byte_string`].
#[allow(clippy::ptr_arg)]
pub fn write_cow_string(f: &mut Formatter, cow_string: &Cow<[u8]>) -> fmt::Result {
    match cow_string {
        Cow::Owned(s) => {
            write!(f, "Owned(")?;
            write_byte_string(f, s)?;
        }
        Cow::Borrowed(s) => {
            write!(f, "Borrowed(")?;
            write_byte_string(f, s)?;
        }
    }
    write!(f, ")")
}

/// Writes a byte slice to the formatter as a human-readable string: printable
/// ASCII bytes are written as-is, everything else is written in hexadecimal.
pub fn write_byte_string(f: &mut Formatter, byte_string: &[u8]) -> fmt::Result {
    write!(f, "\"")?;
    for b in byte_string {
        match *b {
            32..=33 | 35..=126 => write!(f, "{}", *b as char)?,
            34 => write!(f, "\\\"")?,
            _ => write!(f, "{:#02X}", b)?,
        }
    }
    write!(f, "\"")?;
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A version of [`Cow`] that can borrow from two different buffers, one of them
/// being a deserializer input.
///
/// # Lifetimes
///
/// - `'i`: lifetime of the data that the deserializer borrows from the parsed input
/// - `'s`: lifetime of the data that is owned by a deserializer
pub enum CowRef<'i, 's, B>
where
    B: ToOwned + ?Sized,
{
    /// An input borrowed from the parsed data.
    Input(&'i B),
    /// An input borrowed from the buffer owned by another deserializer.
    Slice(&'s B),
    /// An input taken from an external deserializer, owned by that deserializer.
    Owned(<B as ToOwned>::Owned),
}

impl<B> Deref for CowRef<'_, '_, B>
where
    B: ToOwned + ?Sized,
    B::Owned: Borrow<B>,
{
    type Target = B;

    fn deref(&self) -> &B {
        match *self {
            Self::Input(borrowed) => borrowed,
            Self::Slice(borrowed) => borrowed,
            Self::Owned(ref owned) => owned.borrow(),
        }
    }
}

impl<B> Debug for CowRef<'_, '_, B>
where
    B: ToOwned + ?Sized + Debug,
    B::Owned: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            Self::Input(borrowed) => Debug::fmt(borrowed, f),
            Self::Slice(borrowed) => Debug::fmt(borrowed, f),
            Self::Owned(ref owned) => Debug::fmt(owned, f),
        }
    }
}

impl<'i> CowRef<'i, '_, str> {
    /// Supplies to the visitor a borrowed string, a string slice, or an owned
    /// string depending on the kind of input.
    ///
    /// Calls
    /// - `visitor.visit_borrowed_str` if the data is borrowed from the input
    /// - `visitor.visit_str` if the data is borrowed from another source
    /// - `visitor.visit_string` if the data is owned by this type
    #[cfg(feature = "serialize")]
    pub fn deserialize_str<V, E>(self, visitor: V) -> Result<V::Value, E>
    where
        V: Visitor<'i>,
        E: Error,
    {
        match self {
            Self::Input(s) => visitor.visit_borrowed_str(s),
            Self::Slice(s) => visitor.visit_str(s),
            Self::Owned(s) => visitor.visit_string(s),
        }
    }

    /// Calls [`Visitor::visit_bool`] with `true` or `false` if the text contains a
    /// [valid] boolean representation, otherwise calls [`Self::deserialize_str`].
    ///
    /// The valid boolean representations are only `"true"`, `"false"`, `"1"` and `"0"`.
    ///
    /// [valid]: https://www.w3.org/TR/xmlschema11-2/#boolean
    #[cfg(feature = "serialize")]
    pub fn deserialize_bool<V, E>(self, visitor: V) -> Result<V::Value, E>
    where
        V: Visitor<'i>,
        E: Error,
    {
        match self.as_ref() {
            "1" | "true" => visitor.visit_bool(true),
            "0" | "false" => visitor.visit_bool(false),
            _ => self.deserialize_str(visitor),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around `Vec<u8>` that has a human-readable debug representation:
/// printable ASCII symbols are output as-is, all other output in hexadecimal.
///
/// Also, when the `serialize` feature is on, this type is deserialized using
#[cfg_attr(
    feature = "serialize",
    doc = "[`deserialize_byte_buf`](serde::Deserializer::deserialize_byte_buf) instead of the \
vector's generic [`deserialize_seq`](serde::Deserializer::deserialize_seq)."
)]
#[cfg_attr(
    not(feature = "serialize"),
    doc = "`Deserializer::deserialize_byte_buf` instead of the vector's generic \
`Deserializer::deserialize_seq`."
)]
#[derive(PartialEq, Eq)]
pub struct ByteBuf(pub Vec<u8>);

impl Debug for ByteBuf {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write_byte_string(f, &self.0)
    }
}

#[cfg(feature = "serialize")]
impl<'de> Deserialize<'de> for ByteBuf {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ValueVisitor;

        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = ByteBuf;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("byte data")
            }

            fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(ByteBuf(v.to_vec()))
            }

            fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(ByteBuf(v))
            }
        }

        d.deserialize_byte_buf(ValueVisitor)
    }
}

#[cfg(feature = "serialize")]
impl Serialize for ByteBuf {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around `&[u8]` that has a human-readable debug representation:
/// printable ASCII symbols are output as-is, all other output in hexadecimal.
///
/// Also, when the `serialize` feature is on, this type is deserialized using
#[cfg_attr(
    feature = "serialize",
    doc = "[`deserialize_bytes`](serde::Deserializer::deserialize_bytes) instead of the \
vector's generic [`deserialize_seq`](serde::Deserializer::deserialize_seq)."
)]
#[cfg_attr(
    not(feature = "serialize"),
    doc = "`Deserializer::deserialize_bytes` instead of the vector's generic \
`Deserializer::deserialize_seq`."
)]
#[derive(PartialEq, Eq)]
pub struct Bytes<'de>(pub &'de [u8]);

impl<'de> Debug for Bytes<'de> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write_byte_string(f, self.0)
    }
}

#[cfg(feature = "serialize")]
impl<'de> Deserialize<'de> for Bytes<'de> {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ValueVisitor;

        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = Bytes<'de>;

            fn expecting(&self, f: &mut Formatter) -> fmt::Result {
                f.write_str("borrowed bytes")
            }

            fn visit_borrowed_bytes<E: Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
                Ok(Bytes(v))
            }
        }

        d.deserialize_bytes(ValueVisitor)
    }
}

#[cfg(feature = "serialize")]
impl<'de> Serialize for Bytes<'de> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(self.0)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns `true` if the byte is XML white space (space, tab, carriage return or
/// line feed).
#[inline]
pub const fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\r' | b'\n' | b'\t')
}

/// Iterator over the byte offsets of every occurrence of a needle byte in a
/// haystack, mirroring `memchr::memchr_iter` used by quick-xml 0.41. Built on
/// top of the Pure-Rust [`memchr`](oxixml_support::bytes::memchr).
#[derive(Debug)]
pub(crate) struct MemchrIter<'a> {
    haystack: &'a [u8],
    offset: usize,
    needles: [u8; 3],
    count: u8,
}

impl<'a> MemchrIter<'a> {
    /// Iterates over positions of `needle` in `haystack`.
    #[inline]
    pub(crate) const fn one(needle: u8, haystack: &'a [u8]) -> Self {
        Self {
            haystack,
            offset: 0,
            needles: [needle, 0, 0],
            count: 1,
        }
    }

    /// Iterates over positions of any of the three needles in `haystack`.
    #[inline]
    pub(crate) const fn three(n1: u8, n2: u8, n3: u8, haystack: &'a [u8]) -> Self {
        Self {
            haystack,
            offset: 0,
            needles: [n1, n2, n3],
            count: 3,
        }
    }
}

impl Iterator for MemchrIter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let rest = &self.haystack[self.offset..];
        let found = if self.count == 1 {
            memchr(self.needles[0], rest)
        } else {
            memchr3(self.needles[0], self.needles[1], self.needles[2], rest)
        };
        found.map(|i| {
            let absolute = self.offset + i;
            self.offset = absolute + 1;
            absolute
        })
    }
}

impl FusedIterator for MemchrIter<'_> {}

/// Returns an iterator over the byte offsets of every occurrence of `needle`.
///
/// Equivalent to `memchr::memchr_iter` from quick-xml 0.41.
#[inline]
pub(crate) fn memchr_iter(needle: u8, haystack: &[u8]) -> MemchrIter<'_> {
    MemchrIter::one(needle, haystack)
}

/// Returns an iterator over the byte offsets of every occurrence of any of the
/// three needles.
///
/// Equivalent to `memchr::memchr3_iter` from quick-xml 0.41.
#[inline]
pub(crate) fn memchr3_iter(n1: u8, n2: u8, n3: u8, haystack: &[u8]) -> MemchrIter<'_> {
    MemchrIter::three(n1, n2, n3, haystack)
}

/// Calculates the name length from element-like content. The name is the first
/// word in `bytes`, where the word boundary is any XML white space character.
///
/// 'Whitespace' refers to the definition used by [`is_whitespace`].
#[inline]
pub const fn name_len(mut bytes: &[u8]) -> usize {
    // A pattern-matching based approach (instead of indexing) allows making the
    // function `const`.
    let mut len = 0;
    while let [first, rest @ ..] = bytes {
        if is_whitespace(*first) {
            break;
        }
        len += 1;
        bytes = rest;
    }
    len
}

/// Returns a byte slice with leading XML white space bytes removed.
///
/// 'Whitespace' refers to the definition used by [`is_whitespace`].
#[inline]
pub const fn trim_xml_start(mut bytes: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = bytes {
        if is_whitespace(*first) {
            bytes = rest;
        } else {
            break;
        }
    }
    bytes
}

/// Returns a byte slice with trailing XML white space bytes removed.
///
/// 'Whitespace' refers to the definition used by [`is_whitespace`].
#[inline]
pub const fn trim_xml_end(mut bytes: &[u8]) -> &[u8] {
    while let [rest @ .., last] = bytes {
        if is_whitespace(*last) {
            bytes = rest;
        } else {
            break;
        }
    }
    bytes
}

/// Returns a string slice with XML white space characters removed from both sides.
///
/// 'Whitespace' refers to the definition used by [`is_whitespace`].
#[inline]
pub fn trim_xml_spaces(text: &str) -> &str {
    let bytes = trim_xml_end(trim_xml_start(text.as_bytes()));
    // Removing XML space characters (a subset of ASCII) from a `&str` never
    // invalidates the UTF-8 encoding, so this always succeeds.
    core::str::from_utf8(bytes).unwrap_or_default()
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Splits a string into pieces that can each be part of a single `CDATA` section.
///
/// Because a `CDATA` section cannot contain the `]]>` sequence, the string is
/// split between `]]` and `>`.
#[derive(Debug, Clone)]
pub(crate) struct CDataIterator<'a> {
    /// The unprocessed data which should be emitted as `CDATA` pieces. At each
    /// iteration, the processed data is cut from this slice.
    unprocessed: &'a str,
    finished: bool,
}

impl<'a> CDataIterator<'a> {
    /// Creates a new iterator over the `CDATA`-safe pieces of `value`.
    pub const fn new(value: &'a str) -> Self {
        Self {
            unprocessed: value,
            finished: false,
        }
    }
}

impl<'a> Iterator for CDataIterator<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.finished {
            return None;
        }

        let bytes = self.unprocessed.as_bytes();
        let mut base = 0;
        while let Some(rel) = memchr(b'>', &bytes[base..]) {
            let gt = base + rel;
            let (slice, rest) = self.unprocessed.split_at(gt);
            if slice.ends_with("]]") {
                self.unprocessed = rest;
                return Some(slice);
            }
            base = gt + 1;
        }

        self.finished = true;
        Some(self.unprocessed)
    }
}

impl FusedIterator for CDataIterator<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_byte_string_hex_and_ascii() {
        let bytes = ByteBuf(vec![10, 32, 32, 32, 32, 32, 32, 32, 32]);
        assert_eq!(format!("{:?}", bytes), "\"0xA        \"");
    }

    #[test]
    fn write_byte_string_url() {
        let bytes = ByteBuf(b"http://www.w3.org/2002/07/owl#".to_vec());
        assert_eq!(
            format!("{:?}", bytes),
            r##""http://www.w3.org/2002/07/owl#""##
        );
    }

    #[test]
    fn write_byte_string_quote() {
        let bytes = ByteBuf(b"Class IRI=\"#B\"".to_vec());
        assert_eq!(format!("{:?}", bytes), r##""Class IRI=\"#B\"""##);
    }

    #[test]
    fn name_len_boundaries() {
        assert_eq!(super::name_len(b""), 0);
        assert_eq!(super::name_len(b" abc"), 0);
        assert_eq!(super::name_len(b" \t\r\n"), 0);

        assert_eq!(super::name_len(b"abc"), 3);
        assert_eq!(super::name_len(b"abc "), 3);

        assert_eq!(super::name_len(b"a bc"), 1);
        assert_eq!(super::name_len(b"ab\tc"), 2);
        assert_eq!(super::name_len(b"ab\rc"), 2);
        assert_eq!(super::name_len(b"ab\nc"), 2);
    }

    #[test]
    fn trim_start() {
        assert_eq!(Bytes(super::trim_xml_start(b"")), Bytes(b""));
        assert_eq!(Bytes(super::trim_xml_start(b"abc")), Bytes(b"abc"));
        assert_eq!(
            Bytes(super::trim_xml_start(b"\r\n\t ab \t\r\nc \t\r\n")),
            Bytes(b"ab \t\r\nc \t\r\n")
        );
    }

    #[test]
    fn trim_end() {
        assert_eq!(Bytes(super::trim_xml_end(b"")), Bytes(b""));
        assert_eq!(Bytes(super::trim_xml_end(b"abc")), Bytes(b"abc"));
        assert_eq!(
            Bytes(super::trim_xml_end(b"\r\n\t ab \t\r\nc \t\r\n")),
            Bytes(b"\r\n\t ab \t\r\nc")
        );
    }

    #[test]
    fn cdata_split() {
        let pieces: Vec<_> = CDataIterator::new("foo]]>bar]]>baz").collect();
        assert_eq!(pieces, vec!["foo]]", ">bar]]", ">baz"]);
        let pieces: Vec<_> = CDataIterator::new("no marker").collect();
        assert_eq!(pieces, vec!["no marker"]);
    }
}
