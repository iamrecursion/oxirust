//! X.500 distinguished names, and what it means for two of them to be the same.
//!
//! Chaining a certificate to its issuer is a *name* comparison: the issuer
//! field of the child has to name the subject of the parent. Getting that
//! comparison wrong in either direction is a real failure — too strict and
//! ordinary chains stop building, too loose and one certificate authority can
//! be made to stand in for another.
//!
//! [RFC 5280 §7.1](https://www.rfc-editor.org/rfc/rfc5280#section-7.1) asks for
//! the LDAP matching rules, which in full means the string preparation of
//! [RFC 4518](https://www.rfc-editor.org/rfc/rfc4518) — transcoding, case
//! folding, Unicode normalisation and insignificant-space handling. Two levels
//! of that are implemented here:
//!
//! 1. **the encodings are identical.** This is the case in practice: a
//!    conforming authority copies the subject of the issuing certificate into
//!    the issuer field of the one it issues, octet for octet
//!    ([RFC 5280 §4.1.2.4](https://www.rfc-editor.org/rfc/rfc5280#section-4.1.2.4)).
//! 2. **the names are structurally the same** — same sequence of relative
//!    names, same attribute types, and values that agree after decoding to
//!    text, folding case and collapsing insignificant whitespace. This is what
//!    rescues a chain whose two halves were produced by tools that disagreed
//!    about `PrintableString` versus `UTF8String`.
//!
//! Unicode normalisation (NFKC) is *not* performed: it needs tables this
//! workspace does not carry, and a name that differs only by a compatibility
//! composition is not something a certificate authority emits. Two names that
//! would match only under NFKC are treated as different, which is the
//! fail-closed direction.

use std::fmt;

use crate::crypto::der::{Der, DerError, tag};
use crate::x509::error::CertificateError;
use crate::x509::oid;

/// A parsed X.500 `Name`, as it appears in a certificate's `issuer` and
/// `subject` fields.
///
/// [`PartialEq`] is encoding equality; [`matches`](DistinguishedName::matches)
/// is the relation certificate chaining actually uses. [`Display`](fmt::Display)
/// writes the compact
/// [RFC 4514](https://www.rfc-editor.org/rfc/rfc4514) form that error messages
/// name a certificate by.
#[derive(Debug, Clone)]
pub struct DistinguishedName {
    encoding: Vec<u8>,
    relative_names: Vec<Vec<Attribute>>,
}

/// One `AttributeTypeAndValue` of a relative distinguished name.
#[derive(Debug, Clone)]
struct Attribute {
    /// The contents of the type `OBJECT IDENTIFIER`.
    oid: Vec<u8>,
    /// The whole DER encoding of the value, tag and length included — what
    /// RFC 4514 hex-escapes for a value it cannot render as text.
    encoding: Vec<u8>,
    /// The value decoded as text, when it is one of the string types.
    text: Option<String>,
    /// The same text with case folded and insignificant whitespace collapsed.
    folded: Option<String>,
}

impl DistinguishedName {
    /// Parses a DER `Name`, which is the whole `SEQUENCE OF
    /// RelativeDistinguishedName` including its tag and length.
    ///
    /// # Errors
    ///
    /// [`CertificateError::Malformed`] if the encoding is not a `Name`.
    pub(crate) fn parse(encoding: &[u8]) -> Result<Self, CertificateError> {
        Self::read(encoding)
            .map_err(|error| CertificateError::Malformed(format!("X.500 Name: {error}")))
    }

    fn read(encoding: &[u8]) -> Result<Self, DerError> {
        let mut outer = Der::new(encoding);
        let mut sequence = outer.read_sequence()?;
        outer.finish()?;
        let mut relative_names = Vec::new();
        while !sequence.is_empty() {
            let mut set = Der::new(sequence.read(tag::SET)?);
            let mut attributes = Vec::new();
            while !set.is_empty() {
                let mut pair = set.read_sequence()?;
                let oid = pair.read(tag::OID)?.to_vec();
                let (value_tag, whole, contents) = pair.read_any()?;
                pair.finish()?;
                let text = decode_directory_string(value_tag, contents);
                attributes.push(Attribute {
                    oid,
                    encoding: whole.to_vec(),
                    folded: text.as_deref().map(fold),
                    text,
                });
            }
            relative_names.push(attributes);
        }
        Ok(Self {
            encoding: encoding.to_vec(),
            relative_names,
        })
    }

    /// The DER encoding the name was read from.
    #[must_use]
    pub fn encoding(&self) -> &[u8] {
        &self.encoding
    }

    /// Whether this name and `other` denote the same directory entry, under the
    /// two levels of comparison the module documentation describes.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        if self.encoding == other.encoding {
            return true;
        }
        if self.relative_names.len() != other.relative_names.len() {
            return false;
        }
        self.relative_names
            .iter()
            .zip(&other.relative_names)
            .all(|(left, right)| relative_names_match(left, right))
    }

    /// Whether the name has no relative names at all — the empty `Name` that a
    /// certificate is only allowed to use for a subject it identifies some
    /// other way.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.relative_names.is_empty()
    }
}

impl PartialEq for DistinguishedName {
    /// Encoding equality. Use [`matches`](DistinguishedName::matches) to ask
    /// whether two names denote the same entry.
    fn eq(&self, other: &Self) -> bool {
        self.encoding == other.encoding
    }
}

impl Eq for DistinguishedName {}

impl fmt::Display for DistinguishedName {
    /// Writes the name in the string form of
    /// [RFC 4514 §2](https://www.rfc-editor.org/rfc/rfc4514#section-2): the
    /// relative names most specific first, each attribute as `type=value`, and
    /// a value that is not text as `#` followed by the hexadecimal of its
    /// encoding.
    ///
    /// Attribute types outside the short-name table are written as dotted
    /// decimal. This is a rendering for people reading an error message, not a
    /// canonical form; nothing compares these strings.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.relative_names.is_empty() {
            return f.write_str("<empty>");
        }
        for (index, relative_name) in self.relative_names.iter().rev().enumerate() {
            if index > 0 {
                f.write_str(",")?;
            }
            for (position, attribute) in relative_name.iter().enumerate() {
                if position > 0 {
                    f.write_str("+")?;
                }
                match oid::attribute_short_name(&attribute.oid) {
                    Some(name) => f.write_str(name)?,
                    None => f.write_str(&oid::dotted(&attribute.oid))?,
                }
                f.write_str("=")?;
                match &attribute.text {
                    Some(text) => f.write_str(&escape(text))?,
                    None => {
                        f.write_str("#")?;
                        for byte in &attribute.encoding {
                            write!(f, "{byte:02x}")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Whether two relative distinguished names hold the same set of attributes.
///
/// A relative name is a `SET`, so its members are unordered even though DER
/// fixes an encoding order; the comparison is therefore by set membership
/// rather than position.
fn relative_names_match(left: &[Attribute], right: &[Attribute]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .all(|attribute| right.iter().any(|other| attributes_match(attribute, other)))
}

/// Whether two attributes have the same type and an equal value.
fn attributes_match(left: &Attribute, right: &Attribute) -> bool {
    if left.oid != right.oid {
        return false;
    }
    match (&left.folded, &right.folded) {
        (Some(left), Some(right)) => left == right,
        // A value neither side could read as text is compared as it was
        // encoded, which is the only honest thing left to do with it.
        (None, None) => left.encoding == right.encoding,
        _ => false,
    }
}

/// Decodes one of the `DirectoryString` alternatives, plus the two other string
/// types certificates use for `emailAddress` and `domainComponent`.
///
/// Returns [`None`] for a value that is not a string type at all, or for one
/// whose octets are not valid in the encoding its tag claims.
fn decode_directory_string(value_tag: u8, contents: &[u8]) -> Option<String> {
    match value_tag {
        // UTF8String.
        0x0c => String::from_utf8(contents.to_vec()).ok(),
        // PrintableString, NumericString, IA5String, VisibleString: all ASCII
        // subsets, and this crate does not need to police which subset.
        0x13 | 0x12 | 0x16 | 0x1a => contents
            .is_ascii()
            .then(|| contents.iter().map(|byte| char::from(*byte)).collect()),
        // TeletexString. Its real repertoire is a T.61 escape soup nobody
        // implements; every producer that still emits one means Latin-1, and
        // that is what is read here.
        0x14 => Some(contents.iter().map(|byte| char::from(*byte)).collect()),
        // BMPString: UTF-16 big-endian.
        0x1e => {
            if contents.len() % 2 != 0 {
                return None;
            }
            let units: Vec<u16> = contents
                .chunks_exact(2)
                .map(|pair| {
                    let mut octets = [0_u8; 2];
                    octets.copy_from_slice(pair);
                    u16::from_be_bytes(octets)
                })
                .collect();
            String::from_utf16(&units).ok()
        }
        // UniversalString: UTF-32 big-endian.
        0x1c => {
            if contents.len() % 4 != 0 {
                return None;
            }
            contents
                .chunks_exact(4)
                .map(|quad| {
                    let mut octets = [0_u8; 4];
                    octets.copy_from_slice(quad);
                    char::from_u32(u32::from_be_bytes(octets))
                })
                .collect::<Option<String>>()
        }
        _ => None,
    }
}

/// Folds a directory string for comparison: case folded, leading and trailing
/// whitespace removed, and every internal run of whitespace collapsed to one
/// space — the "insignificant space handling" of
/// [RFC 4518 §2.6.1](https://www.rfc-editor.org/rfc/rfc4518#section-2.6.1),
/// without the Unicode normalisation step.
fn fold(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for (index, word) in text.split_whitespace().enumerate() {
        if index > 0 {
            folded.push(' ');
        }
        folded.extend(word.chars().flat_map(char::to_lowercase));
    }
    folded
}

/// Escapes an attribute value for the RFC 4514 string form.
fn escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    let mut characters = value.chars().peekable();
    let mut position = 0_usize;
    while let Some(character) = characters.next() {
        let leading = position == 0;
        let trailing = characters.peek().is_none();
        match character {
            '"' | '+' | ',' | ';' | '<' | '>' | '\\' => {
                escaped.push('\\');
                escaped.push(character);
            }
            '\0' => escaped.push_str("\\00"),
            ' ' if leading || trailing => escaped.push_str("\\ "),
            '#' if leading => escaped.push_str("\\#"),
            _ => escaped.push(character),
        }
        position += 1;
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one DER TLV.
    fn tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        if contents.len() < 0x80 {
            out.push(u8::try_from(contents.len()).unwrap_or(0));
        } else {
            out.push(0x81);
            out.push(u8::try_from(contents.len()).unwrap_or(0xff));
        }
        out.extend_from_slice(contents);
        out
    }

    /// A single-attribute `Name`, with the attribute value in whichever string
    /// encoding the caller asks for.
    fn name(attribute_oid: &[u8], value_tag: u8, value: &[u8]) -> Vec<u8> {
        let mut pair = tlv(tag::OID, attribute_oid);
        pair.extend(tlv(value_tag, value));
        let set = tlv(tag::SET, &tlv(tag::SEQUENCE, &pair));
        tlv(tag::SEQUENCE, &set)
    }

    const COMMON_NAME: &[u8] = &[0x55, 0x04, 0x03];
    const ORGANISATION: &[u8] = &[0x55, 0x04, 0x0a];

    #[test]
    fn an_identical_encoding_matches() -> Result<(), CertificateError> {
        let encoding = name(COMMON_NAME, 0x13, b"OxiXML Test");
        let left = DistinguishedName::parse(&encoding)?;
        let right = DistinguishedName::parse(&encoding)?;
        assert_eq!(left, right);
        assert!(left.matches(&right));
        assert_eq!(left.to_string(), "CN=OxiXML Test");
        Ok(())
    }

    /// The case the second level of comparison exists for: the same name
    /// written by two tools that chose different string types, different case
    /// and different spacing.
    #[test]
    fn the_same_name_in_a_different_encoding_still_matches() -> Result<(), CertificateError> {
        let printable = DistinguishedName::parse(&name(COMMON_NAME, 0x13, b"OxiXML  Test "))?;
        let utf8 = DistinguishedName::parse(&name(COMMON_NAME, 0x0c, b"oxixml test"))?;
        assert_ne!(printable, utf8, "the encodings differ");
        assert!(printable.matches(&utf8));
        assert!(utf8.matches(&printable));
        Ok(())
    }

    #[test]
    fn a_different_name_does_not_match() -> Result<(), CertificateError> {
        let left = DistinguishedName::parse(&name(COMMON_NAME, 0x13, b"OxiXML Test"))?;
        let right = DistinguishedName::parse(&name(COMMON_NAME, 0x13, b"OxiXML Other"))?;
        assert!(!left.matches(&right));
        // The same text under a different attribute type is a different name.
        let other_type = DistinguishedName::parse(&name(ORGANISATION, 0x13, b"OxiXML Test"))?;
        assert!(!left.matches(&other_type));
        assert_eq!(other_type.to_string(), "O=OxiXML Test");
        Ok(())
    }

    /// A `BMPString` and a `UniversalString` holding the same characters must
    /// reach the same folded text, and a broken one must reach none at all.
    #[test]
    fn the_wide_string_types_decode() -> Result<(), CertificateError> {
        let bmp = DistinguishedName::parse(&name(COMMON_NAME, 0x1e, b"\x00A\x00b"))?;
        let universal =
            DistinguishedName::parse(&name(COMMON_NAME, 0x1c, b"\x00\x00\x00a\x00\x00\x00B"))?;
        assert!(bmp.matches(&universal));
        assert_eq!(bmp.to_string(), "CN=Ab");

        // An odd number of octets is not a BMPString; the value stays opaque
        // and compares by its encoding alone.
        let truncated = DistinguishedName::parse(&name(COMMON_NAME, 0x1e, b"\x00A\x00"))?;
        assert!(!truncated.matches(&bmp));
        assert!(truncated.to_string().starts_with("CN=#1e"), "{truncated}");
        Ok(())
    }

    /// Relative names hold a `SET`, so two encodings that list the same
    /// attributes in different orders name the same entry.
    #[test]
    fn a_multi_valued_relative_name_compares_as_a_set() -> Result<(), CertificateError> {
        let build = |first: (&[u8], &[u8]), second: (&[u8], &[u8])| {
            let mut members = Vec::new();
            for (oid, value) in [first, second] {
                let mut pair = tlv(tag::OID, oid);
                pair.extend(tlv(0x13, value));
                members.extend(tlv(tag::SEQUENCE, &pair));
            }
            tlv(tag::SEQUENCE, &tlv(tag::SET, &members))
        };
        let left =
            DistinguishedName::parse(&build((COMMON_NAME, b"alice"), (ORGANISATION, b"example")))?;
        let right =
            DistinguishedName::parse(&build((ORGANISATION, b"example"), (COMMON_NAME, b"alice")))?;
        assert!(left.matches(&right));
        Ok(())
    }

    /// RFC 4514 puts the most specific relative name first, which is the
    /// reverse of the encoding order.
    #[test]
    fn the_rendering_is_most_specific_first() -> Result<(), CertificateError> {
        let mut sequence = Vec::new();
        for (oid, value) in [
            (ORGANISATION, b"OxiXML".as_slice()),
            (COMMON_NAME, b"alice"),
        ] {
            let mut pair = tlv(tag::OID, oid);
            pair.extend(tlv(0x13, value));
            sequence.extend(tlv(tag::SET, &tlv(tag::SEQUENCE, &pair)));
        }
        let name = DistinguishedName::parse(&tlv(tag::SEQUENCE, &sequence))?;
        assert_eq!(name.to_string(), "CN=alice,O=OxiXML");
        Ok(())
    }

    #[test]
    fn special_characters_are_escaped() -> Result<(), CertificateError> {
        let name = DistinguishedName::parse(&name(COMMON_NAME, 0x0c, b" a,b+c "))?;
        assert_eq!(name.to_string(), "CN=\\ a\\,b\\+c\\ ");
        Ok(())
    }

    #[test]
    fn a_malformed_name_is_refused() {
        assert!(DistinguishedName::parse(&[0x30, 0x01]).is_err());
        // A `SEQUENCE` where the grammar wants a `SET`.
        assert!(DistinguishedName::parse(&tlv(tag::SEQUENCE, &tlv(tag::SEQUENCE, &[]))).is_err());
    }

    #[test]
    fn an_empty_name_renders_and_matches_itself() -> Result<(), CertificateError> {
        let empty = DistinguishedName::parse(&tlv(tag::SEQUENCE, &[]))?;
        assert!(empty.is_empty());
        assert_eq!(empty.to_string(), "<empty>");
        assert!(empty.matches(&empty));
        Ok(())
    }
}
