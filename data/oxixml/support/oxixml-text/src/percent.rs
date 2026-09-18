//! Percent-encoding and decoding ([RFC 3986]), parameterized by the set of ASCII bytes that
//! may appear literally.
//!
//! [`AsciiSet`] is a const-constructible 128-bit bitmask (one bit per ASCII code point;
//! non-ASCII bytes are never members and are therefore always percent-encoded). A handful of
//! [RFC 3986] production-derived sets are provided as constants for the IRI components that
//! need distinct treatment: [`UNRESERVED`], [`PATH_SEGMENT`]/[`PATH`], [`QUERY`],
//! [`FRAGMENT`], and [`USERINFO`].
//!
//! Encoding a `&str` percent-encodes each UTF-8 byte not in the allowed set; multi-byte
//! sequences for non-ASCII characters are therefore always fully percent-encoded, byte by
//! byte, exactly as [RFC 3986 §2.5] requires. Decoding is available either as raw bytes
//! ([`percent_decode`]) or with strict UTF-8 validation ([`percent_decode_to_string`]).
//!
//! [RFC 3986]: https://www.rfc-editor.org/rfc/rfc3986
//! [RFC 3986 §2.5]: https://www.rfc-editor.org/rfc/rfc3986#section-2.5

use std::error::Error;
use std::fmt;

const UPPER_HEX: &[u8; 16] = b"0123456789ABCDEF";

/// A set of ASCII bytes (0..128), represented as a 128-bit const-constructible bitmask.
///
/// Bytes 128..=255 are never members: percent-encoding a byte outside the ASCII range is
/// always required (this is how non-ASCII UTF-8 content ends up fully percent-encoded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsciiSet([u64; 2]);

impl AsciiSet {
    /// The empty set: every byte is percent-encoded.
    #[must_use]
    pub const fn empty() -> Self {
        Self([0, 0])
    }

    /// Returns a copy of this set with `byte` added. A no-op for `byte >= 128`.
    #[must_use]
    pub const fn add(mut self, byte: u8) -> Self {
        if byte < 128 {
            let index = (byte / 64) as usize;
            let bit = byte % 64;
            self.0[index] |= 1u64 << bit;
        }
        self
    }

    /// Returns a copy of this set with every byte in `start..=end` added.
    #[must_use]
    pub const fn add_range(mut self, start: u8, end: u8) -> Self {
        let mut byte = start;
        while byte <= end {
            self = self.add(byte);
            if byte == u8::MAX {
                break;
            }
            byte += 1;
        }
        self
    }

    /// Returns the union of two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self([self.0[0] | other.0[0], self.0[1] | other.0[1]])
    }

    /// Returns whether `byte` is a member of this set. Always `false` for `byte >= 128`.
    #[must_use]
    pub const fn contains(&self, byte: u8) -> bool {
        if byte >= 128 {
            return false;
        }
        let index = (byte / 64) as usize;
        let bit = byte % 64;
        (self.0[index] >> bit) & 1 == 1
    }
}

/// [RFC 3986 §2.3] `unreserved` characters: `ALPHA / DIGIT / "-" / "." / "_" / "~"`.
///
/// These never need percent-encoding in any URI/IRI component.
///
/// [RFC 3986 §2.3]: https://www.rfc-editor.org/rfc/rfc3986#section-2.3
pub const UNRESERVED: AsciiSet = AsciiSet::empty()
    .add_range(b'A', b'Z')
    .add_range(b'a', b'z')
    .add_range(b'0', b'9')
    .add(b'-')
    .add(b'.')
    .add(b'_')
    .add(b'~');

/// [RFC 3986 §2.2] `sub-delims`: `"!" / "$" / "&" / "'" / "(" / ")" / "*" / "+" / "," / ";" / "="`.
const SUB_DELIMS: AsciiSet = AsciiSet::empty()
    .add(b'!')
    .add(b'$')
    .add(b'&')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'*')
    .add(b'+')
    .add(b',')
    .add(b';')
    .add(b'=');

/// Bytes allowed unescaped within a single `pchar` (a path segment, RFC 3986 §3.3, without
/// the `/` segment separator): `unreserved / sub-delims / ":" / "@"`.
pub const PATH_SEGMENT: AsciiSet = UNRESERVED.union(SUB_DELIMS).add(b':').add(b'@');

/// Bytes allowed unescaped within a full `path` (RFC 3986 §3.3): [`PATH_SEGMENT`] plus the
/// `/` segment separator, which is left intact rather than being percent-encoded.
pub const PATH: AsciiSet = PATH_SEGMENT.add(b'/');

/// Bytes allowed unescaped within a `query` (RFC 3986 §3.4): `pchar / "/" / "?"`.
pub const QUERY: AsciiSet = PATH_SEGMENT.add(b'/').add(b'?');

/// Bytes allowed unescaped within a `fragment` (RFC 3986 §3.5): `pchar / "/" / "?"`.
pub const FRAGMENT: AsciiSet = PATH_SEGMENT.add(b'/').add(b'?');

/// Bytes allowed unescaped within `userinfo` (RFC 3986 §3.2.1): `unreserved / sub-delims / ":"`.
pub const USERINFO: AsciiSet = UNRESERVED.union(SUB_DELIMS).add(b':');

/// An error produced while percent-decoding text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PercentError {
    /// A `%` at `position` is not followed by two hex digits (the input ends too soon).
    Truncated {
        /// Byte offset of the `%` that starts the incomplete escape.
        position: usize,
    },
    /// The byte at `position`, expected to be a hex digit following `%`, is not one.
    InvalidHex {
        /// Byte offset of the offending byte.
        position: usize,
        /// The offending byte.
        byte: u8,
    },
    /// The decoded bytes are not valid UTF-8 (only returned by
    /// [`percent_decode_to_string`]).
    InvalidUtf8 {
        /// The number of leading decoded bytes that did form valid UTF-8.
        valid_up_to: usize,
    },
}

impl fmt::Display for PercentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Truncated { position } => {
                write!(f, "truncated percent-escape at position {position}")
            }
            Self::InvalidHex { position, byte } => {
                write!(
                    f,
                    "invalid percent-escape hex digit {byte:#04x} at position {position}"
                )
            }
            Self::InvalidUtf8 { valid_up_to } => {
                write!(
                    f,
                    "invalid UTF-8 after byte {valid_up_to} of decoded output"
                )
            }
        }
    }
}

impl Error for PercentError {}

/// Percent-encodes `input`, leaving bytes in `allowed` untouched and encoding every other
/// byte as `%XX` (upper-case hex).
#[must_use]
pub fn percent_encode(input: &[u8], allowed: &AsciiSet) -> String {
    let mut out = String::with_capacity(input.len());
    for &byte in input {
        if allowed.contains(byte) {
            out.push(byte as char);
        } else {
            out.push('%');
            out.push(UPPER_HEX[(byte >> 4) as usize] as char);
            out.push(UPPER_HEX[(byte & 0x0F) as usize] as char);
        }
    }
    out
}

/// Percent-encodes the UTF-8 bytes of `input`. Non-ASCII characters are always encoded byte
/// by byte, since [`AsciiSet`] only ever admits ASCII bytes.
#[must_use]
pub fn percent_encode_str(input: &str, allowed: &AsciiSet) -> String {
    percent_encode(input.as_bytes(), allowed)
}

/// Percent-decodes `input` into raw bytes. `%XX` escapes are decoded; every other byte is
/// passed through unchanged. No UTF-8 validation is performed — use
/// [`percent_decode_to_string`] if `input` is expected to decode to text.
///
/// # Errors
///
/// Returns [`PercentError::Truncated`] if a `%` is not followed by two more bytes, or
/// [`PercentError::InvalidHex`] if either of those bytes is not an ASCII hex digit.
pub fn percent_decode(input: &str) -> Result<Vec<u8>, PercentError> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err(PercentError::Truncated { position: index });
            }
            let high = hex_value(bytes[index + 1]).ok_or(PercentError::InvalidHex {
                position: index + 1,
                byte: bytes[index + 1],
            })?;
            let low = hex_value(bytes[index + 2]).ok_or(PercentError::InvalidHex {
                position: index + 2,
                byte: bytes[index + 2],
            })?;
            out.push((high << 4) | low);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    Ok(out)
}

/// Percent-decodes `input` and validates that the result is well-formed UTF-8.
///
/// # Errors
///
/// Returns any [`percent_decode`] error, or [`PercentError::InvalidUtf8`] if the decoded
/// bytes are not valid UTF-8.
pub fn percent_decode_to_string(input: &str) -> Result<String, PercentError> {
    let bytes = percent_decode(input)?;
    String::from_utf8(bytes).map_err(|error| PercentError::InvalidUtf8 {
        valid_up_to: error.utf8_error().valid_up_to(),
    })
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_support::rng::{Rng, SplitMix64};

    #[test]
    fn unreserved_set_contains_expected_bytes() {
        for byte in b'A'..=b'Z' {
            assert!(UNRESERVED.contains(byte));
        }
        for byte in b'0'..=b'9' {
            assert!(UNRESERVED.contains(byte));
        }
        for byte in [b'-', b'.', b'_', b'~'] {
            assert!(UNRESERVED.contains(byte));
        }
        assert!(!UNRESERVED.contains(b' '));
        assert!(!UNRESERVED.contains(b'/'));
        assert!(!UNRESERVED.contains(128));
        assert!(!UNRESERVED.contains(255));
    }

    #[test]
    fn per_component_sets_differ() {
        // '/' is a path separator: kept literal in PATH, escaped within a single segment.
        assert!(PATH.contains(b'/'));
        assert!(!PATH_SEGMENT.contains(b'/'));

        // '?' starts a query: kept literal within query/fragment, escaped in a path segment.
        assert!(QUERY.contains(b'?'));
        assert!(FRAGMENT.contains(b'?'));
        assert!(!PATH_SEGMENT.contains(b'?'));

        // ':' separates userinfo from host, and is a `pchar`, so it's allowed unescaped in
        // both userinfo and path contexts.
        assert!(USERINFO.contains(b':'));
        assert!(PATH_SEGMENT.contains(b':'));

        // '@' terminates userinfo; escaping it there avoids ambiguity, but it's a normal
        // `pchar` in a path segment.
        assert!(!USERINFO.contains(b'@'));
        assert!(PATH_SEGMENT.contains(b'@'));
    }

    #[test]
    fn encode_percent_escapes_disallowed_bytes() {
        assert_eq!(percent_encode_str("a b/c", &PATH_SEGMENT), "a%20b%2Fc");
        assert_eq!(percent_encode_str("a b/c", &PATH), "a%20b/c");
        assert_eq!(percent_encode_str("hello", &UNRESERVED), "hello");
        assert_eq!(percent_encode_str("100%", &UNRESERVED), "100%25");
    }

    #[test]
    fn encode_non_ascii_is_utf8_byte_escaped() {
        // "café" — 'é' is U+00E9, UTF-8 bytes 0xC3 0xA9.
        assert_eq!(percent_encode_str("café", &UNRESERVED), "caf%C3%A9");
    }

    #[test]
    fn decode_round_trips_encoded_text() -> Result<(), PercentError> {
        assert_eq!(percent_decode_to_string("caf%C3%A9")?, "café".to_string());
        assert_eq!(percent_decode_to_string("a%20b%2Fc")?, "a b/c".to_string());
        assert_eq!(
            percent_decode_to_string("no-escapes")?,
            "no-escapes".to_string()
        );
        Ok(())
    }

    #[test]
    fn decode_accepts_lower_and_upper_hex() -> Result<(), PercentError> {
        assert_eq!(percent_decode("%2f")?, vec![b'/']);
        assert_eq!(percent_decode("%2F")?, vec![b'/']);
        assert_eq!(percent_decode("%2f")?, percent_decode("%2F")?);
        Ok(())
    }

    #[test]
    fn decode_rejects_truncated_escape() {
        assert_eq!(
            percent_decode("%2"),
            Err(PercentError::Truncated { position: 0 })
        );
        assert_eq!(
            percent_decode("%"),
            Err(PercentError::Truncated { position: 0 })
        );
        assert_eq!(
            percent_decode("abc%"),
            Err(PercentError::Truncated { position: 3 })
        );
    }

    #[test]
    fn decode_rejects_invalid_hex_digit() {
        assert_eq!(
            percent_decode("%zz"),
            Err(PercentError::InvalidHex {
                position: 1,
                byte: b'z'
            })
        );
    }

    #[test]
    fn decode_to_string_rejects_invalid_utf8() {
        // %FF is never a valid standalone UTF-8 byte.
        let err = percent_decode_to_string("%FF").expect_err("must reject invalid UTF-8");
        assert_eq!(err, PercentError::InvalidUtf8 { valid_up_to: 0 });
    }

    #[test]
    fn property_round_trip_random_strings() -> Result<(), PercentError> {
        let mut rng = SplitMix64::new(0xA1DA_A1DA_A1DA_A1DA);
        let corpus: &[char] = &[
            'a', 'Z', '0', '-', '.', '_', '~', ' ', '/', '?', '#', '@', ':', '%', 'é', '漢',
        ];
        for _ in 0..200 {
            let len = rng.gen_range(0..12) as usize;
            let mut sample = String::new();
            for _ in 0..len {
                let index = rng.gen_range(0..corpus.len() as u64) as usize;
                sample.push(corpus[index]);
            }
            let encoded = percent_encode_str(&sample, &UNRESERVED);
            assert_eq!(
                percent_decode_to_string(&encoded)?,
                sample,
                "sample={sample:?}"
            );
        }
        Ok(())
    }
}
