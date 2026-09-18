//! RFC 4648 base16 (hexadecimal) encoding and decoding, for `xsd:hexBinary`.
//!
//! Encoding comes in lower- and upper-case flavors ([`encode_lower`] / [`encode_upper`]);
//! `xsd:hexBinary`'s canonical lexical form is upper-case. Decoding ([`decode`]) accepts a
//! mix of both cases, as the `xsd:hexBinary` lexical space does, and rejects anything else:
//! odd-length input, non-hex-digit bytes, and non-ASCII input.

use std::error::Error;
use std::fmt;

const LOWER_DIGITS: &[u8; 16] = b"0123456789abcdef";
const UPPER_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// An error produced while decoding hex text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexError {
    /// The input has an odd number of characters; a whole number of bytes needs pairs of
    /// hex digits.
    OddLength {
        /// The (odd) length of the offending input, in bytes.
        length: usize,
    },
    /// A byte at `position` is not an ASCII hex digit (`0-9`, `a-f`, `A-F`).
    InvalidByte {
        /// Byte offset of the offending byte within the input.
        position: usize,
        /// The offending byte.
        byte: u8,
    },
}

impl fmt::Display for HexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::OddLength { length } => {
                write!(f, "odd-length hex input ({length} characters)")
            }
            Self::InvalidByte { position, byte } => {
                write!(f, "invalid hex digit {byte:#04x} at position {position}")
            }
        }
    }
}

impl Error for HexError {}

/// Encodes `input` as lower-case hex text (two characters per byte).
#[must_use]
pub fn encode_lower(input: &[u8]) -> String {
    encode(input, LOWER_DIGITS)
}

/// Encodes `input` as upper-case hex text (two characters per byte). This is the canonical
/// lexical form for `xsd:hexBinary`.
#[must_use]
pub fn encode_upper(input: &[u8]) -> String {
    encode(input, UPPER_DIGITS)
}

fn encode(input: &[u8], digits: &[u8; 16]) -> String {
    let mut out = String::with_capacity(input.len() * 2);
    for &byte in input {
        out.push(digits[(byte >> 4) as usize] as char);
        out.push(digits[(byte & 0x0F) as usize] as char);
    }
    out
}

/// Decodes hex text into bytes, accepting either case (or a mix of both) per digit.
///
/// # Errors
///
/// Returns [`HexError::OddLength`] if `input` has an odd number of bytes, or
/// [`HexError::InvalidByte`] if any byte is not an ASCII hex digit.
pub fn decode(input: &str) -> Result<Vec<u8>, HexError> {
    let bytes = input.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(HexError::OddLength {
            length: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for (pair_index, pair) in bytes.chunks_exact(2).enumerate() {
        let base = pair_index * 2;
        let high = hex_value(pair[0]).ok_or(HexError::InvalidByte {
            position: base,
            byte: pair[0],
        })?;
        let low = hex_value(pair[1]).ok_or(HexError::InvalidByte {
            position: base + 1,
            byte: pair[1],
        })?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

/// Maps a single ASCII hex-digit byte to its 4-bit value, accepting either case.
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
    fn encode_vectors() {
        assert_eq!(encode_lower(b""), "");
        assert_eq!(encode_lower(&[0xDE, 0xAD, 0xBE, 0xEF]), "deadbeef");
        assert_eq!(encode_upper(&[0xDE, 0xAD, 0xBE, 0xEF]), "DEADBEEF");
        assert_eq!(encode_lower(&[0x00, 0x01, 0x0F, 0xFF]), "00010fff");
    }

    #[test]
    fn decode_vectors() -> Result<(), HexError> {
        assert_eq!(decode("")?, Vec::<u8>::new());
        assert_eq!(decode("deadbeef")?, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(decode("DEADBEEF")?, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(decode("DeAdBeEf")?, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        Ok(())
    }

    #[test]
    fn decode_rejects_odd_length() {
        assert_eq!(decode("abc"), Err(HexError::OddLength { length: 3 }));
    }

    #[test]
    fn decode_rejects_invalid_digit() {
        assert_eq!(
            decode("zz"),
            Err(HexError::InvalidByte {
                position: 0,
                byte: b'z'
            })
        );
        assert_eq!(
            decode("aZ"),
            Err(HexError::InvalidByte {
                position: 1,
                byte: b'Z'
            })
        );
    }

    #[test]
    fn decode_rejects_non_ascii() {
        // "é" is a two-byte UTF-8 sequence; neither byte is an ASCII hex digit.
        let err = decode("é").expect_err("non-ASCII input must be rejected");
        assert!(matches!(err, HexError::InvalidByte { .. }));
    }

    #[test]
    fn property_round_trip_random_bytes() -> Result<(), HexError> {
        let mut rng = SplitMix64::new(0x0E0E_0E0E_0E0E_0E0Eu64);
        for len in 0..64usize {
            let mut data = vec![0u8; len];
            rng.fill_bytes(&mut data);
            assert_eq!(decode(&encode_lower(&data))?, data, "len={len} lower");
            assert_eq!(decode(&encode_upper(&data))?, data, "len={len} upper");
        }
        Ok(())
    }
}
