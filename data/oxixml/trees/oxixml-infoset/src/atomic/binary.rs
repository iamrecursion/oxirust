//! Lexical mappings for the two XSD binary datatypes, `xs:hexBinary` and `xs:base64Binary`.
//!
//! Both are defined by [XML Schema 1.1 Part 2 §3.3.15/§3.3.16](https://www.w3.org/TR/xmlschema11-2/#hexBinary).
//! The canonical forms are upper-case hexadecimal and RFC 2045 base64 with no line breaks and
//! canonical (zero-padded) final quantum, exactly as produced here.

use crate::error::{ErrorCode, XdmError};

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `bytes` as the canonical `xs:hexBinary` lexical form (upper-case, two digits per
/// octet).
#[must_use]
pub(crate) fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

/// Parses an `xs:hexBinary` lexical form. Both cases are accepted on input, per the
/// specification's lexical space `([0-9a-fA-F]{2})*`.
pub(crate) fn decode_hex(lexical: &str) -> Result<Vec<u8>, XdmError> {
    let bytes = lexical.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(invalid("xs:hexBinary", lexical, "odd number of hex digits"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = hex_digit(pair[0]).ok_or_else(|| {
            invalid(
                "xs:hexBinary",
                lexical,
                "characters outside [0-9a-fA-F] are not hex digits",
            )
        })?;
        let low = hex_digit(pair[1]).ok_or_else(|| {
            invalid(
                "xs:hexBinary",
                lexical,
                "characters outside [0-9a-fA-F] are not hex digits",
            )
        })?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Encodes `bytes` as the canonical `xs:base64Binary` lexical form: no whitespace, `=` padding,
/// and a final quantum whose unused low-order bits are zero.
#[must_use]
pub(crate) fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);
    for chunk in &mut chunks {
        let value = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
        for shift in [18, 12, 6, 0] {
            out.push(char::from(
                BASE64_ALPHABET[((value >> shift) & 0x3f) as usize],
            ));
        }
    }
    match chunks.remainder() {
        [a] => {
            let value = u32::from(*a) << 16;
            out.push(char::from(BASE64_ALPHABET[((value >> 18) & 0x3f) as usize]));
            out.push(char::from(BASE64_ALPHABET[((value >> 12) & 0x3f) as usize]));
            out.push('=');
            out.push('=');
        }
        [a, b] => {
            let value = (u32::from(*a) << 16) | (u32::from(*b) << 8);
            out.push(char::from(BASE64_ALPHABET[((value >> 18) & 0x3f) as usize]));
            out.push(char::from(BASE64_ALPHABET[((value >> 12) & 0x3f) as usize]));
            out.push(char::from(BASE64_ALPHABET[((value >> 6) & 0x3f) as usize]));
            out.push('=');
        }
        _ => {}
    }
    out
}

/// Parses an `xs:base64Binary` lexical form.
///
/// Whitespace is permitted anywhere between the base64 characters (the datatype's `whiteSpace`
/// facet is `collapse`, and the lexical space explicitly allows the groups to be separated).
/// Non-canonical final quanta — those whose unused low-order bits are non-zero — are rejected,
/// matching the specification's lexical space, which fixes the final character to one of
/// `[AQgw]` (two padding characters) or `[AEIMQUYcgkosw048]` (one padding character).
pub(crate) fn decode_base64(lexical: &str) -> Result<Vec<u8>, XdmError> {
    let mut symbols: Vec<u8> = Vec::with_capacity(lexical.len());
    let mut padding = 0_usize;
    for character in lexical.chars() {
        if character.is_whitespace() {
            continue;
        }
        if character == '=' {
            padding += 1;
            if padding > 2 {
                return Err(invalid(
                    "xs:base64Binary",
                    lexical,
                    "at most two padding characters are allowed",
                ));
            }
            continue;
        }
        if padding != 0 {
            return Err(invalid(
                "xs:base64Binary",
                lexical,
                "base64 characters may not follow the padding",
            ));
        }
        let index = base64_index(character).ok_or_else(|| {
            invalid(
                "xs:base64Binary",
                lexical,
                "character is outside the base64 alphabet",
            )
        })?;
        symbols.push(index);
    }
    if (symbols.len() + padding) % 4 != 0 {
        return Err(invalid(
            "xs:base64Binary",
            lexical,
            "the number of base64 characters (including padding) must be a multiple of four",
        ));
    }
    match padding {
        0 => {}
        1 if symbols.len() % 4 == 3 => {}
        2 if symbols.len() % 4 == 2 => {}
        _ => {
            return Err(invalid(
                "xs:base64Binary",
                lexical,
                "the padding length does not match the number of base64 characters",
            ));
        }
    }

    let mut out = Vec::with_capacity(symbols.len() * 3 / 4);
    let mut full = symbols.chunks_exact(4);
    for chunk in &mut full {
        let value = (u32::from(chunk[0]) << 18)
            | (u32::from(chunk[1]) << 12)
            | (u32::from(chunk[2]) << 6)
            | u32::from(chunk[3]);
        out.push(((value >> 16) & 0xff) as u8);
        out.push(((value >> 8) & 0xff) as u8);
        out.push((value & 0xff) as u8);
    }
    match full.remainder() {
        [] => {}
        [a, b] => {
            if b & 0x0f != 0 {
                return Err(invalid(
                    "xs:base64Binary",
                    lexical,
                    "the final quantum is not canonical: its unused low-order bits are set",
                ));
            }
            out.push((((u32::from(*a) << 18) | (u32::from(*b) << 12)) >> 16) as u8);
        }
        [a, b, c] => {
            if c & 0x03 != 0 {
                return Err(invalid(
                    "xs:base64Binary",
                    lexical,
                    "the final quantum is not canonical: its unused low-order bits are set",
                ));
            }
            let value = (u32::from(*a) << 18) | (u32::from(*b) << 12) | (u32::from(*c) << 6);
            out.push(((value >> 16) & 0xff) as u8);
            out.push(((value >> 8) & 0xff) as u8);
        }
        _ => {
            return Err(invalid(
                "xs:base64Binary",
                lexical,
                "a trailing group of one base64 character carries no octet",
            ));
        }
    }
    Ok(out)
}

fn base64_index(character: char) -> Option<u8> {
    let byte = u8::try_from(u32::from(character)).ok()?;
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn invalid(type_name: &str, lexical: &str, reason: &str) -> XdmError {
    XdmError::new(
        ErrorCode::Forg0001,
        format!("`{lexical}` is not a valid {type_name}: {reason}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_canonicalizes_to_upper_case() -> Result<(), XdmError> {
        assert_eq!(encode_hex(b""), "");
        assert_eq!(encode_hex(&[0x0f, 0xa0]), "0FA0");
        assert_eq!(decode_hex("0fa0")?, vec![0x0f, 0xa0]);
        assert_eq!(decode_hex("0FA0")?, vec![0x0f, 0xa0]);
        assert!(decode_hex("0F0").is_err());
        assert!(decode_hex("0G").is_err());
        Ok(())
    }

    #[test]
    fn base64_matches_the_rfc_4648_test_vectors() -> Result<(), XdmError> {
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode_base64(plain.as_bytes()), encoded, "encoding {plain}");
            assert_eq!(
                decode_base64(encoded)?,
                plain.as_bytes(),
                "decoding {encoded}"
            );
        }
        Ok(())
    }

    #[test]
    fn base64_tolerates_whitespace_but_rejects_non_canonical_quanta() -> Result<(), XdmError> {
        assert_eq!(decode_base64("Zm9v YmFy")?, b"foobar");
        assert_eq!(decode_base64("Zm9\nvYmFy")?, b"foobar");
        // `Zh==` would decode to `f` but sets bits the canonical form leaves zero.
        assert!(decode_base64("Zh==").is_err());
        assert!(decode_base64("Zm9=").is_err());
        assert!(decode_base64("Zg=").is_err());
        assert!(decode_base64("Z===").is_err());
        assert!(decode_base64("Zm9v=").is_err());
        assert!(decode_base64("Zm9*").is_err());
        Ok(())
    }
}
