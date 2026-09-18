//! RFC 4648 base64 encoding and decoding.
//!
//! Two alphabets are supported ([`Alphabet::Standard`] and [`Alphabet::UrlSafe`]), each with
//! optional `=` padding on encode. Two decoders are provided:
//!
//! * [`Base64Config::decode`] — **strict**: rejects bytes outside the alphabet, malformed or
//!   inconsistent padding, and a "non-canonical" final quantum (a trailing partial group whose
//!   unused low bits are not all zero, per [RFC 4648 §3.5]).
//! * [`Base64Config::decode_forgiving`] — the same validation, but ASCII whitespace
//!   (`#x20 #x9 #xD #xA`) anywhere in the input is skipped first. This is the lexical space
//!   required by `xsd:base64Binary`, whose value may be wrapped across multiple lines.
//!
//! Both a config-struct API ([`Base64Config`]) and plain function wrappers for the common
//! cases ([`base64_encode`], [`base64_decode`], ...) are provided.
//!
//! [RFC 4648 §3.5]: https://www.rfc-editor.org/rfc/rfc4648#section-3.5

use std::error::Error;
use std::fmt;

const STANDARD_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const URL_SAFE_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
const PAD: u8 = b'=';

/// Which of the two RFC 4648 base64 alphabets to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alphabet {
    /// The standard alphabet (`+`, `/`), RFC 4648 §4.
    Standard,
    /// The URL- and filename-safe alphabet (`-`, `_`), RFC 4648 §5.
    UrlSafe,
}

impl Alphabet {
    /// The 64-byte encode table: index `n` (0..64) maps to the ASCII byte for symbol `n`.
    const fn encode_table(self) -> &'static [u8; 64] {
        match self {
            Self::Standard => STANDARD_ALPHABET,
            Self::UrlSafe => URL_SAFE_ALPHABET,
        }
    }

    /// Maps an ASCII byte back to its 6-bit symbol value, or `None` if `byte` is not part of
    /// this alphabet (and is not the `=` padding character, which is handled separately).
    fn decode_symbol(self, byte: u8) -> Option<u8> {
        let table = self.encode_table();
        table.iter().position(|&b| b == byte).map(|pos| pos as u8)
    }
}

/// Whether [`Base64Config::encode`] appends `=` padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Padding {
    /// Pad the output with `=` so its length is always a multiple of 4 (RFC 4648 §4/§5).
    Include,
    /// Omit padding (RFC 4648 §3.2).
    Omit,
}

/// An error produced while decoding base64 text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base64Error {
    /// A byte at `position` is not part of the selected alphabet (and is not `=`).
    InvalidByte {
        /// Byte offset of the offending byte within the input passed to `decode`.
        position: usize,
        /// The offending byte.
        byte: u8,
    },
    /// The number of data symbols is not consistent with any valid base64 encoding (for
    /// example a final group of exactly one symbol, which cannot represent a whole byte).
    InvalidLength,
    /// `=` padding is present but malformed: in the wrong position, the wrong count for the
    /// number of trailing data symbols, or otherwise inconsistent with RFC 4648 §4.
    InvalidPadding,
    /// The final (partial) group decodes successfully, but its unused low-order bits are not
    /// all zero. RFC 4648 §3.5 calls such input non-canonical; strict decoders must reject it.
    NonCanonicalFinalQuantum,
}

impl fmt::Display for Base64Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidByte { position, byte } => {
                write!(f, "invalid base64 byte {byte:#04x} at position {position}")
            }
            Self::InvalidLength => f.write_str("invalid base64 input length"),
            Self::InvalidPadding => f.write_str("invalid base64 padding"),
            Self::NonCanonicalFinalQuantum => {
                f.write_str("non-canonical base64 final quantum: unused bits are not zero")
            }
        }
    }
}

impl Error for Base64Error {}

/// A base64 alphabet + padding configuration, usable to encode and decode.
///
/// Constants are provided for the four common combinations; [`Base64Config::new`] builds any
/// other combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Base64Config {
    alphabet: Alphabet,
    padding: Padding,
}

impl Base64Config {
    /// The standard alphabet, with `=` padding (the RFC 4648 §4 default).
    pub const STANDARD: Self = Self::new(Alphabet::Standard, Padding::Include);
    /// The standard alphabet, without padding.
    pub const STANDARD_NO_PAD: Self = Self::new(Alphabet::Standard, Padding::Omit);
    /// The URL- and filename-safe alphabet, with `=` padding.
    pub const URL_SAFE: Self = Self::new(Alphabet::UrlSafe, Padding::Include);
    /// The URL- and filename-safe alphabet, without padding (the common web convention).
    pub const URL_SAFE_NO_PAD: Self = Self::new(Alphabet::UrlSafe, Padding::Omit);

    /// Builds a configuration from an explicit alphabet and padding choice.
    #[must_use]
    pub const fn new(alphabet: Alphabet, padding: Padding) -> Self {
        Self { alphabet, padding }
    }

    /// Encodes `input` as base64 text using this configuration's alphabet and padding.
    #[must_use]
    pub fn encode(&self, input: &[u8]) -> String {
        let table = self.alphabet.encode_table();
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        let mut chunks = input.chunks_exact(3);
        for chunk in &mut chunks {
            let n = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
            out.push(table[((n >> 18) & 0x3F) as usize] as char);
            out.push(table[((n >> 12) & 0x3F) as usize] as char);
            out.push(table[((n >> 6) & 0x3F) as usize] as char);
            out.push(table[(n & 0x3F) as usize] as char);
        }
        match chunks.remainder() {
            [b0] => {
                let n = u32::from(*b0) << 16;
                out.push(table[((n >> 18) & 0x3F) as usize] as char);
                out.push(table[((n >> 12) & 0x3F) as usize] as char);
                if self.padding == Padding::Include {
                    out.push('=');
                    out.push('=');
                }
            }
            [b0, b1] => {
                let n = (u32::from(*b0) << 16) | (u32::from(*b1) << 8);
                out.push(table[((n >> 18) & 0x3F) as usize] as char);
                out.push(table[((n >> 12) & 0x3F) as usize] as char);
                out.push(table[((n >> 6) & 0x3F) as usize] as char);
                if self.padding == Padding::Include {
                    out.push('=');
                }
            }
            _ => {}
        }
        out
    }

    /// Decodes strict base64 `input`: every byte must be an alphabet symbol or (only as a
    /// well-formed trailing group) `=`, and the final partial quantum's unused bits must be
    /// zero.
    ///
    /// # Errors
    ///
    /// Returns [`Base64Error`] if `input` contains a byte outside the alphabet (ASCII
    /// whitespace included — use [`Base64Config::decode_forgiving`] to tolerate that),
    /// malformed padding, an impossible length, or a non-canonical final quantum.
    pub fn decode(&self, input: &str) -> Result<Vec<u8>, Base64Error> {
        self.decode_bytes(input.as_bytes())
    }

    /// Decodes `input` the same way as [`Base64Config::decode`], except ASCII whitespace
    /// (`#x20 #x9 #xD #xA`) is stripped from `input` first. This is the lexical space
    /// `xsd:base64Binary` requires (values may be wrapped across multiple lines).
    ///
    /// # Errors
    ///
    /// Returns [`Base64Error`] under the same conditions as [`Base64Config::decode`], with
    /// positions reported against the whitespace-stripped input.
    pub fn decode_forgiving(&self, input: &str) -> Result<Vec<u8>, Base64Error> {
        let stripped: Vec<u8> = input
            .bytes()
            .filter(|&b| !matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
            .collect();
        self.decode_bytes(&stripped)
    }

    /// Core strict decoder, operating on raw bytes that have already had any tolerated
    /// whitespace removed.
    fn decode_bytes(&self, cleaned: &[u8]) -> Result<Vec<u8>, Base64Error> {
        if cleaned.is_empty() {
            return Ok(Vec::new());
        }

        // Trailing `=` padding, if any, must form an unbroken run at the very end.
        let mut data_end = cleaned.len();
        let mut pad_count = 0usize;
        while data_end > 0 && cleaned[data_end - 1] == PAD {
            data_end -= 1;
            pad_count += 1;
        }
        if pad_count > 2 {
            return Err(Base64Error::InvalidPadding);
        }
        let data = &cleaned[..data_end];
        if data.contains(&PAD) {
            // `=` appeared somewhere other than a trailing run.
            return Err(Base64Error::InvalidPadding);
        }

        let remainder = data.len() % 4;
        match remainder {
            0 => {
                if pad_count != 0 {
                    return Err(Base64Error::InvalidPadding);
                }
            }
            1 => return Err(Base64Error::InvalidLength),
            2 => {
                if pad_count != 0 && pad_count != 2 {
                    return Err(Base64Error::InvalidPadding);
                }
            }
            3 => {
                if pad_count != 0 && pad_count != 1 {
                    return Err(Base64Error::InvalidPadding);
                }
            }
            _ => unreachable!("n % 4 is always in 0..4"),
        }

        let mut out = Vec::with_capacity(data.len() / 4 * 3 + 2);
        let full_quads = data.len() / 4;
        for quad_index in 0..full_quads {
            let base = quad_index * 4;
            let mut symbols = [0u8; 4];
            for (offset, symbol) in symbols.iter_mut().enumerate() {
                let position = base + offset;
                *symbol = self.symbol_at(data, position)?;
            }
            let n = (u32::from(symbols[0]) << 18)
                | (u32::from(symbols[1]) << 12)
                | (u32::from(symbols[2]) << 6)
                | u32::from(symbols[3]);
            out.push((n >> 16) as u8);
            out.push((n >> 8) as u8);
            out.push(n as u8);
        }

        let tail = &data[full_quads * 4..];
        match tail.len() {
            0 => {}
            2 => {
                let b0 = self.symbol_at(data, full_quads * 4)?;
                let b1 = self.symbol_at(data, full_quads * 4 + 1)?;
                let n = (u32::from(b0) << 6) | u32::from(b1);
                if n & 0xF != 0 {
                    return Err(Base64Error::NonCanonicalFinalQuantum);
                }
                out.push((n >> 4) as u8);
            }
            3 => {
                let b0 = self.symbol_at(data, full_quads * 4)?;
                let b1 = self.symbol_at(data, full_quads * 4 + 1)?;
                let b2 = self.symbol_at(data, full_quads * 4 + 2)?;
                let n = (u32::from(b0) << 12) | (u32::from(b1) << 6) | u32::from(b2);
                if n & 0x3 != 0 {
                    return Err(Base64Error::NonCanonicalFinalQuantum);
                }
                out.push((n >> 10) as u8);
                out.push((n >> 2) as u8);
            }
            _ => unreachable!("tail length was validated to be 0, 2 or 3 above"),
        }

        Ok(out)
    }

    /// Decodes the alphabet symbol at `position` within `data`, reporting `position` in the
    /// resulting error on failure.
    fn symbol_at(&self, data: &[u8], position: usize) -> Result<u8, Base64Error> {
        let byte = data[position];
        self.alphabet
            .decode_symbol(byte)
            .ok_or(Base64Error::InvalidByte { position, byte })
    }
}

/// Encodes `input` as base64 using the standard alphabet with `=` padding.
#[must_use]
pub fn base64_encode(input: &[u8]) -> String {
    Base64Config::STANDARD.encode(input)
}

/// Encodes `input` as base64 using the URL-safe alphabet with `=` padding.
#[must_use]
pub fn base64_encode_urlsafe(input: &[u8]) -> String {
    Base64Config::URL_SAFE.encode(input)
}

/// Strictly decodes standard-alphabet base64 text.
///
/// # Errors
///
/// See [`Base64Config::decode`].
pub fn base64_decode(input: &str) -> Result<Vec<u8>, Base64Error> {
    Base64Config::STANDARD.decode(input)
}

/// Strictly decodes URL-safe-alphabet base64 text.
///
/// # Errors
///
/// See [`Base64Config::decode`].
pub fn base64_decode_urlsafe(input: &str) -> Result<Vec<u8>, Base64Error> {
    Base64Config::URL_SAFE.decode(input)
}

/// Decodes standard-alphabet base64 text, tolerating embedded ASCII whitespace — the
/// `xsd:base64Binary` lexical space.
///
/// # Errors
///
/// See [`Base64Config::decode_forgiving`].
pub fn base64_decode_forgiving(input: &str) -> Result<Vec<u8>, Base64Error> {
    Base64Config::STANDARD.decode_forgiving(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_support::rng::{Rng, SplitMix64};

    // RFC 4648 §10 test vectors.
    const RFC_VECTORS: &[(&str, &str)] = &[
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ];

    #[test]
    fn rfc4648_encode_vectors() {
        for &(plain, encoded) in RFC_VECTORS {
            assert_eq!(
                base64_encode(plain.as_bytes()),
                encoded,
                "encode({plain:?})"
            );
        }
    }

    #[test]
    fn rfc4648_decode_vectors() -> Result<(), Base64Error> {
        for &(plain, encoded) in RFC_VECTORS {
            assert_eq!(
                base64_decode(encoded)?,
                plain.as_bytes(),
                "decode({encoded:?})"
            );
        }
        Ok(())
    }

    #[test]
    fn no_pad_variant_round_trips() -> Result<(), Base64Error> {
        for &(plain, _) in RFC_VECTORS {
            let encoded = Base64Config::STANDARD_NO_PAD.encode(plain.as_bytes());
            assert!(!encoded.contains('='));
            assert_eq!(
                Base64Config::STANDARD_NO_PAD.decode(&encoded)?,
                plain.as_bytes()
            );
            // The unpadded encoder must also decode fine through the padded decoder's
            // strict validation, since padding is optional on decode.
            assert_eq!(Base64Config::STANDARD.decode(&encoded)?, plain.as_bytes());
        }
        Ok(())
    }

    #[test]
    fn url_safe_alphabet_round_trips() -> Result<(), Base64Error> {
        let data = [0xFB, 0xFF, 0xBF];
        let standard = base64_encode(&data);
        let url_safe = base64_encode_urlsafe(&data);
        assert_ne!(standard, url_safe);
        assert!(url_safe.contains('_') || url_safe.contains('-'));
        assert_eq!(base64_decode_urlsafe(&url_safe)?, data);
        Ok(())
    }

    #[test]
    fn strict_decode_rejects_non_alphabet_byte() {
        let err = base64_decode("Zm9v!m9v").expect_err("must reject '!'");
        assert!(matches!(err, Base64Error::InvalidByte { byte: b'!', .. }));
    }

    #[test]
    fn strict_decode_rejects_ascii_whitespace() {
        // "Zm9vYmFy" (8 symbols, a multiple of 4) encodes "foobar"; swapping the 5th symbol
        // for a space keeps the overall length valid so the failure comes from the alphabet
        // check, not a length check.
        let err = base64_decode("Zm9v mFy").expect_err("must reject embedded whitespace");
        assert_eq!(
            err,
            Base64Error::InvalidByte {
                position: 4,
                byte: b' '
            }
        );
    }

    #[test]
    fn strict_decode_rejects_misplaced_padding() {
        assert_eq!(base64_decode("=Zg=="), Err(Base64Error::InvalidPadding));
        assert_eq!(base64_decode("Z=g="), Err(Base64Error::InvalidPadding));
        assert_eq!(base64_decode("Zg==="), Err(Base64Error::InvalidPadding));
        // Complete quads never take padding.
        assert_eq!(base64_decode("Zm9v="), Err(Base64Error::InvalidPadding));
    }

    #[test]
    fn strict_decode_rejects_impossible_length() {
        assert_eq!(base64_decode("Z"), Err(Base64Error::InvalidLength));
        assert_eq!(base64_decode("Zm9vZ"), Err(Base64Error::InvalidLength));
    }

    #[test]
    fn strict_decode_rejects_non_canonical_final_quantum() {
        // Two-symbol tail: "AB" = symbols (0, 1); n = (0 << 6) | 1 = 1, whose low 4 bits
        // (the bits a 1-byte payload never uses) are non-zero, so this is non-canonical.
        let err = base64_decode("AB==").expect_err("non-canonical final quantum (2 symbols)");
        assert_eq!(err, Base64Error::NonCanonicalFinalQuantum);

        // Three-symbol tail: "AAB" = symbols (0, 0, 1); n = 1, whose low 2 bits (unused by a
        // 2-byte payload) are non-zero, so this is also non-canonical.
        let err = base64_decode("AAB=").expect_err("non-canonical final quantum (3 symbols)");
        assert_eq!(err, Base64Error::NonCanonicalFinalQuantum);
    }

    #[test]
    fn forgiving_decode_skips_xml_whitespace() -> Result<(), Base64Error> {
        let input = "Zm9v\n  Ym\tFy\r\n";
        assert_eq!(base64_decode_forgiving(input)?, b"foobar");
        Ok(())
    }

    #[test]
    fn forgiving_decode_still_rejects_non_canonical() {
        let err = base64_decode_forgiving("A B =\t=").expect_err("still non-canonical");
        assert_eq!(err, Base64Error::NonCanonicalFinalQuantum);
    }

    #[test]
    fn property_round_trip_random_bytes() -> Result<(), Base64Error> {
        let mut rng = SplitMix64::new(0xB64B_64B6_4B64_B64B);
        for len in 0..96usize {
            let mut data = vec![0u8; len];
            rng.fill_bytes(&mut data);
            let encoded = base64_encode(&data);
            assert_eq!(base64_decode(&encoded)?, data, "len={len}");
            let encoded_url = base64_encode_urlsafe(&data);
            assert_eq!(
                base64_decode_urlsafe(&encoded_url)?,
                data,
                "len={len} urlsafe"
            );
            let no_pad = Base64Config::STANDARD_NO_PAD.encode(&data);
            assert_eq!(
                Base64Config::STANDARD_NO_PAD.decode(&no_pad)?,
                data,
                "len={len} no_pad"
            );
        }
        Ok(())
    }

    #[test]
    fn empty_input_round_trips() -> Result<(), Base64Error> {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_decode("")?, Vec::<u8>::new());
        Ok(())
    }
}
