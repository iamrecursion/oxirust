//! Byte-order-mark detection.

use crate::label::Encoding;

/// Detect a byte-order mark (BOM) at the start of `bytes`, one of the three signatures relevant
/// to the encodings this crate supports (see
/// [XML 1.0 Appendix F.1](https://www.w3.org/TR/xml/#sec-guessing)):
///
/// | Bytes            | Encoding      |
/// |-------------------|--------------|
/// | `EF BB BF`         | [`Encoding::Utf8`] |
/// | `FE FF`             | [`Encoding::Utf16Be`] |
/// | `FF FE`             | [`Encoding::Utf16Le`] |
///
/// Returns the detected encoding together with the number of leading bytes that make up the
/// BOM, so a caller can skip exactly that many bytes before decoding the rest of the stream.
/// Returns [`None`] if `bytes` does not start with any of the three signatures (this includes
/// streams that are simply too short to contain one).
///
/// # Examples
///
/// ```
/// use oxixml_encoding::{detect_bom, Encoding};
///
/// assert_eq!(detect_bom(&[0xEF, 0xBB, 0xBF, b'<']), Some((Encoding::Utf8, 3)));
/// assert_eq!(detect_bom(&[0xFF, 0xFE, b'<', 0x00]), Some((Encoding::Utf16Le, 2)));
/// assert_eq!(detect_bom(b"<?xml"), None);
/// ```
#[must_use]
pub fn detect_bom(bytes: &[u8]) -> Option<(Encoding, usize)> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        Some((Encoding::Utf8, 3))
    } else if bytes.starts_with(&[0xFE, 0xFF]) {
        Some((Encoding::Utf16Be, 2))
    } else if bytes.starts_with(&[0xFF, 0xFE]) {
        Some((Encoding::Utf16Le, 2))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_utf8_bom() {
        assert_eq!(detect_bom(&[0xEF, 0xBB, 0xBF]), Some((Encoding::Utf8, 3)));
        assert_eq!(
            detect_bom(&[0xEF, 0xBB, 0xBF, b'a', b'b']),
            Some((Encoding::Utf8, 3))
        );
    }

    #[test]
    fn detects_utf16_be_bom() {
        assert_eq!(detect_bom(&[0xFE, 0xFF]), Some((Encoding::Utf16Be, 2)));
    }

    #[test]
    fn detects_utf16_le_bom() {
        assert_eq!(detect_bom(&[0xFF, 0xFE]), Some((Encoding::Utf16Le, 2)));
    }

    #[test]
    fn no_bom_returns_none() {
        assert_eq!(detect_bom(b"<?xml version=\"1.0\"?>"), None);
        assert_eq!(detect_bom(b""), None);
        assert_eq!(detect_bom(&[0x00]), None);
    }

    #[test]
    fn utf32_bom_is_not_recognized() {
        // 00 00 FE FF is the UTF-32BE BOM; this crate has no UTF-32 support, so it must not be
        // misreported as one of the three encodings it does support.
        assert_eq!(detect_bom(&[0x00, 0x00, 0xFE, 0xFF]), None);
    }
}
