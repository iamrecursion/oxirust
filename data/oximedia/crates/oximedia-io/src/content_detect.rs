//! Format auto-detection by buffer content.
//!
//! [`ContentDetector`] provides three orthogonal detection methods:
//!
//! 1. **Text encoding** — inspects BOM markers and byte statistics to classify
//!    a buffer as UTF-8, UTF-16 (LE/BE), Latin-1, or plain ASCII.
//!
//! 2. **Binary vs. text** — heuristic based on null-byte density and
//!    non-printable byte ratio.
//!
//! 3. **Media type** — delegates to [`FormatDetector`] for magic-byte
//!    identification and maps the result to a broad [`MediaType`] category.

use crate::format_detector::{FormatDetector, MediaFormat};

// ─────────────────────────────────────────────────────────────────────────────
// TextEncoding
// ─────────────────────────────────────────────────────────────────────────────

/// Character encoding detected from a data buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextEncoding {
    /// Unicode UTF-8 (with or without a leading BOM `EF BB BF`).
    Utf8,
    /// UTF-16 Little-Endian (BOM `FF FE`).
    Utf16Le,
    /// UTF-16 Big-Endian (BOM `FE FF`).
    Utf16Be,
    /// ISO 8859-1 / Windows-1252 (bytes in 0x80–0xFF range present).
    Latin1,
    /// Pure 7-bit ASCII (only printable characters and common control chars).
    Ascii,
}

impl std::fmt::Display for TextEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextEncoding::Utf8 => write!(f, "UTF-8"),
            TextEncoding::Utf16Le => write!(f, "UTF-16LE"),
            TextEncoding::Utf16Be => write!(f, "UTF-16BE"),
            TextEncoding::Latin1 => write!(f, "Latin-1"),
            TextEncoding::Ascii => write!(f, "ASCII"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MediaType
// ─────────────────────────────────────────────────────────────────────────────

/// Broad media type category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Video container (MP4, MKV, WebM, MXF, etc.).
    Video,
    /// Audio file (FLAC, WAV, MP3, AAC, etc.).
    Audio,
    /// Image (JPEG, PNG, WebP, EXR, DPX, etc.).
    Image,
    /// Compressed archive or data container (ZIP, GZ, XZ, etc.).
    Archive,
    /// Plain-text content (SRT, VTT, SVG, etc.).
    Text,
    /// Binary data that did not match any known media format.
    Binary,
    /// Content type could not be determined (empty or ambiguous buffer).
    Unknown,
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaType::Video => write!(f, "Video"),
            MediaType::Audio => write!(f, "Audio"),
            MediaType::Image => write!(f, "Image"),
            MediaType::Archive => write!(f, "Archive"),
            MediaType::Text => write!(f, "Text"),
            MediaType::Binary => write!(f, "Binary"),
            MediaType::Unknown => write!(f, "Unknown"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Number of bytes examined by the binary-detection heuristic.
const BINARY_PROBE_BYTES: usize = 8192;

/// Null-byte ratio threshold above which a buffer is considered binary.
const NULL_BYTE_THRESHOLD: f64 = 0.01; // 1 %

/// Non-printable byte ratio threshold above which a buffer is considered binary.
const NON_PRINTABLE_THRESHOLD: f64 = 0.30; // 30 %

/// Stateless content-based format detector.
pub struct ContentDetector;

impl ContentDetector {
    // ── Text encoding detection ───────────────────────────────────────────────

    /// Detect the text encoding of `data`.
    ///
    /// The detection proceeds in this order:
    ///
    /// 1. **BOM checks** — `EF BB BF` → UTF-8, `FF FE` → UTF-16LE,
    ///    `FE FF` → UTF-16BE.
    /// 2. **Pure ASCII** — all bytes are printable ASCII or common control
    ///    characters (`\t`, `\n`, `\r`).
    /// 3. **Valid UTF-8** — the entire slice passes `std::str::from_utf8`.
    /// 4. **Latin-1** — any byte in the `0x80–0xFF` range is present.
    /// 5. **Fallback** — returns [`TextEncoding::Ascii`].
    #[must_use]
    pub fn detect_encoding(data: &[u8]) -> TextEncoding {
        if data.is_empty() {
            return TextEncoding::Ascii;
        }

        // ── BOM detection ─────────────────────────────────────────────────────
        if data.len() >= 3 && data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF {
            return TextEncoding::Utf8;
        }
        if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xFE {
            return TextEncoding::Utf16Le;
        }
        if data.len() >= 2 && data[0] == 0xFE && data[1] == 0xFF {
            return TextEncoding::Utf16Be;
        }

        // ── Pure ASCII check ──────────────────────────────────────────────────
        // A byte is "ASCII-compatible" if it is a printable ASCII character
        // (0x20–0x7E) or one of the three common whitespace control codes.
        let is_ascii_compat =
            |b: u8| -> bool { (0x20..=0x7E).contains(&b) || b == 0x09 || b == 0x0A || b == 0x0D };
        if data.iter().copied().all(is_ascii_compat) {
            return TextEncoding::Ascii;
        }

        // ── Valid UTF-8 check ─────────────────────────────────────────────────
        if std::str::from_utf8(data).is_ok() {
            return TextEncoding::Utf8;
        }

        // ── Latin-1 heuristic ─────────────────────────────────────────────────
        // If any byte is in the extended range it's likely Latin-1 / Windows-1252.
        if data.iter().any(|&b| b >= 0x80) {
            return TextEncoding::Latin1;
        }

        // Fallback (should be unreachable given the ASCII-compat check above,
        // but provides a safe default).
        TextEncoding::Ascii
    }

    // ── Binary detection ──────────────────────────────────────────────────────

    /// Return `true` when `data` appears to be binary content.
    ///
    /// The heuristic examines up to the first `BINARY_PROBE_BYTES` bytes.
    /// A buffer is considered binary when:
    ///
    /// - More than 1 % of the examined bytes are null (`0x00`), **or**
    /// - More than 30 % of the examined bytes are non-printable control
    ///   characters (bytes `< 0x08` or in the range `0x0E–0x1F`, excluding
    ///   tab `0x09`, LF `0x0A`, and CR `0x0D`).
    #[must_use]
    pub fn is_binary(data: &[u8]) -> bool {
        if data.is_empty() {
            return false;
        }
        let probe = &data[..data.len().min(BINARY_PROBE_BYTES)];
        let total = probe.len() as f64;
        let mut null_count = 0usize;
        let mut non_printable_count = 0usize;

        for &b in probe {
            if b == 0x00 {
                null_count += 1;
            }
            // Non-printable control characters (excluding \t, \n, \r).
            if b < 0x08 || (0x0E..=0x1F).contains(&b) {
                non_printable_count += 1;
            }
        }

        let null_ratio = null_count as f64 / total;
        let non_printable_ratio = non_printable_count as f64 / total;

        null_ratio > NULL_BYTE_THRESHOLD || non_printable_ratio > NON_PRINTABLE_THRESHOLD
    }

    // ── Media type detection ──────────────────────────────────────────────────

    /// Detect the broad media type of `data` using magic-byte inspection.
    ///
    /// Delegates to [`FormatDetector::detect`] for magic-byte identification
    /// and maps the result to a [`MediaType`] category.  When the format is
    /// [`MediaFormat::Unknown`] this method falls back to [`Self::is_binary`]
    /// to distinguish [`MediaType::Binary`] from [`MediaType::Text`].
    #[must_use]
    pub fn detect_media_type(data: &[u8]) -> MediaType {
        if data.is_empty() {
            return MediaType::Unknown;
        }

        let detection = FormatDetector::detect(data);
        Self::media_format_to_type(detection.format, data)
    }

    /// Map a [`MediaFormat`] to the corresponding [`MediaType`].
    fn media_format_to_type(format: MediaFormat, data: &[u8]) -> MediaType {
        // Use the convenience helpers on MediaFormat where possible.
        if format.is_video() {
            return MediaType::Video;
        }
        if format.is_audio() {
            return MediaType::Audio;
        }
        if format.is_image() {
            return MediaType::Image;
        }

        match format {
            // Archive / compression formats.
            MediaFormat::Zip
            | MediaFormat::Tar
            | MediaFormat::Gz
            | MediaFormat::Bz2
            | MediaFormat::Xz
            | MediaFormat::Zstd => MediaType::Archive,

            // Text subtitle formats.
            MediaFormat::Srt | MediaFormat::Vtt | MediaFormat::Ass | MediaFormat::Svg => {
                MediaType::Text
            }

            // Unknown: fall back to binary heuristic.
            MediaFormat::Unknown => {
                if Self::is_binary(data) {
                    MediaType::Binary
                } else {
                    MediaType::Text
                }
            }

            // Anything else that is_video / is_audio / is_image did not catch
            // (should not happen, but provide a safe default).
            _ => MediaType::Binary,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TextEncoding::Utf8 via BOM ────────────────────────────────────────────

    #[test]
    fn test_encoding_utf8_bom() {
        let data = b"\xEF\xBB\xBFHello, world!";
        assert_eq!(ContentDetector::detect_encoding(data), TextEncoding::Utf8);
    }

    #[test]
    fn test_encoding_utf8_no_bom() {
        let data = "Hello, world! ✓ café".as_bytes();
        assert_eq!(ContentDetector::detect_encoding(data), TextEncoding::Utf8);
    }

    // ── TextEncoding::Utf16Le / Utf16Be via BOM ───────────────────────────────

    #[test]
    fn test_encoding_utf16_le_bom() {
        let data = b"\xFF\xFE\x48\x00\x65\x00"; // LE BOM + "He" in UTF-16LE
        assert_eq!(
            ContentDetector::detect_encoding(data),
            TextEncoding::Utf16Le
        );
    }

    #[test]
    fn test_encoding_utf16_be_bom() {
        let data = b"\xFE\xFF\x00\x48\x00\x65"; // BE BOM + "He" in UTF-16BE
        assert_eq!(
            ContentDetector::detect_encoding(data),
            TextEncoding::Utf16Be
        );
    }

    // ── TextEncoding::Ascii ───────────────────────────────────────────────────

    #[test]
    fn test_encoding_ascii_printable() {
        let data = b"Hello World 123";
        assert_eq!(ContentDetector::detect_encoding(data), TextEncoding::Ascii);
    }

    #[test]
    fn test_encoding_ascii_with_crlf() {
        let data = b"line1\r\nline2\r\n";
        assert_eq!(ContentDetector::detect_encoding(data), TextEncoding::Ascii);
    }

    #[test]
    fn test_encoding_ascii_with_tab() {
        let data = b"col1\tcol2\tcol3";
        assert_eq!(ContentDetector::detect_encoding(data), TextEncoding::Ascii);
    }

    #[test]
    fn test_encoding_empty_returns_ascii() {
        assert_eq!(ContentDetector::detect_encoding(&[]), TextEncoding::Ascii);
    }

    // ── TextEncoding::Latin1 ──────────────────────────────────────────────────

    #[test]
    fn test_encoding_latin1_extended_bytes() {
        // Invalid UTF-8 but valid Latin-1
        let data = b"Caf\xe9 au lait"; // 0xE9 = 'é' in Latin-1, but invalid UTF-8 here
        assert_eq!(ContentDetector::detect_encoding(data), TextEncoding::Latin1);
    }

    #[test]
    fn test_encoding_latin1_high_bytes() {
        let data = &[0x80u8, 0x9F, 0xA0, 0xFF];
        assert_eq!(ContentDetector::detect_encoding(data), TextEncoding::Latin1);
    }

    // ── is_binary ─────────────────────────────────────────────────────────────

    #[test]
    fn test_is_binary_empty() {
        assert!(!ContentDetector::is_binary(&[]));
    }

    #[test]
    fn test_is_binary_plain_text() {
        let text = b"This is plain ASCII text.\nNo binary bytes here.\n";
        assert!(!ContentDetector::is_binary(text));
    }

    #[test]
    fn test_is_binary_null_bytes() {
        // 5 null bytes in 100 bytes = 5% > 1% threshold.
        let mut data = vec![0x41u8; 100]; // 'A' * 100
        data[10] = 0x00;
        data[20] = 0x00;
        data[30] = 0x00;
        data[40] = 0x00;
        data[50] = 0x00;
        assert!(ContentDetector::is_binary(&data));
    }

    #[test]
    fn test_is_binary_jpeg_magic() {
        // JPEG files start with 0xFF 0xD8 — binary.
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert!(ContentDetector::is_binary(&data));
    }

    #[test]
    fn test_is_binary_utf8_text() {
        let text = "The quick brown fox jumps over the lazy dog. 1234567890!".as_bytes();
        assert!(!ContentDetector::is_binary(text));
    }

    // ── detect_media_type ─────────────────────────────────────────────────────

    #[test]
    fn test_media_type_empty_returns_unknown() {
        assert_eq!(ContentDetector::detect_media_type(&[]), MediaType::Unknown);
    }

    #[test]
    fn test_media_type_jpeg_is_image() {
        // JPEG magic: FF D8
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46];
        assert_eq!(ContentDetector::detect_media_type(&data), MediaType::Image);
    }

    #[test]
    fn test_media_type_png_is_image() {
        // PNG magic: 89 50 4E 47 0D 0A 1A 0A
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(ContentDetector::detect_media_type(&data), MediaType::Image);
    }

    #[test]
    fn test_media_type_flac_is_audio() {
        // FLAC magic: 66 4C 61 43 = "fLaC"
        let data = b"fLaC\x00\x00\x00\x22";
        assert_eq!(ContentDetector::detect_media_type(data), MediaType::Audio);
    }

    #[test]
    fn test_media_type_wav_is_audio() {
        // WAV: RIFF....WAVE
        let data = b"RIFF\x00\x00\x00\x00WAVE";
        assert_eq!(ContentDetector::detect_media_type(data), MediaType::Audio);
    }

    #[test]
    fn test_media_type_zip_is_archive() {
        // ZIP magic: 50 4B 03 04
        let data = [0x50, 0x4B, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00];
        assert_eq!(
            ContentDetector::detect_media_type(&data),
            MediaType::Archive
        );
    }

    #[test]
    fn test_media_type_gz_is_archive() {
        // GZ magic: 1F 8B
        let data = [0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(
            ContentDetector::detect_media_type(&data),
            MediaType::Archive
        );
    }

    #[test]
    fn test_media_type_unknown_binary_is_binary() {
        // Random binary data that doesn't match any format.
        let data = [
            0x00u8, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(ContentDetector::detect_media_type(&data), MediaType::Binary);
    }

    #[test]
    fn test_media_type_unknown_text_is_text() {
        // Plain text that doesn't match any media format.
        let data = b"1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n";
        // FormatDetector will return Unknown for SRT-like content.
        let mt = ContentDetector::detect_media_type(data);
        assert!(
            matches!(mt, MediaType::Text | MediaType::Unknown),
            "expected Text or Unknown, got {mt:?}"
        );
    }

    // ── MediaType display ─────────────────────────────────────────────────────

    #[test]
    fn test_media_type_display() {
        assert_eq!(MediaType::Video.to_string(), "Video");
        assert_eq!(MediaType::Audio.to_string(), "Audio");
        assert_eq!(MediaType::Image.to_string(), "Image");
        assert_eq!(MediaType::Archive.to_string(), "Archive");
        assert_eq!(MediaType::Text.to_string(), "Text");
        assert_eq!(MediaType::Binary.to_string(), "Binary");
        assert_eq!(MediaType::Unknown.to_string(), "Unknown");
    }

    // ── TextEncoding display ──────────────────────────────────────────────────

    #[test]
    fn test_text_encoding_display() {
        assert_eq!(TextEncoding::Utf8.to_string(), "UTF-8");
        assert_eq!(TextEncoding::Utf16Le.to_string(), "UTF-16LE");
        assert_eq!(TextEncoding::Utf16Be.to_string(), "UTF-16BE");
        assert_eq!(TextEncoding::Latin1.to_string(), "Latin-1");
        assert_eq!(TextEncoding::Ascii.to_string(), "ASCII");
    }
}
