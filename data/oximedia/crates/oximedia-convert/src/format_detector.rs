//! Format detection for media files based on file extension and magic bytes.
//!
//! Provides a `FormatDetector` that identifies `MediaFormat` from a file path
//! or raw byte header, along with `FormatInfo` describing format capabilities.

#![allow(dead_code)]

/// Enumeration of well-known media formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaFormat {
    // ── Video containers ──────────────────────────────────────────────────
    Mp4,
    Mkv,
    Avi,
    Mov,
    Webm,
    Flv,
    Ts,
    M2ts,
    Mxf,
    // ── Audio containers ──────────────────────────────────────────────────
    Mp3,
    Aac,
    Flac,
    Ogg,
    Wav,
    Aiff,
    Opus,
    // ── Image formats ─────────────────────────────────────────────────────
    Jpeg,
    Png,
    Tiff,
    Webp,
    Dpx,
    Exr,
    // ── Unknown ───────────────────────────────────────────────────────────
    Unknown,
}

impl MediaFormat {
    /// Returns `true` if this format is primarily a video container.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            Self::Mp4
                | Self::Mkv
                | Self::Avi
                | Self::Mov
                | Self::Webm
                | Self::Flv
                | Self::Ts
                | Self::M2ts
                | Self::Mxf
        )
    }

    /// Returns `true` if this format is primarily an audio container.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            Self::Mp3 | Self::Aac | Self::Flac | Self::Ogg | Self::Wav | Self::Aiff | Self::Opus
        )
    }

    /// Returns `true` if this format is an image format.
    #[must_use]
    pub fn is_image(&self) -> bool {
        matches!(
            self,
            Self::Jpeg | Self::Png | Self::Tiff | Self::Webp | Self::Dpx | Self::Exr
        )
    }

    /// Canonical file extension for this format (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::Avi => "avi",
            Self::Mov => "mov",
            Self::Webm => "webm",
            Self::Flv => "flv",
            Self::Ts => "ts",
            Self::M2ts => "m2ts",
            Self::Mxf => "mxf",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
            Self::Wav => "wav",
            Self::Aiff => "aiff",
            Self::Opus => "opus",
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Tiff => "tiff",
            Self::Webp => "webp",
            Self::Dpx => "dpx",
            Self::Exr => "exr",
            Self::Unknown => "bin",
        }
    }

    /// MIME type string for this format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Mp4 => "video/mp4",
            Self::Mkv => "video/x-matroska",
            Self::Avi => "video/x-msvideo",
            Self::Mov => "video/quicktime",
            Self::Webm => "video/webm",
            Self::Flv => "video/x-flv",
            Self::Ts | Self::M2ts => "video/mp2t",
            Self::Mxf => "application/mxf",
            Self::Mp3 => "audio/mpeg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Ogg => "audio/ogg",
            Self::Wav => "audio/wav",
            Self::Aiff => "audio/aiff",
            Self::Opus => "audio/opus",
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Tiff => "image/tiff",
            Self::Webp => "image/webp",
            Self::Dpx => "image/x-dpx",
            Self::Exr => "image/x-exr",
            Self::Unknown => "application/octet-stream",
        }
    }
}

/// Metadata / capability descriptor for a `MediaFormat`.
#[derive(Debug, Clone, PartialEq)]
pub struct FormatInfo {
    /// The format this info relates to.
    pub format: MediaFormat,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// Whether the format stores data without lossy compression.
    pub lossless: bool,
    /// Whether the format supports embedded subtitles.
    pub supports_subtitles: bool,
    /// Whether the format supports multiple audio tracks.
    pub supports_multi_audio: bool,
    /// Maximum supported bit depth (for image/video formats).
    pub max_bit_depth: u8,
}

impl FormatInfo {
    /// Returns `true` if this format is strictly lossless.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.lossless
    }
}

/// Detects the `MediaFormat` of a media asset.
#[derive(Debug, Clone, Default)]
pub struct FormatDetector;

impl FormatDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detect format from a file extension string (with or without a leading
    /// dot).  Case-insensitive.
    #[must_use]
    pub fn detect_from_extension(&self, ext: &str) -> MediaFormat {
        let ext = ext.trim_start_matches('.').to_ascii_lowercase();
        match ext.as_str() {
            "mp4" | "m4v" | "m4a" => MediaFormat::Mp4,
            "mkv" | "webm_mkv" => MediaFormat::Mkv,
            "avi" => MediaFormat::Avi,
            "mov" | "qt" => MediaFormat::Mov,
            "webm" => MediaFormat::Webm,
            "flv" => MediaFormat::Flv,
            "ts" | "mts" => MediaFormat::Ts,
            "m2ts" | "mts2" => MediaFormat::M2ts,
            "mxf" => MediaFormat::Mxf,
            "mp3" | "mp2" => MediaFormat::Mp3,
            "aac" | "m4a_aac" => MediaFormat::Aac,
            "flac" => MediaFormat::Flac,
            "ogg" | "oga" => MediaFormat::Ogg,
            "wav" | "wave" => MediaFormat::Wav,
            "aif" | "aiff" => MediaFormat::Aiff,
            "opus" => MediaFormat::Opus,
            "jpg" | "jpeg" => MediaFormat::Jpeg,
            "png" => MediaFormat::Png,
            "tif" | "tiff" => MediaFormat::Tiff,
            "webp" => MediaFormat::Webp,
            "dpx" => MediaFormat::Dpx,
            "exr" => MediaFormat::Exr,
            _ => MediaFormat::Unknown,
        }
    }

    /// Detect format from a byte slice representing the file header (magic
    /// bytes).  At least 12 bytes should be provided for reliable detection.
    #[must_use]
    pub fn detect_from_header(&self, header: &[u8]) -> MediaFormat {
        if header.len() < 4 {
            return MediaFormat::Unknown;
        }

        // MP4 / MOV / M4V / ISOM: ftyp box at offset 4.
        if header.len() >= 8 && &header[4..8] == b"ftyp" {
            // Distinguish MOV vs MP4 by brand.
            if header.len() >= 12 {
                let brand = &header[8..12];
                if brand == b"qt  " {
                    return MediaFormat::Mov;
                }
            }
            return MediaFormat::Mp4;
        }

        // Matroska / WebM.
        if header.len() >= 4 && header[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            return MediaFormat::Mkv;
        }

        // AVI: RIFF....AVI
        if header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"AVI " {
            return MediaFormat::Avi;
        }

        // FLV.
        if header.len() >= 3 && &header[0..3] == b"FLV" {
            return MediaFormat::Flv;
        }

        // MPEG-TS: sync byte 0x47 at offset 0, 188, 376...
        if header[0] == 0x47 {
            return MediaFormat::Ts;
        }

        // WAV: RIFF....WAVE
        if header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"WAVE" {
            return MediaFormat::Wav;
        }

        // FLAC: fLaC
        if header.len() >= 4 && &header[0..4] == b"fLaC" {
            return MediaFormat::Flac;
        }

        // OGG: OggS
        if header.len() >= 4 && &header[0..4] == b"OggS" {
            return MediaFormat::Ogg;
        }

        // MP3: ID3 tag or MPEG frame sync.
        if header.len() >= 3 && &header[0..3] == b"ID3" {
            return MediaFormat::Mp3;
        }
        if header.len() >= 2 && header[0] == 0xFF && (header[1] & 0xE0 == 0xE0) {
            return MediaFormat::Mp3;
        }

        // JPEG: FFD8FF
        if header.len() >= 3 && header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF {
            return MediaFormat::Jpeg;
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if header.len() >= 8 && header[0..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
            return MediaFormat::Png;
        }

        // TIFF: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
        if header.len() >= 4
            && ((&header[0..4] == b"II\x2A\x00") || (&header[0..4] == b"MM\x00\x2A"))
        {
            return MediaFormat::Tiff;
        }

        // RIFF WEBP
        if header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"WEBP" {
            return MediaFormat::Webp;
        }

        // DPX: SDPX or XPDS
        if header.len() >= 4 && (&header[0..4] == b"SDPX" || &header[0..4] == b"XPDS") {
            return MediaFormat::Dpx;
        }

        // OpenEXR: 76 2F 31 01
        if header.len() >= 4 && header[0..4] == [0x76, 0x2F, 0x31, 0x01] {
            return MediaFormat::Exr;
        }

        MediaFormat::Unknown
    }

    /// Retrieve a `FormatInfo` descriptor for the given format.
    #[must_use]
    pub fn format_info(&self, format: MediaFormat) -> FormatInfo {
        match format {
            MediaFormat::Mp4 => FormatInfo {
                format,
                display_name: "MPEG-4",
                lossless: false,
                supports_subtitles: true,
                supports_multi_audio: true,
                max_bit_depth: 10,
            },
            MediaFormat::Mkv => FormatInfo {
                format,
                display_name: "Matroska",
                lossless: false,
                supports_subtitles: true,
                supports_multi_audio: true,
                max_bit_depth: 12,
            },
            MediaFormat::Flac => FormatInfo {
                format,
                display_name: "FLAC",
                lossless: true,
                supports_subtitles: false,
                supports_multi_audio: false,
                max_bit_depth: 32,
            },
            MediaFormat::Wav => FormatInfo {
                format,
                display_name: "WAV",
                lossless: true,
                supports_subtitles: false,
                supports_multi_audio: false,
                max_bit_depth: 32,
            },
            MediaFormat::Tiff => FormatInfo {
                format,
                display_name: "TIFF",
                lossless: true,
                supports_subtitles: false,
                supports_multi_audio: false,
                max_bit_depth: 16,
            },
            MediaFormat::Dpx => FormatInfo {
                format,
                display_name: "DPX",
                lossless: true,
                supports_subtitles: false,
                supports_multi_audio: false,
                max_bit_depth: 16,
            },
            MediaFormat::Exr => FormatInfo {
                format,
                display_name: "OpenEXR",
                lossless: true,
                supports_subtitles: false,
                supports_multi_audio: false,
                max_bit_depth: 32,
            },
            _ => FormatInfo {
                format,
                display_name: "Unknown",
                lossless: false,
                supports_subtitles: false,
                supports_multi_audio: false,
                max_bit_depth: 8,
            },
        }
    }
}

// ── Fast magic-byte detection ─────────────────────────────────────────────────

/// Identify a [`MediaFormat`] by inspecting only the first **16 bytes** of a
/// file header.  This is substantially faster than reading the whole file
/// because no seek or full I/O is required.
///
/// Returns `None` when fewer than 4 bytes are supplied or the signature is not
/// recognised.
///
/// Supported signatures (in priority order):
///
/// | Format | Signature |
/// |--------|-----------|
/// | MP4/MOV | `ftyp` at offset 4 |
/// | MKV/WebM | `\x1A\x45\xDF\xA3` |
/// | AVI | `RIFF` + `AVI ` |
/// | WAV | `RIFF` + `WAVE` |
/// | FLAC | `fLaC` |
/// | OGG | `OggS` |
/// | MP3 | `ID3` or `\xFF\xFB` sync |
/// | JPEG | `\xFF\xD8\xFF` |
/// | PNG | `\x89PNG\r\n\x1A\n` |
#[must_use]
pub fn detect_by_magic_bytes(buf: &[u8]) -> Option<MediaFormat> {
    if buf.len() < 4 {
        return None;
    }

    // MP4 / MOV: ISO Base Media file — "ftyp" box at bytes 4–7.
    // We need at least 8 bytes for this check.
    if buf.len() >= 8 && &buf[4..8] == b"ftyp" {
        // Further disambiguate MOV vs MP4 by the major brand (bytes 8–11).
        if buf.len() >= 12 && &buf[8..12] == b"qt  " {
            return Some(MediaFormat::Mov);
        }
        return Some(MediaFormat::Mp4);
    }

    // Matroska / WebM: EBML header magic.
    if buf[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        // Both Matroska and WebM share the same EBML header; report as Mkv
        // (the distinction requires reading the DocType element further in).
        return Some(MediaFormat::Mkv);
    }

    // RIFF-based formats — need at least 12 bytes to read the sub-type.
    if buf.len() >= 12 && &buf[0..4] == b"RIFF" {
        let sub_type = &buf[8..12];
        if sub_type == b"AVI " {
            return Some(MediaFormat::Avi);
        }
        if sub_type == b"WAVE" {
            return Some(MediaFormat::Wav);
        }
    }

    // FLAC
    if &buf[0..4] == b"fLaC" {
        return Some(MediaFormat::Flac);
    }

    // OGG
    if &buf[0..4] == b"OggS" {
        return Some(MediaFormat::Ogg);
    }

    // MP3 — ID3 tag header
    if buf.len() >= 3 && &buf[0..3] == b"ID3" {
        return Some(MediaFormat::Mp3);
    }

    // MP3 — raw MPEG frame sync bytes (0xFF 0xFB / 0xFF 0xFA / 0xFF 0xF3 …)
    // The sync word is 0xFF followed by a byte with bits 7–5 all set.
    if buf.len() >= 2 && buf[0] == 0xFF && (buf[1] & 0xE0 == 0xE0) {
        return Some(MediaFormat::Mp3);
    }

    // JPEG — SOI marker
    if buf.len() >= 3 && buf[0] == 0xFF && buf[1] == 0xD8 && buf[2] == 0xFF {
        return Some(MediaFormat::Jpeg);
    }

    // PNG — 8-byte signature
    if buf.len() >= 8 && buf[0..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some(MediaFormat::Png);
    }

    None
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> FormatDetector {
        FormatDetector::new()
    }

    #[test]
    fn test_is_video_mp4() {
        assert!(MediaFormat::Mp4.is_video());
    }

    #[test]
    fn test_is_audio_wav() {
        assert!(MediaFormat::Wav.is_audio());
    }

    #[test]
    fn test_is_image_png() {
        assert!(MediaFormat::Png.is_image());
    }

    #[test]
    fn test_extension_mp4() {
        assert_eq!(MediaFormat::Mp4.extension(), "mp4");
    }

    #[test]
    fn test_extension_jpeg() {
        assert_eq!(MediaFormat::Jpeg.extension(), "jpg");
    }

    #[test]
    fn test_mime_type_webm() {
        assert_eq!(MediaFormat::Webm.mime_type(), "video/webm");
    }

    #[test]
    fn test_detect_from_extension_mp4() {
        let d = detector();
        assert_eq!(d.detect_from_extension("mp4"), MediaFormat::Mp4);
        assert_eq!(d.detect_from_extension(".MP4"), MediaFormat::Mp4);
    }

    #[test]
    fn test_detect_from_extension_unknown() {
        let d = detector();
        assert_eq!(d.detect_from_extension("xyz"), MediaFormat::Unknown);
    }

    #[test]
    fn test_detect_from_extension_flac() {
        let d = detector();
        assert_eq!(d.detect_from_extension("flac"), MediaFormat::Flac);
    }

    #[test]
    fn test_detect_from_header_png() {
        let d = detector();
        let header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(d.detect_from_header(&header), MediaFormat::Png);
    }

    #[test]
    fn test_detect_from_header_jpeg() {
        let d = detector();
        let header = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert_eq!(d.detect_from_header(&header), MediaFormat::Jpeg);
    }

    #[test]
    fn test_detect_from_header_flac() {
        let d = detector();
        let header = b"fLaC\x00\x00\x00\x22";
        assert_eq!(d.detect_from_header(header), MediaFormat::Flac);
    }

    #[test]
    fn test_detect_from_header_mkv() {
        let d = detector();
        let header = [0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00];
        assert_eq!(d.detect_from_header(&header), MediaFormat::Mkv);
    }

    #[test]
    fn test_detect_from_header_too_short() {
        let d = detector();
        assert_eq!(d.detect_from_header(&[0x00, 0x01]), MediaFormat::Unknown);
    }

    #[test]
    fn test_format_info_lossless_flac() {
        let d = detector();
        let info = d.format_info(MediaFormat::Flac);
        assert!(info.is_lossless());
    }

    #[test]
    fn test_format_info_not_lossless_mp4() {
        let d = detector();
        let info = d.format_info(MediaFormat::Mp4);
        assert!(!info.is_lossless());
    }

    #[test]
    fn test_format_info_supports_subtitles_mkv() {
        let d = detector();
        let info = d.format_info(MediaFormat::Mkv);
        assert!(info.supports_subtitles);
    }

    // ── detect_by_magic_bytes ─────────────────────────────────────────────────

    #[test]
    fn magic_mp4_ftyp() {
        // Minimal MP4 header: 4 size bytes then "ftyp" then brand "isom"
        let buf = [
            0x00, 0x00, 0x00, 0x20, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm', 0x00, 0x00,
            0x00, 0x01,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Mp4));
    }

    #[test]
    fn magic_mov_qt_brand() {
        let buf = [
            0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p', b'q', b't', b' ', b' ', 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Mov));
    }

    #[test]
    fn magic_mkv_ebml_header() {
        let buf = [
            0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x42, 0x86,
            0x81, 0x01,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Mkv));
    }

    #[test]
    fn magic_webm_same_as_mkv() {
        // WebM uses the same EBML magic as Matroska; both detected as Mkv
        let buf = [
            0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Mkv));
    }

    #[test]
    fn magic_avi_riff_avi() {
        let buf = [
            b'R', b'I', b'F', b'F', 0x24, 0x00, 0x00, 0x00, b'A', b'V', b'I', b' ', 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Avi));
    }

    #[test]
    fn magic_wav_riff_wave() {
        let buf = [
            b'R', b'I', b'F', b'F', 0x24, 0x00, 0x00, 0x00, b'W', b'A', b'V', b'E', 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Wav));
    }

    #[test]
    fn magic_flac() {
        let buf = [
            b'f', b'L', b'a', b'C', 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x0A, 0xC4, 0x00, 0x00,
            0x0A, 0xC4,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Flac));
    }

    #[test]
    fn magic_ogg() {
        let buf = [
            b'O', b'g', b'g', b'S', 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Ogg));
    }

    #[test]
    fn magic_mp3_id3() {
        let buf = [
            b'I', b'D', b'3', 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Mp3));
    }

    #[test]
    fn magic_mp3_sync_bytes() {
        // 0xFF 0xFB = MPEG-1 Layer 3, no CRC, 128 kbps
        let buf = [
            0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Mp3));
    }

    #[test]
    fn magic_jpeg() {
        let buf = [
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Jpeg));
    }

    #[test]
    fn magic_png() {
        let buf = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52,
        ];
        assert_eq!(detect_by_magic_bytes(&buf), Some(MediaFormat::Png));
    }

    #[test]
    fn magic_too_short_returns_none() {
        assert_eq!(detect_by_magic_bytes(&[0xFF, 0xD8]), None);
    }

    #[test]
    fn magic_unknown_data_returns_none() {
        let buf = [0x00u8; 16];
        assert_eq!(detect_by_magic_bytes(&buf), None);
    }
}
