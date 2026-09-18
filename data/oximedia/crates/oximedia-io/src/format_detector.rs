#![allow(dead_code)]
//! Magic-byte based media format detection.
//!
//! [`FormatDetector`] inspects the leading bytes of a data buffer to identify
//! the container or codec format, returning a [`FormatDetection`] that includes
//! confidence, MIME type, canonical extension, and a human-readable description.

/// All media/archive formats recognised by this detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaFormat {
    // ── Video containers ─────────────────────────────────────────────────────
    /// MPEG-4 Part 14 container (`ftyp` box).
    Mp4,
    /// Matroska video container.
    Mkv,
    /// Audio-Video Interleave (Microsoft).
    Avi,
    /// QuickTime movie container.
    Mov,
    /// WebM (subset of Matroska).
    Webm,
    /// Flash Video container.
    Flv,
    /// MPEG-2 Transport Stream.
    Ts,
    /// Blu-ray MPEG-2 Transport Stream.
    M2ts,
    /// Material Exchange Format.
    Mxf,
    /// Ogg container.
    Ogg,
    // ── Audio ─────────────────────────────────────────────────────────────────
    /// MPEG Audio Layer III.
    Mp3,
    /// Free Lossless Audio Codec.
    Flac,
    /// Waveform Audio File Format.
    Wav,
    /// Advanced Audio Coding (raw ADTS stream).
    Aac,
    /// Opus audio (inside Ogg).
    Opus,
    /// Vorbis audio (inside Ogg).
    Vorbis,
    /// Audio Interchange File Format.
    Aiff,
    /// Sun/NeXT audio format.
    Au,
    // ── Image ─────────────────────────────────────────────────────────────────
    /// JPEG / JFIF image.
    Jpeg,
    /// Portable Network Graphics.
    Png,
    /// Graphics Interchange Format.
    Gif,
    /// WebP image.
    Webp,
    /// Windows Bitmap.
    Bmp,
    /// Tagged Image File Format.
    Tiff,
    /// Scalable Vector Graphics (XML text).
    Svg,
    /// High Efficiency Image Container.
    Heic,
    /// AV1 Image File Format.
    Avif,
    // ── Subtitle ──────────────────────────────────────────────────────────────
    /// SubRip Text subtitles.
    Srt,
    /// Web Video Text Tracks.
    Vtt,
    /// Advanced SubStation Alpha subtitles.
    Ass,
    // ── Archive ───────────────────────────────────────────────────────────────
    /// ZIP archive.
    Zip,
    /// Unix tape-archive (uncompressed).
    Tar,
    /// Gzip-compressed data.
    Gz,
    /// Bzip2-compressed data.
    Bz2,
    /// XZ-compressed data.
    Xz,
    /// Zstandard-compressed data.
    Zstd,
    // ── Professional / Cinema ───────────────────────────────────────────────
    /// Digital Picture Exchange.
    Dpx,
    /// OpenEXR high dynamic range image.
    Exr,
    /// Digital Negative (Adobe DNG, based on TIFF).
    Dng,
    /// JPEG XL image.
    Jxl,
    /// Y4M (YUV4MPEG2) raw video.
    Y4m,
    /// FFV1 video (raw bitstream, Matroska-wrapped detection via MKV).
    Ffv1,
    /// CAF (Core Audio Format).
    Caf,
    /// MPEG Program Stream.
    Ps,
    // ── Fallback ──────────────────────────────────────────────────────────────
    /// Format could not be determined.
    Unknown,
}

impl MediaFormat {
    /// Returns `true` for video-container formats.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            MediaFormat::Mp4
                | MediaFormat::Mkv
                | MediaFormat::Avi
                | MediaFormat::Mov
                | MediaFormat::Webm
                | MediaFormat::Flv
                | MediaFormat::Ts
                | MediaFormat::M2ts
                | MediaFormat::Mxf
                | MediaFormat::Ogg
                | MediaFormat::Y4m
                | MediaFormat::Ffv1
                | MediaFormat::Ps
        )
    }

    /// Returns `true` for audio formats.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            MediaFormat::Mp3
                | MediaFormat::Flac
                | MediaFormat::Wav
                | MediaFormat::Aac
                | MediaFormat::Opus
                | MediaFormat::Vorbis
                | MediaFormat::Aiff
                | MediaFormat::Au
                | MediaFormat::Caf
        )
    }

    /// Returns `true` for image formats.
    #[must_use]
    pub fn is_image(&self) -> bool {
        matches!(
            self,
            MediaFormat::Jpeg
                | MediaFormat::Png
                | MediaFormat::Gif
                | MediaFormat::Webp
                | MediaFormat::Bmp
                | MediaFormat::Tiff
                | MediaFormat::Svg
                | MediaFormat::Heic
                | MediaFormat::Avif
                | MediaFormat::Dpx
                | MediaFormat::Exr
                | MediaFormat::Dng
                | MediaFormat::Jxl
        )
    }

    /// Returns the canonical file extension (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            MediaFormat::Mp4 => "mp4",
            MediaFormat::Mkv => "mkv",
            MediaFormat::Avi => "avi",
            MediaFormat::Mov => "mov",
            MediaFormat::Webm => "webm",
            MediaFormat::Flv => "flv",
            MediaFormat::Ts => "ts",
            MediaFormat::M2ts => "m2ts",
            MediaFormat::Mxf => "mxf",
            MediaFormat::Ogg => "ogg",
            MediaFormat::Mp3 => "mp3",
            MediaFormat::Flac => "flac",
            MediaFormat::Wav => "wav",
            MediaFormat::Aac => "aac",
            MediaFormat::Opus => "opus",
            MediaFormat::Vorbis => "ogg",
            MediaFormat::Aiff => "aiff",
            MediaFormat::Au => "au",
            MediaFormat::Jpeg => "jpg",
            MediaFormat::Png => "png",
            MediaFormat::Gif => "gif",
            MediaFormat::Webp => "webp",
            MediaFormat::Bmp => "bmp",
            MediaFormat::Tiff => "tiff",
            MediaFormat::Svg => "svg",
            MediaFormat::Heic => "heic",
            MediaFormat::Avif => "avif",
            MediaFormat::Srt => "srt",
            MediaFormat::Vtt => "vtt",
            MediaFormat::Ass => "ass",
            MediaFormat::Zip => "zip",
            MediaFormat::Tar => "tar",
            MediaFormat::Gz => "gz",
            MediaFormat::Bz2 => "bz2",
            MediaFormat::Xz => "xz",
            MediaFormat::Zstd => "zst",
            MediaFormat::Dpx => "dpx",
            MediaFormat::Exr => "exr",
            MediaFormat::Dng => "dng",
            MediaFormat::Jxl => "jxl",
            MediaFormat::Y4m => "y4m",
            MediaFormat::Ffv1 => "ffv1",
            MediaFormat::Caf => "caf",
            MediaFormat::Ps => "mpg",
            MediaFormat::Unknown => "bin",
        }
    }

    /// Returns the primary MIME type for this format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            MediaFormat::Mp4 => "video/mp4",
            MediaFormat::Mkv => "video/x-matroska",
            MediaFormat::Avi => "video/x-msvideo",
            MediaFormat::Mov => "video/quicktime",
            MediaFormat::Webm => "video/webm",
            MediaFormat::Flv => "video/x-flv",
            MediaFormat::Ts => "video/mp2t",
            MediaFormat::M2ts => "video/mp2t",
            MediaFormat::Mxf => "application/mxf",
            MediaFormat::Ogg => "video/ogg",
            MediaFormat::Mp3 => "audio/mpeg",
            MediaFormat::Flac => "audio/flac",
            MediaFormat::Wav => "audio/wav",
            MediaFormat::Aac => "audio/aac",
            MediaFormat::Opus => "audio/ogg; codecs=opus",
            MediaFormat::Vorbis => "audio/ogg; codecs=vorbis",
            MediaFormat::Aiff => "audio/aiff",
            MediaFormat::Au => "audio/basic",
            MediaFormat::Jpeg => "image/jpeg",
            MediaFormat::Png => "image/png",
            MediaFormat::Gif => "image/gif",
            MediaFormat::Webp => "image/webp",
            MediaFormat::Bmp => "image/bmp",
            MediaFormat::Tiff => "image/tiff",
            MediaFormat::Svg => "image/svg+xml",
            MediaFormat::Heic => "image/heic",
            MediaFormat::Avif => "image/avif",
            MediaFormat::Srt => "text/plain",
            MediaFormat::Vtt => "text/vtt",
            MediaFormat::Ass => "text/x-ssa",
            MediaFormat::Zip => "application/zip",
            MediaFormat::Tar => "application/x-tar",
            MediaFormat::Gz => "application/gzip",
            MediaFormat::Bz2 => "application/x-bzip2",
            MediaFormat::Xz => "application/x-xz",
            MediaFormat::Zstd => "application/zstd",
            MediaFormat::Dpx => "image/x-dpx",
            MediaFormat::Exr => "image/x-exr",
            MediaFormat::Dng => "image/x-adobe-dng",
            MediaFormat::Jxl => "image/jxl",
            MediaFormat::Y4m => "video/x-raw-yuv",
            MediaFormat::Ffv1 => "video/x-ffv1",
            MediaFormat::Caf => "audio/x-caf",
            MediaFormat::Ps => "video/mpeg",
            MediaFormat::Unknown => "application/octet-stream",
        }
    }

    /// Returns a short human-readable description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            MediaFormat::Mp4 => "MPEG-4 Part 14 video container",
            MediaFormat::Mkv => "Matroska video container",
            MediaFormat::Avi => "Audio Video Interleave container",
            MediaFormat::Mov => "Apple QuickTime movie container",
            MediaFormat::Webm => "WebM video container",
            MediaFormat::Flv => "Flash Video container",
            MediaFormat::Ts => "MPEG-2 Transport Stream",
            MediaFormat::M2ts => "Blu-ray MPEG-2 Transport Stream",
            MediaFormat::Mxf => "Material Exchange Format",
            MediaFormat::Ogg => "Ogg multimedia container",
            MediaFormat::Mp3 => "MPEG Audio Layer III",
            MediaFormat::Flac => "Free Lossless Audio Codec",
            MediaFormat::Wav => "Waveform Audio File Format",
            MediaFormat::Aac => "Advanced Audio Coding",
            MediaFormat::Opus => "Opus audio codec",
            MediaFormat::Vorbis => "Vorbis audio codec",
            MediaFormat::Aiff => "Audio Interchange File Format",
            MediaFormat::Au => "Sun/NeXT AU audio",
            MediaFormat::Jpeg => "JPEG image",
            MediaFormat::Png => "Portable Network Graphics image",
            MediaFormat::Gif => "Graphics Interchange Format image",
            MediaFormat::Webp => "WebP image",
            MediaFormat::Bmp => "Windows Bitmap image",
            MediaFormat::Tiff => "Tagged Image File Format",
            MediaFormat::Svg => "Scalable Vector Graphics",
            MediaFormat::Heic => "High Efficiency Image Container",
            MediaFormat::Avif => "AV1 Image File Format",
            MediaFormat::Srt => "SubRip Text subtitles",
            MediaFormat::Vtt => "Web Video Text Tracks",
            MediaFormat::Ass => "Advanced SubStation Alpha subtitles",
            MediaFormat::Zip => "ZIP archive",
            MediaFormat::Tar => "Unix tar archive",
            MediaFormat::Gz => "Gzip compressed data",
            MediaFormat::Bz2 => "Bzip2 compressed data",
            MediaFormat::Xz => "XZ compressed data",
            MediaFormat::Zstd => "Zstandard compressed data",
            MediaFormat::Dpx => "Digital Picture Exchange image",
            MediaFormat::Exr => "OpenEXR high dynamic range image",
            MediaFormat::Dng => "Adobe Digital Negative",
            MediaFormat::Jxl => "JPEG XL image",
            MediaFormat::Y4m => "YUV4MPEG2 raw video",
            MediaFormat::Ffv1 => "FFV1 lossless video codec",
            MediaFormat::Caf => "Core Audio Format",
            MediaFormat::Ps => "MPEG Program Stream",
            MediaFormat::Unknown => "Unknown binary data",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection result
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by [`FormatDetector::detect`].
#[derive(Debug, Clone)]
pub struct FormatDetection {
    /// Identified format.
    pub format: MediaFormat,
    /// Detection confidence in the range `0.0..=1.0`.
    pub confidence: f32,
    /// Primary MIME type string.
    pub mime_type: &'static str,
    /// Canonical file extension (without leading dot).
    pub extension: &'static str,
    /// Short human-readable description of the format.
    pub description: &'static str,
}

impl FormatDetection {
    fn new(format: MediaFormat, confidence: f32) -> Self {
        Self {
            mime_type: format.mime_type(),
            extension: format.extension(),
            description: format.description(),
            format,
            confidence,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless magic-byte based format detector.
#[derive(Debug, Default, Clone)]
pub struct FormatDetector;

impl FormatDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detect the format of `data` by inspecting magic bytes.
    ///
    /// Returns a [`FormatDetection`] with the best guess and a confidence
    /// value (`1.0` = definitive match, lower values indicate partial matches).
    #[must_use]
    pub fn detect(data: &[u8]) -> FormatDetection {
        // Nothing to inspect
        if data.is_empty() {
            return FormatDetection::new(MediaFormat::Unknown, 0.0);
        }

        // ── JPEG ─────────────────────────────────────────────────────────────
        if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
            return FormatDetection::new(MediaFormat::Jpeg, 1.0);
        }

        // ── PNG ───────────────────────────────────────────────────────────────
        if data.len() >= 8 && data[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
            return FormatDetection::new(MediaFormat::Png, 1.0);
        }

        // ── GIF ───────────────────────────────────────────────────────────────
        if data.len() >= 6 && (&data[..6] == b"GIF87a" || &data[..6] == b"GIF89a") {
            return FormatDetection::new(MediaFormat::Gif, 1.0);
        }

        // ── FLAC ──────────────────────────────────────────────────────────────
        if data.len() >= 4 && &data[..4] == b"fLaC" {
            return FormatDetection::new(MediaFormat::Flac, 1.0);
        }

        // ── OGG family (Ogg, Opus, Vorbis) ───────────────────────────────────
        if data.len() >= 4 && &data[..4] == b"OggS" {
            // Peek into the first logical-bitstream page for codec identification.
            // The codec packet header starts at byte 28 inside the page.
            if data.len() >= 36 && &data[28..36] == b"OpusHead" {
                return FormatDetection::new(MediaFormat::Opus, 1.0);
            }
            // Vorbis identification header starts with \x01vorbis
            if data.len() >= 35 && data[28] == 0x01 && &data[29..35] == b"vorbi" {
                return FormatDetection::new(MediaFormat::Vorbis, 1.0);
            }
            return FormatDetection::new(MediaFormat::Ogg, 0.95);
        }

        // ── RIFF-based formats: WAV, AVI, WEBP ───────────────────────────────
        if data.len() >= 12 && &data[..4] == b"RIFF" {
            if &data[8..12] == b"WAVE" {
                return FormatDetection::new(MediaFormat::Wav, 1.0);
            }
            if &data[8..12] == b"AVI " {
                return FormatDetection::new(MediaFormat::Avi, 1.0);
            }
            if &data[8..12] == b"WEBP" {
                return FormatDetection::new(MediaFormat::Webp, 1.0);
            }
            // RIFF but unknown sub-format
            return FormatDetection::new(MediaFormat::Unknown, 0.3);
        }

        // ── FORM-based formats: AIFF ──────────────────────────────────────────
        if data.len() >= 12 && &data[..4] == b"FORM" {
            if &data[8..12] == b"AIFF" || &data[8..12] == b"AIFC" {
                return FormatDetection::new(MediaFormat::Aiff, 1.0);
            }
        }

        // ── MKV / WebM (EBML) ─────────────────────────────────────────────────
        if data.len() >= 4 && data[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            // Scan up to 64 bytes for the DocType element (0x42 0x82) and
            // its string value to distinguish WebM from Matroska.
            let scan_end = data.len().min(128);
            let doctype_marker = [0x42u8, 0x82];
            let mut idx = 4;
            let mut found_webm = false;
            let mut found_mkv = false;
            while idx + 1 < scan_end {
                if data[idx] == doctype_marker[0] && data[idx + 1] == doctype_marker[1] {
                    // Next byte is the size, then the string.
                    if idx + 2 < scan_end {
                        let size = data[idx + 2] as usize;
                        let str_start = idx + 3;
                        let str_end = str_start.saturating_add(size).min(scan_end);
                        if str_end > str_start {
                            let doctype = &data[str_start..str_end];
                            if doctype == b"webm" {
                                found_webm = true;
                                break;
                            } else if doctype == b"matroska" {
                                found_mkv = true;
                                break;
                            }
                        }
                    }
                }
                idx += 1;
            }
            if found_webm {
                return FormatDetection::new(MediaFormat::Webm, 1.0);
            }
            if found_mkv {
                return FormatDetection::new(MediaFormat::Mkv, 1.0);
            }
            // EBML magic but DocType not found yet — default to Matroska.
            return FormatDetection::new(MediaFormat::Mkv, 0.8);
        }

        // ── MP4 / MOV (ISO Base Media) ────────────────────────────────────────
        // The `ftyp` box can appear at offset 0 or after initial mdat boxes.
        // Check the most common case: ftyp at offset 4.
        if let Some(fmt) = Self::probe_isobmff(data) {
            return fmt;
        }

        // ── FLV ───────────────────────────────────────────────────────────────
        if data.len() >= 3 && &data[..3] == b"FLV" {
            return FormatDetection::new(MediaFormat::Flv, 1.0);
        }

        // ── MPEG-TS (sync-byte heuristic) ─────────────────────────────────────
        // 0x47 appears at byte 0, and again at 188 or 192 for M2TS.
        if data.len() >= 192 && data[0] == 0x47 {
            if data[188] == 0x47 {
                return FormatDetection::new(MediaFormat::Ts, 0.9);
            }
            if data[4] == 0x47 {
                return FormatDetection::new(MediaFormat::M2ts, 0.85);
            }
        }
        if data.len() >= 1 && data[0] == 0x47 {
            return FormatDetection::new(MediaFormat::Ts, 0.5);
        }

        // ── MXF ───────────────────────────────────────────────────────────────
        if data.len() >= 4 && data[..4] == [0x06, 0x0E, 0x2B, 0x34] {
            return FormatDetection::new(MediaFormat::Mxf, 1.0);
        }

        // ── MP3 ───────────────────────────────────────────────────────────────
        if data.len() >= 3 && &data[..3] == b"ID3" {
            return FormatDetection::new(MediaFormat::Mp3, 1.0);
        }
        // Sync word: 0xFF 0xFB (MPEG1 Layer3 CBR) or similar frame sync.
        if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0 {
            // Could be MPEG audio; check common Layer-III sync patterns.
            let layer = (data[1] >> 1) & 0x03;
            if layer == 0x01 {
                return FormatDetection::new(MediaFormat::Mp3, 0.9);
            }
        }

        // ── AAC (ADTS) ────────────────────────────────────────────────────────
        if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xF6) == 0xF0 {
            return FormatDetection::new(MediaFormat::Aac, 0.9);
        }

        // ── BMP ───────────────────────────────────────────────────────────────
        if data.len() >= 2 && &data[..2] == b"BM" {
            return FormatDetection::new(MediaFormat::Bmp, 1.0);
        }

        // ── TIFF ──────────────────────────────────────────────────────────────
        if data.len() >= 4 {
            if data[..4] == [0x49, 0x49, 0x2A, 0x00] {
                return FormatDetection::new(MediaFormat::Tiff, 1.0);
            }
            if data[..4] == [0x4D, 0x4D, 0x00, 0x2A] {
                return FormatDetection::new(MediaFormat::Tiff, 1.0);
            }
        }

        // ── ZIP ───────────────────────────────────────────────────────────────
        if data.len() >= 2 && data[..2] == [0x50, 0x4B] {
            return FormatDetection::new(MediaFormat::Zip, 1.0);
        }

        // ── GZIP ──────────────────────────────────────────────────────────────
        if data.len() >= 2 && data[..2] == [0x1F, 0x8B] {
            return FormatDetection::new(MediaFormat::Gz, 1.0);
        }

        // ── Bzip2 ─────────────────────────────────────────────────────────────
        if data.len() >= 3 && &data[..3] == b"BZh" {
            return FormatDetection::new(MediaFormat::Bz2, 1.0);
        }

        // ── XZ ────────────────────────────────────────────────────────────────
        if data.len() >= 6 && data[..6] == [0xFD, b'7', b'z', b'X', b'Z', 0x00] {
            return FormatDetection::new(MediaFormat::Xz, 1.0);
        }

        // ── Zstandard ─────────────────────────────────────────────────────────
        if data.len() >= 4 && data[..4] == [0xFD, 0x2F, 0xB5, 0x28] {
            return FormatDetection::new(MediaFormat::Zstd, 1.0);
        }

        // ── Sun AU ────────────────────────────────────────────────────────────
        if data.len() >= 4 && &data[..4] == b".snd" {
            return FormatDetection::new(MediaFormat::Au, 1.0);
        }

        // ── SVG (XML heuristic) ───────────────────────────────────────────────
        if Self::looks_like_svg(data) {
            return FormatDetection::new(MediaFormat::Svg, 0.85);
        }

        // ── DPX (SMPTE 268M) ──────────────────────────────────────────────────
        // Big-endian magic "SDPX" or little-endian "XPDS"
        if data.len() >= 4 && (&data[..4] == b"SDPX" || &data[..4] == b"XPDS") {
            return FormatDetection::new(MediaFormat::Dpx, 1.0);
        }

        // ── OpenEXR ──────────────────────────────────────────────────────────
        // Magic number 0x76, 0x2F, 0x31, 0x01
        if data.len() >= 4 && data[..4] == [0x76, 0x2F, 0x31, 0x01] {
            return FormatDetection::new(MediaFormat::Exr, 1.0);
        }

        // ── JPEG XL ──────────────────────────────────────────────────────────
        // Codestream: 0xFF 0x0A; Container: 0x00 0x00 0x00 0x0C 0x4A 0x58 0x4C 0x20
        if data.len() >= 2 && data[0] == 0xFF && data[1] == 0x0A {
            return FormatDetection::new(MediaFormat::Jxl, 1.0);
        }
        if data.len() >= 12 && data[..8] == [0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20] {
            return FormatDetection::new(MediaFormat::Jxl, 1.0);
        }

        // ── Y4M (YUV4MPEG2) ─────────────────────────────────────────────────
        if data.len() >= 10 && &data[..10] == b"YUV4MPEG2 " {
            return FormatDetection::new(MediaFormat::Y4m, 1.0);
        }

        // ── CAF (Core Audio Format) ──────────────────────────────────────────
        if data.len() >= 4 && &data[..4] == b"caff" {
            return FormatDetection::new(MediaFormat::Caf, 1.0);
        }

        // ── MPEG Program Stream ──────────────────────────────────────────────
        // Pack start code: 0x00 0x00 0x01 0xBA
        if data.len() >= 4 && data[..4] == [0x00, 0x00, 0x01, 0xBA] {
            return FormatDetection::new(MediaFormat::Ps, 0.9);
        }

        // ── SRT (plain-text subtitle heuristic) ──────────────────────────────
        if Self::looks_like_srt(data) {
            return FormatDetection::new(MediaFormat::Srt, 0.75);
        }

        // ── VTT ───────────────────────────────────────────────────────────────
        if data.len() >= 6 && &data[..6] == b"WEBVTT" {
            return FormatDetection::new(MediaFormat::Vtt, 1.0);
        }

        // ── ASS/SSA subtitle ──────────────────────────────────────────────────
        if Self::looks_like_ass(data) {
            return FormatDetection::new(MediaFormat::Ass, 0.9);
        }

        FormatDetection::new(MediaFormat::Unknown, 0.0)
    }

    /// Scan for an ISO Base Media File Format `ftyp` or `moov` box, which
    /// identifies MP4, MOV, and related containers.
    fn probe_isobmff(data: &[u8]) -> Option<FormatDetection> {
        // Scan first 64 bytes for a box whose type field is `ftyp` or `moov`.
        let scan_end = data.len().min(512);
        let mut offset = 0usize;

        while offset + 8 <= scan_end {
            // Box size is big-endian u32 at `offset`.
            let size = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;

            let box_type = &data[offset + 4..offset + 8];

            if box_type == b"ftyp" {
                // Read the major brand (next 4 bytes).
                if offset + 12 <= data.len() {
                    let brand = &data[offset + 8..offset + 12];
                    if brand == b"qt  " || brand == b"MSNV" {
                        return Some(FormatDetection::new(MediaFormat::Mov, 1.0));
                    }
                }
                return Some(FormatDetection::new(MediaFormat::Mp4, 1.0));
            }

            if box_type == b"moov" {
                return Some(FormatDetection::new(MediaFormat::Mp4, 0.9));
            }

            // Move to the next box; guard against infinite loops / corrupt data.
            if size < 8 {
                break;
            }
            offset = offset.saturating_add(size);
        }

        None
    }

    /// Heuristic: does the data start with an XML/SVG declaration or `<svg`?
    fn looks_like_svg(data: &[u8]) -> bool {
        let s = match std::str::from_utf8(data.get(..256).unwrap_or(data)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let trimmed = s.trim_start();
        trimmed.starts_with("<?xml") && trimmed.contains("<svg") || trimmed.starts_with("<svg")
    }

    /// Heuristic: does the data look like an ASS/SSA subtitle file?
    fn looks_like_ass(data: &[u8]) -> bool {
        let s = match std::str::from_utf8(data.get(..256).unwrap_or(data)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let trimmed = s.trim_start();
        trimmed.starts_with("[Script Info]")
    }

    /// Heuristic: does the data look like an SRT subtitle file?
    /// First non-empty line should be a sequence number (a single integer).
    fn looks_like_srt(data: &[u8]) -> bool {
        let s = match std::str::from_utf8(data.get(..512).unwrap_or(data)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let mut lines = s.lines().filter(|l| !l.trim().is_empty());
        if let Some(first) = lines.next() {
            return first.trim().parse::<u64>().is_ok();
        }
        false
    }

    /// Fallback: determine format from a file extension string (case-insensitive).
    ///
    /// Returns [`MediaFormat::Unknown`] when the extension is unrecognised.
    #[must_use]
    pub fn detect_from_extension(ext: &str) -> MediaFormat {
        match ext.to_ascii_lowercase().trim_start_matches('.') {
            "mp4" | "m4v" | "m4p" | "m4b" => MediaFormat::Mp4,
            "mkv" => MediaFormat::Mkv,
            "avi" => MediaFormat::Avi,
            "mov" | "qt" => MediaFormat::Mov,
            "webm" => MediaFormat::Webm,
            "flv" => MediaFormat::Flv,
            "ts" | "mts" => MediaFormat::Ts,
            "m2ts" | "m2t" => MediaFormat::M2ts,
            "mxf" => MediaFormat::Mxf,
            "ogg" | "ogv" | "ogx" => MediaFormat::Ogg,
            "mp3" => MediaFormat::Mp3,
            "flac" => MediaFormat::Flac,
            "wav" | "wave" => MediaFormat::Wav,
            "aac" => MediaFormat::Aac,
            "opus" => MediaFormat::Opus,
            "oga" => MediaFormat::Vorbis,
            "aiff" | "aif" => MediaFormat::Aiff,
            "au" | "snd" => MediaFormat::Au,
            "jpg" | "jpeg" | "jfif" => MediaFormat::Jpeg,
            "png" => MediaFormat::Png,
            "gif" => MediaFormat::Gif,
            "webp" => MediaFormat::Webp,
            "bmp" | "dib" => MediaFormat::Bmp,
            "tiff" | "tif" => MediaFormat::Tiff,
            "svg" | "svgz" => MediaFormat::Svg,
            "heic" | "heif" => MediaFormat::Heic,
            "avif" => MediaFormat::Avif,
            "srt" => MediaFormat::Srt,
            "vtt" => MediaFormat::Vtt,
            "ass" | "ssa" => MediaFormat::Ass,
            "zip" => MediaFormat::Zip,
            "tar" => MediaFormat::Tar,
            "gz" | "gzip" => MediaFormat::Gz,
            "bz2" | "bzip2" => MediaFormat::Bz2,
            "xz" => MediaFormat::Xz,
            "zst" | "zstd" => MediaFormat::Zstd,
            "dpx" => MediaFormat::Dpx,
            "exr" => MediaFormat::Exr,
            "dng" => MediaFormat::Dng,
            "jxl" => MediaFormat::Jxl,
            "y4m" => MediaFormat::Y4m,
            "ffv1" => MediaFormat::Ffv1,
            "caf" => MediaFormat::Caf,
            "mpg" | "mpeg" | "vob" => MediaFormat::Ps,
            _ => MediaFormat::Unknown,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: build a minimal ftyp box ──────────────────────────────────────
    fn mp4_ftyp_bytes() -> Vec<u8> {
        let mut v = vec![0u8, 0, 0, 20]; // box size = 20
        v.extend_from_slice(b"ftyp");
        v.extend_from_slice(b"isom"); // major brand
        v.extend_from_slice(b"\x00\x00\x02\x00"); // minor version
        v.extend_from_slice(b"isom"); // compatible brand
        v
    }

    fn mov_ftyp_bytes() -> Vec<u8> {
        let mut v = vec![0u8, 0, 0, 16];
        v.extend_from_slice(b"ftyp");
        v.extend_from_slice(b"qt  "); // QuickTime brand
        v.extend_from_slice(b"\x00\x00\x00\x00");
        v
    }

    fn mkv_bytes() -> Vec<u8> {
        // Minimal EBML header with DocType = "matroska"
        let mut v: Vec<u8> = vec![0x1A, 0x45, 0xDF, 0xA3]; // EBML ID
                                                           // Pad to give scanner room
        v.extend_from_slice(&[0x00u8; 10]);
        v.push(0x42);
        v.push(0x82);
        v.push(8); // size
        v.extend_from_slice(b"matroska");
        v
    }

    fn webm_bytes() -> Vec<u8> {
        let mut v: Vec<u8> = vec![0x1A, 0x45, 0xDF, 0xA3];
        v.extend_from_slice(&[0x00u8; 10]);
        v.push(0x42);
        v.push(0x82);
        v.push(4);
        v.extend_from_slice(b"webm");
        v
    }

    // ── Video ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_detect_mp4() {
        let data = mp4_ftyp_bytes();
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Mp4);
        assert!(det.confidence >= 0.9);
    }

    #[test]
    fn test_detect_mov() {
        let data = mov_ftyp_bytes();
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Mov);
        assert!(det.confidence >= 0.9);
    }

    #[test]
    fn test_detect_mkv() {
        let data = mkv_bytes();
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Mkv);
        assert!(det.confidence >= 0.8);
    }

    #[test]
    fn test_detect_webm() {
        let data = webm_bytes();
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Webm);
        assert!(det.confidence >= 0.9);
    }

    #[test]
    fn test_detect_avi() {
        let mut data = b"RIFF".to_vec();
        data.extend_from_slice(&[0x00u8; 4]);
        data.extend_from_slice(b"AVI ");
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Avi);
    }

    #[test]
    fn test_detect_flv() {
        let data = b"FLV\x01\x05\x00\x00\x00\x09";
        let det = FormatDetector::detect(data);
        assert_eq!(det.format, MediaFormat::Flv);
    }

    #[test]
    fn test_detect_mxf() {
        let data = [0x06u8, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Mxf);
    }

    #[test]
    fn test_detect_ogg() {
        let mut data = b"OggS".to_vec();
        data.extend_from_slice(&[0u8; 50]);
        let det = FormatDetector::detect(&data);
        // Generic Ogg (no codec header available in our stub).
        assert!(
            det.format == MediaFormat::Ogg
                || det.format == MediaFormat::Opus
                || det.format == MediaFormat::Vorbis
        );
    }

    // ── Audio ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_detect_flac() {
        let data = b"fLaC\x00\x00\x00\x22";
        let det = FormatDetector::detect(data);
        assert_eq!(det.format, MediaFormat::Flac);
    }

    #[test]
    fn test_detect_wav() {
        let mut data = b"RIFF".to_vec();
        data.extend_from_slice(&[0x24u8, 0x00, 0x00, 0x00]);
        data.extend_from_slice(b"WAVE");
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Wav);
    }

    #[test]
    fn test_detect_mp3_id3() {
        let data = b"ID3\x04\x00\x00";
        let det = FormatDetector::detect(data);
        assert_eq!(det.format, MediaFormat::Mp3);
    }

    #[test]
    fn test_detect_aiff() {
        let mut data = b"FORM".to_vec();
        data.extend_from_slice(&[0x00u8; 4]);
        data.extend_from_slice(b"AIFF");
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Aiff);
    }

    // ── Image ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_detect_jpeg() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Jpeg);
    }

    #[test]
    fn test_detect_png() {
        let data = [0x89u8, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Png);
    }

    #[test]
    fn test_detect_gif() {
        let data = b"GIF89a\x01\x00\x01\x00";
        let det = FormatDetector::detect(data);
        assert_eq!(det.format, MediaFormat::Gif);
    }

    #[test]
    fn test_detect_webp() {
        let mut data = b"RIFF".to_vec();
        data.extend_from_slice(&[0x24u8, 0, 0, 0]);
        data.extend_from_slice(b"WEBP");
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Webp);
    }

    #[test]
    fn test_detect_bmp() {
        let data = b"BM\x36\x00\x00\x00";
        let det = FormatDetector::detect(data);
        assert_eq!(det.format, MediaFormat::Bmp);
    }

    #[test]
    fn test_detect_tiff_little_endian() {
        let data = [0x49u8, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Tiff);
    }

    // ── Archive ───────────────────────────────────────────────────────────────

    #[test]
    fn test_detect_zip() {
        let data = [0x50u8, 0x4B, 0x03, 0x04];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Zip);
    }

    #[test]
    fn test_detect_gz() {
        let data = [0x1Fu8, 0x8B, 0x08, 0x00];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Gz);
    }

    #[test]
    fn test_detect_zstd() {
        let data = [0xFDu8, 0x2F, 0xB5, 0x28, 0x00];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Zstd);
    }

    // ── Extension fallback ────────────────────────────────────────────────────

    #[test]
    fn test_extension_fallback_mp4() {
        assert_eq!(
            FormatDetector::detect_from_extension("mp4"),
            MediaFormat::Mp4
        );
    }

    #[test]
    fn test_extension_fallback_unknown() {
        assert_eq!(
            FormatDetector::detect_from_extension("xyz"),
            MediaFormat::Unknown
        );
    }

    // ── Category predicates ───────────────────────────────────────────────────

    #[test]
    fn test_is_video() {
        assert!(MediaFormat::Mp4.is_video());
        assert!(!MediaFormat::Mp4.is_audio());
        assert!(!MediaFormat::Mp4.is_image());
    }

    #[test]
    fn test_is_audio() {
        assert!(MediaFormat::Flac.is_audio());
        assert!(!MediaFormat::Flac.is_video());
        assert!(!MediaFormat::Flac.is_image());
    }

    #[test]
    fn test_is_image() {
        assert!(MediaFormat::Jpeg.is_image());
        assert!(!MediaFormat::Jpeg.is_video());
        assert!(!MediaFormat::Jpeg.is_audio());
    }

    // ── Unknown ───────────────────────────────────────────────────────────────

    #[test]
    fn test_detect_unknown() {
        let data = [0x00u8, 0x01, 0x02, 0x03, 0x04];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Unknown);
    }

    #[test]
    fn test_detect_empty() {
        let det = FormatDetector::detect(&[]);
        assert_eq!(det.format, MediaFormat::Unknown);
        assert_eq!(det.confidence, 0.0);
    }

    // ── MIME type helper ──────────────────────────────────────────────────────

    #[test]
    fn test_mime_type_jpeg() {
        let det = FormatDetector::detect(&[0xFF, 0xD8, 0xFF, 0xE0]);
        assert_eq!(det.mime_type, "image/jpeg");
    }

    #[test]
    fn test_mime_type_mp4() {
        let data = mp4_ftyp_bytes();
        let det = FormatDetector::detect(&data);
        assert_eq!(det.mime_type, "video/mp4");
    }

    // ── Opus inside Ogg ───────────────────────────────────────────────────────

    #[test]
    fn test_detect_opus_in_ogg() {
        let mut data = b"OggS".to_vec();
        // Pad bytes 4..28 (24 bytes of Ogg page header fields)
        data.extend_from_slice(&[0u8; 24]);
        // Codec identification packet at byte 28
        data.extend_from_slice(b"OpusHead");
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Opus);
    }

    // ── New format detection tests ──────────────────────────────────────────

    #[test]
    fn test_detect_dpx_big_endian() {
        let mut data = b"SDPX".to_vec();
        data.extend_from_slice(&[0u8; 20]);
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Dpx);
        assert_eq!(det.confidence, 1.0);
    }

    #[test]
    fn test_detect_dpx_little_endian() {
        let mut data = b"XPDS".to_vec();
        data.extend_from_slice(&[0u8; 20]);
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Dpx);
    }

    #[test]
    fn test_detect_exr() {
        let data = [0x76u8, 0x2F, 0x31, 0x01, 0x02, 0x00, 0x00, 0x00];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Exr);
        assert_eq!(det.confidence, 1.0);
    }

    #[test]
    fn test_detect_jxl_codestream() {
        let data = [0xFFu8, 0x0A, 0x00, 0x00];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Jxl);
    }

    #[test]
    fn test_detect_jxl_container() {
        let mut data = vec![0x00u8, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20];
        data.extend_from_slice(&[0x0D, 0x0A, 0x87, 0x0A]);
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Jxl);
    }

    #[test]
    fn test_detect_y4m() {
        let data = b"YUV4MPEG2 W320 H240 F30:1 Ip A0:0 C420jpeg\n";
        let det = FormatDetector::detect(data);
        assert_eq!(det.format, MediaFormat::Y4m);
        assert_eq!(det.confidence, 1.0);
    }

    #[test]
    fn test_detect_caf() {
        let mut data = b"caff".to_vec();
        data.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]);
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Caf);
    }

    #[test]
    fn test_detect_mpeg_ps() {
        let data = [0x00u8, 0x00, 0x01, 0xBA, 0x44, 0x00, 0x04, 0x00];
        let det = FormatDetector::detect(&data);
        assert_eq!(det.format, MediaFormat::Ps);
    }

    #[test]
    fn test_detect_ass_subtitle() {
        let data = b"[Script Info]\n; Script generated by Aegisub\nTitle: Test\n";
        let det = FormatDetector::detect(data);
        assert_eq!(det.format, MediaFormat::Ass);
    }

    #[test]
    fn test_dpx_is_image() {
        assert!(MediaFormat::Dpx.is_image());
        assert!(!MediaFormat::Dpx.is_video());
        assert!(!MediaFormat::Dpx.is_audio());
    }

    #[test]
    fn test_exr_metadata() {
        assert_eq!(MediaFormat::Exr.extension(), "exr");
        assert_eq!(MediaFormat::Exr.mime_type(), "image/x-exr");
        assert!(MediaFormat::Exr.description().contains("OpenEXR"));
    }

    #[test]
    fn test_extension_fallback_dpx() {
        assert_eq!(
            FormatDetector::detect_from_extension("dpx"),
            MediaFormat::Dpx
        );
    }

    #[test]
    fn test_extension_fallback_exr() {
        assert_eq!(
            FormatDetector::detect_from_extension("exr"),
            MediaFormat::Exr
        );
    }

    #[test]
    fn test_extension_fallback_jxl() {
        assert_eq!(
            FormatDetector::detect_from_extension("jxl"),
            MediaFormat::Jxl
        );
    }

    #[test]
    fn test_extension_fallback_y4m() {
        assert_eq!(
            FormatDetector::detect_from_extension("y4m"),
            MediaFormat::Y4m
        );
    }

    #[test]
    fn test_extension_fallback_caf() {
        assert_eq!(
            FormatDetector::detect_from_extension("caf"),
            MediaFormat::Caf
        );
    }

    #[test]
    fn test_extension_fallback_mpg() {
        assert_eq!(
            FormatDetector::detect_from_extension("mpg"),
            MediaFormat::Ps
        );
        assert_eq!(
            FormatDetector::detect_from_extension("mpeg"),
            MediaFormat::Ps
        );
    }

    // ── Truncated / edge-case tests ─────────────────────────────────────────

    #[test]
    fn test_detect_single_byte() {
        let det = FormatDetector::detect(&[0x00]);
        // Should not panic, may or may not detect
        assert!(det.confidence <= 1.0);
    }

    #[test]
    fn test_detect_two_bytes_only() {
        let det = FormatDetector::detect(&[0xFF, 0xD8]); // JPEG start
        assert_eq!(det.format, MediaFormat::Jpeg);
    }

    #[test]
    fn test_detect_truncated_png() {
        // PNG needs 8 bytes; provide only 4
        let data = [0x89u8, b'P', b'N', b'G'];
        let det = FormatDetector::detect(&data);
        // Should NOT detect as PNG (insufficient bytes)
        assert_ne!(det.format, MediaFormat::Png);
    }

    #[test]
    fn test_detect_truncated_riff() {
        // RIFF header needs 12 bytes; provide only 6
        let data = b"RIFF\x00\x00";
        let det = FormatDetector::detect(data);
        // Should not detect specific RIFF format
        assert_ne!(det.format, MediaFormat::Wav);
        assert_ne!(det.format, MediaFormat::Avi);
    }

    #[test]
    fn test_new_formats_all_have_extensions() {
        let formats = [
            MediaFormat::Dpx,
            MediaFormat::Exr,
            MediaFormat::Dng,
            MediaFormat::Jxl,
            MediaFormat::Y4m,
            MediaFormat::Ffv1,
            MediaFormat::Caf,
            MediaFormat::Ps,
        ];
        for fmt in formats {
            assert!(!fmt.extension().is_empty(), "{:?} has empty extension", fmt);
            assert!(!fmt.mime_type().is_empty(), "{:?} has empty mime_type", fmt);
            assert!(
                !fmt.description().is_empty(),
                "{:?} has empty description",
                fmt
            );
        }
    }
}
