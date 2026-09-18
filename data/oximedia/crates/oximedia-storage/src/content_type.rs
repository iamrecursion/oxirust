#![allow(dead_code)]
//! Content-type detection from file extension.
//!
//! Provides `ContentTypeDetector` which maps file extensions to MIME types,
//! and extends `UploadOptions` with automatic detection when `content_type` is `None`.

/// Content-type detector based on file extension lookup.
///
/// Covers 50+ common media, document, archive, and web extensions.
pub struct ContentTypeDetector;

impl ContentTypeDetector {
    /// Return the MIME content-type string for the given lowercase file extension
    /// (without leading dot), or `None` if the extension is not recognised.
    ///
    /// The extension lookup is case-insensitive internally: callers may pass
    /// `"MP4"` or `"mp4"` and get the same result.
    pub fn from_extension(ext: &str) -> Option<String> {
        let lower = ext.to_lowercase();
        let mime = match lower.as_str() {
            // ── Video ──────────────────────────────────────────────────────────────
            "mp4" | "m4v" => "video/mp4",
            "mkv" => "video/x-matroska",
            "avi" => "video/x-msvideo",
            "mov" | "qt" => "video/quicktime",
            "wmv" => "video/x-ms-wmv",
            "flv" => "video/x-flv",
            "webm" => "video/webm",
            "ogv" => "video/ogg",
            "3gp" => "video/3gpp",
            "3g2" => "video/3gpp2",
            "ts" | "m2ts" | "mts" => "video/mp2t",
            "mpg" | "mpeg" => "video/mpeg",
            "m2v" => "video/mpeg",
            "mxf" => "application/mxf",
            "dv" => "video/x-dv",
            "y4m" => "video/x-yuv4mpeg",
            "ivf" => "video/x-ivf",

            // ── Audio ──────────────────────────────────────────────────────────────
            "mp3" => "audio/mpeg",
            "aac" => "audio/aac",
            "flac" => "audio/flac",
            "wav" | "wave" => "audio/wav",
            "ogg" | "oga" => "audio/ogg",
            "opus" => "audio/opus",
            "m4a" => "audio/mp4",
            "wma" => "audio/x-ms-wma",
            "aiff" | "aif" => "audio/aiff",
            "amr" => "audio/amr",
            "caf" => "audio/x-caf",
            "mid" | "midi" => "audio/midi",
            "ra" | "ram" => "audio/x-realaudio",
            "dts" => "audio/vnd.dts",
            "ac3" => "audio/ac3",
            "eac3" => "audio/eac3",
            "spx" => "audio/ogg",
            "vorbis" => "audio/vorbis",

            // ── Image ──────────────────────────────────────────────────────────────
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            "tiff" | "tif" => "image/tiff",
            "svg" | "svgz" => "image/svg+xml",
            "ico" => "image/x-icon",
            "heic" | "heif" => "image/heif",
            "avif" => "image/avif",
            "jxl" => "image/jxl",
            "raw" | "dng" | "arw" | "cr2" | "cr3" | "nef" | "orf" | "rw2" | "pef" => "image/x-raw",
            "exr" => "image/x-exr",
            "dpx" => "image/x-dpx",
            "psd" => "image/vnd.adobe.photoshop",
            "xcf" => "image/x-xcf",

            // ── Subtitles / Captions ────────────────────────────────────────────
            "srt" => "text/plain",
            "vtt" => "text/vtt",
            "ass" | "ssa" => "text/x-ssa",
            "ttml" => "application/ttml+xml",
            "dfxp" => "application/ttml+xml",
            "scc" => "text/x-scc",
            "sbv" => "text/plain",

            // ── Manifest / Playlist ─────────────────────────────────────────────
            "m3u8" => "application/vnd.apple.mpegurl",
            "m3u" => "audio/x-mpegurl",
            "mpd" => "application/dash+xml",
            "ism" | "ismc" | "ismd" => "application/vnd.ms-sstr+xml",

            // ── Archives ────────────────────────────────────────────────────────
            "zip" => "application/zip",
            "tar" => "application/x-tar",
            "gz" | "tgz" => "application/gzip",
            "bz2" => "application/x-bzip2",
            "xz" => "application/x-xz",
            "zst" => "application/zstd",
            "7z" => "application/x-7z-compressed",
            "rar" => "application/vnd.rar",
            "lz4" => "application/x-lz4",
            "oarc" => "application/x-oarc",

            // ── Documents ───────────────────────────────────────────────────────
            "pdf" => "application/pdf",
            "xml" => "application/xml",
            "json" => "application/json",
            "csv" => "text/csv",
            "txt" => "text/plain",
            "html" | "htm" => "text/html",
            "css" => "text/css",
            "js" | "mjs" => "application/javascript",
            "wasm" => "application/wasm",

            // ── Font ────────────────────────────────────────────────────────────
            "ttf" => "font/ttf",
            "otf" => "font/otf",
            "woff" => "font/woff",
            "woff2" => "font/woff2",

            // ── Binary / misc ────────────────────────────────────────────────────
            "bin" | "dat" => "application/octet-stream",

            _ => return None,
        };
        Some(mime.to_string())
    }

    /// Detect content-type from a file path by inspecting the extension.
    ///
    /// Returns `None` if the path has no extension or the extension is unknown.
    pub fn from_path(path: &std::path::Path) -> Option<String> {
        let ext = path.extension()?.to_str()?;
        Self::from_extension(ext)
    }
}

/// Extension trait that adds `resolve_content_type` to `UploadOptions`.
pub trait UploadOptionsExt {
    /// Fill `content_type` from the file path extension when it is currently `None`.
    ///
    /// Returns `true` if detection succeeded and the field was filled in.
    fn resolve_content_type(&mut self, path: &std::path::Path) -> bool;
}

impl UploadOptionsExt for crate::UploadOptions {
    fn resolve_content_type(&mut self, path: &std::path::Path) -> bool {
        if self.content_type.is_none() {
            if let Some(ct) = ContentTypeDetector::from_path(path) {
                self.content_type = Some(ct);
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UploadOptions;
    use std::path::Path;

    // ── Extension → MIME ───────────────────────────────────────────────────────

    #[test]
    fn test_mp4() {
        assert_eq!(
            ContentTypeDetector::from_extension("mp4"),
            Some("video/mp4".to_string())
        );
    }

    #[test]
    fn test_mkv() {
        assert_eq!(
            ContentTypeDetector::from_extension("mkv"),
            Some("video/x-matroska".to_string())
        );
    }

    #[test]
    fn test_mp3() {
        assert_eq!(
            ContentTypeDetector::from_extension("mp3"),
            Some("audio/mpeg".to_string())
        );
    }

    #[test]
    fn test_flac() {
        assert_eq!(
            ContentTypeDetector::from_extension("flac"),
            Some("audio/flac".to_string())
        );
    }

    #[test]
    fn test_wav() {
        assert_eq!(
            ContentTypeDetector::from_extension("wav"),
            Some("audio/wav".to_string())
        );
    }

    #[test]
    fn test_opus() {
        assert_eq!(
            ContentTypeDetector::from_extension("opus"),
            Some("audio/opus".to_string())
        );
    }

    #[test]
    fn test_png() {
        assert_eq!(
            ContentTypeDetector::from_extension("png"),
            Some("image/png".to_string())
        );
    }

    #[test]
    fn test_jpg_jpeg() {
        assert_eq!(
            ContentTypeDetector::from_extension("jpg"),
            Some("image/jpeg".to_string())
        );
        assert_eq!(
            ContentTypeDetector::from_extension("jpeg"),
            Some("image/jpeg".to_string())
        );
    }

    #[test]
    fn test_gif() {
        assert_eq!(
            ContentTypeDetector::from_extension("gif"),
            Some("image/gif".to_string())
        );
    }

    #[test]
    fn test_webp() {
        assert_eq!(
            ContentTypeDetector::from_extension("webp"),
            Some("image/webp".to_string())
        );
    }

    #[test]
    fn test_webm() {
        assert_eq!(
            ContentTypeDetector::from_extension("webm"),
            Some("video/webm".to_string())
        );
    }

    #[test]
    fn test_m3u8() {
        assert_eq!(
            ContentTypeDetector::from_extension("m3u8"),
            Some("application/vnd.apple.mpegurl".to_string())
        );
    }

    #[test]
    fn test_srt() {
        assert_eq!(
            ContentTypeDetector::from_extension("srt"),
            Some("text/plain".to_string())
        );
    }

    #[test]
    fn test_vtt() {
        assert_eq!(
            ContentTypeDetector::from_extension("vtt"),
            Some("text/vtt".to_string())
        );
    }

    #[test]
    fn test_pdf() {
        assert_eq!(
            ContentTypeDetector::from_extension("pdf"),
            Some("application/pdf".to_string())
        );
    }

    #[test]
    fn test_json() {
        assert_eq!(
            ContentTypeDetector::from_extension("json"),
            Some("application/json".to_string())
        );
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(
            ContentTypeDetector::from_extension("MP4"),
            Some("video/mp4".to_string())
        );
        assert_eq!(
            ContentTypeDetector::from_extension("PNG"),
            Some("image/png".to_string())
        );
    }

    #[test]
    fn test_unknown_extension_returns_none() {
        assert_eq!(ContentTypeDetector::from_extension("xyz"), None);
        assert_eq!(ContentTypeDetector::from_extension(""), None);
    }

    #[test]
    fn test_from_path() {
        let p = Path::new("video/clip.mp4");
        assert_eq!(
            ContentTypeDetector::from_path(p),
            Some("video/mp4".to_string())
        );
    }

    #[test]
    fn test_from_path_no_extension() {
        let p = Path::new("Makefile");
        assert_eq!(ContentTypeDetector::from_path(p), None);
    }

    // ── UploadOptionsExt ───────────────────────────────────────────────────────

    #[test]
    fn test_upload_options_ext_fills_when_none() {
        let mut opts = UploadOptions::default();
        let p = Path::new("clip.mkv");
        let filled = opts.resolve_content_type(p);
        assert!(filled);
        assert_eq!(opts.content_type.as_deref(), Some("video/x-matroska"));
    }

    #[test]
    fn test_upload_options_ext_no_override_when_set() {
        let mut opts = UploadOptions {
            content_type: Some("application/octet-stream".to_string()),
            ..Default::default()
        };
        let p = Path::new("clip.mp4");
        let filled = opts.resolve_content_type(p);
        assert!(!filled);
        assert_eq!(
            opts.content_type.as_deref(),
            Some("application/octet-stream")
        );
    }

    #[test]
    fn test_dng_raw_image() {
        assert_eq!(
            ContentTypeDetector::from_extension("dng"),
            Some("image/x-raw".to_string())
        );
    }

    #[test]
    fn test_jxl() {
        assert_eq!(
            ContentTypeDetector::from_extension("jxl"),
            Some("image/jxl".to_string())
        );
    }

    #[test]
    fn test_ts_video() {
        assert_eq!(
            ContentTypeDetector::from_extension("ts"),
            Some("video/mp2t".to_string())
        );
    }
}
