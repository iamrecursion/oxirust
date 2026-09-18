// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Subtitle extraction from standalone subtitle files.
//!
//! This module handles extraction and re-serialisation of subtitle content.
//! Container-embedded subtitle extraction (demux from MKV, MP4, etc.) returns
//! `UnsupportedFormat` until the container demux API exposes a subtitle packet
//! reader.

use crate::{ConversionError, Result};
use std::path::{Path, PathBuf};

/// Extractor for subtitle streams from media or standalone subtitle files.
#[derive(Debug, Clone)]
pub struct SubtitleExtractor {
    convert_format: Option<super::convert::SubtitleFormat>,
}

impl SubtitleExtractor {
    /// Create a new subtitle extractor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            convert_format: None,
        }
    }

    /// Set the output format for extracted subtitles.
    #[must_use]
    pub fn with_format(mut self, format: super::convert::SubtitleFormat) -> Self {
        self.convert_format = Some(format);
        self
    }

    /// Extract and optionally re-format a standalone subtitle file.
    ///
    /// When the input is a standalone subtitle file (.srt, .ass, .vtt, etc.)
    /// and `with_format` was called, the file is parsed and re-serialised.
    /// Otherwise the file is copied as-is.
    ///
    /// For container files (.mkv, .mp4, .ts …) this returns
    /// [`ConversionError::UnsupportedFormat`] until full container demux
    /// support is integrated.
    ///
    /// Returns the number of subtitle events extracted.
    ///
    /// # Errors
    ///
    /// Returns an error if the input file does not exist, cannot be read, or
    /// if the detected format cannot be parsed.
    pub async fn extract<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        stream_index: usize,
    ) -> Result<usize> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Reject container formats — not yet integrated.
        if matches!(
            ext.as_str(),
            "mkv" | "webm" | "mp4" | "m4v" | "ts" | "mts" | "m2ts" | "avi" | "mov"
        ) {
            return Err(ConversionError::UnsupportedFormat(format!(
                "Extracting embedded subtitle tracks from container files ({ext}) is not yet \
                 integrated. Convert the subtitle track externally first.",
            )));
        }

        // For standalone subtitle files stream_index is irrelevant (must be 0).
        if stream_index != 0 {
            return Err(ConversionError::NoTrack(format!(
                "Subtitle file has exactly one stream; requested index {stream_index} is out of range",
            )));
        }

        // Delegate to SubtitleConverter for format conversion, or plain copy.
        let converter = super::convert::SubtitleConverter::new();

        if let Some(target_fmt) = self.convert_format {
            converter.convert(input, output, target_fmt).await
        } else {
            // Detect source format to validate it is a known subtitle type.
            let _fmt = converter.detect_format(input)?;
            std::fs::copy(input, output).map_err(ConversionError::Io)?;
            // Count events for reporting.
            let text = std::fs::read_to_string(input).map_err(ConversionError::Io)?;
            let count = text
                .split("\n\n")
                .map(str::trim)
                .filter(|b| !b.is_empty())
                .count();
            Ok(count)
        }
    }

    /// Extract all subtitle streams from a media file.
    ///
    /// For container files this always returns an empty Vec (not yet
    /// supported). For standalone subtitle files it returns a single-element
    /// Vec describing the file itself.
    pub async fn extract_all<P: AsRef<Path>>(
        &self,
        input: P,
        output_dir: P,
    ) -> Result<Vec<SubtitleStreamInfo>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Reject container files — not yet integrated.
        if matches!(
            ext.as_str(),
            "mkv" | "webm" | "mp4" | "m4v" | "ts" | "mts" | "m2ts" | "avi" | "mov"
        ) {
            return Err(ConversionError::UnsupportedFormat(format!(
                "Extracting embedded subtitle tracks from container files ({ext}) is not yet \
                 integrated.",
            )));
        }

        let converter = super::convert::SubtitleConverter::new();
        let fmt = converter.detect_format(input)?;

        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("subtitle");
        let out_name = format!("{stem}.{}", fmt.extension());
        let out_path = output_dir.join(&out_name);

        std::fs::create_dir_all(output_dir).map_err(ConversionError::Io)?;

        let count = if let Some(target_fmt) = self.convert_format {
            converter.convert(input, &out_path, target_fmt).await?
        } else {
            std::fs::copy(input, &out_path).map_err(ConversionError::Io)?;
            let text = std::fs::read_to_string(input).map_err(ConversionError::Io)?;
            text.split("\n\n")
                .map(str::trim)
                .filter(|b| !b.is_empty())
                .count()
        };

        let info = SubtitleStreamInfo::new(0, fmt.extension().to_string()).with_title(out_name);

        let _ = count; // event count is in SubtitleStreamInfo for future extension

        Ok(vec![info])
    }

    /// List subtitle streams in a media file.
    ///
    /// For standalone subtitle files this returns a single-element Vec. For
    /// container files it returns an empty Vec (not yet supported).
    pub fn list_streams<P: AsRef<Path>>(&self, input: P) -> Result<Vec<SubtitleStreamInfo>> {
        let input = input.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if matches!(
            ext.as_str(),
            "mkv" | "webm" | "mp4" | "m4v" | "ts" | "mts" | "m2ts" | "avi" | "mov"
        ) {
            // Container files — demux not yet integrated.
            return Ok(Vec::new());
        }

        let converter = super::convert::SubtitleConverter::new();
        match converter.detect_format(input) {
            Ok(fmt) => Ok(vec![SubtitleStreamInfo::new(
                0,
                fmt.extension().to_string(),
            )]),
            Err(_) => Ok(Vec::new()),
        }
    }

    /// Extract subtitle by language code.
    ///
    /// For standalone files this always returns an error because language
    /// metadata is not available outside a container.
    pub async fn extract_by_language<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        language: &str,
    ) -> Result<usize> {
        let streams = self.list_streams(&input)?;

        let stream = streams
            .iter()
            .find(|s| s.language.as_deref() == Some(language))
            .ok_or_else(|| {
                ConversionError::NoTrack(format!(
                    "No subtitle stream found for language: {language}"
                ))
            })?;

        self.extract(input, output, stream.index).await
    }

    /// Extract the first subtitle stream (stream index 0).
    pub async fn extract_first<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<usize> {
        self.extract(input, output, 0).await
    }

    /// Return the output path that `extract_all` would produce for the given
    /// input, without actually performing extraction.
    #[must_use]
    pub fn expected_output_path(&self, input: &Path, output_dir: &Path) -> Option<PathBuf> {
        let stem = input.file_stem().and_then(|s| s.to_str())?;
        let converter = super::convert::SubtitleConverter::new();
        let fmt = converter.detect_format(input).ok()?;
        let ext = self
            .convert_format
            .map(|f| f.extension())
            .unwrap_or_else(|| fmt.extension());
        Some(output_dir.join(format!("{stem}.{ext}")))
    }
}

impl Default for SubtitleExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a subtitle stream.
#[derive(Debug, Clone)]
pub struct SubtitleStreamInfo {
    /// Stream index
    pub index: usize,
    /// Codec name
    pub codec: String,
    /// Language code (e.g., "eng", "spa")
    pub language: Option<String>,
    /// Stream title/description
    pub title: Option<String>,
    /// Whether this is the default stream
    pub is_default: bool,
    /// Whether this is a forced subtitle
    pub is_forced: bool,
}

impl SubtitleStreamInfo {
    /// Create a new subtitle stream info.
    #[must_use]
    pub fn new(index: usize, codec: String) -> Self {
        Self {
            index,
            codec,
            language: None,
            title: None,
            is_default: false,
            is_forced: false,
        }
    }

    /// Set the language.
    pub fn with_language<S: Into<String>>(mut self, language: S) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the title.
    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set whether this is the default stream.
    #[must_use]
    pub fn with_default(mut self, is_default: bool) -> Self {
        self.is_default = is_default;
        self
    }

    /// Set whether this is a forced subtitle.
    #[must_use]
    pub fn with_forced(mut self, is_forced: bool) -> Self {
        self.is_forced = is_forced;
        self
    }

    /// Get a description of this stream.
    #[must_use]
    pub fn description(&self) -> String {
        let mut parts = vec![format!("Stream {}", self.index)];

        if let Some(lang) = &self.language {
            parts.push(format!("Language: {lang}"));
        }

        if let Some(title) = &self.title {
            parts.push(format!("\"{title}\""));
        }

        if self.is_default {
            parts.push("(default)".to_string());
        }

        if self.is_forced {
            parts.push("(forced)".to_string());
        }

        parts.join(" - ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_creation() {
        let extractor = SubtitleExtractor::new();
        assert!(extractor.convert_format.is_none());
    }

    #[test]
    fn test_extractor_with_format() {
        let extractor =
            SubtitleExtractor::new().with_format(super::super::convert::SubtitleFormat::Srt);

        assert!(extractor.convert_format.is_some());
    }

    #[test]
    fn test_stream_info_creation() {
        let info = SubtitleStreamInfo::new(0, "srt".to_string());
        assert_eq!(info.index, 0);
        assert_eq!(info.codec, "srt");
    }

    #[test]
    fn test_stream_info_builder() {
        let info = SubtitleStreamInfo::new(0, "srt".to_string())
            .with_language("eng")
            .with_title("English")
            .with_default(true)
            .with_forced(false);

        assert_eq!(info.language, Some("eng".to_string()));
        assert_eq!(info.title, Some("English".to_string()));
        assert!(info.is_default);
        assert!(!info.is_forced);
    }

    #[test]
    fn test_stream_description() {
        let info = SubtitleStreamInfo::new(0, "srt".to_string())
            .with_language("eng")
            .with_title("English")
            .with_default(true);

        let desc = info.description();
        assert!(desc.contains("Stream 0"));
        assert!(desc.contains("eng"));
        assert!(desc.contains("English"));
        assert!(desc.contains("default"));
    }

    #[tokio::test]
    async fn test_extract_missing_file_errors() {
        let extractor = SubtitleExtractor::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_sub__.srt");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_sub_out__.srt");
        let result = extractor.extract(&input, &output, 0).await;
        assert!(
            matches!(result, Err(ConversionError::Io(_))),
            "expected Io error, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_extract_container_unsupported() {
        let extractor = SubtitleExtractor::new();
        // Use a path that ends in .mkv (file need not exist — we detect by extension first).
        // Actually the function checks existence first, so we need a file.
        let tmp = std::env::temp_dir().join("oximedia_convert_test_dummy.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy mkv");

        let result = extractor
            .extract(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_test_dummy_out.srt"),
                0,
            )
            .await;
        assert!(
            matches!(result, Err(ConversionError::UnsupportedFormat(_))),
            "expected UnsupportedFormat for container, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_stream_index_out_of_range() {
        use std::io::Write;
        let tmp = std::env::temp_dir().join("oximedia_convert_test_extract_idx.srt");
        {
            let mut f = std::fs::File::create(&tmp).expect("create tmp srt");
            f.write_all(b"1\n00:00:01,000 --> 00:00:04,000\nHello\n\n")
                .expect("write");
        }
        let extractor = SubtitleExtractor::new();
        let result = extractor
            .extract(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_test_extract_idx_out.srt"),
                1,
            )
            .await;
        assert!(
            matches!(result, Err(ConversionError::NoTrack(_))),
            "expected NoTrack for index 1, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_srt_with_format_conversion() {
        use std::io::Write;
        let tmp_dir = std::env::temp_dir();
        let srt_path = tmp_dir.join("oximedia_convert_test_extract_srt.srt");
        let vtt_path = tmp_dir.join("oximedia_convert_test_extract_srt_out.vtt");

        let content = "1\n00:00:01,000 --> 00:00:04,000\nHello!\n\n";
        {
            let mut f = std::fs::File::create(&srt_path).expect("create tmp srt");
            f.write_all(content.as_bytes()).expect("write");
        }

        let extractor =
            SubtitleExtractor::new().with_format(super::super::convert::SubtitleFormat::WebVtt);
        let count = extractor
            .extract(&srt_path, &vtt_path, 0)
            .await
            .expect("extract should succeed");
        assert_eq!(count, 1);
        let text = std::fs::read_to_string(&vtt_path).expect("read vtt");
        assert!(text.starts_with("WEBVTT"));

        let _ = std::fs::remove_file(&srt_path);
        let _ = std::fs::remove_file(&vtt_path);
    }
}
