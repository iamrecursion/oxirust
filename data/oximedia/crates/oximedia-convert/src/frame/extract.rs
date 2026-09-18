// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Frame extraction from video files.
//!
//! `FrameExtractor` extracts individual decoded video frames as image files.
//! Full integration (demux → decode → encode-to-PNG/JPEG) requires the codec
//! decode API; until that is wired through, `extract_at` and related methods
//! return [`ConversionError::UnsupportedFormat`]. The builder API, format
//! helpers, and validation logic are fully implemented.

use crate::{ConversionError, Result};
use std::path::{Path, PathBuf};

/// Extractor for video frames as images.
#[derive(Debug, Clone)]
pub struct FrameExtractor {
    format: ImageFormat,
    quality: u32,
}

impl FrameExtractor {
    /// Create a new frame extractor (default: JPEG, quality 90).
    #[must_use]
    pub fn new() -> Self {
        Self {
            format: ImageFormat::Jpeg,
            quality: 90,
        }
    }

    /// Set the output image format.
    #[must_use]
    pub fn with_format(mut self, format: ImageFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the output quality (1–100 for JPEG/WebP; clamped).
    #[must_use]
    pub fn with_quality(mut self, quality: u32) -> Self {
        self.quality = quality.clamp(1, 100);
        self
    }

    /// Extract a single frame at a specific time (in seconds from start).
    ///
    /// Returns [`ConversionError::InvalidTimestamp`] for negative timestamps
    /// and [`ConversionError::UnsupportedFormat`] until the codec decode path
    /// is integrated.
    pub async fn extract_at<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        time_seconds: f64,
    ) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if time_seconds < 0.0 {
            return Err(ConversionError::InvalidTimestamp);
        }

        let _ = output;

        Err(ConversionError::UnsupportedFormat(
            "Frame extraction requires the codec decode pipeline (demux → decode → image encode), \
             which is not yet integrated."
                .to_string(),
        ))
    }

    /// Extract frames at regular intervals.
    ///
    /// Returns an empty list or a structured error depending on whether the
    /// input file is valid.
    pub async fn extract_interval<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        interval_seconds: f64,
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if interval_seconds <= 0.0 {
            return Err(ConversionError::InvalidInput(
                "Interval must be greater than zero".to_string(),
            ));
        }

        let _ = output_dir;

        Err(ConversionError::UnsupportedFormat(
            "Interval frame extraction requires the codec decode pipeline, which is not yet \
             integrated."
                .to_string(),
        ))
    }

    /// Extract frames at specific times (in seconds from start).
    pub async fn extract_multiple<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        times: &[f64],
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if times.is_empty() {
            return Ok(Vec::new());
        }

        for &t in times {
            if t < 0.0 {
                return Err(ConversionError::InvalidTimestamp);
            }
        }

        let _ = output_dir;

        Err(ConversionError::UnsupportedFormat(
            "Multi-frame extraction requires the codec decode pipeline, which is not yet \
             integrated."
                .to_string(),
        ))
    }

    /// Extract all frames in a time range.
    pub async fn extract_range<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        range: &super::FrameRange,
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if range.start < 0.0 {
            return Err(ConversionError::InvalidTimestamp);
        }

        if let Some(end) = range.end {
            if end < range.start {
                return Err(ConversionError::InvalidInput(
                    "Frame range end must not be before start".to_string(),
                ));
            }
        }

        let _ = output_dir;

        Err(ConversionError::UnsupportedFormat(
            "Range-based frame extraction requires the codec decode pipeline, which is not yet \
             integrated."
                .to_string(),
        ))
    }

    /// Generate the expected output filename for a frame extracted at `time_seconds`.
    ///
    /// Used for pre-flight path validation without actually extracting.
    #[must_use]
    pub fn output_filename_for(&self, base_name: &str, time_seconds: f64) -> String {
        format!("{base_name}_t{time_seconds:.3}.{}", self.format.extension())
    }

    /// Extract as JPEG.
    #[must_use]
    pub fn as_jpeg(self) -> Self {
        self.with_format(ImageFormat::Jpeg)
    }

    /// Extract as PNG.
    #[must_use]
    pub fn as_png(self) -> Self {
        self.with_format(ImageFormat::Png)
    }

    /// Extract as WebP.
    #[must_use]
    pub fn as_webp(self) -> Self {
        self.with_format(ImageFormat::WebP)
    }
}

impl Default for FrameExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported image formats for frame extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// JPEG — lossy, small file size.
    Jpeg,
    /// PNG — lossless.
    Png,
    /// WebP — lossy or lossless, very small.
    WebP,
    /// BMP — uncompressed.
    Bmp,
}

impl ImageFormat {
    /// Get the file extension (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::WebP => "webp",
            Self::Bmp => "bmp",
        }
    }

    /// Get the MIME type.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::WebP => "image/webp",
            Self::Bmp => "image/bmp",
        }
    }

    /// Check if this format supports quality settings.
    #[must_use]
    pub fn supports_quality(&self) -> bool {
        matches!(self, Self::Jpeg | Self::WebP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_creation() {
        let extractor = FrameExtractor::new();
        assert_eq!(extractor.format, ImageFormat::Jpeg);
        assert_eq!(extractor.quality, 90);
    }

    #[test]
    fn test_extractor_settings() {
        let extractor = FrameExtractor::new()
            .with_format(ImageFormat::Png)
            .with_quality(100);

        assert_eq!(extractor.format, ImageFormat::Png);
        assert_eq!(extractor.quality, 100);
    }

    #[test]
    fn test_quality_clamping() {
        let extractor = FrameExtractor::new().with_quality(150);
        assert_eq!(extractor.quality, 100);

        let extractor = FrameExtractor::new().with_quality(0);
        assert_eq!(extractor.quality, 1);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::WebP.extension(), "webp");
        assert_eq!(ImageFormat::Bmp.extension(), "bmp");
    }

    #[test]
    fn test_format_mime_type() {
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::WebP.mime_type(), "image/webp");
    }

    #[test]
    fn test_format_quality_support() {
        assert!(ImageFormat::Jpeg.supports_quality());
        assert!(ImageFormat::WebP.supports_quality());
        assert!(!ImageFormat::Png.supports_quality());
        assert!(!ImageFormat::Bmp.supports_quality());
    }

    #[test]
    fn test_convenience_methods() {
        let extractor = FrameExtractor::new().as_png();
        assert_eq!(extractor.format, ImageFormat::Png);

        let extractor = FrameExtractor::new().as_webp();
        assert_eq!(extractor.format, ImageFormat::WebP);
    }

    #[test]
    fn test_output_filename_for() {
        let extractor = FrameExtractor::new().as_png();
        let name = extractor.output_filename_for("frame", 12.5);
        assert_eq!(name, "frame_t12.500.png");
    }

    #[test]
    fn test_output_filename_jpeg() {
        let extractor = FrameExtractor::new().as_jpeg();
        let name = extractor.output_filename_for("thumb", 0.0);
        assert_eq!(name, "thumb_t0.000.jpg");
    }

    #[tokio::test]
    async fn test_extract_at_missing_file_errors() {
        let extractor = FrameExtractor::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_frame__.mkv");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_frame_out__.png");
        let result = extractor.extract_at(&input, &output, 1.0).await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_extract_at_negative_timestamp_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_frame_neg_ts.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = FrameExtractor::new();
        let result = extractor
            .extract_at(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_frame_neg_ts_out.png"),
                -1.0,
            )
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidTimestamp)),
            "expected InvalidTimestamp, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_interval_zero_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_frame_zero_interval.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = FrameExtractor::new();
        let result = extractor
            .extract_interval(&tmp, std::env::temp_dir(), 0.0)
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for zero interval, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_multiple_empty_returns_empty() {
        let tmp = std::env::temp_dir().join("oximedia_convert_frame_empty_times.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = FrameExtractor::new();
        let result = extractor
            .extract_multiple(&tmp, std::env::temp_dir(), &[])
            .await;
        assert!(matches!(result, Ok(ref v) if v.is_empty()));
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_range_negative_start_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_frame_neg_range.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = FrameExtractor::new();
        let range = super::super::FrameRange::new(-1.0, Some(5.0));
        let result = extractor
            .extract_range(&tmp, std::env::temp_dir(), &range)
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidTimestamp)),
            "expected InvalidTimestamp, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_extract_range_end_before_start_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_frame_bad_range.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let extractor = FrameExtractor::new();
        let range = super::super::FrameRange::new(10.0, Some(5.0));
        let result = extractor
            .extract_range(&tmp, std::env::temp_dir(), &range)
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for end < start, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
