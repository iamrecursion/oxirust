// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Thumbnail generation from video files.
//!
//! `ThumbnailGenerator` generates image thumbnails at fractional or absolute
//! positions within a video. Full integration requires the codec decode and
//! image resize paths; until those are wired through the `generate_*` methods
//! return [`ConversionError::UnsupportedFormat`]. The builder API, position
//! math, and output-path naming logic are fully implemented.

use crate::{ConversionError, Result};
use std::path::{Path, PathBuf};

/// Generator for video thumbnails.
#[derive(Debug, Clone)]
pub struct ThumbnailGenerator {
    width: u32,
    height: u32,
    time_position: ThumbnailPosition,
    format: super::super::frame::extract::ImageFormat,
    /// Output file pattern; `%02d` is replaced by the thumbnail index.
    output_pattern: String,
}

impl ThumbnailGenerator {
    /// Create a new thumbnail generator (default: 320×180, middle of video, JPEG).
    #[must_use]
    pub fn new() -> Self {
        Self {
            width: 320,
            height: 180,
            time_position: ThumbnailPosition::Middle,
            format: super::super::frame::extract::ImageFormat::Jpeg,
            output_pattern: "thumb_%02d".to_string(),
        }
    }

    /// Set the thumbnail width.
    #[must_use]
    pub fn with_width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set the thumbnail height.
    #[must_use]
    pub fn with_height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Set the thumbnail dimensions.
    #[must_use]
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the time position for extraction.
    #[must_use]
    pub fn with_position(mut self, position: ThumbnailPosition) -> Self {
        self.time_position = position;
        self
    }

    /// Set the output format.
    #[must_use]
    pub fn with_format(mut self, format: super::super::frame::extract::ImageFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the output filename pattern (`%02d` → zero-padded index).
    #[must_use]
    pub fn with_output_pattern<S: Into<String>>(mut self, pattern: S) -> Self {
        self.output_pattern = pattern.into();
        self
    }

    // ── Duration validation helpers ─────────────────────────────────────────

    fn validate_size(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(ConversionError::InvalidInput(
                "Thumbnail width and height must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    // ── Output path generation ──────────────────────────────────────────────

    /// Generate the output path for thumbnail index `idx` in `output_dir`.
    #[must_use]
    pub fn output_path_for(&self, output_dir: &Path, idx: usize) -> PathBuf {
        let name = self.output_pattern.replace("%02d", &format!("{idx:02}"));
        output_dir.join(format!("{name}.{}", self.format.extension()))
    }

    /// Generate a thumbnail from a video at the configured position.
    ///
    /// Returns [`ConversionError::UnsupportedFormat`] until the codec decode
    /// path is integrated.
    pub async fn generate<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        self.validate_size()?;

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        let _ = output;

        Err(ConversionError::UnsupportedFormat(
            "Thumbnail generation requires the codec decode pipeline (demux → decode → resize → \
             image encode), which is not yet integrated."
                .to_string(),
        ))
    }

    /// Generate multiple thumbnails at evenly-spaced positions.
    pub async fn generate_multiple<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        count: usize,
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        self.validate_size()?;

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        let _ = output_dir;

        Err(ConversionError::UnsupportedFormat(
            "Thumbnail generation requires the codec decode pipeline, which is not yet integrated."
                .to_string(),
        ))
    }

    /// Generate a thumbnail at a specific time (in seconds from start).
    pub async fn generate_at<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        time_seconds: f64,
    ) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        self.validate_size()?;

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
            "Thumbnail generation requires the codec decode pipeline, which is not yet integrated."
                .to_string(),
        ))
    }

    /// Generate thumbnails at fractional or absolute positions (0.0–1.0 or
    /// seconds).
    ///
    /// Each element of `positions` is interpreted as:
    /// - A value in 0.0–1.0: fractional position in the video.
    /// - A value > 1.0: absolute position in seconds.
    ///
    /// `total_duration_secs` is used only for fractional positions.
    pub async fn generate_at_positions<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        positions: &[f64],
        _total_duration_secs: f64,
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        self.validate_size()?;

        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }

        if positions.is_empty() {
            return Ok(Vec::new());
        }

        for &p in positions {
            if p < 0.0 {
                return Err(ConversionError::InvalidTimestamp);
            }
        }

        let _ = output_dir;

        Err(ConversionError::UnsupportedFormat(
            "Thumbnail generation requires the codec decode pipeline, which is not yet integrated."
                .to_string(),
        ))
    }

    // ── Preset constructors ─────────────────────────────────────────────────

    /// Create a 16:9 thumbnail (640×360).
    #[must_use]
    pub fn widescreen() -> Self {
        Self::new().with_size(640, 360)
    }

    /// Create a 4:3 thumbnail (640×480).
    #[must_use]
    pub fn standard() -> Self {
        Self::new().with_size(640, 480)
    }

    /// Create a small thumbnail (160×90).
    #[must_use]
    pub fn small() -> Self {
        Self::new().with_size(160, 90)
    }

    /// Create a large thumbnail (1280×720).
    #[must_use]
    pub fn large() -> Self {
        Self::new().with_size(1280, 720)
    }
}

impl Default for ThumbnailGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Position in the video for thumbnail extraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThumbnailPosition {
    /// Start of the video (t = 0).
    Start,
    /// Middle of the video.
    Middle,
    /// Near the end of the video (last second).
    End,
    /// Specific absolute time in seconds.
    Time(f64),
    /// Fractional position through the video (0.0 to 1.0).
    Percent(f64),
}

impl ThumbnailPosition {
    /// Calculate the actual time in seconds based on total video duration.
    #[must_use]
    pub fn calculate_time(&self, duration_seconds: f64) -> f64 {
        match self {
            Self::Start => 0.0,
            Self::Middle => duration_seconds / 2.0,
            Self::End => duration_seconds.max(1.0) - 1.0,
            Self::Time(t) => *t,
            Self::Percent(p) => duration_seconds * p.clamp(0.0, 1.0),
        }
    }

    /// Return `true` if this position requires knowing the total duration.
    #[must_use]
    pub fn requires_duration(&self) -> bool {
        matches!(self, Self::Middle | Self::End | Self::Percent(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::extract::ImageFormat;

    #[test]
    fn test_generator_creation() {
        let gen = ThumbnailGenerator::new();
        assert_eq!(gen.width, 320);
        assert_eq!(gen.height, 180);
    }

    #[test]
    fn test_generator_size() {
        let gen = ThumbnailGenerator::new().with_size(640, 480);
        assert_eq!(gen.width, 640);
        assert_eq!(gen.height, 480);
    }

    #[test]
    fn test_presets() {
        let gen = ThumbnailGenerator::widescreen();
        assert_eq!(gen.width, 640);
        assert_eq!(gen.height, 360);

        let gen = ThumbnailGenerator::small();
        assert_eq!(gen.width, 160);
        assert_eq!(gen.height, 90);
    }

    #[test]
    fn test_position_calculation() {
        let duration = 100.0;

        assert_eq!(ThumbnailPosition::Start.calculate_time(duration), 0.0);
        assert_eq!(ThumbnailPosition::Middle.calculate_time(duration), 50.0);
        assert_eq!(ThumbnailPosition::End.calculate_time(duration), 99.0);
        assert_eq!(ThumbnailPosition::Time(25.0).calculate_time(duration), 25.0);
        assert_eq!(
            ThumbnailPosition::Percent(0.75).calculate_time(duration),
            75.0
        );
    }

    #[test]
    fn test_position_percent_clamp() {
        // Clamps at 1.0
        assert_eq!(ThumbnailPosition::Percent(1.5).calculate_time(100.0), 100.0);
        // Clamps at 0.0
        assert_eq!(ThumbnailPosition::Percent(-0.5).calculate_time(100.0), 0.0);
    }

    #[test]
    fn test_position_requires_duration() {
        assert!(ThumbnailPosition::Middle.requires_duration());
        assert!(ThumbnailPosition::End.requires_duration());
        assert!(ThumbnailPosition::Percent(0.5).requires_duration());
        assert!(!ThumbnailPosition::Start.requires_duration());
        assert!(!ThumbnailPosition::Time(5.0).requires_duration());
    }

    #[test]
    fn test_output_path_for() {
        let gen = ThumbnailGenerator::new().with_output_pattern("frame_%02d");
        let dir = std::env::temp_dir();
        assert_eq!(gen.output_path_for(&dir, 0), dir.join("frame_00.jpg"));
        assert_eq!(gen.output_path_for(&dir, 5), dir.join("frame_05.jpg"));
    }

    #[test]
    fn test_output_path_png() {
        let gen = ThumbnailGenerator::new()
            .with_format(ImageFormat::Png)
            .with_output_pattern("thumb_%02d");
        let dir = PathBuf::from("/out");
        assert_eq!(
            gen.output_path_for(&dir, 1),
            PathBuf::from("/out/thumb_01.png")
        );
    }

    #[test]
    fn test_validate_size_zero_width_errors() {
        let gen = ThumbnailGenerator::new().with_size(0, 180);
        assert!(
            matches!(gen.validate_size(), Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for zero width"
        );
    }

    #[test]
    fn test_validate_size_zero_height_errors() {
        let gen = ThumbnailGenerator::new().with_size(320, 0);
        assert!(
            matches!(gen.validate_size(), Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for zero height"
        );
    }

    #[tokio::test]
    async fn test_generate_missing_file_errors() {
        let gen = ThumbnailGenerator::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_thumb__.mkv");
        let output = std::env::temp_dir().join("__oximedia_nonexistent_thumb_out__.jpg");
        let result = gen.generate(&input, &output).await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_generate_at_negative_time_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_thumb_neg_time.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let gen = ThumbnailGenerator::new();
        let result = gen
            .generate_at(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_thumb_neg_time_out.jpg"),
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
    async fn test_generate_multiple_zero_count_returns_empty() {
        let tmp = std::env::temp_dir().join("oximedia_convert_thumb_zero_count.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let gen = ThumbnailGenerator::new();
        let result = gen.generate_multiple(&tmp, std::env::temp_dir(), 0).await;
        assert!(
            matches!(result, Ok(ref v) if v.is_empty()),
            "expected empty vec, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_generate_at_positions_empty_returns_empty() {
        let tmp = std::env::temp_dir().join("oximedia_convert_thumb_empty_pos.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let gen = ThumbnailGenerator::new();
        let result = gen
            .generate_at_positions(&tmp, std::env::temp_dir(), &[], 60.0)
            .await;
        assert!(
            matches!(result, Ok(ref v) if v.is_empty()),
            "expected empty vec"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_generate_at_positions_negative_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_thumb_neg_pos.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let gen = ThumbnailGenerator::new();
        let result = gen
            .generate_at_positions(&tmp, std::env::temp_dir(), &[-1.0, 0.5], 60.0)
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidTimestamp)),
            "expected InvalidTimestamp, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn test_generate_zero_size_errors() {
        let tmp = std::env::temp_dir().join("oximedia_convert_thumb_zero_size.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");
        let gen = ThumbnailGenerator::new().with_size(0, 0);
        let result = gen
            .generate(&tmp, std::env::temp_dir().join("out.jpg"))
            .await;
        assert!(
            matches!(result, Err(ConversionError::InvalidInput(_))),
            "expected InvalidInput for zero size, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
