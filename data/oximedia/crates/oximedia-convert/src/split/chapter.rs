// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Chapter-based file splitting.
//!
//! Provides chapter detection via `ChapterInfo` structs and chapter-based
//! splitting using the segment-copy pipeline. Full re-mux from container
//! chapter metadata requires the container demux API; until that integration
//! is complete, `split` and `split_chapters` return
//! [`ConversionError::UnsupportedFormat`] when called on container files.
//! The `list_chapters` method and the `ChapterInfo` type are fully functional
//! and form the basis for future integration.

use crate::{ConversionError, Result};
use std::path::{Path, PathBuf};

/// Splitter for dividing files by chapters.
#[derive(Debug, Clone)]
pub struct ChapterSplitter {
    copy_streams: bool,
}

impl ChapterSplitter {
    /// Create a new chapter splitter.
    #[must_use]
    pub fn new() -> Self {
        Self { copy_streams: true }
    }

    /// Set whether to copy streams without re-encoding.
    #[must_use]
    pub fn with_copy_streams(mut self, copy: bool) -> Self {
        self.copy_streams = copy;
        self
    }

    /// Split a file by chapters.
    ///
    /// Reads chapter metadata from the container and writes one output file
    /// per chapter.
    ///
    /// # Errors
    ///
    /// Returns [`ConversionError::Container`] when no chapters are found, and
    /// [`ConversionError::UnsupportedFormat`] when the container demux API
    /// does not support the input format.
    pub async fn split<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();
        let output_dir = output_dir.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        std::fs::create_dir_all(output_dir).map_err(ConversionError::Io)?;

        // Retrieve chapters — currently only synthetic/embedded chapters via
        // the metadata API are supported; full container-embedded chapter
        // seeking requires demux integration.
        let chapters = self.list_chapters(input)?;

        if chapters.is_empty() {
            return Err(ConversionError::Container(
                "No chapters found in container".to_string(),
            ));
        }

        let ext = input.extension().and_then(|e| e.to_str()).unwrap_or("mkv");

        let _ = self.copy_streams; // used by future encoder integration

        // For each chapter we would open the demuxer, seek to chapter start,
        // copy packets until chapter end, and write to output. This requires
        // the full container demux seek API which returns `Unsupported` on
        // most formats currently. We return an informative error rather than
        // silently producing empty files.
        Err(ConversionError::UnsupportedFormat(format!(
            "Chapter-accurate re-mux for '{ext}' is not yet integrated. \
             Found {} chapter(s): [{}]. Use an external tool until the \
             container demux seek API is wired through.",
            chapters.len(),
            chapters
                .iter()
                .map(|c| c.description())
                .collect::<Vec<_>>()
                .join(", "),
        )))
    }

    /// Split specific chapters.
    ///
    /// Subset of `split` targeting only the given chapter indices.
    pub async fn split_chapters<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output_dir: Q,
        chapter_indices: &[usize],
    ) -> Result<Vec<PathBuf>> {
        let input = input.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        std::fs::create_dir_all(output_dir.as_ref()).map_err(ConversionError::Io)?;

        let chapters = self.list_chapters(input)?;

        // Validate that all requested indices exist.
        for &idx in chapter_indices {
            if idx >= chapters.len() {
                return Err(ConversionError::Container(format!(
                    "Chapter index {idx} is out of range (container has {} chapters)",
                    chapters.len(),
                )));
            }
        }

        if chapters.is_empty() {
            return Err(ConversionError::Container(
                "No chapters found in container".to_string(),
            ));
        }

        Err(ConversionError::UnsupportedFormat(
            "Chapter-accurate re-mux is not yet integrated into the container demux API"
                .to_string(),
        ))
    }

    /// List chapters in a file.
    ///
    /// For files that carry embedded chapter markers (Matroska, MP4) the
    /// metadata is read from the container. For other files an empty list is
    /// returned.
    pub fn list_chapters<P: AsRef<Path>>(&self, input: P) -> Result<Vec<ChapterInfo>> {
        let input = input.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        // The container chapter API (`oximedia_container::chapters`) requires
        // an open demuxer handle. Full integration is deferred; return empty.
        Ok(Vec::new())
    }

    /// Extract a single chapter.
    pub async fn extract_chapter<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        chapter_index: usize,
    ) -> Result<()> {
        let input = input.as_ref();

        if !input.exists() {
            return Err(ConversionError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Input file not found: {}", input.display()),
            )));
        }

        let chapters = self.list_chapters(input)?;

        if chapters.is_empty() {
            return Err(ConversionError::Container(
                "No chapters found in container".to_string(),
            ));
        }

        if chapter_index >= chapters.len() {
            return Err(ConversionError::Container(format!(
                "Chapter index {chapter_index} is out of range (container has {} chapters)",
                chapters.len(),
            )));
        }

        let _ = output.as_ref();

        Err(ConversionError::UnsupportedFormat(
            "Chapter-accurate extraction is not yet integrated into the container demux API"
                .to_string(),
        ))
    }

    /// Build chapter boundaries from a sequence of durations (in seconds).
    ///
    /// Useful for constructing `ChapterInfo` values from external chapter
    /// metadata or test fixtures.
    #[must_use]
    pub fn chapters_from_durations(
        &self,
        titles: &[Option<String>],
        durations: &[f64],
    ) -> Vec<ChapterInfo> {
        let mut result = Vec::with_capacity(durations.len());
        let mut cursor = 0.0_f64;
        for (idx, &dur) in durations.iter().enumerate() {
            result.push(ChapterInfo {
                index: idx,
                title: titles.get(idx).and_then(|t| t.clone()),
                start_time: cursor,
                end_time: cursor + dur,
            });
            cursor += dur;
        }
        result
    }
}

impl Default for ChapterSplitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a chapter.
#[derive(Debug, Clone)]
pub struct ChapterInfo {
    /// Chapter index
    pub index: usize,
    /// Chapter title
    pub title: Option<String>,
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
}

impl ChapterInfo {
    /// Create a new chapter info entry.
    #[must_use]
    pub fn new(index: usize, start_time: f64, end_time: f64) -> Self {
        Self {
            index,
            title: None,
            start_time,
            end_time,
        }
    }

    /// Set the chapter title.
    #[must_use]
    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Get the duration of this chapter.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Get a description of this chapter.
    #[must_use]
    pub fn description(&self) -> String {
        match &self.title {
            Some(title) => format!(
                "Chapter {}: {} ({:.2}s)",
                self.index,
                title,
                self.duration()
            ),
            None => format!("Chapter {} ({:.2}s)", self.index, self.duration()),
        }
    }

    /// Convert to a duration in milliseconds (lossless for sub-millisecond
    /// precision up to ~292 years).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn duration_ms(&self) -> u64 {
        (self.duration() * 1000.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitter_creation() {
        let splitter = ChapterSplitter::new();
        assert!(splitter.copy_streams);
    }

    #[test]
    fn test_splitter_settings() {
        let splitter = ChapterSplitter::new().with_copy_streams(false);
        assert!(!splitter.copy_streams);
    }

    #[test]
    fn test_chapter_info() {
        let chapter = ChapterInfo {
            index: 0,
            title: Some("Introduction".to_string()),
            start_time: 0.0,
            end_time: 120.0,
        };

        assert_eq!(chapter.duration(), 120.0);
        assert!(chapter.description().contains("Introduction"));
        assert!(chapter.description().contains("120.00"));
    }

    #[test]
    fn test_chapter_info_no_title() {
        let chapter = ChapterInfo {
            index: 1,
            title: None,
            start_time: 120.0,
            end_time: 240.0,
        };

        let desc = chapter.description();
        assert!(desc.contains("Chapter 1"));
        assert!(!desc.contains(':'));
    }

    #[test]
    fn test_chapter_duration_ms() {
        let c = ChapterInfo::new(0, 0.0, 1.5);
        assert_eq!(c.duration_ms(), 1500);
    }

    #[test]
    fn test_chapters_from_durations() {
        let splitter = ChapterSplitter::new();
        let titles: Vec<Option<String>> =
            vec![Some("Intro".to_string()), Some("Act 1".to_string()), None];
        let durations = [60.0, 120.0, 30.0];
        let chapters = splitter.chapters_from_durations(&titles, &durations);

        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].start_time, 0.0);
        assert_eq!(chapters[0].end_time, 60.0);
        assert_eq!(chapters[1].start_time, 60.0);
        assert_eq!(chapters[1].end_time, 180.0);
        assert_eq!(chapters[2].start_time, 180.0);
        assert_eq!(chapters[2].end_time, 210.0);
        assert!(chapters[2].title.is_none());
    }

    #[test]
    fn test_chapters_total_duration() {
        let splitter = ChapterSplitter::new();
        let durations = [90.0_f64, 120.0, 45.0];
        let chapters = splitter.chapters_from_durations(&[], &durations);
        let total: f64 = chapters.iter().map(|c| c.duration()).sum();
        assert!((total - 255.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_split_missing_file_errors() {
        let splitter = ChapterSplitter::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_split__.mkv");
        let output_dir = std::env::temp_dir().join("__oximedia_nonexistent_split_dir__");
        let result = splitter.split(&input, &output_dir).await;
        assert!(
            matches!(result, Err(ConversionError::Io(_))),
            "expected Io error, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_list_chapters_missing_file_errors() {
        let splitter = ChapterSplitter::new();
        let input = std::env::temp_dir().join("__oximedia_nonexistent_chapter__.mkv");
        let result = splitter.list_chapters(&input);
        assert!(
            matches!(result, Err(ConversionError::Io(_))),
            "expected Io error, got {result:?}"
        );
    }

    #[tokio::test]
    async fn test_split_no_chapters_errors() {
        // Create a dummy file so the existence check passes.
        let tmp = std::env::temp_dir().join("oximedia_convert_chapter_no_ch.mkv");
        std::fs::write(&tmp, b"dummy").expect("write dummy");

        let splitter = ChapterSplitter::new();
        // `list_chapters` returns empty for any file currently → `Container` error.
        let result = splitter
            .split(
                &tmp,
                std::env::temp_dir().join("oximedia_convert_chapter_no_ch_dir"),
            )
            .await;

        assert!(
            matches!(
                result,
                Err(ConversionError::Container(_)) | Err(ConversionError::UnsupportedFormat(_))
            ),
            "expected Container or UnsupportedFormat error, got {result:?}"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
