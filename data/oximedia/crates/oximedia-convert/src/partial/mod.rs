// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Partial conversion support (time ranges, chapters, stream selection).

use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Time range for partial conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: Duration,
    /// End time (None for end of file)
    pub end: Option<Duration>,
}

impl TimeRange {
    /// Create a new time range.
    #[must_use]
    pub const fn new(start: Duration, end: Option<Duration>) -> Self {
        Self { start, end }
    }

    /// Create from seconds.
    pub fn from_seconds(start: f64, end: Option<f64>) -> Self {
        Self {
            start: Duration::from_secs_f64(start),
            end: end.map(Duration::from_secs_f64),
        }
    }

    /// Get duration of the range.
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.end.map(|end| end.saturating_sub(self.start))
    }

    /// Validate the time range.
    pub fn validate(&self) -> Result<()> {
        if let Some(end) = self.end {
            if end <= self.start {
                return Err(ConversionError::InvalidInput(
                    "End time must be greater than start time".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Chapter selection for conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChapterSelection {
    /// Chapter indices to include
    pub chapters: Vec<usize>,
}

impl ChapterSelection {
    /// Create a new chapter selection.
    #[must_use]
    pub fn new(chapters: Vec<usize>) -> Self {
        Self { chapters }
    }

    /// Select a single chapter.
    #[must_use]
    pub fn single(chapter: usize) -> Self {
        Self {
            chapters: vec![chapter],
        }
    }

    /// Select a range of chapters.
    #[must_use]
    pub fn range(start: usize, end: usize) -> Self {
        Self {
            chapters: (start..=end).collect(),
        }
    }

    /// Select all chapters.
    #[must_use]
    pub fn all() -> Self {
        Self {
            chapters: Vec::new(), // Empty means all
        }
    }

    /// Check if all chapters are selected.
    #[must_use]
    pub fn is_all(&self) -> bool {
        self.chapters.is_empty()
    }
}

/// Stream selection for conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamSelection {
    /// Video stream indices to include
    pub video_streams: Vec<usize>,
    /// Audio stream indices to include
    pub audio_streams: Vec<usize>,
    /// Subtitle stream indices to include
    pub subtitle_streams: Vec<usize>,
}

impl StreamSelection {
    /// Create a new stream selection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            video_streams: Vec::new(),
            audio_streams: Vec::new(),
            subtitle_streams: Vec::new(),
        }
    }

    /// Select all streams.
    #[must_use]
    pub fn all() -> Self {
        Self::new() // Empty vectors mean all
    }

    /// Select first video and audio streams only.
    #[must_use]
    pub fn default_streams() -> Self {
        Self {
            video_streams: vec![0],
            audio_streams: vec![0],
            subtitle_streams: Vec::new(),
        }
    }

    /// Add video stream.
    #[must_use]
    pub fn with_video(mut self, index: usize) -> Self {
        self.video_streams.push(index);
        self
    }

    /// Add audio stream.
    #[must_use]
    pub fn with_audio(mut self, index: usize) -> Self {
        self.audio_streams.push(index);
        self
    }

    /// Add subtitle stream.
    #[must_use]
    pub fn with_subtitle(mut self, index: usize) -> Self {
        self.subtitle_streams.push(index);
        self
    }

    /// Check if all streams are selected.
    #[must_use]
    pub fn is_all(&self) -> bool {
        self.video_streams.is_empty()
            && self.audio_streams.is_empty()
            && self.subtitle_streams.is_empty()
    }
}

impl Default for StreamSelection {
    fn default() -> Self {
        Self::new()
    }
}

/// Partial conversion configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartialConversion {
    /// Time range (if specified)
    pub time_range: Option<TimeRange>,
    /// Chapter selection (if specified)
    pub chapters: Option<ChapterSelection>,
    /// Stream selection
    pub streams: StreamSelection,
}

impl PartialConversion {
    /// Create a new partial conversion config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            time_range: None,
            chapters: None,
            streams: StreamSelection::all(),
        }
    }

    /// Set time range.
    #[must_use]
    pub fn with_time_range(mut self, range: TimeRange) -> Self {
        self.time_range = Some(range);
        self
    }

    /// Set chapter selection.
    #[must_use]
    pub fn with_chapters(mut self, chapters: ChapterSelection) -> Self {
        self.chapters = Some(chapters);
        self
    }

    /// Set stream selection.
    #[must_use]
    pub fn with_streams(mut self, streams: StreamSelection) -> Self {
        self.streams = streams;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if let Some(ref range) = self.time_range {
            range.validate()?;
        }

        if self.time_range.is_some() && self.chapters.is_some() {
            return Err(ConversionError::InvalidInput(
                "Cannot specify both time range and chapter selection".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for PartialConversion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range() {
        let range = TimeRange::from_seconds(10.0, Some(60.0));
        assert_eq!(range.start, Duration::from_secs(10));
        assert_eq!(range.end, Some(Duration::from_secs(60)));
        assert_eq!(range.duration(), Some(Duration::from_secs(50)));
        assert!(range.validate().is_ok());
    }

    #[test]
    fn test_time_range_invalid() {
        let range = TimeRange::from_seconds(60.0, Some(10.0));
        assert!(range.validate().is_err());
    }

    #[test]
    fn test_chapter_selection() {
        let single = ChapterSelection::single(0);
        assert_eq!(single.chapters, vec![0]);

        let range = ChapterSelection::range(0, 5);
        assert_eq!(range.chapters, vec![0, 1, 2, 3, 4, 5]);

        let all = ChapterSelection::all();
        assert!(all.is_all());
    }

    #[test]
    fn test_stream_selection() {
        let streams = StreamSelection::new()
            .with_video(0)
            .with_audio(1)
            .with_subtitle(0);

        assert_eq!(streams.video_streams, vec![0]);
        assert_eq!(streams.audio_streams, vec![1]);
        assert_eq!(streams.subtitle_streams, vec![0]);
    }

    #[test]
    fn test_stream_selection_default() {
        let streams = StreamSelection::default_streams();
        assert_eq!(streams.video_streams, vec![0]);
        assert_eq!(streams.audio_streams, vec![0]);
        assert!(streams.subtitle_streams.is_empty());
    }

    #[test]
    fn test_partial_conversion() {
        let partial = PartialConversion::new()
            .with_time_range(TimeRange::from_seconds(10.0, Some(60.0)))
            .with_streams(StreamSelection::default_streams());

        assert!(partial.time_range.is_some());
        assert!(partial.validate().is_ok());
    }

    #[test]
    fn test_partial_conversion_conflicting() {
        let partial = PartialConversion::new()
            .with_time_range(TimeRange::from_seconds(10.0, Some(60.0)))
            .with_chapters(ChapterSelection::single(0));

        assert!(partial.validate().is_err());
    }
}
