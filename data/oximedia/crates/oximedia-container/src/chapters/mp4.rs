//! MP4 chapter support.
//!
//! Provides chapter handling for MP4 containers (Nero/QuickTime style).

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};

/// An MP4 chapter (QuickTime/Nero style).
#[derive(Debug, Clone)]
pub struct Mp4Chapter {
    /// Start time in milliseconds.
    pub start_time_ms: u64,
    /// Chapter title.
    pub title: String,
}

impl Mp4Chapter {
    /// Creates a new MP4 chapter.
    #[must_use]
    pub fn new(start_time_ms: u64, title: impl Into<String>) -> Self {
        Self {
            start_time_ms,
            title: title.into(),
        }
    }

    /// Returns the start time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn start_time_secs(&self) -> f64 {
        self.start_time_ms as f64 / 1000.0
    }
}

/// MP4 chapter track (`QuickTime`).
#[derive(Debug, Clone)]
pub struct Mp4ChapterTrack {
    chapters: Vec<Mp4Chapter>,
}

impl Mp4ChapterTrack {
    /// Creates a new MP4 chapter track.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chapters: Vec::new(),
        }
    }

    /// Adds a chapter.
    pub fn add_chapter(&mut self, chapter: Mp4Chapter) {
        self.chapters.push(chapter);
    }

    /// Returns all chapters.
    #[must_use]
    pub fn chapters(&self) -> &[Mp4Chapter] {
        &self.chapters
    }

    /// Sorts chapters by start time.
    pub fn sort(&mut self) {
        self.chapters.sort_by_key(|ch| ch.start_time_ms);
    }

    /// Validates the chapter track.
    ///
    /// # Errors
    ///
    /// Returns `Err` if chapter times are not monotonically increasing.
    pub fn validate(&self) -> OxiResult<()> {
        // Check for monotonically increasing times
        let mut last_time = 0;
        for chapter in &self.chapters {
            if chapter.start_time_ms < last_time {
                return Err(OxiError::InvalidData(
                    "Chapter times are not monotonically increasing".into(),
                ));
            }
            last_time = chapter.start_time_ms;
        }
        Ok(())
    }
}

impl Default for Mp4ChapterTrack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp4_chapter() {
        let chapter = Mp4Chapter::new(5000, "Chapter 1");
        assert_eq!(chapter.start_time_ms, 5000);
        assert_eq!(chapter.title, "Chapter 1");
        assert!((chapter.start_time_secs() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_mp4_chapter_track() {
        let mut track = Mp4ChapterTrack::new();
        track.add_chapter(Mp4Chapter::new(0, "Intro"));
        track.add_chapter(Mp4Chapter::new(5000, "Main"));

        assert_eq!(track.chapters().len(), 2);
        assert!(track.validate().is_ok());
    }
}
