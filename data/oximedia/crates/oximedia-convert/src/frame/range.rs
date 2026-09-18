// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Frame range selection for extraction.

/// Range specification for frame extraction.
#[derive(Debug, Clone)]
pub struct FrameRange {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds (None = until end)
    pub end: Option<f64>,
    /// Frame rate (frames per second)
    pub fps: Option<f64>,
    /// Maximum number of frames
    pub max_frames: Option<usize>,
}

impl FrameRange {
    /// Create a new frame range from start to end.
    #[must_use]
    pub fn new(start: f64, end: Option<f64>) -> Self {
        Self {
            start,
            end,
            fps: None,
            max_frames: None,
        }
    }

    /// Create a range for the entire video.
    #[must_use]
    pub fn all() -> Self {
        Self {
            start: 0.0,
            end: None,
            fps: None,
            max_frames: None,
        }
    }

    /// Create a range for a specific duration.
    #[must_use]
    pub fn from_duration(start: f64, duration: f64) -> Self {
        Self {
            start,
            end: Some(start + duration),
            fps: None,
            max_frames: None,
        }
    }

    /// Set the target frame rate.
    #[must_use]
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Set the maximum number of frames.
    #[must_use]
    pub fn with_max_frames(mut self, max: usize) -> Self {
        self.max_frames = Some(max);
        self
    }

    /// Get the duration of this range.
    #[must_use]
    pub fn duration(&self) -> Option<f64> {
        self.end.map(|end| end - self.start)
    }

    /// Check if a time is within this range.
    #[must_use]
    pub fn contains(&self, time: f64) -> bool {
        if time < self.start {
            return false;
        }

        match self.end {
            Some(end) => time <= end,
            None => true,
        }
    }

    /// Calculate the number of frames in this range.
    #[must_use]
    pub fn frame_count(&self, video_duration: f64, video_fps: f64) -> usize {
        let end = self.end.unwrap_or(video_duration);
        let duration = end - self.start;
        let fps = self.fps.unwrap_or(video_fps);

        let calculated = (duration * fps) as usize;

        match self.max_frames {
            Some(max) => calculated.min(max),
            None => calculated,
        }
    }

    /// Extract one frame per second.
    #[must_use]
    pub fn one_per_second() -> Self {
        Self::all().with_fps(1.0)
    }

    /// Extract one frame per minute.
    #[must_use]
    pub fn one_per_minute() -> Self {
        Self::all().with_fps(1.0 / 60.0)
    }

    /// Extract a specific number of frames evenly distributed.
    #[must_use]
    pub fn evenly_distributed(count: usize) -> Self {
        Self::all().with_max_frames(count)
    }
}

impl Default for FrameRange {
    fn default() -> Self {
        Self::all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_creation() {
        let range = FrameRange::new(10.0, Some(20.0));
        assert_eq!(range.start, 10.0);
        assert_eq!(range.end, Some(20.0));
    }

    #[test]
    fn test_range_all() {
        let range = FrameRange::all();
        assert_eq!(range.start, 0.0);
        assert_eq!(range.end, None);
    }

    #[test]
    fn test_range_duration() {
        let range = FrameRange::from_duration(5.0, 10.0);
        assert_eq!(range.start, 5.0);
        assert_eq!(range.end, Some(15.0));
        assert_eq!(range.duration(), Some(10.0));
    }

    #[test]
    fn test_contains() {
        let range = FrameRange::new(10.0, Some(20.0));

        assert!(!range.contains(5.0));
        assert!(range.contains(10.0));
        assert!(range.contains(15.0));
        assert!(range.contains(20.0));
        assert!(!range.contains(25.0));
    }

    #[test]
    fn test_frame_count() {
        let range = FrameRange::new(0.0, Some(10.0));
        let count = range.frame_count(100.0, 30.0);
        assert_eq!(count, 300);

        let range = FrameRange::new(0.0, Some(10.0)).with_max_frames(100);
        let count = range.frame_count(100.0, 30.0);
        assert_eq!(count, 100);
    }

    #[test]
    fn test_convenience_methods() {
        let range = FrameRange::one_per_second();
        assert_eq!(range.fps, Some(1.0));

        let range = FrameRange::one_per_minute();
        assert_eq!(range.fps, Some(1.0 / 60.0));

        let range = FrameRange::evenly_distributed(10);
        assert_eq!(range.max_frames, Some(10));
    }
}
