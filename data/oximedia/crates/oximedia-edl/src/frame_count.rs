#![allow(dead_code)]
//! Frame counting and timecode conversion utilities for EDLs.
//!
//! This module provides [`FrameCount`] for converting between absolute frame
//! numbers and `HH:MM:SS:FF` timecode strings, [`FrameCounter`] for
//! accumulating frame counts, and [`FrameCountRange`] for representing
//! inclusive frame ranges.

use std::fmt;

/// Represents an absolute frame number at a given frame rate.
///
/// The frame count is always non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameCount {
    /// Absolute frame number (0-based).
    frames: u64,
    /// Frames per second (integer, e.g. 24, 25, 30).
    fps: u32,
}

impl FrameCount {
    /// Create a new frame count.
    ///
    /// # Panics
    ///
    /// Panics if `fps` is zero.
    #[must_use]
    pub fn new(frames: u64, fps: u32) -> Self {
        assert!(fps > 0, "fps must be > 0");
        Self { frames, fps }
    }

    /// Create from hours, minutes, seconds, and frames.
    ///
    /// # Panics
    ///
    /// Panics if `fps` is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_hmsf(hours: u32, minutes: u32, seconds: u32, frame: u32, fps: u32) -> Self {
        assert!(fps > 0, "fps must be > 0");
        let total_seconds = u64::from(hours) * 3600 + u64::from(minutes) * 60 + u64::from(seconds);
        let total_frames = total_seconds * u64::from(fps) + u64::from(frame);
        Self {
            frames: total_frames,
            fps,
        }
    }

    /// Absolute frame number.
    #[must_use]
    pub const fn frames(&self) -> u64 {
        self.frames
    }

    /// Frames per second.
    #[must_use]
    pub const fn fps(&self) -> u32 {
        self.fps
    }

    /// Convert to `HH:MM:SS:FF` timecode string.
    #[must_use]
    pub fn to_timecode(&self) -> String {
        let fps64 = u64::from(self.fps);
        let total_seconds = self.frames / fps64;
        let ff = self.frames % fps64;
        let ss = total_seconds % 60;
        let total_minutes = total_seconds / 60;
        let mm = total_minutes % 60;
        let hh = total_minutes / 60;
        format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
    }

    /// Duration in seconds (floating point).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_seconds(&self) -> f64 {
        self.frames as f64 / f64::from(self.fps)
    }

    /// Add frames, returning a new `FrameCount`.
    #[must_use]
    pub fn add_frames(self, n: u64) -> Self {
        Self {
            frames: self.frames.saturating_add(n),
            fps: self.fps,
        }
    }

    /// Subtract frames, clamping at zero.
    #[must_use]
    pub fn sub_frames(self, n: u64) -> Self {
        Self {
            frames: self.frames.saturating_sub(n),
            fps: self.fps,
        }
    }

    /// Difference in frames between two counts (absolute value).
    #[must_use]
    pub fn distance(self, other: Self) -> u64 {
        self.frames.abs_diff(other.frames)
    }
}

impl fmt::Display for FrameCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_timecode())
    }
}

/// An accumulator that tracks a running frame count.
#[derive(Debug, Clone)]
pub struct FrameCounter {
    /// Current accumulated frames.
    current: u64,
    /// Frames per second.
    fps: u32,
    /// Number of additions performed.
    additions: u64,
}

impl FrameCounter {
    /// Create a new counter starting at zero.
    #[must_use]
    pub fn new(fps: u32) -> Self {
        assert!(fps > 0, "fps must be > 0");
        Self {
            current: 0,
            fps,
            additions: 0,
        }
    }

    /// Add frames to the counter.
    pub fn add(&mut self, frames: u64) {
        self.current = self.current.saturating_add(frames);
        self.additions += 1;
    }

    /// Reset to zero.
    pub fn reset(&mut self) {
        self.current = 0;
        self.additions = 0;
    }

    /// Current frame count.
    #[must_use]
    pub const fn current(&self) -> u64 {
        self.current
    }

    /// Number of additions performed.
    #[must_use]
    pub const fn additions(&self) -> u64 {
        self.additions
    }

    /// Snapshot the current value as a [`FrameCount`].
    #[must_use]
    pub fn snapshot(&self) -> FrameCount {
        FrameCount::new(self.current, self.fps)
    }

    /// Duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.current as f64 / f64::from(self.fps)
    }
}

/// An inclusive range of frames `[start, end]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameCountRange {
    /// Start frame count.
    start: FrameCount,
    /// End frame count (inclusive).
    end: FrameCount,
}

impl FrameCountRange {
    /// Create a new frame range.  `start` and `end` must share the same fps.
    ///
    /// # Panics
    ///
    /// Panics if `start.fps != end.fps` or if `start > end`.
    #[must_use]
    pub fn new(start: FrameCount, end: FrameCount) -> Self {
        assert_eq!(start.fps, end.fps, "fps must match");
        assert!(start.frames <= end.frames, "start must be <= end");
        Self { start, end }
    }

    /// Start of the range.
    #[must_use]
    pub const fn start(&self) -> FrameCount {
        self.start
    }

    /// End of the range (inclusive).
    #[must_use]
    pub const fn end(&self) -> FrameCount {
        self.end
    }

    /// Number of frames in the range (inclusive).
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.end.frames - self.start.frames + 1
    }

    /// Duration of the range in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.frame_count() as f64 / f64::from(self.start.fps)
    }

    /// Whether a frame number falls inside this range.
    #[must_use]
    pub fn contains(&self, frame: u64) -> bool {
        frame >= self.start.frames && frame <= self.end.frames
    }

    /// Whether two ranges overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start.frames <= other.end.frames && other.start.frames <= self.end.frames
    }

    /// Start timecode string.
    #[must_use]
    pub fn start_timecode(&self) -> String {
        self.start.to_timecode()
    }

    /// End timecode string.
    #[must_use]
    pub fn end_timecode(&self) -> String {
        self.end.to_timecode()
    }
}

impl fmt::Display for FrameCountRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} - {}", self.start, self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_count_creation() {
        let fc = FrameCount::new(100, 25);
        assert_eq!(fc.frames(), 100);
        assert_eq!(fc.fps(), 25);
    }

    #[test]
    fn test_frame_count_from_hmsf() {
        let fc = FrameCount::from_hmsf(1, 0, 0, 0, 25);
        assert_eq!(fc.frames(), 90_000); // 3600 * 25
    }

    #[test]
    fn test_frame_count_to_timecode() {
        let fc = FrameCount::new(90_000, 25);
        assert_eq!(fc.to_timecode(), "01:00:00:00");

        let fc2 = FrameCount::new(25 * 61 + 12, 25); // 1min 1sec 12fr
        assert_eq!(fc2.to_timecode(), "00:01:01:12");
    }

    #[test]
    fn test_frame_count_to_seconds() {
        let fc = FrameCount::new(75, 25);
        let secs = fc.to_seconds();
        assert!((secs - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_count_add_sub() {
        let fc = FrameCount::new(100, 25);
        assert_eq!(fc.add_frames(50).frames(), 150);
        assert_eq!(fc.sub_frames(30).frames(), 70);
        assert_eq!(fc.sub_frames(200).frames(), 0); // clamped
    }

    #[test]
    fn test_frame_count_distance() {
        let a = FrameCount::new(100, 25);
        let b = FrameCount::new(150, 25);
        assert_eq!(a.distance(b), 50);
        assert_eq!(b.distance(a), 50);
    }

    #[test]
    fn test_frame_count_display() {
        let fc = FrameCount::new(0, 30);
        assert_eq!(format!("{fc}"), "00:00:00:00");
    }

    #[test]
    fn test_frame_count_ordering() {
        let a = FrameCount::new(10, 25);
        let b = FrameCount::new(20, 25);
        assert!(a < b);
    }

    #[test]
    fn test_frame_counter_basic() {
        let mut counter = FrameCounter::new(25);
        assert_eq!(counter.current(), 0);
        counter.add(100);
        counter.add(50);
        assert_eq!(counter.current(), 150);
        assert_eq!(counter.additions(), 2);
    }

    #[test]
    fn test_frame_counter_reset() {
        let mut counter = FrameCounter::new(25);
        counter.add(100);
        counter.reset();
        assert_eq!(counter.current(), 0);
        assert_eq!(counter.additions(), 0);
    }

    #[test]
    fn test_frame_counter_snapshot() {
        let mut counter = FrameCounter::new(30);
        counter.add(90);
        let snap = counter.snapshot();
        assert_eq!(snap.frames(), 90);
        assert_eq!(snap.fps(), 30);
    }

    #[test]
    fn test_frame_counter_duration() {
        let mut counter = FrameCounter::new(25);
        counter.add(75);
        let dur = counter.duration_seconds();
        assert!((dur - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_count_range_basic() {
        let start = FrameCount::new(100, 25);
        let end = FrameCount::new(199, 25);
        let range = FrameCountRange::new(start, end);
        assert_eq!(range.frame_count(), 100);
    }

    #[test]
    fn test_frame_count_range_contains() {
        let range = FrameCountRange::new(FrameCount::new(10, 25), FrameCount::new(20, 25));
        assert!(range.contains(10));
        assert!(range.contains(15));
        assert!(range.contains(20));
        assert!(!range.contains(9));
        assert!(!range.contains(21));
    }

    #[test]
    fn test_frame_count_range_overlaps() {
        let a = FrameCountRange::new(FrameCount::new(0, 25), FrameCount::new(50, 25));
        let b = FrameCountRange::new(FrameCount::new(25, 25), FrameCount::new(75, 25));
        let c = FrameCountRange::new(FrameCount::new(51, 25), FrameCount::new(100, 25));
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_frame_count_range_duration() {
        let range = FrameCountRange::new(FrameCount::new(0, 25), FrameCount::new(24, 25));
        let dur = range.duration_seconds();
        assert!((dur - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_count_range_display() {
        let range = FrameCountRange::new(FrameCount::new(0, 25), FrameCount::new(24, 25));
        let s = format!("{range}");
        assert!(s.contains(" - "));
    }
}
