//! Segment timeline management for DASH live streaming.
//!
//! This module provides structures and utilities for managing segment timelines
//! in DASH live streaming scenarios, including dynamic timeline updates and
//! segment number tracking.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use crate::dash::mpd::{SegmentTimeline, SegmentTimelineEntry};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Maximum number of timeline entries to keep in memory.
const MAX_TIMELINE_ENTRIES: usize = 1000;

/// Manages a dynamic segment timeline for live streaming.
///
/// This structure maintains a rolling window of segment timeline entries,
/// updating as new segments are generated and old ones are removed.
#[derive(Debug, Clone)]
pub struct TimelineManager {
    /// Timeline entries in presentation order.
    entries: VecDeque<TimelineEntry>,
    /// Timescale (units per second).
    timescale: u32,
    /// Target segment duration in timescale units.
    target_duration: u64,
    /// Current segment number.
    current_number: u64,
    /// Current presentation time in timescale units.
    current_time: u64,
    /// Availability start time (when the stream started).
    availability_start: SystemTime,
    /// Maximum number of entries to retain.
    max_entries: usize,
}

/// A single timeline entry with metadata.
#[derive(Debug, Clone, Copy)]
struct TimelineEntry {
    /// Entry number.
    number: u64,
    /// Start time in timescale units.
    start: u64,
    /// Duration in timescale units.
    duration: u64,
    /// Wall clock time when this segment was created.
    wall_clock_time: SystemTime,
}

impl TimelineManager {
    /// Creates a new timeline manager.
    ///
    /// # Arguments
    ///
    /// * `timescale` - Timescale in units per second
    /// * `target_duration` - Target segment duration
    /// * `availability_start` - Stream start time
    #[must_use]
    pub fn new(timescale: u32, target_duration: Duration, availability_start: SystemTime) -> Self {
        let target_duration_units = Self::duration_to_units(target_duration, timescale);
        Self {
            entries: VecDeque::new(),
            timescale,
            target_duration: target_duration_units,
            current_number: 1,
            current_time: 0,
            availability_start,
            max_entries: MAX_TIMELINE_ENTRIES,
        }
    }

    /// Adds a new segment to the timeline.
    ///
    /// # Arguments
    ///
    /// * `duration` - Actual duration of the segment
    ///
    /// # Returns
    ///
    /// The segment number for the added segment.
    pub fn add_segment(&mut self, duration: Duration) -> u64 {
        let duration_units = Self::duration_to_units(duration, self.timescale);
        let number = self.current_number;

        let entry = TimelineEntry {
            number,
            start: self.current_time,
            duration: duration_units,
            wall_clock_time: SystemTime::now(),
        };

        self.entries.push_back(entry);

        // Update state
        self.current_time += duration_units;
        self.current_number += 1;

        // Trim old entries if necessary
        while self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }

        number
    }

    /// Removes segments older than the specified duration from the timeline.
    ///
    /// This is used to implement the time-shift buffer depth.
    ///
    /// # Arguments
    ///
    /// * `buffer_depth` - How far back in time to keep segments
    pub fn trim_old_segments(&mut self, buffer_depth: Duration) {
        let buffer_depth_units = Self::duration_to_units(buffer_depth, self.timescale);

        if self.current_time <= buffer_depth_units {
            return;
        }

        let cutoff_time = self.current_time - buffer_depth_units;

        while let Some(entry) = self.entries.front() {
            if entry.start + entry.duration <= cutoff_time {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    /// Returns the current segment number.
    #[must_use]
    pub const fn current_number(&self) -> u64 {
        self.current_number
    }

    /// Returns the current presentation time in seconds.
    #[must_use]
    pub fn current_time_secs(&self) -> f64 {
        self.current_time as f64 / self.timescale as f64
    }

    /// Returns the current presentation time in timescale units.
    #[must_use]
    pub const fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Returns the timescale.
    #[must_use]
    pub const fn timescale(&self) -> u32 {
        self.timescale
    }

    /// Returns the number of segments in the timeline.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the earliest segment number available.
    #[must_use]
    pub fn earliest_segment_number(&self) -> Option<u64> {
        self.entries.front().map(|e| e.number)
    }

    /// Returns the latest segment number available.
    #[must_use]
    pub fn latest_segment_number(&self) -> Option<u64> {
        self.entries.back().map(|e| e.number)
    }

    /// Generates a `SegmentTimeline` for MPD generation.
    ///
    /// This creates a compact representation by grouping consecutive segments
    /// with the same duration using repeat counts.
    #[must_use]
    pub fn to_segment_timeline(&self) -> SegmentTimeline {
        let mut timeline = SegmentTimeline::new();

        if self.entries.is_empty() {
            return timeline;
        }

        let mut current_duration = 0u64;
        let mut current_start = 0u64;
        let mut repeat_count = -1i32;

        for (i, entry) in self.entries.iter().enumerate() {
            if i == 0 {
                current_duration = entry.duration;
                current_start = entry.start;
                repeat_count = 0;
            } else if entry.duration == current_duration {
                repeat_count += 1;
            } else {
                // Flush current entry
                let timeline_entry = SegmentTimelineEntry::new(current_duration)
                    .with_start(current_start)
                    .with_repeat(repeat_count);
                timeline.add_entry(timeline_entry);

                // Start new entry
                current_duration = entry.duration;
                current_start = entry.start;
                repeat_count = 0;
            }
        }

        // Flush final entry
        if !self.entries.is_empty() {
            let timeline_entry = SegmentTimelineEntry::new(current_duration)
                .with_start(current_start)
                .with_repeat(repeat_count);
            timeline.add_entry(timeline_entry);
        }

        timeline
    }

    /// Returns segment information for a specific segment number.
    #[must_use]
    pub fn get_segment_info(&self, number: u64) -> Option<SegmentInfo> {
        self.entries
            .iter()
            .find(|e| e.number == number)
            .map(|e| SegmentInfo {
                number: e.number,
                start_time: e.start,
                duration: e.duration,
                timescale: self.timescale,
            })
    }

    /// Returns all segment numbers currently in the timeline.
    pub fn segment_numbers(&self) -> Vec<u64> {
        self.entries.iter().map(|e| e.number).collect()
    }

    /// Returns the availability start time.
    #[must_use]
    pub const fn availability_start_time(&self) -> SystemTime {
        self.availability_start
    }

    /// Returns the wall clock time for a specific segment.
    #[must_use]
    pub fn segment_wall_clock_time(&self, number: u64) -> Option<SystemTime> {
        self.entries
            .iter()
            .find(|e| e.number == number)
            .map(|e| e.wall_clock_time)
    }

    /// Returns the publish time (time of the latest segment).
    #[must_use]
    pub fn publish_time(&self) -> SystemTime {
        self.entries
            .back()
            .map(|e| e.wall_clock_time)
            .unwrap_or(self.availability_start)
    }

    /// Returns the presentation time offset.
    ///
    /// This is the start time of the earliest available segment.
    #[must_use]
    pub fn presentation_time_offset(&self) -> u64 {
        self.entries.front().map(|e| e.start).unwrap_or(0)
    }

    /// Sets the maximum number of timeline entries to retain.
    pub fn set_max_entries(&mut self, max: usize) {
        self.max_entries = max;
        while self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
    }

    /// Converts a duration to timescale units.
    fn duration_to_units(duration: Duration, timescale: u32) -> u64 {
        (duration.as_secs_f64() * f64::from(timescale)) as u64
    }

    /// Formats a system time as ISO 8601.
    #[must_use]
    pub fn format_system_time(time: SystemTime) -> String {
        let duration_since_epoch = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);

        let secs = duration_since_epoch.as_secs();
        let millis = duration_since_epoch.subsec_millis();

        // Simple ISO 8601 formatting
        // This is a simplified version; a full implementation would use chrono
        let days_since_epoch = secs / 86400;
        let year = 1970 + (days_since_epoch / 365); // Simplified year calculation
        let day_of_year = days_since_epoch % 365;
        let month = (day_of_year / 30) + 1; // Very simplified month
        let day = (day_of_year % 30) + 1;

        let time_of_day = secs % 86400;
        let hour = time_of_day / 3600;
        let minute = (time_of_day % 3600) / 60;
        let second = time_of_day % 60;

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            year, month, day, hour, minute, second, millis
        )
    }
}

/// Simplified segment information.
#[derive(Debug, Clone, Copy)]
pub struct SegmentInfo {
    /// Segment number.
    pub number: u64,
    /// Start time in timescale units.
    pub start_time: u64,
    /// Duration in timescale units.
    pub duration: u64,
    /// Timescale.
    pub timescale: u32,
}

impl SegmentInfo {
    /// Returns the start time in seconds.
    #[must_use]
    pub fn start_time_secs(&self) -> f64 {
        self.start_time as f64 / self.timescale as f64
    }

    /// Returns the duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.duration as f64 / self.timescale as f64
    }

    /// Returns the end time in timescale units.
    #[must_use]
    pub const fn end_time(&self) -> u64 {
        self.start_time + self.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_manager_creation() {
        let start = SystemTime::now();
        let manager = TimelineManager::new(90000, Duration::from_secs(2), start);

        assert_eq!(manager.timescale(), 90000);
        assert_eq!(manager.current_number(), 1);
        assert_eq!(manager.current_time(), 0);
        assert_eq!(manager.segment_count(), 0);
    }

    #[test]
    fn test_add_segments() {
        let start = SystemTime::now();
        let mut manager = TimelineManager::new(90000, Duration::from_secs(2), start);

        let num1 = manager.add_segment(Duration::from_secs(2));
        assert_eq!(num1, 1);
        assert_eq!(manager.current_number(), 2);
        assert_eq!(manager.segment_count(), 1);

        let num2 = manager.add_segment(Duration::from_secs(2));
        assert_eq!(num2, 2);
        assert_eq!(manager.segment_count(), 2);
    }

    #[test]
    fn test_trim_old_segments() {
        let start = SystemTime::now();
        let mut manager = TimelineManager::new(90000, Duration::from_secs(2), start);

        // Add 10 segments
        for _ in 0..10 {
            manager.add_segment(Duration::from_secs(2));
        }
        assert_eq!(manager.segment_count(), 10);

        // Trim to keep only 10 seconds (5 segments)
        manager.trim_old_segments(Duration::from_secs(10));
        assert_eq!(manager.segment_count(), 5);
    }

    #[test]
    fn test_segment_timeline_generation() {
        let start = SystemTime::now();
        let mut manager = TimelineManager::new(90000, Duration::from_secs(2), start);

        // Add segments with consistent duration
        for _ in 0..5 {
            manager.add_segment(Duration::from_secs(2));
        }

        let timeline = manager.to_segment_timeline();
        assert!(!timeline.entries.is_empty());
    }

    #[test]
    fn test_get_segment_info() {
        let start = SystemTime::now();
        let mut manager = TimelineManager::new(90000, Duration::from_secs(2), start);

        manager.add_segment(Duration::from_secs(2));
        manager.add_segment(Duration::from_secs(2));

        let info = manager.get_segment_info(1).expect("should succeed in test");
        assert_eq!(info.number, 1);
        assert_eq!(info.start_time, 0);
    }

    #[test]
    fn test_earliest_latest_segments() {
        let start = SystemTime::now();
        let mut manager = TimelineManager::new(90000, Duration::from_secs(2), start);

        manager.add_segment(Duration::from_secs(2));
        manager.add_segment(Duration::from_secs(2));
        manager.add_segment(Duration::from_secs(2));

        assert_eq!(manager.earliest_segment_number(), Some(1));
        assert_eq!(manager.latest_segment_number(), Some(3));
    }
}
