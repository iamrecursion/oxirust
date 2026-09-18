//! Subtitle timing adjustment and synchronization.
//!
//! This module provides utilities for:
//! - Timing offset (shift all subtitles)
//! - Timing scale (speed up/slow down)
//! - Gap management (avoid overlaps)
//! - Duration constraints
//! - Frame-accurate timing

use crate::{Subtitle, SubtitleError, SubtitleResult};

/// Timing adjuster for subtitle synchronization.
pub struct TimingAdjuster {
    /// Minimum subtitle duration in milliseconds.
    pub min_duration_ms: i64,
    /// Maximum subtitle duration in milliseconds.
    pub max_duration_ms: i64,
    /// Minimum gap between subtitles in milliseconds.
    pub min_gap_ms: i64,
    /// Maximum characters per second (for automatic duration).
    pub max_chars_per_second: f32,
}

impl TimingAdjuster {
    /// Create a new timing adjuster with custom parameters.
    #[must_use]
    pub const fn new(
        min_duration_ms: i64,
        max_duration_ms: i64,
        min_gap_ms: i64,
        max_chars_per_second: f32,
    ) -> Self {
        Self {
            min_duration_ms,
            max_duration_ms,
            min_gap_ms,
            max_chars_per_second,
        }
    }

    /// Create a timing adjuster with default parameters.
    ///
    /// Defaults:
    /// - Min duration: 1000ms (1 second)
    /// - Max duration: 7000ms (7 seconds)
    /// - Min gap: 100ms
    /// - Max chars/second: 20
    #[must_use]
    pub const fn default_adjuster() -> Self {
        Self::new(1000, 7000, 100, 20.0)
    }

    /// Adjust a single subtitle's timing.
    pub fn adjust(&self, subtitle: &mut Subtitle) {
        let duration = subtitle.end_time - subtitle.start_time;

        // Calculate optimal duration based on text length
        let text_len = subtitle.text.chars().count() as f32;
        let optimal_duration = ((text_len / self.max_chars_per_second) * 1000.0) as i64;

        // Apply constraints
        let mut new_duration = duration;

        if duration < self.min_duration_ms {
            new_duration = self.min_duration_ms;
        } else if duration > self.max_duration_ms {
            new_duration = self.max_duration_ms;
        }

        // Use optimal duration if within bounds
        if optimal_duration >= self.min_duration_ms && optimal_duration <= self.max_duration_ms {
            new_duration = optimal_duration;
        }

        subtitle.end_time = subtitle.start_time + new_duration;
    }

    /// Adjust timing for a list of subtitles.
    ///
    /// This ensures:
    /// - All subtitles meet duration constraints
    /// - Minimum gaps between subtitles
    /// - No overlaps
    pub fn adjust_list(&mut self, subtitles: &mut [Subtitle]) {
        for i in 0..subtitles.len() {
            self.adjust(&mut subtitles[i]);

            // Ensure gap to next subtitle
            if i + 1 < subtitles.len() {
                let current_end = subtitles[i].end_time;
                let next_start = subtitles[i + 1].start_time;

                if next_start < current_end + self.min_gap_ms {
                    // Shorten current subtitle to make room
                    let required_end = next_start - self.min_gap_ms;
                    if required_end > subtitles[i].start_time {
                        subtitles[i].end_time = required_end;
                    } else {
                        // If we can't fit the gap, move next subtitle
                        subtitles[i + 1].start_time = current_end + self.min_gap_ms;
                        subtitles[i + 1].end_time += self.min_gap_ms;
                    }
                }
            }
        }
    }

    /// Fix overlapping subtitles.
    pub fn fix_overlaps(&self, subtitles: &mut [Subtitle]) {
        for i in 0..subtitles.len().saturating_sub(1) {
            if subtitles[i].end_time > subtitles[i + 1].start_time {
                // Adjust end time to just before next subtitle starts
                subtitles[i].end_time = subtitles[i + 1].start_time - self.min_gap_ms;

                // Ensure minimum duration
                if subtitles[i].end_time < subtitles[i].start_time + self.min_duration_ms {
                    subtitles[i].end_time = subtitles[i].start_time + self.min_duration_ms;
                }
            }
        }
    }
}

impl Default for TimingAdjuster {
    fn default() -> Self {
        Self::default_adjuster()
    }
}

/// Apply a time offset to all subtitles.
///
/// # Examples
///
/// ```
/// # use oximedia_subtitle::{Subtitle, timing::offset_subtitles};
/// let mut subtitles = vec![
///     Subtitle::new(1000, 2000, "Hello".to_string()),
///     Subtitle::new(3000, 4000, "World".to_string()),
/// ];
///
/// // Shift all subtitles forward by 500ms
/// offset_subtitles(&mut subtitles, 500);
/// assert_eq!(subtitles[0].start_time, 1500);
/// assert_eq!(subtitles[0].end_time, 2500);
/// ```
pub fn offset_subtitles(subtitles: &mut [Subtitle], offset_ms: i64) {
    for subtitle in subtitles {
        subtitle.start_time += offset_ms;
        subtitle.end_time += offset_ms;

        // Ensure non-negative times
        if subtitle.start_time < 0 {
            subtitle.end_time -= subtitle.start_time;
            subtitle.start_time = 0;
        }
    }
}

/// Scale subtitle timing by a factor.
///
/// # Examples
///
/// ```
/// # use oximedia_subtitle::{Subtitle, timing::scale_subtitles};
/// let mut subtitles = vec![
///     Subtitle::new(1000, 2000, "Hello".to_string()),
///     Subtitle::new(3000, 4000, "World".to_string()),
/// ];
///
/// // Speed up by 10% (factor 0.9)
/// scale_subtitles(&mut subtitles, 0.9);
/// assert_eq!(subtitles[0].start_time, 900);
/// assert_eq!(subtitles[0].end_time, 1800);
/// ```
pub fn scale_subtitles(subtitles: &mut [Subtitle], scale: f64) {
    for subtitle in subtitles {
        subtitle.start_time = (f64::from(subtitle.start_time as i32) * scale) as i64;
        subtitle.end_time = (f64::from(subtitle.end_time as i32) * scale) as i64;
    }
}

/// Linear time correction between two points.
///
/// This is useful when you know two sync points and want to correct
/// the timing drift between them.
///
/// # Arguments
///
/// * `subtitles` - Subtitles to adjust
/// * `ref1_time` - First reference time in original timing
/// * `ref1_correct` - Correct time for first reference
/// * `ref2_time` - Second reference time in original timing
/// * `ref2_correct` - Correct time for second reference
///
/// # Errors
///
/// Returns error if reference times are equal.
pub fn linear_correction(
    subtitles: &mut [Subtitle],
    ref1_time: i64,
    ref1_correct: i64,
    ref2_time: i64,
    ref2_correct: i64,
) -> SubtitleResult<()> {
    if ref1_time == ref2_time {
        return Err(SubtitleError::InvalidTimestamp(
            "Reference times must be different".to_string(),
        ));
    }

    let slope =
        f64::from((ref2_correct - ref1_correct) as i32) / f64::from((ref2_time - ref1_time) as i32);
    let intercept = f64::from(ref1_correct as i32) - slope * f64::from(ref1_time as i32);

    for subtitle in subtitles {
        subtitle.start_time = (slope * f64::from(subtitle.start_time as i32) + intercept) as i64;
        subtitle.end_time = (slope * f64::from(subtitle.end_time as i32) + intercept) as i64;
    }

    Ok(())
}

/// Convert timestamps to frame numbers.
///
/// # Arguments
///
/// * `timestamp_ms` - Timestamp in milliseconds
/// * `fps` - Frames per second
///
/// # Examples
///
/// ```
/// # use oximedia_subtitle::timing::ms_to_frame;
/// assert_eq!(ms_to_frame(1000, 24.0), 24);
/// assert_eq!(ms_to_frame(500, 30.0), 15);
/// ```
#[must_use]
pub fn ms_to_frame(timestamp_ms: i64, fps: f64) -> i64 {
    ((f64::from(timestamp_ms as i32) / 1000.0) * fps).round() as i64
}

/// Convert frame numbers to timestamps.
///
/// # Arguments
///
/// * `frame` - Frame number
/// * `fps` - Frames per second
///
/// # Examples
///
/// ```
/// # use oximedia_subtitle::timing::frame_to_ms;
/// assert_eq!(frame_to_ms(24, 24.0), 1000);
/// assert_eq!(frame_to_ms(15, 30.0), 500);
/// ```
#[must_use]
pub fn frame_to_ms(frame: i64, fps: f64) -> i64 {
    ((f64::from(frame as i32) / fps) * 1000.0).round() as i64
}

/// Snap subtitle times to frame boundaries.
///
/// This ensures subtitles start and end on exact frame boundaries.
pub fn snap_to_frames(subtitles: &mut [Subtitle], fps: f64) {
    for subtitle in subtitles {
        let start_frame = ms_to_frame(subtitle.start_time, fps);
        let end_frame = ms_to_frame(subtitle.end_time, fps);

        subtitle.start_time = frame_to_ms(start_frame, fps);
        subtitle.end_time = frame_to_ms(end_frame, fps);
    }
}

/// Remove duplicate subtitles.
///
/// Removes subtitles with identical timing and text.
pub fn remove_duplicates(subtitles: &mut Vec<Subtitle>) {
    subtitles.sort_by_key(|s| s.start_time);

    let mut i = 0;
    while i + 1 < subtitles.len() {
        if subtitles[i].start_time == subtitles[i + 1].start_time
            && subtitles[i].end_time == subtitles[i + 1].end_time
            && subtitles[i].text == subtitles[i + 1].text
        {
            subtitles.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

/// Sort subtitles by start time.
pub fn sort_by_time(subtitles: &mut [Subtitle]) {
    subtitles.sort_by_key(|s| s.start_time);
}

/// Merge overlapping or adjacent subtitles with the same text.
pub fn merge_similar(subtitles: &mut Vec<Subtitle>, max_gap_ms: i64) {
    if subtitles.is_empty() {
        return;
    }

    sort_by_time(subtitles);

    let mut i = 0;
    while i + 1 < subtitles.len() {
        let current_end = subtitles[i].end_time;
        let next_start = subtitles[i + 1].start_time;
        let gap = next_start - current_end;

        if gap <= max_gap_ms && subtitles[i].text == subtitles[i + 1].text {
            // Merge
            subtitles[i].end_time = subtitles[i + 1].end_time;
            subtitles.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

/// Split long subtitles into shorter segments.
///
/// # Arguments
///
/// * `subtitles` - Subtitles to split
/// * `max_duration_ms` - Maximum duration per segment
pub fn split_long_subtitles(subtitles: &mut Vec<Subtitle>, max_duration_ms: i64) {
    let mut i = 0;
    while i < subtitles.len() {
        let duration = subtitles[i].end_time - subtitles[i].start_time;

        if duration > max_duration_ms {
            let num_segments = ((duration as f64) / (max_duration_ms as f64)).ceil() as i64;
            let segment_duration = duration / num_segments;

            let original = subtitles[i].clone();
            subtitles[i].end_time = subtitles[i].start_time + segment_duration;

            for seg in 1..num_segments {
                let new_sub = Subtitle {
                    id: None,
                    start_time: original.start_time + seg * segment_duration,
                    end_time: if seg == num_segments - 1 {
                        original.end_time
                    } else {
                        original.start_time + (seg + 1) * segment_duration
                    },
                    text: original.text.clone(),
                    style: original.style.clone(),
                    position: original.position,
                    animations: original.animations.clone(),
                };
                subtitles.insert(i + seg as usize, new_sub);
            }

            i += num_segments as usize;
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_subtitles() {
        let mut subs = vec![
            Subtitle::new(1000, 2000, "Test 1".to_string()),
            Subtitle::new(3000, 4000, "Test 2".to_string()),
        ];

        offset_subtitles(&mut subs, 500);

        assert_eq!(subs[0].start_time, 1500);
        assert_eq!(subs[0].end_time, 2500);
        assert_eq!(subs[1].start_time, 3500);
        assert_eq!(subs[1].end_time, 4500);
    }

    #[test]
    fn test_scale_subtitles() {
        let mut subs = vec![Subtitle::new(1000, 2000, "Test".to_string())];

        scale_subtitles(&mut subs, 2.0);

        assert_eq!(subs[0].start_time, 2000);
        assert_eq!(subs[0].end_time, 4000);
    }

    #[test]
    fn test_ms_to_frame() {
        assert_eq!(ms_to_frame(1000, 24.0), 24);
        assert_eq!(ms_to_frame(1000, 30.0), 30);
        assert_eq!(ms_to_frame(500, 24.0), 12);
    }

    #[test]
    fn test_frame_to_ms() {
        assert_eq!(frame_to_ms(24, 24.0), 1000);
        assert_eq!(frame_to_ms(30, 30.0), 1000);
        assert_eq!(frame_to_ms(12, 24.0), 500);
    }

    #[test]
    fn test_timing_adjuster() {
        let adjuster = TimingAdjuster::default_adjuster();
        let mut sub = Subtitle::new(0, 500, "Short".to_string());

        adjuster.adjust(&mut sub);

        // Should be adjusted to minimum duration
        assert_eq!(sub.end_time - sub.start_time, 1000);
    }

    #[test]
    fn test_fix_overlaps() {
        let adjuster = TimingAdjuster::default_adjuster();
        let mut subs = vec![
            Subtitle::new(1000, 2000, "Test 1".to_string()),
            Subtitle::new(2100, 3100, "Test 2".to_string()),
        ];

        adjuster.fix_overlaps(&mut subs);

        // First subtitle should end before second starts (with gap)
        assert!(subs[0].end_time < subs[1].start_time);
    }

    #[test]
    fn test_remove_duplicates() {
        let mut subs = vec![
            Subtitle::new(1000, 2000, "Test".to_string()),
            Subtitle::new(1000, 2000, "Test".to_string()),
            Subtitle::new(3000, 4000, "Other".to_string()),
        ];

        remove_duplicates(&mut subs);

        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_merge_similar() {
        let mut subs = vec![
            Subtitle::new(1000, 2000, "Test".to_string()),
            Subtitle::new(2100, 3000, "Test".to_string()),
        ];

        merge_similar(&mut subs, 200);

        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].start_time, 1000);
        assert_eq!(subs[0].end_time, 3000);
    }

    #[test]
    fn test_linear_correction() {
        let mut subs = vec![
            Subtitle::new(1000, 2000, "Test 1".to_string()),
            Subtitle::new(3000, 4000, "Test 2".to_string()),
        ];

        // Correct drift: 1000ms -> 1100ms, 3000ms -> 3300ms
        linear_correction(&mut subs, 1000, 1100, 3000, 3300).expect("should succeed in test");

        assert_eq!(subs[0].start_time, 1100);
        assert_eq!(subs[1].start_time, 3300);
    }

    #[test]
    fn test_split_long_subtitles() {
        let mut subs = vec![Subtitle::new(0, 10000, "Long subtitle".to_string())];

        split_long_subtitles(&mut subs, 3000);

        assert!(subs.len() > 1);
        assert!(subs.iter().all(|s| s.end_time - s.start_time <= 3000));
    }
}
