#![allow(dead_code)]
//! Cue timing calculations and adjustments for subtitles.
//!
//! Provides utilities for shifting, scaling, snapping, and validating
//! subtitle cue timing, including frame-rate conversion and gap analysis.

/// A time span in milliseconds representing a cue's start and end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CueSpan {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
}

impl CueSpan {
    /// Create a new cue span.
    pub fn new(start_ms: i64, end_ms: i64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration in milliseconds.
    pub fn duration(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Check if this span overlaps with another.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Check if a timestamp falls within this span.
    pub fn contains(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }

    /// Gap (in ms) between this span's end and the next span's start.
    /// Returns negative if they overlap.
    pub fn gap_to(&self, next: &Self) -> i64 {
        next.start_ms - self.end_ms
    }

    /// Shift both start and end by a signed offset in milliseconds.
    pub fn shift(&self, offset_ms: i64) -> Self {
        Self {
            start_ms: self.start_ms + offset_ms,
            end_ms: self.end_ms + offset_ms,
        }
    }

    /// Scale both start and end around the origin by a factor.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            start_ms: (self.start_ms as f64 * factor).round() as i64,
            end_ms: (self.end_ms as f64 * factor).round() as i64,
        }
    }

    /// Clamp both start and end to a valid range.
    pub fn clamp(&self, min_ms: i64, max_ms: i64) -> Self {
        Self {
            start_ms: self.start_ms.max(min_ms).min(max_ms),
            end_ms: self.end_ms.max(min_ms).min(max_ms),
        }
    }

    /// Check if start is before end and both are non-negative.
    pub fn is_valid(&self) -> bool {
        self.start_ms >= 0 && self.end_ms > self.start_ms
    }
}

/// Snap a timestamp to the nearest frame boundary.
///
/// `fps` is the frame rate (e.g. 23.976, 24.0, 25.0, 29.97, 30.0).
pub fn snap_to_frame(timestamp_ms: i64, fps: f64) -> i64 {
    if fps <= 0.0 {
        return timestamp_ms;
    }
    let frame_duration_ms = 1000.0 / fps;
    let frame_index = (timestamp_ms as f64 / frame_duration_ms).round();
    (frame_index * frame_duration_ms).round() as i64
}

/// Convert timing from one frame rate to another.
pub fn convert_frame_rate(timestamp_ms: i64, from_fps: f64, to_fps: f64) -> i64 {
    if from_fps <= 0.0 || to_fps <= 0.0 {
        return timestamp_ms;
    }
    let frame_index = timestamp_ms as f64 * from_fps / 1000.0;
    (frame_index * 1000.0 / to_fps).round() as i64
}

/// Compute the minimum gap between consecutive cue spans (sorted by start time).
pub fn minimum_gap(spans: &[CueSpan]) -> Option<i64> {
    if spans.len() < 2 {
        return None;
    }
    let mut min = i64::MAX;
    for w in spans.windows(2) {
        let gap = w[0].gap_to(&w[1]);
        if gap < min {
            min = gap;
        }
    }
    Some(min)
}

/// Enforce a minimum gap between consecutive cues by adjusting end times.
///
/// Returns adjusted spans. Spans must be sorted by start time.
pub fn enforce_minimum_gap(spans: &[CueSpan], min_gap_ms: i64) -> Vec<CueSpan> {
    if spans.is_empty() {
        return Vec::new();
    }
    let mut result = vec![spans[0]];
    for &current in &spans[1..] {
        let prev = result
            .last()
            .expect("result is non-empty: initialized with spans[0]");
        let gap = prev.gap_to(&current);
        if gap < min_gap_ms {
            // Shorten the previous cue's end to create the gap
            let adjusted_end = current.start_ms - min_gap_ms;
            let last = result
                .last_mut()
                .expect("result is non-empty: initialized with spans[0]");
            if adjusted_end > last.start_ms {
                last.end_ms = adjusted_end;
            }
        }
        result.push(current);
    }
    result
}

/// Shift all cue spans by a fixed offset.
pub fn shift_all(spans: &[CueSpan], offset_ms: i64) -> Vec<CueSpan> {
    spans.iter().map(|s| s.shift(offset_ms)).collect()
}

/// Compute total display time of all cue spans.
pub fn total_display_time(spans: &[CueSpan]) -> i64 {
    spans.iter().map(|s| s.duration().max(0)).sum()
}

/// Compute average cue duration.
pub fn average_duration(spans: &[CueSpan]) -> Option<f64> {
    if spans.is_empty() {
        return None;
    }
    let total: i64 = spans.iter().map(|s| s.duration()).sum();
    Some(total as f64 / spans.len() as f64)
}

/// Find overlapping pairs in a list of cue spans (sorted by start time).
pub fn find_overlaps(spans: &[CueSpan]) -> Vec<(usize, usize)> {
    let mut overlaps = Vec::new();
    for i in 0..spans.len() {
        for j in (i + 1)..spans.len() {
            if spans[j].start_ms >= spans[i].end_ms {
                break;
            }
            if spans[i].overlaps(&spans[j]) {
                overlaps.push((i, j));
            }
        }
    }
    overlaps
}

/// Compute the density of subtitles as a ratio of display time to total duration.
pub fn subtitle_density(spans: &[CueSpan], total_duration_ms: i64) -> f64 {
    if total_duration_ms <= 0 {
        return 0.0;
    }
    let display = total_display_time(spans);
    display as f64 / total_duration_ms as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cue_span_duration() {
        let s = CueSpan::new(1000, 3000);
        assert_eq!(s.duration(), 2000);
    }

    #[test]
    fn test_cue_span_overlaps() {
        let a = CueSpan::new(1000, 3000);
        let b = CueSpan::new(2000, 4000);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_cue_span_no_overlap() {
        let a = CueSpan::new(1000, 2000);
        let b = CueSpan::new(2000, 3000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_cue_span_contains() {
        let s = CueSpan::new(1000, 3000);
        assert!(s.contains(1500));
        assert!(s.contains(1000));
        assert!(!s.contains(3000));
        assert!(!s.contains(500));
    }

    #[test]
    fn test_cue_span_gap_to() {
        let a = CueSpan::new(1000, 2000);
        let b = CueSpan::new(2500, 3500);
        assert_eq!(a.gap_to(&b), 500);
    }

    #[test]
    fn test_cue_span_shift() {
        let s = CueSpan::new(1000, 2000);
        let shifted = s.shift(500);
        assert_eq!(shifted.start_ms, 1500);
        assert_eq!(shifted.end_ms, 2500);
    }

    #[test]
    fn test_cue_span_scale() {
        let s = CueSpan::new(1000, 2000);
        let scaled = s.scale(2.0);
        assert_eq!(scaled.start_ms, 2000);
        assert_eq!(scaled.end_ms, 4000);
    }

    #[test]
    fn test_cue_span_clamp() {
        let s = CueSpan::new(-100, 5000);
        let clamped = s.clamp(0, 4000);
        assert_eq!(clamped.start_ms, 0);
        assert_eq!(clamped.end_ms, 4000);
    }

    #[test]
    fn test_cue_span_is_valid() {
        assert!(CueSpan::new(0, 1000).is_valid());
        assert!(!CueSpan::new(1000, 1000).is_valid());
        assert!(!CueSpan::new(-1, 1000).is_valid());
    }

    #[test]
    fn test_snap_to_frame_24fps() {
        let snapped = snap_to_frame(1042, 24.0);
        // Frame duration ~41.667ms, frame 25 = 1042ms -> 25 * 41.667 = 1041.67
        let frame_dur: f64 = 1000.0 / 24.0;
        let expected = (((1042.0_f64 / frame_dur).round()) * frame_dur).round() as i64;
        assert_eq!(snapped, expected);
    }

    #[test]
    fn test_snap_to_frame_zero_fps() {
        assert_eq!(snap_to_frame(1000, 0.0), 1000);
    }

    #[test]
    fn test_convert_frame_rate() {
        // 1 second at 24fps -> should still be ~1 second at 25fps
        let converted = convert_frame_rate(1000, 24.0, 25.0);
        // 1000ms * 24fps / 1000 = 24 frames. 24 frames * 1000 / 25fps = 960ms
        assert_eq!(converted, 960);
    }

    #[test]
    fn test_minimum_gap() {
        let spans = vec![
            CueSpan::new(0, 1000),
            CueSpan::new(1200, 2000),
            CueSpan::new(2100, 3000),
        ];
        assert_eq!(minimum_gap(&spans), Some(100));
    }

    #[test]
    fn test_minimum_gap_single() {
        assert!(minimum_gap(&[CueSpan::new(0, 1000)]).is_none());
    }

    #[test]
    fn test_enforce_minimum_gap() {
        let spans = vec![CueSpan::new(0, 1000), CueSpan::new(1050, 2000)];
        let adjusted = enforce_minimum_gap(&spans, 100);
        assert_eq!(adjusted[0].end_ms, 950);
        assert_eq!(adjusted[1].start_ms, 1050);
    }

    #[test]
    fn test_shift_all() {
        let spans = vec![CueSpan::new(0, 1000), CueSpan::new(2000, 3000)];
        let shifted = shift_all(&spans, 500);
        assert_eq!(shifted[0].start_ms, 500);
        assert_eq!(shifted[1].start_ms, 2500);
    }

    #[test]
    fn test_total_display_time() {
        let spans = vec![CueSpan::new(0, 1000), CueSpan::new(2000, 3500)];
        assert_eq!(total_display_time(&spans), 2500);
    }

    #[test]
    fn test_average_duration() {
        let spans = vec![CueSpan::new(0, 1000), CueSpan::new(2000, 4000)];
        let avg = average_duration(&spans).expect("should succeed in test");
        assert!((avg - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_duration_empty() {
        assert!(average_duration(&[]).is_none());
    }

    #[test]
    fn test_find_overlaps() {
        let spans = vec![
            CueSpan::new(0, 2000),
            CueSpan::new(1000, 3000),
            CueSpan::new(4000, 5000),
        ];
        let overlaps = find_overlaps(&spans);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0], (0, 1));
    }

    #[test]
    fn test_subtitle_density() {
        let spans = vec![CueSpan::new(0, 500), CueSpan::new(500, 1000)];
        let density = subtitle_density(&spans, 2000);
        assert!((density - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_subtitle_density_zero_duration() {
        assert!((subtitle_density(&[], 0) - 0.0).abs() < f64::EPSILON);
    }
}
