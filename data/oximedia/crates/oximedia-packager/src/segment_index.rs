#![allow(dead_code)]
//! Segment index for tracking and querying packaged media segments.
//!
//! Provides a fast lookup structure for segment metadata such as byte ranges,
//! timecodes, durations, and keyframe flags. Useful for building DASH `SegmentTimeline`
//! and HLS `#EXTINF` entries.

use std::collections::BTreeMap;
use std::time::Duration;

/// Information about a single indexed segment.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedSegment {
    /// Zero-based segment number.
    pub number: u64,
    /// Presentation start time.
    pub start_time: Duration,
    /// Duration of this segment.
    pub duration: Duration,
    /// Byte offset from the beginning of the media file or container.
    pub byte_offset: u64,
    /// Size in bytes.
    pub byte_size: u64,
    /// Whether this segment starts with a keyframe.
    pub starts_with_keyframe: bool,
    /// Optional URI or path for this segment.
    pub uri: Option<String>,
}

impl IndexedSegment {
    /// Compute the end time (start + duration).
    #[must_use]
    pub fn end_time(&self) -> Duration {
        self.start_time + self.duration
    }

    /// Compute the byte range end (offset + size).
    #[must_use]
    pub fn byte_end(&self) -> u64 {
        self.byte_offset + self.byte_size
    }

    /// Return the byte range as a `(start, end)` tuple.
    #[must_use]
    pub fn byte_range(&self) -> (u64, u64) {
        (self.byte_offset, self.byte_end())
    }

    /// Format the byte range as an HTTP Content-Range header value (without "bytes " prefix).
    #[must_use]
    pub fn content_range_value(&self, total_size: u64) -> String {
        format!(
            "{}-{}/{}",
            self.byte_offset,
            self.byte_end().saturating_sub(1),
            total_size
        )
    }
}

/// A collection of segments indexed by segment number.
#[derive(Debug, Clone)]
pub struct SegmentIndex {
    /// Ordered map: segment number -> segment data.
    segments: BTreeMap<u64, IndexedSegment>,
    /// Total number of bytes across all segments.
    total_bytes: u64,
    /// Total duration across all segments.
    total_duration: Duration,
}

impl SegmentIndex {
    /// Create an empty segment index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: BTreeMap::new(),
            total_bytes: 0,
            total_duration: Duration::ZERO,
        }
    }

    /// Insert a segment into the index.
    pub fn insert(&mut self, segment: IndexedSegment) {
        self.total_bytes += segment.byte_size;
        let end = segment.end_time();
        if end > self.total_duration {
            self.total_duration = end;
        }
        self.segments.insert(segment.number, segment);
    }

    /// Look up a segment by its number.
    #[must_use]
    pub fn get(&self, number: u64) -> Option<&IndexedSegment> {
        self.segments.get(&number)
    }

    /// Return the total number of segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Return `true` if the index contains no segments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Total byte size of all segments.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Total presentation duration.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Average segment duration.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_duration(&self) -> Duration {
        if self.segments.is_empty() {
            return Duration::ZERO;
        }
        let total_ms = self.total_duration.as_millis() as f64;
        let avg_ms = total_ms / self.segments.len() as f64;
        Duration::from_secs_f64(avg_ms / 1000.0)
    }

    /// Average segment byte size.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_byte_size(&self) -> u64 {
        if self.segments.is_empty() {
            return 0;
        }
        self.total_bytes / self.segments.len() as u64
    }

    /// Find the segment that contains the given presentation time.
    #[must_use]
    pub fn find_by_time(&self, time: Duration) -> Option<&IndexedSegment> {
        self.segments
            .values()
            .find(|s| time >= s.start_time && time < s.end_time())
    }

    /// Find the segment that contains the given byte offset.
    #[must_use]
    pub fn find_by_byte_offset(&self, offset: u64) -> Option<&IndexedSegment> {
        self.segments
            .values()
            .find(|s| offset >= s.byte_offset && offset < s.byte_end())
    }

    /// Return all keyframe segments.
    #[must_use]
    pub fn keyframe_segments(&self) -> Vec<&IndexedSegment> {
        self.segments
            .values()
            .filter(|s| s.starts_with_keyframe)
            .collect()
    }

    /// Return an ordered slice of all segments (by number).
    #[must_use]
    pub fn all_segments(&self) -> Vec<&IndexedSegment> {
        self.segments.values().collect()
    }

    /// Remove segments older than a given segment number (for live window).
    pub fn trim_before(&mut self, keep_from: u64) {
        let to_remove: Vec<u64> = self.segments.range(..keep_from).map(|(&k, _)| k).collect();
        for key in to_remove {
            if let Some(seg) = self.segments.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(seg.byte_size);
            }
        }
        // Recalculate total_duration from remaining
        self.total_duration = self
            .segments
            .values()
            .map(IndexedSegment::end_time)
            .max()
            .unwrap_or(Duration::ZERO);
    }

    /// Return the last segment number, or `None` if empty.
    #[must_use]
    pub fn last_number(&self) -> Option<u64> {
        self.segments.keys().next_back().copied()
    }
}

impl Default for SegmentIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn seg(
        num: u64,
        start_ms: u64,
        dur_ms: u64,
        offset: u64,
        size: u64,
        kf: bool,
    ) -> IndexedSegment {
        IndexedSegment {
            number: num,
            start_time: Duration::from_millis(start_ms),
            duration: Duration::from_millis(dur_ms),
            byte_offset: offset,
            byte_size: size,
            starts_with_keyframe: kf,
            uri: Some(format!("segment_{num}.m4s")),
        }
    }

    #[test]
    fn test_empty_index() {
        let idx = SegmentIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.total_bytes(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        assert_eq!(idx.len(), 1);
        assert!(idx.get(0).is_some());
        assert!(idx.get(1).is_none());
    }

    #[test]
    fn test_total_bytes() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        idx.insert(seg(1, 6000, 6000, 5000, 4500, true));
        assert_eq!(idx.total_bytes(), 9500);
    }

    #[test]
    fn test_total_duration() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        idx.insert(seg(1, 6000, 6000, 5000, 4500, true));
        assert_eq!(idx.total_duration(), Duration::from_secs(12));
    }

    #[test]
    fn test_find_by_time() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        idx.insert(seg(1, 6000, 6000, 5000, 4500, true));
        let found = idx.find_by_time(Duration::from_secs(3));
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").number, 0);
        let found2 = idx.find_by_time(Duration::from_secs(7));
        assert_eq!(found2.expect("should succeed in test").number, 1);
    }

    #[test]
    fn test_find_by_time_not_found() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        assert!(idx.find_by_time(Duration::from_secs(99)).is_none());
    }

    #[test]
    fn test_find_by_byte_offset() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        idx.insert(seg(1, 6000, 6000, 5000, 4500, true));
        let found = idx.find_by_byte_offset(2500);
        assert_eq!(found.expect("should succeed in test").number, 0);
        let found2 = idx.find_by_byte_offset(7000);
        assert_eq!(found2.expect("should succeed in test").number, 1);
    }

    #[test]
    fn test_keyframe_segments() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        idx.insert(seg(1, 6000, 6000, 5000, 4500, false));
        idx.insert(seg(2, 12000, 6000, 9500, 5200, true));
        let kf = idx.keyframe_segments();
        assert_eq!(kf.len(), 2);
    }

    #[test]
    fn test_trim_before() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        idx.insert(seg(1, 6000, 6000, 5000, 4500, true));
        idx.insert(seg(2, 12000, 6000, 9500, 5200, true));
        idx.trim_before(2);
        assert_eq!(idx.len(), 1);
        assert!(idx.get(0).is_none());
        assert!(idx.get(2).is_some());
    }

    #[test]
    fn test_last_number() {
        let mut idx = SegmentIndex::new();
        assert!(idx.last_number().is_none());
        idx.insert(seg(0, 0, 6000, 0, 5000, true));
        idx.insert(seg(5, 6000, 6000, 5000, 4500, true));
        assert_eq!(idx.last_number(), Some(5));
    }

    #[test]
    fn test_average_duration() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 4000, 0, 5000, true));
        idx.insert(seg(1, 4000, 8000, 5000, 4500, true));
        let avg = idx.average_duration();
        // total = 12s, 2 segs => 6s average
        assert!((avg.as_secs_f64() - 6.0).abs() < 0.1);
    }

    #[test]
    fn test_average_byte_size() {
        let mut idx = SegmentIndex::new();
        idx.insert(seg(0, 0, 6000, 0, 4000, true));
        idx.insert(seg(1, 6000, 6000, 4000, 6000, true));
        assert_eq!(idx.average_byte_size(), 5000);
    }

    #[test]
    fn test_content_range_value() {
        let s = seg(0, 0, 6000, 100, 500, true);
        let cr = s.content_range_value(10000);
        assert_eq!(cr, "100-599/10000");
    }

    #[test]
    fn test_byte_range() {
        let s = seg(0, 0, 6000, 200, 300, true);
        assert_eq!(s.byte_range(), (200, 500));
    }

    #[test]
    fn test_default_index() {
        let idx = SegmentIndex::default();
        assert!(idx.is_empty());
    }
}
