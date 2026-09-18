#![allow(dead_code)]
//! Segment list management for adaptive streaming packagers.
//!
//! Tracks ordered collections of media, init, and index segments together
//! with timing metadata.

/// The role played by a segment in an adaptive stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentType {
    /// Initialisation segment (moov/sidx for fMP4, or similar).
    InitSegment,
    /// A regular media segment carrying audio/video data.
    MediaSegment,
    /// A segment index (sidx box or similar pointer structure).
    IndexSegment,
}

impl SegmentType {
    /// Returns `true` if this type carries decodable media data.
    #[must_use]
    pub fn is_media(self) -> bool {
        matches!(self, Self::MediaSegment)
    }

    /// A short ASCII label for the type.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::InitSegment => "init",
            Self::MediaSegment => "media",
            Self::IndexSegment => "index",
        }
    }
}

impl std::fmt::Display for SegmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Metadata for a single segment.
#[derive(Debug, Clone)]
pub struct MediaSegment {
    /// Absolute or relative URI of the segment.
    uri: String,
    /// Type of segment.
    segment_type: SegmentType,
    /// Duration in milliseconds (0 for init/index segments).
    duration_ms: u64,
    /// Byte offset into the containing file (for byte-range requests).
    byte_offset: Option<u64>,
    /// Byte length (for byte-range requests).
    byte_length: Option<u64>,
    /// Sequence number (optional).
    sequence_number: Option<u64>,
}

impl MediaSegment {
    /// Create a new media segment descriptor.
    pub fn new(uri: impl Into<String>, segment_type: SegmentType, duration_ms: u64) -> Self {
        Self {
            uri: uri.into(),
            segment_type,
            duration_ms,
            byte_offset: None,
            byte_length: None,
            sequence_number: None,
        }
    }

    /// Set byte-range information.
    #[must_use]
    pub fn with_byte_range(mut self, offset: u64, length: u64) -> Self {
        self.byte_offset = Some(offset);
        self.byte_length = Some(length);
        self
    }

    /// Set the sequence number.
    #[must_use]
    pub fn with_sequence(mut self, seq: u64) -> Self {
        self.sequence_number = Some(seq);
        self
    }

    /// Returns the segment URI.
    #[must_use]
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Returns the segment type.
    #[must_use]
    pub fn segment_type(&self) -> SegmentType {
        self.segment_type
    }

    /// Returns the segment duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    /// Returns the byte offset if set.
    #[must_use]
    pub fn byte_offset(&self) -> Option<u64> {
        self.byte_offset
    }

    /// Returns the byte length if set.
    #[must_use]
    pub fn byte_length(&self) -> Option<u64> {
        self.byte_length
    }

    /// Returns the sequence number if set.
    #[must_use]
    pub fn sequence_number(&self) -> Option<u64> {
        self.sequence_number
    }
}

/// An ordered list of segments for a single rendition.
#[derive(Debug, Default)]
pub struct SegmentList {
    segments: Vec<MediaSegment>,
}

impl SegmentList {
    /// Create an empty segment list.
    #[must_use]
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Append a segment to the list.
    pub fn add(&mut self, segment: MediaSegment) {
        self.segments.push(segment);
    }

    /// Returns the total duration of all media segments in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ms).sum()
    }

    /// Returns the number of segments in the list.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns a slice of all segments.
    #[must_use]
    pub fn segments(&self) -> &[MediaSegment] {
        &self.segments
    }

    /// Returns only segments of the given type.
    #[must_use]
    pub fn segments_of_type(&self, t: SegmentType) -> Vec<&MediaSegment> {
        self.segments
            .iter()
            .filter(|s| s.segment_type == t)
            .collect()
    }

    /// Returns `true` if the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Returns the init segment, if any.
    #[must_use]
    pub fn init_segment(&self) -> Option<&MediaSegment> {
        self.segments
            .iter()
            .find(|s| s.segment_type == SegmentType::InitSegment)
    }

    /// Removes segments older than the given sequence number.
    pub fn prune_before_sequence(&mut self, min_seq: u64) {
        self.segments
            .retain(|s| s.sequence_number.map_or(true, |seq| seq >= min_seq));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_type_is_media() {
        assert!(SegmentType::MediaSegment.is_media());
        assert!(!SegmentType::InitSegment.is_media());
        assert!(!SegmentType::IndexSegment.is_media());
    }

    #[test]
    fn test_segment_type_label() {
        assert_eq!(SegmentType::InitSegment.label(), "init");
        assert_eq!(SegmentType::MediaSegment.label(), "media");
        assert_eq!(SegmentType::IndexSegment.label(), "index");
    }

    #[test]
    fn test_segment_type_display() {
        assert_eq!(SegmentType::MediaSegment.to_string(), "media");
    }

    #[test]
    fn test_media_segment_duration() {
        let seg = MediaSegment::new("seg0.m4s", SegmentType::MediaSegment, 4000);
        assert_eq!(seg.duration_ms(), 4000);
    }

    #[test]
    fn test_media_segment_uri() {
        let seg = MediaSegment::new("init.mp4", SegmentType::InitSegment, 0);
        assert_eq!(seg.uri(), "init.mp4");
    }

    #[test]
    fn test_media_segment_byte_range() {
        let seg = MediaSegment::new("stream.mp4", SegmentType::MediaSegment, 2000)
            .with_byte_range(0, 1024);
        assert_eq!(seg.byte_offset(), Some(0));
        assert_eq!(seg.byte_length(), Some(1024));
    }

    #[test]
    fn test_media_segment_sequence() {
        let seg = MediaSegment::new("s1.m4s", SegmentType::MediaSegment, 2000).with_sequence(5);
        assert_eq!(seg.sequence_number(), Some(5));
    }

    #[test]
    fn test_segment_list_empty() {
        let list = SegmentList::new();
        assert!(list.is_empty());
        assert_eq!(list.segment_count(), 0);
        assert_eq!(list.total_duration_ms(), 0);
    }

    #[test]
    fn test_segment_list_add_and_count() {
        let mut list = SegmentList::new();
        list.add(MediaSegment::new("init.mp4", SegmentType::InitSegment, 0));
        list.add(MediaSegment::new("s1.m4s", SegmentType::MediaSegment, 4000));
        list.add(MediaSegment::new("s2.m4s", SegmentType::MediaSegment, 4000));
        assert_eq!(list.segment_count(), 3);
    }

    #[test]
    fn test_segment_list_total_duration() {
        let mut list = SegmentList::new();
        list.add(MediaSegment::new("s1.m4s", SegmentType::MediaSegment, 3000));
        list.add(MediaSegment::new("s2.m4s", SegmentType::MediaSegment, 3500));
        assert_eq!(list.total_duration_ms(), 6500);
    }

    #[test]
    fn test_segment_list_of_type() {
        let mut list = SegmentList::new();
        list.add(MediaSegment::new("init.mp4", SegmentType::InitSegment, 0));
        list.add(MediaSegment::new("s1.m4s", SegmentType::MediaSegment, 2000));
        let media = list.segments_of_type(SegmentType::MediaSegment);
        assert_eq!(media.len(), 1);
    }

    #[test]
    fn test_segment_list_init_segment() {
        let mut list = SegmentList::new();
        list.add(MediaSegment::new("init.mp4", SegmentType::InitSegment, 0));
        assert!(list.init_segment().is_some());
        assert_eq!(
            list.init_segment().expect("should succeed in test").uri(),
            "init.mp4"
        );
    }

    #[test]
    fn test_segment_list_prune() {
        let mut list = SegmentList::new();
        list.add(MediaSegment::new("s0.m4s", SegmentType::MediaSegment, 2000).with_sequence(0));
        list.add(MediaSegment::new("s1.m4s", SegmentType::MediaSegment, 2000).with_sequence(1));
        list.add(MediaSegment::new("s2.m4s", SegmentType::MediaSegment, 2000).with_sequence(2));
        list.prune_before_sequence(1);
        assert_eq!(list.segment_count(), 2);
    }
}
