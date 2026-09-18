//! CPL (Composition Playlist) segment building and introspection.
//!
//! Provides lightweight types for constructing and analysing segments of an
//! IMF Composition Playlist independently of the full CPL parser.

#![allow(dead_code)]

/// The type of essence carried by a virtual track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackKind {
    /// Main picture / video track.
    MainImage,
    /// Main audio track.
    MainAudio,
    /// Subtitle or caption track.
    Subtitle,
    /// Ancillary data track.
    AncillaryData,
}

impl TrackKind {
    /// Returns a short string label used in XML element names.
    #[must_use]
    pub fn xml_label(&self) -> &str {
        match self {
            Self::MainImage => "MainImageSequence",
            Self::MainAudio => "MainAudioSequence",
            Self::Subtitle => "SubtitlesSequence",
            Self::AncillaryData => "AncillaryDataSequence",
        }
    }

    /// Returns `true` for tracks that carry audio essence.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::MainAudio)
    }
}

/// A single resource entry within a segment: a time-range into a track file.
#[derive(Debug, Clone)]
pub struct SegmentResource {
    /// Unique resource UUID (arbitrary string for this lightweight model).
    pub id: String,
    /// UUID of the MXF track file.
    pub asset_id: String,
    /// Edit rate as `(numerator, denominator)`.
    pub edit_rate: (u32, u32),
    /// First frame within the track file to use.
    pub entry_point: u64,
    /// Number of frames to use from the track file.
    pub source_duration: u64,
    /// How many times to repeat this resource.
    pub repeat_count: u32,
}

impl SegmentResource {
    /// Create a new `SegmentResource`.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        edit_rate: (u32, u32),
        entry_point: u64,
        source_duration: u64,
        repeat_count: u32,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            edit_rate,
            entry_point,
            source_duration,
            repeat_count,
        }
    }

    /// Effective duration including repeats.
    #[must_use]
    pub fn effective_duration(&self) -> u64 {
        self.source_duration * u64::from(self.repeat_count)
    }

    /// Returns `true` if the resource refers to a valid (non-empty) asset.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && !self.asset_id.is_empty()
            && self.source_duration > 0
            && self.repeat_count > 0
    }
}

/// A segment within a CPL: an ordered list of resources for one virtual track.
#[derive(Debug, Clone)]
pub struct CplSegment {
    /// Unique segment UUID.
    pub id: String,
    /// Kind of track this segment belongs to.
    pub track_kind: TrackKind,
    /// Ordered resource references.
    pub resources: Vec<SegmentResource>,
}

impl CplSegment {
    /// Create an empty segment.
    #[must_use]
    pub fn new(id: impl Into<String>, track_kind: TrackKind) -> Self {
        Self {
            id: id.into(),
            track_kind,
            resources: Vec::new(),
        }
    }

    /// Append a resource to the segment.
    pub fn push(&mut self, resource: SegmentResource) {
        self.resources.push(resource);
    }

    /// Total effective duration of all resources.
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.resources.iter().map(|r| r.effective_duration()).sum()
    }

    /// Number of resource references.
    #[must_use]
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Validate all contained resources; returns a list of error strings.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.id.is_empty() {
            errors.push("Segment has an empty ID".to_string());
        }
        for (i, r) in self.resources.iter().enumerate() {
            if !r.is_valid() {
                errors.push(format!("Resource at index {i} is invalid"));
            }
        }
        errors
    }
}

/// A sequence of CPL segments making up a full composition.
#[derive(Debug, Clone, Default)]
pub struct CplSequence {
    /// Ordered segments.
    pub segments: Vec<CplSegment>,
}

impl CplSequence {
    /// Create an empty sequence.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a segment to the sequence.
    pub fn add_segment(&mut self, segment: CplSegment) {
        self.segments.push(segment);
    }

    /// Total duration across all segments.
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.segments.iter().map(|s| s.total_duration()).sum()
    }

    /// Count of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Run validation on all segments and collect errors.
    #[must_use]
    pub fn validate_all(&self) -> Vec<String> {
        self.segments.iter().flat_map(|s| s.validate()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resource(id: &str, asset: &str, dur: u64, repeat: u32) -> SegmentResource {
        SegmentResource::new(id, asset, (24, 1), 0, dur, repeat)
    }

    // --- TrackKind ---

    #[test]
    fn test_track_kind_xml_labels() {
        assert_eq!(TrackKind::MainImage.xml_label(), "MainImageSequence");
        assert_eq!(TrackKind::MainAudio.xml_label(), "MainAudioSequence");
        assert_eq!(TrackKind::Subtitle.xml_label(), "SubtitlesSequence");
        assert_eq!(
            TrackKind::AncillaryData.xml_label(),
            "AncillaryDataSequence"
        );
    }

    #[test]
    fn test_track_kind_is_audio() {
        assert!(TrackKind::MainAudio.is_audio());
        assert!(!TrackKind::MainImage.is_audio());
        assert!(!TrackKind::Subtitle.is_audio());
    }

    // --- SegmentResource ---

    #[test]
    fn test_resource_effective_duration_with_repeat() {
        let r = make_resource("r1", "a1", 100, 3);
        assert_eq!(r.effective_duration(), 300);
    }

    #[test]
    fn test_resource_effective_duration_no_repeat() {
        let r = make_resource("r1", "a1", 50, 1);
        assert_eq!(r.effective_duration(), 50);
    }

    #[test]
    fn test_resource_is_valid_true() {
        let r = make_resource("r1", "a1", 100, 1);
        assert!(r.is_valid());
    }

    #[test]
    fn test_resource_is_valid_false_empty_id() {
        let r = make_resource("", "a1", 100, 1);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_resource_is_valid_false_zero_duration() {
        let r = make_resource("r1", "a1", 0, 1);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_resource_is_valid_false_zero_repeat() {
        let r = make_resource("r1", "a1", 100, 0);
        assert!(!r.is_valid());
    }

    // --- CplSegment ---

    #[test]
    fn test_segment_starts_empty() {
        let seg = CplSegment::new("seg-1", TrackKind::MainImage);
        assert_eq!(seg.resource_count(), 0);
        assert_eq!(seg.total_duration(), 0);
    }

    #[test]
    fn test_segment_push_and_duration() {
        let mut seg = CplSegment::new("seg-1", TrackKind::MainImage);
        seg.push(make_resource("r1", "a1", 100, 1));
        seg.push(make_resource("r2", "a2", 50, 2));
        assert_eq!(seg.resource_count(), 2);
        assert_eq!(seg.total_duration(), 200); // 100 + 50*2
    }

    #[test]
    fn test_segment_validate_valid() {
        let mut seg = CplSegment::new("seg-1", TrackKind::MainAudio);
        seg.push(make_resource("r1", "a1", 200, 1));
        assert!(seg.validate().is_empty());
    }

    #[test]
    fn test_segment_validate_empty_id() {
        let seg = CplSegment::new("", TrackKind::MainImage);
        let errors = seg.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("empty ID"));
    }

    #[test]
    fn test_segment_validate_invalid_resource() {
        let mut seg = CplSegment::new("seg-1", TrackKind::MainImage);
        seg.push(make_resource("r1", "", 100, 1)); // empty asset_id → invalid
        let errors = seg.validate();
        assert!(!errors.is_empty());
    }

    // --- CplSequence ---

    #[test]
    fn test_sequence_starts_empty() {
        let seq = CplSequence::new();
        assert_eq!(seq.segment_count(), 0);
        assert_eq!(seq.total_duration(), 0);
    }

    #[test]
    fn test_sequence_add_segment_and_total_duration() {
        let mut seq = CplSequence::new();
        let mut seg = CplSegment::new("s1", TrackKind::MainImage);
        seg.push(make_resource("r1", "a1", 100, 1));
        seq.add_segment(seg);

        let mut seg2 = CplSegment::new("s2", TrackKind::MainImage);
        seg2.push(make_resource("r2", "a2", 200, 1));
        seq.add_segment(seg2);

        assert_eq!(seq.segment_count(), 2);
        assert_eq!(seq.total_duration(), 300);
    }

    #[test]
    fn test_sequence_validate_all_clean() {
        let mut seq = CplSequence::new();
        let mut seg = CplSegment::new("s1", TrackKind::MainAudio);
        seg.push(make_resource("r1", "a1", 48000, 1));
        seq.add_segment(seg);
        assert!(seq.validate_all().is_empty());
    }

    #[test]
    fn test_sequence_validate_all_propagates_errors() {
        let mut seq = CplSequence::new();
        seq.add_segment(CplSegment::new("", TrackKind::MainImage)); // invalid: empty ID
        let errors = seq.validate_all();
        assert!(!errors.is_empty());
    }
}
