//! Music structure segmentation.
//!
//! Provides types and utilities for describing the formal structure of a
//! piece of music: intro, verse, chorus, bridge, outro, and more.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// SegmentType
// ---------------------------------------------------------------------------

/// Semantic label for a music section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentType {
    /// Opening section before the first verse.
    Intro,
    /// Narrative verse section.
    Verse,
    /// Pre-chorus build-up section.
    PreChorus,
    /// Hook / chorus section (high energy, often repeated).
    Chorus,
    /// Contrasting bridge section.
    Bridge,
    /// Closing section.
    Outro,
    /// Instrumental solo.
    Solo,
    /// Breakdown / stripped-back section.
    Break,
    /// Section whose type could not be determined.
    Unknown,
}

impl SegmentType {
    /// Short text label for this segment type.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Intro => "intro",
            Self::Verse => "verse",
            Self::PreChorus => "pre-chorus",
            Self::Chorus => "chorus",
            Self::Bridge => "bridge",
            Self::Outro => "outro",
            Self::Solo => "solo",
            Self::Break => "break",
            Self::Unknown => "unknown",
        }
    }

    /// Typical duration in musical bars for this section type.
    #[must_use]
    pub const fn typical_duration_bars(&self) -> u32 {
        match self {
            Self::Verse | Self::Chorus | Self::Solo => 8,
            Self::Intro
            | Self::PreChorus
            | Self::Bridge
            | Self::Outro
            | Self::Break
            | Self::Unknown => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// MusicSegment
// ---------------------------------------------------------------------------

/// A labelled time segment within a piece of music.
#[derive(Debug, Clone, PartialEq)]
pub struct MusicSegment {
    /// Start position in milliseconds.
    pub start_ms: u64,
    /// End position in milliseconds.
    pub end_ms: u64,
    /// Semantic type of this segment.
    pub segment_type: SegmentType,
    /// Confidence score (0.0–1.0).
    pub confidence: f64,
    /// Index used when the same type appears multiple times (0-based).
    pub repeat_index: u32,
}

impl MusicSegment {
    /// Create a new music segment.
    #[must_use]
    pub const fn new(
        start_ms: u64,
        end_ms: u64,
        segment_type: SegmentType,
        confidence: f64,
        repeat_index: u32,
    ) -> Self {
        Self {
            start_ms,
            end_ms,
            segment_type,
            confidence,
            repeat_index,
        }
    }

    /// Duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Return `true` if this segment is a climax section (Chorus or Solo).
    #[must_use]
    pub const fn is_climax(&self) -> bool {
        matches!(self.segment_type, SegmentType::Chorus | SegmentType::Solo)
    }
}

// ---------------------------------------------------------------------------
// StructureAnalysis
// ---------------------------------------------------------------------------

/// Complete structural analysis of a music piece.
#[derive(Debug, Clone)]
pub struct StructureAnalysis {
    /// All detected segments in chronological order.
    pub segments: Vec<MusicSegment>,
    /// Total duration of the piece in milliseconds.
    pub total_ms: u64,
}

impl StructureAnalysis {
    /// Create a new empty structure analysis for a piece of given duration.
    #[must_use]
    pub const fn new(total_ms: u64) -> Self {
        Self {
            segments: Vec::new(),
            total_ms,
        }
    }

    /// Add a segment to the analysis.
    ///
    /// Segments are stored in the order they are added.
    pub fn add_segment(&mut self, seg: MusicSegment) {
        self.segments.push(seg);
    }

    /// Compute the fraction of total duration occupied by chorus sections.
    ///
    /// Returns 0.0 if `total_ms` is 0.
    #[must_use]
    pub fn chorus_ratio(&self) -> f64 {
        if self.total_ms == 0 {
            return 0.0;
        }
        let chorus_ms: u64 = self
            .segments
            .iter()
            .filter(|s| s.segment_type == SegmentType::Chorus)
            .map(MusicSegment::duration_ms)
            .sum();
        chorus_ms as f64 / self.total_ms as f64
    }

    /// Return the duration of the intro section, if one exists.
    #[must_use]
    pub fn intro_duration_ms(&self) -> Option<u64> {
        self.segments
            .iter()
            .find(|s| s.segment_type == SegmentType::Intro)
            .map(MusicSegment::duration_ms)
    }

    /// Return the start time of the first climax section (chorus or solo).
    ///
    /// This can be used to find the "first drop" in electronic music or the
    /// first chorus in a pop song.
    #[must_use]
    pub fn first_drop_ms(&self) -> Option<u64> {
        self.segments
            .iter()
            .find(|s| s.is_climax())
            .map(|s| s.start_ms)
    }

    /// Return the number of chorus sections detected.
    #[must_use]
    pub fn chorus_count(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| s.segment_type == SegmentType::Chorus)
            .count()
    }

    /// Return `true` if an intro section is present.
    #[must_use]
    pub fn has_intro(&self) -> bool {
        self.segments
            .iter()
            .any(|s| s.segment_type == SegmentType::Intro)
    }

    /// Return `true` if an outro section is present.
    #[must_use]
    pub fn has_outro(&self) -> bool {
        self.segments
            .iter()
            .any(|s| s.segment_type == SegmentType::Outro)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: u64, end: u64, t: SegmentType) -> MusicSegment {
        MusicSegment::new(start, end, t, 0.9, 0)
    }

    #[test]
    fn test_segment_type_label() {
        assert_eq!(SegmentType::Intro.label(), "intro");
        assert_eq!(SegmentType::Chorus.label(), "chorus");
        assert_eq!(SegmentType::PreChorus.label(), "pre-chorus");
        assert_eq!(SegmentType::Verse.label(), "verse");
        assert_eq!(SegmentType::Bridge.label(), "bridge");
        assert_eq!(SegmentType::Outro.label(), "outro");
        assert_eq!(SegmentType::Solo.label(), "solo");
        assert_eq!(SegmentType::Break.label(), "break");
        assert_eq!(SegmentType::Unknown.label(), "unknown");
    }

    #[test]
    fn test_segment_type_typical_bars() {
        assert_eq!(SegmentType::Intro.typical_duration_bars(), 4);
        assert_eq!(SegmentType::Verse.typical_duration_bars(), 8);
        assert_eq!(SegmentType::Chorus.typical_duration_bars(), 8);
        assert_eq!(SegmentType::Bridge.typical_duration_bars(), 4);
        assert_eq!(SegmentType::PreChorus.typical_duration_bars(), 4);
    }

    #[test]
    fn test_music_segment_duration_ms() {
        let s = seg(1000, 5000, SegmentType::Verse);
        assert_eq!(s.duration_ms(), 4000);
    }

    #[test]
    fn test_music_segment_duration_saturating() {
        // end < start should not panic
        let s = MusicSegment::new(5000, 1000, SegmentType::Verse, 0.9, 0);
        assert_eq!(s.duration_ms(), 0);
    }

    #[test]
    fn test_music_segment_is_climax_chorus() {
        let s = seg(0, 30_000, SegmentType::Chorus);
        assert!(s.is_climax());
    }

    #[test]
    fn test_music_segment_is_climax_solo() {
        let s = seg(0, 30_000, SegmentType::Solo);
        assert!(s.is_climax());
    }

    #[test]
    fn test_music_segment_is_not_climax() {
        let s = seg(0, 30_000, SegmentType::Verse);
        assert!(!s.is_climax());
    }

    #[test]
    fn test_structure_analysis_new_empty() {
        let sa = StructureAnalysis::new(240_000);
        assert!(sa.segments.is_empty());
        assert_eq!(sa.total_ms, 240_000);
    }

    #[test]
    fn test_structure_analysis_add_segment() {
        let mut sa = StructureAnalysis::new(240_000);
        sa.add_segment(seg(0, 30_000, SegmentType::Intro));
        assert_eq!(sa.segments.len(), 1);
    }

    #[test]
    fn test_structure_analysis_chorus_ratio() {
        let mut sa = StructureAnalysis::new(200_000);
        sa.add_segment(seg(0, 50_000, SegmentType::Verse));
        sa.add_segment(seg(50_000, 100_000, SegmentType::Chorus)); // 50_000 ms
        sa.add_segment(seg(100_000, 150_000, SegmentType::Verse));
        sa.add_segment(seg(150_000, 200_000, SegmentType::Chorus)); // 50_000 ms
                                                                    // 100_000 / 200_000 = 0.5
        let ratio = sa.chorus_ratio();
        assert!((ratio - 0.5).abs() < 1e-9, "Expected 0.5, got {ratio}");
    }

    #[test]
    fn test_structure_analysis_chorus_ratio_no_total() {
        let sa = StructureAnalysis::new(0);
        assert_eq!(sa.chorus_ratio(), 0.0);
    }

    #[test]
    fn test_structure_analysis_intro_duration() {
        let mut sa = StructureAnalysis::new(240_000);
        sa.add_segment(seg(0, 15_000, SegmentType::Intro));
        sa.add_segment(seg(15_000, 60_000, SegmentType::Verse));
        assert_eq!(sa.intro_duration_ms(), Some(15_000));
    }

    #[test]
    fn test_structure_analysis_intro_duration_absent() {
        let mut sa = StructureAnalysis::new(240_000);
        sa.add_segment(seg(0, 60_000, SegmentType::Verse));
        assert_eq!(sa.intro_duration_ms(), None);
    }

    #[test]
    fn test_structure_analysis_first_drop_ms() {
        let mut sa = StructureAnalysis::new(240_000);
        sa.add_segment(seg(0, 15_000, SegmentType::Intro));
        sa.add_segment(seg(15_000, 30_000, SegmentType::Verse));
        sa.add_segment(seg(30_000, 60_000, SegmentType::Chorus));
        assert_eq!(sa.first_drop_ms(), Some(30_000));
    }

    #[test]
    fn test_structure_analysis_first_drop_absent() {
        let mut sa = StructureAnalysis::new(240_000);
        sa.add_segment(seg(0, 60_000, SegmentType::Verse));
        assert_eq!(sa.first_drop_ms(), None);
    }

    #[test]
    fn test_structure_analysis_chorus_count() {
        let mut sa = StructureAnalysis::new(300_000);
        sa.add_segment(seg(0, 30_000, SegmentType::Verse));
        sa.add_segment(seg(30_000, 60_000, SegmentType::Chorus));
        sa.add_segment(seg(60_000, 90_000, SegmentType::Verse));
        sa.add_segment(seg(90_000, 120_000, SegmentType::Chorus));
        assert_eq!(sa.chorus_count(), 2);
    }

    #[test]
    fn test_structure_analysis_has_intro_outro() {
        let mut sa = StructureAnalysis::new(240_000);
        sa.add_segment(seg(0, 15_000, SegmentType::Intro));
        sa.add_segment(seg(15_000, 200_000, SegmentType::Verse));
        sa.add_segment(seg(200_000, 240_000, SegmentType::Outro));
        assert!(sa.has_intro());
        assert!(sa.has_outro());
    }
}
