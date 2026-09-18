#![allow(dead_code)]
//! High-level music summary — intro/verse/chorus markers derived from structure analysis.

/// Identifies a broad structural role for a section of a song.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionRole {
    /// Opening of the track (before first verse or chorus).
    Intro,
    /// Verse section — carries the narrative.
    Verse,
    /// Chorus — main hook, typically highest energy.
    Chorus,
    /// Bridge or mid-section contrast.
    Bridge,
    /// Outro / fade-out.
    Outro,
    /// Section whose role could not be determined.
    Unknown,
}

impl SectionRole {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Intro => "Intro",
            Self::Verse => "Verse",
            Self::Chorus => "Chorus",
            Self::Bridge => "Bridge",
            Self::Outro => "Outro",
            Self::Unknown => "Unknown",
        }
    }
}

/// A single structural section within a song summary.
#[derive(Debug, Clone)]
pub struct SummarySection {
    /// Role classification of this section.
    pub role: SectionRole,
    /// Start position in milliseconds.
    pub start_ms: u32,
    /// End position in milliseconds.
    pub end_ms: u32,
    /// Confidence score 0.0–1.0 for the role assignment.
    pub confidence: f32,
}

impl SummarySection {
    /// Create a new section.
    #[must_use]
    pub fn new(role: SectionRole, start_ms: u32, end_ms: u32, confidence: f32) -> Self {
        Self {
            role,
            start_ms,
            end_ms,
            confidence,
        }
    }

    /// Duration of this section in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` when this section alone represents a complete, minimal song
    /// (i.e. its role is Unknown and it covers > 90 s of content).
    #[must_use]
    pub fn represents_full_song(&self) -> bool {
        self.role == SectionRole::Unknown && self.duration_ms() > 90_000
    }
}

/// An ordered list of summary sections.
#[derive(Debug, Clone, Default)]
pub struct SummarySectionList {
    sections: Vec<SummarySection>,
}

impl SummarySectionList {
    /// Create an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a section. Sections should be added in chronological order.
    pub fn add(&mut self, section: SummarySection) {
        self.sections.push(section);
    }

    /// Total duration covered by all sections, in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u32 {
        self.sections.iter().map(SummarySection::duration_ms).sum()
    }

    /// Number of sections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Returns `true` when the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Iterate over sections.
    pub fn iter(&self) -> impl Iterator<Item = &SummarySection> {
        self.sections.iter()
    }
}

/// High-level structural summary of a complete music track.
#[derive(Debug, Clone)]
pub struct MusicSummary {
    /// All detected sections in chronological order.
    pub sections: SummarySectionList,
    /// Total track duration in milliseconds.
    pub total_duration_ms: u32,
}

impl MusicSummary {
    /// Build a summary from a `SummarySectionList` and total duration.
    #[must_use]
    pub fn new(sections: SummarySectionList, total_duration_ms: u32) -> Self {
        Self {
            sections,
            total_duration_ms,
        }
    }

    /// End time (ms) of the intro section, or `None` if no intro was detected.
    #[must_use]
    pub fn intro_end_ms(&self) -> Option<u32> {
        self.sections
            .iter()
            .find(|s| s.role == SectionRole::Intro)
            .map(|s| s.end_ms)
    }

    /// Start time (ms) of the first chorus, or `None` if no chorus was detected.
    #[must_use]
    pub fn chorus_start_ms(&self) -> Option<u32> {
        self.sections
            .iter()
            .find(|s| s.role == SectionRole::Chorus)
            .map(|s| s.start_ms)
    }

    /// Start time of the outro, or `None` if none detected.
    #[must_use]
    pub fn outro_start_ms(&self) -> Option<u32> {
        self.sections
            .iter()
            .find(|s| s.role == SectionRole::Outro)
            .map(|s| s.start_ms)
    }

    /// Count of chorus sections.
    #[must_use]
    pub fn chorus_count(&self) -> usize {
        self.sections
            .iter()
            .filter(|s| s.role == SectionRole::Chorus)
            .count()
    }

    /// Average confidence across all sections.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_confidence(&self) -> f32 {
        let n = self.sections.len();
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = self.sections.iter().map(|s| s.confidence).sum();
        sum / n as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_list() -> SummarySectionList {
        let mut list = SummarySectionList::new();
        list.add(SummarySection::new(SectionRole::Intro, 0, 8_000, 0.9));
        list.add(SummarySection::new(SectionRole::Verse, 8_000, 40_000, 0.85));
        list.add(SummarySection::new(
            SectionRole::Chorus,
            40_000,
            72_000,
            0.95,
        ));
        list.add(SummarySection::new(
            SectionRole::Verse,
            72_000,
            104_000,
            0.8,
        ));
        list.add(SummarySection::new(
            SectionRole::Chorus,
            104_000,
            136_000,
            0.92,
        ));
        list.add(SummarySection::new(
            SectionRole::Outro,
            136_000,
            180_000,
            0.75,
        ));
        list
    }

    #[test]
    fn test_section_role_labels() {
        assert_eq!(SectionRole::Intro.label(), "Intro");
        assert_eq!(SectionRole::Chorus.label(), "Chorus");
        assert_eq!(SectionRole::Unknown.label(), "Unknown");
    }

    #[test]
    fn test_section_duration() {
        let s = SummarySection::new(SectionRole::Verse, 10_000, 40_000, 0.8);
        assert_eq!(s.duration_ms(), 30_000);
    }

    #[test]
    fn test_represents_full_song_false_for_short() {
        let s = SummarySection::new(SectionRole::Unknown, 0, 60_000, 0.5);
        assert!(!s.represents_full_song());
    }

    #[test]
    fn test_represents_full_song_true_for_long_unknown() {
        let s = SummarySection::new(SectionRole::Unknown, 0, 200_000, 0.5);
        assert!(s.represents_full_song());
    }

    #[test]
    fn test_represents_full_song_false_for_known_role() {
        let s = SummarySection::new(SectionRole::Chorus, 0, 200_000, 0.9);
        assert!(!s.represents_full_song());
    }

    #[test]
    fn test_list_total_duration() {
        let list = make_list();
        assert_eq!(list.total_duration_ms(), 180_000);
    }

    #[test]
    fn test_list_len() {
        let list = make_list();
        assert_eq!(list.len(), 6);
        assert!(!list.is_empty());
    }

    #[test]
    fn test_summary_intro_end() {
        let summary = MusicSummary::new(make_list(), 180_000);
        assert_eq!(summary.intro_end_ms(), Some(8_000));
    }

    #[test]
    fn test_summary_chorus_start() {
        let summary = MusicSummary::new(make_list(), 180_000);
        assert_eq!(summary.chorus_start_ms(), Some(40_000));
    }

    #[test]
    fn test_summary_chorus_count() {
        let summary = MusicSummary::new(make_list(), 180_000);
        assert_eq!(summary.chorus_count(), 2);
    }

    #[test]
    fn test_summary_outro_start() {
        let summary = MusicSummary::new(make_list(), 180_000);
        assert_eq!(summary.outro_start_ms(), Some(136_000));
    }

    #[test]
    fn test_summary_average_confidence() {
        let summary = MusicSummary::new(make_list(), 180_000);
        let avg = summary.average_confidence();
        assert!(avg > 0.0 && avg <= 1.0);
    }

    #[test]
    fn test_summary_no_intro_returns_none() {
        let mut list = SummarySectionList::new();
        list.add(SummarySection::new(SectionRole::Chorus, 0, 30_000, 0.9));
        let summary = MusicSummary::new(list, 30_000);
        assert_eq!(summary.intro_end_ms(), None);
    }
}
