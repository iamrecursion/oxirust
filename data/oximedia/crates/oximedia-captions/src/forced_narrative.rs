//! Forced narrative / foreign language captions.
//!
//! Forced narrative captions are displayed only when foreign-language or
//! non-diegetic text appears on screen — for example, translated signs,
//! alien speech, or untranslated foreign dialogue in an otherwise English
//! film.

#![allow(dead_code)]
#![allow(missing_docs)]

// ── ForcedNarrativeEntry ──────────────────────────────────────────────────────

/// A single forced-narrative caption entry.
#[derive(Debug, Clone)]
pub struct ForcedNarrativeEntry {
    /// First frame on which this entry is displayed (inclusive).
    pub start_frame: u64,
    /// Last frame on which this entry is displayed (inclusive).
    pub end_frame: u64,
    /// Caption text to display.
    pub text: String,
    /// BCP-47 language tag of the caption text (e.g. `"en"`, `"fr"`, `"ja"`).
    pub language: String,
}

impl ForcedNarrativeEntry {
    /// Create a new forced narrative entry.
    pub fn new(
        start_frame: u64,
        end_frame: u64,
        text: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            start_frame,
            end_frame,
            text: text.into(),
            language: language.into(),
        }
    }

    /// Number of frames this entry spans.
    ///
    /// Returns 0 if `start_frame > end_frame` (degenerate entry).
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` when the caption language is not English (`"en"`).
    ///
    /// Forced narrative tracks typically consist of translations into the
    /// primary audience language; entries where the caption language differs
    /// from English are flagged as foreign-only.
    #[must_use]
    pub fn is_foreign_only(&self) -> bool {
        self.language != "en"
    }
}

// ── ForcedNarrativeTrack ──────────────────────────────────────────────────────

/// A collection of forced-narrative entries for a single language track.
#[derive(Debug, Default)]
pub struct ForcedNarrativeTrack {
    /// Caption entries in this track.
    pub entries: Vec<ForcedNarrativeEntry>,
    /// BCP-47 language tag of the source (original) material.
    pub source_language: String,
}

impl ForcedNarrativeTrack {
    /// Create a new, empty forced-narrative track.
    pub fn new(source_language: impl Into<String>) -> Self {
        Self {
            entries: Vec::new(),
            source_language: source_language.into(),
        }
    }

    /// Append an entry to the track.
    pub fn add(&mut self, entry: ForcedNarrativeEntry) {
        self.entries.push(entry);
    }

    /// All entries whose frame range overlaps `[start, end]`.
    ///
    /// An entry overlaps when its `start_frame` is ≤ `end` **and** its
    /// `end_frame` is ≥ `start`.
    #[must_use]
    pub fn entries_in_range(&self, start: u64, end: u64) -> Vec<&ForcedNarrativeEntry> {
        self.entries
            .iter()
            .filter(|e| e.start_frame <= end && e.end_frame >= start)
            .collect()
    }

    /// All entries that are foreign-only (language != `"en"`).
    #[must_use]
    pub fn foreign_entries(&self) -> Vec<&ForcedNarrativeEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_foreign_only())
            .collect()
    }

    /// Sum of [`ForcedNarrativeEntry::duration_frames`] across all entries.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.entries
            .iter()
            .map(ForcedNarrativeEntry::duration_frames)
            .sum()
    }

    /// Total number of entries in this track.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn en_entry(start: u64, end: u64) -> ForcedNarrativeEntry {
        ForcedNarrativeEntry::new(start, end, "English text", "en")
    }

    fn fr_entry(start: u64, end: u64) -> ForcedNarrativeEntry {
        ForcedNarrativeEntry::new(start, end, "Texte français", "fr")
    }

    // ── ForcedNarrativeEntry ──

    #[test]
    fn test_entry_duration_frames_basic() {
        let e = en_entry(100, 200);
        assert_eq!(e.duration_frames(), 100);
    }

    #[test]
    fn test_entry_duration_frames_zero() {
        let e = en_entry(50, 50);
        assert_eq!(e.duration_frames(), 0);
    }

    #[test]
    fn test_entry_is_foreign_only_english() {
        assert!(!en_entry(0, 10).is_foreign_only());
    }

    #[test]
    fn test_entry_is_foreign_only_french() {
        assert!(fr_entry(0, 10).is_foreign_only());
    }

    #[test]
    fn test_entry_is_foreign_only_japanese() {
        let e = ForcedNarrativeEntry::new(0, 24, "日本語テキスト", "ja");
        assert!(e.is_foreign_only());
    }

    // ── ForcedNarrativeTrack ──

    #[test]
    fn test_track_entry_count_empty() {
        let track = ForcedNarrativeTrack::new("en");
        assert_eq!(track.entry_count(), 0);
    }

    #[test]
    fn test_track_add_and_count() {
        let mut track = ForcedNarrativeTrack::new("en");
        track.add(en_entry(0, 24));
        track.add(fr_entry(30, 60));
        assert_eq!(track.entry_count(), 2);
    }

    #[test]
    fn test_track_total_duration_frames() {
        let mut track = ForcedNarrativeTrack::new("en");
        track.add(en_entry(0, 100)); // 100 frames
        track.add(fr_entry(200, 250)); // 50 frames
        assert_eq!(track.total_duration_frames(), 150);
    }

    #[test]
    fn test_track_entries_in_range_overlap() {
        let mut track = ForcedNarrativeTrack::new("en");
        track.add(en_entry(10, 30));
        track.add(en_entry(50, 80));
        // Query [20, 60]: overlaps both (10-30 and 50-80)
        let found = track.entries_in_range(20, 60);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_track_entries_in_range_no_overlap() {
        let mut track = ForcedNarrativeTrack::new("en");
        track.add(en_entry(0, 10));
        // Query [20, 30]: no overlap
        let found = track.entries_in_range(20, 30);
        assert!(found.is_empty());
    }

    #[test]
    fn test_track_entries_in_range_exact_boundary() {
        let mut track = ForcedNarrativeTrack::new("en");
        track.add(en_entry(0, 24)); // end = 24
                                    // Query [24, 48]: start == entry end → should match
        let found = track.entries_in_range(24, 48);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_track_foreign_entries() {
        let mut track = ForcedNarrativeTrack::new("en");
        track.add(en_entry(0, 24));
        track.add(fr_entry(30, 60));
        track.add(ForcedNarrativeEntry::new(70, 100, "Español", "es"));
        let foreign = track.foreign_entries();
        assert_eq!(foreign.len(), 2);
    }

    #[test]
    fn test_track_foreign_entries_none() {
        let mut track = ForcedNarrativeTrack::new("en");
        track.add(en_entry(0, 24));
        track.add(en_entry(30, 60));
        assert!(track.foreign_entries().is_empty());
    }

    #[test]
    fn test_track_total_duration_empty() {
        let track = ForcedNarrativeTrack::new("fr");
        assert_eq!(track.total_duration_frames(), 0);
    }
}
