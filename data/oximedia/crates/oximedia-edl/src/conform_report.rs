//! EDL conform report types and operations.
//!
//! A conform report describes the status of matching EDL clips to
//! physical media sources, used during the online conforming process.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Status of a clip conform operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformStatus {
    /// Clip was fully matched to a source file.
    Matched,
    /// Clip was partially matched (e.g., wrong reel or timecode offset).
    PartialMatch,
    /// No matching source was found.
    NotFound,
    /// Multiple conflicting sources were found.
    Conflict,
}

impl ConformStatus {
    /// Returns true if the clip is considered fully resolved (no further action needed).
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Matched)
    }

    /// Returns a human-readable description of the status.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Matched => "Fully matched",
            Self::PartialMatch => "Partial match",
            Self::NotFound => "Not found",
            Self::Conflict => "Conflict",
        }
    }
}

impl std::fmt::Display for ConformStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// A single entry in a conform report.
#[derive(Debug, Clone)]
pub struct ConformEntry {
    /// Name of the clip as referenced in the EDL.
    pub clip_name: String,
    /// Reel/tape identifier.
    pub reel_id: String,
    /// Current conform status.
    pub status: ConformStatus,
    /// Path to the matched source file (None if not found).
    pub source_path: Option<String>,
    /// Confidence score for the match (0.0 to 1.0).
    pub confidence: f32,
}

impl ConformEntry {
    /// Create a new conform entry.
    #[must_use]
    pub fn new(
        clip_name: impl Into<String>,
        reel_id: impl Into<String>,
        status: ConformStatus,
        source_path: Option<String>,
        confidence: f32,
    ) -> Self {
        Self {
            clip_name: clip_name.into(),
            reel_id: reel_id.into(),
            status,
            source_path,
            confidence,
        }
    }

    /// Returns true if the clip has an online (available) source.
    #[must_use]
    pub fn is_online(&self) -> bool {
        self.source_path.is_some() && self.status == ConformStatus::Matched
    }

    /// Returns the source path if the clip is online.
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.source_path.as_deref()
    }

    /// Returns the confidence level as a percentage string.
    #[must_use]
    pub fn confidence_percent(&self) -> String {
        format!("{:.1}%", self.confidence * 100.0)
    }
}

/// A complete EDL conform report for a sequence.
#[derive(Debug)]
pub struct ConformReport {
    /// All conform entries.
    pub entries: Vec<ConformEntry>,
    /// Total number of clips in the original EDL.
    pub total_clips: u32,
}

impl ConformReport {
    /// Create a new empty conform report.
    #[must_use]
    pub fn new(total_clips: u32) -> Self {
        Self {
            entries: Vec::new(),
            total_clips,
        }
    }

    /// Add an entry to the report.
    pub fn add_entry(&mut self, entry: ConformEntry) {
        self.entries.push(entry);
    }

    /// Returns the number of online (fully matched) clips.
    #[must_use]
    pub fn online_count(&self) -> u32 {
        self.entries.iter().filter(|e| e.is_online()).count() as u32
    }

    /// Returns the number of offline (not fully matched) clips.
    #[must_use]
    pub fn offline_count(&self) -> u32 {
        self.entries.iter().filter(|e| !e.is_online()).count() as u32
    }

    /// Returns the conform rate as a fraction (0.0 to 1.0).
    ///
    /// Calculated as `online_count / total_clips`.
    #[must_use]
    pub fn conform_rate(&self) -> f32 {
        if self.total_clips == 0 {
            return 0.0;
        }
        self.online_count() as f32 / self.total_clips as f32
    }

    /// Returns all offline (not fully matched) entries.
    #[must_use]
    pub fn offline_entries(&self) -> Vec<&ConformEntry> {
        self.entries.iter().filter(|e| !e.is_online()).collect()
    }

    /// Returns all entries with a specific status.
    #[must_use]
    pub fn entries_with_status(&self, status: ConformStatus) -> Vec<&ConformEntry> {
        self.entries.iter().filter(|e| e.status == status).collect()
    }

    /// Returns true if all clips are fully conformed.
    #[must_use]
    pub fn is_fully_conformed(&self) -> bool {
        self.online_count() == self.total_clips && self.total_clips > 0
    }

    /// Returns the average confidence across all entries.
    #[must_use]
    pub fn average_confidence(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.entries.iter().map(|e| e.confidence).sum();
        sum / self.entries.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matched_entry(name: &str, reel: &str) -> ConformEntry {
        ConformEntry::new(
            name,
            reel,
            ConformStatus::Matched,
            Some(format!("/media/{}.mov", name)),
            1.0,
        )
    }

    fn make_offline_entry(name: &str, reel: &str) -> ConformEntry {
        ConformEntry::new(name, reel, ConformStatus::NotFound, None, 0.0)
    }

    #[test]
    fn test_conform_status_is_resolved() {
        assert!(ConformStatus::Matched.is_resolved());
        assert!(!ConformStatus::PartialMatch.is_resolved());
        assert!(!ConformStatus::NotFound.is_resolved());
        assert!(!ConformStatus::Conflict.is_resolved());
    }

    #[test]
    fn test_conform_status_display() {
        assert_eq!(ConformStatus::Matched.to_string(), "Fully matched");
        assert_eq!(ConformStatus::NotFound.to_string(), "Not found");
        assert_eq!(ConformStatus::Conflict.to_string(), "Conflict");
    }

    #[test]
    fn test_conform_entry_is_online_matched() {
        let entry = make_matched_entry("shot001", "A001");
        assert!(entry.is_online());
    }

    #[test]
    fn test_conform_entry_is_online_not_found() {
        let entry = make_offline_entry("shot002", "A002");
        assert!(!entry.is_online());
    }

    #[test]
    fn test_conform_entry_partial_match_not_online() {
        let entry = ConformEntry::new(
            "shot003",
            "A003",
            ConformStatus::PartialMatch,
            Some("/media/shot003_alt.mov".to_string()),
            0.7,
        );
        // PartialMatch means source_path is set, but is_online checks for Matched status
        assert!(!entry.is_online());
    }

    #[test]
    fn test_conform_entry_confidence_percent() {
        let entry = ConformEntry::new("s1", "R1", ConformStatus::Matched, None, 0.95);
        assert_eq!(entry.confidence_percent(), "95.0%");
    }

    #[test]
    fn test_conform_entry_source() {
        let entry = make_matched_entry("shot001", "A001");
        assert_eq!(entry.source(), Some("/media/shot001.mov"));
    }

    #[test]
    fn test_conform_report_online_count() {
        let mut report = ConformReport::new(3);
        report.add_entry(make_matched_entry("s1", "R1"));
        report.add_entry(make_matched_entry("s2", "R2"));
        report.add_entry(make_offline_entry("s3", "R3"));

        assert_eq!(report.online_count(), 2);
        assert_eq!(report.offline_count(), 1);
    }

    #[test]
    fn test_conform_report_conform_rate() {
        let mut report = ConformReport::new(4);
        report.add_entry(make_matched_entry("s1", "R1"));
        report.add_entry(make_matched_entry("s2", "R2"));
        report.add_entry(make_offline_entry("s3", "R3"));
        report.add_entry(make_offline_entry("s4", "R4"));

        let rate = report.conform_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_conform_report_conform_rate_zero_total() {
        let report = ConformReport::new(0);
        assert_eq!(report.conform_rate(), 0.0);
    }

    #[test]
    fn test_conform_report_offline_entries() {
        let mut report = ConformReport::new(3);
        report.add_entry(make_matched_entry("s1", "R1"));
        report.add_entry(make_offline_entry("s2", "R2"));
        report.add_entry(make_offline_entry("s3", "R3"));

        let offline = report.offline_entries();
        assert_eq!(offline.len(), 2);
    }

    #[test]
    fn test_conform_report_entries_with_status() {
        let mut report = ConformReport::new(4);
        report.add_entry(make_matched_entry("s1", "R1"));
        report.add_entry(ConformEntry::new(
            "s2",
            "R2",
            ConformStatus::Conflict,
            None,
            0.5,
        ));
        report.add_entry(ConformEntry::new(
            "s3",
            "R3",
            ConformStatus::PartialMatch,
            Some("/media/s3.mov".to_string()),
            0.6,
        ));
        report.add_entry(make_offline_entry("s4", "R4"));

        let conflicts = report.entries_with_status(ConformStatus::Conflict);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].clip_name, "s2");
    }

    #[test]
    fn test_conform_report_is_fully_conformed() {
        let mut report = ConformReport::new(2);
        report.add_entry(make_matched_entry("s1", "R1"));
        report.add_entry(make_matched_entry("s2", "R2"));

        assert!(report.is_fully_conformed());
    }

    #[test]
    fn test_conform_report_not_fully_conformed() {
        let mut report = ConformReport::new(2);
        report.add_entry(make_matched_entry("s1", "R1"));
        report.add_entry(make_offline_entry("s2", "R2"));

        assert!(!report.is_fully_conformed());
    }

    #[test]
    fn test_conform_report_average_confidence() {
        let mut report = ConformReport::new(2);
        report.add_entry(ConformEntry::new(
            "s1",
            "R1",
            ConformStatus::Matched,
            Some("/m/s1.mov".to_string()),
            1.0,
        ));
        report.add_entry(ConformEntry::new(
            "s2",
            "R2",
            ConformStatus::PartialMatch,
            None,
            0.6,
        ));

        let avg = report.average_confidence();
        assert!((avg - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_conform_report_empty_average_confidence() {
        let report = ConformReport::new(0);
        assert_eq!(report.average_confidence(), 0.0);
    }
}
