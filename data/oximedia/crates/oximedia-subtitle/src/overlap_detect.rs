//! Subtitle overlap detection for OxiMedia.
//!
//! Identifies subtitle cues that overlap in time, classifies the overlap
//! as full or partial, and summarises the findings in a report.

#![allow(dead_code)]

/// Describes how two subtitle cues overlap in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapType {
    /// One cue is entirely contained within the other's time span.
    Full,
    /// The cues share a partial time window (start/end boundaries cross).
    Partial,
}

impl OverlapType {
    /// Severity level: Full overlaps are more disruptive than partial ones.
    #[must_use]
    pub fn severity(self) -> u8 {
        match self {
            Self::Full => 2,
            Self::Partial => 1,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Partial => "partial",
        }
    }
}

/// A single detected overlap between two subtitle cues.
#[derive(Debug, Clone)]
pub struct SubtitleOverlap {
    /// Index of the first (earlier-starting) cue.
    pub cue_a: usize,
    /// Index of the second cue.
    pub cue_b: usize,
    /// Start of the overlap window in milliseconds.
    pub overlap_start_ms: i64,
    /// End of the overlap window in milliseconds.
    pub overlap_end_ms: i64,
    /// Classification of the overlap.
    pub overlap_type: OverlapType,
}

impl SubtitleOverlap {
    /// Returns the duration of the overlap in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.overlap_end_ms - self.overlap_start_ms).max(0)
    }

    /// Returns `true` when the overlap is a full containment.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.overlap_type == OverlapType::Full
    }

    /// Severity (delegated to `OverlapType`).
    #[must_use]
    pub fn severity(&self) -> u8 {
        self.overlap_type.severity()
    }
}

/// A subtitle cue used as input for overlap detection.
#[derive(Debug, Clone)]
pub struct DetectableCue {
    /// Cue index.
    pub index: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Text content.
    pub text: String,
}

impl DetectableCue {
    /// Creates a new `DetectableCue`.
    #[must_use]
    pub fn new(index: usize, start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Returns `true` when this cue overlaps with `other`.
    #[must_use]
    pub fn overlaps_with(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Returns the overlap window with `other`, if any.
    #[must_use]
    pub fn overlap_window(&self, other: &Self) -> Option<(i64, i64)> {
        let start = self.start_ms.max(other.start_ms);
        let end = self.end_ms.min(other.end_ms);
        if start < end {
            Some((start, end))
        } else {
            None
        }
    }

    /// Determines the overlap type with `other`.
    #[must_use]
    pub fn overlap_type_with(&self, other: &Self) -> OverlapType {
        // Full overlap: one cue is entirely contained in the other
        if (self.start_ms >= other.start_ms && self.end_ms <= other.end_ms)
            || (other.start_ms >= self.start_ms && other.end_ms <= self.end_ms)
        {
            OverlapType::Full
        } else {
            OverlapType::Partial
        }
    }
}

/// Detects overlapping subtitle cues in a track.
#[derive(Debug, Default)]
pub struct OverlapDetector {
    /// If `true`, adjacent cues that share a boundary (start == end) are not flagged.
    pub allow_touching: bool,
}

impl OverlapDetector {
    /// Creates a new `OverlapDetector`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allows cues whose boundaries touch (end_ms == start_ms) without flagging them.
    #[must_use]
    pub fn with_allow_touching(mut self, allow: bool) -> Self {
        self.allow_touching = allow;
        self
    }

    /// Scans `cues` for overlapping pairs and returns all detected overlaps.
    ///
    /// The input does not need to be sorted; all pairs are checked.
    #[must_use]
    pub fn find_overlaps(&self, cues: &[DetectableCue]) -> Vec<SubtitleOverlap> {
        let mut overlaps = Vec::new();
        for i in 0..cues.len() {
            for j in (i + 1)..cues.len() {
                let a = &cues[i];
                let b = &cues[j];

                // Determine if they actually overlap
                let does_overlap = if self.allow_touching {
                    // Touching boundaries count as overlap
                    a.start_ms <= b.end_ms && b.start_ms <= a.end_ms
                } else {
                    // Strictly overlapping (touching is not overlap)
                    a.start_ms < b.end_ms && b.start_ms < a.end_ms
                };

                if does_overlap {
                    if let Some((start, end)) = a.overlap_window(b) {
                        let overlap_type = a.overlap_type_with(b);
                        overlaps.push(SubtitleOverlap {
                            cue_a: a.index,
                            cue_b: b.index,
                            overlap_start_ms: start,
                            overlap_end_ms: end,
                            overlap_type,
                        });
                    }
                }
            }
        }
        overlaps
    }
}

/// Summary report for subtitle overlap detection.
#[derive(Debug, Clone)]
pub struct OverlapReport {
    /// Total number of overlapping pairs detected.
    pub total_overlaps: usize,
    /// Number of full-containment overlaps.
    pub full_overlap_count: usize,
    /// Number of partial overlaps.
    pub partial_overlap_count: usize,
    /// Maximum overlap duration in milliseconds.
    pub max_overlap_ms: i64,
    /// All detected overlaps, in detection order.
    pub overlaps: Vec<SubtitleOverlap>,
}

impl OverlapReport {
    /// Builds a report from a list of detected overlaps.
    #[must_use]
    pub fn from_overlaps(overlaps: Vec<SubtitleOverlap>) -> Self {
        let total_overlaps = overlaps.len();
        let full_overlap_count = overlaps.iter().filter(|o| o.is_full()).count();
        let partial_overlap_count = total_overlaps - full_overlap_count;
        let max_overlap_ms = overlaps.iter().map(|o| o.duration_ms()).max().unwrap_or(0);
        Self {
            total_overlaps,
            full_overlap_count,
            partial_overlap_count,
            max_overlap_ms,
            overlaps,
        }
    }

    /// Returns `true` when at least one full-containment overlap was detected.
    #[must_use]
    pub fn has_full_overlaps(&self) -> bool {
        self.full_overlap_count > 0
    }

    /// Returns `true` when no overlaps were detected.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.total_overlaps == 0
    }

    /// Returns overlaps involving a specific cue index.
    #[must_use]
    pub fn overlaps_for_cue(&self, cue_index: usize) -> Vec<&SubtitleOverlap> {
        self.overlaps
            .iter()
            .filter(|o| o.cue_a == cue_index || o.cue_b == cue_index)
            .collect()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cue(index: usize, start_ms: i64, end_ms: i64) -> DetectableCue {
        DetectableCue::new(index, start_ms, end_ms, format!("Cue {index}"))
    }

    #[test]
    fn test_overlap_type_severity() {
        assert_eq!(OverlapType::Full.severity(), 2);
        assert_eq!(OverlapType::Partial.severity(), 1);
    }

    #[test]
    fn test_overlap_type_label() {
        assert_eq!(OverlapType::Full.label(), "full");
        assert_eq!(OverlapType::Partial.label(), "partial");
    }

    #[test]
    fn test_subtitle_overlap_duration_ms() {
        let o = SubtitleOverlap {
            cue_a: 0,
            cue_b: 1,
            overlap_start_ms: 1000,
            overlap_end_ms: 1500,
            overlap_type: OverlapType::Partial,
        };
        assert_eq!(o.duration_ms(), 500);
    }

    #[test]
    fn test_subtitle_overlap_is_full() {
        let o = SubtitleOverlap {
            cue_a: 0,
            cue_b: 1,
            overlap_start_ms: 0,
            overlap_end_ms: 1000,
            overlap_type: OverlapType::Full,
        };
        assert!(o.is_full());
    }

    #[test]
    fn test_detectable_cue_overlaps_with() {
        let a = make_cue(0, 0, 2000);
        let b = make_cue(1, 1000, 3000);
        assert!(a.overlaps_with(&b));
    }

    #[test]
    fn test_detectable_cue_no_overlap() {
        let a = make_cue(0, 0, 1000);
        let b = make_cue(1, 1000, 2000); // touching, not overlapping
        assert!(!a.overlaps_with(&b));
    }

    #[test]
    fn test_detectable_cue_overlap_window() {
        let a = make_cue(0, 0, 2000);
        let b = make_cue(1, 1000, 3000);
        assert_eq!(a.overlap_window(&b), Some((1000, 2000)));
    }

    #[test]
    fn test_detectable_cue_overlap_type_full() {
        let outer = make_cue(0, 0, 5000);
        let inner = make_cue(1, 1000, 3000);
        assert_eq!(outer.overlap_type_with(&inner), OverlapType::Full);
    }

    #[test]
    fn test_detectable_cue_overlap_type_partial() {
        let a = make_cue(0, 0, 2000);
        let b = make_cue(1, 1000, 3000);
        assert_eq!(a.overlap_type_with(&b), OverlapType::Partial);
    }

    #[test]
    fn test_detector_no_overlaps_sequential() {
        let cues = vec![
            make_cue(0, 0, 1000),
            make_cue(1, 1000, 2000),
            make_cue(2, 2000, 3000),
        ];
        let overlaps = OverlapDetector::new().find_overlaps(&cues);
        assert!(overlaps.is_empty());
    }

    #[test]
    fn test_detector_finds_partial_overlap() {
        let cues = vec![make_cue(0, 0, 2000), make_cue(1, 1500, 3000)];
        let overlaps = OverlapDetector::new().find_overlaps(&cues);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].overlap_type, OverlapType::Partial);
    }

    #[test]
    fn test_detector_finds_full_overlap() {
        let cues = vec![make_cue(0, 0, 5000), make_cue(1, 1000, 3000)];
        let overlaps = OverlapDetector::new().find_overlaps(&cues);
        assert_eq!(overlaps.len(), 1);
        assert!(overlaps[0].is_full());
    }

    #[test]
    fn test_report_has_full_overlaps() {
        let o = SubtitleOverlap {
            cue_a: 0,
            cue_b: 1,
            overlap_start_ms: 0,
            overlap_end_ms: 1000,
            overlap_type: OverlapType::Full,
        };
        let report = OverlapReport::from_overlaps(vec![o]);
        assert!(report.has_full_overlaps());
        assert!(!report.is_clean());
    }

    #[test]
    fn test_report_is_clean_when_no_overlaps() {
        let report = OverlapReport::from_overlaps(vec![]);
        assert!(report.is_clean());
        assert!(!report.has_full_overlaps());
    }

    #[test]
    fn test_report_overlaps_for_cue() {
        let cues = vec![
            make_cue(0, 0, 2000),
            make_cue(1, 1500, 3000),
            make_cue(2, 5000, 7000),
        ];
        let overlaps = OverlapDetector::new().find_overlaps(&cues);
        let report = OverlapReport::from_overlaps(overlaps);
        let for_cue_0 = report.overlaps_for_cue(0);
        assert_eq!(for_cue_0.len(), 1);
        let for_cue_2 = report.overlaps_for_cue(2);
        assert!(for_cue_2.is_empty());
    }
}
