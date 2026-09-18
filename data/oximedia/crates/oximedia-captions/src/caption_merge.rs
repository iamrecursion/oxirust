//! Caption track merging with configurable conflict-resolution strategies.
//!
//! Combines two or more caption tracks into a single output track,
//! handling overlapping time ranges according to the selected
//! `MergeStrategy`.

#![allow(dead_code)]

/// How to resolve conflicts when two source tracks overlap in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MergeStrategy {
    /// Keep captions from the first (primary) track; discard overlapping
    /// entries from the second track.
    PrimaryWins,
    /// Keep captions from the second track when there is a conflict.
    SecondaryWins,
    /// Place captions from both tracks side-by-side using additional rows.
    /// Overlapping captions both appear simultaneously.
    Interleave,
    /// Concatenate text from both overlapping captions into a single entry
    /// separated by a pipe character (`|`).
    ConcatenateText,
}

impl MergeStrategy {
    /// Human-readable name of the strategy.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PrimaryWins => "primary-wins",
            Self::SecondaryWins => "secondary-wins",
            Self::Interleave => "interleave",
            Self::ConcatenateText => "concatenate-text",
        }
    }
}

/// A lightweight caption entry used by the merger.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeCaption {
    /// Display start in milliseconds.
    pub start_ms: u64,
    /// Display end in milliseconds.
    pub end_ms: u64,
    /// Caption text.
    pub text: String,
    /// Source track identifier (`0` = primary, `1` = secondary, etc.).
    pub source: u8,
}

impl MergeCaption {
    /// Create a new caption entry.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, text: impl Into<String>, source: u8) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
            source,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` when this caption overlaps `other` in time.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }
}

/// Statistics from a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeReport {
    /// Total captions in the primary track.
    pub primary_count: usize,
    /// Total captions in the secondary track.
    pub secondary_count: usize,
    /// Number of time-range conflicts detected.
    pub conflict_count: usize,
    /// Number of captions dropped due to conflict resolution.
    pub dropped_count: usize,
    /// Total captions in the merged output.
    pub output_count: usize,
    /// Strategy used for this merge.
    pub strategy: Option<MergeStrategy>,
}

impl MergeReport {
    /// Returns `true` when no conflicts occurred.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.conflict_count == 0
    }
}

/// Merges two caption tracks according to a configurable strategy.
#[derive(Debug, Clone)]
pub struct CaptionMerger {
    strategy: MergeStrategy,
    /// Gap in milliseconds to insert between adjacent merged captions.
    gap_ms: u64,
}

impl CaptionMerger {
    /// Create a merger with the given strategy.
    #[must_use]
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            strategy,
            gap_ms: 0,
        }
    }

    /// Set the minimum gap between merged captions.
    #[must_use]
    pub fn with_gap(mut self, gap_ms: u64) -> Self {
        self.gap_ms = gap_ms;
        self
    }

    /// Current strategy.
    #[must_use]
    pub fn strategy(&self) -> MergeStrategy {
        self.strategy
    }

    /// Merge `primary` and `secondary` into a single sorted track.
    ///
    /// Returns the merged captions and a `MergeReport`.
    #[must_use]
    pub fn merge(
        &self,
        primary: &[MergeCaption],
        secondary: &[MergeCaption],
    ) -> (Vec<MergeCaption>, MergeReport) {
        let mut report = MergeReport {
            primary_count: primary.len(),
            secondary_count: secondary.len(),
            strategy: Some(self.strategy),
            ..Default::default()
        };

        let mut output: Vec<MergeCaption> = Vec::new();

        match self.strategy {
            MergeStrategy::PrimaryWins => {
                // Add all primary captions.
                output.extend_from_slice(primary);
                // Add secondary captions that do not overlap any primary.
                for sec in secondary {
                    if primary.iter().any(|p| p.overlaps(sec)) {
                        report.conflict_count += 1;
                        report.dropped_count += 1;
                    } else {
                        output.push(sec.clone());
                    }
                }
            }
            MergeStrategy::SecondaryWins => {
                // Add all secondary captions.
                output.extend_from_slice(secondary);
                // Add primary captions that do not overlap any secondary.
                for prim in primary {
                    if secondary.iter().any(|s| s.overlaps(prim)) {
                        report.conflict_count += 1;
                        report.dropped_count += 1;
                    } else {
                        output.push(prim.clone());
                    }
                }
            }
            MergeStrategy::Interleave => {
                // Count conflicts but keep all entries.
                for sec in secondary {
                    if primary.iter().any(|p| p.overlaps(sec)) {
                        report.conflict_count += 1;
                    }
                }
                output.extend_from_slice(primary);
                output.extend_from_slice(secondary);
            }
            MergeStrategy::ConcatenateText => {
                // For each primary caption, check for overlapping secondary.
                let mut used_secondary = vec![false; secondary.len()];
                for prim in primary {
                    let mut merged = prim.clone();
                    for (j, sec) in secondary.iter().enumerate() {
                        if prim.overlaps(sec) {
                            report.conflict_count += 1;
                            merged.text = format!("{} | {}", merged.text, sec.text);
                            used_secondary[j] = true;
                        }
                    }
                    output.push(merged);
                }
                // Add secondary captions that were not merged.
                for (used, sec) in used_secondary.iter().zip(secondary.iter()) {
                    if !used {
                        output.push(sec.clone());
                    }
                }
            }
        }

        // Apply gap enforcement and sort by start time.
        output.sort_by_key(|c| c.start_ms);
        if self.gap_ms > 0 {
            for i in 1..output.len() {
                let prev_end = output[i - 1].end_ms;
                if output[i].start_ms < prev_end + self.gap_ms {
                    output[i].start_ms = prev_end + self.gap_ms;
                }
            }
        }

        report.output_count = output.len();
        (output, report)
    }

    /// Merge more than two tracks sequentially using the primary strategy.
    ///
    /// The first track is primary; subsequent tracks are merged as secondary
    /// in order.
    #[must_use]
    pub fn merge_many(
        &self,
        tracks: &[Vec<MergeCaption>],
    ) -> (Vec<MergeCaption>, Vec<MergeReport>) {
        if tracks.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut current = tracks[0].clone();
        let mut reports = Vec::new();
        for track in &tracks[1..] {
            let (merged, report) = self.merge(&current, track);
            current = merged;
            reports.push(report);
        }
        (current, reports)
    }
}

impl Default for CaptionMerger {
    fn default() -> Self {
        Self::new(MergeStrategy::PrimaryWins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(start: u64, end: u64, text: &str, src: u8) -> MergeCaption {
        MergeCaption::new(start, end, text, src)
    }

    #[test]
    fn test_overlaps_true() {
        let a = cap(0, 1000, "a", 0);
        let b = cap(500, 1500, "b", 1);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_false_adjacent() {
        let a = cap(0, 1000, "a", 0);
        let b = cap(1000, 2000, "b", 1);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_overlaps_false_gap() {
        let a = cap(0, 1000, "a", 0);
        let b = cap(1500, 2500, "b", 1);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_primary_wins_no_conflict() {
        let merger = CaptionMerger::new(MergeStrategy::PrimaryWins);
        let primary = vec![cap(0, 1000, "P1", 0)];
        let secondary = vec![cap(1100, 2100, "S1", 1)];
        let (out, report) = merger.merge(&primary, &secondary);
        assert_eq!(out.len(), 2);
        assert_eq!(report.conflict_count, 0);
        assert!(report.is_clean());
    }

    #[test]
    fn test_primary_wins_drops_secondary_on_conflict() {
        let merger = CaptionMerger::new(MergeStrategy::PrimaryWins);
        let primary = vec![cap(0, 2000, "P1", 0)];
        let secondary = vec![cap(500, 1500, "S1", 1)]; // overlaps
        let (out, report) = merger.merge(&primary, &secondary);
        assert_eq!(out.len(), 1);
        assert_eq!(report.dropped_count, 1);
        assert_eq!(out[0].text, "P1");
    }

    #[test]
    fn test_secondary_wins_drops_primary_on_conflict() {
        let merger = CaptionMerger::new(MergeStrategy::SecondaryWins);
        let primary = vec![cap(0, 2000, "P1", 0)];
        let secondary = vec![cap(500, 1500, "S1", 1)];
        let (out, report) = merger.merge(&primary, &secondary);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "S1");
        assert_eq!(report.dropped_count, 1);
    }

    #[test]
    fn test_interleave_keeps_all() {
        let merger = CaptionMerger::new(MergeStrategy::Interleave);
        let primary = vec![cap(0, 2000, "P1", 0)];
        let secondary = vec![cap(500, 1500, "S1", 1)];
        let (out, report) = merger.merge(&primary, &secondary);
        assert_eq!(out.len(), 2);
        assert_eq!(report.conflict_count, 1);
        assert_eq!(report.dropped_count, 0);
    }

    #[test]
    fn test_concatenate_text_merges_overlapping() {
        let merger = CaptionMerger::new(MergeStrategy::ConcatenateText);
        let primary = vec![cap(0, 2000, "Hello", 0)];
        let secondary = vec![cap(500, 1500, "World", 1)];
        let (out, _report) = merger.merge(&primary, &secondary);
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("Hello"));
        assert!(out[0].text.contains("World"));
    }

    #[test]
    fn test_output_sorted_by_start() {
        let merger = CaptionMerger::new(MergeStrategy::Interleave);
        let primary = vec![cap(1000, 2000, "P1", 0)];
        let secondary = vec![cap(0, 500, "S1", 1)];
        let (out, _) = merger.merge(&primary, &secondary);
        assert!(out[0].start_ms <= out[1].start_ms);
    }

    #[test]
    fn test_gap_enforcement() {
        let merger = CaptionMerger::new(MergeStrategy::PrimaryWins).with_gap(200);
        let primary = vec![cap(0, 1000, "P1", 0)];
        let secondary = vec![cap(1050, 2000, "S1", 1)]; // gap of 50ms < 200ms
        let (out, _) = merger.merge(&primary, &secondary);
        // secondary start should be pushed to 1000 + 200 = 1200
        assert!(out[1].start_ms >= 1200);
    }

    #[test]
    fn test_merge_report_fields() {
        let merger = CaptionMerger::default();
        let primary = vec![cap(0, 1000, "P", 0), cap(2000, 3000, "P2", 0)];
        let secondary = vec![cap(500, 1500, "S", 1)];
        let (_out, report) = merger.merge(&primary, &secondary);
        assert_eq!(report.primary_count, 2);
        assert_eq!(report.secondary_count, 1);
        assert_eq!(report.strategy, Some(MergeStrategy::PrimaryWins));
    }

    #[test]
    fn test_merge_many_empty() {
        let merger = CaptionMerger::default();
        let (out, reports) = merger.merge_many(&[]);
        assert!(out.is_empty());
        assert!(reports.is_empty());
    }

    #[test]
    fn test_merge_many_single_track() {
        let merger = CaptionMerger::default();
        let tracks = vec![vec![cap(0, 1000, "only", 0)]];
        let (out, reports) = merger.merge_many(&tracks);
        assert_eq!(out.len(), 1);
        assert!(reports.is_empty());
    }

    #[test]
    fn test_merge_strategy_as_str() {
        assert_eq!(MergeStrategy::Interleave.as_str(), "interleave");
        assert_eq!(MergeStrategy::ConcatenateText.as_str(), "concatenate-text");
    }

    #[test]
    fn test_duration_ms() {
        let c = cap(500, 1500, "x", 0);
        assert_eq!(c.duration_ms(), 1000);
    }
}
