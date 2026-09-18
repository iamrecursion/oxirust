#![allow(dead_code)]
//! Subtitle comparison and diffing.
//!
//! Provides tools to compare two subtitle tracks, detecting
//! differences in timing, text content, and ordering. Useful
//! for QC workflows, version comparisons, and translation checks.

/// Represents the type of difference found between two subtitle entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffKind {
    /// Only present in the first track (deleted or missing in second).
    OnlyInFirst,
    /// Only present in the second track (added in second).
    OnlyInSecond,
    /// Text content differs between matched entries.
    TextChanged,
    /// Timing differs between matched entries.
    TimingChanged,
    /// Both text and timing differ.
    BothChanged,
    /// Entries are identical.
    Identical,
}

/// A single subtitle entry for comparison purposes.
#[derive(Clone, Debug)]
pub struct DiffEntry {
    /// Index in the original track.
    pub index: usize,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Text content.
    pub text: String,
}

impl DiffEntry {
    /// Create a new diff entry.
    pub fn new(index: usize, start_ms: i64, end_ms: i64, text: impl Into<String>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
        }
    }

    /// Check if timing matches another entry within a tolerance.
    pub fn timing_matches(&self, other: &Self, tolerance_ms: i64) -> bool {
        (self.start_ms - other.start_ms).abs() <= tolerance_ms
            && (self.end_ms - other.end_ms).abs() <= tolerance_ms
    }

    /// Check if text matches another entry (case-sensitive).
    pub fn text_matches(&self, other: &Self) -> bool {
        self.text == other.text
    }

    /// Check if text matches another entry (case-insensitive).
    pub fn text_matches_ignore_case(&self, other: &Self) -> bool {
        self.text.to_lowercase() == other.text.to_lowercase()
    }
}

/// A single diff result between two subtitle entries.
#[derive(Clone, Debug)]
pub struct DiffResult {
    /// Kind of difference.
    pub kind: DiffKind,
    /// Entry from the first track (if present).
    pub first: Option<DiffEntry>,
    /// Entry from the second track (if present).
    pub second: Option<DiffEntry>,
}

impl DiffResult {
    /// Check if this result represents a change.
    pub fn is_changed(&self) -> bool {
        self.kind != DiffKind::Identical
    }
}

/// Configuration for subtitle diffing.
#[derive(Clone, Debug)]
pub struct DiffConfig {
    /// Timing tolerance in milliseconds for matching.
    pub timing_tolerance_ms: i64,
    /// Whether to ignore case when comparing text.
    pub ignore_case: bool,
    /// Whether to trim whitespace before comparing text.
    pub trim_whitespace: bool,
    /// Maximum timing difference (ms) to still consider entries as "matched".
    pub max_match_distance_ms: i64,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            timing_tolerance_ms: 50,
            ignore_case: false,
            trim_whitespace: true,
            max_match_distance_ms: 2000,
        }
    }
}

/// Compare two subtitle tracks and produce a list of differences.
pub fn diff_tracks(
    first: &[DiffEntry],
    second: &[DiffEntry],
    config: &DiffConfig,
) -> Vec<DiffResult> {
    let mut results = Vec::new();
    let mut matched_second: Vec<bool> = vec![false; second.len()];

    for entry_a in first {
        let mut best_match: Option<(usize, i64)> = None;

        for (j, entry_b) in second.iter().enumerate() {
            if matched_second[j] {
                continue;
            }
            let dist = (entry_a.start_ms - entry_b.start_ms).abs();
            if dist <= config.max_match_distance_ms {
                if let Some((_, best_dist)) = best_match {
                    if dist < best_dist {
                        best_match = Some((j, dist));
                    }
                } else {
                    best_match = Some((j, dist));
                }
            }
        }

        if let Some((j, _)) = best_match {
            matched_second[j] = true;
            let entry_b = &second[j];

            let timing_ok = entry_a.timing_matches(entry_b, config.timing_tolerance_ms);
            let text_a = if config.trim_whitespace {
                entry_a.text.trim().to_string()
            } else {
                entry_a.text.clone()
            };
            let text_b = if config.trim_whitespace {
                entry_b.text.trim().to_string()
            } else {
                entry_b.text.clone()
            };
            let text_ok = if config.ignore_case {
                text_a.to_lowercase() == text_b.to_lowercase()
            } else {
                text_a == text_b
            };

            let kind = match (timing_ok, text_ok) {
                (true, true) => DiffKind::Identical,
                (false, true) => DiffKind::TimingChanged,
                (true, false) => DiffKind::TextChanged,
                (false, false) => DiffKind::BothChanged,
            };

            results.push(DiffResult {
                kind,
                first: Some(entry_a.clone()),
                second: Some(entry_b.clone()),
            });
        } else {
            results.push(DiffResult {
                kind: DiffKind::OnlyInFirst,
                first: Some(entry_a.clone()),
                second: None,
            });
        }
    }

    // Entries only in the second track
    for (j, entry_b) in second.iter().enumerate() {
        if !matched_second[j] {
            results.push(DiffResult {
                kind: DiffKind::OnlyInSecond,
                first: None,
                second: Some(entry_b.clone()),
            });
        }
    }

    results
}

/// Count the number of each type of difference.
pub fn count_diff_kinds(results: &[DiffResult]) -> DiffSummary {
    let mut summary = DiffSummary::default();
    for r in results {
        match r.kind {
            DiffKind::Identical => summary.identical += 1,
            DiffKind::TextChanged => summary.text_changed += 1,
            DiffKind::TimingChanged => summary.timing_changed += 1,
            DiffKind::BothChanged => summary.both_changed += 1,
            DiffKind::OnlyInFirst => summary.only_in_first += 1,
            DiffKind::OnlyInSecond => summary.only_in_second += 1,
        }
    }
    summary
}

/// Summary of differences between two subtitle tracks.
#[derive(Clone, Debug, Default)]
pub struct DiffSummary {
    /// Number of identical entries.
    pub identical: usize,
    /// Number of entries with text changes.
    pub text_changed: usize,
    /// Number of entries with timing changes.
    pub timing_changed: usize,
    /// Number of entries with both text and timing changes.
    pub both_changed: usize,
    /// Number of entries only in the first track.
    pub only_in_first: usize,
    /// Number of entries only in the second track.
    pub only_in_second: usize,
}

impl DiffSummary {
    /// Total number of entries compared.
    pub fn total(&self) -> usize {
        self.identical
            + self.text_changed
            + self.timing_changed
            + self.both_changed
            + self.only_in_first
            + self.only_in_second
    }

    /// Total number of changes (non-identical results).
    pub fn total_changes(&self) -> usize {
        self.total() - self.identical
    }

    /// Percentage of entries that are identical.
    pub fn identity_percentage(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 100.0;
        }
        (self.identical as f64 / total as f64) * 100.0
    }
}

/// Compute the Levenshtein edit distance between two strings.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(index: usize, start: i64, end: i64, text: &str) -> DiffEntry {
        DiffEntry::new(index, start, end, text)
    }

    #[test]
    fn test_diff_entry_timing_matches() {
        let a = make_entry(0, 1000, 3000, "Hello");
        let b = make_entry(0, 1020, 3010, "Hello");
        assert!(a.timing_matches(&b, 50));
        assert!(!a.timing_matches(&b, 10));
    }

    #[test]
    fn test_diff_entry_text_matches() {
        let a = make_entry(0, 0, 1000, "Hello");
        let b = make_entry(0, 0, 1000, "Hello");
        assert!(a.text_matches(&b));
    }

    #[test]
    fn test_diff_entry_text_matches_ignore_case() {
        let a = make_entry(0, 0, 1000, "Hello");
        let b = make_entry(0, 0, 1000, "hello");
        assert!(!a.text_matches(&b));
        assert!(a.text_matches_ignore_case(&b));
    }

    #[test]
    fn test_diff_identical_tracks() {
        let track = vec![
            make_entry(0, 0, 1000, "Hello"),
            make_entry(1, 2000, 3000, "World"),
        ];
        let config = DiffConfig::default();
        let results = diff_tracks(&track, &track, &config);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.kind == DiffKind::Identical));
    }

    #[test]
    fn test_diff_text_changed() {
        let first = vec![make_entry(0, 0, 1000, "Hello")];
        let second = vec![make_entry(0, 0, 1000, "Goodbye")];
        let config = DiffConfig::default();
        let results = diff_tracks(&first, &second, &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind, DiffKind::TextChanged);
    }

    #[test]
    fn test_diff_timing_changed() {
        let first = vec![make_entry(0, 0, 1000, "Hello")];
        let second = vec![make_entry(0, 200, 1200, "Hello")];
        let config = DiffConfig::default();
        let results = diff_tracks(&first, &second, &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind, DiffKind::TimingChanged);
    }

    #[test]
    fn test_diff_only_in_first() {
        let first = vec![make_entry(0, 0, 1000, "Hello")];
        let second: Vec<DiffEntry> = vec![];
        let config = DiffConfig::default();
        let results = diff_tracks(&first, &second, &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind, DiffKind::OnlyInFirst);
    }

    #[test]
    fn test_diff_only_in_second() {
        let first: Vec<DiffEntry> = vec![];
        let second = vec![make_entry(0, 0, 1000, "Hello")];
        let config = DiffConfig::default();
        let results = diff_tracks(&first, &second, &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind, DiffKind::OnlyInSecond);
    }

    #[test]
    fn test_diff_result_is_changed() {
        let r = DiffResult {
            kind: DiffKind::Identical,
            first: None,
            second: None,
        };
        assert!(!r.is_changed());

        let r2 = DiffResult {
            kind: DiffKind::TextChanged,
            first: None,
            second: None,
        };
        assert!(r2.is_changed());
    }

    #[test]
    fn test_count_diff_kinds() {
        let results = vec![
            DiffResult {
                kind: DiffKind::Identical,
                first: None,
                second: None,
            },
            DiffResult {
                kind: DiffKind::TextChanged,
                first: None,
                second: None,
            },
            DiffResult {
                kind: DiffKind::OnlyInFirst,
                first: None,
                second: None,
            },
        ];
        let summary = count_diff_kinds(&results);
        assert_eq!(summary.identical, 1);
        assert_eq!(summary.text_changed, 1);
        assert_eq!(summary.only_in_first, 1);
        assert_eq!(summary.total(), 3);
    }

    #[test]
    fn test_diff_summary_identity_percentage() {
        let summary = DiffSummary {
            identical: 8,
            text_changed: 2,
            ..DiffSummary::default()
        };
        assert!((summary.identity_percentage() - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_summary_empty() {
        let summary = DiffSummary::default();
        assert!((summary.identity_percentage() - 100.0).abs() < 1e-10);
        assert_eq!(summary.total(), 0);
        assert_eq!(summary.total_changes(), 0);
    }

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_different() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_edit_distance_empty() {
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", ""), 0);
    }

    #[test]
    fn test_diff_both_changed() {
        let first = vec![make_entry(0, 0, 1000, "Hello")];
        let second = vec![make_entry(0, 200, 1200, "Goodbye")];
        let config = DiffConfig::default();
        let results = diff_tracks(&first, &second, &config);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind, DiffKind::BothChanged);
    }

    #[test]
    fn test_diff_ignore_case_config() {
        let first = vec![make_entry(0, 0, 1000, "Hello")];
        let second = vec![make_entry(0, 0, 1000, "hello")];
        let mut config = DiffConfig::default();
        config.ignore_case = true;
        let results = diff_tracks(&first, &second, &config);
        assert_eq!(results[0].kind, DiffKind::Identical);
    }
}
