#![allow(dead_code)]

//! Caption diff engine for comparing caption tracks and detecting changes.
//!
//! Provides tools for computing structural and textual differences between
//! caption tracks, useful for QC workflows, version comparison, and
//! translation verification.

use std::collections::HashMap;
use std::fmt;

/// The type of change detected between two caption entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DiffKind {
    /// Caption was added (present only in the right track).
    Added,
    /// Caption was removed (present only in the left track).
    Removed,
    /// Caption text was modified.
    TextChanged,
    /// Caption timing was modified but text stayed the same.
    TimingChanged,
    /// Both text and timing were modified.
    BothChanged,
    /// Caption is identical in both tracks.
    Unchanged,
}

impl fmt::Display for DiffKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Added => write!(f, "ADDED"),
            Self::Removed => write!(f, "REMOVED"),
            Self::TextChanged => write!(f, "TEXT_CHANGED"),
            Self::TimingChanged => write!(f, "TIMING_CHANGED"),
            Self::BothChanged => write!(f, "BOTH_CHANGED"),
            Self::Unchanged => write!(f, "UNCHANGED"),
        }
    }
}

/// A single caption entry used in diff comparison.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiffCaption {
    /// Index of this caption in its track.
    pub index: usize,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// The text content of the caption.
    pub text: String,
}

impl DiffCaption {
    /// Create a new diff caption entry.
    #[must_use]
    pub fn new(index: usize, start_ms: u64, end_ms: u64, text: String) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text,
        }
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A single diff entry representing one change between two caption tracks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffEntry {
    /// The kind of change.
    pub kind: DiffKind,
    /// The caption from the left (original) track, if any.
    pub left: Option<DiffCaption>,
    /// The caption from the right (modified) track, if any.
    pub right: Option<DiffCaption>,
}

impl DiffEntry {
    /// Create a new diff entry.
    #[must_use]
    pub fn new(kind: DiffKind, left: Option<DiffCaption>, right: Option<DiffCaption>) -> Self {
        Self { kind, left, right }
    }

    /// Returns true if this entry represents an actual change.
    #[must_use]
    pub fn is_change(&self) -> bool {
        self.kind != DiffKind::Unchanged
    }
}

/// Summary statistics for a diff result.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DiffSummary {
    /// Number of captions added.
    pub added: usize,
    /// Number of captions removed.
    pub removed: usize,
    /// Number of captions with text changes.
    pub text_changed: usize,
    /// Number of captions with timing changes.
    pub timing_changed: usize,
    /// Number of captions with both changes.
    pub both_changed: usize,
    /// Number of unchanged captions.
    pub unchanged: usize,
    /// Total captions in the left track.
    pub left_total: usize,
    /// Total captions in the right track.
    pub right_total: usize,
}

impl DiffSummary {
    /// Total number of changes.
    #[must_use]
    pub fn total_changes(&self) -> usize {
        self.added + self.removed + self.text_changed + self.timing_changed + self.both_changed
    }

    /// Percentage of captions that are unchanged (based on the left track).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity_pct(&self) -> f64 {
        if self.left_total == 0 && self.right_total == 0 {
            return 100.0;
        }
        let total = self.left_total.max(self.right_total) as f64;
        if total == 0.0 {
            return 100.0;
        }
        (self.unchanged as f64 / total) * 100.0
    }
}

/// Configuration for the diff engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffConfig {
    /// Maximum timing tolerance in milliseconds for considering captions as matched.
    pub timing_tolerance_ms: u64,
    /// Whether to ignore whitespace differences in text comparison.
    pub ignore_whitespace: bool,
    /// Whether to ignore case differences in text comparison.
    pub ignore_case: bool,
    /// Whether to ignore punctuation differences.
    pub ignore_punctuation: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            timing_tolerance_ms: 100,
            ignore_whitespace: false,
            ignore_case: false,
            ignore_punctuation: false,
        }
    }
}

/// The result of diffing two caption tracks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffResult {
    /// All diff entries.
    pub entries: Vec<DiffEntry>,
    /// Summary statistics.
    pub summary: DiffSummary,
}

impl DiffResult {
    /// Returns only the changed entries.
    #[must_use]
    pub fn changes_only(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.is_change()).collect()
    }
}

/// Normalizes text according to diff config settings.
fn normalize_text(text: &str, config: &DiffConfig) -> String {
    let mut result = text.to_string();
    if config.ignore_whitespace {
        result = result.split_whitespace().collect::<Vec<_>>().join(" ");
    }
    if config.ignore_case {
        result = result.to_lowercase();
    }
    if config.ignore_punctuation {
        result.retain(|c| c.is_alphanumeric() || c.is_whitespace());
    }
    result
}

/// Check if two captions match by timing within tolerance.
fn timings_match(left: &DiffCaption, right: &DiffCaption, tolerance_ms: u64) -> bool {
    let start_diff = if left.start_ms > right.start_ms {
        left.start_ms - right.start_ms
    } else {
        right.start_ms - left.start_ms
    };
    let end_diff = if left.end_ms > right.end_ms {
        left.end_ms - right.end_ms
    } else {
        right.end_ms - left.end_ms
    };
    start_diff <= tolerance_ms && end_diff <= tolerance_ms
}

/// Compute the diff between two caption tracks.
#[must_use]
pub fn diff_captions(
    left: &[DiffCaption],
    right: &[DiffCaption],
    config: &DiffConfig,
) -> DiffResult {
    let mut entries = Vec::new();
    let mut right_matched: HashMap<usize, bool> = HashMap::new();

    for l in left {
        let normalized_left = normalize_text(&l.text, config);
        let mut best_match: Option<(usize, DiffKind)> = None;

        for (ri, r) in right.iter().enumerate() {
            if right_matched.get(&ri).copied().unwrap_or(false) {
                continue;
            }
            let tm = timings_match(l, r, config.timing_tolerance_ms);
            let normalized_right = normalize_text(&r.text, config);
            let text_eq = normalized_left == normalized_right;

            if tm && text_eq {
                best_match = Some((ri, DiffKind::Unchanged));
                break;
            } else if tm && !text_eq {
                best_match = Some((ri, DiffKind::TextChanged));
            } else if !tm && text_eq && best_match.is_none() {
                best_match = Some((ri, DiffKind::TimingChanged));
            }
        }

        match best_match {
            Some((ri, kind)) => {
                right_matched.insert(ri, true);
                entries.push(DiffEntry::new(
                    kind,
                    Some(l.clone()),
                    Some(right[ri].clone()),
                ));
            }
            None => {
                // Try a broader search for text matches that had both changes
                let mut found = false;
                for (ri, r) in right.iter().enumerate() {
                    if right_matched.get(&ri).copied().unwrap_or(false) {
                        continue;
                    }
                    let normalized_right = normalize_text(&r.text, config);
                    // Use Levenshtein-like similarity: if texts share >50% chars, consider BothChanged
                    let sim = text_similarity(&normalized_left, &normalized_right);
                    if sim > 0.5 {
                        right_matched.insert(ri, true);
                        entries.push(DiffEntry::new(
                            DiffKind::BothChanged,
                            Some(l.clone()),
                            Some(right[ri].clone()),
                        ));
                        found = true;
                        break;
                    }
                }
                if !found {
                    entries.push(DiffEntry::new(DiffKind::Removed, Some(l.clone()), None));
                }
            }
        }
    }

    // Any unmatched right entries are additions
    for (ri, r) in right.iter().enumerate() {
        if !right_matched.get(&ri).copied().unwrap_or(false) {
            entries.push(DiffEntry::new(DiffKind::Added, None, Some(r.clone())));
        }
    }

    let summary = compute_summary(&entries, left.len(), right.len());

    DiffResult { entries, summary }
}

/// Compute simple text similarity ratio (0.0 to 1.0).
#[allow(clippy::cast_precision_loss)]
fn text_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_len = a_chars.len().max(b_chars.len());
    let common = a_chars
        .iter()
        .zip(b_chars.iter())
        .filter(|(ac, bc)| ac == bc)
        .count();
    common as f64 / max_len as f64
}

/// Compute summary from entries.
fn compute_summary(entries: &[DiffEntry], left_total: usize, right_total: usize) -> DiffSummary {
    let mut summary = DiffSummary {
        left_total,
        right_total,
        ..Default::default()
    };
    for e in entries {
        match e.kind {
            DiffKind::Added => summary.added += 1,
            DiffKind::Removed => summary.removed += 1,
            DiffKind::TextChanged => summary.text_changed += 1,
            DiffKind::TimingChanged => summary.timing_changed += 1,
            DiffKind::BothChanged => summary.both_changed += 1,
            DiffKind::Unchanged => summary.unchanged += 1,
        }
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(index: usize, start: u64, end: u64, text: &str) -> DiffCaption {
        DiffCaption::new(index, start, end, text.to_string())
    }

    #[test]
    fn test_identical_tracks() {
        let left = vec![cap(0, 0, 1000, "Hello"), cap(1, 1500, 3000, "World")];
        let right = vec![cap(0, 0, 1000, "Hello"), cap(1, 1500, 3000, "World")];
        let result = diff_captions(&left, &right, &DiffConfig::default());
        assert_eq!(result.summary.unchanged, 2);
        assert_eq!(result.summary.total_changes(), 0);
    }

    #[test]
    fn test_text_change() {
        let left = vec![cap(0, 0, 1000, "Hello")];
        let right = vec![cap(0, 0, 1000, "Goodbye")];
        let result = diff_captions(&left, &right, &DiffConfig::default());
        assert_eq!(result.summary.text_changed, 1);
    }

    #[test]
    fn test_timing_change() {
        let left = vec![cap(0, 0, 1000, "Hello")];
        let right = vec![cap(0, 500, 1500, "Hello")];
        let result = diff_captions(&left, &right, &DiffConfig::default());
        assert_eq!(result.summary.timing_changed, 1);
    }

    #[test]
    fn test_added_caption() {
        let left = vec![cap(0, 0, 1000, "Hello")];
        let right = vec![cap(0, 0, 1000, "Hello"), cap(1, 2000, 3000, "New caption")];
        let result = diff_captions(&left, &right, &DiffConfig::default());
        assert_eq!(result.summary.added, 1);
        assert_eq!(result.summary.unchanged, 1);
    }

    #[test]
    fn test_removed_caption() {
        let left = vec![cap(0, 0, 1000, "Hello"), cap(1, 2000, 3000, "Bye")];
        let right = vec![cap(0, 0, 1000, "Hello")];
        let result = diff_captions(&left, &right, &DiffConfig::default());
        assert_eq!(result.summary.removed, 1);
        assert_eq!(result.summary.unchanged, 1);
    }

    #[test]
    fn test_empty_tracks() {
        let result = diff_captions(&[], &[], &DiffConfig::default());
        assert_eq!(result.summary.total_changes(), 0);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn test_ignore_whitespace() {
        let left = vec![cap(0, 0, 1000, "Hello  World")];
        let right = vec![cap(0, 0, 1000, "Hello World")];
        let config = DiffConfig {
            ignore_whitespace: true,
            ..Default::default()
        };
        let result = diff_captions(&left, &right, &config);
        assert_eq!(result.summary.unchanged, 1);
    }

    #[test]
    fn test_ignore_case() {
        let left = vec![cap(0, 0, 1000, "Hello World")];
        let right = vec![cap(0, 0, 1000, "hello world")];
        let config = DiffConfig {
            ignore_case: true,
            ..Default::default()
        };
        let result = diff_captions(&left, &right, &config);
        assert_eq!(result.summary.unchanged, 1);
    }

    #[test]
    fn test_timing_tolerance() {
        let left = vec![cap(0, 0, 1000, "Hello")];
        let right = vec![cap(0, 50, 1050, "Hello")];
        let config = DiffConfig {
            timing_tolerance_ms: 100,
            ..Default::default()
        };
        let result = diff_captions(&left, &right, &config);
        assert_eq!(result.summary.unchanged, 1);
    }

    #[test]
    fn test_similarity_pct_identical() {
        let left = vec![cap(0, 0, 1000, "A"), cap(1, 2000, 3000, "B")];
        let result = diff_captions(&left, &left, &DiffConfig::default());
        let pct = result.summary.similarity_pct();
        assert!((pct - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_pct_empty() {
        let summary = DiffSummary::default();
        assert!((summary.similarity_pct() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diff_kind_display() {
        assert_eq!(DiffKind::Added.to_string(), "ADDED");
        assert_eq!(DiffKind::Removed.to_string(), "REMOVED");
        assert_eq!(DiffKind::TextChanged.to_string(), "TEXT_CHANGED");
        assert_eq!(DiffKind::Unchanged.to_string(), "UNCHANGED");
    }

    #[test]
    fn test_caption_duration() {
        let c = cap(0, 500, 2000, "test");
        assert_eq!(c.duration_ms(), 1500);
    }

    #[test]
    fn test_changes_only() {
        let left = vec![cap(0, 0, 1000, "A"), cap(1, 2000, 3000, "B")];
        let right = vec![cap(0, 0, 1000, "A"), cap(1, 2000, 3000, "C")];
        let result = diff_captions(&left, &right, &DiffConfig::default());
        let changes = result.changes_only();
        assert_eq!(changes.len(), 1);
        assert!(changes[0].is_change());
    }

    #[test]
    fn test_ignore_punctuation() {
        let left = vec![cap(0, 0, 1000, "Hello, World!")];
        let right = vec![cap(0, 0, 1000, "Hello World")];
        let config = DiffConfig {
            ignore_punctuation: true,
            ..Default::default()
        };
        let result = diff_captions(&left, &right, &config);
        assert_eq!(result.summary.unchanged, 1);
    }
}
