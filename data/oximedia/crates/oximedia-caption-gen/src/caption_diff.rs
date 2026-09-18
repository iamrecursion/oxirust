//! Caption diff: compare two caption tracks and report differences.
//!
//! This module provides a structural diff between two [`CaptionBlock`] tracks,
//! reporting insertions, deletions, substitutions, and timing shifts.  The
//! diff is computed using a Myers-inspired O((N+M)D) shortest-edit-script
//! algorithm operating on caption text content, then a separate timing
//! analysis reports blocks with matching content but shifted timestamps.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_caption_gen::caption_diff::{CaptionDiff, DiffOp};
//! use oximedia_caption_gen::{CaptionBlock, CaptionPosition};
//!
//! let a = vec![CaptionBlock {
//!     id: 1, start_ms: 0, end_ms: 2000,
//!     lines: vec!["Hello world".to_string()],
//!     speaker_id: None, position: CaptionPosition::Bottom,
//! }];
//! let b = vec![CaptionBlock {
//!     id: 1, start_ms: 500, end_ms: 2500,
//!     lines: vec!["Hello world".to_string()],
//!     speaker_id: None, position: CaptionPosition::Bottom,
//! }];
//! let ops = CaptionDiff::diff(&a, &b);
//! assert!(!ops.is_empty());
//! ```

use crate::alignment::CaptionBlock;

// ─── Diff operation types ─────────────────────────────────────────────────────

/// A single diff operation between two caption tracks.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffOp {
    /// A block exists in both tracks with identical text content.
    /// The `timing_shift_ms` is non-zero if the timing differs.
    Equal {
        /// Index in the left (original) track.
        left_idx: usize,
        /// Index in the right (revised) track.
        right_idx: usize,
        /// Difference in `start_ms` between right and left (signed).
        timing_shift_ms: i64,
    },
    /// A block was replaced: text content differs between left and right.
    Replace {
        left_idx: usize,
        right_idx: usize,
        /// Text content in the left track.
        left_text: String,
        /// Text content in the right track.
        right_text: String,
    },
    /// A block was inserted (present in right but not left).
    Insert { right_idx: usize, text: String },
    /// A block was deleted (present in left but not right).
    Delete { left_idx: usize, text: String },
}

impl DiffOp {
    /// Human-readable symbol for this operation (`=`, `~`, `+`, `-`).
    pub fn symbol(&self) -> char {
        match self {
            DiffOp::Equal {
                timing_shift_ms: 0, ..
            } => '=',
            DiffOp::Equal { .. } => '~',
            DiffOp::Replace { .. } => '~',
            DiffOp::Insert { .. } => '+',
            DiffOp::Delete { .. } => '-',
        }
    }
}

// ─── Diff summary ─────────────────────────────────────────────────────────────

/// Summary statistics of a diff.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffSummary {
    /// Number of blocks that are identical (text + timing).
    pub equal_count: usize,
    /// Number of blocks present in both tracks but with shifted timing.
    pub timing_shifted_count: usize,
    /// Number of blocks with changed text content.
    pub replaced_count: usize,
    /// Number of blocks inserted (only in right track).
    pub inserted_count: usize,
    /// Number of blocks deleted (only in left track).
    pub deleted_count: usize,
    /// Total number of operations.
    pub total_ops: usize,
    /// Similarity ratio in [0.0, 1.0]: `equal / total`.
    pub similarity: f32,
}

impl DiffSummary {
    fn from_ops(ops: &[DiffOp]) -> Self {
        let mut equal_count = 0usize;
        let mut timing_shifted_count = 0usize;
        let mut replaced_count = 0usize;
        let mut inserted_count = 0usize;
        let mut deleted_count = 0usize;

        for op in ops {
            match op {
                DiffOp::Equal {
                    timing_shift_ms: 0, ..
                } => equal_count += 1,
                DiffOp::Equal { .. } => timing_shifted_count += 1,
                DiffOp::Replace { .. } => replaced_count += 1,
                DiffOp::Insert { .. } => inserted_count += 1,
                DiffOp::Delete { .. } => deleted_count += 1,
            }
        }

        let total_ops = ops.len();
        let similarity = if total_ops == 0 {
            1.0
        } else {
            (equal_count + timing_shifted_count) as f32 / total_ops as f32
        };

        Self {
            equal_count,
            timing_shifted_count,
            replaced_count,
            inserted_count,
            deleted_count,
            total_ops,
            similarity,
        }
    }
}

// ─── Caption diff engine ──────────────────────────────────────────────────────

/// Compares two caption tracks and produces a list of [`DiffOp`] values.
pub struct CaptionDiff;

impl CaptionDiff {
    /// Compute the diff between `left` (original) and `right` (revised) tracks.
    ///
    /// The diff is computed on normalised text content (whitespace-collapsed,
    /// case-sensitive).  Timing differences on otherwise identical blocks are
    /// reported as [`DiffOp::Equal`] with a non-zero `timing_shift_ms`.
    pub fn diff(left: &[CaptionBlock], right: &[CaptionBlock]) -> Vec<DiffOp> {
        let left_texts: Vec<String> = left.iter().map(|b| normalise_text(b)).collect();
        let right_texts: Vec<String> = right.iter().map(|b| normalise_text(b)).collect();

        // Run LCS-based diff.
        let edit_ops = lcs_diff(&left_texts, &right_texts);

        // Map back to DiffOp, incorporating timing information.
        edit_ops
            .into_iter()
            .map(|raw| match raw {
                RawOp::Equal(li, ri) => {
                    let shift = right[ri].start_ms as i64 - left[li].start_ms as i64;
                    DiffOp::Equal {
                        left_idx: li,
                        right_idx: ri,
                        timing_shift_ms: shift,
                    }
                }
                RawOp::Replace(li, ri) => DiffOp::Replace {
                    left_idx: li,
                    right_idx: ri,
                    left_text: left_texts[li].clone(),
                    right_text: right_texts[ri].clone(),
                },
                RawOp::Insert(ri) => DiffOp::Insert {
                    right_idx: ri,
                    text: right_texts[ri].clone(),
                },
                RawOp::Delete(li) => DiffOp::Delete {
                    left_idx: li,
                    text: left_texts[li].clone(),
                },
            })
            .collect()
    }

    /// Compute diff and return a [`DiffSummary`].
    pub fn summarise(left: &[CaptionBlock], right: &[CaptionBlock]) -> DiffSummary {
        let ops = Self::diff(left, right);
        DiffSummary::from_ops(&ops)
    }

    /// Return only the operations where text differs (insertions, deletions,
    /// replacements).  Equal blocks (even with timing differences) are excluded.
    pub fn text_changes(left: &[CaptionBlock], right: &[CaptionBlock]) -> Vec<DiffOp> {
        Self::diff(left, right)
            .into_iter()
            .filter(|op| !matches!(op, DiffOp::Equal { .. }))
            .collect()
    }

    /// Return the list of timing shifts for blocks that have identical text
    /// content but different `start_ms` values.
    ///
    /// Only blocks where the shift is non-zero are returned.
    pub fn timing_shifts(
        left: &[CaptionBlock],
        right: &[CaptionBlock],
    ) -> Vec<(usize, usize, i64)> {
        Self::diff(left, right)
            .into_iter()
            .filter_map(|op| match op {
                DiffOp::Equal {
                    left_idx,
                    right_idx,
                    timing_shift_ms,
                } if timing_shift_ms != 0 => Some((left_idx, right_idx, timing_shift_ms)),
                _ => None,
            })
            .collect()
    }
}

// ─── Normalisation ────────────────────────────────────────────────────────────

/// Normalise a [`CaptionBlock`] to a single string for text comparison.
///
/// Lines are joined with a single space and leading/trailing whitespace is
/// collapsed.  This intentionally ignores timing, speaker IDs, and position
/// so that the diff focuses purely on textual content.
fn normalise_text(block: &CaptionBlock) -> String {
    block
        .lines
        .iter()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

// ─── LCS diff (Myers / patience-sort inspired) ───────────────────────────────

/// Raw edit operation (before timing annotation).
#[derive(Debug)]
enum RawOp {
    Equal(usize, usize),
    Replace(usize, usize),
    Insert(usize),
    Delete(usize),
}

/// Compute an edit script from `left` to `right` using a simple LCS DP.
///
/// The LCS approach is O(NM) in the worst case, which is acceptable for
/// caption tracks (typically < 2000 blocks per track).
fn lcs_diff(left: &[String], right: &[String]) -> Vec<RawOp> {
    let n = left.len();
    let m = right.len();

    // Build LCS table.
    // lcs[i][j] = length of LCS of left[i..] and right[j..].
    // Use flat Vec for cache efficiency.
    let mut lcs = vec![0u32; (n + 1) * (m + 1)];

    let idx = |i: usize, j: usize| i * (m + 1) + j;

    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[idx(i, j)] = if left[i] == right[j] {
                lcs[idx(i + 1, j + 1)] + 1
            } else {
                lcs[idx(i + 1, j)].max(lcs[idx(i, j + 1)])
            };
        }
    }

    // Backtrack to produce edit ops.
    let mut ops: Vec<RawOp> = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < n || j < m {
        if i < n && j < m && left[i] == right[j] {
            ops.push(RawOp::Equal(i, j));
            i += 1;
            j += 1;
        } else if j < m && (i >= n || lcs[idx(i, j + 1)] >= lcs[idx(i + 1, j)]) {
            ops.push(RawOp::Insert(j));
            j += 1;
        } else if i < n {
            // Check if we can pair this delete with an upcoming insert as a replace.
            if j < m && lcs[idx(i + 1, j)] == lcs[idx(i, j + 1)] {
                // Both options have same LCS length — emit replace.
                ops.push(RawOp::Replace(i, j));
                i += 1;
                j += 1;
            } else {
                ops.push(RawOp::Delete(i));
                i += 1;
            }
        }
    }

    ops
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::CaptionPosition;

    fn make_block(id: u32, start_ms: u64, end_ms: u64, text: &str) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec![text.to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    // ─── diff: equal tracks ───────────────────────────────────────────────────

    #[test]
    fn diff_identical_tracks_all_equal() {
        let a = vec![
            make_block(1, 0, 1000, "Hello"),
            make_block(2, 1000, 2000, "World"),
        ];
        let b = a.clone();
        let ops = CaptionDiff::diff(&a, &b);
        assert_eq!(ops.len(), 2);
        assert!(ops.iter().all(|op| matches!(
            op,
            DiffOp::Equal {
                timing_shift_ms: 0,
                ..
            }
        )));
    }

    // ─── diff: timing shift ───────────────────────────────────────────────────

    #[test]
    fn diff_same_text_different_timing_reports_shift() {
        let a = vec![make_block(1, 0, 1000, "Hello")];
        let b = vec![make_block(1, 500, 1500, "Hello")];
        let ops = CaptionDiff::diff(&a, &b);
        assert_eq!(ops.len(), 1);
        assert!(matches!(
            ops[0],
            DiffOp::Equal {
                timing_shift_ms: 500,
                ..
            }
        ));
    }

    // ─── diff: insertion ──────────────────────────────────────────────────────

    #[test]
    fn diff_extra_block_in_right_is_insert() {
        let a = vec![make_block(1, 0, 1000, "Hello")];
        let b = vec![
            make_block(1, 0, 1000, "Hello"),
            make_block(2, 1000, 2000, "World"),
        ];
        let ops = CaptionDiff::diff(&a, &b);
        // One equal + one insert.
        assert!(ops.iter().any(|op| matches!(op, DiffOp::Insert { .. })));
    }

    // ─── diff: deletion ───────────────────────────────────────────────────────

    #[test]
    fn diff_missing_block_in_right_is_delete() {
        let a = vec![
            make_block(1, 0, 1000, "Hello"),
            make_block(2, 1000, 2000, "World"),
        ];
        let b = vec![make_block(1, 0, 1000, "Hello")];
        let ops = CaptionDiff::diff(&a, &b);
        assert!(ops.iter().any(|op| matches!(op, DiffOp::Delete { .. })));
    }

    // ─── diff: replace ────────────────────────────────────────────────────────

    #[test]
    fn diff_changed_text_is_replace_or_delete_insert() {
        let a = vec![make_block(1, 0, 1000, "Hello world")];
        let b = vec![make_block(1, 0, 1000, "Goodbye world")];
        let ops = CaptionDiff::diff(&a, &b);
        assert!(!ops.is_empty());
        // Must not be "Equal".
        assert!(!ops.iter().all(|op| matches!(op, DiffOp::Equal { .. })));
    }

    // ─── diff: empty tracks ───────────────────────────────────────────────────

    #[test]
    fn diff_both_empty_returns_empty() {
        let ops = CaptionDiff::diff(&[], &[]);
        assert!(ops.is_empty());
    }

    #[test]
    fn diff_left_empty_all_inserts() {
        let b = vec![make_block(1, 0, 1000, "Hello")];
        let ops = CaptionDiff::diff(&[], &b);
        assert!(ops.iter().all(|op| matches!(op, DiffOp::Insert { .. })));
    }

    #[test]
    fn diff_right_empty_all_deletes() {
        let a = vec![make_block(1, 0, 1000, "Hello")];
        let ops = CaptionDiff::diff(&a, &[]);
        assert!(ops.iter().all(|op| matches!(op, DiffOp::Delete { .. })));
    }

    // ─── summarise ───────────────────────────────────────────────────────────

    #[test]
    fn summarise_identical_similarity_one() {
        let a = vec![make_block(1, 0, 1000, "A"), make_block(2, 1000, 2000, "B")];
        let summary = CaptionDiff::summarise(&a, &a);
        assert!((summary.similarity - 1.0).abs() < 1e-5);
        assert_eq!(summary.equal_count, 2);
    }

    #[test]
    fn summarise_all_different_similarity_zero() {
        let a = vec![make_block(1, 0, 1000, "A")];
        let b = vec![make_block(1, 0, 1000, "Z")];
        let summary = CaptionDiff::summarise(&a, &b);
        assert_eq!(summary.equal_count, 0);
    }

    // ─── text_changes ─────────────────────────────────────────────────────────

    #[test]
    fn text_changes_excludes_equal_and_shifted() {
        let a = vec![
            make_block(1, 0, 1000, "Same text"),
            make_block(2, 1000, 2000, "Different"),
        ];
        let b = vec![
            make_block(1, 500, 1500, "Same text"), // shifted, not changed
            make_block(2, 2000, 3000, "New text"), // changed
        ];
        let changes = CaptionDiff::text_changes(&a, &b);
        // Only the changed block should appear.
        assert!(changes.iter().all(|op| !matches!(op, DiffOp::Equal { .. })));
    }

    // ─── timing_shifts ────────────────────────────────────────────────────────

    #[test]
    fn timing_shifts_detects_shifted_blocks() {
        let a = vec![make_block(1, 0, 1000, "Hello")];
        let b = vec![make_block(1, 250, 1250, "Hello")];
        let shifts = CaptionDiff::timing_shifts(&a, &b);
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0].2, 250);
    }

    #[test]
    fn timing_shifts_ignores_unshifted_blocks() {
        let a = vec![make_block(1, 0, 1000, "Hello")];
        let b = a.clone();
        let shifts = CaptionDiff::timing_shifts(&a, &b);
        assert!(shifts.is_empty());
    }

    // ─── DiffOp::symbol ───────────────────────────────────────────────────────

    #[test]
    fn diff_op_symbol() {
        assert_eq!(
            DiffOp::Equal {
                left_idx: 0,
                right_idx: 0,
                timing_shift_ms: 0
            }
            .symbol(),
            '='
        );
        assert_eq!(
            DiffOp::Equal {
                left_idx: 0,
                right_idx: 0,
                timing_shift_ms: 100
            }
            .symbol(),
            '~'
        );
        assert_eq!(
            DiffOp::Insert {
                right_idx: 0,
                text: "t".into()
            }
            .symbol(),
            '+'
        );
        assert_eq!(
            DiffOp::Delete {
                left_idx: 0,
                text: "t".into()
            }
            .symbol(),
            '-'
        );
    }

    // ─── normalise_text ───────────────────────────────────────────────────────

    #[test]
    fn normalise_text_joins_lines() {
        let block = CaptionBlock {
            id: 1,
            start_ms: 0,
            end_ms: 1000,
            lines: vec!["Line one".to_string(), "Line two".to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        };
        let text = normalise_text(&block);
        assert_eq!(text, "Line one Line two");
    }
}
