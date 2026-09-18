#![allow(dead_code)]
//! Diff viewer for comparing versions of collaborative documents or timelines.
//!
//! Produces a list of `DiffBlock`s, each containing `DiffLine`s tagged as
//! `Unchanged`, `Added`, or `Removed`, giving reviewers a clear picture of
//! what changed between two text representations.

/// A single line in a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLine {
    /// Line present in both the old and new versions.
    Unchanged(String),
    /// Line that was added in the new version.
    Added(String),
    /// Line that was removed from the old version.
    Removed(String),
}

impl DiffLine {
    /// Returns a tag string describing the type of this line.
    pub fn line_type(&self) -> &'static str {
        match self {
            DiffLine::Unchanged(_) => "unchanged",
            DiffLine::Added(_) => "added",
            DiffLine::Removed(_) => "removed",
        }
    }

    /// The raw text content of this line.
    pub fn content(&self) -> &str {
        match self {
            DiffLine::Unchanged(s) | DiffLine::Added(s) | DiffLine::Removed(s) => s.as_str(),
        }
    }

    /// Returns `true` if the line represents a change (not `Unchanged`).
    pub fn is_changed(&self) -> bool {
        !matches!(self, DiffLine::Unchanged(_))
    }
}

/// A contiguous block of diff lines (e.g. a hunk).
#[derive(Debug, Clone)]
pub struct DiffBlock {
    /// Sequential block number, 0-indexed.
    pub index: usize,
    /// All lines in this block.
    pub lines: Vec<DiffLine>,
}

impl DiffBlock {
    /// Create a new diff block.
    pub fn new(index: usize, lines: Vec<DiffLine>) -> Self {
        Self { index, lines }
    }

    /// Count of lines in this block that represent a change (Added or Removed).
    pub fn changed_lines(&self) -> usize {
        self.lines.iter().filter(|l| l.is_changed()).count()
    }

    /// Count of added lines in this block.
    pub fn added_lines(&self) -> usize {
        self.lines
            .iter()
            .filter(|l| matches!(l, DiffLine::Added(_)))
            .count()
    }

    /// Count of removed lines in this block.
    pub fn removed_lines(&self) -> usize {
        self.lines
            .iter()
            .filter(|l| matches!(l, DiffLine::Removed(_)))
            .count()
    }

    /// Returns `true` if this block contains at least one changed line.
    pub fn has_changes(&self) -> bool {
        self.changed_lines() > 0
    }
}

/// Compares two text documents and produces a sequence of `DiffBlock`s.
///
/// The implementation uses a simple line-level longest-common-subsequence (LCS)
/// algorithm, suitable for moderate-sized documents (< 50 k lines).
#[derive(Debug, Default)]
pub struct DiffViewer {
    /// Number of context lines to include around each change.
    context_lines: usize,
}

impl DiffViewer {
    /// Create a new viewer with the given context-line count.
    pub fn new(context_lines: usize) -> Self {
        Self { context_lines }
    }

    /// Compare `old` and `new` texts and return a list of `DiffBlock`s.
    ///
    /// Each block represents a hunk of consecutive changes (or context).
    pub fn compare(&self, old: &str, new: &str) -> Vec<DiffBlock> {
        let old_lines: Vec<&str> = old.lines().collect();
        let new_lines: Vec<&str> = new.lines().collect();

        let all_lines = self.diff_lines(&old_lines, &new_lines);
        self.group_into_blocks(all_lines)
    }

    /// Count of blocks that contain at least one changed line.
    pub fn changed_block_count(&self, blocks: &[DiffBlock]) -> usize {
        blocks.iter().filter(|b| b.has_changes()).count()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Produce a flat list of `DiffLine`s via LCS.
    fn diff_lines<'a>(&self, old: &[&'a str], new: &[&'a str]) -> Vec<DiffLine> {
        let m = old.len();
        let n = new.len();

        // Build the LCS table.
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in (0..m).rev() {
            for j in (0..n).rev() {
                if old[i] == new[j] {
                    dp[i][j] = dp[i + 1][j + 1] + 1;
                } else {
                    dp[i][j] = dp[i + 1][j].max(dp[i][j + 1]);
                }
            }
        }

        // Walk the table to emit diff lines.
        let mut result = Vec::new();
        let mut i = 0usize;
        let mut j = 0usize;

        while i < m || j < n {
            if i < m && j < n && old[i] == new[j] {
                result.push(DiffLine::Unchanged(old[i].to_string()));
                i += 1;
                j += 1;
            } else if j < n && (i >= m || dp[i][j + 1] >= dp[i + 1][j]) {
                result.push(DiffLine::Added(new[j].to_string()));
                j += 1;
            } else {
                result.push(DiffLine::Removed(old[i].to_string()));
                i += 1;
            }
        }

        result
    }

    /// Group a flat diff into `DiffBlock`s using the configured context window.
    fn group_into_blocks(&self, lines: Vec<DiffLine>) -> Vec<DiffBlock> {
        if lines.is_empty() {
            return vec![];
        }

        // Find indices of changed lines.
        let changed_indices: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter_map(|(i, l)| if l.is_changed() { Some(i) } else { None })
            .collect();

        if changed_indices.is_empty() {
            // No changes – single unchanged block.
            return vec![DiffBlock::new(0, lines)];
        }

        // Build hunk ranges [start, end) inclusive of context.
        let ctx = self.context_lines;
        let mut hunks: Vec<(usize, usize)> = Vec::new();

        for &ci in &changed_indices {
            let start = ci.saturating_sub(ctx);
            let end = (ci + ctx + 1).min(lines.len());

            if let Some(last) = hunks.last_mut() {
                if start <= last.1 {
                    last.1 = last.1.max(end);
                } else {
                    hunks.push((start, end));
                }
            } else {
                hunks.push((start, end));
            }
        }

        hunks
            .into_iter()
            .enumerate()
            .map(|(idx, (start, end))| DiffBlock::new(idx, lines[start..end].to_vec()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn viewer() -> DiffViewer {
        DiffViewer::new(1)
    }

    // DiffLine tests

    #[test]
    fn test_unchanged_line_type() {
        let l = DiffLine::Unchanged("hello".to_string());
        assert_eq!(l.line_type(), "unchanged");
    }

    #[test]
    fn test_added_line_type() {
        let l = DiffLine::Added("world".to_string());
        assert_eq!(l.line_type(), "added");
    }

    #[test]
    fn test_removed_line_type() {
        let l = DiffLine::Removed("bye".to_string());
        assert_eq!(l.line_type(), "removed");
    }

    #[test]
    fn test_unchanged_is_not_changed() {
        assert!(!DiffLine::Unchanged("x".to_string()).is_changed());
    }

    #[test]
    fn test_added_is_changed() {
        assert!(DiffLine::Added("x".to_string()).is_changed());
    }

    #[test]
    fn test_removed_is_changed() {
        assert!(DiffLine::Removed("x".to_string()).is_changed());
    }

    #[test]
    fn test_line_content() {
        assert_eq!(DiffLine::Added("abc".to_string()).content(), "abc");
    }

    // DiffBlock tests

    #[test]
    fn test_block_changed_lines_count() {
        let block = DiffBlock::new(
            0,
            vec![
                DiffLine::Unchanged("a".to_string()),
                DiffLine::Added("b".to_string()),
                DiffLine::Removed("c".to_string()),
            ],
        );
        assert_eq!(block.changed_lines(), 2);
    }

    #[test]
    fn test_block_has_changes_true() {
        let block = DiffBlock::new(0, vec![DiffLine::Added("x".to_string())]);
        assert!(block.has_changes());
    }

    #[test]
    fn test_block_has_changes_false() {
        let block = DiffBlock::new(0, vec![DiffLine::Unchanged("x".to_string())]);
        assert!(!block.has_changes());
    }

    #[test]
    fn test_block_added_removed_counts() {
        let block = DiffBlock::new(
            0,
            vec![
                DiffLine::Added("a".to_string()),
                DiffLine::Added("b".to_string()),
                DiffLine::Removed("c".to_string()),
            ],
        );
        assert_eq!(block.added_lines(), 2);
        assert_eq!(block.removed_lines(), 1);
    }

    // DiffViewer tests

    #[test]
    fn test_identical_texts_no_changes() {
        let v = DiffViewer::new(0);
        let text = "line1\nline2\nline3";
        let blocks = v.compare(text, text);
        assert_eq!(v.changed_block_count(&blocks), 0);
    }

    #[test]
    fn test_single_line_added() {
        let v = DiffViewer::new(0);
        let old = "a\nb";
        let new = "a\nc\nb";
        let blocks = v.compare(old, new);
        assert!(v.changed_block_count(&blocks) > 0);
    }

    #[test]
    fn test_single_line_removed() {
        let v = DiffViewer::new(0);
        let old = "a\nb\nc";
        let new = "a\nc";
        let blocks = v.compare(old, new);
        assert!(v.changed_block_count(&blocks) > 0);
    }

    #[test]
    fn test_empty_old_all_added() {
        let v = DiffViewer::new(0);
        let blocks = v.compare("", "line1\nline2");
        let total_added: usize = blocks.iter().map(|b| b.added_lines()).sum();
        assert_eq!(total_added, 2);
    }

    #[test]
    fn test_empty_new_all_removed() {
        let v = DiffViewer::new(0);
        let blocks = v.compare("line1\nline2", "");
        let total_removed: usize = blocks.iter().map(|b| b.removed_lines()).sum();
        assert_eq!(total_removed, 2);
    }

    #[test]
    fn test_changed_block_count() {
        let v = DiffViewer::new(0);
        let blocks = v.compare("a\nb", "a\nc");
        assert_eq!(v.changed_block_count(&blocks), 1);
    }

    #[test]
    fn test_both_empty_no_blocks() {
        let v = DiffViewer::new(0);
        let blocks = v.compare("", "");
        assert!(blocks.is_empty());
    }
}
